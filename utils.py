import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter, Image, ImageOps
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101, Flowers102
from datasets.imagenet import ImageNet
from datasets.cub200 import CUB
from datasets.dogs import Dogs
from datasets.datasets import Pets, Food101, DTD, SUN397, MIT67
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from tqdm import tqdm

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")

random.seed(0)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
    
class KNNValidataion(object):
    def __init__(self, args, model, c):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        if args.gpu is not None:
            self.device = torch.device(args.gpu)
        self.args = args
        self.c = c
        base_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        memory_dataset = ImageNet(root = args.dir, subset = args.subset, split='train', transform = base_transform)
        test_dataset = ImageNet(root = args.dir, subset = args.subset, split='val', transform = base_transform)
        self.memory_loader = DataLoader(memory_dataset, batch_size=256, shuffle=False, pin_memory=True, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, drop_last=False)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        self.model.eval()
        total_top1, total_num, feature_bank = 0.0, 0, []
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        with torch.no_grad():
            for data,_ in tqdm(self.memory_loader):
                data = data.to(self.device)
                features = self.model(data)
                features = nn.functional.normalize(features, dim=-1)
                feature_bank.append(features)
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)

            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                features = self.model(data)
                features = nn.functional.normalize(features, dim=-1)

                total_num += data.size(0)
                sim_matrix = torch.mm(features, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.args.k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * self.args.k, self.c, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
                # score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, self.c), dim=1)
                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        return total_top1 / total_num * 100
    
    def eval(self):
        return self._topk_retrieval()

def get_transfer_dataset(args):
    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.functional.InterpolationMode.BICUBIC), 
                                    transforms.CenterCrop(224),transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if args.data == 'food101':
        trainval = Food101(root=os.path.join(args.dir, 'food-101'), split='train', transform=transform)
        train_data, val_data = random_split(trainval, [68175, 7575], generator=generator(42))
        test_data = Food101(root=os.path.join(args.dir, 'food-101'), split='test',  transform=transform)
        num_classes = 101
        
    elif args.data =='CIFAR10':
        trainval = CIFAR10(root=args.dir, train=True, transform=transform)
        train_data, val_data = random_split(trainval, [45000, 5000], generator=generator(43))
        test_data = CIFAR10(root=args.dir, train=False, transform=transform)
        num_classes = 10
        
    elif args.data == 'CIFAR100':
        trainval = CIFAR100(root=args.dir, train=True, transform=transform)
        train_data, val_data = random_split(trainval, [45000, 5000], generator=generator(44))
        test_data = CIFAR100(root=args.dir, train=False, transform=transform)
        num_classes = 100

    elif args.data == 'sun397':
        trn_indices, val_indices = torch.load('./datasets/split/sun397.pth')
        trainval = SUN397(root = os.path.join(args.dir, 'SUN397'), split='Training', transform=transform)
        train_data = Subset(trainval, trn_indices)
        val_data   = Subset(trainval, val_indices)
        test_data = SUN397(root = os.path.join(args.dir, 'SUN397'), split='Testing', transform=transform)
        num_classes = 397

    elif args.data == 'dtd':
        train_data = DTD(root=os.path.join(args.dir, 'dtd'), split='train', transform=transform)
        val_data = DTD(root=os.path.join(args.dir, 'dtd'), split='val',   transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = DTD(root=os.path.join(args.dir, 'dtd'), split='test',  transform=transform)
        num_classes = 47

    elif args.data == 'pets':
        trainval = Pets(root=os.path.join(args.dir, 'pets'), split='trainval', transform=transform)
        train_data, val_data = random_split(trainval, [2940, 740], generator=generator(49))
        test_data = Pets(root=os.path.join(args.dir, 'pets'), split='test', transform=transform)
        num_classes = 37

    elif args.data == 'caltech101':
        transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(root=args.dir, transform=transform)
        trn_indices, val_indices, tst_indices = torch.load('./datasets/split/caltech101.pth')
        train_data = Subset(D, trn_indices)
        val_data = Subset(D, val_indices)
        trainval = ConcatDataset([train_data, val_data])
        test_data= Subset(D, tst_indices)
        num_classes = 101

    elif args.data == 'flowers102':
        train_data = Flowers102(root = args.dir, split='train', transform=transform)
        val_data = Flowers102(root = args.dir, split='val', transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = Flowers102(root = args.dir, split='test', transform=transform)
        num_classes = 102

    elif args.data == 'mit67':
        trainval = MIT67(root = os.path.join(args.dir, 'mit67'), split='Train', transform=transform)
        test_data = MIT67(root = os.path.join(args.dir, 'mit67'), split='Test', transform=transform)
        train_data, val_data = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif args.data == 'cub200':
        transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))
        train_data = CUB(os.path.join(args.dir, 'CUB_200_2011'), 'train', transform=transform)
        val_data = CUB(os.path.join(args.dir, 'CUB_200_2011'), 'valid', transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = CUB(os.path.join(args.dir, 'CUB_200_2011'), 'test', transform=transform)
        num_classes = 200

    elif args.data == 'dog':
        trn_indices, val_indices = torch.load('./datasets/split/dog.pth')
        trainval = Dogs(args.dir, train=True, transform=transform)
        train_data = Subset(trainval, trn_indices)
        val_data = Subset(trainval, val_indices)
        test_data = Dogs(args.dir, train=False, transform = transform)
        num_classes = 120
        
    else:  
        warnings.warn('You have chosen wrong data')
        
    return trainval, train_data, val_data, test_data, num_classes

