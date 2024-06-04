#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
from prettytable import PrettyTable
from collections import defaultdict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import *
from learn2learn.vision.datasets import *
from datasets.gtsrb import traffic
from sklearn.linear_model import LogisticRegression

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class FewShotBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, N, K, Q, num_iterations):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_iterations = num_iterations

        labels = [label for _, label in dataset]
        self.label2idx = defaultdict(list)
        for i, y in enumerate(labels):
            self.label2idx[y].append(i)

        few_labels = [y for y, indices in self.label2idx.items() if len(indices) <= self.K]
        for y in few_labels:
            del self.label2idx[y]

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        label_set = set(list(self.label2idx.keys()))
        for _ in range(self.num_iterations):
            labels = random.sample(label_set, self.N)
            indices = []
            for y in labels:
                if len(self.label2idx[y]) >= self.K+self.Q:
                    indices.extend(list(random.sample(self.label2idx[y], self.K+self.Q)))
                else:
                    tmp_indices = [i for i in self.label2idx[y]]
                    random.shuffle(tmp_indices)
                    indices.extend(tmp_indices[:self.K] + np.random.choice(tmp_indices[self.K:], size=self.Q).tolist())
            yield indices


def load_fewshot_datasets(args):
    path = '/data4/jeongheon_oh/code_folder/data/fewshot'
    transform = transforms.Compose([transforms.Resize(224,transforms.functional.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.data == 'aircraft':
        data = FGVCAircraft(path, mode='all', transform=transform)
    elif args.data == 'birds':
        data = CUBirds200(path, mode='all', transform=transform)
    elif args.data == 'fc100':
        data = FC100(path, mode='test', transform=transform)
    elif args.data == 'flowers':
        data = VGGFlower102(path, mode='all', transform=transform)
    elif args.data == 'fungi':
        data = FGVCFungi(path, mode='all', transform=transform)
    elif args.data == 'texture':
        data = DescribableTextures(path, mode='all', transform=transform)
    elif args.data == 'omniglot':
        transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))
        data = FullOmniglot(path, transform=transform)
    elif args.data == 'traffic':
        data = traffic(path, split='test', transform=transform)
    else:
        warnings.warn('You have chosen wrong data')
    return data


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--seed', default=100, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

# additional configs for dataset:
parser.add_argument('--data', default='aircraft', type=str, help='aircraft, birds, fc100, flowers, fungi, texture')

# configs for few-shot learning
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--Q', type=int, default=16)
parser.add_argument('--num-tasks', type=int, default=2000)

best_acc1 = 0

def main():
    global args
    args = parser.parse_args()

    global best_acc1

    # create dir
    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    # init log
    log = logger(path=save_dir, log_name="fewshot_log.txt")
    args_table = get_args_table(vars(args))
    log.info(str(args_table)+'\n')

    # gpu and seed
    if args.gpu is not None:
        log.info("Use GPU: {} for training".format(args.gpu))
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)
    log.info("Use Seed : {} for replication".format(args.seed))

    # Data loading code
    log.info("Data preparing")
    datasets = load_fewshot_datasets(args)
    build_sampler = partial(FewShotBatchSampler, N = args.N, K=args.K, Q=args.Q, num_iterations=args.num_tasks)
    build_dataloader = partial(DataLoader, num_workers = args.workers)
    test_loader = build_dataloader(datasets, batch_sampler = build_sampler(datasets))

    # create model
    log.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](zero_init_residual=True)
    model.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            log.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

            log.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()
    print(model)

    model.eval()
    all_accuracies = []
    for i, (batch, _) in enumerate(test_loader):
        with torch.no_grad():
            batch = batch.cuda(args.gpu)
            B, C, H, W = batch.shape
            batch = batch.view(args.N, args.K+args.Q, C, H, W)
            
            train_batch = batch[:, :args.K].reshape(args.N*args.K, C, H, W)
            test_batch = batch[:, args.K:].reshape(args.N*args.Q, C, H, W)
            train_labels = torch.arange(args.N).unsqueeze(1).repeat(1, args.K).view(-1).cuda(args.gpu)
            test_labels = torch.arange(args.N).unsqueeze(1).repeat(1, args.Q).view(-1).cuda(args.gpu)

        with torch.no_grad():
            X_train = model(train_batch)
            Y_train = train_labels
            
            X_test = model(test_batch)
            Y_test = test_labels
        
        classifier = LogisticRegression(solver='liblinear').fit(X_train.cpu().numpy(), Y_train.cpu().numpy())
        preds = classifier.predict(X_test.cpu().numpy())
        acc = np.mean((Y_test.cpu().numpy() == preds).astype(float))
        all_accuracies.append(acc)
        if (i + 1) % 10 == 0:
            log.info(f'{i+1:3d} | {acc:.4f} (mean: {np.mean(all_accuracies):.4f})')
    avg = np.mean(all_accuracies)
    std = np.std(all_accuracies) * 1.96 / np.sqrt(len(all_accuracies))
    log.info('\n')
    log.info(f'Mean: {avg:.4f} ± {std:.4f}')


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

if __name__ == '__main__':
    main()
