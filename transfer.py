import argparse
import os
from prettytable import PrettyTable
from copy import deepcopy
import time
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=100, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

# additional configs:
parser.add_argument('--dir', metavar='DIR', help='path to dataset')
parser.add_argument('--data', default='CIFAR100', type=str, help='Choose from CIFAR100, CIFAR10, cub200, flowers102')
parser.add_argument('--metric', type=str, default='top1', help='top1, class-avg')

best_acc, best_w, best_classifier = 0, 0., None
def main():
    global args
    args = parser.parse_args()

    global best_acc, best_w, best_classifier

    # create dir
    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    # init log
    log = logger(path=save_dir, log_name="linear_log.txt")
    args_table = get_args_table(vars(args))
    log.info(str(args_table)+'\n')

    # gpu and seed
    if args.gpu is not None:
        log.info("Use GPU: {} for training".format(args.gpu))
    cudnn.benchmark = True
    setup_seed(args.seed)
    log.info("Use Seed : {} for replication".format(args.seed))

    # Data loading code
    log.info("Data : {} preparing".format(args.data))
    _, train_data, val_data, test_data, num_classes = get_transfer_dataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size = 256, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size = 256, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    log.info('Dataset contains {}/{}/{} train/val/test samples'.format(len(train_data), len(val_data), len(test_data)))

    # create model
    log.info("=> creating model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](num_classes = num_classes, zero_init_residual=True)
    classifier = deepcopy(backbone.fc)
    backbone.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            log.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = backbone.load_state_dict(state_dict, strict=False)
            print(msg)
            
            log.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        backbone = backbone.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        backbone = backbone.cuda()
        classifier = classifier.cuda()

    log.info('Collecting features...')
    start_data_time = time.time()
    X_train, Y_train = collect_features(backbone, train_loader, args)
    X_val, Y_val = collect_features(backbone, val_loader, args)
    X_test, Y_test = collect_features(backbone, test_loader, args)
    
    optim_kwargs = {
        'line_search_fn': 'strong_wolfe',
        'max_iter': 5000,
        'lr': 1.,
        'tolerance_grad': 1e-10,
        'tolerance_change': 0,
    }
    total_data_time = time.time() - start_data_time
    total_data_time_str = str(datetime.timedelta(seconds=int(total_data_time)))
    log.info('Collecting features done {} time...'.format(total_data_time_str))
    
    start_time = time.time()
    for w in torch.logspace(-6, 5, steps=45).tolist():
        optimizer = torch.optim.LBFGS(classifier.parameters(), **optim_kwargs)
        optimizer.step(build_step(X_train, Y_train, classifier, optimizer, w))
        acc = compute_accuracy(X_val, Y_val, classifier, args.metric)
    
        if best_acc < acc:
            best_acc = acc
            best_w = w
            best_classifier = deepcopy(classifier)

        log.info(f'w={w:.4e}, acc={acc:.4f}')
                    
    log.info('\n')
    log.info(f'BEST for Val: w={best_w:.4e}, acc={best_acc:.4f}')
    
    X = torch.cat([X_train, X_val], 0)
    Y = torch.cat([Y_train, Y_val], 0)
    optimizer = torch.optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
    optimizer.step(build_step(X, Y, best_classifier, optimizer, best_w))
    acc = compute_accuracy(X_test, Y_test, best_classifier, args.metric)
    log.info('\n')    
    log.info('Metrix : {}, Test acc : {:.4f}'.format(args.metric, acc))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Total time {}'.format(total_time_str))

def collect_features(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        for x, y in dataloader:
            z = model(x.cuda(args.gpu))
            features.append(z.detach())
            labels.append(y.to(z.device))
        features = torch.cat(features, 0).detach()
        labels = torch.cat(labels, 0).detach()
    return features, labels

def build_step(X, Y, classifier, optimizer, w):
    def step():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step

def compute_accuracy(X, Y, classifier, metric):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
        if metric == 'top1':
            acc = (preds == Y).float().mean().item()
        elif metric == 'class-avg':
            total, count = 0., 0.
            for y in range(0, Y.max().item()+1):
                masks = Y == y
                if masks.sum() > 0:
                    total += (preds[masks] == y).float().mean().item()
                    count += 1
            acc = total / count
        else:
            raise Exception(f'Unknown metric: {metric}')
    return acc

def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

if __name__ == '__main__':
    main()

