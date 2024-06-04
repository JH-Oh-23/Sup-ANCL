#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import shutil
from prettytable import PrettyTable
from tqdm import tqdm
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
# for distributed 학습
import torch.multiprocessing as mp
import torch.distributed as dist

from datasets.imagenet import ImageNet
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=30.0, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W', help='weight decay (default: 0.)', dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dir', default='', type=str, metavar='PATH', help='Path to dataset')
parser.add_argument('--subset', default=100, type=int, help='Imagenet subset 100 or None')

# additional configs:
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')


best_acc1 = 0
start_epoch = 1
log = None

def main():
    args = parser.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)
        print("Use Seed : {} for replication".format(args.seed))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, log, start_epoch
    args.gpu = gpu

    if not (args.multiprocessing_distributed and args.gpu != 0):
        # create dir
        save_dir = os.path.join(args.save_dir, args.experiment)
        if os.path.exists(save_dir) is not True:
            os.system("mkdir -p {}".format(save_dir))

        # init log
        log = logger(path=save_dir, log_name="linear_log.txt")
        args_table = get_args_table(vars(args))
        log.info(str(args_table)+'\n')


    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    if log is not None:
        log.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            if log is not None:
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

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            if log is not None:
                log.info("=> loaded pre-trained model '{}'".format(args.pretrained))

        else:
            if log is not None:
                log.info("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    
    # Data loading code
    if log is not None:
        log.info("Data preparing")
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = ImageNet(root = args.dir, subset = args.subset, split = 'train', transform = train_transform)
    val_data = ImageNet(root = args.dir, subset = args.subset, split = 'val', transform = test_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size = 256, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_loss, train_acc_1, train_acc_5 = train(train_loader, model, criterion, optimizer, epoch, log, args)
        
        # evaluate on validation set
        val_loss, val_acc_1, val_acc_5 = validate(val_loader, model, criterion, epoch, log, args)

        # remember best acc@1 and save checkpoint
        is_best = val_acc_1 > best_acc1
        best_acc1 = max(val_acc_1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),'best_acc1':best_acc1}, save_dir, is_best)
            if epoch == start_epoch:
                sanity_check(model.state_dict(), args.pretrained)
    if log is not None:
        log.info('\n')
        log.info('Best accuracy : {:.3f}%'.format(best_acc1))


def train(train_loader, model, criterion, optimizer, epoch, log, args):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    if log is not None: train_bar = tqdm(train_loader)
    else: train_bar = train_loader 
    for images, target in train_bar:
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_num += images.size(0)
        total_loss += loss.item() * images.size(0)
        prediction = torch.argsort(output, dim=-1, descending=True)
        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        if log is not None: train_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.3f}% ACC@5: {:.3f}%'
                                .format('Train', epoch, args.epochs, total_loss / total_num,
                                        total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
    if log is not None:
        log.info("current lr is {:.2f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        log.info('Train Epoch : [{}/{}] Loss: {:.4f} ACC@1: {:.3f}% ACC@5: {:.3f}%'
                        .format(epoch, args.epochs, total_loss/total_num, total_correct_1/total_num*100, total_correct_5/total_num*100)) 

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def validate(val_loader, model, criterion, epoch, log, args):
    # switch to evaluate mode
    model.eval()
    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    if log is not None: val_bar = tqdm(val_loader)
    else: val_bar = val_loader
    with torch.no_grad():
        for images, target in val_bar:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_num += images.size(0)
            total_loss += loss.item() * images.size(0)
            prediction = torch.argsort(output, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            if log is not None: val_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.3f}% ACC@5: {:.3f}%'
                                    .format('Test', epoch, args.epochs, total_loss / total_num,
                                            total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
    if log is not None:
        log.info('Test Epoch : [{}/{}] Loss: {:.4f} ACC@1: {:.3f}% ACC@5: {:.3f}%'
                        .format(epoch, args.epochs, total_loss/total_num, total_correct_1/total_num*100, total_correct_5/total_num*100))
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def save_checkpoint(state, path, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))

def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    for milestone in args.schedule:
        lr *= 0.1 if (epoch-1) >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
