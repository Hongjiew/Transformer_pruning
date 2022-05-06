# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from vit import VisionTransformerTeacher, checkpoint_filter_fn, _cfg
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from tokenrank_vit import TokenRankVisionTransformer
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from functools import partial



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--model_name', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--retain_rate_list', nargs='+', type=float, help='Retain Rate')
    parser.add_argument('--prune_list', nargs='+', type=int, help='Prune List')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    if args.model_name == 'tokenrank_deit_tiny_3':

        patch_size = 16
        layers = 12
        # prune_list = [2,5,8]
        heads = 3
        mlp_ratio = 4.
        dims = 192
        qkv_bias = True
        tau_imp = 0.5
        # retain_rate = 0.8
        
        model = TokenRankVisionTransformer(
                prune_list=args.prune_list, patch_size=patch_size,  embed_dim=dims, depth=layers,
                num_heads=heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, tau_imp = tau_imp,
                retain_rate_list = args.retain_rate_list)
        #load pretrained
        model_path = '../model_weights/deit_tiny_patch16_224-a1311bcf.pth'
        checkpoint= torch.load(model_path,map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('successfully loaded from pre-trained weights:', model_path)

    elif args.model_name == 'tokenrank_deit_small_3':

        patch_size = 16
        layers = 12
        # prune_list = [9,10,11]
        heads = 6
        mlp_ratio = 4.
        dims = 384
        qkv_bias = True
        tau_imp = 0.5
        # retain_rate = 0.8
        
        model = TokenRankVisionTransformer(
                prune_list=args.prune_list, patch_size=patch_size,  embed_dim=dims, depth=layers,
                num_heads=heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, tau_imp = tau_imp,
                retain_rate_list = args.retain_rate_list)
        #load pretrained
        model_path = '../model_weights/deit_small_patch16_224-cd65a155.pth'
        checkpoint= torch.load(model_path,map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('successfully loaded from pre-trained weights:', model_path)

    elif args.model_name == 'tokenrank_deit_base_3':

        patch_size = 16
        layers = 12
        # prune_list = [2,5,8]
        heads =12
        mlp_ratio = 4.
        dims = 768
        qkv_bias = True
        tau_imp = 0.5
        # retain_rate = 0.8

        model = TokenRankVisionTransformer(
                prune_list=args.prune_list, patch_size=patch_size,  embed_dim=dims, depth=layers,
                num_heads=heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, tau_imp = tau_imp,
                retain_rate_list = args.retain_rate_list )
        #load pretrained
        model_path = 'model_weights/deit_base_patch16_224-b5f2ef4d.pth'
        checkpoint = torch.load(model_path,map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('successfully loaded from pre-trained weights:', model_path)

    
    else:
        raise NotImplementedError

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(data_loader_val, model, criterion)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)