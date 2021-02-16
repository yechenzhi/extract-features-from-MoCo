#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataset.ImageFolder import ImageFolder
from tqdm import tqdm

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MSR-VTT10K Feaure Extraction')
# parser.add_argument('data', metavar='DIR', default='/data3/ycz/SSL/msrvtt10k/VisualSearch',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='/data3/ycz/SSL/moco/checkpoint_0200.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')


def main():
    args = parser.parse_args()
    # args.data='/data3/ycz/SSL/msrvtt10k/VisualSearch'
    #load data
    datadir= '/data/home/caz/VisualSearch/msrvtt10ktest/ImageData/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = ImageFolder(
        datadir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    print("number of images:",len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    #creat model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    
   
    #load model
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']

        #print keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
        # model.load_state_dict(state_dict, strict=False)
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
    

    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules).cuda()
    for p in model.parameters():
        p.requires_grad = False


    # extract feature from each frame to a dictionary
    dic={}
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(dataloader,desc='compute features'):
            x = x.cuda()
            r = model(x).squeeze()
            for i,y_ in enumerate(y):
                y_ = y_.item()
                if y_ not in dic:
                    dic[y_] = []
                dic[y_].append(r[i])   
    

                
    path="/data3/ycz/SSL/extracted_features/moco_tce_v2_200ep_pretrain_msrvtt.txt"
    f=open(path,'w+')

    for key,value in tqdm(dic.items(),desc='write features to txt'):
        key=str(key)
        key="video"+key+" "

        sum=torch.zeros(2048).cuda()
        for v in value:
            sum=sum+v
        sum=sum/len(value)
        sum=sum/(torch.norm(sum)) #normalization, not necessary.
        value=sum.cpu().numpy().tolist()
        value=' '.join([str(v) for v in value])
        f.write(key)
        f.write(str(value).lstrip('[').rstrip(']')+"\n")

    

if __name__ == '__main__':
    main()