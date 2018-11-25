import sys
sys.path.append("../srp")
sys.path.append("../")
from data.utils import DatasetContainer
from models.lstm import SRM_LSTM
import argparse
import torch.optim as optim
import numpy as np
import torch
import os
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from modules.criterions import cross_entropy
from srp.modules.encoders import rnn_encoder
from srp.modules.decoders import linear_decoder
from utils import *
from collections import defaultdict,Counter




def train(args):
    use_cuda = args.use_cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path

    dataset_container = DatasetContainer('serial',args.batch_size)
    dataset_container.create_vocab(data_path)
    dataset_container.load_sequences(data_path)

    iseqs = dataset_container.iseqs
    item2count = Counter([intc.pid for iseq in iseqs for intc in iseq if intc.fold in ('val')])
    items_count_sorted = sorted(item2count.items(), key=lambda x: -x[1])
    topk_items = set([dataset_container.item2id[item] for item,count in items_count_sorted[0:10]])



    for key in ['seen', 'unseen']:
        for period in ['train','val', 'test']:
            total = 0.0
            hits = 0.0
            batch_iter = dataset_container.batch_iterator(key,period)
            for batch in batch_iter:
                targets = batch['outputs']
                masks = batch['masks']
                targets = targets*masks
                batch_size,max_len = targets.size()
                targets = Counter(map(int,list(targets.view(batch_size*max_len).numpy())))
                total += masks.sum().item()
                hits += sum([targets[pid] for pid in targets if pid in topk_items])
            print(key,period,hits,total,hits/total)



