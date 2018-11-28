import sys
sys.path.append("../srp")
sys.path.append("../")
from data.utils import MovielensDatasetContainer
import argparse
import torch.optim as optim
import torch
import os
from torch import nn
from mtl.modules.encoders import rnn_encoder,ff_encoder
from mtl.modules.decoders import linear_decoder
from models.lstm import SingleTaskMovieModel
from utils import *
import json
from data.probe_utils import *

data_path = ""
batch_size = 16
save_path = ""


dataset_container = MovielensDatasetContainer(batch_size)
dataset_container.load_sequences(data_path)
seqs = dataset_container.datasets
trunc_seqs = truncate_sequences(seqs,100)
shuff_seqs = partial_shuffle(trunc_seqs,10,mode='global')
dataset_container.batches['probe'] = dataset_container.prepare_batches(shuff_seqs,batch_size,'dev')
batch_iter = dataset_container.batch_iterator('probe')
model = torch.load(save_path)
result = evaluate(model,batch_iter,['movie'],None)
print(result)