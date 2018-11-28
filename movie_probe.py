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

data_path = "../../data/ml-20m/sequences_with_splits.sample.json"
batch_size = 16
model_path = "experiments/1/model.pt"


dataset_container = MovielensDatasetContainer(batch_size)
dataset_container.load_sequences(data_path)
seqs = dataset_container.datasets
useqs = unstack_sequences(seqs)

print("number of sequences after unstacking is",len(useqs))
trunc_seqs = truncate_sequences(useqs,100)
print("number of sequences after truncating is",len(trunc_seqs))
shuff_seqs = partial_shuffle(trunc_seqs,0,mode='global')
print("number of sequences after partial shiffling is",len(shuff_seqs))
dataset_container.batches['probe'] = dataset_container.prepare_batches(shuff_seqs,batch_size,'dev')
batch_iter = dataset_container.batch_iterator('probe')
model = torch.load(model_path)
result = evaluate(model,batch_iter,['movie'],None)
print(result['movie']['metrics'])