import os
import sys
from itertools import product

sys.path.append("../")
sys.path.append("../srp")
sys.path.append("data/")

from srp.modules.encoders import rnn_encoder
from srp.modules.decoders import linear_decoder
from srp import driver
import srp.data.utils as datautils
import pickle


class Args:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


DATA_PATH = "../../data/amazon/books.pkl"
args = {}
args['data_path'] = DATA_PATH
args['bidirectional'] = 0
args['dropout'] = 0
args['hidden_dim'] = 100
args['embedding_dim'] = 100
args['num_layers'] = 1
args['exp_dir'] = "experiments"
args['patience'] = 20
args['use_cuda'] = 1
args['optimizer'] = 'adam'
args['num_epochs'] = 20
args['learning_rate'] = 0.001
args['momentum'] = 0.9
args['batch_size'] = 64
args['clip_grad'] = 5
args['encoder'] = 'rnn'

for hidden_dim in [50,100,200]:
    for num_layers in [1,2]:
        for encoder in ['rnn','hrnn']:

            args['encoder'] = encoder
            embedding_dim = hidden_dim
            args['hidden_dim'] = hidden_dim
            args['embedding_dim'] = embedding_dim
            args['num_layers'] = num_layers
            args['model_id'] = "_".join([str(hidden_dim),str(num_layers)])
            print("Training model for args",args)
            options = Args(args=args)
            driver.train(options)
