import os
import sys

sys.path.append("../")
sys.path.append("../srp")
sys.path.append("data/")

from srp.modules.encoders import rnn_encoder
from srp.modules.decoders import linear_decoder
from srp import driver
import srp.data.utils as datautils
import pickle

DATA_PATH = "../../data/amazon/books.pkl"
args = {}
args['data_path'] = DATA_PATH
args['bidirectional'] = 0
args['dropout'] = 0
args['hidden_dim'] = 100
args['embedding_dim'] = 100
args['num_layers'] = 1
args['exp_dir'] = "experiments"
args['patience'] = 2
args['use_cuda'] = 0
args['optimizer'] = 'adam'
args['num_epochs'] = 10
args['learning_rate'] = 0.0001
args['momentum'] = 0.9
args['batch_size'] = 64
args['clip_grad'] = 5
args['model_id'] = '1'
args['encoder'] = 'rnn'
class Args:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
            
options = Args(args=args)
driver.train(options)
