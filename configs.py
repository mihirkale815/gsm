import os
import sys

sys.path.append("../")
sys.path.append("../srp")
sys.path.append("data/")

from srp.modules.encoders import rnn_encoder
from srp.modules.decoders import linear_decoder
from srp import driver
from srp import pop
import srp.data.utils as datautils
import pickle

DATA_PATH = "../../data/amazon/books.pkl"
args = {}
args['data_path'] = DATA_PATH
args['bidirectional'] = 0
args['patience'] = 15
args['use_cuda'] = 1
args['optimizer'] = 'adam'
args['num_epochs'] = 30
args['learning_rate'] = 0.004
args['momentum'] = 0.9
args['batch_size'] = 8
args['clip_grad'] = 5
args['use_year_scale'] = False
args['use_month_features'] = False
args['month_feature_type'] = 'simple'

class Args:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
        self.args = args





args['use_month_features'] = False
args['multi_label_loss'] = False
args['dropout'] = 0.4
args['encoder'] = 'rnn'
args['num_layers'] = 2
args['use_month_features'] = False
args['hidden_dim'] = 256
args['embedding_dim'] = 256


paths = [("../../data/movietweetings/movietweetings.pkl","experiments/movietweetings/")]
paths.append(("../../data/amazon/books.pkl","experiments/books/"))
#paths.append(("../../data/amazon/baby.pkl","experiments/baby/"))



for data_path,exp_dir in paths:
    args['data_path'] = data_path
    args['exp_dir'] = exp_dir
    model_id = 30
    args['window_size'] = 3
    args['mll_weight'] = 0

    opts = [('hrnn',1),('cumrnn',1),('cumrnn',2),('cumrnn',3)]
    for encoder,num_layers in opts:
        args['encoder'] = encoder
        args['num_layers'] = num_layers
        args['use_year_scale'] = True
        model_id+=1
        args['model_id'] = model_id
        driver.train(Args(args=args))
        print("*" * 100)

    args['use_year_scale'] = False
    args['encoder'] = 'hrnn'
    args['num_layers'] = 1
    model_id += 1
    args['model_id'] = model_id
    driver.train(Args(args=args))
    print("*" * 100)
