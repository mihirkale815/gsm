import os
import sys

sys.path.append("../")
sys.path.append("../mtl")
sys.path.append("data/")

from mtl import driver




DATA_PATH = "../../data/goodreads/goodreads_sequence_with_splits_young_adult.sample.json"
args = {}
args['data_path'] = DATA_PATH
args['patience'] = 3
args['use_cuda'] = 1
args['optimizer'] = 'adam'
args['num_epochs'] = 2
args['learning_rate'] = 0.004
args['momentum'] = 0.9
args['batch_size'] = 2
args['clip_grad'] = 5
args['embedding_dim'] = 128
args['hidden_dim'] = 128
args['dropout'] = 0.2
args['num_layers'] = 2
args['pad_index'] = 0
args['device'] = 'cpu'
args['max_num_trial'] = 3
args['exp_dir'] = "experiments/"
args['model_id'] = 1
args['is_book_task'] = True
args['is_author_task'] = True
args['encoder'] = 'rnn'
args['num_mos_decoders'] = 10
args['is_weight_loss'] = True
args['combine_logits'] = True



class Args:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
        self.args = args

driver.train(Args(args=args))


