import os
import sys

sys.path.append("../")
sys.path.append("../mtl")
sys.path.append("data/")

from mtl import movie_driver




DATA_PATH = "../../data/ml-20m/sequences_with_splits.json"
DATA_PATH_SAMPLE = "../../data/ml-20m/sequences_with_splits.sample.json"
args = {}
args['data_path'] = DATA_PATH
args['patience'] = 3
args['use_cuda'] = 1
args['optimizer'] = 'adam'
args['num_epochs'] = 2
args['learning_rate'] = 0.004
args['momentum'] = 0.9
args['batch_size'] = 8
args['clip_grad'] = 5
args['embedding_dim'] = 4
args['hidden_dim'] = 4
args['dropout'] = 0.2
args['num_layers'] = 2
args['pad_index'] = 0
args['device'] = 'cpu'
args['max_num_trial'] = 3
args['exp_dir'] = "experiments/"
args['model_id'] = 1
args['is_movie_task'] = True
args['is_genre_task'] = False
args['movie_embedding_dim'] = 4
args['genre_embedding_dim'] = 4
args['encoder'] = 'rnn'
args['num_mos_decoders'] = 11
args['is_weight_loss'] = True
args['tie_mixture_weights'] = 'full'
args['tie_weights'] = True
args['optim'] = 'adam'
args['split_hidden'] = False
args['decoder'] = 'moc'


class Args:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
        self.args = args

movie_driver.train(Args(args=args))


