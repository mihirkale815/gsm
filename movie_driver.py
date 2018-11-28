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



def train(args):
    print("Preparing experiment with arguments",list(args.args.items()))
    use_cuda = args.use_cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path

    is_movie_task = args.is_movie_task
    is_genre_task = args.is_genre_task
    is_weight_loss = args.is_weight_loss
    num_mos_decoders = args.num_mos_decoders
    is_mtl = is_genre_task and is_movie_task
    task_ids = []
    if is_genre_task: task_ids.append('genre')
    if is_movie_task: task_ids.append('movie')

    dataset_container = MovielensDatasetContainer(args.batch_size)
    dataset_container.load_sequences(data_path)


    exp_dir = os.path.join(args.exp_dir,str(args.model_id))
    dev_save_dir = os.path.join(exp_dir,"dev")
    test_save_dir = os.path.join(exp_dir, "test")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(dev_save_dir)
        os.makedirs(test_save_dir)
    max_patience = args.patience
    model_state_path = os.path.join(exp_dir,"model_state_dict.pt")
    model_path = os.path.join(exp_dir, "model.pt")
    log_path = os.path.join(exp_dir, "logs.txt")


    movie_vocab_size = len(dataset_container.movie2id)
    genre_vocab_size = len(dataset_container.genre2id)

    if args.encoder == 'rnn':
        encoder_args = {}
        encoder_args['movie_vocab_size'] = movie_vocab_size
        encoder_args['genre_vocab_size'] = genre_vocab_size
        encoder_args['movie_embedding_dim'] = args.movie_embedding_dim
        encoder_args['genre_embedding_dim'] = args.genre_embedding_dim
        encoder_args['hidden_dim'] = args.hidden_dim
        encoder_args['dropout'] = args.dropout
        encoder_args['num_layers'] = args.num_layers
        encoder_args['pad_index'] = args.pad_index
        encoder_args['device'] = device
        encoder = rnn_encoder.MovieRNNEncoder(encoder_args)

    decoder_args = {}

    if is_mtl:
        pass
        #decoder_args['movie_input_dim'] = args.hidden_dim
        #decoder_args['genre_input_dim'] = args.hidden_dim
        #decoder_args['book_output_dim'] = book_vocab_size
        #decoder_args['author_output_dim'] = author_vocab_size
        #decoder = linear_decoder.MultiTaskLinearDecoder(decoder_args)
        #model = MultiTaskModel(encoder, decoder).to(device)
    else:
        if is_movie_task :
            decoder_args['input_dim'] = args.hidden_dim
            decoder_args['output_dim'] = movie_vocab_size
            decoder_args['num_mos_decoders'] = num_mos_decoders
            decoder_args['split_hidden'] = args.split_hidden

        if args.decoder == 'mos' : decoder = linear_decoder.MOSDecoder(decoder_args)
        elif args.decoder == 'moc': decoder = linear_decoder.MOCDecoder(decoder_args)
        else : decoder = linear_decoder.LinearDecoder(decoder_args)
        model_args = {}
        model_args['tie_mixture_weights'] = args.tie_mixture_weights
        model_args['tie_weights'] = args.tie_weights
        model = SingleTaskMovieModel(task_ids[0],encoder, decoder,model_args).to(device)

    print(model)
    print_num_trainable_params(model)

    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1); scheduler.step()

    best_dev_metric = -float("inf")
    hist_dev_metrics = [best_dev_metric]
    hist_train_metrics = [float("inf")]
    num_trial = 0

    patience = 0
    max_epochs = args.num_epochs

    for epoch in range(max_epochs):


        batch_iter = dataset_container.batch_iterator('train')
        epoch_train_losses = []
        num_elements = 0

        for idx, batch in enumerate(batch_iter):
            torch.set_grad_enabled(True)
            model.train()

            batch = move_batch_to_device(batch)

            if is_mtl and is_weight_loss:
                output = model.compute_weighted_loss(batch)
            else : output = model.compute_loss(batch)
            loss = output['loss']
            num_elements += batch[task_ids[0]]['masks'].sum().item()
            epoch_train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            torch.cuda.empty_cache()

            #if idx%10 == 0 : print("Completed %d batches" % (idx))

        hist_train_metrics.append(sum(epoch_train_losses)/num_elements)
        result = evaluate(model, dataset_container.batch_iterator('dev'),task_ids)
        print("epoch %d train loss = %.2f" % (epoch + 1, hist_train_metrics[-1]))
        display_results(result)
        dev_metric = sum([result[task_id]['metrics']['metrics']['hitrate'][10] for task_id in task_ids])
        is_better = len(hist_dev_metrics) == 2 or dev_metric > max(hist_dev_metrics)
        hist_dev_metrics.append(dev_metric)

        if is_better:
            patience = 0
            print('save currently the best model to [%s]' % model_path)  # , file=sys.stderr)
            torch.save(model.state_dict(), model_state_path)
            torch.save(model, model_path)

            # You may also save the optimizer's state
        elif patience < max_patience:
            patience += 1
            print('hit patience %d' % patience)  # , file=sys.stderr)

            if patience == args.patience:
                num_trial += 1
                print('hit #%d trial' % num_trial)  # , file=sys.stderr)
                if num_trial == args.max_num_trial:
                    print('early stop!', )  # file=sys.stderr)
                    break

                # decay learning rate, and restore from previously best checkpoint
                scheduler.step()
                print("Modified learning rate...")
                print('loading previously best model and decaying learning rate')  # , file=sys.stderr)
                for param_group in optimizer.param_groups:
                    print("lr for param group =", param_group['lr'])

                # load model
                model.load_state_dict(torch.load(model_path))

                # print('restore parameters of the optimizers', file=sys.stderr)
                # You may also need to load the state of the optimizer saved before
                #optimizer.load_state_dict(torch.load(optimizer_save_path))

                # reset patience
                patience = 0



    model.load_state_dict(torch.load(model_path))

#    result = evaluate(model, dataset_container.batch_iterator('train'),task_ids)
#    print("TRAIN SET");display_results(result)

    result = evaluate(model, dataset_container.batch_iterator('dev'),task_ids,dev_save_dir)
    print("DEV SET");display_results(result)

    result = evaluate(model, dataset_container.batch_iterator('test'),task_ids,test_save_dir)
    print("TEST SET"); display_results(result)


    #json.dump(res,open(os.path.join(exp_dir,"summary.json"),"w"))



def display_results(result):
    for task_id in result:
        metrics = result[task_id]['metrics']
        if 100 in  metrics['metrics']['hitrate'] :
            hitrate100 = metrics['metrics']['hitrate'][100]
        if 10  in metrics['metrics']['hitrate']:
            hitrate10 = metrics['metrics']['hitrate'][10]
        if 5 in metrics['metrics']['hitrate']:
            hitrate5 = metrics['metrics']['hitrate'][5]
        loss = metrics['loss']
        print(task_id)
        print("loss = %.2f hitrate@10 = %.2f " % (loss, 100.0 * hitrate10))
        #print("loss = %.2f hitrate@100 = %.2f hitrate@10 = %.2f hitrate@5 = %.2f" %(loss,100.0*hitrate100,100.0*hitrate10,100.0*hitrate5))
    print("*"*50)


def print_num_trainable_params(model):
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters =",params )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bilstm for sequence tagging')
    parser.add_argument('--data_path', action="store", dest="data_dir", type=str, default="data")
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=16)
    parser.add_argument('--embedding_dim', action="store", dest="embedding_dim", type=int, default=100)
    parser.add_argument('--hidden_dim', action="store", dest="hidden_dim", type=int, default=100)
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.0001)
    parser.add_argument('--momentum', action="store", dest="momentum", type=float, default=0.9)
    parser.add_argument('--num_epochs', action="store", dest="num_epochs", type=int, default=10)
    parser.add_argument('--patience', action="store", dest="patience", type=int, default=3)
    parser.add_argument('--early_stopping', action="store", dest="early_stopping", type=int, default=1)
    parser.add_argument('--exp_dir', action="store", dest="exp_dir", type=str, default="experiments/")
    parser.add_argument('--model_id', action="store", dest="model_id", type=str, default="1")
    parser.add_argument('--use_cuda', action="store", dest="use_cuda", type=int, default=0)
    parser.add_argument('--clip_grad', action="store", dest="clip_grad", type=float, default=5.0)
    parser.add_argument('--dropout', action="store", dest="dropout", type=float, default=0.5)
    parser.add_argument('--optimizer', action="store", dest="optimizer", type=str, default='adam')
    parser.add_argument('--metric', action="store", dest="metric", type=str, default='hitrate')
    parser.add_argument('--bidirectional', action="store", dest="bidirectional", type=int, default=0)

    args = parser.parse_args()
