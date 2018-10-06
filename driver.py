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


def train(args):
    use_cuda = args.use_cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path

    dataset_container = DatasetContainer()
    dataset_container.create_vocab(data_path)
    dataset_container.load_sequences(data_path)


    exp_dir = os.path.join(args.exp_dir,args.model_id)
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    patience = args.patience
    model_path = os.path.join(exp_dir,"model.pt")
    log_path = os.path.join(exp_dir, "logs.txt")
    vocab_size = len(dataset_container.item2id)

    #sys.stdout = open(log_path, 'w')

    if args.encoder == 'hrnn': encoder_fn = rnn_encoder.HRNNEncoder
    elif args.encoder == 'rnn': encoder_fn = rnn_encoder.RNNEncoder

    encoder = encoder_fn(
                  vocab_size = vocab_size,
                  hidden_dim  = args.hidden_dim,
                  embedding_dim = args.embedding_dim,
                  dropout = args.dropout,
                  num_layers = args.num_layers,
                  device = device,
                  bidirectional = args.bidirectional,
                  cell='lstm',
                  pad_index = 0)
    decoder = linear_decoder.LinearDecoder(input_dim=args.hidden_dim, output_dim=vocab_size)
    model = SRM_LSTM(encoder,decoder).to(device)
    print(model)

    model_args = {}
    model_args['vocab_size'] = vocab_size
    model_args['hidden_dim'] = args.hidden_dim
    model_args['embedding_dim'] = args.embedding_dim
    model_args['dropout'] = args.dropout
    model_args['device'] = use_cuda
    model_args['num_layers'] = 1
    model_args['pad_index'] = 0


    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else :
        raise NotImplementedError("optimizer",args.optimizer,"not available")

    lambda1 = lambda epoch: 1 / (1 + 0.05 * epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)


    train_losses = []
    best_dev_loss = float("inf")
    dev_losses = [best_dev_loss]
    epochs_since_improvement = 0

    for epoch in range(args.num_epochs):

        print(">> Epoch = ", epoch+1)
        train_loss,dev_loss = 0,0
        scheduler.step()

        batch_iter = dataset_container.batch_iterator('seen','train')
        epoch_train_losses = []
        for idx, batch in enumerate(batch_iter):
            torch.set_grad_enabled(True)
            model.train()

            batch = move_batch_to_device(batch)

            logits = model(batch)

            loss = cross_entropy(batch['outputs'],logits,batch['masks'])
            epoch_train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)


        #for key in ['seen', 'unseen']:
        #    for period in ['train', 'val', 'test']:
        #        batch_iter = dataset_container.batch_iterator(key, period)
        #        loss = get_loss(model, batch_iter)
        #        print(key + "-" + period,round(loss,3))

        batch_iter = dataset_container.batch_iterator('seen', 'val')
        actual, predicted, metrics = evaluate(model, batch_iter)
        dev_loss = -metrics['hitrate'][10]
        #dev_loss = get_loss(model,batch_iter)
        print(">> dev loss =", round(dev_loss,3))
        dev_losses.append(dev_loss)

        prev_dev_loss,dev_loss = dev_losses[-2],dev_losses[-1]


        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), model_path)
            torch.save(model_args, model_path+"_args")
            print(">> updated best model")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement > patience:
            print(">> lost patience with the model")
            break

    print("LATEST MODEL")
    full_eval(model, dataset_container)

    print("BEST MODEL")
    model.load_state_dict(torch.load(model_path))
    full_eval(model, dataset_container)

def full_eval(model,dataset_container):
    for key in ['seen', 'unseen']:
        for period in ['train', 'val', 'test']:
            print(key + "-" + period)

            batch_iter = dataset_container.batch_iterator(key, period)
            actual, predicted, metrics = evaluate(model, batch_iter)
            print("hitrate = ", metrics['hitrate'][10])

            batch_iter = dataset_container.batch_iterator(key, period)
            loss = get_loss(model, batch_iter)
            print("loss = ", round(loss, 3))





def get_loss(model,batch_iter):
    torch.set_grad_enabled(False)
    model.eval()
    losses = []
    num_batches = 0.0
    for idx, batch in enumerate(batch_iter):
        batch = move_batch_to_device(batch)
        logits = model(batch)
        loss = cross_entropy(batch['outputs'], logits, batch['masks'])
        losses.append(loss.item())
        num_batches += 1
    return sum(losses)/num_batches

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
    parser.add_argument('--use_cuda', action="store", dest="use_cuda", type=int, default=0)
    parser.add_argument('--clip_grad', action="store", dest="clip_grad", type=float, default=5.0)
    parser.add_argument('--dropout', action="store", dest="dropout", type=float, default=0.5)
    parser.add_argument('--optimizer', action="store", dest="optimizer", type=str, default='adam')
    parser.add_argument('--metric', action="store", dest="metric", type=str, default='hitrate')
    parser.add_argument('--bidirectional', action="store", dest="bidirectional", type=int, default=0)

    args = parser.parse_args()
