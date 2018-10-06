from data_utils import DatasetContainer, load_word_vectors
from model import LSTM_CRF,LSTM_Softmax
from aux_model import LSTM_Softmax_Aux
import argparse
import torch.optim as optim
import numpy
import torch
import os
from utils import move_batch_to_device
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from utils import *




parser = argparse.ArgumentParser(description='bilstm for sequence tagging')
parser.add_argument('--data_dir', action="store", dest="data_dir", type=str,default="data")
parser.add_argument('--train_path', action="store", dest="train_path", type=str,default="train.data.quad")
parser.add_argument('--dev_path', action="store", dest="dev_path", type=str,default="dev.data.quad")
parser.add_argument('--test_path', action="store", dest="test_path", type=str)
parser.add_argument('--embedding_path', action="store", dest="embedding_path", type=str,default=None)
parser.add_argument('--batch_size', action="store", dest="batch_size", type=int,default=1)
parser.add_argument('--embedding_dim', action="store", dest="embedding_dim", type=int,default=100)
parser.add_argument('--hidden_dim', action="store", dest="hidden_dim", type=int,default=100)
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float,default=0.001)
parser.add_argument('--momentum', action="store", dest="momentum", type=float,default=0.9)
parser.add_argument('--num_epochs', action="store", dest="num_epochs", type=int,default=10)
parser.add_argument('--patience', action="store", dest="patience", type=int,default=3)
parser.add_argument('--early_stopping', action="store", dest="early_stopping", type=int,default=1)
parser.add_argument('--output_path', action="store", dest="train_output_path", type=str)
parser.add_argument('--model_path', action="store", dest="model_path", type=str, default="models/model")
parser.add_argument('--exp_dir', action="store", dest="exp_dir", type=str, default="experiments/")
parser.add_argument('--use_cuda', action="store", dest="use_cuda", type=int,default=0)
parser.add_argument('--clip_grad', action="store", dest="clip_grad", type=float,default=5.0)
parser.add_argument('--dropout', action="store", dest="dropout", type=float,default=0.5)
parser.add_argument('--optimizer', action="store", dest="optimizer", type=str, default='sgd')

args = parser.parse_args()
use_cuda = args.use_cuda
device = torch.device("cuda" if use_cuda==1 else "cpu")

arch = args.arch

train_path = args.data_dir + "/train.data"
dev_path = args.data_dir + "/dev.data"
test_path = args.data_dir + "/test.data"
exp_dir = args.exp_dir


vocab_file_paths = [train_path]
dataset_container = DatasetContainer()
dataset_container.create_vocab(vocab_file_paths)

dataset_container.add_split(train_path,"train")
dataset_container.add_split(dev_path,"dev")
dataset_container.add_split(test_path,"test")

patience = args.patience
model_path = args.model_path
vocab_size = len(dataset_container.word2id)


model = LSTM_Softmax(vocab_size,
                 args.hidden_dim,
                 args.embedding_dim,
                 dropout=args.dropout,
                 device = device,
                 num_layers=1,
                 pad_index=0).to(device)
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
else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

lambda1 = lambda epoch: 1 / (1 + 0.05 * epoch)
scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

train_iter = dataset_container.get_batch_iterator("train",args.train_batch_size)
dev_iter = dataset_container.get_batch_iterator("dev",args.dev_batch_size)
test_iter = dataset_container.get_batch_iterator("test",args.dev_batch_size)


prev_dev_epoch_f1 = None
best_dev_f1 = None
epochs_since_improvement = 0

for epoch in range(args.num_epochs):

    print("epoch = ", epoch+1)
    epoch_loss = 0

    scheduler.step()

    for idx, batch in enumerate(train_iter):

        batch = move_batch_to_device(batch,device)

        model.train()

        optimizer.zero_grad()

        loss = model.loss(batch)
        epoch_loss += loss.item()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()



    print("Train Loss = ", epoch_loss)
    dev_loss = get_loss_for_dataset(dev_iter, model)
    print("Dev Loss = ", dev_loss)








    if best_dev_f1 is None or dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        torch.save(model.state_dict(), model_path)
        torch.save(model_args, model_path+"_args")
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement > patience:
        print("Lost patience with the model")
        break



