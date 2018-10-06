import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class RNNEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 dropout,
                 num_layers,
                 pad_index,
                 device,
                 bidirectional,
                 cell):
        super(RNNEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device
        self.bidirectional = bidirectional

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.bidirectional : self.W = nn.Linear(2*self.hidden_dim, self.embedding_dim)

        if cell == 'lstm' :
            self.rnn = nn.LSTM(self.embedding_dim,
                               self.hidden_dim,
                               batch_first=True,
                               bidirectional=self.bidirectional,
                               num_layers=self.num_layers)
        else : raise  NotImplementedError(cell,"not implemented")


    def forward(self, batch):
        lens = batch['lens']
        sequences = batch['inputs']

        embeddings = self.embedding_layer(sequences)
        embeddings = self.dropout(embeddings)

        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states, _ = self.rnn(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        if self.bidirectional : hidden_states = self.W(hidden_states)

        return hidden_states



class HRNNEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 dropout,
                 num_layers,
                 pad_index,
                 device,
                 bidirectional,
                 cell,
                 kth=5):
        super(HRNNEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device
        self.bidirectional = bidirectional
        self.kth = kth

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.W = nn.Linear(2*self.hidden_dim, self.embedding_dim)

        if cell == 'lstm' :
            self.rnn = nn.LSTM(self.embedding_dim,
                               self.hidden_dim,
                               batch_first=True,
                               bidirectional=self.bidirectional,
                               num_layers=self.num_layers)
            self.rnn2 = nn.LSTM(self.hidden_dim,
                           self.hidden_dim,
                           batch_first=True,
                           bidirectional=self.bidirectional,
                           num_layers=self.num_layers)

        else : raise  NotImplementedError(cell,"not implemented")


    def forward(self, batch):
        lens = batch['lens']
        sequences = batch['inputs']

        embeddings = self.embedding_layer(sequences)
        embeddings = self.dropout(embeddings)

        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states, _ = self.rnn(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        batch_size,max_len,_ = hidden_states.size()
        ds_hidden_states = hidden_states[:,np.arange(0,max_len,self.kth),:]
        packed_input = pack_padded_sequence(ds_hidden_states, [1 + k/self.kth for k in lens], batch_first=True)
        packed_hidden_states, _ = self.rnn2(packed_input)
        hidden_states_l2, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states_l2 = self.dropout(hidden_states_l2)

        _,max_len_l2,_ = hidden_states_l2.size()
        hidden_states_l2_expanded = torch.zeros(hidden_states.size()).to(self.device)
        for i in range(0,max_len_l2):
            start= i*self.kth
            end = min(start+self.kth,max_len)
            hidden_states_l2_expanded[:,start:end,:] = hidden_states_l2[:,i:i+1,:]

        hidden_states = torch.cat((hidden_states,hidden_states_l2_expanded),dim=2)
        hidden_states = self.W(hidden_states)

        return hidden_states

