import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from crf import ConditionalRandomField,viterbi_decode
from crf_b import ConditionalRandomFieldTrans
import numpy as np
from data_utils import pad_sequence

class LSTM_Softmax(nn.Module):

    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 dropout,
                 num_layers,
                 pad_index,
                 device):

        super(LSTM_Softmax, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p = dropout)
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm_l1 = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.lstm_l2 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    def forward(self, batch):

        lens = batch['lens']
        sequences = batch['sequences']
        max_len = max(lens)
        batch_size = len(sequences)

        embeddings = self.embedding_layer(sequences)
        embeddings = self.dropout(embeddings)

        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states,_ = self.lstm(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        logits = self.hidden2tag(hidden_states)

        logits = logits.view(batch_size * max_len, self.tagset_size)

        scores = F.log_softmax(logits, dim=1)
        _, predicted_tag_sequence = torch.max(scores, dim=1)
        predicted_tag_sequence = predicted_tag_sequence.view(batch_size,max_len)

        return scores, predicted_tag_sequence


    def loss(self,batch):

        lens = batch['lens']
        max_len = max(lens)
        batch_size = len(lens)
        tag_sequences = batch['tag_sequences']

        scores, predicted_tag_sequence = self.forward(batch)

        return self.loss_fn(scores, tag_sequences.view(batch_size * max_len))




class LSTM_CRF(nn.Module):

    def __init__(self,tagset_size,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 pretrained_embeddings,
                 dropout,
                 num_layers,
                 pad_index,
                 device,
                 fine_tune=True,
                 bidirectional=True):

        super(LSTM_CRF, self).__init__()

        self.tagset_size = tagset_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device


        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        if type(pretrained_embeddings) == torch.Tensor:
            self.embedding_layer.weight.data.copy_(pretrained_embeddings)

        if not fine_tune:
            self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional)

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        self.crf = ConditionalRandomField(self.tagset_size,1,2)

    def get_lstm_feats(self, batch):

        lens = batch['lens']
        word_sequences = batch['word_sequences']
        max_len = max(lens)
        batch_size = len(word_sequences)

        embeddings = self.embedding_layer(word_sequences)
        embeddings = self.dropout(embeddings)

        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states,_ = self.lstm(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        logits = self.hidden2tag(hidden_states)

        return logits
        #logits = logits.view(batch_size * max_len, self.tagset_size)


    def loss(self,batch):
        logits = self.get_lstm_feats(batch)
        mask= batch['mask'].squeeze(1)
        return self.crf.forward(logits, batch['tag_sequences'], mask)

    def forward(self,batch):
        logits = self.get_lstm_feats(batch)
        mask = batch['mask'].squeeze(1)
        all_tags = self.crf.viterbi_tags(logits.to('cpu'),mask.to('cpu'))
        max_len = max(batch['lens'])
        for i in range(len(all_tags)) :
            all_tags[i] += [0 for i in range(max_len - len(all_tags[i]))]
            #print(all_tags[i])
        return None,torch.tensor(all_tags)


class LSTM_CRF_Softmax(nn.Module):

    def __init__(self,tagset_size,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 pretrained_embeddings,
                 dropout,
                 num_layers,
                 pad_index,
                 device,
                 fine_tune=True,
                 bidirectional=True):

        super(LSTM_CRF_Softmax, self).__init__()

        self.tagset_size = tagset_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device


        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        if type(pretrained_embeddings) == torch.Tensor:
            self.embedding_layer.weight.data.copy_(pretrained_embeddings)

        if not fine_tune:
            self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional)

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        self.crf = ConditionalRandomField(self.tagset_size,1,2)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    def get_lstm_feats(self, batch):

        lens = batch['lens']
        word_sequences = batch['word_sequences']
        max_len = max(lens)
        batch_size = len(word_sequences)

        embeddings = self.embedding_layer(word_sequences)
        embeddings = self.dropout(embeddings)

        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states,_ = self.lstm(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        logits = self.hidden2tag(hidden_states)

        return logits
        #logits = logits.view(batch_size * max_len, self.tagset_size)


    def loss(self,batch):
        lens = batch['lens']
        max_len = max(lens)
        batch_size = len(lens)
        logits = self.get_lstm_feats(batch)
        mask= batch['mask'].squeeze(1)
        self.crf_loss  = self.crf.forward(logits, batch['tag_sequences'], mask)

        logits = logits.view(batch_size * max_len, self.tagset_size)
        scores = F.log_softmax(logits, dim=1)
        self.softmax_loss = self.loss_fn(scores, batch['tag_sequences'].view(batch_size * max_len))
        return self.crf_loss + self.softmax_loss

    def forward(self,batch):
        logits = self.get_lstm_feats(batch)
        mask = batch['mask'].squeeze(1)
        all_tags = self.crf.viterbi_tags(logits.to('cpu'),mask.to('cpu'))
        max_len = max(batch['lens'])
        for i in range(len(all_tags)) :
            all_tags[i] += [0 for i in range(max_len - len(all_tags[i]))]
            #print(all_tags[i])
        return None,torch.tensor(all_tags)

