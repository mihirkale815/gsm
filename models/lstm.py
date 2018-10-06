import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np




class SRM_LSTM(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 tie_weights=True):

        super(SRM_LSTM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        if tie_weights:
            if not self.encoder.bidirectional and self.encoder.embedding_dim != self.decoder.input_dim:
                raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ')
            self.decoder.decoder.weight = self.encoder.embedding_layer.weight


    def forward(self, batch):

        hidden_states = self.encoder(batch)
        logits = self.decoder(hidden_states)
        #batch_size, max_len, vocab_size = logits.size()
        #logits = logits.view(batch_size * max_len, vocab_size)
        #scores = F.log_softmax(logits, dim=1)
        #scores = scores.view(batch_size,max_len,vocab_size)
        return logits







