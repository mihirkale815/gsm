import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np




class SRMMultiLabel(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 multi_label_decoder,
                 tie_weights=True):

        super(SRMMultiLabel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.multi_label_decoder = multi_label_decoder

        if tie_weights:
            if not self.encoder.bidirectional and self.encoder.embedding_dim != self.decoder.input_dim:
                raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ')
            self.decoder.decoder.weight = self.encoder.embedding_layer.weight
            self.multi_label_decoder.decoder.weight = self.encoder.embedding_layer.weight


    def forward(self, batch):

        hidden_states = self.encoder(batch)
        logits = self.decoder(hidden_states)
        multi_label_logits = self.multi_label_decoder(hidden_states)
        if len(logits.size()) == 4 :
            batch_size, max_year, max_len, vocab_size = logits.size()
            logits = logits.view(batch_size*max_year, max_len, vocab_size)

            batch_size, max_year, max_len, vocab_size = multi_label_logits.size()
            multi_label_logits = multi_label_logits.view(batch_size * max_year, max_len, vocab_size)


        return logits,multi_label_logits







