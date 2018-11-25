import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class FFEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 embedding_dim,
                 dropout,
                 num_layers,
                 pad_index,
                 device,
                 use_month_features=True,
                 month_feature_type='simple'):
        super(FFEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.device = device
        self.use_month_features = use_month_features
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gap_emb_size = 10
        self.month_embedding_layer = nn.Embedding(300, self.gap_emb_size)
        self.projection_layer = nn.Linear(2*embedding_dim,embedding_dim)
        self.mlp = nn.Sequential(*(nn.Linear(embedding_dim,embedding_dim) for _ in range(self.num_layers)))
        self.bidirectional = False





    def forward(self, batch):
        sequences = batch['inputs']
        batch_size,max_len = sequences.size()

        embeddings = self.embedding_layer(sequences)
        embeddings = self.dropout(embeddings)


        cum_embeddings = self.embedding_layer(sequences)
        denom = torch.Tensor([1.0/n for n in range(1,max_len+1)]).unsqueeze(0).unsqueeze(2).to(self.device)
        cummean_embeds = torch.cumsum(cum_embeddings,dim=1)*denom

        hidden_states = torch.cat([embeddings, cummean_embeds], dim=2)
        hidden_states = self.mlp(self.projection_layer(hidden_states))

        return hidden_states