import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class RNNEncoder(nn.Module):
    def __init__(self,args):
        super(RNNEncoder, self).__init__()

        self.book_vocab_size = args['book_vocab_size']
        self.author_vocab_size = args['author_vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.dropout = nn.Dropout(p=args['dropout'])
        self.num_layers = args['num_layers']
        self.pad_index = args['pad_index']
        self.device = args['device']
        self.book_embedding_layer = nn.Embedding(self.book_vocab_size, self.embedding_dim)
        self.author_embedding_layer = nn.Embedding(self.author_vocab_size, self.embedding_dim)
        self.rnn_input_dim = 2*self.embedding_dim



        self.rnn = nn.LSTM(self.rnn_input_dim,
                               self.hidden_dim,
                               batch_first=True,
                               num_layers=self.num_layers)

        self.BOOK_TASK_ID = 'book'
        self.AUTHOR_TASK_ID = 'author'

    def forward(self, batch):
        lens = batch['lens']
        book_sequences = batch[self.BOOK_TASK_ID]['inputs']
        author_sequences = batch[self.AUTHOR_TASK_ID]['inputs']
        book_embeddings = self.book_embedding_layer(book_sequences)
        author_embeddings = self.author_embedding_layer(author_sequences)
        book_embeddings = self.dropout(book_embeddings)
        author_embeddings = self.dropout(author_embeddings)
        embeddings = torch.cat([book_embeddings, author_embeddings], dim=2)
        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states, _ = self.rnn(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        output = {}
        output['hidden_states'] = hidden_states
        return output



class MovieRNNEncoder(nn.Module):
    def __init__(self,args):
        super(MovieRNNEncoder, self).__init__()

        self.movie_vocab_size = args['movie_vocab_size']
        self.genre_vocab_size = args['genre_vocab_size']
        self.movie_embedding_dim = args['movie_embedding_dim']
        self.genre_embedding_dim = args['genre_embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.dropout = nn.Dropout(p=args['dropout'])
        self.num_layers = args['num_layers']
        self.pad_index = args['pad_index']
        self.device = args['device']
        self.movie_embedding_layer = nn.Embedding(self.movie_vocab_size, self.movie_embedding_dim)
        self.genre_embedding_layer = nn.Embedding(self.genre_vocab_size, self.genre_embedding_dim)
        self.rnn_input_dim = self.movie_embedding_dim + self.genre_embedding_dim



        self.rnn = nn.LSTM(self.rnn_input_dim,
                               self.hidden_dim,
                               batch_first=True,
                               num_layers=self.num_layers)

        self.MOVIE_TASK_ID = 'movie'
        self.GENRE_TASK_ID = 'genre'

    def forward(self, batch):
        lens = batch['lens']
        movie_sequences = batch[self.MOVIE_TASK_ID]['inputs']
        genre_sequences = batch[self.GENRE_TASK_ID]['inputs']
        movie_embeddings = self.movie_embedding_layer(movie_sequences)
        genre_embeddings = self.genre_embedding_layer(genre_sequences)
        genre_embeddings = torch.mean(genre_embeddings,dim=2)
        movie_embeddings = self.dropout(movie_embeddings)
        genre_embeddings = self.dropout(genre_embeddings)
        embeddings = torch.cat([movie_embeddings, genre_embeddings], dim=2)
        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_hidden_states, _ = self.rnn(packed_input)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)

        output = {}
        output['hidden_states'] = hidden_states
        return output




