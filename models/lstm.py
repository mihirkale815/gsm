import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from modules import criterions




class SingleTaskModel(nn.Module):

    def __init__(self,
                 task_id,
                 encoder,
                 decoder,
                 tie_weights=True):

        super(SingleTaskModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.task_id = task_id

        if tie_weights:
            if task_id == 'book':
                if  self.encoder.embedding_dim != self.decoder.input_dim:
                    raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ',
                                 self.decoder.input_dim,self.encoder.embedding_dim)
                self.decoder.decoder.weight = self.encoder.book_embedding_layer.weight
            if task_id == 'author':
                if  self.encoder.embedding_dim != self.decoder.input_dim:
                    raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ',
                                 self.decoder.input_dim,self.encoder.embedding_dim)
                self.decoder.decoder.weight = self.encoder.author_embedding_layer.weight



    def forward(self, batch):

        hidden_states = self.encoder(batch)['hidden_states']
        logits = self.decoder(hidden_states)['logits']
        output = {self.task_id:{}}
        output[self.task_id]['logits'] = logits
        return output

    def compute_loss(self,batch):
        output = self.forward(batch)
        loss = criterions.cross_entropy(batch[self.task_id]['targets'], output[self.task_id]['logits'], batch['masks'])
        output[self.task_id]['loss'] = loss
        output['loss'] = loss
        return output

class MultiTaskModel(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 tie_weights=True):

        super(MultiTaskModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.BOOK_TASK_ID = 'book'
        self.AUTHOR_TASK_ID = 'author'

        if tie_weights:
            if self.encoder.embedding_dim != self.decoder.book_decoder.input_dim:
                raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ')
            if self.encoder.embedding_dim != self.decoder.author_decoder.input_dim:
                raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ')
            self.decoder.book_decoder.decoder.weight = self.encoder.book_embedding_layer.weight
            self.decoder.author_decoder.decoder.weight = self.encoder.author_embedding_layer.weight
            print("Weights have been tied")
        #lgs = log sigma square
        self.lgs1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.lgs2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)


    def forward(self, batch):

        hidden_states = self.encoder(batch)['hidden_states']
        decoder_output = self.decoder(hidden_states)
        return decoder_output


    def compute_loss(self,batch):
        output = self.compute_individual_losses(batch)
        output['loss'] = output[self.BOOK_TASK_ID]['loss'] + output[self.AUTHOR_TASK_ID]['loss']
        return output

    def compute_individual_losses(self,batch):
        output = self.forward(batch)
        book_loss = criterions.cross_entropy(batch[self.BOOK_TASK_ID]['targets'],
                                             output[self.BOOK_TASK_ID]['logits'], batch['masks'])
        author_loss = criterions.cross_entropy(batch[self.AUTHOR_TASK_ID]['targets'],
                                               output[self.AUTHOR_TASK_ID]['logits'], batch['masks'])
        output[self.BOOK_TASK_ID]['loss'] = book_loss
        output[self.AUTHOR_TASK_ID]['loss'] = author_loss
        return output

    def compute_weighted_loss(self,batch):
        output = self.compute_individual_losses(batch)
        book_loss = output[self.BOOK_TASK_ID]['loss']
        author_loss = output[self.AUTHOR_TASK_ID]['loss']

        book_loss  = book_loss * torch.exp(-self.lgs1) + self.lgs1/2.0
        author_loss = author_loss * torch.exp(-self.lgs2) + self.lgs2/2.0

        output['loss'] = book_loss + author_loss
        output[self.BOOK_TASK_ID]['loss'] = book_loss
        output[self.AUTHOR_TASK_ID]['loss'] = author_loss
        return output


class SingleTaskMovieModel(nn.Module):

    def __init__(self,
                 task_id,
                 encoder,
                 decoder,
                 args):

        super(SingleTaskMovieModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.task_id = task_id
        tie_weights = args['tie_weights']
        tie_mixture_weights = args['tie_mixture_weights']

        if tie_weights:
            if task_id == 'movie':
                #if  self.encoder.movie_embedding_dim != self.decoder.input_dim:
                #    raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ',
                #                 self.decoder.input_dim,self.encoder.embedding_dim)
                self.decoder.decoder.weight = self.encoder.movie_embedding_layer.weight
            if task_id == 'genre':
                #if  self.encoder.genre_embedding_dim != self.decoder.input_dim:
                #    raise ValueError('When using the tied flag, embedding_dim should be equal to input_dim ',
                #                 self.decoder.input_dim,self.encoder.embedding_dim)
                self.decoder.decoder.weight = self.encoder.genre_embedding_layer.weight

        if tie_mixture_weights == 'full':
            self.decoder.prior.weight = self.encoder.genre_embedding_layer.weight
        elif tie_mixture_weights == 'partial':
            genre_emb_size = self.encoder.genre_embedding_layer.embedding_dim
            self.decoder.prior.weight[:,:genre_emb_size] = self.encoder.genre_embedding_layer.weight


    def forward(self, batch):

        hidden_states = self.encoder(batch)['hidden_states']
        logits = self.decoder(hidden_states)['logits']
        output = {self.task_id:{}}
        output[self.task_id]['logits'] = logits
        return output

    def compute_loss(self,batch):
        output = self.forward(batch)
        loss = criterions.cross_entropy(batch[self.task_id]['targets'], output[self.task_id]['logits'],
                                        batch[self.task_id]['masks'])
        output[self.task_id]['loss'] = loss
        output['loss'] = loss
        return output





