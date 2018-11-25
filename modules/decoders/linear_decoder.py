import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LinearDecoder(nn.Module):

    def __init__(self, args):
        super(LinearDecoder, self).__init__()
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.linear = nn.Linear(self.input_dim, self.input_dim)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(self.input_dim, self.output_dim)


    def forward(self, hidden_states):
        self.intermediate_hidden_states = self.tanh(self.linear(hidden_states))
        logits = self.decoder(self.intermediate_hidden_states)
        output = {}
        output['logits'] = logits
        return output


class MOSDecoder(nn.Module):

    def __init__(self, args):
        super(MOSDecoder, self).__init__()

        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.num_experts = args['num_mos_decoders']
        self.split_hidden = args['split_hidden']


        self.prior = nn.Linear(self.input_dim, self.num_experts)
        self.latent = nn.Sequential(nn.Linear(self.input_dim,self.num_experts*self.input_dim),nn.Tanh())
        self.decoder = nn.Linear(self.input_dim,self.output_dim)



    def forward(self, hidden_states):
        batch_size,maxlen,hidden_size = hidden_states.size()

        mixture_hidden_states = hidden_states[:,:,: int(hidden_size/2)]

        if self.split_hidden :
            mixture_logits = self.prior(mixture_hidden_states).contiguous().view(-1, self.num_experts)
        else:
            mixture_logits = self.prior(hidden_states).contiguous().view(-1,self.num_experts)

        latent = self.latent(hidden_states)
        mixture_coefficients = nn.functional.softmax(mixture_logits,-1)
        logits = self.decoder(latent.view(-1,self.input_dim))

        probs = nn.functional.softmax(logits.view(-1,self.output_dim),-1).view(-1,self.num_experts,self.output_dim)
        probs = (probs * mixture_coefficients.unsqueeze(2).expand_as(probs)).sum(1)
        probs = probs.view(batch_size,maxlen,self.output_dim)
        logits = torch.log(probs)

        output = {}
        output['logits'] = logits
        return output



class MOCDecoder(nn.Module):

    def __init__(self, args):
        super(MOCDecoder, self).__init__()

        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.num_experts = args['num_mos_decoders']
        self.split_hidden = args['split_hidden']


        self.prior = nn.Linear(self.input_dim, self.num_experts)
        self.latent = nn.Sequential(nn.Linear(self.input_dim,self.num_experts*self.input_dim),nn.Tanh())
        self.decoder = nn.Linear(self.input_dim,self.output_dim)



    def forward(self, hidden_states):
        batch_size,maxlen,hidden_size = hidden_states.size()

        if self.split_hidden :
            mixture_hidden_states = hidden_states[:, :, : int(hidden_size / 2)]
            mixture_logits = self.prior(mixture_hidden_states).contiguous().view(-1, self.num_experts)
        else:
            mixture_logits = self.prior(hidden_states).contiguous().view(-1,self.num_experts)



        latent = self.latent(hidden_states).view(-1,self.input_dim,self.num_experts)

        mixture_coefficients = nn.functional.softmax(mixture_logits,-1)

        mixture_coefficients = mixture_coefficients.unsqueeze(1).expand_as(latent)

        latent =  (latent * mixture_coefficients).sum(2)



        logits = self.decoder(latent.view(-1,self.input_dim))


        logits = logits.view(batch_size,maxlen,self.output_dim)


        output = {}
        output['logits'] = logits
        return output



class MultiTaskLinearDecoder(nn.Module):

    def __init__(self, args):
        super(MultiTaskLinearDecoder, self).__init__()
        book_args,author_args = {},{}

        self.is_share = False
        self.book_projection = nn.Linear(in_features=args['book_input_dim']+args['author_input_dim'],
                                         out_features=args['book_input_dim'])
        book_args['input_dim'] = args['book_input_dim']
        book_args['output_dim'] = args['book_output_dim']
        self.book_decoder = LinearDecoder(book_args)
        self.tanh = nn.Tanh()

        author_args['input_dim'] = args['author_input_dim']
        author_args['output_dim'] = args['author_output_dim']
        self.author_decoder = LinearDecoder(author_args)

        self.BOOK_TASK_ID = 'book'
        self.AUTHOR_TASK_ID = 'author'

        self.book2author = args['book2author']
        self.author2book_indices = [self.book2author[book_id] for book_id in range(0, len(self.book2author))]
        self.is_combine_logits = args['combine_logits']

    def forward(self, hidden_states):
        author_logits = self.author_decoder(hidden_states)['logits']

        if self.is_share :
            hidden_states = self.tanh(self.book_projection(torch.cat([hidden_states,
                                                            self.author_decoder.intermediate_hidden_states],dim=2)))
        book_logits = self.book_decoder(hidden_states)['logits']
        if self.is_combine_logits : book_logits = self.combine_logits(book_logits,author_logits)
        output = {self.BOOK_TASK_ID:{},self.AUTHOR_TASK_ID:{}}
        output[self.BOOK_TASK_ID]['logits'] = book_logits
        output[self.AUTHOR_TASK_ID]['logits'] = author_logits
        return output

    def combine_logits(self,book_logits,author_logits):
        # book_logits shape : batch_size,maxlen,book_vocab_size
        # author_logits shape : batch_size,maxlen,author_vocab_size
        return author_logits[:,:,self.author2book_indices] + book_logits



