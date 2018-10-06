import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LinearDecoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decoder = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, hidden_states):
        return self.decoder(hidden_states)