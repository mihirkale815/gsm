import torch.nn as nn
import torch

def cross_entropy(actual,predicted,mask=None,pad_index=0):
    batch_size, max_len, vocab_size = predicted.size()
    actual = actual*mask
    predicted = predicted.view(batch_size * max_len,vocab_size)
    actual =  actual.view(batch_size * max_len)
    loss = nn.CrossEntropyLoss(ignore_index=pad_index)(predicted,actual)
    return loss