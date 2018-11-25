import torch.nn as nn
import torch

def cross_entropy(actual,predicted,mask,pad_index=0):
    actual = actual*mask
    batch_size, max_len, vocab_size = predicted.size()
    predicted = predicted.contiguous().view(batch_size * max_len,vocab_size)
    actual =  actual.contiguous().view(batch_size * max_len)
    loss = nn.CrossEntropyLoss(ignore_index=pad_index,reduction='sum')(predicted,actual)
    return loss

