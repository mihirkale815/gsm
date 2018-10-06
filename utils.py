import copy
import torch
import torch.nn as nn
import numpy as np
from collections import Counter,defaultdict
import bottleneck


def display_metrics(metrics,split):
    print('>'*3,split,'<'*5)
    for metric,val in metrics.items():
        print(metric,val)


def move_batch_to_device(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key,val in batch.items():
        if type(val) == torch.Tensor: batch[key] = val.to(device)
    return batch

def calculate_metrics(actual,predicted,topklist=(10,)):
    metrics = {'hitrate':{}}
    for topk in topklist:
        metrics['hitrate'][topk] = calculate_hitrate(actual,predicted[:,0:topk])
    return metrics

def calculate_hitrate(actual,predicted):
    correct = len([1 for act,pred in zip(actual,predicted) if act in pred])
    return float(correct)/len(actual)


def sample_negatives_slow(predictions,targets):
    num_samlpes, vocab_size = predictions.shape
    sampled_predictions = np.zeros((num_samlpes, 100))
    for idx, target in enumerate(targets):
        candidates = [i for i in range(1, vocab_size) if i != target]
        negatives = np.random.choice(candidates, 99, replace=False)
        sampled_predictions[idx] = predictions[idx][np.append(target, negatives)]
    return sampled_predictions

def sample_negatives(predictions,targets,topk=10):
    #implementation is wrong
    num_samlpes, vocab_size = predictions.shape
    rows = np.arange(num_samlpes).repeat(topk).reshape(num_samlpes, topk)
    cols = np.random.randint(1,vocab_size,(num_samlpes, topk))
    cols[:,0] = targets
    return predictions[rows,cols],np.zeros(num_samlpes)

def process_predictions(batch,predictions,topk = 10):

    batch_size, max_len, vocab_size = predictions.size()
    n_rows,n_cols = batch_size*max_len,vocab_size
    predictions = predictions.view(batch_size * max_len, vocab_size).detach().cpu().numpy()
    targets = batch['outputs'].view(batch_size * max_len).detach().cpu().numpy()

    topk_unsorted_indices = bottleneck.argpartition(-predictions,topk,axis=1)[:,0:topk]
    rows = np.arange(n_rows).repeat(topk).reshape(n_rows,topk)
    topk_indices_sorted_relative = predictions[rows,topk_unsorted_indices].argsort(axis=1)
    topk_indices_sorted = topk_unsorted_indices[rows,topk_indices_sorted_relative]
    #predictions = topk_indices_sorted
    predictions = topk_unsorted_indices

    masks = batch['masks'].view(batch_size * max_len).detach().cpu().numpy()
    predicted = predictions[masks == 1]
    actual = targets[masks==1]
    return actual,predicted

def evaluate(model,dataset_iter):
    torch.set_grad_enabled(False)
    model.eval()
    predicted,actual = [],[]
    for idx,batch in enumerate(dataset_iter):
        batch = move_batch_to_device(batch)
        predictions = model(batch)
        act,pred = process_predictions(batch,predictions)
        predicted.append(pred)
        actual.append(act)
    predicted = np.vstack(predicted)
    actual = np.hstack(actual)
    print("num interactions",predicted.shape, actual.shape)
    metrics = calculate_metrics(actual,predicted)
    return actual,predicted,metrics