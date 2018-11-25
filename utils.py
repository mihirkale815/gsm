import copy
import torch
import torch.nn as nn
import numpy as np
from collections import Counter,defaultdict
import bottleneck
import os
import json


def move_batch_to_device(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k,d in batch.items():
        if type(d) == torch.Tensor: batch[k] = d.to(device)
        if type(d) == dict:
            for key, val in d.items():
                if type(val) == torch.Tensor: d[key] = val.to(device)
            batch[k] = d
    return batch

def calculate_metrics(actual,predicted,topklist=(10,)):
    metrics = {'hitrate':{}}
    for topk in topklist:
        metrics['hitrate'][topk] = calculate_hitrate(actual,predicted[:,0:topk])
    return metrics

def calculate_hitrate(actual,predicted):
    correct = len([1 for act,pred in zip(actual,predicted) if act in pred])
    return float(correct)/len(actual)

def process_predictions(targets,predictions,masks,topk):

    batch_size, max_len, vocab_size = predictions.size()
    masks = masks.view(batch_size * max_len).detach().cpu().numpy()
    targets = targets.view(batch_size * max_len).detach().cpu().numpy()
    predictions = predictions.view(batch_size * max_len, vocab_size).detach().cpu().numpy()

    predictions = predictions[masks == 1]
    targets = targets[masks==1]


    n_rows, n_cols = predictions.shape
    topk_unsorted_indices = bottleneck.argpartition(-predictions,topk,axis=1)[:,0:topk]
    #rows = np.arange(n_rows).repeat(topk).reshape(n_rows,topk)
    #topk_indices_sorted_relative = predictions[rows,topk_unsorted_indices].argsort(axis=1)
    #topk_indices_sorted = topk_unsorted_indices[rows,topk_indices_sorted_relative]
    #predictions = topk_indices_sorted
    predictions = topk_unsorted_indices

    return targets,predictions


def evaluate(model,batch_iter,task_ids,save_path=None):
    torch.set_grad_enabled(False)
    model.eval()
    all_predicted = {task_id:[] for task_id in task_ids}
    all_targets = {task_id: [] for task_id in task_ids}
    losses = {task_id: [] for task_id in task_ids}
    num_elements = {task_id: 0 for task_id in task_ids}
    for idx,batch in enumerate(batch_iter):
        batch = move_batch_to_device(batch)
        output = model.compute_loss(batch)
        for task_id in task_ids:
            loss = output[task_id]['loss']
            predictions = output[task_id]['logits']
            targets = batch[task_id]['targets']
            masks = batch[task_id]['masks']
            losses[task_id].append(loss.item())
            num_elements[task_id] += batch[task_id]['masks'].sum().item()
            targs,preds = process_predictions(targets,predictions,masks,10)
            all_predicted[task_id].append(preds)
            all_targets[task_id].append(targs)

    result = {}
    for task_id in task_ids:
        predicted = np.vstack(all_predicted[task_id])
        targets = np.hstack(all_targets[task_id])
        metrics = {'loss': sum(losses[task_id]) / num_elements[task_id],
                   'metrics': calculate_metrics(targets, predicted)}
        result[task_id] = {'predicted':predicted,'target':targets,'metrics':metrics}
        if save_path:
            persist_model_output(targets, predicted, metrics, save_path)
    print("number of predictions = ",result[task_id]['target'].shape)
    return result


def persist_model_output(targets,predicted,metrics,save_path):
    np.savez(os.path.join(save_path,"targets"),targets)
    np.savez(os.path.join(save_path,"predicted"),predicted)
    json.dump(metrics,open(os.path.join(save_path,"metrics"),"w"))

