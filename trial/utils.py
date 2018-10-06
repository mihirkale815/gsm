import torch
import copy
import os
import numpy as np
import json

def move_batch_to_device(batch,device):
    new_batch = copy.deepcopy(batch)
    new_batch['mask'] = new_batch['mask'].to(device)
    new_batch['item_id_seqs'] = new_batch['item_id_seqs'].to(device)
    if 'ratings_seqs' in batch : new_batch['ratings_seqs'] = new_batch['ratings_seqs'].to(device)
    return new_batch


def get_output(model,dataset_iter):
    model.eval()
    Ratings = []
    Preds = []
    Items = []
    Users = []
    for batch in dataset_iter:
        lens = batch['lens']
        max_len = max(lens)
        batch_size = len(lens)
        batch = move_batch_to_device(batch, model.device)
        predictions = list(model(batch).view(batch_size*max_len))
        ratings = list(batch['ratings_seqs'].view(batch_size*max_len))
        items = list(batch['item_seqs'].view(batch_size*max_len))
        users = list(batch['user_ids'].view(batch_size*max_len))
        Users.extend(users)
        Ratings.extend(ratings)
        Preds.extend(predictions)
        Items.extend(items)
    return Ratings,Preds,Items

def get_loss_for_dataset(data_iter,model):
    model.eval()
    epoch_loss = 0
    for idx ,batch in enumerate(data_iter):
        batch = move_batch_to_device(batch,model.device)
        loss = model.loss(batch)
        epoch_loss+=loss.item()
    return epoch_loss




