import json
from datetime import datetime
from interactions import Interaction, InteractionSequence
from collections import Counter, defaultdict
import torch
import numpy as np
import json
import pickle
import copy
from random import shuffle

PAD_TOKEN_ID = 0

def filter_user(interaction_sequence,
                min_dt,
                max_dt,
                max_user_ratings,
                min_user_ratings,
                min_active_years):
    uid = interaction_sequence.uid
    interaction_sequence = InteractionSequence(uid,
                                               [intc for intc in interaction_sequence if min_dt <= intc.dt <= max_dt])
    if len(interaction_sequence) > max_user_ratings: return None
    if len(interaction_sequence) < min_user_ratings: return None
    max_obs_year = interaction_sequence.get_max_dt().year
    min_obs_year = interaction_sequence.get_min_dt().year
    if max_obs_year != max_dt.year: return None
    if max(max_obs_year) - min(min_obs_year) < min_active_years: return None
    return interaction_sequence


def create_split(interaction_sequences, fold, min_dt, max_dt):
    for interaction_sequence in interaction_sequences:
        for interaction in interaction_sequence:
            if min_dt <= interaction <= max_dt:
                interaction.fold = fold
    return interaction_sequences


def remove_oov_items(interaction_sequences):
    train_items = set([[intc.pid for intc in interaction_sequence if intc.fold == 'train'] for interaction_sequence in
                       interaction_sequences])
    rest_items = set([[intc.pid for intc in interaction_sequence if intc.fold != 'train'] for interaction_sequence in
                      interaction_sequences])
    oov_items = rest_items.difference(train_items)
    for interaction_sequence in interaction_sequences:
        interaction_sequence.remove_items(oov_items)
    return interaction_sequences



def create_vocabulary_from_interactions(sequence_interactions):

    item2id = {'pad':PAD_TOKEN_ID}
    for sequence_interaction in sequence_interactions:
        for interaction in sequence_interaction:
            item = interaction.pid
            if item not in item2id: item2id[item] = len(item2id)
    id2item = {v:k for k,v in item2id.items()}

    return item2id,id2item



def pad_sequence(sequence,pad_token,max_len):
    sequence += [pad_token for i in range(max_len - len(sequence))]
    return sequence






class DatasetContainer():

    def __init__(self):
        self.datasets = {}
        self.freeze_vocab = False
        self.item2id, self.id2item = {}, {}
        self.batches = {}
        self.batch_size = 16

    def create_vocab(self,data_path):
        interaction_sequences = pickle.load(open(data_path,"rb"))
        if self.freeze_vocab : raise ValueError("Vocab has been freezed")
        self.item2id, self.id2item = create_vocabulary_from_interactions(interaction_sequences)
        self.freeze_vocab = True

    def set_item_vocab(self,item2id):
        self.item2id  = item2id
        self.id2item = {v:k for k, v in self.item2id.items()}

    def load_sequences(self,data_path,sort=True):
        interaction_sequences = pickle.load(open(data_path,"rb"))
        np.random.shuffle(interaction_sequences)
        self.iseqs = interaction_sequences

        seen_partition_index = int(0.9*len(self.iseqs))

        interaction_sequences = self.iseqs[0:seen_partition_index]
        if sort : interaction_sequences = sorted(interaction_sequences,key=lambda x : len(x))
        self.datasets['seen'] = interaction_sequences

        interaction_sequences = self.iseqs[seen_partition_index:]
        if sort: interaction_sequences = sorted(interaction_sequences, key=lambda x: len(x))
        '''for i in range(0,len(interaction_sequences)):
            for intc in interaction_sequences[i]:
                intc.fold = 'test'''
        self.datasets['unseen'] = interaction_sequences

        self.batches = {'seen':{},'unseen':{}}


        self.batches['seen']['train'] = self.prepare_train_batches(self.datasets['seen'],self.batch_size,'train')
        self.batches['seen']['val'] =self.prepare_batches(self.datasets['seen'],self.batch_size, 'val')
        self.batches['seen']['test'] = self.prepare_batches(self.datasets['seen'], self.batch_size, 'test')

        self.batches['unseen']['train'] = self.prepare_train_batches(self.datasets['unseen'], self.batch_size, 'train')
        self.batches['unseen']['val'] = self.prepare_batches(self.datasets['unseen'], self.batch_size, 'val')
        self.batches['unseen']['test'] = self.prepare_batches(self.datasets['unseen'],self.batch_size, 'test')




    def batch_iterator(self,key,period):
        shuffle(self.batches[key][period])
        for batch in self.batches[key][period]:
            yield batch

    def prepare_batches(self,iseqs,batch_size,split):
        print("Preparing batches for",split)
        num_batches = int(len(iseqs)/batch_size)
        batch_start_indices = [batch_size*i for i in range(num_batches)]
        batches = []
        for batch_start_index in batch_start_indices:
            end = min(len(iseqs),batch_start_index+batch_size)
            batches.append(self.batchify(iseqs[batch_start_index:end],split))
        print("Number of interactions = ", sum([batch['masks'].sum() for batch in batches]))
        return batches

    def prepare_train_batches(self,iseqs,batch_size,split):
        print("Preparing batches for",split)
        num_batches = int(len(iseqs)/batch_size)
        batch_start_indices = [batch_size*i for i in range(num_batches)]
        batches = []
        for batch_start_index in batch_start_indices:
            end = min(len(iseqs),batch_start_index+batch_size)
            batches.append(self.batchify_train(iseqs[batch_start_index:end],split))
        print("Number of interactions = ", sum([batch['masks'].sum() for batch in batches]))
        return batches


    def batchify(self,iseqs,split):

        lens,inputs,outputs,masks = [],[],[],[]
        for iseq in iseqs:
            inp,outp,mask = [],[],[]
            for i in range(0,len(iseq)-1):
                intc,next_intc = iseq[i],iseq[i+1]
                if split == 'train' and intc.fold != split : continue
                if split == 'val' and intc.fold == 'test': continue
                inp.append(self.item2id[intc.pid])
                outp.append(self.item2id[next_intc.pid])
                mask.append(1 if split==intc.fold else 0)
            lens.append(len(inp))
            inputs.append(inp)
            outputs.append(outp)
            masks.append(mask)

        lens = np.array(lens)
        maxlen = max(lens)

        pad_token = PAD_TOKEN_ID
        sorted_indices = np.argsort(-lens)

        batch = {}
        batch['lens'] = list(map(int, lens[sorted_indices]))

        inputs = [pad_sequence(inp, pad_token, maxlen) for inp in inputs]
        inputs = torch.tensor(inputs)
        inputs = inputs[sorted_indices, :]
        batch['inputs'] = inputs

        outputs = [pad_sequence(outp, pad_token, maxlen) for outp in outputs]
        outputs = torch.tensor(outputs)
        outputs = outputs[sorted_indices, :]
        batch['outputs'] = outputs

        masks = [pad_sequence(mask, pad_token, maxlen) for mask in masks]
        masks = torch.tensor(masks)
        masks = masks[sorted_indices,:]
        batch['masks'] = masks

        return batch


    def batchify_train(self,iseqs,split):

        lens,inputs,outputs,masks = [],[],[],[]
        for iseq in iseqs:
            inp,outp,mask = [],[],[]
            for i in range(0,len(iseq)-1):
                intc,next_intc = iseq[i],iseq[i+1]
                if intc.fold != 'train' : continue
                inp.append(self.item2id[intc.pid])
                outp.append(self.item2id[next_intc.pid])
                mask.append(1 if split==intc.fold else 0)
            lens.append(len(inp))
            inputs.append(inp)
            outputs.append(outp)
            masks.append(mask)

        lens = np.array(lens)
        maxlen = max(lens)

        pad_token = PAD_TOKEN_ID
        sorted_indices = np.argsort(-lens)

        batch = {}
        batch['lens'] = list(map(int, lens[sorted_indices]))

        inputs = [pad_sequence(inp, pad_token, maxlen) for inp in inputs]
        inputs = torch.tensor(inputs)
        inputs = inputs[sorted_indices, :]
        batch['inputs'] = inputs

        outputs = [pad_sequence(outp, pad_token, maxlen) for outp in outputs]
        outputs = torch.tensor(outputs)
        outputs = outputs[sorted_indices, :]
        batch['outputs'] = outputs

        masks = [pad_sequence(mask, pad_token, maxlen) for mask in masks]
        masks = torch.tensor(masks)
        masks = masks[sorted_indices,:]
        batch['masks'] = masks

        return batch







def sample_classes(targets,vocab_size,K=100,ignore_index=PAD_TOKEN_ID):
    num_samples = len(targets)
    sampled_classes = np.zeros((num_samples, K))
    for idx, target in enumerate(targets):
        candidates = [i for i in range(0, vocab_size) if i != target and i!= ignore_index]
        negatives = np.random.choice(candidates, K-1, replace=False)
        sampled_classes[idx] = negatives
    return sampled_classes




