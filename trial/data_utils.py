import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import numpy as np
import json

def sentence_generator(path):
    f = open(path)
    for line in f:
        yield json.loads(line)
    f.close()


def create_vocabulary_from_files(file_paths):

    item2id = {'pad':0}

    for file_path in file_paths:
        gen = sentence_generator(file_path)
        for user_id,tagged_sentence in gen:
            for item,rating,ts,split in tagged_sentence:
                if item not in item2id: item2id[item] = len(item2id)


    id2item = {v:k for k,v in item2id.items()}

    return item2id,id2item


def create_rows(file_path,item2id):
    rows = []
    gen = sentence_generator(file_path)
    for user_id,seq in gen:
        items = []
        ratings = []
        user_ids = []
        splits = []
        tss = []
        for (item,rating,ts,split) in seq:
            if item not in item2id : continue
            items.append(item2id[item])
            ratings.append(rating)
            splits.append(split)
            tss.append(ts)
            user_ids.append(user_id)
        seq_length = len(items)
        rows.append({'user_id':user_ids,'items':items, 'ratings':ratings, 'len':seq_length, 'tss':tss, 'splits':splits})
    return rows



def pad_sequence(sequence,pad_token,max_len):
    sequence += [pad_token for i in range(max_len - len(sequence))]
    return sequence



class DatasetContainer():

    def __init__(self):
        self.datasets = {}
        self.freeze_vocab = False
        self.item2id, self.id2item = {}, {}

    def create_vocab(self,vocab_file_paths):
        if self.freeze_vocab : raise ValueError("Vocab has been freezed")
        self.item2id, self.id2item = create_vocabulary_from_files(vocab_file_paths)
        self.freeze_vocab = True

    def set_word_vocab(self,word2id):
        self.item2id  = word2id
        self.id2item = {v:k for k, v in self.item2id.items()}

    def set_tag_vocab(self,tag2id):
        self.tag2id  = tag2id
        self.id2tag = {v:k for k,v in self.tag2id.items()}

    def add_split(self,data_path,key,sort=True):
        rows = create_rows(data_path, self.item2id)
        if sort : rows = sorted(rows,key=lambda x : x['len'])
        self.datasets[key] = SequenceTaggingDataset(rows)

    def get_batch_iterator(self,key,batch_size):
        return DataLoader(self.datasets[key], batch_size=batch_size,
                          shuffle=False, sampler=None,
                          batch_sampler=None, collate_fn=collate_pad)


class SequenceTaggingDataset(Dataset):

    def __init__(self,rows):
        self.rows = rows

    def __getitem__(self,index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)










