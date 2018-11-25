import json
from datetime import datetime
from collections import namedtuple
from collections import Counter, defaultdict
import torch
import numpy as np
import json
import pickle
import copy
from random import shuffle

PAD_TOKEN_ID = 0


def create_vocabulary_from_interactions(sequence_interactions):
    item2id = {'pad':PAD_TOKEN_ID}
    for sequence_interaction in sequence_interactions:
        for interaction in sequence_interaction:
            item = interaction
            if item not in item2id: item2id[item] = len(item2id)
    id2item = {v:k for k,v in item2id.items()}

    return item2id,id2item



def filter_by_count(sequence_interactions):
    book2count = defaultdict(int)
    total_events = 0
    for seq in sequence_interactions:
        for event in seq:
            if event.split == 'train' : book2count[event.book_id] += 1
            total_events += 1
    print("events before filtering =",total_events)

    filtered_sequence_interactions = [[event for event in seq if book2count[event.book_id]>=20]
                                      for seq in sequence_interactions]
    print("events after filtering =",sum([len(seq) for seq in filtered_sequence_interactions]))
    return filtered_sequence_interactions



def pad_sequence(sequence,pad_token,max_len):
    sequence += [pad_token for i in range(max_len - len(sequence))]
    return sequence

def pad_sequences(sequences,pad_token,max_len):
    sequences += [[pad_token] for i in range(max_len - len(sequences))]
    return sequences



class DatasetContainer():

    def __init__(self,batch_size):
        self.datasets = {}
        self.freeze_vocab = False
        self.batches = {}
        self.batch_size = batch_size
        self.BOOK_TASK_ID = 'book'
        self.AUTHOR_TASK_ID = 'author'
        self.Event = namedtuple('Event', 'book_id author_id ts split')

    def set_identifiers(self,is_book_task,is_author_task):
        self.is_book_task = is_book_task
        self.is_author_task = is_author_task
        self.is_mtl = self.is_book_task and self.is_author_task

    def create_vocab(self,interaction_sequences):
        if self.freeze_vocab : raise ValueError("Vocab has been freezed")

        book_interaction_sequences = [ [event.book_id for event in seq] for seq in interaction_sequences]
        self.book2id, self.id2book = create_vocabulary_from_interactions(book_interaction_sequences)

        author_interaction_sequences = [[event.author_id for event in seq] for seq in interaction_sequences]
        self.author2id, self.id2author = create_vocabulary_from_interactions(author_interaction_sequences)

        self.book2author = self.create_book2author_mapping(interaction_sequences,self.book2id,self.author2id)
        self.freeze_vocab = True

    def create_book2author_mapping(self,interaction_sequences,book2id,author2id):
        book2author = {PAD_TOKEN_ID:PAD_TOKEN_ID}
        for seq in interaction_sequences:
            for event in seq:
                book_idx = book2id[event.book_id]
                author_idx = author2id[event.author_id]
                book2author[book_idx] = author_idx
        return book2author


    def convert_to_event(self,tup):
        return self.Event(book_id=tup[0], ts=tup[1], author_id=tup[2], split=tup[4])

    def load_sequences(self,data_path,sort=True):
        print("loading sequences....")
        f = open(data_path)

        interaction_sequences = [[self.convert_to_event(event)
                                  for event in json.loads(line)]
                                 for line in open(data_path)]
        interaction_sequences = filter_by_count(interaction_sequences)
        self.create_vocab(interaction_sequences)
        print("loading complete")

        print("shuffling sequences...")
        np.random.seed(0)
        np.random.shuffle(interaction_sequences)
        print("shuffling comlpete")

        print("sorting sequences by length...")
        if sort : interaction_sequences = sorted(interaction_sequences,key=lambda x : len(x))
        self.datasets = interaction_sequences
        print("sequences sorted...")

        self.batches = {}

        self.batches['train'] = self.prepare_batches(self.datasets,self.batch_size,'train')
        self.batches['dev'] =self.prepare_batches(self.datasets,self.batch_size, 'dev')
        self.batches['test'] = self.prepare_batches(self.datasets, self.batch_size, 'test')

    def batch_iterator(self,split):
        shuffle(self.batches[split])
        for batch in self.batches[split]:
            yield batch

    def prepare_batches(self,iseqs,batch_size,split):
        print("preparing batches for",split)
        num_batches = int(len(iseqs)/batch_size)
        batch_start_indices = [batch_size*i for i in range(num_batches)]
        batches = []
        for batch_start_index in batch_start_indices:
            end = min(len(iseqs),batch_start_index+batch_size)
            batches.append(self.batchify(iseqs[batch_start_index:end],split))
        print("number of interactions = ", sum([batch['masks'].sum() for batch in batches]))
        return batches

    def batchify(self,iseqs,split):

        lens = []
        book_inputs = []
        author_inputs = []

        book_targets = []
        author_targets = []
        masks = []

        for iseq in iseqs:
            book_inp = []
            author_inp = []
            book_target = []
            author_target = []
            mask = []

            for i in range(0,len(iseq)-1):
                intc,next_intc = iseq[i],iseq[i+1]
                if split == 'train' and intc.split != split : continue
                if split == 'dev' and intc.split == 'test': continue
                book_inp.append(self.book2id[intc.book_id])
                book_target.append(self.book2id[next_intc.book_id])

                author_inp.append(self.author2id[intc.author_id])
                author_target.append(self.author2id[next_intc.author_id])

                mask.append(1 if split==intc.split else 0)

            if len(book_inp) == 0 : continue
            lens.append(len(book_inp))
            book_inputs.append(book_inp)
            author_inputs.append(author_inp)
            book_targets.append(book_target)
            author_targets.append(author_target)
            masks.append(mask)


        return self.serial_batch(lens,book_inputs,book_targets,author_inputs,author_targets,masks)

    def serial_batch(self,lens,book_inputs,book_targets,author_inputs,author_targets,masks):
        lens = np.array(lens)
        maxlen = max(lens)

        sorted_indices = np.argsort(-lens)

        batch = {self.BOOK_TASK_ID : {}, self.AUTHOR_TASK_ID: {}}
        batch['lens'] = list(map(int, lens[sorted_indices]))

        batch[self.BOOK_TASK_ID]['inputs'] = convert_to_tensor(book_inputs,maxlen,sorted_indices)
        batch[self.BOOK_TASK_ID]['targets'] = convert_to_tensor(book_targets,maxlen,sorted_indices)

        batch[self.AUTHOR_TASK_ID]['inputs'] = convert_to_tensor(author_inputs, maxlen, sorted_indices)
        batch[self.AUTHOR_TASK_ID]['targets'] = convert_to_tensor(author_targets, maxlen, sorted_indices)

        batch['masks'] = convert_to_tensor(masks,maxlen,sorted_indices)
        batch[self.AUTHOR_TASK_ID]['masks'] = batch['masks']
        batch[self.BOOK_TASK_ID]['masks'] = batch['masks']

        return batch

class MovielensDatasetContainer():

    def __init__(self,batch_size):
        self.datasets = {}
        self.freeze_vocab = False
        self.batches = {}
        self.batch_size = batch_size
        self.MOVIE_TASK_ID = 'movie'
        self.GENRE_TASK_ID = 'genre'
        self.ignore_genres = ['IMAX','(no genres listed)','Film-Noir','Western']

    def create_vocab(self,interaction_sequences):
        if self.freeze_vocab : raise ValueError("Vocab has been freezed")

        movie_interaction_sequences = [ [event['movie_id'] for event in seq] for seq in interaction_sequences]
        self.movie2id, self.id2movie = create_vocabulary_from_interactions(movie_interaction_sequences)

        genres_interaction_sequences = [[genre for event in seq for genre in event['genres']] for seq in interaction_sequences]
        self.genre2id, self.id2genre = create_vocabulary_from_interactions(genres_interaction_sequences)
        self.freeze_vocab = True

    def load_sequences(self,data_path,sort=True):
        print("loading sequences....")
        f = open(data_path)

        interaction_sequences = [[self.convert_to_event(event)
                                  for event in json.loads(line)]
                                 for line in open(data_path)]
        #interaction_sequences = filter_by_count(interaction_sequences)
        self.create_vocab(interaction_sequences)
        print("loading complete")

        print("shuffling sequences...")
        np.random.seed(0)
        np.random.shuffle(interaction_sequences)
        print("shuffling comlpete")

        print("sorting sequences by length...")
        if sort : interaction_sequences = sorted(interaction_sequences,key=lambda x : len(x))
        self.datasets = interaction_sequences
        print("sequences sorted...")

        self.batches = {}

        self.batches['train'] = self.prepare_batches(self.datasets,self.batch_size,'train')
        self.batches['dev'] =self.prepare_batches(self.datasets,self.batch_size, 'dev')
        self.batches['test'] = self.prepare_batches(self.datasets, self.batch_size, 'test')

    def batch_iterator(self,split):
        shuffle(self.batches[split])
        for batch in self.batches[split]:
            yield batch

    def prepare_batches(self,iseqs,batch_size,split):
        print("preparing batches for",split)
        num_batches = int(len(iseqs)/batch_size)
        batch_start_indices = [batch_size*i for i in range(num_batches)]
        batches = []
        for batch_start_index in batch_start_indices:
            end = min(len(iseqs),batch_start_index+batch_size)
            batches.append(self.batchify(iseqs[batch_start_index:end],split))
        print("number of interactions = ", sum([batch[self.MOVIE_TASK_ID]['masks'].sum() for batch in batches]))
        return batches


    def convert_to_event(self,row):
        self.allowed_genres = set(['Adventure',
         'Documentary',
         'Horror',
         'Crime',
         'Action',
         'Romance',
         'Thriller',
         'Comedy',
         'Drama',
         'Other'])
        user_id = row[0]
        movie_id = row[1]
        genres = row[2] + ['Other']
        genres = [genre for genre in genres if genre in self.allowed_genres]
        ts = row[3]
        split = row[4]
        return {'movie_id':movie_id,'split':split,'genres':genres}


    def set_identifiers(self,is_movie_task,is_genre_task):
        self.is_movie_task = is_movie_task
        self.is_genre_task = is_genre_task
        self.is_mtl = self.is_genre_task and self.is_movie_task



    def batchify(self,iseqs,split):

        lens = []
        movie_inputs = []
        genre_inputs = []

        movie_targets = []
        genre_targets = []
        movie_masks = []
        genre_masks = []

        for iseq in iseqs:
            movie_inp = []
            genre_inp = []
            movie_target = []
            genre_target = []
            genre_mask = []
            movie_mask = []

            for i in range(0,len(iseq)-1):
                intc,next_intc = iseq[i],iseq[i+1]
                if split == 'train' and intc['split'] != split : continue
                if split == 'dev' and intc['split'] == 'test': continue
                movie_inp.append(self.movie2id[intc['movie_id']])
                movie_target.append(self.movie2id[next_intc['movie_id']])

                genre_inp.append([self.genre2id[genre] for genre in intc['genres']])
                genre_target.append([self.genre2id[genre] for genre in next_intc['genres']])


                movie_mask.append(1 if split==intc['split'] else 0)
                genre_mask.append([1]*len(intc['genres']) if split == intc['split'] else [0]*len(intc['genres']))

            if len(movie_inp) == 0 : continue
            lens.append(len(movie_inp))

            movie_inputs.append(movie_inp)
            genre_inputs.append(genre_inp)

            movie_targets.append(movie_target)
            genre_targets.append(genre_target)

            movie_masks.append(movie_mask)
            genre_masks.append(genre_mask)

        return self.serial_batch(lens,movie_inputs,movie_targets,genre_inputs,genre_targets,movie_masks,genre_masks)

    def serial_batch(self,lens,movie_inputs,movie_targets,genre_inputs,genre_targets,movie_masks,genre_masks):
        lens = np.array(lens)
        maxlen = max(lens)

        sorted_indices = np.argsort(-lens)

        batch = {self.MOVIE_TASK_ID : {}, self.GENRE_TASK_ID: {}}
        batch['lens'] = list(map(int, lens[sorted_indices]))

        batch[self.MOVIE_TASK_ID]['inputs'] = convert_to_tensor(movie_inputs,maxlen,sorted_indices)
        batch[self.MOVIE_TASK_ID]['targets'] = convert_to_tensor(movie_targets,maxlen,sorted_indices)

        max_genres_inp = max([len(genres) for inp in genre_inputs for genres in inp])
        max_genres_targ = max([len(genres) for targ in genre_targets for genres in targ])
        max_genres = max(max_genres_inp,max_genres_targ)

        batch[self.GENRE_TASK_ID]['inputs'] = convert_to_tensor_genres(genre_inputs, max_genres,maxlen, sorted_indices)
        batch[self.GENRE_TASK_ID]['targets'] = convert_to_tensor_genres(genre_targets, max_genres,maxlen, sorted_indices)

        batch[self.MOVIE_TASK_ID]['masks'] = convert_to_tensor(movie_masks,maxlen,sorted_indices)
        batch[self.GENRE_TASK_ID]['masks'] = convert_to_tensor_genres(genre_masks, max_genres,maxlen, sorted_indices)

        return batch

def convert_to_tensor_genres(data,max_genres,max_len,indices=None):
    data = [pad_sequences(seq,PAD_TOKEN_ID,max_len) for seq in data]
    data = [[pad_sequence(genre_list,PAD_TOKEN_ID,max_genres) for genre_list in seq]
                    for seq in data]
    data = torch.tensor(data)
    data = data[indices,:]
    return data

def convert_to_tensor(data,maxlen,indices):
    data = [pad_sequence(row, PAD_TOKEN_ID, maxlen) for row in data]
    data = torch.tensor(data)
    data = data[indices, :]
    return data



def sample_classes(targets,vocab_size,K=100,ignore_index=PAD_TOKEN_ID):
    num_samples = len(targets)
    sampled_classes = np.zeros((num_samples, K))
    for idx, target in enumerate(targets):
        candidates = [i for i in range(0, vocab_size) if i != target and i!= ignore_index]
        negatives = np.random.choice(candidates, K-1, replace=False)
        sampled_classes[idx] = negatives
    return sampled_classes




