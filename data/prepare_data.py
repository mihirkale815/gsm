import json
from datetime import datetime
from interactions import Interaction, InteractionSequence
from collections import Counter, defaultdict
import numpy as np
import pickle

def filter_user(interaction_sequence,
                min_dt,
                max_dt,
                max_user_ratings,
                min_user_ratings,
                min_active_years):
    interaction_sequence.remove_after(max_dt)
    interaction_sequence.remove_before(min_dt)
    if len(interaction_sequence) > max_user_ratings: return None
    if len(interaction_sequence) < min_user_ratings: return None
    max_obs_year = interaction_sequence.get_max_dt().year
    min_obs_year = interaction_sequence.get_min_dt().year
    if max_obs_year != max_dt.year: return None
    if max_obs_year - min_obs_year + 1 < min_active_years: return None
    return interaction_sequence


def create_split(interaction_sequences, fold, min_dt, max_dt):
    for interaction_sequence in interaction_sequences:
        for interaction in interaction_sequence.interactions:
            if min_dt <= interaction.dt <= max_dt:
                interaction.fold = fold
    return interaction_sequences


def remove_oov_items(interaction_sequences):
    train_items = set([ intc.pid  for interaction_sequence in interaction_sequences
                        for intc in interaction_sequence.interactions if intc.fold == 'train'])
    rest_items = set([intc.pid for interaction_sequence in interaction_sequences
                       for intc in interaction_sequence.interactions if intc.fold != 'train'])
    oov_items = rest_items.difference(train_items)
    for interaction_sequence in interaction_sequences:
        interaction_sequence.remove_items(oov_items)
    return interaction_sequences


def display_summary(interaction_sequences):
    print("number of sequences = ",len(interaction_sequences))
    num_train_intc,num_val_intc,num_test_intc = 0,0,0
    item_set = set()
    users_train,users_val,users_test = set(),set(),set()
    pid_counts = defaultdict(int)
    for interaction_sequence in interaction_sequences:
        uid = interaction_sequence.uid
        num_test_intc-=1 # sequence of length n gives n-1 samples
        for intc in interaction_sequence.interactions:
            pid_counts[intc.pid] += 1
            if intc.fold == 'train' :
                num_train_intc+=1
                users_train.add(uid)
            if intc.fold == 'val':
                num_val_intc += 1
                users_val.add(uid)
            if intc.fold == 'test':
                num_test_intc += 1
                users_test.add(uid)
            item_set.add(intc.pid)
    num_items = len(item_set)
    num_users_train, num_users_val, num_users_test = len(users_train),len(users_val),len(users_test)
    print("train interactions = ", num_train_intc)
    print("validation interactions = ", num_val_intc)
    print("test interactions = ", num_test_intc)
    print("number of items = ", num_items)
    print("number of users in train,val,test = ",num_users_train, num_users_val, num_users_test)
    print(np.histogram(list(pid_counts.values()),bins=[0,10,50,100,1000]))

args = None
input_path = "/Users/mihirkale.s/PycharmProjects/srm/data/amazon/books.json" #args.input_path
output_path = "/Users/mihirkale.s/PycharmProjects/srm/data/amazon/books.pkl" #args.output_path
min_item_count = 40 #args.min_item_count
min_user_ratings = 20 #args.min_user_ratings
max_user_ratings = 500 #args.max_user_ratings

min_val_year = 2014 #args.val_year
min_val_month = 3 #args.val_month
max_val_year = 2014 #args.val_year
max_val_month = 3 #args.val_month

min_test_year = 2014 #args.test_year
min_test_month = 4 #args.test_month
max_test_year = 2014 #args.test_year
max_test_month = 8 #args.test_month

min_year = 2004
max_year = 2014
min_active_years = 4
min_dt = datetime(min_year,1,1)
max_dt = datetime(max_year,8,1)

min_val_dt = datetime(min_val_year,min_val_month,1)
max_val_dt = datetime(max_val_year,max_val_month,30)
min_test_dt = datetime(min_test_year,min_test_month,30)
max_test_dt = datetime(max_test_year,max_test_month,30)

userdatalist = [json.loads(line.strip("\n")) for line in open(input_path)]
interaction_sequences = []
for data in userdatalist:
    uid = data['uid']
    interaction_sequence = InteractionSequence(uid,[ Interaction(uid,pid,ts) for ts,pid in data['history']])
    interaction_sequences.append(interaction_sequence)

print('*'*10,"Raw data",'*'*10)
display_summary(interaction_sequences)

#remove low frequency items
pid_counts = Counter([intc.pid for seq in interaction_sequences for intc in seq.interactions])
pid_counts = {pid:count for pid,count in pid_counts.items() if count>min_item_count}
chosen_pids = set(pid_counts.keys())
for interaction_sequence in interaction_sequences:
    interaction_sequence.keep_items(chosen_pids)
print('*'*10,"Removed infrequent products",'*'*10)
display_summary(interaction_sequences)

# filter users based on various constraints
interaction_sequences = [filter_user(intc,min_dt,
                                    max_dt,
                                    max_user_ratings,
                                    min_user_ratings,
                                    min_active_years)
                         for intc in interaction_sequences]
interaction_sequences = [intc for intc in interaction_sequences if intc]
print('*'*10,"Filtered Users",'*'*10)
display_summary(interaction_sequences)

# tag interactions with split
interaction_sequences = create_split(interaction_sequences,'val',min_val_dt,max_val_dt)
interaction_sequences = create_split(interaction_sequences,'test',min_test_dt,max_test_dt)
print('*'*10,"Created Splits",'*'*10)
display_summary(interaction_sequences)


# remove items that are not present in train
interaction_sequences = remove_oov_items(interaction_sequences)
print('*'*10,"Removed OOV items",'*'*10)

display_summary(interaction_sequences)

pickle.dump(interaction_sequences,open(output_path,"wb"))