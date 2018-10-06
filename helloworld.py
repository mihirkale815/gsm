from spotlight.cross_validation import random_train_test_split,user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight import losses
from spotlight.sequence.representations import LSTMNet
from spotlight.interactions import Interactions
import pandas as pd
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.evaluation import sequence_mrr_score,sequence_precision_recall_score
import numpy as np

'''
dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)

model = ExplicitFactorizationModel(n_iter=3)
model.fit(train,verbose=True)

rmse = rmse_score(model, test)

print(rmse)
'''

def convert_to_int(keys):
    key2idx = {'dummy':0}
    for key in keys :
        if key not in key2idx : key2idx[key] = len(key2idx)
    return np.array([key2idx[key] for key in keys], dtype=np.int32),key2idx



random_state = np.random.RandomState(0)

df = pd.read_csv("../../data/amazon/movies.csv")
timestamps=df['ts'].values
item_ids,item2idx = convert_to_int(df['pid'].values)
user_ids,user2idx = convert_to_int(df['uid'].values)


#dataset = Interactions(timestamps=timestamps[0:1000],item_ids=item_ids[0:1000],user_ids=user_ids[0:1000])
dataset = Interactions(timestamps=timestamps,item_ids=item_ids,user_ids=user_ids)


train, rest = user_based_train_test_split(dataset,
                                          random_state=random_state)
test, validation = user_based_train_test_split(rest,
                                               test_percentage=0.2,
                                               random_state=random_state)

min_sequence_length = 10
max_sequence_length = 200
step_size = 1

train = train.to_sequence(max_sequence_length=max_sequence_length,
                          min_sequence_length=min_sequence_length,
                          step_size=step_size)
test = test.to_sequence(max_sequence_length=max_sequence_length,
                        min_sequence_length=min_sequence_length,
                        step_size=step_size)
validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)

net = LSTMNet(len(set(item2idx)), embedding_dim=32, item_embedding_layer=None, sparse=False)
model = ImplicitSequenceModel(loss='adaptive_hinge',
                              representation=net,
                              batch_size=32,
                              learning_rate=0.01,
                              l2=10e-6,
                              n_iter=10,
                              use_cuda=False,
                              random_state=random_state)
model.fit(train, verbose=True)

test_mrr = sequence_mrr_score(model, test)
val_mrr = sequence_mrr_score(model, validation)
train_mrr = sequence_mrr_score(model, train)

print(test_mrr.mean(),val_mrr.mean(),train_mrr.mean())

for (split,split_name) in ((train,"train"),(validation,"validation"),(test,"test")):
    for k in (5,10,50,100):
        precision,recall = sequence_precision_recall_score(model, split, k=k, exclude_preceding=False)
        print(split_name,"precision at",k,precision.mean())
        print(split_name,"recall at", k,recall.mean())

