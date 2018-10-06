from utils import *
import pickle


class HierarchicalDatasetContainer():

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