from utils import MovielensDatasetContainer
import random




def unstack_sequences(seqs):
    split_name = 'dev'
    useqs = []
    for idx,seq in enumerate(seqs):
        for event_idx,event in seq:
            if event['split'] == split_name:
                subseq = seq[0:event_idx+1]
                for i in range(0,event_idx-1) : subseq[i]['split'] = 'train'
                useqs.append(subseq)
    return useqs


def truncate_sequences(seqs,trunc_len):
    tseqs = [seq[-trunc_len:] for seq in seqs]
    tseqs = [seq for seq in tseqs if len(seq) == trunc_len]
    return tseqs

def partial_shuffle(seqs,index,mode='global'):
    shuff_seqs = []
    for seq in seqs:
        seq_part_1 = seq[0:index]
        seq_part_2 = seq[index:-1]
        target = [seq[-1]]
        if mode == 'global' : random.shuffle(seq_part_1)
        if mode == 'local' : random.shuffle(seq_part_2)
        shuff_seqs.append(seq_part_1 + seq_part_2 + target)
    return shuff_seqs



