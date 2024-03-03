import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def pad_sequence_collator(batch):
    '''
    pad sequence of different length to the same length
    batch: list of (data, label)
    data.shape = [seq_len, 4]
    label.shape = []
    '''
    data, labels = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels)
    return data, labels

def event_to_tensor(x): 
    x = torch.from_numpy(np.stack([x['t'], x['x'], x['y'], x['p']], axis=0).T)  # x.shape = [seq_len, 4]
    return x