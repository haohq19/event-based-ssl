import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from spikingjelly.datasets import dvs128_gesture, n_mnist

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


def transform_event_list(x, H, W, seq_len): 
    x = np.stack([x['t'], x['x'], x['y'], x['p']], axis=0, dtype=np.float32)  # x.shape = [4, max_seq_len]
    x = x[:, :seq_len + 1]  # select first seq_len + 1 events
    x = torch.from_numpy(x.T)  # x.shape = [seq_len + 1, 4]
    start = x[0, 0]  # start time
    # normalize t to relative time
    x[:, 0] = x[:, 0] - start  
    # normalize x, y to [0, 1]
    x[:, 1] = x[:, 1] / H
    x[:, 2] = x[:, 2] / W
    return x

def get_data_loader(args):
    if args.dataset == 'dvs128_gesture':
        dataset = dvs128_gesture.DVS128Gesture
        H, W = dataset.get_H_W()
        seq_len = args.ctx_len
        transform = partial(transform_event_list, H=H, W=W, seq_len=seq_len)
        train_dataset = dataset(root=args.root, train=True, data_type='event', transform=transform)
        val_dataset = dataset(root=args.root, train=False, data_type='event', transform=transform)
    elif args.dataset == 'n_mnist':
        dataset = n_mnist.NMNIST
        H, W = dataset.get_H_W()
        seq_len = args.seq_len
        transform = partial(transform_event_list, H=34, W=34, seq_len=seq_len)
        train_dataset = dataset(root=args.root, train=True, data_type='event', transform=transform)
        val_dataset = dataset(root=args.root, train=False, data_type='event', transform=transform)
    else:
        raise NotImplementedError(args.dataset)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=pad_sequence_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=pad_sequence_collator)

    return train_loader, val_loader