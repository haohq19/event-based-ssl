import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from spikingjelly.datasets import dvs128_gesture, n_mnist, n_caltech101, cifar10_dvs

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
    # normalize t to relative time
    x[:, 0] = x[:, 0] -  x[0, 0] 
    # normalize x, y to [0, 1]
    x[:, 1] = x[:, 1] / H
    x[:, 2] = x[:, 2] / W
    return x


def split_dataset_to_train_valid(dataset, train_ratio=0.9):
    valid_ratio = 1 - train_ratio
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_ratio, valid_ratio])
    return train_dataset, valid_dataset 


def get_data_loader(args):
    if args.dataset == 'dvs128_gesture':
        dataset_type = dvs128_gesture.DVS128Gesture
        H, W = dataset_type.get_H_W()
        seq_len = args.seq_len
        transform = partial(transform_event_list, H=H, W=W, seq_len=seq_len)
        train_dataset = dataset_type(root=args.root, train=True, data_type='event', transform=transform)
        valid_dataset = dataset_type(root=args.root, train=False, data_type='event', transform=transform)
    elif args.dataset == 'n_mnist':
        dataset_type = n_mnist.NMNIST
        H, W = dataset_type.get_H_W()
        seq_len = args.seq_len
        transform = partial(transform_event_list, H=H, W=W, seq_len=seq_len)
        dataset = dataset_type(root=args.root, data_type='event', transform=transform)
        train_dataset, valid_dataset = split_dataset_to_train_valid(dataset, train_ratio=0.9)
    elif args.dataset == 'n_caltech101':
        dataset_type = n_caltech101.NCaltech101
        H, W = dataset_type.get_H_W()
        seq_len = args.seq_len
        transform = partial(transform_event_list, H=H, W=W, seq_len=seq_len)
        dataset = dataset_type(root=args.root, data_type='event', transform=transform)
        train_dataset, valid_dataset = split_dataset_to_train_valid(dataset, train_ratio=0.9)
    else:
        raise NotImplementedError(args.dataset)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=pad_sequence_collator)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=pad_sequence_collator)

    return train_loader, val_loader