import os
import argparse
import random
import hashlib
import yaml
import numpy as np
import torch
import torch.nn as nn
from models.causal_event_model.causal_event_model_for_discrete_event_vae_encoding import CausalEventModelForDiscreteEventVAEEncoding
from models.linear_probe.linear_probe import LinearProbe
from utils.data import get_data_loader
from utils.distributed import init_ddp
from engines.transfer import cache_representations, get_data_loader_from_cached_representations, transfer

_seed_ = 2024
random.seed(2024)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='transfer discrete event VAE encoder')
    # data
    parser.add_argument('--dataset', default='n_mnist', type=str, help='dataset')
    parser.add_argument('--root', default='datasets/NMNIST', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--nclasses', default=10, type=int, help='number of classes')
    # model
    parser.add_argument('--d_model', default=128, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--seq_len', default=2048, type=int, help='context length')
    parser.add_argument('--nevents_per_token', default=64, type=int, help='number of events per token')
    parser.add_argument('--dim_embedding', default=256, type=int, help='dimension of embedding')
    parser.add_argument('--vocab_size', default=1024, type=int, help='vocabulary size')
    parser.add_argument('--pretrained', default='/home/haohq/event-based-ssl/outputs/pretrain_causal_event_model_for_discrete_event_vae_encoding/n_mnist_lr0.001_dmd128_nly4_T2048_ntk64_dem256_vsz1024/checkpoint/checkpoint_epoch200.pth', type=str, help='path to pre-trained weights')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--output_dir', default='outputs/transfer_causal_event_model_for_discrete_event_vae_encoding', help='dir to save')
    parser.add_argument('--test', help='the test mode', action='store_true')
    return parser.parse_args()


def load_pretrained_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError('pretrained weights not found at {}'.format(pretrained_path))
    else:
        checkpoint = torch.load(pretrained_path)
        if 'model' in checkpoint.keys():
            pretrained_weights = checkpoint['model']
        else:
            pretrained_weights = checkpoint
    if pretrained_weights is not None:
        model.load_state_dict(pretrained_weights, strict=False)
        print('load pretrained weights from {}'.format(pretrained_path))
    else:
        raise ValueError('pretrained weights is None')
    return model


def get_output_dir(args):
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, f'lr{args.lr}_wd{args.weight_decay}_dmd{args.d_model}_nly{args.num_layers}_T{args.seq_len}')
    output_dir += f'_ntk{args.nevents_per_token}_dem{args.dim_embedding}_vsz{args.vocab_size}'
    sha256_hash = hashlib.sha256(args.pretrained.encode()).hexdigest()
    output_dir += '_pt'
    output_dir += f'{sha256_hash[:8]}'
    if args.test:
        output_dir += '_test'
    return output_dir


def main(args):
    init_ddp(args)
    torch.cuda.set_device(args.device_id)
    
    # output
    output_dir = get_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))

    # model
    model = CausalEventModelForDiscreteEventVAEEncoding(
        d_event=4,
        d_model=args.d_model, 
        num_layers=args.num_layers, 
        dim_feedforward=4*args.d_model, 
        nevents_per_token=args.nevents_per_token,
        dim_embedding=args.dim_embedding,
        vocab_size=args.vocab_size,
    )
    if args.pretrained:
        model = load_pretrained_weights(model, args.pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    
    # cache
    sha256_hash = hashlib.sha256(args.pretrained.encode()).hexdigest()
    cache_dir = os.path.join(args.output_dir, args.dataset, 'cache' + f'_{sha256_hash[:8]}')  # output_dir/dataset_name/cache_{hash}/
    args.cache_dir = cache_dir
    train_loader, valid_loader = get_data_loader(args)
    cache_representations(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        cache_dir=cache_dir,
    )
    train_loader, valid_loader = get_data_loader_from_cached_representations(args)
    
    # linear_probe
    model = LinearProbe(args.d_model * args.num_layers, args.nclasses)
    model.cuda()

    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # print and save args
    print(args)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    transfer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    args = parser_args()
    main(args)