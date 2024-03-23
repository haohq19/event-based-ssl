import os
import argparse
import random
import glob
import numpy as np
import torch
import yaml
from models.causal_event_model.causal_event_model_for_discrete_event_vae_encoding import CausalEventModelForDiscreteEventVAEEncoding
from utils.data import get_data_loader
from utils.distributed import init_ddp, is_master
from engines.pretrain import pretrain

_seed_ = 2024
random.seed(2024)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='causal event pretraining')
    # data
    parser.add_argument('--dataset', default='n_mnist', type=str, help='dataset')
    parser.add_argument('--root', default='datasets/NMNIST', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    # model
    parser.add_argument('--d_model', default=128, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--seq_len', default=2048, type=int, help='sequence length')
    parser.add_argument('--dim_embedding', default=256, type=int, help='dimension of embedding')
    parser.add_argument('--nevents_per_token', default=64, type=int, help='number of events per token')
    parser.add_argument('--vocab_size', default=1024, type=int, help='vocabulary size')
    parser.add_argument('--pretrained', default='/home/haohq/test/outputs/discrete_event_vae/n_mnist_lr0.001_T64_dem256_ntk1024_dlt32_dhd32_nep50_stp1_gma0.99_klw1_0.025_tmp4_0.0625_0.2_exp_bce/checkpoints/checkpoint_50.pth', type=str, help='path to pretrained dvae model')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain_causal_event_model_for_discrete_event_vae_encoding', help='dir to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--resume', help='resume from latest checkpoint', action='store_true')
    parser.add_argument('--test', help='debug mode', action='store_true')
    # distributed
    parser.add_argument('--world-size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--backend', default='gloo', help='distributed backend')
    return parser.parse_args()


def get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_lr{args.lr}_dmd{args.d_model}_nly{args.num_layers}_T{args.seq_len}')
    output_dir += '_ntk' + str(args.nevents_per_token) + '_dem' + str(args.dim_embedding) + '_vsz' + str(args.vocab_size)
    if args.test:
        output_dir += '_test'
    return output_dir


def main(args):
    # init distributed data parallel
    init_ddp(args)
    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

    # data
    train_loader, val_loader = get_data_loader(args)

    # output_dir
    output_dir = get_output_dir(args)
    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
            os.makedirs(os.path.join(output_dir, 'checkpoint'))

    # resume
    state_dict = None
    if args.resume:
        checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint/*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)  # get the latest checkpoint
            state_dict = torch.load(latest_checkpoint)
            print('load checkpoint from {}'.format(latest_checkpoint))

    # model
    model = CausalEventModelForDiscreteEventVAEEncoding(
        d_event=4,
        d_model=args.d_model, 
        num_layers=args.num_layers, 
        dim_feedforward=4*args.d_model, 
        nevents_per_token=args.nevents_per_token,
        dim_embedding=args.dim_embedding,
        vocab_size=args.vocab_size,
        pretrained_discrete_event_vae_root=args.pretrained,
    )
    print('model size: {:.2f}M'.format(model.num_params / 1e6))
    if state_dict:
        model.load_state_dict(state_dict['model'])
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
   
    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
        epoch = state_dict['epoch']
   
    # print and save args
    print(args)
    if is_master():
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    # pretrain
    pretrain(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        save_freq=args.save_freq,
        distributed=args.distributed,
        local_rank=args.local_rank if args.distributed else None,
    )

if __name__ == '__main__':
    args = parser_args()
    main(args)

'''
python pretrain_casual_event_model_for_discrete_event_vae_encoding.py
python -m torchrun --nproc_per_node=8 pretrain_casual_event_model_for_discrete_event_vae_encoding.py
'''