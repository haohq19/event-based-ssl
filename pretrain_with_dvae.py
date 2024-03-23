# pretrain model with causal event predict
# haohq19@gmail.com

import os
import argparse
import random
import glob
import numpy as np
import torch
import tqdm
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.causal_event_model.causal_event_model import CausalEventModel
from models.transformer_decoder.model import TransformerDecoder
from models.discrete_event_vae.discrete_event_vae import dVAEOutput
from utils.data import get_data_loader
from utils.distributed import init_ddp, is_master, global_meters_all_sum, save_on_master

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
    parser.add_argument('--dvae_root', default='/home/haohq/test/outputs/discrete_event_vae/n_mnist_lr0.001_T64_dem256_ntk1024_dlt32_dhd32_nep50_stp1_gma0.99_klw1_0.025_tmp4_0.0625_0.2_exp_bce/checkpoints/checkpoint_50.pth', type=str, help='path to pretrained dvae model')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain_with_dvae/', help='path where to save')
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

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    save_freq: int,
    args: argparse.Namespace,
):  
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
        print('log saved to {}'.format(output_dir + '/log'))
    
    # train
    epoch = epoch
    while(epoch < nepochs):
        print('epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        if is_master():
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, _ in train_loader:
            # data.shape = [batch, seq_len, d_event]
            input = data.cuda(non_blocking=True)
            loss = model(input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if is_master():
                process_bar.set_description('loss: {:.3f}'.format(loss.item()))
            epoch_loss += loss.item() * data.size(0)
            step += 1
            if is_master():
                tb_writer.add_scalar(tag='step/loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
                process_bar.update(1)
        if args.distributed:
            epoch_loss = global_meters_all_sum(args, epoch_loss)
        if is_master():
            tb_writer.add_scalar('train/loss', epoch_loss/nsamples_per_epoch, epoch + 1)
            process_bar.close()
        print('train_avg_loss: {:.3f}'.format(epoch_loss/nsamples_per_epoch))
        
        # validate
        model.eval()
        nsamples_per_epoch = len(val_loader.dataset)
        epoch_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                input = data.cuda(non_blocking=True)
                loss = model(input)
                epoch_loss += loss.item() * data.size(0)
        if args.distributed:
            epoch_loss = global_meters_all_sum(args, epoch_loss)
        if is_master():
            tb_writer.add_scalar('valid/loss', epoch_loss/nsamples_per_epoch, epoch + 1)
        print('valid_avg_loss: {:.3f}'.format(epoch_loss/nsamples_per_epoch))

        epoch += 1

        # save
        if epoch % save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoint/checkpoint_epoch{}.pth'.format(epoch)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('saved checkpoint to {}'.format(output_dir))


class CausalEventModelWithdVAEOutput(nn.Module):
    def __init__(self, d_event, d_model, num_layers, dim_feedforward, nevents_per_token, dim_embedding, vocab_size, dvae_root):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.nevents_per_token = nevents_per_token
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.dvae_root = dvae_root

        self.decoder = CausalEventModel(
            d_event=d_event,
            d_model=d_model,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            d_out=vocab_size,
        )

        self.output = dVAEOutput(
            nevents_per_token=nevents_per_token,
            dim_embedding=dim_embedding,
            vocab_size=vocab_size,
        )

        checkpoint = torch.load(dvae_root)
        pretrained_weight = checkpoint['model']
        self.output.dvae.load_state_dict(pretrained_weight, strict=False)
        for p in self.output.dvae.parameters():
            p.requires_grad = False

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        output = self.decoder(x)
        loss = self.output(output, x)
        return loss


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
    model = CausalEventModelWithdVAEOutput(
        d_event=4,
        d_model=args.d_model, 
        num_layers=args.num_layers, 
        dim_feedforward=4*args.d_model, 
        nevents_per_token=args.nevents_per_token,
        dim_embedding=args.dim_embedding,
        vocab_size=args.vocab_size,
        dvae_root=args.dvae_root,
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

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        save_freq=args.save_freq,
        args=args,
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)

'''
python -m torchrun --nproc_per_node=8 pretrain.py
'''