# pretrain model with causal event predict
# haohq19@gmail.com

import os
import argparse
import random
import glob
import numpy as np
import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.causal_event_model.model import CausalEventModel
from models.loss.product_loss import ProductLoss
from models.loss.dual_head_loss import DualHeadL2Loss, DualHeadL1Loss
from utils.data import get_data_loader

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
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    # model
    parser.add_argument('--d_model', default=512, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--seq_len', default=1024, type=int, help='context length')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain/', help='path where to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--resume', help='resume from checkpoint', action='store_true')
    parser.add_argument('--test', help='the test mode', action='store_true')
    return parser.parse_args()



def get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_lr{args.lr}_dmodel{args.d_model}_nlayers{args.num_layers}_T{args.seq_len}')
    
    if args.criterion == 'ProductLoss':
        output_dir += '_PL'
    elif args.criterion == 'DualHeadLoss':
        output_dir += '_DHL2'
    elif args.criterion == 'MSELoss':
        output_dir += '_MSE'
    elif args.criterion == 'DualHeadL1Loss':
        output_dir += '_DHL1'
    else:
        raise NotImplementedError(args.criterion)

    if args.test:
        output_dir += '_test'

    return output_dir

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
):  
    tb_writer = SummaryWriter(output_dir + '/log')
    print('log saved to {}'.format(output_dir + '/log'))

    torch.cuda.empty_cache()
    
    # train 
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        total = len(train_loader.dataset)
        total_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, _ in train_loader:
            input = data[:, :args.seq_len, :] # select first seq_len events]
            target = data[:, 1:args.seq_len+1, 1:]  # auto-regressive and ignore t
            # to cuda
            input = input.cuda()
            target = target.cuda()
            output = model(input)  # output.shape = [batch, seq_len, d_event]
            
            loss = criterion(output, target)
            
            # pseudo_loss
            # pseudo_output = input[:, :, 1:3].repeat(1, 1, 2)  # pseudo_output.shape = [batch, seq_len, 2]
            # pseudo_loss = criterion(pseudo_output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            process_bar.set_description('loss: {:.3f}'.format(loss.item()))
            total_loss += loss.item() * data.size(0)
            step += 1
            tb_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
            # tb_writer.add_scalar(tag='pseudo_loss', scalar_value=pseudo_loss.item(), global_step=epoch * nsteps_per_epoch + step)
            process_bar.update(1)
         
        total_loss /= total
        tb_writer.add_scalar('train_loss', total_loss, epoch + 1)
        process_bar.close()
        print('train_loss: {:.3f}'.format(total_loss))
        
        # validate
        model.eval()
        total = len(val_loader.dataset)
        total_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                input = data[:, :args.seq_len, :] # select first seq_len events]
                target = data[:, 1:args.seq_len+1, 1:]  # auto-regressive and ignore t
                # to cuda
                input = input.cuda()
                target = target.cuda()
                output = model(input)  # output.shape = [batch, seq_len, d_event]

                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
    
        total_loss = total_loss / total
        tb_writer.add_scalar('valid_loss', total_loss, epoch + 1)
        print('valid_loss: {:.3f}'.format(total_loss))

        epoch += 1

        # save
        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            save_name = 'checkpoint/checkpoint_epoch{}.pth'.format(epoch)
            torch.save(checkpoint, os.path.join(output_dir, save_name))
            print('saved checkpoint to {}'.format(output_dir))


def main(args):

    # criterion
    criterion = DualHeadL1Loss()
    args.criterion = criterion.__class__.__name__
    print(args)

    # device
    torch.cuda.set_device(args.device_id)

     # data
    train_loader, val_loader = get_data_loader(args)

    # output_dir
    output_dir = get_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))

    # --resume
    state_dict = None
    if args.resume:
        checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint/*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            state_dict = torch.load(latest_checkpoint)
            print('load checkpoint from {}'.format(latest_checkpoint))

    # model
    model = CausalEventModel(d_event=4, d_model=args.d_model, num_layers=args.num_layers)
    if state_dict:
        model.load_state_dict(state_dict['model'])
    model.cuda()
    
    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
        epoch = state_dict['epoch']
   

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        args=args
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)

