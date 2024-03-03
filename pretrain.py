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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.datasets import dvs128_gesture
from models.causal_event_model.model import CausalEventModel
from models.loss.product_loss import ProductLoss
from models.loss.dual_head_loss import DualHeadLoss
from utils import pad_sequence_collator as collate_fn
from utils import event_to_tensor as transform

_seed_ = 2024
random.seed(2024)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='event-based causal event pretrain')
    # data
    parser.add_argument('--dataset', default='dvs128_gesture', type=str, help='dataset')
    parser.add_argument('--root', default='datasets/DVS128Gesture', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    # model
    parser.add_argument('--d_model', default=512, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--ctx_len', default=4096, type=int, help='context length')
    # run
    parser.add_argument('--device_id', default=7, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain/', help='path where to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--resume', help='resume from checkpoint', action='store_true')
    parser.add_argument('--test', help='the test mode', action='store_true')
    return parser.parse_args()



def load_data(args):
    if args.dataset == 'dvs128_gesture':
        train_dataset = dvs128_gesture.DVS128Gesture(root=args.root, train=True, data_type='event', transform=transform)
        val_dataset = dvs128_gesture.DVS128Gesture(root=args.root, train=False, data_type='event', transform=transform)
    else:
        raise NotImplementedError(args.dataset)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return train_loader, val_loader

def load_model(args):
    model = CausalEventModel(d_event=4, d_model=args.d_model, num_layers=args.num_layers)
    return model


def get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_lr{args.lr}_dmodel{args.d_model}_nlayers{args.num_layers}_T{args.ctx_len}')
    
    if args.criterion == 'ProductLoss':
        output_dir += '_PL'
    elif args.criterion == 'DualHeadLoss':
        output_dir += '_DHL'
    elif args.criterion == 'MSELoss':
        output_dir += '_MSE'
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
            input = data[:, :args.ctx_len, :] # select first ctx_len events]
            target = data[:, 1:args.ctx_len+1, 1:]  # auto-regressive and ignore t
            # transform to relative time
            start = input[:, 0:1, 0]  # start time
            input[:, :, 0] = input[:, :, 0] - start
            # to cuda
            input = input.cuda().float()
            target = target.cuda().float()
            output = model(input)  # output.shape = [batch, ctx_len, d_event]
            pseudo_output = input[:, :, 1:3].repeat(1, 1, 2)  # pseudo_output.shape = [batch, ctx_len, 2]
            loss = criterion(output, target)
            pseudo_loss = criterion(pseudo_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            process_bar.set_description('loss: {:.3f}'.format(loss.item()))
            total_loss += loss.item() * data.size(0)
            step += 1
            tb_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
            tb_writer.add_scalar(tag='pseudo_loss', scalar_value=pseudo_loss.item(), global_step=epoch * nsteps_per_epoch + step)
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
                input = data[:, :args.ctx_len, :] # select first ctx_len events]
                target = data[:, 1:args.ctx_len+1, 1:]  # auto-regressive and ignore t
                # transform to relative time
                start = input[:, 0:1, 0]  # start time
                input[:, :, 0] = input[:, :, 0] - start
                # to cuda
                input = input.cuda().float()
                target = target.cuda().float()
                output = model(input)  # output.shape = [batch, ctx_len, d_event]
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

    print(args)
    torch.cuda.set_device(args.device_id)

     # data
    train_loader, val_loader = load_data(args)

    # criterion
    criterion = DualHeadLoss()
    args.criterion = criterion.__class__.__name__
    
    # resume
    output_dir = get_output_dir(args)
    state_dict = None
    if args.resume:
        checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint/*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            state_dict = torch.load(latest_checkpoint)
            print('load checkpoint from {}'.format(latest_checkpoint))

    # model
    model = load_model(args)
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

    # output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))
   

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

