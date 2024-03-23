import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.distributed import is_master, global_meters_all_sum, save_on_master


def pretrain(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    save_freq: int,
    distributed: bool = False,
    local_rank: int = None,
):  
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
    print('log saved to {}'.format(output_dir + '/log'))
    
    epoch = epoch
    while(epoch < nepochs):
        print('epoch {}/{}'.format(epoch+1, nepochs))

        # train
        model.train()
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        if is_master():
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, _ in train_loader:
            input = data.cuda(non_blocking=True)
            loss = model(input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_master():
                process_bar.set_description('loss: {:.3f}'.format(loss.item()))
                process_bar.update(1)
                tb_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
            
            epoch_loss += loss.item() * data.size(0)
            step += 1
        if distributed:
            epoch_loss = global_meters_all_sum(local_rank, epoch_loss)
        epoch_loss /= nsamples_per_epoch
        if is_master():
            process_bar.close()
            tb_writer.add_scalar('train/loss', epoch_loss, epoch + 1)
        print('train_avg_loss: {:.3f}'.format(epoch_loss))
        
        # valid
        model.eval()
        nsamples_per_epoch = len(val_loader.dataset)
        epoch_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                input = data.cuda(non_blocking=True)
                loss = model(input)
                epoch_loss += loss.item() * data.size(0)
        if distributed:
            epoch_loss = global_meters_all_sum(local_rank, epoch_loss)
        epoch_loss = epoch_loss / nsamples_per_epoch
        if is_master():
            tb_writer.add_scalar('valid/loss', epoch_loss, epoch + 1)
        print('valid_avg_loss: {:.3f}'.format(epoch_loss))

        epoch += 1

        # save
        if epoch % save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoint/checkpoint_epoch{}.pth'.format(epoch)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('saved checkpoint to {}'.format(output_dir))
    
