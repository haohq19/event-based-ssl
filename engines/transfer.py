import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def cache_representations(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cache_dir: str,
):  
    if os.path.exists(cache_dir):
        print('cached feature map already exists')
    else:
        os.makedirs(cache_dir)
    
    with torch.no_grad():
        model.eval()
        
        # train
        features = []
        labels = []
        nsteps_per_epoch = len(train_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in train_loader:
            input = data.cuda(non_blocking=True)
            output, hidden = model(input, return_hidden=True)
            feature_map = hidden.detach().cpu().numpy()        # feature_map.shape = [batch, dim_hidden]
            features.append(feature_map)
            labels.append(label)                               # labels.shape = [batch]
            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=0)            # features.shape = [N, dim_hidden]
        labels = np.concatenate(labels, axis=0)                # labels.shape = [N]
        np.save(os.path.join(cache_dir, 'train_features.npy'), features)
        np.save(os.path.join(cache_dir, 'train_labels.npy'), labels)

        # valid
        features = []
        labels = []
        nsteps_per_epoch = len(valid_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in valid_loader:
            input = data.cuda(non_blocking=True)
            output, hidden = model(input, return_hidden=True)
            feature_map = hidden.detach().cpu().numpy()       # feature_map.shape = [batch, dim_hidden]
            features.append(feature_map)
            labels.append(label)                              # labels.shape = [batch]
            process_bar.update(1)
        process_bar.close()
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'valid_features.npy'), features)
        np.save(os.path.join(cache_dir, 'valid_labels.npy'), labels)

        print('cached feature map saved to {}'.format(cache_dir))


def get_data_loader_from_cached_representations(
        cache_dir,
        batch_size,
        num_workers,
    ):
    train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'train_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'train_labels.npy'))))
    valid_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_labels.npy'))))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, valid_loader


def transfer(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
):  
    
    tb_writer = SummaryWriter(output_dir + '/log')
    print('log saved to {}'.format(output_dir + '/log'))

    epoch = epoch
    best_model = None
    best_valid_acc = 0
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))

        # train
        model.train()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in train_loader:
            input = data.cuda(non_blocking=True)
            target = label.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

            process_bar.set_description('loss: {:.3f}'.format(loss.item()))
            process_bar.update(1)
            tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
            epoch_loss += loss.item() * data.size(0)
            step += 1
        
        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch

        tb_writer.add_scalar('train/acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('train/acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('train/loss', epoch_loss / nsamples_per_epoch, epoch + 1)
        process_bar.close()
        print('train_cor@1: {}, train_cor@5: {}, train_total: {}'.format(top1_correct, top5_correct, nsamples_per_epoch))
        print('train_acc@1: {}, train_acc@5: {}, train_avg_loss: {}'.format(top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch))
        
        # valid
        model.eval()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(valid_loader.dataset)
        epoch_loss = 0
        with torch.no_grad():
            for data, label in valid_loader:
                input = data.cuda(non_blocking=True)
                target = label.cuda(non_blocking=True)
                output = model(input)
                loss = criterion(output, target)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()

                epoch_loss += loss.item() * data.size(0)

        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch

        tb_writer.add_scalar('valid/acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('valid/acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('valid/loss', epoch_loss / nsamples_per_epoch, epoch + 1)
        print('valid_cor@1: {}, valid_cor@5: {}, valid_total: {}'.format(top1_correct, top5_correct, nsamples_per_epoch))
        print('valid_acc@1: {}, valid_acc@5: {}, valid_loss: {}'.format(top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch))

        epoch += 1

        # save best
        if top1_accuracy >= best_valid_acc:
            best_valid_acc = top1_accuracy
            best_model = model
            checkpoint = {
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoint/best_{}.pth'.format(top1_accuracy)
            torch.save(checkpoint, os.path.join(output_dir, save_name))
            print('saved best model to {}'.format(output_dir))
    print('best_valid_acc@1: {}'.format(best_valid_acc))
    return best_model