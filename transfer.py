# transfer causal event pretrained representations to downstream tasks
# haohq19@gmail.com

import os
import argparse
import random
import numpy as np
import torch
import tqdm
import torch.nn as nn
import hashlib
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from spikingjelly.datasets import dvs128_gesture
from models.causal_event_model.model import CausalEventModel
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
    parser = argparse.ArgumentParser(description='transfer causal evnent pretrained representations to downstream tasks')
    # data
    parser.add_argument('--dataset', default='dvs128_gesture', type=str, help='dataset')
    parser.add_argument('--root', default='datasets/DVS128Gesture', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--nclasses', default=11, type=int, help='number of classes')
    # model
    parser.add_argument('--d_model', default=512, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--ctx_len', default=4096, type=int, help='context length')
    parser.add_argument('--pretrained_path', default='outputs/random/dmodel512_nlayers4_T4096_test/checkpoint/checkpoint.pth', type=str, help='path to pre-trained weights')
    # run
    parser.add_argument('--device_id', default=7, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--output_dir', default='outputs/transfer/', help='path where to save')
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
    # load pretrained weights
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(args.pretrained_path)
    else:
        checkpoint = torch.load(args.pretrained_path)
        if 'model' in checkpoint.keys():
            pretrained_weights = checkpoint['model']
        else:
            pretrained_weights = checkpoint

    # model
    model = CausalEventModel(d_event=4, d_model=args.d_model, num_layers=args.num_layers)
    if pretrained_weights is not None:
        model.load_state_dict(pretrained_weights)
        print('load pretrained weights from {}'.format(args.pretrained_path))
    else:
        raise NotImplementedError(args.model)
    
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_output_dir(args):
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, f'lr{args.lr}_wd{args.weight_decay}_dmodel{args.d_model}_nlayers{args.num_layers}_T{args.ctx_len}')

    sha256_hash = hashlib.sha256(args.pretrained_path.encode()).hexdigest()
    output_dir += '_pt'
    output_dir += f'_{sha256_hash[:8]}'

    if args.test:
        output_dir += '_test'

    return output_dir


class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features):
        # feature.shape = [batch_size, in_features]
        output = self.fc(features)
        return output # output.shape = [batch_size, num_classes]


def cache_representations(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cache_dir: str,
):  
    if os.path.exists(cache_dir):
        print('cached feature map already exists')
        train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'train_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'train_labels.npy'))))
        val_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_labels.npy'))))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        return train_loader, valid_loader
    else:
        os.makedirs(cache_dir)
    
    with torch.no_grad():
        model.eval()
        # train_loader
        features = []
        labels = []
        nsteps_per_epoch = len(train_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in train_loader:
            input = data[:, :args.ctx_len, :] # select first ctx_len events]
            start = input[:, 0:1, 0]  # start time
            input[:, :, 0] = input[:, :, 0] - start  # transform to relative time
            input = input.cuda().float()
            model(input)
            feature_map = model.hidden.cpu().numpy()  # N, D
            features.append(feature_map)
            labels.append(label)
            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=0)  # N, D
        labels = np.concatenate(labels, axis=0)  # N
        np.save(os.path.join(cache_dir, 'train_features.npy'), features)
        np.save(os.path.join(cache_dir, 'train_labels.npy'), labels)

        # val_loader
        features = []
        labels = []
        nsteps_per_epoch = len(valid_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in valid_loader:
            input = data[:, :args.ctx_len, :] # select first ctx_len events]
            start = input[:, 0:1, 0]  # start time
            input[:, :, 0] = input[:, :, 0] - start  # transform to relative time
            input = input.cuda().float()
            model(input)
            feature_map = model.hidden.cpu().numpy()  # N, D
            features.append(feature_map)
            labels.append(label)
            process_bar.update(1)
        process_bar.close()
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'valid_features.npy'), features)
        np.save(os.path.join(cache_dir, 'valid_labels.npy'), labels)

        print('cached feature map saved to {}'.format(cache_dir))
    
    train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'train_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'train_labels.npy'))))
    valid_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'valid_labels.npy'))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    return train_loader, valid_loader

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
):  
    
    tb_writer = SummaryWriter(output_dir + '/log')
    print('log saved to {}'.format(output_dir + '/log'))

    # train 
    epoch = epoch
    best_model = None
    best_acc = 0
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        top1_correct = 0
        top5_correct = 0
        total = len(train_loader.dataset)
        total_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        import tqdm
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in train_loader:
            input = data.cuda(non_blocking=True)
            target = label.cuda(non_blocking=True)  # N
            output = model(input)  # N, C
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item() * input.shape[1]
            step += 1
            tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
            process_bar.update(1)
        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        total_loss = total_loss / total 
        tb_writer.add_scalar('train_acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('train_acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('train_loss', total_loss, epoch + 1)
        process_bar.close()
        print('train_cor@1: {}, train_cor@5: {}, train_total: {}'.format(top1_correct, top5_correct, total))
        print('train_acc@1: {}, train_acc@5: {}, train_loss: {}'.format(top1_accuracy, top5_accuracy, total_loss))
        
        # evaluate
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = len(valid_loader.dataset)
        total_loss = 0
        with torch.no_grad():
            for data, label in valid_loader:
                data = input.cuda(non_blocking=True)
                target = label.cuda(non_blocking=True)
                output = model(input)
                loss = criterion(output, target)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                total_loss += loss.item() * input.shape[1]
        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        total_loss = total_loss / total
        tb_writer.add_scalar('val_acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('val_acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('val_loss', total_loss, epoch + 1)
        print('val_cor@1: {}, val_cor@5: {}, val_total: {}'.format(top1_correct, top5_correct, total))
        print('val_acc@1: {}, val_acc@5: {}, val_loss: {}'.format(top1_accuracy, top5_accuracy, total_loss))

        epoch += 1

        # save best
        if top1_accuracy >= best_acc:
            best_acc = top1_accuracy
            best_model = model
            checkpoint = {
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            save_name = 'checkpoint/best_{}.pth'.format(top1_accuracy)
            torch.save(checkpoint, os.path.join(output_dir, save_name))
            print('saved best model to {}'.format(output_dir))
    print('best_val_acc@1: {}'.format(best_acc))
    return best_model


def main(args):
    print(args)
    torch.cuda.set_device(args.device_id)

    # criterion
    criterion = nn.CrossEntropyLoss()
    
    # output
    output_dir = get_output_dir(args)  # outputs/transfer/dataset_name/lr_wd_dmodel_nl_T_pt/
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))

    # model
    model = load_model(args)
    model.cuda()
    
    # data
    sha256_hash = hashlib.sha256(args.pretrained_path.encode()).hexdigest()
    cache_dir = os.path.join(args.output_dir, args.dataset, 'cache' + f'_{sha256_hash[:8]}')  # outputs/transfer/dataset_name/cache_{hash}/
    train_loader, valid_loader = None, None
    if not os.path.exists(cache_dir):
        train_loader, valid_loader = load_data(args)
    train_loader, valid_loader = cache_representations(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        cache_dir=cache_dir,
    )

    # linear_probe
    linear_probe = LinearProbe(args.d_model, args.nclasses)
    linear_probe.cuda()

    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, linear_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    train(
        model=linear_probe,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        args=args,
    )

if __name__ == '__main__':
    args = parser_args()
    main(args)