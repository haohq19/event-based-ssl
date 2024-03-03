# random initialize the model and save weights to the output directory
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
    parser = argparse.ArgumentParser(description='random weights')
    # model
    parser.add_argument('--d_model', default=512, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--ctx_len', default=4096, type=int, help='context length')
    # run
    parser.add_argument('--device_id', default=7, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--output_dir', default='outputs/random/', help='path where to save')
    parser.add_argument('--test', help='the test mode', action='store_true')
    return parser.parse_args()


def load_model(args):

    model = CausalEventModel(d_event=4, d_model=args.d_model, num_layers=args.num_layers)
    return model


def get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'dmodel{args.d_model}_nlayers{args.num_layers}_T{args.ctx_len}')

    if args.test:
        output_dir += '_test'

    return output_dir

def save(
    model: nn.Module,
    output_dir: str,
    args: argparse.Namespace,
):  

    checkpoint = {
        'model': model.state_dict(),
        'args': args,
    }
    save_name = 'checkpoint/checkpoint.pth'
    torch.save(checkpoint, os.path.join(output_dir, save_name))
    print('saved checkpoint to {}'.format(output_dir))


def main(args):
    print(args)
    torch.cuda.set_device(args.device_id)

    # resume
    output_dir = get_output_dir(args)

    # model
    model = load_model(args)
    model = model.cuda()

    # output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))
   
    save(model=model, output_dir=output_dir, args=args)
    

if __name__ == '__main__':
    args = parser_args()
    main(args)

