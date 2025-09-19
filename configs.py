import argparse
import os
from typing import Dict

import torch

class Config():
    
    def __init__(self, params: argparse.Namespace):

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        ### ---------- experiment configs ---------- ###
        self.data_dir = params.data_dir
        self.save_dir = params.save_dir
        self.load_iter = params.load_iter
        self.load_dir = params.load_dir
        self.mode = params.mode

        ### ---------- quantization configs ---------- ###
        self.quantize = params.quantize
        self.quantize_bits = params.quantize_bits
        self.save_bits = params.save_bits
        self.noise_std = params.noise_std

        # check quantize bits and save bits       
        if self.quantize_bits >= self.save_bits:
            raise ValueError("The save bits should be larger than the quantize bits.")

        ### ---------- trainer configs ---------- ###
        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.weight_decay = params.weight_decay
        self.lr_scheduler = params.lr_scheduler

        self.eval_interval = params.eval_interval
        self.save_interval = params.save_interval

        ### ---------- model configs ---------- ###
        self.num_channels = 0
        self.num_lods = 0
        self.n_frequencies = 8
        self.n_levels = 7
        self.n_features_per_level = 8
        self.base_resolution = 4
        self.n_neurons = 16
        self.n_hidden_layers = 1

def get_args():
    parser = argparse.ArgumentParser()

    ### ---------- experiment configs ---------- ###
    parser.add_argument('--data_dir', type=str, default='data',
                        help='root directory of dataset')
    parser.add_argument('--save_dir', type=str,
                        help='directory of dataset')
    parser.add_argument('--load_iter', type=int, default=0,
                        help='0 -> do not load model, -1 means the newest')
    parser.add_argument('--load_dir', type=str,
                        help='')
    parser.add_argument('--mode', type=str, default="train",
                        help='',
                        choices=['train', 'infer'])
    
    ### ---------- quantization configs ---------- ###
    parser.add_argument('--quantize', type=bool, default=False,
                        help='whether to quantize the model or not')
    parser.add_argument('--quantize_bits', type=int, default=4,
                        help='choose the bits to quantize',
                        choices=[2, 4, 8, 16])
    parser.add_argument('--save_bits', type=int, default=16,
                        help='choose the bits to quantize',
                        choices=[8, 16, 32, 64])
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    ### ---------- trainer configs ---------- ###
    parser.add_argument('--max_iter', type=int, default=400000,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='the interval of iteration for evalation')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='the interval of iteration for saving model')

    return parser.parse_args()