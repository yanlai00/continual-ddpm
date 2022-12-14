from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from eps_models.unet import UNet
import wandb
from pathlib import Path
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10, SplitCIFAR10
from trainers.trainer import Trainer
from eps_models.mlp import Mlp
import numpy as np
import argparse

def main(args):
    if args.wandb:
        wandb.init(project="ddpm", entity="yanlaiy", name=args.wandb_name)

    trainer = Trainer()

    # Override config defaults
    # TODO add a more general method for this
    trainer.batch_size = 128
    trainer.epochs = 1000
    trainer.image_channels = 3 # 1
    trainer.n_samples = 64
    trainer.dataset = SplitCIFAR10(trainer.image_size, target=args.target)
    trainer.wandb = args.wandb
    trainer.wandb_name = args.wandb_name

    # Initialize, start and run the training loop
    trainer.init()
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise Diffusion Probabilitic Models')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    parser.add_argument('--target', type=int, help='class number')
    args = parser.parse_args()
    main(args)

