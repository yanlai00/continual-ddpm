from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from eps_models.unet import UNet
import wandb
from pathlib import Path
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10
from trainers.trainer import Trainer
from trainers.conditional_trainer import ClassConditionedTrainer
from eps_models.mlp import Mlp
import numpy as np
import argparse

def main(args):
    if args.wandb:
        wandb.init(project="ddpm", entity="yanlaiy", name=args.wandb_name)

    trainer = ClassConditionedTrainer()
    trainer.batch_size = 64
    trainer.epochs = 675
    trainer.image_channels = 3
    trainer.n_samples = 32
    trainer.num_classes = 10
    trainer.dataset = CIFAR10(trainer.image_size)
    trainer.wandb = args.wandb
    trainer.wandb_name = args.wandb_name

    # Initialize, start and run the training loop
    trainer.init()
    trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12112022_135829/checkpoint_325.pt'))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise Diffusion Probabilitic Models')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    args = parser.parse_args()
    main(args)
