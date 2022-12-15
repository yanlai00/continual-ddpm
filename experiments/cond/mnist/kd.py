from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from eps_models.unet import UNet
import wandb
from pathlib import Path
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D
from trainers.trainer import Trainer
from trainers.conditional_distillation import ConditionalDistillationTrainer
from eps_models.mlp import Mlp
import numpy as np
import argparse

def main(args):
    if args.wandb:
        wandb.init(project="ddpm", entity="yanlaiy", name=args.wandb_name)

    trainer = ConditionalDistillationTrainer()
    trainer.image_channels = 1
    trainer.batch_size = 128
    trainer.epochs = 100
    trainer.n_samples = 128
    trainer.datasets = [SplitMNIST(trainer.image_size, target=t) for t in range(10)]
    trainer.wandb = args.wandb
    trainer.wandb_name = args.wandb_name

    # Initialize, start and run the training loop
    trainer.init()
    trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12122022_204352mnist_cond_replay/checkpoint_100.pt'))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise Diffusion Probabilitic Models')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    args = parser.parse_args()
    main(args)
