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
from eps_models.mlp import Mlp
import numpy as np
import argparse

def main(args):
    if args.wandb:
        wandb.init(project="ddpm", entity="yanlaiy", name=args.wandb_name)

    trainer = Trainer()
    # trainer = ContinualTrainer()

    # Override config defaults
    # TODO add a more general method for this
    trainer.batch_size = 128
    trainer.epochs = 100
    trainer.image_channels = 1
    trainer.n_samples = 128
    # trainer.datasets = [SplitMNIST(trainer.image_size, target=t) for t in range(1, 2)]
    # trainer.dataset = Gaussian2D(mean=np.array([0., 0.]), cov=np.array([[1., 0.], [0., 1.]]), num_samples=10000)
    trainer.dataset = MNIST(trainer.image_size)
    # trainer.dataset = CIFAR10(trainer.image_size)
    trainer.wandb = args.wandb
    trainer.wandb_name = args.wandb_name

    # Initialize, start and run the training loop
    trainer.init()
    # trainer.eps_model = Mlp(1, 1, [100, 100])
    # trainer.eps_model.load_state_dict(torch.load('/home/yy2694/ddpm/experiments/10282022_152633/checkpoint_29.pt'))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise Diffusion Probabilitic Models')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    args = parser.parse_args()
    main(args)

