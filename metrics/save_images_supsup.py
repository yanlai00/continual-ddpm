import torch
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10
from trainers.trainer import Trainer
from trainers.conditional_trainer import ClassConditionedTrainer
from trainers.supsup_trainer import SupSupTrainer
import argparse
from torchvision.utils import save_image
import os

# trainer = ClassConditionedTrainer()
trainer = SupSupTrainer()
trainer.batch_size = 500
trainer.image_channels = 1
trainer.n_samples = 128
trainer.datasets = [SplitMNIST(trainer.image_size, target=t) for t in range(10)]
num_classes = 10

save_dir = '/home/yy2694/continual-ddpm/images/mnist_cond_supsup'
os.mkdir(save_dir)
for class_idx in range(num_classes):
    os.mkdir(f'{save_dir}/{class_idx}')

trainer.init()
trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12142022_020625/checkpoint_599.pt'))
num_iterations = 1

step = 0
for i in range(num_iterations):
    for class_idx in range(num_classes):
        trainer.eps_model.apply(lambda m: setattr(m, "task", class_idx)) # for SupSup
        images = trainer.sample(class_idx, trainer.batch_size)
        for t in range(trainer.batch_size):
            save_image(images[t], f'{save_dir}/{class_idx}/img{i*trainer.batch_size+t}.png')

