import torch
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10
from trainers.trainer import Trainer
from trainers.ewc import EWCTrainer
import argparse
from torchvision.utils import save_image
import os

trainer = EWCTrainer()
trainer.batch_size = 500
trainer.image_channels = 1
trainer.n_samples = 128
num_classes = 10
trainer.dataset = MNIST(trainer.image_size)
trainer.datasets = [SplitMNIST(trainer.image_size, target=t) for t in range(num_classes)]

save_dir = '/home/yy2694/continual-ddpm/images/mnist_uncond_ewc/'
os.mkdir(save_dir)

trainer.init()
# trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12022022_144833/checkpoint_99.pt'))
trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12142022_214037mnist_uncond_ewc/checkpoint_1000.pt'))

num_iterations = 1

step = 0
for i in range(num_iterations):
    images = trainer.sample(trainer.batch_size)
    for t in range(trainer.batch_size):
        save_image(images[t], f'{save_dir}/img{i*trainer.batch_size+t}.png')

