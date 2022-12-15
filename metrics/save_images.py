import torch
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10
from trainers.trainer import Trainer
import argparse
from torchvision.utils import save_image
import os

save_dir = '/home/yy2694/continual-ddpm/images/mnist_uncond_joint/'
os.mkdir(save_dir)

trainer = Trainer()

trainer.batch_size = 500
trainer.image_channels = 1
trainer.n_samples = 128
trainer.dataset = MNIST(trainer.image_size)

trainer.init()
trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12022022_144833/checkpoint_99.pt'))

num_iterations = 10000 // trainer.batch_size

step = 0
for i in range(num_iterations):
    images = trainer.sample(trainer.batch_size)
    for t in range(trainer.batch_size):
        save_image(images[t], f'{save_dir}/img{i*trainer.batch_size+t}.png')

