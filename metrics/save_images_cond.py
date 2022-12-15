import torch
from datasets import CelebA, MNIST, SplitMNIST, Gaussian2D, CIFAR10
from trainers.trainer import Trainer
from trainers.conditional_trainer import ClassConditionedTrainer
import argparse
from torchvision.utils import save_image
import os

save_dir = '/home/yy2694/continual-ddpm/images/mnist_cond_supsup/'
os.mkdir(save_dir)

trainer = ClassConditionedTrainer()

trainer.batch_size = 500
trainer.image_channels = 1
trainer.n_samples = 128
trainer.dataset = MNIST(trainer.image_size)
num_classes = 10

trainer.init()
# trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12122022_204352mnist_cond_replay/checkpoint_1000.pt'))
trainer.eps_model.load_state_dict(torch.load('/home/yy2694/continual-ddpm/checkpoints/12122022_204352mnist_cond_replay/checkpoint_1000.pt'))
num_iterations = 1

step = 0
for i in range(num_iterations):
    for class_idx in range(num_classes):
        images = trainer.sample(class_idx, trainer.batch_size)
        for t in range(trainer.batch_size):
            save_image(images[t], f'{save_dir}/{class_idx}/img{i*trainer.batch_size+t}.png')

