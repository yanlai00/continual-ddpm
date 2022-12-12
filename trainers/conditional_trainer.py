from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from diffusion.class_conditioned_ddpm import ClassConditionedDenoiseDiffusion
from eps_models.unet import UNet
from eps_models.unet_class_conditioned import ClassConditionedUNet
from pathlib import Path
from datetime import datetime
from torch.utils.data import ConcatDataset
import wandb
from ewc import EWC
import torch.nn.functional as F
from trainers.trainer import Trainer, get_exp_path

class ClassConditionedTrainer(Trainer):

    # Class conditioned
    num_classes: int

    def init(self):
        self.eps_model = ClassConditionedUNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create DDPM class
        self.diffusion = ClassConditionedDenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )
        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        self.step = 0
        self.exp_path = get_exp_path(name=self.wandb_name)

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        for batch_idx, (data, labels) in enumerate(self.data_loader):
            # Increment global step
            self.step += 1
            # Move data to device
            data = data.to(self.device)
            labels = labels.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data, labels)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            if self.wandb:
                wandb.log({'loss': loss}, step=self.step)
            else:
                if self.step % 5000 == 0:
                    print(f'loss at step {self.step}: {loss}')

    def sample(self, class_idx, n_samples=64):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            # Sample Initial Image (Random Gaussian Noise)
            x = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                time_vec = x.new_full((n_samples,), t, dtype=torch.long)
                labels = x.new_full((n_samples,), class_idx, dtype=torch.long)
                x = self.diffusion.p_sample(x, time_vec, labels)
            # Log samples
            if self.wandb:
                wandb.log({f'samples_{class_idx}': wandb.Image(x)}, step=self.step)
            return x

    def run(self):
        """
        ### Training loop
        """
        for epoch in range(self.epochs):
            if epoch % 5 == 0:
                # Sample some images
                for class_idx in range(self.num_classes):
                    self.sample(class_idx, self.n_samples)
                    print(f"Finish Generating Class {class_idx}")
            # Train the model
            self.train()
            if (epoch+1) % 5 == 0:
                # Save the eps model
                torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{epoch+1}.pt'))
