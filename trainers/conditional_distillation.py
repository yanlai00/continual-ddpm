from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from diffusion.class_conditioned_ddpm import ClassConditionedDenoiseDiffusion
from masked_models.unet import ClassConditionedUNet
from pathlib import Path
from datetime import datetime
from torch.utils.data import ConcatDataset
import wandb
from ewc import EWC
import torch.nn.functional as F
from trainers.trainer import Trainer, Gaussian2DTrainer, ClassConditionedGaussian2DTrainer, get_exp_path
from trainers.continual_trainer import ContinualTrainer
from trainers.continual_conditional_trainer import ContinualConditionalTrainer

class ConditionalDistillationTrainer(ContinualConditionalTrainer):
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
            if self.diffusion.has_copy:
                self.diffusion.generate_auxilary_data(data)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            if self.diffusion.has_copy:
                distillation_loss = self.diffusion.distillation_loss()
                total_loss = loss + 5 * distillation_loss
            else:
                total_loss = loss
            # Compute gradients
            total_loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            if self.diffusion.has_copy:
                wandb.log({'loss': loss, 'distillation_loss': distillation_loss}, step=self.step)
            else:
                wandb.log({'loss': loss}, step=self.step)
            

    def run(self):
        """
        ### Training loop
        """
        for experience_id in range(self.n_experiences):
            self.dataset = self.datasets[experience_id]
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
            if experience_id > 0:
                self.diffusion.save_model_copy()

            if experience_id == 0:
                epochs = 100
            else:
                epochs = self.epochs
            
            for epoch in range(epochs):
                # Sample some images
                if epoch == 0 or (epoch+1) % 20 == 0:
                    for class_idx in range(self.num_classes):
                        self.sample(class_idx, self.n_samples)
                        print(f"Finish Generating Class {class_idx}")
                # Train the model
                self.train()
                # Save the eps model
                if (epoch+1) % 20 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + (epoch+1)
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            
            print(f"Finished Experience {experience_id}")

    def sample(self, n_samples=100):
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
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                t_vec = x.new_full((n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)
            # Log samples
            if self.wandb:
                wandb.log({'samples': wandb.Image(x)}, step=self.step)
