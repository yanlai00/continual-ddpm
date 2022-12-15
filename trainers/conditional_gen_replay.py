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
from trainers.trainer import Trainer, get_exp_path
from trainers.continual_trainer import ContinualTrainer
from trainers.continual_conditional_trainer import ContinualConditionalTrainer

class ConditionalGenerativeReplayTrainer(ContinualConditionalTrainer):

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        for batch_idx, (data, labels) in enumerate(self.data_loader):
            if batch_idx == 0:
                if self.diffusion.has_copy:
                    batch_size = data.shape[0]
                    replay_batch_size = self.batch_size // (self.experience_id + 1)
                    assert (batch_size + replay_batch_size * self.experience_id) == self.batch_size
                    gen_replay_data = []
                    gen_replay_labels = []
                    for class_idx in range(self.experience_id + 1):
                        gen_replay_data.append(self._sample(class_idx=class_idx, n_samples=replay_batch_size))
                        gen_replay_labels.append(torch.ones(replay_batch_size) * class_idx)
                    gen_replay_data = torch.cat(gen_replay_data, dim=0)
                    gen_replay_labels = torch.cat(gen_replay_labels, dim=0)
            # Increment global step
            self.step += 1
            # Move data to device
            data = data.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data, labels)
            if self.diffusion.has_copy:
                replay_loss = self.diffusion.loss(gen_replay_data, gen_replay_labels)
                total_loss = (loss * batch_size + replay_loss * replay_batch_size) / self.batch_size
            else:
                total_loss = loss
            # Compute gradients
            total_loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            wandb.log({'loss': total_loss}, step=self.step)
            

    def run(self):
        """
        ### Training loop
        """
        for experience_id in range(self.n_experiences):
            self.experience_id = experience_id
            self.dataset = self.datasets[experience_id]
            batch_size = self.batch_size - (self.batch_size // (experience_id + 1)) * experience_id
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=True, pin_memory=True)
            if experience_id > 0:
                self.diffusion.save_model_copy()

            for epoch in range(self.epochs):
                # Sample some images
                if epoch == 0 or (epoch+1) % 50 == 0:
                    self.sample(self.n_samples)
                # Train the model
                self.train()
                # Save the eps model
                if (epoch+1) % 50 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + (epoch+1)
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            
            print(f"Finished Experience {experience_id}")
        
    def _sample(self, class_idx, n_samples=64):
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
            return x