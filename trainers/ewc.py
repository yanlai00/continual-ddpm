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

class EWCGaussian2DTrainer(ContinualTrainer):
    importance: int = 1000
    samples_per_task: int = 200

    def run(self):
        self.old_tasks_data = []
        self.ewc = None
        for experience_id in range(self.n_experiences):
            self.dataset = self.datasets[experience_id]
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
            if self.old_tasks_data:
                self.ewc = EWC(self.eps_model, self.diffusion, self.old_tasks_data, self.importance)
            for epoch in range(self.epochs):
                # Sample some images
                if epoch == 0 or (epoch+1) % 20 == 0:
                    self.sample(self.n_samples)
                # Train the model
                self.train()
                # Save the eps model
                if (epoch+1) % 20 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + (epoch+1)
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            print(f"Finished Experience {experience_id}")

            # TODO: Edit this with subset and concat dataset 
            self.old_tasks_data += list(self.dataset.get_sample(self.samples_per_task))

    def train(self):
        # Iterate through the dataset
        for (data, labels) in self.data_loader:
            # Increment global step
            self.step += 1
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)

            if self.ewc is not None:
                ewc_loss = self.importance * self.ewc.penalty(self.eps_model)
                total_loss = loss + ewc_loss
            else:
                total_loss = loss

            # Compute gradients
            total_loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            if self.ewc is not None:
                wandb.log({'loss': loss, 'ewc_loss': ewc_loss}, step=self.step)
            else:
                wandb.log({'loss': loss}, step=self.step)
