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
from trainers.continual_conditional_trainer import ContinualConditionalTrainer

class SupSupTrainer(ContinualConditionalTrainer):

    def init(self):
        self.n_experiences = len(self.datasets)
        self.dataset = self.datasets[0]
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
        self.step = 0
        self.exp_path = get_exp_path(name=self.wandb_name)

    def run(self):
        for experience_id in range(self.n_experiences):
            self.dataset = self.datasets[experience_id]
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)

            print(f"Training for task {experience_id}")
            self.eps_model.apply(lambda m: setattr(m, "task", experience_id))
            for p in self.eps_model.parameters():
                p.grad = None

            self.optimizer = torch.optim.Adam([p for p in self.eps_model.parameters() if p.requires_grad], lr=self.learning_rate)

            for epoch in range(self.epochs):
                # Sample some images
                if (epoch+1) % 10 == 0:
                    # Sample some images
                    for class_idx in range(min(experience_id + 1, self.num_classes)):
                        self.eps_model.apply(lambda m: setattr(m, "task", class_idx))
                        self.sample(class_idx, self.n_samples)
                        print(f"Finish Generating Class {class_idx}")
                # Train the model
                self.eps_model.apply(lambda m: setattr(m, "task", experience_id))
                self.train()
                # Save the eps model
                if (epoch+1) % 10 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + epoch
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            
            print(f"Finished Experience {experience_id}")
