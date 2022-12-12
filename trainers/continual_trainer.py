from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from diffusion.class_conditioned_ddpm import ClassConditionedDenoiseDiffusion
from eps_models.unet import UNet
from eps_models.unet_class_conditioned import ClassConditionedUNet
from eps_models.mlp import Mlp, ClassConditionedMlp
from pathlib import Path
from datetime import datetime
from torch.utils.data import ConcatDataset
import wandb
from ewc import EWC
import torch.nn.functional as F
from trainers.trainer import Trainer

class ContinualTrainer(Trainer):

    datasets: List[torch.utils.data.Dataset]

    def init(self):
        self.n_experiences = len(self.datasets)
        self.dataset = self.datasets[0]
        super().init()

    def run(self):
        for experience_id in range(self.n_experiences):
            self.dataset = self.datasets[experience_id]
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)

            if experience_id == 0:
                epochs = 100
            else:
                epochs = self.epochs
            
            for epoch in range(epochs):
                # Sample some images
                if epoch == 0 or (epoch+1) % 10 == 0:
                    self.sample(self.n_samples)
                # Train the model
                self.train()
                # Save the eps model
                if (epoch+1) % 10 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + epoch
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            
            print(f"Finished Experience {experience_id}")
