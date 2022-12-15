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

class ReplayTrainer(ContinualTrainer):

    n_replay_samples: int = 50

    def run(self):
        """
        ### Training loop
        """
        for experience_id in range(self.n_experiences):
            if experience_id == 0:
                self.prev_datasets = []
                self.dataset = self.datasets[experience_id]
            else:
                prev_task = experience_id - 1
                replay_dataset = torch.utils.data.Subset(self.datasets[prev_task], list(range(self.n_replay_samples)))
                self.prev_datasets.append(replay_dataset)
                curr_dataset = self.datasets[experience_id]
                multipler = max(len(curr_dataset) // len(replay_dataset), 1)
                self.dataset = torch.utils.data.ConcatDataset([*(self.prev_datasets * multipler), curr_dataset])
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)

            epochs = self.epochs // (experience_id + 1)

            for epoch in range(epochs):
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


