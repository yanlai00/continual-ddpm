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
from trainers.trainer import Trainer, get_exp_path

class Gaussian2DTrainer(Trainer):

    def init(self):
        # self.eps_model = Mlp(2, 2, [100, 100]).to(self.device)
        self.eps_model = Mlp(6, 6, [100, 100]).to(self.device)

        self.diffusion = DenoiseDiffusion(
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

    def sample(self, n_samples=10000):
        """
        ### Sample images
        """
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            x = torch.randn((n_samples, 6), device=self.device)
            # y = F.one_hot(0 * torch.ones(n_samples), num_classes=4)
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
            return x

    def run(self):
        """
        ### Training loop
        """
        for epoch in range(self.epochs):
            # Train the model
            self.train()
            # Sample some images
            self.sample(self.n_samples)
            if (epoch+1) % 20 == 0:
                # Save the eps model
                torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{epoch}.pt'))

class ClassConditionedGaussian2DTrainer(Gaussian2DTrainer):

    # Class conditioned
    num_classes: int
    class_embed_size: int

    def init(self):
        self.eps_model = ClassConditionedMlp(2, 2, [100, 100], self.num_classes).to(self.device)

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
            loss = self.diffusion.loss(data, c=labels)
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

    def sample(self, n_samples=64, class_idx=-1):
        """
        ### Sample images
        """
        with torch.no_grad():
            x = torch.randn([n_samples, 2], device=self.device)
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                if class_idx != -1:
                    x = self.diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long), x.new_full((n_samples,), class_idx, dtype=torch.long))
                else:
                    x = self.diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
            return x


class ContinualGaussian2DTrainer(Gaussian2DTrainer):
    datasets: List[torch.utils.data.Dataset]

    def init(self):
        self.n_experiences = len(self.datasets)
        self.dataset = self.datasets[0]
        super().init()

    def run(self):
        for experience_id in range(self.n_experiences):
            self.dataset = self.datasets[experience_id]
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)

            for epoch in range(self.epochs):
                # Train the model
                self.train()
                # Sample some images
                self.sample(self.n_samples)
                # Save the eps model
                if (epoch+1) % 20 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + epoch
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            
            print(f"Finished Experience {experience_id}")

class EWCGaussian2DTrainer(ContinualGaussian2DTrainer):
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
                # Train the model
                self.train()
                # Sample some images
                self.sample(class_idx=experience_id)
                # Save the eps model
                if (epoch+1) % 20 == 0:
                    # Save the eps model
                    cumulative_epoch = experience_id * self.epochs + epoch
                    torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{cumulative_epoch}.pt'))
            print(f"Finished Experience {experience_id}")
            # import pdb; pdb.set_trace()
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
                loss += self.importance * self.ewc.penalty(self.eps_model)

            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            # wandb.log({'loss': loss}, step=self.step)
            if self.step % 5000 == 0:
                print(f'loss at step {self.step}: {loss}')
            # self.sample()


class SelfGenerativeReplayGaussian2DTrainer(ContinualGaussian2DTrainer):
    def run(self):
        for experience_id in range(self.n_experiences):
            
            with torch.no_grad():
                n_samples = len(self.datasets[experience_id]) // 127
                self_generated_data = self.sample(n_samples).cpu().numpy()
            self_generated_dataset = torch.utils.data.Dataset(self_generated_data)
            self.data = ConcatDataset(self.datasets[experience_id], self_generated_dataset)
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
            super().run()

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        for batch_idx, data in enumerate(self.data_loader):
            # Increment global step
            self.step += 1
            # Move data to device
            data = data.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            # wandb.log({'loss': loss}, step=self.step)
            if self.step % 5000 == 0:
                print(f'loss at step {self.step}: {loss}')
            # self.sample()
