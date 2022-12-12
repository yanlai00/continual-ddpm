import torch
import torchvision
import os
import torch.utils.data
from PIL import Image
import numpy as np
import wandb
from typing import List
import math
from torch.utils.data.sampler import RandomSampler
import torch.nn.functional as F

def get_data_path():
    return os.environ.get('DATA')

class Gaussian2D(torch.utils.data.Dataset):
    def __init__(self, mean, cov, num_samples, label=0):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.length = num_samples

        self.samples = torch.tensor(np.random.multivariate_normal(mean, cov, num_samples), dtype=torch.float)
        self.labels = self.samples.new_full((num_samples,), label, dtype=torch.long)
        # table = wandb.Table(data=self.samples.numpy(), columns = ["x", "y"])
        # wandb.log({"data" : wandb.plot.scatter(table, "x", "y")}) 

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.samples[index], self.labels[index]
    
    def get_sample(self, num_samples: int):
        indices = torch.randperm(len(self))[0:num_samples]
        return [self.__getitem__(index) for index in indices]

class Gaussian2DWithClass(Gaussian2D):
    def __init__(self, mean, cov, num_samples, label, num_classes):
        super().__init__(mean, cov, num_samples, label)
        self.labels = F.one_hot(self.labels, num_classes=num_classes).float()
        self.samples = torch.cat([self.samples, self.labels], dim=1)

    def __getitem__(self, index: int):
        return self.samples[index], self.labels[index]

class MixGaussian2D(torch.utils.data.Dataset):
    def __init__(self, datasets: List[Gaussian2D]):
        super().__init__()
        self.samples = torch.cat([dataset.samples for dataset in datasets], dim=0)
        self.labels = torch.cat([dataset.samples for dataset in datasets], dim=0)
        self.length = sum([len(dataset) for dataset in datasets])
        
    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.samples[index], self.labels[index]

    def get_sample(self, num_samples: int):
        indices = torch.randperm(len(self))[0:num_samples]
        return self.samples[indices], self.labels[indices]

class CelebA(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()

        # CelebA images folder
        folder = os.path.join(get_data_path(), 'CelebA')
        # List of files
        self._files = os.listdir(folder)
        self._files = [os.path.join(folder, file) for file in self._files]

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


class MNIST(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(get_data_path()), train=True, download=False, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)

class CIFAR10(torchvision.datasets.CIFAR10):
    """
    ### CIFAR10 dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(get_data_path()), train=True, download=False, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)

class SplitCIFAR10(torchvision.datasets.CIFAR10):
    """
    ### CIFAR10 dataset
    """

    def __init__(self, image_size, target):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(get_data_path()), train=True, download=False, transform=transform)
        self.targets = torch.Tensor(self.targets)
        self.data = self.data[self.targets == target]

    def __getitem__(self, item):
        return super().__getitem__(item)

class SplitMNIST(torchvision.datasets.MNIST):
    """
    ### MNIST dataset with a specific label
    """

    def __init__(self, image_size, target):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(get_data_path()), train=True, download=False, transform=transform)
        self.data = self.data[self.targets == target]

    def __getitem__(self, item):
        return super().__getitem__(item)

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.lengths = []
        self.length = 0
        for dataset in datasets:
            self.lengths.append(self.length)
            self.length += len(dataset)
        self.lengths.append(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        for i in range(len(self.lengths)):
            if index >= self.lengths[i] and index < self.lengths[i+1]:
                return self.datasets[i].__getitem__(index - self.lengths[i])

# class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
#     def __init__(self, dataset, batch_size):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.number_of_datasets = len(dataset.datasets)
#         self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])

#     def __len__(self):
#         return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

#     def __iter__(self):
#         samplers_list = []
#         sampler_iterators = []
#         for dataset_idx in range(self.number_of_datasets):
#             cur_dataset = self.dataset.datasets[dataset_idx]
#             sampler = RandomSampler(cur_dataset)
#             samplers_list.append(sampler)
#             cur_sampler_iterator = sampler.__iter__()
#             sampler_iterators.append(cur_sampler_iterator)

#         push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
#         step = self.batch_size * self.number_of_datasets
#         samples_to_grab = self.batch_size
#         # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
#         epoch_samples = self.largest_dataset_size * self.number_of_datasets

#         final_samples_list = []  # this is a list of indexes from the combined dataset
#         for _ in range(0, epoch_samples, step):
#             for i in range(self.number_of_datasets):
#                 cur_batch_sampler = sampler_iterators[i]
#                 cur_samples = []
#                 for _ in range(samples_to_grab):
#                     try:
#                         cur_sample_org = cur_batch_sampler.__next__()
#                         cur_sample = cur_sample_org + push_index_val[i]
#                         cur_samples.append(cur_sample)
#                     except StopIteration:
#                         # got to the end of iterator - restart the iterator and continue to get samples
#                         # until reaching "epoch_samples"
#                         sampler_iterators[i] = samplers_list[i].__iter__()
#                         cur_batch_sampler = sampler_iterators[i]
#                         cur_sample_org = cur_batch_sampler.__next__()
#                         cur_sample = cur_sample_org + push_index_val[i]
#                         cur_samples.append(cur_sample)
#                 final_samples_list.extend(cur_samples)

#         return iter(final_samples_list)