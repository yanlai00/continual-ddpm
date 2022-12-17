# Continual Learning for Diffusion Models

This is a PyTorch implementation of different continual learning strategies for class-incremental image generation with [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM).

## Requirements
```
python=3.9  
pytorch=1.10  
torchvision=0.11  
tqdm
scipy
```

## Running Continual Learning Experiments

### Train DDPM Models
The experiment config files can be found in the `experiments/` folder. Here are some examples to run the training code:  
Unconditional image generation on MNIST with knowledge distillation:
```
python -m experiments.uncond.mnist.kd
```
Conditional image generation on MNIST with sequential fine-tuning:
```
python -m experiments.cond.mnist.sft
```
Unconditional image generation on CIFAR with joint training:
```
python -m experiments.cond.cifar.joint
```

### Generate Images from Checkpoint
Change the `save_dir` variable and the checkpoint loading path in `metrics/save_images.py`. Then run
```
python -m metrics.save_images
```

### Compute the FID Score
```
python -m metrics.save_images
```
