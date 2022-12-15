import math
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from eps_models.unet import TimeEmbedding, Swish, AttentionBlock, DownBlock, UpBlock, MiddleBlock, Upsample, Downsample, UNet

class ClassConditionedUNet(UNet):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2, num_classes: int = 10):
        super().__init__(image_channels, n_channels, ch_mults, is_attn, n_blocks)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, n_channels * 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_idx: torch.Tensor):
        t = self.time_emb(t)
        t += self.class_emb(class_idx)

        return self.unet_forward(x, t)