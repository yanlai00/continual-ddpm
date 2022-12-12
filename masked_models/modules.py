import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import masked_models.module_util as module_util

from masked_models.supsup_args import args as pargs

from scipy.stats import ortho_group

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d

class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )

class MultitaskNonAffineBN(nn.Module):
    def __init__(self, dim):
        super(MultitaskNonAffineBN, self).__init__()
        self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(pargs.num_tasks)])
        self.task = 0

    def forward(self, x):
        return self.bns[self.task](x)

class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        # Turn the gradient on the weights off
        self.weight.requires_grad = False

        # default sparsity
        self.sparsity = pargs.sparsity

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

# Conv from What's Hidden in a Randomly Weighted Neural Network?
class MultitaskMaskConv(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.ParameterList(
            [
                nn.Parameter(module_util.mask_init(self))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.weight.requires_grad = False

        self.sparsity = pargs.sparsity

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    module_util.get_subnet(self.scores[j].abs(), self.sparsity)
                    for j in range(pargs.num_tasks)
                ]
            ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(
            self.scores[self.task].abs(), self.sparsity
        )
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        return f"MultitaskMaskConv({self.in_channels}, {self.out_channels})"

# TransposeConv
class MultitaskMaskTransposeConv(nn.ConvTranspose2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.ParameterList(
            [
                nn.Parameter(module_util.mask_init(self))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.weight.requires_grad = False

        self.sparsity = pargs.sparsity

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    module_util.get_subnet(self.scores[j].abs(), self.sparsity)
                    for j in range(pargs.num_tasks)
                ]
            ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(
            self.scores[self.task].abs(), self.sparsity
        )
        w = self.weight * subnet
        x = F.conv_transpose2d(
            x, w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        return x

    def __repr__(self):
        return f"MultitaskMaskConv({self.in_channels}, {self.out_channels})"

# Init from What's Hidden with masking from Mallya et al. (Piggyback)
class FastMultitaskMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks)]
        )

        self.weight.requires_grad = False

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    module_util.get_subnet_fast(self.scores[j])
                    for j in range(pargs.num_tasks)
                ]
            ),
        )


    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        subnet = module_util.GetSubnetFast.apply(self.scores[self.task])

        w = self.weight * subnet

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        return f"FastMultitaskMaskConv({self.in_channels}, {self.out_channels})"


class BatchEnsembles(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = nn.ParameterList(
            [
                nn.Parameter(module_util.rank_one_init(self).unsqueeze(1))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.t = nn.ParameterList(
            [
                nn.Parameter(module_util.rank_one_initv2(self).unsqueeze(0))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.weight.requires_grad = False

    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    torch.mm(self.s[j], self.t[j]).view(*self.weight.shape)
                    for j in range(pargs.num_tasks)
                ]
            ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        subnet = torch.mm(self.s[self.task], self.t[self.task]).view(
            *self.weight.shape
        )
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class VectorizedBatchEnsembles(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = nn.ParameterList(
            [
                nn.Parameter(module_util.rank_one_init(self).unsqueeze(0))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.t = nn.ParameterList(
            [
                nn.Parameter(module_util.rank_one_initv2(self).unsqueeze(0))
                for _ in range(pargs.num_tasks)
            ]
        )

    def forward(self, x):
        batch_sz = x.size(0)
        if self.task >= 0:
            new_x = x * self.t[self.task].repeat(batch_sz, 1).view(
                batch_sz, self.in_channels, 1, 1
            )
        else:
            multiplier = torch.stack([self.t[j % self.num_tasks_learned].flatten() for j in range(batch_sz)], 0).view(batch_sz, self.in_channels, 1, 1)
            new_x = x * multiplier
        out = F.conv2d(
            new_x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.task >= 0:
            new_out = out * self.s[self.task].repeat(batch_sz, 1).view(
                batch_sz, self.out_channels, 1, 1
            )
        else:
            multiplier = torch.stack([self.s[j % self.num_tasks_learned].flatten() for j in range(batch_sz)], 0).view(batch_sz, self.out_channels, 1, 1)
            new_out = out * multiplier
        return new_out

    def __repr__(self):
        return f"VectorizedBatchEnsembles({self.in_channels}, {self.out_channels})"


class IndividualHeads(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(self.weight.data.clone()) for _ in range(pargs.num_tasks)]
        )
        self.weight.requires_grad = False

    def forward(self, x):
        w = self.scores[self.task]
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        return f"IndividualHeads({self.in_channels}, {self.out_channels})"


class FastHopMaskBN(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        super(FastHopMaskBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.scores = nn.ParameterList(
            [
                nn.Parameter(module_util.bn_mask_initv2(self))
                for _ in range(pargs.num_tasks)
            ]
        )
        self.d = num_features
        self.register_parameter("score", nn.Parameter(module_util.bn_mask_init(self)))

    def cache_masks(self):
        with torch.no_grad():
            d = self.d
            W = torch.zeros(d, d).to(pargs.device)
            for j in range(self.num_tasks_learned):
                x = 2 * module_util.get_subnet_fast(self.scores[j]) - 1
                heb = torch.ger(x, x) - torch.eye(d).to(pargs.device)
                h = W.mm(x.unsqueeze(1)).squeeze()
                pre = torch.ger(x, h)
                W = W + (1.0 / d) * (heb - pre - pre.t())
                # W = W + (1. / d) * heb

            self.register_buffer("W", W)

    def clear_masks(self):
        self.register_buffer("W", None)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        subnet = module_util.GetSubnetFast.apply(self.scores[self.task])
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            subnet,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )


class PSPRotation(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if pargs.ortho_group:
            self.contexts = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.from_numpy(
                            ortho_group.rvs(self.in_channels).astype("float32")
                        )
                    )
                    for _ in range(pargs.num_tasks)
                ]
            )
        else:
            self.contexts = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.nn.init.orthogonal_(
                            torch.Tensor(self.in_channels, self.in_channels)
                        )
                    )
                    for _ in range(pargs.num_tasks)
                ]
            )

        self.scores = nn.ParameterList(
            [
                nn.Parameter(module_util.pspinit(self).squeeze())
                for _ in range(pargs.num_tasks)
            ]
        )

    def cache_weights(self, t):
        out = torch.stack([self.scores[j].mm(self.contexts[j]) for j in range(t)]).sum(
            dim=0
        )
        self.register_buffer("weight_sum", out)

    def cache_masks(self):
        self.register_buffer(
            "stacked", torch.stack([self.contexts[j] for j in range(pargs.num_tasks)]),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):

        x = x.squeeze().t()
        out = self.contexts[self.task].mm(x)
        out = self.weight.squeeze().mm(out)
        out = out.t()
        out = out.view(*out.size(), 1, 1)
        return out



class StackedFastMultitaskMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.ParameterList(
            [nn.Parameter(module_util.mask_init(self)) for _ in range(pargs.num_tasks)]
        )

        self.weight.requires_grad = False

    def forward(self, x):
        subnet = module_util.GetSubnetFast.apply(self.scores[self.task])
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = pargs.num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(module_util.mask_init(self))
                for _ in range(self.num_tasks)
            ]
        )
        
        # Keep weights untrained
        self.weight.requires_grad = False
        self.sparsity = pargs.sparsity
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    module_util.GetSubnet.apply(self.scores[j], self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        # Subnet forward pass (given task info in self.task)
        subnet = module_util.GetSubnet.apply(self.scores[self.task], self.sparsity)
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x


    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_features}, {self.out_features})"