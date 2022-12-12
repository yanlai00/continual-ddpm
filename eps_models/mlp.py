from typing import Optional, Tuple, Union, List, Callable
import torch
from torch import nn
from torch.nn import functional as F
from utils import identity, fanin_init
from eps_models.unet import TimeEmbedding

class ClassEmbedding(nn.Module):
    """
    ### Embeddings for $c$
    """

    def __init__(self, num_classes: int, class_embed_size: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_embed_size = class_embed_size
        self.fcs = nn.Sequential(
            nn.Linear(num_classes, class_embed_size),
            nn.ReLU(),
            nn.Linear(class_embed_size, class_embed_size)
        )

    def forward(self, c: torch.Tensor):
        if not (isinstance(c, torch.FloatTensor) or isinstance(c, torch.cuda.FloatTensor)):
            emb = F.one_hot(c, num_classes=self.num_classes).float()
            emb = self.fcs(emb)
        else:
            emb = self.fcs(c)

        return emb

class Mlp(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: List[int],
            init_w: float = 1e-4,
            hidden_activation: Callable = F.relu,
            output_activation: Callable = identity,
            hidden_init: Callable = fanin_init,
            b_init_value: float = 0.,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        self.time_fcs = []
        in_size = input_size

        n_channels = hidden_sizes[0]
        self.time_emb_size = n_channels * 4
        self.time_emb = TimeEmbedding(self.time_emb_size)

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            time_fc = nn.Linear(self.time_emb_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            hidden_init(time_fc.weight)
            fc.bias.data.fill_(b_init_value)
            time_fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.__setattr__("time_fc{}".format(i), time_fc)
            self.fcs.append(fc)
            self.time_fcs.append(time_fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x += self.time_fcs[i](t)
            x = self.hidden_activation(x)
        preactivation = self.last_fc(x)
        output = self.output_activation(preactivation)
        return output

class ClassConditionedMlp(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: List[int],
            num_classes: int,
            class_embed_size: Optional[int] = None,
            init_w: float = 1e-4,
            hidden_activation: Callable = F.relu,
            output_activation: Callable = identity,
            hidden_init: Callable = fanin_init,
            b_init_value: float = 0.,
    ):
        super().__init__()
        
        n_channels = hidden_sizes[0]
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_classes = num_classes
        if class_embed_size is None:
            class_embed_size = n_channels
        self.class_embed_size = class_embed_size
        self.fcs = []
        self.time_fcs = []
        self.class_fcs = []
        in_size = input_size

        self.time_emb_size = n_channels * 4
        self.time_emb = TimeEmbedding(self.time_emb_size)

        self.class_emb = ClassEmbedding(self.num_classes, self.class_embed_size)

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            time_fc = nn.Linear(self.time_emb_size, next_size)
            class_fc = nn.Linear(self.class_embed_size, next_size)

            in_size = next_size

            hidden_init(fc.weight)
            hidden_init(time_fc.weight)
            hidden_init(class_fc.weight)

            fc.bias.data.fill_(b_init_value)
            time_fc.bias.data.fill_(b_init_value)
            class_fc.bias.data.fill_(b_init_value)

            self.__setattr__("fc{}".format(i), fc)
            self.__setattr__("time_fc{}".format(i), time_fc)
            self.__setattr__("class_fc{}".format(i), class_fc)

            self.fcs.append(fc)
            self.time_fcs.append(time_fc)
            self.class_fcs.append(class_fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)


    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None):
        if c is not None:
            c = self.class_emb(c)
        t = self.time_emb(t)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x += self.time_fcs[i](t)
            if c is not None:
                x += self.class_fcs[i](c)
            x = self.hidden_activation(x)
        preactivation = self.last_fc(x)
        output = self.output_activation(preactivation)
        return output

class ClassConcatMlp(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: List[int],
            num_classes: int = 0,
            class_embed_size: Optional[int] = None,
            init_w: float = 1e-4,
            hidden_activation: Callable = F.relu,
            output_activation: Callable = identity,
            hidden_init: Callable = fanin_init,
            b_init_value: float = 0.,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_classes = num_classes
        if class_embed_size is None:
            class_embed_size = num_classes
        self.class_embed_size = class_embed_size
        self.fcs = []
        self.time_fcs = []
        in_size = input_size + class_embed_size

        n_channels = hidden_sizes[0]
        self.time_emb_size = n_channels * 4
        self.time_emb = TimeEmbedding(self.time_emb_size)

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            time_fc = nn.Linear(self.time_emb_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            hidden_init(time_fc.weight)
            fc.bias.data.fill_(b_init_value)
            time_fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.__setattr__("time_fc{}".format(i), time_fc)
            self.fcs.append(fc)
            self.time_fcs.append(time_fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)


    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None):
        if c is not None:
            c = F.one_hot(c, num_classes=self.num_classes).float()
            x = torch.cat([x, c], dim=1)
        t = self.time_emb(t)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x += self.time_fcs[i](t)
            x = self.hidden_activation(x)
        preactivation = self.last_fc(x)
        output = self.output_activation(preactivation)
        return output


