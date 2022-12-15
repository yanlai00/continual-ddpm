from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather # Used for Image Data
# from utils import gather2d as gather # Used for Gaussian 2D Data

from diffusion.ddpm import DenoiseDiffusion

class ClassConditionedDenoiseDiffusion(DenoiseDiffusion):
    """
    ## Denoise Diffusion
    """

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, class_idx: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t, class_idx)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        # import pdb; pdb.set_trace()
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, class_idx: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t, class_idx)

        # MSE loss
        return F.mse_loss(noise, eps_theta)

    def generate_auxilary_data(self, x0: torch.Tensor, experience_id: int):
        with torch.no_grad():
            batch_size, image_size = x0.shape[0], x0.shape[1:]
            self.aux_t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
            self.aux_label = torch.randint(0, experience_id, (batch_size,), device=x0.device, dtype=torch.long)
            self.aux_data = torch.randn((batch_size, *image_size), dtype=x0.dtype, device=x0.device) # TODO: Other methods of generating auxilary data
            self.aux_noise = self.eps_model_copy(self.aux_data, self.aux_t, self.aux_label)

    def distillation_loss(self):
        eps_theta = self.eps_model(self.aux_data, self.aux_t, self.aux_label)
        return F.l1_loss(self.aux_noise, eps_theta)
