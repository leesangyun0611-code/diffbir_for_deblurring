from typing import overload, Tuple, Optional
import torch
from torch.nn import functional as F


class Guidance:

    def __init__(
        self, scale: float, t_start: int, t_stop: int, space: str, repeat: int
    ) -> "Guidance":
        """
        Initialize restoration guidance.

        Args:
            scale (float): Gradient scale (denoted as `s` in our paper).
            t_start (int), t_stop (int): Guidance active when t < t_start and t > t_stop.
            space (str): rgb or latent
            repeat (int): Number of guidance updates per sampling step
        """
        self.scale = scale * 3000
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None              # RGB target, usually cond_img * 2 - 1
        self.target_latent = None       # latent target, usually clean cond["c_img"]
        self.space = space
        self.repeat = repeat

    def load_target(
        self,
        target: torch.Tensor,
        target_latent: Optional[torch.Tensor] = None,
    ) -> None:
        self.target = target
        self.target_latent = target_latent

    def __call__(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        pred_x0 = pred_x0.detach().clone()
        target_x0 = target_x0.detach().clone()
        return self._forward(target_x0, pred_x0, t)

    @overload
    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]: ...


class MSEGuidance(Guidance):

    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
        g = -torch.autograd.grad(loss, pred_x0)[0] * self.scale
        return g, loss.item()


class WeightedMSEGuidance(Guidance):

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        # convert RGB to grayscale
        rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        target = torch.sum(
            target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True
        )

        # sobel kernels
        G_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        G_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
        G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
        G = torch.stack((G_x, G_y))

        target = F.pad(target, (1, 1, 1, 1), mode="replicate")
        grad = F.conv2d(target, G, stride=1)
        mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

        n, c, h, w = mag.size()
        block_size = 2
        blocks = (
            mag.view(n, c, h // block_size, block_size, w // block_size, block_size)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        block_mean = (
            blocks.sum(dim=(-2, -1), keepdim=True)
            .tanh()
            .repeat(1, 1, 1, 1, block_size, block_size)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        block_mean = block_mean.view(n, c, h, w)
        weight_map = 1 - block_mean
        return weight_map

    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        g = -torch.autograd.grad(loss, pred_x0)[0] * self.scale
        return g, loss.item()