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

    def __init__(
        self,
        scale: float,
        t_start: int,
        t_stop: int,
        space: str,
        repeat: int,
        weight_mode: str = "lowfreq",
        weight_floor: float = 0.0,
        weight_gamma: float = 1.0,
        block_size: int = 2,
    ) -> "WeightedMSEGuidance":
        """
        Weighted MSE guidance with selectable frequency emphasis.

        Args:
            weight_mode (str): "lowfreq" (original DiffBIR-style) or "highfreq"
            weight_floor (float): minimum weight floor in [0, 1)
            weight_gamma (float): exponent for weight shaping, >= 1.0 recommended
            block_size (int): patch size for patch-level gradient aggregation
        """
        super().__init__(scale, t_start, t_stop, space, repeat)

        if weight_mode not in ["lowfreq", "highfreq"]:
            raise ValueError(
                f"Unsupported weight_mode: {weight_mode}. "
                "Choose from ['lowfreq', 'highfreq']."
            )
        if not (0.0 <= weight_floor < 1.0):
            raise ValueError(
                f"weight_floor must satisfy 0.0 <= weight_floor < 1.0, got {weight_floor}"
            )
        if weight_gamma <= 0.0:
            raise ValueError(
                f"weight_gamma must be > 0.0, got {weight_gamma}"
            )
        if block_size < 1:
            raise ValueError(
                f"block_size must be >= 1, got {block_size}"
            )

        self.weight_mode = weight_mode
        self.weight_floor = weight_floor
        self.weight_gamma = weight_gamma
        self.block_size = block_size

    def _to_grayscale(self, target: torch.Tensor) -> torch.Tensor:
        if target.size(1) == 3:
            rgb_to_gray_kernel = torch.tensor(
                [0.2989, 0.5870, 0.1140],
                dtype=target.dtype,
                device=target.device,
            ).view(1, 3, 1, 1)
            target = torch.sum(target * rgb_to_gray_kernel, dim=1, keepdim=True)
        elif target.size(1) == 1:
            pass
        else:
            # Fallback: average over channels if unexpected channel size appears
            target = target.mean(dim=1, keepdim=True)
        return target

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        # convert RGB to grayscale
        target = self._to_grayscale(target)

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
        block_size = self.block_size

        # pad to multiples of block_size if needed
        pad_h = (block_size - (h % block_size)) % block_size
        pad_w = (block_size - (w % block_size)) % block_size
        if pad_h > 0 or pad_w > 0:
            mag = F.pad(mag, (0, pad_w, 0, pad_h), mode="replicate")

        _, _, h_pad, w_pad = mag.size()

        blocks = (
            mag.view(
                n,
                c,
                h_pad // block_size,
                block_size,
                w_pad // block_size,
                block_size,
            )
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
        block_mean = block_mean.view(n, c, h_pad, w_pad)

        # crop back to original size
        if pad_h > 0 or pad_w > 0:
            block_mean = block_mean[:, :, :h, :w]

        block_mean = block_mean.clamp(0.0, 1.0)

        # original DiffBIR-style: low-frequency regions get larger weights
        if self.weight_mode == "lowfreq":
            weight_map = 1.0 - block_mean
        # new experiment: high-frequency regions get larger weights
        elif self.weight_mode == "highfreq":
            weight_map = block_mean
        else:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

        # optional shaping
        weight_map = weight_map.clamp(0.0, 1.0)
        weight_map = weight_map.pow(self.weight_gamma)
        weight_map = self.weight_floor + (1.0 - self.weight_floor) * weight_map
        weight_map = weight_map.clamp(0.0, 1.0)

        return weight_map

    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            # target_x0 is assumed to be in [-1, 1], so convert to [0, 1] before Sobel
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        g = -torch.autograd.grad(loss, pred_x0)[0] * self.scale
        return g, loss.item()