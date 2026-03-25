from typing import Literal, overload, Dict, Optional, Tuple
import math
import torch
from torch import nn
import numpy as np

from ..model.cldm import ControlLDM
from ..utils.cond_fn import Guidance, WeightedMSEGuidance


class Sampler(nn.Module):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "Sampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.training_betas = betas
        self.training_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}
        self.parameterization = parameterization
        self.rescale_cfg = rescale_cfg

    def register(
        self, name: str, value: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=dtype))

    def get_cfg_scale(self, default_cfg_scale: float, model_t: int) -> float:
        if self.rescale_cfg and default_cfg_scale > 1:
            cfg_scale = 1 + default_cfg_scale * (
                (1 - math.cos(math.pi * ((1000 - model_t) / 1000) ** 5.0)) / 2
            )
        else:
            cfg_scale = default_cfg_scale
        return cfg_scale

    def _should_apply_guidance(
        self,
        cond_fn: Optional[Guidance],
        model_t: torch.Tensor,
    ) -> bool:
        if cond_fn is None:
            return False
        if cond_fn.scale == 0:
            return False
        t_scalar = int(model_t[0].item())
        return (t_scalar < cond_fn.t_start) and (t_scalar > cond_fn.t_stop)

    def _apply_restoration_guidance(
        self,
        model: ControlLDM,
        pred_x0: torch.Tensor,
        model_t: torch.Tensor,
        cond_fn: Optional[Guidance],
        guidance_decoder_tiled: bool = False,
        guidance_decoder_tile_size: int = -1,
    ) -> torch.Tensor:
        if not self._should_apply_guidance(cond_fn, model_t):
            return pred_x0

        out = pred_x0
        t_scalar = int(model_t[0].item())

        for _ in range(cond_fn.repeat):
            # latent guidance: only safe for plain MSE
            if cond_fn.space == "latent":
                if isinstance(cond_fn, WeightedMSEGuidance):
                    raise ValueError(
                        "Weighted MSE guidance requires G_SPACE='rgb'. "
                        "For latent guidance, use G_LOSS='mse'."
                    )
                if cond_fn.target_latent is None:
                    raise ValueError("cond_fn.target_latent is None.")
                g, _ = cond_fn(cond_fn.target_latent, out, t_scalar)
                out = out + g
            else:
                # rgb guidance: decode latent -> RGB, compute loss in RGB, backprop to latent
                if cond_fn.target is None:
                    raise ValueError("cond_fn.target is None.")

                with torch.enable_grad():
                    z = out.detach().clone().requires_grad_(True)
                    pred_rgb = model.vae_decode(
                        z,
                        guidance_decoder_tiled,
                        guidance_decoder_tile_size,
                    )

                    if isinstance(cond_fn, WeightedMSEGuidance):
                        with torch.no_grad():
                            w = cond_fn._get_weight((cond_fn.target + 1) / 2)
                        loss = ((pred_rgb - cond_fn.target).pow(2) * w).mean((1, 2, 3)).sum()
                    else:
                        loss = (pred_rgb - cond_fn.target).pow(2).mean((1, 2, 3)).sum()

                    g = -torch.autograd.grad(loss, z)[0] * cond_fn.scale

                out = out + g

        return out

    @overload
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = True,
        cond_fn: Optional[Guidance] = None,
        guidance_decoder_tiled: bool = False,
        guidance_decoder_tile_size: int = -1,
    ) -> torch.Tensor:
        ...