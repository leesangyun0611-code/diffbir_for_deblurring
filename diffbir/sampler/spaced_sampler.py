from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn
from ..utils.cond_fn import Guidance, TextGuidance, WeightedMSEGuidance


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride

        all_steps += taken_steps
        start_idx += size

    return set(all_steps)


class SpacedSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "SpacedSampler":
        super().__init__(betas, parameterization, rescale_cfg)

    def make_schedule(self, num_steps: int) -> None:
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32)

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            model_cond = model(x, model_t, cond)
            model_uncond = model(x, model_t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

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

    def _should_apply_any_guidance(
        self,
        cond_fn: Optional[Guidance],
        model_t: torch.Tensor,
        step_index: int,
    ) -> bool:
        if cond_fn is None:
            return False
        text_guidance = getattr(cond_fn, "text_guidance", None)
        return self._should_apply_guidance(cond_fn, model_t) or (
            text_guidance is not None and text_guidance.should_apply(step_index)
        )

    def _apply_restoration_guidance(
        self,
        model: ControlLDM,
        pred_x0: torch.Tensor,
        model_t: torch.Tensor,
        cond_fn: Optional[Guidance],
        step_index: int,
        guidance_decoder_tiled: bool = False,
        guidance_decoder_tile_size: int = -1,
    ) -> torch.Tensor:
        if not self._should_apply_any_guidance(cond_fn, model_t, step_index):
            return pred_x0

        out = pred_x0
        t_scalar = int(model_t[0].item())
        use_restoration_guidance = self._should_apply_guidance(cond_fn, model_t)
        text_guidance: Optional[TextGuidance] = getattr(cond_fn, "text_guidance", None)

        # weighted MSE guidance is defined in RGB space
        use_rgb_guidance = (cond_fn.space == "rgb") or isinstance(
            cond_fn, WeightedMSEGuidance
        )

        for _ in range(cond_fn.repeat):
            if use_restoration_guidance and not use_rgb_guidance:
                target_latent = getattr(cond_fn, "target_latent", None)
                if target_latent is None:
                    raise ValueError("Guidance target_latent is not set.")

                g, _ = cond_fn(target_latent, out, t_scalar)
                out = out + g
            elif use_restoration_guidance:
                if cond_fn.target is None:
                    raise ValueError("Guidance RGB target is not set.")

                with torch.enable_grad():
                    z = out.detach().clone().float().requires_grad_(True)

                    # force the guidance branch to run in fp32
                    with torch.autocast(device_type=z.device.type, enabled=False):
                        pred_rgb = model.vae_decode(
                            z,
                            guidance_decoder_tiled,
                            guidance_decoder_tile_size,
                        ).float()

                        target_rgb = cond_fn.target.float()

                        if isinstance(cond_fn, WeightedMSEGuidance):
                            with torch.no_grad():
                                w = cond_fn._get_weight((target_rgb + 1) / 2).float()
                            loss = ((pred_rgb - target_rgb).pow(2) * w).mean(
                                (1, 2, 3)
                            ).sum()
                        else:
                            loss = (pred_rgb - target_rgb).pow(2).mean(
                                (1, 2, 3)
                            ).sum()

                    g = -torch.autograd.grad(loss, z)[0] * float(cond_fn.scale)

                out = out + g

            if text_guidance is not None and text_guidance.should_apply(step_index):
                with torch.enable_grad():
                    z = out.detach().clone().float().requires_grad_(True)
                    with torch.autocast(device_type=z.device.type, enabled=False):
                        pred_rgb = model.vae_decode(
                            z,
                            guidance_decoder_tiled,
                            guidance_decoder_tile_size,
                        ).float()
                        loss = text_guidance.loss(pred_rgb)

                    g = -torch.autograd.grad(loss, z)[0] * float(text_guidance.scale)
                    if text_guidance.grad_clip > 0:
                        g = g.clamp(-text_guidance.grad_clip, text_guidance.grad_clip)
                    grad_max = float(g.detach().abs().max().item())
                    print(
                        f"[TextGuidance] step={step_index} t={t_scalar} "
                        f"loss={float(loss.detach().item()):.6f} grad_max={grad_max:.6f}"
                    )

                out = out + g

        return out

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        cond_fn: Optional[Guidance] = None,
        step_index: int = -1,
        guidance_decoder_tiled: bool = False,
        guidance_decoder_tile_size: int = -1,
    ) -> torch.Tensor:
        model_output = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)

        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)

        pred_x0 = self._apply_restoration_guidance(
            model=model,
            pred_x0=pred_x0,
            model_t=model_t,
            cond_fn=cond_fn,
            step_index=step_index,
            guidance_decoder_tiled=guidance_decoder_tiled,
            guidance_decoder_tile_size=guidance_decoder_tile_size,
        )

        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
        return x_prev

    @torch.no_grad()
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
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        cond_fn: Optional[Guidance] = None,
        guidance_decoder_tiled: bool = False,
        guidance_decoder_tile_size: int = -1,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if cond_fn is not None and getattr(cond_fn, "text_guidance", None) is not None:
            cond_fn.text_guidance.configure_steps(steps)

        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )

        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
                cond_fn=cond_fn,
                step_index=i,
                guidance_decoder_tiled=guidance_decoder_tiled,
                guidance_decoder_tile_size=guidance_decoder_tile_size,
            )

        if tiled:
            model.forward = forward
        return x
