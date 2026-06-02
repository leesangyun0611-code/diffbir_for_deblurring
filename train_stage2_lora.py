import copy
import os
from argparse import ArgumentParser

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from diffbir.model import ControlLDM, Diffusion, SwinIR
from diffbir.model.lora import (
    count_parameters,
    inject_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_as_trainable,
)
from diffbir.sampler import SpacedSampler
from diffbir.utils.common import instantiate_from_config, log_txt_as_img, to


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape) -> torch.Tensor:
    out = arr.to(device=timesteps.device)[timesteps].float()
    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)


def _predict_xstart(
    diffusion: Diffusion,
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
) -> torch.Tensor:
    if diffusion.parameterization == "x0":
        return model_output
    if diffusion.parameterization == "eps":
        sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, x_t.shape)
        return (
            x_t
            - _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * model_output
        ) / sqrt_alpha.clamp_min(1e-8)
    if diffusion.parameterization == "v":
        return (
            _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * model_output
        )
    raise NotImplementedError(diffusion.parameterization)


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 3:
        gray = x[:, :1] * 0.2989 + x[:, 1:2] * 0.5870 + x[:, 2:3] * 0.1140
    else:
        gray = x.mean(dim=1, keepdim=True)
    gx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    gray = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    edge = torch.sqrt(F.conv2d(gray, gx).pow(2) + F.conv2d(gray, gy).pow(2) + 1e-12)
    flat = edge.flatten(1)
    scale = flat.amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (edge / scale).clamp(0, 1)


def diffusion_text_weighted_loss(
    model: ControlLDM,
    diffusion: Diffusion,
    x_start: torch.Tensor,
    t: torch.Tensor,
    cond,
    gt_rgb: torch.Tensor,
    text_mask: torch.Tensor | None,
    text_loss_weight: float,
    text_box_weight: float,
    text_edge_weight: float,
    cond_anchor: torch.Tensor | None = None,
    text_anchor_weight: float = 0.0,
    text_anchor_max_timestep: int = 500,
):
    noise = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    model_output = model(x_noisy, t, cond)

    if diffusion.parameterization == "x0":
        target = x_start
    elif diffusion.parameterization == "eps":
        target = noise
    elif diffusion.parameterization == "v":
        target = diffusion.get_v(x_start, noise, t)
    else:
        raise NotImplementedError(diffusion.parameterization)

    error = F.mse_loss(model_output, target, reduction="none")
    base_loss = error.mean()
    if text_mask is None or text_mask.detach().sum().item() <= 0:
        zero = base_loss.new_tensor(0.0)
        return base_loss, base_loss.detach(), zero, zero

    mask = text_mask.to(device=error.device, dtype=error.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) != 1:
        mask = mask[:, :1]
    mask = F.interpolate(mask, size=error.shape[-2:], mode="area").clamp(0, 1)

    gt01 = ((gt_rgb + 1.0) * 0.5).clamp(0, 1)
    edge = _sobel_edges(gt01)
    edge = F.interpolate(edge, size=error.shape[-2:], mode="area").clamp(0, 1)

    zero = base_loss.new_tensor(0.0)
    text_loss = zero
    if text_loss_weight > 0:
        text_weight = mask * (1.0 + float(text_box_weight) + float(text_edge_weight) * edge)
        denom = (text_weight.sum() * error.size(1)).clamp_min(1.0)
        text_loss = (error * text_weight).sum() / denom

    anchor_loss = zero
    if text_anchor_weight > 0 and cond_anchor is not None:
        pred_x0 = _predict_xstart(diffusion, x_noisy, t, model_output)
        anchor_mask = mask
        cond_anchor = cond_anchor.to(device=pred_x0.device, dtype=pred_x0.dtype)
        if cond_anchor.shape[-2:] != pred_x0.shape[-2:]:
            cond_anchor = F.interpolate(cond_anchor, size=pred_x0.shape[-2:], mode="bilinear", align_corners=False)
        active = (t <= int(text_anchor_max_timestep)).to(dtype=pred_x0.dtype).view(-1, 1, 1, 1)
        anchor_mask = anchor_mask * active
        denom = (anchor_mask.sum() * pred_x0.size(1)).clamp_min(1.0)
        anchor_loss = ((pred_x0 - cond_anchor).abs() * anchor_mask).sum() / denom

    total_loss = (
        base_loss
        + float(text_loss_weight) * text_loss
        + float(text_anchor_weight) * anchor_loss
    )
    return total_loss, base_loss.detach(), text_loss.detach(), anchor_loss.detach()


def load_lora_state_into_injected_controlnet(controlnet: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["lora"]
    missing, unexpected = controlnet.load_state_dict(state, strict=False)
    missing_lora = [key for key in missing if "lora_" in key]
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "Failed to resume LoRA checkpoint cleanly. "
            f"missing LoRA keys: {missing_lora[:10]}, "
            f"unexpected LoRA keys: {unexpected_lora[:10]}"
        )


def main(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.image_every is not None:
        cfg.train.image_every = args.image_every
    if args.train_text_loss:
        text_mask_dir = args.text_mask_dir or getattr(cfg.dataset.train.params, "mask_dir", "")
        if not text_mask_dir:
            raise ValueError("--train_text_loss requires --text_mask_dir or dataset.train.params.mask_dir.")
        cfg.dataset.train.params.mask_dir = text_mask_dir
        if accelerator.is_main_process:
            print(
                "[TextGuidance][Train] enabled: "
                f"text_mask_dir={text_mask_dir}, "
                f"text_loss_weight={args.text_loss_weight}, "
                f"text_box_weight={args.text_rgb_weight}, "
                f"text_edge_weight={args.text_edge_weight}, "
                f"text_anchor_weight={args.text_anchor_weight}, "
                f"text_anchor_max_timestep={args.text_anchor_max_timestep}"
            )

    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {len(unused)}\n"
            f"missing weights: {len(missing)}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                "strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {len(init_with_new_zero)}\n"
                f"weights initialized from scratch: {len(init_with_scratch)}"
            )

    lora_cfg = cfg.train.lora
    injected = inject_lora(
        cldm.controlnet,
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        use_conv=lora_cfg.use_conv,
    )
    lora_resume = getattr(cfg.train, "lora_resume", "")
    if lora_resume:
        load_lora_state_into_injected_controlnet(cldm.controlnet, lora_resume)
    mark_only_lora_as_trainable(cldm.controlnet)
    total_params, trainable_params = count_parameters(cldm.controlnet)
    if accelerator.is_main_process:
        print(f"Injected LoRA into {len(injected)} modules")
        if lora_resume:
            print(f"Resume LoRA from checkpoint: {lora_resume}")
        print(f"ControlNet parameters: {total_params:,}")
        print(f"Trainable LoRA parameters: {trainable_params:,}")

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    opt = torch.optim.AdamW(
        lora_parameters(cldm.controlnet),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} image pairs")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training LoRA for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            batch = to(batch, device)
            batch = batch_transform(batch)
            if len(batch) == 4:
                gt, lq, prompt, text_mask = batch
            else:
                gt, lq, prompt = batch
                text_mask = None
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            if text_mask is not None:
                text_mask = rearrange(text_mask, "b h w c -> b c h w").contiguous().float()

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(clean, prompt)
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            if args.train_text_loss:
                loss, base_loss, text_loss, anchor_loss = diffusion_text_weighted_loss(
                    cldm,
                    diffusion,
                    z_0,
                    t,
                    cond_aug,
                    gt,
                    text_mask,
                    args.text_loss_weight,
                    args.text_rgb_weight,
                    args.text_edge_weight,
                    cond_anchor=cond["c_img"],
                    text_anchor_weight=args.text_anchor_weight,
                    text_anchor_max_timestep=args.text_anchor_max_timestep,
                )
            else:
                loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
                base_loss = loss.detach()
                text_loss = loss.new_tensor(0.0)
                anchor_loss = loss.new_tensor(0.0)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)
                    writer.add_scalar("loss/base_step", base_loss.item(), global_step)
                    if args.train_text_loss:
                        writer.add_scalar("loss/text_step", text_loss.item(), global_step)
                        writer.add_scalar("loss/anchor_step", anchor_loss.item(), global_step)

            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "lora": lora_state_dict(pure_cldm.controlnet),
                        "config": OmegaConf.to_container(lora_cfg, resolve=True),
                    }
                    ckpt_path = f"{ckpt_dir}/lora_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if cfg.train.image_every > 0 and (
                global_step % cfg.train.image_every == 0 or global_step == 1
            ):
                n = min(cfg.train.num_log_images, gt.shape[0])
                log_clean = clean[:n]
                log_cond = {k: v[:n] for k, v in cond.items()}
                log_cond_aug = {k: v[:n] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:n], lq[:n]
                log_prompt = prompt[:n]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=cfg.train.sample_steps,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        checkpoint = {
            "lora": lora_state_dict(pure_cldm.controlnet),
            "config": OmegaConf.to_container(lora_cfg, resolve=True),
        }
        torch.save(checkpoint, f"{ckpt_dir}/lora_final.pt")
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--image_every", type=int, default=None)
    parser.add_argument("--train_text_loss", action="store_true")
    parser.add_argument("--text_loss_weight", type=float, default=0.0)
    parser.add_argument("--text_rgb_weight", type=float, default=0.1)
    parser.add_argument("--text_edge_weight", type=float, default=1.0)
    parser.add_argument("--text_anchor_weight", type=float, default=0.0)
    parser.add_argument("--text_anchor_max_timestep", type=int, default=500)
    parser.add_argument("--text_mask_dir", type=str, default="")
    parser.add_argument("--freeze_stage1", action="store_true", default=True)
    parser.add_argument("--finetune_stage2_only", action="store_true")
    parser.add_argument("--lora_text_finetune", action="store_true")
    args = parser.parse_args()
    main(args)
