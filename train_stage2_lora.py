import copy
import os
from argparse import ArgumentParser

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
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


def main(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.image_every is not None:
        cfg.train.image_every = args.image_every
    if args.train_text_loss and accelerator.is_main_process:
        print(
            "[TextGuidance][Train] Text-loss hooks are enabled for configuration "
            "tracking, but online OCR is not run in the training loop. Use "
            "--text_mask_dir with precomputed masks when wiring the auxiliary "
            "loss into diffusion training."
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
    mark_only_lora_as_trainable(cldm.controlnet)
    total_params, trainable_params = count_parameters(cldm.controlnet)
    if accelerator.is_main_process:
        print(f"Injected LoRA into {len(injected)} modules")
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
            gt, lq, prompt = batch_transform(batch)
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

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

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
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
    parser.add_argument("--text_mask_dir", type=str, default="")
    parser.add_argument("--freeze_stage1", action="store_true", default=True)
    parser.add_argument("--finetune_stage2_only", action="store_true")
    parser.add_argument("--lora_text_finetune", action="store_true")
    args = parser.parse_args()
    main(args)
