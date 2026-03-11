import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffbir.utils.common import instantiate_from_config
from scripts.dataset_gopro_stage1 import GoProStage1Dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lq_dir",
        type=str,
        default=os.path.expanduser("~/datasets/gopro_deblur/gopro_deblur/blur/images"),
    )
    parser.add_argument(
        "--hq_dir",
        type=str,
        default=os.path.expanduser("~/datasets/gopro_deblur/gopro_deblur/sharp/images"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.expanduser("~/diffbir/experiments/gopro_stage1_swinir"),
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=os.path.expanduser(
            "~/diffbir/DiffBIR/weights/realesrgan_s4_swinir_100k.pth"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/swinir.yaml",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_swinir_model(config_path: str, ckpt_path: str, device: str):
    model = instantiate_from_config(OmegaConf.load(config_path))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "params" in ckpt:
            state_dict = ckpt["params"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Init] loaded ckpt: {ckpt_path}")
    print(f"[Init] missing keys: {len(missing)}")
    print(f"[Init] unexpected keys: {len(unexpected)}")

    model = model.to(device)
    return model


def save_checkpoint(save_dir, epoch, model, optimizer):
    ckpt_path = Path(save_dir) / f"swinir_stage1_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"[SAVE] {ckpt_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = GoProStage1Dataset(
        lq_dir=args.lq_dir,
        hq_dir=args.hq_dir,
        patch_size=args.patch_size,
        training=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = load_swinir_model(args.config, args.init_ckpt, args.device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] start from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            lq = batch["lq"].to(args.device, non_blocking=True)
            hq = batch["hq"].to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lq)
            loss = criterion(pred, hq)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % args.log_every == 0:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"step {step:04d}/{len(loader):04d} "
                    f"loss={loss.item():.6f}"
                )

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.save_dir, epoch, model, optimizer)


if __name__ == "__main__":
    main()