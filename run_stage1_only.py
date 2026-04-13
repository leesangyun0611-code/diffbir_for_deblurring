from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from accelerate.utils import set_seed

from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.inference.loop import MODELS
from diffbir.model import SwinIR, SCUNet
from diffbir.model.restormer_from_clone import RestormerFromClone

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def check_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to cpu")
        return "cpu"
    if device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS unavailable, fallback to cpu")
            return "cpu"
    return device


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        out_h, out_w = size, size
    elif h < w:
        out_h, out_w = size, int(w * (size / h))
    else:
        out_h, out_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(out_h, out_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph = (h + multiple - 1) // multiple * multiple - h
    pw = (w + multiple - 1) // multiple * multiple - w
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def pad_reflect_to_multiple(imgs: torch.Tensor, multiple: int):
    _, _, h, w = imgs.size()
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return imgs, h, w
    imgs = F.pad(imgs, pad=(0, pw, 0, ph), mode="reflect")
    return imgs, h, w


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = (
        x.clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    return x


def load_default_cleaner(task: str, version: str, device: str):
    if task != "denoise":
        raise NotImplementedError(
            "This stage1-only script currently supports task='denoise' only. "
            "For your current deblurring experiment, that is the correct path."
        )

    if version == "v1":
        config = "configs/inference/swinir.yaml"
        weight = MODELS["swinir_general"]
    elif version in ["v2", "v2.1"]:
        config = "configs/inference/scunet.yaml"
        weight = MODELS["scunet_psnr"]
    else:
        raise ValueError(f"Unsupported version for default cleaner: {version}")

    cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
    model_weight = load_model_from_url(weight)
    cleaner.load_state_dict(model_weight, strict=True)
    cleaner.eval().to(device)
    return cleaner


@torch.no_grad()
def apply_default_cleaner(
    cleaner,
    lq: torch.Tensor,
    task: str,
    version: str,
) -> torch.Tensor:
    # Pure stage1-only output:
    # - no stage2
    # - no forced short-edge resize to 512
    # - only the minimum padding needed by the cleaner
    if task != "denoise":
        raise NotImplementedError("Only task='denoise' is implemented.")

    if version == "v1":
        h0, w0 = lq.shape[2:]
        x = pad_to_multiples_of(lq, multiple=64)
        out = cleaner(x)[:, :, :h0, :w0]
    elif version in ["v2", "v2.1"]:
        out = cleaner(lq)
    else:
        raise ValueError(f"Unsupported version: {version}")

    return out.clamp(0, 1)


@torch.no_grad()
def apply_restormer_cleaner(cleaner, lq: torch.Tensor) -> torch.Tensor:
    x, h0, w0 = pad_reflect_to_multiple(lq, multiple=8)
    out = cleaner(x)[:, :, :h0, :w0]
    return out.clamp(0, 1)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--cleaner_type",
        type=str,
        default="default",
        choices=["default", "restormer"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="denoise",
        choices=["denoise"],
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2.1",
        choices=["v1", "v2", "v2.1"],
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=231)

    parser.add_argument("--restormer_repo", type=str, default="third_party/Restormer")
    parser.add_argument(
        "--restormer_task",
        type=str,
        default="Motion_Deblurring",
        choices=[
            "Motion_Deblurring",
            "Real_Denoising",
            "Gaussian_Color_Denoising",
            "Gaussian_Gray_Denoising",
        ],
    )
    parser.add_argument("--restormer_ckpt", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
    elif in_path.is_file():
        files = [in_path]
    else:
        raise FileNotFoundError(f"Input path not found: {in_path}")

    if args.cleaner_type == "default":
        cleaner = load_default_cleaner(args.task, args.version, args.device)
    else:
        cleaner = RestormerFromClone(
            repo_dir=args.restormer_repo,
            task=args.restormer_task,
            ckpt_path=args.restormer_ckpt,
        ).eval().to(args.device)

    rows = []

    for img_path in files:
        img = np.array(Image.open(img_path).convert("RGB"))
        lq = (
            torch.tensor(img, dtype=torch.float32, device=args.device)
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        if args.cleaner_type == "default":
            out = apply_default_cleaner(cleaner, lq, args.task, args.version)
        else:
            out = apply_restormer_cleaner(cleaner, lq)

        out_img = tensor_to_uint8_image(out)[0]
        save_path = out_dir / img_path.name
        Image.fromarray(out_img).save(save_path)

        rows.append(
            {
                "file_name": img_path.name,
                "height": out_img.shape[0],
                "width": out_img.shape[1],
            }
        )
        print(f"[SAVE] {save_path}")

    csv_path = out_dir / "stage1_only_files.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "height", "width"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print(f"Saved {len(rows)} stage1-only images to: {out_dir}")
    print(f"Saved file list CSV to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()