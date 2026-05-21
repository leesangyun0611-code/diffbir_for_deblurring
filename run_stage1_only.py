from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.inference.loop import MODELS
from diffbir.model import SwinIR, SCUNet
from diffbir.model.restormer_from_clone import RestormerFromClone
from diffbir.model.nafnet_from_clone import NAFNetFromClone
from diffbir.model.mprnet_from_clone import MPRNetFromClone
from diffbir.pipeline import (
    SwinIRPipeline,
    SCUNetPipeline,
    BSRNetPipeline,
)
from diffbir.model import RRDBNet

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


def resize_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == gt.shape[:2]:
        return pred
    h, w = gt.shape[:2]
    return np.array(Image.fromarray(pred).resize((w, h), Image.BICUBIC))


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    psnr = peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0)
    ssim = structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def compute_lpips(lpips_metric, pred: np.ndarray, gt: np.ndarray, device: str) -> float:
    pred_tensor = (
        torch.from_numpy(pred)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(127.5)
        .sub(1.0)
        .to(device)
    )
    gt_tensor = (
        torch.from_numpy(gt)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(127.5)
        .sub(1.0)
        .to(device)
    )
    with torch.no_grad():
        score = lpips_metric(pred_tensor, gt_tensor)
    return float(score.item())


def load_default_cleaner(task: str, version: str, device: str):
    if task == "denoise":
        if version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
            cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        elif version in ["v2", "v2.1"]:
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
            cleaner = instantiate_from_config(OmegaConf.load(config))
        else:
            raise ValueError(f"Unsupported version for denoise: {version}")
    elif task == "sr":
        if version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
            cleaner: RRDBNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        elif version in ["v2", "v2.1"]:
            config = "configs/inference/bsrnet.yaml"
            weight = MODELS["bsrnet"]
            cleaner = instantiate_from_config(OmegaConf.load(config))
        else:
            raise ValueError(f"Unsupported version for sr: {version}")
    elif task == "face":
        config = "configs/inference/swinir.yaml"
        weight = MODELS["swinir_face"]
        cleaner: SwinIR = instantiate_from_config(OmegaConf.load(config))
    else:
        raise ValueError(f"Unsupported task: {task}")

    model_weight = load_model_from_url(weight)
    cleaner.load_state_dict(model_weight, strict=True)
    cleaner.eval().to(device)
    return cleaner


@torch.no_grad()
def apply_restormer_cleaner(cleaner, lq: torch.Tensor) -> torch.Tensor:
    x, h0, w0 = pad_reflect_to_multiple(lq, multiple=8)
    out = cleaner(x)[:, :, :h0, :w0]
    return out.clamp(0, 1)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default="GT")
    parser.add_argument("--metrics_csv", type=str, default="stage1_metrics.csv")
    parser.add_argument("--resize_pred_to_gt", action="store_true")

    parser.add_argument(
        "--cleaner_type",
        type=str,
        default="default",
        choices=["default", "restormer", "nafnet", "mprnet"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="denoise",
        choices=["denoise", "sr", "face"],
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2.1",
        choices=["v1", "v2", "v2.1"],
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument("--cleaner_tiled", action="store_true")
    parser.add_argument("--cleaner_tile_size", type=int, default=512)
    parser.add_argument("--cleaner_tile_stride", type=int, default=256)

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
    parser.add_argument("--nafnet_repo", type=str, default="third_party/NAFNet")
    parser.add_argument("--nafnet_ckpt", type=str, default="")
    parser.add_argument("--mprnet_repo", type=str, default="third_party/MPRNet")
    parser.add_argument("--mprnet_ckpt", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)

    autocast_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]

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
    elif args.cleaner_type == "restormer":
        cleaner = RestormerFromClone(
            repo_dir=args.restormer_repo,
            task=args.restormer_task,
            ckpt_path=args.restormer_ckpt,
        ).eval().to(args.device)
    elif args.cleaner_type == "nafnet":
        cleaner = NAFNetFromClone(
            repo_dir=args.nafnet_repo,
            ckpt_path=args.nafnet_ckpt,
        ).eval().to(args.device)
    elif args.cleaner_type == "mprnet":
        cleaner = MPRNetFromClone(
            repo_dir=args.mprnet_repo,
            ckpt_path=args.mprnet_ckpt,
        ).eval().to(args.device)
    else:
        raise ValueError(f"Unsupported cleaner_type: {args.cleaner_type}")

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    lpips_metric = None
    if gt_dir is not None and gt_dir.exists():
        import lpips

        lpips_metric = lpips.LPIPS(net="alex").to(args.device)
    else:
        print(f"[INFO] GT directory not found or empty: {args.gt_dir}. Stage1 metrics will be skipped.")

    pipeline = None
    if args.cleaner_type == "default":
        if args.task == "denoise":
            pipeline_class = SwinIRPipeline if args.version == "v1" else SCUNetPipeline
        elif args.task == "sr":
            pipeline_class = SwinIRPipeline if args.version == "v1" else BSRNetPipeline
        elif args.task == "face":
            pipeline_class = SwinIRPipeline
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        if pipeline_class == BSRNetPipeline:
            pipeline = pipeline_class(cleaner, None, None, None, args.device, args.upscale)
        else:
            pipeline = pipeline_class(cleaner, None, None, None, args.device)

    file_rows = []
    metric_rows = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for img_path in files:
        img = np.array(Image.open(img_path).convert("RGB"))
        if args.cleaner_type == "default" and args.task in ["denoise", "face"]:
            img = np.array(
                Image.fromarray(img).resize(
                    tuple(int(x * args.upscale) for x in Image.fromarray(img).size),
                    Image.BICUBIC,
                )
            )
        elif args.cleaner_type == "default" and args.task == "sr" and args.version == "v1":
            img = np.array(
                Image.fromarray(img).resize(
                    tuple(int(x * args.upscale) for x in Image.fromarray(img).size),
                    Image.BICUBIC,
                )
            )
        lq = (
            torch.tensor(img, dtype=torch.float32, device=args.device)
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        with torch.autocast(args.device, autocast_dtype):
            if args.cleaner_type == "default":
                pipeline.set_output_size(lq.size())
                out = pipeline.apply_cleaner(
                    lq,
                    args.cleaner_tiled,
                    args.cleaner_tile_size,
                    args.cleaner_tile_stride,
                )
            else:
                out = apply_restormer_cleaner(cleaner, lq)

        out_img = tensor_to_uint8_image(out)[0]
        save_path = out_dir / img_path.name
        Image.fromarray(out_img).save(save_path)

        file_rows.append(
            {
                "file_name": img_path.name,
                "height": out_img.shape[0],
                "width": out_img.shape[1],
            }
        )
        print(f"[SAVE] {save_path}")

        if gt_dir is None or lpips_metric is None:
            continue

        gt_path = gt_dir / img_path.name
        if not gt_path.exists():
            print(f"[SKIP] GT not found for {img_path.name}")
            continue

        gt_img = np.array(Image.open(gt_path).convert("RGB"))
        metric_img = out_img
        if metric_img.shape[:2] != gt_img.shape[:2]:
            if not args.resize_pred_to_gt:
                print(
                    f"[SKIP] Size mismatch for {img_path.name}: "
                    f"pred={metric_img.shape[:2]}, gt={gt_img.shape[:2]}"
                )
                continue
            metric_img = resize_to_match(metric_img, gt_img)

        psnr, ssim = compute_psnr_ssim(metric_img, gt_img)
        lpips_score = compute_lpips(lpips_metric, metric_img, gt_img, args.device)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_score)
        metric_rows.append(
            {
                "file_name": img_path.name,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_score,
            }
        )
        print(
            f"[METRIC] {img_path.name}: "
            f"PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_score:.6f}"
        )

    csv_path = out_dir / "stage1_only_files.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "height", "width"])
        writer.writeheader()
        writer.writerows(file_rows)

    if metric_rows:
        avg_psnr = float(np.mean(psnr_values))
        avg_ssim = float(np.mean(ssim_values))
        avg_lpips = float(np.mean(lpips_values))
        metric_rows.append(
            {
                "file_name": "AVERAGE",
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "lpips": avg_lpips,
            }
        )
        metrics_path = out_dir / args.metrics_csv
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(metric_rows)

        print("=" * 60)
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.6f}")
        print(f"Saved metrics CSV to: {metrics_path}")

    print("=" * 60)
    print(f"Saved {len(file_rows)} stage1-only images to: {out_dir}")
    print(f"Saved file list CSV to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
