import os
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.inference.pretrained_models import MODELS
from diffbir.pipeline import SwinIRPipeline, SCUNetPipeline, BSRNetPipeline


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread_rgb_uint8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imsave_rgb_uint8(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def resize_to_hw(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def calc_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def calc_lpips(lpips_metric, pred: np.ndarray, gt: np.ndarray, device: str) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
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


def build_file_map(folder: str) -> dict[str, str]:
    d = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            d[p.stem] = str(p)
    return d


def preprocess_like_diffbir(lq_pil: Image.Image, task: str, version: str, upscale: float) -> Image.Image:
    """
    Match DiffBIR's task-specific `after_load_lq()` behavior.
    """
    if task == "denoise":
        # BIDInferenceLoop.after_load_lq(): always bicubic upscale by `upscale`
        new_size = tuple(int(x * upscale) for x in lq_pil.size)
        lq_pil = lq_pil.resize(new_size, Image.BICUBIC)
    elif task == "face":
        # BFRInferenceLoop.after_load_lq(): always bicubic upscale by `upscale`
        new_size = tuple(int(x * upscale) for x in lq_pil.size)
        lq_pil = lq_pil.resize(new_size, Image.BICUBIC)
    elif task == "sr":
        # BSRInferenceLoop.after_load_lq():
        # only v1 bicubic upscale before stage1 (default cleaner)
        if version == "v1":
            new_size = tuple(int(x * upscale) for x in lq_pil.size)
            lq_pil = lq_pil.resize(new_size, Image.BICUBIC)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return lq_pil


def load_stage1_pipeline(task: str, version: str, device: str, upscale: float):
    """
    Instantiate the exact stage1 cleaner + stage1 pipeline used by DiffBIR inference.
    We pass cldm/diffusion/cond_fn as None because apply_cleaner() does not use them.
    """
    if task == "denoise":
        if version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
            pipeline_cls = SwinIRPipeline
        elif version == "v2":
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
            pipeline_cls = SCUNetPipeline
        elif version == "v2.1":
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
            pipeline_cls = SCUNetPipeline
        else:
            raise ValueError(version)

        cleaner = instantiate_from_config(OmegaConf.load(config))
        cleaner.load_state_dict(load_model_from_url(weight), strict=True)
        cleaner.eval().to(device)
        pipeline = pipeline_cls(cleaner, None, None, None, device)

    elif task == "sr":
        if version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
            cleaner = instantiate_from_config(OmegaConf.load(config))
            cleaner.load_state_dict(load_model_from_url(weight), strict=True)
            cleaner.eval().to(device)
            pipeline = SwinIRPipeline(cleaner, None, None, None, device)

        elif version == "v2":
            config = "configs/inference/bsrnet.yaml"
            weight = MODELS["bsrnet"]
            cleaner = instantiate_from_config(OmegaConf.load(config))
            cleaner.load_state_dict(load_model_from_url(weight), strict=True)
            cleaner.eval().to(device)
            pipeline = BSRNetPipeline(cleaner, None, None, None, device, upscale)

        elif version == "v2.1":
            config = "configs/inference/bsrnet.yaml"
            weight = MODELS["bsrnet"]
            cleaner = instantiate_from_config(OmegaConf.load(config))
            cleaner.load_state_dict(load_model_from_url(weight), strict=True)
            cleaner.eval().to(device)
            pipeline = BSRNetPipeline(cleaner, None, None, None, device, upscale)

        else:
            raise ValueError(version)

    elif task == "face":
        config = "configs/inference/swinir.yaml"
        weight = MODELS["swinir_face"]
        cleaner = instantiate_from_config(OmegaConf.load(config))
        cleaner.load_state_dict(load_model_from_url(weight), strict=True)
        cleaner.eval().to(device)
        pipeline = SwinIRPipeline(cleaner, None, None, None, device)

    else:
        raise ValueError(f"Unsupported task: {task}")

    return pipeline


@torch.no_grad()
def extract_stage1_rgb_uint8(
    pipeline,
    lq_rgb_uint8: np.ndarray,
    device: str,
    cleaner_tiled: bool,
    cleaner_tile_size: int,
    cleaner_tile_stride: int,
) -> np.ndarray:
    """
    Run the exact stage1 cleaner path used by DiffBIR and return RGB uint8 image.
    """
    lq_tensor = (
        torch.tensor(lq_rgb_uint8[None], dtype=torch.float32, device=device)
        .div(255.0)
        .clamp(0, 1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    pipeline.set_output_size(lq_tensor.size())
    cond_img = pipeline.apply_cleaner(
        lq_tensor,
        cleaner_tiled,
        cleaner_tile_size,
        cleaner_tile_stride,
    )

    stage1 = (
        (cond_img * 255.0)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()[0]
    )
    return stage1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sr", "denoise", "face"])
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2", "v2.1"])
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_lpips", action="store_true")
    parser.add_argument("--lpips_model", type=str, default="alex", choices=["alex", "vgg"])
    parser.add_argument("--cleaner_tiled", action="store_true")
    parser.add_argument("--cleaner_tile_size", type=int, default=512)
    parser.add_argument("--cleaner_tile_stride", type=int, default=256)
    parser.add_argument(
        "--metric_resize_to_gt",
        action="store_true",
        help="Resize input/stage1 to GT size before metric if shapes differ.",
    )
    args = parser.parse_args()

    stage1_dir = os.path.join(args.output_dir, "stage1_outputs")
    os.makedirs(stage1_dir, exist_ok=True)

    pipeline = load_stage1_pipeline(
        task=args.task,
        version=args.version,
        device=args.device,
        upscale=args.upscale,
    )

    input_map = build_file_map(args.input_dir)
    gt_map = build_file_map(args.gt_dir)

    common = sorted(set(input_map.keys()) & set(gt_map.keys()))
    if not common:
        raise RuntimeError("No matched file stems between input_dir and gt_dir")

    lpips_metric = None
    if args.eval_lpips:
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LPIPS evaluation requires 'lpips' to be installed. "
                "Install with: pip install lpips"
            ) from exc
        lpips_metric = lpips.LPIPS(net=args.lpips_model).to(args.device)

    csv_path = os.path.join(args.output_dir, "stage1_metrics.csv")
    rows = []

    input_psnr_all, input_ssim_all = [], []
    stage1_psnr_all, stage1_ssim_all = [], []
    input_lpips_all = []
    stage1_lpips_all = []

    for stem in common:
        lq_pil = Image.open(input_map[stem]).convert("RGB")
        lq_pil = preprocess_like_diffbir(lq_pil, args.task, args.version, args.upscale)
        lq_proc = np.array(lq_pil)

        stage1 = extract_stage1_rgb_uint8(
            pipeline=pipeline,
            lq_rgb_uint8=lq_proc,
            device=args.device,
            cleaner_tiled=args.cleaner_tiled,
            cleaner_tile_size=args.cleaner_tile_size,
            cleaner_tile_stride=args.cleaner_tile_stride,
        )

        gt = imread_rgb_uint8(gt_map[stem])

        stage1_save_path = os.path.join(stage1_dir, f"{stem}.png")
        imsave_rgb_uint8(stage1_save_path, stage1)

        input_for_metric = lq_proc
        stage1_for_metric = stage1

        if input_for_metric.shape != gt.shape or stage1_for_metric.shape != gt.shape:
            if args.metric_resize_to_gt:
                input_for_metric = resize_to_hw(input_for_metric, gt.shape[:2])
                stage1_for_metric = resize_to_hw(stage1_for_metric, gt.shape[:2])
            else:
                raise ValueError(
                    f"[{stem}] shape mismatch:\n"
                    f"  input={input_for_metric.shape}\n"
                    f"  stage1={stage1_for_metric.shape}\n"
                    f"  gt={gt.shape}\n"
                    f"Use --metric_resize_to_gt if this is expected."
                )

        input_psnr, input_ssim = calc_psnr_ssim(input_for_metric, gt)
        stage1_psnr, stage1_ssim = calc_psnr_ssim(stage1_for_metric, gt)
        input_lpips = None
        stage1_lpips = None
        if lpips_metric is not None:
            input_lpips = calc_lpips(lpips_metric, input_for_metric, gt, args.device)
            stage1_lpips = calc_lpips(lpips_metric, stage1_for_metric, gt, args.device)

        input_psnr_all.append(input_psnr)
        input_ssim_all.append(input_ssim)
        stage1_psnr_all.append(stage1_psnr)
        stage1_ssim_all.append(stage1_ssim)
        if input_lpips is not None:
            input_lpips_all.append(input_lpips)
        if stage1_lpips is not None:
            stage1_lpips_all.append(stage1_lpips)

        rows.append(
            {
                "image": stem,
                "input_psnr": input_psnr,
                "input_ssim": input_ssim,
                "input_lpips": input_lpips,
                "stage1_psnr": stage1_psnr,
                "stage1_ssim": stage1_ssim,
                "stage1_lpips": stage1_lpips,
                "delta_psnr": stage1_psnr - input_psnr,
                "delta_ssim": stage1_ssim - input_ssim,
                "stage1_path": stage1_save_path,
            }
        )

        print(
            f"[{stem}] "
            f"input PSNR/SSIM = {input_psnr:.4f}/{input_ssim:.4f}, "
            f"stage1 PSNR/SSIM = {stage1_psnr:.4f}/{stage1_ssim:.4f}, "
            f"delta = {stage1_psnr - input_psnr:+.4f}/{stage1_ssim - input_ssim:+.4f}"
        )
        if input_lpips is not None:
            print(f"[{stem}] input LPIPS = {input_lpips:.6f}")
        if stage1_lpips is not None:
            print(f"[{stem}] stage1 LPIPS = {stage1_lpips:.6f}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "input_psnr",
                "input_ssim",
                "input_lpips",
                "stage1_psnr",
                "stage1_ssim",
                "stage1_lpips",
                "delta_psnr",
                "delta_ssim",
                "stage1_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Average =====")
    print(f"input  PSNR: {np.mean(input_psnr_all):.4f}")
    print(f"input  SSIM: {np.mean(input_ssim_all):.4f}")
    if input_lpips_all:
        print(f"input  LPIPS: {np.mean(input_lpips_all):.6f}")
    print(f"stage1 PSNR: {np.mean(stage1_psnr_all):.4f}")
    print(f"stage1 SSIM: {np.mean(stage1_ssim_all):.4f}")
    if stage1_lpips_all:
        print(f"stage1 LPIPS: {np.mean(stage1_lpips_all):.6f}")
    print(f"delta  PSNR: {np.mean(np.array(stage1_psnr_all) - np.array(input_psnr_all)):.4f}")
    print(f"delta  SSIM: {np.mean(np.array(stage1_ssim_all) - np.array(input_ssim_all)):.4f}")
    print(f"\nSaved stage1 images to: {stage1_dir}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
