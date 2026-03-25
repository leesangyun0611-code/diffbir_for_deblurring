import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == gt.shape[:2]:
        return pred
    h, w = gt.shape[:2]
    return np.array(Image.fromarray(pred).resize((w, h), Image.BICUBIC))


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0

    psnr = peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0)
    ssim = structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with predicted/restored images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with GT images (same file names).")
    parser.add_argument(
        "--resize_pred_to_gt",
        action="store_true",
        help="Resize prediction to GT size before metric computation if size mismatch exists.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional CSV path to save per-image and average metrics.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

    rows = []
    psnr_list = []
    ssim_list = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"[SKIP] GT not found for {pred_path.name}")
            continue

        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)

        if pred.shape[:2] != gt.shape[:2]:
            if args.resize_pred_to_gt:
                print(
                    f"[RESIZE] {pred_path.name}: pred {pred.shape[1]}x{pred.shape[0]} "
                    f"-> gt {gt.shape[1]}x{gt.shape[0]}"
                )
                pred = resize_to_match(pred, gt)
            else:
                print(
                    f"[SKIP] Size mismatch for {pred_path.name}: "
                    f"pred={pred.shape[:2]}, gt={gt.shape[:2]}"
                )
                continue

        psnr, ssim = compute_metrics(pred, gt)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        row = {
            "file_name": pred_path.name,
            "psnr": psnr,
            "ssim": ssim,
        }
        rows.append(row)

        print(f"{pred_path.name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")

    if len(rows) == 0:
        print("No valid matched files were evaluated.")
        return

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    print("=" * 60)
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("=" * 60)

    if args.save_csv:
        save_path = Path(args.save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(
                {
                    "file_name": "AVERAGE",
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                }
            )
        print(f"Saved metrics CSV to: {save_path}")


if __name__ == "__main__":
    main()