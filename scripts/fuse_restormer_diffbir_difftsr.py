import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from skimage.metrics import structural_similarity


Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Restormer, DiffBIR, and DiffTSR candidates with OCR-gated text replacement."
    )
    parser.add_argument("--input", type=Path, required=True, help="Blur input image, used for diagnostics only.")
    parser.add_argument("--restormer", type=Path, required=True, help="Restormer output R.")
    parser.add_argument("--diffbir", type=Path, required=True, help="DiffBIR output D.")
    parser.add_argument("--difftsr", type=Path, required=True, help="DiffTSR text candidate T.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth sharp image.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--langs", default="korean,en")
    parser.add_argument("--det_scales", default="1.0,2.0,3.0")
    parser.add_argument("--ocr_conf_margin", type=float, default=0.05)
    parser.add_argument("--min_conf", type=float, default=0.10)
    parser.add_argument("--min_area", type=int, default=40)
    parser.add_argument("--pad_ratio", type=float, default=0.18)
    parser.add_argument("--max_boxes", type=int, default=0)
    parser.add_argument("--text_mode", choices=["confidence", "difftsr"], default="confidence")
    parser.add_argument("--nontext_mode", choices=["conservative", "diffbir"], default="conservative")
    parser.add_argument("--nontext_max_alpha", type=float, default=0.08)
    parser.add_argument("--nontext_diff_limit", type=float, default=12.0)
    parser.add_argument("--nontext_texture_margin", type=float, default=10.0)
    parser.add_argument("--use_gpu_ocr", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips_model", default="alex", choices=["alex", "vgg"])
    return parser.parse_args()


def load_rgb(path: Path, size=None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def image_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def np_to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> List[str]:
    return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


def load_paddleocrs(langs: Sequence[str], use_gpu: bool):
    from paddleocr import PaddleOCR

    return [
        (lang, PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False))
        for lang in langs
    ]


def normalize_ocr_result(result: Any) -> List[Tuple[List[List[float]], str, float]]:
    if not result:
        return []
    if len(result) == 1 and isinstance(result[0], list):
        result = result[0]

    items = []
    for det in result:
        if not det or len(det) < 2:
            continue
        text_score = det[1]
        if not text_score or len(text_score) < 2:
            continue
        items.append((det[0], str(text_score[0]), float(text_score[1])))
    return items


def normalize_rec_result(result: Any) -> Tuple[str, float]:
    if not result:
        return "", 0.0
    candidate = result[0] if isinstance(result, list) else result
    if isinstance(candidate, list) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, tuple) and len(candidate) >= 2:
        return str(candidate[0]), float(candidate[1])
    if isinstance(candidate, list) and len(candidate) >= 2:
        return str(candidate[0]), float(candidate[1])
    return "", 0.0


def run_det(ocr: Any, image: Image.Image, scale: float, lang: str):
    if scale != 1.0:
        width, height = image.size
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BICUBIC,
        )

    arr = cv2.cvtColor(image_to_np(image), cv2.COLOR_RGB2BGR)
    detections = normalize_ocr_result(ocr.ocr(arr, cls=True))
    if scale == 1.0:
        return [(quad, text, conf, lang, scale) for quad, text, conf in detections]

    return [
        ([[pt[0] / scale, pt[1] / scale] for pt in quad], text, conf, lang, scale)
        for quad, text, conf in detections
    ]


def run_rec(ocr: Any, image: Image.Image) -> Tuple[str, float]:
    arr = cv2.cvtColor(image_to_np(image), cv2.COLOR_RGB2BGR)
    return normalize_rec_result(ocr.ocr(arr, det=False, cls=True))


def box_from_quad(quad: Sequence[Sequence[float]], width: int, height: int, pad_ratio: float) -> Box:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    )


def area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = area((x1, y1, x2, y2))
    denom = area(a) + area(b) - inter
    return float(inter / denom) if denom > 0 else 0.0


def prepare_boxes(detections, size, min_conf, min_area, pad_ratio, max_boxes):
    width, height = size
    boxes: List[Dict[str, Any]] = []
    for idx, detection in enumerate(detections):
        quad, text, conf, lang, scale = detection
        box = box_from_quad(quad, width, height, pad_ratio)
        if conf < min_conf or area(box) < min_area:
            continue
        boxes.append(
            {"id": idx, "box": box, "det_text": text, "det_conf": conf, "lang": lang, "scale": scale}
        )

    kept: List[Dict[str, Any]] = []
    for item in sorted(boxes, key=lambda row: row["det_conf"], reverse=True):
        if all(box_iou(item["box"], prev["box"]) < 0.70 for prev in kept):
            kept.append(item)
    return kept[:max_boxes] if max_boxes > 0 else kept


def draw_boxes(image: Image.Image, boxes: Sequence[Dict[str, Any]], path: Path) -> None:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        color = "lime" if item.get("use_t") else "red"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = (
            f'{item["lang"]}@{item["scale"]}: R={item["r_conf"]:.2f}, '
            f'T={item["t_conf"]:.2f}'
        )
        draw.text((x1, max(0, y1 - 14)), label, fill=color)
    out.save(path)


def build_text_mask(size, boxes: Sequence[Dict[str, Any]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for item in boxes:
        draw.rectangle(item["box"], fill=255)
    return mask


def stroke_mask_for_crop(r_crop: Image.Image, t_crop: Image.Image) -> np.ndarray:
    r_gray = cv2.cvtColor(image_to_np(r_crop), cv2.COLOR_RGB2GRAY)
    t_gray = cv2.cvtColor(image_to_np(t_crop), cv2.COLOR_RGB2GRAY)
    edges = cv2.max(cv2.Canny(r_gray, 50, 150), cv2.Canny(t_gray, 50, 150))
    diff = cv2.absdiff(r_gray, t_gray)
    _, diff_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
    mask = cv2.max(edges, diff_mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    if mask.max() == 0:
        mask[:] = 160
    return mask.astype(np.float32) / 255.0


def conservative_nontext_blend(
    r_arr: np.ndarray,
    d_arr: np.ndarray,
    text_binary: np.ndarray,
    max_alpha: float,
    diff_limit: float,
    texture_margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r_gray = cv2.cvtColor(r_arr, cv2.COLOR_RGB2GRAY)
    d_gray = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
    r_lap = np.abs(cv2.Laplacian(r_gray, cv2.CV_32F, ksize=3))
    d_lap = np.abs(cv2.Laplacian(d_gray, cv2.CV_32F, ksize=3))
    texture_gain = np.clip((d_lap - r_lap - texture_margin) / 24.0, 0.0, 1.0)

    diff = np.mean(np.abs(d_arr.astype(np.float32) - r_arr.astype(np.float32)), axis=2)
    similarity_gate = np.clip((diff_limit - diff) / max(diff_limit, 1e-6), 0.0, 1.0)
    text_gate = 1.0 - cv2.GaussianBlur(text_binary.astype(np.float32) / 255.0, (0, 0), 7.0)
    alpha = max_alpha * texture_gain * similarity_gate * text_gate
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
    base = r_arr.astype(np.float32) * (1.0 - alpha[..., None]) + d_arr.astype(np.float32) * alpha[..., None]
    return base, alpha


def calc_metrics(pred: np.ndarray, gt: np.ndarray, device: str, lpips_model: str) -> Dict[str, float]:
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mse = np.mean((pred_f - gt_f) ** 2)
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))

    metric = lpips.LPIPS(net=lpips_model).to(device)
    metric.eval()
    pred_t = torch.from_numpy(pred.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_score = float(metric(pred_t, gt_t).item())
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_score}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    r_img = load_rgb(args.restormer)
    d_img = load_rgb(args.diffbir, r_img.size)
    t_img = load_rgb(args.difftsr, r_img.size)
    gt_img = load_rgb(args.gt, r_img.size)
    input_img = load_rgb(args.input, r_img.size)

    langs = parse_str_list(args.langs)
    scales = parse_float_list(args.det_scales)
    ocrs = load_paddleocrs(langs, args.use_gpu_ocr)

    detections = []
    for lang, ocr in ocrs:
        for scale in scales:
            detections.extend(run_det(ocr, r_img, scale, lang))
    boxes = prepare_boxes(detections, r_img.size, args.min_conf, args.min_area, args.pad_ratio, args.max_boxes)

    text_binary_img = build_text_mask(r_img.size, boxes)
    text_binary = np.asarray(text_binary_img, dtype=np.uint8)
    r_arr = image_to_np(r_img)
    d_arr = image_to_np(d_img)
    t_arr = image_to_np(t_img)
    if args.nontext_mode == "diffbir":
        base = d_arr.astype(np.float32)
        text_soft = cv2.GaussianBlur(text_binary.astype(np.float32) / 255.0, (0, 0), 7.0)
        nontext_alpha = 1.0 - text_soft
    else:
        base, nontext_alpha = conservative_nontext_blend(
            r_arr,
            d_arr,
            text_binary,
            args.nontext_max_alpha,
            args.nontext_diff_limit,
            args.nontext_texture_margin,
        )

    rows = []
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        r_crop = r_img.crop((x1, y1, x2, y2))
        t_crop = t_img.crop((x1, y1, x2, y2))
        rec_ocr = next(ocr for lang, ocr in ocrs if lang == item["lang"])
        r_text, r_conf = run_rec(rec_ocr, r_crop)
        t_text, t_conf = run_rec(rec_ocr, t_crop)
        use_t = args.text_mode == "difftsr" or t_conf > r_conf + args.ocr_conf_margin
        item.update({"r_text": r_text, "r_conf": r_conf, "t_text": t_text, "t_conf": t_conf, "use_t": use_t})

        if use_t or args.nontext_mode == "diffbir":
            if use_t:
                source_crop = t_crop
                source_arr = t_arr
                mask = stroke_mask_for_crop(r_crop, t_crop)
            else:
                source_crop = r_crop
                source_arr = r_arr
                base_crop = np_to_image(base[y1:y2, x1:x2, :])
                mask = stroke_mask_for_crop(source_crop, base_crop)
            crop_base = base[y1:y2, x1:x2, :]
            crop_source = source_arr[y1:y2, x1:x2, :].astype(np.float32)
            base[y1:y2, x1:x2, :] = crop_base * (1.0 - mask[..., None]) + crop_source * mask[..., None]

        rows.append(
            {
                "box": item["box"],
                "lang": item["lang"],
                "scale": item["scale"],
                "det_text": item["det_text"],
                "det_conf": item["det_conf"],
                "r_text": r_text,
                "r_conf": r_conf,
                "t_text": t_text,
                "t_conf": t_conf,
                "use_t": use_t,
            }
        )

    fused = np.clip(base, 0, 255).astype(np.uint8)
    out_path = args.output_dir / "000652_fused.png"
    np_to_image(fused).save(out_path)
    text_binary_img.save(args.output_dir / "000652_text_mask_binary.png")
    Image.fromarray(np.clip(nontext_alpha * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        args.output_dir / "000652_nontext_diffbir_alpha.png"
    )
    draw_boxes(input_img, boxes, args.output_dir / "000652_ocr_boxes.png")

    metrics = calc_metrics(fused, image_to_np(gt_img), args.device, args.lpips_model)
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerow({"file_name": out_path.name, **metrics})

    summary = {
        "input": str(args.input),
        "restormer": str(args.restormer),
        "diffbir": str(args.diffbir),
        "difftsr": str(args.difftsr),
        "gt": str(args.gt),
        "output": str(out_path),
        "langs": langs,
        "det_scales": scales,
        "ocr_conf_margin": args.ocr_conf_margin,
        "text_mode": args.text_mode,
        "nontext_mode": args.nontext_mode,
        "num_boxes": len(boxes),
        "num_t_replacements": sum(1 for item in boxes if item.get("use_t")),
        "mean_nontext_diffbir_alpha": float(nontext_alpha.mean()),
        "max_nontext_diffbir_alpha": float(nontext_alpha.max()),
        "boxes": rows,
        "metrics": metrics,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
