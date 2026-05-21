import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend OCR-detected text regions from stage1 into stage2 outputs."
    )
    parser.add_argument("--stage1_dir", type=Path, required=True)
    parser.add_argument("--stage2_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--det_source", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--lang", default="korean")
    parser.add_argument(
        "--langs",
        default="",
        help="Comma-separated PaddleOCR languages. Overrides --lang, e.g. korean,en.",
    )
    parser.add_argument("--ocr_scale", type=float, default=3.0)
    parser.add_argument("--use_gpu_ocr", action="store_true")
    parser.add_argument("--min_conf", type=float, default=0.10)
    parser.add_argument("--min_area", type=int, default=40)
    parser.add_argument("--pad_ratio", type=float, default=0.20)
    parser.add_argument("--feather", type=int, default=5)
    parser.add_argument("--max_boxes", type=int, default=0, help="0 means all boxes.")
    parser.add_argument("--save_diagnostics", action="store_true")
    parser.add_argument(
        "--diagnostics_dir",
        type=Path,
        default=None,
        help="Directory for text detection boxes/masks. Defaults to output_dir/text_detection.",
    )
    return parser.parse_args()


def parse_langs(args: argparse.Namespace) -> List[str]:
    raw = args.langs if args.langs else args.lang
    langs = [item.strip() for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(langs))


def load_paddleocr(lang: str, use_gpu: bool):
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)


def load_paddleocrs(langs: List[str], use_gpu: bool):
    return [(lang, load_paddleocr(lang, use_gpu)) for lang in langs]


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


def run_ocr(ocr: Any, image: Image.Image, scale: float, lang: str = ""):
    if scale != 1.0:
        w, h = image.size
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BICUBIC,
        )

    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    detections = normalize_ocr_result(ocr.ocr(arr, cls=True))
    if scale == 1.0:
        return [(quad, text, conf, lang) for quad, text, conf in detections]

    return [
        ([[pt[0] / scale, pt[1] / scale] for pt in quad], text, conf, lang)
        for quad, text, conf in detections
    ]


def run_multi_ocr(ocrs, image: Image.Image, scale: float):
    detections = []
    for lang, ocr in ocrs:
        detections.extend(run_ocr(ocr, image, scale, lang=lang))
    return detections


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
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = area((x1, y1, x2, y2))
    denom = area(a) + area(b) - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_boxes(boxes: List[Dict[str, Any]], iou_thresh: float = 0.70) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for item in sorted(boxes, key=lambda row: row["conf"], reverse=True):
        if all(box_iou(item["box"], prev["box"]) < iou_thresh for prev in kept):
            kept.append(item)
    return kept


def prepare_boxes(detections, size, min_conf, min_area, pad_ratio, max_boxes):
    width, height = size
    boxes: List[Dict[str, Any]] = []
    for idx, detection in enumerate(detections):
        if len(detection) == 4:
            quad, text, conf, lang = detection
        else:
            quad, text, conf = detection
            lang = ""
        box = box_from_quad(quad, width, height, pad_ratio)
        if conf < min_conf or area(box) < min_area:
            continue
        boxes.append({"id": idx, "box": box, "text": text, "conf": conf, "lang": lang})

    boxes = nms_boxes(boxes)
    boxes.sort(key=lambda item: item["conf"], reverse=True)
    return boxes[:max_boxes] if max_boxes > 0 else boxes


def build_binary_mask(size, boxes) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for item in boxes:
        draw.rectangle(item["box"], fill=255)
    return mask


def build_mask(size, boxes, feather: int) -> Image.Image:
    mask = build_binary_mask(size, boxes)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def draw_boxes(image: Image.Image, boxes, path: Path) -> None:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        lang = f'[{item["lang"]}] ' if item.get("lang") else ""
        draw.text((x1, max(0, y1 - 12)), f'{lang}{item["text"]} {item["conf"]:.2f}', fill="red")
    out.save(path)


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def find_stage1_path(stage1_dir: Path, stage2_file: Path):
    direct = stage1_dir / stage2_file.name
    if direct.exists():
        return direct
    stem = stage2_file.stem.rsplit("_", 1)[0] if "_" in stage2_file.stem else stage2_file.stem
    for ext in [stage2_file.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        candidate = stage1_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resize_to(image: Image.Image, size) -> Image.Image:
    return image if image.size == size else image.resize(size, Image.Resampling.BICUBIC)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or (args.output_dir / "text_detection")
    if args.save_diagnostics:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    langs = parse_langs(args)
    ocrs = load_paddleocrs(langs, args.use_gpu_ocr)
    rows = []
    skipped = []

    for stage2_path in image_files(args.stage2_dir):
        stage1_path = find_stage1_path(args.stage1_dir, stage2_path)
        if stage1_path is None:
            skipped.append({"file_name": stage2_path.name, "reason": "matching stage1 image not found"})
            continue

        stage2 = Image.open(stage2_path).convert("RGB")
        stage1 = resize_to(Image.open(stage1_path).convert("RGB"), stage2.size)
        det_image = stage1 if args.det_source == "stage1" else stage2
        boxes = prepare_boxes(
            run_multi_ocr(ocrs, det_image, args.ocr_scale),
            stage2.size,
            args.min_conf,
            args.min_area,
            args.pad_ratio,
            args.max_boxes,
        )
        binary_mask = build_binary_mask(stage2.size, boxes)
        mask = binary_mask.filter(ImageFilter.GaussianBlur(radius=args.feather)) if args.feather > 0 else binary_mask
        out = Image.composite(stage1, stage2, mask)
        out_path = args.output_dir / stage2_path.name
        out.save(out_path)

        if args.save_diagnostics:
            binary_mask.save(diagnostics_dir / f"{stage2_path.stem}_text_mask_binary.png")
            mask.save(diagnostics_dir / f"{stage2_path.stem}_text_mask_soft.png")
            draw_boxes(det_image, boxes, diagnostics_dir / f"{stage2_path.stem}_text_boxes.png")

        row = {
            "file_name": stage2_path.name,
            "stage1_path": str(stage1_path),
            "stage2_path": str(stage2_path),
            "output_path": str(out_path),
            "num_boxes": len(boxes),
            "mean_conf": float(np.mean([b["conf"] for b in boxes])) if boxes else 0.0,
            "texts": " | ".join(b["text"] for b in boxes),
            "boxes_json": json.dumps(boxes, ensure_ascii=False),
        }
        rows.append(row)
        print(f"[text-blend] {stage2_path.name}: boxes={row['num_boxes']} -> {out_path}")

    with open(args.output_dir / "text_blend_summary.csv", "w", newline="") as f:
        fieldnames = ["file_name", "stage1_path", "stage2_path", "output_path", "num_boxes", "mean_conf", "texts", "boxes_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage1_dir": str(args.stage1_dir),
        "stage2_dir": str(args.stage2_dir),
        "output_dir": str(args.output_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "langs": langs,
        "num_processed": len(rows),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }
    with open(args.output_dir / "text_blend_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not rows:
        raise SystemExit("No images were blended. Check matching file names in stage1_dir and stage2_dir.")


if __name__ == "__main__":
    main()
