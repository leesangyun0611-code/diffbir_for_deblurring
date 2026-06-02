import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from diffbir.utils.text_detection import (
    build_text_detector,
    polygons_to_mask,
    postprocess_text_mask,
    to_easyocr_uint8_rgb,
)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute OCR detection masks for paired training.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--detector", default="easyocr", choices=["easyocr", "none"])
    parser.add_argument("--easyocr_langs", default="ko,en")
    parser.add_argument("--easyocr_text_threshold", type=float, default=0.2)
    parser.add_argument("--easyocr_low_text", type=float, default=0.1)
    parser.add_argument("--easyocr_link_threshold", type=float, default=0.2)
    parser.add_argument("--easyocr_canvas_size", type=int, default=3200)
    parser.add_argument("--easyocr_mag_ratio", type=float, default=3.0)
    parser.add_argument("--text_min_confidence", type=float, default=0.3)
    parser.add_argument("--text_min_area", type=float, default=8)
    parser.add_argument("--mask_dilate", type=int, default=2)
    parser.add_argument("--mask_blur", type=float, default=1.5)
    parser.add_argument("--save_binary", action="store_true")
    return parser.parse_args()


def image_files(path: Path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary_dir = args.output_dir / "binary"
    if args.save_binary:
        binary_dir.mkdir(parents=True, exist_ok=True)

    detector = build_text_detector(
        args.detector,
        langs=[x.strip() for x in args.easyocr_langs.split(",") if x.strip()],
        text_min_confidence=args.text_min_confidence,
        easyocr_text_threshold=args.easyocr_text_threshold,
        easyocr_low_text=args.easyocr_low_text,
        easyocr_link_threshold=args.easyocr_link_threshold,
        easyocr_canvas_size=args.easyocr_canvas_size,
        easyocr_mag_ratio=args.easyocr_mag_ratio,
    )

    files = image_files(args.input_dir)
    for idx, path in enumerate(files, start=1):
        image = Image.open(path).convert("RGB")
        image_np = to_easyocr_uint8_rgb(image)
        detections = detector.detect(image_np, source="gt")
        binary = polygons_to_mask(
            [det["polygon"] for det in detections],
            image_np.shape,
            min_area=args.text_min_area,
        )
        soft = postprocess_text_mask(binary, dilate=args.mask_dilate, blur=args.mask_blur)
        out = (np.asarray(soft).clip(0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(out).save(args.output_dir / f"{path.stem}.png")
        if args.save_binary:
            binary_out = (np.asarray(binary).clip(0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(binary_out).save(binary_dir / f"{path.stem}.png")
        print(f"[precompute-text-mask] {idx}/{len(files)} {path.name}: boxes={len(detections)}")


if __name__ == "__main__":
    main()
