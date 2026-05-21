import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from diffbir.utils.text_detection import (
    build_text_detector,
    detections_to_jsonable,
    polygons_to_mask,
    postprocess_text_mask,
    save_text_mask_debug,
    to_easyocr_uint8_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug text localization without running DiffBIR inference.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector", type=str, default="easyocr", choices=["easyocr", "none"])
    parser.add_argument("--langs", type=str, default="ko,en")
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--low_text", type=float, default=0.1)
    parser.add_argument("--link_threshold", type=float, default=0.2)
    parser.add_argument("--canvas_size", type=int, default=3200)
    parser.add_argument("--mag_ratio", type=float, default=3.0)
    parser.add_argument("--text_min_confidence", type=float, default=0.1)
    parser.add_argument("--text_min_area", type=float, default=0)
    parser.add_argument("--mask_dilate", type=int, default=5)
    parser.add_argument("--mask_blur", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="debug_text_detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    detector_input = to_easyocr_uint8_rgb(image)
    Image.fromarray(detector_input).save(output_dir / "detector_input.png")

    detector = build_text_detector(
        args.detector,
        langs=[x.strip() for x in args.langs.split(",") if x.strip()],
        text_min_confidence=args.text_min_confidence,
        easyocr_text_threshold=args.text_threshold,
        easyocr_low_text=args.low_text,
        easyocr_link_threshold=args.link_threshold,
        easyocr_canvas_size=args.canvas_size,
        easyocr_mag_ratio=args.mag_ratio,
    )
    detections = detector.detect(detector_input, source="debug")
    polygons = [det["polygon"] for det in detections]
    binary_mask = polygons_to_mask(polygons, detector_input.shape, min_area=args.text_min_area)
    soft_mask = postprocess_text_mask(binary_mask, dilate=args.mask_dilate, blur=args.mask_blur)
    mask_nonzero_ratio = float((soft_mask > 0).mean())

    debug_info = detector.get_debug_info()
    debug_source = debug_info.get("debug", {})
    print(f"raw_detect_count={debug_source.get('raw_detect_count', 0)}")
    print(f"raw_readtext_count={debug_source.get('raw_readtext_count', 0)}")
    print(f"final_count={len(detections)}")
    print(f"mask_nonzero_ratio={mask_nonzero_ratio:.6f}")

    save_text_mask_debug(
        str(output_dir),
        detector_input,
        detector_input,
        binary_mask,
        soft_mask,
        detections,
        {
            "image": args.image,
            "detector": args.detector,
            "langs": args.langs,
            "text_min_confidence": args.text_min_confidence,
            "text_min_area": args.text_min_area,
            "easyocr_text_threshold": args.text_threshold,
            "easyocr_low_text": args.low_text,
            "easyocr_link_threshold": args.link_threshold,
            "easyocr_canvas_size": args.canvas_size,
            "easyocr_mag_ratio": args.mag_ratio,
            "mask_nonzero_ratio": mask_nonzero_ratio,
        },
        detection_debug=debug_info,
    )

    os.replace(output_dir / "text_detector_input_lq.png", output_dir / "detector_input.png")
    os.replace(output_dir / "text_mask_binary.png", output_dir / "mask_binary.png")
    os.replace(output_dir / "text_mask_soft.png", output_dir / "mask_soft.png")
    os.replace(output_dir / "text_mask_overlay_lq.png", output_dir / "overlay.png")

    with open(output_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": debug_info,
                "detections": detections_to_jsonable(detections),
                "mask_nonzero_ratio": mask_nonzero_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
