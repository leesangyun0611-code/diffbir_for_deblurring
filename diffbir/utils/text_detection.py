import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


Detection = Dict[str, Any]


class BaseTextDetector:
    def detect(self, image: np.ndarray, source: str = "stage1") -> List[Detection]:
        raise NotImplementedError


@dataclass
class EasyOCRTextDetector(BaseTextDetector):
    langs: Sequence[str] = ("ko", "en")
    text_min_confidence: float = 0.3
    easyocr_text_threshold: float = 0.4
    easyocr_low_text: float = 0.2
    easyocr_link_threshold: float = 0.4
    easyocr_canvas_size: int = 2560
    easyocr_mag_ratio: float = 2.0

    def __post_init__(self) -> None:
        self.reader = None
        self.available = True
        try:
            import easyocr

            self.reader = easyocr.Reader(list(self.langs), gpu=False)
        except ImportError:
            self.available = False
            print(
                "EasyOCR is not installed. Text mask will be empty. "
                "Install with: pip install easyocr"
            )
        except Exception as exc:
            self.available = False
            print(f"[TextDetection] EasyOCR initialization failed: {exc}")

    def detect(self, image: np.ndarray, source: str = "stage1") -> List[Detection]:
        if not self.available or self.reader is None:
            return []

        kwargs = dict(
            detail=1,
            paragraph=False,
            text_threshold=self.easyocr_text_threshold,
            low_text=self.easyocr_low_text,
            link_threshold=self.easyocr_link_threshold,
            canvas_size=self.easyocr_canvas_size,
            mag_ratio=self.easyocr_mag_ratio,
        )
        try:
            results = self.reader.readtext(image, **kwargs)
        except TypeError:
            results = self.reader.readtext(image, detail=1, paragraph=False)

        detections: List[Detection] = []
        for item in results:
            if len(item) < 3:
                continue
            polygon, text, score = item[0], item[1], float(item[2])
            if score < self.text_min_confidence:
                continue
            detections.append(
                {
                    "polygon": np.asarray(polygon, dtype=np.float32),
                    "score": score,
                    "source": source,
                    "text": text,
                }
            )
        return detections


class EmptyTextDetector(BaseTextDetector):
    def detect(self, image: np.ndarray, source: str = "stage1") -> List[Detection]:
        return []


def parse_langs(langs: str | Sequence[str]) -> List[str]:
    if isinstance(langs, str):
        return [x.strip() for x in langs.split(",") if x.strip()]
    return list(langs)


def build_text_detector(name: str, **kwargs: Any) -> BaseTextDetector:
    name = (name or "none").lower()
    if name == "easyocr":
        return EasyOCRTextDetector(**kwargs)
    if name in {"none", "empty"}:
        return EmptyTextDetector()
    raise ValueError(f"Unsupported text detector backend: {name}")


def _fill_polygons_with_pil(
    polygons: Sequence[np.ndarray],
    image_shape: Tuple[int, int] | Tuple[int, int, int],
) -> np.ndarray:
    h, w = image_shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        pts = [(float(x), float(y)) for x, y in np.asarray(poly).reshape(-1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygons_to_mask(
    polygons: Sequence[np.ndarray],
    image_shape: Tuple[int, int] | Tuple[int, int, int],
    min_area: float = 0,
) -> np.ndarray:
    h, w = image_shape[:2]
    valid_polys = [
        np.asarray(poly, dtype=np.float32)
        for poly in polygons
        if polygon_area(np.asarray(poly)) >= min_area
    ]
    if not valid_polys:
        return np.zeros((h, w), dtype=np.float32)

    try:
        import cv2

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [p.astype(np.int32) for p in valid_polys], 255)
        return mask.astype(np.float32) / 255.0
    except ImportError:
        return _fill_polygons_with_pil(valid_polys, image_shape)


def postprocess_text_mask(mask: np.ndarray, dilate: int = 5, blur: float = 1.0) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    try:
        import cv2

        if dilate and dilate > 0:
            kernel = np.ones((int(dilate), int(dilate)), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        if blur and blur > 0:
            k = max(3, int(round(float(blur) * 4)) | 1)
            mask = cv2.GaussianBlur(mask, (k, k), float(blur))
    except ImportError:
        pass
    return mask.astype(np.float32).clip(0.0, 1.0)


def detections_to_jsonable(detections: Sequence[Detection]) -> List[Detection]:
    out: List[Detection] = []
    for det in detections:
        item = dict(det)
        if "polygon" in item:
            item["polygon"] = np.asarray(item["polygon"]).tolist()
        out.append(item)
    return out


def save_text_mask_debug(
    debug_dir: str,
    lq_image: np.ndarray,
    stage1_image: np.ndarray,
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
    detections: Sequence[Detection],
    config: Dict[str, Any],
) -> None:
    os.makedirs(debug_dir, exist_ok=True)

    def save_rgb(name: str, image: np.ndarray) -> None:
        Image.fromarray(np.asarray(image).clip(0, 255).astype(np.uint8)).save(
            os.path.join(debug_dir, name)
        )

    def save_mask(name: str, mask: np.ndarray) -> None:
        Image.fromarray((np.asarray(mask).clip(0, 1) * 255).astype(np.uint8)).save(
            os.path.join(debug_dir, name)
        )

    def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = np.asarray(image).clip(0, 255).astype(np.float32)
        m = np.asarray(mask).clip(0, 1)[..., None]
        color = np.zeros_like(img)
        color[..., 0] = 255.0
        return (img * (1.0 - 0.35 * m) + color * (0.35 * m)).clip(0, 255).astype(np.uint8)

    save_rgb("input_lq.png", lq_image)
    save_rgb("stage1_for_text_detection.png", stage1_image)
    save_mask("text_mask_binary.png", binary_mask > 0)
    save_mask("text_mask_soft.png", soft_mask)
    save_rgb("text_mask_overlay_lq.png", overlay(lq_image, soft_mask))
    save_rgb("text_mask_overlay_stage1.png", overlay(stage1_image, soft_mask))

    with open(os.path.join(debug_dir, "text_detections.json"), "w", encoding="utf-8") as f:
        json.dump(detections_to_jsonable(detections), f, ensure_ascii=False, indent=2)
    with open(os.path.join(debug_dir, "text_guidance_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
