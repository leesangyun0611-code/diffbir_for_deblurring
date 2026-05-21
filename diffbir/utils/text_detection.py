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

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


def to_easyocr_uint8_rgb(image: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().float().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    elif isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image for EasyOCR, got shape {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image for EasyOCR, got shape {image.shape}")

    image = image.astype(np.float32, copy=False)
    min_v = float(np.nanmin(image)) if image.size else 0.0
    max_v = float(np.nanmax(image)) if image.size else 0.0
    if min_v < 0.0:
        image = (image + 1.0) * 127.5
    elif max_v <= 1.5:
        image = image * 255.0
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).round().astype(np.uint8)


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
        self.init_error = ""
        self.last_debug: Dict[str, Any] = {}
        try:
            import easyocr

            self.reader = easyocr.Reader(list(self.langs), gpu=False, recognizer=False)
            print("[TextDetection] EasyOCR backend initialized in detection-only mode (recognizer=False).")
        except ImportError:
            self.available = False
            self.init_error = "EasyOCR is not installed."
            print(
                "EasyOCR is not installed. Text mask will be empty. "
                "Install with: pip install easyocr"
            )
        except Exception as exc:
            self.available = False
            self.init_error = str(exc)
            print(f"[TextDetection] EasyOCR initialization failed: {exc}")

    def _empty_debug(self, image: np.ndarray, source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "image_minmax": [
                int(image.min()) if image.size else 0,
                int(image.max()) if image.size else 0,
            ],
            "is_hwc": bool(image.ndim == 3),
            "is_uint8": bool(image.dtype == np.uint8),
            "channel_count": int(image.shape[2]) if image.ndim == 3 else None,
            "is_rgb": True,
            "raw_detect_count": 0,
            "raw_readtext_count": 0,
            "after_conf_filter_count": 0,
            "after_area_filter_count": 0,
            "final_count": 0,
            "detector_available": bool(self.available and self.reader is not None),
            "init_error": self.init_error,
            "thresholds": self._thresholds(),
            "first_polygons": [],
        }

    def _thresholds(self) -> Dict[str, Any]:
        return {
            "text_min_confidence": self.text_min_confidence,
            "text_min_area": None,
            "easyocr_text_threshold": self.easyocr_text_threshold,
            "easyocr_low_text": self.easyocr_low_text,
            "easyocr_link_threshold": self.easyocr_link_threshold,
            "easyocr_canvas_size": self.easyocr_canvas_size,
            "easyocr_mag_ratio": self.easyocr_mag_ratio,
        }

    def _box_to_polygon(self, box: Any) -> np.ndarray | None:
        arr = np.asarray(box, dtype=np.float32)
        if arr.size == 4 and arr.ndim == 1:
            x_min, x_max, y_min, y_max = arr.tolist()
            return np.asarray(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )
        arr = arr.reshape(-1, 2) if arr.size >= 8 else arr
        if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] == 2:
            return arr.astype(np.float32)
        return None

    def _flatten_easyocr_boxes(self, boxes: Any) -> List[Any]:
        if boxes is None:
            return []
        arr = np.asarray(boxes, dtype=object)
        if arr.ndim == 1 and arr.size == 4 and all(np.isscalar(x) for x in boxes):
            return [boxes]
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 4:
            return [boxes]
        if isinstance(boxes, tuple):
            out: List[Any] = []
            for item in boxes:
                out.extend(self._flatten_easyocr_boxes(item))
            return out
        if isinstance(boxes, list):
            if not boxes:
                return []
            first = np.asarray(boxes[0], dtype=object)
            if first.ndim >= 1 and first.size in (4, 8) and not np.isscalar(boxes[0]):
                return boxes
            if len(boxes) == 1 and isinstance(boxes[0], list):
                return self._flatten_easyocr_boxes(boxes[0])
            out: List[Any] = []
            for item in boxes:
                if isinstance(item, list) and item and not (
                    np.asarray(item, dtype=object).ndim == 1 and np.asarray(item, dtype=object).size in (4, 8)
                ):
                    out.extend(self._flatten_easyocr_boxes(item))
                else:
                    out.append(item)
            return out
        return [boxes]

    def detect(self, image: np.ndarray, source: str = "stage1") -> List[Detection]:
        image = to_easyocr_uint8_rgb(image)
        debug = self._empty_debug(image, source)
        if not self.available or self.reader is None:
            print(
                f"[TextDetectionDebug] source={source} detector unavailable. "
                f"init_error={self.init_error}"
            )
            self.last_debug[source] = debug
            return []

        detect_kwargs = dict(
            text_threshold=self.easyocr_text_threshold,
            low_text=self.easyocr_low_text,
            link_threshold=self.easyocr_link_threshold,
            canvas_size=self.easyocr_canvas_size,
            mag_ratio=self.easyocr_mag_ratio,
        )
        try:
            horizontal_list, free_list = self.reader.detect(image, **detect_kwargs)
        except TypeError:
            horizontal_list, free_list = self.reader.detect(image)
        except Exception as exc:
            print(f"[TextDetectionDebug] source={source} detect() failed: {exc}")
            horizontal_list, free_list = [], []

        raw_boxes = self._flatten_easyocr_boxes(horizontal_list) + self._flatten_easyocr_boxes(free_list)
        polygons = []
        for box in raw_boxes:
            poly = self._box_to_polygon(box)
            if poly is not None:
                polygons.append(poly)

        raw_readtext_count = 0
        if getattr(self.reader, "recognizer", None) is not None:
            try:
                read_results = self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    text_threshold=self.easyocr_text_threshold,
                    low_text=self.easyocr_low_text,
                    link_threshold=self.easyocr_link_threshold,
                    canvas_size=self.easyocr_canvas_size,
                    mag_ratio=self.easyocr_mag_ratio,
                )
                raw_readtext_count = len(read_results)
            except Exception as exc:
                print(f"[TextDetectionDebug] source={source} readtext() debug failed: {exc}")

        debug["raw_detect_count"] = len(polygons)
        debug["raw_readtext_count"] = raw_readtext_count
        debug["after_conf_filter_count"] = len(polygons)

        detections: List[Detection] = []
        for polygon in polygons:
            detections.append(
                {
                    "polygon": np.asarray(polygon, dtype=np.float32),
                    "score": 1.0,
                    "source": source,
                    "text": "",
                }
            )
        debug["after_area_filter_count"] = len(detections)
        debug["final_count"] = len(detections)
        debug["first_polygons"] = [
            np.asarray(det["polygon"], dtype=np.float32).tolist()
            for det in detections[:10]
        ]
        self.last_debug[source] = debug

        print(f"[TextDetectionDebug] source={source}")
        print(f"[TextDetectionDebug] input_shape={debug['image_shape']}")
        print(f"[TextDetectionDebug] input_dtype={debug['image_dtype']}")
        print(f"[TextDetectionDebug] input_minmax={debug['image_minmax']}")
        print(f"[TextDetectionDebug] is_hwc={debug['is_hwc']} is_uint8={debug['is_uint8']} channel_count={debug['channel_count']}")
        print(f"[TextDetectionDebug] raw_detect_count={debug['raw_detect_count']}")
        print(f"[TextDetectionDebug] raw_readtext_count={debug['raw_readtext_count']}")
        print(f"[TextDetectionDebug] after_conf_filter_count={debug['after_conf_filter_count']}")
        print(f"[TextDetectionDebug] after_area_filter_count={debug['after_area_filter_count']}")
        print(f"[TextDetectionDebug] final_count={debug['final_count']}")
        print(f"[TextDetectionDebug] first_10_polygons={debug['first_polygons']}")
        return detections

    def get_debug_info(self) -> Dict[str, Any]:
        return self.last_debug


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
    valid_polys = []
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        poly[:, 0] = np.clip(poly[:, 0], 0, max(w - 1, 0))
        poly[:, 1] = np.clip(poly[:, 1], 0, max(h - 1, 0))
        if min_area > 0 and polygon_area(poly) < min_area:
            continue
        valid_polys.append(poly)
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
    detection_debug: Dict[str, Any] | None = None,
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
    save_rgb("text_detector_input_lq.png", lq_image)
    save_rgb("text_detector_input_stage1.png", stage1_image)
    save_mask("text_mask_binary.png", binary_mask > 0)
    save_mask("text_mask_soft.png", soft_mask)
    save_rgb("text_mask_overlay_lq.png", overlay(lq_image, soft_mask))
    save_rgb("text_mask_overlay_stage1.png", overlay(stage1_image, soft_mask))

    payload = {
        "sources": detection_debug or {},
        "detections": detections_to_jsonable(detections),
    }
    with open(os.path.join(debug_dir, "text_detections.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(debug_dir, "text_guidance_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
