from typing import List, Tuple
import os
import random

import numpy as np
from PIL import Image
import torch.utils.data as data


class PairedIdentityBatchTransform:
    def __call__(self, batch):
        return batch


class PairedImageDataset(data.Dataset):
    def __init__(
        self,
        lq_dir: str,
        gt_dir: str,
        mask_dir: str = "",
        out_size: int = 512,
        crop_type: str = "random",
        prompt: str = "",
        p_empty_prompt: float = 1.0,
        extensions: Tuple[str, ...] = ("png", "jpg", "jpeg", "webp"),
    ) -> None:
        super().__init__()
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.out_size = out_size
        self.crop_type = crop_type
        self.prompt = prompt
        self.p_empty_prompt = p_empty_prompt
        self.extensions = tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)
        assert crop_type in ["none", "center", "random"]

        lq_names = {
            name
            for name in os.listdir(lq_dir)
            if os.path.splitext(name)[1].lower() in self.extensions
        }
        gt_names = {
            name
            for name in os.listdir(gt_dir)
            if os.path.splitext(name)[1].lower() in self.extensions
        }
        self.names: List[str] = sorted(lq_names & gt_names)
        missing_gt = sorted(lq_names - gt_names)
        missing_lq = sorted(gt_names - lq_names)
        if not self.names:
            raise ValueError(f"No paired images found in {lq_dir} and {gt_dir}.")
        if missing_gt or missing_lq:
            raise ValueError(
                "Image pair names do not match. "
                f"Missing GT: {missing_gt[:5]}, missing LQ: {missing_lq[:5]}"
            )

    def _load_pair(self, name: str) -> Tuple[Image.Image, Image.Image, Image.Image | None]:
        lq = Image.open(os.path.join(self.lq_dir, name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")
        if lq.size != gt.size:
            raise ValueError(f"Pair {name} has different sizes: {lq.size} vs {gt.size}")
        mask = None
        if self.mask_dir:
            stem = os.path.splitext(name)[0]
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                candidate = os.path.join(self.mask_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                raise FileNotFoundError(f"Missing text mask for {name} in {self.mask_dir}")
            mask = Image.open(mask_path).convert("L")
            if mask.size != gt.size:
                raise ValueError(f"Mask {mask_path} has different size: {mask.size} vs {gt.size}")
        return lq, gt, mask

    def _resize_for_crop(
        self, lq: Image.Image, gt: Image.Image, mask: Image.Image | None = None
    ) -> Tuple[Image.Image, Image.Image, Image.Image | None]:
        if self.crop_type == "none":
            if lq.size != (self.out_size, self.out_size):
                raise ValueError(
                    f"crop_type='none' expects {self.out_size}x{self.out_size}, got {lq.size}"
                )
            return lq, gt, mask

        width, height = lq.size
        scale = self.out_size / min(width, height)
        new_size = (round(width * scale), round(height * scale))
        if new_size != lq.size:
            lq = lq.resize(new_size, resample=Image.BICUBIC)
            gt = gt.resize(new_size, resample=Image.BICUBIC)
            if mask is not None:
                mask = mask.resize(new_size, resample=Image.NEAREST)
        return lq, gt, mask

    def _crop_pair(
        self, lq: Image.Image, gt: Image.Image, mask: Image.Image | None = None
    ) -> Tuple[Image.Image, Image.Image, Image.Image | None]:
        if self.crop_type == "none":
            return lq, gt, mask

        width, height = lq.size
        if width < self.out_size or height < self.out_size:
            raise ValueError(f"Image is too small for {self.out_size} crop: {lq.size}")
        if self.crop_type == "center":
            left = (width - self.out_size) // 2
            top = (height - self.out_size) // 2
        else:
            left = random.randint(0, width - self.out_size)
            top = random.randint(0, height - self.out_size)
        box = (left, top, left + self.out_size, top + self.out_size)
        cropped_mask = mask.crop(box) if mask is not None else None
        return lq.crop(box), gt.crop(box), cropped_mask

    def __getitem__(self, index: int):
        name = self.names[index]
        lq, gt, mask = self._load_pair(name)
        lq, gt, mask = self._resize_for_crop(lq, gt, mask)
        lq, gt, mask = self._crop_pair(lq, gt, mask)

        lq = (np.array(lq) / 255.0).astype(np.float32)
        gt = (np.array(gt) / 255.0).astype(np.float32)
        prompt = "" if random.random() < self.p_empty_prompt else self.prompt

        if mask is None:
            return (gt * 2 - 1).astype(np.float32), lq, prompt

        mask = (np.array(mask) / 255.0).astype(np.float32)
        return (gt * 2 - 1).astype(np.float32), lq, prompt, mask[..., None]

    def __len__(self) -> int:
        return len(self.names)
