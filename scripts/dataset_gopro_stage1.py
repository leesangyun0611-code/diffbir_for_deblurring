import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


class GoProStage1Dataset(Dataset):
    def __init__(
        self,
        lq_dir: str,
        hq_dir: str,
        patch_size: int = 256,
        training: bool = True,
    ):
        self.lq_dir = Path(lq_dir)
        self.hq_dir = Path(hq_dir)
        self.patch_size = patch_size
        self.training = training

        if not self.lq_dir.exists():
            raise FileNotFoundError(f"LQ dir not found: {self.lq_dir}")
        if not self.hq_dir.exists():
            raise FileNotFoundError(f"HQ dir not found: {self.hq_dir}")

        self.pairs = []
        for lq_path in sorted(self.lq_dir.iterdir()):
            if lq_path.suffix.lower() not in IMG_EXTS:
                continue
            hq_path = self.hq_dir / lq_path.name
            if hq_path.exists():
                self.pairs.append((lq_path, hq_path))

        if not self.pairs:
            raise ValueError("No matched LQ-HQ pairs found.")

        print(f"[Dataset] matched pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, lq: Image.Image, hq: Image.Image):
        w, h = lq.size
        ps = self.patch_size

        if w < ps or h < ps:
            new_w = max(w, ps)
            new_h = max(h, ps)
            lq = lq.resize((new_w, new_h), Image.BICUBIC)
            hq = hq.resize((new_w, new_h), Image.BICUBIC)
            w, h = lq.size

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        lq = lq.crop((x, y, x + ps, y + ps))
        hq = hq.crop((x, y, x + ps, y + ps))
        return lq, hq

    def _augment(self, lq: Image.Image, hq: Image.Image):
        if random.random() < 0.5:
            lq = lq.transpose(Image.FLIP_LEFT_RIGHT)
            hq = hq.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            lq = lq.transpose(Image.FLIP_TOP_BOTTOM)
            hq = hq.transpose(Image.FLIP_TOP_BOTTOM)

        rot_k = random.randint(0, 3)
        if rot_k > 0:
            angle = 90 * rot_k
            lq = lq.rotate(angle)
            hq = hq.rotate(angle)

        return lq, hq

    def __getitem__(self, idx: int):
        lq_path, hq_path = self.pairs[idx]
        lq = Image.open(lq_path).convert("RGB")
        hq = Image.open(hq_path).convert("RGB")

        if self.training:
            lq, hq = self._random_crop(lq, hq)
            lq, hq = self._augment(lq, hq)

        lq = pil_to_tensor(lq)
        hq = pil_to_tensor(hq)

        return {
            "lq": lq,
            "hq": hq,
            "name": lq_path.name,
        }