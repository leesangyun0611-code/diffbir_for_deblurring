from pathlib import Path
from runpy import run_path

import torch
import torch.nn as nn


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class MPRNetFromClone(nn.Module):
    def __init__(self, repo_dir: str, ckpt_path: str):
        super().__init__()

        repo_dir = Path(repo_dir)
        arch_path = repo_dir / "Deblurring" / "MPRNet.py"
        if not arch_path.exists():
            raise FileNotFoundError(f"MPRNet arch not found: {arch_path}")

        if not ckpt_path:
            raise ValueError("mprnet_ckpt is empty")

        loaded = run_path(str(arch_path))
        MPRNet = loaded["MPRNet"]

        self.model = MPRNet()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt)
        else:
            state = ckpt

        state = _strip_module_prefix(state)
        self.model.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out
