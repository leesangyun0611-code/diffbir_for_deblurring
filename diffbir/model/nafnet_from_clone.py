from pathlib import Path
from runpy import run_path
import sys

import torch
import torch.nn as nn


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class NAFNetFromClone(nn.Module):
    def __init__(self, repo_dir: str, ckpt_path: str):
        super().__init__()

        repo_dir = Path(repo_dir)
        arch_path = repo_dir / "basicsr" / "models" / "archs" / "NAFNet_arch.py"
        if not arch_path.exists():
            raise FileNotFoundError(f"NAFNet arch not found: {arch_path}")

        if not ckpt_path:
            raise ValueError("nafnet_ckpt is empty")

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        loaded = run_path(str(arch_path))
        NAFNetLocal = loaded["NAFNetLocal"]

        # GoPro width64 config from options/test/GoPro/NAFNet-width64.yml
        self.model = NAFNetLocal(
            img_channel=3,
            width=64,
            enc_blk_nums=[1, 1, 1, 28],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1, 1],
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "params" in ckpt:
                state = ckpt["params"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        state = _strip_module_prefix(state)
        self.model.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
