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


class RestormerFromClone(nn.Module):
    def __init__(self, repo_dir: str, task: str, ckpt_path: str):
        super().__init__()

        repo_dir = Path(repo_dir)
        arch_path = repo_dir / "basicsr" / "models" / "archs" / "restormer_arch.py"
        if not arch_path.exists():
            raise FileNotFoundError(f"Restormer arch not found: {arch_path}")

        if not ckpt_path:
            raise ValueError("restormer_ckpt is empty")

        loaded = run_path(str(arch_path))
        Restormer = loaded["Restormer"]

        params = {
            "inp_channels": 3,
            "out_channels": 3,
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "LayerNorm_type": "WithBias",
            "dual_pixel_task": False,
        }

        if task in ["Real_Denoising", "Gaussian_Color_Denoising"]:
            params["LayerNorm_type"] = "BiasFree"
        elif task == "Gaussian_Gray_Denoising":
            params["inp_channels"] = 1
            params["out_channels"] = 1
            params["LayerNorm_type"] = "BiasFree"

        self.model = Restormer(**params)

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