from typing import Dict, List, Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def _matches(name: str, target_modules: List[str]) -> bool:
    if not target_modules:
        return True
    return any(target in name for target in target_modules)


def inject_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: List[str] | None = None,
    use_conv: bool = False,
    prefix: str = "",
) -> List[str]:
    target_modules = target_modules or []
    injected = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches(full_name, target_modules):
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            injected.append(full_name)
        elif use_conv and isinstance(child, nn.Conv2d) and _matches(full_name, target_modules):
            setattr(module, child_name, LoRAConv2d(child, rank, alpha, dropout))
            injected.append(full_name)
        else:
            injected.extend(
                inject_lora(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                    use_conv=use_conv,
                    prefix=full_name,
                )
            )
    return injected


def mark_only_lora_as_trainable(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def lora_parameters(module: nn.Module):
    return [param for name, param in module.named_parameters() if "lora_" in name]


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.state_dict().items()
        if "lora_" in name
    }


def load_lora_checkpoint(module: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    state = checkpoint["lora"]
    inject_lora(
        module,
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config.get("dropout", 0.0),
        target_modules=config.get("target_modules", []),
        use_conv=config.get("use_conv", False),
    )
    missing, unexpected = module.load_state_dict(state, strict=False)
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    missing_lora = [key for key in missing if "lora_" in key]
    if unexpected_lora or missing_lora:
        raise RuntimeError(
            "Failed to load LoRA checkpoint cleanly. "
            f"missing LoRA keys: {missing_lora[:10]}, "
            f"unexpected LoRA keys: {unexpected_lora[:10]}"
        )


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
