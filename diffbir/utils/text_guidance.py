from typing import Literal

import torch
from torch.nn import functional as F


def charbonnier_loss(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")

    channels = x.size(1)
    gx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    gx = gx.repeat(channels, 1, 1, 1)
    gy = gy.repeat(channels, 1, 1, 1)

    x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    edge_x = F.conv2d(x, gx, groups=channels)
    edge_y = F.conv2d(x, gy, groups=channels)
    return torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-12)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (value * mask).sum(dim=(1, 2, 3)).div(denom).sum()


def masked_rgb_loss(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    mask: torch.Tensor,
    loss_type: Literal["charbonnier", "l1", "mse"] = "charbonnier",
) -> torch.Tensor:
    diff = x_pred - x_ref
    if loss_type == "charbonnier":
        value = charbonnier_loss(diff)
    elif loss_type == "l1":
        value = diff.abs()
    elif loss_type == "mse":
        value = diff.pow(2)
    else:
        raise ValueError(f"Unsupported RGB loss type: {loss_type}")
    return _masked_mean(value, mask)


def masked_edge_loss(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_edges = sobel_edges(x_pred)
    ref_edges = sobel_edges(x_ref)
    return _masked_mean((pred_edges - ref_edges).abs(), mask)


def compute_text_guidance_loss(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    text_mask: torch.Tensor,
    rgb_weight: float = 0.1,
    edge_weight: float = 1.0,
    loss_type: Literal["charbonnier", "l1", "mse"] = "charbonnier",
) -> torch.Tensor:
    if text_mask is None:
        return x_pred.new_tensor(0.0)
    if text_mask.sum().item() <= 0:
        return x_pred.new_tensor(0.0)

    if text_mask.ndim == 3:
        text_mask = text_mask.unsqueeze(1)
    text_mask = text_mask.to(device=x_pred.device, dtype=x_pred.dtype)
    if text_mask.shape[-2:] != x_pred.shape[-2:]:
        text_mask = F.interpolate(text_mask, size=x_pred.shape[-2:], mode="bilinear", align_corners=False)
    text_mask = text_mask.clamp(0.0, 1.0)

    rgb = masked_rgb_loss(x_pred, x_ref, text_mask, loss_type=loss_type)
    edge = masked_edge_loss(x_pred, x_ref, text_mask)
    return float(rgb_weight) * rgb + float(edge_weight) * edge
