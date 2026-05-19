from typing import overload, Tuple
import os
import time

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

from .sampler import (
    SpacedSampler,
    DDIMSampler,
    DPMSolverSampler,
    EDMSampler,
)
from .utils.cond_fn import Guidance
from .utils.common import (
    wavelet_reconstruction,
    trace_vram_usage,
    make_tiled_fn,
    VRAMPeakMonitor,
)
from .utils.text_detection import (
    build_text_detector,
    polygons_to_mask,
    postprocess_text_mask,
    save_text_mask_debug,
)
from .model import ControlLDM, Diffusion, RRDBNet


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        out_h, out_w = size, size
    elif h < w:
        out_h, out_w = size, int(w * (size / h))
    else:
        out_h, out_w = int(h * (size / w)), size

    return F.interpolate(imgs, size=(out_h, out_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def tock(t, name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[TIME] {name}: {time.perf_counter() - t:.4f} sec")


class Pipeline:

    def __init__(
        self,
        cleaner: nn.Module,
        cldm: ControlLDM,
        diffusion: Diffusion,
        cond_fn: Guidance | None,
        device: str,
    ) -> None:
        self.cleaner = cleaner
        self.cldm = cldm
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.device = device
        self.output_size: Tuple[int, int] = None
        self.text_args = None
        self.text_detector = None
        self.text_debug_stem = None

    def configure_text_guidance(self, args) -> None:
        self.text_args = args
        if not getattr(args, "text_guidance", False):
            return
        self.text_detector = build_text_detector(
            args.text_detector,
            langs=[x.strip() for x in args.easyocr_langs.split(",") if x.strip()],
            text_min_confidence=args.text_min_confidence,
            easyocr_text_threshold=args.easyocr_text_threshold,
            easyocr_low_text=args.easyocr_low_text,
            easyocr_link_threshold=args.easyocr_link_threshold,
            easyocr_canvas_size=args.easyocr_canvas_size,
            easyocr_mag_ratio=args.easyocr_mag_ratio,
        )
        print(f"[TextGuidance] detector backend: {args.text_detector}")
        print(f"[TextGuidance] selected Stage 1 model: {getattr(args, 'stage1_model', 'default')}")
        print(f"[TextGuidance] Stage 1 checkpoint: {getattr(args, 'stage1_ckpt', '')}")
        print(f"[TextGuidance] Stage 1 config: {getattr(args, 'stage1_config', '')}")

    def set_output_size(self, lq_size: Tuple[int]) -> None:
        h, w = lq_size[2:]
        self.output_size = (h, w)

    @overload
    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        ...

    def apply_cldm(
        self,
        cond_img: torch.Tensor,
        steps: int,
        strength: float,
        vae_encoder_tiled: bool,
        vae_encoder_tile_size: int,
        vae_decoder_tiled: bool,
        vae_decoder_tile_size: int,
        cldm_tiled: bool,
        cldm_tile_size: int,
        cldm_tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        start_point_type: str,
        sampler_type: str,
        noise_aug: int,
        rescale_cfg: bool,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> torch.Tensor:
        bs, _, h0, w0 = cond_img.shape

        if not vae_encoder_tiled and not cldm_tiled:
            cond_img = pad_to_multiples_of(cond_img, multiple=64)
        else:
            cond_img = pad_to_multiples_of(cond_img, multiple=8)

        if vae_encoder_tiled and (
            cond_img.size(2) < vae_encoder_tile_size
            or cond_img.size(3) < vae_encoder_tile_size
        ):
            print("[VAE Encoder]: the input size is tiny and unnecessary to tile.")
            vae_encoder_tiled = False

        if vae_encoder_tiled:
            if vae_encoder_tile_size % 8 != 0:
                raise ValueError("VAE encoder tile size must be a multiple of 8")

        with VRAMPeakMonitor("encoding condition image"):
            cond = self.cldm.prepare_condition(
                cond_img,
                [pos_prompt] * bs,
                vae_encoder_tiled,
                vae_encoder_tile_size,
            )
            uncond = self.cldm.prepare_condition(
                cond_img,
                [neg_prompt] * bs,
                vae_encoder_tiled,
                vae_encoder_tile_size,
            )

        h1, w1 = cond["c_img"].shape[2:]
        guidance_latent_target = cond["c_img"].detach().clone()

        if cldm_tiled and (h1 < cldm_tile_size // 8 or w1 < cldm_tile_size // 8):
            print("[Diffusion]: the input size is tiny and unnecessary to tile.")
            cldm_tiled = False

        if not cldm_tiled:
            cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
            uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)
        else:
            if cldm_tile_size % 64 != 0:
                raise ValueError("Diffusion tile size must be a multiple of 64")

        h2, w2 = cond["c_img"].shape[2:]

        if start_point_type == "cond":
            x_0 = cond["c_img"]
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full(
                    (bs,),
                    self.diffusion.num_timesteps - 1,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device),
            )
        else:
            x_T = torch.randn((bs, 4, h2, w2), dtype=torch.float32, device=self.device)

        if noise_aug > 0:
            cond["c_img"] = self.diffusion.q_sample(
                x_start=cond["c_img"],
                t=torch.full(size=(bs,), fill_value=noise_aug, device=self.device),
                noise=torch.randn_like(cond["c_img"]),
            )
            uncond["c_img"] = cond["c_img"].detach().clone()

        if self.cond_fn:
            self.cond_fn.load_target(cond_img * 2 - 1)
            self.cond_fn.target_latent = guidance_latent_target

        control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13

        betas = self.diffusion.betas
        parameterization = self.diffusion.parameterization
        if sampler_type == "spaced":
            sampler = SpacedSampler(betas, parameterization, rescale_cfg)
        elif sampler_type == "ddim":
            sampler = DDIMSampler(betas, parameterization, rescale_cfg, eta=0)
        elif sampler_type.startswith("dpm"):
            sampler = DPMSolverSampler(
                betas, parameterization, rescale_cfg, sampler_type
            )
        elif sampler_type.startswith("edm"):
            sampler = EDMSampler(
                betas,
                parameterization,
                rescale_cfg,
                sampler_type,
                s_churn,
                s_tmin,
                s_tmax,
                s_noise,
                eta,
                order,
            )
        else:
            raise NotImplementedError(sampler_type)

        with VRAMPeakMonitor("sampling"):
            sample_kwargs = dict(
                model=self.cldm,
                device=self.device,
                steps=steps,
                x_size=(bs, 4, h2, w2),
                cond=cond,
                uncond=uncond,
                cfg_scale=cfg_scale,
                tiled=cldm_tiled,
                tile_size=cldm_tile_size // 8,
                tile_stride=cldm_tile_stride // 8,
                x_T=x_T,
                progress=True,
            )

            if sampler_type == "spaced":
                sample_kwargs.update(
                    cond_fn=self.cond_fn,
                    guidance_decoder_tiled=vae_decoder_tiled,
                    guidance_decoder_tile_size=vae_decoder_tile_size // 8,
                )

            z = sampler.sample(**sample_kwargs)
            z = z[..., :h1, :w1]

        if vae_decoder_tiled and (
            h1 < vae_decoder_tile_size // 8 or w1 < vae_decoder_tile_size // 8
        ):
            print("[VAE Decoder]: the input size is tiny and unnecessary to tile.")
            vae_decoder_tiled = False

        with VRAMPeakMonitor("decoding generated latent"):
            x = self.cldm.vae_decode(
                z,
                vae_decoder_tiled,
                vae_decoder_tile_size // 8,
            )

        x = x[:, :, :h0, :w0]
        self.cldm.control_scales = control_scales
        return x

    def _tensor_to_uint8_rgb(self, x: torch.Tensor) -> np.ndarray:
        x = x.detach().float().clamp(0, 1)
        x = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).round()
        return x.clip(0, 255).astype(np.uint8)

    def _prepare_text_guidance(self, lq_tensor: torch.Tensor, cond_img: torch.Tensor) -> None:
        if self.cond_fn is None:
            return
        text_guidance = getattr(self.cond_fn, "text_guidance", None)
        if text_guidance is None:
            return
        if lq_tensor.size(0) != 1 or cond_img.size(0) != 1:
            raise NotImplementedError("Text guidance currently supports batch_size=1.")

        args = self.text_args
        if self.text_detector is None:
            raise RuntimeError("Text guidance is enabled but text detector is not configured.")

        lq_for_mask = F.interpolate(
            lq_tensor,
            size=cond_img.shape[-2:],
            mode="bicubic",
            antialias=True,
        ).clamp(0, 1)
        lq_np = self._tensor_to_uint8_rgb(lq_for_mask)
        stage1_np = self._tensor_to_uint8_rgb(cond_img)

        lq_detections = []
        stage1_detections = []
        if args.text_mask_source in ["lq", "union"]:
            lq_detections = self.text_detector.detect(lq_np, source="lq")
        if args.text_mask_source in ["stage1", "union"]:
            stage1_detections = self.text_detector.detect(stage1_np, source="stage1")

        detections = lq_detections + stage1_detections
        polygons = [det["polygon"] for det in detections]
        binary_mask = polygons_to_mask(
            polygons,
            stage1_np.shape,
            min_area=args.text_min_area,
        )
        soft_mask = postprocess_text_mask(
            binary_mask,
            dilate=args.text_mask_dilate,
            blur=args.text_mask_blur,
        )

        mask_tensor = (
            torch.from_numpy(soft_mask)
            .to(device=cond_img.device, dtype=cond_img.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        text_guidance.load_target(cond_img.detach() * 2 - 1, mask_tensor.detach())

        nonzero_ratio = float((soft_mask > 0).mean())
        print(f"[TextGuidance] LQ detections: {len(lq_detections)}")
        print(f"[TextGuidance] Stage 1 detections: {len(stage1_detections)}")
        print(f"[TextGuidance] final union detections: {len(detections)}")
        print(f"[TextGuidance] mask nonzero ratio: {nonzero_ratio:.6f}")

        if getattr(args, "save_text_mask", False):
            base_debug_dir = (
                args.text_debug_dir
                if getattr(args, "text_debug_dir", None)
                else os.path.join(args.output, "text_guidance_debug")
            )
            debug_dir = base_debug_dir
            if self.text_debug_stem:
                debug_dir = os.path.join(base_debug_dir, self.text_debug_stem)
            save_text_mask_debug(
                debug_dir,
                lq_np,
                stage1_np,
                binary_mask,
                soft_mask,
                detections,
                {
                    "text_guidance": True,
                    "text_detector": args.text_detector,
                    "text_mask_source": args.text_mask_source,
                    "text_guidance_scale": args.text_guidance_scale,
                    "text_rgb_weight": args.text_rgb_weight,
                    "text_edge_weight": args.text_edge_weight,
                    "text_guidance_start": args.text_guidance_start,
                    "text_guidance_stop": args.text_guidance_stop,
                    "text_guidance_mode": args.text_guidance_mode,
                    "text_grad_clip": args.text_grad_clip,
                    "text_mask_dilate": args.text_mask_dilate,
                    "text_mask_blur": args.text_mask_blur,
                    "text_min_confidence": args.text_min_confidence,
                    "text_min_area": args.text_min_area,
                    "easyocr_langs": args.easyocr_langs,
                },
            )
            print(f"[TextGuidance] saved debug masks to {debug_dir}")

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        cleaner_tiled: bool,
        cleaner_tile_size: int,
        cleaner_tile_stride: int,
        vae_encoder_tiled: bool,
        vae_encoder_tile_size: int,
        vae_decoder_tiled: bool,
        vae_decoder_tile_size: int,
        cldm_tiled: bool,
        cldm_tile_size: int,
        cldm_tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        start_point_type: str,
        sampler_type: str,
        noise_aug: int,
        rescale_cfg: bool,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> np.ndarray:
        t_total = tick()

        t = tick()
        lq_tensor = (
            torch.tensor(lq, dtype=torch.float32, device=self.device)
            .div(255)
            .clamp(0, 1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.set_output_size(lq_tensor.size())
        tock(t, "input numpy -> tensor")

        t = tick()
        with VRAMPeakMonitor("applying cleaner"):
            cond_img = self.apply_cleaner(
                lq_tensor, cleaner_tiled, cleaner_tile_size, cleaner_tile_stride
            )
        tock(t, "stage1 cleaner")
        self._prepare_text_guidance(lq_tensor, cond_img)

        assert all(x >= 512 for x in cond_img.shape[2:]), (
            "The resolution of stage-1 model output should be greater than 512, "
            "since it will be used as condition for stage-2 model."
        )

        t = tick()
        sample = self.apply_cldm(
            cond_img,
            steps,
            strength,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            cldm_tile_stride,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            start_point_type,
            sampler_type,
            noise_aug,
            rescale_cfg,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            eta,
            order,
        )
        tock(t, "stage2 cldm")

        t = tick()
        sample = F.interpolate(
            wavelet_reconstruction((sample + 1) / 2, cond_img),
            size=self.output_size,
            mode="bicubic",
            antialias=True,
        )
        sample = (
            (sample * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        tock(t, "final postprocess")

        tock(t_total, "pipeline total")
        return sample


class BSRNetPipeline(Pipeline):

    def __init__(
        self,
        cleaner: RRDBNet,
        cldm: ControlLDM,
        diffusion: Diffusion,
        cond_fn: Guidance | None,
        device: str,
        upscale: float,
    ) -> None:
        super().__init__(cleaner, cldm, diffusion, cond_fn, device)
        self.upscale = upscale

    def set_output_size(self, lq_size: Tuple[int]) -> None:
        h, w = lq_size[2:]
        self.output_size = (int(h * self.upscale), int(w * self.upscale))

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[BSRNet]: the input size is tiny and unnecessary to tile.")
            tiled = False

        if tiled:
            model = make_tiled_fn(
                self.cleaner,
                tile_size,
                tile_stride,
                scale_type="up",
                scale=4,
            )
        else:
            model = self.cleaner

        output_upscale4 = model(lq)
        if min(self.output_size) < 512:
            output = resize_short_edge_to(output_upscale4, size=512)
        else:
            output = F.interpolate(
                output_upscale4, size=self.output_size, mode="bicubic", antialias=True
            )
        return output


class SwinIRPipeline(Pipeline):

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[SwinIR]: the input size is tiny and unnecessary to tile.")
            tiled = False
        if tiled:
            if tile_size % 64 != 0:
                raise ValueError("SwinIR (cleaner) tile size must be a multiple of 64")

        if not tiled:
            if min(lq.shape[2:]) < 512:
                lq = resize_short_edge_to(lq, size=512)
            h0, w0 = lq.shape[2:]
            lq = pad_to_multiples_of(lq, multiple=64)
            output = self.cleaner(lq)[:, :, :h0, :w0]
        else:
            tiled_model = make_tiled_fn(
                self.cleaner,
                size=tile_size,
                stride=tile_stride,
            )
            output = tiled_model(lq)
            if min(output.shape[2:]) < 512:
                output = resize_short_edge_to(output, size=512)
        return output


class SCUNetPipeline(Pipeline):

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[SCUNet]: the input size is tiny and unnecessary to tile.")
            tiled = False

        if tiled:
            model = make_tiled_fn(
                self.cleaner,
                tile_size,
                tile_stride,
            )
        else:
            model = self.cleaner

        output = model(lq)
        if min(output.shape[2:]) < 512:
            output = resize_short_edge_to(output, size=512)
        return output


class RestormerPipeline(Pipeline):

    @staticmethod
    def _pad_reflect_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.size()
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return x, h, w
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def _run_tiled(
        self, x: torch.Tensor, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        tile = min(tile_size, h, w)

        if tile % 8 != 0:
            raise ValueError("Restormer cleaner tile size must be a multiple of 8")
        if tile_stride <= 0:
            raise ValueError("Restormer cleaner tile stride must be positive")
        if tile_stride > tile:
            raise ValueError("Restormer cleaner tile stride must be <= tile size")

        h_idx_list = list(range(0, max(h - tile, 0), tile_stride)) + [h - tile]
        w_idx_list = list(range(0, max(w - tile, 0), tile_stride)) + [w - tile]

        E = torch.zeros((b, c, h, w), device=x.device, dtype=x.dtype)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = x[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = self.cleaner(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch)
                W[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch_mask)

        return E.div_(W.clamp_min(1e-8))

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if min(lq.shape[2:]) < 512:
            lq = resize_short_edge_to(lq, size=512)

        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[Restormer]: input is smaller than tile size, disable cleaner tiling.")
            tiled = False

        lq, h0, w0 = self._pad_reflect_to_multiple(lq, multiple=8)

        if not tiled:
            output = self.cleaner(lq)
        else:
            output = self._run_tiled(lq, tile_size, tile_stride)

        output = output[:, :, :h0, :w0]

        if min(output.shape[2:]) < 512:
            output = resize_short_edge_to(output, size=512)

        return output


class NAFNetPipeline(Pipeline):

    @staticmethod
    def _pad_reflect_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.size()
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return x, h, w
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if min(lq.shape[2:]) < 512:
            lq = resize_short_edge_to(lq, size=512)

        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[NAFNet]: input is smaller than tile size, disable cleaner tiling.")
            tiled = False

        lq, h0, w0 = self._pad_reflect_to_multiple(lq, multiple=8)

        if tiled:
            model = make_tiled_fn(self.cleaner, tile_size, tile_stride)
            output = model(lq)
        else:
            output = self.cleaner(lq)

        output = output[:, :, :h0, :w0]
        if min(output.shape[2:]) < 512:
            output = resize_short_edge_to(output, size=512)
        return output


class MPRNetPipeline(Pipeline):

    @staticmethod
    def _pad_reflect_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.size()
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return x, h, w
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if min(lq.shape[2:]) < 512:
            lq = resize_short_edge_to(lq, size=512)

        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[MPRNet]: input is smaller than tile size, disable cleaner tiling.")
            tiled = False

        lq, h0, w0 = self._pad_reflect_to_multiple(lq, multiple=8)

        if tiled:
            model = make_tiled_fn(self.cleaner, tile_size, tile_stride)
            output = model(lq)
        else:
            output = self.cleaner(lq)

        output = output[:, :, :h0, :w0]
        if min(output.shape[2:]) < 512:
            output = resize_short_edge_to(output, size=512)
        return output
