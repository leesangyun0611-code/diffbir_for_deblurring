import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline, NAFNetPipeline, MPRNetPipeline, RestormerPipeline
from ..model import SwinIR
from ..model.restormer_from_clone import RestormerFromClone
from ..model.nafnet_from_clone import NAFNetFromClone
from ..model.mprnet_from_clone import MPRNetFromClone


class BFRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        if getattr(self.args, "cleaner_type", "default") == "restormer":
            self.cleaner = RestormerFromClone(
                repo_dir=self.args.restormer_repo,
                task=self.args.restormer_task,
                ckpt_path=self.args.restormer_ckpt,
            )
            self.cleaner.eval().to(self.args.device)
            return
        if getattr(self.args, "cleaner_type", "default") == "nafnet":
            self.cleaner = NAFNetFromClone(
                repo_dir=self.args.nafnet_repo,
                ckpt_path=self.args.nafnet_ckpt,
            )
            self.cleaner.eval().to(self.args.device)
            return
        if getattr(self.args, "cleaner_type", "default") == "mprnet":
            self.cleaner = MPRNetFromClone(
                repo_dir=self.args.mprnet_repo,
                ckpt_path=self.args.mprnet_ckpt,
            )
            self.cleaner.eval().to(self.args.device)
            return

        config = self.args.stage1_config or "configs/inference/swinir.yaml"
        self.cleaner: SwinIR = instantiate_from_config(OmegaConf.load(config))
        weight_path = self.args.stage1_ckpt or MODELS["swinir_face"]
        weight = load_model_from_url(weight_path) if weight_path.startswith("http") else torch.load(weight_path, map_location="cpu")
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if getattr(self.args, "cleaner_type", "default") == "restormer":
            pipeline_class = RestormerPipeline
        elif getattr(self.args, "cleaner_type", "default") == "nafnet":
            pipeline_class = NAFNetPipeline
        elif getattr(self.args, "cleaner_type", "default") == "mprnet":
            pipeline_class = MPRNetPipeline
        else:
            pipeline_class = SwinIRPipeline

        self.pipeline = pipeline_class(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
