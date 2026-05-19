import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
)
from ..pipeline import (
    SwinIRPipeline,
    SCUNetPipeline,
    RestormerPipeline,
    NAFNetPipeline,
    MPRNetPipeline,
)
from ..model import SwinIR, SCUNet
from ..model.restormer_from_clone import RestormerFromClone
from ..model.nafnet_from_clone import NAFNetFromClone
from ..model.mprnet_from_clone import MPRNetFromClone


class BIDInferenceLoop(InferenceLoop):

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
        if getattr(self.args, "stage1_model", "default") == "swinir":
            config = self.args.stage1_config or "configs/inference/swinir.yaml"
            weight = self.args.stage1_ckpt or (
                MODELS["swinir_general"]
                if self.args.version == "v1"
                else MODELS["swinir_realesrgan"]
            )
            self.cleaner: SwinIR = instantiate_from_config(OmegaConf.load(config))
            model_weight = load_model_from_url(weight) if weight.startswith("http") else torch.load(weight, map_location="cpu")
            self.cleaner.load_state_dict(model_weight, strict=True)
            self.cleaner.eval().to(self.args.device)
            return

        if self.args.version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
        elif self.args.version in ["v2", "v2.1"]:
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
        else:
            raise ValueError(f"Unsupported version for BIDInferenceLoop: {self.args.version}")

        self.cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        model_weight = load_model_from_url(weight)
        self.cleaner.load_state_dict(model_weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if getattr(self.args, "cleaner_type", "default") == "restormer":
            pipeline_class = RestormerPipeline
        elif getattr(self.args, "cleaner_type", "default") == "nafnet":
            pipeline_class = NAFNetPipeline
        elif getattr(self.args, "cleaner_type", "default") == "mprnet":
            pipeline_class = MPRNetPipeline
        elif getattr(self.args, "stage1_model", "default") == "swinir":
            pipeline_class = SwinIRPipeline
        elif self.args.version == "v1":
            pipeline_class = SwinIRPipeline
        elif self.args.version in ["v2", "v2.1"]:
            pipeline_class = SCUNetPipeline
        else:
            raise ValueError(f"Unsupported version for BIDInferenceLoop: {self.args.version}")

        self.pipeline = pipeline_class(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
