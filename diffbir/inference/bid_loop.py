import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import (
    SwinIRPipeline,
    SCUNetPipeline,
)
from ..model import SwinIR, SCUNet


class BIDInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
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
        if self.args.version == "v1":
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