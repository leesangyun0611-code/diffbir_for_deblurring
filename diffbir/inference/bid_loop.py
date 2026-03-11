import os
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
        elif self.args.version == "v2":
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
        else:
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_realesrgan"]

        self.cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))

        if getattr(self.args, "stage1_ckpt", ""):
            ckpt = torch.load(self.args.stage1_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            print(f"load fine-tuned stage1 checkpoint: {self.args.stage1_ckpt}")
        else:
            state_dict = load_model_from_url(weight)
            print("load default pretrained stage1 checkpoint")

        self.cleaner.load_state_dict(state_dict, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if self.args.version == "v1" or self.args.version == "v2.1":
            pipeline_class = SwinIRPipeline
        else:
            pipeline_class = SCUNetPipeline

        self.pipeline = pipeline_class(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        return super().after_load_lq(lq)

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        auto_cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]

        use_external_stage1 = bool(getattr(self.args, "external_stage1_dir", ""))

        for lq in self.load_lq():
            with torch.no_grad():
                caption = self.captioner(lq)

            pos_prompt = ", ".join(
                [text for text in [caption, self.args.pos_prompt] if text]
            )
            neg_prompt = self.args.neg_prompt

            lq_np = self.after_load_lq(lq)

            if use_external_stage1:
                file_stem = self.loop_ctx["file_stem"]
                ext_candidates = [".png", ".jpg", ".jpeg"]
                stage1_path = None

                for ext in ext_candidates:
                    cand = os.path.join(self.args.external_stage1_dir, file_stem + ext)
                    if os.path.exists(cand):
                        stage1_path = cand
                        break

                if stage1_path is None:
                    raise FileNotFoundError(
                        f"External stage1 image not found for {file_stem} "
                        f"in {self.args.external_stage1_dir}"
                    )

                stage1_img = Image.open(stage1_path).convert("RGB")
                stage1_np = np.array(stage1_img)

                self.pipeline.external_cond_img = stage1_np
                print(f"[INFO] Using external stage1 image: {stage1_path}")
            else:
                self.pipeline.external_cond_img = None

            n_samples = self.args.n_samples
            batch_size = self.args.batch_size
            num_batches = (n_samples + batch_size - 1) // batch_size
            samples = []

            for i in range(num_batches):
                n_inputs = min((i + 1) * batch_size, n_samples) - i * batch_size

                self.pipeline.debug_file_stem = self.loop_ctx["file_stem"]
                self.pipeline.debug_save_dir = self.save_dir

                with torch.autocast(self.args.device, auto_cast_type):
                    batch_samples = self.pipeline.run(
                        np.tile(lq_np[None], (n_inputs, 1, 1, 1)),
                        self.args.steps,
                        self.args.strength,
                        self.args.cleaner_tiled,
                        self.args.cleaner_tile_size,
                        self.args.cleaner_tile_stride,
                        self.args.vae_encoder_tiled,
                        self.args.vae_encoder_tile_size,
                        self.args.vae_decoder_tiled,
                        self.args.vae_decoder_tile_size,
                        self.args.cldm_tiled,
                        self.args.cldm_tile_size,
                        self.args.cldm_tile_stride,
                        pos_prompt,
                        neg_prompt,
                        self.args.cfg_scale,
                        self.args.start_point_type,
                        self.args.sampler,
                        self.args.noise_aug,
                        self.args.rescale_cfg,
                        self.args.s_churn,
                        self.args.s_tmin,
                        self.args.s_tmax,
                        self.args.s_noise,
                        self.args.eta,
                        self.args.order,
                    )
                samples.extend(list(batch_samples))

            self.save(samples, pos_prompt, neg_prompt)