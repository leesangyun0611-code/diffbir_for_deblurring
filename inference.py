from argparse import ArgumentParser, Namespace
from pathlib import Path
import importlib.util

import torch
from PIL import Image
from accelerate.utils import set_seed

from diffbir.inference import (
    BSRInferenceLoop,
    BFRInferenceLoop,
    BIDInferenceLoop,
    UnAlignedBFRInferenceLoop,
    CustomInferenceLoop,
)


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


def check_pyiqa_available() -> None:
    """
    Ensure pyiqa is installed when IQA evaluation is requested.
    """
    if importlib.util.find_spec("pyiqa") is None:
        raise ImportError(
            "MANIQA evaluation requires 'pyiqa', but it is not installed.\n"
            "Install it with:\n"
            "  pip install pyiqa"
        )


DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)


def maybe_resize_input_to_small(input_path: str, max_side: int = 511):
    """
    Resize only large input images so that BOTH width and height are <= max_side,
    while preserving aspect ratio.

    Behavior:
    - If input is a directory:
        save processed files into <input_dir>_resized
    - If input is a single file:
        save processed file into <parent_dir>/<stem>_resized<suffix>
    - Small images are copied without resizing
    - Original inputs are never modified

    Returns:
        new_input_path: str
            Original path if no resize was needed, otherwise resized file/dir path
        resized_any: bool
            Whether any image was actually resized
    """
    src = Path(input_path)
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _save_processed(in_file: Path, out_file: Path) -> bool:
        with Image.open(in_file) as img:
            w, h = img.size
            out_file.parent.mkdir(parents=True, exist_ok=True)

            if w <= max_side and h <= max_side:
                img.save(out_file)
                return False

            scale = min(max_side / w, max_side / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            new_w = min(new_w, max_side)
            new_h = min(new_h, max_side)

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized.save(out_file)

            print(f"[AutoResize] {in_file.name}: {w}x{h} -> {new_w}x{new_h}")
            return True

    if src.is_file():
        if src.suffix.lower() not in valid_exts:
            return input_path, False

        out_file = src.parent / f"{src.stem}_resized{src.suffix}"
        resized = _save_processed(src, out_file)
        return (str(out_file) if resized else input_path), resized

    if src.is_dir():
        files = sorted(
            [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        )

        if not files:
            return input_path, False

        resize_dir = src.parent / f"{src.name}_resized"
        resize_dir.mkdir(parents=True, exist_ok=True)

        resized_any = False
        for f in files:
            out_file = resize_dir / f.name
            resized = _save_processed(f, out_file)
            if resized:
                resized_any = True

        return (str(resize_dir) if resized_any else input_path), resized_any

    return input_path, False


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # model parameters
    parser.add_argument(
        "--task",
        type=str,
        default="sr",
        choices=["sr", "face", "denoise", "unaligned_face"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )
    parser.add_argument(
        "--upscale", type=float, default=4, help="Upscale factor of output."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2.1",
        choices=["v1", "v2", "v2.1", "custom"],
        help="DiffBIR model version.",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="",
        help="Path to training config. Only works when version is custom.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to saved checkpoint. Only works when version is custom.",
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        default="",
        help="Path to a ControlNet LoRA checkpoint trained by train_stage2_lora.py.",
    )

    # sampling parameters
    parser.add_argument(
        "--sampler",
        type=str,
        default="edm_dpm++_3m_sde",
        choices=[
            "dpm++_m2",
            "spaced",
            "ddim",
            "edm_euler",
            "edm_euler_a",
            "edm_heun",
            "edm_dpm_2",
            "edm_dpm_2_a",
            "edm_lms",
            "edm_dpm++_2s_a",
            "edm_dpm++_sde",
            "edm_dpm++_2m",
            "edm_dpm++_2m_sde",
            "edm_dpm++_3m_sde",
        ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Sampling steps. More steps, more details.",
    )
    parser.add_argument(
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),
    )
    parser.add_argument(
        "--cleaner_tiled",
        action="store_true",
        help="Enable tiled inference for stage-1 model, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cleaner_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cleaner_tile_stride", type=int, default=256, help="Stride between tiles."
    )

    # Restormer stage-1 cleaner options
    parser.add_argument(
        "--cleaner_type",
        type=str,
        default="default",
        choices=["default", "restormer", "nafnet", "mprnet"],
        help="Override stage-1 cleaner. Use 'restormer' to replace the default DiffBIR cleaner.",
    )
    parser.add_argument(
        "--stage1_model",
        type=str,
        default="default",
        choices=["default", "swinir", "nafnet", "restormer", "mprnet"],
        help=(
            "Stage-1 restoration selector. 'default' and 'swinir' preserve the "
            "repo's built-in DiffBIR cleaner path; nafnet/restormer/mprnet map "
            "to the existing cleaner_type adapters."
        ),
    )
    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default="",
        help="Optional alias for the selected stage-1 adapter checkpoint.",
    )
    parser.add_argument(
        "--stage1_config",
        type=str,
        default="",
        help="Reserved optional stage-1 config path for future adapters.",
    )
    parser.add_argument(
        "--restormer_repo",
        type=str,
        default="third_party/Restormer",
        help="Path to cloned Restormer repository.",
    )
    parser.add_argument(
        "--restormer_task",
        type=str,
        default="Motion_Deblurring",
        choices=[
            "Motion_Deblurring",
            "Real_Denoising",
            "Gaussian_Color_Denoising",
            "Gaussian_Gray_Denoising",
        ],
        help="Restormer pretrained task type.",
    )
    parser.add_argument(
        "--restormer_ckpt",
        type=str,
        default="",
        help="Path to local Restormer checkpoint.",
    )
    parser.add_argument(
        "--nafnet_repo",
        type=str,
        default="third_party/NAFNet",
        help="Path to cloned NAFNet repository.",
    )
    parser.add_argument(
        "--nafnet_ckpt",
        type=str,
        default="",
        help="Path to NAFNet checkpoint.",
    )
    parser.add_argument(
        "--mprnet_repo",
        type=str,
        default="third_party/MPRNet",
        help="Path to cloned MPRNet repository.",
    )
    parser.add_argument(
        "--mprnet_ckpt",
        type=str,
        default="",
        help="Path to MPRNet checkpoint.",
    )

    parser.add_argument(
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--captioner",
        type=str,
        choices=["none", "llava", "ram"],
        default="llava",
        help="Select a model to describe the content of your input image.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default=DEFAULT_POS_PROMPT,
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=DEFAULT_NEG_PROMPT,
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=6.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Nothing to say.")

    # guidance parameters
    parser.add_argument(
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(
        "--g_scale",
        type=float,
        default=0.0,
        help="Guidance strength / step size for restoration guidance.",
    )
    parser.add_argument(
        "--g_start",
        type=int,
        default=1001,
        help="Apply guidance only when t < g_start.",
    )
    parser.add_argument(
        "--g_stop",
        type=int,
        default=-1,
        help="Apply guidance only when t > g_stop.",
    )
    parser.add_argument(
        "--g_space",
        type=str,
        default="latent",
        choices=["latent", "rgb"],
        help="Space where restoration guidance loss is computed.",
    )
    parser.add_argument(
        "--g_repeat",
        type=int,
        default=1,
        help="Number of guidance updates per sampling step.",
    )
    parser.add_argument(
        "--g_weight_mode",
        type=str,
        default="lowfreq",
        choices=["lowfreq", "highfreq"],
        help=(
            "Weight-map mode for weighted MSE guidance. "
            "'lowfreq' reproduces the original low-frequency-emphasis behavior, "
            "while 'highfreq' emphasizes high-frequency regions."
        ),
    )
    parser.add_argument(
        "--g_weight_floor",
        type=float,
        default=0.0,
        help=(
            "Minimum floor value for the weighted guidance map. "
            "Useful to avoid weights becoming too close to zero."
        ),
    )
    parser.add_argument(
        "--g_weight_gamma",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to the weight map before floor scaling. "
            "Values > 1 sharpen the contrast of the weighting."
        ),
    )
    parser.add_argument(
        "--g_block_size",
        type=int,
        default=2,
        help=(
            "Patch size used for patch-level Sobel gradient aggregation "
            "in weighted MSE guidance."
        ),
    )

    # text-aware guidance parameters
    parser.add_argument("--text_guidance", action="store_true", help="Enable text-aware Stage 2 guidance.")
    parser.add_argument("--text_detector", type=str, default="easyocr", choices=["easyocr", "none"])
    parser.add_argument(
        "--text_mask_source",
        type=str,
        default="union",
        choices=["stage1", "lq", "union"],
        help="Image source used to detect text regions.",
    )
    parser.add_argument("--text_guidance_scale", type=float, default=0.3)
    parser.add_argument("--text_rgb_weight", type=float, default=0.1)
    parser.add_argument("--text_edge_weight", type=float, default=1.0)
    parser.add_argument("--text_guidance_start", type=float, default=0.0)
    parser.add_argument("--text_guidance_stop", type=float, default=0.35)
    parser.add_argument(
        "--text_guidance_mode",
        type=str,
        default="late",
        choices=["late", "early", "all", "fraction"],
    )
    parser.add_argument("--text_mask_dilate", type=int, default=5)
    parser.add_argument("--text_mask_blur", type=float, default=1.0)
    parser.add_argument("--text_min_confidence", type=float, default=0.3)
    parser.add_argument("--text_min_area", type=float, default=0)
    parser.add_argument("--easyocr_langs", type=str, default="ko,en")
    parser.add_argument("--easyocr_text_threshold", type=float, default=0.4)
    parser.add_argument("--easyocr_low_text", type=float, default=0.2)
    parser.add_argument("--easyocr_link_threshold", type=float, default=0.4)
    parser.add_argument("--easyocr_canvas_size", type=int, default=2560)
    parser.add_argument("--easyocr_mag_ratio", type=float, default=2.0)
    parser.add_argument("--save_text_mask", action="store_true")
    parser.add_argument("--text_debug_dir", type=str, default="")
    parser.add_argument("--text_grad_clip", type=float, default=0.05)
    parser.add_argument(
        "--text_guidance_loss",
        type=str,
        default="edge_rgb",
        choices=["edge_rgb", "charbonnier", "l1", "mse"],
    )
    parser.add_argument(
        "--text_regional_noise",
        action="store_true",
        help=(
            "Use text-aware regional noise initialization: text regions start "
            "from lower-noise Stage 1 latent while non-text regions keep the "
            "normal diffusion start."
        ),
    )
    parser.add_argument(
        "--text_noise_timestep_ratio",
        type=float,
        default=0.15,
        help="Timestep ratio used for text-region start noise. Lower preserves Stage 1 text more.",
    )
    parser.add_argument(
        "--nontext_noise_timestep_ratio",
        type=float,
        default=1.0,
        help="Timestep ratio used for non-text start noise. 1.0 matches the original cond start.",
    )
    parser.add_argument(
        "--text_noise_scale",
        type=float,
        default=1.0,
        help="Extra multiplier for random noise injected into text-region q_sample.",
    )
    parser.add_argument(
        "--nontext_noise_scale",
        type=float,
        default=1.0,
        help="Extra multiplier for random noise injected into non-text q_sample.",
    )
    parser.add_argument(
        "--text_latent_anchor",
        action="store_true",
        help=(
            "After each Stage 2 sampler step, softly anchor text-region latent "
            "back to the Stage 1 condition latent."
        ),
    )
    parser.add_argument(
        "--text_latent_anchor_alpha",
        type=float,
        default=0.7,
        help="Text latent anchoring strength in [0, 1]. Higher follows Stage 1 text more.",
    )

    # common parameters
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder that contains your low-quality images.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for each image."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save restored results."
    )
    parser.add_argument("--seed", type=int, default=231)

    # optional auto resize
    parser.add_argument(
        "--auto_resize_input",
        action="store_true",
        help="Enable auto resize for large input images before inference.",
    )
    parser.add_argument(
        "--max_input_side",
        type=int,
        default=511,
        help="Maximum side length used when auto resizing input images.",
    )

    # IQA / MANIQA parameters
    parser.add_argument(
        "--eval_maniqa",
        action="store_true",
        help="Evaluate final restored outputs with MANIQA after inference.",
    )
    parser.add_argument(
        "--maniqa_model",
        type=str,
        default="maniqa-pipal",
        choices=["maniqa", "maniqa-kadid", "maniqa-pipal"],
        help="MANIQA variant used by pyiqa.",
    )
    parser.add_argument(
        "--eval_lpips",
        action="store_true",
        help="Evaluate final restored outputs with LPIPS after inference.",
    )
    parser.add_argument(
        "--lpips_model",
        type=str,
        default="alex",
        choices=["alex", "vgg"],
        help="LPIPS network architecture.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="",
        help="Directory with GT images for LPIPS evaluation (same file names as input).",
    )
    parser.add_argument(
        "--iqa_csv",
        type=str,
        default="iqa_results.csv",
        help="CSV filename used to save IQA results inside the output directory.",
    )

    # mps has not been tested
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--llava_bit", type=str, default="4", choices=["16", "8", "4"])
    parser.add_argument(
        "--start_point_noise_scale",
        type=float,
        default=0.0,
        help="Noise scale when using cond start point.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage1_model in ["restormer", "nafnet", "mprnet"]:
        args.cleaner_type = args.stage1_model
    elif args.stage1_model == "swinir":
        args.cleaner_type = "default"

    if args.stage1_ckpt:
        if args.cleaner_type == "restormer":
            args.restormer_ckpt = args.stage1_ckpt
        elif args.cleaner_type == "nafnet":
            args.nafnet_ckpt = args.stage1_ckpt
        elif args.cleaner_type == "mprnet":
            args.mprnet_ckpt = args.stage1_ckpt

    if args.text_guidance and args.sampler != "spaced":
        print(
            "[TextGuidance] Warning: text guidance is currently inserted in the "
            "'spaced' sampler guidance path. Use --sampler spaced for text-aware guidance."
        )
    if args.text_guidance:
        print("[TextGuidanceArgs][inference.py] text_guidance=", args.text_guidance)
        print("[TextGuidanceArgs][inference.py] text_detector=", args.text_detector)
        print("[TextGuidanceArgs][inference.py] text_mask_source=", args.text_mask_source)
        print("[TextGuidanceArgs][inference.py] text_min_confidence=", args.text_min_confidence)
        print("[TextGuidanceArgs][inference.py] text_min_area=", args.text_min_area)
        print("[TextGuidanceArgs][inference.py] easyocr_langs=", args.easyocr_langs)
        print("[TextGuidanceArgs][inference.py] easyocr_text_threshold=", args.easyocr_text_threshold)
        print("[TextGuidanceArgs][inference.py] easyocr_low_text=", args.easyocr_low_text)
        print("[TextGuidanceArgs][inference.py] easyocr_link_threshold=", args.easyocr_link_threshold)
        print("[TextGuidanceArgs][inference.py] easyocr_canvas_size=", args.easyocr_canvas_size)
        print("[TextGuidanceArgs][inference.py] easyocr_mag_ratio=", args.easyocr_mag_ratio)
        print("[TextGuidanceArgs][inference.py] save_text_mask=", args.save_text_mask)
        print("[TextGuidanceArgs][inference.py] text_regional_noise=", args.text_regional_noise)
        print("[TextGuidanceArgs][inference.py] text_noise_timestep_ratio=", args.text_noise_timestep_ratio)
        print("[TextGuidanceArgs][inference.py] nontext_noise_timestep_ratio=", args.nontext_noise_timestep_ratio)
        print("[TextGuidanceArgs][inference.py] text_noise_scale=", args.text_noise_scale)
        print("[TextGuidanceArgs][inference.py] nontext_noise_scale=", args.nontext_noise_scale)
        print("[TextGuidanceArgs][inference.py] text_latent_anchor=", args.text_latent_anchor)
        print("[TextGuidanceArgs][inference.py] text_latent_anchor_alpha=", args.text_latent_anchor_alpha)

    if args.auto_resize_input:
        args.input, resized = maybe_resize_input_to_small(
            args.input,
            max_side=args.max_input_side,
        )
        if resized:
            print(f"[AutoResize] Using resized input path: {args.input}")
        else:
            print("[AutoResize] Input already small enough. No resize applied.")
    else:
        print("[AutoResize] Disabled. Using original input path.")

    args.device = check_device(args.device)
    set_seed(args.seed)

    if args.eval_maniqa:
        check_pyiqa_available()
        if not args.iqa_csv.lower().endswith(".csv"):
            raise ValueError("--iqa_csv must end with '.csv'")
        print(f"[IQA] MANIQA evaluation enabled: {args.maniqa_model}")
        print(f"[IQA] Result CSV will be saved as: {args.iqa_csv}")

    if args.eval_lpips:
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "LPIPS evaluation requires 'lpips', but it is not installed.\n"
                "Install it with:\n"
                "  pip install lpips"
            )
        print(f"[IQA] LPIPS evaluation enabled: {args.lpips_model}")
        print(f"[IQA] Result CSV will be saved as: {args.iqa_csv}")

    if args.version != "custom":
        loops = {
            "sr": BSRInferenceLoop,
            "denoise": BIDInferenceLoop,
            "face": BFRInferenceLoop,
            "unaligned_face": UnAlignedBFRInferenceLoop,
        }
        loops[args.task](args).run()
    else:
        CustomInferenceLoop(args).run()

    print("done!")


if __name__ == "__main__":
    main()
