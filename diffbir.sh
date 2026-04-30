#!/usr/bin/env bash
set -eo pipefail

# ==========================================
# DiffBIR single experiment launcher
# Run with: bash diffbir.sh
# ==========================================

# ---------- Device ----------
GPU=0
DEVICE="cuda"                 # cpu / cuda / mps
PRECISION="fp16"              # fp32 / fp16 / bf16
SEED=42

# ---------- Current experiment ----------
EXPERIMENT_NAME="exp3_wmse_rgb_highfreq_s1.5_gamma1_denoise_lora2000_test_image_text"
TASK="denoise"                # sr / face / denoise / unaligned_face
GUIDANCE=true                 # true / false
G_LOSS="w_mse"                # mse / w_mse
G_SCALE=1.5
G_SPACE="rgb"                 # latent / rgb
G_WEIGHT_MODE="highfreq"      # lowfreq / highfreq
G_WEIGHT_GAMMA=1.0

# ---------- Task / version ----------
VERSION="v2.1"                # v1 / v2 / v2.1 / custom
UPSCALE=1

# ---------- Paths ----------
INPUT_DIR="inputs/demo/test_image_text"
GT_DIR="GT/test_image_text_GT"
RESULTS_ROOT="results"
OUTPUT_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}"

# ---------- Optional input resize ----------
AUTO_RESIZE=false             # true / false
MAX_INPUT_SIDE=511            # only used when AUTO_RESIZE=true

# ---------- Sampling ----------
SAMPLER="spaced"              # spaced / ddim / dpm++_m2 / edm_euler / ...
STEPS=40
CFG_SCALE=4
RESCALE_CFG=false             # true / false

# ---------- Start point ----------
START_POINT_TYPE="cond"       # cond / noise
START_POINT_NOISE_SCALE=""    # e.g. 0.3, 1.0 ; leave empty to disable

# ---------- Batch / outputs ----------
N_SAMPLES=1
BATCH_SIZE=1

# ---------- Prompt / caption ----------
CAPTIONER="none"              # none / llava / ram
POS_PROMPT="a sharp, clean, natural photo, no motion blur, high detail"
NEG_PROMPT="motion blur, blurry, ghosting, smear, low quality, artifacts"

# ---------- Tile toggles ----------
CLEANER_TILED=false
CLDM_TILED=false
VAE_ENCODER_TILED=false
VAE_DECODER_TILED=false

# ---------- Tile sizes / strides ----------
CLEANER_TILE_SIZE=512
CLEANER_TILE_STRIDE=256

CLDM_TILE_SIZE=256
CLDM_TILE_STRIDE=128

VAE_ENCODER_TILE_SIZE=1024
VAE_DECODER_TILE_SIZE=256

# ---------- Restormer stage-1 cleaner ----------
# Roll back to the original stage-1 path at any time with: CLEANER_TYPE="default"
CLEANER_TYPE="restormer"       # default / restormer / nafnet / mprnet
RESTORMER_REPO="third_party/Restormer"
RESTORMER_TASK="Motion_Deblurring"
RESTORMER_CKPT="${RESTORMER_REPO}/Motion_Deblurring/pretrained_models/motion_deblurring.pth"
NAFNET_REPO="third_party/NAFNet"
NAFNET_CKPT="weights/NAFNet-GoPro-width64.pth"
MPRNET_REPO="third_party/MPRNet"
MPRNET_CKPT="weights/MPRNet-Deblurring.pth"

# ---------- Optional noise / sampler extras ----------
NOISE_AUG=0
ETA=0
ORDER=2
STRENGTH=1.0
S_CHURN=0
S_TMIN=0
S_TMAX=999
S_NOISE=1.0

# ---------- Guidance shared options ----------
G_START=1001
G_STOP=1
G_REPEAT=1
G_WEIGHT_FLOOR=0.0
G_BLOCK_SIZE=2

# ---------- MANIQA evaluation ----------
EVAL_MANIQA=true             # true / false
MANIQA_MODEL="maniqa-pipal"  # maniqa / maniqa-kadid / maniqa-pipal
IQA_CSV="iqa_results.csv"

# ---------- LPIPS evaluation ----------
EVAL_LPIPS=true              # true / false
LPIPS_MODEL="alex"           # alex / vgg

# ---------- GT-based metric evaluation ----------
EVAL_METRICS=true            # true / false
RESIZE_PRED_TO_GT=false      # true / false
METRICS_CSV="metrics_psnr_ssim.csv"

# ---------- Custom model paths (optional) ----------
TRAIN_CFG=""
CKPT=""
LORA_CKPT="experiments/stage2_lora_paired/checkpoints/lora_0002000.pt"

# ==========================================
# Build inference command
# ==========================================
CMD=(python inference.py)

# Basic
CMD+=(--task "$TASK")
CMD+=(--version "$VERSION")
CMD+=(--upscale "$UPSCALE")
CMD+=(--input "$INPUT_DIR")
CMD+=(--output "$OUTPUT_DIR")
if [ "$EVAL_LPIPS" = true ] && [ -n "$GT_DIR" ]; then
  CMD+=(--gt_dir "$GT_DIR")
fi
CMD+=(--n_samples "$N_SAMPLES")
CMD+=(--batch_size "$BATCH_SIZE")
CMD+=(--sampler "$SAMPLER")
CMD+=(--steps "$STEPS")
CMD+=(--cfg_scale "$CFG_SCALE")
CMD+=(--start_point_type "$START_POINT_TYPE")
CMD+=(--captioner "$CAPTIONER")
CMD+=(--precision "$PRECISION")
CMD+=(--seed "$SEED")
CMD+=(--device "$DEVICE")

# Stage-1 cleaner selection
CMD+=(--cleaner_type "$CLEANER_TYPE")
if [ "$CLEANER_TYPE" = "restormer" ]; then
  CMD+=(--restormer_repo "$RESTORMER_REPO")
  CMD+=(--restormer_task "$RESTORMER_TASK")
  CMD+=(--restormer_ckpt "$RESTORMER_CKPT")
elif [ "$CLEANER_TYPE" = "nafnet" ]; then
  CMD+=(--nafnet_repo "$NAFNET_REPO")
  CMD+=(--nafnet_ckpt "$NAFNET_CKPT")
elif [ "$CLEANER_TYPE" = "mprnet" ]; then
  CMD+=(--mprnet_repo "$MPRNET_REPO")
  CMD+=(--mprnet_ckpt "$MPRNET_CKPT")
fi

# Optional input resize
if [ "$AUTO_RESIZE" = true ]; then
  CMD+=(--auto_resize_input)
  CMD+=(--max_input_side "$MAX_INPUT_SIDE")
fi

# Optional booleans
if [ "$RESCALE_CFG" = true ]; then
  CMD+=(--rescale_cfg)
fi

if [ "$CLEANER_TILED" = true ]; then
  CMD+=(--cleaner_tiled)
  CMD+=(--cleaner_tile_size "$CLEANER_TILE_SIZE")
  CMD+=(--cleaner_tile_stride "$CLEANER_TILE_STRIDE")
fi

if [ "$CLDM_TILED" = true ]; then
  CMD+=(--cldm_tiled)
  CMD+=(--cldm_tile_size "$CLDM_TILE_SIZE")
  CMD+=(--cldm_tile_stride "$CLDM_TILE_STRIDE")
fi

if [ "$VAE_ENCODER_TILED" = true ]; then
  CMD+=(--vae_encoder_tiled)
  CMD+=(--vae_encoder_tile_size "$VAE_ENCODER_TILE_SIZE")
fi

if [ "$VAE_DECODER_TILED" = true ]; then
  CMD+=(--vae_decoder_tiled)
  CMD+=(--vae_decoder_tile_size "$VAE_DECODER_TILE_SIZE")
fi

if [ "$GUIDANCE" = true ]; then
  CMD+=(--guidance)
  CMD+=(--g_loss "$G_LOSS")
  CMD+=(--g_scale "$G_SCALE")
  CMD+=(--g_start "$G_START")
  CMD+=(--g_stop "$G_STOP")
  CMD+=(--g_space "$G_SPACE")
  CMD+=(--g_repeat "$G_REPEAT")
  CMD+=(--g_weight_mode "$G_WEIGHT_MODE")
  CMD+=(--g_weight_floor "$G_WEIGHT_FLOOR")
  CMD+=(--g_weight_gamma "$G_WEIGHT_GAMMA")
  CMD+=(--g_block_size "$G_BLOCK_SIZE")
fi

# MANIQA
if [ "$EVAL_MANIQA" = true ]; then
  CMD+=(--eval_maniqa)
  CMD+=(--maniqa_model "$MANIQA_MODEL")
fi

# LPIPS
if [ "$EVAL_LPIPS" = true ]; then
  CMD+=(--eval_lpips)
  CMD+=(--lpips_model "$LPIPS_MODEL")
fi

# IQA CSV (if any IQA evaluation is enabled)
if [ "$EVAL_MANIQA" = true ] || [ "$EVAL_LPIPS" = true ]; then
  CMD+=(--iqa_csv "$IQA_CSV")
fi

# Optional scalar/string params
if [ -n "$START_POINT_NOISE_SCALE" ]; then
  CMD+=(--start_point_noise_scale "$START_POINT_NOISE_SCALE")
fi

if [ -n "$POS_PROMPT" ]; then
  CMD+=(--pos_prompt "$POS_PROMPT")
fi

if [ -n "$NEG_PROMPT" ]; then
  CMD+=(--neg_prompt "$NEG_PROMPT")
fi

if [ -n "$TRAIN_CFG" ]; then
  CMD+=(--train_cfg "$TRAIN_CFG")
fi

if [ -n "$CKPT" ]; then
  CMD+=(--ckpt "$CKPT")
fi

if [ -n "$LORA_CKPT" ]; then
  CMD+=(--lora_ckpt "$LORA_CKPT")
fi

# Optional sampler extras
CMD+=(--noise_aug "$NOISE_AUG")
CMD+=(--eta "$ETA")
CMD+=(--order "$ORDER")
CMD+=(--strength "$STRENGTH")
CMD+=(--s_churn "$S_CHURN")
CMD+=(--s_tmin "$S_TMIN")
CMD+=(--s_tmax "$S_TMAX")
CMD+=(--s_noise "$S_NOISE")

# ==========================================
# Run inference
# ==========================================
echo "=========================================="
echo "Running DiffBIR with the following config:"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "GPU=$GPU"
echo "TASK=$TASK"
echo "VERSION=$VERSION"
echo "UPSCALE=$UPSCALE"
echo "INPUT_DIR=$INPUT_DIR"
echo "GT_DIR=$GT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "AUTO_RESIZE=$AUTO_RESIZE"
echo "MAX_INPUT_SIDE=$MAX_INPUT_SIDE"
echo "SAMPLER=$SAMPLER"
echo "STEPS=$STEPS"
echo "CFG_SCALE=$CFG_SCALE"
echo "START_POINT_TYPE=$START_POINT_TYPE"
echo "PRECISION=$PRECISION"
echo "CLEANER_TYPE=$CLEANER_TYPE"
echo "RESTORMER_REPO=$RESTORMER_REPO"
echo "RESTORMER_TASK=$RESTORMER_TASK"
echo "RESTORMER_CKPT=$RESTORMER_CKPT"
echo "NAFNET_REPO=$NAFNET_REPO"
echo "NAFNET_CKPT=$NAFNET_CKPT"
echo "MPRNET_REPO=$MPRNET_REPO"
echo "MPRNET_CKPT=$MPRNET_CKPT"
echo "CLEANER_TILED=$CLEANER_TILED"
echo "CLDM_TILED=$CLDM_TILED"
echo "VAE_ENCODER_TILED=$VAE_ENCODER_TILED"
echo "VAE_DECODER_TILED=$VAE_DECODER_TILED"
echo "GUIDANCE=$GUIDANCE"
echo "G_LOSS=$G_LOSS"
echo "G_SCALE=$G_SCALE"
echo "G_START=$G_START"
echo "G_STOP=$G_STOP"
echo "G_SPACE=$G_SPACE"
echo "G_REPEAT=$G_REPEAT"
echo "G_WEIGHT_MODE=$G_WEIGHT_MODE"
echo "G_WEIGHT_FLOOR=$G_WEIGHT_FLOOR"
echo "G_WEIGHT_GAMMA=$G_WEIGHT_GAMMA"
echo "G_BLOCK_SIZE=$G_BLOCK_SIZE"
echo "EVAL_MANIQA=$EVAL_MANIQA"
echo "MANIQA_MODEL=$MANIQA_MODEL"
echo "EVAL_LPIPS=$EVAL_LPIPS"
echo "LPIPS_MODEL=$LPIPS_MODEL"
echo "IQA_CSV=$IQA_CSV"
echo "EVAL_METRICS=$EVAL_METRICS"
echo "RESIZE_PRED_TO_GT=$RESIZE_PRED_TO_GT"
echo "METRICS_CSV=$METRICS_CSV"
echo "LORA_CKPT=$LORA_CKPT"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

# ==========================================
# Run GT-based metric evaluation
# ==========================================
if [ "$EVAL_METRICS" = true ]; then
  EVAL_CMD=(
    python eval_metrics.py
    --pred_dir "$OUTPUT_DIR"
    --gt_dir "$GT_DIR"
    --save_csv "$OUTPUT_DIR/$METRICS_CSV"
  )

  if [ "$RESIZE_PRED_TO_GT" = true ]; then
    EVAL_CMD+=(--resize_pred_to_gt)
  fi

  echo "=========================================="
  echo "Running metric evaluation..."
  echo "PRED_DIR=$OUTPUT_DIR"
  echo "GT_DIR=$GT_DIR"
  echo "RESIZE_PRED_TO_GT=$RESIZE_PRED_TO_GT"
  echo "METRICS_CSV=$OUTPUT_DIR/$METRICS_CSV"
  echo "=========================================="

  "${EVAL_CMD[@]}"
fi

# ==========================================
# Print MANIQA summary
# ==========================================
if [ "$EVAL_MANIQA" = true ]; then
  IQA_PATH="$OUTPUT_DIR/$IQA_CSV"

  if [ -f "$IQA_PATH" ]; then
    echo "=========================================="
    echo "MANIQA summary"
    echo "IQA_PATH=$IQA_PATH"
    echo "=========================================="

    python - <<PY
import pandas as pd

csv_path = r"$IQA_PATH"
df = pd.read_csv(csv_path)

if "maniqa" not in df.columns:
    raise ValueError(f"'maniqa' column not found in {csv_path}")

for _, row in df.iterrows():
    print(f"{row['file_name']}: MANIQA={row['maniqa']:.6f}")

print("============================================================")
print(f"Average MANIQA: {df['maniqa'].mean():.6f}")
PY
  else
    echo "[WARN] MANIQA CSV not found: $IQA_PATH"
  fi
fi

# ==========================================
# Print Notion-friendly summary
# ==========================================
if [ "$EVAL_MANIQA" = true ] || [ "$EVAL_LPIPS" = true ] || [ "$EVAL_METRICS" = true ]; then
  echo "=========================================="
  echo "Notion summary"
  echo "Copy the next lines into Notion:"
  echo "=========================================="
  python scripts/notion_metrics.py "$OUTPUT_DIR" --iqa_csv "$IQA_CSV" --metrics_csv "$METRICS_CSV" --format blocks
fi
