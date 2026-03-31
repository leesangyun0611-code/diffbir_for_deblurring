#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# DiffBIR experiment launcher
# Run with: bash diffbir.sh
# ==========================================

# ---------- Device ----------
GPU=1
DEVICE="cuda"                 # cpu / cuda / mps
PRECISION="fp16"              # fp32 / fp16 / bf16
SEED=45

# ---------- Task / version ----------
TASK="denoise"                # sr / face / denoise / unaligned_face
VERSION="v2.1"                # v1 / v2 / v2.1 / custom
UPSCALE=1

# ---------- Paths ----------
INPUT_DIR="inputs/demo/mytest"
OUTPUT_DIR="results/v21_denoise_spaced_scale1_RG_s=2.0_latent"

# ---------- Optional input resize ----------
AUTO_RESIZE=false             # true / false
MAX_INPUT_SIDE=511            # only used when AUTO_RESIZE=true

# ---------- Sampling ----------
SAMPLER="spaced"              # spaced / ddim / dpm++_m2 / edm_euler / ...
STEPS=50
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
POS_PROMPT="realistic photo, natural color, clear structure, clean edges"
NEG_PROMPT="motion blur, smear, ghosting, ringing, oversmoothed, artifacts, distorted details"

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

# ---------- Optional noise / sampler extras ----------
NOISE_AUG=0
ETA=0
ORDER=2
STRENGTH=1.0
S_CHURN=0
S_TMIN=0
S_TMAX=999
S_NOISE=1.0

# ---------- Guidance ----------
GUIDANCE=true                # true / false
G_LOSS="mse"                # mse / w_mse
G_SCALE=2.0
G_START=1001
G_STOP=1
G_SPACE="latent"                 # latent / rgb
G_REPEAT=1

# ---------- Weighted guidance options (for G_LOSS=w_mse) ----------
G_WEIGHT_MODE="lowfreq"      # lowfreq / highfreq
G_WEIGHT_FLOOR=0.0
G_WEIGHT_GAMMA=1.0
G_BLOCK_SIZE=2

# ---------- MANIQA evaluation ----------
EVAL_MANIQA=true              # true / false
MANIQA_MODEL="maniqa-pipal"   # maniqa / maniqa-kadid / maniqa-pipal
IQA_CSV="iqa_results.csv"

# ---------- GT-based metric evaluation ----------
EVAL_METRICS=true             # true / false
GT_DIR="GT"
RESIZE_PRED_TO_GT=false       # true / false

# ---------- Custom model paths (optional) ----------
TRAIN_CFG=""
CKPT=""

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
echo "GPU=$GPU"
echo "TASK=$TASK"
echo "VERSION=$VERSION"
echo "UPSCALE=$UPSCALE"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "AUTO_RESIZE=$AUTO_RESIZE"
echo "MAX_INPUT_SIDE=$MAX_INPUT_SIDE"
echo "SAMPLER=$SAMPLER"
echo "STEPS=$STEPS"
echo "CFG_SCALE=$CFG_SCALE"
echo "START_POINT_TYPE=$START_POINT_TYPE"
echo "PRECISION=$PRECISION"
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
echo "IQA_CSV=$IQA_CSV"
echo "EVAL_METRICS=$EVAL_METRICS"
echo "GT_DIR=$GT_DIR"
echo "RESIZE_PRED_TO_GT=$RESIZE_PRED_TO_GT"
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
  )

  if [ "$RESIZE_PRED_TO_GT" = true ]; then
    EVAL_CMD+=(--resize_pred_to_gt)
  fi

  echo "=========================================="
  echo "Running metric evaluation..."
  echo "PRED_DIR=$OUTPUT_DIR"
  echo "GT_DIR=$GT_DIR"
  echo "RESIZE_PRED_TO_GT=$RESIZE_PRED_TO_GT"
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