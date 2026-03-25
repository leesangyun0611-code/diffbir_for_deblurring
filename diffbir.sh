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
TASK="sr"                     # sr / face / denoise / unaligned_face
VERSION="v2.1"                # v1 / v2 / v2.1 / custom
UPSCALE=1

# ---------- Paths ----------
INPUT_DIR="inputs/demo/mytest"
OUTPUT_DIR="results/v21_sr_spaced_scale1_RG_s=2.0_rgb"

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
POS_PROMPT="sharp, detailed, high quality"
NEG_PROMPT="blurry, low quality, artifacts"

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
GUIDANCE=true              # true / false
G_LOSS="w_mse"                # mse / w_mse
G_SCALE=2.0             # restoration guidance strength
G_START=1001                  # guidance active when t < G_START
G_STOP=1                     # guidance active when t > G_STOP
G_SPACE="rgb"              # latent / rgb
G_REPEAT=1                    # guidance updates per step

# ---------- Metric evaluation ----------
EVAL_METRICS=true             # true / false
GT_DIR="GT"                   # GT image directory (same filenames as outputs)
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
echo "EVAL_METRICS=$EVAL_METRICS"
echo "GT_DIR=$GT_DIR"
echo "RESIZE_PRED_TO_GT=$RESIZE_PRED_TO_GT"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

# ==========================================
# Run metric evaluation
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