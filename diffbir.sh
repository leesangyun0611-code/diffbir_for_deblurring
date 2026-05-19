#!/usr/bin/env bash
set -eo pipefail

# ==========================================
# DiffBIR single experiment launcher
# Run with: bash diffbir.sh
# ==========================================

# ---------- Device ----------
GPU=${GPU:-0}
DEVICE=${DEVICE:-cuda}                 # cpu / cuda / mps
PRECISION=${PRECISION:-fp16}           # fp32 / fp16 / bf16
SEED=${SEED:-42}

# ---------- Current experiment ----------
EXPERIMENT_NAME=${EXPERIMENT_NAME:-text_guidance_test}
TASK=${TASK:-sr}                       # sr / face / denoise / unaligned_face
GUIDANCE=${GUIDANCE:-false}            # true / false
G_LOSS=${G_LOSS:-w_mse}                # mse / w_mse
G_SCALE=${G_SCALE:-0.0}
G_SPACE=${G_SPACE:-rgb}                # latent / rgb
G_WEIGHT_MODE=${G_WEIGHT_MODE:-highfreq} # lowfreq / highfreq
G_WEIGHT_GAMMA=${G_WEIGHT_GAMMA:-1.0}

# ---------- Task / version ----------
VERSION=${VERSION:-v2.1}               # v1 / v2 / v2.1 / custom
UPSCALE=${UPSCALE:-1}

# ---------- Paths ----------
INPUT_DIR=${INPUT:-${INPUT_DIR:-inputs/demo/mytest}}
GT_DIR=${GT_DIR:-}
RESULTS_ROOT=${RESULTS_ROOT:-results}
OUTPUT_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
OUTPUT_DIR=${OUTPUT:-${OUTPUT_DIR:-results/text_guidance_test}}

# ---------- Optional input resize ----------
AUTO_RESIZE=${AUTO_RESIZE:-false}      # true / false
MAX_INPUT_SIDE=${MAX_INPUT_SIDE:-511}  # only used when AUTO_RESIZE=true

# ---------- Sampling ----------
SAMPLER=${SAMPLER:-spaced}             # spaced / ddim / dpm++_m2 / edm_euler / ...
STEPS=${STEPS:-50}
CFG_SCALE=${CFG_SCALE:-4}
RESCALE_CFG=${RESCALE_CFG:-false}      # true / false

# ---------- Start point ----------
START_POINT_TYPE=${START_POINT_TYPE:-cond} # cond / noise
START_POINT_NOISE_SCALE=${START_POINT_NOISE_SCALE:-} # e.g. 0.3, 1.0 ; leave empty to disable

# ---------- Batch / outputs ----------
N_SAMPLES=${N_SAMPLES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}

# ---------- Prompt / caption ----------
CAPTIONER=${CAPTIONER:-none}           # none / llava / ram
POS_PROMPT=${POS_PROMPT:-"a sharp, clean, natural photo, no motion blur, high detail"}
NEG_PROMPT=${NEG_PROMPT:-"motion blur, blurry, ghosting, smear, low quality, artifacts"}

# ---------- Tile toggles ----------
CLEANER_TILED=${CLEANER_TILED:-false}
CLDM_TILED=${CLDM_TILED:-false}
VAE_ENCODER_TILED=${VAE_ENCODER_TILED:-false}
VAE_DECODER_TILED=${VAE_DECODER_TILED:-false}

# ---------- Tile sizes / strides ----------
CLEANER_TILE_SIZE=${CLEANER_TILE_SIZE:-512}
CLEANER_TILE_STRIDE=${CLEANER_TILE_STRIDE:-256}

CLDM_TILE_SIZE=${CLDM_TILE_SIZE:-256}
CLDM_TILE_STRIDE=${CLDM_TILE_STRIDE:-128}

VAE_ENCODER_TILE_SIZE=${VAE_ENCODER_TILE_SIZE:-1024}
VAE_DECODER_TILE_SIZE=${VAE_DECODER_TILE_SIZE:-256}

# ---------- Restormer stage-1 cleaner ----------
# Roll back to the original stage-1 path with: STAGE1_MODEL=default
STAGE1_MODEL=${STAGE1_MODEL:-swinir}   # options: default, swinir, nafnet, restormer, mprnet
STAGE1_CKPT=${STAGE1_CKPT:-}
STAGE1_CONFIG=${STAGE1_CONFIG:-}
CLEANER_TYPE=${CLEANER_TYPE:-default} # legacy alias used by existing code
RESTORMER_REPO=${RESTORMER_REPO:-third_party/Restormer}
RESTORMER_TASK=${RESTORMER_TASK:-Motion_Deblurring}
RESTORMER_CKPT=${RESTORMER_CKPT:-${RESTORMER_REPO}/Motion_Deblurring/pretrained_models/motion_deblurring.pth}
NAFNET_REPO=${NAFNET_REPO:-third_party/NAFNet}
NAFNET_CKPT=${NAFNET_CKPT:-weights/NAFNet-GoPro-width64.pth}
MPRNET_REPO=${MPRNET_REPO:-third_party/MPRNet}
MPRNET_CKPT=${MPRNET_CKPT:-weights/MPRNet-Deblurring.pth}

# ---------- Text-aware guidance ----------
TEXT_GUIDANCE=${TEXT_GUIDANCE:-true}
TEXT_DETECTOR=${TEXT_DETECTOR:-easyocr}
TEXT_MASK_SOURCE=${TEXT_MASK_SOURCE:-union}
TEXT_GUIDANCE_SCALE=${TEXT_GUIDANCE_SCALE:-0.3}
TEXT_RGB_WEIGHT=${TEXT_RGB_WEIGHT:-0.1}
TEXT_EDGE_WEIGHT=${TEXT_EDGE_WEIGHT:-1.0}
TEXT_GUIDANCE_START=${TEXT_GUIDANCE_START:-0.0}
TEXT_GUIDANCE_STOP=${TEXT_GUIDANCE_STOP:-0.35}
TEXT_GUIDANCE_MODE=${TEXT_GUIDANCE_MODE:-late}
TEXT_GRAD_CLIP=${TEXT_GRAD_CLIP:-0.05}
TEXT_GUIDANCE_LOSS=${TEXT_GUIDANCE_LOSS:-edge_rgb}
TEXT_MASK_DILATE=${TEXT_MASK_DILATE:-5}
TEXT_MASK_BLUR=${TEXT_MASK_BLUR:-1.0}
TEXT_MIN_CONFIDENCE=${TEXT_MIN_CONFIDENCE:-0.3}
TEXT_MIN_AREA=${TEXT_MIN_AREA:-0}
EASYOCR_LANGS=${EASYOCR_LANGS:-ko,en}
EASYOCR_TEXT_THRESHOLD=${EASYOCR_TEXT_THRESHOLD:-0.4}
EASYOCR_LOW_TEXT=${EASYOCR_LOW_TEXT:-0.2}
EASYOCR_LINK_THRESHOLD=${EASYOCR_LINK_THRESHOLD:-0.4}
EASYOCR_CANVAS_SIZE=${EASYOCR_CANVAS_SIZE:-2560}
EASYOCR_MAG_RATIO=${EASYOCR_MAG_RATIO:-2.0}
SAVE_TEXT_MASK=${SAVE_TEXT_MASK:-true}
TEXT_DEBUG_DIR=${TEXT_DEBUG_DIR:-}

# ---------- Optional noise / sampler extras ----------
NOISE_AUG=${NOISE_AUG:-0}
ETA=${ETA:-0}
ORDER=${ORDER:-2}
STRENGTH=${STRENGTH:-1.0}
S_CHURN=${S_CHURN:-0}
S_TMIN=${S_TMIN:-0}
S_TMAX=${S_TMAX:-999}
S_NOISE=${S_NOISE:-1.0}

# ---------- Guidance shared options ----------
G_START=${G_START:-1001}
G_STOP=${G_STOP:-1}
G_REPEAT=${G_REPEAT:-1}
G_WEIGHT_FLOOR=${G_WEIGHT_FLOOR:-0.0}
G_BLOCK_SIZE=${G_BLOCK_SIZE:-2}

# ---------- MANIQA evaluation ----------
EVAL_MANIQA=${EVAL_MANIQA:-false}      # true / false
MANIQA_MODEL=${MANIQA_MODEL:-maniqa-pipal} # maniqa / maniqa-kadid / maniqa-pipal
IQA_CSV=${IQA_CSV:-iqa_results.csv}

# ---------- LPIPS evaluation ----------
EVAL_LPIPS=${EVAL_LPIPS:-false}        # true / false
LPIPS_MODEL=${LPIPS_MODEL:-alex}       # alex / vgg

# ---------- GT-based metric evaluation ----------
EVAL_METRICS=${EVAL_METRICS:-false}    # true / false
RESIZE_PRED_TO_GT=${RESIZE_PRED_TO_GT:-false} # true / false
METRICS_CSV=${METRICS_CSV:-metrics_psnr_ssim.csv}

# ---------- Custom model paths (optional) ----------
TRAIN_CFG=${TRAIN_CFG:-}
CKPT=${CKPT:-}
LORA_CKPT=${LORA_CKPT:-}

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
CMD+=(--stage1_model "$STAGE1_MODEL")
CMD+=(--cleaner_type "$CLEANER_TYPE")
if [ -n "$STAGE1_CKPT" ]; then
  CMD+=(--stage1_ckpt "$STAGE1_CKPT")
fi
if [ -n "$STAGE1_CONFIG" ]; then
  CMD+=(--stage1_config "$STAGE1_CONFIG")
fi
if [ "$STAGE1_MODEL" = "restormer" ] || [ "$CLEANER_TYPE" = "restormer" ]; then
  CMD+=(--restormer_repo "$RESTORMER_REPO")
  CMD+=(--restormer_task "$RESTORMER_TASK")
  CMD+=(--restormer_ckpt "${STAGE1_CKPT:-$RESTORMER_CKPT}")
elif [ "$STAGE1_MODEL" = "nafnet" ] || [ "$CLEANER_TYPE" = "nafnet" ]; then
  CMD+=(--nafnet_repo "$NAFNET_REPO")
  CMD+=(--nafnet_ckpt "${STAGE1_CKPT:-$NAFNET_CKPT}")
elif [ "$STAGE1_MODEL" = "mprnet" ] || [ "$CLEANER_TYPE" = "mprnet" ]; then
  CMD+=(--mprnet_repo "$MPRNET_REPO")
  CMD+=(--mprnet_ckpt "${STAGE1_CKPT:-$MPRNET_CKPT}")
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

if [ "$TEXT_GUIDANCE" = true ]; then
  CMD+=(--text_guidance)
  CMD+=(--text_detector "$TEXT_DETECTOR")
  CMD+=(--text_mask_source "$TEXT_MASK_SOURCE")
  CMD+=(--text_guidance_scale "$TEXT_GUIDANCE_SCALE")
  CMD+=(--text_rgb_weight "$TEXT_RGB_WEIGHT")
  CMD+=(--text_edge_weight "$TEXT_EDGE_WEIGHT")
  CMD+=(--text_guidance_start "$TEXT_GUIDANCE_START")
  CMD+=(--text_guidance_stop "$TEXT_GUIDANCE_STOP")
  CMD+=(--text_guidance_mode "$TEXT_GUIDANCE_MODE")
  CMD+=(--text_grad_clip "$TEXT_GRAD_CLIP")
  CMD+=(--text_guidance_loss "$TEXT_GUIDANCE_LOSS")
  CMD+=(--text_mask_dilate "$TEXT_MASK_DILATE")
  CMD+=(--text_mask_blur "$TEXT_MASK_BLUR")
  CMD+=(--text_min_confidence "$TEXT_MIN_CONFIDENCE")
  CMD+=(--text_min_area "$TEXT_MIN_AREA")
  CMD+=(--easyocr_langs "$EASYOCR_LANGS")
  CMD+=(--easyocr_text_threshold "$EASYOCR_TEXT_THRESHOLD")
  CMD+=(--easyocr_low_text "$EASYOCR_LOW_TEXT")
  CMD+=(--easyocr_link_threshold "$EASYOCR_LINK_THRESHOLD")
  CMD+=(--easyocr_canvas_size "$EASYOCR_CANVAS_SIZE")
  CMD+=(--easyocr_mag_ratio "$EASYOCR_MAG_RATIO")
  if [ "$SAVE_TEXT_MASK" = true ]; then
    CMD+=(--save_text_mask)
  fi
  if [ -n "$TEXT_DEBUG_DIR" ]; then
    CMD+=(--text_debug_dir "$TEXT_DEBUG_DIR")
  fi
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
echo "STAGE1_MODEL=$STAGE1_MODEL"
echo "STAGE1_CKPT=$STAGE1_CKPT"
echo "STAGE1_CONFIG=$STAGE1_CONFIG"
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
echo "TEXT_GUIDANCE=$TEXT_GUIDANCE"
echo "TEXT_DETECTOR=$TEXT_DETECTOR"
echo "TEXT_MASK_SOURCE=$TEXT_MASK_SOURCE"
echo "TEXT_GUIDANCE_SCALE=$TEXT_GUIDANCE_SCALE"
echo "TEXT_RGB_WEIGHT=$TEXT_RGB_WEIGHT"
echo "TEXT_EDGE_WEIGHT=$TEXT_EDGE_WEIGHT"
echo "TEXT_GUIDANCE_START=$TEXT_GUIDANCE_START"
echo "TEXT_GUIDANCE_STOP=$TEXT_GUIDANCE_STOP"
echo "TEXT_GUIDANCE_MODE=$TEXT_GUIDANCE_MODE"
echo "TEXT_GRAD_CLIP=$TEXT_GRAD_CLIP"
echo "TEXT_MASK_DILATE=$TEXT_MASK_DILATE"
echo "TEXT_MASK_BLUR=$TEXT_MASK_BLUR"
echo "TEXT_MIN_CONFIDENCE=$TEXT_MIN_CONFIDENCE"
echo "TEXT_MIN_AREA=$TEXT_MIN_AREA"
echo "EASYOCR_LANGS=$EASYOCR_LANGS"
echo "EASYOCR_TEXT_THRESHOLD=$EASYOCR_TEXT_THRESHOLD"
echo "EASYOCR_LOW_TEXT=$EASYOCR_LOW_TEXT"
echo "EASYOCR_LINK_THRESHOLD=$EASYOCR_LINK_THRESHOLD"
echo "EASYOCR_CANVAS_SIZE=$EASYOCR_CANVAS_SIZE"
echo "EASYOCR_MAG_RATIO=$EASYOCR_MAG_RATIO"
echo "SAVE_TEXT_MASK=$SAVE_TEXT_MASK"
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
