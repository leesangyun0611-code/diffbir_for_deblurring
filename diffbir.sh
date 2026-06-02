#!/usr/bin/env bash
set -eo pipefail

# ==========================================
# DiffBIR experiment launcher
# EXPERIMENT=1: exp3_ baseline Stage2
# EXPERIMENT=2: exp4_ Stage2 + text guidance
# EXPERIMENT=3: exp5_ Stage2 non-text + Stage1 text blend
# ==========================================

# ---------- Core ----------
EXPERIMENT=${EXPERIMENT:-2}            # 실험 선택, options: 1=baseline Stage2 / 2=text guidance / 3=Stage1 text blend
GPU=${GPU:-1}                          # 사용할 GPU id, 예: 0 / 1
DEVICE=${DEVICE:-cuda}                 # 실행 device, options: cuda / cpu / mps
PRECISION=${PRECISION:-fp16}           # 연산 precision, options: fp16 / fp32 / bf16
SEED=${SEED:-42}                       # random seed, 결과 재현성 제어

INPUT_DIR=${INPUT:-${INPUT_DIR:-inputs/demo/input_test_one}} # input image folder
GT_DIR=${GT_DIR:-GT/GT_one} # GT folder, PSNR/SSIM/LPIPS 평가용
RESULTS_DIR=${RESULTS_DIR:-results}    # 결과 root directory

TASK=${TASK:-sr}                       # DiffBIR task, options: sr / denoise / face / unaligned_face
VERSION=${VERSION:-v2.1}               # DiffBIR version, options: v1 / v2 / v2.1
UPSCALE=${UPSCALE:-1}                  # SR upscale factor
STAGE1_MODEL=${STAGE1_MODEL:-restormer} # Stage1 restoration model, options: default / swinir / restormer / nafnet / mprnet
LORA_CKPT=${LORA_CKPT-experiments/stage2_lora_text_train_20260528/checkpoints/lora_final.pt} # optional ControlNet LoRA checkpoint. Set LORA_CKPT="" to disable LoRA.

SAMPLER=${SAMPLER:-spaced}             # Stage2 sampler, text guidance는 현재 spaced에서 적용
STEPS=${STEPS:-50}                     # denoising steps, 클수록 느림
CFG_SCALE=${CFG_SCALE:-2.0}            # prompt guidance scale, 낮추면 hallucination 감소 가능
START_POINT_TYPE=${START_POINT_TYPE:-cond} # Stage2 start point, options: cond / noise
STRENGTH=${STRENGTH:-0.7}              # ControlNet strength, 낮추면 Stage2 자유도 감소

# ---------- Existing restoration guidance ----------
GUIDANCE=${GUIDANCE:-true}             # 기존 Restoration Guidance 사용 여부, options: true / false
G_LOSS=${G_LOSS:-w_mse}                # restoration guidance loss, options: mse / w_mse
G_SCALE=${G_SCALE:-1.0}                # restoration guidance strength
G_SPACE=${G_SPACE:-rgb}                # guidance 계산 space, options: rgb / latent
G_WEIGHT_MODE=${G_WEIGHT_MODE:-highfreq} # w_mse weight 방향, options: highfreq / lowfreq
G_WEIGHT_GAMMA=${G_WEIGHT_GAMMA:-1.0}  # w_mse weight contrast, 클수록 weight 차이 증가
G_REPEAT=${G_REPEAT:-1}                # step당 guidance update 반복 횟수, 클수록 느림

# ---------- Text guidance for EXPERIMENT=2, optional in EXPERIMENT=3 ----------
TEXT_GUIDANCE_SCALE=${TEXT_GUIDANCE_SCALE:-0.8} # text guidance 전체 strength
TEXT_RGB_WEIGHT=${TEXT_RGB_WEIGHT:-1.0} # text RGB/color consistency weight
TEXT_EDGE_WEIGHT=${TEXT_EDGE_WEIGHT:-3.0} # text stroke/edge consistency weight
TEXT_GUIDANCE_MODE=${TEXT_GUIDANCE_MODE:-late} # text guidance schedule, options: all / late / early / fraction
TEXT_GUIDANCE_STOP=${TEXT_GUIDANCE_STOP:-0.35} # late에서는 마지막 fraction만 text guidance 적용
TEXT_GRAD_CLIP=${TEXT_GRAD_CLIP:-0.05} # text gradient clipping, 과보정/NaN 방지

TEXT_REGIONAL_NOISE=${TEXT_REGIONAL_NOISE:-true} # text/non-text 시작 noise 분리, options: true / false
TEXT_NOISE_TIMESTEP_RATIO=${TEXT_NOISE_TIMESTEP_RATIO:-${TEXT_NOISE_RATIO:-0.01}} # text region start noise timestep ratio, 낮을수록 Stage1 text 보존
NONTEXT_NOISE_TIMESTEP_RATIO=${NONTEXT_NOISE_TIMESTEP_RATIO:-${NON_TEXT_NOISE_RATIO:-1.0}} # non-text start noise ratio, 1.0이면 기존 full-noise
TEXT_NOISE_SCALE=${TEXT_NOISE_SCALE:-0.2} # text region random noise scale, 낮을수록 text 보존
NONTEXT_NOISE_SCALE=${NONTEXT_NOISE_SCALE:-1.0} # non-text random noise scale

TEXT_LATENT_ANCHOR=${TEXT_LATENT_ANCHOR:-true} # 매 step 후 text latent를 Stage1 latent에 anchoring, options: true / false
TEXT_LATENT_ANCHOR_ALPHA=${TEXT_LATENT_ANCHOR_ALPHA:-0.85} # anchoring strength, 1.0에 가까울수록 Stage1 text 유지

# ---------- Text detection ----------
TEXT_SPOTTING_MODULE=${TEXT_SPOTTING_MODULE:-easyocr} # EXPERIMENT=2 text spotting backend, 현재 options: easyocr
TEXT_MASK_SOURCE=${TEXT_MASK_SOURCE:-union} # mask source, options: union / stage1 / lq
TEXT_MASK_DILATE=${TEXT_MASK_DILATE:-2} # text mask dilation pixel, 글자 주변까지 보호
TEXT_MASK_BLUR=${TEXT_MASK_BLUR:-1.5}  # soft mask blur sigma, 경계 완화
TEXT_MIN_CONFIDENCE=${TEXT_MIN_CONFIDENCE:-0.30} # EasyOCR confidence threshold, 낮을수록 recall 증가
TEXT_MIN_AREA=${TEXT_MIN_AREA:-8}      # 최소 text polygon area, 작은 글자면 0 권장
EASYOCR_LANGS=${EASYOCR_LANGS:-ko,en}  # EasyOCR language hints, 예: ko,en / en
EASYOCR_TEXT_THRESHOLD=${EASYOCR_TEXT_THRESHOLD:-0.2} # EasyOCR text threshold, 낮을수록 흐린 글자 검출 증가
EASYOCR_LOW_TEXT=${EASYOCR_LOW_TEXT:-0.1} # EasyOCR low_text threshold, 낮을수록 작은/약한 글자 검출 증가
EASYOCR_LINK_THRESHOLD=${EASYOCR_LINK_THRESHOLD:-0.2} # EasyOCR character link threshold
EASYOCR_CANVAS_SIZE=${EASYOCR_CANVAS_SIZE:-3200} # EasyOCR canvas size, 클수록 작은 글자 유리하지만 느림
EASYOCR_MAG_RATIO=${EASYOCR_MAG_RATIO:-3.0} # EasyOCR magnification ratio, 클수록 작은 글자 유리하지만 느림

# ---------- EXPERIMENT=3 text blend ----------
EXP5_USE_TEXT_GUIDANCE=${EXP5_USE_TEXT_GUIDANCE:-false} # EXPERIMENT=3의 Stage2에도 text guidance 적용할지, options: true / false
TEXT_BLEND_SPOTTING_MODULE=${TEXT_BLEND_SPOTTING_MODULE:-paddleocr} # EXPERIMENT=3 blend용 OCR backend, 현재 options: paddleocr
TEXT_BLEND_DET_SOURCE=${TEXT_BLEND_DET_SOURCE:-stage1} # blend box 검출 source, options: stage1 / stage2
TEXT_BLEND_OCR_LANGS=${TEXT_BLEND_OCR_LANGS:-korean,en} # PaddleOCR languages, 예: korean,en / en
TEXT_BLEND_OCR_SCALE=${TEXT_BLEND_OCR_SCALE:-3.0} # OCR 입력 확대 배율, 작은 글자 검출용
TEXT_BLEND_MIN_CONF=${TEXT_BLEND_MIN_CONF:-0.10} # PaddleOCR confidence threshold
TEXT_BLEND_MIN_AREA=${TEXT_BLEND_MIN_AREA:-40} # 최소 box area, 작게 하면 작은 글자 recall 증가
TEXT_BLEND_PAD_RATIO=${TEXT_BLEND_PAD_RATIO:-0.20} # box padding ratio, 글자 주변까지 Stage1 blend
TEXT_BLEND_FEATHER=${TEXT_BLEND_FEATHER:-5} # blend mask feather radius, 경계 부드럽게
TEXT_BLEND_USE_GPU_OCR=${TEXT_BLEND_USE_GPU_OCR:-false} # PaddleOCR GPU 사용 여부

# ---------- Metrics ----------
EVAL_METRICS=${EVAL_METRICS:-true}     # PSNR/SSIM 평가, options: true / false
RESIZE_PRED_TO_GT=${RESIZE_PRED_TO_GT:-false} # metric 전 pred를 GT 크기로 resize, options: true / false
EVAL_MANIQA=${EVAL_MANIQA:-false}      # MANIQA 평가, options: true / false
EVAL_LPIPS=${EVAL_LPIPS:-false}        # LPIPS 평가, options: true / false
LPIPS_MODEL=${LPIPS_MODEL:-alex}       # LPIPS backbone, options: alex / vgg
MANIQA_MODEL=${MANIQA_MODEL:-maniqa-pipal} # MANIQA model, options: maniqa / maniqa-kadid / maniqa-pipal

# ---------- Less frequently changed paths/checkpoints ----------
RESTORMER_REPO=${RESTORMER_REPO:-third_party/Restormer}
RESTORMER_TASK=${RESTORMER_TASK:-Motion_Deblurring}
RESTORMER_CKPT=${RESTORMER_CKPT:-${RESTORMER_REPO}/Motion_Deblurring/pretrained_models/motion_deblurring.pth}
NAFNET_REPO=${NAFNET_REPO:-third_party/NAFNet}
NAFNET_CKPT=${NAFNET_CKPT:-weights/NAFNet-GoPro-width64.pth}
MPRNET_REPO=${MPRNET_REPO:-third_party/MPRNet}
MPRNET_CKPT=${MPRNET_CKPT:-weights/MPRNet-Deblurring.pth}
STAGE1_CKPT=${STAGE1_CKPT:-}

CLEANER_TILED=${CLEANER_TILED:-false}
CLDM_TILED=${CLDM_TILED:-false}
VAE_ENCODER_TILED=${VAE_ENCODER_TILED:-false}
VAE_DECODER_TILED=${VAE_DECODER_TILED:-false}
CLEANER_TILE_SIZE=${CLEANER_TILE_SIZE:-512}
CLEANER_TILE_STRIDE=${CLEANER_TILE_STRIDE:-256}
CLDM_TILE_SIZE=${CLDM_TILE_SIZE:-256}
CLDM_TILE_STRIDE=${CLDM_TILE_STRIDE:-128}
VAE_ENCODER_TILE_SIZE=${VAE_ENCODER_TILE_SIZE:-1024}
VAE_DECODER_TILE_SIZE=${VAE_DECODER_TILE_SIZE:-256}

N_SAMPLES=${N_SAMPLES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
CAPTIONER=${CAPTIONER:-none}
POS_PROMPT=${POS_PROMPT:-"a sharp, clean, natural photo, no motion blur, high detail"}
NEG_PROMPT=${NEG_PROMPT:-"motion blur, blurry, ghosting, smear, low quality, artifacts"}
NOISE_AUG=${NOISE_AUG:-0}
ETA=${ETA:-0}
ORDER=${ORDER:-2}
S_CHURN=${S_CHURN:-0}
S_TMIN=${S_TMIN:-0}
S_TMAX=${S_TMAX:-999}
S_NOISE=${S_NOISE:-1.0}

# ---------- Mode selection ----------
case "$EXPERIMENT" in
  1)
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-exp3_1_baseline}
    TEXT_GUIDANCE=false
    PRESERVE_TEXT=false
    ;;
  2)
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-exp5_1_conservative_text_anchor}
    TEXT_GUIDANCE=true
    PRESERVE_TEXT=false
    ;;
  3)
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-exp5_1_text_blend}
    TEXT_GUIDANCE=$EXP5_USE_TEXT_GUIDANCE
    PRESERVE_TEXT=true
    ;;
  *)
    echo "[ERROR] EXPERIMENT must be 1, 2, or 3. Got: $EXPERIMENT"
    exit 1
    ;;
esac

EXPERIMENT_DIR=${OUTPUT:-${RESULTS_DIR}/${EXPERIMENT_NAME}}
RESULT_DIR=${EXPERIMENT_DIR}/result
TEXT_DETECTION_DIR=${EXPERIMENT_DIR}/text_detection
METRICS_DIR=${EXPERIMENT_DIR}/metrics
STAGE2_RAW_DIR=${EXPERIMENT_DIR}/stage2_raw
STAGE1_BLEND_DIR=${EXPERIMENT_DIR}/stage1

if [ "$PRESERVE_TEXT" = true ]; then
  OUTPUT_DIR="$STAGE2_RAW_DIR"
  FINAL_PRED_DIR="$RESULT_DIR"
else
  OUTPUT_DIR="$RESULT_DIR"
  FINAL_PRED_DIR="$RESULT_DIR"
fi

mkdir -p "$RESULT_DIR" "$TEXT_DETECTION_DIR" "$METRICS_DIR"

# ---------- Build Stage2 command ----------
CMD=(python inference.py)
CMD+=(--task "$TASK" --version "$VERSION" --upscale "$UPSCALE")
CMD+=(--input "$INPUT_DIR" --output "$OUTPUT_DIR")
CMD+=(--n_samples "$N_SAMPLES" --batch_size "$BATCH_SIZE")
CMD+=(--sampler "$SAMPLER" --steps "$STEPS" --cfg_scale "$CFG_SCALE")
CMD+=(--start_point_type "$START_POINT_TYPE")
CMD+=(--captioner "$CAPTIONER" --precision "$PRECISION" --seed "$SEED" --device "$DEVICE")
CMD+=(--stage1_model "$STAGE1_MODEL" --cleaner_type default)

if [ -n "$LORA_CKPT" ]; then
  CMD+=(--lora_ckpt "$LORA_CKPT")
fi

if [ -n "$STAGE1_CKPT" ]; then
  CMD+=(--stage1_ckpt "$STAGE1_CKPT")
fi

if [ "$STAGE1_MODEL" = "restormer" ]; then
  CMD+=(--restormer_repo "$RESTORMER_REPO" --restormer_task "$RESTORMER_TASK")
  CMD+=(--restormer_ckpt "${STAGE1_CKPT:-$RESTORMER_CKPT}")
elif [ "$STAGE1_MODEL" = "nafnet" ]; then
  CMD+=(--nafnet_repo "$NAFNET_REPO" --nafnet_ckpt "${STAGE1_CKPT:-$NAFNET_CKPT}")
elif [ "$STAGE1_MODEL" = "mprnet" ]; then
  CMD+=(--mprnet_repo "$MPRNET_REPO" --mprnet_ckpt "${STAGE1_CKPT:-$MPRNET_CKPT}")
fi

if [ "$CLEANER_TILED" = true ]; then
  CMD+=(--cleaner_tiled --cleaner_tile_size "$CLEANER_TILE_SIZE" --cleaner_tile_stride "$CLEANER_TILE_STRIDE")
fi
if [ "$CLDM_TILED" = true ]; then
  CMD+=(--cldm_tiled --cldm_tile_size "$CLDM_TILE_SIZE" --cldm_tile_stride "$CLDM_TILE_STRIDE")
fi
if [ "$VAE_ENCODER_TILED" = true ]; then
  CMD+=(--vae_encoder_tiled --vae_encoder_tile_size "$VAE_ENCODER_TILE_SIZE")
fi
if [ "$VAE_DECODER_TILED" = true ]; then
  CMD+=(--vae_decoder_tiled --vae_decoder_tile_size "$VAE_DECODER_TILE_SIZE")
fi

if [ "$GUIDANCE" = true ]; then
  CMD+=(--guidance --g_loss "$G_LOSS" --g_scale "$G_SCALE")
  CMD+=(--g_start 1001 --g_stop 1 --g_space "$G_SPACE" --g_repeat "$G_REPEAT")
  CMD+=(--g_weight_mode "$G_WEIGHT_MODE" --g_weight_gamma "$G_WEIGHT_GAMMA")
fi

if [ "$TEXT_GUIDANCE" = true ]; then
  CMD+=(--text_guidance --text_detector "$TEXT_SPOTTING_MODULE" --text_mask_source "$TEXT_MASK_SOURCE")
  CMD+=(--text_guidance_scale "$TEXT_GUIDANCE_SCALE")
  CMD+=(--text_rgb_weight "$TEXT_RGB_WEIGHT" --text_edge_weight "$TEXT_EDGE_WEIGHT")
  CMD+=(--text_guidance_start 0.0 --text_guidance_stop "$TEXT_GUIDANCE_STOP")
  CMD+=(--text_guidance_mode "$TEXT_GUIDANCE_MODE" --text_grad_clip "$TEXT_GRAD_CLIP")
  CMD+=(--text_guidance_loss edge_rgb)
  CMD+=(--text_mask_dilate "$TEXT_MASK_DILATE" --text_mask_blur "$TEXT_MASK_BLUR")
  CMD+=(--text_min_confidence "$TEXT_MIN_CONFIDENCE" --text_min_area "$TEXT_MIN_AREA")
  CMD+=(--easyocr_langs "$EASYOCR_LANGS")
  CMD+=(--easyocr_text_threshold "$EASYOCR_TEXT_THRESHOLD")
  CMD+=(--easyocr_low_text "$EASYOCR_LOW_TEXT")
  CMD+=(--easyocr_link_threshold "$EASYOCR_LINK_THRESHOLD")
  CMD+=(--easyocr_canvas_size "$EASYOCR_CANVAS_SIZE")
  CMD+=(--easyocr_mag_ratio "$EASYOCR_MAG_RATIO")
  CMD+=(--save_text_mask --text_debug_dir "$TEXT_DETECTION_DIR")

  if [ "$TEXT_REGIONAL_NOISE" = true ]; then
    CMD+=(--text_regional_noise)
  fi
  CMD+=(--text_noise_timestep_ratio "$TEXT_NOISE_TIMESTEP_RATIO")
  CMD+=(--nontext_noise_timestep_ratio "$NONTEXT_NOISE_TIMESTEP_RATIO")
  CMD+=(--text_noise_scale "$TEXT_NOISE_SCALE")
  CMD+=(--nontext_noise_scale "$NONTEXT_NOISE_SCALE")

  if [ "$TEXT_LATENT_ANCHOR" = true ]; then
    CMD+=(--text_latent_anchor)
  fi
  CMD+=(--text_latent_anchor_alpha "$TEXT_LATENT_ANCHOR_ALPHA")
fi

if [ "$EVAL_MANIQA" = true ]; then
  CMD+=(--eval_maniqa --maniqa_model "$MANIQA_MODEL")
fi
if [ "$EVAL_LPIPS" = true ] && [ -n "$GT_DIR" ]; then
  CMD+=(--eval_lpips --lpips_model "$LPIPS_MODEL" --gt_dir "$GT_DIR")
fi
if [ "$EVAL_MANIQA" = true ] || [ "$EVAL_LPIPS" = true ]; then
  CMD+=(--iqa_csv "../metrics/iqa_results.csv")
fi

CMD+=(--noise_aug "$NOISE_AUG" --eta "$ETA" --order "$ORDER" --strength "$STRENGTH")
CMD+=(--s_churn "$S_CHURN" --s_tmin "$S_TMIN" --s_tmax "$S_TMAX" --s_noise "$S_NOISE")
CMD+=(--pos_prompt "$POS_PROMPT" --neg_prompt "$NEG_PROMPT")

echo "=========================================="
echo "DiffBIR experiment"
echo "EXPERIMENT=$EXPERIMENT"
echo "EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "TEXT_DETECTION_DIR=$TEXT_DETECTION_DIR"
echo "METRICS_DIR=$METRICS_DIR"
echo "TEXT_GUIDANCE=$TEXT_GUIDANCE"
echo "PRESERVE_TEXT=$PRESERVE_TEXT"
echo "STAGE1_MODEL=$STAGE1_MODEL"
echo "INPUT_DIR=$INPUT_DIR"
echo "GT_DIR=$GT_DIR"
echo "LORA_CKPT=$LORA_CKPT"
echo "=========================================="

# ---------- EXPERIMENT=3: Stage1 image for text replacement ----------
if [ "$PRESERVE_TEXT" = true ]; then
  STAGE1_CLEANER_TYPE=default
  if [ "$STAGE1_MODEL" = "restormer" ] || [ "$STAGE1_MODEL" = "nafnet" ] || [ "$STAGE1_MODEL" = "mprnet" ]; then
    STAGE1_CLEANER_TYPE="$STAGE1_MODEL"
  fi

  STAGE1_CMD=(
    python run_stage1_only.py
    --input "$INPUT_DIR"
    --output "$STAGE1_BLEND_DIR"
    --gt_dir "$GT_DIR"
    --task "$TASK"
    --version "$VERSION"
    --device "$DEVICE"
    --precision "$PRECISION"
    --seed "$SEED"
    --upscale "$UPSCALE"
    --cleaner_type "$STAGE1_CLEANER_TYPE"
    --restormer_repo "$RESTORMER_REPO"
    --restormer_task "$RESTORMER_TASK"
    --restormer_ckpt "${STAGE1_CKPT:-$RESTORMER_CKPT}"
    --nafnet_repo "$NAFNET_REPO"
    --nafnet_ckpt "${STAGE1_CKPT:-$NAFNET_CKPT}"
    --mprnet_repo "$MPRNET_REPO"
    --mprnet_ckpt "${STAGE1_CKPT:-$MPRNET_CKPT}"
  )

  if [ "$CLEANER_TILED" = true ]; then
    STAGE1_CMD+=(--cleaner_tiled --cleaner_tile_size "$CLEANER_TILE_SIZE" --cleaner_tile_stride "$CLEANER_TILE_STRIDE")
  fi

  echo "Generating Stage1 images for EXPERIMENT=3..."
  CUDA_VISIBLE_DEVICES="$GPU" "${STAGE1_CMD[@]}"
fi

# ---------- Stage2 ----------
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

# ---------- EXPERIMENT=3: blend Stage1 text with Stage2 non-text ----------
if [ "$PRESERVE_TEXT" = true ]; then
  TEXT_BLEND_CMD=(
    python scripts/blend_stage1_text_regions.py
    --stage1_dir "$STAGE1_BLEND_DIR"
    --stage2_dir "$STAGE2_RAW_DIR"
    --output_dir "$RESULT_DIR"
    --det_source "$TEXT_BLEND_DET_SOURCE"
    --langs "$TEXT_BLEND_OCR_LANGS"
    --ocr_scale "$TEXT_BLEND_OCR_SCALE"
    --min_conf "$TEXT_BLEND_MIN_CONF"
    --min_area "$TEXT_BLEND_MIN_AREA"
    --pad_ratio "$TEXT_BLEND_PAD_RATIO"
    --feather "$TEXT_BLEND_FEATHER"
    --diagnostics_dir "$TEXT_DETECTION_DIR"
    --save_diagnostics
  )
  if [ "$TEXT_BLEND_USE_GPU_OCR" = true ]; then
    TEXT_BLEND_CMD+=(--use_gpu_ocr)
  fi

  echo "Blending Stage1 text regions into Stage2 output..."
  CUDA_VISIBLE_DEVICES="$GPU" "${TEXT_BLEND_CMD[@]}"
fi

# ---------- Metrics ----------
if [ "$EVAL_METRICS" = true ] && [ -n "$GT_DIR" ]; then
  METRICS_CMD=(
    python eval_metrics.py
    --pred_dir "$FINAL_PRED_DIR"
    --gt_dir "$GT_DIR"
    --save_csv "$METRICS_DIR/metrics_psnr_ssim.csv"
  )
  if [ "$RESIZE_PRED_TO_GT" = true ]; then
    METRICS_CMD+=(--resize_pred_to_gt)
  fi

  echo "Running PSNR/SSIM metrics..."
  "${METRICS_CMD[@]}"
fi

echo "=========================================="
echo "Done"
echo "Result images: $RESULT_DIR"
echo "Text detection: $TEXT_DETECTION_DIR"
echo "Metrics: $METRICS_DIR"
echo "=========================================="
