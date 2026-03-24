#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# DiffBIR experiment launcher
# Edit only the variables in this section.
# Run with: bash diffbir.sh
# ==========================================

# ---------- Device ----------
GPU=1
DEVICE="cuda"                 # cpu / cuda / mps
PRECISION="fp32"             # fp32 / fp16 / bf16
SEED=123                       # -1 for random seed, or set a specific integer for reproducibility

# ---------- Task / version ----------
TASK="denoise"                  # sr / face / denoise / unaligned_face
VERSION="v2.1"               # v1 / v2 / v2.1 / custom
UPSCALE=1

# ---------- Paths ----------
INPUT_DIR="inputs/demo/mytest"
OUTPUT_DIR="results/v21_spaced_s50_cfg4_cond_notile_fp32_gpu1_bid"

# ---------- Sampling ----------
SAMPLER="spaced"             # spaced / ddim / dpm++_m2 / edm_euler / ...
STEPS=50
CFG_SCALE=4
RESCALE_CFG=false            # true / false

# ---------- Start point ----------
START_POINT_TYPE="cond"      # cond / noise
START_POINT_NOISE_SCALE=""   # e.g. 0.3, 1.0 ; leave empty to disable, 낮을수록 cond많이 유지

# ---------- Batch / outputs ----------
N_SAMPLES=1
BATCH_SIZE=1

# ---------- Prompt / caption ----------
CAPTIONER="none"             # none / llava / ram
POS_PROMPT="sharp, detailed, high quality"
NEG_PROMPT="blurry, low quality, artifacts"

# ---------- Tile toggles ----------
CLEANER_TILED=false #Stage1
CLDM_TILED=false  #controlnet+LDM
VAE_ENCODER_TILED=false #VAE encoder
VAE_DECODER_TILED=false #VAE decoder

# ---------- Tile sizes / strides ----------
CLEANER_TILE_SIZE=512
CLEANER_TILE_STRIDE=256

CLDM_TILE_SIZE=256
CLDM_TILE_STRIDE=128

VAE_ENCODER_TILE_SIZE=1024
VAE_DECODER_TILE_SIZE=256
#Tile size는 모델이 한 번에 처리하는 최대 크기보다 작거나 같아야 함. (예: 512x512 모델이면 tile size <= 512)
#Stride는 tile이 겹치는 정도를 조절. stride < tile_size이면 tile이 겹쳐서 처리됨. 겹치는 부분이 많을수록 경계 아티팩트 감소에 도움될 수 있지만 처리 시간 증가 가능. 실험을 통해 적절한 tile size와 stride 조합을 찾아보는 것을 권장. 예: tile_size=512, stride=256은 50% 겹치는 타일을 생성.

# ---------- Optional noise / sampler extras ----------
NOISE_AUG=0 
#condition latent(c_img) 자체에 추가로 noise를 넣는 옵션.
#stage1 condition을 일부러 흐리게 만들어 robustness를 볼 때 사용.
#보통 0으로 두는 것이 안전.
#값이 커질수록 condition 신뢰도가 떨어져 품질 저하 가능.
ETA=0.0 #DDIM sampler에서 stochasticity를 조절.
ORDER=2 #DPM-Solver 계열 sampler에서 사용하는 적분 차수(order). 높을수록 정확
STRENGTH=1.0  #일부 sampler/설정에서 초기 latent를 얼마나 강하게 유지/변형할지 관련된 계수
S_CHURN=0
S_TMIN=0
S_TMAX=999
S_NOISE=1.0

# ---------- Guidance ----------
GUIDANCE=false               # true / false
G_LOSS="mse"                 # mse / w_mse
G_SCALE=0.0

# ---------- Custom model paths (optional) ----------
TRAIN_CFG=""                 # e.g. configs/train/train.yaml
CKPT=""                      # e.g. weights/custom.ckpt

# ==========================================
# Build command
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
# Run
# ==========================================
echo "=========================================="
echo "Running DiffBIR with the following config:"
echo "GPU=$GPU"
echo "TASK=$TASK"
echo "VERSION=$VERSION"
echo "UPSCALE=$UPSCALE"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SAMPLER=$SAMPLER"
echo "STEPS=$STEPS"
echo "CFG_SCALE=$CFG_SCALE"
echo "START_POINT_TYPE=$START_POINT_TYPE"
echo "PRECISION=$PRECISION"
echo "CLEANER_TILED=$CLEANER_TILED"
echo "CLDM_TILED=$CLDM_TILED"
echo "VAE_ENCODER_TILED=$VAE_ENCODER_TILED"
echo "VAE_DECODER_TILED=$VAE_DECODER_TILED"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"