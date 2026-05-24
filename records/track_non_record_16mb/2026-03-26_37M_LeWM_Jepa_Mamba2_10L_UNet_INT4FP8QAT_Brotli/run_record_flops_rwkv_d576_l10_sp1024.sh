#!/bin/bash
set -euo pipefail

# Local 4090 proxy for the LeWM/Mamba2 10-minute record compute budget.
#
# Official LeWM/Mamba2 BPE8192 record:
#   6,090 steps * 524,288 tokens/step.
#
# This launcher uses the local SP1024 cache because the published local cache in
# this checkout does not include fineweb10B_sp8192. It matches the train
# parameter-token budget on the best same-size RWKV candidate tested locally:
#   Mamba2 SP1024 train params: 34,500,268
#   RWKV d576 L10 SP1024 train params: 34,005,952
#   target steps = round(6090 * 34,500,268 / 34,005,952) = 6,179

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

cd "$SCRIPT_DIR"

RUN_ID=record_flops_rwkv_d576_l10_sp1024 \
TOKENIZER=bpe \
TOKENIZER_PATH="$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
DATA_PATH="$ROOT_DIR/data/datasets/fineweb10B_sp1024" \
BACKBONE=rwkv7 \
RWKV_HEAD_SIZE=64 \
RWKV_BACKEND=cuda \
RWKV_CHUNK_LEN=16 \
MODEL_DIM=576 \
NUM_LAYERS=10 \
D_STATE=64 \
D_CONV=4 \
EXPAND=1 \
MLP_EVERY=2 \
MLP_MULT=4 \
ACTIVATION=relu2 \
EMBED_DIM=336 \
PREDICTOR_HIDDEN_MULT=4 \
PROJECTOR_TYPE=linear \
TIE_EMBEDDINGS=1 \
LOGIT_SOFTCAP=15 \
SOFTCAP_TYPE=poly \
JEPA_WEIGHT=1.0 \
JEPA_STEPS=3 \
CE_WEIGHT=1.0 \
SIGREG_LAMBDA=1.0 \
SIGREG_SCHEDULE=0 \
DETACH_TARGETS=0 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=64 \
ITERATIONS=6179 \
WARMUP_STEPS=10 \
WARMDOWN_FRACTION=0.15 \
MAX_WALLCLOCK_SECONDS=0 \
MATRIX_LR=0.02 \
SCALAR_LR=0.01 \
EMBED_LR=0.01 \
MUON_BACKEND_STEPS=3 \
MUON_WD=0.04 \
ADAM_WD=0.05 \
GRAD_CLIP_NORM=1.0 \
QUANT_BITS=4 \
FP_STORAGE=FP8 \
QAT_FRACTION=1.0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
VAL_MAX_TOKENS=0 \
TEMP_SCALING=1 \
TEMP_CALIB=1.02 \
SLIDING_EVAL=0 \
CHECKPOINT_EVERY=100 \
CHECKPOINT_DIR=./checkpoints_record_flops_rwkv_d576_l10_sp1024 \
SEED=42 \
COMPILE_MODE=default \
OMP_NUM_THREADS=1 "$PYTHON" "$SCRIPT_DIR/train_jepa_ssm.py"
