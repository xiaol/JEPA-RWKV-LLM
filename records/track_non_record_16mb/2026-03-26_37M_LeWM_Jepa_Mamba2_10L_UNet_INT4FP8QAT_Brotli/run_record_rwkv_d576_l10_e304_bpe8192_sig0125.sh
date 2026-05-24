#!/bin/bash
set -euo pipefail

# 4090-compatible BPE8192 RWKV-7 comparison.
#
# The original 8xH100 LeWM BPE run computes SIGReg on 64 sequences per GPU:
#   524,288 tokens / 8 GPUs / 1,024 seq_len = 64 sequences.
# A single 4090 cannot fit the strict equivalent (GRAD_ACCUM_STEPS=8 gives
# 64 sequences per microbatch and OOMs). With GRAD_ACCUM_STEPS=64, SIGReg sees
# 8 sequences per microbatch, so SIGREG_LAMBDA is scaled by 8/64 = 0.125 to
# keep the regularization contribution comparable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID="${RUN_ID:-record_rwkv_d576_l10_e304_bpe8192_sig0125}" \
EMBED_DIM="${EMBED_DIM:-304}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}" \
SIGREG_LAMBDA="${SIGREG_LAMBDA:-0.125}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_record_rwkv_d576_l10_e304_bpe8192_sig0125}" \
bash "$SCRIPT_DIR/run_record_rwkv_d576_l10_bpe8192.sh"
