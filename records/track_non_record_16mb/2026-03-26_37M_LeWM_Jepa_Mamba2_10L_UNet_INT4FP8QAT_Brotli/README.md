# JEPA + RWKV-7 LeWM Record

This repo is a contest record for the OpenAI `parameter-golf` benchmark. For the upstream challenge context and the baseline reference, see the original repo: https://github.com/openai/parameter-golf

## Result

| Config | Sliding BPB | Standard BPB | Artifact | Notes |
|---|---:|---:|---:|---|
| RWKV-7 BPE8192 local run | 1.2036 | 1.2651 | 14.95 MB | 4090-comparable proxy |
| Mamba2 BPE8192 10min record | 1.2566 | 1.2721 | 15.50 MB | 8xH100 SXM reference |

The local RWKV-7 run is slightly better on both BPB metrics and fits under the 16 MB limit.

## How To Use

Use the launcher scripts in this folder:

```bash
bash run_record_rwkv_d576_l10_bpe8192.sh
bash run_record_rwkv_d576_l10_e304_bpe8192_sig0125.sh
```

The second script is the comparable local 4090 setup. It uses the BPE8192 data/tokenizer, RWKV-7 backbone, and adjusted SIGReg scaling for the single-GPU batch size.

## Reference

- Upstream contest repo: https://github.com/openai/parameter-golf
- Simple baseline record: https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
- This record's training code: `train_jepa_ssm.py`
