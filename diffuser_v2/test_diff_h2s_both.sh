#!/bin/bash
#SBATCH --job-name=eval_h2s_both
#SBATCH --account=ar85
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=eval_h2s_both_output.txt
#SBATCH --error=eval_h2s_both_error.txt

set -euo pipefail

export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
module unload cuda || true
module load cuda/12.2.0

source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion

cd /fs04/scratch2/ar85/singyu/SOKE

python - <<'PY'
import os
import torch
print("[CUDA-CHECK] torch =", torch.__version__)
print("[CUDA-CHECK] built CUDA =", torch.version.cuda)
print("[CUDA-CHECK] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("[CUDA-CHECK] cuda_available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA precheck failed. Abort evaluation instead of silently running on CPU.")
PY

REPORT_DIR="/fs04/scratch2/ar85/singyu/SOKE/diffuser_v2/eval_reports"
mkdir -p "${REPORT_DIR}"

# 可通过环境变量覆盖，默认测 best ckpt
CKPT_NAME="${CKPT_NAME:-net_best_mpjpe.tar}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_INFER_STEPS="${NUM_INFER_STEPS:-50}"

RAG_OFF_CKPT="/fs04/scratch2/ar85/singyu/SOKE/checkpoints/HIERARCHICAL/h2s_v2_lenpos_rag_off_cached_bf16"
RAG_ON_CKPT="/fs04/scratch2/ar85/singyu/SOKE/checkpoints/HIERARCHICAL/h2s_v2_lenpos_rag_on_cached_bf16"

run_eval () {
  local CKPT_DIR="$1"
  local TAG="$2"

  echo "[EVAL] ${TAG} | ckpt=${CKPT_DIR} | file=${CKPT_NAME}"
  python diffuser_v2/test_diffusion_how2sign.py \
    --checkpoint_dir "${CKPT_DIR}" \
    --ckpt_name "${CKPT_NAME}" \
    --split test \
    --device cuda \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_infer_steps "${NUM_INFER_STEPS}" \
    --report_dir "${REPORT_DIR}"
}

nvidia-smi
run_eval "${RAG_OFF_CKPT}" "rag_off"
run_eval "${RAG_ON_CKPT}" "rag_on"

echo "[DONE] reports at ${REPORT_DIR}"
