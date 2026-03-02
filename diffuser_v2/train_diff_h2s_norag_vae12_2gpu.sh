#!/bin/bash
#SBATCH --job-name=diff_h2s_vae12_rag_off_2g
#SBATCH --account=ar85
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=fit
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=diff_h2s_vae12_rag_off_2gpu_output.txt
#SBATCH --error=diff_h2s_vae12_rag_off_2gpu_error.txt

export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
export OMP_NUM_THREADS=8
export PYTORCH_NVML_BASED_CUDA_CHECK=1
module unload cuda
module load cuda/12.2.0

source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion
nvidia-smi

torchrun --standalone --nproc_per_node=2 train_denoiser_v2_how2sign_vae12.py \
    --name h2s_vae12_lenpos_rag_off_cached_bf16_2gpu \
    --vae_path /fs04/scratch2/ar85/singyu/SOKE/checkpoints/HIERARCHICAL/vae12_hier_12d \
    --use_latent_cache \
    --disable_rag \
    --amp_dtype bf16 \
    --batch_size 8 \
    --num_workers 6 \
    --max_epoch 1000 \
    --gloss_layers 1

echo "Task ${SLURM_JOB_ID} finished."
