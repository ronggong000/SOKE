#!/bin/bash
#SBATCH --job-name=vae12_hier
#SBATCH --account=ar85
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=fit
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=vae12_train_output.txt
#SBATCH --error=vae12_train_error.txt

export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
module unload cuda
module load cuda/12.2.0

source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion

nvidia-smi
python mymodel/vae_2/run_vae12_training2_fixed_length.py \
  --amp_dtype bf16 \
  --is_continue

echo "Task finished."
