#!/bin/bash
#SBATCH --job-name=eval_vae12
#SBATCH --account=ar85
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=test_vae12_hier_12d_output.txt
#SBATCH --error=test_vae12_hier_12d_error.txt

export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
module unload cuda
module load cuda/12.2.0

source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion

cd /home/smuk0019/ar85_scratch2/singyu/SOKE || exit 1
export PYTHONPATH=/home/smuk0019/ar85_scratch2/singyu/SOKE

nvidia-smi
python -u test_my_vae.py --experiment vae12_hier_12d --checkpoint_name best --split test --progress_every 1

echo "Task finished."
