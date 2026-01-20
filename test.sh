#!/bin/bash
#SBATCH --job-name=test  # Job的名称
#SBATCH --account=ar85                   # 你的账户
#SBATCH --qos=fitq                       # 你的QoS
#SBATCH --nodes=1                        # 每个任务使用1个节点
#SBATCH --ntasks-per-node=1              # 每个节点运行1个任务
#SBATCH --cpus-per-task=16               # 为每个任务分配16个CPU核心
#SBATCH --partition=gpu                  # 分区
#SBATCH --gres=gpu:L40S:1                # 每个任务需要1个A100 GPU
#SBATCH --mem=128G                       # 每个任务的内存
#SBATCH --time=24:00:00                # Job最长运行时间

# Set the file for output (stdout)
#SBATCH --output=test_output.txt

# Set the file for error log (stderr)
#SBATCH --error=test_error.txt
export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
module unload cuda
module load cuda/12.2.0

# 3. [关键修正] 手动 source conda.sh
# 这行命令让当前脚本拥有使用 conda 的能力
source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion
nvidia-smi
python test_my_vae.py --ckpt_path /home/smuk0019/ar85_scratch2/singyu/SOKE/checkpoints/vae/qvae_b256h1024_L1_fingerdistinct/model/latest.tar --split test
echo "Task $SLURM_ARRAY_TASK_ID finished."
