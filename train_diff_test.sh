#!/bin/bash
#SBATCH --job-name=diff_asl_test_no # Job的名称
#SBATCH --account=ar85                   # 你的账户
#SBATCH --qos=fitq                       # 你的QoS
#SBATCH --nodes=1                        # 每个任务使用1个节点
#SBATCH --ntasks-per-node=1              # 每个节点运行1个任务
#SBATCH --cpus-per-task=16               # 为每个任务分配16个CPU核心
#SBATCH --partition=fit                  # 分区
#SBATCH --gres=gpu:A100:1                # 每个任务需要1个A100 GPU
#SBATCH --mem=128G                       # 每个任务的内存
#SBATCH --time=24:00:00                # Job最长运行时间

# Set the file for output (stdout)
#SBATCH --output=diff_asl__test_no_output.txt

# Set the file for error log (stderr)
#SBATCH --error=diff_asl_test_no_error.txt
export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
module unload cuda
module load cuda/12.2.0

# 3. [关键修正] 手动 source conda.sh
# 这行命令让当前脚本拥有使用 conda 的能力
source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion
nvidia-smi
python diffuser_v2/test_diffusion_aslcitizen.py     --checkpoint_dir /home/smuk0019/ar85_scratch2/singyu/SOKE/checkpoints/HIERARCHICAL/aslc_step1_full_no_wt     --report_dir /fs04/scratch2/ar85/singyu/SOKE/diffuser_v2/eval_reports
echo "Task $SLURM_ARRAY_TASK_ID finished."
