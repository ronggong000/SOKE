#!/bin/bash
#SBATCH --job-name=diff_asl_full_no_wt  # Job的名称
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
#SBATCH --output=diff_asl_full_no_wt_output.txt

# Set the file for error log (stderr)
#SBATCH --error=diff_asl_full_no_wt_error.txt
export HF_HOME="/fs04/scratch2/ar85/singyu/cache/huggingface"
module unload cuda
module load cuda/12.2.0

# 3. [关键修正] 手动 source conda.sh
# 这行命令让当前脚本拥有使用 conda 的能力
source /home/smuk0019/ar85_scratch2/singyu/miniconda3/etc/profile.d/conda.sh
conda activate sign_motion
nvidia-smi
python train_denoiser_v2.py \
    --name aslc_step1_full_no_wt \
    --dataset_name HIERARCHICAL \
    --vae_path /fs04/scratch2/ar85/singyu/SOKE/checkpoints/vae/qvae_b256h1024_L1_fingerdistinct \
    --train_data_dir /fs04/scratch2/ar85/singyu/aslcitizen/aslcitizen_npz \
    --val_data_dir /fs04/scratch2/ar85/singyu/aslcitizen/aslcitizen_npz \
    --train_csv_path /fs04/scratch2/ar85/singyu/ncslgr/aslcitizen/train.csv \
    --val_csv_path /fs04/scratch2/ar85/singyu/ncslgr/aslcitizen/val.csv \
    --gloss_vocab_size 4000 \
    --batch_size 16 \
    --num_workers 8 \
    --max_epoch 500 \
    --eval_every_e 10 \
    --milestones 100 200 300 \
    --gamma 0.5 \
    --save_latest 10000 \
    --full_train_csv_paths \
      /fs04/scratch2/ar85/singyu/ncslgr/aslcitizen/train.csv \
      /fs04/scratch2/ar85/singyu/ncslgr/aslcitizen/val.csv \
      /fs04/scratch2/ar85/singyu/ncslgr/aslcitizen/test.csv \
echo "Task $SLURM_ARRAY_TASK_ID finished."

