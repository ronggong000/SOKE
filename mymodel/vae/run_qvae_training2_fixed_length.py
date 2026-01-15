import wandb
import os
from os.path import join as pjoin
import torch
torch.autograd.set_detect_anomaly(True)
# Enable cuDNN autotuner for optimized performance

import random

import sys
import importlib 
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from qvae_option_fixed_length import arg_parse
from qvae_model_rod3_fixed_length import VAE
from qvae_trainer_rod3_fixed_length import VAETrainer


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from evaluator_rod3_fixed_length import MotionEvaluator
#from motion_dataset_rod3_fixed_length import create_data_loaders

#os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True
def print_mem_usage(stage=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{stage}] CUDA Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # allow cuDNN benchmark for speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # 1 load setting
    opt = arg_parse(True)
    fixseed(opt.seed)
    
    # --- W&B 初始化 (在这里添加新代码) ---
    # 初始化 wandb 项目
    wandb.init(
        project="vae-motion-synthesis",  # 替换成你的项目名称
        name=opt.name,                   # 使用你为实验设定的名称作为 wandb 的运行名称
        config=opt                       # 将所有超参数 (opt) 保存到 wandb
    )
    # ------------------------------------
    
    # 2. create VAE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VAE(opt).to(device)
    # Initialize mixed-precision scaler
    scaler = GradScaler()
    print_mem_usage("After VAE model")
    # --- W&B 监控模型 (在这里添加新代码) ---
    # (可选但推荐) 监控模型的梯度和参数
    wandb.watch(net, log="gradients", log_freq=100)
    # ------------------------------------

    num_params = sum(param.numel() for param in net.parameters())
    print(f'Total trainable parameters of all models: {num_params/1_000_000:.2f}M')

    # 3. load dataset
    print("Creating data loaders...")
    try:
        # 3.1. 根据 opt.data_format 构建模块名
        data_format_name = f"{opt.data_format}"
        print(f"Attempting to import data loader from module: {data_format_name}")
        
        # 3.2. 使用 importlib 动态导入模块
        data_format = importlib.import_module(data_format_name)
        
        # 3.3. 从导入的模块中获取 create_data_loaders 函数
        create_data_loaders = data_format.create_data_loaders
        
        # 3.4. 调用函数创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data_dir=opt.train_data_dir,
            val_data_dir=opt.val_data_dir,
            test_data_dir=opt.test_data_dir,
            batch_size=opt.batch_size,
            config=opt
        )
    except ImportError:
        print(f"Error: Could not import the dataset module '{data_format_name}'.")
        print("Please make sure you have a file named '{data_format_name}.py' in your Python path,")
        sys.exit(1) # 导入失败，直接退出程序
    except AttributeError:
        print(f"Error: The module '{data_format_name}' does not have a function named 'create_data_loaders'.")
        sys.exit(1) # 函数不存在，直接退出程序
    # -----------------------------------

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print_mem_usage("After dataset loaded")
    # 4. evaluation setup
    evaluator = MotionEvaluator(opt=opt)

    # 5. train with mixed precision
    trainer = VAETrainer(opt, net, scaler=scaler)
    # Wrap forward/backward calls inside autocast in trainer implementation
    trainer.train(train_loader, val_loader, evaluator)
