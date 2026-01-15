import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from os.path import join as pjoin

import os
import time
import numpy as np
from collections import OrderedDict, defaultdict
from datetime import datetime

#from utils.eval_t2m import evaluation_vae, test_vae
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import smplx
from torch.cuda.amp import autocast, GradScaler

def save_eval_summary(metrics_dict, save_dir="results", joint_names=None, epoch=None, prefix=""):
    """
    保存评估结果的summary和热力图
    
    Args:
        metrics_dict: 评估指标字典
        save_dir: 保存目录
        joint_names: 关节名称列表
        epoch: 当前epoch，如果为None则使用时间戳
        prefix: 文件前缀，用于区分不同阶段的评估（如"pretrain", "epoch"等）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名后缀
    if epoch is not None:
        if prefix:
            suffix = f"{prefix}_epoch_{epoch:03d}"
        else:
            suffix = f"epoch_{epoch:03d}"
    else:
        # 使用时间戳，适用于训练前评估
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            suffix = f"{prefix}_{timestamp}"
        else:
            suffix = f"pretrain_{timestamp}"

    # ===== 1. 热力图（MPJPE per joint） =====
    #ric_joint = metrics_dict["ric_mpjpe"]["mpjpe_per_joint"].cpu().numpy()
    rot_joint = metrics_dict["rot_mpjpe"]["mpjpe_per_joint"].cpu().numpy()
    #assert len(ric_joint) == len(joint_names), f"ric: {len(ric_joint)}, namelist: {len(joint_names)}"
    heat_data = [rot_joint]
    heat_labels = ["rot"]

    plt.figure(figsize=(max(10, len(rot_joint) * 0.3), 2.5))
    ax = sns.heatmap(
        heat_data,
        cmap="YlOrRd",
        cbar=True,
        annot=False,
        xticklabels=joint_names if joint_names is not None else [str(i) for i in range(len(rot_joint))],
        yticklabels=heat_labels
    )
    ax.set_title(f"MPJPE per joint (mm) - {suffix}")
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, f"mpjpe_per_joint_heatmap_{suffix}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"✅ Saved heatmap to {heatmap_path}")

    # ===== 2. 打印和保存其他指标 =====
    summary_rows = []

    for key, value_dict in metrics_dict.items():
        for subkey, val in value_dict.items():
            if isinstance(val, torch.Tensor):
                val = val.item() if val.dim() == 0 else val.cpu().numpy()
            summary_rows.append({
                "Category": key,
                "Metric": subkey,
                "Value": val
            })

    df_all = pd.DataFrame(summary_rows)

    # 🔹 只用于终端打印：跳过过长的行
    df_print = df_all[~df_all["Metric"].isin(["pa_mpjpe_per_joint", "mpjpe_per_joint"])]

    print(f"\n📊 Evaluation Summary ({suffix}):\n")
    print(df_print.to_string(index=False, float_format=lambda x: "%.3f" % x if isinstance(x, float) else str(x)))

    # 🔹 保存全部字段（包括大数组）
    table_path = os.path.join(save_dir, f"evaluation_summary_{suffix}.csv")
    df_all.to_csv(table_path, index=False)
    print(f"✅ Saved evaluation summary table to {table_path}")

@torch.no_grad()
def save_eval_visualization_sample(opt, model, val_loader, save_path):
    model.eval()
    for batch_data in val_loader:
        # unpack batch
        motion, rod3_data, mask, lengths = batch_data

        motion = motion.to(opt.device)
        mask = mask.to(opt.device)

        # forward pass
        pred_motion, loss_dict = model.forward(motion, mask)

        # 保存第一个样本（index=0）的结果
        to_save = {
            "input": motion[0].cpu().numpy(),        # [T, D]
            "output": pred_motion[0].cpu().numpy(),  # [T, D]
            "length": lengths[0].item()              # 原始长度（用于变长序列可视化时裁剪）
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **to_save)

        print(f"✅ Saved inference output for visualization to {save_path}")
        break  # 只处理第一个 batch

def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


    # Header: epoch/iter info
    if epoch is not None and inner_iter is not None:
        print(f"[Epoch {epoch:02d} | Iter {inner_iter:04d}] ", end='')

    # Progress
    percent = niter_state / total_niters if total_niters > 0 else 0
    progress_info = time_since(start_time, percent)
    print(f"[{niter_state}/{total_niters} ({percent * 100:.1f}%) | Elapsed {progress_info}]", end=' ')

    # Optional: teacher forcing ratio
    if tf_ratio is not None:
        print(f"TF-ratio: {tf_ratio:.3f}", end=' ')
    if sl_steps is not None:
        print(f"SL-steps: {sl_steps}", end=' ')

    # Losses
    if isinstance(losses, dict):
        loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in losses.items()])
        print(f"| Losses: {loss_str}")
    else:
        print()  # 损失为空时换行

def def_value():
    return 0.0

class VAETrainer:
    def __init__(self, opt, vae, scaler=None):
        self.opt = opt
        self.vae = vae
        self.scaler = scaler
        self.device = opt.device
        self.smplx_model = None
        
        # --- 1. 初始化损失函数 ---
        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            
            # 【修改点 1】添加 reduction='none'，这样我们才能在后面给手指加权
            if opt.recon_loss == "mse":
                self.recon_criterion = nn.MSELoss(reduction='none') 
            else:
                self.recon_criterion = nn.L1Loss(reduction='none')

            if opt.mesh_loss == "l1_smooth":
                self.mesh_criterion = nn.SmoothL1Loss(reduction='none')
            elif opt.mesh_loss == "mse": # 兼容 mse
                self.mesh_criterion = nn.MSELoss(reduction='none')
            else: # 默认 l1
                self.mesh_criterion = nn.L1Loss(reduction='none')
        # --- 2. 加载 SMPL-X 模型 ---
        try:
            self.smplx_model = smplx.create(
                model_path=opt.smplx_model_path,
                model_type='smplx',
                gender='neutral',
                use_pca=False,
                flat_hand_mean=True,
                batch_size=opt.batch_size * opt.max_length,
            ).to(self.device).eval()
        except Exception as e:
            print(f"SMPLX模型加载失败: {e}")

        # --- 3. 【核心修正】根据模式准备基础变量 ---
        # 检查是否处于手指降维模式
        is_reduce_dim_mode = hasattr(opt, 'reduce_dim_finger') and opt.reduce_dim_finger
        
        if is_reduce_dim_mode:
            print("VAETrainer: Initializing in HETEROGENEOUS (reduce_dim_finger) mode.")
            # 在此模式下，opt.joint_feature_dims 必须存在
            if not hasattr(opt, 'joint_feature_dims'):
                raise AttributeError("`reduce_dim_finger` is True, but `opt.joint_feature_dims` is not defined in options.")
            joint_dims_for_calc = opt.joint_feature_dims
        else:
            print("VAETrainer: Initializing in UNIFORM (3D axis-angle) mode.")
            # 在统一模式下，我们手动创建 joint_dims 列表
            joint_dims_for_calc = [3] * opt.joints_num

        # --- 4. 准备用于 Mesh Loss 的旋转数据索引 ---
        # 这个逻辑现在对两种模式都通用
        self.rot_indices = []
        current_idx = 0
        for dim in joint_dims_for_calc:
            # 旋转总是每个关节特征的前3维
            self.rot_indices.extend([current_idx, current_idx + 1, current_idx + 2])
            current_idx += dim

        # --- 5. 根据 finger_loss_weight 设置加权损失 ---
        # 权重 > 1.0 时才启用加权
        if hasattr(opt, 'finger_loss_weight') and opt.finger_loss_weight != 1.0:
            self.finger_loss_weight = opt.finger_loss_weight
            print(f"VAETrainer: Weighted loss ENABLED with finger_loss_weight = {self.finger_loss_weight}")

            # a. 准备 Reconstruction Loss 的手指特征索引
            self.rec_finger_indices = []
            current_idx = 0
            for i in range(self.opt.joints_num):
                dim = joint_dims_for_calc[i]
                if i in self.opt.hand_joint_indices:
                    self.rec_finger_indices.extend(range(current_idx, current_idx + dim))
                current_idx += dim
            
            # b. 准备 Mesh Loss 的手部顶点索引
            all_verts_list = self.opt.UPPER_BODY_VERTEX + self.opt.LEFT_HAND_VERTEX + self.opt.RIGHT_HAND_VERTEX
            hand_verts_set = set(self.opt.LEFT_HAND_VERTEX + self.opt.RIGHT_HAND_VERTEX)
            hand_vtx_indices_list = [i for i, v_id in enumerate(all_verts_list) if v_id in hand_verts_set]
            self.hand_vertex_indices = torch.tensor(hand_vtx_indices_list, device=self.device, dtype=torch.long)
        else:
            print("VAETrainer: Weighted loss DISABLED.")
            self.finger_loss_weight = 1.0
            self.rec_finger_indices = None
            self.hand_vertex_indices = None
        self.ALL_SELECTED_VERTICES = self.opt.UPPER_BODY_VERTEX + self.opt.LEFT_HAND_VERTEX + self.opt.RIGHT_HAND_VERTEX
        self.body_indices = torch.tensor(
            self.opt.SELECTED_JOINT_INDICES_BODY_ONLY, 
            device=self.device, 
            dtype=torch.long
        )
    # --- 新增方法: 监控潜空间分布 ---
    def monitor_latent_stats(self, val_loader):
        self.vae.eval()
        all_valid_z = []
        
        print("📊 Monitoring Latent Space Statistics...")
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                # 1. 解包数据
                motion, lengths = batch_data
                motion = motion.to(self.device)
                lengths = lengths.to(self.device)
                
                # 2. 获取连续潜变量 Z (Continuous Latent)
                # 注意：根据你的 VAE 代码，encode 返回的是 (z, loss_dict)
                z, _ = self.vae.encode(motion) # z shape: [B, T, J, D]
                
                # 3. 处理 Padding (关键步骤！)
                # z 的时间维度 T 包含了 padding，必须根据 lengths 掩码掉
                B, T, J, D = z.shape
                
                # 创建 mask: [B, T] -> [B, T, 1, 1]
                mask = (torch.arange(T, device=self.device)[None, :] < lengths[:, None])
                mask = mask.unsqueeze(-1).unsqueeze(-1) # 广播到 J 和 D
                
                # 4. 提取有效数据
                # masked_select 会把数据展平为 [N_total_valid_elements]
                valid_z_batch = torch.masked_select(z, mask)
                
                # 为了节省显存，转到 CPU 并存入列表
                all_valid_z.append(valid_z_batch.cpu())
                
                # 为了速度，只统计前 20 个 Batch 就足够代表分布了
                if i > 20: 
                    break
        
        # 5. 拼接所有有效数据
        if len(all_valid_z) > 0:
            full_z = torch.cat(all_valid_z)
            
            # 6. 计算统计量
            z_mean = full_z.mean().item()
            z_std = full_z.std().item()
            z_max = full_z.max().item()
            z_min = full_z.min().item()
            
            print(f"   -> Latent Mean: {z_mean:.4f} (Ideal: ~0.0)")
            print(f"   -> Latent Std : {z_std:.4f}  (Ideal: ~1.0)")
            
            return {
                "latent_stats/mean": z_mean,
                "latent_stats/std": z_std,
                "latent_stats/max": z_max,
                "latent_stats/min": z_min
            }
        else:
            return {}
    # --- 在 VAETrainer 类内部 ---
    def eval_process(self, evaluator, val_loader, selected_names, epoch, it): # 增加 it 
        print("starting eval")
        # 1. 原有的评估逻辑 (Reconstruction Metrics)
        evaluation_results = evaluator.calculate_metrics(self.vae, val_loader, self.smplx_model)
        # --- W&B 记录详细评估指标 (在这里添加新代码) ---
        eval_log_dict = {}
        # 展平嵌套的字典以方便记录
        for key, value_dict in evaluation_results.items():
            for subkey, val in value_dict.items():
                # 我们只记录标量值，忽略 per_joint 的数组
                if "per_joint" not in subkey:
                    metric_name = f"eval/{key}_{subkey}" # e.g., "eval/rot_mpjpe_mpjpe_body"
                    eval_log_dict[metric_name] = val.item() if isinstance(val, torch.Tensor) else val
        # ==================== 【新增代码开始】 ====================
        # 2. 潜空间分布检查
        latent_stats = self.monitor_latent_stats(val_loader)
        
        # 将 latent 统计数据合并到 log 字典中
        eval_log_dict.update(latent_stats)
        
        # 简单的健康检查报警 (在终端打印警告)
        if epoch is not None: # 只在正式训练 eval 时检查
            std = latent_stats.get("latent_stats/std", 1.0)
            if std > 1.5 or std < 0.5:
                print(f"⚠️ WARNING: Latent STD is abnormal ({std:.4f})! Diffusion model training might fail.")
                print(f"   Suggestion: Adjust KL weight (lambda_kl) or check input normalization.")
        # ==================== 【新增代码结束】 ====================
        
        wandb.log(eval_log_dict, step=it)

        # 注意：这里的 save_eval_summary 会生成一个热力图
        # 我们可以顺便把这个图也上传到 wandb
        save_dir = self.opt.save_root
        if epoch is None:
            prefix = "pretrain"
        else:
            prefix = "eval"

        save_eval_summary(
            metrics_dict=evaluation_results,
            save_dir=save_dir,
            joint_names=selected_names,
            epoch=epoch,
            prefix=prefix
        )

        # 从 save_eval_summary 获取 heatmap 路径并上传
        # (这部分逻辑需要与 save_eval_summary 内部的文件名生成逻辑保持一致)
        if epoch is None:
             # 对于 pretrain，save_eval_summary 使用时间戳，我们无法直接预测文件名
             # 最简单的办法是假设只有一个 pretrain heatmap，或者修改 save_eval_summary 返回路径
             # 为简单起见，这里我们暂时只上传 epoch > 0 时的 heatmap
             pass
        else:
            suffix = f"{prefix}_epoch_{epoch:03d}"
            heatmap_path = os.path.join(save_dir, f"mpjpe_per_joint_heatmap_{suffix}.png")
            if os.path.exists(heatmap_path):
                wandb.log({"eval/MPJPE_Heatmap": wandb.Image(heatmap_path)}, step=it)

    def train_forward(self, batch_data, epoch):
        # 1. 解包数据并移动到设备
        motion, lengths = batch_data  # motion:[B, T, D_flat], lengths:[B]
        motion = motion.to(self.opt.device)
        lengths = lengths.to(self.opt.device)
        B, T, D_flat = motion.shape 
        
        # 2. VAE 前向传播 (Dual Path)
        # 返回: 连续输出, 量化输出, 连续Z, 量化Z, 损失字典
        out_cont, out_quant, z_cont, z_quant, loss_dict = self.vae(motion)

        # ==================== 【完整版: Codebook Reset 机制】 ====================
        # 策略：利用当前 Batch 丰富的 z_cont 特征来激活死码
        # 触发频率：约每 50 个 Batch 触发一次 (概率 0.02)
        if self.vae.training and torch.rand(1).item() < 0.02:
            with torch.no_grad():
                # 直接把完整的 z_cont 传进去，Model 自己知道怎么切
                n_total, stats = self.vae.reset_all_codebooks(z_cont)
                
                if n_total > 0:
                    # Optional: 打印重置详情
                    msg = ", ".join([f"{k}:{v}" for k,v in stats.items()])
                    print(f"[Reset] {msg}")
                    pass
        # =======================================================================

        # 3. 创建 Mask
        mask = torch.arange(T, device=self.opt.device)[None, :] < lengths[:, None]
        N_frames = B * T
        
        # 4. 准备 Ground Truth 的 SMPL-X 数据 (完整补全)
        # ---------------------------------------------------------
        if self.opt.data_format == 'motion_dataset_rod3_fixed_length_dk':
            gt_rot = motion[:, :, self.rot_indices]
            all_gt = gt_rot.reshape(N_frames, self.opt.joints_num, 3).contiguous()
        else:
            all_gt = motion.reshape(N_frames, self.opt.joints_num, 3).contiguous()

        # 定义切分函数 (复用)
        def split_smplx_local(x):
            # x shape: [N_frames, J, 3]
            body = x[:, :13]   # 0-12
            lhand = x[:, 13:28] # 13-27
            rhand = x[:, 28:43] # 28-42
            
            # 还原到 SMPL-X 完整关节 (22个身体 + 手)
            restored = torch.zeros(x.shape[0], 22, 3, device=self.device, dtype=x.dtype)
            restored[:, self.body_indices] = body 
            # 返回: body(排除root), lhand, rhand
            return restored[:, 1:], lhand, rhand

        # 计算 GT Vertices
        gt_body, gt_lh, gt_rh = split_smplx_local(all_gt)
        with torch.no_grad():
            out_gt = self.smplx_model(body_pose=gt_body, left_hand_pose=gt_lh, right_hand_pose=gt_rh)
        
        # 提取 GT 顶点并 Reshape [B, T, V, 3]
        verts_gt_full = out_gt.vertices[:, self.ALL_SELECTED_VERTICES, :].reshape(B, T, -1, 3)
        valid_verts_gt = verts_gt_full[mask] # [N_valid, V, 3]
        # ---------------------------------------------------------

        # 5. 定义计算 Mesh Loss 的闭包 (用于 Cont 和 Quant 两路)
        def compute_weighted_mesh_loss(pred_motion):
            # a. Reshape
            if self.opt.data_format == 'motion_dataset_rod3_fixed_length_dk':
                pred_rot = pred_motion[:, :, self.rot_indices]
                all_pred = pred_rot.reshape(N_frames, self.opt.joints_num, 3).contiguous()
            else:
                all_pred = pred_motion.reshape(N_frames, self.opt.joints_num, 3).contiguous()
            
            # b. Split & Forward
            pd_body, pd_lh, pd_rh = split_smplx_local(all_pred)
            out_pd = self.smplx_model(body_pose=pd_body, left_hand_pose=pd_lh, right_hand_pose=pd_rh)
            verts_pd = out_pd.vertices[:, self.ALL_SELECTED_VERTICES, :].reshape(B, T, -1, 3)
            
            # c. Masking
            valid_verts_pd = verts_pd[mask] # [N_valid, V, 3]
            
            # ==================== 【修改点 2 START】 ====================
            # d. Error Calculation (直接使用 init 里定义的 loss 函数)
            # 因为我们设置了 reduction='none'，这里返回的 error 形状和输入一样，是 [N_valid, V, 3]
            error = self.mesh_criterion(valid_verts_pd, valid_verts_gt)
            # ==================== 【修改点 2 END】 ====================
            
            # e. Weighting (手指加权)
            if self.opt.finger_loss_weight != 1.0:
                # 构造权重: [1, N_verts, 1]
                weights = torch.ones(valid_verts_pd.shape[1], device=self.device)
                weights[self.hand_vertex_indices] = self.finger_loss_weight
                weights = weights.view(1, -1, 1)
                return (error * weights).mean()
            else:
                return error.mean()

        # 6. 计算双路 Reconstruction Loss
        loss_mesh_cont = compute_weighted_mesh_loss(out_cont)
        
        # 关键：量化路也要算 Mesh Loss，这样 Decoder 才会去适应 z_q
        loss_mesh_quant = compute_weighted_mesh_loss(out_quant)

        # 7. 计算一致性损失 (Consistency Loss)
        # -----------------------------------------------------------
        # A. Latent Consistency (Commitment): 
        #    拉近 z_cont 和 z_quant。
        #    detach() 是标准操作：只拉动 Encoder (z_cont) 去靠近 Codebook (z_quant)，
        #    而不希望把 Codebook 拉乱 (Codebook 更新由 Quantizer 内部 loss 负责)。
        loss_latent_consist = torch.mean((z_cont - z_quant.detach())**2)
        
        # B. Output Consistency (Self-Distillation):
        #    让 Quantized Output 去模仿 Continuous Output。
        #    这比单纯模仿 GT 更容易，因为 Continuous Output 包含了模型自身的偏置，
        #    这能让量化路更快收敛。
        #loss_output_consist = torch.mean((out_quant - out_cont.detach())**2)
        # -----------------------------------------------------------

        # 8. 汇总 Loss
        loss_kl = loss_dict["loss_kl"]
        loss_quant = loss_dict.get("loss_quant", 0.0) # 包含 embedding loss
        
        # 权重配置 (建议放入 opt)
        w_q_recon = getattr(self.opt, 'lambda_q_recon', 1.0)
        w_consist = getattr(self.opt, 'lambda_consistency', 0.5)
        w_quant_loss = getattr(self.opt, 'lambda_quant', 1.0)
        
        total_loss = loss_mesh_cont + \
                     (w_q_recon * loss_mesh_quant) + \
                     (w_consist * loss_latent_consist) + \
                     (w_quant_loss * loss_quant) + \
                     (self.opt.lambda_kl * loss_kl)

        # 记录详细 Loss 供 W&B 监控
        loss_dict["loss_mesh_cont"] = loss_mesh_cont
        loss_dict["loss_mesh_quant"] = loss_mesh_quant
        loss_dict["loss_consist"]   = loss_latent_consist
        loss_dict["loss_total"]     = total_loss
        
        return total_loss, loss_dict

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr


    def save(self, file_name, epoch, total_iter):
        state = {
            "vae": self.vae.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, evaluator):
        self.vae.to(self.opt.device)

        # optimizer
        self.optim = torch.optim.AdamW(self.vae.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        
        # eval before train - 使用pretrain前缀和时间戳
        selected_names = [self.opt.SMPLX_JOINT_LANDMARK_NAMES[i] for i in self.opt.SELECTED_JOINT_LANDMARK_INDICES]
        self.vae.eval()
        # 【最佳实践】在这里包裹 no_grad，从根源上阻止梯度计算
        with torch.no_grad():
            self.eval_process(evaluator, val_loader, selected_names, None, it)
        
        # 训练前评估结束后，别忘了将模型切换回训练模式
        self.vae.train() 
        # training loop
        while epoch < self.opt.max_epoch:
            self.vae.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # forward
                self.optim.zero_grad()
                with autocast():
                    loss, loss_dict = self.train_forward(batch_data, epoch)


                # --- 【核心修改】梯度裁剪逻辑 ---
                # 1. 照常计算缩放后的梯度
                self.scaler.scale(loss).backward()

                # 2. 在裁剪前，必须先 unscale 梯度
                self.scaler.unscale_(self.optim)

                # 3. 对 unscale 后的梯度进行裁剪，1.0 是一个常用的最大范数阈值
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)

                # 4. 优化器执行一步
                self.scaler.step(self.optim)

                # 5. 更新 scaler 的缩放因子
                self.scaler.update()
                
                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                # --- Codebook Reset Strategy ---
                # 每 500 个 step 检查一次
                if it % 500 == 0:
                    # 收集当前 batch 的 z (为了从中采样)
                    # 我们需要再次 encode 一下或者缓存之前的 z
                    # 为了简单，我们从当前 batch 重新 encode
                    with torch.no_grad():
                        # 1. 重新 Encode 当前 batch 以获取最新特征
                        z_curr, _ = self.vae.encode(batch_data[0].to(self.device))
                        
                        # 2. 调用模型内部封装好的 Reset 方法
                        # 它会自动处理所有分组（无论是 Default, Arm Mirror 还是 Finger Distinct）
                        n_total, stats = self.vae.reset_all_codebooks(z_curr)
                        
                        # 3. 打印日志
                        if n_total > 0:
                            msg = ", ".join([f"{k}:{v}" for k,v in stats.items()])
                            print(f"[Iter {it}] Codebook Reset: {msg}")
                for tag, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        logs[tag] += value.item()
                    else:
                        logs[tag] += value

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                    # --- W&B 记录训练损失 (在这里添加新代码) ---
                    # 构造一个带 'train/' 前缀的字典并记录
                    train_log_dict = {"train/" + k: v for k, v in mean_loss.items()}
                    train_log_dict['lr'] = self.optim.param_groups[0]['lr'] # 额外记录学习率
                    wandb.log(train_log_dict, step=it)
                    # ------------------------------------

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            print('Validation time:')
            self.vae.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_dict = self.train_forward(batch_data,epoch)

                    # ==================== 【BUG 修复区 START (通用版)】 ====================
                    val_log["loss"] += loss.item()
                    for tag, value in loss_dict.items():
                        # --- 核心修复 2.0: 动态识别非标量 ---
                        
                        # 1. 如果 key 名字里明确写了是 indices，直接跳过
                        if "indices" in tag:
                            continue
                        
                        # 2. 双重保险：检查 Tensor 的维度
                        # 如果是张量且包含多于 1 个元素，绝对不能 .item()
                        if isinstance(value, torch.Tensor):
                            if value.numel() > 1:
                                continue
                            val_log[tag] += value.item()
                        else:
                            val_log[tag] += value
                    # ==================== 【BUG 修复区 END】 ====================

            
            # --- W&B 记录验证损失 (在这里修改) ---
            # 构造一个带 'val/' 前缀的字典并记录
            val_log_dict = {}
            msg = "Validation loss: "
            for tag, value in val_log.items():
                # --- 修复: 除以 len(val_loader) 来获得平均值 ---
                avg_val = value / len(val_loader)
                self.logger.add_scalar('Val/%s'%tag, avg_val, epoch)
                msg += "%s: %.8f, " % (tag, avg_val)
                val_log_dict["val/" + tag] = avg_val # 填充字典
            print(msg)
            wandb.log(val_log_dict, step=it) # 在迭代步上记录验证结果
            # mean_loss = OrderedDict() # 这一行似乎是多余的，注释掉
            # -----------------------------------
            
            # evaluation - 使用epoch信息
            if epoch % self.opt.eval_every_e == 0:
                self.vae.eval()
                with torch.no_grad():
                    self.eval_process(evaluator, val_loader, selected_names, None, it)
                self.vae.train()
