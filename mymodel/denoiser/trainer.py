from typing import List, Union
import wandb
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import sys
import time
import numpy as np
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_denoiser, test_denoiser
from utils.utils import print_current_loss, attn2img
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from physical_evaluator import SignPhysicalEvaluator
smplx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../..",'smplx')
sys.path.append(smplx_path)
import smplx
from torch.amp import autocast # 仅保留引用防止报错，实际不使用

def def_value():
    return 0.0

def lengths_to_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    max_frames = max_len if max_len is not None else torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


class DenoiserTrainer:
    def __init__(self, opt, denoiser, vae, scheduler):
        self.opt = opt
        self.denoiser = denoiser.to(opt.device)
        self.vae = vae.to(opt.device)
        self.noise_scheduler = scheduler
        
        # 1. 挂载物理评估器
        self.physical_evaluator = SignPhysicalEvaluator(opt, opt.device)
        
        # 2. 初始化 SMPL-X (仅在旋转模式下需要，但保留逻辑以防万一)
        self.smplx_model = None
        if not getattr(opt, 'xyz', False):
            try:
                max_smplx_batch = opt.batch_size * opt.max_motion_length 
                self.smplx_model = smplx.create(
                    model_path=smplx_path, 
                    model_type='smplx', 
                    gender='neutral', 
                    use_pca=False,
                    flat_hand_mean=True, 
                    batch_size=max_smplx_batch
                ).to(opt.device).eval()
                print(f"✅ SMPL-X Model initialized with static capacity: {max_smplx_batch}")
            except Exception as e:
                print(f"❌ SMPL-X Load Error: {e}")
            
        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()
            elif opt.recon_loss == "l2":
                self.recon_criterion = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"Reconstruction loss {opt.recon_loss} not implemented")
            
        if opt.is_train:
            log_path = pjoin(opt.model_dir, "train_log.txt")
            self.log_file = open(log_path, "a", encoding="utf-8")
            self.log_to_file(f"=== Training session started at {time.ctime()} ===")
        
        # 物理损失权重预计算
        self.all_verts_indices = opt.UPPER_BODY_VERTEX + opt.LEFT_HAND_VERTEX + opt.RIGHT_HAND_VERTEX
        hand_verts_set = set(opt.LEFT_HAND_VERTEX + opt.RIGHT_HAND_VERTEX)
        
        hand_vtx_rel_indices = [i for i, v_id in enumerate(self.all_verts_indices) if v_id in hand_verts_set]
        self.hand_vertex_indices = torch.tensor(hand_vtx_rel_indices, device=opt.device, dtype=torch.long)

        self.vertex_weights = torch.ones(len(self.all_verts_indices), device=opt.device)
        self.vertex_weights[self.hand_vertex_indices] = opt.finger_loss_weight
        self.vertex_weights = self.vertex_weights.view(1, -1, 1) 

        self.body_indices = torch.tensor(opt.SELECTED_JOINT_INDICES_BODY_ONLY, device=opt.device)
        # 【修复】彻底移除 scaler 初始化，防止误用
    def vae_encode_raw(self, x_raw: torch.Tensor):
        """
        x_raw: [B, T, D_flat] raw motion (axis-angle or xyz)
        返回：z, kl_dict
        兼容：
        - 新版 VAE：有 mean/std，需要归一化
        - 旧版 VAE：没有 mean/std，直接 encode
        """
        # 1) 处理 mean/std（若不存在则跳过）
        if hasattr(self.vae, "mean") and hasattr(self.vae, "std"):
            mean = self.vae.mean
            std = self.vae.std
            x_in = (x_raw - mean) / (std + 1e-8)
        else:
            x_in = x_raw

        # 2) encode（兼容返回结构）
        out = self.vae.encode(x_in)

        # encode 可能返回 (z, dict) 或 dict 或 只返回 z
        if isinstance(out, tuple) and len(out) == 2:
            z, info = out
        else:
            z, info = out, {}
        if not hasattr(self, "_printed_latent_info"):
            print("latent dtype/shape:", z.dtype, z.shape)
            self._printed_latent_info = True
        if info is None:
            info = {}
        return z, info
        

    def vae_decode_to_raw(self, z: torch.Tensor):
        """
        z: latent
        返回：x_raw [B, T, D_flat]（物理量级）
        兼容：
        - 新版 VAE：decode 输出 norm，需要 denorm
        - 旧版 VAE：decode 直接输出 raw
        """
        out = self.vae.decode(z)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if hasattr(self.vae, "mean") and hasattr(self.vae, "std"):
            mean = self.vae.mean
            std = self.vae.std
            x_raw = out * std + mean
        else:
            x_raw = out
        return x_raw

    def log_to_file(self, message):
        print(message)
        if hasattr(self, 'log_file'):
            self.log_file.write(message + "\n")
            self.log_file.flush()
    def train_forward(self, batch_data, epoch):
        """
        batch_data: (text, motion, masks, m_lens, names)
        text:  List[str] 或 (text_emb[B,L,D], text_mask[B,L])
        motion: [B, T_pad, D_flat]
        masks:  [B, T_pad] (1=valid, 0=pad)
        m_lens: [B]
        """
        text, motion, masks, m_lens, names = batch_data
        device = self.opt.device

        # ===== cond drop（兼容 str 或 (emb,mask)）=====
        p_drop = float(getattr(self.opt, "cond_drop_prob", 0.0))
        if p_drop > 0:
            if isinstance(text, tuple) and len(text) == 2 and torch.is_tensor(text[0]):
                text_emb, text_mask = text
                if np.random.rand(1) < p_drop:
                    text_mask = torch.zeros_like(text_mask, dtype=torch.bool)
                text = (text_emb, text_mask)
            else:
                text = ["" if np.random.rand(1) < p_drop else t for t in text]

        motion = motion.to(device, dtype=torch.float32)
        masks = masks.to(device)
        m_lens = m_lens.to(device, dtype=torch.long)

        B, T_pad, D_flat = motion.shape

        # ===== 0) Trim：只去掉 batch padding 尾巴（不裁内容）=====
        T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
        T_valid = max(1, min(T_valid, T_pad))
        if T_valid < T_pad:
            motion = motion[:, :T_valid]
            masks = masks[:, :T_valid]

        # ===== 1) Encode =====
        with torch.no_grad():
            latent, _ = self.vae_encode_raw(motion)  # [B, Tz, ...]
            if latent.dim() == 3:
                Bb, Tz, JD = latent.shape
                if JD % 3 == 0:
                    latent = latent.view(Bb, Tz, JD // 3, 3)

            # 动态 ratio（更稳）
            Tm = motion.shape[1]
            Tz = latent.shape[1]
            downsample_ratio = max(1, Tm // Tz)
            curr_m_lens = torch.clamp(m_lens // downsample_ratio, min=0, max=Tz)
            len_mask = lengths_to_mask(curr_m_lens).to(device)  # [B, Tz]

        # ===== 2) Diffusion =====
        timesteps = torch.randint(0, self.opt.num_train_timesteps, (B,), device=device).long()

        noise = torch.randn_like(latent)
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        pred, attn_list = self.denoiser.forward(noisy_latent, timesteps, text, len_mask=len_mask)
        pred = pred * len_mask[..., None, None].float()

        # ===== 3) Base loss：prediction_type 三选一 =====
        loss_dict = {}
        loss = torch.tensor(0.0, device=device)

        pred_w = float(getattr(self.opt, "pred_loss_weight", 1.0))

        if self.opt.prediction_type == "sample":
            base = self.recon_criterion(pred, latent)
            loss = loss + pred_w * base
            loss_dict["loss_sample"] = base
            x0_hat = pred

        elif self.opt.prediction_type == "epsilon":
            base = self.recon_criterion(pred, noise)
            loss = loss + pred_w * base
            loss_dict["loss_eps"] = base

            a = self.noise_scheduler.alphas_cumprod[timesteps].to(device)  # [B]
            sa = torch.sqrt(a).view(B, 1, 1, 1)
            som = torch.sqrt(1.0 - a).view(B, 1, 1, 1)
            x0_hat = (noisy_latent - som * pred) / (sa + 1e-8)

        elif self.opt.prediction_type == "v_prediction":
            vel = self.noise_scheduler.get_velocity(latent, noise, timesteps)
            base = self.recon_criterion(pred, vel)
            loss = loss + pred_w * base
            loss_dict["loss_vel"] = base

            a = self.noise_scheduler.alphas_cumprod[timesteps].to(device)  # [B]
            sa = torch.sqrt(a).view(B, 1, 1, 1)
            som = torch.sqrt(1.0 - a).view(B, 1, 1, 1)
            x0_hat = sa * noisy_latent - som * pred

        else:
            raise ValueError(f"prediction_type must be one of ['sample','epsilon','v_prediction'], got {self.opt.prediction_type}")

        # ===== 4) Dist loss（不管 prediction_type 是啥，都能加）=====
        dist_w = float(getattr(self.opt, "dist_loss_weight", 0.0))
        finger_w = float(getattr(self.opt, "finger_loss_weight", 10.0))
        mesh_threshold = int(getattr(self.opt, "mesh_threshold", 200))
        warmup_epochs = int(getattr(self.opt, "warmup_epochs", 0))

        loss_dist = torch.tensor(0.0, device=device)

        if dist_w > 0.0 and epoch >= warmup_epochs:
            valid_idx = torch.where(timesteps < mesh_threshold)[0]
            if valid_idx.numel() > 0:
                x0_hat_v = x0_hat[valid_idx]
                decoded = self.vae_decode_to_raw(x0_hat_v)
                gt = motion[valid_idx]
                m_mask = masks[valid_idx].bool()

                if getattr(self.opt, "xyz", False):
                    # ===== XYZ：joint-space MSE（按 joint 平均）+ 手指*10（安全过滤索引） =====
                    # decoded -> [Nv, T, J, 3]
                    if decoded.dim() == 3:
                        Nv, Td, Dd = decoded.shape
                        J_dec = Dd // 3
                        decoded = decoded.view(Nv, Td, J_dec, 3)
                    elif decoded.dim() == 4:
                        J_dec = decoded.shape[2]
                    else:
                        raise ValueError(f"Unexpected decoded shape in xyz mode: {decoded.shape}")

                    # gt -> [Nv, T, J, 3]
                    Nv, Tg, Dg = gt.shape
                    J_gt = Dg // 3
                    gt_j = gt.view(Nv, Tg, J_gt, 3)

                    # 对齐时间
                    T_use = min(decoded.shape[1], gt_j.shape[1], m_mask.shape[1])
                    decoded = decoded[:, :T_use]
                    gt_j = gt_j[:, :T_use]
                    m_mask = m_mask[:, :T_use]

                    # 对齐关节数（保险：以最小 J 为准，避免 silent mismatch）
                    J_use = min(decoded.shape[2], gt_j.shape[2])
                    decoded = decoded[:, :, :J_use]
                    gt_j = gt_j[:, :, :J_use]

                    if m_mask.any():
                        # per_joint: [Nv, T, J]
                        diff = decoded - gt_j
                        per_joint = (diff ** 2).mean(dim=-1)

                        # --- 关键：手指索引安全处理 ---
                        finger_joint_idx = None

                        # 1) 优先用 opt.finger_joint_indices（应该是 joint-level）
                        if hasattr(self.opt, "finger_joint_indices") and self.opt.finger_joint_indices is not None:
                            finger_joint_idx = torch.as_tensor(self.opt.finger_joint_indices, device=device, dtype=torch.long)

                        # 2) 否则尝试用你已有的 flat rec_finger_indices（0..D_flat-1），转成 joint idx
                        elif hasattr(self, "rec_finger_indices") and self.rec_finger_indices is not None:
                            flat_idx = torch.as_tensor(self.rec_finger_indices, device=device, dtype=torch.long)
                            finger_joint_idx = torch.unique(flat_idx // 3)

                        if finger_joint_idx is not None and finger_w != 1.0:
                            # 过滤越界，避免 CUDA index out of bounds
                            finger_joint_idx = finger_joint_idx[(finger_joint_idx >= 0) & (finger_joint_idx < J_use)]
                            if finger_joint_idx.numel() > 0:
                                wj = torch.ones((J_use,), device=device, dtype=per_joint.dtype)
                                wj[finger_joint_idx] = finger_w
                                per_joint = per_joint * wj.view(1, 1, -1)

                        # 有效帧取出来再平均
                        loss_dist = per_joint[m_mask].mean()
                        loss = loss + dist_w * loss_dist

                else:
                    # 旋转/SMPLX dist loss 你后面再整理，这里先保持你原逻辑（不动）
                    pass

        loss_dict["loss_dist"] = loss_dist
        loss_dict["loss_total"] = loss

        return loss, attn_list, loss_dict

    @torch.no_grad()
    def generate(self, batch_data, need_attn=False):
        self.denoiser.eval()
        self.vae.eval()

        text, motion, masks, m_lens, names = batch_data
        device = self.opt.device

        motion = motion.to(device, dtype=torch.float32)
        m_lens = m_lens.to(device, dtype=torch.long)

        # ===== 0) Trim batch padding（不裁内容，只去掉尾部 pad）=====
        B, T_pad, D_flat = motion.shape
        T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
        T_valid = max(1, min(T_valid, T_pad))
        if T_valid < T_pad:
            motion = motion[:, :T_valid]
            masks = masks[:, :T_valid]
            T_pad = T_valid

        # ===== 1) Encode：得到 latent z，Tz 由 VAE 决定 =====
        z, _ = self.vae_encode_raw(motion)  # ✅ raw->norm->encode
        # z 形状可能是 [B, Tz, J, C] 或 [B, Tz, JD]
        if z.dim() == 3:
            Bz, Tz, JD = z.shape
            J = getattr(self.opt, "joints_num", None)
            if J is None:
                # 如果没写 joints_num，就用 JD//3 推断（只在 xyz latent=3 的场景）
                J = JD // 3
            z = z.view(Bz, Tz, J, -1)
        else:
            Tz = z.shape[1]

        # ===== 2) 动态 downsample_ratio：用 motion 和 z 的时间长度推断 =====
        # （必须跟 train_forward 一致，不能写死 //4）
        downsample_ratio = max(1, motion.shape[1] // Tz)
        z_lens = torch.clamp(m_lens // downsample_ratio, min=0, max=Tz)
        len_mask = lengths_to_mask(z_lens).to(device)  # [B, Tz]

        # ===== 3) CFG 输入文本 =====
        # text 可能是 List[str]，也可能是 (emb,mask)
        if self.opt.classifier_free_guidance:
            if isinstance(text, tuple) and len(text) == 2 and torch.is_tensor(text[0]):
                # embedding 模式：uncond -> mask 全 False
                text_emb, text_mask = text
                uncond_mask = torch.zeros_like(text_mask, dtype=torch.bool)
                input_text = (torch.cat([text_emb, text_emb], dim=0),
                            torch.cat([uncond_mask, text_mask], dim=0))
            else:
                input_text = [""] * len(text) + list(text)
        else:
            input_text = text

        # ===== 4) Noise init：latents 的 shape 必须和 z 一样 =====
        latents = torch.randn_like(z) * float(getattr(self.noise_scheduler, "init_noise_sigma", 1.0))
        latents = latents * len_mask[..., None, None].float()

        # ===== 5) DDIM / DPMSolver 等推理步 =====
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        timesteps = self.noise_scheduler.timesteps.to(device)

        skel_attn_weights, temp_attn_weights, cross_attn_weights = [], [], []
        for timestep in timesteps:
            if self.opt.classifier_free_guidance:
                input_latents = torch.cat([latents, latents], dim=0)
                input_len_mask = torch.cat([len_mask, len_mask], dim=0)
            else:
                input_latents = latents
                input_len_mask = len_mask

            pred, attn = self.denoiser.forward(
                input_latents,
                timestep,
                input_text,
                len_mask=input_len_mask,
                need_attn=need_attn,
                use_cached_clip=True
            )

            if self.opt.classifier_free_guidance:
                pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                pred = pred_uncond + self.opt.cond_scale * (pred_cond - pred_uncond)

            latents = self.noise_scheduler.step(pred, timestep, latents).prev_sample
            latents = latents * len_mask[..., None, None].float()

            if need_attn:
                skel_attn_weights.append(attn[0])
                temp_attn_weights.append(attn[1])
                cross_attn_weights.append(attn[2])

        # ===== 6) Decode：latent -> raw motion =====
        pred_motion = self.vae_decode_to_raw(latents)  # ✅ decode->denorm
        if isinstance(pred_motion, (tuple, list)):
            pred_motion = pred_motion[0]

        # pred_motion 可能是 [B, Tm, D_flat] 或 [B, Tm, J, 3]
        if pred_motion.dim() == 3:
            Bp, Tp, Dp = pred_motion.shape
            J = getattr(self.opt, "joints_num", None)
            if J is not None and Dp % J == 0:
                pred_motion = pred_motion.view(Bp, Tp, J, -1)

        if need_attn:
            attn_weights = (
                torch.stack(skel_attn_weights, dim=1),
                torch.stack(temp_attn_weights, dim=1),
                torch.stack(cross_attn_weights, dim=1),
            )
        else:
            attn_weights = (None, None, None)

        self.denoiser.remove_clip_cache()
        return pred_motion, attn_weights

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    

    def save(self, file_name, epoch, total_iter):
        state = {
            "denoiser": self.denoiser.state_dict_without_clip(),
            "optim": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        missing_keys, unexpected_keys = self.denoiser.load_state_dict(checkpoint["denoiser"], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith("clip_model.") for k in missing_keys])

        try:
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except:
            print("Fail to load optimizer and lr_scheduler")
        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)

        # 优化器
        self.optim = torch.optim.AdamW(self.denoiser.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
        print(f"Iters Per Epoch, Training: {len(train_loader)}, Validation: {len(eval_val_loader)}")
        logs = defaultdict(def_value, OrderedDict())
        
        # 初始评估
        metrics = evaluation_denoiser(
            self.opt.model_dir, 
            eval_val_loader, 
            self.denoiser, 
            self.generate, 
            self.logger, 
            epoch,
            physical_evaluator=self.physical_evaluator,
            smplx_model=self.smplx_model,
            opt=self.opt
        )
        print(f"MPJPE BODY {metrics.get('how2sign_MPJPE_body', 1000.0)}") 
        
        best_mpjpe = 1000.0

        # === 训练循环 ===
        while epoch < self.opt.max_epoch:
            torch.cuda.empty_cache()
            self.denoiser.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                self.optim.zero_grad()
                
                # 【修复】移除了 autocast，强制使用 Float32
                loss, attn_list, loss_dict = self.train_forward(batch_data, epoch)

                # 【修复】增加 NaN 检测
                if torch.isnan(loss):
                    print(f"❌ Critical Warning: Loss is NaN at Epoch {epoch} Step {it}. Skipping backward to prevent crash.")
                    print(f"Loss Dict: {loss_dict}")
                    # 可选：如果希望 NaN 就退出，可以 sys.exit(1)，这里选择跳过该 batch
                    continue
                
                loss.backward()
                
                # 【修复】梯度裁剪，防止 XYZ 数据爆炸
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                
                self.optim.step()

                # log
                logs["lr"] += self.optim.param_groups[0]["lr"]
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    
                    train_log_dict = {f"train/{k}": v for k, v in mean_loss.items()}
                    train_log_dict["lr"] = self.optim.param_groups[0]["lr"]
                    wandb.log(train_log_dict, step=it)

                    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in mean_loss.items()])
                    msg = f"[Ep {epoch:03d} | It {it:06d}] {loss_str} | lr: {train_log_dict['lr']:.6f}"
                    self.log_to_file(msg) 

                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
            
            # 【修复】将 LR 更新移到 Epoch 循环末尾
            if it >= self.opt.warm_up_iter:
                self.lr_scheduler.step()
            
            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1
            
            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                metrics = evaluation_denoiser(
                    self.opt.model_dir, 
                    eval_val_loader, 
                    self.denoiser, 
                    self.generate, 
                    self.logger, 
                    epoch,
                    physical_evaluator=self.physical_evaluator,
                    smplx_model=self.smplx_model,
                    opt=self.opt
                )
                if isinstance(metrics, dict):
                    wandb_metrics = {f"eval/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                                     for k, v in metrics.items()}
                    wandb.log(wandb_metrics, step=it)

                current_mpjpe = metrics.get('how2sign_MPJPE_body', 1000.0) 
                if isinstance(current_mpjpe, torch.Tensor): current_mpjpe = current_mpjpe.item()
                
                if current_mpjpe < best_mpjpe:
                    best_mpjpe = current_mpjpe
                    self.save(pjoin(self.opt.model_dir, 'net_best_mpjpe.tar'), epoch, it)
                    self.log_to_file(f"--> --> MPJPE Improved to {best_mpjpe:.5f}!!!")