from typing import List, Union
import wandb
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import sys
import json

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
        # === mesh loss 滑窗最大长度（固定 SMPL-X capacity）===
        self.mesh_loss_window = int(getattr(opt, "mesh_loss_window", 256))  # 128/256/...
        self._smplx_Bcap = int(getattr(opt, "batch_size", 1))
        max_smplx_batch = self._smplx_Bcap * self.mesh_loss_window

        self.smplx_model = smplx.create(
            model_path=smplx_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            flat_hand_mean=True,
            batch_size=max_smplx_batch
        ).to(opt.device).eval()
        print(f"✅ SMPL-X Model initialized with static capacity: Bcap={self._smplx_Bcap} W={self.mesh_loss_window} => {max_smplx_batch}")

            
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
    def _lazy_init_rag(self, device):
        """
        延迟初始化 RAG 相关资源：
        - wmap（gloss -> code seq 的 json）
        - dataset_metadata.json 里推导每个 slot 的 codebook_size -> pad_token_ids
        兼容两种格式：
            A) meta["codebook_sizes"] 直接给了 per-slot size
            B) meta["groups"][...]["codebook_size"] + meta["slot2q_idx"]（你现在用的）
        - build_blueprint_batch 函数句柄
        """
        if getattr(self, "_rag_ready", False):
            return
        if not bool(getattr(self.opt, "use_rag", False)):
            self._rag_ready = False
            return

        import os, json
        import torch

        # 1) import build_blueprint_batch（复用你 MaskGIT 的实现）
        try:
            from models.denoiser.rag import build_blueprint_batch
        except Exception as e:
            raise ImportError(
                "Failed to import build_blueprint_batch from models.denoiser.rag. "
                "Please make sure you have models/denoiser/rag.py and it defines build_blueprint_batch."
            ) from e
        self._build_blueprint_batch = build_blueprint_batch

        # 2) 读 metadata
        meta_path = getattr(self.opt, "rag_metadata_path", None)
        if meta_path is None:
            dataset_root = getattr(self.opt, "rag_dataset_root", None) or getattr(self.opt, "dataset_root", None)
            if dataset_root is None:
                raise ValueError("use_rag=True but rag_metadata_path/rag_dataset_root/dataset_root is not set")
            meta_name = getattr(self.opt, "rag_metadata_filename", "dataset_metadata.json")
            meta_path = os.path.join(dataset_root, meta_name)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 3) 推导 rag_K
        # 优先用 opt.rag_K，其次用 meta["K"]，否则报错
        if hasattr(self.opt, "rag_K"):
            rag_K = int(getattr(self.opt, "rag_K"))
        elif "K" in meta:
            rag_K = int(meta["K"])
        else:
            raise ValueError(f"[RAG] Cannot determine K. Please set opt.rag_K or ensure metadata has key 'K'. meta_path={meta_path}")

        self._rag_K = rag_K

        # 4) 推导每个 slot 的 codebook_size

        # 4.2 你现在的格式：groups + slot2q_idx

        if "slot2q_idx" not in meta or "groups" not in meta:
            raise ValueError(
                f"[RAG] metadata must contain either 'codebook_sizes' OR ('slot2q_idx' and 'groups'). "
                f"Got keys={list(meta.keys())} in {meta_path}"
            )

        slot2q_idx = meta["slot2q_idx"]
        groups = meta["groups"]

        if len(slot2q_idx) < rag_K:
            raise ValueError(f"[RAG] slot2q_idx length={len(slot2q_idx)} < rag_K={rag_K} ({meta_path})")

        # 建 q_idx -> codebook_size 映射
        qidx2size = {}
        for g in groups:
            if not isinstance(g, dict):
                continue
            if "q_idx" not in g or "codebook_size" not in g:
                continue
            q = int(g["q_idx"])
            sz = int(g["codebook_size"])
            qidx2size[q] = sz

        # 生成 per-slot codebook_size
        codebook_sizes = []
        for k in range(rag_K):
            q = int(slot2q_idx[k])
            if q not in qidx2size:
                raise ValueError(f"[RAG] q_idx={q} (from slot2q_idx[{k}]) not found in groups q_idx list. ({meta_path})")
            codebook_sizes.append(int(qidx2size[q]))

        self._rag_codebook_sizes = codebook_sizes

        # pad_token_id 规则：正常 token [0..cb-1], mask=cb, pad=cb+1
        self._rag_pad_token_ids = torch.tensor([int(cb) + 1 for cb in codebook_sizes], device=device, dtype=torch.long)

        # 让 denoiser 能初始化 rag_token_embs
        setattr(self.denoiser, "_rag_codebook_sizes", codebook_sizes)

        # 5) 读 wmap（必须与 rag.build_blueprint_batch 期望格式一致）
        from models.denoiser.rag import _load_wlasl_map

        # dataset_root：优先 rag_dataset_root，其次 dataset_root；都没有就用 metadata 所在目录
        dataset_root = getattr(self.opt, "rag_dataset_root", None) or getattr(self.opt, "dataset_root", None)
        if dataset_root is None:
            dataset_root = os.path.dirname(meta_path)

        self._rag_wmap = _load_wlasl_map(dataset_root)

        self._rag_ready = True

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
        #print(message)
        if hasattr(self, 'log_file'):
            self.log_file.write(message + "\n")
            self.log_file.flush()
    def train_forward(self, batch_data, epoch):
        """
        batch_data: (text, motion, masks, m_lens, names)
        text:
        - List[str]
        - 或 (text_emb[B,L,D], text_mask[B,L], raw_texts[List[str]])
        motion: [B, T_pad, D_flat]
        masks:  [B, T_pad]
        m_lens: [B]
        """
        text, motion, masks, m_lens, names = batch_data
        device = self.opt.device

        # ===== cond drop（兼容 str 或 (emb,mask,raw_texts)）=====
        p_drop = float(getattr(self.opt, "cond_drop_prob", 0.0))
        dropped = False
        if p_drop > 0:
            if isinstance(text, tuple) and len(text) >= 2 and torch.is_tensor(text[0]):
                text_emb, text_mask = text[0], text[1]
                raw_texts = text[2] if len(text) >= 3 else None

                if np.random.rand(1) < p_drop:
                    text_mask = torch.zeros_like(text_mask, dtype=torch.bool)
                    dropped = True
                    if raw_texts is not None:
                        raw_texts = [""] * len(raw_texts)

                if raw_texts is not None:
                    text = (text_emb, text_mask, raw_texts)
                else:
                    text = (text_emb, text_mask)
            else:
                # List[str]
                new_text = []
                for t in text:
                    if np.random.rand(1) < p_drop:
                        new_text.append("")
                    else:
                        new_text.append(t)
                text = new_text

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


        if bool(getattr(self.opt, "use_rag", False)) and isinstance(text, tuple) and len(text) >= 3:
            # text 必须是 (text_emb, text_mask, raw_texts, ...)
            if (not torch.is_tensor(text[0])) or text[0].dim() != 3:
                raise ValueError(f"[RAG] text[0] must be a tensor [B,L,D], got {type(text[0])} shape={getattr(text[0], 'shape', None)}")
            if (not torch.is_tensor(text[1])) or text[1].dim() != 2:
                raise ValueError(f"[RAG] text[1] must be a tensor [B,L] mask, got {type(text[1])} shape={getattr(text[1], 'shape', None)}")

            self._lazy_init_rag(device=device)

            raw_texts = text[2]
            if not isinstance(raw_texts, (list, tuple)):
                raise ValueError(f"[RAG] text[2] must be List[str]/Tuple[str], got {type(raw_texts)}")

            bp_tokens, bp_pad_mask, bp_stats = self._build_blueprint_batch(
                glosses=raw_texts,
                wmap=self._rag_wmap,
                pad_token_ids=self._rag_pad_token_ids,
                device=device,
                K=self._rag_K,
                max_words=int(getattr(self.opt, "rag_max_words", 64)),
                per_word_max_T=int(getattr(self.opt, "rag_per_word_max_T", 48)),
                total_max_T=int(getattr(self.opt, "rag_total_max_T", 384)),
                names=names,
                epoch=epoch,
                mode="train",
            )

            # (optional) print hit-rate occasionally to verify it's working
            if epoch % 100 == 0:

                Tb = int(bp_tokens.shape[1])
                if (not self.denoiser.training):
                    # eval: 只打印第一个 batch
                    if not getattr(self, "_printed_rag_eval_once", False):
                        print(f"[WLASL] hit_rate={bp_stats['hit_rate']:.3f} Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})")

                        # ✅ 这里别再用 text.shape —— text 是 tuple
                        L = int(text[0].shape[1])          # text_emb 的 L
                        print(f"[COND] text_L={L} bp_Tb={Tb} cond_L={L+Tb}")
                        self._printed_rag_eval_once = True

        pred, attn_list = self.denoiser.forward(
            noisy_latent,
            timesteps,
            text,
            len_mask=len_mask,
            blueprint_tokens=bp_tokens,
            blueprint_pad_mask=bp_pad_mask,
        )
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
                    # ===== ROT：SMPLX vertex loss（按 vertex 平均）+ 手部顶点*10，支持分组 index =====
                    # decoded 期望是 [Nv, T, 43*3] 或 [Nv, T, 43, 3]
                    if decoded.dim() == 3:
                        Nv, Td, Dd = decoded.shape
                        decoded_r = decoded.view(Nv, Td, 43, 3)
                    elif decoded.dim() == 4:
                        decoded_r = decoded
                    else:
                        raise ValueError(f"Unexpected decoded shape in rot mode: {decoded.shape}")

                    gt_r = gt.view(gt.shape[0], gt.shape[1], 43, 3)

                    T_use = min(decoded_r.shape[1], gt_r.shape[1], m_mask.shape[1])
                    decoded_r = decoded_r[:, :T_use]
                    gt_r = gt_r[:, :T_use]
                    m_mask = m_mask[:, :T_use]

                    # === mesh loss 滑窗抽样：固定喂给 SMPL-X 的帧数 = Bcap * W，避免变长 ===
                    Bcur = decoded_r.shape[0]
                    W = int(getattr(self, "mesh_loss_window", 256))
                    Bcap = int(getattr(self, "_smplx_Bcap", Bcur))

                    # 只在 mesh loss 分支里做 pad 到 Bcap（最后一个 batch 可能 < opt.batch_size）
                    # [Bcap, T_use, 43, 3] / [Bcap, T_use]
                    if Bcur < Bcap:
                        pad_n = Bcap - Bcur
                        decoded_r = torch.cat([decoded_r, torch.zeros((pad_n, T_use, 43, 3), device=device, dtype=decoded_r.dtype)], dim=0)
                        gt_r      = torch.cat([gt_r,      torch.zeros((pad_n, T_use, 43, 3), device=device, dtype=gt_r.dtype)], dim=0)
                        m_mask    = torch.cat([m_mask,    torch.zeros((pad_n, T_use),       device=device, dtype=torch.bool)], dim=0)
                        Bcur = Bcap

                    # 每个样本有效长度（默认前段连续 True）
                    lengths = m_mask.long().sum(dim=1)  # [Bcap]
                    starts = torch.zeros((Bcap,), device=device, dtype=torch.long)

                    # 采样后的窗口张量
                    pd_win = torch.zeros((Bcap, W, 43, 3), device=device, dtype=decoded_r.dtype)
                    gt_win = torch.zeros((Bcap, W, 43, 3), device=device, dtype=gt_r.dtype)
                    win_mask = torch.zeros((Bcap, W), device=device, dtype=torch.bool)  # True=有效帧

                    for b in range(Bcap):
                        L = int(lengths[b].item())
                        if L <= 0:
                            starts[b] = 0
                            continue

                        if L <= W:
                            s = 0
                            pd_win[b, :L] = decoded_r[b, :L]
                            gt_win[b, :L] = gt_r[b, :L]
                            win_mask[b, :L] = m_mask[b, :L]   # 只拷贝有效段，后面保持 False
                        else:
                            # train 随机连续裁剪；eval 固定从 0（你也可以改成 center）
                            if self.denoiser.training:
                                s = int(torch.randint(low=0, high=L - W + 1, size=(1,), device=device).item())
                            else:
                                s = 0
                            pd_win[b] = decoded_r[b, s:s+W]
                            gt_win[b] = gt_r[b, s:s+W]
                            win_mask[b, :] = m_mask[b, s:s+W]

                        starts[b] = s

                    # 记录：每个样本窗口起点 & 有效长度（你要“记录有效帧是哪些”，这俩足够还原）
                    loss_dict["mesh_win_start"] = starts.detach().cpu()
                    loss_dict["mesh_win_len"]   = lengths.detach().cpu()
                    #loss_dict["mesh_win_W"]     = int(W)

                    # flatten 到 [Bcap*W, 43, 3]，并给出有效帧 mask
                    pd_v = pd_win.reshape(Bcap * W, 43, 3)
                    gt_v = gt_win.reshape(Bcap * W, 43, 3)
                    valid_flat = win_mask.reshape(Bcap * W)
                    def split_smplx(x):
                        body = x[:, :13]
                        lhand = x[:, 13:28]
                        rhand = x[:, 28:43]
                        restored = torch.zeros(x.shape[0], 22, 3, device=device, dtype=x.dtype)
                        # 你 VAE 里用 self.body_indices 缓存映射，这里也照用
                        restored[:, self.body_indices] = body
                        return restored[:, 1:], lhand, rhand
                    if not valid_flat.any():
                        loss_dist = torch.tensor(0.0, device=device)
                        # 这一步直接跳过 SMPL-X forward，避免 None 输入
                        # 如果你在一个更大的 loss 分支内部，就用 `loss_dict["loss_dist"]=loss_dist` 后 `continue`
                        loss = loss + dist_w * loss_dist

                    else:


                        if (pd_v is not None) and (gt_v is not None):
                            gt_body, gt_lh, gt_rh = split_smplx(gt_v)
                            pd_body, pd_lh, pd_rh = split_smplx(pd_v)
                        else:
                            gt_body = gt_lh = gt_rh = pd_body = pd_lh = pd_rh = None

                        with torch.no_grad():
                            out_gt = self.smplx_model(body_pose=gt_body, left_hand_pose=gt_lh, right_hand_pose=gt_rh)
                        out_pd = self.smplx_model(body_pose=pd_body, left_hand_pose=pd_lh, right_hand_pose=pd_rh)

                        # 取你关心的顶点集合
                        verts_gt = out_gt.vertices[:, self.all_verts_indices, :]  # [Nf, V, 3]
                        verts_pd = out_pd.vertices[:, self.all_verts_indices, :]  # [Nf, V, 3]

                        # elementwise error
                        mesh_loss_type = getattr(self.opt, "recon_loss", "l1")
                        mesh_mm_scale   = float(getattr(self.opt, "mesh_mm_scale", 1000.0))
                        verts_gt_mm = verts_gt * mesh_mm_scale
                        verts_pd_mm = verts_pd * mesh_mm_scale
                        if mesh_loss_type == "l1":
                            elem = torch.nn.functional.smooth_l1_loss(verts_pd_mm, verts_gt_mm, reduction="none")
                        elif mesh_loss_type == "l1_smooth":
                            elem = torch.nn.functional.smooth_l1_loss(verts_pd, verts_gt, reduction="none")
                        elif mesh_loss_type == "l2":
                            elem = (verts_pd_mm - verts_gt_mm) ** 2

                        else:
                            raise NotImplementedError(f"recon_loss={mesh_loss_type} not supported for mesh loss")

                        # 只在有效帧上算（valid_flat: [B*W] bool）
                        # self.vertex_weights: [1, V, 1]（你 init 里已经把手部顶点加权了）
                        elem_valid = elem[valid_flat]  # [N_valid, V, 3]
                        loss_dist = (elem_valid * self.vertex_weights).mean()

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

        # ===== 0) Trim batch padding =====
        B, T_pad, D_flat = motion.shape
        T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
        T_valid = max(1, min(T_valid, T_pad))
        if T_valid < T_pad:
            motion = motion[:, :T_valid]
            masks = masks[:, :T_valid]
            T_pad = T_valid

        # ===== 1) Encode =====
        z, _ = self.vae_encode_raw(motion)
        if z.dim() == 3:
            Bz, Tz, JD = z.shape
            J = getattr(self.opt, "joints_num", None)
            if J is None:
                J = JD // 3
            z = z.view(Bz, Tz, J, -1)
        else:
            Tz = z.shape[1]

        downsample_ratio = max(1, motion.shape[1] // Tz)
        z_lens = torch.clamp(m_lens // downsample_ratio, min=0, max=Tz)
        len_mask = lengths_to_mask(z_lens).to(device)  # [B, Tz]

        # ===== 2) CFG 输入文本（支持 (emb,mask,raw_texts)）=====
        input_text = text
        if self.opt.classifier_free_guidance:
            if isinstance(text, tuple) and len(text) >= 2 and torch.is_tensor(text[0]):
                text_emb, text_mask = text[0], text[1]
                raw_texts = text[2] if len(text) >= 3 else None

                uncond_mask = torch.zeros_like(text_mask, dtype=torch.bool)

                if raw_texts is not None:
                    raw_u = [""] * len(raw_texts)
                    input_text = (
                        torch.cat([text_emb, text_emb], dim=0),
                        torch.cat([uncond_mask, text_mask], dim=0),
                        raw_u + list(raw_texts),
                    )
                else:
                    input_text = (
                        torch.cat([text_emb, text_emb], dim=0),
                        torch.cat([uncond_mask, text_mask], dim=0),
                    )
            else:
                input_text = [""] * len(text) + list(text)

        # ===== 3) RAG blueprint（一次性 build，循环里复用）=====
        bp_tokens, bp_pad_mask = None, None
        if bool(getattr(self.opt, "use_rag", False)) and isinstance(input_text, tuple) and len(input_text) >= 3:
            self._lazy_init_rag(device=device)
            raw_texts = input_text[2]
            bp_tokens, bp_pad_mask, bp_stats = self._build_blueprint_batch(
                glosses=raw_texts,
                wmap=self._rag_wmap,
                pad_token_ids=self._rag_pad_token_ids,
                device=device,
                K=self._rag_K,
                max_words=int(getattr(self.opt, "rag_max_words", 64)),
                per_word_max_T=int(getattr(self.opt, "rag_per_word_max_T", 48)),
                total_max_T=int(getattr(self.opt, "rag_total_max_T", 384)),
                names=names,
                mode="infer",
            )
            Tb = int(bp_tokens.shape[1])
            if (not self.denoiser.training):
                # eval: 只打印第一个 batch
                if not getattr(self, "_printed_rag_eval_once", False):
                    print(f"[WLASL] hit_rate={bp_stats['hit_rate']:.3f} Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})")

                    # ✅ 这里别再用 text.shape —— text 是 tuple
                    L = int(text[0].shape[1])          # text_emb 的 L
                    print(f"[COND] text_L={L} bp_Tb={Tb} cond_L={L+Tb}")
                    self._printed_rag_eval_once = True
        # ===== 4) Noise init =====
        latents = torch.randn_like(z) * float(getattr(self.noise_scheduler, "init_noise_sigma", 1.0))
        latents = latents * len_mask[..., None, None].float()

        # ===== 5) 推理步 =====
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
                use_cached_clip=True,
                blueprint_tokens=bp_tokens,
                blueprint_pad_mask=bp_pad_mask,
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

        # ===== 6) Decode =====
        pred_motion = self.vae_decode_to_raw(latents)
        if isinstance(pred_motion, (tuple, list)):
            pred_motion = pred_motion[0]

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

        # ===== 关键修复：在 load_state_dict 前把 RAG 子模块建出来 =====
        den_sd = checkpoint["denoiser"]
        has_rag_in_ckpt = any(k.startswith("rag_") for k in den_sd.keys())

        if has_rag_in_ckpt and bool(getattr(self.opt, "use_rag", False)):
            # 1) 先初始化 trainer 侧的 RAG 资源（会把 codebook_sizes 塞进 denoiser._rag_codebook_sizes）
            self._lazy_init_rag(self.opt.device)

            # 2) 强制让 denoiser 创建 rag_* 子模块（否则 load 会看到一堆 unexpected）
            if hasattr(self.denoiser, "_maybe_init_rag"):
                self.denoiser._maybe_init_rag(self.opt.device)

        missing_keys, unexpected_keys = self.denoiser.load_state_dict(den_sd, strict=False)

        # ===== 更合理的断言规则 =====
        # 允许：layer=0 时 ckpt 里存在 rag_encoder.*（因为当前 rag_encoder=None）
        allow_unexpected = []
        if bool(getattr(self.opt, "use_rag", False)) and getattr(self.denoiser, "rag_encoder", None) is None:
            allow_unexpected.append("rag_encoder.")

        unexpected_bad = [k for k in unexpected_keys if not any(k.startswith(p) for p in allow_unexpected)]
        assert len(unexpected_bad) == 0, f"Unexpected keys (not allowed): {unexpected_bad[:30]}"

        # missing_keys 只允许 clip（你原本逻辑）
        assert all([k.startswith("clip_model.") for k in missing_keys]), f"Missing keys: {missing_keys[:30]}"

        # optim / scheduler
        self.optim.load_state_dict(checkpoint["optim"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

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
        self._printed_rag_eval_once = False
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

            logs = defaultdict(def_value, OrderedDict())
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
                    # 只记录标量；像 mesh_win_start/mesh_win_len 这种 [B] 的调试张量直接跳过
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            continue
                        logs[tag] += value.detach().float().item()
                    elif isinstance(value, (float, int)):
                        logs[tag] += float(value)
                    else:
                        continue

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
            # ===== epoch end: print/log once =====
            mean_loss = OrderedDict()
            denom = max(1, len(train_loader))
            for tag, value in logs.items():
                avg_v = value / denom
                mean_loss[tag] = avg_v
                self.logger.add_scalar(f'Train/{tag}', avg_v, it)

            train_log_dict = {f"train/{k}": v for k, v in mean_loss.items()}
            train_log_dict["lr"] = self.optim.param_groups[0]["lr"]
            wandb.log(train_log_dict, step=it)

            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in mean_loss.items()])
            msg = f"[Ep {epoch:03d} | It {it:06d}] {loss_str} | lr: {train_log_dict['lr']:.6f}"

            self.log_to_file(msg)
            print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=(len(train_loader)-1))

            # 【修复】将 LR 更新移到 Epoch 循环末尾
            if it >= self.opt.warm_up_iter:
                self.lr_scheduler.step()
            
            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1
            
            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                self._printed_rag_eval_once = False
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