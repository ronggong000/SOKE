from typing import List, Union
import contextlib
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import json

import time
import numpy as np
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_denoiser, test_denoiser
from utils.utils import print_current_loss, attn2img
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from physical_evaluator import SignPhysicalEvaluator
from vae_adapter import extract_pose_from_motion_tensor
import smplx
from torch.amp import autocast # 仅保留引用防止报错，实际不使用
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except Exception:
    class _WandbStub:
        @staticmethod
        def log(*args, **kwargs):
            return

    wandb = _WandbStub()

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
        self.is_master = bool(getattr(opt, "is_master", True))
        self.is_distributed = bool(getattr(opt, "distributed", False))
        self.denoiser = denoiser.to(opt.device)
        self.vae = vae.to(opt.device)
        self.noise_scheduler = scheduler
        
        self.mesh_loss_window = int(getattr(opt, "mesh_loss_window", 256))
        self._smplx_Bcap = int(getattr(opt, "batch_size", 1))
        self.smplx_model_path = str(getattr(opt, "smplx_model_path", "") or "")
        if not self.smplx_model_path:
            self.smplx_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "deps", "smpl_models")
            )

        if self.is_master:
            self.physical_evaluator = SignPhysicalEvaluator(opt, opt.device)
            max_smplx_batch = self._smplx_Bcap * self.mesh_loss_window
            self.smplx_model = smplx.create(
                model_path=self.smplx_model_path,
                model_type='smplx',
                gender='neutral',
                use_pca=False,
                flat_hand_mean=True,
                batch_size=max_smplx_batch
            ).to(opt.device).eval()
            print(
                f"✅ SMPL-X Model initialized with static capacity: Bcap={self._smplx_Bcap} "
                f"W={self.mesh_loss_window} => {max_smplx_batch}, path={self.smplx_model_path}"
            )
        else:
            self.physical_evaluator = None
            self.smplx_model = None

            
        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir) if self.is_master else None
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()
            elif opt.recon_loss == "l2":
                self.recon_criterion = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"Reconstruction loss {opt.recon_loss} not implemented")
            
        if opt.is_train and self.is_master:
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
        amp_dtype = str(getattr(opt, "amp_dtype", "none") or "none").strip().lower()
        if amp_dtype == "bf16":
            self._amp_enabled = True
            self._amp_dtype = torch.bfloat16
            self._grad_scaler = None
        elif amp_dtype == "fp16":
            self._amp_enabled = True
            self._amp_dtype = torch.float16
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=(opt.device.type == "cuda"))
        else:
            self._amp_enabled = False
            self._amp_dtype = torch.float32
            self._grad_scaler = None
        self._amp_use = bool(self._amp_enabled and (opt.device.type == "cuda"))
        if self.is_master:
            print(f"[AMP] mode={amp_dtype} enabled={self._amp_use}")

    def _dist_barrier(self):
        if self.is_distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _denoiser_module(self):
        return self.denoiser.module if isinstance(self.denoiser, DDP) else self.denoiser

    def _autocast_ctx(self):
        if self._amp_use:
            return autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()
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

        import json
        import os
        import torch

        # 1) import build_blueprint_batch
        try:
            from models.denoiser.rag import (
                _load_wlasl_map,
                build_blueprint_batch,
                infer_rag_layout,
                resolve_rag_metadata_path,
                resolve_rag_wmap_source,
            )
        except Exception as e:
            raise ImportError(
                "Failed to import build_blueprint_batch from models.denoiser.rag. "
                "Please make sure you have models/denoiser/rag.py and it defines build_blueprint_batch."
            ) from e
        self._build_blueprint_batch = build_blueprint_batch

        # 2) 读 metadata
        meta_path = resolve_rag_metadata_path(
            rag_metadata_path=getattr(self.opt, "rag_metadata_path", None),
            rag_wmap_path=getattr(self.opt, "rag_wmap_path", None),
            rag_dataset_root=getattr(self.opt, "rag_dataset_root", None),
            dataset_root=getattr(self.opt, "dataset_root", None),
            meta_name=getattr(self.opt, "rag_metadata_filename", "dataset_metadata.json"),
        )
        if not meta_path:
            raise ValueError("use_rag=True but rag metadata cannot be resolved from rag_metadata_path/rag_wmap_path/dataset_root")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 3) 推导 rag_K / slot 子集 / codebook_sizes
        rag_layout = infer_rag_layout(
            meta,
            rag_k=getattr(self.opt, "rag_K", None),
            rag_slot_names=getattr(self.opt, "rag_slot_names", ""),
        )
        rag_K = int(rag_layout["rag_k"])
        codebook_sizes = list(rag_layout["codebook_sizes"])

        self._rag_K = rag_K
        self._rag_slot_names = list(rag_layout["slot_names"])

        self._rag_codebook_sizes = codebook_sizes

        # pad_token_id 规则：正常 token [0..cb-1], mask=cb, pad=cb+1
        self._rag_pad_token_ids = torch.tensor([int(cb) + 1 for cb in codebook_sizes], device=device, dtype=torch.long)

        # 让 denoiser 能初始化 rag_token_embs
        setattr(self._denoiser_module(), "_rag_codebook_sizes", codebook_sizes)

        # 5) 读 token 词典/样本表
        wmap_source = resolve_rag_wmap_source(
            rag_wmap_path=getattr(self.opt, "rag_wmap_path", None),
            rag_dataset_root=getattr(self.opt, "rag_dataset_root", None),
            dataset_root=getattr(self.opt, "dataset_root", None),
            meta_path=meta_path,
        )
        if not wmap_source:
            raise FileNotFoundError(
                f"[RAG] cannot resolve token source from rag_wmap_path={getattr(self.opt, 'rag_wmap_path', None)}"
            )

        self._rag_wmap = _load_wlasl_map(
            wmap_source,
            rag_meta=meta,
            slot_indices=rag_layout["slot_indices"],
            gloss_csv_dir=getattr(self.opt, "rag_gloss_csv_dir", ""),
            gloss_source_col=getattr(self.opt, "rag_gloss_source_col", "Video file"),
            gloss_target_col=getattr(self.opt, "rag_gloss_target_col", "my_gloss"),
            rag_weight_dir=getattr(self.opt, "rag_weight_dir", ""),
        )
        print(
            f"[RAG] trainer ready: meta={meta_path} source={wmap_source} "
            f"K={rag_K} slots={rag_layout['slot_names']}"
        )

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
            if self.is_master:
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

    def _unpack_batch(self, batch_data):
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) >= 6:
                text, motion, masks, m_lens, names, frame_weights = batch_data[:6]
                return text, motion, masks, m_lens, names, frame_weights
            if len(batch_data) == 5:
                text, motion, masks, m_lens, names = batch_data
                return text, motion, masks, m_lens, names, None
        raise ValueError(f"Unexpected batch_data format in trainer: type={type(batch_data)}")

    def _vae_time_pool_steps(self) -> int:
        if hasattr(self.vae, "opt") and hasattr(self.vae.opt, "n_layers"):
            try:
                return max(1, int(self.vae.opt.n_layers))
            except Exception:
                pass
        unit_length = int(getattr(self.opt, "unit_length", 4))
        if unit_length >= 2 and (unit_length & (unit_length - 1)) == 0:
            return max(1, int(round(np.log2(unit_length))))
        return 2

    def _reduce_recon(self, pred, target, len_mask, temporal_weight=None):
        if self.opt.recon_loss == "l1":
            point = torch.abs(pred - target)
        elif self.opt.recon_loss == "l1_smooth":
            point = F.smooth_l1_loss(pred, target, reduction="none")
        elif self.opt.recon_loss == "l2":
            point = (pred - target) ** 2
        else:
            raise NotImplementedError(f"Reconstruction loss {self.opt.recon_loss} not implemented")

        w = len_mask.float()
        if temporal_weight is not None:
            w = w * temporal_weight.float().clamp(min=0.0, max=1.0)
        w = w.unsqueeze(-1).unsqueeze(-1)
        num = (point * w).sum()
        tail_dim = 1
        for s in point.shape[2:]:
            tail_dim *= int(s)
        den = (w.sum() * tail_dim).clamp(min=1e-6)
        return num / den

    def _pool_frame_weight_to_latent(self, frame_weights, latent_len, len_mask):
        if frame_weights is None:
            return None
        if frame_weights.dim() == 1:
            frame_weights = frame_weights.unsqueeze(0)
        if frame_weights.dim() > 2:
            frame_weights = frame_weights.view(frame_weights.shape[0], frame_weights.shape[1], -1)[..., 0]

        fw = frame_weights.float().clamp(min=0.0, max=1.0).unsqueeze(1)  # [B,1,T]
        for _ in range(self._vae_time_pool_steps()):
            if fw.shape[-1] <= 1:
                break
            fw = F.avg_pool1d(fw, kernel_size=2, stride=2)
        if fw.shape[-1] != int(latent_len):
            fw = F.interpolate(fw, size=int(latent_len), mode="linear", align_corners=False)
        fw = fw.squeeze(1) * len_mask.float()
        return fw.clamp(min=0.0, max=1.0)

    def train_forward(self, batch_data, epoch):
        """
        batch_data: (text, motion, masks, m_lens, names[, frame_weights])

        V3 训练前向：
        - condition 只走 gloss token-level（vocab） + RAG blueprint
        - 不依赖 CLIP / text embedding
        - cond drop / mismatch 会同时作用到 gloss + rag
        """
        text, motion, masks, m_lens, names, frame_weights = self._unpack_batch(batch_data)
        device = self.opt.device

        # -----------------------------
        # 0) 解析 raw_texts（List[str] 或 List[[eng,gloss]]）
        # -----------------------------
        raw_texts = None
        if isinstance(text, tuple) and len(text) >= 3 and isinstance(text[2], (list, tuple)):
            raw_texts = list(text[2])
        elif isinstance(text, (list, tuple)):
            raw_texts = list(text)
        else:
            # 极端 fallback
            B0 = int(motion.shape[0]) if torch.is_tensor(motion) else 1
            raw_texts = [""] * B0

        # 抽取 gloss 字符串（供 rag）
        def _extract_gloss(rt):
            if isinstance(rt, (list, tuple)) and len(rt) >= 2:
                return "" if rt[1] is None else str(rt[1])
            return "" if rt is None else str(rt)

        # -----------------------------
        # 1) cond drop：每个样本独立 drop（同时影响 gloss + rag）
        # -----------------------------
        p_drop = float(getattr(self.opt, "cond_drop_prob", 0.0))
        if p_drop > 0:
            B0 = len(raw_texts)
            drop = (torch.rand(B0, device=device) < p_drop).tolist()
            if any(drop):
                new_raw = []
                for i, rt in enumerate(raw_texts):
                    if drop[i]:
                        if isinstance(rt, (list, tuple)) and len(rt) >= 2:
                            new_raw.append(["", ""])
                        else:
                            new_raw.append("")
                    else:
                        new_raw.append(rt)
                raw_texts = new_raw

        glosses = [_extract_gloss(rt) for rt in raw_texts]

        # -----------------------------
        # 2) motion -> latent（保持你原逻辑）
        # -----------------------------
        masks = masks.to(device)
        m_lens = m_lens.to(device, dtype=torch.long)
        use_cached_latent = torch.is_tensor(motion) and motion.ndim == 4

        if use_cached_latent:
            latent = motion.to(device=device, dtype=torch.float32)
            B, Tz, _, _ = latent.shape
            curr_m_lens = torch.clamp(m_lens, min=0, max=Tz)
            len_mask = lengths_to_mask(curr_m_lens, max_len=Tz).to(device)
            if frame_weights is not None:
                frame_weights = frame_weights.to(device, dtype=torch.float32)
                if frame_weights.dim() > 2:
                    frame_weights = frame_weights.view(frame_weights.shape[0], frame_weights.shape[1], -1)[..., 0]
                if frame_weights.shape[1] != Tz:
                    tmp = frame_weights.unsqueeze(1)
                    tmp = F.interpolate(tmp, size=Tz, mode="linear", align_corners=False)
                    frame_weights = tmp.squeeze(1)
        else:
            motion = motion.to(device, dtype=torch.float32)
            B, T_pad, D_flat = motion.shape

            # ===== 0) Trim：只去掉 batch padding 尾巴（不裁内容）=====
            T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
            T_valid = max(1, min(T_valid, T_pad))
            if T_valid < T_pad:
                motion = motion[:, :T_valid]
                masks = masks[:, :T_valid]
            if frame_weights is not None:
                frame_weights = frame_weights.to(device, dtype=torch.float32)
                if frame_weights.dim() > 2:
                    frame_weights = frame_weights.view(frame_weights.shape[0], frame_weights.shape[1], -1)[..., 0]
                if frame_weights.shape[1] > T_valid:
                    frame_weights = frame_weights[:, :T_valid]
                elif frame_weights.shape[1] < T_valid:
                    tmp = frame_weights.unsqueeze(1)
                    tmp = F.interpolate(tmp, size=T_valid, mode="linear", align_corners=False)
                    frame_weights = tmp.squeeze(1)

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

        temporal_weight = None
        if bool(getattr(self.opt, "enable_custom_weight", False)):
            if frame_weights is None:
                if not getattr(self, "_warned_missing_custom_weight", False):
                    print("[CustomWeight][WARN] enable_custom_weight=True but batch has no frame_weights. Fallback to unweighted loss.")
                    self._warned_missing_custom_weight = True
            else:
                if use_cached_latent:
                    temporal_weight = frame_weights.float().clamp(0.0, 1.0) * len_mask.float()
                else:
                    temporal_weight = self._pool_frame_weight_to_latent(frame_weights, latent.shape[1], len_mask)

        # ===== 2) Diffusion =====
        timesteps = torch.randint(0, self.opt.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn_like(latent)
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # -----------------------------
        # 3) RAG blueprint（若开启则一定参与 condition）
        # -----------------------------
        bp_tokens, bp_pad_mask, bp_weights = None, None, None
        if bool(getattr(self.opt, "use_rag", False)):
            self._lazy_init_rag(device=device)
            bp_tokens, bp_pad_mask, bp_weights, bp_stats = self._build_blueprint_batch(
                glosses=glosses,  # 只传 gloss 字符串
                wmap=self._rag_wmap,
                pad_token_ids=self._rag_pad_token_ids,
                device=device,
                K=self._rag_K,
                max_words=int(getattr(self.opt, "rag_max_words", 64)),
                per_word_max_T=int(getattr(self.opt, "rag_per_word_max_T", 1)),
                total_max_T=int(getattr(self.opt, "rag_total_max_T", 384)),
                frame_subsample=int(getattr(self.opt, "rag_frame_subsample", 0)),
                slot_names=getattr(self, "_rag_slot_names", None),
                weight_key=str(getattr(self.opt, "rag_weight_key", "soft_w")),
                weight_max_mix=float(getattr(self.opt, "rag_weight_max_mix", 0.5)),
                names=names,
                epoch=epoch,
                mode="train",
            )

            if epoch % 100 == 0 and not getattr(self, "_printed_rag_train_once", False):
                print(f"[WLASL] hit_rate={bp_stats['hit_rate']:.3f} Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})")
                self._printed_rag_train_once = True

        # -----------------------------
        # 4) forward good cond
        # -----------------------------
        pred, attn_list = self.denoiser.forward(
            noisy_latent,
            timesteps,
            raw_texts,
            len_mask=len_mask,
            blueprint_tokens=bp_tokens,
            blueprint_weights=bp_weights,
            blueprint_pad_mask=bp_pad_mask,
            use_cached_clip=True,   # V3: no-op
        )
        pred = pred * len_mask[..., None, None].float()

        # -----------------------------
        # 5) base loss（保持你原逻辑）
        # -----------------------------
        loss_dict = {}
        loss = torch.tensor(0.0, device=device)
        pred_w = float(getattr(self.opt, "pred_loss_weight", 1.0))

        if self.opt.prediction_type == "sample":
            target = latent
            base = self.recon_criterion(pred, target)
            loss_dict["loss_sample"] = base

        elif self.opt.prediction_type == "epsilon":
            target = noise
            base = self.recon_criterion(pred, target)
            loss_dict["loss_eps"] = base

        elif self.opt.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
            if temporal_weight is not None:
                base = self._reduce_recon(pred, target, len_mask, temporal_weight=temporal_weight)
            else:
                base = self.recon_criterion(pred, target)
            loss_dict["loss_vel"] = base

        else:
            raise ValueError(f"prediction_type must be one of ['sample','epsilon','v_prediction'], got {self.opt.prediction_type}")

        loss = loss + pred_w * base

        # -----------------------------
        # 6) mismatch ranking：同步错配 gloss + rag
        # -----------------------------
        mm_w = float(getattr(self.opt, "mismatch_text_weight", 0.0))
        mm_margin = float(getattr(self.opt, "mismatch_text_margin", 0.0))
        if mm_w > 0:
            perm = torch.randperm(B, device=device).tolist()

            bad_raw = []
            bad_glosses = []
            for i in range(B):
                rt_i = raw_texts[i]
                gj = _extract_gloss(raw_texts[perm[i]])

                if isinstance(rt_i, (list, tuple)) and len(rt_i) >= 2:
                    bad_raw.append([rt_i[0], gj])
                else:
                    bad_raw.append(gj)
                bad_glosses.append(gj)

            bp_tokens_bad, bp_pad_mask_bad, bp_weights_bad = bp_tokens, bp_pad_mask, bp_weights
            if bool(getattr(self.opt, "use_rag", False)):
                bp_tokens_bad, bp_pad_mask_bad, bp_weights_bad, _ = self._build_blueprint_batch(
                    glosses=bad_glosses,
                    wmap=self._rag_wmap,
                    pad_token_ids=self._rag_pad_token_ids,
                    device=device,
                    K=self._rag_K,
                    max_words=int(getattr(self.opt, "rag_max_words", 64)),
                    per_word_max_T=int(getattr(self.opt, "rag_per_word_max_T", 1)),
                    total_max_T=int(getattr(self.opt, "rag_total_max_T", 384)),
                    frame_subsample=int(getattr(self.opt, "rag_frame_subsample", 0)),
                    slot_names=getattr(self, "_rag_slot_names", None),
                    weight_key=str(getattr(self.opt, "rag_weight_key", "soft_w")),
                    weight_max_mix=float(getattr(self.opt, "rag_weight_max_mix", 0.5)),
                    names=names,
                    epoch=epoch,
                    mode="train",
                )

            pred_bad, _ = self.denoiser.forward(
                noisy_latent,
                timesteps,
                bad_raw,
                len_mask=len_mask,
                blueprint_tokens=bp_tokens_bad,
                blueprint_weights=bp_weights_bad,
                blueprint_pad_mask=bp_pad_mask_bad,
                use_cached_clip=True,   # V3: no-op
            )
            pred_bad = pred_bad * len_mask[..., None, None].float()
            if self.opt.prediction_type == "v_prediction" and temporal_weight is not None:
                base_bad = self._reduce_recon(pred_bad, target, len_mask, temporal_weight=temporal_weight)
            else:
                base_bad = self.recon_criterion(pred_bad, target)

            if mm_margin > 0:
                rank = F.relu(mm_margin + base - base_bad)
            else:
                rank = F.relu(base - base_bad)

            loss = loss + mm_w * rank
            loss_dict["loss_mismatch_rank"] = rank.detach()
            loss_dict["loss_mismatch_bad"] = base_bad.detach()

        return loss, attn_list, loss_dict

    @torch.no_grad()
    def generate(self, batch_data, need_attn=False):
        """
        batch_data: (text, motion, masks, m_lens, names[, frame_weights])
        V3：
        - condition 只走 gloss token-level（vocab） + RAG blueprint
        - CFG 时：uncond 分支会同时 drop gloss + rag
        """
        text, motion, masks, m_lens, names, _ = self._unpack_batch(batch_data)
        device = self.opt.device

        # ===== 1) 解析 raw_texts（List[str] 或 List[[eng,gloss]]）=====
        if isinstance(text, tuple) and len(text) >= 3 and isinstance(text[2], (list, tuple)):
            raw_texts = list(text[2])
        elif isinstance(text, (list, tuple)):
            raw_texts = list(text)
        else:
            B0 = int(motion.shape[0]) if torch.is_tensor(motion) else 1
            raw_texts = [""] * B0

        def _extract_gloss(rt):
            if isinstance(rt, (list, tuple)) and len(rt) >= 2:
                return "" if rt[1] is None else str(rt[1])
            return "" if rt is None else str(rt)

        # ===== 2) CFG：构造 input_text（2B）=====
        if bool(getattr(self.opt, "classifier_free_guidance", False)):
            raw_u = []
            for rt in raw_texts:
                if isinstance(rt, (list, tuple)) and len(rt) >= 2:
                    raw_u.append(["", ""])
                else:
                    raw_u.append("")
            input_text = raw_u + list(raw_texts)
        else:
            input_text = raw_texts

        # ===== 3) motion -> latent（保持你原逻辑）=====
        motion = motion.to(device, dtype=torch.float32)
        masks = masks.to(device)
        m_lens = m_lens.to(device, dtype=torch.long)

        B, T_pad, D_flat = motion.shape
        T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
        T_valid = max(1, min(T_valid, T_pad))
        if T_valid < T_pad:
            motion = motion[:, :T_valid]
            masks = masks[:, :T_valid]

        with torch.no_grad():
            latents, _ = self.vae_encode_raw(motion)
            if latents.dim() == 3:
                Bb, Tz, JD = latents.shape
                if JD % 3 == 0:
                    latents = latents.view(Bb, Tz, JD // 3, 3)

            Tm = motion.shape[1]
            Tz = latents.shape[1]
            downsample_ratio = max(1, Tm // Tz)
            curr_m_lens = torch.clamp(m_lens // downsample_ratio, min=0, max=Tz)
            len_mask = lengths_to_mask(curr_m_lens).to(device)

        # ===== 4) RAG blueprint：根据 input_text 构建（无论 input_text 是 pair 还是 str）=====
        bp_tokens, bp_pad_mask, bp_weights = None, None, None
        if bool(getattr(self.opt, "use_rag", False)):
            self._lazy_init_rag(device=device)
            glosses = [_extract_gloss(rt) for rt in input_text]
            bp_tokens, bp_pad_mask, bp_weights, bp_stats = self._build_blueprint_batch(
                glosses=glosses,
                wmap=self._rag_wmap,
                pad_token_ids=self._rag_pad_token_ids,
                device=device,
                K=self._rag_K,
                max_words=int(getattr(self.opt, "rag_max_words", 64)),
                per_word_max_T=int(getattr(self.opt, "rag_per_word_max_T", 1)),
                total_max_T=int(getattr(self.opt, "rag_total_max_T", 384)),
                frame_subsample=int(getattr(self.opt, "rag_frame_subsample", 0)),
                slot_names=getattr(self, "_rag_slot_names", None),
                weight_key=str(getattr(self.opt, "rag_weight_key", "soft_w")),
                weight_max_mix=float(getattr(self.opt, "rag_weight_max_mix", 0.5)),
                names=(names + names) if bool(getattr(self.opt, "classifier_free_guidance", False)) else names,
                epoch=0,
                mode="infer",
            )

            if not getattr(self, "_printed_rag_infer_once", False):
                print(f"[WLASL] infer hit_rate={bp_stats['hit_rate']:.3f} Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})")
                self._printed_rag_infer_once = True

        # ===== 5) 推理步 =====
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        timesteps = self.noise_scheduler.timesteps.to(device)

        skel_attn_weights, temp_attn_weights, cross_attn_weights = [], [], []
        for timestep in timesteps:
            if bool(getattr(self.opt, "classifier_free_guidance", False)):
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
                use_cached_clip=True,  # V3: no-op
                blueprint_tokens=bp_tokens,
                blueprint_weights=bp_weights,
                blueprint_pad_mask=bp_pad_mask,
            )

            if bool(getattr(self.opt, "classifier_free_guidance", False)):
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
        pred_motion = extract_pose_from_motion_tensor(pred_motion, self.vae.opt)

        if need_attn:
            attn_weights = (
                torch.stack(skel_attn_weights, dim=1),
                torch.stack(temp_attn_weights, dim=1),
                torch.stack(cross_attn_weights, dim=1),
            )
        else:
            attn_weights = (None, None, None)

        self._denoiser_module().remove_clip_cache()
        return pred_motion, attn_weights


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    

    def save(self, file_name, epoch, total_iter):
        module = self._denoiser_module()
        state = {
            "denoiser": module.state_dict_without_clip(),
            "optim": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        module = self._denoiser_module()

        # ===== 关键修复：在 load_state_dict 前把 RAG 子模块建出来 =====
        den_sd = checkpoint["denoiser"]
        has_rag_in_ckpt = any(k.startswith("rag_") for k in den_sd.keys())

        if has_rag_in_ckpt and bool(getattr(self.opt, "use_rag", False)):
            # 1) 先初始化 trainer 侧的 RAG 资源（会把 codebook_sizes 塞进 denoiser._rag_codebook_sizes）
            self._lazy_init_rag(self.opt.device)

            # 2) 强制让 denoiser 创建 rag_* 子模块（否则 load 会看到一堆 unexpected）
            if hasattr(module, "_maybe_init_rag"):
                module._maybe_init_rag(self.opt.device)

        missing_keys, unexpected_keys = module.load_state_dict(den_sd, strict=False)

        # V3 compatibility:
        # - allow legacy keys from v1/v2 ckpt (clip_model/word_emb/cache)
        # - allow rag_encoder.* when current rag_layers=0
        allow_unexpected_prefixes = ["clip_model.", "word_emb.", "_cache_"]
        if bool(getattr(self.opt, "use_rag", False)) and getattr(module, "rag_encoder", None) is None:
            allow_unexpected_prefixes.append("rag_encoder.")

        unexpected_bad = [
            k for k in unexpected_keys
            if not any(k.startswith(p) for p in allow_unexpected_prefixes)
        ]
        if unexpected_bad:
            raise AssertionError(f"Unexpected keys (not allowed): {unexpected_bad[:30]}")

        # For missing keys we keep resume permissive for v2->v3 migration.
        missing_bad = [
            k for k in missing_keys
            if not (
                k.startswith("clip_model.")
                or k.startswith("word_emb.")
                or "_cache_" in k
            )
        ]
        if missing_bad:
            if self.is_master:
                print(f"[Resume] missing keys (kept for compatibility): {missing_bad[:30]}")

        # optim / scheduler
        # v2->v3 may change parameter groups (removed word_emb/clip path), so keep this tolerant.
        try:
            self.optim.load_state_dict(checkpoint["optim"])
        except Exception as e:
            if self.is_master:
                print(f"[Resume] skip optimizer state due to param-group mismatch: {e}")

        try:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except Exception as e:
            if self.is_master:
                print(f"[Resume] skip lr_scheduler state: {e}")

        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)
        if self.is_distributed and not isinstance(self.denoiser, DDP):
            device_idx = int(getattr(self.opt, "device_index", getattr(self.opt, "local_rank", 0)))
            self.denoiser = DDP(
                self.denoiser,
                device_ids=[device_idx],
                output_device=device_idx,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        # 优化器
        self.optim = torch.optim.AdamW(self.denoiser.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            if os.path.isfile(model_dir):
                epoch, it = self.resume(model_dir)
                if self.is_master:
                    print("Load model epoch:%d iterations:%d"%(epoch, it))
            else:
                if self.is_master:
                    print(f"[Resume][WARN] --is_continue is set but checkpoint not found: {model_dir}. Start from scratch.")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_master:
            print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
            print(f"Iters Per Epoch, Training: {len(train_loader)}, Validation: {(len(eval_val_loader) if eval_val_loader is not None else 0)}")
        logs = defaultdict(def_value, OrderedDict())
        self._printed_rag_eval_once = False
        # # 初始评估
        # metrics = evaluation_denoiser(
        #     self.opt.model_dir, 
        #     eval_val_loader, 
        #     self.denoiser, 
        #     self.generate, 
        #     self.logger, 
        #     epoch,
        #     physical_evaluator=self.physical_evaluator,
        #     smplx_model=self.smplx_model,
        #     opt=self.opt
        # )
        # print(f"MPJPE BODY {metrics.get('how2sign_MPJPE_body', 1000.0)}") 
        
        best_mpjpe = 1000.0

        # === 训练循环 ===
        while epoch < self.opt.max_epoch:
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if batch_sampler is None and hasattr(train_loader, "loader"):
                batch_sampler = getattr(train_loader.loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)

            logs = defaultdict(def_value, OrderedDict())
            torch.cuda.empty_cache()
            self.denoiser.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                self.optim.zero_grad()
                with self._autocast_ctx():
                    loss, attn_list, loss_dict = self.train_forward(batch_data, epoch)

                # 【修复】增加 NaN 检测
                if torch.isnan(loss):
                    if self.is_master:
                        print(f"❌ Critical Warning: Loss is NaN at Epoch {epoch} Step {it}. Skipping backward to prevent crash.")
                        print(f"Loss Dict: {loss_dict}")
                    # 可选：如果希望 NaN 就退出，可以 sys.exit(1)，这里选择跳过该 batch
                    continue
                
                if self._grad_scaler is not None and self._amp_use:
                    self._grad_scaler.scale(loss).backward()
                    self._grad_scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                    self._grad_scaler.step(self.optim)
                    self._grad_scaler.update()
                else:
                    loss.backward()
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

                if self.is_master and it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
            # ===== epoch end: print/log once =====
            mean_loss = OrderedDict()
            denom = max(1, len(train_loader))
            for tag, value in logs.items():
                avg_v = value / denom
                mean_loss[tag] = avg_v
                if self.logger is not None:
                    self.logger.add_scalar(f'Train/{tag}', avg_v, it)

            train_log_dict = {f"train/{k}": v for k, v in mean_loss.items()}
            train_log_dict["lr"] = self.optim.param_groups[0]["lr"]
            if self.is_master:
                wandb.log(train_log_dict, step=it)

            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in mean_loss.items()])
            msg = f"[Ep {epoch:03d} | It {it:06d}] {loss_str} | lr: {train_log_dict['lr']:.6f}"

            self.log_to_file(msg)
            if self.is_master:
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=(len(train_loader)-1))

            # 【修复】将 LR 更新移到 Epoch 循环末尾
            if it >= self.opt.warm_up_iter:
                self.lr_scheduler.step()
            
            if self.is_master:
                self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1
            self._dist_barrier()
            
            # evaluation
            if self.is_master and eval_val_loader is not None and epoch % self.opt.eval_every_e == 0:
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
                if not isinstance(metrics, dict) or len(metrics) == 0:
                    self.log_to_file("[Eval][WARN] Empty metrics returned. Skip best-checkpoint update.")
                    continue
                if isinstance(metrics, dict):
                    wandb_metrics = {f"eval/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                                     for k, v in metrics.items()}
                    wandb.log(wandb_metrics, step=it)

                current_mpjpe = metrics.get('how2sign_MPJPE_body', 1000.0) 
                if isinstance(current_mpjpe, torch.Tensor): current_mpjpe = current_mpjpe.item()
                if not np.isfinite(float(current_mpjpe)):
                    self.log_to_file(f"[Eval][WARN] Non-finite MPJPE ({current_mpjpe}). Skip best-checkpoint update.")
                    continue
                
                if current_mpjpe < best_mpjpe:
                    best_mpjpe = current_mpjpe
                    self.save(pjoin(self.opt.model_dir, 'net_best_mpjpe.tar'), epoch, it)
                    self.log_to_file(f"--> --> MPJPE Improved to {best_mpjpe:.5f}!!!")
            self._dist_barrier()
