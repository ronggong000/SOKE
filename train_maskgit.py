#train_maskgit.py
import os
import json
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import importlib.util
from pathlib import Path
from mymodel.maskgit.dataset_maskgit import SignMotionTokenDataset, pad_collate, health_scan, load_metadata
from mymodel.maskgit.maskgit_model import MaskGITTransformer
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from mGPT.utils.joints_list import SELECTED_JOINT_INDICES,SELECTED_JOINT_INDICES_NEIGHBOR_LIST
from types import SimpleNamespace
import yaml
from functools import partial
# =========================
# CONFIG（你只要改这里）
# =========================
CONFIG = {
 
    # 【新增】QVAE 的路径，用于可视化解码
    "qvae_config_path": "checkpoints/vae/qvae_b256h1024_L1_fingerdistinct/opt.txt", # 你的 QVAE 配置文件
    "qvae_model_path": "checkpoints/vae/qvae_b256h1024_L1_fingerdistinct/model/latest.tar",   # 你的 QVAE 权重
    # dataset_root 是你 extract_code_dataset.py 输出 json 的目录（包含 train_dataset.json / val_dataset.json / dataset_metadata.json）
    "dataset_root": "checkpoints/vae/qvae_b256h1024_L1_fingerdistinct",

    # (Optional) 多路 embedding：如果你更想用列表配置，可以直接提供 bases。
    # 例："text_emb_bases": ["checkpoints/text_embeddings", "checkpoints/gloss_embeddings"]
    # 如果该字段非空，则优先使用它（忽略 text_emb_base/gloss_emb_base）。
    "text_emb_bases": ["data/text_embedding", "data/gloss_embedding"],
    "text_emb_base": "",
    "gloss_emb_base": "",

    "text_source": "text",
    # 保存目录
    "save_dir": "checkpoints/maskgit_v1_double_emb_24depth",

    # training
    "epochs": 300,
    "batch_size": 16,
    "num_workers": 8,
    "lr": 2e-4,
    "lr_warmup_epochs": 10,   # 每次进入新 stage，都 warmup 这几个 epoch
    "lr_min": 2e-5,           # stage 内 cosine decay 的最低 lr（不想 decay 就设成等于 lr）
    "weight_decay": 0.01,
    "grad_clip": 1.0,

    # model
    "dim": 512,
    "depth": 24,
    "heads": 8,
    "dropout": 0.1,
    "max_seq_len":4096,
    "max_text_len": 256,   # 只做截断，不做 tokenizer（你存的是 encoder 输出）
    "text_dim": 1024,
    #dataset
    "max_len": 256,        # motion token length T'（建议 >= 训练集 95% 分位）
    # mask schedule
    # "mask_ratio_min": 0.6,     # 每 batch 采样 [r_min, 1.0]
    # "p_textonly_train": 0.4,   # 训练时一定比例强制 text-only（全 mask）
    "mask_ratio_min_schedule": {"type":"cosine","start":0.15,"end":0.55,"warmup_epochs":30},
    "p_textonly_train_schedule": {"type":"linear","start":0.00,"end":0.15,"warmup_epochs":30},

    # RAG-sim（结构化检索画布）
    # 训练时额外抽一部分 batch：只保留少量连续锚点 span，其余全部 mask。
    # 这会逼模型学会："给少量碎片 -> 补全 + 缝合"，更贴近你未来真正的 RAG 推理。
    # rag-sim 先别急着上强度，等模型能稳定复原再加
    "p_rag_sim_train_schedule": {"type":"linear","start":0.0,"end":0.10,"warmup_epochs":50},
    "rag_sim_keep_ratio_min": 0.10,
    "rag_sim_keep_ratio_max": 0.30,
    "rag_sim_span_min": 8,
    "rag_sim_span_max": 32,
    # 每个被 mask 的时刻，尽量 mask 到足够多的 slot，不然 body/arm 很容易吃不到梯度
    "mask_slot_frac": 1.0,   # 先 1.0；后面稳定了你再降到 0.7
    "mask_mode": "span",       # 或 "mix"
    "mask_span_max": 64,       # 训练早期可以更大
    # InfoNCE（强约束）
    "infonce_weight": 0.5,#0.5
    "infonce_temp": 0.1,

    # eval
    "eval_every": 5,
    "save_every": 10,
    "healthscan_n": 2000,

    "early_stop": False,
    "early_stop_patience": 5,      # 连续 5 次 eval 没变好就停（eval_every=5 => 最多容忍 25 个 epoch）
    "early_stop_min_delta": 0.0,   # 需要更严格就设 1e-3 / 5e-4
    "early_stop_min_epochs": 10,   # 训练前 10 个 epoch 不允许早停（建议有）
    # reproducibility
    "seed": 1234,

    # =========================
    # Debug / curriculum knobs
    # =========================
    # If >0, use a tiny subset to sanity-check overfitting.
    "debug_train_n": 0,
    "debug_val_n": 0,

    # Stage training (loss only on selected slots)
    #   stage=1: body_1
    #   stage=2: body_1 + lh_1 + rh_1
    #   stage=3: all slots
    # resume
    "resume_path": "/home/smuk0019/ar85_scratch2/singyu/SOKE/checkpoints/maskgit_v1_double_emb_24depth/maskgit_best.tar",        # path to a checkpoint .tar (optional)
    "resume_last": False,     # if True and resume_path is empty, try <save_dir>/maskgit_last.tar
    "resume_optim": False,     # restore optimizer/scaler when resuming

    # length predictor (text -> length bins). Set length_loss_weight=0 to disable training this head.
    "length_bin_size": 1,     # length bin size in *motion-token timesteps* (NOT original frames)
    "length_num_bins": 0,     # 0 => auto = ceil(max_len / bin_size)
    "length_loss_weight": 0.0,

    # time shift augmentation (token-level). 0 disables.
    "time_shift_aug_max": 0,#1  # e.g., 1/2/4. With VQ compress=4, shift=1 token ~= 4 frames.

    # shift-tolerant CE (token-level). 0 disables. Recommend 1-2.
    "shift_tolerant_max": 0,#1

    # stage_mode:
    #   - "manual": always use CONFIG["stage"]
    #   - "auto": use CONFIG["stage_schedule"] (epoch -> stage)
    "stage_mode": "auto",
    "stage": 2,
    "stage_schedule": [(0, 1), (10, 2)],   # stage1 至少 20 epoch

    # 2-step iterative training (MaskGIT-style refinement)
    "iterative_train": False,
    "iter_steps": 2,
    "iter_keep_frac": 0.5,          # keep top-confidence fraction of currently-masked positions after step1
    "iter_loss_w1": 0.0,            # weight for step1 MLM loss
    "iter_loss_w2": 1.0,            # weight for step2 MLM loss

    "slot_weights": {
    "body": 4.0,
    "l_arm": 2.0,
    "r_arm": 2.0,

    "l_thumb": 0.6, "l_index": 0.6, "l_middle": 0.6, "l_ring": 0.6, "l_pinky": 0.6,
    "r_thumb": 0.6, "r_index": 0.6, "r_middle": 0.6, "r_ring": 0.6, "r_pinky": 0.6,
    },

    "cfg_drop_prob": 0.10,   # CFG: 训练时丢条件概率
}
def _scheduled_value(epoch: int, schedule, default_value: float) -> float:
    """
    schedule:
      - None: use default_value
      - list/tuple of (start_epoch, value): piecewise constant
      - dict: {"type":"linear","start":0.4,"end":0.8,"warmup_epochs":10}
    """
    if schedule is None:
        return float(default_value)

    # dict-based schedule
    if isinstance(schedule, dict):
        stype = str(schedule.get("type", "linear")).lower()
        if stype == "linear":
            start = float(schedule.get("start", default_value))
            end = float(schedule.get("end", default_value))
            warm = int(schedule.get("warmup_epochs", 1))
            warm = max(1, warm)
            t = min(max(epoch, 0), warm) / float(warm)
            return float(start + (end - start) * t)
        
        if stype == "cosine":
            start = float(schedule.get("start", default_value))
            end = float(schedule.get("end", default_value))
            warm = int(schedule.get("warmup_epochs", 1))
            warm = max(1, warm)
            t = min(max(epoch, 0), warm) / float(warm)  # 0..1
            t = 0.5 - 0.5 * math.cos(math.pi * t)       # cosine ease-in/out
            return float(start + (end - start) * t)

        if stype == "arccos":
            start = float(schedule.get("start", default_value))
            end = float(schedule.get("end", default_value))
            warm = int(schedule.get("warmup_epochs", 1))
            warm = max(1, warm)
            t = min(max(epoch, 0), warm) / float(warm)  # 0..1
            t = math.acos(1.0 - 2.0 * t) / math.pi      # arccos map to 0..1
            return float(start + (end - start) * t)

        else:
            # fallback: treat as constant
            raise ValueError(f"unknown schedule type: {stype}")

    # list-based schedule
    if isinstance(schedule, (list, tuple)) and len(schedule) > 0:
        # expect list of (start_epoch, value)
        val = float(schedule[0][1])
        for (start_ep, v) in schedule:
            if epoch >= int(start_ep):
                val = float(v)
        return float(val)

    return float(default_value)
def _get_stage_epoch_range(epoch: int, total_epochs: int):
    """
    Returns (stage_start_epoch, stage_end_epoch_exclusive) based on CONFIG["stage_schedule"].
    Works even if stage_mode != "auto" (falls back to whole training as one stage).
    """
    if str(CONFIG.get("stage_mode", "manual")).lower() != "auto":
        return 0, int(total_epochs)

    sched = list(CONFIG.get("stage_schedule", []))
    if len(sched) == 0:
        return 0, int(total_epochs)

    # ensure sorted by start epoch
    sched = sorted([(int(a), int(b)) for (a, b) in sched], key=lambda x: x[0])

    start = 0
    end = int(total_epochs)
    for i, (s_ep, _st) in enumerate(sched):
        if epoch >= s_ep:
            start = s_ep
            end = int(total_epochs) if (i + 1 >= len(sched)) else int(sched[i + 1][0])
    return int(start), int(end)


def _compute_stagewise_lr(epoch: int):
    """
    Stage-wise LR schedule with restart:
      - warmup for CONFIG["lr_warmup_epochs"] epochs starting at stage_start
      - then cosine decay to CONFIG["lr_min"] within this stage
    """
    base_lr = float(CONFIG["lr"])
    min_lr = float(CONFIG.get("lr_min", base_lr))
    warm = int(CONFIG.get("lr_warmup_epochs", 0))
    warm = max(0, warm)

    total_epochs = int(CONFIG["epochs"])
    stage_start, stage_end = _get_stage_epoch_range(epoch, total_epochs)
    stage_len = max(1, stage_end - stage_start)
    e = int(epoch - stage_start)  # 0..stage_len-1

    # warmup
    if warm > 0 and e < warm:
        return base_lr * float(e + 1) / float(warm)

    # cosine decay within stage
    if stage_len <= warm:
        return base_lr

    decay_len = max(1, stage_len - warm)
    t = float(min(max(e - warm, 0), decay_len)) / float(decay_len)  # 0..1
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    return float(lr)


def _set_optimizer_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)

def get_mask_hparams_for_epoch(epoch: int):
    """Return current-epoch augmentation hyper-params.

    Returns:
      (mask_ratio_min, p_textonly, p_rag_sim)

    Where:
      - mask_ratio_min: lower bound of random masking ratio (for normal MaskGIT MLM)
      - p_textonly:     probability of full-mask (text-only) batches
      - p_rag_sim:      probability of "RAG-sim" batches (sparse anchor spans kept, rest masked)

    Notes:
      - p_textonly and p_rag_sim are applied sequentially using one random draw:
          u < p_textonly            => text-only
          u < p_textonly+p_rag_sim  => rag-sim
          else                      => normal mask

      - If schedules are provided, they take precedence over auto warmup.
    """
    # final targets (optional)
    r_final = float(CONFIG.get("mask_ratio_min", 0.8))
    p_text_final = float(CONFIG.get("p_textonly_train", 0.0))
    p_rag_final = float(CONFIG.get("p_rag_sim_train", 0.0))

    # schedules (optional)
    r_sched = CONFIG.get("mask_ratio_min_schedule", None)
    p_text_sched = CONFIG.get("p_textonly_train_schedule", None)
    p_rag_sched = CONFIG.get("p_rag_sim_train_schedule", None)

    if (r_sched is not None) or (p_text_sched is not None) or (p_rag_sched is not None):
        r = _scheduled_value(epoch, r_sched, r_final)
        p_text = _scheduled_value(epoch, p_text_sched, p_text_final)
        p_rag = _scheduled_value(epoch, p_rag_sched, p_rag_final)

        r = float(min(max(r, 0.0), 1.0))
        p_text = float(min(max(p_text, 0.0), 1.0))
        p_rag = float(min(max(p_rag, 0.0), 1.0))
        return r, p_text, p_rag

    # --- auto warmup fallback (for r and p_text only) ---
    total_epochs = int(CONFIG.get("epochs", 100))
    warm = int(CONFIG.get("mask_schedule_warmup_epochs", min(10, max(1, total_epochs // 5))))
    warm = max(1, warm)

    r_start = float(CONFIG.get("mask_ratio_min_start", max(0.35, r_final - 0.25)))
    p_text_start = float(CONFIG.get("p_textonly_train_start", max(0.0, p_text_final - 0.35)))

    t = min(max(epoch, 0), warm) / float(warm)
    r = r_start + (r_final - r_start) * t
    p_text = p_text_start + (p_text_final - p_text_start) * t

    r = float(min(max(r, 0.0), 1.0))
    p_text = float(min(max(p_text, 0.0), 1.0))

    # by default rag-sim stays off unless user sets schedule/value
    p_rag = float(min(max(p_rag_final, 0.0), 1.0))
    return r, p_text, p_rag

def load_config_from_path(config_path: str) -> dict:
    """
    Load config dict from:
      - .py file containing a dict named CONFIG (recommended) or a function get_config()
      - .json file containing a dict
    Returns a dict.
    """
    p = Path(config_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"[CONFIG] file not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".py":
        spec = importlib.util.spec_from_file_location("user_config", str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"[CONFIG] cannot import python config: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # executes user's config script

        if hasattr(mod, "CONFIG"):
            cfg = getattr(mod, "CONFIG")
        elif hasattr(mod, "get_config"):
            cfg = mod.get_config()
        else:
            raise ValueError(f"[CONFIG] {p} must define `CONFIG = {{...}}` or `def get_config(): ...`")

        if not isinstance(cfg, dict):
            raise TypeError(f"[CONFIG] python config must be a dict, got: {type(cfg)}")

        return cfg

    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise TypeError(f"[CONFIG] json config must be a dict, got: {type(cfg)}")
        return cfg

    raise ValueError(f"[CONFIG] unsupported config suffix: {suffix} (use .py or .json)")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train MaskGIT (config via file path)")
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="Path to config file (.py or .json). "
#              "For .py, define `CONFIG = {...}` (recommended).",
#     )
#     return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_key_padding_mask(text_pad_mask: torch.Tensor, motion_pad_mask: torch.Tensor) -> torch.Tensor:
    # True = pad
    return torch.cat([text_pad_mask, motion_pad_mask], dim=1)
def sample_mask_map(
    lengths: torch.Tensor,
    T: int,
    device,
    r_min: float,
    r_max: float = 1.0,
    K: int = None,
    allowed_slots=None,   # list[int] or tensor[int], if provided: only mask these slots
) -> torch.Tensor:
    """
    returns: mask_map_slot [B,T,K] bool
      True means THIS slot at THIS timestep is masked (only within valid range)

    CONFIG:
      - mask_mode: "span" | "random" | "mix" (default="span")
      - mask_span_min/max/p_geometric
      - mask_slot_frac: fraction of (allowed) slots to mask at a masked timestep (default=0.5)
    """
    B = int(lengths.shape[0])

    # --- determine K robustly ---
    if K is None:
        # try global slot_names
        K_guess = None
        try:
            if "slot_names" in globals() and globals()["slot_names"] is not None:
                K_guess = int(len(globals()["slot_names"]))
        except Exception:
            K_guess = None
        if K_guess is None:
            K_guess = int(CONFIG.get("K_override_for_mask", 13))
        K = int(K_guess)
    else:
        K = int(K)

    # --- allowed slot indices ---
    if allowed_slots is None:
        allowed = torch.arange(K, device=device, dtype=torch.long)
    else:
        if isinstance(allowed_slots, (list, tuple)):
            allowed = torch.tensor(list(allowed_slots), device=device, dtype=torch.long)
        elif isinstance(allowed_slots, torch.Tensor):
            allowed = allowed_slots.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"allowed_slots must be list/tuple or torch.Tensor, got {type(allowed_slots)}")
        if allowed.numel() == 0:
            allowed = torch.arange(K, device=device, dtype=torch.long)

    # --- ratios ---
    mask_slot_frac = float(CONFIG.get("mask_slot_frac", 0.5))
    mask_slot_frac = float(min(max(mask_slot_frac, 0.05), 1.0))

    mode = str(CONFIG.get("mask_mode", "span")).lower()
    span_min = int(CONFIG.get("mask_span_min", 4))
    span_max = int(CONFIG.get("mask_span_max", 32))
    p_geo = float(CONFIG.get("mask_span_p_geometric", 0.2))
    mix_rand_frac = float(CONFIG.get("mask_mix_rand_frac", 0.1))
    force_cover_all = bool(CONFIG.get("mask_force_cover_all", True))

    span_min = max(1, span_min)
    span_max = max(span_min, span_max)
    p_geo = float(min(max(p_geo, 1e-4), 0.999))
    mix_rand_frac = float(min(max(mix_rand_frac, 0.0), 1.0))

    r = float(np.random.uniform(r_min, r_max))

    # ---- sample time-mask [B,T] ----
    time_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    for i in range(B):
        valid = int(lengths[i].item())
        if valid <= 0:
            continue

        m = max(1, int(valid * r))
        if m >= valid:
            time_mask[i, :valid] = True
            continue

        if mode == "random":
            perm = torch.randperm(valid, device=device)
            time_mask[i, perm[:m]] = True
            continue

        if mode == "mix":
            m_rand = max(1, int(m * mix_rand_frac))
            m_span = max(1, m - m_rand)
        else:
            m_rand = 0
            m_span = m

        covered = 0
        max_tries = 10 * (m_span // max(1, span_min) + 1)
        tries = 0

        while covered < m_span and tries < max_tries:
            tries += 1
            geom = torch.distributions.Geometric(
                probs=torch.tensor(p_geo, device=device)
            ).sample().long().item()
            span_len = span_min + int(geom)
            span_len = min(span_len, span_max)

            if valid - span_len <= 0:
                start = 0
                span_len = valid
            else:
                start = int(torch.randint(0, valid - span_len + 1, (1,), device=device).item())
            end = start + span_len

            before = int(time_mask[i, start:end].sum().item())
            time_mask[i, start:end] = True
            after = int(time_mask[i, start:end].sum().item())
            covered += (after - before)

        if force_cover_all and covered < m_span:
            need = m_span - covered
            avail = (~time_mask[i, :valid]).nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                take = min(int(need), int(avail.numel()))
                pick = avail[torch.randperm(avail.numel(), device=device)[:take]]
                time_mask[i, pick] = True

        if m_rand > 0:
            avail = (~time_mask[i, :valid]).nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                take = min(int(m_rand), int(avail.numel()))
                pick = avail[torch.randperm(avail.numel(), device=device)[:take]]
                time_mask[i, pick] = True

        # exact m
        cur = int(time_mask[i, :valid].sum().item())
        if cur > m:
            extra = cur - m
            masked_pos = time_mask[i, :valid].nonzero(as_tuple=False).squeeze(-1)
            drop = masked_pos[torch.randperm(masked_pos.numel(), device=device)[:extra]]
            time_mask[i, drop] = False
        elif cur < m:
            need = m - cur
            avail = (~time_mask[i, :valid]).nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                take = min(int(need), int(avail.numel()))
                pick = avail[torch.randperm(avail.numel(), device=device)[:take]]
                time_mask[i, pick] = True

    # ---- expand to slot-wise mask [B,T,K] ----
    mask_map_slot = torch.zeros((B, T, K), dtype=torch.bool, device=device)

    # number of masked slots is relative to allowed set
    A = int(allowed.numel())
    num_mask_slots = max(1, int(round(A * mask_slot_frac)))

    for i in range(B):
        valid = int(lengths[i].item())
        if valid <= 0:
            continue
        t_idx = time_mask[i, :valid].nonzero(as_tuple=False).squeeze(-1)
        if t_idx.numel() == 0:
            continue

        for t in t_idx.tolist():
            perm = allowed[torch.randperm(A, device=device)]
            sel = perm[:num_mask_slots]
            mask_map_slot[i, t, sel] = True

    return mask_map_slot


def sample_rag_sim_mask_map(
    lengths: torch.Tensor,
    T: int,
    K: int,
    device,
    keep_ratio_min: float,
    keep_ratio_max: float,
    span_min: int,
    span_max: int,
) -> torch.Tensor:
    """Simulate a structured RAG canvas.

    We keep a few contiguous *anchor spans* (unmasked), and mask everything else.
    This matches your real RAG usage where you paste retrieved motion fragments
    and let MaskGIT inpaint the rest.

    Returns:
      mask_map: [B,T,K] bool, True means masked.
    """
    B = int(lengths.shape[0])
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    keep_ratio_min = float(min(max(keep_ratio_min, 0.0), 1.0))
    keep_ratio_max = float(min(max(keep_ratio_max, 0.0), 1.0))
    if keep_ratio_max < keep_ratio_min:
        keep_ratio_min, keep_ratio_max = keep_ratio_max, keep_ratio_min

    mask_map = torch.zeros((B, T, K), dtype=torch.bool, device=device)

    for b in range(B):
        valid = int(lengths[b].item())
        if valid <= 0:
            continue

        # start: everything masked
        # (pad part stays unmasked; motion_pad_mask will ignore it anyway)
        keep_ratio = float(np.random.uniform(keep_ratio_min, keep_ratio_max))
        keep_target = max(1, int(round(valid * keep_ratio)))

        keep_time = torch.zeros((valid,), dtype=torch.bool, device=device)

        # sample spans until we cover enough kept time steps
        tries = 0
        max_tries = 20 + 5 * (keep_target // max(1, span_min) + 1)
        while int(keep_time.sum().item()) < keep_target and tries < max_tries:
            tries += 1
            span_len = int(np.random.randint(span_min, span_max + 1))
            span_len = min(span_len, valid)
            if valid - span_len <= 0:
                start = 0
            else:
                start = int(torch.randint(0, valid - span_len + 1, (1,), device=device).item())
            end = start + span_len
            keep_time[start:end] = True

        # if still not enough, fill remaining with random time indices
        cur = int(keep_time.sum().item())
        if cur < keep_target:
            need = keep_target - cur
            avail = (~keep_time).nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                take = min(int(need), int(avail.numel()))
                pick = avail[torch.randperm(avail.numel(), device=device)[:take]]
                keep_time[pick] = True

        # build mask: masked where NOT kept
        mask_time = ~keep_time  # [valid]
        mask_map[b, :valid, :] = mask_time.unsqueeze(-1).expand(valid, K)

    return mask_map


def get_stage(epoch: int) -> int:
    """
    2-stage curriculum:
      stage=1: upper_body + arms
      stage=2: all slots
    """
    mode = str(CONFIG.get("stage_mode", "manual")).lower()
    if mode == "manual":
        return int(CONFIG.get("stage", 2))

    sched = CONFIG.get("stage_schedule", [(0, 2)])
    # expect list of (start_epoch, stage)
    stage = int(sched[0][1])
    for (start_ep, st) in sched:
        if epoch >= int(start_ep):
            stage = int(st)
    return stage


def stage_active_slots(stage: int, slot_names: list) -> list:
    """
    stage=1: upper_body + arms
    stage>=2: all slots

    slot_names examples (your qvae):
      upper_body, l_arm, r_arm, l_index, ..., r_thumb
    """
    if stage <= 1:
        active = []
        for n in slot_names:
            nl = n.lower()
            if ("body" in nl) or ("torso" in nl) or ("arm" in nl):
                active.append(n)

        # safety fallback
        if len(active) == 0:
            return list(slot_names)
        return active

    return list(slot_names)


def slots_to_indices(all_slots: list, active_slots: list) -> list:
    idx = []
    for s in active_slots:
        if s not in all_slots:
            raise ValueError(f"active slot '{s}' not in slots={all_slots}")
        idx.append(all_slots.index(s))
    return idx


def maybe_make_debug_subset(ds, n: int, seed: int = 0):
    """Return a deterministic Subset(ds, idxs) if n>0."""
    if n is None:
        return ds
    n = int(n)
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator()
    g.manual_seed(int(seed))
    idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
    return torch.utils.data.Subset(ds, idxs)


@torch.no_grad()
def _pick_top_confidence_positions(logits: dict, active_slot_names: list, mask_map: torch.Tensor, keep_frac: float):
    """Pick a subset of currently-masked *time positions* to keep (fill with step1 prediction).

    Supports:
      - mask_map: [B,T]   (time-wise)
      - mask_map: [B,T,K] (slot-wise)  -> will be reduced to [B,T] by any(slot)

    Returns:
      keep_map: [B,T] bool, True means "unmask this time step with step1 argmax".
    """
    if mask_map.dim() == 3:
        mask_time = mask_map.any(dim=-1)  # [B,T]
    elif mask_map.dim() == 2:
        mask_time = mask_map
    else:
        raise ValueError(f"mask_map must be [B,T] or [B,T,K], got dim={mask_map.dim()}")

    B, T = mask_time.shape
    device = mask_time.device
    keep_map = torch.zeros((B, T), dtype=torch.bool, device=device)

    if keep_frac <= 0:
        return keep_map

    conf_list = []
    for name in active_slot_names:
        prob = torch.softmax(logits[name].float(), dim=-1)  # [B,T,V]
        conf, _ = prob.max(dim=-1)                          # [B,T]
        conf_list.append(conf)

    conf_pos = torch.stack(conf_list, dim=0).mean(dim=0)    # [B,T]

    for b in range(B):
        masked_idx = torch.nonzero(mask_time[b], as_tuple=False).squeeze(-1)
        m = int(masked_idx.numel())
        if m <= 0:
            continue
        k = max(1, int(round(m * float(keep_frac))))
        conf_b = conf_pos[b, masked_idx]
        topk = torch.topk(conf_b, k=min(k, m), largest=True).indices
        keep_idx = masked_idx[topk]
        keep_map[b, keep_idx] = True

    return keep_map
def slice_mask_map_for_active(mask_map, active_idx):
    """
    Slice a [B,T,K] mask_map down to active slots only.

    Supports:
      - mask_map: None, [B,T], or [B,T,K]
      - active_idx: list/tuple[int] or 1D torch.Tensor[int] or None

    Returns:
      - If mask_map is [B,T] -> unchanged
      - If mask_map is [B,T,K] -> [B,T,K_active]
    """
    import torch

    if mask_map is None:
        return None

    # If we have no active_idx, do nothing
    if active_idx is None:
        return mask_map

    # Normalize active_idx to a python list of ints
    if isinstance(active_idx, torch.Tensor):
        active_list = active_idx.detach().cpu().tolist()
    elif isinstance(active_idx, (list, tuple)):
        active_list = list(active_idx)
    else:
        raise TypeError(f"active_idx must be list/tuple/Tensor/None, got {type(active_idx)}")

    # If empty, return an empty K dimension for 3D map, or unchanged for 2D map
    if len(active_list) == 0:
        if mask_map.dim() == 3:
            B, T, _K = mask_map.shape
            return mask_map[:, :, :0]
        return mask_map

    # [B,T] time-mask: nothing to slice
    if mask_map.dim() == 2:
        return mask_map

    # [B,T,K] slot-mask: slice along K
    if mask_map.dim() == 3:
        # Make sure indices are on the same device for index_select
        idx_t = torch.tensor(active_list, dtype=torch.long, device=mask_map.device)
        return torch.index_select(mask_map, dim=2, index=idx_t)

    raise ValueError(f"mask_map must be [B,T] or [B,T,K], got shape={tuple(mask_map.shape)}")

def compute_losses_and_acc(
    logits: dict,                 # name -> [B,T,V]
    targets: torch.Tensor,        # [B,T,K]
    mask_map: torch.Tensor,       # [B,T,K] or [B,T]  True=masked
    pad_token_ids: torch.Tensor,  # [K]
    slot_names: list,
    slot_weights: dict,
):
    """
    CE only on masked positions, ignoring PAD.

    Supports:
      - mask_map: [B,T,K] (slot-wise masking)  ✅ recommended
      - mask_map: [B,T]   (time-wise masking)  (will be expanded to [B,T,K])

    Normalization:
      - Loss is weighted by slot_weights and then normalized by total effective weight
        (only counting slots that had at least 1 valid token), so loss scale is comparable
        across different K / weight settings.
    """
    B, T, K = targets.shape
    device = targets.device

    # --- make mask_map [B,T,K] ---
    if mask_map.dim() == 2:
        # [B,T] -> [B,T,K]
        mask_map_k = mask_map.unsqueeze(-1).expand(B, T, K)
    elif mask_map.dim() == 3:
        mask_map_k = mask_map
        if mask_map_k.shape != (B, T, K):
            raise ValueError(f"mask_map shape mismatch: got {tuple(mask_map_k.shape)}, expect {(B,T,K)}")
    else:
        raise ValueError(f"mask_map must be [B,T] or [B,T,K], got dim={mask_map.dim()}")

    total_loss = targets.new_tensor(0.0)
    weight_sum = 0.0

    per_slot_loss = {}
    per_slot_acc = {}
    acc_list = []

    for k, name in enumerate(slot_names):
        logit = logits[name]   # [B,T,V]
        y = targets[..., k]    # [B,T]

        pad_id = int(pad_token_ids[k].item())
        valid = mask_map_k[..., k] & (y != pad_id)  # ✅ slot-wise valid mask

        n_valid = int(valid.sum().item())
        if n_valid == 0:
            per_slot_loss[name] = 0.0
            per_slot_acc[name] = 0.0
            acc_list.append(0.0)
            continue

        logit_v = logit[valid]  # [N,V]
        y_v = y[valid]          # [N]

        loss_k = F.cross_entropy(logit_v, y_v, reduction="mean")
        w = float(slot_weights.get(name, 1.0))

        total_loss = total_loss + loss_k * w
        weight_sum += w

        pred = torch.argmax(logit_v, dim=-1)
        acc = (pred == y_v).float().mean().item()

        per_slot_loss[name] = float(loss_k.item())
        per_slot_acc[name] = float(acc)
        acc_list.append(acc)

    # Normalize: keep loss scale stable (only over slots that contributed)
    if weight_sum > 0:
        total_loss = total_loss / weight_sum

    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    return total_loss, per_slot_loss, per_slot_acc, mean_acc

def apply_time_shift_aug(motion, lengths, motion_pad_mask, pad_token_ids, ts_max: int):
    """
    Non-circular time shift augmentation on token sequence.

    motion: [B, T, K]  (Long)
    lengths: [B]
    motion_pad_mask: [B, T] True=PAD
    pad_token_ids: [K] (Long)  PAD token id per slot
    """
    if ts_max is None or int(ts_max) <= 0:
        return motion, lengths, motion_pad_mask

    B, T, K = motion.shape
    device = motion.device
    ts_max = int(ts_max)

    out = motion.clone()
    out_pad_mask = motion_pad_mask.clone()

    # [K] on correct device/dtype
    pad_ids = pad_token_ids.to(device=device, dtype=out.dtype)  # [K]

    for b in range(B):
        L = int(lengths[b].item())
        if L <= 1:
            continue

        delta = int(torch.randint(-ts_max, ts_max + 1, (1,), device=device).item())
        if delta == 0:
            continue

        # copy valid region
        tmp = out[b, :L].clone()  # [L, K]

        # reset valid region to PAD: broadcasting [K] -> [L,K]
        out[b, :L] = pad_ids.unsqueeze(0).expand(L, K)

        if delta > 0:
            # shift RIGHT by delta
            if delta < L:
                out[b, delta:L] = tmp[0 : L - delta]
        else:
            # shift LEFT by s
            s = -delta
            if s < L:
                out[b, 0 : L - s] = tmp[s:L]

        # enforce pad mask from lengths
        out_pad_mask[b, :L] = False
        out_pad_mask[b, L:] = True

    return out, lengths, out_pad_mask
# def _shift_targets_and_mask(
#     targets: torch.Tensor,        # [B,T,K]
#     mask_map: torch.Tensor,       # [B,T]
#     delta: int,
#     pad_token_ids: torch.Tensor,  # [K]
# ):
#     """
#     Align logits[t] with targets[t+delta] (delta>0 looks ahead).
#     Returns shifted_targets, shifted_mask.
#     """
#     B, T, K = targets.shape
#     shifted = targets.clone()
#     shifted_mask = mask_map.clone()

#     if delta > 0:
#         # shifted[:, :T-d] = targets[:, d:]
#         shifted[:, : T - delta] = targets[:, delta:]
#         # pad last delta
#         pad_row = pad_token_ids.view(1, 1, K).to(targets.device)
#         shifted[:, T - delta :] = pad_row
#         shifted_mask[:, T - delta :] = False
#     elif delta < 0:
#         s = -delta
#         shifted[:, s:] = targets[:, : T - s]
#         pad_row = pad_token_ids.view(1, 1, K).to(targets.device)
#         shifted[:, :s] = pad_row
#         shifted_mask[:, :s] = False

#     return shifted, shifted_mask


# def compute_losses_and_acc_shift_tolerant(
#     logits: dict,                 # name -> [B,T,V]
#     targets: torch.Tensor,         # [B,T,K]
#     mask_map: torch.Tensor,        # [B,T] True=masked
#     pad_token_ids: torch.Tensor,   # [K]
#     slot_names: list,
#     slot_weights: dict,
#     max_shift: int,
# ):
#     """
#     Shift-tolerant CE: compute CE for deltas in [-max_shift, +max_shift],
#     then take the minimum loss (hard min). This makes the objective less
#     sensitive to small temporal misalignment.

#     NOTE:
#       - This is still token-level CE; it won't solve multimodality, but it helps
#         the "shift by 1 => 100% wrong" pathology.
#     """
#     if max_shift <= 0:
#         return compute_losses_and_acc(logits, targets, mask_map, pad_token_ids, slot_names, slot_weights)

#     best = None
#     best_out = None
#     for delta in range(-max_shift, max_shift + 1):
#         tgt_d, mask_d = _shift_targets_and_mask(targets, mask_map, delta, pad_token_ids)
#         out = compute_losses_and_acc(logits, tgt_d, mask_d, pad_token_ids, slot_names, slot_weights)
#         loss_d = out[0]
#         if best is None or loss_d.item() < best:
#             best = loss_d.item()
#             best_out = out

#     return best_out
def compute_losses_and_acc_shift_tolerant(logits, targets, mask_map, pad_token_ids, slot_names, slot_weights, max_shift):
    # 13 tokens 结构比较紧密，建议暂时关闭 shift tolerant，直接调用上面的
    return compute_losses_and_acc(logits, targets, mask_map, pad_token_ids, slot_names, slot_weights)

def info_nce_inbatch(text_vec: torch.Tensor, motion_vec: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Strong symmetric in-batch InfoNCE.
    Expect initial loss around log(B) (~3.5 for B=32).
    """
    B = text_vec.shape[0]
    if B <= 1:
        return text_vec.new_tensor(0.0)

    t = F.normalize(text_vec, dim=-1)
    m = F.normalize(motion_vec, dim=-1)

    logits = (t @ m.t()) / max(temperature, 1e-6)  # [B,B]
    labels = torch.arange(B, device=logits.device)

    loss_t2m = F.cross_entropy(logits, labels)
    loss_m2t = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_t2m + loss_m2t)

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    slot_names,
    slot_weights,
    mask_token_ids: torch.Tensor,  # [K]
    pad_token_ids: torch.Tensor,   # [K]
    epoch: int,
):
    """
    Training with:
      1) curriculum stages (loss computed only on selected slots)
      2) optional 2-step iterative refinement (MaskGIT-style)
      3) epoch schedules: mask_ratio_min, p_textonly_train

    Returns:
      (train_loss, train_acc, train_rank, train_len_loss, hparams_dict)
    """
    model.train()

    # ---- curriculum: which slots participate in the loss this epoch ----
    stage = get_stage(epoch)
    active_slots = stage_active_slots(stage, slot_names)
    active_idx = slots_to_indices(slot_names, active_slots)
    active_pad_ids = pad_token_ids[active_idx]
    active_slot_weights = {n: float(slot_weights.get(n, 1.0)) for n in active_slots}

    # ---- epoch schedules ----
    r_min, p_textonly, p_rag_sim = get_mask_hparams_for_epoch(epoch)
    r_max = 1.0

    infonce_w = float(CONFIG.get("infonce_weight", 1.0))
    infonce_temp = float(CONFIG.get("infonce_temp", 0.07))

    # length head config
    len_w = float(CONFIG.get("length_loss_weight", 0.0))
    len_bin = int(CONFIG.get("length_bin_size", 1))
    max_len = int(CONFIG.get("max_len", 1024))
    num_bins = int(CONFIG.get("length_num_bins", 0))
    if num_bins <= 0:
        num_bins = (max_len + len_bin - 1) // len_bin

    ts_max = int(CONFIG.get("time_shift_aug_max", 0))
    shift_tol = int(CONFIG.get("shift_tolerant_max", 0))

    iterative = bool(CONFIG.get("iterative_train", True))
    iter_keep_frac = float(CONFIG.get("iter_keep_frac", 0.5))
    iter_w1 = float(CONFIG.get("iter_loss_w1", 0.5))
    iter_w2 = float(CONFIG.get("iter_loss_w2", 1.0))

    running_loss = 0.0
    running_rank = 0.0
    running_acc = 0.0
    running_len_loss = 0.0
    n_steps = 0

    # show schedule values in tqdm header
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train s{stage}] rmin={r_min:.3f} ptxt={p_textonly:.3f} prag={p_rag_sim:.3f}")
    for batch in pbar:
        if len(batch) == 6:
            motion, text, lengths, motion_pad_mask, text_pad_mask, names = batch
            glosses = None
        else:
            motion, text, lengths, motion_pad_mask, text_pad_mask, names, glosses = batch
        motion = motion.to(device)  # [B,T,K]
        text = text.to(device)      # [B,L,1024]
        lengths = lengths.to(device)
        motion_pad_mask = motion_pad_mask.to(device)  # [B,T] True=pad
        text_pad_mask = text_pad_mask.to(device)      # [B,L] True=pad
        # # === SANITY CHECK (add right after text_pad_mask/motion_pad_mask to(device)) ===
        # with torch.no_grad():
        #     # 1) text embedding 是否全 0 / 是否几乎一样
        #     t_norm = text.float().norm(dim=-1).mean().item()
        #     t_std  = text.float().std().item()

        #     # 2) text_pad_mask True 的比例（True=PAD 才对）
        #     pad_ratio = text_pad_mask.float().mean().item()
        #     valid_cnt = (~text_pad_mask).sum(dim=1)  # 每条样本有效 token 数

        #     # 3) motion_pad_mask 是否和 lengths 对得上
        #     motion_valid_cnt = (~motion_pad_mask).sum(dim=1)

        #     print(f"[SANITY] text_norm_mean={t_norm:.4f} text_std={t_std:.4f} "
        #         f"text_pad_ratio={pad_ratio:.3f} valid_text_cnt(min/mean/max)="
        #         f"{valid_cnt.min().item()}/{valid_cnt.float().mean().item():.1f}/{valid_cnt.max().item()} "
        #         f"motion_valid_cnt(min/mean/max)={motion_valid_cnt.min().item()}/"
        #         f"{motion_valid_cnt.float().mean().item():.1f}/{motion_valid_cnt.max().item()} "
        #         f"len(min/mean/max)={lengths.min().item()}/{lengths.float().mean().item():.1f}/{lengths.max().item()}")
        # optional time shift augmentation (only on motion tokens)

        # ===== CFG training: randomly drop text condition =====
        cfg_p = float(CONFIG.get("cfg_drop_prob", 0.0))
        cfg_drop = None
        if cfg_p > 0:
            B0 = int(motion.shape[0])
            cfg_drop = (torch.rand((B0,), device=device) < cfg_p)  # [B]
            if cfg_drop.any():
                # null text embedding + keep at least 1 valid token to avoid "all-masked attention -> NaN"
                text = text.clone()
                text_pad_mask = text_pad_mask.clone()
                text[cfg_drop] = 0.0
                text_pad_mask[cfg_drop] = True
                text_pad_mask[cfg_drop, 0] = False

        if ts_max > 0:
            motion, lengths, motion_pad_mask = apply_time_shift_aug(
                motion, lengths, motion_pad_mask, pad_token_ids, ts_max
            )

        B, T, K = motion.shape
        # ---- blueprint retrieval (WLASL) ----
        use_rag = bool(CONFIG.get("use_rag", False))
        bp_tokens = None
        bp_pad_mask = None
        bp_stats = None

        if use_rag:
            if ("_load_wlasl_map" not in globals()) or ("build_blueprint_batch" not in globals()):
                print("[RAG] use_rag=True but _load_wlasl_map/build_blueprint_batch not found in this script. Disable rag for this run.")
                use_rag = False
            else:
                wmap = _load_wlasl_map(CONFIG.get("dataset_root", ""))
                bp_tokens, bp_pad_mask, bp_stats = build_blueprint_batch(
                    glosses=glosses,
                    wmap=wmap,
                    pad_token_ids=pad_token_ids,   # [K]
                    device=device,
                    K=K,
                    max_words=int(CONFIG.get("bp_max_words", 64)),
                    per_word_max_T=int(CONFIG.get("bp_per_word_max_T", 48)),
                    total_max_T=int(CONFIG.get("bp_total_max_T", 384)),
                )
                if n_steps % 200 == 0:
                    print(f"[WLASL] hit_rate={bp_stats['hit_rate']:.3f} Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})")
                    L = int(text.shape[1])
                    Tb = int(bp_tokens.shape[1])
                    print(f"[COND] text_L={L} bp_Tb={Tb} cond_L={L+Tb}")
        # mask map
        if np.random.rand() < p_textonly:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            mask_map = valid
        else:
            mask_map = sample_mask_map(lengths, T, device, r_min, r_max)  # [B,T,K] now

        # apply mask to all slots
        x0 = motion.clone()
        x0[mask_map] = mask_token_ids.view(1, 1, K).expand(B, T, K)[mask_map]

        key_padding_mask = build_key_padding_mask(text_pad_mask, motion_pad_mask)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            # ===== forward =====
            if iterative:
                logits1, reps1 = model(x0, text, key_padding_mask=key_padding_mask, return_reps=True, blueprint_tokens=bp_tokens,blueprint_pad_mask=bp_pad_mask)
                logits1_active = {n: logits1[n] for n in active_slots}
                tgt_active = motion[:, :, active_idx]
                mask_map_loss1 = slice_mask_map_for_active(mask_map, active_idx)

                if shift_tol > 0:
                    loss1, _, _, acc1 = compute_losses_and_acc_shift_tolerant(
                        logits1_active, tgt_active, mask_map_loss1, active_pad_ids, active_slots, active_slot_weights, shift_tol
                    )
                else:
                    loss1, _, _, acc1 = compute_losses_and_acc(
                        logits1_active, tgt_active, mask_map_loss1, active_pad_ids, active_slots, active_slot_weights
                    )

                keep_map = _pick_top_confidence_positions(logits1, active_slots, mask_map, keep_frac=iter_keep_frac)

                x1 = x0.clone()
                for k_i, name in enumerate(slot_names):
                    pred = torch.argmax(logits1[name], dim=-1)
                    x1[..., k_i][keep_map] = pred[keep_map]

                if mask_map.dim() == 3:
                    mask_map2 = mask_map & (~keep_map.unsqueeze(-1))   # [B,T,1] -> broadcast to [B,T,K]
                else:
                    mask_map2 = mask_map & (~keep_map)
                mask_map_loss2 = slice_mask_map_for_active(mask_map2, active_idx)

                logits2, reps2 = model(x1, text, key_padding_mask=key_padding_mask, return_reps=True, blueprint_tokens=bp_tokens,blueprint_pad_mask=bp_pad_mask)
                logits2_active = {n: logits2[n] for n in active_slots}

                if shift_tol > 0:
                    loss2, _, _, acc2 = compute_losses_and_acc_shift_tolerant(
                        logits2_active, tgt_active, mask_map_loss2, active_pad_ids, active_slots, active_slot_weights, shift_tol
                    )
                else:
                    loss2, _, _, acc2 = compute_losses_and_acc(
                        logits2_active, tgt_active, mask_map_loss2, active_pad_ids, active_slots, active_slot_weights
                    )

                mlm_loss = (iter_w1 * loss1 + iter_w2 * loss2) / max(iter_w1 + iter_w2, 1e-6)
                mean_acc = acc2
                reps = reps2
            else:
                logits, reps = model(x0, text, key_padding_mask=key_padding_mask, return_reps=True,
                     blueprint_tokens=bp_tokens, blueprint_pad_mask=bp_pad_mask)
                logits_active = {n: logits[n] for n in active_slots}
                tgt_active = motion[:, :, active_idx]
                mask_map_loss = slice_mask_map_for_active(mask_map, active_idx)

                if shift_tol > 0:
                    mlm_loss, _, _, mean_acc = compute_losses_and_acc_shift_tolerant(
                        logits_active, tgt_active, mask_map_loss, active_pad_ids, active_slots, active_slot_weights, shift_tol
                    )
                else:
                    mlm_loss, _, _, mean_acc = compute_losses_and_acc(
                        logits_active, tgt_active, mask_map_loss, active_pad_ids, active_slots, active_slot_weights
                    )

            # ===== length loss =====
            len_loss = mlm_loss.new_tensor(0.0)
            if len_w > 0 and isinstance(reps, dict) and ("len_logits" in reps):
                cond_mask = None if (cfg_drop is None) else (~cfg_drop)
                if (cond_mask is None) or cond_mask.any():
                    if cond_mask is None:
                        tgt_bin = torch.clamp((lengths - 1) // len_bin, 0, num_bins - 1).long()
                        len_logits = reps["len_logits"]
                    else:
                        tgt_bin = torch.clamp((lengths[cond_mask] - 1) // len_bin, 0, num_bins - 1).long()
                        len_logits = reps["len_logits"][cond_mask]
                    len_loss = torch.nn.functional.cross_entropy(len_logits, tgt_bin)

            # ===== rank/InfoNCE =====
            rank_loss = mlm_loss.new_tensor(0.0)
            if infonce_w > 0 and isinstance(reps, dict) and ("text" in reps) and ("motion" in reps):
                cond_mask = None if (cfg_drop is None) else (~cfg_drop)
                if (cond_mask is None) or (cond_mask.sum().item() >= 2):
                    if cond_mask is None:
                        rank_loss = info_nce_inbatch(reps["text"], reps["motion"], temperature=infonce_temp)
                    else:
                        rank_loss = info_nce_inbatch(reps["text"][cond_mask], reps["motion"][cond_mask], temperature=infonce_temp)

            loss = mlm_loss + infonce_w * rank_loss + len_w * len_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG.get("grad_clip", 1.0)))
            optimizer.step()

        running_loss += float(loss.item())
        running_rank += float(rank_loss.item())
        running_acc += float(mean_acc)
        running_len_loss += float(len_loss.item())
        n_steps += 1

        if n_steps % 50 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mlm": f"{mlm_loss.item():.4f}",
                "len": f"{len_loss.item():.4f}" if len_w > 0 else "off",
                "rank": f"{rank_loss.item():.4f}",
                "m_acc": f"{mean_acc*100:.1f}%",
                "rmin": f"{r_min:.2f}",
                "ptxt": f"{p_textonly:.2f}",
                "slots": ",".join(active_slots),
            })
        else:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "m_acc": f"{mean_acc*100:.1f}%",
                "rmin": f"{r_min:.2f}",
                "ptxt": f"{p_textonly:.2f}",
                "len": f"{len_loss.item():.3f}" if len_w > 0 else "off"
            })

    hparams = {
        "stage": int(stage),
        "active_slots": list(active_slots),
        "mask_ratio_min": float(r_min),
        "p_textonly_train": float(p_textonly),
    }

    return (
        running_loss / max(1, n_steps),
        running_acc / max(1, n_steps),
        running_rank / max(1, n_steps),
        running_len_loss / max(1, n_steps),
        hparams,
    )
@torch.no_grad()
def eval_one(
    model,
    loader,
    device,
    slot_names,
    slot_weights,
    mask_token_ids: torch.Tensor,
    pad_token_ids: torch.Tensor,
    epoch: int,
    mode: str = "normal",
):
    """
    mode:
      - normal: random mask like train (r in [mask_ratio_min(epoch), 1.0])
      - text_only: mask all valid positions
      - text_shuffle: shuffle text in batch (strong diag)
      - text_zero: zero out text embeddings
    Metrics are computed ONLY on active slots for the current stage.
    """
    model.eval()

    stage = get_stage(epoch)
    active_slots = stage_active_slots(stage, slot_names)
    active_idx = slots_to_indices(slot_names, active_slots)
    active_pad_ids = pad_token_ids[active_idx]
    active_slot_weights = {n: float(slot_weights.get(n, 1.0)) for n in active_slots}

    r_min, _, p_rag_sim = get_mask_hparams_for_epoch(epoch)
    r_max = 1.0

    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    pbar = tqdm(loader, desc=f"[Val:{mode} s{stage}] rmin={r_min:.3f}")
    for batch in pbar:
        if len(batch) == 6:
            motion, text, lengths, motion_pad_mask, text_pad_mask, names = batch
            glosses = None
        else:
            motion, text, lengths, motion_pad_mask, text_pad_mask, names, glosses = batch
        motion = motion.to(device)
        text = text.to(device)
        lengths = lengths.to(device)
        motion_pad_mask = motion_pad_mask.to(device)
        text_pad_mask = text_pad_mask.to(device)

        B, T, K = motion.shape

        if mode == "text_only":
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            mask_map = valid
        else:
            mask_map = sample_mask_map(lengths, T, device, r_min, r_max)  # [B,T,K] now

        x = motion.clone()
        x[mask_map] = mask_token_ids.view(1, 1, K).expand(B, T, K)[mask_map]

        # text diagnostics
        if mode == "text_shuffle":
            perm = torch.randperm(B, device=device)
            text = text[perm]
            text_pad_mask = text_pad_mask[perm]
        elif mode == "text_zero":
            text = torch.zeros_like(text)

        key_padding_mask = build_key_padding_mask(text_pad_mask, motion_pad_mask)

        logits = model(x, text, key_padding_mask=key_padding_mask, return_reps=False)

        logits_active = {n: logits[n] for n in active_slots}
        tgt_active = motion[:, :, active_idx]
        mask_map_loss = slice_mask_map_for_active(mask_map, active_idx)

        loss, _, _, mean_acc = compute_losses_and_acc(
            logits_active, tgt_active, mask_map_loss, active_pad_ids, active_slots, active_slot_weights
        )

        total_loss += float(loss.item())
        total_acc += float(mean_acc)
        n_steps += 1

    return total_loss / max(1, n_steps), total_acc / max(1, n_steps)


def save_ckpt(path, model, optimizer, scaler, epoch, meta):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "meta": meta,
        "config": CONFIG,
    }
    torch.save(payload, path)
def load_ckpt(path, model, optimizer=None, scaler=None, map_location="cpu", resume_optim: bool = True):
    payload = torch.load(path, map_location=map_location)

    def _strip_module_prefix(state_dict: dict):
        if not isinstance(state_dict, dict):
            return state_dict
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    sd = payload.get("model", None)
    if sd is None:
        sd = payload.get("state_dict", None)
    if sd is None:
        raise RuntimeError("Checkpoint missing 'model' (or 'state_dict').")

    sd = _strip_module_prefix(sd)

    # 1) try strict load first (keep old behavior)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        # 2) allow only blueprint/rag keys to be missing
        #    if mismatch is NOT only bp_*, re-raise
        tmp = model.load_state_dict(sd, strict=False)
        missing = list(getattr(tmp, "missing_keys", []))
        unexpected = list(getattr(tmp, "unexpected_keys", []))

        def _is_bp_key(k: str) -> bool:
            return k.startswith("bp_") or k.startswith("blueprint_") or k.startswith("rag_")

        only_bp_missing = (len(missing) > 0) and all(_is_bp_key(k) for k in missing)
        only_bp_unexpected = (len(unexpected) > 0) and all(_is_bp_key(k) for k in unexpected)

        if (len(unexpected) > 0 and not only_bp_unexpected) or (len(missing) > 0 and not only_bp_missing):
            # real mismatch -> don't silently continue
            raise RuntimeError(
                f"[load_ckpt] Strict load failed and mismatched keys are not only bp_/rag keys.\n"
                f"Original error: {e}\n"
                f"missing_keys (first 20): {missing[:20]}\n"
                f"unexpected_keys (first 20): {unexpected[:20]}"
            )

        # bp-only mismatch -> accept strict=False, warn once
        print(
            "[load_ckpt] ⚠️ Loaded checkpoint with strict=False (bp_/rag keys mismatched).\n"
            f"  missing bp/rag keys: {len(missing)}\n"
            f"  unexpected bp/rag keys: {len(unexpected)}\n"
            "  => blueprint modules will use random init (this is expected when loading old non-RAG ckpt)."
        )

    if resume_optim and optimizer is not None and ("optim" in payload):
        optimizer.load_state_dict(payload["optim"])
    if resume_optim and scaler is not None and ("scaler" in payload):
        scaler.load_state_dict(payload["scaler"])

    start_epoch = int(payload.get("epoch", 0))
    meta = payload.get("meta", None)
    return start_epoch, meta, payload
def load_qvae_for_vis(config_path, model_path, device):
    """动态加载 QVAE 模型"""
    print(f"🎨 Loading QVAE for visualization...")
    print(f"   Config: {config_path}")
    print(f"   Model:  {model_path}")

    # 1. 动态导入 QVAE 类 (假设它在当前目录或 PYTHONPATH 下)
    #sys.path.append("../VAE")
    from mymodel.vae.qvae_model_rod3_fixed_length import VAE

    # 2. 加载配置 (复用你之前的 load_config_from_txt)
    # 这里为了独立性，简单内联一个简易版，或者你可以复用之前的
    def _load_simple_opt(path):
        opt = argparse.Namespace()
        with open(path, 'r') as f:
            for line in f:
                if ':' in line and '---' not in line:
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    # 简单类型转换
                    if v.lower() == 'true': v = True
                    elif v.lower() == 'false': v = False
                    elif v.isdigit(): v = int(v)
                    else:
                        try: v = float(v)
                        except: pass
                    setattr(opt, k, v)
        # 补丁：确保 device 存在
        opt.device = device
        return opt

    opt = _load_simple_opt(config_path)
    opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    # 3. 初始化并加载权重
    qvae = VAE(opt).to(device)
    ckpt = torch.load(model_path, map_location=device)
    # 兼容不同的保存格式
    if isinstance(ckpt, dict):
        if 'vae' in ckpt: qvae.load_state_dict(ckpt['vae'])
        elif 'model' in ckpt: qvae.load_state_dict(ckpt['model'])
        else: qvae.load_state_dict(ckpt)
    else:
        qvae.load_state_dict(ckpt)
    
    qvae.eval()
    return qvae
@torch.no_grad()
def visualize_and_save(maskgit_model, val_loader, config, device):
    """
    Run generation and save as full SMPL-X format npz.
    Compatible with BOTH:
      - non-RAG batch: (motion_gt, text_emb, lengths, motion_pad_mask, text_pad_mask, names)
      - RAG batch:     (motion_gt, text_emb, lengths, motion_pad_mask, text_pad_mask, names, glosses)
    """
    print("\n🎬 Starting Visualization Generation (Full SMPL-X format)...")

    # 1) Prepare QVAE
    qvae_cfg = config.get("qvae_config_path")
    qvae_pth = config.get("qvae_model_path")
    if not qvae_cfg or not qvae_pth:
        print("⚠️ Skipping visualization: 'qvae_config_path' or 'qvae_model_path' not set.")
        return

    qvae = load_qvae_for_vis(qvae_cfg, qvae_pth, device)
    if qvae is None:
        return

    maskgit_model.eval()

    # 2) Get 1 batch
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return

    # 3) Unpack batch (6 or 7)
    glosses = None
    if isinstance(batch, (list, tuple)) and len(batch) == 6:
        motion_gt, text_emb, lengths, motion_pad_mask, text_pad_mask, names = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 7:
        motion_gt, text_emb, lengths, motion_pad_mask, text_pad_mask, names, glosses = batch
    else:
        raise ValueError(f"[visualize_and_save] Unexpected batch format, len={len(batch) if isinstance(batch,(list,tuple)) else type(batch)}")

    # 4) (Optional) build blueprint for this batch if glosses exist
    bp_tokens = None
    bp_pad_mask = None
    if glosses is not None:
        try:
            wmap = _load_wlasl_map(config.get("dataset_root", ""))
            meta = load_metadata(config["dataset_root"])
            codebook_sizes = list(meta["codebook_sizes"])
            pad_token_ids = torch.tensor([cb + 1 for cb in codebook_sizes], dtype=torch.long, device=device)

            K = int(motion_gt.shape[-1])
            bp_tokens, bp_pad_mask, _ = build_blueprint_batch(
                glosses=glosses,
                wmap=wmap,
                pad_token_ids=pad_token_ids,
                device=device,
                K=K,
                max_words=int(config.get("bp_max_words", 64)),
                per_word_max_T=int(config.get("bp_per_word_max_T", 48)),
                total_max_T=int(config.get("bp_total_max_T", 384)),
            )
        except Exception as e:
            print(f"[visualize_and_save] ⚠️ build blueprint failed, fallback to non-RAG vis. err={e}")
            bp_tokens = None
            bp_pad_mask = None

    # 5) Take top 3
    num_samples = min(3, motion_gt.shape[0])
    results_dir = os.path.join(config["save_dir"], "vis_results")
    os.makedirs(results_dir, exist_ok=True)

    def _call_generate(model, **kwargs):
        # make it work for both models with/without blueprint kwargs
        try:
            return model.generate(**kwargs)
        except TypeError:
            kwargs.pop("blueprint_tokens", None)
            kwargs.pop("blueprint_pad_mask", None)
            return model.generate(**kwargs)

    for i in range(num_samples):
        curr_text = text_emb[i:i+1].to(device)
        curr_text_mask = text_pad_mask[i:i+1].to(device)
        curr_name = names[i]
        gt_len = int(lengths[i].item())

        print(f"   Generating sample {i+1}/{num_samples}: {curr_name} (GT Len: {gt_len})")

        pred_len = gt_len  # debug: force GT length

        pred_tokens = _call_generate(
            maskgit_model,
            text_emb=curr_text,
            text_pad_mask=curr_text_mask,
            seq_len=pred_len,
            num_steps=int(config.get("vis_num_steps", 10)),
            temperature=float(config.get("vis_temperature", 1.0)),
            cfg_scale=float(config.get("vis_cfg_scale", 2.0)),
            blueprint_tokens=None if bp_tokens is None else bp_tokens[i:i+1],
            blueprint_pad_mask=None if bp_pad_mask is None else bp_pad_mask[i:i+1],
        )

        recon_motion = qvae.decode_from_tokens(pred_tokens)

        gt_tokens = motion_gt[i:i+1, :gt_len, :].to(device)
        gt_motion = qvae.decode_from_tokens(gt_tokens)

        def save_amass_npz(motion_tensor, save_name, suffix):
            poses_data = motion_tensor[0].cpu().numpy()
            poses_43 = poses_data.reshape(-1, 43, 3)
            T = poses_43.shape[0]

            poses_55 = np.zeros((T, 55, 3), dtype=np.float32)
            poses_55[:, SELECTED_JOINT_INDICES, :] = poses_43

            poses_flat = poses_55.reshape(T, -1)
            amass_data = {
                "poses": poses_flat,
                "trans": np.zeros((T, 3), dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "mocap_framerate": 24,
                "gender": "neutral",
                "surface_model_type": "smplx",
                "num_betas": 10,
                "num_dmpls": 8,
            }
            final_path = os.path.join(results_dir, f"{save_name}_{suffix}.npz")
            np.savez(final_path, **amass_data)

        save_amass_npz(recon_motion, curr_name, f"pred_len{pred_len}")
        save_amass_npz(gt_motion, curr_name, f"gt_len{gt_len}")

    print(f"✅ Visualization saved to {results_dir}")

def debug_print_and_assert_meta(meta: dict):
    slots = list(meta["slots"])
    K = int(meta["K"])
    slot2q = list(meta["slot2q_idx"])
    q2s = dict(meta["q_idx_to_size"])
    cb_sizes = list(meta["codebook_sizes"])
    gname = dict(meta.get("group_name_by_q", {}))

    assert len(slots) == K, (len(slots), K)
    assert len(slot2q) == K, (len(slot2q), K)
    assert len(cb_sizes) == K, (len(cb_sizes), K)

    # q_idx must exist in q2s
    for q in set(int(x) for x in slot2q):
        assert q in q2s and int(q2s[q]) > 0, f"q_idx {q} missing/invalid in q_idx_to_size"

    # per-slot cb size must match q_idx size
    for i in range(K):
        expect = int(q2s[int(slot2q[i])])
        assert int(cb_sizes[i]) == expect, f"slot {i} cb mismatch: codebook_sizes[{i}]={cb_sizes[i]} vs q2s[{slot2q[i]}]={expect}"

    print("\n========== [Metadata Summary] ==========")
    print(f"K={K} | num_groups={meta.get('num_groups', 'NA')} | family={meta.get('model_family','?')}")
    print("slot_idx | slot_name    | q_idx | group_name     | codebook_size | MASK_ID | PAD_ID")
    for i in range(K):
        q = int(slot2q[i])
        cb = int(cb_sizes[i])
        mask_id = cb
        pad_id = cb + 1
        gn = gname.get(q, f"q{q}")
        print(f"{i:7d} | {slots[i]:11s} | {q:5d} | {gn:13s} | {cb:12d} | {mask_id:7d} | {pad_id:6d}")
    print("========================================\n")
def save_training_report(save_dir: str, meta: dict, history: list, best_val: float, best_path: str, final_path: str, run_seconds: float):
    """
    Writes:
      - <save_dir>/train_report.json
      - <save_dir>/train_report.md
      - <save_dir>/runs.jsonl (append one-line summary per run)
    """
    import time
    import json
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = os.path.join(save_dir, "train_report.json")
    report_md_path = os.path.join(save_dir, "train_report.md")
    runs_jsonl_path = os.path.join(save_dir, "runs.jsonl")

    # summarize last epoch
    last = history[-1] if len(history) > 0 else {}
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "save_dir": save_dir,
        "dataset_root": str(CONFIG.get("dataset_root", "")),
        "text_emb_base": str(CONFIG.get("text_emb_base", "")),
        "epochs": int(CONFIG.get("epochs", 0)),
        "batch_size": int(CONFIG.get("batch_size", 0)),
        "lr": float(CONFIG.get("lr", 0.0)),
        "dim": int(CONFIG.get("dim", 0)),
        "depth": int(CONFIG.get("depth", 0)),
        "heads": int(CONFIG.get("heads", 0)),
        "dropout": float(CONFIG.get("dropout", 0.0)),
        "mask_ratio_min_final": float(CONFIG.get("mask_ratio_min", 0.0)),
        "p_textonly_train_final": float(CONFIG.get("p_textonly_train", 0.0)),
        "best_val_loss": float(best_val),
        "best_ckpt": best_path,
        "final_ckpt": final_path,
        "run_seconds": float(run_seconds),
        "last_epoch": int(last.get("epoch", 0)),
        "last_train_loss": float(last.get("train_loss", 0.0) or 0.0),
        "last_train_macc": float(last.get("train_macc", 0.0) or 0.0),
        "last_val_loss": float(last.get("val_loss", 0.0) or 0.0) if ("val_loss" in last) else None,
        "last_val_macc": float(last.get("val_macc", 0.0) or 0.0) if ("val_macc" in last) else None,
    }

    payload = {
        "summary": summary,
        "config": CONFIG,
        "meta": meta,
        "history": history,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Markdown report (compact + comparable)
    lines = []
    lines.append(f"# MaskGIT Training Report ({run_id})")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- save_dir: `{save_dir}`")
    lines.append(f"- dataset_root: `{summary['dataset_root']}`")
    lines.append(f"- epochs: {summary['epochs']} | batch_size: {summary['batch_size']} | lr: {summary['lr']}")
    lines.append(f"- model: dim={summary['dim']} depth={summary['depth']} heads={summary['heads']} dropout={summary['dropout']}")
    lines.append(f"- final schedule targets: mask_ratio_min={summary['mask_ratio_min_final']} | p_textonly_train={summary['p_textonly_train_final']}")
    lines.append(f"- best_val_loss: {summary['best_val_loss']:.6f} (ckpt: `{summary['best_ckpt']}`)")
    lines.append(f"- final_ckpt: `{summary['final_ckpt']}`")
    lines.append(f"- runtime: {summary['run_seconds']:.1f} sec")
    lines.append("")

    lines.append("## Slots / Groups")
    slots = list(meta.get("slots", []))
    slot2q = list(meta.get("slot2q_idx", []))
    q2s = dict(meta.get("q_idx_to_size", {}))
    gname = dict(meta.get("group_name_by_q", {}))
    lines.append("")
    lines.append("| slot_idx | slot_name | q_idx | group | codebook_size |")
    lines.append("|---:|---|---:|---|---:|")
    for i, n in enumerate(slots):
        q = int(slot2q[i]) if i < len(slot2q) else -1
        cb = int(q2s.get(q, -1))
        gn = str(gname.get(q, f"q{q}"))
        lines.append(f"| {i} | {n} | {q} | {gn} | {cb} |")
    lines.append("")

    lines.append("## Epoch History")
    lines.append("")
    lines.append("| ep | stage | r_min | p_txt | train_loss | train_macc | rank | len_loss | val_loss | val_macc | diag |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for h in history:
        ep = int(h.get("epoch", 0))
        st = int(h.get("stage", -1)) if h.get("stage", None) is not None else -1
        rmin = h.get("mask_ratio_min", None)
        ptxt = h.get("p_textonly_train", None)
        tl = h.get("train_loss", None)
        ta = h.get("train_macc", None)
        rk = h.get("train_rank", None)
        ll = h.get("train_len_loss", None)
        vl = h.get("val_loss", None)
        va = h.get("val_macc", None)
        dg = h.get("diag_hint", "")
        lines.append(
            f"| {ep} | {st} | {rmin if rmin is not None else ''} | {ptxt if ptxt is not None else ''} | "
            f"{tl if tl is not None else ''} | {ta if ta is not None else ''} | {rk if rk is not None else ''} | {ll if ll is not None else ''} | "
            f"{vl if vl is not None else ''} | {va if va is not None else ''} | {dg} |"
        )

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # append runs.jsonl
    try:
        with open(runs_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to append runs.jsonl: {e}")

    print(f"📝 Saved report: {report_json_path}")
    print(f"📝 Saved report: {report_md_path}")

class EarlyStopper:
    """
    Early stopping on a scalar metric (default: val_loss, minimize).
    Only call step() when you have a fresh evaluation metric (e.g., every eval_every epochs).

    Args:
      patience: number of eval points with no improvement allowed
      min_delta: minimum improvement required to reset patience
      mode: "min" or "max"
      min_epochs: do not early-stop before this epoch number (1-based)
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min", min_epochs: int = 0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode).lower().strip()
        self.min_epochs = int(min_epochs)

        if self.mode not in ("min", "max"):
            raise ValueError(f"EarlyStopper mode must be 'min' or 'max', got: {mode}")

        self.best = None
        self.bad_count = 0
        self.best_epoch = None

    def _is_improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        else:
            return value > (self.best + self.min_delta)

    def step(self, value: float, epoch_1based: int) -> bool:
        """
        Returns True if should STOP now.
        """
        epoch_1based = int(epoch_1based)
        value = float(value)

        if epoch_1based < self.min_epochs:
            # still update best, but do not stop
            if self._is_improved(value):
                self.best = value
                self.best_epoch = epoch_1based
                self.bad_count = 0
            return False

        if self._is_improved(value):
            self.best = value
            self.best_epoch = epoch_1based
            self.bad_count = 0
            return False

        self.bad_count += 1
        return self.bad_count >= self.patience



def preflight_check_text_embeddings(dataset_root: str, emb_dirs: list, split: str, n_check: int = 50):
    """
    强制检查 embedding 文件是否真的存在。
    emb_dirs: 例如 ["C:/.../text_embedding/train", "C:/.../gloss_embedding/train"]
    """
    import os, json, random

    jpath = os.path.join(dataset_root, f"{split}_dataset.json")
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"[Preflight] dataset json not found: {jpath}")

    data = json.load(open(jpath, "r", encoding="utf-8"))
    if len(data) == 0:
        raise RuntimeError(f"[Preflight] {split}_dataset.json is empty")

    # 随机抽样检查
    idxs = list(range(len(data)))
    random.shuffle(idxs)
    idxs = idxs[:min(n_check, len(idxs))]

    missing = []
    for i in idxs:
        name = data[i]["name"]
        fn = name + ".pt"
        for d in emb_dirs:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                missing.append((name, p))

    if len(missing) > 0:
        # 只打印前几个，避免刷屏
        msg = "\n".join([f"  name={nm}  missing={p}" for nm, p in missing[:10]])
        raise FileNotFoundError(
            f"[Preflight] Missing embedding files! (showing first {min(10,len(missing))})\n{msg}\n"
            f"Tips: emb_dirs 应该是 *split目录*，例如 .../text_embedding/train，而不是根目录。\n"
        )

    print(f"[Preflight] OK: checked {len(idxs)} samples, all embedding files exist in {len(emb_dirs)} dirs.")

def main():
    import time

    t0 = time.time()

    set_seed(int(CONFIG.get("seed", 1234)))
    ensure_dir(CONFIG["save_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load metadata (drives K/slots/codebooks) ----
    meta_path = os.path.join(CONFIG["dataset_root"], "dataset_metadata.json")
    meta = load_metadata(meta_path)
    slot_names = list(meta["slots"])
    codebook_sizes = list(meta["codebook_sizes"])
    K = int(meta["K"])

    print("🚀 MaskGIT Training Started")
    print(f"   K={K}")
    print(f"   slots={slot_names}")

    # per-slot special ids
    mask_token_ids = torch.tensor([cb for cb in codebook_sizes], dtype=torch.long, device=device)      # MASK = cb
    pad_token_ids = torch.tensor([cb + 1 for cb in codebook_sizes], dtype=torch.long, device=device)  # PAD = cb+1

    # ---- datasets ----
    # Support multi-condition embeddings (e.g., English + Gloss) by passing a list of directories.
    # Priority: CONFIG['text_emb_bases'] (list) > (text_emb_base + optional gloss_emb_base)
    def _build_text_emb_dirs(split: str):
        bases = CONFIG.get("text_emb_bases", [])
        out = []
        if isinstance(bases, (list, tuple)) and len(bases) > 0:
            for b in bases:
                if b:
                    out.append(os.path.join(str(b), split))
        else:
            b0 = CONFIG.get("text_emb_base", "")
            if b0:
                out.append(os.path.join(str(b0), split))
            b1 = CONFIG.get("gloss_emb_base", "")
            if b1:
                out.append(os.path.join(str(b1), split))
        return out

    train_emb_dir = _build_text_emb_dirs("train")
    val_emb_dir = _build_text_emb_dirs("val")
    text_source = str(CONFIG.get("text_source", "text"))
    max_text_len = CONFIG.get("max_text_len", None)

    print(f"   Text embedding dirs (train): {train_emb_dir}")
    print(f"   Text embedding dirs (val):   {val_emb_dir}")
    preflight_check_text_embeddings(CONFIG["dataset_root"], train_emb_dir, "train", n_check=50)
    preflight_check_text_embeddings(CONFIG["dataset_root"], val_emb_dir, "val", n_check=50)

    train_ds_full = SignMotionTokenDataset(
        dataset_root=CONFIG["dataset_root"],
        split="train",
        text_emb_dir=train_emb_dir,
        max_len=int(CONFIG["max_len"]),
        max_text_len=max_text_len,
        text_source=text_source,
        meta=meta,
    )
    val_ds_full = SignMotionTokenDataset(
        dataset_root=CONFIG["dataset_root"],
        split="val",
        text_emb_dir=val_emb_dir,
        max_len=int(CONFIG["max_len"]),
        max_text_len=max_text_len,
        text_source=text_source,
        meta=meta,
    )

    train_ds = maybe_make_debug_subset(train_ds_full, int(CONFIG.get("debug_train_n", 0)), seed=int(CONFIG.get("seed", 0)))
    val_ds = maybe_make_debug_subset(val_ds_full, int(CONFIG.get("debug_val_n", 0)), seed=int(CONFIG.get("seed", 0)) + 1)

    # health scan
    health_scan(train_ds_full, "train", n_scan=int(CONFIG.get("healthscan_n", 2000)))
    health_scan(val_ds_full, "val", n_scan=len(val_ds))

    # loaders
    collate = partial(pad_collate, codebook_sizes=codebook_sizes)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=(int(CONFIG["num_workers"]) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        persistent_workers=(max(0, int(CONFIG["num_workers"]) // 2) > 0),
    )

    # ---- slot weights (optional) ----
    slot_weights = {n: 1.0 for n in slot_names}
    if "slot_weights" in CONFIG and isinstance(CONFIG["slot_weights"], dict):
        for k, v in CONFIG["slot_weights"].items():
            if k in slot_weights:
                slot_weights[k] = float(v)

    # print only weights != 1.0 (helps verify it is actually loaded)
    _sw_non1 = {k:v for k,v in slot_weights.items() if abs(float(v) - 1.0) > 1e-9}
    if len(_sw_non1) > 0:
        print(f"   slot_weights override: {_sw_non1}")


    # ---- model ----
    model = MaskGITTransformer(
        slot_names=slot_names,
        codebook_sizes=codebook_sizes,
        dim=int(CONFIG["dim"]),
        depth=int(CONFIG["depth"]),
        heads=int(CONFIG["heads"]),
        text_dim=int(CONFIG["text_dim"]),
        max_seq_len=int(CONFIG["max_seq_len"]),
        dropout=float(CONFIG["dropout"]),
        length_bin_size=int(CONFIG.get("length_bin_size", 1)),
        length_num_bins=int(CONFIG.get("length_num_bins", 0)),
        slot2q_idx=meta.get("slot2q_idx", None),
        q_idx_to_size=meta.get("q_idx_to_size", None),
        tie_groups=True,
        flatten_spatiotemporal=True,
        use_blueprint=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(CONFIG["lr"]), weight_decay=float(CONFIG["weight_decay"]))
    #scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")) #fp16
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 no scaler

    # ---- resume (optional) ----
    start_epoch = 0
    resume_path = str(CONFIG.get("resume_path", "")).strip()
    if (not resume_path) and bool(CONFIG.get("resume_last", False)):
        cand = os.path.join(CONFIG["save_dir"], "maskgit_last.tar")
        if os.path.exists(cand):
            resume_path = cand
        else:
            raise ValueError(f"resume_last is True but no last checkpoint found at: {cand}")
    print("resume_path =", repr(resume_path))

    print("resume_exists =", os.path.exists(resume_path))
    if resume_path and os.path.exists(resume_path):
        se, meta2, payload = load_ckpt(
            resume_path,
            model,
            optimizer=optimizer,
            scaler=scaler,
            map_location="cpu",
            resume_optim=bool(CONFIG.get("resume_optim", True)),
        )
        start_epoch = int(se)
        print(f"↩️ Resumed from: {resume_path} (start_epoch={start_epoch})")

    best_val = 1e9
    best_path = os.path.join(CONFIG["save_dir"], "maskgit_best.tar")
    final_path = os.path.join(CONFIG["save_dir"], "maskgit_last.tar")

    # ---- early stopping ----
    early_stop_enabled = bool(CONFIG.get("early_stop", True))
    early_stop_patience = int(CONFIG.get("early_stop_patience", 5))
    early_stop_min_delta = float(CONFIG.get("early_stop_min_delta", 0.0))
    early_stop_min_epochs = int(CONFIG.get("early_stop_min_epochs", 0))
    stopper = EarlyStopper(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        mode="min",
        min_epochs=early_stop_min_epochs,
    )

    # ---- history for report ----
    history = []
    last_trained_epoch_1based = start_epoch

    for epoch in range(start_epoch, int(CONFIG["epochs"])):
        # --- NEW: stage-wise LR restart (warmup + cosine decay) ---
        cur_lr = _compute_stagewise_lr(epoch)
        _set_optimizer_lr(optimizer, cur_lr)
        train_loss, train_acc, train_rank, train_len_loss, hparams = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            slot_names, slot_weights, mask_token_ids, pad_token_ids, epoch
        )

        epoch_1based = int(epoch + 1)
        last_trained_epoch_1based = epoch_1based

        hist_entry = {
            "epoch": epoch_1based,
            "stage": int(hparams.get("stage", -1)),
            "mask_ratio_min": float(hparams.get("mask_ratio_min", 0.0)),
            "p_textonly_train": float(hparams.get("p_textonly_train", 0.0)),
            "train_loss": float(train_loss),
            "train_macc": float(train_acc),
            "train_rank": float(train_rank),
            "train_len_loss": float(train_len_loss),
        }

        log_line = (
            f"[Epoch {epoch_1based}] train_loss={train_loss:.4f} "
            f"train_macc={train_acc*100:.1f}% rank={train_rank:.4f} len={train_len_loss:.4f} "
            f"| rmin={hist_entry['mask_ratio_min']:.2f} ptxt={hist_entry['p_textonly_train']:.2f}"
        )

        # eval
        did_eval = ((epoch_1based) % int(CONFIG["eval_every"]) == 0)
        if did_eval:
            val_loss, val_acc = eval_one(model, val_loader, device, slot_names, slot_weights, mask_token_ids, pad_token_ids, epoch=epoch, mode="normal")
            t_only_loss, t_only_acc = eval_one(model, val_loader, device, slot_names, slot_weights, mask_token_ids, pad_token_ids, epoch=epoch, mode="text_only")
            shuf_loss, shuf_acc = eval_one(model, val_loader, device, slot_names, slot_weights, mask_token_ids, pad_token_ids, epoch=epoch, mode="text_shuffle")
            zero_loss, zero_acc = eval_one(model, val_loader, device, slot_names, slot_weights, mask_token_ids, pad_token_ids, epoch=epoch, mode="text_zero")

            diag_hint = "USING_TEXT"
            if (abs(shuf_loss - t_only_loss) < 0.2) and (abs(zero_loss - t_only_loss) < 0.2):
                diag_hint = "IGNORING_TEXT?"

            hist_entry.update({
                "val_loss": float(val_loss),
                "val_macc": float(val_acc),
                "textonly_loss": float(t_only_loss),
                "textonly_macc": float(t_only_acc),
                "shuf_loss": float(shuf_loss),
                "shuf_macc": float(shuf_acc),
                "zero_loss": float(zero_loss),
                "zero_macc": float(zero_acc),
                "diag_hint": str(diag_hint),
            })

            log_line += (
                f" | val_loss={val_loss:.4f} val_macc={val_acc*100:.1f}%"
                f" | textonly={t_only_loss:.4f}"
                f" | shuf={shuf_loss:.4f} zero={zero_loss:.4f}"
                f" | diag={diag_hint}"
            )

            # save best
            if val_loss < best_val:
                best_val = float(val_loss)
                save_ckpt(best_path, model, optimizer, scaler, epoch_1based, meta)
                print(f"💾 Saved checkpoint: {best_path}")

            # early stop check (only on eval epochs)
            if early_stop_enabled:
                should_stop = stopper.step(val_loss, epoch_1based)
                if should_stop:
                    print(
                        f"🛑 Early stopping triggered at epoch {epoch_1based}. "
                        f"Best val_loss={stopper.best:.6f} at epoch {stopper.best_epoch}. "
                        f"(patience={early_stop_patience}, min_delta={early_stop_min_delta})"
                    )
                    history.append(hist_entry)
                    break

        history.append(hist_entry)
        print(log_line)

        # periodic save
        if (epoch_1based) % int(CONFIG["save_every"]) == 0:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"maskgit_ep{epoch_1based}.tar")
            save_ckpt(ckpt_path, model, optimizer, scaler, epoch_1based, meta)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    # always save "last" for the final trained epoch (even if early-stopped)
    save_ckpt(final_path, model, optimizer, scaler, last_trained_epoch_1based, meta)

    # your visualization
    visualize_and_save(model, val_loader, CONFIG, device)

    # save report
    run_seconds = time.time() - t0
    save_training_report(
        save_dir=CONFIG["save_dir"],
        meta=meta,
        history=history,
        best_val=best_val,
        best_path=best_path,
        final_path=final_path,
        run_seconds=run_seconds,
    )

    print("✅ Training Finished.")




if __name__ == "__main__":
    # args = parse_args()

    # user_cfg = load_config_from_path(args.config)

    # merge: user_cfg overrides defaults in this file
    base_cfg = dict(CONFIG)  # CONFIG is your hardcoded default above
    #base_cfg.update(user_cfg)

    # overwrite global CONFIG so existing code keeps working
    CONFIG = base_cfg
    #CONFIG["_config_path"] = str(Path(args.config).expanduser().resolve())

    main()