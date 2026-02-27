    
import os
import json
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------
# Gloss token normalization (keep fingerspelling spans)
# ----------------------
FS_BEGIN_SET = {"fs_begin","<fs_begin>", "[fs_begin]"}
FS_END_SET   = {"fs_end", "<fs_end>", "[fs_end]"}

def normalize_gloss_for_tokens(gloss: str) -> str:
    """
    Normalize a pseudo-gloss string for *token-level conditioning*.

    Goals:
      - KEEP fingerspelling spans between FS_BEGIN..FS_END (do NOT delete inside tokens).
      - Unify FS markers to canonical tokens: 'FS_BEGIN' and 'FS_END'
      - Avoid aggressive punctuation stripping that would break gloss tokens like 'STAND-UP' or 'ZOOM-IN'.

    Notes:
      - Tokens inside FS spans are kept almost as-is (only stripped of surrounding whitespace).
      - Tokens outside FS spans are kept as-is too; you can add more cleanup later if needed,
        but be conservative first to avoid losing semantics.
    """
    s = "" if gloss is None else str(gloss).strip()
    if not s:
        return ""
    toks = s.split()
    out = []
    in_fs = False
    for tok in toks:
        t = tok.strip()
        if not t:
            continue
        key = t.lower()

        if key in FS_BEGIN_SET:
            out.append("FS_BEGIN")
            in_fs = True
            continue
        if key in FS_END_SET:
            out.append("FS_END")
            in_fs = False
            continue

        # Keep token; be conservative
        out.append(t)

    return " ".join(out)


def _infer_sep(path: str) -> str:
    """
    Robust separator inference for csv/tsv.
    - If file suffix suggests TSV, use '\t'
    - Else sniff first non-empty line for tabs vs commas.
    """
    lower = str(path).lower()
    if lower.endswith(".tsv") or lower.endswith(".tab") or lower.endswith(".txt"):
        return "\t"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                # simple sniff
                if s.count("\t") >= s.count(",") and s.count("\t") > 0:
                    return "\t"
                return ","
    except Exception:
        pass
    return ","


def _col_lookup(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"Missing column. Need one of: {candidates}. Got columns={list(df.columns)}")


def _normalize_sample_id(value: Any) -> str:
    sid = "" if value is None else str(value).strip()
    if not sid:
        return ""
    sid = sid.replace("\\", "/").split("/")[-1]
    low = sid.lower()
    if low.endswith("_aioswilor.npz"):
        sid = sid[:-len("_aioswilor.npz")]
    elif low.endswith(".npz") or low.endswith(".mp4"):
        sid = sid[:-4]
    return sid.strip()


def _load_metadata_cache(cache_path: str) -> Dict[str, int]:
    """Load metadata cache and return {filename: length}."""
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    if isinstance(obj, dict):
        if isinstance(obj.get("items", None), list):
            items = obj["items"]
        elif isinstance(obj.get("metadata", None), list):
            items = obj["metadata"]
        else:
            items = []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []

    out: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name.endswith(".npz"):
            continue
        try:
            length = int(item.get("len", 0))
        except Exception:
            continue
        if length > 0:
            out[name] = length
    return out


def _save_metadata_cache(cache_path: str, meta_map: Dict[str, int]) -> None:
    payload = {
        "version": 2,
        "items": [{"name": k, "len": int(v)} for k, v in sorted(meta_map.items())],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class SignDiffusionDataset(Dataset):
    """
    Dataset for diffusion training.

    IMPORTANT CHANGE (to match trainer_patched/model_patched):
      - Always return BOTH sentence and gloss as a pair of strings:
          text_pair = [sentence_str, gloss_str]
      - No precomputed embeddings are loaded here.
      - No is_gloss switch; both are always read.

    __getitem__ returns:
      - no custom weight: (text_pair, motion_flat[T,D], valid_len, sample_id)
      - with custom weight: (text_pair, motion_flat[T,D], valid_len, sample_id, frame_weight[T])
    """
    def __init__(
        self,
        data_dir,
        csv_path,
        max_length,
        config=None,
        is_train=True,
        only_gloss=True,
        enable_custom_weight=False,
        custom_weight_dir="",
        custom_weight_key="soft_w",
        custom_weight_precheck=False,
    ):
        # keep is_gloss arg only for backward compatibility; ignored.
        self.data_dir = data_dir
        self.max_length = max_length
        self.config = config
        self.is_train = is_train
        self.only_gloss = only_gloss
        self.enable_custom_weight = bool(enable_custom_weight)
        self.custom_weight_dir = str(custom_weight_dir or "").strip()
        self.custom_weight_key = str(custom_weight_key or "soft_w").strip()
        self.custom_weight_precheck = bool(custom_weight_precheck)
        # ===== 1) Load text labels (sentence + gloss) =====
        sep = _infer_sep(csv_path)
        df = pd.read_csv(csv_path, sep=sep)

        col_name = _col_lookup(df, ["SENTENCE_NAME", "sentence_name", "name", "SAMPLE_ID", "sample_id", "Video file", "video file", "video_file"])
        try:
            col_sent = _col_lookup(df, ["SENTENCE", "sentence", "TEXT", "text"])
        except KeyError:
            col_sent = None
        col_glos = _col_lookup(df, ["GLOSS", "gloss", "PSEUDO_GLOSS", "pseudo_gloss"])

        # caption_dict: sample_id -> (sentence, gloss)
        self.caption_dict = {}
        sid_list = df[col_name].astype(str).tolist()
        sent_list = df[col_sent].tolist() if col_sent is not None else [""] * len(sid_list)
        glos_list = df[col_glos].tolist()
        for sid, sent, glos in zip(sid_list, sent_list, glos_list):
            sid = _normalize_sample_id(sid)
            if not sid:
                continue
            sent = "" if sent is None else str(sent)
            glos = "" if glos is None else str(glos)
            glos = normalize_gloss_for_tokens(glos)
            self.caption_dict[sid] = (sent, glos)

        # ===== 2) Read/build samples + lengths (cache is global to data_dir, not split-specific) =====
        cache_path = os.path.join(data_dir, "dataset_metadata.json")
        self.samples: List[Tuple[str, str]] = []
        self.lengths: List[int] = []

        def _name_to_sample_id(filename: str):
            return _normalize_sample_id(filename)

        all_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        all_files.sort()

        meta_map = _load_metadata_cache(cache_path)
        if meta_map:
            print(f"[Dataset] Loading cached metadata from {cache_path} ...")
        else:
            print("[Dataset] ⚠️ metadata cache not found/invalid, scanning npz lengths (slow) ...")

        file_set = set(all_files)
        stale_keys = [k for k in meta_map.keys() if k not in file_set]
        if stale_keys:
            for k in stale_keys:
                meta_map.pop(k, None)
            print(f"[Dataset] Removed {len(stale_keys)} stale cache entries.")

        missing = [f for f in all_files if f not in meta_map]
        if missing:
            print(f"[Dataset] Filling metadata for {len(missing)} missing files ...")
            for f in missing:
                path = os.path.join(self.data_dir, f)
                try:
                    with np.load(path, mmap_mode="r") as data:
                        arr = data["joints_xyz"] if bool(getattr(self.config, "xyz", False)) else data["poses"]
                        meta_map[f] = int(arr.shape[0])
                except Exception:
                    continue
            try:
                _save_metadata_cache(cache_path, meta_map)
                print(f"[Dataset] ✅ Metadata cache updated: {cache_path}")
            except Exception:
                pass

        for f in all_files:
            sid = _name_to_sample_id(f)
            if sid not in self.caption_dict:
                continue
            T = int(meta_map.get(f, 0))
            if self.max_length is not None:
                T = min(T, int(self.max_length))
            if T <= 0:
                continue
            self.samples.append((f, sid))
            self.lengths.append(T)

        if self.enable_custom_weight:
            if not self.custom_weight_dir:
                raise ValueError("enable_custom_weight=True but custom_weight_dir is empty.")
            if not os.path.isdir(self.custom_weight_dir):
                raise FileNotFoundError(f"custom_weight_dir does not exist: {self.custom_weight_dir}")
            if self.custom_weight_precheck:
                missing_sidecar = [
                    file_name
                    for (file_name, _) in self.samples
                    if not os.path.isfile(self._sidecar_path(file_name))
                ]
                if missing_sidecar:
                    preview = ", ".join(missing_sidecar[:10])
                    raise FileNotFoundError(
                        f"[CustomWeight] Missing sidecar npz for {len(missing_sidecar)}/{len(self.samples)} samples. "
                        f"First missing: {preview}"
                    )
                print(
                    f"[CustomWeight] precheck passed: {len(self.samples)} samples have sidecar under "
                    f"{self.custom_weight_dir}"
                )

        print(f"[Dataset] Loaded {len(self.samples)} valid samples (csv-matched) from metadata.")

        print(f"Diffusion Dataset: Loaded {len(self.samples)} samples from {data_dir}")

    def calculate_stats(self):
        """
        Compute mean/std for the training set (same as your original logic).
        """
        cache_path = os.path.join(self.data_dir, f"mean_std_cache_{'xyz' if self.config.xyz else 'rot'}.pt")

        if os.path.exists(cache_path):
            print(f"Loading stats from cache: {cache_path}")
            stats = torch.load(cache_path, map_location="cpu")
            return stats["mean"], stats["std"]

        print(f"📊 Calculating stats from scratch (XYZ mode: {self.config.xyz})...")
        all_data = []

        samples_to_scan = self.samples if len(self.samples) < 10000 else random.sample(self.samples, 5000)

        for file_name, _ in samples_to_scan:
            filepath = os.path.join(self.data_dir, file_name)
            with np.load(filepath) as data:
                raw = data["joints_xyz"] if self.config.xyz else data["poses"]
                feat = raw[:, self.config.SELECTED_JOINT_INDICES, :]
                all_data.append(feat.reshape(-1, feat.shape[1] * 3))

        if not all_data:
            raise ValueError("No data loaded for stats calculation!")

        all_data = np.concatenate(all_data, axis=0)

        if self.config.xyz:
            mean = np.mean(all_data, axis=0)
            std = np.std(all_data, axis=0)
            std[std < 1e-5] = 1.0
        else:
            mean = np.zeros(all_data.shape[1])
            std = np.ones(all_data.shape[1])
            for j in range(0, all_data.shape[1], 3):
                joint_std = np.sqrt(np.mean(all_data[:, j:j + 3] ** 2))
                std[j:j + 3] = joint_std if joint_std > 1e-5 else 1.0

        mean_tensor = torch.from_numpy(mean).float()
        std_tensor = torch.from_numpy(std).float()

        torch.save({"mean": mean_tensor, "std": std_tensor}, cache_path)
        print(f"✅ Stats calculated and saved to {cache_path}")

        return mean_tensor, std_tensor

    def __len__(self):
        return len(self.samples)

    def _sidecar_path(self, file_name: str) -> str:
        return os.path.join(self.custom_weight_dir, file_name)

    def _load_frame_weight(self, file_name: str, target_len: int) -> torch.Tensor:
        path = self._sidecar_path(file_name)
        if not os.path.isfile(path):
            return torch.ones((target_len,), dtype=torch.float32)

        try:
            with np.load(path, mmap_mode="r") as data:
                if self.custom_weight_key not in data:
                    return torch.ones((target_len,), dtype=torch.float32)
                w = np.asarray(data[self.custom_weight_key], dtype=np.float32)
        except Exception:
            return torch.ones((target_len,), dtype=torch.float32)

        if w.ndim >= 3:
            w = w[:, 0, 0]
        elif w.ndim == 2:
            w = w[:, 0]
        else:
            w = w.reshape(-1)

        if w.size <= 0:
            return torch.ones((target_len,), dtype=torch.float32)

        wt = torch.from_numpy(w).float().clamp_(0.0, 1.0).view(1, 1, -1)
        if wt.shape[-1] != int(target_len):
            wt = F.interpolate(wt, size=int(target_len), mode="linear", align_corners=False)
        return wt.view(-1).contiguous()

    def __getitem__(self, idx):
        file_name, sample_id = self.samples[idx]
        sent, glos = self.caption_dict[sample_id]
        glos = normalize_gloss_for_tokens(glos)
        if self.only_gloss:
            text_pair = ["", glos]
        else:  
            text_pair = [sent, glos]

        path = os.path.join(self.data_dir, file_name)
        with np.load(path, mmap_mode="r") as data:
            raw_motion = data["joints_xyz"] if self.config.xyz else data["poses"]
            motion = raw_motion[:, self.config.SELECTED_JOINT_INDICES, :]  # [T, J, 3]
            T = int(motion.shape[0])

            # cap (NOT random crop)
            if self.max_length is not None and T > self.max_length:
                motion = motion[: self.max_length]
                T = int(self.max_length)

            motion_flat = torch.from_numpy(motion).float().reshape(T, -1)  # [T, D]
            valid_len = T

        if self.enable_custom_weight:
            frame_weight = self._load_frame_weight(file_name, target_len=T)
            return text_pair, motion_flat, valid_len, sample_id, frame_weight

        return text_pair, motion_flat, valid_len, sample_id


def diffusion_collate_fn(batch):
    """
    batch:
      - no custom weight: (text_pair=[sentence,gloss], motion_flat[T,D], valid_len, sample_id)
      - with custom weight: (text_pair=[sentence,gloss], motion_flat[T,D], valid_len, sample_id, frame_weight[T])

    returns:
      text_out: List[[sentence,gloss]] length B
      motions:  [B, T_max, D]
      masks:    [B, T_max]  (1=valid,0=pad)
      lengths:  [B]
      names:    List[str]
      frame_weights (optional): [B, T_max]
    """
    has_weight = len(batch[0]) >= 5
    if has_weight:
        texts, motions_list, lengths, names, frame_weights_list = zip(*batch)
    else:
        texts, motions_list, lengths, names = zip(*batch)
        frame_weights_list = None

    lengths = torch.tensor(lengths, dtype=torch.long)
    B = len(motions_list)
    D = motions_list[0].shape[-1]

    # motion pad by repeating last frame
    safe_motions = []
    safe_lengths = []
    for m, L in zip(motions_list, lengths.tolist()):
        if L is None or L <= 0:
            safe_motions.append(torch.zeros((1, D), dtype=torch.float32))
            safe_lengths.append(0)
        else:
            safe_motions.append(m)
            safe_lengths.append(int(L))

    lengths = torch.tensor(safe_lengths, dtype=torch.long)
    T_max = int(max([m.shape[0] for m in safe_motions]))

    motions = torch.zeros((B, T_max, D), dtype=torch.float32)
    masks = torch.zeros((B, T_max), dtype=torch.float32)

    for i, (m, L) in enumerate(zip(safe_motions, lengths.tolist())):
        Ti = int(m.shape[0])
        motions[i, :Ti] = m
        if Ti < T_max:
            motions[i, Ti:] = m[-1:].repeat(T_max - Ti, 1)
        if L > 0:
            masks[i, :L] = 1.0

    # text_out: keep as List[[sentence, gloss]]
    text_out: List[List[str]] = []
    for t in texts:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            text_out.append([str(t[0]), str(t[1])])
        else:
            # fallback: treat as sentence only
            text_out.append([str(t) if t is not None else "", ""])

    if has_weight and frame_weights_list is not None:
        frame_weights = torch.ones((B, T_max), dtype=torch.float32)
        for i, fw in enumerate(frame_weights_list):
            if fw is None:
                continue
            wi = fw.float().view(-1)
            if wi.numel() <= 0:
                continue
            Ti = int(min(wi.shape[0], T_max))
            frame_weights[i, :Ti] = wi[:Ti]
            if Ti < T_max:
                frame_weights[i, Ti:] = wi[Ti - 1]
        return text_out, motions, masks, lengths, list(names), frame_weights

    return text_out, motions, masks, lengths, list(names)
