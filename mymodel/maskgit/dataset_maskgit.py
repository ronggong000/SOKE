import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
def load_metadata(meta_path_or_root: str, metadata_filename: str = "dataset_metadata.json"):
    """
    Canonical metadata loader used by BOTH train_maskgit.py and SignMotionTokenDataset.

    Accepts either:
      - full path to dataset_metadata.json
      - dataset_root directory (will join with metadata_filename)

    Guarantees these keys in returned dict:
      - slots: list[str]
      - K: int
      - slot2q_idx: list[int]  (len=K)
      - q_idx_to_size: dict[int,int]
      - codebook_sizes: list[int] (len=K)  # per-slot vocab size (WITHOUT +2)
      - num_groups: int
      - group_name_by_q: dict[int,str]  (optional)
    """
    import os, json

    if meta_path_or_root.endswith(".json"):
        meta_path = meta_path_or_root
    else:
        meta_path = os.path.join(meta_path_or_root, metadata_filename)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "slots" not in meta or "K" not in meta:
        raise RuntimeError(f"Bad metadata: missing slots/K in {meta_path}")

    slots = list(meta["slots"])
    K = int(meta["K"])
    if len(slots) != K:
        raise RuntimeError(f"Bad metadata: len(slots)={len(slots)} != K={K} in {meta_path}")

    groups = meta.get("groups", None)

    # ---- New shared-codebook format (QVAE) ----
    if isinstance(groups, list) and len(groups) > 0:
        slot2q_idx = [None] * K
        q_idx_to_size = {}
        group_name_by_q = {}

        for g in groups:
            q = int(g["q_idx"])
            ids = list(map(int, g["ids"]))
            cb = int(g.get("codebook_size", 0) or 0)
            gname = str(g.get("name", f"q{q}"))

            if cb > 0:
                q_idx_to_size.setdefault(q, cb)
            group_name_by_q.setdefault(q, gname)

            for sid in ids:
                if sid < 0 or sid >= K:
                    raise RuntimeError(f"Bad group ids: {sid} out of range K={K}")
                if slot2q_idx[sid] is not None and int(slot2q_idx[sid]) != q:
                    raise RuntimeError(f"Slot {sid} assigned to multiple q_idx: {slot2q_idx[sid]} vs {q}")
                slot2q_idx[sid] = q

        if any(v is None for v in slot2q_idx):
            bad = [i for i, v in enumerate(slot2q_idx) if v is None]
            raise RuntimeError(f"Some slots missing q_idx mapping in metadata: {bad}")

        default_size = int(meta.get("codebook_size", 0) or 0)
        for q in set(int(x) for x in slot2q_idx):
            q_idx_to_size.setdefault(q, default_size)

        if any(int(v) <= 0 for v in q_idx_to_size.values()):
            raise RuntimeError(f"Some codebook_size is invalid: {q_idx_to_size} (meta_path={meta_path})")

        codebook_sizes = [int(q_idx_to_size[int(slot2q_idx[k])]) for k in range(K)]
        num_groups = len(set(int(x) for x in slot2q_idx))

        meta["slots"] = slots
        meta["K"] = K
        meta["slot2q_idx"] = [int(x) for x in slot2q_idx]
        meta["q_idx_to_size"] = {int(k): int(v) for k, v in q_idx_to_size.items()}
        meta["codebook_sizes"] = [int(x) for x in codebook_sizes]
        meta["num_groups"] = int(num_groups)
        meta["group_name_by_q"] = {int(k): str(v) for k, v in group_name_by_q.items()}
        return meta

    # ---- Old formats fallback ----
    if isinstance(meta.get("codebook_sizes", None), list) and len(meta["codebook_sizes"]) == K:
        codebook_sizes = [int(x) for x in meta["codebook_sizes"]]
        slot2q_idx = list(range(K))
        q_idx_to_size = {int(i): int(cb) for i, cb in enumerate(codebook_sizes)}
        meta["slots"] = slots
        meta["K"] = K
        meta["slot2q_idx"] = slot2q_idx
        meta["q_idx_to_size"] = q_idx_to_size
        meta["codebook_sizes"] = codebook_sizes
        meta["num_groups"] = K
        meta["group_name_by_q"] = {int(i): f"slot_{i}" for i in range(K)}
        return meta

    cb = int(meta.get("codebook_size", 0) or 0)
    if cb <= 0:
        raise RuntimeError(f"Cannot infer codebook size from metadata in {meta_path}. keys={list(meta.keys())}")

    slot2q_idx = [0] * K
    q_idx_to_size = {0: cb}
    codebook_sizes = [cb] * K
    meta["slots"] = slots
    meta["K"] = K
    meta["slot2q_idx"] = slot2q_idx
    meta["q_idx_to_size"] = q_idx_to_size
    meta["codebook_sizes"] = codebook_sizes
    meta["num_groups"] = 1
    meta["group_name_by_q"] = {0: "shared"}
    return meta
def load_dataset_metadata(dataset_root: str, metadata_filename: str = "dataset_metadata.json"):
    """
    Backward-compatible wrapper.
    Returns (meta, slots, K, slot2q_idx, q_idx_to_size)
    """
    meta = load_metadata(dataset_root, metadata_filename=metadata_filename)
    return (
        meta,
        list(meta["slots"]),
        int(meta["K"]),
        list(meta["slot2q_idx"]),
        dict(meta["q_idx_to_size"]),
    )
def parse_code_matrix(entry: dict, K: int):
    import numpy as np

    if "code_matrix" in entry:
        arr = np.asarray(entry["code_matrix"], dtype=np.int64)
    elif "code_seq" in entry:
        # 兼容旧字段名
        arr = np.asarray(entry["code_seq"], dtype=np.int64)
    else:
        raise RuntimeError(f"Entry missing code_matrix/code_seq. keys={list(entry.keys())[:20]}")

    if arr.ndim == 2:
        if arr.shape[1] != K:
            raise RuntimeError(f"code_matrix shape mismatch: got {arr.shape}, expected [:,{K}]")
        return arr

    # 兼容 flat：shape=[T*K]，需要 reshape
    if arr.ndim == 1:
        shape = entry.get("code_matrix_shape", None)
        if shape is not None and len(shape) == 2:
            T, K2 = int(shape[0]), int(shape[1])
            if K2 != K:
                raise RuntimeError(f"code_matrix_shape says K={K2} but metadata K={K}")
            if arr.size != T * K:
                raise RuntimeError(f"flat code size={arr.size} != T*K={T*K}")
            return arr.reshape(T, K)

        T = int(entry.get("frames", 0) or 0)
        if T <= 0:
            raise RuntimeError("flat code_matrix but cannot infer T (missing frames/code_matrix_shape)")
        if arr.size != T * K:
            raise RuntimeError(f"flat code size={arr.size} != T*K={T*K}")
        return arr.reshape(T, K)

    raise RuntimeError(f"Unsupported code_matrix ndim={arr.ndim}")
class SignMotionTokenDataset(torch.utils.data.Dataset):
    """
    读取 extract_code_dataset.py 导出的数据：
      - {split}_dataset_with_gloss.json: list[{"name","text","code_matrix","frames",...}]
      - dataset_metadata.json: {slots, K, groups(q_idx/ids/codebook_size), ...}
      - text_emb: 可选，按 name 找 .pt
    """
    def __init__(
        self,
        dataset_root: str,
        split: str,
        text_emb_dir,
        max_len: int = 256,
        max_text_len: int = None,
        metadata_filename: str = "dataset_metadata.json",
        text_source: str = "text",
        return_global_ids: bool = False,
        global_id_mode: str = "per_slot",
        meta: dict = None,
    ):
        import os, json

        self.dataset_root = dataset_root
        self.split = split
        # text_emb_dir can be:
        #   - str: a single embedding directory
        #   - list/tuple[str]: multiple embedding directories (will be concatenated on seq dim)
        #   - None/"": no text embedding (will fallback to zeros)
        if isinstance(text_emb_dir, (list, tuple)):
            self.text_emb_dirs = [str(x) for x in text_emb_dir if str(x)]
        elif text_emb_dir is None:
            self.text_emb_dirs = []
        else:
            s = str(text_emb_dir)
            self.text_emb_dirs = [s] if s else []

        self.text_source = str(text_source)
        self.max_len = int(max_len)
        self.max_text_len = None if max_text_len is None else int(max_text_len)
        self.return_global_ids = bool(return_global_ids)
        self.global_id_mode = str(global_id_mode)

        # ✅ 单一来源：load_metadata（避免重复解析/读文件）
        if meta is None:
            meta = load_metadata(dataset_root, metadata_filename=metadata_filename)

        self.meta = meta
        self.slots = list(meta["slots"])
        self.K = int(meta["K"])
        self.slot2q_idx = list(meta["slot2q_idx"])
        self.q_idx_to_size = dict(meta["q_idx_to_size"])
        self.codebook_sizes = list(meta["codebook_sizes"])  # len=K, WITHOUT +2

        # per-slot vocab size（同组 slot 会一样）
        self.vocab_size_per_slot = [int(self.codebook_sizes[k]) for k in range(self.K)]

        # 可选：global id offsets（不推荐；你现在做 group tying 不需要）
        if self.return_global_ids:
            self.slot_offsets, self.global_vocab_size = self._build_global_offsets(
                self.slot2q_idx, self.q_idx_to_size, mode=self.global_id_mode
            )
        else:
            self.slot_offsets, self.global_vocab_size = None, None

        # load split json
        split_path = os.path.join(dataset_root, f"{split}_dataset_with_gloss.json")
        with open(split_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not isinstance(self.items, list):
            raise RuntimeError(f"Bad split json: {split_path}")

    def _build_global_offsets(self, slot2q_idx, q_idx_to_size, mode: str):
        # ⚠️ 不推荐：只有你把 token 扁平成单 stream 才有意义
        mode = mode.lower().strip()
        K = len(slot2q_idx)

        if mode == "per_slot":
            # 每个 slot 一段独立词表（会破坏“共享码本权重共享”）
            offsets = []
            cur = 0
            for k in range(K):
                offsets.append(cur)
                cur += int(q_idx_to_size[int(slot2q_idx[k])])
            return offsets, int(cur)

        if mode == "per_group":
            # 同一个 q_idx 的 slot 用同一个 offset（允许共享 embedding，但你仍然不该扁平）
            uniq = sorted(set(int(x) for x in slot2q_idx))
            base = {}
            cur = 0
            for q in uniq:
                base[q] = cur
                cur += int(q_idx_to_size[q])
            offsets = [base[int(slot2q_idx[k])] for k in range(K)]
            return offsets, int(cur)

        raise ValueError(f"Unknown global_id_mode={mode}")

    def __len__(self):
        return len(self.items)

    def _load_text_embedding(self, emb_dir: str, name: str) -> torch.Tensor:
        """Load a precomputed embedding tensor from <emb_dir>/<name>.pt.

        Return shape must be [L, 1024]. If missing or invalid, returns [1,1024] zeros.
        """



        if emb_dir is None:
            return torch.zeros(1, 1024, dtype=torch.float32)

        emb_path = os.path.join(str(emb_dir), f"{name}.pt")
        if not os.path.exists(emb_path):
            return torch.zeros(1, 1024, dtype=torch.float32)

        x = torch.load(emb_path, map_location="cpu")

        if not torch.is_tensor(x):
            return torch.zeros(1, 1024, dtype=torch.float32)
        if x.dim() != 2 or x.shape[1] != 1024:
            return torch.zeros(1, 1024, dtype=torch.float32)
        if x.shape[0] <= 0:
            return torch.zeros(1, 1024, dtype=torch.float32)
        return x.float().contiguous()

    def __getitem__(self, idx: int):


        entry = self.items[idx]
        name = entry.get("name", "")
        gloss = entry.get("gloss", "")
        # 1) tokens: 强制按 metadata 的 slots 顺序读取
        tokens_np = parse_code_matrix(entry, self.K)  # [T, K]
        T = int(tokens_np.shape[0])

        # 2) 截断到 max_len（dataset 这里别乱改列顺序）
        if self.max_len is not None and T > self.max_len:
            tokens_np = tokens_np[: self.max_len]
            T = int(self.max_len)

        tokens_local = torch.from_numpy(tokens_np).long()  # [T, K]

        # 3) 可选：global ids（不推荐默认关）
        if self.return_global_ids:
            off = torch.tensor(self.slot_offsets, dtype=torch.long).view(1, self.K)  # [1,K]
            tokens_global = tokens_local + off
        else:
            tokens_global = None

        # 4) text embedding：支持多路 embedding（按 seq dim 拼接）
        #    - self.text_emb_dirs = [english_emb_dir, gloss_emb_dir, ...]
        if len(self.text_emb_dirs) == 0:
            text_emb = torch.zeros(1, 1024, dtype=torch.float32)
        else:
            parts = [self._load_text_embedding(d, name) for d in self.text_emb_dirs]
            text_emb = torch.cat(parts, dim=0)

        # 可选：截断 text embedding 总长度（防止 L 太长显存爆）
        if self.max_text_len is not None and self.max_text_len > 0 and text_emb.shape[0] > self.max_text_len:
            text_emb = text_emb[: self.max_text_len]

        # 5) 返回：保持你原训练代码的返回格式
        # 推荐训练直接用 tokens_local（无 offset）
        if self.return_global_ids:
            return tokens_global, text_emb, T, name, gloss
        else:
            return tokens_local, text_emb, T, name, gloss
def pad_collate(batch, codebook_sizes):
    tokens_list, text_list, len_list, names, glosses = zip(*batch)
    B = len(tokens_list)
    K = tokens_list[0].shape[1]

    Tmax = max(int(t) for t in len_list)
    # 至少为 1，防止空 batch 报错
    Tmax = max(1, Tmax)
    
    Lmax = max(int(x.shape[0]) for x in text_list)
    Lmax = max(1, Lmax)

    pad_ids = torch.tensor([cb + 1 for cb in codebook_sizes], dtype=torch.long)

    motion = torch.empty((B, Tmax, K), dtype=torch.long)
    motion_pad_mask = torch.ones((B, Tmax), dtype=torch.bool)
    
    # 填充 Pad ID
    motion[:] = pad_ids.unsqueeze(0).unsqueeze(0)

    for i in range(B):
        T = int(len_list[i])
        if T > 0:
            motion[i, :T] = tokens_list[i]
            motion_pad_mask[i, :T] = False

    text = torch.zeros((B, Lmax, 1024), dtype=torch.float32)
    text_pad_mask = torch.ones((B, Lmax), dtype=torch.bool)

    for i in range(B):
        L = text_list[i].shape[0]
        if L > 0:
            text[i, :L] = text_list[i]
            text_pad_mask[i, :L] = False

    lengths = torch.tensor(len_list, dtype=torch.long)
    return motion, text, lengths, motion_pad_mask, text_pad_mask, list(names), list(glosses)

def health_scan(dataset, split_name, n_scan=2000):
    print(f"\n🧪 [HealthScan:{split_name}] Checking data integrity...")
    N = len(dataset)
    n = min(N, n_scan)
    idxs = list(range(N))
    random.shuffle(idxs)
    idxs = idxs[:n]
    
    bad_count = 0
    for idx in idxs:
        try:
            _ = dataset[idx]
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            bad_count += 1
            
    if bad_count == 0:
        print(f"✅ Scanned {n} samples. All good.")
    else:
        print(f"⚠️ Found {bad_count} bad samples in scan.")