import os
import json
import torch
import numpy as np
import random
import zlib
# =========================
# RAG Blueprint (WLASL) helpers
# =========================

_WLASL_CACHE = {"root": None, "map": None}
def _load_wlasl_map(dataset_root: str) -> dict:
    """
    Load WLASL token dictionary from:
      <dataset_root>/wlasl_qvae_tokens.jsonl   (preferred)
      fallback: <dataset_root>/wlasl_qvae_tokens.json

    JSONL line format example:
      {"gloss":"1 DOLLAR","samples":[{"id":"...","tokens":[...]} , ...]}

    Returns:
      wmap: dict[str, list[entry]]
        entry: {"video_id":..., "gloss":..., "tokens":[...], "shape":[T,13]}
    """
    global _WLASL_CACHE
    dataset_root = str(dataset_root)

    if _WLASL_CACHE["map"] is not None and _WLASL_CACHE["root"] == dataset_root:
        return _WLASL_CACHE["map"]

    jsonl_path = os.path.join(dataset_root, "aslcitizen_qvae_tokens.jsonl")
    json_path  = os.path.join(dataset_root, "aslcitizen_qvae_tokens.json")

    wmap = {}

    # ---- preferred: JSONL (streaming, no giant single-line json) ----
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"[WLASL] bad json at line {ln} in {jsonl_path}") from e

                gloss = str(obj.get("gloss", "")).strip().lower()
                if not gloss:
                    continue

                samples = obj.get("samples", [])
                if not isinstance(samples, list) or len(samples) == 0:
                    continue

                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    vid = s.get("id", None)
                    toks = s.get("tokens", None)
                    if toks is None:
                        continue
                    # toks should be flat [T*K]
                    if not isinstance(toks, list) or len(toks) == 0:
                        continue

                    K = 13
                    if (len(toks) % K) != 0:
                        # 不符合 T*13 的，直接跳过（避免后面 reshape 崩）
                        continue
                    T = len(toks) // K

                    entry = {
                        "video_id": str(vid) if vid is not None else "",
                        "gloss": gloss,
                        "tokens": toks,
                        "shape": [int(T), int(K)],
                    }
                    wmap.setdefault(gloss, []).append(entry)

        _WLASL_CACHE["root"] = dataset_root
        _WLASL_CACHE["map"] = wmap
        print(f"[WLASL] loaded {len(wmap)} gloss keys from {jsonl_path}")
        return wmap

    # ---- fallback: JSON ----
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[WLASL] not found: {jsonl_path} or {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k, v in data.items():
            g = str(k).strip().lower()
            if not g:
                continue
            if isinstance(v, list):
                wmap[g] = v
            else:
                wmap[g] = [v]
    elif isinstance(data, list):
        for e in data:
            g = str(e.get("gloss", "")).strip().lower()
            if not g:
                continue
            wmap.setdefault(g, []).append(e)
    else:
        raise TypeError(f"[WLASL] unexpected json type: {type(data)}")

    _WLASL_CACHE["root"] = dataset_root
    _WLASL_CACHE["map"] = wmap
    print(f"[WLASL] loaded {len(wmap)} gloss keys from {json_path}")
    return wmap


def _normalize_gloss_sentence(gloss_sentence: str) -> list[str]:
    """
    Input example:
      "AND I DISCUSS EARLIER FS_BEGIN L_P L_I ... FS_END PART COME-FROM ..."

    Rules:
      - skip FS_BEGIN / FS_END
      - if token starts with "L_" => drop "L_" and keep the letter
      - lowercase everything
    """
    if gloss_sentence is None:
        return []
    s = str(gloss_sentence).strip()
    if not s:
        return []

    out = []
    for raw in s.split():
        tok = raw.strip()
        if not tok:
            continue

        up = tok.upper()
        if up in ("FS_BEGIN", "FS_END"):
            continue

        if up.startswith("L_") and len(tok) >= 3:
            tok = tok[2:]  # "L_P" -> "P"
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(tok)

    return out

def _lookup_wlasl_entry(word: str, wmap: dict, rng: "random.Random|None" = None):
    """
    Try exact match first, then mild variants.
    If multiple samples exist for the same gloss, optionally random-sample one for augmentation.
    """
    if not word:
        return None

    candidates = [word]
    # common variants
    if "-" in word:
        candidates.append(word.replace("-", " "))
    if "_" in word:
        candidates.append(word.replace("_", " "))

    seen = set()
    for c in candidates:
        c = c.strip().lower()
        if not c or c in seen:
            continue
        seen.add(c)

        if c in wmap and isinstance(wmap[c], list) and len(wmap[c]) > 0:
            lst = wmap[c]
            if rng is None or len(lst) == 1:
                return lst[0]
            # ✅ 随机抽一个 sample（但 rng 可控 -> 可复现）
            return lst[rng.randrange(len(lst))]

    return None


def _entry_tokens_to_matrix(entry: dict, K_expected: int = 13):
    """
    entry["tokens"]: flat list length = T*K
    entry["shape"]: [T,13]
    return: np.ndarray [T,K]
    """
    shape = entry.get("shape", None)
    flat = entry.get("tokens", None)
    if shape is None or flat is None:
        return None

    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        T, K = int(shape[0]), int(shape[1])
    else:
        return None

    if K_expected is not None and int(K) != int(K_expected):
        return None

    flat = np.asarray(flat, dtype=np.int64)
    if flat.size != T * K:
        return None

    return flat.reshape(T, K)
def build_blueprint_batch(
    glosses: list,
    wmap: dict,
    pad_token_ids: torch.Tensor,   # [K]
    device,
    K: int = 13,
    max_words: int = 64,
    per_word_max_T: int = 48,
    total_max_T: int = 384,
    # 可复现抽样增强
    names: list | None = None,
    epoch: int = 0,
    rng: "random.Random|None" = None,
    mode: str = "train",          # "train"|"eval"|"infer"
    seed: int | None = None,      # 额外扰动（可选）
):
    """
    V3 (simple):
      - 每个 gloss 单词 -> 随机抽一个 motion token 序列
      - 只取该序列的“中间帧”(50%) 的 13 个 token (K=13)
      - 所以每个单词最多贡献 1 帧，Tb <= max_words
      - 这样显著降低 Tb，提升速度并降低噪声

    输出：
      bp_tokens:   [B, Tb, K] (long)
      bp_pad_mask: [B, Tb]    (bool, True=pad)
      stats: dict(hit_rate, total_words, hit_words, Tb)
    """
    import numpy as np
    import torch
    import random

    if glosses is None:
        glosses = []
    B = len(glosses)

    # pad_token_ids -> [K] list[int]
    if torch.is_tensor(pad_token_ids):
        pad_ids = pad_token_ids.detach().cpu().long().tolist()
    else:
        pad_ids = list(pad_token_ids)
    assert len(pad_ids) == K, f"pad_token_ids must have length K={K}, got {len(pad_ids)}"

    # RNG：尽量可复现（与原逻辑兼容）
    if rng is None:
        base = 0
        if names is not None:
            # 轻量 hash，避免 python hash 随机性
            for n in names:
                s = "" if n is None else str(n)
                for ch in s[:64]:
                    base = (base * 131 + ord(ch)) & 0xFFFFFFFF
        base = (base + int(epoch) * 10007) & 0xFFFFFFFF
        if seed is not None:
            base = (base + int(seed)) & 0xFFFFFFFF
        rng = random.Random(base)

    seq_list = []
    len_list = []

    total_words = 0
    hit_words = 0

    # 逐样本构建：每个词 -> 1 帧 (K token)
    for i, g in enumerate(glosses):
        toks = _normalize_gloss_sentence(g)
        if max_words is not None and max_words > 0:
            toks = toks[: int(max_words)]
        total_words += len(toks)

        frames = []
        for w in toks:
            entries = wmap.get(w, None)
            if not entries:
                continue
            entry = rng.choice(entries)

            tok_mat = np.asarray(entry.get("tokens", None))
            if tok_mat.size == 0:
                continue

            # tok_mat: [T, K] 或 [T*K]
            if tok_mat.ndim == 1:
                if tok_mat.shape[0] % K != 0:
                    continue
                tok_mat = tok_mat.reshape(tok_mat.shape[0] // K, K)
            if tok_mat.ndim != 2 or tok_mat.shape[1] != K:
                continue
            if tok_mat.shape[0] <= 0:
                continue

            mid = tok_mat.shape[0] // 2
            frames.append(tok_mat[mid:mid + 1])  # [1,K]
            hit_words += 1

        if len(frames) == 0:
            # 至少保留 1 个 pad 帧，len=0 表示全 pad
            seq = np.asarray(pad_ids, dtype=np.int64).reshape(1, K)
            L = 0
        else:
            seq = np.concatenate(frames, axis=0).astype(np.int64)  # [Tb_i,K], Tb_i <= max_words
            L = int(seq.shape[0])

        # total_max_T / per_word_max_T 在 V3 里基本无意义，但保持接口安全：
        if total_max_T is not None and total_max_T > 0:
            seq = seq[: int(total_max_T)]
            L = min(L, int(total_max_T))

        seq_list.append(seq)
        len_list.append(L)

    Tb = int(max(len_list)) if len_list else 0
    Tb = max(1, Tb)  # 保证张量非空

    # init outputs (pad by default)
    pad_vec = torch.tensor(pad_ids, device=device, dtype=torch.long).view(1, 1, K)
    bp_tokens = pad_vec.expand(B, Tb, K).clone()
    bp_pad_mask = torch.ones((B, Tb), device=device, dtype=torch.bool)  # True=pad

    for i, (seq, L) in enumerate(zip(seq_list, len_list)):
        if L > 0:
            t = torch.from_numpy(seq[:L]).to(device=device, dtype=torch.long)
            bp_tokens[i, :L] = t
            bp_pad_mask[i, :L] = False

    hit_rate = float(hit_words) / float(max(1, total_words))

    stats = {
        "hit_rate": hit_rate,
        "total_words": int(total_words),
        "hit_words": int(hit_words),
        "Tb": int(Tb),
    }
    return bp_tokens, bp_pad_mask, stats
