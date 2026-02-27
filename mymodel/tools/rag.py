import os
import json
import torch
import numpy as np

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


def _lookup_wlasl_entry(word: str, wmap: dict):
    """
    Try exact match first, then mild variants.
    """
    if not word:
        return None

    candidates = [word]
    # common variants
    if "-" in word:
        candidates.append(word.replace("-", " "))
    if "_" in word:
        candidates.append(word.replace("_", " "))
    # keep only unique
    seen = set()
    for c in candidates:
        c = c.strip().lower()
        if not c or c in seen:
            continue
        seen.add(c)
        if c in wmap and len(wmap[c]) > 0:
            # pick the first (you can later random-sample for augmentation)
            return wmap[c][0]
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
):
    """
    For each sample gloss sentence:
      - tokenize/normalize
      - lookup each word in WLASL dict
      - concat their motion token matrices along time: [sumTi, K]
      - pad to batch max (clipped by total_max_T)

    Returns:
      bp: [B, Tb, K] long
      bp_pad_mask: [B, Tb] bool (True=PAD)
      stats: dict with hit rate info
    """
    B = len(glosses)
    pad_ids = pad_token_ids.to(device=device, dtype=torch.long)  # [K]

    seqs = []
    hit_words = 0
    total_words = 0

    for g in glosses:
        words = _normalize_gloss_sentence(g)
        if max_words is not None and max_words > 0:
            words = words[: int(max_words)]

        mats = []
        for w in words:
            total_words += 1
            e = _lookup_wlasl_entry(w, wmap)
            if e is None:
                continue
            mat = _entry_tokens_to_matrix(e, K_expected=K)
            if mat is None:
                continue

            # clip per-word length to keep memory stable
            if per_word_max_T is not None and per_word_max_T > 0:
                mat = mat[: int(per_word_max_T)]

            mats.append(mat)
            hit_words += 1

        if len(mats) == 0:
            seq = np.zeros((0, K), dtype=np.int64)
        else:
            seq = np.concatenate(mats, axis=0)  # [Tsum, K]

        # clip total length
        if total_max_T is not None and total_max_T > 0:
            seq = seq[: int(total_max_T)]

        seqs.append(torch.from_numpy(seq).long().to(device))

    Tb = max([s.shape[0] for s in seqs] + [0])
    Tb = int(min(Tb, total_max_T)) if (total_max_T is not None and total_max_T > 0) else int(Tb)

    if Tb <= 0:
        bp = pad_ids.view(1, 1, K).expand(B, 1, K).clone()
        bp_pad_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
    else:
        bp = pad_ids.view(1, 1, K).expand(B, Tb, K).clone()
        bp_pad_mask = torch.ones((B, Tb), dtype=torch.bool, device=device)

        for i, s in enumerate(seqs):
            Li = int(min(s.shape[0], Tb))
            if Li > 0:
                bp[i, :Li] = s[:Li]
                bp_pad_mask[i, :Li] = False

    stats = {
        "hit_words": int(hit_words),
        "total_words": int(total_words),
        "hit_rate": float(hit_words / max(1, total_words)),
        "Tb": int(Tb),
    }
    return bp, bp_pad_mask, stats
