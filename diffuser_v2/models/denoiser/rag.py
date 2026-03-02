import csv
import json
import os
import random
import zlib

import numpy as np
import torch
from mGPT.utils.joints_list import SELECTED_JOINT_LANDMARK_INDICES_BODY_ONLY


_SUPPORTED_RAG_TOKEN_FILES = (
    "aslcitizen_qvae_tokens.jsonl",
    "aslcitizen_qvae_tokens.json",
    "aslcitizen_dataset.json",
)
_WLASL_CACHE = {}
_RAG_WEIGHT_LOOKUP_CACHE = {}
_RAG_SIDE_WEIGHT_CACHE = {}

_BODY_WEIGHT_IDS = set(SELECTED_JOINT_LANDMARK_INDICES_BODY_ONLY)
_LEFT_HAND_WEIGHT_IDS = set(list(range(25, 40)) + [66, 67, 68, 69, 70])
_RIGHT_HAND_WEIGHT_IDS = set(list(range(40, 55)) + [71, 72, 73, 74, 75])
_FULL_WEIGHT_IDS = list(SELECTED_JOINT_LANDMARK_INDICES_BODY_ONLY) + list(range(25, 40)) + list(range(40, 55)) + [66, 67, 68, 69, 70, 71, 72, 73, 74, 75]


def _split_csv_arg(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _normalize_slot_name(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_sample_id(value) -> str:
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


def _resolve_path(path: str | None, base_dir: str | None = None) -> str:
    path = str(path or "").strip()
    if not path:
        return ""
    if os.path.isabs(path) or base_dir is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _build_npz_lookup(root_dir: str | None):
    resolved = _resolve_path(root_dir)
    if not resolved:
        return {}
    if resolved in _RAG_WEIGHT_LOOKUP_CACHE:
        return _RAG_WEIGHT_LOOKUP_CACHE[resolved]
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"[RAG] weight dir not found: {resolved}")

    sid2path = {}
    for dirpath, _, filenames in os.walk(resolved):
        for filename in filenames:
            if not filename.lower().endswith(".npz"):
                continue
            sid = _normalize_sample_id(filename)
            if sid and sid not in sid2path:
                sid2path[sid] = os.path.join(dirpath, filename)
    _RAG_WEIGHT_LOOKUP_CACHE[resolved] = sid2path
    print(f"[RAG] indexed {len(sid2path)} weight sidecars from {resolved}")
    return sid2path


def resolve_rag_metadata_path(
    rag_metadata_path: str | None = None,
    rag_wmap_path: str | None = None,
    rag_dataset_root: str | None = None,
    dataset_root: str | None = None,
    meta_name: str = "dataset_metadata.json",
    base_dir: str | None = None,
):
    candidates = []

    meta_path = _resolve_path(rag_metadata_path, base_dir)
    if meta_path:
        candidates.append(meta_path)

    wmap_path = _resolve_path(rag_wmap_path, base_dir)
    if wmap_path:
        if os.path.isdir(wmap_path):
            candidates.append(os.path.join(wmap_path, meta_name))
        else:
            candidates.append(os.path.join(os.path.dirname(wmap_path), meta_name))

    for root in (rag_dataset_root, dataset_root):
        resolved = _resolve_path(root, base_dir)
        if resolved:
            candidates.append(os.path.join(resolved, meta_name))

    if base_dir:
        candidates.append(os.path.join(os.path.abspath(base_dir), meta_name))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand
    return None


def resolve_rag_wmap_source(
    rag_wmap_path: str | None = None,
    rag_dataset_root: str | None = None,
    dataset_root: str | None = None,
    meta_path: str | None = None,
    base_dir: str | None = None,
):
    candidates = []

    wmap_path = _resolve_path(rag_wmap_path, base_dir)
    if wmap_path:
        candidates.append(wmap_path)

    for root in (rag_dataset_root, dataset_root):
        resolved = _resolve_path(root, base_dir)
        if resolved:
            candidates.append(resolved)

    if meta_path:
        candidates.append(os.path.dirname(os.path.abspath(meta_path)))
    if base_dir:
        candidates.append(os.path.abspath(base_dir))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            if os.path.basename(cand) in _SUPPORTED_RAG_TOKEN_FILES:
                return cand
            continue
        if os.path.isdir(cand):
            for filename in _SUPPORTED_RAG_TOKEN_FILES:
                full = os.path.join(cand, filename)
                if os.path.isfile(full):
                    return full
    return None


def _infer_all_codebook_sizes(meta: dict):
    if isinstance(meta.get("codebook_sizes", None), list):
        return [int(x) for x in meta["codebook_sizes"]]

    if "slot2q_idx" not in meta or "groups" not in meta:
        raise ValueError(
            f"[RAG] metadata must contain codebook_sizes OR (slot2q_idx + groups). "
            f"keys={list(meta.keys())}"
        )

    slot2q_idx = meta["slot2q_idx"]
    groups = meta["groups"]
    qidx2size = {}
    for group in groups:
        if isinstance(group, dict) and "q_idx" in group and "codebook_size" in group:
            qidx2size[int(group["q_idx"])] = int(group["codebook_size"])

    codebook_sizes = []
    for idx, q_idx in enumerate(slot2q_idx):
        q_idx = int(q_idx)
        if q_idx not in qidx2size:
            raise ValueError(f"[RAG] q_idx={q_idx} missing in groups for slot {idx}")
        codebook_sizes.append(int(qidx2size[q_idx]))
    return codebook_sizes


def infer_rag_layout(meta: dict, rag_k=None, rag_slot_names=None):
    all_sizes = _infer_all_codebook_sizes(meta)
    meta_k = int(meta.get("K", len(all_sizes)))
    max_k = min(meta_k, len(all_sizes))
    slots = list(meta.get("slots", []))

    requested_slot_names = _split_csv_arg(rag_slot_names)
    if requested_slot_names:
        if not slots:
            raise ValueError("[RAG] rag_slot_names was set but metadata has no 'slots' field")
        name2idx = {_normalize_slot_name(name): idx for idx, name in enumerate(slots)}
        slot_indices = []
        slot_names = []
        for raw_name in requested_slot_names:
            key = _normalize_slot_name(raw_name)
            if key not in name2idx:
                raise ValueError(
                    f"[RAG] slot '{raw_name}' not found in metadata slots={slots}"
                )
            idx = int(name2idx[key])
            slot_indices.append(idx)
            slot_names.append(str(slots[idx]))
    else:
        if rag_k is None:
            use_k = max_k
        else:
            use_k = min(int(rag_k), max_k)
        if use_k <= 0:
            raise ValueError(f"[RAG] invalid rag_K after metadata clamp: {use_k}")
        slot_indices = list(range(use_k))
        slot_names = [
            str(slots[idx]) if idx < len(slots) else f"slot_{idx}"
            for idx in slot_indices
        ]

    codebook_sizes = []
    for idx in slot_indices:
        if idx < 0 or idx >= len(all_sizes):
            raise ValueError(f"[RAG] slot index {idx} out of range for codebook_sizes")
        codebook_sizes.append(int(all_sizes[idx]))

    return {
        "slot_indices": slot_indices,
        "slot_names": slot_names,
        "codebook_sizes": codebook_sizes,
        "rag_k": len(slot_indices),
    }


def preconfigure_rag_opt(opt, base_dir: str | None = None):
    if not bool(getattr(opt, "use_rag", False)):
        return None

    meta_path = resolve_rag_metadata_path(
        rag_metadata_path=getattr(opt, "rag_metadata_path", None),
        rag_wmap_path=getattr(opt, "rag_wmap_path", None),
        rag_dataset_root=getattr(opt, "rag_dataset_root", None),
        dataset_root=getattr(opt, "dataset_root", None),
        meta_name=getattr(opt, "rag_metadata_filename", "dataset_metadata.json"),
        base_dir=base_dir,
    )
    if not meta_path or not os.path.isfile(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    layout = infer_rag_layout(
        meta,
        rag_k=getattr(opt, "rag_K", None),
        rag_slot_names=getattr(opt, "rag_slot_names", ""),
    )

    opt.rag_metadata_path = meta_path
    opt.rag_K = int(layout["rag_k"])
    opt.rag_codebook_sizes = list(layout["codebook_sizes"])
    opt.rag_slot_indices = list(layout["slot_indices"])
    opt.rag_slot_names_resolved = list(layout["slot_names"])

    wmap_source = resolve_rag_wmap_source(
        rag_wmap_path=getattr(opt, "rag_wmap_path", None),
        rag_dataset_root=getattr(opt, "rag_dataset_root", None),
        dataset_root=getattr(opt, "dataset_root", None),
        meta_path=meta_path,
        base_dir=base_dir,
    )
    if wmap_source:
        opt.rag_wmap_path = wmap_source

    return {
        "meta_path": meta_path,
        "wmap_source": wmap_source,
        **layout,
    }


def _normalize_gloss_key(gloss: str) -> str:
    return str(gloss or "").strip().lower()


def _load_gloss_remap(csv_dir_or_path: str | None, source_col: str, target_col: str):
    resolved = _resolve_path(csv_dir_or_path)
    if not resolved:
        return {}

    if os.path.isfile(resolved):
        csv_paths = [resolved]
    elif os.path.isdir(resolved):
        csv_paths = sorted(
            os.path.join(resolved, name)
            for name in os.listdir(resolved)
            if name.lower().endswith(".csv")
        )
    else:
        raise FileNotFoundError(f"[RAG] gloss csv path not found: {resolved}")

    mapping = {}
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            if source_col not in reader.fieldnames or target_col not in reader.fieldnames:
                raise KeyError(
                    f"[RAG] csv {csv_path} must contain '{source_col}' and '{target_col}'. "
                    f"got={reader.fieldnames}"
                )
            for row in reader:
                sid = _normalize_sample_id(row.get(source_col))
                gloss = _normalize_gloss_key(row.get(target_col))
                if sid and gloss:
                    mapping[sid] = gloss
    print(
        f"[RAG] loaded gloss remap {len(mapping)} entries from "
        f"{resolved} ({source_col} -> {target_col})"
    )
    return mapping


def _entry_tokens_to_matrix(entry: dict, default_k: int | None = None):
    tokens = entry.get("tokens", None)
    shape = entry.get("shape", None)
    if tokens is None:
        tokens = entry.get("code_matrix", None)
    if tokens is None:
        return None

    tok_mat = np.asarray(tokens, dtype=np.int64)
    if tok_mat.ndim == 2:
        return tok_mat
    if tok_mat.ndim != 1:
        return None

    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        t_len, k_len = int(shape[0]), int(shape[1])
    elif default_k is not None and int(default_k) > 0 and tok_mat.size % int(default_k) == 0:
        k_len = int(default_k)
        t_len = tok_mat.size // k_len
    else:
        return None

    if tok_mat.size != t_len * k_len:
        return None
    return tok_mat.reshape(t_len, k_len)


def _select_slot_columns(tok_mat: np.ndarray, slot_indices):
    if slot_indices is None:
        return tok_mat
    return tok_mat[:, list(slot_indices)]


def _build_entry(video_id: str, gloss: str, tok_mat: np.ndarray):
    return {
        "video_id": str(video_id or ""),
        "gloss": _normalize_gloss_key(gloss),
        "tokens": tok_mat.astype(np.int64).tolist(),
        "shape": [int(tok_mat.shape[0]), int(tok_mat.shape[1])],
    }


def _slot_group_ids_from_name(slot_name: str):
    key = _normalize_slot_name(slot_name)
    if key == "body":
        return _BODY_WEIGHT_IDS
    if key in {"left_hand", "lhand", "left"}:
        return _LEFT_HAND_WEIGHT_IDS
    if key in {"right_hand", "rhand", "right"}:
        return _RIGHT_HAND_WEIGHT_IDS
    return None


def _prepare_selected_joint_ids(meta_joint_ids, n_joint: int):
    joint_ids = list(meta_joint_ids or [])
    if len(joint_ids) >= n_joint:
        return joint_ids[:n_joint]
    if len(joint_ids) == 0:
        return _FULL_WEIGHT_IDS[:n_joint]
    padded = joint_ids + _FULL_WEIGHT_IDS[len(joint_ids):n_joint]
    return padded[:n_joint]


def _compute_slot_weight_matrix(joint_weight: np.ndarray, selected_joint_ids, slot_names, max_mix: float):
    if joint_weight.ndim != 2:
        raise ValueError(f"joint_weight must be [T,J], got {joint_weight.shape}")
    time_len, n_joint = joint_weight.shape
    selected_joint_ids = _prepare_selected_joint_ids(selected_joint_ids, n_joint)
    slot_weights = []
    max_mix = float(max_mix)
    max_mix = max(0.0, min(1.0, max_mix))

    for slot_name in slot_names:
        group_ids = _slot_group_ids_from_name(slot_name)
        if group_ids is None:
            cols = list(range(n_joint))
        else:
            cols = [idx for idx, joint_id in enumerate(selected_joint_ids) if joint_id in group_ids]
        if len(cols) == 0:
            slot_weights.append(np.ones((time_len,), dtype=np.float32))
            continue
        sub = joint_weight[:, cols].astype(np.float32)
        mean_v = sub.mean(axis=1)
        max_v = sub.max(axis=1)
        mix_v = (1.0 - max_mix) * mean_v + max_mix * max_v
        slot_weights.append(mix_v.astype(np.float32))

    if len(slot_weights) == 0:
        return np.ones((time_len, 0), dtype=np.float32)
    return np.stack(slot_weights, axis=1).astype(np.float32)


def _align_weight_time(slot_weight: np.ndarray, target_t: int):
    if slot_weight.shape[0] == target_t:
        return slot_weight
    if slot_weight.shape[0] <= 0:
        return np.ones((target_t, slot_weight.shape[1]), dtype=np.float32)
    xs = np.arange(slot_weight.shape[0], dtype=np.float32)
    xi = np.linspace(0, slot_weight.shape[0] - 1, int(target_t), dtype=np.float32)
    out = []
    for k in range(slot_weight.shape[1]):
        out.append(np.interp(xi, xs, slot_weight[:, k]).astype(np.float32))
    return np.stack(out, axis=1).astype(np.float32)


def _load_entry_slot_weights(
    entry: dict,
    slot_names,
    expected_t: int,
    weight_key: str = "soft_w",
    weight_max_mix: float = 0.5,
):
    weight_path = str(entry.get("weight_path", "") or "").strip()
    if not weight_path or (not os.path.isfile(weight_path)):
        return None

    slot_names = list(slot_names or [])
    cache_key = (weight_path, tuple(slot_names), str(weight_key), float(weight_max_mix))
    if cache_key in _RAG_SIDE_WEIGHT_CACHE:
        slot_weight = _RAG_SIDE_WEIGHT_CACHE[cache_key]
    else:
        with np.load(weight_path, allow_pickle=False) as data:
            if weight_key not in data:
                raise KeyError(f"[RAG] weight key '{weight_key}' missing in {weight_path}; got={list(data.files)}")
            joint_weight = np.asarray(data[weight_key], dtype=np.float32)
            if joint_weight.ndim == 3 and joint_weight.shape[-1] == 1:
                joint_weight = joint_weight[..., 0]
            if joint_weight.ndim != 2:
                raise ValueError(f"[RAG] expected sidecar weight [T,J] or [T,J,1], got {joint_weight.shape} in {weight_path}")

            meta_joint_ids = []
            if "meta_json" in data:
                try:
                    meta = json.loads(bytes(data["meta_json"].tolist()).decode("utf-8", errors="replace"))
                    meta_joint_ids = list(meta.get("selected_joints", []))
                except Exception:
                    meta_joint_ids = []

        slot_weight = _compute_slot_weight_matrix(
            joint_weight,
            meta_joint_ids,
            slot_names=slot_names,
            max_mix=weight_max_mix,
        )
        _RAG_SIDE_WEIGHT_CACHE[cache_key] = slot_weight

    return _align_weight_time(slot_weight, int(expected_t))


def _load_wlasl_map(
    dataset_root: str,
    rag_meta: dict | None = None,
    slot_indices=None,
    gloss_csv_dir: str | None = None,
    gloss_source_col: str = "Video file",
    gloss_target_col: str = "my_gloss",
    rag_weight_dir: str | None = None,
) -> dict:
    global _WLASL_CACHE

    source_path = resolve_rag_wmap_source(rag_wmap_path=dataset_root)
    if not source_path:
        raise FileNotFoundError(f"[RAG] cannot resolve token source from: {dataset_root}")

    slot_key = tuple(slot_indices) if slot_indices is not None else None
    gloss_csv_resolved = _resolve_path(gloss_csv_dir)
    weight_dir_resolved = _resolve_path(rag_weight_dir)
    cache_key = (
        os.path.abspath(source_path),
        slot_key,
        gloss_csv_resolved,
        weight_dir_resolved,
        str(gloss_source_col),
        str(gloss_target_col),
    )
    if cache_key in _WLASL_CACHE:
        return _WLASL_CACHE[cache_key]

    default_k = None
    if rag_meta is not None and "K" in rag_meta:
        default_k = int(rag_meta["K"])
    elif slot_indices is not None:
        default_k = len(slot_indices)

    gloss_remap = _load_gloss_remap(gloss_csv_resolved, gloss_source_col, gloss_target_col) if gloss_csv_resolved else {}
    weight_lookup = _build_npz_lookup(rag_weight_dir) if str(rag_weight_dir or "").strip() else {}
    source_name = os.path.basename(source_path)
    wmap = {}

    if source_name.endswith(".jsonl"):
        with open(source_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"[RAG] bad json at line {line_no} in {source_path}") from exc

                obj_gloss = _normalize_gloss_key(obj.get("gloss", ""))
                samples = obj.get("samples", [])
                if not obj_gloss or not isinstance(samples, list):
                    continue

                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    sid = _normalize_sample_id(sample.get("id"))
                    gloss = gloss_remap.get(sid, obj_gloss)
                    tok_mat = _entry_tokens_to_matrix(sample, default_k=default_k)
                    if tok_mat is None:
                        continue
                    if slot_indices is not None:
                        tok_mat = _select_slot_columns(tok_mat, slot_indices)
                    item = _build_entry(sid, gloss, tok_mat)
                    if sid in weight_lookup:
                        item["weight_path"] = weight_lookup[sid]
                    wmap.setdefault(gloss, []).append(item)

    elif source_name == "aslcitizen_dataset.json":
        with open(source_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError(f"[RAG] expected list in {source_path}, got {type(data)}")

        for entry in data:
            if not isinstance(entry, dict):
                continue
            sid = _normalize_sample_id(entry.get("source_file") or entry.get("name"))
            gloss = gloss_remap.get(sid, _normalize_gloss_key(entry.get("text", "")))
            if not gloss:
                continue
            tok_mat = _entry_tokens_to_matrix({"tokens": entry.get("code_matrix")}, default_k=default_k)
            if tok_mat is None:
                continue
            if slot_indices is not None:
                tok_mat = _select_slot_columns(tok_mat, slot_indices)
            item = _build_entry(entry.get("name", sid), gloss, tok_mat)
            if sid in weight_lookup:
                item["weight_path"] = weight_lookup[sid]
            wmap.setdefault(gloss, []).append(item)

    else:
        with open(source_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            iterable = []
            for gloss, value in data.items():
                items = value if isinstance(value, list) else [value]
                for item in items:
                    iterable.append((gloss, item))
        elif isinstance(data, list):
            iterable = [(_normalize_gloss_key(item.get("gloss", "")), item) for item in data]
        else:
            raise TypeError(f"[RAG] unexpected json type in {source_path}: {type(data)}")

        for gloss, entry in iterable:
            if not isinstance(entry, dict):
                continue
            sid = _normalize_sample_id(entry.get("video_id") or entry.get("id") or entry.get("name"))
            final_gloss = gloss_remap.get(sid, _normalize_gloss_key(gloss))
            if not final_gloss:
                continue
            tok_mat = _entry_tokens_to_matrix(entry, default_k=default_k)
            if tok_mat is None:
                continue
            if slot_indices is not None:
                tok_mat = _select_slot_columns(tok_mat, slot_indices)
            item = _build_entry(sid, final_gloss, tok_mat)
            if sid in weight_lookup:
                item["weight_path"] = weight_lookup[sid]
            wmap.setdefault(final_gloss, []).append(item)

    _WLASL_CACHE[cache_key] = wmap
    print(f"[RAG] loaded {len(wmap)} gloss keys from {source_path}")
    return wmap


def _normalize_gloss_sentence(gloss_sentence: str) -> list[str]:
    if gloss_sentence is None:
        return []
    sentence = str(gloss_sentence).strip()
    if not sentence:
        return []

    output = []
    for raw in sentence.split():
        token = raw.strip()
        if not token:
            continue

        upper = token.upper()
        if upper in ("FS_BEGIN", "FS_END"):
            continue
        if upper.startswith("L_") and len(token) >= 3:
            token = token[2:]
        token = token.strip().lower()
        if token:
            output.append(token)
    return output


def _lookup_wlasl_entry(word: str, wmap: dict, rng: "random.Random|None" = None):
    if not word:
        return None

    candidates = [word]
    if "-" in word:
        candidates.append(word.replace("-", " "))
    if "_" in word:
        candidates.append(word.replace("_", " "))

    seen = set()
    for cand in candidates:
        cand = cand.strip().lower()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if cand in wmap and isinstance(wmap[cand], list) and len(wmap[cand]) > 0:
            items = wmap[cand]
            if rng is None or len(items) == 1:
                return items[0]
            return items[rng.randrange(len(items))]
    return None


def build_blueprint_batch(
    glosses: list,
    wmap: dict,
    pad_token_ids: torch.Tensor,
    device,
    K: int = 13,
    max_words: int = 64,
    per_word_max_T: int = 48,
    total_max_T: int = 384,
    names: list | None = None,
    epoch: int = 0,
    rng: "random.Random|None" = None,
    mode: str = "train",
    seed: int | None = None,
    frame_subsample: int = 0,
    slot_names=None,
    weight_key: str = "soft_w",
    weight_max_mix: float = 0.5,
):
    if glosses is None:
        glosses = []
    batch_size = len(glosses)

    if torch.is_tensor(pad_token_ids):
        pad_ids = pad_token_ids.detach().cpu().long().tolist()
    else:
        pad_ids = list(pad_token_ids)
    assert len(pad_ids) == K, f"pad_token_ids must have length K={K}, got {len(pad_ids)}"
    # Special ids per slot:
    #   normal token: [0 .. cb-1]
    #   unk token:    cb      (derived from pad_id - 1)
    #   pad token:    cb + 1  (pad_ids)
    unk_ids = [int(pid) - 1 for pid in pad_ids]

    if rng is None:
        base = 0
        if names is not None:
            for name in names:
                text = "" if name is None else str(name)
                for ch in text[:64]:
                    base = (base * 131 + ord(ch)) & 0xFFFFFFFF
        base = (base + int(epoch) * 10007) & 0xFFFFFFFF
        if seed is not None:
            base = (base + int(seed)) & 0xFFFFFFFF
        rng = random.Random(base)

    seq_list = []
    len_list = []
    weight_list = []
    total_words = 0
    hit_words = 0
    unk_words = 0
    frame_subsample = int(frame_subsample or 0)

    for gloss in glosses:
        tokens = _normalize_gloss_sentence(gloss)
        if max_words is not None and max_words > 0:
            tokens = tokens[: int(max_words)]
        total_words += len(tokens)

        frames = []
        frame_weights = []
        for word in tokens:
            entries = wmap.get(word, None)
            if not entries:
                frames.append(np.asarray(unk_ids, dtype=np.int64).reshape(1, K))
                frame_weights.append(np.ones((1, K), dtype=np.float32))
                unk_words += 1
                continue

            entry = rng.choice(entries)
            tok_mat = np.asarray(entry.get("tokens", None), dtype=np.int64)
            if tok_mat.size == 0:
                continue
            if tok_mat.ndim == 1:
                if tok_mat.shape[0] % K != 0:
                    continue
                tok_mat = tok_mat.reshape(tok_mat.shape[0] // K, K)
            if tok_mat.ndim != 2 or tok_mat.shape[1] != K or tok_mat.shape[0] <= 0:
                continue
            slot_weight = _load_entry_slot_weights(
                entry,
                slot_names=slot_names or [f"slot_{i}" for i in range(K)],
                expected_t=tok_mat.shape[0],
                weight_key=weight_key,
                weight_max_mix=weight_max_mix,
            )
            if slot_weight is None:
                slot_weight = np.ones((tok_mat.shape[0], K), dtype=np.float32)

            if frame_subsample > 0:
                tok_mat = tok_mat[::frame_subsample]
                slot_weight = slot_weight[::frame_subsample]
                if tok_mat.shape[0] <= 0:
                    continue
                if per_word_max_T is not None and int(per_word_max_T) > 1:
                    tok_mat = tok_mat[: int(per_word_max_T)]
                    slot_weight = slot_weight[: int(per_word_max_T)]
                frames.append(tok_mat)
                frame_weights.append(slot_weight.astype(np.float32))
            else:
                mid = tok_mat.shape[0] // 2
                frames.append(tok_mat[mid:mid + 1])
                frame_weights.append(slot_weight[mid:mid + 1].astype(np.float32))

            hit_words += 1

        if len(frames) == 0:
            seq = np.asarray(pad_ids, dtype=np.int64).reshape(1, K)
            weight_seq = np.ones((1, K), dtype=np.float32)
            valid_len = 0
        else:
            seq = np.concatenate(frames, axis=0).astype(np.int64)
            weight_seq = np.concatenate(frame_weights, axis=0).astype(np.float32)
            valid_len = int(seq.shape[0])

        if total_max_T is not None and total_max_T > 0:
            seq = seq[: int(total_max_T)]
            weight_seq = weight_seq[: int(total_max_T)]
            valid_len = min(valid_len, int(total_max_T))

        seq_list.append(seq)
        len_list.append(valid_len)
        weight_list.append(weight_seq)

    tb = int(max(len_list)) if len_list else 0
    tb = max(1, tb)

    pad_vec = torch.tensor(pad_ids, device=device, dtype=torch.long).view(1, 1, K)
    bp_tokens = pad_vec.expand(batch_size, tb, K).clone()
    bp_pad_mask = torch.ones((batch_size, tb), device=device, dtype=torch.bool)
    bp_weights = torch.ones((batch_size, tb, K), device=device, dtype=torch.float32)

    for idx, (seq, valid_len, weight_seq) in enumerate(zip(seq_list, len_list, weight_list)):
        if valid_len > 0:
            token_tensor = torch.from_numpy(seq[:valid_len]).to(device=device, dtype=torch.long)
            bp_tokens[idx, :valid_len] = token_tensor
            bp_pad_mask[idx, :valid_len] = False
            weight_tensor = torch.from_numpy(weight_seq[:valid_len]).to(device=device, dtype=torch.float32)
            bp_weights[idx, :valid_len] = weight_tensor

    stats = {
        "hit_rate": float(hit_words) / float(max(1, total_words)),
        "total_words": int(total_words),
        "hit_words": int(hit_words),
        "unk_words": int(unk_words),
        "Tb": int(tb),
        "frame_subsample": int(frame_subsample),
        "mode": str(mode),
    }
    return bp_tokens, bp_pad_mask, bp_weights, stats
