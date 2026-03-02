import os
import sys
import json
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

try:
    import yaml
except Exception:
    yaml = None

from models.denoiser.model_patched import Denoiser

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SOKE_ROOT not in sys.path:
    sys.path.append(_SOKE_ROOT)

from vae_adapter import (
    decode_latent_to_pose3d,
    infer_latent_shape_from_vae,
    load_vae_model,
    prepare_vae_opt,
)


def _require_cuda_device() -> torch.device:
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        raise RuntimeError(
            "Diffusion inference requires CUDA, but torch.cuda.is_available() is False. "
            f"torch={torch.__version__} built_cuda={torch.version.cuda} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    return torch.device("cuda")

# -------------------------
# config loader (txt/yaml)
# -------------------------
def _convert_string_to_type(s: str):
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.startswith("[") and s.endswith("]"):
        items = s[1:-1].split(",")
        return [_convert_string_to_type(item) for item in items if item.strip() != ""]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def load_config(path: str) -> SimpleNamespace:
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("pyyaml is required to load .yaml/.yml configs")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cleaned = {}
        for k, v in cfg.items():
            if isinstance(v, dict) and "value" in v:
                cleaned[k] = v["value"]
            else:
                cleaned[k] = v
        return SimpleNamespace(**cleaned)

    if path.endswith(".txt"):
        cfg = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "---" in line:
                    continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = _convert_string_to_type(parts[1])
                    cfg[key] = val
        return SimpleNamespace(**cfg)

    raise ValueError(f"Unsupported config format: {path}")


def _infer_sep(path: str) -> str:
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
                if s.count("\t") >= s.count(",") and s.count("\t") > 0:
                    return "\t"
                return ","
    except Exception:
        pass
    return ","


def _col_lookup(df: pd.DataFrame, candidates):
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"Missing column. Need one of {candidates}. got={list(df.columns)}")


def _col_lookup_optional(df: pd.DataFrame, candidates):
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None


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


def _resolve_checkpoint_layout(checkpoint_dir: str):
    cdir = os.path.abspath(checkpoint_dir)
    den_opt = os.path.join(cdir, "opt.txt")
    den_ckpt = os.path.join(cdir, "model", "latest.tar")
    if not os.path.isfile(den_opt):
        raise FileNotFoundError(f"Checkpoint opt not found: {den_opt}")
    if not os.path.isfile(den_ckpt):
        raise FileNotFoundError(f"Checkpoint latest.tar not found: {den_ckpt}")
    return den_opt, den_ckpt


def _default_vae_paths_from_denoiser_opt(den_opt):
    vae_dir = str(getattr(den_opt, "vae_path", "") or "").strip()
    if not vae_dir:
        return None, None
    vae_opt = os.path.join(vae_dir, "opt.txt")
    vae_ckpt = os.path.join(vae_dir, "model", "latest.tar")
    if not os.path.isfile(vae_opt):
        vae_opt = None
    if not os.path.isfile(vae_ckpt):
        vae_ckpt = None
    return vae_opt, vae_ckpt


def _build_sid_to_npz(data_dir: str):
    sid2npz = {}
    for name in os.listdir(data_dir):
        if not name.lower().endswith(".npz"):
            continue
        sid = _normalize_sample_id(name)
        if sid and sid not in sid2npz:
            sid2npz[sid] = os.path.join(data_dir, name)
    return sid2npz


def _load_split_rows(csv_path: str, split_type: str = "auto"):
    sep = _infer_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    sentence_id_candidates = ["SENTENCE_NAME", "sentence_name", "name", "sample_id", "SAMPLE_ID"]
    word_id_candidates = ["Video file", "video file", "video_file", "VIDEO_FILE", "item_name", "ITEM_NAME"]
    split_type = str(split_type or "auto").strip().lower()
    if split_type not in {"auto", "word", "sentence"}:
        raise ValueError(f"split_type must be one of ['auto','word','sentence'], got {split_type}")

    if split_type == "word":
        id_candidates = word_id_candidates + sentence_id_candidates
    elif split_type == "sentence":
        id_candidates = sentence_id_candidates + word_id_candidates
    else:
        # auto:先句子后单词（不破坏 how2sign），找不到再落到单词列
        id_candidates = sentence_id_candidates + word_id_candidates

    col_name = _col_lookup(df, id_candidates)
    col_gloss = _col_lookup(df, ["GLOSS", "gloss", "PSEUDO_GLOSS", "pseudo_gloss"])
    rows = []
    for _, row in df.iterrows():
        sid = _normalize_sample_id(row[col_name])
        if not sid:
            continue
        gloss = "" if pd.isna(row[col_gloss]) else str(row[col_gloss])
        rows.append({"sample_id": sid, "gloss": gloss})
    print(f"[CSV] split_type={split_type} id_col='{col_name}' gloss_col='{col_gloss}' rows={len(rows)}")
    return rows


def _resolve_num_frames_from_npz(npz_path: str) -> int:
    with np.load(npz_path, mmap_mode="r") as data:
        if "poses" in data:
            return int(data["poses"].shape[0])
        if "joints_xyz" in data:
            return int(data["joints_xyz"].shape[0])
    raise KeyError(f"No 'poses' or 'joints_xyz' in {npz_path}")


# -------------------------
# Gloss normalization (keep fingerspelling spans)
# -------------------------
FS_BEGIN_SET = {"fs_begin", "fsbegin", "fs-start", "fs_start", "fsbegin:", "<fs_begin>", "[fs_begin]"}
FS_END_SET   = {"fs_end", "fsend", "fs-stop", "fs_stop", "fsend:", "<fs_end>", "[fs_end]"}

def normalize_gloss_for_tokens(gloss: str) -> str:
    """
    - Keep tokens inside FS_BEGIN..FS_END (do NOT delete)
    - Normalize FS markers to: FS_BEGIN / FS_END
    - Otherwise keep tokens as-is (conservative)
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
        out.append(t)
    return " ".join(out)


def load_vae(vae_opt_path: str, vae_ckpt_path: str, device: torch.device):
    vae_opt = load_config(vae_opt_path)
    if not hasattr(vae_opt, "data_format"):
        vae_opt.data_format = "motion_dataset_rod3_fixed_length"
    vae_opt = prepare_vae_opt(vae_opt, device=device)
    return load_vae_model(vae_opt, vae_ckpt_path)


# -------------------------
# denoiser loader
# -------------------------
def _resolve_rag_metadata_path(opt, base_dir: str):
    from models.denoiser.rag import resolve_rag_metadata_path as _resolve_meta

    return _resolve_meta(
        rag_metadata_path=getattr(opt, "rag_metadata_path", None),
        rag_wmap_path=getattr(opt, "rag_wmap_path", None),
        rag_dataset_root=getattr(opt, "rag_dataset_root", None),
        dataset_root=getattr(opt, "dataset_root", None) or getattr(opt, "vae_path", None),
        meta_name=getattr(opt, "rag_metadata_filename", "dataset_metadata.json"),
        base_dir=base_dir,
    )


def _resolve_rag_wmap_root(opt, base_dir: str):
    from models.denoiser.rag import resolve_rag_wmap_source as _resolve_source

    return _resolve_source(
        rag_wmap_path=getattr(opt, "rag_wmap_path", None),
        rag_dataset_root=getattr(opt, "rag_dataset_root", None),
        dataset_root=getattr(opt, "dataset_root", None) or getattr(opt, "vae_path", None),
        meta_path=getattr(opt, "rag_metadata_path", None),
        base_dir=base_dir,
    )


def _infer_rag_codebook_sizes(meta: dict, rag_k: int):
    if isinstance(meta.get("codebook_sizes", None), list):
        sizes = [int(x) for x in meta["codebook_sizes"]]
        if len(sizes) < rag_k:
            raise ValueError(f"[RAG] codebook_sizes length={len(sizes)} < rag_K={rag_k}")
        return sizes[:rag_k]

    if "slot2q_idx" not in meta or "groups" not in meta:
        raise ValueError("[RAG] metadata must contain codebook_sizes OR (slot2q_idx + groups)")

    slot2q_idx = meta["slot2q_idx"]
    groups = meta["groups"]
    if len(slot2q_idx) < rag_k:
        raise ValueError(f"[RAG] slot2q_idx length={len(slot2q_idx)} < rag_K={rag_k}")

    qidx2size = {}
    for g in groups:
        if isinstance(g, dict) and "q_idx" in g and "codebook_size" in g:
            qidx2size[int(g["q_idx"])] = int(g["codebook_size"])

    sizes = []
    for k in range(rag_k):
        q = int(slot2q_idx[k])
        if q not in qidx2size:
            raise ValueError(f"[RAG] q_idx={q} missing in metadata groups")
        sizes.append(int(qidx2size[q]))
    return sizes


def _load_rag_resources(opt, denoiser_opt_path: str, device: torch.device, strict_rag: bool = True):
    if not bool(getattr(opt, "use_rag", False)):
        return None

    base_dir = os.path.dirname(os.path.abspath(denoiser_opt_path))
    try:
        from models.denoiser.rag import (
            _load_wlasl_map,
            build_blueprint_batch,
            infer_rag_layout,
        )
    except Exception as exc:
        if strict_rag:
            raise
        print(f"[RAG][WARN] Failed to import rag helpers: {exc}")
        return None

    meta_path = _resolve_rag_metadata_path(opt, base_dir)
    if not meta_path or (not os.path.isfile(meta_path)):
        msg = f"[RAG] metadata not found. opt.rag_metadata_path={getattr(opt, 'rag_metadata_path', None)}"
        if strict_rag:
            raise FileNotFoundError(msg)
        print(f"[RAG][WARN] {msg}")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rag_layout = infer_rag_layout(
        meta,
        rag_k=getattr(opt, "rag_K", None),
        rag_slot_names=getattr(opt, "rag_slot_names", ""),
    )
    rag_k = int(rag_layout["rag_k"])
    codebook_sizes = list(rag_layout["codebook_sizes"])
    pad_token_ids = torch.tensor([int(cb) + 1 for cb in codebook_sizes], device=device, dtype=torch.long)

    wmap_source = _resolve_rag_wmap_root(opt, base_dir)
    if not wmap_source:
        msg = f"[RAG] cannot resolve token source from rag_wmap_path={getattr(opt, 'rag_wmap_path', None)}"
        if strict_rag:
            raise FileNotFoundError(msg)
        print(f"[RAG][WARN] {msg}")
        return None

    wmap = _load_wlasl_map(
        wmap_source,
        rag_meta=meta,
        slot_indices=rag_layout["slot_indices"],
        gloss_csv_dir=getattr(opt, "rag_gloss_csv_dir", ""),
        gloss_source_col=getattr(opt, "rag_gloss_source_col", "Video file"),
        gloss_target_col=getattr(opt, "rag_gloss_target_col", "my_gloss"),
        rag_weight_dir=getattr(opt, "rag_weight_dir", ""),
    )
    print(f"[RAG] ready: meta={meta_path} source={wmap_source} K={rag_k} slots={rag_layout['slot_names']}")
    return {
        "build_blueprint_batch": build_blueprint_batch,
        "wmap": wmap,
        "pad_token_ids": pad_token_ids,
        "rag_k": rag_k,
        "codebook_sizes": codebook_sizes,
        "slot_names": list(rag_layout["slot_names"]),
    }


def load_denoiser(
    denoiser_opt_path: str,
    denoiser_ckpt_path: str,
    vae_latent_dim: int,
    device: torch.device,
    strict_rag: bool = True,
):
    opt = load_config(denoiser_opt_path)
    opt.device = device

    # ===== V3 defaults =====
    opt.use_precomputed_text_emb = False
    opt.use_gloss_tokens = True
    opt.gloss_embed_mode = "vocab"
    opt.use_cond_film = bool(getattr(opt, "use_cond_film", True))
    opt.gloss_layers = int(getattr(opt, "gloss_layers", 0))
    opt.gloss_heads  = int(getattr(opt, "gloss_heads", 8))
    opt.use_rag = bool(getattr(opt, "use_rag", True))

    # try to discover vocab path if missing in old opt.txt
    vocab_path = str(getattr(opt, "gloss_vocab_path", "") or "").strip()
    if not vocab_path or (not os.path.isfile(vocab_path)):
        base_dir = os.path.dirname(os.path.abspath(denoiser_opt_path))
        candidates = [
            os.path.join(base_dir, "gloss_vocab.json"),
            os.path.join(base_dir, "gloss_vocab_v3.json"),
            os.path.join(base_dir, "gloss_vocab_how2sign_dictionary.json"),
            os.path.join(base_dir, "vocab", "gloss_vocab.json"),
            os.path.join(base_dir, "model", "gloss_vocab_v3.json"),
            os.path.join(base_dir, "model", "gloss_vocab.json"),
            os.path.join(base_dir, "model", "gloss_vocab_how2sign_dictionary.json"),
            "gloss_vocab.json",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                vocab_path = cand
                break
    if not vocab_path or (not os.path.isfile(vocab_path)):
        raise FileNotFoundError(
            "V3 inference requires gloss vocab json. Set gloss_vocab_path in opt/config."
        )
    opt.gloss_vocab_path = vocab_path

    print(f"[DenoiserOpt] use_precomputed_text_emb={opt.use_precomputed_text_emb}, "
          f"use_gloss_tokens={opt.use_gloss_tokens}, gloss_embed_mode={opt.gloss_embed_mode}, "
          f"gloss_layers={opt.gloss_layers}, use_cond_film={opt.use_cond_film}")

    if bool(getattr(opt, "use_rag", False)):
        try:
            from models.denoiser.rag import preconfigure_rag_opt
            preconfigure_rag_opt(opt, base_dir=os.path.dirname(os.path.abspath(denoiser_opt_path)))
        except Exception:
            if strict_rag:
                raise

    denoiser = Denoiser(opt, vae_latent_dim).to(device)
    denoiser.eval()

    rag_resources = None
    if bool(getattr(opt, "use_rag", False)):
        rag_resources = _load_rag_resources(opt, denoiser_opt_path, device=device, strict_rag=strict_rag)
        if rag_resources is not None:
            setattr(denoiser, "_rag_codebook_sizes", rag_resources["codebook_sizes"])
            if hasattr(denoiser, "_maybe_init_rag"):
                denoiser._maybe_init_rag(device)

    ckpt = torch.load(denoiser_ckpt_path, map_location="cpu")
    state = ckpt["denoiser"] if (isinstance(ckpt, dict) and "denoiser" in ckpt) else ckpt

    missing, unexpected = denoiser.load_state_dict(state, strict=False)

    # ===== 关键：这里不允许 gloss_* 还出现在 unexpected 里 =====
    bad_unexpected = [k for k in unexpected if k.startswith("gloss_") or k.startswith("cond_") or "cond_gate" in k]
    if len(bad_unexpected) > 0:
        print("[FATAL] Denoiser structure mismatch. These keys exist in ckpt but not in model:")
        for k in bad_unexpected[:50]:
            print("  -", k)
        raise RuntimeError(
            "Your inference Denoiser does NOT include gloss/cond modules. "
            "Make sure you replaced models/denoiser/model.py with the patched version "
            "and opt.gloss_layers matches the checkpoint."
        )

    # Allow v2->v3 compatibility keys.
    allow_unexpected_prefixes = ["clip_model.", "word_emb.", "_cache_"]
    if bool(getattr(opt, "use_rag", False)) and getattr(denoiser, "rag_encoder", None) is None:
        allow_unexpected_prefixes.append("rag_encoder.")
    unexpected = [
        k for k in unexpected
        if not any(k.startswith(p) for p in allow_unexpected_prefixes)
    ]
    missing = [
        k for k in missing
        if not (k.startswith("clip_model.") or k.startswith("word_emb.") or "_cache_" in k)
    ]

    if len(unexpected) > 0:
        print("[Warn] Unexpected keys (non-clip):", unexpected[:20])
    if len(missing) > 0:
        print("[Warn] Missing keys (non-clip):", missing[:20])

    return denoiser, opt, rag_resources



# -------------------------
# decode + expand joints + save AMASS
# -------------------------
def expand_selected_poses_to_full_smplx(
    selected_poses: np.ndarray,
    selected_joint_indices: list,
    full_joint_count: int = 55,
    fill_mode: str = "zero",
    fill_full_poses: np.ndarray = None,
) -> np.ndarray:
    if selected_poses.ndim != 3 or selected_poses.shape[-1] != 3:
        raise ValueError(f"selected_poses must be [T, Js, 3], got {selected_poses.shape}")
    T, Js, _ = selected_poses.shape
    if len(selected_joint_indices) != Js:
        raise ValueError(f"len(selected_joint_indices) must equal Js ({Js}), got {len(selected_joint_indices)}")
    if fill_mode not in ("zero", "ref"):
        raise ValueError(f"fill_mode must be 'zero' or 'ref', got {fill_mode}")

    if fill_mode == "ref":
        if fill_full_poses is None:
            raise ValueError("fill_mode='ref' requires fill_full_poses")
        if fill_full_poses.shape != (T, full_joint_count, 3):
            raise ValueError(f"fill_full_poses must be [T,{full_joint_count},3], got {fill_full_poses.shape}")
        full_poses = fill_full_poses.astype(np.float32, copy=True)
    else:
        full_poses = np.zeros((T, full_joint_count, 3), dtype=np.float32)

    for i, orig_idx in enumerate(selected_joint_indices):
        if orig_idx < 0 or orig_idx >= full_joint_count:
            raise ValueError(f"orig_idx out of range: {orig_idx} (full_joint_count={full_joint_count})")
        full_poses[:, orig_idx, :] = selected_poses[:, i, :].astype(np.float32, copy=False)

    return full_poses

def save_amass_npz(
    output_path: str,
    poses_full: np.ndarray,
    framerate: int = 30,
    gender: str = "neutral",
    surface_model_type: str = "smplx",
):
    if poses_full.ndim != 3 or poses_full.shape[-1] != 3:
        raise ValueError(f"poses_full must be [T, J, 3], got {poses_full.shape}")
    frame_count = int(poses_full.shape[0])

    amass_data = {
        "gender": gender,
        "surface_model_type": surface_model_type,
        "mocap_framerate": int(framerate),
        "mocap_time_length": float(frame_count) / float(framerate),
        "trans": np.zeros((frame_count, 3), dtype=np.float32),
        "poses": poses_full.astype(np.float32, copy=False),
        "betas": np.zeros(10, dtype=np.float32),
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, **amass_data)

    print(f"✅ Saved AMASS npz: {output_path}")
    print("poses:", amass_data["poses"].shape, amass_data["poses"].dtype)

@torch.no_grad()
def vae_decode_to_raw(vae, z: torch.Tensor):
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(vae, "mean") and hasattr(vae, "std"):
        out = out * vae.std + vae.mean
    return out


# -------------------------
# sampling (pure diffusion from noise)
# -------------------------
@torch.no_grad()
def sample_from_noise_gloss_only(
    denoiser,
    den_opt,
    scheduler,
    latents_shape,
    gloss_text: str,
    cfg_scale: float = 3.0,
    seed: int = 123,
    rag_resources=None,
    sample_name: str = "",
):
    """
    Pure diffusion: x_T ~ N(0, I), denoise to 0 with optional CFG.
    text input format for your patched denoiser:
        List[[sentence, gloss]]  with sentence == ""
    """
    device = next(denoiser.parameters()).device
    B = int(latents_shape[0])
    Tz = int(latents_shape[1])

    if B != 1:
        raise ValueError(f"This helper expects B=1, got latents_shape={latents_shape}")

    len_mask = torch.ones((1, Tz), device=device, dtype=torch.bool)

    n_steps = int(getattr(den_opt, "num_inference_timesteps", 50))
    scheduler.set_timesteps(n_steps)
    timesteps = scheduler.timesteps.to(device)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    x = torch.randn(latents_shape, device=device, generator=g) * float(getattr(scheduler, "init_noise_sigma", 1.0))

    gloss_text = normalize_gloss_for_tokens(gloss_text)

    # build conditional & unconditional text batches
    text_cond = [["", gloss_text]]
    text_uncond = [["", ""]]   # empty gloss

    bp_tokens_cond, bp_pad_mask_cond, bp_weights_cond = None, None, None
    bp_tokens_cfg, bp_pad_mask_cfg, bp_weights_cfg = None, None, None
    if rag_resources is not None and bool(getattr(den_opt, "use_rag", False)):
        build_blueprint_batch = rag_resources["build_blueprint_batch"]
        rag_k = int(rag_resources["rag_k"])
        pad_token_ids = rag_resources["pad_token_ids"]
        common_kwargs = dict(
            wmap=rag_resources["wmap"],
            pad_token_ids=pad_token_ids,
            device=device,
            K=rag_k,
            max_words=int(getattr(den_opt, "rag_max_words", 64)),
            per_word_max_T=int(getattr(den_opt, "rag_per_word_max_T", 1)),
            total_max_T=int(getattr(den_opt, "rag_total_max_T", 384)),
            frame_subsample=int(getattr(den_opt, "rag_frame_subsample", 0)),
            slot_names=rag_resources.get("slot_names"),
            weight_key=str(getattr(den_opt, "rag_weight_key", "soft_w")),
            weight_max_mix=float(getattr(den_opt, "rag_weight_max_mix", 0.5)),
            epoch=0,
            mode="infer",
        )
        bp_tokens_cond, bp_pad_mask_cond, bp_weights_cond, bp_stats = build_blueprint_batch(
            glosses=[gloss_text],
            names=[sample_name],
            **common_kwargs,
        )
        print(
            f"[RAG][infer] sample={sample_name or '<none>'} hit_rate={bp_stats['hit_rate']:.3f} "
            f"Tb={bp_stats['Tb']} (hit={bp_stats['hit_words']}/{bp_stats['total_words']})"
        )
        if cfg_scale is not None and float(cfg_scale) > 1.0:
            bp_tokens_cfg, bp_pad_mask_cfg, bp_weights_cfg, _ = build_blueprint_batch(
                glosses=["", gloss_text],
                names=["", sample_name],
                **common_kwargs,
            )

    for t in timesteps:
        if cfg_scale is not None and float(cfg_scale) > 1.0:
            # duplicate batch (2B)
            x_in = torch.cat([x, x], dim=0)
            len_mask_in = torch.cat([len_mask, len_mask], dim=0)
            text_in = text_uncond + text_cond  # length 2

            pred, _ = denoiser.forward(
                x_in,
                t,
                text_in,
                len_mask=len_mask_in,
                need_attn=False,
                use_cached_clip=False,
                blueprint_tokens=bp_tokens_cfg,
                blueprint_weights=bp_weights_cfg,
                blueprint_pad_mask=bp_pad_mask_cfg,
            )
            pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
        else:
            pred, _ = denoiser.forward(
                x,
                t,
                text_cond,
                len_mask=len_mask,
                need_attn=False,
                use_cached_clip=False,
                blueprint_tokens=bp_tokens_cond,
                blueprint_weights=bp_weights_cond,
                blueprint_pad_mask=bp_pad_mask_cond,
            )

        x = scheduler.step(pred, t, x).prev_sample

    return x


def _build_scheduler(den_opt):
    from diffusers import DDIMScheduler

    prediction_type = str(getattr(den_opt, "prediction_type", "epsilon"))
    scheduler_pred = prediction_type if prediction_type != "mesh" else "v_prediction"
    return DDIMScheduler(
        num_train_timesteps=int(getattr(den_opt, "num_train_timesteps", 1000)),
        beta_start=float(getattr(den_opt, "beta_start", 0.0001)),
        beta_end=float(getattr(den_opt, "beta_end", 0.02)),
        beta_schedule=str(getattr(den_opt, "beta_schedule", "linear")),
        prediction_type=str(scheduler_pred),
        clip_sample=False,
    )


def _infer_latent_shape(vae, num_frames: int, device: torch.device):
    return infer_latent_shape_from_vae(vae, num_frames=int(num_frames), device=device)


def _decode_and_expand_to_full(vae, z_hat: torch.Tensor, target_frames: int):
    poses_sel = decode_latent_to_pose3d(vae, z_hat)[0].detach().cpu().numpy()

    target = int(target_frames)
    t_len = poses_sel.shape[0]
    if t_len < target:
        pad = np.repeat(poses_sel[-1:, :, :], target - t_len, axis=0)
        poses_sel = np.concatenate([poses_sel, pad], axis=0)
    else:
        poses_sel = poses_sel[:target]

    from mGPT.utils.joints_list import SELECTED_JOINT_INDICES

    if poses_sel.shape[1] == 55:
        return poses_sel.astype(np.float32, copy=False)
    if poses_sel.shape[1] == len(SELECTED_JOINT_INDICES):
        return expand_selected_poses_to_full_smplx(
            selected_poses=poses_sel,
            selected_joint_indices=SELECTED_JOINT_INDICES,
            full_joint_count=55,
            fill_mode="zero",
        )
    raise RuntimeError(f"Decoded joints={poses_sel.shape[1]}, expected 55 or {len(SELECTED_JOINT_INDICES)}")


def _resolve_gloss_from_csv(csv_path: str, sample_name: str, split_type: str = "auto"):
    rows = _load_split_rows(csv_path, split_type=split_type)
    for row in rows:
        if row["sample_id"] == _normalize_sample_id(sample_name):
            return row["gloss"]
    raise RuntimeError(f"sample_name not found in csv: {sample_name}")


def _run_one_sample(
    vae,
    denoiser,
    den_opt,
    rag_resources,
    scheduler,
    sample_id: str,
    gloss_text: str,
    num_frames: int,
    cfg_scale: float,
    seed: int,
    framerate: int,
    out_npz: str,
):
    gloss_text = normalize_gloss_for_tokens(gloss_text)
    latents_shape = _infer_latent_shape(vae, num_frames=int(num_frames), device=next(denoiser.parameters()).device)
    z_hat = sample_from_noise_gloss_only(
        denoiser=denoiser,
        den_opt=den_opt,
        scheduler=scheduler,
        latents_shape=latents_shape,
        gloss_text=gloss_text,
        cfg_scale=float(cfg_scale),
        seed=int(seed),
        rag_resources=rag_resources,
        sample_name=sample_id,
    )
    poses_full = _decode_and_expand_to_full(vae, z_hat, target_frames=int(num_frames))
    save_amass_npz(
        output_path=out_npz,
        poses_full=poses_full,
        framerate=int(framerate),
        gender="neutral",
        surface_model_type="smplx",
    )
    return {
        "sample_id": sample_id,
        "num_frames": int(num_frames),
        "out_npz": out_npz,
    }


def _run_checkpoint_random_batch(
    checkpoint_dir: str,
    shared_rows,
    device: torch.device,
    args,
):
    denoiser_opt_path, denoiser_ckpt_path = _resolve_checkpoint_layout(checkpoint_dir)
    den_cfg = load_config(denoiser_opt_path)

    vae_opt_path = args.vae_opt
    vae_ckpt_path = args.vae_ckpt
    if (not vae_opt_path) or (not vae_ckpt_path):
        d_vae_opt, d_vae_ckpt = _default_vae_paths_from_denoiser_opt(den_cfg)
        vae_opt_path = vae_opt_path or d_vae_opt
        vae_ckpt_path = vae_ckpt_path or d_vae_ckpt
    if not vae_opt_path or not vae_ckpt_path:
        raise RuntimeError("Cannot resolve VAE paths. Provide --vae_opt and --vae_ckpt explicitly.")

    vae = load_vae(vae_opt_path, vae_ckpt_path, device)
    first_frames = int(shared_rows[0]["num_frames"])
    vae_latent_dim = int(_infer_latent_shape(vae, num_frames=first_frames, device=device)[-1])

    denoiser, den_opt, rag_resources = load_denoiser(
        denoiser_opt_path,
        denoiser_ckpt_path,
        vae_latent_dim,
        device,
        strict_rag=bool(args.strict_rag),
    )
    den_opt.num_inference_timesteps = int(args.num_infer_steps)
    scheduler = _build_scheduler(den_opt)

    ckpt_name = os.path.basename(os.path.abspath(checkpoint_dir.rstrip("/")))
    ckpt_out_dir = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(ckpt_out_dir, exist_ok=True)

    outputs = []
    for row in shared_rows:
        sid = row["sample_id"]
        out_npz = os.path.join(ckpt_out_dir, f"{sid}.npz")
        out_item = _run_one_sample(
            vae=vae,
            denoiser=denoiser,
            den_opt=den_opt,
            rag_resources=rag_resources,
            scheduler=scheduler,
            sample_id=sid,
            gloss_text=row["gloss"],
            num_frames=int(row["num_frames"]),
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed),
            framerate=int(args.framerate),
            out_npz=out_npz,
        )
        outputs.append(out_item)
    return ckpt_name, outputs


def main():
    parser = argparse.ArgumentParser("Gloss-only diffusion inference -> VAE decode -> AMASS npz")

    # legacy direct paths
    parser.add_argument("--denoiser_opt", type=str, default=None)
    parser.add_argument("--denoiser_ckpt", type=str, default=None)
    parser.add_argument("--vae_opt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)

    # checkpoint-dir mode
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint folder containing opt.txt and model/latest.tar")
    parser.add_argument("--checkpoint_dirs", nargs="+", default=None, help="multiple checkpoint folders for side-by-side generation")
    parser.add_argument("--strict_rag", action="store_true", help="fail fast when use_rag=True but rag resources are missing")
    parser.add_argument("--allow_missing_rag", dest="strict_rag", action="store_false", help="warn instead of fail when rag resources are missing")
    parser.set_defaults(strict_rag=True)

    # sample source
    parser.add_argument("--gloss", type=str, default=None, help="gloss sentence used for conditioning")
    parser.add_argument("--csv", type=str, default=None, help="csv/tsv with columns SENTENCE_NAME,SENTENCE,GLOSS")
    parser.add_argument("--sample_name", type=str, default=None, help="SENTENCE_NAME to lookup gloss in --csv")
    parser.add_argument("--split_csv", type=str, default=None, help="split csv for random batch mode")
    parser.add_argument("--split_type", type=str, default="auto", choices=["auto", "word", "sentence"], help="CSV schema type for id column mapping")
    parser.add_argument("--data_dir", type=str, default=None, help="npz data dir for random batch mode (for GT length)")

    # inference settings
    parser.add_argument("--cfg_scale", default=7.5, type=float)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--num_infer_steps", default=50, type=int)
    parser.add_argument("--num_frames", type=int, default=None, help="target frames; required in legacy mode unless --use_gt_length")
    parser.add_argument("--use_gt_length", action="store_true", help="use GT npz length for each sample")
    parser.add_argument("--num_random_samples", type=int, default=0, help="randomly select N samples from split_csv")

    # outputs
    parser.add_argument("--framerate", default=30, type=int)
    parser.add_argument("--out_npz", type=str, default=None, help="legacy single output npz")
    parser.add_argument("--output_dir", type=str, default=None, help="batch mode output root")

    args = parser.parse_args()
    device = _require_cuda_device()
    print("Device:", device)

    ckpt_dirs = []
    if args.checkpoint_dirs:
        ckpt_dirs.extend(args.checkpoint_dirs)
    if args.checkpoint_dir:
        ckpt_dirs.append(args.checkpoint_dir)

    # -------------------------
    # batch random mode (new)
    # -------------------------
    if ckpt_dirs:
        if not args.output_dir:
            raise RuntimeError("checkpoint-dir mode requires --output_dir")

        seed_rng = random.Random(int(args.seed))
        first_den_opt_path, _ = _resolve_checkpoint_layout(ckpt_dirs[0])
        first_opt = load_config(first_den_opt_path)

        split_csv = args.split_csv or str(getattr(first_opt, "val_csv_path", "") or "")
        data_dir = args.data_dir or str(getattr(first_opt, "val_data_dir", "") or "")
        if not split_csv or not os.path.isfile(split_csv):
            raise FileNotFoundError(f"split_csv not found: {split_csv}")
        if not data_dir or (not os.path.isdir(data_dir)):
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        rows = _load_split_rows(split_csv, split_type=args.split_type)
        sid2npz = _build_sid_to_npz(data_dir)
        rows = [r for r in rows if r["sample_id"] in sid2npz]
        if not rows:
            raise RuntimeError(f"No csv rows matched npz files in {data_dir}")

        if args.num_random_samples <= 0:
            raise RuntimeError("checkpoint-dir mode requires --num_random_samples > 0")
        seed_rng.shuffle(rows)
        chosen = rows[: int(args.num_random_samples)]
        if len(chosen) < int(args.num_random_samples):
            print(f"[WARN] requested {args.num_random_samples}, only {len(chosen)} matched samples.")

        shared_rows = []
        for idx, row in enumerate(chosen):
            sid = row["sample_id"]
            npz_path = sid2npz[sid]
            if args.use_gt_length:
                num_frames = _resolve_num_frames_from_npz(npz_path)
            else:
                if args.num_frames is None:
                    raise RuntimeError("batch mode requires --num_frames when --use_gt_length is not set")
                num_frames = int(args.num_frames)
            shared_rows.append({
                "sample_id": sid,
                "gloss": row["gloss"],
                "num_frames": int(num_frames),
                "npz_path": npz_path,
                "index": idx,
            })

        all_outputs = {}
        for ckpt in ckpt_dirs:
            ckpt_name, outputs = _run_checkpoint_random_batch(ckpt, shared_rows, device=device, args=args)
            all_outputs[ckpt_name] = outputs

        meta = {
            "mode": "checkpoint_random_batch",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": int(args.seed),
            "cfg_scale": float(args.cfg_scale),
            "num_infer_steps": int(args.num_infer_steps),
            "framerate": int(args.framerate),
            "split_csv": split_csv,
            "data_dir": data_dir,
            "shared_samples": shared_rows,
            "outputs": all_outputs,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        meta_path = os.path.join(args.output_dir, "run_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved run metadata: {meta_path}")
        return

    # -------------------------
    # legacy single-sample mode
    # -------------------------
    if not args.denoiser_opt or not args.denoiser_ckpt:
        raise RuntimeError("Legacy mode requires --denoiser_opt and --denoiser_ckpt")

    gloss_text = args.gloss
    sample_name = args.sample_name or "single_sample"
    if gloss_text is None:
        if args.csv is None or args.sample_name is None:
            raise RuntimeError("Provide either --gloss, OR (--csv and --sample_name).")
        gloss_text = _resolve_gloss_from_csv(args.csv, args.sample_name, split_type=args.split_type)
    gloss_text = normalize_gloss_for_tokens(gloss_text)
    print("Gloss:", gloss_text)

    den_cfg_for_vae = load_config(args.denoiser_opt)
    vae_opt_path = args.vae_opt
    vae_ckpt_path = args.vae_ckpt
    if (not vae_opt_path) or (not vae_ckpt_path):
        d_vae_opt, d_vae_ckpt = _default_vae_paths_from_denoiser_opt(den_cfg_for_vae)
        vae_opt_path = vae_opt_path or d_vae_opt
        vae_ckpt_path = vae_ckpt_path or d_vae_ckpt
    if not vae_opt_path or not vae_ckpt_path:
        raise RuntimeError("Cannot resolve VAE paths. Provide --vae_opt and --vae_ckpt explicitly.")
    if not args.out_npz:
        raise RuntimeError("Legacy mode requires --out_npz")

    if args.use_gt_length:
        if not args.data_dir or not args.sample_name:
            raise RuntimeError("--use_gt_length in legacy mode requires --data_dir and --sample_name")
        sid2npz = _build_sid_to_npz(args.data_dir)
        sid = _normalize_sample_id(args.sample_name)
        if sid not in sid2npz:
            raise RuntimeError(f"sample_name {args.sample_name} not found in data_dir")
        num_frames = _resolve_num_frames_from_npz(sid2npz[sid])
    else:
        if args.num_frames is None:
            raise RuntimeError("Legacy mode requires --num_frames (or set --use_gt_length)")
        num_frames = int(args.num_frames)

    vae = load_vae(vae_opt_path, vae_ckpt_path, device)
    vae_latent_dim = int(_infer_latent_shape(vae, num_frames=num_frames, device=device)[-1])
    denoiser, den_opt, rag_resources = load_denoiser(
        args.denoiser_opt,
        args.denoiser_ckpt,
        vae_latent_dim,
        device,
        strict_rag=bool(args.strict_rag),
    )
    den_opt.num_inference_timesteps = int(args.num_infer_steps)
    scheduler = _build_scheduler(den_opt)

    _run_one_sample(
        vae=vae,
        denoiser=denoiser,
        den_opt=den_opt,
        rag_resources=rag_resources,
        scheduler=scheduler,
        sample_id=sample_name,
        gloss_text=gloss_text,
        num_frames=num_frames,
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed),
        framerate=int(args.framerate),
        out_npz=args.out_npz,
    )


if __name__ == "__main__":
    main()
