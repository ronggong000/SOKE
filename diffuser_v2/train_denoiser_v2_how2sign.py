import copy
import inspect
import json
import os
import random
import math
import sys
import time
from os.path import join as pjoin

import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from diffusers import DDIMScheduler

from models.denoiser.model_patched import Denoiser
from models.denoiser.trainer_patched import DenoiserTrainer
from latent_cache_dataset import CachedLatentDataset, latent_cache_collate_fn
from options.denoiser_option_v2 import arg_parse
from sign_diffusion_dataset_patched import SignDiffusionDataset, diffusion_collate_fn, normalize_gloss_for_tokens
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from mGPT.utils.joints_list import (
    SMPLX_JOINT_LANDMARK_NAMES,
    SELECTED_JOINT_INDICES,
    SELECTED_JOINT_LANDMARK_INDICES,
    SELECTED_JOINT_LANDMARK_BODY_EVAL,
    SELECTED_JOINT_LANDMARK_LHAND_EVAL,
    SELECTED_JOINT_LANDMARK_RHAND_EVAL,
    SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,
    SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,
    SELECTED_JOINT_INDICES_BODY_ONLY,
    SELECTED_JOINT_INDICES_NEIGHBOR_LIST,
)
from mGPT.utils.smplx_vertex_group import UPPER_BODY_VERTEX, LEFT_HAND_VERTEX, RIGHT_HAND_VERTEX

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_DIR", pjoin(_THIS_DIR, "wandb"))
if _SOKE_ROOT not in sys.path:
    sys.path.append(_SOKE_ROOT)
sys.path.append(os.path.join(_SOKE_ROOT, "mymodel", "vae"))
from qvae_model_rod3_fixed_length import VAE as MyVAE

try:
    import wandb
except Exception:
    class _WandbStub:
        @staticmethod
        def init(*args, **kwargs):
            print("[WARN] wandb is not installed. Continuing without wandb logging.")
            return _WandbStub()

        @staticmethod
        def log(*args, **kwargs):
            return

        @staticmethod
        def finish(*args, **kwargs):
            return

    wandb = _WandbStub()

_HOW2SIGN_ROOT = "/fs04/scratch2/ar85/singyu/how2sign"
_SOKE_DATA_ROOT = os.path.join(_SOKE_ROOT, "data")

_HOW2SIGN_TRAIN_DATA_DIR = os.path.join(_HOW2SIGN_ROOT, "align_denoised_front")
_HOW2SIGN_VAL_DATA_DIR = os.path.join(_HOW2SIGN_ROOT, "align_denoised_front_val")
_HOW2SIGN_TEST_DATA_DIR = os.path.join(_HOW2SIGN_ROOT, "align_denoised_front_test")
_HOW2SIGN_TRAIN_CSV = os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_train.csv")
_HOW2SIGN_VAL_CSV = os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_val.csv")
_HOW2SIGN_TEST_CSV = os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_test.csv")
_HOW2SIGN_DICT_JSON = os.path.join(_SOKE_DATA_ROOT, "TEXT_TO_GLOSS_DICTIONARY.json")


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_size=200, drop_last=False, shuffle=True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

    def __iter__(self):
        indices = self.sorted_indices[:]
        if self.shuffle:
            buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
            random.shuffle(buckets)
            indices = []
            for b in buckets:
                random.shuffle(b)
                indices.extend(b)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


class LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max(1, int(max_batches))
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


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


def _iter_dictionary_entries(obj):
    if isinstance(obj, list):
        for item in obj:
            yield item
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            if isinstance(v, str):
                yield v
            elif isinstance(v, list):
                for vv in v:
                    yield vv
        return
    return


def _clean_dict_token(token_obj):
    tok = "" if token_obj is None else str(token_obj).strip()
    if len(tok) >= 2 and ((tok[0] == '"' and tok[-1] == '"') or (tok[0] == "'" and tok[-1] == "'")):
        tok = tok[1:-1].strip()
    return tok


def build_gloss_vocab_from_dictionary(opt):
    if not os.path.isfile(_HOW2SIGN_DICT_JSON):
        raise FileNotFoundError(f"how2sign dictionary not found: {_HOW2SIGN_DICT_JSON}")

    with open(_HOW2SIGN_DICT_JSON, "r", encoding="utf-8-sig") as f:
        raw_obj = json.load(f)

    tokens = set()
    for item in _iter_dictionary_entries(raw_obj):
        tok = _clean_dict_token(item)
        if not tok:
            continue
        norm = normalize_gloss_for_tokens(tok)
        for t in norm.split():
            tt = t.strip()
            if tt:
                tokens.add(tt)

    if not tokens:
        raise ValueError(f"No tokens loaded from dictionary: {_HOW2SIGN_DICT_JSON}")

    stoi = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok in sorted(tokens):
        if tok not in stoi:
            stoi[tok] = len(stoi)

    out_path = pjoin(opt.model_dir, "gloss_vocab_how2sign_dictionary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi}, f, ensure_ascii=False)

    opt.gloss_vocab_path = out_path
    print(f"[HOW2SIGN] built gloss vocab from dictionary: size={len(stoi)} -> {out_path}")
    return out_path


def report_unknown_token_coverage(csv_path: str, vocab_stoi: dict):
    sep = _infer_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    col_gloss = _col_lookup(df, ["GLOSS", "gloss", "Gloss", "PSEUDO_GLOSS", "pseudo_gloss"])

    known = 0
    unk = 0
    for g in df[col_gloss].fillna("").astype(str).tolist():
        norm = normalize_gloss_for_tokens(g)
        for tok in norm.split():
            if tok in vocab_stoi:
                known += 1
            else:
                unk += 1

    total = max(1, known + unk)
    print(
        "[HOW2SIGN] train token coverage by dictionary vocab: "
        f"known={known} unk={unk} unk_ratio={unk / total:.4f}"
    )


def force_how2sign_paths(opt):
    opt.train_data_dir = _HOW2SIGN_TRAIN_DATA_DIR
    opt.val_data_dir = _HOW2SIGN_VAL_DATA_DIR
    opt.test_data_dir = _HOW2SIGN_TEST_DATA_DIR
    opt.train_csv_path = _HOW2SIGN_TRAIN_CSV
    opt.val_csv_path = _HOW2SIGN_VAL_CSV
    opt.test_csv_path = _HOW2SIGN_TEST_CSV

    opt.train_only_gloss = True
    opt.gloss_use_positional = True
    opt.enable_length_cond = True

    missing = []
    for p in [
        opt.train_data_dir,
        opt.val_data_dir,
        opt.test_data_dir,
        opt.train_csv_path,
        opt.val_csv_path,
        opt.test_csv_path,
        _HOW2SIGN_DICT_JSON,
    ]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError("Missing required how2sign paths:\n" + "\n".join(missing))

    print("[HOW2SIGN] hardcoded paths active:")
    print(f"  train_data_dir={opt.train_data_dir}")
    print(f"  val_data_dir={opt.val_data_dir}")
    print(f"  train_csv_path={opt.train_csv_path}")
    print(f"  val_csv_path={opt.val_csv_path}")
    print(f"  test_csv_path={opt.test_csv_path}")


def save_runtime_opt(opt):
    args = vars(opt)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, "opt.txt")
    with open(file_name, "wt", encoding="utf-8") as opt_file:
        opt_file.write("------------ Options -------------\n")
        for k, v in sorted(args.items()):
            opt_file.write("%s: %s\n" % (str(k), str(v)))
        opt_file.write("-------------- End ----------------\n")
    print(f"[HOW2SIGN] runtime options saved to: {file_name}")


def load_and_freeze_vae(opt):
    if hasattr(opt, "vae_path") and opt.vae_path:
        vae_dir = opt.vae_path
    else:
        vae_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name)

    opt_path = pjoin(vae_dir, "opt.txt")
    print(f"Loading VAE config from: {opt_path}")
    vae_opt = get_opt(opt_path, opt.device)
    vae_opt.SMPLX_JOINT_LANDMARK_NAMES = SMPLX_JOINT_LANDMARK_NAMES
    vae_opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES = SELECTED_JOINT_LANDMARK_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_BODY_EVAL = SELECTED_JOINT_LANDMARK_BODY_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_LHAND_EVAL = SELECTED_JOINT_LANDMARK_LHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_RHAND_EVAL = SELECTED_JOINT_LANDMARK_RHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX = SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX
    vae_opt.SELECTED_JOINT_INDICES_BODY_ONLY = SELECTED_JOINT_INDICES_BODY_ONLY
    vae_opt.UPPER_BODY_VERTEX = UPPER_BODY_VERTEX
    vae_opt.LEFT_HAND_VERTEX = LEFT_HAND_VERTEX
    vae_opt.RIGHT_HAND_VERTEX = RIGHT_HAND_VERTEX
    vae_opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    vae_opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    vae_opt.joints_num = len(SELECTED_JOINT_INDICES)
    model = MyVAE(vae_opt)

    ckpt_path = pjoin(vae_dir, "model", "latest.tar")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "vae" in ckpt:
        model.load_state_dict(ckpt["vae"])
    else:
        model.load_state_dict(ckpt)
    model.freeze()
    model.to(opt.device)
    return model


def validate_v2_contract(opt, denoiser):
    warnings = []

    named_params = [n for n, _ in denoiser.named_parameters()]
    if any(n.startswith("clip_model.") for n in named_params):
        warnings.append("Detected clip_model.* params; V2/V3 path should not create CLIP modules.")
    if any(n.startswith("word_emb.") for n in named_params):
        warnings.append("Detected word_emb.* params; V2/V3 path should not create word_emb modules.")

    if not bool(getattr(denoiser, "use_cond_film", False)):
        warnings.append("FiLM conditioning is disabled (use_cond_film=False).")
    if float(getattr(opt, "mismatch_text_weight", 0.0)) > 0 and not bool(getattr(opt, "use_rag", False)):
        warnings.append("mismatch_text_weight>0 but use_rag=False; mismatch loss cannot include RAG blueprint.")

    try:
        from models.denoiser import trainer_patched as trainer_mod

        src = inspect.getsource(trainer_mod.DenoiserTrainer.train_forward)
        if "bp_tokens_bad" not in src or "blueprint_tokens=bp_tokens_bad" not in src:
            warnings.append("Mismatch branch does not appear to rebuild/use bad RAG blueprint.")
    except Exception as exc:
        warnings.append(f"Could not introspect trainer mismatch path: {exc}")

    try:
        from models.denoiser import rag as rag_mod

        rag_src = inspect.getsource(rag_mod.build_blueprint_batch)
        if "mid = tok_mat.shape[0] // 2" not in rag_src:
            warnings.append("RAG midpoint-per-word rule not detected in build_blueprint_batch.")
    except Exception as exc:
        warnings.append(f"Could not introspect RAG midpoint rule: {exc}")

    if warnings:
        print("[V3-CHECK][WARN] Feature contract warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[V3-CHECK] Feature contract checks passed (no CLIP/word_emb + FiLM + mismatch/RAG midpoint).")


def validate_dataset_pair_mode(dataset, split_name: str):
    if len(dataset) == 0:
        print(f"[V3-CHECK][WARN] {split_name} dataset is empty.")
        return
    sample = dataset[0]
    text_obj = sample[0] if isinstance(sample, (list, tuple)) and len(sample) > 0 else None
    if not (isinstance(text_obj, (list, tuple)) and len(text_obj) >= 2):
        print(f"[V3-CHECK][WARN] {split_name} dataset text is not [sentence, gloss] pair mode.")
    else:
        print(f"[V3-CHECK] {split_name} dataset returns [sentence, gloss] pair mode.")


def _cache_dtype(dtype_name: str):
    name = str(dtype_name or "float16").strip().lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported latent cache dtype: {dtype_name}")


@torch.no_grad()
def build_split_latent_cache(split_name: str, dataset, vae, opt, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    store_dtype = _cache_dtype(getattr(opt, "latent_cache_dtype", "float16"))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(getattr(opt, "latent_cache_batch_size", 16))),
        shuffle=False,
        num_workers=max(0, int(getattr(opt, "latent_cache_workers", 4))),
        drop_last=False,
        collate_fn=diffusion_collate_fn,
    )

    print(
        f"[LatentCache] building split={split_name} -> {out_path} "
        f"(samples={len(dataset)}, batch={loader.batch_size}, dtype={store_dtype})"
    )
    vae.eval()
    start = time.time()

    names = []
    texts = []
    frame_lengths = []
    latent_lengths = []
    offsets = [0]
    chunks = []
    seen = 0

    for step, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) >= 6:
            text, motion, masks, m_lens, batch_names, _ = batch[:6]
        else:
            text, motion, masks, m_lens, batch_names = batch

        motion = motion.to(opt.device, dtype=torch.float32)
        m_lens = m_lens.to(opt.device, dtype=torch.long)

        T_pad = int(motion.shape[1])
        T_valid = int(m_lens.max().item()) if m_lens.numel() > 0 else T_pad
        T_valid = max(1, min(T_valid, T_pad))
        if T_valid < T_pad:
            motion = motion[:, :T_valid]

        if hasattr(vae, "mean") and hasattr(vae, "std"):
            x_in = (motion - vae.mean) / (vae.std + 1e-8)
        else:
            x_in = motion

        enc_out = vae.encode(x_in)
        if isinstance(enc_out, tuple) and len(enc_out) == 2:
            z, _ = enc_out
        else:
            z = enc_out

        if z.dim() == 3:
            Bb, Tz, JD = z.shape
            if JD % 3 == 0:
                z = z.view(Bb, Tz, JD // 3, 3)
        if z.dim() != 4:
            raise RuntimeError(f"Expected latent shape [B,Tz,J,D], got {tuple(z.shape)}")

        Tm = int(motion.shape[1])
        Tz = int(z.shape[1])
        downsample_ratio = max(1, Tm // max(1, Tz))
        z_lens = torch.clamp(m_lens // downsample_ratio, min=1, max=Tz)

        B = int(z.shape[0])
        for i in range(B):
            li = int(z_lens[i].item())
            zi = z[i, :li].detach().cpu().to(store_dtype)
            chunks.append(zi)
            offsets.append(offsets[-1] + li)

            names.append(str(batch_names[i]))
            t_i = text[i] if i < len(text) else ["", ""]
            if isinstance(t_i, (list, tuple)) and len(t_i) >= 2:
                texts.append([str(t_i[0]), str(t_i[1])])
            else:
                texts.append([str(t_i), ""])
            frame_lengths.append(int(m_lens[i].item()))
            latent_lengths.append(li)

        seen += B
        if step % 50 == 0 or seen == len(dataset):
            elapsed = time.time() - start
            print(f"[LatentCache] split={split_name} progress {seen}/{len(dataset)} elapsed={elapsed/60:.1f}m")

    if chunks:
        latents = torch.cat(chunks, dim=0).contiguous()
        J = int(latents.shape[1])
        D = int(latents.shape[2])
    else:
        J = int(getattr(vae.opt, "joints_num", 13))
        D = int(getattr(vae.opt, "latent_dim", 128))
        latents = torch.empty((0, J, D), dtype=store_dtype)

    payload = {
        "version": 1,
        "split": split_name,
        "names": names,
        "texts": texts,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "frame_lengths": torch.tensor(frame_lengths, dtype=torch.long),
        "latent_lengths": torch.tensor(latent_lengths, dtype=torch.long),
        "latents": latents,
        "meta": {
            "dtype": str(store_dtype),
            "num_samples": len(names),
            "sum_latent_steps": int(latents.shape[0]),
            "latent_shape": [int(latents.shape[1]), int(latents.shape[2])],
            "built_at_unix": int(time.time()),
        },
    }
    torch.save(payload, out_path)
    size_gb = os.path.getsize(out_path) / (1024 ** 3)
    elapsed = time.time() - start
    print(
        f"[LatentCache] done split={split_name} samples={len(names)} "
        f"sumTz={int(latents.shape[0])} file={out_path} size={size_gb:.2f}GiB "
        f"time={elapsed/60:.1f}m"
    )


def resolve_cache_paths(opt):
    cache_dir = str(getattr(opt, "latent_cache_dir", "") or "").strip()
    if not cache_dir:
        cache_dir = pjoin(_SOKE_ROOT, "checkpoints", "HIERARCHICAL", "latent_cache_how2sign")
    os.makedirs(cache_dir, exist_ok=True)
    opt.latent_cache_dir = cache_dir
    return {
        "train": pjoin(cache_dir, "train_latents.pt"),
        "val": pjoin(cache_dir, "val_latents.pt"),
        "test": pjoin(cache_dir, "test_latents.pt"),
    }


def main():
    opt = arg_parse(True)
    force_how2sign_paths(opt)

    if bool(getattr(opt, "tiny_debug", False)) and int(getattr(opt, "max_epoch", 1)) > 1:
        print(f"[TINY] overriding max_epoch {opt.max_epoch} -> 1")
        opt.max_epoch = 1

    if bool(getattr(opt, "tiny_debug", False)) and bool(getattr(opt, "tiny_disable_wandb", True)):
        os.environ["WANDB_MODE"] = "disabled"

    save_runtime_opt(opt)
    vocab_path = build_gloss_vocab_from_dictionary(opt)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_stoi = json.load(f).get("stoi", {})
    report_unknown_token_coverage(opt.train_csv_path, vocab_stoi)

    fixseed(opt.seed)

    print(
        f"[HOW2SIGN] use_rag={getattr(opt, 'use_rag', None)} "
        f"gloss_use_positional={getattr(opt, 'gloss_use_positional', None)} "
        f"enable_length_cond={getattr(opt, 'enable_length_cond', None)} "
        f"amp_dtype={getattr(opt, 'amp_dtype', None)} "
        f"use_latent_cache={getattr(opt, 'use_latent_cache', None)}"
    )

    vae = load_and_freeze_vae(opt)
    denoiser = Denoiser(opt, vae.opt.latent_dim)
    validate_v2_contract(opt, denoiser)

    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    num_params = sum(param.numel() for param in denoiser.parameters_without_clip())
    print(f"Total trainable parameters of all models: {num_params / 1_000_000:.3f}M")

    train_cfg = copy.deepcopy(opt)
    val_cfg = copy.deepcopy(opt)
    test_cfg = copy.deepcopy(opt)

    train_dataset_raw = SignDiffusionDataset(
        data_dir=opt.train_data_dir,
        csv_path=opt.train_csv_path,
        max_length=opt.max_motion_length,
        config=train_cfg,
        is_train=True,
        only_gloss=bool(getattr(opt, "train_only_gloss", True)),
        enable_custom_weight=bool(getattr(opt, "enable_custom_weight", False)),
        custom_weight_dir=str(getattr(opt, "custom_weight_dir", "") or ""),
        custom_weight_key=str(getattr(opt, "custom_weight_key", "soft_w") or "soft_w"),
        custom_weight_precheck=bool(getattr(opt, "custom_weight_precheck", False)),
    )

    val_dataset = SignDiffusionDataset(
        data_dir=opt.val_data_dir,
        csv_path=opt.val_csv_path,
        max_length=opt.max_motion_length,
        config=val_cfg,
        is_train=False,
        only_gloss=bool(getattr(opt, "train_only_gloss", True)),
        enable_custom_weight=bool(getattr(opt, "enable_custom_weight", False)),
        custom_weight_dir=str(getattr(opt, "custom_weight_dir", "") or ""),
        custom_weight_key=str(getattr(opt, "custom_weight_key", "soft_w") or "soft_w"),
        custom_weight_precheck=False,
    )

    validate_dataset_pair_mode(train_dataset_raw, "train")
    validate_dataset_pair_mode(val_dataset, "val")

    train_dataset = train_dataset_raw
    train_collate_fn = diffusion_collate_fn
    use_latent_cache = bool(getattr(opt, "use_latent_cache", False))
    if use_latent_cache and bool(getattr(opt, "enable_custom_weight", False)):
        raise ValueError("use_latent_cache=True is incompatible with enable_custom_weight=True in this script.")

    cache_paths = resolve_cache_paths(opt)
    should_build_cache = bool(getattr(opt, "build_latent_cache", False) or use_latent_cache)
    if should_build_cache:
        build_all = bool(getattr(opt, "latent_cache_build_all_splits", True))
        need_train = bool(getattr(opt, "rebuild_latent_cache", False) or (not os.path.isfile(cache_paths["train"])))
        need_val = build_all and bool(getattr(opt, "rebuild_latent_cache", False) or (not os.path.isfile(cache_paths["val"])))
        need_test = build_all and bool(getattr(opt, "rebuild_latent_cache", False) or (not os.path.isfile(cache_paths["test"])))

        if need_train:
            build_split_latent_cache("train", train_dataset_raw, vae, opt, cache_paths["train"])
        else:
            print(f"[LatentCache] reuse existing train cache: {cache_paths['train']}")

        if need_val:
            build_split_latent_cache("val", val_dataset, vae, opt, cache_paths["val"])
        elif build_all:
            print(f"[LatentCache] reuse existing val cache: {cache_paths['val']}")

        if need_test:
            test_dataset = SignDiffusionDataset(
                data_dir=opt.test_data_dir,
                csv_path=opt.test_csv_path,
                max_length=opt.max_motion_length,
                config=test_cfg,
                is_train=False,
                only_gloss=bool(getattr(opt, "train_only_gloss", True)),
                enable_custom_weight=bool(getattr(opt, "enable_custom_weight", False)),
                custom_weight_dir=str(getattr(opt, "custom_weight_dir", "") or ""),
                custom_weight_key=str(getattr(opt, "custom_weight_key", "soft_w") or "soft_w"),
                custom_weight_precheck=False,
            )
            build_split_latent_cache("test", test_dataset, vae, opt, cache_paths["test"])
        elif build_all:
            print(f"[LatentCache] reuse existing test cache: {cache_paths['test']}")

    if use_latent_cache:
        if not os.path.isfile(cache_paths["train"]):
            raise FileNotFoundError(
                f"train latent cache missing: {cache_paths['train']} "
                "Use --build_latent_cache or --rebuild_latent_cache."
            )
        train_dataset = CachedLatentDataset(
            cache_paths["train"],
            only_gloss=bool(getattr(opt, "train_only_gloss", True)),
        )
        train_collate_fn = latent_cache_collate_fn
        print(f"[LatentCache] training will use cached train split: {cache_paths['train']}")

    use_tiny = bool(getattr(opt, "tiny_debug", False))
    loader_workers = 0 if use_tiny else int(getattr(opt, "num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(
            lengths=train_dataset.lengths,
            batch_size=opt.batch_size,
            bucket_size=64,
            drop_last=True,
            shuffle=True,
        ),
        num_workers=loader_workers,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=loader_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=diffusion_collate_fn,
    )

    if use_tiny:
        train_batches = int(getattr(opt, "tiny_train_batches", 2))
        val_batches = int(getattr(opt, "tiny_val_batches", 1))
        train_loader = LimitedLoader(train_loader, train_batches)
        val_loader = LimitedLoader(val_loader, val_batches)
        print(
            f"[TINY] tiny_debug enabled: train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} batch_size={opt.batch_size}"
        )

    wandb_mode = None
    if bool(getattr(opt, "tiny_debug", False)) and bool(getattr(opt, "tiny_disable_wandb", True)):
        wandb_mode = "disabled"
    wandb_kwargs = {
        "project": "Sign_Diffusion",
        "name": opt.name,
        "config": vars(opt),
    }
    if wandb_mode is not None:
        wandb_kwargs["mode"] = wandb_mode
    wandb.init(**wandb_kwargs)

    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    trainer.train(
        train_loader,
        val_loader,
        val_loader,
        eval_wrapper=None,
        plot_eval=None,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
