import torch
import numpy as np
import os
import sys
import inspect
import copy
import json
import pandas as pd
from torch.utils.data import DataLoader
from os.path import join as pjoin

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SOKE_ROOT not in sys.path:
    sys.path.append(_SOKE_ROOT)

from diffusers import DDIMScheduler

#from models.vae.model import VAE
from models.denoiser.model_patched import Denoiser
from models.denoiser.trainer_patched import DenoiserTrainer
from options.denoiser_option_v2 import arg_parse

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils import paramUtil

#from data.t2m_dataset import Text2MotionDataset
#from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from sign_diffusion_dataset_patched import SignDiffusionDataset, diffusion_collate_fn

sys.path.append(os.path.join(_SOKE_ROOT, "mymodel", "vae"))
from qvae_model_rod3_fixed_length import VAE as MyVAE
from mGPT.utils.joints_list import SMPLX_JOINT_LANDMARK_NAMES, SELECTED_JOINT_INDICES,SELECTED_JOINT_LANDMARK_INDICES,SELECTED_JOINT_LANDMARK_BODY_EVAL,SELECTED_JOINT_LANDMARK_LHAND_EVAL,SELECTED_JOINT_LANDMARK_RHAND_EVAL, SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,SELECTED_JOINT_INDICES_BODY_ONLY,SELECTED_JOINT_INDICES_NEIGHBOR_LIST
from mGPT.utils.smplx_vertex_group import UPPER_BODY_VERTEX, LEFT_HAND_VERTEX, RIGHT_HAND_VERTEX
import math
import random
from torch.utils.data import Sampler
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_DIR", pjoin(_THIS_DIR, "wandb"))

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

class BucketBatchSampler(Sampler):
    """
    先按长度分桶，再在桶内打乱，最后按 batch_size 组成 batch。
    - lengths: list[int]，每个样本的原始长度（你可以在初始化 dataset 时预读一遍长度，或缓存到 csv）
    """
    def __init__(self, lengths, batch_size, bucket_size=200, drop_last=False, shuffle=True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # 全部 index 按长度排序
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

    def __iter__(self):
        indices = self.sorted_indices[:]
        if self.shuffle:
            # 把排序后的序列切成若干大桶，然后桶内打乱 + 桶顺序打乱
            buckets = [indices[i:i+self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
            random.shuffle(buckets)
            indices = []
            for b in buckets:
                random.shuffle(b)
                indices.extend(b)

        # 组 batch
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
    """Limit number of batches per epoch for smoke tests."""
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
def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, opt.kinematic_chain, joint, title=caption, fps=20)
        
def load_and_freeze_vae(opt):
    # --- 修改部分：支持直接指定 VAE 路径 ---
    # 如果你在启动命令中传入了 --vae_path，则直接使用，否则使用原有的拼接逻辑
    if hasattr(opt, 'vae_path') and opt.vae_path:
        vae_dir = opt.vae_path
    else:
        vae_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name)
    
    opt_path = pjoin(vae_dir, 'opt.txt')
    print(f"Loading VAE config from: {opt_path}")
    vae_opt = get_opt(opt_path, opt.device)
    vae_opt.SMPLX_JOINT_LANDMARK_NAMES = SMPLX_JOINT_LANDMARK_NAMES
    vae_opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES = SELECTED_JOINT_LANDMARK_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_BODY_EVAL = SELECTED_JOINT_LANDMARK_BODY_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_LHAND_EVAL=SELECTED_JOINT_LANDMARK_LHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_RHAND_EVAL=SELECTED_JOINT_LANDMARK_RHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX=SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX
    vae_opt.SELECTED_JOINT_INDICES_BODY_ONLY=SELECTED_JOINT_INDICES_BODY_ONLY
    vae_opt.UPPER_BODY_VERTEX = UPPER_BODY_VERTEX
    vae_opt.LEFT_HAND_VERTEX = LEFT_HAND_VERTEX
    vae_opt.RIGHT_HAND_VERTEX = RIGHT_HAND_VERTEX
    vae_opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    vae_opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    vae_opt.joints_num = len(SELECTED_JOINT_INDICES)
    model = MyVAE(vae_opt)
    
    # 根据你的 VAE 保存逻辑，权重文件通常在 model 子文件夹下
    ckpt_path = pjoin(vae_dir, 'model', 'latest.tar') 
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if "vae" in ckpt:
        model.load_state_dict(ckpt["vae"])
    else:
        model.load_state_dict(ckpt)
        
    model.freeze()
    model.to(opt.device)
    return model


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


def ensure_gloss_vocab(opt):
    path = str(getattr(opt, "gloss_vocab_path", "") or "").strip()
    if path and os.path.isfile(path):
        return path

    csv_candidates = []
    train_csv = str(getattr(opt, "train_csv_path", "") or "").strip()
    if train_csv:
        csv_candidates.append(train_csv)
    csv_candidates.extend([
        pjoin(opt.data_root, "merge_train.csv"),
        pjoin(opt.data_root, "how2sign_realigned_train.csv"),
        pjoin(opt.data_root, "train.csv"),
    ])
    csv_path = None
    for p in csv_candidates:
        if os.path.isfile(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            f"No train csv found for auto vocab build. tried={csv_candidates}. "
            "Please pass --gloss_vocab_path explicitly."
        )

    sep = _infer_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    col_gloss = _col_lookup(df, ["GLOSS", "gloss", "Gloss", "PSEUDO_GLOSS", "pseudo_gloss"])

    from sign_diffusion_dataset_patched import normalize_gloss_for_tokens

    tokens = set()
    for g in df[col_gloss].fillna("").astype(str).tolist():
        gg = normalize_gloss_for_tokens(g)
        for tok in gg.split():
            t = tok.strip()
            if t:
                tokens.add(t)

    stoi = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok in sorted(tokens):
        if tok not in stoi:
            stoi[tok] = len(stoi)

    out_path = pjoin(opt.model_dir, "gloss_vocab_v3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi}, f, ensure_ascii=False)

    print(f"[V3] built gloss vocab from {csv_path}: size={len(stoi)} -> {out_path}")
    opt.gloss_vocab_path = out_path
    return out_path


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
    except Exception as e:
        warnings.append(f"Could not introspect trainer mismatch path: {e}")

    try:
        from models.denoiser import rag as rag_mod
        rag_src = inspect.getsource(rag_mod.build_blueprint_batch)
        if "mid = tok_mat.shape[0] // 2" not in rag_src:
            warnings.append("RAG midpoint-per-word rule not detected in build_blueprint_batch.")
    except Exception as e:
        warnings.append(f"Could not introspect RAG midpoint rule: {e}")

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


if __name__ == '__main__':
    opt = arg_parse(True)
    if bool(getattr(opt, "use_rag", False)):
        from models.denoiser.rag import preconfigure_rag_opt
        rag_info = preconfigure_rag_opt(opt)
        if rag_info is not None:
            print(
                f"[RAG] preconfigured: K={rag_info['rag_k']} slots={rag_info['slot_names']} "
                f"meta={rag_info['meta_path']} source={rag_info['wmap_source']}"
            )
    fixseed(opt.seed)
    if bool(getattr(opt, "tiny_debug", False)) and int(getattr(opt, "max_epoch", 1)) > 1:
        print(f"[TINY] overriding max_epoch {opt.max_epoch} -> 1")
        opt.max_epoch = 1
    if getattr(opt, 'xyz', False):
        print(opt.xyz)
    if bool(getattr(opt, "tiny_debug", False)) and bool(getattr(opt, "tiny_disable_wandb", True)):
        os.environ["WANDB_MODE"] = "disabled"

    ensure_gloss_vocab(opt)

    print(
        f"[V3] gloss_embed_mode={getattr(opt, 'gloss_embed_mode', None)} "
        f"gloss_layers={getattr(opt, 'gloss_layers', None)} use_rag={getattr(opt, 'use_rag', None)} "
        f"rag_layers={getattr(opt, 'rag_layers', None)} rag_per_word_max_T={getattr(opt, 'rag_per_word_max_T', None)}"
    )
    # models & noise scheduler
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
    print('Total trainable parameters of all models: {}M'.format(num_params/1_000_000))

    # evaluation setup
    #wrapper_opt = get_opt(opt.dataset_opt_path, torch.device('cuda'))
    #eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_wrapper=None
    #eval_val_loader, _ = get_dataset_motion_loader(opt.dataset_opt_path, 32, 'val', device=opt.device)

    # dataset & dataloader
    #mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    #std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    # train_split_file = pjoin(opt.data_root, 'train.txt')
    # val_split_file = pjoin(opt.data_root, 'val.txt')
    train_cfg = copy.deepcopy(opt)

    # --- val cfg ---
    val_cfg = copy.deepcopy(opt)


    train_dataset = SignDiffusionDataset(
        data_dir=opt.train_data_dir, # 指向你的 npz 目录
        csv_path=opt.train_csv_path,
        max_length=opt.max_motion_length, # 扩散模型的最大步数
        config=train_cfg, # 沿用 VAE 的关节配置
        is_train=True,
        only_gloss=bool(getattr(opt, "train_only_gloss", True)),
        enable_custom_weight=bool(getattr(opt, "enable_custom_weight", False)),
        custom_weight_dir=str(getattr(opt, "custom_weight_dir", "") or ""),
        custom_weight_key=str(getattr(opt, "custom_weight_key", "soft_w") or "soft_w"),
        custom_weight_precheck=bool(getattr(opt, "custom_weight_precheck", False)),

    )

    val_dataset = SignDiffusionDataset(
        data_dir=opt.val_data_dir, # 指向你的 npz 目录
        csv_path=opt.val_csv_path,
        max_length=opt.max_motion_length, # 扩散模型的最大步数
        config=val_cfg, # 沿用 VAE 的关节配置
        is_train=False,
        only_gloss=bool(getattr(opt, "train_only_gloss", True)),
        enable_custom_weight=bool(getattr(opt, "enable_custom_weight", False)),
        custom_weight_dir=str(getattr(opt, "custom_weight_dir", "") or ""),
        custom_weight_key=str(getattr(opt, "custom_weight_key", "soft_w") or "soft_w"),
        custom_weight_precheck=bool(getattr(opt, "custom_weight_precheck", False)),
    )
    validate_dataset_pair_mode(train_dataset, "train")
    validate_dataset_pair_mode(val_dataset, "val")
    use_tiny = bool(getattr(opt, "tiny_debug", False))
    loader_workers = 0 if use_tiny else opt.num_workers
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=BucketBatchSampler(
        lengths=train_dataset.lengths,    
        batch_size=opt.batch_size,
        bucket_size=64,
        drop_last=True,
        shuffle=True
        ),
        num_workers=loader_workers, 
        collate_fn=diffusion_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        num_workers=loader_workers, 
        shuffle=False, 
        drop_last=False,
        collate_fn=diffusion_collate_fn
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

    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    #val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    # [新增] 初始化 WandB
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
    # train
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    #trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=plot_t2m)
    # 我们不再依赖 eval_wrapper，而是依赖 trainer 内部挂载的 physical_evaluator
    trainer.train(
        train_loader, 
        val_loader, 
        val_loader, 
        eval_wrapper=None, # 明确传入 None
        plot_eval=plot_t2m
    )
