# run_vis_from_ckpt.py
import os
import argparse
import torch
from functools import partial
from torch.utils.data import DataLoader

from mymodel.maskgit.dataset_maskgit import SignMotionTokenDataset, pad_collate, load_metadata
from mymodel.maskgit.maskgit_model import MaskGITTransformer
from train_maskgit_rag import visualize_and_save  # 直接复用你训练脚本最后那套存 npz 的逻辑


def _strip_module_prefix(state_dict: dict):
    # 兼容 DDP / DataParallel 的 "module."
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_text_emb_dirs(config: dict, split: str):
    # 完全照你 train_maskgit.py 的优先级：text_emb_bases(list) > text_emb_base(+gloss_emb_base)
    bases = config.get("text_emb_bases", [])
    out = []
    if isinstance(bases, (list, tuple)) and len(bases) > 0:
        for b in bases:
            if b:
                out.append(os.path.join(str(b), split))
    else:
        b0 = config.get("text_emb_base", "")
        if b0:
            out.append(os.path.join(str(b0), split))
        b1 = config.get("gloss_emb_base", "")
        if b1:
            out.append(os.path.join(str(b1), split))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to *.tar checkpoint")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", type=str, default="", help="override config['save_dir'] for outputs")
    ap.add_argument("--batch_size", type=int, default=0, help="override batch size (0 = use config)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load ckpt
    payload = torch.load(args.ckpt, map_location="cpu")

    # 2) pull config (优先从 ckpt 里拿；没有就当你自己改成了别的格式)
    config = payload.get("config", None)
    if config is None:
        raise RuntimeError("Checkpoint has no 'config'. 请确认你保存 ckpt 时把 CONFIG 一起存进去了。")

    if args.out_dir:
        config["save_dir"] = args.out_dir
    os.makedirs(config["save_dir"], exist_ok=True)

    # 3) metadata -> build model
    meta = load_metadata(config["dataset_root"])
    slot_names = list(meta["slots"])
    codebook_sizes = list(meta["codebook_sizes"])
    slot2q_idx = list(meta.get("slot2q_idx", [i for i in range(len(slot_names))]))
    q_idx_to_size = dict(meta.get("q_idx_to_size", {i: codebook_sizes[i] for i in range(len(slot_names))}))

    # max_seq_len：用 config 里显式的 max_seq_len，没有就用 max_len（你的 dataset 截断长度）
    max_seq_len = int(config.get("max_seq_len", config.get("max_len", 1024)))

    model = MaskGITTransformer(
        slot_names=slot_names,
        codebook_sizes=codebook_sizes,
        dim=int(config.get("dim", 512)),
        depth=int(config.get("depth", 8)),
        heads=int(config.get("heads", 8)),
        text_dim=int(config.get("text_dim", 1024)),
        max_seq_len=max_seq_len,
        dropout=float(config.get("dropout", 0.1)),
        # group tying（你现在是 7 codebooks / 13 slots 的那个模式）
        slot2q_idx=slot2q_idx,
        q_idx_to_size=q_idx_to_size,
        tie_groups=bool(config.get("tie_groups", True)),
    ).to(device)

    # 4) load weights (兼容旧 ckpt：允许 bp_* 缺失)
    state_dict = payload.get("model", None)
    if state_dict is None:
        state_dict = payload.get("state_dict", None)
    if state_dict is None:
        raise RuntimeError("Checkpoint missing 'model' (or 'state_dict').")

    state_dict = _strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        inc = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(inc, "missing_keys", []))
        unexpected = list(getattr(inc, "unexpected_keys", []))

        def _is_bp_key(k: str) -> bool:
            return k.startswith("bp_") or k.startswith("blueprint_") or k.startswith("rag_")

        if (missing and not all(_is_bp_key(k) for k in missing)) or (unexpected and not all(_is_bp_key(k) for k in unexpected)):
            raise RuntimeError(
                f"[run_vis] Strict load failed and mismatch isn't only bp_/rag.\n"
                f"Original error: {e}\n"
                f"missing_keys (first 20): {missing[:20]}\n"
                f"unexpected_keys (first 20): {unexpected[:20]}"
            )

        print(
            "[run_vis] ⚠️ Loaded ckpt with strict=False (bp_/rag mismatch only). "
            "This is expected when visualizing old non-RAG checkpoints."
        )

    # 5) build loader (只取一批做可视化，所以 num_workers=0 最稳)
    emb_dirs = _build_text_emb_dirs(config, args.split)
    text_source = str(config.get("text_source", "text"))
    max_text_len = config.get("max_text_len", None)

    ds = SignMotionTokenDataset(
        dataset_root=config["dataset_root"],
        split=args.split,
        text_emb_dir=emb_dirs,
        max_len=int(config.get("max_len", 1024)),
        max_text_len=max_text_len,
        text_source=text_source,
        meta=meta,
    )

    bs = int(args.batch_size) if int(args.batch_size) > 0 else int(config.get("batch_size", 8))
    collate = partial(pad_collate, codebook_sizes=codebook_sizes)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )

    # 6) reuse your existing exporter: will save into config["save_dir"]/vis_results
    visualize_and_save(model, loader, config, device)


if __name__ == "__main__":
    main()
