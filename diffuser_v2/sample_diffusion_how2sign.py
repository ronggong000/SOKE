import argparse
import json
import os
import random
from datetime import datetime

import torch

from diffusion_sample_to_npz_v2 import (
    _build_sid_to_npz,
    _load_split_rows,
    _resolve_num_frames_from_npz,
    _run_checkpoint_random_batch,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOW2SIGN_ROOT = "/fs04/scratch2/ar85/singyu/how2sign"
_SOKE_DATA_ROOT = os.path.join(_SOKE_ROOT, "data")

_SPLIT_DEFAULTS = {
    "train": (
        os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_train.csv"),
        os.path.join(_HOW2SIGN_ROOT, "align_denoised_front"),
    ),
    "val": (
        os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_val.csv"),
        os.path.join(_HOW2SIGN_ROOT, "align_denoised_front_val"),
    ),
    "test": (
        os.path.join(_SOKE_DATA_ROOT, "how2sign_realigned_test.csv"),
        os.path.join(_HOW2SIGN_ROOT, "align_denoised_front_test"),
    ),
}


def main():
    parser = argparse.ArgumentParser("Random how2sign sampling to npz using diffusion v2")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dirs", nargs="+", default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--split_csv", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_random_samples", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=os.path.join(_THIS_DIR, "outputs", "how2sign_random_samples"))
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--strict_rag", action="store_true", help="fail when checkpoint expects RAG but resources missing")
    parser.add_argument("--allow_missing_rag", dest="strict_rag", action="store_false")
    parser.set_defaults(strict_rag=True)
    parser.add_argument("--vae_opt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    args = parser.parse_args()

    ckpt_dirs = []
    if args.checkpoint_dirs:
        ckpt_dirs.extend(args.checkpoint_dirs)
    if args.checkpoint_dir:
        ckpt_dirs.append(args.checkpoint_dir)
    if not ckpt_dirs:
        raise RuntimeError("Provide --checkpoint_dir or --checkpoint_dirs")

    default_csv, default_data_dir = _SPLIT_DEFAULTS[args.split]
    split_csv = args.split_csv or default_csv
    data_dir = args.data_dir or default_data_dir
    if not os.path.isfile(split_csv):
        raise FileNotFoundError(f"split_csv not found: {split_csv}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    rows = _load_split_rows(split_csv, split_type="sentence")
    sid2npz = _build_sid_to_npz(data_dir)
    rows = [r for r in rows if r["sample_id"] in sid2npz]
    if not rows:
        raise RuntimeError(f"No csv rows matched npz files in {data_dir}")

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    chosen = rows[: int(args.num_random_samples)]
    if len(chosen) < int(args.num_random_samples):
        print(f"[WARN] requested {args.num_random_samples}, only {len(chosen)} matched samples.")

    shared_rows = []
    for idx, row in enumerate(chosen):
        sid = row["sample_id"]
        npz_path = sid2npz[sid]
        num_frames = _resolve_num_frames_from_npz(npz_path)
        shared_rows.append(
            {
                "sample_id": sid,
                "gloss": row["gloss"],
                "num_frames": int(num_frames),
                "npz_path": npz_path,
                "index": idx,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_outputs = {}
    for ckpt in ckpt_dirs:
        ckpt_name, outputs = _run_checkpoint_random_batch(ckpt, shared_rows, device=device, args=args)
        all_outputs[ckpt_name] = outputs

    meta = {
        "mode": "how2sign_random_batch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "num_infer_steps": int(args.num_infer_steps),
        "framerate": int(args.framerate),
        "split": args.split,
        "split_csv": split_csv,
        "data_dir": data_dir,
        "shared_samples": shared_rows,
        "outputs": all_outputs,
    }
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved run metadata: {meta_path}")


if __name__ == "__main__":
    main()
