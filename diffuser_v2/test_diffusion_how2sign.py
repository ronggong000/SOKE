import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from diffusion_sample_to_npz_v2 import (
    _build_scheduler,
    _default_vae_paths_from_denoiser_opt,
    _infer_latent_shape,
    load_config,
    load_denoiser,
    load_vae,
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


def _to_python(v):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return [float(x) for x in v.detach().cpu().view(-1).tolist()]
    if isinstance(v, (float, int, str, bool)):
        return v
    return str(v)


def _split_metrics(metrics: dict):
    mr = {}
    dtw = {}
    for k, v in metrics.items():
        if "DTW" in k:
            dtw[k] = v
        else:
            mr[k] = v
    return mr, dtw


def main():
    parser = argparse.ArgumentParser("Evaluate diffusion checkpoint on how2sign split with MR+DTW")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="folder with opt.txt and model/latest.tar")
    parser.add_argument("--ckpt_name", default="latest.tar", type=str, help="checkpoint file under model/")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--split_csv", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_infer_steps", default=None, type=int)
    parser.add_argument("--strict_rag", action="store_true", help="fail when checkpoint expects RAG but resources missing")
    parser.add_argument("--allow_missing_rag", dest="strict_rag", action="store_false")
    parser.set_defaults(strict_rag=True)
    parser.add_argument("--vae_opt", default=None, type=str)
    parser.add_argument("--vae_ckpt", default=None, type=str)
    parser.add_argument("--report_dir", default=None, type=str)
    args = parser.parse_args()

    from models.denoiser.trainer_patched import DenoiserTrainer
    from sign_diffusion_dataset_patched import SignDiffusionDataset, diffusion_collate_fn
    from utils.eval_t2m import test_denoiser

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    den_opt_path = os.path.join(checkpoint_dir, "opt.txt")
    den_ckpt_path = os.path.join(checkpoint_dir, "model", args.ckpt_name)
    if not os.path.isfile(den_opt_path):
        raise FileNotFoundError(f"Missing opt.txt: {den_opt_path}")
    if not os.path.isfile(den_ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {den_ckpt_path}")

    den_cfg = load_config(den_opt_path)
    default_csv, default_data_dir = _SPLIT_DEFAULTS[args.split]
    split_csv = args.split_csv or default_csv
    data_dir = args.data_dir or default_data_dir

    if not split_csv or (not os.path.isfile(split_csv)):
        raise FileNotFoundError(f"split_csv not found: {split_csv}")
    if not data_dir or (not os.path.isdir(data_dir)):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vae_opt_path = args.vae_opt
    vae_ckpt_path = args.vae_ckpt
    if (not vae_opt_path) or (not vae_ckpt_path):
        d_vae_opt, d_vae_ckpt = _default_vae_paths_from_denoiser_opt(den_cfg)
        vae_opt_path = vae_opt_path or d_vae_opt
        vae_ckpt_path = vae_ckpt_path or d_vae_ckpt
    if not vae_opt_path or not vae_ckpt_path:
        raise RuntimeError("Cannot resolve VAE paths. Provide --vae_opt and --vae_ckpt.")

    vae = load_vae(vae_opt_path, vae_ckpt_path, device)

    max_len = int(getattr(den_cfg, "max_motion_length", 2048))
    vae_latent_dim = int(_infer_latent_shape(vae, num_frames=max_len, device=device)[-1])
    denoiser, den_opt, _ = load_denoiser(
        den_opt_path,
        den_ckpt_path,
        vae_latent_dim,
        device,
        strict_rag=bool(args.strict_rag),
    )

    den_opt.device = device
    den_opt.is_train = False
    if args.num_infer_steps is not None:
        den_opt.num_inference_timesteps = int(args.num_infer_steps)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(getattr(den_opt, "batch_size", 16))
    eval_dataset = SignDiffusionDataset(
        data_dir=data_dir,
        csv_path=split_csv,
        max_length=int(getattr(den_opt, "max_motion_length", 2048)),
        config=den_opt,
        is_train=False,
        only_gloss=bool(getattr(den_opt, "train_only_gloss", True)),
        enable_custom_weight=bool(getattr(den_opt, "enable_custom_weight", False)),
        custom_weight_dir=str(getattr(den_opt, "custom_weight_dir", "") or ""),
        custom_weight_key=str(getattr(den_opt, "custom_weight_key", "soft_w") or "soft_w"),
        custom_weight_precheck=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        shuffle=False,
        drop_last=False,
        collate_fn=diffusion_collate_fn,
    )

    scheduler = _build_scheduler(den_opt)
    trainer = DenoiserTrainer(den_opt, denoiser, vae, scheduler)
    denoiser.eval()
    vae.eval()

    metrics = test_denoiser(
        eval_loader,
        trainer.generate,
        trainer.physical_evaluator,
        trainer.smplx_model,
        den_opt,
    )
    mr_metrics, dtw_metrics = _split_metrics(metrics)

    print("==== Diffusion Evaluation Metrics ====")
    print("MRMetrics:")
    for k, v in mr_metrics.items():
        print(f"  {k}: {_to_python(v)}")
    print("TM2TMetrics:")
    for k, v in dtw_metrics.items():
        print(f"  {k}: {_to_python(v)}")

    report_dir = args.report_dir or os.path.join(_THIS_DIR, "eval_reports")
    os.makedirs(report_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.abspath(checkpoint_dir.rstrip("/")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = os.path.join(report_dir, f"{exp_name}_{args.split}_full_metrics_{stamp}.json")
    report_txt = os.path.join(report_dir, f"{exp_name}_{args.split}_full_metrics_{stamp}.txt")

    payload = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_file": den_ckpt_path,
        "split": args.split,
        "split_csv": split_csv,
        "data_dir": data_dir,
        "num_samples": len(eval_dataset),
        "batch_size": batch_size,
        "num_inference_timesteps": int(getattr(den_opt, "num_inference_timesteps", 50)),
        "metrics": {k: _to_python(v) for k, v in metrics.items()},
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("==== Diffusion Evaluation Metrics ====\n")
        f.write("MRMetrics:\n")
        for k, v in mr_metrics.items():
            f.write(f"  {k}: {_to_python(v)}\n")
        f.write("TM2TMetrics:\n")
        for k, v in dtw_metrics.items():
            f.write(f"  {k}: {_to_python(v)}\n")

    print(f"Saved report json: {report_json}")
    print(f"Saved report txt: {report_txt}")


if __name__ == "__main__":
    main()
