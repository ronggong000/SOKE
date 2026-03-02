import os
import json
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from diffusion_sample_to_npz_v2 import (
    load_config,
    load_vae,
    load_denoiser,
    _build_scheduler,
    _default_vae_paths_from_denoiser_opt,
    _infer_latent_shape,
)


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


def _aslc_alias(metrics: dict):
    out = dict(metrics)
    for k, v in list(metrics.items()):
        if k.startswith("how2sign_"):
            out["aslcitizen_" + k[len("how2sign_"):]] = v
    return out


def _require_cuda_device() -> torch.device:
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        raise RuntimeError(
            "Diffusion evaluation requires CUDA, but torch.cuda.is_available() is False. "
            f"torch={torch.__version__} built_cuda={torch.version.cuda} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    return torch.device("cuda")


def main():
    parser = argparse.ArgumentParser("Evaluate diffusion checkpoint on ASLcitizen split with MR+DTW")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="folder with opt.txt and model/latest.tar")
    parser.add_argument("--ckpt_name", default="latest.tar", type=str, help="checkpoint file under model/")
    parser.add_argument("--split_csv", default=None, type=str, help="csv path for evaluation split")
    parser.add_argument("--data_dir", default=None, type=str, help="npz directory for evaluation split")
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

    from sign_diffusion_dataset_patched import SignDiffusionDataset, diffusion_collate_fn
    from models.denoiser.trainer_patched import DenoiserTrainer
    from utils.eval_t2m import test_denoiser

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    den_opt_path = os.path.join(checkpoint_dir, "opt.txt")
    den_ckpt_path = os.path.join(checkpoint_dir, "model", args.ckpt_name)
    if not os.path.isfile(den_opt_path):
        raise FileNotFoundError(f"Missing opt.txt: {den_opt_path}")
    if not os.path.isfile(den_ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {den_ckpt_path}")

    den_cfg = load_config(den_opt_path)
    split_csv = args.split_csv or str(getattr(den_cfg, "val_csv_path", "") or "")
    data_dir = args.data_dir or str(getattr(den_cfg, "val_data_dir", "") or "")
    if not split_csv or (not os.path.isfile(split_csv)):
        raise FileNotFoundError(f"split_csv not found: {split_csv}")
    if not data_dir or (not os.path.isdir(data_dir)):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    device = _require_cuda_device()
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

    raw_metrics = test_denoiser(
        eval_loader,
        trainer.generate,
        trainer.physical_evaluator,
        trainer.smplx_model,
        den_opt,
    )
    metrics = _aslc_alias(raw_metrics)
    mr_metrics, dtw_metrics = _split_metrics(metrics)

    print("==== Diffusion Evaluation Metrics ====")
    print("MRMetrics:")
    for k, v in mr_metrics.items():
        print(f"  {k}: {_to_python(v)}")
    print("TM2TMetrics:")
    for k, v in dtw_metrics.items():
        print(f"  {k}: {_to_python(v)}")

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = os.path.join(os.path.dirname(__file__), "eval_reports")
    os.makedirs(report_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.abspath(checkpoint_dir.rstrip("/")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = os.path.join(report_dir, f"{exp_name}_val_full_metrics_{stamp}.json")
    report_txt = os.path.join(report_dir, f"{exp_name}_val_full_metrics_{stamp}.txt")

    payload = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_file": den_ckpt_path,
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
