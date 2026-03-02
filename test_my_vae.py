import argparse
import ast
import importlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import smplx
import torch

SOKE_ROOT = Path(__file__).resolve().parent
if str(SOKE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOKE_ROOT))

from mGPT.utils.joints_list import (
    SMPLX_JOINT_LANDMARK_NAMES,
    SELECTED_JOINT_INDICES,
    SELECTED_JOINT_INDICES_BODY_ONLY,
    SELECTED_JOINT_INDICES_NEIGHBOR_LIST,
    SELECTED_JOINT_LANDMARK_BODY_EVAL,
    SELECTED_JOINT_LANDMARK_INDICES,
    SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,
    SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,
    SELECTED_JOINT_LANDMARK_LHAND_EVAL,
    SELECTED_JOINT_LANDMARK_RHAND_EVAL,
)
from mGPT.utils.smplx_vertex_group import LEFT_HAND_VERTEX, RIGHT_HAND_VERTEX, UPPER_BODY_VERTEX
from mymodel.tools.evaluator_rod3_fixed_length import MotionEvaluator

PRESET_EXPERIMENTS = OrderedDict(
    {
        "vae12_hier_12d": {
            "checkpoint_dir": SOKE_ROOT / "checkpoints" / "HIERARCHICAL" / "vae12_hier_12d",
            "model_kind": "vae12",
            "recon_mode": "cont",
        },
        "vqvae_not_hier_3p_b96h192": {
            "checkpoint_dir": SOKE_ROOT / "checkpoints" / "HIERARCHICAL" / "vqvae_not_hier_3p_b96h192",
            "model_kind": "vqvae_not",
            "recon_mode": "quant",
        },
        "vqvae_not_hier_b96h192": {
            "checkpoint_dir": SOKE_ROOT / "checkpoints" / "HIERARCHICAL" / "vqvae_not_hier_b96h192",
            "model_kind": "vqvae_not",
            "recon_mode": "quant",
        },
    }
)


def _log(message: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{stamp}] {message}", flush=True)


def _convert_string_to_type(value: str):
    value = value.strip()
    if value == "":
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config_from_txt(path: Path) -> SimpleNamespace:
    config_dict = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "---" in line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value = _convert_string_to_type(parts[1].strip())
            config_dict[key] = value
    return SimpleNamespace(**config_dict)


def _resolve_path(path_value):
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((SOKE_ROOT / path).resolve())


def _import_data_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as direct_err:
        last_err = direct_err

    short_name = name.split(".")[-1]
    data_dir = SOKE_ROOT / "mGPT" / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    try:
        return importlib.import_module(short_name)
    except Exception as short_err:
        last_err = short_err

    if not name.startswith("mGPT.data."):
        try:
            return importlib.import_module(f"mGPT.data.{short_name}")
        except Exception as pkg_err:
            last_err = pkg_err

    raise ImportError(f"Unable to import dataset module '{name}': {last_err}")


def _infer_model_kind(opt: SimpleNamespace, checkpoint_dir: Path) -> str:
    name = str(getattr(opt, "name", checkpoint_dir.name)).lower()
    data_format = str(getattr(opt, "data_format", ""))
    if getattr(opt, "per_joint_dim", None) == 12 or data_format.endswith("_dk") or "vae12" in name:
        return "vae12"
    if "vqvae_not" in name or hasattr(opt, "codebook_grouping"):
        return "vqvae_not"
    return "qvae"


def _attach_common_metadata(opt: SimpleNamespace, model_kind: str, device: torch.device):
    opt.device = device
    opt.is_train = False
    opt.smplx_model_path = _resolve_path(getattr(opt, "smplx_model_path", "deps/smpl_models"))
    opt.train_data_dir = _resolve_path(getattr(opt, "train_data_dir", "../how2sign/align_denoised_front"))
    opt.val_data_dir = _resolve_path(getattr(opt, "val_data_dir", "../how2sign/align_denoised_front_val"))
    opt.test_data_dir = _resolve_path(getattr(opt, "test_data_dir", "../how2sign/align_denoised_front_test"))

    opt.SMPLX_JOINT_LANDMARK_NAMES = SMPLX_JOINT_LANDMARK_NAMES
    opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    opt.SELECTED_JOINT_LANDMARK_INDICES = SELECTED_JOINT_LANDMARK_INDICES
    opt.SELECTED_JOINT_LANDMARK_BODY_EVAL = SELECTED_JOINT_LANDMARK_BODY_EVAL
    opt.SELECTED_JOINT_LANDMARK_LHAND_EVAL = SELECTED_JOINT_LANDMARK_LHAND_EVAL
    opt.SELECTED_JOINT_LANDMARK_RHAND_EVAL = SELECTED_JOINT_LANDMARK_RHAND_EVAL
    opt.SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST
    opt.SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX = SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX
    opt.SELECTED_JOINT_INDICES_BODY_ONLY = SELECTED_JOINT_INDICES_BODY_ONLY
    opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    opt.UPPER_BODY_VERTEX = UPPER_BODY_VERTEX
    opt.LEFT_HAND_VERTEX = LEFT_HAND_VERTEX
    opt.RIGHT_HAND_VERTEX = RIGHT_HAND_VERTEX

    opt.joints_num = len(SELECTED_JOINT_INDICES)
    opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    opt.hand_joint_indices = list(range(13, 43))
    opt.non_hand_joint_indices = list(range(13))
    opt.xyz = bool(getattr(opt, "xyz", False))

    if model_kind == "vae12":
        opt.per_joint_dim = int(getattr(opt, "per_joint_dim", 12))
        if not hasattr(opt, "data_format"):
            opt.data_format = "motion_dataset_rod3_fixed_length_dk"
    else:
        opt.per_joint_dim = int(getattr(opt, "per_joint_dim", 3))
        if not hasattr(opt, "data_format"):
            opt.data_format = "motion_dataset_rod3_fixed_length"

    opt.batch_size = int(getattr(opt, "batch_size", 32))
    opt.max_length = int(getattr(opt, "max_length", 256))
    opt.num_workers = int(getattr(opt, "num_workers", 0))
    return opt


def _create_model(opt: SimpleNamespace, model_kind: str):
    if model_kind == "vae12":
        from mymodel.vae_2.vae12_model_rod3_fixed_length import VAE

        return VAE(opt)
    if model_kind == "vqvae_not":
        from mymodel.vae_2.vqvae_not_model_rod3_fixed_length import VQVAE

        return VQVAE(opt)
    if model_kind == "qvae":
        from mymodel.vae.qvae_model_rod3_fixed_length import VAE

        return VAE(opt)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint_state(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    candidates = []
    if isinstance(checkpoint, dict):
        for key in ("vae", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                candidates.append((key, checkpoint[key]))
        candidates.append(("root", checkpoint))
    else:
        candidates.append(("root", checkpoint))

    errors = []
    for label, state_dict in candidates:
        try:
            model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
            _log(f"Loaded checkpoint from '{checkpoint_path}' using key '{label}'.")
            return
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError(
        f"Unable to load checkpoint '{checkpoint_path}'. Tried: {' | '.join(errors)}"
    )


def _create_eval_loader(opt: SimpleNamespace, split: str):
    data_module = _import_data_module(opt.data_format)
    train_loader, val_loader, test_loader = data_module.create_data_loaders(
        train_data_dir=opt.train_data_dir,
        val_data_dir=opt.val_data_dir,
        test_data_dir=opt.test_data_dir,
        batch_size=opt.batch_size,
        config=opt,
    )
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def _create_smplx_model(opt: SimpleNamespace):
    model = smplx.create(
        model_path=opt.smplx_model_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=opt.batch_size * opt.max_length,
    ).to(opt.device)
    model.eval()
    return model


def _metrics_to_python(metrics):
    out = OrderedDict()
    for family, values in metrics.items():
        out[family] = OrderedDict()
        for key, value in values.items():
            if torch.is_tensor(value):
                out[family][key] = float(value.detach().cpu().item())
            else:
                out[family][key] = float(value)
    return out


def _build_runs(args):
    runs = []
    if args.model_path is not None:
        model_path = Path(_resolve_path(args.model_path))
        if args.config_path is not None:
            config_path = Path(_resolve_path(args.config_path))
        else:
            config_path = model_path.parents[1] / "opt.txt"
        name = args.run_name or model_path.parents[1].name
        runs.append(
            {
                "name": name,
                "model_path": model_path,
                "config_path": config_path,
                "model_kind": args.model_kind,
                "recon_mode": args.recon_mode,
            }
        )
        return runs

    selected = args.experiment if args.experiment else list(PRESET_EXPERIMENTS.keys())
    for name in selected:
        spec = PRESET_EXPERIMENTS[name]
        checkpoint_dir = spec["checkpoint_dir"]
        runs.append(
            {
                "name": name,
                "model_path": checkpoint_dir / "model" / f"{args.checkpoint_name}.tar",
                "config_path": checkpoint_dir / "opt.txt",
                "model_kind": spec["model_kind"],
                "recon_mode": spec["recon_mode"],
            }
        )
    return runs


def _evaluate_one(run, args, device):
    config_path = Path(run["config_path"])
    model_path = Path(run["model_path"])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    opt = load_config_from_txt(config_path)
    model_kind = run["model_kind"]
    if model_kind == "auto":
        model_kind = _infer_model_kind(opt, config_path.parent)
    recon_mode = run["recon_mode"] if run["recon_mode"] != "auto" else None

    opt = _attach_common_metadata(opt, model_kind=model_kind, device=device)
    if args.batch_size is not None:
        opt.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        opt.num_workers = int(args.num_workers)

    _log(f"===== Evaluating {run['name']} =====")
    _log(f"model_kind={model_kind} | checkpoint={model_path}")
    _log(f"config={config_path} | data_format={opt.data_format} | split={args.split}")

    model = _create_model(opt, model_kind).to(device)
    _load_checkpoint_state(model, model_path)
    model.eval()

    _log(f"{run['name']}: building dataloader")
    eval_loader = _create_eval_loader(opt, args.split)
    try:
        num_batches = len(eval_loader)
    except TypeError:
        num_batches = "unknown"
    _log(f"{run['name']}: dataloader ready | batch_size={opt.batch_size} | batches={num_batches}")

    _log(f"{run['name']}: building SMPL-X model")
    smplx_model = _create_smplx_model(opt)
    _log(f"{run['name']}: SMPL-X model ready")

    evaluator = MotionEvaluator(opt=opt, model_kind=model_kind, recon_mode=recon_mode)
    metrics = evaluator.calculate_metrics(
        model,
        eval_loader,
        smplx_model,
        split=args.split,
        max_batches=args.max_batches,
        progress_every=args.progress_every,
        run_name=run["name"],
    )
    _log(f"{run['name']}: evaluation complete")
    return _metrics_to_python(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE/VQ-VAE motion reconstruction metrics on How2Sign."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=list(PRESET_EXPERIMENTS.keys()),
        help="Preset experiment name. Repeat this flag to evaluate multiple presets.",
    )
    parser.add_argument("--checkpoint_name", default="best", choices=["best", "latest"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--save_json", type=str, default="")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--model_kind", choices=["auto", "vae12", "vqvae_not", "qvae"], default="auto")
    parser.add_argument("--recon_mode", choices=["auto", "cont", "quant"], default="auto")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script, but no GPU is available.")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    _log(f"Using device: {device}")

    runs = _build_runs(args)
    results = OrderedDict()
    for run in runs:
        results[run["name"]] = _evaluate_one(run, args, device)

    _log("===== Evaluation Summary =====")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)

    if args.save_json:
        save_path = Path(_resolve_path(args.save_json))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        _log(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()
