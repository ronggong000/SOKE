import argparse
import ast
import os

import smplx
import torch

from mymodel.tools.evaluator_rod3_fixed_length import MotionEvaluator
from mymodel.vae.qvae_model_rod3_fixed_length import VAE
from mymodel.vae.qvae_option_fixed_length import arg_parse
from mGPT.data.motion_dataset_rod3_fixed_length import create_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE checkpoints with SOKE metrics.")
    parser.add_argument("--cfg", type=str, default="", help="Unused, for CLI compatibility.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to VAE checkpoint (latest.tar).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    return parser.parse_args()


def _coerce_value(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_opt_from_txt(opt, opt_path: str):
    if not os.path.exists(opt_path):
        return opt
    with open(opt_path, "r", encoding="utf-8") as opt_file:
        for line in opt_file:
            line = line.strip()
            if not line or line.startswith("-"):
                continue
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            value = _coerce_value(raw_value.strip())
            setattr(opt, key, value)
    return opt


def load_checkpoint(vae, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["vae"] if isinstance(checkpoint, dict) and "vae" in checkpoint else checkpoint
    vae.load_state_dict(state_dict, strict=True)


def main():
    cli_args = parse_args()
    opt = arg_parse(False)
    opt.ckpt_path = cli_args.ckpt_path
    opt.split = cli_args.split

    opt_path = os.path.join(os.path.dirname(os.path.dirname(cli_args.ckpt_path)), "opt.txt")
    opt = load_opt_from_txt(opt, opt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device

    vae = VAE(opt).to(device).eval()
    load_checkpoint(vae, opt.ckpt_path, device)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data_dir=opt.train_data_dir,
        val_data_dir=opt.val_data_dir,
        test_data_dir=opt.test_data_dir,
        batch_size=opt.batch_size,
        config=opt,
    )

    if cli_args.split == "train":
        eval_loader = train_loader
    elif cli_args.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    smplx_model = smplx.create(
        model_path=opt.smplx_model_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=opt.batch_size * opt.max_length,
    ).to(device)
    smplx_model.eval()

    evaluator = MotionEvaluator(opt=opt)
    metrics = evaluator.calculate_metrics(vae, eval_loader, smplx_model)

    print("==== VAE Evaluation Metrics ====")
    for metric_name, metric_values in metrics.items():
        print(f"{metric_name}:")
        for key, value in metric_values.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
