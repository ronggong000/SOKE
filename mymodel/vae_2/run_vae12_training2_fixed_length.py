import importlib
import random
import sys
from pathlib import Path

# Ensure project root is importable even when this script is launched by path.
SOKE_ROOT = Path(__file__).resolve().parents[2]
if str(SOKE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOKE_ROOT))

import numpy as np
import torch

from mymodel.vae_2.vae12_model_rod3_fixed_length import VAE
from mymodel.vae_2.vae12_option_fixed_length import arg_parse
from mymodel.vae_2.vae12_trainer_rod3_fixed_length import VAETrainer

try:
    import wandb
except Exception:
    wandb = None


def fixseed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _import_data_module(name: str):
    # 1) direct import
    try:
        return importlib.import_module(name)
    except Exception as direct_err:
        last_err = direct_err

    # 2) import by short module name from SOKE/mGPT/data without touching mGPT.data package __init__
    short_name = name.split(".")[-1]
    data_dir = SOKE_ROOT / "mGPT" / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    try:
        return importlib.import_module(short_name)
    except Exception as short_err:
        last_err = short_err

    # 3) package style fallback
    if not name.startswith("mGPT.data."):
        try:
            return importlib.import_module(f"mGPT.data.{short_name}")
        except Exception as pkg_err:
            last_err = pkg_err

    raise ImportError(f"Unable to import dataset module '{name}': {last_err}")


def main():
    opt = arg_parse(True)
    fixseed(opt.seed)

    if wandb is not None:
        try:
            wandb.init(project="vae-motion-synthesis", name=opt.name, config=vars(opt))
        except Exception as e:
            print(f"WandB init skipped: {e}")

    net = VAE(opt).to(opt.device)

    data_module = _import_data_module(opt.data_format)
    create_data_loaders = data_module.create_data_loaders
    train_loader, val_loader, _ = create_data_loaders(
        train_data_dir=opt.train_data_dir,
        val_data_dir=opt.val_data_dir,
        test_data_dir=opt.test_data_dir,
        batch_size=opt.batch_size,
        config=opt,
    )

    mean, std = train_loader.dataset.calculate_stats()
    net.set_stats(mean.to(opt.device), std.to(opt.device))

    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
