import argparse
import os
from os.path import join as pjoin

import torch

from mGPT.utils.smplx_vertex_group import LEFT_HAND_VERTEX, RIGHT_HAND_VERTEX, UPPER_BODY_VERTEX
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


def arg_parse(is_train: bool = False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic setup
    parser.add_argument("--name", type=str, default="vqvae_not_hier_b96h192")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="", help="Unused compatibility argument.")
    parser.add_argument("--smplx_model_path", type=str, default="deps/smpl_models")

    # dataloader
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HIERARCHICAL",
        choices=["SMPLX_SL", "HAND_CENTRIC", "HIERARCHICAL"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--train_data_dir", type=str, default="../how2sign/align_denoised_front")
    parser.add_argument("--val_data_dir", type=str, default="../how2sign/align_denoised_front_val")
    parser.add_argument("--test_data_dir", type=str, default="../how2sign/align_denoised_front_test")
    parser.add_argument(
        "--data_format",
        type=str,
        default="motion_dataset_rod3_fixed_length",
        help="Import path for data module.",
    )

    # optimization
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--warm_up_iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"], help="Autocast dtype for training.")
    parser.add_argument("--milestones", nargs="+", type=int, default=[150_000, 250_000])
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--recon_loss", type=str, default="mse", choices=["mse", "l1", "l1_smooth"])
    parser.add_argument("--mesh_loss", type=str, default="l1_smooth", choices=["mse", "l1", "l1_smooth"])
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_q_recon", type=float, default=1.0)
    parser.add_argument("--lambda_quant", type=float, default=1.0)
    parser.add_argument("--finger_loss_weight", type=float, default=1.0)

    # architecture
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_extra_layers", type=int, default=1)
    parser.add_argument("--norm", type=str, default="none", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal_downsample", action="store_true", help="Enable temporal pooling (disabled by default).")

    parser.add_argument("--codebook_size_body", type=int, default=96)
    parser.add_argument("--codebook_size_hand", type=int, default=192)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument(
        "--codebook_grouping",
        type=str,
        default="finger_distinct",
        choices=["default", "arm_mirror", "thumb_sep", "finger_distinct", "full_book"],
    )

    # logging/checkpoint
    parser.add_argument("--is_continue", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument("--eval_every_e", type=int, default=10)

    opt = parser.parse_args()

    if opt.gpu_id == -1:
        opt.device = torch.device("cpu")
    else:
        torch.cuda.set_device(opt.gpu_id)
        opt.device = torch.device(f"cuda:{opt.gpu_id}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.is_train = is_train

    # static metadata
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

    opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    opt.joints_num = len(SELECTED_JOINT_INDICES)
    opt.hand_joint_indices = list(range(13, 43))
    opt.non_hand_joint_indices = list(range(13))
    opt.reduce_dim_finger = False
    opt.xyz = False

    if is_train:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt", encoding="utf-8") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(vars(opt).items()):
                opt_file.write(f"{k}: {v}\n")
            opt_file.write("-------------- End ----------------\n")

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    return opt
