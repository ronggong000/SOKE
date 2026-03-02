import os
import sys
from typing import Any

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SOKE_ROOT not in sys.path:
    sys.path.append(_SOKE_ROOT)


def inject_joint_constants(vae_opt):
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

    vae_opt.SMPLX_JOINT_LANDMARK_NAMES = SMPLX_JOINT_LANDMARK_NAMES
    vae_opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES = SELECTED_JOINT_LANDMARK_INDICES
    vae_opt.SELECTED_JOINT_LANDMARK_BODY_EVAL = SELECTED_JOINT_LANDMARK_BODY_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_LHAND_EVAL = SELECTED_JOINT_LANDMARK_LHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_RHAND_EVAL = SELECTED_JOINT_LANDMARK_RHAND_EVAL
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST
    vae_opt.SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX = SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX
    vae_opt.SELECTED_JOINT_INDICES_BODY_ONLY = SELECTED_JOINT_INDICES_BODY_ONLY
    vae_opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    vae_opt.UPPER_BODY_VERTEX = UPPER_BODY_VERTEX
    vae_opt.LEFT_HAND_VERTEX = LEFT_HAND_VERTEX
    vae_opt.RIGHT_HAND_VERTEX = RIGHT_HAND_VERTEX
    vae_opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    vae_opt.joints_num = len(SELECTED_JOINT_INDICES)
    return vae_opt


def resolve_vae_family(opt_like: Any) -> str:
    explicit = str(getattr(opt_like, "vae_family", "") or "").strip().lower()
    if explicit in {"qvae", "vae12"}:
        return explicit

    data_format = str(getattr(opt_like, "data_format", "") or "").strip().lower()
    per_joint_dim = int(getattr(opt_like, "per_joint_dim", 3) or 3)
    input_dim = int(getattr(opt_like, "input_dim", 0) or 0)

    if per_joint_dim == 12 or data_format.endswith("_dk") or input_dim == 516:
        return "vae12"
    return "qvae"


def resolve_motion_repr(opt_like: Any) -> str:
    explicit = str(getattr(opt_like, "motion_repr", "") or "").strip().lower()
    if explicit in {"pose3d", "dk12"}:
        return explicit
    if resolve_vae_family(opt_like) == "vae12":
        return "dk12"
    return "pose3d"


def prepare_vae_opt(vae_opt, device=None):
    if device is not None:
        vae_opt.device = device
    vae_opt = inject_joint_constants(vae_opt)
    vae_opt.vae_family = resolve_vae_family(vae_opt)
    vae_opt.motion_repr = resolve_motion_repr(vae_opt)
    if not hasattr(vae_opt, "pose_key"):
        vae_opt.pose_key = "poses"
    if not hasattr(vae_opt, "ric_key"):
        vae_opt.ric_key = "smpl_kp3d_ric"
    if not hasattr(vae_opt, "fps"):
        vae_opt.fps = 24.0
    return vae_opt


def _load_vae_class(vae_family: str):
    if vae_family == "vae12":
        from mymodel.vae_2.vae12_model_rod3_fixed_length import VAE
        return VAE
    from mymodel.vae.qvae_model_rod3_fixed_length import VAE
    return VAE


def load_vae_model(vae_opt, vae_ckpt_path: str):
    vae_family = resolve_vae_family(vae_opt)
    vae_opt = prepare_vae_opt(vae_opt, getattr(vae_opt, "device", None))
    VAE = _load_vae_class(vae_family)
    vae = VAE(vae_opt).to(vae_opt.device)

    ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "vae" in ckpt:
        state = ckpt["vae"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("VAE checkpoint is not a dict")

    vae.load_state_dict(state, strict=True)
    if hasattr(vae, "freeze"):
        vae.freeze()
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

    vae.eval()
    setattr(vae, "_soke_vae_family", vae_family)
    setattr(vae, "_soke_motion_repr", resolve_motion_repr(vae_opt))
    return vae


def _ensure_j3(arr: np.ndarray, key_name: str) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
        return arr.reshape(arr.shape[0], arr.shape[-1] // 3, 3).astype(np.float32, copy=False)
    raise ValueError(f"Invalid {key_name} shape {arr.shape}, expected [T,J,3] or [T,J*3].")


def _select_joints(arr_j3: np.ndarray, config) -> np.ndarray:
    sel = list(getattr(config, "SELECTED_JOINT_INDICES", []))
    if not sel:
        return arr_j3
    if arr_j3.shape[1] == len(sel):
        return arr_j3
    max_sel = max(sel)
    if arr_j3.shape[1] > max_sel:
        return arr_j3[:, sel, :]
    raise ValueError(f"Joint axis too small ({arr_j3.shape[1]}) for selected index max {max_sel}.")


def _compute_velocity(x_j3: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(x_j3, dtype=np.float32)
    vel[1:] = (x_j3[1:] - x_j3[:-1]) * float(fps)
    return vel


def build_motion_feature_from_npz(data, config):
    motion_repr = resolve_motion_repr(config)
    if motion_repr == "pose3d":
        key = "joints_xyz" if bool(getattr(config, "xyz", False)) else str(getattr(config, "pose_key", "poses"))
        raw = _ensure_j3(np.array(data[key]), key)
        return _select_joints(raw, config).astype(np.float32, copy=False)

    pose_key = str(getattr(config, "pose_key", "poses"))
    ric_key = str(getattr(config, "ric_key", "smpl_kp3d_ric"))
    fps = float(getattr(config, "fps", 24.0))

    pose = _select_joints(_ensure_j3(np.array(data[pose_key]), pose_key), config)
    ric = _select_joints(_ensure_j3(np.array(data[ric_key]), ric_key), config)
    rot_vel = _compute_velocity(pose, fps)
    ric_vel = _compute_velocity(ric, fps)
    feat = np.concatenate([pose, ric, rot_vel, ric_vel], axis=-1)
    return feat.astype(np.float32, copy=False)


def get_motion_feature_dim(opt_like: Any) -> int:
    joints_num = int(getattr(opt_like, "joints_num", len(getattr(opt_like, "SELECTED_JOINT_INDICES", [])) or 43))
    motion_repr = resolve_motion_repr(opt_like)
    per_joint_dim = 12 if motion_repr == "dk12" else 3
    return joints_num * per_joint_dim


def normalize_motion_for_vae(vae, x_raw: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "mean") and hasattr(vae, "std"):
        return (x_raw - vae.mean) / (vae.std + 1e-8)
    return x_raw


@torch.no_grad()
def infer_latent_shape_from_vae(vae, num_frames: int, device: torch.device):
    d_flat = get_motion_feature_dim(vae.opt)
    dummy = torch.zeros((1, int(num_frames), d_flat), device=device, dtype=torch.float32)
    enc_out = vae.encode(normalize_motion_for_vae(vae, dummy))
    z = enc_out[0] if isinstance(enc_out, tuple) else enc_out
    if z.dim() != 4:
        raise RuntimeError(f"vae.encode(dummy) must return [B,Tz,J,D], got {tuple(z.shape)}")
    return z.shape


def extract_pose_from_motion_tensor(motion: torch.Tensor, opt_like: Any) -> torch.Tensor:
    if motion.dim() == 4:
        if motion.shape[-1] < 3:
            raise RuntimeError(f"Unexpected motion last dim < 3: {tuple(motion.shape)}")
        return motion[..., :3].contiguous()

    if motion.dim() != 3:
        raise RuntimeError(f"Expected motion tensor dim 3 or 4, got {tuple(motion.shape)}")

    joints_num = int(getattr(opt_like, "joints_num", len(getattr(opt_like, "SELECTED_JOINT_INDICES", [])) or 43))
    d_flat = int(motion.shape[-1])
    if d_flat % joints_num != 0:
        raise RuntimeError(f"Motion flat dim {d_flat} is not divisible by joints_num={joints_num}")

    per_joint_dim = d_flat // joints_num
    motion = motion.view(motion.shape[0], motion.shape[1], joints_num, per_joint_dim)
    if per_joint_dim < 3:
        raise RuntimeError(f"Unexpected per_joint_dim={per_joint_dim} < 3")
    return motion[..., :3].contiguous()


@torch.no_grad()
def decode_latent_to_pose3d(vae, z: torch.Tensor) -> torch.Tensor:
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(vae, "mean") and hasattr(vae, "std"):
        out = out * vae.std + vae.mean
    return extract_pose_from_motion_tensor(out, vae.opt)
