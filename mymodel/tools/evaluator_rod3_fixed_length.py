from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional

import torch

from mGPT.metrics.mr import MRMetrics
from mGPT.metrics.t2m import TM2TMetrics
from mGPT.utils.joints_list import SMPLX_JOINT_NAMES


class MotionEvaluator:
    def __init__(self, opt, model_kind: str = "qvae", recon_mode: Optional[str] = None):
        self.opt = opt
        self.device = opt.device
        self.model_kind = model_kind
        self.recon_mode = recon_mode or self._default_recon_mode(model_kind)
        self._joints_num = len(opt.SELECTED_JOINT_INDICES)
        self._smplx_joint_count = len(SMPLX_JOINT_NAMES)
        self._name_counter = 0

        self._mr_metrics = MRMetrics(
            njoints=opt.joints_num,
            jointstype="humanml3d",
            force_in_meter=True,
        )
        self._dtw_metrics = TM2TMetrics(
            cfg=SimpleNamespace(),
            dataname="how2sign",
        )

    def _default_recon_mode(self, model_kind: str) -> str:
        if model_kind == "vae12":
            return "cont"
        return "quant"

    def _extract_pose_features(self, motion: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = motion.shape
        if feat_dim == self._joints_num * 3:
            return motion
        if feat_dim % self._joints_num != 0:
            raise ValueError(
                f"Feature dim {feat_dim} is not divisible by joints_num {self._joints_num}."
            )
        per_joint_dim = feat_dim // self._joints_num
        if per_joint_dim < 3:
            raise ValueError(f"Invalid per-joint feature dim {per_joint_dim}.")
        motion = motion.view(batch_size, seq_len, self._joints_num, per_joint_dim)
        pose = motion[..., :3].contiguous().view(batch_size, seq_len, self._joints_num * 3)
        return pose

    def _expand_to_smplx_pose(self, motion: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = motion.shape
        if feat_dim % 3 != 0:
            raise ValueError(
                f"Expected pose features divisible by 3, got D={feat_dim}."
            )
        joint_dim = feat_dim // 3
        if joint_dim != self._joints_num:
            raise ValueError(
                f"Expected {self._joints_num} joints, got {joint_dim}."
            )
        motion = motion.view(batch_size, seq_len, joint_dim, 3)
        full_pose = torch.zeros(
            batch_size,
            seq_len,
            self._smplx_joint_count,
            3,
            device=motion.device,
            dtype=motion.dtype,
        )
        full_pose[:, :, self.opt.SELECTED_JOINT_INDICES, :] = motion
        return full_pose

    def _smplx_forward(self, smplx_model, full_pose: torch.Tensor):
        batch_size, seq_len = full_pose.shape[:2]
        flat_pose = full_pose.reshape(batch_size * seq_len, self._smplx_joint_count, 3)
        body_pose = flat_pose[:, 1:22].reshape(batch_size * seq_len, -1)
        left_hand_pose = flat_pose[:, 25:40].reshape(batch_size * seq_len, -1)
        right_hand_pose = flat_pose[:, 40:55].reshape(batch_size * seq_len, -1)
        output = smplx_model(
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        return output.vertices, output.joints

    def _reconstruct_motion(self, model, motion: torch.Tensor) -> torch.Tensor:
        if self.model_kind == "vae12":
            out = model(motion)
            if isinstance(out, tuple):
                return out[0]
            return out

        if self.recon_mode == "cont":
            out = model(motion, only_cont=True)
        elif self.recon_mode == "quant":
            out = model(motion, only_quant=True)
        else:
            raise ValueError(f"Unsupported recon_mode: {self.recon_mode}")

        if not isinstance(out, tuple) or len(out) == 0:
            raise RuntimeError("Model reconstruction output is not in the expected tuple format.")
        return out[0]

    def calculate_metrics(
        self,
        model,
        val_loader,
        smplx_model,
        split: str = "test",
        max_batches: Optional[int] = None,
        progress_every: int = 1,
        run_name: str = "eval",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        model.eval()
        self._mr_metrics.reset()
        self._dtw_metrics.reset()
        self._name_counter = 0

        try:
            total_batches = len(val_loader)
            if max_batches is not None:
                total_batches = min(total_batches, max_batches)
        except TypeError:
            total_batches = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                current_batch = batch_idx + 1
                if progress_every > 0 and ((current_batch - 1) % progress_every == 0):
                    total_txt = total_batches if total_batches is not None else "?"
                    print(f"[{run_name}] batch {current_batch}/{total_txt} start", flush=True)

                motion, lengths = batch
                motion = motion.to(self.device)
                lengths = lengths.to(self.device)

                pred_motion = self._reconstruct_motion(model, motion)
                motion_pose = self._extract_pose_features(motion)
                pred_pose = self._extract_pose_features(pred_motion)

                pose_ref = self._expand_to_smplx_pose(motion_pose)
                pose_rst = self._expand_to_smplx_pose(pred_pose)

                vertices_ref, joints_ref = self._smplx_forward(smplx_model, pose_ref)
                vertices_rst, joints_rst = self._smplx_forward(smplx_model, pose_rst)

                lengths_list = lengths.detach().cpu().tolist()
                batch_size = motion.shape[0]
                src = ["how2sign"] * batch_size
                names = [f"sample_{self._name_counter + i}" for i in range(batch_size)]
                self._name_counter += batch_size

                self._mr_metrics.update(
                    feats_rst=pred_pose,
                    feats_ref=motion_pose,
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=lengths_list,
                    src=src,
                    name=names,
                )
                self._dtw_metrics.update(
                    feats_rst=pred_pose,
                    feats_ref=motion_pose,
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=lengths_list,
                    lengths_rst=lengths_list,
                    split=split,
                    src=src,
                    name=names,
                )

                if progress_every > 0 and (current_batch % progress_every == 0):
                    print(f"[{run_name}] batch {current_batch} done", flush=True)

        print(f"[{run_name}] computing final metrics", flush=True)
        mr_metrics = self._mr_metrics.compute(sanity_flag=False)
        dtw_metrics = self._dtw_metrics.compute(sanity_flag=False)
        return {"MRMetrics": mr_metrics, "TM2TMetrics": dtw_metrics}
