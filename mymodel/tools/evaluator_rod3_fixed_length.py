from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import torch

from mGPT.metrics.mr import MRMetrics
from mGPT.metrics.t2m import TM2TMetrics
from mGPT.utils.joints_list import SMPLX_JOINT_NAMES


@dataclass
class EvalResult:
    mr_metrics: Dict[str, torch.Tensor]
    dtw_metrics: Dict[str, torch.Tensor]


class MotionEvaluator:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self._mr_metrics = MRMetrics(
            njoints=opt.joints_num,
            jointstype="humanml3d",
            force_in_meter=True,
        )
        self._dtw_metrics = TM2TMetrics(
            cfg=SimpleNamespace(),
            dataname="how2sign",
        )
        self._smplx_joint_count = len(SMPLX_JOINT_NAMES)
        self._name_counter = 0

    def _expand_to_smplx_pose(self, motion: torch.Tensor) -> torch.Tensor:
        """Expand selected 43-joint axis-angle to full SMPL-X (55 joints)."""
        batch_size, seq_len, feat_dim = motion.shape
        if feat_dim % 3 != 0:
            raise ValueError(
                f"Expected axis-angle features, got D={feat_dim} not divisible by 3."
            )
        joint_dim = feat_dim // 3
        if joint_dim != len(self.opt.SELECTED_JOINT_INDICES):
            raise ValueError(
                f"Expected {len(self.opt.SELECTED_JOINT_INDICES)} joints, got {joint_dim}."
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
        global_orient = flat_pose[:, 0]
        body_pose = flat_pose[:, 1:22].reshape(batch_size * seq_len, -1)
        jaw_pose = flat_pose[:, 22].reshape(batch_size * seq_len, -1)
        lhand_pose = flat_pose[:, 25:40].reshape(batch_size * seq_len, -1)
        rhand_pose = flat_pose[:, 40:55].reshape(batch_size * seq_len, -1)

        betas = torch.zeros(batch_size * seq_len, 10, device=full_pose.device)
        expression = torch.zeros(batch_size * seq_len, 10, device=full_pose.device)
        zero_pose = torch.zeros(batch_size * seq_len, 3, device=full_pose.device)
        output = smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=expression,
        )
        return output.vertices, output.joints

    def calculate_metrics(self, vae, val_loader, smplx_model) -> Dict[str, Dict[str, torch.Tensor]]:
        vae.eval()
        self._mr_metrics.reset()
        self._dtw_metrics.reset()

        with torch.no_grad():
            for batch in val_loader:
                motion, lengths = batch
                motion = motion.to(self.device)
                lengths = lengths.to(self.device)

                out_cont, out_quant, _, _, _ = vae(motion)
                pred_motion = out_quant

                pose_ref = self._expand_to_smplx_pose(motion)
                pose_rst = self._expand_to_smplx_pose(pred_motion)

                vertices_ref, joints_ref = self._smplx_forward(smplx_model, pose_ref)
                vertices_rst, joints_rst = self._smplx_forward(smplx_model, pose_rst)

                lengths_list = lengths.detach().cpu().tolist()
                batch_size = motion.shape[0]
                src = ["how2sign"] * batch_size
                names = [f"sample_{self._name_counter + i}" for i in range(batch_size)]
                self._name_counter += batch_size

                self._mr_metrics.update(
                    feats_rst=pred_motion,
                    feats_ref=motion,
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=lengths_list,
                    src=src,
                    name=names,
                )
                self._dtw_metrics.update(
                    feats_rst=pred_motion,
                    feats_ref=motion,
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=lengths_list,
                    lengths_rst=lengths_list,
                    split="test",
                    src=src,
                    name=names,
                )

        mr_metrics = self._mr_metrics.compute(sanity_flag=False)
        dtw_metrics = self._dtw_metrics.compute(sanity_flag=False)
        return {"MRMetrics": mr_metrics, "TM2TMetrics": dtw_metrics}
