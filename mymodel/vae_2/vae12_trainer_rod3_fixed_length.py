import os
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import smplx


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _loss_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if torch.is_tensor(x) else x


class VAETrainer:
    def __init__(self, opt, vae, scaler=None):
        self.opt = opt
        self.vae = vae
        self.device = opt.device
        self.logger = SummaryWriter(opt.log_dir) if (opt.is_train and SummaryWriter is not None) else None

        amp_dtype = str(getattr(opt, "amp_dtype", "bf16")).lower()
        if amp_dtype not in {"fp16", "bf16", "none"}:
            raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
        self.use_amp = self.device.type == "cuda" and amp_dtype != "none"
        self.amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        scaler_enabled = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = scaler if scaler is not None else amp.GradScaler("cuda", enabled=scaler_enabled)

        if opt.recon_loss == "mse":
            self.recon_criterion = nn.MSELoss(reduction="none")
        elif opt.recon_loss == "l1_smooth":
            self.recon_criterion = nn.SmoothL1Loss(reduction="none")
        else:
            self.recon_criterion = nn.L1Loss(reduction="none")

        if opt.mesh_loss == "mse":
            self.mesh_criterion = nn.MSELoss(reduction="none")
        elif opt.mesh_loss == "l1_smooth":
            self.mesh_criterion = nn.SmoothL1Loss(reduction="none")
        else:
            self.mesh_criterion = nn.L1Loss(reduction="none")

        self.smplx_model = None
        try:
            self.smplx_model = smplx.create(
                model_path=opt.smplx_model_path,
                model_type="smplx",
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
                batch_size=opt.batch_size * opt.max_length,
            ).to(self.device).eval()
        except Exception as e:
            print(f"[{_now()}] SMPLX model init failed: {e}")

        self.ALL_SELECTED_VERTICES = opt.UPPER_BODY_VERTEX + opt.LEFT_HAND_VERTEX + opt.RIGHT_HAND_VERTEX
        self.body_indices = torch.tensor(opt.SELECTED_JOINT_INDICES_BODY_ONLY, device=self.device, dtype=torch.long)

        all_verts_list = self.ALL_SELECTED_VERTICES
        hand_verts_set = set(opt.LEFT_HAND_VERTEX + opt.RIGHT_HAND_VERTEX)
        hand_vtx_indices_list = [i for i, v_id in enumerate(all_verts_list) if v_id in hand_verts_set]
        self.hand_vertex_indices = torch.tensor(hand_vtx_indices_list, device=self.device, dtype=torch.long)

    def _split_components(self, x_flat: torch.Tensor):
        # [B, T, J*12] -> 4x [B, T, J, 3]
        b, t, _ = x_flat.shape
        x = x_flat.view(b, t, self.opt.joints_num, self.opt.per_joint_dim)
        pose = x[..., 0:3]
        ric = x[..., 3:6]
        rot_vel = x[..., 6:9]
        ric_vel = x[..., 9:12]
        return pose, ric, rot_vel, ric_vel

    def _split_smplx_local(self, x_j3: torch.Tensor):
        # x_j3: [N, 43, 3]
        body = x_j3[:, :13]
        lhand = x_j3[:, 13:28]
        rhand = x_j3[:, 28:43]
        restored = torch.zeros(x_j3.shape[0], 22, 3, device=x_j3.device, dtype=x_j3.dtype)
        restored[:, self.body_indices] = body
        return restored[:, 1:], lhand, rhand

    def _component_loss(self, pred: torch.Tensor, gt: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        # pred/gt: [B, T, J, 3]
        valid_pred = pred[mask_bt]
        valid_gt = gt[mask_bt]
        err = self.recon_criterion(valid_pred, valid_gt)
        return err.mean()

    def _mesh_loss(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        if self.smplx_model is None:
            return torch.zeros((), device=self.device)

        b, t, j, c = pred_pose.shape
        n = b * t
        all_pred = pred_pose.reshape(n, j, c).contiguous()
        all_gt = gt_pose.reshape(n, j, c).contiguous()

        gt_body, gt_lh, gt_rh = self._split_smplx_local(all_gt)
        pd_body, pd_lh, pd_rh = self._split_smplx_local(all_pred)

        with torch.no_grad():
            out_gt = self.smplx_model(body_pose=gt_body, left_hand_pose=gt_lh, right_hand_pose=gt_rh)
        out_pd = self.smplx_model(body_pose=pd_body, left_hand_pose=pd_lh, right_hand_pose=pd_rh)

        verts_gt = out_gt.vertices[:, self.ALL_SELECTED_VERTICES, :].reshape(b, t, -1, 3)
        verts_pd = out_pd.vertices[:, self.ALL_SELECTED_VERTICES, :].reshape(b, t, -1, 3)

        err = self.mesh_criterion(verts_pd[mask_bt], verts_gt[mask_bt])

        if getattr(self.opt, "finger_loss_weight", 1.0) != 1.0:
            w = torch.ones(err.shape[1], device=self.device)
            w[self.hand_vertex_indices] = self.opt.finger_loss_weight
            err = err * w.view(1, -1, 1)

        return err.mean()

    def train_forward(self, batch_data, epoch: int):
        motion, lengths = batch_data
        motion = motion.to(self.device)
        lengths = lengths.to(self.device)

        b, t, _ = motion.shape
        mask = (torch.arange(t, device=self.device)[None, :] < lengths[:, None])

        pred_motion, loss_dict = self.vae(motion)

        gt_pose, gt_ric, gt_rot_vel, gt_ric_vel = self._split_components(motion)
        pd_pose, pd_ric, pd_rot_vel, pd_ric_vel = self._split_components(pred_motion)

        loss_pose = self._component_loss(pd_pose, gt_pose, mask)
        loss_ric = self._component_loss(pd_ric, gt_ric, mask)
        loss_rot_vel = self._component_loss(pd_rot_vel, gt_rot_vel, mask)
        loss_ric_vel = self._component_loss(pd_ric_vel, gt_ric_vel, mask)

        loss_recon = (
            self.opt.lambda_recon_pose * loss_pose
            + self.opt.lambda_recon_ric * loss_ric
            + self.opt.lambda_recon_rot_vel * loss_rot_vel
            + self.opt.lambda_recon_ric_vel * loss_ric_vel
        )

        loss_mesh = self._mesh_loss(pd_pose, gt_pose, mask)
        loss_kl = loss_dict["loss_kl"]

        total = loss_recon + self.opt.lambda_mesh * loss_mesh + self.opt.lambda_kl * loss_kl

        metrics = OrderedDict()
        metrics["loss_total"] = total.detach()
        metrics["loss_recon"] = loss_recon.detach()
        metrics["loss_pose"] = loss_pose.detach()
        metrics["loss_ric"] = loss_ric.detach()
        metrics["loss_rot_vel"] = loss_rot_vel.detach()
        metrics["loss_ric_vel"] = loss_ric_vel.detach()
        metrics["loss_mesh"] = loss_mesh.detach()
        metrics["loss_kl"] = loss_kl.detach()
        return total, metrics

    def save(self, file_name, epoch, total_iter, optim, scheduler):
        state = {
            "vae": self.vae.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)

    def resume(self, model_dir, optim, scheduler):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vae.load_state_dict(checkpoint["vae"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["epoch"], checkpoint["total_iter"]

    def _log_metrics(self, prefix: str, metrics: OrderedDict, step: int):
        if self.logger is None:
            return
        for k, v in metrics.items():
            if torch.is_tensor(v):
                scalar = float(v.detach().mean().item())
            else:
                scalar = float(v)
            self.logger.add_scalar(f"{prefix}/{k}", scalar, step)

    def train(self, train_loader, val_loader):
        self.vae.to(self.device)
        optim = torch.optim.AdamW(
            self.vae.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch0 = 0
        total_iter = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            if os.path.exists(model_dir):
                epoch0, total_iter = self.resume(model_dir, optim, scheduler)
                print(f"[{_now()}] Resume from epoch={epoch0}, iter={total_iter}")

        best_val = float("inf")
        start = time.time()

        for epoch in range(epoch0, self.opt.max_epoch):
            self.vae.train()
            running = defaultdict(float)

            for i, batch_data in enumerate(train_loader):
                total_iter += 1
                optim.zero_grad(set_to_none=True)

                with amp.autocast(device_type=self.amp_device, enabled=self.use_amp, dtype=self.amp_dtype):
                    loss, metrics = self.train_forward(batch_data, epoch)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    optim.step()

                for k, v in metrics.items():
                    running[k] += float(v.item())

                if total_iter % self.opt.log_every == 0:
                    mean_metrics = OrderedDict((k, running[k] / self.opt.log_every) for k in running.keys())
                    elapsed = time.time() - start
                    msg = " | ".join([f"{k}:{v:.4f}" for k, v in mean_metrics.items()])
                    print(f"[{_now()}] Epoch {epoch:03d} Iter {i:04d} TotalIter {total_iter} Elapsed {elapsed/60:.1f}m | {msg}")
                    self._log_metrics("train", mean_metrics, total_iter)
                    running = defaultdict(float)

                if total_iter % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, total_iter, optim, scheduler)

            scheduler.step()

            # validation
            self.vae.eval()
            val_running = defaultdict(float)
            n_val = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    loss, metrics = self.train_forward(batch_data, epoch)
                    n_val += 1
                    for k, v in metrics.items():
                        val_running[k] += float(v.item())

            if n_val > 0:
                val_mean = OrderedDict((k, val_running[k] / n_val) for k in val_running.keys())
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in val_mean.items()])
                print(f"[{_now()}] Validation epoch {epoch:03d} | {msg}")
                self._log_metrics("val", val_mean, total_iter)

                if val_mean["loss_total"] < best_val:
                    best_val = val_mean["loss_total"]
                    self.save(pjoin(self.opt.model_dir, "best.tar"), epoch, total_iter, optim, scheduler)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, total_iter, optim, scheduler)
