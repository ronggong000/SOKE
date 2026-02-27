import os
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import torch
import torch.nn as nn
from torch import amp
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import smplx


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class VQVAETrainer:
    def __init__(self, opt, model, scaler=None):
        self.opt = opt
        self.model = model
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

    def _split_smplx_local(self, x_j3: torch.Tensor):
        # x_j3: [N, 43, 3]
        body = x_j3[:, :13]
        lhand = x_j3[:, 13:28]
        rhand = x_j3[:, 28:43]
        restored = torch.zeros(x_j3.shape[0], 22, 3, device=x_j3.device, dtype=x_j3.dtype)
        restored[:, self.body_indices] = body
        return restored[:, 1:], lhand, rhand

    def _mesh_loss(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        if self.smplx_model is None:
            return torch.zeros((), device=self.device)

        b, t, d = pred_motion.shape
        n = b * t
        all_pred = pred_motion.reshape(n, self.opt.joints_num, 3).contiguous()
        all_gt = gt_motion.reshape(n, self.opt.joints_num, 3).contiguous()

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
        del epoch
        motion, lengths = batch_data
        motion = motion.to(self.device)
        lengths = lengths.to(self.device)

        b, t, _ = motion.shape
        mask = (torch.arange(t, device=self.device)[None, :] < lengths[:, None])

        out_quant, z_quant, loss_dict = self.model(motion, only_quant=True)

        rec_err = self.recon_criterion(out_quant[mask], motion[mask]).mean()
        mesh_err = self._mesh_loss(out_quant, motion, mask)
        quant_err = loss_dict.get("loss_quant", torch.zeros((), device=self.device))

        total = self.opt.lambda_recon * rec_err + self.opt.lambda_q_recon * mesh_err + self.opt.lambda_quant * quant_err

        metrics = OrderedDict()
        metrics["loss_total"] = total.detach()
        metrics["loss_recon"] = rec_err.detach()
        metrics["loss_mesh_quant"] = mesh_err.detach()
        metrics["loss_quant"] = quant_err.detach() if torch.is_tensor(quant_err) else torch.tensor(quant_err)

        # occasional dead-code reset
        if self.model.training and torch.rand(1).item() < 0.02:
            with torch.no_grad():
                z_curr = self.model.encode(motion)
                n_reset, _ = self.model.reset_all_codebooks(z_curr)
                metrics["code_reset"] = torch.tensor(float(n_reset), device=self.device)

        return total, metrics

    def save(self, file_name, epoch, total_iter, optim, scheduler):
        state = {
            "model": self.model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)

    def resume(self, model_dir, optim, scheduler):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
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
        self.model.to(self.device)

        optim = torch.optim.AdamW(
            self.model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay
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
            self.model.train()
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
            self.model.eval()
            val_running = defaultdict(float)
            n_val = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    _, metrics = self.train_forward(batch_data, epoch)
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
