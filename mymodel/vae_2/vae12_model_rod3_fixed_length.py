import torch
import torch.nn as nn

from mymodel.vae_2.vae12_skeleton_rod3_fixed_length import (
    MultiLinear,
    MotionEncoder,
    MotionDecoder,
    STConvEncoder,
    STConvDecoder,
)


def _expert_num_by_dataset(dataset_name: str) -> int:
    if dataset_name == "SMPLX_SL":
        return 5
    if dataset_name == "HAND_CENTRIC":
        return 9
    if dataset_name == "HIERARCHICAL":
        return 13
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


class VAE(nn.Module):
    """VAE for 12D per-joint features.

    Feature order per joint:
    [pose(3), ric(3), rot_vel(3), ric_vel(3)]
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.joints_num = len(opt.SELECTED_JOINT_INDICES)
        self.per_joint_dim = int(getattr(opt, "per_joint_dim", 12))
        d_flat = self.joints_num * self.per_joint_dim

        # Normalization buffers (pose dims can be kept identity by stats).
        self.register_buffer("mean", torch.zeros(d_flat))
        self.register_buffer("std", torch.ones(d_flat))

        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        expert_num = _expert_num_by_dataset(opt.dataset_name)
        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, expert_num)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError(
                f"Stats shape mismatch: expected {self.mean.shape}, got mean={mean.shape}, std={std.shape}"
            )
        self.mean.copy_(mean)
        self.std.copy_(std)
        print("VAE12: loaded normalization stats into buffers.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (self.std + 1e-8)) + self.mean

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_norm: torch.Tensor):
        x = self.motion_enc(x_norm)
        x = self.conv_enc(x)
        x = self.dist(x)
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        loss_kl = 0.5 * torch.mean(mu.pow(2) + torch.exp(logvar) - logvar - 1.0)
        return z, {"loss_kl": loss_kl, "mu": mu, "logvar": logvar}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_dec(z)
        x = self.motion_dec(x)
        return x

    def forward(self, x: torch.Tensor, only_cont: bool = False, only_quant: bool = False):
        del only_quant  # VAE has no quant path; keep signature for compatibility.

        x = x.to(self.opt.device)
        x_norm = self._normalize(x)
        z, loss_dict = self.encode(x_norm)
        out_norm = self.decode(z)
        out = self._denormalize(out_norm)

        if only_cont:
            return out, z, loss_dict
        return out, loss_dict
