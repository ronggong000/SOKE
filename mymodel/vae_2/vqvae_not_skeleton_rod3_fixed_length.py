import torch
import torch.nn as nn

from mymodel.vae.qvae_skeleton_rod3_fixed_length import (
    MultiLinear,
    MotionEncoder,
    MotionDecoder,
    ResSTConv,
    STPool,
    STUnpool,
    VectorQuantizer,
    adj_list_to_edges,
)


class STPoolToggle(STPool):
    """Skeleton pooling always enabled; temporal pooling is optional."""

    def __init__(self, joint_selection=None, depth=0, temporal_downsample=True):
        super().__init__(joint_selection=joint_selection, depth=depth)
        self.temporal_downsample = temporal_downsample

    def forward(self, x):
        # x: [B, T, J, D]
        b, t, _, d = x.size()
        out = torch.matmul(self.skeleton_pool, x)  # [B, T, J_out, D]
        if not self.temporal_downsample:
            return out

        j_out = out.size(2)
        out = out.permute(0, 2, 3, 1).reshape(b * j_out, d, t)
        out = self.temporal_pool(out)
        out = out.reshape(b, j_out, d, -1).permute(0, 3, 1, 2)
        return out


class STUnpoolToggle(STUnpool):
    """Skeleton unpooling always enabled; temporal upsampling is optional."""

    def __init__(self, skeleton_mapping, temporal_downsample=True):
        super().__init__(skeleton_mapping=skeleton_mapping)
        self.temporal_downsample = temporal_downsample

    def forward(self, x):
        # x: [B, T, J_in, D]
        b, t, _, d = x.size()
        out = torch.matmul(self.skeleton_unpool, x)  # [B, T, J_out, D]
        if not self.temporal_downsample:
            return out

        j_out = out.size(2)
        out = out.permute(0, 2, 3, 1).reshape(b * j_out, d, t)
        out = self.temporal_unpool(out)
        out = out.reshape(b, j_out, d, -1).permute(0, 3, 1, 2)
        return out


class STConvEncoderNoTime(nn.Module):
    def __init__(self, opt):
        super().__init__()

        temporal_downsample = bool(getattr(opt, "temporal_downsample", False))

        self.edge_list = [adj_list_to_edges(opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST)]
        self.mapping_list = []
        self.layers = nn.ModuleList()

        for i in range(opt.n_layers):
            block = nn.ModuleList()
            for _ in range(opt.n_extra_layers):
                block.append(
                    ResSTConv(
                        self.edge_list[-1],
                        opt.latent_dim,
                        opt.kernel_size,
                        activation=opt.activation,
                        norm=opt.norm,
                        dropout=opt.dropout,
                    )
                )
            block.append(
                ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout,
                )
            )
            pool = STPoolToggle(opt.dataset_name, i, temporal_downsample=temporal_downsample)
            block.append(pool)
            self.layers.append(block)
            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


class STConvDecoderNoTime(nn.Module):
    def __init__(self, opt, encoder: STConvEncoderNoTime):
        super().__init__()

        temporal_downsample = bool(getattr(opt, "temporal_downsample", False))

        self.layers = nn.ModuleList()
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for _ in range(opt.n_layers):
            block = nn.ModuleList()
            block.append(
                STUnpoolToggle(
                    skeleton_mapping=mapping_list.pop(),
                    temporal_downsample=temporal_downsample,
                )
            )
            edge_list.pop()
            for _ in range(opt.n_extra_layers):
                block.append(
                    ResSTConv(
                        edge_list[-1],
                        opt.latent_dim,
                        opt.kernel_size,
                        activation=opt.activation,
                        norm=opt.norm,
                        dropout=opt.dropout,
                    )
                )
            block.append(
                ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout,
                )
            )
            self.layers.append(block)

    def forward(self, x):
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


__all__ = [
    "MultiLinear",
    "MotionEncoder",
    "MotionDecoder",
    "VectorQuantizer",
    "STConvEncoderNoTime",
    "STConvDecoderNoTime",
]
