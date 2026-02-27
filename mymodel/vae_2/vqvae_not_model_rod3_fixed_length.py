import torch
import torch.nn as nn

from mymodel.vae_2.vqvae_not_skeleton_rod3_fixed_length import (
    MotionEncoder,
    MotionDecoder,
    STConvEncoderNoTime,
    STConvDecoderNoTime,
    VectorQuantizer,
)


class VQVAE(nn.Module):
    """Pure VQ-VAE without temporal downsampling.

    - Input/Output: poses only, shape [B, T, J*3]
    - Latent tokens: one token per hierarchical node per frame
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt.latent_dim

        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)
        self.conv_enc = STConvEncoderNoTime(opt)
        self.conv_dec = STConvDecoderNoTime(opt, self.conv_enc)

        if opt.dataset_name != "HIERARCHICAL":
            raise ValueError("vqvae_not currently expects dataset_name='HIERARCHICAL' (13 latent nodes).")

        self.cb_size_body = getattr(opt, "codebook_size_body", 96)
        self.cb_size_hand = getattr(opt, "codebook_size_hand", 192)
        self.commitment_cost = getattr(opt, "commitment_cost", 0.25)

        self.quantizers = nn.ModuleList()
        self.grouping_schedule = self._setup_quantizers(opt.codebook_grouping)

    def _setup_quantizers(self, strategy):
        schedule = []

        def add_group(name, node_ids, size):
            q = VectorQuantizer(size, self.latent_dim, self.commitment_cost)
            self.quantizers.append(q)
            q_idx = len(self.quantizers) - 1
            schedule.append({"name": name, "ids": node_ids, "q_idx": q_idx})

        if strategy == "default":
            add_group("body_arms", [0, 1, 2], self.cb_size_body)
            add_group("shared_hands", list(range(3, 13)), self.cb_size_hand)
        elif strategy == "arm_mirror":
            add_group("torso", [0], self.cb_size_body)
            add_group("shared_arms", [1, 2], self.cb_size_body)
            add_group("shared_hands", list(range(3, 13)), self.cb_size_hand)
        elif strategy == "thumb_sep":
            add_group("torso", [0], self.cb_size_body)
            add_group("shared_arms", [1, 2], self.cb_size_body)
            add_group("shared_thumbs", [7, 12], self.cb_size_hand)
            add_group("shared_fingers", [3, 4, 5, 6, 8, 9, 10, 11], self.cb_size_hand)
        elif strategy == "finger_distinct":
            add_group("torso", [0], self.cb_size_body)
            add_group("shared_arms", [1, 2], self.cb_size_body)
            add_group("idx", [3, 8], self.cb_size_hand)
            add_group("mid", [4, 9], self.cb_size_hand)
            add_group("pnk", [5, 10], self.cb_size_hand)
            add_group("rng", [6, 11], self.cb_size_hand)
            add_group("tmb", [7, 12], self.cb_size_hand)
        elif strategy == "full_book":
            add_group("node_0_torso", [0], self.cb_size_body)
            add_group("node_1_larm", [1], self.cb_size_body)
            add_group("node_2_rarm", [2], self.cb_size_body)
            for i, name in enumerate(["l_idx", "l_mid", "l_pnk", "l_rng", "l_tmb"]):
                add_group(name, [3 + i], self.cb_size_hand)
            for i, name in enumerate(["r_idx", "r_mid", "r_pnk", "r_rng", "r_tmb"]):
                add_group(name, [8 + i], self.cb_size_hand)
        else:
            raise ValueError(f"Unknown codebook grouping strategy: {strategy}")

        print(f"Codebook Strategy [{strategy}]: Created {len(self.quantizers)} quantizers.")
        return schedule

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.motion_enc(x)
        x = self.conv_enc(x)
        return x

    def decode(self, z):
        z = self.conv_dec(z)
        z = self.motion_dec(z)
        return z

    def _quantize(self, z_cont):
        z_quant = torch.zeros_like(z_cont)
        loss_dict = {}
        total_quant_loss = 0.0

        for group in self.grouping_schedule:
            name = group["name"]
            ids = group["ids"]
            q_idx = group["q_idx"]
            quantizer = self.quantizers[q_idx]

            z_slice = z_cont[:, :, ids, :]
            loss, z_q_slice, perp, idx = quantizer(z_slice)
            z_quant[:, :, ids, :] = z_q_slice

            total_quant_loss = total_quant_loss + loss
            loss_dict[f"perplexity_{name}"] = perp
            if not self.training:
                loss_dict[f"indices_{name}"] = idx

        loss_dict["loss_quant"] = total_quant_loss
        return z_quant, loss_dict

    def forward(self, x, only_cont: bool = False, only_quant: bool = False):
        x = x.to(self.opt.device)
        z_cont = self.encode(x)

        if only_cont:
            out_cont = self.decode(z_cont)
            return out_cont, z_cont, {"loss_quant": torch.zeros((), device=x.device)}

        z_quant, loss_dict = self._quantize(z_cont)
        out_quant = self.decode(z_quant)

        if only_quant:
            return out_quant, z_quant, loss_dict

        out_cont = self.decode(z_cont)
        return out_cont, out_quant, z_cont, z_quant, loss_dict

    @torch.no_grad()
    def encode_to_tokens(self, x):
        x = x.to(self.opt.device)
        z_cont = self.encode(x)
        tokens = torch.zeros(
            z_cont.shape[0],
            z_cont.shape[1],
            z_cont.shape[2],
            device=z_cont.device,
            dtype=torch.long,
        )

        for group in self.grouping_schedule:
            ids = group["ids"]
            q = self.quantizers[group["q_idx"]]
            z_slice = z_cont[:, :, ids, :]
            _, _, _, idx = q(z_slice)
            tokens[:, :, ids] = idx

        return tokens

    @torch.no_grad()
    def decode_from_tokens(self, tokens):
        b, t, k = tokens.shape
        if k != 13:
            raise ValueError(f"Expected 13 tokens/frame for HIERARCHICAL, got {k}.")

        z_quant = torch.zeros(b, t, k, self.latent_dim, device=tokens.device)

        for group in self.grouping_schedule:
            ids = group["ids"]
            q = self.quantizers[group["q_idx"]]
            emb_weight = q._embedding.weight
            group_tokens = tokens[..., ids]
            group_vectors = torch.nn.functional.embedding(group_tokens, emb_weight)
            z_quant[..., ids, :] = group_vectors

        motion = self.decode(z_quant)
        return motion

    def reset_all_codebooks(self, z_current):
        reset_stats = {}
        total_resets = 0

        for group in self.grouping_schedule:
            name = group["name"]
            ids = group["ids"]
            q = self.quantizers[group["q_idx"]]
            z_slice = z_current[:, :, ids, :]
            z_pool = z_slice.reshape(-1, z_slice.shape[-1])
            n_reset = q.reset_codebook(z_pool)
            if n_reset > 0:
                reset_stats[name] = n_reset
                total_resets += n_reset

        return total_resets, reset_stats
