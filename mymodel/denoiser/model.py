"""
This code was inspired by the denoiser implementation in the Motion Latent Diffusion
    - https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/architectures/mld_denoiser.py
"""

from typing import List
import torch
import torch.nn as nn

from models.denoiser.clip import FrozenCLIPTextEncoder
from models.denoiser.embedding import TimestepEmbedding, PositionalEmbedding
from models.denoiser.transformer import SkipTransformer

class InputProcess(nn.Module):
    def __init__(self, opt, in_features):
        super(InputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim),
        )

    def forward(self, x):
        return self.layers(x)

class OutputProcess(nn.Module):
    def __init__(self, opt, out_features):
        super(OutputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(opt.latent_dim),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, out_features),
        )
    
    def forward(self, x):
        return self.layers(x)

class Denoiser(nn.Module):
    def __init__(self, opt, vae_dim):
        super(Denoiser, self).__init__()

        self.opt = opt
        self.latent_dim = opt.latent_dim
        

        # input & output process
        self.input_process = InputProcess(opt, vae_dim)
        self.output_process = OutputProcess(opt, vae_dim)
        
        # timestep embedding
        self.timestep_emb = TimestepEmbedding(self.latent_dim)

        # CLIP/T5 text encoder
        # self.clip_model = FrozenCLIPTextEncoder(opt)
        # self.word_emb = nn.Linear(self.clip_dim, self.latent_dim)
        # self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768 # ViT-L/14
        self.use_precomputed_text_emb = bool(getattr(opt, "use_precomputed_text_emb", False))

        if self.use_precomputed_text_emb:
            # 例如 T5-large: 1024
            self.text_in_dim = int(getattr(opt, "text_emb_dim", 1024))
            self.clip_model = None  # 不加载 CLIP
        else:
            self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768  # 你的旧逻辑
            self.text_in_dim = self.clip_dim
            self.clip_model = FrozenCLIPTextEncoder(opt)

        # 不管来自哪种 encoder，统一投到 latent_dim
        self.word_emb = nn.Linear(self.text_in_dim, self.latent_dim)

        # cache（仅对 CLIP 模式有意义；embedding 模式一般不需要 cache）
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None
        # positional embedding
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)

        # transformer
        self.transformer = SkipTransformer(opt)

        # ===== RAG / blueprint encoder（新增）=====
        self.use_rag = bool(getattr(opt, "use_rag", False))
        self.rag_K = int(getattr(opt, "rag_K", 13))
        self.rag_layers = int(getattr(opt, "rag_layers", 2))
        self.rag_heads = int(getattr(opt, "rag_heads", 8))
        self.rag_max_T = int(getattr(opt, "rag_max_T", 384))

        # codebook_sizes 可能由 trainer 在运行时塞进来：self._rag_codebook_sizes = [...]
        self._rag_inited = False
        self._rag_codebook_sizes = getattr(opt, "rag_codebook_sizes", None)

        self.rag_token_embs = None
        self.rag_slot_emb = None
        self.rag_fuse = None
        self.rag_pos = None
        self.rag_encoder = None
        self.rag_ln = None

    def _maybe_init_rag(self, device):
        if (not self.use_rag) or self._rag_inited:
            return
        if self._rag_codebook_sizes is None:
            raise ValueError(
                "use_rag=True but rag_codebook_sizes is None. "
                "Set opt.rag_codebook_sizes or let trainer set denoiser._rag_codebook_sizes before first forward."
            )

        codebook_sizes = list(self._rag_codebook_sizes)
        if len(codebook_sizes) < self.rag_K:
            raise ValueError(f"rag_codebook_sizes length={len(codebook_sizes)} < rag_K={self.rag_K}")
        codebook_sizes = codebook_sizes[: self.rag_K]

        # 每个 slot 一个 embedding：vocab = codebook_size + 2（[0..cb-1]=正常, cb=mask, cb+1=pad）
        self.rag_token_embs = nn.ModuleList([
            nn.Embedding(int(cb) + 2, self.latent_dim) for cb in codebook_sizes
        ]).to(device)

        self.rag_slot_emb = nn.Embedding(self.rag_K, self.latent_dim).to(device)
        self.rag_fuse = nn.Linear(self.rag_K * self.latent_dim, self.latent_dim).to(device)
        self.rag_pos = nn.Embedding(self.rag_max_T, self.latent_dim).to(device)

        # ✅ rag_layers == 0：不创建 TransformerEncoder（否则 forward 会 IndexError）
        if self.rag_layers < 0:
            raise ValueError(f"rag_layers must be >= 0, got {self.rag_layers}")

        if self.rag_layers == 0:
            self.rag_encoder = None
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.rag_heads,
                dim_feedforward=4 * self.latent_dim,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.rag_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.rag_layers).to(device)

        self.rag_ln = nn.LayerNorm(self.latent_dim).to(device)
        self._rag_inited = True


    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if "clip_model" not in name]
    
    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        remove_weights = [e for e in state_dict.keys() if "clip_model." in e or "_cache_" in e]
        for e in remove_weights:
            del state_dict[e]
        return state_dict
    
    def remove_clip_cache(self):
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None
    def forward(
        self, x, timestep_emb, text, len_mask=None, need_attn=False,
        fixed_sa=None, fixed_ta=None, fixed_ca=None, use_cached_clip=False,
        blueprint_tokens=None, blueprint_pad_mask=None
    ):
        """
        x: [B, T, J, D]
        timestep_emb: [B] or [1]
        text:
        - List[str]  (旧模式：CLIP)
        - (text_emb, text_mask) (新模式：预提取 embedding)
            text_emb: [B, L, D_text], text_mask: [B, L] bool, True=valid
        """

        # input process
        x = self.input_process(x)
        B, T, J, D = x.size()

        # diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # ===== text embedding（兼容两种输入）=====
        if isinstance(text, tuple) and len(text) >= 2 and torch.is_tensor(text[0]):
            text_emb, text_mask = text[0], text[1]
            word_emb = self.word_emb(text_emb.to(device=x.device, dtype=x.dtype))
            ca_mask = text_mask.to(device=x.device, dtype=torch.bool)  # True=valid token
        else:
            if not hasattr(self, "clip_model") or self.clip_model is None:
                word_emb = torch.zeros((B, 1, D), device=x.device, dtype=x.dtype)
                ca_mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
            else:
                if use_cached_clip and all(e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]):
                    word_emb = self._cache_word_emb
                    ca_mask = self._cache_ca_mask
                else:
                    word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)  # ca_mask: True=valid
                    word_emb = self.word_emb(word_emb)
                    if use_cached_clip:
                        self._cache_word_emb = word_emb
                        self._cache_ca_mask = ca_mask
                        self._cache_tokens_pos = token_pos

        # ===== RAG blueprint：embed/fuse -> (可选) rag_encoder -> concat 到 cross-attn memory =====
        if self.use_rag and blueprint_tokens is not None:
            self._maybe_init_rag(device=x.device)

            bp = blueprint_tokens.to(device=x.device, dtype=torch.long)  # [B, Tb, K]
            Bb, Tb, K = bp.shape
            if K != self.rag_K:
                raise ValueError(f"blueprint_tokens K={K} != rag_K={self.rag_K}")

            if blueprint_pad_mask is None:
                blueprint_pad_mask = torch.zeros((Bb, Tb), device=x.device, dtype=torch.bool)
            else:
                blueprint_pad_mask = blueprint_pad_mask.to(device=x.device, dtype=torch.bool)

            # slot-wise embedding + slot id embedding
            slot_ids = torch.arange(self.rag_K, device=x.device, dtype=torch.long)
            slot_add = self.rag_slot_emb(slot_ids).view(1, 1, self.rag_K, D)  # [1,1,K,D]

            bp_slots = []
            for k in range(self.rag_K):
                ek = self.rag_token_embs[k](bp[:, :, k])  # [B,Tb,D]
                bp_slots.append(ek)
            bp_slots = torch.stack(bp_slots, dim=2) + slot_add  # [B,Tb,K,D]

            bp_h = self.rag_fuse(bp_slots.reshape(Bb, Tb, self.rag_K * D))  # [B,Tb,D]

            # pos
            if Tb > self.rag_max_T:
                bp_h = bp_h + self.rag_pos.weight[:Tb].unsqueeze(0)
            else:
                pos_ids = torch.arange(Tb, device=x.device, dtype=torch.long)
                bp_h = bp_h + self.rag_pos(pos_ids).unsqueeze(0)

            # ✅ rag_layers==0：不走 TransformerEncoder；rag_layers>0 才走
            if self.rag_encoder is not None:
                bp_h = self.rag_encoder(bp_h, src_key_padding_mask=blueprint_pad_mask)  # [B,Tb,D]

            bp_h = self.rag_ln(bp_h)

            # concat into cross-attn keys/values
            word_emb = torch.cat([word_emb, bp_h.to(dtype=word_emb.dtype)], dim=1)  # [B, L+Tb, D]
            bp_valid = ~blueprint_pad_mask
            ca_mask = torch.cat([ca_mask, bp_valid], dim=1)  # True=valid

        # positional embedding
        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        # ===== attention masks =====
        # 约定：len_mask True=valid, False=pad
        if len_mask is not None:
            if len_mask.dtype != torch.bool:
                len_mask = len_mask.to(torch.bool)
            if len_mask.ndim != 2 or len_mask.shape[0] != B or len_mask.shape[1] != T:
                raise ValueError(f"len_mask must be [B,T]={B,T}, got {tuple(len_mask.shape)}")

            # ✅ 给 temporal-attn 用：pad mask shape [B*J, T]，避免 T*J vs T 冲突
            sa_pad_mask = (~len_mask).repeat_interleave(J, dim=0)  # [B*J, T]
        else:
            sa_pad_mask = None

        # transformer
        x, attn_weights = self.transformer.forward(
            x,
            timestep_emb,
            word_emb,
            sa_mask=sa_pad_mask,     # True=mask掉（padding）
            ca_mask=~ca_mask,        # True=mask掉（padding）
            need_attn=need_attn,
            fixed_sa=fixed_sa,
            fixed_ta=fixed_ta,
            fixed_ca=fixed_ca
        )

        # output process
        x = self.output_process(x)
        return x, attn_weights
