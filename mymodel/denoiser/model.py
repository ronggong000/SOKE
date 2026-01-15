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

        # cache for CLIP embedding
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None
    
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

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False,
            fixed_sa=None, fixed_ta=None, fixed_ca=None, use_cached_clip=False):
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
        # 情况 A：预提取 embedding
        if isinstance(text, tuple) and len(text) == 2 and torch.is_tensor(text[0]):
            text_emb, text_mask = text
            word_emb = self.word_emb(text_emb.to(device=x.device, dtype=x.dtype))
            ca_mask = text_mask.to(device=x.device)  # True=valid token

        # 情况 B：字符串 caption（只有在 clip_model 存在时可用）
        else:
            if not hasattr(self, "clip_model") or self.clip_model is None:
                # 你现在是“跳过 encoder 以加速”，如果 eval 还传字符串，这里直接退化成 uncond（不炸训练）
                # uncond：给一个长度=1的空 token
                word_emb = torch.zeros((B, 1, D), device=x.device, dtype=x.dtype)
                ca_mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
            else:
                if use_cached_clip and all(e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]):
                    word_emb = self._cache_word_emb
                    ca_mask = self._cache_ca_mask
                    token_pos = self._cache_tokens_pos
                else:
                    word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)  # ca_mask: True=valid
                    word_emb = self.word_emb(word_emb)
                    if use_cached_clip:
                        self._cache_word_emb = word_emb
                        self._cache_ca_mask = ca_mask
                        self._cache_tokens_pos = token_pos

        # positional embedding
        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        # attention masks
        if len_mask is not None:
            # [B, T] -> [B, T*J] (repeat for joints)
            len_mask = len_mask.repeat_interleave(J, dim=0)

        # transformer
        x, attn_weights = self.transformer.forward(
            x,
            timestep_emb,
            word_emb,
            sa_mask=None if len_mask is None else ~len_mask,
            ca_mask=~ca_mask,  # 注意：transformer 里是“True=mask掉”
            need_attn=need_attn,
            fixed_sa=fixed_sa,
            fixed_ta=fixed_ta,
            fixed_ca=fixed_ca
        )

        # output process
        x = self.output_process(x)
        return x, attn_weights