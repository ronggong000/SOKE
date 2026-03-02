"""
Denoiser V3:
- No CLIP path.
- No text projection linear (word_emb).
- Condition memory is unified: [NULL] + gloss tokens (+ rag tokens if provided).
"""

import json
import os
from typing import List

import torch
import torch.nn as nn

from models.denoiser.embedding import PositionalEmbedding, TimestepEmbedding
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
        self.latent_dim = int(opt.latent_dim)

        self.input_process = InputProcess(opt, vae_dim)
        self.output_process = OutputProcess(opt, vae_dim)

        self.timestep_emb = TimestepEmbedding(self.latent_dim)
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)
        self.transformer = SkipTransformer(opt)

        # V3 hard rule: always use gloss vocabulary tokens.
        self.use_gloss_tokens = True
        self.gloss_embed_mode = "vocab"

        self.gloss_pad_id = int(getattr(opt, "gloss_pad_id", 0))
        self.gloss_unk_id = int(getattr(opt, "gloss_unk_id", 1))
        self.gloss_bos_id = int(getattr(opt, "gloss_bos_id", 2))
        self.gloss_eos_id = int(getattr(opt, "gloss_eos_id", 3))

        vocab_path = str(getattr(opt, "gloss_vocab_path", "") or "").strip()
        if not vocab_path or (not os.path.isfile(vocab_path)):
            raise ValueError(
                "V3 requires gloss vocab file. Set opt.gloss_vocab_path to an existing json file."
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        stoi = obj.get("stoi", obj) if isinstance(obj, dict) else obj
        if not isinstance(stoi, dict):
            raise ValueError(
                "gloss_vocab_path must be a dict json or {'stoi': {...}} json."
            )

        self._gloss_stoi = stoi
        vsize_opt = getattr(opt, "gloss_vocab_size", None)
        if isinstance(vsize_opt, str):
            vv = vsize_opt.strip().lower()
            if vv in ("", "none", "null"):
                vsize_opt = None
        if vsize_opt is None:
            self._gloss_vocab_size = int(len(stoi))
        else:
            try:
                vsize_opt = int(vsize_opt)
            except Exception:
                vsize_opt = int(len(stoi))
            self._gloss_vocab_size = int(len(stoi)) if vsize_opt <= 0 else vsize_opt
        self.gloss_emb = nn.Embedding(self._gloss_vocab_size, self.latent_dim)
        self.gloss_ln = nn.LayerNorm(self.latent_dim)
        self.gloss_use_positional = bool(getattr(opt, "gloss_use_positional", False))
        self.gloss_max_tokens = max(8, int(getattr(opt, "gloss_max_tokens", 512)))
        if self.gloss_use_positional:
            self.gloss_pos_emb = nn.Embedding(self.gloss_max_tokens, self.latent_dim)
        else:
            self.gloss_pos_emb = None

        self.gloss_layers = int(getattr(opt, "gloss_layers", 0))
        self.gloss_heads = int(getattr(opt, "gloss_heads", 8))
        if self.gloss_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.gloss_heads,
                dim_feedforward=4 * self.latent_dim,
                dropout=float(getattr(opt, "dropout", 0.1)),
                batch_first=True,
                activation="gelu",
            )
            self.gloss_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.gloss_layers)
        else:
            self.gloss_encoder = None

        # A learnable null condition token for stable CFG/unconditional branch.
        self.null_cond_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        self.use_cond_film = bool(getattr(opt, "use_cond_film", True))
        if self.use_cond_film:
            self.cond_proj = nn.Linear(self.latent_dim, self.latent_dim)
            self.cond_gate = nn.Parameter(torch.tensor(0.0))

        self.enable_length_cond = bool(getattr(opt, "enable_length_cond", False))
        self.length_cond_max_len = max(8, int(getattr(opt, "length_cond_max_len", 1024)))
        self.length_cond_as_token = bool(getattr(opt, "length_cond_as_token", True))
        if self.enable_length_cond:
            self.length_cond_emb = nn.Embedding(self.length_cond_max_len + 1, self.latent_dim)
            self.length_cond_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.length_cond_emb = None
            self.length_cond_gate = None

        self.use_rag = bool(getattr(opt, "use_rag", False))
        self.rag_K = int(getattr(opt, "rag_K", 13))
        self.rag_layers = int(getattr(opt, "rag_layers", 2))
        self.rag_heads = int(getattr(opt, "rag_heads", 8))
        self.rag_max_T = int(getattr(opt, "rag_max_T", 384))
        self.rag_weight_gate_scale = float(getattr(opt, "rag_weight_gate_scale", 1.0))

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
            raise ValueError(
                f"rag_codebook_sizes length={len(codebook_sizes)} < rag_K={self.rag_K}"
            )
        codebook_sizes = codebook_sizes[: self.rag_K]

        self.rag_token_embs = nn.ModuleList(
            [nn.Embedding(int(cb) + 2, self.latent_dim) for cb in codebook_sizes]
        ).to(device)
        self.rag_slot_emb = nn.Embedding(self.rag_K, self.latent_dim).to(device)
        self.rag_fuse = nn.Linear(self.rag_K * self.latent_dim, self.latent_dim).to(device)
        self.rag_pos = nn.Embedding(self.rag_max_T, self.latent_dim).to(device)

        if self.rag_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.rag_heads,
                dim_feedforward=4 * self.latent_dim,
                dropout=float(getattr(self.opt, "dropout", 0.1)),
                batch_first=True,
                activation="gelu",
            )
            self.rag_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.rag_layers).to(device)
        else:
            self.rag_encoder = None

        self.rag_ln = nn.LayerNorm(self.latent_dim).to(device)
        self._rag_inited = True

    def parameters_without_clip(self):
        return list(self.parameters())

    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        remove_weights = [
            k
            for k in state_dict.keys()
            if k.startswith("clip_model.") or k.startswith("word_emb.") or "_cache_" in k
        ]
        for k in remove_weights:
            del state_dict[k]
        return state_dict

    def remove_clip_cache(self):
        # Kept for interface compatibility with trainer/inference code.
        return

    def _split_text_gloss_list(self, text_list):
        """text_list: List[str] or List[[eng, gloss]] -> (eng_list, gloss_list)."""
        eng, gloss = [], []
        for t in text_list:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                eng.append("" if t[0] is None else str(t[0]))
                gloss.append("" if t[1] is None else str(t[1]))
            else:
                eng.append("" if t is None else str(t))
                gloss.append("")
        return eng, gloss

    def _tokenize_gloss(self, g: str):
        g = "" if g is None else str(g).strip()
        return g.split() if g else []

    def _encode_gloss_batch(self, gloss_texts: List[str], device):
        """Return gloss_ids [B,Lg], gloss_mask [B,Lg] with True as valid."""
        ids_list = []
        stoi = self._gloss_stoi

        for g in gloss_texts:
            toks = self._tokenize_gloss(g)

            ids = []
            if self.gloss_bos_id >= 0:
                ids.append(self.gloss_bos_id)
            for tok in toks:
                ids.append(int(stoi.get(tok, self.gloss_unk_id)))
            if self.gloss_eos_id >= 0:
                ids.append(self.gloss_eos_id)

            ids_list.append(ids)

        B = len(ids_list)
        Lg = max(1, max(len(x) for x in ids_list)) if B > 0 else 1

        gloss_ids = torch.full((B, Lg), self.gloss_pad_id, device=device, dtype=torch.long)
        gloss_mask = torch.zeros((B, Lg), device=device, dtype=torch.bool)

        for i, ids in enumerate(ids_list):
            if not ids:
                continue
            l = min(len(ids), Lg)
            gloss_ids[i, :l] = torch.tensor(ids[:l], device=device, dtype=torch.long)
            gloss_mask[i, :l] = True

        return gloss_ids, gloss_mask

    def _masked_mean(self, x, mask):
        """x:[B,L,D], mask:[B,L] True=valid -> [B,D]."""
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    def _extract_gloss_texts(self, text, batch_size: int) -> List[str]:
        """
        Accepts:
        - List[str]
        - List[[eng, gloss]]
        - (text_emb, text_mask, raw_texts, ...)
        """
        raw_texts = None
        if isinstance(text, tuple) and len(text) >= 3 and isinstance(text[2], (list, tuple)):
            raw_texts = list(text[2])
        elif isinstance(text, tuple) and len(text) >= 2 and torch.is_tensor(text[0]):
            # Legacy precomputed tuple without raw text.
            raw_texts = [""] * batch_size
        elif isinstance(text, (list, tuple)) and len(text) == batch_size:
            raw_texts = list(text)
        else:
            raw_texts = [""] * batch_size

        gloss_texts = []
        for rt in raw_texts:
            if isinstance(rt, (list, tuple)) and len(rt) >= 2:
                gloss_texts.append("" if rt[1] is None else str(rt[1]))
            else:
                gloss_texts.append("" if rt is None else str(rt))
        return gloss_texts

    def _infer_length_ids(self, len_mask, batch_size: int, time_steps: int, device):
        if len_mask is not None:
            if len_mask.dtype != torch.bool:
                len_mask = len_mask.to(torch.bool)
            if len_mask.ndim != 2 or len_mask.shape[0] != batch_size or len_mask.shape[1] != time_steps:
                raise ValueError(
                    f"len_mask must be [B,T]={batch_size,time_steps}, got {tuple(len_mask.shape)}"
                )
            lengths = len_mask.long().sum(dim=1)
        else:
            lengths = torch.full((batch_size,), int(time_steps), device=device, dtype=torch.long)
        return lengths.clamp(min=1, max=self.length_cond_max_len)

    def forward(
        self,
        x,
        timestep_emb,
        text,
        len_mask=None,
        need_attn=False,
        fixed_sa=None,
        fixed_ta=None,
        fixed_ca=None,
        use_cached_clip=False,
        blueprint_tokens=None,
        blueprint_weights=None,
        blueprint_pad_mask=None,
    ):
        del use_cached_clip  # V3: no CLIP cache path

        x = self.input_process(x)
        B, T, J, D = x.size()

        cond = self.timestep_emb(timestep_emb).expand(B, D)
        len_ids = None
        len_vec = None
        if self.enable_length_cond:
            len_ids = self._infer_length_ids(len_mask, batch_size=B, time_steps=T, device=x.device)
            len_vec = self.length_cond_emb(len_ids).to(dtype=cond.dtype)
            cond = cond + torch.sigmoid(self.length_cond_gate) * len_vec

        gloss_texts = self._extract_gloss_texts(text, B)
        gloss_ids, gloss_mask = self._encode_gloss_batch(gloss_texts, device=x.device)

        cond_mem = self.gloss_emb(gloss_ids)
        if self.gloss_pos_emb is not None:
            Lg = cond_mem.shape[1]
            pos_ids = torch.arange(Lg, device=x.device, dtype=torch.long).clamp(max=self.gloss_max_tokens - 1)
            cond_mem = cond_mem + self.gloss_pos_emb(pos_ids).unsqueeze(0).to(dtype=cond_mem.dtype)
        if self.gloss_encoder is not None:
            cond_mem = self.gloss_encoder(cond_mem, src_key_padding_mask=~gloss_mask)
        cond_mem = self.gloss_ln(cond_mem)
        cond_mask = gloss_mask

        if self.enable_length_cond and self.length_cond_as_token:
            if len_vec is None:
                len_ids = self._infer_length_ids(len_mask, batch_size=B, time_steps=T, device=x.device)
                len_vec = self.length_cond_emb(len_ids).to(dtype=cond_mem.dtype)
            len_tok = len_vec.to(dtype=cond_mem.dtype).unsqueeze(1)
            len_tok_mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
            cond_mem = torch.cat([len_tok, cond_mem], dim=1)
            cond_mask = torch.cat([len_tok_mask, cond_mask], dim=1)

        if self.use_rag and blueprint_tokens is not None:
            self._maybe_init_rag(device=x.device)

            bp = blueprint_tokens.to(device=x.device, dtype=torch.long)
            Bb, Tb, K = bp.shape
            if Bb != B:
                raise ValueError(f"blueprint batch {Bb} != motion batch {B}")
            if K != self.rag_K:
                raise ValueError(f"blueprint_tokens K={K} != rag_K={self.rag_K}")

            if blueprint_pad_mask is None:
                blueprint_pad_mask = torch.zeros((Bb, Tb), device=x.device, dtype=torch.bool)
            else:
                blueprint_pad_mask = blueprint_pad_mask.to(device=x.device, dtype=torch.bool)

            slot_ids = torch.arange(self.rag_K, device=x.device, dtype=torch.long)
            slot_add = self.rag_slot_emb(slot_ids).view(1, 1, self.rag_K, D)

            bp_slots = []
            for k in range(self.rag_K):
                bp_slots.append(self.rag_token_embs[k](bp[:, :, k]))
            bp_slots = torch.stack(bp_slots, dim=2) + slot_add

            if blueprint_weights is not None:
                bw = blueprint_weights.to(device=x.device, dtype=bp_slots.dtype)
                if bw.ndim == 2:
                    bw = bw.unsqueeze(-1)
                if bw.ndim != 3 or bw.shape[0] != Bb or bw.shape[1] != Tb:
                    raise ValueError(
                        f"blueprint_weights must be [B,Tb] or [B,Tb,K], got {tuple(bw.shape)} for batch={(Bb, Tb, self.rag_K)}"
                    )
                if bw.shape[2] == 1:
                    bw = bw.expand(Bb, Tb, self.rag_K)
                elif bw.shape[2] != self.rag_K:
                    raise ValueError(
                        f"blueprint_weights K={bw.shape[2]} != rag_K={self.rag_K}"
                    )
                gate = 1.0 + self.rag_weight_gate_scale * (2.0 * bw - 1.0)
                gate = gate.clamp(min=0.25, max=2.0)
                bp_slots = bp_slots * gate.unsqueeze(-1)

            bp_h = self.rag_fuse(bp_slots.reshape(Bb, Tb, self.rag_K * D))

            if self.rag_max_T <= 0:
                raise ValueError(f"rag_max_T must be > 0, got {self.rag_max_T}")
            pos_ids = torch.arange(Tb, device=x.device, dtype=torch.long).clamp(max=self.rag_max_T - 1)
            bp_h = bp_h + self.rag_pos(pos_ids).unsqueeze(0)

            if self.rag_encoder is not None:
                bp_h = self.rag_encoder(bp_h, src_key_padding_mask=blueprint_pad_mask)
            bp_h = self.rag_ln(bp_h)

            cond_mem = torch.cat([cond_mem, bp_h.to(dtype=cond_mem.dtype)], dim=1)
            cond_mask = torch.cat([cond_mask, ~blueprint_pad_mask], dim=1)

        null_tok = self.null_cond_token.to(dtype=cond_mem.dtype).expand(B, 1, D)
        null_mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
        cond_mem = torch.cat([null_tok, cond_mem], dim=1)
        cond_mask = torch.cat([null_mask, cond_mask], dim=1)

        if self.use_cond_film:
            pooled = self._masked_mean(cond_mem, cond_mask)
            gate = torch.sigmoid(self.cond_gate)
            cond = cond + gate * self.cond_proj(pooled.to(dtype=cond.dtype))

        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        if len_mask is not None:
            if len_mask.dtype != torch.bool:
                len_mask = len_mask.to(torch.bool)
            if len_mask.ndim != 2 or len_mask.shape[0] != B or len_mask.shape[1] != T:
                raise ValueError(f"len_mask must be [B,T]={B,T}, got {tuple(len_mask.shape)}")
            sa_pad_mask = (~len_mask).repeat_interleave(J, dim=0)
        else:
            sa_pad_mask = None

        x, attn_weights = self.transformer.forward(
            x,
            cond,
            cond_mem,
            sa_mask=sa_pad_mask,
            ca_mask=~cond_mask,
            need_attn=need_attn,
            fixed_sa=fixed_sa,
            fixed_ta=fixed_ta,
            fixed_ca=fixed_ca,
        )

        x = self.output_process(x)
        return x, attn_weights
