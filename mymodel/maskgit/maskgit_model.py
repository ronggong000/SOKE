#maskgit_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MotionCrossAttnBlock(nn.Module):
    """
    motion self-attn -> motion->text cross-attn -> FFN
    batch_first=True
    key_padding_mask: True = PAD (ignored)
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, motion_x, text_kv, motion_pad_mask, text_pad_mask):
        # self-attn on motion
        h = self.norm1(motion_x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=motion_pad_mask, need_weights=False,average_attn_weights=False)
        motion_x = motion_x + h

        # cross-attn: motion queries, text keys/values
        h = self.norm2(motion_x)
        h, _ = self.cross_attn(h, text_kv, text_kv, key_padding_mask=text_pad_mask,need_weights=False,average_attn_weights=False)
        motion_x = motion_x + h

        # ffn
        h = self.ff(self.norm3(motion_x))
        motion_x = motion_x + h
        return motion_x


class MaskGITTransformer(nn.Module):
    """
    tokens:   [B, T, K]  each slot in [0..codebook-1], PAD=(codebook+1), MASK=(codebook)
    text_emb: [B, L, text_dim]
    """
    def __init__(
        self,
        slot_names,
        codebook_sizes,
        dim=512,
        depth=8,
        heads=8,
        text_dim=1024,
        max_seq_len=4096,
        dropout=0.1,
        length_bin_size: int = 1,
        length_num_bins: int = 0,
        # --- optional: share 7 codebooks across 13 slots ---
        slot2q_idx: list = None,          # len=K, each slot -> q_idx
        q_idx_to_size: dict = None,       # q_idx -> codebook_size
        tie_groups: bool = True,
        flatten_spatiotemporal: bool = True,
        use_blueprint: bool = False,
    ):
        super().__init__()
        self.slot_names = list(slot_names)
        self.codebook_sizes = [int(x) for x in codebook_sizes]
        self.K = len(self.slot_names)
        assert self.K == len(self.codebook_sizes)

        self.dim = int(dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.text_dim = int(text_dim)
        self.max_seq_len = int(max_seq_len)
        self.flatten_spatiotemporal = bool(flatten_spatiotemporal)

        # --- group tying (recommended for 13 slots share 7 codebooks) ---
        self.use_group_tying = bool(tie_groups) and (slot2q_idx is not None) and (q_idx_to_size is not None)

        if self.use_group_tying:
            if len(slot2q_idx) != self.K:
                raise ValueError(f"slot2q_idx len={len(slot2q_idx)} != K={self.K}")
            self.slot2q_idx = [int(x) for x in slot2q_idx]

            # unique q indices may not be contiguous; remap -> gid
            q_vals = sorted(set(self.slot2q_idx))
            self.q_vals = q_vals
            self.q_to_gid = {q: i for i, q in enumerate(q_vals)}
            self.slot_gid = [self.q_to_gid[q] for q in self.slot2q_idx]  # len=K
            self.G = len(q_vals)

            # group codebook sizes
            self.q_idx_to_size = {int(k): int(v) for k, v in q_idx_to_size.items()}
            self.group_codebook_sizes = [int(self.q_idx_to_size[q]) for q in q_vals]

            # sanity: per-slot cb must match its group cb
            for k in range(self.K):
                cb_expect = int(self.group_codebook_sizes[self.slot_gid[k]])
                if int(self.codebook_sizes[k]) != cb_expect:
                    raise ValueError(
                        f"codebook_sizes[{k}]={self.codebook_sizes[k]} != group(cb of q={self.slot2q_idx[k]})={cb_expect}. "
                        f"(metadata/codebook_sizes mismatch)"
                    )

            # shared embeddings & heads per group
            self.group_token_embs = nn.ModuleList([
                nn.Embedding(cb + 2, self.dim) for cb in self.group_codebook_sizes
            ])
            self.group_heads = nn.ModuleList([
                nn.Linear(self.dim, cb + 2) for cb in self.group_codebook_sizes
            ])

            # slot identity embeddings (break symmetry while still sharing weights)
            self.slot_in_emb = nn.Embedding(self.K, self.dim)
            self.slot_out_emb = nn.Embedding(self.K, self.dim)

            # buffers for fast indexing
            self.register_buffer(
                "slot_ids",
                torch.arange(self.K, dtype=torch.long),
                persistent=False,
            )

            # list of slot indices per gid
            self.gid_to_slot_ids = []
            for gid in range(self.G):
                self.gid_to_slot_ids.append([i for i, g in enumerate(self.slot_gid) if g == gid])

        else:
            # per-slot emb and head (legacy)
            self.token_embs = nn.ModuleList()
            self.heads_dict = nn.ModuleDict()
            for name, cb in zip(self.slot_names, self.codebook_sizes):
                vocab = int(cb) + 2
                self.token_embs.append(nn.Embedding(vocab, self.dim))
                self.heads_dict[name] = nn.Linear(self.dim, vocab)
            print("Warning: MaskGITTransformer using legacy per-slot embeddings/heads (no group tying).")
            # ---- NEW: slot identity embeddings (also for legacy path) ----
            self.slot_in_emb = nn.Embedding(self.K, self.dim)
            self.slot_out_emb = nn.Embedding(self.K, self.dim)
            self.register_buffer(
                "slot_ids",
                torch.arange(self.K, dtype=torch.long),
                persistent=False,
            )
        # fuse slots -> motion dim
        self.slot_proj = None if self.flatten_spatiotemporal else nn.Linear(self.K * self.dim, self.dim)

        # text projection
        self.text_proj = nn.Linear(self.text_dim, self.dim)
        self.text_ln = nn.LayerNorm(self.dim)

        # motion pos embedding
        self.motion_pos = nn.Embedding(self.max_seq_len, self.dim)

        # cache pos ids
        self.register_buffer(
            "pos_ids",
            torch.arange(self.max_seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False
        )

        self.blocks = nn.ModuleList([
            MotionCrossAttnBlock(self.dim, self.heads, dropout=dropout)
            for _ in range(self.depth)
        ])
        self.final_ln = nn.LayerNorm(self.dim)

        # length head
        self.length_bin_size = int(length_bin_size)
        if int(length_num_bins) > 0:
            self.length_num_bins = int(length_num_bins)
        else:
            self.length_num_bins = int(math.ceil(self.max_seq_len / max(1, self.length_bin_size)))
        self.length_head = nn.Linear(self.dim, self.length_num_bins)

        self.apply(self._init_weights)

        # ===== Blueprint (RAG) encoder =====
        self.use_blueprint = use_blueprint  # 你也可以改成从外部参数控制
        self.bp_fuse = nn.Linear(self.K * self.dim, self.dim)

        bp_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=8, dim_feedforward=self.dim * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.bp_encoder = nn.TransformerEncoder(bp_layer, num_layers=3)
        self.bp_ln = nn.LayerNorm(self.dim)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    @staticmethod
    def _split_key_padding_mask(tokens, text_emb, key_padding_mask, *, flatten_spatiotemporal: bool = False, K: int = 13):
        """
        key_padding_mask: [B, L + T] (non-flatten)  or  [B, L + S] (flatten), True=PAD
        Returns:
        text_pad:  [B, L] or None
        motion_pad:[B, T] / [B, S] or None
        """
        if key_padding_mask is None:
            return None, None

        B = tokens.shape[0]
        L = text_emb.shape[1]
        # tokens may be [B,T,K] or [B,S] depending on your implementation; we only use it for sanity
        # We infer T/K from mask length rather than from tokens.

        M = int(key_padding_mask.shape[1] - L)
        if M <= 0:
            raise ValueError(f"key_padding_mask shape {key_padding_mask.shape}, L={L}")

        text_pad = key_padding_mask[:, :L].contiguous()    # [B,L]
        motion_pad = key_padding_mask[:, L:].contiguous()  # [B,M]  (M is T or S)

        # ✅ if no padding anywhere, pass None to let attention use fast path
        if not text_pad.any():
            text_pad = None
        if not motion_pad.any():
            motion_pad = None

        # ✅ if we are in flatten_spatiotemporal mode AND current motion_pad is [B,T], expand to [B,T*K]
        # We can only do this when M looks like T (not already S). Heuristic: if M % K == 0, it *could* be S already.
        if flatten_spatiotemporal and (motion_pad is not None):
            # if the mask length corresponds to T (not S), you should repeat_interleave
            # simplest and safest: only repeat when you KNOW you built mask as L+T.
            # here we detect it by checking whether M matches tokens' time dimension if available.
            if tokens.dim() >= 2 and tokens.shape[1] == M:
                # tokens second dim equals M => M is T
                motion_pad = motion_pad.repeat_interleave(int(K), dim=1)  # [B,T*K]

        return text_pad, motion_pad

    @staticmethod
    def _masked_mean(x, pad_mask):
        # x [B,N,D], pad_mask [B,N] True=pad
        keep = (~pad_mask).float().unsqueeze(-1)
        denom = keep.sum(dim=1).clamp_min(1.0)
        return (x * keep).sum(dim=1) / denom

        return x
    def forward(self, tokens, text_emb, key_padding_mask=None, return_reps: bool = False,blueprint_tokens=None, blueprint_pad_mask=None):
        B, T, K = tokens.shape
        assert K == self.K, f"K mismatch: got {K}, expect {self.K}"
        S = int(T * self.K) if self.flatten_spatiotemporal else int(T)
        if S > self.max_seq_len:
            raise ValueError(f"SeqLen={S} exceeds max_seq_len={self.max_seq_len}")

        # split padding masks (auto-expands motion mask to [B,T*K] when flatten)
        if key_padding_mask is None:
            text_pad = None
            motion_pad = None
        else:
            text_pad, motion_pad = self._split_key_padding_mask(
                tokens, text_emb, key_padding_mask, K=self.K, flatten_spatiotemporal=self.flatten_spatiotemporal
            )

        # ---- slot embeddings -> x_slots [B,T,K,D] ----
        if self.use_group_tying:
            x_slots = torch.zeros((B, T, self.K, self.dim), device=tokens.device, dtype=torch.float32)
            slot_in = self.slot_in_emb(self.slot_ids.to(tokens.device))  # [K,D]

            for gid, slot_ids in enumerate(self.gid_to_slot_ids):
                if len(slot_ids) == 0:
                    continue
                tok = tokens[:, :, slot_ids]  # [B,T,S]
                emb = self.group_token_embs[gid](tok)  # [B,T,S,D]
                emb = emb + slot_in[slot_ids].view(1, 1, len(slot_ids), self.dim)
                x_slots[:, :, slot_ids, :] = emb
        else:
            # legacy per-slot embedding -> stack to [B,T,K,D]
            xs = []
            for k in range(self.K):
                xs.append(self.token_embs[k](tokens[..., k]).unsqueeze(2))  # [B,T,1,D]
            x_slots = torch.cat(xs, dim=2)  # [B,T,K,D]

        # ---- choose fusion vs flatten ----
        if not self.flatten_spatiotemporal:
            # old fusion: [B,T,K,D] -> [B,T,K*D] -> [B,T,D]
            feat_cat = x_slots.reshape(B, T, self.K * self.dim)
            motion_x = self.slot_proj(feat_cat)  # [B,T,D]

            # time pos emb (per time)
            motion_x = motion_x + self.motion_pos(self.pos_ids[:, :T])

            # text proj
            text_h = self.text_ln(self.text_proj(text_emb))  # [B,L,D]
            # ===== Blueprint KV (optional) =====
            if (blueprint_tokens is not None) and (blueprint_pad_mask is not None) and self.use_blueprint:
                bp = blueprint_tokens  # [B,Tb,K], int64
                B2, Tb, K2 = bp.shape
                assert B2 == B and K2 == self.K

                # embed blueprint tokens into [B,Tb,K,D] using SAME embedding logic as motion tokens
                if self.use_group_tying:
                    bp_slots = torch.zeros((B, Tb, self.K, self.dim), device=bp.device, dtype=torch.float32)
                    slot_in = self.slot_in_emb(self.slot_ids.to(bp.device))  # [K,D]
                    for gid, slot_ids in enumerate(self.gid_to_slot_ids):
                        if len(slot_ids) == 0:
                            continue
                        tok = bp[:, :, slot_ids]  # [B,Tb,S]
                        emb = self.group_token_embs[gid](tok)  # [B,Tb,S,D]
                        emb = emb + slot_in[slot_ids].view(1, 1, len(slot_ids), self.dim)
                        bp_slots[:, :, slot_ids, :] = emb
                else:
                    xs = []
                    for k in range(self.K):
                        xs.append(self.token_embs[k](bp[..., k]).unsqueeze(2))
                    bp_slots = torch.cat(xs, dim=2)  # [B,Tb,K,D]

                # fuse K -> D  (keep blueprint length = Tb, not Tb*K)
                bp_cat = bp_slots.reshape(B, Tb, self.K * self.dim)
                bp_h = self.bp_fuse(bp_cat)  # [B,Tb,D]

                # add time pos (reuse motion_pos; Tb should be <= max_seq_len)
                bp_h = bp_h + self.motion_pos(torch.arange(Tb, device=bp.device, dtype=torch.long)).view(1, Tb, self.dim)

                # shallow encoder
                bp_h = self.bp_encoder(bp_h, src_key_padding_mask=blueprint_pad_mask)
                bp_h = self.bp_ln(bp_h)

                # concat to text condition for cross-attn KV
                text_h = torch.cat([text_h, bp_h], dim=1)
                if text_pad is None:
                    # no text pad given -> assume all valid
                    text_pad = torch.zeros((B, text_h.shape[1]), dtype=torch.bool, device=text_h.device)
                    text_pad[:, :text_emb.shape[1]] = False
                    text_pad[:, text_emb.shape[1]:] = blueprint_pad_mask
                else:
                    text_pad = torch.cat([text_pad, blueprint_pad_mask], dim=1)
            for blk in self.blocks:
                motion_x = blk(motion_x, text_h, motion_pad, text_pad)

            motion_h = self.final_ln(motion_x)  # [B,T,D]

            # output heads keep same as before
            if self.use_group_tying:
                slot_out = self.slot_out_emb(self.slot_ids.to(tokens.device))  # [K,D]
                logits = {}
                for k, name in enumerate(self.slot_names):
                    gid = self.slot_gid[k]
                    h_k = motion_h + slot_out[k].view(1, 1, self.dim)
                    logits[name] = self.group_heads[gid](h_k)
            else:
                logits = {name: self.heads_dict[name](motion_h) for name in self.slot_names}

            if not return_reps:
                return logits

            # reps (time-level)
            if text_pad is None:
                text_rep = text_h.mean(dim=1)
            else:
                text_rep = self._masked_mean(text_h, text_pad)

            if motion_pad is None:
                motion_rep = motion_h.mean(dim=1)
            else:
                motion_rep = self._masked_mean(motion_h, motion_pad)

            reps = {
                "motion": motion_rep,
                "text": text_rep,
                "len_logits": self.length_head(text_rep),
            }
            return logits, reps

        # ---- NEW: flatten spatio-temporal: [B,T,K,D] -> [B,T*K,D] ----
        S = int(T * self.K)
        x_seq = x_slots.reshape(B, S, self.dim)  # [B,T*K,D]

        # time pos emb repeats every K tokens: time_id = i // K
        time_ids = torch.arange(T, device=tokens.device, dtype=torch.long).repeat_interleave(self.K)  # [T*K]
        x_seq = x_seq + self.motion_pos(time_ids).view(1, S, self.dim)

        # text proj
        text_h = self.text_ln(self.text_proj(text_emb))  # [B,L,D]
        # ===== Blueprint KV (optional) =====
        if (blueprint_tokens is not None) and (blueprint_pad_mask is not None) and self.use_blueprint:
            bp = blueprint_tokens  # [B,Tb,K], int64
            B2, Tb, K2 = bp.shape
            assert B2 == B and K2 == self.K

            # embed blueprint tokens into [B,Tb,K,D] using SAME embedding logic as motion tokens
            if self.use_group_tying:
                bp_slots = torch.zeros((B, Tb, self.K, self.dim), device=bp.device, dtype=torch.float32)
                slot_in = self.slot_in_emb(self.slot_ids.to(bp.device))  # [K,D]
                for gid, slot_ids in enumerate(self.gid_to_slot_ids):
                    if len(slot_ids) == 0:
                        continue
                    tok = bp[:, :, slot_ids]  # [B,Tb,S]
                    emb = self.group_token_embs[gid](tok)  # [B,Tb,S,D]
                    emb = emb + slot_in[slot_ids].view(1, 1, len(slot_ids), self.dim)
                    bp_slots[:, :, slot_ids, :] = emb
            else:
                xs = []
                for k in range(self.K):
                    xs.append(self.token_embs[k](bp[..., k]).unsqueeze(2))
                bp_slots = torch.cat(xs, dim=2)  # [B,Tb,K,D]

            # fuse K -> D  (keep blueprint length = Tb, not Tb*K)
            bp_cat = bp_slots.reshape(B, Tb, self.K * self.dim)
            bp_h = self.bp_fuse(bp_cat)  # [B,Tb,D]

            # add time pos (reuse motion_pos; Tb should be <= max_seq_len)
            bp_h = bp_h + self.motion_pos(torch.arange(Tb, device=bp.device, dtype=torch.long)).view(1, Tb, self.dim)

            # shallow encoder
            bp_h = self.bp_encoder(bp_h, src_key_padding_mask=blueprint_pad_mask)
            bp_h = self.bp_ln(bp_h)

            # concat to text condition for cross-attn KV
            text_h = torch.cat([text_h, bp_h], dim=1)
            if text_pad is None:
                # no text pad given -> assume all valid
                text_pad = torch.zeros((B, text_h.shape[1]), dtype=torch.bool, device=text_h.device)
                text_pad[:, :text_emb.shape[1]] = False
                text_pad[:, text_emb.shape[1]:] = blueprint_pad_mask
            else:
                text_pad = torch.cat([text_pad, blueprint_pad_mask], dim=1)
        # motion_pad already expanded to [B,T*K] if key_padding_mask was [B,L+T]
        for blk in self.blocks:
            x_seq = blk(x_seq, text_h, motion_pad, text_pad)

        motion_h_seq = self.final_ln(x_seq)  # [B,T*K,D]

        # reshape back to [B,T,K,D] for per-slot heads
        motion_h_slots = motion_h_seq.view(B, T, self.K, self.dim)

        # ---- output heads: still return dict{name: [B,T,V]} so training script stays same ----
        if self.use_group_tying:
            slot_out = self.slot_out_emb(self.slot_ids.to(tokens.device))  # [K,D]
            logits = {}
            for k, name in enumerate(self.slot_names):
                gid = self.slot_gid[k]
                h_k = motion_h_slots[:, :, k, :] + slot_out[k].view(1, 1, self.dim)  # [B,T,D]
                logits[name] = self.group_heads[gid](h_k)  # [B,T,V]
        else:
            logits = {}
            for k, name in enumerate(self.slot_names):
                logits[name] = self.heads_dict[name](motion_h_slots[:, :, k, :])

        if not return_reps:
            return logits

        # reps: keep time-level meaning (average over K first, then masked mean over T)
        if text_pad is None:
            text_rep = text_h.mean(dim=1)
        else:
            text_rep = self._masked_mean(text_h, text_pad)

        motion_time = motion_h_slots.mean(dim=2)  # [B,T,D]
        if key_padding_mask is None:
            motion_rep = motion_time.mean(dim=1)
        else:
            # original motion_pad in training is [B,T], but inside model we expanded to [B,T*K] for attention.
            # For reps we want [B,T] semantics: derive from lengths mask implied by motion_pad of shape [B,T] if possible.
            # Best effort: if motion_pad came in expanded, compress it back.
            if motion_pad is not None and motion_pad.shape[1] == S:
                motion_pad_T = motion_pad.view(B, T, self.K).all(dim=2)  # [B,T] True only if all K are pad
            else:
                motion_pad_T = None

            if motion_pad_T is None:
                motion_rep = motion_time.mean(dim=1)
            else:
                motion_rep = self._masked_mean(motion_time, motion_pad_T)

        reps = {
            "motion": motion_rep,
            "text": text_rep,
            "len_logits": self.length_head(text_rep),
        }
        return logits, reps


    @staticmethod
    def cosine_schedule(step: int, total_steps: int) -> float:
        """Return keep ratio in [0,1] using cosine schedule."""
        if total_steps <= 1:
            return 1.0
        t = float(step) / float(total_steps - 1)
        # 0 -> 1 smoothly
        return float(0.5 - 0.5 * math.cos(math.pi * t))
    @torch.no_grad()
    def generate(
        self,
        text_emb: torch.Tensor,                 # [B,L,text_dim]
        text_pad_mask: torch.Tensor = None,     # [B,L] True=pad
        seq_len: int = 60,
        num_steps: int = 10,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        blueprint_tokens: torch.Tensor = None,  # [B,Tb,K] or None
        blueprint_pad_mask: torch.Tensor = None # [B,Tb] True=pad or None
    ) -> torch.Tensor:
        device = text_emb.device
        B = text_emb.shape[0]
        T = int(seq_len)
        S = int(T * self.K) if self.flatten_spatiotemporal else int(T)
        if S > self.max_seq_len:
            raise ValueError(f"SeqLen={S} exceeds max_seq_len={self.max_seq_len}")

        # per-slot special ids
        mask_ids = torch.tensor([cb for cb in self.codebook_sizes], device=device, dtype=torch.long)  # [K]

        # init all masked
        tokens = mask_ids.view(1, 1, self.K).expand(B, T, self.K).clone()
        known = torch.zeros((B, T), dtype=torch.bool, device=device)

        if text_pad_mask is None:
            text_pad_mask = torch.zeros((B, text_emb.shape[1]), dtype=torch.bool, device=device)

        # motion 无 pad，这里仍然需要把 text_pad_mask 传进去（cross-attn 用）
        # key_padding_mask 仍按 [text, motion(T)] 构造，flatten 时由模型内部 repeat 到 S
        key_padding_mask = torch.cat(
            [text_pad_mask, torch.zeros((B, T), dtype=torch.bool, device=device)],
            dim=1
        )

        last_preds = None
        temp = max(float(temperature), 1e-6)

        for step in range(int(num_steps)):
            # ===== CFG inference =====
            if float(cfg_scale) == 1.0:
                logits = self(
                    tokens, text_emb,
                    key_padding_mask=key_padding_mask,
                    return_reps=False,
                    blueprint_tokens=blueprint_tokens,
                    blueprint_pad_mask=blueprint_pad_mask
                )
            else:
                # build unconditional text: zero emb + only token0 is valid
                text_u = torch.zeros_like(text_emb)
                text_u_mask = torch.ones_like(text_pad_mask)
                text_u_mask[:, 0] = False
                key_padding_mask_u = torch.cat(
                    [text_u_mask, torch.zeros((B, T), dtype=torch.bool, device=device)],
                    dim=1
                )

                # conditional: include blueprint (if provided)
                logits_c = self(
                    tokens, text_emb,
                    key_padding_mask=key_padding_mask,
                    return_reps=False,
                    blueprint_tokens=blueprint_tokens,
                    blueprint_pad_mask=blueprint_pad_mask
                )
                # unconditional: drop ALL conditions, including blueprint
                logits_u = self(
                    tokens, text_u,
                    key_padding_mask=key_padding_mask_u,
                    return_reps=False,
                    blueprint_tokens=None,
                    blueprint_pad_mask=None
                )

                logits = {}
                s = float(cfg_scale)
                for name in self.slot_names:
                    logits[name] = logits_u[name] + s * (logits_c[name] - logits_u[name])

            conf_pos = 0.0
            preds = []

            # 不做 softmax 大张量：用 logsumexp 求 max prob
            for k, name in enumerate(self.slot_names):
                logit = logits[name] / temp  # [B,T,V]
                max_logit, pred_k = torch.max(logit, dim=-1)  # [B,T]
                log_denom = torch.logsumexp(logit, dim=-1)    # [B,T]
                conf_k = torch.exp(max_logit - log_denom)     # [B,T] = max prob
                preds.append(pred_k)
                conf_pos = conf_pos + conf_k

            conf_pos = conf_pos / float(self.K)  # [B,T]
            last_preds = preds

            keep_ratio = self.cosine_schedule(step, int(num_steps))
            target_keep = max(1, int(round(T * keep_ratio)))

            # 向量化更新 known：每行选出需要的新位置
            cur_known_cnt = known.sum(dim=1)  # [B]
            to_select = (target_keep - cur_known_cnt).clamp(min=0)  # [B]
            max_sel = int(to_select.max().item())

            if max_sel > 0:
                scores = conf_pos.masked_fill(known, float("-inf"))  # 已知位置不参与选择
                topk_idx = torch.topk(scores, k=max_sel, dim=1, largest=True).indices  # [B,max_sel]

                sel_mask = (torch.arange(max_sel, device=device).unsqueeze(0) < to_select.unsqueeze(1))  # [B,max_sel]
                new_known = torch.zeros_like(known)
                new_known.scatter_(1, topk_idx, sel_mask)
                known = known | new_known

            # fill known positions with predictions for all slots
            for k in range(self.K):
                tokens_k = tokens[..., k]
                tokens_k[known] = preds[k][known]
                tokens[..., k] = tokens_k

        # replace any remaining masks with last-step preds
        if last_preds is not None:
            for k in range(self.K):
                m = tokens[..., k] == mask_ids[k]
                if m.any():
                    tokens[..., k][m] = last_preds[k][m]

        # clamp away special ids
        for k, cb in enumerate(self.codebook_sizes):
            tokens[..., k].clamp_(0, cb - 1)

        return tokens


    @torch.no_grad()
    def generate_from_text(
        self,
        text_emb: torch.Tensor,              # [B,L,text_dim]
        text_pad_mask: torch.Tensor = None,  # [B,L] True=pad
        num_steps: int = 10,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        bin_to_length: str = "center",
        min_len: int = 1,
        blueprint_tokens: torch.Tensor = None,   # [B,Tb,K] or None
        blueprint_pad_mask: torch.Tensor = None  # [B,Tb] or None
    ):
        """
        Convenience inference API:
        1) predict length from text
        2) generate tokens with that length
        Returns:
        tokens: [B, T, K]
        pred_len: [B]
        len_logits: [B, num_bins]
        len_probs: [B, num_bins]
        """
        pred_len, len_logits, len_probs = self.predict_length(
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            bin_to_length=bin_to_length,
        )
        # 保险：长度下限/上限
        T = int(pred_len.max().item())
        T = max(int(min_len), min(T, int(self.max_seq_len)))

        tokens = self.generate(
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            seq_len=T,
            num_steps=num_steps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            blueprint_tokens=blueprint_tokens,
            blueprint_pad_mask=blueprint_pad_mask,
        )
        return tokens, pred_len, len_logits, len_probs