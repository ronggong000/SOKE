import os
from typing import List

import torch
from torch.utils.data import Dataset


def _to_text_pair(item, only_gloss: bool):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        sent = "" if item[0] is None else str(item[0])
        gloss = "" if item[1] is None else str(item[1])
    else:
        sent = ""
        gloss = "" if item is None else str(item)
    if only_gloss:
        return ["", gloss]
    return [sent, gloss]


class CachedLatentDataset(Dataset):
    """
    Cache payload format (.pt):
      - names: List[str], length N
      - texts: List[[sentence, gloss]], length N
      - offsets: LongTensor [N+1]
      - latent_lengths: LongTensor [N]
      - frame_lengths: LongTensor [N]
      - latents: Tensor [sum(latent_lengths), J, D]
    """

    def __init__(self, cache_path: str, only_gloss: bool = True):
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"latent cache not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")

        self.cache_path = cache_path
        self.only_gloss = bool(only_gloss)
        self.names: List[str] = list(payload.get("names", []))
        self.texts: List[List[str]] = list(payload.get("texts", []))
        self.offsets: torch.Tensor = payload["offsets"].long()
        self.latent_lengths: torch.Tensor = payload["latent_lengths"].long()
        self.frame_lengths: torch.Tensor = payload["frame_lengths"].long()
        self.latents: torch.Tensor = payload["latents"]
        self.lengths = self.latent_lengths.tolist()

        n = len(self.names)
        if self.offsets.numel() != n + 1:
            raise ValueError(f"offsets must be [N+1], got {self.offsets.shape} vs N={n}")
        if self.latent_lengths.numel() != n:
            raise ValueError(f"latent_lengths must be [N], got {self.latent_lengths.shape} vs N={n}")
        if self.frame_lengths.numel() != n:
            raise ValueError(f"frame_lengths must be [N], got {self.frame_lengths.shape} vs N={n}")
        if len(self.texts) != n:
            raise ValueError(f"texts length={len(self.texts)} != names length={n}")
        if self.latents.ndim != 3:
            raise ValueError(f"latents must be 3D [sumT, J, D], got {self.latents.shape}")

        print(
            f"[LatentCache] Loaded {n} samples from {cache_path} | "
            f"latents={tuple(self.latents.shape)} dtype={self.latents.dtype}"
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        s = int(self.offsets[idx].item())
        e = int(self.offsets[idx + 1].item())
        z = self.latents[s:e]
        if z.numel() == 0:
            # Safety fallback: one-step zero latent.
            z = torch.zeros((1, self.latents.shape[1], self.latents.shape[2]), dtype=self.latents.dtype)
        text_pair = _to_text_pair(self.texts[idx], only_gloss=self.only_gloss)
        latent_len = int(z.shape[0])
        return text_pair, z, latent_len, self.names[idx]


def latent_cache_collate_fn(batch):
    """
    batch items:
      (text_pair, latents[Tz,J,D], latent_len, sample_id)
    return:
      text_out, latents[B,Tz_max,J,D], masks[B,Tz_max], lengths[B], names
    """
    texts, z_list, lengths, names = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    B = len(z_list)
    J = int(z_list[0].shape[1])
    D = int(z_list[0].shape[2])
    T_max = int(max(z.shape[0] for z in z_list))

    latents = torch.zeros((B, T_max, J, D), dtype=z_list[0].dtype)
    masks = torch.zeros((B, T_max), dtype=torch.float32)

    for i, (z, L) in enumerate(zip(z_list, lengths.tolist())):
        Ti = int(z.shape[0])
        latents[i, :Ti] = z
        if L > 0:
            masks[i, :L] = 1.0

    text_out: List[List[str]] = []
    for t in texts:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            text_out.append([str(t[0]), str(t[1])])
        else:
            text_out.append([str(t) if t is not None else "", ""])

    return text_out, latents, masks, lengths, list(names)
