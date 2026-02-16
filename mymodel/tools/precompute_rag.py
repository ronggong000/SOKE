
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid RAG precompute (v2): BM25 + cosine fusion, with NO neighbor spillover.

Why v2:
- v1 used context windows (prev/next tokens) for single-token fallback, which can
  "spill" a neighbor's meaning onto the current token (你说的“扩窗后识别成隔壁单词”).
- v2 makes a hard rule:
    * multi-word phrases: ONLY matched by exact n-gram dictionary (deterministic)
    * single-token fallback: ONLY query the token itself (no prev/next windows)
    * plus a strict lexical gate: the chosen alias/canonical must contain the token
      (or be a slash form that contains it), otherwise reject -> unk

Output:
  rag_precompute_{train,val,test}.jsonl
  gloss_seq is ALWAYS same length as tokens_filtered.
"""

import os
import re
import json
import csv
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1) Text normalization + FS restore
# ============================================================

def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^A-Za-z0-9\s/]+", " ", s)  # keep slash for this/it style
    s = " ".join(s.strip().lower().split())
    return s


def restore_fingerspelling(text: str) -> str:
    if text is None:
        return ""
    s = " ".join(str(text).strip().split())
    pat = re.compile(r"\bFS_BEGIN\b(.*?)\bFS_END\b", flags=re.IGNORECASE)

    def repl(m):
        mid = m.group(1) or ""
        letters = re.findall(r"\b[lL][_-]?([A-Za-z])\b", mid)
        if not letters:
            return " "
        word = "".join([ch.upper() for ch in letters])
        return " " + word + " "

    s = pat.sub(repl, s)
    return " ".join(s.split())


def tokenize_for_rag(gloss_sent: str) -> list[str]:
    gloss_sent = restore_fingerspelling(gloss_sent)
    s = normalize_text(gloss_sent)
    return s.split()


# ============================================================
# 2) JSON / JSONL loaders (aliases + alias2canonical)
# ============================================================

def load_aliases_any(path: str) -> list[str]:
    if path.endswith(".jsonl"):
        aliases = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "alias" in obj:
                    aliases.append(str(obj["alias"]))
                elif isinstance(obj, str):
                    aliases.append(obj)
                else:
                    raise ValueError(f"[aliases.jsonl] bad line {ln}: {type(obj)}")
        return aliases

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], str):
            return [str(x) for x in data]
        if isinstance(data[0], dict) and "alias" in data[0]:
            return [str(x["alias"]) for x in data if isinstance(x, dict) and "alias" in x]

    raise ValueError(f"Unsupported aliases json format: {path} root={type(data)}")


def load_alias2canonical_any(path: str) -> dict[str, str]:
    if path.endswith(".jsonl"):
        m = {}
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"[alias2canonical.jsonl] bad line {ln}: {type(obj)}")
                a = obj.get("alias", None)
                c = obj.get("canonical", None)
                if a is None or c is None:
                    raise ValueError(f"[alias2canonical.jsonl] missing keys at line {ln}: {obj}")
                m[str(a)] = str(c)
        return m

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        out = {}
        for obj in data:
            if isinstance(obj, dict) and "alias" in obj and "canonical" in obj:
                out[str(obj["alias"])] = str(obj["canonical"])
        return out

    raise ValueError(f"Unsupported alias2canonical json format: {path} root={type(data)}")


def normalize_alias2canonical(alias2canonical_raw: dict[str, str]) -> dict[str, str]:
    out = {}
    for k, v in alias2canonical_raw.items():
        kk = normalize_text(k)
        vv = normalize_text(v)
        if kk:
            out[kk] = vv
    return out


def build_slash_part_map(alias2canonical: dict[str, str]) -> dict[str, str]:
    part2canon = {}
    for a, c in alias2canonical.items():
        for s in (a, c):
            if "/" in s:
                parts = [p.strip() for p in s.split("/") if p.strip()]
                if len(parts) >= 2:
                    for p in parts:
                        part2canon.setdefault(p, c)
    return part2canon


# ============================================================
# 3) Encoder (transformers mean pooling)
# ============================================================

class TextEncoder:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        self.cache = {}

    @torch.no_grad()
    def encode_one(self, text: str) -> np.ndarray:
        text = text.strip()
        if text in self.cache:
            return self.cache[text]

        enc = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc).last_hidden_state  # [1,L,D]
        attn = enc["attention_mask"].unsqueeze(-1)  # [1,L,1]
        pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)  # [1,D]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        v = pooled[0].detach().cpu().numpy().astype(np.float32)

        self.cache[text] = v
        return v


# ============================================================
# 4) Lexical BM25 (pure python)
# ============================================================

class BM25OkapiLite:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N > 0 else 0.0

        df = {}
        for doc in corpus_tokens:
            seen = set(doc)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.df = df

        self.idf = {}
        for t, n in df.items():
            self.idf[t] = np.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

        self.tf = []
        for doc in corpus_tokens:
            m = {}
            for t in doc:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        if self.N == 0:
            return np.zeros((0,), dtype=np.float32)

        scores = np.zeros((self.N,), dtype=np.float32)
        if not query_tokens:
            return scores

        for t in query_tokens:
            idf = self.idf.get(t, 0.0)
            if idf == 0.0:
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                scores[i] += idf * (f * (self.k1 + 1.0) / (denom + 1e-9))
        return scores

    def topk(self, query_tokens: list[str], k: int = 50):
        scores = self.get_scores(query_tokens)
        if scores.size == 0:
            return []
        kk = min(int(k), scores.shape[0])
        if kk <= 0:
            return []
        idx = np.argpartition(-scores, kk - 1)[:kk]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


# ============================================================
# 5) Vector retrieval + Hybrid fusion
# ============================================================

def topk_cosine(q: np.ndarray, index_mat: np.ndarray, k: int = 25):
    sims = index_mat @ q
    kk = min(int(k), sims.shape[0])
    if kk <= 0:
        return []
    idx = np.argpartition(-sims, kk - 1)[:kk]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def minmax_norm_scores(pairs: list[tuple[int, float]]) -> dict[int, float]:
    if not pairs:
        return {}
    vals = np.array([s for _, s in pairs], dtype=np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax - vmin < 1e-8:
        return {i: 1.0 for i, _ in pairs}
    return {i: float((s - vmin) / (vmax - vmin)) for i, s in pairs}


def build_alias_tokens_for_bm25(aliases_norm: list[str]) -> list[list[str]]:
    docs = []
    for a in aliases_norm:
        toks = []
        for t in a.split():
            if "/" in t:
                parts = [p.strip() for p in t.split("/") if p.strip()]
                toks.extend(parts if parts else [t])
            else:
                toks.append(t)
        docs.append(toks)
    return docs


def hybrid_top1_token(tok: str,
                      encoder: TextEncoder,
                      index_mat: np.ndarray,
                      bm25: BM25OkapiLite,
                      k_vec: int,
                      k_bm25: int,
                      alpha: float):
    qv = encoder.encode_one(tok)
    vec_pairs = topk_cosine(qv, index_mat, k=k_vec)
    vec_norm = minmax_norm_scores(vec_pairs)

    bm_pairs = bm25.topk([tok], k=k_bm25)
    bm_norm = minmax_norm_scores(bm_pairs)

    cand_ids = set(vec_norm.keys()) | set(bm_norm.keys())
    if not cand_ids:
        return None, 0.0

    best_i = None
    best_s = -1.0
    for i in cand_ids:
        s = alpha * vec_norm.get(i, 0.0) + (1.0 - alpha) * bm_norm.get(i, 0.0)
        if s > best_s:
            best_s = s
            best_i = i

    return best_i, float(best_s)


# ============================================================
# 6) Matching: exact phrase ngram + hybrid token fallback (ALIGNED)
# ============================================================

def make_phrase_dict(alias2canonical: dict[str, str]) -> dict[tuple[str, ...], str]:
    d = {}
    for a, c in alias2canonical.items():
        toks = tuple(a.split())
        if toks:
            d[toks] = c
    return d


def token_in_alias(tok: str, s: str) -> bool:
    if not tok or not s:
        return False
    for t in s.split():
        if t == tok:
            return True
        if "/" in t:
            parts = [p.strip() for p in t.split("/") if p.strip()]
            if tok in parts:
                return True
    return False

def match_tokens_hybrid(tokens: list[str],
                        encoder: TextEncoder,
                        index_mat: np.ndarray,
                        aliases_norm: list[str],
                        alias2canonical: dict[str, str],
                        slash_part2canon: dict[str, str],
                        bm25: BM25OkapiLite,
                        phrase_dict: dict[tuple[str, ...], str],
                        max_ngram: int,
                        k_vec: int,
                        k_bm25: int,
                        alpha: float,
                        score_thr: float,
                        span_mode: str = "repeat",          # "repeat" | "head"
                        cont_token: str = "__CONT__"):      # span continuation marker
    """
    Return:
      gloss_seq: same length as tokens
      gloss_mask: same length; 1 = supervise this position, 0 = ignore (continuation)
    span_mode:
      - "repeat": span 内每个 token 都填同一 canonical（旧行为）
      - "head":   span 只在第一个 token 填 canonical，其余填 cont_token，mask=0
    """
    n = len(tokens)
    out = ["unk"] * n
    mask = [1] * n
    used = [False] * n

    STOPWORDS = {
        "also", "actually", "really", "very", "just", "there", "here", "then",
        "a", "an", "the", "to", "of", "and", "or", "in", "on", "at", "for", "with",
        "is", "am", "are", "was", "were", "be", "been", "being",
        "do", "does", "did",
    }

    # 1) exact phrase match (longest-first), aligned
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue

        matched = False
        for L in range(min(max_ngram, n - i), 0, -1):
            span = tuple(tokens[i:i+L])
            canon = phrase_dict.get(span, None)
            if canon is None:
                continue

            if span_mode == "head":
                # head-only
                out[i] = canon
                mask[i] = 1
                used[i] = True
                for k in range(i + 1, i + L):
                    out[k] = cont_token
                    mask[k] = 0
                    used[k] = True
            else:
                # repeat
                for k in range(i, i + L):
                    out[k] = canon
                    mask[k] = 1
                    used[k] = True

            i += L
            matched = True
            break

        if not matched:
            i += 1

    # 2) exact single token (including slash-part)
    for i in range(n):
        if used[i]:
            continue
        tok = tokens[i]
        if tok in alias2canonical:
            out[i] = alias2canonical[tok]
            mask[i] = 1
            used[i] = True
            continue
        sp = slash_part2canon.get(tok, None)
        if sp is not None:
            out[i] = sp
            mask[i] = 1
            used[i] = True
            continue

    # helper: strict gate
    def token_in_alias(tok: str, s: str) -> bool:
        if not tok or not s:
            return False
        for t in s.split():
            if t == tok:
                return True
            if "/" in t:
                parts = [p.strip() for p in t.split("/") if p.strip()]
                if tok in parts:
                    return True
        return False

    # 3) hybrid token fallback (NO context + strict gate)
    for i in range(n):
        if used[i]:
            continue
        tok = tokens[i]
        if (not tok) or (tok in STOPWORDS) or (len(tok) < 2):
            out[i] = "unk"
            mask[i] = 1
            used[i] = True
            continue

        idx1, s1 = hybrid_top1_token(tok, encoder, index_mat, bm25, k_vec, k_bm25, alpha)
        if idx1 is None or s1 < score_thr:
            out[i] = "unk"
            mask[i] = 1
            used[i] = True
            continue

        alias_norm = aliases_norm[idx1]
        canon = alias2canonical.get(alias_norm, alias_norm)

        if not token_in_alias(tok, alias_norm) and not token_in_alias(tok, canon):
            out[i] = "unk"
            mask[i] = 1
            used[i] = True
            continue

        out[i] = canon
        mask[i] = 1
        used[i] = True

    return out, mask


# ============================================================
# 7) TSV IO + precompute
# ============================================================

def read_tsv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert "SENTENCE_NAME" in reader.fieldnames and "GLOSS" in reader.fieldnames, \
            f"TSV must have columns SENTENCE_NAME and GLOSS, got {reader.fieldnames}"
        for r in reader:
            rows.append(r)
    return rows


def write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def precompute_split(tsv_path, out_path, matcher_kwargs: dict,
                     qc_n: int = 30, qc_show: int = 16):
    rows = read_tsv(tsv_path)

    out_rows = []
    hit_words = 0
    total_sup = 0
    unk_words = 0

    for r in rows:
        name = r["SENTENCE_NAME"].strip()
        gloss_raw = r["GLOSS"]

        toks = tokenize_for_rag(gloss_raw)
        matched, gloss_mask = match_tokens_hybrid(toks, **matcher_kwargs)

        # 统计只看 gloss_mask==1 的位置（head-only 模式会自动忽略 __CONT__）
        for g, m in zip(matched, gloss_mask):
            if m != 1:
                continue
            total_sup += 1
            if g == "unk":
                unk_words += 1
            else:
                hit_words += 1

        if len(out_rows) < qc_n and len(toks) > 0:
            show = min(qc_show, len(toks))
            print(f"[QC] {name} | toks[:{show}]={toks[:show]}")
            print(f"[QC] matched[:{show}]={matched[:show]}")
            print(f"[QC] mask[:{show}]={gloss_mask[:show]}")
            unk_pos = [i for i, (g, m) in enumerate(zip(matched[:show], gloss_mask[:show])) if (m == 1 and g == "unk")]
            if unk_pos:
                print(f"[QC] unk_pos[:{show}]={unk_pos} | unk_toks={[toks[i] for i in unk_pos]}")
            denom = max(1, sum(gloss_mask))
            this_hit = sum(1 for g, m in zip(matched, gloss_mask) if m == 1 and g != "unk")
            this_unk = sum(1 for g, m in zip(matched, gloss_mask) if m == 1 and g == "unk")
            print(f"[QC] sent_hit_rate={(this_hit/denom):.3f} hit={this_hit}/{denom} unk={this_unk}")

        out_rows.append({
            "name": name,
            "gloss_raw": gloss_raw,
            "tokens_filtered": toks,
            "gloss_seq": matched,
            "gloss_mask": gloss_mask,   # <-- 训练时用这个 mask 忽略 __CONT__
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".jsonl"):
        write_jsonl(out_path, out_rows)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False)

    hit_rate = (hit_words / max(1, total_sup))
    unk_rate = (unk_words / max(1, total_sup))
    print(f"✅ saved: {out_path}")
    print(f"✅ stats: hit_rate={hit_rate:.3f} unk_rate={unk_rate:.3f} "
          f"(hit={hit_words}/{total_sup}, unk={unk_words}/{total_sup}) rows={len(rows)}")
# ============================================================
# 8) main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--thr", type=float, default=0.75)  # kept for compatibility (unused)
    ap.add_argument("--qc_n", type=int, default=35)
    ap.add_argument("--qc_show", type=int, default=16)

    ap.add_argument("--train_tsv", type=str, default="/home/smuk0019/ar85_scratch2/singyu/how2sign/how2sign_train_gloss.csv")
    ap.add_argument("--val_tsv", type=str, default="/home/smuk0019/ar85_scratch2/singyu/how2sign/how2sign_val_gloss.csv")
    ap.add_argument("--test_tsv", type=str, default="/home/smuk0019/ar85_scratch2/singyu/how2sign/how2sign_test_gloss.csv")

    ap.add_argument("--index_npy", type=str, default="/home/smuk0019/ar85_scratch2/singyu/SOKE/mymodel/tools/output/embeddings.npy",
                    help="embeddings.npy [N,D] normalized float32")
    ap.add_argument("--aliases", type=str, default="/home/smuk0019/ar85_scratch2/singyu/SOKE/mymodel/tools/output/aliases.jsonl",
                    help="aliases.json or .jsonl, length N")
    ap.add_argument("--alias2canonical", type=str, default="/home/smuk0019/ar85_scratch2/singyu/SOKE/mymodel/tools/output/alias2canonical.jsonl",
                    help="alias2canonical.json or .jsonl")

    ap.add_argument("--sbert", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_dir", type=str, default="output")

    ap.add_argument("--max_ngram", type=int, default=3)
    ap.add_argument("--k_vec", type=int, default=25)
    ap.add_argument("--k_bm25", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--score_thr", type=float, default=0.45, help="higher=less wrong matches")

    # NEW: span control
    ap.add_argument("--span_mode", type=str, default="head", choices=["head", "repeat"],
                    help="head: phrase only labels first token, others __CONT__; repeat: label all tokens")
    ap.add_argument("--cont_token", type=str, default="__CONT__", help="continuation marker when span_mode=head")

    args = ap.parse_args()

    index_mat = np.load(args.index_npy).astype(np.float32)
    assert index_mat.ndim == 2, f"index must be [N,D], got {index_mat.shape}"
    norms = np.linalg.norm(index_mat, axis=1, keepdims=True)
    index_mat = index_mat / np.clip(norms, 1e-9, None)

    aliases_raw = load_aliases_any(args.aliases)
    alias2canonical_raw = load_alias2canonical_any(args.alias2canonical)
    alias2canonical = normalize_alias2canonical(alias2canonical_raw)

    assert len(aliases_raw) == index_mat.shape[0], f"aliases len {len(aliases_raw)} != index N {index_mat.shape[0]}"

    aliases_norm = [normalize_text(a) for a in aliases_raw]
    slash_part2canon = build_slash_part_map(alias2canonical)
    phrase_dict = make_phrase_dict(alias2canonical)

    bm25_docs = build_alias_tokens_for_bm25(aliases_norm)
    bm25 = BM25OkapiLite(bm25_docs)

    print(f"[RAG] loaded aliases: {len(aliases_norm)} from {args.aliases}")
    print(f"[RAG] loaded alias2canonical: {len(alias2canonical)} from {args.alias2canonical}")
    print(f"[RAG] bm25 docs: {len(bm25_docs)} avgdl={bm25.avgdl:.2f}")
    print(f"[RAG] slash parts: {len(slash_part2canon)}")
    print(f"[RAG] phrase entries: {len(phrase_dict)}")
    print(f"[RAG] span_mode={args.span_mode} cont_token={args.cont_token}")

    encoder = TextEncoder(args.sbert)

    matcher_kwargs = dict(
        encoder=encoder,
        index_mat=index_mat,
        aliases_norm=aliases_norm,
        alias2canonical=alias2canonical,
        slash_part2canon=slash_part2canon,
        bm25=bm25,
        phrase_dict=phrase_dict,
        max_ngram=int(args.max_ngram),
        k_vec=int(args.k_vec),
        k_bm25=int(args.k_bm25),
        alpha=float(args.alpha),
        score_thr=float(args.score_thr),
        span_mode=str(args.span_mode),
        cont_token=str(args.cont_token),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    precompute_split(
        args.train_tsv,
        os.path.join(args.out_dir, "rag_precompute_train.jsonl"),
        matcher_kwargs,
        qc_n=args.qc_n, qc_show=args.qc_show
    )
    precompute_split(
        args.val_tsv,
        os.path.join(args.out_dir, "rag_precompute_val.jsonl"),
        matcher_kwargs,
        qc_n=args.qc_n, qc_show=args.qc_show
    )
    precompute_split(
        args.test_tsv,
        os.path.join(args.out_dir, "rag_precompute_test.jsonl"),
        matcher_kwargs,
        qc_n=args.qc_n, qc_show=args.qc_show
    )

if __name__ == "__main__":
    main()
