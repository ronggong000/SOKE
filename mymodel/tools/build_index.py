import os, json, argparse
import numpy as np


def iter_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def build_syn_maps(msasl_syn_path: str):
    syn_groups = load_json(msasl_syn_path)
    word2group = {}
    for g in syn_groups:
        if not isinstance(g, list):
            continue
        items = [norm(x) for x in g if str(x).strip()]
        items = [x for x in items if x]
        if len(items) < 2:
            continue
        S = set(items)
        for w in S:
            word2group[w] = S
    return word2group


def expand_aliases_for_gloss(gloss: str, word2group: dict):
    """
    只做 MSASL_synonym 扩展，不做 ' ' <-> '-' <-> '_' 变体。
    """
    g = norm(gloss)
    aliases = {g}

    toks = g.split()
    for i, t in enumerate(toks):
        if t in word2group:
            for alt in word2group[t]:
                new = toks[:]
                new[i] = alt
                aliases.add(" ".join(new))

    return {a for a in aliases if a}


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_jsonl", type=str, required=True, help="merged_by_gloss.jsonl（每行一个 gloss）")
    ap.add_argument("--msasl_syn", type=str, default="MSASL_synonym.json")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--sbert", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    word2group = build_syn_maps(args.msasl_syn)

    # 1) canonical glosses
    canonical_glosses = []
    for obj in iter_jsonl(args.merged_jsonl):
        g = obj.get("gloss", None)
        if g is None:
            continue
        g = norm(g)
        if g:
            canonical_glosses.append(g)
    canonical_glosses = sorted(set(canonical_glosses))
    canon_set = set(canonical_glosses)

    # 2) aliases + alias2canonical (自映射优先)
    aliases = []
    alias2canonical = {}

    for cg in canonical_glosses:
        for a in sorted(expand_aliases_for_gloss(cg, word2group)):
            if a in canon_set:
                # 关键：canonical 自己永远映射回自己
                if a not in alias2canonical:
                    alias2canonical[a] = a
                    aliases.append(a)
                elif alias2canonical[a] != a:
                    alias2canonical[a] = a
                continue

            # 其它别名：第一次出现占坑
            if a not in alias2canonical:
                alias2canonical[a] = cg
                aliases.append(a)

    # 3) SBERT encode
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    model = SentenceTransformer(args.sbert)
    emb = model.encode(aliases, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    # 4) save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), emb)

    write_jsonl(os.path.join(args.out_dir, "canonical_glosses.jsonl"),
               [{"gloss": g} for g in canonical_glosses])

    write_jsonl(os.path.join(args.out_dir, "aliases.jsonl"),
               [{"alias": a} for a in aliases])

    write_jsonl(os.path.join(args.out_dir, "alias2canonical.jsonl"),
               [{"alias": a, "canonical": alias2canonical[a]} for a in sorted(alias2canonical.keys())])

    print(f"✅ canonical glosses: {len(canonical_glosses)}")
    print(f"✅ aliases built    : {len(aliases)}")
    print(f"✅ alias2canonical  : {len(alias2canonical)}")
    print(f"✅ saved to         : {args.out_dir}")


if __name__ == "__main__":
    main()
