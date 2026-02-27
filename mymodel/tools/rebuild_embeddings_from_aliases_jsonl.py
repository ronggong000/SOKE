import os, json, argparse
import numpy as np

def iter_aliases_jsonl(p):
    aliases = []
    with open(p, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "alias" in obj:
                aliases.append(str(obj["alias"]))
            else:
                raise ValueError(f"bad line {ln}: {obj}")
    return aliases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases_jsonl", type=str, default="output/aliases.jsonl")
    ap.add_argument("--out_npy", type=str, default="output/embeddings.npy")
    ap.add_argument("--sbert", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    aliases = iter_aliases_jsonl(args.aliases_jsonl)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.sbert)
    emb = model.encode(aliases, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, emb)
    print(f"✅ aliases: {len(aliases)}")
    print(f"✅ saved embeddings: {args.out_npy} shape={emb.shape}")

if __name__ == "__main__":
    main()
