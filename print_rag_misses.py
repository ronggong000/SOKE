import os
import json
from collections import Counter

# 1) 直接 import 你现有 rag.py 里的函数（按你项目真实 import 路径二选一）
#    A. 如果你是在项目根目录运行，且 rag.py 就在同目录：用下面这行
from mymodel.tools.rag import _load_wlasl_map, _normalize_gloss_sentence, _lookup_wlasl_entry



def iter_gloss_sentences_from_json(json_path: str):
    """
    从 dataset json 里尽可能稳健地取出 gloss sentence。
    你的数据字段可能叫 text / gloss / glosses / etc，所以这里做了兜底。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{json_path} must be a list of samples, got {type(data)}")

    for e in data:
        if not isinstance(e, dict):
            continue

        # 你项目里最常见的是 text（或 gloss）。这里做多 key 兜底。
        s = (
            e.get("gloss_sentence")
            or e.get("gloss")
            or e.get("glosses")
            or e.get("asl_gloss")
            or e.get("text")
            or ""
        )

        # glosses 可能是 list[str]
        if isinstance(s, list):
            s = " ".join([str(x) for x in s])

        s = str(s).strip()
        if s:
            yield s


def scan_misses(dataset_root: str, split_json: str, topk: int = 10, max_samples: int = 0):
    """
    dataset_root: 里面要有 wlasl_qvae_tokens.json （rag.py 会从这加载）
    split_json: train_dataset.json / val_dataset.json 的路径
    topk: 打印多少个 miss 词
    max_samples: >0 时只扫前 N 条（用于快速调试）
    """
    wmap = _load_wlasl_map(dataset_root)

    miss_counter = Counter()
    hit_words = 0
    total_words = 0

    for i, sent in enumerate(iter_gloss_sentences_from_json(split_json)):
        if max_samples and i >= int(max_samples):
            break

        words = _normalize_gloss_sentence(sent)
        for w in words:
            total_words += 1
            e = _lookup_wlasl_entry(w, wmap)
            if e is None:
                miss_counter[w] += 1
            else:
                hit_words += 1

    hit_rate = hit_words / max(1, total_words)

    print(f"\n=== Scan: {os.path.basename(split_json)} ===")
    print(f"Total words: {total_words}")
    print(f"Hit words:   {hit_words}")
    print(f"Hit rate:    {hit_rate:.4f}")
    print(f"Unique misses: {len(miss_counter)}")

    print(f"\nTop-{topk} missed words (by frequency):")
    for w, c in miss_counter.most_common(topk):
        print(f"  {w:<24s}  count={c}")

    return {
        "total_words": total_words,
        "hit_words": hit_words,
        "hit_rate": hit_rate,
        "unique_misses": len(miss_counter),
        "miss_counter": miss_counter,
    }


def main():
    # ======= 你只需要改这几行 =======
    DATASET_ROOT = "checkpoints/vae/qvae_b256h1024_L1_fingerdistinct"
    TRAIN_JSON   = os.path.join(DATASET_ROOT, "train_dataset.json")
    VAL_JSON     = os.path.join(DATASET_ROOT, "val_dataset.json")

    TOPK = 10
    MAX_SAMPLES = 0  # 0=全量；比如 2000 可以快速扫

    # ======= 跑两份 =======
    scan_misses(DATASET_ROOT, TRAIN_JSON, topk=TOPK, max_samples=MAX_SAMPLES)
    scan_misses(DATASET_ROOT, VAL_JSON, topk=TOPK, max_samples=MAX_SAMPLES)


if __name__ == "__main__":
    main()
