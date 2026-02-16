import os
import json
import argparse
import hashlib
from collections import defaultdict


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_int_list(x):
    # tokens 可能是 numpy 序列序列化后，或 list[int]，或 list[list[int]]
    if x is None:
        return None
    if isinstance(x, list):
        # 2D -> flatten
        if len(x) > 0 and isinstance(x[0], list):
            out = []
            for row in x:
                out.extend(row)
            return [int(v) for v in out]
        return [int(v) for v in x]
    return None


def infer_seq_len(entry: dict, tokens: list[int], default_num_nodes: int = 13):
    """
    目标：返回 token 序列的“帧长度 T”（更常用）
    优先级：
      1) entry['shape'] = [T, N] / (T, N)
      2) len(tokens) // N（N 来自 shape[1] 或 default_num_nodes）
      3) len(tokens)
    """
    shape = entry.get("shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
        try:
            T = int(shape[0])
            if T > 0:
                return T
        except Exception:
            pass

    # 如果 shape 提供了 N，用它；否则用默认 13
    N = default_num_nodes
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            N = int(shape[1])
        except Exception:
            N = default_num_nodes

    if tokens is None:
        return 0

    if N > 0 and len(tokens) % N == 0:
        T = len(tokens) // N
        return int(T)

    return int(len(tokens))


def normalize_entry(entry: dict, default_num_nodes: int = 13):
    """
    兼容不同输入字段：
      - id: 优先 video_id，其次 id，再其次 instance_id
      - gloss: gloss
      - tokens: tokens 或 code_seq 或 code_matrix(会flatten)
      - seq_len: 推断出来的 T
    返回: (gloss, sample_dict) 或 (None, None)
    """
    gloss = entry.get("gloss", None)
    if gloss is None:
        return None, None
    gloss = str(gloss).strip()
    if gloss == "":
        return None, None

    sid = entry.get("video_id", None)
    if sid is None:
        sid = entry.get("id", None)
    if sid is None:
        sid = entry.get("instance_id", None)
    sid = "UNKNOWN_ID" if sid is None else str(sid).strip()

    tokens = entry.get("tokens", None)
    if tokens is None:
        tokens = entry.get("code_seq", None)
    if tokens is None:
        tokens = entry.get("code_matrix", None)

    tokens = _as_int_list(tokens)
    if tokens is None:
        # 没 token 直接丢弃
        return None, None

    seq_len = infer_seq_len(entry, tokens, default_num_nodes=default_num_nodes)

    sample = {
        "id": sid,
        "tokens": tokens,
        "seq_len": seq_len,
    }
    return gloss, sample


def sample_fingerprint(gloss: str, sample: dict):
    """
    用于去重：gloss + id + tokens hash
    """
    h = hashlib.md5()
    h.update(gloss.encode("utf-8"))
    h.update(b"|")
    h.update(sample["id"].encode("utf-8"))
    h.update(b"|")
    # tokens 很长，用 hash 避免内存爆
    h.update(hashlib.md5(bytes(",".join(map(str, sample["tokens"])), "utf-8")).digest())
    return h.hexdigest()


def merge_two_files(in1: str, in2: str, out_path: str, default_num_nodes: int = 13):
    data1 = load_json(in1)
    #data2 = load_json(in2)

    merged = defaultdict(list)
    seen = set()

    def add_list(data, src_name: str):
        if not isinstance(data, list):
            raise ValueError(f"{src_name} is not a JSON list. Got type={type(data)}")
        for entry in data:
            if not isinstance(entry, dict):
                continue
            gloss, sample = normalize_entry(entry, default_num_nodes=default_num_nodes)
            if gloss is None:
                continue
            # 加一个来源字段方便追踪（你不想要可以删掉）
            sample["source"] = src_name

            fp = sample_fingerprint(gloss, sample)
            if fp in seen:
                continue
            seen.add(fp)
            merged[gloss].append(sample)

    add_list(data1, os.path.basename(in1))
    #add_list(data2, os.path.basename(in2))

    # 输出：dict[str, {"gloss": str, "samples": [...] }]
    # 同时做个稳定排序：gloss 排序；samples 按 seq_len 降序再按 id
    out_obj = {}
    for gloss in sorted(merged.keys()):
        samples = merged[gloss]
        samples.sort(key=lambda s: (-int(s.get("seq_len", 0)), s.get("id", "")))
        out_obj[gloss] = {"gloss": gloss, "samples": samples}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False)

    print(f"✅ Merged glosses: {len(out_obj)}")
    print(f"✅ Total samples : {sum(len(v['samples']) for v in out_obj.values())}")
    print(f"✅ Saved to      : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1", type=str, required=True, help="wlasl2000_qvae_tokens.json 路径")
    parser.add_argument("--in2", type=str, required=True, help="aslcitizen_qvae_tokens.json 路径")
    parser.add_argument("--out", type=str, required=True, help="合并后的输出 json 路径")
    parser.add_argument("--num_nodes", type=int, default=13, help="每帧节点数，用于无 shape 时推断 seq_len（默认 13）")
    args = parser.parse_args()

    merge_two_files(args.in1, args.in2, args.out, default_num_nodes=args.num_nodes)


if __name__ == "__main__":
    main()
