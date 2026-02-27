import argparse
import json
import os
import re
from collections import defaultdict

def normalize_gloss(g: str, case_mode: str = "upper") -> str:
    """
    规则（按你纠正后的要求）：
      1) 仅删除最前面的 'seed' 这4个字母（不删 seed 后面的内容）
         - seed1 DOLLAR -> 1 DOLLAR
         - seedA-LINE BOB -> A-LINE BOB
         - 但如果整个词就是 'seed'，则保留 'seed'
      2) 删除所有 '-'（用空格替换）
         - STAND-UP -> STAND UP
      3) '_' -> 空格；多空格合一
      4) 去掉尾部变体编号：' <digits>'（例如 'ARM 2' -> 'ARM'）
      5) 大小写统一（默认 upper；也支持 lower/keep）
    """
    if g is None:
        return ""

    g = str(g).strip()
    if g == "":
        return ""

    # 1) 仅删除前缀 'seed'（只删这4个字母）
    #    如果整体就是 'seed'（忽略大小写/空格），则保留不动
    if g.lower() != "seed" and g.lower().startswith("seed"):
        g = g[4:]  # 删除 'seed'
        # 删除 seed 后面紧跟的分隔符/空格（但不删除字母数字）
        g = g.lstrip(" _-").strip()

    # 2) 删除所有 '-'（用空格替换）
    g = g.replace("-", " ")

    # 3) '_' 也当作分隔符
    g = g.replace("_", " ")

    # 多空格合一
    g = re.sub(r"\s+", " ", g).strip()

    # 4) 去掉尾部变体编号（仅数字）
    g = re.sub(r"\s+\d+$", "", g).strip()

    # 再合一次空格
    g = re.sub(r"\s+", " ", g).strip()

    # 5) 大小写
    if case_mode == "upper":
        g = g.upper()
    elif case_mode == "lower":
        g = g.lower()
    elif case_mode == "keep":
        pass
    else:
        raise ValueError(f"Unknown case_mode: {case_mode}")

    return g


def load_merged_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict, got {type(data)}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="merged_gloss_tokens.json（顶层 dict: gloss -> {gloss,samples}）")
    ap.add_argument("--out_json", required=True, help="输出清洗后的 json 路径")
    ap.add_argument("--case", default="upper", choices=["upper", "lower", "keep"], help="gloss 大小写归一化方式")
    ap.add_argument("--write_aliases", action="store_true", help="是否在输出里保留 aliases（原始 gloss 变体列表）")
    args = ap.parse_args()

    data = load_merged_dict(args.in_json)

    # new_key -> samples merged
    merged_samples = defaultdict(list)
    # new_key -> set(original variants)
    aliases = defaultdict(set)

    total_in_gloss = len(data)
    total_in_samples = 0

    for orig_gloss, payload in data.items():
        samples = payload.get("samples", [])
        if not isinstance(samples, list):
            continue

        total_in_samples += len(samples)

        new_gloss = normalize_gloss(orig_gloss, case_mode=args.case)
        if new_gloss == "":
            continue

        aliases[new_gloss].add(str(orig_gloss))

        # samples 直接拼上去
        merged_samples[new_gloss].extend(samples)

    # 构建输出 dict
    out = {}
    for g in sorted(merged_samples.keys()):
        # 给每个 gloss 的 samples 排个序：seq_len 降序，id 次序
        samples = merged_samples[g]
        samples.sort(key=lambda s: (-int(s.get("seq_len", 0)), str(s.get("id", ""))))

        obj = {
            "gloss": g,
            "samples": samples
        }
        if args.write_aliases:
            obj["aliases"] = sorted(aliases[g])

        out[g] = obj

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    total_out_gloss = len(out)
    total_out_samples = sum(len(v["samples"]) for v in out.values())

    # 统计：有多少 gloss 被合并（aliases>1）
    merged_groups = sum(1 for g in out.keys() if len(aliases[g]) > 1)

    print(f"✅ Input glosses  : {total_in_gloss}")
    print(f"✅ Input samples  : {total_in_samples}")
    print(f"✅ Output glosses : {total_out_gloss}")
    print(f"✅ Output samples : {total_out_samples}")
    print(f"✅ Merged groups  : {merged_groups} (aliases>1)")
    print(f"✅ Saved to       : {args.out_json}")


if __name__ == "__main__":
    main()
