import os, json, torch

dataset_root = "checkpoints/vae/qvae_b256h1024_L1_fingerdistinct"
text_dir = "data/text_embedding/train"  # 没有就删掉这行
gloss_dir = "data/gloss_embedding/train"  # 没有就删掉这行

jpath = os.path.join(dataset_root, "train_dataset.json")
data = json.load(open(jpath, "r", encoding="utf-8"))
name = data[0]["name"]

def check_one(base, name):
    # 你如果实际是 .pt 以外的后缀，改这里
    p = os.path.join(base, name + ".pt")
    print("path:", p, "exists:", os.path.exists(p))
    if os.path.exists(p):
        obj = torch.load(p, map_location="cpu")
        x = obj
        if isinstance(obj, dict):
            print("keys:", list(obj.keys())[:10])
            # 你按实际 key 改：emb / text / feat ...
            for k in ["emb", "text", "feat", "hidden", "last_hidden_state"]:
                if k in obj:
                    x = obj[k]
                    break
        x = torch.as_tensor(x).float()
        print("shape:", tuple(x.shape), "norm:", x.norm().item(), "std:", x.std().item())

print("sample name:", name)
check_one(text_dir, name)
check_one(gloss_dir, name)