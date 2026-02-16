import torch
import numpy as np
import os
import sys
import json
import re
import yaml
import argparse
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace

from mGPT.utils.joints_list import SMPLX_JOINT_LANDMARK_NAMES, SELECTED_JOINT_INDICES,SELECTED_JOINT_LANDMARK_INDICES,SELECTED_JOINT_LANDMARK_BODY_EVAL,SELECTED_JOINT_LANDMARK_LHAND_EVAL,SELECTED_JOINT_LANDMARK_RHAND_EVAL, SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,SELECTED_JOINT_INDICES_BODY_ONLY,SELECTED_JOINT_INDICES_NEIGHBOR_LIST


# 确保导入的是你提供的新模型文件
from mymodel.vae.qvae_model_rod3_fixed_length import VAE
def build_videoid_to_gloss_map(wlasl_json_path: str):
    """
    Robust loader for WLASL_v0.3.json.

    Handles:
      - root is list[dict]  (common)
      - root is dict with a list field (e.g. {"data":[...]} or {"glosses":[...]} etc)

    Stores multiple keys per video_id:
      - original string
      - stripped leading zeros
      - (optional) zero-padded to 5 digits if numeric (helps some filename formats)
    """
    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # unwrap common dict containers
    if isinstance(data, dict):
        for k in ("data", "glosses", "annotations", "items"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise TypeError(f"WLASL json root must be list (or dict containing a list). Got {type(data)}")

    vid2gloss = {}

    def add_vid(vid_str: str, gloss: str):
        vid_str = str(vid_str).strip()
        if vid_str == "":
            return
        gloss = str(gloss).strip()
        if gloss == "":
            return

        # original
        vid2gloss.setdefault(vid_str, gloss)

        # strip leading zeros
        vid_norm = vid_str.lstrip("0")
        if vid_norm == "":
            vid_norm = "0"
        vid2gloss.setdefault(vid_norm, gloss)

        # optional: pad numeric ids to 5 digits (00295 style)
        if vid_norm.isdigit():
            vid_pad5 = vid_norm.zfill(5)
            vid2gloss.setdefault(vid_pad5, gloss)

    for item in data:
        if not isinstance(item, dict):
            continue
        gloss = item.get("gloss", None)
        instances = item.get("instances", None)

        # some variants might use "videos"
        if instances is None:
            instances = item.get("videos", [])

        if not gloss or not isinstance(instances, list):
            continue

        for inst in instances:
            if not isinstance(inst, dict):
                continue
            vid = inst.get("video_id", None)
            if vid is None:
                continue
            add_vid(vid, gloss)

    return vid2gloss
# =============== 配置加载工具 ===============
def _convert_string_to_type(s):
    s = s.strip()
    if s.lower() == 'true': return True
    if s.lower() == 'false': return False
    if s.startswith('[') and s.endswith(']'):
        items = s[1:-1].split(',')
        return [_convert_string_to_type(item) for item in items]
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s

def load_config(path):
    if path.endswith('.txt'):
        config_dict = {}
        with open(path, 'r') as f:
            for line in f:
                if '---' in line: continue
                parts = line.split(':', 1)
                if len(parts) == 2:
                    config_dict[parts[0].strip()] = _convert_string_to_type(parts[1].strip())
        return SimpleNamespace(**config_dict)
    else:
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
        cleaned = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in config_dict.items()}
        return SimpleNamespace(**cleaned)

# --- 1. 文件名解析 ---
def parse_filename(filename):
    """
    支持三种命名：
      1) gloss + sep + ID   例如: DISAGREEMENT_836681859074_aioswilor.npz
      2) ID + sep + gloss   例如: 836681859074-DISAGREEMENT_aioswilor.npz
      3) 只有 ID            例如: 00295_aioswilor.npz  或  00295.npz

    返回: (gloss_or_none, video_id_str)
      - 当文件名里没有 gloss 时，gloss_or_none = None
    """
    name = os.path.basename(filename)
    if name.endswith('.npz'):
        name = name[:-4]
    name = name.replace('_aioswilor', '')

    # 3) 只有ID（全数字）
    m = re.match(r'^(\d+)$', name)
    if m:
        video_id = m.group(1)
        return None, video_id

    # 先尝试：ID 在前（纯数字ID）
    m = re.match(r'^(\d+)[_-](.+)$', name)
    if m:
        video_id = m.group(1)
        gloss = m.group(2).replace('_', ' ').strip()
        return gloss, video_id

    # 再尝试：ID 在后（纯数字ID）
    m = re.match(r'^(.+)[_-](\d+)$', name)
    if m:
        gloss = m.group(1).replace('_', ' ').strip()
        video_id = m.group(2)
        return gloss, video_id

    raise ValueError(f"无法解析的文件名格式: {filename}")
class WLASLDataset(Dataset):
    def __init__(self, data_dir, config, vid2gloss=None):
        self.data_dir = data_dir
        self.config = config
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

        # 对应你模型需要的关节数量 (43个关节)
        self.indices = SELECTED_JOINT_INDICES
        self.max_length = int(getattr(config, 'max_length', 256))

        # video_id -> gloss 映射（允许为空：为空就只能用文件名里的 gloss）
        self.vid2gloss = vid2gloss or {}

    def __len__(self):
        return len(self.data_files)

    def _lookup_gloss(self, gloss_from_name, video_id_str):
        """
        优先级：
          1) 文件名里带的 gloss
          2) vid2gloss 映射
          3) "UNKNOWN"
        同时兼容前导0：会尝试 video_id 原样 & 去前导0
        """
        if gloss_from_name is not None and str(gloss_from_name).strip() != "":
            return gloss_from_name

        vid = str(video_id_str).strip()
        if vid in self.vid2gloss:
            return self.vid2gloss[vid]

        vid_norm = vid.lstrip("0")
        if vid_norm == "":
            vid_norm = "0"
        if vid_norm in self.vid2gloss:
            return self.vid2gloss[vid_norm]

        return "UNKNOWN"

    def __getitem__(self, idx):
        filename = self.data_files[idx]
        filepath = os.path.join(self.data_dir, filename)

        gloss_from_name, video_id = parse_filename(filename)
        gloss = self._lookup_gloss(gloss_from_name, video_id)

        try:
            with np.load(filepath) as data:
                poses = data['poses'][:, self.indices, :]  # [T, J, 3]
                n_frames = int(poses.shape[0])

                if n_frames <= 0:
                    raise ValueError(f"空的姿态序列: {filepath}")

                # 固定长度处理 (Padding/Crop)
                if n_frames < self.max_length:
                    padding = np.repeat(poses[-1:], self.max_length - n_frames, axis=0)
                    poses = np.concatenate([poses, padding], axis=0)
                else:
                    poses = poses[:self.max_length]

                motion_flat = poses.reshape(self.max_length, -1).astype(np.float32)
                return torch.from_numpy(motion_flat), gloss, video_id, min(n_frames, self.max_length)

        except Exception as e:
            raise ValueError(f"无法加载或处理文件: {filepath} | err={repr(e)}")


def collate_fn(batch):
    motions, glosses, vids, lens = zip(*batch)
    return torch.stack(motions), list(glosses), list(vids), list(lens)

# --- 3. 核心提取逻辑 ---
@torch.no_grad()
def extract_tokens(opt, model, loader):
    model.eval()

    # 根据 finger_distinct 策略定义的组名顺序
    group_names = ['torso', 'shared_arms', 'idx', 'mid', 'pnk', 'rng', 'tmb']

    print(f"🚀 Extracting tokens using 'finger_distinct' strategy (JSONL output)...")

    os.makedirs(opt.save_root, exist_ok=True)
    save_path = pjoin(opt.save_root, "wlasl2000_qvae_tokens.jsonl")

    n_written = 0
    with open(save_path, "w", encoding="utf-8") as f_out:
        for batch in tqdm(loader):
            motions, glosses, vids, actual_lens = batch
            motions = motions.to(opt.device)

            # Forward: 得到 out_cont, out_quant, z_cont, z_quant, loss_dict
            _, _, _, _, loss_dict = model(motions)

            for i in range(len(glosses)):
                if glosses[i] == "ERROR":
                    continue

                all_group_indices = []
                for name in group_names:
                    key = f"indices_{name}"
                    if key in loss_dict:
                        idx_tensor = loss_dict[key][i].detach().cpu().numpy()  # [T, N]
                        all_group_indices.append(idx_tensor)

                if not all_group_indices:
                    continue

                combined_tokens = np.concatenate(all_group_indices, axis=1)  # [T, 13]

                # 如果你想按真实长度截断（不存 padding token），取消下一行注释
                # combined_tokens = combined_tokens[:int(actual_lens[i])]

                record = {
                    "video_id": str(vids[i]),
                    "gloss": str(glosses[i]),
                    "tokens": combined_tokens.flatten().astype(int).tolist(),
                    "shape": [int(combined_tokens.shape[0]), int(combined_tokens.shape[1])],
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"✅ JSONL saved: {save_path}")
    print(f"✅ records    : {n_written}")

# --- 4. Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="权重 .tar 文件")
    parser.add_argument("--config_path", type=str, required=True, help="配置 .yaml 或 .txt")
    parser.add_argument("--data_dir", type=str, required=True, help="WLASL npz 文件夹")
    parser.add_argument("--save_dir", type=str, default="./output_tokens")
    parser.add_argument("--wlasl_json", type=str, default=None, help="WLASL_v0.3.json 路径（可选；不填则默认 data_dir 同目录下找）")
    args = parser.parse_args()

    # 加载配置
    opt = load_config(args.config_path)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.save_root = args.save_dir
    opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    # 关键：确保策略是 finger_distinct，否则 loss_dict 里的 key 对不上
    if not hasattr(opt, 'codebook_grouping'):
        opt.codebook_grouping = 'finger_distinct'
        print("⚠️ Warning: 'codebook_grouping' not found in config, forcing 'finger_distinct'")

    # 初始化模型
    model = VAE(opt).to(opt.device)
    
    # 加载权重
    ckpt = torch.load(args.model_path, map_location=opt.device)
    state_dict = ckpt['vae'] if 'vae' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model_path}")

        # ========== 构建 video_id -> gloss 映射 ==========
    if args.wlasl_json is None:
        guess_path = os.path.join(args.data_dir, "WLASL_v0.3.json")
        if os.path.isfile(guess_path):
            args.wlasl_json = guess_path

    vid2gloss = {}
    if args.wlasl_json is not None and os.path.isfile(args.wlasl_json):
        vid2gloss = build_videoid_to_gloss_map(args.wlasl_json)
        print(f"Loaded WLASL json mapping: {args.wlasl_json} | #vid={len(vid2gloss)}")
    else:
        print("⚠️ Warning: WLASL json not provided/found. Gloss will rely on filename or become UNKNOWN.")

    # 数据加载
    dataset = WLASLDataset(args.data_dir, opt, vid2gloss=vid2gloss)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


    extract_tokens(opt, model, loader)