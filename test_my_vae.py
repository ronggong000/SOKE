import argparse
import ast
import os

import smplx
import torch
from types import SimpleNamespace
from mymodel.tools.evaluator_rod3_fixed_length import MotionEvaluator
from mymodel.vae.qvae_model_rod3_fixed_length import VAE
from mymodel.vae.qvae_option_fixed_length import arg_parse
from mGPT.data.motion_dataset_rod3_fixed_length import create_data_loaders


from mGPT.utils.joints_list import SMPLX_JOINT_LANDMARK_NAMES, SELECTED_JOINT_INDICES,SELECTED_JOINT_LANDMARK_INDICES,SELECTED_JOINT_LANDMARK_BODY_EVAL,SELECTED_JOINT_LANDMARK_LHAND_EVAL,SELECTED_JOINT_LANDMARK_RHAND_EVAL, SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,SELECTED_JOINT_INDICES_BODY_ONLY,SELECTED_JOINT_INDICES_NEIGHBOR_LIST

def _convert_string_to_type(s):
    s = s.strip()
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    if s.startswith('[') and s.endswith(']'):
        items = s[1:-1].split(',')
        return [_convert_string_to_type(item) for item in items]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def load_config_from_txt(path):
    print(f"正在从 TXT 文件 '{path}' 加载配置...")
    config_dict = {}
    with open(path, 'r') as f:
        for line in f:
            if '---' in line:
                continue
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                config_dict[key] = _convert_string_to_type(value_str)
    return SimpleNamespace(**config_dict)


def chunk_pad(chunk, max_length):
    original_len = chunk.shape[0]
    if original_len < max_length:
        pad_len = max_length - original_len
        last_frame = chunk[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        chunk = np.concatenate([chunk, padding], axis=0)
    return chunk, original_len


def main():


    parser = argparse.ArgumentParser(description="使用训练好的 QVAE/双路VAE 模型进行动作重建，并导出 npz 以便肉眼检查")
    parser.add_argument("--model_path", default="checkpoints/vae/qvae_b256h1024_L1_fingerdistinct/model/latest.tar", type=str, help="已训练的模型权重文件路径 (.tar)")
    parser.add_argument("--config_path", default="checkpoints/vae/qvae_b256h1024_L1_fingerdistinct/opt.txt", type=str, help="配置文件路径 (支持 .yaml 或 .txt)")
    #parser.add_argument("--mode", default="quant", choices=["quant", "cont", "both"],help="cont=连续路, quant=码本路, both=两路都保存（推荐用于对比）")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    args = parser.parse_args()

    opt = load_config_from_txt(args.config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    print(f"计算设备: {device}")

    # 2) 补齐旧配置可能缺字段
    if not hasattr(opt, 'data_format'):
        opt.data_format = 'motion_dataset_rod3_fixed_length'

    opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    opt.joints_num = len(SELECTED_JOINT_INDICES)
    opt.smplx_model_path = "deps/smpl_models"
    # 3) 加载模型
    print(f"正在从 '{args.model_path}' 加载模型权重...")
    vae = VAE(opt).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    # 兼容两种保存方式：checkpoint['vae'] / checkpoint['model'] / 直接 state_dict
    if isinstance(checkpoint, dict) and 'vae' in checkpoint:
        vae.load_state_dict(checkpoint['vae'], strict=True)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        vae.load_state_dict(checkpoint['model'], strict=True)
    elif isinstance(checkpoint, dict):
        # 有些人直接 torch.save(model.state_dict())
        try:
            vae.load_state_dict(checkpoint, strict=True)
        except Exception:
            # 最后兜底：如果里面是其他key，直接报错让你看
            raise RuntimeError(f"checkpoint keys: {list(checkpoint.keys())}")
    else:
        raise RuntimeError("checkpoint 格式不对：不是 dict")

    vae.eval()
    print("模型加载成功。")

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data_dir=opt.train_data_dir,
        val_data_dir=opt.val_data_dir,
        test_data_dir=opt.test_data_dir,
        batch_size=opt.batch_size,
        config=opt,
    )

    if args.split == "train":
        eval_loader = train_loader
    elif args.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    smplx_model = smplx.create(
        model_path=opt.smplx_model_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=opt.batch_size * opt.max_length,
    ).to(device)
    smplx_model.eval()

    evaluator = MotionEvaluator(opt=opt)
    metrics = evaluator.calculate_metrics(vae, eval_loader, smplx_model)

    print("==== VAE Evaluation Metrics ====")
    for metric_name, metric_values in metrics.items():
        print(f"{metric_name}:")
        for key, value in metric_values.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
