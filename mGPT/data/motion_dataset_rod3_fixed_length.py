import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import random
import json
class SignMotionFixedLengthDataset(Dataset):
    def __init__(self, data_dir, max_length, is_train=True,file_list=None,config=None):
        self.data_dir = data_dir
        self.max_length = max_length
        self.is_train = is_train
        self.config=config
        # 1. 尝试读取预生成的元数据缓存 (解决启动慢的问题)
        # 假设你在 data_dir 下放了一个 metadata.json，里面存了文件名和长度
        cache_path = os.path.join(data_dir, "dataset_metadata.json")
        
        if os.path.exists(cache_path):
            print(f"Loading cached metadata from {cache_path}...")
            with open(cache_path, 'r') as f:
                metadata = json.load(f)
                # metadata 结构建议: [{"name": "xxx.npz", "len": 100}, ...]
                # 过滤逻辑放在这里
                self.samples = []
                for item in metadata:
                    if item['len'] >= 8: # 你的过滤条件
                        # 验证/测试集逻辑需要在这里展开，或者只存基本信息，getitem里动态切
                        # 为保持与你原代码逻辑一致，这里简化处理，假设 metadata 已过滤
                        if is_train:
                            self.samples.append(item['name'])
                            self.file_lengths = {x['name']: x['len'] for x in metadata}
                        else:
                            # 验证集切片逻辑 (稍微复杂点，如果 metadata 只存了文件名和总长)
                            T = item['len']
                            T = (T // 4) * 4 # 对齐
                            for start_idx in range(0, T, max_length):
                                self.samples.append((item['name'], start_idx))
        else:
            print("⚠️ Cache not found! Scanning directory (this will be slow)...")
            # --- 你的原始逻辑 (慢) ---
            # 建议：第一次运行完这个慢逻辑后，把结果保存成 json，下次就快了
            self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
            self.file_lengths = {}
            self.samples = []
            
            temp_metadata = [] # 用于保存缓存
            
            for filename in self.data_files:
                path = os.path.join(data_dir, filename)
                try:
                    # 优化点：使用 mmap_mode='r' 读取头部，不加载数据进内存，速度快很多
                    with np.load(path, mmap_mode='r') as data:
                        # 只是读取 shape，非常快
                        if self.config.xyz==True:
                            shape = data['joints_xyz'].shape 
                        else:
                            shape = data['poses'].shape 
                        T = shape[0]
                        
                        # 过滤逻辑
                        T_aligned = (T // 4) * 4
                        
                        if T_aligned >= 8:
                            self.file_lengths[filename] = T_aligned
                            temp_metadata.append({"name": filename, "len": T_aligned})
                            
                            if is_train:
                                self.samples.append(filename)
                            else:
                                for start_idx in range(0, T_aligned, max_length):
                                    self.samples.append((filename, start_idx))
                except Exception as e:
                    pass
            
            # 自动保存缓存，下次就不用等了
            try:
                with open(cache_path, 'w') as f:
                    json.dump(temp_metadata, f)
                print(f"✅ Created metadata cache at {cache_path}")
            except:
                pass

        print(f"Loaded {len(self.samples)} samples.")
    def calculate_stats(self):
        """
        计算统计量：
        - XYZ 模式：逐维 mean/std
        - ROT(轴角) 模式：mean 强制 0，更稳的 std 方案：
            rot_norm = "none"      -> std=1 不缩放（推荐先试，最不容易伤模型）
            rot_norm = "std_dim"   -> 每个维度真实 std（推荐）
            rot_norm = "std_joint" -> 每个关节(3维)用同一个 std（更保守）
        """
        print(f"📊 Calculating stats (XYZ mode: {self.config.xyz})...")
        all_data = []
        files_to_scan = self.samples if self.is_train else []

        for filename in files_to_scan:
            filepath = os.path.join(self.data_dir, filename)
            with np.load(filepath) as data:
                feat = data['joints_xyz'] if self.config.xyz else data['poses']
                feat = feat[:, self.config.SELECTED_JOINT_INDICES, :]  # [T, J, 3]
                all_data.append(feat.reshape(-1, feat.shape[1] * 3))   # [T, J*3]

        if len(all_data) == 0:
            raise RuntimeError("No training files found to calculate stats.")

        all_data = np.concatenate(all_data, axis=0).astype(np.float64)  # [Total_Frames, D]

        eps = 1e-5

        if self.config.xyz:
            mean = np.mean(all_data, axis=0)
            std = np.std(all_data, axis=0)
            std[std < eps] = 1.0
            return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

        # ---------------- ROT mode (axis-angle) ----------------
        # mean 必须保持 0（不然“无旋转”状态被平移）
        mean = np.zeros(all_data.shape[1], dtype=np.float64)

        # 选择 rot 归一化策略：尽量不要求你改 config，没有就走默认
        rot_norm = getattr(self.config, "rot_norm", "none")
        # 可选："none" / "std_dim" / "std_joint"

        if rot_norm == "none":
            std = np.ones(all_data.shape[1], dtype=np.float64)

        elif rot_norm == "std_joint":
            # 每个关节 3 维用一个 std（用真正 std，不用 RMS）
            std = np.ones(all_data.shape[1], dtype=np.float64)
            for j in range(0, all_data.shape[1], 3):
                joint = all_data[:, j:j+3]                 # [N, 3]
                joint_std = np.std(joint)                  # 标量：对这 3 维整体的真实 std
                if joint_std < eps:
                    joint_std = 1.0
                std[j:j+3] = joint_std

        else:
            # "std_dim"：逐维真实 std（通常最好）
            std = np.std(all_data, axis=0)
            std[std < eps] = 1.0

        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            filename = self.samples[idx]
            T = self.file_lengths[filename]
            if T > self.max_length:
                start_idx = random.randint(0, T - self.max_length)
            else:
                start_idx = 0
        else:
            filename, start_idx = self.samples[idx]

        filepath = os.path.join(self.data_dir, filename)
        # 2. 优化 Runtime 读取：使用 mmap_mode='r'
        # 这样当你做切片时，只会从硬盘读取你需要的那一小块数据，而不是整个文件
        # 这能显著降低内存峰值 (Memory Spike)
        try:
            with np.load(filepath, mmap_mode='r') as data:
                # 注意：mmap 返回的是磁盘上的视图
                # data['poses'] 此时没有读入内存
                
                # 这一步切片操作会触发实际的磁盘读取，但只读这一小块
                # 必须加上 .copy() 或者 np.array() 把它真正变成内存里的 array，否则转 Tensor 会报错
                if hasattr(self.config, 'xyz') and self.config.xyz==True:
                    full_motion_slice = data['joints_xyz'][start_idx : start_idx + self.max_length, self.config.SELECTED_JOINT_INDICES, :]
                else:
                    full_motion_slice = data['poses'][start_idx : start_idx + self.max_length, self.config.SELECTED_JOINT_INDICES, :]
                
                # 这里的切片逻辑需要稍微调整，因为我们不能先读 full 再切，那样就失去 mmap 的意义了
                # 现在的逻辑：直接读取需要的 time slice 和 joint slice
                
                # 为了安全，先读取到 numpy (这就进入内存了，但只有一小块)
                motion_data = np.array(full_motion_slice) 
                
        except Exception as e:
            # 容错处理：返回全0或者随机数据防止 crash
            print(f"Error loading {filename}: {e}")
            return torch.zeros(self.max_length, len(self.config.SELECTED_JOINT_INDICES)*3), torch.tensor(0)

        # 3. 后处理 (Padding 等)
        # 因为我们上面是直接按 start_idx + max_length 切的，可能切出来的长度不够
        original_len = motion_data.shape[0]
        
        if original_len < self.max_length:
            pad_len = self.max_length - original_len
            last_frame = motion_data[-1:]
            padding = np.repeat(last_frame, pad_len, axis=0)
            motion_data = np.concatenate([motion_data, padding], axis=0)

        # Flatten
        motion_flat = motion_data.reshape(self.max_length, -1).astype(np.float32)
            
        return torch.from_numpy(motion_flat), torch.tensor(original_len)

def simple_collate_fn(batch):
    # batch 现在是一个元组列表: [(motion1, len1), (motion2, len2), ...]
    motions, lengths = zip(*batch)
    
    # motions 是一个张量元组，lengths 是一个张量元组
    # 将它们分别堆叠成一个大的批次张量
    stacked_motions = torch.stack(motions, dim=0)
    stacked_lengths = torch.stack(lengths, dim=0)
    
    # 返回两个张量：一个是批次数据，另一个是对应的长度
    return stacked_motions, stacked_lengths

def create_data_loaders(train_data_dir, val_data_dir, test_data_dir, batch_size, config=None):
    """
    从三个独立的文件夹创建训练、验证和测试的 DataLoader。
    """
    num_workers = config.num_workers
    max_length = config.max_length

    # 1. 为训练集创建 Dataset 实例
    # 它会自动扫描 train_data_dir 文件夹
    train_dataset = SignMotionFixedLengthDataset(
        data_dir=train_data_dir, 
        max_length=max_length, 
        is_train=True, 
        file_list=None, # 设为 None, 让 Dataset 自己扫描
        config=config
    )
    
    # 2. 为验证集创建 Dataset 实例
    # 它会自动扫描 val_data_dir 文件夹
    val_dataset = SignMotionFixedLengthDataset(
        data_dir=val_data_dir, 
        max_length=max_length, 
        is_train=False, 
        file_list=None, # 设为 None, 让 Dataset 自己扫描
        config=config
    )

    # 3. 为测试集创建 Dataset 实例
    # 它会自动扫描 test_data_dir 文件夹
    test_dataset = SignMotionFixedLengthDataset(
        data_dir=test_data_dir, 
        max_length=max_length, 
        is_train=False, # 测试集与验证集一样，is_train=False
        file_list=None, # 设为 None, 让 Dataset 自己扫描
        config=config
    )

    # 4. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # 训练集需要打乱
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # 验证集不需要打乱
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # 测试集不需要打乱
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=True,
        drop_last=True, # 你可以根据需要决定测试集是否 drop_last
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"Data loaders created.")
    return train_loader, val_loader, test_loader