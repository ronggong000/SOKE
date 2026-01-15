import torch
import torch.nn as nn
from qvae_skeleton_rod3_fixed_length import MultiLinear, MotionEncoder, MotionDecoder, STConvEncoder, STConvDecoder, VectorQuantizer


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        
        self.opt = opt

        # 1. Encoder & Decoder (参数独立，不改)
        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)
        
        if opt.dataset_name =="SMPLX_SL": expert_num=5
        elif opt.dataset_name =="HAND_CENTRIC": expert_num=9
        elif opt.dataset_name =="HIERARCHICAL": expert_num=13
        else: raise ValueError(f"Unknown joint_selection: {opt.dataset_name}")

        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, expert_num)
        # ==================== 【动态量化器配置】 ====================
        # 获取参数
        self.latent_dim = opt.latent_dim
        self.cb_size_body = getattr(opt, 'codebook_size_body', 256) # Body通常小一点
        self.cb_size_hand = getattr(opt, 'codebook_size_hand', 1024)
        self.commitment_cost = getattr(opt, 'commitment_cost', 0.25)

        # 存储量化器模块的列表 (nn.ModuleList 才能被 pytorch 识别参数)
        self.quantizers = nn.ModuleList()
        
        # 存储分组逻辑: List of dicts [{'ids': [...], 'quantizer_idx': int, 'name': str}]
        self.grouping_schedule = self._setup_quantizers(opt.codebook_grouping)

    def _setup_quantizers(self, strategy):
        """
        根据 opt.codebook_grouping 定义分组策略
        返回 schedule list，并初始化 self.quantizers
        
        节点索引回顾 (HIERARCHICAL):
        0: Torso
        1: L-Arm, 2: R-Arm
        3: L-Index, 4: L-Middle, 5: L-Pinky, 6: L-Ring, 7: L-Thumb
        8: R-Index, 9: R-Middle, 10: R-Pinky, 11: R-Ring, 12: R-Thumb
        """
        schedule = []
        
        # 辅助函数：添加一个量化器并记录分组
        def add_group(name, node_ids, size):
            # 创建新的量化器
            q = VectorQuantizer(size, self.latent_dim, self.commitment_cost)
            self.quantizers.append(q)
            q_idx = len(self.quantizers) - 1
            # 记录
            schedule.append({'name': name, 'ids': node_ids, 'q_idx': q_idx})

        if strategy == 'default':
            # 配置1: 保持现状 (Body+Arms=1, Hands=1)
            # Body (0,1,2)
            add_group('body_arms', [0, 1, 2], self.cb_size_body)
            # Shared Hands (3-12)
            add_group('shared_hands', list(range(3, 13)), self.cb_size_hand)

        elif strategy == 'arm_mirror':
            # 配置2: Torso独立, Arms镜像共享, Hands共享
            # Torso (0)
            add_group('torso', [0], self.cb_size_body)
            # Shared Arms (1, 2)
            add_group('shared_arms', [1, 2], self.cb_size_body) # Arm 动作少，用 body size 够了
            # Shared Hands (3-12)
            add_group('shared_hands', list(range(3, 13)), self.cb_size_hand)

        elif strategy == 'thumb_sep':
            # 配置3: 在配置2基础上，大拇指单独出来共享，其他4指共享
            # Torso (0)
            add_group('torso', [0], self.cb_size_body)
            # Shared Arms (1, 2)
            add_group('shared_arms', [1, 2], self.cb_size_body)
            # Shared Thumbs (7, 12)
            add_group('shared_thumbs', [7, 12], self.cb_size_hand)
            # Shared Fingers (3,4,5,6 + 8,9,10,11)
            finger_ids = [3, 4, 5, 6, 8, 9, 10, 11]
            add_group('shared_fingers', finger_ids, self.cb_size_hand)

        elif strategy == 'finger_distinct':
            # 配置4: Torso, Arms, 加上 5 根手指各一个码本 (Total 7 Codebooks)
            # Torso (0)
            add_group('torso', [0], self.cb_size_body)
            # Shared Arms (1, 2)
            add_group('shared_arms', [1, 2], self.cb_size_body)
            
            # Index (3, 8)
            add_group('idx', [3, 8], self.cb_size_hand)
            # Middle (4, 9)
            add_group('mid', [4, 9], self.cb_size_hand)
            # Pinky (5, 10)
            add_group('pnk', [5, 10], self.cb_size_hand)
            # Ring (6, 11)
            add_group('rng', [6, 11], self.cb_size_hand)
            # Thumb (7, 12)
            add_group('tmb', [7, 12], self.cb_size_hand)
        # ==================== 【新增: Full Book Mode】 ====================
        elif strategy == 'full_book':
            # 配置5: 13个节点，每个节点都有自己独立的码本 (Total 13 Codebooks)
            # 这就是完全不共享，数据最稀疏，参数量最大，用来做 Baseline
            
            # Node 0: Torso
            add_group('node_0_torso', [0], self.cb_size_body)
            # Node 1, 2: Arms
            add_group('node_1_larm', [1], self.cb_size_body)
            add_group('node_2_rarm', [2], self.cb_size_body)
            
            # Node 3-7: Left Hand
            hand_names = ['l_idx', 'l_mid', 'l_pnk', 'l_rng', 'l_tmb']
            for i, name in enumerate(hand_names):
                add_group(name, [3 + i], self.cb_size_hand)
                
            # Node 8-12: Right Hand
            hand_names_r = ['r_idx', 'r_mid', 'r_pnk', 'r_rng', 'r_tmb']
            for i, name in enumerate(hand_names_r):
                add_group(name, [8 + i], self.cb_size_hand)
        else:
            raise ValueError(f"Unknown codebook grouping strategy: {strategy}")
            
        print(f"Codebook Strategy [{strategy}]: Created {len(self.quantizers)} quantizers.")
        return schedule

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.motion_enc(x)
        x = self.conv_enc(x)

        # latent space
        x = self.dist(x)
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        
        return z, {"loss_kl": loss_kl}
    
    def decode(self, x):
        x = self.conv_dec(x)
        x = self.motion_dec(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.to(self.opt.device)

        # 1. Encode -> Continuous Z [B, T, 13, D]
        z_cont, loss_dict = self.encode(x)
        
        # 2. Dynamic Quantization Loop
        # 我们需要构建一个跟 z_cont 形状一样的 z_quant
        z_quant = torch.zeros_like(z_cont)
        
        total_quant_loss = 0.0
        
        # 临时存储 indices 用于 logging (Optional)
        all_indices = {} # name -> indices

        for group in self.grouping_schedule:
            name = group['name']
            ids = group['ids'] # List of node indices, e.g., [1, 2]
            q_idx = group['q_idx']
            quantizer = self.quantizers[q_idx]
            
            # a. 切分 Z: [B, T, len(ids), D]
            # 注意：ids 必须转为 list 才能正确索引
            z_slice = z_cont[:, :, ids, :]
            
            # b. 量化 (隐式镜像核心：左右手特征都在 z_slice 里，喂给同一个 quantizer)
            loss, z_q_slice, perp, idx = quantizer(z_slice)
            
            # c. 填回 Z_quant (Scatter back)
            # 这里的 ids 是 node 的绝对索引，可以直接赋值
            z_quant[:, :, ids, :] = z_q_slice
            
            # d. 累加 Loss 和记录指标
            total_quant_loss += loss
            
            # 记录 Usage (Perplexity)
            loss_dict[f"perplexity_{name}"] = perp
            
            # 记录 Indices (仅 Eval 模式)
            if not self.training:
                # 记录时我们为了区分，可以加上名字
                loss_dict[f"indices_{name}"] = idx

        # 3. Dual Decoding
        out_cont = self.decode(z_cont)
        out_quant = self.decode(z_quant)

        loss_dict["loss_quant"] = total_quant_loss
        
        return out_cont, out_quant, z_cont, z_quant, loss_dict

    def reset_all_codebooks(self, z_current):
        """
        封装了 Reset 逻辑，供 Trainer 调用。
        z_current: [B, T, 13, D] 当前 batch 的连续潜变量
        """
        reset_stats = {}
        total_resets = 0
        
        for group in self.grouping_schedule:
            name = group['name']
            ids = group['ids']
            q_idx = group['q_idx']
            quantizer = self.quantizers[q_idx]
            
            # 提取当前 batch 中属于该组的特征作为 pool
            z_slice = z_current[:, :, ids, :] # [B, T, N_nodes, D]
            
            # 展平: [B*T*N_nodes, D]
            z_pool = z_slice.reshape(-1, z_slice.shape[-1])
            
            # 执行 Reset
            n_reset = quantizer.reset_codebook(z_pool)
            
            if n_reset > 0:
                reset_stats[name] = n_reset
                total_resets += n_reset
                
        return total_resets, reset_stats
