import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from rotation_utils import CorrectionTransform, axis_angle_to_matrix, matrix_to_axis_angle,rotation_6d_to_matrix

from torch.amp import autocast 


def adj_list_to_edges(adj_list):
    edges = []
    for i, adj in enumerate(adj_list):
        for j in adj:
            if i < j:
                edges.append((i, j))
    return edges

def edges_to_adj_list(edges):
    max_idx = -1
    for i, j in edges:
        max_idx = max(max_idx, i, j)

    adj_list = [[] for _ in range(max_idx + 1)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return adj_list

def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_norm(name, dim):
    if name.lower() == "layer":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch":
        return nn.BatchNorm1d(dim)
    elif name.lower() == "group":
        return nn.GroupNorm(32, dim)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization function: {name}")


class GraphConv(nn.Module):
    """
    Graph Convolution.
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)
    
    
    def forward(self, x, adj_matrix):
        """
        x: [B, T, J, D]
        adj_matrix: [J, J]

        return: linear1(x) + sum_{j \in N(i)} linear2(x_j)
        """

        h1 = self.linear1(x) # [B, T, J, D]
        h2 = torch.einsum('ij,btjd->btid', adj_matrix, self.linear2(x))
        h2 = h2 / (adj_matrix.sum(dim=-1, keepdim=True).unsqueeze(0).unsqueeze(0) + 1e-6)
        out = h1 + h2

        return out


class STConv(nn.Module):
    """
    Skeleto-Temporal Convolution.
    """
    def __init__(
        self,
        edges,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
    ):
        assert kernel_size % 2 == 1, f"kernel_size should be odd number, but got {kernel_size}."

        super(STConv, self).__init__()

        # grpah convolution
        self.graph_conv = GraphConv(in_channels, out_channels, bias=bias)
        self.adj_matrix = nn.Parameter(self._get_adj_matrix(edges), requires_grad=False) # symmetric matrix

        # temporal convolution
        self.temp_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
    
    def _get_adj_matrix(self, edges, add_self_loop=True):
        max_idx = -1
        for i, j in edges:
            max_idx = max(max_idx, i, j)

        adj_matrix = torch.zeros(max_idx + 1, max_idx + 1)
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        if add_self_loop:
            for i in range(max_idx + 1):
                adj_matrix[i, i] = 1
        
        return adj_matrix

    def forward(self, x):
        B, T, J, D = x.size()
        assert x.shape[2] == self.adj_matrix.shape[0], f"x.shape={x.shape}, adj_matrix.shape={self.adj_matrix.shape}"
        # graph conv
        graph_out = self.graph_conv.forward(x, self.adj_matrix)

        # temporal conv
        temp_in = x.permute(0, 2, 3, 1).reshape(B * J, D, T)
        temp_out = self.temp_conv(temp_in)
        temp_out = temp_out.reshape(B, J, -1, T).permute(0, 3, 1, 2)

        out = graph_out + temp_out

        return out
    
class ResSTConv(nn.Module):
    """
    Residual Skeleto-Temporal Convolution.
    """
    def __init__(
        self,
        edges,
        dim_channels,
        kernel_size,
        bias=True,
        activation="gelu",
        norm="none",
        dropout=0.1,
    ):
        super(ResSTConv, self).__init__()
        self.norm1 = get_norm(norm, dim_channels)
        self.act1 = get_activation(activation)
        self.st_conv1 = STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias)
        
        self.norm2 = get_norm(norm, dim_channels)
        self.act2 = get_activation(activation)
        self.st_conv2 = STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 记录原始输入，用于残差连接
        identity = x
        
        # 第一个卷积块
        out = self.norm1(x)
        out = self.act1(out)
        out = self.st_conv1(out) # 调用独立的层
        
        # 第二个卷积块
        out = self.norm2(out)
        out = self.act2(out)
        out = self.st_conv2(out) # 调用独立的层
        
        out = self.dropout(out)
        
        # 添加残差连接
        out = identity + out
        
        return out

class MultiLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_experts,
        bias: bool = True,
    ):
        super(MultiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: [*, num_experts, in_features]
        out: [*, num_experts, out_features]
        """
        out = torch.matmul(x.unsqueeze(-2), self.weight).squeeze(-2)
        if self.bias is not None:
            out += self.bias
        return out
    

class STPool(nn.Module):
    """
    Skeleto-Temporal Pooling.
    """
    def __init__(
        self,
        joint_selection=None,#["SMPLX_SL","HAND_CENTRIC","HIERARCHICAL"]
        depth=0,
    ):
        if not joint_selection in ["SMPLX_SL","HAND_CENTRIC","HIERARCHICAL"]:
            raise ValueError(f"joint_selection should be 'SMPLX_SL', but got {joint_selection} ")
        
        super(STPool, self).__init__()

        self.skeleton_pool, self.skeleton_mapping, self.new_edges = self._get_skeleton_pooling(joint_selection, depth)
        self.skeleton_pool = nn.Parameter(self.skeleton_pool, requires_grad=False) # [J_out, J_in]
        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def _get_skeleton_pooling(self, joint_selection, depth):
        if joint_selection == "SMPLX_SL":
            if depth == 0:
                # 43 -> 16 (与原来相同)
                weight = torch.zeros(16, 43)
                mapping = [
                    ((0, 1, 2, 3), 0),      # torso
                    ((4, 7, 9, 11), 1),     # left_arm
                    ((5, 8, 10, 12), 2),    # right_arm
                    ((9,11,25), 3),         # left_wrist
                    ((10,12,40), 4),        # right_wrist
                    ((3,6), 5),             # head
                    ((13, 14, 15), 6),      # left_index
                    ((16, 17, 18), 7),      # left_middle
                    ((19, 20, 21), 8),      # left_pinky
                    ((22, 23, 24), 9),      # left_ring
                    ((25, 26, 27), 10),     # left_thumb
                    ((28, 29, 30), 11),     # right_index
                    ((31, 32, 33), 12),     # right_middle
                    ((34, 35, 36), 13),     # right_pinky
                    ((37, 38, 39), 14),     # right_ring
                    ((40, 41, 42), 15),     # right_thumb
                ]
                new_edges = adj_list_to_edges([
                    [1, 2, 5], [0, 3], [0, 4], [1, 6,7,8,9,10], [2, 11,12,13,14,15],
                    [0], [3], [3], [3], [3], [3], [4], [4], [4], [4], [4],
                ])
            elif depth == 1:
                # 16 -> 5 (与原来相同)
                weight = torch.zeros(5, 16)
                mapping = [
                    ((0, 5), 0), # upper_body
                    ((1, 3), 1), # left_arm_hand
                    ((2, 4), 2), # right_arm_hand
                    ((6, 7, 8, 9, 10), 3), # left_fingers
                    ((11, 12, 13, 14, 15), 4), # right_fingers
                ]
                new_edges = adj_list_to_edges([[1, 2], [0, 3], [0, 4], [1], [2]])
            else: # 最后一层，保持不变
                weight = torch.eye(5)
                mapping = [((i,), i) for i in range(5)]
                new_edges = adj_list_to_edges([[1, 2], [0, 3], [0, 4], [1], [2]])

        elif joint_selection == "HAND_CENTRIC":
            if depth == 0:
                # 43 -> 16 (与 SMPLX_SL 的第一层相同)
                weight = torch.zeros(16, 43)
                mapping = [((0,1,2,3),0),((4,7,9,11),1),((5,8,10,12),2),((9,11,25),3),((10,12,40),4),((3,6),5),((13,14,15),6),((16,17,18),7),((19,20,21),8),((22,23,24),9),((25,26,27),10),((28,29,30),11),((31,32,33),12),((34,35,36),13),((37,38,39),14),((40,41,42),15)]
                new_edges = adj_list_to_edges([[1,2,5],[0,3],[0,4],[1,6,7,8,9,10],[2,11,12,13,14,15],[0],[3],[3],[3],[3],[3],[4],[4],[4],[4],[4]])
            elif depth == 1:
                # 16 -> 9 (保留手部细节)
                weight = torch.zeros(9, 16)
                mapping = [
                    ((0, 5), 0),             # 0: upper_body (torso, head)
                    ((1,), 1),               # 1: left_arm
                    ((2,), 2),               # 2: right_arm
                    ((3,), 3),               # 3: left_wrist
                    ((4,), 4),               # 4: right_wrist
                    ((10,), 5),              # 5: left_thumb
                    ((15,), 6),              # 6: right_thumb
                    ((6, 7, 8, 9), 7),       # 7: left_other_fingers
                    ((11, 12, 13, 14), 8),   # 8: right_other_fingers
                ]
                new_edges = adj_list_to_edges([
                    [1, 2],      # 0: upper_body -> arms
                    [0, 3],      # 1: left_arm -> body, wrist
                    [0, 4],      # 2: right_arm -> body, wrist
                    [1, 5, 7],   # 3: left_wrist -> arm, thumb, fingers
                    [2, 6, 8],   # 4: right_wrist -> arm, thumb, fingers
                    [3], [4], [3], [4] # 5,6,7,8: leaves
                ])
            else: # 最后一层
                weight = torch.eye(9)
                mapping = [((i,), i) for i in range(9)]
                new_edges = adj_list_to_edges([[1,2],[0,3],[0,4],[1,5,7],[2,6,8],[3],[4],[3],[4]])
        elif joint_selection == "HIERARCHICAL":
            if depth == 0:
                # 与 SMPLX_SL 第一层一致：43 -> 16
                weight = torch.zeros(16, 43)
                mapping = [
                    ((0, 1, 2, 3), 0),      # torso =spines+neck
                    ((4, 7, 9, 11), 1),     # left_arm =collar+shoulder+elbow+wrist
                    ((5, 8, 10, 12), 2),    # right_arm
                    ((9,11,25), 3),         # left_wrist = elbow + wrist + thumb1
                    ((10,12,40), 4),        # right_wrist
                    ((3,6), 5),             # head = neck + head
                    ((13, 14, 15), 6),      # left_index
                    ((16, 17, 18), 7),      # left_middle
                    ((19, 20, 21), 8),      # left_pinky
                    ((22, 23, 24), 9),      # left_ring
                    ((25, 26, 27),10),      # left_thumb
                    ((28, 29, 30),11),      # right_index
                    ((31, 32, 33),12),      # right_middle
                    ((34, 35, 36),13),      # right_pinky
                    ((37, 38, 39),14),      # right_ring
                    ((40, 41, 42),15),      # right_thumb
                ]
                new_edges = adj_list_to_edges([
                    [1, 2, 5], [0, 3], [0, 4], [1, 6,7,8,9,10], [2, 11,12,13,14,15],
                    [0], [3], [3], [3], [3], [3], [4], [4], [4], [4], [4],
                ])

            elif depth == 1:
                # 16 -> 13：upper_body、左右臂+腕、10根手指（左5+右5，全部保留）
                # 这里的输入通道索引基于 depth=0 的输出顺序：
                # 0:torso, 5:head, 1:left_arm, 3:left_wrist, 2:right_arm, 4:right_wrist,
                # 6..10: 左手 [index, middle, pinky, ring, thumb]
                # 11..15: 右手 [index, middle, pinky, ring, thumb]
                weight = torch.zeros(13, 16)
                mapping = [
                    ((0, 5), 0),        # 0: upper_body (torso + head)
                    ((1, 3), 1),        # 1: left_arm_wrist
                    ((2, 4), 2),        # 2: right_arm_wrist
                    ((6,), 3),          # 3: left_index
                    ((7,), 4),          # 4: left_middle
                    ((8,), 5),          # 5: left_pinky
                    ((9,), 6),          # 6: left_ring
                    ((10,), 7),         # 7: left_thumb
                    ((11,), 8),         # 8: right_index
                    ((12,), 9),         # 9: right_middle
                    ((13,),10),         # 10: right_pinky
                    ((14,),11),         # 11: right_ring
                    ((15,),12),         # 12: right_thumb
                ]
                new_edges = adj_list_to_edges([
                    [1, 2],        # 0: upper_body <-> arms
                    [0, 3, 4, 5, 6, 7],     # 1: left_arm_wrist <-> left fingers
                    [0, 8, 9,10,11,12],     # 2: right_arm_wrist <-> right fingers
                    [1], [1], [1], [1], [1],   # 3..7: 左手五指 -> left_arm_wrist
                    [2], [2], [2], [2], [2],   # 8..12: 右手五指 -> right_arm_wrist
                ])

            else:
                # 最后一层：保持不变
                weight = torch.eye(13)
                mapping = [((i,), i) for i in range(13)]
                new_edges = adj_list_to_edges([
                    [1, 2], [0, 3, 4, 5, 6, 7], [0, 8, 9,10,11,12],
                    [1], [1], [1], [1], [1],
                    [2], [2], [2], [2], [2],
                ])

            # —— 通用：按照 mapping 填权重并归一化 ——
            for joints, idx in mapping:
                weight[idx, joints] = 1
            weight_sum = weight.sum(dim=1, keepdim=True)
            weight_sum[weight_sum == 0] = 1
            weight = weight / weight_sum

        else:
            raise ValueError(f"Unknown joint_selection strategy: {joint_selection}")

        for joints, idx in mapping:
            weight[idx, joints] = 1
        
        # Normalize weights
        weight_sum = weight.sum(dim=1, keepdim=True)
        weight_sum[weight_sum == 0] = 1 # Avoid division by zero for empty groups
        weight = weight / weight_sum

        return weight, mapping, new_edges


    def forward(self, x):
        """
        x: [B, T, J, D]
        out: [B, T // 2, J_out, D]
        """
        B, T, J_in, D = x.size()

        # skeleton pooling
        out = torch.matmul(self.skeleton_pool, x) # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal pooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_pool(out)
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2) # [B, T // 2, J_out, D]

        return out
    
class STUnpool(nn.Module):
    """
    Skeleton-Temporal Unpooling.
    """
    def __init__(
        self,
        skeleton_mapping,
    ):
        super(STUnpool, self).__init__()
        self.skeleton_unpool = nn.Parameter(self._get_skeleton_unpool(skeleton_mapping), requires_grad=False) # [J_out, J_in]
        self.temporal_unpool = nn.Upsample(scale_factor=2, mode="linear")
        
    def _get_skeleton_unpool(self, skeleton_mapping):
        max_idx = -1
        for joints, idx in skeleton_mapping:
            max_idx = max(max_idx, *joints)

        weight = torch.zeros(max_idx + 1, len(skeleton_mapping))
        for joints, idx in skeleton_mapping:
            weight[joints, idx] = 1
            
        return weight
    
    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T * upsample_rate, J_in, D]
        """

        B, T, J_in, D = x.size()

        # skeleton unpooling
        out = torch.matmul(self.skeleton_unpool, x) # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal unpooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_unpool(out)
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2) # [B, T * upsample_rate, J_out, D]

        return out


class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()
        self.opt = opt
        self.latent_dim = opt.latent_dim
        self.feature_type = opt.data_format
        self.joints_num = len(opt.SELECTED_JOINT_INDICES)

        if self.feature_type == 'motion_dataset_rod3_fixed_length_dk':
            # --- 版本1的逻辑：为每个关节创建独立的MLP ---
            print("MotionEncoder: Initializing in HETEROGENEOUS mode.")
            self.joint_feature_dims = opt.joint_feature_dims
            self.feature_spans = tuple(self.joint_feature_dims)
            
            self.layers = nn.ModuleList()
            for input_dim in self.joint_feature_dims:
                self.layers.append(nn.Sequential(
                    nn.Linear(input_dim, self.latent_dim),
                    get_activation(opt.activation),
                    nn.Linear(self.latent_dim, self.latent_dim),
                ))
        
        # ==================== 【代码修改区-开始】 ====================
        elif self.feature_type == 'motion_dataset_rod3_fixed_length' and opt.reduce_dim_finger==True:
            print("MotionEncoder: Initializing in Canonical Axis-Angle mode.")
            
            # --- 1. 加载新的变换字典 ---
            print(f"Loading learned per-joint transforms from: {opt.transform_path}")
            learned_transforms_dict = torch.load(opt.transform_path, map_location='cpu')

            # --- 2. 从字典手动构建完整的 R_corr 和 R_corr_inv 矩阵 ---
            R_corr_all_hands = torch.zeros(30, 3, 3)
            reflect = torch.eye(3); reflect[0, 0] = -1

            # 假设右手关节在43关节列表中的索引是28到42
            for i in range(15):
                joint_idx_in_43 = 28 + i
                
                if joint_idx_in_43 in learned_transforms_dict:
                    learned_6d = learned_transforms_dict[joint_idx_in_43]
                    
                    R_corr_right_joint = rotation_6d_to_matrix(learned_6d)
                    R_corr_left_joint = reflect @ R_corr_right_joint @ reflect
                    
                    # 存放到正确的位置
                    R_corr_all_hands[i] = R_corr_left_joint       # 左手0-14
                    R_corr_all_hands[15 + i] = R_corr_right_joint # 右手15-29
                else:
                    print(f"Warning: Transform for joint index {joint_idx_in_43} not found in file. Using identity.")
                    R_corr_all_hands[i] = torch.eye(3)
                    R_corr_all_hands[15 + i] = torch.eye(3)

            # --- 3. 将构建好的矩阵存入 buffer ---
            self.register_buffer("R_corr_buf", R_corr_all_hands.clone())
            self.register_buffer("R_corr_inv_buf", torch.inverse(R_corr_all_hands).clone())
            
            # --- 4. 【性能优化】建立分组的 MultiLinear 层 ---
            self.joint_feature_dims = opt.joint_feature_dims
            self.feature_spans = tuple(self.joint_feature_dims)
            
            # a. 找到所有独特的输入维度并为它们创建分组
            self.dim_groups = {}
            for i, dim in enumerate(self.joint_feature_dims):
                if dim not in self.dim_groups:
                    self.dim_groups[dim] = []
                self.dim_groups[dim].append(i)

            # b. 为每个分组创建一个高效的 MultiLinear 层
            self.group_layers1 = nn.ModuleDict()
            self.group_layers2 = nn.ModuleDict()
            for dim, joint_indices in self.dim_groups.items():
                num_joints_in_group = len(joint_indices)
                # Key must be a string for ModuleDict
                key = str(dim)
                self.group_layers1[key] = MultiLinear(dim, self.latent_dim, num_joints_in_group)
                self.group_layers2[key] = MultiLinear(self.latent_dim, self.latent_dim, num_joints_in_group)

            self.activation = get_activation(opt.activation)
        # ==================== 【代码修改区-结束】 ====================

        elif self.feature_type == 'motion_dataset_rod3_fixed_length':
            # --- 版本2的逻辑：使用高效的Conv1d ---
            print("MotionEncoder: Initializing in UNIFORM mode.")
            input_dim = 3
            self.layer1 = nn.Conv1d(self.joints_num * input_dim, 
                                      self.joints_num * self.latent_dim, 
                                      kernel_size=1, 
                                      groups=self.joints_num)
            self.activation = get_activation(opt.activation)
            self.layer2 = nn.Conv1d(self.joints_num * self.latent_dim, 
                                      self.joints_num * self.latent_dim, 
                                      kernel_size=1, 
                                      groups=self.joints_num)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def forward(self, x):
        B, T, _ = x.size()

        if self.feature_type == 'motion_dataset_rod3_fixed_length_dk':
            x_split = torch.split(x, self.feature_spans, dim=-1)
            out_list = [self.layers[i](x_split[i]) for i in range(self.joints_num)]
            out = torch.stack(out_list, dim=2)
        
        # ==================== 【代码修改区-开始】 ====================
        elif self.feature_type == 'motion_dataset_rod3_fixed_length' and self.opt.reduce_dim_finger==True:
            x_poses = x.view(B, T, self.joints_num, 3)

            # 1. 分离手部和非手部关节
            x_hands_aa = x_poses[:, :, self.opt.hand_joint_indices, :]

            # 2. 几何变换: 原始轴角 -> 规范化轴角
            with torch.amp.autocast(device_type='cuda', enabled=False):
                x_hands_aa_f32 = x_hands_aa.view(-1, 3).to(torch.float32)
                
                # 原始轴角 -> 旋转矩阵
                R_smplx_hands = axis_angle_to_matrix(x_hands_aa_f32).view(B, T, -1, 3, 3)

                # 在矩阵空间进行修正
                R_corr_inv = self.R_corr_inv_buf.to(x.device)
                R_canonical = torch.matmul(R_corr_inv.unsqueeze(0).unsqueeze(0), R_smplx_hands)

                # 【关键改动】修正后的旋转矩阵 -> 修正后的轴角
                aa_canonical_f32 = matrix_to_axis_angle(R_canonical.view(-1, 3, 3)).view(B, T, -1, 3)

            aa_canonical = aa_canonical_f32.to(x.dtype)

            # 3. 根据关节类型提取1D, 2D或3D特征
            reduced_features = []
            for i in range(self.joints_num):
                if i in self.opt.non_hand_joint_indices:
                    # 非手部关节直接使用原始的3D轴角
                    reduced_features.append(x_poses[:, :, i, :])
                else:
                    # 手部关节使用降维后的规范化轴角
                    relative_hand_idx = self.opt.hand_joint_indices.index(i)
                    if relative_hand_idx in self.opt.mcp_indices_in_hand:
                        # MCP关节保留2个维度 (Y, Z分量)
                        reduced_features.append(aa_canonical[:, :, relative_hand_idx, 1:])
                    else:
                        # PIP/DIP关节只保留1个维度 (Z分量)
                        reduced_features.append(aa_canonical[:, :, relative_hand_idx, 2:])
            
            x_reduced_flat = torch.cat(reduced_features, dim=-1)


            # 4. 【性能优化】通过分组MLP高效编码
            x_split = torch.split(x_reduced_flat, self.feature_spans, dim=-1)
            
            # a. 初始化一个列表来存储每个关节的输出
            results = [None] * self.joints_num

            # b. 遍历每个维度分组
            for dim, joint_indices in self.dim_groups.items():
                key = str(dim)
                
                # i. 收集这个分组需要的所有输入张量
                group_inputs_list = [x_split[i] for i in joint_indices]
                
                # ii. 将它们堆叠成一个批次，送入该分组的 MultiLinear 层
                #    Shape becomes [B, T, num_joints_in_group, dim]
                group_inputs_stacked = torch.stack(group_inputs_list, dim=2)
                
                # iii. 一次性完成该分组所有关节的计算
                out = self.group_layers1[key](group_inputs_stacked)
                out = self.activation(out)
                group_outputs = self.group_layers2[key](out)
                
                # iv. 将计算结果“散布”回它们在结果列表中的正确位置
                for i, joint_idx in enumerate(joint_indices):
                    results[joint_idx] = group_outputs[:, :, i, :]
                    
            # c. 最后，将所有关节的有序结果堆叠成最终输出
            out = torch.stack(results, dim=2)
        # ==================== 【代码修改区-结束】 ====================
            
        elif self.feature_type == 'motion_dataset_rod3_fixed_length':
            x_permuted = x.permute(0, 2, 1)
            out = self.layer1(x_permuted)
            out = self.activation(out)
            out = self.layer2(out)
            out = out.reshape(B, self.joints_num, self.latent_dim, T).permute(0, 3, 1, 2)
        
        return out

class MotionDecoder(nn.Module):
    def __init__(self, opt):
        super(MotionDecoder, self).__init__()
        self.opt = opt
        self.latent_dim = opt.latent_dim
        self.feature_type = opt.data_format
        self.joints_num = len(opt.SELECTED_JOINT_INDICES)
        
        if self.feature_type == 'motion_dataset_rod3_fixed_length_dk':
            # 这个分支的逻辑保持不变，因为它已经使用了高效的 MultiLinear
            print("MotionDecoder: Initializing in HETEROGENEOUS mode.")
            self.joint_feature_dims = opt.joint_feature_dims
            
            self.layers = nn.ModuleList()
            for output_dim in self.joint_feature_dims:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    get_activation(opt.activation),
                    nn.Linear(self.latent_dim, output_dim),
                ))

        # ==================== 【代码修改区-开始】 ====================
        elif self.feature_type == 'motion_dataset_rod3_fixed_length' and opt.reduce_dim_finger==True:
            print("MotionDecoder: Initializing in Canonical Axis-Angle mode with EFFICIENT GROUP PROCESSING.")
            
            # --- 1 & 2. 加载和构建几何变换矩阵 (这部分逻辑不变) ---
            print(f"Loading learned per-joint transforms from: {opt.transform_path}")
            learned_transforms_dict = torch.load(opt.transform_path, map_location='cpu')
            R_corr_all_hands = torch.zeros(30, 3, 3)
            reflect = torch.eye(3); reflect[0, 0] = -1
            for i in range(15):
                joint_idx_in_43 = 28 + i
                if joint_idx_in_43 in learned_transforms_dict:
                    learned_6d = learned_transforms_dict[joint_idx_in_43]
                    R_corr_right_joint = rotation_6d_to_matrix(learned_6d)
                    R_corr_left_joint = reflect @ R_corr_right_joint @ reflect
                    R_corr_all_hands[i] = R_corr_left_joint
                    R_corr_all_hands[15 + i] = R_corr_right_joint
                else:
                    print(f"Warning: Transform for joint index {joint_idx_in_43} not found in file. Using identity.")
                    R_corr_all_hands[i] = torch.eye(3)
                    R_corr_all_hands[15 + i] = torch.eye(3)

            # --- 3. 将矩阵存入 buffer (这部分逻辑不变) ---
            self.register_buffer("R_corr_buf", R_corr_all_hands.clone())
            self.register_buffer("R_corr_inv_buf", torch.inverse(R_corr_all_hands).clone())
            
            # --- 4.【性能优化】建立分组的 MultiLinear 层 ---
            self.joint_feature_dims = opt.joint_feature_dims
            
            # a. 找到所有独特的输出维度并为它们创建分组
            self.dim_groups = {}
            for i, dim in enumerate(self.joint_feature_dims):
                if dim not in self.dim_groups:
                    self.dim_groups[dim] = []
                self.dim_groups[dim].append(i)

            # b. 为每个分组创建一个高效的 MultiLinear MLP (两层)
            self.group_layers1 = nn.ModuleDict()
            self.group_layers2 = nn.ModuleDict()
            for dim, joint_indices in self.dim_groups.items():
                num_joints_in_group = len(joint_indices)
                key = str(dim) # ModuleDict 的 key 必须是字符串
                self.group_layers1[key] = MultiLinear(self.latent_dim, self.latent_dim, num_joints_in_group)
                self.group_layers2[key] = MultiLinear(self.latent_dim, dim, num_joints_in_group)
            
            self.activation = get_activation(opt.activation)

        # ==================== 【代码修改区-结束】 ====================

        elif self.feature_type == 'motion_dataset_rod3_fixed_length':
            # 这个分支的逻辑保持不变
            print("MotionDecoder: Initializing in UNIFORM mode.")
            output_dim = 3
            self.layer1 = nn.Conv1d(self.joints_num * self.latent_dim, 
                                      self.joints_num * self.latent_dim, 
                                      kernel_size=1, 
                                      groups=self.joints_num)
            self.activation = get_activation(opt.activation)
            self.layer2 = nn.Conv1d(self.joints_num * self.latent_dim, 
                                      self.joints_num * output_dim, 
                                      kernel_size=1, 
                                      groups=self.joints_num)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def forward(self, x):
        B, T, J, D = x.size()
        
        if self.feature_type == 'motion_dataset_rod3_fixed_length_dk':
            # 这个分支的逻辑保持不变
            out_list = [self.layers[i](x[:, :, i, :]) for i in range(self.joints_num)]
            motion = torch.cat(out_list, dim=-1)

        # ==================== 【代码修改区-开始】 ====================
        elif self.feature_type == 'motion_dataset_rod3_fixed_length' and self.opt.reduce_dim_finger==True:
            
            # 1.【性能优化】通过分组MLP高效解码
            # a. 初始化一个列表来存储每个关节的输出
            out_list = [None] * self.joints_num
            
            # b. 遍历每个维度分组
            for dim, joint_indices in self.dim_groups.items():
                key = str(dim)
                
                # i. 收集这个分组的输入潜变量 [B, T, num_joints_in_group, D_latent]
                group_inputs = x[:, :, joint_indices, :]
                
                # ii. 一次性完成该分组所有关节的计算
                out = self.group_layers1[key](group_inputs)
                out = self.activation(out)
                group_outputs = self.group_layers2[key](out) # -> [B, T, num_joints_in_group, dim]
                
                # iii. 将计算结果“散布”回它们在结果列表中的正确位置
                for i, joint_idx in enumerate(joint_indices):
                    out_list[joint_idx] = group_outputs[:, :, i, :]
            
            # --- 从这里开始，后续的所有逻辑都保持不变，因为`out_list`的结构和之前完全一样 ---

            # 2. 分离与升维: 重建完整的3D规范化轴角
            pred_non_hands_aa_list = []
            pred_hands_aa_canonical_list = []
            for i in range(self.joints_num):
                if i in self.opt.non_hand_joint_indices:
                    pred_non_hands_aa_list.append(out_list[i])
                else:
                    relative_hand_idx = self.opt.hand_joint_indices.index(i)
                    zero_pad = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)
                    if relative_hand_idx in self.opt.mcp_indices_in_hand:
                        # 2D -> 3D: 补上被丢弃的X分量
                        pred_hands_aa_canonical_list.append(torch.cat([zero_pad, out_list[i]], dim=-1))
                    else:
                        # 1D -> 3D: 补上被丢弃的X, Y分量
                        pred_hands_aa_canonical_list.append(torch.cat([zero_pad, zero_pad, out_list[i]], dim=-1))

            pred_non_hands_aa = torch.stack(pred_non_hands_aa_list, dim=2)
            pred_hands_aa_canonical = torch.stack(pred_hands_aa_canonical_list, dim=2)

            # 3. 几何变换: 规范化轴角 -> 原始轴角
            with torch.amp.autocast(device_type='cuda', enabled=False):
                pred_hands_aa_canonical_f32 = pred_hands_aa_canonical.to(torch.float32)
                R_canonical_pred = axis_angle_to_matrix(pred_hands_aa_canonical_f32.reshape(-1, 3)).view(B, T, -1, 3, 3)
                R_corr = self.R_corr_buf.to(x.device, dtype=torch.float32)
                R_smplx_pred = torch.matmul(R_corr.unsqueeze(0).unsqueeze(0), R_canonical_pred)
                aa_smplx_pred_f32 = matrix_to_axis_angle(
                    R_smplx_pred.reshape(-1, 3, 3)
                ).view(B, T, -1, 3)

            aa_smplx_pred = aa_smplx_pred_f32.to(x.dtype)
            
            # 4. 最终组合
            final_poses = torch.zeros(B, T, self.joints_num, 3, device=x.device, dtype=x.dtype)
            final_poses[:, :, self.opt.non_hand_joint_indices, :] = pred_non_hands_aa.to(final_poses.dtype)
            final_poses[:, :, self.opt.hand_joint_indices, :] = aa_smplx_pred.to(final_poses.dtype)
            
            motion = final_poses.view(B, T, -1)
        # ==================== 【代码修改区-结束】 ====================

        elif self.feature_type == 'motion_dataset_rod3_fixed_length':
            # 这个分支的逻辑保持不变
            x_flat = x.reshape(B, T, J * D)
            x_permuted = x_flat.permute(0, 2, 1)
            out = self.layer1(x_permuted)
            out = self.activation(out)
            out = self.layer2(out)
            motion = out.permute(0, 2, 1)
            
        return motion

class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        # topology
        self.edge_list = [adj_list_to_edges(opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST)]
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):
            block = nn.ModuleList()
            for _ in range(opt.n_extra_layers):
                block.append(ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            block.append(ResSTConv(
                self.edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))
            pool = STPool(opt.dataset_name, i)
            block.append(pool)
            self.layers.append(block)
            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)
    def forward(self, x):
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

class STConvDecoder(nn.Module):
    def __init__(self, opt, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        # network modules
        self.layers = nn.ModuleList()

        # build network
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(opt.n_layers):
            block = nn.ModuleList()
            # 1. 先unpool
            block.append(STUnpool(skeleton_mapping=mapping_list.pop()))
            # 2. unpool后J变大，pop掉edge_list，edge_list[-1]就是unpool后J的邻接
            edge_list.pop()
            # 3. conv
            for _ in range(opt.n_extra_layers):
                block.append(ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            block.append(ResSTConv(
                edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))
            self.layers.append(block)

    def forward(self, x):
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


# ==================== 【新增代码: 矢量量化器】 ====================
class VectorQuantizer(nn.Module):
    """
    标准 Vector Quantizer 模块。
    用于 Hybrid VAE 的辅助分支，学习离散码本。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        # 码本 (Codebook)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # 用于统计 usage，方便 reset
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        
    def forward(self, inputs):
        # inputs: [B, T, J, D]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Quantize
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Metrics
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Update usage statistics (for external reset logic)
        if self.training:
            with torch.no_grad():
                # Simple moving average or just accumulation
                current_usage = encodings.sum(0)
                self.code_usage = self.code_usage * 0.99 + current_usage * 0.01

        return loss, quantized, perplexity, encoding_indices.view(input_shape[:-1])

    def reset_codebook(self, inputs, threshold=0.01):
        """
        Reset dead codes to random inputs.
        inputs: [B*T*J, D] batch of current latent vectors
        """
        with torch.no_grad():
            # Find dead codes
            dead_codes = torch.where(self.code_usage < threshold)[0]
            if len(dead_codes) == 0:
                return 0
            
            # Select random inputs to replace them
            # Flatten inputs first
            flat_inputs = inputs.view(-1, self._embedding_dim)
            indices = torch.randperm(flat_inputs.shape[0])[:len(dead_codes)]
            selected_inputs = flat_inputs[indices]
            
            # Assign
            self._embedding.weight.data[dead_codes] = selected_inputs
            
            # Reset usage
            self.code_usage[dead_codes] = 1.0 # Prevent immediate re-reset
            
            return len(dead_codes)