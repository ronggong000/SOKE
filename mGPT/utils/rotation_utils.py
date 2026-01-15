import torch
import torch.nn as nn
import math
# --- PyTorch 旋转转换辅助函数 ---
# def axis_angle_to_matrix(a):
#     angle = torch.norm(a, dim=-1, keepdim=True)
#     axis = a / (angle + 1e-8)
#     cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
#     K = torch.zeros((a.shape[0], 3, 3), device=a.device, dtype=a.dtype)
#     K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
#     K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
#     K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
#     I = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[0], -1, -1)
#     return I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * torch.matmul(K, K)

def axis_angle_to_matrix(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    具备【数值稳定 + 梯度稳定】的轴角转旋转矩阵。
    关键点：
    1) 分母 theta_sq 绝不为 0（clamp）
    2) 小角度用泰勒展开，避免 0/0 与不稳定梯度
    3) 核心计算用 float32，最后 cast 回原 dtype
    """
    orig_dtype = v.dtype
    v32 = v.to(torch.float32)

    batch_shape = v32.shape[:-1]

    theta_sq = torch.sum(v32 * v32, dim=-1, keepdim=True)                # [..., 1]
    theta_sq_safe = torch.clamp(theta_sq, min=eps)                       # 防止除 0
    theta = torch.sqrt(theta_sq_safe)                                    # [..., 1]

    # A = sin(theta)/theta
    # B = (1-cos(theta))/theta^2
    A = torch.sin(theta) / theta                                         # [..., 1]
    B = (1.0 - torch.cos(theta)) / theta_sq_safe                         # [..., 1]

    # 小角度泰勒展开（用 theta_sq 避免额外 sqrt 带来的奇异导数问题）
    small = theta_sq < 1e-6
    theta4 = theta_sq * theta_sq
    A_taylor = 1.0 - theta_sq / 6.0 + theta4 / 120.0
    B_taylor = 0.5 - theta_sq / 24.0 + theta4 / 720.0

    A = torch.where(small, A_taylor, A)
    B = torch.where(small, B_taylor, B)

    # 构造反对称矩阵 K
    wx, wy, wz = v32.unbind(dim=-1)
    O = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([O,  -wz,  wy], dim=-1),
        torch.stack([wz,  O,  -wx], dim=-1),
        torch.stack([-wy, wx,  O], dim=-1),
    ], dim=-2)                                                           # [..., 3, 3]
    K2 = K @ K

    I = torch.eye(3, device=v32.device, dtype=v32.dtype).view(*(1,) * len(batch_shape), 3, 3)
    R = I + A[..., None] * K + B[..., None] * K2

    return R.to(orig_dtype)
def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_euler_xyz(matrix):
    sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = sy < 1e-6
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0
    x[singular] = xs[singular]
    y[singular] = ys[singular]
    z[singular] = zs[singular]
    return torch.stack([x, y, z], dim=-1)

def euler_xyz_to_matrix(eulers):
    # ... (实现 euler -> matrix 的转换, zxy的反向)
    # 这是一个简化的 XYZ 顺序实现
    x, y, z = eulers[..., 0], eulers[..., 1], eulers[..., 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    
    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    rx = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1).view(eulers.shape[:-1] + (3, 3))
    ry = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1).view(eulers.shape[:-1] + (3, 3))
    rz = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1).view(eulers.shape[:-1] + (3, 3))
    
    return rz @ ry @ rx # XYZ aplication order
def matrix_to_euler_xyz_safe(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    稳定的 R -> 欧拉(XYZ)
    R: [..., 3, 3]  (建议 float32)
    返回: [..., 3]   (rx, ry, rz)
    """
    r00 = R[..., 0, 0]; r01 = R[..., 0, 1]; r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]; r11 = R[..., 1, 1]; r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]; r21 = R[..., 2, 1]; r22 = R[..., 2, 2]

    # ry = asin(-r02)，先 clamp 防越界
    ry = torch.asin(torch.clamp(-r02, min=-1.0 + 1e-6, max=1.0 - 1e-6))
    cy = torch.cos(ry)
    near_gimbal = (cy.abs() < 1e-6)

    # 一般分支
    rx_general = torch.atan2(r12, r22)
    rz_general = torch.atan2(r01, r00)

    # 万向节分支（cy ≈ 0）
    rx_gimbal = torch.atan2(-r21, r11)
    rz_gimbal = torch.zeros_like(rz_general)

    rx = torch.where(near_gimbal, rx_gimbal, rx_general)
    rz = torch.where(near_gimbal, rz_gimbal, rz_general)

    # wrap 到 [-pi, pi]
    pi = 3.141592653589793
    def wrap_pi(a: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a + pi, 2*pi) - pi

    rx = wrap_pi(rx); ry = wrap_pi(ry); rz = wrap_pi(rz)
    euler = torch.stack([rx, ry, rz], dim=-1)
    euler = torch.nan_to_num(euler, nan=0.0, posinf=1e6, neginf=-1e6)
    return euler

# def matrix_to_axis_angle(R):
#     """
#     更稳定的旋转矩阵 -> 轴角转换函数。
#     通过在 clamp 时保留 epsilon 来避免 acos 的梯度 NaN 问题。
#     """
#     # 记录原始类型并在内部使用 float32 计算
#     original_dtype = R.dtype
#     R32 = R.to(torch.float32)

#     # --- 【核心修正】在 clamp 时保留一个微小的 epsilon ---
#     trace = torch.einsum('bii->b', R32)
#     val_to_acos = (trace - 1) / 2
#     angle = torch.acos(torch.clamp(val_to_acos, -1.0 + 1e-6, 1.0 - 1e-6))
    
#     axis = torch.stack([
#         R32[:, 2, 1] - R32[:, 1, 2],
#         R32[:, 0, 2] - R32[:, 2, 0],
#         R32[:, 1, 0] - R32[:, 0, 1]
#     ], dim=-1)
    
#     sin_angle = torch.sin(angle)
#     # 增加 epsilon 防止除以零
#     axis = axis / (2 * sin_angle.unsqueeze(-1) + 1e-8)
    
#     # 最终结果转回原始类型
#     return (axis * angle.unsqueeze(-1)).to(original_dtype)

def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    稳定版矩阵转轴角：用 atan2 避免 acos 在 cos=1 附近的导数奇异。
    额外做了：
    - float32 核心计算
    - sin_theta 分母 clamp
    """
    orig_dtype = R.dtype
    R32 = R.to(torch.float32)

    v_vec = torch.stack([
        R32[..., 2, 1] - R32[..., 1, 2],
        R32[..., 0, 2] - R32[..., 2, 0],
        R32[..., 1, 0] - R32[..., 0, 1]
    ], dim=-1)

    sin_theta = 0.5 * torch.linalg.norm(v_vec, dim=-1, keepdim=True)     # [..., 1]
    cos_theta = 0.5 * (torch.einsum('...ii->...', R32)[..., None] - 1.0) # [..., 1]

    theta = torch.atan2(sin_theta, cos_theta)                             # [..., 1]

    # theta/(2*sin_theta) 的稳定形式
    sin_theta_safe = torch.clamp(sin_theta, min=eps)
    s = theta / (2.0 * sin_theta_safe)

    # 小角度泰勒：theta/(2*sin_theta) ~ 0.5 + theta^2/12
    small = theta < 1e-4
    s_taylor = 0.5 + (theta * theta) / 12.0
    s = torch.where(small, s_taylor, s)

    aa = v_vec * s
    return aa.to(orig_dtype)
# --- 核心模型：可训练的修正变换 ---
class CorrectionTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_hand_correction_6d = nn.Parameter(torch.randn(15, 6))
        reflect = torch.eye(3)
        reflect[0, 0] = -1
        self.register_buffer('reflection_matrix', reflect)

    def forward(self):
        R_corr_right = rotation_6d_to_matrix(self.right_hand_correction_6d)
        R_corr_left = self.reflection_matrix @ R_corr_right @ self.reflection_matrix
        return torch.cat([R_corr_left, R_corr_right], dim=0)