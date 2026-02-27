import torch
from metrics.mr import MRMetrics
from metrics.t2m import TM2TMetrics
import os
import smplx
def lengths_to_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    # 增加 max_len 参数，确保生成的 Mask 宽度与 Padding 后的张量一致
    max_frames = max_len if max_len is not None else torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask

class SignPhysicalEvaluator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.smplx_model_path = str(getattr(opt, "smplx_model_path", "") or "")
        if not self.smplx_model_path:
            self.smplx_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "deps", "smpl_models")
            )
        from metrics.mr import MRMetrics
        from metrics.t2m import TM2TMetrics
        self.mr_metrics = MRMetrics(njoints=opt.joints_num, dist_sync_on_step=True)
        self.dtw_metrics = TM2TMetrics(cfg=None, dataname='how2sign', dist_sync_on_step=True)
        self.body_indices = torch.tensor(self.opt.SELECTED_JOINT_INDICES_BODY_ONLY, device=self.device)

    def _get_joints_and_vertices(self, rotations, smplx_model):
        """
        rotations: [N_valid, 43, 3] or [N_valid, 129]
        """
        # --- 核心修复 1: 强制确保输入是 3D 关节格式 [N, 43, 3] ---
        if rotations.dim() == 2:
            rotations = rotations.reshape(-1, 43, 3)
    
        curr_batch = rotations.shape[0]
        if smplx_model.batch_size != curr_batch:
            smplx_model = smplx.create(
                model_path=self.smplx_model_path, 
                model_type='smplx', 
                gender='neutral', 
                use_pca=False,
                flat_hand_mean=True, 
                batch_size=curr_batch
            ).to(self.device).eval()

        # 现在 rotations 是 [N, 43, 3]，切片会得到 [N, 13, 3]
        body = rotations[:, :13]
        lhand = rotations[:, 13:28]
        rhand = rotations[:, 28:43]
        
        restored_body = torch.zeros(rotations.shape[0], 22, 3, device=self.device)
        restored_body[:, self.body_indices] = body
        
        output = smplx_model(
            body_pose=restored_body[:, 1:], 
            left_hand_pose=lhand, 
            right_hand_pose=rhand,
            return_verts=True
        )
        return output.joints, output.vertices, smplx_model
    def update(self, pred_rot, gt_rot, lengths, smplx_model, compute_dtw=False, src="how2sign", names=None):
        B, T = pred_rot.shape[:2]
        device = pred_rot.device
        # --- 统一形状修复：强制转为 [B, T, J, 3] ---
        if pred_rot.ndim == 3:
            pred_rot = pred_rot.reshape(B, T, -1, 3)
        if gt_rot.ndim == 3:
            gt_rot = gt_rot.reshape(B, T, -1, 3)
       
        # --- XYZ 模式跳过 SMPL-X 逻辑 ---
        if getattr(self.opt, 'xyz', False):
            # 此时 pred_rot/gt_rot 形状应为 [B, T, 43, 3]
            with torch.no_grad():
                for i in range(B):
                    cur_len = lengths[i]
                    p_joint = pred_rot[i, :cur_len] # [L, 43, 3]
                    g_joint = gt_rot[i, :cur_len]   # [L, 43, 3]

                    # --- 身体部分 (0-13 关节) ---
                    body_p = p_joint[:, :13, :] # [L, 13, 3]
                    body_g = g_joint[:, :13, :] # [L, 13, 3]

                    # Pelvis 对齐：减去每帧第 0 个关节的坐标
                    # .mean(dim=1, keepdim=True) 会导致维度变成 [L, 1, 3]，从而实现广播
                    pelvis_p = body_p[:, :1, :] # [L, 1, 3]
                    pelvis_g = body_g[:, :1, :] # [L, 1, 3]
                    
                    body_p_aligned = body_p - pelvis_p
                    body_g_aligned = body_g - pelvis_g
                    
                    # 计算 MPJPE：先算关节距离，再算 13 个关节的均值，最后累加所有帧
                    dist_body = torch.linalg.norm(body_p_aligned - body_g_aligned, dim=-1).mean(dim=-1).sum()
                    
                    # --- 手部部分 (13-43 关节) ---
                    hand_p = p_joint[:, 13:, :] # [L, 30, 3]
                    hand_g = g_joint[:, 13:, :] # [L, 30, 3]
                    
                    # 手部通常按腕关节对齐，这里简化为直接计算 (或不进行额外对齐)
                    dist_hand = torch.linalg.norm(hand_p - hand_g, dim=-1).mean(dim=-1).sum()
                    #new_count = getattr(self.mr_metrics, f'{src}_count').to(device) + cur_len
                    # 获取当前状态并确保是 Tensor
                    prev_count = torch.as_tensor(getattr(self.mr_metrics, f'{src}_count'), device=device)
                    prev_body = torch.as_tensor(getattr(self.mr_metrics, f'{src}_MPJPE_body'), device=device)
                    prev_hand = torch.as_tensor(getattr(self.mr_metrics, f'{src}_MPJPE_hand'), device=device)

                    # 累加并重新设置
                    setattr(self.mr_metrics, f'{src}_count', prev_count + cur_len)
                    setattr(self.mr_metrics, f'{src}_MPJPE_body', prev_body + dist_body)
                    setattr(self.mr_metrics, f'{src}_MPJPE_hand', prev_hand + dist_hand)

            return smplx_model
        # 提取有效帧
         # --- 核心修复 2: 显式传入 T，解决 IndexError ---
        mask_3d = lengths_to_mask(torch.tensor(lengths, device=device), max_len=T)
        pred_valid = pred_rot[mask_3d] 
        gt_valid = gt_rot[mask_3d]

        with torch.no_grad():
            j_pd, v_pd, smplx_model = self._get_joints_and_vertices(pred_valid, smplx_model)
            j_gt, v_gt, smplx_model = self._get_joints_and_vertices(gt_valid, smplx_model)

        # 还原 4D
        j_pd_padded = torch.zeros(B, T, j_pd.shape[1], 3, device=device)
        v_pd_padded = torch.zeros(B, T, v_pd.shape[1], 3, device=device)
        j_gt_padded = torch.zeros(B, T, j_gt.shape[1], 3, device=device)
        v_gt_padded = torch.zeros(B, T, v_gt.shape[1], 3, device=device)

        j_pd_padded[mask_3d] = j_pd
        v_pd_padded[mask_3d] = v_pd
        j_gt_padded[mask_3d] = j_gt
        v_gt_padded[mask_3d] = v_gt

        # --- 核心修复 3: 展平前两维后再传给 mr_metrics，解决 RuntimeError ---
        # mr.py 内部会根据 lengths 和 B 重新切分，它期望输入是 [B*T, J, 3]
        self.mr_metrics.update(
            feats_rst=None, feats_ref=None,
            joints_rst=j_pd_padded.reshape(-1, j_pd_padded.shape[2], 3), 
            joints_ref=j_gt_padded.reshape(-1, j_gt_padded.shape[2], 3),
            vertices_rst=v_pd_padded.reshape(-1, v_pd_padded.shape[2], 3), 
            vertices_ref=v_gt_padded.reshape(-1, v_gt_padded.shape[2], 3),
            lengths=lengths, src=[src]*B, name=names
        )

        # TM2TMetrics 更新 (DTW) - 仅在测试时开启，因为非常慢
        if compute_dtw:
            self.dtw_metrics.update(
                feats_rst=None, feats_ref=None,
                joints_rst=j_pd_padded.reshape(-1, j_pd_padded.shape[2], 3), joints_ref=j_gt_padded.reshape(-1, j_gt_padded.shape[2], 3),
                vertices_rst=v_pd_padded.reshape(-1, v_pd_padded.shape[2], 3), vertices_ref=v_gt_padded.reshape(-1, v_gt_padded.shape[2], 3),
                lengths=lengths, lengths_rst=lengths, # 假设生成长度一致
                split='test', src=[src]*B, name=names
            )
        return smplx_model
