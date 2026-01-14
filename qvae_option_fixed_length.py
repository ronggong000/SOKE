import argparse
import os
import torch
from os.path import join as pjoin
#from utils import paramUtil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from joints_list import SMPLX_JOINT_LANDMARK_NAMES, SELECTED_JOINT_INDICES,SELECTED_JOINT_LANDMARK_INDICES,SELECTED_JOINT_LANDMARK_BODY_EVAL,SELECTED_JOINT_LANDMARK_LHAND_EVAL,SELECTED_JOINT_LANDMARK_RHAND_EVAL, SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,SELECTED_JOINT_INDICES_BODY_ONLY,SELECTED_JOINT_INDICES_NEIGHBOR_LIST
from smplx_vertex_group import LEFT_HAND_VERTEX,RIGHT_HAND_VERTEX,UPPER_BODY_VERTEX
def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## basic setup
    parser.add_argument("--name", type=str, default="vae_rod3_fixed_length_h2s", help="Name of this trial")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")

    ## dataloader
    parser.add_argument("--dataset_name", type=str, default="SMPLX_SL", help="dataset directory", choices=['SMPLX_SL','HAND_CENTRIC','HIERARCHICAL'])
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--max_length", type=int, default=256, help="training motion length")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument('--train_data_dir', type=str, default='../../how2sign/align_denoised_front')
    parser.add_argument('--val_data_dir', type=str, default='../../how2sign/align_denoised_front_val')
    parser.add_argument('--test_data_dir', type=str, default='../../how2sign/align_denoised_front_test')
    parser.add_argument('--data_format', type=str, 
                        default='motion_dataset_rod3_fixed_length', 
                        choices=['motion_dataset_rod3_fixed_length','motion_dataset_rod3_fixed_length_dk'],
                        help='Data representation format.')
    parser.add_argument("--transform_path", type=str, default="../tools/per_joint_transforms.pth", 
                        help="Path to the trained correction transform .pth file.")
    parser.add_argument("--reduce_dim_finger", type=bool, default=False)
    ## optimization
    parser.add_argument("--max_epoch", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="number of total epoch for warmup")
    parser.add_argument("--warm_up_iter", default=2000, type=int, help="number of total iterations for warmup")
    parser.add_argument("--lr", default=2e-4, type=float, help="max learning rate")
    parser.add_argument("--milestones", default=[150_000, 250_000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")

    parser.add_argument("--recon_loss", type=str, default="mse", help="reconstruction loss")
    parser.add_argument("--mesh_loss", type=str, default="l1_smooth", help="mesh loss")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="reconstruction loss weight")
    # parser.add_argument("--lambda_pos", type=float, default=0.5, help="position loss weight")
    # parser.add_argument("--lambda_vel", type=float, default=0.5, help="velocity loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.02, help="kl loss weight") # used when vae
    parser.add_argument("--finger_loss_weight", type=float, default=10.0, help="finger_loss_weight loss")
    # 在 optimization 部分添加
    parser.add_argument("--commitment_cost", type=float, default=1.0, help="commitment cost for VQ")
    parser.add_argument("--lambda_q_recon", type=float, default=1.0, help="weight for quantized reconstruction loss")
    parser.add_argument("--lambda_consistency", type=float, default=0.5, help="weight for latent consistency loss")
    parser.add_argument("--lambda_quant", type=float, default=1.0, help="quant loss weight") # used when vqvae
    
    ## vae arch
    parser.add_argument("--latent_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--n_layers", type=int, default=2, help="num of layers")
    parser.add_argument("--n_extra_layers", type=int, default=1, help="num of extra layers")
    parser.add_argument("--norm", type=str, default="none", help="normalization", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", help="activation function", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--codebook_size_body", type=int, default=256, help="codebooksize_body")
    parser.add_argument("--codebook_size_hand", type=int, default=1024, help="codebooksize_hand")

    parser.add_argument('--codebook_grouping', type=str, default='default', 
                    choices=['default', 'arm_mirror', 'thumb_sep', 'finger_distinct','full_book'],
                    help='Strategy for codebook sharing and splitting.')

    ## other
    parser.add_argument("--is_continue", action="store_true", help="Name of this trial")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--log_every", default=10, type=int, help="iter log frequency")
    parser.add_argument("--save_latest", default=500, type=int, help="iter save latest model frequency")
    parser.add_argument("--eval_every_e", default=10, type=int, help="save eval results every n epoch")

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    args = vars(opt)

    opt.fps = 30

    
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    opt.SMPLX_JOINT_LANDMARK_NAMES = SMPLX_JOINT_LANDMARK_NAMES
    opt.SELECTED_JOINT_INDICES = SELECTED_JOINT_INDICES
    opt.SELECTED_JOINT_LANDMARK_INDICES = SELECTED_JOINT_LANDMARK_INDICES
    opt.SELECTED_JOINT_LANDMARK_BODY_EVAL = SELECTED_JOINT_LANDMARK_BODY_EVAL
    opt.SELECTED_JOINT_LANDMARK_LHAND_EVAL=SELECTED_JOINT_LANDMARK_LHAND_EVAL
    opt.SELECTED_JOINT_LANDMARK_RHAND_EVAL=SELECTED_JOINT_LANDMARK_RHAND_EVAL
    opt.SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST
    opt.SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX=SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX
    opt.SELECTED_JOINT_INDICES_BODY_ONLY=SELECTED_JOINT_INDICES_BODY_ONLY
    opt.UPPER_BODY_VERTEX = UPPER_BODY_VERTEX
    opt.LEFT_HAND_VERTEX = LEFT_HAND_VERTEX
    opt.RIGHT_HAND_VERTEX = RIGHT_HAND_VERTEX
    opt.SELECTED_JOINT_INDICES_NEIGHBOR_LIST = SELECTED_JOINT_INDICES_NEIGHBOR_LIST
    opt.joints_landmark_num = len(SELECTED_JOINT_LANDMARK_INDICES)
    opt.joints_num = len(SELECTED_JOINT_INDICES)

        # 定义手部关节在43个关节列表中的索引
    opt.hand_joint_indices = list(range(13, 43)) # 30个手部关节
    opt.non_hand_joint_indices = list(range(13)) # 13个非手部关节
# --- 仅在 data_format == 'motion_dataset_rod3_fixed_length_dk' 时需要 ---
    if opt.data_format=="motion_dataset_rod3_fixed_length_dk":
    

        # 1. 定义关节分组 (与Dataset脚本中保持一致)
        # 假设有55个总关节，你选择了43个
        # 注意：这里的索引应该是你在 SELECTED_JOINT_INDICES 中使用的关节的原始索引
        opt.GROUP1_FINGERS = set(range(25, 55))
        opt.GROUP2_WRISTS = {20, 21}
        opt.GROUP3_ARMS_HEAD = {16, 17, 18, 19, 12, 15}
        opt.GROUP4_TORSO_COLLAR = {3, 6, 9, 13, 14}

        # 2. 计算每个关节的特征维度 (最关键的一步)
        # 这个计算逻辑必须严格遵循你的 Dataset 脚本
        # poses_slice (3) + rot_vel_slice (3) + rot_accel_slice (3) + relative_pos_slice (3) = 12
        opt.DIM_FINGERS = 12 
        # poses_slice (3) + rot_vel_slice (3) + relative_pos_slice (3) = 9
        opt.DIM_WRISTS = 9
        # poses_slice (3) + relative_pos_slice (3) = 6
        opt.DIM_ARMS_HEAD = 6
        # poses_slice (3) = 3
        opt.DIM_TORSO = 3
        # 3. 创建一个列表，按 SELECTED_JOINT_INDICES 的顺序记录每个关节的维度
        opt.joint_feature_dims = []
        for idx in SELECTED_JOINT_INDICES: 
            if idx in opt.GROUP1_FINGERS:
                opt.joint_feature_dims.append(opt.DIM_FINGERS)
            elif idx in opt.GROUP2_WRISTS:
                opt.joint_feature_dims.append(opt.DIM_WRISTS)
            elif idx in opt.GROUP3_ARMS_HEAD:
                opt.joint_feature_dims.append(opt.DIM_ARMS_HEAD)
            elif idx in opt.GROUP4_TORSO_COLLAR:
                opt.joint_feature_dims.append(opt.DIM_TORSO)
            else:
                raise e 

        # 验证一下
        assert len(opt.joint_feature_dims) == len(opt.SELECTED_JOINT_INDICES)
    elif opt.data_format=="motion_dataset_rod3_fixed_length" and opt.reduce_dim_finger==True:

        
        # 定义MCP和PIP/DIP关节 (在30个手部关节中的相对索引)
        opt.mcp_indices_in_hand = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        opt.pip_dip_indices_in_hand = [i for i in range(30) if i not in opt.mcp_indices_in_hand]

        # 动态计算每个关节的特征维度
        opt.joint_feature_dims = []
        for i in range(len(opt.SELECTED_JOINT_INDICES)): # 遍历43个关节
            if i in opt.non_hand_joint_indices:
                opt.joint_feature_dims.append(3) # 非手部关节保持3D轴角
            else:
                relative_hand_idx = opt.hand_joint_indices.index(i)
                if relative_hand_idx in opt.mcp_indices_in_hand:
                    opt.joint_feature_dims.append(2) # MCP关节降维到2D (Y, Z)
                else:
                    opt.joint_feature_dims.append(1) # PIP/DIP关节降维到1D (Z)
        
        # 总输入维度
        opt.input_dim = sum(opt.joint_feature_dims)


    return opt