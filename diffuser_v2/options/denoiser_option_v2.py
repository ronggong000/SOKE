import argparse
import os
import sys
import torch
from os.path import join as pjoin
from utils import paramUtil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFUSER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SOKE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SOKE_ROOT, ".."))
_DEFAULT_DATA_ROOT = os.path.join(_WORKSPACE_ROOT, "how2sign")
_DEFAULT_CHECKPOINTS_DIR = os.path.join(_SOKE_ROOT, "checkpoints")
_DEFAULT_SMPLX_MODEL_PATH = os.path.join(_SOKE_ROOT, "deps", "smpl_models")
_DEFAULT_VAE_DIR = os.path.join(_SOKE_ROOT, "checkpoints", "vae", "qvae_b256h1024_L1_fingerdistinct")

if _SOKE_ROOT not in sys.path:
    sys.path.append(_SOKE_ROOT)

from mGPT.utils.joints_list import SMPLX_JOINT_LANDMARK_NAMES, SELECTED_JOINT_INDICES,SELECTED_JOINT_LANDMARK_INDICES,SELECTED_JOINT_LANDMARK_BODY_EVAL,SELECTED_JOINT_LANDMARK_LHAND_EVAL,SELECTED_JOINT_LANDMARK_RHAND_EVAL, SELECTED_JOINT_LANDMARK_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_LANDMARK_INDICES_LANDMARK_INDEX,SELECTED_JOINT_INDICES_BODY_ONLY,SELECTED_JOINT_INDICES_NEIGHBOR_LIST,SELECTED_JOINT_INDICES_HAND_ONLY
from mGPT.utils.smplx_vertex_group import LEFT_HAND_VERTEX,RIGHT_HAND_VERTEX,UPPER_BODY_VERTEX


def _require_cuda_device(gpu_id: int) -> torch.device:
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        raise RuntimeError(
            "Diffusion training requires CUDA, but torch.cuda.is_available() is False. "
            f"torch={torch.__version__} built_cuda={torch.version.cuda} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    torch.cuda.set_device(int(gpu_id))
    return torch.device("cuda:" + str(int(gpu_id)))


def _resolve_dist_env():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return local_rank, rank, world_size


def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vae_path", type=str, default="", help="Absolute path to the pre-trained VAE folder")
    ## basic setup
    parser.add_argument("--name", type=str, default="denoiser_default", help="Name of this trial")
    parser.add_argument("--vae_name", type=str, default="vae_default", help="Name of the vae model.")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    ## dataloader
    parser.add_argument("--dataset_name", type=str, default="HIERARCHICAL", help="dataset directory", choices=['SMPLX_SL','HAND_CENTRIC','HIERARCHICAL'])
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--max_motion_length", type=int, default=2048, help="Max length of motion")
    parser.add_argument("--unit_length", type=int, default=4, help="Downscale ratio of VAE")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--train_data_dir", type=str, default="", help="Directory containing train npz files")
    parser.add_argument("--train_csv_path", type=str, default="", help="CSV/TSV for train split labels")
    parser.add_argument("--val_data_dir", type=str, default="", help="Directory containing val npz files")
    parser.add_argument("--val_csv_path", type=str, default="", help="CSV/TSV for val split labels")
    parser.add_argument("--train_only_gloss", dest="train_only_gloss", action="store_true", help="Use gloss-only conditioning text pair ['', gloss]")
    parser.add_argument("--train_with_sentence", dest="train_only_gloss", action="store_false", help="Use sentence+gloss conditioning text pair [sentence, gloss]")
    parser.set_defaults(train_only_gloss=True)
    parser.add_argument("--xyz", type=bool, default=False)
    ## optimization
    parser.add_argument("--max_epoch", default=1000, type=int, help="number of total epochs to run")
    parser.add_argument("--warm_up_iter", default=50, type=int, help="number of total iterations for warmup")
    parser.add_argument("--lr", default=1e-5, type=float, help="max learning rate")
    parser.add_argument("--milestones", default=[100, 200, 300], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument("--gamma", default=0.7, type=float, help="learning rate decay")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--recon_loss", type=str, default="l1", help="reconstruction loss", choices=["l1", "l1_smooth", "l2"])
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--dist_loss_weight", default=0.1, type=float)
    parser.add_argument("--finger_loss_weight", default=10.0, type=float)
    parser.add_argument("--amp_dtype", type=str, default="none", choices=["none", "bf16", "fp16"], help="Mixed precision dtype for denoiser training")
    #parser.add_argument("--mesh_loss", default=20, type=int)
    ## denosier arch
    parser.add_argument("--clip_version", type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"], help="Legacy field; V3 does not use CLIP")
    parser.add_argument("--latent_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--n_layers", type=int, default=5, help="num of layers")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--ff_dim", type=int, default=1024, help="feedforward dimension")
    parser.add_argument("--norm", type=str, default="layer", help="normalization", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", help="activation function", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
    parser.add_argument("--cond_drop_prob", type=float, default=0.1, help="Dropout ratio of condition for classifier-free guidance")
    parser.add_argument("--cond_scale", type=float, default=7.5, help="classifier-free guidance scale factor for condition")
    parser.add_argument("--mismatch_text_weight", type=float, default=0.1, help="weight of negative text")
    parser.add_argument("--mismatch_text_margin", type=float, default=0.05, help="margin of negative text")
    parser.add_argument("--t_high_noise_pow", type=float, default=2.0, help="power for t high noise sampling")  # op

    
    # rag
    parser.add_argument("--use_rag", dest="use_rag", action="store_true", help="Enable RAG blueprint conditioning")
    parser.add_argument("--disable_rag", dest="use_rag", action="store_false", help="Disable RAG blueprint conditioning")
    parser.set_defaults(use_rag=True)
    parser.add_argument("--rag_metadata_path", type=str, default=os.path.join(_DEFAULT_VAE_DIR, "dataset_metadata.json"))
    parser.add_argument("--rag_wmap_path", type=str, default=os.path.join(_DEFAULT_VAE_DIR, "aslcitizen_qvae_tokens.json"))
    parser.add_argument("--rag_slot_names", type=str, default="", help="Comma-separated slot names from rag metadata, e.g. left_hand,right_hand")
    parser.add_argument("--rag_frame_subsample", type=int, default=0, help="<=0 keeps old midpoint-only RAG; >0 keeps every N-th token frame per gloss word")
    parser.add_argument("--rag_gloss_csv_dir", type=str, default="", help="Optional csv dir/file for remapping Video file -> my_gloss when loading RAG tokens")
    parser.add_argument("--rag_gloss_source_col", type=str, default="Video file")
    parser.add_argument("--rag_gloss_target_col", type=str, default="my_gloss")
    parser.add_argument("--rag_weight_dir", type=str, default="", help="Optional sidecar npz root for RAG token-frame weights")
    parser.add_argument("--rag_weight_key", type=str, default="soft_w", help="Key inside RAG sidecar npz, e.g. soft_w")
    parser.add_argument("--rag_weight_max_mix", type=float, default=0.5, help="Blend ratio for slot weight aggregation: (1-mix)*mean + mix*max")
    parser.add_argument("--rag_weight_gate_scale", type=float, default=1.0, help="Gate scale on RAG slot embeddings. 1.0 means gate ~= 2*w")
    parser.add_argument("--rag_layers", type=int, default=0)
    parser.add_argument("--rag_heads", type=int, default=8)
    parser.add_argument("--rag_K", type=int, default=13)
    parser.add_argument("--rag_max_T", type=int, default=384)
    parser.add_argument("--rag_max_words", type=int, default=64)
    parser.add_argument("--rag_per_word_max_T", type=int, default=1, help="When rag_frame_subsample>0, values >1 cap sampled token rows per gloss word; 1 means keep the full sampled sequence")
    parser.add_argument("--rag_total_max_T", type=int, default=384)

    parser.add_argument("--gloss_layers", type=int, default=0)
    parser.add_argument("--gloss_heads", type=int, default=8)
    parser.add_argument("--gloss_vocab_path", type=str, default="", help="Required by V3 gloss vocab mode")
    parser.add_argument("--gloss_vocab_size", type=int, default=0, help="Optional override; <=0 means infer from vocab")
    parser.add_argument("--gloss_pad_id", type=int, default=0)
    parser.add_argument("--gloss_unk_id", type=int, default=1)
    parser.add_argument("--gloss_bos_id", type=int, default=2)
    parser.add_argument("--gloss_eos_id", type=int, default=3)
    parser.add_argument("--gloss_use_positional", dest="gloss_use_positional", action="store_true", help="Add positional embeddings on gloss token sequence")
    parser.add_argument("--gloss_disable_positional", dest="gloss_use_positional", action="store_false", help="Disable positional embeddings on gloss token sequence")
    parser.set_defaults(gloss_use_positional=False)
    parser.add_argument("--gloss_max_tokens", type=int, default=512, help="Max gloss token length for positional embedding")
    parser.add_argument("--enable_length_cond", action="store_true", help="Inject latent sequence length as explicit condition")
    parser.add_argument("--length_cond_max_len", type=int, default=1024, help="Max latent length index for length embedding")
    parser.add_argument("--length_cond_as_token", dest="length_cond_as_token", action="store_true", help="Append a length token into cross-attention memory")
    parser.add_argument("--length_cond_no_token", dest="length_cond_as_token", action="store_false", help="Do not append a length token into cross-attention memory")
    parser.set_defaults(length_cond_as_token=True)

    # parser.add_argument("--additive_attn", action="store_true", help="Use additive attention of skeletal and temporal dimensions")
    # parser.add_argument("--skel_attn_first", action="store_true", help="Use skeletal attention first")
    # parser.add_argument("--flat_attn", action="store_true", help="Use flat attention for skeletal and temporal dimensions")
    # parser.add_argument("--no_cross_attn", action="store_true", help="Use cross attention for skeletal and temporal dimensions")
    # parser.add_argument("--no_film", action="store_true", help="Not using FiLM for conditioning and use element-wise addition instead")

    ## diffusion scheduler
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps")
    parser.add_argument("--num_inference_timesteps", type=int, default=50, help="Number of inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.00085, help="Beta start")
    parser.add_argument("--beta_end", type=float, default=0.012, help="Beta end")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear", help="Beta schedule", choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    parser.add_argument("--prediction_type", type=str, default="v_prediction", help="Prediction type", choices=["epsilon", "sample", "v_prediction"])

    ## log
    parser.add_argument("--is_continue", action="store_true", help="Name of this trial")
    parser.add_argument("--checkpoints_dir", type=str, default=_DEFAULT_CHECKPOINTS_DIR, help="models are saved here")
    parser.add_argument("--log_every", default=5, type=int, help="iter log frequency")
    parser.add_argument("--save_latest", default=500, type=int, help="iter save latest model frequency")
    parser.add_argument("--eval_every_e", default=25, type=int, help="save eval results every n epoch")
    parser.add_argument("--smplx_model_path", type=str, default=_DEFAULT_SMPLX_MODEL_PATH, help="SMPL/SMPLH/SMPLX model directory")

    # custom per-frame sidecar weights (e.g. gradcam soft_w)
    parser.add_argument("--enable_custom_weight", action="store_true", help="Enable sidecar per-frame loss weighting")
    parser.add_argument("--custom_weight_dir", type=str, default="", help="Root dir of sidecar weight npz files")
    parser.add_argument("--custom_weight_key", type=str, default="soft_w", help="Key inside sidecar npz (e.g. soft_w)")
    parser.add_argument("--custom_weight_precheck", action="store_true", help="Fail early if any sample has no sidecar weight file")
    parser.add_argument("--use_latent_cache", action="store_true", help="Use precomputed latent cache for train split")
    parser.add_argument("--build_latent_cache", action="store_true", help="Build latent cache before training")
    parser.add_argument("--rebuild_latent_cache", action="store_true", help="Force rebuild latent cache files")
    parser.add_argument("--latent_cache_dir", type=str, default="", help="Directory containing split latent cache .pt files")
    parser.add_argument("--latent_cache_dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Storage dtype for latent cache")
    parser.add_argument("--latent_cache_batch_size", type=int, default=16, help="Batch size used when building latent cache")
    parser.add_argument("--latent_cache_workers", type=int, default=4, help="Workers used when building latent cache")
    parser.add_argument("--latent_cache_build_all_splits", dest="latent_cache_build_all_splits", action="store_true", help="Build train/val/test cache files")
    parser.add_argument("--latent_cache_build_train_only", dest="latent_cache_build_all_splits", action="store_false", help="Build only train cache file")
    parser.set_defaults(latent_cache_build_all_splits=True)

    # full ASLcitizen pretrain helper mode
    parser.add_argument("--full_aslcitizen_mode", action="store_true", help="Train on merged ASLcitizen train+val+test split")
    parser.add_argument("--full_train_csv_paths", nargs="+", default=[], help="CSV paths to merge as full train set")

    # tiny smoke-run mode
    parser.add_argument("--tiny_debug", action="store_true", help="Enable tiny smoke training mode")
    parser.add_argument("--tiny_train_batches", type=int, default=2, help="Number of train batches per epoch in tiny mode")
    parser.add_argument("--tiny_val_batches", type=int, default=1, help="Number of val batches per eval in tiny mode")
    parser.add_argument("--tiny_disable_wandb", dest="tiny_disable_wandb", action="store_true", help="Disable wandb in tiny mode")
    parser.add_argument("--tiny_enable_wandb", dest="tiny_disable_wandb", action="store_false", help="Enable wandb in tiny mode")
    parser.set_defaults(tiny_disable_wandb=True)

    opt = parser.parse_args()
    opt.classifier_free_guidance = opt.cond_scale > 1.0
    opt.local_rank, opt.rank, opt.world_size = _resolve_dist_env()
    opt.distributed = int(opt.world_size) > 1
    opt.is_master = int(opt.rank) == 0

    target_gpu = int(opt.local_rank) if opt.distributed else int(opt.gpu_id)
    if target_gpu < 0:
        raise RuntimeError("Diffusion training requires a CUDA GPU. --gpu_id must be >= 0.")
    opt.device = _require_cuda_device(target_gpu)
    opt.device_index = target_gpu

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin(_DIFFUSER_ROOT, 'log', opt.dataset_name, opt.name)

    #opt.use_precomputed_text_emb = True
    #opt.text_emb_dir = "/home/smuk0019/ar85_scratch2/singyu/unet/translator/checkpoints/maskgit_v1/gloss_embeddings/train"  # 训练集
    #opt.text_emb_dim = 1024
    #opt.text_emb_preload = True  
    #opt.text_emb_preload_limit_gb = 0.0 
    #opt.text_emb_preload_fp16=True
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # V3 hard defaults: gloss vocab conditioning + rag enabled
    opt.use_gloss_tokens = True
    opt.gloss_embed_mode = "vocab"
    opt.use_cond_film = True
    opt.use_rag = bool(getattr(opt, "use_rag", True))
    # if opt.dataset_name == "HIERARCHICAL":
    #     opt.data_root = './dataset/humanml3d/'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 22
    #     opt.pose_dim = 263
    #     opt.contact_joints = [7, 10, 8, 11]
    #     opt.fps = 20
    #     opt.radius = 4
    #     opt.kinematic_chain = paramUtil.t2m_kinematic_chain
    #     opt.dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    if opt.dataset_name == 'HIERARCHICAL':
        opt.data_root = _DEFAULT_DATA_ROOT
        opt.motion_dir = pjoin(opt.data_root, 'align_denoised_front_joints_relative')
        opt.text_dir = pjoin(opt.data_root, 'how2sign_realigned_train.csv')
        opt.joints_num = 43
        opt.pose_dim = 129
        opt.contact_joints = []
        opt.fps = 24
        opt.max_motion_length = 2048
        opt.max_motion_frame = 2048
        #opt.max_motion_token = 55
    else:
        raise KeyError('Dataset Does not Exists')

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
    opt.finger_joint_indices = SELECTED_JOINT_INDICES_HAND_ONLY
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.train_data_dir = str(getattr(opt, "train_data_dir", "") or "").strip() or pjoin(opt.data_root, "align_denoised_front")
    opt.train_csv_path = str(getattr(opt, "train_csv_path", "") or "").strip() or pjoin(opt.data_root, "merge_train.csv")
    opt.val_data_dir = str(getattr(opt, "val_data_dir", "") or "").strip() or pjoin(opt.data_root, "align_denoised_front_val")
    opt.val_csv_path = str(getattr(opt, "val_csv_path", "") or "").strip() or pjoin(opt.data_root, "merge_val.csv")

    # Auto-discover gloss vocab path if user did not provide it.
    if not str(getattr(opt, "gloss_vocab_path", "")).strip():
        vocab_candidates = [
            pjoin(opt.data_root, "gloss_vocab.json"),
            pjoin(opt.data_root, "gloss_vocab_v3.json"),
            pjoin(opt.data_root, "vocab_gloss.json"),
            pjoin(opt.data_root, "vocab", "gloss_vocab.json"),
            pjoin(".", "gloss_vocab.json"),
        ]
        for cand in vocab_candidates:
            if os.path.isfile(cand):
                opt.gloss_vocab_path = cand
                break

    args = vars(opt)

    opt.is_train = is_train
    if is_train and opt.is_master:
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
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
            
    return opt
