from train_denoiser_v2_how2sign import main as run_base_main


if __name__ == "__main__":
    run_base_main(
        default_overrides={
            "vae_family": "vae12",
            "motion_repr": "dk12",
            "vae_path": "/fs04/scratch2/ar85/singyu/SOKE/checkpoints/HIERARCHICAL/vae12_hier_12d",
        }
    )
