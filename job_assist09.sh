# !/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=7 python run.py --log_path=logs/assist09_tune.log --config_path=config.yaml
