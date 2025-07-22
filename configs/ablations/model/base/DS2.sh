#!/bin/bash

#SBATCH --job-name=ablation_model_base_DS2
#SBATCH --output=logs/ablation_model_base_DS2.out
#SBATCH --error=logs/ablation_model_base_DS2.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:0
#SBATCH --mem=150G
    

source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG            # verbose compile log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1   

cd ~/eval_improver/improver

lake build eval_improver
sleep 5


./improver run pipeline --config /home/riyaza/eval_improver/improver/configs/ablations/model/base/DS2.yaml