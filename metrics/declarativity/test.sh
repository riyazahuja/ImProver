#!/bin/bash

#SBATCH --job-name=rerun_decl
#SBATCH --output=logs/final/rerun_decl.out
#SBATCH --error=logs/final/rerun_decl.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G





source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp


cd /home/$USER/eval_improver/improver
lake build eval_improver
sleep 0.5


./improver run eval --run_id baseline_decl --cpus 64
./improver run analysis --run_id baseline_decl
