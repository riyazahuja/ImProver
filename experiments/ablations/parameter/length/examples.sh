#!/bin/bash

#SBATCH --job-name=pa_le
#SBATCH --output=logs/parameter_ablation/length/examples.out
#SBATCH --error=logs/parameter_ablation/length/examples.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G
    

source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG            # verbose compile log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1   

cd ~/eval_improver/improver

lake build eval_improver
sleep 5


export run_id="parameter_ablation_length_examples"
export annotation=false
export context=0
export rag=0
export examples=-1


./improver run pipeline --run_id $run_id --annotation $annotation --context $context --rag $rag --examples $examples --config /home/riyaza/eval_improver/improver/experiments/ablations/parameter/length/base.yaml