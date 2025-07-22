source $HOME/miniconda3/bin/activate env

export HF_HOME="/data/user_data/riyaza/HF"

python /home/riyaza/eval_improver/improver/experiments/expert_iteration.py \
    --inf-template /home/riyaza/eval_improver/improver/experiments/results/base_infer.yaml \
    --train-template /home/riyaza/eval_improver/improver/experiments/results/base_train.yaml \
    --base-model deepseek-ai/DeepSeek-Prover-V2-7B \
    --base-name length_full \
    --iterations 5 \
    --cpus 16 \
    --gres gpu:A6000:8 \
    --mem 150G \
    --output-dir /data/user_data/riyaza/saved_models/length \
