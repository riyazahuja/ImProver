#!/bin/bash
#SBATCH --job-name=improver_modeltheory
#SBATCH --output=/home/shivansg/ImProver/logs/improver_modeltheory_%j.out
#SBATCH --error=/home/shivansg/ImProver/logs/improver_modeltheory_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/shivansg/ImProver

echo "============================================================"
echo "Generating ImProver prompts for ModelTheory"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Run the prompts generation
./improver prompts get \
    --dataset_path /home/shivansg/ImProver/data/modeltheory_dataset.json \
    --prompts_id modeltheory_prompts \
    --cpus 8

echo "============================================================"
echo "Counting generated prompts..."
find /home/shivansg/ImProver/prompts/modeltheory_prompts -name "*.json" | wc -l
echo "Job Complete"
echo "============================================================"
