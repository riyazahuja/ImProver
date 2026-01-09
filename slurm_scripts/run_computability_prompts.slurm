#!/bin/bash
#SBATCH --job-name=improver_computability
#SBATCH --output=/home/shivansg/ImProver/logs/improver_computability_%j.out
#SBATCH --error=/home/shivansg/ImProver/logs/improver_computability_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/shivansg/ImProver

echo "============================================================"
echo "Generating ImProver prompts for Computability"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Run the prompts generation
./improver prompts get \
    --dataset_path /home/shivansg/ImProver/data/computability_dataset.json \
    --prompts_id computability_prompts \
    --cpus 8

echo "============================================================"
echo "Counting generated prompts..."
find /home/shivansg/ImProver/prompts/computability_prompts -name "*.json" | wc -l
echo "Job Complete"
echo "============================================================"
