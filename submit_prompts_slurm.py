#!/usr/bin/env python3
"""
Submit Slurm jobs to generate ImProver prompts for all missing topics in parallel.
"""
import json
import subprocess
from pathlib import Path

MATHLIB_ROOT = Path("/data/user_data/shivansg/mathlib4_v4.17.0/Mathlib")
IMPROVER_ROOT = Path("/home/shivansg/ImProver")

TOPICS = [
    "Combinatorics", "Computability", "Logic", "ModelTheory", 
    "NumberTheory", "Probability"
]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=improver_{topic_lower}
#SBATCH --output=/home/shivansg/ImProver/logs/improver_{topic_lower}_%j.out
#SBATCH --error=/home/shivansg/ImProver/logs/improver_{topic_lower}_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/shivansg/ImProver

echo "============================================================"
echo "Generating ImProver prompts for {topic}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Run the prompts generation
./improver prompts get \\
    --dataset_path /home/shivansg/ImProver/data/{topic_lower}_dataset.json \\
    --prompts_id {topic_lower}_prompts \\
    --cpus 8

echo "============================================================"
echo "Counting generated prompts..."
find /home/shivansg/ImProver/prompts/{topic_lower}_prompts -name "*.json" | wc -l
echo "Job Complete"
echo "============================================================"
"""

def main():
    # Create logs directory
    Path("/home/shivansg/ImProver/logs").mkdir(exist_ok=True)
    Path("/home/shivansg/ImProver/slurm_scripts").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Submitting ImProver Prompts Generation Jobs")
    print("=" * 60)
    
    submitted_jobs = []
    
    for topic in TOPICS:
        print(f"\nProcessing {topic}...")
        
        # Find all .lean files in the topic
        topic_dir = MATHLIB_ROOT / topic
        if not topic_dir.exists():
            print(f"  Warning: {topic_dir} does not exist")
            continue
        
        files = [f"Mathlib/{topic}/{f.relative_to(topic_dir)}" 
                 for f in topic_dir.rglob("*.lean")]
        
        print(f"  Found {len(files)} files")
        
        # Create dataset JSON
        dataset = {"train": {"mathlib4": files}}
        dataset_path = IMPROVER_ROOT / "data" / f"{topic.lower()}_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"  Created {dataset_path}")
        
        # Create Slurm script
        slurm_content = SLURM_TEMPLATE.format(topic=topic, topic_lower=topic.lower())
        slurm_path = IMPROVER_ROOT / "slurm_scripts" / f"run_{topic.lower()}_prompts.slurm"
        with open(slurm_path, "w") as f:
            f.write(slurm_content)
        
        # Submit job
        result = subprocess.run(
            ["sbatch", str(slurm_path)],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  Submitted job {job_id}")
            submitted_jobs.append((topic, job_id))
        else:
            print(f"  FAILED: {result.stderr}")
    
    print()
    print("=" * 60)
    print(f"Submitted {len(submitted_jobs)} jobs:")
    for topic, job_id in submitted_jobs:
        print(f"  {topic}: {job_id}")
    print("Check status with 'squeue -u shivansg'")
    print("=" * 60)

if __name__ == "__main__":
    main()
