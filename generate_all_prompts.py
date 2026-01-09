#!/usr/bin/env python3
"""
Generate ImProver prompts (content_sorry) for all Mathlib topics.
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

def main():
    for topic in TOPICS:
        print(f"\n{'='*60}")
        print(f"Processing {topic}")
        print(f"{'='*60}")
        
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
        
        # Run prompts get
        prompts_id = f"{topic.lower()}_prompts"
        cmd = [
            "./improver", "prompts", "get",
            "--dataset_path", str(dataset_path),
            "--prompts_id", prompts_id,
            "--cpus", "8"
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=IMPROVER_ROOT, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"  STDERR: {result.stderr}")
        
        # Count generated prompts
        prompts_dir = IMPROVER_ROOT / "prompts" / prompts_id / "src"
        if prompts_dir.exists():
            n_files = len(list(prompts_dir.rglob("*.json")))
            print(f"  Generated {n_files} prompt files")
        else:
            print(f"  Warning: No prompts generated")

if __name__ == "__main__":
    main()
