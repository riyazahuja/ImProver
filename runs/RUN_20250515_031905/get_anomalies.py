import os
import json

def get_anomalies():
    eval_dir = "/home/riyaza/eval_improver/improver/runs/RUN_20250515_031905/evals"
    prompt_dir = "/home/riyaza/eval_improver/improver/prompts/length"
    
    anomalies = []
    
    # Get all JSON files recursively in both directories
    eval_files = []
    prompt_files = []
    
    # Recursively find all JSON files in eval directory
    for root, _, files in os.walk(eval_dir):
        for file in files:
            if file.endswith('.json'):
                # Store relative path for comparison
                rel_path = os.path.relpath(os.path.join(root, file), eval_dir)
                eval_files.append(rel_path)
    
    # Recursively find all JSON files in prompt directory
    for root, _, files in os.walk(prompt_dir):
        for file in files:
            if file.endswith('.json'):
                # Store relative path for comparison
                rel_path = os.path.relpath(os.path.join(root, file), prompt_dir)
                prompt_files.append(rel_path)
    
    # Check eval files against prompt files
    for eval_file in eval_files:
        # Check if corresponding file exists in prompt directory
        if eval_file not in prompt_files:
            anomalies.append(f"{eval_file}: Missing corresponding file in prompt directory")
            print(f"[ANOMALY] Missing corresponding file in prompt directory: {eval_file}")
            continue
        
        # Compare number of items in both files
        try:
            with open(os.path.join(eval_dir, eval_file), 'r') as f1:
                eval_data = json.load(f1)
            
            with open(os.path.join(prompt_dir, eval_file), 'r') as f2:
                prompt_data = json.load(f2)
                
            if len(eval_data) != len(prompt_data):
                anomalies.append(f"{eval_file}: Different number of items (eval: {len(eval_data)}, prompt: {len(prompt_data)})")
                print(f"[ANOMALY] Different number of items in {eval_file}: eval {len(eval_data)}, prompt {len(prompt_data)}")
        except Exception as e:
            anomalies.append(f"{eval_file}: Error processing file - {str(e)}")
            print(f"[ANOMALY] Error processing file {eval_file}: {str(e)}")
    
    # Check for files in prompt directory that don't have counterparts in eval directory
    for prompt_file in prompt_files:
        if prompt_file not in eval_files:
            anomalies.append(f"{prompt_file}: Missing corresponding file in eval directory")
            print(f"[ANOMALY] Missing corresponding file in eval directory: {prompt_file}")
    
    return anomalies

if __name__ == "__main__":
    anomalies = get_anomalies()
    if anomalies:
        print("\n\n\nFound anomalies:")
        for anomaly in anomalies:
            print(f"- {anomaly}")
    else:
        print("No anomalies found. All files have corresponding matches with the same number of items.")
