import argparse
import os
import json
import subprocess
import sys
def get_parser():

    parser = argparse.ArgumentParser(description="Reload metric(s).")
    parser.add_argument("--names", default=None, help="Name of the metric to reload, comma-separated. If not provided, reloads all metrics.")
    parser.add_argument("--rag_id", default=None, help="RAG ID")
    parser.add_argument("--k", default=5, help="Number of RAG results to use for each prompt")

    return parser

def main(args):

    if args.names:
        names = [name.strip() for name in args.names.split(",")]
    else:
        metrics_dir = "metrics"
        
        names = [name.strip() for name in os.listdir(metrics_dir) 
                if os.path.isdir(os.path.join(metrics_dir, name)) and name != "__pycache__"]
    
    example_modules = []
    for name in names:
        metric_path = os.path.join("metrics", name)
        config_path = os.path.join(metric_path, "config.json")
        
        if not os.path.exists(config_path):
            print(f"Metric {name} does not have a config file. Skipping.")
            continue
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        
        example_file = config["examples"]["example_file"]
        example_data = config["examples"]["example_data"]
        if example_file:
            example_module = example_file.replace(os.path.sep, ".").replace(".lean", "")
            example_modules.append((example_module, example_data))
    
    # Build metrics.examples
    try:
        subprocess.run(["lake", "build", "metrics.examples"], check=True)
        print("Built metrics.examples")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build metrics.examples: {e}")
    
    # Build get_examples
    try:
        subprocess.run(["lake", "build", "get_examples"], check=True)
        print("Built get_examples")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build get_examples: {e}")
    
    # Run get_examples for each module
    for example_module, output_path in example_modules:
        try:
            cmd = ["lake", "exe", "get_examples", example_module, output_path, sys.executable, args.rag_id, str(args.k)]
            subprocess.run(cmd, check=True)
            print(f"Extracted examples from {example_module} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to extract examples from {example_module}: {e}")
    
    # Build metrics.router
    try:
        subprocess.run(["lake", "build", "metrics.router"], check=True)
        print("Built metrics.router")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build metrics.router: {e}")
    
    
    
    
    
    
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
