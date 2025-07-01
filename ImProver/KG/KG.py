import argparse
from build_vector_db import main as vec_main
from build_combined_db import main as combined_main
from compute_class3_edges import main as c3_main
from heuristic_filter import main as filter_main
from insert_neo4j import main as neo4j_main
import torch




'''
All arguments:

dataset_path: required
split: optional, default is "train"
prompts_id: required
KG_id: optional, default is "KG_" + current timestamp
prompts_dir: optional, default is ".prompts"
KG_dir: optional, default is ".knowledge_graphs"
embedding_model: optional, default is "Qwen/Qwen3-Embedding-0.6B

k: optional, default is 40
threshold: optional, default is 0.35

heuristic_model: optional, default is "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
cpus: optional, default is number of available CPUs
gpus: optional, default is number of available GPUs
n: optional, default is 1
run_inference: optional, default is True
augment_DB: optional, default is True
training_data: optional, default is True

neo4j_uri: optional, default is "bolt://localhost:7687"
neo4j_user: optional, default is "neo4j"
neo4j_pass: optional, default is "12345678"


'''




def main():
    parser = argparse.ArgumentParser(description="Build and process Knowledge Graph")
    
    # Required arguments
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("prompts_id", type=str, help="ID for prompts")
    
    # Optional arguments with defaults
    parser.add_argument("--KG_id", type=str, help="ID for Knowledge Graph",
                        default=f"KG_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--prompts_dir", type=str, default=".prompts", help="Directory for prompts")
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs", help="Directory for knowledge graphs")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", 
                        help="Embedding model to use")
    
    # Parameters for compute_class3_edges
    parser.add_argument("--k", type=int, default=40, help="Number of neighbors to consider")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold")
    
    # Parameters for heuristic_filter
    parser.add_argument("--heuristic_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model for heuristic filtering")
    parser.add_argument("--cpus", type=int, 
                        default=__import__('multiprocessing').cpu_count(),
                        help="Number of CPUs to use")
    
    # Try to get GPU count
    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0
    
    parser.add_argument("--gpus", type=int, default=available_gpus, help="Number of GPUs to use")
    parser.add_argument("--n", type=int, default=1, help="Majority vote value")
    parser.add_argument("--run_inference", action="store_true", default=True, 
                        help="Whether to run inference")
    parser.add_argument("--augment_DB", action="store_true", default=True,
                        help="Whether to augment filteredDB")
    parser.add_argument("--training_data", action="store_true", default=True,
                        help="Whether to generate training data")
    
    # Neo4j parameters
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j_user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j_pass", type=str, default="12345678", help="Neo4j password")
    
    args = parser.parse_args()
    
    # Build vector DB
    print("[IMPROVER: Building vector database...]")
    vec_args = argparse.Namespace(
        prompts_id=args.prompts_id,
        KG_id=args.KG_id,
        prompts_dir=args.prompts_dir,
        KG_dir=args.KG_dir,
        model=args.embedding_model
    )
    vec_main(vec_args)
    
    # Build combined DB
    print("[IMPROVER: Building combined database...]")
    combined_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        KG_id=args.KG_id,
        split=args.split,
        KG_dir=args.KG_dir
    )
    combined_main(combined_args)
    
    # Compute class 3 edges
    print("[IMPROVER: Computing class 3 edges...]")
    c3_args = argparse.Namespace(
        KG_id=args.KG_id,
        KG_dir=args.KG_dir,
        model=args.embedding_model,
        k=args.k,
        threshold=args.threshold
    )
    c3_main(c3_args)
    
    # Apply heuristic filter
    print("[IMPROVER: Applying heuristic filtering...]")
    filter_args = argparse.Namespace(
        KG_id=args.KG_id,
        KG_dir=args.KG_dir,
        model=args.heuristic_model,
        cpus=args.cpus,
        gpus=args.gpus,
        n=args.n,
        run_inference=args.run_inference,
        augment_DB=args.augment_DB,
        training_data=args.training_data
    )
    filter_main(filter_args)
    
    # Insert into Neo4j
    print("[IMPROVER: Inserting into Neo4j...]")
    neo4j_args = argparse.Namespace(
        KG_id=args.KG_id,
        KG_dir=args.KG_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass
    )
    neo4j_main(neo4j_args)
    
    print(f"[IMPROVER: Knowledge Graph {args.KG_id} successfully built and processed.]")

if __name__ == "__main__":
    main()




'''
# Informalize theorems
parser = argparse.ArgumentParser(description="Build vector DB for informal theorems")
    parser.add_argument("prompts_id", type=str)
    parser.add_argument("KG_id", type=str, nargs='?', default="KG_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--prompts_dir", type=str, default=".prompts")
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()
    
    
#compute c3

parser = argparse.ArgumentParser(description="Compute class3 edges")
    parser.add_argument("KG_id", type=str)
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    # parser.add_argument("--chroma_dir", type=str, default = "chroma_db", help="Path to chroma db")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")

    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    
    
# combined 

parser = argparse.ArgumentParser(description="Build combined KG database")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("KG_id", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    args = parser.parse_args()
    
    
# heuristic filter


 parser = argparse.ArgumentParser(description="Filter KG for ImProver")
    parser.add_argument("KG_id", type=str)

    parser.add_argument(
        "--KG_dir",
        type=str,
        default=".knowledge_graphs",
        help="Directory to get KG (default: .knowledge_graphs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to use",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0

    parser.add_argument(
        "--gpus",
        type=int,
        default=available_gpus,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Majority vote value (default: 1)"
    )
    parser.add_argument(
        "--run_inference", type=bool, action=argparse.BooleanOptionalAction , default=True, help="Whether to run inference (default: True)"
    )
    parser.add_argument(
        "--augment_DB",  action=argparse.BooleanOptionalAction , type=bool, default=True, help="Whether to augment filteredDB (default: True)"
    )
    parser.add_argument(
        "--training_data", action=argparse.BooleanOptionalAction , type=bool, default=True, help="Whether to generate training data (default: True)"
    )


# neo4j

parser = argparse.ArgumentParser(description="Export KG with class3 edges")
    parser.add_argument("KG_id", type=str)

    parser.add_argument("KG_dir", type=str, help="Path to KG directory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_pass", type=str, default="12345678")
    args = parser.parse_args()
'''