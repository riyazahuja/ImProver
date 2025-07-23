from rag import add_to_db

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add informal proof data to the database.")
    parser.add_argument("--prompt_id", type=str, required=True, help="ID of the prompt to add data for.")
    parser.add_argument("--k", type=int, default=6, help="Number of documents to retrieve.")
    
    args = parser.parse_args()
    
    add_to_db(prompt_id=args.prompt_id, k=args.k)