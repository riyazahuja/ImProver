import argparse
import json
import os
import sys
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

#!/usr/bin/env python3
"""
Extract theorem dependencies from a Neo4j knowledge graph and create a training dataset
for theorem generation based on lemmas.
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract theorem dependencies from Neo4j and create a training dataset')
    parser.add_argument('--uri', required=True, help='Neo4j URI')
    parser.add_argument('--username', required=True, help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output JSONL file')
    parser.add_argument('--system_prompt', default='Given the following lemma, generate a theorem that depends on it:', 
                        help='System prompt to prepend to each lemma')
    return parser.parse_args()

def extract_theorem_dependencies(uri, username, password, system_prompt):
    """
    Extract theorem dependencies from Neo4j where both theorems have isCorrect=true.
    
    Args:
        uri: Neo4j URI
        username: Neo4j username
        password: Neo4j password
        system_prompt: Prompt to prepend to each lemma
        
    Returns:
        List of dictionaries containing instruction and output pairs
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
    except (ServiceUnavailable, AuthError) as e:
        print(f"Error connecting to Neo4j: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Query to find theorem dependencies where both are correct
    query = """
    MATCH (theorem:Theorem)-[:DEPENDS_ON]->(lemma:Theorem)
    WHERE theorem.isCorrect = true AND lemma.isCorrect = true
    AND theorem.content IS NOT NULL AND lemma.content IS NOT NULL
    RETURN theorem.content AS theorem_content, lemma.content AS lemma_content
    """
    
    results = []
    try:
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                theorem_content = record["theorem_content"]
                lemma_content = record["lemma_content"]
                
                # Skip empty content
                if not isinstance(theorem_content, str) or not isinstance(lemma_content, str):
                    continue
                if not theorem_content.strip() or not lemma_content.strip():
                    continue
                
                # Create a training example
                example = {
                    "instruction": f"{system_prompt}\n\n{lemma_content}",
                    "output": theorem_content
                }
                results.append(example)
    except Exception as e:
        print(f"Error querying Neo4j: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()
    
    return results

def save_jsonl(data, output_path):
    """Save data as JSONL file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} examples to {output_path}")
    except IOError as e:
        print(f"Error saving JSONL file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except IOError as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract theorem dependencies
    print("Extracting theorem dependencies from Neo4j...")
    training_data = extract_theorem_dependencies(
        args.uri, 
        args.username, 
        args.password,
        args.system_prompt
    )
    
    if not training_data:
        print("Warning: No theorem dependencies found", file=sys.stderr)
    else:
        print(f"Found {len(training_data)} theorem-lemma pairs")
    
    # Save as JSONL
    output_path = os.path.join(args.output_dir, 'conjecturer.jsonl')
    save_jsonl(training_data, output_path)
    print("Done!")

if __name__ == "__main__":
    main()