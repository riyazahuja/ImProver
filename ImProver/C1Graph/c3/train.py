import argparse
import json
from neo4j import GraphDatabase

# Define constants
SYSTEM_PROMPT = "Please continue the following proof:"

def parse_args():
    parser = argparse.ArgumentParser(description='Extract theorem pairs from Neo4j and create training data')
    parser.add_argument('--informal', action=argparse.BooleanOptionalAction, default=True, help='Include informal statements in output')
    parser.add_argument('--output', default='alpaca_training.jsonl', help='Output file path')
    return parser.parse_args()

def create_alpaca_entry(a_props, b_props, include_informal=False):
    """Create an entry in Alpaca format"""
    statement = b_props.get('informalStatement', '')
    conj = a_props.get('text', '')
    if ":=" in conj:
        theorem_part, _ = conj.split(":=", 1)  # Split at first occurrence
        conj = f"{theorem_part}:= by sorry"
        
    instruction = f"""You are a Lean4 library builder and (formal) mathematician. Given a Lean4 theorem and proof (referred to as the seed theorem) conjecture a formal theorem statement. 
More explicitly, given a seed theorem, come up with a conjecture that builds off of and expands upon that theorem that may be correct, and is novel, interesting, and useful. {'You will also be given a high-level informal statement (Wrapped in <STATEMENT>...</STATEMENT>) of the seed theorem to help you understand it and formulate your conjecture.' if include_informal and statement.strip() != '' else ''}
This conjecture should be a formal lean4 theorem statement (you can leave the proof as \":= by sorry\"). 
Feel free to first explore related ideas and concepts at a high level in informal mathematics, but for the final output, be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags. 
{f'<STATEMENT>\n{statement}\n</STATEMENT>' if include_informal and statement.strip() != '' else ''}

<CURRENT>
{b_props.get('text', '')}
</CURRENT>"""    
    
    output = "<IMPROVED>\n"+ conj + "\n</IMPROVED>"
    
    return {
        "instruction": instruction,
        "output": output
    }

def main():
    args = parse_args()
    
    # Connect to Neo4j
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Query to find relationships where a --STRONGLY_DEPENDS_ON--> b and both are correct
    query = """
    MATCH (a)-[:STRONGLY_DEPENDS_ON]->(b)
    WHERE a.isCorrect = true AND b.isCorrect = true
    RETURN a, b
    """
    
    entries = []
    
    with driver.session() as session:
        results = session.run(query)
        
        for record in results:
            a_node = record["a"]
            b_node = record["b"]
            
            # Convert node properties to dictionaries
            a_props = dict(a_node)
            b_props = dict(b_node)
            
            entry = create_alpaca_entry(a_props, b_props, include_informal=args.informal)
            entries.append(entry)
    
    # Write to JSONL file
    with open(args.output, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created {len(entries)} entries in {args.output}")
    
    # Close the driver
    driver.close()

if __name__ == "__main__":
    main()