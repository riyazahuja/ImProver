import os
import json
from neo4j import GraphDatabase
import argparse
import duckdb
from tqdm import tqdm

# KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG"


def init_node(tx, theorem):
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text
        SET t.isExtracted = $isExtracted
        SET t.isOriginal = $isOriginal
        """,
        name=theorem["name"],
        text=theorem["text"],
        module=theorem["module"],
        isExtracted=theorem["isExtracted"],
        isOriginal=theorem["module"] in files,
    )


def create_nodes(tx, theorem, args):
    c1_dependencies = theorem.get("C1_dependencies", [])
    c2_dependencies = theorem.get("C2_dependencies", [])
    dependencies = c1_dependencies + c2_dependencies

    init_node(tx, theorem)

    for dep in dependencies:
        init_node(tx, dep)

    for dep in dependencies:
        tx.run(
            """
            MATCH (t:Theorem {name: $theorem_name, module: $theorem_module})
            MATCH (d:Theorem {name: $dep_name, module: $dep_module})
            MERGE (t)-[:DEPENDS_ON]->(d)
            """,
            theorem_name=theorem["name"],
            theorem_module=theorem["module"],
            dep_name=dep["name"],
            dep_module=dep["module"],
        )

    # if heuristic filtering:
    # Add code to check in DuckDB database
    try:

        # Connect to the DuckDB database
        db_path = os.path.join(args.KG_path, "filtered/data.duckdb")
        if os.path.exists(db_path):
            conn = duckdb.connect(db_path)

            # Query the database for the theorem
            result = conn.execute(
                """
                SELECT * 
                FROM theorems 
                WHERE name = ? AND module = ?
                """,
                [theorem["name"], theorem["module"]],
            ).fetchone()

            if result:
                print(f"Found in filtered database: {result}")
            else:
                print(
                    f"Sorry! Theorem {theorem['name']} from {theorem['module']} not found in filtered database"
                )

            conn.close()
    except Exception as e:
        print(f"Error accessing DuckDB database: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate KG for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--KG_path",
        type=str,
        default="KG2.75",
        help="Directory to get KG raws (default: KG2.75)",
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j_user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--neo4j_pass",
        type=str,
        default="1234",
        help="Neo4j password (default: 1234)",
    )

    # parser.add_argument(
    #     "--cpus",
    #     type=int,
    #     default=cpu_count(),
    #     help="Number of CPUs to use (default: all available)",
    # )

    args = parser.parse_args()
    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass)
    )

    with open(args.dataset_path, "r") as f:
        all = json.load(f)
        dataset = all[args.split]
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]
    files = set([f.replace(".lean", "").replace("/", ".") for f in files_to_process])

    with driver.session() as session:
        # Get the list of all JSON files first
        all_files = []
        for root, _, files in os.walk(args.KG_path):
            for file in files:
                if file.endswith(".json"):
                    all_files.append((root, file))

        # Create the outer progress bar for files
        for root, file in tqdm(all_files, desc="Processing files"):
            module_path = os.path.relpath(os.path.join(root, file), args.KG_path)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)

            # Inner progress bar for theorems in the current file
            for thm in tqdm(theorems, desc=f"Processing {module}", leave=False):
                session.write_transaction(create_nodes, thm, args)

    driver.close()

    # asyncio.run(main_async(args))


# KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG2.75"
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASS = "12P@ssword21"
# DATASET_PATH = "/Users/ahuja/Desktop/ImProver-fresh/dataset.json"
