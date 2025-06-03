import os
import json
from neo4j import GraphDatabase

KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG"  # your path
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12P@ssword21"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def create_nodes(tx, theorem, module):
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text
        """,
        name=theorem["name"],
        text=theorem["text"],
        module=module,
    )
    for dep in theorem["dependencies"]:
        tx.run(
            """
            MERGE (d:Theorem {name: $dep_name, module: $dep_module})
            SET d.text = $dep_text
            WITH d
            MATCH (t:Theorem {name: $theorem_name, module: $theorem_module})
            MERGE (t)-[:DEPENDS_ON]->(d)
            """,
            dep_name=dep["name"],
            dep_text=dep["text"],
            dep_module=dep["module"],
            theorem_name=theorem["name"],
            theorem_module=module,
        )


with driver.session() as session:
    for root, _, files in os.walk(KG_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue
            module_path = os.path.relpath(os.path.join(root, file), KG_PATH)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
                print(f"Added {os.path.relpath(os.path.join(root, file),KG_PATH)}")
                for theorem in theorems:
                    session.write_transaction(create_nodes, theorem, module)

driver.close()
