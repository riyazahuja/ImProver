import os
import json
from neo4j import GraphDatabase
import hashlib

KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12P@ssword21"  # replace with actual

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# Hash a module name to a hex color code
def module_to_color(module):
    h = hashlib.md5(module.encode()).hexdigest()
    return "#" + h[:6]


def create_nodes(tx, theorem, module):
    color = module_to_color(module)
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text,
            t.color = $color
        """,
        name=theorem["name"],
        text=theorem["text"],
        module=module,
        color=color,
    )
    for dep in theorem["dependencies"]:
        dep_color = module_to_color(dep["module"])
        tx.run(
            """
            MERGE (d:Theorem {name: $dep_name, module: $dep_module})
            SET d.text = $dep_text,
                d.color = $dep_color
            WITH d
            MATCH (t:Theorem {name: $theorem_name, module: $theorem_module})
            MERGE (d)-[:DEPENDS_ON]->(t)
            """,
            dep_name=dep["name"],
            dep_text=dep["text"],
            dep_module=dep["module"],
            dep_color=dep_color,
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
                for theorem in theorems:
                    session.write_transaction(create_nodes, theorem, module)

driver.close()
