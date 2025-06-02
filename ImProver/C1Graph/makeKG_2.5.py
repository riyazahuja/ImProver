import os
import json
from neo4j import GraphDatabase

# KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG"
KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG2.75"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12P@ssword21"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def init_node(tx, theorem):
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text
        SET t.c2 = $c2
        """,
        name=theorem["name"],
        text=theorem["text"],
        module=theorem["module"],
        c2=theorem["isExtracted"],
    )


def create_nodes(tx, theorem):
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
            theorem_module=module,
            dep_name=dep["name"],
            dep_module=dep["module"],
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
            # with open(os.path.join(root, file).replace("KG", "KG2.5"), "r") as f:
            #     C2Data = json.load(f)
            for thm in theorems:
                session.write_transaction(create_nodes, thm)


driver.close()
