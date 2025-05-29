import os
import json
from neo4j import GraphDatabase

KG_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG"
KG2_PATH = "/Users/ahuja/Desktop/ImProver-fresh/KG2.5"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12P@ssword21"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def create_nodes(tx, theorem, c2, module):
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text
        SET t.c2 = $c2
        """,
        name=theorem["name"],
        text=theorem["text"],
        module=module,
        c2=False,
    )
    for dep in theorem["dependencies"]:
        tx.run(
            """
            MERGE (d:Theorem {name: $dep_name, module: $dep_module})
            SET d.text = $dep_text
            SET d.c2 = False
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
    for dep in c2["dependencies"]:
        tx.run(
            """
            MERGE (d:Theorem {name: $dep_name, module: $dep_module})
            SET d.text = $dep_text
            SET d.c2 = True
            WITH d
            MATCH (t:Theorem {name: $theorem_name, module: $theorem_module})
            MERGE (t)-[:DEPENDS_ON]->(d)
            """,
            dep_name=f'{theorem["name"]}_{dep["name"]}',
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
                print(f)
                theorems = json.load(f)
            with open(os.path.join(root, file).replace("KG", "KG2.5"), "r") as f:
                C2Data = json.load(f)
            for i in range(min(len(theorems), len(C2Data))):
                theorem = theorems[i]
                c2 = C2Data[i]
                session.write_transaction(create_nodes, theorem, c2, module)

driver.close()
