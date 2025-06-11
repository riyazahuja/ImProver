import os
import json
import argparse
import duckdb
from neo4j import GraphDatabase
from tqdm import tqdm


def init_node(tx, theorem):
    text = theorem.get("text", None)
    isExtracted = theorem.get("isExtracted", None)
    isOriginal = theorem.get("isOriginal", None)
    
    tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        """,
        name=theorem["name"],
        module=theorem["module"],
    )
    
    if text is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.text = $text
        """,
        name=theorem["name"],
        text=text,
        module=theorem["module"],
        )
    
    if isExtracted is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.isExtracted = $isExtracted
        """,
        name=theorem["name"],
        isExtracted=isExtracted,
        module=theorem["module"],
        )
    
    if isOriginal is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.isOriginal = $isOriginal
        """,
        name=theorem["name"],
        isOriginal=isOriginal,
        module=theorem["module"],
        )


def create_edges(tx, thm, deps, rel):
    for dep in deps:
        tx.run(
            """
            MATCH (t:Theorem {name: $theorem_name, module: $theorem_module})
            MATCH (d:Theorem {name: $dep_name, module: $dep_module})
            MERGE (t)-[r:%s]->(d)
            """ % rel,
            theorem_name=thm["name"],
            theorem_module=thm["module"],
            dep_name=dep["name"],
            dep_module=dep["module"],
        )


def process_row(tx, row):
    thm = {
        "name": row["name"],
        "module": row["module"],
        "text": row["text"],
        "isExtracted": row["isExtracted"],
        "isOriginal": row["isOriginal"],
    }
    c1 = json.loads(row["C1Dependencies"]) if row["C1Dependencies"] else []
    c2 = json.loads(row["C2Dependencies"]) if row["C2Dependencies"] else []
    c3 = json.loads(row["C3Dependencies"]) if row["C3Dependencies"] else []

    init_node(tx, thm)
    for dep in c1 + c2 + c3:
        init_node(tx, dep)

    create_edges(tx, thm, c1, "DEPENDS_ON")
    create_edges(tx, thm, c2, "DEPENDS_ON")
    create_edges(tx, thm, c3, "INFORMALLY_DEPENDS_ON")

def remove_informal_cycles(session):
    """Remove bidirectional INFORMALLY_DEPENDS_ON relationships"""
    # Find all bidirectional relationships (a->b and b->a)
    result = session.run("""
        MATCH (a:Theorem)-[r1:INFORMALLY_DEPENDS_ON]->(b:Theorem)
        MATCH (b)-[r2:INFORMALLY_DEPENDS_ON]->(a)
        WHERE id(a) < id(b)  // Avoid duplicate pairs
        RETURN id(r1) AS r1_id, id(r2) AS r2_id
    """)
    
    bidirectional_pairs = [(record["r1_id"], record["r2_id"]) for record in result]
    
    if bidirectional_pairs:
        # Flatten the list of tuples to a single list of relationship IDs
        rel_ids = [rel_id for pair in bidirectional_pairs for rel_id in pair]
        print(f"Removing {len(rel_ids)} INFORMALLY_DEPENDS_ON relationships that form bidirectional dependencies")
        
        session.run("""
            UNWIND $rel_ids AS id
            MATCH ()-[r]-()
            WHERE id(r) = id
            DELETE r
        """, rel_ids=rel_ids)
    else:
        print("No bidirectional INFORMALLY_DEPENDS_ON relationships found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KG with class3 edges")
    parser.add_argument("db_path", type=str, help="Path to combined duckdb")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_pass", type=str, default="12345678")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    # con = duckdb.connect(args.db_path, read_only=True)
    # rows = con.execute("SELECT * FROM theorems").fetchall()
    # cols = [c[1] for c in con.execute("PRAGMA table_info('theorems')").fetchall()]
    # con.close()

    with driver.session() as session:
        # for row in tqdm(rows, desc="Uploading"):
        #     row_dict = dict(zip(cols, row))
        #     session.write_transaction(process_row, row_dict)
        
        # After creating all nodes and edges, remove cycles
        print("Checking for cycles in INFORMALLY_DEPENDS_ON relationships...")
        remove_informal_cycles(session)

    driver.close()
