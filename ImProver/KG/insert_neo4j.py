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
    isCorrect = theorem.get("isCorrect", None)
    informalStatement = theorem.get("informalStatement", None)
    informalProof = theorem.get("informalProof", None)
    errorMessages = theorem.get("errorMessages", None)
    
    
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
    
    if isCorrect is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.isCorrect = $isCorrect
        """,
        name=theorem["name"],
        isCorrect=isCorrect,
        module=theorem["module"],
        )
    
    if informalStatement is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.informalStatement = $informalStatement
        """,
        name=theorem["name"],
        informalStatement=informalStatement,
        module=theorem["module"],
        )
    
    if informalProof is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.informalProof = $informalProof
        """,
        name=theorem["name"],
        informalProof=informalProof,
        module=theorem["module"],
        )
    
    if errorMessages is not None:
        tx.run(
        """
        MERGE (t:Theorem {name: $name, module: $module})
        SET t.errorMessages = $errorMessages
        """,
        name=theorem["name"],
        errorMessages=errorMessages,
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


def process_row(tx, row, con):
    thm = {
        "name": row["name"],
        "module": row["module"],
        "text": row["text"],
        "isExtracted": row["isExtracted"],
        "isOriginal": row["isOriginal"],
        "isCorrect": row["isCorrect"],
        "informalStatement": row["informalStatement"],
        "informalProof": row["informalProof"],
        "errorMessages": row["errorMsgs"]
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
    
    dependencies = c1 + c2
    result = con.execute(
            f"""
            SELECT core_dependencies
            FROM run_data 
            WHERE name = '{thm["name"].replace("'","''")}' AND module = '{thm["module"].replace("'","''")}'
            """,
        ).fetchone()

    try:
        core_dependency_indices = json.loads(result[-1])
        if len(core_dependency_indices) == 0:
            core_dependencies = dependencies
        else:
            core_dependencies = [
                dependencies[i] for i in core_dependency_indices if i < len(dependencies)
            ]
        
        create_edges(tx, thm, core_dependencies, "STRONGLY_DEPENDS_ON")
        
        
        #optionally, induce strong out-neighbors in c2 children
        for dep in c2:
            # Find out-neighbors of the dep
            neighbor_result = tx.run(
                """
                MATCH (d:Theorem {name: $dep_name, module: $dep_module})-[:DEPENDS_ON]->(n:Theorem)
                RETURN n.name as name, n.module as module
                """,
                dep_name=dep["name"],
                dep_module=dep["module"]
            ).data()
            
            # Convert to list of (name, module) pairs
            sub_dependencies = [(entry["name"], entry["module"]) for entry in neighbor_result]
            
            # Filter to those also in core_dependencies
            core_dep_pairs = [(dep["name"], dep["module"]) for dep in core_dependencies]
            sub_core_dependencies = [sub_dep for sub_dep in sub_dependencies if sub_dep in core_dep_pairs]
            
            create_edges(tx, dep, sub_core_dependencies, "STRONGLY_DEPENDS_ON")
            
            
    except Exception as e:
        # print(
        #     f"Sorry! Theorem {thm['name']} from {thm['module']} not found in filtered database, or error!\n\n{e}"
        # )
        pass

    

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

def main(args):
    
    
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    
    db_path = os.path.join("knowledge_graphs", args.KG_id,"combined.duckdb")
    
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("SELECT * FROM theorems").fetchall()
    cols = [c[1] for c in con.execute("PRAGMA table_info('theorems')").fetchall()]
    con.close()

    filtered_path = os.path.join("knowledge_graphs", args.KG_id,"filtered_data.duckdb")
    con = duckdb.connect(filtered_path, read_only=True)
    
    
    
    with driver.session() as session:
        for row in tqdm(rows, desc="Uploading"):
            row_dict = dict(zip(cols, row))
            session.write_transaction(process_row, row_dict, con)
        
        # After creating all nodes and edges, remove cycles
        print("Checking for cycles in INFORMALLY_DEPENDS_ON relationships...")
        remove_informal_cycles(session)

    driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KG with class3 edges")
    parser.add_argument("KG_id", type=str)

    # parser.add_argument("KG_dir", type=str, help="Path to KG directory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_pass", type=str, default="12345678")
    args = parser.parse_args()
    main(args)

