import os
import json
import argparse
import asyncio
import multiprocessing
from types import SimpleNamespace
from typing import List, Dict, Iterable
import re
import duckdb
import pandas as pd
from neo4j import GraphDatabase
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
import torch
import random
import tqdm
from efficient.inference import run_inference
import time

def make_conjecturer_prompt_old(conjecture: str) -> str:
    return f"You are a Lean4 library builder and (formal) mathematician. Given a Lean4 theorem and proof (referred to as the seed theorem) conjecture a formal theorem statement. More explicitly, given a seed theorem, come up with a conjecture that builds off of and expands upon that theorem that may be correct, and is novel, interesting, and useful. This conjecture should be a formal lean4 theorem statement (you can leave the proof as \":= by sorry\"). Feel free to first explore related ideas and concepts at a high level in informal mathematics, but for the final output, be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n<CURRENT>{conjecture}</CURRENT>\n\n<IMPROVED>"

def make_conjecturer_prompt(conjecture: str) -> str:
    return f"You are a Lean4 library builder and (formal) mathematician. Given a Lean4 theorem and proof (referred to as the seed theorem) conjecture a formal theorem statement. More explicitly, given a seed theorem, come up with a conjecture that builds off of and expands upon that theorem that may be correct, and is novel, interesting, and useful. This conjecture should be a formal lean4 theorem statement (you can leave the proof as \":= by sorry\"). Feel free to first explore related ideas and concepts at a high level in informal mathematics, but for the final output, be sure to output your final response as a novel, interesting, and distinct Lean4 theorem conjecture wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n<CURRENT>{conjecture}</CURRENT>\n\n<IMPROVED>"

def make_prover_prompt(theorem: str) -> str:
    return f"You are an expert Lean4 theorem proving assistant and formal mathematician. Prove the current theorem (wrapped in <CURRENT>...</CURRENT>) with a correct, formal, and complete (sorry-free) Lean4 proof. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n<CURRENT>{theorem}</CURRENT>\n\n<IMPROVED>"
    


class CoTrainer:
    def __init__(self, conj_model: str, prov_model: str, neo4j_uri: str,
                 neo4j_user: str, neo4j_pass: str, chroma_dir: str):
        # self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        client = PersistentClient(path=chroma_dir)
        # embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="Qwen/Qwen3-Embedding-0.6B"
        # )
        model = "Qwen/Qwen3-Embedding-0.6B"  # or any other embedding model you prefer
        embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model,
        device="cuda",                      # push model to available GPU(s)
        model_kwargs={
            "device_map": "cuda:0",           # shard across multiple GPUs if present
            "torch_dtype": torch.float16    # cut VRAM/RAM usage in half
        }
    )
        self.collection = client.get_or_create_collection("informal_theorems", embedding_function=embed)
        device_idx = 0 if torch.cuda.is_available() else -1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # keep local models for training but generation is handled by vLLM
        self.conj_tokenizer = AutoTokenizer.from_pretrained(conj_model)
        self.conj_model = AutoModelForCausalLM.from_pretrained(conj_model)
        self.prov_tokenizer = AutoTokenizer.from_pretrained(prov_model)
        self.prov_model = AutoModelForCausalLM.from_pretrained(prov_model)
        self.conj_model_path = conj_model
        self.prov_model_path = prov_model
        self.embed_fn = embed

    def get_seeds(self, frontier: bool = True) -> List[Dict]:
        # with self.driver.session() as session:
        #     if frontier:
        #         query = (
        #             "MATCH (t:Theorem) WHERE NOT EXISTS{ MATCH (:Theorem)-[:DEPENDS_ON]->(t) } "
        #             "RETURN t.module AS module, t.name AS name, t.text AS text"
        #         )
        #     else:
        #         query = "MATCH (t:Theorem) RETURN t.module AS module, t.name AS name, t.text AS text"
        #     res = session.run(query)
        #     return [r.data() for r in res]
        with open("/home/riyaza/eval_improver/improver/ImProver/records_small.json", "r", encoding="utf-8-sig") as f:
            records = json.load(f)

        return [{r['keys'][i] : r['_fields'][i] for i in range(len(r['keys']))} for r in records]


    def similarity(self, a: str, b: str) -> float:
        emb = self.embed_fn([a, b])
        import numpy as np

        v1, v2 = np.array(emb[0]), np.array(emb[1])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def novel_score(self, text: str) -> float:
        res = self.collection.query(query_texts=[text], n_results=1)
        return res["distances"][0][0]

    def update_kg(self, module: str, name: str, text: str, deps: List[Dict]):
        # with self.driver.session() as session:
        #     session.run(
        #         "MERGE (t:Theorem {name:$name, module:$module}) "
        #         "SET t.text=$text, t.isExtracted=false, t.isOriginal=true, t.isConjectured=true",
        #         name=name, module=module, text=text,
        #     )
        #     for dep in deps:
        #         session.run(
        #             "MATCH (t:Theorem {name:$tname, module:$tmod}) MATCH (d:Theorem {name:$dname, module:$dmod}) "
        #             "MERGE (t)-[:DEPENDS_ON]->(d)",
        #             tname=name, tmod=module, dname=dep["name"], dmod=dep["module"],
        #         )
        pass
    
    def add_vector(self, module: str, name: str, statement: str, proof: str):
        doc = f"{statement}\n\n{proof}"
        self.collection.add(
            documents=[doc],
            ids=[f"{module}:{name}"],
            metadatas=[{"module": module, "name": name}]
        )

    def batch_generate(self, prompts, n: int, model_path: str, ray_init : bool = True, metric = "completion") -> List[List[str]]:
        
        
        df = pd.DataFrame({
            "module": [p["module"] for p in prompts],
            "decl": [p["decl"] for p in prompts],
            "decl_idx": list(range(len(prompts))),
            "raw_prompt": [p["prompt"] for p in prompts],
        })
        
        args = SimpleNamespace(
            cpus=12,#multiprocessing.cpu_count(),
            gpus=max(1, torch.cuda.device_count()),
            n=n,
            model=model_path,
            output_dir="cotraining_runs",
            dataset_path="",
            split="",
            metric=metric,
            annotation=False,
            context=0,
            rag=0,
        )

        run_dir = run_inference(df, args,ray_init=ray_init)
        con = duckdb.connect(os.path.join(run_dir, "data.duckdb"))
        df_out = con.execute(
            "SELECT decl_idx, prompt_idx, answer FROM run_data ORDER BY decl_idx, prompt_idx"
        ).fetchdf()
        con.close()
        results = [[] for _ in range(len(prompts))]
        for _, row in df_out.iterrows():
            results[int(row["decl_idx"])].append(row["answer"])
        return run_dir, results


        
    async def _eval_file(self, file, run_dir, metric="completion"):
        st = time.time()
        output_path = os.path.join(
            run_dir, "evals", file.replace(".", "/")+".json"
        )
        
        cmd = [
            "lake",
            "exe",
            "eval_improver",
            file,
            metric,
            os.path.join(run_dir),
            output_path
        ]
        print(" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                )
        
        try:
            await asyncio.wait_for(proc.wait(),20*60)
            print(f">>> success on {file}! (took {time.time()-st}s)\n")
            return
        except asyncio.TimeoutError:
            proc.kill()
            print(f">>> [TIME-OUT] {file} (> {20*60}s)")
            return
        except Exception as e:
            print(f">>> Exception running improver on {file}: {str(e)}")
            return
    
    async def lean_check_batch(self, run_dir,metric="completion") -> List[bool]:
        print("Beginning batch evaluation of Lean proofs...")
        con = duckdb.connect(os.path.join(run_dir, "data.duckdb"))
        modules_df = con.execute("SELECT DISTINCT module FROM run_data").fetchdf()
        modules = modules_df["module"].tolist()
        con.close()

        cpus = 12#multiprocessing.cpu_count()
        
        semaphore = asyncio.Semaphore(cpus)
        progress_bar = tqdm.tqdm(total=len(modules), desc="Processing files")

        async def worker(f):
            async with semaphore:
                ok = await self._eval_file(f, run_dir, metric)
                progress_bar.update(1)
                return ok
        tasks = [asyncio.create_task(worker(f)) for f in modules]
        await asyncio.gather(*tasks)
        progress_bar.close()
        
        print("Batch evaluation completed. Making database...")
        evals_dir = os.path.join(run_dir, "evals")
        db_path = os.path.join(run_dir, "eval.duckdb")
        con = duckdb.connect(db_path)
        
        #   SAFE MODE
        # con.execute(f"CREATE TABLE IF NOT EXISTS evaluation_results AS SELECT * FROM '{evals_dir}/**/*.json';")
        
        #   UN-SAFE MODE, (but probably better lol)
        con.execute("DROP TABLE IF EXISTS evaluation_results;")
        con.execute(f"CREATE TABLE evaluation_results AS SELECT * FROM '{evals_dir}/**/*.json';")

        con.close()
        print(f"Evaluation results saved to {db_path}")

    def train_model(self, model, tokenizer, jsonl_path: str, epochs: int, out_dir: str):
        if not os.path.exists(jsonl_path):
            return
        data = load_dataset("json", data_files=jsonl_path)["train"]

        def tokenize_fn(batch):
            texts = [f"{i['input']}\n{i['output']}" for i in batch]
            return tokenizer(texts, truncation=True)

        tokenized = data.map(tokenize_fn, batched=True, remove_columns=["input", "output"])
        args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            learning_rate=5e-5,
            logging_steps=10,
            save_strategy="no",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.save_model(out_dir)
        return trainer.model

    def train_conjecturer(self, jsonl_path: str, epochs: int = 1):
        model = self.train_model(
            self.conj_model,
            self.conj_tokenizer,
            jsonl_path,
            epochs,
            os.path.join("cotraining_models", "conjecturer"),
        )
        if model is not None:
            self.conj_model = model
            self.conj_model_path = os.path.join("cotraining_models", "conjecturer")

    def train_prover(self, jsonl_path: str, epochs: int = 1):
        model = self.train_model(
            self.prov_model,
            self.prov_tokenizer,
            jsonl_path,
            epochs,
            os.path.join("cotraining_models", "prover"),
        )
        if model is not None:
            self.prov_model = model
            self.prov_model_path = os.path.join("cotraining_models", "prover")

    def run_iteration(self, k: int = 100, c: int = 100, best_of_n: int = 32,
                      t: float = 0.25, t_prime: float = 0.25,
                      related_thresh: float = 0.3, novel_thresh: float = 0.8,
                      frontier: bool = True):
        seeds = self.get_seeds(frontier)
        random.shuffle(seeds)
        seeds = seeds[:k]
        print("=======================================")
        print(f"Selected {len(seeds)} seeds for co-training iteration.")
        print("=======================================")
        # generate many conjectures per seed using batch inference
        prompts = [
            {"prompt" : make_conjecturer_prompt(seed['text']),
                "module": seed["module"],
                "decl": seed["name"]}
            for seed in seeds
        ]
        conj_data_path, conj_lists = self.batch_generate(prompts, c, self.conj_model_path, metric="conjecturer")
        print("=======================================")
        print(f"Ran inference on conjectures to get {len(conj_lists)} conjecture lists.")
        print("=======================================")
        
        asyncio.run(self.lean_check_batch(conj_data_path, "conjecturer"))

        eval_conn = duckdb.connect(os.path.join(conj_data_path, "eval.duckdb"))
        conj_df = eval_conn.execute(
        "SELECT decl_idx, new_trimmed, decl, module FROM evaluation_results WHERE new_correct=true ORDER BY decl_idx"
        ).fetchdf()
        eval_conn.close()
        
        # # Open the conjecture database and extract the generated conjectures
        # con = duckdb.connect(os.path.join(conj_data_path, "data.duckdb"))
        # conj_df = con.execute(
        #     "SELECT decl_idx, answer, decl, module FROM run_data ORDER BY decl_idx"
        # ).fetchdf()
        # con.close()
        
        # Parse conjectures and create proof prompts
        conjectures = []
        seed_index = []
        proof_prompts = []
        
        for _, row in conj_df.iterrows():
            conjecture = row["new_trimmed"]
            conjectures.append(conjecture)
            seed_index.append(int(row["decl_idx"]))
            
            proof_prompts.append({
            "prompt": make_prover_prompt(conjecture),
            "module": row["module"],
            "decl": row["decl"]
            })
        
        # Generate proofs using batch inference
        prover_data_path, proof_lists = self.batch_generate(proof_prompts, best_of_n, self.prov_model_path, ray_init=False)
        print("=======================================")
        print(f"Generated {sum(len(lst) for lst in proof_lists)} proofs across all conjectures.")
        print("=======================================")
        print(f"Checking validity of proofs.")
        # prover_data_path = "/home/riyaza/eval_improver/improver/cotraining_runs/RUN_20250613_111733"
        asyncio.run(self.lean_check_batch(prover_data_path))
        # Read proof validity results from the evaluation database
        proof_valid = []

        eval_conn = duckdb.connect(os.path.join(prover_data_path, "eval.duckdb"))
        eval_df = eval_conn.execute(
        "SELECT decl_idx, original_prompt, new_correct, new_raw, new_trimmed, new_errors FROM evaluation_results ORDER BY decl_idx"
        ).fetchdf()
        eval_conn.close()
        
        # Create a mapping from seed theorems to conjecture results
        theorem_to_conjectures = {}
        
        for _, row in eval_df.iterrows():
            decl_idx = int(row['decl_idx'])
            seed_idx = seed_index[decl_idx]
            seed = seeds[seed_idx]
            seed_key = (seed['module'], seed['name'])
            
            if seed_key not in theorem_to_conjectures:
                theorem_to_conjectures[seed_key] = {}
            
            theorem_to_conjectures[seed_key][decl_idx] = {
                'original_prompt': row['original_prompt'],
                'new_correct': row['new_correct'],
                'new_raw': row['new_raw'],
                'new_trimmed': row['new_trimmed'],
                'new_errors': row['new_errors']
            }
            
        # Parse the theorem_to_conjectures dictionary
        raw_dataset = {}


        for seed_key, conjectures in theorem_to_conjectures.items():
            raw_dataset[seed_key] = {}
            
            # Dictionary to track conjecture statistics
            conjecture_stats = {}
            
            # Process each proof attempt
            for decl_idx, data in conjectures.items():
                original_prompt = data['original_prompt']
                
                # First try to extract from <CURRENT> tags
                match = re.search(r'<CURRENT>(.*?)</CURRENT>', original_prompt, re.DOTALL)
                conjecture_str = None
                if match:
                    conjecture_str = match.group(1).strip()
                else:
                    # # Extract from prompt format "Provide a Lean4 proof for the following theorem:\n{conjecture}\nproof"
                    # parts = original_prompt.split("following theorem:\n", 1)
                    # if len(parts) > 1:
                    # conjecture_parts = parts[1].split("\nproof", 1)
                    # if len(conjecture_parts) > 1:
                    #     conjecture_str = conjecture_parts[0].strip()
                    # else:
                    #     conjecture_str = parts[1].strip()
                    # else:
                    continue  # Skip if we can't parse
            
                # Initialize stats for this conjecture if needed
                if conjecture_str not in conjecture_stats:
                    conjecture_stats[conjecture_str] = {
                    'attempts': 0,
                    'correct': 0,
                    'proof': None
                    }
            
            # Track the attempt
            conjecture_stats[conjecture_str]['attempts'] += 1
            
            # If this attempt was correct, update the stats
            if data['new_correct']:
                conjecture_stats[conjecture_str]['correct'] += 1
                if conjecture_stats[conjecture_str]['proof'] is None:
                    conjecture_stats[conjecture_str]['proof'] = data['new_trimmed']
            
            # Finalize the raw_dataset with the required format
            for conjecture_str, stats in conjecture_stats.items():
                pass_rate = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0
                proof = stats['proof'] if pass_rate > 0 else None
                raw_dataset[seed_key][conjecture_str] = (proof, pass_rate)
        
        
        conj_data, prov_data = [], []
        
        # Create directories for training data
        os.makedirs("cotraining_data", exist_ok=True)
        
        # Process the raw_dataset
        for seed_key, conjectures in raw_dataset.items():
            seed_module, seed_name = seed_key
            seed_text = next((s["text"] for s in seeds if s["module"] == seed_module and s["name"] == seed_name), None)
            
            if not seed_text:
                continue
            
            for conjecture_str, (proof, pass_rate) in conjectures.items():
                # Apply filters
                if pass_rate == 0 or pass_rate > t_prime:
                    continue
                
                # Check relevance and novelty
                rel = self.similarity(seed_text, conjecture_str)
                nov = self.novel_score(conjecture_str)
                if rel < related_thresh or nov < novel_thresh:
                    continue
                
                # Skip if no valid proof
                if not proof:
                    continue
                
                # Generate a unique name for the conjecture
                name = f"conj_{seed_name}_{abs(hash(conjecture_str))}"
                
                # Add to training data
                conj_data.append({
                    "input": seed_text,
                    "output": conjecture_str,
                    "module": seed_module,
                    "seed_name": seed_name,
                    "name": name,
                })
                
                prov_data.append({
                    "input": conjecture_str,
                    "output": proof,
                    "module": seed_module,
                    "name": name,
                })
                
                # Update knowledge graph and vector database
                # self.update_kg(seed_module, name, conjecture_str, [{"module": seed_module, "name": seed_name}])
                # self.add_vector(seed_module, name, conjecture_str, proof)
            

        # Write data to files
        with open("cotraining_data/conjecturer.jsonl", "w") as f:
            for item in conj_data:
                f.write(json.dumps(item) + "\n")
            
        with open("cotraining_data/prover.jsonl", "w") as f:
            for item in prov_data:
                f.write(json.dumps(item) + "\n")
        # Train models on the newly collected data
        # self.train_conjecturer("cotraining_data/conjecturer.jsonl")
        # self.train_prover("cotraining_data/prover.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run offline co-training loop")
    parser.add_argument("conjecturer_model", type=str, help="HF model for conjecturer")
    parser.add_argument("prover_model", type=str, help="HF model for prover")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_pass", type=str, default="12345678")
    parser.add_argument("--chroma_dir", type=str, default="chroma_db")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    loop = CoTrainer(args.conjecturer_model, args.prover_model,
                     args.neo4j_uri, args.neo4j_user, args.neo4j_pass,
                     args.chroma_dir)
    for _ in range(args.iterations):
        loop.run_iteration()

if __name__ == "__main__":
    main()
