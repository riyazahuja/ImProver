import os
import json
import argparse
import asyncio
import multiprocessing
from types import SimpleNamespace
from typing import List, Dict, Iterable

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

from ImProver.efficient.inference import run_inference

class CoTrainer:
    def __init__(self, conj_model: str, prov_model: str, neo4j_uri: str,
                 neo4j_user: str, neo4j_pass: str, chroma_dir: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        client = PersistentClient(path=chroma_dir)
        embed = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
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
        with self.driver.session() as session:
            if frontier:
                query = (
                    "MATCH (t:Theorem) WHERE NOT EXISTS{ MATCH (:Theorem)-[:DEPENDS_ON]->(t) } "
                    "RETURN t.module AS module, t.name AS name, t.text AS text"
                )
            else:
                query = "MATCH (t:Theorem) RETURN t.module AS module, t.name AS name, t.text AS text"
            res = session.run(query)
            return [r.data() for r in res]


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
        with self.driver.session() as session:
            session.run(
                "MERGE (t:Theorem {name:$name, module:$module}) "
                "SET t.text=$text, t.isExtracted=false, t.isOriginal=true, t.isConjectured=true",
                name=name, module=module, text=text,
            )
            for dep in deps:
                session.run(
                    "MATCH (t:Theorem {name:$tname, module:$tmod}) MATCH (d:Theorem {name:$dname, module:$dmod}) "
                    "MERGE (t)-[:DEPENDS_ON]->(d)",
                    tname=name, tmod=module, dname=dep["name"], dmod=dep["module"],
                )

    def add_vector(self, module: str, name: str, statement: str, proof: str):
        doc = f"{statement}\n\n{proof}"
        self.collection.add(
            documents=[doc],
            ids=[f"{module}:{name}"],
            metadatas=[{"module": module, "name": name}]
        )

    def batch_generate(self, prompts: Iterable[str], n: int, model_path: str) -> List[List[str]]:
        df = pd.DataFrame({
            "file_path": [str(i) for i in range(len(prompts))],
            "decl": [str(i) for i in range(len(prompts))],
            "decl_idx": list(range(len(prompts))),
            "raw_prompt": list(prompts),
        })

        args = SimpleNamespace(
            cpus=multiprocessing.cpu_count(),
            gpus=max(1, torch.cuda.device_count()),
            n=n,
            model=model_path,
            output_dir="cotraining_runs",
            dataset_path="",
            split="",
            metric="",
            annotation=False,
            context=0,
            rag=0,
        )

        run_dir = run_inference(df, args)
        con = duckdb.connect(os.path.join(run_dir, "data.duckdb"))
        df_out = con.execute(
            "SELECT decl_idx, prompt_idx, answer FROM run_data ORDER BY decl_idx, prompt_idx"
        ).fetchdf()
        con.close()
        results = [[] for _ in range(len(prompts))]
        for _, row in df_out.iterrows():
            results[int(row["decl_idx"])].append(row["answer"])
        return results

    async def _lean_check_async(self, module: str, name: str, code: str) -> bool:
        path = os.path.join("temp_eval", module.replace(".", "/"))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{name}.lean")
        with open(file_path, "w") as f:
            f.write(f"import {module}\n\n")
            f.write(code)
        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "env",
                "lean",
                file_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=60)
            return proc.returncode == 0
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            return False

    def lean_check_batch(self, items: Iterable[Dict], concurrency: int = None) -> List[bool]:
        async def run_all():
            sem = asyncio.Semaphore(concurrency or multiprocessing.cpu_count())

            async def worker(it):
                async with sem:
                    return await self._lean_check_async(it["module"], it["name"], it["code"])

            tasks = [asyncio.create_task(worker(it)) for it in items]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())

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

    def run_iteration(self, k: int = 10, c: int = 3, best_of_n: int = 4,
                      t: float = 0.25, t_prime: float = 0.25,
                      related_thresh: float = 0.3, novel_thresh: float = 0.8,
                      frontier: bool = True):
        seeds = self.get_seeds(frontier)
        random.shuffle(seeds)
        seeds = seeds[:k]

        # generate many conjectures per seed using batch inference
        prompts = [
            f"Propose a challenging yet correct Lean theorem related to:\n{seed['text']}\nThe statement is:"
            for seed in seeds
        ]
        conj_lists = self.batch_generate(prompts, c, self.conj_model_path)

        conjectures = []
        seed_index = []
        for idx, lst in enumerate(conj_lists):
            for text in lst:
                conjectures.append(text)
                seed_index.append(idx)

        # generate many proofs for each conjecture
        proof_prompts = [
            f"Provide a Lean4 proof for the following theorem:\n{text}\nproof"
            for text in conjectures
        ]
        proof_lists = self.batch_generate(proof_prompts, best_of_n, self.prov_model_path)

        flat_items = []
        for cidx, lst in enumerate(proof_lists):
            seed_idx = seed_index[cidx]
            module = seeds[seed_idx]["module"]
            for j, proof in enumerate(lst):
                flat_items.append({"module": module, "name": f"tmp_{cidx}_{j}", "code": proof})

        proof_valid = self.lean_check_batch(flat_items)

        conj_data, prov_data = [], []
        idx = 0
        for cidx, proofs in enumerate(proof_lists):
            valid_flags = proof_valid[idx: idx + len(proofs)]
            idx += len(proofs)
            pass_rate = sum(valid_flags) / max(1, len(proofs))
            if pass_rate == 0 or pass_rate > t_prime:
                continue

            seed_idx = seed_index[cidx]
            seed = seeds[seed_idx]
            conjecture = conjectures[cidx]

            rel = self.similarity(seed["text"], conjecture)
            nov = self.novel_score(conjecture)
            if rel < related_thresh or nov < novel_thresh:
                continue

            valid_proofs = [p for p, v in zip(proofs, valid_flags) if v]
            if not valid_proofs:
                continue

            proof = valid_proofs[0]
            name = f"conj_{abs(hash(conjecture))}"
            conj_data.append({
                "input": seed["text"],
                "output": conjecture,
                "module": seed["module"],
                "seed_name": seed["name"],
                "name": name,
            })
            prov_data.append({
                "input": conjecture,
                "output": proof,
                "module": seed["module"],
                "name": name,
            })
            self.update_kg(seed["module"], name, conjecture, [{"module": seed["module"], "name": seed["name"]}])
            self.add_vector(seed["module"], name, conjecture, proof)
        os.makedirs("cotraining_data", exist_ok=True)
        with open("cotraining_data/conjecturer.jsonl", "w") as f:
            for item in conj_data:
                f.write(json.dumps(item) + "\n")
        with open("cotraining_data/prover.jsonl", "w") as f:
            for item in prov_data:
                f.write(json.dumps(item) + "\n")
        # Train models on the newly collected data
        self.train_conjecturer("cotraining_data/conjecturer.jsonl")
        self.train_prover("cotraining_data/prover.jsonl")


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
