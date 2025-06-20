import os
import json
import re
import argparse
from tqdm import tqdm
import pandas as pd
import duckdb
import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import multiprocessing
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer



def build_prompt(thm, include_context=False):
    
    context = ""
    context_prompt = "You will additionally be given the (formal) context/dependencies of the theorem (such as referenced lemmas), which you should use to inform your informalization of the theorem and proof. This context will be wrapped in <CONTEXT>...</CONTEXT> tags."
    if include_context:
        deps = thm.get("C1_dependencies", [])
        dep_texts = "\n\n".join(d.get("content", "").strip() for d in deps)
        if dep_texts:
            context = f"<CONTEXT>\n{dep_texts}\n</CONTEXT>\n"
            
            # TODO: Add context example?
            
            
    theorem_text = thm.get("id", {}).get("content", "").strip()
    
    prompt = f"""You are an expert in mathematics and formal theorem proving. Your task is to provide an informal statement and informal step-by-step proof for the following Lean4 formal theorem and proof.

Namely, you will be given a formal theorem and proof in Lean4 (wrapped in <FORMAL>...</FORMAL> tags), and you need to (1) provide an informal statement of the theorem in natural language (wrap this part of your output in <STATEMENT>...</STATEMENT> tags), 
and (2) provide an informal proof of the theorem in natural language by translating the formal proof tactic by tactic to produce a step-by-step aligned human-readable proof (wrapped in <PROOF>...</PROOF> tags). Do not skip any steps or omit and details in the proof.
{context_prompt if include_context else ""}

Consider the following simple example (wrapped in <EXAMPLE>...</EXAMPLE> tags):
<EXAMPLE>

Input:
<FORMAL>

theorem primes_infinite : ∀ n, ∃ p > n, Nat.Prime p := by
  intro n
  have : 2 ≤ Nat.factorial (n + 1) + 1 := by
    apply Nat.succ_le_succ
    exact Nat.succ_le_of_lt (Nat.factorial_pos _)
  rcases exists_prime_factor this with ⟨p, pp, pdvd⟩
  refine ⟨p, ?_, pp⟩
  show p > n
  by_contra ple
  push_neg at ple
  have : p ∣ Nat.factorial (n + 1) := by
    apply Nat.dvd_factorial
    apply pp.pos
    linarith
  have : p ∣ 1 := by
    convert Nat.dvd_sub' pdvd this
    simp
  show False
  have := Nat.le_of_dvd zero_lt_one this
  linarith [pp.two_le]

</FORMAL>

Output:
<STATEMENT>

For every natural number $n$, there is a prime number $p$ that is larger than $n$.
Equivalently, there are infinitely many primes.

</STATEMENT>
<PROOF>

First, we fix an arbitrary natural number $n$. 
Then, we note that $(n + 1)!+1$ is at least $2$, by definition of factorial and addition properties.
We then note that there exists a prime factor $p$ of $(n + 1)!+1$.
We aim to show that $p$ is greater than $n$, by first assuming for the sake of contradiction that $p$ is not greater than $n$.
Then as $p \\le n$, and $p$ is positive, $p$ divides $(n + 1)!$.
As $p$ divides both $(n + 1)!+1$ and $(n + 1)!$, it must also divide their difference, which is $1$.
Thus, $p$ must be at most $1$, which contradicts the fact that $p$ is a prime number, as primes are at least $2$.
Thus, we have a contradiction, and therefore $p$ must be greater than $n$.

</PROOF>
</EXAMPLE>

Now, informalize the following theorem and proof, which is wrapped in <FORMAL>...</FORMAL> tags, and be sure to wrap your informal statement in <STATEMENT>...</STATEMENT> tags and your informal proof in <PROOF>...</PROOF> tags.

{context + "\n\n" if include_context else ""}<FORMAL>\n{theorem_text}\n</FORMAL>
"""
    
    

    return prompt


def collect_prompts(kg_dir, dataset_path, split="train", include_context=False):
    with open(dataset_path, "r") as f:
        all_ds = json.load(f)
        dataset = all_ds[split]
    files = []
    for repo in dataset.values():
        files.extend(repo)
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files)

    prompts = []
    truncation_count = 0
    for root, _, files in os.walk(kg_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            if file == "config.json":
                continue
            module_path = os.path.relpath(os.path.join(root, file), kg_dir)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
            for thm in theorems:
                thm_module = thm.get("id", {}).get("module", module)
                if thm_module not in modules:
                    continue
                prompt = build_prompt(thm, include_context)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > MAX_PROMPT_TOKENS:
                    # ----- OPTION A: truncate to last MAX_PROMPT_TOKENS tokens
                    tokens = tokens[-MAX_PROMPT_TOKENS:]
                    prompt = tokenizer.decode(tokens)
                    truncation_count += 1


                prompts.append({
                    "prompt": prompt,
                    "module": thm_module,
                    "name": thm.get("id", {}).get("name"),
                    "text": thm.get("id", {}).get("content"),
                })
    print(f"Total prompts: {len(prompts)}")
    print(f"Truncated prompts: {truncation_count}")
    return pd.DataFrame(prompts)


def run_inference(df, args):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    DataContext.get_current().wait_for_min_actors_s = 1800

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": max(1, args.cpus // max(1, args.gpus)), "GPU": 1},
        concurrency=max(1, args.gpus),
        engine_kwargs={
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
            "max_model_len": 16384,
            "max_num_batched_tokens": 65536,
        },
        max_concurrent_batches=32,
        batch_size=32,
    )

    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["prompt"]}],
            sampling_params=dict(truncate_prompt_tokens=16384 - 2048, max_tokens=2048),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    ds = ray.data.from_pandas(df).repartition(max(1, args.gpus) * 8)
    ds = processor(ds).materialize()

    output_dir = os.path.join(args.KG_dir, args.KG_id, "informal_data")
    os.makedirs(output_dir, exist_ok=True)
    ds.write_parquet(f"local://{output_dir}")
    return output_dir


def populate_database(output_dir, kg_dir):
    db_path = os.path.join(kg_dir, "informal_data.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS informal_data")
    con.execute(f"CREATE TABLE informal_data AS SELECT * FROM read_parquet('{output_dir}/*.parquet')")
    con.execute("ALTER TABLE informal_data ADD COLUMN IF NOT EXISTS informal_statement TEXT")
    con.execute("ALTER TABLE informal_data ADD COLUMN IF NOT EXISTS informal_proof TEXT")
    rows = con.execute("SELECT rowid, answer FROM informal_data").fetchall()
    for rowid, answer in tqdm(rows, desc="Parsing outputs"):
        stmt_match = re.search(r"<STATEMENT>([\s|\S]*?)</STATEMENT>", answer)
        proof_match = re.search(r"<PROOF>([\s|\S]*?)</PROOF>", answer)
        stmt = stmt_match.group(1).strip() if stmt_match else ""
        proof = proof_match.group(1).strip() if proof_match else ""
        con.execute(
            "UPDATE informal_data SET informal_statement=?, informal_proof=? WHERE rowid=?",
            (stmt, proof, rowid),
        )
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Informalize theorems")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("KG_id", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    parser.add_argument("--include_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0

    parser.add_argument(
        "--gpus",
        type=int,
        default=available_gpus,
        help="Number of GPUs to use (default: all available)",
    )
    args = parser.parse_args()
    
    MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    df = collect_prompts(os.path.join(args.KG_dir,args.KG_id), args.dataset_path, args.split, args.include_context)
    if len(df) == 0:
        print("No theorems to process")
        exit()
    output_dir = run_inference(df, args)
    populate_database(output_dir, os.path.join(args.KG_dir,args.KG_id))
