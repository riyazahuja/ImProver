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


def build_prompt(thm, include_context=False):
    context = ""
    if include_context:
        deps = thm.get("C1_dependencies", [])
        dep_texts = "\n\n".join(d.get("text", "").strip() for d in deps)
        if dep_texts:
            context = f"<CONTEXT>\n{dep_texts}\n</CONTEXT>\n"
    theorem_text = thm.get("text", "").strip()
    prompt = (
        f"<THEOREM>\n{theorem_text}\n</THEOREM>\n"
        f"{context}"
        "Provide an informal statement and proof following the formal proof step by step."
        " Wrap the statement in <INFORMAL_STATEMENT> tags and the proof in <INFORMAL_PROOF> tags."
    )
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
    for root, _, files in os.walk(kg_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            if "filtered" in root and "config" in file:
                continue
            module_path = os.path.relpath(os.path.join(root, file), kg_dir)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
            for thm in theorems:
                thm_module = thm.get("module", module)
                if thm_module not in modules:
                    continue
                prompt = build_prompt(thm, include_context)
                prompts.append({
                    "prompt": prompt,
                    "module": thm_module,
                    "name": thm.get("name"),
                    "text": thm.get("text"),
                    "isExtracted": thm.get("isExtracted", False),
                    "isOriginal": True,
                })
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
            sampling_params=dict(truncate_prompt_tokens=16384 - 512, max_tokens=512),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    ds = ray.data.from_pandas(df).repartition(max(1, args.gpus) * 4)
    ds = processor(ds).materialize()

    output_dir = os.path.join(args.KG_dir, "class3", "data")
    os.makedirs(output_dir, exist_ok=True)
    ds.write_parquet(f"local://{output_dir}")
    return output_dir


def populate_database(output_dir, kg_dir):
    db_path = os.path.join(kg_dir, "class3", "informal_data.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS informal_data")
    con.execute(f"CREATE TABLE informal_data AS SELECT * FROM read_parquet('{output_dir}/*.parquet')")
    con.execute("ALTER TABLE informal_data ADD COLUMN IF NOT EXISTS informal_statement TEXT")
    con.execute("ALTER TABLE informal_data ADD COLUMN IF NOT EXISTS informal_proof TEXT")
    rows = con.execute("SELECT rowid, answer FROM informal_data").fetchall()
    for rowid, answer in tqdm(rows, desc="Parsing outputs"):
        stmt_match = re.search(r"<INFORMAL_STATEMENT>([\s\S]*?)</INFORMAL_STATEMENT>", answer)
        proof_match = re.search(r"<INFORMAL_PROOF>([\s\S]*?)</INFORMAL_PROOF>", answer)
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
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--KG_dir", type=str, default="KG2.75")
    parser.add_argument("--include_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    df = collect_prompts(args.KG_dir, args.dataset_path, args.split, args.include_context)
    if len(df) == 0:
        print("No theorems to process")
        exit()
    output_dir = run_inference(df, args)
    populate_database(output_dir, args.KG_dir)
