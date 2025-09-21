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
from packaging.version import Version



def build_prompt(thm, include_context=False):
    
    context = ""
    context_prompt = "You will additionally be given the (formal) context/dependencies of the theorem (such as referenced lemmas), which you should use to inform your informalization of the theorem and proof. This context will be wrapped in <CONTEXT>...</CONTEXT> tags."
    # if include_context:
    #     deps = thm.get("C1_dependencies", [])
    #     dep_texts = "\n\n".join(d.get("content", "").strip() for d in deps)
    #     if dep_texts:
    #         context = f"<CONTEXT>\n{dep_texts}\n</CONTEXT>\n"
            
            # TODO: Add context example?
            
            
    theorem_text = thm.get("content", "").strip()
    
#     prompt = f"""You are an expert in mathematics and formal theorem proving. Your task is to provide an informal statement and informal step-by-step proof for the following Lean4 formal theorem and proof.

# Namely, you will be given a formal theorem and proof in Lean4 (wrapped in <FORMAL>...</FORMAL> tags), and you need to (1) provide an informal statement of the theorem in natural language (wrap this part of your output in <STATEMENT>...</STATEMENT> tags), 
# and (2) provide an informal proof of the theorem in natural language by translating the formal proof tactic by tactic to produce a step-by-step aligned human-readable proof (wrapped in <PROOF>...</PROOF> tags). Do not skip any steps or omit and details in the proof.
# {context_prompt if include_context else ""}

# Consider the following simple example (wrapped in <EXAMPLE>...</EXAMPLE> tags):
# <EXAMPLE>

# Input:
# <FORMAL>

# theorem primes_infinite : ∀ n, ∃ p > n, Nat.Prime p := by
#   intro n
#   have : 2 ≤ Nat.factorial (n + 1) + 1 := by
#     apply Nat.succ_le_succ
#     exact Nat.succ_le_of_lt (Nat.factorial_pos _)
#   rcases exists_prime_factor this with ⟨p, pp, pdvd⟩
#   refine ⟨p, ?_, pp⟩
#   show p > n
#   by_contra ple
#   push_neg at ple
#   have : p ∣ Nat.factorial (n + 1) := by
#     apply Nat.dvd_factorial
#     apply pp.pos
#     linarith
#   have : p ∣ 1 := by
#     convert Nat.dvd_sub' pdvd this
#     simp
#   show False
#   have := Nat.le_of_dvd zero_lt_one this
#   linarith [pp.two_le]

# </FORMAL>

# Output:
# <STATEMENT>

# For every natural number $n$, there is a prime number $p$ that is larger than $n$.
# Equivalently, there are infinitely many primes.

# </STATEMENT>
# <PROOF>

# First, we fix an arbitrary natural number $n$. 
# Then, we note that $(n + 1)!+1$ is at least $2$, by definition of factorial and addition properties.
# We then note that there exists a prime factor $p$ of $(n + 1)!+1$.
# We aim to show that $p$ is greater than $n$, by first assuming for the sake of contradiction that $p$ is not greater than $n$.
# Then as $p \\le n$, and $p$ is positive, $p$ divides $(n + 1)!$.
# As $p$ divides both $(n + 1)!+1$ and $(n + 1)!$, it must also divide their difference, which is $1$.
# Thus, $p$ must be at most $1$, which contradicts the fact that $p$ is a prime number, as primes are at least $2$.
# Thus, we have a contradiction, and therefore $p$ must be greater than $n$.

# </PROOF>
# </EXAMPLE>

# Now, with this example in mind, informalize the following theorem and proof, which is wrapped in <FORMAL>...</FORMAL> tags, and be sure to wrap your informal statement in <STATEMENT>...</STATEMENT> tags and your informal proof in <PROOF>...</PROOF> tags.

# {context + "\n\n" if include_context else ""}<FORMAL>\n{theorem_text}\n</FORMAL>
# """
    
    
    prompt = f"""You are an expert informalizer of formal mathematics to natural language. Namely, given a formal theorem and proof in Lean4,
you will generate an informalized statement of this same theorem in natural language, as well as (2) an informalized, natural language version of the same formal proof 
that is aligned with the informal statement. Namely, when informalizing the proof, you should convert each tactic of the formal proof into a natural language step in the informal proof, and thereby, your informal proof should be written as a sequence of steps.

Consider the following example:

<EXAMPLE>

Input:
<FORMAL>
theorem and_comm' : ∀ a b : Prop, a ∧ b ↔ b ∧ a := by
  intro a b
  constructor <;>
  . intro h
    exact ⟨h.2, h.1⟩
</FORMAL>

Output:
<STATEMENT>

For all propositions $a$ and $b$, $a \\land b$ is equivalent to $b \\land a$.

</STATEMENT>
<PROOF>

First, we consider arbitrary propositions $a$ and $b$.
We break into the two directions of the equivalence, noting that we will show both directions by the same method.
For the forward direction, we assume that $a \\land b$ are true, so both $a$ and $b$ are true.
So, $b$ and $a$ are true, which means that $b \\land a$ is true. A symmetric argument shows the other direction.
</PROOF>
</EXAMPLE>

Similarly, for a more complex example, consider the following:

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


And for an example that does not require a proof, consider the following:

<EXAMPLE>

Input:
<FORMAL>
inductive Nat where
  | zero : Nat
  | succ (n : Nat) : Nat
</FORMAL>

Output:
<STATEMENT>

The natural numbers are defined inductively, with $0$ being a natural number, and the successor of a natural number being a natural number.

</STATEMENT>
<PROOF></PROOF>

</EXAMPLE>

Now, with these examples in mind, it is now your turn to informalize the following formal statement and proof, which is wrapped in <FORMAL>...</FORMAL> tags.

You may think and reason as much as you want, but ensure that your final answer for (1): the informal statement is wrapped in <STATEMENT>...</STATEMENT> tags, and (2): the informal proof is wrapped in <PROOF>...</PROOF> tags.
Your final answer should have both a <STATEMENT>...</STATEMENT> tag and a <PROOF>...</PROOF> tag, and if there is no formal proof provided in the input, you may simply output <PROOF></PROOF> for the proof after informalizing the statement (i.e. if you are given a theorem without a proof, or a definition/class/etc.).




Input:
<FORMAL>
{theorem_text}
</FORMAL>
"""
    

    return prompt


def collect_prompts(rag_dir, max_depth, tokenizer, MAX_PROMPT_TOKENS, include_context=False):
    
    data_conn = os.path.join(rag_dir, "data.duckdb")
    
    
    
    # decls_path = os.path.join(rag_dir, "decl_data.json")
    # modules_path = os.path.join(rag_dir, "module_data.json")
    # with open(decls_path, "r") as f:
    #     decls = json.load(f)
    # with open(modules_path, "r") as f:
    #     modules = json.load(f)
    
    
    # Use DuckDB to query decls whose module is in module_data and whose module's depth < max_depth
    con = duckdb.connect(data_conn)
    query = f"""
        SELECT d.*, m.depth
        FROM decl_data d
        JOIN module_data m
        ON d.module = m.module
        WHERE m.depth < {max_depth}
    """
    decls = con.execute(query).fetchall()
    columns = [desc[0] for desc in con.description]
    decls = [dict(zip(columns, row)) for row in decls]
    con.close()
    
    
    
    #TEMPORARY: REMOVE EVENTUALLY
    # informal_data_path = "/home/riyaza/eval_improver/improver/rag/final_rag_real/informal_data.duckdb"
    # con = duckdb.connect(informal_data_path)
    # all_distinct_modules = con.execute("SELECT DISTINCT module FROM informal_data").fetchall()
    # all_distinct_modules = [row[0] for row in all_distinct_modules]
    # con.close()
    
    
    
    
    
    
    

    all_prompts = []
    truncation_count = 0
    for decl in decls:
        
        
        # if decl['module'] in all_distinct_modules:
        #     print(f"skipping {decl['module']} as it was already done")
        #     continue
        
        
        prompt = build_prompt(decl, include_context=include_context)
        all_prompts.append({
            "prompt": prompt,
            "module": decl['module'],
            "name": decl['decl'],
            "text": decl['content']
        })

    print(f"Collected {len(all_prompts)} prompts")
    filtered_prompts = all_prompts  # Already filtered by SQL
    
    # all_prompts = []
    # truncation_count = 0
    # for decl in decls:
    #     prompt = build_prompt(decl)
    #     # tokens = tokenizer.encode(prompt, add_special_tokens=False)
    #     # if len(tokens) > MAX_PROMPT_TOKENS:
    #     #     truncation_count += 1
    #     #     tokens = tokens[-MAX_PROMPT_TOKENS:]
    #     #     prompt = tokenizer.decode(tokens)
    #     all_prompts.append({"prompt": prompt,
    #                         "module" : decl['module'],
    #                         "name" : decl['decl'],
    #                         "text" : decl['content']})
        
    
    
    # print(f"Collected {len(all_prompts)} prompts")
    # filtered_prompts = [prompt for prompt in all_prompts if prompt['module'] in modules and modules[prompt['module']]['depth'] <= max_depth]
    # print(f"Filtered to {len(filtered_prompts)} prompts")
    
    
    # filtered_truncated_prompts = []
    # for prompt in filtered_prompts:
    #     tokens = tokenizer.encode(prompt['prompt'], add_special_tokens=False)
    #     if len(tokens) > MAX_PROMPT_TOKENS:
    #         truncation_count += 1
    #         tokens = tokens[-MAX_PROMPT_TOKENS:]
    #         prompt['prompt'] = tokenizer.decode(tokens)
    #     filtered_truncated_prompts.append(prompt)
    # print(f"Truncated to {len(filtered_truncated_prompts)} prompts")
    
    
    # #TEMP REMOVE EVENTUALLY
    # with open(os.path.join(rag_dir, "prompts.json"), "w") as f:
    #     json.dump(filtered_truncated_prompts, f)
    
    df = pd.DataFrame(filtered_prompts)
    return df
    
    


def run_inference(df, args,ray_init=True):
    if args.nccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "0"
    else:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        
    tmp_dir = os.environ.get("RAY_TMPDIR", f"/data/user_data/{os.getenv('USER','user')}/ray_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ray.init(num_cpus=args.cpus, num_gpus=args.gpus, _temp_dir=tmp_dir)
        
    # if ray_init:
    #     try:
    #         ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    #     except:
    #         ray.init(num_cpus=args.cpus, num_gpus=args.gpus, _temp_dir='/data/user_data/riyaza/ray_tmp')
            
    DataContext.get_current().wait_for_min_actors_s = args.ray_timeout
    ctx = DataContext.get_current()
    # ctx.progress_bar = True
    # ctx.execution_options.verbose_progress = True
    
    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"


    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.engine_cpu_resources, "GPU": args.engine_gpu_resources},
        concurrency=args.concurrency,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            
            "gpu_memory_utilization":0.85,
            "swap_space": 16
            
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
            
        },
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
    )

    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["prompt"]}],
            sampling_params=dict(
                truncate_prompt_tokens=args.truncate_prompt_tokens, 
                max_tokens=args.max_tokens),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )
    ds = ray.data.from_pandas(df).repartition(args.num_blocks)

    ds = processor(ds).materialize()

    output_dir = os.path.join("rag", args.rag_id, "informal_data")
    os.makedirs(output_dir, exist_ok=True)
    ds.write_parquet(f"local://{output_dir}")
    return output_dir


def populate_database(output_dir, rag_dir):
    db_path = os.path.join(rag_dir, "informal_data.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS informal_data")
    
    
    con.execute(f"CREATE TABLE informal_data AS SELECT * FROM read_parquet('{output_dir}/*.parquet')")
    # con.execute(f"CREATE TABLE informal_data AS SELECT * FROM read_parquet(['{output_dir}/*.parquet', '/home/riyaza/eval_improver/improver/rag/final_rag_real/informal_data/*.parquet'])")
    
    
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


def main(args):
    
    MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    df = collect_prompts(os.path.join("rag",args.rag_id), args.max_depth, tokenizer, MAX_PROMPT_TOKENS, args.include_context)
    if len(df) == 0:
        print("No theorems to process")
        exit()
    output_dir = run_inference(df, args)
    populate_database(output_dir, os.path.join("rag",args.rag_id))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Informalize theorems")
    parser.add_argument("rag_id", type=str)
    parser.add_argument("--max_depth", type=int, default=2)
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
    
    
    
    
    
    
    
    
    
    
    
    parser.add_argument(
        "--nccl_p2p",
        type=bool,
        default=False,
        help="Enable NCCL P2P - set to false if nvidia-smi topo -m shows SYS between gpus, or something or another about PCIE? A6000 -> false. (default: False)",
    )
    parser.add_argument(
        "--ray_timeout",
        type=int,
        default=1800,
        help="Ray timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=16,
        help="Number of blocks to repartition the dataset into (default: 16)",
    )
    parser.add_argument(
        "--engine_cpu_resources",
        type=int,
        default=multiprocessing.cpu_count() // available_gpus,
        help="Number of CPU resources for the engine (default: cpus // gpus)",
    )
    parser.add_argument(
        "--engine_gpu_resources",
        type=int,
        default=1,
        help="Number of GPU resources for the engine (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=available_gpus,
        help="Concurrency for the engine (default: gpus)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for the engine (default: 1)",
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        type=bool,
        default=True,
        help="Enable chunked prefill for the engine (default: True)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Maximum model length for the engine (default: 16384)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=65536,
        help="Maximum number of batched tokens for the engine (default: 65536)",
    )
    parser.add_argument(
        "--max_concurrent_batches",
        type=int,
        default=32,
        help="Maximum number of concurrent batches for the engine (default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the engine (default: 32)",
    )
    parser.add_argument(
        "--truncate_prompt_tokens",
        type=int,
        default=16384 - 2048,
        help="Number of prompt tokens to truncate (default: 16384 - 2048)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )
    
    
    
    
    
    
    
    
    args = parser.parse_args()
    
    main(args)