
import os
import json
import datetime
import argparse
import multiprocessing
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import duckdb

# Optional: torch is used only to detect available GPUs to keep CLI parity with the original script.
try:
    import torch  # noqa: F401
    AVAILABLE_GPUS = torch.cuda.device_count()
except Exception:
    AVAILABLE_GPUS = 0

# --- OpenAI / Azure OpenAI client setup --------------------------------------

# We intentionally use the official OpenAI Python SDK so this script can talk to:
#   * OpenAI-hosted endpoints (default)
#   * Any OpenAI-compatible endpoint (e.g., a vLLM server) by setting OPENAI_BASE_URL
#   * Azure OpenAI by setting AZURE_OPENAI_* env vars (uses AzureOpenAI client)
#
# Env vars this script understands:
#   OPENAI_API_KEY          -> API key for OpenAI-compatible endpoints (including vLLM if required)
#   OPENAI_BASE_URL         -> Base URL for an OpenAI-compatible endpoint, e.g. http://localhost:8000/v1
#   (Azure) AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
#
# NOTE: We intentionally call the Chat Completions API because it is the most widely supported
# across OpenAI-compatible servers (including vLLM) at the time of writing.

def _get_openai_client():
    """
    Return an instantiated OpenAI or AzureOpenAI client based on environment variables.
    We avoid adding new CLI flags to preserve the original user-facing interface.
    """
    # Prefer Azure client if Azure-specific env vars are present.
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Azure OpenAI environment variables are set but the OpenAI SDK is missing Azure support. "
                "Please install/upgrade `openai` >= 1.0."
            ) from e

        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-06-01"
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client

    # Otherwise, default to the standard OpenAI client, optionally honoring a custom base URL.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI Python SDK is required. Install with `pip install openai`."
        ) from e

    base_url = os.getenv("OPENAI_BASE_URL")  # e.g., http://localhost:8000/v1 for vLLM
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")  # allow generic API_KEY fallback
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or API_KEY) must be set for OpenAI-compatible endpoints.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


# --- Prompt construction (unchanged semantics) --------------------------------

def construct_prompt_core(config_data: Dict[str, Any], item: Dict[str, Any], args) -> str:
    prompt = ""

    if args.file_context != 0:
        prompt += f"<FILE_CONTEXT>\n"
        num_deps = len(item["C0_dependencies"]) if args.file_context == -1 else min(args.file_context, len(item["C0_dependencies"]))
        for context in item["C0_dependencies"][: num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</FILE_CONTEXT>\n\n"

    if args.context != 0:
        prompt += f"<CONTEXT>\n"
        num_deps = len(item["C1_dependencies"]) if args.context == -1 else min(args.context, len(item["C1_dependencies"]))
        for context in item["C1_dependencies"][: num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</CONTEXT>\n\n"

    if args.rag != 0:
        prompt += f"<RAG>\n"
        num_rag = len(item["rag"]) if args.rag == -1 else min(args.rag, len(item["rag"]))
        for rag in item["rag"][: num_rag]:
            prompt += f"<DOC>\n{rag}\n</DOC>\n"
        prompt += f"</RAG>\n\n"

    if args.annotation:
        prompt += f"<ANNOTATION>\n{item['annotation']}\n</ANNOTATION>\n\n"

    if args.goal_state:
        prompt += f"<GOAL_STATE>\n{item['goal_state']}\n</GOAL_STATE>\n\n"

    # Preserve the exact behavior around input_sorry
    if config_data["scoring"]["input_sorry"]:
        current = item["content_sorry"]
    else:
        current = item["id"]["content"]

    # Optional, approximate prompt truncation (to preserve CLI arg semantics)
    # We cannot tokenize reliably without extra deps; approximate 1 token ~= 4 chars.
    if args.truncate_prompt_tokens is not None and args.truncate_prompt_tokens > 0:
        approx_chars = int(args.truncate_prompt_tokens) * 4
        if len(current) > approx_chars:
            current = current[-approx_chars:]

    prompt += f"\n<CURRENT>\n{current}\n</CURRENT>\n\n"
    prompt += "<IMPROVED>"

    return prompt


def construct_prompts(config_data: Dict[str, Any], data: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    # config_data is metric config data
    # data is the prompt data
    idx = 0
    items: List[Dict[str, Any]] = []

    for item in data:
        if item['id']['isExtracted'] or len(item['id']['errorMsgs']) != 0:
            continue

        name = item["id"]["name"]

        prompt = config_data["prompts"]["system_prompt"] + "\n"

        if args.examples != 0:
            prompt += config_data["prompts"]["example_prompt"] + "\n"

        if args.context != 0:
            prompt += config_data["prompts"]["context_prompt"] + "\n"

        if args.file_context != 0:
            prompt += config_data["prompts"]["file_context_prompt"] + "\n"

        if args.rag != 0:
            prompt += config_data["prompts"]["rag_prompt"] + "\n"

        if args.annotation:
            prompt += config_data["prompts"]["annotation_prompt"] + "\n"

        if args.goal_state:
            prompt += config_data["prompts"]["goal_state_prompt"] + "\n"

        prompt += "\n"

        if args.examples != 0:
            example_data_path = config_data["examples"]["example_data"]
            with open(example_data_path, "r") as f:
                examples_data = json.load(f)

            prompt += f"<EXAMPLES>\n\n"
            num_examples = len(examples_data.items()) if args.examples == -1 else min(args.examples, len(examples_data.items()))
            for nameTag, example in list(examples_data.items())[: num_examples]:
                try:
                    ex_prompt = "<EXAMPLE>\n\n"
                    ex_prompt += construct_prompt_core(config_data, example, args)
                    ex_prompt += f"\n{example['improved']}\n</IMPROVED>\n\n"
                    ex_prompt += f"</EXAMPLE>\n\n"
                    prompt += ex_prompt
                except Exception:
                    # swallow malformed example entries to match original behavior
                    pass
            prompt += f"</EXAMPLES>\n\n"

        prompt += construct_prompt_core(config_data, item, args)

        items.append({
            "decl": name,
            "decl_idx": idx,
            "raw_prompt": prompt,
        })
        idx += 1
    return items


# --- Inference (non-batched, OpenAI-compatible) -------------------------------

def _chat_complete(client, model: str, prompt: str, max_tokens: int) -> str:
    """
    Make a single Chat Completions request and return the assistant message text.
    Works with OpenAI, Azure OpenAI, and OpenAI-compatible servers (e.g., vLLM) via base_url.
    """
    # We stick to the chat.completions API for maximum compatibility across providers.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    # Handle both OpenAI and common OpenAI-compatible response shapes.
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        # Fallback: some proxies return `text` in choices for chat
        try:
            return resp.choices[0].text or ""
        except Exception:
            return ""


def run_inference(df: pd.DataFrame, args, ray_init: bool = False) -> str:
    """
    Non-batched inference that iterates row-by-row and calls an OpenAI-compatible endpoint.
    Preserves the user-facing interface and on-disk outputs of the original script.
    """
    client = _get_openai_client()

    # Duplicate rows for best-of-n behavior (preserve original semantics)
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)
    df2 = pd.concat(df2_parts, ignore_index=True)

    size = len(df2)
    print(f"Size of dataset: {size} prompts")

    outputs: List[Dict[str, Any]] = []
    last_print = time.time()

    for row_idx, row in enumerate(df2.itertuples(index=False), start=1):
        raw_prompt: str = getattr(row, "raw_prompt")
        try:
            answer_text = _chat_complete(client, model=args.model, prompt=raw_prompt, max_tokens=args.max_tokens)
        except Exception as e:
            # Retry once after a short backoff to tolerate transient errors.
            time.sleep(2.0)
            try:
                answer_text = _chat_complete(client, model=args.model, prompt=raw_prompt, max_tokens=args.max_tokens)
            except Exception as e2:
                answer_text = f"[ERROR] {type(e2).__name__}: {e2}"

        outputs.append({
            "module": getattr(row, "module", None),
            "decl": getattr(row, "decl"),
            "decl_idx": getattr(row, "decl_idx"),
            "prompt_idx": getattr(row, "prompt_idx"),
            "raw_prompt": raw_prompt,
            "generated_text": answer_text,  # keep for compatibility
            "answer": answer_text,          # this is what downstream expects postprocess to produce
        })

        # Light-weight progress indicator without adding new deps.
        if time.time() - last_print > 5:
            print(f"Processed {row_idx}/{size} prompts...")
            last_print = time.time()

    run_output_dir = os.path.join("evals", args.run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    output_path = os.path.join(run_output_dir, "data")

    # Save run config
    config = vars(args)
    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Write a Parquet file in the same folder structure as the original code.
    os.makedirs(output_path, exist_ok=True)
    df_out = pd.DataFrame(outputs)

    # Try to write via pandas/pyarrow; if unavailable, fall back to DuckDB COPY.
    parquet_file = os.path.join(output_path, "part-00000.parquet")
    try:
        df_out.to_parquet(parquet_file, index=False)
    except Exception:
        # Fallback via DuckDB COPY
        con_tmp = duckdb.connect()
        con_tmp.register("df_out", df_out)
        con_tmp.execute(f"COPY df_out TO '{parquet_file}' (FORMAT PARQUET)")
        con_tmp.unregister("df_out")
        con_tmp.close()

    # Build DuckDB database aggregating the Parquet lake (same table name as original).
    con = duckdb.connect(os.path.join(run_output_dir, "data.duckdb"))
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS run_data AS
        SELECT * FROM read_parquet('{output_path}/*.parquet');
        """
    )
    con.close()

    return run_output_dir


# --- Main CLI (preserve original interface) -----------------------------------

def main(args):
    with open(args.dataset_path, "r") as f:
        all_data = json.load(f)
        dataset = all_data[args.split]

    files_to_process: List[str] = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    prompt_root = os.path.join("prompts", args.prompt_id)
    metric_root = os.path.join("metrics", args.metric)
    config_path = os.path.join(metric_root, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)

    df = pd.DataFrame(columns=["module", "decl", "decl_idx", "raw_prompt"])
    for file_info in files_to_process:
        file = file_info if type(file_info) is str else file_info["file"]

        file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
        module = file.replace(".lean", "").replace("/", ".")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(config_data, data_raw, args)
            print(f"Processing {file_path} with {len(prompt_data)} prompts")

            for item in prompt_data:
                df.loc[len(df)] = [
                    module,
                    item["decl"],
                    item["decl_idx"],
                    item["raw_prompt"],
                ]

    # returns the path to the directory containing run metadata and the parquet lake
    output_path = run_inference(df, args)

    # No ray.shutdown() here; we are not using ray in this implementation.

    # return path for potential downstream consumers
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for ImProver (OpenAI-compatible, non-batched)")

    # Positional args (unchanged)
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")

    # Run ID (unchanged)
    parser.add_argument(
        "--run_id",
        type=str,
        default="RUN_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run ID to use for evaluation (default: run_<timestamp>)",
    )

    # MODEL is now the 'model' string sent to the OpenAI-compatible API (e.g., Azure deployment name or vLLM served name)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",  # sensible default; override with your deployment name / HF model on vLLM
        help="Model/deployment name to send to the OpenAI-compatible endpoint",
    )

    # Split (unchanged)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )

    # We keep the following arguments to preserve CLI compatibility,
    # even if they are not used by this non-batched, API-based implementation.
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="(unused) Number of CPUs (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=AVAILABLE_GPUS,
        help="(unused) Number of GPUs (kept for CLI compatibility)",
    )
    parser.add_argument("--n", type=int, default=1, help="Best-of-n value (default: 1)")
    parser.add_argument(
        "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    )
    parser.add_argument(
        "--goal_state", type=bool, default=False, help="Goal state? (default: False)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context retrievals (default: 0, -1 for all)",
    )
    parser.add_argument(
        "--file_context",
        type=int,
        default=0,
        help="Number of file context items (default: 0, -1 for all)",
    )
    parser.add_argument(
        "--rag", type=int, default=0, help="Number of RAG retrievals (default: 0, max: 10)"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0, -1 for all)",
    )

    # Legacy/unused arguments retained for CLI parity
    parser.add_argument("--nccl_p2p", type=bool, default=False, help="(unused)")
    parser.add_argument("--ray_timeout", type=int, default=1800, help="(unused)")
    parser.add_argument("--num_blocks", type=int, default=16, help="(unused)")
    parser.add_argument("--engine_cpu_resources", type=int, default=1, help="(unused)")
    parser.add_argument("--engine_gpu_resources", type=int, default=1, help="(unused)")
    parser.add_argument("--concurrency", type=int, default=1, help="(unused)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="(unused)")
    parser.add_argument("--enable_chunked_prefill", type=bool, default=True, help="(unused)")
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="(unused) Maximum model length (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=65536,
        help="(unused) Kept for CLI compatibility",
    )
    parser.add_argument(
        "--max_concurrent_batches",
        type=int,
        default=32,
        help="(unused) Kept for CLI compatibility",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="(unused) Kept for CLI compatibility",
    )
    parser.add_argument(
        "--truncate_prompt_tokens",
        type=int,
        default=16384 - 2048,
        help="Approximate prompt truncation budget in tokens (default: 16384 - 2048)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )

    args = parser.parse_args()
    main(args)
