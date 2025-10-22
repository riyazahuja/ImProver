import os
import json
import datetime
import argparse
import multiprocessing
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from tqdm import tqdm
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

        api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or "2025-01-01-preview"
        )
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
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
        "API_KEY"
    )  # allow generic API_KEY fallback
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY (or API_KEY) must be set for OpenAI-compatible endpoints."
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


# --- Prompt construction (updated to match inference.py) --------------------------------


def construct_prompt_core(
    config_data: Dict[str, Any],
    item: Dict[str, Any],
    file_context: int,
    context: int,
    rag: int,
    annotation: bool,
    informal: bool,
    goal_state: bool,
    system: bool,
) -> str:
    prompt = ""

    if file_context != 0:
        prompt += f"<FILE_CONTEXT>\n"
        num_deps = (
            len(item["C0_dependencies"])
            if file_context == -1
            else min(file_context, len(item["C0_dependencies"]))
        )
        for context in item["C0_dependencies"][:num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</FILE_CONTEXT>\n\n"

    if context != 0:
        prompt += f"<CONTEXT>\n"
        num_deps = (
            len(item["C1_dependencies"])
            if context == -1
            else min(context, len(item["C1_dependencies"]))
        )
        for context in item["C1_dependencies"][:num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</CONTEXT>\n\n"

    if rag != 0:
        prompt += f"<RETRIEVED>\n"
        num_rag = len(item["rag"]) if rag == -1 else min(rag, len(item["rag"]))
        for rag in item["rag"][:num_rag]:
            prompt += f"<DOC>\n{rag}\n</DOC>\n"
        prompt += f"</RETRIEVED>\n\n"

    if annotation:
        prompt += f"<ANNOTATION>\n{item['annotation']}\n</ANNOTATION>\n\n"

    if informal:
        prompt += f"<INFORMAL>\nTheorem: {item['informal_statement']}\n\nProof:\n{item['informal_proof']}\n</INFORMAL>\n\n"

    if goal_state:
        prompt += f"<GOAL_STATE>\n{item['goal_state']}\n</GOAL_STATE>\n\n"

    if system:
        prompt += "As a reminder: " + config_data["prompts"]["system_prompt"] + "\n"

    prompt += f"\n<CURRENT>\n{item['content_sorry'] if config_data['scoring']['input_sorry'] else item['id']['content']}\n</CURRENT>\n\n"
    # prompt += "<IMPROVED>"

    return prompt


def construct_prompts(
    config_data: Dict[str, Any], data: List[Dict[str, Any]], args
) -> List[Dict[str, Any]]:
    # config_data is metric config data
    # data is the prompt data
    idx = 0
    items: List[Dict[str, Any]] = []

    for item in data:
        if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
            continue

        name = item["id"]["name"]

        prompt = config_data["prompts"]["system_prompt"] + "\n"
        # prompt = ""

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
        if args.informal:
            try:
                prompt += config_data["prompts"]["informal_prompt"] + "\n"
            except:
                prompt += " An informal (natural language) version of the current theorem and proof has also been provided for reference in your reasoning process to better understand and optimize the structure and intuition behind the theorem (wrapped in <INFORMAL>...</INFORMAL>)."

        if args.goal_state:
            prompt += config_data["prompts"]["goal_state_prompt"] + "\n"

        prompt += "\n"

        if args.examples != 0:
            example_data_path = config_data["examples"]["example_data"]

            with open(example_data_path, "r") as f:
                examples_data = json.load(f)

            prompt += f"<EXAMPLES>\n\n"
            num_examples = (
                len(examples_data.items())
                if args.examples == -1
                else min(args.examples, len(examples_data.items()))
            )
            for nameTag, example in list(examples_data.items())[:num_examples]:
                try:
                    ex_prompt = "<EXAMPLE>\n\n"

                    ex_prompt += construct_prompt_core(
                        config_data, example, 0, 0, 0, False, False, False
                    )

                    ex_prompt += f"\n<IMPROVED>\n{example['improved']}\n</IMPROVED>\n\n"
                    ex_prompt += f"</EXAMPLE>\n\n"
                    prompt += ex_prompt
                except:
                    pass
            prompt += f"</EXAMPLES>\n\n"

        prompt += construct_prompt_core(
            config_data,
            item,
            args.file_context,
            args.context,
            args.rag,
            args.annotation,
            args.informal,
            args.goal_state,
            True,
        )

        data = {
            "decl": name,
            "decl_idx": idx,
            "raw_prompt": prompt,
        }
        items.append(data)
        idx += 1
    return items


# --- Asynchronous inference with rate limiting -------------------------------


class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [
                req_time for req_time in self.requests if now - req_time < 60
            ]

            if len(self.requests) >= self.max_requests_per_minute:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                await asyncio.sleep(sleep_time)
                return await self.acquire()

            self.requests.append(now)


async def _chat_complete_async(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    is_azure: bool = False,
) -> str:
    """
    Make a single asynchronous Chat Completions request and return the assistant message text.
    """
    await rate_limiter.acquire()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        # "temperature": 0.3,
        # "top_p": 0.9,
        # "repetition_penalty": 1.05,
        # "stop": ["</IMPROVED>"]
    }
    # data = {
    #     "input": [{"role": "user", "content": prompt}],
    #     "max_output_tokens": max_tokens,
    #     "model": model,
    # data = {
    #     "messages": [{"role": "user", "content": prompt}],
    #     "max_completion_tokens": max_tokens,
    #     # "temperature": 0.3,
    #     # "top_p": 0.9,
    #     # "repetition_penalty": 1.05,
    #     # "stop": ["</IMPROVED>"]
    # }

    # data = {
    #     "input": [{"role": "user", "content": prompt}],
    #     "max_output_tokens": max_tokens,
    #     "model": model,
    #     # "temperature": 0.3,
    #     # "top_p": 0.9,
    #     # "repetition_penalty": 1.05,
    #     # "stop": ["</IMPROVED>"]
    # }


    # For Azure, the model is in the URL, not in the request body
    if not is_azure:
        data["model"] = model

    try:
        # For Azure, the URL already includes the full path
        if is_azure:
            url = base_url
        else:
            url = f"{base_url}/chat/completions"
        # print(
        #     f"Making request to \n{url}\n with headers \n{headers}\n and data \n{data}\n\n"
        #     + "=" * 50
        # )

        async with session.post(url, headers=headers, json=data) as response:

            if response.status == 429:  # Rate limit hit
                retry_after = int(response.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                return await _chat_complete_async(
                    session,
                    base_url,
                    api_key,
                    model,
                    prompt,
                    max_tokens,
                    rate_limiter,
                    is_azure,
                )

            if response.status != 200:
                error_text = await response.text()
                print(f"API Error {response.status}: {error_text}")
                print(f"Request URL: {url}")
                print(f"Request data: {data}")

            response.raise_for_status()
            result = await response.json()
            print(f"Response status: {response.status}")
            print(f"Response body:\n{result}")
            print("=" * 50)

            # Handle both OpenAI and common OpenAI-compatible response shapes
            try:
                return result["choices"][0]["message"]["content"] or ""
            except Exception:
                # Fallback: some proxies return `text` in choices for chat
                try:
                    return result["choices"][0]["text"] or ""
                except Exception:
                    try:
                        return result["output"][-1]["content"][-1]["text"] or ""
                    except Exception:
                        return ""

    except Exception as e:
        # Retry once after a short backoff
        await asyncio.sleep(2.0)
        try:
            # For Azure, the URL already includes the full path
            if is_azure:
                url = base_url
            else:
                url = f"{base_url}/chat/completions"

            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()

                try:
                    return result["choices"][0]["message"]["content"] or ""
                except Exception:
                    try:
                        return result["choices"][0]["text"] or ""
                    except Exception:
                        try:
                            return result["output"][-1]["content"][-1]["text"] or ""
                        except Exception:
                            return (
                                f"[ERROR] {type(e).__name__}: {e}\n\n{result.__dict__}"
                            )
        except Exception as e2:
            return f"[ERROR] {type(e2).__name__}: {e2}\n\n{result.__dict__}"


async def run_inference_async(df: pd.DataFrame, args) -> str:
    """
    Asynchronous inference that processes prompts concurrently with rate limiting.
    """
    # Get API configuration - handle both Azure and standard OpenAI
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if azure_key and azure_endpoint:
        # Azure OpenAI configuration
        api_key = azure_key
        base_url = azure_endpoint
        # For Azure, we need to use the deployment name as the model
        model_name = args.model
        is_azure = True
    else:
        # Standard OpenAI configuration
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY (or API_KEY) must be set for OpenAI-compatible endpoints."
            )

        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model_name = args.model
        is_azure = False

    # Duplicate rows for best-of-n behavior (preserve original semantics)
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)
    df2 = pd.concat(df2_parts, ignore_index=True)

    size = len(df2)
    print(f"Size of dataset: {size} prompts")

    # Configure rate limiting (adjust based on your API limits)
    rate_limiter = RateLimiter(
        max_requests_per_minute=(
            args.server_rate_limit if hasattr(args, "server_rate_limit") else 60
        )
    )

    outputs: List[Dict[str, Any]] = []

    # Create progress bar
    pbar = tqdm(total=size, desc="Processing prompts", unit="prompt")

    async def process_prompt(row_idx: int, row) -> Dict[str, Any]:
        raw_prompt: str = getattr(row, "raw_prompt")

        # # Optional prompt truncation (approximate)
        # if args.truncate_prompt_tokens is not None and args.truncate_prompt_tokens > 0:
        #     approx_chars = int(args.truncate_prompt_tokens) * 4
        #     if len(raw_prompt) > approx_chars:
        #         raw_prompt = raw_prompt[-approx_chars:]

        answer_text = await _chat_complete_async(
            session,
            base_url,
            api_key,
            model_name,
            raw_prompt,
            args.max_tokens,
            rate_limiter,
            is_azure,
        )
        print(f"Prompt {row_idx} completed.")
        print(f"Answer:\n{answer_text}")
        print("#" * 80)

        result = {
            "module": getattr(row, "module", None),
            "decl": getattr(row, "decl"),
            "decl_idx": getattr(row, "decl_idx"),
            "prompt_idx": getattr(row, "prompt_idx"),
            "raw_prompt": raw_prompt,
            "generated_text": answer_text,  # keep for compatibility
            "answer": answer_text,  # this is what downstream expects postprocess to produce
        }

        pbar.update(1)
        return result

    # Process prompts with controlled concurrency
    semaphore = asyncio.Semaphore(
        args.server_concurrency if hasattr(args, "server_concurrency") else 10
    )

    async def process_with_semaphore(row_idx: int, row):
        async with semaphore:
            return await process_prompt(row_idx, row)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for row_idx, row in enumerate(df2.itertuples(index=False), start=1):
            task = process_with_semaphore(row_idx, row)
            tasks.append(task)

        # Process in batches to avoid overwhelming the system
        batch_size = (
            args.server_concurrency if hasattr(args, "server_concurrency") else 50
        )
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    outputs.append(
                        {
                            "module": None,
                            "decl": "ERROR",
                            "decl_idx": 0,
                            "prompt_idx": 0,
                            "raw_prompt": "",
                            "generated_text": f"[ERROR] {type(result).__name__}: {result}\n\n{result.__dict__}",
                            "answer": f"[ERROR] {type(result).__name__}: {result}\n\n{result.__dict__}",
                        }
                    )
                else:
                    outputs.append(result)

    pbar.close()

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
        DROP TABLE IF EXISTS run_data;
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS run_data AS
        SELECT * FROM read_parquet('{output_path}/*.parquet');
        """
    )
    con.close()

    return run_output_dir


def run_inference(df: pd.DataFrame, args, ray_init: bool = False) -> str:
    """
    Wrapper to run the async inference function.
    """
    return asyncio.run(run_inference_async(df, args))


# --- Main CLI (updated to match inference.py) -----------------------------------


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

    # return path for potential downstream consumers
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for ImProver (OpenAI-compatible, async)"
    )

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

    # Updated arguments to match inference.py
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=AVAILABLE_GPUS,
        help="Number of GPUs (default: all available)",
    )
    parser.add_argument("--n", type=int, default=1, help="Best-of-n value (default: 1)")
    parser.add_argument(
        "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    )
    parser.add_argument(
        "--informal", type=bool, default=False, help="Informal? (default: False)"
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
        "--rag",
        type=int,
        default=0,
        help="Number of RAG retrievals (default: 0, max: 10)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0, -1 for all)",
    )

    # New async-specific arguments
    parser.add_argument(
        "--server_concurrency",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--server_rate_limit",
        type=int,
        default=60,
        help="Maximum requests per minute for rate limiting (default: 60)",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )

    args = parser.parse_args()
    main(args)
