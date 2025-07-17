#!/usr/bin/env python
"""
Synchronous Azure inference script that augments instruction→output training data with *model-generated chain-of-thought* (CoT) traces.

Changes vs prior version:
  • **Batch removed** – only synchronous API calls.
  • **Reasoning model support** – optional use of the *Responses API* with `reasoning` params (effort, summary) for o-series (o4-mini, o3, o1, etc.) or any deployment you choose.
  • Retains Chat Completions path for non-reasoning (GPT-4o, etc.) or when you prefer classic chat semantics.

USAGE EXAMPLES
--------------
# Chat Completions (default)
python azure_infer_training_data_sync.py RUN123 --model gpt-4o-cot-deploy

# Force Responses API (recommended for reasoning deployments; also works w/ chat models)
python azure_infer_training_data_sync.py RUN123 --model o4-mini-deploy --use-responses-api \
    --reasoning-effort medium --reasoning-summary detailed

# Auto-detect: omit --use-responses-api and the script will switch to Responses automatically
# when the deployment name *looks* like an o-series model (starts with "o"). Override with flags.

OUTPUT SCHEMA (per line JSONL)
{
  "instruction": <orig>,
  "original_output": <orig>,
  "cot": <generated_chain_of_thought>,
  "augmented_output": <cot + "\n" + original_output>
}

See inline CLI help (-h) for details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
import pathlib
import typing as t
from dataclasses import dataclass

try:
    import pandas as pd  # optional; only if --csv
except Exception:  # pragma: no cover
    pd = None

# Azure OpenAI unified Python SDK
try:
    from openai import AzureOpenAI
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install --upgrade openai azure-identity pandas") from e

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augment training_data.jsonl with Azure-generated CoT traces (sync only).")
    p.add_argument("run_id", help="Directory key under evals/<run_id>/analysis/training_data.jsonl")
    p.add_argument("--model", required=True, help="Azure *deployment name*.")
    p.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint URL.")
    p.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"), help="Azure OpenAI API key (omit to use Azure AD).")
    p.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"), help="API version (e.g., 2024-10-21, preview).")
    # inference controls
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-concurrent", type=int, default=1, help="Threads for parallel calls.")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-initial", type=float, default=2.0)
    p.add_argument("--retry-max", type=float, default=60.0)
    p.add_argument("--progress-every", type=int, default=25)
    # reasoning opts (Responses API)
    p.add_argument("--use-responses-api", action="store_true", help="Force Responses API (auto when model name starts with 'o' if not set).")
    p.add_argument("--reasoning-effort", choices=["low","medium","high"], default=None, help="Responses API reasoning.effort.")
    p.add_argument("--reasoning-summary", choices=["auto","concise","detailed","none"], default=None, help="Responses API reasoning.summary; 'none' omits.")
    # misc IO
    p.add_argument("--csv", action="store_true", help="Emit CSV alongside JSONL.")
    return p

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a reasoner agent whose purpose is to retroactively construct the reasoning trace of a given output given the input. "
    "You will be provided an instruction and a model_output (tagged). Analyze both and reconstruct the reasoning that produced the model_output, as if you were the original model. "
    "Return ONLY the full chain-of-thought text (no extra prose, no tags). Be detailed and thorough."
)


def build_user_block(instruction: str, output: str) -> str:
    return f"<INSTRUCTION>\n{instruction}\n</INSTRUCTION>\n\n<MODEL_OUTPUT>\n{output}\n</MODEL_OUTPUT>\n"


def build_chat_messages(instruction: str, output: str) -> t.List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_block(instruction, output)},
    ]


def build_responses_input(instruction: str, output: str) -> t.List[dict]:
    # Responses API accepts an "input" list of content items; we pass role blocks for clarity.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_block(instruction, output)},
    ]

# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------

@dataclass
class Example:
    idx: int
    instruction: str
    original_output: str

    def chat_messages(self) -> t.List[dict]:
        return build_chat_messages(self.instruction, self.original_output)

    def responses_input(self) -> t.List[dict]:
        return build_responses_input(self.instruction, self.original_output)


def load_training_examples(run_id: str) -> t.List[Example]:
    path = pathlib.Path("evals") / run_id / "analysis" / "training_data.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    exs: t.List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            exs.append(Example(i, rec["instruction"], rec["output"]))
    return exs

# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def make_client(endpoint: str, api_key: str | None, api_version: str) -> AzureOpenAI:
    if not endpoint:
        raise ValueError("--endpoint or AZURE_OPENAI_ENDPOINT required")
    if api_key:
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
    # Azure AD fallback
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # lazy import
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, azure_ad_token_provider=token_provider)

# ---------------------------------------------------------------------------
# Sync inference helpers
# ---------------------------------------------------------------------------

class RetryableError(Exception):
    pass


def _sleep_with_jitter(base: float, cap: float) -> float:
    import math
    jitter = random.uniform(0, 0.25 * base)
    return min(cap, base * (1.0 + jitter))


def _extract_chat_text(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:  # pragma: no cover
        return ""


def _extract_responses_text(resp) -> str:
    # openai>=1.0 Response object usually exposes .output_text; fall back to concatenation
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                # item.message.content may be list of Text objects; join
                mc = getattr(item, "message", None)
                if mc and getattr(mc, "content", None):
                    for c in mc.content:
                        if c.type == "text":
                            parts.append(c.text)
            elif getattr(item, "type", None) == "output_text":
                parts.append(getattr(item, "text", ""))
        return "".join(parts)
    except Exception:  # pragma: no cover
        return ""


def call_chat_once(client: AzureOpenAI, model: str, messages: t.List[dict], *, temperature: float, top_p: float, seed: int | None):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


def call_responses_once(client: AzureOpenAI, model: str, items: t.List[dict], *, temperature: float, top_p: float, seed: int | None, reasoning_effort: str | None, reasoning_summary: str | None):
    reasoning: dict | None = None
    if reasoning_effort or (reasoning_summary and reasoning_summary != "none"):
        reasoning = {}
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort
        if reasoning_summary and reasoning_summary != "none":
            reasoning["summary"] = reasoning_summary
    return client.responses.create(
        model=model,
        input=items,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        reasoning=reasoning,
    )


def call_with_retries(fn, *, max_retries: int, retry_initial: float, retry_max: float) -> t.Any:
    delay = retry_initial
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # refine by SDK error classes if desired
            if attempt >= max_retries:
                raise
            # honor Retry-After if present
            retry_after = None
            if hasattr(e, "response") and getattr(e.response, "headers", None):  # type: ignore[attr-defined]
                retry_after = e.response.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else _sleep_with_jitter(delay, retry_max)
            time.sleep(sleep_for)
            delay = min(retry_max, delay * 2)
    raise AssertionError("unreachable")


def run_sync_inference(
    client: AzureOpenAI,
    model: str,
    examples: t.List[Example],
    *,
    use_responses_api: bool,
    temperature: float,
    top_p: float,
    seed: int | None,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
    max_concurrent: int,
    max_retries: int,
    retry_initial: float,
    retry_max: float,
    progress_every: int,
) -> t.List[dict]:
    """Iterate examples w/ optional threaded concurrency."""

    def _infer(ex: Example) -> tuple[int, dict]:
        if use_responses_api:
            items = ex.responses_input()
            resp = call_with_retries(lambda: call_responses_once(client, model, items,
                                            temperature=temperature, top_p=top_p, seed=seed,
                                            reasoning_effort=reasoning_effort, reasoning_summary=reasoning_summary),
                                      max_retries=max_retries, retry_initial=retry_initial, retry_max=retry_max)
            cot = _extract_responses_text(resp)
        else:
            msgs = ex.chat_messages()
            resp = call_with_retries(lambda: call_chat_once(client, model, msgs,
                                            temperature=temperature, top_p=top_p, seed=seed),
                                      max_retries=max_retries, retry_initial=retry_initial, retry_max=retry_max)
            cot = _extract_chat_text(resp)
        return ex.idx, _build_record(ex, cot)

    if max_concurrent <= 1:
        out: t.List[dict] = []
        for i, ex in enumerate(examples, 1):
            _, rec = _infer(ex)
            out.append(rec)
            if i % progress_every == 0:
                print(f"Processed {i}/{len(examples)}")
        return out

    import concurrent.futures
    results: t.List[dict | None] = [None] * len(examples)  # type: ignore[list-item]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        for j, (idx, rec) in enumerate(pool.map(_infer, examples), 1):
            results[idx] = rec
            if j % progress_every == 0:
                print(f"Processed {j}/{len(examples)}")
    return t.cast(t.List[dict], results)

# ---------------------------------------------------------------------------
# Record writer
# ---------------------------------------------------------------------------

def _build_record(ex: Example, cot: str) -> dict:
    return {
        "instruction": ex.instruction,
        "original_output": ex.original_output,
        "cot": cot,
        "augmented_output": f"{cot}\n{ex.original_output}" if cot else ex.original_output,
    }


def write_enhanced(run_id: str, records: t.Iterable[dict], *, csv: bool=False) -> pathlib.Path:
    out_path = pathlib.Path("evals") / run_id / "analysis" / "enhanced_training_data.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if csv and pd is not None:
        pd.DataFrame(list(records)).to_csv(out_path.with_suffix(".csv"), index=False)
    return out_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: t.Sequence[str] | None = None) -> int:
    args = get_parser().parse_args(argv)

    if not args.endpoint:
        print("ERROR: no Azure endpoint provided", file=sys.stderr)
        return 2
    if not args.api_key:
        print("INFO: using Azure AD auth (no API key provided)")

    client = make_client(args.endpoint, args.api_key, args.api_version)

    examples = load_training_examples(args.run_id)
    print(f"Loaded {len(examples)} examples.")

    # auto-select responses when model appears to be an o-series reasoning model
    use_responses_api = args.use_responses_api or args.model.lower().startswith("o")

    augmented = run_sync_inference(
        client,
        args.model,
        examples,
        use_responses_api=use_responses_api,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        retry_initial=args.retry_initial,
        retry_max=args.retry_max,
        progress_every=args.progress_every,
    )

    out_path = write_enhanced(args.run_id, augmented, csv=args.csv)
    print(f"Enhanced training data saved to: {out_path}")
    if args.csv and pd is not None:
        print(f"CSV summary saved to: {out_path.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
