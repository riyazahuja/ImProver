import ray
from packaging.version import Version
from ray.data.llm import (
    build_llm_processor,
    ProcessorConfig,
    vLLMEngineProcessorConfig,
    SGLangEngineProcessorConfig,
    HttpRequestProcessorConfig,
)

from ray.llm._internal.batch.processor.base import (
    ProcessorBuilder,
)
from ray.llm._internal.batch.processor.vllm_engine_proc import (
    build_vllm_engine_processor,
)
from ray.llm._internal.batch.processor.sglang_engine_proc import (
    build_sglang_engine_processor,
)
from ray.llm._internal.batch.processor.http_request_proc import (
    build_http_request_processor,
)

from ray.data import DataContext, Dataset
import subprocess
import os
import torch
import pandas as pd
import json
import datetime
import multiprocessing
import argparse
import duckdb
from transformers import AutoTokenizer
from pydantic import Field, BaseModel
from typing import Optional, Dict, Union, Tuple
import pyarrow as pa, pyarrow.parquet as pq
from uuid import uuid4
import re
import numpy as np
import shutil
import math


class Metric(BaseModel):
    name: str = Field(description="The name of the metric.")
    config: Optional[Dict] = Field(
        default=None, description="The configuration for the metric."
    )
    score_fn: Optional[str] = Field(
        default=None, description="The scoring function for the metric."
    )
    sorry_ok: Optional[bool] = Field(
        default=None,
        description="Whether to allow sorry messages in the generated text.",
    )
    correctness_condition: Optional[str] = Field(
        default=None, description="The correctness condition for the metric."
    )
    minmax: Optional[str] = Field(
        default=None, description="Whether to minimize or maximize the metric."
    )
    input_sorry: Optional[bool] = Field(
        default=None, description="Whether to input sorry messages into the metric."
    )

    # Optional fields for storing additional data

    example_file: Optional[str] = Field(
        default=None, description="Path to the example file."
    )
    example_data: Optional[str] = Field(
        default=None, description="Path to the example data."
    )

    system_prompt: Optional[str] = Field(
        default=None, description="The system prompt for the metric."
    )
    annotation_prompt: Optional[str] = Field(
        default=None, description="The annotation prompt for the metric."
    )
    context_prompt: Optional[str] = Field(
        default=None, description="The context prompt for the metric."
    )
    rag_prompt: Optional[str] = Field(
        default=None, description="The RAG prompt for the metric."
    )
    example_prompt: Optional[str] = Field(
        default=None, description="The example prompt for the metric."
    )

    llm_metric: Optional[bool] = Field(
        default=None, description="Whether the metric is an LLM metric."
    )
    metric_model: Optional[str] = Field(
        default=None, description="The model to use for the metric."
    )
    rubric: Optional[Dict] = Field(
        default=None, description="The rubric for the metric."
    )

    def __init__(self, name: str, **kwargs):
        metric_config_path = os.path.join("metrics", name, "config.json")
        if not os.path.exists(metric_config_path):
            raise FileNotFoundError(
                f"Metric config file not found at {metric_config_path}"
            )
        with open(metric_config_path, "r") as f:
            config = json.load(f)

        # Initialize the Pydantic model with all the loaded values
        super().__init__(
            name=name,
            config=config,
            score_fn=config["scoring"].get("score_fn", None),
            sorry_ok=config["scoring"].get("sorry_ok", None),
            correctness_condition=config["scoring"].get("correctness_condition", None),
            minmax=config["scoring"].get("minmax", None),
            input_sorry=config["scoring"].get("input_sorry", None),
            example_file=config["examples"].get("example_file", None),
            example_data=config["examples"].get("example_data", None),
            system_prompt=config["prompts"].get("system_prompt", None),
            annotation_prompt=config["prompts"].get("annotation_prompt", None),
            context_prompt=config["prompts"].get("context_prompt", None),
            rag_prompt=config["prompts"].get("rag_prompt", None),
            example_prompt=config["prompts"].get("example_prompt", None),
            metric_model=config["llm"].get("metric_model", None),
            rubric=config["llm"].get("rubric", None),
            llm_metric=config["llm"].get("llm_metric", None),
            **kwargs,
        )


class ImProverDataset(BaseModel):
    dataset_path: str
    prompt_id: str
    split: Optional[str] = None
    files_to_process: list = Field(default_factory=list)
    theorems_to_process: dict = Field(default_factory=dict)

    def __init__(
        self, dataset_path: str, prompt_id: str, split: Optional[str] = None, **data
    ):
        super().__init__(
            dataset_path=dataset_path, prompt_id=prompt_id, split=split, **data
        )
        self._load_files_to_process()

    def _load_files_to_process(self):
        with open(self.dataset_path, "r") as f:
            all_data = json.load(f)

        if self.split:
            if self.split in all_data:
                dataset_split = all_data[self.split]
            else:
                raise ValueError(f"Split '{self.split}' not found in dataset file.")
        else:
            # If no split is specified, combine all splits.
            dataset_split = {}
            for split_name, split_data in all_data.items():
                for repo, files in split_data.items():
                    if repo not in dataset_split:
                        dataset_split[repo] = []
                    dataset_split[repo].extend(files)

        for repo, files in dataset_split.items():
            for file_info in files:
                if isinstance(file_info, str):
                    self.files_to_process.append(file_info)
                elif isinstance(file_info, dict) and "file" in file_info:
                    file_path = file_info["file"]
                    self.files_to_process.append(file_path)
                    if "theorems" in file_info:
                        if file_path not in self.theorems_to_process:
                            self.theorems_to_process[file_path] = []
                        self.theorems_to_process[file_path].extend(
                            file_info["theorems"]
                        )

        self.files_to_process = sorted(list(set(self.files_to_process)))

    def get_prompts(
        self,
        metric: Metric,
        examples=0,
        context=0,
        file_context=0,
        rag=0,
        annotation=False,
        goal_state=False,
    ) -> pd.DataFrame:
        prompt_root = os.path.join("prompts", self.prompt_id)

        all_prompt_data = []

        config_data = metric.config

        for file in self.files_to_process:
            file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
            module = file.replace(".lean", "").replace("/", ".")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data_raw = json.load(f)

                theorems_in_file = self.theorems_to_process.get(file)
                if theorems_in_file:
                    data_raw = [
                        item
                        for item in data_raw
                        if item["id"]["name"] in theorems_in_file
                    ]

                prompt_data = self._construct_prompts(
                    config_data,
                    data_raw,
                    examples,
                    context,
                    file_context,
                    rag,
                    annotation,
                    goal_state,
                )
                print(f"Processing {file_path} with {len(prompt_data)} prompts")

                for item in prompt_data:
                    all_prompt_data.append(
                        {
                            "module": module,
                            "decl": item["decl"],
                            "decl_idx": item["decl_idx"],
                            "raw_prompt": item["raw_prompt"],
                        }
                    )
        return pd.DataFrame(all_prompt_data)

    def _construct_prompt_core(
        self,
        config_data,
        item,
        file_context,
        context,
        rag,
        annotation,
        goal_state,
        system,
    ):
        prompt = ""

        if file_context != 0:
            prompt += f"<FILE_CONTEXT>\n"
            num_deps = (
                len(item["C0_dependencies"])
                if file_context == -1
                else min(file_context, len(item["C0_dependencies"]))
            )
            for c in item["C0_dependencies"][:num_deps]:
                prompt += f"<ITEM>\n--name={c['name']}\n--type={c['kind']}\n{c['content']}\n</ITEM>\n"
            prompt += f"</FILE_CONTEXT>\n\n"

        if context != 0:
            prompt += f"<CONTEXT>\n"
            num_deps = (
                len(item["C1_dependencies"])
                if context == -1
                else min(context, len(item["C1_dependencies"]))
            )
            for c in item["C1_dependencies"][:num_deps]:
                prompt += f"<ITEM>\n--name={c['name']}\n--type={c['kind']}\n{c['content']}\n</ITEM>\n"
            prompt += f"</CONTEXT>\n\n"

        if rag != 0:
            prompt += f"<RETRIEVED>\n"
            num_rag = len(item["rag"]) if rag == -1 else min(rag, len(item["rag"]))
            for r in item["rag"][:num_rag]:
                prompt += f"<DOC>\n{r}\n</DOC>\n"
            prompt += f"</RETRIEVED>\n\n"

        if annotation:
            prompt += f"<ANNOTATION>\n{item['annotation']}\n</ANNOTATION>\n\n"

        if goal_state:
            prompt += f"<GOAL_STATE>\n{item['goal_state']}\n</GOAL_STATE>\n\n"

        if system:
            prompt += "As a reminder: " + config_data["prompts"]["system_prompt"] + "\n"

        prompt += f"\n<CURRENT>\n{item['content_sorry'] if config_data['scoring']['input_sorry'] else item['id']['content']}\n</CURRENT>\n\n"
        return prompt

    def _construct_prompts(
        self,
        config_data,
        data,
        examples=0,
        context=0,
        file_context=0,
        rag=0,
        annotation=False,
        goal_state=False,
    ):
        idx = 0
        items = []

        for item in data:
            if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
                continue

            name = item["id"]["name"]
            prompt = config_data["prompts"]["system_prompt"] + "\n"

            if examples != 0:
                prompt += config_data["prompts"]["example_prompt"] + "\n"
            if context != 0:
                prompt += config_data["prompts"]["context_prompt"] + "\n"
            if file_context != 0:
                prompt += config_data["prompts"]["file_context_prompt"] + "\n"
            if rag != 0:
                prompt += config_data["prompts"]["rag_prompt"] + "\n"
            if annotation:
                prompt += config_data["prompts"]["annotation_prompt"] + "\n"
            if goal_state:
                prompt += config_data["prompts"]["goal_state_prompt"] + "\n"
            prompt += "\n"

            if examples != 0:
                example_data_path = config_data["examples"]["example_data"]
                with open(example_data_path, "r") as f:
                    examples_data = json.load(f)
                prompt += f"<EXAMPLES>\n\n"
                num_examples = (
                    len(examples_data.items())
                    if examples == -1
                    else min(examples, len(examples_data.items()))
                )
                for nameTag, example in list(examples_data.items())[:num_examples]:
                    try:
                        ex_prompt = "<EXAMPLE>\n\n"
                        ex_prompt += self._construct_prompt_core(
                            config_data, example, 0, 0, 0, False, False, False
                        )
                        ex_prompt += (
                            f"\n<IMPROVED>\n{example['improved']}\n</IMPROVED>\n\n"
                        )
                        ex_prompt += f"</EXAMPLE>\n\n"
                        prompt += ex_prompt
                    except:
                        pass
                prompt += f"</EXAMPLES>\n\n"

            prompt += self._construct_prompt_core(
                config_data,
                item,
                file_context,
                context,
                rag,
                annotation,
                goal_state,
                True,
            )

            data_item = {
                "decl": name,
                "decl_idx": idx,
                "raw_prompt": prompt,
            }
            items.append(data_item)
            idx += 1
        return items

    def to_ray_dataset(
        self, prompts_df: pd.DataFrame, partition_by_module: bool = False
    ):
        ds = ray.data.from_pandas(prompts_df)
        if partition_by_module:
            ds = ds.groupby("module").map_groups(lambda pdf: pdf, batch_format="pandas")
        return ds


class ImProverRunConfig(BaseModel):
    """Configuration for running ImProver."""

    metric: Metric = Field(
        default=None,
        description="The metric to use for evaluation. Should be an instance of Metric class.",
    )
    dataset: ImProverDataset = Field(
        default=None,
        description="The dataset to use for the run. Should be an instance of ImProverDataset class.",
    )

    run_id: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Unique identifier for the run, defaults to current timestamp.",
    )

    annotation: bool = Field(
        default=False, description="Whether to enable annotation. Defaults to False."
    )
    goal_state: bool = Field(
        default=False, description="Whether to enable goal state. Defaults to False."
    )
    context: int = Field(
        default=0,
        description="The number of context items to include in the prompt. Defaults to 0 (no context). Use -1 for all.",
    )
    file_context: int = Field(
        default=0,
        description="The number of file context items to include in the prompt. Defaults to 0 (no file context). Use -1 for all.",
    )
    rag: int = Field(
        default=0,
        description="The number of RAG items to include in the prompt. Defaults to 0 (no RAG). Use -1 for all.",
    )
    examples: int = Field(
        default=0,
        description="The number of examples to include in the prompt. Defaults to 0 (no examples). Use -1 for all.",
    )

    thinking: str = Field(
        default="none",
        description="The thinking strategy to use. Defaults to 'none'. Options: 'none', 'raw', 'synthetic'.",
    )

    def get_prompts(self) -> pd.DataFrame:
        """Get prompts from the dataset based on the metric configuration."""
        if not self.dataset or not self.metric:
            raise ValueError("Dataset and Metric must be set before getting prompts.")

        return self.dataset.get_prompts(
            metric=self.metric,
            examples=self.examples,
            context=self.context,
            file_context=self.file_context,
            rag=self.rag,
            annotation=self.annotation,
            goal_state=self.goal_state,
        )

    def to_ray_dataset(self, partition_by_module: bool = False) -> Dataset:
        """Convert the prompts DataFrame to a Ray Dataset."""
        if not self.dataset:
            raise ValueError("Dataset must be set before converting to Ray Dataset.")

        return self.dataset.to_ray_dataset(
            self.get_prompts(), partition_by_module=partition_by_module
        )


class ImProverProcessorConfig(ProcessorConfig):
    """The ImProver processor configuration."""

    max_block_size: int = Field(
        default=1024,
        description="The maximum block size for Ray Dataset partitions. Defaults to 1024.",
    )
    ray_temp_dir: Optional[str] = Field(
        default=None,
        description="The temporary directory for Ray. Defaults to None, meaning Ray will use the default temp directory.",
    )
    cpus: int = Field(
        default=multiprocessing.cpu_count(),
        description="The number of CPUs to use. Defaults to the number of available CPUs.",
    )
    gpus: int = Field(
        default=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        description="The number of GPUs to use. Defaults to the number of available GPUs.",
    )
    n: int = Field(
        default=1,
        description="The number of samples per prompt for generation. Defaults to 1. This is depreciated in vLLM >=0.10.0",
    )
    m: int = Field(
        default=1,
        description="The maximum number of iterations to run per prompt. Defaults to 1.",
    )

    nccl_p2p: bool = Field(
        default=False,
        description="Whether to enable NCCL P2P. Set to false if nvidia-smi topo -m shows SYS between gpus, or something or another about PCIE? A6000 -> false. Defaults to False.",
    )

    ray_timeout: int = Field(
        default=1800,
        description="The Ray timeout in seconds. Defaults to 1800.",
    )

    sampling_params: Dict = Field(
        default_factory=dict,
        description="(static) Sampling parameters for the LLM.",
    )

    def preprocessor(self, row):
        return dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=self.sampling_params,
        )


def _merge_sp(base, extra):
    out = dict(base or {})
    out.update(extra or {})
    return out


class vLLMEngineImProverProcessorConfig(
    ImProverProcessorConfig, vLLMEngineProcessorConfig
):
    def preprocessor(self, row):
        # Truncate the prompt to fit within the model's context window
        prompt = row["raw_prompt"]
        max_model_len = self.engine_kwargs.get("max_model_len")
        max_tokens = self.sampling_params.get("max_tokens")

        if max_model_len and max_tokens:
            tokenizer = AutoTokenizer.from_pretrained(self.model_source)
            tokens = tokenizer.encode(prompt)

            # Calculate available space for prompt (reserve space for generation)
            max_prompt_tokens = max_model_len - max_tokens

            if len(tokens) > max_prompt_tokens:
                # Truncate tokens and decode back to text
                truncated_tokens = tokens[:max_prompt_tokens]
                prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                print(
                    f"Truncated prompt from {len(tokens)} to {len(truncated_tokens)} tokens"
                )

        new_sampling_params = _merge_sp(
            self.sampling_params,
            dict(
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                stop=["</IMPROVED>"],
                seed=int(row.get("prompt_idx", 0)),
            ),
        )
        return dict(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=new_sampling_params,
        )


ProcessorBuilder.register(
    vLLMEngineImProverProcessorConfig, build_vllm_engine_processor
)


class SGLangEngineImProverProcessorConfig(
    ImProverProcessorConfig, SGLangEngineProcessorConfig
):
    def preprocessor(self, row):
        new_sampling_params = _merge_sp(
            self.sampling_params,
            dict(
                stop=["</IMPROVED>"],
                seed=int(row.get("prompt_idx", 0)),
            ),
        )
        return dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=new_sampling_params,
        )


ProcessorBuilder.register(
    SGLangEngineImProverProcessorConfig, build_sglang_engine_processor
)


class HttpRequestImProverProcessorConfig(
    ImProverProcessorConfig, HttpRequestProcessorConfig
):
    def preprocessor(self, row):
        new_sampling_params = _merge_sp(
            self.sampling_params,
            dict(
                stop=["</IMPROVED>"],
                seed=int(row.get("prompt_idx", 0)),
            ),
        )
        return dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=new_sampling_params,
        )


ProcessorBuilder.register(
    HttpRequestImProverProcessorConfig, build_http_request_processor
)


@ray.remote
class CheckpointCoordinator:
    def __init__(
        self, run_id: str, i: int, expected_inf_chunks: int, expected_modules: int
    ):
        self.run_id, self.i = run_id, i
        self.expected_inf_chunks = expected_inf_chunks
        self.expected_modules = expected_modules
        self.inf_seen = 0
        self.eval_seen = 0
        self.inf_done = False
        self.eval_done = False

    def report_inf_chunk(self):
        if self.inf_done:
            return False
        self.inf_seen += 1
        if self.inf_seen == self.expected_inf_chunks:
            self._build_inf_duckdb()
            self.inf_done = True
            return True
        return False

    def report_eval_module(self):
        if self.eval_done:
            return False
        self.eval_seen += 1
        if self.eval_seen == self.expected_modules:
            self._build_eval_duckdb()
            self.eval_done = True
            return True
        return False

    def _build_inf_duckdb(self):

        cp = os.path.join("evals", self.run_id, f"checkpoint_{self.i}")
        data_glob = os.path.join(cp, "data", "**", "*.parquet")
        with duckdb.connect(
            os.path.join("evals", self.run_id, "checkpoints.duckdb")
        ) as con:
            con.execute(
                f"CREATE OR REPLACE VIEW checkpoint_{self.i}_data AS SELECT * FROM '{data_glob}';"
            )

    def _build_eval_duckdb(self):

        cp = os.path.join("evals", self.run_id, f"checkpoint_{self.i}")
        # prefer parquet; fall back to JSON if you kept JSON
        eval_glob = os.path.join(cp, "eval_parquet", "**", "*.parquet")
        with duckdb.connect(
            os.path.join("evals", self.run_id, "checkpoints.duckdb")
        ) as con:
            con.execute(
                f"CREATE OR REPLACE VIEW checkpoint_{self.i}_eval AS SELECT * FROM '{eval_glob}';"
            )


def duplicate_rows_with_prompt_idx(batch, n):
    df = pd.DataFrame(batch)
    out = pd.concat([df.assign(prompt_idx=i) for i in range(n)], ignore_index=True)
    return out  # <- DataFrame


def tag_chunks_global(pdf: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    # pdf contains ONE module because we call this via groupby("module")
    pdf = (
        pdf.assign(_pl=pdf["raw_prompt"].str.len())
        .sort_values("_pl")
        .reset_index(drop=True)
    )
    pdf["__chunk"] = (pdf.index // max_block_size).astype("int64")
    pdf["__group"] = pdf["module"].iat[0] + "|" + pdf["__chunk"].astype(str)
    return pdf.drop(columns=["_pl"])


def _expected_inf_chunks(prompts_df: pd.DataFrame, n: int, max_block_size: int) -> int:
    # Count rows per module BEFORE duplication
    per_mod = prompts_df.groupby("module").size()  # Series: module -> count
    # After n-duplication
    per_mod_n = per_mod * n
    # Chunks per module after size-based chunking
    chunks_per_mod = (per_mod_n + max_block_size - 1) // max_block_size
    expected_inf_chunks = int(chunks_per_mod.sum())
    expected_modules = int(per_mod.shape[0])  # unique modules
    return expected_inf_chunks, expected_modules


ran_inf_ckpt = False
ran_eval_ckpt = False


def _write_inf_ckpt(pdf: pd.DataFrame, run_id: str, i: int, coord) -> pd.DataFrame:
    mod = pdf["module"].iloc[0]
    global ran_inf_ckpt
    if not ran_inf_ckpt:
        shutil.rmtree(
            os.path.join("evals", run_id, f"checkpoint_{i}", "data"), ignore_errors=True
        )
        os.makedirs(
            os.path.join("evals", run_id, f"checkpoint_{i}", "data"), exist_ok=True
        )
        ran_inf_ckpt = True

    chk = int(pdf["__chunk"].iloc[0])
    print(f"[INF-CKPT] module={mod} chunk={chk} rows={len(pdf)}")
    out_dir = os.path.join(
        "evals", run_id, f"checkpoint_{i}", "data", f"module={mod}", f"chunk={chk}"
    )
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pdf), os.path.join(out_dir, f"part-{uuid4().hex}.parquet")
    )
    ray.get(coord.report_inf_chunk.remote())
    return pdf


def _write_eval_ckpt(pdf: pd.DataFrame, run_id: str, i: int, coord) -> pd.DataFrame:
    mod = pdf["module"].iloc[0]
    global ran_eval_ckpt
    if not ran_eval_ckpt:
        shutil.rmtree(
            os.path.join("evals", run_id, f"checkpoint_{i}", "eval_parquet"),
            ignore_errors=True,
        )
        os.makedirs(
            os.path.join("evals", run_id, f"checkpoint_{i}", "eval_parquet"),
            exist_ok=True,
        )
        ran_eval_ckpt = True
    pdir = os.path.join(
        "evals",
        run_id,
        f"checkpoint_{i}",
        "eval_parquet",
        f"module={mod}",
    )
    os.makedirs(pdir, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pdf), os.path.join(pdir, f"part-{uuid4().hex}.parquet")
    )
    # Optional JSON (per module)
    jdir = os.path.join(
        "evals",
        run_id,
        f"checkpoint_{i}",
        "eval",
        mod.replace(".", os.sep),
    )
    os.makedirs(jdir, exist_ok=True)
    with open(os.path.join(jdir, "results.json"), "w") as f:
        json.dump(pdf.to_dict(orient="records"), f)
    ray.get(coord.report_eval_module.remote())
    return pdf


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)  # last-resort fallback


# ---- Postprocessing helpers ----
def _direction_from_metric(metric: Metric) -> str:
    """Return 'increase' if we should maximize, otherwise 'decrease'."""
    mm = (metric.minmax or "").lower()
    if mm.startswith("min"):  # minimize
        return "decrease"
    # default to maximize
    return "increase"


def _stringify_errors(errs):
    try:
        if errs is None:
            return "(none)"
        if isinstance(errs, str):
            return errs
        if isinstance(errs, (list, tuple)):
            return "\n".join(f"- {str(e)}" for e in errs)
        return str(errs)
    except Exception:
        return str(errs)


def _fmt_opt_float(x, ndigits: int = 4):
    try:
        if x is None:
            return "null"
        if isinstance(x, float) and math.isnan(x):
            return "null"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "null"


def postprocess_to_next_prompt(pdf: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """Build next-iteration prompts by appending an evaluator feedback block.

    Input columns expected (from Lean eval):
      - module, decl, decl_idx, prompt_idx (ignored here),
      - original_prompt, new_score, og_score, delta, new_errors, new_correct
    Output columns:
      - module, decl, decl_idx, raw_prompt
    """
    df = pdf.copy()

    direction = _direction_from_metric(metric)
    # Guarantee columns exist
    for col in [
        "original_prompt",
        "new_score",
        "og_score",
        "delta",
        "new_errors",
        "new_correct",
    ]:
        if col not in df.columns:
            df[col] = None

    def _build_feedback_row(r):
        old_s = _fmt_opt_float(r.get("og_score"))
        new_s = _fmt_opt_float(r.get("new_score"))
        dlt = _fmt_opt_float(r.get("delta"))
        errs = _stringify_errors(r.get("new_errors"))
        correctness = "true" if bool(r.get("new_correct")) else "false"

        instruction = (
            f"If there were errors, fix them. Otherwise, {direction} the '{metric.name}' score. "
            f"Target direction per metric.minmax='{metric.minmax}'."
        )

        feedback = (
            "\n<EVAL_FEEDBACK>\n"
            f"metric={metric.name}\n"
            f"correct={correctness}\n"
            f"old_score={old_s}\n"
            f"new_score={new_s}\n"
            f"delta={dlt}\n"
            f"errors:\n{errs}\n"
            "</EVAL_FEEDBACK>\n\n"
            "<FOLLOW_UP_TASK>\n" + instruction + "\n"
            "Respond by outputting only the improved Lean code between <IMPROVED> and </IMPROVED> tags.\n"
            "</FOLLOW_UP_TASK>\n"
        )

        base_prompt = r.get("original_prompt") or r.get("raw_prompt") or ""
        return (base_prompt + "\n\n" + feedback).strip()

    out = pd.DataFrame(
        {
            "module": df["module"],
            "decl": df["decl"],
            "decl_idx": df["decl_idx"],
            "prompt_idx": df["prompt_idx"],
            "raw_prompt": df.apply(_build_feedback_row, axis=1),
        }
    )
    # We intentionally drop prompt_idx and any eval-only columns here; the next
    # iteration will re-duplicate rows and reassign prompt_idx.
    return out


def _eval_one_module(
    pdf: pd.DataFrame, metric: Metric  # , output_dir: str
) -> pd.DataFrame:
    mod = pdf["module"].iloc[0]
    # Keep only specified columns and ensure messages is a string
    cols_to_keep = [
        "module",
        "decl",
        "decl_idx",
        "generated_text",
        "raw_prompt",
        "messages",
        "prompt_idx",
    ]
    pdf = pdf[cols_to_keep]

    # Ensure messages column is a string (JSON)
    if "messages" in pdf.columns and pdf["messages"].dtype == "object":
        pdf["messages"] = pdf["messages"].apply(
            lambda x: json.dumps(x, default=_json_safe) if not isinstance(x, str) else x
        )

    payload = pdf.to_json(orient="records")

    # output_dir = os.path.join(output_dir, "eval", mod.replace(".", os.sep) + ".json")
    # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    payload_bytes = payload.encode()

    cmd = [
        "lake",
        "exe",
        "eval_improver",
        mod,
        metric.name,
        # output_dir,
        str(metric.sorry_ok),
        metric.correctness_condition,
        str(len(payload_bytes)),
    ]
    res = subprocess.run(
        cmd, input=payload, text=True, capture_output=True, encoding="utf-8"
    )
    if res.returncode != 0:
        return pd.DataFrame(
            [
                {
                    "module": mod,
                    "status": "failed",
                    "error": res.stderr,
                    "out": res.stdout,
                }
            ]
        )
    raw_out = res.stdout.strip()
    final_answer_match = re.search(
        r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", raw_out, re.DOTALL
    )
    if final_answer_match:
        final_answer_content = final_answer_match.group(1).strip()
    else:
        final_answer_content = ""
    out = json.loads(final_answer_content)
    df = pd.DataFrame(out)
    # Keep stable keys so we can postprocess/join if needed:
    keys = ["module", "decl", "decl_idx", "prompt_idx"]
    if not all(k in df.columns for k in keys):
        for k in keys:
            if k not in df.columns and k in pdf.columns:
                df[k] = pdf[k]
    return df


# # # AFTER eval: re-split big module batches back into sub-blocks
# def rechunk_after_eval(batch, max_block_size: int) -> pd.DataFrame:
#     df = batch if isinstance(batch, pd.DataFrame) else pd.DataFrame(batch)
#     df = df.sort_values(["module"])  # or by any size proxy you have
#     df["_rank"] = df.groupby("module").cumcount()
#     df["__chunk"] = (df["_rank"] // max_block_size).astype("int64")
#     df["__group"] = df["module"] + "|" + df["__chunk"].astype(str)
#     return df.drop(columns=["_rank"])


def run_inference(
    run_config: ImProverRunConfig, processor_config: ImProverProcessorConfig
):
    if processor_config.nccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "0"
    else:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(
        num_cpus=processor_config.cpus,
        num_gpus=processor_config.gpus,
        _temp_dir=processor_config.ray_temp_dir,
    )
    DataContext.get_current().wait_for_min_actors_s = processor_config.ray_timeout
    DataContext.use_hash_based_shuffle = True

    ## INITIALIZE DATASET
    base_ds = run_config.to_ray_dataset(partition_by_module=True)

    processor = build_llm_processor(
        processor_config,
        preprocess=processor_config.preprocessor,
    )

    # Pre-compute expected blocks per module for the coordinator
    prompts_df = run_config.get_prompts()
    expected_inf_chunks, expected_modules = _expected_inf_chunks(
        prompts_df, processor_config.n, processor_config.max_block_size
    )

    # This will hold the prompts for the *next* iteration
    # 0) Prepare this iteration's dataset: duplicate and chunk within each module

    ds_next = base_ds.map_batches(
        duplicate_rows_with_prompt_idx,
        fn_args=[processor_config.n],
        batch_format="pandas",
    )
    # ds_next = base_ds

    ## CORE LOOP
    for i in range(processor_config.m):
        coord = CheckpointCoordinator.remote(
            run_config.run_id, i, expected_inf_chunks, expected_modules
        )

        print(f"[[ITER {i}]]")
        # if hasattr(ds_next, "count"):
        # print(f"[ITER {i}] Number of base rows: {ds_next.count()}")

        ds_iter = ds_next

        ds_iter = ds_iter.groupby("module").map_groups(
            tag_chunks_global,
            fn_args=[processor_config.max_block_size],
            batch_format="pandas",
        )
        ds_iter = ds_iter.groupby("__group").map_groups(
            lambda pdf: pdf, batch_format="pandas"
        )

        # 1) Inference
        ds_inf = processor(ds_iter)

        # 2) Inference checkpoint: guarantee exactly one write per module|chunk
        ds_inf_ckpt = ds_inf.groupby("__group").map_groups(
            _write_inf_ckpt,
            fn_args=[run_config.run_id, i, coord],
            # lambda pdf: _write_inf_ckpt(pdf, run_config.run_id, i, coord),
            batch_format="pandas",
        )

        # 3) Evaluate once per module
        ds_eval = ds_inf_ckpt.groupby("module").map_groups(
            _eval_one_module,
            fn_args=[run_config.metric],
            # lambda pdf: _eval_one_module(
            #     pdf,
            #     run_config.metric,
            #     os.path.join("evals", run_config.run_id, f"checkpoint_{i}"),
            # ),
            batch_format="pandas",
        )

        # 4) Save eval checkpoints per module + notify coordinator
        ds_eval_ckpt = ds_eval.groupby("module").map_groups(
            _write_eval_ckpt,
            fn_args=[run_config.run_id, i, coord],
            # lambda pdf: _write_eval_ckpt(pdf, run_config.run_id, i, coord),
            batch_format="pandas",
        )

        # 5) Postprocess → prompts for next iteration using evaluator output
        ds_postprocessed = ds_eval_ckpt.map_batches(
            postprocess_to_next_prompt,
            fn_args=[run_config.metric],
            # lambda pdf: postprocess_to_next_prompt(pdf, run_config.metric),
            batch_format="pandas",
        )

        print(f"[ITER {i}] Materializing results and breaking DAG lineage...")
        ds_next = ds_postprocessed.materialize()

    print("All iterations complete.")

    # Materialize the final (postprocessed) dataset
    # ds_next.materialize()


def main(args):

    metric = Metric(args.metric)
    dataset = ImProverDataset(
        dataset_path=args.dataset_path, prompt_id=args.prompt_id, split=args.split
    )

    run_config = ImProverRunConfig(
        metric=metric,
        dataset=dataset,
        run_id=args.run_id,
        annotation=args.annotation,
        goal_state=args.goal_state,
        context=args.context,
        file_context=args.file_context,
        rag=args.rag,
        examples=args.examples,
        thinking="none",  # Default to no thinking strategy
    )

    processor_config = vLLMEngineImProverProcessorConfig(
        max_block_size=args.max_block_size,
        cpus=args.cpus,
        gpus=args.gpus,
        n=args.n,
        m=args.m,
        nccl_p2p=args.nccl_p2p,
        ray_timeout=args.ray_timeout,
        model_source=args.model,
        engine_resources={
            "CPU": args.engine_cpu_resources,
            "GPU": args.engine_gpu_resources,
        },
        concurrency=args.concurrency,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
        sampling_params={
            "max_tokens": args.max_tokens,
            "truncate_prompt_tokens": args.truncate_prompt_tokens,
        },
    )

    # returns the path to the directory containing run metadata and the parquet lake
    run_inference(run_config, processor_config)

    ray.shutdown()

    # we should also initialize + index the duckDB stuff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
    parser.add_argument(
        "--run_id",
        type=str,
        default="RUN_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run ID to use for evaluation (default: run_<timestamp>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Model to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
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
    parser.add_argument("--n", type=int, default=1, help="Best-of-n value (default: 1)")
    parser.add_argument(
        "--m", type=int, default=1, help="Number of iterations (default: 1)"
    )
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
        "--max_block_size",
        type=int,
        default=64,
        help="Maximum block size for Ray Dataset partitions (default: 64)",
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
