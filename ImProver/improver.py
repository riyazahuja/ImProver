import asyncio
import multiprocessing
import os
from types import SimpleNamespace
from pathlib import Path
from .get_prompts import make_config as gp_make_config, main_async as gp_main
from .inference import main as inf_main
from .eval_improver import main_async as eval_main
from .readability import main as readability_main
from .analysis import main as analysis_main


def run_pipeline(config: dict):
    """Run the full evaluation pipeline given a flat configuration dictionary."""

    prompt_id = config.get("prompt_id")
    if prompt_id is None:
        gp_args = {k: config.get(k) for k in [
            "dataset_path",
            "prompt_id",
            "split",
            "prompts_dir",
            "example_dir",
            "cpus",
            "python_cmd",
        ]}
        if gp_args.get("dataset_path") is None:
            raise ValueError("dataset_path is required when prompt_id is not provided")
        gp_args.setdefault("cpus", multiprocessing.cpu_count())
        gp_args.setdefault("prompts_dir", ".prompts")
        gp_args.setdefault("example_dir", ".prompts/.prompt_examples")
        gp_args.setdefault("python_cmd", "python")
        args = SimpleNamespace(**gp_args)
        gp_make_config(args)
        asyncio.run(gp_main(args))
        prompt_id = args.prompt_id

    inf_args = {k: config.get(k) for k in [
        "metric",
        "dataset_path",
        "model",
        "split",
        "prompts_dir",
        "output_dir",
        "cpus",
        "gpus",
        "n",
        "annotation",
        "context",
        "rag",
        "examples",
        "runID",
    ]}
    inf_args.setdefault("cpus", multiprocessing.cpu_count())
    inf_args.setdefault("gpus", 0)
    inf_args.setdefault("output_dir", ".evals/")
    inf_args.setdefault("prompts_dir", ".prompts/")
    inf_args.setdefault("split", "train")
    inf_args["prompt_id"] = prompt_id
    inf_ns = SimpleNamespace(**inf_args)
    run_id = inf_main(inf_ns)

    eval_args = {
        "runID": run_id,
        "inference_dir": config.get("inference_dir", inf_ns.output_dir),
        "cpus": config.get("cpus", multiprocessing.cpu_count()),
    }
    eval_ns = SimpleNamespace(**eval_args)
    asyncio.run(eval_main(eval_ns))

    if inf_ns.metric == "readability":
        read_ns = SimpleNamespace(
            runID=run_id,
            prompts_id=prompt_id,
            inference=config.get("readability_inference", True),
            model=config.get("model", "deepseek-ai/DeepSeek-Prover-V2-7B"),
            split=config.get("split", "train"),
            prompts_dir=config.get("prompts_dir", ".prompts/"),
            output_dir=config.get("output_dir", ".evals/"),
            cpus=config.get("cpus", multiprocessing.cpu_count()),
            gpus=config.get("gpus", 0),
            n=config.get("readability_n", 3),
        )
        readability_main(read_ns)

    analysis_ns = SimpleNamespace(
        RunID=run_id,
        run_dir=config.get("run_dir", inf_ns.output_dir),
        training_data=config.get("training_data", True),
    )
    analysis_main(analysis_ns)
    return run_id


def run_kg_pipeline(config: dict, insert: bool = True):
    """Run the knowledge graph construction pipeline."""

    prompt_id = config.get("prompts_id")
    if prompt_id is None:
        gp_args = {k: config.get(k) for k in [
            "dataset_path",
            "prompt_id",
            "split",
            "prompts_dir",
            "example_dir",
            "cpus",
            "python_cmd",
        ]}
        if gp_args.get("dataset_path") is None:
            raise ValueError("dataset_path is required when prompts_id is not provided")
        gp_args.setdefault("cpus", multiprocessing.cpu_count())
        gp_args.setdefault("prompts_dir", ".prompts")
        gp_args.setdefault("example_dir", ".prompts/.prompt_examples")
        gp_args.setdefault("python_cmd", "python")
        args = SimpleNamespace(**gp_args)
        gp_make_config(args)
        asyncio.run(gp_main(args))
        prompt_id = args.prompt_id

    kg_id = config.get("KG_id", f"KG_{prompt_id}")

    informalize_args = SimpleNamespace(
        dataset_path=config.get("dataset_path"),
        prompts_id=prompt_id,
        split=config.get("split", "train"),
        prompts_dir=config.get("prompts_dir", ".prompts"),
        include_context=config.get("include_context", False),
        model=config.get("model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        cpus=config.get("cpus", multiprocessing.cpu_count()),
        gpus=config.get("gpus", 0),
    )
    from .KG import informalize as kg_informalize

    kg_informalize.main(informalize_args)

    from .KG import build_vector_db as kg_build_vector
    build_vector_args = SimpleNamespace(
        prompts_id=prompt_id,
        KG_id=kg_id,
        prompts_dir=config.get("prompts_dir", ".prompts"),
        KG_dir=config.get("KG_dir", ".knowledge_graphs"),
        model=config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
    )
    kg_build_vector.main(build_vector_args)

    from .KG import compute_class3_edges as kg_c3
    c3_args = SimpleNamespace(
        KG_id=kg_id,
        KG_dir=config.get("KG_dir", ".knowledge_graphs"),
        model=config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        k=config.get("k", 40),
        threshold=config.get("threshold", 0.35),
    )
    kg_c3.main(c3_args)

    from .KG import build_combined_db as kg_combined
    combine_args = SimpleNamespace(
        dataset_path=config.get("dataset_path"),
        KG_id=kg_id,
        split=config.get("split", "train"),
        KG_dir=config.get("KG_dir", ".knowledge_graphs"),
    )
    kg_combined.main(combine_args)

    from .KG import heuristic_filter as kg_filter
    filter_args = SimpleNamespace(
        KG_id=kg_id,
        KG_dir=config.get("KG_dir", ".knowledge_graphs"),
        model=config.get("model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        cpus=config.get("cpus", multiprocessing.cpu_count()),
        gpus=config.get("gpus", 0),
        n=config.get("n", 1),
        run_inference=config.get("run_inference", True),
        augment_DB=config.get("augment_DB", True),
        training_data=config.get("training_data", True),
    )
    kg_filter.main(filter_args)

    if insert:
        from .KG import insert_neo4j as kg_insert
        insert_args = SimpleNamespace(
            KG_id=kg_id,
            KG_dir=config.get("KG_dir", ".knowledge_graphs"),
            neo4j_uri=config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=config.get("neo4j_user", "neo4j"),
            neo4j_pass=config.get("neo4j_pass", "12345678"),
        )
        kg_insert.main(insert_args)

    return kg_id
