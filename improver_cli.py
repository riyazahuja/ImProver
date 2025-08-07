import click
import yaml
import argparse
import asyncio
import multiprocessing
import datetime

def get_default_gpus(params):
    try:
        import torch
        _DEFAULT_GPUS = torch.cuda.device_count()
    except Exception:
        torch = None
        _DEFAULT_GPUS = 0
    if params.get('gpus') is None:
        params['gpus'] = _DEFAULT_GPUS
        
    
    cpus = params.get('cpus')    
    if cpus is None:
        params['cpus'] = multiprocessing.cpu_count()
        cpus = params['cpus']
    else:
        params['engine_cpu_resources'] = int(cpus) // _DEFAULT_GPUS if _DEFAULT_GPUS > 0 else cpus
        params['concurrency'] = _DEFAULT_GPUS if _DEFAULT_GPUS > 0 else 1
    
    return params


    

METRIC_PROMPT_DEFAULTS = {
    "annotation_prompt": " A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response.",
    "context_prompt": " The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>.",
    "rag_prompt": " The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>.",
    "example_prompt": "Here are some examples of such optimization, as wrapped in <EXAMPLES>...</EXAMPLES>. Note that these examples are for illustrative purposes only and should not be copied directly. Instead, use them to understand the kind of optimization expected and apply similar techniques to the current theorem.",
    "goal_state_prompt": " Additionally, the initial goal state of the theorem has been provided to help you better understand the proof statement and ensure the correctness of your response. It is wrapped in <GOAL_STATE>...</GOAL_STATE>.",
    "file_context_prompt": " The file context of the theorem, i.e. the preceding definitions, theorems, etc., have been provided to help understand the context of the theorem and ensure the correctness of your response. It is wrapped in <FILE_CONTEXT>...</FILE_CONTEXT>, with each individual item wrapped in <ITEM>...</ITEM>."
}


def apply_config(params, defaults, config_path):
    if not config_path:
        return params
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    res = {}
    for key, default in defaults.items():
        res[key] = data.get(key, default)
    # for key, value in params.items():
    return res


def extract_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest == 'help':
            continue
        defaults[action.dest] = action.default
    return defaults


def require_params(params, required):
    missing = [k for k in required if params.get(k) is None]
    if missing:
        raise click.UsageError(
            "Missing required parameters: " + ", ".join(missing)
        )


@click.group()
def cli():
    """ImProver command line interface."""
    pass

# ----- Metrics group -----
@cli.group()
def metrics():
    """Metric utilities."""
    pass


@metrics.command('add')
@click.option('--name', required=False, help="Name of the metric.")
@click.option('--system_prompt', required=False, help="System prompt for the metric.")
@click.option('--minmax', default="max")
@click.option('--input_sorry', is_flag=True, default=False)
@click.option('--score_fn', default=None)
@click.option('--sorry_ok', is_flag=True, default=False)
@click.option('--correctness_condition', default='none')
@click.option('--example_file', default=None)
@click.option('--rag_id', default=None)
@click.option('--k', default=5)
@click.option('--llm_metric', is_flag=True, default=False)
@click.option('--metric_model', default=None)
@click.option('--rubric', default=None)
@click.option('--annotation_prompt', default=METRIC_PROMPT_DEFAULTS['annotation_prompt'])
@click.option('--context_prompt', default=METRIC_PROMPT_DEFAULTS['context_prompt'])
@click.option('--rag_prompt', default=METRIC_PROMPT_DEFAULTS['rag_prompt'])
@click.option('--example_prompt', default=METRIC_PROMPT_DEFAULTS['example_prompt'])
@click.option('--goal_state_prompt', default=METRIC_PROMPT_DEFAULTS['goal_state_prompt'])
@click.option('--file_context_prompt', default=METRIC_PROMPT_DEFAULTS['file_context_prompt'])
@click.option('--config', type=click.Path(exists=True), default=None)
def metrics_add(**kwargs):
    """Add a new metric."""
    from ImProver.metrics.add import create_metric, get_parser as metric_parser

    config = kwargs.pop('config')
    parser = metric_parser()
    defaults = extract_defaults(parser)
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['name', 'system_prompt'])
    args = argparse.Namespace(**params)
    create_metric(args)



@metrics.command('reload')
@click.option('--names', default=None)
@click.option('--rag_id', default=None)
@click.option('--k', default=5)
@click.option('--config', type=click.Path(exists=True), default=None)
def metrics_reload(**kwargs):
    """Reload existing metric(s)."""
    from ImProver.metrics.reload import main as reload_main, get_parser as reload_parser

    config = kwargs.pop('config')
    parser = reload_parser()
    defaults = extract_defaults(parser)
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    reload_main(args)



# ----- Get Prompts group -----
@cli.group()
def prompts():
    """Prompt generation utilities."""
    pass


@prompts.command('get')
@click.option('--dataset_path', required=False, help="Path to the dataset.")
@click.option('--prompts_id', default=f"prompts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--rag_id', default=None)
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--timeout', default=1800)
@click.option('--k', default=10)
@click.option('--config', type=click.Path(exists=True), default=None)
def prompts_get(**kwargs):
    """Generate prompts."""
    from ImProver.get_prompts.get_prompts import main as gp_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['dataset_path'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)

    gp_main(args)

# ----- RAG group -----
@cli.group()
def rag():
    """RAG building utilities."""
    pass

@rag.command('build')
@click.option('--dataset_path', required=False, help="Path to the dataset.")
@click.option('--rag_id', default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--max_depth', default=2)
@click.option('--split', default=None)
@click.option('--include_context', is_flag=True, default=False)
@click.option('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)

@click.option('--nccl_p2p', is_flag=True, default=False)
@click.option('--ray_timeout', default=1800)
@click.option('--num_blocks', default=16)
@click.option('--engine_cpu_resources', default=None)
@click.option('--engine_gpu_resources', default=1)
@click.option('--concurrency', default=None)
@click.option('--tensor_parallel_size', default=1)
@click.option('--enable_chunked_prefill', is_flag=True, default=True)
@click.option('--max_model_len', default=16384)
@click.option('--max_num_batched_tokens', default=65536)
@click.option('--max_concurrent_batches', default=32)
@click.option('--batch_size', default=32)
@click.option('--truncate_prompt_tokens', default=14336)
@click.option('--max_tokens', default=2048)


@click.option('--config', type=click.Path(exists=True), default=None)
def rag_build(**kwargs):
    """Build a RAG."""
    from ImProver.build.rag import main as rag_main
    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['dataset_path'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)
    rag_main(args)

@rag.command('preprocess')
@click.option('--dataset_path', required=False, help="Path to the dataset.")
@click.option('--rag_id', default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--split', default=None)
@click.option('--config', type=click.Path(exists=True), default=None)
def rag_preprocess(**kwargs):
    """Preprocess a RAG."""
    from ImProver.build.preprocess_rag import main as preprocess_rag_main
    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['dataset_path'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)
    preprocess_rag_main(args)


@rag.command('informalize')
@click.option('--rag_id', required=False, help="RAG ID to informalize.")
@click.option('--max_depth', default=2)
@click.option('--include_context', is_flag=True, default=False)
@click.option('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)

@click.option('--nccl_p2p', is_flag=True, default=False)
@click.option('--ray_timeout', default=1800)
@click.option('--num_blocks', default=16)
@click.option('--engine_cpu_resources', default=None)
@click.option('--engine_gpu_resources', default=1)
@click.option('--concurrency', default=None)
@click.option('--tensor_parallel_size', default=1)
@click.option('--enable_chunked_prefill', is_flag=True, default=True)
@click.option('--max_model_len', default=16384)
@click.option('--max_num_batched_tokens', default=65536)
@click.option('--max_concurrent_batches', default=32)
@click.option('--batch_size', default=32)
@click.option('--truncate_prompt_tokens', default=14336)
@click.option('--max_tokens', default=2048)

@click.option('--config', type=click.Path(exists=True), default=None)
def rag_informalize(**kwargs):
    """Informalize a RAG."""
    from ImProver.build.informalize import main as informalize_main
    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['rag_id'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)
    informalize_main(args)


@rag.command('build_db')
@click.option('--rag_id', required=False, help="RAG ID to build DB for.")
@click.option('--config', type=click.Path(exists=True), default=None)
def rag_build_db(**kwargs):
    """Build a RAG database."""
    from ImProver.build.build_db import main as build_db_main
    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['rag_id'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)
    build_db_main(args)




# ----- Run group -----
@cli.group()
def run():
    """Run pipeline components."""
    pass


@run.command('inference')
@click.option('--metric', required=False, help="Metric name.")
@click.option('--dataset_path', required=False, help="Path to the dataset.")
@click.option('--prompt_id', required=False, help="Prompt ID.")
@click.option('--run_id', default=f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--model', default='deepseek-ai/DeepSeek-Prover-V2-7B')
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--n', default=1)
@click.option('--annotation', is_flag=True, default=False)
@click.option('--context', default=0)
@click.option('--rag', default=0)
@click.option('--examples', default=0)
@click.option('--goal_state', is_flag=True, default=False)
@click.option('--file_context', default=0)

@click.option('--azure', default=False)
@click.option('--server_concurrency', default=10)
@click.option('--server_rate_limit', default=60)

@click.option('--nccl_p2p', is_flag=True, default=False)
@click.option('--ray_timeout', default=1800)
@click.option('--num_blocks', default=16)
@click.option('--engine_cpu_resources', default=None)
@click.option('--engine_gpu_resources', default=1)
@click.option('--concurrency', default=None)
@click.option('--tensor_parallel_size', default=1)
@click.option('--enable_chunked_prefill', is_flag=True, default=True)
@click.option('--max_model_len', default=16384)
@click.option('--max_num_batched_tokens', default=65536)
@click.option('--max_concurrent_batches', default=32)
@click.option('--batch_size', default=32)
@click.option('--truncate_prompt_tokens', default=14336)
@click.option('--max_tokens', default=2048)

@click.option('--config', type=click.Path(exists=True), default=None)
def run_inference(**kwargs):
    from ImProver.basic.inference import main as inference_main
    from ImProver.basic.inference_server import main as server_inference_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['metric', 'dataset_path', 'prompt_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    if args.azure:
        server_inference_main(args)
    else:
        inference_main(args)


@run.command('eval')
@click.option('--run_id', required=False, help="Run ID to evaluate.")
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--config', type=click.Path(exists=True), default=None)
def run_eval(**kwargs):
    from ImProver.basic.eval_improver import main_async as eval_main_async

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['run_id'])
    args = argparse.Namespace(**params)
    asyncio.run(eval_main_async(args))


@run.command('analysis')
@click.option('--run_id', required=False, help="Run ID to analyze.")
@click.option('--training_data', is_flag=True, default=True)
@click.option('--config', type=click.Path(exists=True), default=None)
@click.option('--thinking', default='none', help="Thinking mode for analysis (default: none). Options: 'none', 'raw'.")
def run_analysis(**kwargs):
    from ImProver.basic.analysis import main as analysis_main, get_parser as analysis_parser

    config = kwargs.pop('config')
    parser = analysis_parser()
    defaults = extract_defaults(parser)
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['run_id'])
    args = argparse.Namespace(**params)
    analysis_main(args)


@run.command('llm_metric')
@click.option('--run_id', required=False, help="Run ID for LLM metric.")
@click.option('--model', default=None)
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--n', default=3)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_llm_metric(**kwargs):
    from ImProver.basic.llm_metric import main as llm_metric_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['run_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    llm_metric_main(args)

@run.command('synthetic_thinking')
@click.option('--run_id', required=False, help="Run ID for synthetic thinking.")
@click.option('--model', required=False, help="Model for synthetic thinking.")
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--num_blocks', default=16)
@click.option('--engine_cpu_resources', default=None)
@click.option('--engine_gpu_resources', default=1)
@click.option('--concurrency', default=None)
@click.option('--tensor_parallel_size', default=1)
@click.option('--enable_chunked_prefill', is_flag=True, default=True)
@click.option('--max_model_len', default=16384)
@click.option('--max_num_batched_tokens', default=65536)
@click.option('--max_concurrent_batches', default=32)
@click.option('--batch_size', default=32)
@click.option('--truncate_prompt_tokens', default=14336)
@click.option('--max_tokens', default=2048)
@click.option('--nccl_p2p', is_flag=True, default=False)
@click.option('--ray_timeout', default=1800)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_synthetic_thinking(**kwargs):
    from ImProver.basic.synth_thinking import main as synthetic_thinking_main
    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['run_id', 'model'])
    params = get_default_gpus(params)
    args = argparse.Namespace(**params)
    synthetic_thinking_main(args)

@run.command('pipeline')
@click.option('--metric', required=False, help="Metric name.")
@click.option('--dataset_path', required=False, help="Path to the dataset.")
@click.option('--prompt_id', required=False, help="Prompt ID.")
@click.option('--model', default='deepseek-ai/DeepSeek-Prover-V2-7B')
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--n', default=1)
@click.option('--annotation', is_flag=True, default=False)
@click.option('--context', default=0)
@click.option('--rag', default=0)
@click.option('--examples', default=0)
@click.option('--goal_state', is_flag=True, default=False)
@click.option('--file_context', default=0)

@click.option('--azure', default=False)
@click.option('--server_concurrency', default=10)
@click.option('--server_rate_limit', default=60)

@click.option('--nccl_p2p', is_flag=True, default=False)
@click.option('--ray_timeout', default=1800)
@click.option('--num_blocks', default=16)
@click.option('--engine_cpu_resources', default=None)
@click.option('--engine_gpu_resources', default=1)
@click.option('--concurrency', default=None)
@click.option('--tensor_parallel_size', default=1)
@click.option('--enable_chunked_prefill', is_flag=True, default=True)
@click.option('--max_model_len', default=16384)
@click.option('--max_num_batched_tokens', default=65536)
@click.option('--max_concurrent_batches', default=32)
@click.option('--batch_size', default=32)
@click.option('--truncate_prompt_tokens', default=14336)
@click.option('--max_tokens', default=2048)

@click.option('--run_id', default=f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--training_data', is_flag=True, default=True)
@click.option('--thinking', default='none', help="Thinking mode for analysis (default: none). Options: 'none', 'raw'.")
@click.option('--config', type=click.Path(exists=True), default=None)
def run_pipeline(**kwargs):
    from ImProver.basic.improver import main as pipeline_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    print(defaults)
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['metric', 'dataset_path', 'prompt_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    pipeline_main(args)
    
    
@run.command('training_data')
@click.option('--run_id', required=False, help="Identifier for the run.")
@click.option('--output_path', default=None, help="Output path for JSONL file (default: evals/[run_id]/analysis/BoN/training_data.jsonl)")
@click.option('--thinking', is_flag=True, default=False, help="Use CoT output instead of regular output (default: False)")
@click.option('--filter_threshold', type=float, default=1.0, help="Threshold for filtering high improvement rate items (default: 1.0)")
@click.option('--prev_run_id', default=None, help="Previous run ID for replay buffer (default: None)")
@click.option('--replay_buffer_split', type=float, default=None, help="Target proportion of replay items (default: None)")
@click.option('--type', type=click.Choice(['sft', 'weighted_sft', 'dpo']), default="sft", help="Type of training dataset (default: sft)")
@click.option('--tau', default=1.0, help="temperature value for weighted SFT")

@click.option('--config', type=click.Path(exists=True), default=None)
def run_pipeline(**kwargs):
    from ImProver.basic.training_data import main as training_data_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    print(defaults)
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['run_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    training_data_main(args)

# ----- KG group -----
@cli.group()
def KG():
    """Knowledge graph utilities."""
    pass


@KG.command('embed')
@click.option('--prompts_id', required=False, help="Prompts ID for KG embedding.")
@click.option('--kg_id', default=f"KG_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_embed(**kwargs):
    from ImProver.KG.build_vector_db import main as embed_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['prompts_id'])
    args = argparse.Namespace(**params)
    embed_main(args)

@KG.command('c3')
@click.option('--prompts_id', required=False, help="Prompts ID for KG c3.")
@click.option('--kg_id', required=False, help="KG ID for c3.")
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--k', default=40)
@click.option('--threshold', default=0.35)
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_c3(**kwargs):
    from ImProver.KG.compute_class3_edges import main as c3_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['prompts_id', 'kg_id'])
    args = argparse.Namespace(**params)
    c3_main(args)

@KG.command('make_db')
@click.option('--dataset_path', required=False, help="Dataset path for KG DB.")
@click.option('--prompts_id', required=False, help="Prompts ID for KG DB.")
@click.option('--kg_id', required=False, help="KG ID for DB.")
@click.option('--split', default='train')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_make_db(**kwargs):
    from ImProver.KG.build_combined_db import main as combined_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['dataset_path', 'prompts_id','kg_id'])
    args = argparse.Namespace(**params)
    combined_main(args)

@KG.command('filter')
@click.option('--kg_id', required=False, help="KG ID to filter.")
@click.option('--heuristic_model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--n', default=1)
@click.option('--run_inference', is_flag=True, default=True)
@click.option('--augment_db', is_flag=True, default=True)
@click.option('--training_data', is_flag=True, default=True)
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_filter(**kwargs):
    from ImProver.KG.heuristic_filter import main as filter_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['kg_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    filter_main(args)

@KG.command('insert')
@click.option('--kg_id', required=False, help="KG ID to insert.")
@click.option('--neo4j_uri', default='bolt://localhost:7687')
@click.option('--neo4j_user', default='neo4j')
@click.option('--neo4j_pass', default='12345678')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_insert(**kwargs):
    from ImProver.KG.insert_neo4j import main as insert_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['kg_id'])
    args = argparse.Namespace(**params)
    insert_main(args)

@KG.command('full')
@click.option('--dataset_path', required=False, help="Dataset path for KG full.")
@click.option('--prompts_id', required=False, help="Prompts ID for KG full.")
@click.option('--kg_id', default=f"KG_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--split', default='train')
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--k', default=40)
@click.option('--threshold', default=0.35)
@click.option('--heuristic_model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=None)
@click.option('--n', default=1)
@click.option('--run_inference', is_flag=True, default=True)
@click.option('--augment_db', is_flag=True, default=True)
@click.option('--training_data', is_flag=True, default=True)
@click.option('--neo4j_uri', default='bolt://localhost:7687')
@click.option('--neo4j_user', default='neo4j')
@click.option('--neo4j_pass', default='12345678')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_full(**kwargs):
    from ImProver.KG.KG import main as kg_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    require_params(params, ['dataset_path', 'prompts_id'])
    params = get_default_gpus(params)

    args = argparse.Namespace(**params)
    # print(args.__dict__)  # Debugging line to print arguments
    kg_main(args)

if __name__ == '__main__':
    cli()