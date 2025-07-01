import click
import yaml
import argparse
import asyncio
import multiprocessing
import datetime
try:
    import torch
    _DEFAULT_GPUS = torch.cuda.device_count()
except Exception:
    torch = None
    _DEFAULT_GPUS = 0

METRIC_PROMPT_DEFAULTS = {
    "annotation_prompt": " A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response.",
    "context_prompt": " The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>.",
    "rag_prompt": " The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>.",
    "example_prompt": "Here are some examples of such optimization, as wrapped in <EXAMPLES>...</EXAMPLES>. Note that these examples are for illustrative purposes only and should not be copied directly. Instead, use them to understand the kind of optimization expected and apply similar techniques to the current theorem.",
}


def apply_config(params, defaults, config_path):
    if not config_path:
        return params
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    res = {}
    for key, default in defaults.items():
        res[key] = data.get(key, default)
    return res


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
@click.argument('name')
@click.argument('system_prompt')
@click.option('--score_fn', default=None)
@click.option('--sorry_ok', is_flag=True, default=False)
@click.option('--correctness_condition', default='none')
@click.option('--example_file', default=None)
@click.option('--llm_metric', is_flag=True, default=False)
@click.option('--metric_model', default=None)
@click.option('--rubric', default=None)
@click.option('--annotation_prompt', default=METRIC_PROMPT_DEFAULTS['annotation_prompt'])
@click.option('--context_prompt', default=METRIC_PROMPT_DEFAULTS['context_prompt'])
@click.option('--rag_prompt', default=METRIC_PROMPT_DEFAULTS['rag_prompt'])
@click.option('--example_prompt', default=METRIC_PROMPT_DEFAULTS['example_prompt'])
@click.option('--config', type=click.Path(exists=True), default=None)
def metrics_add(**kwargs):
    """Add a new metric."""
    from ImProver.metrics.add import create_metric, get_parser as metric_parser

    config = kwargs.pop('config')
    parser = metric_parser()
    defaults = vars(parser.parse_args([]))
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    create_metric(args)

# ----- Get Prompts group -----
@cli.group()
def get_prompts():
    """Prompt generation utilities."""
    pass


@get_prompts.command('default')
@click.argument('dataset_path')
@click.option('--prompts_id', default=f"prompts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--informalize', is_flag=True, default=False)
@click.option('--include_context', is_flag=True, default=False)
@click.option('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--config', type=click.Path(exists=True), default=None)
def get_prompts_default(**kwargs):
    """Generate prompts."""
    from ImProver.get_prompts.get_prompts import make_config as gp_make_config, main_async as gp_main_async

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    gp_make_config(args)
    asyncio.run(gp_main_async(args))


@get_prompts.command('informalize')
@click.argument('dataset_path')
@click.argument('prompts_id')
@click.option('--split', default='train')
@click.option('--include_context', is_flag=True, default=False)
@click.option('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--config', type=click.Path(exists=True), default=None)
def informalize(dataset_path, prompts_id, split, include_context, model, cpus, gpus, config):
    """Informalize theorems."""
    from ImProver.get_prompts.informalize import main as informalize_main

    defaults = dict(dataset_path=dataset_path, prompts_id=prompts_id, split=split,
                    include_context=include_context, model=model, cpus=cpus, gpus=gpus)
    params = apply_config(defaults, defaults, config)
    args = argparse.Namespace(**params)
    informalize_main(args)

# ----- Run group -----
@cli.group()
def run():
    """Run pipeline components."""
    pass


@run.command('inference')
@click.argument('metric')
@click.argument('dataset_path')
@click.argument('prompt_id')
@click.option('--runID', default=f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--model', default='deepseek-ai/DeepSeek-Prover-V2-7B')
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--n', default=1)
@click.option('--annotation', is_flag=True, default=False)
@click.option('--context', default=0)
@click.option('--rag', default=0)
@click.option('--examples', default=0)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_inference(**kwargs):
    from ImProver.basic.inference import main as inference_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    inference_main(args)


@run.command('eval')
@click.argument('runID')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--config', type=click.Path(exists=True), default=None)
def run_eval(**kwargs):
    from ImProver.basic.eval_improver import main_async as eval_main_async

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    asyncio.run(eval_main_async(args))


@run.command('analysis')
@click.argument('runID')
@click.option('--training_data', is_flag=True, default=True)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_analysis(**kwargs):
    from ImProver.basic.analysis import main as analysis_main, get_parser as analysis_parser

    config = kwargs.pop('config')
    parser = analysis_parser()
    defaults = vars(parser.parse_args([]))
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    analysis_main(args)


@run.command('llm_metric')
@click.argument('runID')
@click.argument('prompts_id')
@click.option('--model', default=None)
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--n', default=3)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_llm_metric(**kwargs):
    from ImProver.basic.llm_metric import main as llm_metric_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    llm_metric_main(args)


@run.command('pipeline')
@click.argument('metric')
@click.argument('dataset_path')
@click.argument('prompt_id')
@click.option('--model', default='deepseek-ai/DeepSeek-Prover-V2-7B')
@click.option('--split', default='train')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--n', default=1)
@click.option('--annotation', is_flag=True, default=False)
@click.option('--context', default=0)
@click.option('--rag', default=0)
@click.option('--examples', default=0)
@click.option('--runID', default=f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--training_data', is_flag=True, default=True)
@click.option('--config', type=click.Path(exists=True), default=None)
def run_pipeline(**kwargs):
    from ImProver.basic.improver import main as pipeline_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    pipeline_main(args)

# ----- KG group -----
@cli.group()
def KG():
    """Knowledge graph utilities."""
    pass


@KG.command('embed')
@click.argument('prompts_id')
@click.argument('KG_id', required=False, default=f"KG_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_embed(**kwargs):
    from ImProver.KG.build_vector_db import main as embed_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    embed_main(args)

@KG.command('c3')
@click.argument('KG_id')
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--k', default=40)
@click.option('--threshold', default=0.35)
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_c3(**kwargs):
    from ImProver.KG.compute_class3_edges import main as c3_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    c3_main(args)

@KG.command('make_db')
@click.argument('dataset_path')
@click.argument('KG_id')
@click.option('--split', default='train')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_make_db(**kwargs):
    from ImProver.KG.build_combined_db import main as combined_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    combined_main(args)

@KG.command('filter')
@click.argument('KG_id')
@click.option('--heuristic_model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--n', default=1)
@click.option('--run_inference', is_flag=True, default=True)
@click.option('--augment_DB', is_flag=True, default=True)
@click.option('--training_data', is_flag=True, default=True)
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_filter(**kwargs):
    from ImProver.KG.heuristic_filter import main as filter_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    filter_main(args)

@KG.command('insert')
@click.argument('KG_id')
@click.option('--neo4j_uri', default='bolt://localhost:7687')
@click.option('--neo4j_user', default='neo4j')
@click.option('--neo4j_pass', default='12345678')
@click.option('--config', type=click.Path(exists=True), default=None)
def kg_insert(**kwargs):
    from ImProver.KG.insert_neo4j import main as insert_main

    config = kwargs.pop('config')
    defaults = kwargs.copy()
    params = apply_config(kwargs, defaults, config)
    args = argparse.Namespace(**params)
    insert_main(args)

@KG.command('full')
@click.argument('dataset_path')
@click.argument('prompts_id')
@click.option('--KG_id', default=f"KG_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
@click.option('--split', default='train')
@click.option('--embedding_model', default='Qwen/Qwen3-Embedding-0.6B')
@click.option('--k', default=40)
@click.option('--threshold', default=0.35)
@click.option('--heuristic_model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
@click.option('--cpus', default=multiprocessing.cpu_count())
@click.option('--gpus', default=_DEFAULT_GPUS)
@click.option('--n', default=1)
@click.option('--run_inference', is_flag=True, default=True)
@click.option('--augment_DB', is_flag=True, default=True)
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
    args = argparse.Namespace(**params)
    kg_main(args)

if __name__ == '__main__':
    cli()
