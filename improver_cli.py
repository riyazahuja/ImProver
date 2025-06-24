#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_get_prompts(cfg):
    args = [sys.executable, 'ImProver/get_prompts.py', cfg['dataset_path'], cfg.get('prompt_id', f"PROMPTS_{int(os.path.getmtime(cfg['dataset_path']))}")]
    args += ['--split', cfg.get('split', 'train')]
    args += ['--prompts_dir', cfg.get('prompts_dir', '.prompts')]
    args += ['--example_dir', cfg.get('example_dir', '.prompts/.prompt_examples')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    args += ['--python_cmd', cfg.get('python_cmd', sys.executable)]
    subprocess.run(args, check=True)


def run_inference(cfg):
    if 'runID' not in cfg or cfg['runID'] is None:
        cfg['runID'] = 'RUN_' + str(int(__import__('time').time()))
    args = [sys.executable, 'ImProver/inference.py', cfg['metric'], cfg['dataset_path']]
    args += ['--runID', cfg['runID']]
    args += ['--model', cfg.get('model', 'deepseek-ai/DeepSeek-Prover-V2-7B')]
    args += ['--split', cfg.get('split', 'train')]
    args += ['--prompts_dir', cfg.get('prompts_dir', '.prompts')]
    args += ['--output_dir', cfg.get('output_dir', '.evals')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    args += ['--gpus', str(cfg.get('gpus', 0))]
    args += ['--n', str(cfg.get('n', 1))]
    args += ['--annotation', str(cfg.get('annotation', False))]
    args += ['--context', str(cfg.get('context', 0))]
    args += ['--rag', str(cfg.get('rag', 0))]
    args += ['--examples', str(cfg.get('examples', 0))]
    subprocess.run(args, check=True)
    print('RunID:', cfg['runID'])


def run_eval(cfg):
    args = [sys.executable, 'ImProver/eval_improver.py', cfg['runID']]
    args += ['--inference_dir', cfg.get('output_dir', '.evals')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    subprocess.run(args, check=True)

    if cfg.get('metric') == 'readability':
        read_cfg = cfg.get('readability', {})
        args = [sys.executable, 'ImProver/readability.py', cfg['runID'], cfg['prompt_id']]
        if not read_cfg.get('inference', True):
            args += ['--no-inference']
        args += ['--output_dir', cfg.get('output_dir', '.evals')]
        args += ['--model', read_cfg.get('model', 'deepseek-ai/DeepSeek-Prover-V2-7B')]
        args += ['--prompts_dir', cfg.get('prompts_dir', '.prompts')]
        args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
        args += ['--gpus', str(cfg.get('gpus', 0))]
        args += ['--n', str(read_cfg.get('n', 3))]
        subprocess.run(args, check=True)


def run_analysis(cfg):
    args = [sys.executable, 'ImProver/analysis.py', cfg['runID']]
    args += ['--run_dir', cfg.get('output_dir', '.evals')]
    if not cfg.get('analysis', {}).get('training_data', True):
        args += ['--no-training_data']
    subprocess.run(args, check=True)


def run_pipeline(config_path: str) -> None:
    """Run the full proof-evaluation pipeline via ``improver.py``."""
    args = [sys.executable, "improver.py", "--config", config_path]
    subprocess.run(args, check=True)


def run_kg_get_class2(cfg):
    args = [sys.executable, 'ImProver/KG/get_class2.py', cfg['dataset_path']]
    if cfg.get('KG_id'):
        args += ['--KG_id', cfg['KG_id']]
    args += ['--split', cfg.get('split', 'train')]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    subprocess.run(args, check=True)


def run_kg_informalize(cfg):
    args = [sys.executable, 'ImProver/KG/informalize.py', cfg['dataset_path'], cfg['KG_id']]
    args += ['--split', cfg.get('split', 'train')]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    if cfg.get('include_context'):
        args += ['--include_context']
    args += ['--model', cfg.get('model', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    args += ['--gpus', str(cfg.get('gpus', 0))]
    subprocess.run(args, check=True)


def run_kg_build_vector_db(cfg):
    args = [sys.executable, 'ImProver/KG/build_vector_db.py', cfg['KG_id']]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    args += ['--model', cfg.get('vector_model', 'Qwen/Qwen3-Embedding-0.6B')]
    subprocess.run(args, check=True)


def run_kg_compute_edges(cfg):
    args = [sys.executable, 'ImProver/KG/compute_class3_edges.py', cfg['KG_id']]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    args += ['--model', cfg.get('edge_model', 'Qwen/Qwen3-Embedding-0.6B')]
    args += ['--k', str(cfg.get('edge_k', 40))]
    args += ['--threshold', str(cfg.get('edge_threshold', 0.35))]
    subprocess.run(args, check=True)


def run_kg_build_combined(cfg):
    args = [sys.executable, 'ImProver/KG/build_combined_db.py', cfg['dataset_path'], cfg['KG_id']]
    args += ['--split', cfg.get('split', 'train')]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    subprocess.run(args, check=True)


def run_kg_filter(cfg):
    args = [sys.executable, 'ImProver/KG/heuristic_filter.py', cfg['KG_id']]
    args += ['--KG_dir', cfg.get('KG_dir', '.knowledge_graphs')]
    args += ['--model', cfg.get('heuristic_model', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')]
    args += ['--cpus', str(cfg.get('cpus', os.cpu_count()))]
    args += ['--gpus', str(cfg.get('gpus', 0))]
    args += ['--n', str(cfg.get('heuristic_n', 1))]
    if not cfg.get('heuristic_run_inference', True):
        args += ['--no-run_inference']
    if not cfg.get('heuristic_augment_DB', True):
        args += ['--no-augment_DB']
    if not cfg.get('heuristic_training_data', True):
        args += ['--no-training_data']
    subprocess.run(args, check=True)


def run_kg_insert(cfg):
    args = [sys.executable, 'ImProver/KG/insert_neo4j.py', cfg['KG_id'], cfg.get('KG_dir', '.knowledge_graphs')]
    args += ['--neo4j_uri', cfg.get('neo4j_uri', 'bolt://localhost:7687')]
    args += ['--neo4j_user', cfg.get('neo4j_user', 'neo4j')]
    args += ['--neo4j_pass', cfg.get('neo4j_pass', '12345678')]
    subprocess.run(args, check=True)


def kg_pipeline(cfg, insert=True):
    run_kg_get_class2(cfg)
    run_kg_informalize(cfg)
    run_kg_build_vector_db(cfg)
    run_kg_compute_edges(cfg)
    run_kg_build_combined(cfg)
    run_kg_filter(cfg)
    if insert:
        run_kg_insert(cfg)


def kg_c3(cfg, insert=True):
    run_kg_informalize(cfg)
    run_kg_build_vector_db(cfg)
    run_kg_compute_edges(cfg)
    run_kg_build_combined(cfg)
    run_kg_filter(cfg)
    if insert:
        run_kg_insert(cfg)


def main():
    parser = argparse.ArgumentParser(description='ImProver command line interface')
    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help='Run proof optimization pipeline')
    run_sub = run_parser.add_subparsers(dest='stage')

    parser_run_all = run_sub.add_parser('all', help='Run full pipeline')
    parser_run_all.add_argument('--config', type=str, required=True)

    parser_run_prompts = run_sub.add_parser('prompts', help='Generate prompts')
    parser_run_prompts.add_argument('--config', type=str, required=True)

    parser_run_infer = run_sub.add_parser('inference', help='Run model inference')
    parser_run_infer.add_argument('--config', type=str, required=True)

    parser_run_eval = run_sub.add_parser('eval', help='Evaluate proofs')
    parser_run_eval.add_argument('--config', type=str, required=True)

    parser_run_analysis = run_sub.add_parser('analysis', help='Analyze run')
    parser_run_analysis.add_argument('--config', type=str, required=True)

    kg_parser = subparsers.add_parser('KG', help='Knowledge graph utilities')
    kg_sub = kg_parser.add_subparsers(dest='stage')

    parser_kg_all = kg_sub.add_parser('all', help='Run full KG pipeline')
    parser_kg_all.add_argument('--config', type=str, required=True)

    parser_kg_data = kg_sub.add_parser('data', help='Build KG data only')
    parser_kg_data.add_argument('--config', type=str, required=True)

    parser_kg_insert = kg_sub.add_parser('insert', help='Insert existing KG into neo4j')
    parser_kg_insert.add_argument('--config', type=str, required=True)

    parser_kg_c2 = kg_sub.add_parser('c2', help='Generate class2 data')
    parser_kg_c2.add_argument('--config', type=str, required=True)

    parser_kg_c3 = kg_sub.add_parser('c3', help='Run steps after class2 generation')
    parser_kg_c3.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'run':
        cfg = load_config(args.config)
        if args.stage == 'prompts':
            run_get_prompts(cfg)
        elif args.stage == 'inference':
            run_inference(cfg)
        elif args.stage == 'eval':
            run_eval(cfg)
        elif args.stage == 'analysis':
            run_analysis(cfg)
        elif args.stage == 'all' or args.stage is None:
            run_pipeline(args.config)
        else:
            parser.print_help()
    elif args.command == 'KG':
        cfg = load_config(args.config)
        if args.stage == 'insert':
            run_kg_insert(cfg)
        elif args.stage == 'c2':
            run_kg_get_class2(cfg)
        elif args.stage == 'c3':
            kg_c3(cfg, insert=False)
        elif args.stage == 'data':
            kg_pipeline(cfg, insert=False)
        elif args.stage == 'all' or args.stage is None:
            kg_pipeline(cfg, insert=True)
        else:
            parser.print_help()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
