import asyncio
import argparse
import multiprocessing
import sys
import os
import time
from pathlib import Path
from types import SimpleNamespace
import typer
import yaml

from ImProver.get_prompts import get_parser as gp_parser, make_config as gp_make_config, main_async as gp_main
from ImProver.inference import get_parser as inf_parser, main as inf_main
from ImProver.eval_improver import get_parser as eval_parser, main_async as eval_main
from ImProver.analysis import get_parser as analysis_parser, main as analysis_main
from ImProver.readability import get_parser as read_parser, main as readability_main
from ImProver.improver import run_pipeline, run_kg_pipeline
from ImProver.KG.informalize import get_parser as kg_inf_parser
from ImProver.KG.build_vector_db import get_parser as kg_build_parser
from ImProver.KG.compute_class3_edges import get_parser as kg_c3_parser
from ImProver.KG.build_combined_db import get_parser as kg_combined_parser
from ImProver.KG.heuristic_filter import get_parser as kg_filter_parser
from ImProver.KG.insert_neo4j import get_parser as kg_insert_parser

app = typer.Typer(help="Command line interface for the ImProver research framework")
kg_app = typer.Typer(help="Knowledge graph utilities")
app.add_typer(kg_app, name="KG")


def load_config(path: Path) -> dict:
    if path is None:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _config_to_args(parser: argparse.ArgumentParser, cfg: dict) -> list[str]:
    args = []
    for act in parser._actions:
        if act.dest == "help":
            continue
        if act.dest in cfg and cfg[act.dest] is not None:
            val = cfg[act.dest]
            if not act.option_strings:
                args.append(str(val))
            elif isinstance(act, argparse._BooleanOptionalAction):
                if val:
                    args.append(act.option_strings[0])
                else:
                    if len(act.option_strings) > 1:
                        args.append(act.option_strings[1])
            elif act.nargs == 0:
                if val:
                    args.append(act.option_strings[0])
            else:
                args.append(act.option_strings[0])
                args.append(str(val))
    return args


def _parse_with_config(parser_func, extras: list[str], cfg: dict) -> argparse.Namespace:
    parser = parser_func()
    if "--help" in extras or "-h" in extras:
        parser.print_help()
        raise typer.Exit()
    args = _config_to_args(parser, cfg) + extras
    return parser.parse_args(args)


def _combine_parsers(*parsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    combined = argparse.ArgumentParser(add_help=False)
    seen = set()
    for p in parsers:
        for a in p._actions:
            if a.dest == "help" or a.dest in seen:
                continue
            params = {
                "help": a.help,
                "default": a.default,
                "nargs": a.nargs,
                "choices": getattr(a, "choices", None),
                "type": getattr(a, "type", None),
                "action": a.__class__ if a.option_strings else None,
                "metavar": a.metavar,
            }
            params = {k: v for k, v in params.items() if v is not None}
            if a.option_strings:
                combined.add_argument(*a.option_strings, **params)
            else:
                combined.add_argument(a.dest, **params)
            seen.add(a.dest)
    return combined


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def get_prompts(ctx: typer.Context, config: Path = typer.Option(None, help="YAML config")):
    cfg = load_config(config)
    ns = _parse_with_config(gp_parser, ctx.args, cfg)
    gp_make_config(ns)
    asyncio.run(gp_main(ns))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def inference(ctx: typer.Context, config: Path = typer.Option(None, help="YAML config")):
    cfg = load_config(config)
    ns = _parse_with_config(inf_parser, ctx.args, cfg)
    inf_main(ns)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def eval(ctx: typer.Context, config: Path = typer.Option(None, help="YAML config")):
    cfg = load_config(config)
    ns = _parse_with_config(eval_parser, ctx.args, cfg)
    asyncio.run(eval_main(ns))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def analysis(ctx: typer.Context, config: Path = typer.Option(None, help="YAML config")):
    cfg = load_config(config)
    ns = _parse_with_config(analysis_parser, ctx.args, cfg)
    analysis_main(ns)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, config: Path = typer.Option(..., help="YAML config")):
    cfg = load_config(config)
    union = _combine_parsers(gp_parser(), inf_parser(), eval_parser(), analysis_parser(), read_parser())
    ns = union.parse_args(_config_to_args(union, cfg) + ctx.args)
    run_id = run_pipeline(vars(ns))
    typer.echo(f"Run ID: {run_id}")


@kg_app.command("data", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def kg_data(ctx: typer.Context, config: Path = typer.Option(..., help="YAML config")):
    cfg = load_config(config)
    union = _combine_parsers(kg_inf_parser(), kg_build_parser(), kg_c3_parser(), kg_combined_parser(), kg_filter_parser())
    ns = union.parse_args(_config_to_args(union, cfg) + ctx.args)
    kg_id = run_kg_pipeline(vars(ns), insert=False)
    typer.echo(f"KG ID: {kg_id}")


@kg_app.command("insert", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def kg_insert(ctx: typer.Context, config: Path = typer.Option(..., help="YAML config")):
    cfg = load_config(config)
    ns = _parse_with_config(kg_insert_parser, ctx.args, cfg)
    from ImProver.KG import insert_neo4j as kg_insert_mod
    kg_insert_mod.main(ns)


@kg_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def kg_run(ctx: typer.Context, config: Path = typer.Option(..., help="YAML config")):
    cfg = load_config(config)
    union = _combine_parsers(kg_inf_parser(), kg_build_parser(), kg_c3_parser(), kg_combined_parser(), kg_filter_parser(), kg_insert_parser())
    ns = union.parse_args(_config_to_args(union, cfg) + ctx.args)
    kg_id = run_kg_pipeline(vars(ns), insert=True)
    typer.echo(f"KG ID: {kg_id}")


if __name__ == "__main__":
    app()
