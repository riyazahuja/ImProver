import sys
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from models.structures import *
from models.prompt import *
from evaluate.build_prooftree import *
import pandas as pd
import time
from models.rag import *
import itertools
from tqdm import tqdm
from multiprocessing import cpu_count


def process_instance(thm: AnnotatedTheorem, method):
    start_time = time.time()
    fn, metric, kwargs = method
    og_correct, og_messages, _ = eval_correctness(thm)
    og_score = None
    if og_correct and metric.score_fn != None:
        og_score = metric.score(thm)
    output_thm = fn(thm, metric, **kwargs)

    new_correct, new_messages, output_anno_thm = eval_correctness(output_thm)
    processing_time = time.time() - start_time
    new_score = None
    if new_correct and metric.score_fn != None:
        new_score = metric.score(output_anno_thm)
    if new_correct and og_correct:
        delta = metric.metric(thm, output_anno_thm)
    else:
        delta = None
    # print(
    #     f"=============\n(METHOD : {method[2].get('examples',0)})\nOUTPUT: \n{output_thm}\n\nANNOT:\n{output_anno_thm}\n=============="
    # )

    og_raw = parseTheorem(thm, context=False)
    new_raw = parseTheorem(output_thm, context=False)

    def parse_msg(message):
        return f"{message.content}\n\tat: {message.message_src}"

    og_errors = "\n".join(
        [
            parse_msg(msg)
            for msg in og_messages
            if msg.severity == "error"
            or (msg.severity == "warning" and "sorry" in msg.content)
        ]
    )
    new_errors = "\n".join(
        [
            parse_msg(msg)
            for msg in new_messages
            if msg.severity == "error"
            or (msg.severity == "warning" and "sorry" in msg.content)
        ]
    )

    return {
        "repo": thm.src,
        "file": thm.leanFile,
        "decl": thm.decl,
        "method": fn.__name__,
        "n": kwargs.get("n", None) if fn.__name__ != "prompt_structured" else None,
        "metric": metric.name,
        "model": kwargs.get("model", "gpt-4-turbo"),
        "annotation": kwargs.get("annotation", True),
        "syntax_search": kwargs.get("syntax_search", False),
        "mathlib_search": kwargs.get("mathlib_search", False),
        "examples": kwargs.get("examples", 0),
        "improved_context": kwargs.get("improved_context", False),
        "og_correct": og_correct,
        "og_errors": og_errors,
        "og_score": og_score,
        "new_correct": new_correct,
        "new_errors": new_errors,
        "new_score": new_score,
        "delta": delta,
        "og_raw": og_raw,
        "new_raw": new_raw,
        "time": processing_time,
    }


def process_instances(instances, max_workers=None, show_progress=False):
    if max_workers is None:
        max_workers = len(instances)

    if show_progress:
        with tqdm(total=len(instances), desc="Instances: ") as pbar:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(instances))
            ) as executor:
                futures = [
                    executor.submit(process_instance, i[0], i[1]) for i in instances
                ]
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    else:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(instances))
        ) as executor:
            futures = [executor.submit(process_instance, i[0], i[1]) for i in instances]
    data = [future.result(timeout=1) for future in futures]
    return data


def benchmark_theorem(
    thm: AnnotatedTheorem, methods, max_workers=None, show_progress=False
):
    instances = [(thm, m) for m in methods]
    return process_instances(
        instances, max_workers=max_workers, show_progress=show_progress
    )


def benchmark_file(
    file: AnnotatedFile,
    methods,
    max_workers=None,
    show_progress=False,
):
    thms = file.theorems

    instances = [(t, m) for t in thms for m in methods]
    return process_instances(
        instances, max_workers=max_workers, show_progress=show_progress
    )


def benchmark_repo(repo: Repo, methods, max_workers=None, show_progress=False):
    anno_files = [f for f in repo.files if type(f) == AnnotatedFile]
    thms = []
    for f in anno_files:
        thms.extend(f.theorems)

    instances = [(t, m) for t in thms for m in methods]
    return process_instances(
        instances, max_workers=max_workers, show_progress=show_progress
    )


def save_to_csv(data, path="data.csv"):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path, index=False)


def get_methods(
    fn=[prompt_structured],
    metric=[length_metric()],
    annotation=[False],
    model=["gpt-4-turbo"],
    n=[1],
    syntax_search=[False],
    mathlib_search=[False],
    examples=[0],
    improved_context=[False],
):
    dl = [
        fn,
        annotation,
        model,
        n,
        syntax_search,
        mathlib_search,
        metric,
        examples,
        improved_context,
    ]
    prod = list(itertools.product(*dl))
    return [
        (
            i[0],
            i[6],
            {
                "annotation": i[1],
                "model": i[2],
                "n": i[3],
                "syntax_search": i[4],
                "mathlib_search": i[5],
                "examples": i[7],
                "improved_context": i[8],
            },
        )
        for i in prod
    ]


def baseline(*metrics):
    return get_methods(
        model=["gpt-4o"],
        fn=[prompt_basic],
        metric=metrics,
    )


def improver(*metrics):
    return get_methods(
        model=["gpt-4o"],
        fn=[refinement(best_of_n_n(prompt_flat, 3, max_workers=3), keep_best=True)],
        n=[5],
        annotation=[True],
        examples=[10],
        metric=metrics,
        syntax_search=[True],
        mathlib_search=[True],
        improved_context=[True],
    )


def get_cost(obj, methods):
    price_pt = {
        "gpt-4o-mini": (0.150 / 1000000, 0.600 / 1000000),
        "gpt-4o": (5 / 1000000, 15 / 1000000),
        "gpt-4-turbo": (10 / 1000000, 30 / 1000000),
        "gpt-3.5-turbo-0125": (0.5 / 1000000, 1.5 / 1000000),
    }

    if type(obj) == Repo:
        anno_files = [f for f in obj.files if type(f) == AnnotatedFile]
        thms = [thm for f in anno_files for thm in f.theorems]
    elif type(obj) == AnnotatedFile:
        thms = obj.theorems
    elif type(obj) == AnnotatedTheorem:
        thms = [obj]
    else:
        raise ValueError(f"uhoh: type is \n{type(obj)}")

    def get_instance_cost(obj, method):
        model = method[2].get("model", "gpt-4-turbo")
        fn, metric, kwargs = method
        inp_tok = fn(obj, metric, **kwargs, token=True)
        encoding = tiktoken.encoding_for_model(model)
        output_tok = len(encoding.encode(parseTheorem(obj, context=False)))
        inp_cost, out_cost = price_pt[model]
        price = inp_tok * inp_cost + output_tok * out_cost
        return price

    with tqdm(total=len(thms) * len(methods), desc="instances: ") as pbar:
        with ThreadPoolExecutor(
            max_workers=min(24, len(methods) * len(thms))
        ) as executor:
            futures = [
                executor.submit(get_instance_cost, thm, method)
                for method in methods
                for thm in thms
            ]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
    return sum(future.result() for future in futures)


def no_errors(thms):
    msgs = []
    for thm in thms:
        msgs.extend(thm.messages)
    errors = sum(1 for msg in msgs if msg.severity == "error") + sum(
        1 for msg in msgs if msg.severity == "warning" and "sorry" in msg.content
    )
    return errors == 0


if __name__ == "__main__":

    # setup
    methods = improver(length_metric(), modularity_metric())
    repo = getRepo("equational_theories", "configs/config_eq.json")
    files = {file.file_path: file for file in repo.files if type(file) == AnnotatedFile}
    # print(files.keys())
    f = files["equational_theories/MagmaOp.lean"]

    # estimate the cost
    cost = get_cost(f, methods)
    print(f"${cost}")

    # actually run the tool
    data = []
    for t in f.theorems:
        data.extend(
            benchmark_theorem(
                t,
                methods,
                max_workers=2,
                show_progress=True,
            )
        )
        save_to_csv(data, path=f"equational_op_results.csv")

    # now do statistics on results
    from benchmark.extract import *

    fp = "equational_op_results.csv"
    df = pd.read_csv(fp)
    data = calculate_metrics(filter_data(df), minimax="MIN")
    print(data)
