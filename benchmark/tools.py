import sys
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from models.structures import *
from models.prompt import *
import pandas as pd
import time
from models.rag import *
import itertools
from tqdm import tqdm
from multiprocessing import cpu_count


def extract_data(thm: Theorem, method, trajectory_position):
    fn, metric, kwargs = method
    correct, messages = eval_correctness(thm)

    score = None
    if correct and metric.score_fn != None:
        score = metric.score(thm)

    raw = parseTheorem(thm, context=False)

    errors = "\n".join(
        [
            msg
            for msg in messages
            if "error" in msg or ("warning" in msg and "sorry" in msg)
        ]
    )

    return {
        "repo": thm.repo.name,
        "file": thm.file.get_path(),
        "decl": thm.get_decl(),
        "method": fn.__name__,
        "n": kwargs.get("n", None) if fn.__name__ != "prompt_structured" else None,
        "metric": metric.name,
        "trajectory_position": trajectory_position,
        "model": kwargs.get("model", "gpt-4-turbo"),
        "annotation": kwargs.get("annotation", True),
        "syntax_search": kwargs.get("syntax_search", False),
        "mathlib_search": kwargs.get("mathlib_search", False),
        "examples": kwargs.get("examples", 0),
        "improved_context": kwargs.get("improved_context", False),
        "correct": correct,
        "errors": errors,
        "score": score,
        "raw": raw,
    }


def parse_trajectories(trajectories, method, position="/"):
    if (
        type(trajectories) == tuple
        and trajectories[0] == "BoN"
        and type(trajectories[1]) == list
    ):
        # print("BON")
        output = []
        for subtrajectory in trajectories[1]:
            output += parse_trajectories(
                subtrajectory, method, position + f"BoN({len(trajectories[1])})/"
            )
        return output
    if (
        type(trajectories) == tuple
        and trajectories[0] == "refine"
        and type(trajectories[1]) == list
    ):
        # print("refine")
        output = []
        for i, subtrajectory in enumerate(trajectories[1]):
            output += parse_trajectories(
                subtrajectory,
                method,
                position + f"refine({len(trajectories[1])}; curr = {i})/",
            )
        return output

    elif type(trajectories) == Theorem:
        # print("thm")
        return [extract_data(trajectories, method, position)]
    else:
        raise ValueError(
            f"Invalid trajectories type: {type(trajectories)}, {trajectories}"
        )


def process_instance(thm: Theorem, method, output_trajectories=False):
    start_time = time.time()
    fn, metric, kwargs = method
    og_correct, og_messages = eval_correctness(thm)
    og_score = None
    if og_correct and metric.score_fn != None:
        og_score = metric.score(thm)

    output_thm, trajectories = fn(thm, metric, **kwargs)

    new_correct, new_messages = eval_correctness(output_thm)
    processing_time = time.time() - start_time
    new_score = None
    if new_correct and metric.score_fn != None:
        new_score = metric.score(output_thm)
    if new_correct and og_correct:
        delta = metric.metric(thm, output_thm)
    else:
        delta = None

    og_raw = parseTheorem(thm, context=False)
    new_raw = parseTheorem(output_thm, context=False)

    og_errors = "\n".join(
        [
            msg
            for msg in og_messages
            if "error" in msg or ("warning" in msg and "sorry" in msg)
        ]
    )
    new_errors = "\n".join(
        [
            msg
            for msg in new_messages
            if "error" in msg or ("warning" in msg and "sorry" in msg)
        ]
    )

    return {
        "repo": thm.repo.name,
        "file": thm.file.get_path(),
        "decl": thm.get_decl(),
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
    }, (parse_trajectories(trajectories, method) if output_trajectories else None)


def process_instances(
    instances, max_workers=None, show_progress=False, output_trajectories=False
):
    if max_workers is None:
        max_workers = len(instances)

    if show_progress:
        with tqdm(total=len(instances), desc="Instances: ") as pbar:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(instances))
            ) as executor:
                futures = [
                    executor.submit(
                        process_instance,
                        i[0],
                        i[1],
                        output_trajectories=output_trajectories,
                    )
                    for i in instances
                ]
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    else:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(instances))
        ) as executor:
            futures = [
                executor.submit(
                    process_instance,
                    i[0],
                    i[1],
                    output_trajectories=output_trajectories,
                )
                for i in instances
            ]
    data_raw = [future.result(timeout=1) for future in futures]

    data = [item for item, traj in data_raw]
    if output_trajectories:
        trajectories = []
        for item, traj in data_raw:
            if traj != None:
                trajectories += traj
    else:
        trajectories = None

    return data, trajectories


def benchmark_theorem(
    thm: Theorem,
    methods,
    max_workers=None,
    show_progress=False,
    output_trajectories=False,
):
    instances = [(thm, m) for m in methods]
    return process_instances(
        instances,
        max_workers=max_workers,
        show_progress=show_progress,
        output_trajectories=output_trajectories,
    )


def benchmark_file(
    file: File,
    methods,
    max_workers=None,
    show_progress=False,
    output_trajectories=False,
):
    thms = file.theorems

    instances = [(t, m) for t in thms for m in methods]
    return process_instances(
        instances,
        max_workers=max_workers,
        show_progress=show_progress,
        output_trajectories=output_trajectories,
    )


# def benchmark_repo(
#     repo: Repo,
#     methods,
#     max_workers=None,
#     show_progress=False,
#     output_trajectories=False,
# ):
#     anno_files = [f for f in repo.files]
#     thms = []
#     for f in anno_files:
#         thms.extend(f.theorems)

#     instances = [(t, m) for t in thms for m in methods]
#     return process_instances(
#         instances,
#         max_workers=max_workers,
#         show_progress=show_progress,
#         output_trajectories=output_trajectories,
#     )


def save_to_csv(data, path="data.csv"):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path, index=False)


def get_methods(
    fn=[prompt_flat],
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
        fn=[refinement(best_of_n_n(prompt_basic, 3, max_workers=1), keep_best=True)],
        n=[5],
        annotation=[True],
        # examples=[10],
        metric=metrics,
        # syntax_search=[True],
        # mathlib_search=[True],
    )


def get_cost(obj, methods):
    price_pt = {
        "gpt-4o-mini": (0.150 / 1000000, 0.600 / 1000000),
        "gpt-4o": (5 / 1000000, 15 / 1000000),
        "gpt-4-turbo": (10 / 1000000, 30 / 1000000),
        "gpt-3.5-turbo-0125": (0.5 / 1000000, 1.5 / 1000000),
    }

    if type(obj) == File:
        thms = obj.theorems
    elif type(obj) == Theorem:
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

    methods = improver(length_metric())

    repo = Repo.from_config("configs/config_mathlib.json")
    files = {f.path: f for f in repo.get_files(calculate_modules=False)}

    file = files["Mathlib/Logic/Basic.lean"]
    thms = file.get_theorems()
    d = []

    t = []
    for thm in thms:
        data, traj = benchmark_theorem(
            thm, methods, max_workers=1, show_progress=True, output_trajectories=True
        )
        d.extend(data)
        t.extend(traj)
        save_to_csv(d, "benchmark/data/data_test.csv")
        save_to_csv(t, "benchmark/data/traj_test.csv")
