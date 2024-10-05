from benchmark.tools import *
from benchmark.extract import *


# First define a metric that minimizes the number of rewrite statements:
def rewrite_metric():
    def count_rw(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        pf = [tac.tactic for tac in thm.proof]
        return sum(1 for tactic in pf if tactic.strip().startswith("rw"))

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites Lean 4 proofs to minimize the number of rewrite statements while ensuring their correctness. Specifically, you must generate a new proof of the given theorem that is both correct and contains as little "rw" tactics as possible, and of these rewrite tactics, they do as little rewriting as possible.""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it has less rewrite statements, while also being correct.""",
    )
    examples = (
        []
    )  # ideally should add a few here in a {"input":"content","output":"better content"} format

    return Metric(
        "REWRITE",
        [sys_prompt, user_prompt],
        examples,
        "MIN",
        score_fn=count_rw,
    )


# now we get our repo and files we want to look at, we'll do compfiles for this one:
# Make sure to first run python scripts/build.py --config CONFIG_FILE_PATH

repo = getRepo("compfiles", "configs/config_MIL.json")

files = {f.file_path: f for f in repo.files if type(f) == AnnotatedFile}
file = files["Compfiles/Imo2019P2.lean"]


# We now define the experiments we want to run
improver_experiments = improver(length_metric(), rewrite_metric())
baseline_experiments = baseline(length_metric(), rewrite_metric())


# If we want to see how expensive (in terms of API), we can get an estimate by:
cost = get_cost(file, improver_experiments + baseline_experiments)
print(f"Cost: {cost}")

# And now we can run either at a file-level or on a theorem level, saving after each (we can also do repo level)
baseline_file_path = "demo_baseline.csv"
improver_file_path = "demo_improver.csv"

baseline_data = benchmark_file(
    file, baseline_experiments, max_workers=5, show_progress=True
)
save_to_csv(baseline_data, path=baseline_file_path)

improver_data = []
for thm in file.theorems:
    improver_data.extend(
        benchmark_theorem(thm, improver_experiments, max_workers=1, show_progress=True)
    )
    save_to_csv(improver_data, path=improver_file_path)
# Note that improver is quite memory-hungry, so be careful trying large values of max_workers, it might cause a deadlock if it runs out of memory


# We can now extract and run statistics on the data

experiments = [
    {"n": n, "metric": metric} for n in [0, 5] for metric in ["LENGTH", "REWRITE"]
]

df = pd.concat([pd.read_csv(improver_file_path), pd.read_csv(baseline_file_path)])

data = [
    calculate_metrics(filter_data(basic_df, **method), minimax="MIN")
    for method in basic_methods
]
for i in range(len(data)):
    print(basic_methods[i])
    print("****")
    print(data[i])
    print("====================")

print()
best = best_method = get_best_method(
    basic_df,
    basic_methods,
    minimax="MIN",
)

best_method = get_best_method(
    basic_df,
    basic_methods,
    minimax="MIN",
)
print(f"BEST:\n{best_method}")

# and output a graph

data_dict = {str(basic_methods[i]): data[i] for i in range(len(basic_methods))}
plot_combined_chart(data_dict, minimax="MIN")
