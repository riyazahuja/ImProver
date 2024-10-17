from benchmark.tools import *
from benchmark.extract import *

# OPENAI API KEY IS REQUIRED AS AN ENVVAR

# get our repo and files we want to look at, we'll do compfiles for this one:
# Make sure to first run python scripts/build.py --config configs/config_eq.json

methods = improver(modularity_metric())
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
    save_to_csv(data, "benchmark/data/demo_eq/equational_op_results_all2.csv")

# now do statistics on results
from benchmark.extract import *

experiments = [{"metric": "LENGTH"}, {"metric": "MODULARITY"}]

fp = "benchmark/data/demo_eq/equational_op_results_all.csv"
df = pd.read_csv(fp)
for exp in experiments:
    data = calculate_metrics(
        filter_data(df, **exp),
        minimax="MAX" if exp["metric"] == "MODULARITY" else "MIN",
    )
    print(data)
    print("")

# ^ you can just run this by itself on a csv to extract what you need.
# make sure to set minimax to "MAX" for modularity checking
# Also, make sure to modify the experiments for the variable
# parameters you're testing.
