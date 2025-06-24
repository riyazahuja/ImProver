# ImProver Command Line Interface

The `improver` script provides a convenient way to run the main proof
optimisation and knowledge graph pipelines. Each command accepts a
YAML configuration file that specifies the required arguments for the
underlying scripts. The examples below assume you are in the repository
Add the repository root to your `PATH` or invoke it directly as `./improver`.
root.

## Proof evaluation

Run the full pipeline (prompt extraction, inference, evaluation and
analysis). If no `prompt_id` is provided in the configuration the prompts
will be generated automatically:

```bash
improver run all --config config.yaml
```

Individual stages can be run with the sub‑commands `prompts`,
`inference`, `eval` and `analysis`:

```bash
improver run inference --config config.yaml
```

The configuration file should contain the arguments of
`get_prompts.py`, `inference.py`, `eval_improver.py` and
`analysis.py`. A minimal example:

```yaml
dataset_path: path/to/dataset.json
metric: length
split: train
prompts_dir: .prompts
prompt_id: PROMPT_20240601  # optional
output_dir: .evals
cpus: 4
gpus: 1
```

`runID` is created automatically when running inference if not
supplied.

## Knowledge graph construction

Run every step and insert the final graph into Neo4j:

```bash
improver KG all --config kg.yaml
```

The `data` command performs all steps except the final Neo4j
insertion. Use `c2` to only build the class‑2 database and `c3` to run
the remaining stages (informalisation through filtering). Existing
class‑3 data can be inserted into Neo4j using `insert`.

A minimal `kg.yaml` might look like:

```yaml
dataset_path: path/to/dataset.json
KG_id: KG_20240601
split: train
KG_dir: .knowledge_graphs
cpus: 8
gpus: 1
```

