# ntp-training-data

This repository is a modified version of Scott Morrison's [lean-training-data](https://github.com/semorrison/lean-training-data).


We provide tools for extracting training data based on Lean source code. 


### Running the new tools
To run the full pipeline on all repositories in `configs/config.json`:
```
python scripts/run_multiple.py --cwd {filepath_of_this_repo}
```

To run the pipeline on an individual repository:
```
python scripts/run_pipeline.py --cwd {filepath_of_this_repo}
```
By default, this extracts training data from the Mathlib version specified in `lakefile.lean`. 

On a Macbook Pro (M3 Max, 14 CPU) it takes around 2 hours.

```



To run a tool individually, use `lake exe <tool>`. \
The `run_pipeline.py` script uses Python to call tools in this way and organize the resulting files.

## New tools:
### `training_data`

This produces a `.jsonl` file where each line is an example of the following form:
```json
{
   "state": "{tactic state}",
   "nextTactic" : "{pretty-printed next tactic}",
   "srcUpToTactic" : "{source code in the file up to the tactic invocation}",
   "declUpToTactic" : "{source code in the declaration up to the tactic invocation}",
   "decl": "{declaration without proof (e.g., statement of a theorem)}",
   "declId": "{unique identifier of the declaration}"
}
```




## Usage (from `lean-training-data`)

* Install [`elan`](https://github.com/leanprover/elan) by running

```shell
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```
* Run `lake exe cache get` (this downloads precompiled binaries for `Mathlib`).
* Run `lake build`
* Run `lake exe <tool>`, where `<tool>` is one of the programs documented below.

