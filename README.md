# ntp-training-data

This repository is a modified version of Scott Morrison's [lean-training-data](https://github.com/semorrison/lean-training-data).


We provide tools for extracting training data based on Lean source code, and for creating instruction-tuning data for language models.


## Running extraction
To run the full pipeline on all repositories in `configs/config.json`:
```
python scripts/extract_repos.py --cwd {filepath_of_this_repo}
```

On a Macbook Pro (M3 Max, 14 CPU) it takes around 2 hours to run the extractions on mathlib.


To run a tool individually, use `lake exe <tool>`. \
The `run_pipeline.py` script uses Python to call tools in this way and organize the resulting files.



#### Extraction tools:
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

### `full_proof_training_data`

This produces a `.jsonl` file where each line is an example of the following form:
```json
{
   "srcUpToDecl":"{source code in the file up to the declaration}",
   "decl": "{declaration without proof (e.g., statement of a theorem)}",
   "declId": "{unique identifier of the declaration}",
   "proof":"{proof}"
}
```

### `state_comments`

This produces Lean source files with proof states interleaved as comments after each tactic.

## Running instruction tuning data generation
After extraction, you can generate various forms of (prompt, completion) examples for fine-tuning language models.

To do so, run:
```
python scripts/instruction_tuning.py --prompt context_state_tactic
```
See `python scripts/instruction_tuning.py -h` for other options for `--prompt` or other settings.

The prompt includes a natural language description of the task, commonly referred to as an "instruction" (hence the name instruction tuning data).


## Other setup docs from `lean-training-data`

You may find these useful during setup.

* Install [`elan`](https://github.com/leanprover/elan) by running

```shell
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```
* Run `lake exe cache get` (this downloads precompiled binaries for `Mathlib`).
* Run `lake build`
* Run `lake exe <tool>`, where `<tool>` is one of the programs documented below.

