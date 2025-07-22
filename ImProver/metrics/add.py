"""
all arguments:

name: required
system_prompt: required
score_fn : optional, default is null

sorry_ok? : optional, default is False
correctness_condition? : Optional, options are "none", "equal", "not_equal", and "compiles"

example_file: optional, default is null

metrics_dir: optional, default is ".metrics"

llm_metric: optional, default is false
metric_model: optional, default is null
rubric: optional, json, default is null

metrics_dir: optional, default is ".metrics"


this outputs a new folder in the metrics_dir with the name of the metric
and a config file that contains all the arguments used to create the metric. 

it also converts the metric example
file into a json file. need to modify the following files to handle this arbitrary metric format (a la /home/riyaza/eval_improver/improver/.prompts/.prompt_examples/prompt_to_template.py).

- readability.py needs to be refactored to handle llm metrics (must calculate + parse score according to rubric)
- eval_improver.lean should take in args for sorry_ok?, correctness_condition? as well as import the metric router file
- metric router should be a lean file that maps the name to the scoring function (that takes in a compilationStep and returns a float)
    - each scoring function should be in a standalone file in a dedicated folder, and the router should import all of them (ideally through a middle-man)
    - the add function should add the corresponding code to the metric router file. In a sense, this will be a double meta program.
- inference should get the new prompt and example_file from the config. which we then generate a json file for and use to come up with the standard examples format

also default creates the [name].lean file in the metrics_dir/name with the function spec, if path not provided via score_fn. Path must be in current repo.

then modifies the metrics_dir/router.lean file to include the new metric.
"""
import os
import json
import sys
import argparse
import subprocess


def create_metric(args):
    metric_path = os.path.join("metrics", args.name)
    os.makedirs(metric_path, exist_ok=True)


    #INIT EXAMPLE FILE

    # Convert example file to JSON if provided
    # Handle example file - create template if not provided
    if args.example_file is None:
        # Create a local Lean example template
        example_lean_path = os.path.join(metric_path, f"examples.lean")
        with open(example_lean_path, "w", encoding="utf-8") as ef:
            ef.write(f"""-- Auto-generated example template for {args.name}
-- TODO: Add + tag example theorems and proofs here

import ImProver.metrics.tagger

@[improver_example test, version unoptimized]
example : True := by
  sorry

@[improver_example test, version optimized]
example : True := by
  trivial

    """)
        args.example_file = example_lean_path

    # Convert example file path to Lean module
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    example_module = os.path.relpath(args.example_file, root_path).replace(os.path.sep, ".").replace(".lean", "")

    # Set output JSON path for extracted examples
    example_json_path = os.path.join(metric_path, f"examples.json")

    # Run lake exe get_examples to convert Lean examples to JSON
    try:
        cmd = [
            "lake", "exe", "get_examples", 
            example_module, 
            example_json_path,
            sys.executable
        ]
        subprocess.run(cmd, check=True)
        print(" ".join(cmd))
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to extract examples from {example_module}: {e}")
        # Create empty JSON file as fallback
        with open(example_json_path, "w", encoding="utf-8") as f:
            json.dump([], f)
            
    
    # Update metrics/examples.lean to import the example module
    metrics_examples_path = os.path.join("metrics", "examples.lean")
    import_line = f"import {example_module}"

    if os.path.exists(metrics_examples_path):
        with open(metrics_examples_path, "r", encoding="utf-8") as f:
            contents = f.read()
        
        if import_line not in contents:
            # Add import at the beginning of the file
            updated_contents = f"{import_line}\n{contents}"
            with open(metrics_examples_path, "w", encoding="utf-8") as f:
                f.write(updated_contents)
    else:
        # Create the file with the import
        with open(metrics_examples_path, "w", encoding="utf-8") as f:
            f.write(f"{import_line}\n")
            
    
    # INIT SCORING FUNCTION
    score_fn = args.score_fn
    # Create a Lean function file if no external score_fn path is provided
    if not args.llm_metric:
        if args.score_fn is None:
            lean_file_path = os.path.join(metric_path, f"{args.name}.lean")
            with open(lean_file_path, "w", encoding="utf-8") as lf:
                lf.write(f"""import TrainingData.Frontend
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
-- Auto-generated scoring function schema for {args.name}
def {args.name}_score (cs : CompilationStep) : IO Float :=
-- TODO: Implement custom score logic
pure 0.0
""")
            score_fn = lean_file_path
        
        score_module = score_fn.replace(os.path.sep, ".").replace(".lean", "")

        # Update the router.lean file to import and route this metric
        router_file = os.path.join("metrics", "router.lean")
        if not os.path.exists(router_file):
            with open(router_file, "w", encoding="utf-8") as rf:
                rf.write(f"""import {score_module}
open Lean Elab IO

def route_metric (name : String) (cs : CompilationStep) : IO Float := match name with
| "{args.name}" => {args.name}_score cs
| _ => pure 0.0
""")
        else:
            with open(router_file, "r", encoding="utf-8") as rf:
                contents = rf.read()

            add_contents = contents.replace("| _ => pure 0.0", f"| \"{args.name}\" => {args.name}_score cs\n| _ => pure 0.0")
            new_contents = f"""import {score_module}
{add_contents}
"""
            if f"| \"{args.name}\" => {args.name}_score cs" not in contents:
                with open(router_file, "w", encoding="utf-8") as rf:
                    rf.write(new_contents)
        # don't want redundant matches

    rubric = None
    if type(args.rubric) is str:
        try:
            rubric = json.loads(args.rubric)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON rubric: {e}")
    else:
        rubric = args.rubric
    
    
    config = {
        "name": args.name,
        "scoring": {
        "score_fn": score_fn,
        "sorry_ok": args.sorry_ok,
        "correctness_condition": args.correctness_condition,
        "minmax": args.minmax,
        "input_sorry": args.input_sorry,
        # "delta_type": args.delta_type,
        },
        "examples": {
        "example_file": args.example_file,
        "example_data": example_json_path,    
        },
        "prompts": {

        "system_prompt": args.system_prompt,
        "annotation_prompt": args.annotation_prompt,
        "context_prompt": args.context_prompt,
        "rag_prompt": args.rag_prompt,
        "example_prompt": args.example_prompt,
        "goal_state_prompt": args.goal_state_prompt,
        "file_context_prompt": args.file_context_prompt,
        },
        "llm": {


        "llm_metric": args.llm_metric,
        "metric_model": args.metric_model,
        "rubric": rubric
        }
    }
    with open(os.path.join(metric_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        

def get_parser():
    prompt_defaults = {
        "annotation_prompt": " A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response.",
    "context_prompt": " The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>.",
    "rag_prompt": " The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>.",
    "example_prompt": " Here are some examples of such optimization, as wrapped in <EXAMPLES>...</EXAMPLES>. Note that these examples are for illustrative purposes only and should not be copied directly. Instead, use them to understand the kind of optimization expected and apply similar techniques to the current theorem.",
    "goal_state_prompt": " Additionally, the initial goal state of the theorem has been provided to help you better understand the proof statement and ensure the correctness of your response. It is wrapped in <GOAL_STATE>...</GOAL_STATE>.",
    "file_context_prompt": " The file context of the theorem, i.e. the preceding definitions, theorems, etc., have been provided to help understand the context of the theorem and ensure the correctness of your response. It is wrapped in <FILE_CONTEXT>...</FILE_CONTEXT>, with each individual item wrapped in <ITEM>...</ITEM>."
    }
    
    parser = argparse.ArgumentParser(description="Create a metric configuration.")
    parser.add_argument("name", help="Name of the metric")
    parser.add_argument("system_prompt", help="System prompt")
    parser.add_argument("--score_fn", default=None, help="Optional score function")
    parser.add_argument("--sorry_ok", action="store_true", help="Allow sorry_ok")
    parser.add_argument("--correctness_condition", default="none", help="Condition for correctness")
    parser.add_argument("--example_file", default=None, help="Path to example file")
    parser.add_argument("--llm_metric", action="store_true", help="Use an LLM-based metric")
    parser.add_argument("--metric_model", default=None, help="Model name for LLM metric")
    parser.add_argument("--rubric", default=None, help="JSON rubric")

    parser.add_argument("--minmax", default="max", help="Minimize or maximize the metric score")
    parser.add_argument("--input_sorry", action="store_true", default=False, help="Send sorry'd proof as the input")
    # parser.add_argument("--delta_type", action="relative", default=False, help="How to calculate the delta between the old and new score: relative (%) or absolute (-).")
    
    parser.add_argument("--annotation_prompt", default=prompt_defaults['annotation_prompt'], help="annotation prompt")
    parser.add_argument("--context_prompt", default=prompt_defaults['context_prompt'], help="context prompt")
    parser.add_argument("--rag_prompt", default=prompt_defaults['rag_prompt'], help="rag prompt")
    parser.add_argument("--example_prompt", default=prompt_defaults['example_prompt'], help="example prompt")
    parser.add_argument("--goal_state_prompt", default=prompt_defaults['goal_state_prompt'], help="goal state prompt")
    parser.add_argument("--file_context_prompt", default=prompt_defaults['file_context_prompt'], help="file context prompt")
    return parser

def main(args):
   

    if args.llm_metric:
        if args.score_fn is not None:
            raise ValueError("When llm_metric is true, score_fn must be None.")
        if not args.metric_model:
            raise ValueError("When llm_metric is true, metric_model cannot be None.")
        if not args.rubric:
            raise ValueError("When llm_metric is true, rubric cannot be None.")
    else:
        if args.score_fn is None:
            # Prepare to auto-create a Lean file
            pass

    
    create_metric( args
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
