import ImProver.prompting.state_comments
import ImProver.prompting.context
import Cli
import ImProver.prompting.prompts
import ImProver.inference.inference
import ImProver.evaluation.eval
import ImProver.utils
import ImProver.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports



import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
-- import Compfiles

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true




def getPrompts (mod : Name) (metric : String) (outputDirectory : String) : IO Unit := do

  let fileName := (← findLean mod).toString
  -- let mut trajectories_json := []
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    let isThm? := match ci with
      | .thmInfo _ => true
      | _ => false

    let pf_env := cmd.after
    let ctx : Core.Context := {fileName := "", fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1
      | none => false



    if not isThm? || not isHuman then
      continue

    targets_new := targets_new.push (cmd, ci)

  -- IO.println s!"Found {targets_new.size} targets"

  let targets_with_prompts ← get_prompt_eval_batched mod metric targets_new


  let json_path := outputDirectory ++ "/" ++ metric ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  IO.println s!"Writing to {json_path}"
  -- let trajectories := Json.arr (trajectories_json.toArray)
  -- match json_path with
  -- | some path =>
  if not (← System.FilePath.pathExists json_path) then
    let parent := System.FilePath.parent json_path
    match parent with
    | some path =>
      IO.println path
      IO.FS.createDirAll path
    | none => pure ()


  IO.FS.writeFile json_path (targets_with_prompts.compress)
  -- | none => pure ()



def getPromptsCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let metric := args.positionalArg! "metric" |>.as! String
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let mod :Name := module


  getPrompts mod metric outputDirectory
  return 0


def get_prompts : Cmd := `[Cli|
  get_prompts VIA getPromptsCLI; ["0.0.1"]
"Generate prompts for ImProver."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    metric : String; "Metric to use for evaluation."
    outputDirectory : String; "Where to save the Json output."
]


def main (args : List String) : IO UInt32 :=
  get_prompts.validate args


#eval getPrompts `Mathlib.Logic.Hydra "length" "prompts"
