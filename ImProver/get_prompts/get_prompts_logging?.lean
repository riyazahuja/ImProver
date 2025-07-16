import ImProver.online.prompting.state_comments
import ImProver.online.prompting.context
import Cli
import ImProver.online.prompting.prompts
import ImProver.online.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports
import TrainingData.TreeParser
import TrainingData.ExtractGoal
import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import ImProver.online.c2
import ImProver.get_prompts.utils
import ImProver.get_prompts.where_with_end
import Lean.Elab.Command


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


def getPrompts (mod : Name) (outputDirectory : String) (python_cmd : String) (theorems : List String): IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  -- let scope_import := "import ImProver.get_prompts.where_with_end\n"
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets do
    -- let isThm? := match ci with
    --   | .thmInfo _ => true
    --   | _ => false

    let pf_env := cmd.after
    let ctx : Core.Context := {fileName := "", fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1 -- currently modified to return something that is not necessarily a theorem
      | none => false
    if --not isThm? ||
      not isHuman then
      continue


    let curr_name_variants :=
      let fullName := ci.name.toString
      let nameParts := fullName.splitOn "."
      let rec buildVariants (remaining : List String) (acc : List String) :=
        match remaining with
        | [] => acc
        | _ :: rest =>
          let currVariant := ".".intercalate remaining
          buildVariants rest (currVariant :: acc)
      buildVariants nameParts []

    let included? := curr_name_variants.map (fun n => theorems.contains n) |>.any id
    -- IO.println s!"Checking {ci.name.toString} against {theorems} => {included?}"
    -- IO.println s!"Current name variants: {curr_name_variants}"
    -- IO.println ""
    if (not theorems.isEmpty && not included?) then
      continue
    targets_new := targets_new.push (cmd, ci)

  IO.println s!"==== Got {targets_new.size} targets from {mod.toString} ===="

  let outputs_raw ← getPromptsAux targets_new mod python_cmd fileName
  let outputs := outputs_raw.map (fun x => x.1) |>.flatten

  if outputs.length != targets_new.size then
    IO.println s!"Warning: {outputs.length} outputs generated, but {targets_new.size} targets were processed. Some targets may not have produced valid prompts."
    return 1

  let json_data := Json.arr <| outputs.toArray.map (ToJson.toJson)

  let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  IO.println s!"Writing to {json_path}"

  if not (← System.FilePath.pathExists json_path) then
    let parent := System.FilePath.parent json_path
    match parent with
    | some path =>
      IO.println path
      IO.FS.createDirAll path
    | none => return 1



  IO.FS.writeFile json_path (ToString.toString json_data)
  return 0




def getPromptsCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let python_cmd := args.positionalArg! "pythonCommand" |>.as! String
  let mod :Name := module
  let theorems_raw : String := match args.flag? "theorems" with
  | some x => x |>.as! String
  | none => ""
  let theorems : List String := if theorems_raw.isEmpty then [] else theorems_raw.splitOn ","


  -- IO.println theorems
  getPrompts mod outputDirectory python_cmd theorems




def get_prompts : Cmd := `[Cli|
  get_prompts VIA getPromptsCLI; ["0.0.1"]
"Generate prompts for ImProver."

  FLAGS:
    theorems : String; "List of theorems to include in the prompts, separated by \",\". If empty, all theorems in the module will be used."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    outputDirectory : String; "Where to save the Json output."
    pythonCommand : String; "Path to python executable."

  EXTENSIONS:
    defaultValues! #[("theorems", "")]
]


def main (args : List String) : IO UInt32 :=
  get_prompts.validate args
