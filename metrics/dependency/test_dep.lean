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
import metrics.dependency.dependency

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


def getPrompts (mod : Name) (theorems : List String): IO Unit := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  -- let scope_import := "import ImProver.get_prompts.where_with_end\n"
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  -- let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets.filter (fun (_, ci) => theorems.isEmpty || ci.name.toString ∈ theorems) do
    -- let isThm? := match ci with
    --   | .thmInfo _ => true
    --   | _ => false

    let pf_env := cmd.after--.env
    let ctx : Core.Context := {fileName := fileName, fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1 -- currently modified to return something that is not necessarily a theorem
      | none => false
    if --not isThm? ||
      not isHuman then
      continue

    -- let dependency_score ← dependency_score cmd
    let context ← get_context cmd ["axiom", "def", "theorem", "opaque", "quot", "inductive", "constructor", "recursor"] (some mod)
    let dependency_score :=
      let external_deps := context.filter (fun c =>
      match c.kind with
      | "theorem (internal)" => true
      | "theorem" => true
      | _ => false)
      external_deps.length |>.toFloat


    IO.println s!"{ci.name.toString} {dependency_score}"
    IO.println s!"{cmd.commandStateBefore.scopes.map (fun s => s.header)}"
    IO.println s!"{cmd.src.toString}"
    IO.println s!"{"\n----------\n".intercalate (context.map (fun c => s!">> [{c.name}]{c.kind} (from {c.module}):\n{c.text}"))}"
    IO.println "================================================"








def getPromptsCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName

  let mod :Name := module
  let theorems_raw : String := match args.flag? "theorems" with
  | some x => x |>.as! String
  | none => ""
  let theorems : List String := if theorems_raw.isEmpty then [] else theorems_raw.splitOn ","


  -- IO.println theorems
  getPrompts mod  theorems
  return 0


def get_prompts : Cmd := `[Cli|
  get_prompts VIA getPromptsCLI; ["0.0.1"]
"Generate prompts for ImProver."

  FLAGS:
    theorems : String; "List of theorems to include in the prompts, separated by \",\". If empty, all theorems in the module will be used."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."

  EXTENSIONS:
    defaultValues! #[("theorems", "")]
]


def main (args : List String) : IO UInt32 :=
  get_prompts.validate args




#eval main ["Carleson.ToMathlib.Annulus", "--theorems", "Set.EAnnulus.ci_eq_annulus"]
