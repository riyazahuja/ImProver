import ImProver.prompting.state_comments
import ImProver.prompting.context
import Cli
-- import ImProver.prompting.prompts
-- import ImProver.inference.inference
-- import ImProver.evaluation.eval
-- import ImProver.utils
-- import ImProver.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports
import ImProver.ProofTree.getPfTree

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
-- import Compfiles

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


structure TheoremData where
  name : Name
  module : Name
  text : Option String := none
  C1_dependencies : Array TheoremData := #[]
  C2_dependencies : Array TheoremData := #[]
  isExtracted : Bool := false
  -- fromSrc : Bool := false
  deriving Inhabited, ToJson, FromJson, Repr


def getKG (mod : Name) (outputDirectory : String): IO Unit := do
  searchPathRef.set compile_time_search_path%
  IO.println mod.toString
  let fileName := (← findLean mod).toString
  -- let mut trajectories_json := []
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets do
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
  let mut outputs := []
  for (cmd, ci) in targets_new do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"

    let C1_raw ← get_context cmd ["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremData:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, text := some ctx.text})



    let C2_raw ← splitC2 fileName cmd

    let mut split_data : List TheoremData := []
    let mut C2_dependencies : List TheoremData := []

    for ((thm, deps),idx) in C2_raw.zipIdx do
      let inner : TheoremData :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          text := thm,
          isExtracted := true}
      C2_dependencies := inner :: C2_dependencies

      let deps' : List TheoremData := deps.map (fun ctx => {name := ctx.name, module := ctx.module, text := some ctx.text})
      let outer : TheoremData :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          text := some cmd.src.toString,
          C1_dependencies := deps'.toArray,
          isExtracted := true}
      split_data := outer :: split_data


    let mainData : TheoremData :=
      {name := ci.name,
        module := mod,
        text := some cmd.src.toString,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
        isExtracted := false}

    outputs := mainData :: split_data ++ outputs


  let json_data := Json.arr <| outputs.toArray.map (ToJson.toJson)

  let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  IO.println s!"Writing to {json_path}"

  if not (← System.FilePath.pathExists json_path) then
    let parent := System.FilePath.parent json_path
    match parent with
    | some path =>
      IO.println path
      IO.FS.createDirAll path
    | none => pure ()



  IO.FS.writeFile json_path (ToString.toString json_data)




def getKGCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let mod :Name := module


  getKG mod outputDirectory
  return 0


def get_KG : Cmd := `[Cli|
  get_KG VIA getKGCLI; ["0.0.1"]
"Generate C1 KG."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    outputDirectory : String; "Where to save the Json output."
]


def main (args : List String) : IO UInt32 :=
  get_KG.validate args




-- #eval getKG `PFR.BoundingMutual "C1Graph"

-- #eval getPrompts `MIL.C07_Hierarchies.solutions.Solutions_S01_Basics "length" "temp" "prompt_examples"
