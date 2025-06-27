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


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


def getPrompts (mod : Name) (outputDirectory : String) (python_cmd : String) (theorems : List String): IO Unit := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
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

    if (not theorems.isEmpty && not included?) then
      continue
    targets_new := targets_new.push (cmd, ci)

  IO.println s!"==== Got {targets_new.size} targets from {mod.toString} ===="

  let rag_strings : Array Json ← do
      let items ← retrieve_batch_indep targets_new python_cmd
      let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
      pure x


  IO.println s!"==== Got {rag_strings.size} prompts from RAG ===="


  let mut outputs := []
  for ((cmd, ci), rag) in targets_new.zip rag_strings do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"



    let srcCommand := cmd.src.toString

    -- eventually want annotation on partial proofs, but for now, ignore
    let annotation_string : String ← insert_state_comments cmd


    -- let context_string : Json ← do
    --     let context ← get_context cmd
    --     pure <| Json.arr <| context.map (fun c : ExternalContext => Json.mkObj [
    --         ("name", Json.str c.name.toString),
    --         ("context_item_type", Json.str c.kind),
    --         ("content", Json.str c.text)
    --       ]) |>.toArray

    let pfAsSorry := proofAsSorry cmd |>.getD ""

    let initialGoal ←  getInitialProofState2 cmd


    let C1_raw ← get_context cmd --["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremID:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text, kind:= ctx.kind})

    let C2_raw ← splitC2 fileName cmd "spawned"

    let mut extracted_thms : List TheoremData := []
    let mut C2_dependencies : List TheoremID := []
    -- errors = none means didn't compile, some [] means no errors, some [errors] means there were errors
    for ((pp_thm, stx_thm, deps, errors), idx) in C2_raw.zipIdx do

      let split_thm : TheoremID :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          content := pp_thm,
          compilationAlias := stx_thm,
          isExtracted := true,
          errorMsgs := errors.toArray
        }

      C2_dependencies := split_thm :: C2_dependencies

      let split_data : TheoremData :=
        { id := split_thm,
          C1_dependencies := deps.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text}) |>.toArray,
          C2_dependencies := #[]
        }
      extracted_thms := split_data :: extracted_thms

    let id : TheoremID := {name := ci.name, module := mod, content := some srcCommand, compilationAlias := some srcCommand}

    let mainData : TheoremData :=
      { id := id,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
        annotation := annotation_string,
        content_sorry := pfAsSorry,
        goal := initialGoal,
        rag := rag
        }

    outputs := mainData :: extracted_thms ++ outputs


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





  -- -- let prompt_data : Array (ConstantInfo × Json × Json × Json × Json) ← cmds_ci.mapM (fun (cmd,(ci : ConstantInfo)) => do

  -- --   return (ci, Json.str srcCommand, Json.str pfAsSorry, Json.str annotation_string, context_string)
  -- -- )

  -- IO.println s!"==== GOT {prompt_data.size} prompts from {mod}!!! ===="

  -- let rag_strings : Array Json ← do
  --     let items ← retrieve_batch_indep cmds_ci python_cmd
  --     let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
  --     pure x
  -- IO.println s!"==== GOT {rag_strings.size} prompts from RAG!!! ===="


  -- let prompt_data := prompt_data.zip rag_strings |>.map (fun ((ci,srcCommand, pfAsSorry, annotation_string,context_string),rag_string) =>
  --   let data := Json.mkObj [
  --     -- ("system", Json.str main_prompt),
  --     -- ("example_prompt", Json.str example_prompt),
  --     -- ("examples", example_json),
  --     -- ("context_prompt", Json.str context_prompt),
  --     ("context", context_string),
  --     -- ("rag_prompt", Json.str rag_prompt),
  --     ("rag", rag_string),
  --     -- ("annotation_prompt", Json.str annotation_prompt),
  --     ("annotation", annotation_string),
  --     ("current", srcCommand),
  --     ("current_sorry", pfAsSorry),
  --     ]
  --   ((ci : ConstantInfo).name.toString, data)
  --   )

  -- pure <| Json.mkObj prompt_data.toList














  -- let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  -- IO.println s!"Writing to {json_path}"
  -- -- let trajectories := Json.arr (trajectories_json.toArray)
  -- -- match json_path with
  -- -- | some path =>
  -- if not (← System.FilePath.pathExists json_path) then
  --   let parent := System.FilePath.parent json_path
  --   match parent with
  --   | some path =>
  --     IO.println path
  --     IO.FS.createDirAll path
  --   | none => pure ()

  -- IO.println s!"Path exists, now writing:\n{targets_with_prompts}"

  -- IO.FS.writeFile json_path (targets_with_prompts.compress)
  -- -- | none => pure ()






def getPromptsCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let python_cmd := args.positionalArg! "pythonCommand" |>.as! String
  let mod :Name := module
  let theorems_raw : String := args.positionalArg! "theorems" |>.as! String
  let theorems : List String := if theorems_raw.isEmpty then [] else theorems_raw.splitOn ","



  getPrompts mod outputDirectory python_cmd theorems
  return 0


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
