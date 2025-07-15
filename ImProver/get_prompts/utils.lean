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
import ImProver.get_prompts.where_with_end

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true



def getNormalId (target : CompilationStep × ConstantInfo) (mod : Name): IO TheoremID := do
  let (cmd, ci) := target
  -- name : Name
  -- module : Name
  -- content : Option String := none
  -- compilationAlias : Option String := none
  -- isExtracted : Bool := false
  -- errorMsgs : Array String := #[]
  -- kind : String := "theorem"
  let msgs ← cmd.msgs.filterMapM (fun msg => do
        let m ← msg.data.toString
        if msg.severity != .error then
          return none
        return some m)
  let kind := getKind cmd.after.constants ci.name
  return {
    name := ci.name,
    module := mod,
    content := some cmd.src.toString,
    compilationAlias := some cmd.src.toString,
    isExtracted := false,
    errorMsgs := msgs.toArray,
    kind := kind
  }

open Lean.Elab.Command



def getScopes (cmd : CompilationStep) (fileName : String) : IO (String × String) := do

  let ctx : Command.Context := {
    fileName := fileName,
    fileMap := cmd.src.toString.toFileMap,
    tacticCache? := none,
    snap? := none,
    cancelTk? := none
  }
  let state := cmd.commandStateBefore

  let ((prescopes_raw, postscopes_raw), _) ← CommandElabM.toIO whereWithEndCore ctx state
  let prescopes ← prescopes_raw.toString
  let postscopes ← postscopes_raw.toString
  IO.println s!"==== Got scopes ===="
  IO.println s!"Prescopes: {prescopes}"
  IO.println s!"Postscopes: {postscopes}"
  return (prescopes, postscopes)



def getPromptsAux (targets_new : Array (CompilationStep × ConstantInfo)) (mod : Name) (python_cmd : String) (fileName: String) : IO (List (List TheoremData × Nat)) := do
  let rag_strings : Array Json ← do
    let items ← if targets_new.isEmpty then pure #[] else retrieve_batch_indep targets_new python_cmd
    let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
    pure x


  IO.println s!"==== Got {rag_strings.size} prompts from RAG ===="

  -- let targets_new_with_id : Array (CompilationStep × ConstantInfo × (Array (CompilationStep × ConstantInfo))) :=
  --   targets_new.mapIdx (fun i target => (target.1, target.2, targets_new.extract 0 i))
  let targets_new_with_id : Array (CompilationStep × ConstantInfo × TheoremID) ←
    targets_new.mapM (fun target => do
      let id ← getNormalId target mod
      return (target.1, target.2, id))

  let targets_new_cumulative : Array (CompilationStep × ConstantInfo × TheoremID × (Array TheoremID)) :=
    targets_new_with_id.mapIdx (fun i (cmd, ci, id) =>
      let deps := targets_new_with_id.extract 0 i |>.map (fun (_, _, dep_id) => dep_id)
      (cmd, ci, id, deps))



  let mut outputs := []
  for (((cmd, ci, id, prev_ids), rag), target_idx) in (targets_new_cumulative.zip rag_strings).zipIdx do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"
    -- eventually want annotation on partial proofs, but for now, ignore
    let annotation_string : String ← insert_state_comments cmd


    -- let context_string : Json ← do
    --     let context ← get_context cmd
    --     pure <| Json.arr <| context.map (fun c : ExternalContext => Json.mkObj [
    --         ("name", Json.str c.name.toString),
    --         ("context_item_type", Json.str c.kind),
    --         ("content", Json.str c.text)
    --       ]) |>.toArray
    let (prescopes, postscopes) ← getScopes cmd fileName

    let pfAsSorry := proofAsSorry cmd |>.getD ""

    let initialGoal ←  getInitialProofState2 cmd


    let C1_raw ← get_context cmd --["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremID:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text, kind:= ctx.kind, compilationAlias := some ctx.text})

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

      -- we don't get all the fancy data for splits bc we are lazy...
      let split_data : TheoremData :=
        { id := split_thm,
          prescopes := prescopes,
          postscopes := postscopes,
          C0_dependencies := prev_ids, --idk yet whether to keep this
          C1_dependencies := deps.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text}) |>.toArray,
          C2_dependencies := #[]
        }
      extracted_thms := split_data :: extracted_thms

    -- let id : TheoremID := {name := ci.name, module := mod, content := some srcCommand, compilationAlias := some srcCommand}
    -- let id ← getNormalId (cmd, ci) mod


    let mainData : TheoremData :=
      { id := id,
        C0_dependencies := prev_ids,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
        annotation := annotation_string,
        content_sorry := pfAsSorry,
        goal := initialGoal,
        rag := rag,
        prescopes := prescopes,
        postscopes := postscopes
        }

    outputs := ((mainData :: extracted_thms),target_idx) :: outputs

  return outputs
