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


def getPromptsAux (targets_new : Array (CompilationStep × ConstantInfo)) (mod : Name) (python_cmd : String) (fileName: String) : IO (List (List TheoremData × Nat)) := do
  let rag_strings : Array Json ← do
    let items ← if targets_new.isEmpty then pure #[] else retrieve_batch_indep targets_new python_cmd
    let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
    pure x


  IO.println s!"==== Got {rag_strings.size} prompts from RAG ===="


  let mut outputs := []
  for (((cmd, ci), rag), target_idx) in (targets_new.zip rag_strings).zipIdx do
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

    outputs := ((mainData :: extracted_thms),target_idx) :: outputs

  return outputs
