
import Cli
import ImProver.prompting.state_comments
import ImProver.prompting.context
import ImProver.utils
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
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





def initialize_retrieval (config : ImProverConfig) (step: CompilationStep) : IO ImProverConfig := do
  let env : NameMap (Array Name) := step.after.importGraph
  let nodes := env.toList.map (fun (n, _) => n) |>.eraseDup
  let output : ImProverConfig := {
    config with
    retrievalFilter := nodes
  }
  return output

def retrieve (step : CompilationStep) (config: ImProverConfig) : IO (List String) := do
  let query : String ← insert_state_comments step

  let data : Json := Json.mkObj
    [("query", Json.str query),
      ("k", Json.num <| JsonNumber.fromNat config.rag?),
    --  ("imports", Json.arr <| List.toArray <| config.retrievalFilter.map (fun n => Json.str (n.toString)))
    ]

  let out ← IO.Process.output {
    cmd := "/home/riyaza/miniconda3/envs/env/bin/python3",
    args := #["ImProver/prompting/rag.py", data.compress]
  }

  let stdout := out.stdout.trim
  IO.println stdout
  IO.println "ERROR:"
  IO.println out.stderr
  let items := stdout.splitOn "<BREAK>"
  let items := if items.isEmpty then [] else items.take (items.length - 1)
  return items
