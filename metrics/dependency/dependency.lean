import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules
import ImProver.online.utils
import ImProver.online.prompting.context
import ImProver.online.prompting.rag

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Mathlib.Tactic.Change
import Batteries.Lean.HashSet
import Batteries.Data.List.Basic
import Cli

open Lean Core Elab IO Meta Term Command Tactic Cli System


def dependency_score (cmd:CompilationStep) : IO Float := do
  let context ← get_context cmd
  let external_deps := context.filter (fun c =>
    match c.kind with
    | "theorem (internal)" => true
    | "theorem" => true
    | _ => false)
  return external_deps.length |>.toFloat
