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



def completion_score (cs : CompilationStep) : IO Float := do
  let msgs : List String ← cs.msgs.filterMapM (fun msg : Message => do
    let m ← msg.data.toString
    let isSorry := msg.severity == .warning && m.trim == "declaration uses 'sorry'"
    if not (msg.severity == .error || isSorry) then
      return none
    else do
      return some (bombEmoji++m))
  return if cs.trees.length == 0
    then 0
    else msgs.length |>.toFloat
