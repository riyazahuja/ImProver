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

-- may want to upgrade declarativity to use proof tree instead of just haves
def declarativity_score (cs : CompilationStep) : IO Float :=
  let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
  let haves := tac_stx.filter (fun stx =>
    match stx with
    | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
    | _ => false)
  return haves.length |>.toFloat
