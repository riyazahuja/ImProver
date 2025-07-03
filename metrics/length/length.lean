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



def length_score (cs : CompilationStep) : IO Float :=
    return InfoTree.tactics_new cs.trees |>.length |>.toFloat
