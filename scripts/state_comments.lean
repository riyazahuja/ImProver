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

open Lean Elab IO Meta
open Cli System

namespace Lean.Elab.TacticInvocation

def rangesAndGoals (i : TacticInvocation) : IO (Range × String) := do
  return ⟨i.range, (Format.joinSep (← i.goalStateAfter) "\n").pretty 1000000⟩

def rangeAndStates (i : TacticInvocation) : IO (Range × String × String) := do
  return ⟨
    i.range,
    ((← i.mainGoalStateBefore)).pretty 1000000,
    ((← i.mainGoalStateAfter)).pretty 1000000
  ⟩

end Lean.Elab.TacticInvocation

partial def dropEnclosed (L : List (Range × String × String)) : List (Range × String × String) :=
  let L' := L.filter fun ⟨r, _, _⟩ => ¬ L.any fun ⟨r', _, _⟩ => r < r'
  if L' = L then L' else dropEnclosed L'

def formatState (s : String) : List String :=
  if s = "" then ["🎉 no goals"] else
  let lines := (s.splitOn "\n").map fun l =>
    if l.length > 80 then
      l.take 78 ++ " …"
    else
      l
  lines

def String.indent (s : String) (k : Nat) : String := ⟨List.replicate k ' '⟩ ++ s

def stateComment (state: List String) (column: Nat) :=
    ("/-".indent column)
    ++ "\n"
    ++ "\n".intercalate (state.map fun s => s.indent (column + 2))
    ++ "\n"
    ++ ("-/".indent column)
