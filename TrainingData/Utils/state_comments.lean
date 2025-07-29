import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range

open Lean Elab IO Meta
open System

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

/- Helper function to recursively find and keep only the largest disjoint intervals -/
partial def dropEnclosed (L : List (Range × String × String)) : List (Range × String × String) :=
  let L' := L.filter fun ⟨r, _, _⟩ => ¬ L.any fun ⟨r', _, _⟩ => r < r'
  if L' = L then L' else dropEnclosed L'

/- Helper function to format goal state strings:  -/
/- REQUIRES: cutoff_length = None or cutoff_length = Some n for n >= 2 -/
def formatState (s : String) (cutoff_length : Option Nat) : List String :=
  if s = "" then ["🎉 no goals"] else
  let lines := (s.splitOn "\n").map fun line =>
    match cutoff_length with
    | none => line
    | some len =>
      if line.length > len then
        line.take (len-2) ++ " …"
      else
        line
  lines

def String.indent (s : String) (k : Nat) : String := ⟨List.replicate k ' '⟩ ++ s

def stateComment (state: List String) (column: Nat) :=
    ("/-".indent column)
    ++ "\n"
    ++ "\n".intercalate (state.map fun s => s.indent (column + 2))
    ++ "\n"
    ++ ("-/".indent column)


/- Adds comments about the goal state of the proof after each tactic -/
def insert_state_comments (step:CompilationStep) : IO String := do
  /- Get relevant tactic nodes -/
  let tactics := step.trees
    |>.flatMap InfoTree.retainTacticInfo
    |>.flatMap InfoTree.retainOriginal
    |>.flatMap InfoTree.retainSubstantive
    |>.flatMap InfoTree.tactics

  let tacticStates ← tactics.mapM TacticInvocation.rangeAndStates
  let separatedStates := dropEnclosed tacticStates |>.filter fun ⟨⟨⟨l₁, _⟩, ⟨l₂, _⟩⟩, _, _⟩  => l₁ = l₂
  let formattedStates := (separatedStates.map fun ⟨r, sb, sa⟩ => (r, formatState sb (some 80), formatState sa (some 80)))
  /- **TODO**: I changed the logic in runAtDecls below, so now `step.src` is a substring of a different string,
    maybe (all preceding contents ++ this theorem). So the below (might) have to be changed -/

  let mut src := ({str := step.src.str, startPos := 0, stopPos := step.src.stopPos} : Substring).toString.splitOn "\n"
  let mut inserted : Std.HashSet Nat := Std.HashSet.ofList [10000000]

  /- insert each of the goal states into the existing proof string -/
  for item in formattedStates.reverse do
    let ⟨⟨⟨l, c⟩, _⟩, sb, sa⟩ := item
    if sa.contains "🎉 no goals" then
      src := src.insertIdx l <| stateComment sa c
    if inserted.contains (l-1) then
      src := src.set (l-1) <| stateComment sb c
    else
      src := src.insertIdx (l-1) <| stateComment sb c
      inserted := inserted.insert (l-1)
  let out := ("\n".intercalate src)
  let trim_out := ({str := out, startPos := step.src.startPos, stopPos := out.endPos}:Substring).toString
  return trim_out
