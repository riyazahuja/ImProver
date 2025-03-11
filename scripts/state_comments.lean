import TrainingData.Frontend
-- import TrainingData.InfoTree.ToJson
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

/- Helper function to return plaintext of a theorem annotated with goal states -/
def annotateTheorems (targetModule : Name) (decls : Option (List Name)) (proofAsSorry? : Bool) : IO (List String) := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean targetModule).toString

  /- Replace all tactics with "sorry" for faster execution -/
  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)

  /- Process the actual source code from our module -/
  let steps := Lean.Elab.IO.processInput' (← moduleSource targetModule) none (if proofAsSorry? then proofAsSorry else {}) fileName
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut annotatedTheorems := []

  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    if (decls.isSome && !(decls.get!.contains ci_name_stem)) then
      continue
    let state_comments ← insert_state_comments cmd
    if !annotatedTheorems.contains state_comments then
      annotatedTheorems := annotatedTheorems ++ [state_comments]
    -- IO.println s!"{state_comments}"

  return annotatedTheorems

def getModule (args : Cli.Parsed) :=
    args.positionalArg! "module" |>.as! ModuleName

def printAnnotations (args : Cli.Parsed) : IO UInt32 := do
    let module := getModule args
    let annotatedTheorems ← annotateTheorems module none false
    annotatedTheorems.forM IO.println
    return 0

-- #eval annotateTheorems `Mathlib.Tactic.LinearCombination.Lemmas none false
