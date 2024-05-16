import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Mathlib.Tactic.Change
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

def stateComments (args : Cli.Parsed) : IO UInt32 := do
    searchPathRef.set compile_time_search_path%
    let module := args.positionalArg! "module" |>.as! ModuleName
    let mut trees ← moduleInfoTrees module
    trees := trees.bind InfoTree.retainTacticInfo
    trees := trees.bind InfoTree.retainOriginal
    trees := trees.bind InfoTree.retainSubstantive
    let L₁ ← (trees.bind InfoTree.tactics).mapM TacticInvocation.rangeAndStates
    let L₂ := dropEnclosed L₁ |>.filter fun ⟨⟨⟨l₁, _⟩, ⟨l₂, _⟩⟩, _, _⟩  => l₁ = l₂
    let L₃ := (L₂.map fun ⟨r, sb, sa⟩ => (r, formatState sb, formatState sa))
    let mut src := (← moduleSource module).splitOn "\n"

    let mut inserted : HashSet Nat := HashSet.ofList [10000000]
    for item in L₃.reverse do
      let ⟨⟨⟨l, c⟩, _⟩, sb, sa⟩ := item
      let c := if args.hasFlag "indent" then c else 0
      if sa.contains "🎉 no goals" then
        src := src.insertNth l $ stateComment sa c
      if inserted.contains (l-1) then
        src := src.set (l-1) $ stateComment sb c
      else
        src := src.insertNth (l-1) $ stateComment sb c
        inserted := inserted.insert (l-1)

    let out := ("\n".intercalate src)
    IO.println out
    return 0

/-- Setting up command line options and help text for `lake exe state_comments`. -/
def state_comments : Cmd := `[Cli|
  state_comments VIA stateComments; ["0.0.1"]
"Modify a Lean file by inserting comments after every tactic invocation showing the goal.
Prints the modified source code to stdout."

  FLAGS:
    "indent";  "Indent the state comments to the column of their corresponding tactic."

  ARGS:
    module : ModuleName; "Lean module to compile and annotate with state comments."
]

/-- `lake exe state_comments` -/
def main (args : List String) : IO UInt32 :=
  state_comments.validate args
