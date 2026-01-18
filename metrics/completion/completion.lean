import TrainingData.Frontend

open Lean Core Elab IO Meta Term Command Tactic System



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
