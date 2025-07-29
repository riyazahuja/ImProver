import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
open Lean Core Elab IO Meta Term Command Tactic System

-- may want to upgrade declarativity to use proof tree instead of just haves
def declarativity_score (cs : CompilationStep) : IO Float :=
  let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
  let haves := tac_stx.filter (fun stx =>
    match stx with
    | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
    | _ => false)
  return haves.length |>.toFloat
