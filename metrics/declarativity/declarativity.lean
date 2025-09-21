import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System

-- may want to upgrade declarativity to use proof tree instead of just haves
-- def declarativity_score (cs : CompilationStep) : IO Float :=
--   let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
--   let haves := tac_stx.filter (fun stx =>
--     match stx with
--     | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
--     | _ => false)
--   return haves.length |>.toFloat

partial def getSpawnedGoalsCount (tree : ProofTree) (acc : Nat := 0) : Nat :=
  let counts := tree.children.map (fun child => getSpawnedGoalsCount child acc)
  let curr := tree.spawned_children.size
  counts.foldl (fun a b => a + b) curr

def declarativity_score (cs : CompilationStep) : IO Float := do
  let tree? := getProofTree <| (← (cs.trees.filterMapM (BetterParser)) ).flatMap (fun result => result.steps)

  match tree? with
  | none => return (0 : Float)
  | some tree =>
    let spawned_goals_count := getSpawnedGoalsCount tree |>.toFloat
    return spawned_goals_count

  -- let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
  -- let haves := tac_stx.filter (fun stx =>
  --   match stx with
  --   | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
  --   | _ => false)
  -- return haves.length |>.toFloat
