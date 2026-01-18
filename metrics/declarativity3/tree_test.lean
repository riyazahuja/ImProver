import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System


def getCompilationSteps (mod : Name) (decl : Name) (new_proof : String) : IO (Option CompilationStep × Option CompilationStep) := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none => return (none, none)
  | some (target_cmd, _) => do

    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})

    let new_target ← elaborated_steps.head?

    match new_target with
    | none =>
      return (target_cmd, none)
    | some new_target => do
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      return (some target_cmd, if correct then some new_target else none)



def extractProofTree (mod : Name) (decl : Name) (new_proof : String) : IO (Option ProofTree × Option ProofTree) := do
  let (old,new) ← getCompilationSteps mod decl new_proof
  let old_tree ← match old with
    | none => pure none
    | some old_cs => do
      let steps := (← old_cs.trees.filterMapM BetterParser).flatMap (fun x => x.steps)
      pure (getProofTree steps)
  let new_tree ← match new with
    | none => pure none
    | some new_cs => do
      let steps := (← new_cs.trees.filterMapM BetterParser).flatMap (fun x => x.steps)
      pure (getProofTree steps)
  return (old_tree, new_tree)



def new_proof := "lemma KD5_weakerThan_KD45 : (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
  -- Introduce a lemma to handle the subset relationship between the axioms of KD5 and KD45
  have h₁ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms → (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
    intro h
    -- Apply the lemma that establishes the weakening relation given a subset of axioms
    apply normal_weakerThan_of_subset
    -- Use the given subset relation to conclude the proof
    <;> assumption
  -- Prove the subset relation between the axioms of KD5 and KD45
  have h₂ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms := by
    -- Introduce the axioms of KD5 and verify they are included in the axioms of KD45
    intro φ hφ
    cases' hφ with hφ hφ
    <;> simp_all [LO.Modal.Hilbert.KD5, LO.Modal.Hilbert.KD45]
    <;> aesop
  -- Combine the results to conclude the weakening relation
  exact h₁ h₂"

#eval do extractProofTree `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof
