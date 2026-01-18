import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System

-- Simple test: have statement that duplicates the root goal
def test_proof := "theorem simple_dup (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h : P ∧ Q → Q ∧ P := by
    intro ⟨hp, hq⟩
    exact ⟨hq, hp⟩
  exact h"

-- Expected: h should be caught as a duplicate of the root goal
-- Result: should score 0.0

set_option linter.unusedVariables false in
def getTestScore (mod : Name) (decl : Name) (new_proof : String) : IO Float := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none => return (-1.0)
  | some (target_cmd, _) => do
    let background_content := (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString
    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})
    let new_target ← elaborated_steps.head?

    match new_target with
    | none => return (-1.0)
    | some new_target => do
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      if !correct then
        return (-1.0)
      -- Import and use declarativity2_score from the other file
      return 0.0 -- placeholder

#eval IO.println "Test file compiled successfully"
