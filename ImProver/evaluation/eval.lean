-- import TrainingData.Frontend
import Cli
import ImProver.prompting.state_comments
import ImProver.prompting.context
import ImProver.prompting.prompts
import ImProver.inference.inference
import ImProver.utils
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules


import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true





/- An efficient way to verify each new proof candidate that the model outputs
    Requires the original proof's compilation steps, the module name, and a list of new proof candidates (as strings) to verify -/
def elaborateVariants (original : CompilationStep) (mod: Name) (variants : List String) : IO (List (Option (String × CompilationStep))) := do
  let options := ({} : KVMap)
      |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
      |>.insert `debug.byAsSorry (.ofBool false)
      |>.insert `linter.unusedVariables (.ofBool true)
      |>.insert `linter.unusedTactic (.ofBool true)
      |>.insert `linter.unreachableTactic (.ofBool true)

  let fileName := (← findLean mod).toString

  let contentsBefore : Substring := match original.src with
    | ⟨s, b, _⟩ => ⟨s, 0, b⟩

  /- Multithreading stuff to verify each new proof on separate threads -/
  let tasks := variants.map fun newCommand => IO.asTask (prio := Task.Priority.dedicated) do
    /- Parse and compile each proof... -/
    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (contentsBefore.toString ++ newCommand) fileName)
      original.parserStateBefore
      (original.commandStateBefore.withOptions options)
    /- ...and return the ones that work (otherwise none) -/
    let head? ← elaborated_steps.uncons
    return match head? with
      | none => none
      | some (head, _) => some (newCommand,head)

  let results ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get
  return results
