import TrainingData.Utils.context

open Lean Core Elab IO Meta Term Command Tactic System


def dependency_score (cmd:CompilationStep) : IO Float := do
  let context ← get_context cmd
  let external_deps := context.filter (fun c =>
    match c.kind with
    -- | "theorem (internal)" => true
    | "theorem" => true
    | _ => false)
  return external_deps.length |>.toFloat
