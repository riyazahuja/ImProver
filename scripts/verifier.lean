import TrainingData.Frontend
import Cli
import scripts.state_comments

open Lean Core Elab IO Meta Term Tactic

set_option autoImplicit true

-- we begin by configuring the modules/theorems we want to improve and the improver settings
-- improver settings are given by annotation, rag, best of n, refinement, etc. and metric config
-- metric config is given as in evaluate/metrics
-- theorems are given by a list of names, and modules are given by a list of names


def insert_state_comments (step:CompilationStep) : IO String := do
  let mut trees := step.trees
  trees := trees.flatMap InfoTree.retainTacticInfo
  trees := trees.flatMap InfoTree.retainOriginal
  trees := trees.flatMap InfoTree.retainSubstantive

  let L₁ ← (trees.flatMap InfoTree.tactics).mapM TacticInvocation.rangeAndStates
  let L₂ := dropEnclosed L₁ |>.filter fun ⟨⟨⟨l₁, _⟩, ⟨l₂, _⟩⟩, _, _⟩  => l₁ = l₂
  let L₃ := (L₂.map fun ⟨r, sb, sa⟩ => (r, formatState sb, formatState sa))
  let mut src := ({str:=step.src.str, stopPos := step.src.stopPos, startPos := 0} : Substring).toString.splitOn "\n"
  let mut inserted : Std.HashSet Nat := Std.HashSet.ofList [10000000]
  for item in L₃.reverse do
    let ⟨⟨⟨l, c⟩, _⟩, sb, sa⟩ := item
    if sa.contains "🎉 no goals" then
      src := src.insertIdx l $ stateComment sa c
    if inserted.contains (l-1) then
      src := src.set (l-1) $ stateComment sb c
    else
      src := src.insertIdx (l-1) $ stateComment sb c
      inserted := inserted.insert (l-1)

  let out := ("\n".intercalate src)
  return out

/-- Our verifier needs two parts: First evaluate a whole module (for each module in config)
with the proof as sorry option on. this should return a bunch of compilationSteps.
Then on each decl in this module in the include list, we run the "improvement loop"

This "loop" for will take in the now will literally just print out the
theorem with the proof states interleaved.

Then we will elaborate this string on the environment before the command which created that declaration.
then we will print out the theorem with the proof states interleaved.
--/

def runAtDecls (mod : Name) (decls : Option (List Name) := none): IO Unit := do
  let proofAsSorry := ({} : KVMap).insert `debug.proofAsSorry (.ofBool true)
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none proofAsSorry (← findLean mod).toString

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  for (cmd, ci) in targets do
    if decls.isSome && !(decls.get!.contains ci.name) then
      continue

    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    unless cmd.msgs.isEmpty do
      throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    let contents := cmd.src.toString
    IO.println s!"COMPILATION STEP CONTENTS:\n {contents}"
    let prev_state := cmd.before
    --now, let's add the comment to the theorem, and run it after the prev env
    let elaborated_steps := Lean.Elab.IO.processInput' contents (some prev_state) {} (← findLean mod).toString

    let head? ← elaborated_steps.uncons
    match head? with
    | none =>
      IO.println s!"No elaborated steps"
    | some (head, _) =>
      -- Should probably check that ci is actually in the diff? But the below code is a bit finnicky with namespaces.
      -- works fine without it anyways

      -- if not (head.after.constants.map₂.contains ci.name) then
      --   IO.eprintln s!"Expected {ci.name} to be in the elaborated steps, but it was not:\n {(head.diff.map (fun info=>info.name))}"
      -- else
      IO.println s!"AFTER ELAB CONTENTS:\n {← insert_state_comments head}"


      let thm_str ← insert_state_comments head
      let context := ({str:=head.src.str,startPos := 0, stopPos := head.src.startPos} : Substring).toString
      let metric := s!"LENGTH"
      let llm_output_str ← IO.Process.output {
        cmd := ".venv/bin/python3",
        args := #["scripts/model.py", thm_str, context, metric]
      }
      let llm_output := llm_output_str.stdout
      let llm_err := llm_output_str.stderr

      IO.println s!"LLM OUTPUT:\n {llm_output}"
      IO.println s!"LLM ERR:\n {llm_err}"



#eval runAtDecls `Mathlib.Logic.Hydra (some [`Relation.cutExpand_le_invImage_lex])
