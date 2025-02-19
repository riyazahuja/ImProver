-- import TrainingData.Frontend
import Cli
import scripts.state_comments

open Lean Core Elab IO Meta Term Command Tactic

set_option autoImplicit true

-- we begin by configuring the modules/theorems we want to improve and the improver settings
-- improver settings are given by annotation, rag, best of n, refinement, etc. and metric config
-- metric config is given as in evaluate/metrics
-- theorems are given by a list of names, and modules are given by a list of names


def insert_state_comments (step:CompilationStep) (pre_elab_str: Option String := none) : IO String := do
  let mut trees := step.trees
  trees := trees.flatMap InfoTree.retainTacticInfo
  trees := trees.flatMap InfoTree.retainOriginal
  trees := trees.flatMap InfoTree.retainSubstantive

  let L₁ ← (trees.flatMap InfoTree.tactics).mapM TacticInvocation.rangeAndStates
  let L₂ := dropEnclosed L₁ |>.filter fun ⟨⟨⟨l₁, _⟩, ⟨l₂, _⟩⟩, _, _⟩  => l₁ = l₂
  let L₃ := (L₂.map fun ⟨r, sb, sa⟩ => (r, formatState sb, formatState sa))
  /- **TODO**: I changed the logic in runAtDecls below, so now `step.src` is a substring of a different string,
    maybe (all preceding contents ++ this theorem). So the below (might) have to be changed -/
  let mut src := match pre_elab_str with
                  | none => ({str:=step.src.str, stopPos := step.src.stopPos, startPos := 0} : Substring).toString.splitOn "\n"
                  | some str => (({str:=step.src.str, stopPos := step.src.startPos, startPos := 0} : Substring).toString ++ str).splitOn "\n"
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

def _root_.Lean.Elab.Command.State.withOptions (state : Command.State) (options : Options) :=
  { state with
    scopes := state.scopes.map fun s : Scope =>
      { s with opts := Id.run do
          let mut opts := s.opts
          for (k, v) in options do
            opts := opts.insert k v
          opts } }

/-- Our verifier needs two parts: First evaluate a whole module (for each module in config)
with the proof as sorry option on. this should return a bunch of compilationSteps.
Then on each decl in this module in the include list, we run the "improvement loop"

This "loop" for will take in the now will literally just print out the
theorem with the proof states interleaved.

Then we will elaborate this string on the environment before the command which created that declaration.
then we will print out the theorem with the proof states interleaved.
--/

def runAtDecls (mod : Name) (decls : Option (List Name) := none): IO Unit := do
  let fileName := (← findLean mod).toString

  /- TODO: I don't know if proofAsSorry is actually working -/
  let proofAsSorry := ({} : KVMap).insert `debug.proofAsSorry (.ofBool true)
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none proofAsSorry fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  for (cmd, ci) in targets do
    if decls.isSome && !(decls.get!.contains ci.name) then
      continue

    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    unless cmd.msgs.isEmpty do
      throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    let contentsBefore : Substring := match cmd.src with
      | ⟨s, b, _⟩ => ⟨s, 0, b⟩
    let srcCommand := cmd.src.toString
    IO.println s!"COMPILATION STEP CONTENTS:\n{srcCommand.dropRightWhile (· == '\n')}"

    /- Presumably, interaction with the LLM improver agent happens here.
      Given e.g. the srcCommand (source theorem before improvement),
      or e.g. ← insert_state_comments cmd (source theorem before improvement + state comments),
      the LLM outputs its proof improvement candidates to newCommandCandidates.
      Here we (1) don't do any changes and (2) replace rfl by sorry as a toy example. -/
    let newCommandCandidates := [
      srcCommand,
      srcCommand.replace "rfl" "sorry"
    ]

    let options := ({} : KVMap)
      |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
      |>.insert `debug.proofAsSorry (.ofBool false) -- turn proof checking back on

    /- Run proof candidates in parallel -/
    let tasks := newCommandCandidates.map fun newCommand => IO.asTask do
      let mut msgs := #[]
      let mut correct := false
      let elaborated_steps := Lean.Elab.IO.compilationSteps
        (Parser.mkInputContext (contentsBefore.toString ++ newCommand) fileName)
        cmd.parserStateBefore
        (cmd.commandStateBefore.withOptions options)

      let head? ← elaborated_steps.uncons
      match head? with
      | none =>
        msgs := msgs.push s!"No elaborated steps"
      | some (head, _) =>
        -- Should probably check that ci is actually in the diff? But the below code is a bit finnicky with namespaces.
        -- works fine without it anyways

        -- if not (head.after.constants.map₂.contains ci.name) then
        --   IO.eprintln s!"Expected {ci.name} to be in the elaborated steps, but it was not:\n {(head.diff.map (fun info=>info.name))}"
        -- else
        msgs := msgs.push s!"NEW COMMAND:\n{newCommand}"
        msgs := msgs.push s!"CONSTANTS:\n{head.before.constants.map₂.toList.map (fun (x,v)=>x)}"
        msgs := msgs.push s!"CONSTANTS:\n{head.after.constants.map₂.toList.map (fun (x,v)=>x)}"
        msgs := msgs.push s!"AFTER ELAB CONTENTS:\n {← insert_state_comments head}"

        /- Any errors in the improved proof will be caught here -/
        for m in head.msgs do
          msgs := msgs.push (bombEmoji ++ (← m.data.toString))
        correct := head.msgs.isEmpty

      return (correct, msgs)

    let results ← tasks.mapM fun (t : BaseIO _) => do
      IO.ofExcept <| (← t).get
    for result in results do
      IO.println "============================================="
      let (correct, msgs) := result
      IO.println s!"Correct: {correct}"
      for msg in msgs do
        IO.println msg


#eval runAtDecls `temp.temp

#eval runAtDecls `Mathlib.Logic.Hydra -- (some [`Relation.cutExpand_le_invImage_lex])
