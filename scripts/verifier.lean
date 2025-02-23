-- import TrainingData.Frontend
import Cli
import scripts.state_comments
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true

structure ImprovedTheoremInstance where
  originalTheorem : String
  modelOutput : String
  stateComments : String
  correct : Bool
  metricScore : Nat
  msgs : List String

structure ImProverConfig where
  mod : Name
  decls : Option (List Name) := none
  jsonPath : Option String := none

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
                  | none => step.src.str.splitOn "\n"
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


def promptModel (cmd : CompilationStep) (model : String := "nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B_full")
  (endpoint: String := "http://0.0.0.0:8000/v1/chat/completions") (best_of_n : Nat := 1) : IO (List String) := do
  let srcCommand := cmd.src.toString
  -- IO.println s!"srcCommand:\n{srcCommand.dropRightWhile (· == '\n')}"

  let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  let jsonPayload : Json := Json.mkObj [
      ("model", Json.str model),
      ("messages", Json.arr #[Json.mkObj [("role",Json.str "user"),("content", Json.str prompt)]]),
      ("max_tokens", Json.num <| JsonNumber.fromNat 4096)
    ]
  let args := #[
    "-X", "POST",
    "-H", "Content-Type: application/json",
    "-d", s!"{jsonPayload.compress}",
    endpoint
  ]

  let tasks := List.range (best_of_n) |>.map fun _ => IO.asTask (prio := Task.Priority.dedicated) do

    let out_json ← IO.Process.output {
      cmd := "curl",
      args := args
    }
    return out_json

  let newCommandCandidates ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get


  let newCommandCandidates ← newCommandCandidates.mapM (fun out_json => do
    let out_json_parsed : Json := (match Json.parse out_json.stdout with
                          | Except.error _ => none
                          | Except.ok msg => some msg).get!

    let out := match out_json_parsed with
              | Json.obj kvs => match kvs.find compare "choices" with
                | some (Json.arr choices) => match choices[0]? with
                  | some (Json.obj choice) => match choice.find compare "message" with
                    | some (Json.obj message) => match message.find compare "content" with
                      | some (Json.str s) => some s
                      | _ => none
                    | _ =>none
                  | _ => none
                | _ => none
              | _ => none

    let modelOutput := out.get!

    let tagOpen  := "<IMPROVED>"
    let tagClose := "</IMPROVED>"
    let trimmed_out := modelOutput.stripPrefix tagOpen |>.stripSuffix tagClose

    return trimmed_out)

  return newCommandCandidates


def promptModel_debug (cmd : CompilationStep) : IO (List String) := do
  let srcCommand := cmd.src.toString
  return ["--DEBUG\n"++srcCommand]


def elaborateVariants (original : CompilationStep) (mod: Name) (variants : List String) : IO (List (Option (String × CompilationStep))) := do
  let options := ({} : KVMap)
      |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
      |>.insert `debug.byAsSorry (.ofBool false) -- proofAsSorry not working???
      |>.insert `linter.unusedVariables (.ofBool true)
      |>.insert `linter.unusedTactic (.ofBool true)
      |>.insert `linter.unreachableTactic (.ofBool true)

  let fileName := (← findLean mod).toString

  let contentsBefore : Substring := match original.src with
    | ⟨s, b, _⟩ => ⟨s, 0, b⟩

  let tasks := variants.map fun newCommand => IO.asTask (prio := Task.Priority.dedicated) do
    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (contentsBefore.toString ++ newCommand) fileName)
      original.parserStateBefore
      (original.commandStateBefore.withOptions options)

    let head? ← elaborated_steps.uncons
    return match head? with
      | none => none
      | some (head, _) => some (newCommand,head)

  let results ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get
  return results


def ImProver (config : ImProverConfig): IO Unit := do
  let ⟨mod, decls, json_path⟩ := config
  let fileName := (← findLean mod).toString
  let mut trajectories_json := []
  /- TODO: I don't know if proofAsSorry is actually working -/
  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none proofAsSorry fileName
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)
  for (cmd, ci) in targets do
    if decls.isSome && !(decls.get!.contains ci.name) then
      continue
    IO.println s!"============================================="
    IO.println s!"Processing {ci.name} in {mod}"


    let tacs :=  InfoTree.tactics_new cmd.trees
    let tacs ← tacs.mapM (fun t => t.pp)
    IO.println s!"Tactics: {tacs.length}"
    IO.println s!"Tactics: {tacs}"
    IO.println s!"---------------------------------------------"

    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    -- unless cmd.msgs.isEmpty do
    --   throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    -- let newCommandCandidates ← promptModel cmd (best_of_n := 5)
    let newCommandCandidates ← promptModel_debug cmd
    let resultantSteps := (← elaborateVariants cmd mod newCommandCandidates).filterMap (fun x => x)

    let instances : List ImprovedTheoremInstance ← resultantSteps.mapM (fun (model_output,head) => do
      let correct := head.msgs.isEmpty
      let metric_score := InfoTree.tactics_new head.trees |>.length

      let state_comments ← insert_state_comments head

      let msgs ← head.msgs.mapM (fun msg => do
        let m ← msg.data.toString
        return bombEmoji++m)

      return ⟨cmd.src.toString, model_output, state_comments, correct, metric_score, msgs⟩
    )


    for i in instances do
      IO.println "-------------------------------------------------"
      let ⟨original, newCommand,elabed, correct, metric, msgs⟩ := i
      IO.println s!"Original:\n {original}"
      IO.println s!"Model Output:\n {newCommand}"
      IO.println s!"Annotated:\n {elabed}"
      IO.println s!"Correct: {correct}"
      IO.println s!"Metric: {metric}"
      for msg in msgs do
        IO.println msg
      IO.println "-------------------------------------------------"

    let trajectories_json_new := instances.map (fun i =>
      let ⟨original,newCmd,elabed, correct,metric,msgs⟩ := i
      Json.mkObj [
        ("module", Json.str mod.toString),
        ("original", original),
        ("new", Json.str newCmd),
        ("annotated",Json.str elabed),
        ("correct",Json.bool correct),
        ("metric",Json.num <| JsonNumber.fromNat metric),
        ("errors", Json.str ("\n\n".intercalate msgs))
        ])
    trajectories_json := trajectories_json ++ trajectories_json_new

  let trajectories := Json.arr (trajectories_json.toArray)
  match json_path with
  | some path =>
    IO.FS.writeFile path (trajectories.compress)
  | none => pure ()




#eval ImProver {mod:=`temp.temp, decls:=(some [`theorem1])}
