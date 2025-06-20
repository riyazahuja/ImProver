-- import TrainingData.Frontend
import Cli
import ImProver.online.prompting.state_comments
import ImProver.online.prompting.context
import ImProver.online.prompting.prompts
import ImProver.online.utils
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules
import Lean.Declaration

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true



-- SHOULD PROBABLY ADD STUFF TO
-- check if the original theorem `Name is the name of the CompilationStep
-- but for now its all ok.

/- An efficient way to verify each new proof candidate that the model outputs
    Requires the original proof's compilation steps, the module name, and a list of new proof candidates (as strings) to verify -/
def elaborateVariants (original : CompilationStep) (mod: Name) (variants : List String) : IO (List (String × (Option CompilationStep))) := do
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
      | none => (newCommand, none)
      | some (head, _) => (newCommand,some head)

  let results ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get
  return results


-- note that this calls get_prompt FOR EACH INSTANCE: BAD IF YOU'RE DOING RAG
def calculateInstances (ci : ConstantInfo) (cmd : CompilationStep)
(resultantSteps : List (String × CompilationStep)) (config : ImProverConfig)
: IO (List ImprovedTheoremInstance) := do
  let metric_name := config.metric
  let oldMsgs ← cmd.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))



  let old_correct := oldMsgs.isEmpty && cmd.trees.length > 0
  -- let old_score := if old_correct then some (tacs.length.toFloat) else none
  -- let old_score := if old_correct then some (get_metric metric_name cmd) else none
  let old_score ← if old_correct then do
    pure <| some (← get_metric metric_name cmd) else
    pure none

  let instances : List ImprovedTheoremInstance ← resultantSteps.mapM (fun (model_output,head) => do

    let state_comments ← insert_state_comments head

    let msgs ← head.msgs.filterMapM (fun msg => do
      if msg.severity != .error then
        return none
      let m ← msg.data.toString
      return some (bombEmoji++m))

    let correct := msgs.isEmpty && head.trees.length > 0
    -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
    let metric_score ← if correct then do
      pure <| some (← get_metric metric_name head) else
      pure none

    let delta := if correct && old_correct then (
        if old_score.get! == 0 then
          -- some (-1 : Float)
          none
        else
          some ((old_score.get! - metric_score.get!) / (old_score.get!))
        )
      else none

    let original_prompt ← get_prompt config.prompt config cmd
    let name : String := ci.name.toString

    return ImprovedTheoremInstance.mk name
      cmd.src.toString
      model_output
      state_comments
      old_correct
      correct
      old_score
      metric_score
      delta
      oldMsgs
      msgs
      original_prompt
      config
  )
  return instances



def calculateInstancesWithPrompt (ci : ConstantInfo) (cmd : CompilationStep)
(resultantSteps : List (String × Option CompilationStep)) (config : ImProverConfig) (prompt: String)
: IO (List ImprovedTheoremInstance) := do

  let metric_name := config.metric
  let oldMsgs ← cmd.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))



  let old_correct := oldMsgs.isEmpty && cmd.trees.length > 0 && cmd.src.toString.trim != ""
  -- let old_score := if old_correct then some (tacs.length.toFloat) else none
  let old_score ← if old_correct then do pure <| some (← get_metric metric_name cmd) else pure none


  let mut instances : List ImprovedTheoremInstance := []

  for (model_output,head?) in resultantSteps do
    match head? with
    | none =>
      let out := (ImprovedTheoremInstance.mk ci.name.toString
        cmd.src.toString
        model_output
        model_output
        old_correct
        false
        old_score
        none
        none
        oldMsgs
        [bombEmoji++"Unknown Error"]
        prompt
        config)

      instances := out :: instances
    | some head =>
      let state_comments ← insert_state_comments head

      let msgs ← head.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))

      let correct := msgs.isEmpty && head.trees.length > 0 && model_output.trim != ""
        && (head.diff.map (·.name) |>.contains ci.name)
      -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
      let metric_score ← if correct then do pure <| some (← get_metric metric_name head) else pure none

      let delta := if correct && old_correct then (
          if old_score.get! == 0 then
            -- some (-1 : Float)
            none
          else
            some ((old_score.get! - metric_score.get!) / (old_score.get!))
          )
        else none

      let name : String := ci.name.toString

      let out := ImprovedTheoremInstance.mk name
        cmd.src.toString
        model_output
        state_comments
        old_correct
        correct
        old_score
        metric_score
        delta
        oldMsgs
        msgs
        prompt
        config

      instances := out :: instances

  return instances















-- for llm metric
def calculateInstances_batched (ci : ConstantInfo) (cmd : CompilationStep)
(resultantSteps : List (String × CompilationStep)) (config : ImProverConfig)
: IO (List ImprovedTheoremInstance) := do

  let metric_name := config.metric
  let oldMsgs ← cmd.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))



  let old_correct := oldMsgs.isEmpty && cmd.trees.length > 0

  let old_score ← if old_correct then do pure <| some (← get_metric metric_name cmd) else pure none
  -- this one's scoring to the steps_with_scores thing...


  let steps_with_scores ← resultantSteps.mapM (fun (model_output, head) => do
    let model := config.model
    let endpoint := config.endpoint
    let best_of_n := config.best_of_n
    let prompt_name := config.prompt

    -- TODO ACTUALLY MAKE THE PROMPT FUNCTION FOR SCORING

    let prompt : String := s!"PROMPT FOR {model_output}"

    let jsonPayload : Json := Json.mkObj [
        ("model", Json.str model),
        ("messages", Json.arr #[Json.mkObj [("role",Json.str "user"),("content", Json.str prompt)]]),
        ("max_tokens", Json.num <| JsonNumber.fromNat 1024)
      ]
    -- Call Python script with JSON payload
    let out ← IO.Process.output {
      cmd := "/home/riyaza/miniconda3/envs/env/bin/python3",
      args := #["ImProver/inference/send_batched.py", jsonPayload.compress, toString best_of_n, endpoint]
    }

    let stdout := out.stdout.trim
    let contents := match stdout.splitOn "<RESPONSE>" |>.reverse with
    | []   => ""
    | last :: _ => last

    let responses := Json.parse (contents)
      |>.toOption.getD (Json.arr #[])
      |>.getArr?.toOption.getD (#[])
      |>.map (fun j => j.getStr?.toOption.getD "")

    -- TODO FIX THIS FOR MODEL OUTPUT !!!!!!!
    let newCommandCandidates := responses.map (fun response =>
      let tagOpen  := "<IMPROVED>"
      let tagClose := "</IMPROVED>"
      response.stripPrefix tagOpen |>.stripSuffix tagClose
      )

    return (model_output,head,0.0)

  )


  let instances : List ImprovedTheoremInstance ← steps_with_scores.mapM (fun (model_output,head,score) => do

    let state_comments ← insert_state_comments head

    let msgs ← head.msgs.filterMapM (fun msg => do
      if msg.severity != .error then
        return none
      let m ← msg.data.toString
      return some (bombEmoji++m))

    let correct := msgs.isEmpty && head.trees.length > 0
    -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
    let metric_score := if correct then some score else none

    let delta := if correct && old_correct then (
        if old_score.get! == 0 then
          some (-1 : Float)
        else
          some ((old_score.get! - metric_score.get!) / (old_score.get!))
        )
      else none

    let original_prompt ← get_prompt config.prompt config cmd
    let name : String := ci.name.toString

    return ImprovedTheoremInstance.mk name
      cmd.src.toString
      model_output
      state_comments
      old_correct
      correct
      old_score
      metric_score
      delta
      oldMsgs
      msgs
      original_prompt
      config
  )
  return instances
