-- import TrainingData.Frontend
import Cli
import ImProver.prompting.state_comments
import ImProver.prompting.context
import ImProver.prompting.prompts
import ImProver.prompting.rag
import ImProver.evaluation.eval
import ImProver.utils
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules
import Lean.Util.Trace

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


/- Returns a dummy response (for debugging when model is offline) -/
def promptModel_debug (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  let prompt_name := config.prompt


  let prompt ← get_prompt prompt_name config cmd
  IO.println s!"Prompt:\n{prompt}"

  let srcCommand := cmd.src.toString
  let bon := config.best_of_n
  return List.range bon |>.map (fun i => s!"--DEBUG: {i}\n{srcCommand}")


def promptModel_server (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do

  let model := config.model
  let endpoint := config.endpoint
  let best_of_n := config.best_of_n
  let prompt_name := config.prompt
  -- let srcCommand ← if annotation? then (insert_state_comments cmd) else pure cmd.src.toString

  -- -- IO.println s!"srcCommand:\n{srcCommand.dropRightWhile (· == '\n')}"
  -- let annotation_prompt : String := s!" The goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. "
  -- let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.{if annotation? then annotation_prompt else " "}Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  let prompt ← get_prompt prompt_name config cmd
  IO.println s!"Prompt:\n{prompt}"
  let jsonPayload : Json := Json.mkObj [
      ("model", Json.str model),
      ("messages", Json.arr #[Json.mkObj [("role",Json.str "user"),("content", Json.str prompt)]]),
      ("max_tokens", Json.num <| JsonNumber.fromNat <| 256)
    ]
  let args := #[
    "-X", "POST",
    "-H", "Content-Type: application/json",
    "-d", s!"{jsonPayload.compress}",
    endpoint
  ]

  /- In parallel, send json via POST request using curl to the endpoint and await responses -/
  let tasks := List.range (best_of_n) |>.map fun _ => IO.asTask (prio := Task.Priority.dedicated) do

    let out_json ← IO.Process.output {
      cmd := "curl",
      args := args
    }
    return out_json

  let newCommandCandidates ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get

  /- Parse response as JSON -/
  let newCommandCandidates ← newCommandCandidates.mapM (fun out_json => do
    let out_json_parsed : Json := (match Json.parse out_json.stdout with
                          | Except.error _ => none
                          | Except.ok msg => some msg).get!

    /- Find correct field -/
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

    /- Cut out context/tags -/
    let tagOpen  := "<IMPROVED>"
    let tagClose := "</IMPROVED>"
    let trimmed_out := modelOutput.stripPrefix tagOpen |>.stripSuffix tagClose

    return trimmed_out)

  return newCommandCandidates




def promptModel_batched (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  let model := config.model
  let endpoint := config.endpoint
  let best_of_n := config.best_of_n
  let prompt_name := config.prompt
  -- let srcCommand ← if annotation? then (insert_state_comments cmd) else pure cmd.src.toString

  -- IO.println s!"srcCommand:\n{srcCommand.dropRightWhile (· == '\n')}"
  -- let annotation_prompt : String := s!" The goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. "
  -- let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.{if annotation? then annotation_prompt else " "}Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  let prompt ← get_prompt prompt_name config cmd
  IO.println s!"Prompt:\n{prompt}"
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
  IO.println responses
  -- Process each response to extract content between IMPROVED tags
  let newCommandCandidates := responses.map (fun response =>
    let tagOpen  := "<IMPROVED>"
    let tagClose := "</IMPROVED>"
    response.stripPrefix tagOpen |>.stripSuffix tagClose)

  return newCommandCandidates.toList



def promptModel_debug_raw (prompt : String) (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  IO.println "IN DEBUG"
  IO.println s!"Prompt:\n{prompt}"

  let srcCommand := cmd.src.toString
  let bon := config.best_of_n
  return List.range bon |>.map (fun i => s!"--DEBUG: {i}\n{srcCommand}")

def promptModel_batched_raw (prompt : String) (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  let model := config.model
  let endpoint := config.endpoint
  let best_of_n := config.best_of_n

  IO.println s!"Prompt:\n{prompt}"
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
  IO.println responses
  -- Process each response to extract content between IMPROVED tags
  let newCommandCandidates := responses.map (fun response =>
    let tagOpen  := "<IMPROVED>"
    let tagClose := "</IMPROVED>"
    response.stripPrefix tagOpen |>.stripSuffix tagClose)

  return newCommandCandidates.toList


def score (step : CompilationStep) (config : ImProverConfig): IO Float := do
  let metric_name := config.metric
  let msgs ← step.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))
  let correct := msgs.isEmpty

  let metric_score := if correct then get_metric metric_name step else -1

  return metric_score

def promptModel_refine (cmd : CompilationStep) (config : ImProverConfig) (num_steps : Nat) (keep_best? : Bool := True): IO (CompilationStep × List (List CompilationStep)) := do
  let targetModule := config.targetModule
  let model := config.model
  let endpoint := config.endpoint
  let best_of_n := config.best_of_n
  let prompt_name := config.prompt

  let build_payload := fun c => do
    let prompt ← get_prompt prompt_name config c
    return Json.mkObj [
      ("model", Json.str model),
      ("messages", Json.arr #[Json.mkObj [("role",Json.str "user"),("content", Json.str prompt)]]),
      ("max_tokens", Json.num <| JsonNumber.fromNat 256)
    ]

  let process_output (out : Process.Output) : IO (List String) := do
    let stdout := out.stdout.trim
    let contents := match stdout.splitOn "<RESPONSE>" |>.reverse with
    | []   => ""
    | last :: _ => last

    let responses := Json.parse (contents)
      |>.toOption.getD (Json.arr #[])
      |>.getArr?.toOption.getD (#[])
      |>.map (fun j => j.getStr?.toOption.getD "")
    IO.println responses
    -- Process each response to extract content between IMPROVED tags
    let newCommandCandidates := responses.map (fun response =>
      let tagOpen  := "<IMPROVED>"
      let tagClose := "</IMPROVED>"
      response.stripPrefix tagOpen |>.stripSuffix tagClose)
    return newCommandCandidates.toList

  let mut best : CompilationStep := cmd
  let mut traj : List (List CompilationStep) := []
  for _ in List.range num_steps do
    let jsonPayload ← build_payload best
    let out ← IO.Process.output {
      cmd := "/home/riyaza/miniconda3/envs/.venv10/bin/python3",
      args := #["ImProver/evaluation/send_batched.py", jsonPayload.compress, toString best_of_n, endpoint]
    }
    let variants ← process_output out
    let resultantSteps := (← elaborateVariants best targetModule variants) |>.filterMap (fun x => x) |>.map (fun x => x.2)
    traj := resultantSteps :: traj
    let cmp := fun curr_best new => do
      let old_score : Float ← score curr_best config
      let new_score : Float ← score new config
      let old_pos : Bool := old_score ≥ 0
      let new_pos : Bool := new_score ≥ 0
      let out : CompilationStep := match (old_pos, new_pos) with
      | (true, true) => if new_score < old_score then new else curr_best
      | (true, false) => curr_best
      | (false, true) => new
      | (false, false) => curr_best
      return out

    let best_result : CompilationStep ← resultantSteps.foldlM cmp best

    if keep_best? then
      best ← cmp best_result best
    else
      best := best_result
  return (best, traj.reverse)







/- Prompts the model running on an available web interface
  Takes a (compiled) theorem, a model name, an endpoint (URL to interface), and the number of separate attempts the model should make (best_of_n) -/
def promptModel (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  match config.model with
  | "DEBUG" => return ← promptModel_debug cmd config
  | _ => return ← promptModel_batched cmd config

def promptModel_raw (prompt : String) (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  match config.model with
  | "DEBUG" => return ← promptModel_debug_raw prompt cmd config
  | _ => return ← promptModel_batched_raw prompt cmd config
