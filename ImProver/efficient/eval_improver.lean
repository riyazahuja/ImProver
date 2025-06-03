import ImProver.prompting.state_comments
import ImProver.prompting.context
import Cli
import ImProver.prompting.prompts
import ImProver.inference.inference
import ImProver.evaluation.eval
import ImProver.utils
import ImProver.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports



import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
-- import Compfiles

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true



structure Instance where
  module : String
  decl : String
  og_correct : Bool
  og_errors : List String
  og_score : Option Float
  new_correct : Bool
  new_errors : List String
  new_score : Option Float
  delta : Option Float
  og_raw : String
  new_raw : String
  new_trimmed : String
  original_prompt : String
deriving Inhabited, ToJson


def String.getTagged (s: String) (tag : String) : Option String :=
  s.splitAtString s!"<{tag}>" |>.getD (("", "")) |>.2 |>.splitAtString (s!"</{tag}>") |>.getD (("", "")) |>.1


def getInstances (preinstances : Array (CompilationStep × ConstantInfo × String × Option CompilationStep × String))
(metric : String) (mod : String)
: IO (List Instance) := do

  let mut instances : List Instance := []

  for (original, ci, model_output,new?, prompt) in preinstances do

    let oldMsgs ← original.msgs.filterMapM (fun msg => do
          if msg.severity != .error then
            return none
          let m ← msg.data.toString
          return some (bombEmoji++m))



    let old_correct := oldMsgs.isEmpty && original.trees.length > 0 && original.src.toString.trim != ""
    -- let old_score := if old_correct then some (tacs.length.toFloat) else none
    let old_score ← if old_correct then do pure <| some (← get_metric metric original) else pure none


    let contentsBefore : Substring := match original.src with
      | ⟨s, b, _⟩ => ⟨s, 0, b⟩
    let trimmed_output := model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim
    -- remove everything before the first <IMPROVED> tag and after the first </IMPROVED> tag
    let trimmed_output := match model_output.trim.getTagged "IMPROVED" with
      | some x => x
      | none => trimmed_output

    match new? with
    | none =>
      let out := (Instance.mk mod
        ci.name.toString
        old_correct
        oldMsgs
        old_score
        false
        [bombEmoji++"Unknown Error, CompilationStep not found"]
        none
        none
        original.src.toString
        model_output
        trimmed_output
        prompt
        )

      instances := out :: instances
    | some head =>

      let msgs ← head.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))

      let correct := msgs.isEmpty && head.trees.length > 0 && model_output.trim != ""
        && (head.diff.map (·.name) |>.contains ci.name)
      -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
      let metric_score ← if correct then do pure <| some (← get_metric metric head) else pure none

      let delta := if correct && old_correct then (
          if old_score.get! == 0 then
            none
          else
            some ((old_score.get! - metric_score.get!) / (old_score.get!))
          )
        else none

      let out := Instance.mk mod
        ci.name.toString
        old_correct
        oldMsgs
        old_score
        correct
        msgs
        metric_score
        delta
        original.src.toString
        model_output
        trimmed_output
        prompt

      instances := out :: instances

  return instances


-- def String.splitAtString (s : String) (pattern : String): Option (String × String) :=
--   if h : pattern.endPos.1 = 0 then none
--   else
--     have hPatt := Nat.zero_lt_of_ne_zero h
--     let rec loop (pos : String.Pos) :=
--       if h : pos.byteIdx + pattern.endPos.byteIdx > s.endPos.byteIdx then
--         none
--       else
--         have := Nat.lt_of_lt_of_le (Nat.add_lt_add_left hPatt _) (Nat.ge_of_not_lt h)
--         if s.substrEq pos pattern 0 pattern.endPos.byteIdx then
--           -- Found a match, return split strings
--           let before := s.extract 0 pos
--           let after := s.extract (pos + pattern) s.endPos
--           some (before, after)
--         else
--           have := Nat.sub_lt_sub_left this (lt_next s pos)
--           loop (s.next pos)
--       termination_by s.endPos.1 - pos.1
--     loop 0


-- #eval "hello <IMPROVED> world </IMPROVED>" |>.getTagged "IMPROVED"
/--
Return type used internally by `withTimeout`.
-/
inductive TimeoutResult (α : Type) where
  | success (val : α)
  | timeout

/--
Run a computation with a timeout.
-/
def withTimeout (timeout : UInt32) (x : IO α) : IO α := do
  let timeoutTask ← IO.asTask <| IO.sleep timeout >>= fun _ => return TimeoutResult.timeout
  let mainTask ← IO.asTask (prio := .dedicated) <| TimeoutResult.success <$> x
  match ← IO.waitAny [mainTask, timeoutTask] with
  | .ok <| .success a =>
    IO.cancel timeoutTask
    return a
  | .ok <| .timeout =>
    IO.cancel mainTask
    throw <| .userError s!"Operation timed out after {timeout}ms"
  | .error e =>
    IO.cancel mainTask
    IO.cancel timeoutTask
    throw e

def evalImprover (mod : Name) (promptFile : String) (metric : String) (runPath : String) (outputPath : String) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  -- let mut trajectories_json := []
  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)

  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none proofAsSorry fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets do
    let isThm? := match ci with
      | .thmInfo _ => true
      | _ => false

    let pf_env := cmd.after
    let ctx : Core.Context := {fileName := "", fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1
      | none => false



    if not isThm? || not isHuman then
      continue

    targets_new := targets_new.push (cmd, ci)



  -- for each target (cmd, ci) in targets_new, we want to
  -- get all the model outputs for the target and evaluate all of
  -- them in parallel into CompilationStep's and ConstantInfo's
  -- then we want to evaluate metrics and output to json.
  -- in python, analyze these jsons as a big csv

  IO.println s!"Found {targets_new.size} targets"
  -- convert to tasks!!!
  -- let mut preinstances := []
  -- for (cmd, ci) in targets_new do
  --type: : Array (BaseIO (Task (Except Error (Array (CompilationStep × ConstantInfo × String × String)))))
  let preinstances_runner  := (targets_new.map fun (cmd,ci) => (do--IO.asTask do
    let SQL_escaped_name := ci.name.toString.replace "'" "''"
    let SQL_escaped_file := promptFile.replace "'" "''"
    let SQL_cmd : String := s!"SELECT * FROM run_data WHERE decl = '{SQL_escaped_name}' AND file_path = '{SQL_escaped_file}';"
    IO.println s!"== [[{ci.name}]] =="
    let output ← IO.Process.output {
      cmd := "duckdb",--/home/riyaza/.local/bin/duckdb",
      args := #[s!"{runPath}/data.duckdb", "--readonly", "--json", "-c", SQL_cmd]}

    if output.exitCode != 0 then
      IO.println s!"Error running duckdb: {output.stderr}"
      -- break
      return #[]

    let json? := output.stdout
    -- IO.println s!"DuckDB output: {json?}"
    -- IO.println s!"DuckDB err: {output.stderr}"
    -- IO.println s!"DuckDB exit code: {output.exitCode}"
    -- IO.println s!"[==> variant_tuples?"--\n===={json?}\n===="
    -- IO.println "\n\n"
    -- IO.println s!"DuckDB output: {json?}"
    -- IO.println "\n\n"
    let variant_tuples? :=
      let json := Json.parse json? |>.toOption.get!
      match json with
      | .arr variants =>
        some (variants.filterMap (fun v =>
          let model_answer := (v : Json).getObjVal? "answer"
          let prompt := (v : Json).getObjVal? "prompt"
          match (model_answer.toOption, prompt.toOption) with
          | (some (Json.str answer), some (Json.str prompt)) => some (cmd, ci, answer, prompt)
          | _ => none
        ))
      | _ =>
        none
    -- IO.println s!"<== variant_tuples? completed]"

    return (variant_tuples?.getD #[])
    -- IO.sleep 1000
  ))

  let preinstances := (← preinstances_runner.mapM id--(fun (t : BaseIO _) => do
   -- IO.ofExcept (← t).get
  ) |>.flatten

  IO.println s!"Found {preinstances.size} variants"

  let options := ({} : KVMap)
      |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
      |>.insert `debug.byAsSorry (.ofBool false)
      |>.insert `linter.unusedVariables (.ofBool true)
      |>.insert `linter.unusedTactic (.ofBool true)
      |>.insert `linter.unreachableTactic (.ofBool true)



  /- Multithreading stuff to verify each new proof on separate threads -/
  let tasks := preinstances.map fun (original, ci, model_output, prompt) => do --IO.asTask do
    IO.println s!"Evaluating {ci.name}"
    let contentsBefore : Substring := match original.src with
      | ⟨s, b, _⟩ => ⟨s, 0, b⟩
    let trimmed_output := model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim
    -- remove everything before the first <IMPROVED> tag and after the first </IMPROVED> tag
    let trimmed_output := match model_output.trim.getTagged "IMPROVED" with
      | some x => x
      | none => trimmed_output
    IO.println s!"trimmed output (length: {trimmed_output.length})"



    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (contentsBefore.toString ++ trimmed_output) fileName)
      original.parserStateBefore
      (original.commandStateBefore.withOptions options)

    IO.println s!"Made MLList"

    /- ...and return the ones that work (otherwise none) -/
    -- if trimmed_output.length == 320 then
    --   IO.println s!"Skipping {ci.name} because trimmed output is too short\n\n{trimmed_output}"
    --   return (original, ci, model_output, none, prompt)


    try
      let head? ← withTimeout 10000 elaborated_steps.uncons
      IO.println "DONE"
      IO.println "============="
      return match head? with
        | none => (original, ci, model_output, none, prompt)
        | some (head, _) => (original, ci, model_output,some head, prompt)

    catch e =>
      IO.println s!"Error elaborating {ci.name}: {e}"
      IO.println "============="
      return (original, ci, model_output, none, prompt)

    -- let head? ← elaborated_steps.uncons

  let results ← tasks.mapM id --fun (t : BaseIO _) => do
    --IO.ofExcept <| (← t).get

  IO.println " ==== GETTING INSTANCES ==== "
  let instances ← getInstances results metric mod.toString
  IO.println " ==== DONE GETTING INSTANCES ==== "

  let outputJson := Json.arr <| instances.map (fun i => ToJson.toJson i) |>.toArray

  IO.println s!"Writing to {outputPath}"
  -- let trajectories := Json.arr (trajectories_json.toArray)
  -- match json_path with
  -- | some path =>
  if not (← System.FilePath.pathExists outputPath) then
    let parent := System.FilePath.parent outputPath
    match parent with
    | some path =>
      IO.println path
      IO.FS.createDirAll path
    | none => pure ()


  IO.FS.writeFile outputPath (outputJson.compress)

  -- let valid := if (preinstances.size == targets_new.size) && (targets_new.size == instances.length) then
  --   0
  -- else
  --   1

  return 0

  -- return valid
  -- | none => pure ()



def evalImproverCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let promptFile := args.positionalArg! "promptFile" |>.as! String
  let metric := args.positionalArg! "metric" |>.as! String
  let runPath := args.positionalArg! "runPath" |>.as! String
  let outputPath := args.positionalArg! "outputPath" |>.as! String
  let mod :Name := module


  evalImprover mod promptFile metric runPath outputPath


def eval_improver : Cmd := `[Cli|
  eval_improver VIA evalImproverCLI; ["0.0.1"]
"Evaluate ImProver."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    promptFile : String; "File path to prompt data."
    metric : String; "Metric to use for evaluation."
    runPath : String; "Path to the run DB."
    outputPath : String; "Where to save the Json output."
]


def main (args : List String) : IO UInt32 :=
  eval_improver.validate args


-- #eval evalImprover `Mathlib.Logic.Hydra "length" "runs/RUN_20250515_031905" "temp.json"
