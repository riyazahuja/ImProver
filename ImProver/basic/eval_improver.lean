import Cli
import TrainingData.Utils.c2
import TrainingData.Utils.HumanTheorem
import metrics.router

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
  decl_idx : String
  recgen : Option (List TheoremData) := none
deriving Inhabited, ToJson


def String.getTagged (s: String) (tag : String) : Option String :=
  s.splitAtString s!"<{tag}>" |>.getD (("", "")) |>.2 |>.splitAtString (s!"</{tag}>") |>.getD (("", "")) |>.1

def String.getBetween (s: String) (left : String) (right : String) : Option String :=
  s.splitAtString left |>.getD (("", "")) |>.2 |>.splitAtString (right) |>.getD (("", "")) |>.1


def getInitialProofState3 (cmd : CompilationStep) : IO String := do
  let env := cmd.after
  let ci? := cmd.diff.get? 0

  if ci?.isSome then
    try
      let ci := ci?.get!
      let (state, _, _) ← MetaM.toIO (ctxCore := { fileName := "", fileMap := default }) (sCore := { env }) do
        -- forallTelescope transforms ∀ n : Nat, 0 + n = n to _args = #[n : Nat] and typ = 0 + n = n
        forallTelescope ci.type fun _args typ => do
          let g ← mkFreshExprMVar typ
          g.mvarId!.withContext do
            -- We disable some pretty-printing options,
            -- e.g. Nat is not pretty-printed as ℕ
            -- HAdd.hAdd is not pretty-printed as +
            let state ← withOptions (fun o => o.set `pp.notation false |>.set `pp.fullNames true) <| Meta.ppGoal g.mvarId!
            return state.pretty (width := 100000000)
      return state
    catch _ =>
      let backup:=  match InfoTree.tactics_new cmd.trees |>.get? 0 with
      | some t => t.mainGoalStateBefore
      | _ => pure default
      return (← backup).pretty (width := 100000000)

  else
    let backup:=  match InfoTree.tactics_new cmd.trees |>.get? 0 with
    | some t => t.mainGoalStateBefore
    | _ => pure default
    return (← backup).pretty (width := 100000000)

def isDefEq (a : CompilationStep) (b : CompilationStep) : IO Bool := -- eventually upgrade to check all items in diff
  let env := a.after
  -- if a.diff.isEmpty || b.diff.isEmpty then
    -- return false
  let a_ci? := a.diff.get? 0
  let b_ci? := b.diff.get? 0

  if a_ci?.isNone || b_ci?.isNone then
    return false
  else do
    let a_ci := a_ci?.get!
    let b_ci := b_ci?.get!

    let (eq, _,_) ← MetaM.toIO (ctxCore := { fileName := "", fileMap := default }) (sCore := { env }) do
      Meta.isDefEq a_ci.type b_ci.type
    return eq




def getInstances (preinstances : Array (CompilationStep × ConstantInfo × String × String × Option CompilationStep × String × String))
(metric : String) (mod : String) (sorryOk : Bool) (correctnessCondition : String)
: IO (List Instance) := do
  -- correctness condition is either
  -- "none" => no type mandates
  -- "eq" => name + types must be equal
  -- "neq" => name + types must not be equal

  let mut instances : List Instance := []

  for (original, ci, model_output,trimmed_output, new?, prompt, decl_idx) in preinstances do

    let oldMsgs ← original.msgs.filterMapM (fun msg => do
        let m ← msg.data.toString
        let isSorry := not sorryOk && msg.severity == .warning && m.trim == "declaration uses 'sorry'"
        if msg.severity != .error && not isSorry then
          return none
        return some (bombEmoji++m))



    let old_correct := oldMsgs.isEmpty && original.trees.length > 0 && original.src.toString.trim != ""
    -- let old_score := if old_correct then some (tacs.length.toFloat) else none
    let old_score ← if old_correct then do pure <| some (← route_metric metric original) else pure none


    -- let contentsBefore : Substring := match original.src with
    --   | ⟨s, b, _⟩ => ⟨s, 0, b⟩
    -- let trimmed_output := model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim
    -- -- remove everything before the first <IMPROVED> tag and after the first </IMPROVED> tag
    -- let trimmed_output := match model_output.trim.getTagged "IMPROVED" with
    --   | some x => x
    --   | none => trimmed_output

    match new? with
    | none =>
      let out := {
        module := mod,
        decl := ci.name.toString,
        og_correct := old_correct,
        og_errors := oldMsgs,
        og_score := old_score,
        new_correct := false,
        new_errors := [bombEmoji++"Unknown Error, CompilationStep not found"],
        new_score := none,
        delta := none,
        og_raw := original.src.toString,
        new_raw := model_output,
        new_trimmed := trimmed_output,
        original_prompt := prompt,
        decl_idx := decl_idx
      }

      instances := out :: instances
    | some head =>

      let msgs ← head.msgs.filterMapM (fun msg => do
        let m ← msg.data.toString
        let isSorry := not sorryOk && msg.severity == .warning && m.trim == "declaration uses 'sorry'"
        if not (msg.severity == .error || isSorry) then
          return none
        return some (bombEmoji++m))
      -- IO.println "Checking correctness..."


      -- let io_correct : IO Bool := match metric with
      -- | "conjecturer" => do
      --   IO.println "hello!"
      --   let defn_eq? ←  isDefEq head original
      --   -- let new_goal ← getInitialProofState2 head
      --   -- let old_goal ←  getInitialProofState2 original
      --   -- let eq? := new_goal == old_goal
      --   let output := msgs.isEmpty && head.trees.length > 0 && trimmed_output.trim != "" && not defn_eq?--new_goal != old_goal
      --   -- if output then
      --     -- IO.println s!">>> New goal: {new_goal}"
      --     -- IO.println s!">>> Old goal: {old_goal}"
      --   pure output
      -- | _ => do
      --   IO.println "world!"
      --   let contains? := (head.diff.map (·.name) |>.contains ci.name)
      --   let equal_types? := if contains? then
      --     let eqs := head.diff.filter (fun c => c.name == ci.name)
      --     eqs.map (fun c => c.type == ci.type) |>.any id
      --   else
      --     false
      --   pure <| msgs.isEmpty && head.trees.length > 0 && trimmed_output.trim != ""
      --   -- && equal_types? && contains?
      --   -- MUST BE A BETTER WAY TO DO THIS!!! ^^^^

      let io_correct : IO Bool := match correctnessCondition with
      | "none" => pure <| msgs.isEmpty && head.trees.length > 0 && trimmed_output.trim != ""
      | "eq" => do
        let contains? := (head.diff.map (·.name) |>.contains ci.name)
        let equal_types? := if contains? then
          let eqs := head.diff.filter (fun c => c.name == ci.name)
          eqs.map (fun c => c.type == ci.type) |>.any id
        else
          false
        pure <| msgs.isEmpty && head.trees.length > 0 && trimmed_output.trim != ""
        && equal_types? && contains?
      | "neq" => do
        let defn_eq? ←  isDefEq head original

        let output := msgs.isEmpty && head.trees.length > 0 && trimmed_output.trim != ""
          && not defn_eq?
        pure output
      | _ => pure false


      let correct ← io_correct

      if correct then
        IO.println s!">>>> Correct: {correct}"
        let msg_raw ← head.msgs.mapM (fun msg => msg.data.toString)
        -- let msg_raw := msg_raw.map (fun s=> s.trim == "declaration uses 'sorry'")
        IO.println s!"Messages: {"||".intercalate msg_raw}"
        IO.println head.src.toString


      -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
      let metric_score ← if correct then do pure <| some (← route_metric metric head) else pure none

      let delta := if correct && old_correct then (
          if old_score.get! == 0 then
            none
          else
            some ((metric_score.get!- old_score.get!) / (old_score.get!))
          )

        else none

      let final_trimmed := if correct then
        head.src.toString
      else
        trimmed_output

      let out := {
        module := mod,
        decl := ci.name.toString,
        og_correct := old_correct,
        og_errors := oldMsgs,
        og_score := old_score,
        new_correct := correct,
        new_errors := msgs,
        new_score := metric_score,
        delta := delta,
        og_raw := original.src.toString,
        new_raw := model_output,
        -- new_trimmed := trimmed_output,
        new_trimmed := final_trimmed,
        original_prompt := prompt,
        decl_idx := decl_idx
      }

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



def evalImprover (mod : Name) (metric : String) (runPath : String) (outputPath : String) (sorryOk : Bool) (correctnessCondition : String) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%
  IO.println s!"Running eval_improver for {mod} with metric {metric}"

  let fileName := (← findLean mod).toString
  let options := ({} : KVMap)
      |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
      |>.insert `debug.byAsSorry (.ofBool false)
      |>.insert `linter.unusedVariables (.ofBool true)
      |>.insert `linter.unusedTactic (.ofBool true)
      |>.insert `linter.unreachableTactic (.ofBool true)

  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none options fileName

  -- IO.println s!"Processed compilation steps for {mod}"

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  for (cmd, ci) in targets do
    IO.println s!"Processing target: {ci.name} ({cmd.src.toString})"
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
  let preinstances_runner := (targets_new.map fun (cmd,ci) => (do--IO.asTask do
    let SQL_escaped_name := ci.name.toString.replace "'" "''"
    -- let SQL_escaped_file := "prompts_test." ++ mod.toString.replace "'" "''"
    let SQL_escaped_file := mod.toString.replace "'" "''"
    let SQL_cmd : String := s!"SELECT * FROM run_data WHERE decl = '{SQL_escaped_name}' AND module = '{SQL_escaped_file}';"
    IO.println s!"== [[{ci.name}]] =="
    let output ← IO.Process.output {
      cmd := "duckdb"--"/home/riyaza/.local/bin/duckdb",
      args := #[s!"{runPath}/data.duckdb", "--readonly", "--json", "-c", SQL_cmd]}
    -- IO.println s!"DuckDB output: {output.stdout}"
    -- IO.println s!"DuckDB err: {output.stderr}"
    -- IO.println s!"SQL command: {SQL_cmd}"
    if output.exitCode != 0 then
      -- IO.println s!"Error running duckdb: {output.stderr}"
      -- break
      return some #[]

    let json? := output.stdout
    -- IO.println s!"DuckDB output: {json?}"
    -- IO.println s!"DuckDB err: {output.stderr}"
    -- IO.println s!"DuckDB exit code: {output.exitCode}"
    -- IO.println s!"[==> variant_tuples?"--\n===={json?}\n===="
    -- IO.println "\n\n"
    -- IO.println s!"DuckDB output: {json?}"
    -- IO.println "\n\n"
    let json?? := Json.parse json? |>.toOption
    if json??.isNone then
      IO.println s!">>> Error parsing JSON: {json?}"
      return none

    IO.println s!"json: {json??.get!.compress}"

    let variant_tuples? :=
      let json := json??.get!
      match json with
      | .arr variants =>
        some (variants.filterMap (fun v =>
          let model_answer := (v : Json).getObjVal? "answer" |>.toOption
          let prompt := match (v : Json).getObjVal? "prompt" |>.toOption with
            | some p => some p
            | none => (v: Json).getObjVal? "raw_prompt" |>.toOption
          let decl_idx := (v : Json).getObjVal? "decl_idx" |>.toOption
          -- match (model_answer.toOption, prompt.toOption) with
          -- | (some (Json.str answer), some (Json.str prompt)) => some (cmd, ci, answer, prompt)
          -- | _ => none
          match (model_answer, prompt, decl_idx) with
          | (some (Json.str answer), some (Json.str prompt), some (Json.str decl_idx)) => some (cmd, ci, answer, prompt, decl_idx)
          | (some (Json.str answer), some (Json.str prompt), some (Json.num decl_idx)) => some (cmd, ci, answer, prompt, decl_idx.toString)
          | _ => none
        ))
      | _ =>
        none
    -- IO.println s!"<== variant_tuples? completed]"
    return some (variant_tuples?.getD #[])
    -- IO.sleep 1000
  ))

  let preinstances' ← preinstances_runner.mapM id
  let preinstances :=  preinstances'.filterMap id |>.flatten


  IO.println s!"Found {preinstances.size} variants"



  /- Multithreading stuff to verify each new proof on separate threads -/
  let tasks := preinstances.map fun (original, ci, model_output, prompt,decl_idx) => do --IO.asTask do

    IO.println s!"Evaluating {ci.name}"
    let contentsBefore : Substring := match original.src with
      | ⟨s, b, _⟩ => ⟨s, 0, b⟩
    -- IO.println "--------"
    -- IO.println s!"Original:"
    -- IO.println original.src.toString
    -- IO.println "--------"
    -- IO.println s!"New:"
    -- IO.println model_output
    -- IO.println "--------"
    -- IO.println s!"{model_output.trim.splitAtString "<IMPROVED>"}"
    -- IO.println "--------"
    -- IO.println s!"{model_output.trim.splitAtString "</IMPROVED>"}"

    let nonthinking_tokens := match model_output.trim.splitAtString "</think>" with
      | some (_, after) => after
      | none => model_output.trim
    let trimmed_output? := match (nonthinking_tokens.trim.splitAtString "<IMPROVED>", nonthinking_tokens.trim.splitAtString "</IMPROVED>") with
    | (some (_, after), none) => some after
    | (none, some (before, _)) => some before
    | (some (_, after), some _) =>
      let rest := after.trim.splitAtString "</IMPROVED>"
      match rest with
      | some (l,_) => some l
      | none => none
    | _ => none
    let trimmed_output := trimmed_output?.getD (model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim)
    -- let trimmed_output := model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim
    -- remove everything before the first <IMPROVED> tag and after the first </IMPROVED> tag
    -- let trimmed_output := match model_output.trim.getTagged "IMPROVED" with -- first match things in <IMPROVED>...</IMPROVED>
    --   | some x => x
    --   | none =>
    --     let endTagged := model_output.trim.splitAtString "</IMPROVED>"
    --     match endTagged with
    --     | some (before, _) => before
    --     | none =>
    --       let endTagged := model_output.trim.splitAtString "<IMPROVED>"
    --       match endTagged with
    --       | some (_, after) => after
    --       | none =>
    --         let half_tagged := model_output.trim.getBetween "</IMPROVED>" "</IMPROVED>"
    --         match half_tagged with
    --         | some x => x
    --         | none => model_output.trim.replace "<IMPROVED>" "" |>.replace "</IMPROVED>" "" |>.trim

    -- IO.println s!"trimmed output (length: {trimmed_output.length})"



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
        | none => (original, ci, model_output, trimmed_output, none, prompt, decl_idx)
        | some (head, _) => (original, ci, model_output,trimmed_output, some head, prompt, decl_idx)

    catch e =>
      IO.println s!"Error elaborating {ci.name}: {e}"
      IO.println "============="
      return (original, ci, model_output, trimmed_output, none, prompt, decl_idx)

    -- let head? ← elaborated_steps.uncons

  let results ← tasks.mapM id --fun (t : BaseIO _) => do
    --IO.ofExcept <| (← t).get

  IO.println " ==== GETTING INSTANCES ==== "
  let instances ← getInstances results metric mod.toString sorryOk correctnessCondition
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
  let metric := args.positionalArg! "metric" |>.as! String
  let runPath := args.positionalArg! "runPath" |>.as! String
  let outputPath := args.positionalArg! "outputPath" |>.as! String
  let mod :Name := module

  let sorryOk_raw := args.positionalArg! "sorryOk" |>.as! String
  let sorryOk := sorryOk_raw == "true" || sorryOk_raw == "1" || sorryOk_raw == "True"
  let correctnessCondition := args.positionalArg! "correctnessCondition" |>.as! String


  evalImprover mod metric runPath outputPath sorryOk correctnessCondition


def eval_improver : Cmd := `[Cli|
  eval_improver VIA evalImproverCLI; ["0.0.1"]
"Evaluate ImProver."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    metric : String; "Metric to use for evaluation."
    runPath : String; "Path to the run DB."
    outputPath : String; "Where to save the Json output."
    sorryOk : String; "Whether to allow 'sorry' in the output."
    correctnessCondition : String; "Condition to check correctness of the output."
]


def main (args : List String) : IO UInt32 :=
  eval_improver.validate args


-- #eval evalImprover `Mathlib.Logic.Hydra "length" "runs/RUN_20250515_031905" "temp.json"
