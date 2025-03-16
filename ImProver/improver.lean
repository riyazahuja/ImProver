-- import TrainingData.Frontend
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




/-- Main call to the ImProver framework
    Processes the theorems given as a list in `decls` in the module `targetModule`,
      and writes the original and improved proofs, along with relevant metrics, to the JSON file at `json_path` (if any) -/
def ImProver (config : ImProverConfig): IO Unit := do
  searchPathRef.set compile_time_search_path%

  -- let ⟨targetModule, decls, _, _, _, _,_, proofAsSorry?, json_path, metric_name, _⟩ := config
  let targetModule := config.targetModule
  let decls := config.decls
  let proofAsSorry? := config.proofAsSorry?
  let json_path := config.jsonPath
  let metric_name := config.metric


  let fileName := (← findLean targetModule).toString
  let mut trajectories_json := []

  /- Handle incomplete proofs with "sorry" in them -/
  /- TODO: I don't know if proofAsSorry is actually working -/

  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)

  /- Process the actual source code from our module -/
  let steps := Lean.Elab.IO.processInput' (← moduleSource targetModule) none (if proofAsSorry? then proofAsSorry else {}) fileName
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)


  let config ← if config.rag? == 0 then pure config
    else
      let fst ← steps.uncons
      let fst_step := match fst with
        | some (c, _) => initialize_retrieval config c
        | none => pure config
      fst_step


  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    if decls.isSome && !(decls.get!.contains ci_name_stem) then
      continue
    IO.println s!"============================================="
    IO.println s!"Processing {ci.name} in {targetModule}"

    /- For each declaration, get the individual tactics involved in each one -/
    let tacs :=  InfoTree.tactics_new cmd.trees
    let tacs ← tacs.mapM (fun t => t.pp)
    IO.println s!"Number of tactics: {tacs.length}"
    IO.println s!"Tactics: {tacs}"

    IO.println s!"---------------------------------------------"

    /- Print out what the verifier is yelling at us about -/
    let oldMsgs ← cmd.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))

    for m in oldMsgs do IO.eprintln m

    let old_correct := oldMsgs.isEmpty
    -- let old_score := if old_correct then some (tacs.length.toFloat) else none
    let old_score := if old_correct then some (get_metric metric_name cmd) else none

    /- Prompt the model for an improved version of the proof -/
    let newCommandCandidates ← promptModel cmd config

    /- Verify the new proof candidates -/
    let resultantSteps := (← elaborateVariants cmd targetModule newCommandCandidates).filterMap (fun x => x)

    /- Create a list of structures that contain each original theorem, the model's (possibly) improved version, whether it worked, the goal state after each tactic, and relevant metrics -/
    let instances : List ImprovedTheoremInstance ← resultantSteps.mapM (fun (model_output,head) => do

      let state_comments ← insert_state_comments head

      let msgs ← head.msgs.filterMapM (fun msg => do
        if msg.severity != .error then
          return none
        let m ← msg.data.toString
        return some (bombEmoji++m))

      let correct := msgs.isEmpty
      -- let metric_score := if correct then some (InfoTree.tactics_new head.trees |>.length |>.toFloat) else none
      let metric_score := if correct then some (get_metric metric_name head) else none

      let delta := if correct && old_correct then (
          if old_score.get! == 0 then
            some (-1 : Float)
          else
            some ((old_score.get! - metric_score.get!) / (old_score.get!))
          )
        else none

      let original_prompt ← get_prompt config.prompt config cmd
      let utilization ← if config.rag? == 0 then pure 0.0 else calculate_utilization original_prompt model_output



      return ImprovedTheoremInstance.mk ci.name.toString
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
        utilization
        config
    )

    /- Print out the results for each instance -/
    for i in instances do
      IO.println "-------------------------------------------------"

      let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, _, msgs, _, utilization, _⟩ := i
      IO.println s!"Name:\n {name}"
      IO.println s!"Original:\n {original}"
      IO.println s!"Correct: {oldCorrect}"
      IO.println s!"Metric: {oldScore}"
      IO.println s!"\n\nModel Output:\n {modelOutput}"
      IO.println s!"Correct: {newCorrect}"
      IO.println s!"Metric: {newScore}"
      IO.println s!"Delta: {delta}"
      IO.println s!"RAG Utilization: {utilization}"
      for msg in msgs do
        IO.println msg
      IO.println "-------------------------------------------------"

    /- Make a JSON with the info we've gathered -/
    let trajectories_json_new := instances.map (fun i =>
      let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, oldMsgs, msgs, og_prompt, utilization, config⟩ := i
      Json.mkObj [
        ("module", Json.str config.targetModule.toString),
        ("decl", Json.str name),
        ("method", Json.str "best_of_n"),
        ("n", Json.num <| JsonNumber.fromNat config.best_of_n),
        ("metric", Json.str "LENGTH"),
        ("model", Json.str config.model),
        ("annotation", Json.bool config.annotation?),
        ("context", Json.bool config.context?),
        ("rag", Json.num <| JsonNumber.fromNat config.rag?),
        ("rag_utilization", Json.num <| (JsonNumber.fromFloat? utilization |>.getRight?).get!),
        ("og_correct", Json.bool oldCorrect),
        ("og_errors", "\n\n".intercalate oldMsgs),
        ("og_score", Json.num <| (JsonNumber.fromFloat? (oldScore.getD (-1)) |>.getRight?).get!),
        ("new_correct", Json.bool newCorrect),
        ("new_errors", "\n\n".intercalate msgs),
        ("new_score", Json.num <| (JsonNumber.fromFloat? (newScore.getD (-1)) |>.getRight?).get!),
        ("delta", Json.num <| (JsonNumber.fromFloat? (delta.getD (-1)) |>.getRight?).get!),
        ("og_raw", Json.str original),
        ("new_raw", Json.str modelOutput),
        ("original_prompt", Json.str og_prompt),
        ("time",Json.num <| JsonNumber.fromInt (-1))
        ])
    trajectories_json := trajectories_json ++ trajectories_json_new
  /- If a path to a JSON has been provided, then write all the info there -/
  let trajectories := Json.arr (trajectories_json.toArray)
  match json_path with
  | some path =>
    IO.FS.writeFile path (trajectories.compress)
  | none => pure ()


/-- Configures a command-line interface for ImProver -/
def ImProver_CLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "module" |>.as! ModuleName
  let mod :Name := module
  let decls := args.flag! "decls" |>.as! String
  let json_path := args.flag! "json_path" |>.as! String
  let decls := if decls == "" then none else some (decls.splitOn "," |>.map String.toName)
  let json_path := if json_path == "" then none else some json_path
  let model := args.flag! "model" |>.as! String
  let endpoint := args.flag! "endpoint" |>.as! String
  let best_of_n := args.flag! "best_of_n" |>.as! Nat
  let annotation := args.flag! "annotation" |>.as! Bool
  let context := args.flag! "context" |>.as! Bool
  let rag := args.flag! "rag" |>.as! Nat

  let proofAsSorry := args.flag! "proofAsSorry" |>.as! Bool

  let config : ImProverConfig :=
    {targetModule:=mod, decls:=decls, model:=model, endpoint:=endpoint, best_of_n:=best_of_n, annotation?:=annotation, context? := context, rag? := rag, proofAsSorry?:=proofAsSorry, jsonPath:=json_path}

  ImProver config
  return 0

/-- Setting up command line options and help text for `lake exe state_comments`. -/
def improver : Cmd := `[Cli|
  improver VIA ImProver_CLI; ["0.0.1"]
"Run ImProver on (specific decls in a) Lean file."

  FLAGS:
    decls: String; "(comma-separated) List of declarations to process. (Blank for all)"
    model : String; "Model to use. (Default: DEBUG)"
    endpoint : String; "Endpoint to use. (Default: http://0.0.0.0:8000/v1/chat/completions)"
    best_of_n : Nat; "Number of attempts to make. (Default: 1)"
    annotation : Bool; "Forward proof states to model. (Default: false)"
    context : Bool; "Forward context to model. (Default: false)"
    rag : Nat; "Number of mathlib documents to retrieve. (Default: 0)"
    proofAsSorry : Bool; "Convert all tactics to \"sorry\" for faster execution. (Default: false)"
    json_path : String; "Path to save the JSON output. (Blank for stdout)"


  ARGS:
    module : ModuleName; "Lean module to compile and annotate with state comments."

  EXTENSIONS:
    defaultValues! #[("decls", ""), ("json_path", ""),
    ("model", "DEBUG"), ("endpoint", "http://0.0.0.0:8000/v1/chat/completions"),
    ("best_of_n", "1"), ("annotation", "false"), ("context", "false"), ("rag", "0"), ("proofAsSorry", "false")]
]

/-- `lake exe state_comments` -/
def main (args : List String) : IO UInt32 :=
  improver.validate args



def test_config : ImProverConfig := {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`t8])}
-- def test_config : ImProverConfig := {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`t8]), best_of_n:= 1,model:="Llama-8B"}


-- #eval ImProver test_config


-- #eval ImProver {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`theorem8]), best_of_n:= 1, context?:=true}
-- #eval ImProver {targetModule:=`temp.temp, decls:=(some [`theorem8]), best_of_n:= 1, context?:=true}
