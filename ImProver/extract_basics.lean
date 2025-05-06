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


def find_anomalies_dbg (file : String) (content : String)  : IO Unit := do
  let buf ← IO.FS.readFile file
  IO.FS.writeFile file (buf ++ "\n" ++ content)

def displayTime (pair : Nat× Nat) : String :=
  let (st, et) := pair
  let s := (et - st) / 1_000_000_000
  s!"{s}"


def ppDeclWithoutProof (module: ModuleName) (info: CommandInfo) : IO String := do
    -- (magic value) if this command is a declaration like theorem/def T := proof/definition
    -- then the := syntax occurs at `stx[1][3][0]`
    IO.println s!"{info.stx[1][3]}"
    if info.stx[1][3][0].getAtomVal == ":=" then
      let declStart := info.stx.getPos?.getD 0
      let proofStart := info.stx[1][3].getPos?.getD 0
      let proofEnd := info.stx.getTailPos?.getD 0
      let moduleSource ← moduleSource module
      let decl := (Substring.mk moduleSource declStart proofStart).toString
      let proof := (Substring.mk moduleSource proofStart proofEnd).toString
      return decl ++ ":= by sorry"
    else
      return ""

def to_completion_format (cmd : CompilationStep) (targetModule : ModuleName): IO String := do
  let findCommandNodes (t : InfoTree) : List CommandInfo :=
      let infos := t.findAllInfo none fun i => match i with
        | .ofCommandInfo _ => true
        | _ => false
      infos.filterMap fun p => match p with
      | (.ofCommandInfo i, _, _) => (i)
      | _ => none
  let ci := cmd.trees.flatMap (fun t => findCommandNodes t)
  let decls ← ci.mapM (fun c => ppDeclWithoutProof targetModule c)
  match decls with
  | [] => pure ""
  | x::_ =>
    pure x


/-- Main call to the ImProver framework
    Processes the theorems given as a list in `decls` in the module `targetModule`,
      and writes the original and improved proofs, along with relevant metrics, to the JSON file at `json_path` (if any) -/
def ImProver (config : ImProverConfig): IO Unit := do
  searchPathRef.set compile_time_search_path%



  let targetModule := config.targetModule
  let decls := config.decls
  let proofAsSorry? := config.proofAsSorry?
  let json_path := config.jsonPath



  let fileName := (← findLean targetModule).toString
  -- let mut trajectories_json := []



  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)


  let steps := Lean.Elab.IO.processInput' (← moduleSource targetModule) none (if proofAsSorry? then proofAsSorry else {}) fileName



  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)



  let mut targets_new : Array (CompilationStep × ConstantInfo) := #[]

  let anomaly_file := s!"anomalies/{config.model}/err_{config.targetModule.toString.replace "." "_"}.txt"
  IO.FS.createDirAll s!"anomalies/{config.model}"
  IO.FS.writeFile anomaly_file ""
  let log_anomaly := find_anomalies_dbg anomaly_file


  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    let isThm? := match ci with
      | .thmInfo _ => true
      | _ => false

    let pf_env := cmd.after
    let ctx : Core.Context := {fileName := "", fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1
      | none => false


    if (decls.isSome && !(decls.get!.contains ci_name_stem)) || not isThm? || not isHuman then
      continue
    targets_new := targets_new.push (cmd, ci)



  let targets_with_prompts ← get_prompt_batched config.prompt config targets_new


  let content := if targets_with_prompts.size != targets_new.size then
    s!"Generated {targets_with_prompts.size} prompts for {targets_new.size} targets. [ERR]"
  else
    s!"Good prompt generation [SUCC]"
  log_anomaly content

  for ((cmd, ci), prompt) in targets_new.zip targets_with_prompts do
    IO.println s!"============================================="
    IO.println s!"Processing {ci.name} in {targetModule}"
    IO.println cmd.src.toString
    -- IO.println cmd.src.str

    let findCommandNodes (t : InfoTree) : List CommandInfo :=
      let infos := t.findAllInfo none fun i => match i with
        | .ofCommandInfo _ => true
        | _ => false
      infos.filterMap fun p => match p with
      | (.ofCommandInfo i, _, _) => (i)
      | _ => none
    let ci := cmd.trees.flatMap (fun t => findCommandNodes t)
    let decls ← ci.mapM (fun c => ppDeclWithoutProof targetModule c)
    IO.println s!"Decls: {"\n\n".intercalate decls}"

    IO.println s!"---------------------------------------------"

    let data := InfoTree.tactics_new (cmd.trees) |>.map (fun x => x.info.stx)--(fun x => x.1) |>.map (fun x => s!"{x.stx}")

    -- IO.println s!"{data.map (fun x => "#"++ s!"{x}" ++ "\n\n")}"
    for x in data do

      IO.println s!"{x}"
      -- IO.println s!"******-----******"


      let s := match x with
        | Syntax.node _ k _ => s!"stx : {k}\n[-----------]"
        | _ => "UH OH"
      IO.println s
    let haves := data.filter (fun stx =>
      match stx with
      | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
      | _ => false)
    IO.println s!"{haves.map (fun x => "#"++ s!"{x}" ++ "\n\n")}"

    IO.println s!"=============================================="
  --   /- For each declaration, get the individual tactics involved in each one -/
  --   let tacs :=  InfoTree.tactics_new cmd.trees
  --   let tacs ← tacs.mapM (fun t => t.pp)
  --   IO.println s!"Number of tactics: {tacs.length}"
  --   IO.println s!"Tactics: {tacs}"

  --   IO.println s!"---------------------------------------------"

  --   /- Prompt the model for an improved version of the proof -/
  --   let newCommandCandidates ← promptModel_raw prompt cmd config anomaly_file

  --   for i in newCommandCandidates do
  --     IO.println s!"Model response: {i}"
  --     IO.println s!"---------------------------------------------"

  --   let content := if (newCommandCandidates.length != config.best_of_n) then
  --       s!"[{ci.name}] {newCommandCandidates.length} model responses recieved vs {config.best_of_n} expected [ERR]"
  --     else
  --       s!"[{ci.name}] good model responses [SUCC]"
  --   log_anomaly content

  --   /- Verify the new proof candidates -/
  --   let resultantSteps ← elaborateVariants cmd targetModule newCommandCandidates


  --   let content := if (resultantSteps.length != config.best_of_n) then
  --       s!"[{ci.name}] {resultantSteps.length} model responses elaborated vs {config.best_of_n} expected [ERR]"
  --     else
  --       s!"[{ci.name}] Good elaboration [SUCC]"
  --   log_anomaly content

  --   /- Create a list of structures that contain each original theorem, the model's (possibly) improved version, whether it worked, the goal state after each tactic, and relevant metrics -/
  --   let instances ← calculateInstancesWithPrompt ci cmd resultantSteps config prompt


  --   let content := if (instances.length != config.best_of_n) then
  --       s!"[{ci.name}] {instances.length} instances calculated vs {config.best_of_n} expected [ERR]"
  --     else
  --       s!"[{ci.name}] Good instance calculation [SUCC]"
  --   log_anomaly content



  --   /- Print out the results for each instance -/
  --   for i in instances do
  --     IO.println "-------------------------------------------------"

  --     let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, _, msgs, _, _⟩ := i
  --     IO.println s!"Name:\n {name}"
  --     IO.println s!"Original:\n {original}"
  --     IO.println s!"Correct: {oldCorrect}"
  --     IO.println s!"Metric: {oldScore}"
  --     IO.println s!"\n\nModel Output:\n {modelOutput}"
  --     IO.println s!"Correct: {newCorrect}"
  --     IO.println s!"Metric: {newScore}"
  --     IO.println s!"Delta: {delta}"
  --     for msg in msgs do
  --       IO.println msg
  --     IO.println "-------------------------------------------------"

  --   /- Make a JSON with the info we've gathered -/
  --   let trajectories_json_new := instances.map (fun i =>
  --     let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, oldMsgs, msgs, og_prompt, config⟩ := i
  --     Json.mkObj [
  --       ("module", Json.str config.targetModule.toString),
  --       ("decl", Json.str name),
  --       ("method", Json.str "best_of_n"),
  --       ("n", Json.num <| JsonNumber.fromNat config.best_of_n),
  --       ("metric", Json.str "LENGTH"),
  --       ("model", Json.str config.model),
  --       ("annotation", Json.bool config.annotation?),
  --       ("context", Json.bool config.context?),
  --       ("rag", Json.num <| JsonNumber.fromNat config.rag?),
  --       ("og_correct", Json.bool oldCorrect),
  --       ("og_errors", "\n\n".intercalate oldMsgs),
  --       ("og_score", Json.num <| (JsonNumber.fromFloat? (oldScore.getD (-1)) |>.getRight?).get!),
  --       ("new_correct", Json.bool newCorrect),
  --       ("new_errors", "\n\n".intercalate msgs),
  --       ("new_score", Json.num <| (JsonNumber.fromFloat? (newScore.getD (-1)) |>.getRight?).get!),
  --       ("delta", Json.num <| (JsonNumber.fromFloat? (delta.getD (-1)) |>.getRight?).get!),
  --       ("og_raw", Json.str original),
  --       ("new_raw", Json.str modelOutput),
  --       ("original_prompt", Json.str og_prompt),
  --       ("time",Json.num <| JsonNumber.fromInt (-1))
  --       ])
  --   trajectories_json := trajectories_json ++ trajectories_json_new
  -- /- If a path to a JSON has been provided, then write all the info there -/

  -- let trajectories := Json.arr (trajectories_json.toArray)
  -- match json_path with
  -- | some path =>
  --   IO.FS.writeFile path (trajectories.compress)
  -- | none => pure ()




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
  let example_file :=
   let ef := args.flag! "example_file" |>.as! String
   match ef with
   | "" => none
   | ef => some ef

  let config : ImProverConfig :=
    {targetModule:=mod, decls:=decls, model:=model, endpoint:=endpoint, best_of_n:=best_of_n, annotation?:=annotation, context? := context, rag? := rag, proofAsSorry?:=proofAsSorry, jsonPath:=json_path, example_file:=example_file}

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
    example_file : String; "Path to examples. (Blank for no examples)"


  ARGS:
    module : ModuleName; "Lean module to compile and annotate with state comments."

  EXTENSIONS:
    defaultValues! #[("decls", ""), ("json_path", ""),
    ("model", "DEBUG"), ("endpoint", "http://0.0.0.0:8000/v1/chat/completions"),
    ("best_of_n", "1"), ("annotation", "false"), ("context", "false"), ("rag", "0"), ("proofAsSorry", "false"), ("example_file", "")
    ]
]

/-- `lake exe state_comments` -/
def main (args : List String) : IO UInt32 :=
  improver.validate args




-- def test_config : ImProverConfig := {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`C04_S01_8])}
-- def test_config : ImProverConfig := {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets}
def test_config : ImProverConfig := {targetModule:=`temp.temp}
-- #eval ImProver test_config
-- def test_config : ImProverConfig := {targetModule:=`Compfiles.Imo2010P3, example_file:=some "improver_outputs_new/combined_examples/base_examples.txt"}

-- def test_config : ImProverConfig := {targetModule:=`temp.temp, decls:=(some [`theorem2]), rag?:=5}
-- def test_config : ImProverConfig := {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`t8]), best_of_n:= 1,model:="Llama-8B"}






#eval ImProver test_config


-- #eval ImProver {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`theorem8]), best_of_n:= 1, context?:=true}
-- #eval ImProver {targetModule:=`temp.temp, decls:=(some [`theorem8]), best_of_n:= 1, context?:=true}
