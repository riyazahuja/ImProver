-- import TrainingData.Frontend
import Cli
import scripts.state_comments
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true

/- Configuration options for ImProver below -/
structure ImProverConfig where
  targetModule : Name
  decls : Option (List Name) := none
  model : String := "DEBUG"
  endpoint : String := "http://0.0.0.0:8000/v1/chat/completions"
  best_of_n : Nat := 1
  annotation? : Bool := false
  context? : Bool := false
  proofAsSorry : Bool := false
  jsonPath : Option String := none
  metric := "length"
  prompt := "default"

structure ExternalContext where
  name : Name
  kind : String
  module : Name
  text : String

/- Helper structure for containing info about potentially improved theorems (used in ImProver below) -/
structure ImprovedTheoremInstance where
  name : String
  originalTheorem : String
  modelOutput : String
  annotatedOutput : String
  oldCorrect : Bool
  newCorrect : Bool
  oldScore : Option Float
  newScore : Option Float
  delta : Option Float
  old_msgs : List String
  new_msgs : List String
  config : ImProverConfig


/- Metric of theorem improvement: number of tactics used in a theorem -/
def metric_length (cmd:CompilationStep) :=
  InfoTree.tactics_new cmd.trees |>.length |>.toFloat

/- Returns metric function from name (for ease of use from command line) -/
def get_metric (metric_name : String) : CompilationStep → Float :=
  match metric_name with
  | "length" => metric_length
  | _ => fun _ => 0.0

/- Model system prompt asking it to shorten the length of the theorem -/
def length_prompt (config : ImProverConfig) (cmd : CompilationStep) : IO String := do
  let srcCommand ← if config.annotation? then (insert_state_comments cmd) else pure cmd.src.toString
  let annotation_prompt : String := s!" The goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. "
  let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.{if config.annotation? then annotation_prompt else " "}Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  return prompt

/- Returns the prompt function from a name -/
def get_prompt (prompt_name : String) :=
  match prompt_name with
  | _ => length_prompt


def getUsedConstantsAsSet (t : TacticInfo) : NameSet :=
  let set := t.goalsBefore
    |>.filterMap t.mctxAfter.getExprAssignmentCore?
    |>.map Expr.getUsedConstantsAsSet
    |>.foldl .union .empty

  set

def getKind (const_map : ConstMap) (m : Name) : String :=
  let local_const := const_map.map₂
  let ext_const := const_map.map₁
  let c := local_const.find? m
  match c with
  | none => match ext_const[m]? with
    | some d => match d with
      | .axiomInfo _ => "axiom"
      | .defnInfo _ => "def"
      | .thmInfo _ => "theorem"
      | .opaqueInfo _ => "opaque"
      | .quotInfo _ => "quot"
      | .inductInfo _ => "inductive"
      | .ctorInfo _ => "constructor"
      | .recInfo _ => "recursor"
    | none => "Not Found"
  | some c => match c with
    | .axiomInfo _ => "axiom (internal)"
    | .defnInfo _ => "def (internal)"
    | .thmInfo _ => "theorem (internal)"
    | .opaqueInfo _ => "opaque (internal)"
    | .quotInfo _ => "quot (internal)"
    | .inductInfo _ => "inductive (internal)"
    | .ctorInfo _ => "constructor (internal)"
    | .recInfo _ => "recursor (internal)"

def get_context (step:CompilationStep) : IO (List ExternalContext) := do
  let constants := step.trees
    |>.flatMap InfoTree.retainTacticInfo
    |>.flatMap InfoTree.retainOriginal
    |>.flatMap InfoTree.retainSubstantive
    |>.flatMap (fun t => t.findTacticNodes.map (fun ⟨i, _⟩ => (getUsedConstantsAsSet i).toList))
    |>.flatten
    |>.eraseDups

  let pf_env := step.commandStateBefore.env
  let modules := constants.map (fun c => (c,pf_env.getModuleFor? c |>.getD (Name.anonymous)))
  let consts_mods_kind := modules.map (fun (c, m) => (c, m, getKind pf_env.constants c))
  let mods := (modules.map fun x => x.2) |>.eraseDups |>.filter fun m => m != Name.anonymous
  let constant_info ← CoreM.withImportModules mods.toArray do
    let mut out := []
    for (c, module, kind) in consts_mods_kind do
      let rgs := ((← findDeclarationRanges? c).getD default).range
      -- let module := ((pf_env.getModuleFor? c).getD (Name.anonymous))
      let modulePath ← findLean module
      let fileContents ← IO.FS.readFile modulePath.toString
      -- get the source code within range
      let declTextList := if rgs.pos.line == 0 then [] else
        fileContents.splitOn "\n"
          |>.drop (rgs.pos.line - 1)
          |>.take (rgs.endPos.line - rgs.pos.line + 1)
      let declText := "\n".intercalate declTextList

      out := (ExternalContext.mk c kind module declText)::out
    return out
  return constant_info




/- Returns a dummy response (for debugging when model is offline) -/
def promptModel_debug (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  let srcCommand := cmd.src.toString
  let bon := config.best_of_n
  return List.range bon |>.map (fun i => s!"--DEBUG: {i}\n{srcCommand}")


def promptModel_server (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  let ⟨_,_,model, endpoint, best_of_n, annotation?, context?, _, _, _, prompt_name⟩ := config

  -- let srcCommand ← if annotation? then (insert_state_comments cmd) else pure cmd.src.toString

  -- -- IO.println s!"srcCommand:\n{srcCommand.dropRightWhile (· == '\n')}"
  -- let annotation_prompt : String := s!" The goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. "
  -- let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.{if annotation? then annotation_prompt else " "}Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  let prompt ← get_prompt prompt_name config cmd

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
  let ⟨_,_,model, endpoint, best_of_n, annotation?,context?, _, _, _, prompt_name⟩ := config

  let srcCommand ← if annotation? then (insert_state_comments cmd) else pure cmd.src.toString

  -- IO.println s!"srcCommand:\n{srcCommand.dropRightWhile (· == '\n')}"
  -- let annotation_prompt : String := s!" The goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. "
  -- let prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.{if annotation? then annotation_prompt else " "}Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  let prompt ← get_prompt prompt_name config cmd

  let jsonPayload : Json := Json.mkObj [
      ("model", Json.str model),
      ("messages", Json.arr #[Json.mkObj [("role",Json.str "user"),("content", Json.str prompt)]]),
      ("max_tokens", Json.num <| JsonNumber.fromNat 256)
    ]
  -- Call Python script with JSON payload
  let out ← IO.Process.output {
    cmd := "/home/riyaza/miniconda3/envs/.venv10/bin/python3",
    args := #["scripts/send_batched.py", jsonPayload.compress, toString best_of_n, endpoint]
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


/- Prompts the model running on an available web interface
  Takes a (compiled) theorem, a model name, an endpoint (URL to interface), and the number of separate attempts the model should make (best_of_n) -/
def promptModel (cmd : CompilationStep) (config : ImProverConfig) : IO (List String) := do
  match config.model with
  | "DEBUG" => return ← promptModel_debug cmd config
  | _ => return ← promptModel_batched cmd config




/- Not sure what this is for but the file doesn't run without it :| -/
def _root_.Lean.Elab.Command.State.withOptions (state : Command.State) (options : Options) :=
  { state with
    scopes := state.scopes.map fun s : Scope =>
      { s with opts := Id.run do
          let mut opts := s.opts
          for (k, v) in options do
            opts := opts.insert k v
          opts } }


/- An efficient way to verify each new proof candidate that the model outputs
    Requires the original proof's compilation steps, the module name, and a list of new proof candidates (as strings) to verify -/
def elaborateVariants (original : CompilationStep) (mod: Name) (variants : List String) : IO (List (Option (String × CompilationStep))) := do
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
      | none => none
      | some (head, _) => some (newCommand,head)

  let results ← tasks.mapM fun (t : BaseIO _) => do
    IO.ofExcept <| (← t).get
  return results

/-- Main call to the ImProver framework
    Processes the theorems given as a list in `decls` in the module `targetModule`,
      and writes the original and improved proofs, along with relevant metrics, to the JSON file at `json_path` (if any) -/
def ImProver (config : ImProverConfig): IO Unit := do
  searchPathRef.set compile_time_search_path%
  let ⟨targetModule, decls, _, _, _, _,_, proofAsSorry?, json_path, metric_name, _⟩ := config
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

      return ⟨ci.name.toString, cmd.src.toString, model_output, state_comments, old_correct, correct, old_score, metric_score, delta, oldMsgs, msgs, config⟩
    )

    /- Print out the results for each instance -/
    for i in instances do
      IO.println "-------------------------------------------------"

      let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, _, msgs, _⟩ := i
      IO.println s!"Name:\n {name}"
      IO.println s!"Original:\n {original}"
      IO.println s!"Correct: {oldCorrect}"
      IO.println s!"Metric: {oldScore}"
      IO.println s!"\n\nModel Output:\n {modelOutput}"
      IO.println s!"Correct: {newCorrect}"
      IO.println s!"Metric: {newScore}"
      IO.println s!"Delta: {delta}"
      for msg in msgs do
        IO.println msg
      IO.println "-------------------------------------------------"

    /- Make a JSON with the info we've gathered -/
    let trajectories_json_new := instances.map (fun i =>
      let ⟨name, original, modelOutput, _, oldCorrect, newCorrect, oldScore, newScore, delta, oldMsgs, msgs, config⟩ := i

      Json.mkObj [

        ("module", Json.str config.targetModule.toString),
        ("decl", Json.str name),
        ("method", Json.str "best_of_n"),
        ("n", Json.num <| JsonNumber.fromNat config.best_of_n),
        ("metric", Json.str "LENGTH"),
        ("model", Json.str config.model),
        ("annotation", Json.bool config.annotation?),
        ("syntax_search", Json.bool false),
        ("mathlib_search", Json.bool false),
        ("examples", Json.num <| JsonNumber.fromNat 0),
        ("og_correct", Json.bool oldCorrect),
        ("og_errors", "\n\n".intercalate oldMsgs),
        ("og_score", Json.num <| (JsonNumber.fromFloat? (oldScore.getD (-1)) |>.getRight?).get!),
        ("new_correct", Json.bool newCorrect),
        ("new_errors", "\n\n".intercalate msgs),
        ("new_score", Json.num <| (JsonNumber.fromFloat? (newScore.getD (-1)) |>.getRight?).get!),
        ("delta", Json.num <| (JsonNumber.fromFloat? (delta.getD (-1)) |>.getRight?).get!),
        ("og_raw", Json.str original),
        ("new_raw", Json.str modelOutput),
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
  let proofAsSorry := args.flag! "proofAsSorry" |>.as! Bool

  let config : ImProverConfig := {targetModule:=mod, decls:=decls, model:=model, endpoint:=endpoint, best_of_n:=best_of_n, annotation?:=annotation, proofAsSorry:=proofAsSorry, jsonPath:=json_path}

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
    proofAsSorry : Bool; "Convert all tactics to \"sorry\" for faster execution. (Default: false)"
    json_path : String; "Path to save the JSON output. (Blank for stdout)"

  ARGS:
    module : ModuleName; "Lean module to compile and annotate with state comments."

  EXTENSIONS:
    defaultValues! #[("decls", ""), ("json_path", ""),
    ("model", "DEBUG"), ("endpoint", "http://0.0.0.0:8000/v1/chat/completions"),
    ("best_of_n", "1"), ("annotation", "false"), ("proofAsSorry", "false")]
]

/-- `lake exe state_comments` -/
def main (args : List String) : IO UInt32 :=
  improver.validate args


#eval ImProver {targetModule:=`MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets, decls:=(some [`t1]), best_of_n:= 1}
