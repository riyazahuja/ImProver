import ImProver.online.prompting.state_comments
import ImProver.online.prompting.context
import Cli
import ImProver.online.prompting.prompts
import ImProver.online.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports
import TrainingData.TreeParser
import TrainingData.ExtractGoal
import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true





partial def maxEndPos (bp : ProofTree) (pos : Nat := 0) : Nat :=
  let curr := bp.node.tailPos.map (fun x => x.byteIdx) |>.getD pos
  let children := bp.children.map (fun child => maxEndPos child curr)
  children.foldl (init := curr) (fun acc x => max acc x)

-- given a list of possible theorems, return the first that compiles without errors
def test_extracted_theorems (thms : List String)
  (contentsBefore : Substring) (cmd : CompilationStep)
  (options : Options) (fileName : String)
  (best : (String × IO (List String)) := (thms.getLast!, pure ["Unexpected error: Failed to compile"]))
  : IO ((String × IO (List String))) := do

  match thms with
  | [] =>
    IO.println s!"No more theorems to test, returning best: {best.1}"
    return best
  | thm :: rest =>
    IO.println s!"Testing theorem: {thm}"

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (contentsBefore.toString ++ thm) fileName)
      cmd.parserStateBefore
      (cmd.commandStateBefore.withOptions options)

    let head? ← elaborated_steps.uncons
    let cstep? : Option CompilationStep := match head? with
    | none => none
    | some (cstep, _) => some cstep

    let msgs : IO (List String) := match cstep? with
      | none => return ["Unexpected error: failed to compile"]
      | some cstep => do
        let filtered_msg := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.error)
        let strMsg ← filtered_msg.mapM (fun (msg : Message) => msg.toString)
        return strMsg
    IO.println s!"Messages: {← msgs}"
    IO.println "===================="
    if (← msgs).isEmpty then
      return (thm, pure [])
    else
      let new_best2 := (← best.2) ++ (← msgs)
      let new_best := (best.1, pure new_best2)
      test_extracted_theorems rest contentsBefore cmd options fileName new_best

-- given two theorems, compile the latter and get its error msgs
def test_extracted_theorems' (thms : List String)
  (contentsBefore : Substring) (cmd : CompilationStep)
  (options : Options) (fileName : String)
  : IO (Option (String × String × IO (List String))) := do

  match thms with
  | pp::stx::[] =>
    IO.println s!"Testing theorem: {pp}"

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (contentsBefore.toString ++ stx) fileName)
      cmd.parserStateBefore
      (cmd.commandStateBefore.withOptions options)

    let head? ← elaborated_steps.uncons
    let cstep? : Option CompilationStep := match head? with
    | none => none
    | some (cstep, _) => some cstep

    let msgs : IO (List String) := match cstep? with
      | none => return ["Unexpected error: failed to compile"]
      | some cstep => do
        let filtered_msg := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.error)
        let strMsg ← filtered_msg.mapM (fun (msg : Message) => msg.toString)
        return strMsg
    IO.println s!"Messages: {← msgs}"
    IO.println "===================="
    return some (pp, stx, msgs)
  | _ => return none


partial def getIndentSize (line : String) : Nat :=
  let rec countLeadingSpaces (idx : Nat) : Nat :=
    if idx < line.length then
      match line.get ⟨idx⟩ with
      | ' ' => countLeadingSpaces (idx + 1)
      | '\t' => countLeadingSpaces (idx + 2) -- assuming 2 spaces per tab
      | _ => idx
    else
      idx

  countLeadingSpaces 0

def removeCommonIndent (s : String) : String :=
    let lines := s.splitOn "\n"
    -- Handle empty string or single line without indentation
    if lines.isEmpty || lines.all (fun line => line.trim.isEmpty) then
      s
    else
      -- Find common indent by finding minimum number of leading spaces in non-empty lines
      let nonEmptyLines := lines.filter (fun line => !line.trim.isEmpty)
      let commonIndent := nonEmptyLines.map getIndentSize |>.min?.getD 0

      -- Remove common indent from each line
      let trimmedLines := lines.map (fun line =>
        if line.length ≤ commonIndent || line.trim.isEmpty then
          line.trim
        else
          line.drop commonIndent)

      "\n".intercalate trimmedLines



-- given a theorem cmd, return a list of split (theorem (pp), theorem syntax, context items, error msgs)
def splitC2 (fileName : String) (cmd : CompilationStep) (breakpointType : String := "all_splits")
  : IO (List (String × String × List ExternalContext × (List String))) := do

  IO.println s!"Processing: {cmd.src.toString}"


  let dependencies ← get_context cmd ["theorem", "theorem (internal)"]
  IO.println s!"Dependencies: {dependencies.map (fun x => x.name.toString)}"


  let tree? := getProofTree <| (← (cmd.trees.filterMapM (BetterParser)) ).flatMap (fun result => result.steps)
  if tree?.isNone then
    IO.println s!"Failed to parse the proof tree."
    return []

  let tree := tree?.get!
  IO.println s!"Proof Tree: \n{tree}\n"
  let breakpoints : List ProofTree := tree.getBreakpointsWithDescendents breakpointType
    |>.filter (fun pt => pt.node.pos.isSome)

  IO.println s!"Breakpoints: {breakpoints.map (fun pt => pt.node.tacticString)}"
  let new_thm := insertBreakpointsFromTree' cmd.src.toString (breakpoints.map (fun pt => pt.node))

  IO.println s!"New theorem: \n{new_thm}\n"

  let contentsBefore : Substring := match cmd.src with
  | ⟨s, b, _⟩ => ⟨s, 0, b⟩

  let options := ({} : KVMap)
    |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
    |>.insert `debug.byAsSorry (.ofBool false)
    |>.insert `linter.unusedVariables (.ofBool true)
    |>.insert `linter.unusedTactic (.ofBool true)
    |>.insert `linter.unreachableTactic (.ofBool true)

  let elaborated_steps := Lean.Elab.IO.compilationSteps
    (Parser.mkInputContext (contentsBefore.toString ++ new_thm) fileName)
    cmd.parserStateBefore
    (cmd.commandStateBefore.withOptions options)

  let head? ← elaborated_steps.uncons
  let outputted : Option CompilationStep := match head? with
  | none => none
  | some (cstep, _) => some cstep

  let grouped_msgs : List (List String) ← match outputted with
  | none => return []
  | some cstep =>
    let filtered_msgs := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.information)
    let grouped := filtered_msgs.splitBy (fun (m m': Message) => m.pos == m'.pos && m.endPos == m'.endPos)
    grouped.mapM (fun msgs => msgs.mapM (fun msg => msg.toString) )

  -- let msgs : IO (List String) := match outputted with
  --   | none => pure []
  --   | some cstep => do
  --     let filtered_msg := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.information)
  --     let output ← filtered_msg.mapM (fun msg => msg.toString)
  --     return output

  let grouped_msgs := grouped_msgs.map (fun msgs => msgs.reverse)


  let getProof (bp : ProofTree) : String :=
    let pos := bp.node.pos.get!
    let endPos := ⟨maxEndPos bp pos.byteIdx⟩
    let sstr : Substring := ⟨cmd.src.str,pos,endPos⟩
    sstr.toString

  -- let indent (s : String) : String :=
  --   "\n".intercalate <| s.splitOn "\n"
  --   |>.map (fun line => "  " ++ line)


  let breakpoint_splits : List (List String × String) := (grouped_msgs).zip (breakpoints.map getProof)
  let splits : List (List String):=
    breakpoint_splits.map (fun (thms, pf) =>
      let (firstLine, rest) :=
        match pf.splitOn "\n" with
        | [] => ("", "")
        | [line] => (line, "")
        | first :: rest => (first, "\n".intercalate rest)
      let cleanedRest := removeCommonIndent rest
      let cleanProof := firstLine ++ (if rest == "" then "" else "\n" ++ cleanedRest)
      let thms_raw := thms.map (fun thm => s!"lemma {thm} := by\n{cleanProof}")
      thms_raw

      )

  -- IO.println s!"Split Theorems: \n{"\n".intercalate <| splits}\n"

  let mut output := []
  -- must be 2 thms, first is pp, second is syntax
  for (thms,bp) in splits.zip breakpoints do
    let pos := bp.node.pos.get!
    let endPos : String.Pos := ⟨maxEndPos bp pos.byteIdx⟩

    let dependencies_filtered := dependencies.filter (fun dep =>
      dep.pos.isSome && dep.endPos.isSome &&
      dep.pos.get!.byteIdx >= pos.byteIdx &&
      dep.endPos.get!.byteIdx <= endPos.byteIdx)

    let compiled? ← test_extracted_theorems' thms contentsBefore cmd options fileName

    if compiled?.isNone then
      IO.println s!"Failed to compile any of the split theorems: {thms}"
      continue

    let (pp_thm, stx_thm, io_msgs) := compiled?.get!
    let msgs : (List String) ← io_msgs



    IO.println s!"Split Theorem: \n{pp_thm}\n"
    IO.println s!"Dependencies: \n{dependencies_filtered.map (fun x => x.name.toString)}\n"
    IO.println s!"Messages: \n{msgs}\n"
    output := (pp_thm, stx_thm, dependencies_filtered, msgs) :: output


  IO.println "===================="


  return output

structure TheoremID where
  name : Name
  module : Name
  content : Option String := none
  compilationAlias : Option String := none
  isExtracted : Bool := false
  errorMsgs : Array String := #[]
  kind : String := "theorem"
  deriving Inhabited, ToJson, FromJson, Repr

structure TheoremData where
  id : TheoremID
  annotation : String := ""
  content_sorry : String := ""
  goal : String := ""
  rag : Json := Json.arr #[]
  C1_dependencies : Array TheoremID := #[]
  C2_dependencies : Array TheoremID := #[]

  -- fromSrc : Bool := false
  deriving Inhabited, ToJson, FromJson




def getPrompts (mod : Name) (outputDirectory : String) (python_cmd : String) (theorems : List String): IO Unit := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

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

    let curr_name_variants :=
      let fullName := ci.name.toString
      let nameParts := fullName.splitOn "."
      let rec buildVariants (remaining : List String) (acc : List String) :=
        match remaining with
        | [] => acc
        | _ :: rest =>
          let currVariant := ".".intercalate remaining
          buildVariants rest (currVariant :: acc)
      buildVariants nameParts []

    let included? := curr_name_variants.map (fun n => theorems.contains n) |>.any id

    if (not theorems.isEmpty && not included?) then
      continue
    targets_new := targets_new.push (cmd, ci)

  IO.println s!"==== Got {targets_new.size} targets from {mod.toString} ===="

  let rag_strings : Array Json ← do
      let items ← retrieve_batch_indep targets_new python_cmd
      let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
      pure x


  IO.println s!"==== Got {rag_strings.size} prompts from RAG ===="


  let mut outputs := []
  for ((cmd, ci), rag) in targets_new.zip rag_strings do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"



    let srcCommand := cmd.src.toString

    -- eventually want annotation on partial proofs, but for now, ignore
    let annotation_string : String ← insert_state_comments cmd


    -- let context_string : Json ← do
    --     let context ← get_context cmd
    --     pure <| Json.arr <| context.map (fun c : ExternalContext => Json.mkObj [
    --         ("name", Json.str c.name.toString),
    --         ("context_item_type", Json.str c.kind),
    --         ("content", Json.str c.text)
    --       ]) |>.toArray

    let pfAsSorry := proofAsSorry cmd |>.getD ""

    let initialGoal ←  getInitialProofState2 cmd


    let C1_raw ← get_context cmd --["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremID:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text, kind:= ctx.kind})

    let C2_raw ← splitC2 fileName cmd "spawned"

    let mut extracted_thms : List TheoremData := []
    let mut C2_dependencies : List TheoremID := []
    -- errors = none means didn't compile, some [] means no errors, some [errors] means there were errors
    for ((pp_thm, stx_thm, deps, errors), idx) in C2_raw.zipIdx do

      let split_thm : TheoremID :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          content := pp_thm,
          compilationAlias := stx_thm,
          isExtracted := true,
          errorMsgs := errors.toArray
        }

      C2_dependencies := split_thm :: C2_dependencies

      let split_data : TheoremData :=
        { id := split_thm,
          C1_dependencies := deps.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text}) |>.toArray,
          C2_dependencies := #[]
        }
      extracted_thms := split_data :: extracted_thms

    let id : TheoremID := {name := ci.name, module := mod, content := some srcCommand, compilationAlias := some srcCommand}

    let mainData : TheoremData :=
      { id := id,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
        annotation := annotation_string,
        content_sorry := pfAsSorry,
        goal := initialGoal,
        rag := rag
        }

    outputs := mainData :: extracted_thms ++ outputs


  let json_data := Json.arr <| outputs.toArray.map (ToJson.toJson)

  let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  IO.println s!"Writing to {json_path}"

  if not (← System.FilePath.pathExists json_path) then
    let parent := System.FilePath.parent json_path
    match parent with
    | some path =>
      IO.println path
      IO.FS.createDirAll path
    | none => pure ()



  IO.FS.writeFile json_path (ToString.toString json_data)





  -- -- let prompt_data : Array (ConstantInfo × Json × Json × Json × Json) ← cmds_ci.mapM (fun (cmd,(ci : ConstantInfo)) => do

  -- --   return (ci, Json.str srcCommand, Json.str pfAsSorry, Json.str annotation_string, context_string)
  -- -- )

  -- IO.println s!"==== GOT {prompt_data.size} prompts from {mod}!!! ===="

  -- let rag_strings : Array Json ← do
  --     let items ← retrieve_batch_indep cmds_ci python_cmd
  --     let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
  --     pure x
  -- IO.println s!"==== GOT {rag_strings.size} prompts from RAG!!! ===="


  -- let prompt_data := prompt_data.zip rag_strings |>.map (fun ((ci,srcCommand, pfAsSorry, annotation_string,context_string),rag_string) =>
  --   let data := Json.mkObj [
  --     -- ("system", Json.str main_prompt),
  --     -- ("example_prompt", Json.str example_prompt),
  --     -- ("examples", example_json),
  --     -- ("context_prompt", Json.str context_prompt),
  --     ("context", context_string),
  --     -- ("rag_prompt", Json.str rag_prompt),
  --     ("rag", rag_string),
  --     -- ("annotation_prompt", Json.str annotation_prompt),
  --     ("annotation", annotation_string),
  --     ("current", srcCommand),
  --     ("current_sorry", pfAsSorry),
  --     ]
  --   ((ci : ConstantInfo).name.toString, data)
  --   )

  -- pure <| Json.mkObj prompt_data.toList














  -- let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
  -- IO.println s!"Writing to {json_path}"
  -- -- let trajectories := Json.arr (trajectories_json.toArray)
  -- -- match json_path with
  -- -- | some path =>
  -- if not (← System.FilePath.pathExists json_path) then
  --   let parent := System.FilePath.parent json_path
  --   match parent with
  --   | some path =>
  --     IO.println path
  --     IO.FS.createDirAll path
  --   | none => pure ()

  -- IO.println s!"Path exists, now writing:\n{targets_with_prompts}"

  -- IO.FS.writeFile json_path (targets_with_prompts.compress)
  -- -- | none => pure ()






def getPromptsCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let python_cmd := args.positionalArg! "pythonCommand" |>.as! String
  let mod :Name := module
  let theorems_raw : String := args.positionalArg! "theorems" |>.as! String
  let theorems : List String := if theorems_raw.isEmpty then [] else theorems_raw.splitOn ","



  getPrompts mod outputDirectory python_cmd theorems
  return 0


def get_prompts : Cmd := `[Cli|
  get_prompts VIA getPromptsCLI; ["0.0.1"]
"Generate prompts for ImProver."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    outputDirectory : String; "Where to save the Json output."
    pythonCommand : String; "Path to python executable."
    theorems : String; "List of theorems to include in the prompts, separated by \",\". If empty, all theorems in the module will be used."

]


def main (args : List String) : IO UInt32 :=
  get_prompts.validate args
