import ImProver.prompting.state_comments
import ImProver.prompting.context
import Cli
-- import ImProver.prompting.prompts
-- import ImProver.inference.inference
-- import ImProver.evaluation.eval
-- import ImProver.utils
-- import ImProver.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports
import TrainingData.TreeParser
import TrainingData.ExtractGoal

-- import ImProver.ProofTree.getPfTree

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
-- import Compfiles

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true




def _root_.Lean.Elab.Command.State.withOptions (state : Command.State) (options : Options) :=
  { state with
    scopes := state.scopes.map fun s : Scope =>
      { s with opts := Id.run do
          let mut opts := s.opts
          for (k, v) in options do
            opts := opts.insert k v
          opts } }


partial def maxEndPos (bp : ProofTree) (pos : Nat := 0) : Nat :=
  let curr := bp.node.tailPos.map (fun x => x.byteIdx) |>.getD pos
  let children := bp.children.map (fun child => maxEndPos child curr)
  children.foldl (init := curr) (fun acc x => max acc x)


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



def splitC2 (fileName : String) (cmd : CompilationStep) (breakpointType : String := "all_splits")
  : IO (List (String × List ExternalContext × (List String))) := do

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
      thms.map (fun thm => s!"lemma {thm} := by\n{cleanProof}"))
  -- IO.println s!"Split Theorems: \n{"\n".intercalate <| splits}\n"

  let mut output := []

  for (thms,bp) in splits.zip breakpoints do
    let pos := bp.node.pos.get!
    let endPos : String.Pos := ⟨maxEndPos bp pos.byteIdx⟩

    let dependencies_filtered := dependencies.filter (fun dep =>
      dep.pos.isSome && dep.endPos.isSome &&
      dep.pos.get!.byteIdx >= pos.byteIdx &&
      dep.endPos.get!.byteIdx <= endPos.byteIdx)

    let compiled := test_extracted_theorems thms contentsBefore cmd options fileName
    let (thm, io_msgs) ← compiled
    let msgs : (List String) ← io_msgs



    IO.println s!"Split Theorem: \n{thm}\n"
    IO.println s!"Dependencies: \n{dependencies_filtered.map (fun x => x.name.toString)}\n"
    IO.println s!"Messages: \n{msgs}\n"
    output := (thm, dependencies_filtered, msgs) :: output


  IO.println "===================="


  return output

structure TheoremID where
  name : Name
  module : Name
  content : Option String := none
  isExtracted : Bool := false
  errorMsgs : Array String := #[]
  deriving Inhabited, ToJson, FromJson, Repr

structure TheoremData where
  id : TheoremID
  C1_dependencies : Array TheoremID := #[]
  C2_dependencies : Array TheoremID := #[]

  -- fromSrc : Bool := false
  deriving Inhabited, ToJson, FromJson, Repr


def getKG (mod : Name) (outputDirectory : String): IO Unit := do
  searchPathRef.set compile_time_search_path%
  IO.println mod.toString
  let fileName := (← findLean mod).toString
  -- let mut trajectories_json := []
  let steps := Lean.Elab.IO.processInput' s!"import TrainingData.ExtractGoal\n{(← moduleSource mod)}" none {} fileName

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

  -- IO.println s!"Found {targets_new.size} targets"
  let mut outputs := []
  for (cmd, ci) in targets_new do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"

    let C1_raw ← get_context cmd ["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremID:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text})

    let C2_raw ← splitC2 fileName cmd "spawned"

    let mut extracted_thms : List TheoremData := []
    let mut C2_dependencies : List TheoremID := []
    -- errors = none means didn't compile, some [] means no errors, some [errors] means there were errors
    for ((thm, deps, errors), idx) in C2_raw.zipIdx do

      let split_thm : TheoremID :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          content := thm,
          isExtracted := true,
          errorMsgs := errors.toArray}

      C2_dependencies := split_thm :: C2_dependencies

      let split_data : TheoremData :=
        { id := split_thm,
          C1_dependencies := deps.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text}) |>.toArray,
          C2_dependencies := #[]
        }
      extracted_thms := split_data :: extracted_thms
      -- let deps' : List TheoremData := deps.map (fun ctx => {name := ctx.name, module := ctx.module, text := some ctx.text})
      -- let outer : TheoremData :=
      --   {name := s!"extracted_split_{ci.name}_{idx}".toName,
      --     module := mod,
      --     text := some thm,
      --     C1_dependencies := deps'.toArray,
      --     isExtracted := true,
      --     errorMsgs := errors'}
      -- split_data := outer :: split_data

    let id : TheoremID := {name := ci.name, module := mod, content := some cmd.src.toString}

    let mainData : TheoremData :=
      { id := id,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
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




def getKGCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  let mod :Name := module


  getKG mod outputDirectory
  return 0


def get_KG : Cmd := `[Cli|
  get_KG VIA getKGCLI; ["0.0.1"]
"Generate C1 KG."


  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    outputDirectory : String; "Where to save the Json output."
]


def main (args : List String) : IO UInt32 :=
  get_KG.validate args





-- #evPFal getKG `PFR.HomPFR "KG2.76"

-- #eval getPrompts `MIL.C07_Hierarchies.solutions.Solutions_S01_Basics "length" "temp" "prompt_examples"
