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
