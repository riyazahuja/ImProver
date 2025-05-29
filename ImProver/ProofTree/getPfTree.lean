import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import TrainingData.TreeParser
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import ImProver.prompting.context
import TrainingData.Utils.HumanTheorem
-- import Batteries.Lean.Util.Path
import Batteries.Data.String.Basic
import Mathlib.Tactic.Change
import Cli
import ImProver.ProofTree.Utils


open Lean Elab IO Meta
open Lean Core Elab IO Meta Term Command Tactic Cli



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


def splitC2 (fileName : String) (cmd : CompilationStep) (breakpointType : String := "all_splits") : IO (List (String × List ExternalContext)) := do

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

  let msgs : IO (List String) := match outputted with
    | none => pure []
    | some cstep => do
      let filtered_msg := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.information)
      let output ← filtered_msg.mapM (fun msg => msg.toString)
      return output



  let getProof (bp : ProofTree) : String :=
    let pos := bp.node.pos.get!
    let endPos := ⟨maxEndPos bp pos.byteIdx⟩
    let sstr : Substring := ⟨cmd.src.str,pos,endPos⟩
    sstr.toString

  let indent (s : String) : String :=
    "\n".intercalate <| s.splitOn "\n"
    |>.map (fun line => "  " ++ line)


  let breakpoint_splits : List (String × String) := (← msgs).zip (breakpoints.map getProof)
  let splits := breakpoint_splits.map (fun (thm, pf) => thm.replace "sorry" s!"by\n{indent pf}")
  -- IO.println s!"Split Theorems: \n{"\n".intercalate <| splits}\n"

  let mut output := []

  for (thm,bp) in splits.zip breakpoints do
    let pos := bp.node.pos.get!
    let endPos : String.Pos := ⟨maxEndPos bp pos.byteIdx⟩

    let dependencies_filtered := dependencies.filter (fun dep =>
      dep.pos.isSome && dep.endPos.isSome &&
      dep.pos.get!.byteIdx >= pos.byteIdx &&
      dep.endPos.get!.byteIdx <= endPos.byteIdx)



    IO.println s!"Split Theorem: \n{thm}\n"
    IO.println s!"Dependencies: \n{dependencies_filtered.map (fun x => x.name.toString)}\n"
    output := (thm, dependencies_filtered) :: output


  IO.println "===================="


  -- -- if evaling only!!!
  -- let mut outputs := []
  -- for thm in splits do

  --   let elaborated_steps := Lean.Elab.IO.compilationSteps
  --     (Parser.mkInputContext (contentsBefore.toString ++ thm) fileName)
  --     cmd.parserStateBefore
  --     (cmd.commandStateBefore.withOptions options)

  --   let head? ← elaborated_steps.uncons
  --   let cstep : Option CompilationStep := match head? with
  --   | none => none
  --   | some (cstep, _) => some cstep

  --   let msgs : IO (List String) := match outputted with
  --     | none => pure []
  --     | some cstep => do
  --       let filtered_msg := cstep.msgs.filter (fun (msg : Message) => msg.severity == MessageSeverity.error)
  --       let output ← filtered_msg.mapM (fun msg => msg.toString)
  --       return output

  --   if not <| (← msgs).isEmpty then
  --     IO.println s!"Failed to elaborate the split theorem: {thm}"
  --     IO.println s!"Errors: \n{"\n".intercalate (← msgs)}"
  --     continue
  --   else
  --     -- IO.println s!"Successfully elaborated the split theorem: {thm}"
  --     outputs := cstep :: outputs



  -- IO.println "SPLITS:"
  --   -- IO.println s!"Split Theorem: \n{thm}\n"
  -- let o := outputs.filterMap (fun cs => match cs with
  --   | none => none
  --   | some cstep => cstep.src.toString)
  -- IO.println s!"{"\n".intercalate o}\n"


  -- make a dependency of cmd for each split

  -- for each split, add its dependencies


  return output


def trainingData (args : Cli.Parsed) : IO UInt32 := do
    searchPathRef.set compile_time_search_path%

    let mod := args.positionalArg! "module" |>.as! ModuleName
    let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String

    let fileName := (← findLean mod).toString
    -- let mut trajectories_json := []
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

      targets_new := targets_new.push (cmd, ci)

    let data ← targets_new.mapM (fun (cmd, ci) => do
      let x ← splitC2 fileName cmd
      return (ci, x)
      )

    let basic := data.map (fun (ci, splits) => -- for each theorem, and its splits
      let dependencies := splits.zipIdx.map (fun ((thm, _), idx) =>
        Json.mkObj [
          ("name", Json.str s!"extracted_split_{ci.name}_{idx}"),
          ("module", Json.str mod.toString),
          ("text", Json.str thm)
          ]
        )
      Json.mkObj [
        ("name", Json.str ci.name.toString),
        ("module", Json.str mod.toString),
        ("dependencies", Json.arr dependencies.toArray)]
      )
    let dependency_deps := data.map (fun (ci, splits) => -- for each theorem, and its splits
      splits.toArray.zipIdx.filterMap (fun ((_, deps), idx) =>
        let dep_deps := deps.map (fun (dep : ExternalContext) =>
          Json.mkObj [
            ("name", Json.str dep.name.toString),
            ("module", Json.str dep.module.toString),
            ("text", Json.str (ExternalContext.text dep))
            ])

        if dep_deps.isEmpty then
          none -- skip if no dependencies
        else
          some <| Json.mkObj [
          ("name", Json.str s!"extracted_split_{ci.name}_{idx}"),
          ("module", Json.str mod.toString),
          ("dependencies", Json.arr dep_deps.toArray)
          ]
        )
      ) |>.flatten

    let output := Json.arr (basic ++ dependency_deps)

    let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
    IO.println s!"Writing to {json_path}"

    if not (← System.FilePath.pathExists json_path) then
      let parent := System.FilePath.parent json_path
      match parent with
      | some path =>
        IO.println path
        IO.FS.createDirAll path
      | none => pure ()



    IO.FS.writeFile json_path (output.compress)



    -- )
    -- now, we have the splits for each theorem in this file
    -- that creates edges split -> theorem,
    -- but we also want edges st for all u -> theorem, if u is used in a split s, we draw u -> s

    -- luckily, we now have ranges of each split, so now we just need ranges of context.



    -- let thmAnnotatedTrees : List (ConstantInfo × CompilationStep × List InfoTree) := targets_new.map (fun (cmd, ci) => (ci, cmd, cmd.trees) )|>.toList


    -- -- let thmAnnotatedTrees : List (String × List InfoTree) := thmAnnotatedTrees_enum.map (fun (s,ts) => (s,ts.map (fun (_,t) =>t) |>.reverse))
    -- let parsedTrees : List (ConstantInfo × CompilationStep  × (IO (List Result))) := thmAnnotatedTrees.map (fun (ci,cmd,ts) => (ci,cmd,ts.filterMapM (BetterParser)))


    -- let mut outputs : List Json := []
    -- -- let mut PTs := []
    -- for (ci,cmd,results) in parsedTrees do
    --   let results ← results
    --   let steps := results.flatMap (fun result => result.steps)

    --   -- IO.println s!"Theorem: \n{cmd.src.toString}\n"
    --   let PT_real? := getProofTree steps

    --   if PT_real?.isNone then
    --     continue

    --   let PT_real := getProofTree steps |>.get!
    --   IO.println s!"Theorem: \n{cmd.src.toString}"

    --   let edges ← get_context cmd ["theorem", "theorem (internal)"]
    --   IO.println s!"Context: \n{edges.map (fun x => x.name.toString)}\n"
    --   IO.println s!"ProofTree: \n{PT_real}\n"
    --   -- let PT_json : Json := toJson PT_real
    --   -- IO.println s!"ProofTree JSON: \n{PT_json}\n\n"

    --   let breakpoints := PT_real.getBreakpoints
    --   IO.println s!"Breakpoints: \n{breakpoints.map (fun ps => ps.tacticString)}\n\n"

    --   let new_thm := insertBreakpointsFromTree cmd.src.toString PT_real
    --   -- IO.println s!"New theorem: \n{new_thm}\n\n"

    --   let contentsBefore : Substring := match cmd.src with
    --   | ⟨s, b, _⟩ => ⟨s, 0, b⟩

    --   let options := ({} : KVMap)
    --     |>.insert `maxHeartbeats (.ofNat 200000) -- TODO determine a heartbeat count
    --     |>.insert `debug.byAsSorry (.ofBool false)
    --     |>.insert `linter.unusedVariables (.ofBool true)
    --     |>.insert `linter.unusedTactic (.ofBool true)
    --     |>.insert `linter.unreachableTactic (.ofBool true)

    --   let elaborated_steps := Lean.Elab.IO.compilationSteps
    --     (Parser.mkInputContext (contentsBefore.toString ++ new_thm) fileName)
    --     cmd.parserStateBefore
    --     (cmd.commandStateBefore.withOptions options)

    --   let head? ← elaborated_steps.uncons
    --   let outputted : Option CompilationStep := match head? with
    --   | none => none
    --   | some (cstep, _) => some cstep

    --   match outputted with
    --   | none => IO.println s!"Failed to elaborate the new theorem."
    --   | some cstep =>
    --     let filtered_msg := cstep.msgs.filter (fun msg => msg.severity == .information)
    --     let msgs ← filtered_msg.mapM (fun msg => msg.toString)

    --     IO.println s!"Split Theorems: \n{"\n".intercalate msgs}\n"
    --     IO.println "===================="


    --     let deps : List Json := msgs.zipIdx.map (fun (msg,idx) =>
    --       Json.mkObj [
    --         ("name", Json.str s!"extracted_split_{idx}"),
    --         ("text", Json.str msg),
    --         ("module", Json.str mod.toString)
    --       ]

    --       )

    --     let this : Json := Json.mkObj [
    --       ("name", Json.str ci.name.toString),
    --       ("module", Json.str mod.toString),
    --       ("dependencies", Json.arr deps.toArray),
    --     ]
    --     -- return this

    --     outputs := this :: outputs

    --     -- NOTE: only splits theorems with no errors (no sorry?), and is a tactic proof.
    --     -- It ignores things like <;> (probably, leads to weird behavior)
    -- let json_path := outputDirectory ++ "/" ++ mod.toString.replace "." "/" ++ ".json"
    -- IO.println s!"Writing to {json_path}"

    -- if not (← System.FilePath.pathExists json_path) then
    --   let parent := System.FilePath.parent json_path
    --   match parent with
    --   | some path =>
    --     IO.println path
    --     IO.FS.createDirAll path
    --   | none => pure ()



    -- IO.FS.writeFile json_path (Json.arr outputs.toArray |>.compress)




    return 0


/-- Setting up command line options and help text for `lake exe training_data`. -/
def training_data : Cmd := `[Cli|
  training_data VIA trainingData; ["0.0.1"]
"Export training data from the given file."

  ARGS:
    module : ModuleName; "Lean module to compile and export training data."
    outputDirectory : String; "Where to save the Json output."
]

/-- `lake exe training_data` -/
def main (args : List String) : IO UInt32 :=
  training_data.validate args





-- #eval main ["Mathlib.Logic.Hydra", "/Users/ahuja/Desktop/ImProver-fresh/KG3"]

-- #eval main ["ImProver.ProofTree.Basic", ""]

-- #eval main ["MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets", "/Users/ahuja/Desktop/ImProver-fresh/KG3"]

-- #eval main ["PFR.Main", ""]
