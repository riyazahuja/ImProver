import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import TrainingData.TreeParser
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Batteries.Lean.Util.Path
import Batteries.Data.String.Basic
import Mathlib.Tactic.Change
import ImportGraph.RequiredModules
import Cli
open Lean Elab Term Command Frontend Parser

open Lean Elab IO Meta
open Cli System

--CREDITS TO KIM MORRISON's training_data repo (citation in paper) for a lot of the background and helpers in this

def DeclIdMap := HashMap String (List Json)

def addToMap (map : DeclIdMap) (declId : String) (jsonObj : Json) : DeclIdMap :=
  match map.find? declId with
  | some jsonList => map.insert declId (jsonObj :: jsonList)
  | none => map.insert declId [jsonObj]

def groupByDecl (idJsons : List (String × Json)) : IO DeclIdMap := do
  let mut map : DeclIdMap := HashMap.empty
  for ⟨declId, json⟩ in idJsons do
    map := addToMap map declId json
  return map

def mapToJson (map : DeclIdMap) : List Json :=
  let entries := map.toList
  let jsonEntries : List Json := entries.map fun (declId, jsonList) =>
    Json.mkObj [
      ("declId", declId),
      ("tacticExamples", Json.arr jsonList.toArray)
    ]
  jsonEntries

def generateRandomHash (length : Nat := 15): IO String := do
  let chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".toList
  let mut hash := ""
  for _ in List.range length do
    hash := hash ++ (chars.get! (← IO.rand 1 (chars.length-1))).toString
  return hash

def findCommandInfo (t : InfoTree) : List (CommandInfo × ContextInfo) :=
  let infos := t.findAllInfo none fun i => match i with
    | .ofCommandInfo _ => true
    | _ => false
  infos.filterMap fun p => match p with
  | (.ofCommandInfo i, some ctx, _) => (i, ctx)
  | _ => none

def ElabDeclInfo := (Range × CommandInfo)

def getElabDeclInfo (trees : List InfoTree) : IO (List ElabDeclInfo) := do
    let mut out  := []
    for tree in trees do
      let infos := findCommandInfo tree
      for ⟨cmdInfo, ctxInfo⟩ in infos do
        out := (FileMap.stxRange ctxInfo.fileMap cmdInfo.stx, cmdInfo) :: out
    return out

def ppCommandInfo (info : CommandInfo) : String :=
  info.stx.prettyPrint.pretty

def getElabDeclOfTacticInvocation (elabDeclInfo : List ElabDeclInfo) (ti: TacticInvocation) :
  Option ElabDeclInfo := do
    let ⟨s, e⟩ := FileMap.stxRange ti.ctx.fileMap ti.info.stx
    elabDeclInfo.find? fun ⟨⟨s', e'⟩, _⟩ => s' <= s && e <= e'

def makeElabDeclId (info: ElabDeclInfo) (module: Name) (hash: String) : String :=
  let ⟨x,y⟩ := info.fst.fst
  let declId := s!"{module}.{x}_{y}.{hash}"
  declId

def getInvocationTrees (module: ModuleName) : IO (List InfoTree) := do
    let mut trees ← moduleInfoTrees module
    trees := trees.bind InfoTree.retainTacticInfo
    trees := trees.bind InfoTree.retainOriginal
    trees := trees.bind InfoTree.retainSubstantive
    return trees


namespace Lean.Elab.TacticInvocation

def tacticPP (module : ModuleName) (i: TacticInvocation) : IO String := do
   return (Substring.mk (← moduleSource module)
   (i.info.stx.getPos?.getD 0)
   (i.info.stx.getTailPos?.getD 0)).toString

def ppCommandInfo (module: ModuleName) (info : CommandInfo) : IO String :=
   return (Substring.mk (← moduleSource module)
   (info.stx.getPos?.getD 0)
   (info.stx.getTailPos?.getD 0)).toString

def ppDeclWithoutProof (module: ModuleName) (info: CommandInfo) : IO String := do
    let ppDecl ← ppCommandInfo module info
    let decl := (ppDecl.splitOn ":=").headD ""
    return decl

def trainingData' (elabDeclInfo: ElabDeclInfo) (module : ModuleName) (hash : String) (i : TacticInvocation) : IO (String × Json) := do
  let declId := makeElabDeclId elabDeclInfo module hash
  let sourceUpToTactic := Substring.mk (← moduleSource module) 0 (i.info.stx.getPos?.getD 0)
  let declUpToTactic := Substring.mk (← moduleSource module)
    (elabDeclInfo.snd.stx.getPos?.getD 0) (i.info.stx.getPos?.getD 0)
  let prev_state := ((← i.goalState).map (fun x => x.pretty)).toArray
  let next_state := ((← i.goalStateAfter).map (fun x => x.pretty)).toArray
  let tactic ← tacticPP module i
  let decl ← ppDeclWithoutProof module elabDeclInfo.snd
  let (start, tail) := i.range

  let (thm_start, thm_tail) := elabDeclInfo.fst


  --let goalsBefore : Array String := i.info.goalsBefore.map (fun x => x.name.toString) |>.toArray
  --let goalsAfter : Array String := i.info.goalsAfter.map (fun x => x.name.toString) |>.toArray
  --let mctxBefore : Array (String × String):= i.info.mctxBefore.eAssignment.toList.map (fun (k,v) => (k.name.toString,v.dbgToString)) |>.toArray
  --let mctxAfter : Array (String × String):= i.info.mctxAfter.eAssignment.toList.map (fun (k,v) => (k.name.toString,v.dbgToString)) |>.toArray


  --let childrenJson ← i.children.toList.mapM (fun x=> x.toJson (some i.ctx))




  let pf_json : Json :=
    Json.mkObj [
      ("declId", Json.str declId),
      ("decl", Json.str decl),
      ("srcUpToTactic", Json.str sourceUpToTactic.toString),
      ("declUpToTactic", Json.str declUpToTactic.toString),
      ("prevState", Json.arr (prev_state.map (fun x => Json.str x))),
      ("nextState", Json.arr (next_state.map (fun x => Json.str x))),
      ("tactic", Json.str tactic),
      ("startPos", Json.mkObj [("line", start.line),("column",start.column)]),
      ("endPos", Json.mkObj [("line", tail.line),("column",tail.column)]),
      ("thm_startPos", Json.mkObj [("line", thm_start.line),("column",thm_start.column)]),
      ("thm_endPos", Json.mkObj [("line", thm_tail.line),("column",thm_tail.column)])--,
      --("goalsBefore",Json.arr (goalsBefore.map (fun x => Json.str x))),
      --("goalsAfter",Json.arr (goalsAfter.map (fun x => Json.str x))),
      --("mctxBefore", Json.arr (mctxBefore.map (fun (x,y) => Json.mkObj [("key",Json.str x),("value",Json.str y)])) ),
      --("mctxAfter", Json.arr (mctxAfter.map (fun (x,y) => Json.mkObj [("key",Json.str x),("value",Json.str y)])) ),
      --("children", Json.arr childrenJson.toArray),
    ]

  return (declId, pf_json)

end Lean.Elab.TacticInvocation


-- def references (info : ConstantInfo) (env : Environment): HashSet (Name × Option Name) :=
--   let temp : NameSet := info.getUsedConstantsAsSet
--   let temp2 := temp.toList

--   let get_mod := temp2.map (fun x=> (x,env.getModuleFor? x))
--   let refs := HashSet.empty.insertMany get_mod
--   refs


def trainingData (args : Cli.Parsed) : IO UInt32 := do
  searchPathRef.set compile_time_search_path%

  -- let t0 ← IO.monoMsNow
  let module : ModuleName := args.positionalArg! "module" |>.as! ModuleName
  let steps ← compileModule module
  -- let t1 ← IO.monoMsNow
  -- IO.println s!"[Timing] compileModule took {t1 - t0}ms"

  -- let t2 ← IO.monoMsNow
  let infos ← getElabDeclInfo (steps.bind (fun c => c.trees))
  let trees ← getInvocationTrees module
  let hash ← generateRandomHash
  -- let t3 ← IO.monoMsNow
  -- IO.println s!"[Timing] Gathering info and invocation trees took {t3 - t2}ms"

  -- let t4 ← IO.monoMsNow
  let mut msgs := []
  let raw_msgs ← moduleMessages module
  for msg in raw_msgs do
    let s ← msg.toJson
    msgs := s::msgs
  -- let t5 ← IO.monoMsNow
  -- IO.println s!"[Timing] Collecting messages took {t5 - t4}ms"

  -- let t6 ← IO.monoMsNow
  let mut idJsons : List (String × Json) := []
  let mut thmAnnotatedTrees_enum : List (String × List (Nat × InfoTree)) := []

  for (idx,t) in trees.enum do
    for tac in t.tactics_new do
    match getElabDeclOfTacticInvocation infos tac with
    | some elabDeclInfo => do
      let json ← tac.trainingData' elabDeclInfo module hash
      if not <| thmAnnotatedTrees_enum.any (fun (s,_) => s==json.1) then
      thmAnnotatedTrees_enum := (json.1,[(idx,t)]) :: thmAnnotatedTrees_enum
      else
      thmAnnotatedTrees_enum := thmAnnotatedTrees_enum.map (fun (s,ts) => if (s==json.1 && (not (ts.any (fun (i,_) => i==idx)))) then (s,(idx,t)::ts) else (s,ts))
      idJsons := json :: idJsons
    | none => pure ()
  -- let t7 ← IO.monoMsNow
  -- IO.println s!"[Timing] Constructing thmAnnotatedTrees took {t7 - t6}ms"

  -- let t8 ← IO.monoMsNow
  let thmAnnotatedTrees : List (String × List InfoTree) :=
    thmAnnotatedTrees_enum.map (fun (s,ts) => (s,ts.map (fun (_,t) => t) |>.reverse))
  let parsedTrees : List (String × (IO (List Result))) :=
    thmAnnotatedTrees.map (fun (s,ts) => (s, ts.filterMapM (BetterParser)))
  -- let t9 ← IO.monoMsNow
  -- IO.println s!"[Timing] Preparing trees and parsers took {t9 - t8}ms"

  -- let t10 ← IO.monoMsNow
  let mut PTs := []
  for (s,results) in parsedTrees do
    let results ← results
    let steps := results.bind (fun result => result.steps)
    let PT : List (String × List Nat × List Nat) := getProofTree steps
    let PTJson := Json.arr <| PT.map (fun (tac,xs) => Json.mkObj (
      [("tactic",tac),
      ("children",Json.arr <| xs.1.map (fun x => Json.num <| JsonNumber.fromNat x) |>.toArray),
      ("spawned_children",Json.arr <| xs.2.map (fun x => Json.num <| JsonNumber.fromNat x) |>.toArray)]
    )) |>.toArray
    PTs := (s,PTJson) :: PTs
  let PTsJson := Json.mkObj PTs
  -- let t11 ← IO.monoMsNow
  -- IO.println s!"[Timing] Proof tree parsing and construction took {t11 - t10}ms"

  -- let t12 ← IO.monoMsNow
  let out := idJsons.reverse.map fun (_, j) => j
  let tactics := Json.arr out.toArray
  let messages := Json.arr msgs.toArray
  let output := Json.mkObj ([
    ("tactics",tactics),
    ("messages",messages),
    ("proofTrees", PTsJson)
  ])
  IO.println output.compress
  -- let t13 ← IO.monoMsNow
  -- IO.println s!"[Timing] Final JSON construction and printing took {t13 - t12}ms"

  return 0


/-- Setting up command line options and help text for `lake exe training_data`. -/
def training_data : Cmd := `[Cli|
  training_data VIA trainingData; ["0.0.1"]
"Export training data from the given file."

  ARGS:
    module : ModuleName; "Lean module to compile and export training data."
]

/-- `lake exe training_data` -/
def main (args : List String) : IO UInt32 :=
  training_data.validate args



--#eval main ["Tests.MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets"]
