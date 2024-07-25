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
import Cli

open Lean Elab IO Meta
open Cli System


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
  let ⟨x, y⟩ := info.fst.fst
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


  let goalsBefore : Array String := i.info.goalsBefore.map (fun x => x.name.toString) |>.toArray
  let goalsAfter : Array String := i.info.goalsAfter.map (fun x => x.name.toString) |>.toArray
  let mctxBefore : Array (String × String):= i.info.mctxBefore.eAssignment.toList.map (fun (k,v) => (k.name.toString,v.dbgToString)) |>.toArray
  let mctxAfter : Array (String × String):= i.info.mctxAfter.eAssignment.toList.map (fun (k,v) => (k.name.toString,v.dbgToString)) |>.toArray


  let childrenJson ← i.children.toList.mapM (fun x=> x.toJson (some i.ctx))




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
      ("thm_endPos", Json.mkObj [("line", thm_tail.line),("column",thm_tail.column)]),
      ("goalsBefore",Json.arr (goalsBefore.map (fun x => Json.str x))),
      ("goalsAfter",Json.arr (goalsAfter.map (fun x => Json.str x))),
      ("mctxBefore", Json.arr (mctxBefore.map (fun (x,y) => Json.mkObj [("key",Json.str x),("value",Json.str y)])) ),
      ("mctxAfter", Json.arr (mctxAfter.map (fun (x,y) => Json.mkObj [("key",Json.str x),("value",Json.str y)])) ),
      ("children", Json.arr childrenJson.toArray),
    ]

  return (declId, pf_json)




end Lean.Elab.TacticInvocation

/-
def getProofTrees (data : List (String × Json)) : IO (List (String × List (Nat × Nat))):= do

  let split_thms : List (String × (List Json)) := data.foldl
    (fun acc (s, j) =>
      match acc.find? (fun (t, _) => t = s) with
      | some (_,_) =>
        acc.map (fun (t, js') => if t = s then (t, js' ++ [j]) else (t, js'))
      | none => acc ++ [(s, [j])]
    )
    []



  let extractJsonStr (json:Json) := match json with
        | Json.str s => some s
        | _ => none

  let deJsonifyStrArr (json: Json):=
    match json with
    | Json.arr elems => some (elems.map (fun x => extractJsonStr x |>.getD ""))
    | _ => none

  let deJsonifyStrStrArr (json: Json):=
    match json with
    | Json.arr elems => (
      let extract (elem:Json) :=
        match elem with
        | Json.obj kvpairs => (
          let kvpairs := kvpairs.find compare
          let key := kvpairs "key" |>.getD (Json.null)
          let value := kvpairs "value" |>.getD (Json.null)
          some (extractJsonStr key |>.getD "",extractJsonStr value |>.getD "")
        )
        | _ => none
      some (elems.map (fun x => extract x |>.getD ("","")))
    )
    | _ => none

  let get_tactic (elem:Json) :=
    match elem with
        | Json.obj kvpairs => (
          let kvpairs := kvpairs.find compare
          let tactic := kvpairs "tactic" |>.getD ""
          some (extractJsonStr tactic |>.getD "")
        )
        | _ => none
  IO.println <| split_thms.map (fun (s,js) => (s,js.map (fun js => get_tactic js |>.getD "")))

  let deJsonify (elem : Json) :=
    match elem with
        | Json.obj kvpairs => (
          let kvpairs := kvpairs.find compare
          let before := kvpairs "goalsBefore" |>.getD (Json.null)
          let after := kvpairs "goalsAfter" |>.getD (Json.null)
          let mctxAfter := kvpairs "mctxAfter" |>.getD (Json.null)
          some (deJsonifyStrArr before |>.getD Array.empty,deJsonifyStrArr after|>.getD Array.empty,deJsonifyStrStrArr mctxAfter|>.getD Array.empty)
        )
        | _ => none


  let mctxInside (old : Nat × Json) (new : Nat × Json) : Bool :=
    let old_data := (old.1,deJsonify (old.2) |>.getD (Array.empty,Array.empty,Array.empty))
    let new_data := (new.1,deJsonify (new.2) |>.getD (Array.empty,Array.empty,Array.empty))

    let old_before := old_data.2.1
    let mctxAfter := old_data.2.2.2
    let search (db : Array (String × String)) (key : String) : Option String :=
      let filtered := db.filter (fun (k,_) => key == k)
      if filtered.toList.isEmpty then none
      else some (filtered.get! 0).2

    let after_ctx := old_before.filterMap (fun b4txt => search mctxAfter b4txt)
    let before := new_data.2.1

    before.any (fun a => after_ctx.any (fun b => b.containsSubstr a))

  let enum_thms := split_thms.map (fun (s,js) => (s,js.enum))
  let get_combos (nodes : List (Nat × Json)) :=
    let prod := nodes.bind (fun a => nodes.map (fun b=> (a,b)))
    prod.filter (fun (a,b) => a == b)

  let get_insides (nodes : List (Nat × Json)) :=
    let combos := get_combos nodes
    combos.filterMap (fun (old,new) => if mctxInside old new then some (old.1,new.1) else none)

  let node1 := ((enum_thms.head!).2.get! 1)
  let node2 := ((enum_thms.head!).2.get! 3)
  IO.println node1.1
  IO.println (mctxInside node1 node2)
  return enum_thms.map (fun (s,js)=>(s,get_insides js))
-/


def trainingData (args : Cli.Parsed) : IO UInt32 := do
    searchPathRef.set compile_time_search_path%

    let module := args.positionalArg! "module" |>.as! ModuleName
    let infos ← getElabDeclInfo (← moduleInfoTrees module)
    let trees ← getInvocationTrees module
    let hash ← generateRandomHash

    --let msgs ← moduleMessages module
    let mut msgs := []
    let raw_msgs ← moduleMessages module

    for msg in raw_msgs do
      let s ← msg.toJson
      msgs := s::msgs


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

    let thmAnnotatedTrees : List (String × List InfoTree) := thmAnnotatedTrees_enum.map (fun (s,ts) => (s,ts.map (fun (_,t) =>t) |>.reverse))



    let parsedTrees : List (String × (IO (List Result))) := thmAnnotatedTrees.map (fun (s,ts) => (s,ts.filterMapM (BetterParser)))


    --let stepsJson (res : Result) := Json.arr <| res.steps.map (fun ps => toJson ps) |>.toArray

    let mut PTs := []
    for (s,results) in parsedTrees do
      let results ← results
      let steps := results.bind (fun result => result.steps)
      /-
      let getgoalBefore (step:ProofStep) := step.goalBefore.hyps.map (fun hyp => hyp.id)
      let getgoalsAfter (step:ProofStep) := step.goalsAfter.bind (fun goal => goal.hyps.map (fun hyp => hyp.id))
      let getgoalsSpawned (step:ProofStep) := step.spawnedGoals.bind (fun goal => goal.hyps.map (fun hyp => hyp.id))
      let parsed := steps.enum.map (fun (idx,step) => s!"[{idx}] Tactic: {step.tacticString}\ngoalsBefore: {getgoalBefore step}\ndependencies: {step.tacticDependsOn}\ngoalsAfter: {getgoalsAfter step}\nspawnedGoals:{getgoalsSpawned step}\n")
      -/

      let PT : List (String × List Nat) := getProofTree steps
      let PTJson := Json.arr <| PT.map (fun (s,xs) => Json.mkObj ([("tactic",s),("children",Json.arr <| xs.map (fun x => Json.num <| JsonNumber.fromNat x) |>.toArray)])) |>.toArray
      PTs := (s,PTJson) :: PTs
    let PTsJson := Json.mkObj PTs
    --let goalsJson (res : Result) := Json.arr <| res.allGoals.toArray.map (fun gi => toJson gi)--Json.arr <| res.steps.map (fun ps => toJson ps) |>.toArray

    --
    --let duh2 := Json.arr <| parsedTrees.map goalsJson |>.toArray
    --IO.println duh
    --IO.println "========="
    --IO.println duh2
    let rev := idJsons.reverse
    --let PTs ← getProofTrees rev




    --IO.println (rev.map (fun (_,j) => deJsonify j |>.get!))
    --IO.println PTs
    /-
    let PTs_json :=
      PTs.map (fun (s,data) => Json.mkObj ([
        ("declID", Json.str s),
        ("mctxChildren", Json.arr (data.map (fun (old,new) =>
          Json.mkObj ([
          ("oldIdx", Json.num (JsonNumber.fromNat old)),
          ("newIdx", Json.num (JsonNumber.fromNat new))
          ])) |>.toArray)
        )
      ]))
    -/
    let out := rev.map fun (_, j) => j

    let tactics := Json.arr out.toArray
    let messages := Json.arr msgs.toArray
    let output := Json.mkObj ([
      ("tactics",tactics),
      ("messages",messages),
      ("proofTrees", PTsJson)
    ])

    IO.println output.compress

    -- for item in out do
    --   IO.println item.compress
    --   IO.println "====LINE===="

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
