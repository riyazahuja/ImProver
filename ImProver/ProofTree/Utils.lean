import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import TrainingData.TreeParser
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
-- import Batteries.Lean.Util.Path
import Batteries.Data.String.Basic
import Mathlib.Tactic.Change
import Cli

open Lean Elab IO Meta
open Cli System Std


def DeclIdMap := Std.HashMap String (List Json)


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

def ElabDeclInfo := (_root_.Range × CommandInfo)

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

    ]

  return (declId, pf_json)

end Lean.Elab.TacticInvocation
