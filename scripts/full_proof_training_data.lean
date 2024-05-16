import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Std.Lean.Util.Path
import Std.Data.String.Basic
import Mathlib.Tactic.Change
import TrainingData.Utils.Range
import Cli

open Lean Elab IO Meta
open Cli System


def DeclIdMap := HashMap String Json

def addToMap (map : DeclIdMap) (declId : String) (jsonObj : Json) : DeclIdMap :=
  match map.find? declId with
  | some _ => map
  | none => map.insert declId jsonObj

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

def ppCommandInfo (module: ModuleName) (info : CommandInfo) : IO String :=
   return (Substring.mk (← moduleSource module)
   (info.stx.getPos?.getD 0)
   (info.stx.getTailPos?.getD 0)).toString

def makeElabDeclId (info: ElabDeclInfo) (module: Name) (hash: String) : String :=
  let ⟨x, y⟩ := info.fst.fst
  let declId := s!"{module}.{x}_{y}.{hash}"
  declId

def ppDeclAndProof (module: ModuleName) (info: CommandInfo) : IO (String × String) := do
    let ppDecl ← ppCommandInfo module info
    let decl := (ppDecl.splitOn ":=").headD ""
    let proof := ((ppDecl.dropPrefix? decl).getD "").toString
    return (decl, proof)

def validateDecl (decl : String) (keep : Bool) : IO Bool :=
  return keep

def validateProof (proof : String) (keep : Bool) : IO Bool :=
  return (keep
    && proof.trim != ""
    && !proof.containsSubstr "sorry")

def trainingData' (elabDeclInfo: ElabDeclInfo) (module : ModuleName) (hash : String) : IO (Bool × (String × Json)) := do
  let declId := makeElabDeclId elabDeclInfo module hash
  let cmdInfo := elabDeclInfo.snd
  let sourceUpToDecl := Substring.mk (← moduleSource module) 0 (cmdInfo.stx.getPos?.getD 0)
  let ⟨decl, proof⟩ ← ppDeclAndProof module elabDeclInfo.snd

  let keep ← validateDecl decl true
  let keep ← validateProof proof keep

  let json : Json :=
    Json.mkObj [
      ("declId", Json.str declId),
      ("decl", Json.str decl),
      ("proof", Json.str proof),
      ("srcUpToDecl", Json.str sourceUpToDecl.toString)
    ]
  return ⟨keep, (declId, json)⟩

def trainingData (args : Cli.Parsed) : IO UInt32 := do
    searchPathRef.set compile_time_search_path%

    let module := args.positionalArg! "module" |>.as! ModuleName
    let infos ← getElabDeclInfo (← moduleInfoTrees module)
    let hash ← generateRandomHash

    let mut declMap : DeclIdMap  := HashMap.empty
    let mut jsons : List Json := []
    for elabDeclInfo in infos do
      let ⟨keep, ⟨id, json⟩⟩  ← (trainingData' elabDeclInfo module hash)
      if keep then
        match declMap.find? id with
        | some id => pure ()
        | none => do
          jsons := json :: jsons
          declMap := addToMap declMap id json

    let out := jsons

    for item in out do
      IO.println item.compress
    return 0


/-- Setting up command line options and help text for `lake exe fullproof_training_data`. -/
def fullproof_training_data : Cmd := `[Cli|
  fullproof_training_data VIA trainingData; ["0.0.1"]
"Export training data from the given file."

  ARGS:
    module : ModuleName; "Lean module to compile and export training data."
]

/-- `lake exe training_data` -/
def main (args : List String) : IO UInt32 :=
  fullproof_training_data.validate args
