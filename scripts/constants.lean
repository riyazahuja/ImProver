/-
Adapted from lean-training-data/scripts/premises.lean
(Changed: output declarations restricted to the input module(s).)

Copyright (c) 2023 Scott Morrison. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/

import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import Batteries.Lean.Util.Path
import ImportGraph.RequiredModules

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

/-!
Generate declaration dependencies up to a target file (defaulting to all of Mathlib).

* Declarations are separated by `---`.
* In each block the first declaration is the theorem or definition we are analyzing,
* Subsequent indented declarations are those used in its proof or definition.
* Declarations prefixed with a `* ` appear in explicit arguments.
  (This approximates "is visible in the pretty printed form".)
* Declarations prefixed with a `s ` are used by the simplifier.

-/

open Lean Meta

def isAuxLemma : Name → Bool
| .num (.str _ "_auxLemma") _ => true
| _ => false

partial def Lean.ConstantInfo.getUsedConstants' (c : ConstantInfo)
    (constantsMap : HashMap Name ConstantInfo)
    (unfolding : Name → Bool := isAuxLemma) : NameSet × NameSet × String := Id.run do
  let mut direct : NameSet := ∅
  let mut unfolded : NameSet := ∅
  for n in c.getUsedConstantsAsSet do
    if unfolding n then
      if let some c := constantsMap.find? n then
        unfolded := unfolded ++ c.getUsedConstantsAsSet
    else
      direct := direct.insert n

  let kind := match c with
    | .axiomInfo _ => "axiom"
    | .defnInfo _ => "def"
    | .thmInfo _ => "theorem"
    | .opaqueInfo _ => "opaque"
    | .quotInfo _ => "quot"
    | .inductInfo _ => "inductive"
    | .ctorInfo _ => "constructor"
    | .recInfo _ => "recursor"


  return (direct, unfolded, kind)

/--
Traverse all constants from imported files,
collecting the constants which are used in either the type or value of the theorem or definition.

A predicate `unfolding` picks out a class of constants which should not be returned,
but instead replaced with the set of constants appearing in their type or value.
The typical use case is to unfold `_auxLemma`s generated dynamically by the simplifier.
-/
def allUsedConstants (moduleNames : Array Name) (unfolding : Name → Bool := isAuxLemma) :
    CoreM (NameMap (NameSet × NameSet × String)) := do
  let constantsMap := (← getEnv).constants.map₁
  --let constantsMap := constantsMap.toList.filter (fun (n,c) => n.toString.containsSubstr "PFR")
  let const2ModIdx := (← getEnv).const2ModIdx
  let moduleIdxs := moduleNames.filterMap (← getEnv).getModuleIdx?
  --let sIdx := const2ModIdx.toList.map (fun x=> toString x)
  --IO.println sIdx
  let mut result : NameMap (NameSet × NameSet × String) := ∅
  for (n, c) in constantsMap do
    if !(n.toString.containsSubstr "PFR") then continue
    -- We omit all internally generated auxiliary statements.
    -- We restrict to declarations in the module
    if let some modIdx := const2ModIdx.find? n then
      if moduleIdxs.contains modIdx && ! (← n.isBlackListed) then
        result := result.insert n (c.getUsedConstants' constantsMap unfolding)
  return result

/--
Traverse an expression, collecting `Name`s,
but do not descend into implicit arguments.
-/
partial def Lean.Expr.explicitConstants : Expr → MetaM NameSet
| .app f x => do
  -- We wrap with `try?` here because on tiny fraction of declarations in Mathlib,
  -- e.g. `Computation.exists_of_mem_parallel`, this fails with an error like
  -- `function expected ?m.88 ?m.93`.
  match (← try? (inferType f)) with
  | some (.forallE _ _ _ .default) => return (← f.explicitConstants) ++ (← x.explicitConstants)
  | _ => f.explicitConstants
| .lam _ t b _ => do b.instantiate1 (← mkFreshExprMVar t) |>.explicitConstants
| .forallE _ t b _ => do b.instantiate1 (← mkFreshExprMVar t) |>.explicitConstants
| .letE n t v b _ => return (← v.explicitConstants)
    ++ (← withLetDecl n t v fun fvar => (b.instantiate1 fvar).explicitConstants)
| .const n _ => return NameSet.empty.insert n
| .mdata _ e => e.explicitConstants
| _ => return NameSet.empty

/--
Collect all the constants used in the values of theorem or definition,
ignoring implicit arguments of functions.
-/
def Lean.ConstantInfo.explicitConstants (c : ConstantInfo) : MetaM NameSet := do
  match c.value? with
  | some e => e.explicitConstants
  | none => return ∅

/--
Traverse all constants from imported files,
collecting the constants which are used in the value of the theorem or definition,
ignoring implicit arguments of functions.
-/
def allExplicitConstants (moduleNames : Array Name) : MetaM (NameMap NameSet) := do
  let r := (← getEnv).constants.map₁
  let const2ModIdx := (← getEnv).const2ModIdx
  let moduleIdxs := moduleNames.filterMap (← getEnv).getModuleIdx?
  let mut result : NameMap NameSet := ∅
  for (n, c) in r do
    -- We omit all internally generated auxiliary statements.
    -- We restrict to declarations in the module
    if let some modIdx := const2ModIdx.find? n then
      if moduleIdxs.contains modIdx && ! (← n.isBlackListed) then
        result := result.insert n (← c.explicitConstants)
  return result

def getKind (const_map : HashMap Name ConstantInfo) (m : Name) : String :=
  let c := const_map.find? m
  match c with
  | none => "Not Found"
  | some c => match c with
    | .axiomInfo _ => "axiom"
    | .defnInfo _ => "def"
    | .thmInfo _ => "theorem"
    | .opaqueInfo _ => "opaque"
    | .quotInfo _ => "quot"
    | .inductInfo _ => "inductive"
    | .ctorInfo _ => "constructor"
    | .recInfo _ => "recursor"


def findCommandInfo (t : InfoTree) : List (CommandInfo × ContextInfo) :=
  let infos := t.findAllInfo none fun i => match i with
    | .ofCommandInfo _ => true
    | _ => false
  infos.filterMap fun p => match p with
  | (.ofCommandInfo i, some ctx, _) => (i, ctx)
  | _ => none

def getRanges (tree : InfoTree) : IO (List (Range × CommandInfo)) := do
    let infos := findCommandInfo tree
    let mut out := []
    for (cmdInfo, ctxInfo) in infos do
      if cmdInfo.elaborator = "Lean.Elab.Command.elabDeclaration".toName then
        out := (FileMap.stxRange ctxInfo.fileMap cmdInfo.stx, cmdInfo) :: out
    return out


def ppCommandInfo (module: Name) (info : CommandInfo) : IO String := do
  return (Substring.mk (← moduleSource module)
   (info.stx.getPos?.getD 0)
   (info.stx.getTailPos?.getD 0)).toString

def getDependencyData (module : Name) (name : Name) : IO (List (Range × String)) := do
  try
    let steps ← compileModule module
    let steps := steps.filter (fun cmd => (cmd.after.constants.map₂.contains name) && !(cmd.before.constants.map₂.contains name))
    let trees := steps.bind (fun c => c.trees)
    let mut outputs := []
    for tree in trees do
      let rngs ← getRanges tree
      for (rng, info) in rngs do
        let i ← ppCommandInfo module info
        outputs := (rng,i) :: outputs
    return outputs
  catch _ =>
    return []


def main (args : List String) : IO UInt32 := do
  let options := Options.empty.insert `maxHeartbeats (0 : Nat)
  let modules := match args with
  | [] => #[`Mathlib]
  | args => args.toArray.map fun s => s.toName
  searchPathRef.set compile_time_search_path%

  CoreM.withImportModules modules (options := options) do
    let env ← getEnv
    let allConstants ← allUsedConstants modules
    let explicitConstants ← MetaM.run' (allExplicitConstants modules)
    let const_map := env.constants.map₁
    let mut output := []
    let module := modules[0]!
    for (n, (d, _, kind)) in allConstants do

      let explicit := explicitConstants.find? n |>.getD ∅

      let map_fn (m : Name) : CoreM (Option Json):= do
        let src_module := env.getModuleFor? m |>.getD (Name.anonymous)
        let dep ← getDependencyData src_module m
        if dep.isEmpty then return none
        let range := dep.map (fun (r,_) =>
          Json.mkObj [
            ("start", Json.mkObj [
                ("line", r.1.line),
                ("column",r.1.column)
              ]
            ),
            ("end", Json.mkObj [
              ("line", r.2.line),
              ("column",r.2.column)
              ]
            )]) |>.toArray[0]!
          let content := dep.map (fun (_,s) => Json.str s) |>.toArray[0]!
        return some (Json.mkObj [
          ("name", Json.str m.toString),
          ("module", Json.str src_module.toString),
          ("explicit", Json.bool (explicit.contains m)),
          ("direct", Json.bool (d.contains m)),
          ("kind", Json.str <| getKind const_map m),
          ("content", content),
          ("range",range)
        ])

      let dependents ← (explicit).toList.filterMapM map_fn
      let parent_data ← getDependencyData module n

      let range := parent_data.map (fun (r,_) =>
          Json.mkObj [
            ("start", Json.mkObj [
                ("line", r.1.line),
                ("column",r.1.column)
              ]
            ),
            ("end", Json.mkObj [
              ("line", r.2.line),
              ("column",r.2.column)
              ]
            )]) |>.toArray[0]!
      let content := parent_data.map (fun (_,s) => Json.str s) |>.toArray[0]!

      let json := Json.mkObj [
        ("name", Json.str n.toString),
        ("dependents", Json.arr dependents.toArray),
        ("kind", Json.str kind), -- for now, useful for doing the matching step
        ("content", content),
        ("range",range),
        ("module", Json.str module.toString),
      ]
      output := json::output
    let final := Json.arr output.toArray
    IO.println final.compress
  return 0
