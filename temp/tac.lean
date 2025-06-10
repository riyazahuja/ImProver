import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import Batteries
import Lean
import Cli

open Lean Elab IO Meta
open Cli System

def findCommandInfo (t : InfoTree) : List (CommandInfo × ContextInfo) :=
  let infos := t.findAllInfo none fun i => match i with
    | .ofCommandInfo _ => true
    | _ => false
  infos.filterMap fun p => match p with
  | (.ofCommandInfo i, some ctx, _) => (i, ctx)
  | _ => none

structure ElabDeclInfo where
  range : Range
  cmdInfo : CommandInfo
  currNamespace : Name

def getElabDeclInfo (trees : List InfoTree) : IO (List ElabDeclInfo) := do
    let mut out  := []
    for tree in trees do
      let infos := findCommandInfo tree
      for ⟨cmdInfo, ctxInfo⟩ in infos do
        out := ⟨FileMap.stxRange ctxInfo.fileMap cmdInfo.stx, cmdInfo, ctxInfo.currNamespace⟩ :: out
    return out

def ppCommandInfo (info : CommandInfo) : String :=
  info.stx.prettyPrint.pretty

def getElabDeclOfTacticInvocation (elabDeclInfo : List ElabDeclInfo) (ti: TacticInvocation) :
  Option ElabDeclInfo := do
    let ⟨s, e⟩ := FileMap.stxRange ti.ctx.fileMap ti.info.stx
    elabDeclInfo.find? fun ⟨⟨s', e'⟩, _, _⟩ => s' <= s && e <= e'

def getInvocationTrees (module: ModuleName) : IO (List InfoTree) := do
    let mut trees ← moduleInfoTrees module
    trees := trees.flatMap InfoTree.retainTacticInfo
    trees := trees.flatMap InfoTree.retainOriginal
    trees := trees.flatMap InfoTree.retainSubstantive
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
    -- (magic value) if this command is a declaration like theorem/def T := proof/definition
    -- then the := syntax occurs at `stx[1][3][0]`
    if info.stx[1][3][0].getAtomVal == ":=" then
      let declStart := info.stx.getPos?.getD 0
      let proofStart := info.stx[1][3].getPos?.getD 0
      let proofEnd := info.stx.getTailPos?.getD 0
      let moduleSource ← moduleSource module
      let decl := (Substring.mk moduleSource declStart proofStart).toString
      let proof := (Substring.mk moduleSource proofStart proofEnd).toString
      return decl
    else
      return ""

open PrettyPrinter Delaborator SubExpr in
/-- This delaborates a type to the format of a signature of a constant.
It is copied from the type-printing part of delabConstWithSignature.
-/
def delabTypeSignature (name : Name) : Delab := do
  let e ← getExpr
  descend e 1 <|
    delabConstWithSignature.delabParams {} (mkIdent name) #[]

open Tactic PrettyPrinter in
def stateAsSignature (noRevertFVarIds : Array FVarId) : TacticM Format := do
  -- This is copied from extract_goal in mathlib
  let name : Name := `extracted
  withOptions (pp.funBinderTypes.set · true |> (pp.universes.set · false) |> (pp.coercions.types.set · true)) do
  withoutModifyingEnv <| withoutModifyingState do
    let g ← getMainGoal
    let g ← do
      if (← g.getType >>= instantiateMVars).consumeMData.isConstOf ``False then
        -- In a contradiction proof, it is not very helpful to clear all hypotheses!
        pure g
      else
        g.cleanup
    let (g, _) ← g.renameInaccessibleFVars
    let fvarIds := (← g.getDecl).lctx.getFVarIds
    -- We deviate from extract_goal by not reverting all free variables for conciseness
    -- The set of free variables not reverted will be the ones in the initial state
    let revertFvarIds := fvarIds.filter fun id => !noRevertFVarIds.contains id
    let (_, g) ← g.revert (clearAuxDeclsInsteadOfRevert := true) revertFvarIds
    let ty ← g.getType
    -- Also, we deviate from extract_goal by
    -- zeta reducing the type so the resulting type doesn't have `let` binders
    let ty ← zetaReduce ty
    let ty ← instantiateMVars ty
    let ty ← Term.levelMVarToParam ty
    let (stx, _) ← delabCore ty (delab := delabTypeSignature name)
    PrettyPrinter.ppTerm ⟨stx⟩  -- HACK: not a term

section Test
elab "extract_goal" : tactic => do
  let fvars := (← getLCtx).getFVarIds
  let l := fvars.size
  for i in [:l] do
    logInfo (← stateAsSignature (fvars.take i))
  -- logInfo (← stateAsSignature (fvars.take 1))
  -- logInfo (← stateAsSignature (fvars.take 2))
  -- logInfo (← stateAsSignature (fvars.take 3))
  -- logInfo (← stateAsSignature (fvars.take 4))
/--
info: extracted {α : Type u_1} (a : α) [inst : Add α] : a + a = a + a
---
info: extracted {α : Type u_1} (a : α) [inst : Add α] : a + a = a + a
---
info: extracted (a : α) [inst : Add α] : a + a = a + a
---
info: extracted [inst : Add α] : a + a = a + a
---
info: extracted : a + a = a + a
-/

example {α} (a : α) [Add α] : a + a = a + a := by extract_goal; rfl
end Test

def stateAsSignatureMetaM (g : MVarId) (noRevertFVarIds : Array FVarId) : MetaM Format := do
  g.withContext do
    Term.TermElabM.run' do
      -- We do not revert free variables occuring in the initial proof state for conciseness
      let (fmt, _) ← stateAsSignature noRevertFVarIds { elaborator := .anonymous } |>.run { goals := [g] }
      return fmt.pretty 1000000000

open Term in
def trainingData' (elabDeclInfo: ElabDeclInfo) (module : ModuleName) (i : TacticInvocation) (initFVars? : Name → Option (Array FVarId)) :
    IO (Name × (Array FVarId) × Json) := do
  let declName := i.ctx.parentDecl?.getD Name.anonymous
  let declNameString := match declName with | Name.anonymous => "" | n => n.toString
  let srcUpToTactic := Substring.mk (← moduleSource module) 0 (i.info.stx.getPos?.getD 0)
  let declUpToTactic := Substring.mk (← moduleSource module)
    (elabDeclInfo.cmdInfo.stx.getPos?.getD 0) (i.info.stx.getPos?.getD 0)

  -- let fvars ← i.ctx.runMetaM {} <| Meta.withMCtx i.info.mctxBefore <| do
  --   return (← getLCtx).getFVarIds
  let fvars? ← i.runMetaMGoalsBefore fun gs => gs[0]?.mapM fun g => do
    return (← g.getDecl).lctx.getFVarIds
  let fvars := fvars?.getD #[]

  let statesAsSignatures ← i.runMetaMGoalsBefore fun gs => gs.mapM fun g => do
    let fmt ← stateAsSignatureMetaM g ((initFVars? declName).getD fvars)
    return fmt.pretty 1000000000
  let states := (← i.goalState).map (·.pretty 1000000000)
  let statesAfter := (← i.goalStateAfter).map (·.pretty 1000000000)
  let statesAfterAsSignatures ← i.runMetaMGoalsAfter fun gs => gs.mapM fun g => do
    let fmt ← stateAsSignatureMetaM g ((initFVars? declName).getD fvars)
    return fmt.pretty 1000000000
  let nextTactic ← tacticPP module i
  let decl ← ppDeclWithoutProof module elabDeclInfo.cmdInfo

  let data : Json :=
    Json.mkObj [
      ("declName", Json.str declNameString),
      ("decl", Json.str decl),
      ("srcUpToTactic", Json.str srcUpToTactic.toString),
      ("declUpToTactic", Json.str declUpToTactic.toString),
      ("states", toJson states),
      ("statesAsSignatures", toJson statesAsSignatures),
      ("statesAfter", toJson statesAfter),
      ("statesAfterAsSignatures", toJson statesAfterAsSignatures),
      ("nextTactic", Json.str nextTactic)
    ]
  return (declName, fvars, data)

end Lean.Elab.TacticInvocation


def trainingData (args : Cli.Parsed) : IO UInt32 := do
    searchPathRef.set compile_time_search_path%


    let module := args.positionalArg! "module" |>.as! ModuleName
    let infos ← getElabDeclInfo (← moduleInfoTrees module)
    let trees ← getInvocationTrees module

    let mut results : NameMap (Array FVarId × Array Json) := ∅
    for t in trees do
      for t in t.tactics do

        match getElabDeclOfTacticInvocation infos t with
        | some elabDeclInfo => do
          let (name, fvars, json) ← t.trainingData' elabDeclInfo module
            (fun name => (results.find? name).map Prod.fst)
          match results.find? name with
          | none =>
            -- If this is the first tactic of the theorem,
            -- use its free variables of the state before as the initial free variables
            results := results.insert name (fvars, #[json])
          | some (initFVars, jsons) =>
            -- For subsequent tactics,
            -- we exclude the initial free variables
            -- IO.println s!"{name}: {initFVars.map (·.name)}"
            results := results.insert name (initFVars, jsons.push json)
        | none => pure ()

    for (name, _, jsons) in results do
      let item := json%{
        "declName": $(name), "tactics": $(jsons)
      }
      IO.println item.compress

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
