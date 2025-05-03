import Cli
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules


import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Mathlib.Tactic.Change
import Batteries.Lean.HashSet
import Batteries.Data.List.Basic
import Cli


open Lean Core Elab IO Meta Term Command Tactic Cli System

set_option autoImplicit true



structure ExternalContext where
  name : Name
  kind : String
  module : Name
  text : String



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

def getExplicitConstantsAsSet (t : TacticInfo) : MetaM (List Name) := do
  let set ← t.goalsBefore
    |>.filterMap t.mctxAfter.getExprAssignmentCore?
    |>.mapM Expr.explicitConstants

  let out : NameSet := set.foldl .union .empty
  let out2 : List Name := out.toList
  return out2

partial def go (s : Syntax) (acc : Array Name) : Array Name :=
  match s with
  | Syntax.ident _ _ name _ => acc.push name
  | Syntax.node _ _ args => args.foldl (fun acc s' => go s' acc) acc
  | _ => acc

def getConstants (step : CompilationStep) : MetaM (List Name) := do
  let idents : Array Name := go step.stx #[]
  return idents.toList

def getUsedConstantsAsSet (t : TacticInfo) : NameSet :=
  let set := t.goalsBefore
    |>.filterMap t.mctxAfter.getExprAssignmentCore?
    |>.map Expr.getUsedConstantsAsSet
    -- |>.map Expr
    |>.foldl .union .empty

  set


def getKind (const_map : ConstMap) (m : Name) : String :=
  let local_const := const_map.map₂
  let ext_const := const_map.map₁
  let c := local_const.find? m
  match c with
  | none => match ext_const[m]? with
    | some d => match d with
      | .axiomInfo _ => "axiom"
      | .defnInfo _ => "def"
      | .thmInfo _ => "theorem"
      | .opaqueInfo _ => "opaque"
      | .quotInfo _ => "quot"
      | .inductInfo _ => "inductive"
      | .ctorInfo _ => "constructor"
      | .recInfo _ => "recursor"
    | none => "Not Found"
  | some c => match c with
    | .axiomInfo _ => "axiom (internal)"
    | .defnInfo _ => "def (internal)"
    | .thmInfo _ => "theorem (internal)"
    | .opaqueInfo _ => "opaque (internal)"
    | .quotInfo _ => "quot (internal)"
    | .inductInfo _ => "inductive (internal)"
    | .ctorInfo _ => "constructor (internal)"
    | .recInfo _ => "recursor (internal)"

def isAuxLemma : Name → Bool
| .num (.str _ "_auxLemma") _ => true
| _ => false

def get_context (step:CompilationStep) : IO (List ExternalContext) := do
  IO.println "HELLO"
  -- let tactics := step.trees
  --   |>.flatMap InfoTree.retainTacticInfo
  --   |>.flatMap InfoTree.retainOriginal
  --   |>.flatMap InfoTree.retainSubstantive

  let pf_env := step.commandStateBefore.env
  let ctx : Core.Context := {fileName := "", fileMap := default}
  let state : Core.State := {env := pf_env}
  -- let metaExplicitConstants := tactics.mapM (fun t => t.findTacticNodes.mapM (fun ⟨i,_⟩ => (getExplicitConstantsAsSet i)))
  -- let explicit_constants_raw ← MetaM.toIO metaExplicitConstants ctx state
  -- let constants := explicit_constants_raw.1.flatMap .flatten |>.eraseDups
  let constants ← MetaM.toIO (getConstants step) ctx state
  let constants := constants.1.eraseDups

  let modules := constants.map (fun c => (c,pf_env.getModuleFor? c |>.getD (Name.anonymous)))

  let consts_mods_kind := modules.map (fun (c, m) => (c, m, getKind pf_env.constants c))

  let mods := (modules.map fun x => x.2) |>.eraseDups |>.filter fun m => m != Name.anonymous

  let allowed_kinds := ["theorem", "def","theorem (internal)", "def (internal)"]
  let constant_info ← CoreM.withImportModules mods.toArray do
    let mut out := []
    for (c, module, kind) in consts_mods_kind do
      if isAuxLemma c || kind ∉ allowed_kinds || module.isAnonymous then
        continue
      IO.println s!"extracting {c}, {module}"
      let rgs := ((← findDeclarationRanges? c).getD default).range
      -- let module := ((pf_env.getModuleFor? c).getD (Name.anonymous))
      IO.println s!"rg: {rgs.pos} -> {rgs.endPos}"
      let modulePath ← findLean module
      let fileContents ← IO.FS.readFile modulePath.toString
      -- get the source code within range
      let declTextList := if rgs.pos.line == 0 then [] else
        fileContents.splitOn "\n"
          |>.drop (rgs.pos.line - 1)
          |>.take (rgs.endPos.line - rgs.pos.line + 1)
      let declText := "\n".intercalate declTextList

      out := (ExternalContext.mk c kind module declText)::out
    return out
  return constant_info
