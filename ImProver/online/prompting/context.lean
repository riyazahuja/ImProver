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


open Lean Core Elab IO Meta Term Command Tactic Cli System String

set_option autoImplicit true



structure ExternalContext where
  name : Name
  kind : String
  module : Name
  text : String
  parent : CompilationStep
  pos : Option Pos := none
  endPos : Option Pos := none


instance : BEq ExternalContext where
  beq a b := a.name == b.name
  && a.kind.trim == b.kind.trim
  && a.module == b.module
  && a.text.trim == b.text.trim
  && a.parent.src.trim == b.parent.src.trim
  && a.parent.stx == b.parent.stx
  && a.pos == b.pos
  && a.endPos == b.endPos



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


partial def go (s : Syntax) (acc : Array (Name × Option Pos × Option Pos)) : Array (Name × Option Pos × Option Pos) :=
  match s with
  | Syntax.ident _ _ name _ => acc.push (name, s.getPos?, s.getTailPos?)
  | Syntax.node _ _ args => args.foldl (fun acc s' => go s' acc) acc
  | _ => acc

def getConstants (step : CompilationStep) : MetaM (List (Name × Option Pos × Option Pos)) := do
  let idents := go step.stx #[]
  return idents.toList

def getUsedConstantsAsSet (t : TacticInfo) : NameSet :=
  let set := t.goalsBefore
    |>.filterMap t.mctxAfter.getExprAssignmentCore?
    |>.map Expr.getUsedConstantsAsSet
    -- |>.map Expr
    |>.foldl .union .empty

  set


def getKind' (c : ConstantInfo) : String :=
  match c with
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

def getKind (const_map : ConstMap) (m : Name) : String :=
  let local_const := const_map.map₂
  let ext_const := const_map.map₁
  let c := local_const.find? m
  match c with
  | none => match ext_const[m]? with
    | some d => getKind' d
      -- | .axiomInfo _ => "axiom"
      -- | .defnInfo _ => "def"
      -- | .thmInfo _ => "theorem"
      -- | .opaqueInfo _ => "opaque"
      -- | .quotInfo _ => "quot"
      -- | .inductInfo _ => "inductive"
      -- | .ctorInfo _ => "constructor"
      -- | .recInfo _ => "recursor"
    | none => "Not Found"
  | some c => getKind' c --match c with
    -- | .axiomInfo _ => "axiom (internal)"
    -- | .defnInfo _ => "def (internal)"
    -- | .thmInfo _ => "theorem (internal)"
    -- | .opaqueInfo _ => "opaque (internal)"
    -- | .quotInfo _ => "quot (internal)"
    -- | .inductInfo _ => "inductive (internal)"
    -- | .ctorInfo _ => "constructor (internal)"
    -- | .recInfo _ => "recursor (internal)"

def isAuxLemma : Name → Bool
| .num (.str _ "_auxLemma") _ => true
| _ => false


/-- Return the name of the module in which a declaration was defined. -/
def Environment.getModuleForWithSelf? (env : Environment) (declName : Name) (curr_mod : Option Name) : Option Name :=
  match env.getModuleIdxFor? declName with
  | none =>
    if env.constants.map₂.contains declName then do
      curr_mod.getD env.header.mainModule
    else do
      none
  | some idx => do
    env.header.moduleNames[idx.toNat]!


def get_context (step:CompilationStep)
  (allowed_kinds : List String := ["theorem", "def","theorem (internal)", "def (internal)"] )
  (module : Option Name := none)
  : IO (List ExternalContext) := do

  let pf_env := step.commandStateBefore.env
  let ctx : Core.Context := {fileName := "", fileMap := default}
  let state : Core.State := {env := pf_env}

  let constants ← MetaM.toIO (getConstants step) ctx state
  let constants := constants.1.eraseDups
  -- IO.println s!"constants: {"\n".intercalate (constants.map (fun (c, pos, endPos) => s!"{c}"))}"
  let scopes := step.commandStateBefore.scopes.reverse.filterMap (fun s =>
    let content := s.header.trim
    if content == "" then none else some content
    ) |>.toArray



  let consts_mods_kind : List (Name × Option Pos × Option Pos × Name × String) ← constants.filterMapM (fun (c, pos, endPos) => do
    let prefixes : List Name ← do
      if scopes.isEmpty then pure []
      else do
        let mut acc := Name.str Name.anonymous scopes[0]!
        let mut out := [acc]
        for i in [1:scopes.size] do
          acc := Name.str acc scopes[i]!
          out := out ++ [acc]
        pure out

    let possibilities := c :: prefixes.map (fun p => Name.append p c)
    let included? := pf_env.constants.find? c |>.map (fun x => x.all)
    -- IO.println s!"included? {c} : {included?}"
    -- IO.println s!"possibilities: {"\n".intercalate (possibilities.map (fun p => p.toString))}"

    -- IO.println "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    let included? := possibilities.filterMap (fun p => pf_env.constants.find? p)
    match included? with
    | [] => return (c, pos, endPos, Name.anonymous, "Not Found")
    | fst :: _ =>
      let curr_mod := Environment.getModuleForWithSelf? pf_env fst.name module
      return (fst.name, pos, endPos, curr_mod |>.getD Name.anonymous, getKind pf_env.constants fst.name)



    )
  -- IO.println "########################################################"
  -- IO.println s!"constants: \n{"\n".intercalate (pf_env.constants.map₂.toList.map (fun (c, x) => c.toString))}"
  -- IO.println "########################################################"
  -- IO.println s!"consts_mods_kind: {"\n".intercalate (consts_mods_kind.map (fun (c, pos, endPos, m, kind) => s!"{c} ({kind}) : {m}"))}"
  let temp_custom_beq : BEq (Name × Option Pos × Option Pos × Name × String) :=
    ⟨fun (c1, _, _, m1, _) (c2, _, _, m2, _) => c1 = c2 && m1 = m2⟩


  let consts_mods_kind := @List.eraseDups _ temp_custom_beq <| consts_mods_kind.filter fun (_, _, _, m, _) => m != Name.anonymous
  let mods := (consts_mods_kind.map fun (_, _, _, m, _) => m) |>.eraseDups

  -- let allowed_kinds := ["theorem", "def","theorem (internal)", "def (internal)"]
  let constant_info ← CoreM.withImportModules mods.toArray do
    let mut out := []
    for (c, pos, endPos, module, kind) in consts_mods_kind do
      if isAuxLemma c || kind ∉ allowed_kinds || module.isAnonymous then
        continue
      -- IO.println s!"extracting {c}, {module}"
      let rgs := ((← findDeclarationRanges? c).getD default).range
      -- let module := ((pf_env.getModuleFor? c).getD (Name.anonymous))
      -- IO.println s!"rg: {rgs.pos} -> {rgs.endPos}"
      let modulePath ← findLean module
      let fileContents ← IO.FS.readFile modulePath.toString
      -- get the source code within range
      let declTextList := if rgs.pos.line == 0 then [] else
        fileContents.splitOn "\n"
          |>.drop (rgs.pos.line - 1)
          |>.take (rgs.endPos.line - rgs.pos.line + 1)
      let declText := "\n".intercalate declTextList

      out := (ExternalContext.mk c kind module declText step pos endPos)::out
    return out
  let contexts := constant_info.eraseDups
  return contexts
