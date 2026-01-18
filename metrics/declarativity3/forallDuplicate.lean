import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System


-- Step 1: EdgeStatus type
inductive EdgeStatus
  | normal
  | effective
  | ineffective
  deriving BEq, Inhabited

-- Reason for ineffectiveness
inductive IneffectiveReason
  | trivial (workSteps : Nat) (hasDeps : Bool)
  | duplicate (precedingGoalType : String)
  | noDependents
  | allDependentsIneffective (dependentIds : List Nat)
  deriving Inhabited

-- Extended status with reason
structure EdgeStatusInfo where
  status : EdgeStatus
  tacticStr : String
  reason : Option IneffectiveReason
  deriving Inhabited

-- Edge identifier: unique key for each proof step
def EdgeId := Nat
  deriving BEq, Hashable, Inhabited, ToString

-- Create a unique identifier for a ProofTree node
def getEdgeId (tree : ProofTree) : EdgeId :=
  tree.node.goalBefore.id.name.hash.toNat

-- Map from EdgeId to EdgeStatusInfo
abbrev StatusMap := Std.HashMap EdgeId EdgeStatusInfo

-- Helper to get tactic string from ProofTree
def getTacticStr (tree : ProofTree) : String :=
  tree.node.tacticString









/-- Collect all free variable ids in an expression. -/
partial def gatherFVarIds (e : Expr) (acc : Std.HashSet FVarId := {}) : Std.HashSet FVarId :=
  match e with
  | .fvar fid        => acc.insert fid
  | .app f a         => gatherFVarIds a (gatherFVarIds f acc)
  | .lam _ ty bd _   => gatherFVarIds bd (gatherFVarIds ty acc)
  | .forallE _ ty bd _ => gatherFVarIds bd (gatherFVarIds ty acc)
  | .letE _ ty v b _ => gatherFVarIds b (gatherFVarIds v (gatherFVarIds ty acc))
  | .mdata _ b       => gatherFVarIds b acc
  | .proj _ _ b      => gatherFVarIds b acc
  | _                => acc

/-- Replace fvars that are **not** in the current local context with fresh mvars. -/
partial def replaceUnknownFVarsWithMVars (e : Expr) : MetaM Expr := do
  let lctx ← getLCtx
  let rec go (e : Expr) : MetaM Expr := do
    match e with
    | .fvar fid =>
      match lctx.find? fid with
      | some _ => pure e
      | none   =>
        let u ← mkFreshLevelMVar
        let α ← mkFreshExprMVar (mkSort u)
        mkFreshExprMVar α
    | .app f a           => return .app (← go f) (← go a)
    | .lam n ty b bi     => return .lam n (← go ty) (← go b) bi
    | .forallE n ty b bi => return .forallE n (← go ty) (← go b) bi
    | .letE n ty v b nd  => return .letE n (← go ty) (← go v) (← go b) nd
    | .mdata md b        => return .mdata md (← go b)
    | .proj s i b        => return .proj s i (← go b)
    | e                  => pure e
  go e

/-- Make universe levels flexible everywhere (helps defEq). -/
partial def loosenLevels (e : Expr) : MetaM Expr := do
  let cacheRef ← IO.mkRef ({} : Std.HashMap Name (Array Level))
  let rec go (e : Expr) : MetaM Expr := do
    match e with
    | .const nm lvls =>
      let cache ← cacheRef.get
      match cache.get? nm with
      | some newLvls => return .const nm newLvls.toList
      | none =>
        let newLvls ← lvls.mapM (fun _ => mkFreshLevelMVar)
        cacheRef.set (cache.insert nm newLvls.toArray)
        return .const nm newLvls
    | .app f a           => return .app (← go f) (← go a)
    | .lam n ty b bi     => return .lam n (← go ty) (← go b) bi
    | .forallE n ty b bi => return .forallE n (← go ty) (← go b) bi
    | .letE n ty v b nd  => return .letE n (← go ty) (← go v) (← go b) nd
    | .mdata md b        => return .mdata md (← go b)
    | .proj s i b        => return .proj s i (← go b)
    | _                  => pure e
  go e

/-- Definitional equality "modulo ∀": peel all Π/∀, standardize parameters, then check `isDefEq`.

    Strategy:
    1. Replace external fvars with mvars (prevents "unknown free variable" errors)
    2. Loosen universe levels (helps unification)
    3. Peel ALL Π-binders (∀ and →) from both sides
    4. Abstract bodies into lambdas over peeled parameters
    5. Apply lambdas with fresh mvars (standardizes parameter names)
    6. Check definitional equality on the applications

    This handles cases like:
    - Child: `∀ C : Set α, M.Circuit C → C.Nonempty`
    - Parent: `C.Nonempty` (where C and M.Circuit C are in context)
    - After peeling and standardizing: both reduce to `?C.Nonempty` → Match! -/
def defEqModuloForallMeta (a b : Expr) : MetaM Bool := do
  -- Step 1: Replace all external fvars with fresh mvars
  -- This handles fvars from the original proof context that don't exist in current lctx
  let a0 ← replaceUnknownFVarsWithMVars a
  let b0 ← replaceUnknownFVarsWithMVars b

  -- Step 2: Make universe levels flexible (helps unification)
  let a1 ← loosenLevels a0
  let b1 ← loosenLevels b0

  -- Step 3: Peel ALL Π-binders (∀ and →) from child
  forallTelescopeReducing a1 fun paramsA bodyA => do
    -- Child must have foralls to be a forall-abstraction duplicate
    if paramsA.isEmpty then
      return false

    -- Step 4: Peel ALL Π-binders from parent (might have none)
    forallTelescopeReducing b1 fun paramsB bodyB => do
      -- Step 5: Abstract child body over its parameters
      -- This creates: λ (p₁ : T₁) ... (pₙ : Tₙ) => bodyA
      let lamA ← mkLambdaFVars paramsA bodyA

      -- Step 6: Apply lambda with fresh mvars
      -- This gives: bodyA[p₁ := ?m₁, ..., pₙ := ?mₙ]
      let mvarsA ← paramsA.mapM (fun p => do
        let ty ← inferType p
        mkFreshExprMVar ty)
      let instA := mkAppN lamA mvarsA

      -- Step 7: Do same for parent (if it had any foralls)
      let instB ← if paramsB.isEmpty then
        -- Parent has no foralls, just use the body directly
        pure bodyB
      else
        -- Parent also had foralls, abstract and apply
        let lamB ← mkLambdaFVars paramsB bodyB
        let mvarsB ← paramsB.mapM (fun p => do
          let ty ← inferType p
          mkFreshExprMVar ty)
        pure (mkAppN lamB mvarsB)

      -- Step 8: Check definitional equality
      -- instA and instB now have standardized parameters (mvars)
      try
        isDefEq instA instB
      catch _ =>
        -- Handle any type errors gracefully
        return false




def defEqModuloForall (env : Environment) (a b : Expr) : IO Bool := do
  -- IO.println s!"Checking defEqModuloForall between:\n A: {a}\n B: {b}"
  let coreCtx : Core.Context := { fileName := "<internal>", fileMap := default, options := {} }
  let coreState : Core.State := { env := env }
  let m := defEqModuloForallMeta a b
  let (output, _)← (m.run').toIO coreCtx coreState
  return output




-- edge depth i for e: A -> B means that B's depth is i
partial def getAllEdgesWithDepth (tree : ProofTree) (depth : Nat := 0) : List (Nat × EdgeStatus × ProofStep × ProofTree) := Id.run do
  let mut curr := []
  for child in tree.children.toList do
    let status := if tree.spawned_children.toList.contains child then EdgeStatus.effective else EdgeStatus.normal
    let recursive := getAllEdgesWithDepth child (depth + 1)
    curr := (depth, status, tree.node, child) :: curr ++ recursive
  curr

partial def initializeStatusMap (env : Environment) (tree : ProofTree) : IO StatusMap := do
  let mut statusMap : StatusMap := {}

  let allEdgesWithDepth := getAllEdgesWithDepth tree |>.toArray

  for (_, status, parent, child) in allEdgesWithDepth do
    let edgeId := getEdgeId child
    let childTacticStr := getTacticStr child
    let tacticStr := s!"([{parent.tacticString}] -> [{childTacticStr}])"

    if status == EdgeStatus.effective then
      let childGoal := child.node.goalBefore
      let parentGoal := parent.goalBefore

      let dupToParent ← defEqModuloForall env childGoal.typeExpr parent.goalBefore.typeExpr
      if dupToParent then
        let info : EdgeStatusInfo := {
          status := .ineffective
          tacticStr := tacticStr
          reason := some (IneffectiveReason.duplicate s!"{parentGoal.type} (mod ∀)")
        }
        statusMap := statusMap.insert edgeId info
        continue

    else
      -- normal edge
      let info : EdgeStatusInfo := {
        status := .normal
        tacticStr := tacticStr
        reason := none
      }
      statusMap := statusMap.insert edgeId info

  return statusMap



def formatReason (reason : IneffectiveReason) (statusMap : StatusMap) : String :=
  match reason with
  | IneffectiveReason.trivial workSteps hasDeps =>
    s!"TRIVIAL (workSteps={workSteps}, hasDeps={hasDeps})"
  | IneffectiveReason.duplicate precedingType =>
    s!"DUPLICATE (precedingType={precedingType})"
  | IneffectiveReason.noDependents =>
    "NO_DEPENDENTS (nothing uses hypotheses introduced by this edge)"
  | IneffectiveReason.allDependentsIneffective depIds =>
    let depTactics := depIds.filterMap (fun id =>
      statusMap.get? id |>.map (fun info => s!"{id}:{info.tacticStr}"))
    s!"ALL_DEPENDENTS_INEFFECTIVE (depends on: {depTactics})"

/-- Log spawned edge status -/
def logSpawnedEdgeStatus (edgeId : EdgeId) (info : EdgeStatusInfo) (statusMap : StatusMap) : IO Unit := do
  let statusStr := match info.status with
    | EdgeStatus.effective => "EFFECTIVE"
    | EdgeStatus.ineffective => "INEFFECTIVE"
    | EdgeStatus.normal => "NORMAL"

  let reasonStr := match info.reason with
    | none => "N/A"
    | some r => formatReason r statusMap

  IO.println s!"[EdgeId={edgeId}] Status={statusStr} | Tactic: {info.tacticStr} | Reason: {reasonStr}"





def computeStatus (cs : CompilationStep) : IO Unit := do
  let steps := (← cs.trees.filterMapM BetterParser).flatMap (·.steps)
  IO.println s!"Total steps in compilation: {steps.length}"
  match getProofTree steps with
  | none => return
  | some tree =>
    IO.println tree
    let statusMap ← initializeStatusMap cs.after tree

    IO.println "\n==============================================="
    for (edgeId, info) in statusMap.toList do
      logSpawnedEdgeStatus edgeId info statusMap
    IO.println "===============================================\n"
    return



def getStatus (mod : Name) (decl : Name) (new_proof : String) : IO Unit := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none => return
  | some (target_cmd, _) => do
    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})
    let new_target ← elaborated_steps.head?

    match new_target with
    | none => return
    | some new_target => do
      computeStatus new_target
      return






def repeat_main_goal4 := "theorem foo2 (n : ℕ) : n=n → n=n := by
  have h₁ : ∀ n : ℕ, n=n → n=n := by
    intro n hn
    have duh : ∀ m : ℕ, m=m := by
      intro m
      rfl
    exact duh n
  intro hn
  exact h₁ n hn"

#eval do
  IO.println "\n=== Testing repeat_main_goal4 ==="
  getStatus `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal4






-- def matroid := "lemma Matroid.Circuit.nonempty {M : Matroid α} {C : Set α} (hC : M.Circuit C) : C.Nonempty := by
--   -- Extract the property that a circuit must be nonempty
--   have h_nonempty : ∀ C : Set α, M.Circuit C → C.Nonempty := by
--     intro C hC
--     -- Assume for contradiction that the circuit is empty
--     by_contra! h_empty
--     -- Rewrite the assumption to show the empty set cannot be a circuit
--     rw [h_empty] at hC
--     -- Derive a contradiction since an empty set cannot be a circuit
--     exact hC.not_empty
--   -- Apply the extracted property to conclude the proof
--   apply h_nonempty
--   exact hC"

-- #eval do
--   IO.println "\n=== Testing matroid ==="
--   getStatus `Seymour.Matroid.Notions.Circuit `Matroid.Circuit.nonempty matroid


-- def singleton := "theorem op_eq_singleton_iff (x y : TSet γ) (z : TSet β) :
--     op hβ hγ x y = singleton hβ z ↔ singleton hγ x = z ∧ singleton hγ y = z := by
--   -- Define the equivalence for the operation op resulting in a singleton set
--   have h1 : ∀ x y z, op hβ hγ x y = singleton hβ z ↔ singleton hγ x = z ∧ singleton hγ y = z := by
--     intro x y z
--     rw [op, up_eq_singleton_iff, and_congr_right_iff]
--     rintro rfl
--     simp only [up_eq_singleton_iff, true_and, singleton_inj]
--   -- Apply the established equivalence
--   exact h1 x y z"

-- #eval do
--   IO.println "\n=== Testing singleton ==="
--   getStatus `ConNF.Model.Hailperin `ConNF.TSet.op_eq_singleton_iff singleton




-- def fermi :="lemma fermionicProj_mem_bosonic (a : 𝓕.FieldOpAlgebra) (ha : a ∈ statSubmodule .bosonic) :
--     fermionicProj a = 0 := by
--   -- Introduce a helper lemma to handle the sum of projections
--   have h₁ : fermionicProj a = 0 := by
--     have h₂ := bosonicProj_add_fermionicProj a
--     rw [bosonicProj_mem_bosonic a ha] at h₂
--     simpa using h₂
--   exact h₁"


-- #eval do
--   IO.println "\n=== Testing singleton ==="
--   getStatus `HepLean.PerturbationTheory.FieldOpAlgebra.Grading `FieldSpecification.FieldOpAlgebra.fermionicProj_mem_bosonic fermi
