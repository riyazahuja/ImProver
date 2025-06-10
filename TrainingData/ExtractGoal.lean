import Batteries
open Lean Elab IO Meta


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

elab "better_extract_goal" : tactic => do
  let fvars := (← getLCtx).getFVarIds
  let limit := 2
  let l := fvars.size
  if l > limit then
    for i in [:limit/2] do
      logInfo (← stateAsSignature (fvars.take i))
    let rest := List.range' (l-limit/2) (limit/2)
    for i in rest do
      logInfo (← stateAsSignature (fvars.take i))
  else
    for i in [:l] do
      logInfo (← stateAsSignature (fvars.take i))

example {α} (a : α) [Add α] : a + a = a + a := by better_extract_goal; rfl

example (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  constructor
  . better_extract_goal;exact h.2
  . better_extract_goal;exact h.1
