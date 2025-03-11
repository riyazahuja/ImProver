/-- `change? term` unifies `term` with the current goal, then suggests explicit `change` syntax
that uses the resulting unified term.

If `term` is not present, `change?` suggests the current goal itself. This is useful after tactics
which transform the goal while maintaining definitional equality, such as `dsimp`; those preceding
tactic calls can then be deleted.
```lean
example : (fun x : Nat => x) 0 = 1 := by
  change? 0 = _  -- `Try this: change 0 = 1`
```
-/
syntax (name := change?) "change?" (ppSpace colGt term)? : tactic


open Lean Meta Elab.Tactic Meta.Tactic.TryThis in
elab_rules : tactic
| `(tactic|change?%$tk $[$sop:term]?) => withMainContext do
  let stx ← getRef
  let expr ← match sop with
    | none => getMainTarget
    | some sop => do
      let tgt ← getMainTarget
      let ex ← withRef sop <| elabTermEnsuringType sop (← inferType tgt)
      if !(← isDefEq ex tgt) then throwErrorAt sop "\
        The term{indentD ex}\n\
        is not defeq to the goal:{indentD tgt}"
      instantiateMVars ex
  let dstx ← delabToRefinableSyntax expr
  addSuggestion tk (← `(tactic| change $dstx)) (origSpan? := stx)

