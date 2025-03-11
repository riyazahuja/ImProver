open Lean Elab Tactic Meta in
/-- Backwards compatibility shim for `generalize`. -/
elab "generalize'" h:ident " : " t:term:51 " = " x:ident : tactic => do
  withMainContext do
      let mut xIdents := #[]
      let mut hIdents := #[]
      let mut args := #[]
      hIdents := hIdents.push h
      let expr ← elabTerm t none
      xIdents := xIdents.push x
      args := args.push { hName? := h.getId, expr, xName? := x.getId : GeneralizeArg }
      let hyps := #[]
      let mvarId ← getMainGoal
      mvarId.withContext do
        let (_, newVars, mvarId) ← mvarId.generalizeHyp args hyps (transparency := default)
        mvarId.withContext do
          for v in newVars, id in xIdents ++ hIdents do
            Term.addLocalVarInfo id (.fvar v)
          replaceMainGoal [mvarId]

