/-- Type check the given expression, and trace its type. -/
elab tk:"type_check " e:term : tactic => do
  Tactic.withMainContext do
    let e ← Term.elabTermAndSynthesize e none
    check e
    let type ← inferType e
    Lean.logInfoAt tk m!"{← Lean.instantiateMVars type}"

