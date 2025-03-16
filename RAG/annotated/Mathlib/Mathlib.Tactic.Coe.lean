/-- Elaborator for the `(↑)`, `(⇑)`, and `(↥)` notations. -/
def elabPartiallyAppliedCoe (sym : String) (expectedType : Expr)
    (mkCoe : (expectedType x : Expr) → TermElabM Expr) : TermElabM Expr := do
  let expectedType ← instantiateMVars expectedType
  let Expr.forallE _ a b .. := expectedType | do
    tryPostpone
    throwError "({sym}) must have a function type, not{indentExpr expectedType}"
  if b.hasLooseBVars then
    tryPostpone
    throwError "({sym}) must have a non-dependent function type, not{indentExpr expectedType}"
  if a.hasExprMVar then tryPostpone
  let f ← withLocalDeclD `x a fun x ↦ do
    mkLambdaFVars #[x] (← mkCoe b x)
  return f.etaExpanded?.getD f


/-- Partially applied coercion.  Equivalent to the η-reduction of `(↑ ·)` -/
elab "(" "↑" ")" : term <= expectedType =>
  elabPartiallyAppliedCoe "↑" expectedType fun b x => do
    if b.hasExprMVar then tryPostpone
    if let .some e ← coerce? x b then
      return e
    else
      throwError "cannot coerce{indentExpr x}\nto type{indentExpr b}"


/-- Partially applied function coercion.  Equivalent to the η-reduction of `(⇑ ·)` -/
elab "(" "⇑" ")" : term <= expectedType =>
  elabPartiallyAppliedCoe "⇑" expectedType fun b x => do
    if let some ty ← coerceToFunction? x then
      ensureHasType b ty
    else
      throwError "cannot coerce to function{indentExpr x}"


/-- Partially applied type coercion.  Equivalent to the η-reduction of `(↥ ·)` -/
elab "(" "↥" ")" : term <= expectedType =>
  elabPartiallyAppliedCoe "↥" expectedType fun b x => do
    if let some ty ← coerceToSort? x then
      ensureHasType b ty
    else
      throwError "cannot coerce to sort{indentExpr x}"


