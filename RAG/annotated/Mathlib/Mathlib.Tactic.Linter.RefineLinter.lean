/-- The "refine" linter flags usages of the `refine'` tactic.

The tactics `refine` and `refine'` are similar, but they handle meta-variables slightly differently.
This means that they are not completely interchangeable, nor can one completely replace the other.
However, `refine` is more readable and (heuristically) tends to be more efficient on average.
-/
register_option linter.refine : Bool := {
  defValue := false
  descr := "enable the refine linter"
}


/-- `getRefine' t` returns all usages of the `refine'` tactic in the input syntax `t`. -/
partial
def getRefine' : Syntax → Array Syntax
  | stx@(.node _ kind args) =>
    let rargs := (args.map getRefine').flatten
    if kind == ``Lean.Parser.Tactic.refine' then rargs.push stx else rargs
  | _ => default


@[inherit_doc linter.refine]
def refineLinter : Linter where run := withSetOptionIn fun _stx => do
  unless Linter.getLinterValue linter.refine (← getOptions) do
    return
  if (← MonadState.get).messages.hasErrors then
    return
  for stx in (getRefine' _stx) do
    Linter.logLint linter.refine stx
      "The `refine'` tactic is discouraged: \
      please strongly consider using `refine` or `apply` instead."


initialize addLinter refineLinter


