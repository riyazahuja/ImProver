/-- Whether a syntax element is an `obtain` tactic call without a provided proof. -/
def isObtainWithoutProof : Syntax → Bool
  -- Using the `obtain` tactic without a proof requires proving a type;
  -- a pattern is optional.
  | `(tactic|obtain : $_type) | `(tactic|obtain $_pat : $_type) => true
  | _ => false


/-- Deprecated alias for `Mathlib.Linter.Style.isObtainWithoutProof`. -/
@[deprecated isObtainWithoutProof (since := "2024-12-07")]
def is_obtain_without_proof := @isObtainWithoutProof


/-- The `oldObtain` linter emits a warning upon uses of the "stream-of-conciousness" variants
of the `obtain` tactic, i.e. with the proof postponed. -/
register_option linter.oldObtain : Bool := {
  defValue := false
  descr := "enable the `oldObtain` linter"
}


/-- The `oldObtain` linter: see docstring above -/
def oldObtainLinter : Linter where run := withSetOptionIn fun stx => do
    unless Linter.getLinterValue linter.oldObtain (← getOptions) do
      return
    if (← MonadState.get).messages.hasErrors then
      return
    if let some head := stx.find? isObtainWithoutProof then
      Linter.logLint linter.oldObtain head m!"Please remove stream-of-conciousness `obtain` syntax"


initialize addLinter oldObtainLinter


