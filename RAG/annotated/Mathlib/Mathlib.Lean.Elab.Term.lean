/-- Fully elaborates the term `patt`, allowing typeclass inference failure,
but while setting `errToSorry` to false.
Typeclass failures result in plain metavariables.
Instantiates all assigned metavariables. -/
def elabPattern (patt : Term) (expectedType? : Option Expr) : TermElabM Expr := do
  withTheReader Term.Context ({ · with ignoreTCFailures := true, errToSorry := false }) <|
    withSynthesizeLight do
      let t ← elabTerm patt expectedType?
      synthesizeSyntheticMVars (postpone := .no) (ignoreStuckTC := true)
      instantiateMVars t


