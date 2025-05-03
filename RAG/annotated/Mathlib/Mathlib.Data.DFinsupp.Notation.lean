/-- `DFinsupp` elaborator for `single₀`. -/
@[term_elab Finsupp.stxSingle₀]
def elabSingle₀ : Elab.Term.TermElab
  | `(term| single₀ $i $x) => fun ty? => do
    Elab.Term.tryPostponeIfNoneOrMVar ty?
    let .some ty := ty? | Elab.throwUnsupportedSyntax
    let_expr DFinsupp _ _ _ := ← Meta.withReducible (Meta.whnf ty) | Elab.throwUnsupportedSyntax
    Elab.Term.elabTerm (← `(DFinsupp.single $i $x)) ty?
  | _ => fun _ => Elab.throwUnsupportedSyntax


/-- `DFinsupp` elaborator for `update₀`. -/
@[term_elab Finsupp.stxUpdate₀]
def elabUpdate₀ : Elab.Term.TermElab
  | `(term| update₀ $f $i $x) => fun ty? => do
    Elab.Term.tryPostponeIfNoneOrMVar ty?
    let .some ty := ty? | Elab.throwUnsupportedSyntax
    let_expr DFinsupp _ _ _ := ← Meta.withReducible (Meta.whnf ty) | Elab.throwUnsupportedSyntax
    Elab.Term.elabTerm (← `(DFinsupp.update $f $i $x)) ty?
  | _ => fun _ => Elab.throwUnsupportedSyntax


/-- Unexpander for the `fun₀ | i => x` notation. -/
@[app_unexpander DFinsupp.single]
def singleUnexpander : Lean.PrettyPrinter.Unexpander
  | `($_ $pat $val) => `(fun₀ | $pat => $val)
  | _ => throw ()


/-- Unexpander for the `fun₀ | i => x` notation. -/
@[app_unexpander DFinsupp.update]
def updateUnexpander : Lean.PrettyPrinter.Unexpander
  | `($_ $f $pat $val) => match f with
    | `(fun₀ $xs:matchAlt*) => `(fun₀ $xs:matchAlt* | $pat => $val)
    | _ => throw ()
  | _ => throw ()


/-- Display `DFinsupp` using `fun₀` notation. -/
unsafe instance {α : Type*} {β : α → Type*} [Repr α] [∀ i, Repr (β i)] [∀ i, Zero (β i)] :
    Repr (Π₀ a, β a) where
  reprPrec f p :=
    let vals :=
      ((f.support'.unquot.val.map fun i => (repr i, repr (f i))).filter
        (fun p => toString p.2 != "0")).unquot
    let vals_dedup := vals.foldl (fun xs x => x :: xs.eraseP (toString ·.1 == toString x.1)) []
    if vals.length = 0 then
      "0"
    else
      let ret : Std.Format := f!"fun₀" ++ .nest 2 (
        .group (.join <| vals_dedup.map fun a =>
          .line ++ .group (f!"| {a.1} =>" ++ .line ++ a.2)))
      if p ≥ leadPrec then Format.paren ret else ret


