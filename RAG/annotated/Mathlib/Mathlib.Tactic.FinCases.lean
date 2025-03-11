/-- If `e` is of the form `x ∈ (A : List α)`, `x ∈ (A : Finset α)`, or `x ∈ (A : Multiset α)`,
return `some α`, otherwise `none`. -/
def getMemType {m : Type → Type} [Monad m] [MonadError m] (e : Expr) : m (Option Expr) := do
  match e.getAppFnArgs with
  | (``Membership.mem, #[_, type, _, _, _]) =>
    match type.getAppFnArgs with
    | (``List, #[α])     => return α
    | (``Multiset, #[α]) => return α
    | (``Finset, #[α])   => return α
    | _ => throwError "Hypothesis must be of type `x ∈ (A : List α)`, `x ∈ (A : Finset α)`, \
                       or `x ∈ (A : Multiset α)`"
  | _ => return none


/--
Recursively runs the `cases` tactic on a hypothesis `h`.
As long as two goals are produced, `cases` is called recursively on the second goal,
and we return a list of the first goals which appeared.

This is useful for hypotheses of the form `h : a ∈ [l₁, l₂, ...]`,
which will be transformed into a sequence of goals with hypotheses `h : a = l₁`, `h : a = l₂`,
and so on.
Cases are named according to the order in which they are generated as tracked by `counter`
and prefixed with `userNamePre`.
-/
partial def unfoldCases (g : MVarId) (h : FVarId)
    (userNamePre : Name := .anonymous) (counter := 0) : MetaM (List MVarId) := do
  let gs ← g.cases h
  try
    let #[g₁, g₂] := gs | throwError "unexpected number of cases"
    g₁.mvarId.setUserName (.str userNamePre s!"{counter}")
    let gs ← unfoldCases g₂.mvarId g₂.fields[2]!.fvarId! userNamePre (counter+1)
    return g₁.mvarId :: gs
  catch _ => return []


/-- Implementation of the `fin_cases` tactic. -/
partial def finCasesAt (g : MVarId) (hyp : FVarId) : MetaM (List MVarId) := g.withContext do
  let type ← hyp.getType >>= instantiateMVars
  match ← getMemType type with
  | some _ => unfoldCases g hyp (userNamePre := ← g.getTag)
  | none =>
    -- Deal with `x : A`, where `[Fintype A]` is available:
    let inst ← synthInstance (← mkAppM ``Fintype #[type])
    let elems ← mkAppOptM ``Fintype.elems #[type, inst]
    let t ← mkAppM ``Membership.mem #[elems, .fvar hyp]
    let v ← mkAppOptM ``Fintype.complete #[type, inst, Expr.fvar hyp]
    let (fvar, g) ← (← g.assert `this t v).intro1P
    finCasesAt g fvar


/--
`fin_cases h` performs case analysis on a hypothesis of the form
`h : A`, where `[Fintype A]` is available, or
`h : a ∈ A`, where `A : Finset X`, `A : Multiset X` or `A : List X`.

As an example, in
```
example (f : ℕ → Prop) (p : Fin 3) (h0 : f 0) (h1 : f 1) (h2 : f 2) : f p.val := by
  fin_cases p; simp
  all_goals assumption
```
after `fin_cases p; simp`, there are three goals, `f 0`, `f 1`, and `f 2`.
-/
syntax (name := finCases) "fin_cases " ("*" <|> term,+) (" with " term,+)? : tactic


@[tactic finCases] elab_rules : tactic
  | `(tactic| fin_cases $[$hyps:ident],*) => withMainContext <| focus do
    for h in hyps do
      allGoals <| liftMetaTactic (finCasesAt · (← getFVarId h))


