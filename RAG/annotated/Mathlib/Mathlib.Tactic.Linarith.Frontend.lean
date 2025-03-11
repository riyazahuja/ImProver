/-- A configuration object for `linarith`. -/
structure LinarithConfig : Type where
  /-- Discharger to prove that a candidate linear combination of hypothesis is zero. -/
  -- TODO There should be a def for this, rather than calling `evalTactic`?
  discharger : TacticM Unit := do evalTactic (← `(tactic| ring1))
  -- We can't actually store a `Type` here,
  -- as we want `LinarithConfig : Type` rather than ` : Type 1`,
  -- so that we can define `elabLinarithConfig : Lean.Syntax → Lean.Elab.TermElabM LinarithConfig`.
  -- For now, we simply don't support restricting the type.
  -- (restrict_type : Option Type := none)
  /-- Prove goals which are not linear comparisons by first calling `exfalso`. -/
  exfalso : Bool := true
  /-- Transparency mode for identifying atomic expressions in comparisons. -/
  transparency : TransparencyMode := .reducible
  /-- Split conjunctions in hypotheses. -/
  splitHypotheses : Bool := true
  /-- Split `≠` in hypotheses, by branching in cases `<` and `>`. -/
  splitNe : Bool := false
  /-- Override the list of preprocessors. -/
  preprocessors : List GlobalBranchingPreprocessor := defaultPreprocessors
  /-- Specify an oracle for identifying candidate contradictions.
  `.simplexAlgorithmSparse`, `.simplexAlgorithmSparse`, and `.fourierMotzkin` are available. -/
  oracle : CertificateOracle := .simplexAlgorithmSparse


/--
`cfg.updateReducibility reduce_default` will change the transparency setting of `cfg` to
`default` if `reduce_default` is true. In this case, it also sets the discharger to `ring!`,
since this is typically needed when using stronger unification.
-/
def LinarithConfig.updateReducibility (cfg : LinarithConfig) (reduce_default : Bool) :
    LinarithConfig :=
  if reduce_default then
    { cfg with transparency := .default, discharger := do evalTactic (← `(tactic| ring1!)) }
  else cfg


/--
If `e` is a comparison `a R b` or the negation of a comparison `¬ a R b`, found in the target,
`getContrLemma e` returns the name of a lemma that will change the goal to an
implication, along with the type of `a` and `b`.

For example, if `e` is `(a : ℕ) < b`, returns ``(`lt_of_not_ge, ℕ)``.
-/
def getContrLemma (e : Expr) : MetaM (Name × Expr) := do
  match ← e.ineqOrNotIneq? with
  | (true, Ineq.lt, t, _) => pure (``lt_of_not_ge, t)
  | (true, Ineq.le, t, _) => pure (``le_of_not_gt, t)
  | (true, Ineq.eq, t, _) => pure (``eq_of_not_lt_of_not_gt, t)
  | (false, _, t, _) => pure (``Not.intro, t)


/--
`applyContrLemma` inspects the target to see if it can be moved to a hypothesis by negation.
For example, a goal `⊢ a ≤ b` can become `a > b ⊢ false`.
If this is the case, it applies the appropriate lemma and introduces the new hypothesis.
It returns the type of the terms in the comparison (e.g. the type of `a` and `b` above) and the
newly introduced local constant.
Otherwise returns `none`.
-/
def applyContrLemma (g : MVarId) : MetaM (Option (Expr × Expr) × MVarId) := do
  try
    let (nm, tp) ← getContrLemma (← withReducible g.getType')
    let [g] ← g.apply (← mkConst' nm) | failure
    let (f, g) ← g.intro1P
    return (some (tp, .fvar f), g)
  catch _ => return (none, g)


/-- A map of keys to values, where the keys are `Expr` up to defeq and one key can be
associated to multiple values. -/
abbrev ExprMultiMap α := Array (Expr × List α)


/-- Retrieves the list of values at a key, as well as the index of the key for later modification.
(If the key is not in the map it returns `self.size` as the index.) -/
def ExprMultiMap.find {α : Type} (self : ExprMultiMap α) (k : Expr) : MetaM (Nat × List α) := do
  for h : i in [:self.size] do
    let (k', vs) := self[i]'h.2
    if ← isDefEq k' k then
      return (i, vs)
  return (self.size, [])


/-- Insert a new value into the map at key `k`. This does a defeq check with all other keys
in the map. -/
def ExprMultiMap.insert {α : Type} (self : ExprMultiMap α) (k : Expr) (v : α) :
    MetaM (ExprMultiMap α) := do
  for h : i in [:self.size] do
    if ← isDefEq (self[i]'h.2).1 k then
      return self.modify i fun (k, vs) => (k, v::vs)
  return self.push (k, [v])


/--
`partitionByType l` takes a list `l` of proofs of comparisons. It sorts these proofs by
the type of the variables in the comparison, e.g. `(a : ℚ) < 1` and `(b : ℤ) > c` will be separated.
Returns a map from a type to a list of comparisons over that type.
-/
def partitionByType (l : List Expr) : MetaM (ExprMultiMap Expr) :=
  l.foldlM (fun m h => do m.insert (← typeOfIneqProof h) h) #[]


/--
Given a list `ls` of lists of proofs of comparisons, `findLinarithContradiction cfg ls` will try to
prove `False` by calling `linarith` on each list in succession. It will stop at the first proof of
`False`, and fail if no contradiction is found with any list.
-/
def findLinarithContradiction (cfg : LinarithConfig) (g : MVarId) (ls : List (List Expr)) :
    MetaM Expr :=
  try
    ls.firstM (fun L => proveFalseByLinarith cfg.transparency cfg.oracle cfg.discharger g L)
  catch e => throwError "linarith failed to find a contradiction\n{g}\n{e.toMessageData}"



/--
Given a list `hyps` of proofs of comparisons, `runLinarith cfg hyps prefType`
preprocesses `hyps` according to the list of preprocessors in `cfg`.
This results in a list of branches (typically only one),
each of which must succeed in order to close the goal.

In each branch, we partition the list of hypotheses by type, and run `linarith` on each class
in the partition; one of these must succeed in order for `linarith` to succeed on this branch.
If `prefType` is given, it will first use the class of proofs of comparisons over that type.
-/
-- If it succeeds, the passed metavariable should have been assigned.
def runLinarith (cfg : LinarithConfig) (prefType : Option Expr) (g : MVarId)
    (hyps : List Expr) : MetaM Unit := do
  let singleProcess (g : MVarId) (hyps : List Expr) : MetaM Expr := g.withContext do
    linarithTraceProofs s!"after preprocessing, linarith has {hyps.length} facts:" hyps
    let hyp_set ← partitionByType hyps
    trace[linarith] "hypotheses appear in {hyp_set.size} different types"
      if let some t := prefType then
        let (i, vs) ← hyp_set.find t
        proveFalseByLinarith cfg.transparency cfg.oracle cfg.discharger g vs <|>
        findLinarithContradiction cfg g ((hyp_set.eraseIdxIfInBounds i).toList.map (·.2))
      else findLinarithContradiction cfg g (hyp_set.toList.map (·.2))
  let mut preprocessors := cfg.preprocessors
  if cfg.splitNe then
    preprocessors := Linarith.removeNe :: preprocessors
  if cfg.splitHypotheses then
    preprocessors := Linarith.splitConjunctions.globalize.branching :: preprocessors
  let branches ← preprocess preprocessors g hyps
  for (g, es) in branches do
    let r ← singleProcess g es
    g.assign r
  -- Verify that we closed the goal. Failure here should only result from a bad `Preprocessor`.
  (Expr.mvar g).ensureHasNoMVars

-- /--
-- `filterHyps restr_type hyps` takes a list of proofs of comparisons `hyps`, and filters it
-- to only those that are comparisons over the type `restr_type`.
-- -/
-- def filterHyps (restr_type : Expr) (hyps : List Expr) : MetaM (List Expr) :=
--   hyps.filterM (fun h => do
--     let ht ← inferType h
--     match getContrLemma ht with
--     | some (_, htype) => isDefEq htype restr_type
--     | none => return false)


/--
`linarith only_on hyps cfg` tries to close the goal using linear arithmetic. It fails
if it does not succeed at doing this.

* `hyps` is a list of proofs of comparisons to include in the search.
* If `only_on` is true, the search will be restricted to `hyps`. Otherwise it will use all
  comparisons in the local context.
* If `cfg.transparency := semireducible`,
  it will unfold semireducible definitions when trying to match atomic expressions.
-/
partial def linarith (only_on : Bool) (hyps : List Expr) (cfg : LinarithConfig := {})
    (g : MVarId) : MetaM Unit := g.withContext do
  -- if the target is an equality, we run `linarith` twice, to prove ≤ and ≥.
  if (← whnfR (← instantiateMVars (← g.getType))).isEq then
    trace[linarith] "target is an equality: splitting"
    if let some [g₁, g₂] ← try? (g.apply (← mkConst' ``eq_of_not_lt_of_not_gt)) then
      linarith only_on hyps cfg g₁
      linarith only_on hyps cfg g₂
      return

  /- If we are proving a comparison goal (and not just `False`), we consider the type of the
    elements in the comparison to be the "preferred" type. That is, if we find comparison
    hypotheses in multiple types, we will run `linarith` on the goal type first.
    In this case we also receive a new variable from moving the goal to a hypothesis.
    Otherwise, there is no preferred type and no new variable; we simply change the goal to `False`.
  -/

  let (g, target_type, new_var) ← match ← applyContrLemma g with
  | (none, g) =>
    if cfg.exfalso then
      trace[linarith] "using exfalso"
      pure (← g.exfalso, none, none)
    else
      pure (g, none, none)
  | (some (t, v), g) => pure (g, some t, some v)

  g.withContext do
  -- set up the list of hypotheses, considering the `only_on` and `restrict_type` options
    let hyps ← (if only_on then return new_var.toList ++ hyps
      else return (← getLocalHyps).toList ++ hyps)

    -- TODO in mathlib3 we could specify a restriction to a single type.
    -- I haven't done that here because I don't know how to store a `Type` in `LinarithConfig`.
    -- There's only one use of the `restrict_type` configuration option in mathlib3,
    -- and it can be avoided just by using `linarith only`.

    linarithTraceProofs "linarith is running on the following hypotheses:" hyps
    runLinarith cfg target_type g hyps


/-- Syntax for the arguments of `linarith`, after the optional `!`. -/
syntax linarithArgsRest := optConfig (&" only")? (" [" term,* "]")?


/--
`linarith` attempts to find a contradiction between hypotheses that are linear (in)equalities.
Equivalently, it can prove a linear inequality by assuming its negation and proving `False`.

In theory, `linarith` should prove any goal that is true in the theory of linear arithmetic over
the rationals. While there is some special handling for non-dense orders like `Nat` and `Int`,
this tactic is not complete for these theories and will not prove every true goal. It will solve
goals over arbitrary types that instantiate `LinearOrderedCommRing`.

An example:
```lean
example (x y z : ℚ) (h1 : 2*x < 3*y) (h2 : -4*x + 2*z < 0)
        (h3 : 12*y - 4* z < 0) : False := by
  linarith
```

`linarith` will use all appropriate hypotheses and the negation of the goal, if applicable.
Disequality hypotheses require case splitting and are not normally considered
(see the `splitNe` option below).

`linarith [t1, t2, t3]` will additionally use proof terms `t1, t2, t3`.

`linarith only [h1, h2, h3, t1, t2, t3]` will use only the goal (if relevant), local hypotheses
`h1`, `h2`, `h3`, and proofs `t1`, `t2`, `t3`. It will ignore the rest of the local context.

`linarith!` will use a stronger reducibility setting to try to identify atoms. For example,
```lean
example (x : ℚ) : id x ≥ x := by
  linarith
```
will fail, because `linarith` will not identify `x` and `id x`. `linarith!` will.
This can sometimes be expensive.

`linarith (config := { .. })` takes a config object with five
optional arguments:
* `discharger` specifies a tactic to be used for reducing an algebraic equation in the
  proof stage. The default is `ring`. Other options include `simp` for basic
  problems.
* `transparency` controls how hard `linarith` will try to match atoms to each other. By default
  it will only unfold `reducible` definitions.
* If `splitHypotheses` is true, `linarith` will split conjunctions in the context into separate
  hypotheses.
* If `splitNe` is `true`, `linarith` will case split on disequality hypotheses.
  For a given `x ≠ y` hypothesis, `linarith` is run with both `x < y` and `x > y`,
  and so this runs linarith exponentially many times with respect to the number of
  disequality hypotheses. (`false` by default.)
* If `exfalso` is `false`, `linarith` will fail when the goal is neither an inequality nor `False`.
  (`true` by default.)
* `restrict_type` (not yet implemented in mathlib4)
  will only use hypotheses that are inequalities over `tp`. This is useful
  if you have e.g. both integer and rational valued inequalities in the local context, which can
  sometimes confuse the tactic.

A variant, `nlinarith`, does some basic preprocessing to handle some nonlinear goals.

The option `set_option trace.linarith true` will trace certain intermediate stages of the `linarith`
routine.
-/
syntax (name := linarith) "linarith" "!"? linarithArgsRest : tactic


@[inherit_doc linarith] macro "linarith!" rest:linarithArgsRest : tactic =>
  `(tactic| linarith ! $rest:linarithArgsRest)


/--
An extension of `linarith` with some preprocessing to allow it to solve some nonlinear arithmetic
problems. (Based on Coq's `nra` tactic.) See `linarith` for the available syntax of options,
which are inherited by `nlinarith`; that is, `nlinarith!` and `nlinarith only [h1, h2]` all work as
in `linarith`. The preprocessing is as follows:

* For every subterm `a ^ 2` or `a * a` in a hypothesis or the goal,
  the assumption `0 ≤ a ^ 2` or `0 ≤ a * a` is added to the context.
* For every pair of hypotheses `a1 R1 b1`, `a2 R2 b2` in the context, `R1, R2 ∈ {<, ≤, =}`,
  the assumption `0 R' (b1 - a1) * (b2 - a2)` is added to the context (non-recursively),
  where `R ∈ {<, ≤, =}` is the appropriate comparison derived from `R1, R2`.
-/
syntax (name := nlinarith) "nlinarith" "!"? linarithArgsRest : tactic

@[inherit_doc nlinarith] macro "nlinarith!" rest:linarithArgsRest : tactic =>
  `(tactic| nlinarith ! $rest:linarithArgsRest)


/-- Elaborate `t` in a way that is suitable for linarith. -/
def elabLinarithArg (tactic : Name) (t : Term) : TacticM Expr := Term.withoutErrToSorry do
  let (e, mvars) ← elabTermWithHoles t none tactic
  unless mvars.isEmpty do
    throwErrorAt t "Argument passed to {tactic} has metavariables:{indentD e}"
  return e


/--
Allow elaboration of `LinarithConfig` arguments to tactics.
-/
declare_config_elab elabLinarithConfig Linarith.LinarithConfig


elab_rules : tactic
  | `(tactic| linarith $[!%$bang]? $cfg:optConfig $[only%$o]? $[[$args,*]]?) => withMainContext do
    let args ← ((args.map (TSepArray.getElems)).getD {}).mapM (elabLinarithArg `linarith)
    let cfg := (← elabLinarithConfig cfg).updateReducibility bang.isSome
    commitIfNoEx do liftMetaFinishingTactic <| Linarith.linarith o.isSome args.toList cfg

-- TODO restore this when `add_tactic_doc` is ported
-- add_tactic_doc
-- { name       := "linarith",
--   category   := doc_category.tactic,
--   decl_names := [`tactic.interactive.linarith],
--   tags       := ["arithmetic", "decision procedure", "finishing"] }


elab_rules : tactic
  | `(tactic| nlinarith $[!%$bang]? $cfg:optConfig $[only%$o]? $[[$args,*]]?) => withMainContext do
    let args ← ((args.map (TSepArray.getElems)).getD {}).mapM (elabLinarithArg `nlinarith)
    let cfg := (← elabLinarithConfig cfg).updateReducibility bang.isSome
    let cfg := { cfg with
      preprocessors := cfg.preprocessors.concat nlinarithExtras }
    commitIfNoEx do liftMetaFinishingTactic <| Linarith.linarith o.isSome args.toList cfg

-- TODO restore this when `add_tactic_doc` is ported
-- add_tactic_doc
-- { name       := "nlinarith",
--   category   := doc_category.tactic,
--   decl_names := [`tactic.interactive.nlinarith],
--   tags       := ["arithmetic", "decision procedure", "finishing"] }

