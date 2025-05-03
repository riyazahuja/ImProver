/--
Given `[EstimatorData a ε]`
* a term `e : ε` can be interpreted via `bound a e : α` as a lower bound for `a`, and
* we can ask for an improved lower bound via `improve a e : Option ε`.

The value `a` in `α` that we are estimating is hidden inside a `Thunk` to avoid evaluation.
 -/
class EstimatorData (a : Thunk α) (ε : Type*) where
  /-- The value of the bound for `a` representation by a term of `ε`. -/
  bound : ε → α
  /-- Generate an improved lower bound. -/
  improve : ε → Option ε


/--
Given `[Estimator a ε]`
* we have `bound a e ≤ a.get`, and
* `improve a e` returns none iff `bound a e = a.get`,
  and otherwise it returns a strictly better bound.
-/
class Estimator [Preorder α] (a : Thunk α) (ε : Type*) extends EstimatorData a ε where
  /-- The calculated bounds are always lower bounds. -/
  bound_le e : bound e ≤ a.get
  /-- Calling `improve` either gives a strictly better bound,
  or a proof that the current bound is exact. -/
  improve_spec e : match improve e with
    | none => bound e = a.get
    | some e' => bound e < bound e'


/-- A trivial estimator, containing the actual value. -/
abbrev Estimator.trivial.{u} {α : Type u} (a : α) : Type u := { b : α // b = a }


instance {a : α} : Bot (Estimator.trivial a) := ⟨⟨a, rfl⟩⟩


instance : WellFoundedGT Unit where
  wf := ⟨fun .unit => ⟨Unit.unit, nofun⟩⟩


instance (a : α) : WellFoundedGT (Estimator.trivial a) :=
  let f : Estimator.trivial a ≃o Unit := RelIso.ofUniqueOfRefl _ _
  let f' : Estimator.trivial a ↪o Unit := f.toOrderEmbedding
  f'.wellFoundedGT


instance {a : α} : Estimator (Thunk.pure a) (Estimator.trivial a) where
  bound b := b.val
  improve _ := none
  bound_le b := b.prop.le
  improve_spec b := b.prop


attribute [local instance] WellFoundedGT.toWellFoundedRelation in
/-- Implementation of `Estimator.improveUntil`. -/
def Estimator.improveUntilAux
    (a : Thunk α) (p : α → Bool) [Estimator a ε]
    [WellFoundedGT (range (bound a : ε → α))]
    (e : ε) (r : Bool) : Except (Option ε) ε :=
    if p (bound a e) then
      return e
    else
      match improve a e, improve_spec e with
      | none, _ => .error <| if r then none else e
      | some e', _ =>
        improveUntilAux a p e' true
termination_by (⟨_, mem_range_self e⟩ : range (bound a))


/--
Improve an estimate until it satisfies a predicate,
or else return the best available estimate, if any improvement was made.
-/
def Estimator.improveUntil (a : Thunk α) (p : α → Bool)
    [Estimator a ε] [WellFoundedGT (range (bound a : ε → α))] (e : ε) :
    Except (Option ε) ε :=
  Estimator.improveUntilAux a p e false


attribute [local instance] WellFoundedGT.toWellFoundedRelation in
/--
If `Estimator.improveUntil a p e` returns `some e'`, then `bound a e'` satisfies `p`.
Otherwise, that value `a` must not satisfy `p`.
-/
theorem Estimator.improveUntilAux_spec (a : Thunk α) (p : α → Bool)
    [Estimator a ε] [WellFoundedGT (range (bound a : ε → α))] (e : ε) (r : Bool) :
    match Estimator.improveUntilAux a p e r with
    | .error _ => ¬ p a.get
    | .ok e' => p (bound a e') := by
  /-
    α : Type u_1
    ε : Type u_2
    inst✝² : Preorder α
    a : Thunk α
    p : α → Bool
    inst✝¹ : Estimator a ε
    inst✝ : WellFoundedGT ↑(Set.range (EstimatorData.bound a))
    e : ε
    r : Bool
    ⊢ Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (Estimator.improveUnt …
  -/
  rw [Estimator.improveUntilAux]
  /-
    α : Type u_1
    ε : Type u_2
    inst✝² : Preorder α
    a : Thunk α
    p : α → Bool
    inst✝¹ : Estimator a ε
    inst✝ : WellFoundedGT ↑(Set.range (EstimatorData.bound a))
    e : ε
    r : Bool
    ⊢ Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (ite (Eq (p (Estimato …
  -/
  by_cases h : p (bound a e)
    /-
      case pos
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a : Thunk α
      p : α → Bool
      inst✝¹ : Estimator a ε
      inst✝ : WellFoundedGT ↑(Set.range (EstimatorData.bound a))
      e : ε
      r : Bool
      h : Eq (p (EstimatorData.bound a e)) Bool.true
      ⊢ Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (ite (Eq (p (Estimato …
    -/
  · simp only [h]; exact h
                   /-
                     🎉 no goals
                   -/
    /-
      case neg
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a : Thunk α
      p : α → Bool
      inst✝¹ : Estimator a ε
      inst✝ : WellFoundedGT ↑(Set.range (EstimatorData.bound a))
      e : ε
      r : Bool
      h : Not (Eq (p (EstimatorData.bound a e)) Bool.true)
      ⊢ Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (ite (Eq (p (Estimato …
    -/
  · simp only [h]
    match improve a e, improve_spec e with
    | none, eq =>
      simp only [Bool.not_eq_true]
      rw [eq] at h
      exact Bool.bool_eq_false h
    | some e', _ =>
      exact Estimator.improveUntilAux_spec a p e' true
termination_by (⟨_, mem_range_self e⟩ : range (bound a))


/--
If `Estimator.improveUntil a p e` returns `some e'`, then `bound a e'` satisfies `p`.
Otherwise, that value `a` must not satisfy `p`.
-/
theorem Estimator.improveUntil_spec
    (a : Thunk α) (p : α → Bool) [Estimator a ε] [WellFoundedGT (range (bound a : ε → α))] (e : ε) :
    match Estimator.improveUntil a p e with
    | .error _ => ¬ p a.get
    | .ok e' => p (bound a e') :=
  Estimator.improveUntilAux_spec a p e false


@[simps]
instance [Add α] {a b : Thunk α} (εa εb : Type*) [EstimatorData a εa] [EstimatorData b εb] :
    EstimatorData (a + b) (εa × εb) where
  bound e := bound a e.1 + bound b e.2
  improve e := match improve a e.1 with
  | some e' => some { e with fst := e' }
  | none => match improve b e.2 with
    | some e' => some { e with snd := e' }
    | none => none


instance (a b : Thunk ℕ) {εa εb : Type*} [Estimator a εa] [Estimator b εb] :
    Estimator (a + b) (εa × εb) where
  bound_le e :=
    Nat.add_le_add (Estimator.bound_le e.1) (Estimator.bound_le e.2)
  improve_spec e := by
    /-
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a b : Thunk Nat
      εa : Type u_3
      εb : Type u_4
      inst✝¹ : Estimator a εa
      inst✝ : Estimator b εb
      e : Prod εa εb
      ⊢ Estimator.match_1 (Prod εa εb) (fun x => Prop) (EstimatorData.improve (HAdd. …
    -/
    dsimp
    /-
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a b : Thunk Nat
      εa : Type u_3
      εb : Type u_4
      inst✝¹ : Estimator a εa
      inst✝ : Estimator b εb
      e : Prod εa εb
      ⊢ Estimator.match_1 (Prod εa εb) (fun x => Prop) (instEstimatorDataHAddThunkPr …
    -/
    have s₁ := Estimator.improve_spec (a := a) e.1
    /-
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a b : Thunk Nat
      εa : Type u_3
      εb : Type u_4
      inst✝¹ : Estimator a εa
      inst✝ : Estimator b εb
      e : Prod εa εb
      s₁ : Estimator.match_1 εa (fun x => Prop) (EstimatorData.improve a e.1) (fun _ …
      ⊢ Estimator.match_1 (Prod εa εb) (fun x => Prop) (instEstimatorDataHAddThunkPr …
    -/
    have s₂ := Estimator.improve_spec (a := b) e.2
    /-
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a b : Thunk Nat
      εa : Type u_3
      εb : Type u_4
      inst✝¹ : Estimator a εa
      inst✝ : Estimator b εb
      e : Prod εa εb
      s₁ : Estimator.match_1 εa (fun x => Prop) (EstimatorData.improve a e.1) (fun _ …
      s₂ : Estimator.match_1 εb (fun x => Prop) (EstimatorData.improve b e.2) (fun _ …
      ⊢ Estimator.match_1 (Prod εa εb) (fun x => Prop) (instEstimatorDataHAddThunkPr …
    -/
    revert s₁ s₂
    /-
      α : Type u_1
      ε : Type u_2
      inst✝² : Preorder α
      a b : Thunk Nat
      εa : Type u_3
      εb : Type u_4
      inst✝¹ : Estimator a εa
      inst✝ : Estimator b εb
      e : Prod εa εb
      ⊢ (Estimator.match_1 εa (fun x => Prop) (EstimatorData.improve a e.1) (fun _ = …
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    cases improve a e.fst <;> cases improve b e.snd <;> intro s₁ s₂ <;> simp_all only
      /-
        case none.some
        α : Type u_1
        ε : Type u_2
        inst✝² : Preorder α
        a b : Thunk Nat
        εa : Type u_3
        εb : Type u_4
        inst✝¹ : Estimator a εa
        inst✝ : Estimator b εb
        e : Prod εa εb
        val✝ : εb
        s₁ : Eq (EstimatorData.bound a e.1) a.get
        s₂ : LT.lt (EstimatorData.bound b e.2) (EstimatorData.bound b val✝)
        ⊢ LT.lt (HAdd.hAdd a.get (EstimatorData.bound b e.2)) (HAdd.hAdd a.get (Estima …
      -/
    · apply Nat.add_lt_add_left s₂
      /-
        🎉 no goals
      -/
      /-
        case some.none
        α : Type u_1
        ε : Type u_2
        inst✝² : Preorder α
        a b : Thunk Nat
        εa : Type u_3
        εb : Type u_4
        inst✝¹ : Estimator a εa
        inst✝ : Estimator b εb
        e : Prod εa εb
        val✝ : εa
        s₁ : LT.lt (EstimatorData.bound a e.1) (EstimatorData.bound a val✝)
        s₂ : Eq (EstimatorData.bound b e.2) b.get
        ⊢ LT.lt (HAdd.hAdd (EstimatorData.bound a e.1) b.get) (HAdd.hAdd (EstimatorDat …
      -/
    · apply Nat.add_lt_add_right s₁
      /-
        🎉 no goals
      -/
      /-
        case some.some
        α : Type u_1
        ε : Type u_2
        inst✝² : Preorder α
        a b : Thunk Nat
        εa : Type u_3
        εb : Type u_4
        inst✝¹ : Estimator a εa
        inst✝ : Estimator b εb
        e : Prod εa εb
        val✝¹ : εa
        val✝ : εb
        s₁ : LT.lt (EstimatorData.bound a e.1) (EstimatorData.bound a val✝¹)
        s₂ : LT.lt (EstimatorData.bound b e.2) (EstimatorData.bound b val✝)
        ⊢ LT.lt (HAdd.hAdd (EstimatorData.bound a e.1) (EstimatorData.bound b e.2)) (H …
      -/
    · apply Nat.add_lt_add_right s₁
      /-
        🎉 no goals
      -/


/--
An estimator for `(a, b)` can be turned into an estimator for `a`,
simply by repeatedly running `improve` until the first factor "improves".
The hypothesis that `>` is well-founded on `{ q // q ≤ (a, b) }` ensures this terminates.
-/
structure Estimator.fst
    (p : Thunk (α × β)) (ε : Type*) [Estimator p ε] where
  /-- The wrapped bound for a value in `α × β`,
  which we will use as a bound for the first component. -/
  inner : ε


instance {a : Thunk α} [Estimator a ε] : WellFoundedGT (range (bound a : ε → α)) :=
  let f : range (bound a : ε → α) ↪o { x // x ≤ a.get } :=
                               /-
                                 α : Type u_1
                                 ε : Type u_2
                                 β : Type u_3
                                 inst✝³ : PartialOrder α
                                 inst✝² : PartialOrder β
                                 inst✝¹ : ∀ (a : α), WellFoundedGT (Subtype fun x => LE.le x a)
                                 a : Thunk α
                                 inst✝ : Estimator a ε
                                 ⊢ ∀ (a_1 : α), Membership.mem (Set.range (EstimatorData.bound a)) a_1 → LE.le  …
                               -/
    Subtype.orderEmbedding (by rintro _ ⟨e, rfl⟩; exact Estimator.bound_le e)
                                                  /-
                                                    🎉 no goals
                                                  -/
  f.wellFoundedGT


instance [DecidableRel ((· : α) < ·)] {a : Thunk α} {b : Thunk β}
    (ε : Type*) [Estimator (a.prod b) ε] [∀ (p : α × β), WellFoundedGT { q // q ≤ p }] :
    EstimatorData a (Estimator.fst (a.prod b) ε) where
  bound e := (bound (a.prod b) e.inner).1
  improve e :=
    let bd := (bound (a.prod b) e.inner).1
    Estimator.improveUntil (a.prod b) (fun p => bd < p.1) e.inner
      |>.toOption |>.map Estimator.fst.mk


/-- Given an estimator for a pair, we can extract an estimator for the first factor. -/
-- This isn't an instance as at the sole use case we need to provide
-- the instance arguments by hand anyway.
def Estimator.fstInst [DecidableRel ((· : α) < ·)] [∀ (p : α × β), WellFoundedGT { q // q ≤ p }]
    (a : Thunk α) (b : Thunk β) (i : Estimator (a.prod b) ε) :
    Estimator a (Estimator.fst (a.prod b) ε) where
  bound_le e := (Estimator.bound_le e.inner : bound (a.prod b) e.inner ≤ (a.get, b.get)).1
  improve_spec e := by
    /-
      α : Type u_1
      ε : Type u_2
      β : Type u_3
      inst✝⁴ : PartialOrder α
      inst✝³ : PartialOrder β
      inst✝² : ∀ (a : α), WellFoundedGT (Subtype fun x => LE.le x a)
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : ∀ (p : Prod α β), WellFoundedGT (Subtype fun q => LE.le q p)
      a : Thunk α
      b : Thunk β
      i : Estimator (a.prod b) ε
      e : Estimator.fst (a.prod b) ε
      ⊢ Estimator.match_1 (Estimator.fst (a.prod b) ε) (fun x => Prop) (EstimatorDat …
    -/
    let bd := (bound (a.prod b) e.inner).1
    /-
      α : Type u_1
      ε : Type u_2
      β : Type u_3
      inst✝⁴ : PartialOrder α
      inst✝³ : PartialOrder β
      inst✝² : ∀ (a : α), WellFoundedGT (Subtype fun x => LE.le x a)
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : ∀ (p : Prod α β), WellFoundedGT (Subtype fun q => LE.le q p)
      a : Thunk α
      b : Thunk β
      i : Estimator (a.prod b) ε
      e : Estimator.fst (a.prod b) ε
      bd : α := (EstimatorData.bound (a.prod b) e.inner).1
      ⊢ Estimator.match_1 (Estimator.fst (a.prod b) ε) (fun x => Prop) (EstimatorDat …
    -/
    have := Estimator.improveUntil_spec (a.prod b) (fun p => bd < p.1) e.inner
    /-
      α : Type u_1
      ε : Type u_2
      β : Type u_3
      inst✝⁴ : PartialOrder α
      inst✝³ : PartialOrder β
      inst✝² : ∀ (a : α), WellFoundedGT (Subtype fun x => LE.le x a)
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : ∀ (p : Prod α β), WellFoundedGT (Subtype fun q => LE.le q p)
      a : Thunk α
      b : Thunk β
      i : Estimator (a.prod b) ε
      e : Estimator.fst (a.prod b) ε
      bd : α := (EstimatorData.bound (a.prod b) e.inner).1
      this : Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (Estimator.impro …
      ⊢ Estimator.match_1 (Estimator.fst (a.prod b) ε) (fun x => Prop) (EstimatorDat …
    -/
    revert this
    /-
      α : Type u_1
      ε : Type u_2
      β : Type u_3
      inst✝⁴ : PartialOrder α
      inst✝³ : PartialOrder β
      inst✝² : ∀ (a : α), WellFoundedGT (Subtype fun x => LE.le x a)
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : ∀ (p : Prod α β), WellFoundedGT (Subtype fun q => LE.le q p)
      a : Thunk α
      b : Thunk β
      i : Estimator (a.prod b) ε
      e : Estimator.fst (a.prod b) ε
      bd : α := (EstimatorData.bound (a.prod b) e.inner).1
      ⊢ (Estimator.improveUntilAux_spec.match_1 (fun x => Prop) (Estimator.improveUn …
    -/
    simp only [EstimatorData.improve, decide_eq_true_eq]
    match Estimator.improveUntil (a.prod b) _ _ with
    | .error _ =>
      simp only [Option.map_none']
      exact fun w =>
        eq_of_le_of_not_lt
          (Estimator.bound_le e.inner : bound (a.prod b) e.inner ≤ (a.get, b.get)).1 w
    | .ok e' => exact fun w => w


