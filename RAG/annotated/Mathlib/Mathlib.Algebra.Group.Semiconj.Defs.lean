/-- `x` is semiconjugate to `y` by `a`, if `a * x = y * a`. -/
@[to_additive "`x` is additive semiconjugate to `y` by `a` if `a + x = y + a`"]
def SemiconjBy [Mul M] (a x y : M) : Prop :=
  a * x = y * a


/-- Equality behind `SemiconjBy a x y`; useful for rewriting. -/
@[to_additive "Equality behind `AddSemiconjBy a x y`; useful for rewriting."]
protected theorem eq [Mul S] {a x y : S} (h : SemiconjBy a x y) : a * x = y * a :=
  h


/-- If `a` semiconjugates `x` to `y` and `x'` to `y'`,
then it semiconjugates `x * x'` to `y * y'`. -/
@[to_additive (attr := simp) "If `a` semiconjugates `x` to `y` and `x'` to `y'`,
then it semiconjugates `x + x'` to `y + y'`."]
theorem mul_right (h : SemiconjBy a x y) (h' : SemiconjBy a x' y') :
    SemiconjBy a (x * x') (y * y') := by
  /-
    S : Type u_1
    inst✝ : Semigroup S
    a x y x' y' : S
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ SemiconjBy a (HMul.hMul x x') (HMul.hMul y y')
  -/
  unfold SemiconjBy
  -- TODO this could be done using `assoc_rw` if/when this is ported to mathlib4
  /-
    S : Type u_1
    inst✝ : Semigroup S
    a x y x' y' : S
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ Eq (HMul.hMul a (HMul.hMul x x')) (HMul.hMul (HMul.hMul y y') a)
  -/
  rw [← mul_assoc, h.eq, mul_assoc, h'.eq, ← mul_assoc]
  /-
    🎉 no goals
  -/


/-- If `b` semiconjugates `x` to `y` and `a` semiconjugates `y` to `z`, then `a * b`
semiconjugates `x` to `z`. -/
@[to_additive "If `b` semiconjugates `x` to `y` and `a` semiconjugates `y` to `z`, then `a + b`
semiconjugates `x` to `z`."]
theorem mul_left (ha : SemiconjBy a y z) (hb : SemiconjBy b x y) : SemiconjBy (a * b) x z := by
  /-
    S : Type u_1
    inst✝ : Semigroup S
    a b x y z : S
    ha : SemiconjBy a y z
    hb : SemiconjBy b x y
    ⊢ SemiconjBy (HMul.hMul a b) x z
  -/
  unfold SemiconjBy
  /-
    S : Type u_1
    inst✝ : Semigroup S
    a b x y z : S
    ha : SemiconjBy a y z
    hb : SemiconjBy b x y
    ⊢ Eq (HMul.hMul (HMul.hMul a b) x) (HMul.hMul z (HMul.hMul a b))
  -/
  rw [mul_assoc, hb.eq, ← mul_assoc, ha.eq, mul_assoc]
  /-
    🎉 no goals
  -/


/-- The relation “there exists an element that semiconjugates `a` to `b`” on a semigroup
is transitive. -/
@[to_additive "The relation “there exists an element that semiconjugates `a` to `b`” on an additive
semigroup is transitive."]
protected theorem transitive : Transitive fun a b : S ↦ ∃ c, SemiconjBy c a b
  | _, _, _, ⟨x, hx⟩, ⟨y, hy⟩ => ⟨y * x, hy.mul_left hx⟩


/-- Any element semiconjugates `1` to `1`. -/
@[to_additive (attr := simp) "Any element semiconjugates `0` to `0`."]
                                                   /-
                                                     M : Type u_2
                                                     inst✝ : MulOneClass M
                                                     a : M
                                                     ⊢ SemiconjBy a 1 1
                                                   -/
theorem one_right (a : M) : SemiconjBy a 1 1 := by rw [SemiconjBy, mul_one, one_mul]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- One semiconjugates any element to itself. -/
@[to_additive (attr := simp) "Zero semiconjugates any element to itself."]
theorem one_left (x : M) : SemiconjBy 1 x x :=
  Eq.symm <| one_right x


/-- The relation “there exists an element that semiconjugates `a` to `b`” on a monoid (or, more
generally, on `MulOneClass` type) is reflexive. -/
@[to_additive "The relation “there exists an element that semiconjugates `a` to `b`” on an additive
monoid (or, more generally, on an `AddZeroClass` type) is reflexive."]
protected theorem reflexive : Reflexive fun a b : M ↦ ∃ c, SemiconjBy c a b
  | a => ⟨1, one_left a⟩


@[to_additive (attr := simp)]
theorem pow_right {a x y : M} (h : SemiconjBy a x y) (n : ℕ) : SemiconjBy a (x ^ n) (y ^ n) := by
  induction n with
  | zero =>
    rw [pow_zero, pow_zero]
    exact SemiconjBy.one_right _
  | succ n ih =>
    rw [pow_succ, pow_succ]
    exact ih.mul_right h


/-- `a` semiconjugates `x` to `a * x * a⁻¹`. -/
@[to_additive "`a` semiconjugates `x` to `a + x + -a`."]
theorem conj_mk (a x : G) : SemiconjBy a x (a * x * a⁻¹) := by
  /-
    G : Type u_3
    inst✝ : Group G
    a x : G
    ⊢ SemiconjBy a x (HMul.hMul (HMul.hMul a x) (Inv.inv a))
  -/
  unfold SemiconjBy; rw [mul_assoc, inv_mul_cancel, mul_one]
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp)]
theorem conj_iff {a x y b : G} :
    SemiconjBy (b * a * b⁻¹) (b * x * b⁻¹) (b * y * b⁻¹) ↔ SemiconjBy a x y := by
  /-
    G : Type u_3
    inst✝ : Group G
    a x y b : G
    ⊢ Iff (SemiconjBy (HMul.hMul (HMul.hMul b a) (Inv.inv b)) (HMul.hMul (HMul.hMu …
  -/
  unfold SemiconjBy
  /-
    G : Type u_3
    inst✝ : Group G
    a x y b : G
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul (HMul.hMul b a) (Inv.inv b)) (HMul.hMul (HMul. …
  -/
  simp only [← mul_assoc, inv_mul_cancel_right]
  /-
    G : Type u_3
    inst✝ : Group G
    a x y b : G
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul (HMul.hMul b a) x) (Inv.inv b)) (HMul.hMul (HM …
  -/
  repeat rw [mul_assoc]
  /-
    G : Type u_3
    inst✝ : Group G
    a x y b : G
    ⊢ Iff (Eq (HMul.hMul b (HMul.hMul a (HMul.hMul x (Inv.inv b)))) (HMul.hMul b ( …
  -/
  rw [mul_left_cancel_iff, ← mul_assoc, ← mul_assoc, mul_right_cancel_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem semiconjBy_iff_eq [CancelCommMonoid M] {a x y : M} : SemiconjBy a x y ↔ x = y :=
                                                                  /-
                                                                    M : Type u_2
                                                                    inst✝ : CancelCommMonoid M
                                                                    a x y : M
                                                                    h : Eq x y
                                                                    ⊢ SemiconjBy a x y
                                                                  -/
  ⟨fun h => mul_left_cancel (h.trans (mul_comm _ _)), fun h => by rw [h, SemiconjBy, mul_comm]⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

