/-- Two elements commute if `a * b = b * a`. -/
@[to_additive "Two elements additively commute if `a + b = b + a`"]
def Commute [Mul S] (a b : S) : Prop :=
  SemiconjBy a b b


/--
Two elements `a` and `b` commute if `a * b = b * a`.
-/
@[to_additive]
theorem commute_iff_eq [Mul S] (a b : S) : Commute a b ↔ a * b = b * a := Iff.rfl


/-- Equality behind `Commute a b`; useful for rewriting. -/
@[to_additive "Equality behind `AddCommute a b`; useful for rewriting."]
protected theorem eq {a b : S} (h : Commute a b) : a * b = b * a :=
  h


/-- Any element commutes with itself. -/
@[to_additive (attr := refl, simp) "Any element commutes with itself."]
protected theorem refl (a : S) : Commute a a :=
  Eq.refl (a * a)


/-- If `a` commutes with `b`, then `b` commutes with `a`. -/
@[to_additive (attr := symm) "If `a` commutes with `b`, then `b` commutes with `a`."]
protected theorem symm {a b : S} (h : Commute a b) : Commute b a :=
  Eq.symm h


@[to_additive]
protected theorem semiconjBy {a b : S} (h : Commute a b) : SemiconjBy a b b :=
  h


@[to_additive]
protected theorem symm_iff {a b : S} : Commute a b ↔ Commute b a :=
  ⟨Commute.symm, Commute.symm⟩


@[to_additive]
instance : IsRefl S Commute :=
  ⟨Commute.refl⟩

-- This instance is useful for `Finset.noncommProd`

@[to_additive]
instance on_isRefl {f : G → S} : IsRefl G fun a b => Commute (f a) (f b) :=
  ⟨fun _ => Commute.refl _⟩


/-- If `a` commutes with both `b` and `c`, then it commutes with their product. -/
@[to_additive (attr := simp)
"If `a` commutes with both `b` and `c`, then it commutes with their sum."]
theorem mul_right (hab : Commute a b) (hac : Commute a c) : Commute a (b * c) :=
  SemiconjBy.mul_right hab hac
-- I think `ₓ` is necessary because of the `mul` vs `HMul` distinction


/-- If both `a` and `b` commute with `c`, then their product commutes with `c`. -/
@[to_additive (attr := simp)
"If both `a` and `b` commute with `c`, then their product commutes with `c`."]
theorem mul_left (hac : Commute a c) (hbc : Commute b c) : Commute (a * b) c :=
  SemiconjBy.mul_left hac hbc
-- I think `ₓ` is necessary because of the `mul` vs `HMul` distinction


@[to_additive]
protected theorem right_comm (h : Commute b c) (a : S) : a * b * c = a * c * b := by
  /-
    S : Type u_3
    inst✝ : Semigroup S
    b c : S
    h : Commute b c
    a : S
    ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul (HMul.hMul a c) b)
  -/
  simp only [mul_assoc, h.eq]
  /-
    🎉 no goals
  -/
-- I think `ₓ` is necessary because of the `mul` vs `HMul` distinction


@[to_additive]
protected theorem left_comm (h : Commute a b) (c) : a * (b * c) = b * (a * c) := by
  /-
    S : Type u_3
    inst✝ : Semigroup S
    a b : S
    h : Commute a b
    c : S
    ⊢ Eq (HMul.hMul a (HMul.hMul b c)) (HMul.hMul b (HMul.hMul a c))
  -/
  simp only [← mul_assoc, h.eq]
  /-
    🎉 no goals
  -/
-- I think `ₓ` is necessary because of the `mul` vs `HMul` distinction


@[to_additive]
protected theorem mul_mul_mul_comm (hbc : Commute b c) (a d : S) :
                                            /-
                                              S : Type u_3
                                              inst✝ : Semigroup S
                                              b c : S
                                              hbc : Commute b c
                                              a d : S
                                              ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HMul.hMul a c) (H …
                                            -/
    a * b * (c * d) = a * c * (b * d) := by simp only [hbc.left_comm, mul_assoc]
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive]
protected theorem all [CommMagma S] (a b : S) : Commute a b :=
  mul_comm a b


@[to_additive (attr := simp)]
theorem one_right (a : M) : Commute a 1 :=
  SemiconjBy.one_right a
-- I think `ₓ` is necessary because `One.toOfNat1` appears in the Lean 4 version


@[to_additive (attr := simp)]
theorem one_left (a : M) : Commute 1 a :=
  SemiconjBy.one_left a
-- I think `ₓ` is necessary because `One.toOfNat1` appears in the Lean 4 version


@[to_additive (attr := simp)]
theorem pow_right (h : Commute a b) (n : ℕ) : Commute a (b ^ n) :=
  SemiconjBy.pow_right h n
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`


@[to_additive (attr := simp)]
theorem pow_left (h : Commute a b) (n : ℕ) : Commute (a ^ n) b :=
  (h.symm.pow_right n).symm
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`

-- todo: should nat power be called `nsmul` here?

@[to_additive (attr := simp)]
theorem pow_pow (h : Commute a b) (m n : ℕ) : Commute (a ^ m) (b ^ n) :=
  (h.pow_left m).pow_right n
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`


@[to_additive]
theorem self_pow (a : M) (n : ℕ) : Commute a (a ^ n) :=
  (Commute.refl a).pow_right n
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`


@[to_additive]
theorem pow_self (a : M) (n : ℕ) : Commute (a ^ n) a :=
  (Commute.refl a).pow_left n
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`


@[to_additive]
theorem pow_pow_self (a : M) (m n : ℕ) : Commute (a ^ m) (a ^ n) :=
  (Commute.refl a).pow_pow m n
-- `MulOneClass.toHasMul` vs. `MulOneClass.toMul`


@[to_additive] lemma mul_pow (h : Commute a b) : ∀ n, (a * b) ^ n = a ^ n * b ^ n
            /-
              M : Type u_2
              inst✝ : Monoid M
              a b : M
              h : Commute a b
              ⊢ Eq (HPow.hPow (HMul.hMul a b) 0) (HMul.hMul (HPow.hPow a 0) (HPow.hPow b 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, pow_zero, one_mul]
            /-
              🎉 no goals
            -/
                /-
                  M : Type u_2
                  inst✝ : Monoid M
                  a b : M
                  h : Commute a b
                  n : Nat
                  ⊢ Eq (HPow.hPow (HMul.hMul a b) (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow a (HAdd …
                -/
  | n + 1 => by simp only [pow_succ', h.mul_pow n, ← mul_assoc, (h.pow_left n).right_comm]
                /-
                  🎉 no goals
                -/


@[to_additive]
                                                                            /-
                                                                              G : Type u_1
                                                                              inst✝ : DivisionMonoid G
                                                                              a b : G
                                                                              hab : Commute a b
                                                                              ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (Inv.inv b))
                                                                            -/
protected theorem mul_inv (hab : Commute a b) : (a * b)⁻¹ = a⁻¹ * b⁻¹ := by rw [hab.eq, mul_inv_rev]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[to_additive]
                                                                        /-
                                                                          G : Type u_1
                                                                          inst✝ : DivisionMonoid G
                                                                          a b : G
                                                                          hab : Commute a b
                                                                          ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (Inv.inv b))
                                                                        -/
protected theorem inv (hab : Commute a b) : (a * b)⁻¹ = a⁻¹ * b⁻¹ := by rw [hab.eq, mul_inv_rev]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[to_additive AddCommute.zsmul_add]
protected lemma mul_zpow (h : Commute a b) : ∀ n : ℤ, (a * b) ^ n = a ^ n * b ^ n
                     /-
                       G : Type u_1
                       inst✝ : DivisionMonoid G
                       a b : G
                       h : Commute a b
                       n : Nat
                       ⊢ Eq (HPow.hPow (HMul.hMul a b) ↑n) (HMul.hMul (HPow.hPow a ↑n) (HPow.hPow b ↑ …
                     -/
  | (n : ℕ)    => by simp [zpow_natCast, h.mul_pow n]
                     /-
                       🎉 no goals
                     -/
                     /-
                       G : Type u_1
                       inst✝ : DivisionMonoid G
                       a b : G
                       h : Commute a b
                       n : Nat
                       ⊢ Eq (HPow.hPow (HMul.hMul a b) (Int.negSucc n)) (HMul.hMul (HPow.hPow a (Int. …
                     -/
  | .negSucc n => by simp [h.mul_pow, (h.pow_pow _ _).eq, mul_inv_rev]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
protected theorem mul_inv_cancel (h : Commute a b) : a * b * a⁻¹ = b := by
  /-
    G : Type u_1
    inst✝ : Group G
    a b : G
    h : Commute a b
    ⊢ Eq (HMul.hMul (HMul.hMul a b) (Inv.inv a)) b
  -/
  rw [h.eq, mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_inv_cancel_assoc (h : Commute a b) : a * (b * a⁻¹) = b := by
  /-
    G : Type u_1
    inst✝ : Group G
    a b : G
    h : Commute a b
    ⊢ Eq (HMul.hMul a (HMul.hMul b (Inv.inv a))) b
  -/
  rw [← mul_assoc, h.mul_inv_cancel]
  /-
    🎉 no goals
  -/


