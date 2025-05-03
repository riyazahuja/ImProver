@[to_additive (attr := simp)]
theorem inv_inv_symm_iff : SemiconjBy a⁻¹ x⁻¹ y⁻¹ ↔ SemiconjBy a y x := by
  /-
    G : Type u_1
    inst✝ : DivisionMonoid G
    a x y : G
    ⊢ Iff (SemiconjBy (Inv.inv a) (Inv.inv x) (Inv.inv y)) (SemiconjBy a y x)
  -/
  simp_rw [SemiconjBy, ← mul_inv_rev, inv_inj, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨_, inv_inv_symm⟩ := inv_inv_symm_iff


@[to_additive (attr := simp)] lemma inv_symm_left_iff : SemiconjBy a⁻¹ y x ↔ SemiconjBy a x y := by
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    ⊢ Iff (SemiconjBy (Inv.inv a) y x) (SemiconjBy a x y)
  -/
  simp_rw [SemiconjBy, eq_mul_inv_iff_mul_eq, mul_assoc, inv_mul_eq_iff_eq_mul, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨_, inv_symm_left⟩ := inv_symm_left_iff


@[to_additive (attr := simp)] lemma inv_right_iff : SemiconjBy a x⁻¹ y⁻¹ ↔ SemiconjBy a x y := by
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    ⊢ Iff (SemiconjBy a (Inv.inv x) (Inv.inv y)) (SemiconjBy a x y)
  -/
  rw [← inv_symm_left_iff, inv_inv_symm_iff]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨_, inv_right⟩ := inv_right_iff


@[to_additive (attr := simp)] lemma zpow_right (h : SemiconjBy a x y) :
    ∀ m : ℤ, SemiconjBy a (x ^ m) (y ^ m)
                     /-
                       G : Type u_1
                       inst✝ : Group G
                       a x y : G
                       h : SemiconjBy a x y
                       n : Nat
                       ⊢ SemiconjBy a (HPow.hPow x ↑n) (HPow.hPow y ↑n)
                     -/
  | (n : ℕ)    => by simp [zpow_natCast, h.pow_right n]
                     /-
                       🎉 no goals
                     -/
  | .negSucc n => by
    /-
      G : Type u_1
      inst✝ : Group G
      a x y : G
      h : SemiconjBy a x y
      n : Nat
      ⊢ SemiconjBy a (HPow.hPow x (Int.negSucc n)) (HPow.hPow y (Int.negSucc n))
    -/
    simp only [zpow_negSucc, inv_right_iff]
    /-
      G : Type u_1
      inst✝ : Group G
      a x y : G
      h : SemiconjBy a x y
      n : Nat
      ⊢ SemiconjBy a (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y (HAdd.hAdd n 1))
    -/
    apply pow_right h
    /-
      🎉 no goals
    -/


variable (a) in
@[to_additive] lemma eq_one_iff (h : SemiconjBy a x y): x = 1 ↔ y = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    h : SemiconjBy a x y
    ⊢ Iff (Eq x 1) (Eq y 1)
  -/
  rw [← conj_eq_one_iff (a := a) (b := x), h.eq, mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


