/-- If `a` semiconjugates a unit `x` to a unit `y`, then it semiconjugates `x⁻¹` to `y⁻¹`. -/
@[to_additive "If `a` semiconjugates an additive unit `x` to an additive unit `y`, then it
semiconjugates `-x` to `-y`."]
theorem units_inv_right {a : M} {x y : Mˣ} (h : SemiconjBy a x y) : SemiconjBy a ↑x⁻¹ ↑y⁻¹ :=
  calc
                                           /-
                                             M : Type u_1
                                             inst✝ : Monoid M
                                             a : M
                                             x y : Units M
                                             h : SemiconjBy a ↑x ↑y
                                             ⊢ Eq (HMul.hMul a ↑(Inv.inv x)) (HMul.hMul (HMul.hMul (↑(Inv.inv y)) (HMul.hMu …
                                           -/
    a * ↑x⁻¹ = ↑y⁻¹ * (y * a) * ↑x⁻¹ := by rw [Units.inv_mul_cancel_left]
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             M : Type u_1
                                             inst✝ : Monoid M
                                             a : M
                                             x y : Units M
                                             h : SemiconjBy a ↑x ↑y
                                             ⊢ Eq (HMul.hMul (HMul.hMul (↑(Inv.inv y)) (HMul.hMul (↑y) a)) ↑(Inv.inv x)) (H …
                                           -/
    _        = ↑y⁻¹ * a              := by rw [← h.eq, mul_assoc, Units.mul_inv_cancel_right]
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive (attr := simp)]
theorem units_inv_right_iff {a : M} {x y : Mˣ} : SemiconjBy a ↑x⁻¹ ↑y⁻¹ ↔ SemiconjBy a x y :=
  ⟨units_inv_right, units_inv_right⟩


/-- If a unit `a` semiconjugates `x` to `y`, then `a⁻¹` semiconjugates `y` to `x`. -/
@[to_additive "If an additive unit `a` semiconjugates `x` to `y`, then `-a` semiconjugates `y` to
`x`."]
theorem units_inv_symm_left {a : Mˣ} {x y : M} (h : SemiconjBy (↑a) x y) : SemiconjBy (↑a⁻¹) y x :=
  calc
                                           /-
                                             M : Type u_1
                                             inst✝ : Monoid M
                                             a : Units M
                                             x y : M
                                             h : SemiconjBy (↑a) x y
                                             ⊢ Eq (HMul.hMul (↑(Inv.inv a)) y) (HMul.hMul (↑(Inv.inv a)) (HMul.hMul (HMul.h …
                                           -/
    ↑a⁻¹ * y = ↑a⁻¹ * (y * a * ↑a⁻¹) := by rw [Units.mul_inv_cancel_right]
                                           /-
                                             🎉 no goals
                                           -/
                       /-
                         M : Type u_1
                         inst✝ : Monoid M
                         a : Units M
                         x y : M
                         h : SemiconjBy (↑a) x y
                         ⊢ Eq (HMul.hMul (↑(Inv.inv a)) (HMul.hMul (HMul.hMul y ↑a) ↑(Inv.inv a))) (HMu …
                       -/
    _ = x * ↑a⁻¹ := by rw [← h.eq, ← mul_assoc, Units.inv_mul_cancel_left]
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
theorem units_inv_symm_left_iff {a : Mˣ} {x y : M} : SemiconjBy (↑a⁻¹) y x ↔ SemiconjBy (↑a) x y :=
  ⟨units_inv_symm_left, units_inv_symm_left⟩


@[to_additive]
theorem units_val {a x y : Mˣ} (h : SemiconjBy a x y) : SemiconjBy (a : M) x y :=
  congr_arg Units.val h


@[to_additive]
theorem units_of_val {a x y : Mˣ} (h : SemiconjBy (a : M) x y) : SemiconjBy a x y :=
  Units.ext h


@[to_additive (attr := simp)]
theorem units_val_iff {a x y : Mˣ} : SemiconjBy (a : M) x y ↔ SemiconjBy a x y :=
  ⟨units_of_val, units_val⟩


@[to_additive (attr := simp)]
lemma units_zpow_right {a : M} {x y : Mˣ} (h : SemiconjBy a x y) :
    ∀ m : ℤ, SemiconjBy a ↑(x ^ m) ↑(y ^ m)
                  /-
                    M : Type u_1
                    inst✝ : Monoid M
                    a : M
                    x y : Units M
                    h : SemiconjBy a ↑x ↑y
                    n : Nat
                    ⊢ SemiconjBy a ↑(HPow.hPow x ↑n) ↑(HPow.hPow y ↑n)
                  -/
  | (n : ℕ) => by simp only [zpow_natCast, Units.val_pow_eq_pow_val, h, pow_right]
                  /-
                    🎉 no goals
                  -/
                 /-
                   M : Type u_1
                   inst✝ : Monoid M
                   a : M
                   x y : Units M
                   h : SemiconjBy a ↑x ↑y
                   n : Nat
                   ⊢ SemiconjBy a ↑(HPow.hPow x (Int.negSucc n)) ↑(HPow.hPow y (Int.negSucc n))
                 -/
  | -[n+1] => by simp only [zpow_negSucc, Units.val_pow_eq_pow_val, units_inv_right, h, pow_right]
                 /-
                   🎉 no goals
                 -/


/-- `a` semiconjugates `x` to `a * x * a⁻¹`. -/
@[to_additive "`a` semiconjugates `x` to `a + x + -a`."]
lemma mk_semiconjBy (u : Mˣ) (x : M) : SemiconjBy (↑u) x (u * x * ↑u⁻¹) := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    u : Units M
    x : M
    ⊢ SemiconjBy (↑u) x (HMul.hMul (HMul.hMul (↑u) x) ↑(Inv.inv u))
  -/
  unfold SemiconjBy; rw [Units.inv_mul_cancel_right]
                     /-
                       🎉 no goals
                     -/


lemma conj_pow (u : Mˣ) (x : M) (n : ℕ) :
    ((↑u : M) * x * (↑u⁻¹ : M)) ^ n = (u : M) * x ^ n * (↑u⁻¹ : M) :=
  eq_divp_iff_mul_eq.2 ((u.mk_semiconjBy x).pow_right n).eq.symm


lemma conj_pow' (u : Mˣ) (x : M) (n : ℕ) :
    ((↑u⁻¹ : M) * x * (u : M)) ^ n = (↑u⁻¹ : M) * x ^ n * (u : M) := u⁻¹.conj_pow x n


