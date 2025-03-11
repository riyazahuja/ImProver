theorem eq_X_add_C_of_degree_le_one (h : degree p ≤ 1) : p = C (p.coeff 1) * X + C (p.coeff 0) :=
  ext fun n =>
                      /-
                        R : Type u
                        inst✝ : Semiring R
                        p : Polynomial R
                        h : LE.le p.degree 1
                        n : Nat
                        ⊢ Eq (p.coeff Nat.zero) ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Poly …
                      -/
    Nat.casesOn n (by simp) fun n =>
                      /-
                        🎉 no goals
                      -/
                        /-
                          R : Type u
                          inst✝ : Semiring R
                          p : Polynomial R
                          h : LE.le p.degree 1
                          n✝ n : Nat
                          ⊢ Eq (p.coeff Nat.zero.succ) ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) …
                        -/
      Nat.casesOn n (by simp [coeff_C]) fun m => by
                        /-
                          🎉 no goals
                        -/
        -- Porting note: `by decide` → `Iff.mpr ..`
        have : degree p < m.succ.succ := lt_of_le_of_lt h
          (Iff.mpr WithBot.coe_lt_coe <| Nat.succ_lt_succ <| Nat.zero_lt_succ m)
        simp [coeff_eq_zero_of_degree_lt this, coeff_C, Nat.succ_ne_zero, coeff_X, Nat.succ_inj',
          @eq_comm ℕ 0]


theorem eq_X_add_C_of_degree_eq_one (h : degree p = 1) :
    p = C p.leadingCoeff * X + C (p.coeff 0) :=
  (eq_X_add_C_of_degree_le_one h.le).trans
        /-
          R : Type u
          inst✝ : Semiring R
          p : Polynomial R
          h : Eq p.degree 1
          ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomia …
        -/
    (by rw [← Nat.cast_one] at h; rw [leadingCoeff, natDegree_eq_of_degree_eq_some h])
                                  /-
                                    🎉 no goals
                                  -/


theorem eq_X_add_C_of_natDegree_le_one (h : natDegree p ≤ 1) :
    p = C (p.coeff 1) * X + C (p.coeff 0) :=
  eq_X_add_C_of_degree_le_one <| degree_le_of_natDegree_le h


theorem Monic.eq_X_add_C (hm : p.Monic) (hnd : p.natDegree = 1) : p = X + C (p.coeff 0) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hnd : Eq p.natDegree 1
    ⊢ Eq p (HAdd.hAdd Polynomial.X (Polynomial.C (p.coeff 0)))
  -/
  rw [← one_mul X, ← C_1, ← hm.coeff_natDegree, hnd, ← eq_X_add_C_of_natDegree_le_one hnd.le]
  /-
    🎉 no goals
  -/


theorem exists_eq_X_add_C_of_natDegree_le_one (h : natDegree p ≤ 1) : ∃ a b, p = C a * X + C b :=
  ⟨p.coeff 1, p.coeff 0, eq_X_add_C_of_natDegree_le_one h⟩


theorem zero_le_degree_iff : 0 ≤ degree p ↔ p ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (LE.le 0 p.degree) (Ne p 0)
  -/
  rw [← not_lt, Nat.WithBot.lt_zero_iff, degree_eq_bot]
  /-
    🎉 no goals
  -/


theorem ne_zero_of_coe_le_degree (hdeg : ↑n ≤ p.degree) : p ≠ 0 :=
  zero_le_degree_iff.mp <| (WithBot.coe_le_coe.mpr n.zero_le).trans hdeg


theorem le_natDegree_of_coe_le_degree (hdeg : ↑n ≤ p.degree) : n ≤ p.natDegree :=
  -- Porting note: `.. ▸ ..` → `rwa [..] at ..`
  WithBot.coe_le_coe.mp <| by
    /-
      R : Type u
      n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      hdeg : LE.le (↑n) p.degree
      ⊢ LE.le ↑n ↑p.natDegree
    -/
    rwa [degree_eq_natDegree <| ne_zero_of_coe_le_degree hdeg] at hdeg
    /-
      🎉 no goals
    -/


theorem degree_linear_le : degree (C a * X + C b) ≤ 1 :=
  degree_add_le_of_degree_le (degree_C_mul_X_le _) <| le_trans degree_C_le Nat.WithBot.coe_nonneg


theorem degree_linear_lt : degree (C a * X + C b) < 2 :=
  degree_linear_le.trans_lt <| WithBot.coe_lt_coe.mpr one_lt_two


@[simp]
theorem degree_linear (ha : a ≠ 0) : degree (C a * X + C b) = 1 := by
  /-
    R : Type u
    a b : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).de …
  -/
  rw [degree_add_eq_left_of_degree_lt <| degree_C_lt_degree_C_mul_X ha, degree_C_mul_X ha]
  /-
    🎉 no goals
  -/


theorem natDegree_linear_le : natDegree (C a * X + C b) ≤ 1 :=
  natDegree_le_of_degree_le degree_linear_le


theorem natDegree_linear (ha : a ≠ 0) : natDegree (C a * X + C b) = 1 := by
  /-
    R : Type u
    a b : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).na …
  -/
  rw [natDegree_add_C, natDegree_C_mul_X a ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_linear (ha : a ≠ 0) : leadingCoeff (C a * X + C b) = a := by
  rw [add_comm, leadingCoeff_add_of_degree_lt (degree_C_lt_degree_C_mul_X ha),
    leadingCoeff_C_mul_X]


theorem degree_quadratic_le : degree (C a * X ^ 2 + C b * X + C c) ≤ 2 := by
  simpa only [add_assoc] using
    degree_add_le_of_degree_le (degree_C_mul_X_pow_le 2 a)
      (le_trans degree_linear_le <| WithBot.coe_le_coe.mpr one_le_two)


theorem degree_quadratic_lt : degree (C a * X ^ 2 + C b * X + C c) < 3 :=
  degree_quadratic_le.trans_lt <| WithBot.coe_lt_coe.mpr <| lt_add_one 2


theorem degree_linear_lt_degree_C_mul_X_sq (ha : a ≠ 0) :
    degree (C b * X + C c) < degree (C a * X ^ 2) := by
  /-
    R : Type u
    a b c : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (Polynomial.C b) Polynomial.X) (Polynomial.C c)) …
  -/
  simpa only [degree_C_mul_X_pow 2 ha] using degree_linear_lt
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_quadratic (ha : a ≠ 0) : degree (C a * X ^ 2 + C b * X + C c) = 2 := by
  rw [add_assoc, degree_add_eq_left_of_degree_lt <| degree_linear_lt_degree_C_mul_X_sq ha,
    degree_C_mul_X_pow 2 ha]
  /-
    R : Type u
    a b c : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (↑2) 2
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem natDegree_quadratic_le : natDegree (C a * X ^ 2 + C b * X + C c) ≤ 2 :=
  natDegree_le_of_degree_le degree_quadratic_le


theorem natDegree_quadratic (ha : a ≠ 0) : natDegree (C a * X ^ 2 + C b * X + C c) = 2 :=
  natDegree_eq_of_degree_eq_some <| degree_quadratic ha


@[simp]
theorem leadingCoeff_quadratic (ha : a ≠ 0) : leadingCoeff (C a * X ^ 2 + C b * X + C c) = a := by
  rw [add_assoc, add_comm, leadingCoeff_add_of_degree_lt <| degree_linear_lt_degree_C_mul_X_sq ha,
    leadingCoeff_C_mul_X_pow]


theorem degree_cubic_le : degree (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) ≤ 3 := by
  simpa only [add_assoc] using
    degree_add_le_of_degree_le (degree_C_mul_X_pow_le 3 a)
      (le_trans degree_quadratic_le <| WithBot.coe_le_coe.mpr <| Nat.le_succ 2)


theorem degree_cubic_lt : degree (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) < 4 :=
  degree_cubic_le.trans_lt <| WithBot.coe_lt_coe.mpr <| lt_add_one 3


theorem degree_quadratic_lt_degree_C_mul_X_cb (ha : a ≠ 0) :
    degree (C b * X ^ 2 + C c * X + C d) < degree (C a * X ^ 3) := by
  /-
    R : Type u
    a b c d : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C b) (HPow.hPow Polynomia …
  -/
  simpa only [degree_C_mul_X_pow 3 ha] using degree_quadratic_lt
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_cubic (ha : a ≠ 0) : degree (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) = 3 := by
  rw [add_assoc, add_assoc, ← add_assoc (C b * X ^ 2),
    degree_add_eq_left_of_degree_lt <| degree_quadratic_lt_degree_C_mul_X_cb ha,
    degree_C_mul_X_pow 3 ha]
  /-
    R : Type u
    a b c d : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (↑3) 3
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem natDegree_cubic_le : natDegree (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) ≤ 3 :=
  natDegree_le_of_degree_le degree_cubic_le


theorem natDegree_cubic (ha : a ≠ 0) : natDegree (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) = 3 :=
  natDegree_eq_of_degree_eq_some <| degree_cubic ha


@[simp]
theorem leadingCoeff_cubic (ha : a ≠ 0) :
    leadingCoeff (C a * X ^ 3 + C b * X ^ 2 + C c * X + C d) = a := by
  rw [add_assoc, add_assoc, ← add_assoc (C b * X ^ 2), add_comm,
    leadingCoeff_add_of_degree_lt <| degree_quadratic_lt_degree_C_mul_X_cb ha,
    leadingCoeff_C_mul_X_pow]


