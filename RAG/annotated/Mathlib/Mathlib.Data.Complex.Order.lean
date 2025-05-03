/-- We put a partial order on ℂ so that `z ≤ w` exactly if `w - z` is real and nonnegative.
Complex numbers with different imaginary parts are incomparable.
-/
protected def partialOrder : PartialOrder ℂ where
  le z w := z.re ≤ w.re ∧ z.im = w.im
  lt z w := z.re < w.re ∧ z.im = w.im
  lt_iff_le_not_le z w := by
    /-
      z w : Complex
      ⊢ Iff (LT.lt z w) (And (LE.le z w) (Not (LE.le w z)))
    -/
    dsimp
    /-
      z w : Complex
      ⊢ Iff (And (LT.lt z.re w.re) (Eq z.im w.im)) (And (And (LE.le z.re w.re) (Eq z …
    -/
    rw [lt_iff_le_not_le]
    /-
      z w : Complex
      ⊢ Iff (And (And (LE.le z.re w.re) (Not (LE.le w.re z.re))) (Eq z.im w.im)) (An …
    -/
    tauto
    /-
      🎉 no goals
    -/
  le_refl _ := ⟨le_rfl, rfl⟩
  le_trans _ _ _ h₁ h₂ := ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩
  le_antisymm _ _ h₁ h₂ := ext (h₁.1.antisymm h₂.1) h₁.2


theorem le_def {z w : ℂ} : z ≤ w ↔ z.re ≤ w.re ∧ z.im = w.im :=
  Iff.rfl


theorem lt_def {z w : ℂ} : z < w ↔ z.re < w.re ∧ z.im = w.im :=
  Iff.rfl


theorem nonneg_iff {z : ℂ} : 0 ≤ z ↔ 0 ≤ z.re ∧ 0 = z.im :=
  le_def


theorem pos_iff {z : ℂ} : 0 < z ↔ 0 < z.re ∧ 0 = z.im :=
  lt_def


theorem nonpos_iff {z : ℂ} : z ≤ 0 ↔ z.re ≤ 0 ∧ z.im = 0 :=
  le_def


theorem neg_iff {z : ℂ} : z < 0 ↔ z.re < 0 ∧ z.im = 0 :=
  lt_def


@[simp, norm_cast]
                                                                 /-
                                                                   x y : Real
                                                                   ⊢ Iff (LE.le ↑x ↑y) (LE.le x y)
                                                                 -/
theorem real_le_real {x y : ℝ} : (x : ℂ) ≤ (y : ℂ) ↔ x ≤ y := by simp [le_def, ofReal]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp, norm_cast]
                                                                 /-
                                                                   x y : Real
                                                                   ⊢ Iff (LT.lt ↑x ↑y) (LT.lt x y)
                                                                 -/
theorem real_lt_real {x y : ℝ} : (x : ℂ) < (y : ℂ) ↔ x < y := by simp [lt_def, ofReal]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp, norm_cast]
theorem zero_le_real {x : ℝ} : (0 : ℂ) ≤ (x : ℂ) ↔ 0 ≤ x :=
  real_le_real


@[simp, norm_cast]
theorem zero_lt_real {x : ℝ} : (0 : ℂ) < (x : ℂ) ↔ 0 < x :=
  real_lt_real


theorem not_le_iff {z w : ℂ} : ¬z ≤ w ↔ w.re < z.re ∨ z.im ≠ w.im := by
  /-
    z w : Complex
    ⊢ Iff (Not (LE.le z w)) (Or (LT.lt w.re z.re) (Ne z.im w.im))
  -/
  rw [le_def, not_and_or, not_le]
  /-
    🎉 no goals
  -/


theorem not_lt_iff {z w : ℂ} : ¬z < w ↔ w.re ≤ z.re ∨ z.im ≠ w.im := by
  /-
    z w : Complex
    ⊢ Iff (Not (LT.lt z w)) (Or (LE.le w.re z.re) (Ne z.im w.im))
  -/
  rw [lt_def, not_and_or, not_lt]
  /-
    🎉 no goals
  -/


theorem not_le_zero_iff {z : ℂ} : ¬z ≤ 0 ↔ 0 < z.re ∨ z.im ≠ 0 :=
  not_le_iff


theorem not_lt_zero_iff {z : ℂ} : ¬z < 0 ↔ 0 ≤ z.re ∨ z.im ≠ 0 :=
  not_lt_iff


theorem eq_re_of_ofReal_le {r : ℝ} {z : ℂ} (hz : (r : ℂ) ≤ z) : z = z.re := by
  /-
    r : Real
    z : Complex
    hz : LE.le (↑r) z
    ⊢ Eq z ↑z.re
  -/
  rw [eq_comm, ← conj_eq_iff_re, conj_eq_iff_im, ← (Complex.le_def.1 hz).2, Complex.ofReal_im]
  /-
    🎉 no goals
  -/


@[simp]
lemma re_eq_abs {z : ℂ} : z.re = abs z ↔ 0 ≤ z :=
  have : 0 ≤ abs z := apply_nonneg abs z
  ⟨fun h ↦ ⟨h.symm ▸ this, (abs_re_eq_abs.1 <| h.symm ▸ _root_.abs_of_nonneg this).symm⟩,
                      /-
                        z : Complex
                        this : LE.le 0 (Complex.abs z)
                        x✝ : LE.le 0 z
                        h₁ : LE.le (Complex.re 0) z.re
                        h₂ : Eq (Complex.im 0) z.im
                        ⊢ Eq z.re (Complex.abs z)
                      -/
    fun ⟨h₁, h₂⟩ ↦ by rw [← abs_re_eq_abs.2 h₂.symm, _root_.abs_of_nonneg h₁]⟩
                      /-
                        🎉 no goals
                      -/


@[simp]
lemma neg_re_eq_abs {z : ℂ} : -z.re = abs z ↔ z ≤ 0 := by
  /-
    z : Complex
    ⊢ Iff (Eq (Neg.neg z.re) (Complex.abs z)) (LE.le z 0)
  -/
  rw [← neg_re, ← abs.map_neg, re_eq_abs]
  /-
    z : Complex
    ⊢ Iff (LE.le 0 (Neg.neg z)) (LE.le z 0)
  -/
  exact neg_nonneg.and <| eq_comm.trans neg_eq_zero
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            z : Complex
                                                            ⊢ Iff (Eq z.re (Neg.neg (Complex.abs z))) (LE.le z 0)
                                                          -/
lemma re_eq_neg_abs {z : ℂ} : z.re = -abs z ↔ z ≤ 0 := by rw [← neg_eq_iff_eq_neg, neg_re_eq_abs]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma monotone_ofReal : Monotone ofReal := by
  /-
    ⊢ Monotone Complex.ofReal
  -/
  intro x y hxy
  /-
    x y : Real
    hxy : LE.le x y
    ⊢ LE.le ↑x ↑y
  -/
  simp only [ofRealHom_eq_coe, real_le_real, hxy]
  /-
    🎉 no goals
  -/


private alias ⟨_, ofReal_pos⟩ := zero_lt_real

private alias ⟨_, ofReal_nonneg⟩ := zero_le_real

private alias ⟨_, ofReal_ne_zero_of_ne_zero⟩ := ofReal_ne_zero


/-- Extension for the `positivity` tactic: `Complex.ofReal` is positive/nonnegative/nonzero if its
input is. -/
@[positivity Complex.ofReal _, Complex.ofReal _]
def evalComplexOfReal : PositivityExt where eval {u α} _ _ e := do
  -- TODO: Can we avoid duplicating the code?
  match u, α, e with
  | 0, ~q(ℂ), ~q(Complex.ofReal $a) =>
    assumeInstancesCommute
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa => return .positive q(ofReal_pos $pa)
    | .nonnegative pa => return .nonnegative q(ofReal_nonneg $pa)
    | .nonzero pa => return .nonzero q(ofReal_ne_zero_of_ne_zero $pa)
    | _ => return .none
  | 0, ~q(ℂ), ~q(Complex.ofReal $a) =>
    assumeInstancesCommute
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa => return .positive q(ofReal_pos $pa)
    | .nonnegative pa => return .nonnegative q(ofReal_nonneg $pa)
    | .nonzero pa => return .nonzero q(ofReal_ne_zero_of_ne_zero $pa)
    | _ => return .none
  | _, _ => throwError "not Complex.ofReal"


