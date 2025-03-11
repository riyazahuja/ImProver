/-- Square root of a nonnegative real number. -/
-- Porting note (kmill): `pp_nodot` has no affect here
-- unless RFC https://github.com/leanprover/lean4/issues/6178 leads to dot notation pp for CoeFun
@[pp_nodot]
noncomputable def sqrt : ℝ≥0 ≃o ℝ≥0 :=
  OrderIso.symm <| powOrderIso 2 two_ne_zero


@[simp] lemma sq_sqrt (x : ℝ≥0) : sqrt x ^ 2 = x := sqrt.symm_apply_apply _


@[simp] lemma sqrt_sq (x : ℝ≥0) : sqrt (x ^ 2) = x := sqrt.apply_symm_apply _


                                                                  /-
                                                                    x : NNReal
                                                                    ⊢ Eq (HMul.hMul (NNReal.sqrt x) (NNReal.sqrt x)) x
                                                                  -/
@[simp] lemma mul_self_sqrt (x : ℝ≥0) : sqrt x * sqrt x = x := by rw [← sq, sq_sqrt]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                               /-
                                                                 x : NNReal
                                                                 ⊢ Eq (NNReal.sqrt (HMul.hMul x x)) x
                                                               -/
@[simp] lemma sqrt_mul_self (x : ℝ≥0) : sqrt (x * x) = x := by rw [← sq, sqrt_sq]
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma sqrt_le_sqrt : sqrt x ≤ sqrt y ↔ x ≤ y := sqrt.le_iff_le


lemma sqrt_lt_sqrt : sqrt x < sqrt y ↔ x < y := sqrt.lt_iff_lt


lemma sqrt_eq_iff_eq_sq : sqrt x = y ↔ x = y ^ 2 := sqrt.toEquiv.apply_eq_iff_eq_symm_apply


lemma sqrt_le_iff_le_sq : sqrt x ≤ y ↔ x ≤ y ^ 2 := sqrt.to_galoisConnection _ _


lemma le_sqrt_iff_sq_le : x ≤ sqrt y ↔ x ^ 2 ≤ y := (sqrt.symm.to_galoisConnection _ _).symm


@[deprecated (since := "2024-02-14")] alias sqrt_le_sqrt_iff := sqrt_le_sqrt

@[deprecated (since := "2024-02-14")] alias sqrt_lt_sqrt_iff := sqrt_lt_sqrt

@[deprecated (since := "2024-02-14")] alias sqrt_le_iff := sqrt_le_iff_le_sq

@[deprecated (since := "2024-02-14")] alias le_sqrt_iff := le_sqrt_iff_sq_le

@[deprecated (since := "2024-02-14")] alias sqrt_eq_iff_sq_eq := sqrt_eq_iff_eq_sq


                                                      /-
                                                        x : NNReal
                                                        ⊢ Iff (Eq (NNReal.sqrt x) 0) (Eq x 0)
                                                      -/
@[simp] lemma sqrt_eq_zero : sqrt x = 0 ↔ x = 0 := by simp [sqrt_eq_iff_eq_sq]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                     /-
                                                       x : NNReal
                                                       ⊢ Iff (Eq (NNReal.sqrt x) 1) (Eq x 1)
                                                     -/
@[simp] lemma sqrt_eq_one : sqrt x = 1 ↔ x = 1 := by simp [sqrt_eq_iff_eq_sq]
                                                     /-
                                                       🎉 no goals
                                                     -/


                                           /-
                                             ⊢ Eq (NNReal.sqrt 0) 0
                                           -/
@[simp] lemma sqrt_zero : sqrt 0 = 0 := by simp
                                           /-
                                             🎉 no goals
                                           -/


                                          /-
                                            ⊢ Eq (NNReal.sqrt 1) 1
                                          -/
@[simp] lemma sqrt_one : sqrt 1 = 1 := by simp
                                          /-
                                            🎉 no goals
                                          -/


                                                     /-
                                                       x : NNReal
                                                       ⊢ Iff (LE.le (NNReal.sqrt x) 1) (LE.le x 1)
                                                     -/
@[simp] lemma sqrt_le_one : sqrt x ≤ 1 ↔ x ≤ 1 := by rw [← sqrt_one, sqrt_le_sqrt, sqrt_one]
                                                     /-
                                                       🎉 no goals
                                                     -/

                                                     /-
                                                       x : NNReal
                                                       ⊢ Iff (LE.le 1 (NNReal.sqrt x)) (LE.le 1 x)
                                                     -/
@[simp] lemma one_le_sqrt : 1 ≤ sqrt x ↔ 1 ≤ x := by rw [← sqrt_one, sqrt_le_sqrt, sqrt_one]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem sqrt_mul (x y : ℝ≥0) : sqrt (x * y) = sqrt x * sqrt y := by
  /-
    x y : NNReal
    ⊢ Eq (NNReal.sqrt (HMul.hMul x y)) (HMul.hMul (NNReal.sqrt x) (NNReal.sqrt y))
  -/
  rw [sqrt_eq_iff_eq_sq, mul_pow, sq_sqrt, sq_sqrt]
  /-
    🎉 no goals
  -/


/-- `NNReal.sqrt` as a `MonoidWithZeroHom`. -/
noncomputable def sqrtHom : ℝ≥0 →*₀ ℝ≥0 :=
  ⟨⟨sqrt, sqrt_zero⟩, sqrt_one, sqrt_mul⟩


theorem sqrt_inv (x : ℝ≥0) : sqrt x⁻¹ = (sqrt x)⁻¹ :=
  map_inv₀ sqrtHom x


theorem sqrt_div (x y : ℝ≥0) : sqrt (x / y) = sqrt x / sqrt y :=
  map_div₀ sqrtHom x y


@[continuity, fun_prop]
theorem continuous_sqrt : Continuous sqrt := sqrt.continuous


                                                    /-
                                                      x : NNReal
                                                      ⊢ Iff (LT.lt 0 (NNReal.sqrt x)) (LT.lt 0 x)
                                                    -/
@[simp] theorem sqrt_pos : 0 < sqrt x ↔ 0 < x := by simp [pos_iff_ne_zero]
                                                    /-
                                                      🎉 no goals
                                                    -/


alias ⟨_, sqrt_pos_of_pos⟩ := sqrt_pos


/-- The square root of a real number. This returns 0 for negative inputs.

This has notation `√x`. Note that `√x⁻¹` is parsed as `√(x⁻¹)`. -/
noncomputable def sqrt (x : ℝ) : ℝ :=
  NNReal.sqrt (Real.toNNReal x)

-- TODO: replace this with a typeclass

@[inherit_doc]
prefix:max "√" => Real.sqrt


@[simp, norm_cast]
theorem coe_sqrt {x : ℝ≥0} : (NNReal.sqrt x : ℝ) = √(x : ℝ) := by
  /-
    x : NNReal
    ⊢ Eq (↑(NNReal.sqrt x)) (↑x).sqrt
  -/
  rw [Real.sqrt, Real.toNNReal_coe]
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_sqrt : Continuous (√· : ℝ → ℝ) :=
  NNReal.continuous_coe.comp <| NNReal.continuous_sqrt.comp continuous_real_toNNReal


                                                              /-
                                                                x : Real
                                                                h : LE.le x 0
                                                                ⊢ Eq x.sqrt 0
                                                              -/
theorem sqrt_eq_zero_of_nonpos (h : x ≤ 0) : sqrt x = 0 := by simp [sqrt, Real.toNNReal_eq_zero.2 h]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] theorem sqrt_nonneg (x : ℝ) : 0 ≤ √x := NNReal.coe_nonneg _


@[simp]
theorem mul_self_sqrt (h : 0 ≤ x) : √x * √x = x := by
  /-
    x : Real
    h : LE.le 0 x
    ⊢ Eq (HMul.hMul x.sqrt x.sqrt) x
  -/
  rw [Real.sqrt, ← NNReal.coe_mul, NNReal.mul_self_sqrt, Real.coe_toNNReal _ h]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_mul_self (h : 0 ≤ x) : √(x * x) = x :=
  (mul_self_inj_of_nonneg (sqrt_nonneg _) h).1 (mul_self_sqrt (mul_self_nonneg _))


theorem sqrt_eq_cases : √x = y ↔ y * y = x ∧ 0 ≤ y ∨ x < 0 ∧ y = 0 := by
  /-
    x y : Real
    ⊢ Iff (Eq x.sqrt y) (Or (And (Eq (HMul.hMul y y) x) (LE.le 0 y)) (And (LT.lt x …
  -/
  constructor
    /-
      case mp
      x y : Real
      ⊢ Eq x.sqrt y → Or (And (Eq (HMul.hMul y y) x) (LE.le 0 y)) (And (LT.lt x 0) ( …
    -/
  · rintro rfl
    /-
      case mp
      x : Real
      ⊢ Or (And (Eq (HMul.hMul x.sqrt x.sqrt) x) (LE.le 0 x.sqrt)) (And (LT.lt x 0)  …
    -/
    rcases le_or_lt 0 x with hle | hlt
      /-
        case mp.inl
        x : Real
        hle : LE.le 0 x
        ⊢ Or (And (Eq (HMul.hMul x.sqrt x.sqrt) x) (LE.le 0 x.sqrt)) (And (LT.lt x 0)  …
      -/
    · exact Or.inl ⟨mul_self_sqrt hle, sqrt_nonneg x⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        x : Real
        hlt : LT.lt x 0
        ⊢ Or (And (Eq (HMul.hMul x.sqrt x.sqrt) x) (LE.le 0 x.sqrt)) (And (LT.lt x 0)  …
      -/
    · exact Or.inr ⟨hlt, sqrt_eq_zero_of_nonpos hlt.le⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x y : Real
      ⊢ Or (And (Eq (HMul.hMul y y) x) (LE.le 0 y)) (And (LT.lt x 0) (Eq y 0)) → Eq  …
    -/
  · rintro (⟨rfl, hy⟩ | ⟨hx, rfl⟩)
    /-
      case mpr.inl.intro
      y : Real
      hy : LE.le 0 y
      ⊢ Eq (HMul.hMul y y).sqrt y
    -/
    exacts [sqrt_mul_self hy, sqrt_eq_zero_of_nonpos hx.le]
    /-
      🎉 no goals
    -/


theorem sqrt_eq_iff_mul_self_eq (hx : 0 ≤ x) (hy : 0 ≤ y) : √x = y ↔ x = y * y :=
               /-
                 x y : Real
                 hx : LE.le 0 x
                 hy : LE.le 0 y
                 h : Eq x.sqrt y
                 ⊢ Eq x (HMul.hMul y y)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [← h, mul_self_sqrt hx], fun h => by rw [h, sqrt_mul_self hy]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated sqrt_eq_iff_mul_self_eq (since := "2024-08-25")]
theorem sqrt_eq_iff_eq_mul_self (hx : 0 ≤ x) (hy : 0 ≤ y) : √x = y ↔ y * y = x := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Iff (Eq x.sqrt y) (Eq (HMul.hMul y y) x)
  -/
  rw [sqrt_eq_iff_mul_self_eq hx hy, eq_comm]
  /-
    🎉 no goals
  -/


theorem sqrt_eq_iff_mul_self_eq_of_pos (h : 0 < y) : √x = y ↔ y * y = x := by
  /-
    x y : Real
    h : LT.lt 0 y
    ⊢ Iff (Eq x.sqrt y) (Eq (HMul.hMul y y) x)
  -/
  simp [sqrt_eq_cases, h.ne', h.le]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_eq_one : √x = 1 ↔ x = 1 :=
  calc
    √x = 1 ↔ 1 * 1 = x := sqrt_eq_iff_mul_self_eq_of_pos zero_lt_one
                    /-
                      x : Real
                      ⊢ Iff (Eq (HMul.hMul 1 1) x) (Eq x 1)
                    -/
    _ ↔ x = 1 := by rw [eq_comm, mul_one]
                    /-
                      🎉 no goals
                    -/


@[simp]
                                               /-
                                                 x : Real
                                                 h : LE.le 0 x
                                                 ⊢ Eq (HPow.hPow x.sqrt 2) x
                                               -/
theorem sq_sqrt (h : 0 ≤ x) : √x ^ 2 = x := by rw [sq, mul_self_sqrt h]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                 /-
                                                   x : Real
                                                   h : LE.le 0 x
                                                   ⊢ Eq (HPow.hPow x 2).sqrt x
                                                 -/
theorem sqrt_sq (h : 0 ≤ x) : √(x ^ 2) = x := by rw [sq, sqrt_mul_self h]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem sqrt_eq_iff_eq_sq (hx : 0 ≤ x) (hy : 0 ≤ y) : √x = y ↔ x = y ^ 2 := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Iff (Eq x.sqrt y) (Eq x (HPow.hPow y 2))
  -/
  rw [sq, sqrt_eq_iff_mul_self_eq hx hy]
  /-
    🎉 no goals
  -/


@[deprecated sqrt_eq_iff_eq_sq (since := "2024-08-25")]
theorem sqrt_eq_iff_sq_eq (hx : 0 ≤ x) (hy : 0 ≤ y) : √x = y ↔ y ^ 2 = x := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Iff (Eq x.sqrt y) (Eq (HPow.hPow y 2) x)
  -/
  rw [sqrt_eq_iff_eq_sq hx hy, eq_comm]
  /-
    🎉 no goals
  -/


theorem sqrt_mul_self_eq_abs (x : ℝ) : √(x * x) = |x| := by
  /-
    x : Real
    ⊢ Eq (HMul.hMul x x).sqrt (abs x)
  -/
  rw [← abs_mul_abs_self x, sqrt_mul_self (abs_nonneg _)]
  /-
    🎉 no goals
  -/


                                                      /-
                                                        x : Real
                                                        ⊢ Eq (HPow.hPow x 2).sqrt (abs x)
                                                      -/
theorem sqrt_sq_eq_abs (x : ℝ) : √(x ^ 2) = |x| := by rw [sq, sqrt_mul_self_eq_abs]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                 /-
                                   ⊢ Eq (Real.sqrt 0) 0
                                 -/
theorem sqrt_zero : √0 = 0 := by simp [Real.sqrt]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
                                /-
                                  ⊢ Eq (Real.sqrt 1) 1
                                -/
theorem sqrt_one : √1 = 1 := by simp [Real.sqrt]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem sqrt_le_sqrt_iff (hy : 0 ≤ y) : √x ≤ √y ↔ x ≤ y := by
  /-
    x y : Real
    hy : LE.le 0 y
    ⊢ Iff (LE.le x.sqrt y.sqrt) (LE.le x y)
  -/
  rw [Real.sqrt, Real.sqrt, NNReal.coe_le_coe, NNReal.sqrt_le_sqrt, toNNReal_le_toNNReal_iff hy]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_lt_sqrt_iff (hx : 0 ≤ x) : √x < √y ↔ x < y :=
  lt_iff_lt_of_le_iff_le (sqrt_le_sqrt_iff hx)


theorem sqrt_lt_sqrt_iff_of_pos (hy : 0 < y) : √x < √y ↔ x < y := by
  /-
    x y : Real
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt x.sqrt y.sqrt) (LT.lt x y)
  -/
  rw [Real.sqrt, Real.sqrt, NNReal.coe_lt_coe, NNReal.sqrt_lt_sqrt, toNNReal_lt_toNNReal_iff hy]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
theorem sqrt_le_sqrt (h : x ≤ y) : √x ≤ √y := by
  /-
    x y : Real
    h : LE.le x y
    ⊢ LE.le x.sqrt y.sqrt
  -/
  rw [Real.sqrt, Real.sqrt, NNReal.coe_le_coe, NNReal.sqrt_le_sqrt]
  /-
    x y : Real
    h : LE.le x y
    ⊢ LE.le x.toNNReal y.toNNReal
  -/
  exact toNNReal_le_toNNReal h
  /-
    🎉 no goals
  -/


@[gcongr, bound]
theorem sqrt_lt_sqrt (hx : 0 ≤ x) (h : x < y) : √x < √y :=
  (sqrt_lt_sqrt_iff hx).2 h


theorem sqrt_le_left (hy : 0 ≤ y) : √x ≤ y ↔ x ≤ y ^ 2 := by
  rw [sqrt, ← Real.le_toNNReal_iff_coe_le hy, NNReal.sqrt_le_iff_le_sq, sq, ← Real.toNNReal_mul hy,
    Real.toNNReal_le_toNNReal_iff (mul_self_nonneg y), sq]


theorem sqrt_le_iff : √x ≤ y ↔ 0 ≤ y ∧ x ≤ y ^ 2 := by
  /-
    x y : Real
    ⊢ Iff (LE.le x.sqrt y) (And (LE.le 0 y) (LE.le x (HPow.hPow y 2)))
  -/
  rw [← and_iff_right_of_imp fun h => (sqrt_nonneg x).trans h, and_congr_right_iff]
  /-
    x y : Real
    ⊢ LE.le 0 y → Iff (LE.le x.sqrt y) (LE.le x (HPow.hPow y 2))
  -/
  exact sqrt_le_left
  /-
    🎉 no goals
  -/


theorem sqrt_lt (hx : 0 ≤ x) (hy : 0 ≤ y) : √x < y ↔ x < y ^ 2 := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Iff (LT.lt x.sqrt y) (LT.lt x (HPow.hPow y 2))
  -/
  rw [← sqrt_lt_sqrt_iff hx, sqrt_sq hy]
  /-
    🎉 no goals
  -/


theorem sqrt_lt' (hy : 0 < y) : √x < y ↔ x < y ^ 2 := by
  /-
    x y : Real
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt x.sqrt y) (LT.lt x (HPow.hPow y 2))
  -/
  rw [← sqrt_lt_sqrt_iff_of_pos (pow_pos hy _), sqrt_sq hy.le]
  /-
    🎉 no goals
  -/


/-- Note: if you want to conclude `x ≤ √y`, then use `Real.le_sqrt_of_sq_le`.
If you have `x > 0`, consider using `Real.le_sqrt'` -/
theorem le_sqrt (hx : 0 ≤ x) (hy : 0 ≤ y) : x ≤ √y ↔ x ^ 2 ≤ y :=
  le_iff_le_iff_lt_iff_lt.2 <| sqrt_lt hy hx


theorem le_sqrt' (hx : 0 < x) : x ≤ √y ↔ x ^ 2 ≤ y :=
  le_iff_le_iff_lt_iff_lt.2 <| sqrt_lt' hx


theorem abs_le_sqrt (h : x ^ 2 ≤ y) : |x| ≤ √y := by
  /-
    x y : Real
    h : LE.le (HPow.hPow x 2) y
    ⊢ LE.le (abs x) y.sqrt
  -/
  rw [← sqrt_sq_eq_abs]; exact sqrt_le_sqrt h
                         /-
                           🎉 no goals
                         -/


theorem sq_le (h : 0 ≤ y) : x ^ 2 ≤ y ↔ -√y ≤ x ∧ x ≤ √y := by
  /-
    x y : Real
    h : LE.le 0 y
    ⊢ Iff (LE.le (HPow.hPow x 2) y) (And (LE.le (Neg.neg y.sqrt) x) (LE.le x y.sqr …
  -/
  constructor
    /-
      case mp
      x y : Real
      h : LE.le 0 y
      ⊢ LE.le (HPow.hPow x 2) y → And (LE.le (Neg.neg y.sqrt) x) (LE.le x y.sqrt)
    -/
  · simpa only [abs_le] using abs_le_sqrt
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x y : Real
      h : LE.le 0 y
      ⊢ And (LE.le (Neg.neg y.sqrt) x) (LE.le x y.sqrt) → LE.le (HPow.hPow x 2) y
    -/
  · rw [← abs_le, ← sq_abs]
    /-
      case mpr
      x y : Real
      h : LE.le 0 y
      ⊢ LE.le (abs x) y.sqrt → LE.le (HPow.hPow (abs x) 2) y
    -/
    exact (le_sqrt (abs_nonneg x) h).mp
    /-
      🎉 no goals
    -/


theorem neg_sqrt_le_of_sq_le (h : x ^ 2 ≤ y) : -√y ≤ x :=
  ((sq_le ((sq_nonneg x).trans h)).mp h).1


theorem le_sqrt_of_sq_le (h : x ^ 2 ≤ y) : x ≤ √y :=
  ((sq_le ((sq_nonneg x).trans h)).mp h).2


@[simp]
theorem sqrt_inj (hx : 0 ≤ x) (hy : 0 ≤ y) : √x = √y ↔ x = y := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Iff (Eq x.sqrt y.sqrt) (Eq x y)
  -/
  simp [le_antisymm_iff, hx, hy]
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          x : Real
                                                          h : LE.le 0 x
                                                          ⊢ Iff (Eq x.sqrt 0) (Eq x 0)
                                                        -/
theorem sqrt_eq_zero (h : 0 ≤ x) : √x = 0 ↔ x = 0 := by simpa using sqrt_inj h le_rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem sqrt_eq_zero' : √x = 0 ↔ x ≤ 0 := by
  /-
    x : Real
    ⊢ Iff (Eq x.sqrt 0) (LE.le x 0)
  -/
  rw [sqrt, NNReal.coe_eq_zero, NNReal.sqrt_eq_zero, Real.toNNReal_eq_zero]
  /-
    🎉 no goals
  -/


                                                        /-
                                                          x : Real
                                                          h : LE.le 0 x
                                                          ⊢ Iff (Ne x.sqrt 0) (Ne x 0)
                                                        -/
theorem sqrt_ne_zero (h : 0 ≤ x) : √x ≠ 0 ↔ x ≠ 0 := by rw [not_iff_not, sqrt_eq_zero h]
                                                        /-
                                                          🎉 no goals
                                                        -/


                                             /-
                                               x : Real
                                               ⊢ Iff (Ne x.sqrt 0) (LT.lt 0 x)
                                             -/
theorem sqrt_ne_zero' : √x ≠ 0 ↔ 0 < x := by rw [← not_le, not_iff_not, sqrt_eq_zero']
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem sqrt_pos : 0 < √x ↔ 0 < x :=
                                        /-
                                          x : Real
                                          ⊢ Iff (LE.le x.sqrt 0) (Eq x.sqrt 0)
                                        -/
  lt_iff_lt_of_le_iff_le (Iff.trans (by simp [le_antisymm_iff, sqrt_nonneg]) sqrt_eq_zero')
                                        /-
                                          🎉 no goals
                                        -/


lemma sqrt_le_sqrt_iff' (hx : 0 < x) : √x ≤ √y ↔ x ≤ y := by
  /-
    x y : Real
    hx : LT.lt 0 x
    ⊢ Iff (LE.le x.sqrt y.sqrt) (LE.le x y)
  -/
  obtain hy | hy := le_total y 0
  · exact iff_of_false ((sqrt_eq_zero_of_nonpos hy).trans_lt <| sqrt_pos.2 hx).not_le
      (hy.trans_lt hx).not_le
    /-
      case inr
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le 0 y
      ⊢ Iff (LE.le x.sqrt y.sqrt) (LE.le x y)
    -/
  · exact sqrt_le_sqrt_iff hy
    /-
      🎉 no goals
    -/


@[simp] lemma one_le_sqrt : 1 ≤ √x ↔ 1 ≤ x := by
  /-
    x : Real
    ⊢ Iff (LE.le 1 x.sqrt) (LE.le 1 x)
  -/
  rw [← sqrt_one, sqrt_le_sqrt_iff' zero_lt_one, sqrt_one]
  /-
    🎉 no goals
  -/


@[simp] lemma sqrt_le_one : √x ≤ 1 ↔ x ≤ 1 := by
  /-
    x : Real
    ⊢ Iff (LE.le x.sqrt 1) (LE.le x 1)
  -/
  rw [← sqrt_one, sqrt_le_sqrt_iff zero_le_one, sqrt_one]
  /-
    🎉 no goals
  -/


/-- Extension for the `positivity` tactic: a square root of a strictly positive nonnegative real is
positive. -/
@[positivity NNReal.sqrt _]
def evalNNRealSqrt : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(NNReal), ~q(NNReal.sqrt $a) =>
    let ra ← core  q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa => pure (.positive q(NNReal.sqrt_pos_of_pos $pa))
    | _ => failure -- this case is dealt with by generic nonnegativity of nnreals
  | _, _, _ => throwError "not NNReal.sqrt"


/-- Extension for the `positivity` tactic: a square root is nonnegative, and is strictly positive if
its input is. -/
@[positivity √ _]
def evalSqrt : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(√$a) =>
    let ra ← catchNone <| core q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa => pure (.positive q(Real.sqrt_pos_of_pos $pa))
    | _ => pure (.nonnegative q(Real.sqrt_nonneg $a))
  | _, _, _ => throwError "not Real.sqrt"


@[simp]
theorem sqrt_mul {x : ℝ} (hx : 0 ≤ x) (y : ℝ) : √(x * y) = √x * √y := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (HMul.hMul x y).sqrt (HMul.hMul x.sqrt y.sqrt)
  -/
  simp_rw [Real.sqrt, ← NNReal.coe_mul, NNReal.coe_inj, Real.toNNReal_mul hx, NNReal.sqrt_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_mul' (x) {y : ℝ} (hy : 0 ≤ y) : √(x * y) = √x * √y := by
  /-
    x y : Real
    hy : LE.le 0 y
    ⊢ Eq (HMul.hMul x y).sqrt (HMul.hMul x.sqrt y.sqrt)
  -/
  rw [mul_comm, sqrt_mul hy, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_inv (x : ℝ) : √x⁻¹ = (√x)⁻¹ := by
  /-
    x : Real
    ⊢ Eq (Inv.inv x).sqrt (Inv.inv x.sqrt)
  -/
  rw [Real.sqrt, Real.toNNReal_inv, NNReal.sqrt_inv, NNReal.coe_inv, Real.sqrt]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_div {x : ℝ} (hx : 0 ≤ x) (y : ℝ) : √(x / y) = √x / √y := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (HDiv.hDiv x y).sqrt (HDiv.hDiv x.sqrt y.sqrt)
  -/
  rw [division_def, sqrt_mul hx, sqrt_inv, division_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem sqrt_div' (x) {y : ℝ} (hy : 0 ≤ y) : √(x / y) = √x / √y := by
  /-
    x y : Real
    hy : LE.le 0 y
    ⊢ Eq (HDiv.hDiv x y).sqrt (HDiv.hDiv x.sqrt y.sqrt)
  -/
  rw [division_def, sqrt_mul' x (inv_nonneg.2 hy), sqrt_inv, division_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem div_sqrt : x / √x = √x := by
  /-
    x : Real
    ⊢ Eq (HDiv.hDiv x x.sqrt) x.sqrt
  -/
  rcases le_or_lt x 0 with h | h
    /-
      case inl
      x : Real
      h : LE.le x 0
      ⊢ Eq (HDiv.hDiv x x.sqrt) x.sqrt
    -/
  · rw [sqrt_eq_zero'.mpr h, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      h : LT.lt 0 x
      ⊢ Eq (HDiv.hDiv x x.sqrt) x.sqrt
    -/
  · rw [div_eq_iff (sqrt_ne_zero'.mpr h), mul_self_sqrt h.le]
    /-
      🎉 no goals
    -/


                                               /-
                                                 x : Real
                                                 ⊢ Eq (HDiv.hDiv x.sqrt x) (HDiv.hDiv 1 x.sqrt)
                                               -/
theorem sqrt_div_self' : √x / x = 1 / √x := by rw [← div_sqrt, one_div_div, div_sqrt]
                                               /-
                                                 🎉 no goals
                                               -/


                                              /-
                                                x : Real
                                                ⊢ Eq (HDiv.hDiv x.sqrt x) (Inv.inv x.sqrt)
                                              -/
theorem sqrt_div_self : √x / x = (√x)⁻¹ := by rw [sqrt_div_self', one_div]
                                              /-
                                                🎉 no goals
                                              -/


theorem lt_sqrt (hx : 0 ≤ x) : x < √y ↔ x ^ 2 < y := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Iff (LT.lt x y.sqrt) (LT.lt (HPow.hPow x 2) y)
  -/
  rw [← sqrt_lt_sqrt_iff (sq_nonneg _), sqrt_sq hx]
  /-
    🎉 no goals
  -/


theorem sq_lt : x ^ 2 < y ↔ -√y < x ∧ x < √y := by
  /-
    x y : Real
    ⊢ Iff (LT.lt (HPow.hPow x 2) y) (And (LT.lt (Neg.neg y.sqrt) x) (LT.lt x y.sqr …
  -/
  rw [← abs_lt, ← sq_abs, lt_sqrt (abs_nonneg _)]
  /-
    🎉 no goals
  -/


theorem neg_sqrt_lt_of_sq_lt (h : x ^ 2 < y) : -√y < x :=
  (sq_lt.mp h).1


theorem lt_sqrt_of_sq_lt (h : x ^ 2 < y) : x < √y :=
  (sq_lt.mp h).2


theorem lt_sq_of_sqrt_lt (h : √x < y) : x < y ^ 2 := by
  /-
    x y : Real
    h : LT.lt x.sqrt y
    ⊢ LT.lt x (HPow.hPow y 2)
  -/
  have hy := x.sqrt_nonneg.trans_lt h
  /-
    x y : Real
    h : LT.lt x.sqrt y
    hy : LT.lt 0 y
    ⊢ LT.lt x (HPow.hPow y 2)
  -/
  rwa [← sqrt_lt_sqrt_iff_of_pos (sq_pos_of_pos hy), sqrt_sq hy.le]
  /-
    🎉 no goals
  -/


/-- The natural square root is at most the real square root -/
theorem nat_sqrt_le_real_sqrt {a : ℕ} : ↑(Nat.sqrt a) ≤ √(a : ℝ) := by
  /-
    a : Nat
    ⊢ LE.le (↑a.sqrt) (↑a).sqrt
  -/
  rw [Real.le_sqrt (Nat.cast_nonneg _) (Nat.cast_nonneg _)]
  /-
    a : Nat
    ⊢ LE.le (HPow.hPow (↑a.sqrt) 2) ↑a
  -/
  norm_cast
  /-
    a : Nat
    ⊢ LE.le (HPow.hPow a.sqrt 2) a
  -/
  exact Nat.sqrt_le' a
  /-
    🎉 no goals
  -/


/-- The real square root is less than the natural square root plus one -/
theorem real_sqrt_lt_nat_sqrt_succ {a : ℕ} : √(a : ℝ) < Nat.sqrt a + 1 := by
  /-
    a : Nat
    ⊢ LT.lt (↑a).sqrt (HAdd.hAdd (↑a.sqrt) 1)
  -/
  rw [sqrt_lt (by simp)] <;> norm_cast
    /-
      a : Nat
      ⊢ LT.lt a (HPow.hPow (HAdd.hAdd a.sqrt 1) 2)
    -/
  · exact Nat.lt_succ_sqrt' a
    /-
      🎉 no goals
    -/
    /-
      a : Nat
      ⊢ LE.le 0 (HAdd.hAdd a.sqrt 1)
    -/
  · exact Nat.le_add_left 0 (Nat.sqrt a + 1)
    /-
      🎉 no goals
    -/


/-- The real square root is at most the natural square root plus one -/
theorem real_sqrt_le_nat_sqrt_succ {a : ℕ} : √(a : ℝ) ≤ Nat.sqrt a + 1 :=
  real_sqrt_lt_nat_sqrt_succ.le


/-- The floor of the real square root is the same as the natural square root. -/
@[simp]
theorem floor_real_sqrt_eq_nat_sqrt {a : ℕ} : ⌊√(a : ℝ)⌋ = Nat.sqrt a := by
  /-
    a : Nat
    ⊢ Eq (Int.floor (↑a).sqrt) ↑a.sqrt
  -/
  rw [Int.floor_eq_iff]
  /-
    a : Nat
    ⊢ And (LE.le (↑↑a.sqrt) (↑a).sqrt) (LT.lt (↑a).sqrt (HAdd.hAdd (↑↑a.sqrt) 1))
  -/
  exact ⟨nat_sqrt_le_real_sqrt, real_sqrt_lt_nat_sqrt_succ⟩
  /-
    🎉 no goals
  -/


/-- The natural floor of the real square root is the same as the natural square root. -/
@[simp]
theorem nat_floor_real_sqrt_eq_nat_sqrt {a : ℕ} : ⌊√(a : ℝ)⌋₊ = Nat.sqrt a := by
  /-
    a : Nat
    ⊢ Eq (Nat.floor (↑a).sqrt) a.sqrt
  -/
  rw [Nat.floor_eq_iff (sqrt_nonneg a)]
  /-
    a : Nat
    ⊢ And (LE.le (↑a.sqrt) (↑a).sqrt) (LT.lt (↑a).sqrt (HAdd.hAdd (↑a.sqrt) 1))
  -/
  exact ⟨nat_sqrt_le_real_sqrt, real_sqrt_lt_nat_sqrt_succ⟩
  /-
    🎉 no goals
  -/


/-- Bernoulli's inequality for exponent `1 / 2`, stated using `sqrt`. -/
theorem sqrt_one_add_le (h : -1 ≤ x) : √(1 + x) ≤ 1 + x / 2 := by
  /-
    x : Real
    h : LE.le (-1) x
    ⊢ LE.le (HAdd.hAdd 1 x).sqrt (HAdd.hAdd 1 (HDiv.hDiv x 2))
  -/
  refine sqrt_le_iff.mpr ⟨by linarith, ?_⟩
  calc 1 + x
    _ ≤ 1 + x + (x / 2) ^ 2 := le_add_of_nonneg_right <| sq_nonneg _
    _ = _ := by ring


theorem Filter.Tendsto.sqrt {f : α → ℝ} {l : Filter α} {x : ℝ} (h : Tendsto f l (𝓝 x)) :
    Tendsto (fun x => √(f x)) l (𝓝 (√x)) :=
  (continuous_sqrt.tendsto _).comp h


nonrec theorem ContinuousWithinAt.sqrt (h : ContinuousWithinAt f s x) :
    ContinuousWithinAt (fun x => √(f x)) s x :=
  h.sqrt


@[fun_prop]
nonrec theorem ContinuousAt.sqrt (h : ContinuousAt f x) : ContinuousAt (fun x => √(f x)) x :=
  h.sqrt


@[fun_prop]
theorem ContinuousOn.sqrt (h : ContinuousOn f s) : ContinuousOn (fun x => √(f x)) s :=
  fun x hx => (h x hx).sqrt


@[continuity, fun_prop]
theorem Continuous.sqrt (h : Continuous f) : Continuous fun x => √(f x) :=
  continuous_sqrt.comp h


/-- **Cauchy-Schwarz inequality** for finsets using square roots in `ℝ≥0`. -/
lemma sum_mul_le_sqrt_mul_sqrt (s : Finset ι) (f g : ι → ℝ≥0) :
    ∑ i ∈ s, f i * g i ≤ sqrt (∑ i ∈ s, f i ^ 2) * sqrt (∑ i ∈ s, g i ^ 2) :=
  (le_sqrt_iff_sq_le.2 <| sum_mul_sq_le_sq_mul_sq _ _ _).trans_eq <| sqrt_mul _ _


/-- **Cauchy-Schwarz inequality** for finsets using square roots in `ℝ≥0`. -/
lemma sum_sqrt_mul_sqrt_le (s : Finset ι) (f g : ι → ℝ≥0) :
    ∑ i ∈ s, sqrt (f i) * sqrt (g i) ≤ sqrt (∑ i ∈ s, f i) * sqrt (∑ i ∈ s, g i) := by
  /-
    ι : Type u_2
    s : Finset ι
    f g : ι → NNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul (NNReal.sqrt (f i)) (NNReal.sqrt (g i))) (HM …
  -/
  simpa [*] using sum_mul_le_sqrt_mul_sqrt _ (fun x ↦ sqrt (f x)) (fun x ↦ sqrt (g x))
  /-
    🎉 no goals
  -/


/-- **Cauchy-Schwarz inequality** for finsets using square roots in `ℝ`. -/
lemma sum_mul_le_sqrt_mul_sqrt (s : Finset ι) (f g : ι → ℝ) :
    ∑ i ∈ s, f i * g i ≤ √(∑ i ∈ s, f i ^ 2) * √(∑ i ∈ s, g i ^ 2) :=
  (le_sqrt_of_sq_le <| sum_mul_sq_le_sq_mul_sq _ _ _).trans_eq <| sqrt_mul
                             /-
                               ι : Type u_2
                               s : Finset ι
                               f g : ι → Real
                               x✝¹ : ι
                               x✝ : Membership.mem s x✝¹
                               ⊢ LE.le 0 (HPow.hPow (f x✝¹) 2)
                             -/
    (sum_nonneg fun _ _ ↦ by positivity) _
                             /-
                               🎉 no goals
                             -/


/-- **Cauchy-Schwarz inequality** for finsets using square roots in `ℝ`. -/
lemma sum_sqrt_mul_sqrt_le (s : Finset ι) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i) :
    ∑ i ∈ s, √(f i) * √(g i) ≤ √(∑ i ∈ s, f i) * √(∑ i ∈ s, g i) := by
  /-
    ι : Type u_2
    f g : ι → Real
    s : Finset ι
    hf : ∀ (i : ι), LE.le 0 (f i)
    hg : ∀ (i : ι), LE.le 0 (g i)
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i).sqrt (g i).sqrt) (HMul.hMul (s.sum fun …
  -/
  simpa [*] using sum_mul_le_sqrt_mul_sqrt _ (fun x ↦ √(f x)) (fun x ↦ √(g x))
  /-
    🎉 no goals
  -/


