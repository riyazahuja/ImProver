/-- The sign function that maps negative real numbers to -1, positive numbers to 1, and 0
otherwise. -/
noncomputable def sign (r : ℝ) : ℝ :=
  if r < 0 then -1 else if 0 < r then 1 else 0


                                                             /-
                                                               r : Real
                                                               hr : LT.lt r 0
                                                               ⊢ Eq r.sign (-1)
                                                             -/
theorem sign_of_neg {r : ℝ} (hr : r < 0) : sign r = -1 := by rw [sign, if_pos hr]
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                            /-
                                                              r : Real
                                                              hr : LT.lt 0 r
                                                              ⊢ Eq r.sign 1
                                                            -/
theorem sign_of_pos {r : ℝ} (hr : 0 < r) : sign r = 1 := by rw [sign, if_pos hr, if_neg hr.not_lt]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
                                     /-
                                       ⊢ Eq (Real.sign 0) 0
                                     -/
theorem sign_zero : sign 0 = 0 := by rw [sign, if_neg (lt_irrefl _), if_neg (lt_irrefl _)]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem sign_one : sign 1 = 1 :=
                    /-
                      ⊢ LT.lt 0 1
                    -/
  sign_of_pos <| by norm_num
                    /-
                      🎉 no goals
                    -/


theorem sign_apply_eq (r : ℝ) : sign r = -1 ∨ sign r = 0 ∨ sign r = 1 := by
  /-
    r : Real
    ⊢ Or (Eq r.sign (-1)) (Or (Eq r.sign 0) (Eq r.sign 1))
  -/
  obtain hn | rfl | hp := lt_trichotomy r (0 : ℝ)
    /-
      case inl
      r : Real
      hn : LT.lt r 0
      ⊢ Or (Eq r.sign (-1)) (Or (Eq r.sign 0) (Eq r.sign 1))
    -/
  · exact Or.inl <| sign_of_neg hn
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Or (Eq (Real.sign 0) (-1)) (Or (Eq (Real.sign 0) 0) (Eq (Real.sign 0) 1))
    -/
  · exact Or.inr <| Or.inl <| sign_zero
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      hp : LT.lt 0 r
      ⊢ Or (Eq r.sign (-1)) (Or (Eq r.sign 0) (Eq r.sign 1))
    -/
  · exact Or.inr <| Or.inr <| sign_of_pos hp
    /-
      🎉 no goals
    -/


/-- This lemma is useful for working with `ℝˣ` -/
theorem sign_apply_eq_of_ne_zero (r : ℝ) (h : r ≠ 0) : sign r = -1 ∨ sign r = 1 :=
  h.lt_or_lt.imp sign_of_neg sign_of_pos


@[simp]
theorem sign_eq_zero_iff {r : ℝ} : sign r = 0 ↔ r = 0 := by
  /-
    r : Real
    ⊢ Iff (Eq r.sign 0) (Eq r 0)
  -/
  refine ⟨fun h => ?_, fun h => h.symm ▸ sign_zero⟩
  /-
    r : Real
    h : Eq r.sign 0
    ⊢ Eq r 0
  -/
  obtain hn | rfl | hp := lt_trichotomy r (0 : ℝ)
    /-
      case inl
      r : Real
      h : Eq r.sign 0
      hn : LT.lt r 0
      ⊢ Eq r 0
    -/
  · rw [sign_of_neg hn, neg_eq_zero] at h
    /-
      case inl
      r : Real
      h : Eq 1 0
      hn : LT.lt r 0
      ⊢ Eq r 0
    -/
    exact (one_ne_zero h).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      h : Eq (Real.sign 0) 0
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      h : Eq r.sign 0
      hp : LT.lt 0 r
      ⊢ Eq r 0
    -/
  · rw [sign_of_pos hp] at h
    /-
      case inr.inr
      r : Real
      h : Eq 1 0
      hp : LT.lt 0 r
      ⊢ Eq r 0
    -/
    exact (one_ne_zero h).elim
    /-
      🎉 no goals
    -/


theorem sign_intCast (z : ℤ) : sign (z : ℝ) = ↑(Int.sign z) := by
  /-
    z : Int
    ⊢ Eq (↑z).sign ↑z.sign
  -/
  obtain hn | rfl | hp := lt_trichotomy z (0 : ℤ)
  · rw [sign_of_neg (Int.cast_lt_zero.mpr hn), Int.sign_eq_neg_one_of_neg hn, Int.cast_neg,
      Int.cast_one]
    /-
      case inr.inl
      ⊢ Eq (↑0).sign ↑(Int.sign 0)
    -/
  · rw [Int.cast_zero, sign_zero, Int.sign_zero, Int.cast_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      z : Int
      hp : LT.lt 0 z
      ⊢ Eq (↑z).sign ↑z.sign
    -/
  · rw [sign_of_pos (Int.cast_pos.mpr hp), Int.sign_eq_one_of_pos hp, Int.cast_one]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias sign_int_cast := sign_intCast


theorem sign_neg {r : ℝ} : sign (-r) = -sign r := by
  /-
    r : Real
    ⊢ Eq (Neg.neg r).sign (Neg.neg r.sign)
  -/
  obtain hn | rfl | hp := lt_trichotomy r (0 : ℝ)
    /-
      case inl
      r : Real
      hn : LT.lt r 0
      ⊢ Eq (Neg.neg r).sign (Neg.neg r.sign)
    -/
  · rw [sign_of_neg hn, sign_of_pos (neg_pos.mpr hn), neg_neg]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Eq (-0).sign (Neg.neg (Real.sign 0))
    -/
  · rw [sign_zero, neg_zero, sign_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      hp : LT.lt 0 r
      ⊢ Eq (Neg.neg r).sign (Neg.neg r.sign)
    -/
  · rw [sign_of_pos hp, sign_of_neg (neg_lt_zero.mpr hp)]
    /-
      🎉 no goals
    -/


theorem sign_mul_nonneg (r : ℝ) : 0 ≤ sign r * r := by
  /-
    r : Real
    ⊢ LE.le 0 (HMul.hMul r.sign r)
  -/
  obtain hn | rfl | hp := lt_trichotomy r (0 : ℝ)
    /-
      case inl
      r : Real
      hn : LT.lt r 0
      ⊢ LE.le 0 (HMul.hMul r.sign r)
    -/
  · rw [sign_of_neg hn]
    /-
      case inl
      r : Real
      hn : LT.lt r 0
      ⊢ LE.le 0 (HMul.hMul (-1) r)
    -/
    exact mul_nonneg_of_nonpos_of_nonpos (by norm_num) hn.le
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ LE.le 0 (HMul.hMul (Real.sign 0) 0)
    -/
  · rw [mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      hp : LT.lt 0 r
      ⊢ LE.le 0 (HMul.hMul r.sign r)
    -/
  · rw [sign_of_pos hp, one_mul]
    /-
      case inr.inr
      r : Real
      hp : LT.lt 0 r
      ⊢ LE.le 0 r
    -/
    exact hp.le
    /-
      🎉 no goals
    -/


theorem sign_mul_pos_of_ne_zero (r : ℝ) (hr : r ≠ 0) : 0 < sign r * r := by
  /-
    r : Real
    hr : Ne r 0
    ⊢ LT.lt 0 (HMul.hMul r.sign r)
  -/
  refine lt_of_le_of_ne (sign_mul_nonneg r) fun h => hr ?_
  /-
    r : Real
    hr : Ne r 0
    h : Eq 0 (HMul.hMul r.sign r)
    ⊢ Eq r 0
  -/
  have hs0 := (zero_eq_mul.mp h).resolve_right hr
  /-
    r : Real
    hr : Ne r 0
    h : Eq 0 (HMul.hMul r.sign r)
    hs0 : Eq r.sign 0
    ⊢ Eq r 0
  -/
  exact sign_eq_zero_iff.mp hs0
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_sign (r : ℝ) : (sign r)⁻¹ = sign r := by
  /-
    r : Real
    ⊢ Eq (Inv.inv r.sign) r.sign
  -/
  obtain hn | hz | hp := sign_apply_eq r
    /-
      case inl
      r : Real
      hn : Eq r.sign (-1)
      ⊢ Eq (Inv.inv r.sign) r.sign
    -/
  · rw [hn]
    /-
      case inl
      r : Real
      hn : Eq r.sign (-1)
      ⊢ Eq (Inv.inv (-1)) (-1)
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      r : Real
      hz : Eq r.sign 0
      ⊢ Eq (Inv.inv r.sign) r.sign
    -/
  · rw [hz]
    /-
      case inr.inl
      r : Real
      hz : Eq r.sign 0
      ⊢ Eq (Inv.inv 0) 0
    -/
    exact inv_zero
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      hp : Eq r.sign 1
      ⊢ Eq (Inv.inv r.sign) r.sign
    -/
  · rw [hp]
    /-
      case inr.inr
      r : Real
      hp : Eq r.sign 1
      ⊢ Eq (Inv.inv 1) 1
    -/
    exact inv_one
    /-
      🎉 no goals
    -/


@[simp]
theorem sign_inv (r : ℝ) : sign r⁻¹ = sign r := by
  /-
    r : Real
    ⊢ Eq (Inv.inv r).sign r.sign
  -/
  obtain hn | rfl | hp := lt_trichotomy r (0 : ℝ)
    /-
      case inl
      r : Real
      hn : LT.lt r 0
      ⊢ Eq (Inv.inv r).sign r.sign
    -/
  · rw [sign_of_neg hn, sign_of_neg (inv_lt_zero.mpr hn)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Eq (Inv.inv 0).sign (Real.sign 0)
    -/
  · rw [sign_zero, inv_zero, sign_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      r : Real
      hp : LT.lt 0 r
      ⊢ Eq (Inv.inv r).sign r.sign
    -/
  · rw [sign_of_pos hp, sign_of_pos (inv_pos.mpr hp)]
    /-
      🎉 no goals
    -/


