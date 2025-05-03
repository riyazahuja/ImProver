/-- The real logarithm function, equal to the inverse of the exponential for `x > 0`,
to `log |x|` for `x < 0`, and to `0` for `0`. We use this unconventional extension to
`(-∞, 0]` as it gives the formula `log (x * y) = log x + log y` for all nonzero `x` and `y`, and
the derivative of `log` is `1/x` away from `0`. -/
@[pp_nodot]
noncomputable def log (x : ℝ) : ℝ :=
  if hx : x = 0 then 0 else expOrderIso.symm ⟨|x|, abs_pos.2 hx⟩


theorem log_of_ne_zero (hx : x ≠ 0) : log x = expOrderIso.symm ⟨|x|, abs_pos.2 hx⟩ :=
  dif_neg hx


theorem log_of_pos (hx : 0 < x) : log x = expOrderIso.symm ⟨x, hx⟩ := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.log x) (Real.expOrderIso.symm ⟨x, hx⟩)
  -/
  rw [log_of_ne_zero hx.ne']
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.expOrderIso.symm ⟨abs x, ⋯⟩) (Real.expOrderIso.symm ⟨x, hx⟩)
  -/
  congr
  /-
    case h.e_6.h.e_val
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (abs x) x
  -/
  exact abs_of_pos hx
  /-
    🎉 no goals
  -/


theorem exp_log_eq_abs (hx : x ≠ 0) : exp (log x) = |x| := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ Eq (Real.exp (Real.log x)) (abs x)
  -/
  rw [log_of_ne_zero hx, ← coe_expOrderIso_apply, OrderIso.apply_symm_apply, Subtype.coe_mk]
  /-
    🎉 no goals
  -/


theorem exp_log (hx : 0 < x) : exp (log x) = x := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.exp (Real.log x)) x
  -/
  rw [exp_log_eq_abs hx.ne']
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (abs x) x
  -/
  exact abs_of_pos hx
  /-
    🎉 no goals
  -/


theorem exp_log_of_neg (hx : x < 0) : exp (log x) = -x := by
  /-
    x : Real
    hx : LT.lt x 0
    ⊢ Eq (Real.exp (Real.log x)) (Neg.neg x)
  -/
  rw [exp_log_eq_abs (ne_of_lt hx)]
  /-
    x : Real
    hx : LT.lt x 0
    ⊢ Eq (abs x) (Neg.neg x)
  -/
  exact abs_of_neg hx
  /-
    🎉 no goals
  -/


theorem le_exp_log (x : ℝ) : x ≤ exp (log x) := by
  /-
    x : Real
    ⊢ LE.le x (Real.exp (Real.log x))
  -/
  by_cases h_zero : x = 0
    /-
      case pos
      x : Real
      h_zero : Eq x 0
      ⊢ LE.le x (Real.exp (Real.log x))
    -/
  · rw [h_zero, log, dif_pos rfl, exp_zero]
    /-
      case pos
      x : Real
      h_zero : Eq x 0
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Real
      h_zero : Not (Eq x 0)
      ⊢ LE.le x (Real.exp (Real.log x))
    -/
  · rw [exp_log_eq_abs h_zero]
    /-
      case neg
      x : Real
      h_zero : Not (Eq x 0)
      ⊢ LE.le x (abs x)
    -/
    exact le_abs_self _
    /-
      🎉 no goals
    -/


@[simp]
theorem log_exp (x : ℝ) : log (exp x) = x :=
  exp_injective <| exp_log (exp_pos x)


theorem surjOn_log : SurjOn log (Ioi 0) univ := fun x _ => ⟨exp x, exp_pos x, log_exp x⟩


theorem log_surjective : Surjective log := fun x => ⟨exp x, log_exp x⟩


@[simp]
theorem range_log : range log = univ :=
  log_surjective.range_eq


@[simp]
theorem log_zero : log 0 = 0 :=
  dif_pos rfl


@[simp]
theorem log_one : log 1 = 0 :=
                      /-
                        ⊢ Eq (Real.exp (Real.log 1)) (Real.exp 0)
                      -/
  exp_injective <| by rw [exp_log zero_lt_one, exp_zero]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem log_abs (x : ℝ) : log |x| = log x := by
  /-
    x : Real
    ⊢ Eq (Real.log (abs x)) (Real.log x)
  -/
  by_cases h : x = 0
    /-
      case pos
      x : Real
      h : Eq x 0
      ⊢ Eq (Real.log (abs x)) (Real.log x)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Real
      h : Not (Eq x 0)
      ⊢ Eq (Real.log (abs x)) (Real.log x)
    -/
  · rw [← exp_eq_exp, exp_log_eq_abs h, exp_log_eq_abs (abs_pos.2 h).ne', abs_abs]
    /-
      🎉 no goals
    -/


@[simp]
                                                        /-
                                                          x : Real
                                                          ⊢ Eq (Real.log (Neg.neg x)) (Real.log x)
                                                        -/
theorem log_neg_eq_log (x : ℝ) : log (-x) = log x := by rw [← log_abs x, ← log_abs (-x), abs_neg]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem sinh_log {x : ℝ} (hx : 0 < x) : sinh (log x) = (x - x⁻¹) / 2 := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.sinh (Real.log x)) (HDiv.hDiv (HSub.hSub x (Inv.inv x)) 2)
  -/
  rw [sinh_eq, exp_neg, exp_log hx]
  /-
    🎉 no goals
  -/


theorem cosh_log {x : ℝ} (hx : 0 < x) : cosh (log x) = (x + x⁻¹) / 2 := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (Real.cosh (Real.log x)) (HDiv.hDiv (HAdd.hAdd x (Inv.inv x)) 2)
  -/
  rw [cosh_eq, exp_neg, exp_log hx]
  /-
    🎉 no goals
  -/


theorem surjOn_log' : SurjOn log (Iio 0) univ := fun x _ =>
                                          /-
                                            x : Real
                                            x✝ : Membership.mem Set.univ x
                                            ⊢ Eq (Real.log (Neg.neg (Real.exp x))) x
                                          -/
  ⟨-exp x, neg_lt_zero.2 <| exp_pos x, by rw [log_neg_eq_log, log_exp]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem log_mul (hx : x ≠ 0) (hy : y ≠ 0) : log (x * y) = log x + log y :=
  exp_injective <| by
    /-
      x y : Real
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Eq (Real.exp (Real.log (HMul.hMul x y))) (Real.exp (HAdd.hAdd (Real.log x) ( …
    -/
    rw [exp_log_eq_abs (mul_ne_zero hx hy), exp_add, exp_log_eq_abs hx, exp_log_eq_abs hy, abs_mul]
    /-
      🎉 no goals
    -/


theorem log_div (hx : x ≠ 0) (hy : y ≠ 0) : log (x / y) = log x - log y :=
  exp_injective <| by
    /-
      x y : Real
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Eq (Real.exp (Real.log (HDiv.hDiv x y))) (Real.exp (HSub.hSub (Real.log x) ( …
    -/
    rw [exp_log_eq_abs (div_ne_zero hx hy), exp_sub, exp_log_eq_abs hx, exp_log_eq_abs hy, abs_div]
    /-
      🎉 no goals
    -/


@[simp]
theorem log_inv (x : ℝ) : log x⁻¹ = -log x := by
  /-
    x : Real
    ⊢ Eq (Real.log (Inv.inv x)) (Neg.neg (Real.log x))
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    x : Real
    hx : Not (Eq x 0)
    ⊢ Eq (Real.log (Inv.inv x)) (Neg.neg (Real.log x))
  -/
  rw [← exp_eq_exp, exp_log_eq_abs (inv_ne_zero hx), exp_neg, exp_log_eq_abs hx, abs_inv]
  /-
    🎉 no goals
  -/


theorem log_le_log_iff (h : 0 < x) (h₁ : 0 < y) : log x ≤ log y ↔ x ≤ y := by
  /-
    x y : Real
    h : LT.lt 0 x
    h₁ : LT.lt 0 y
    ⊢ Iff (LE.le (Real.log x) (Real.log y)) (LE.le x y)
  -/
  rw [← exp_le_exp, exp_log h, exp_log h₁]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
lemma log_le_log (hx : 0 < x) (hxy : x ≤ y) : log x ≤ log y :=
  (log_le_log_iff hx (hx.trans_le hxy)).2 hxy


@[gcongr, bound]
theorem log_lt_log (hx : 0 < x) (h : x < y) : log x < log y := by
  /-
    x y : Real
    hx : LT.lt 0 x
    h : LT.lt x y
    ⊢ LT.lt (Real.log x) (Real.log y)
  -/
  rwa [← exp_lt_exp, exp_log hx, exp_log (lt_trans hx h)]
  /-
    🎉 no goals
  -/


theorem log_lt_log_iff (hx : 0 < x) (hy : 0 < y) : log x < log y ↔ x < y := by
  /-
    x y : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (Real.log x) (Real.log y)) (LT.lt x y)
  -/
  rw [← exp_lt_exp, exp_log hx, exp_log hy]
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       x y : Real
                                                                       hx : LT.lt 0 x
                                                                       ⊢ Iff (LE.le (Real.log x) y) (LE.le x (Real.exp y))
                                                                     -/
theorem log_le_iff_le_exp (hx : 0 < x) : log x ≤ y ↔ x ≤ exp y := by rw [← exp_le_exp, exp_log hx]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                     /-
                                                                       x y : Real
                                                                       hx : LT.lt 0 x
                                                                       ⊢ Iff (LT.lt (Real.log x) y) (LT.lt x (Real.exp y))
                                                                     -/
theorem log_lt_iff_lt_exp (hx : 0 < x) : log x < y ↔ x < exp y := by rw [← exp_lt_exp, exp_log hx]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                     /-
                                                                       x y : Real
                                                                       hy : LT.lt 0 y
                                                                       ⊢ Iff (LE.le x (Real.log y)) (LE.le (Real.exp x) y)
                                                                     -/
theorem le_log_iff_exp_le (hy : 0 < y) : x ≤ log y ↔ exp x ≤ y := by rw [← exp_le_exp, exp_log hy]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                     /-
                                                                       x y : Real
                                                                       hy : LT.lt 0 y
                                                                       ⊢ Iff (LT.lt x (Real.log y)) (LT.lt (Real.exp x) y)
                                                                     -/
theorem lt_log_iff_exp_lt (hy : 0 < y) : x < log y ↔ exp x < y := by rw [← exp_lt_exp, exp_log hy]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem log_pos_iff (hx : 0 < x) : 0 < log x ↔ 1 < x := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt 0 (Real.log x)) (LT.lt 1 x)
  -/
  rw [← log_one]
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.log 1) (Real.log x)) (LT.lt 1 x)
  -/
  exact log_lt_log_iff zero_lt_one hx
  /-
    🎉 no goals
  -/


@[bound]
theorem log_pos (hx : 1 < x) : 0 < log x :=
  (log_pos_iff (lt_trans zero_lt_one hx)).2 hx


theorem log_pos_of_lt_neg_one (hx : x < -1) : 0 < log x := by
  /-
    x : Real
    hx : LT.lt x (-1)
    ⊢ LT.lt 0 (Real.log x)
  -/
  rw [← neg_neg x, log_neg_eq_log]
  /-
    x : Real
    hx : LT.lt x (-1)
    ⊢ LT.lt 0 (Real.log (Neg.neg x))
  -/
  have : 1 < -x := by linarith
  /-
    x : Real
    hx : LT.lt x (-1)
    this : LT.lt 1 (Neg.neg x)
    ⊢ LT.lt 0 (Real.log (Neg.neg x))
  -/
  exact log_pos this
  /-
    🎉 no goals
  -/


theorem log_neg_iff (h : 0 < x) : log x < 0 ↔ x < 1 := by
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.log x) 0) (LT.lt x 1)
  -/
  rw [← log_one]
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ Iff (LT.lt (Real.log x) (Real.log 1)) (LT.lt x 1)
  -/
  exact log_lt_log_iff h zero_lt_one
  /-
    🎉 no goals
  -/


@[bound]
theorem log_neg (h0 : 0 < x) (h1 : x < 1) : log x < 0 :=
  (log_neg_iff h0).2 h1


theorem log_neg_of_lt_zero (h0 : x < 0) (h1 : -1 < x) : log x < 0 := by
  /-
    x : Real
    h0 : LT.lt x 0
    h1 : LT.lt (-1) x
    ⊢ LT.lt (Real.log x) 0
  -/
  rw [← neg_neg x, log_neg_eq_log]
  /-
    x : Real
    h0 : LT.lt x 0
    h1 : LT.lt (-1) x
    ⊢ LT.lt (Real.log (Neg.neg x)) 0
  -/
  have h0' : 0 < -x := by linarith
  /-
    x : Real
    h0 : LT.lt x 0
    h1 : LT.lt (-1) x
    h0' : LT.lt 0 (Neg.neg x)
    ⊢ LT.lt (Real.log (Neg.neg x)) 0
  -/
  have h1' : -x < 1 := by linarith
  /-
    x : Real
    h0 : LT.lt x 0
    h1 : LT.lt (-1) x
    h0' : LT.lt 0 (Neg.neg x)
    h1' : LT.lt (Neg.neg x) 1
    ⊢ LT.lt (Real.log (Neg.neg x)) 0
  -/
  exact log_neg h0' h1'
  /-
    🎉 no goals
  -/


                                                              /-
                                                                x : Real
                                                                hx : LT.lt 0 x
                                                                ⊢ Iff (LE.le 0 (Real.log x)) (LE.le 1 x)
                                                              -/
theorem log_nonneg_iff (hx : 0 < x) : 0 ≤ log x ↔ 1 ≤ x := by rw [← not_lt, log_neg_iff hx, not_lt]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[bound]
theorem log_nonneg (hx : 1 ≤ x) : 0 ≤ log x :=
  (log_nonneg_iff (zero_lt_one.trans_le hx)).2 hx


                                                              /-
                                                                x : Real
                                                                hx : LT.lt 0 x
                                                                ⊢ Iff (LE.le (Real.log x) 0) (LE.le x 1)
                                                              -/
theorem log_nonpos_iff (hx : 0 < x) : log x ≤ 0 ↔ x ≤ 1 := by rw [← not_lt, log_pos_iff hx, not_lt]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem log_nonpos_iff' (hx : 0 ≤ x) : log x ≤ 0 ↔ x ≤ 1 := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ Iff (LE.le (Real.log x) 0) (LE.le x 1)
  -/
  rcases hx.eq_or_lt with (rfl | hx)
    /-
      case inl
      hx : LE.le 0 0
      ⊢ Iff (LE.le (Real.log 0) 0) (LE.le 0 1)
    -/
  · simp [le_refl, zero_le_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : Real
    hx✝ : LE.le 0 x
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (Real.log x) 0) (LE.le x 1)
  -/
  exact log_nonpos_iff hx
  /-
    🎉 no goals
  -/


@[bound]
theorem log_nonpos (hx : 0 ≤ x) (h'x : x ≤ 1) : log x ≤ 0 :=
  (log_nonpos_iff' hx).2 h'x


theorem log_natCast_nonneg (n : ℕ) : 0 ≤ log n := by
  if hn : n = 0 then
    simp [hn]
  else
    have : (1 : ℝ) ≤ n := mod_cast Nat.one_le_of_lt <| Nat.pos_of_ne_zero hn
    exact log_nonneg this


@[deprecated (since := "2024-04-17")]
alias log_nat_cast_nonneg := log_natCast_nonneg


theorem log_neg_natCast_nonneg (n : ℕ) : 0 ≤ log (-n) := by
  /-
    n : Nat
    ⊢ LE.le 0 (Real.log (Neg.neg ↑n))
  -/
  rw [← log_neg_eq_log, neg_neg]
  /-
    n : Nat
    ⊢ LE.le 0 (Real.log ↑n)
  -/
  exact log_natCast_nonneg _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias log_neg_nat_cast_nonneg := log_neg_natCast_nonneg


theorem log_intCast_nonneg (n : ℤ) : 0 ≤ log n := by
  cases lt_trichotomy 0 n with
  | inl hn =>
      have : (1 : ℝ) ≤ n := mod_cast hn
      exact log_nonneg this
  | inr hn =>
      cases hn with
      | inl hn => simp [hn.symm]
      | inr hn =>
          have : (1 : ℝ) ≤ -n := by rw [← neg_zero, ← lt_neg] at hn; exact mod_cast hn
          rw [← log_neg_eq_log]
          exact log_nonneg this


@[deprecated (since := "2024-04-17")]
alias log_int_cast_nonneg := log_intCast_nonneg


theorem strictMonoOn_log : StrictMonoOn log (Set.Ioi 0) := fun _ hx _ _ hxy => log_lt_log hx hxy


theorem strictAntiOn_log : StrictAntiOn log (Set.Iio 0) := by
  /-
    ⊢ StrictAntiOn Real.log (Set.Iio 0)
  -/
  rintro x (hx : x < 0) y (hy : y < 0) hxy
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.log y) (Real.log x)
  -/
  rw [← log_abs y, ← log_abs x]
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (Real.log (abs y)) (Real.log (abs x))
  -/
  refine log_lt_log (abs_pos.2 hy.ne) ?_
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : LT.lt x y
    ⊢ LT.lt (abs y) (abs x)
  -/
  rwa [abs_of_neg hy, abs_of_neg hx, neg_lt_neg_iff]
  /-
    🎉 no goals
  -/


theorem log_injOn_pos : Set.InjOn log (Set.Ioi 0) :=
  strictMonoOn_log.injOn


theorem log_lt_sub_one_of_pos (hx1 : 0 < x) (hx2 : x ≠ 1) : log x < x - 1 := by
  have h : log x ≠ 0 := by
    rwa [← log_one, log_injOn_pos.ne_iff hx1]
    exact mem_Ioi.mpr zero_lt_one
  /-
    x : Real
    hx1 : LT.lt 0 x
    hx2 : Ne x 1
    h : Ne (Real.log x) 0
    ⊢ LT.lt (Real.log x) (HSub.hSub x 1)
  -/
  linarith [add_one_lt_exp h, exp_log hx1]
  /-
    🎉 no goals
  -/


theorem eq_one_of_pos_of_log_eq_zero {x : ℝ} (h₁ : 0 < x) (h₂ : log x = 0) : x = 1 :=
  log_injOn_pos (Set.mem_Ioi.2 h₁) (Set.mem_Ioi.2 zero_lt_one) (h₂.trans Real.log_one.symm)


theorem log_ne_zero_of_pos_of_ne_one {x : ℝ} (hx_pos : 0 < x) (hx : x ≠ 1) : log x ≠ 0 :=
  mt (eq_one_of_pos_of_log_eq_zero hx_pos) hx


@[simp]
theorem log_eq_zero {x : ℝ} : log x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  /-
    x : Real
    ⊢ Iff (Eq (Real.log x) 0) (Or (Eq x 0) (Or (Eq x 1) (Eq x (-1))))
  -/
  constructor
    /-
      case mp
      x : Real
      ⊢ Eq (Real.log x) 0 → Or (Eq x 0) (Or (Eq x 1) (Eq x (-1)))
    -/
  · intro h
    /-
      case mp
      x : Real
      h : Eq (Real.log x) 0
      ⊢ Or (Eq x 0) (Or (Eq x 1) (Eq x (-1)))
    -/
    rcases lt_trichotomy x 0 with (x_lt_zero | rfl | x_gt_zero)
      /-
        case mp.inl
        x : Real
        h : Eq (Real.log x) 0
        x_lt_zero : LT.lt x 0
        ⊢ Or (Eq x 0) (Or (Eq x 1) (Eq x (-1)))
      -/
    · refine Or.inr (Or.inr (neg_eq_iff_eq_neg.mp ?_))
      /-
        case mp.inl
        x : Real
        h : Eq (Real.log x) 0
        x_lt_zero : LT.lt x 0
        ⊢ Eq (Neg.neg x) 1
      -/
      rw [← log_neg_eq_log x] at h
      /-
        case mp.inl
        x : Real
        h : Eq (Real.log (Neg.neg x)) 0
        x_lt_zero : LT.lt x 0
        ⊢ Eq (Neg.neg x) 1
      -/
      exact eq_one_of_pos_of_log_eq_zero (neg_pos.mpr x_lt_zero) h
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inl
        h : Eq (Real.log 0) 0
        ⊢ Or (Eq 0 0) (Or (Eq 0 1) (Eq 0 (-1)))
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inr
        x : Real
        h : Eq (Real.log x) 0
        x_gt_zero : LT.lt 0 x
        ⊢ Or (Eq x 0) (Or (Eq x 1) (Eq x (-1)))
      -/
    · exact Or.inr (Or.inl (eq_one_of_pos_of_log_eq_zero x_gt_zero h))
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x : Real
      ⊢ Or (Eq x 0) (Or (Eq x 1) (Eq x (-1))) → Eq (Real.log x) 0
    -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  · rintro (rfl | rfl | rfl) <;> simp only [log_one, log_zero, log_neg_eq_log]
                                 /-
                                   🎉 no goals
                                 -/


theorem log_ne_zero {x : ℝ} : log x ≠ 0 ↔ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 := by
  /-
    x : Real
    ⊢ Iff (Ne (Real.log x) 0) (And (Ne x 0) (And (Ne x 1) (Ne x (-1))))
  -/
  simpa only [not_or] using log_eq_zero.not
  /-
    🎉 no goals
  -/


@[simp]
theorem log_pow (x : ℝ) (n : ℕ) : log (x ^ n) = n * log x := by
  induction n with
  | zero => simp
  | succ n ih =>
    rcases eq_or_ne x 0 with (rfl | hx)
    · simp
    · rw [pow_succ, log_mul (pow_ne_zero _ hx) hx, ih, Nat.cast_succ, add_mul, one_mul]


@[simp]
theorem log_zpow (x : ℝ) (n : ℤ) : log (x ^ n) = n * log x := by
  /-
    x : Real
    n : Int
    ⊢ Eq (Real.log (HPow.hPow x n)) (HMul.hMul (↑n) (Real.log x))
  -/
  induction n
    /-
      case ofNat
      x : Real
      a✝ : Nat
      ⊢ Eq (Real.log (HPow.hPow x (Int.ofNat a✝))) (HMul.hMul (↑(Int.ofNat a✝)) (Rea …
    -/
  · rw [Int.ofNat_eq_coe, zpow_natCast, log_pow, Int.cast_natCast]
    /-
      🎉 no goals
    -/
  /-
    case negSucc
    x : Real
    a✝ : Nat
    ⊢ Eq (Real.log (HPow.hPow x (Int.negSucc a✝))) (HMul.hMul (↑(Int.negSucc a✝))  …
  -/
  rw [zpow_negSucc, log_inv, log_pow, Int.cast_negSucc, Nat.cast_add_one, neg_mul_eq_neg_mul]
  /-
    🎉 no goals
  -/


theorem log_sqrt {x : ℝ} (hx : 0 ≤ x) : log (√x) = log x / 2 := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ Eq (Real.log x.sqrt) (HDiv.hDiv (Real.log x) 2)
  -/
  rw [eq_div_iff, mul_comm, ← Nat.cast_two, ← log_pow, sq_sqrt hx]
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ Ne 2 0
  -/
  exact two_ne_zero
  /-
    🎉 no goals
  -/


theorem log_le_sub_one_of_pos {x : ℝ} (hx : 0 < x) : log x ≤ x - 1 := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LE.le (Real.log x) (HSub.hSub x 1)
  -/
  rw [le_sub_iff_add_le]
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LE.le (HAdd.hAdd (Real.log x) 1) x
  -/
  convert add_one_le_exp (log x)
  /-
    case h.e'_4
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq x (Real.exp (Real.log x))
  -/
  rw [exp_log hx]
  /-
    🎉 no goals
  -/


lemma one_sub_inv_le_log_of_pos (hx : 0 < x) : 1 - x⁻¹ ≤ log x := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LE.le (HSub.hSub 1 (Inv.inv x)) (Real.log x)
  -/
  simpa [add_comm] using log_le_sub_one_of_pos (inv_pos.2 hx)
  /-
    🎉 no goals
  -/


/-- See `Real.log_le_sub_one_of_pos` for the stronger version when `x ≠ 0`. -/
lemma log_le_self (hx : 0 ≤ x) : log x ≤ x := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le (Real.log x) x
  -/
  obtain rfl | hx := hx.eq_or_lt
    /-
      case inl
      hx : LE.le 0 0
      ⊢ LE.le (Real.log 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx✝ : LE.le 0 x
      hx : LT.lt 0 x
      ⊢ LE.le (Real.log x) x
    -/
  · exact (log_le_sub_one_of_pos hx).trans (by linarith)
    /-
      🎉 no goals
    -/


/-- See `Real.one_sub_inv_le_log_of_pos` for the stronger version when `x ≠ 0`. -/
lemma neg_inv_le_log (hx : 0 ≤ x) : -x⁻¹ ≤ log x := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le (Neg.neg (Inv.inv x)) (Real.log x)
  -/
  rw [neg_le, ← log_inv]; exact log_le_self <| inv_nonneg.2 hx
                          /-
                            🎉 no goals
                          -/


/-- Bound for `|log x * x|` in the interval `(0, 1]`. -/
theorem abs_log_mul_self_lt (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 1) : |log x * x| < 1 := by
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  have : 0 < 1 / x := by simpa only [one_div, inv_pos] using h1
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    this : LT.lt 0 (HDiv.hDiv 1 x)
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  replace := log_le_sub_one_of_pos this
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    this : LE.le (Real.log (HDiv.hDiv 1 x)) (HSub.hSub (HDiv.hDiv 1 x) 1)
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  replace : log (1 / x) < 1 / x := by linarith
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    this : LT.lt (Real.log (HDiv.hDiv 1 x)) (HDiv.hDiv 1 x)
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  rw [log_div one_ne_zero h1.ne', log_one, zero_sub, lt_div_iff₀ h1] at this
  have aux : 0 ≤ -log x * x := by
    refine mul_nonneg ?_ h1.le
    rw [← log_inv]
    apply log_nonneg
    rw [← le_inv_comm₀ h1 zero_lt_one, inv_one]
    exact h2
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    this : LT.lt (HMul.hMul (Neg.neg (Real.log x)) x) 1
    aux : LE.le 0 (HMul.hMul (Neg.neg (Real.log x)) x)
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  rw [← abs_of_nonneg aux, neg_mul, abs_neg] at this
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    this : LT.lt (abs (HMul.hMul (Real.log x) x)) 1
    aux : LE.le 0 (HMul.hMul (Neg.neg (Real.log x)) x)
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) x)) 1
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- The real logarithm function tends to `+∞` at `+∞`. -/
theorem tendsto_log_atTop : Tendsto log atTop atTop :=
                                 /-
                                   ⊢ Filter.Tendsto (fun x => Real.log (Real.exp x)) Filter.atTop Filter.atTop
                                 -/
  tendsto_comp_exp_atTop.1 <| by simpa only [log_exp] using tendsto_id
                                 /-
                                   🎉 no goals
                                 -/


theorem tendsto_log_nhdsWithin_zero : Tendsto log (𝓝[≠] 0) atBot := by
  /-
    ⊢ Filter.Tendsto Real.log (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0 …
  -/
  rw [← show _ = log from funext log_abs]
  /-
    ⊢ Filter.Tendsto (fun x => Real.log (abs x)) (nhdsWithin 0 (HasCompl.compl (Si …
  -/
  refine Tendsto.comp (g := log) ?_ tendsto_abs_nhdsWithin_zero
  /-
    ⊢ Filter.Tendsto Real.log (nhdsWithin 0 (Set.Ioi 0)) Filter.atBot
  -/
  simpa [← tendsto_comp_exp_atBot] using tendsto_id
  /-
    🎉 no goals
  -/


lemma tendsto_log_nhdsWithin_zero_right : Tendsto log (𝓝[>] 0) atBot :=
  tendsto_log_nhdsWithin_zero.mono_left <| nhdsWithin_mono _ fun _ h ↦ ne_of_gt h


theorem continuousOn_log : ContinuousOn log {0}ᶜ := by
  simp (config := { unfoldPartialApp := true }) only [continuousOn_iff_continuous_restrict,
    restrict]
  /-
    ⊢ Continuous fun x => Real.log ↑x
  -/
  conv in log _ => rw [log_of_ne_zero (show (x : ℝ) ≠ 0 from x.2)]
  /-
    ⊢ Continuous fun x => Real.expOrderIso.symm ⟨abs ↑x, ⋯⟩
  -/
  exact expOrderIso.symm.continuous.comp (continuous_subtype_val.norm.subtype_mk _)
  /-
    🎉 no goals
  -/


/-- The real logarithm is continuous as a function from nonzero reals. -/
@[fun_prop]
theorem continuous_log : Continuous fun x : { x : ℝ // x ≠ 0 } => log x :=
  continuousOn_iff_continuous_restrict.1 <| continuousOn_log.mono fun _ => id


/-- The real logarithm is continuous as a function from positive reals. -/
@[fun_prop]
theorem continuous_log' : Continuous fun x : { x : ℝ // 0 < x } => log x :=
  continuousOn_iff_continuous_restrict.1 <| continuousOn_log.mono fun _ hx => ne_of_gt hx


theorem continuousAt_log (hx : x ≠ 0) : ContinuousAt log x :=
  (continuousOn_log x hx).continuousAt <| isOpen_compl_singleton.mem_nhds hx


@[simp]
theorem continuousAt_log_iff : ContinuousAt log x ↔ x ≠ 0 := by
  /-
    x : Real
    ⊢ Iff (ContinuousAt Real.log x) (Ne x 0)
  -/
  refine ⟨?_, continuousAt_log⟩
  /-
    x : Real
    ⊢ ContinuousAt Real.log x → Ne x 0
  -/
  rintro h rfl
  exact not_tendsto_nhds_of_tendsto_atBot tendsto_log_nhdsWithin_zero _
    (h.tendsto.mono_left inf_le_left)


theorem log_prod {α : Type*} (s : Finset α) (f : α → ℝ) (hf : ∀ x ∈ s, f x ≠ 0) :
    log (∏ i ∈ s, f i) = ∑ i ∈ s, log (f i) := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Real
    hf : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    ⊢ Eq (Real.log (s.prod fun i => f i)) (s.sum fun i => Real.log (f i))
  -/
  induction' s using Finset.cons_induction_on with a s ha ih
    /-
      case h₁
      α : Type u_1
      f : α → Real
      hf : ∀ (x : α), Membership.mem EmptyCollection.emptyCollection x → Ne (f x) 0
      ⊢ Eq (Real.log (EmptyCollection.emptyCollection.prod fun i => f i)) (EmptyColl …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      f : α → Real
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : (∀ (x : α), Membership.mem s x → Ne (f x) 0) → Eq (Real.log (s.prod fun i …
      hf : ∀ (x : α), Membership.mem (Finset.cons a s ha) x → Ne (f x) 0
      ⊢ Eq (Real.log ((Finset.cons a s ha).prod fun i => f i)) ((Finset.cons a s ha) …
    -/
  · rw [Finset.forall_mem_cons] at hf
    /-
      case h₂
      α : Type u_1
      f : α → Real
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : (∀ (x : α), Membership.mem s x → Ne (f x) 0) → Eq (Real.log (s.prod fun i …
      hf : And (Ne (f a) 0) (∀ (x : α), Membership.mem s x → Ne (f x) 0)
      ⊢ Eq (Real.log ((Finset.cons a s ha).prod fun i => f i)) ((Finset.cons a s ha) …
    -/
    simp [ih hf.2, log_mul hf.1 (Finset.prod_ne_zero_iff.2 hf.2)]
    /-
      🎉 no goals
    -/


protected theorem _root_.Finsupp.log_prod {α β : Type*} [Zero β] (f : α →₀ β) (g : α → β → ℝ)
    (hg : ∀ a, g a (f a) = 0 → f a = 0) : log (f.prod g) = f.sum fun a b ↦ log (g a b) :=
  log_prod _ _ fun _x hx h₀ ↦ Finsupp.mem_support_iff.1 hx <| hg _ h₀


theorem log_nat_eq_sum_factorization (n : ℕ) :
    log n = n.factorization.sum fun p t => t * log p := by
  /-
    n : Nat
    ⊢ Eq (Real.log ↑n) (n.factorization.sum fun p t => HMul.hMul (↑t) (Real.log ↑p))
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      ⊢ Eq (Real.log ↑0) ((Nat.factorization 0).sum fun p t => HMul.hMul (↑t) (Real. …
    -/
  · simp -- relies on junk values of `log` and `Nat.factorization`
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : Ne n 0
      ⊢ Eq (Real.log ↑n) (n.factorization.sum fun p t => HMul.hMul (↑t) (Real.log ↑p))
    -/
  · simp only [← log_pow, ← Nat.cast_pow]
    /-
      case inr
      n : Nat
      hn : Ne n 0
      ⊢ Eq (Real.log ↑n) (n.factorization.sum fun p t => Real.log ↑(HPow.hPow p t))
    -/
    rw [← Finsupp.log_prod, ← Nat.cast_finsupp_prod, Nat.factorization_prod_pow_eq_self hn]
    /-
      case inr.hg
      n : Nat
      hn : Ne n 0
      ⊢ ∀ (a : Nat), Eq (↑(HPow.hPow a (n.factorization a))) 0 → Eq (n.factorization …
    -/
    intro p hp
    /-
      case inr.hg
      n : Nat
      hn : Ne n 0
      p : Nat
      hp : Eq (↑(HPow.hPow p (n.factorization p))) 0
      ⊢ Eq (n.factorization p) 0
    -/
    rw [pow_eq_zero (Nat.cast_eq_zero.1 hp), Nat.factorization_zero_right]
    /-
      🎉 no goals
    -/


theorem tendsto_pow_log_div_mul_add_atTop (a b : ℝ) (n : ℕ) (ha : a ≠ 0) :
    Tendsto (fun x => log x ^ n / (a * x + b)) atTop (𝓝 0) :=
  ((tendsto_div_pow_mul_exp_add_atTop a b n ha.symm).comp tendsto_log_atTop).congr' <| by
    /-
      a b : Real
      n : Nat
      ha : Ne a 0
      ⊢ Filter.atTop.EventuallyEq (Function.comp (fun x => HDiv.hDiv (HPow.hPow x n) …
    -/
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with x hx using by simp [exp_log hx]
    /-
      🎉 no goals
    -/


theorem isLittleO_pow_log_id_atTop {n : ℕ} : (fun x => log x ^ n) =o[atTop] id := by
  /-
    n : Nat
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HPow.hPow (Real.log x) n) id
  -/
  rw [Asymptotics.isLittleO_iff_tendsto']
    /-
      n : Nat
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow (Real.log x) n) (id x)) Filter …
    -/
  · simpa using tendsto_pow_log_div_mul_add_atTop 1 0 n one_ne_zero
    /-
      🎉 no goals
    -/
  /-
    n : Nat
    ⊢ Filter.Eventually (fun x => Eq (id x) 0 → Eq (HPow.hPow (Real.log x) n) 0) F …
  -/
  filter_upwards [eventually_ne_atTop (0 : ℝ)] with x h₁ h₂ using (h₁ h₂).elim
  /-
    🎉 no goals
  -/


theorem isLittleO_log_id_atTop : log =o[atTop] id :=
  isLittleO_pow_log_id_atTop.congr_left fun _ => pow_one _


theorem isLittleO_const_log_atTop {c : ℝ} : (fun _ => c) =o[atTop] log := by
  refine Asymptotics.isLittleO_of_tendsto' ?_
    <| Tendsto.div_atTop (a := c) (by simp) tendsto_log_atTop
  /-
    c : Real
    ⊢ Filter.Eventually (fun x => Eq (Real.log x) 0 → Eq c 0) Filter.atTop
  -/
  filter_upwards [eventually_gt_atTop 1] with x hx
  /-
    case h
    c x : Real
    hx : LT.lt 1 x
    ⊢ Eq (Real.log x) 0 → Eq c 0
  -/
  aesop (add safe forward log_pos)
  /-
    🎉 no goals
  -/


/-- `Real.exp` as a `PartialHomeomorph` with `source = univ` and `target = {z | 0 < z}`. -/
@[simps] noncomputable def expPartialHomeomorph : PartialHomeomorph ℝ ℝ where
  toFun := Real.exp
  invFun := Real.log
  source := univ
  target := Ioi (0 : ℝ)
  map_source' x _ := exp_pos x
  map_target' _ _ := mem_univ _
                      /-
                        x y x✝¹ : Real
                        x✝ : Membership.mem Set.univ x✝¹
                        ⊢ Eq (Real.log (Real.exp x✝¹)) x✝¹
                      -/
  left_inv' _ _ := by simp
                      /-
                        🎉 no goals
                      -/
  right_inv' _ hx := exp_log hx
  open_source := isOpen_univ
  open_target := isOpen_Ioi
  continuousOn_toFun := continuousOn_exp
  continuousOn_invFun x hx := (continuousAt_log (ne_of_gt hx)).continuousWithinAt


theorem Filter.Tendsto.log {f : α → ℝ} {l : Filter α} {x : ℝ} (h : Tendsto f l (𝓝 x)) (hx : x ≠ 0) :
    Tendsto (fun x => log (f x)) l (𝓝 (log x)) :=
  (continuousAt_log hx).tendsto.comp h


@[fun_prop]
theorem Continuous.log (hf : Continuous f) (h₀ : ∀ x, f x ≠ 0) : Continuous fun x => log (f x) :=
  continuousOn_log.comp_continuous hf h₀


@[fun_prop]
nonrec theorem ContinuousAt.log (hf : ContinuousAt f a) (h₀ : f a ≠ 0) :
    ContinuousAt (fun x => log (f x)) a :=
  hf.log h₀


nonrec theorem ContinuousWithinAt.log (hf : ContinuousWithinAt f s a) (h₀ : f a ≠ 0) :
    ContinuousWithinAt (fun x => log (f x)) s a :=
  hf.log h₀


@[fun_prop]
theorem ContinuousOn.log (hf : ContinuousOn f s) (h₀ : ∀ x ∈ s, f x ≠ 0) :
    ContinuousOn (fun x => log (f x)) s := fun x hx => (hf x hx).log (h₀ x hx)


theorem tendsto_log_comp_add_sub_log (y : ℝ) :
    Tendsto (fun x : ℝ => log (x + y) - log x) atTop (𝓝 0) := by
  have : Tendsto (fun x ↦ 1 + y / x) atTop (𝓝 (1 + 0)) :=
    tendsto_const_nhds.add (tendsto_const_nhds.div_atTop tendsto_id)
  /-
    y : Real
    this : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv y x)) Filter.atTop (nhd …
    ⊢ Filter.Tendsto (fun x => HSub.hSub (Real.log (HAdd.hAdd x y)) (Real.log x))  …
  -/
  rw [← comap_exp_nhds_exp, exp_zero, tendsto_comap_iff, ← add_zero (1 : ℝ)]
  /-
    y : Real
    this : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv y x)) Filter.atTop (nhd …
    ⊢ Filter.Tendsto (Function.comp Real.exp fun x => HSub.hSub (Real.log (HAdd.hA …
  -/
  refine this.congr' ?_
  /-
    y : Real
    this : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv y x)) Filter.atTop (nhd …
    ⊢ Filter.atTop.EventuallyEq (fun x => HAdd.hAdd 1 (HDiv.hDiv y x)) (Function.c …
  -/
  filter_upwards [eventually_gt_atTop (0 : ℝ), eventually_gt_atTop (-y)] with x hx₀ hxy
  /-
    case h
    y : Real
    this : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv y x)) Filter.atTop (nhd …
    x : Real
    hx₀ : LT.lt 0 x
    hxy : LT.lt (Neg.neg y) x
    ⊢ Eq (HAdd.hAdd 1 (HDiv.hDiv y x)) (Function.comp Real.exp (fun x => HSub.hSub …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  rw [comp_apply, exp_sub, exp_log, exp_log, one_add_div] <;> linarith
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem tendsto_log_nat_add_one_sub_log : Tendsto (fun k : ℕ => log (k + 1) - log k) atTop (𝓝 0) :=
  (tendsto_log_comp_add_sub_log 1).comp tendsto_natCast_atTop_atTop


lemma log_nonneg_of_isNat {n : ℕ} (h : NormNum.IsNat e n) : 0 ≤ Real.log (e : ℝ) := by
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsNat e n
    ⊢ LE.le 0 (Real.log e)
  -/
  rw [NormNum.IsNat.to_eq h rfl]
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsNat e n
    ⊢ LE.le 0 (Real.log ↑n)
  -/
  exact Real.log_natCast_nonneg _
  /-
    🎉 no goals
  -/


lemma log_pos_of_isNat {n : ℕ} (h : NormNum.IsNat e n) (w : Nat.blt 1 n = true) :
    0 < Real.log (e : ℝ) := by
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsNat e n
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 0 (Real.log e)
  -/
  rw [NormNum.IsNat.to_eq h rfl]
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsNat e n
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 0 (Real.log ↑n)
  -/
  apply Real.log_pos
  /-
    case hx
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsNat e n
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 1 ↑n
  -/
  simpa using w
  /-
    🎉 no goals
  -/


lemma log_nonneg_of_isNegNat {n : ℕ} (h : NormNum.IsInt e (.negOfNat n)) :
    0 ≤ Real.log (e : ℝ) := by
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    ⊢ LE.le 0 (Real.log e)
  -/
  rw [NormNum.IsInt.neg_to_eq h rfl]
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    ⊢ LE.le 0 (Real.log (Neg.neg ↑n))
  -/
  exact Real.log_neg_natCast_nonneg _
  /-
    🎉 no goals
  -/


lemma log_pos_of_isNegNat {n : ℕ} (h : NormNum.IsInt e (.negOfNat n)) (w : Nat.blt 1 n = true) :
    0 < Real.log (e : ℝ) := by
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 0 (Real.log e)
  -/
  rw [NormNum.IsInt.neg_to_eq h rfl]
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 0 (Real.log (Neg.neg ↑n))
  -/
  rw [Real.log_neg_eq_log]
  /-
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 0 (Real.log ↑n)
  -/
  apply Real.log_pos
  /-
    case hx
    e : Real
    n : Nat
    h : Mathlib.Meta.NormNum.IsInt e (Int.negOfNat n)
    w : Eq (Nat.blt 1 n) Bool.true
    ⊢ LT.lt 1 ↑n
  -/
  simpa using w
  /-
    🎉 no goals
  -/


lemma log_pos_of_isRat {n : ℤ} :
    (NormNum.IsRat e n d) → (decide ((1 : ℚ) < n / d)) → (0 < Real.log (e : ℝ))
  | ⟨inv, eq⟩, h => by
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt 1 (HDiv.hDiv ↑n ↑d))) Bool.true
      ⊢ LT.lt 0 (Real.log e)
    -/
    rw [eq, invOf_eq_inv, ← div_eq_mul_inv]
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt 1 (HDiv.hDiv ↑n ↑d))) Bool.true
      ⊢ LT.lt 0 (Real.log (HDiv.hDiv ↑n ↑d))
    -/
    have : 1 < (n : ℝ) / d := by exact_mod_cast of_decide_eq_true h
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt 1 (HDiv.hDiv ↑n ↑d))) Bool.true
      this : LT.lt 1 (HDiv.hDiv ↑n ↑d)
      ⊢ LT.lt 0 (Real.log (HDiv.hDiv ↑n ↑d))
    -/
    exact Real.log_pos this
    /-
      🎉 no goals
    -/


lemma log_pos_of_isRat_neg {n : ℤ} :
    (NormNum.IsRat e n d) → (decide (n / d < (-1 : ℚ))) → (0 < Real.log (e : ℝ))
  | ⟨inv, eq⟩, h => by
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) (-1))) Bool.true
      ⊢ LT.lt 0 (Real.log e)
    -/
    rw [eq, invOf_eq_inv, ← div_eq_mul_inv]
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) (-1))) Bool.true
      ⊢ LT.lt 0 (Real.log (HDiv.hDiv ↑n ↑d))
    -/
    have : (n : ℝ) / d < -1 := by exact_mod_cast of_decide_eq_true h
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) (-1))) Bool.true
      this : LT.lt (HDiv.hDiv ↑n ↑d) (-1)
      ⊢ LT.lt 0 (Real.log (HDiv.hDiv ↑n ↑d))
    -/
    exact Real.log_pos_of_lt_neg_one this
    /-
      🎉 no goals
    -/


lemma log_nz_of_isRat {n : ℤ} : (NormNum.IsRat e n d) → (decide ((0 : ℚ) < n / d))
    → (decide (n / d < (1 : ℚ))) → (Real.log (e : ℝ) ≠ 0)
  | ⟨inv, eq⟩, h₁, h₂ => by
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt 0 (HDiv.hDiv ↑n ↑d))) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 1)) Bool.true
      ⊢ Ne (Real.log e) 0
    -/
    rw [eq, invOf_eq_inv, ← div_eq_mul_inv]
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt 0 (HDiv.hDiv ↑n ↑d))) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 1)) Bool.true
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    have h₁' : 0 < (n : ℝ) / d := by exact_mod_cast of_decide_eq_true h₁
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt 0 (HDiv.hDiv ↑n ↑d))) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 1)) Bool.true
      h₁' : LT.lt 0 (HDiv.hDiv ↑n ↑d)
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    have h₂' : (n : ℝ) / d < 1 := by exact_mod_cast of_decide_eq_true h₂
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt 0 (HDiv.hDiv ↑n ↑d))) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 1)) Bool.true
      h₁' : LT.lt 0 (HDiv.hDiv ↑n ↑d)
      h₂' : LT.lt (HDiv.hDiv ↑n ↑d) 1
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    exact ne_of_lt <| Real.log_neg h₁' h₂'
    /-
      🎉 no goals
    -/


lemma log_nz_of_isRat_neg {n : ℤ} : (NormNum.IsRat e n d) → (decide (n / d < (0 : ℚ)))
    → (decide ((-1 : ℚ) < n / d)) → (Real.log (e : ℝ) ≠ 0)
  | ⟨inv, eq⟩, h₁, h₂ => by
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 0)) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (-1) (HDiv.hDiv ↑n ↑d))) Bool.true
      ⊢ Ne (Real.log e) 0
    -/
    rw [eq, invOf_eq_inv, ← div_eq_mul_inv]
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 0)) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (-1) (HDiv.hDiv ↑n ↑d))) Bool.true
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    have h₁' : (n : ℝ) / d < 0 := by exact_mod_cast of_decide_eq_true h₁
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 0)) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (-1) (HDiv.hDiv ↑n ↑d))) Bool.true
      h₁' : LT.lt (HDiv.hDiv ↑n ↑d) 0
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    have h₂' : -1 < (n : ℝ) / d := by exact_mod_cast of_decide_eq_true h₂
    /-
      e : Real
      d : Nat
      n : Int
      inv : Invertible ↑d
      eq : Eq e (HMul.hMul (↑n) (Invertible.invOf ↑d))
      h₁ : Eq (Decidable.decide (LT.lt (HDiv.hDiv ↑n ↑d) 0)) Bool.true
      h₂ : Eq (Decidable.decide (LT.lt (-1) (HDiv.hDiv ↑n ↑d))) Bool.true
      h₁' : LT.lt (HDiv.hDiv ↑n ↑d) 0
      h₂' : LT.lt (-1) (HDiv.hDiv ↑n ↑d)
      ⊢ Ne (Real.log (HDiv.hDiv ↑n ↑d)) 0
    -/
    exact ne_of_lt <| Real.log_neg_of_lt_zero h₁' h₂'
    /-
      🎉 no goals
    -/


/-- Extension for the `positivity` tactic: `Real.log` of a natural number is always nonnegative. -/
@[positivity Real.log (Nat.cast _)]
def evalLogNatCast : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Real.log (Nat.cast $a)) =>
    assertInstancesCommute
    pure (.nonnegative q(Real.log_natCast_nonneg $a))
  | _, _, _ => throwError "not Real.log"


/-- Extension for the `positivity` tactic: `Real.log` of an integer is always nonnegative. -/
@[positivity Real.log (Int.cast _)]
def evalLogIntCast : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Real.log (Int.cast $a)) =>
    assertInstancesCommute
    pure (.nonnegative q(Real.log_intCast_nonneg $a))
  | _, _, _ => throwError "not Real.log"


/-- Extension for the `positivity` tactic: `Real.log` of a numeric literal. -/
@[positivity Real.log _]
def evalLogNatLit : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Real.log $a) =>
    match ← NormNum.derive a with
    | .isNat (_ : Q(AddMonoidWithOne ℝ)) lit p =>
      assumeInstancesCommute
      have p : Q(NormNum.IsNat $a $lit) := p
      if 1 < lit.natLit! then
        let p' : Q(Nat.blt 1 $lit = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.positive q(log_pos_of_isNat $p $p'))
      else
        pure (.nonnegative q(log_nonneg_of_isNat $p))
    | .isNegNat _ lit p =>
      assumeInstancesCommute
      have p : Q(NormNum.IsInt $a (Int.negOfNat $lit)) := p
      if 1 < lit.natLit! then
        let p' : Q(Nat.blt 1 $lit = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.positive q(log_pos_of_isNegNat $p $p'))
      else
        pure (.nonnegative q(log_nonneg_of_isNegNat $p))
    | .isRat (i : Q(DivisionRing ℝ)) q n d p =>
      assumeInstancesCommute
                    /-
                      $d✝ : Nat
                      $x✝³ : Bool
                      «$α» : Type := Real
                      $x✝² : Prod Lean.Expr Bool
                      «$a» : Real
                      «$e» : «$α» := Real.log «$a»
                      «$n» : Int
                      «$d» : Nat
                      $x✝¹ : Zero «$α» := Real.instZero
                      $x✝ : PartialOrder «$α» := Real.partialOrder
                      «$i» : DivisionRing Real := Real.instDivisionRing
                      «$p» : Mathlib.Meta.NormNum.IsRat «$a» «$n» «$d»
                      ⊢ Sort ?u.98700
                    -/
      have p : Q(by clear! «$i»; exact NormNum.IsRat $a $n $d) := p
                                 /-
                                   🎉 no goals
                                 -/
      if 0 < q ∧ q < 1 then
        let w₁ : Q(decide ((0 : ℚ) < $n / $d) = true) := (q(Eq.refl true) : Lean.Expr)
        let w₂ : Q(decide ($n / $d < (1 : ℚ)) = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.nonzero q(log_nz_of_isRat $p $w₁ $w₂))
      else if 1 < q then
        let w : Q(decide ((1 : ℚ) < $n / $d) = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.positive q(log_pos_of_isRat $p $w))
      else if -1 < q ∧ q < 0 then
        let w₁ : Q(decide ($n / $d < (0 : ℚ)) = true) := (q(Eq.refl true) : Lean.Expr)
        let w₂ : Q(decide ((-1 : ℚ) < $n / $d) = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.nonzero q(log_nz_of_isRat_neg $p $w₁ $w₂))
      else if q < -1 then
        let w : Q(decide ($n / $d < (-1 : ℚ)) = true) := (q(Eq.refl true) : Lean.Expr)
        pure (.positive q(log_pos_of_isRat_neg $p $w))
      else
        failure
    | _ => failure
  | _, _, _ => throwError "not Real.log"


