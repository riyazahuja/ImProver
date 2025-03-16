/-- For 0 < x, we have sin x < x. -/
theorem sin_lt (h : 0 < x) : sin x < x := by
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ LT.lt (Real.sin x) x
  -/
  cases' lt_or_le 1 x with h' h'
    /-
      case inl
      x : Real
      h : LT.lt 0 x
      h' : LT.lt 1 x
      ⊢ LT.lt (Real.sin x) x
    -/
  · exact (sin_le_one x).trans_lt h'
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    ⊢ LT.lt (Real.sin x) x
  -/
  have hx : |x| = x := abs_of_nonneg h.le
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    ⊢ LT.lt (Real.sin x) x
  -/
  have := le_of_abs_le (sin_bound <| show |x| ≤ 1 by rwa [hx])
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (HSub.hSub (Real.sin x) (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LT.lt (Real.sin x) x
  -/
  rw [sub_le_iff_le_add', hx] at this
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Real.sin x) (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LT.lt (Real.sin x) x
  -/
  apply this.trans_lt
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Real.sin x) (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LT.lt (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6)) (HMul.hMul (HPo …
  -/
  rw [sub_add, sub_lt_self_iff, sub_pos, div_eq_mul_inv (x ^ 3)]
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Real.sin x) (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LT.lt (HMul.hMul (HPow.hPow x 4) (5 / 96)) (HMul.hMul (HPow.hPow x 3) (Inv.i …
  -/
  refine mul_lt_mul' ?_ (by norm_num) (by norm_num) (pow_pos h 3)
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Real.sin x) (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LE.le (HPow.hPow x 4) (HPow.hPow x 3)
  -/
  apply pow_le_pow_of_le_one h.le h'
  /-
    case inr
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Real.sin x) (HAdd.hAdd (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6 …
    ⊢ LE.le 3 4
  -/
  norm_num
  /-
    🎉 no goals
  -/


lemma sin_le (hx : 0 ≤ x) : sin x ≤ x := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le (Real.sin x) x
  -/
  obtain rfl | hx := hx.eq_or_lt
    /-
      case inl
      hx : LE.le 0 0
      ⊢ LE.le (Real.sin 0) 0
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
      ⊢ LE.le (Real.sin x) x
    -/
  · exact (sin_lt hx).le
    /-
      🎉 no goals
    -/


                                            /-
                                              x : Real
                                              hx : LT.lt x 0
                                              ⊢ LT.lt x (Real.sin x)
                                            -/
lemma lt_sin (hx : x < 0) : x < sin x := by simpa using sin_lt <| neg_pos.2 hx
                                            /-
                                              🎉 no goals
                                            -/

                                            /-
                                              x : Real
                                              hx : LE.le x 0
                                              ⊢ LE.le x (Real.sin x)
                                            -/
lemma le_sin (hx : x ≤ 0) : x ≤ sin x := by simpa using sin_le <| neg_nonneg.2 hx
                                            /-
                                              🎉 no goals
                                            -/


theorem lt_sin_mul {x : ℝ} (hx : 0 < x) (hx' : x < 1) : x < sin (π / 2 * x) := by
  simpa [mul_comm x] using
    strictConcaveOn_sin_Icc.2 ⟨le_rfl, pi_pos.le⟩ ⟨pi_div_two_pos.le, half_le_self pi_pos.le⟩
      pi_div_two_pos.ne (sub_pos.2 hx') hx


theorem le_sin_mul {x : ℝ} (hx : 0 ≤ x) (hx' : x ≤ 1) : x ≤ sin (π / 2 * x) := by
  simpa [mul_comm x] using
    strictConcaveOn_sin_Icc.concaveOn.2 ⟨le_rfl, pi_pos.le⟩
      ⟨pi_div_two_pos.le, half_le_self pi_pos.le⟩ (sub_nonneg.2 hx') hx


theorem mul_lt_sin {x : ℝ} (hx : 0 < x) (hx' : x < π / 2) : 2 / π * x < sin x := by
  /-
    x : Real
    hx : LT.lt 0 x
    hx' : LT.lt x (HDiv.hDiv Real.pi 2)
    ⊢ LT.lt (HMul.hMul (HDiv.hDiv 2 Real.pi) x) (Real.sin x)
  -/
  rw [← inv_div]
  simpa [-inv_div, mul_inv_cancel_left₀ pi_div_two_pos.ne'] using @lt_sin_mul ((π / 2)⁻¹ * x)
    (mul_pos (inv_pos.2 pi_div_two_pos) hx) (by rwa [← div_eq_inv_mul, div_lt_one pi_div_two_pos])


/-- One half of **Jordan's inequality**.

In the range `[0, π / 2]`, we have a linear lower bound on `sin`. The other half is given by
`Real.sin_le`.
-/
theorem mul_le_sin {x : ℝ} (hx : 0 ≤ x) (hx' : x ≤ π / 2) : 2 / π * x ≤ sin x := by
  /-
    x : Real
    hx : LE.le 0 x
    hx' : LE.le x (HDiv.hDiv Real.pi 2)
    ⊢ LE.le (HMul.hMul (HDiv.hDiv 2 Real.pi) x) (Real.sin x)
  -/
  rw [← inv_div]
  simpa [-inv_div, mul_inv_cancel_left₀ pi_div_two_pos.ne'] using @le_sin_mul ((π / 2)⁻¹ * x)
    (mul_nonneg (inv_nonneg.2 pi_div_two_pos.le) hx)
    (by rwa [← div_eq_inv_mul, div_le_one pi_div_two_pos])


/-- Half of **Jordan's inequality** for negative values. -/
lemma sin_le_mul (hx : -(π / 2) ≤ x) (hx₀ : x ≤ 0) : sin x ≤ 2 / π * x := by
  /-
    x : Real
    hx : LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) x
    hx₀ : LE.le x 0
    ⊢ LE.le (Real.sin x) (HMul.hMul (HDiv.hDiv 2 Real.pi) x)
  -/
  simpa using mul_le_sin (neg_nonneg.2 hx₀) (neg_le.2 hx)
  /-
    🎉 no goals
  -/


/-- Half of **Jordan's inequality** for absolute values. -/
lemma mul_abs_le_abs_sin (hx : |x| ≤ π / 2) : 2 / π * |x| ≤ |sin x| := by
  /-
    x : Real
    hx : LE.le (abs x) (HDiv.hDiv Real.pi 2)
    ⊢ LE.le (HMul.hMul (HDiv.hDiv 2 Real.pi) (abs x)) (abs (Real.sin x))
  -/
  wlog hx₀ : 0 ≤ x
  /-
    case inr
    x : Real
    hx : LE.le (abs x) (HDiv.hDiv Real.pi 2)
    this : ∀ {x : Real}, LE.le (abs x) (HDiv.hDiv Real.pi 2) → LE.le 0 x → LE.le ( …
    hx₀ : Not (LE.le 0 x)
    ⊢ LE.le (HMul.hMul (HDiv.hDiv 2 Real.pi) (abs x)) (abs (Real.sin x))
  -/
  case inr => simpa using this (by rwa [abs_neg]) <| neg_nonneg.2 <| le_of_not_le hx₀
  /-
    x✝ x : Real
    hx : LE.le (abs x) (HDiv.hDiv Real.pi 2)
    hx₀ : LE.le 0 x
    ⊢ LE.le (HMul.hMul (HDiv.hDiv 2 Real.pi) (abs x)) (abs (Real.sin x))
  -/
  rw [abs_of_nonneg hx₀] at hx ⊢
  /-
    x✝ x : Real
    hx : LE.le x (HDiv.hDiv Real.pi 2)
    hx₀ : LE.le 0 x
    ⊢ LE.le (HMul.hMul (HDiv.hDiv 2 Real.pi) x) (abs (Real.sin x))
  -/
  exact (mul_le_sin hx₀ hx).trans (le_abs_self _)
  /-
    🎉 no goals
  -/


lemma sin_sq_lt_sq (hx : x ≠ 0) : sin x ^ 2 < x ^ 2 := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ LT.lt (HPow.hPow (Real.sin x) 2) (HPow.hPow x 2)
  -/
  wlog hx₀ : 0 < x
  case inr =>
    simpa using this (neg_ne_zero.2 hx) <| neg_pos_of_neg <| hx.lt_of_le <| le_of_not_lt hx₀
  /-
    x✝ x : Real
    hx : Ne x 0
    hx₀ : LT.lt 0 x
    ⊢ LT.lt (HPow.hPow (Real.sin x) 2) (HPow.hPow x 2)
  -/
  rcases le_or_lt x 1 with hxπ | hxπ
  case inl =>
    exact pow_lt_pow_left₀ (sin_lt hx₀)
      (sin_nonneg_of_nonneg_of_le_pi hx₀.le (by linarith [two_le_pi])) (by simp)
  case inr =>
    exact (sin_sq_le_one _).trans_lt (by rwa [one_lt_sq_iff₀ hx₀.le])


lemma sin_sq_le_sq : sin x ^ 2 ≤ x ^ 2 := by
  /-
    x : Real
    ⊢ LE.le (HPow.hPow (Real.sin x) 2) (HPow.hPow x 2)
  -/
  rcases eq_or_ne x 0 with rfl | hx
  /-
    case inl
    ⊢ LE.le (HPow.hPow (Real.sin 0) 2) (HPow.hPow 0 2)
  -/
  case inl => simp
  /-
    case inr
    x : Real
    hx : Ne x 0
    ⊢ LE.le (HPow.hPow (Real.sin x) 2) (HPow.hPow x 2)
  -/
  case inr => exact (sin_sq_lt_sq hx).le
  /-
    🎉 no goals
  -/


lemma abs_sin_lt_abs (hx : x ≠ 0) : |sin x| < |x| := sq_lt_sq.1 (sin_sq_lt_sq hx)

lemma abs_sin_le_abs : |sin x| ≤ |x| := sq_le_sq.1 sin_sq_le_sq


lemma one_sub_sq_div_two_lt_cos (hx : x ≠ 0) : 1 - x ^ 2 / 2 < cos x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ LT.lt (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2)) (Real.cos x)
  -/
  have := (sin_sq_lt_sq (by positivity)).trans_eq' (sin_sq_eq_half_sub (x / 2)).symm
  /-
    x : Real
    hx : Ne x 0
    this : LT.lt (HSub.hSub (1 / 2) (HDiv.hDiv (Real.cos (HMul.hMul 2 (HDiv.hDiv x …
    ⊢ LT.lt (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2)) (Real.cos x)
  -/
  ring_nf at this
  /-
    x : Real
    hx : Ne x 0
    this : LT.lt (HAdd.hAdd (1 / 2) (HMul.hMul (Real.cos x) (-1 / 2))) (HMul.hMul  …
    ⊢ LT.lt (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2)) (Real.cos x)
  -/
  linarith
  /-
    🎉 no goals
  -/


lemma one_sub_sq_div_two_le_cos : 1 - x ^ 2 / 2 ≤ cos x := by
  /-
    x : Real
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2)) (Real.cos x)
  -/
  rcases eq_or_ne x 0 with rfl | hx
  /-
    case inl
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (HPow.hPow 0 2) 2)) (Real.cos 0)
  -/
  case inl => simp
  /-
    case inr
    x : Real
    hx : Ne x 0
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2)) (Real.cos x)
  -/
  case inr => exact (one_sub_sq_div_two_lt_cos hx).le
  /-
    🎉 no goals
  -/


/-- Half of **Jordan's inequality** for `cos`. -/
lemma one_sub_mul_le_cos (hx₀ : 0 ≤ x) (hx : x ≤ π / 2) : 1 - 2 / π * x ≤ cos x := by
  simpa [sin_pi_div_two_sub, mul_sub, div_mul_div_comm, mul_comm π, pi_pos.ne']
    using mul_le_sin (x := π / 2 - x) (by simpa) (by simpa)


/-- Half of **Jordan's inequality** for `cos` and negative values. -/
lemma one_add_mul_le_cos (hx₀ : -(π / 2) ≤ x) (hx : x ≤ 0) : 1 + 2 / π * x ≤ cos x := by
  /-
    x : Real
    hx₀ : LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) x
    hx : LE.le x 0
    ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv 2 Real.pi) x)) (Real.cos x)
  -/
  simpa using one_sub_mul_le_cos (x := -x) (by linarith) (by linarith)
  /-
    🎉 no goals
  -/


lemma cos_le_one_sub_mul_cos_sq (hx : |x| ≤ π) : cos x ≤ 1 - 2 / π ^ 2 * x ^ 2 := by
  /-
    x : Real
    hx : LE.le (abs x) Real.pi
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  wlog hx₀ : 0 ≤ x
  /-
    case inr
    x : Real
    hx : LE.le (abs x) Real.pi
    this : ∀ {x : Real}, LE.le (abs x) Real.pi → LE.le 0 x → LE.le (Real.cos x) (H …
    hx₀ : Not (LE.le 0 x)
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  case inr => simpa using this (by rwa [abs_neg]) <| neg_nonneg.2 <| le_of_not_le hx₀
  /-
    x✝ x : Real
    hx : LE.le (abs x) Real.pi
    hx₀ : LE.le 0 x
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  rw [abs_of_nonneg hx₀] at hx
  /-
    x✝ x : Real
    hx : LE.le x Real.pi
    hx₀ : LE.le 0 x
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  have : x / π ≤ sin (x / 2) := by simpa using mul_le_sin (x := x / 2) (by positivity) (by linarith)
  /-
    x✝ x : Real
    hx : LE.le x Real.pi
    hx₀ : LE.le 0 x
    this : LE.le (HDiv.hDiv x Real.pi) (Real.sin (HDiv.hDiv x 2))
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  have := (pow_le_pow_left₀ (by positivity) this 2).trans_eq (sin_sq_eq_half_sub _)
  /-
    x✝ x : Real
    hx : LE.le x Real.pi
    hx₀ : LE.le 0 x
    this✝ : LE.le (HDiv.hDiv x Real.pi) (Real.sin (HDiv.hDiv x 2))
    this : LE.le (HPow.hPow (HDiv.hDiv x Real.pi) 2) (HSub.hSub (1 / 2) (HDiv.hDiv …
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HDiv.hDiv 2 (HPow.hPow Real.pi 2 …
  -/
  ring_nf at this ⊢
  /-
    x✝ x : Real
    hx : LE.le x Real.pi
    hx₀ : LE.le 0 x
    this✝ : LE.le (HDiv.hDiv x Real.pi) (Real.sin (HDiv.hDiv x 2))
    this : LE.le (HMul.hMul (HPow.hPow x 2) (HPow.hPow (Inv.inv Real.pi) 2)) (HAdd …
    ⊢ LE.le (Real.cos x) (HSub.hSub 1 (HMul.hMul (HMul.hMul (HPow.hPow x 2) (HPow. …
  -/
  linarith
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias two_div_pi_mul_le_sin := mul_le_sin

@[deprecated (since := "2024-08-29")] alias sin_le_two_div_pi_mul := sin_le_mul

@[deprecated (since := "2024-08-29")] alias one_sub_two_div_pi_mul_le_cos := one_sub_mul_le_cos

@[deprecated (since := "2024-08-29")] alias cos_quadratic_upper_bound := cos_le_one_sub_mul_cos_sq


/-- For 0 < x ≤ 1 we have x - x ^ 3 / 4 < sin x.

This is also true for x > 1, but it's nontrivial for x just above 1. This inequality is not
tight; the tighter inequality is sin x > x - x ^ 3 / 6 for all x > 0, but this inequality has
a simpler proof. -/
theorem sin_gt_sub_cube {x : ℝ} (h : 0 < x) (h' : x ≤ 1) : x - x ^ 3 / 4 < sin x := by
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (Real.sin x)
  -/
  have hx : |x| = x := abs_of_nonneg h.le
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (Real.sin x)
  -/
  have := neg_le_of_abs_le (sin_bound <| show |x| ≤ 1 by rwa [hx])
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (Neg.neg (HMul.hMul (HPow.hPow (abs x) 4) (5 / 96))) (HSub.hSub ( …
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (Real.sin x)
  -/
  rw [le_sub_iff_add_le, hx] at this
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub.h …
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (Real.sin x)
  -/
  refine lt_of_lt_of_le ?_ this
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub.h …
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (HAdd.hAdd (Neg.neg (HMul. …
  -/
  have : x ^ 3 / ↑4 - x ^ 3 / ↑6 = x ^ 3 * 12⁻¹ := by norm_num [div_eq_mul_inv, ← mul_sub]
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this✝ : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub. …
    this : Eq (HSub.hSub (HDiv.hDiv (HPow.hPow x 3) 4) (HDiv.hDiv (HPow.hPow x 3)  …
    ⊢ LT.lt (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 4)) (HAdd.hAdd (Neg.neg (HMul. …
  -/
  rw [add_comm, sub_add, sub_neg_eq_add, sub_lt_sub_iff_left, ← lt_sub_iff_add_lt', this]
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this✝ : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub. …
    this : Eq (HSub.hSub (HDiv.hDiv (HPow.hPow x 3) 4) (HDiv.hDiv (HPow.hPow x 3)  …
    ⊢ LT.lt (HMul.hMul (HPow.hPow x 4) (5 / 96)) (HMul.hMul (HPow.hPow x 3) (Inv.i …
  -/
  refine mul_lt_mul' ?_ (by norm_num) (by norm_num) (pow_pos h 3)
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this✝ : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub. …
    this : Eq (HSub.hSub (HDiv.hDiv (HPow.hPow x 3) 4) (HDiv.hDiv (HPow.hPow x 3)  …
    ⊢ LE.le (HPow.hPow x 4) (HPow.hPow x 3)
  -/
  apply pow_le_pow_of_le_one h.le h'
  /-
    x : Real
    h : LT.lt 0 x
    h' : LE.le x 1
    hx : Eq (abs x) x
    this✝ : LE.le (HAdd.hAdd (Neg.neg (HMul.hMul (HPow.hPow x 4) (5 / 96))) (HSub. …
    this : Eq (HSub.hSub (HDiv.hDiv (HPow.hPow x 3) 4) (HDiv.hDiv (HPow.hPow x 3)  …
    ⊢ LE.le 3 4
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- The derivative of `tan x - x` is `1/(cos x)^2 - 1` away from the zeroes of cos. -/
theorem deriv_tan_sub_id (x : ℝ) (h : cos x ≠ 0) :
    deriv (fun y : ℝ => tan y - y) x = 1 / cos x ^ 2 - 1 :=
                         /-
                           x : Real
                           h : Ne (Real.cos x) 0
                           ⊢ HasDerivAt (fun y => HSub.hSub (Real.tan y) y) (HSub.hSub (HDiv.hDiv 1 (HPow …
                         -/
  HasDerivAt.deriv <| by simpa using (hasDerivAt_tan h).add (hasDerivAt_id x).neg
                         /-
                           🎉 no goals
                         -/


/-- For all `0 < x < π/2` we have `x < tan x`.

This is proved by checking that the function `tan x - x` vanishes
at zero and has non-negative derivative. -/
theorem lt_tan {x : ℝ} (h1 : 0 < x) (h2 : x < π / 2) : x < tan x := by
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    ⊢ LT.lt x (Real.tan x)
  -/
  let U := Ico 0 (π / 2)
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    ⊢ LT.lt x (Real.tan x)
  -/
  have intU : interior U = Ioo 0 (π / 2) := interior_Ico
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    ⊢ LT.lt x (Real.tan x)
  -/
  have half_pi_pos : 0 < π / 2 := div_pos pi_pos two_pos
  have cos_pos {y : ℝ} (hy : y ∈ U) : 0 < cos y := by
    exact cos_pos_of_mem_Ioo (Ico_subset_Ioo_left (neg_lt_zero.mpr half_pi_pos) hy)
  have sin_pos {y : ℝ} (hy : y ∈ interior U) : 0 < sin y := by
    rw [intU] at hy
    exact sin_pos_of_mem_Ioo (Ioo_subset_Ioo_right (div_le_self pi_pos.le one_le_two) hy)
  have tan_cts_U : ContinuousOn tan U := by
    apply ContinuousOn.mono continuousOn_tan
    intro z hz
    simp only [mem_setOf_eq]
    exact (cos_pos hz).ne'
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    half_pi_pos : LT.lt 0 (HDiv.hDiv Real.pi 2)
    cos_pos : ∀ {y : Real}, Membership.mem U y → LT.lt 0 (Real.cos y)
    sin_pos : ∀ {y : Real}, Membership.mem (interior U) y → LT.lt 0 (Real.sin y)
    tan_cts_U : ContinuousOn Real.tan U
    ⊢ LT.lt x (Real.tan x)
  -/
  have tan_minus_id_cts : ContinuousOn (fun y : ℝ => tan y - y) U := tan_cts_U.sub continuousOn_id
  have deriv_pos (y : ℝ) (hy : y ∈ interior U) : 0 < deriv (fun y' : ℝ => tan y' - y') y := by
    have := cos_pos (interior_subset hy)
    simp only [deriv_tan_sub_id y this.ne', one_div, gt_iff_lt, sub_pos]
    norm_cast
    have bd2 : cos y ^ 2 < 1 := by
      apply lt_of_le_of_ne y.cos_sq_le_one
      rw [cos_sq']
      simpa only [Ne, sub_eq_self, sq_eq_zero_iff] using (sin_pos hy).ne'
    rwa [lt_inv_comm₀, inv_one]
    · exact zero_lt_one
    simpa only [sq, mul_self_pos] using this.ne'
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    half_pi_pos : LT.lt 0 (HDiv.hDiv Real.pi 2)
    cos_pos : ∀ {y : Real}, Membership.mem U y → LT.lt 0 (Real.cos y)
    sin_pos : ∀ {y : Real}, Membership.mem (interior U) y → LT.lt 0 (Real.sin y)
    tan_cts_U : ContinuousOn Real.tan U
    tan_minus_id_cts : ContinuousOn (fun y => HSub.hSub (Real.tan y) y) U
    deriv_pos : ∀ (y : Real), Membership.mem (interior U) y → LT.lt 0 (deriv (fun  …
    ⊢ LT.lt x (Real.tan x)
  -/
  have mono := strictMonoOn_of_deriv_pos (convex_Ico 0 (π / 2)) tan_minus_id_cts deriv_pos
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    half_pi_pos : LT.lt 0 (HDiv.hDiv Real.pi 2)
    cos_pos : ∀ {y : Real}, Membership.mem U y → LT.lt 0 (Real.cos y)
    sin_pos : ∀ {y : Real}, Membership.mem (interior U) y → LT.lt 0 (Real.sin y)
    tan_cts_U : ContinuousOn Real.tan U
    tan_minus_id_cts : ContinuousOn (fun y => HSub.hSub (Real.tan y) y) U
    deriv_pos : ∀ (y : Real), Membership.mem (interior U) y → LT.lt 0 (deriv (fun  …
    mono : StrictMonoOn (fun y => HSub.hSub (Real.tan y) y) (Set.Ico 0 (HDiv.hDiv  …
    ⊢ LT.lt x (Real.tan x)
  -/
  have zero_in_U : (0 : ℝ) ∈ U := by rwa [left_mem_Ico]
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    half_pi_pos : LT.lt 0 (HDiv.hDiv Real.pi 2)
    cos_pos : ∀ {y : Real}, Membership.mem U y → LT.lt 0 (Real.cos y)
    sin_pos : ∀ {y : Real}, Membership.mem (interior U) y → LT.lt 0 (Real.sin y)
    tan_cts_U : ContinuousOn Real.tan U
    tan_minus_id_cts : ContinuousOn (fun y => HSub.hSub (Real.tan y) y) U
    deriv_pos : ∀ (y : Real), Membership.mem (interior U) y → LT.lt 0 (deriv (fun  …
    mono : StrictMonoOn (fun y => HSub.hSub (Real.tan y) y) (Set.Ico 0 (HDiv.hDiv  …
    zero_in_U : Membership.mem U 0
    ⊢ LT.lt x (Real.tan x)
  -/
  have x_in_U : x ∈ U := ⟨h1.le, h2⟩
  /-
    x : Real
    h1 : LT.lt 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    U : Set Real := Set.Ico 0 (HDiv.hDiv Real.pi 2)
    intU : Eq (interior U) (Set.Ioo 0 (HDiv.hDiv Real.pi 2))
    half_pi_pos : LT.lt 0 (HDiv.hDiv Real.pi 2)
    cos_pos : ∀ {y : Real}, Membership.mem U y → LT.lt 0 (Real.cos y)
    sin_pos : ∀ {y : Real}, Membership.mem (interior U) y → LT.lt 0 (Real.sin y)
    tan_cts_U : ContinuousOn Real.tan U
    tan_minus_id_cts : ContinuousOn (fun y => HSub.hSub (Real.tan y) y) U
    deriv_pos : ∀ (y : Real), Membership.mem (interior U) y → LT.lt 0 (deriv (fun  …
    mono : StrictMonoOn (fun y => HSub.hSub (Real.tan y) y) (Set.Ico 0 (HDiv.hDiv  …
    zero_in_U : Membership.mem U 0
    x_in_U : Membership.mem U x
    ⊢ LT.lt x (Real.tan x)
  -/
  simpa only [tan_zero, sub_zero, sub_pos] using mono zero_in_U x_in_U h1
  /-
    🎉 no goals
  -/


theorem le_tan {x : ℝ} (h1 : 0 ≤ x) (h2 : x < π / 2) : x ≤ tan x := by
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LT.lt x (HDiv.hDiv Real.pi 2)
    ⊢ LE.le x (Real.tan x)
  -/
  rcases eq_or_lt_of_le h1 with (rfl | h1')
    /-
      case inl
      h1 : LE.le 0 0
      h2 : LT.lt 0 (HDiv.hDiv Real.pi 2)
      ⊢ LE.le 0 (Real.tan 0)
    -/
  · rw [tan_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      h1 : LE.le 0 x
      h2 : LT.lt x (HDiv.hDiv Real.pi 2)
      h1' : LT.lt 0 x
      ⊢ LE.le x (Real.tan x)
    -/
  · exact le_of_lt (lt_tan h1' h2)
    /-
      🎉 no goals
    -/


theorem cos_lt_one_div_sqrt_sq_add_one {x : ℝ} (hx1 : -(3 * π / 2) ≤ x) (hx2 : x ≤ 3 * π / 2)
    (hx3 : x ≠ 0) : cos x < (1 / √(x ^ 2 + 1) : ℝ) := by
  suffices ∀ {y : ℝ}, 0 < y → y ≤ 3 * π / 2 → cos y < 1 / sqrt (y ^ 2 + 1) by
    rcases lt_or_lt_iff_ne.mpr hx3.symm with ⟨h⟩
    · exact this h hx2
    · convert this (by linarith : 0 < -x) (by linarith) using 1
      · rw [cos_neg]
      · rw [neg_sq]
  /-
    x : Real
    hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
    hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    hx3 : Ne x 0
    ⊢ ∀ {y : Real}, LT.lt 0 y → LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2) → LT.l …
  -/
  intro y hy1 hy2
  /-
    x : Real
    hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
    hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    hx3 : Ne x 0
    y : Real
    hy1 : LT.lt 0 y
    hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    ⊢ LT.lt (Real.cos y) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow y 2) 1).sqrt)
  -/
  have hy3 : ↑0 < y ^ 2 + 1 := by linarith [sq_nonneg y]
  /-
    x : Real
    hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
    hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    hx3 : Ne x 0
    y : Real
    hy1 : LT.lt 0 y
    hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    hy3 : LT.lt 0 (HAdd.hAdd (HPow.hPow y 2) 1)
    ⊢ LT.lt (Real.cos y) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow y 2) 1).sqrt)
  -/
  rcases lt_or_le y (π / 2) with (hy2' | hy1')
  · -- Main case : `0 < y < π / 2`
    /-
      case inl
      x : Real
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
      hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hx3 : Ne x 0
      y : Real
      hy1 : LT.lt 0 y
      hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hy3 : LT.lt 0 (HAdd.hAdd (HPow.hPow y 2) 1)
      hy2' : LT.lt y (HDiv.hDiv Real.pi 2)
      ⊢ LT.lt (Real.cos y) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow y 2) 1).sqrt)
    -/
    have hy4 : 0 < cos y := cos_pos_of_mem_Ioo ⟨by linarith, hy2'⟩
    rw [← abs_of_nonneg (cos_nonneg_of_mem_Icc ⟨by linarith, hy2'.le⟩), ←
      abs_of_nonneg (one_div_nonneg.mpr (sqrt_nonneg _)), ← sq_lt_sq, div_pow, one_pow,
      sq_sqrt hy3.le, lt_one_div (pow_pos hy4 _) hy3, ← inv_one_add_tan_sq hy4.ne', one_div,
      inv_inv, add_comm, add_lt_add_iff_left, sq_lt_sq, abs_of_pos hy1,
      abs_of_nonneg (tan_nonneg_of_nonneg_of_le_pi_div_two hy1.le hy2'.le)]
    /-
      case inl
      x : Real
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
      hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hx3 : Ne x 0
      y : Real
      hy1 : LT.lt 0 y
      hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hy3 : LT.lt 0 (HAdd.hAdd (HPow.hPow y 2) 1)
      hy2' : LT.lt y (HDiv.hDiv Real.pi 2)
      hy4 : LT.lt 0 (Real.cos y)
      ⊢ LT.lt y (Real.tan y)
    -/
    exact Real.lt_tan hy1 hy2'
    /-
      🎉 no goals
    -/
  · -- Easy case : `π / 2 ≤ y ≤ 3 * π / 2`
    /-
      case inr
      x : Real
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
      hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hx3 : Ne x 0
      y : Real
      hy1 : LT.lt 0 y
      hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hy3 : LT.lt 0 (HAdd.hAdd (HPow.hPow y 2) 1)
      hy1' : LE.le (HDiv.hDiv Real.pi 2) y
      ⊢ LT.lt (Real.cos y) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow y 2) 1).sqrt)
    -/
    refine lt_of_le_of_lt ?_ (one_div_pos.mpr <| sqrt_pos_of_pos hy3)
    /-
      case inr
      x : Real
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
      hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hx3 : Ne x 0
      y : Real
      hy1 : LT.lt 0 y
      hy2 : LE.le y (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hy3 : LT.lt 0 (HAdd.hAdd (HPow.hPow y 2) 1)
      hy1' : LE.le (HDiv.hDiv Real.pi 2) y
      ⊢ LE.le (Real.cos y) 0
    -/
    exact cos_nonpos_of_pi_div_two_le_of_le hy1' (by linarith [pi_pos])
    /-
      🎉 no goals
    -/


theorem cos_le_one_div_sqrt_sq_add_one {x : ℝ} (hx1 : -(3 * π / 2) ≤ x) (hx2 : x ≤ 3 * π / 2) :
    cos x ≤ (1 : ℝ) / √(x ^ 2 + 1) := by
  /-
    x : Real
    hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
    hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
    ⊢ LE.le (Real.cos x) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow x 2) 1).sqrt)
  -/
  rcases eq_or_ne x 0 with (rfl | hx3)
    /-
      case inl
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) 0
      hx2 : LE.le 0 (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      ⊢ LE.le (Real.cos 0) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow 0 2) 1).sqrt)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx1 : LE.le (Neg.neg (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)) x
      hx2 : LE.le x (HDiv.hDiv (HMul.hMul 3 Real.pi) 2)
      hx3 : Ne x 0
      ⊢ LE.le (Real.cos x) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow x 2) 1).sqrt)
    -/
  · exact (cos_lt_one_div_sqrt_sq_add_one hx1 hx2 hx3).le
    /-
      🎉 no goals
    -/


