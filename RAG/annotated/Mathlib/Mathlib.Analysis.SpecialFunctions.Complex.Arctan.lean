/-- The complex arctangent, defined via the complex logarithm. -/
noncomputable def arctan (z : ℂ) : ℂ := -I / 2 * log ((1 + z * I) / (1 - z * I))


theorem tan_arctan {z : ℂ} (h₁ : z ≠ I) (h₂ : z ≠ -I) : tan (arctan z) = z := by
  /-
    z : Complex
    h₁ : Ne z Complex.I
    h₂ : Ne z (Neg.neg Complex.I)
    ⊢ Eq (Complex.tan z.arctan) z
  -/
  unfold tan sin cos
  rw [div_div_eq_mul_div, div_mul_cancel₀ _ two_ne_zero, ← div_mul_eq_mul_div,
    -- multiply top and bottom by `exp (arctan z * I)`
    ← mul_div_mul_right _ _ (exp_ne_zero (arctan z * I)), sub_mul, add_mul,
    ← exp_add, neg_mul, neg_add_cancel, exp_zero, ← exp_add, ← two_mul]
  have z₁ : 1 + z * I ≠ 0 := by
    contrapose! h₁
    rw [add_eq_zero_iff_neg_eq, ← div_eq_iff I_ne_zero, div_I, neg_one_mul, neg_neg] at h₁
    exact h₁.symm
  have z₂ : 1 - z * I ≠ 0 := by
    contrapose! h₂
    rw [sub_eq_zero, ← div_eq_iff I_ne_zero, div_I, one_mul] at h₂
    exact h₂.symm
  have key : exp (2 * (arctan z * I)) = (1 + z * I) / (1 - z * I) := by
    rw [arctan, ← mul_rotate, ← mul_assoc,
      show 2 * (I * (-I / 2)) = 1 by field_simp, one_mul, exp_log]
    · exact div_ne_zero z₁ z₂
  -- multiply top and bottom by `1 - z * I`
  rw [key, ← mul_div_mul_right _ _ z₂, sub_mul, add_mul, div_mul_cancel₀ _ z₂, one_mul,
    show _ / _ * I = -(I * I) * z by ring, I_mul_I, neg_neg, one_mul]


/-- `cos z` is nonzero when the bounds in `arctan_tan` are met (`z` lies in the vertical strip
`-π / 2 < z.re < π / 2` and `z ≠ π / 2`). -/
lemma cos_ne_zero_of_arctan_bounds {z : ℂ} (h₀ : z ≠ π / 2) (h₁ : -(π / 2) < z.re)
    (h₂ : z.re ≤ π / 2) : cos z ≠ 0 := by
  /-
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    ⊢ Ne (Complex.cos z) 0
  -/
  refine cos_ne_zero_iff.mpr (fun k ↦ ?_)
  /-
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    k : Int
    ⊢ Ne z (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) ↑Real.pi) 2)
  -/
  rw [ne_eq, Complex.ext_iff, not_and_or] at h₀ ⊢
  /-
    z : Complex
    h₀ : Or (Not (Eq z.re (HDiv.hDiv (↑Real.pi) 2).re)) (Not (Eq z.im (HDiv.hDiv ( …
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    k : Int
    ⊢ Or (Not (Eq z.re (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) ↑Real. …
  -/
  norm_cast at h₀ ⊢
  /-
    z : Complex
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    k : Int
    h₀ : Or (Not (Eq z.re (HDiv.hDiv Real.pi 2))) (Not (Eq z.im 0))
    ⊢ Or (Not (Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real …
  -/
  cases' h₀ with nr ni
    /-
      case inl
      z : Complex
      h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      k : Int
      nr : Not (Eq z.re (HDiv.hDiv Real.pi 2))
      ⊢ Or (Not (Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real …
    -/
  · left; contrapose! nr
    /-
      case inl.h
      z : Complex
      h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      k : Int
      nr : Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real.pi) 2)
      ⊢ Eq z.re (HDiv.hDiv Real.pi 2)
    -/
    rw [nr, mul_div_assoc, neg_eq_neg_one_mul, mul_lt_mul_iff_of_pos_right (by positivity)] at h₁
    /-
      case inl.h
      z : Complex
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      k : Int
      h₁ : LT.lt (-1) ↑(HAdd.hAdd (HMul.hMul 2 k) 1)
      nr : Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real.pi) 2)
      ⊢ Eq z.re (HDiv.hDiv Real.pi 2)
    -/
    rw [nr, ← one_mul (π / 2), mul_div_assoc, mul_le_mul_iff_of_pos_right (by positivity)] at h₂
    /-
      case inl.h
      z : Complex
      k : Int
      h₂ : LE.le (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) 1
      h₁ : LT.lt (-1) ↑(HAdd.hAdd (HMul.hMul 2 k) 1)
      nr : Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real.pi) 2)
      ⊢ Eq z.re (HDiv.hDiv Real.pi 2)
    -/
    norm_cast at h₁ h₂
    /-
      case inl.h
      z : Complex
      k : Int
      nr : Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real.pi) 2)
      h₁ : LT.lt (Int.negSucc 0) (HAdd.hAdd (HMul.hMul 2 k) 1)
      h₂ : LE.le (HAdd.hAdd (HMul.hMul 2 k) 1) 1
      ⊢ Eq z.re (HDiv.hDiv Real.pi 2)
    -/
    change -1 < _ at h₁
    /-
      case inl.h
      z : Complex
      k : Int
      nr : Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real.pi) 2)
      h₂ : LE.le (HAdd.hAdd (HMul.hMul 2 k) 1) 1
      h₁ : LT.lt (-1) (HAdd.hAdd (HMul.hMul 2 k) 1)
      ⊢ Eq z.re (HDiv.hDiv Real.pi 2)
    -/
    rwa [show 2 * k + 1 = 1 by omega, Int.cast_one, one_mul] at nr
    /-
      🎉 no goals
    -/
    /-
      case inr
      z : Complex
      h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      k : Int
      ni : Not (Eq z.im 0)
      ⊢ Or (Not (Eq z.re (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k) 1)) Real …
    -/
  · exact Or.inr ni
    /-
      🎉 no goals
    -/


theorem arctan_tan {z : ℂ} (h₀ : z ≠ π / 2) (h₁ : -(π / 2) < z.re) (h₂ : z.re ≤ π / 2) :
    arctan (tan z) = z := by
  /-
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    ⊢ Eq (Complex.tan z).arctan z
  -/
  have h := cos_ne_zero_of_arctan_bounds h₀ h₁ h₂
  /-
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    h : Ne (Complex.cos z) 0
    ⊢ Eq (Complex.tan z).arctan z
  -/
  unfold arctan tan
  -- multiply top and bottom by `cos z`
  /-
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    h : Ne (Complex.cos z) 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (Complex.log (HDiv.hDiv (HAd …
  -/
  rw [← mul_div_mul_right (1 + _) _ h, add_mul, sub_mul, one_mul, ← mul_rotate, mul_div_cancel₀ _ h]
  conv_lhs =>
    enter [2, 1, 2]
    rw [sub_eq_add_neg, ← neg_mul, ← sin_neg, ← cos_neg]
  rw [← exp_mul_I, ← exp_mul_I, ← exp_sub, show z * I - -z * I = 2 * (I * z) by ring, log_exp,
    show -I / 2 * (2 * (I * z)) = -(I * I) * z by ring, I_mul_I, neg_neg, one_mul]
  /-
    case hx₁
    z : Complex
    h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
    h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
    h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
    h : Ne (Complex.cos z) 0
    ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul 2 (HMul.hMul Complex.I z)).im
  -/
  all_goals norm_num
    /-
      case hx₁
      z : Complex
      h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
      h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      h : Ne (Complex.cos z) 0
      ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul 2 z.re)
    -/
  · rwa [← div_lt_iff₀' two_pos, neg_div]
    /-
      🎉 no goals
    -/
    /-
      case hx₂
      z : Complex
      h₀ : Ne z (HDiv.hDiv (↑Real.pi) 2)
      h₁ : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.re
      h₂ : LE.le z.re (HDiv.hDiv Real.pi 2)
      h : Ne (Complex.cos z) 0
      ⊢ LE.le (HMul.hMul 2 z.re) Real.pi
    -/
  · rwa [← le_div_iff₀' two_pos]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem ofReal_arctan (x : ℝ) : (Real.arctan x : ℂ) = arctan x := by
  /-
    x : Real
    ⊢ Eq (↑(Real.arctan x)) (↑x).arctan
  -/
  conv_rhs => rw [← Real.tan_arctan x]
  /-
    x : Real
    ⊢ Eq (↑(Real.arctan x)) (↑(Real.tan (Real.arctan x))).arctan
  -/
  rw [ofReal_tan, arctan_tan]
  /-
    case h₀
    x : Real
    ⊢ Ne (↑(Real.arctan x)) (HDiv.hDiv (↑Real.pi) 2)
  -/
  all_goals norm_cast
    /-
      case h₀
      x : Real
      ⊢ Not (Eq (Real.arctan x) (HDiv.hDiv Real.pi 2))
    -/
  · rw [← ne_eq]; exact (Real.arctan_lt_pi_div_two _).ne
                  /-
                    🎉 no goals
                  -/
    /-
      case h₁
      x : Real
      ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (Real.arctan x)
    -/
  · exact Real.neg_pi_div_two_lt_arctan _
    /-
      🎉 no goals
    -/
    /-
      case h₂
      x : Real
      ⊢ LE.le (Real.arctan x) (HDiv.hDiv Real.pi 2)
    -/
  · exact (Real.arctan_lt_pi_div_two _).le
    /-
      🎉 no goals
    -/


/-- The argument of `1 + z` for `z` in the open unit disc is always in `(-π / 2, π / 2)`. -/
lemma arg_one_add_mem_Ioo {z : ℂ} (hz : ‖z‖ < 1) : (1 + z).arg ∈ Set.Ioo (-(π / 2)) (π / 2) := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
  -/
  rw [Set.mem_Ioo, ← abs_lt, abs_arg_lt_pi_div_two_iff, add_re, one_re, ← neg_lt_iff_pos_add']
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ Or (LT.lt (-1) z.re) (Eq (HAdd.hAdd 1 z) 0)
  -/
  exact Or.inl (abs_lt.mp ((abs_re_le_abs z).trans_lt (norm_eq_abs z ▸ hz))).1
  /-
    🎉 no goals
  -/


/-- We can combine the logs in `log (1 + z * I) + -log (1 - z * I)` into one.
This is only used in `hasSum_arctan`. -/
lemma hasSum_arctan_aux {z : ℂ} (hz : ‖z‖ < 1) :
    log (1 + z * I) + -log (1 - z * I) = log ((1 + z * I) / (1 - z * I)) := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ Eq (HAdd.hAdd (Complex.log (HAdd.hAdd 1 (HMul.hMul z Complex.I))) (Neg.neg ( …
  -/
  have z₁ := mem_slitPlane_iff_arg.mp (mem_slitPlane_of_norm_lt_one (z := z * I) (by simpa))
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    ⊢ Eq (HAdd.hAdd (Complex.log (HAdd.hAdd 1 (HMul.hMul z Complex.I))) (Neg.neg ( …
  -/
  have z₂ := mem_slitPlane_iff_arg.mp (mem_slitPlane_of_norm_lt_one (z := -(z * I)) (by simpa))
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HAdd.hAdd 1 (Neg.neg (HMul.hMul z Complex.I))).arg Real.pi) (Ne  …
    ⊢ Eq (HAdd.hAdd (Complex.log (HAdd.hAdd 1 (HMul.hMul z Complex.I))) (Neg.neg ( …
  -/
  rw [← sub_eq_add_neg] at z₂
  rw [← log_inv _ z₂.1, ← (log_mul_eq_add_log_iff z₁.2 (inv_eq_zero.ne.mpr z₂.2)).mpr,
    div_eq_mul_inv]
  -- `log_mul_eq_add_log_iff` requires a bound on `arg (1 + z * I) + arg (1 - z * I)⁻¹`.
  -- `arg_one_add_mem_Ioo` provides sufficiently tight bounds on both terms
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  have b₁ := arg_one_add_mem_Ioo (z := z * I) (by simpa)
  have b₂ : arg (1 - z * I)⁻¹ ∈ Set.Ioo (-(π / 2)) (π / 2) := by
    simp_rw [arg_inv, z₂.1, ite_false, Set.neg_mem_Ioo_iff, neg_neg, sub_eq_add_neg]
    exact arg_one_add_mem_Ioo (by simpa)
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    b₁ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    b₂ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  have c₁ := add_lt_add b₁.1 b₂.1
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    b₁ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    b₂ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    c₁ : LT.lt (HAdd.hAdd (Neg.neg (HDiv.hDiv Real.pi 2)) (Neg.neg (HDiv.hDiv Real …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  have c₂ := add_lt_add b₁.2 b₂.2
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    b₁ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    b₂ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    c₁ : LT.lt (HAdd.hAdd (Neg.neg (HDiv.hDiv Real.pi 2)) (Neg.neg (HDiv.hDiv Real …
    c₂ : LT.lt (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg (Inv.inv (HSub …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  rw [show -(π / 2) + -(π / 2) = -π by ring] at c₁
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    b₁ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    b₂ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    c₁ : LT.lt (Neg.neg Real.pi) (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul z Complex.I)). …
    c₂ : LT.lt (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg (Inv.inv (HSub …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  rw [show π / 2 + π / 2 = π by ring] at c₂
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    z₁ : And (Ne (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HAdd.hAdd …
    z₂ : And (Ne (HSub.hSub 1 (HMul.hMul z Complex.I)).arg Real.pi) (Ne (HSub.hSub …
    b₁ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    b₂ : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    c₁ : LT.lt (Neg.neg Real.pi) (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul z Complex.I)). …
    c₂ : LT.lt (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul z Complex.I)).arg (Inv.inv (HSub …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (HAdd.hAdd 1 ( …
  -/
  exact ⟨c₁, c₂.le⟩
  /-
    🎉 no goals
  -/


/-- The power series expansion of `Complex.arctan`, valid on the open unit disc. -/
theorem hasSum_arctan {z : ℂ} (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ ↦ (-1) ^ n * z ^ (2 * n + 1) / ↑(2 * n + 1)) (arctan z) := by
  have := ((hasSum_taylorSeries_log (z := z * I) (by simpa)).add
    (hasSum_taylorSeries_neg_log (z := z * I) (by simpa))).mul_left (-I / 2)
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun i => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HAdd.hAdd …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  simp_rw [← add_div, ← add_one_mul, hasSum_arctan_aux hz] at this
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun i => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  replace := (Nat.divModEquiv 2).symm.hasSum_iff.mpr this
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (Function.comp (fun i => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  dsimp [Function.comp_def] at this
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun x => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  simp_rw [← mul_comm 2 _] at this
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun x => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  refine this.prod_fiberwise fun k => ?_
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun x => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    k : Nat
    ⊢ HasSum (fun c => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv (HMu …
  -/
  dsimp only
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun x => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    k : Nat
    ⊢ HasSum (fun c => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv (HMu …
  -/
  convert hasSum_fintype (_ : Fin 2 → ℂ) using 1
  rw [Fin.sum_univ_two, Fin.val_zero, Fin.val_one, Odd.neg_one_pow (n := 2 * k + 0 + 1) (by simp),
    neg_add_cancel, zero_mul, zero_div, mul_zero, zero_add,
    show 2 * k + 1 + 1 = 2 * (k + 1) by ring, Even.neg_one_pow (n := 2 * (k + 1)) (by simp),
    ← mul_div_assoc (_ / _), ← mul_assoc, show -I / 2 * (1 + 1) = -I by ring]
  /-
    case h.e'_6
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : HasSum (fun x => HMul.hMul (HDiv.hDiv (Neg.neg Complex.I) 2) (HDiv.hDiv …
    k : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) k) (HPow.hPow z (HAdd.hAdd (HMul.hM …
  -/
  congr 1
  rw [mul_pow, pow_succ' I, pow_mul, I_sq,
    show -I * _ = -(I * I) * (-1) ^ k * z ^ (2 * k + 1) by ring, I_mul_I, neg_neg, one_mul]


/-- The power series expansion of `Real.arctan`, valid on `-1 < x < 1`. -/
theorem Real.hasSum_arctan {x : ℝ} (hx : ‖x‖ < 1) :
    HasSum (fun n : ℕ => (-1) ^ n * x ^ (2 * n + 1) / ↑(2 * n + 1)) (arctan x) :=
                                              /-
                                                x : Real
                                                hx : LT.lt (Norm.norm x) 1
                                                ⊢ LT.lt (Norm.norm ↑x) 1
                                              -/
  mod_cast Complex.hasSum_arctan (z := x) (by simpa)
                                              /-
                                                🎉 no goals
                                              -/

