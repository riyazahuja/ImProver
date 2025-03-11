/-- `arg` returns values in the range (-π, π], such that for `x ≠ 0`,
  `sin (arg x) = x.im / x.abs` and `cos (arg x) = x.re / x.abs`,
  `arg 0` defaults to `0` -/
noncomputable def arg (x : ℂ) : ℝ :=
  if 0 ≤ x.re then Real.arcsin (x.im / abs x)
  else if 0 ≤ x.im then Real.arcsin ((-x).im / abs x) + π else Real.arcsin ((-x).im / abs x) - π


theorem sin_arg (x : ℂ) : Real.sin (arg x) = x.im / abs x := by
  /-
    x : Complex
    ⊢ Eq (Real.sin x.arg) (HDiv.hDiv x.im (Complex.abs x))
  -/
  unfold arg; split_ifs <;>
    simp [sub_eq_add_neg, arg,
      Real.sin_arcsin (abs_le.1 (abs_im_div_abs_le_one x)).1 (abs_le.1 (abs_im_div_abs_le_one x)).2,
      Real.sin_add, neg_div, Real.arcsin_neg, Real.sin_neg]


theorem cos_arg {x : ℂ} (hx : x ≠ 0) : Real.cos (arg x) = x.re / abs x := by
  /-
    x : Complex
    hx : Ne x 0
    ⊢ Eq (Real.cos x.arg) (HDiv.hDiv x.re (Complex.abs x))
  -/
  rw [arg]
  /-
    x : Complex
    hx : Ne x 0
    ⊢ Eq (Real.cos (ite (LE.le 0 x.re) (Real.arcsin (HDiv.hDiv x.im (Complex.abs x …
  -/
  split_ifs with h₁ h₂
    /-
      case pos
      x : Complex
      hx : Ne x 0
      h₁ : LE.le 0 x.re
      ⊢ Eq (Real.cos (Real.arcsin (HDiv.hDiv x.im (Complex.abs x)))) (HDiv.hDiv x.re …
    -/
  · rw [Real.cos_arcsin]
    /-
      case pos
      x : Complex
      hx : Ne x 0
      h₁ : LE.le 0 x.re
      ⊢ Eq (HSub.hSub 1 (HPow.hPow (HDiv.hDiv x.im (Complex.abs x)) 2)).sqrt (HDiv.h …
    -/
    field_simp [Real.sqrt_sq, (abs.pos hx).le, *]
    /-
      🎉 no goals
    -/
    /-
      case pos
      x : Complex
      hx : Ne x 0
      h₁ : Not (LE.le 0 x.re)
      h₂ : LE.le 0 x.im
      ⊢ Eq (Real.cos (HAdd.hAdd (Real.arcsin (HDiv.hDiv (Neg.neg x).im (Complex.abs  …
    -/
  · rw [Real.cos_add_pi, Real.cos_arcsin]
    field_simp [Real.sqrt_div (sq_nonneg _), Real.sqrt_sq_eq_abs,
      _root_.abs_of_neg (not_le.1 h₁), *]
    /-
      case neg
      x : Complex
      hx : Ne x 0
      h₁ : Not (LE.le 0 x.re)
      h₂ : Not (LE.le 0 x.im)
      ⊢ Eq (Real.cos (HSub.hSub (Real.arcsin (HDiv.hDiv (Neg.neg x).im (Complex.abs  …
    -/
  · rw [Real.cos_sub_pi, Real.cos_arcsin]
    field_simp [Real.sqrt_div (sq_nonneg _), Real.sqrt_sq_eq_abs,
      _root_.abs_of_neg (not_le.1 h₁), *]


@[simp]
theorem abs_mul_exp_arg_mul_I (x : ℂ) : ↑(abs x) * exp (arg x * I) = x := by
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (↑(Complex.abs x)) (Complex.exp (HMul.hMul (↑x.arg) Complex.I) …
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      ⊢ Eq (HMul.hMul (↑(Complex.abs 0)) (Complex.exp (HMul.hMul (↑(Complex.arg 0))  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Complex
      hx : Ne x 0
      ⊢ Eq (HMul.hMul (↑(Complex.abs x)) (Complex.exp (HMul.hMul (↑x.arg) Complex.I) …
    -/
  · have : abs x ≠ 0 := abs.ne_zero hx
    /-
      case inr
      x : Complex
      hx : Ne x 0
      this : Ne (Complex.abs x) 0
      ⊢ Eq (HMul.hMul (↑(Complex.abs x)) (Complex.exp (HMul.hMul (↑x.arg) Complex.I) …
    -/
                          /-
                            🎉 no goals
                          -/
    apply Complex.ext <;> field_simp [sin_arg, cos_arg hx, this, mul_comm (abs x)]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem abs_mul_cos_add_sin_mul_I (x : ℂ) : (abs x * (cos (arg x) + sin (arg x) * I) : ℂ) = x := by
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (↑(Complex.abs x)) (HAdd.hAdd (Complex.cos ↑x.arg) (HMul.hMul  …
  -/
  rw [← exp_mul_I, abs_mul_exp_arg_mul_I]
  /-
    🎉 no goals
  -/


@[simp]
lemma abs_mul_cos_arg (x : ℂ) : abs x * Real.cos (arg x) = x.re := by
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (Complex.abs x) (Real.cos x.arg)) x.re
  -/
  simpa [-abs_mul_cos_add_sin_mul_I] using congr_arg re (abs_mul_cos_add_sin_mul_I x)
  /-
    🎉 no goals
  -/


@[simp]
lemma abs_mul_sin_arg (x : ℂ) : abs x * Real.sin (arg x) = x.im := by
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (Complex.abs x) (Real.sin x.arg)) x.im
  -/
  simpa [-abs_mul_cos_add_sin_mul_I] using congr_arg im (abs_mul_cos_add_sin_mul_I x)
  /-
    🎉 no goals
  -/


theorem abs_eq_one_iff (z : ℂ) : abs z = 1 ↔ ∃ θ : ℝ, exp (θ * I) = z := by
  /-
    z : Complex
    ⊢ Iff (Eq (Complex.abs z) 1) (Exists fun θ => Eq (Complex.exp (HMul.hMul (↑θ)  …
  -/
  refine ⟨fun hz => ⟨arg z, ?_⟩, ?_⟩
  · calc
      exp (arg z * I) = abs z * exp (arg z * I) := by rw [hz, ofReal_one, one_mul]
      _ = z := abs_mul_exp_arg_mul_I z

    /-
      case refine_2
      z : Complex
      ⊢ (Exists fun θ => Eq (Complex.exp (HMul.hMul (↑θ) Complex.I)) z) → Eq (Comple …
    -/
  · rintro ⟨θ, rfl⟩
    /-
      case refine_2.intro
      θ : Real
      ⊢ Eq (Complex.abs (Complex.exp (HMul.hMul (↑θ) Complex.I))) 1
    -/
    exact Complex.abs_exp_ofReal_mul_I θ
    /-
      🎉 no goals
    -/


@[simp]
theorem range_exp_mul_I : (Set.range fun x : ℝ => exp (x * I)) = Metric.sphere 0 1 := by
  /-
    ⊢ Eq (Set.range fun x => Complex.exp (HMul.hMul (↑x) Complex.I)) (Metric.spher …
  -/
  ext x
  /-
    case h
    x : Complex
    ⊢ Iff (Membership.mem (Set.range fun x => Complex.exp (HMul.hMul (↑x) Complex. …
  -/
  simp only [mem_sphere_zero_iff_norm, norm_eq_abs, abs_eq_one_iff, Set.mem_range]
  /-
    🎉 no goals
  -/


theorem arg_mul_cos_add_sin_mul_I {r : ℝ} (hr : 0 < r) {θ : ℝ} (hθ : θ ∈ Set.Ioc (-π) π) :
    arg (r * (cos θ + sin θ * I)) = θ := by
  /-
    r : Real
    hr : LT.lt 0 r
    θ : Real
    hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
    ⊢ Eq (HMul.hMul (↑r) (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) C …
  -/
  simp only [arg, map_mul, abs_cos_add_sin_mul_I, abs_of_nonneg hr.le, mul_one]
  simp only [re_ofReal_mul, im_ofReal_mul, neg_im, ← ofReal_cos, ← ofReal_sin, ←
    mk_eq_add_mul_I, neg_div, mul_div_cancel_left₀ _ hr.ne', mul_nonneg_iff_right_nonneg_of_pos hr]
  /-
    r : Real
    hr : LT.lt 0 r
    θ : Real
    hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
    ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
  -/
  by_cases h₁ : θ ∈ Set.Icc (-(π / 2)) (π / 2)
    /-
      case pos
      r : Real
      hr : LT.lt 0 r
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      h₁ : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
      ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
    -/
  · rw [if_pos]
    /-
      case pos
      r : Real
      hr : LT.lt 0 r
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      h₁ : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
      ⊢ Eq (Real.arcsin (Real.sin θ)) θ
    -/
    exacts [Real.arcsin_sin' h₁, Real.cos_nonneg_of_mem_Icc h₁]
    /-
      🎉 no goals
    -/
    /-
      case neg
      r : Real
      hr : LT.lt 0 r
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      h₁ : Not (Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv R …
      ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
    -/
  · rw [Set.mem_Icc, not_and_or, not_le, not_le] at h₁
    /-
      case neg
      r : Real
      hr : LT.lt 0 r
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      h₁ : Or (LT.lt θ (Neg.neg (HDiv.hDiv Real.pi 2))) (LT.lt (HDiv.hDiv Real.pi 2) …
      ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
    -/
    cases' h₁ with h₁ h₁
      /-
        case neg.inl
        r : Real
        hr : LT.lt 0 r
        θ : Real
        hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
        h₁ : LT.lt θ (Neg.neg (HDiv.hDiv Real.pi 2))
        ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
      -/
    · replace hθ := hθ.1
      have hcos : Real.cos θ < 0 := by
        rw [← neg_pos, ← Real.cos_add_pi]
        refine Real.cos_pos_of_mem_Ioo ⟨?_, ?_⟩ <;> linarith
      /-
        case neg.inl
        r : Real
        hr : LT.lt 0 r
        θ : Real
        h₁ : LT.lt θ (Neg.neg (HDiv.hDiv Real.pi 2))
        hθ : LT.lt (Neg.neg Real.pi) θ
        hcos : LT.lt (Real.cos θ) 0
        ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
      -/
      have hsin : Real.sin θ < 0 := Real.sin_neg_of_neg_of_neg_pi_lt (by linarith) hθ
      rw [if_neg, if_neg, ← Real.sin_add_pi, Real.arcsin_sin, add_sub_cancel_right] <;> [linarith;
        linarith; exact hsin.not_le; exact hcos.not_le]
      /-
        case neg.inr
        r : Real
        hr : LT.lt 0 r
        θ : Real
        hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
        h₁ : LT.lt (HDiv.hDiv Real.pi 2) θ
        ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
      -/
    · replace hθ := hθ.2
      /-
        case neg.inr
        r : Real
        hr : LT.lt 0 r
        θ : Real
        h₁ : LT.lt (HDiv.hDiv Real.pi 2) θ
        hθ : LE.le θ Real.pi
        ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
      -/
      have hcos : Real.cos θ < 0 := Real.cos_neg_of_pi_div_two_lt_of_lt h₁ (by linarith)
      /-
        case neg.inr
        r : Real
        hr : LT.lt 0 r
        θ : Real
        h₁ : LT.lt (HDiv.hDiv Real.pi 2) θ
        hθ : LE.le θ Real.pi
        hcos : LT.lt (Real.cos θ) 0
        ⊢ Eq (ite (LE.le 0 (Real.cos θ)) (Real.arcsin (Real.sin θ)) (ite (LE.le 0 (Rea …
      -/
      have hsin : 0 ≤ Real.sin θ := Real.sin_nonneg_of_mem_Icc ⟨by linarith, hθ⟩
      rw [if_neg, if_pos, ← Real.sin_sub_pi, Real.arcsin_sin, sub_add_cancel] <;> [linarith;
        linarith; exact hsin; exact hcos.not_le]


theorem arg_cos_add_sin_mul_I {θ : ℝ} (hθ : θ ∈ Set.Ioc (-π) π) : arg (cos θ + sin θ * I) = θ := by
  /-
    θ : Real
    hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
    ⊢ Eq (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) Complex.I)).arg θ
  -/
  rw [← one_mul (_ + _), ← ofReal_one, arg_mul_cos_add_sin_mul_I zero_lt_one hθ]
  /-
    🎉 no goals
  -/


lemma arg_exp_mul_I (θ : ℝ) :
    arg (exp (θ * I)) = toIocMod (mul_pos two_pos Real.pi_pos) (-π) θ := by
  /-
    θ : Real
    ⊢ Eq (Complex.exp (HMul.hMul (↑θ) Complex.I)).arg (toIocMod ⋯ (Neg.neg Real.pi …
  -/
  convert arg_cos_add_sin_mul_I (θ := toIocMod (mul_pos two_pos Real.pi_pos) (-π) θ) _ using 2
  · rw [← exp_mul_I, eq_sub_of_add_eq <| toIocMod_add_toIocDiv_zsmul _ _ θ, ofReal_sub,
      ofReal_zsmul, ofReal_mul, ofReal_ofNat, exp_mul_I_periodic.sub_zsmul_eq]
    /-
      θ : Real
      ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (toIocMod ⋯ (Neg.neg Real …
    -/
  · convert toIocMod_mem_Ioc _ _ _
    /-
      case h.e'_4.h.e'_4
      θ : Real
      ⊢ Eq Real.pi (HAdd.hAdd (Neg.neg Real.pi) (HMul.hMul 2 Real.pi))
    -/
    ring
    /-
      🎉 no goals
    -/


@[simp]
                                   /-
                                     ⊢ Eq (Complex.arg 0) 0
                                   -/
theorem arg_zero : arg 0 = 0 := by simp [arg, le_refl]
                                   /-
                                     🎉 no goals
                                   -/


theorem ext_abs_arg {x y : ℂ} (h₁ : abs x = abs y) (h₂ : x.arg = y.arg) : x = y := by
  /-
    x y : Complex
    h₁ : Eq (Complex.abs x) (Complex.abs y)
    h₂ : Eq x.arg y.arg
    ⊢ Eq x y
  -/
  rw [← abs_mul_exp_arg_mul_I x, ← abs_mul_exp_arg_mul_I y, h₁, h₂]
  /-
    🎉 no goals
  -/


theorem ext_abs_arg_iff {x y : ℂ} : x = y ↔ abs x = abs y ∧ arg x = arg y :=
  ⟨fun h => h ▸ ⟨rfl, rfl⟩, and_imp.2 ext_abs_arg⟩


theorem arg_mem_Ioc (z : ℂ) : arg z ∈ Set.Ioc (-π) π := by
  /-
    z : Complex
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  have hπ : 0 < π := Real.pi_pos
  /-
    z : Complex
    hπ : LT.lt 0 Real.pi
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  rcases eq_or_ne z 0 with (rfl | hz)
    /-
      case inl
      hπ : LT.lt 0 Real.pi
      ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (Complex.arg 0)
    -/
  · simp [hπ, hπ.le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  rcases existsUnique_add_zsmul_mem_Ioc Real.two_pi_pos (arg z) (-π) with ⟨N, hN, -⟩
  /-
    case inr.intro.intro
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    N : Int
    hN : Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (H …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  rw [two_mul, neg_add_cancel_left, ← two_mul, zsmul_eq_mul] at hN
  /-
    case inr.intro.intro
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    N : Int
    hN : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd z.arg (HMul …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  rw [← abs_mul_cos_add_sin_mul_I z, ← cos_add_int_mul_two_pi _ N, ← sin_add_int_mul_two_pi _ N]
  /-
    case inr.intro.intro
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    N : Int
    hN : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd z.arg (HMul …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HMul.hMul (↑(Complex.abs …
  -/
  have := arg_mul_cos_add_sin_mul_I (abs.pos hz) hN
  /-
    case inr.intro.intro
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    N : Int
    hN : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd z.arg (HMul …
    this : Eq (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos ↑(HAdd.hAdd z. …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HMul.hMul (↑(Complex.abs …
  -/
  push_cast at this
  /-
    case inr.intro.intro
    z : Complex
    hπ : LT.lt 0 Real.pi
    hz : Ne z 0
    N : Int
    hN : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd z.arg (HMul …
    this : Eq (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos (HAdd.hAdd (↑z …
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HMul.hMul (↑(Complex.abs …
  -/
  rwa [this]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_arg : Set.range arg = Set.Ioc (-π) π :=
  (Set.range_subset_iff.2 arg_mem_Ioc).antisymm fun _ hx => ⟨_, arg_cos_add_sin_mul_I hx⟩


theorem arg_le_pi (x : ℂ) : arg x ≤ π :=
  (arg_mem_Ioc x).2


theorem neg_pi_lt_arg (x : ℂ) : -π < arg x :=
  (arg_mem_Ioc x).1


theorem abs_arg_le_pi (z : ℂ) : |arg z| ≤ π :=
  abs_le.2 ⟨(neg_pi_lt_arg z).le, arg_le_pi z⟩


@[simp]
theorem arg_nonneg_iff {z : ℂ} : 0 ≤ arg z ↔ 0 ≤ z.im := by
  /-
    z : Complex
    ⊢ Iff (LE.le 0 z.arg) (LE.le 0 z.im)
  -/
  rcases eq_or_ne z 0 with (rfl | h₀); · simp
                                         /-
                                           🎉 no goals
                                         -/
  calc
    0 ≤ arg z ↔ 0 ≤ Real.sin (arg z) :=
      ⟨fun h => Real.sin_nonneg_of_mem_Icc ⟨h, arg_le_pi z⟩, by
        contrapose!
        intro h
        exact Real.sin_neg_of_neg_of_neg_pi_lt h (neg_pi_lt_arg _)⟩
    _ ↔ _ := by rw [sin_arg, le_div_iff₀ (abs.pos h₀), zero_mul]


@[simp]
theorem arg_neg_iff {z : ℂ} : arg z < 0 ↔ z.im < 0 :=
  lt_iff_lt_of_le_iff_le arg_nonneg_iff


theorem arg_real_mul (x : ℂ) {r : ℝ} (hr : 0 < r) : arg (r * x) = arg x := by
  /-
    x : Complex
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (HMul.hMul (↑r) x).arg x.arg
  -/
  rcases eq_or_ne x 0 with (rfl | hx); · rw [mul_zero]
                                         /-
                                           🎉 no goals
                                         -/
  conv_lhs =>
    rw [← abs_mul_cos_add_sin_mul_I x, ← mul_assoc, ← ofReal_mul,
      arg_mul_cos_add_sin_mul_I (mul_pos hr (abs.pos hx)) x.arg_mem_Ioc]


theorem arg_mul_real {r : ℝ} (hr : 0 < r) (x : ℂ) : arg (x * r) = arg x :=
  mul_comm x r ▸ arg_real_mul x hr


theorem arg_eq_arg_iff {x y : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) :
    arg x = arg y ↔ (abs y / abs x : ℂ) * x = y := by
  simp only [ext_abs_arg_iff, map_mul, map_div₀, abs_ofReal, abs_abs,
    div_mul_cancel₀ _ (abs.ne_zero hx), eq_self_iff_true, true_and]
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq x.arg y.arg) (Eq (HMul.hMul (HDiv.hDiv ↑(Complex.abs y) ↑(Complex.ab …
  -/
  rw [← ofReal_div, arg_real_mul]
  /-
    case hr
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ LT.lt 0 (HDiv.hDiv (Complex.abs y) (Complex.abs x))
  -/
  exact div_pos (abs.pos hy) (abs.pos hx)
  /-
    🎉 no goals
  -/


@[simp]
                                  /-
                                    ⊢ Eq (Complex.arg 1) 0
                                  -/
theorem arg_one : arg 1 = 0 := by simp [arg, zero_le_one]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
                                         /-
                                           ⊢ Eq (-1).arg Real.pi
                                         -/
theorem arg_neg_one : arg (-1) = π := by simp [arg, le_refl, not_le.2 (zero_lt_one' ℝ)]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                    /-
                                      ⊢ Eq Complex.I.arg (HDiv.hDiv Real.pi 2)
                                    -/
theorem arg_I : arg I = π / 2 := by simp [arg, le_refl]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
                                              /-
                                                ⊢ Eq (Neg.neg Complex.I).arg (Neg.neg (HDiv.hDiv Real.pi 2))
                                              -/
theorem arg_neg_I : arg (-I) = -(π / 2) := by simp [arg, le_refl]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem tan_arg (x : ℂ) : Real.tan (arg x) = x.im / x.re := by
  /-
    x : Complex
    ⊢ Eq (Real.tan x.arg) (HDiv.hDiv x.im x.re)
  -/
  by_cases h : x = 0
    /-
      case pos
      x : Complex
      h : Eq x 0
      ⊢ Eq (Real.tan x.arg) (HDiv.hDiv x.im x.re)
    -/
  · simp only [h, zero_div, Complex.zero_im, Complex.arg_zero, Real.tan_zero, Complex.zero_re]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Complex
    h : Not (Eq x 0)
    ⊢ Eq (Real.tan x.arg) (HDiv.hDiv x.im x.re)
  -/
  rw [Real.tan_eq_sin_div_cos, sin_arg, cos_arg h, div_div_div_cancel_right₀ (abs.ne_zero h)]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      x : Real
                                                                      hx : LE.le 0 x
                                                                      ⊢ Eq (↑x).arg 0
                                                                    -/
theorem arg_ofReal_of_nonneg {x : ℝ} (hx : 0 ≤ x) : arg x = 0 := by simp [arg, hx]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp, norm_cast]
lemma natCast_arg {n : ℕ} : arg n = 0 :=
  ofReal_natCast n ▸ arg_ofReal_of_nonneg n.cast_nonneg


@[simp]
lemma ofNat_arg {n : ℕ} [n.AtLeastTwo] : arg (no_index (OfNat.ofNat n)) = 0 :=
  natCast_arg


theorem arg_eq_zero_iff {z : ℂ} : arg z = 0 ↔ 0 ≤ z.re ∧ z.im = 0 := by
  /-
    z : Complex
    ⊢ Iff (Eq z.arg 0) (And (LE.le 0 z.re) (Eq z.im 0))
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      z : Complex
      h : Eq z.arg 0
      ⊢ And (LE.le 0 z.re) (Eq z.im 0)
    -/
  · rw [← abs_mul_cos_add_sin_mul_I z, h]
    /-
      case refine_1
      z : Complex
      h : Eq z.arg 0
      ⊢ And (LE.le 0 (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos ↑0) (HMul …
    -/
    simp [abs.nonneg]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      z : Complex
      ⊢ And (LE.le 0 z.re) (Eq z.im 0) → Eq z.arg 0
    -/
  · cases' z with x y
    /-
      case refine_2.mk
      x y : Real
      ⊢ And (LE.le 0 { re := x, im := y }.re) (Eq { re := x, im := y }.im 0) → Eq {  …
    -/
    rintro ⟨h, rfl : y = 0⟩
    /-
      case refine_2.mk.intro
      x : Real
      h : LE.le 0 { re := x, im := 0 }.re
      ⊢ Eq { re := x, im := 0 }.arg 0
    -/
    exact arg_ofReal_of_nonneg h
    /-
      🎉 no goals
    -/


open ComplexOrder in
lemma arg_eq_zero_iff_zero_le {z : ℂ} : arg z = 0 ↔ 0 ≤ z := by
  /-
    z : Complex
    ⊢ Iff (Eq z.arg 0) (LE.le 0 z)
  -/
  rw [arg_eq_zero_iff, eq_comm, nonneg_iff]
  /-
    🎉 no goals
  -/


theorem arg_eq_pi_iff {z : ℂ} : arg z = π ↔ z.re < 0 ∧ z.im = 0 := by
  /-
    z : Complex
    ⊢ Iff (Eq z.arg Real.pi) (And (LT.lt z.re 0) (Eq z.im 0))
  -/
  by_cases h₀ : z = 0
    /-
      case pos
      z : Complex
      h₀ : Eq z 0
      ⊢ Iff (Eq z.arg Real.pi) (And (LT.lt z.re 0) (Eq z.im 0))
    -/
  · simp [h₀, lt_irrefl, Real.pi_ne_zero.symm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    z : Complex
    h₀ : Not (Eq z 0)
    ⊢ Iff (Eq z.arg Real.pi) (And (LT.lt z.re 0) (Eq z.im 0))
  -/
  constructor
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ Eq z.arg Real.pi → And (LT.lt z.re 0) (Eq z.im 0)
    -/
  · intro h
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg Real.pi
      ⊢ And (LT.lt z.re 0) (Eq z.im 0)
    -/
    rw [← abs_mul_cos_add_sin_mul_I z, h]
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg Real.pi
      ⊢ And (LT.lt (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos ↑Real.pi) ( …
    -/
    simp [h₀]
    /-
      🎉 no goals
    -/
    /-
      case neg.mpr
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ And (LT.lt z.re 0) (Eq z.im 0) → Eq z.arg Real.pi
    -/
  · cases' z with x y
    /-
      case neg.mpr.mk
      x y : Real
      h₀ : Not (Eq { re := x, im := y } 0)
      ⊢ And (LT.lt { re := x, im := y }.re 0) (Eq { re := x, im := y }.im 0) → Eq {  …
    -/
    rintro ⟨h : x < 0, rfl : y = 0⟩
    /-
      case neg.mpr.mk.intro
      x : Real
      h : LT.lt x 0
      h₀ : Not (Eq { re := x, im := 0 } 0)
      ⊢ Eq { re := x, im := 0 }.arg Real.pi
    -/
    rw [← arg_neg_one, ← arg_real_mul (-1) (neg_pos.2 h)]
    /-
      case neg.mpr.mk.intro
      x : Real
      h : LT.lt x 0
      h₀ : Not (Eq { re := x, im := 0 } 0)
      ⊢ Eq { re := x, im := 0 }.arg (HMul.hMul (↑(Neg.neg x)) (-1)).arg
    -/
    simp [← ofReal_def]
    /-
      🎉 no goals
    -/


open ComplexOrder in
lemma arg_eq_pi_iff_lt_zero {z : ℂ} : arg z = π ↔ z < 0 := arg_eq_pi_iff


theorem arg_lt_pi_iff {z : ℂ} : arg z < π ↔ 0 ≤ z.re ∨ z.im ≠ 0 := by
  /-
    z : Complex
    ⊢ Iff (LT.lt z.arg Real.pi) (Or (LE.le 0 z.re) (Ne z.im 0))
  -/
  rw [(arg_le_pi z).lt_iff_ne, not_iff_comm, not_or, not_le, Classical.not_not, arg_eq_pi_iff]
  /-
    🎉 no goals
  -/


theorem arg_ofReal_of_neg {x : ℝ} (hx : x < 0) : arg x = π :=
  arg_eq_pi_iff.2 ⟨hx, rfl⟩


theorem arg_eq_pi_div_two_iff {z : ℂ} : arg z = π / 2 ↔ z.re = 0 ∧ 0 < z.im := by
  /-
    z : Complex
    ⊢ Iff (Eq z.arg (HDiv.hDiv Real.pi 2)) (And (Eq z.re 0) (LT.lt 0 z.im))
  -/
  by_cases h₀ : z = 0; · simp [h₀, lt_irrefl, Real.pi_div_two_pos.ne]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    z : Complex
    h₀ : Not (Eq z 0)
    ⊢ Iff (Eq z.arg (HDiv.hDiv Real.pi 2)) (And (Eq z.re 0) (LT.lt 0 z.im))
  -/
  constructor
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ Eq z.arg (HDiv.hDiv Real.pi 2) → And (Eq z.re 0) (LT.lt 0 z.im)
    -/
  · intro h
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg (HDiv.hDiv Real.pi 2)
      ⊢ And (Eq z.re 0) (LT.lt 0 z.im)
    -/
    rw [← abs_mul_cos_add_sin_mul_I z, h]
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg (HDiv.hDiv Real.pi 2)
      ⊢ And (Eq (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos ↑(HDiv.hDiv Re …
    -/
    simp [h₀]
    /-
      🎉 no goals
    -/
    /-
      case neg.mpr
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ And (Eq z.re 0) (LT.lt 0 z.im) → Eq z.arg (HDiv.hDiv Real.pi 2)
    -/
  · cases' z with x y
    /-
      case neg.mpr.mk
      x y : Real
      h₀ : Not (Eq { re := x, im := y } 0)
      ⊢ And (Eq { re := x, im := y }.re 0) (LT.lt 0 { re := x, im := y }.im) → Eq {  …
    -/
    rintro ⟨rfl : x = 0, hy : 0 < y⟩
    /-
      case neg.mpr.mk.intro
      y : Real
      h₀ : Not (Eq { re := 0, im := y } 0)
      hy : LT.lt 0 y
      ⊢ Eq { re := 0, im := y }.arg (HDiv.hDiv Real.pi 2)
    -/
    rw [← arg_I, ← arg_real_mul I hy, ofReal_mul', I_re, I_im, mul_zero, mul_one]
    /-
      🎉 no goals
    -/


theorem arg_eq_neg_pi_div_two_iff {z : ℂ} : arg z = -(π / 2) ↔ z.re = 0 ∧ z.im < 0 := by
  /-
    z : Complex
    ⊢ Iff (Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2))) (And (Eq z.re 0) (LT.lt z.im  …
  -/
  by_cases h₀ : z = 0; · simp [h₀, lt_irrefl, Real.pi_ne_zero]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    z : Complex
    h₀ : Not (Eq z 0)
    ⊢ Iff (Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2))) (And (Eq z.re 0) (LT.lt z.im  …
  -/
  constructor
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2)) → And (Eq z.re 0) (LT.lt z.im 0)
    -/
  · intro h
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2))
      ⊢ And (Eq z.re 0) (LT.lt z.im 0)
    -/
    rw [← abs_mul_cos_add_sin_mul_I z, h]
    /-
      case neg.mp
      z : Complex
      h₀ : Not (Eq z 0)
      h : Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2))
      ⊢ And (Eq (HMul.hMul (↑(Complex.abs z)) (HAdd.hAdd (Complex.cos ↑(Neg.neg (HDi …
    -/
    simp [h₀]
    /-
      🎉 no goals
    -/
    /-
      case neg.mpr
      z : Complex
      h₀ : Not (Eq z 0)
      ⊢ And (Eq z.re 0) (LT.lt z.im 0) → Eq z.arg (Neg.neg (HDiv.hDiv Real.pi 2))
    -/
  · cases' z with x y
    /-
      case neg.mpr.mk
      x y : Real
      h₀ : Not (Eq { re := x, im := y } 0)
      ⊢ And (Eq { re := x, im := y }.re 0) (LT.lt { re := x, im := y }.im 0) → Eq {  …
    -/
    rintro ⟨rfl : x = 0, hy : y < 0⟩
    /-
      case neg.mpr.mk.intro
      y : Real
      h₀ : Not (Eq { re := 0, im := y } 0)
      hy : LT.lt y 0
      ⊢ Eq { re := 0, im := y }.arg (Neg.neg (HDiv.hDiv Real.pi 2))
    -/
    rw [← arg_neg_I, ← arg_real_mul (-I) (neg_pos.2 hy), mk_eq_add_mul_I]
    /-
      case neg.mpr.mk.intro
      y : Real
      h₀ : Not (Eq { re := 0, im := y } 0)
      hy : LT.lt y 0
      ⊢ Eq (HAdd.hAdd (↑0) (HMul.hMul (↑y) Complex.I)).arg (HMul.hMul (↑(Neg.neg y)) …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem arg_of_re_nonneg {x : ℂ} (hx : 0 ≤ x.re) : arg x = Real.arcsin (x.im / abs x) :=
  if_pos hx


theorem arg_of_re_neg_of_im_nonneg {x : ℂ} (hx_re : x.re < 0) (hx_im : 0 ≤ x.im) :
    arg x = Real.arcsin ((-x).im / abs x) + π := by
  /-
    x : Complex
    hx_re : LT.lt x.re 0
    hx_im : LE.le 0 x.im
    ⊢ Eq x.arg (HAdd.hAdd (Real.arcsin (HDiv.hDiv (Neg.neg x).im (Complex.abs x))) …
  -/
  simp only [arg, hx_re.not_le, hx_im, if_true, if_false]
  /-
    🎉 no goals
  -/


theorem arg_of_re_neg_of_im_neg {x : ℂ} (hx_re : x.re < 0) (hx_im : x.im < 0) :
    arg x = Real.arcsin ((-x).im / abs x) - π := by
  /-
    x : Complex
    hx_re : LT.lt x.re 0
    hx_im : LT.lt x.im 0
    ⊢ Eq x.arg (HSub.hSub (Real.arcsin (HDiv.hDiv (Neg.neg x).im (Complex.abs x))) …
  -/
  simp only [arg, hx_re.not_le, hx_im.not_le, if_false]
  /-
    🎉 no goals
  -/


theorem arg_of_im_nonneg_of_ne_zero {z : ℂ} (h₁ : 0 ≤ z.im) (h₂ : z ≠ 0) :
    arg z = Real.arccos (z.re / abs z) := by
  /-
    z : Complex
    h₁ : LE.le 0 z.im
    h₂ : Ne z 0
    ⊢ Eq z.arg (Real.arccos (HDiv.hDiv z.re (Complex.abs z)))
  -/
  rw [← cos_arg h₂, Real.arccos_cos (arg_nonneg_iff.2 h₁) (arg_le_pi _)]
  /-
    🎉 no goals
  -/


theorem arg_of_im_pos {z : ℂ} (hz : 0 < z.im) : arg z = Real.arccos (z.re / abs z) :=
  arg_of_im_nonneg_of_ne_zero hz.le fun h => hz.ne' <| h.symm ▸ rfl


theorem arg_of_im_neg {z : ℂ} (hz : z.im < 0) : arg z = -Real.arccos (z.re / abs z) := by
  /-
    z : Complex
    hz : LT.lt z.im 0
    ⊢ Eq z.arg (Neg.neg (Real.arccos (HDiv.hDiv z.re (Complex.abs z))))
  -/
  have h₀ : z ≠ 0 := mt (congr_arg im) hz.ne
  /-
    z : Complex
    hz : LT.lt z.im 0
    h₀ : Ne z 0
    ⊢ Eq z.arg (Neg.neg (Real.arccos (HDiv.hDiv z.re (Complex.abs z))))
  -/
  rw [← cos_arg h₀, ← Real.cos_neg, Real.arccos_cos, neg_neg]
  /-
    case hx₁
    z : Complex
    hz : LT.lt z.im 0
    h₀ : Ne z 0
    ⊢ LE.le 0 (Neg.neg z.arg)
  -/
  exacts [neg_nonneg.2 (arg_neg_iff.2 hz).le, neg_le.2 (neg_pi_lt_arg z).le]
  /-
    🎉 no goals
  -/


theorem arg_conj (x : ℂ) : arg (conj x) = if arg x = π then π else -arg x := by
  simp_rw [arg_eq_pi_iff, arg, neg_im, conj_im, conj_re, abs_conj, neg_div, neg_neg,
    Real.arcsin_neg]
  /-
    x : Complex
    ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
  -/
  rcases lt_trichotomy x.re 0 with (hr | hr | hr) <;>
    /-
      case inl
      x : Complex
      hr : LT.lt x.re 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
    rcases lt_trichotomy x.im 0 with (hi | hi | hi)
    /-
      case inl.inl
      x : Complex
      hr : LT.lt x.re 0
      hi : LT.lt x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.not_le, hi.le, hi.ne, not_le.2 hi, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case inl.inr.inl
      x : Complex
      hr : LT.lt x.re 0
      hi : Eq x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.not_le, hi]
    /-
      🎉 no goals
    -/
    /-
      case inl.inr.inr
      x : Complex
      hr : LT.lt x.re 0
      hi : LT.lt 0 x.im
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.not_le, hi.ne.symm, hi.le, not_le.2 hi, sub_eq_neg_add]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.inl
      x : Complex
      hr : Eq x.re 0
      hi : LT.lt x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.inr.inl
      x : Complex
      hr : Eq x.re 0
      hi : Eq x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.inr.inr
      x : Complex
      hr : Eq x.re 0
      hi : LT.lt 0 x.im
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inl
      x : Complex
      hr : LT.lt 0 x.re
      hi : LT.lt x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.le, hi.ne]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inl
      x : Complex
      hr : LT.lt 0 x.re
      hi : Eq x.im 0
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.le, hr.le.not_lt]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inr
      x : Complex
      hr : LT.lt 0 x.re
      hi : LT.lt 0 x.im
      ⊢ Eq (ite (LE.le 0 x.re) (Neg.neg (Real.arcsin (HDiv.hDiv x.im (Complex.abs x) …
    -/
  · simp [hr, hr.le, hr.le.not_lt]
    /-
      🎉 no goals
    -/


theorem arg_inv (x : ℂ) : arg x⁻¹ = if arg x = π then π else -arg x := by
  /-
    x : Complex
    ⊢ Eq (Inv.inv x).arg (ite (Eq x.arg Real.pi) Real.pi (Neg.neg x.arg))
  -/
  rw [← arg_conj, inv_def, mul_comm]
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (↑(Inv.inv (Complex.normSq x))) ((starRingEnd Complex) x)).arg …
  -/
  by_cases hx : x = 0
    /-
      case pos
      x : Complex
      hx : Eq x 0
      ⊢ Eq (HMul.hMul (↑(Inv.inv (Complex.normSq x))) ((starRingEnd Complex) x)).arg …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Complex
      hx : Not (Eq x 0)
      ⊢ Eq (HMul.hMul (↑(Inv.inv (Complex.normSq x))) ((starRingEnd Complex) x)).arg …
    -/
  · exact arg_real_mul (conj x) (by simp [hx])
    /-
      🎉 no goals
    -/


                                                              /-
                                                                x : Complex
                                                                ⊢ Eq (_root_.abs (Inv.inv x).arg) (_root_.abs x.arg)
                                                              -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
@[simp] lemma abs_arg_inv (x : ℂ) : |x⁻¹.arg| = |x.arg| := by rw [arg_inv]; split_ifs <;> simp [*]
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/

-- TODO: Replace the next two lemmas by general facts about periodic functions

lemma abs_eq_one_iff' : abs x = 1 ↔ ∃ θ ∈ Set.Ioc (-π) π, exp (θ * I) = x := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.abs x) 1) (Exists fun θ => And (Membership.mem (Set.Ioc (Ne …
  -/
  rw [abs_eq_one_iff]
  /-
    x : Complex
    ⊢ Iff (Exists fun θ => Eq (Complex.exp (HMul.hMul (↑θ) Complex.I)) x) (Exists  …
  -/
  constructor
    /-
      case mp
      x : Complex
      ⊢ (Exists fun θ => Eq (Complex.exp (HMul.hMul (↑θ) Complex.I)) x) → Exists fun …
    -/
  · rintro ⟨θ, rfl⟩
    /-
      case mp.intro
      θ : Real
      ⊢ Exists fun θ_1 => And (Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ_ …
    -/
    refine ⟨toIocMod (mul_pos two_pos Real.pi_pos) (-π) θ, ?_, ?_⟩
      /-
        case mp.intro.refine_1
        θ : Real
        ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (toIocMod ⋯ (Neg.neg Real …
      -/
    · convert toIocMod_mem_Ioc _ _ _
      /-
        case h.e'_4.h.e'_4
        θ : Real
        ⊢ Eq Real.pi (HAdd.hAdd (Neg.neg Real.pi) (HMul.hMul 2 Real.pi))
      -/
      ring
      /-
        🎉 no goals
      -/
    · rw [eq_sub_of_add_eq <| toIocMod_add_toIocDiv_zsmul _ _ θ, ofReal_sub,
      ofReal_zsmul, ofReal_mul, ofReal_ofNat, exp_mul_I_periodic.sub_zsmul_eq]
    /-
      case mpr
      x : Complex
      ⊢ (Exists fun θ => And (Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ)  …
    -/
  · rintro ⟨θ, _, rfl⟩
    /-
      case mpr.intro.intro
      θ : Real
      left✝ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Exists fun θ_1 => Eq (Complex.exp (HMul.hMul (↑θ_1) Complex.I)) (Complex.exp …
    -/
    exact ⟨θ, rfl⟩
    /-
      🎉 no goals
    -/


lemma image_exp_Ioc_eq_sphere : (fun θ : ℝ ↦ exp (θ * I)) '' Set.Ioc (-π) π = sphere 0 1 := by
  /-
    ⊢ Eq (Set.image (fun θ => Complex.exp (HMul.hMul (↑θ) Complex.I)) (Set.Ioc (Ne …
  -/
  ext; simpa using abs_eq_one_iff'.symm
       /-
         🎉 no goals
       -/


theorem arg_le_pi_div_two_iff {z : ℂ} : arg z ≤ π / 2 ↔ 0 ≤ re z ∨ im z < 0 := by
  /-
    z : Complex
    ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (Or (LE.le 0 z.re) (LT.lt z.im 0))
  -/
  rcases le_or_lt 0 (re z) with hre | hre
    /-
      case inl
      z : Complex
      hre : LE.le 0 z.re
      ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (Or (LE.le 0 z.re) (LT.lt z.im 0))
    -/
  · simp only [hre, arg_of_re_nonneg hre, Real.arcsin_le_pi_div_two, true_or]
    /-
      🎉 no goals
    -/
  /-
    case inr
    z : Complex
    hre : LT.lt z.re 0
    ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (Or (LE.le 0 z.re) (LT.lt z.im 0))
  -/
  simp only [hre.not_le, false_or]
  /-
    case inr
    z : Complex
    hre : LT.lt z.re 0
    ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (LT.lt z.im 0)
  -/
  rcases le_or_lt 0 (im z) with him | him
    /-
      case inr.inl
      z : Complex
      hre : LT.lt z.re 0
      him : LE.le 0 z.im
      ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (LT.lt z.im 0)
    -/
  · simp only [him.not_lt]
    rw [iff_false, not_le, arg_of_re_neg_of_im_nonneg hre him, ← sub_lt_iff_lt_add, half_sub,
      Real.neg_pi_div_two_lt_arcsin, neg_im, neg_div, neg_lt_neg_iff, div_lt_one, ←
      _root_.abs_of_nonneg him, abs_im_lt_abs]
    /-
      case inr.inl
      z : Complex
      hre : LT.lt z.re 0
      him : LE.le 0 z.im
      ⊢ Ne z.re 0
    -/
    exacts [hre.ne, abs.pos <| ne_of_apply_ne re hre.ne]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      z : Complex
      hre : LT.lt z.re 0
      him : LT.lt z.im 0
      ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) (LT.lt z.im 0)
    -/
  · simp only [him]
    /-
      case inr.inr
      z : Complex
      hre : LT.lt z.re 0
      him : LT.lt z.im 0
      ⊢ Iff (LE.le z.arg (HDiv.hDiv Real.pi 2)) True
    -/
    rw [iff_true, arg_of_re_neg_of_im_neg hre him]
    /-
      case inr.inr
      z : Complex
      hre : LT.lt z.re 0
      him : LT.lt z.im 0
      ⊢ LE.le (HSub.hSub (Real.arcsin (HDiv.hDiv (Neg.neg z).im (Complex.abs z))) Re …
    -/
    exact (sub_le_self _ Real.pi_pos.le).trans (Real.arcsin_le_pi_div_two _)
    /-
      🎉 no goals
    -/


theorem neg_pi_div_two_le_arg_iff {z : ℂ} : -(π / 2) ≤ arg z ↔ 0 ≤ re z ∨ 0 ≤ im z := by
  /-
    z : Complex
    ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (Or (LE.le 0 z.re) (LE.le  …
  -/
  rcases le_or_lt 0 (re z) with hre | hre
    /-
      case inl
      z : Complex
      hre : LE.le 0 z.re
      ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (Or (LE.le 0 z.re) (LE.le  …
    -/
  · simp only [hre, arg_of_re_nonneg hre, Real.neg_pi_div_two_le_arcsin, true_or]
    /-
      🎉 no goals
    -/
  /-
    case inr
    z : Complex
    hre : LT.lt z.re 0
    ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (Or (LE.le 0 z.re) (LE.le  …
  -/
  simp only [hre.not_le, false_or]
  /-
    case inr
    z : Complex
    hre : LT.lt z.re 0
    ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (LE.le 0 z.im)
  -/
  rcases le_or_lt 0 (im z) with him | him
    /-
      case inr.inl
      z : Complex
      hre : LT.lt z.re 0
      him : LE.le 0 z.im
      ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (LE.le 0 z.im)
    -/
  · simp only [him]
    /-
      case inr.inl
      z : Complex
      hre : LT.lt z.re 0
      him : LE.le 0 z.im
      ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) True
    -/
    rw [iff_true, arg_of_re_neg_of_im_nonneg hre him]
    /-
      case inr.inl
      z : Complex
      hre : LT.lt z.re 0
      him : LE.le 0 z.im
      ⊢ LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) (HAdd.hAdd (Real.arcsin (HDiv.hDiv (Ne …
    -/
    exact (Real.neg_pi_div_two_le_arcsin _).trans (le_add_of_nonneg_right Real.pi_pos.le)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      z : Complex
      hre : LT.lt z.re 0
      him : LT.lt z.im 0
      ⊢ Iff (LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (LE.le 0 z.im)
    -/
  · simp only [him.not_le]
    rw [iff_false, not_le, arg_of_re_neg_of_im_neg hre him, sub_lt_iff_lt_add', ←
      sub_eq_add_neg, sub_half, Real.arcsin_lt_pi_div_two, div_lt_one, neg_im, ← abs_of_neg him,
      abs_im_lt_abs]
    /-
      case inr.inr
      z : Complex
      hre : LT.lt z.re 0
      him : LT.lt z.im 0
      ⊢ Ne z.re 0
    -/
    exacts [hre.ne, abs.pos <| ne_of_apply_ne re hre.ne]
    /-
      🎉 no goals
    -/


lemma neg_pi_div_two_lt_arg_iff {z : ℂ} : -(π / 2) < arg z ↔ 0 < re z ∨ 0 ≤ im z := by
  /-
    z : Complex
    ⊢ Iff (LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) z.arg) (Or (LT.lt 0 z.re) (LE.le  …
  -/
  rw [lt_iff_le_and_ne, neg_pi_div_two_le_arg_iff, ne_comm, Ne, arg_eq_neg_pi_div_two_iff]
  /-
    z : Complex
    ⊢ Iff (And (Or (LE.le 0 z.re) (LE.le 0 z.im)) (Not (And (Eq z.re 0) (LT.lt z.i …
  -/
  rcases lt_trichotomy z.re 0 with hre | hre | hre
    /-
      case inl
      z : Complex
      hre : LT.lt z.re 0
      ⊢ Iff (And (Or (LE.le 0 z.re) (LE.le 0 z.im)) (Not (And (Eq z.re 0) (LT.lt z.i …
    -/
  · simp [hre.ne, hre.not_le, hre.not_lt]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      z : Complex
      hre : Eq z.re 0
      ⊢ Iff (And (Or (LE.le 0 z.re) (LE.le 0 z.im)) (Not (And (Eq z.re 0) (LT.lt z.i …
    -/
  · simp [hre]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      z : Complex
      hre : LT.lt 0 z.re
      ⊢ Iff (And (Or (LE.le 0 z.re) (LE.le 0 z.im)) (Not (And (Eq z.re 0) (LT.lt z.i …
    -/
  · simp [hre, hre.le, hre.ne']
    /-
      🎉 no goals
    -/


lemma arg_lt_pi_div_two_iff {z : ℂ} : arg z < π / 2 ↔ 0 < re z ∨ im z < 0 ∨ z = 0 := by
  /-
    z : Complex
    ⊢ Iff (LT.lt z.arg (HDiv.hDiv Real.pi 2)) (Or (LT.lt 0 z.re) (Or (LT.lt z.im 0 …
  -/
  rw [lt_iff_le_and_ne, arg_le_pi_div_two_iff, Ne, arg_eq_pi_div_two_iff]
  /-
    z : Complex
    ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
  -/
  rcases lt_trichotomy z.re 0 with hre | hre | hre
    /-
      case inl
      z : Complex
      hre : LT.lt z.re 0
      ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
    -/
  · have : z ≠ 0 := by simp [Complex.ext_iff, hre.ne]
    /-
      case inl
      z : Complex
      hre : LT.lt z.re 0
      this : Ne z 0
      ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
    -/
    simp [hre.ne, hre.not_le, hre.not_lt, this]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      z : Complex
      hre : Eq z.re 0
      ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
    -/
  · have : z = 0 ↔ z.im = 0 := by simp [Complex.ext_iff, hre]
    /-
      case inr.inl
      z : Complex
      hre : Eq z.re 0
      this : Iff (Eq z 0) (Eq z.im 0)
      ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
    -/
    simp [hre, this, or_comm, le_iff_eq_or_lt]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      z : Complex
      hre : LT.lt 0 z.re
      ⊢ Iff (And (Or (LE.le 0 z.re) (LT.lt z.im 0)) (Not (And (Eq z.re 0) (LT.lt 0 z …
    -/
  · simp [hre, hre.le, hre.ne']
    /-
      🎉 no goals
    -/


@[simp]
theorem abs_arg_le_pi_div_two_iff {z : ℂ} : |arg z| ≤ π / 2 ↔ 0 ≤ re z := by
  rw [abs_le, arg_le_pi_div_two_iff, neg_pi_div_two_le_arg_iff, ← or_and_left, ← not_le,
    and_not_self_iff, or_false]


@[simp]
theorem abs_arg_lt_pi_div_two_iff {z : ℂ} : |arg z| < π / 2 ↔ 0 < re z ∨ z = 0 := by
  /-
    z : Complex
    ⊢ Iff (LT.lt (_root_.abs z.arg) (HDiv.hDiv Real.pi 2)) (Or (LT.lt 0 z.re) (Eq  …
  -/
  rw [abs_lt, arg_lt_pi_div_two_iff, neg_pi_div_two_lt_arg_iff, ← or_and_left]
  /-
    z : Complex
    ⊢ Iff (Or (LT.lt 0 z.re) (And (LE.le 0 z.im) (Or (LT.lt z.im 0) (Eq z 0)))) (O …
  -/
  rcases eq_or_ne z 0 with hz | hz
    /-
      case inl
      z : Complex
      hz : Eq z 0
      ⊢ Iff (Or (LT.lt 0 z.re) (And (LE.le 0 z.im) (Or (LT.lt z.im 0) (Eq z 0)))) (O …
    -/
  · simp [hz]
    /-
      🎉 no goals
    -/
    /-
      case inr
      z : Complex
      hz : Ne z 0
      ⊢ Iff (Or (LT.lt 0 z.re) (And (LE.le 0 z.im) (Or (LT.lt z.im 0) (Eq z 0)))) (O …
    -/
  · simp_rw [hz, or_false, ← not_lt, not_and_self_iff, or_false]
    /-
      🎉 no goals
    -/


@[simp]
theorem arg_conj_coe_angle (x : ℂ) : (arg (conj x) : Real.Angle) = -arg x := by
  /-
    x : Complex
    ⊢ Eq (↑((starRingEnd Complex) x).arg) (Neg.neg ↑x.arg)
  -/
                             /-
                               🎉 no goals
                             -/
  by_cases h : arg x = π <;> simp [arg_conj, h]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem arg_inv_coe_angle (x : ℂ) : (arg x⁻¹ : Real.Angle) = -arg x := by
  /-
    x : Complex
    ⊢ Eq (↑(Inv.inv x).arg) (Neg.neg ↑x.arg)
  -/
                             /-
                               🎉 no goals
                             -/
  by_cases h : arg x = π <;> simp [arg_inv, h]
                             /-
                               🎉 no goals
                             -/


theorem arg_neg_eq_arg_sub_pi_of_im_pos {x : ℂ} (hi : 0 < x.im) : arg (-x) = arg x - π := by
  /-
    x : Complex
    hi : LT.lt 0 x.im
    ⊢ Eq (Neg.neg x).arg (HSub.hSub x.arg Real.pi)
  -/
  rw [arg_of_im_pos hi, arg_of_im_neg (show (-x).im < 0 from Left.neg_neg_iff.2 hi)]
  /-
    x : Complex
    hi : LT.lt 0 x.im
    ⊢ Eq (Neg.neg (Real.arccos (HDiv.hDiv (Neg.neg x).re (Complex.abs (Neg.neg x)) …
  -/
  simp [neg_div, Real.arccos_neg]
  /-
    🎉 no goals
  -/


theorem arg_neg_eq_arg_add_pi_of_im_neg {x : ℂ} (hi : x.im < 0) : arg (-x) = arg x + π := by
  /-
    x : Complex
    hi : LT.lt x.im 0
    ⊢ Eq (Neg.neg x).arg (HAdd.hAdd x.arg Real.pi)
  -/
  rw [arg_of_im_neg hi, arg_of_im_pos (show 0 < (-x).im from Left.neg_pos_iff.2 hi)]
  /-
    x : Complex
    hi : LT.lt x.im 0
    ⊢ Eq (Real.arccos (HDiv.hDiv (Neg.neg x).re (Complex.abs (Neg.neg x)))) (HAdd. …
  -/
  simp [neg_div, Real.arccos_neg, add_comm, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem arg_neg_eq_arg_sub_pi_iff {x : ℂ} :
    arg (-x) = arg x - π ↔ 0 < x.im ∨ x.im = 0 ∧ x.re < 0 := by
  /-
    x : Complex
    ⊢ Iff (Eq (Neg.neg x).arg (HSub.hSub x.arg Real.pi)) (Or (LT.lt 0 x.im) (And ( …
  -/
  rcases lt_trichotomy x.im 0 with (hi | hi | hi)
  · simp [hi, hi.ne, hi.not_lt, arg_neg_eq_arg_add_pi_of_im_neg, sub_eq_add_neg, ←
      add_eq_zero_iff_eq_neg, Real.pi_ne_zero]
    /-
      case inr.inl
      x : Complex
      hi : Eq x.im 0
      ⊢ Iff (Eq (Neg.neg x).arg (HSub.hSub x.arg Real.pi)) (Or (LT.lt 0 x.im) (And ( …
    -/
  · rw [(ext rfl hi : x = x.re)]
    /-
      case inr.inl
      x : Complex
      hi : Eq x.im 0
      ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HSub.hSub (↑x.re).arg Real.pi)) (Or (LT.lt 0 (↑ …
    -/
    rcases lt_trichotomy x.re 0 with (hr | hr | hr)
      /-
        case inr.inl.inl
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt x.re 0
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HSub.hSub (↑x.re).arg Real.pi)) (Or (LT.lt 0 (↑ …
      -/
    · rw [arg_ofReal_of_neg hr, ← ofReal_neg, arg_ofReal_of_nonneg (Left.neg_pos_iff.2 hr).le]
      /-
        case inr.inl.inl
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt x.re 0
        ⊢ Iff (Eq 0 (HSub.hSub Real.pi Real.pi)) (Or (LT.lt 0 (↑x.re).im) (And (Eq (↑x …
      -/
      simp [hr]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inl
        x : Complex
        hi : Eq x.im 0
        hr : Eq x.re 0
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HSub.hSub (↑x.re).arg Real.pi)) (Or (LT.lt 0 (↑ …
      -/
    · simp [hr, hi, Real.pi_ne_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inr
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt 0 x.re
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HSub.hSub (↑x.re).arg Real.pi)) (Or (LT.lt 0 (↑ …
      -/
    · rw [arg_ofReal_of_nonneg hr.le, ← ofReal_neg, arg_ofReal_of_neg (Left.neg_neg_iff.2 hr)]
      /-
        case inr.inl.inr.inr
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt 0 x.re
        ⊢ Iff (Eq Real.pi (HSub.hSub 0 Real.pi)) (Or (LT.lt 0 (↑x.re).im) (And (Eq (↑x …
      -/
      simp [hr.not_lt, ← add_eq_zero_iff_eq_neg, Real.pi_ne_zero]
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      x : Complex
      hi : LT.lt 0 x.im
      ⊢ Iff (Eq (Neg.neg x).arg (HSub.hSub x.arg Real.pi)) (Or (LT.lt 0 x.im) (And ( …
    -/
  · simp [hi, arg_neg_eq_arg_sub_pi_of_im_pos]
    /-
      🎉 no goals
    -/


theorem arg_neg_eq_arg_add_pi_iff {x : ℂ} :
    arg (-x) = arg x + π ↔ x.im < 0 ∨ x.im = 0 ∧ 0 < x.re := by
  /-
    x : Complex
    ⊢ Iff (Eq (Neg.neg x).arg (HAdd.hAdd x.arg Real.pi)) (Or (LT.lt x.im 0) (And ( …
  -/
  rcases lt_trichotomy x.im 0 with (hi | hi | hi)
    /-
      case inl
      x : Complex
      hi : LT.lt x.im 0
      ⊢ Iff (Eq (Neg.neg x).arg (HAdd.hAdd x.arg Real.pi)) (Or (LT.lt x.im 0) (And ( …
    -/
  · simp [hi, arg_neg_eq_arg_add_pi_of_im_neg]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x : Complex
      hi : Eq x.im 0
      ⊢ Iff (Eq (Neg.neg x).arg (HAdd.hAdd x.arg Real.pi)) (Or (LT.lt x.im 0) (And ( …
    -/
  · rw [(ext rfl hi : x = x.re)]
    /-
      case inr.inl
      x : Complex
      hi : Eq x.im 0
      ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HAdd.hAdd (↑x.re).arg Real.pi)) (Or (LT.lt (↑x. …
    -/
    rcases lt_trichotomy x.re 0 with (hr | hr | hr)
      /-
        case inr.inl.inl
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt x.re 0
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HAdd.hAdd (↑x.re).arg Real.pi)) (Or (LT.lt (↑x. …
      -/
    · rw [arg_ofReal_of_neg hr, ← ofReal_neg, arg_ofReal_of_nonneg (Left.neg_pos_iff.2 hr).le]
      /-
        case inr.inl.inl
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt x.re 0
        ⊢ Iff (Eq 0 (HAdd.hAdd Real.pi Real.pi)) (Or (LT.lt (↑x.re).im 0) (And (Eq (↑x …
      -/
      simp [hr.not_lt, ← two_mul, Real.pi_ne_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inl
        x : Complex
        hi : Eq x.im 0
        hr : Eq x.re 0
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HAdd.hAdd (↑x.re).arg Real.pi)) (Or (LT.lt (↑x. …
      -/
    · simp [hr, hi, Real.pi_ne_zero.symm]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr.inr
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt 0 x.re
        ⊢ Iff (Eq (Neg.neg ↑x.re).arg (HAdd.hAdd (↑x.re).arg Real.pi)) (Or (LT.lt (↑x. …
      -/
    · rw [arg_ofReal_of_nonneg hr.le, ← ofReal_neg, arg_ofReal_of_neg (Left.neg_neg_iff.2 hr)]
      /-
        case inr.inl.inr.inr
        x : Complex
        hi : Eq x.im 0
        hr : LT.lt 0 x.re
        ⊢ Iff (Eq Real.pi (HAdd.hAdd 0 Real.pi)) (Or (LT.lt (↑x.re).im 0) (And (Eq (↑x …
      -/
      simp [hr]
      /-
        🎉 no goals
      -/
  · simp [hi, hi.ne.symm, hi.not_lt, arg_neg_eq_arg_sub_pi_of_im_pos, sub_eq_add_neg, ←
      add_eq_zero_iff_neg_eq, Real.pi_ne_zero]


theorem arg_neg_coe_angle {x : ℂ} (hx : x ≠ 0) : (arg (-x) : Real.Angle) = arg x + π := by
  /-
    x : Complex
    hx : Ne x 0
    ⊢ Eq (↑(Neg.neg x).arg) (HAdd.hAdd ↑x.arg ↑Real.pi)
  -/
  rcases lt_trichotomy x.im 0 with (hi | hi | hi)
    /-
      case inl
      x : Complex
      hx : Ne x 0
      hi : LT.lt x.im 0
      ⊢ Eq (↑(Neg.neg x).arg) (HAdd.hAdd ↑x.arg ↑Real.pi)
    -/
  · rw [arg_neg_eq_arg_add_pi_of_im_neg hi, Real.Angle.coe_add]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x : Complex
      hx : Ne x 0
      hi : Eq x.im 0
      ⊢ Eq (↑(Neg.neg x).arg) (HAdd.hAdd ↑x.arg ↑Real.pi)
    -/
  · rw [(ext rfl hi : x = x.re)]
    /-
      case inr.inl
      x : Complex
      hx : Ne x 0
      hi : Eq x.im 0
      ⊢ Eq (↑(Neg.neg ↑x.re).arg) (HAdd.hAdd ↑(↑x.re).arg ↑Real.pi)
    -/
    rcases lt_trichotomy x.re 0 with (hr | hr | hr)
    · rw [arg_ofReal_of_neg hr, ← ofReal_neg, arg_ofReal_of_nonneg (Left.neg_pos_iff.2 hr).le, ←
        Real.Angle.coe_add, ← two_mul, Real.Angle.coe_two_pi, Real.Angle.coe_zero]
      /-
        case inr.inl.inr.inl
        x : Complex
        hx : Ne x 0
        hi : Eq x.im 0
        hr : Eq x.re 0
        ⊢ Eq (↑(Neg.neg ↑x.re).arg) (HAdd.hAdd ↑(↑x.re).arg ↑Real.pi)
      -/
    · exact False.elim (hx (ext hr hi))
      /-
        🎉 no goals
      -/
    · rw [arg_ofReal_of_nonneg hr.le, ← ofReal_neg, arg_ofReal_of_neg (Left.neg_neg_iff.2 hr),
        Real.Angle.coe_zero, zero_add]
    /-
      case inr.inr
      x : Complex
      hx : Ne x 0
      hi : LT.lt 0 x.im
      ⊢ Eq (↑(Neg.neg x).arg) (HAdd.hAdd ↑x.arg ↑Real.pi)
    -/
  · rw [arg_neg_eq_arg_sub_pi_of_im_pos hi, Real.Angle.coe_sub, Real.Angle.sub_coe_pi_eq_add_coe_pi]
    /-
      🎉 no goals
    -/


theorem arg_mul_cos_add_sin_mul_I_eq_toIocMod {r : ℝ} (hr : 0 < r) (θ : ℝ) :
    arg (r * (cos θ + sin θ * I)) = toIocMod Real.two_pi_pos (-π) θ := by
  have hi : toIocMod Real.two_pi_pos (-π) θ ∈ Set.Ioc (-π) π := by
    convert toIocMod_mem_Ioc _ _ θ
    ring
  /-
    r : Real
    hr : LT.lt 0 r
    θ : Real
    hi : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (toIocMod Real.two_pi_ …
    ⊢ Eq (HMul.hMul (↑r) (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) C …
  -/
  convert arg_mul_cos_add_sin_mul_I hr hi using 3
  /-
    case h.e'_2.h.e'_1.h.e'_6
    r : Real
    hr : LT.lt 0 r
    θ : Real
    hi : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (toIocMod Real.two_pi_ …
    ⊢ Eq (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) Complex.I)) (HAdd …
  -/
  simp [toIocMod, cos_sub_int_mul_two_pi, sin_sub_int_mul_two_pi]
  /-
    🎉 no goals
  -/


theorem arg_cos_add_sin_mul_I_eq_toIocMod (θ : ℝ) :
    arg (cos θ + sin θ * I) = toIocMod Real.two_pi_pos (-π) θ := by
  /-
    θ : Real
    ⊢ Eq (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) Complex.I)).arg ( …
  -/
  rw [← one_mul (_ + _), ← ofReal_one, arg_mul_cos_add_sin_mul_I_eq_toIocMod zero_lt_one]
  /-
    🎉 no goals
  -/


theorem arg_mul_cos_add_sin_mul_I_sub {r : ℝ} (hr : 0 < r) (θ : ℝ) :
    arg (r * (cos θ + sin θ * I)) - θ = 2 * π * ⌊(π - θ) / (2 * π)⌋ := by
  rw [arg_mul_cos_add_sin_mul_I_eq_toIocMod hr, toIocMod_sub_self, toIocDiv_eq_neg_floor,
    zsmul_eq_mul]
  /-
    r : Real
    hr : LT.lt 0 r
    θ : Real
    ⊢ Eq (HMul.hMul (↑(Neg.neg (Neg.neg (Int.floor (HDiv.hDiv (HSub.hSub (HAdd.hAd …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem arg_cos_add_sin_mul_I_sub (θ : ℝ) :
    arg (cos θ + sin θ * I) - θ = 2 * π * ⌊(π - θ) / (2 * π)⌋ := by
  /-
    θ : Real
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Complex.cos ↑θ) (HMul.hMul (Complex.sin ↑θ) Comple …
  -/
  rw [← one_mul (_ + _), ← ofReal_one, arg_mul_cos_add_sin_mul_I_sub zero_lt_one]
  /-
    🎉 no goals
  -/


theorem arg_mul_cos_add_sin_mul_I_coe_angle {r : ℝ} (hr : 0 < r) (θ : Real.Angle) :
    (arg (r * (Real.Angle.cos θ + Real.Angle.sin θ * I)) : Real.Angle) = θ := by
  /-
    r : Real
    hr : LT.lt 0 r
    θ : Real.Angle
    ⊢ Eq (↑(HMul.hMul (↑r) (HAdd.hAdd (↑θ.cos) (HMul.hMul (↑θ.sin) Complex.I))).ar …
  -/
  induction' θ using Real.Angle.induction_on with θ
  /-
    case h
    r : Real
    hr : LT.lt 0 r
    θ : Real
    ⊢ Eq ↑(HMul.hMul (↑r) (HAdd.hAdd (↑(↑θ).cos) (HMul.hMul (↑(↑θ).sin) Complex.I) …
  -/
  rw [Real.Angle.cos_coe, Real.Angle.sin_coe, Real.Angle.angle_eq_iff_two_pi_dvd_sub]
  /-
    case h
    r : Real
    hr : LT.lt 0 r
    θ : Real
    ⊢ Exists fun k => Eq (HSub.hSub (HMul.hMul (↑r) (HAdd.hAdd (↑(Real.cos θ)) (HM …
  -/
  use ⌊(π - θ) / (2 * π)⌋
  /-
    case h
    r : Real
    hr : LT.lt 0 r
    θ : Real
    ⊢ Eq (HSub.hSub (HMul.hMul (↑r) (HAdd.hAdd (↑(Real.cos θ)) (HMul.hMul (↑(Real. …
  -/
  exact mod_cast arg_mul_cos_add_sin_mul_I_sub hr θ
  /-
    🎉 no goals
  -/


theorem arg_cos_add_sin_mul_I_coe_angle (θ : Real.Angle) :
    (arg (Real.Angle.cos θ + Real.Angle.sin θ * I) : Real.Angle) = θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (↑(HAdd.hAdd (↑θ.cos) (HMul.hMul (↑θ.sin) Complex.I)).arg) θ
  -/
  rw [← one_mul (_ + _), ← ofReal_one, arg_mul_cos_add_sin_mul_I_coe_angle zero_lt_one]
  /-
    🎉 no goals
  -/


theorem arg_mul_coe_angle {x y : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) :
    (arg (x * y) : Real.Angle) = arg x + arg y := by
  convert arg_mul_cos_add_sin_mul_I_coe_angle (mul_pos (abs.pos hx) (abs.pos hy))
      (arg x + arg y : Real.Angle) using
    3
  simp_rw [← Real.Angle.coe_add, Real.Angle.sin_coe, Real.Angle.cos_coe, ofReal_cos, ofReal_sin,
    cos_add_sin_I, ofReal_add, add_mul, exp_add, ofReal_mul]
  rw [mul_assoc, mul_comm (exp _), ← mul_assoc (abs y : ℂ), abs_mul_exp_arg_mul_I, mul_comm y, ←
    mul_assoc, abs_mul_exp_arg_mul_I]


theorem arg_div_coe_angle {x y : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) :
    (arg (x / y) : Real.Angle) = arg x - arg y := by
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (↑(HDiv.hDiv x y).arg) (HSub.hSub ↑x.arg ↑y.arg)
  -/
  rw [div_eq_mul_inv, arg_mul_coe_angle hx (inv_ne_zero hy), arg_inv_coe_angle, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem arg_coe_angle_toReal_eq_arg (z : ℂ) : (arg z : Real.Angle).toReal = arg z := by
  /-
    z : Complex
    ⊢ Eq (↑z.arg).toReal z.arg
  -/
  rw [Real.Angle.toReal_coe_eq_self_iff_mem_Ioc]
  /-
    z : Complex
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) z.arg
  -/
  exact arg_mem_Ioc _
  /-
    🎉 no goals
  -/


theorem arg_coe_angle_eq_iff_eq_toReal {z : ℂ} {θ : Real.Angle} :
    (arg z : Real.Angle) = θ ↔ arg z = θ.toReal := by
  /-
    z : Complex
    θ : Real.Angle
    ⊢ Iff (Eq (↑z.arg) θ) (Eq z.arg θ.toReal)
  -/
  rw [← Real.Angle.toReal_inj, arg_coe_angle_toReal_eq_arg]
  /-
    🎉 no goals
  -/


@[simp]
theorem arg_coe_angle_eq_iff {x y : ℂ} : (arg x : Real.Angle) = arg y ↔ arg x = arg y := by
  /-
    x y : Complex
    ⊢ Iff (Eq ↑x.arg ↑y.arg) (Eq x.arg y.arg)
  -/
  simp_rw [← Real.Angle.toReal_inj, arg_coe_angle_toReal_eq_arg]
  /-
    🎉 no goals
  -/


lemma arg_mul_eq_add_arg_iff {x y : ℂ} (hx₀ : x ≠ 0) (hy₀ : y ≠ 0) :
    (x * y).arg = x.arg + y.arg ↔ arg x + arg y ∈ Set.Ioc (-π) π := by
  rw [← arg_coe_angle_toReal_eq_arg, arg_mul_coe_angle hx₀ hy₀, ← Real.Angle.coe_add,
      Real.Angle.toReal_coe_eq_self_iff_mem_Ioc]


alias ⟨_, arg_mul⟩ := arg_mul_eq_add_arg_iff


open ComplexOrder in
/-- An alternative description of the slit plane as consisting of nonzero complex numbers
whose argument is not π. -/
lemma mem_slitPlane_iff_arg {z : ℂ} : z ∈ slitPlane ↔ z.arg ≠ π ∧ z ≠ 0 := by
  /-
    z : Complex
    ⊢ Iff (Membership.mem Complex.slitPlane z) (And (Ne z.arg Real.pi) (Ne z 0))
  -/
  simp only [mem_slitPlane_iff_not_le_zero, le_iff_lt_or_eq, ne_eq, arg_eq_pi_iff_lt_zero, not_or]
  /-
    🎉 no goals
  -/


lemma slitPlane_arg_ne_pi {z : ℂ} (hz : z ∈ slitPlane) : z.arg ≠ Real.pi :=
  (mem_slitPlane_iff_arg.mp hz).1


theorem arg_eq_nhds_of_re_pos (hx : 0 < x.re) : arg =ᶠ[𝓝 x] fun x => Real.arcsin (x.im / abs x) :=
  ((continuous_re.tendsto _).eventually (lt_mem_nhds hx)).mono fun _ hy => arg_of_re_nonneg hy.le


theorem arg_eq_nhds_of_re_neg_of_im_pos (hx_re : x.re < 0) (hx_im : 0 < x.im) :
    arg =ᶠ[𝓝 x] fun x => Real.arcsin ((-x).im / abs x) + π := by
  suffices h_forall_nhds : ∀ᶠ y : ℂ in 𝓝 x, y.re < 0 ∧ 0 < y.im from
    h_forall_nhds.mono fun y hy => arg_of_re_neg_of_im_nonneg hy.1 hy.2.le
  /-
    x : Complex
    hx_re : LT.lt x.re 0
    hx_im : LT.lt 0 x.im
    ⊢ Filter.Eventually (fun y => And (LT.lt y.re 0) (LT.lt 0 y.im)) (nhds x)
  -/
  refine IsOpen.eventually_mem ?_ (⟨hx_re, hx_im⟩ : x.re < 0 ∧ 0 < x.im)
  exact
    IsOpen.and (isOpen_lt continuous_re continuous_zero) (isOpen_lt continuous_zero continuous_im)


theorem arg_eq_nhds_of_re_neg_of_im_neg (hx_re : x.re < 0) (hx_im : x.im < 0) :
    arg =ᶠ[𝓝 x] fun x => Real.arcsin ((-x).im / abs x) - π := by
  suffices h_forall_nhds : ∀ᶠ y : ℂ in 𝓝 x, y.re < 0 ∧ y.im < 0 from
    h_forall_nhds.mono fun y hy => arg_of_re_neg_of_im_neg hy.1 hy.2
  /-
    x : Complex
    hx_re : LT.lt x.re 0
    hx_im : LT.lt x.im 0
    ⊢ Filter.Eventually (fun y => And (LT.lt y.re 0) (LT.lt y.im 0)) (nhds x)
  -/
  refine IsOpen.eventually_mem ?_ (⟨hx_re, hx_im⟩ : x.re < 0 ∧ x.im < 0)
  exact
    IsOpen.and (isOpen_lt continuous_re continuous_zero) (isOpen_lt continuous_im continuous_zero)


theorem arg_eq_nhds_of_im_pos (hz : 0 < im z) : arg =ᶠ[𝓝 z] fun x => Real.arccos (x.re / abs x) :=
  ((continuous_im.tendsto _).eventually (lt_mem_nhds hz)).mono fun _ => arg_of_im_pos


theorem arg_eq_nhds_of_im_neg (hz : im z < 0) : arg =ᶠ[𝓝 z] fun x => -Real.arccos (x.re / abs x) :=
  ((continuous_im.tendsto _).eventually (gt_mem_nhds hz)).mono fun _ => arg_of_im_neg


theorem continuousAt_arg (h : x ∈ slitPlane) : ContinuousAt arg x := by
  have h₀ : abs x ≠ 0 := by
    rw [abs.ne_zero_iff]
    exact slitPlane_ne_zero h
  /-
    x : Complex
    h : Membership.mem Complex.slitPlane x
    h₀ : Ne (Complex.abs x) 0
    ⊢ ContinuousAt Complex.arg x
  -/
  rw [mem_slitPlane_iff, ← lt_or_lt_iff_ne] at h
  /-
    x : Complex
    h : Or (LT.lt 0 x.re) (Or (LT.lt x.im 0) (LT.lt 0 x.im))
    h₀ : Ne (Complex.abs x) 0
    ⊢ ContinuousAt Complex.arg x
  -/
  rcases h with (hx_re | hx_im | hx_im)
  exacts [(Real.continuousAt_arcsin.comp
          (continuous_im.continuousAt.div continuous_abs.continuousAt h₀)).congr
      (arg_eq_nhds_of_re_pos hx_re).symm,
    (Real.continuous_arccos.continuousAt.comp
            (continuous_re.continuousAt.div continuous_abs.continuousAt h₀)).neg.congr
      (arg_eq_nhds_of_im_neg hx_im).symm,
    (Real.continuous_arccos.continuousAt.comp
          (continuous_re.continuousAt.div continuous_abs.continuousAt h₀)).congr
      (arg_eq_nhds_of_im_pos hx_im).symm]


theorem tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0)
    (him : z.im = 0) : Tendsto arg (𝓝[{ z : ℂ | z.im < 0 }] z) (𝓝 (-π)) := by
  suffices H : Tendsto (fun x : ℂ => Real.arcsin ((-x).im / abs x) - π)
      (𝓝[{ z : ℂ | z.im < 0 }] z) (𝓝 (-π)) by
    refine H.congr' ?_
    have : ∀ᶠ x : ℂ in 𝓝 z, x.re < 0 := continuous_re.tendsto z (gt_mem_nhds hre)
    filter_upwards [self_mem_nhdsWithin, mem_nhdsWithin_of_mem_nhds this] with _ him hre
    rw [arg, if_neg hre.not_le, if_neg him.not_le]
  convert (Real.continuousAt_arcsin.comp_continuousWithinAt
    ((continuous_im.continuousAt.comp_continuousWithinAt continuousWithinAt_neg).div
      continuous_abs.continuousWithinAt _)
    ).sub_const π using 1
    /-
      case h.e'_5
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      ⊢ Eq (nhds (Neg.neg Real.pi)) (nhds (HSub.hSub (Function.comp Real.arcsin (HDi …
    -/
  · simp [him]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      ⊢ Ne (Complex.abs z) 0
    -/
  · lift z to ℝ using him
    /-
      case convert_3.intro
      z : Real
      hre : LT.lt (↑z).re 0
      ⊢ Ne (Complex.abs ↑z) 0
    -/
    simpa using hre.ne
    /-
      🎉 no goals
    -/


theorem continuousWithinAt_arg_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0) (him : z.im = 0) :
    ContinuousWithinAt arg { z : ℂ | 0 ≤ z.im } z := by
  have : arg =ᶠ[𝓝[{ z : ℂ | 0 ≤ z.im }] z] fun x => Real.arcsin ((-x).im / abs x) + π := by
    have : ∀ᶠ x : ℂ in 𝓝 z, x.re < 0 := continuous_re.tendsto z (gt_mem_nhds hre)
    filter_upwards [self_mem_nhdsWithin (s := { z : ℂ | 0 ≤ z.im }),
      mem_nhdsWithin_of_mem_nhds this] with _ him hre
    rw [arg, if_neg hre.not_le, if_pos him]
  /-
    z : Complex
    hre : LT.lt z.re 0
    him : Eq z.im 0
    this : (nhdsWithin z (setOf fun z => LE.le 0 z.im)).EventuallyEq Complex.arg f …
    ⊢ ContinuousWithinAt Complex.arg (setOf fun z => LE.le 0 z.im) z
  -/
  refine ContinuousWithinAt.congr_of_eventuallyEq ?_ this ?_
  · refine
      (Real.continuousAt_arcsin.comp_continuousWithinAt
            ((continuous_im.continuousAt.comp_continuousWithinAt continuousWithinAt_neg).div
              continuous_abs.continuousWithinAt ?_)).add
        tendsto_const_nhds
    /-
      case refine_1
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      this : (nhdsWithin z (setOf fun z => LE.le 0 z.im)).EventuallyEq Complex.arg f …
      ⊢ Ne (Complex.abs z) 0
    -/
    lift z to ℝ using him
    /-
      case refine_1.intro
      z : Real
      hre : LT.lt (↑z).re 0
      this : (nhdsWithin (↑z) (setOf fun z => LE.le 0 z.im)).EventuallyEq Complex.ar …
      ⊢ Ne (Complex.abs ↑z) 0
    -/
    simpa using hre.ne
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      this : (nhdsWithin z (setOf fun z => LE.le 0 z.im)).EventuallyEq Complex.arg f …
      ⊢ Eq z.arg (HAdd.hAdd (Real.arcsin (HDiv.hDiv (Neg.neg z).im (Complex.abs z))) …
    -/
  · rw [arg, if_neg hre.not_le, if_pos him.ge]
    /-
      🎉 no goals
    -/


theorem tendsto_arg_nhdsWithin_im_nonneg_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0)
    (him : z.im = 0) : Tendsto arg (𝓝[{ z : ℂ | 0 ≤ z.im }] z) (𝓝 π) := by
  simpa only [arg_eq_pi_iff.2 ⟨hre, him⟩] using
    (continuousWithinAt_arg_of_re_neg_of_im_zero hre him).tendsto


theorem continuousAt_arg_coe_angle (h : x ≠ 0) : ContinuousAt ((↑) ∘ arg : ℂ → Real.Angle) x := by
  /-
    x : Complex
    h : Ne x 0
    ⊢ ContinuousAt (Function.comp Real.Angle.coe Complex.arg) x
  -/
  by_cases hs : x ∈ slitPlane
    /-
      case pos
      x : Complex
      h : Ne x 0
      hs : Membership.mem Complex.slitPlane x
      ⊢ ContinuousAt (Function.comp Real.Angle.coe Complex.arg) x
    -/
  · exact Real.Angle.continuous_coe.continuousAt.comp (continuousAt_arg hs)
    /-
      🎉 no goals
    -/
  · rw [← Function.comp_id (((↑) : ℝ → Real.Angle) ∘ arg),
      (funext_iff.2 fun _ => (neg_neg _).symm : (id : ℂ → ℂ) = Neg.neg ∘ Neg.neg), ←
      Function.comp_assoc]
    /-
      case neg
      x : Complex
      h : Ne x 0
      hs : Not (Membership.mem Complex.slitPlane x)
      ⊢ ContinuousAt (Function.comp (Function.comp (Function.comp Real.Angle.coe Com …
    -/
    refine ContinuousAt.comp ?_ continuous_neg.continuousAt
    suffices ContinuousAt (Function.update (((↑) ∘ arg) ∘ Neg.neg : ℂ → Real.Angle) 0 π) (-x) by
      rwa [continuousAt_update_of_ne (neg_ne_zero.2 h)] at this
    have ha :
      Function.update (((↑) ∘ arg) ∘ Neg.neg : ℂ → Real.Angle) 0 π = fun z =>
        (arg z : Real.Angle) + π := by
      rw [Function.update_eq_iff]
      exact ⟨by simp, fun z hz => arg_neg_coe_angle hz⟩
    /-
      case neg
      x : Complex
      h : Ne x 0
      hs : Not (Membership.mem Complex.slitPlane x)
      ha : Eq (Function.update (Function.comp (Function.comp Real.Angle.coe Complex. …
      ⊢ ContinuousAt (Function.update (Function.comp (Function.comp Real.Angle.coe C …
    -/
    rw [ha]
    /-
      case neg
      x : Complex
      h : Ne x 0
      hs : Not (Membership.mem Complex.slitPlane x)
      ha : Eq (Function.update (Function.comp (Function.comp Real.Angle.coe Complex. …
      ⊢ ContinuousAt (fun z => HAdd.hAdd ↑z.arg ↑Real.pi) (Neg.neg x)
    -/
    replace hs := mem_slitPlane_iff.mpr.mt hs
    /-
      case neg
      x : Complex
      h : Ne x 0
      ha : Eq (Function.update (Function.comp (Function.comp Real.Angle.coe Complex. …
      hs : Not (Or (LT.lt 0 x.re) (Ne x.im 0))
      ⊢ ContinuousAt (fun z => HAdd.hAdd ↑z.arg ↑Real.pi) (Neg.neg x)
    -/
    push_neg at hs
    refine
      (Real.Angle.continuous_coe.continuousAt.comp (continuousAt_arg (Or.inl ?_))).add
        continuousAt_const
    /-
      case neg
      x : Complex
      h : Ne x 0
      ha : Eq (Function.update (Function.comp (Function.comp Real.Angle.coe Complex. …
      hs : And (LE.le x.re 0) (Eq x.im 0)
      ⊢ LT.lt 0 (Neg.neg x).re
    -/
    rw [neg_re, neg_pos]
    /-
      case neg
      x : Complex
      h : Ne x 0
      ha : Eq (Function.update (Function.comp (Function.comp Real.Angle.coe Complex. …
      hs : And (LE.le x.re 0) (Eq x.im 0)
      ⊢ LT.lt x.re 0
    -/
    exact hs.1.lt_of_ne fun h0 => h (Complex.ext_iff.2 ⟨h0, hs.2⟩)
    /-
      🎉 no goals
    -/


