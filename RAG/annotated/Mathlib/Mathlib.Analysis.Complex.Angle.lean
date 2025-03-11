/-- The angle between two non-zero complex numbers is the absolute value of the argument of their
quotient.

Note that this does not hold when `x` or `y` is `0` as the LHS is `π / 2` while the RHS is `0`. -/
lemma angle_eq_abs_arg (hx : x ≠ 0) (hy : y ≠ 0) : angle x y = |(x / y).arg| := by
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (InnerProductGeometry.angle x y) (_root_.abs (HDiv.hDiv x y).arg)
  -/
  refine Real.arccos_eq_of_eq_cos (abs_nonneg _) (abs_arg_le_pi _) ?_
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) (Re …
  -/
  rw [Real.cos_abs, Complex.cos_arg (div_ne_zero hx hy)]
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) (HD …
  -/
  have := (map_ne_zero Complex.abs).2 hx
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    this : Ne (Complex.abs x) 0
    ⊢ Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) (HD …
  -/
  have := (map_ne_zero Complex.abs).2 hy
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    this✝ : Ne (Complex.abs x) 0
    this : Ne (Complex.abs y) 0
    ⊢ Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) (HD …
  -/
  simp [div_eq_mul_inv, Complex.normSq_eq_norm_sq]
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    this✝ : Ne (Complex.abs x) 0
    this : Ne (Complex.abs y) 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x.re y.re) (HMul.hMul x.im y.im)) (HMul. …
  -/
  field_simp
  /-
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    this✝ : Ne (Complex.abs x) 0
    this : Ne (Complex.abs y) 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x.re y.re) (HMul.hMul x.im y.im)) (HMul. …
  -/
  ring
  /-
    🎉 no goals
  -/


                                                              /-
                                                                y : Complex
                                                                hy : Ne y 0
                                                                ⊢ Eq (InnerProductGeometry.angle 1 y) (_root_.abs y.arg)
                                                              -/
lemma angle_one_left (hy : y ≠ 0) : angle 1 y = |y.arg| := by simp [angle_eq_abs_arg, hy]
                                                              /-
                                                                🎉 no goals
                                                              -/

                                                               /-
                                                                 x : Complex
                                                                 hx : Ne x 0
                                                                 ⊢ Eq (InnerProductGeometry.angle x 1) (_root_.abs x.arg)
                                                               -/
lemma angle_one_right (hx : x ≠ 0) : angle x 1 = |x.arg| := by simp [angle_eq_abs_arg, hx]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp] lemma angle_mul_left (ha : a ≠ 0) (x y : ℂ) : angle (a * x) (a * y) = angle x y := by
  /-
    a : Complex
    ha : Ne a 0
    x y : Complex
    ⊢ Eq (InnerProductGeometry.angle (HMul.hMul a x) (HMul.hMul a y)) (InnerProduc …
  -/
  obtain rfl | hx := eq_or_ne x 0 <;> obtain rfl | hy := eq_or_ne y 0 <;>
    /-
      case inl.inl
      a : Complex
      ha : Ne a 0
      ⊢ Eq (InnerProductGeometry.angle (HMul.hMul a 0) (HMul.hMul a 0)) (InnerProduc …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [angle_eq_abs_arg, mul_div_mul_left, *]
    /-
      🎉 no goals
    -/


@[simp] lemma angle_mul_right (ha : a ≠ 0) (x y : ℂ) : angle (x * a) (y * a) = angle x y := by
  /-
    a : Complex
    ha : Ne a 0
    x y : Complex
    ⊢ Eq (InnerProductGeometry.angle (HMul.hMul x a) (HMul.hMul y a)) (InnerProduc …
  -/
  simp [mul_comm, angle_mul_left ha]
  /-
    🎉 no goals
  -/


lemma angle_div_left_eq_angle_mul_right (a x y : ℂ) : angle (x / a) y = angle x (y * a) := by
  /-
    a x y : Complex
    ⊢ Eq (InnerProductGeometry.angle (HDiv.hDiv x a) y) (InnerProductGeometry.angl …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      x y : Complex
      ⊢ Eq (InnerProductGeometry.angle (HDiv.hDiv x 0) y) (InnerProductGeometry.angl …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      a x y : Complex
      ha : Ne a 0
      ⊢ Eq (InnerProductGeometry.angle (HDiv.hDiv x a) y) (InnerProductGeometry.angl …
    -/
  · rw [← angle_mul_right ha, div_mul_cancel₀ _ ha]
    /-
      🎉 no goals
    -/


lemma angle_div_right_eq_angle_mul_left (a x y : ℂ) : angle x (y / a) = angle (x * a) y := by
  /-
    a x y : Complex
    ⊢ Eq (InnerProductGeometry.angle x (HDiv.hDiv y a)) (InnerProductGeometry.angl …
  -/
  rw [angle_comm, angle_div_left_eq_angle_mul_right, angle_comm]
  /-
    🎉 no goals
  -/


lemma angle_exp_exp (x y : ℝ) :
    angle (exp (x * I)) (exp (y * I)) = |toIocMod Real.two_pi_pos (-π) (x - y)| := by
  simp_rw [angle_eq_abs_arg (exp_ne_zero _) (exp_ne_zero _), ← exp_sub, ← sub_mul, ← ofReal_sub,
    arg_exp_mul_I]


lemma angle_exp_one (x : ℝ) : angle (exp (x * I)) 1 = |toIocMod Real.two_pi_pos (-π) x| := by
  /-
    x : Real
    ⊢ Eq (InnerProductGeometry.angle (Complex.exp (HMul.hMul (↑x) Complex.I)) 1) ( …
  -/
  simpa using angle_exp_exp x 0
  /-
    🎉 no goals
  -/


/-- Chord-length is a multiple of arc-length up to constants. -/
lemma norm_sub_mem_Icc_angle (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ‖x - y‖ ∈ Icc (2 / π * angle x y) (angle x y) := by
  /-
    x y : Complex
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm y) 1
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
  -/
  wlog h : y = 1
    /-
      case inr
      x y : Complex
      hx : Eq (Norm.norm x) 1
      hy : Eq (Norm.norm y) 1
      this : ∀ {x y : Complex}, Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq y 1 → M …
      h : Not (Eq y 1)
      ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
    -/
  · have := @this (x / y) 1 (by simp only [norm_div, hx, hy, div_one]) norm_one rfl
    rwa [angle_div_left_eq_angle_mul_right, div_sub_one, norm_div, hy, div_one, one_mul]
      at this
    /-
      case inr
      x y : Complex
      hx : Eq (Norm.norm x) 1
      hy : Eq (Norm.norm y) 1
      this✝ : ∀ {x y : Complex}, Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq y 1 →  …
      h : Not (Eq y 1)
      this : Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductG …
      ⊢ Ne y 0
    -/
    rintro rfl
    /-
      case inr
      x : Complex
      hx : Eq (Norm.norm x) 1
      this✝ : ∀ {x y : Complex}, Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq y 1 →  …
      hy : Eq (Norm.norm 0) 1
      h : Not (Eq 0 1)
      this : Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductG …
      ⊢ False
    -/
    simp at hy
    /-
      🎉 no goals
    -/
  /-
    x✝ y✝ x y : Complex
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm y) 1
    h : Eq y 1
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
  -/
  subst y
  /-
    x✝ y x : Complex
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm 1) 1
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
  -/
  rw [norm_eq_abs, abs_eq_one_iff'] at hx
  /-
    x✝ y x : Complex
    hx : Exists fun θ => And (Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ …
    hy : Eq (Norm.norm 1) 1
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
  -/
  obtain ⟨θ, hθ, rfl⟩ := hx
  /-
    case intro.intro
    x y : Complex
    hy : Eq (Norm.norm 1) 1
    θ : Real
    hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (InnerProductGeomet …
  -/
  rw [angle_exp_one, exp_mul_I, add_sub_right_comm, (toIocMod_eq_self _).2]
    /-
      case intro.intro
      x y : Complex
      hy : Eq (Norm.norm 1) 1
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (_root_.abs θ)) (_r …
    -/
  · norm_cast
    /-
      case intro.intro
      x y : Complex
      hy : Eq (Norm.norm 1) 1
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (_root_.abs θ)) (_r …
    -/
    rw [norm_eq_abs, abs_add_mul_I]
    /-
      case intro.intro
      x y : Complex
      hy : Eq (Norm.norm 1) 1
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Membership.mem (Set.Icc (HMul.hMul (HDiv.hDiv 2 Real.pi) (_root_.abs θ)) (_r …
    -/
    refine ⟨Real.le_sqrt_of_sq_le ?_, ?_⟩
      /-
        case intro.intro.refine_1
        x y : Complex
        hy : Eq (Norm.norm 1) 1
        θ : Real
        hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
        ⊢ LE.le (HPow.hPow (HMul.hMul (HDiv.hDiv 2 Real.pi) (_root_.abs θ)) 2) (HAdd.h …
      -/
    · rw [mul_pow, ← _root_.abs_pow, abs_sq]
      calc
        _ = 2 * (1 - (1 - 2 / π ^ 2 * θ ^ 2)) := by ring
        _ ≤ 2 * (1 - θ.cos) := by
            gcongr; exact Real.cos_le_one_sub_mul_cos_sq <| abs_le.2 <| Ioc_subset_Icc_self hθ
        _  = _ := by linear_combination -θ.cos_sq_add_sin_sq
      /-
        case intro.intro.refine_2
        x y : Complex
        hy : Eq (Norm.norm 1) 1
        θ : Real
        hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
        ⊢ LE.le (HAdd.hAdd (HPow.hPow (HSub.hSub (Real.cos θ) 1) 2) (HPow.hPow (Real.s …
      -/
    · rw [Real.sqrt_le_left (by positivity), ← _root_.abs_pow, abs_sq]
      calc
        _ = 2 * (1 - θ.cos) := by linear_combination θ.cos_sq_add_sin_sq
        _ ≤ 2 * (1 - (1 - θ ^ 2 / 2)) := by gcongr; exact Real.one_sub_sq_div_two_le_cos
        _ = _ := by ring
    /-
      case intro.intro
      x y : Complex
      hy : Eq (Norm.norm 1) 1
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (HMul …
    -/
  · convert hθ
    /-
      case h.e'_4.h.e'_4
      x y : Complex
      hy : Eq (Norm.norm 1) 1
      θ : Real
      hθ : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) θ
      ⊢ Eq (HAdd.hAdd (Neg.neg Real.pi) (HMul.hMul 2 Real.pi)) Real.pi
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Chord-length is always less than arc-length. -/
lemma norm_sub_le_angle (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) : ‖x - y‖ ≤ angle x y :=
  (norm_sub_mem_Icc_angle hx hy).2


/-- Chord-length is always greater than a multiple of arc-length. -/
lemma mul_angle_le_norm_sub (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) : 2 / π * angle x y ≤ ‖x - y‖ :=
  (norm_sub_mem_Icc_angle hx hy).1


/-- Arc-length is always less than a multiple of chord-length. -/
lemma angle_le_mul_norm_sub (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) : angle x y ≤ π / 2 * ‖x - y‖ := by
  /-
    x y : Complex
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm y) 1
    ⊢ LE.le (InnerProductGeometry.angle x y) (HMul.hMul (HDiv.hDiv Real.pi 2) (Nor …
  -/
  rw [← div_le_iff₀' <| by positivity, div_eq_inv_mul, inv_div]; exact mul_angle_le_norm_sub hx hy
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


