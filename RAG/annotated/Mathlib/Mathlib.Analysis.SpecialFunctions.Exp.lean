theorem exp_bound_sq (x z : ℂ) (hz : ‖z‖ ≤ 1) :
    ‖exp (x + z) - exp x - z • exp x‖ ≤ ‖exp x‖ * ‖z‖ ^ 2 :=
  calc
    ‖exp (x + z) - exp x - z * exp x‖ = ‖exp x * (exp z - 1 - z)‖ := by
      /-
        x z : Complex
        hz : LE.le (Norm.norm z) 1
        ⊢ Eq (Norm.norm (HSub.hSub (HSub.hSub (Complex.exp (HAdd.hAdd x z)) (Complex.e …
      -/
      congr
      /-
        case e_a
        x z : Complex
        hz : LE.le (Norm.norm z) 1
        ⊢ Eq (HSub.hSub (HSub.hSub (Complex.exp (HAdd.hAdd x z)) (Complex.exp x)) (HMu …
      -/
      rw [exp_add]
      /-
        case e_a
        x z : Complex
        hz : LE.le (Norm.norm z) 1
        ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (Complex.exp x) (Complex.exp z)) (Comple …
      -/
      ring
      /-
        🎉 no goals
      -/
    _ = ‖exp x‖ * ‖exp z - 1 - z‖ := norm_mul _ _
    _ ≤ ‖exp x‖ * ‖z‖ ^ 2 :=
      mul_le_mul_of_nonneg_left (abs_exp_sub_one_sub_id_le hz) (norm_nonneg _)


theorem locally_lipschitz_exp {r : ℝ} (hr_nonneg : 0 ≤ r) (hr_le : r ≤ 1) (x y : ℂ)
    (hyx : ‖y - x‖ < r) : ‖exp y - exp x‖ ≤ (1 + r) * ‖exp x‖ * ‖y - x‖ := by
  /-
    r : Real
    hr_nonneg : LE.le 0 r
    hr_le : LE.le r 1
    x y : Complex
    hyx : LT.lt (Norm.norm (HSub.hSub y x)) r
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.exp y) (Complex.exp x))) (HMul.hMul (HM …
  -/
  have hy_eq : y = x + (y - x) := by abel
  have hyx_sq_le : ‖y - x‖ ^ 2 ≤ r * ‖y - x‖ := by
    rw [pow_two]
    exact mul_le_mul hyx.le le_rfl (norm_nonneg _) hr_nonneg
  have h_sq : ∀ z, ‖z‖ ≤ 1 → ‖exp (x + z) - exp x‖ ≤ ‖z‖ * ‖exp x‖ + ‖exp x‖ * ‖z‖ ^ 2 := by
    intro z hz
    have : ‖exp (x + z) - exp x - z • exp x‖ ≤ ‖exp x‖ * ‖z‖ ^ 2 := exp_bound_sq x z hz
    rw [← sub_le_iff_le_add', ← norm_smul z]
    exact (norm_sub_norm_le _ _).trans this
  calc
    ‖exp y - exp x‖ = ‖exp (x + (y - x)) - exp x‖ := by nth_rw 1 [hy_eq]
    _ ≤ ‖y - x‖ * ‖exp x‖ + ‖exp x‖ * ‖y - x‖ ^ 2 := h_sq (y - x) (hyx.le.trans hr_le)
    _ ≤ ‖y - x‖ * ‖exp x‖ + ‖exp x‖ * (r * ‖y - x‖) :=
      (add_le_add_left (mul_le_mul le_rfl hyx_sq_le (sq_nonneg _) (norm_nonneg _)) _)
    _ = (1 + r) * ‖exp x‖ * ‖y - x‖ := by ring

-- Porting note: proof by term mode `locally_lipschitz_exp zero_le_one le_rfl x`
-- doesn't work because `‖y - x‖` and `dist y x` don't unify

@[continuity]
theorem continuous_exp : Continuous exp :=
  continuous_iff_continuousAt.mpr fun x =>
    continuousAt_of_locally_lipschitz zero_lt_one (2 * ‖exp x‖)
      (fun y ↦ by
        /-
          x y : Complex
          ⊢ LT.lt (Dist.dist y x) 1 → LE.le (Dist.dist (Complex.exp y) (Complex.exp x))  …
        -/
        convert locally_lipschitz_exp zero_le_one le_rfl x y using 2
        /-
          case h'.h.e'_4
          x y : Complex
          a✝ : LT.lt (Norm.norm (HSub.hSub y x)) 1
          ⊢ Eq (HMul.hMul (HMul.hMul 2 (Norm.norm (Complex.exp x))) (Dist.dist y x)) (HM …
        -/
        congr
        /-
          case h'.h.e'_4.e_a.e_a
          x y : Complex
          a✝ : LT.lt (Norm.norm (HSub.hSub y x)) 1
          ⊢ Eq 2 (HAdd.hAdd 1 1)
        -/
        ring)
        /-
          🎉 no goals
        -/


theorem continuousOn_exp {s : Set ℂ} : ContinuousOn exp s :=
  continuous_exp.continuousOn


lemma exp_sub_sum_range_isBigO_pow (n : ℕ) :
    (fun x ↦ exp x - ∑ i ∈ Finset.range n, x ^ i / i !) =O[𝓝 0] (· ^ n) := by
  /-
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Complex.exp x) ((Finset.ran …
  -/
  rcases (zero_le n).eq_or_lt with rfl | hn
    /-
      case inl
      ⊢ Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Complex.exp x) ((Finset.ran …
    -/
  · simpa using continuous_exp.continuousAt.norm.isBoundedUnder_le
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : LT.lt 0 n
      ⊢ Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Complex.exp x) ((Finset.ran …
    -/
  · refine .of_bound (n.succ / (n ! * n)) ?_
    /-
      case inr
      n : Nat
      hn : LT.lt 0 n
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (Complex.exp x) ((Fi …
    -/
    rw [NormedAddCommGroup.nhds_zero_basis_norm_lt.eventually_iff]
    /-
      case inr
      n : Nat
      hn : LT.lt 0 n
      ⊢ Exists fun i => And (LT.lt 0 i) (∀ ⦃x : Complex⦄, Membership.mem (setOf fun  …
    -/
    refine ⟨1, one_pos, fun x hx ↦ ?_⟩
    /-
      case inr
      n : Nat
      hn : LT.lt 0 n
      x : Complex
      hx : Membership.mem (setOf fun y => LT.lt (Norm.norm y) 1) x
      ⊢ LE.le (Norm.norm (HSub.hSub (Complex.exp x) ((Finset.range n).sum fun i => H …
    -/
    convert exp_bound hx.out.le hn using 1
    /-
      case h.e'_4
      n : Nat
      hn : LT.lt 0 n
      x : Complex
      hx : Membership.mem (setOf fun y => LT.lt (Norm.norm y) 1) x
      ⊢ Eq (HMul.hMul (HDiv.hDiv (↑n.succ) (HMul.hMul ↑n.factorial ↑n)) (Norm.norm ( …
    -/
    field_simp [mul_comm]
    /-
      🎉 no goals
    -/


lemma exp_sub_sum_range_succ_isLittleO_pow (n : ℕ) :
    (fun x ↦ exp x - ∑ i ∈ Finset.range (n + 1), x ^ i / i !) =o[𝓝 0] (· ^ n) :=
  (exp_sub_sum_range_isBigO_pow (n + 1)).trans_isLittleO <| isLittleO_pow_pow n.lt_succ_self


theorem Filter.Tendsto.cexp {l : Filter α} {f : α → ℂ} {z : ℂ} (hf : Tendsto f l (𝓝 z)) :
    Tendsto (fun x => exp (f x)) l (𝓝 (exp z)) :=
  (continuous_exp.tendsto _).comp hf


nonrec
theorem ContinuousWithinAt.cexp (h : ContinuousWithinAt f s x) :
    ContinuousWithinAt (fun y => exp (f y)) s x :=
  h.cexp


@[fun_prop]
nonrec
theorem ContinuousAt.cexp (h : ContinuousAt f x) : ContinuousAt (fun y => exp (f y)) x :=
  h.cexp


@[fun_prop]
theorem ContinuousOn.cexp (h : ContinuousOn f s) : ContinuousOn (fun y => exp (f y)) s :=
  fun x hx => (h x hx).cexp


@[fun_prop]
theorem Continuous.cexp (h : Continuous f) : Continuous fun y => exp (f y) :=
  continuous_iff_continuousAt.2 fun _ => h.continuousAt.cexp


/-- The complex exponential function is uniformly continuous on left half planes. -/
lemma UniformlyContinuousOn.cexp (a : ℝ) : UniformContinuousOn exp {x : ℂ | x.re ≤ a} := by
  /-
    a : Real
    ⊢ UniformContinuousOn Complex.exp (setOf fun x => LE.le x.re a)
  -/
  have : Continuous (cexp - 1) := Continuous.sub (Continuous.cexp continuous_id') continuous_one
  /-
    a : Real
    this : Continuous (HSub.hSub Complex.exp 1)
    ⊢ UniformContinuousOn Complex.exp (setOf fun x => LE.le x.re a)
  -/
  rw [Metric.uniformContinuousOn_iff, Metric.continuous_iff'] at *
  /-
    a : Real
    this : ∀ (a : Complex) (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT. …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex),  …
  -/
  intro ε hε
  simp only [gt_iff_lt, Pi.sub_apply, Pi.one_apply, dist_sub_eq_dist_add_right,
    sub_add_cancel] at this
  /-
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex), Membership.mem (setOf fun  …
  -/
  have ha : 0 < ε / (2 * Real.exp a) := by positivity
  /-
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex), Membership.mem (setOf fun  …
  -/
  have H := this 0 (ε / (2 * Real.exp a)) ha
  /-
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    H : Filter.Eventually (fun x => LT.lt (Dist.dist (Complex.exp x) (Complex.exp  …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex), Membership.mem (setOf fun  …
  -/
  rw [Metric.eventually_nhds_iff] at H
  /-
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    H : Exists fun ε_1 => And (GT.gt ε_1 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0 …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex), Membership.mem (setOf fun  …
  -/
  obtain ⟨δ, hδ⟩ := H
  /-
    case intro
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    hδ : And (GT.gt δ 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0) δ → LT.lt (Dist.d …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : Complex), Membership.mem (setOf fun  …
  -/
  refine ⟨δ, hδ.1, ?_⟩
  /-
    case intro
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    hδ : And (GT.gt δ 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0) δ → LT.lt (Dist.d …
    ⊢ ∀ (x : Complex), Membership.mem (setOf fun x => LE.le x.re a) x → ∀ (y : Com …
  -/
  intros x _ y hy hxy
  /-
    case intro
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    hδ : And (GT.gt δ 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0) δ → LT.lt (Dist.d …
    x : Complex
    a✝ : Membership.mem (setOf fun x => LE.le x.re a) x
    y : Complex
    hy : Membership.mem (setOf fun x => LE.le x.re a) y
    hxy : LT.lt (Dist.dist x y) δ
    ⊢ LT.lt (Dist.dist (Complex.exp x) (Complex.exp y)) ε
  -/
  have h3 := hδ.2 (y := x - y) (by simpa only [dist_zero_right, norm_eq_abs] using hxy)
  /-
    case intro
    a ε : Real
    hε : GT.gt ε 0
    this : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT. …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    hδ : And (GT.gt δ 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0) δ → LT.lt (Dist.d …
    x : Complex
    a✝ : Membership.mem (setOf fun x => LE.le x.re a) x
    y : Complex
    hy : Membership.mem (setOf fun x => LE.le x.re a) y
    hxy : LT.lt (Dist.dist x y) δ
    h3 : LT.lt (Dist.dist (Complex.exp (HSub.hSub x y)) (Complex.exp 0)) (HDiv.hDi …
    ⊢ LT.lt (Dist.dist (Complex.exp x) (Complex.exp y)) ε
  -/
  rw [dist_eq_norm, exp_zero] at *
  have : cexp x - cexp y = cexp y * (cexp (x - y) - 1) := by
      rw [mul_sub_one, ← exp_add]
      ring_nf
  /-
    case intro
    a ε : Real
    hε : GT.gt ε 0
    this✝ : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    hδ : And (GT.gt δ 0) (∀ ⦃y : Complex⦄, LT.lt (Dist.dist y 0) δ → LT.lt (Dist.d …
    x : Complex
    a✝ : Membership.mem (setOf fun x => LE.le x.re a) x
    y : Complex
    hy : Membership.mem (setOf fun x => LE.le x.re a) y
    hxy : LT.lt (Norm.norm (HSub.hSub x y)) δ
    h3 : LT.lt (Norm.norm (HSub.hSub (Complex.exp (HSub.hSub x y)) 1)) (HDiv.hDiv  …
    this : Eq (HSub.hSub (Complex.exp x) (Complex.exp y)) (HMul.hMul (Complex.exp  …
    ⊢ LT.lt (Norm.norm (HSub.hSub (Complex.exp x) (Complex.exp y))) ε
  -/
  rw [this, mul_comm]
  have hya : ‖cexp y‖ ≤ Real.exp a := by
    simp only [norm_eq_abs, abs_exp, Real.exp_le_exp]
    exact hy
  simp only [gt_iff_lt, dist_zero_right, norm_eq_abs, Set.mem_setOf_eq, norm_mul,
    Complex.abs_exp] at *
  /-
    case intro
    a ε : Real
    this✝ : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    x : Complex
    a✝ : LE.le x.re a
    y : Complex
    hy : LE.le y.re a
    hxy : LT.lt (Complex.abs (HSub.hSub x y)) δ
    h3 : LT.lt (Complex.abs (HSub.hSub (Complex.exp (HSub.hSub x y)) 1)) (HDiv.hDi …
    this : Eq (HSub.hSub (Complex.exp x) (Complex.exp y)) (HMul.hMul (Complex.exp  …
    hε : LT.lt 0 ε
    hδ : And (LT.lt 0 δ) (∀ ⦃y : Complex⦄, LT.lt (Complex.abs y) δ → LT.lt (Dist.d …
    hya : LE.le (Real.exp y.re) (Real.exp a)
    ⊢ LT.lt (HMul.hMul (Complex.abs (HSub.hSub (Complex.exp (HSub.hSub x y)) 1)) ( …
  -/
  apply lt_of_le_of_lt (mul_le_mul h3.le hya (Real.exp_nonneg y.re) (le_of_lt ha))
  have hrr : ε / (2 * a.exp) * a.exp = ε / 2 := by
    nth_rw 2 [mul_comm]
    field_simp [mul_assoc]
  /-
    case intro
    a ε : Real
    this✝ : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    x : Complex
    a✝ : LE.le x.re a
    y : Complex
    hy : LE.le y.re a
    hxy : LT.lt (Complex.abs (HSub.hSub x y)) δ
    h3 : LT.lt (Complex.abs (HSub.hSub (Complex.exp (HSub.hSub x y)) 1)) (HDiv.hDi …
    this : Eq (HSub.hSub (Complex.exp x) (Complex.exp y)) (HMul.hMul (Complex.exp  …
    hε : LT.lt 0 ε
    hδ : And (LT.lt 0 δ) (∀ ⦃y : Complex⦄, LT.lt (Complex.abs y) δ → LT.lt (Dist.d …
    hya : LE.le (Real.exp y.re) (Real.exp a)
    hrr : Eq (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a))) (Real.exp a)) (HD …
    ⊢ LT.lt (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a))) (Real.exp a)) ε
  -/
  rw [hrr]
  /-
    case intro
    a ε : Real
    this✝ : ∀ (a : Complex) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT …
    ha : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a)))
    δ : Real
    x : Complex
    a✝ : LE.le x.re a
    y : Complex
    hy : LE.le y.re a
    hxy : LT.lt (Complex.abs (HSub.hSub x y)) δ
    h3 : LT.lt (Complex.abs (HSub.hSub (Complex.exp (HSub.hSub x y)) 1)) (HDiv.hDi …
    this : Eq (HSub.hSub (Complex.exp x) (Complex.exp y)) (HMul.hMul (Complex.exp  …
    hε : LT.lt 0 ε
    hδ : And (LT.lt 0 δ) (∀ ⦃y : Complex⦄, LT.lt (Complex.abs y) δ → LT.lt (Dist.d …
    hya : LE.le (Real.exp y.re) (Real.exp a)
    hrr : Eq (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 (Real.exp a))) (Real.exp a)) (HD …
    ⊢ LT.lt (HDiv.hDiv ε 2) ε
  -/
  exact div_two_lt_of_pos hε
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_exp : Continuous exp :=
  Complex.continuous_re.comp Complex.continuous_ofReal.cexp


theorem continuousOn_exp {s : Set ℝ} : ContinuousOn exp s :=
  continuous_exp.continuousOn


lemma exp_sub_sum_range_isBigO_pow (n : ℕ) :
    (fun x ↦ exp x - ∑ i ∈ Finset.range n, x ^ i / i !) =O[𝓝 0] (· ^ n) := by
  have := (Complex.exp_sub_sum_range_isBigO_pow n).comp_tendsto
    (Complex.continuous_ofReal.tendsto' 0 0 rfl)
  /-
    n : Nat
    this : Asymptotics.IsBigO (nhds 0) (Function.comp (fun x => HSub.hSub (Complex …
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Real.exp x) ((Finset.range  …
  -/
  simp only [Function.comp_def] at this
  /-
    n : Nat
    this : Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Complex.exp ↑x) ((Fins …
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => HSub.hSub (Real.exp x) ((Finset.range  …
  -/
  norm_cast at this
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.rexp {l : Filter α} {f : α → ℝ} {z : ℝ} (hf : Tendsto f l (𝓝 z)) :
    Tendsto (fun x => exp (f x)) l (𝓝 (exp z)) :=
  (continuous_exp.tendsto _).comp hf


nonrec
theorem ContinuousWithinAt.rexp (h : ContinuousWithinAt f s x) :
    ContinuousWithinAt (fun y ↦ exp (f y)) s x :=
  h.rexp

@[deprecated (since := "2024-05-09")] alias ContinuousWithinAt.exp := ContinuousWithinAt.rexp


@[fun_prop]
nonrec
theorem ContinuousAt.rexp (h : ContinuousAt f x) : ContinuousAt (fun y ↦ exp (f y)) x :=
  h.rexp

@[deprecated (since := "2024-05-09")] alias ContinuousAt.exp := ContinuousAt.rexp


@[fun_prop]
theorem ContinuousOn.rexp (h : ContinuousOn f s) :
    ContinuousOn (fun y ↦ exp (f y)) s :=
  fun x hx ↦ (h x hx).rexp

@[deprecated (since := "2024-05-09")] alias ContinuousOn.exp := ContinuousOn.rexp


@[fun_prop]
theorem Continuous.rexp (h : Continuous f) : Continuous fun y ↦ exp (f y) :=
  continuous_iff_continuousAt.2 fun _ ↦ h.continuousAt.rexp

@[deprecated (since := "2024-05-09")] alias Continuous.exp := Continuous.rexp


theorem exp_half (x : ℝ) : exp (x / 2) = √(exp x) := by
  /-
    x : Real
    ⊢ Eq (Real.exp (HDiv.hDiv x 2)) (Real.exp x).sqrt
  -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  rw [eq_comm, sqrt_eq_iff_eq_sq, sq, ← exp_add, add_halves] <;> exact (exp_pos _).le
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The real exponential function tends to `+∞` at `+∞`. -/
theorem tendsto_exp_atTop : Tendsto exp atTop atTop := by
  have A : Tendsto (fun x : ℝ => x + 1) atTop atTop :=
    tendsto_atTop_add_const_right atTop 1 tendsto_id
  /-
    A : Filter.Tendsto (fun x => HAdd.hAdd x 1) Filter.atTop Filter.atTop
    ⊢ Filter.Tendsto Real.exp Filter.atTop Filter.atTop
  -/
  have B : ∀ᶠ x in atTop, x + 1 ≤ exp x := eventually_atTop.2 ⟨0, fun x _ => add_one_le_exp x⟩
  /-
    A : Filter.Tendsto (fun x => HAdd.hAdd x 1) Filter.atTop Filter.atTop
    B : Filter.Eventually (fun x => LE.le (HAdd.hAdd x 1) (Real.exp x)) Filter.atTop
    ⊢ Filter.Tendsto Real.exp Filter.atTop Filter.atTop
  -/
  exact tendsto_atTop_mono' atTop B A
  /-
    🎉 no goals
  -/


/-- The real exponential function tends to `0` at `-∞` or, equivalently, `exp(-x)` tends to `0`
at `+∞` -/
theorem tendsto_exp_neg_atTop_nhds_zero : Tendsto (fun x => exp (-x)) atTop (𝓝 0) :=
  (tendsto_inv_atTop_zero.comp tendsto_exp_atTop).congr fun x => (exp_neg x).symm


/-- The real exponential function tends to `1` at `0`. -/
theorem tendsto_exp_nhds_zero_nhds_one : Tendsto exp (𝓝 0) (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto Real.exp (nhds 0) (nhds 1)
  -/
  convert continuous_exp.tendsto 0
  /-
    case h.e'_5.h.e'_3
    ⊢ Eq 1 (Real.exp 0)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem tendsto_exp_atBot : Tendsto exp atBot (𝓝 0) :=
  (tendsto_exp_neg_atTop_nhds_zero.comp tendsto_neg_atBot_atTop).congr fun x =>
    congr_arg exp <| neg_neg x


theorem tendsto_exp_atBot_nhdsGT : Tendsto exp atBot (𝓝[>] 0) :=
  tendsto_inf.2 ⟨tendsto_exp_atBot, tendsto_principal.2 <| Eventually.of_forall exp_pos⟩


@[deprecated (since := "2024-12-22")]
alias tendsto_exp_atBot_nhdsWithin := tendsto_exp_atBot_nhdsGT


@[simp]
theorem isBoundedUnder_ge_exp_comp (l : Filter α) (f : α → ℝ) :
    IsBoundedUnder (· ≥ ·) l fun x => exp (f x) :=
  isBoundedUnder_of ⟨0, fun _ => (exp_pos _).le⟩


@[simp]
theorem isBoundedUnder_le_exp_comp {f : α → ℝ} :
    (IsBoundedUnder (· ≤ ·) l fun x => exp (f x)) ↔ IsBoundedUnder (· ≤ ·) l f :=
  exp_monotone.isBoundedUnder_le_comp_iff tendsto_exp_atTop


/-- The function `exp(x)/x^n` tends to `+∞` at `+∞`, for any natural number `n` -/
theorem tendsto_exp_div_pow_atTop (n : ℕ) : Tendsto (fun x => exp x / x ^ n) atTop atTop := by
  /-
    n : Nat
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Real.exp x) (HPow.hPow x n)) Filter.atTo …
  -/
  refine (atTop_basis_Ioi.tendsto_iff (atTop_basis' 1)).2 fun C hC₁ => ?_
  /-
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    ⊢ Exists fun ia => And True (∀ (x : Real), Membership.mem (Set.Ioi ia) x → Mem …
  -/
  have hC₀ : 0 < C := zero_lt_one.trans_le hC₁
  /-
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    ⊢ Exists fun ia => And True (∀ (x : Real), Membership.mem (Set.Ioi ia) x → Mem …
  -/
  have : 0 < (exp 1 * C)⁻¹ := inv_pos.2 (mul_pos (exp_pos _) hC₀)
  obtain ⟨N, hN⟩ : ∃ N : ℕ, ∀ k ≥ N, (↑k : ℝ) ^ n / exp 1 ^ k < (exp 1 * C)⁻¹ :=
    eventually_atTop.1
      ((tendsto_pow_const_div_const_pow_of_one_lt n (one_lt_exp_iff.2 zero_lt_one)).eventually
        (gt_mem_nhds this))
  /-
    case intro
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    this : LT.lt 0 (Inv.inv (HMul.hMul (Real.exp 1) C))
    N : Nat
    hN : ∀ (k : Nat), GE.ge k N → LT.lt (HDiv.hDiv (HPow.hPow (↑k) n) (HPow.hPow ( …
    ⊢ Exists fun ia => And True (∀ (x : Real), Membership.mem (Set.Ioi ia) x → Mem …
  -/
  simp only [← exp_nat_mul, mul_one, div_lt_iff₀, exp_pos, ← div_eq_inv_mul] at hN
  /-
    case intro
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    this : LT.lt 0 (Inv.inv (HMul.hMul (Real.exp 1) C))
    N : Nat
    hN : ∀ (k : Nat), GE.ge k N → LT.lt (HPow.hPow (↑k) n) (HDiv.hDiv (Real.exp ↑k …
    ⊢ Exists fun ia => And True (∀ (x : Real), Membership.mem (Set.Ioi ia) x → Mem …
  -/
  refine ⟨N, trivial, fun x hx => ?_⟩
  /-
    case intro
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    this : LT.lt 0 (Inv.inv (HMul.hMul (Real.exp 1) C))
    N : Nat
    hN : ∀ (k : Nat), GE.ge k N → LT.lt (HPow.hPow (↑k) n) (HDiv.hDiv (Real.exp ↑k …
    x : Real
    hx : Membership.mem (Set.Ioi ↑N) x
    ⊢ Membership.mem (Set.Ici C) (HDiv.hDiv (Real.exp x) (HPow.hPow x n))
  -/
  rw [Set.mem_Ioi] at hx
  /-
    case intro
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    this : LT.lt 0 (Inv.inv (HMul.hMul (Real.exp 1) C))
    N : Nat
    hN : ∀ (k : Nat), GE.ge k N → LT.lt (HPow.hPow (↑k) n) (HDiv.hDiv (Real.exp ↑k …
    x : Real
    hx : LT.lt (↑N) x
    ⊢ Membership.mem (Set.Ici C) (HDiv.hDiv (Real.exp x) (HPow.hPow x n))
  -/
  have hx₀ : 0 < x := (Nat.cast_nonneg N).trans_lt hx
  /-
    case intro
    n : Nat
    C : Real
    hC₁ : LE.le 1 C
    hC₀ : LT.lt 0 C
    this : LT.lt 0 (Inv.inv (HMul.hMul (Real.exp 1) C))
    N : Nat
    hN : ∀ (k : Nat), GE.ge k N → LT.lt (HPow.hPow (↑k) n) (HDiv.hDiv (Real.exp ↑k …
    x : Real
    hx : LT.lt (↑N) x
    hx₀ : LT.lt 0 x
    ⊢ Membership.mem (Set.Ici C) (HDiv.hDiv (Real.exp x) (HPow.hPow x n))
  -/
  rw [Set.mem_Ici, le_div_iff₀ (pow_pos hx₀ _), ← le_div_iff₀' hC₀]
  calc
    x ^ n ≤ ⌈x⌉₊ ^ n := by gcongr; exact Nat.le_ceil _
    _ ≤ exp ⌈x⌉₊ / (exp 1 * C) := mod_cast (hN _ (Nat.lt_ceil.2 hx).le).le
    _ ≤ exp (x + 1) / (exp 1 * C) := by gcongr; exact (Nat.ceil_lt_add_one hx₀.le).le
    _ = exp x / C := by rw [add_comm, exp_add, mul_div_mul_left _ _ (exp_pos _).ne']


/-- The function `x^n * exp(-x)` tends to `0` at `+∞`, for any natural number `n`. -/
theorem tendsto_pow_mul_exp_neg_atTop_nhds_zero (n : ℕ) :
    Tendsto (fun x => x ^ n * exp (-x)) atTop (𝓝 0) :=
  (tendsto_inv_atTop_zero.comp (tendsto_exp_div_pow_atTop n)).congr fun x => by
    /-
      n : Nat
      x : Real
      ⊢ Eq (Function.comp (fun r => Inv.inv r) (fun x => HDiv.hDiv (Real.exp x) (HPo …
    -/
    rw [comp_apply, inv_eq_one_div, div_div_eq_mul_div, one_mul, div_eq_mul_inv, exp_neg]
    /-
      🎉 no goals
    -/


/-- The function `(b * exp x + c) / (x ^ n)` tends to `+∞` at `+∞`, for any natural number
`n` and any real numbers `b` and `c` such that `b` is positive. -/
theorem tendsto_mul_exp_add_div_pow_atTop (b c : ℝ) (n : ℕ) (hb : 0 < b) :
    Tendsto (fun x => (b * exp x + c) / x ^ n) atTop atTop := by
  /-
    b c : Real
    n : Nat
    hb : LT.lt 0 b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul b (Real.exp x)) c)  …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      b c : Real
      hb : LT.lt 0 b
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul b (Real.exp x)) c)  …
    -/
  · simp only [pow_zero, div_one]
    /-
      case inl
      b c : Real
      hb : LT.lt 0 b
      ⊢ Filter.Tendsto (fun x => HAdd.hAdd (HMul.hMul b (Real.exp x)) c) Filter.atTo …
    -/
    exact (tendsto_exp_atTop.const_mul_atTop hb).atTop_add tendsto_const_nhds
    /-
      🎉 no goals
    -/
  /-
    case inr
    b c : Real
    n : Nat
    hb : LT.lt 0 b
    hn : Ne n 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul b (Real.exp x)) c)  …
  -/
  simp only [add_div, mul_div_assoc]
  exact
    ((tendsto_exp_div_pow_atTop n).const_mul_atTop hb).atTop_add
      (tendsto_const_nhds.div_atTop (tendsto_pow_atTop hn))


/-- The function `(x ^ n) / (b * exp x + c)` tends to `0` at `+∞`, for any natural number
`n` and any real numbers `b` and `c` such that `b` is nonzero. -/
theorem tendsto_div_pow_mul_exp_add_atTop (b c : ℝ) (n : ℕ) (hb : 0 ≠ b) :
    Tendsto (fun x => x ^ n / (b * exp x + c)) atTop (𝓝 0) := by
  have H : ∀ d e, 0 < d → Tendsto (fun x : ℝ => x ^ n / (d * exp x + e)) atTop (𝓝 0) := by
    intro b' c' h
    convert (tendsto_mul_exp_add_div_pow_atTop b' c' n h).inv_tendsto_atTop using 1
    ext x
    simp
  /-
    b c : Real
    n : Nat
    hb : Ne 0 b
    H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b ( …
  -/
  cases' lt_or_gt_of_ne hb with h h
    /-
      case inl
      b c : Real
      n : Nat
      hb : Ne 0 b
      H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
      h : LT.lt 0 b
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b ( …
    -/
  · exact H b c h
    /-
      🎉 no goals
    -/
    /-
      case inr
      b c : Real
      n : Nat
      hb : Ne 0 b
      H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
      h : GT.gt 0 b
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b ( …
    -/
  · convert (H (-b) (-c) (neg_pos.mpr h)).neg using 1
      /-
        case h.e'_3
        b c : Real
        n : Nat
        hb : Ne 0 b
        H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
        h : GT.gt 0 b
        ⊢ Eq (fun x => HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b (Real.exp x)) …
      -/
    · ext x
      /-
        case h.e'_3.h
        b c : Real
        n : Nat
        hb : Ne 0 b
        H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
        h : GT.gt 0 b
        x : Real
        ⊢ Eq (HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b (Real.exp x)) c)) (Neg …
      -/
      field_simp
      /-
        case h.e'_3.h
        b c : Real
        n : Nat
        hb : Ne 0 b
        H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
        h : GT.gt 0 b
        x : Real
        ⊢ Eq (HDiv.hDiv (HPow.hPow x n) (HAdd.hAdd (HMul.hMul b (Real.exp x)) c)) (HDi …
      -/
      rw [← neg_add (b * exp x) c, neg_div_neg_eq]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5
        b c : Real
        n : Nat
        hb : Ne 0 b
        H : ∀ (d e : Real), LT.lt 0 d → Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow  …
        h : GT.gt 0 b
        ⊢ Eq (nhds 0) (nhds (-0))
      -/
    · rw [neg_zero]
      /-
        🎉 no goals
      -/


/-- `Real.exp` as an order isomorphism between `ℝ` and `(0, +∞)`. -/
def expOrderIso : ℝ ≃o Ioi (0 : ℝ) :=
  StrictMono.orderIsoOfSurjective _ (exp_strictMono.codRestrict exp_pos) <|
    (continuous_exp.subtype_mk _).surjective
          /-
            α : Type u_1
            x y z : Real
            l : Filter α
            ⊢ Filter.Tendsto (fun x => ⟨Real.exp x, ⋯⟩) Filter.atTop Filter.atTop
          -/
      (by rw [tendsto_Ioi_atTop]; simp only [tendsto_exp_atTop])
                                  /-
                                    🎉 no goals
                                  -/
          /-
            α : Type u_1
            x y z : Real
            l : Filter α
            ⊢ Filter.Tendsto (fun x => ⟨Real.exp x, ⋯⟩) Filter.atBot Filter.atBot
          -/
      (by rw [tendsto_Ioi_atBot]; simp only [tendsto_exp_atBot_nhdsGT])
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem coe_expOrderIso_apply (x : ℝ) : (expOrderIso x : ℝ) = exp x :=
  rfl


@[simp]
theorem coe_comp_expOrderIso : (↑) ∘ expOrderIso = exp :=
  rfl


@[simp]
theorem range_exp : range exp = Set.Ioi 0 := by
  /-
    ⊢ Eq (Set.range Real.exp) (Set.Ioi 0)
  -/
  rw [← coe_comp_expOrderIso, range_comp, expOrderIso.range_eq, image_univ, Subtype.range_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_exp_atTop : map exp atTop = atTop := by
  /-
    ⊢ Eq (Filter.map Real.exp Filter.atTop) Filter.atTop
  -/
  rw [← coe_comp_expOrderIso, ← Filter.map_map, OrderIso.map_atTop, map_val_Ioi_atTop]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_exp_atTop : comap exp atTop = atTop := by
  /-
    ⊢ Eq (Filter.comap Real.exp Filter.atTop) Filter.atTop
  -/
  rw [← map_exp_atTop, comap_map exp_injective, map_exp_atTop]
  /-
    🎉 no goals
  -/


@[simp]
theorem tendsto_exp_comp_atTop {f : α → ℝ} :
    Tendsto (fun x => exp (f x)) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Filter.Tendsto (fun x => Real.exp (f x)) l Filter.atTop) (Filter.Tendst …
  -/
  simp_rw [← comp_apply (f := exp), ← tendsto_comap_iff, comap_exp_atTop]
  /-
    🎉 no goals
  -/


theorem tendsto_comp_exp_atTop {f : ℝ → α} :
    Tendsto (fun x => f (exp x)) atTop l ↔ Tendsto f atTop l := by
  /-
    α : Type u_1
    l : Filter α
    f : Real → α
    ⊢ Iff (Filter.Tendsto (fun x => f (Real.exp x)) Filter.atTop l) (Filter.Tendst …
  -/
  simp_rw [← comp_apply (g := exp), ← tendsto_map'_iff, map_exp_atTop]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_exp_atBot : map exp atBot = 𝓝[>] 0 := by
  /-
    ⊢ Eq (Filter.map Real.exp Filter.atBot) (nhdsWithin 0 (Set.Ioi 0))
  -/
  rw [← coe_comp_expOrderIso, ← Filter.map_map, expOrderIso.map_atBot, ← map_coe_Ioi_atBot]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_exp_nhdsGT_zero : comap exp (𝓝[>] 0) = atBot := by
  /-
    ⊢ Eq (Filter.comap Real.exp (nhdsWithin 0 (Set.Ioi 0))) Filter.atBot
  -/
  rw [← map_exp_atBot, comap_map exp_injective]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias comap_exp_nhdsWithin_Ioi_zero := comap_exp_nhdsGT_zero


theorem tendsto_comp_exp_atBot {f : ℝ → α} :
    Tendsto (fun x => f (exp x)) atBot l ↔ Tendsto f (𝓝[>] 0) l := by
  /-
    α : Type u_1
    l : Filter α
    f : Real → α
    ⊢ Iff (Filter.Tendsto (fun x => f (Real.exp x)) Filter.atBot l) (Filter.Tendst …
  -/
  rw [← map_exp_atBot, tendsto_map'_iff]
  /-
    α : Type u_1
    l : Filter α
    f : Real → α
    ⊢ Iff (Filter.Tendsto (fun x => f (Real.exp x)) Filter.atBot l) (Filter.Tendst …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_exp_nhds_zero : comap exp (𝓝 0) = atBot :=
                                                  /-
                                                    ⊢ Eq (Filter.comap Real.exp (nhdsWithin 0 (Set.range Real.exp))) Filter.atBot
                                                  -/
  (comap_nhdsWithin_range exp 0).symm.trans <| by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem tendsto_exp_comp_nhds_zero {f : α → ℝ} :
    Tendsto (fun x => exp (f x)) l (𝓝 0) ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Filter.Tendsto (fun x => Real.exp (f x)) l (nhds 0)) (Filter.Tendsto f  …
  -/
  simp_rw [← comp_apply (f := exp), ← tendsto_comap_iff, comap_exp_nhds_zero]
  /-
    🎉 no goals
  -/


theorem isOpenEmbedding_exp : IsOpenEmbedding exp :=
  isOpen_Ioi.isOpenEmbedding_subtypeVal.comp expOrderIso.toHomeomorph.isOpenEmbedding


@[deprecated (since := "2024-10-18")]
alias openEmbedding_exp := isOpenEmbedding_exp


@[simp]
theorem map_exp_nhds (x : ℝ) : map exp (𝓝 x) = 𝓝 (exp x) :=
  isOpenEmbedding_exp.map_nhds_eq x


@[simp]
theorem comap_exp_nhds_exp (x : ℝ) : comap exp (𝓝 (exp x)) = 𝓝 x :=
  (isOpenEmbedding_exp.nhds_eq_comap x).symm


theorem isLittleO_pow_exp_atTop {n : ℕ} : (fun x : ℝ => x ^ n) =o[atTop] Real.exp := by
  simpa [isLittleO_iff_tendsto fun x hx => ((exp_pos x).ne' hx).elim] using
    tendsto_div_pow_mul_exp_add_atTop 1 0 n zero_ne_one


@[simp]
theorem isBigO_exp_comp_exp_comp {f g : α → ℝ} :
    ((fun x => exp (f x)) =O[l] fun x => exp (g x)) ↔ IsBoundedUnder (· ≤ ·) l (f - g) :=
  Iff.trans (isBigO_iff_isBoundedUnder_le_div <| Eventually.of_forall fun _ => exp_ne_zero _) <| by
    /-
      α : Type u_1
      l : Filter α
      f g : α → Real
      ⊢ Iff (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => HDiv.hDiv ( …
    -/
    simp only [norm_eq_abs, abs_exp, ← exp_sub, isBoundedUnder_le_exp_comp, Pi.sub_def]
    /-
      🎉 no goals
    -/


@[simp]
theorem isTheta_exp_comp_exp_comp {f g : α → ℝ} :
    ((fun x => exp (f x)) =Θ[l] fun x => exp (g x)) ↔
      IsBoundedUnder (· ≤ ·) l fun x => |f x - g x| := by
  simp only [isBoundedUnder_le_abs, ← isBoundedUnder_le_neg, neg_sub, IsTheta,
    isBigO_exp_comp_exp_comp, Pi.sub_def]


@[simp]
theorem isLittleO_exp_comp_exp_comp {f g : α → ℝ} :
    ((fun x => exp (f x)) =o[l] fun x => exp (g x)) ↔ Tendsto (fun x => g x - f x) l atTop := by
  simp only [isLittleO_iff_tendsto, exp_ne_zero, ← exp_sub, ← tendsto_neg_atTop_iff, false_imp_iff,
    imp_true_iff, tendsto_exp_comp_nhds_zero, neg_sub]


theorem isLittleO_one_exp_comp {f : α → ℝ} :
    ((fun _ => 1 : α → ℝ) =o[l] fun x => exp (f x)) ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => 1) fun x => Real.exp (f x)) (Filter.T …
  -/
  simp only [← exp_zero, isLittleO_exp_comp_exp_comp, sub_zero]
  /-
    🎉 no goals
  -/


/-- `Real.exp (f x)` is bounded away from zero along a filter if and only if this filter is bounded
from below under `f`. -/
@[simp]
theorem isBigO_one_exp_comp {f : α → ℝ} :
    ((fun _ => 1 : α → ℝ) =O[l] fun x => exp (f x)) ↔ IsBoundedUnder (· ≥ ·) l f := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Asymptotics.IsBigO l (fun x => 1) fun x => Real.exp (f x)) (Filter.IsBo …
  -/
  simp only [← exp_zero, isBigO_exp_comp_exp_comp, Pi.sub_def, zero_sub, isBoundedUnder_le_neg]
  /-
    🎉 no goals
  -/


/-- `Real.exp (f x)` is bounded away from zero along a filter if and only if this filter is bounded
from below under `f`. -/
theorem isBigO_exp_comp_one {f : α → ℝ} :
    (fun x => exp (f x)) =O[l] (fun _ => 1 : α → ℝ) ↔ IsBoundedUnder (· ≤ ·) l f := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Asymptotics.IsBigO l (fun x => Real.exp (f x)) fun x => 1) (Filter.IsBo …
  -/
  simp only [isBigO_one_iff, norm_eq_abs, abs_exp, isBoundedUnder_le_exp_comp]
  /-
    🎉 no goals
  -/


/-- `Real.exp (f x)` is bounded away from zero and infinity along a filter `l` if and only if
`|f x|` is bounded from above along this filter. -/
@[simp]
theorem isTheta_exp_comp_one {f : α → ℝ} :
    (fun x => exp (f x)) =Θ[l] (fun _ => 1 : α → ℝ) ↔ IsBoundedUnder (· ≤ ·) l fun x => |f x| := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Asymptotics.IsTheta l (fun x => Real.exp (f x)) fun x => 1) (Filter.IsB …
  -/
  simp only [← exp_zero, isTheta_exp_comp_exp_comp, sub_zero]
  /-
    🎉 no goals
  -/


lemma summable_exp_nat_mul_iff {a : ℝ} :
    Summable (fun n : ℕ ↦ exp (n * a)) ↔ a < 0 := by
  simp only [exp_nat_mul, summable_geometric_iff_norm_lt_one, norm_of_nonneg (exp_nonneg _),
    exp_lt_one_iff]


lemma summable_exp_neg_nat : Summable fun n : ℕ ↦ exp (-n) := by
  /-
    ⊢ Summable fun n => Real.exp (Neg.neg ↑n)
  -/
  simpa only [mul_neg_one] using summable_exp_nat_mul_iff.mpr neg_one_lt_zero
  /-
    🎉 no goals
  -/


lemma summable_pow_mul_exp_neg_nat_mul (k : ℕ) {r : ℝ} (hr : 0 < r) :
    Summable fun n : ℕ ↦ n ^ k * exp (-r * n) := by
  /-
    k : Nat
    r : Real
    hr : LT.lt 0 r
    ⊢ Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (Real.exp (HMul.hMul (Neg.neg …
  -/
  simp_rw [mul_comm (-r), exp_nat_mul]
  /-
    k : Nat
    r : Real
    hr : LT.lt 0 r
    ⊢ Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow (Real.exp (Neg.neg …
  -/
  apply summable_pow_mul_geometric_of_norm_lt_one
  /-
    case hr
    k : Nat
    r : Real
    hr : LT.lt 0 r
    ⊢ LT.lt (Norm.norm (Real.exp (Neg.neg r))) 1
  -/
  rwa [norm_of_nonneg (exp_nonneg _), exp_lt_one_iff, neg_lt_zero]
  /-
    🎉 no goals
  -/


open Real in
/-- If `f` has sum `a`, then `exp ∘ f` has product `exp a`. -/
lemma HasSum.rexp {ι} {f : ι → ℝ} {a : ℝ} (h : HasSum f a) : HasProd (rexp ∘ f) (rexp a) :=
  Tendsto.congr (fun s ↦ exp_sum s f) <| Tendsto.rexp h


@[simp]
theorem comap_exp_cobounded : comap exp (cobounded ℂ) = comap re atTop :=
  calc
    comap exp (cobounded ℂ) = comap re (comap Real.exp atTop) := by
      /-
        ⊢ Eq (Filter.comap Complex.exp (Bornology.cobounded Complex)) (Filter.comap Co …
      -/
      simp only [← comap_norm_atTop, Complex.norm_eq_abs, comap_comap, Function.comp_def, abs_exp]
      /-
        🎉 no goals
      -/
                             /-
                               ⊢ Eq (Filter.comap Complex.re (Filter.comap Real.exp Filter.atTop)) (Filter.co …
                             -/
    _ = comap re atTop := by rw [Real.comap_exp_atTop]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem comap_exp_nhds_zero : comap exp (𝓝 0) = comap re atBot :=
  calc
    comap exp (𝓝 0) = comap re (comap Real.exp (𝓝 0)) := by
      /-
        ⊢ Eq (Filter.comap Complex.exp (nhds 0)) (Filter.comap Complex.re (Filter.coma …
      -/
      simp only [comap_comap, ← comap_abs_nhds_zero, Function.comp_def, abs_exp]
      /-
        🎉 no goals
      -/
                             /-
                               ⊢ Eq (Filter.comap Complex.re (Filter.comap Real.exp (nhds 0))) (Filter.comap  …
                             -/
    _ = comap re atBot := by rw [Real.comap_exp_nhds_zero]
                             /-
                               🎉 no goals
                             -/


theorem comap_exp_nhdsNE : comap exp (𝓝[≠] 0) = comap re atBot := by
  /-
    ⊢ Eq (Filter.comap Complex.exp (nhdsWithin 0 (HasCompl.compl (Singleton.single …
  -/
  have : (exp ⁻¹' {0})ᶜ = Set.univ := eq_univ_of_forall exp_ne_zero
  /-
    this : Eq (HasCompl.compl (Set.preimage Complex.exp (Singleton.singleton 0)))  …
    ⊢ Eq (Filter.comap Complex.exp (nhdsWithin 0 (HasCompl.compl (Singleton.single …
  -/
  simp [nhdsWithin, comap_exp_nhds_zero, this]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias comap_exp_nhdsWithin_zero := comap_exp_nhdsNE


theorem tendsto_exp_nhds_zero_iff {α : Type*} {l : Filter α} {f : α → ℂ} :
    Tendsto (fun x => exp (f x)) l (𝓝 0) ↔ Tendsto (fun x => re (f x)) l atBot := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Complex
    ⊢ Iff (Filter.Tendsto (fun x => Complex.exp (f x)) l (nhds 0)) (Filter.Tendsto …
  -/
  simp_rw [← comp_apply (f := exp), ← tendsto_comap_iff, comap_exp_nhds_zero, tendsto_comap_iff]
  /-
    α : Type u_1
    l : Filter α
    f : α → Complex
    ⊢ Iff (Filter.Tendsto (Function.comp Complex.re f) l Filter.atBot) (Filter.Ten …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Complex.abs (Complex.exp z) → ∞` as `Complex.re z → ∞`. -/
theorem tendsto_exp_comap_re_atTop : Tendsto exp (comap re atTop) (cobounded ℂ) :=
  comap_exp_cobounded ▸ tendsto_comap


/-- `Complex.exp z → 0` as `Complex.re z → -∞`. -/
theorem tendsto_exp_comap_re_atBot : Tendsto exp (comap re atBot) (𝓝 0) :=
  comap_exp_nhds_zero ▸ tendsto_comap


theorem tendsto_exp_comap_re_atBot_nhdsNE : Tendsto exp (comap re atBot) (𝓝[≠] 0) :=
  comap_exp_nhdsNE ▸ tendsto_comap


@[deprecated (since := "2024-12-22")]
alias tendsto_exp_comap_re_atBot_nhdsWithin := tendsto_exp_comap_re_atBot_nhdsNE


open Complex in
/-- If `f` has sum `a`, then `exp ∘ f` has product `exp a`. -/
lemma HasSum.cexp {ι : Type*} {f : ι → ℂ} {a : ℂ} (h : HasSum f a) : HasProd (cexp ∘ f) (cexp a) :=
  Filter.Tendsto.congr (fun s ↦ exp_sum s f) <| Filter.Tendsto.cexp h

