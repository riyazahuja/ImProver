/-- The (point-wise) product of `log : ℕ → ℂ` with `f`. -/
noncomputable abbrev LSeries.logMul (f : ℕ → ℂ) (n : ℕ) : ℂ := log n * f n


/-- The derivative of the terms of an L-series. -/
lemma LSeries.hasDerivAt_term (f : ℕ → ℂ) (n : ℕ) (s : ℂ) :
    HasDerivAt (fun z ↦ term f z n) (-(term (logMul f) s n)) s := by
  /-
    f : Nat → Complex
    n : Nat
    s : Complex
    ⊢ HasDerivAt (fun z => LSeries.term f z n) (Neg.neg (LSeries.term (LSeries.log …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      f : Nat → Complex
      s : Complex
      ⊢ HasDerivAt (fun z => LSeries.term f z 0) (Neg.neg (LSeries.term (LSeries.log …
    -/
  · simp only [term_zero, neg_zero, hasDerivAt_const]
    /-
      🎉 no goals
    -/
  simp_rw [term_of_ne_zero hn, ← neg_div, ← neg_mul, mul_comm, mul_div_assoc, div_eq_mul_inv,
    ← cpow_neg]
  exact HasDerivAt.const_mul (f n) (by simpa only [mul_comm, ← mul_neg_one (log n), ← mul_assoc]
    using (hasDerivAt_neg' s).const_cpow (Or.inl <| Nat.cast_ne_zero.mpr hn))

/- This lemma proves two things at once, since their proofs are intertwined; we give separate
non-private lemmas below that extract the two statements. -/

private lemma LSeries.LSeriesSummable_logMul_and_hasDerivAt {f : ℕ → ℂ} {s : ℂ}
    (h : abscissaOfAbsConv f < s.re) :
    LSeriesSummable (logMul f) s ∧ HasDerivAt (LSeries f) (-LSeries (logMul f) s) s := by
  -- The L-series of `f` is summable at some real `x < re s`.
  /-
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  obtain ⟨x, hxs, hf⟩ := LSeriesSummable_lt_re_of_abscissaOfAbsConv_lt_re h
  /-
    case intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  obtain ⟨y, hxy, hys⟩ := exists_between hxs
  -- We work in the right half-plane `y < re z`, for some `y` such that `x < y < re s`, on which
  -- we have a uniform summable bound on `‖term f z ·‖`.
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  let S : Set ℂ := {z | y < z.re}
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  have h₀ : Summable (fun n ↦ ‖term f x n‖) := summable_norm_iff.mpr hf
  have h₁ (n) : DifferentiableOn ℂ (term f · n) S :=
    fun z _ ↦ (hasDerivAt_term f n _).differentiableAt.differentiableWithinAt
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    h₀ : Summable fun n => Norm.norm (LSeries.term f (↑x) n)
    h₁ : ∀ (n : Nat), DifferentiableOn Complex (fun x => LSeries.term f x n) S
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  have h₂ : IsOpen S := isOpen_lt continuous_const continuous_re
  have h₃ (n z) (hz : z ∈ S) : ‖term f z n‖ ≤ ‖term f x n‖ :=
    norm_term_le_of_re_le_re f (by simpa using (hxy.trans hz).le) n
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    h₀ : Summable fun n => Norm.norm (LSeries.term f (↑x) n)
    h₁ : ∀ (n : Nat), DifferentiableOn Complex (fun x => LSeries.term f x n) S
    h₂ : IsOpen S
    h₃ : ∀ (n : Nat) (z : Complex), Membership.mem S z → LE.le (Norm.norm (LSeries …
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  have H := hasSum_deriv_of_summable_norm h₀ h₁ h₂ h₃ hys
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    h₀ : Summable fun n => Norm.norm (LSeries.term f (↑x) n)
    h₁ : ∀ (n : Nat), DifferentiableOn Complex (fun x => LSeries.term f x n) S
    h₂ : IsOpen S
    h₃ : ∀ (n : Nat) (z : Complex), Membership.mem S z → LE.le (Norm.norm (LSeries …
    H : HasSum (fun i => deriv (fun x => LSeries.term f x i) s) (deriv (fun w => t …
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  simp_rw [(hasDerivAt_term f _ _).deriv] at H
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    h₀ : Summable fun n => Norm.norm (LSeries.term f (↑x) n)
    h₁ : ∀ (n : Nat), DifferentiableOn Complex (fun x => LSeries.term f x n) S
    h₂ : IsOpen S
    h₃ : ∀ (n : Nat) (z : Complex), Membership.mem S z → LE.le (Norm.norm (LSeries …
    H : HasSum (fun i => Neg.neg (LSeries.term (LSeries.logMul f) s i)) (deriv (fu …
    ⊢ And (LSeriesSummable (LSeries.logMul f) s) (HasDerivAt (LSeries f) (Neg.neg  …
  -/
  refine ⟨summable_neg_iff.mp H.summable, ?_⟩
  /-
    case intro.intro.intro.intro
    f : Nat → Complex
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    x : Real
    hxs : LT.lt x s.re
    hf : LSeriesSummable f ↑x
    y : Real
    hxy : LT.lt x y
    hys : LT.lt y s.re
    S : Set Complex := setOf fun z => LT.lt y z.re
    h₀ : Summable fun n => Norm.norm (LSeries.term f (↑x) n)
    h₁ : ∀ (n : Nat), DifferentiableOn Complex (fun x => LSeries.term f x n) S
    h₂ : IsOpen S
    h₃ : ∀ (n : Nat) (z : Complex), Membership.mem S z → LE.le (Norm.norm (LSeries …
    H : HasSum (fun i => Neg.neg (LSeries.term (LSeries.logMul f) s i)) (deriv (fu …
    ⊢ HasDerivAt (LSeries f) (Neg.neg (LSeries (LSeries.logMul f) s)) s
  -/
  have H' := differentiableOn_tsum_of_summable_norm h₀ h₁ h₂ h₃
  simpa only [← H.tsum_eq, tsum_neg]
    using (H'.differentiableAt <| IsOpen.mem_nhds h₂ hys).hasDerivAt


/-- If `re s` is greater than the abscissa of absolute convergence of `f`, then the L-series
of `f` is differentiable with derivative the negative of the L-series of the point-wise
product of `log` with `f`. -/
lemma LSeries_hasDerivAt {f : ℕ → ℂ} {s : ℂ} (h : abscissaOfAbsConv f < s.re) :
    HasDerivAt (LSeries f) (- LSeries (logMul f) s) s :=
  (LSeriesSummable_logMul_and_hasDerivAt h).2


/-- If `re s` is greater than the abscissa of absolute convergence of `f`, then
the derivative of this L-series at `s` is the negative of the L-series of `log * f`. -/
lemma LSeries_deriv {f : ℕ → ℂ} {s : ℂ} (h : abscissaOfAbsConv f < s.re) :
    deriv (LSeries f) s = - LSeries (logMul f) s :=
  (LSeries_hasDerivAt h).deriv


/-- The derivative of the L-series of `f` agrees with the negative of the L-series of
`log * f` on the right half-plane of absolute convergence. -/
lemma LSeries_deriv_eqOn {f : ℕ → ℂ} :
    {s | abscissaOfAbsConv f < s.re}.EqOn (deriv (LSeries f)) (- LSeries (logMul f)) :=
  deriv_eqOn (isOpen_re_gt_EReal _) fun _ hs ↦ (LSeries_hasDerivAt hs).hasDerivWithinAt


/-- If the L-series of `f` is summable at `s` and `re s < re s'`, then the L-series of the
point-wise product of `log` with `f` is summable at `s'`. -/
lemma LSeriesSummable_logMul_of_lt_re {f : ℕ → ℂ} {s : ℂ} (h : abscissaOfAbsConv f < s.re) :
    LSeriesSummable (logMul f) s :=
  (LSeriesSummable_logMul_and_hasDerivAt h).1


/-- The abscissa of absolute convergence of the point-wise product of `log` and `f`
is the same as that of `f`. -/
@[simp]
lemma LSeries.abscissaOfAbsConv_logMul {f : ℕ → ℂ} :
    abscissaOfAbsConv (logMul f) = abscissaOfAbsConv f := by
  /-
    f : Nat → Complex
    ⊢ Eq (LSeries.abscissaOfAbsConv (LSeries.logMul f)) (LSeries.abscissaOfAbsConv …
  -/
  apply le_antisymm <;> refine abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable' fun s hs ↦ ?_
    /-
      case a
      f : Nat → Complex
      s : Real
      hs : LT.lt (LSeries.abscissaOfAbsConv f) ↑s
      ⊢ LSeriesSummable (LSeries.logMul f) ↑s
    -/
  · exact LSeriesSummable_logMul_of_lt_re <| by simp [hs]
    /-
      🎉 no goals
    -/
  · refine (LSeriesSummable_of_abscissaOfAbsConv_lt_re <| by simp only [ofReal_re, hs])
      |>.norm.of_norm_bounded_eventually_nat (‖term (logMul f) s ·‖) ?_
    /-
      case a
      f : Nat → Complex
      s : Real
      hs : LT.lt (LSeries.abscissaOfAbsConv (LSeries.logMul f)) ↑s
      ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (LSeries.term f (↑s) i)) ((fun  …
    -/
    filter_upwards [Filter.eventually_ge_atTop <| max 1 (Nat.ceil (Real.exp 1))] with n hn
    simp only [term_of_ne_zero (show n ≠ 0 by omega), logMul, norm_mul, mul_div_assoc,
      ← natCast_log, norm_real]
    /-
      case h
      f : Nat → Complex
      s : Real
      hs : LT.lt (LSeries.abscissaOfAbsConv (LSeries.logMul f)) ↑s
      n : Nat
      hn : LE.le (Max.max 1 (Nat.ceil (Real.exp 1))) n
      ⊢ LE.le (Norm.norm (HDiv.hDiv (f n) (HPow.hPow ↑n ↑s))) (HMul.hMul (Norm.norm  …
    -/
    refine le_mul_of_one_le_left (norm_nonneg _) (.trans ?_ <| Real.le_norm_self _)
    /-
      case h
      f : Nat → Complex
      s : Real
      hs : LT.lt (LSeries.abscissaOfAbsConv (LSeries.logMul f)) ↑s
      n : Nat
      hn : LE.le (Max.max 1 (Nat.ceil (Real.exp 1))) n
      ⊢ LE.le 1 (Real.log ↑n)
    -/
    rw [← Real.log_exp 1]
    /-
      case h
      f : Nat → Complex
      s : Real
      hs : LT.lt (LSeries.abscissaOfAbsConv (LSeries.logMul f)) ↑s
      n : Nat
      hn : LE.le (Max.max 1 (Nat.ceil (Real.exp 1))) n
      ⊢ LE.le (Real.log (Real.exp 1)) (Real.log ↑n)
    -/
    exact Real.log_le_log (Real.exp_pos 1) <| Nat.ceil_le.mp <| (le_max_right _ _).trans hn
    /-
      🎉 no goals
    -/


/-- The abscissa of absolute convergence of the point-wise product of a power of `log` and `f`
is the same as that of `f`. -/
@[simp]
lemma LSeries.absicssaOfAbsConv_logPowMul {f : ℕ → ℂ} {m : ℕ} :
    abscissaOfAbsConv (logMul^[m] f) = abscissaOfAbsConv f := by
  /-
    f : Nat → Complex
    m : Nat
    ⊢ Eq (LSeries.abscissaOfAbsConv (Nat.iterate LSeries.logMul m f)) (LSeries.abs …
  -/
  induction' m with n ih
    /-
      case zero
      f : Nat → Complex
      ⊢ Eq (LSeries.abscissaOfAbsConv (Nat.iterate LSeries.logMul 0 f)) (LSeries.abs …
    -/
  · simp only [Function.iterate_zero, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case succ
      f : Nat → Complex
      n : Nat
      ih : Eq (LSeries.abscissaOfAbsConv (Nat.iterate LSeries.logMul n f)) (LSeries. …
      ⊢ Eq (LSeries.abscissaOfAbsConv (Nat.iterate LSeries.logMul (HAdd.hAdd n 1) f) …
    -/
  · simp only [ih, Function.iterate_succ', Function.comp_def, abscissaOfAbsConv_logMul]
    /-
      🎉 no goals
    -/


/-- If `re s` is greater than the abscissa of absolute convergence of `f`, then
the `m`th derivative of this L-series is `(-1)^m` times the L-series of `log^m * f`. -/
lemma LSeries_iteratedDeriv {f : ℕ → ℂ} (m : ℕ) {s : ℂ} (h : abscissaOfAbsConv f < s.re) :
    iteratedDeriv m (LSeries f) s = (-1) ^ m * LSeries (logMul^[m] f) s := by
  /-
    f : Nat → Complex
    m : Nat
    s : Complex
    h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
    ⊢ Eq (iteratedDeriv m (LSeries f) s) (HMul.hMul (HPow.hPow (-1) m) (LSeries (N …
  -/
  induction' m with m ih generalizing s
    /-
      case zero
      f : Nat → Complex
      s : Complex
      h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
      ⊢ Eq (iteratedDeriv 0 (LSeries f) s) (HMul.hMul (HPow.hPow (-1) 0) (LSeries (N …
    -/
  · simp only [iteratedDeriv_zero, pow_zero, Function.iterate_zero, id_eq, one_mul]
    /-
      🎉 no goals
    -/
  · have ih' : {s | abscissaOfAbsConv f < re s}.EqOn (iteratedDeriv m (LSeries f))
        ((-1) ^ m * LSeries (logMul^[m] f)) := fun _ hs ↦ ih hs
    /-
      case succ
      f : Nat → Complex
      m : Nat
      ih : ∀ {s : Complex}, LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re → Eq (iterated …
      s : Complex
      h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
      ih' : Set.EqOn (iteratedDeriv m (LSeries f)) (HMul.hMul (HPow.hPow (-1) m) (LS …
      ⊢ Eq (iteratedDeriv (HAdd.hAdd m 1) (LSeries f) s) (HMul.hMul (HPow.hPow (-1)  …
    -/
    have := derivWithin_congr ih' (ih h)
    /-
      case succ
      f : Nat → Complex
      m : Nat
      ih : ∀ {s : Complex}, LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re → Eq (iterated …
      s : Complex
      h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
      ih' : Set.EqOn (iteratedDeriv m (LSeries f)) (HMul.hMul (HPow.hPow (-1) m) (LS …
      this : Eq (derivWithin (iteratedDeriv m (LSeries f)) (setOf fun s => LT.lt (LS …
      ⊢ Eq (iteratedDeriv (HAdd.hAdd m 1) (LSeries f) s) (HMul.hMul (HPow.hPow (-1)  …
    -/
    simp_rw [derivWithin_of_isOpen (isOpen_re_gt_EReal _) h] at this
    /-
      case succ
      f : Nat → Complex
      m : Nat
      ih : ∀ {s : Complex}, LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re → Eq (iterated …
      s : Complex
      h : LT.lt (LSeries.abscissaOfAbsConv f) ↑s.re
      ih' : Set.EqOn (iteratedDeriv m (LSeries f)) (HMul.hMul (HPow.hPow (-1) m) (LS …
      this : Eq (deriv (iteratedDeriv m (LSeries f)) s) (deriv (HMul.hMul (HPow.hPow …
      ⊢ Eq (iteratedDeriv (HAdd.hAdd m 1) (LSeries f) s) (HMul.hMul (HPow.hPow (-1)  …
    -/
    rw [iteratedDeriv_succ, this]
    simp only [Pi.mul_def, Pi.pow_apply, Pi.neg_apply, Pi.one_apply, deriv_const_mul_field',
      pow_succ, mul_assoc, neg_one_mul, Function.iterate_succ', Function.comp_def,
      LSeries_deriv <| absicssaOfAbsConv_logPowMul.symm ▸ h]


/-- The L-series of `f` is complex differentiable in its open half-plane of absolute
convergence. -/
lemma LSeries_differentiableOn (f : ℕ → ℂ) :
    DifferentiableOn ℂ (LSeries f) {s | abscissaOfAbsConv f < s.re} :=
  fun _ hz ↦ (LSeries_hasDerivAt hz).differentiableAt.differentiableWithinAt


/-- The L-series of `f` is holomorphic on its open half-plane of absolute convergence. -/
lemma LSeries_analyticOnNhd (f : ℕ → ℂ) :
    AnalyticOnNhd ℂ (LSeries f) {s | abscissaOfAbsConv f < s.re} :=
  (LSeries_differentiableOn f).analyticOnNhd <| isOpen_re_gt_EReal _


lemma LSeries_analyticOn (f : ℕ → ℂ) :
    AnalyticOn ℂ (LSeries f) {s | abscissaOfAbsConv f < s.re} :=
  (LSeries_analyticOnNhd f).analyticOn

