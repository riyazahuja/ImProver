/-- Log-convexity of the Gamma function on the positive reals (stated in multiplicative form),
proved using the Hölder inequality applied to Euler's integral. -/
theorem Gamma_mul_add_mul_le_rpow_Gamma_mul_rpow_Gamma {s t a b : ℝ} (hs : 0 < s) (ht : 0 < t)
    (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
    Gamma (a * s + b * t) ≤ Gamma s ^ a * Gamma t ^ b := by
  -- We will apply Hölder's inequality, for the conjugate exponents `p = 1 / a`
  -- and `q = 1 / b`, to the functions `f a s` and `f b t`, where `f` is as follows:
  /-
    s t a b : Real
    hs : LT.lt 0 s
    ht : LT.lt 0 t
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))) (HMul.hMul (H …
  -/
  let f : ℝ → ℝ → ℝ → ℝ := fun c u x => exp (-c * x) * x ^ (c * (u - 1))
  /-
    s t a b : Real
    hs : LT.lt 0 s
    ht : LT.lt 0 t
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
    ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))) (HMul.hMul (H …
  -/
  have e : IsConjExponent (1 / a) (1 / b) := Real.isConjExponent_one_div ha hb hab
  /-
    s t a b : Real
    hs : LT.lt 0 s
    ht : LT.lt 0 t
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
    e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
    ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))) (HMul.hMul (H …
  -/
  have hab' : b = 1 - a := by linarith
  /-
    s t a b : Real
    hs : LT.lt 0 s
    ht : LT.lt 0 t
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
    e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
    hab' : Eq b (HSub.hSub 1 a)
    ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))) (HMul.hMul (H …
  -/
  have hst : 0 < a * s + b * t := by positivity
  -- some properties of f:
  have posf : ∀ c u x : ℝ, x ∈ Ioi (0 : ℝ) → 0 ≤ f c u x := fun c u x hx =>
    mul_nonneg (exp_pos _).le (rpow_pos_of_pos hx _).le
  have posf' : ∀ c u : ℝ, ∀ᵐ x : ℝ ∂volume.restrict (Ioi 0), 0 ≤ f c u x := fun c u =>
    (ae_restrict_iff' measurableSet_Ioi).mpr (ae_of_all _ (posf c u))
  have fpow :
    ∀ {c x : ℝ} (_ : 0 < c) (u : ℝ) (_ : 0 < x), exp (-x) * x ^ (u - 1) = f c u x ^ (1 / c) := by
    intro c x hc u hx
    dsimp only [f]
    rw [mul_rpow (exp_pos _).le ((rpow_nonneg hx.le) _), ← exp_mul, ← rpow_mul hx.le]
    congr 2 <;> field_simp [hc.ne']; ring
  -- show `f c u` is in `ℒp` for `p = 1/c`:
  have f_mem_Lp :
    ∀ {c u : ℝ} (hc : 0 < c) (hu : 0 < u),
      Memℒp (f c u) (ENNReal.ofReal (1 / c)) (volume.restrict (Ioi 0)) := by
    intro c u hc hu
    have A : ENNReal.ofReal (1 / c) ≠ 0 := by
      rwa [Ne, ENNReal.ofReal_eq_zero, not_le, one_div_pos]
    have B : ENNReal.ofReal (1 / c) ≠ ∞ := ENNReal.ofReal_ne_top
    rw [← memℒp_norm_rpow_iff _ A B, ENNReal.toReal_ofReal (one_div_nonneg.mpr hc.le),
      ENNReal.div_self A B, memℒp_one_iff_integrable]
    · apply Integrable.congr (GammaIntegral_convergent hu)
      refine eventuallyEq_of_mem (self_mem_ae_restrict measurableSet_Ioi) fun x hx => ?_
      dsimp only
      rw [fpow hc u hx]
      congr 1
      exact (norm_of_nonneg (posf _ _ x hx)).symm
    · refine ContinuousOn.aestronglyMeasurable ?_ measurableSet_Ioi
      refine (Continuous.continuousOn ?_).mul (continuousOn_of_forall_continuousAt fun x hx => ?_)
      · exact continuous_exp.comp (continuous_const.mul continuous_id')
      · exact continuousAt_rpow_const _ _ (Or.inl (mem_Ioi.mp hx).ne')
  -- now apply Hölder:
  /-
    s t a b : Real
    hs : LT.lt 0 s
    ht : LT.lt 0 t
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
    e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
    hab' : Eq b (HSub.hSub 1 a)
    hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
    posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
    posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
    fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
    f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
    ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))) (HMul.hMul (H …
  -/
  rw [Gamma_eq_integral hs, Gamma_eq_integral ht, Gamma_eq_integral hst]
  convert
    MeasureTheory.integral_mul_le_Lp_mul_Lq_of_nonneg e (posf' a s) (posf' b t) (f_mem_Lp ha hs)
      (f_mem_Lp hb ht) using
    1
    /-
      case h.e'_3
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
  · refine setIntegral_congr_fun measurableSet_Ioi fun x hx => ?_
    /-
      case h.e'_3
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.hSub (HAdd.hAdd (HMu …
    -/
    dsimp only
    have A : exp (-x) = exp (-a * x) * exp (-b * x) := by
      rw [← exp_add, ← add_mul, ← neg_add, hab, neg_one_mul]
    have B : x ^ (a * s + b * t - 1) = x ^ (a * (s - 1)) * x ^ (b * (t - 1)) := by
      rw [← rpow_add hx, hab']; congr 1; ring
    /-
      case h.e'_3
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      A : Eq (Real.exp (Neg.neg x)) (HMul.hMul (Real.exp (HMul.hMul (Neg.neg a) x))  …
      B : Eq (HPow.hPow x (HSub.hSub (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t)) 1)) …
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.hSub (HAdd.hAdd (HMu …
    -/
    rw [A, B]
    /-
      case h.e'_3
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      A : Eq (Real.exp (Neg.neg x)) (HMul.hMul (Real.exp (HMul.hMul (Neg.neg a) x))  …
      B : Eq (HPow.hPow x (HSub.hSub (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t)) 1)) …
      ⊢ Eq (HMul.hMul (HMul.hMul (Real.exp (HMul.hMul (Neg.neg a) x)) (Real.exp (HMu …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      ⊢ Eq (HMul.hMul (HPow.hPow (MeasureTheory.integral (MeasureTheory.MeasureSpace …
    -/
  · rw [one_div_one_div, one_div_one_div]
    /-
      case h.e'_4
      s t a b : Real
      hs : LT.lt 0 s
      ht : LT.lt 0 t
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      f : Real → Real → Real → Real := fun c u x => HMul.hMul (Real.exp (HMul.hMul ( …
      e : (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
      hab' : Eq b (HSub.hSub 1 a)
      hst : LT.lt 0 (HAdd.hAdd (HMul.hMul a s) (HMul.hMul b t))
      posf : ∀ (c u x : Real), Membership.mem (Set.Ioi 0) x → LE.le 0 (f c u x)
      posf' : ∀ (c u : Real), Filter.Eventually (fun x => LE.le 0 (f c u x)) (Measur …
      fpow : ∀ {c x : Real}, LT.lt 0 c → ∀ (u : Real), LT.lt 0 x → Eq (HMul.hMul (Re …
      f_mem_Lp : ∀ {c u : Real}, LT.lt 0 c → LT.lt 0 u → MeasureTheory.Memℒp (f c u) …
      ⊢ Eq (HMul.hMul (HPow.hPow (MeasureTheory.integral (MeasureTheory.MeasureSpace …
    -/
                /-
                  🎉 no goals
                -/
    congr 2 <;> exact setIntegral_congr_fun measurableSet_Ioi fun x hx => fpow (by assumption) _ hx
                /-
                  🎉 no goals
                -/


theorem convexOn_log_Gamma : ConvexOn ℝ (Ioi 0) (log ∘ Gamma) := by
  /-
    ⊢ ConvexOn Real (Set.Ioi 0) (Function.comp Real.log Real.Gamma)
  -/
  refine convexOn_iff_forall_pos.mpr ⟨convex_Ioi _, fun x hx y hy a b ha hb hab => ?_⟩
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    hy : Membership.mem (Set.Ioi 0) y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (Function.comp Real.log Real.Gamma (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  have : b = 1 - a := by linarith
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    hy : Membership.mem (Set.Ioi 0) y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    this : Eq b (HSub.hSub 1 a)
    ⊢ LE.le (Function.comp Real.log Real.Gamma (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  subst this
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    hy : Membership.mem (Set.Ioi 0) y
    a : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 (HSub.hSub 1 a)
    hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
    ⊢ LE.le (Function.comp Real.log Real.Gamma (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  simp_rw [Function.comp_apply, smul_eq_mul]
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    hy : Membership.mem (Set.Ioi 0) y
    a : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 (HSub.hSub 1 a)
    hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
    ⊢ LE.le (Real.log (Real.Gamma (HAdd.hAdd (HMul.hMul a x) (HMul.hMul (HSub.hSub …
  -/
  simp only [mem_Ioi] at hx hy
  /-
    x y a : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 (HSub.hSub 1 a)
    hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ LE.le (Real.log (Real.Gamma (HAdd.hAdd (HMul.hMul a x) (HMul.hMul (HSub.hSub …
  -/
  rw [← log_rpow, ← log_rpow, ← log_mul]
    /-
      x y a : Real
      ha : LT.lt 0 a
      hb : LT.lt 0 (HSub.hSub 1 a)
      hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
      hx : LT.lt 0 x
      hy : LT.lt 0 y
      ⊢ LE.le (Real.log (Real.Gamma (HAdd.hAdd (HMul.hMul a x) (HMul.hMul (HSub.hSub …
    -/
  · gcongr
    /-
      case hxy
      x y a : Real
      ha : LT.lt 0 a
      hb : LT.lt 0 (HSub.hSub 1 a)
      hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
      hx : LT.lt 0 x
      hy : LT.lt 0 y
      ⊢ LE.le (Real.Gamma (HAdd.hAdd (HMul.hMul a x) (HMul.hMul (HSub.hSub 1 a) y))) …
    -/
    exact Gamma_mul_add_mul_le_rpow_Gamma_mul_rpow_Gamma hx hy ha hb hab
    /-
      🎉 no goals
    -/
  /-
    case hx
    x y a : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 (HSub.hSub 1 a)
    hab : Eq (HAdd.hAdd a (HSub.hSub 1 a)) 1
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Ne (HPow.hPow (Real.Gamma x) a) 0
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


theorem convexOn_Gamma : ConvexOn ℝ (Ioi 0) Gamma := by
  refine
    ((convexOn_exp.subset (subset_univ _) ?_).comp convexOn_log_Gamma
          (exp_monotone.monotoneOn _)).congr
      fun x hx => exp_log (Gamma_pos_of_pos hx)
  /-
    ⊢ Convex Real (Set.image (Function.comp Real.log Real.Gamma) (Set.Ioi 0))
  -/
  rw [convex_iff_isPreconnected]
  /-
    ⊢ IsPreconnected (Set.image (Function.comp Real.log Real.Gamma) (Set.Ioi 0))
  -/
  refine isPreconnected_Ioi.image _ fun x hx => ContinuousAt.continuousWithinAt ?_
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ ContinuousAt (Function.comp Real.log Real.Gamma) x
  -/
  refine (differentiableAt_Gamma fun m => ?_).continuousAt.log (Gamma_pos_of_pos hx).ne'
  /-
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    m : Nat
    ⊢ Ne x (Neg.neg ↑m)
  -/
  exact (neg_lt_iff_pos_add.mpr (add_pos_of_pos_of_nonneg (mem_Ioi.mp hx) (Nat.cast_nonneg m))).ne'
  /-
    🎉 no goals
  -/


/-- The function `n ↦ x log n + log n! - (log x + ... + log (x + n))`, which we will show tends to
`log (Gamma x)` as `n → ∞`. -/
def logGammaSeq (x : ℝ) (n : ℕ) : ℝ :=
  x * log n + log n ! - ∑ m ∈ Finset.range (n + 1), log (x + m)


theorem f_nat_eq (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hn : n ≠ 0) :
    f n = f 1 + log (n - 1)! := by
  /-
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    ⊢ Eq (f ↑n) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub n 1).factorial))
  -/
  refine Nat.le_induction (by simp) (fun m hm IH => ?_) n (Nat.one_le_iff_ne_zero.2 hn)
  /-
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    m : Nat
    hm : LE.le 1 m
    IH : Eq (f ↑m) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub m 1).factorial))
    ⊢ Eq (f ↑(HAdd.hAdd m 1)) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub (HAdd.hAdd m  …
  -/
  have A : 0 < (m : ℝ) := Nat.cast_pos.2 hm
  /-
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    m : Nat
    hm : LE.le 1 m
    IH : Eq (f ↑m) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub m 1).factorial))
    A : LT.lt 0 ↑m
    ⊢ Eq (f ↑(HAdd.hAdd m 1)) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub (HAdd.hAdd m  …
  -/
  simp only [hf_feq A, Nat.cast_add, Nat.cast_one, Nat.add_succ_sub_one, add_zero]
  rw [IH, add_assoc, ← log_mul (Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero _)) A.ne', ←
    Nat.cast_mul]
  /-
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    m : Nat
    hm : LE.le 1 m
    IH : Eq (f ↑m) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub m 1).factorial))
    A : LT.lt 0 ↑m
    ⊢ Eq (HAdd.hAdd (f 1) (Real.log ↑(HMul.hMul (HSub.hSub m 1).factorial m))) (HA …
  -/
  conv_rhs => rw [← Nat.succ_pred_eq_of_pos hm, Nat.factorial_succ, mul_comm]
  /-
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    m : Nat
    hm : LE.le 1 m
    IH : Eq (f ↑m) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub m 1).factorial))
    A : LT.lt 0 ↑m
    ⊢ Eq (HAdd.hAdd (f 1) (Real.log ↑(HMul.hMul (HSub.hSub m 1).factorial m))) (HA …
  -/
  congr
  /-
    case e_a.e_x.e_a.e_a
    f : Real → Real
    n : Nat
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    m : Nat
    hm : LE.le 1 m
    IH : Eq (f ↑m) (HAdd.hAdd (f 1) (Real.log ↑(HSub.hSub m 1).factorial))
    A : LT.lt 0 ↑m
    ⊢ Eq m (HAdd.hAdd m.pred 1)
  -/
  exact (Nat.succ_pred_eq_of_pos hm).symm
  /-
    🎉 no goals
  -/


theorem f_add_nat_eq (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hx : 0 < x) (n : ℕ) :
    f (x + n) = f x + ∑ m ∈ Finset.range n, log (x + m) := by
  induction n with
  | zero => simp
  | succ n hn =>
    have : x + n.succ = x + n + 1 := by push_cast; ring
    rw [this, hf_feq, hn]
    · rw [Finset.range_succ, Finset.sum_insert Finset.not_mem_range_self]
      abel
    · linarith [(Nat.cast_nonneg n : 0 ≤ (n : ℝ))]


/-- Linear upper bound for `f (x + n)` on unit interval -/
theorem f_add_nat_le (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hn : n ≠ 0) (hx : 0 < x) (hx' : x ≤ 1) :
    f (n + x) ≤ f n + x * log n := by
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    hx : LT.lt 0 x
    hx' : LE.le x 1
    ⊢ LE.le (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log ↑n)))
  -/
  have hn' : 0 < (n : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    hx : LT.lt 0 x
    hx' : LE.le x 1
    hn' : LT.lt 0 ↑n
    ⊢ LE.le (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log ↑n)))
  -/
  have : f n + x * log n = (1 - x) * f n + x * f (n + 1) := by rw [hf_feq hn']; ring
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : Ne n 0
    hx : LT.lt 0 x
    hx' : LE.le x 1
    hn' : LT.lt 0 ↑n
    this : Eq (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log ↑n))) (HAdd.hAdd (HMul.hMul …
    ⊢ LE.le (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log ↑n)))
  -/
  rw [this, (by ring : (n : ℝ) + x = (1 - x) * n + x * (n + 1))]
  simpa only [smul_eq_mul] using
    hf_conv.2 hn' (by linarith : 0 < (n + 1 : ℝ)) (by linarith : 0 ≤ 1 - x) hx.le (by linarith)


/-- Linear lower bound for `f (x + n)` on unit interval -/
theorem f_add_nat_ge (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hn : 2 ≤ n) (hx : 0 < x) :
    f n + x * log (n - 1) ≤ f (n + x) := by
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : LE.le 2 n
    hx : LT.lt 0 x
    ⊢ LE.le (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log (HSub.hSub (↑n) 1)))) (f (HAd …
  -/
  have npos : 0 < (n : ℝ) - 1 := by rw [← Nat.cast_one, sub_pos, Nat.cast_lt]; omega
  have c :=
    (convexOn_iff_slope_mono_adjacent.mp <| hf_conv).2 npos (by linarith : 0 < (n : ℝ) + x)
      (by linarith : (n : ℝ) - 1 < (n : ℝ)) (by linarith)
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : LE.le 2 n
    hx : LT.lt 0 x
    npos : LT.lt 0 (HSub.hSub (↑n) 1)
    c : LE.le (HDiv.hDiv (HSub.hSub (f ↑n) (f (HSub.hSub (↑n) 1))) (HSub.hSub (↑n) …
    ⊢ LE.le (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log (HSub.hSub (↑n) 1)))) (f (HAd …
  -/
  rw [add_sub_cancel_left, sub_sub_cancel, div_one] at c
  have : f (↑n - 1) = f n - log (↑n - 1) := by
    -- Porting note: was
    -- nth_rw_rhs 1 [(by ring : (n : ℝ) = ↑n - 1 + 1)]
    -- rw [hf_feq npos, add_sub_cancel]
    rw [eq_sub_iff_add_eq, ← hf_feq npos, sub_add_cancel]
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hn : LE.le 2 n
    hx : LT.lt 0 x
    npos : LT.lt 0 (HSub.hSub (↑n) 1)
    c : LE.le (HSub.hSub (f ↑n) (f (HSub.hSub (↑n) 1))) (HDiv.hDiv (HSub.hSub (f ( …
    this : Eq (f (HSub.hSub (↑n) 1)) (HSub.hSub (f ↑n) (Real.log (HSub.hSub (↑n) 1 …
    ⊢ LE.le (HAdd.hAdd (f ↑n) (HMul.hMul x (Real.log (HSub.hSub (↑n) 1)))) (f (HAd …
  -/
  rwa [this, le_div_iff₀ hx, sub_sub_cancel, le_sub_iff_add_le, mul_comm _ x, add_comm] at c
  /-
    🎉 no goals
  -/


theorem logGammaSeq_add_one (x : ℝ) (n : ℕ) :
    logGammaSeq (x + 1) n = logGammaSeq x (n + 1) + log x - (x + 1) * (log (n + 1) - log n) := by
  /-
    x : Real
    n : Nat
    ⊢ Eq (Real.BohrMollerup.logGammaSeq (HAdd.hAdd x 1) n) (HSub.hSub (HAdd.hAdd ( …
  -/
  dsimp only [Nat.factorial_succ, logGammaSeq]
  /-
    x : Real
    n : Nat
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd x 1) (Real.log ↑n)) (Real.log …
  -/
  conv_rhs => rw [Finset.sum_range_succ', Nat.cast_zero, add_zero]
  /-
    x : Real
    n : Nat
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd x 1) (Real.log ↑n)) (Real.log …
  -/
  rw [Nat.cast_mul, log_mul]; rotate_left
    /-
      case hx
      x : Real
      n : Nat
      ⊢ Ne (↑(HAdd.hAdd n 1)) 0
    -/
  · rw [Nat.cast_ne_zero]; exact Nat.succ_ne_zero n
                           /-
                             🎉 no goals
                           -/
    /-
      case hy
      x : Real
      n : Nat
      ⊢ Ne (↑n.factorial) 0
    -/
  · rw [Nat.cast_ne_zero]; exact Nat.factorial_ne_zero n
                           /-
                             🎉 no goals
                           -/
  have :
    ∑ m ∈ Finset.range (n + 1), log (x + 1 + ↑m) =
      ∑ k ∈ Finset.range (n + 1), log (x + ↑(k + 1)) := by
    congr! 2 with m
    push_cast
    abel
  /-
    x : Real
    n : Nat
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => Real.log (HAdd.hAdd (HA …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd x 1) (Real.log ↑n)) (Real.log …
  -/
  rw [← this, Nat.cast_add_one n]
  /-
    x : Real
    n : Nat
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun m => Real.log (HAdd.hAdd (HA …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd x 1) (Real.log ↑n)) (Real.log …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem le_logGammaSeq (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hx : 0 < x) (hx' : x ≤ 1) (n : ℕ) :
    f x ≤ f 1 + x * log (n + 1) - x * log n + logGammaSeq x n := by
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hx' : LE.le x 1
    n : Nat
    ⊢ LE.le (f x) (HAdd.hAdd (HSub.hSub (HAdd.hAdd (f 1) (HMul.hMul x (Real.log (H …
  -/
  rw [logGammaSeq, ← add_sub_assoc, le_sub_iff_add_le, ← f_add_nat_eq (@hf_feq) hx, add_comm x]
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hx' : LE.le x 1
    n : Nat
    ⊢ LE.le (f (HAdd.hAdd (↑(HAdd.hAdd n 1)) x)) (HAdd.hAdd (HSub.hSub (HAdd.hAdd  …
  -/
  refine (f_add_nat_le hf_conv (@hf_feq) (Nat.add_one_ne_zero n) hx hx').trans (le_of_eq ?_)
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hx' : LE.le x 1
    n : Nat
    ⊢ Eq (HAdd.hAdd (f ↑(HAdd.hAdd n 1)) (HMul.hMul x (Real.log ↑(HAdd.hAdd n 1))) …
  -/
  rw [f_nat_eq @hf_feq (by omega : n + 1 ≠ 0), Nat.add_sub_cancel, Nat.cast_add_one]
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hx' : LE.le x 1
    n : Nat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (f 1) (Real.log ↑n.factorial)) (HMul.hMul x (Real.l …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem ge_logGammaSeq (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hx : 0 < x) (hn : n ≠ 0) :
    f 1 + logGammaSeq x n ≤ f x := by
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hn : Ne n 0
    ⊢ LE.le (HAdd.hAdd (f 1) (Real.BohrMollerup.logGammaSeq x n)) (f x)
  -/
  dsimp [logGammaSeq]
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hn : Ne n 0
    ⊢ LE.le (HAdd.hAdd (f 1) (HSub.hSub (HAdd.hAdd (HMul.hMul x (Real.log ↑n)) (Re …
  -/
  rw [← add_sub_assoc, sub_le_iff_le_add, ← f_add_nat_eq (@hf_feq) hx, add_comm x _]
  /-
    f : Real → Real
    x : Real
    n : Nat
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    hn : Ne n 0
    ⊢ LE.le (HAdd.hAdd (f 1) (HAdd.hAdd (HMul.hMul x (Real.log ↑n)) (Real.log ↑n.f …
  -/
  refine le_trans (le_of_eq ?_) (f_add_nat_ge hf_conv @hf_feq ?_ hx)
    /-
      case refine_1
      f : Real → Real
      x : Real
      n : Nat
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hn : Ne n 0
      ⊢ Eq (HAdd.hAdd (f 1) (HAdd.hAdd (HMul.hMul x (Real.log ↑n)) (Real.log ↑n.fact …
    -/
  · rw [f_nat_eq @hf_feq, Nat.add_sub_cancel, Nat.cast_add_one, add_sub_cancel_right]
      /-
        case refine_1
        f : Real → Real
        x : Real
        n : Nat
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        hx : LT.lt 0 x
        hn : Ne n 0
        ⊢ Eq (HAdd.hAdd (f 1) (HAdd.hAdd (HMul.hMul x (Real.log ↑n)) (Real.log ↑n.fact …
      -/
    · ring
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        f : Real → Real
        x : Real
        n : Nat
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        hx : LT.lt 0 x
        hn : Ne n 0
        ⊢ Ne (HAdd.hAdd n 1) 0
      -/
    · exact Nat.succ_ne_zero _
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      f : Real → Real
      x : Real
      n : Nat
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hn : Ne n 0
      ⊢ LE.le 2 (HAdd.hAdd n 1)
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem tendsto_logGammaSeq_of_le_one (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hx : 0 < x) (hx' : x ≤ 1) :
    Tendsto (logGammaSeq x) atTop (𝓝 <| f x - f 1) := by
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' (f := logGammaSeq x)
    (g := fun n ↦ f x - f 1 - x * (log (n + 1) - log n)) ?_ tendsto_const_nhds ?_ ?_
    /-
      case refine_1
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      ⊢ Filter.Tendsto (fun n => HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMul x (HSu …
    -/
  · have : f x - f 1 = f x - f 1 - x * 0 := by ring
    /-
      case refine_1
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      this : Eq (HSub.hSub (f x) (f 1)) (HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMu …
      ⊢ Filter.Tendsto (fun n => HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMul x (HSu …
    -/
    nth_rw 2 [this]
    /-
      case refine_1
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      this : Eq (HSub.hSub (f x) (f 1)) (HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMu …
      ⊢ Filter.Tendsto (fun n => HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMul x (HSu …
    -/
    exact Tendsto.sub tendsto_const_nhds (tendsto_log_nat_add_one_sub_log.const_mul _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      ⊢ Filter.Eventually (fun b => LE.le ((fun n => HSub.hSub (HSub.hSub (f x) (f 1 …
    -/
  · filter_upwards with n
    /-
      case refine_2.h
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      n : Nat
      ⊢ LE.le (HSub.hSub (HSub.hSub (f x) (f 1)) (HMul.hMul x (HSub.hSub (Real.log ( …
    -/
    rw [sub_le_iff_le_add', sub_le_iff_le_add']
    /-
      case refine_2.h
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      n : Nat
      ⊢ LE.le (f x) (HAdd.hAdd (f 1) (HAdd.hAdd (HMul.hMul x (HSub.hSub (Real.log (H …
    -/
    convert le_logGammaSeq hf_conv (@hf_feq) hx hx' n using 1
    /-
      case h.e'_4
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      n : Nat
      ⊢ Eq (HAdd.hAdd (f 1) (HAdd.hAdd (HMul.hMul x (HSub.hSub (Real.log (HAdd.hAdd  …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      f : Real → Real
      x : Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      hx : LT.lt 0 x
      hx' : LE.le x 1
      ⊢ Filter.Eventually (fun b => LE.le (Real.BohrMollerup.logGammaSeq x b) (HSub. …
    -/
  · show ∀ᶠ n : ℕ in atTop, logGammaSeq x n ≤ f x - f 1
    filter_upwards [eventually_ne_atTop 0] with n hn using
      le_sub_iff_add_le'.mpr (ge_logGammaSeq hf_conv hf_feq hx hn)


theorem tendsto_logGammaSeq (hf_conv : ConvexOn ℝ (Ioi 0) f)
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = f y + log y) (hx : 0 < x) :
    Tendsto (logGammaSeq x) atTop (𝓝 <| f x - f 1) := by
  suffices ∀ m : ℕ, ↑m < x → x ≤ m + 1 → Tendsto (logGammaSeq x) atTop (𝓝 <| f x - f 1) by
    refine this ⌈x - 1⌉₊ ?_ ?_
    · rcases lt_or_le x 1 with ⟨⟩
      · rwa [Nat.ceil_eq_zero.mpr (by linarith : x - 1 ≤ 0), Nat.cast_zero]
      · convert Nat.ceil_lt_add_one (by linarith : 0 ≤ x - 1)
        abel
    · rw [← sub_le_iff_le_add]; exact Nat.le_ceil _
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    ⊢ ∀ (m : Nat), LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Filter.Tendsto (Rea …
  -/
  intro m
  /-
    f : Real → Real
    x : Real
    hf_conv : ConvexOn Real (Set.Ioi 0) f
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
    hx : LT.lt 0 x
    m : Nat
    ⊢ LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Filter.Tendsto (Real.BohrMolleru …
  -/
  induction' m with m hm generalizing x
    /-
      case zero
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      x : Real
      hx : LT.lt 0 x
      ⊢ LT.lt (↑0) x → LE.le x (HAdd.hAdd (↑0) 1) → Filter.Tendsto (Real.BohrMolleru …
    -/
  · rw [Nat.cast_zero, zero_add]
    /-
      case zero
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      x : Real
      hx : LT.lt 0 x
      ⊢ LT.lt 0 x → LE.le x 1 → Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Fil …
    -/
    exact fun _ hx' => tendsto_logGammaSeq_of_le_one hf_conv (@hf_feq) hx hx'
    /-
      🎉 no goals
    -/
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      hm : ∀ {x : Real}, LT.lt 0 x → LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Fil …
      x : Real
      hx : LT.lt 0 x
      ⊢ LT.lt (↑(HAdd.hAdd m 1)) x → LE.le x (HAdd.hAdd (↑(HAdd.hAdd m 1)) 1) → Filt …
    -/
  · intro hy hy'
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      hm : ∀ {x : Real}, LT.lt 0 x → LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Fil …
      x : Real
      hx : LT.lt 0 x
      hy : LT.lt (↑(HAdd.hAdd m 1)) x
      hy' : LE.le x (HAdd.hAdd (↑(HAdd.hAdd m 1)) 1)
      ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
    -/
    rw [Nat.cast_succ, ← sub_le_iff_le_add] at hy'
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      hm : ∀ {x : Real}, LT.lt 0 x → LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Fil …
      x : Real
      hx : LT.lt 0 x
      hy : LT.lt (↑(HAdd.hAdd m 1)) x
      hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
      ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
    -/
    rw [Nat.cast_succ, ← lt_sub_iff_add_lt] at hy
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      hm : ∀ {x : Real}, LT.lt 0 x → LT.lt (↑m) x → LE.le x (HAdd.hAdd (↑m) 1) → Fil …
      x : Real
      hx : LT.lt 0 x
      hy : LT.lt (↑m) (HSub.hSub x 1)
      hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
      ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
    -/
    specialize hm ((Nat.cast_nonneg _).trans_lt hy) hy hy'
    -- now massage gauss_product n (x - 1) into gauss_product (n - 1) x
    have :
      ∀ᶠ n : ℕ in atTop,
        logGammaSeq (x - 1) n =
          logGammaSeq x (n - 1) + x * (log (↑(n - 1) + 1) - log ↑(n - 1)) - log (x - 1) := by
      refine Eventually.mp (eventually_ge_atTop 1) (Eventually.of_forall fun n hn => ?_)
      have := logGammaSeq_add_one (x - 1) (n - 1)
      rw [sub_add_cancel, Nat.sub_add_cancel hn] at this
      rw [this]
      ring
    replace hm :=
      ((Tendsto.congr' this hm).add (tendsto_const_nhds : Tendsto (fun _ => log (x - 1)) _ _)).comp
        (tendsto_add_atTop_nat 1)
    have :
      ((fun x_1 : ℕ =>
            (fun n : ℕ =>
                  logGammaSeq x (n - 1) + x * (log (↑(n - 1) + 1) - log ↑(n - 1)) - log (x - 1))
                x_1 +
              (fun b : ℕ => log (x - 1)) x_1) ∘
          fun a : ℕ => a + 1) =
        fun n => logGammaSeq x n + x * (log (↑n + 1) - log ↑n) := by
      ext1 n
      dsimp only [Function.comp_apply]
      rw [sub_add_cancel, Nat.add_sub_cancel]
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      x : Real
      hx : LT.lt 0 x
      hy : LT.lt (↑m) (HSub.hSub x 1)
      hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
      this✝ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.hS …
      hm : Filter.Tendsto (Function.comp (fun x_1 => HAdd.hAdd (HSub.hSub (HAdd.hAdd …
      this : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAdd …
      ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
    -/
    rw [this] at hm
    /-
      case succ
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) f
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
      m : Nat
      x : Real
      hx : LT.lt 0 x
      hy : LT.lt (↑m) (HSub.hSub x 1)
      hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
      this✝ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.hS …
      hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
      this : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAdd …
      ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
    -/
    convert hm.sub (tendsto_log_nat_add_one_sub_log.const_mul x) using 2
      /-
        case h.e'_3.h
        f : Real → Real
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        m : Nat
        x : Real
        hx : LT.lt 0 x
        hy : LT.lt (↑m) (HSub.hSub x 1)
        hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
        this✝ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.hS …
        hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
        this : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAdd …
        x✝ : Nat
        ⊢ Eq (Real.BohrMollerup.logGammaSeq x x✝) (HSub.hSub (HAdd.hAdd (Real.BohrMoll …
      -/
    · ring
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.e'_3
        f : Real → Real
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        m : Nat
        x : Real
        hx : LT.lt 0 x
        hy : LT.lt (↑m) (HSub.hSub x 1)
        hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
        this✝ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.hS …
        hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
        this : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAdd …
        ⊢ Eq (HSub.hSub (f x) (f 1)) (HSub.hSub (HAdd.hAdd (HSub.hSub (f (HSub.hSub x  …
      -/
    · have := hf_feq ((Nat.cast_nonneg m).trans_lt hy)
      /-
        case h.e'_5.h.e'_3
        f : Real → Real
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        m : Nat
        x : Real
        hx : LT.lt 0 x
        hy : LT.lt (↑m) (HSub.hSub x 1)
        hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
        this✝¹ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.h …
        hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
        this✝ : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAd …
        this : Eq (f (HAdd.hAdd (HSub.hSub x 1) 1)) (HAdd.hAdd (f (HSub.hSub x 1)) (Re …
        ⊢ Eq (HSub.hSub (f x) (f 1)) (HSub.hSub (HAdd.hAdd (HSub.hSub (f (HSub.hSub x  …
      -/
      rw [sub_add_cancel] at this
      /-
        case h.e'_5.h.e'_3
        f : Real → Real
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        m : Nat
        x : Real
        hx : LT.lt 0 x
        hy : LT.lt (↑m) (HSub.hSub x 1)
        hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
        this✝¹ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.h …
        hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
        this✝ : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAd …
        this : Eq (f x) (HAdd.hAdd (f (HSub.hSub x 1)) (Real.log (HSub.hSub x 1)))
        ⊢ Eq (HSub.hSub (f x) (f 1)) (HSub.hSub (HAdd.hAdd (HSub.hSub (f (HSub.hSub x  …
      -/
      rw [this]
      /-
        case h.e'_5.h.e'_3
        f : Real → Real
        hf_conv : ConvexOn Real (Set.Ioi 0) f
        hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HAdd.hAdd (f y) (Re …
        m : Nat
        x : Real
        hx : LT.lt 0 x
        hy : LT.lt (↑m) (HSub.hSub x 1)
        hy' : LE.le (HSub.hSub x 1) (HAdd.hAdd (↑m) 1)
        this✝¹ : Filter.Eventually (fun n => Eq (Real.BohrMollerup.logGammaSeq (HSub.h …
        hm : Filter.Tendsto (fun n => HAdd.hAdd (Real.BohrMollerup.logGammaSeq x n) (H …
        this✝ : Eq (Function.comp (fun x_1 => HAdd.hAdd ((fun n => HSub.hSub (HAdd.hAd …
        this : Eq (f x) (HAdd.hAdd (f (HSub.hSub x 1)) (Real.log (HSub.hSub x 1)))
        ⊢ Eq (HSub.hSub (HAdd.hAdd (f (HSub.hSub x 1)) (Real.log (HSub.hSub x 1))) (f  …
      -/
      ring
      /-
        🎉 no goals
      -/


theorem tendsto_log_gamma {x : ℝ} (hx : 0 < x) :
    Tendsto (logGammaSeq x) atTop (𝓝 <| log (Gamma x)) := by
  have : log (Gamma x) = (log ∘ Gamma) x - (log ∘ Gamma) 1 := by
    simp_rw [Function.comp_apply, Gamma_one, log_one, sub_zero]
  /-
    x : Real
    hx : LT.lt 0 x
    this : Eq (Real.log (Real.Gamma x)) (HSub.hSub (Function.comp Real.log Real.Ga …
    ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (Real.lo …
  -/
  rw [this]
  /-
    x : Real
    hx : LT.lt 0 x
    this : Eq (Real.log (Real.Gamma x)) (HSub.hSub (Function.comp Real.log Real.Ga …
    ⊢ Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub.hS …
  -/
  refine BohrMollerup.tendsto_logGammaSeq convexOn_log_Gamma (fun {y} hy => ?_) hx
  rw [Function.comp_apply, Gamma_add_one hy.ne', log_mul hy.ne' (Gamma_pos_of_pos hy).ne', add_comm,
    Function.comp_apply]


/-- The **Bohr-Mollerup theorem**: the Gamma function is the *unique* log-convex, positive-valued
function on the positive reals which satisfies `f 1 = 1` and `f (x + 1) = x * f x` for all `x`. -/
theorem eq_Gamma_of_log_convex {f : ℝ → ℝ} (hf_conv : ConvexOn ℝ (Ioi 0) (log ∘ f))
    (hf_feq : ∀ {y : ℝ}, 0 < y → f (y + 1) = y * f y) (hf_pos : ∀ {y : ℝ}, 0 < y → 0 < f y)
    (hf_one : f 1 = 1) : EqOn f Gamma (Ioi (0 : ℝ)) := by
  suffices EqOn (log ∘ f) (log ∘ Gamma) (Ioi (0 : ℝ)) from
    fun x hx ↦ log_injOn_pos (hf_pos hx) (Gamma_pos_of_pos hx) (this hx)
  /-
    f : Real → Real
    hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
    hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
    hf_one : Eq (f 1) 1
    ⊢ Set.EqOn (Function.comp Real.log f) (Function.comp Real.log Real.Gamma) (Set …
  -/
  intro x hx
  /-
    f : Real → Real
    hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
    hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
    hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
    hf_one : Eq (f 1) 1
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (Function.comp Real.log f x) (Function.comp Real.log Real.Gamma x)
  -/
  have e1 := BohrMollerup.tendsto_logGammaSeq hf_conv ?_ hx
    /-
      case refine_2
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
      hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
      hf_one : Eq (f 1) 1
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      e1 : Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (HSub …
      ⊢ Eq (Function.comp Real.log f x) (Function.comp Real.log Real.Gamma x)
    -/
  · rw [Function.comp_apply (f := log) (g := f) (x := 1), hf_one, log_one, sub_zero] at e1
    /-
      case refine_2
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
      hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
      hf_one : Eq (f 1) 1
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      e1 : Filter.Tendsto (Real.BohrMollerup.logGammaSeq x) Filter.atTop (nhds (Func …
      ⊢ Eq (Function.comp Real.log f x) (Function.comp Real.log Real.Gamma x)
    -/
    exact tendsto_nhds_unique e1 (BohrMollerup.tendsto_log_gamma hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
      hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
      hf_one : Eq (f 1) 1
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ ∀ {y : Real}, LT.lt 0 y → Eq (Function.comp Real.log f (HAdd.hAdd y 1)) (HAd …
    -/
  · intro y hy
    /-
      case refine_1
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
      hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
      hf_one : Eq (f 1) 1
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      y : Real
      hy : LT.lt 0 y
      ⊢ Eq (Function.comp Real.log f (HAdd.hAdd y 1)) (HAdd.hAdd (Function.comp Real …
    -/
    rw [Function.comp_apply, Function.comp_apply, hf_feq hy, log_mul hy.ne' (hf_pos hy).ne']
    /-
      case refine_1
      f : Real → Real
      hf_conv : ConvexOn Real (Set.Ioi 0) (Function.comp Real.log f)
      hf_feq : ∀ {y : Real}, LT.lt 0 y → Eq (f (HAdd.hAdd y 1)) (HMul.hMul y (f y))
      hf_pos : ∀ {y : Real}, LT.lt 0 y → LT.lt 0 (f y)
      hf_one : Eq (f 1) 1
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      y : Real
      hy : LT.lt 0 y
      ⊢ Eq (HAdd.hAdd (Real.log y) (Real.log (f y))) (HAdd.hAdd (Real.log (f y)) (Re …
    -/
    ring
    /-
      🎉 no goals
    -/


                                      /-
                                        ⊢ Eq (Real.Gamma 2) 1
                                      -/
theorem Gamma_two : Gamma 2 = 1 := by simp [Nat.factorial_one]
                                      /-
                                        🎉 no goals
                                      -/


theorem Gamma_three_div_two_lt_one : Gamma (3 / 2) < 1 := by
  -- This can also be proved using the closed-form evaluation of `Gamma (1 / 2)` in
  -- `Mathlib/Analysis/SpecialFunctions/Gaussian.lean`, but we give a self-contained proof using
  -- log-convexity to avoid unnecessary imports.
  /-
    ⊢ LT.lt (Real.Gamma (3 / 2)) 1
  -/
  have A : (0 : ℝ) < 3 / 2 := by norm_num
  have :=
    BohrMollerup.f_add_nat_le convexOn_log_Gamma (fun {y} hy => ?_) two_ne_zero one_half_pos
      (by norm_num : 1 / 2 ≤ (1 : ℝ))
  /-
    case refine_2
    A : LT.lt 0 (3 / 2)
    this : LE.le (Function.comp Real.log Real.Gamma (HAdd.hAdd (↑2) (1 / 2))) (HAd …
    ⊢ LT.lt (Real.Gamma (3 / 2)) 1
  -/
  swap
  · rw [Function.comp_apply, Gamma_add_one hy.ne', log_mul hy.ne' (Gamma_pos_of_pos hy).ne',
      add_comm, Function.comp_apply]
  rw [Function.comp_apply, Function.comp_apply, Nat.cast_two, Gamma_two, log_one, zero_add,
    (by norm_num : (2 : ℝ) + 1 / 2 = 3 / 2 + 1), Gamma_add_one A.ne',
    log_mul A.ne' (Gamma_pos_of_pos A).ne', ← le_sub_iff_add_le',
    log_le_iff_le_exp (Gamma_pos_of_pos A)] at this
  /-
    case refine_2
    A : LT.lt 0 (3 / 2)
    this : LE.le (Real.Gamma (3 / 2)) (Real.exp (HSub.hSub (HMul.hMul (1 / 2) (Rea …
    ⊢ LT.lt (Real.Gamma (3 / 2)) 1
  -/
  refine this.trans_lt (exp_lt_one_iff.mpr ?_)
  /-
    case refine_2
    A : LT.lt 0 (3 / 2)
    this : LE.le (Real.Gamma (3 / 2)) (Real.exp (HSub.hSub (HMul.hMul (1 / 2) (Rea …
    ⊢ LT.lt (HSub.hSub (HMul.hMul (1 / 2) (Real.log 2)) (Real.log (3 / 2))) 0
  -/
  rw [mul_comm, ← mul_div_assoc, div_sub' _ _ (2 : ℝ) two_ne_zero]
  /-
    case refine_2
    A : LT.lt 0 (3 / 2)
    this : LE.le (Real.Gamma (3 / 2)) (Real.exp (HSub.hSub (HMul.hMul (1 / 2) (Rea …
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HMul.hMul (Real.log 2) 1) (HMul.hMul 2 (Real.lo …
  -/
  refine div_neg_of_neg_of_pos ?_ two_pos
  rw [sub_neg, mul_one, ← Nat.cast_two, ← log_pow, ← exp_lt_exp, Nat.cast_two, exp_log two_pos,
      exp_log] <;>
    /-
      case refine_2
      A : LT.lt 0 (3 / 2)
      this : LE.le (Real.Gamma (3 / 2)) (Real.exp (HSub.hSub (HMul.hMul (1 / 2) (Rea …
      ⊢ LT.lt 2 (HPow.hPow (3 / 2) 2)
    -/
    /-
      🎉 no goals
    -/
    norm_num
    /-
      🎉 no goals
    -/


theorem Gamma_strictMonoOn_Ici : StrictMonoOn Gamma (Ici 2) := by
  convert
    convexOn_Gamma.strict_mono_of_lt (by norm_num : (0 : ℝ) < 3 / 2)
      (by norm_num : (3 / 2 : ℝ) < 2) (Gamma_two.symm ▸ Gamma_three_div_two_lt_one)
  /-
    case h.e'_6
    ⊢ Eq (Set.Ici 2) (Inter.inter (Set.Ioi 0) (Set.Ici 2))
  -/
  symm
  /-
    case h.e'_6
    ⊢ Eq (Inter.inter (Set.Ioi 0) (Set.Ici 2)) (Set.Ici 2)
  -/
  rw [inter_eq_right]
  /-
    case h.e'_6
    ⊢ HasSubset.Subset (Set.Ici 2) (Set.Ioi 0)
  -/
  exact fun x hx => two_pos.trans_le <| mem_Ici.mp hx
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for the doubling formula (we'll show this is equal to `Gamma s`) -/
def doublingGamma (s : ℝ) : ℝ :=
  Gamma (s / 2) * Gamma (s / 2 + 1 / 2) * 2 ^ (s - 1) / √π


theorem doublingGamma_add_one (s : ℝ) (hs : s ≠ 0) :
    doublingGamma (s + 1) = s * doublingGamma s := by
  rw [doublingGamma, doublingGamma, (by abel : s + 1 - 1 = s - 1 + 1), add_div, add_assoc,
    add_halves (1 : ℝ), Gamma_add_one (div_ne_zero hs two_ne_zero), rpow_add two_pos, rpow_one]
  /-
    s : Real
    hs : Ne s 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (Real.Gamma (HAdd.hAdd (HDiv.hDiv s 2) ( …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem doublingGamma_one : doublingGamma 1 = 1 := by
  simp_rw [doublingGamma, Gamma_one_half_eq, add_halves (1 : ℝ), sub_self, Gamma_one, mul_one,
    rpow_zero, mul_one, div_self (sqrt_ne_zero'.mpr pi_pos)]


theorem log_doublingGamma_eq :
    EqOn (log ∘ doublingGamma)
      (fun s => log (Gamma (s / 2)) + log (Gamma (s / 2 + 1 / 2)) + s * log 2 - log (2 * √π))
      (Ioi 0) := by
  /-
    ⊢ Set.EqOn (Function.comp Real.log Real.doublingGamma) (fun s => HSub.hSub (HA …
  -/
  intro s hs
  /-
    s : Real
    hs : Membership.mem (Set.Ioi 0) s
    ⊢ Eq (Function.comp Real.log Real.doublingGamma s) ((fun s => HSub.hSub (HAdd. …
  -/
  have h1 : √π ≠ 0 := sqrt_ne_zero'.mpr pi_pos
  /-
    s : Real
    hs : Membership.mem (Set.Ioi 0) s
    h1 : Ne Real.pi.sqrt 0
    ⊢ Eq (Function.comp Real.log Real.doublingGamma s) ((fun s => HSub.hSub (HAdd. …
  -/
  have h2 : Gamma (s / 2) ≠ 0 := (Gamma_pos_of_pos <| div_pos hs two_pos).ne'
  have h3 : Gamma (s / 2 + 1 / 2) ≠ 0 :=
    (Gamma_pos_of_pos <| add_pos (div_pos hs two_pos) one_half_pos).ne'
  /-
    s : Real
    hs : Membership.mem (Set.Ioi 0) s
    h1 : Ne Real.pi.sqrt 0
    h2 : Ne (Real.Gamma (HDiv.hDiv s 2)) 0
    h3 : Ne (Real.Gamma (HAdd.hAdd (HDiv.hDiv s 2) (1 / 2))) 0
    ⊢ Eq (Function.comp Real.log Real.doublingGamma s) ((fun s => HSub.hSub (HAdd. …
  -/
  have h4 : (2 : ℝ) ^ (s - 1) ≠ 0 := (rpow_pos_of_pos two_pos _).ne'
  rw [Function.comp_apply, doublingGamma, log_div (mul_ne_zero (mul_ne_zero h2 h3) h4) h1,
    log_mul (mul_ne_zero h2 h3) h4, log_mul h2 h3, log_rpow two_pos, log_mul two_ne_zero h1]
  /-
    s : Real
    hs : Membership.mem (Set.Ioi 0) s
    h1 : Ne Real.pi.sqrt 0
    h2 : Ne (Real.Gamma (HDiv.hDiv s 2)) 0
    h3 : Ne (Real.Gamma (HAdd.hAdd (HDiv.hDiv s 2) (1 / 2))) 0
    h4 : Ne (HPow.hPow 2 (HSub.hSub s 1)) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Real.log (Real.Gamma (HDiv.hDiv s 2)))  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem doublingGamma_log_convex_Ioi : ConvexOn ℝ (Ioi (0 : ℝ)) (log ∘ doublingGamma) := by
  /-
    ⊢ ConvexOn Real (Set.Ioi 0) (Function.comp Real.log Real.doublingGamma)
  -/
  refine (((ConvexOn.add ?_ ?_).add ?_).add_const _).congr log_doublingGamma_eq.symm
  · convert
      convexOn_log_Gamma.comp_affineMap (DistribMulAction.toLinearMap ℝ ℝ (1 / 2 : ℝ)).toAffineMap
      using 1
      /-
        case h.e'_9
        ⊢ Eq (Set.Ioi 0) (Set.preimage (⇑(DistribMulAction.toLinearMap Real Real (1 /  …
      -/
    · simpa only [zero_div] using (preimage_const_mul_Ioi (0 : ℝ) one_half_pos).symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_10
        ⊢ Eq (fun s => Real.log (Real.Gamma (HDiv.hDiv s 2))) (Function.comp (Function …
      -/
    · ext1 x
      -- Porting note: was
      -- change log (Gamma (x / 2)) = log (Gamma ((1 / 2 : ℝ) • x))
      /-
        case h.e'_10.h
        x : Real
        ⊢ Eq (Real.log (Real.Gamma (HDiv.hDiv x 2))) (Function.comp (Function.comp Rea …
      -/
      simp only [LinearMap.coe_toAffineMap, Function.comp_apply, DistribMulAction.toLinearMap_apply]
      /-
        case h.e'_10.h
        x : Real
        ⊢ Eq (Real.log (Real.Gamma (HDiv.hDiv x 2))) (Real.log (Real.Gamma (HSMul.hSMu …
      -/
      rw [smul_eq_mul, mul_comm, mul_one_div]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ⊢ ConvexOn Real (Set.Ioi 0) fun s => Real.log (Real.Gamma (HAdd.hAdd (HDiv.hDi …
    -/
  · refine ConvexOn.subset ?_ (Ioi_subset_Ioi <| neg_one_lt_zero.le) (convex_Ioi _)
    convert
      convexOn_log_Gamma.comp_affineMap
        ((DistribMulAction.toLinearMap ℝ ℝ (1 / 2 : ℝ)).toAffineMap +
          AffineMap.const ℝ ℝ (1 / 2 : ℝ)) using 1
      /-
        case h.e'_9
        ⊢ Eq (Set.Ioi (-1)) (Set.preimage (⇑(HAdd.hAdd (DistribMulAction.toLinearMap R …
      -/
    · change Ioi (-1 : ℝ) = ((fun x : ℝ => x + 1 / 2) ∘ fun x : ℝ => (1 / 2 : ℝ) * x) ⁻¹' Ioi 0
      rw [preimage_comp, preimage_add_const_Ioi, zero_sub,
        preimage_const_mul_Ioi (_ : ℝ) one_half_pos, neg_div, div_self (@one_half_pos ℝ _).ne']
      /-
        case h.e'_10
        ⊢ Eq (fun s => Real.log (Real.Gamma (HAdd.hAdd (HDiv.hDiv s 2) (1 / 2)))) (Fun …
      -/
    · ext1 x
      /-
        case h.e'_10.h
        x : Real
        ⊢ Eq (Real.log (Real.Gamma (HAdd.hAdd (HDiv.hDiv x 2) (1 / 2)))) (Function.com …
      -/
      change log (Gamma (x / 2 + 1 / 2)) = log (Gamma ((1 / 2 : ℝ) • x + 1 / 2))
      /-
        case h.e'_10.h
        x : Real
        ⊢ Eq (Real.log (Real.Gamma (HAdd.hAdd (HDiv.hDiv x 2) (1 / 2)))) (Real.log (Re …
      -/
      rw [smul_eq_mul, mul_comm, mul_one_div]
      /-
        🎉 no goals
      -/
  · simpa only [mul_comm _ (log _)] using
      (convexOn_id (convex_Ioi (0 : ℝ))).smul (log_pos one_lt_two).le


theorem doublingGamma_eq_Gamma {s : ℝ} (hs : 0 < s) : doublingGamma s = Gamma s := by
  refine
    eq_Gamma_of_log_convex doublingGamma_log_convex_Ioi
      (fun {y} hy => doublingGamma_add_one y hy.ne') (fun {y} hy => ?_) doublingGamma_one hs
  apply_rules [mul_pos, Gamma_pos_of_pos, add_pos, inv_pos_of_pos, rpow_pos_of_pos, two_pos,
    one_pos, sqrt_pos_of_pos pi_pos]


/-- Legendre's doubling formula for the Gamma function, for positive real arguments. Note that
we shall later prove this for all `s` as `Real.Gamma_mul_Gamma_add_half` (superseding this result)
but this result is needed as an intermediate step. -/
theorem Gamma_mul_Gamma_add_half_of_pos {s : ℝ} (hs : 0 < s) :
    Gamma s * Gamma (s + 1 / 2) = Gamma (2 * s) * 2 ^ (1 - 2 * s) * √π := by
  rw [← doublingGamma_eq_Gamma (mul_pos two_pos hs), doublingGamma,
    mul_div_cancel_left₀ _ (two_ne_zero' ℝ), (by abel : 1 - 2 * s = -(2 * s - 1)),
    rpow_neg zero_le_two]
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (HMul.hMul (Real.Gamma s) (Real.Gamma (HAdd.hAdd s (1 / 2)))) (HMul.hMul  …
  -/
  field_simp [(sqrt_pos_of_pos pi_pos).ne', (rpow_pos_of_pos two_pos (2 * s - 1)).ne']
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (HMul.hMul (HMul.hMul (Real.Gamma s) (Real.Gamma (HDiv.hDiv (HAdd.hAdd (H …
  -/
  ring
  /-
    🎉 no goals
  -/


