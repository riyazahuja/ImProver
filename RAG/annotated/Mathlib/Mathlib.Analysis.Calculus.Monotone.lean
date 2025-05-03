/-- If `(f y - f x) / (y - x)` converges to a limit as `y` tends to `x`, then the same goes if
`y` is shifted a little bit, i.e., `f (y + (y-x)^2) - f x) / (y - x)` converges to the same limit.
This lemma contains a slightly more general version of this statement (where one considers
convergence along some subfilter, typically `𝓝[<] x` or `𝓝[>] x`) tailored to the application
to almost everywhere differentiability of monotone functions. -/
theorem tendsto_apply_add_mul_sq_div_sub {f : ℝ → ℝ} {x a c d : ℝ} {l : Filter ℝ} (hl : l ≤ 𝓝[≠] x)
    (hf : Tendsto (fun y => (f y - d) / (y - x)) l (𝓝 a))
    (h' : Tendsto (fun y => y + c * (y - x) ^ 2) l l) :
    Tendsto (fun y => (f (y + c * (y - x) ^ 2) - d) / (y - x)) l (𝓝 a) := by
  have L : Tendsto (fun y => (y + c * (y - x) ^ 2 - x) / (y - x)) l (𝓝 1) := by
    have : Tendsto (fun y => 1 + c * (y - x)) l (𝓝 (1 + c * (x - x))) := by
      apply Tendsto.mono_left _ (hl.trans nhdsWithin_le_nhds)
      exact ((tendsto_id.sub_const x).const_mul c).const_add 1
    simp only [_root_.sub_self, add_zero, mul_zero] at this
    apply Tendsto.congr' (Eventually.filter_mono hl _) this
    filter_upwards [self_mem_nhdsWithin] with y hy
    field_simp [sub_ne_zero.2 hy]
    ring
  /-
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    ⊢ Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f (HAdd.hAdd y (HMul.hMul c ( …
  -/
  have Z := (hf.comp h').mul L
  /-
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    Z : Filter.Tendsto (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (H …
    ⊢ Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f (HAdd.hAdd y (HMul.hMul c ( …
  -/
  rw [mul_one] at Z
  /-
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    Z : Filter.Tendsto (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (H …
    ⊢ Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f (HAdd.hAdd y (HMul.hMul c ( …
  -/
  apply Tendsto.congr' _ Z
  /-
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    Z : Filter.Tendsto (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (H …
    ⊢ l.EventuallyEq (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (HSu …
  -/
  have : ∀ᶠ y in l, y + c * (y - x) ^ 2 ≠ x := by apply Tendsto.mono_right h' hl self_mem_nhdsWithin
  /-
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    Z : Filter.Tendsto (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (H …
    this : Filter.Eventually (fun y => Ne (HAdd.hAdd y (HMul.hMul c (HPow.hPow (HS …
    ⊢ l.EventuallyEq (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (HSu …
  -/
  filter_upwards [this] with y hy
  /-
    case h
    f : Real → Real
    x a c d : Real
    l : Filter Real
    hl : LE.le l (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))
    hf : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.hSub y x)) l …
    h' : Filter.Tendsto (fun y => HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y …
    L : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (HAdd.hAdd y (HMul.hMul c (H …
    Z : Filter.Tendsto (fun x_1 => HMul.hMul (Function.comp (fun y => HDiv.hDiv (H …
    this : Filter.Eventually (fun y => Ne (HAdd.hAdd y (HMul.hMul c (HPow.hPow (HS …
    y : Real
    hy : Ne (HAdd.hAdd y (HMul.hMul c (HPow.hPow (HSub.hSub y x) 2))) x
    ⊢ Eq (HMul.hMul (Function.comp (fun y => HDiv.hDiv (HSub.hSub (f y) d) (HSub.h …
  -/
  field_simp [sub_ne_zero.2 hy]
  /-
    🎉 no goals
  -/


/-- A Stieltjes function is almost everywhere differentiable, with derivative equal to the
Radon-Nikodym derivative of the associated Stieltjes measure with respect to Lebesgue. -/
theorem StieltjesFunction.ae_hasDerivAt (f : StieltjesFunction) :
    ∀ᵐ x, HasDerivAt f (rnDeriv f.measure volume x).toReal x := by
  /- Denote by `μ` the Stieltjes measure associated to `f`.
    The general theorem `VitaliFamily.ae_tendsto_rnDeriv` ensures that `μ [x, y] / (y - x)` tends
    to the Radon-Nikodym derivative as `y` tends to `x` from the right. As `μ [x,y] = f y - f (x^-)`
    and `f (x^-) = f x` almost everywhere, this gives differentiability on the right.
    On the left, `μ [y, x] / (x - y)` again tends to the Radon-Nikodym derivative.
    As `μ [y, x] = f x - f (y^-)`, this is not exactly the right result, so one uses a sandwiching
    argument to deduce the convergence for `(f x - f y) / (x - y)`. -/
  filter_upwards [VitaliFamily.ae_tendsto_rnDeriv (vitaliFamily (volume : Measure ℝ) 1) f.measure,
    rnDeriv_lt_top f.measure volume, f.countable_leftLim_ne.ae_not_mem volume] with x hx h'x h''x
  -- Limit on the right, following from differentiation of measures
  have L1 :
    Tendsto (fun y => (f y - f x) / (y - x)) (𝓝[>] x) (𝓝 (rnDeriv f.measure volume x).toReal) := by
    apply Tendsto.congr' _
      ((ENNReal.tendsto_toReal h'x.ne).comp (hx.comp (Real.tendsto_Icc_vitaliFamily_right x)))
    filter_upwards [self_mem_nhdsWithin]
    rintro y (hxy : x < y)
    simp only [comp_apply, StieltjesFunction.measure_Icc, Real.volume_Icc, Classical.not_not.1 h''x]
    rw [← ENNReal.ofReal_div_of_pos (sub_pos.2 hxy), ENNReal.toReal_ofReal]
    exact div_nonneg (sub_nonneg.2 (f.mono hxy.le)) (sub_pos.2 hxy).le
  -- Limit on the left, following from differentiation of measures. Its form is not exactly the one
  -- we need, due to the appearance of a left limit.
  have L2 : Tendsto (fun y => (leftLim f y - f x) / (y - x)) (𝓝[<] x)
      (𝓝 (rnDeriv f.measure volume x).toReal) := by
    apply Tendsto.congr' _
      ((ENNReal.tendsto_toReal h'x.ne).comp (hx.comp (Real.tendsto_Icc_vitaliFamily_left x)))
    filter_upwards [self_mem_nhdsWithin]
    rintro y (hxy : y < x)
    simp only [comp_apply, StieltjesFunction.measure_Icc, Real.volume_Icc]
    rw [← ENNReal.ofReal_div_of_pos (sub_pos.2 hxy), ENNReal.toReal_ofReal, ← neg_neg (y - x),
      div_neg, neg_div', neg_sub, neg_sub]
    exact div_nonneg (sub_nonneg.2 (f.mono.leftLim_le hxy.le)) (sub_pos.2 hxy).le
  -- Shifting a little bit the limit on the left, by `(y - x)^2`.
  have L3 : Tendsto (fun y => (leftLim f (y + 1 * (y - x) ^ 2) - f x) / (y - x)) (𝓝[<] x)
      (𝓝 (rnDeriv f.measure volume x).toReal) := by
    apply tendsto_apply_add_mul_sq_div_sub (nhdsLT_le_nhdsNE x) L2
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · apply Tendsto.mono_left _ nhdsWithin_le_nhds
      have : Tendsto (fun y : ℝ => y + ↑1 * (y - x) ^ 2) (𝓝 x) (𝓝 (x + ↑1 * (x - x) ^ 2)) :=
        tendsto_id.add (((tendsto_id.sub_const x).pow 2).const_mul ↑1)
      simpa using this
    · filter_upwards [Ioo_mem_nhdsLT <| show x - 1 < x by simp]
      rintro y ⟨hy : x - 1 < y, h'y : y < x⟩
      rw [mem_Iio]
      nlinarith
  -- Deduce the correct limit on the left, by sandwiching.
  have L4 :
    Tendsto (fun y => (f y - f x) / (y - x)) (𝓝[<] x) (𝓝 (rnDeriv f.measure volume x).toReal) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' L3 L2
    · filter_upwards [self_mem_nhdsWithin]
      rintro y (hy : y < x)
      refine div_le_div_of_nonpos_of_le (by linarith) ((sub_le_sub_iff_right _).2 ?_)
      apply f.mono.le_leftLim
      have : ↑0 < (x - y) ^ 2 := sq_pos_of_pos (sub_pos.2 hy)
      linarith
    · filter_upwards [self_mem_nhdsWithin]
      rintro y (hy : y < x)
      refine div_le_div_of_nonpos_of_le (by linarith) ?_
      simpa only [sub_le_sub_iff_right] using f.mono.leftLim_le (le_refl y)
  -- prove the result by splitting into left and right limits.
  /-
    case h
    f : StieltjesFunction
    x : Real
    hx : Filter.Tendsto (fun a => HDiv.hDiv (f.measure a) (MeasureTheory.MeasureSp …
    h'x : LT.lt (f.measure.rnDeriv MeasureTheory.MeasureSpace.volume x) Top.top
    h''x : Not (Ne (Function.leftLim (↑f) x) (↑f x))
    L1 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (↑f y) (↑f x)) (HSub.hSub y …
    L2 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (Function.leftLim (↑f) y) ( …
    L3 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (Function.leftLim (↑f) (HAd …
    L4 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (↑f y) (↑f x)) (HSub.hSub y …
    ⊢ HasDerivAt (↑f) (f.measure.rnDeriv MeasureTheory.MeasureSpace.volume x).toRe …
  -/
  rw [hasDerivAt_iff_tendsto_slope, slope_fun_def_field, ← nhdsLT_sup_nhdsGT, tendsto_sup]
  /-
    case h
    f : StieltjesFunction
    x : Real
    hx : Filter.Tendsto (fun a => HDiv.hDiv (f.measure a) (MeasureTheory.MeasureSp …
    h'x : LT.lt (f.measure.rnDeriv MeasureTheory.MeasureSpace.volume x) Top.top
    h''x : Not (Ne (Function.leftLim (↑f) x) (↑f x))
    L1 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (↑f y) (↑f x)) (HSub.hSub y …
    L2 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (Function.leftLim (↑f) y) ( …
    L3 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (Function.leftLim (↑f) (HAd …
    L4 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (↑f y) (↑f x)) (HSub.hSub y …
    ⊢ And (Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub (↑f b) (↑f x)) (HSub.hSub …
  -/
  exact ⟨L4, L1⟩
  /-
    🎉 no goals
  -/


/-- A monotone function is almost everywhere differentiable, with derivative equal to the
Radon-Nikodym derivative of the associated Stieltjes measure with respect to Lebesgue. -/
theorem Monotone.ae_hasDerivAt {f : ℝ → ℝ} (hf : Monotone f) :
    ∀ᵐ x, HasDerivAt f (rnDeriv hf.stieltjesFunction.measure volume x).toReal x := by
  /- We already know that the Stieltjes function associated to `f` (i.e., `g : x ↦ f (x^+)`) is
    differentiable almost everywhere. We reduce to this statement by sandwiching values of `f` with
    values of `g`, by shifting with `(y - x)^2` (which has no influence on the relevant
    scale `y - x`.)-/
  filter_upwards [hf.stieltjesFunction.ae_hasDerivAt,
    hf.countable_not_continuousAt.ae_not_mem volume] with x hx h'x
  have A : hf.stieltjesFunction x = f x := by
    rw [Classical.not_not, hf.continuousAt_iff_leftLim_eq_rightLim] at h'x
    apply le_antisymm _ (hf.le_rightLim (le_refl _))
    rw [← h'x]
    exact hf.leftLim_le (le_refl _)
  rw [hasDerivAt_iff_tendsto_slope, (nhdsLT_sup_nhdsGT x).symm, tendsto_sup,
    slope_fun_def_field, A] at hx
  -- prove differentiability on the right, by sandwiching with values of `g`
  have L1 : Tendsto (fun y => (f y - f x) / (y - x)) (𝓝[>] x)
      (𝓝 (rnDeriv hf.stieltjesFunction.measure volume x).toReal) := by
    -- limit of a helper function, with a small shift compared to `g`
    have : Tendsto (fun y => (hf.stieltjesFunction (y + -1 * (y - x) ^ 2) - f x) / (y - x)) (𝓝[>] x)
        (𝓝 (rnDeriv hf.stieltjesFunction.measure volume x).toReal) := by
      apply tendsto_apply_add_mul_sq_div_sub (nhdsGT_le_nhdsNE x) hx.2
      apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
      · apply Tendsto.mono_left _ nhdsWithin_le_nhds
        have : Tendsto (fun y : ℝ => y + -↑1 * (y - x) ^ 2) (𝓝 x) (𝓝 (x + -↑1 * (x - x) ^ 2)) :=
          tendsto_id.add (((tendsto_id.sub_const x).pow 2).const_mul (-1))
        simpa using this
      · filter_upwards [Ioo_mem_nhdsGT <| show x < x + 1 by simp]
        rintro y ⟨hy : x < y, h'y : y < x + 1⟩
        rw [mem_Ioi]
        nlinarith
    -- apply the sandwiching argument, with the helper function and `g`
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' this hx.2
    · filter_upwards [self_mem_nhdsWithin] with y hy
      rw [mem_Ioi, ← sub_pos] at hy
      gcongr
      exact hf.rightLim_le (by nlinarith)
    · filter_upwards [self_mem_nhdsWithin] with y hy
      rw [mem_Ioi, ← sub_pos] at hy
      gcongr
      exact hf.le_rightLim le_rfl
  -- prove differentiability on the left, by sandwiching with values of `g`
  have L2 : Tendsto (fun y => (f y - f x) / (y - x)) (𝓝[<] x)
      (𝓝 (rnDeriv hf.stieltjesFunction.measure volume x).toReal) := by
    -- limit of a helper function, with a small shift compared to `g`
    have : Tendsto (fun y => (hf.stieltjesFunction (y + -1 * (y - x) ^ 2) - f x) / (y - x)) (𝓝[<] x)
        (𝓝 (rnDeriv hf.stieltjesFunction.measure volume x).toReal) := by
      apply tendsto_apply_add_mul_sq_div_sub (nhdsLT_le_nhdsNE x) hx.1
      apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
      · apply Tendsto.mono_left _ nhdsWithin_le_nhds
        have : Tendsto (fun y : ℝ => y + -↑1 * (y - x) ^ 2) (𝓝 x) (𝓝 (x + -↑1 * (x - x) ^ 2)) :=
          tendsto_id.add (((tendsto_id.sub_const x).pow 2).const_mul (-1))
        simpa using this
      · filter_upwards [Ioo_mem_nhdsLT <| show x - 1 < x by simp]
        rintro y hy
        rw [mem_Ioo] at hy
        rw [mem_Iio]
        nlinarith
    -- apply the sandwiching argument, with `g` and the helper function
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hx.1 this
    · filter_upwards [self_mem_nhdsWithin]
      rintro y hy
      rw [mem_Iio, ← sub_neg] at hy
      apply div_le_div_of_nonpos_of_le hy.le
      exact (sub_le_sub_iff_right _).2 (hf.le_rightLim (le_refl _))
    · filter_upwards [self_mem_nhdsWithin]
      rintro y hy
      rw [mem_Iio, ← sub_neg] at hy
      have : 0 < (y - x) ^ 2 := sq_pos_of_neg hy
      apply div_le_div_of_nonpos_of_le hy.le
      exact (sub_le_sub_iff_right _).2 (hf.rightLim_le (by linarith))
  -- conclude global differentiability
  /-
    case h
    f : Real → Real
    hf : Monotone f
    x : Real
    hx : And (Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub (↑hf.stieltjesFunction …
    h'x : Not (Not (ContinuousAt f x))
    A : Eq (↑hf.stieltjesFunction x) (f x)
    L1 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x …
    L2 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x …
    ⊢ HasDerivAt f (hf.stieltjesFunction.measure.rnDeriv MeasureTheory.MeasureSpac …
  -/
  rw [hasDerivAt_iff_tendsto_slope, slope_fun_def_field, ← nhdsLT_sup_nhdsGT, tendsto_sup]
  /-
    case h
    f : Real → Real
    hf : Monotone f
    x : Real
    hx : And (Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub (↑hf.stieltjesFunction …
    h'x : Not (Not (ContinuousAt f x))
    A : Eq (↑hf.stieltjesFunction x) (f x)
    L1 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x …
    L2 : Filter.Tendsto (fun y => HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x …
    ⊢ And (Filter.Tendsto (fun b => HDiv.hDiv (HSub.hSub (f b) (f x)) (HSub.hSub b …
  -/
  exact ⟨L2, L1⟩
  /-
    🎉 no goals
  -/


/-- A monotone real function is differentiable Lebesgue-almost everywhere. -/
theorem Monotone.ae_differentiableAt {f : ℝ → ℝ} (hf : Monotone f) :
    ∀ᵐ x, DifferentiableAt ℝ f x := by
  /-
    f : Real → Real
    hf : Monotone f
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae Mea …
  -/
  filter_upwards [hf.ae_hasDerivAt] with x hx using hx.differentiableAt
  /-
    🎉 no goals
  -/


/-- A real function which is monotone on a set is differentiable Lebesgue-almost everywhere on
this set. This version does not assume that `s` is measurable. For a formulation with
`volume.restrict s` assuming that `s` is measurable, see `MonotoneOn.ae_differentiableWithinAt`.
-/
theorem MonotoneOn.ae_differentiableWithinAt_of_mem {f : ℝ → ℝ} {s : Set ℝ} (hf : MonotoneOn f s) :
    ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /- We use a global monotone extension of `f`, and argue that this extension is differentiable
    almost everywhere. Such an extension need not exist (think of `1/x` on `(0, +∞)`), but it exists
    if one restricts first the function to a compact interval `[a, b]`. -/
  /-
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  apply ae_of_mem_of_ae_of_mem_inter_Ioo
  /-
    case h
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    ⊢ ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filter …
  -/
  intro a b as bs _
  obtain ⟨g, hg, gf⟩ : ∃ g : ℝ → ℝ, Monotone g ∧ EqOn f g (s ∩ Icc a b) :=
    (hf.mono inter_subset_left).exists_monotone_extension
      (hf.map_bddBelow inter_subset_left ⟨a, fun x hx => hx.2.1, as⟩)
      (hf.map_bddAbove inter_subset_left ⟨b, fun x hx => hx.2.2, bs⟩)
  /-
    case h.intro.intro
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    ⊢ Filter.Eventually (fun x => Membership.mem (Inter.inter s (Set.Ioo a b)) x → …
  -/
  filter_upwards [hg.ae_differentiableAt] with x hx
  /-
    case h
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    x : Real
    hx : DifferentiableAt Real g x
    ⊢ Membership.mem (Inter.inter s (Set.Ioo a b)) x → DifferentiableWithinAt Real …
  -/
  intro h'x
  /-
    case h
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    x : Real
    hx : DifferentiableAt Real g x
    h'x : Membership.mem (Inter.inter s (Set.Ioo a b)) x
    ⊢ DifferentiableWithinAt Real f s x
  -/
  apply hx.differentiableWithinAt.congr_of_eventuallyEq _ (gf ⟨h'x.1, h'x.2.1.le, h'x.2.2.le⟩)
  /-
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    x : Real
    hx : DifferentiableAt Real g x
    h'x : Membership.mem (Inter.inter s (Set.Ioo a b)) x
    ⊢ (nhdsWithin x s).EventuallyEq f g
  -/
  have : Ioo a b ∈ 𝓝[s] x := nhdsWithin_le_nhds (Ioo_mem_nhds h'x.2.1 h'x.2.2)
  /-
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    x : Real
    hx : DifferentiableAt Real g x
    h'x : Membership.mem (Inter.inter s (Set.Ioo a b)) x
    this : Membership.mem (nhdsWithin x s) (Set.Ioo a b)
    ⊢ (nhdsWithin x s).EventuallyEq f g
  -/
  filter_upwards [self_mem_nhdsWithin, this] with y hy h'y
  /-
    case h
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    a b : Real
    as : Membership.mem s a
    bs : Membership.mem s b
    a✝ : LT.lt a b
    g : Real → Real
    hg : Monotone g
    gf : Set.EqOn f g (Inter.inter s (Set.Icc a b))
    x : Real
    hx : DifferentiableAt Real g x
    h'x : Membership.mem (Inter.inter s (Set.Ioo a b)) x
    this : Membership.mem (nhdsWithin x s) (Set.Ioo a b)
    y : Real
    hy : Membership.mem s y
    h'y : Membership.mem (Set.Ioo a b) y
    ⊢ Eq (f y) (g y)
  -/
  exact gf ⟨hy, h'y.1.le, h'y.2.le⟩
  /-
    🎉 no goals
  -/


/-- A real function which is monotone on a set is differentiable Lebesgue-almost everywhere on
this set. This version assumes that `s` is measurable and uses `volume.restrict s`.
For a formulation without measurability assumption,
see `MonotoneOn.ae_differentiableWithinAt_of_mem`. -/
theorem MonotoneOn.ae_differentiableWithinAt {f : ℝ → ℝ} {s : Set ℝ} (hf : MonotoneOn f s)
    (hs : MeasurableSet s) : ∀ᵐ x ∂volume.restrict s, DifferentiableWithinAt ℝ f s x := by
  /-
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => DifferentiableWithinAt Real f s x) (MeasureTheor …
  -/
  rw [ae_restrict_iff' hs]
  /-
    f : Real → Real
    s : Set Real
    hf : MonotoneOn f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  exact hf.ae_differentiableWithinAt_of_mem
  /-
    🎉 no goals
  -/

