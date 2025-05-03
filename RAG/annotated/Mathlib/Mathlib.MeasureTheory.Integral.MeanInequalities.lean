theorem lintegral_mul_le_one_of_lintegral_rpow_eq_one {p q : ℝ} (hpq : p.IsConjExponent q)
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hf_norm : ∫⁻ a, f a ^ p ∂μ = 1)
    (hg_norm : ∫⁻ a, g a ^ q ∂μ = 1) : (∫⁻ a, (f * g) a ∂μ) ≤ 1 := by
  calc
    (∫⁻ a : α, (f * g) a ∂μ) ≤
        ∫⁻ a : α, f a ^ p / ENNReal.ofReal p + g a ^ q / ENNReal.ofReal q ∂μ :=
      lintegral_mono fun a => young_inequality (f a) (g a) hpq
    _ = 1 := by
      simp only [div_eq_mul_inv]
      rw [lintegral_add_left']
      · rw [lintegral_mul_const'' _ (hf.pow_const p), lintegral_mul_const', hf_norm, hg_norm,
          one_mul, one_mul, hpq.inv_add_inv_conj_ennreal]
        simp [hpq.symm.pos]
      · exact (hf.pow_const _).mul_const _


/-- Function multiplied by the inverse of its p-seminorm `(∫⁻ f^p ∂μ) ^ 1/p`-/
def funMulInvSnorm (f : α → ℝ≥0∞) (p : ℝ) (μ : Measure α) : α → ℝ≥0∞ := fun a =>
  f a * ((∫⁻ c, f c ^ p ∂μ) ^ (1 / p))⁻¹


theorem fun_eq_funMulInvSnorm_mul_eLpNorm {p : ℝ} (f : α → ℝ≥0∞)
    (hf_nonzero : (∫⁻ a, f a ^ p ∂μ) ≠ 0) (hf_top : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤) {a : α} :
    f a = funMulInvSnorm f p μ a * (∫⁻ c, f c ^ p ∂μ) ^ (1 / p) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f : α → ENNReal
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    a : α
    ⊢ Eq (f a) (HMul.hMul (ENNReal.funMulInvSnorm f p μ a) (HPow.hPow (MeasureTheo …
  -/
  simp [funMulInvSnorm, mul_assoc, ENNReal.inv_mul_cancel, hf_nonzero, hf_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias fun_eq_funMulInvSnorm_mul_snorm := fun_eq_funMulInvSnorm_mul_eLpNorm


theorem funMulInvSnorm_rpow {p : ℝ} (hp0 : 0 < p) {f : α → ℝ≥0∞} {a : α} :
    funMulInvSnorm f p μ a ^ p = f a ^ p * (∫⁻ c, f c ^ p ∂μ)⁻¹ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LT.lt 0 p
    f : α → ENNReal
    a : α
    ⊢ Eq (HPow.hPow (ENNReal.funMulInvSnorm f p μ a) p) (HMul.hMul (HPow.hPow (f a …
  -/
  rw [funMulInvSnorm, mul_rpow_of_nonneg _ _ (le_of_lt hp0)]
  suffices h_inv_rpow : ((∫⁻ c : α, f c ^ p ∂μ) ^ (1 / p))⁻¹ ^ p = (∫⁻ c : α, f c ^ p ∂μ)⁻¹ by
    rw [h_inv_rpow]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LT.lt 0 p
    f : α → ENNReal
    a : α
    ⊢ Eq (HPow.hPow (Inv.inv (HPow.hPow (MeasureTheory.lintegral μ fun c => HPow.h …
  -/
  rw [inv_rpow, ← rpow_mul, one_div_mul_cancel hp0.ne', rpow_one]
  /-
    🎉 no goals
  -/


theorem lintegral_rpow_funMulInvSnorm_eq_one {p : ℝ} (hp0_lt : 0 < p) {f : α → ℝ≥0∞}
    (hf_nonzero : (∫⁻ a, f a ^ p ∂μ) ≠ 0) (hf_top : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤) :
    ∫⁻ c, funMulInvSnorm f p μ c ^ p ∂μ = 1 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0_lt : LT.lt 0 p
    f : α → ENNReal
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun c => HPow.hPow (ENNReal.funMulInvSnorm f p …
  -/
  simp_rw [funMulInvSnorm_rpow hp0_lt]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0_lt : LT.lt 0 p
    f : α → ENNReal
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun c => HMul.hMul (HPow.hPow (f c) p) (Inv.in …
  -/
  rw [lintegral_mul_const', ENNReal.mul_inv_cancel hf_nonzero hf_top]
  /-
    case hr
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0_lt : LT.lt 0 p
    f : α → ENNReal
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    ⊢ Ne (Inv.inv (MeasureTheory.lintegral μ fun c => HPow.hPow (f c) p)) Top.top
  -/
  rwa [inv_ne_top]
  /-
    🎉 no goals
  -/


/-- Hölder's inequality in case of finite non-zero integrals -/
theorem lintegral_mul_le_Lp_mul_Lq_of_ne_zero_of_ne_top {p q : ℝ} (hpq : p.IsConjExponent q)
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hf_nontop : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤)
    (hg_nontop : (∫⁻ a, g a ^ q ∂μ) ≠ ⊤) (hf_nonzero : (∫⁻ a, f a ^ p ∂μ) ≠ 0)
    (hg_nonzero : (∫⁻ a, g a ^ q ∂μ) ≠ 0) :
    (∫⁻ a, (f * g) a ∂μ) ≤ (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) * (∫⁻ a, g a ^ q ∂μ) ^ (1 / q) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_nontop : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nontop : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) Top.top
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  let npf := (∫⁻ c : α, f c ^ p ∂μ) ^ (1 / p)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_nontop : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nontop : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) Top.top
    hf_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    npf : ENNReal := HPow.hPow (MeasureTheory.lintegral μ fun c => HPow.hPow (f c) …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  let nqg := (∫⁻ c : α, g c ^ q ∂μ) ^ (1 / q)
  calc
    (∫⁻ a : α, (f * g) a ∂μ) =
        ∫⁻ a : α, (funMulInvSnorm f p μ * funMulInvSnorm g q μ) a * (npf * nqg) ∂μ := by
      refine lintegral_congr fun a => ?_
      rw [Pi.mul_apply, fun_eq_funMulInvSnorm_mul_eLpNorm f hf_nonzero hf_nontop,
        fun_eq_funMulInvSnorm_mul_eLpNorm g hg_nonzero hg_nontop, Pi.mul_apply]
      ring
    _ ≤ npf * nqg := by
      rw [lintegral_mul_const' (npf * nqg) _
          (by simp [npf, nqg, hf_nontop, hg_nontop, hf_nonzero, hg_nonzero, ENNReal.mul_eq_top])]
      refine mul_le_of_le_one_left' ?_
      have hf1 := lintegral_rpow_funMulInvSnorm_eq_one hpq.pos hf_nonzero hf_nontop
      have hg1 := lintegral_rpow_funMulInvSnorm_eq_one hpq.symm.pos hg_nonzero hg_nontop
      exact lintegral_mul_le_one_of_lintegral_rpow_eq_one hpq (hf.mul_const _) hf1 hg1


theorem ae_eq_zero_of_lintegral_rpow_eq_zero {p : ℝ} (hp0 : 0 ≤ p) {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_zero : ∫⁻ a, f a ^ p ∂μ = 0) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  rw [lintegral_eq_zero_iff' (hf.pow_const p)] at hf_zero
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : (MeasureTheory.ae μ).EventuallyEq (fun x => HPow.hPow (f x) p) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  filter_upwards [hf_zero] with x
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : (MeasureTheory.ae μ).EventuallyEq (fun x => HPow.hPow (f x) p) 0
    x : α
    ⊢ Eq (HPow.hPow (f x) p) (0 x) → Eq (f x) (0 x)
  -/
  rw [Pi.zero_apply, ← not_imp_not]
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : (MeasureTheory.ae μ).EventuallyEq (fun x => HPow.hPow (f x) p) 0
    x : α
    ⊢ Not (Eq (f x) 0) → Not (Eq (HPow.hPow (f x) p) 0)
  -/
  exact fun hx => (rpow_pos_of_nonneg (pos_iff_ne_zero.2 hx) hp0).ne'
  /-
    🎉 no goals
  -/


theorem lintegral_mul_eq_zero_of_lintegral_rpow_eq_zero {p : ℝ} (hp0 : 0 ≤ p) {f g : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_zero : ∫⁻ a, f a ^ p ∂μ = 0) : (∫⁻ a, (f * g) a ∂μ) = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) 0
  -/
  rw [← @lintegral_zero_fun α _ μ]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (MeasureTheory.linte …
  -/
  refine lintegral_congr_ae ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f g) 0
  -/
  suffices h_mul_zero : f * g =ᵐ[μ] 0 * g by rwa [zero_mul] at h_mul_zero
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f g) (HMul.hMul 0 g)
  -/
  have hf_eq_zero : f =ᵐ[μ] 0 := ae_eq_zero_of_lintegral_rpow_eq_zero hp0 hf hf_zero
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    hp0 : LE.le 0 p
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
    hf_eq_zero : (MeasureTheory.ae μ).EventuallyEq f 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f g) (HMul.hMul 0 g)
  -/
  exact hf_eq_zero.mul (ae_eq_refl g)
  /-
    🎉 no goals
  -/


theorem lintegral_mul_le_Lp_mul_Lq_of_ne_zero_of_eq_top {p q : ℝ} (hp0_lt : 0 < p) (hq0 : 0 ≤ q)
    {f g : α → ℝ≥0∞} (hf_top : ∫⁻ a, f a ^ p ∂μ = ⊤) (hg_nonzero : (∫⁻ a, g a ^ q ∂μ) ≠ 0) :
    (∫⁻ a, (f * g) a ∂μ) ≤ (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) * (∫⁻ a, g a ^ q ∂μ) ^ (1 / q) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hp0_lt : LT.lt 0 p
    hq0 : LE.le 0 q
    f g : α → ENNReal
    hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  refine le_trans le_top (le_of_eq ?_)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hp0_lt : LT.lt 0 p
    hq0 : LE.le 0 q
    f g : α → ENNReal
    hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    ⊢ Eq Top.top (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hP …
  -/
  have hp0_inv_lt : 0 < 1 / p := by simp [hp0_lt]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hp0_lt : LT.lt 0 p
    hq0 : LE.le 0 q
    f g : α → ENNReal
    hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    hp0_inv_lt : LT.lt 0 (HDiv.hDiv 1 p)
    ⊢ Eq Top.top (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hP …
  -/
  rw [hf_top, ENNReal.top_rpow_of_pos hp0_inv_lt]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hp0_lt : LT.lt 0 p
    hq0 : LE.le 0 q
    f g : α → ENNReal
    hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_nonzero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
    hp0_inv_lt : LT.lt 0 (HDiv.hDiv 1 p)
    ⊢ Eq Top.top (HMul.hMul Top.top (HPow.hPow (MeasureTheory.lintegral μ fun a => …
  -/
  simp [hq0, hg_nonzero]
  /-
    🎉 no goals
  -/


/-- Hölder's inequality for functions `α → ℝ≥0∞`. The integral of the product of two functions
is bounded by the product of their `ℒp` and `ℒq` seminorms when `p` and `q` are conjugate
exponents. -/
theorem lintegral_mul_le_Lp_mul_Lq (μ : Measure α) {p q : ℝ} (hpq : p.IsConjExponent q)
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    (∫⁻ a, (f * g) a ∂μ) ≤ (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) * (∫⁻ a, g a ^ q ∂μ) ^ (1 / q) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  by_cases hf_zero : ∫⁻ a, f a ^ p ∂μ = 0
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
    -/
  · refine Eq.trans_le ?_ (zero_le _)
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0
      ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) 0
    -/
    exact lintegral_mul_eq_zero_of_lintegral_rpow_eq_zero hpq.nonneg hf hf_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  by_cases hg_zero : ∫⁻ a, g a ^ q ∂μ = 0
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
    -/
  · refine Eq.trans_le ?_ (zero_le _)
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
      ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) 0
    -/
    rw [mul_comm]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0
      ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul g f a) 0
    -/
    exact lintegral_mul_eq_zero_of_lintegral_rpow_eq_zero hpq.symm.nonneg hg hg_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
    hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  by_cases hf_top : ∫⁻ a, f a ^ p ∂μ = ⊤
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
      hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
    -/
  · exact lintegral_mul_le_Lp_mul_Lq_of_ne_zero_of_eq_top hpq.pos hpq.symm.nonneg hf_top hg_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
    hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  by_cases hg_top : ∫⁻ a, g a ^ q ∂μ = ⊤
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) Top.top
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
    -/
  · rw [mul_comm, mul_comm ((∫⁻ a : α, f a ^ p ∂μ) ^ (1 / p))]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
      hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) Top.top
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul g f a) (HMul.hMul (HPow. …
    -/
    exact lintegral_mul_le_Lp_mul_Lq_of_ne_zero_of_eq_top hpq.symm.pos hpq.nonneg hg_top hf_zero
    /-
      🎉 no goals
    -/
  -- non-⊤ non-zero case
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) 0)
    hg_zero : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) 0)
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) q) Top.top)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul f g a) (HMul.hMul (HPow. …
  -/
  exact ENNReal.lintegral_mul_le_Lp_mul_Lq_of_ne_zero_of_ne_top hpq hf hf_top hg_top hf_zero hg_zero
  /-
    🎉 no goals
  -/


/-- A different formulation of Hölder's inequality for two functions, with two exponents that sum to
1, instead of reciprocals of  -/
theorem lintegral_mul_norm_pow_le {α} [MeasurableSpace α] {μ : Measure α}
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ)
    {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) (hpq : p + q = 1) :
    ∫⁻ a, f a ^ p * g a ^ q ∂μ ≤ (∫⁻ a, f a ∂μ) ^ p * (∫⁻ a, g a ∂μ) ^ q := by
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    p q : Real
    hp : LE.le 0 p
    hq : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
  -/
  rcases hp.eq_or_lt with rfl|hp
    /-
      case inl
      α : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      q : Real
      hq : LE.le 0 q
      hp : LE.le 0 0
      hpq : Eq (HAdd.hAdd 0 q) 1
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) 0) (HPo …
    -/
  · rw [zero_add] at hpq
    /-
      case inl
      α : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      q : Real
      hq : LE.le 0 q
      hp : LE.le 0 0
      hpq : Eq q 1
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) 0) (HPo …
    -/
    simp [hpq]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    p q : Real
    hp✝ : LE.le 0 p
    hq : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    hp : LT.lt 0 p
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
  -/
  rcases hq.eq_or_lt with rfl|hq
    /-
      case inr.inl
      α : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      p : Real
      hp✝ : LE.le 0 p
      hp : LT.lt 0 p
      hq : LE.le 0 0
      hpq : Eq (HAdd.hAdd p 0) 1
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
    -/
  · rw [add_zero] at hpq
    /-
      case inr.inl
      α : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      p : Real
      hp✝ : LE.le 0 p
      hp : LT.lt 0 p
      hq : LE.le 0 0
      hpq : Eq p 1
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
    -/
    simp [hpq]
    /-
      🎉 no goals
    -/
  have h2p : 1 < 1 / p := by
    rw [one_div, one_lt_inv₀ hp]
    linarith
  /-
    case inr.inr
    α : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    p q : Real
    hp✝ : LE.le 0 p
    hq✝ : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    hp : LT.lt 0 p
    hq : LT.lt 0 q
    h2p : LT.lt 1 (HDiv.hDiv 1 p)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
  -/
  have h2pq : (1 / p)⁻¹ + (1 / q)⁻¹ = 1 := by simp [hp.ne', hq.ne', hpq]
  /-
    case inr.inr
    α : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    p q : Real
    hp✝ : LE.le 0 p
    hq✝ : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    hp : LT.lt 0 p
    hq : LT.lt 0 q
    h2p : LT.lt 1 (HDiv.hDiv 1 p)
    h2pq : Eq (HAdd.hAdd (Inv.inv (HDiv.hDiv 1 p)) (Inv.inv (HDiv.hDiv 1 q))) 1
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
  -/
  have := ENNReal.lintegral_mul_le_Lp_mul_Lq μ ⟨h2p, h2pq⟩ (hf.pow_const p) (hg.pow_const q)
  /-
    case inr.inr
    α : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    p q : Real
    hp✝ : LE.le 0 p
    hq✝ : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    hp : LT.lt 0 p
    hq : LT.lt 0 q
    h2p : LT.lt 1 (HDiv.hDiv 1 p)
    h2pq : Eq (HAdd.hAdd (Inv.inv (HDiv.hDiv 1 p)) (Inv.inv (HDiv.hDiv 1 q))) 1
    this : LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (fun x => HPow.hPow …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (HPow.hPow (f a) p) (HPo …
  -/
  simpa [← ENNReal.rpow_mul, hp.ne', hq.ne'] using this
  /-
    🎉 no goals
  -/


/-- A version of Hölder with multiple arguments -/
theorem lintegral_prod_norm_pow_le {α ι : Type*} [MeasurableSpace α] {μ : Measure α}
    (s : Finset ι) {f : ι → α → ℝ≥0∞} (hf : ∀ i ∈ s, AEMeasurable (f i) μ)
    {p : ι → ℝ} (hp : ∑ i ∈ s, p i = 1) (h2p : ∀ i ∈ s, 0 ≤ p i) :
    ∫⁻ a, ∏ i ∈ s, f i a ^ p i ∂μ ≤ ∏ i ∈ s, (∫⁻ a, f i a ∂μ) ^ p i := by
  classical
  induction s using Finset.induction generalizing p with
  | empty =>
    simp at hp
  | @insert i₀ s hi₀ ih =>
    rcases eq_or_ne (p i₀) 1 with h2i₀|h2i₀
    · simp only [hi₀, not_false_eq_true, prod_insert]
      have h2p : ∀ i ∈ s, p i = 0 := by
        simpa [hi₀, h2i₀, sum_eq_zero_iff_of_nonneg (fun i hi ↦ h2p i <| mem_insert_of_mem hi)]
          using hp
      calc ∫⁻ a, f i₀ a ^ p i₀ * ∏ i ∈ s, f i a ^ p i ∂μ
          = ∫⁻ a, f i₀ a ^ p i₀ * ∏ i ∈ s, 1 ∂μ := by
            congr! 3 with x
            apply prod_congr rfl fun i hi ↦ by rw [h2p i hi, ENNReal.rpow_zero]
        _ ≤ (∫⁻ a, f i₀ a ∂μ) ^ p i₀ * ∏ i ∈ s, 1 := by simp [h2i₀]
        _ = (∫⁻ a, f i₀ a ∂μ) ^ p i₀ * ∏ i ∈ s, (∫⁻ a, f i a ∂μ) ^ p i := by
            congr 1
            apply prod_congr rfl fun i hi ↦ by rw [h2p i hi, ENNReal.rpow_zero]
    · have hpi₀ : 0 ≤ 1 - p i₀ := by
        simp_rw [sub_nonneg, ← hp, single_le_sum h2p (mem_insert_self ..)]
      have h2pi₀ : 1 - p i₀ ≠ 0 := by
        rwa [sub_ne_zero, ne_comm]
      let q := fun i ↦ p i / (1 - p i₀)
      have hq : ∑ i ∈ s, q i = 1 := by
        rw [← Finset.sum_div, ← sum_insert_sub hi₀, hp, div_self h2pi₀]
      have h2q : ∀ i ∈ s, 0 ≤ q i :=
        fun i hi ↦ div_nonneg (h2p i <| mem_insert_of_mem hi) hpi₀
      calc ∫⁻ a, ∏ i ∈ insert i₀ s, f i a ^ p i ∂μ
          = ∫⁻ a, f i₀ a ^ p i₀ * ∏ i ∈ s, f i a ^ p i ∂μ := by simp [hi₀]
        _ = ∫⁻ a, f i₀ a ^ p i₀ * (∏ i ∈ s, f i a ^ q i) ^ (1 - p i₀) ∂μ := by
            simp [q, ← ENNReal.prod_rpow_of_nonneg hpi₀, ← ENNReal.rpow_mul,
              div_mul_cancel₀ (h := h2pi₀)]
        _ ≤ (∫⁻ a, f i₀ a ∂μ) ^ p i₀ * (∫⁻ a, ∏ i ∈ s, f i a ^ q i ∂μ) ^ (1 - p i₀) := by
            apply ENNReal.lintegral_mul_norm_pow_le
            · exact hf i₀ <| mem_insert_self ..
            · exact s.aemeasurable_prod fun i hi ↦ (hf i <| mem_insert_of_mem hi).pow_const _
            · exact h2p i₀ <| mem_insert_self ..
            · exact hpi₀
            · apply add_sub_cancel
        _ ≤ (∫⁻ a, f i₀ a ∂μ) ^ p i₀ * (∏ i ∈ s, (∫⁻ a, f i a ∂μ) ^ q i) ^ (1 - p i₀) := by
            gcongr -- behavior of gcongr is heartbeat-dependent, which makes code really fragile...
            exact ih (fun i hi ↦ hf i <| mem_insert_of_mem hi) hq h2q
        _ = (∫⁻ a, f i₀ a ∂μ) ^ p i₀ * ∏ i ∈ s, (∫⁻ a, f i a ∂μ) ^ p i := by
            simp [q, ← ENNReal.prod_rpow_of_nonneg hpi₀, ← ENNReal.rpow_mul,
              div_mul_cancel₀ (h := h2pi₀)]
        _ = ∏ i ∈ insert i₀ s, (∫⁻ a, f i a ∂μ) ^ p i := by simp [hi₀]


/-- A version of Hölder with multiple arguments, one of which plays a distinguished role. -/
theorem lintegral_mul_prod_norm_pow_le {α ι : Type*} [MeasurableSpace α] {μ : Measure α}
    (s : Finset ι) {g : α →  ℝ≥0∞} {f : ι → α → ℝ≥0∞} (hg : AEMeasurable g μ)
    (hf : ∀ i ∈ s, AEMeasurable (f i) μ) (q : ℝ) {p : ι → ℝ} (hpq : q + ∑ i ∈ s, p i = 1)
    (hq :  0 ≤ q) (hp : ∀ i ∈ s, 0 ≤ p i) :
    ∫⁻ a, g a ^ q * ∏ i ∈ s, f i a ^ p i ∂μ ≤
      (∫⁻ a, g a ∂μ) ^ q * ∏ i ∈ s, (∫⁻ a, f i a ∂μ) ^ p i := by
  suffices
    ∫⁻ t, ∏ j ∈ insertNone s, Option.elim j (g t) (fun j ↦ f j t) ^ Option.elim j q p ∂μ
    ≤ ∏ j ∈ insertNone s, (∫⁻ t, Option.elim j (g t) (fun j ↦ f j t) ∂μ) ^ Option.elim j q p by
    simpa using this
  /-
    α : Type u_2
    ι : Type u_3
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    g : α → ENNReal
    f : ι → α → ENNReal
    hg : AEMeasurable g μ
    hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
    q : Real
    p : ι → Real
    hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
    hq : LE.le 0 q
    hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
    ⊢ LE.le (MeasureTheory.lintegral μ fun t => (Finset.insertNone s).prod fun j = …
  -/
  refine ENNReal.lintegral_prod_norm_pow_le _ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_2
      ι : Type u_3
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Finset ι
      g : α → ENNReal
      f : ι → α → ENNReal
      hg : AEMeasurable g μ
      hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
      q : Real
      p : ι → Real
      hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
      hq : LE.le 0 q
      hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
      ⊢ ∀ (i : Option ι), Membership.mem (Finset.insertNone s) i → AEMeasurable (fun …
    -/
  · rintro (_|i) hi
      /-
        case refine_1.none
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        hi : Membership.mem (Finset.insertNone s) Option.none
        ⊢ AEMeasurable (fun t => Option.none.elim (g t) fun j => f j t) μ
      -/
    · exact hg
      /-
        🎉 no goals
      -/
      /-
        case refine_1.some
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        i : ι
        hi : Membership.mem (Finset.insertNone s) (Option.some i)
        ⊢ AEMeasurable (fun t => (Option.some i).elim (g t) fun j => f j t) μ
      -/
    · refine hf i ?_
      /-
        case refine_1.some
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        i : ι
        hi : Membership.mem (Finset.insertNone s) (Option.some i)
        ⊢ Membership.mem s i
      -/
      simpa using hi
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      ι : Type u_3
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Finset ι
      g : α → ENNReal
      f : ι → α → ENNReal
      hg : AEMeasurable g μ
      hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
      q : Real
      p : ι → Real
      hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
      hq : LE.le 0 q
      hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
      ⊢ Eq ((Finset.insertNone s).sum fun i => i.elim q p) 1
    -/
  · simp_rw [sum_insertNone, Option.elim]
    /-
      case refine_2
      α : Type u_2
      ι : Type u_3
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Finset ι
      g : α → ENNReal
      f : ι → α → ENNReal
      hg : AEMeasurable g μ
      hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
      q : Real
      p : ι → Real
      hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
      hq : LE.le 0 q
      hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
      ⊢ Eq (HAdd.hAdd q (s.sum fun x => p x)) 1
    -/
    exact hpq
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      ι : Type u_3
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Finset ι
      g : α → ENNReal
      f : ι → α → ENNReal
      hg : AEMeasurable g μ
      hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
      q : Real
      p : ι → Real
      hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
      hq : LE.le 0 q
      hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
      ⊢ ∀ (i : Option ι), Membership.mem (Finset.insertNone s) i → LE.le 0 (i.elim q …
    -/
  · rintro (_|i) hi
      /-
        case refine_3.none
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        hi : Membership.mem (Finset.insertNone s) Option.none
        ⊢ LE.le 0 (Option.none.elim q p)
      -/
    · exact hq
      /-
        🎉 no goals
      -/
      /-
        case refine_3.some
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        i : ι
        hi : Membership.mem (Finset.insertNone s) (Option.some i)
        ⊢ LE.le 0 ((Option.some i).elim q p)
      -/
    · refine hp i ?_
      /-
        case refine_3.some
        α : Type u_2
        ι : Type u_3
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Finset ι
        g : α → ENNReal
        f : ι → α → ENNReal
        hg : AEMeasurable g μ
        hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
        q : Real
        p : ι → Real
        hpq : Eq (HAdd.hAdd q (s.sum fun i => p i)) 1
        hq : LE.le 0 q
        hp : ∀ (i : ι), Membership.mem s i → LE.le 0 (p i)
        i : ι
        hi : Membership.mem (Finset.insertNone s) (Option.some i)
        ⊢ Membership.mem s i
      -/
      simpa using hi
      /-
        🎉 no goals
      -/


theorem lintegral_rpow_add_lt_top_of_lintegral_rpow_lt_top {p : ℝ} {f g : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_top : (∫⁻ a, f a ^ p ∂μ) < ⊤) (hg_top : (∫⁻ a, g a ^ p ∂μ) < ⊤)
    (hp1 : 1 ≤ p) : (∫⁻ a, (f + g) a ^ p ∂μ) < ⊤ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_top : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    hp1 : LE.le 1 p
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) Top …
  -/
  have hp0_lt : 0 < p := lt_of_lt_of_le zero_lt_one hp1
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg_top : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    hp1 : LE.le 1 p
    hp0_lt : LT.lt 0 p
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) Top …
  -/
  have hp0 : 0 ≤ p := le_of_lt hp0_lt
  calc
    (∫⁻ a : α, (f a + g a) ^ p ∂μ) ≤
        ∫⁻ a, (2 : ℝ≥0∞) ^ (p - 1) * f a ^ p + (2 : ℝ≥0∞) ^ (p - 1) * g a ^ p ∂μ := by
      refine lintegral_mono fun a => ?_
      dsimp only
      have h_zero_lt_half_rpow : (0 : ℝ≥0∞) < (1 / 2 : ℝ≥0∞) ^ p := by
        rw [← ENNReal.zero_rpow_of_pos hp0_lt]
        exact ENNReal.rpow_lt_rpow (by simp [zero_lt_one]) hp0_lt
      have h_rw : (1 / 2 : ℝ≥0∞) ^ p * (2 : ℝ≥0∞) ^ (p - 1) = 1 / 2 := by
        rw [sub_eq_add_neg, ENNReal.rpow_add _ _ two_ne_zero ENNReal.coe_ne_top, ← mul_assoc, ←
          ENNReal.mul_rpow_of_nonneg _ _ hp0, one_div,
          ENNReal.inv_mul_cancel two_ne_zero ENNReal.coe_ne_top, ENNReal.one_rpow, one_mul,
          ENNReal.rpow_neg_one]
      rw [← ENNReal.mul_le_mul_left (ne_of_lt h_zero_lt_half_rpow).symm _]
      · rw [mul_add, ← mul_assoc, ← mul_assoc, h_rw, ← ENNReal.mul_rpow_of_nonneg _ _ hp0, mul_add]
        refine
          ENNReal.rpow_arith_mean_le_arith_mean2_rpow (1 / 2 : ℝ≥0∞) (1 / 2 : ℝ≥0∞) (f a) (g a) ?_
            hp1
        rw [ENNReal.div_add_div_same, one_add_one_eq_two,
          ENNReal.div_self two_ne_zero ENNReal.coe_ne_top]
      · rw [← lt_top_iff_ne_top]
        refine ENNReal.rpow_lt_top_of_nonneg hp0 ?_
        rw [one_div, ENNReal.inv_ne_top]
        exact two_ne_zero
    _ < ⊤ := by
      have h_two : (2 : ℝ≥0∞) ^ (p - 1) ≠ ⊤ :=
        ENNReal.rpow_ne_top_of_nonneg (by simp [hp1]) ENNReal.coe_ne_top
      rw [lintegral_add_left', lintegral_const_mul'' _ (hf.pow_const p),
        lintegral_const_mul' _ _ h_two, ENNReal.add_lt_top]
      · exact ⟨ENNReal.mul_lt_top h_two.lt_top hf_top, ENNReal.mul_lt_top h_two.lt_top hg_top⟩
      · exact (hf.pow_const p).const_mul _


theorem lintegral_Lp_mul_le_Lq_mul_Lr {α} [MeasurableSpace α] {p q r : ℝ} (hp0_lt : 0 < p)
    (hpq : p < q) (hpqr : 1 / p = 1 / q + 1 / r) (μ : Measure α) {f g : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    (∫⁻ a, (f * g) a ^ p ∂μ) ^ (1 / p) ≤
      (∫⁻ a, f a ^ q ∂μ) ^ (1 / q) * (∫⁻ a, g a ^ r ∂μ) ^ (1 / r) := by
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  have hp0_ne : p ≠ 0 := (ne_of_lt hp0_lt).symm
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  have hp0 : 0 ≤ p := le_of_lt hp0_lt
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    hp0 : LE.le 0 p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  have hq0_lt : 0 < q := lt_of_le_of_lt hp0 hpq
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    hp0 : LE.le 0 p
    hq0_lt : LT.lt 0 q
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  have hq0_ne : q ≠ 0 := (ne_of_lt hq0_lt).symm
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    hp0 : LE.le 0 p
    hq0_lt : LT.lt 0 q
    hq0_ne : Ne q 0
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  have h_one_div_r : 1 / r = 1 / p - 1 / q := by rw [hpqr]; simp
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    hp0 : LE.le 0 p
    hq0_lt : LT.lt 0 q
    hq0_ne : Ne q 0
    h_one_div_r : Eq (HDiv.hDiv 1 r) (HSub.hSub (HDiv.hDiv 1 p) (HDiv.hDiv 1 q))
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  let p2 := q / p
  /-
    α : Type u_2
    inst✝ : MeasurableSpace α
    p q r : Real
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp0_ne : Ne p 0
    hp0 : LE.le 0 p
    hq0_lt : LT.lt 0 q
    hq0_ne : Ne q 0
    h_one_div_r : Eq (HDiv.hDiv 1 r) (HSub.hSub (HDiv.hDiv 1 p) (HDiv.hDiv 1 q))
    p2 : Real := HDiv.hDiv q p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul f  …
  -/
  let q2 := p2.conjExponent
  have hp2q2 : p2.IsConjExponent q2 :=
    .conjExponent (by simp [p2, q2, _root_.lt_div_iff₀, hpq, hp0_lt])
  calc
    (∫⁻ a : α, (f * g) a ^ p ∂μ) ^ (1 / p) = (∫⁻ a : α, f a ^ p * g a ^ p ∂μ) ^ (1 / p) := by
      simp_rw [Pi.mul_apply, ENNReal.mul_rpow_of_nonneg _ _ hp0]
    _ ≤ ((∫⁻ a, f a ^ (p * p2) ∂μ) ^ (1 / p2) *
        (∫⁻ a, g a ^ (p * q2) ∂μ) ^ (1 / q2)) ^ (1 / p) := by
      gcongr
      simp_rw [ENNReal.rpow_mul]
      exact ENNReal.lintegral_mul_le_Lp_mul_Lq μ hp2q2 (hf.pow_const _) (hg.pow_const _)
    _ = (∫⁻ a : α, f a ^ q ∂μ) ^ (1 / q) * (∫⁻ a : α, g a ^ r ∂μ) ^ (1 / r) := by
      rw [@ENNReal.mul_rpow_of_nonneg _ _ (1 / p) (by simp [hp0]), ← ENNReal.rpow_mul, ←
        ENNReal.rpow_mul]
      have hpp2 : p * p2 = q := by
        symm
        rw [mul_comm, ← div_eq_iff hp0_ne]
      have hpq2 : p * q2 = r := by
        rw [← inv_inv r, ← one_div, ← one_div, h_one_div_r]
        field_simp [p2, q2, Real.conjExponent, hp0_ne, hq0_ne]
      simp_rw [div_mul_div_comm, mul_one, mul_comm p2, mul_comm q2, hpp2, hpq2]


theorem lintegral_mul_rpow_le_lintegral_rpow_mul_lintegral_rpow {p q : ℝ}
    (hpq : p.IsConjExponent q) {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ)
    (hf_top : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤) :
    (∫⁻ a, f a * g a ^ (p - 1) ∂μ) ≤
      (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) * (∫⁻ a, g a ^ p ∂μ) ^ (1 / q) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (f a) (HPow.hPow (g a) ( …
  -/
  refine le_trans (ENNReal.lintegral_mul_le_Lp_mul_Lq μ hpq hf (hg.pow_const _)) ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    ⊢ LE.le (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f …
  -/
  by_cases hf_zero_rpow : (∫⁻ a : α, f a ^ p ∂μ) ^ (1 / p) = 0
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
      hf_zero_rpow : Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f  …
      ⊢ LE.le (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f …
    -/
  · rw [hf_zero_rpow, zero_mul]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Real
      hpq : p.IsConjExponent q
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
      hf_zero_rpow : Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f  …
      ⊢ LE.le 0 (HMul.hMul 0 (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPo …
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
  have hf_top_rpow : (∫⁻ a : α, f a ^ p ∂μ) ^ (1 / p) ≠ ⊤ := by
    by_contra h
    refine hf_top ?_
    have hp_not_neg : ¬p < 0 := by simp [hpq.nonneg]
    simpa [hpq.pos, hp_not_neg] using h
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hf_zero_rpow : Not (Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPo …
    hf_top_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f a …
    ⊢ LE.le (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f …
  -/
  refine (ENNReal.mul_le_mul_left hf_zero_rpow hf_top_rpow).mpr (le_of_eq ?_)
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hf_zero_rpow : Not (Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPo …
    hf_top_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f a …
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HPow.hPow (g a) …
  -/
  congr
  /-
    case neg.e_a.e_f
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hf_zero_rpow : Not (Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPo …
    hf_top_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f a …
    ⊢ Eq (fun a => HPow.hPow (HPow.hPow (g a) (HSub.hSub p 1)) q) fun a => HPow.hP …
  -/
  ext1 a
  /-
    case neg.e_a.e_f.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hf_zero_rpow : Not (Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPo …
    hf_top_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (f a …
    a : α
    ⊢ Eq (HPow.hPow (HPow.hPow (g a) (HSub.hSub p 1)) q) (HPow.hPow (g a) p)
  -/
  rw [← ENNReal.rpow_mul, hpq.sub_one_mul_conj]
  /-
    🎉 no goals
  -/


theorem lintegral_rpow_add_le_add_eLpNorm_mul_lintegral_rpow_add {p q : ℝ}
    (hpq : p.IsConjExponent q) {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hf_top : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤) (hg : AEMeasurable g μ) (hg_top : (∫⁻ a, g a ^ p ∂μ) ≠ ⊤) :
    (∫⁻ a, (f + g) a ^ p ∂μ) ≤
      ((∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p)) *
        (∫⁻ a, (f a + g a) ^ p ∂μ) ^ (1 / q) := by
  calc
    (∫⁻ a, (f + g) a ^ p ∂μ) ≤ ∫⁻ a, (f + g) a * (f + g) a ^ (p - 1) ∂μ := by
      gcongr with a
      by_cases h_zero : (f + g) a = 0
      · rw [h_zero, ENNReal.zero_rpow_of_pos hpq.pos]
        exact zero_le _
      by_cases h_top : (f + g) a = ⊤
      · rw [h_top, ENNReal.top_rpow_of_pos hpq.sub_one_pos, ENNReal.top_mul_top]
        exact le_top
      refine le_of_eq ?_
      nth_rw 2 [← ENNReal.rpow_one ((f + g) a)]
      rw [← ENNReal.rpow_add _ _ h_zero h_top, add_sub_cancel]
    _ = (∫⁻ a : α, f a * (f + g) a ^ (p - 1) ∂μ) + ∫⁻ a : α, g a * (f + g) a ^ (p - 1) ∂μ := by
      have h_add_m : AEMeasurable (fun a : α => (f + g) a ^ (p - 1 : ℝ)) μ :=
        (hf.add hg).pow_const _
      have h_add_apply :
        (∫⁻ a : α, (f + g) a * (f + g) a ^ (p - 1) ∂μ) =
          ∫⁻ a : α, (f a + g a) * (f + g) a ^ (p - 1) ∂μ :=
        rfl
      simp_rw [h_add_apply, add_mul]
      rw [lintegral_add_left' (hf.mul h_add_m)]
    _ ≤
        ((∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p)) *
          (∫⁻ a, (f a + g a) ^ p ∂μ) ^ (1 / q) := by
      rw [add_mul]
      gcongr
      · exact lintegral_mul_rpow_le_lintegral_rpow_mul_lintegral_rpow hpq hf (hf.add hg) hf_top
      · exact lintegral_mul_rpow_le_lintegral_rpow_mul_lintegral_rpow hpq hg (hf.add hg) hg_top


@[deprecated (since := "2024-07-27")]
alias lintegral_rpow_add_le_add_snorm_mul_lintegral_rpow_add :=
  lintegral_rpow_add_le_add_eLpNorm_mul_lintegral_rpow_add


private theorem lintegral_Lp_add_le_aux {p q : ℝ} (hpq : p.IsConjExponent q) {f g : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hf_top : (∫⁻ a, f a ^ p ∂μ) ≠ ⊤) (hg : AEMeasurable g μ)
    (hg_top : (∫⁻ a, g a ^ p ∂μ) ≠ ⊤) (h_add_zero : (∫⁻ a, (f + g) a ^ p ∂μ) ≠ 0)
    (h_add_top : (∫⁻ a, (f + g) a ^ p ∂μ) ≠ ⊤) :
    (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≤
      (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg : AEMeasurable g μ
    hg_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    h_add_zero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a …
    h_add_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  have hp_not_nonpos : ¬p ≤ 0 := by simp [hpq.pos]
  have htop_rpow : (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≠ ⊤ := by
    by_contra h
    exact h_add_top (@ENNReal.rpow_eq_top_of_nonneg _ (1 / p) (by simp [hpq.nonneg]) h)
  have h0_rpow : (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≠ 0 := by
    simp [h_add_zero, h_add_top, hpq.nonneg, hp_not_nonpos, -Pi.add_apply]
  suffices h :
    1 ≤
      (∫⁻ a : α, (f + g) a ^ p ∂μ) ^ (-(1 / p)) *
        ((∫⁻ a : α, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a : α, g a ^ p ∂μ) ^ (1 / p)) by
    rwa [← mul_le_mul_left h0_rpow htop_rpow, ← mul_assoc, ← rpow_add _ _ h_add_zero h_add_top, ←
      sub_eq_add_neg, _root_.sub_self, rpow_zero, one_mul, mul_one] at h
  have h :
    (∫⁻ a : α, (f + g) a ^ p ∂μ) ≤
      ((∫⁻ a : α, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a : α, g a ^ p ∂μ) ^ (1 / p)) *
        (∫⁻ a : α, (f + g) a ^ p ∂μ) ^ (1 / q) :=
    lintegral_rpow_add_le_add_eLpNorm_mul_lintegral_rpow_add hpq hf hf_top hg hg_top
  have h_one_div_q : 1 / q = 1 - 1 / p := by
    nth_rw 2 [← hpq.inv_add_inv_conj]
    ring
  simp_rw [h_one_div_q, sub_eq_add_neg 1 (1 / p), ENNReal.rpow_add _ _ h_add_zero h_add_top,
    rpow_one] at h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg : AEMeasurable g μ
    hg_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    h_add_zero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a …
    h_add_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) …
    hp_not_nonpos : Not (LE.le p 0)
    htop_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd. …
    h0_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hA …
    h_one_div_q : Eq (HDiv.hDiv 1 q) (HSub.hSub 1 (HDiv.hDiv 1 p))
    h : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) ( …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow  …
  -/
  conv_rhs at h => enter [2]; rw [mul_comm]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg : AEMeasurable g μ
    hg_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    h_add_zero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a …
    h_add_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) …
    hp_not_nonpos : Not (LE.le p 0)
    htop_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd. …
    h0_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hA …
    h_one_div_q : Eq (HDiv.hDiv 1 q) (HSub.hSub 1 (HDiv.hDiv 1 p))
    h : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) ( …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow  …
  -/
  conv_lhs at h => rw [← one_mul (∫⁻ a : α, (f + g) a ^ p ∂μ)]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hf_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
    hg : AEMeasurable g μ
    hg_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
    h_add_zero : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a …
    h_add_top : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) …
    hp_not_nonpos : Not (LE.le p 0)
    htop_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd. …
    h0_rpow : Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hA …
    h_one_div_q : Eq (HDiv.hDiv 1 q) (HSub.hSub 1 (HDiv.hDiv 1 p))
    h : LE.le (HMul.hMul 1 (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAd …
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow  …
  -/
  rwa [← mul_assoc, ENNReal.mul_le_mul_right h_add_zero h_add_top, mul_comm] at h
  /-
    🎉 no goals
  -/


/-- **Minkowski's inequality for functions** `α → ℝ≥0∞`: the `ℒp` seminorm of the sum of two
functions is bounded by the sum of their `ℒp` seminorms. -/
theorem lintegral_Lp_add_le {p : ℝ} {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ)
    (hp1 : 1 ≤ p) :
    (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≤
      (∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  have hp_pos : 0 < p := lt_of_lt_of_le zero_lt_one hp1
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  by_cases hf_top : ∫⁻ a, f a ^ p ∂μ = ⊤
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top
      ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
    -/
  · simp [hf_top, hp_pos]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  by_cases hg_top : ∫⁻ a, g a ^ p ∂μ = ⊤
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top
      ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
    -/
  · simp [hg_top, hp_pos]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  by_cases h1 : p = 1
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
      h1 : Eq p 1
      ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
    -/
  · refine le_of_eq ?_
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
      h1 : Eq p 1
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a …
    -/
    simp_rw [h1, one_div_one, ENNReal.rpow_one]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
      h1 : Eq p 1
      ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd f g a) (HAdd.hAdd (MeasureT …
    -/
    exact lintegral_add_left' hf _
    /-
      🎉 no goals
    -/
  have hp1_lt : 1 < p := by
    refine lt_of_le_of_ne hp1 ?_
    symm
    exact h1
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
    h1 : Not (Eq p 1)
    hp1_lt : LT.lt 1 p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  have hpq := Real.IsConjExponent.conjExponent hp1_lt
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
    h1 : Not (Eq p 1)
    hp1_lt : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  by_cases h0 : (∫⁻ a, (f + g) a ^ p ∂μ) = 0
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
      h1 : Not (Eq p 1)
      hp1_lt : LT.lt 1 p
      hpq : p.IsConjExponent p.conjExponent
      h0 : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) 0
      ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
    -/
  · rw [h0, @ENNReal.zero_rpow_of_pos (1 / p) (by simp [lt_of_lt_of_le zero_lt_one hp1])]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : Real
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
      hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
      h1 : Not (Eq p 1)
      hp1_lt : LT.lt 1 p
      hpq : p.IsConjExponent p.conjExponent
      h0 : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) 0
      ⊢ LE.le 0 (HAdd.hAdd (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow  …
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
  have htop : (∫⁻ a, (f + g) a ^ p ∂μ) ≠ ⊤ := by
    rw [← Ne] at hf_top hg_top
    rw [← lt_top_iff_ne_top] at hf_top hg_top ⊢
    exact lintegral_rpow_add_lt_top_of_lintegral_rpow_lt_top hf hf_top hg_top hp1
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hf_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (f a) p) Top.top)
    hg_top : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (g a) p) Top.top)
    h1 : Not (Eq p 1)
    hp1_lt : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    h0 : Not (Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p …
    htop : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f g a) p) T …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  exact lintegral_Lp_add_le_aux hpq hf hf_top hg hg_top h0 htop
  /-
    🎉 no goals
  -/


/-- Variant of Minkowski's inequality for functions `α → ℝ≥0∞` in `ℒp` with `p ≤ 1`: the `ℒp`
seminorm of the sum of two functions is bounded by a constant multiple of the sum
of their `ℒp` seminorms. -/
theorem lintegral_Lp_add_le_of_le_one {p : ℝ} {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hp0 : 0 ≤ p)
    (hp1 : p ≤ 1) :
    (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≤
      (2 : ℝ≥0∞) ^ (1 / p - 1) * ((∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p)) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Real
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hp0 : LE.le 0 p
    hp1 : LE.le p 1
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
  -/
  rcases eq_or_lt_of_le hp0 with (rfl | hp)
    /-
      case inl
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hp0 : LE.le 0 0
      hp1 : LE.le 0 1
      ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HAdd.hAdd f  …
    -/
  · simp only [Pi.add_apply, rpow_zero, lintegral_one, _root_.div_zero, zero_sub]
    /-
      case inl
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hp0 : LE.le 0 0
      hp1 : LE.le 0 1
      ⊢ LE.le 1 (HMul.hMul (HPow.hPow 2 (-1)) (HAdd.hAdd 1 1))
    -/
    norm_num
    /-
      case inl
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hp0 : LE.le 0 0
      hp1 : LE.le 0 1
      ⊢ LE.le 1 (HMul.hMul (HPow.hPow 2 (-1)) 2)
    -/
    rw [rpow_neg, rpow_one, ENNReal.inv_mul_cancel two_ne_zero two_ne_top]
    /-
      🎉 no goals
    -/
  calc
    (∫⁻ a, (f + g) a ^ p ∂μ) ^ (1 / p) ≤ ((∫⁻ a, f a ^ p ∂μ) + ∫⁻ a, g a ^ p ∂μ) ^ (1 / p) := by
      rw [← lintegral_add_left' (hf.pow_const p)]
      gcongr with a
      exact rpow_add_le_add_rpow _ _ hp0 hp1
    _ ≤ (2 : ℝ≥0∞) ^ (1 / p - 1) * ((∫⁻ a, f a ^ p ∂μ) ^ (1 / p) + (∫⁻ a, g a ^ p ∂μ) ^ (1 / p)) :=
      rpow_add_le_mul_rpow_add_rpow _ _ ((one_le_div hp).2 hp1)


/-- Hölder's inequality for functions `α → ℝ≥0`. The integral of the product of two functions
is bounded by the product of their `ℒp` and `ℒq` seminorms when `p` and `q` are conjugate
exponents. -/
theorem NNReal.lintegral_mul_le_Lp_mul_Lq {p q : ℝ} (hpq : p.IsConjExponent q) {f g : α → ℝ≥0}
    (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    (∫⁻ a, (f * g) a ∂μ) ≤
      (∫⁻ a, (f a : ℝ≥0∞) ^ p ∂μ) ^ (1 / p) * (∫⁻ a, (g a : ℝ≥0∞) ^ q ∂μ) ^ (1 / q) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → NNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(HMul.hMul f g a)) (HMul.hMul (HP …
  -/
  simp_rw [Pi.mul_apply, ENNReal.coe_mul]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → NNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul ↑(f a) ↑(g a)) (HMul.hMu …
  -/
  exact ENNReal.lintegral_mul_le_Lp_mul_Lq μ hpq hf.coe_nnreal_ennreal hg.coe_nnreal_ennreal
  /-
    🎉 no goals
  -/


