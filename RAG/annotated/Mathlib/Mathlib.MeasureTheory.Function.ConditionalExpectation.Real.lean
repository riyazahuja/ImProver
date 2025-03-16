theorem rnDeriv_ae_eq_condexp {hm : m ≤ m0} [hμm : SigmaFinite (μ.trim hm)] {f : α → ℝ}
    (hf : Integrable f μ) :
    SignedMeasure.rnDeriv ((μ.withDensityᵥ f).trim hm) (μ.trim hm) =ᵐ[μ] μ[f|m] := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.SignedMeasure.rnDeriv ((μ.w …
  -/
  refine ae_eq_condexp_of_forall_setIntegral_eq hm hf ?_ ?_ ?_
  · exact fun _ _ _ => (integrable_of_integrable_trim hm
      (SignedMeasure.integrable_rnDeriv ((μ.withDensityᵥ f).trim hm) (μ.trim hm))).integrableOn
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs _
    conv_rhs => rw [← hf.withDensityᵥ_trim_eq_integral hm hs,
      ← SignedMeasure.withDensityᵥ_rnDeriv_eq ((μ.withDensityᵥ f).trim hm) (μ.trim hm)
        (hf.withDensityᵥ_trim_absolutelyContinuous hm)]
    rw [withDensityᵥ_apply
      (SignedMeasure.integrable_rnDeriv ((μ.withDensityᵥ f).trim hm) (μ.trim hm)) hs,
      ← setIntegral_trim hm _ hs]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      s : Set α
      hs : MeasurableSet s
      a✝ : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.SignedMeasure.rnDeriv ((μ.wi …
    -/
    exact (SignedMeasure.measurable_rnDeriv _ _).stronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      hμm : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m (MeasureTheory.SignedMeasure.rnDeriv ( …
    -/
  · exact (SignedMeasure.measurable_rnDeriv _ _).stronglyMeasurable.aeStronglyMeasurable'
    /-
      🎉 no goals
    -/

-- TODO: the following couple of lemmas should be generalized and proved using Jensen's inequality
-- for the conditional expectation (not in mathlib yet) .

theorem eLpNorm_one_condexp_le_eLpNorm (f : α → ℝ) : eLpNorm (μ[f|m]) 1 μ ≤ eLpNorm f 1 μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  by_cases hf : Integrable f μ
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  swap; · rw [condexp_undef hf, eLpNorm_zero]; exact zero_le _
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hm : LE.le m m0
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  swap; · rw [condexp_of_not_le hm, eLpNorm_zero]; exact zero_le _
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hm : LE.le m m0
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  by_cases hsig : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hm : LE.le m m0
    hsig : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
  -/
  swap; · rw [condexp_of_not_sigmaFinite hm hsig, eLpNorm_zero]; exact zero_le _
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  calc
    eLpNorm (μ[f|m]) 1 μ ≤ eLpNorm (μ[(|f|)|m]) 1 μ := by
      refine eLpNorm_mono_ae ?_
      filter_upwards [condexp_mono hf hf.abs
        (ae_of_all μ (fun x => le_abs_self (f x) : ∀ x, f x ≤ |f x|)),
        EventuallyLE.trans (condexp_neg f).symm.le
          (condexp_mono hf.neg hf.abs
          (ae_of_all μ (fun x => neg_le_abs (f x) : ∀ x, -f x ≤ |f x|)))] with x hx₁ hx₂
      exact abs_le_abs hx₁ hx₂
    _ = eLpNorm f 1 μ := by
      rw [eLpNorm_one_eq_lintegral_nnnorm, eLpNorm_one_eq_lintegral_nnnorm,
        ← ENNReal.toReal_eq_toReal (hasFiniteIntegral_iff_nnnorm.mp integrable_condexp.2).ne
          (hasFiniteIntegral_iff_nnnorm.mp hf.2).ne,
        ← integral_norm_eq_lintegral_nnnorm
          (stronglyMeasurable_condexp.mono hm).aestronglyMeasurable,
        ← integral_norm_eq_lintegral_nnnorm hf.1]
      simp_rw [Real.norm_eq_abs]
      rw (config := {occs := .pos [2]}) [← integral_condexp hm]
      refine integral_congr_ae ?_
      have : 0 ≤ᵐ[μ] μ[(|f|)|m] := by
        rw [← condexp_zero]
        exact condexp_mono (integrable_zero _ _ _) hf.abs
          (ae_of_all μ (fun x => abs_nonneg (f x) : ∀ x, 0 ≤ |f x|))
      filter_upwards [this] with x hx
      exact abs_eq_self.2 hx


@[deprecated (since := "2024-07-27")]
alias snorm_one_condexp_le_snorm := eLpNorm_one_condexp_le_eLpNorm


theorem integral_abs_condexp_le (f : α → ℝ) : ∫ x, |(μ[f|m]) x| ∂μ ≤ ∫ x, |f x| ∂μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
  -/
  by_cases hm : m ≤ m0
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hm : LE.le m m0
    ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
  -/
  swap
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : Not (LE.le m m0)
      ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
    -/
  · simp_rw [condexp_of_not_le hm, Pi.zero_apply, abs_zero, integral_zero]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : Not (LE.le m m0)
      ⊢ LE.le 0 (MeasureTheory.integral μ fun x => abs (f x))
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hm : LE.le m m0
    ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
  -/
  by_cases hfint : Integrable f μ
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
  -/
  swap
  · simp only [condexp_undef hfint, Pi.zero_apply, abs_zero, integral_const, Algebra.id.smul_eq_mul,
      mul_zero]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ LE.le 0 (MeasureTheory.integral μ fun x => abs (f x))
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.integral μ fun x => abs (MeasureTheory.condexp m μ f x) …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae, integral_eq_lintegral_of_nonneg_ae]
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (abs (MeasureTheory …
    -/
  · apply ENNReal.toReal_mono <;> simp_rw [← Real.norm_eq_abs, ofReal_norm_eq_coe_nnnorm]
      /-
        case pos.hb
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hm : LE.le m m0
        hfint : MeasureTheory.Integrable f μ
        ⊢ Ne (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f a))) Top.top
      -/
    · exact hfint.2.ne
      /-
        🎉 no goals
      -/
      /-
        case pos.h
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hm : LE.le m m0
        hfint : MeasureTheory.Integrable f μ
        ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (MeasureTheory.con …
      -/
    · rw [← eLpNorm_one_eq_lintegral_nnnorm, ← eLpNorm_one_eq_lintegral_nnnorm]
      /-
        case pos.h
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hm : LE.le m m0
        hfint : MeasureTheory.Integrable f μ
        ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp m μ f) 1 μ) (MeasureTheo …
      -/
      exact eLpNorm_one_condexp_le_eLpNorm _
      /-
        🎉 no goals
      -/
    /-
      case pos.hf
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => abs (f x)
    -/
  · filter_upwards with x using abs_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case pos.hfm
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => abs (f x)) μ
    -/
  · simp_rw [← Real.norm_eq_abs]
    /-
      case pos.hfm
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => Norm.norm (f x)) μ
    -/
    exact hfint.1.norm
    /-
      🎉 no goals
    -/
    /-
      case pos.hf
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => abs (MeasureTheory.condexp m μ  …
    -/
  · filter_upwards with x using abs_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case pos.hfm
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => abs (MeasureTheory.condexp m μ  …
    -/
  · simp_rw [← Real.norm_eq_abs]
    /-
      case pos.hfm
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => Norm.norm (MeasureTheory.condex …
    -/
    exact (stronglyMeasurable_condexp.mono hm).aestronglyMeasurable.norm
    /-
      🎉 no goals
    -/


theorem setIntegral_abs_condexp_le {s : Set α} (hs : MeasurableSet[m] s) (f : α → ℝ) :
    ∫ x in s, |(μ[f|m]) x| ∂μ ≤ ∫ x in s, |f x| ∂μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
  -/
  by_cases hnm : m ≤ m0
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    hnm : LE.le m m0
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
  -/
  swap
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      f : α → Real
      hnm : Not (LE.le m m0)
      ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
    -/
  · simp_rw [condexp_of_not_le hnm, Pi.zero_apply, abs_zero, integral_zero]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      f : α → Real
      hnm : Not (LE.le m m0)
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x => abs (f x))
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    hnm : LE.le m m0
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
  -/
  by_cases hfint : Integrable f μ
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
  -/
  swap
  · simp only [condexp_undef hfint, Pi.zero_apply, abs_zero, integral_const, Algebra.id.smul_eq_mul,
      mul_zero]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      f : α → Real
      hnm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x => abs (f x))
    -/
    positivity
    /-
      🎉 no goals
    -/
  have : ∫ x in s, |(μ[f|m]) x| ∂μ = ∫ x, |(μ[s.indicator f|m]) x| ∂μ := by
    rw [← integral_indicator (hnm _ hs)]
    refine integral_congr_ae ?_
    have : (fun x => |(μ[s.indicator f|m]) x|) =ᵐ[μ] fun x => |s.indicator (μ[f|m]) x| :=
      (condexp_indicator hfint hs).fun_comp abs
    refine EventuallyEq.trans (Eventually.of_forall fun x => ?_) this.symm
    rw [← Real.norm_eq_abs, norm_indicator_eq_indicator_norm]
    simp only [Real.norm_eq_abs]
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    this : Eq (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.c …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.con …
  -/
  rw [this, ← integral_indicator (hnm _ hs)]
  refine (integral_abs_condexp_le _).trans
    (le_of_eq <| integral_congr_ae <| Eventually.of_forall fun x => ?_)
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → Real
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    this : Eq (MeasureTheory.integral (μ.restrict s) fun x => abs (MeasureTheory.c …
    x : α
    ⊢ Eq ((fun x => abs (s.indicator f x)) x) (s.indicator (fun x => abs (f x)) x)
  -/
  simp_rw [← Real.norm_eq_abs, norm_indicator_eq_indicator_norm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_abs_condexp_le := setIntegral_abs_condexp_le


/-- If the real valued function `f` is bounded almost everywhere by `R`, then so is its conditional
expectation. -/
theorem ae_bdd_condexp_of_ae_bdd {R : ℝ≥0} {f : α → ℝ} (hbdd : ∀ᵐ x ∂μ, |f x| ≤ R) :
    ∀ᵐ x ∂μ, |(μ[f|m]) x| ≤ R := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
  -/
  by_cases hnm : m ≤ m0
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
  -/
  swap
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : Not (LE.le m m0)
      ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
    -/
  · simp_rw [condexp_of_not_le hnm, Pi.zero_apply, abs_zero]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : Not (LE.le m m0)
      ⊢ Filter.Eventually (fun x => LE.le 0 ↑R) (MeasureTheory.ae μ)
    -/
    exact Eventually.of_forall fun _ => R.coe_nonneg
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
  -/
  by_cases hfint : Integrable f μ
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
  -/
  swap
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
    -/
  · simp_rw [condexp_undef hfint]
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      ⊢ Filter.Eventually (fun x => LE.le (abs (0 x)) ↑R) (MeasureTheory.ae μ)
    -/
    filter_upwards [hbdd] with x hx
    /-
      case h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      x : α
      hx : LE.le (abs (f x)) ↑R
      ⊢ LE.le (abs (0 x)) ↑R
    -/
    rw [Pi.zero_apply, abs_zero]
    /-
      case h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : Not (MeasureTheory.Integrable f μ)
      x : α
      hx : LE.le (abs (f x)) ↑R
      ⊢ LE.le 0 ↑R
    -/
    exact (abs_nonneg _).trans hx
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    ⊢ Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x)) ↑R)  …
  -/
  by_contra h
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : Not (Filter.Eventually (fun x => LE.le (abs (MeasureTheory.condexp m μ f x …
    ⊢ False
  -/
  change μ _ ≠ 0 at h
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : Ne (μ (HasCompl.compl (setOf fun x => (fun x => LE.le (abs (MeasureTheory. …
    ⊢ False
  -/
  simp only [← zero_lt_iff, Set.compl_def, Set.mem_setOf_eq, not_le] at h
  suffices (μ {x | ↑R < |(μ[f|m]) x|}).toReal * ↑R < (μ {x | ↑R < |(μ[f|m]) x|}).toReal * ↑R by
    exact this.ne rfl
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
    ⊢ LT.lt (HMul.hMul (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m …
  -/
  refine lt_of_lt_of_le (setIntegral_gt_gt R.coe_nonneg ?_ h.ne') ?_
    /-
      case pos.refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
      ⊢ MeasureTheory.IntegrableOn (fun x => abs (MeasureTheory.condexp m μ f x)) (s …
    -/
  · exact integrable_condexp.abs.integrableOn
    /-
      🎉 no goals
    -/
  /-
    case pos.refine_2
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (setOf fun x => LT.lt (↑R) (abs (M …
  -/
  refine (setIntegral_abs_condexp_le ?_ _).trans ?_
    /-
      case pos.refine_2.refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      R : NNReal
      f : α → Real
      hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
      hnm : LE.le m m0
      hfint : MeasureTheory.Integrable f μ
      h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
      ⊢ MeasurableSet (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x …
    -/
  · simp_rw [← Real.norm_eq_abs]
    exact @measurableSet_lt _ _ _ _ _ m _ _ _ _ _ measurable_const
      stronglyMeasurable_condexp.norm.measurable
  simp only [← smul_eq_mul, ← setIntegral_const, NNReal.val_eq_coe, RCLike.ofReal_real_eq_id,
    _root_.id]
  /-
    case pos.refine_2.refine_2
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (setOf fun x => LT.lt (↑R) (abs (M …
  -/
  refine setIntegral_mono_ae hfint.abs.integrableOn ?_ hbdd
  refine ⟨aestronglyMeasurable_const, lt_of_le_of_lt ?_
    (integrable_condexp.integrableOn : IntegrableOn (μ[f|m]) {x | ↑R < |(μ[f|m]) x|} μ).2⟩
  refine setLIntegral_mono
    (stronglyMeasurable_condexp.mono hnm).measurable.nnnorm.coe_nnreal_ennreal fun x hx => ?_
  /-
    case pos.refine_2.refine_2
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
    x : α
    hx : Membership.mem (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ …
    ⊢ LE.le (ENorm.enorm ((fun x => ↑R) x)) (ENorm.enorm (MeasureTheory.condexp m  …
  -/
  rw [enorm_eq_nnnorm, enorm_eq_nnnorm, ENNReal.coe_le_coe, Real.nnnorm_of_nonneg R.coe_nonneg]
  /-
    case pos.refine_2.refine_2
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    R : NNReal
    f : α → Real
    hbdd : Filter.Eventually (fun x => LE.le (abs (f x)) ↑R) (MeasureTheory.ae μ)
    hnm : LE.le m m0
    hfint : MeasureTheory.Integrable f μ
    h : LT.lt 0 (μ (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ f x) …
    x : α
    hx : Membership.mem (setOf fun x => LT.lt (↑R) (abs (MeasureTheory.condexp m μ …
    ⊢ LE.le ⟨↑R, ⋯⟩ (NNNorm.nnnorm (MeasureTheory.condexp m μ f x))
  -/
  exact Subtype.mk_le_mk.2 (le_of_lt hx)
  /-
    🎉 no goals
  -/


/-- Given an integrable function `g`, the conditional expectations of `g` with respect to
a sequence of sub-σ-algebras is uniformly integrable. -/
theorem Integrable.uniformIntegrable_condexp {ι : Type*} [IsFiniteMeasure μ] {g : α → ℝ}
    (hint : Integrable g μ) {ℱ : ι → MeasurableSpace α} (hℱ : ∀ i, ℱ i ≤ m0) :
    UniformIntegrable (fun i => μ[g|ℱ i]) 1 μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    ⊢ MeasureTheory.UniformIntegrable (fun i => MeasureTheory.condexp (ℱ i) μ g) 1 μ
  -/
  let A : MeasurableSpace α := m0
  have hmeas : ∀ n, ∀ C, MeasurableSet {x | C ≤ ‖(μ[g|ℱ n]) x‖₊} := fun n C =>
    measurableSet_le measurable_const (stronglyMeasurable_condexp.mono (hℱ n)).measurable.nnnorm
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    ⊢ MeasureTheory.UniformIntegrable (fun i => MeasureTheory.condexp (ℱ i) μ g) 1 μ
  -/
  have hg : Memℒp g 1 μ := memℒp_one_iff_integrable.2 hint
  refine uniformIntegrable_of le_rfl ENNReal.one_ne_top
    (fun n => (stronglyMeasurable_condexp.mono (hℱ n)).aestronglyMeasurable) fun ε hε => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  by_cases hne : eLpNorm g 1 μ = 0
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      g : α → Real
      hint : MeasureTheory.Integrable g μ
      ℱ : ι → MeasurableSpace α
      hℱ : ∀ (i : ι), LE.le (ℱ i) m0
      A : MeasurableSpace α := m0
      hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
      hg : MeasureTheory.Memℒp g 1 μ
      ε : Real
      hε : LT.lt 0 ε
      hne : Eq (MeasureTheory.eLpNorm g 1 μ) 0
      ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
    -/
  · rw [eLpNorm_eq_zero_iff hg.1 one_ne_zero] at hne
    refine ⟨0, fun n => (le_of_eq <|
      (eLpNorm_eq_zero_iff ((stronglyMeasurable_condexp.mono (hℱ n)).aestronglyMeasurable.indicator
        (hmeas n 0)) one_ne_zero).2 ?_).trans (zero_le _)⟩
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      g : α → Real
      hint : MeasureTheory.Integrable g μ
      ℱ : ι → MeasurableSpace α
      hℱ : ∀ (i : ι), LE.le (ℱ i) m0
      A : MeasurableSpace α := m0
      hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
      hg : MeasureTheory.Memℒp g 1 μ
      ε : Real
      hε : LT.lt 0 ε
      hne : (MeasureTheory.ae μ).EventuallyEq g 0
      n : ι
      ⊢ (MeasureTheory.ae μ).EventuallyEq ((setOf fun x => LE.le 0 (NNNorm.nnnorm (M …
    -/
    filter_upwards [condexp_congr_ae (m := ℱ n) hne] with x hx
    /-
      case h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      g : α → Real
      hint : MeasureTheory.Integrable g μ
      ℱ : ι → MeasurableSpace α
      hℱ : ∀ (i : ι), LE.le (ℱ i) m0
      A : MeasurableSpace α := m0
      hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
      hg : MeasureTheory.Memℒp g 1 μ
      ε : Real
      hε : LT.lt 0 ε
      hne : (MeasureTheory.ae μ).EventuallyEq g 0
      n : ι
      x : α
      hx : Eq (MeasureTheory.condexp (ℱ n) μ g x) (MeasureTheory.condexp (ℱ n) μ 0 x)
      ⊢ Eq ((setOf fun x => LE.le 0 (NNNorm.nnnorm (MeasureTheory.condexp (ℱ n) μ g  …
    -/
    simp only [zero_le', Set.setOf_true, Set.indicator_univ, Pi.zero_apply, hx, condexp_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨δ, hδ, h⟩ := hg.eLpNorm_indicator_le le_rfl ENNReal.one_ne_top hε
  /-
    case neg.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le (M …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  set C : ℝ≥0 := ⟨δ, hδ.le⟩⁻¹ * (eLpNorm g 1 μ).toNNReal with hC
  /-
    case neg.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le (M …
    C : NNReal := HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal
    hC : Eq C (HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal)
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  have hCpos : 0 < C := mul_pos (inv_pos.2 hδ) (ENNReal.toNNReal_pos hne hg.eLpNorm_lt_top.ne)
  have : ∀ n, μ {x : α | C ≤ ‖(μ[g|ℱ n]) x‖₊} ≤ ENNReal.ofReal δ := by
    intro n
    have := mul_meas_ge_le_pow_eLpNorm' μ one_ne_zero ENNReal.one_ne_top
      ((stronglyMeasurable_condexp (m := ℱ n) (μ := μ) (f := g)).mono (hℱ n)).aestronglyMeasurable C
    rw [ENNReal.one_toReal, ENNReal.rpow_one, ENNReal.rpow_one, mul_comm, ←
      ENNReal.le_div_iff_mul_le (Or.inl (ENNReal.coe_ne_zero.2 hCpos.ne'))
        (Or.inl ENNReal.coe_lt_top.ne)] at this
    simp_rw [ENNReal.coe_le_coe] at this
    refine this.trans ?_
    rw [ENNReal.div_le_iff_le_mul (Or.inl (ENNReal.coe_ne_zero.2 hCpos.ne'))
        (Or.inl ENNReal.coe_lt_top.ne),
      hC, Nonneg.inv_mk, ENNReal.coe_mul, ENNReal.coe_toNNReal hg.eLpNorm_lt_top.ne, ← mul_assoc, ←
      ENNReal.ofReal_eq_coe_nnreal, ← ENNReal.ofReal_mul hδ.le, mul_inv_cancel₀ hδ.ne',
      ENNReal.ofReal_one, one_mul]
    exact eLpNorm_one_condexp_le_eLpNorm _
  /-
    case neg.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le (M …
    C : NNReal := HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal
    hC : Eq C (HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal)
    hCpos : LT.lt 0 C
    this : ∀ (n : ι), LE.le (μ (setOf fun x => LE.le C (NNNorm.nnnorm (MeasureTheo …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  refine ⟨C, fun n => le_trans ?_ (h {x : α | C ≤ ‖(μ[g|ℱ n]) x‖₊} (hmeas n C) (this n))⟩
  have hmeasℱ : MeasurableSet[ℱ n] {x : α | C ≤ ‖(μ[g|ℱ n]) x‖₊} :=
    @measurableSet_le _ _ _ _ _ (ℱ n) _ _ _ _ _ measurable_const
      (@Measurable.nnnorm _ _ _ _ _ (ℱ n) _ stronglyMeasurable_condexp.measurable)
  /-
    case neg.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le (M …
    C : NNReal := HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal
    hC : Eq C (HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal)
    hCpos : LT.lt 0 C
    this : ∀ (n : ι), LE.le (μ (setOf fun x => LE.le C (NNNorm.nnnorm (MeasureTheo …
    n : ι
    hmeasℱ : MeasurableSet (setOf fun x => LE.le C (NNNorm.nnnorm (MeasureTheory.c …
    ⊢ LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm.nnnorm (Measur …
  -/
  rw [← eLpNorm_congr_ae (condexp_indicator hint hmeasℱ)]
  /-
    case neg.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : α → Real
    hint : MeasureTheory.Integrable g μ
    ℱ : ι → MeasurableSpace α
    hℱ : ∀ (i : ι), LE.le (ℱ i) m0
    A : MeasurableSpace α := m0
    hmeas : ∀ (n : ι) (C : NNReal), MeasurableSet (setOf fun x => LE.le C (NNNorm. …
    hg : MeasureTheory.Memℒp g 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hne : Not (Eq (MeasureTheory.eLpNorm g 1 μ) 0)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le (M …
    C : NNReal := HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal
    hC : Eq C (HMul.hMul (Inv.inv ⟨δ, ⋯⟩) (MeasureTheory.eLpNorm g 1 μ).toNNReal)
    hCpos : LT.lt 0 C
    this : ∀ (n : ι), LE.le (μ (setOf fun x => LE.le C (NNNorm.nnnorm (MeasureTheo …
    n : ι
    hmeasℱ : MeasurableSet (setOf fun x => LE.le C (NNNorm.nnnorm (MeasureTheory.c …
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.condexp (ℱ n) μ ((setOf fun x => …
  -/
  exact eLpNorm_one_condexp_le_eLpNorm _
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `condexp_stronglyMeasurable_mul`. -/
theorem condexp_stronglyMeasurable_simpleFunc_mul (hm : m ≤ m0) (f : @SimpleFunc α m ℝ) {g : α → ℝ}
    (hg : Integrable g μ) : μ[(f * g : α → ℝ)|m] =ᵐ[μ] f * μ[g|m] := by
  have : ∀ (s c) (f : α → ℝ), Set.indicator s (Function.const α c) * f = s.indicator (c • f) := by
    intro s c f
    ext1 x
    by_cases hx : x ∈ s
    · simp only [hx, Pi.mul_apply, Set.indicator_of_mem, Pi.smul_apply, Algebra.id.smul_eq_mul,
        Function.const_apply]
    · simp only [hx, Pi.mul_apply, Set.indicator_of_not_mem, not_false_iff, zero_mul]
  apply @SimpleFunc.induction _ _ m _ (fun f => _)
    (fun c s hs => ?_) (fun g₁ g₂ _ h_eq₁ h_eq₂ => ?_) f
  · -- Porting note: if not classical, `DecidablePred fun x ↦ x ∈ s` cannot be synthesised
    -- for `Set.piecewise_eq_indicator`
    classical simp only [@SimpleFunc.const_zero _ _ m, @SimpleFunc.coe_piecewise _ _ m,
      @SimpleFunc.coe_const _ _ m, @SimpleFunc.coe_zero _ _ m, Set.piecewise_eq_indicator]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : MeasureTheory.SimpleFunc α Real
      g : α → Real
      hg : MeasureTheory.Integrable g μ
      this : ∀ (s : Set α) (c : Real) (f : α → Real), Eq (HMul.hMul (s.indicator (Fu …
      c : Real
      s : Set α
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (s.i …
    -/
    rw [this, this]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : MeasureTheory.SimpleFunc α Real
      g : α → Real
      hg : MeasureTheory.Integrable g μ
      this : ∀ (s : Set α) (c : Real) (f : α → Real), Eq (HMul.hMul (s.indicator (Fu …
      c : Real
      s : Set α
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (s.indicator (H …
    -/
    refine (condexp_indicator (hg.smul c) hs).trans ?_
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : MeasureTheory.SimpleFunc α Real
      g : α → Real
      hg : MeasureTheory.Integrable g μ
      this : ∀ (s : Set α) (c : Real) (f : α → Real), Eq (HMul.hMul (s.indicator (Fu …
      c : Real
      s : Set α
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (MeasureTheory.condexp m μ (H …
    -/
    filter_upwards [condexp_smul (m := m) (m0 := m0) c g] with x hx
    /-
      case h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : MeasureTheory.SimpleFunc α Real
      g : α → Real
      hg : MeasureTheory.Integrable g μ
      this : ∀ (s : Set α) (c : Real) (f : α → Real), Eq (HMul.hMul (s.indicator (Fu …
      c : Real
      s : Set α
      hs : MeasurableSet s
      x : α
      hx : Eq (MeasureTheory.condexp m μ (HSMul.hSMul c g) x) (HSMul.hSMul c (Measur …
      ⊢ Eq (s.indicator (MeasureTheory.condexp m μ (HSMul.hSMul c g)) x) (s.indicato …
    -/
    classical simp_rw [Set.indicator_apply, hx]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : MeasureTheory.SimpleFunc α Real
      g : α → Real
      hg : MeasureTheory.Integrable g μ
      this : ∀ (s : Set α) (c : Real) (f : α → Real), Eq (HMul.hMul (s.indicator (Fu …
      g₁ g₂ : MeasureTheory.SimpleFunc α Real
      x✝ : Disjoint (Function.support ⇑g₁) (Function.support ⇑g₂)
      h_eq₁ : (fun f => (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ …
      h_eq₂ : (fun f => (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (⇑(H …
    -/
  · have h_add := @SimpleFunc.coe_add _ _ m _ g₁ g₂
    calc
      μ[⇑(g₁ + g₂) * g|m] =ᵐ[μ] μ[(⇑g₁ + ⇑g₂) * g|m] := by
        refine condexp_congr_ae (EventuallyEq.mul ?_ EventuallyEq.rfl); rw [h_add]
      _ =ᵐ[μ] μ[⇑g₁ * g|m] + μ[⇑g₂ * g|m] := by
        rw [add_mul]; exact condexp_add (hg.simpleFunc_mul' hm _) (hg.simpleFunc_mul' hm _)
      _ =ᵐ[μ] ⇑g₁ * μ[g|m] + ⇑g₂ * μ[g|m] := EventuallyEq.add h_eq₁ h_eq₂
      _ =ᵐ[μ] ⇑(g₁ + g₂) * μ[g|m] := by rw [h_add, add_mul]


theorem condexp_stronglyMeasurable_mul_of_bound (hm : m ≤ m0) [IsFiniteMeasure μ] {f g : α → ℝ}
    (hf : StronglyMeasurable[m] f) (hg : Integrable g μ) (c : ℝ) (hf_bound : ∀ᵐ x ∂μ, ‖f x‖ ≤ c) :
    μ[f * g|m] =ᵐ[μ] f * μ[g|m] := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  let fs := hf.approxBounded c
  have hfs_tendsto : ∀ᵐ x ∂μ, Tendsto (fs · x) atTop (𝓝 (f x)) :=
    hf.tendsto_approxBounded_ae hf_bound
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
    hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Eq μ 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
    -/
  · simp only [hμ, ae_zero]; norm_cast
                             /-
                               🎉 no goals
                             -/
  /-
    case neg
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
    hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
    hμ : Not (Eq μ 0)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  have : (ae μ).NeBot := ae_neBot.2 hμ
  have hc : 0 ≤ c := by
    rcases hf_bound.exists with ⟨_x, hx⟩
    exact (norm_nonneg _).trans hx
  /-
    case neg
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
    hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
    hμ : Not (Eq μ 0)
    this : (MeasureTheory.ae μ).NeBot
    hc : LE.le 0 c
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  have hfs_bound : ∀ n x, ‖fs n x‖ ≤ c := hf.norm_approxBounded_le hc
  have : μ[f * μ[g|m]|m] = f * μ[g|m] := by
    refine condexp_of_stronglyMeasurable hm (hf.mul stronglyMeasurable_condexp) ?_
    exact integrable_condexp.bdd_mul' (hf.mono hm).aestronglyMeasurable hf_bound
  /-
    case neg
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
    hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
    hμ : Not (Eq μ 0)
    this✝ : (MeasureTheory.ae μ).NeBot
    hc : LE.le 0 c
    hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
    this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  rw [← this]
  refine tendsto_condexp_unique (fun n x => fs n x * g x) (fun n x => fs n x * (μ[g|m]) x) (f * g)
    (f * μ[g|m]) ?_ ?_ ?_ ?_ (c * ‖g ·‖) ?_ (c * ‖(μ[g|m]) ·‖) ?_ ?_ ?_ ?_
  · exact fun n => hg.bdd_mul' ((SimpleFunc.stronglyMeasurable (fs n)).mono hm).aestronglyMeasurable
      (Eventually.of_forall (hfs_bound n))
  · exact fun n => integrable_condexp.bdd_mul'
      ((SimpleFunc.stronglyMeasurable (fs n)).mono hm).aestronglyMeasurable
      (Eventually.of_forall (hfs_bound n))
    /-
      case neg.refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => (fun n x => HMul.hMul ( …
    -/
  · filter_upwards [hfs_tendsto] with x hx
    /-
      case h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      x : α
      hx : Filter.Tendsto (fun x_1 => (fs x_1) x) Filter.atTop (nhds (f x))
      ⊢ Filter.Tendsto (fun n => HMul.hMul ((fs n) x) (g x)) Filter.atTop (nhds (HMu …
    -/
    exact hx.mul tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_4
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => (fun n x => HMul.hMul ( …
    -/
  · filter_upwards [hfs_tendsto] with x hx
    /-
      case h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      x : α
      hx : Filter.Tendsto (fun x_1 => (fs x_1) x) Filter.atTop (nhds (f x))
      ⊢ Filter.Tendsto (fun n => HMul.hMul ((fs n) x) (MeasureTheory.condexp m μ g x …
    -/
    exact hx.mul tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_5
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul c (Norm.norm (g x))) μ
    -/
  · exact hg.norm.const_mul c
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_6
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul c (Norm.norm (MeasureTheory.con …
    -/
  · exact integrable_condexp.norm.const_mul c
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_7
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm ((fun n x => HMul. …
    -/
  · refine fun n => Eventually.of_forall fun x => ?_
    /-
      case neg.refine_7
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      n : Nat
      x : α
      ⊢ LE.le (Norm.norm ((fun n x => HMul.hMul ((fs n) x) (g x)) n x)) ((fun x => H …
    -/
    exact (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right (hfs_bound n x) (norm_nonneg _))
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_8
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun x => LE.le (Norm.norm ((fun n x => HMul. …
    -/
  · refine fun n => Eventually.of_forall fun x => ?_
    /-
      case neg.refine_8
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      n : Nat
      x : α
      ⊢ LE.le (Norm.norm ((fun n x => HMul.hMul ((fs n) x) (MeasureTheory.condexp m  …
    -/
    exact (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right (hfs_bound n x) (norm_nonneg _))
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_9
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      ⊢ ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (( …
    -/
  · intro n
    /-
      case neg.refine_9
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ ((fun n x => HM …
    -/
    simp_rw [← Pi.mul_apply]
    /-
      case neg.refine_9
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.Integrable g μ
      c : Real
      hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
      fs : Nat → MeasureTheory.SimpleFunc α Real := hf.approxBounded c
      hfs_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun x_1 => (fs x_1)  …
      hμ : Not (Eq μ 0)
      this✝ : (MeasureTheory.ae μ).NeBot
      hc : LE.le 0 c
      hfs_bound : ∀ (n : Nat) (x : α), LE.le (Norm.norm ((fs n) x)) c
      this : Eq (MeasureTheory.condexp m μ (HMul.hMul f (MeasureTheory.condexp m μ g …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ fun x => HMul.h …
    -/
    refine (condexp_stronglyMeasurable_simpleFunc_mul hm _ hg).trans ?_
    rw [condexp_of_stronglyMeasurable hm
      ((SimpleFunc.stronglyMeasurable _).mul stronglyMeasurable_condexp) _]
    exact integrable_condexp.bdd_mul'
      ((SimpleFunc.stronglyMeasurable (fs n)).mono hm).aestronglyMeasurable
      (Eventually.of_forall (hfs_bound n))


theorem condexp_stronglyMeasurable_mul_of_bound₀ (hm : m ≤ m0) [IsFiniteMeasure μ] {f g : α → ℝ}
    (hf : AEStronglyMeasurable' m f μ) (hg : Integrable g μ) (c : ℝ)
    (hf_bound : ∀ᵐ x ∂μ, ‖f x‖ ≤ c) : μ[f * g|m] =ᵐ[μ] f * μ[g|m] := by
  have : μ[f * g|m] =ᵐ[μ] μ[hf.mk f * g|m] :=
    condexp_congr_ae (EventuallyEq.mul hf.ae_eq_mk EventuallyEq.rfl)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  refine this.trans ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (Mea …
  -/
  have : f * μ[g|m] =ᵐ[μ] hf.mk f * μ[g|m] := EventuallyEq.mul hf.ae_eq_mk EventuallyEq.rfl
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (Mea …
  -/
  refine EventuallyEq.trans ?_ this.symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (Mea …
  -/
  refine condexp_stronglyMeasurable_mul_of_bound hm hf.stronglyMeasurable_mk hg c ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (MeasureTheory.AEStronglyMeasur …
  -/
  filter_upwards [hf_bound, hf.ae_eq_mk] with x hxc hx_eq
  /-
    case h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hg : MeasureTheory.Integrable g μ
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    x : α
    hxc : LE.le (Norm.norm (f x)) c
    hx_eq : Eq (f x) (MeasureTheory.AEStronglyMeasurable'.mk f hf x)
    ⊢ LE.le (Norm.norm (MeasureTheory.AEStronglyMeasurable'.mk f hf x)) c
  -/
  rwa [← hx_eq]
  /-
    🎉 no goals
  -/


/-- Pull-out property of the conditional expectation. -/
theorem condexp_stronglyMeasurable_mul {f g : α → ℝ} (hf : StronglyMeasurable[m] f)
    (hfg : Integrable (f * g) μ) (hg : Integrable g μ) : μ[f * g|m] =ᵐ[μ] f * μ[g|m] := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  by_cases hm : m ≤ m0; swap; · simp_rw [condexp_of_not_le hm]; rw [mul_zero]
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  by_cases hμm : SigmaFinite (μ.trim hm)
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  swap; · simp_rw [condexp_of_not_sigmaFinite hm hμm]; rw [mul_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  haveI : SigmaFinite (μ.trim hm) := hμm
  /-
    case pos
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  obtain ⟨sets, sets_prop, h_univ⟩ := hf.exists_spanning_measurableSet_norm_le hm μ
  /-
    case pos.intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    sets_prop : ∀ (n : Nat), And (MeasurableSet (sets n)) (And (LT.lt (μ (sets n)) …
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  simp_rw [forall_and] at sets_prop
  /-
    case pos.intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    sets_prop : And (∀ (x : Nat), MeasurableSet (sets x)) (And (∀ (x : Nat), LT.lt …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  obtain ⟨h_meas, h_finite, h_norm⟩ := sets_prop
  suffices ∀ n, ∀ᵐ x ∂μ, x ∈ sets n → (μ[f * g|m]) x = f x * (μ[g|m]) x by
    rw [← ae_all_iff] at this
    filter_upwards [this] with x hx
    obtain ⟨i, hi⟩ : ∃ i, x ∈ sets i := by
      have h_mem : x ∈ ⋃ i, sets i := by rw [h_univ]; exact Set.mem_univ _
      simpa using h_mem
    exact hx i hi
  /-
    case pos.intro.intro.intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    h_meas : ∀ (x : Nat), MeasurableSet (sets x)
    h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
    h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
    ⊢ ∀ (n : Nat), Filter.Eventually (fun x => Membership.mem (sets n) x → Eq (Mea …
  -/
  refine fun n => ae_imp_of_ae_restrict ?_
  suffices (μ.restrict (sets n))[f * g|m] =ᵐ[μ.restrict (sets n)] f * (μ.restrict (sets n))[g|m] by
    refine (condexp_restrict_ae_eq_restrict hm (h_meas n) hfg).symm.trans ?_
    exact this.trans (EventuallyEq.rfl.mul (condexp_restrict_ae_eq_restrict hm (h_meas n) hg))
  suffices (μ.restrict (sets n))[(sets n).indicator f * g|m] =ᵐ[μ.restrict (sets n)]
      (sets n).indicator f * (μ.restrict (sets n))[g|m] by
    refine EventuallyEq.trans ?_ (this.trans ?_)
    · exact
        condexp_congr_ae ((indicator_ae_eq_restrict <| hm _ <| h_meas n).symm.mul EventuallyEq.rfl)
    · exact (indicator_ae_eq_restrict <| hm _ <| h_meas n).mul EventuallyEq.rfl
  have : IsFiniteMeasure (μ.restrict (sets n)) := by
    constructor
    rw [Measure.restrict_apply_univ]
    exact h_finite n
  /-
    case pos.intro.intro.intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    h_meas : ∀ (x : Nat), MeasurableSet (sets x)
    h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
    h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
    n : Nat
    this : MeasureTheory.IsFiniteMeasure (μ.restrict (sets n))
    ⊢ (MeasureTheory.ae (μ.restrict (sets n))).EventuallyEq (MeasureTheory.condexp …
  -/
  refine condexp_stronglyMeasurable_mul_of_bound hm (hf.indicator (h_meas n)) hg.integrableOn n ?_
  /-
    case pos.intro.intro.intro.intro
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    h_meas : ∀ (x : Nat), MeasurableSet (sets x)
    h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
    h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
    n : Nat
    this : MeasureTheory.IsFiniteMeasure (μ.restrict (sets n))
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm ((sets n).indicator f x)) ↑n) ( …
  -/
  filter_upwards with x
  /-
    case pos.intro.intro.intro.intro.h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.StronglyMeasurable f
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    hm : LE.le m m0
    hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    sets : Nat → Set α
    h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
    h_meas : ∀ (x : Nat), MeasurableSet (sets x)
    h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
    h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
    n : Nat
    this : MeasureTheory.IsFiniteMeasure (μ.restrict (sets n))
    x : α
    ⊢ LE.le (Norm.norm ((sets n).indicator f x)) ↑n
  -/
  by_cases hxs : x ∈ sets n
    /-
      case pos
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
      hg : MeasureTheory.Integrable g μ
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      sets : Nat → Set α
      h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
      h_meas : ∀ (x : Nat), MeasurableSet (sets x)
      h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
      h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
      n : Nat
      this : MeasureTheory.IsFiniteMeasure (μ.restrict (sets n))
      x : α
      hxs : Membership.mem (sets n) x
      ⊢ LE.le (Norm.norm ((sets n).indicator f x)) ↑n
    -/
  · simpa only [hxs, Set.indicator_of_mem] using h_norm n x hxs
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : MeasureTheory.StronglyMeasurable f
      hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
      hg : MeasureTheory.Integrable g μ
      hm : LE.le m m0
      hμm this✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      sets : Nat → Set α
      h_univ : Eq (Set.iUnion fun i => sets i) Set.univ
      h_meas : ∀ (x : Nat), MeasurableSet (sets x)
      h_finite : ∀ (x : Nat), LT.lt (μ (sets x)) Top.top
      h_norm : ∀ (x : Nat) (x_1 : α), Membership.mem (sets x) x_1 → LE.le (Norm.norm …
      n : Nat
      this : MeasureTheory.IsFiniteMeasure (μ.restrict (sets n))
      x : α
      hxs : Not (Membership.mem (sets n) x)
      ⊢ LE.le (Norm.norm ((sets n).indicator f x)) ↑n
    -/
  · simp only [hxs, Set.indicator_of_not_mem, not_false_iff, _root_.norm_zero, Nat.cast_nonneg]
    /-
      🎉 no goals
    -/


/-- Pull-out property of the conditional expectation. -/
theorem condexp_stronglyMeasurable_mul₀ {f g : α → ℝ} (hf : AEStronglyMeasurable' m f μ)
    (hfg : Integrable (f * g) μ) (hg : Integrable g μ) : μ[f * g|m] =ᵐ[μ] f * μ[g|m] := by
  have : μ[f * g|m] =ᵐ[μ] μ[hf.mk f * g|m] :=
    condexp_congr_ae (hf.ae_eq_mk.mul EventuallyEq.rfl)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    this : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul f g) …
  -/
  refine this.trans ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    this : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (Mea …
  -/
  have : f * μ[g|m] =ᵐ[μ] hf.mk f * μ[g|m] := hf.ae_eq_mk.mul EventuallyEq.rfl
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMul (Mea …
  -/
  refine (condexp_stronglyMeasurable_mul hf.stronglyMeasurable_mk ?_ hg).trans this.symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ MeasureTheory.Integrable (HMul.hMul (MeasureTheory.AEStronglyMeasurable'.mk  …
  -/
  refine (integrable_congr ?_).mp hfg
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.AEStronglyMeasurable' m f μ
    hfg : MeasureTheory.Integrable (HMul.hMul f g) μ
    hg : MeasureTheory.Integrable g μ
    this✝ : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m μ (HMul.hMu …
    this : (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f (MeasureTheory.condexp m …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HMul.hMul f g) (HMul.hMul (MeasureTheory. …
  -/
  exact hf.ae_eq_mk.mul EventuallyEq.rfl
  /-
    🎉 no goals
  -/


