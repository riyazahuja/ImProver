/-- The `ℝ≥0∞`-valued variance of a real-valued random variable defined as the Lebesgue integral of
`(X - 𝔼[X])^2`. -/
def evariance {Ω : Type*} {_ : MeasurableSpace Ω} (X : Ω → ℝ) (μ : Measure Ω) : ℝ≥0∞ :=
  ∫⁻ ω, (‖X ω - μ[X]‖₊ : ℝ≥0∞) ^ 2 ∂μ


/-- The `ℝ`-valued variance of a real-valued random variable defined by applying `ENNReal.toReal`
to `evariance`. -/
def variance {Ω : Type*} {_ : MeasurableSpace Ω} (X : Ω → ℝ) (μ : Measure Ω) : ℝ :=
  (evariance X μ).toReal


theorem _root_.MeasureTheory.Memℒp.evariance_lt_top [IsFiniteMeasure μ] (hX : Memℒp X 2 μ) :
    evariance X μ < ∞ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  have := ENNReal.pow_lt_top (hX.sub <| memℒp_const <| μ[X]).2 2
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    this : LT.lt (HPow.hPow (MeasureTheory.eLpNorm (HSub.hSub X fun x => MeasureTh …
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  rw [eLpNorm_eq_lintegral_rpow_nnnorm two_ne_zero ENNReal.two_ne_top, ← ENNReal.rpow_two] at this
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    this : LT.lt (HPow.hPow (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hP …
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  simp only [ENNReal.toReal_ofNat, Pi.sub_apply, ENNReal.one_toReal, one_div] at this
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    this : LT.lt (HPow.hPow (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hP …
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  rw [← ENNReal.rpow_mul, inv_mul_cancel₀ (two_ne_zero : (2 : ℝ) ≠ 0), ENNReal.rpow_one] at this
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    this : LT.lt (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm (H …
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  simp_rw [ENNReal.rpow_two] at this
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    this : LT.lt (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm (H …
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem evariance_eq_top [IsFiniteMeasure μ] (hXm : AEStronglyMeasurable X μ) (hX : ¬Memℒp X 2 μ) :
    evariance X μ = ∞ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    ⊢ Eq (ProbabilityTheory.evariance X μ) Top.top
  -/
  by_contra h
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    h : Not (Eq (ProbabilityTheory.evariance X μ) Top.top)
    ⊢ False
  -/
  rw [← Ne, ← lt_top_iff_ne_top] at h
  have : Memℒp (fun ω => X ω - μ[X]) 2 μ := by
    refine ⟨hXm.sub aestronglyMeasurable_const, ?_⟩
    rw [eLpNorm_eq_lintegral_rpow_nnnorm two_ne_zero ENNReal.two_ne_top]
    simp only [ENNReal.toReal_ofNat, ENNReal.one_toReal, ENNReal.rpow_two, Ne]
    exact ENNReal.rpow_lt_top_of_nonneg (by linarith) h.ne
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    h : LT.lt (ProbabilityTheory.evariance X μ) Top.top
    this : MeasureTheory.Memℒp (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ …
    ⊢ False
  -/
  refine hX ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    h : LT.lt (ProbabilityTheory.evariance X μ) Top.top
    this : MeasureTheory.Memℒp (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ …
    ⊢ MeasureTheory.Memℒp X 2 μ
  -/
  convert this.add (memℒp_const μ[X])
  /-
    case h.e'_6
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    h : LT.lt (ProbabilityTheory.evariance X μ) Top.top
    this : MeasureTheory.Memℒp (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ …
    ⊢ Eq X (HAdd.hAdd (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ fun x => …
  -/
  ext ω
  /-
    case h.e'_6.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hXm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    h : LT.lt (ProbabilityTheory.evariance X μ) Top.top
    this : MeasureTheory.Memℒp (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ …
    ω : Ω
    ⊢ Eq (X ω) (HAdd.hAdd (fun ω => HSub.hSub (X ω) (MeasureTheory.integral μ fun  …
  -/
  rw [Pi.add_apply, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem evariance_lt_top_iff_memℒp [IsFiniteMeasure μ] (hX : AEStronglyMeasurable X μ) :
    evariance X μ < ∞ ↔ Memℒp X 2 μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ Iff (LT.lt (ProbabilityTheory.evariance X μ) Top.top) (MeasureTheory.Memℒp X …
  -/
  refine ⟨?_, MeasureTheory.Memℒp.evariance_lt_top⟩
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ LT.lt (ProbabilityTheory.evariance X μ) Top.top → MeasureTheory.Memℒp X 2 μ
  -/
  contrapose
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ Not (MeasureTheory.Memℒp X 2 μ) → Not (LT.lt (ProbabilityTheory.evariance X  …
  -/
  rw [not_lt, top_le_iff]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ Not (MeasureTheory.Memℒp X 2 μ) → Eq (ProbabilityTheory.evariance X μ) Top.top
  -/
  exact evariance_eq_top hX
  /-
    🎉 no goals
  -/


theorem _root_.MeasureTheory.Memℒp.ofReal_variance_eq [IsFiniteMeasure μ] (hX : Memℒp X 2 μ) :
    ENNReal.ofReal (variance X μ) = evariance X μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    ⊢ Eq (ENNReal.ofReal (ProbabilityTheory.variance X μ)) (ProbabilityTheory.evar …
  -/
  rw [variance, ENNReal.ofReal_toReal]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hX : MeasureTheory.Memℒp X 2 μ
    ⊢ Ne (ProbabilityTheory.evariance X μ) Top.top
  -/
  exact hX.evariance_lt_top.ne
  /-
    🎉 no goals
  -/


theorem evariance_eq_lintegral_ofReal (X : Ω → ℝ) (μ : Measure Ω) :
    evariance X μ = ∫⁻ ω, ENNReal.ofReal ((X ω - μ[X]) ^ 2) ∂μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.evariance X μ) (MeasureTheory.lintegral μ fun ω => ENN …
  -/
  rw [evariance]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub …
  -/
  congr
  /-
    case e_f
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (fun ω => HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (X ω) (MeasureTheory.inte …
  -/
  ext1 ω
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (X ω) (MeasureTheory.integral μ fu …
  -/
  rw [pow_two, ← ENNReal.coe_mul, ← nnnorm_mul, ← pow_two]
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (↑(NNNorm.nnnorm (HPow.hPow (HSub.hSub (X ω) (MeasureTheory.integral μ fu …
  -/
  congr
  /-
    case e_f.h.e_a
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (NNNorm.nnnorm (HPow.hPow (HSub.hSub (X ω) (MeasureTheory.integral μ fun  …
  -/
  exact (Real.toNNReal_eq_nnnorm_of_nonneg <| sq_nonneg _).symm
  /-
    🎉 no goals
  -/


theorem _root_.MeasureTheory.Memℒp.variance_eq_of_integral_eq_zero (hX : Memℒp X 2 μ)
    (hXint : μ[X] = 0) : variance X μ = μ[X ^ (2 : Nat)] := by
  rw [variance, evariance_eq_lintegral_ofReal, ← ofReal_integral_eq_lintegral_ofReal,
      ENNReal.toReal_ofReal (by positivity)] <;>
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : MeasureTheory.Memℒp X 2 μ
      hXint : Eq (MeasureTheory.integral μ fun x => X x) 0
      ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (HSub.hSub (X x) (MeasureThe …
    -/
    simp_rw [hXint, sub_zero]
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : MeasureTheory.Memℒp X 2 μ
      hXint : Eq (MeasureTheory.integral μ fun x => X x) 0
      ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (X x) 2) (MeasureTheory.inte …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hfi
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : MeasureTheory.Memℒp X 2 μ
      hXint : Eq (MeasureTheory.integral μ fun x => X x) 0
      ⊢ MeasureTheory.Integrable (fun ω => HPow.hPow (X ω) 2) μ
    -/
  · convert hX.integrable_norm_rpow two_ne_zero ENNReal.two_ne_top with ω
    simp only [Pi.sub_apply, Real.norm_eq_abs, ENNReal.toReal_ofNat, ENNReal.one_toReal,
      Real.rpow_two, sq_abs, abs_pow]
    /-
      case f_nn
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : MeasureTheory.Memℒp X 2 μ
      hXint : Eq (MeasureTheory.integral μ fun x => X x) 0
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun ω => HPow.hPow (X ω) 2
    -/
  · exact ae_of_all _ fun ω => pow_two_nonneg _
    /-
      🎉 no goals
    -/


theorem _root_.MeasureTheory.Memℒp.variance_eq [IsFiniteMeasure μ] (hX : Memℒp X 2 μ) :
    variance X μ = μ[(X - fun _ => μ[X] :) ^ (2 : Nat)] := by
  rw [variance, evariance_eq_lintegral_ofReal, ← ofReal_integral_eq_lintegral_ofReal,
    ENNReal.toReal_ofReal (by positivity)]
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (HSub.hSub (X x) (MeasureThe …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · convert (hX.sub <| memℒp_const μ[X]).integrable_norm_rpow two_ne_zero ENNReal.two_ne_top
      with ω
    simp only [Pi.sub_apply, Real.norm_eq_abs, ENNReal.toReal_ofNat, ENNReal.one_toReal,
      Real.rpow_two, sq_abs, abs_pow]
    /-
      case f_nn
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun ω => HPow.hPow (HSub.hSub (X ω) (Mea …
    -/
  · exact ae_of_all _ fun ω => pow_two_nonneg _
    /-
      🎉 no goals
    -/


@[simp]
                                                 /-
                                                   Ω : Type u_1
                                                   m : MeasurableSpace Ω
                                                   μ : MeasureTheory.Measure Ω
                                                   ⊢ Eq (ProbabilityTheory.evariance 0 μ) 0
                                                 -/
theorem evariance_zero : evariance 0 μ = 0 := by simp [evariance]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem evariance_eq_zero_iff (hX : AEMeasurable X μ) :
    evariance X μ = 0 ↔ X =ᵐ[μ] fun _ => μ[X] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    hX : AEMeasurable X μ
    ⊢ Iff (Eq (ProbabilityTheory.evariance X μ) 0) ((MeasureTheory.ae μ).Eventuall …
  -/
  rw [evariance, lintegral_eq_zero_iff']
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    hX : AEMeasurable X μ
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (fun ω => HPow.hPow (↑(NNNorm.nnnorm  …
  -/
  constructor <;> intro hX <;> filter_upwards [hX] with ω hω
  · simpa only [Pi.zero_apply, sq_eq_zero_iff, ENNReal.coe_eq_zero, nnnorm_eq_zero, sub_eq_zero]
      using hω
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX✝ : AEMeasurable X μ
      hX : (MeasureTheory.ae μ).EventuallyEq X fun x => MeasureTheory.integral μ fun …
      ω : Ω
      hω : Eq (X ω) (MeasureTheory.integral μ fun x => X x)
      ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (X ω) (MeasureTheory.integral μ fu …
    -/
  · rw [hω]
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX✝ : AEMeasurable X μ
      hX : (MeasureTheory.ae μ).EventuallyEq X fun x => MeasureTheory.integral μ fun …
      ω : Ω
      hω : Eq (X ω) (MeasureTheory.integral μ fun x => X x)
      ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (MeasureTheory.integral μ fun x => …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : AEMeasurable X μ
      ⊢ AEMeasurable (fun ω => HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (X ω) (MeasureT …
    -/
  · exact (hX.sub_const _).ennnorm.pow_const _ -- TODO `measurability` and `fun_prop` fail
    /-
      🎉 no goals
    -/


theorem evariance_mul (c : ℝ) (X : Ω → ℝ) (μ : Measure Ω) :
    evariance (fun ω => c * X ω) μ = ENNReal.ofReal (c ^ 2) * evariance X μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.evariance (fun ω => HMul.hMul c (X ω)) μ) (HMul.hMul ( …
  -/
  rw [evariance, evariance, ← lintegral_const_mul' _ _ ENNReal.ofReal_lt_top.ne]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub …
  -/
  congr
  /-
    case e_f
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (fun ω => HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (HMul.hMul c (X ω)) (Meas …
  -/
  ext1 ω
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (HSub.hSub (HMul.hMul c (X ω)) (MeasureTheory …
  -/
  rw [ENNReal.ofReal, ← ENNReal.coe_pow, ← ENNReal.coe_pow, ← ENNReal.coe_mul]
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq ↑(HPow.hPow (NNNorm.nnnorm (HSub.hSub (HMul.hMul c (X ω)) (MeasureTheory. …
  -/
  congr
  rw [← sq_abs, ← Real.rpow_two, Real.toNNReal_rpow_of_nonneg (abs_nonneg _), NNReal.rpow_two,
    ← mul_pow, Real.toNNReal_mul_nnnorm _ (abs_nonneg _)]
  /-
    case e_f.h.e_a
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (HPow.hPow (NNNorm.nnnorm (HSub.hSub (HMul.hMul c (X ω)) (MeasureTheory.i …
  -/
  conv_rhs => rw [← nnnorm_norm, norm_mul, norm_abs_eq_norm, ← norm_mul, nnnorm_norm, mul_sub]
  /-
    case e_f.h.e_a
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (HPow.hPow (NNNorm.nnnorm (HSub.hSub (HMul.hMul c (X ω)) (MeasureTheory.i …
  -/
  congr
  /-
    case e_f.h.e_a.e_a.e_a.e_a
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul c (X x)) (HMul.hMul c (Measu …
  -/
  rw [mul_comm]
  /-
    case e_f.h.e_a.e_a.e_a.e_a
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ω : Ω
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul c (X x)) (HMul.hMul (Measure …
  -/
  simp_rw [← smul_eq_mul, ← integral_smul_const, smul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


scoped notation "eVar[" X "]" => ProbabilityTheory.evariance X MeasureTheory.MeasureSpace.volume


@[simp]
theorem variance_zero (μ : Measure Ω) : variance 0 μ = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.variance 0 μ) 0
  -/
  simp only [variance, evariance_zero, ENNReal.zero_toReal]
  /-
    🎉 no goals
  -/


theorem variance_nonneg (X : Ω → ℝ) (μ : Measure Ω) : 0 ≤ variance X μ :=
  ENNReal.toReal_nonneg


theorem variance_mul (c : ℝ) (X : Ω → ℝ) (μ : Measure Ω) :
    variance (fun ω => c * X ω) μ = c ^ 2 * variance X μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.variance (fun ω => HMul.hMul c (X ω)) μ) (HMul.hMul (H …
  -/
  rw [variance, evariance_mul, ENNReal.toReal_mul, ENNReal.toReal_ofReal (sq_nonneg _)]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    c : Real
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (HMul.hMul (HPow.hPow c 2) (ProbabilityTheory.evariance X μ).toReal) (HMu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem variance_smul (c : ℝ) (X : Ω → ℝ) (μ : Measure Ω) :
    variance (c • X) μ = c ^ 2 * variance X μ :=
  variance_mul c X μ


theorem variance_smul' {A : Type*} [CommSemiring A] [Algebra A ℝ] (c : A) (X : Ω → ℝ)
    (μ : Measure Ω) : variance (c • X) μ = c ^ 2 • variance X μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    A : Type u_2
    inst✝¹ : CommSemiring A
    inst✝ : Algebra A Real
    c : A
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.variance (HSMul.hSMul c X) μ) (HSMul.hSMul (HPow.hPow  …
  -/
  convert variance_smul (algebraMap A ℝ c) X μ using 1
    /-
      case h.e'_2
      Ω : Type u_1
      m : MeasurableSpace Ω
      A : Type u_2
      inst✝¹ : CommSemiring A
      inst✝ : Algebra A Real
      c : A
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      ⊢ Eq (ProbabilityTheory.variance (HSMul.hSMul c X) μ) (ProbabilityTheory.varia …
    -/
  · congr; simp only [algebraMap_smul]
           /-
             🎉 no goals
           -/
    /-
      case h.e'_3
      Ω : Type u_1
      m : MeasurableSpace Ω
      A : Type u_2
      inst✝¹ : CommSemiring A
      inst✝ : Algebra A Real
      c : A
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      ⊢ Eq (HSMul.hSMul (HPow.hPow c 2) (ProbabilityTheory.variance X μ)) (HMul.hMul …
    -/
  · simp only [Algebra.smul_def, map_pow]
    /-
      🎉 no goals
    -/


scoped notation "Var[" X "]" => ProbabilityTheory.variance X MeasureTheory.MeasureSpace.volume


theorem variance_def' [IsProbabilityMeasure μ] {X : Ω → ℝ} (hX : Memℒp X 2 μ) :
    variance X μ = μ[X ^ 2] - μ[X] ^ 2 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    X : Ω → Real
    hX : MeasureTheory.Memℒp X 2 μ
    ⊢ Eq (ProbabilityTheory.variance X μ) (HSub.hSub (MeasureTheory.integral μ fun …
  -/
  rw [hX.variance_eq, sub_sq', integral_sub', integral_add']; rotate_left
    /-
      case hf
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ MeasureTheory.Integrable (HPow.hPow X 2) μ
    -/
  · exact hX.integrable_sq
    /-
      🎉 no goals
    -/
    /-
      case hg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ MeasureTheory.Integrable (HPow.hPow (fun x => MeasureTheory.integral μ fun x …
    -/
  · apply integrable_const
    /-
      🎉 no goals
    -/
    /-
      case hf
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ MeasureTheory.Integrable (HAdd.hAdd (HPow.hPow X 2) (HPow.hPow (fun x => Mea …
    -/
  · apply hX.integrable_sq.add
    /-
      case hf
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ MeasureTheory.Integrable (HPow.hPow (fun x => MeasureTheory.integral μ fun x …
    -/
    apply integrable_const
    /-
      🎉 no goals
    -/
    /-
      case hg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ MeasureTheory.Integrable (HMul.hMul (HMul.hMul 2 X) fun x => MeasureTheory.i …
    -/
  · exact ((hX.integrable one_le_two).const_mul 2).mul_const' _
    /-
      🎉 no goals
    -/
  simp only [Pi.pow_apply, integral_const, measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul,
    Pi.mul_apply, Pi.ofNat_apply, Nat.cast_ofNat, integral_mul_right, integral_mul_left]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    X : Ω → Real
    hX : MeasureTheory.Memℒp X 2 μ
    ⊢ Eq (HSub.hSub (HAdd.hAdd (MeasureTheory.integral μ fun a => HPow.hPow (X a)  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem variance_le_expectation_sq [IsProbabilityMeasure μ] {X : Ω → ℝ}
    (hm : AEStronglyMeasurable X μ) : variance X μ ≤ μ[X ^ 2] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    X : Ω → Real
    hm : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ LE.le (ProbabilityTheory.variance X μ) (MeasureTheory.integral μ fun x => HP …
  -/
  by_cases hX : Memℒp X 2 μ
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hm : MeasureTheory.AEStronglyMeasurable X μ
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ LE.le (ProbabilityTheory.variance X μ) (MeasureTheory.integral μ fun x => HP …
    -/
  · rw [variance_def' hX]
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hm : MeasureTheory.AEStronglyMeasurable X μ
      hX : MeasureTheory.Memℒp X 2 μ
      ⊢ LE.le (HSub.hSub (MeasureTheory.integral μ fun x => HPow.hPow X 2 x) (HPow.h …
    -/
    simp only [sq_nonneg, sub_le_self_iff]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    X : Ω → Real
    hm : MeasureTheory.AEStronglyMeasurable X μ
    hX : Not (MeasureTheory.Memℒp X 2 μ)
    ⊢ LE.le (ProbabilityTheory.variance X μ) (MeasureTheory.integral μ fun x => HP …
  -/
  rw [variance, evariance_eq_lintegral_ofReal, ← integral_eq_lintegral_of_nonneg_ae]
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hm : MeasureTheory.AEStronglyMeasurable X μ
      hX : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ LE.le (MeasureTheory.integral μ fun a => HPow.hPow (HSub.hSub (X a) (Measure …
    -/
  · by_cases hint : Integrable X μ; swap
      /-
        case neg
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : Not (MeasureTheory.Integrable X μ)
        ⊢ LE.le (MeasureTheory.integral μ fun a => HPow.hPow (HSub.hSub (X a) (Measure …
      -/
    · simp only [integral_undef hint, Pi.pow_apply, Pi.sub_apply, sub_zero]
      /-
        case neg
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : Not (MeasureTheory.Integrable X μ)
        ⊢ LE.le (MeasureTheory.integral μ fun a => HPow.hPow (X a) 2) (MeasureTheory.i …
      -/
      exact le_rfl
      /-
        🎉 no goals
      -/
      /-
        case pos
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        ⊢ LE.le (MeasureTheory.integral μ fun a => HPow.hPow (HSub.hSub (X a) (Measure …
      -/
    · rw [integral_undef]
        /-
          case pos
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X : Ω → Real
          hm : MeasureTheory.AEStronglyMeasurable X μ
          hX : Not (MeasureTheory.Memℒp X 2 μ)
          hint : MeasureTheory.Integrable X μ
          ⊢ LE.le 0 (MeasureTheory.integral μ fun x => HPow.hPow X 2 x)
        -/
      · exact integral_nonneg fun a => sq_nonneg _
        /-
          🎉 no goals
        -/
      /-
        case pos
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        ⊢ Not (MeasureTheory.Integrable (fun a => HPow.hPow (HSub.hSub (X a) (MeasureT …
      -/
      intro h
      have A : Memℒp (X - fun ω : Ω => μ[X]) 2 μ :=
        (memℒp_two_iff_integrable_sq (hint.aestronglyMeasurable.sub aestronglyMeasurable_const)).2 h
      /-
        case pos
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        h : MeasureTheory.Integrable (fun a => HPow.hPow (HSub.hSub (X a) (MeasureTheo …
        A : MeasureTheory.Memℒp (HSub.hSub X fun ω => MeasureTheory.integral μ fun x = …
        ⊢ False
      -/
      have B : Memℒp (fun _ : Ω => μ[X]) 2 μ := memℒp_const _
      /-
        case pos
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        h : MeasureTheory.Integrable (fun a => HPow.hPow (HSub.hSub (X a) (MeasureTheo …
        A : MeasureTheory.Memℒp (HSub.hSub X fun ω => MeasureTheory.integral μ fun x = …
        B : MeasureTheory.Memℒp (fun x => MeasureTheory.integral μ fun x => X x) 2 μ
        ⊢ False
      -/
      apply hX
      /-
        case pos
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        h : MeasureTheory.Integrable (fun a => HPow.hPow (HSub.hSub (X a) (MeasureTheo …
        A : MeasureTheory.Memℒp (HSub.hSub X fun ω => MeasureTheory.integral μ fun x = …
        B : MeasureTheory.Memℒp (fun x => MeasureTheory.integral μ fun x => X x) 2 μ
        ⊢ MeasureTheory.Memℒp X 2 μ
      -/
      convert A.add B
      /-
        case h.e'_6
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hm : MeasureTheory.AEStronglyMeasurable X μ
        hX : Not (MeasureTheory.Memℒp X 2 μ)
        hint : MeasureTheory.Integrable X μ
        h : MeasureTheory.Integrable (fun a => HPow.hPow (HSub.hSub (X a) (MeasureTheo …
        A : MeasureTheory.Memℒp (HSub.hSub X fun ω => MeasureTheory.integral μ fun x = …
        B : MeasureTheory.Memℒp (fun x => MeasureTheory.integral μ fun x => X x) 2 μ
        ⊢ Eq X (HAdd.hAdd (HSub.hSub X fun ω => MeasureTheory.integral μ fun x => X x) …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case neg.hf
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hm : MeasureTheory.AEStronglyMeasurable X μ
      hX : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun ω => HPow.hPow (HSub.hSub (X ω) (Mea …
    -/
  · exact Eventually.of_forall fun x => sq_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case neg.hfm
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hm : MeasureTheory.AEStronglyMeasurable X μ
      hX : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => HPow.hPow (HSub.hSub (X ω) (Mea …
    -/
  · exact (AEMeasurable.pow_const (hm.aemeasurable.sub_const _) _).aestronglyMeasurable
    /-
      🎉 no goals
    -/


theorem evariance_def' [IsProbabilityMeasure μ] {X : Ω → ℝ} (hX : AEStronglyMeasurable X μ) :
    evariance X μ = (∫⁻ ω, (‖X ω‖₊ ^ 2 :) ∂μ) - ENNReal.ofReal (μ[X] ^ 2) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    X : Ω → Real
    hX : MeasureTheory.AEStronglyMeasurable X μ
    ⊢ Eq (ProbabilityTheory.evariance X μ) (HSub.hSub (MeasureTheory.lintegral μ f …
  -/
  by_cases hℒ : Memℒp X 2 μ
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : MeasureTheory.Memℒp X 2 μ
      ⊢ Eq (ProbabilityTheory.evariance X μ) (HSub.hSub (MeasureTheory.lintegral μ f …
    -/
  · rw [← hℒ.ofReal_variance_eq, variance_def' hℒ, ENNReal.ofReal_sub _ (sq_nonneg _)]
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : MeasureTheory.Memℒp X 2 μ
      ⊢ Eq (HSub.hSub (ENNReal.ofReal (MeasureTheory.integral μ fun x => HPow.hPow X …
    -/
    congr
    /-
      case pos.e_a
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : MeasureTheory.Memℒp X 2 μ
      ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral μ fun x => HPow.hPow X 2 x)) (Mea …
    -/
    rw [lintegral_coe_eq_integral]
      /-
        case pos.e_a
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hX : MeasureTheory.AEStronglyMeasurable X μ
        hℒ : MeasureTheory.Memℒp X 2 μ
        ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral μ fun x => HPow.hPow X 2 x)) (ENN …
      -/
    · congr 2 with ω
      /-
        case pos.e_a.e_r.e_f.h
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hX : MeasureTheory.AEStronglyMeasurable X μ
        hℒ : MeasureTheory.Memℒp X 2 μ
        ω : Ω
        ⊢ Eq (HPow.hPow X 2 ω) ↑(HPow.hPow (NNNorm.nnnorm (X ω)) 2)
      -/
      simp only [Pi.pow_apply, NNReal.coe_pow, coe_nnnorm, Real.norm_eq_abs, Even.pow_abs even_two]
      /-
        🎉 no goals
      -/
      /-
        case pos.e_a.hfi
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X : Ω → Real
        hX : MeasureTheory.AEStronglyMeasurable X μ
        hℒ : MeasureTheory.Memℒp X 2 μ
        ⊢ MeasureTheory.Integrable (fun x => ↑(HPow.hPow (NNNorm.nnnorm (X x)) 2)) μ
      -/
    · exact hℒ.abs.integrable_sq
      /-
        🎉 no goals
      -/
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ Eq (ProbabilityTheory.evariance X μ) (HSub.hSub (MeasureTheory.lintegral μ f …
    -/
  · symm
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ Eq (HSub.hSub (MeasureTheory.lintegral μ fun ω => ↑(HPow.hPow (NNNorm.nnnorm …
    -/
    rw [evariance_eq_top hX hℒ, ENNReal.sub_eq_top_iff]
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ And (Eq (MeasureTheory.lintegral μ fun ω => ↑(HPow.hPow (NNNorm.nnnorm (X ω) …
    -/
    refine ⟨?_, ENNReal.ofReal_ne_top⟩
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : Not (MeasureTheory.Memℒp X 2 μ)
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ↑(HPow.hPow (NNNorm.nnnorm (X ω)) 2)) …
    -/
    rw [Memℒp, not_and] at hℒ
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : MeasureTheory.AEStronglyMeasurable X μ → Not (LT.lt (MeasureTheory.eLpNor …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ↑(HPow.hPow (NNNorm.nnnorm (X ω)) 2)) …
    -/
    specialize hℒ hX
    simp only [eLpNorm_eq_lintegral_rpow_nnnorm two_ne_zero ENNReal.two_ne_top, not_lt, top_le_iff,
      ENNReal.toReal_ofNat, one_div, ENNReal.rpow_eq_top_iff, inv_lt_zero, inv_pos, and_true,
      or_iff_not_imp_left, not_and_or, zero_lt_two] at hℒ
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      hℒ : (Not (Not (Eq (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnn …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ↑(HPow.hPow (NNNorm.nnnorm (X ω)) 2)) …
    -/
    exact mod_cast hℒ fun _ => zero_le_two
    /-
      🎉 no goals
    -/


/-- **Chebyshev's inequality** for `ℝ≥0∞`-valued variance. -/
theorem meas_ge_le_evariance_div_sq {X : Ω → ℝ} (hX : AEStronglyMeasurable X μ) {c : ℝ≥0}
    (hc : c ≠ 0) : μ {ω | ↑c ≤ |X ω - μ[X]|} ≤ evariance X μ / c ^ 2 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Ω → Real
    hX : MeasureTheory.AEStronglyMeasurable X μ
    c : NNReal
    hc : Ne c 0
    ⊢ LE.le (μ (setOf fun ω => LE.le (↑c) (abs (HSub.hSub (X ω) (MeasureTheory.int …
  -/
  have A : (c : ℝ≥0∞) ≠ 0 := by rwa [Ne, ENNReal.coe_eq_zero]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Ω → Real
    hX : MeasureTheory.AEStronglyMeasurable X μ
    c : NNReal
    hc : Ne c 0
    A : Ne (↑c) 0
    ⊢ LE.le (μ (setOf fun ω => LE.le (↑c) (abs (HSub.hSub (X ω) (MeasureTheory.int …
  -/
  have B : AEStronglyMeasurable (fun _ : Ω => μ[X]) μ := aestronglyMeasurable_const
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Ω → Real
    hX : MeasureTheory.AEStronglyMeasurable X μ
    c : NNReal
    hc : Ne c 0
    A : Ne (↑c) 0
    B : MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral μ fun  …
    ⊢ LE.le (μ (setOf fun ω => LE.le (↑c) (abs (HSub.hSub (X ω) (MeasureTheory.int …
  -/
  convert meas_ge_le_mul_pow_eLpNorm μ two_ne_zero ENNReal.two_ne_top (hX.sub B) A using 1
    /-
      case h.e'_3
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      c : NNReal
      hc : Ne c 0
      A : Ne (↑c) 0
      B : MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral μ fun  …
      ⊢ Eq (μ (setOf fun ω => LE.le (↑c) (abs (HSub.hSub (X ω) (MeasureTheory.integr …
    -/
  · congr
    simp only [Pi.sub_apply, ENNReal.coe_le_coe, ← Real.norm_eq_abs, ← coe_nnnorm,
      NNReal.coe_le_coe, ENNReal.ofReal_coe_nnreal]
    /-
      case h.e'_4
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      c : NNReal
      hc : Ne c 0
      A : Ne (↑c) 0
      B : MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral μ fun  …
      ⊢ Eq (HDiv.hDiv (ProbabilityTheory.evariance X μ) (HPow.hPow (↑c) 2)) (HMul.hM …
    -/
  · rw [eLpNorm_eq_lintegral_rpow_nnnorm two_ne_zero ENNReal.two_ne_top]
    simp only [show ENNReal.ofNNReal (c ^ 2) = (ENNReal.ofNNReal c) ^ 2 by norm_cast,
      ENNReal.toReal_ofNat, one_div, Pi.sub_apply]
    /-
      case h.e'_4
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      c : NNReal
      hc : Ne c 0
      A : Ne (↑c) 0
      B : MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral μ fun  …
      ⊢ Eq (HDiv.hDiv (ProbabilityTheory.evariance X μ) (HPow.hPow (↑c) 2)) (HMul.hM …
    -/
    rw [div_eq_mul_inv, ENNReal.inv_pow, mul_comm, ENNReal.rpow_two]
    /-
      case h.e'_4
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Ω → Real
      hX : MeasureTheory.AEStronglyMeasurable X μ
      c : NNReal
      hc : Ne c 0
      A : Ne (↑c) 0
      B : MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral μ fun  …
      ⊢ Eq (HMul.hMul (HPow.hPow (Inv.inv ↑c) 2) (ProbabilityTheory.evariance X μ))  …
    -/
    congr
    simp_rw [← ENNReal.rpow_mul, inv_mul_cancel₀ (two_ne_zero : (2 : ℝ) ≠ 0), ENNReal.rpow_two,
      ENNReal.rpow_one, evariance]


/-- **Chebyshev's inequality**: one can control the deviation probability of a real random variable
from its expectation in terms of the variance. -/
theorem meas_ge_le_variance_div_sq [IsFiniteMeasure μ] {X : Ω → ℝ} (hX : Memℒp X 2 μ) {c : ℝ}
    (hc : 0 < c) : μ {ω | c ≤ |X ω - μ[X]|} ≤ ENNReal.ofReal (variance X μ / c ^ 2) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : Ω → Real
    hX : MeasureTheory.Memℒp X 2 μ
    c : Real
    hc : LT.lt 0 c
    ⊢ LE.le (μ (setOf fun ω => LE.le c (abs (HSub.hSub (X ω) (MeasureTheory.integr …
  -/
  rw [ENNReal.ofReal_div_of_pos (sq_pos_of_ne_zero hc.ne.symm), hX.ofReal_variance_eq]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : Ω → Real
    hX : MeasureTheory.Memℒp X 2 μ
    c : Real
    hc : LT.lt 0 c
    ⊢ LE.le (μ (setOf fun ω => LE.le c (abs (HSub.hSub (X ω) (MeasureTheory.integr …
  -/
  convert @meas_ge_le_evariance_div_sq _ _ _ _ hX.1 c.toNNReal (by simp [hc]) using 1
    /-
      case h.e'_3
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      c : Real
      hc : LT.lt 0 c
      ⊢ Eq (μ (setOf fun ω => LE.le c (abs (HSub.hSub (X ω) (MeasureTheory.integral  …
    -/
  · simp only [Real.coe_toNNReal', max_le_iff, abs_nonneg, and_true]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      c : Real
      hc : LT.lt 0 c
      ⊢ Eq (HDiv.hDiv (ProbabilityTheory.evariance X μ) (ENNReal.ofReal (HPow.hPow c …
    -/
  · rw [ENNReal.ofReal_pow hc.le]
    /-
      case h.e'_4
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : Ω → Real
      hX : MeasureTheory.Memℒp X 2 μ
      c : Real
      hc : LT.lt 0 c
      ⊢ Eq (HDiv.hDiv (ProbabilityTheory.evariance X μ) (HPow.hPow (ENNReal.ofReal c …
    -/
    rfl
    /-
      🎉 no goals
    -/

-- Porting note: supplied `MeasurableSpace Ω` argument of `h` by unification

/-- The variance of the sum of two independent random variables is the sum of the variances. -/
theorem IndepFun.variance_add [IsProbabilityMeasure μ] {X Y : Ω → ℝ} (hX : Memℒp X 2 μ)
    (hY : Memℒp Y 2 μ) (h : IndepFun X Y μ) : variance (X + Y) μ = variance X μ + variance Y μ :=
  calc
    variance (X + Y) μ = μ[fun a => X a ^ 2 + Y a ^ 2 + 2 * X a * Y a] - μ[X + Y] ^ 2 := by
      /-
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X Y : Ω → Real
        hX : MeasureTheory.Memℒp X 2 μ
        hY : MeasureTheory.Memℒp Y 2 μ
        h : ProbabilityTheory.IndepFun X Y μ
        ⊢ Eq (ProbabilityTheory.variance (HAdd.hAdd X Y) μ) (HSub.hSub (MeasureTheory. …
      -/
      simp [variance_def' (hX.add hY), add_sq']
      /-
        🎉 no goals
      -/
    _ = μ[X ^ 2] + μ[Y ^ 2] + (2 : ℝ) * μ[X * Y] - (μ[X] + μ[Y]) ^ 2 := by
      /-
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X Y : Ω → Real
        hX : MeasureTheory.Memℒp X 2 μ
        hY : MeasureTheory.Memℒp Y 2 μ
        h : ProbabilityTheory.IndepFun X Y μ
        ⊢ Eq (HSub.hSub (MeasureTheory.integral μ fun x => (fun a => HAdd.hAdd (HAdd.h …
      -/
      simp only [Pi.add_apply, Pi.pow_apply, Pi.mul_apply, mul_assoc]
      /-
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X Y : Ω → Real
        hX : MeasureTheory.Memℒp X 2 μ
        hY : MeasureTheory.Memℒp Y 2 μ
        h : ProbabilityTheory.IndepFun X Y μ
        ⊢ Eq (HSub.hSub (MeasureTheory.integral μ fun x => HAdd.hAdd (HAdd.hAdd (HPow. …
      -/
      rw [integral_add, integral_add, integral_add, integral_mul_left]
        /-
          case hf
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable X μ
        -/
      · exact hX.integrable one_le_two
        /-
          🎉 no goals
        -/
        /-
          case hg
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable Y μ
        -/
      · exact hY.integrable one_le_two
        /-
          🎉 no goals
        -/
        /-
          case hf
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable (fun a => HPow.hPow (X a) 2) μ
        -/
      · exact hX.integrable_sq
        /-
          🎉 no goals
        -/
        /-
          case hg
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable (fun a => HPow.hPow (Y a) 2) μ
        -/
      · exact hY.integrable_sq
        /-
          🎉 no goals
        -/
        /-
          case hf
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable (fun x => HAdd.hAdd (HPow.hPow (X x) 2) (HPow.hPow  …
        -/
      · exact hX.integrable_sq.add hY.integrable_sq
        /-
          🎉 no goals
        -/
        /-
          case hg
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable (fun x => HMul.hMul 2 (HMul.hMul (X x) (Y x))) μ
        -/
      · apply Integrable.const_mul
        /-
          case hg.h
          Ω : Type u_1
          m : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝ : MeasureTheory.IsProbabilityMeasure μ
          X Y : Ω → Real
          hX : MeasureTheory.Memℒp X 2 μ
          hY : MeasureTheory.Memℒp Y 2 μ
          h : ProbabilityTheory.IndepFun X Y μ
          ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (X x) (Y x)) μ
        -/
        exact h.integrable_mul (hX.integrable one_le_two) (hY.integrable one_le_two)
        /-
          🎉 no goals
        -/
    _ = μ[X ^ 2] + μ[Y ^ 2] + 2 * (μ[X] * μ[Y]) - (μ[X] + μ[Y]) ^ 2 := by
      /-
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X Y : Ω → Real
        hX : MeasureTheory.Memℒp X 2 μ
        hY : MeasureTheory.Memℒp Y 2 μ
        h : ProbabilityTheory.IndepFun X Y μ
        ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral μ fun x => HPow. …
      -/
      congr
      /-
        case e_a.e_a.e_a
        Ω : Type u_1
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        X Y : Ω → Real
        hX : MeasureTheory.Memℒp X 2 μ
        hY : MeasureTheory.Memℒp Y 2 μ
        h : ProbabilityTheory.IndepFun X Y μ
        ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul X Y x) (HMul.hMul (MeasureTh …
      -/
      exact h.integral_mul_of_integrable (hX.integrable one_le_two) (hY.integrable one_le_two)
      /-
        🎉 no goals
      -/
                                          /-
                                            Ω : Type u_1
                                            m : MeasurableSpace Ω
                                            μ : MeasureTheory.Measure Ω
                                            inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                            X Y : Ω → Real
                                            hX : MeasureTheory.Memℒp X 2 μ
                                            hY : MeasureTheory.Memℒp Y 2 μ
                                            h : ProbabilityTheory.IndepFun X Y μ
                                            ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral μ fun x => HPow. …
                                          -/
    _ = variance X μ + variance Y μ := by simp only [variance_def', hX, hY, Pi.pow_apply]; ring
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/

-- Porting note: supplied `MeasurableSpace Ω` argument of `hs`, `h` by unification

/-- The variance of a finite sum of pairwise independent random variables is the sum of the
variances. -/
theorem IndepFun.variance_sum [IsProbabilityMeasure μ] {ι : Type*} {X : ι → Ω → ℝ}
    {s : Finset ι} (hs : ∀ i ∈ s, Memℒp (X i) 2 μ)
    (h : Set.Pairwise ↑s fun i j => IndepFun (X i) (X j) μ) :
    variance (∑ i ∈ s, X i) μ = ∑ i ∈ s, variance (X i) μ := by
  classical
  induction' s using Finset.induction_on with k s ks IH
  · simp only [Finset.sum_empty, variance_zero]
  rw [variance_def' (memℒp_finset_sum' _ hs), sum_insert ks, sum_insert ks]
  simp only [add_sq']
  calc
    μ[(X k ^ 2 + (∑ i ∈ s, X i) ^ 2 + 2 * X k * ∑ i ∈ s, X i : Ω → ℝ)] - μ[X k + ∑ i ∈ s, X i] ^ 2 =
        μ[X k ^ 2] + μ[(∑ i ∈ s, X i) ^ 2] + μ[2 * X k * ∑ i ∈ s, X i] -
          (μ[X k] + μ[∑ i ∈ s, X i]) ^ 2 := by
      rw [integral_add', integral_add', integral_add']
      · exact Memℒp.integrable one_le_two (hs _ (mem_insert_self _ _))
      · apply integrable_finset_sum' _ fun i hi => ?_
        exact Memℒp.integrable one_le_two (hs _ (mem_insert_of_mem hi))
      · exact Memℒp.integrable_sq (hs _ (mem_insert_self _ _))
      · apply Memℒp.integrable_sq
        exact memℒp_finset_sum' _ fun i hi => hs _ (mem_insert_of_mem hi)
      · apply Integrable.add
        · exact Memℒp.integrable_sq (hs _ (mem_insert_self _ _))
        · apply Memℒp.integrable_sq
          exact memℒp_finset_sum' _ fun i hi => hs _ (mem_insert_of_mem hi)
      · rw [mul_assoc]
        apply Integrable.const_mul _ (2 : ℝ)
        rw [mul_sum, sum_fn]
        apply integrable_finset_sum _ fun i hi => ?_
        apply IndepFun.integrable_mul _ (Memℒp.integrable one_le_two (hs _ (mem_insert_self _ _)))
          (Memℒp.integrable one_le_two (hs _ (mem_insert_of_mem hi)))
        apply h (mem_insert_self _ _) (mem_insert_of_mem hi)
        exact fun hki => ks (hki.symm ▸ hi)
    _ = variance (X k) μ + variance (∑ i ∈ s, X i) μ +
        (μ[2 * X k * ∑ i ∈ s, X i] - 2 * μ[X k] * μ[∑ i ∈ s, X i]) := by
      rw [variance_def' (hs _ (mem_insert_self _ _)),
        variance_def' (memℒp_finset_sum' _ fun i hi => hs _ (mem_insert_of_mem hi))]
      ring
    _ = variance (X k) μ + variance (∑ i ∈ s, X i) μ := by
      simp_rw [Pi.mul_apply, Pi.ofNat_apply, Nat.cast_ofNat, sum_apply, mul_sum, mul_assoc,
        add_right_eq_self]
      rw [integral_finset_sum s fun i hi => ?_]; swap
      · apply Integrable.const_mul _ (2 : ℝ)
        apply IndepFun.integrable_mul _ (Memℒp.integrable one_le_two (hs _ (mem_insert_self _ _)))
          (Memℒp.integrable one_le_two (hs _ (mem_insert_of_mem hi)))
        apply h (mem_insert_self _ _) (mem_insert_of_mem hi)
        exact fun hki => ks (hki.symm ▸ hi)
      rw [integral_finset_sum s fun i hi =>
          Memℒp.integrable one_le_two (hs _ (mem_insert_of_mem hi)),
        mul_sum, mul_sum, ← sum_sub_distrib]
      apply Finset.sum_eq_zero fun i hi => ?_
      rw [integral_mul_left, IndepFun.integral_mul', sub_self]
      · apply h (mem_insert_self _ _) (mem_insert_of_mem hi)
        exact fun hki => ks (hki.symm ▸ hi)
      · exact Memℒp.aestronglyMeasurable (hs _ (mem_insert_self _ _))
      · exact Memℒp.aestronglyMeasurable (hs _ (mem_insert_of_mem hi))
    _ = variance (X k) μ + ∑ i ∈ s, variance (X i) μ := by
      rw [IH (fun i hi => hs i (mem_insert_of_mem hi))
          (h.mono (by simp only [coe_insert, Set.subset_insert]))]


/-- **The Bhatia-Davis inequality on variance**

The variance of a random variable `X` satisfying `a ≤ X ≤ b` almost everywhere is at most
`(b - 𝔼 X) * (𝔼 X - a)`. -/
lemma variance_le_sub_mul_sub [IsProbabilityMeasure μ] {a b : ℝ} {X : Ω → ℝ}
    (h : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b) (hX : AEMeasurable X μ) :
    variance X μ ≤ (b - μ[X]) * (μ[X] - a) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    a b : Real
    X : Ω → Real
    h : Filter.Eventually (fun ω => Membership.mem (Set.Icc a b) (X ω)) (MeasureTh …
    hX : AEMeasurable X μ
    ⊢ LE.le (ProbabilityTheory.variance X μ) (HMul.hMul (HSub.hSub b (MeasureTheor …
  -/
  have ha : ∀ᵐ ω ∂μ, a ≤ X ω := h.mono fun ω h => h.1
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    a b : Real
    X : Ω → Real
    h : Filter.Eventually (fun ω => Membership.mem (Set.Icc a b) (X ω)) (MeasureTh …
    hX : AEMeasurable X μ
    ha : Filter.Eventually (fun ω => LE.le a (X ω)) (MeasureTheory.ae μ)
    ⊢ LE.le (ProbabilityTheory.variance X μ) (HMul.hMul (HSub.hSub b (MeasureTheor …
  -/
  have hb : ∀ᵐ ω ∂μ, X ω ≤ b := h.mono fun ω h => h.2
  have hX_int₂ : Integrable (fun ω ↦ -X ω ^ 2) μ :=
    (memℒp_of_bounded h hX.aestronglyMeasurable 2).integrable_sq.neg
  have hX_int₁ : Integrable (fun ω ↦ (a + b) * X ω) μ :=
    ((integrable_const (max |a| |b|)).mono' hX.aestronglyMeasurable
      (by filter_upwards [ha, hb] with ω using abs_le_max_abs_abs)).const_mul (a + b)
  have h0 : 0 ≤ - μ[X ^ 2] + (a + b) * μ[X] - a * b :=
    calc
      _ ≤ ∫ ω, (b - X ω) * (X ω - a) ∂μ := by
        apply integral_nonneg_of_ae
        filter_upwards [ha, hb] with ω ha' hb'
        exact mul_nonneg (by linarith : 0 ≤ b - X ω) (by linarith : 0 ≤ X ω - a)
      _ = ∫ ω, - X ω ^ 2 + (a + b) * X ω - a * b ∂μ :=
        integral_congr_ae <| ae_of_all μ fun ω ↦ by ring
      _ = ∫ ω, - X ω ^ 2 + (a + b) * X ω ∂μ - ∫ _, a * b ∂μ :=
        integral_sub (hX_int₂.add hX_int₁) (integrable_const (a * b))
      _ = ∫ ω, - X ω ^ 2 + (a + b) * X ω ∂μ - a * b := by simp
      _ = - μ[X ^ 2] + (a + b) * μ[X] - a * b := by
        simp [← integral_neg, ← integral_mul_left, integral_add hX_int₂ hX_int₁]
  calc
    _ ≤ (a + b) * μ[X] - a * b - μ[X] ^ 2 := by
      rw [variance_def' (memℒp_of_bounded h hX.aestronglyMeasurable 2)]
      linarith
    _ = (b - μ[X]) * (μ[X] - a) := by ring


/-- **Popoviciu's inequality on variance**

The variance of a random variable `X` satisfying `a ≤ X ≤ b` almost everywhere is at most
`((b - a) / 2) ^ 2`. -/
lemma variance_le_sq_of_bounded [IsProbabilityMeasure μ] {a b : ℝ} {X : Ω → ℝ}
    (h : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b) (hX : AEMeasurable X μ) :
    variance X μ ≤ ((b - a) / 2) ^ 2 :=
  calc
    _ ≤ (b - μ[X]) * (μ[X] - a) := variance_le_sub_mul_sub h hX
                                                           /-
                                                             Ω : Type u_1
                                                             m : MeasurableSpace Ω
                                                             μ : MeasureTheory.Measure Ω
                                                             inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                                             a b : Real
                                                             X : Ω → Real
                                                             h : Filter.Eventually (fun ω => Membership.mem (Set.Icc a b) (X ω)) (MeasureTh …
                                                             hX : AEMeasurable X μ
                                                             ⊢ Eq (HMul.hMul (HSub.hSub b (MeasureTheory.integral μ fun x => X x)) (HSub.hS …
                                                           -/
    _ = ((b - a) / 2) ^ 2 - (μ[X] - (b + a) / 2) ^ 2 := by ring
                                                           /-
                                                             🎉 no goals
                                                           -/
    _ ≤ ((b - a) / 2) ^ 2 := sub_le_self _ (sq_nonneg _)


