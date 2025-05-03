theorem pow_mul_meas_ge_le_eLpNorm (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf : AEStronglyMeasurable f μ) (ε : ℝ≥0∞) :
    (ε * μ { x | ε ≤ (‖f x‖₊ : ℝ≥0∞) ^ p.toReal }) ^ (1 / p.toReal) ≤ eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    ⊢ LE.le (HPow.hPow (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hPow (↑(NNNor …
  -/
  rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_ne_zero hp_ne_top]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    ⊢ LE.le (HPow.hPow (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hPow (↑(NNNor …
  -/
  gcongr
  /-
    case h₁
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hPow (↑(NNNorm.nnnorm (f …
  -/
  exact mul_meas_ge_le_lintegral₀ (hf.ennnorm.pow_const _) ε
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias pow_mul_meas_ge_le_snorm := pow_mul_meas_ge_le_eLpNorm


theorem mul_meas_ge_le_pow_eLpNorm (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf : AEStronglyMeasurable f μ) (ε : ℝ≥0∞) :
    ε * μ { x | ε ≤ (‖f x‖₊ : ℝ≥0∞) ^ p.toReal } ≤ eLpNorm f p μ ^ p.toReal := by
  have : 1 / p.toReal * p.toReal = 1 := by
    refine one_div_mul_cancel ?_
    rw [Ne, ENNReal.toReal_eq_zero_iff]
    exact not_or_intro hp_ne_zero hp_ne_top
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    this : Eq (HMul.hMul (HDiv.hDiv 1 p.toReal) p.toReal) 1
    ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hPow (↑(NNNorm.nnnorm (f …
  -/
  rw [← ENNReal.rpow_one (ε * μ { x | ε ≤ (‖f x‖₊ : ℝ≥0∞) ^ p.toReal }), ← this, ENNReal.rpow_mul]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    this : Eq (HMul.hMul (HDiv.hDiv 1 p.toReal) p.toReal) 1
    ⊢ LE.le (HPow.hPow (HPow.hPow (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hP …
  -/
  gcongr
  /-
    case h₁
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    this : Eq (HMul.hMul (HDiv.hDiv 1 p.toReal) p.toReal) 1
    ⊢ LE.le (HPow.hPow (HMul.hMul ε (μ (setOf fun x => LE.le ε (HPow.hPow (↑(NNNor …
  -/
  exact pow_mul_meas_ge_le_eLpNorm μ hp_ne_zero hp_ne_top hf ε
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias mul_meas_ge_le_pow_snorm := mul_meas_ge_le_pow_eLpNorm


/-- A version of Chebyshev-Markov's inequality using Lp-norms. -/
theorem mul_meas_ge_le_pow_eLpNorm' (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf : AEStronglyMeasurable f μ) (ε : ℝ≥0∞) :
    ε ^ p.toReal * μ { x | ε ≤ ‖f x‖₊ } ≤ eLpNorm f p μ ^ p.toReal := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    ⊢ LE.le (HMul.hMul (HPow.hPow ε p.toReal) (μ (setOf fun x => LE.le ε ↑(NNNorm. …
  -/
  convert mul_meas_ge_le_pow_eLpNorm μ hp_ne_zero hp_ne_top hf (ε ^ p.toReal) using 4
  /-
    case h.e'_3.h.e'_6.h.e'_6.h.e'_2
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    ⊢ Eq (fun x => LE.le ε ↑(NNNorm.nnnorm (f x))) fun x => LE.le (HPow.hPow ε p.t …
  -/
  ext x
  /-
    case h.e'_3.h.e'_6.h.e'_6.h.e'_2.h.a
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    x : α
    ⊢ Iff (LE.le ε ↑(NNNorm.nnnorm (f x))) (LE.le (HPow.hPow ε p.toReal) (HPow.hPo …
  -/
  rw [ENNReal.rpow_le_rpow_iff (ENNReal.toReal_pos hp_ne_zero hp_ne_top)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias mul_meas_ge_le_pow_snorm' := mul_meas_ge_le_pow_eLpNorm'


theorem meas_ge_le_mul_pow_eLpNorm (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf : AEStronglyMeasurable f μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    μ { x | ε ≤ ‖f x‖₊ } ≤ ε⁻¹ ^ p.toReal * eLpNorm f p μ ^ p.toReal := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ LE.le (μ (setOf fun x => LE.le ε ↑(NNNorm.nnnorm (f x)))) (HMul.hMul (HPow.h …
  -/
  by_cases h : ε = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      f : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : MeasureTheory.AEStronglyMeasurable f μ
      ε : ENNReal
      hε : Ne ε 0
      h : Eq ε Top.top
      ⊢ LE.le (μ (setOf fun x => LE.le ε ↑(NNNorm.nnnorm (f x)))) (HMul.hMul (HPow.h …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    hε : Ne ε 0
    h : Not (Eq ε Top.top)
    ⊢ LE.le (μ (setOf fun x => LE.le ε ↑(NNNorm.nnnorm (f x)))) (HMul.hMul (HPow.h …
  -/
  have hεpow : ε ^ p.toReal ≠ 0 := (ENNReal.rpow_pos (pos_iff_ne_zero.2 hε) h).ne.symm
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    hε : Ne ε 0
    h : Not (Eq ε Top.top)
    hεpow : Ne (HPow.hPow ε p.toReal) 0
    ⊢ LE.le (μ (setOf fun x => LE.le ε ↑(NNNorm.nnnorm (f x)))) (HMul.hMul (HPow.h …
  -/
  have hεpow' : ε ^ p.toReal ≠ ∞ := ENNReal.rpow_ne_top_of_nonneg ENNReal.toReal_nonneg h
  rw [ENNReal.inv_rpow, ← ENNReal.mul_le_mul_left hεpow hεpow', ← mul_assoc,
    ENNReal.mul_inv_cancel hεpow hεpow', one_mul]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ε : ENNReal
    hε : Ne ε 0
    h : Not (Eq ε Top.top)
    hεpow : Ne (HPow.hPow ε p.toReal) 0
    hεpow' : Ne (HPow.hPow ε p.toReal) Top.top
    ⊢ LE.le (HMul.hMul (HPow.hPow ε p.toReal) (μ (setOf fun x => LE.le ε ↑(NNNorm. …
  -/
  exact mul_meas_ge_le_pow_eLpNorm' μ hp_ne_zero hp_ne_top hf ε
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias meas_ge_le_mul_pow_snorm := meas_ge_le_mul_pow_eLpNorm


theorem Memℒp.meas_ge_lt_top' {μ : Measure α} (hℒp : Memℒp f p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    μ { x | ε ≤ ‖f x‖₊ } < ∞ := by
  apply (meas_ge_le_mul_pow_eLpNorm μ hp_ne_zero hp_ne_top hℒp.aestronglyMeasurable hε).trans_lt
    (ENNReal.mul_lt_top ?_ ?_)
    /-
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : α → E
      μ : MeasureTheory.Measure α
      hℒp : MeasureTheory.Memℒp f p μ
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      ⊢ LT.lt (HPow.hPow (Inv.inv ε) p.toReal) Top.top
    -/
  · simp [hε, lt_top_iff_ne_top]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : α → E
      μ : MeasureTheory.Measure α
      hℒp : MeasureTheory.Memℒp f p μ
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      ⊢ LT.lt (HPow.hPow (MeasureTheory.eLpNorm f p μ) p.toReal) Top.top
    -/
  · simp [hℒp.eLpNorm_lt_top.ne, lt_top_iff_ne_top]
    /-
      🎉 no goals
    -/


theorem Memℒp.meas_ge_lt_top {μ : Measure α} (hℒp : Memℒp f p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) {ε : ℝ≥0} (hε : ε ≠ 0) :
    μ { x | ε ≤ ‖f x‖₊ } < ∞ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : α → E
    μ : MeasureTheory.Measure α
    hℒp : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ε : NNReal
    hε : Ne ε 0
    ⊢ LT.lt (μ (setOf fun x => LE.le ε (NNNorm.nnnorm (f x)))) Top.top
  -/
  simp_rw [← ENNReal.coe_le_coe]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : α → E
    μ : MeasureTheory.Measure α
    hℒp : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ε : NNReal
    hε : Ne ε 0
    ⊢ LT.lt (μ (setOf fun x => LE.le ↑ε ↑(NNNorm.nnnorm (f x)))) Top.top
  -/
  apply hℒp.meas_ge_lt_top' hp_ne_zero hp_ne_top (by simp [hε])
  /-
    🎉 no goals
  -/


