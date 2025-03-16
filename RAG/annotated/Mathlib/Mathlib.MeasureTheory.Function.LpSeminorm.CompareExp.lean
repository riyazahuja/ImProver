theorem eLpNorm'_le_eLpNorm'_mul_rpow_measure_univ {p q : ℝ} (hp0_lt : 0 < p) (hpq : p ≤ q)
    (hf : AEStronglyMeasurable f μ) :
    eLpNorm' f p μ ≤ eLpNorm' f q μ * μ Set.univ ^ (1 / p - 1 / q) := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
  -/
  have hq0_lt : 0 < q := lt_of_lt_of_le hp0_lt hpq
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
  -/
  by_cases hpq_eq : p = q
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : Real
      hp0_lt : LT.lt 0 p
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hq0_lt : LT.lt 0 q
      hpq_eq : Eq p q
      ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
    -/
  · rw [hpq_eq, sub_self, ENNReal.rpow_zero, mul_one]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
  -/
  have hpq : p < q := lt_of_le_of_ne hpq hpq_eq
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq✝ : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    hpq : LT.lt p q
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
  -/
  let g := fun _ : α => (1 : ℝ≥0∞)
  have h_rw : (∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ p ∂μ) = ∫⁻ a, ((‖f a‖₊ : ℝ≥0∞) * g a) ^ p ∂μ :=
    lintegral_congr fun a => by simp [g]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq✝ : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    hpq : LT.lt p q
    g : α → ENNReal := fun x => 1
    h_rw : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNorm' f q  …
  -/
  repeat' rw [eLpNorm'_eq_lintegral_nnnorm]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq✝ : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    hpq : LT.lt p q
    g : α → ENNReal := fun x => 1
    h_rw : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnn …
  -/
  rw [h_rw]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq✝ : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    hpq : LT.lt p q
    g : α → ENNReal := fun x => 1
    h_rw : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul (↑ …
  -/
  let r := p * q / (q - p)
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq✝ : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hq0_lt : LT.lt 0 q
    hpq_eq : Not (Eq p q)
    hpq : LT.lt p q
    g : α → ENNReal := fun x => 1
    h_rw : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
    r : Real := HDiv.hDiv (HMul.hMul p q) (HSub.hSub q p)
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul (↑ …
  -/
  have hpqr : 1 / p = 1 / q + 1 / r := by field_simp [r, hp0_lt.ne', hq0_lt.ne']
  calc
    (∫⁻ a : α, (↑‖f a‖₊ * g a) ^ p ∂μ) ^ (1 / p) ≤
        (∫⁻ a : α, ↑‖f a‖₊ ^ q ∂μ) ^ (1 / q) * (∫⁻ a : α, g a ^ r ∂μ) ^ (1 / r) :=
      ENNReal.lintegral_Lp_mul_le_Lq_mul_Lr hp0_lt hpq hpqr μ hf.ennnorm aemeasurable_const
    _ = (∫⁻ a : α, ↑‖f a‖₊ ^ q ∂μ) ^ (1 / q) * μ Set.univ ^ (1 / p - 1 / q) := by
      rw [hpqr]; simp [r, g]


@[deprecated (since := "2024-07-27")]
alias snorm'_le_snorm'_mul_rpow_measure_univ := eLpNorm'_le_eLpNorm'_mul_rpow_measure_univ


theorem eLpNorm'_le_eLpNormEssSup_mul_rpow_measure_univ {q : ℝ} (hq_pos : 0 < q) :
    eLpNorm' f q μ ≤ eLpNormEssSup f μ * μ Set.univ ^ (1 / q) := by
  have h_le : (∫⁻ a : α, (‖f a‖₊ : ℝ≥0∞) ^ q ∂μ) ≤ ∫⁻ _ : α, eLpNormEssSup f μ ^ q ∂μ := by
    refine lintegral_mono_ae ?_
    have h_nnnorm_le_eLpNorm_ess_sup := coe_nnnorm_ae_le_eLpNormEssSup f μ
    exact h_nnnorm_le_eLpNorm_ess_sup.mono fun x hx => by gcongr
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    q : Real
    hq_pos : LT.lt 0 q
    h_le : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f …
    ⊢ LE.le (MeasureTheory.eLpNorm' f q μ) (HMul.hMul (MeasureTheory.eLpNormEssSup …
  -/
  rw [eLpNorm', ← ENNReal.rpow_one (eLpNormEssSup f μ)]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    q : Real
    hq_pos : LT.lt 0 q
    h_le : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm  …
  -/
  nth_rw 2 [← mul_inv_cancel₀ (ne_of_lt hq_pos).symm]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    q : Real
    hq_pos : LT.lt 0 q
    h_le : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm  …
  -/
  rw [ENNReal.rpow_mul, one_div, ← ENNReal.mul_rpow_of_nonneg _ _ (by simp [hq_pos.le] : 0 ≤ q⁻¹)]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    q : Real
    hq_pos : LT.lt 0 q
    h_le : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm  …
  -/
  gcongr
  /-
    case h₁
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    q : Real
    hq_pos : LT.lt 0 q
    h_le : LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm (f a)) q) ( …
  -/
  rwa [lintegral_const] at h_le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_le_snormEssSup_mul_rpow_measure_univ := eLpNorm'_le_eLpNormEssSup_mul_rpow_measure_univ


theorem eLpNorm_le_eLpNorm_mul_rpow_measure_univ {p q : ℝ≥0∞} (hpq : p ≤ q)
    (hf : AEStronglyMeasurable f μ) :
    eLpNorm f p μ ≤ eLpNorm f q μ * μ Set.univ ^ (1 / p.toReal - 1 / q.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
    -/
  · simp [hp0, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  rw [← Ne] at hp0
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  have hp0_lt : 0 < p := lt_of_le_of_ne (zero_le _) hp0.symm
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  have hq0_lt : 0 < q := lt_of_lt_of_le hp0_lt hpq
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  by_cases hq_top : q = ∞
  · simp only [hq_top, _root_.div_zero, one_div, ENNReal.top_toReal, sub_zero, eLpNorm_exponent_top,
      GroupWithZero.inv_zero]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNormEssSup  …
    -/
    by_cases hp_top : p = ∞
    · simp only [hp_top, ENNReal.rpow_zero, mul_one, ENNReal.top_toReal, sub_zero,
        GroupWithZero.inv_zero, eLpNorm_exponent_top]
      /-
        case pos
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        μ : MeasureTheory.Measure α
        f : α → E
        p q : ENNReal
        hpq : LE.le p q
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hp0 : Ne p 0
        hp0_lt : LT.lt 0 p
        hq0_lt : LT.lt 0 q
        hq_top : Eq q Top.top
        hp_top : Eq p Top.top
        ⊢ LE.le (MeasureTheory.eLpNormEssSup f μ) (MeasureTheory.eLpNormEssSup f μ)
      -/
      exact le_rfl
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      hp_top : Not (Eq p Top.top)
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNormEssSup  …
    -/
    rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      hp_top : Not (Eq p Top.top)
      ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (HMul.hMul (MeasureTheory.eLpNor …
    -/
    have hp_pos : 0 < p.toReal := ENNReal.toReal_pos hp0_lt.ne' hp_top
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (HMul.hMul (MeasureTheory.eLpNor …
    -/
    refine (eLpNorm'_le_eLpNormEssSup_mul_rpow_measure_univ hp_pos).trans (le_of_eq ?_)
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      ⊢ Eq (HMul.hMul (MeasureTheory.eLpNormEssSup f μ) (HPow.hPow (μ Set.univ) (HDi …
    -/
    congr
    /-
      case neg.e_a.e_a
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : ENNReal
      hpq : LE.le p q
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hp0 : Ne p 0
      hp0_lt : LT.lt 0 p
      hq0_lt : LT.lt 0 q
      hq_top : Eq q Top.top
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      ⊢ Eq (HDiv.hDiv 1 p.toReal) (Inv.inv p.toReal)
    -/
    exact one_div _
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    hq_top : Not (Eq q Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  have hp_lt_top : p < ∞ := hpq.trans_lt (lt_top_iff_ne_top.mpr hq_top)
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    hq_top : Not (Eq q Top.top)
    hp_lt_top : LT.lt p Top.top
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  have hp_pos : 0 < p.toReal := ENNReal.toReal_pos hp0_lt.ne' hp_lt_top.ne
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    hq_top : Not (Eq q Top.top)
    hp_lt_top : LT.lt p Top.top
    hp_pos : LT.lt 0 p.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (MeasureTheory.eLpNorm f q μ) …
  -/
  rw [eLpNorm_eq_eLpNorm' hp0_lt.ne.symm hp_lt_top.ne, eLpNorm_eq_eLpNorm' hq0_lt.ne.symm hq_top]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    hq_top : Not (Eq q Top.top)
    hp_lt_top : LT.lt p Top.top
    hp_pos : LT.lt 0 p.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (HMul.hMul (MeasureTheory.eLpNor …
  -/
  have hpq_real : p.toReal ≤ q.toReal := ENNReal.toReal_mono hq_top hpq
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : ENNReal
    hpq : LE.le p q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hp0 : Ne p 0
    hp0_lt : LT.lt 0 p
    hq0_lt : LT.lt 0 q
    hq_top : Not (Eq q Top.top)
    hp_lt_top : LT.lt p Top.top
    hp_pos : LT.lt 0 p.toReal
    hpq_real : LE.le p.toReal q.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (HMul.hMul (MeasureTheory.eLpNor …
  -/
  exact eLpNorm'_le_eLpNorm'_mul_rpow_measure_univ hp_pos hpq_real hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_mul_rpow_measure_univ := eLpNorm_le_eLpNorm_mul_rpow_measure_univ


theorem eLpNorm'_le_eLpNorm'_of_exponent_le {p q : ℝ} (hp0_lt : 0 < p)
    (hpq : p ≤ q) (μ : Measure α) [IsProbabilityMeasure μ] (hf : AEStronglyMeasurable f μ) :
    eLpNorm' f p μ ≤ eLpNorm' f q μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq : LE.le p q
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (MeasureTheory.eLpNorm' f q μ)
  -/
  have h_le_μ := eLpNorm'_le_eLpNorm'_mul_rpow_measure_univ hp0_lt hpq hf
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    p q : Real
    hp0_lt : LT.lt 0 p
    hpq : LE.le p q
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_le_μ : LE.le (MeasureTheory.eLpNorm' f p μ) (HMul.hMul (MeasureTheory.eLpNor …
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (MeasureTheory.eLpNorm' f q μ)
  -/
  rwa [measure_univ, ENNReal.one_rpow, mul_one] at h_le_μ
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_le_snorm'_of_exponent_le := eLpNorm'_le_eLpNorm'_of_exponent_le


theorem eLpNorm'_le_eLpNormEssSup {q : ℝ} (hq_pos : 0 < q) [IsProbabilityMeasure μ] :
    eLpNorm' f q μ ≤ eLpNormEssSup f μ :=
                                                                        /-
                                                                          α : Type u_1
                                                                          E : Type u_2
                                                                          m : MeasurableSpace α
                                                                          inst✝¹ : NormedAddCommGroup E
                                                                          μ : MeasureTheory.Measure α
                                                                          f : α → E
                                                                          q : Real
                                                                          hq_pos : LT.lt 0 q
                                                                          inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                                                          ⊢ Eq (HMul.hMul (MeasureTheory.eLpNormEssSup f μ) (HPow.hPow (μ Set.univ) (HDi …
                                                                        -/
  (eLpNorm'_le_eLpNormEssSup_mul_rpow_measure_univ hq_pos).trans_eq (by simp [measure_univ])
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-07-27")]
alias snorm'_le_snormEssSup := eLpNorm'_le_eLpNormEssSup


theorem eLpNorm_le_eLpNorm_of_exponent_le {p q : ℝ≥0∞} (hpq : p ≤ q) [IsProbabilityMeasure μ]
    (hf : AEStronglyMeasurable f μ) : eLpNorm f p μ ≤ eLpNorm f q μ :=
                                                                        /-
                                                                          α : Type u_1
                                                                          E : Type u_2
                                                                          m : MeasurableSpace α
                                                                          inst✝¹ : NormedAddCommGroup E
                                                                          μ : MeasureTheory.Measure α
                                                                          f : α → E
                                                                          p q : ENNReal
                                                                          hpq : LE.le p q
                                                                          inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                                                          hf : MeasureTheory.AEStronglyMeasurable f μ
                                                                          ⊢ Eq (HMul.hMul (MeasureTheory.eLpNorm f q μ) (HPow.hPow (μ Set.univ) (HSub.hS …
                                                                        -/
  (eLpNorm_le_eLpNorm_mul_rpow_measure_univ hpq hf).trans (le_of_eq (by simp [measure_univ]))
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_of_exponent_le := eLpNorm_le_eLpNorm_of_exponent_le


theorem eLpNorm'_lt_top_of_eLpNorm'_lt_top_of_exponent_le {p q : ℝ} [IsFiniteMeasure μ]
    (hf : AEStronglyMeasurable f μ) (hfq_lt_top : eLpNorm' f q μ < ∞) (hp_nonneg : 0 ≤ p)
    (hpq : p ≤ q) : eLpNorm' f p μ < ∞ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
    hp_nonneg : LE.le 0 p
    hpq : LE.le p q
    ⊢ LT.lt (MeasureTheory.eLpNorm' f p μ) Top.top
  -/
  rcases le_or_lt p 0 with hp_nonpos | hp_pos
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
      hp_nonneg : LE.le 0 p
      hpq : LE.le p q
      hp_nonpos : LE.le p 0
      ⊢ LT.lt (MeasureTheory.eLpNorm' f p μ) Top.top
    -/
  · rw [le_antisymm hp_nonpos hp_nonneg]
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f : α → E
      p q : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
      hp_nonneg : LE.le 0 p
      hpq : LE.le p q
      hp_nonpos : LE.le p 0
      ⊢ LT.lt (MeasureTheory.eLpNorm' f 0 μ) Top.top
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : α → E
    p q : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
    hp_nonneg : LE.le 0 p
    hpq : LE.le p q
    hp_pos : LT.lt 0 p
    ⊢ LT.lt (MeasureTheory.eLpNorm' f p μ) Top.top
  -/
  have hq_pos : 0 < q := lt_of_lt_of_le hp_pos hpq
  calc
    eLpNorm' f p μ ≤ eLpNorm' f q μ * μ Set.univ ^ (1 / p - 1 / q) :=
      eLpNorm'_le_eLpNorm'_mul_rpow_measure_univ hp_pos hpq hf
    _ < ∞ := by
      rw [ENNReal.mul_lt_top_iff]
      refine Or.inl ⟨hfq_lt_top, ENNReal.rpow_lt_top_of_nonneg ?_ (measure_ne_top μ Set.univ)⟩
      rwa [le_sub_comm, sub_zero, one_div, one_div, inv_le_inv₀ hq_pos hp_pos]


@[deprecated (since := "2024-07-27")]
alias snorm'_lt_top_of_snorm'_lt_top_of_exponent_le :=
  eLpNorm'_lt_top_of_eLpNorm'_lt_top_of_exponent_le


theorem Memℒp.memℒp_of_exponent_le {p q : ℝ≥0∞} [IsFiniteMeasure μ] {f : α → E} (hfq : Memℒp f q μ)
    (hpq : p ≤ q) : Memℒp f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hfq : MeasureTheory.Memℒp f q μ
    hpq : LE.le p q
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  cases' hfq with hfq_m hfq_lt_top
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Eq p 0
      ⊢ MeasureTheory.Memℒp f p μ
    -/
  · rwa [hp0, memℒp_zero_iff_aestronglyMeasurable]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Not (Eq p 0)
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  rw [← Ne] at hp0
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  refine ⟨hfq_m, ?_⟩
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Ne p 0
      hp_top : Eq p Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    -/
  · have hq_top : q = ∞ := by rwa [hp_top, top_le_iff] at hpq
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Ne p 0
      hp_top : Eq p Top.top
      hq_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    -/
    rw [hp_top]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Ne p 0
      hp_top : Eq p Top.top
      hq_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm f Top.top μ) Top.top
    -/
    rwa [hq_top] at hfq_lt_top
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  have hp_pos : 0 < p.toReal := ENNReal.toReal_pos hp0 hp_top
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  by_cases hq_top : q = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Ne p 0
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      hq_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    -/
  · rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
      hp0 : Ne p 0
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      hq_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm' f p.toReal μ) Top.top
    -/
    rw [hq_top, eLpNorm_exponent_top] at hfq_lt_top
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
      hp0 : Ne p 0
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      hq_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm' f p.toReal μ) Top.top
    -/
    refine lt_of_le_of_lt (eLpNorm'_le_eLpNormEssSup_mul_rpow_measure_univ hp_pos) ?_
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
      hp0 : Ne p 0
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      hq_top : Eq q Top.top
      ⊢ LT.lt (HMul.hMul (MeasureTheory.eLpNormEssSup f μ) (HPow.hPow (μ Set.univ) ( …
    -/
    refine ENNReal.mul_lt_top hfq_lt_top ?_
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p q : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → E
      hpq : LE.le p q
      hfq_m : MeasureTheory.AEStronglyMeasurable f μ
      hfq_lt_top : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
      hp0 : Ne p 0
      hp_top : Not (Eq p Top.top)
      hp_pos : LT.lt 0 p.toReal
      hq_top : Eq q Top.top
      ⊢ LT.lt (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 p.toReal)) Top.top
    -/
    exact ENNReal.rpow_lt_top_of_nonneg (by simp [hp_pos.le]) (measure_ne_top μ Set.univ)
    /-
      🎉 no goals
    -/
  have hq0 : q ≠ 0 := by
    by_contra hq_eq_zero
    have hp_eq_zero : p = 0 := le_antisymm (by rwa [hq_eq_zero] at hpq) (zero_le _)
    rw [hp_eq_zero, ENNReal.zero_toReal] at hp_pos
    exact (lt_irrefl _) hp_pos
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    hq_top : Not (Eq q Top.top)
    hq0 : Ne q 0
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  have hpq_real : p.toReal ≤ q.toReal := ENNReal.toReal_mono hq_top hpq
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    hq_top : Not (Eq q Top.top)
    hq0 : Ne q 0
    hpq_real : LE.le p.toReal q.toReal
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm f q μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    hq_top : Not (Eq q Top.top)
    hq0 : Ne q 0
    hpq_real : LE.le p.toReal q.toReal
    ⊢ LT.lt (MeasureTheory.eLpNorm' f p.toReal μ) Top.top
  -/
  rw [eLpNorm_eq_eLpNorm' hq0 hq_top] at hfq_lt_top
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p q : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → E
    hpq : LE.le p q
    hfq_m : MeasureTheory.AEStronglyMeasurable f μ
    hfq_lt_top : LT.lt (MeasureTheory.eLpNorm' f q.toReal μ) Top.top
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    hq_top : Not (Eq q Top.top)
    hq0 : Ne q 0
    hpq_real : LE.le p.toReal q.toReal
    ⊢ LT.lt (MeasureTheory.eLpNorm' f p.toReal μ) Top.top
  -/
  exact eLpNorm'_lt_top_of_eLpNorm'_lt_top_of_exponent_le hfq_m hfq_lt_top hp_pos.le hpq_real
  /-
    🎉 no goals
  -/


theorem eLpNorm_le_eLpNorm_top_mul_eLpNorm (p : ℝ≥0∞) (f : α → E) {g : α → F}
    (hg : AEStronglyMeasurable g μ) (b : E → F → G)
    (h : ∀ᵐ x ∂μ, ‖b (f x) (g x)‖₊ ≤ ‖f x‖₊ * ‖g x‖₊) :
    eLpNorm (fun x => b (f x) (g x)) p μ ≤ eLpNorm f ∞ μ * eLpNorm g p μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : α → E
    g : α → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : α → E
      g : α → F
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
  · simp_rw [hp_top, eLpNorm_exponent_top]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : α → E
      g : α → F
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNormEssSup (fun x => b (f x) (g x)) μ) (HMul.hMul (M …
    -/
    refine le_trans (essSup_mono_ae <| h.mono fun a ha => ?_) (ENNReal.essSup_mul_le _ _)
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : α → E
      g : α → F
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hp_top : Eq p Top.top
      a : α
      ha : LE.le (NNNorm.nnnorm (b (f a) (g a))) (HMul.hMul (NNNorm.nnnorm (f a)) (N …
      ⊢ LE.le ((fun x => ENorm.enorm ((fun x => b (f x) (g x)) x)) a) (HMul.hMul (fu …
    -/
    simp_rw [Pi.mul_apply, enorm_eq_nnnorm, ← ENNReal.coe_mul, ENNReal.coe_le_coe]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : α → E
      g : α → F
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hp_top : Eq p Top.top
      a : α
      ha : LE.le (NNNorm.nnnorm (b (f a) (g a))) (HMul.hMul (NNNorm.nnnorm (f a)) (N …
      ⊢ LE.le (NNNorm.nnnorm (b (f a) (g a))) (HMul.hMul (NNNorm.nnnorm (f a)) (NNNo …
    -/
    exact ha
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : α → E
    g : α → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hp_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : α → E
      g : α → F
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hp_top : Not (Eq p Top.top)
      hp_zero : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
  · simp only [hp_zero, eLpNorm_exponent_zero, mul_zero, le_zero_iff]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : α → E
    g : α → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hp_top : Not (Eq p Top.top)
    hp_zero : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_zero hp_top, eLpNorm_exponent_top, eLpNormEssSup]
  calc
    (∫⁻ x, (‖b (f x) (g x)‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ^ (1 / p.toReal) ≤
        (∫⁻ x, (‖f x‖₊ : ℝ≥0∞) ^ p.toReal * (‖g x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ^ (1 / p.toReal) := by
      gcongr ?_ ^ _
      refine lintegral_mono_ae (h.mono fun a ha => ?_)
      rw [← ENNReal.mul_rpow_of_nonneg _ _ ENNReal.toReal_nonneg]
      refine ENNReal.rpow_le_rpow ?_ ENNReal.toReal_nonneg
      rw [← ENNReal.coe_mul, ENNReal.coe_le_coe]
      exact ha
    _ ≤
        (∫⁻ x, essSup (fun x => (‖f x‖₊ : ℝ≥0∞)) μ ^ p.toReal * (‖g x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ^
          (1 / p.toReal) := by
      gcongr ?_ ^ _
      refine lintegral_mono_ae ?_
      filter_upwards [@ENNReal.ae_le_essSup _ _ μ fun x => (‖f x‖₊ : ℝ≥0∞)] with x hx
      gcongr
    _ = essSup (fun x => (‖f x‖₊ : ℝ≥0∞)) μ *
        (∫⁻ x, (‖g x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ^ (1 / p.toReal) := by
      rw [lintegral_const_mul'']
      swap; · exact hg.nnnorm.aemeasurable.coe_nnreal_ennreal.pow aemeasurable_const
      rw [ENNReal.mul_rpow_of_nonneg]
      swap
      · rw [one_div_nonneg]
        exact ENNReal.toReal_nonneg
      rw [← ENNReal.rpow_mul, one_div, mul_inv_cancel₀, ENNReal.rpow_one]
      rw [Ne, ENNReal.toReal_eq_zero_iff, not_or]
      exact ⟨hp_zero, hp_top⟩


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_top_mul_snorm := eLpNorm_le_eLpNorm_top_mul_eLpNorm


theorem eLpNorm_le_eLpNorm_mul_eLpNorm_top (p : ℝ≥0∞) {f : α → E} (hf : AEStronglyMeasurable f μ)
    (g : α → F) (b : E → F → G) (h : ∀ᵐ x ∂μ, ‖b (f x) (g x)‖₊ ≤ ‖f x‖₊ * ‖g x‖₊) :
    eLpNorm (fun x => b (f x) (g x)) p μ ≤ eLpNorm f p μ * eLpNorm g ∞ μ :=
  calc
    eLpNorm (fun x ↦ b (f x) (g x)) p μ ≤ eLpNorm g ∞ μ * eLpNorm f p μ :=
                                                               /-
                                                                 α : Type u_1
                                                                 E : Type u_2
                                                                 F : Type u_3
                                                                 G : Type u_4
                                                                 m : MeasurableSpace α
                                                                 inst✝² : NormedAddCommGroup E
                                                                 inst✝¹ : NormedAddCommGroup F
                                                                 inst✝ : NormedAddCommGroup G
                                                                 μ : MeasureTheory.Measure α
                                                                 p : ENNReal
                                                                 f : α → E
                                                                 hf : MeasureTheory.AEStronglyMeasurable f μ
                                                                 g : α → F
                                                                 b : E → F → G
                                                                 h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
                                                                 ⊢ Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (flip b (g x) (f x))) (HMul …
                                                               -/
      eLpNorm_le_eLpNorm_top_mul_eLpNorm p g hf (flip b) <| by simpa only [mul_comm] using h
                                                               /-
                                                                 🎉 no goals
                                                               -/
    _ = eLpNorm f p μ * eLpNorm g ∞ μ := mul_comm _ _


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_mul_snorm_top := eLpNorm_le_eLpNorm_mul_eLpNorm_top


theorem eLpNorm'_le_eLpNorm'_mul_eLpNorm' {p q r : ℝ} (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) (b : E → F → G)
    (h : ∀ᵐ x ∂μ, ‖b (f x) (g x)‖₊ ≤ ‖f x‖₊ * ‖g x‖₊) (hp0_lt : 0 < p) (hpq : p < q)
    (hpqr : 1 / p = 1 / q + 1 / r) :
    eLpNorm' (fun x => b (f x) (g x)) p μ ≤ eLpNorm' f q μ * eLpNorm' g r μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    ⊢ LE.le (MeasureTheory.eLpNorm' (fun x => b (f x) (g x)) p μ) (HMul.hMul (Meas …
  -/
  rw [eLpNorm']
  calc
    (∫⁻ a : α, ↑‖b (f a) (g a)‖₊ ^ p ∂μ) ^ (1 / p) ≤
        (∫⁻ a : α, ↑(‖f a‖₊ * ‖g a‖₊) ^ p ∂μ) ^ (1 / p) :=
      (ENNReal.rpow_le_rpow_iff <| one_div_pos.mpr hp0_lt).mpr <|
        lintegral_mono_ae <|
          h.mono fun a ha => (ENNReal.rpow_le_rpow_iff hp0_lt).mpr <| ENNReal.coe_le_coe.mpr <| ha
    _ ≤ _ := ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(HMul.hMul  …
  -/
  simp_rw [eLpNorm', ENNReal.coe_mul]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hp0_lt : LT.lt 0 p
    hpq : LT.lt p q
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (HMul.hMul ↑( …
  -/
  exact ENNReal.lintegral_Lp_mul_le_Lq_mul_Lr hp0_lt hpq hpqr μ hf.ennnorm hg.ennnorm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_le_snorm'_mul_snorm' := eLpNorm'_le_eLpNorm'_mul_eLpNorm'


/-- Hölder's inequality, as an inequality on the `ℒp` seminorm of an elementwise operation
`fun x => b (f x) (g x)`. -/
theorem eLpNorm_le_eLpNorm_mul_eLpNorm_of_nnnorm {p q r : ℝ≥0∞}
    (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) (b : E → F → G)
    (h : ∀ᵐ x ∂μ, ‖b (f x) (g x)‖₊ ≤ ‖f x‖₊ * ‖g x‖₊) (hpqr : 1 / p = 1 / q + 1 / r) :
    eLpNorm (fun x => b (f x) (g x)) p μ ≤ eLpNorm f q μ * eLpNorm g r μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : ENNReal
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
  · simp [hp_zero]
    /-
      🎉 no goals
    -/
  have hq_ne_zero : q ≠ 0 := by
    intro hq_zero
    simp only [hq_zero, hp_zero, one_div, ENNReal.inv_zero, top_add, ENNReal.inv_eq_top] at hpqr
  have hr_ne_zero : r ≠ 0 := by
    intro hr_zero
    simp only [hr_zero, hp_zero, one_div, ENNReal.inv_zero, add_top, ENNReal.inv_eq_top] at hpqr
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : ENNReal
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    hp_zero : Not (Eq p 0)
    hq_ne_zero : Ne q 0
    hr_ne_zero : Ne r 0
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  by_cases hq_top : q = ∞
  · have hpr : p = r := by
      simpa only [hq_top, one_div, ENNReal.inv_top, zero_add, inv_inj] using hpqr
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Eq q Top.top
      hpr : Eq p r
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
    rw [← hpr, hq_top]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Eq q Top.top
      hpr : Eq p r
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
    exact eLpNorm_le_eLpNorm_top_mul_eLpNorm p f hg b h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : ENNReal
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    hp_zero : Not (Eq p 0)
    hq_ne_zero : Ne q 0
    hr_ne_zero : Ne r 0
    hq_top : Not (Eq q Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
  -/
  by_cases hr_top : r = ∞
  · have hpq : p = q := by
      simpa only [hr_top, one_div, ENNReal.inv_top, add_zero, inv_inj] using hpqr
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Eq r Top.top
      hpq : Eq p q
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
    rw [← hpq, hr_top]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Eq r Top.top
      hpq : Eq p q
      ⊢ LE.le (MeasureTheory.eLpNorm (fun x => b (f x) (g x)) p μ) (HMul.hMul (Measu …
    -/
    exact eLpNorm_le_eLpNorm_mul_eLpNorm_top p hf g b h
    /-
      🎉 no goals
    -/
  have hpq : p < q := by
    suffices 1 / q < 1 / p by rwa [one_div, one_div, ENNReal.inv_lt_inv] at this
    rw [hpqr]
    refine ENNReal.lt_add_right ?_ ?_
    · simp only [hq_ne_zero, one_div, Ne, ENNReal.inv_eq_top, not_false_iff]
    · simp only [hr_top, one_div, Ne, ENNReal.inv_eq_zero, not_false_iff]
  rw [eLpNorm_eq_eLpNorm' hp_zero (hpq.trans_le le_top).ne, eLpNorm_eq_eLpNorm' hq_ne_zero hq_top,
    eLpNorm_eq_eLpNorm' hr_ne_zero hr_top]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    m : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    μ : MeasureTheory.Measure α
    f : α → E
    g : α → F
    p q r : ENNReal
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    b : E → F → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
    hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
    hp_zero : Not (Eq p 0)
    hq_ne_zero : Ne q 0
    hr_ne_zero : Ne r 0
    hq_top : Not (Eq q Top.top)
    hr_top : Not (Eq r Top.top)
    hpq : LT.lt p q
    ⊢ LE.le (MeasureTheory.eLpNorm' (fun x => b (f x) (g x)) p.toReal μ) (HMul.hMu …
  -/
  refine eLpNorm'_le_eLpNorm'_mul_eLpNorm' hf hg _ h ?_ ?_ ?_
    /-
      case neg.refine_1
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Not (Eq r Top.top)
      hpq : LT.lt p q
      ⊢ LT.lt 0 p.toReal
    -/
  · exact ENNReal.toReal_pos hp_zero (hpq.trans_le le_top).ne
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Not (Eq r Top.top)
      hpq : LT.lt p q
      ⊢ LT.lt p.toReal q.toReal
    -/
  · exact ENNReal.toReal_strict_mono hq_top hpq
    /-
      🎉 no goals
    -/
  rw [← ENNReal.one_toReal, ← ENNReal.toReal_div, ← ENNReal.toReal_div, ← ENNReal.toReal_div, hpqr,
    ENNReal.toReal_add]
    /-
      case neg.refine_3.ha
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Not (Eq r Top.top)
      hpq : LT.lt p q
      ⊢ Ne (HDiv.hDiv 1 q) Top.top
    -/
  · simp only [hq_ne_zero, one_div, Ne, ENNReal.inv_eq_top, not_false_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_3.hb
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      μ : MeasureTheory.Measure α
      f : α → E
      g : α → F
      p q r : ENNReal
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      b : E → F → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (b (f x) (g x))) (HMul.hM …
      hpqr : Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 q) (HDiv.hDiv 1 r))
      hp_zero : Not (Eq p 0)
      hq_ne_zero : Ne q 0
      hr_ne_zero : Ne r 0
      hq_top : Not (Eq q Top.top)
      hr_top : Not (Eq r Top.top)
      hpq : LT.lt p q
      ⊢ Ne (HDiv.hDiv 1 r) Top.top
    -/
  · simp only [hr_ne_zero, one_div, Ne, ENNReal.inv_eq_top, not_false_iff]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_mul_snorm_of_nnnorm := eLpNorm_le_eLpNorm_mul_eLpNorm_of_nnnorm


/-- Hölder's inequality, as an inequality on the `ℒp` seminorm of an elementwise operation
`fun x => b (f x) (g x)`. -/
theorem eLpNorm_le_eLpNorm_mul_eLpNorm'_of_norm {p q r : ℝ≥0∞} (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) (b : E → F → G)
    (h : ∀ᵐ x ∂μ, ‖b (f x) (g x)‖ ≤ ‖f x‖ * ‖g x‖) (hpqr : 1 / p = 1 / q + 1 / r) :
    eLpNorm (fun x => b (f x) (g x)) p μ ≤ eLpNorm f q μ * eLpNorm g r μ :=
  eLpNorm_le_eLpNorm_mul_eLpNorm_of_nnnorm hf hg b h hpqr


@[deprecated (since := "2024-07-27")]
alias snorm_le_snorm_mul_snorm'_of_norm := eLpNorm_le_eLpNorm_mul_eLpNorm'_of_norm


theorem eLpNorm_smul_le_eLpNorm_top_mul_eLpNorm (p : ℝ≥0∞) (hf : AEStronglyMeasurable f μ)
    (φ : α → 𝕜) : eLpNorm (φ • f) p μ ≤ eLpNorm φ ∞ μ * eLpNorm f p μ :=
  (eLpNorm_le_eLpNorm_top_mul_eLpNorm p φ hf (· • ·)
    (Eventually.of_forall fun _ => nnnorm_smul_le _ _) : _)


@[deprecated (since := "2024-07-27")]
alias snorm_smul_le_snorm_top_mul_snorm := eLpNorm_smul_le_eLpNorm_top_mul_eLpNorm


theorem eLpNorm_smul_le_eLpNorm_mul_eLpNorm_top (p : ℝ≥0∞) (f : α → E) {φ : α → 𝕜}
    (hφ : AEStronglyMeasurable φ μ) : eLpNorm (φ • f) p μ ≤ eLpNorm φ p μ * eLpNorm f ∞ μ :=
  (eLpNorm_le_eLpNorm_mul_eLpNorm_top p hφ f (· • ·)
    (Eventually.of_forall fun _ => nnnorm_smul_le _ _) : _)


@[deprecated (since := "2024-07-27")]
alias snorm_smul_le_snorm_mul_snorm_top := eLpNorm_smul_le_eLpNorm_mul_eLpNorm_top


theorem eLpNorm'_smul_le_mul_eLpNorm' {p q r : ℝ} {f : α → E} (hf : AEStronglyMeasurable f μ)
    {φ : α → 𝕜} (hφ : AEStronglyMeasurable φ μ) (hp0_lt : 0 < p) (hpq : p < q)
    (hpqr : 1 / p = 1 / q + 1 / r) : eLpNorm' (φ • f) p μ ≤ eLpNorm' φ q μ * eLpNorm' f r μ :=
  eLpNorm'_le_eLpNorm'_mul_eLpNorm' hφ hf (· • ·) (Eventually.of_forall fun _ => nnnorm_smul_le _ _)
    hp0_lt hpq hpqr


@[deprecated (since := "2024-07-27")]
alias snorm'_smul_le_mul_snorm' := eLpNorm'_smul_le_mul_eLpNorm'


/-- Hölder's inequality, as an inequality on the `ℒp` seminorm of a scalar product `φ • f`. -/
theorem eLpNorm_smul_le_mul_eLpNorm {p q r : ℝ≥0∞} {f : α → E} (hf : AEStronglyMeasurable f μ)
    {φ : α → 𝕜} (hφ : AEStronglyMeasurable φ μ) (hpqr : 1 / p = 1 / q + 1 / r) :
    eLpNorm (φ • f) p μ ≤ eLpNorm φ q μ * eLpNorm f r μ :=
  (eLpNorm_le_eLpNorm_mul_eLpNorm_of_nnnorm hφ hf (· • ·)
      (Eventually.of_forall fun _ => nnnorm_smul_le _ _) hpqr :
    _)


@[deprecated (since := "2024-07-27")]
alias snorm_smul_le_mul_snorm := eLpNorm_smul_le_mul_eLpNorm


theorem Memℒp.smul {p q r : ℝ≥0∞} {f : α → E} {φ : α → 𝕜} (hf : Memℒp f r μ) (hφ : Memℒp φ q μ)
    (hpqr : 1 / p = 1 / q + 1 / r) : Memℒp (φ • f) p μ :=
  ⟨hφ.1.smul hf.1,
    (eLpNorm_smul_le_mul_eLpNorm hf.1 hφ.1 hpqr).trans_lt
      (ENNReal.mul_lt_top hφ.eLpNorm_lt_top hf.eLpNorm_lt_top)⟩


theorem Memℒp.smul_of_top_right {p : ℝ≥0∞} {f : α → E} {φ : α → 𝕜} (hf : Memℒp f p μ)
    (hφ : Memℒp φ ∞ μ) : Memℒp (φ • f) p μ := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedRing 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MulActionWithZero 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    p : ENNReal
    f : α → E
    φ : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    hφ : MeasureTheory.Memℒp φ Top.top μ
    ⊢ MeasureTheory.Memℒp (HSMul.hSMul φ f) p μ
  -/
  apply hf.smul hφ
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedRing 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MulActionWithZero 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    p : ENNReal
    f : α → E
    φ : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    hφ : MeasureTheory.Memℒp φ Top.top μ
    ⊢ Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 Top.top) (HDiv.hDiv 1 p))
  -/
  simp only [ENNReal.div_top, zero_add]
  /-
    🎉 no goals
  -/


theorem Memℒp.smul_of_top_left {p : ℝ≥0∞} {f : α → E} {φ : α → 𝕜} (hf : Memℒp f ∞ μ)
    (hφ : Memℒp φ p μ) : Memℒp (φ • f) p μ := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedRing 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MulActionWithZero 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    p : ENNReal
    f : α → E
    φ : α → 𝕜
    hf : MeasureTheory.Memℒp f Top.top μ
    hφ : MeasureTheory.Memℒp φ p μ
    ⊢ MeasureTheory.Memℒp (HSMul.hSMul φ f) p μ
  -/
  apply hf.smul hφ
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedRing 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MulActionWithZero 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    p : ENNReal
    f : α → E
    φ : α → 𝕜
    hf : MeasureTheory.Memℒp f Top.top μ
    hφ : MeasureTheory.Memℒp φ p μ
    ⊢ Eq (HDiv.hDiv 1 p) (HAdd.hAdd (HDiv.hDiv 1 p) (HDiv.hDiv 1 Top.top))
  -/
  simp only [ENNReal.div_top, add_zero]
  /-
    🎉 no goals
  -/


theorem Memℒp.mul (hf : Memℒp f r μ) (hφ : Memℒp φ q μ) (hpqr : 1 / p = 1 / q + 1 / r) :
    Memℒp (φ * f) p μ :=
  Memℒp.smul hf hφ hpqr


/-- Variant of `Memℒp.mul` where the function is written as `fun x ↦ φ x * f x`
instead of `φ * f`. -/
theorem Memℒp.mul' (hf : Memℒp f r μ) (hφ : Memℒp φ q μ) (hpqr : 1 / p = 1 / q + 1 / r) :
    Memℒp (fun x ↦ φ x * f x) p μ :=
  Memℒp.smul hf hφ hpqr


theorem Memℒp.mul_of_top_right (hf : Memℒp f p μ) (hφ : Memℒp φ ∞ μ) : Memℒp (φ * f) p μ :=
  Memℒp.smul_of_top_right hf hφ


/-- Variant of `Memℒp.mul_of_top_right` where the function is written as `fun x ↦ φ x * f x`
instead of `φ * f`. -/
theorem Memℒp.mul_of_top_right' (hf : Memℒp f p μ) (hφ : Memℒp φ ∞ μ) :
    Memℒp (fun x ↦ φ x * f x) p μ :=
  Memℒp.smul_of_top_right hf hφ


theorem Memℒp.mul_of_top_left (hf : Memℒp f ∞ μ) (hφ : Memℒp φ p μ) : Memℒp (φ * f) p μ :=
  Memℒp.smul_of_top_left hf hφ


/-- Variant of `Memℒp.mul_of_top_left` where the function is written as `fun x ↦ φ x * f x`
instead of `φ * f`. -/
theorem Memℒp.mul_of_top_left' (hf : Memℒp f ∞ μ) (hφ : Memℒp φ p μ) :
    Memℒp (fun x ↦ φ x * f x) p μ :=
  Memℒp.smul_of_top_left hf hφ


