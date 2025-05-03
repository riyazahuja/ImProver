/-- `(∫ ‖f a‖^q ∂μ) ^ (1/q)`, which is a seminorm on the space of measurable functions for which
this quantity is finite -/
def eLpNorm' {_ : MeasurableSpace α} (f : α → ε) (q : ℝ) (μ : Measure α) : ℝ≥0∞ :=
  (∫⁻ a, ‖f a‖ₑ ^ q ∂μ) ^ (1 / q)


lemma eLpNorm'_eq_lintegral_nnnorm {_ : MeasurableSpace α} (f : α → F) (q : ℝ) (μ : Measure α) :
    eLpNorm' f q μ = (∫⁻ a, ‖f a‖₊ ^ q ∂μ) ^ (1 / q) :=
  rfl


/-- seminorm for `ℒ∞`, equal to the essential supremum of `‖f‖`. -/
def eLpNormEssSup {_ : MeasurableSpace α} (f : α → ε) (μ : Measure α) :=
  essSup (fun x => ‖f x‖ₑ) μ


lemma eLpNormEssSup_eq_essSup_nnnorm {_ : MeasurableSpace α} (f : α → F) (μ : Measure α) :
    eLpNormEssSup f μ = essSup (fun x => (‖f x‖₊ : ℝ≥0∞)) μ :=
  rfl



/-- `ℒp` seminorm, equal to `0` for `p=0`, to `(∫ ‖f a‖^p ∂μ) ^ (1/p)` for `0 < p < ∞` and to
`essSup ‖f‖ μ` for `p = ∞`. -/
def eLpNorm {_ : MeasurableSpace α}
    (f : α → ε) (p : ℝ≥0∞) (μ : Measure α := by volume_tac) : ℝ≥0∞ :=
  if p = 0 then 0 else if p = ∞ then eLpNormEssSup f μ else eLpNorm' f (ENNReal.toReal p) μ


@[deprecated (since := "2024-07-26")] noncomputable alias snorm := eLpNorm


theorem eLpNorm_eq_eLpNorm' (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) {f : α → F} :
                                                          /-
                                                            α : Type u_1
                                                            F : Type u_4
                                                            m0 : MeasurableSpace α
                                                            p : ENNReal
                                                            μ : MeasureTheory.Measure α
                                                            inst✝ : NormedAddCommGroup F
                                                            hp_ne_zero : Ne p 0
                                                            hp_ne_top : Ne p Top.top
                                                            f : α → F
                                                            ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm' f p.toReal μ)
                                                          -/
    eLpNorm f p μ = eLpNorm' f (ENNReal.toReal p) μ := by simp [eLpNorm, hp_ne_zero, hp_ne_top]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-07-27")] alias snorm_eq_snorm' := eLpNorm_eq_eLpNorm'


lemma eLpNorm_nnreal_eq_eLpNorm' {f : α → F} {p : ℝ≥0} (hp : p ≠ 0) :
    eLpNorm f p μ = eLpNorm' f p μ :=
                          /-
                            α : Type u_1
                            F : Type u_4
                            m0 : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            inst✝ : NormedAddCommGroup F
                            f : α → F
                            p : NNReal
                            hp : Ne p 0
                            ⊢ Ne (↑p) 0
                          -/
  eLpNorm_eq_eLpNorm' (by exact_mod_cast hp) ENNReal.coe_ne_top
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-07-27")] alias snorm_nnreal_eq_snorm' := eLpNorm_nnreal_eq_eLpNorm'


theorem eLpNorm_eq_lintegral_rpow_nnnorm (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) {f : α → F} :
    eLpNorm f p μ = (∫⁻ x, (‖f x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (HPow.hPow (MeasureTheory.lintegral μ fun x …
  -/
  rw [eLpNorm_eq_eLpNorm' hp_ne_zero hp_ne_top, eLpNorm'_eq_lintegral_nnnorm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_eq_lintegral_rpow_nnnorm := eLpNorm_eq_lintegral_rpow_nnnorm


lemma eLpNorm_nnreal_eq_lintegral {f : α → F} {p : ℝ≥0} (hp : p ≠ 0) :
    eLpNorm f p μ = (∫⁻ x, ‖f x‖₊ ^ (p : ℝ) ∂μ) ^ (1 / (p : ℝ)) :=
  eLpNorm_nnreal_eq_eLpNorm' hp


@[deprecated (since := "2024-07-27")] alias snorm_nnreal_eq_lintegral := eLpNorm_nnreal_eq_lintegral


theorem eLpNorm_one_eq_lintegral_nnnorm {f : α → F} : eLpNorm f 1 μ = ∫⁻ x, ‖f x‖₊ ∂μ := by
  simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm one_ne_zero ENNReal.coe_ne_top, ENNReal.one_toReal,
    one_div_one, ENNReal.rpow_one]


@[deprecated (since := "2024-07-27")]
alias snorm_one_eq_lintegral_nnnorm := eLpNorm_one_eq_lintegral_nnnorm


@[simp]
                                                                                   /-
                                                                                     α : Type u_1
                                                                                     F : Type u_4
                                                                                     m0 : MeasurableSpace α
                                                                                     μ : MeasureTheory.Measure α
                                                                                     inst✝ : NormedAddCommGroup F
                                                                                     f : α → F
                                                                                     ⊢ Eq (MeasureTheory.eLpNorm f Top.top μ) (MeasureTheory.eLpNormEssSup f μ)
                                                                                   -/
theorem eLpNorm_exponent_top {f : α → F} : eLpNorm f ∞ μ = eLpNormEssSup f μ := by simp [eLpNorm]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[deprecated (since := "2024-07-27")]
alias snorm_exponent_top := eLpNorm_exponent_top


/-- The property that `f:α→E` is ae strongly measurable and `(∫ ‖f a‖^p ∂μ)^(1/p)` is finite
if `p < ∞`, or `essSup f < ∞` if `p = ∞`. -/
def Memℒp {α} {_ : MeasurableSpace α} [TopologicalSpace ε] (f : α → ε) (p : ℝ≥0∞)
    (μ : Measure α := by volume_tac) : Prop :=
  AEStronglyMeasurable f μ ∧ eLpNorm f p μ < ∞


theorem Memℒp.aestronglyMeasurable {f : α → E} {p : ℝ≥0∞} (h : Memℒp f p μ) :
    AEStronglyMeasurable f μ :=
  h.1


theorem lintegral_rpow_nnnorm_eq_rpow_eLpNorm' {f : α → F} (hq0_lt : 0 < q) :
    ∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ q ∂μ = eLpNorm' f q μ ^ q := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq0_lt : LT.lt 0 q
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) q) …
  -/
  rw [eLpNorm'_eq_lintegral_nnnorm, ← ENNReal.rpow_mul, one_div, inv_mul_cancel₀, ENNReal.rpow_one]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq0_lt : LT.lt 0 q
    ⊢ Ne q 0
  -/
  exact (ne_of_lt hq0_lt).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias lintegral_rpow_nnnorm_eq_rpow_snorm' := lintegral_rpow_nnnorm_eq_rpow_eLpNorm'


lemma eLpNorm_nnreal_pow_eq_lintegral {f : α → F} {p : ℝ≥0} (hp : p ≠ 0) :
    eLpNorm f p μ ^ (p : ℝ) = ∫⁻ x, ‖f x‖₊ ^ (p : ℝ) ∂μ := by
  simp [eLpNorm_eq_eLpNorm' (by exact_mod_cast hp) ENNReal.coe_ne_top,
    lintegral_rpow_nnnorm_eq_rpow_eLpNorm' ((NNReal.coe_pos.trans pos_iff_ne_zero).mpr hp)]


@[deprecated (since := "2024-07-27")]
alias snorm_nnreal_pow_eq_lintegral := eLpNorm_nnreal_pow_eq_lintegral


theorem Memℒp.eLpNorm_lt_top {f : α → E} (hfp : Memℒp f p μ) : eLpNorm f p μ < ∞ :=
  hfp.2


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_lt_top := Memℒp.eLpNorm_lt_top


theorem Memℒp.eLpNorm_ne_top {f : α → E} (hfp : Memℒp f p μ) : eLpNorm f p μ ≠ ∞ :=
  ne_of_lt hfp.2


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_ne_top := Memℒp.eLpNorm_ne_top


theorem lintegral_rpow_nnnorm_lt_top_of_eLpNorm'_lt_top {f : α → F} (hq0_lt : 0 < q)
    (hfq : eLpNorm' f q μ < ∞) : (∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ q ∂μ) < ∞ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq0_lt : LT.lt 0 q
    hfq : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) …
  -/
  rw [lintegral_rpow_nnnorm_eq_rpow_eLpNorm' hq0_lt]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq0_lt : LT.lt 0 q
    hfq : LT.lt (MeasureTheory.eLpNorm' f q μ) Top.top
    ⊢ LT.lt (HPow.hPow (MeasureTheory.eLpNorm' f q μ) q) Top.top
  -/
  exact ENNReal.rpow_lt_top_of_nonneg (le_of_lt hq0_lt) (ne_of_lt hfq)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias lintegral_rpow_nnnorm_lt_top_of_snorm'_lt_top :=
  lintegral_rpow_nnnorm_lt_top_of_eLpNorm'_lt_top


theorem lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top {f : α → F} (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) (hfp : eLpNorm f p μ < ∞) : (∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) < ∞ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hfp : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) …
  -/
  apply lintegral_rpow_nnnorm_lt_top_of_eLpNorm'_lt_top
    /-
      case hq0_lt
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hfp : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
      ⊢ LT.lt 0 p.toReal
    -/
  · exact ENNReal.toReal_pos hp_ne_zero hp_ne_top
    /-
      🎉 no goals
    -/
    /-
      case hfq
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hfp : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm' f p.toReal μ) Top.top
    -/
  · simpa [eLpNorm_eq_eLpNorm' hp_ne_zero hp_ne_top] using hfp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias lintegral_rpow_nnnorm_lt_top_of_snorm_lt_top := lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top


theorem eLpNorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top {f : α → F} (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) : eLpNorm f p μ < ∞ ↔ (∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) < ∞ :=
  ⟨lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top hp_ne_zero hp_ne_top, by
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) …
    -/
    intro h
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      h : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    -/
    have hp' := ENNReal.toReal_pos hp_ne_zero hp_ne_top
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      h : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a) …
      hp' : LT.lt 0 p.toReal
      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    -/
    have : 0 < 1 / p.toReal := div_pos zero_lt_one hp'
    simpa [eLpNorm_eq_lintegral_rpow_nnnorm hp_ne_zero hp_ne_top] using
      ENNReal.rpow_lt_top_of_nonneg (le_of_lt this) (ne_of_lt h)⟩


@[deprecated (since := "2024-07-27")]
alias snorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top :=
  eLpNorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top


@[simp]
theorem eLpNorm'_exponent_zero {f : α → F} : eLpNorm' f 0 μ = 1 := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm' f 0 μ) 1
  -/
  rw [eLpNorm', div_zero, ENNReal.rpow_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_exponent_zero := eLpNorm'_exponent_zero


@[simp]
                                                                    /-
                                                                      α : Type u_1
                                                                      F : Type u_4
                                                                      m0 : MeasurableSpace α
                                                                      μ : MeasureTheory.Measure α
                                                                      inst✝ : NormedAddCommGroup F
                                                                      f : α → F
                                                                      ⊢ Eq (MeasureTheory.eLpNorm f 0 μ) 0
                                                                    -/
theorem eLpNorm_exponent_zero {f : α → F} : eLpNorm f 0 μ = 0 := by simp [eLpNorm]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-07-27")]
alias snorm_exponent_zero := eLpNorm_exponent_zero


@[simp]
theorem memℒp_zero_iff_aestronglyMeasurable {f : α → E} :
                                                 /-
                                                   α : Type u_1
                                                   E : Type u_3
                                                   m0 : MeasurableSpace α
                                                   μ : MeasureTheory.Measure α
                                                   inst✝ : NormedAddCommGroup E
                                                   f : α → E
                                                   ⊢ Iff (MeasureTheory.Memℒp f 0 μ) (MeasureTheory.AEStronglyMeasurable f μ)
                                                 -/
    Memℒp f 0 μ ↔ AEStronglyMeasurable f μ := by simp [Memℒp, eLpNorm_exponent_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem eLpNorm'_zero (hp0_lt : 0 < q) : eLpNorm' (0 : α → F) q μ = 0 := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp0_lt : LT.lt 0 q
    ⊢ Eq (MeasureTheory.eLpNorm' 0 q μ) 0
  -/
  simp [eLpNorm'_eq_lintegral_nnnorm, hp0_lt]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_zero := eLpNorm'_zero


@[simp]
theorem eLpNorm'_zero' (hq0_ne : q ≠ 0) (hμ : μ ≠ 0) : eLpNorm' (0 : α → F) q μ = 0 := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hq0_ne : Ne q 0
    hμ : Ne μ 0
    ⊢ Eq (MeasureTheory.eLpNorm' 0 q μ) 0
  -/
  rcases le_or_lt 0 q with hq0 | hq_neg
    /-
      case inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hq0_ne : Ne q 0
      hμ : Ne μ 0
      hq0 : LE.le 0 q
      ⊢ Eq (MeasureTheory.eLpNorm' 0 q μ) 0
    -/
  · exact eLpNorm'_zero (lt_of_le_of_ne hq0 hq0_ne.symm)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hq0_ne : Ne q 0
      hμ : Ne μ 0
      hq_neg : LT.lt q 0
      ⊢ Eq (MeasureTheory.eLpNorm' 0 q μ) 0
    -/
  · simp [eLpNorm'_eq_lintegral_nnnorm, ENNReal.rpow_eq_zero_iff, hμ, hq_neg]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm'_zero' := eLpNorm'_zero'


@[simp]
theorem eLpNormEssSup_zero : eLpNormEssSup (0 : α → F) μ = 0 := by
  simp_rw [eLpNormEssSup_eq_essSup_nnnorm, Pi.zero_apply, nnnorm_zero, ENNReal.coe_zero,
    ← ENNReal.bot_eq_zero]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    ⊢ Eq (essSup (fun x => Bot.bot) μ) Bot.bot
  -/
  exact essSup_const_bot
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_zero := eLpNormEssSup_zero


@[simp]
theorem eLpNorm_zero : eLpNorm (0 : α → F) p μ = 0 := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
    -/
  · simp only [h_top, eLpNorm_exponent_top, eLpNormEssSup_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
  -/
  rw [← Ne] at h0
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    h0 : Ne p 0
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm 0 p μ) 0
  -/
  simp [eLpNorm_eq_eLpNorm' h0 h_top, ENNReal.toReal_pos h0 h_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_zero := eLpNorm_zero


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       F : Type u_4
                                                                       m0 : MeasurableSpace α
                                                                       p : ENNReal
                                                                       μ : MeasureTheory.Measure α
                                                                       inst✝ : NormedAddCommGroup F
                                                                       ⊢ Eq (MeasureTheory.eLpNorm (fun x => 0) p μ) 0
                                                                     -/
theorem eLpNorm_zero' : eLpNorm (fun _ : α => (0 : F)) p μ = 0 := by convert eLpNorm_zero (F := F)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[deprecated (since := "2024-07-27")]
alias snorm_zero' := eLpNorm_zero'


theorem zero_memℒp : Memℒp (0 : α → E) p μ :=
  ⟨aestronglyMeasurable_zero, by
    /-
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      ⊢ LT.lt (MeasureTheory.eLpNorm 0 p μ) Top.top
    -/
    rw [eLpNorm_zero]
    /-
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      ⊢ LT.lt 0 Top.top
    -/
    exact ENNReal.coe_lt_top⟩
    /-
      🎉 no goals
    -/


theorem zero_mem_ℒp' : Memℒp (fun _ : α => (0 : E)) p μ := zero_memℒp (E := E)


theorem eLpNorm'_measure_zero_of_pos {f : α → F} (hq_pos : 0 < q) :
                                           /-
                                             α : Type u_1
                                             F : Type u_4
                                             q : Real
                                             inst✝¹ : NormedAddCommGroup F
                                             inst✝ : MeasurableSpace α
                                             f : α → F
                                             hq_pos : LT.lt 0 q
                                             ⊢ Eq (MeasureTheory.eLpNorm' f q 0) 0
                                           -/
    eLpNorm' f q (0 : Measure α) = 0 := by simp [eLpNorm', hq_pos]
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-07-27")]
alias snorm'_measure_zero_of_pos := eLpNorm'_measure_zero_of_pos


theorem eLpNorm'_measure_zero_of_exponent_zero {f : α → F} : eLpNorm' f 0 (0 : Measure α) = 1 := by
  /-
    α : Type u_1
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm' f 0 0) 1
  -/
  simp [eLpNorm']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_measure_zero_of_exponent_zero := eLpNorm'_measure_zero_of_exponent_zero


theorem eLpNorm'_measure_zero_of_neg {f : α → F} (hq_neg : q < 0) :
                                           /-
                                             α : Type u_1
                                             F : Type u_4
                                             q : Real
                                             inst✝¹ : NormedAddCommGroup F
                                             inst✝ : MeasurableSpace α
                                             f : α → F
                                             hq_neg : LT.lt q 0
                                             ⊢ Eq (MeasureTheory.eLpNorm' f q 0) Top.top
                                           -/
    eLpNorm' f q (0 : Measure α) = ∞ := by simp [eLpNorm', hq_neg]
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-07-27")]
alias snorm'_measure_zero_of_neg := eLpNorm'_measure_zero_of_neg


@[simp]
theorem eLpNormEssSup_measure_zero {f : α → F} : eLpNormEssSup f (0 : Measure α) = 0 := by
  /-
    α : Type u_1
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNormEssSup f 0) 0
  -/
  simp [eLpNormEssSup]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_measure_zero := eLpNormEssSup_measure_zero


@[simp]
theorem eLpNorm_measure_zero {f : α → F} : eLpNorm f p (0 : Measure α) = 0 := by
  /-
    α : Type u_1
    F : Type u_4
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      p : ENNReal
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasurableSpace α
      f : α → F
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      p : ENNReal
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasurableSpace α
      f : α → F
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
    -/
  · simp [h_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
  -/
  rw [← Ne] at h0
  /-
    case neg
    α : Type u_1
    F : Type u_4
    p : ENNReal
    inst✝¹ : NormedAddCommGroup F
    inst✝ : MeasurableSpace α
    f : α → F
    h0 : Ne p 0
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm f p 0) 0
  -/
  simp [eLpNorm_eq_eLpNorm' h0 h_top, eLpNorm', ENNReal.toReal_pos h0 h_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_measure_zero := eLpNorm_measure_zero


                                                                               /-
                                                                                 α : Type u_1
                                                                                 F : Type u_4
                                                                                 p : ENNReal
                                                                                 inst✝¹ : NormedAddCommGroup F
                                                                                 inst✝ : MeasurableSpace α
                                                                                 f : α → F
                                                                                 ⊢ MeasureTheory.Memℒp f p 0
                                                                               -/
@[simp] lemma memℒp_measure_zero {f : α → F} : Memℒp f p (0 : Measure α) := by simp [Memℒp]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem eLpNorm'_neg (f : α → F) (q : ℝ) (μ : Measure α) : eLpNorm' (-f) q μ = eLpNorm' f q μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    q : Real
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.eLpNorm' (Neg.neg f) q μ) (MeasureTheory.eLpNorm' f q μ)
  -/
  simp [eLpNorm'_eq_lintegral_nnnorm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_neg := eLpNorm'_neg


@[simp]
theorem eLpNorm_neg (f : α → F) (p : ℝ≥0∞) (μ : Measure α) : eLpNorm (-f) p μ = eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.eLpNorm (Neg.neg f) p μ) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup F
      f : α → F
      p : ENNReal
      μ : MeasureTheory.Measure α
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm (Neg.neg f) p μ) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm (Neg.neg f) p μ) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup F
      f : α → F
      p : ENNReal
      μ : MeasureTheory.Measure α
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm (Neg.neg f) p μ) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [h_top, eLpNormEssSup_eq_essSup_nnnorm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm (Neg.neg f) p μ) (MeasureTheory.eLpNorm f p μ)
  -/
  simp [eLpNorm_eq_eLpNorm' h0 h_top]
  /-
    🎉 no goals
  -/


lemma eLpNorm_sub_comm (f g : α → E) (p : ℝ≥0∞) (μ : Measure α) :
                                                    /-
                                                      α : Type u_1
                                                      E : Type u_3
                                                      m0 : MeasurableSpace α
                                                      inst✝ : NormedAddCommGroup E
                                                      f g : α → E
                                                      p : ENNReal
                                                      μ : MeasureTheory.Measure α
                                                      ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) (MeasureTheory.eLpNorm (HSub. …
                                                    -/
    eLpNorm (f - g) p μ = eLpNorm (g - f) p μ := by simp [← eLpNorm_neg (f := f - g)]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-07-27")]
alias snorm_neg := eLpNorm_neg


theorem Memℒp.neg {f : α → E} (hf : Memℒp f p μ) : Memℒp (-f) p μ :=
                                     /-
                                       α : Type u_1
                                       E : Type u_3
                                       m0 : MeasurableSpace α
                                       p : ENNReal
                                       μ : MeasureTheory.Measure α
                                       inst✝ : NormedAddCommGroup E
                                       f : α → E
                                       hf : MeasureTheory.Memℒp f p μ
                                       ⊢ LT.lt (MeasureTheory.eLpNorm (Neg.neg f) p μ) Top.top
                                     -/
  ⟨AEStronglyMeasurable.neg hf.1, by simp [hf.right]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem memℒp_neg_iff {f : α → E} : Memℒp (-f) p μ ↔ Memℒp f p μ :=
  ⟨fun h => neg_neg f ▸ h.neg, Memℒp.neg⟩


theorem eLpNorm_indicator_eq_restrict {f : α → E} {s : Set α} (hs : MeasurableSet s) :
    eLpNorm (s.indicator f) p μ = eLpNorm f p (μ.restrict s) := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
  -/
  rcases eq_or_ne p ∞ with rfl | hp
  · simp only [eLpNorm_exponent_top, eLpNormEssSup_eq_essSup_nnnorm,
      ← ENNReal.essSup_indicator_eq_essSup_restrict hs, ENNReal.coe_indicator,
      nnnorm_indicator_eq_indicator_nnnorm]
    /-
      case inr
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hp : Ne p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
    -/
  · rcases eq_or_ne p 0 with rfl | hp₀; · simp
                                          /-
                                            🎉 no goals
                                          -/
    simp only [eLpNorm_eq_lintegral_rpow_nnnorm hp₀ hp, ← lintegral_indicator hs,
      ENNReal.coe_indicator, nnnorm_indicator_eq_indicator_nnnorm]
    /-
      case inr.inr
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hp : Ne p Top.top
      hp₀ : Ne p 0
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hPow (s.indicator (fu …
    -/
    congr with x
    /-
      case inr.inr.e_a.e_f.h
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      s : Set α
      hs : MeasurableSet s
      hp : Ne p Top.top
      hp₀ : Ne p 0
      x : α
      ⊢ Eq (HPow.hPow (s.indicator (fun x => ↑(NNNorm.nnnorm (f x))) x) p.toReal) (s …
    -/
                            /-
                              🎉 no goals
                            -/
    by_cases hx : x ∈ s <;> simp [ENNReal.toReal_pos, *]
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_eq_restrict := eLpNorm_indicator_eq_restrict


theorem eLpNorm'_const (c : F) (hq_pos : 0 < q) :
    eLpNorm' (fun _ : α => c) q μ = (‖c‖₊ : ℝ≥0∞) * μ Set.univ ^ (1 / q) := by
  rw [eLpNorm'_eq_lintegral_nnnorm, lintegral_const,
    ENNReal.mul_rpow_of_nonneg _ _ (by simp [hq_pos.le] : 0 ≤ 1 / q)]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    hq_pos : LT.lt 0 q
    ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow (↑(NNNorm.nnnorm c)) q) (HDiv.hDiv 1 q)) …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    hq_pos : LT.lt 0 q
    ⊢ Eq (HPow.hPow (HPow.hPow (↑(NNNorm.nnnorm c)) q) (HDiv.hDiv 1 q)) ↑(NNNorm.n …
  -/
  rw [← ENNReal.rpow_mul]
  /-
    case e_a
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    hq_pos : LT.lt 0 q
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm c)) (HMul.hMul q (HDiv.hDiv 1 q))) ↑(NNNorm.n …
  -/
  suffices hq_cancel : q * (1 / q) = 1 by rw [hq_cancel, ENNReal.rpow_one]
  /-
    case e_a
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    hq_pos : LT.lt 0 q
    ⊢ Eq (HMul.hMul q (HDiv.hDiv 1 q)) 1
  -/
  rw [one_div, mul_inv_cancel₀ (ne_of_lt hq_pos).symm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_const := eLpNorm'_const


theorem eLpNorm'_const' [IsFiniteMeasure μ] (c : F) (hc_ne_zero : c ≠ 0) (hq_ne_zero : q ≠ 0) :
    eLpNorm' (fun _ : α => c) q μ = (‖c‖₊ : ℝ≥0∞) * μ Set.univ ^ (1 / q) := by
  rw [eLpNorm'_eq_lintegral_nnnorm, lintegral_const,
    ENNReal.mul_rpow_of_ne_top _ (measure_ne_top μ Set.univ)]
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow (↑(NNNorm.nnnorm c)) q) (HDiv.hDiv 1 q)) …
    -/
  · congr
    /-
      case e_a
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ Eq (HPow.hPow (HPow.hPow (↑(NNNorm.nnnorm c)) q) (HDiv.hDiv 1 q)) ↑(NNNorm.n …
    -/
    rw [← ENNReal.rpow_mul]
    /-
      case e_a
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm c)) (HMul.hMul q (HDiv.hDiv 1 q))) ↑(NNNorm.n …
    -/
    suffices hp_cancel : q * (1 / q) = 1 by rw [hp_cancel, ENNReal.rpow_one]
    /-
      case e_a
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ Eq (HMul.hMul q (HDiv.hDiv 1 q)) 1
    -/
    rw [one_div, mul_inv_cancel₀ hq_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ Ne (HPow.hPow (↑(NNNorm.nnnorm c)) q) Top.top
    -/
  · rw [Ne, ENNReal.rpow_eq_top_iff, not_or, not_and_or, not_and_or]
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : F
      hc_ne_zero : Ne c 0
      hq_ne_zero : Ne q 0
      ⊢ And (Or (Not (Eq (↑(NNNorm.nnnorm c)) 0)) (Not (LT.lt q 0))) (Or (Not (Eq (↑ …
    -/
    constructor
      /-
        case left
        α : Type u_1
        F : Type u_4
        m0 : MeasurableSpace α
        q : Real
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup F
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        c : F
        hc_ne_zero : Ne c 0
        hq_ne_zero : Ne q 0
        ⊢ Or (Not (Eq (↑(NNNorm.nnnorm c)) 0)) (Not (LT.lt q 0))
      -/
    · left
      /-
        case left.h
        α : Type u_1
        F : Type u_4
        m0 : MeasurableSpace α
        q : Real
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup F
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        c : F
        hc_ne_zero : Ne c 0
        hq_ne_zero : Ne q 0
        ⊢ Not (Eq (↑(NNNorm.nnnorm c)) 0)
      -/
      rwa [ENNReal.coe_eq_zero, nnnorm_eq_zero]
      /-
        🎉 no goals
      -/
      /-
        case right
        α : Type u_1
        F : Type u_4
        m0 : MeasurableSpace α
        q : Real
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup F
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        c : F
        hc_ne_zero : Ne c 0
        hq_ne_zero : Ne q 0
        ⊢ Or (Not (Eq (↑(NNNorm.nnnorm c)) Top.top)) (Not (LT.lt 0 q))
      -/
    · exact Or.inl ENNReal.coe_ne_top
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-27")]
alias snorm'_const' := eLpNorm'_const'


theorem eLpNormEssSup_const (c : F) (hμ : μ ≠ 0) :
    eLpNormEssSup (fun _ : α => c) μ = (‖c‖₊ : ℝ≥0∞) := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    hμ : Ne μ 0
    ⊢ Eq (MeasureTheory.eLpNormEssSup (fun x => c) μ) ↑(NNNorm.nnnorm c)
  -/
  rw [eLpNormEssSup_eq_essSup_nnnorm, essSup_const _ hμ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_const := eLpNormEssSup_const


theorem eLpNorm'_const_of_isProbabilityMeasure (c : F) (hq_pos : 0 < q) [IsProbabilityMeasure μ] :
                                                        /-
                                                          α : Type u_1
                                                          F : Type u_4
                                                          m0 : MeasurableSpace α
                                                          q : Real
                                                          μ : MeasureTheory.Measure α
                                                          inst✝¹ : NormedAddCommGroup F
                                                          c : F
                                                          hq_pos : LT.lt 0 q
                                                          inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                                          ⊢ Eq (MeasureTheory.eLpNorm' (fun x => c) q μ) ↑(NNNorm.nnnorm c)
                                                        -/
    eLpNorm' (fun _ : α => c) q μ = (‖c‖₊ : ℝ≥0∞) := by simp [eLpNorm'_const c hq_pos, measure_univ]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[deprecated (since := "2024-07-27")]
alias snorm'_const_of_isProbabilityMeasure := eLpNorm'_const_of_isProbabilityMeasure


theorem eLpNorm_const (c : F) (h0 : p ≠ 0) (hμ : μ ≠ 0) :
    eLpNorm (fun _ : α => c) p μ = (‖c‖₊ : ℝ≥0∞) * μ Set.univ ^ (1 / ENNReal.toReal p) := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    h0 : Ne p 0
    hμ : Ne μ 0
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => c) p μ) (HMul.hMul (↑(NNNorm.nnnorm c))  …
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      c : F
      h0 : Ne p 0
      hμ : Ne μ 0
      h_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm (fun x => c) p μ) (HMul.hMul (↑(NNNorm.nnnorm c))  …
    -/
  · simp [h_top, eLpNormEssSup_const c hμ]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    h0 : Ne p 0
    hμ : Ne μ 0
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => c) p μ) (HMul.hMul (↑(NNNorm.nnnorm c))  …
  -/
  simp [eLpNorm_eq_eLpNorm' h0 h_top, eLpNorm'_const, ENNReal.toReal_pos h0 h_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_const := eLpNorm_const


theorem eLpNorm_const' (c : F) (h0 : p ≠ 0) (h_top : p ≠ ∞) :
    eLpNorm (fun _ : α => c) p μ = (‖c‖₊ : ℝ≥0∞) * μ Set.univ ^ (1 / ENNReal.toReal p) := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    c : F
    h0 : Ne p 0
    h_top : Ne p Top.top
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => c) p μ) (HMul.hMul (↑(NNNorm.nnnorm c))  …
  -/
  simp [eLpNorm_eq_eLpNorm' h0 h_top, eLpNorm'_const, ENNReal.toReal_pos h0 h_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_const' := eLpNorm_const'


theorem eLpNorm_const_lt_top_iff {p : ℝ≥0∞} {c : F} (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    eLpNorm (fun _ : α => c) p μ < ∞ ↔ c = 0 ∨ μ Set.univ < ∞ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ Iff (LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top) (Or (Eq c 0) (L …
  -/
  have hp : 0 < p.toReal := ENNReal.toReal_pos hp_ne_zero hp_ne_top
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    ⊢ Iff (LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top) (Or (Eq c 0) (L …
  -/
  by_cases hμ : μ = 0
  · simp only [hμ, Measure.coe_zero, Pi.zero_apply, or_true, ENNReal.zero_lt_top,
      eLpNorm_measure_zero]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    hμ : Not (Eq μ 0)
    ⊢ Iff (LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top) (Or (Eq c 0) (L …
  -/
  by_cases hc : c = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      c : F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hp : LT.lt 0 p.toReal
      hμ : Not (Eq μ 0)
      hc : Eq c 0
      ⊢ Iff (LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top) (Or (Eq c 0) (L …
    -/
  · simp only [hc, true_or, eq_self_iff_true, ENNReal.zero_lt_top, eLpNorm_zero']
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    hμ : Not (Eq μ 0)
    hc : Not (Eq c 0)
    ⊢ Iff (LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top) (Or (Eq c 0) (L …
  -/
  rw [eLpNorm_const' c hp_ne_zero hp_ne_top]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    hμ : Not (Eq μ 0)
    hc : Not (Eq c 0)
    ⊢ Iff (LT.lt (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDi …
  -/
  by_cases hμ_top : μ Set.univ = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      c : F
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hp : LT.lt 0 p.toReal
      hμ : Not (Eq μ 0)
      hc : Not (Eq c 0)
      hμ_top : Eq (μ Set.univ) Top.top
      ⊢ Iff (LT.lt (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDi …
    -/
  · simp [hc, hμ_top, hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    hμ : Not (Eq μ 0)
    hc : Not (Eq c 0)
    hμ_top : Not (Eq (μ Set.univ) Top.top)
    ⊢ Iff (LT.lt (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDi …
  -/
  rw [ENNReal.mul_lt_top_iff]
  simp only [true_and, one_div, ENNReal.rpow_eq_zero_iff, hμ, false_or, or_false,
    ENNReal.coe_lt_top, nnnorm_eq_zero, ENNReal.coe_eq_zero,
    MeasureTheory.Measure.measure_univ_eq_zero, hp, inv_lt_zero, hc, false_and,
    inv_pos, or_self_iff, hμ_top, Ne.lt_top hμ_top, iff_true]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    c : F
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hp : LT.lt 0 p.toReal
    hμ : Not (Eq μ 0)
    hc : Not (Eq c 0)
    hμ_top : Not (Eq (μ Set.univ) Top.top)
    ⊢ LT.lt (HPow.hPow (μ Set.univ) (Inv.inv p.toReal)) Top.top
  -/
  exact ENNReal.rpow_lt_top_of_nonneg (inv_nonneg.mpr hp.le) hμ_top
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_const_lt_top_iff := eLpNorm_const_lt_top_iff


theorem memℒp_const (c : E) [IsFiniteMeasure μ] : Memℒp (fun _ : α => c) p μ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ MeasureTheory.Memℒp (fun x => c) p μ
  -/
  refine ⟨aestronglyMeasurable_const, ?_⟩
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      c : E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h0 : Eq p 0
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h0 : Not (Eq p 0)
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      c : E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h0 : Not (Eq p 0)
      hμ : Eq μ 0
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top
    -/
  · simp [hμ]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h0 : Not (Eq p 0)
    hμ : Not (Eq μ 0)
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) p μ) Top.top
  -/
  rw [eLpNorm_const c h0 hμ]
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h0 : Not (Eq p 0)
    hμ : Not (Eq μ 0)
    ⊢ LT.lt (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 p …
  -/
  refine ENNReal.mul_lt_top ENNReal.coe_lt_top ?_
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h0 : Not (Eq p 0)
    hμ : Not (Eq μ 0)
    ⊢ LT.lt (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 p.toReal)) Top.top
  -/
  refine ENNReal.rpow_lt_top_of_nonneg ?_ (measure_ne_top μ Set.univ)
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    c : E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h0 : Not (Eq p 0)
    hμ : Not (Eq μ 0)
    ⊢ LE.le 0 (HDiv.hDiv 1 p.toReal)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem memℒp_top_const (c : E) : Memℒp (fun _ : α => c) ∞ μ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    c : E
    ⊢ MeasureTheory.Memℒp (fun x => c) Top.top μ
  -/
  refine ⟨aestronglyMeasurable_const, ?_⟩
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    c : E
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) Top.top μ) Top.top
  -/
  by_cases h : μ = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      c : E
      h : Eq μ 0
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) Top.top μ) Top.top
    -/
  · simp only [h, eLpNorm_measure_zero, ENNReal.zero_lt_top]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      c : E
      h : Not (Eq μ 0)
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => c) Top.top μ) Top.top
    -/
  · rw [eLpNorm_const _ ENNReal.top_ne_zero h]
    /-
      case neg
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      c : E
      h : Not (Eq μ 0)
      ⊢ LT.lt (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 T …
    -/
    simp only [ENNReal.top_toReal, div_zero, ENNReal.rpow_zero, mul_one, ENNReal.coe_lt_top]
    /-
      🎉 no goals
    -/


theorem memℒp_const_iff {p : ℝ≥0∞} {c : E} (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    Memℒp (fun _ : α => c) p μ ↔ c = 0 ∨ μ Set.univ < ∞ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    c : E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ Iff (MeasureTheory.Memℒp (fun x => c) p μ) (Or (Eq c 0) (LT.lt (μ Set.univ)  …
  -/
  rw [← eLpNorm_const_lt_top_iff hp_ne_zero hp_ne_top]
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    c : E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ Iff (MeasureTheory.Memℒp (fun x => c) p μ) (LT.lt (MeasureTheory.eLpNorm (fu …
  -/
  exact ⟨fun h => h.2, fun h => ⟨aestronglyMeasurable_const, h⟩⟩
  /-
    🎉 no goals
  -/


lemma eLpNorm'_mono_nnnorm_ae {f : α → F} {g : α → G} (hq : 0 ≤ q) (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ ‖g x‖₊) :
    eLpNorm' f q μ ≤ eLpNorm' g q μ := by
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    hq : LE.le 0 q
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    ⊢ LE.le (MeasureTheory.eLpNorm' f q μ) (MeasureTheory.eLpNorm' g q μ)
  -/
  simp only [eLpNorm'_eq_lintegral_nnnorm]
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    hq : LE.le 0 q
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnn …
  -/
  gcongr ?_ ^ (1/q)
  /-
    case h₁
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    hq : LE.le 0 q
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) …
  -/
  refine lintegral_mono_ae (h.mono fun x hx => ?_)
  /-
    case h₁
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    hq : LE.le 0 q
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    x : α
    hx : LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g x))
    ⊢ LE.le (HPow.hPow (↑(NNNorm.nnnorm (f x))) q) (HPow.hPow (↑(NNNorm.nnnorm (g  …
  -/
  gcongr
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_mono_nnnorm_ae := eLpNorm'_mono_nnnorm_ae


theorem eLpNorm'_mono_ae {f : α → F} {g : α → G} (hq : 0 ≤ q) (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖g x‖) :
    eLpNorm' f q μ ≤ eLpNorm' g q μ :=
  eLpNorm'_mono_nnnorm_ae hq h


@[deprecated (since := "2024-07-27")]
alias snorm'_mono_ae := eLpNorm'_mono_ae


theorem eLpNorm'_congr_nnnorm_ae {f g : α → F} (hfg : ∀ᵐ x ∂μ, ‖f x‖₊ = ‖g x‖₊) :
    eLpNorm' f q μ = eLpNorm' g q μ := by
  have : (fun x => (‖f x‖₊ : ℝ≥0∞) ^ q) =ᵐ[μ] fun x => (‖g x‖₊ : ℝ≥0∞) ^ q :=
    hfg.mono fun x hx => by simp_rw [hx]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f g : α → F
    hfg : Filter.Eventually (fun x => Eq (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g x …
    this : (MeasureTheory.ae μ).EventuallyEq (fun x => HPow.hPow (↑(NNNorm.nnnorm  …
    ⊢ Eq (MeasureTheory.eLpNorm' f q μ) (MeasureTheory.eLpNorm' g q μ)
  -/
  simp only [eLpNorm'_eq_lintegral_nnnorm, lintegral_congr_ae this]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_congr_nnnorm_ae := eLpNorm'_congr_nnnorm_ae


theorem eLpNorm'_congr_norm_ae {f g : α → F} (hfg : ∀ᵐ x ∂μ, ‖f x‖ = ‖g x‖) :
    eLpNorm' f q μ = eLpNorm' g q μ :=
  eLpNorm'_congr_nnnorm_ae <| hfg.mono fun _x hx => NNReal.eq hx


@[deprecated (since := "2024-07-27")]
alias snorm'_congr_norm_ae := eLpNorm'_congr_norm_ae


theorem eLpNorm'_congr_ae {f g : α → F} (hfg : f =ᵐ[μ] g) : eLpNorm' f q μ = eLpNorm' g q μ :=
  eLpNorm'_congr_nnnorm_ae (hfg.fun_comp _)


@[deprecated (since := "2024-07-27")]
alias snorm'_congr_ae := eLpNorm'_congr_ae


theorem eLpNormEssSup_congr_ae {f g : α → F} (hfg : f =ᵐ[μ] g) :
    eLpNormEssSup f μ = eLpNormEssSup g μ :=
  essSup_congr_ae (hfg.fun_comp (((↑) : ℝ≥0 → ℝ≥0∞) ∘ nnnorm))


@[deprecated (since := "2024-07-27")]
alias snormEssSup_congr_ae := eLpNormEssSup_congr_ae


theorem eLpNormEssSup_mono_nnnorm_ae {f g : α → F} (hfg : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ ‖g x‖₊) :
    eLpNormEssSup f μ ≤ eLpNormEssSup g μ :=
  essSup_mono_ae <| hfg.mono fun _x hx => ENNReal.coe_le_coe.mpr hx


@[deprecated (since := "2024-07-27")]
alias snormEssSup_mono_nnnorm_ae := eLpNormEssSup_mono_nnnorm_ae


theorem eLpNorm_mono_nnnorm_ae {f : α → F} {g : α → G} (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ ‖g x‖₊) :
    eLpNorm f p μ ≤ eLpNorm g p μ := by
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p μ)
  -/
  simp only [eLpNorm]
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
    ⊢ LE.le (ite (Eq p 0) 0 (ite (Eq p Top.top) (MeasureTheory.eLpNormEssSup f μ)  …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
      h✝ : Eq p 0
      ⊢ LE.le 0 0
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
      h✝¹ : Not (Eq p 0)
      h✝ : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNormEssSup f μ) (MeasureTheory.eLpNormEssSup g μ)
    -/
  · exact essSup_mono_ae (h.mono fun x hx => ENNReal.coe_le_coe.mpr hx)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (g  …
      h✝¹ : Not (Eq p 0)
      h✝ : Not (Eq p Top.top)
      ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (MeasureTheory.eLpNorm' g p.toRe …
    -/
  · exact eLpNorm'_mono_nnnorm_ae ENNReal.toReal_nonneg h
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_mono_nnnorm_ae := eLpNorm_mono_nnnorm_ae


theorem eLpNorm_mono_ae {f : α → F} {g : α → G} (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖g x‖) :
    eLpNorm f p μ ≤ eLpNorm g p μ :=
  eLpNorm_mono_nnnorm_ae h


@[deprecated (since := "2024-07-27")]
alias snorm_mono_ae := eLpNorm_mono_ae


theorem eLpNorm_mono_ae_real {f : α → F} {g : α → ℝ} (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ g x) :
    eLpNorm f p μ ≤ eLpNorm g p μ :=
  eLpNorm_mono_ae <| h.mono fun _x hx =>
    hx.trans ((le_abs_self _).trans (Real.norm_eq_abs _).symm.le)


@[deprecated (since := "2024-07-27")]
alias snorm_mono_ae_real := eLpNorm_mono_ae_real


theorem eLpNorm_mono_nnnorm {f : α → F} {g : α → G} (h : ∀ x, ‖f x‖₊ ≤ ‖g x‖₊) :
    eLpNorm f p μ ≤ eLpNorm g p μ :=
  eLpNorm_mono_nnnorm_ae (Eventually.of_forall fun x => h x)


@[deprecated (since := "2024-07-27")]
alias snorm_mono_nnnorm := eLpNorm_mono_nnnorm


theorem eLpNorm_mono {f : α → F} {g : α → G} (h : ∀ x, ‖f x‖ ≤ ‖g x‖) :
    eLpNorm f p μ ≤ eLpNorm g p μ :=
  eLpNorm_mono_ae (Eventually.of_forall fun x => h x)


@[deprecated (since := "2024-07-27")]
alias snorm_mono := eLpNorm_mono


theorem eLpNorm_mono_real {f : α → F} {g : α → ℝ} (h : ∀ x, ‖f x‖ ≤ g x) :
    eLpNorm f p μ ≤ eLpNorm g p μ :=
  eLpNorm_mono_ae_real (Eventually.of_forall fun x => h x)


@[deprecated (since := "2024-07-27")]
alias snorm_mono_real := eLpNorm_mono_real


theorem eLpNormEssSup_le_of_ae_nnnorm_bound {f : α → F} {C : ℝ≥0} (hfC : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ C) :
    eLpNormEssSup f μ ≤ C :=
  essSup_le_of_ae_le (C : ℝ≥0∞) <| hfC.mono fun _x hx => ENNReal.coe_le_coe.mpr hx


@[deprecated (since := "2024-07-27")]
alias snormEssSup_le_of_ae_nnnorm_bound := eLpNormEssSup_le_of_ae_nnnorm_bound


theorem eLpNormEssSup_le_of_ae_bound {f : α → F} {C : ℝ} (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    eLpNormEssSup f μ ≤ ENNReal.ofReal C :=
  eLpNormEssSup_le_of_ae_nnnorm_bound <| hfC.mono fun _x hx => hx.trans C.le_coe_toNNReal


@[deprecated (since := "2024-07-27")]
alias snormEssSup_le_of_ae_bound := eLpNormEssSup_le_of_ae_bound


theorem eLpNormEssSup_lt_top_of_ae_nnnorm_bound {f : α → F} {C : ℝ≥0} (hfC : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ C) :
    eLpNormEssSup f μ < ∞ :=
  (eLpNormEssSup_le_of_ae_nnnorm_bound hfC).trans_lt ENNReal.coe_lt_top


@[deprecated (since := "2024-07-27")]
alias snormEssSup_lt_top_of_ae_nnnorm_bound := eLpNormEssSup_lt_top_of_ae_nnnorm_bound


theorem eLpNormEssSup_lt_top_of_ae_bound {f : α → F} {C : ℝ} (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    eLpNormEssSup f μ < ∞ :=
  (eLpNormEssSup_le_of_ae_bound hfC).trans_lt ENNReal.ofReal_lt_top


@[deprecated (since := "2024-07-27")]
alias snormEssSup_lt_top_of_ae_bound := eLpNormEssSup_lt_top_of_ae_bound


theorem eLpNorm_le_of_ae_nnnorm_bound {f : α → F} {C : ℝ≥0} (hfC : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ C) :
    eLpNorm f p μ ≤ C • μ Set.univ ^ p.toReal⁻¹ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul C (HPow.hPow (μ Set.univ) ( …
  -/
  rcases eq_zero_or_neZero μ with rfl | hμ
    /-
      case inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      inst✝ : NormedAddCommGroup F
      f : α → F
      C : NNReal
      hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
      ⊢ LE.le (MeasureTheory.eLpNorm f p 0) (HSMul.hSMul C (HPow.hPow (0 Set.univ) ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
    hμ : NeZero μ
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul C (HPow.hPow (μ Set.univ) ( …
  -/
  by_cases hp : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      C : NNReal
      hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
      hμ : NeZero μ
      hp : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul C (HPow.hPow (μ Set.univ) ( …
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
    hμ : NeZero μ
    hp : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul C (HPow.hPow (μ Set.univ) ( …
  -/
  have : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ ‖(C : ℝ)‖₊ := hfC.mono fun x hx => hx.trans_eq C.nnnorm_eq.symm
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
    hμ : NeZero μ
    hp : Not (Eq p 0)
    this : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm  …
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul C (HPow.hPow (μ Set.univ) ( …
  -/
  refine (eLpNorm_mono_ae this).trans_eq ?_
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) C) (MeasureTheor …
    hμ : NeZero μ
    hp : Not (Eq p 0)
    this : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (NNNorm.nnnorm  …
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => ↑C) p μ) (HSMul.hSMul C (HPow.hPow (μ Se …
  -/
  rw [eLpNorm_const _ hp (NeZero.ne μ), C.nnnorm_eq, one_div, ENNReal.smul_def, smul_eq_mul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_of_ae_nnnorm_bound := eLpNorm_le_of_ae_nnnorm_bound


theorem eLpNorm_le_of_ae_bound {f : α → F} {C : ℝ} (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    eLpNorm f p μ ≤ μ Set.univ ^ p.toReal⁻¹ * ENNReal.ofReal C := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : Real
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae …
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (HPow.hPow (μ Set.univ) (Inv. …
  -/
  rw [← mul_comm]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    C : Real
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae …
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HMul.hMul (ENNReal.ofReal C) (HPow.hPow …
  -/
  exact eLpNorm_le_of_ae_nnnorm_bound (hfC.mono fun x hx => hx.trans C.le_coe_toNNReal)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_of_ae_bound := eLpNorm_le_of_ae_bound


theorem eLpNorm_congr_nnnorm_ae {f : α → F} {g : α → G} (hfg : ∀ᵐ x ∂μ, ‖f x‖₊ = ‖g x‖₊) :
    eLpNorm f p μ = eLpNorm g p μ :=
  le_antisymm (eLpNorm_mono_nnnorm_ae <| EventuallyEq.le hfg)
    (eLpNorm_mono_nnnorm_ae <| (EventuallyEq.symm hfg).le)


@[deprecated (since := "2024-07-27")]
alias snorm_congr_nnnorm_ae := eLpNorm_congr_nnnorm_ae


theorem eLpNorm_congr_norm_ae {f : α → F} {g : α → G} (hfg : ∀ᵐ x ∂μ, ‖f x‖ = ‖g x‖) :
    eLpNorm f p μ = eLpNorm g p μ :=
  eLpNorm_congr_nnnorm_ae <| hfg.mono fun _x hx => NNReal.eq hx


@[deprecated (since := "2024-07-27")]
alias snorm_congr_norm_ae := eLpNorm_congr_norm_ae


open scoped symmDiff in
theorem eLpNorm_indicator_sub_indicator (s t : Set α) (f : α → E) :
    eLpNorm (s.indicator f - t.indicator f) p μ = eLpNorm ((s ∆ t).indicator f) p μ :=
  eLpNorm_congr_norm_ae <| ae_of_all _ fun x ↦ by
    /-
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s t : Set α
      f : α → E
      x : α
      ⊢ Eq (Norm.norm (HSub.hSub (s.indicator f) (t.indicator f) x)) (Norm.norm ((sy …
    -/
    simp only [Pi.sub_apply, Set.apply_indicator_symmDiff norm_neg]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_sub_indicator := eLpNorm_indicator_sub_indicator


@[simp]
theorem eLpNorm'_norm {f : α → F} :
                                                         /-
                                                           α : Type u_1
                                                           F : Type u_4
                                                           m0 : MeasurableSpace α
                                                           q : Real
                                                           μ : MeasureTheory.Measure α
                                                           inst✝ : NormedAddCommGroup F
                                                           f : α → F
                                                           ⊢ Eq (MeasureTheory.eLpNorm' (fun a => Norm.norm (f a)) q μ) (MeasureTheory.eL …
                                                         -/
    eLpNorm' (fun a => ‖f a‖) q μ = eLpNorm' f q μ := by simp [eLpNorm'_eq_lintegral_nnnorm]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-07-27")]
alias snorm'_norm := eLpNorm'_norm


@[simp]
theorem eLpNorm_norm (f : α → F) : eLpNorm (fun x => ‖f x‖) p μ = eLpNorm f p μ :=
  eLpNorm_congr_norm_ae <| Eventually.of_forall fun _ => norm_norm _


@[deprecated (since := "2024-07-27")]
alias snorm_norm := eLpNorm_norm


theorem eLpNorm'_norm_rpow (f : α → F) (p q : ℝ) (hq_pos : 0 < q) :
    eLpNorm' (fun x => ‖f x‖ ^ q) p μ = eLpNorm' f (p * q) μ ^ q := by
  simp_rw [eLpNorm'_eq_lintegral_nnnorm, ← ENNReal.rpow_mul, ← one_div_mul_one_div, one_div,
    mul_assoc, inv_mul_cancel₀ hq_pos.ne.symm, mul_one, ← ofReal_norm_eq_coe_nnnorm,
    Real.norm_eq_abs, abs_eq_self.mpr (Real.rpow_nonneg (norm_nonneg _) _), mul_comm p,
    ← ENNReal.ofReal_rpow_of_nonneg (norm_nonneg _) hq_pos.le, ENNReal.rpow_mul]


@[deprecated (since := "2024-07-27")]
alias snorm'_norm_rpow := eLpNorm'_norm_rpow


theorem eLpNorm_norm_rpow (f : α → F) (hq_pos : 0 < q) :
    eLpNorm (fun x => ‖f x‖ ^ q) p μ = eLpNorm f (p * ENNReal.ofReal q) μ ^ q := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q) p μ) (HPo …
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q) p μ) (HPo …
    -/
  · simp [h0, ENNReal.zero_rpow_of_pos hq_pos]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q) p μ) (HPo …
  -/
  by_cases hp_top : p = ∞
  · simp only [hp_top, eLpNorm_exponent_top, ENNReal.top_mul', hq_pos.not_le,
      ENNReal.ofReal_eq_zero, if_false, eLpNorm_exponent_top, eLpNormEssSup_eq_essSup_nnnorm]
    have h_rpow :
      essSup (fun x : α => (‖‖f x‖ ^ q‖₊ : ℝ≥0∞)) μ =
        essSup (fun x : α => (‖f x‖₊ : ℝ≥0∞) ^ q) μ := by
      congr
      ext1 x
      conv_rhs => rw [← nnnorm_norm]
      rw [← ENNReal.coe_rpow_of_nonneg _ hq_pos.le, ENNReal.coe_inj]
      ext
      push_cast
      rw [Real.norm_rpow_of_nonneg (norm_nonneg _)]
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      h_rpow : Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q)) …
      ⊢ Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q))) μ) (H …
    -/
    rw [h_rpow]
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      h_rpow : Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q)) …
      ⊢ Eq (essSup (fun x => HPow.hPow (↑(NNNorm.nnnorm (f x))) q) μ) (HPow.hPow (es …
    -/
    have h_rpow_mono := ENNReal.strictMono_rpow_of_pos hq_pos
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      h_rpow : Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q)) …
      h_rpow_mono : StrictMono fun x => HPow.hPow x q
      ⊢ Eq (essSup (fun x => HPow.hPow (↑(NNNorm.nnnorm (f x))) q) μ) (HPow.hPow (es …
    -/
    have h_rpow_surj := (ENNReal.rpow_left_bijective hq_pos.ne.symm).2
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      h_rpow : Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q)) …
      h_rpow_mono : StrictMono fun x => HPow.hPow x q
      h_rpow_surj : Function.Surjective fun y => HPow.hPow y q
      ⊢ Eq (essSup (fun x => HPow.hPow (↑(NNNorm.nnnorm (f x))) q) μ) (HPow.hPow (es …
    -/
    let iso := h_rpow_mono.orderIsoOfSurjective _ h_rpow_surj
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      h_rpow : Eq (essSup (fun x => ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) q)) …
      h_rpow_mono : StrictMono fun x => HPow.hPow x q
      h_rpow_surj : Function.Surjective fun y => HPow.hPow y q
      iso : OrderIso ENNReal ENNReal := StrictMono.orderIsoOfSurjective (fun x => HP …
      ⊢ Eq (essSup (fun x => HPow.hPow (↑(NNNorm.nnnorm (f x))) q) μ) (HPow.hPow (es …
    -/
    exact (iso.essSup_apply (fun x => (‖f x‖₊ : ℝ≥0∞)) μ).symm
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q) p μ) (HPo …
  -/
  rw [eLpNorm_eq_eLpNorm' h0 hp_top, eLpNorm_eq_eLpNorm' _ _]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm' (fun x => HPow.hPow (Norm.norm (f x)) q) p.toReal …
  -/
  swap
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ Ne (HMul.hMul p (ENNReal.ofReal q)) 0
    -/
  · refine mul_ne_zero h0 ?_
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hq_pos : LT.lt 0 q
      h0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ Ne (ENNReal.ofReal q) 0
    -/
    rwa [Ne, ENNReal.ofReal_eq_zero, not_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm' (fun x => HPow.hPow (Norm.norm (f x)) q) p.toReal …
  -/
  swap; · exact ENNReal.mul_ne_top hp_top ENNReal.ofReal_ne_top
          /-
            🎉 no goals
          -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm' (fun x => HPow.hPow (Norm.norm (f x)) q) p.toReal …
  -/
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal hq_pos.le]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hq_pos : LT.lt 0 q
    h0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm' (fun x => HPow.hPow (Norm.norm (f x)) q) p.toReal …
  -/
  exact eLpNorm'_norm_rpow f p.toReal q hq_pos
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_norm_rpow := eLpNorm_norm_rpow


theorem eLpNorm_congr_ae {f g : α → F} (hfg : f =ᵐ[μ] g) : eLpNorm f p μ = eLpNorm g p μ :=
  eLpNorm_congr_norm_ae <| hfg.mono fun _x hx => hx ▸ rfl


@[deprecated (since := "2024-07-27")]
alias snorm_congr_ae := eLpNorm_congr_ae


theorem memℒp_congr_ae {f g : α → E} (hfg : f =ᵐ[μ] g) : Memℒp f p μ ↔ Memℒp g p μ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : α → E
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Iff (MeasureTheory.Memℒp f p μ) (MeasureTheory.Memℒp g p μ)
  -/
  simp only [Memℒp, eLpNorm_congr_ae hfg, aestronglyMeasurable_congr hfg]
  /-
    🎉 no goals
  -/


theorem Memℒp.ae_eq {f g : α → E} (hfg : f =ᵐ[μ] g) (hf_Lp : Memℒp f p μ) : Memℒp g p μ :=
  (memℒp_congr_ae hfg).1 hf_Lp


theorem Memℒp.of_le {f : α → E} {g : α → F} (hg : Memℒp g p μ) (hf : AEStronglyMeasurable f μ)
    (hfg : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖g x‖) : Memℒp f p μ :=
  ⟨hf, (eLpNorm_mono_ae hfg).trans_lt hg.eLpNorm_lt_top⟩


alias Memℒp.mono := Memℒp.of_le


theorem Memℒp.mono' {f : α → E} {g : α → ℝ} (hg : Memℒp g p μ) (hf : AEStronglyMeasurable f μ)
    (h : ∀ᵐ a ∂μ, ‖f a‖ ≤ g a) : Memℒp f p μ :=
  hg.mono hf <| h.mono fun _x hx => le_trans hx (le_abs_self _)


theorem Memℒp.congr_norm {f : α → E} {g : α → F} (hf : Memℒp f p μ) (hg : AEStronglyMeasurable g μ)
    (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) : Memℒp g p μ :=
  hf.mono hg <| EventuallyEq.le <| EventuallyEq.symm h


theorem memℒp_congr_norm {f : α → E} {g : α → F} (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) : Memℒp f p μ ↔ Memℒp g p μ :=
  ⟨fun h2f => h2f.congr_norm hg h, fun h2g => h2g.congr_norm hf <| EventuallyEq.symm h⟩


theorem memℒp_top_of_bound {f : α → E} (hf : AEStronglyMeasurable f μ) (C : ℝ)
    (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : Memℒp f ∞ μ :=
  ⟨hf, by
    /-
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      C : Real
      hfC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae …
      ⊢ LT.lt (MeasureTheory.eLpNorm f Top.top μ) Top.top
    -/
    rw [eLpNorm_exponent_top]
    /-
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      C : Real
      hfC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae …
      ⊢ LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
    -/
    exact eLpNormEssSup_lt_top_of_ae_bound hfC⟩
    /-
      🎉 no goals
    -/


theorem Memℒp.of_bound [IsFiniteMeasure μ] {f : α → E} (hf : AEStronglyMeasurable f μ) (C : ℝ)
    (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : Memℒp f p μ :=
  (memℒp_const C).of_le hf (hfC.mono fun _x hx => le_trans hx (le_abs_self _))


theorem memℒp_of_bounded [IsFiniteMeasure μ]
    {a b : ℝ} {f : α → ℝ} (h : ∀ᵐ x ∂μ, f x ∈ Set.Icc a b)
    (hX : AEStronglyMeasurable f μ) (p : ENNReal) : Memℒp f p μ :=
  have ha : ∀ᵐ x ∂μ, a ≤ f x := h.mono fun ω h => h.1
  have hb : ∀ᵐ x ∂μ, f x ≤ b := h.mono fun ω h => h.2
                                           /-
                                             α : Type u_1
                                             m0 : MeasurableSpace α
                                             μ : MeasureTheory.Measure α
                                             inst✝ : MeasureTheory.IsFiniteMeasure μ
                                             a b : Real
                                             f : α → Real
                                             h : Filter.Eventually (fun x => Membership.mem (Set.Icc a b) (f x)) (MeasureTh …
                                             hX : MeasureTheory.AEStronglyMeasurable f μ
                                             p : ENNReal
                                             ha : Filter.Eventually (fun x => LE.le a (f x)) (MeasureTheory.ae μ)
                                             hb : Filter.Eventually (fun x => LE.le (f x) b) (MeasureTheory.ae μ)
                                             ⊢ Filter.Eventually (fun a_1 => LE.le (Norm.norm (f a_1)) (Max.max (abs a) (ab …
                                           -/
  (memℒp_const (max |a| |b|)).mono' hX (by filter_upwards [ha, hb] with x using abs_le_max_abs_abs)
                                           /-
                                             🎉 no goals
                                           -/


@[gcongr, mono]
theorem eLpNorm'_mono_measure (f : α → F) (hμν : ν ≤ μ) (hq : 0 ≤ q) :
    eLpNorm' f q ν ≤ eLpNorm' f q μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hq : LE.le 0 q
    ⊢ LE.le (MeasureTheory.eLpNorm' f q ν) (MeasureTheory.eLpNorm' f q μ)
  -/
  simp_rw [eLpNorm']
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hq : LE.le 0 q
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral ν fun a => HPow.hPow (ENorm.enorm  …
  -/
  gcongr
  /-
    case h₁
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hq : LE.le 0 q
    ⊢ LE.le (MeasureTheory.lintegral ν fun a => HPow.hPow (ENorm.enorm (f a)) q) ( …
  -/
  exact lintegral_mono' hμν le_rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_mono_measure := eLpNorm'_mono_measure


@[gcongr, mono]
theorem eLpNormEssSup_mono_measure (f : α → F) (hμν : ν ≪ μ) :
    eLpNormEssSup f ν ≤ eLpNormEssSup f μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : ν.AbsolutelyContinuous μ
    ⊢ LE.le (MeasureTheory.eLpNormEssSup f ν) (MeasureTheory.eLpNormEssSup f μ)
  -/
  simp_rw [eLpNormEssSup]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : ν.AbsolutelyContinuous μ
    ⊢ LE.le (essSup (fun x => ENorm.enorm (f x)) ν) (essSup (fun x => ENorm.enorm  …
  -/
  exact essSup_mono_measure hμν
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_mono_measure := eLpNormEssSup_mono_measure


@[gcongr, mono]
theorem eLpNorm_mono_measure (f : α → F) (hμν : ν ≤ μ) : eLpNorm f p ν ≤ eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    ⊢ LE.le (MeasureTheory.eLpNorm f p ν) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ ν : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hμν : LE.le ν μ
      hp0 : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm f p ν) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hp0 : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm f p ν) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ ν : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hμν : LE.le ν μ
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f p ν) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [hp_top, eLpNormEssSup_mono_measure f (Measure.absolutelyContinuous_of_le hμν)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm f p ν) (MeasureTheory.eLpNorm f p μ)
  -/
  simp_rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hμν : LE.le ν μ
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal ν) (MeasureTheory.eLpNorm' f p.toRe …
  -/
  exact eLpNorm'_mono_measure f hμν ENNReal.toReal_nonneg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_mono_measure := eLpNorm_mono_measure


theorem Memℒp.mono_measure {f : α → E} (hμν : ν ≤ μ) (hf : Memℒp f p μ) : Memℒp f p ν :=
  ⟨hf.1.mono_measure hμν, (eLpNorm_mono_measure f hμν).trans_lt hf.2⟩


lemma eLpNorm_restrict_le (f : α → F) (p : ℝ≥0∞) (μ : Measure α) (s : Set α) :
    eLpNorm f p (μ.restrict s) ≤ eLpNorm f p μ :=
  eLpNorm_mono_measure f Measure.restrict_le_self


@[deprecated (since := "2024-07-27")]
alias snorm_restrict_le := eLpNorm_restrict_le


/-- For a function `f` with support in `s`, the Lᵖ norms of `f` with respect to `μ` and
`μ.restrict s` are the same. -/
theorem eLpNorm_restrict_eq_of_support_subset {s : Set α} {f : α → F} (hsf : f.support ⊆ s) :
    eLpNorm f p (μ.restrict s) = eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hsf : HasSubset.Subset (Function.support f) s
    hp0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp only [hp_top, eLpNorm_exponent_top, eLpNormEssSup_eq_essSup_nnnorm]
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (essSup (fun x => ↑(NNNorm.nnnorm (f x))) (μ.restrict s)) (essSup (fun x  …
    -/
    apply ENNReal.essSup_restrict_eq_of_support_subset
    /-
      case pos.hsf
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ HasSubset.Subset (Function.support fun x => ↑(NNNorm.nnnorm (f x))) s
    -/
    apply Function.support_subset_iff.2 (fun x hx ↦ ?_)
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      x : α
      hx : Ne (↑(NNNorm.nnnorm (f x))) 0
      ⊢ Membership.mem s x
    -/
    simp only [ne_eq, ENNReal.coe_eq_zero, nnnorm_eq_zero] at hx
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      x : α
      hx : Not (Eq (f x) 0)
      ⊢ Membership.mem s x
    -/
    exact Function.support_subset_iff.1 hsf x hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp_rw [eLpNorm_eq_eLpNorm' hp0 hp_top, eLpNorm'_eq_lintegral_nnnorm]
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral (μ.restrict s) fun a => HPow.hPow (↑( …
    -/
    congr 1
    /-
      case neg.e_a
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HPow.hPow (↑(NNNorm.nnno …
    -/
    apply setLIntegral_eq_of_support_subset
    /-
      case neg.e_a.hsf
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      ⊢ HasSubset.Subset (Function.support fun x => HPow.hPow (↑(NNNorm.nnnorm (f x) …
    -/
    have : ¬(p.toReal ≤ 0) := by simpa only [not_le] using ENNReal.toReal_pos hp0 hp_top
    /-
      case neg.e_a.hsf
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hsf : HasSubset.Subset (Function.support f) s
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      this : Not (LE.le p.toReal 0)
      ⊢ HasSubset.Subset (Function.support fun x => HPow.hPow (↑(NNNorm.nnnorm (f x) …
    -/
    simpa [this] using hsf
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_restrict_eq_of_support_subset := eLpNorm_restrict_eq_of_support_subset


theorem Memℒp.restrict (s : Set α) {f : α → E} (hf : Memℒp f p μ) : Memℒp f p (μ.restrict s) :=
  hf.mono_measure Measure.restrict_le_self


theorem eLpNorm'_smul_measure {p : ℝ} (hp : 0 ≤ p) {f : α → F} (c : ℝ≥0∞) :
    eLpNorm' f p (c • μ) = c ^ (1 / p) * eLpNorm' f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : Real
    hp : LE.le 0 p
    f : α → F
    c : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm' f p (HSMul.hSMul c μ)) (HMul.hMul (HPow.hPow c (H …
  -/
  rw [eLpNorm', lintegral_smul_measure, ENNReal.mul_rpow_of_nonneg, eLpNorm']
  /-
    case hz
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : Real
    hp : LE.le 0 p
    f : α → F
    c : ENNReal
    ⊢ LE.le 0 (HDiv.hDiv 1 p)
  -/
  simp [hp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_smul_measure := eLpNorm'_smul_measure


@[simp] lemma eLpNormEssSup_smul_measure (hc : c ≠ 0) (f : α → F) :
    eLpNormEssSup f (c • μ) = eLpNormEssSup f μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    R : Type u_6
    inst✝³ : Zero R
    inst✝² : SMulWithZero R ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : NoZeroSMulDivisors R ENNReal
    c : R
    hc : Ne c 0
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNormEssSup f (HSMul.hSMul c μ)) (MeasureTheory.eLpNormE …
  -/
  simp_rw [eLpNormEssSup]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup F
    R : Type u_6
    inst✝³ : Zero R
    inst✝² : SMulWithZero R ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : NoZeroSMulDivisors R ENNReal
    c : R
    hc : Ne c 0
    f : α → F
    ⊢ Eq (essSup (fun x => ENorm.enorm (f x)) (HSMul.hSMul c μ)) (essSup (fun x => …
  -/
  exact essSup_smul_measure hc _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_smul_measure := eLpNormEssSup_smul_measure


/-- Use `eLpNorm_smul_measure_of_ne_top` instead. -/
private theorem eLpNorm_smul_measure_of_ne_zero_of_ne_top {p : ℝ≥0∞} (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) {f : α → F} (c : ℝ≥0∞) :
    eLpNorm f p (c • μ) = c ^ (1 / p).toReal • eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  simp_rw [eLpNorm_eq_eLpNorm' hp_ne_zero hp_ne_top]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm' f p.toReal (HSMul.hSMul c μ)) (HSMul.hSMul (HPow. …
  -/
  rw [eLpNorm'_smul_measure ENNReal.toReal_nonneg]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (HMul.hMul (HPow.hPow c (HDiv.hDiv 1 p.toReal)) (MeasureTheory.eLpNorm' f …
  -/
  congr
  /-
    case e_a.e_a
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (HDiv.hDiv 1 p.toReal) (HDiv.hDiv 1 p).toReal
  -/
  simp_rw [one_div]
  /-
    case e_a.e_a
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (Inv.inv p.toReal) (Inv.inv p).toReal
  -/
  rw [ENNReal.toReal_inv]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_smul_measure_of_ne_zero_of_ne_top := eLpNorm_smul_measure_of_ne_zero_of_ne_top


/-- See `eLpNorm_smul_measure_of_ne_zero'` for a version with scalar multiplication by `ℝ≥0`. -/
theorem eLpNorm_smul_measure_of_ne_zero {c : ℝ≥0∞} (hc : c ≠ 0) (f : α → F) (p : ℝ≥0∞)
    (μ : Measure α) : eLpNorm f p (c • μ) = c ^ (1 / p).toReal • eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    c : ENNReal
    hc : Ne c 0
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup F
      c : ENNReal
      hc : Ne c 0
      f : α → F
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    c : ENNReal
    hc : Ne c 0
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      inst✝ : NormedAddCommGroup F
      c : ENNReal
      hc : Ne c 0
      f : α → F
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
    -/
  · simp [hp_top, eLpNormEssSup_smul_measure hc]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    c : ENNReal
    hc : Ne c 0
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  exact eLpNorm_smul_measure_of_ne_zero_of_ne_top hp0 hp_top c
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_smul_measure_of_ne_zero := eLpNorm_smul_measure_of_ne_zero


/-- See `eLpNorm_smul_measure_of_ne_zero` for a version with scalar multiplication by `ℝ≥0∞`. -/
lemma eLpNorm_smul_measure_of_ne_zero' {c : ℝ≥0} (hc : c ≠ 0) (f : α → F) (p : ℝ≥0∞)
    (μ : Measure α) : eLpNorm f p (c • μ) = c ^ p.toReal⁻¹ • eLpNorm f p μ :=
                                                                            /-
                                                                              α : Type u_1
                                                                              F : Type u_4
                                                                              m0 : MeasurableSpace α
                                                                              inst✝ : NormedAddCommGroup F
                                                                              c : NNReal
                                                                              hc : Ne c 0
                                                                              f : α → F
                                                                              p : ENNReal
                                                                              μ : MeasureTheory.Measure α
                                                                              ⊢ Eq (HSMul.hSMul (HPow.hPow (↑c) (HDiv.hDiv 1 p).toReal) (MeasureTheory.eLpNo …
                                                                            -/
  (eLpNorm_smul_measure_of_ne_zero (ENNReal.coe_ne_zero.2 hc) ..).trans (by simp; norm_cast)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- See `eLpNorm_smul_measure_of_ne_top'` for a version with scalar multiplication by `ℝ≥0`. -/
theorem eLpNorm_smul_measure_of_ne_top {p : ℝ≥0∞} (hp_ne_top : p ≠ ∞) (f : α → F) (c : ℝ≥0∞) :
    eLpNorm f p (c • μ) = c ^ (1 / p).toReal • eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    p : ENNReal
    hp_ne_top : Ne p Top.top
    f : α → F
    c : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      hp_ne_top : Ne p Top.top
      f : α → F
      c : ENNReal
      hp0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      hp_ne_top : Ne p Top.top
      f : α → F
      c : ENNReal
      hp0 : Not (Eq p 0)
      ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
    -/
  · exact eLpNorm_smul_measure_of_ne_zero_of_ne_top hp0 hp_ne_top c
    /-
      🎉 no goals
    -/


/-- See `eLpNorm_smul_measure_of_ne_top'` for a version with scalar multiplication by `ℝ≥0∞`. -/
lemma eLpNorm_smul_measure_of_ne_top' (hp : p ≠ ∞) (c : ℝ≥0) (f : α → F) :
    eLpNorm f p (c • μ) = c ^ p.toReal⁻¹ • eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p Top.top
    c : NNReal
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  have : 0 ≤ p.toReal⁻¹ := by positivity
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p Top.top
    c : NNReal
    f : α → F
    this : LE.le 0 (Inv.inv p.toReal)
    ⊢ Eq (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) (HSMul.hSMul (HPow.hPow c ( …
  -/
  refine (eLpNorm_smul_measure_of_ne_top hp ..).trans ?_
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p Top.top
    c : NNReal
    f : α → F
    this : LE.le 0 (Inv.inv p.toReal)
    ⊢ Eq (HSMul.hSMul (HPow.hPow (↑ENNReal.ofNNRealHom.toMonoidWithZeroHom c) (HDi …
  -/
  simp [ENNReal.smul_def, ENNReal.coe_rpow_of_nonneg, this]
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-07-27")]
alias snorm_smul_measure_of_ne_top := eLpNorm_smul_measure_of_ne_top


theorem eLpNorm_one_smul_measure {f : α → F} (c : ℝ≥0∞) :
    eLpNorm f 1 (c • μ) = c * eLpNorm f 1 μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    c : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm f 1 (HSMul.hSMul c μ)) (HMul.hMul c (MeasureTheory …
  -/
  rw [@eLpNorm_smul_measure_of_ne_top _ _ _ μ _ 1 (@ENNReal.coe_ne_top 1) f c]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    c : ENNReal
    ⊢ Eq (HSMul.hSMul (HPow.hPow c (1 / 1).toReal) (MeasureTheory.eLpNorm f 1 μ))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_one_smul_measure := eLpNorm_one_smul_measure


theorem Memℒp.of_measure_le_smul {μ' : Measure α} (c : ℝ≥0∞) (hc : c ≠ ∞) (hμ'_le : μ' ≤ c • μ)
    {f : α → E} (hf : Memℒp f p μ) : Memℒp f p μ' := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ MeasureTheory.Memℒp f p μ'
  -/
  refine ⟨hf.1.mono_ac (Measure.absolutelyContinuous_of_le_smul hμ'_le), ?_⟩
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ') Top.top
  -/
  refine (eLpNorm_mono_measure f hμ'_le).trans_lt ?_
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ LT.lt (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) Top.top
  -/
  by_cases hc0 : c = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      μ' : MeasureTheory.Measure α
      c : ENNReal
      hc : Ne c Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c μ)
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      hc0 : Eq c 0
      ⊢ LT.lt (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) Top.top
    -/
  · simp [hc0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    hc0 : Not (Eq c 0)
    ⊢ LT.lt (MeasureTheory.eLpNorm f p (HSMul.hSMul c μ)) Top.top
  -/
  rw [eLpNorm_smul_measure_of_ne_zero hc0, smul_eq_mul]
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    hc0 : Not (Eq c 0)
    ⊢ LT.lt (HMul.hMul (HPow.hPow c (HDiv.hDiv 1 p).toReal) (MeasureTheory.eLpNorm …
  -/
  refine ENNReal.mul_lt_top (Ne.lt_top ?_) hf.2
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    hc0 : Not (Eq c 0)
    ⊢ Ne (HPow.hPow c (HDiv.hDiv 1 p).toReal) Top.top
  -/
  simp [hc, hc0]
  /-
    🎉 no goals
  -/


theorem Memℒp.smul_measure {f : α → E} {c : ℝ≥0∞} (hf : Memℒp f p μ) (hc : c ≠ ∞) :
    Memℒp f p (c • μ) :=
  hf.of_measure_le_smul c hc le_rfl


theorem eLpNorm_one_add_measure (f : α → F) (μ ν : Measure α) :
    eLpNorm f 1 (μ + ν) = eLpNorm f 1 μ + eLpNorm f 1 ν := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    μ ν : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.eLpNorm f 1 (HAdd.hAdd μ ν)) (HAdd.hAdd (MeasureTheory.eLp …
  -/
  simp_rw [eLpNorm_one_eq_lintegral_nnnorm]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    f : α → F
    μ ν : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.lintegral (HAdd.hAdd μ ν) fun x => ↑(NNNorm.nnnorm (f x))) …
  -/
  rw [lintegral_add_measure _ μ ν]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_one_add_measure := eLpNorm_one_add_measure


theorem eLpNorm_le_add_measure_right (f : α → F) (μ ν : Measure α) {p : ℝ≥0∞} :
    eLpNorm f p μ ≤ eLpNorm f p (μ + ν) :=
  eLpNorm_mono_measure f <| Measure.le_add_right <| le_refl _


@[deprecated (since := "2024-07-27")]
alias snorm_le_add_measure_right := eLpNorm_le_add_measure_right


theorem eLpNorm_le_add_measure_left (f : α → F) (μ ν : Measure α) {p : ℝ≥0∞} :
    eLpNorm f p ν ≤ eLpNorm f p (μ + ν) :=
  eLpNorm_mono_measure f <| Measure.le_add_left <| le_refl _


@[deprecated (since := "2024-07-27")]
alias snorm_le_add_measure_left := eLpNorm_le_add_measure_left


lemma eLpNormEssSup_eq_iSup (hμ : ∀ a, μ {a} ≠ 0) (f : α → E) : eLpNormEssSup f μ = ⨆ a, ↑‖f a‖₊ :=
  essSup_eq_iSup hμ _


@[simp] lemma eLpNormEssSup_count [MeasurableSingletonClass α] (f : α → E) :
    eLpNormEssSup f .count = ⨆ a, ↑‖f a‖₊ := essSup_count _


theorem Memℒp.left_of_add_measure {f : α → E} (h : Memℒp f p (μ + ν)) : Memℒp f p μ :=
  h.mono_measure <| Measure.le_add_right <| le_refl _


theorem Memℒp.right_of_add_measure {f : α → E} (h : Memℒp f p (μ + ν)) : Memℒp f p ν :=
  h.mono_measure <| Measure.le_add_left <| le_refl _


theorem Memℒp.norm {f : α → E} (h : Memℒp f p μ) : Memℒp (fun x => ‖f x‖) p μ :=
                                                                        /-
                                                                          α : Type u_1
                                                                          E : Type u_3
                                                                          m0 : MeasurableSpace α
                                                                          p : ENNReal
                                                                          μ : MeasureTheory.Measure α
                                                                          inst✝ : NormedAddCommGroup E
                                                                          f : α → E
                                                                          h : MeasureTheory.Memℒp f p μ
                                                                          x : α
                                                                          ⊢ LE.le (Norm.norm (Norm.norm (f x))) (Norm.norm (f x))
                                                                        -/
  h.of_le h.aestronglyMeasurable.norm (Eventually.of_forall fun x => by simp)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem memℒp_norm_iff {f : α → E} (hf : AEStronglyMeasurable f μ) :
    Memℒp (fun x => ‖f x‖) p μ ↔ Memℒp f p μ :=
                    /-
                      α : Type u_1
                      E : Type u_3
                      m0 : MeasurableSpace α
                      p : ENNReal
                      μ : MeasureTheory.Measure α
                      inst✝ : NormedAddCommGroup E
                      f : α → E
                      hf : MeasureTheory.AEStronglyMeasurable f μ
                      h : MeasureTheory.Memℒp (fun x => Norm.norm (f x)) p μ
                      ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
                    -/
  ⟨fun h => ⟨hf, by rw [← eLpNorm_norm]; exact h.2⟩, fun h => h.norm⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem eLpNorm'_eq_zero_of_ae_zero {f : α → F} (hq0_lt : 0 < q) (hf_zero : f =ᵐ[μ] 0) :
                             /-
                               α : Type u_1
                               F : Type u_4
                               m0 : MeasurableSpace α
                               q : Real
                               μ : MeasureTheory.Measure α
                               inst✝ : NormedAddCommGroup F
                               f : α → F
                               hq0_lt : LT.lt 0 q
                               hf_zero : (MeasureTheory.ae μ).EventuallyEq f 0
                               ⊢ Eq (MeasureTheory.eLpNorm' f q μ) 0
                             -/
    eLpNorm' f q μ = 0 := by rw [eLpNorm'_congr_ae hf_zero, eLpNorm'_zero hq0_lt]
                             /-
                               🎉 no goals
                             -/


@[deprecated (since := "2024-07-27")]
alias snorm'_eq_zero_of_ae_zero := eLpNorm'_eq_zero_of_ae_zero


theorem eLpNorm'_eq_zero_of_ae_zero' (hq0_ne : q ≠ 0) (hμ : μ ≠ 0) {f : α → F}
    (hf_zero : f =ᵐ[μ] 0) :
                             /-
                               α : Type u_1
                               F : Type u_4
                               m0 : MeasurableSpace α
                               q : Real
                               μ : MeasureTheory.Measure α
                               inst✝ : NormedAddCommGroup F
                               hq0_ne : Ne q 0
                               hμ : Ne μ 0
                               f : α → F
                               hf_zero : (MeasureTheory.ae μ).EventuallyEq f 0
                               ⊢ Eq (MeasureTheory.eLpNorm' f q μ) 0
                             -/
    eLpNorm' f q μ = 0 := by rw [eLpNorm'_congr_ae hf_zero, eLpNorm'_zero' hq0_ne hμ]
                             /-
                               🎉 no goals
                             -/


@[deprecated (since := "2024-07-27")]
alias snorm'_eq_zero_of_ae_zero' := eLpNorm'_eq_zero_of_ae_zero'


theorem ae_eq_zero_of_eLpNorm'_eq_zero {f : α → E} (hq0 : 0 ≤ q) (hf : AEStronglyMeasurable f μ)
    (h : eLpNorm' f q μ = 0) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hq0 : LE.le 0 q
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h : Eq (MeasureTheory.eLpNorm' f q μ) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  rw [eLpNorm'_eq_lintegral_nnnorm, ENNReal.rpow_eq_zero_iff] at h
  cases h with
  | inl h =>
    rw [lintegral_eq_zero_iff' (hf.ennnorm.pow_const q)] at h
    refine h.left.mono fun x hx => ?_
    rw [Pi.zero_apply, ENNReal.rpow_eq_zero_iff] at hx
    cases hx with
    | inl hx =>
      cases' hx with hx _
      rwa [← ENNReal.coe_zero, ENNReal.coe_inj, nnnorm_eq_zero] at hx
    | inr hx =>
      exact absurd hx.left ENNReal.coe_ne_top
  | inr h =>
    exfalso
    rw [one_div, inv_lt_zero] at h
    exact hq0.not_lt h.right


@[deprecated (since := "2024-07-27")]
alias ae_eq_zero_of_snorm'_eq_zero := ae_eq_zero_of_eLpNorm'_eq_zero


theorem eLpNorm'_eq_zero_iff (hq0_lt : 0 < q) {f : α → E} (hf : AEStronglyMeasurable f μ) :
    eLpNorm' f q μ = 0 ↔ f =ᵐ[μ] 0 :=
  ⟨ae_eq_zero_of_eLpNorm'_eq_zero (le_of_lt hq0_lt) hf, eLpNorm'_eq_zero_of_ae_zero hq0_lt⟩


@[deprecated (since := "2024-07-27")]
alias snorm'_eq_zero_iff := eLpNorm'_eq_zero_iff


theorem coe_nnnorm_ae_le_eLpNormEssSup {_ : MeasurableSpace α} (f : α → F) (μ : Measure α) :
    ∀ᵐ x ∂μ, (‖f x‖₊ : ℝ≥0∞) ≤ eLpNormEssSup f μ :=
  ENNReal.ae_le_essSup fun x => (‖f x‖₊ : ℝ≥0∞)


@[deprecated (since := "2024-07-27")]
alias coe_nnnorm_ae_le_snormEssSup := coe_nnnorm_ae_le_eLpNormEssSup


@[simp]
theorem eLpNormEssSup_eq_zero_iff {f : α → F} : eLpNormEssSup f μ = 0 ↔ f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    ⊢ Iff (Eq (MeasureTheory.eLpNormEssSup f μ) 0) ((MeasureTheory.ae μ).Eventuall …
  -/
  simp [EventuallyEq, eLpNormEssSup_eq_essSup_nnnorm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_eq_zero_iff := eLpNormEssSup_eq_zero_iff


theorem eLpNorm_eq_zero_iff {f : α → E} (hf : AEStronglyMeasurable f μ) (h0 : p ≠ 0) :
    eLpNorm f p μ = 0 ↔ f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h0 : Ne p 0
    ⊢ Iff (Eq (MeasureTheory.eLpNorm f p μ) 0) ((MeasureTheory.ae μ).EventuallyEq  …
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      h0 : Ne p 0
      h_top : Eq p Top.top
      ⊢ Iff (Eq (MeasureTheory.eLpNorm f p μ) 0) ((MeasureTheory.ae μ).EventuallyEq  …
    -/
  · rw [h_top, eLpNorm_exponent_top, eLpNormEssSup_eq_zero_iff]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h0 : Ne p 0
    h_top : Not (Eq p Top.top)
    ⊢ Iff (Eq (MeasureTheory.eLpNorm f p μ) 0) ((MeasureTheory.ae μ).EventuallyEq  …
  -/
  rw [eLpNorm_eq_eLpNorm' h0 h_top]
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h0 : Ne p 0
    h_top : Not (Eq p Top.top)
    ⊢ Iff (Eq (MeasureTheory.eLpNorm' f p.toReal μ) 0) ((MeasureTheory.ae μ).Event …
  -/
  exact eLpNorm'_eq_zero_iff (ENNReal.toReal_pos h0 h_top) hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_eq_zero_iff := eLpNorm_eq_zero_iff


theorem eLpNorm_eq_zero_of_ae_zero {f : α → E} (hf : f =ᵐ[μ] 0) : eLpNorm f p μ = 0 := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : (MeasureTheory.ae μ).EventuallyEq f 0
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) 0
  -/
  rw [← eLpNorm_zero (p := p) (μ := μ) (α := α) (F := E)]
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : (MeasureTheory.ae μ).EventuallyEq f 0
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm 0 p μ)
  -/
  exact eLpNorm_congr_ae hf
  /-
    🎉 no goals
  -/


theorem ae_le_eLpNormEssSup {f : α → F} : ∀ᵐ y ∂μ, ‖f y‖₊ ≤ eLpNormEssSup f μ :=
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) fun y  …
  -/
  ae_le_essSup
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias ae_le_snormEssSup := ae_le_eLpNormEssSup


lemma eLpNormEssSup_lt_top_iff_isBoundedUnder :
    eLpNormEssSup f μ < ⊤ ↔ IsBoundedUnder (· ≤ ·) (ae μ) fun x ↦ ‖f x‖₊ where
  mp h := ⟨(eLpNormEssSup f μ).toNNReal, by
    /-
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      h : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
      ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (MeasureTheory.eLpN …
    -/
    simp_rw [← ENNReal.coe_le_coe, ENNReal.coe_toNNReal h.ne]; exact ae_le_eLpNormEssSup⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/
            /-
              α : Type u_1
              F : Type u_4
              m0 : MeasurableSpace α
              μ : MeasureTheory.Measure α
              inst✝ : NormedAddCommGroup F
              f : α → F
              ⊢ (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) fun x …
            -/
  mpr := by rintro ⟨C, hC⟩; exact eLpNormEssSup_lt_top_of_ae_nnnorm_bound (C := C) hC
                            /-
                              🎉 no goals
                            -/


theorem meas_eLpNormEssSup_lt {f : α → F} : μ { y | eLpNormEssSup f μ < ‖f y‖₊ } = 0 :=
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) fun x  …
  -/
  meas_essSup_lt
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias meas_snormEssSup_lt := meas_eLpNormEssSup_lt


lemma eLpNorm_lt_top_of_finite [Finite α] [IsFiniteMeasure μ] : eLpNorm f p μ < ∞ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  obtain rfl | hp₀ := eq_or_ne p 0
    /-
      case inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup F
      f : α → F
      inst✝¹ : Finite α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ LT.lt (MeasureTheory.eLpNorm f 0 μ) Top.top
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  obtain rfl | hp := eq_or_ne p ∞
    /-
      case inr.inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup F
      f : α → F
      inst✝¹ : Finite α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp₀ : Ne Top.top 0
      ⊢ LT.lt (MeasureTheory.eLpNorm f Top.top μ) Top.top
    -/
  · simp only [eLpNorm_exponent_top, eLpNormEssSup_lt_top_iff_isBoundedUnder]
    /-
      case inr.inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup F
      f : α → F
      inst✝¹ : Finite α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp₀ : Ne Top.top 0
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) fun x  …
    -/
    exact .le_of_finite
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    hp : Ne p Top.top
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  rw [eLpNorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top hp₀ hp]
  /-
    case inr.inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    hp : Ne p Top.top
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) …
  -/
  refine IsFiniteMeasure.lintegral_lt_top_of_bounded_to_ennreal μ ?_
  /-
    case inr.inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    hp : Ne p Top.top
    ⊢ Exists fun c => ∀ (x : α), LE.le (HPow.hPow (↑(NNNorm.nnnorm (f x))) p.toRea …
  -/
  simp_rw [← ENNReal.coe_rpow_of_nonneg _ ENNReal.toReal_nonneg]
  /-
    case inr.inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    hp : Ne p Top.top
    ⊢ Exists fun c => ∀ (x : α), LE.le ↑(HPow.hPow (NNNorm.nnnorm (f x)) p.toReal) …
  -/
  norm_cast
  /-
    case inr.inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup F
    f : α → F
    inst✝¹ : Finite α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp₀ : Ne p 0
    hp : Ne p Top.top
    ⊢ Exists fun c => ∀ (x : α), LE.le (HPow.hPow (NNNorm.nnnorm (f x)) p.toReal) c
  -/
  exact Finite.exists_le _
  /-
    🎉 no goals
  -/


@[simp] lemma Memℒp.of_discrete [DiscreteMeasurableSpace α] [Finite α] [IsFiniteMeasure μ] :
    Memℒp f p μ :=
  let ⟨C, hC⟩ := Finite.exists_le (‖f ·‖₊); .of_bound .of_finite C <| .of_forall hC


@[simp] lemma eLpNorm_of_isEmpty [IsEmpty α] (f : α → E) (p : ℝ≥0∞) : eLpNorm f p μ = 0 := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : IsEmpty α
    f : α → E
    p : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) 0
  -/
  simp [Subsingleton.elim f 0]
  /-
    🎉 no goals
  -/


lemma eLpNormEssSup_piecewise {s : Set α} (f g : α → E) [DecidablePred (· ∈ s)]
    (hs : MeasurableSet s) :
    eLpNormEssSup (Set.piecewise s f g) μ
      = max (eLpNormEssSup f (μ.restrict s)) (eLpNormEssSup g (μ.restrict sᶜ)) := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    s : Set α
    f g : α → E
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.eLpNormEssSup (s.piecewise f g) μ) (Max.max (MeasureTheory …
  -/
  simp only [eLpNormEssSup, ← ENNReal.essSup_piecewise hs]
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    s : Set α
    f g : α → E
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ Eq (essSup (fun x => ENorm.enorm (s.piecewise f g x)) μ) (essSup (s.piecewis …
  -/
  congr with x
  /-
    case e_f.h
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    s : Set α
    f g : α → E
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    x : α
    ⊢ Eq (ENorm.enorm (s.piecewise f g x)) (s.piecewise (fun x => ENorm.enorm (f x …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ s <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_piecewise := eLpNormEssSup_piecewise


lemma eLpNorm_top_piecewise {s : Set α} (f g : α → E) [DecidablePred (· ∈ s)]
    (hs : MeasurableSet s) :
    eLpNorm (Set.piecewise s f g) ∞ μ
      = max (eLpNorm f ∞ (μ.restrict s)) (eLpNorm g ∞ (μ.restrict sᶜ)) :=
  eLpNormEssSup_piecewise f g hs


@[deprecated (since := "2024-07-27")]
alias snorm_top_piecewise := eLpNorm_top_piecewise


theorem eLpNormEssSup_map_measure (hg : AEStronglyMeasurable g (Measure.map f μ))
    (hf : AEMeasurable f μ) : eLpNormEssSup g (Measure.map f μ) = eLpNormEssSup (g ∘ f) μ :=
  essSup_map_measure hg.ennnorm hf


@[deprecated (since := "2024-07-27")]
alias snormEssSup_map_measure := eLpNormEssSup_map_measure


theorem eLpNorm_map_measure (hg : AEStronglyMeasurable g (Measure.map f μ))
    (hf : AEMeasurable f μ) : eLpNorm g p (Measure.map f μ) = eLpNorm (g ∘ f) p μ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → E
      hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
      hf : AEMeasurable f μ
      hp_zero : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
    -/
  · simp only [hp_zero, eLpNorm_exponent_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    hp_zero : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → E
      hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
      hf : AEMeasurable f μ
      hp_zero : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
    -/
  · simp_rw [hp_top, eLpNorm_exponent_top]
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → E
      hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
      hf : AEMeasurable f μ
      hp_zero : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNormEssSup g (MeasureTheory.Measure.map f μ)) (MeasureT …
    -/
    exact eLpNormEssSup_map_measure hg hf
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
  -/
  simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_zero hp_top]
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral (MeasureTheory.Measure.map f μ) fun x …
  -/
  rw [lintegral_map' (hg.ennnorm.pow_const p.toReal) hf]
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_map_measure := eLpNorm_map_measure


theorem memℒp_map_measure_iff (hg : AEStronglyMeasurable g (Measure.map f μ))
    (hf : AEMeasurable f μ) : Memℒp g p (Measure.map f μ) ↔ Memℒp (g ∘ f) p μ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → E
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Iff (MeasureTheory.Memℒp g p (MeasureTheory.Measure.map f μ)) (MeasureTheory …
  -/
  simp [Memℒp, eLpNorm_map_measure hg hf, hg.comp_aemeasurable hf, hg]
  /-
    🎉 no goals
  -/


theorem Memℒp.comp_of_map (hg : Memℒp g p (Measure.map f μ)) (hf : AEMeasurable f μ) :
    Memℒp (g ∘ f) p μ :=
  (memℒp_map_measure_iff hg.aestronglyMeasurable hf).1 hg


theorem eLpNorm_comp_measurePreserving {ν : MeasureTheory.Measure β} (hg : AEStronglyMeasurable g ν)
    (hf : MeasurePreserving f μ ν) : eLpNorm (g ∘ f) p μ = eLpNorm g p ν :=
  Eq.symm <| hf.map_eq ▸ eLpNorm_map_measure (hf.map_eq ▸ hg) hf.aemeasurable


@[deprecated (since := "2024-07-27")]
alias snorm_comp_measurePreserving := eLpNorm_comp_measurePreserving


theorem AEEqFun.eLpNorm_compMeasurePreserving {ν : MeasureTheory.Measure β} (g : β →ₘ[ν] E)
    (hf : MeasurePreserving f μ ν) :
    eLpNorm (g.compMeasurePreserving f hf) p μ = eLpNorm g p ν := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    ν : MeasureTheory.Measure β
    g : MeasureTheory.AEEqFun β E ν
    hf : MeasureTheory.MeasurePreserving f μ ν
    ⊢ Eq (MeasureTheory.eLpNorm (↑(g.compMeasurePreserving f hf)) p μ) (MeasureThe …
  -/
  rw [eLpNorm_congr_ae (g.coeFn_compMeasurePreserving _)]
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    ν : MeasureTheory.Measure β
    g : MeasureTheory.AEEqFun β E ν
    hf : MeasureTheory.MeasurePreserving f μ ν
    ⊢ Eq (MeasureTheory.eLpNorm (Function.comp (↑g) f) p μ) (MeasureTheory.eLpNorm …
  -/
  exact eLpNorm_comp_measurePreserving g.aestronglyMeasurable hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias AEEqFun.snorm_compMeasurePreserving := AEEqFun.eLpNorm_compMeasurePreserving


theorem Memℒp.comp_measurePreserving {ν : MeasureTheory.Measure β} (hg : Memℒp g p ν)
    (hf : MeasurePreserving f μ ν) : Memℒp (g ∘ f) p μ :=
  .comp_of_map (hf.map_eq.symm ▸ hg) hf.aemeasurable


theorem _root_.MeasurableEmbedding.eLpNormEssSup_map_measure {g : β → F}
    (hf : MeasurableEmbedding f) : eLpNormEssSup g (Measure.map f μ) = eLpNormEssSup (g ∘ f) μ :=
  hf.essSup_map_measure


@[deprecated (since := "2024-07-27")]
alias _root_.MeasurableEmbedding.snormEssSup_map_measure :=
  _root_.MeasurableEmbedding.eLpNormEssSup_map_measure


theorem _root_.MeasurableEmbedding.eLpNorm_map_measure {g : β → F} (hf : MeasurableEmbedding f) :
    eLpNorm g p (Measure.map f μ) = eLpNorm (g ∘ f) p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → F
    hf : MeasurableEmbedding f
    ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
    -/
  · simp only [hp_zero, eLpNorm_exponent_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → F
    hf : MeasurableEmbedding f
    hp_zero : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
  -/
  by_cases hp : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Not (Eq p 0)
      hp : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
    -/
  · simp_rw [hp, eLpNorm_exponent_top]
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Not (Eq p 0)
      hp : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNormEssSup g (MeasureTheory.Measure.map f μ)) (MeasureT …
    -/
    exact hf.essSup_map_measure
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Not (Eq p 0)
      hp : Not (Eq p Top.top)
      ⊢ Eq (MeasureTheory.eLpNorm g p (MeasureTheory.Measure.map f μ)) (MeasureTheor …
    -/
  · simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_zero hp]
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Not (Eq p 0)
      hp : Not (Eq p Top.top)
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral (MeasureTheory.Measure.map f μ) fun x …
    -/
    rw [hf.lintegral_map]
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      β : Type u_6
      mβ : MeasurableSpace β
      f : α → β
      g : β → F
      hf : MeasurableEmbedding f
      hp_zero : Not (Eq p 0)
      hp : Not (Eq p Top.top)
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias _root_.MeasurableEmbedding.snorm_map_measure := _root_.MeasurableEmbedding.eLpNorm_map_measure


theorem _root_.MeasurableEmbedding.memℒp_map_measure_iff {g : β → F} (hf : MeasurableEmbedding f) :
    Memℒp g p (Measure.map f μ) ↔ Memℒp (g ∘ f) p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    β : Type u_6
    mβ : MeasurableSpace β
    f : α → β
    g : β → F
    hf : MeasurableEmbedding f
    ⊢ Iff (MeasureTheory.Memℒp g p (MeasureTheory.Measure.map f μ)) (MeasureTheory …
  -/
  simp_rw [Memℒp, hf.aestronglyMeasurable_map_iff, hf.eLpNorm_map_measure]
  /-
    🎉 no goals
  -/


theorem _root_.MeasurableEquiv.memℒp_map_measure_iff (f : α ≃ᵐ β) {g : β → F} :
    Memℒp g p (Measure.map f μ) ↔ Memℒp (g ∘ f) p μ :=
  f.measurableEmbedding.memℒp_map_measure_iff


theorem eLpNorm'_le_nnreal_smul_eLpNorm'_of_ae_le_mul {f : α → F} {g : α → G} {c : ℝ≥0}
    (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) {p : ℝ} (hp : 0 < p) :
    eLpNorm' f p μ ≤ c • eLpNorm' g p μ := by
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : Real
    hp : LT.lt 0 p
    ⊢ LE.le (MeasureTheory.eLpNorm' f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm'  …
  -/
  simp_rw [eLpNorm'_eq_lintegral_nnnorm]
  rw [← ENNReal.rpow_le_rpow_iff hp, ENNReal.smul_def, smul_eq_mul,
    ENNReal.mul_rpow_of_nonneg _ _ hp.le]
  simp_rw [← ENNReal.rpow_mul, one_div, inv_mul_cancel₀ hp.ne.symm, ENNReal.rpow_one,
    ← ENNReal.coe_rpow_of_nonneg _ hp.le, ← lintegral_const_mul' _ _ ENNReal.coe_ne_top,
    ← ENNReal.coe_mul]
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : Real
    hp : LT.lt 0 p
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(HPow.hPow (NNNorm.nnnorm (f a))  …
  -/
  apply lintegral_mono_ae
  /-
    case h
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : Real
    hp : LT.lt 0 p
    ⊢ Filter.Eventually (fun a => LE.le ↑(HPow.hPow (NNNorm.nnnorm (f a)) p) ↑(HMu …
  -/
  simp_rw [ENNReal.coe_le_coe, ← NNReal.mul_rpow, NNReal.rpow_le_rpow_iff hp]
  /-
    case h
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : Real
    hp : LT.lt 0 p
    ⊢ Filter.Eventually (fun a => LE.le (NNNorm.nnnorm (f a)) (HMul.hMul c (NNNorm …
  -/
  exact h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_le_nnreal_smul_snorm'_of_ae_le_mul := eLpNorm'_le_nnreal_smul_eLpNorm'_of_ae_le_mul


theorem eLpNormEssSup_le_nnreal_smul_eLpNormEssSup_of_ae_le_mul {f : α → F} {g : α → G} {c : ℝ≥0}
    (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) : eLpNormEssSup f μ ≤ c • eLpNormEssSup g μ :=
  calc
    essSup (fun x => (‖f x‖₊ : ℝ≥0∞)) μ ≤ essSup (fun x => (↑(c * ‖g x‖₊) : ℝ≥0∞)) μ :=
      essSup_mono_ae <| h.mono fun _ hx => ENNReal.coe_le_coe.mpr hx
                                                      /-
                                                        α : Type u_1
                                                        F : Type u_4
                                                        G : Type u_5
                                                        m0 : MeasurableSpace α
                                                        μ : MeasureTheory.Measure α
                                                        inst✝¹ : NormedAddCommGroup F
                                                        inst✝ : NormedAddCommGroup G
                                                        f : α → F
                                                        g : α → G
                                                        c : NNReal
                                                        h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
                                                        ⊢ Eq (essSup (fun x => ↑(HMul.hMul c (NNNorm.nnnorm (g x)))) μ) (essSup (fun x …
                                                      -/
    _ = essSup (fun x => (c * ‖g x‖₊ : ℝ≥0∞)) μ := by simp_rw [ENNReal.coe_mul]
                                                      /-
                                                        🎉 no goals
                                                      -/
    _ = c • essSup (fun x => (‖g x‖₊ : ℝ≥0∞)) μ := ENNReal.essSup_const_mul


@[deprecated (since := "2024-07-27")]
alias snormEssSup_le_nnreal_smul_snormEssSup_of_ae_le_mul :=
  eLpNormEssSup_le_nnreal_smul_eLpNormEssSup_of_ae_le_mul


theorem eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul {f : α → F} {g : α → G} {c : ℝ≥0}
    (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) (p : ℝ≥0∞) : eLpNorm f p μ ≤ c • eLpNorm g p μ := by
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : ENNReal
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm g  …
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      c : NNReal
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
      p : ENNReal
      h0 : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm g  …
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : ENNReal
    h0 : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm g  …
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      c : NNReal
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
      p : ENNReal
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm g  …
    -/
  · rw [h_top]
    /-
      case pos
      α : Type u_1
      F : Type u_4
      G : Type u_5
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      f : α → F
      g : α → G
      c : NNReal
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
      p : ENNReal
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f Top.top μ) (HSMul.hSMul c (MeasureTheory.eLpN …
    -/
    exact eLpNormEssSup_le_nnreal_smul_eLpNormEssSup_of_ae_le_mul h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : ENNReal
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm f p μ) (HSMul.hSMul c (MeasureTheory.eLpNorm g  …
  -/
  simp_rw [eLpNorm_eq_eLpNorm' h0 h_top]
  /-
    case neg
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : NNReal
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (f x)) (HMul.hMul c (NNNo …
    p : ENNReal
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm' f p.toReal μ) (HSMul.hSMul c (MeasureTheory.eL …
  -/
  exact eLpNorm'_le_nnreal_smul_eLpNorm'_of_ae_le_mul h (ENNReal.toReal_pos h0 h_top)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_le_nnreal_smul_snorm_of_ae_le_mul := eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul

-- TODO: add the whole family of lemmas?

private theorem le_mul_iff_eq_zero_of_nonneg_of_neg_of_nonneg {α} [LinearOrderedSemiring α]
    {a b c : α} (ha : 0 ≤ a) (hb : b < 0) (hc : 0 ≤ c) : a ≤ b * c ↔ a = 0 ∧ c = 0 := by
  /-
    α : Type u_6
    inst✝ : LinearOrderedSemiring α
    a b c : α
    ha : LE.le 0 a
    hb : LT.lt b 0
    hc : LE.le 0 c
    ⊢ Iff (LE.le a (HMul.hMul b c)) (And (Eq a 0) (Eq c 0))
  -/
  constructor
    /-
      case mp
      α : Type u_6
      inst✝ : LinearOrderedSemiring α
      a b c : α
      ha : LE.le 0 a
      hb : LT.lt b 0
      hc : LE.le 0 c
      ⊢ LE.le a (HMul.hMul b c) → And (Eq a 0) (Eq c 0)
    -/
  · intro h
    exact
      ⟨(h.trans (mul_nonpos_of_nonpos_of_nonneg hb.le hc)).antisymm ha,
        (nonpos_of_mul_nonneg_right (ha.trans h) hb).antisymm hc⟩
    /-
      case mpr
      α : Type u_6
      inst✝ : LinearOrderedSemiring α
      a b c : α
      ha : LE.le 0 a
      hb : LT.lt b 0
      hc : LE.le 0 c
      ⊢ And (Eq a 0) (Eq c 0) → LE.le a (HMul.hMul b c)
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case mpr.intro
      α : Type u_6
      inst✝ : LinearOrderedSemiring α
      b : α
      hb : LT.lt b 0
      ha hc : LE.le 0 0
      ⊢ LE.le 0 (HMul.hMul b 0)
    -/
    rw [mul_zero]
    /-
      🎉 no goals
    -/


/-- When `c` is negative, `‖f x‖ ≤ c * ‖g x‖` is nonsense and forces both `f` and `g` to have an
`eLpNorm` of `0`. -/
theorem eLpNorm_eq_zero_and_zero_of_ae_le_mul_neg {f : α → F} {g : α → G} {c : ℝ}
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ c * ‖g x‖) (hc : c < 0) (p : ℝ≥0∞) :
    eLpNorm f p μ = 0 ∧ eLpNorm g p μ = 0 := by
  simp_rw [le_mul_iff_eq_zero_of_nonneg_of_neg_of_nonneg (norm_nonneg _) hc (norm_nonneg _),
    norm_eq_zero, eventually_and] at h
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : Real
    hc : LT.lt c 0
    p : ENNReal
    h : And (Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)) (Filter …
    ⊢ And (Eq (MeasureTheory.eLpNorm f p μ) 0) (Eq (MeasureTheory.eLpNorm g p μ) 0)
  -/
  change f =ᵐ[μ] 0 ∧ g =ᵐ[μ] 0 at h
  /-
    α : Type u_1
    F : Type u_4
    G : Type u_5
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedAddCommGroup G
    f : α → F
    g : α → G
    c : Real
    hc : LT.lt c 0
    p : ENNReal
    h : And ((MeasureTheory.ae μ).EventuallyEq f 0) ((MeasureTheory.ae μ).Eventual …
    ⊢ And (Eq (MeasureTheory.eLpNorm f p μ) 0) (Eq (MeasureTheory.eLpNorm g p μ) 0)
  -/
  simp [eLpNorm_congr_ae h.1, eLpNorm_congr_ae h.2]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_eq_zero_and_zero_of_ae_le_mul_neg := eLpNorm_eq_zero_and_zero_of_ae_le_mul_neg


theorem eLpNorm_le_mul_eLpNorm_of_ae_le_mul {f : α → F} {g : α → G} {c : ℝ}
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ c * ‖g x‖) (p : ℝ≥0∞) :
    eLpNorm f p μ ≤ ENNReal.ofReal c * eLpNorm g p μ :=
  eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul
    (h.mono fun _x hx => hx.trans <| mul_le_mul_of_nonneg_right c.le_coe_toNNReal (norm_nonneg _)) _


@[deprecated (since := "2024-07-27")]
alias snorm_le_mul_snorm_of_ae_le_mul := eLpNorm_le_mul_eLpNorm_of_ae_le_mul


theorem Memℒp.of_nnnorm_le_mul {f : α → E} {g : α → F} {c : ℝ≥0} (hg : Memℒp g p μ)
    (hf : AEStronglyMeasurable f μ) (hfg : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) : Memℒp f p μ :=
  ⟨hf,
    (eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul hfg p).trans_lt <|
      ENNReal.mul_lt_top ENNReal.coe_lt_top hg.eLpNorm_lt_top⟩


theorem Memℒp.of_le_mul {f : α → E} {g : α → F} {c : ℝ} (hg : Memℒp g p μ)
    (hf : AEStronglyMeasurable f μ) (hfg : ∀ᵐ x ∂μ, ‖f x‖ ≤ c * ‖g x‖) : Memℒp f p μ :=
  ⟨hf,
    (eLpNorm_le_mul_eLpNorm_of_ae_le_mul hfg p).trans_lt <|
      ENNReal.mul_lt_top ENNReal.ofReal_lt_top hg.eLpNorm_lt_top⟩


theorem eLpNorm'_const_smul_le (hq : 0 < q) : eLpNorm' (c • f) q μ ≤ ‖c‖₊ • eLpNorm' f q μ :=
  eLpNorm'_le_nnreal_smul_eLpNorm'_of_ae_le_mul (Eventually.of_forall fun _ => nnnorm_smul_le ..) hq


@[deprecated (since := "2024-07-27")]
alias snorm'_const_smul_le := eLpNorm'_const_smul_le


theorem eLpNormEssSup_const_smul_le : eLpNormEssSup (c • f) μ ≤ ‖c‖₊ • eLpNormEssSup f μ :=
  eLpNormEssSup_le_nnreal_smul_eLpNormEssSup_of_ae_le_mul
                                      /-
                                        α : Type u_1
                                        F : Type u_4
                                        m0 : MeasurableSpace α
                                        μ : MeasureTheory.Measure α
                                        inst✝³ : NormedAddCommGroup F
                                        𝕜 : Type u_6
                                        inst✝² : NormedRing 𝕜
                                        inst✝¹ : MulActionWithZero 𝕜 F
                                        inst✝ : BoundedSMul 𝕜 F
                                        c : 𝕜
                                        f : α → F
                                        x✝ : α
                                        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f x✝)) (HMul.hMul (NNNorm.nnnorm c) (NNN …
                                      -/
    (Eventually.of_forall fun _ => by simp [nnnorm_smul_le])
                                      /-
                                        🎉 no goals
                                      -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_const_smul_le := eLpNormEssSup_const_smul_le


theorem eLpNorm_const_smul_le : eLpNorm (c • f) p μ ≤ ‖c‖₊ • eLpNorm f p μ :=
  eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul
                                      /-
                                        α : Type u_1
                                        F : Type u_4
                                        m0 : MeasurableSpace α
                                        p : ENNReal
                                        μ : MeasureTheory.Measure α
                                        inst✝³ : NormedAddCommGroup F
                                        𝕜 : Type u_6
                                        inst✝² : NormedRing 𝕜
                                        inst✝¹ : MulActionWithZero 𝕜 F
                                        inst✝ : BoundedSMul 𝕜 F
                                        c : 𝕜
                                        f : α → F
                                        x✝ : α
                                        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f x✝)) (HMul.hMul (NNNorm.nnnorm c) (NNN …
                                      -/
    (Eventually.of_forall fun _ => by simp [nnnorm_smul_le]) _
                                      /-
                                        🎉 no goals
                                      -/


@[deprecated (since := "2024-07-27")]
alias snorm_const_smul_le := eLpNorm_const_smul_le


theorem Memℒp.const_smul (hf : Memℒp f p μ) (c : 𝕜) : Memℒp (c • f) p μ :=
  ⟨AEStronglyMeasurable.const_smul hf.1 c,
    eLpNorm_const_smul_le.trans_lt (ENNReal.mul_lt_top ENNReal.coe_lt_top hf.2)⟩


theorem Memℒp.const_mul {f : α → 𝕜} (hf : Memℒp f p μ) (c : 𝕜) : Memℒp (fun x => c * f x) p μ :=
  hf.const_smul c


theorem eLpNorm'_const_smul {f : α → F} (c : 𝕜) (hq_pos : 0 < q) :
    eLpNorm' (c • f) q μ = ‖c‖₊ • eLpNorm' f q μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    f : α → F
    c : 𝕜
    hq_pos : LT.lt 0 q
    ⊢ Eq (MeasureTheory.eLpNorm' (HSMul.hSMul c f) q μ) (HSMul.hSMul (NNNorm.nnnor …
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      q : Real
      μ : MeasureTheory.Measure α
      inst✝³ : NormedAddCommGroup F
      𝕜 : Type u_6
      inst✝² : NormedDivisionRing 𝕜
      inst✝¹ : Module 𝕜 F
      inst✝ : BoundedSMul 𝕜 F
      f : α → F
      hq_pos : LT.lt 0 q
      ⊢ Eq (MeasureTheory.eLpNorm' (HSMul.hSMul 0 f) q μ) (HSMul.hSMul (NNNorm.nnnor …
    -/
  · simp [eLpNorm'_eq_lintegral_nnnorm, hq_pos]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    f : α → F
    c : 𝕜
    hq_pos : LT.lt 0 q
    hc : Ne c 0
    ⊢ Eq (MeasureTheory.eLpNorm' (HSMul.hSMul c f) q μ) (HSMul.hSMul (NNNorm.nnnor …
  -/
  refine le_antisymm (eLpNorm'_const_smul_le hq_pos) ?_
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    f : α → F
    c : 𝕜
    hq_pos : LT.lt 0 q
    hc : Ne c 0
    ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm c) (MeasureTheory.eLpNorm' f q μ)) (Measur …
  -/
  simpa [hc, le_inv_smul_iff_of_pos] using eLpNorm'_const_smul_le (c := c⁻¹) (f := c • f) hq_pos
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_const_smul := eLpNorm'_const_smul


theorem eLpNormEssSup_const_smul (c : 𝕜) (f : α → F) :
    eLpNormEssSup (c • f) μ = (‖c‖₊ : ℝ≥0∞) * eLpNormEssSup f μ := by
  simp_rw [eLpNormEssSup_eq_essSup_nnnorm, Pi.smul_apply, nnnorm_smul, ENNReal.coe_mul,
    ENNReal.essSup_const_mul]


@[deprecated (since := "2024-07-27")]
alias snormEssSup_const_smul := eLpNormEssSup_const_smul


theorem eLpNorm_const_smul (c : 𝕜) (f : α → F) (p : ℝ≥0∞) (μ : Measure α):
    eLpNorm (c • f) p μ = (‖c‖₊ : ℝ≥0∞) * eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    c : 𝕜
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.eLpNorm (HSMul.hSMul c f) p μ) (HMul.hMul (↑(NNNorm.nnnorm …
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup F
      𝕜 : Type u_6
      inst✝² : NormedDivisionRing 𝕜
      inst✝¹ : Module 𝕜 F
      inst✝ : BoundedSMul 𝕜 F
      f : α → F
      p : ENNReal
      μ : MeasureTheory.Measure α
      ⊢ Eq (MeasureTheory.eLpNorm (HSMul.hSMul 0 f) p μ) (HMul.hMul (↑(NNNorm.nnnorm …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    c : 𝕜
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    hc : Ne c 0
    ⊢ Eq (MeasureTheory.eLpNorm (HSMul.hSMul c f) p μ) (HMul.hMul (↑(NNNorm.nnnorm …
  -/
  refine le_antisymm eLpNorm_const_smul_le ?_
  /-
    case inr
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_6
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : Module 𝕜 F
    inst✝ : BoundedSMul 𝕜 F
    c : 𝕜
    f : α → F
    p : ENNReal
    μ : MeasureTheory.Measure α
    hc : Ne c 0
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (MeasureTheory.eLpNorm f p μ)) (Measur …
  -/
  simpa [hc, le_inv_smul_iff_of_pos] using eLpNorm_const_smul_le (c := c⁻¹) (f := c • f)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_const_smul := eLpNorm_const_smul


lemma eLpNorm_nsmul [NormedSpace ℝ F] (n : ℕ) (f : α → F) :
    eLpNorm (n • f) p μ = n * eLpNorm f p μ := by
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    n : Nat
    f : α → F
    ⊢ Eq (MeasureTheory.eLpNorm (HSMul.hSMul n f) p μ) (HMul.hMul (↑n) (MeasureThe …
  -/
  simpa [Nat.cast_smul_eq_nsmul] using eLpNorm_const_smul (n : ℝ) f ..
  /-
    🎉 no goals
  -/


theorem le_eLpNorm_of_bddBelow (hp : p ≠ 0) (hp' : p ≠ ∞) {f : α → F} (C : ℝ≥0) {s : Set α}
    (hs : MeasurableSet s) (hf : ∀ᵐ x ∂μ, x ∈ s → C ≤ ‖f x‖₊) :
    C • μ s ^ (1 / p.toReal) ≤ eLpNorm f p μ := by
  rw [ENNReal.smul_def, smul_eq_mul, eLpNorm_eq_lintegral_rpow_nnnorm hp hp',
    one_div, ENNReal.le_rpow_inv_iff (ENNReal.toReal_pos hp hp'),
    ENNReal.mul_rpow_of_nonneg _ _ ENNReal.toReal_nonneg, ← ENNReal.rpow_mul,
    inv_mul_cancel₀ (ENNReal.toReal_pos hp hp').ne.symm, ENNReal.rpow_one, ← setLIntegral_const,
    ← lintegral_indicator hs]
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p 0
    hp' : Ne p Top.top
    f : α → F
    C : NNReal
    s : Set α
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => s.indicator (fun x => HPow.hPow (↑ …
  -/
  refine lintegral_mono_ae ?_
  /-
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p 0
    hp' : Ne p Top.top
    f : α → F
    C : NNReal
    s : Set α
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
    ⊢ Filter.Eventually (fun a => LE.le (s.indicator (fun x => HPow.hPow (↑C) p.to …
  -/
  filter_upwards [hf] with x hx
  /-
    case h
    α : Type u_1
    F : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    hp : Ne p 0
    hp' : Ne p Top.top
    f : α → F
    C : NNReal
    s : Set α
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
    x : α
    hx : Membership.mem s x → LE.le C (NNNorm.nnnorm (f x))
    ⊢ LE.le (s.indicator (fun x => HPow.hPow (↑C) p.toReal) x) (HPow.hPow (↑(NNNor …
  -/
  by_cases hxs : x ∈ s
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hp : Ne p 0
      hp' : Ne p Top.top
      f : α → F
      C : NNReal
      s : Set α
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
      x : α
      hx : Membership.mem s x → LE.le C (NNNorm.nnnorm (f x))
      hxs : Membership.mem s x
      ⊢ LE.le (s.indicator (fun x => HPow.hPow (↑C) p.toReal) x) (HPow.hPow (↑(NNNor …
    -/
  · simp only [Set.indicator_of_mem hxs] at hx ⊢
    /-
      case pos
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hp : Ne p 0
      hp' : Ne p Top.top
      f : α → F
      C : NNReal
      s : Set α
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
      x : α
      hx : Membership.mem s x → LE.le C (NNNorm.nnnorm (f x))
      hxs : Membership.mem s x
      ⊢ LE.le (HPow.hPow (↑C) p.toReal) (HPow.hPow (↑(NNNorm.nnnorm (f x))) p.toReal)
    -/
    gcongr
    /-
      case pos.h₁.a
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hp : Ne p 0
      hp' : Ne p Top.top
      f : α → F
      C : NNReal
      s : Set α
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
      x : α
      hx : Membership.mem s x → LE.le C (NNNorm.nnnorm (f x))
      hxs : Membership.mem s x
      ⊢ LE.le C (NNNorm.nnnorm (f x))
    -/
    exact hx hxs
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      hp : Ne p 0
      hp' : Ne p Top.top
      f : α → F
      C : NNReal
      s : Set α
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → LE.le C (NNNorm.nnnorm ( …
      x : α
      hx : Membership.mem s x → LE.le C (NNNorm.nnnorm (f x))
      hxs : Not (Membership.mem s x)
      ⊢ LE.le (s.indicator (fun x => HPow.hPow (↑C) p.toReal) x) (HPow.hPow (↑(NNNor …
    -/
  · simp [Set.indicator_of_not_mem hxs]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias le_snorm_of_bddBelow := le_eLpNorm_of_bddBelow


@[deprecated (since := "2024-06-26")]
alias snorm_indicator_ge_of_bdd_below := le_snorm_of_bddBelow


@[simp] lemma eLpNorm_conj (f : α → 𝕜) (p : ℝ≥0∞) (μ : Measure α) :
                                               /-
                                                 α : Type u_1
                                                 m0 : MeasurableSpace α
                                                 𝕜 : Type u_6
                                                 inst✝ : RCLike 𝕜
                                                 f : α → 𝕜
                                                 p : ENNReal
                                                 μ : MeasureTheory.Measure α
                                                 ⊢ Eq (MeasureTheory.eLpNorm ((starRingEnd (α → 𝕜)) f) p μ) (MeasureTheory.eLpN …
                                               -/
    eLpNorm (conj f) p μ = eLpNorm f p μ := by simp [← eLpNorm_norm]
                                               /-
                                                 🎉 no goals
                                               -/


theorem Memℒp.re (hf : Memℒp f p μ) : Memℒp (fun x => RCLike.re (f x)) p μ := by
  have : ∀ x, ‖RCLike.re (f x)‖ ≤ 1 * ‖f x‖ := by
    intro x
    rw [one_mul]
    exact RCLike.norm_re_le_norm (f x)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    this : ∀ (x : α), LE.le (Norm.norm (RCLike.re (f x))) (HMul.hMul 1 (Norm.norm  …
    ⊢ MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ
  -/
  refine hf.of_le_mul ?_ (Eventually.of_forall this)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    this : ∀ (x : α), LE.le (Norm.norm (RCLike.re (f x))) (HMul.hMul 1 (Norm.norm  …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => RCLike.re (f x)) μ
  -/
  exact RCLike.continuous_re.comp_aestronglyMeasurable hf.1
  /-
    🎉 no goals
  -/


theorem Memℒp.im (hf : Memℒp f p μ) : Memℒp (fun x => RCLike.im (f x)) p μ := by
  have : ∀ x, ‖RCLike.im (f x)‖ ≤ 1 * ‖f x‖ := by
    intro x
    rw [one_mul]
    exact RCLike.norm_im_le_norm (f x)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    this : ∀ (x : α), LE.le (Norm.norm (RCLike.im (f x))) (HMul.hMul 1 (Norm.norm  …
    ⊢ MeasureTheory.Memℒp (fun x => RCLike.im (f x)) p μ
  -/
  refine hf.of_le_mul ?_ (Eventually.of_forall this)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f p μ
    this : ∀ (x : α), LE.le (Norm.norm (RCLike.im (f x))) (HMul.hMul 1 (Norm.norm  …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => RCLike.im (f x)) μ
  -/
  exact RCLike.continuous_im.comp_aestronglyMeasurable hf.1
  /-
    🎉 no goals
  -/


theorem ae_bdd_liminf_atTop_rpow_of_eLpNorm_bdd {p : ℝ≥0∞} {f : ℕ → α → E}
    (hfmeas : ∀ n, Measurable (f n)) (hbdd : ∀ n, eLpNorm (f n) p μ ≤ R) :
    ∀ᵐ x ∂μ, liminf (fun n => ((‖f n x‖₊ : ℝ≥0∞) ^ p.toReal : ℝ≥0∞)) atTop < ∞ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNo …
  -/
  by_cases hp0 : p.toReal = 0
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      p : ENNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      hp0 : Eq p.toReal 0
      ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNo …
    -/
  · simp only [hp0, ENNReal.rpow_zero]
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      p : ENNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      hp0 : Eq p.toReal 0
      ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => 1) Filter.atTop)  …
    -/
    filter_upwards with _
    /-
      case pos.h
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      p : ENNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      hp0 : Eq p.toReal 0
      a✝ : α
      ⊢ LT.lt (Filter.liminf (fun n => 1) Filter.atTop) Top.top
    -/
    rw [liminf_const (1 : ℝ≥0∞)]
    /-
      case pos.h
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      p : ENNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      hp0 : Eq p.toReal 0
      a✝ : α
      ⊢ LT.lt 1 Top.top
    -/
    exact ENNReal.one_lt_top
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp0 : Not (Eq p.toReal 0)
    ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNo …
  -/
  have hp : p ≠ 0 := fun h => by simp [h] at hp0
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp0 : Not (Eq p.toReal 0)
    hp : Ne p 0
    ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNo …
  -/
  have hp' : p ≠ ∞ := fun h => by simp [h] at hp0
  refine
    ae_lt_top (.liminf fun n => (hfmeas n).nnnorm.coe_nnreal_ennreal.pow_const p.toReal)
      (lt_of_le_of_lt
          (lintegral_liminf_le fun n => (hfmeas n).nnnorm.coe_nnreal_ennreal.pow_const p.toReal)
          (lt_of_le_of_lt ?_
            (ENNReal.rpow_lt_top_of_nonneg ENNReal.toReal_nonneg ENNReal.coe_ne_top :
              (R : ℝ≥0∞) ^ p.toReal < ∞))).ne
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp0 : Not (Eq p.toReal 0)
    hp : Ne p 0
    hp' : Ne p Top.top
    ⊢ LE.le (Filter.liminf (fun n => MeasureTheory.lintegral μ fun a => HPow.hPow  …
  -/
  simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm hp hp', one_div] at hbdd
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hp0 : Not (Eq p.toReal 0)
    hp : Ne p 0
    hp' : Ne p Top.top
    hbdd : ∀ (n : Nat), LE.le (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow. …
    ⊢ LE.le (Filter.liminf (fun n => MeasureTheory.lintegral μ fun a => HPow.hPow  …
  -/
  simp_rw [liminf_eq, eventually_atTop]
  exact
    sSup_le fun b ⟨a, ha⟩ =>
      (ha a le_rfl).trans ((ENNReal.rpow_inv_le_iff (ENNReal.toReal_pos hp hp')).1 (hbdd _))


@[deprecated (since := "2024-07-27")]
alias ae_bdd_liminf_atTop_rpow_of_snorm_bdd := ae_bdd_liminf_atTop_rpow_of_eLpNorm_bdd


theorem ae_bdd_liminf_atTop_of_eLpNorm_bdd {p : ℝ≥0∞} (hp : p ≠ 0) {f : ℕ → α → E}
    (hfmeas : ∀ n, Measurable (f n)) (hbdd : ∀ n, eLpNorm (f n) p μ ≤ R) :
    ∀ᵐ x ∂μ, liminf (fun n => (‖f n x‖₊ : ℝ≥0∞)) atTop < ∞ := by
  /-
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    hp : Ne p 0
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm ( …
  -/
  by_cases hp' : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      p : ENNReal
      hp : Ne p 0
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      hp' : Eq p Top.top
      ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm ( …
    -/
  · subst hp'
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hp : Ne Top.top 0
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) Top.top μ) ↑R
      ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm ( …
    -/
    simp_rw [eLpNorm_exponent_top] at hbdd
    have : ∀ n, ∀ᵐ x ∂μ, (‖f n x‖₊ : ℝ≥0∞) < R + 1 := fun n =>
      ae_lt_of_essSup_lt
        (lt_of_le_of_lt (hbdd n) <| ENNReal.lt_add_right ENNReal.coe_ne_top one_ne_zero)
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasurableSpace E
      inst✝ : OpensMeasurableSpace E
      R : NNReal
      f : Nat → α → E
      hfmeas : ∀ (n : Nat), Measurable (f n)
      hp : Ne Top.top 0
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNormEssSup (f n) μ) ↑R
      this : ∀ (n : Nat), Filter.Eventually (fun x => LT.lt (↑(NNNorm.nnnorm (f n x) …
      ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm ( …
    -/
    rw [← ae_all_iff] at this
    filter_upwards [this] with x hx using lt_of_le_of_lt
        (liminf_le_of_frequently_le' <| Frequently.of_forall fun n => (hx n).le)
        (ENNReal.add_lt_top.2 ⟨ENNReal.coe_lt_top, ENNReal.one_lt_top⟩)
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    hp : Ne p 0
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp' : Not (Eq p Top.top)
    ⊢ Filter.Eventually (fun x => LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm ( …
  -/
  filter_upwards [ae_bdd_liminf_atTop_rpow_of_eLpNorm_bdd hfmeas hbdd] with x hx
  /-
    case h
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    hp : Ne p 0
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp' : Not (Eq p Top.top)
    x : α
    hx : LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNorm.nnnorm (f n x))) p.toR …
    ⊢ LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n x))) Filter.atTop) Top.top
  -/
  have hppos : 0 < p.toReal := ENNReal.toReal_pos hp hp'
  have :
    liminf (fun n => (‖f n x‖₊ : ℝ≥0∞) ^ p.toReal) atTop =
      liminf (fun n => (‖f n x‖₊ : ℝ≥0∞)) atTop ^ p.toReal := by
    change
      liminf (fun n => ENNReal.orderIsoRpow p.toReal hppos (‖f n x‖₊ : ℝ≥0∞)) atTop =
        ENNReal.orderIsoRpow p.toReal hppos (liminf (fun n => (‖f n x‖₊ : ℝ≥0∞)) atTop)
    refine (OrderIso.liminf_apply (ENNReal.orderIsoRpow p.toReal _) ?_ ?_ ?_ ?_).symm <;>
      isBoundedDefault
  /-
    case h
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    hp : Ne p 0
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp' : Not (Eq p Top.top)
    x : α
    hx : LT.lt (Filter.liminf (fun n => HPow.hPow (↑(NNNorm.nnnorm (f n x))) p.toR …
    hppos : LT.lt 0 p.toReal
    this : Eq (Filter.liminf (fun n => HPow.hPow (↑(NNNorm.nnnorm (f n x))) p.toRe …
    ⊢ LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n x))) Filter.atTop) Top.top
  -/
  rw [this] at hx
  rw [← ENNReal.rpow_one (liminf (fun n => ‖f n x‖₊) atTop), ← mul_inv_cancel₀ hppos.ne.symm,
    ENNReal.rpow_mul]
  /-
    case h
    α : Type u_1
    E : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    R : NNReal
    p : ENNReal
    hp : Ne p 0
    f : Nat → α → E
    hfmeas : ∀ (n : Nat), Measurable (f n)
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    hp' : Not (Eq p Top.top)
    x : α
    hx : LT.lt (HPow.hPow (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n x))) Filte …
    hppos : LT.lt 0 p.toReal
    this : Eq (Filter.liminf (fun n => HPow.hPow (↑(NNNorm.nnnorm (f n x))) p.toRe …
    ⊢ LT.lt (HPow.hPow (HPow.hPow (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n x) …
  -/
  exact ENNReal.rpow_lt_top_of_nonneg (inv_nonneg.2 hppos.le) hx.ne
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias ae_bdd_liminf_atTop_of_snorm_bdd := ae_bdd_liminf_atTop_of_eLpNorm_bdd


/-- A continuous function with compact support belongs to `L^∞`.
See `Continuous.memℒp_of_hasCompactSupport` for a version for `L^p`. -/
theorem _root_.Continuous.memℒp_top_of_hasCompactSupport
    {X : Type*} [TopologicalSpace X] [MeasurableSpace X] [OpensMeasurableSpace X]
    {f : X → E} (hf : Continuous f) (h'f : HasCompactSupport f) (μ : Measure X) : Memℒp f ⊤ μ := by
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    X : Type u_6
    inst✝² : TopologicalSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    f : X → E
    hf : Continuous f
    h'f : HasCompactSupport f
    μ : MeasureTheory.Measure X
    ⊢ MeasureTheory.Memℒp f Top.top μ
  -/
  borelize E
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    X : Type u_6
    inst✝² : TopologicalSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    f : X → E
    hf : Continuous f
    h'f : HasCompactSupport f
    μ : MeasureTheory.Measure X
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ MeasureTheory.Memℒp f Top.top μ
  -/
  rcases hf.bounded_above_of_compact_support h'f with ⟨C, hC⟩
  /-
    case intro
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    X : Type u_6
    inst✝² : TopologicalSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    f : X → E
    hf : Continuous f
    h'f : HasCompactSupport f
    μ : MeasureTheory.Measure X
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    C : Real
    hC : ∀ (x : X), LE.le (Norm.norm (f x)) C
    ⊢ MeasureTheory.Memℒp f Top.top μ
  -/
  apply memℒp_top_of_bound ?_ C (Filter.Eventually.of_forall hC)
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    X : Type u_6
    inst✝² : TopologicalSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    f : X → E
    hf : Continuous f
    h'f : HasCompactSupport f
    μ : MeasureTheory.Measure X
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    C : Real
    hC : ∀ (x : X), LE.le (Norm.norm (f x)) C
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  exact (hf.stronglyMeasurable_of_hasCompactSupport h'f).aestronglyMeasurable
  /-
    🎉 no goals
  -/


/-- A single function that is `Memℒp f p μ` is tight with respect to `μ`. -/
theorem Memℒp.exists_eLpNorm_indicator_compl_lt {β : Type*} [NormedAddCommGroup β] (hp_top : p ≠ ∞)
    {f : α → β} (hf : Memℒp f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ s : Set α, MeasurableSet s ∧ μ s < ∞ ∧ eLpNorm (sᶜ.indicator f) p μ < ε := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedAddCommGroup β
    hp_top : Ne p Top.top
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
  -/
  rcases eq_or_ne p 0 with rfl | hp₀
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_6
      inst✝ : NormedAddCommGroup β
      f : α → β
      ε : ENNReal
      hε : Ne ε 0
      hp_top : Ne 0 Top.top
      hf : MeasureTheory.Memℒp f 0 μ
      ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
    -/
  · use ∅; simp [pos_iff_ne_zero.2 hε] -- first take care of `p = 0`
           /-
             🎉 no goals
           -/
  · obtain ⟨s, hsm, hs, hε⟩ :
        ∃ s, MeasurableSet s ∧ μ s < ∞ ∧ ∫⁻ a in sᶜ, (‖f a‖₊) ^ p.toReal ∂μ < ε ^ p.toReal := by
      apply exists_setLintegral_compl_lt
      · exact ((eLpNorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top hp₀ hp_top).1 hf.2).ne
      · simp [*]
    /-
      case inr.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      β : Type u_6
      inst✝ : NormedAddCommGroup β
      hp_top : Ne p Top.top
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε✝ : Ne ε 0
      hp₀ : Ne p 0
      s : Set α
      hsm : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      hε : LT.lt (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun a => H …
      ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
    -/
    refine ⟨s, hsm, hs, ?_⟩
    rwa [eLpNorm_indicator_eq_restrict hsm.compl, eLpNorm_eq_lintegral_rpow_nnnorm hp₀ hp_top,
      one_div, ENNReal.rpow_inv_lt_iff]
    /-
      case inr.intro.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      β : Type u_6
      inst✝ : NormedAddCommGroup β
      hp_top : Ne p Top.top
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε✝ : Ne ε 0
      hp₀ : Ne p 0
      s : Set α
      hsm : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      hε : LT.lt (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun a => H …
      ⊢ LT.lt 0 p.toReal
    -/
    simp [ENNReal.toReal_pos, *]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.exists_snorm_indicator_compl_lt := Memℒp.exists_eLpNorm_indicator_compl_lt


