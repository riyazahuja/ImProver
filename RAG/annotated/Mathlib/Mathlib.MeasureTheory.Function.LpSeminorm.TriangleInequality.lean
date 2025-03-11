theorem eLpNorm'_add_le (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ)
    (hq1 : 1 ≤ q) : eLpNorm' (f + g) q μ ≤ eLpNorm' f q μ + eLpNorm' g q μ :=
  calc
    (∫⁻ a, (‖(f + g) a‖₊ : ℝ≥0∞) ^ q ∂μ) ^ (1 / q) ≤
        (∫⁻ a, ((fun a => (‖f a‖₊ : ℝ≥0∞)) + fun a => (‖g a‖₊ : ℝ≥0∞)) a ^ q ∂μ) ^ (1 / q) := by
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        q : Real
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hg : MeasureTheory.AEStronglyMeasurable g μ
        hq1 : LE.le 1 q
        ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnn …
      -/
      gcongr with a
      /-
        case h₁.hfg.h₁
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        q : Real
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hg : MeasureTheory.AEStronglyMeasurable g μ
        hq1 : LE.le 1 q
        a : α
        ⊢ LE.le (↑(NNNorm.nnnorm (HAdd.hAdd f g a))) (HAdd.hAdd (fun a => ↑(NNNorm.nnn …
      -/
      simp only [Pi.add_apply, ← ENNReal.coe_add, ENNReal.coe_le_coe, nnnorm_add_le]
      /-
        🎉 no goals
      -/
    _ ≤ eLpNorm' f q μ + eLpNorm' g q μ := ENNReal.lintegral_Lp_add_le hf.ennnorm hg.ennnorm hq1


@[deprecated (since := "2024-07-27")]
alias snorm'_add_le := eLpNorm'_add_le


theorem eLpNorm'_add_le_of_le_one (hf : AEStronglyMeasurable f μ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    eLpNorm' (f + g) q μ ≤ (2 : ℝ≥0∞) ^ (1 / q - 1) * (eLpNorm' f q μ + eLpNorm' g q μ) :=
  calc
    (∫⁻ a, (‖(f + g) a‖₊ : ℝ≥0∞) ^ q ∂μ) ^ (1 / q) ≤
        (∫⁻ a, ((fun a => (‖f a‖₊ : ℝ≥0∞)) + fun a => (‖g a‖₊ : ℝ≥0∞)) a ^ q ∂μ) ^ (1 / q) := by
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        q : Real
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hq0 : LE.le 0 q
        hq1 : LE.le q 1
        ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnn …
      -/
      gcongr with a
      /-
        case h₁.hfg.h₁
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        q : Real
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hq0 : LE.le 0 q
        hq1 : LE.le q 1
        a : α
        ⊢ LE.le (↑(NNNorm.nnnorm (HAdd.hAdd f g a))) (HAdd.hAdd (fun a => ↑(NNNorm.nnn …
      -/
      simp only [Pi.add_apply, ← ENNReal.coe_add, ENNReal.coe_le_coe, nnnorm_add_le]
      /-
        🎉 no goals
      -/
    _ ≤ (2 : ℝ≥0∞) ^ (1 / q - 1) * (eLpNorm' f q μ + eLpNorm' g q μ) :=
      ENNReal.lintegral_Lp_add_le_of_le_one hf.ennnorm hq0 hq1


@[deprecated (since := "2024-07-27")]
alias snorm'_add_le_of_le_one := eLpNorm'_add_le_of_le_one


theorem eLpNormEssSup_add_le {f g : α → E} :
    eLpNormEssSup (f + g) μ ≤ eLpNormEssSup f μ + eLpNormEssSup g μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    ⊢ LE.le (MeasureTheory.eLpNormEssSup (HAdd.hAdd f g) μ) (HAdd.hAdd (MeasureThe …
  -/
  refine le_trans (essSup_mono_ae (Eventually.of_forall fun x => ?_)) (ENNReal.essSup_add_le _ _)
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    x : α
    ⊢ LE.le ((fun x => ENorm.enorm (HAdd.hAdd f g x)) x) (HAdd.hAdd (fun x => ENor …
  -/
  simp_rw [Pi.add_apply, enorm_eq_nnnorm, ← ENNReal.coe_add, ENNReal.coe_le_coe]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    x : α
    ⊢ LE.le (NNNorm.nnnorm (HAdd.hAdd (f x) (g x))) (HAdd.hAdd (NNNorm.nnnorm (f x …
  -/
  exact nnnorm_add_le _ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_add_le := eLpNormEssSup_add_le


theorem eLpNorm_add_le {f g : α → E} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ)
    (hp1 : 1 ≤ p) : eLpNorm (f + g) p μ ≤ eLpNorm f p μ + eLpNorm g p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hp1 : LE.le 1 p
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HAdd.hAdd (MeasureTheory. …
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      hp1 : LE.le 1 p
      hp0 : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HAdd.hAdd (MeasureTheory. …
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hp1 : LE.le 1 p
    hp0 : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HAdd.hAdd (MeasureTheory. …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      hp1 : LE.le 1 p
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HAdd.hAdd (MeasureTheory. …
    -/
  · simp [hp_top, eLpNormEssSup_add_le]
    /-
      🎉 no goals
    -/
  have hp1_real : 1 ≤ p.toReal := by
    rwa [← ENNReal.one_toReal, ENNReal.toReal_le_toReal ENNReal.one_ne_top hp_top]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hp1 : LE.le 1 p
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    hp1_real : LE.le 1 p.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HAdd.hAdd (MeasureTheory. …
  -/
  repeat rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hp1 : LE.le 1 p
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    hp1_real : LE.le 1 p.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm' (HAdd.hAdd f g) p.toReal μ) (HAdd.hAdd (Measur …
  -/
  exact eLpNorm'_add_le hf hg hp1_real
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_add_le := eLpNorm_add_le


/-- A constant for the inequality `‖f + g‖_{L^p} ≤ C * (‖f‖_{L^p} + ‖g‖_{L^p})`. It is equal to `1`
for `p ≥ 1` or `p = 0`, and `2^(1/p-1)` in the more tricky interval `(0, 1)`. -/
noncomputable def LpAddConst (p : ℝ≥0∞) : ℝ≥0∞ :=
  if p ∈ Set.Ioo (0 : ℝ≥0∞) 1 then (2 : ℝ≥0∞) ^ (1 / p.toReal - 1) else 1


theorem LpAddConst_of_one_le {p : ℝ≥0∞} (hp : 1 ≤ p) : LpAddConst p = 1 := by
  /-
    p : ENNReal
    hp : LE.le 1 p
    ⊢ Eq (MeasureTheory.LpAddConst p) 1
  -/
  rw [LpAddConst, if_neg]
  /-
    case hnc
    p : ENNReal
    hp : LE.le 1 p
    ⊢ Not (Membership.mem (Set.Ioo 0 1) p)
  -/
  intro h
  /-
    case hnc
    p : ENNReal
    hp : LE.le 1 p
    h : Membership.mem (Set.Ioo 0 1) p
    ⊢ False
  -/
  exact lt_irrefl _ (h.2.trans_le hp)
  /-
    🎉 no goals
  -/


theorem LpAddConst_zero : LpAddConst 0 = 1 := by
  /-
    ⊢ Eq (MeasureTheory.LpAddConst 0) 1
  -/
  rw [LpAddConst, if_neg]
  /-
    case hnc
    ⊢ Not (Membership.mem (Set.Ioo 0 1) 0)
  -/
  intro h
  /-
    case hnc
    h : Membership.mem (Set.Ioo 0 1) 0
    ⊢ False
  -/
  exact lt_irrefl _ h.1
  /-
    🎉 no goals
  -/


theorem LpAddConst_lt_top (p : ℝ≥0∞) : LpAddConst p < ∞ := by
  /-
    p : ENNReal
    ⊢ LT.lt (MeasureTheory.LpAddConst p) Top.top
  -/
  rw [LpAddConst]
  /-
    p : ENNReal
    ⊢ LT.lt (ite (Membership.mem (Set.Ioo 0 1) p) (HPow.hPow 2 (HSub.hSub (HDiv.hD …
  -/
  split_ifs with h
    /-
      case pos
      p : ENNReal
      h : Membership.mem (Set.Ioo 0 1) p
      ⊢ LT.lt (HPow.hPow 2 (HSub.hSub (HDiv.hDiv 1 p.toReal) 1)) Top.top
    -/
  · apply ENNReal.rpow_lt_top_of_nonneg _ ENNReal.two_ne_top
    /-
      p : ENNReal
      h : Membership.mem (Set.Ioo 0 1) p
      ⊢ LE.le 0 (HSub.hSub (HDiv.hDiv 1 p.toReal) 1)
    -/
    rw [one_div, sub_nonneg, ← ENNReal.toReal_inv, ← ENNReal.one_toReal]
    /-
      p : ENNReal
      h : Membership.mem (Set.Ioo 0 1) p
      ⊢ LE.le (ENNReal.toReal 1) (Inv.inv p).toReal
    -/
    exact ENNReal.toReal_mono (by simpa using h.1.ne') (ENNReal.one_le_inv.2 h.2.le)
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : ENNReal
      h : Not (Membership.mem (Set.Ioo 0 1) p)
      ⊢ LT.lt 1 Top.top
    -/
  · exact ENNReal.one_lt_top
    /-
      🎉 no goals
    -/


theorem eLpNorm_add_le' (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ)
    (p : ℝ≥0∞) : eLpNorm (f + g) p μ ≤ LpAddConst p * (eLpNorm f p μ + eLpNorm g p μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    p : ENNReal
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HMul.hMul (MeasureTheory. …
  -/
  rcases eq_or_ne p 0 with (rfl | hp)
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) 0 μ) (HMul.hMul (MeasureTheory. …
    -/
  · simp only [eLpNorm_exponent_zero, add_zero, mul_zero, le_zero_iff]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    p : ENNReal
    hp : Ne p 0
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HMul.hMul (MeasureTheory. …
  -/
  rcases lt_or_le p 1 with (h'p | h'p)
    /-
      case inr.inl
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      p : ENNReal
      hp : Ne p 0
      h'p : LT.lt p 1
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HMul.hMul (MeasureTheory. …
    -/
  · simp only [eLpNorm_eq_eLpNorm' hp (h'p.trans ENNReal.one_lt_top).ne]
    /-
      case inr.inl
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      p : ENNReal
      hp : Ne p 0
      h'p : LT.lt p 1
      ⊢ LE.le (MeasureTheory.eLpNorm' (HAdd.hAdd f g) p.toReal μ) (HMul.hMul (Measur …
    -/
    convert eLpNorm'_add_le_of_le_one hf ENNReal.toReal_nonneg _
      /-
        case h.e'_4.h.e'_5
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hg : MeasureTheory.AEStronglyMeasurable g μ
        p : ENNReal
        hp : Ne p 0
        h'p : LT.lt p 1
        ⊢ Eq (MeasureTheory.LpAddConst p) (HPow.hPow 2 (HSub.hSub (HDiv.hDiv 1 p.toRea …
      -/
    · have : p ∈ Set.Ioo (0 : ℝ≥0∞) 1 := ⟨hp.bot_lt, h'p⟩
      /-
        case h.e'_4.h.e'_5
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hg : MeasureTheory.AEStronglyMeasurable g μ
        p : ENNReal
        hp : Ne p 0
        h'p : LT.lt p 1
        this : Membership.mem (Set.Ioo 0 1) p
        ⊢ Eq (MeasureTheory.LpAddConst p) (HPow.hPow 2 (HSub.hSub (HDiv.hDiv 1 p.toRea …
      -/
      simp only [LpAddConst, if_pos this]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.convert_3
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.AEStronglyMeasurable f μ
        hg : MeasureTheory.AEStronglyMeasurable g μ
        p : ENNReal
        hp : Ne p 0
        h'p : LT.lt p 1
        ⊢ LE.le p.toReal 1
      -/
    · simpa using ENNReal.toReal_mono ENNReal.one_ne_top h'p.le
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      p : ENNReal
      hp : Ne p 0
      h'p : LE.le 1 p
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd f g) p μ) (HMul.hMul (MeasureTheory. …
    -/
  · simpa [LpAddConst_of_one_le h'p] using eLpNorm_add_le hf hg h'p
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_add_le' := eLpNorm_add_le'


/-- Technical lemma to control the addition of functions in `L^p` even for `p < 1`: Given `δ > 0`,
there exists `η` such that two functions bounded by `η` in `L^p` have a sum bounded by `δ`. One
could take `η = δ / 2` for `p ≥ 1`, but the point of the lemma is that it works also for `p < 1`.
-/
theorem exists_Lp_half (p : ℝ≥0∞) {δ : ℝ≥0∞} (hδ : δ ≠ 0) :
    ∃ η : ℝ≥0∞,
      0 < η ∧
        ∀ (f g : α → E), AEStronglyMeasurable f μ → AEStronglyMeasurable g μ →
          eLpNorm f p μ ≤ η → eLpNorm g p μ ≤ η → eLpNorm (f + g) p μ < δ := by
  have :
    Tendsto (fun η : ℝ≥0∞ => LpAddConst p * (η + η)) (𝓝[>] 0) (𝓝 (LpAddConst p * (0 + 0))) :=
    (ENNReal.Tendsto.const_mul (tendsto_id.add tendsto_id)
          (Or.inr (LpAddConst_lt_top p).ne)).mono_left
      nhdsWithin_le_nhds
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p δ : ENNReal
    hδ : Ne δ 0
    this : Filter.Tendsto (fun η => HMul.hMul (MeasureTheory.LpAddConst p) (HAdd.h …
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (f g : α → E), MeasureTheory.AEStronglyMe …
  -/
  simp only [add_zero, mul_zero] at this
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p δ : ENNReal
    hδ : Ne δ 0
    this : Filter.Tendsto (fun η => HMul.hMul (MeasureTheory.LpAddConst p) (HAdd.h …
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (f g : α → E), MeasureTheory.AEStronglyMe …
  -/
  rcases (((tendsto_order.1 this).2 δ hδ.bot_lt).and self_mem_nhdsWithin).exists with ⟨η, hη, ηpos⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p δ : ENNReal
    hδ : Ne δ 0
    this : Filter.Tendsto (fun η => HMul.hMul (MeasureTheory.LpAddConst p) (HAdd.h …
    η : ENNReal
    hη : LT.lt (HMul.hMul (MeasureTheory.LpAddConst p) (HAdd.hAdd η η)) δ
    ηpos : LT.lt 0 η
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (f g : α → E), MeasureTheory.AEStronglyMe …
  -/
  refine ⟨η, ηpos, fun f g hf hg Hf Hg => ?_⟩
  calc
    eLpNorm (f + g) p μ ≤ LpAddConst p * (eLpNorm f p μ + eLpNorm g p μ) := eLpNorm_add_le' hf hg p
    _ ≤ LpAddConst p * (η + η) := by gcongr
    _ < δ := hη


theorem eLpNorm_sub_le' (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ)
    (p : ℝ≥0∞) : eLpNorm (f - g) p μ ≤ LpAddConst p * (eLpNorm f p μ + eLpNorm g p μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    p : ENNReal
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) (HMul.hMul (MeasureTheory. …
  -/
  simpa only [sub_eq_add_neg, eLpNorm_neg] using eLpNorm_add_le' hf hg.neg p
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_sub_le' := eLpNorm_sub_le'


theorem eLpNorm_sub_le {f g : α → E} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ)
    (hp : 1 ≤ p) : eLpNorm (f - g) p μ ≤ eLpNorm f p μ + eLpNorm g p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hp : LE.le 1 p
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) (HAdd.hAdd (MeasureTheory. …
  -/
  simpa [LpAddConst_of_one_le hp] using eLpNorm_sub_le' hf hg p
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_sub_le := eLpNorm_sub_le


theorem eLpNorm_add_lt_top {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    eLpNorm (f + g) p μ < ∞ :=
  calc
    eLpNorm (f + g) p μ ≤ LpAddConst p * (eLpNorm f p μ + eLpNorm g p μ) :=
      eLpNorm_add_le' hf.aestronglyMeasurable hg.aestronglyMeasurable p
    _ < ∞ := by
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        ⊢ LT.lt (HMul.hMul (MeasureTheory.LpAddConst p) (HAdd.hAdd (MeasureTheory.eLpN …
      -/
      apply ENNReal.mul_lt_top (LpAddConst_lt_top p)
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        f g : α → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        ⊢ LT.lt (HAdd.hAdd (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p μ) …
      -/
      exact ENNReal.add_lt_top.2 ⟨hf.2, hg.2⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-27")]
alias snorm_add_lt_top := eLpNorm_add_lt_top


theorem eLpNorm'_sum_le {ι} {f : ι → α → E} {s : Finset ι}
    (hfs : ∀ i, i ∈ s → AEStronglyMeasurable (f i) μ) (hq1 : 1 ≤ q) :
    eLpNorm' (∑ i ∈ s, f i) q μ ≤ ∑ i ∈ s, eLpNorm' (f i) q μ :=
  Finset.le_sum_of_subadditive_on_pred (fun f : α → E => eLpNorm' f q μ)
    (fun f => AEStronglyMeasurable f μ) (eLpNorm'_zero (zero_lt_one.trans_le hq1))
    (fun _f _g hf hg => eLpNorm'_add_le hf hg hq1) (fun _f _g hf hg => hf.add hg) _ hfs


@[deprecated (since := "2024-07-27")]
alias snorm'_sum_le := eLpNorm'_sum_le


theorem eLpNorm_sum_le {ι} {f : ι → α → E} {s : Finset ι}
    (hfs : ∀ i, i ∈ s → AEStronglyMeasurable (f i) μ) (hp1 : 1 ≤ p) :
    eLpNorm (∑ i ∈ s, f i) p μ ≤ ∑ i ∈ s, eLpNorm (f i) p μ :=
  Finset.le_sum_of_subadditive_on_pred (fun f : α → E => eLpNorm f p μ)
    (fun f => AEStronglyMeasurable f μ) eLpNorm_zero (fun _f _g hf hg => eLpNorm_add_le hf hg hp1)
    (fun _f _g hf hg => hf.add hg) _ hfs


@[deprecated (since := "2024-07-27")]
alias snorm_sum_le := eLpNorm_sum_le


theorem Memℒp.add {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) : Memℒp (f + g) p μ :=
  ⟨AEStronglyMeasurable.add hf.1 hg.1, eLpNorm_add_lt_top hf hg⟩


theorem Memℒp.sub {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) : Memℒp (f - g) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    ⊢ MeasureTheory.Memℒp (HSub.hSub f g) p μ
  -/
  rw [sub_eq_add_neg]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : α → E
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    ⊢ MeasureTheory.Memℒp (HAdd.hAdd f (Neg.neg g)) p μ
  -/
  exact hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem memℒp_finset_sum {ι} (s : Finset ι) {f : ι → α → E} (hf : ∀ i ∈ s, Memℒp (f i) p μ) :
    Memℒp (fun a => ∑ i ∈ s, f i a) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.Memℒp (fun a => s.sum fun i => f i a) p μ
  -/
  haveI : DecidableEq ι := Classical.decEq _
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ
    this : DecidableEq ι
    ⊢ MeasureTheory.Memℒp (fun a => s.sum fun i => f i a) p μ
  -/
  revert hf
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    this : DecidableEq ι
    ⊢ (∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ) → MeasureThe …
  -/
  refine Finset.induction_on s ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      ι : Type u_3
      s : Finset ι
      f : ι → α → E
      this : DecidableEq ι
      ⊢ (∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → MeasureTheory …
    -/
  · simp only [zero_mem_ℒp', Finset.sum_empty, imp_true_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      ι : Type u_3
      s : Finset ι
      f : ι → α → E
      this : DecidableEq ι
      ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → ((∀ (i : ι), Membership …
    -/
  · intro i s his ih hf
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      ι : Type u_3
      s✝ : Finset ι
      f : ι → α → E
      this : DecidableEq ι
      i : ι
      s : Finset ι
      his : Not (Membership.mem s i)
      ih : (∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ) → Measure …
      hf : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → MeasureTheory.Memℒp …
      ⊢ MeasureTheory.Memℒp (fun a => (Insert.insert i s).sum fun i => f i a) p μ
    -/
    simp only [his, Finset.sum_insert, not_false_iff]
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      ι : Type u_3
      s✝ : Finset ι
      f : ι → α → E
      this : DecidableEq ι
      i : ι
      s : Finset ι
      his : Not (Membership.mem s i)
      ih : (∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ) → Measure …
      hf : ∀ (i_1 : ι), Membership.mem (Insert.insert i s) i_1 → MeasureTheory.Memℒp …
      ⊢ MeasureTheory.Memℒp (fun a => HAdd.hAdd (f i a) (s.sum fun i => f i a)) p μ
    -/
    exact (hf i (s.mem_insert_self i)).add (ih fun j hj => hf j (Finset.mem_insert_of_mem hj))
    /-
      🎉 no goals
    -/


theorem memℒp_finset_sum' {ι} (s : Finset ι) {f : ι → α → E} (hf : ∀ i ∈ s, Memℒp (f i) p μ) :
    Memℒp (∑ i ∈ s, f i) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.Memℒp (s.sum fun i => f i) p μ
  -/
  convert memℒp_finset_sum s hf using 1
  /-
    case h.e'_6
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ
    ⊢ Eq (s.sum fun i => f i) fun a => s.sum fun i => f i a
  -/
  ext x
  /-
    case h.e'_6.h
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    ι : Type u_3
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Memℒp (f i) p μ
    x : α
    ⊢ Eq (s.sum (fun i => f i) x) (s.sum fun i => f i x)
  -/
  simp
  /-
    🎉 no goals
  -/


