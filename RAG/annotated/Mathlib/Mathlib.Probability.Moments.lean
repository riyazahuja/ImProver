/-- Moment of a real random variable, `μ[X ^ p]`. -/
def moment (X : Ω → ℝ) (p : ℕ) (μ : Measure Ω) : ℝ :=
  μ[X ^ p]


/-- Central moment of a real random variable, `μ[(X - μ[X]) ^ p]`. -/
def centralMoment (X : Ω → ℝ) (p : ℕ) (μ : Measure Ω) : ℝ := by
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    X✝ : Ω → Real
    p✝ : Nat
    μ✝ : MeasureTheory.Measure Ω
    X : Ω → Real
    p : Nat
    μ : MeasureTheory.Measure Ω
    ⊢ Real
  -/
  have m := fun (x : Ω) => μ[X] -- Porting note: Lean deems `μ[(X - fun x => μ[X]) ^ p]` ambiguous
  /-
    Ω : Type u_1
    ι : Type u_2
    m✝ : MeasurableSpace Ω
    X✝ : Ω → Real
    p✝ : Nat
    μ✝ : MeasureTheory.Measure Ω
    X : Ω → Real
    p : Nat
    μ : MeasureTheory.Measure Ω
    m : Ω → Real
    ⊢ Real
  -/
  exact μ[(X - m) ^ p]
  /-
    🎉 no goals
  -/


@[simp]
theorem moment_zero (hp : p ≠ 0) : moment 0 p μ = 0 := by
  simp only [moment, hp, zero_pow, Ne, not_false_iff, Pi.zero_apply, integral_const,
    smul_eq_mul, mul_zero, integral_zero]


@[simp]
theorem centralMoment_zero (hp : p ≠ 0) : centralMoment 0 p μ = 0 := by
  simp only [centralMoment, hp, Pi.zero_apply, integral_const, smul_eq_mul,
    mul_zero, zero_sub, Pi.pow_apply, Pi.neg_apply, neg_zero, zero_pow, Ne, not_false_iff]


theorem centralMoment_one' [IsFiniteMeasure μ] (h_int : Integrable X μ) :
    centralMoment X 1 μ = (1 - (μ Set.univ).toReal) * μ[X] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_int : MeasureTheory.Integrable X μ
    ⊢ Eq (ProbabilityTheory.centralMoment X 1 μ) (HMul.hMul (HSub.hSub 1 (μ Set.un …
  -/
  simp only [centralMoment, Pi.sub_apply, pow_one]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_int : MeasureTheory.Integrable X μ
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (X x) (MeasureTheory.integra …
  -/
  rw [integral_sub h_int (integrable_const _)]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_int : MeasureTheory.Integrable X μ
    ⊢ Eq (HSub.hSub (MeasureTheory.integral μ fun a => X a) (MeasureTheory.integra …
  -/
  simp only [sub_mul, integral_const, smul_eq_mul, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem centralMoment_one [IsZeroOrProbabilityMeasure μ] : centralMoment X 1 μ = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.centralMoment X 1 μ) 0
  -/
  rcases eq_zero_or_isProbabilityMeasure μ with rfl | h
    /-
      case inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure 0
      ⊢ Eq (ProbabilityTheory.centralMoment X 1 0) 0
    -/
  · simp [centralMoment]
    /-
      🎉 no goals
    -/
  /-
    case inr
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    h : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.centralMoment X 1 μ) 0
  -/
  by_cases h_int : Integrable X μ
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : MeasureTheory.IsProbabilityMeasure μ
      h_int : MeasureTheory.Integrable X μ
      ⊢ Eq (ProbabilityTheory.centralMoment X 1 μ) 0
    -/
  · rw [centralMoment_one' h_int]
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : MeasureTheory.IsProbabilityMeasure μ
      h_int : MeasureTheory.Integrable X μ
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (μ Set.univ).toReal) (MeasureTheory.integral μ fu …
    -/
    simp only [measure_univ, ENNReal.one_toReal, sub_self, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : MeasureTheory.IsProbabilityMeasure μ
      h_int : Not (MeasureTheory.Integrable X μ)
      ⊢ Eq (ProbabilityTheory.centralMoment X 1 μ) 0
    -/
  · simp only [centralMoment, Pi.sub_apply, pow_one]
    have : ¬Integrable (fun x => X x - integral μ X) μ := by
      refine fun h_sub => h_int ?_
      have h_add : X = (fun x => X x - integral μ X) + fun _ => integral μ X := by ext1 x; simp
      rw [h_add]
      exact h_sub.add (integrable_const _)
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : MeasureTheory.IsProbabilityMeasure μ
      h_int : Not (MeasureTheory.Integrable X μ)
      this : Not (MeasureTheory.Integrable (fun x => HSub.hSub (X x) (MeasureTheory. …
      ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (X x) (MeasureTheory.integra …
    -/
    rw [integral_undef this]
    /-
      🎉 no goals
    -/


theorem centralMoment_two_eq_variance [IsFiniteMeasure μ] (hX : Memℒp X 2 μ) :
                                             /-
                                               Ω : Type u_1
                                               m : MeasurableSpace Ω
                                               X : Ω → Real
                                               μ : MeasureTheory.Measure Ω
                                               inst✝ : MeasureTheory.IsFiniteMeasure μ
                                               hX : MeasureTheory.Memℒp X 2 μ
                                               ⊢ Eq (ProbabilityTheory.centralMoment X 2 μ) (ProbabilityTheory.variance X μ)
                                             -/
    centralMoment X 2 μ = variance X μ := by rw [hX.variance_eq]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Moment generating function of a real random variable `X`: `fun t => μ[exp(t*X)]`. -/
def mgf (X : Ω → ℝ) (μ : Measure Ω) (t : ℝ) : ℝ :=
  μ[fun ω => exp (t * X ω)]


/-- Cumulant generating function of a real random variable `X`: `fun t => log μ[exp(t*X)]`. -/
def cgf (X : Ω → ℝ) (μ : Measure Ω) (t : ℝ) : ℝ :=
  log (mgf X μ t)


@[simp]
theorem mgf_zero_fun : mgf 0 μ t = (μ Set.univ).toReal := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    ⊢ Eq (ProbabilityTheory.mgf 0 μ t) (μ Set.univ).toReal
  -/
  simp only [mgf, Pi.zero_apply, mul_zero, exp_zero, integral_const, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
                                                                 /-
                                                                   Ω : Type u_1
                                                                   m : MeasurableSpace Ω
                                                                   μ : MeasureTheory.Measure Ω
                                                                   t : Real
                                                                   ⊢ Eq (ProbabilityTheory.cgf 0 μ t) (Real.log (μ Set.univ).toReal)
                                                                 -/
theorem cgf_zero_fun : cgf 0 μ t = log (μ Set.univ).toReal := by simp only [cgf, mgf_zero_fun]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
                                                             /-
                                                               Ω : Type u_1
                                                               m : MeasurableSpace Ω
                                                               X : Ω → Real
                                                               t : Real
                                                               ⊢ Eq (ProbabilityTheory.mgf X 0 t) 0
                                                             -/
theorem mgf_zero_measure : mgf X (0 : Measure Ω) t = 0 := by simp only [mgf, integral_zero_measure]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem cgf_zero_measure : cgf X (0 : Measure Ω) t = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    t : Real
    ⊢ Eq (ProbabilityTheory.cgf X 0 t) 0
  -/
  simp only [cgf, log_zero, mgf_zero_measure]
  /-
    🎉 no goals
  -/


@[simp]
theorem mgf_const' (c : ℝ) : mgf (fun _ => c) μ t = (μ Set.univ).toReal * exp (t * c) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t c : Real
    ⊢ Eq (ProbabilityTheory.mgf (fun x => c) μ t) (HMul.hMul (μ Set.univ).toReal ( …
  -/
  simp only [mgf, integral_const, smul_eq_mul]
  /-
    🎉 no goals
  -/

-- @[simp] -- Porting note: `simp only` already proves this

theorem mgf_const (c : ℝ) [IsProbabilityMeasure μ] : mgf (fun _ => c) μ t = exp (t * c) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t c : Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.mgf (fun x => c) μ t) (Real.exp (HMul.hMul t c))
  -/
  simp only [mgf_const', measure_univ, ENNReal.one_toReal, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem cgf_const' [IsFiniteMeasure μ] (hμ : μ ≠ 0) (c : ℝ) :
    cgf (fun _ => c) μ t = log (μ Set.univ).toReal + t * c := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    c : Real
    ⊢ Eq (ProbabilityTheory.cgf (fun x => c) μ t) (HAdd.hAdd (Real.log (μ Set.univ …
  -/
  simp only [cgf, mgf_const']
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    c : Real
    ⊢ Eq (Real.log (HMul.hMul (μ Set.univ).toReal (Real.exp (HMul.hMul t c)))) (HA …
  -/
  rw [log_mul _ (exp_pos _).ne']
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hμ : Ne μ 0
      c : Real
      ⊢ Eq (HAdd.hAdd (Real.log (μ Set.univ).toReal) (Real.log (Real.exp (HMul.hMul  …
    -/
  · rw [log_exp _]
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hμ : Ne μ 0
      c : Real
      ⊢ Ne (μ Set.univ).toReal 0
    -/
  · rw [Ne, ENNReal.toReal_eq_zero_iff, Measure.measure_univ_eq_zero]
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hμ : Ne μ 0
      c : Real
      ⊢ Not (Or (Eq μ 0) (Eq (μ Set.univ) Top.top))
    -/
    simp only [hμ, measure_ne_top μ Set.univ, or_self_iff, not_false_iff]
    /-
      🎉 no goals
    -/


@[simp]
theorem cgf_const [IsProbabilityMeasure μ] (c : ℝ) : cgf (fun _ => c) μ t = t * c := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    c : Real
    ⊢ Eq (ProbabilityTheory.cgf (fun x => c) μ t) (HMul.hMul t c)
  -/
  simp only [cgf, mgf_const, log_exp]
  /-
    🎉 no goals
  -/


@[simp]
theorem mgf_zero' : mgf X μ 0 = (μ Set.univ).toReal := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (ProbabilityTheory.mgf X μ 0) (μ Set.univ).toReal
  -/
  simp only [mgf, zero_mul, exp_zero, integral_const, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/

-- @[simp] -- Porting note: `simp only` already proves this

theorem mgf_zero [IsProbabilityMeasure μ] : mgf X μ 0 = 1 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.mgf X μ 0) 1
  -/
  simp only [mgf_zero', measure_univ, ENNReal.one_toReal]
  /-
    🎉 no goals
  -/


                                                              /-
                                                                Ω : Type u_1
                                                                m : MeasurableSpace Ω
                                                                X : Ω → Real
                                                                μ : MeasureTheory.Measure Ω
                                                                ⊢ Eq (ProbabilityTheory.cgf X μ 0) (Real.log (μ Set.univ).toReal)
                                                              -/
theorem cgf_zero' : cgf X μ 0 = log (μ Set.univ).toReal := by simp only [cgf, mgf_zero']
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem cgf_zero [IsZeroOrProbabilityMeasure μ] : cgf X μ 0 = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.cgf X μ 0) 0
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rcases eq_zero_or_isProbabilityMeasure μ with rfl | h <;> simp [cgf_zero']
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem mgf_undef (hX : ¬Integrable (fun ω => exp (t * X ω)) μ) : mgf X μ t = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    hX : Not (MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ)
    ⊢ Eq (ProbabilityTheory.mgf X μ t) 0
  -/
  simp only [mgf, integral_undef hX]
  /-
    🎉 no goals
  -/


theorem cgf_undef (hX : ¬Integrable (fun ω => exp (t * X ω)) μ) : cgf X μ t = 0 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    hX : Not (MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ)
    ⊢ Eq (ProbabilityTheory.cgf X μ t) 0
  -/
  simp only [cgf, mgf_undef hX, log_zero]
  /-
    🎉 no goals
  -/


theorem mgf_nonneg : 0 ≤ mgf X μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    ⊢ LE.le 0 (ProbabilityTheory.mgf X μ t)
  -/
  unfold mgf; positivity
              /-
                🎉 no goals
              -/


theorem mgf_pos' (hμ : μ ≠ 0) (h_int_X : Integrable (fun ω => exp (t * X ω)) μ) :
    0 < mgf X μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    hμ : Ne μ 0
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LT.lt 0 (ProbabilityTheory.mgf X μ t)
  -/
  simp_rw [mgf]
  have : ∫ x : Ω, exp (t * X x) ∂μ = ∫ x : Ω in Set.univ, exp (t * X x) ∂μ := by
    simp only [Measure.restrict_univ]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    hμ : Ne μ 0
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
    ⊢ LT.lt 0 (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x)))
  -/
  rw [this, setIntegral_pos_iff_support_of_nonneg_ae _ _]
  · have h_eq_univ : (Function.support fun x : Ω => exp (t * X x)) = Set.univ := by
      ext1 x
      simp only [Function.mem_support, Set.mem_univ, iff_true]
      exact (exp_pos _).ne'
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      h_eq_univ : Eq (Function.support fun x => Real.exp (HMul.hMul t (X x))) Set.univ
      ⊢ LT.lt 0 (μ (Inter.inter (Function.support fun x => Real.exp (HMul.hMul t (X  …
    -/
    rw [h_eq_univ, Set.inter_univ _]
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      h_eq_univ : Eq (Function.support fun x => Real.exp (HMul.hMul t (X x))) Set.univ
      ⊢ LT.lt 0 (μ Set.univ)
    -/
    refine Ne.bot_lt ?_
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      h_eq_univ : Eq (Function.support fun x => Real.exp (HMul.hMul t (X x))) Set.univ
      ⊢ Ne (μ Set.univ) Bot.bot
    -/
    simp only [hμ, ENNReal.bot_eq_zero, Ne, Measure.measure_univ_eq_zero, not_false_iff]
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      ⊢ (MeasureTheory.ae (μ.restrict Set.univ)).EventuallyLE 0 fun x => Real.exp (H …
    -/
  · filter_upwards with x
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      x : Ω
      ⊢ LE.le (0 x) (Real.exp (HMul.hMul t (X x)))
    -/
    rw [Pi.zero_apply]
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      x : Ω
      ⊢ LE.le 0 (Real.exp (HMul.hMul t (X x)))
    -/
    exact (exp_pos _).le
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      hμ : Ne μ 0
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      this : Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (X x))) (Me …
      ⊢ MeasureTheory.IntegrableOn (fun x => Real.exp (HMul.hMul t (X x))) Set.univ μ
    -/
  · rwa [integrableOn_univ]
    /-
      🎉 no goals
    -/


theorem mgf_pos [IsProbabilityMeasure μ] (h_int_X : Integrable (fun ω => exp (t * X ω)) μ) :
    0 < mgf X μ t :=
  mgf_pos' (IsProbabilityMeasure.ne_zero μ) h_int_X


lemma mgf_id_map (hX : AEMeasurable X μ) : mgf id (μ.map X) = mgf X μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    hX : AEMeasurable X μ
    ⊢ Eq (ProbabilityTheory.mgf id (MeasureTheory.Measure.map X μ)) (ProbabilityTh …
  -/
  ext t
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    hX : AEMeasurable X μ
    t : Real
    ⊢ Eq (ProbabilityTheory.mgf id (MeasureTheory.Measure.map X μ) t) (Probability …
  -/
  rw [mgf, integral_map hX]
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : AEMeasurable X μ
      t : Real
      ⊢ Eq (MeasureTheory.integral μ fun x => Real.exp (HMul.hMul t (id (X x)))) (Pr …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      hX : AEMeasurable X μ
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (id ω)))  …
    -/
  · exact (measurable_const_mul _).exp.aestronglyMeasurable
    /-
      🎉 no goals
    -/


                                                    /-
                                                      Ω : Type u_1
                                                      m : MeasurableSpace Ω
                                                      X : Ω → Real
                                                      μ : MeasureTheory.Measure Ω
                                                      t : Real
                                                      ⊢ Eq (ProbabilityTheory.mgf (Neg.neg X) μ t) (ProbabilityTheory.mgf X μ (Neg.n …
                                                    -/
theorem mgf_neg : mgf (-X) μ t = mgf X μ (-t) := by simp_rw [mgf, Pi.neg_apply, mul_neg, neg_mul]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                    /-
                                                      Ω : Type u_1
                                                      m : MeasurableSpace Ω
                                                      X : Ω → Real
                                                      μ : MeasureTheory.Measure Ω
                                                      t : Real
                                                      ⊢ Eq (ProbabilityTheory.cgf (Neg.neg X) μ t) (ProbabilityTheory.cgf X μ (Neg.n …
                                                    -/
theorem cgf_neg : cgf (-X) μ t = cgf X μ (-t) := by simp_rw [cgf, mgf_neg]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem mgf_smul_left (α : ℝ) : mgf (α • X) μ t = mgf X μ (α * t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    ⊢ Eq (ProbabilityTheory.mgf (HSMul.hSMul α X) μ t) (ProbabilityTheory.mgf X μ  …
  -/
  simp_rw [mgf, Pi.smul_apply, smul_eq_mul, mul_comm α t, mul_assoc]
  /-
    🎉 no goals
  -/


theorem mgf_const_add (α : ℝ) : mgf (fun ω => α + X ω) μ t = exp (t * α) * mgf X μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    ⊢ Eq (ProbabilityTheory.mgf (fun ω => HAdd.hAdd α (X ω)) μ t) (HMul.hMul (Real …
  -/
  rw [mgf, mgf, ← integral_mul_left]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => (fun ω => Real.exp (HMul.hMul t (HAdd. …
  -/
  congr with x
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    x : Ω
    ⊢ Eq ((fun ω => Real.exp (HMul.hMul t (HAdd.hAdd α (X ω)))) x) (HMul.hMul (Rea …
  -/
  dsimp
  /-
    case e_f.h
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    x : Ω
    ⊢ Eq (Real.exp (HMul.hMul t (HAdd.hAdd α (X x)))) (HMul.hMul (Real.exp (HMul.h …
  -/
  rw [mul_add, exp_add]
  /-
    🎉 no goals
  -/


theorem mgf_add_const (α : ℝ) : mgf (fun ω => X ω + α) μ t = mgf X μ t *  exp (t * α) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t α : Real
    ⊢ Eq (ProbabilityTheory.mgf (fun ω => HAdd.hAdd (X ω) α) μ t) (HMul.hMul (Prob …
  -/
  simp only [add_comm, mgf_const_add, mul_comm]
  /-
    🎉 no goals
  -/


/-- This is a trivial application of `IndepFun.comp` but it will come up frequently. -/
theorem IndepFun.exp_mul {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ) (s t : ℝ) :
    IndepFun (fun ω => exp (s * X ω)) (fun ω => exp (t * Y ω)) μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    s t : Real
    ⊢ ProbabilityTheory.IndepFun (fun ω => Real.exp (HMul.hMul s (X ω))) (fun ω => …
  -/
  have h_meas : ∀ t, Measurable fun x => exp (t * x) := fun t => (measurable_id'.const_mul t).exp
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    s t : Real
    h_meas : ∀ (t : Real), Measurable fun x => Real.exp (HMul.hMul t x)
    ⊢ ProbabilityTheory.IndepFun (fun ω => Real.exp (HMul.hMul s (X ω))) (fun ω => …
  -/
  change IndepFun ((fun x => exp (s * x)) ∘ X) ((fun x => exp (t * x)) ∘ Y) μ
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    s t : Real
    h_meas : ∀ (t : Real), Measurable fun x => Real.exp (HMul.hMul t x)
    ⊢ ProbabilityTheory.IndepFun (Function.comp (fun x => Real.exp (HMul.hMul s x) …
  -/
  exact IndepFun.comp h_indep (h_meas s) (h_meas t)
  /-
    🎉 no goals
  -/


theorem IndepFun.mgf_add {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ)
    (hX : AEStronglyMeasurable (fun ω => exp (t * X ω)) μ)
    (hY : AEStronglyMeasurable (fun ω => exp (t * Y ω)) μ) :
    mgf (X + Y) μ t = mgf X μ t * mgf Y μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (X ω)) …
    hY : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (Y ω)) …
    ⊢ Eq (ProbabilityTheory.mgf (HAdd.hAdd X Y) μ t) (HMul.hMul (ProbabilityTheory …
  -/
  simp_rw [mgf, Pi.add_apply, mul_add, exp_add]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (X ω)) …
    hY : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (Y ω)) …
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul (Real.exp (HMul.hMul t (X x) …
  -/
  exact (h_indep.exp_mul t t).integral_mul hX hY
  /-
    🎉 no goals
  -/


theorem IndepFun.mgf_add' {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ) (hX : AEStronglyMeasurable X μ)
    (hY : AEStronglyMeasurable Y μ) : mgf (X + Y) μ t = mgf X μ t * mgf Y μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    ⊢ Eq (ProbabilityTheory.mgf (HAdd.hAdd X Y) μ t) (HMul.hMul (ProbabilityTheory …
  -/
  have A : Continuous fun x : ℝ => exp (t * x) := by fun_prop
  have h'X : AEStronglyMeasurable (fun ω => exp (t * X ω)) μ :=
    A.aestronglyMeasurable.comp_aemeasurable hX.aemeasurable
  have h'Y : AEStronglyMeasurable (fun ω => exp (t * Y ω)) μ :=
    A.aestronglyMeasurable.comp_aemeasurable hY.aemeasurable
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    hX : MeasureTheory.AEStronglyMeasurable X μ
    hY : MeasureTheory.AEStronglyMeasurable Y μ
    A : Continuous fun x => Real.exp (HMul.hMul t x)
    h'X : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (X ω) …
    h'Y : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (Y ω) …
    ⊢ Eq (ProbabilityTheory.mgf (HAdd.hAdd X Y) μ t) (HMul.hMul (ProbabilityTheory …
  -/
  exact h_indep.mgf_add h'X h'Y
  /-
    🎉 no goals
  -/


theorem IndepFun.cgf_add {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ)
    (h_int_X : Integrable (fun ω => exp (t * X ω)) μ)
    (h_int_Y : Integrable (fun ω => exp (t * Y ω)) μ) :
    cgf (X + Y) μ t = cgf X μ t + cgf Y μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
    ⊢ Eq (ProbabilityTheory.cgf (HAdd.hAdd X Y) μ t) (HAdd.hAdd (ProbabilityTheory …
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      X Y : Ω → Real
      h_indep : ProbabilityTheory.IndepFun X Y μ
      h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
      hμ : Eq μ 0
      ⊢ Eq (ProbabilityTheory.cgf (HAdd.hAdd X Y) μ t) (HAdd.hAdd (ProbabilityTheory …
    -/
  · simp [hμ]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
    hμ : Not (Eq μ 0)
    ⊢ Eq (ProbabilityTheory.cgf (HAdd.hAdd X Y) μ t) (HAdd.hAdd (ProbabilityTheory …
  -/
  simp only [cgf, h_indep.mgf_add h_int_X.aestronglyMeasurable h_int_Y.aestronglyMeasurable]
  /-
    case neg
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
    hμ : Not (Eq μ 0)
    ⊢ Eq (Real.log (HMul.hMul (ProbabilityTheory.mgf X μ t) (ProbabilityTheory.mgf …
  -/
  exact log_mul (mgf_pos' hμ h_int_X).ne' (mgf_pos' hμ h_int_Y).ne'
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable_exp_mul_add {X Y : Ω → ℝ}
    (h_int_X : AEStronglyMeasurable (fun ω => exp (t * X ω)) μ)
    (h_int_Y : AEStronglyMeasurable (fun ω => exp (t * Y ω)) μ) :
    AEStronglyMeasurable (fun ω => exp (t * (X + Y) ω)) μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_int_X : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t ( …
    h_int_Y : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t ( …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t (HAdd.hAd …
  -/
  simp_rw [Pi.add_apply, mul_add, exp_add]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_int_X : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t ( …
    h_int_Y : MeasureTheory.AEStronglyMeasurable (fun ω => Real.exp (HMul.hMul t ( …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => HMul.hMul (Real.exp (HMul.hMul  …
  -/
  exact AEStronglyMeasurable.mul h_int_X h_int_Y
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable_exp_mul_sum {X : ι → Ω → ℝ} {s : Finset ι}
    (h_int : ∀ i ∈ s, AEStronglyMeasurable (fun ω => exp (t * X i ω)) μ) :
    AEStronglyMeasurable (fun ω => exp (t * (∑ i ∈ s, X i) ω)) μ := by
  classical
  induction' s using Finset.induction_on with i s hi_notin_s h_rec h_int
  · simp only [Pi.zero_apply, sum_apply, sum_empty, mul_zero, exp_zero]
    exact aestronglyMeasurable_const
  · have : ∀ i : ι, i ∈ s → AEStronglyMeasurable (fun ω : Ω => exp (t * X i ω)) μ := fun i hi =>
      h_int i (mem_insert_of_mem hi)
    specialize h_rec this
    rw [sum_insert hi_notin_s]
    apply aestronglyMeasurable_exp_mul_add (h_int i (mem_insert_self _ _)) h_rec


theorem IndepFun.integrable_exp_mul_add {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ)
    (h_int_X : Integrable (fun ω => exp (t * X ω)) μ)
    (h_int_Y : Integrable (fun ω => exp (t * Y ω)) μ) :
    Integrable (fun ω => exp (t * (X + Y) ω)) μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
    ⊢ MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (HAdd.hAdd X Y ω))) μ
  -/
  simp_rw [Pi.add_apply, mul_add, exp_add]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X Y : Ω → Real
    h_indep : ProbabilityTheory.IndepFun X Y μ
    h_int_X : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    h_int_Y : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (Y ω))) μ
    ⊢ MeasureTheory.Integrable (fun ω => HMul.hMul (Real.exp (HMul.hMul t (X ω)))  …
  -/
  exact (h_indep.exp_mul t t).integrable_mul h_int_X h_int_Y
  /-
    🎉 no goals
  -/


theorem iIndepFun.integrable_exp_mul_sum [IsFiniteMeasure μ] {X : ι → Ω → ℝ}
    (h_indep : iIndepFun (fun _ => inferInstance) X μ) (h_meas : ∀ i, Measurable (X i))
    {s : Finset ι} (h_int : ∀ i ∈ s, Integrable (fun ω => exp (t * X i ω)) μ) :
    Integrable (fun ω => exp (t * (∑ i ∈ s, X i) ω)) μ := by
  classical
  induction' s using Finset.induction_on with i s hi_notin_s h_rec h_int
  · simp only [Pi.zero_apply, sum_apply, sum_empty, mul_zero, exp_zero]
    exact integrable_const _
  · have : ∀ i : ι, i ∈ s → Integrable (fun ω : Ω => exp (t * X i ω)) μ := fun i hi =>
      h_int i (mem_insert_of_mem hi)
    specialize h_rec this
    rw [sum_insert hi_notin_s]
    refine IndepFun.integrable_exp_mul_add ?_ (h_int i (mem_insert_self _ _)) h_rec
    exact (h_indep.indepFun_finset_sum_of_not_mem h_meas hi_notin_s).symm

-- TODO(vilin97): weaken `h_meas` to `AEMeasurable (X i)` or `AEStronglyMeasurable (X i)` throughout
-- https://github.com/leanprover-community/mathlib4/issues/20367

theorem iIndepFun.mgf_sum {X : ι → Ω → ℝ}
    (h_indep : iIndepFun (fun _ => inferInstance) X μ) (h_meas : ∀ i, Measurable (X i))
    (s : Finset ι) : mgf (∑ i ∈ s, X i) μ t = ∏ i ∈ s, mgf (X i) μ t := by
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X : ι → Ω → Real
    h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
    h_meas : ∀ (i : ι), Measurable (X i)
    s : Finset ι
    ⊢ Eq (ProbabilityTheory.mgf (s.sum fun i => X i) μ t) (s.prod fun i => Probabi …
  -/
  have : IsProbabilityMeasure μ := h_indep.isProbabilityMeasure
  classical
  induction' s using Finset.induction_on with i s hi_notin_s h_rec h_int
  · simp only [sum_empty, mgf_zero_fun, measure_univ, ENNReal.one_toReal, prod_empty]
  · have h_int' : ∀ i : ι, AEStronglyMeasurable (fun ω : Ω => exp (t * X i ω)) μ := fun i =>
      ((h_meas i).const_mul t).exp.aestronglyMeasurable
    rw [sum_insert hi_notin_s,
      IndepFun.mgf_add (h_indep.indepFun_finset_sum_of_not_mem h_meas hi_notin_s).symm (h_int' i)
        (aestronglyMeasurable_exp_mul_sum fun i _ => h_int' i),
      h_rec, prod_insert hi_notin_s]


theorem iIndepFun.cgf_sum {X : ι → Ω → ℝ}
    (h_indep : iIndepFun (fun _ => inferInstance) X μ) (h_meas : ∀ i, Measurable (X i))
    {s : Finset ι} (h_int : ∀ i ∈ s, Integrable (fun ω => exp (t * X i ω)) μ) :
    cgf (∑ i ∈ s, X i) μ t = ∑ i ∈ s, cgf (X i) μ t := by
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X : ι → Ω → Real
    h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
    h_meas : ∀ (i : ι), Measurable (X i)
    s : Finset ι
    h_int : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun ω => Rea …
    ⊢ Eq (ProbabilityTheory.cgf (s.sum fun i => X i) μ t) (s.sum fun i => Probabil …
  -/
  have : IsProbabilityMeasure μ := h_indep.isProbabilityMeasure
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X : ι → Ω → Real
    h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
    h_meas : ∀ (i : ι), Measurable (X i)
    s : Finset ι
    h_int : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun ω => Rea …
    this : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.cgf (s.sum fun i => X i) μ t) (s.sum fun i => Probabil …
  -/
  simp_rw [cgf]
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    t : Real
    X : ι → Ω → Real
    h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
    h_meas : ∀ (i : ι), Measurable (X i)
    s : Finset ι
    h_int : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun ω => Rea …
    this : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (Real.log (ProbabilityTheory.mgf (s.sum fun i => X i) μ t)) (s.sum fun x  …
  -/
  rw [← log_prod _ _ fun j hj => ?_]
    /-
      Ω : Type u_1
      ι : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      X : ι → Ω → Real
      h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
      h_meas : ∀ (i : ι), Measurable (X i)
      s : Finset ι
      h_int : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun ω => Rea …
      this : MeasureTheory.IsProbabilityMeasure μ
      ⊢ Eq (Real.log (ProbabilityTheory.mgf (s.sum fun i => X i) μ t)) (Real.log (s. …
    -/
  · rw [h_indep.mgf_sum h_meas]
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      t : Real
      X : ι → Ω → Real
      h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
      h_meas : ∀ (i : ι), Measurable (X i)
      s : Finset ι
      h_int : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun ω => Rea …
      this : MeasureTheory.IsProbabilityMeasure μ
      j : ι
      hj : Membership.mem s j
      ⊢ Ne (ProbabilityTheory.mgf (X j) μ t) 0
    -/
  · exact (mgf_pos (h_int j hj)).ne'
    /-
      🎉 no goals
    -/


theorem mgf_congr_of_identDistrib
    (X : Ω → ℝ) {Ω' : Type*} {m' : MeasurableSpace Ω'} {μ' : Measure Ω'} (X' : Ω' → ℝ)
    (hident : IdentDistrib X X' μ μ') (t : ℝ) :
    mgf X μ t = mgf X' μ' t := hident.comp (measurable_const_mul t).exp |>.integral_eq


theorem mgf_sum_of_identDistrib
    {X : ι → Ω → ℝ}
    {s : Finset ι} {j : ι}
    (h_meas : ∀ i, Measurable (X i))
    (h_indep : iIndepFun (fun _ => inferInstance) X μ)
    (hident : ∀ i ∈ s, ∀ j ∈ s, IdentDistrib (X i) (X j) μ μ)
    (hj : j ∈ s) (t : ℝ) : mgf (∑ i ∈ s, X i) μ t = mgf (X j) μ t ^ #s := by
  /-
    Ω : Type u_1
    ι : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : ι → Ω → Real
    s : Finset ι
    j : ι
    h_meas : ∀ (i : ι), Measurable (X i)
    h_indep : ProbabilityTheory.iIndepFun (fun x => inferInstance) X μ
    hident : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Proba …
    hj : Membership.mem s j
    t : Real
    ⊢ Eq (ProbabilityTheory.mgf (s.sum fun i => X i) μ t) (HPow.hPow (ProbabilityT …
  -/
  rw [h_indep.mgf_sum h_meas]
  exact Finset.prod_eq_pow_card fun i hi =>
    mgf_congr_of_identDistrib (X i) (X j) (hident i hi j hj) t


/-- **Chernoff bound** on the upper tail of a real random variable. -/
theorem measure_ge_le_exp_mul_mgf [IsFiniteMeasure μ] (ε : ℝ) (ht : 0 ≤ t)
    (h_int : Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | ε ≤ X ω}).toReal ≤ exp (-t * ε) * mgf X μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le 0 t
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (μ (setOf fun ω => LE.le ε (X ω))).toReal (HMul.hMul (Real.exp (HMul.h …
  -/
  rcases ht.eq_or_lt with ht_zero_eq | ht_pos
    /-
      case inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le 0 t
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ht_zero_eq : Eq 0 t
      ⊢ LE.le (μ (setOf fun ω => LE.le ε (X ω))).toReal (HMul.hMul (Real.exp (HMul.h …
    -/
  · rw [ht_zero_eq.symm]
    /-
      case inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le 0 t
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ht_zero_eq : Eq 0 t
      ⊢ LE.le (μ (setOf fun ω => LE.le ε (X ω))).toReal (HMul.hMul (Real.exp (HMul.h …
    -/
    simp only [neg_zero, zero_mul, exp_zero, mgf_zero', one_mul]
    /-
      case inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le 0 t
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ht_zero_eq : Eq 0 t
      ⊢ LE.le (μ (setOf fun ω => LE.le ε (X ω))).toReal (μ Set.univ).toReal
    -/
    gcongr
    /-
      case inl.hb
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le 0 t
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ht_zero_eq : Eq 0 t
      ⊢ Ne (μ Set.univ) Top.top
    -/
    exacts [measure_ne_top _ _, Set.subset_univ _]
    /-
      🎉 no goals
    -/
  calc
    (μ {ω | ε ≤ X ω}).toReal = (μ {ω | exp (t * ε) ≤ exp (t * X ω)}).toReal := by
      congr with ω
      simp only [Set.mem_setOf_eq, exp_le_exp, gt_iff_lt]
      exact ⟨fun h => mul_le_mul_of_nonneg_left h ht_pos.le,
        fun h => le_of_mul_le_mul_left h ht_pos⟩
    _ ≤ (exp (t * ε))⁻¹ * μ[fun ω => exp (t * X ω)] := by
      have : exp (t * ε) * (μ {ω | exp (t * ε) ≤ exp (t * X ω)}).toReal ≤
          μ[fun ω => exp (t * X ω)] :=
        mul_meas_ge_le_integral_of_nonneg (ae_of_all _ fun x => (exp_pos _).le) h_int _
      rwa [mul_comm (exp (t * ε))⁻¹, ← div_eq_mul_inv, le_div_iff₀' (exp_pos _)]
    _ = exp (-t * ε) * mgf X μ t := by rw [neg_mul, exp_neg]; rfl


/-- **Chernoff bound** on the lower tail of a real random variable. -/
theorem measure_le_le_exp_mul_mgf [IsFiniteMeasure μ] (ε : ℝ) (ht : t ≤ 0)
    (h_int : Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | X ω ≤ ε}).toReal ≤ exp (-t * ε) * mgf X μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le t 0
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (μ (setOf fun ω => LE.le (X ω) ε)).toReal (HMul.hMul (Real.exp (HMul.h …
  -/
  rw [← neg_neg t, ← mgf_neg, neg_neg, ← neg_mul_neg (-t)]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le t 0
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (μ (setOf fun ω => LE.le (X ω) ε)).toReal (HMul.hMul (Real.exp (HMul.h …
  -/
  refine Eq.trans_le ?_ (measure_ge_le_exp_mul_mgf (-ε) (neg_nonneg.mpr ht) ?_)
    /-
      case refine_1
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le t 0
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ⊢ Eq (μ (setOf fun ω => LE.le (X ω) ε)).toReal (μ (setOf fun ω => LE.le (Neg.n …
    -/
  · congr with ω
    /-
      case refine_1.e_a.h.e_6.h.h
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le t 0
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ω : Ω
      ⊢ Iff (Membership.mem (setOf fun ω => LE.le (X ω) ε) ω) (Membership.mem (setOf …
    -/
    simp only [Pi.neg_apply, neg_le_neg_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le t 0
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ⊢ MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul (Neg.neg t) (Neg.neg  …
    -/
  · simp_rw [Pi.neg_apply, neg_mul_neg]
    /-
      case refine_2
      Ω : Type u_1
      m : MeasurableSpace Ω
      X : Ω → Real
      μ : MeasureTheory.Measure Ω
      t : Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ε : Real
      ht : LE.le t 0
      h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
      ⊢ MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    -/
    exact h_int
    /-
      🎉 no goals
    -/


/-- **Chernoff bound** on the upper tail of a real random variable. -/
theorem measure_ge_le_exp_cgf [IsFiniteMeasure μ] (ε : ℝ) (ht : 0 ≤ t)
    (h_int : Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | ε ≤ X ω}).toReal ≤ exp (-t * ε + cgf X μ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le 0 t
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (μ (setOf fun ω => LE.le ε (X ω))).toReal (Real.exp (HAdd.hAdd (HMul.h …
  -/
  refine (measure_ge_le_exp_mul_mgf ε ht h_int).trans ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le 0 t
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul (Neg.neg t) ε)) (ProbabilityTheory.mgf …
  -/
  rw [exp_add]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le 0 t
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul (Neg.neg t) ε)) (ProbabilityTheory.mgf …
  -/
  exact mul_le_mul le_rfl (le_exp_log _) mgf_nonneg (exp_pos _).le
  /-
    🎉 no goals
  -/


/-- **Chernoff bound** on the lower tail of a real random variable. -/
theorem measure_le_le_exp_cgf [IsFiniteMeasure μ] (ε : ℝ) (ht : t ≤ 0)
    (h_int : Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | X ω ≤ ε}).toReal ≤ exp (-t * ε + cgf X μ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le t 0
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (μ (setOf fun ω => LE.le (X ω) ε)).toReal (Real.exp (HAdd.hAdd (HMul.h …
  -/
  refine (measure_le_le_exp_mul_mgf ε ht h_int).trans ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le t 0
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul (Neg.neg t) ε)) (ProbabilityTheory.mgf …
  -/
  rw [exp_add]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    t : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ε : Real
    ht : LE.le t 0
    h_int : MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
    ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul (Neg.neg t) ε)) (ProbabilityTheory.mgf …
  -/
  exact mul_le_mul le_rfl (le_exp_log _) mgf_nonneg (exp_pos _).le
  /-
    🎉 no goals
  -/


lemma mgf_dirac {x : ℝ} (hX : μ.map X = .dirac x) (t : ℝ) : mgf X μ t = exp (x * t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    X : Ω → Real
    μ : MeasureTheory.Measure Ω
    x : Real
    hX : Eq (MeasureTheory.Measure.map X μ) (MeasureTheory.Measure.dirac x)
    t : Real
    ⊢ Eq (ProbabilityTheory.mgf X μ t) (Real.exp (HMul.hMul x t))
  -/
  have : IsProbabilityMeasure (μ.map X) := by rw [hX]; infer_instance
  rw [← mgf_id_map (.of_map_ne_zero <| IsProbabilityMeasure.ne_zero _), mgf, hX, integral_dirac,
    mul_comm, id_def]


lemma aemeasurable_exp_mul {X : Ω → ℝ} (t : ℝ) (hX : AEMeasurable X μ) :
    AEStronglyMeasurable (fun ω ↦ rexp (t * X ω)) μ :=
  (measurable_exp.comp_aemeasurable (hX.const_mul t)).aestronglyMeasurable


lemma integrable_exp_mul_of_le [IsFiniteMeasure μ] {X : Ω → ℝ} (t b : ℝ) (ht : 0 ≤ t)
    (hX : AEMeasurable X μ) (hb : ∀ᵐ ω ∂μ, X ω ≤ b) :
    Integrable (fun ω ↦ exp (t * X ω)) μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : Ω → Real
    t b : Real
    ht : LE.le 0 t
    hX : AEMeasurable X μ
    hb : Filter.Eventually (fun ω => LE.le (X ω) b) (MeasureTheory.ae μ)
    ⊢ MeasureTheory.Integrable (fun ω => Real.exp (HMul.hMul t (X ω))) μ
  -/
  refine .of_mem_Icc 0 (rexp (t * b)) (measurable_exp.comp_aemeasurable (hX.const_mul t)) ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : Ω → Real
    t b : Real
    ht : LE.le 0 t
    hX : AEMeasurable X μ
    hb : Filter.Eventually (fun ω => LE.le (X ω) b) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun ω => Membership.mem (Set.Icc 0 (Real.exp (HMul.hMul t …
  -/
  filter_upwards [hb] with ω hb
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : Ω → Real
    t b : Real
    ht : LE.le 0 t
    hX : AEMeasurable X μ
    hb✝ : Filter.Eventually (fun ω => LE.le (X ω) b) (MeasureTheory.ae μ)
    ω : Ω
    hb : LE.le (X ω) b
    ⊢ Membership.mem (Set.Icc 0 (Real.exp (HMul.hMul t b))) (Real.exp (HMul.hMul t …
  -/
  exact ⟨by positivity, by gcongr⟩
  /-
    🎉 no goals
  -/


