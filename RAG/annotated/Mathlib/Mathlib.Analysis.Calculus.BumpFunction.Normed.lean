/-- A bump function normed so that `∫ x, f.normed μ x ∂μ = 1`. -/
protected def normed (μ : Measure E) : E → ℝ := fun x => f x / ∫ x, f x ∂μ


theorem normed_def {μ : Measure E} (x : E) : f.normed μ x = f x / ∫ x, f x ∂μ :=
  rfl


theorem nonneg_normed (x : E) : 0 ≤ f.normed μ x :=
  div_nonneg f.nonneg <| integral_nonneg f.nonneg'


theorem contDiff_normed {n : ℕ∞} : ContDiff ℝ n (f.normed μ) :=
  f.contDiff.div_const _


theorem continuous_normed : Continuous (f.normed μ) :=
  f.continuous.div_const _


theorem normed_sub (x : E) : f.normed μ (c - x) = f.normed μ (c + x) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : HasContDiffBump E
    inst✝ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    x : E
    ⊢ Eq (f.normed μ (HSub.hSub c x)) (f.normed μ (HAdd.hAdd c x))
  -/
  simp_rw [f.normed_def, f.sub]
  /-
    🎉 no goals
  -/


theorem normed_neg (f : ContDiffBump (0 : E)) (x : E) : f.normed μ (-x) = f.normed μ x := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : HasContDiffBump E
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    f : ContDiffBump 0
    x : E
    ⊢ Eq (f.normed μ (Neg.neg x)) (f.normed μ x)
  -/
  simp_rw [f.normed_def, f.neg]
  /-
    🎉 no goals
  -/


protected theorem integrable : Integrable f μ :=
  f.continuous.integrable_of_hasCompactSupport f.hasCompactSupport


protected theorem integrable_normed : Integrable (f.normed μ) μ :=
  f.integrable.div_const _


theorem integral_pos : 0 < ∫ x, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ LT.lt 0 (MeasureTheory.integral μ fun x => ↑f x)
  -/
  refine (integral_pos_iff_support_of_nonneg f.nonneg' f.integrable).mpr ?_
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ LT.lt 0 (μ (Function.support ↑f))
  -/
  rw [f.support_eq]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ LT.lt 0 (μ (Metric.ball c f.rOut))
  -/
  exact measure_ball_pos μ c f.rOut_pos
  /-
    🎉 no goals
  -/


theorem integral_normed : ∫ x, f.normed μ x ∂μ = 1 := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ Eq (MeasureTheory.integral μ fun x => f.normed μ x) 1
  -/
  simp_rw [ContDiffBump.normed, div_eq_mul_inv, mul_comm (f _), ← smul_eq_mul, integral_smul]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ Eq (HSMul.hSMul (Inv.inv (MeasureTheory.integral μ fun x => ↑f x)) (MeasureT …
  -/
  exact inv_mul_cancel₀ f.integral_pos.ne'
  /-
    🎉 no goals
  -/


theorem support_normed_eq : Function.support (f.normed μ) = Metric.ball c f.rOut := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ Eq (Function.support (f.normed μ)) (Metric.ball c f.rOut)
  -/
  unfold ContDiffBump.normed
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ Eq (Function.support fun x => HDiv.hDiv (↑f x) (MeasureTheory.integral μ fun …
  -/
  rw [support_div, f.support_eq, support_const f.integral_pos.ne', inter_univ]
  /-
    🎉 no goals
  -/


theorem tsupport_normed_eq : tsupport (f.normed μ) = Metric.closedBall c f.rOut := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ Eq (tsupport (f.normed μ)) (Metric.closedBall c f.rOut)
  -/
  rw [tsupport, f.support_normed_eq, closure_ball _ f.rOut_pos.ne']
  /-
    🎉 no goals
  -/


theorem hasCompactSupport_normed : HasCompactSupport (f.normed μ) := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ⊢ HasCompactSupport (f.normed μ)
  -/
  simp only [HasCompactSupport, f.tsupport_normed_eq (μ := μ), isCompact_closedBall]
  /-
    🎉 no goals
  -/


theorem tendsto_support_normed_smallSets {ι} {φ : ι → ContDiffBump c} {l : Filter ι}
    (hφ : Tendsto (fun i => (φ i).rOut) l (𝓝 0)) :
    Tendsto (fun i => Function.support fun x => (φ i).normed μ x) l (𝓝 c).smallSets := by
  simp_rw [NormedAddCommGroup.tendsto_nhds_zero, Real.norm_eq_abs,
    abs_eq_self.mpr (φ _).rOut_pos.le] at hφ
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ι : Type u_2
    φ : ι → ContDiffBump c
    l : Filter ι
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (φ x).rOut ε) l
    ⊢ Filter.Tendsto (fun i => Function.support fun x => (φ i).normed μ x) l (nhds …
  -/
  rw [nhds_basis_ball.smallSets.tendsto_right_iff]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ι : Type u_2
    φ : ι → ContDiffBump c
    l : Filter ι
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (φ x).rOut ε) l
    ⊢ ∀ (i : Real), LT.lt 0 i → Filter.Eventually (fun x => Membership.mem (Metric …
  -/
  refine fun ε hε ↦ (hφ ε hε).mono fun i hi ↦ ?_
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ι : Type u_2
    φ : ι → ContDiffBump c
    l : Filter ι
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (φ x).rOut ε) l
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    hi : LT.lt (φ i).rOut ε
    ⊢ Membership.mem (Metric.ball c ε).powerset (Function.support fun x => (φ i).n …
  -/
  rw [(φ i).support_normed_eq]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    ι : Type u_2
    φ : ι → ContDiffBump c
    l : Filter ι
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (φ x).rOut ε) l
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    hi : LT.lt (φ i).rOut ε
    ⊢ Membership.mem (Metric.ball c ε).powerset (Metric.ball c (φ i).rOut)
  -/
  exact ball_subset_ball hi.le
  /-
    🎉 no goals
  -/


theorem integral_normed_smul {X} [NormedAddCommGroup X] [NormedSpace ℝ X]
    [CompleteSpace X] (z : X) : ∫ x, f.normed μ x • z ∂μ = z := by
  /-
    E : Type u_1
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace Real E
    inst✝⁸ : HasContDiffBump E
    inst✝⁷ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝⁶ : BorelSpace E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝³ : μ.IsOpenPosMeasure
    X : Type u_2
    inst✝² : NormedAddCommGroup X
    inst✝¹ : NormedSpace Real X
    inst✝ : CompleteSpace X
    z : X
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f.normed μ x) z) z
  -/
  simp_rw [integral_smul_const, f.integral_normed (μ := μ), one_smul]
  /-
    🎉 no goals
  -/


theorem measure_closedBall_le_integral : (μ (closedBall c f.rIn)).toReal ≤ ∫ x, f x ∂μ := by calc
  (μ (closedBall c f.rIn)).toReal = ∫ x in closedBall c f.rIn, 1 ∂μ := by simp
  _ = ∫ x in closedBall c f.rIn, f x ∂μ := setIntegral_congr_fun measurableSet_closedBall
        (fun x hx ↦ (one_of_mem_closedBall f hx).symm)
  _ ≤ ∫ x, f x ∂μ := setIntegral_le_integral f.integrable (Eventually.of_forall (fun x ↦ f.nonneg))


theorem normed_le_div_measure_closedBall_rIn [μ.IsOpenPosMeasure] (x : E) :
    f.normed μ x ≤ 1 / (μ (closedBall c f.rIn)).toReal := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    x : E
    ⊢ LE.le (f.normed μ x) (HDiv.hDiv 1 (μ (Metric.closedBall c f.rIn)).toReal)
  -/
  rw [normed_def]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    x : E
    ⊢ LE.le (HDiv.hDiv (↑f x) (MeasureTheory.integral μ fun x => ↑f x)) (HDiv.hDiv …
  -/
  gcongr
    /-
      case hd
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : HasContDiffBump E
      inst✝⁴ : MeasurableSpace E
      c : E
      f : ContDiffBump c
      μ : MeasureTheory.Measure E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : μ.IsOpenPosMeasure
      x : E
      ⊢ LT.lt 0 (μ (Metric.closedBall c f.rIn)).toReal
    -/
  · exact ENNReal.toReal_pos (measure_closedBall_pos _ _ f.rIn_pos).ne' measure_closedBall_lt_top.ne
    /-
      🎉 no goals
    -/
    /-
      case hac
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : HasContDiffBump E
      inst✝⁴ : MeasurableSpace E
      c : E
      f : ContDiffBump c
      μ : MeasureTheory.Measure E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : μ.IsOpenPosMeasure
      x : E
      ⊢ LE.le (↑f x) 1
    -/
  · exact f.le_one
    /-
      🎉 no goals
    -/
    /-
      case hdb
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : HasContDiffBump E
      inst✝⁴ : MeasurableSpace E
      c : E
      f : ContDiffBump c
      μ : MeasureTheory.Measure E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : μ.IsOpenPosMeasure
      x : E
      ⊢ LE.le (μ (Metric.closedBall c f.rIn)).toReal (MeasureTheory.integral μ fun x …
    -/
  · exact f.measure_closedBall_le_integral μ
    /-
      🎉 no goals
    -/


theorem integral_le_measure_closedBall : ∫ x, f x ∂μ ≤ (μ (closedBall c f.rOut)).toReal := by calc
  ∫ x, f x ∂μ = ∫ x in closedBall c f.rOut, f x ∂μ := by
    apply (setIntegral_eq_integral_of_forall_compl_eq_zero (fun x hx ↦ ?_)).symm
    apply f.zero_of_le_dist (le_of_lt _)
    simpa using hx
  _ ≤ ∫ x in closedBall c f.rOut, 1 ∂μ := by
    apply setIntegral_mono f.integrable.integrableOn _ (fun x ↦ f.le_one)
    simp [measure_closedBall_lt_top]
  _ = (μ (closedBall c f.rOut)).toReal := by simp


theorem measure_closedBall_div_le_integral [IsAddHaarMeasure μ] (K : ℝ) (h : f.rOut ≤ K * f.rIn) :
    (μ (closedBall c f.rOut)).toReal / K ^ finrank ℝ E ≤ ∫ x, f x ∂μ := by
  have K_pos : 0 < K := by
    simpa [f.rIn_pos, not_lt.2 f.rIn_pos.le] using mul_pos_iff.1 (f.rOut_pos.trans_le h)
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsAddHaarMeasure
    K : Real
    h : LE.le f.rOut (HMul.hMul K f.rIn)
    K_pos : LT.lt 0 K
    ⊢ LE.le (HDiv.hDiv (μ (Metric.closedBall c f.rOut)).toReal (HPow.hPow K (Modul …
  -/
  apply le_trans _ (f.measure_closedBall_le_integral μ)
  rw [div_le_iff₀ (pow_pos K_pos _), addHaar_closedBall' _ _ f.rIn_pos.le,
    addHaar_closedBall' _ _ f.rOut_pos.le, ENNReal.toReal_mul, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (pow_nonneg f.rOut_pos.le _),
    ENNReal.toReal_ofReal (pow_nonneg f.rIn_pos.le _), mul_assoc, mul_comm _ (K ^ _), ← mul_assoc,
    ← mul_pow, mul_comm _ K]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsAddHaarMeasure
    K : Real
    h : LE.le f.rOut (HMul.hMul K f.rIn)
    K_pos : LT.lt 0 K
    ⊢ LE.le (HMul.hMul (HPow.hPow f.rOut (Module.finrank Real E)) (μ (Metric.close …
  -/
  gcongr
  /-
    case h.ha
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsAddHaarMeasure
    K : Real
    h : LE.le f.rOut (HMul.hMul K f.rIn)
    K_pos : LT.lt 0 K
    ⊢ LE.le 0 f.rOut
  -/
  exact f.rOut_pos.le
  /-
    🎉 no goals
  -/


theorem normed_le_div_measure_closedBall_rOut [IsAddHaarMeasure μ] (K : ℝ) (h : f.rOut ≤ K * f.rIn)
    (x : E) :
    f.normed μ x ≤ K ^ finrank ℝ E / (μ (closedBall c f.rOut)).toReal := by
  have K_pos : 0 < K := by
    simpa [f.rIn_pos, not_lt.2 f.rIn_pos.le] using mul_pos_iff.1 (f.rOut_pos.trans_le h)
  have : f x / ∫ y, f y ∂μ ≤ 1 / ∫ y, f y ∂μ := by
    gcongr
    · exact f.integral_pos.le
    · exact f.le_one
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsAddHaarMeasure
    K : Real
    h : LE.le f.rOut (HMul.hMul K f.rIn)
    x : E
    K_pos : LT.lt 0 K
    this : LE.le (HDiv.hDiv (↑f x) (MeasureTheory.integral μ fun y => ↑f y)) (HDiv …
    ⊢ LE.le (f.normed μ x) (HDiv.hDiv (HPow.hPow K (Module.finrank Real E)) (μ (Me …
  -/
  apply this.trans
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : HasContDiffBump E
    inst✝⁴ : MeasurableSpace E
    c : E
    f : ContDiffBump c
    μ : MeasureTheory.Measure E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsAddHaarMeasure
    K : Real
    h : LE.le f.rOut (HMul.hMul K f.rIn)
    x : E
    K_pos : LT.lt 0 K
    this : LE.le (HDiv.hDiv (↑f x) (MeasureTheory.integral μ fun y => ↑f y)) (HDiv …
    ⊢ LE.le (HDiv.hDiv 1 (MeasureTheory.integral μ fun y => ↑f y)) (HDiv.hDiv (HPo …
  -/
  rw [div_le_div_iff₀ f.integral_pos, one_mul, ← div_le_iff₀' (pow_pos K_pos _)]
    /-
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : HasContDiffBump E
      inst✝⁴ : MeasurableSpace E
      c : E
      f : ContDiffBump c
      μ : MeasureTheory.Measure E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝ : μ.IsAddHaarMeasure
      K : Real
      h : LE.le f.rOut (HMul.hMul K f.rIn)
      x : E
      K_pos : LT.lt 0 K
      this : LE.le (HDiv.hDiv (↑f x) (MeasureTheory.integral μ fun y => ↑f y)) (HDiv …
      ⊢ LE.le (HDiv.hDiv (μ (Metric.closedBall c f.rOut)).toReal (HPow.hPow K (Modul …
    -/
  · exact f.measure_closedBall_div_le_integral μ K h
    /-
      🎉 no goals
    -/
  · exact ENNReal.toReal_pos (measure_closedBall_pos _ _ f.rOut_pos).ne'
      measure_closedBall_lt_top.ne


