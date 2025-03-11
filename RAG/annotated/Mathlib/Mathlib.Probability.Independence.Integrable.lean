/-- If a nonzero function belongs to `ℒ^p` and is independent of another function, then
the space is a probability space. -/
lemma Memℒp.isProbabilityMeasure_of_indepFun
    (f : Ω → E) (g : Ω → F) {p : ℝ≥0∞} (hp : p ≠ 0) (hp' : p ≠ ∞)
    (hℒp : Memℒp f p μ) (h'f : ¬ (∀ᵐ ω ∂μ, f ω = 0)) (hindep : IndepFun f g μ) :
    IsProbabilityMeasure μ := by
  obtain ⟨c, c_pos, hc⟩ : ∃ (c : ℝ≥0), 0 < c ∧ 0 < μ {ω | c ≤ ‖f ω‖₊} := by
    contrapose! h'f
    have A (c : ℝ≥0) (hc : 0 < c) : ∀ᵐ ω ∂μ, ‖f ω‖₊ < c := by simpa [ae_iff] using h'f c hc
    obtain ⟨u, -, u_pos, u_lim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), 0 < u n)
      ∧ Tendsto u atTop (𝓝 0) := exists_seq_strictAnti_tendsto (0 : ℝ≥0)
    filter_upwards [ae_all_iff.2 (fun n ↦ A (u n) (u_pos n))] with ω hω
    simpa using ge_of_tendsto' u_lim (fun i ↦ (hω i).le)
  /-
    case intro.intro
    Ω : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : MeasurableSpace F
    f : Ω → E
    g : Ω → F
    p : ENNReal
    hp : Ne p 0
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp f p μ
    h'f : Not (Filter.Eventually (fun ω => Eq (f ω) 0) (MeasureTheory.ae μ))
    hindep : ProbabilityTheory.IndepFun f g μ
    c : NNReal
    c_pos : LT.lt 0 c
    hc : LT.lt 0 (μ (setOf fun ω => LE.le c (NNNorm.nnnorm (f ω))))
    ⊢ MeasureTheory.IsProbabilityMeasure μ
  -/
  have h'c : μ {ω | c ≤ ‖f ω‖₊} < ∞ := hℒp.meas_ge_lt_top hp hp' c_pos.ne'
  have := hindep.measure_inter_preimage_eq_mul {x | c ≤ ‖x‖₊} Set.univ
    (isClosed_le continuous_const continuous_nnnorm).measurableSet MeasurableSet.univ
  /-
    case intro.intro
    Ω : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : MeasurableSpace F
    f : Ω → E
    g : Ω → F
    p : ENNReal
    hp : Ne p 0
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp f p μ
    h'f : Not (Filter.Eventually (fun ω => Eq (f ω) 0) (MeasureTheory.ae μ))
    hindep : ProbabilityTheory.IndepFun f g μ
    c : NNReal
    c_pos : LT.lt 0 c
    hc : LT.lt 0 (μ (setOf fun ω => LE.le c (NNNorm.nnnorm (f ω))))
    h'c : LT.lt (μ (setOf fun ω => LE.le c (NNNorm.nnnorm (f ω)))) Top.top
    this : Eq (μ (Inter.inter (Set.preimage f (setOf fun x => LE.le c (NNNorm.nnno …
    ⊢ MeasureTheory.IsProbabilityMeasure μ
  -/
  simp only [Set.preimage_setOf_eq, Set.preimage_univ, Set.inter_univ] at this
  /-
    case intro.intro
    Ω : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : MeasurableSpace F
    f : Ω → E
    g : Ω → F
    p : ENNReal
    hp : Ne p 0
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp f p μ
    h'f : Not (Filter.Eventually (fun ω => Eq (f ω) 0) (MeasureTheory.ae μ))
    hindep : ProbabilityTheory.IndepFun f g μ
    c : NNReal
    c_pos : LT.lt 0 c
    hc : LT.lt 0 (μ (setOf fun ω => LE.le c (NNNorm.nnnorm (f ω))))
    h'c : LT.lt (μ (setOf fun ω => LE.le c (NNNorm.nnnorm (f ω)))) Top.top
    this : Eq (μ (setOf fun a => LE.le c (NNNorm.nnnorm (f a)))) (HMul.hMul (μ (se …
    ⊢ MeasureTheory.IsProbabilityMeasure μ
  -/
  exact ⟨(ENNReal.mul_eq_left hc.ne' h'c.ne).1 this.symm⟩
  /-
    🎉 no goals
  -/


/-- If a nonzero function is integrable and is independent of another function, then
the space is a probability space. -/
lemma Integrable.isProbabilityMeasure_of_indepFun (f : Ω → E) (g : Ω → F)
    (hf : Integrable f μ) (h'f : ¬ (∀ᵐ ω ∂μ, f ω = 0)) (hindep : IndepFun f g μ) :
    IsProbabilityMeasure μ :=
  Memℒp.isProbabilityMeasure_of_indepFun f g one_ne_zero ENNReal.one_ne_top
    (memℒp_one_iff_integrable.mpr hf) h'f hindep


