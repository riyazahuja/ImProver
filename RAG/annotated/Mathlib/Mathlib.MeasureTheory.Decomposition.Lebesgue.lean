/-- A pair of measures `μ` and `ν` is said to `HaveLebesgueDecomposition` if there exists a
measure `ξ` and a measurable function `f`, such that `ξ` is mutually singular with respect to
`ν` and `μ = ξ + ν.withDensity f`. -/
class HaveLebesgueDecomposition (μ ν : Measure α) : Prop where
  lebesgue_decomposition :
    ∃ p : Measure α × (α → ℝ≥0∞), Measurable p.2 ∧ p.1 ⟂ₘ ν ∧ μ = p.1 + ν.withDensity p.2


open Classical in
/-- If a pair of measures `HaveLebesgueDecomposition`, then `singularPart` chooses the
measure from `HaveLebesgueDecomposition`, otherwise it returns the zero measure. For sigma-finite
measures, `μ = μ.singularPart ν + ν.withDensity (μ.rnDeriv ν)`. -/
noncomputable irreducible_def singularPart (μ ν : Measure α) : Measure α :=
  if h : HaveLebesgueDecomposition μ ν then (Classical.choose h.lebesgue_decomposition).1 else 0


open Classical in
/-- If a pair of measures `HaveLebesgueDecomposition`, then `rnDeriv` chooses the
measurable function from `HaveLebesgueDecomposition`, otherwise it returns the zero function.
For sigma-finite measures, `μ = μ.singularPart ν + ν.withDensity (μ.rnDeriv ν)`. -/
noncomputable irreducible_def rnDeriv (μ ν : Measure α) : α → ℝ≥0∞ :=
  if h : HaveLebesgueDecomposition μ ν then (Classical.choose h.lebesgue_decomposition).2 else 0


theorem haveLebesgueDecomposition_spec (μ ν : Measure α) [h : HaveLebesgueDecomposition μ ν] :
    Measurable (μ.rnDeriv ν) ∧
      μ.singularPart ν ⟂ₘ ν ∧ μ = μ.singularPart ν + ν.withDensity (μ.rnDeriv ν) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.HaveLebesgueDecomposition ν
    ⊢ And (Measurable (μ.rnDeriv ν)) (And ((μ.singularPart ν).MutuallySingular ν)  …
  -/
  rw [singularPart, rnDeriv, dif_pos h, dif_pos h]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.HaveLebesgueDecomposition ν
    ⊢ And (Measurable (Classical.choose ⋯).2) (And ((Classical.choose ⋯).1.Mutuall …
  -/
  exact Classical.choose_spec h.lebesgue_decomposition
  /-
    🎉 no goals
  -/


lemma rnDeriv_of_not_haveLebesgueDecomposition (h : ¬ HaveLebesgueDecomposition μ ν) :
    μ.rnDeriv ν = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Not (μ.HaveLebesgueDecomposition ν)
    ⊢ Eq (μ.rnDeriv ν) 0
  -/
  rw [rnDeriv, dif_neg h]
  /-
    🎉 no goals
  -/


lemma singularPart_of_not_haveLebesgueDecomposition (h : ¬ HaveLebesgueDecomposition μ ν) :
    μ.singularPart ν = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Not (μ.HaveLebesgueDecomposition ν)
    ⊢ Eq (μ.singularPart ν) 0
  -/
  rw [singularPart, dif_neg h]
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem measurable_rnDeriv (μ ν : Measure α) : Measurable <| μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ Measurable (μ.rnDeriv ν)
  -/
  by_cases h : HaveLebesgueDecomposition μ ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.HaveLebesgueDecomposition ν
      ⊢ Measurable (μ.rnDeriv ν)
    -/
  · exact (haveLebesgueDecomposition_spec μ ν).1
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Measurable (μ.rnDeriv ν)
    -/
  · rw [rnDeriv_of_not_haveLebesgueDecomposition h]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Measurable 0
    -/
    exact measurable_zero
    /-
      🎉 no goals
    -/


theorem mutuallySingular_singularPart (μ ν : Measure α) : μ.singularPart ν ⟂ₘ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ (μ.singularPart ν).MutuallySingular ν
  -/
  by_cases h : HaveLebesgueDecomposition μ ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.HaveLebesgueDecomposition ν
      ⊢ (μ.singularPart ν).MutuallySingular ν
    -/
  · exact (haveLebesgueDecomposition_spec μ ν).2.1
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ (μ.singularPart ν).MutuallySingular ν
    -/
  · rw [singularPart_of_not_haveLebesgueDecomposition h]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ MeasureTheory.Measure.MutuallySingular 0 ν
    -/
    exact MutuallySingular.zero_left
    /-
      🎉 no goals
    -/


theorem haveLebesgueDecomposition_add (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] :
    μ = μ.singularPart ν + ν.withDensity (μ.rnDeriv ν) :=
  (haveLebesgueDecomposition_spec μ ν).2.2


/-- For the versions of this lemma where `ν.withDensity (μ.rnDeriv ν)` or `μ.singularPart ν` are
isolated, see `MeasureTheory.Measure.measure_sub_singularPart` and
`MeasureTheory.Measure.measure_sub_rnDeriv`. -/
lemma singularPart_add_rnDeriv (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] :
    μ.singularPart ν + ν.withDensity (μ.rnDeriv ν) = μ := (haveLebesgueDecomposition_add μ ν).symm


/-- For the versions of this lemma where `μ.singularPart ν` or `ν.withDensity (μ.rnDeriv ν)` are
isolated, see `MeasureTheory.Measure.measure_sub_singularPart` and
`MeasureTheory.Measure.measure_sub_rnDeriv`. -/
lemma rnDeriv_add_singularPart (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] :
                                                             /-
                                                               α : Type u_1
                                                               m : MeasurableSpace α
                                                               μ ν : MeasureTheory.Measure α
                                                               inst✝ : μ.HaveLebesgueDecomposition ν
                                                               ⊢ Eq (HAdd.hAdd (ν.withDensity (μ.rnDeriv ν)) (μ.singularPart ν)) μ
                                                             -/
    ν.withDensity (μ.rnDeriv ν) + μ.singularPart ν = μ := by rw [add_comm, singularPart_add_rnDeriv]
                                                             /-
                                                               🎉 no goals
                                                             -/


instance instHaveLebesgueDecompositionZeroLeft : HaveLebesgueDecomposition 0 ν where
                                                                                     /-
                                                                                       α : Type u_1
                                                                                       m : MeasurableSpace α
                                                                                       μ ν : MeasureTheory.Measure α
                                                                                       ⊢ Eq 0 (HAdd.hAdd { fst := 0, snd := 0 }.1 (ν.withDensity { fst := 0, snd := 0 …
                                                                                     -/
  lebesgue_decomposition := ⟨⟨0, 0⟩, measurable_zero, MutuallySingular.zero_left, by simp⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


instance instHaveLebesgueDecompositionZeroRight : HaveLebesgueDecomposition μ 0 where
                                                                                      /-
                                                                                        α : Type u_1
                                                                                        m : MeasurableSpace α
                                                                                        μ ν : MeasureTheory.Measure α
                                                                                        ⊢ Eq μ (HAdd.hAdd { fst := μ, snd := 0 }.1 (MeasureTheory.Measure.withDensity  …
                                                                                      -/
  lebesgue_decomposition := ⟨⟨μ, 0⟩, measurable_zero, MutuallySingular.zero_right, by simp⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance instHaveLebesgueDecompositionSelf : HaveLebesgueDecomposition μ μ where
                                                                                      /-
                                                                                        α : Type u_1
                                                                                        m : MeasurableSpace α
                                                                                        μ ν : MeasureTheory.Measure α
                                                                                        ⊢ Eq μ (HAdd.hAdd { fst := 0, snd := 1 }.1 (μ.withDensity { fst := 0, snd := 1 …
                                                                                      -/
  lebesgue_decomposition := ⟨⟨0, 1⟩, measurable_const, MutuallySingular.zero_left, by simp⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance HaveLebesgueDecomposition.sum_left {ι : Type*} [Countable ι] (μ : ι → Measure α)
    [∀ i, HaveLebesgueDecomposition (μ i) ν] : HaveLebesgueDecomposition (.sum μ) ν :=
  ⟨(.sum fun i ↦ (μ i).singularPart ν, ∑' i, rnDeriv (μ i) ν),
       /-
         α : Type u_1
         m : MeasurableSpace α
         μ✝ ν : MeasureTheory.Measure α
         ι : Type u_2
         inst✝¹ : Countable ι
         μ : ι → MeasureTheory.Measure α
         inst✝ : ∀ (i : ι), (μ i).HaveLebesgueDecomposition ν
         ⊢ Measurable { fst := MeasureTheory.Measure.sum fun i => (μ i).singularPart ν, …
       -/
                   /-
                     🎉 no goals
                   -/
    by dsimp only; fun_prop, by simp [mutuallySingular_singularPart], by
                                /-
                                  🎉 no goals
                                -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ✝ ν : MeasureTheory.Measure α
        ι : Type u_2
        inst✝¹ : Countable ι
        μ : ι → MeasureTheory.Measure α
        inst✝ : ∀ (i : ι), (μ i).HaveLebesgueDecomposition ν
        ⊢ Eq (MeasureTheory.Measure.sum μ) (HAdd.hAdd { fst := MeasureTheory.Measure.s …
      -/
      simp [withDensity_tsum, measurable_rnDeriv, Measure.sum_add_sum, singularPart_add_rnDeriv]⟩
      /-
        🎉 no goals
      -/


instance HaveLebesgueDecomposition.add_left {μ' : Measure α} [HaveLebesgueDecomposition μ ν]
    [HaveLebesgueDecomposition μ' ν] : HaveLebesgueDecomposition (μ + μ') ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν μ' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : μ'.HaveLebesgueDecomposition ν
    ⊢ (HAdd.hAdd μ μ').HaveLebesgueDecomposition ν
  -/
  have : ∀ b, HaveLebesgueDecomposition (cond b μ μ') ν := by simp [*]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν μ' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : μ'.HaveLebesgueDecomposition ν
    this : ∀ (b : Bool), (cond b μ μ').HaveLebesgueDecomposition ν
    ⊢ (HAdd.hAdd μ μ').HaveLebesgueDecomposition ν
  -/
  simpa using sum_left (cond · μ μ')
  /-
    🎉 no goals
  -/


instance haveLebesgueDecompositionSMul' (μ ν : Measure α) [HaveLebesgueDecomposition μ ν]
    (r : ℝ≥0∞) : (r • μ).HaveLebesgueDecomposition ν where
  lebesgue_decomposition := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : ENNReal
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq (HSMu …
    -/
    obtain ⟨hmeas, hsing, hadd⟩ := haveLebesgueDecomposition_spec μ ν
    /-
      case intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : ENNReal
      hmeas : Measurable (μ.rnDeriv ν)
      hsing : (μ.singularPart ν).MutuallySingular ν
      hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq (HSMu …
    -/
    refine ⟨⟨r • μ.singularPart ν, r • μ.rnDeriv ν⟩, hmeas.const_smul _, hsing.smul _, ?_⟩
    /-
      case intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : ENNReal
      hmeas : Measurable (μ.rnDeriv ν)
      hsing : (μ.singularPart ν).MutuallySingular ν
      hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      ⊢ Eq (HSMul.hSMul r μ) (HAdd.hAdd { fst := HSMul.hSMul r (μ.singularPart ν), s …
    -/
    simp only [ENNReal.smul_def]
    /-
      case intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : ENNReal
      hmeas : Measurable (μ.rnDeriv ν)
      hsing : (μ.singularPart ν).MutuallySingular ν
      hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      ⊢ Eq (HSMul.hSMul r μ) (HAdd.hAdd (HSMul.hSMul r (μ.singularPart ν)) (ν.withDe …
    -/
    rw [withDensity_smul _ hmeas, ← smul_add, ← hadd]
    /-
      🎉 no goals
    -/


instance haveLebesgueDecompositionSMul (μ ν : Measure α) [HaveLebesgueDecomposition μ ν]
    (r : ℝ≥0) : (r • μ).HaveLebesgueDecomposition ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ ν✝ μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    r : NNReal
    ⊢ (HSMul.hSMul r μ).HaveLebesgueDecomposition ν
  -/
  rw [ENNReal.smul_def]; infer_instance
                         /-
                           🎉 no goals
                         -/


instance haveLebesgueDecompositionSMulRight (μ ν : Measure α) [HaveLebesgueDecomposition μ ν]
    (r : ℝ≥0) :
    μ.HaveLebesgueDecomposition (r • ν) where
  lebesgue_decomposition := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : NNReal
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular (HSMul.hSMul …
    -/
    obtain ⟨hmeas, hsing, hadd⟩ := haveLebesgueDecomposition_spec μ ν
    /-
      case intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : NNReal
      hmeas : Measurable (μ.rnDeriv ν)
      hsing : (μ.singularPart ν).MutuallySingular ν
      hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular (HSMul.hSMul …
    -/
    by_cases hr : r = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ✝ ν✝ μ ν : MeasureTheory.Measure α
        inst✝ : μ.HaveLebesgueDecomposition ν
        r : NNReal
        hmeas : Measurable (μ.rnDeriv ν)
        hsing : (μ.singularPart ν).MutuallySingular ν
        hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
        hr : Eq r 0
        ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular (HSMul.hSMul …
      -/
    · exact ⟨⟨μ, 0⟩, measurable_const, by simp [hr], by simp⟩
      /-
        🎉 no goals
      -/
    refine ⟨⟨μ.singularPart ν, r⁻¹ • μ.rnDeriv ν⟩, hmeas.const_smul _,
      hsing.mono_ac AbsolutelyContinuous.rfl smul_absolutelyContinuous, ?_⟩
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ✝ ν✝ μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      r : NNReal
      hmeas : Measurable (μ.rnDeriv ν)
      hsing : (μ.singularPart ν).MutuallySingular ν
      hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      hr : Not (Eq r 0)
      ⊢ Eq μ (HAdd.hAdd { fst := μ.singularPart ν, snd := HSMul.hSMul (Inv.inv r) (μ …
    -/
    have : r⁻¹ • rnDeriv μ ν = ((r⁻¹ : ℝ≥0) : ℝ≥0∞) • rnDeriv μ ν := by simp [ENNReal.smul_def]
    rw [this, withDensity_smul _ hmeas, ENNReal.smul_def r, withDensity_smul_measure,
      ← smul_assoc, smul_eq_mul, ENNReal.coe_inv hr, ENNReal.inv_mul_cancel, one_smul]
      /-
        case neg
        α : Type u_1
        m : MeasurableSpace α
        μ✝ ν✝ μ ν : MeasureTheory.Measure α
        inst✝ : μ.HaveLebesgueDecomposition ν
        r : NNReal
        hmeas : Measurable (μ.rnDeriv ν)
        hsing : (μ.singularPart ν).MutuallySingular ν
        hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
        hr : Not (Eq r 0)
        this : Eq (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν)) (HSMul.hSMul (↑(Inv.inv r))  …
        ⊢ Eq μ (HAdd.hAdd { fst := μ.singularPart ν, snd := HSMul.hSMul (Inv.inv ↑r) ( …
      -/
    · exact hadd
      /-
        🎉 no goals
      -/
      /-
        case neg.h0
        α : Type u_1
        m : MeasurableSpace α
        μ✝ ν✝ μ ν : MeasureTheory.Measure α
        inst✝ : μ.HaveLebesgueDecomposition ν
        r : NNReal
        hmeas : Measurable (μ.rnDeriv ν)
        hsing : (μ.singularPart ν).MutuallySingular ν
        hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
        hr : Not (Eq r 0)
        this : Eq (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν)) (HSMul.hSMul (↑(Inv.inv r))  …
        ⊢ Ne (↑r) 0
      -/
    · simp [hr]
      /-
        🎉 no goals
      -/
      /-
        case neg.ht
        α : Type u_1
        m : MeasurableSpace α
        μ✝ ν✝ μ ν : MeasureTheory.Measure α
        inst✝ : μ.HaveLebesgueDecomposition ν
        r : NNReal
        hmeas : Measurable (μ.rnDeriv ν)
        hsing : (μ.singularPart ν).MutuallySingular ν
        hadd : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
        hr : Not (Eq r 0)
        this : Eq (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν)) (HSMul.hSMul (↑(Inv.inv r))  …
        ⊢ Ne (↑r) Top.top
      -/
    · exact ENNReal.coe_ne_top
      /-
        🎉 no goals
      -/


theorem haveLebesgueDecomposition_withDensity (μ : Measure α) {f : α → ℝ≥0∞} (hf : Measurable f) :
    (μ.withDensity f).HaveLebesgueDecomposition μ := ⟨⟨⟨0, f⟩, hf, .zero_left, (zero_add _).symm⟩⟩


instance haveLebesgueDecompositionRnDeriv (μ ν : Measure α) :
    HaveLebesgueDecomposition (ν.withDensity (μ.rnDeriv ν)) ν :=
  haveLebesgueDecomposition_withDensity ν (measurable_rnDeriv _ _)


instance instHaveLebesgueDecompositionSingularPart :
    HaveLebesgueDecomposition (μ.singularPart ν) ν :=
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   m : MeasurableSpace α
                                                                                   μ ν : MeasureTheory.Measure α
                                                                                   ⊢ Eq (μ.singularPart ν) (HAdd.hAdd { fst := μ.singularPart ν, snd := 0 }.1 (ν. …
                                                                                 -/
  ⟨⟨μ.singularPart ν, 0⟩, measurable_zero, mutuallySingular_singularPart μ ν, by simp⟩
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem singularPart_le (μ ν : Measure α) : μ.singularPart ν ≤ μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ LE.le (μ.singularPart ν) μ
  -/
  by_cases hl : HaveLebesgueDecomposition μ ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ LE.le (μ.singularPart ν) μ
    -/
  · conv_rhs => rw [haveLebesgueDecomposition_add μ ν]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ LE.le (μ.singularPart ν) (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnD …
    -/
    exact Measure.le_add_right le_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ LE.le (μ.singularPart ν) μ
    -/
  · rw [singularPart, dif_neg hl]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ LE.le 0 μ
    -/
    exact Measure.zero_le μ
    /-
      🎉 no goals
    -/


theorem withDensity_rnDeriv_le (μ ν : Measure α) : ν.withDensity (μ.rnDeriv ν) ≤ μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ LE.le (ν.withDensity (μ.rnDeriv ν)) μ
  -/
  by_cases hl : HaveLebesgueDecomposition μ ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ LE.le (ν.withDensity (μ.rnDeriv ν)) μ
    -/
  · conv_rhs => rw [haveLebesgueDecomposition_add μ ν]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ LE.le (ν.withDensity (μ.rnDeriv ν)) (HAdd.hAdd (μ.singularPart ν) (ν.withDen …
    -/
    exact Measure.le_add_left le_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ LE.le (ν.withDensity (μ.rnDeriv ν)) μ
    -/
  · rw [rnDeriv, dif_neg hl, withDensity_zero]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ LE.le 0 μ
    -/
    exact Measure.zero_le μ
    /-
      🎉 no goals
    -/


lemma _root_.AEMeasurable.singularPart {β : Type*} {_ : MeasurableSpace β} {f : α → β}
    (hf : AEMeasurable f μ) (ν : Measure α) :
    AEMeasurable f (μ.singularPart ν) :=
  AEMeasurable.mono_measure hf (Measure.singularPart_le _ _)


lemma _root_.AEMeasurable.withDensity_rnDeriv {β : Type*} {_ : MeasurableSpace β} {f : α → β}
    (hf : AEMeasurable f μ) (ν : Measure α) :
    AEMeasurable f (ν.withDensity (μ.rnDeriv ν)) :=
  AEMeasurable.mono_measure hf (Measure.withDensity_rnDeriv_le _ _)


lemma MutuallySingular.singularPart (h : μ ⟂ₘ ν) (ν' : Measure α) :
    μ.singularPart ν' ⟂ₘ ν :=
  h.mono (singularPart_le μ ν') le_rfl


lemma absolutelyContinuous_withDensity_rnDeriv [HaveLebesgueDecomposition ν μ] (hμν : μ ≪ ν) :
    μ ≪ μ.withDensity (ν.rnDeriv μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous ν
    ⊢ μ.AbsolutelyContinuous (μ.withDensity (ν.rnDeriv μ))
  -/
  rw [haveLebesgueDecomposition_add ν μ] at hμν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    ⊢ μ.AbsolutelyContinuous (μ.withDensity (ν.rnDeriv μ))
  -/
  refine AbsolutelyContinuous.mk (fun s _ hνs ↦ ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    s : Set α
    x✝ : MeasurableSet s
    hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
    ⊢ Eq (μ s) 0
  -/
  obtain ⟨t, _, ht1, ht2⟩ := mutuallySingular_singularPart ν μ
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    s : Set α
    x✝ : MeasurableSet s
    hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
    t : Set α
    left✝ : MeasurableSet t
    ht1 : Eq ((ν.singularPart μ) t) 0
    ht2 : Eq (μ (HasCompl.compl t)) 0
    ⊢ Eq (μ s) 0
  -/
  rw [← inter_union_compl s]
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    s : Set α
    x✝ : MeasurableSet s
    hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
    t : Set α
    left✝ : MeasurableSet t
    ht1 : Eq ((ν.singularPart μ) t) 0
    ht2 : Eq (μ (HasCompl.compl t)) 0
    ⊢ Eq (μ (Union.union (Inter.inter s ?intro.intro.intro) (Inter.inter s (HasCom …
  -/
  refine le_antisymm ((measure_union_le (s ∩ t) (s ∩ tᶜ)).trans ?_) (zero_le _)
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    s : Set α
    x✝ : MeasurableSet s
    hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
    t : Set α
    left✝ : MeasurableSet t
    ht1 : Eq ((ν.singularPart μ) t) 0
    ht2 : Eq (μ (HasCompl.compl t)) 0
    ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (Inter.inter s (HasCompl.compl t)) …
  -/
  simp only [nonpos_iff_eq_zero, add_eq_zero]
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : ν.HaveLebesgueDecomposition μ
    hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
    s : Set α
    x✝ : MeasurableSet s
    hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
    t : Set α
    left✝ : MeasurableSet t
    ht1 : Eq ((ν.singularPart μ) t) 0
    ht2 : Eq (μ (HasCompl.compl t)) 0
    ⊢ And (Eq (μ (Inter.inter s t)) 0) (Eq (μ (Inter.inter s (HasCompl.compl t))) 0)
  -/
  constructor
    /-
      case intro.intro.intro.left
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : ν.HaveLebesgueDecomposition μ
      hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
      s : Set α
      x✝ : MeasurableSet s
      hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
      t : Set α
      left✝ : MeasurableSet t
      ht1 : Eq ((ν.singularPart μ) t) 0
      ht2 : Eq (μ (HasCompl.compl t)) 0
      ⊢ Eq (μ (Inter.inter s t)) 0
    -/
  · refine hμν ?_
    /-
      case intro.intro.intro.left
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : ν.HaveLebesgueDecomposition μ
      hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
      s : Set α
      x✝ : MeasurableSet s
      hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
      t : Set α
      left✝ : MeasurableSet t
      ht1 : Eq ((ν.singularPart μ) t) 0
      ht2 : Eq (μ (HasCompl.compl t)) 0
      ⊢ Eq ((HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.rnDeriv μ))) (Inter.inte …
    -/
    simp only [coe_add, Pi.add_apply, add_eq_zero]
    /-
      case intro.intro.intro.left
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : ν.HaveLebesgueDecomposition μ
      hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
      s : Set α
      x✝ : MeasurableSet s
      hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
      t : Set α
      left✝ : MeasurableSet t
      ht1 : Eq ((ν.singularPart μ) t) 0
      ht2 : Eq (μ (HasCompl.compl t)) 0
      ⊢ And (Eq ((ν.singularPart μ) (Inter.inter s t)) 0) (Eq ((μ.withDensity (ν.rnD …
    -/
    constructor
      /-
        case intro.intro.intro.left.left
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝ : ν.HaveLebesgueDecomposition μ
        hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
        s : Set α
        x✝ : MeasurableSet s
        hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
        t : Set α
        left✝ : MeasurableSet t
        ht1 : Eq ((ν.singularPart μ) t) 0
        ht2 : Eq (μ (HasCompl.compl t)) 0
        ⊢ Eq ((ν.singularPart μ) (Inter.inter s t)) 0
      -/
    · exact measure_mono_null Set.inter_subset_right ht1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.left.right
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝ : ν.HaveLebesgueDecomposition μ
        hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
        s : Set α
        x✝ : MeasurableSet s
        hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
        t : Set α
        left✝ : MeasurableSet t
        ht1 : Eq ((ν.singularPart μ) t) 0
        ht2 : Eq (μ (HasCompl.compl t)) 0
        ⊢ Eq ((μ.withDensity (ν.rnDeriv μ)) (Inter.inter s t)) 0
      -/
    · exact measure_mono_null Set.inter_subset_left hνs
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.right
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : ν.HaveLebesgueDecomposition μ
      hμν : μ.AbsolutelyContinuous (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.r …
      s : Set α
      x✝ : MeasurableSet s
      hνs : Eq ((μ.withDensity (ν.rnDeriv μ)) s) 0
      t : Set α
      left✝ : MeasurableSet t
      ht1 : Eq ((ν.singularPart μ) t) 0
      ht2 : Eq (μ (HasCompl.compl t)) 0
      ⊢ Eq (μ (Inter.inter s (HasCompl.compl t))) 0
    -/
  · exact measure_mono_null Set.inter_subset_right ht2
    /-
      🎉 no goals
    -/


lemma AbsolutelyContinuous.withDensity_rnDeriv {ξ : Measure α} [μ.HaveLebesgueDecomposition ν]
    (hξμ : ξ ≪ μ) (hξν : ξ ≪ ν) :
    ξ ≪ ν.withDensity (μ.rnDeriv ν) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ξ : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hξμ : ξ.AbsolutelyContinuous μ
    hξν : ξ.AbsolutelyContinuous ν
    ⊢ ξ.AbsolutelyContinuous (ν.withDensity (μ.rnDeriv ν))
  -/
  conv_rhs at hξμ => rw [μ.haveLebesgueDecomposition_add ν, add_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ξ : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hξμ : ξ.AbsolutelyContinuous (HAdd.hAdd (ν.withDensity (μ.rnDeriv ν)) (μ.singu …
    hξν : ξ.AbsolutelyContinuous ν
    ⊢ ξ.AbsolutelyContinuous (ν.withDensity (μ.rnDeriv ν))
  -/
  refine absolutelyContinuous_of_add_of_mutuallySingular hξμ ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ξ : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hξμ : ξ.AbsolutelyContinuous (HAdd.hAdd (ν.withDensity (μ.rnDeriv ν)) (μ.singu …
    hξν : ξ.AbsolutelyContinuous ν
    ⊢ ξ.MutuallySingular (μ.singularPart ν)
  -/
  exact MutuallySingular.mono_ac (mutuallySingular_singularPart μ ν).symm hξν .rfl
  /-
    🎉 no goals
  -/


lemma absolutelyContinuous_withDensity_rnDeriv_swap [ν.HaveLebesgueDecomposition μ] :
    ν.withDensity (μ.rnDeriv ν) ≪ μ.withDensity (ν.rnDeriv μ) :=
  (withDensity_absolutelyContinuous ν (μ.rnDeriv ν)).withDensity_rnDeriv
    (absolutelyContinuous_of_le (withDensity_rnDeriv_le _ _))


lemma singularPart_eq_zero_of_ac (h : μ ≪ ν) : μ.singularPart ν = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous ν
    ⊢ Eq (μ.singularPart ν) 0
  -/
  rw [← MutuallySingular.self_iff]
  exact MutuallySingular.mono_ac (mutuallySingular_singularPart _ _)
    AbsolutelyContinuous.rfl ((absolutelyContinuous_of_le (singularPart_le _ _)).trans h)


@[simp]
theorem singularPart_zero (ν : Measure α) : (0 : Measure α).singularPart ν = 0 :=
  singularPart_eq_zero_of_ac (AbsolutelyContinuous.zero _)


@[simp]
lemma singularPart_zero_right (μ : Measure α) : μ.singularPart 0 = μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.singularPart 0) μ
  -/
  conv_rhs => rw [haveLebesgueDecomposition_add μ 0]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.singularPart 0) (HAdd.hAdd (μ.singularPart 0) (MeasureTheory.Measure.w …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma singularPart_eq_zero (μ ν : Measure α) [μ.HaveLebesgueDecomposition ν] :
    μ.singularPart ν = 0 ↔ μ ≪ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    ⊢ Iff (Eq (μ.singularPart ν) 0) (μ.AbsolutelyContinuous ν)
  -/
  have h_dec := haveLebesgueDecomposition_add μ ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    ⊢ Iff (Eq (μ.singularPart ν) 0) (μ.AbsolutelyContinuous ν)
  -/
  refine ⟨fun h ↦ ?_, singularPart_eq_zero_of_ac⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    h : Eq (μ.singularPart ν) 0
    ⊢ μ.AbsolutelyContinuous ν
  -/
  rw [h, zero_add] at h_dec
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (ν.withDensity (μ.rnDeriv ν))
    h : Eq (μ.singularPart ν) 0
    ⊢ μ.AbsolutelyContinuous ν
  -/
  rw [h_dec]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (ν.withDensity (μ.rnDeriv ν))
    h : Eq (μ.singularPart ν) 0
    ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous ν
  -/
  exact withDensity_absolutelyContinuous ν _
  /-
    🎉 no goals
  -/


@[simp]
lemma withDensity_rnDeriv_eq_zero (μ ν : Measure α) [μ.HaveLebesgueDecomposition ν] :
    ν.withDensity (μ.rnDeriv ν) = 0 ↔ μ ⟂ₘ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    ⊢ Iff (Eq (ν.withDensity (μ.rnDeriv ν)) 0) (μ.MutuallySingular ν)
  -/
  have h_dec := haveLebesgueDecomposition_add μ ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    ⊢ Iff (Eq (ν.withDensity (μ.rnDeriv ν)) 0) (μ.MutuallySingular ν)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : Eq (ν.withDensity (μ.rnDeriv ν)) 0
      ⊢ μ.MutuallySingular ν
    -/
  · rw [h, add_zero] at h_dec
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (μ.singularPart ν)
      h : Eq (ν.withDensity (μ.rnDeriv ν)) 0
      ⊢ μ.MutuallySingular ν
    -/
    rw [h_dec]
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (μ.singularPart ν)
      h : Eq (ν.withDensity (μ.rnDeriv ν)) 0
      ⊢ (μ.singularPart ν).MutuallySingular ν
    -/
    exact mutuallySingular_singularPart μ ν
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : μ.MutuallySingular ν
      ⊢ Eq (ν.withDensity (μ.rnDeriv ν)) 0
    -/
  · rw [← MutuallySingular.self_iff]
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : μ.MutuallySingular ν
      ⊢ (ν.withDensity (μ.rnDeriv ν)).MutuallySingular (ν.withDensity (μ.rnDeriv ν))
    -/
    rw [h_dec, MutuallySingular.add_left_iff] at h
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : And ((μ.singularPart ν).MutuallySingular ν) ((ν.withDensity (μ.rnDeriv ν)) …
      ⊢ (ν.withDensity (μ.rnDeriv ν)).MutuallySingular (ν.withDensity (μ.rnDeriv ν))
    -/
    refine MutuallySingular.mono_ac h.2 AbsolutelyContinuous.rfl ?_
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : And ((μ.singularPart ν).MutuallySingular ν) ((ν.withDensity (μ.rnDeriv ν)) …
      ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous ν
    -/
    exact withDensity_absolutelyContinuous _ _
    /-
      🎉 no goals
    -/


@[simp]
lemma rnDeriv_eq_zero (μ ν : Measure α) [μ.HaveLebesgueDecomposition ν] :
    μ.rnDeriv ν =ᵐ[ν] 0 ↔ μ ⟂ₘ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    ⊢ Iff ((MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 0) (μ.MutuallySingular ν)
  -/
  rw [← withDensity_rnDeriv_eq_zero, withDensity_eq_zero_iff (measurable_rnDeriv _ _).aemeasurable]
  /-
    🎉 no goals
  -/


lemma rnDeriv_zero (ν : Measure α) : (0 : Measure α).rnDeriv ν =ᵐ[ν] 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    ⊢ (MeasureTheory.ae ν).EventuallyEq (MeasureTheory.Measure.rnDeriv 0 ν) 0
  -/
  rw [rnDeriv_eq_zero]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    ⊢ MeasureTheory.Measure.MutuallySingular 0 ν
  -/
  exact MutuallySingular.zero_left
  /-
    🎉 no goals
  -/


lemma MutuallySingular.rnDeriv_ae_eq_zero (hμν : μ ⟂ₘ ν) :
    μ.rnDeriv ν =ᵐ[ν] 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 0
  -/
  by_cases h : μ.HaveLebesgueDecomposition ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hμν : μ.MutuallySingular ν
      h : μ.HaveLebesgueDecomposition ν
      ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 0
    -/
  · rw [rnDeriv_eq_zero]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hμν : μ.MutuallySingular ν
      h : μ.HaveLebesgueDecomposition ν
      ⊢ μ.MutuallySingular ν
    -/
    exact hμν
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      hμν : μ.MutuallySingular ν
      h : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 0
    -/
  · rw [rnDeriv_of_not_haveLebesgueDecomposition h]
    /-
      🎉 no goals
    -/


@[simp]
theorem singularPart_withDensity (ν : Measure α) (f : α → ℝ≥0∞) :
    (ν.withDensity f).singularPart ν = 0 :=
  singularPart_eq_zero_of_ac (withDensity_absolutelyContinuous _ _)


lemma rnDeriv_singularPart (μ ν : Measure α) :
    (μ.singularPart ν).rnDeriv ν =ᵐ[ν] 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv ν) 0
  -/
  rw [rnDeriv_eq_zero]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ (μ.singularPart ν).MutuallySingular ν
  -/
  exact mutuallySingular_singularPart μ ν
  /-
    🎉 no goals
  -/


@[simp]
lemma singularPart_self (μ : Measure α) : μ.singularPart μ = 0 :=
  singularPart_eq_zero_of_ac Measure.AbsolutelyContinuous.rfl


lemma rnDeriv_self (μ : Measure α) [SigmaFinite μ] : μ.rnDeriv μ =ᵐ[μ] fun _ ↦ 1 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv μ) fun x => 1
  -/
  have h := rnDeriv_add_singularPart μ μ
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (HAdd.hAdd (μ.withDensity (μ.rnDeriv μ)) (μ.singularPart μ)) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv μ) fun x => 1
  -/
  rw [singularPart_self, add_zero] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (μ.withDensity (μ.rnDeriv μ)) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv μ) fun x => 1
  -/
  have h_one : μ = μ.withDensity 1 := by simp
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (μ.withDensity (μ.rnDeriv μ)) μ
    h_one : Eq μ (μ.withDensity 1)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv μ) fun x => 1
  -/
  conv_rhs at h => rw [h_one]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (μ.withDensity (μ.rnDeriv μ)) (μ.withDensity 1)
    h_one : Eq μ (μ.withDensity 1)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv μ) fun x => 1
  -/
  rwa [withDensity_eq_iff_of_sigmaFinite (measurable_rnDeriv _ _).aemeasurable] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (μ.withDensity (μ.rnDeriv μ)) (μ.withDensity 1)
    h_one : Eq μ (μ.withDensity 1)
    ⊢ AEMeasurable 1 μ
  -/
  exact aemeasurable_const
  /-
    🎉 no goals
  -/


lemma singularPart_eq_self [μ.HaveLebesgueDecomposition ν] : μ.singularPart ν = μ ↔ μ ⟂ₘ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    ⊢ Iff (Eq (μ.singularPart ν) μ) (μ.MutuallySingular ν)
  -/
  have h_dec := haveLebesgueDecomposition_add μ ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    ⊢ Iff (Eq (μ.singularPart ν) μ) (μ.MutuallySingular ν)
  -/
  refine ⟨fun h ↦ ?_, fun  h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : Eq (μ.singularPart ν) μ
      ⊢ μ.MutuallySingular ν
    -/
  · rw [← h]
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : Eq (μ.singularPart ν) μ
      ⊢ (μ.singularPart ν).MutuallySingular ν
    -/
    exact mutuallySingular_singularPart _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : μ.MutuallySingular ν
      ⊢ Eq (μ.singularPart ν) μ
    -/
  · conv_rhs => rw [h_dec]
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      h_dec : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
      h : μ.MutuallySingular ν
      ⊢ Eq (μ.singularPart ν) (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeri …
    -/
    rw [(withDensity_rnDeriv_eq_zero _ _).mpr h, add_zero]
    /-
      🎉 no goals
    -/


@[simp]
lemma singularPart_singularPart (μ ν : Measure α) :
    (μ.singularPart ν).singularPart ν = μ.singularPart ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ Eq ((μ.singularPart ν).singularPart ν) (μ.singularPart ν)
  -/
  rw [Measure.singularPart_eq_self]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ (μ.singularPart ν).MutuallySingular ν
  -/
  exact Measure.mutuallySingular_singularPart _ _
  /-
    🎉 no goals
  -/


instance singularPart.instIsFiniteMeasure [IsFiniteMeasure μ] :
    IsFiniteMeasure (μ.singularPart ν) :=
  isFiniteMeasure_of_le μ <| singularPart_le μ ν


instance singularPart.instSigmaFinite [SigmaFinite μ] : SigmaFinite (μ.singularPart ν) :=
  sigmaFinite_of_le μ <| singularPart_le μ ν


instance singularPart.instIsLocallyFiniteMeasure [TopologicalSpace α] [IsLocallyFiniteMeasure μ] :
    IsLocallyFiniteMeasure (μ.singularPart ν) :=
  isLocallyFiniteMeasure_of_le <| singularPart_le μ ν


instance withDensity.instIsFiniteMeasure [IsFiniteMeasure μ] :
    IsFiniteMeasure (ν.withDensity <| μ.rnDeriv ν) :=
  isFiniteMeasure_of_le μ <| withDensity_rnDeriv_le μ ν


instance withDensity.instSigmaFinite [SigmaFinite μ] :
    SigmaFinite (ν.withDensity <| μ.rnDeriv ν) :=
  sigmaFinite_of_le μ <| withDensity_rnDeriv_le μ ν


instance withDensity.instIsLocallyFiniteMeasure [TopologicalSpace α] [IsLocallyFiniteMeasure μ] :
    IsLocallyFiniteMeasure (ν.withDensity <| μ.rnDeriv ν) :=
  isLocallyFiniteMeasure_of_le <| withDensity_rnDeriv_le μ ν


theorem lintegral_rnDeriv_lt_top_of_measure_ne_top (ν : Measure α) {s : Set α} (hs : μ s ≠ ∞) :
    ∫⁻ x in s, μ.rnDeriv ν x ∂ν < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : Ne (μ s) Top.top
    ⊢ LT.lt (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) Top.top
  -/
  by_cases hl : HaveLebesgueDecomposition μ ν
  · suffices (∫⁻ x in toMeasurable μ s, μ.rnDeriv ν x ∂ν) < ∞ from
      lt_of_le_of_lt (lintegral_mono_set (subset_toMeasurable _ _)) this
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : Ne (μ s) Top.top
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ LT.lt (MeasureTheory.lintegral (ν.restrict (MeasureTheory.toMeasurable μ s)) …
    -/
    rw [← withDensity_apply _ (measurableSet_toMeasurable _ _)]
    calc
      _ ≤ (singularPart μ ν) (toMeasurable μ s) + _ := le_add_self
      _ = μ s := by rw [← Measure.add_apply, ← haveLebesgueDecomposition_add, measure_toMeasurable]
      _ < ⊤ := hs.lt_top
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : Ne (μ s) Top.top
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ LT.lt (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) Top.top
    -/
  · simp only [Measure.rnDeriv, dif_neg hl, Pi.zero_apply, lintegral_zero, ENNReal.zero_lt_top]
    /-
      🎉 no goals
    -/


theorem lintegral_rnDeriv_lt_top (μ ν : Measure α) [IsFiniteMeasure μ] :
    ∫⁻ x, μ.rnDeriv ν x ∂ν < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LT.lt (MeasureTheory.lintegral ν fun x => μ.rnDeriv ν x) Top.top
  -/
  rw [← setLIntegral_univ]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LT.lt (MeasureTheory.lintegral (ν.restrict Set.univ) fun x => μ.rnDeriv ν x) …
  -/
  exact lintegral_rnDeriv_lt_top_of_measure_ne_top _ (measure_lt_top _ _).ne
  /-
    🎉 no goals
  -/


lemma integrable_toReal_rnDeriv [IsFiniteMeasure μ] :
    Integrable (fun x ↦ (μ.rnDeriv ν x).toReal) ν :=
  integrable_toReal_of_lintegral_ne_top (Measure.measurable_rnDeriv _ _).aemeasurable
    (Measure.lintegral_rnDeriv_lt_top _ _).ne


/-- The Radon-Nikodym derivative of a sigma-finite measure `μ` with respect to another
measure `ν` is `ν`-almost everywhere finite. -/
theorem rnDeriv_lt_top (μ ν : Measure α) [SigmaFinite μ] : ∀ᵐ x ∂ν, μ.rnDeriv ν x < ∞ := by
  suffices ∀ n, ∀ᵐ x ∂ν, x ∈ spanningSets μ n → μ.rnDeriv ν x < ∞ by
    filter_upwards [ae_all_iff.2 this] with _ hx using hx _ (mem_spanningSetsIndex _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ ∀ (n : Nat), Filter.Eventually (fun x => Membership.mem (MeasureTheory.spann …
  -/
  intro n
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    ⊢ Filter.Eventually (fun x => Membership.mem (MeasureTheory.spanningSets μ n)  …
  -/
  rw [← ae_restrict_iff' (measurableSet_spanningSets _ _)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    ⊢ Filter.Eventually (fun x => LT.lt (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae …
  -/
  apply ae_lt_top (measurable_rnDeriv _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    ⊢ Ne (MeasureTheory.lintegral (ν.restrict (MeasureTheory.spanningSets μ n)) fu …
  -/
  refine (lintegral_rnDeriv_lt_top_of_measure_ne_top _ ?_).ne
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    ⊢ Ne (μ (MeasureTheory.spanningSets μ n)) Top.top
  -/
  exact (measure_spanningSets_lt_top _ _).ne
  /-
    🎉 no goals
  -/


lemma rnDeriv_ne_top (μ ν : Measure α) [SigmaFinite μ] : ∀ᵐ x ∂ν, μ.rnDeriv ν x ≠ ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ Filter.Eventually (fun x => Ne (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae ν)
  -/
  filter_upwards [Measure.rnDeriv_lt_top μ ν] with x hx using hx.ne
  /-
    🎉 no goals
  -/


/-- Given measures `μ` and `ν`, if `s` is a measure mutually singular to `ν` and `f` is a
measurable function such that `μ = s + fν`, then `s = μ.singularPart μ`.

This theorem provides the uniqueness of the `singularPart` in the Lebesgue decomposition theorem,
while `MeasureTheory.Measure.eq_rnDeriv` provides the uniqueness of the
`rnDeriv`. -/
theorem eq_singularPart {s : Measure α} {f : α → ℝ≥0∞} (hf : Measurable f) (hs : s ⟂ₘ ν)
    (hadd : μ = s + ν.withDensity f) : s = μ.singularPart ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    ⊢ Eq s (μ.singularPart ν)
  -/
  have : HaveLebesgueDecomposition μ ν := ⟨⟨⟨s, f⟩, hf, hs, hadd⟩⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    ⊢ Eq s (μ.singularPart ν)
  -/
  obtain ⟨hmeas, hsing, hadd'⟩ := haveLebesgueDecomposition_spec μ ν
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hsing : (μ.singularPart ν).MutuallySingular ν
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    ⊢ Eq s (μ.singularPart ν)
  -/
  obtain ⟨⟨S, hS₁, hS₂, hS₃⟩, ⟨T, hT₁, hT₂, hT₃⟩⟩ := hs, hsing
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (s S) 0
    hS₃ : Eq (ν (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : Eq ((μ.singularPart ν) T) 0
    hT₃ : Eq (ν (HasCompl.compl T)) 0
    ⊢ Eq s (μ.singularPart ν)
  -/
  rw [hadd'] at hadd
  have hνinter : ν (S ∩ T)ᶜ = 0 := by
    rw [compl_inter]
    refine nonpos_iff_eq_zero.1 (le_trans (measure_union_le _ _) ?_)
    rw [hT₃, hS₃, add_zero]
  have heq : s.restrict (S ∩ T)ᶜ = (μ.singularPart ν).restrict (S ∩ T)ᶜ := by
    ext1 A hA
    have hf : ν.withDensity f (A ∩ (S ∩ T)ᶜ) = 0 := by
      refine withDensity_absolutelyContinuous ν _ ?_
      rw [← nonpos_iff_eq_zero]
      exact hνinter ▸ measure_mono inter_subset_right
    have hrn : ν.withDensity (μ.rnDeriv ν) (A ∩ (S ∩ T)ᶜ) = 0 := by
      refine withDensity_absolutelyContinuous ν _ ?_
      rw [← nonpos_iff_eq_zero]
      exact hνinter ▸ measure_mono inter_subset_right
    rw [restrict_apply hA, restrict_apply hA, ← add_zero (s (A ∩ (S ∩ T)ᶜ)), ← hf, ← add_apply, ←
      hadd, add_apply, hrn, add_zero]
  have heq' : ∀ A : Set α, MeasurableSet A → s A = s.restrict (S ∩ T)ᶜ A := by
    intro A hA
    have hsinter : s (A ∩ (S ∩ T)) = 0 := by
      rw [← nonpos_iff_eq_zero]
      exact hS₂ ▸ measure_mono (inter_subset_right.trans inter_subset_left)
    rw [restrict_apply hA, ← diff_eq, AEDisjoint.measure_diff_left hsinter]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hadd : Eq (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))) (HAdd.h …
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (s S) 0
    hS₃ : Eq (ν (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : Eq ((μ.singularPart ν) T) 0
    hT₃ : Eq (ν (HasCompl.compl T)) 0
    hνinter : Eq (ν (HasCompl.compl (Inter.inter S T))) 0
    heq : Eq (s.restrict (HasCompl.compl (Inter.inter S T))) ((μ.singularPart ν).r …
    heq' : ∀ (A : Set α), MeasurableSet A → Eq (s A) ((s.restrict (HasCompl.compl  …
    ⊢ Eq s (μ.singularPart ν)
  -/
  ext1 A hA
  have hμinter : μ.singularPart ν (A ∩ (S ∩ T)) = 0 := by
    rw [← nonpos_iff_eq_zero]
    exact hT₂ ▸ measure_mono (inter_subset_right.trans inter_subset_right)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.h
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hadd : Eq (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))) (HAdd.h …
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (s S) 0
    hS₃ : Eq (ν (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : Eq ((μ.singularPart ν) T) 0
    hT₃ : Eq (ν (HasCompl.compl T)) 0
    hνinter : Eq (ν (HasCompl.compl (Inter.inter S T))) 0
    heq : Eq (s.restrict (HasCompl.compl (Inter.inter S T))) ((μ.singularPart ν).r …
    heq' : ∀ (A : Set α), MeasurableSet A → Eq (s A) ((s.restrict (HasCompl.compl  …
    A : Set α
    hA : MeasurableSet A
    hμinter : Eq ((μ.singularPart ν) (Inter.inter A (Inter.inter S T))) 0
    ⊢ Eq (s A) ((μ.singularPart ν) A)
  -/
  rw [heq' A hA, heq, restrict_apply hA, ← diff_eq, AEDisjoint.measure_diff_left hμinter]
  /-
    🎉 no goals
  -/


theorem singularPart_smul (μ ν : Measure α) (r : ℝ≥0) :
    (r • μ).singularPart ν = r • μ.singularPart ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    r : NNReal
    ⊢ Eq ((HSMul.hSMul r μ).singularPart ν) (HSMul.hSMul r (μ.singularPart ν))
  -/
  by_cases hr : r = 0
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Eq r 0
      ⊢ Eq ((HSMul.hSMul r μ).singularPart ν) (HSMul.hSMul r (μ.singularPart ν))
    -/
  · rw [hr, zero_smul, zero_smul, singularPart_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    r : NNReal
    hr : Not (Eq r 0)
    ⊢ Eq ((HSMul.hSMul r μ).singularPart ν) (HSMul.hSMul r (μ.singularPart ν))
  -/
  by_cases hl : HaveLebesgueDecomposition μ ν
  · refine (eq_singularPart ((measurable_rnDeriv μ ν).const_smul (r : ℝ≥0∞))
          (MutuallySingular.smul r (mutuallySingular_singularPart _ _)) ?_).symm
    rw [withDensity_smul _ (measurable_rnDeriv _ _), ← smul_add,
      ← haveLebesgueDecomposition_add μ ν, ENNReal.smul_def]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Not (Eq r 0)
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Eq ((HSMul.hSMul r μ).singularPart ν) (HSMul.hSMul r (μ.singularPart ν))
    -/
  · rw [singularPart, singularPart, dif_neg hl, dif_neg, smul_zero]
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Not (Eq r 0)
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Not ((HSMul.hSMul r μ).HaveLebesgueDecomposition ν)
    -/
    refine fun hl' ↦ hl ?_
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Not (Eq r 0)
      hl : Not (μ.HaveLebesgueDecomposition ν)
      hl' : (HSMul.hSMul r μ).HaveLebesgueDecomposition ν
      ⊢ μ.HaveLebesgueDecomposition ν
    -/
    rw [← inv_smul_smul₀ hr μ]
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Not (Eq r 0)
      hl : Not (μ.HaveLebesgueDecomposition ν)
      hl' : (HSMul.hSMul r μ).HaveLebesgueDecomposition ν
      ⊢ (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r μ)).HaveLebesgueDecomposition ν
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem singularPart_smul_right (μ ν : Measure α) (r : ℝ≥0) (hr : r ≠ 0) :
    μ.singularPart (r • ν) = μ.singularPart ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    r : NNReal
    hr : Ne r 0
    ⊢ Eq (μ.singularPart (HSMul.hSMul r ν)) (μ.singularPart ν)
  -/
  by_cases hl : HaveLebesgueDecomposition μ ν
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Ne r 0
      hl : μ.HaveLebesgueDecomposition ν
      ⊢ Eq (μ.singularPart (HSMul.hSMul r ν)) (μ.singularPart ν)
    -/
  · refine (eq_singularPart ((measurable_rnDeriv μ ν).const_smul r⁻¹) ?_ ?_).symm
    · exact (mutuallySingular_singularPart μ ν).mono_ac AbsolutelyContinuous.rfl
        smul_absolutelyContinuous
      /-
        case pos.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        ⊢ Eq μ (HAdd.hAdd (μ.singularPart ν) ((HSMul.hSMul r ν).withDensity (HSMul.hSM …
      -/
    · rw [ENNReal.smul_def r, withDensity_smul_measure, ← withDensity_smul]
      /-
        case pos.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        ⊢ Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (HSMul.hSMul (↑r) (HSMul.h …
      -/
      swap; · exact (measurable_rnDeriv _ _).const_smul _
              /-
                🎉 no goals
              -/
      /-
        case pos.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        ⊢ Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (HSMul.hSMul (↑r) (HSMul.h …
      -/
      convert haveLebesgueDecomposition_add μ ν
      /-
        case h.e'_3.h.e'_6.h.e'_4
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        ⊢ Eq (HSMul.hSMul (↑r) (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν))) (μ.rnDeriv ν)
      -/
      ext x
      /-
        case h.e'_3.h.e'_6.h.e'_4.h
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        x : α
        ⊢ Eq (HSMul.hSMul (↑r) (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν)) x) (μ.rnDeriv ν …
      -/
      simp only [Pi.smul_apply]
      /-
        case h.e'_3.h.e'_6.h.e'_4.h
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        r : NNReal
        hr : Ne r 0
        hl : μ.HaveLebesgueDecomposition ν
        x : α
        ⊢ Eq (HSMul.hSMul (↑r) (HSMul.hSMul (Inv.inv r) (μ.rnDeriv ν x))) (μ.rnDeriv ν …
      -/
      rw [← ENNReal.smul_def, smul_inv_smul₀ hr]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Ne r 0
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Eq (μ.singularPart (HSMul.hSMul r ν)) (μ.singularPart ν)
    -/
  · rw [singularPart, singularPart, dif_neg hl, dif_neg]
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Ne r 0
      hl : Not (μ.HaveLebesgueDecomposition ν)
      ⊢ Not (μ.HaveLebesgueDecomposition (HSMul.hSMul r ν))
    -/
    refine fun hl' ↦ hl ?_
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Ne r 0
      hl : Not (μ.HaveLebesgueDecomposition ν)
      hl' : μ.HaveLebesgueDecomposition (HSMul.hSMul r ν)
      ⊢ μ.HaveLebesgueDecomposition ν
    -/
    rw [← inv_smul_smul₀ hr ν]
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      r : NNReal
      hr : Ne r 0
      hl : Not (μ.HaveLebesgueDecomposition ν)
      hl' : μ.HaveLebesgueDecomposition (HSMul.hSMul r ν)
      ⊢ μ.HaveLebesgueDecomposition (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r ν))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem singularPart_add (μ₁ μ₂ ν : Measure α) [HaveLebesgueDecomposition μ₁ ν]
    [HaveLebesgueDecomposition μ₂ ν] :
    (μ₁ + μ₂).singularPart ν = μ₁.singularPart ν + μ₂.singularPart ν := by
  refine (eq_singularPart ((measurable_rnDeriv μ₁ ν).add (measurable_rnDeriv μ₂ ν))
    ((mutuallySingular_singularPart _ _).add_left (mutuallySingular_singularPart _ _)) ?_).symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ₁ μ₂ ν : MeasureTheory.Measure α
    inst✝¹ : μ₁.HaveLebesgueDecomposition ν
    inst✝ : μ₂.HaveLebesgueDecomposition ν
    ⊢ Eq (HAdd.hAdd μ₁ μ₂) (HAdd.hAdd (HAdd.hAdd (μ₁.singularPart ν) (μ₂.singularP …
  -/
  erw [withDensity_add_left (measurable_rnDeriv μ₁ ν)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ₁ μ₂ ν : MeasureTheory.Measure α
    inst✝¹ : μ₁.HaveLebesgueDecomposition ν
    inst✝ : μ₂.HaveLebesgueDecomposition ν
    ⊢ Eq (HAdd.hAdd μ₁ μ₂) (HAdd.hAdd (HAdd.hAdd (μ₁.singularPart ν) (μ₂.singularP …
  -/
  conv_rhs => rw [add_assoc, add_comm (μ₂.singularPart ν), ← add_assoc, ← add_assoc]
  rw [← haveLebesgueDecomposition_add μ₁ ν, add_assoc, add_comm (ν.withDensity (μ₂.rnDeriv ν)),
    ← haveLebesgueDecomposition_add μ₂ ν]


lemma singularPart_restrict (μ ν : Measure α) [HaveLebesgueDecomposition μ ν]
    {s : Set α} (hs : MeasurableSet s) :
    (μ.restrict s).singularPart ν = (μ.singularPart ν).restrict s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.restrict s).singularPart ν) ((μ.singularPart ν).restrict s)
  -/
  refine (Measure.eq_singularPart (f := s.indicator (μ.rnDeriv ν)) ?_ ?_ ?_).symm
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Measurable (s.indicator (μ.rnDeriv ν))
    -/
  · exact (μ.measurable_rnDeriv ν).indicator hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      s : Set α
      hs : MeasurableSet s
      ⊢ ((μ.singularPart ν).restrict s).MutuallySingular ν
    -/
  · exact (Measure.mutuallySingular_singularPart μ ν).restrict s
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : μ.HaveLebesgueDecomposition ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (μ.restrict s) (HAdd.hAdd ((μ.singularPart ν).restrict s) (ν.withDensity  …
    -/
  · ext t
    rw [withDensity_indicator hs, ← restrict_withDensity hs, ← Measure.restrict_add,
      ← μ.haveLebesgueDecomposition_add ν]


lemma measure_sub_singularPart (μ ν : Measure α) [HaveLebesgueDecomposition μ ν]
    [IsFiniteMeasure μ] :
    μ - μ.singularPart ν = ν.withDensity (μ.rnDeriv ν) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (HSub.hSub μ (μ.singularPart ν)) (ν.withDensity (μ.rnDeriv ν))
  -/
  nth_rw 1 [← rnDeriv_add_singularPart μ ν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (HSub.hSub (HAdd.hAdd (ν.withDensity (μ.rnDeriv ν)) (μ.singularPart ν)) ( …
  -/
  exact Measure.add_sub_cancel
  /-
    🎉 no goals
  -/


lemma measure_sub_rnDeriv (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] [IsFiniteMeasure μ] :
    μ - ν.withDensity (μ.rnDeriv ν) = μ.singularPart ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (HSub.hSub μ (ν.withDensity (μ.rnDeriv ν))) (μ.singularPart ν)
  -/
  nth_rw 1 [← singularPart_add_rnDeriv μ ν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (HSub.hSub (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))) ( …
  -/
  exact Measure.add_sub_cancel
  /-
    🎉 no goals
  -/


/-- Given measures `μ` and `ν`, if `s` is a measure mutually singular to `ν` and `f` is a
measurable function such that `μ = s + fν`, then `f = μ.rnDeriv ν`.

This theorem provides the uniqueness of the `rnDeriv` in the Lebesgue decomposition
theorem, while `MeasureTheory.Measure.eq_singularPart` provides the uniqueness of the
`singularPart`. Here, the uniqueness is given in terms of the measures, while the uniqueness in
terms of the functions is given in `eq_rnDeriv`. -/
theorem eq_withDensity_rnDeriv {s : Measure α} {f : α → ℝ≥0∞} (hf : Measurable f) (hs : s ⟂ₘ ν)
    (hadd : μ = s + ν.withDensity f) : ν.withDensity f = ν.withDensity (μ.rnDeriv ν) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  have : HaveLebesgueDecomposition μ ν := ⟨⟨⟨s, f⟩, hf, hs, hadd⟩⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  obtain ⟨hmeas, hsing, hadd'⟩ := haveLebesgueDecomposition_spec μ ν
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hsing : (μ.singularPart ν).MutuallySingular ν
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  obtain ⟨⟨S, hS₁, hS₂, hS₃⟩, ⟨T, hT₁, hT₂, hT₃⟩⟩ := hs, hsing
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (s S) 0
    hS₃ : Eq (ν (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : Eq ((μ.singularPart ν) T) 0
    hT₃ : Eq (ν (HasCompl.compl T)) 0
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  rw [hadd'] at hadd
  have hνinter : ν (S ∩ T)ᶜ = 0 := by
    rw [compl_inter]
    refine nonpos_iff_eq_zero.1 (le_trans (measure_union_le _ _) ?_)
    rw [hT₃, hS₃, add_zero]
  have heq :
    (ν.withDensity f).restrict (S ∩ T) = (ν.withDensity (μ.rnDeriv ν)).restrict (S ∩ T) := by
    ext1 A hA
    have hs : s (A ∩ (S ∩ T)) = 0 := by
      rw [← nonpos_iff_eq_zero]
      exact hS₂ ▸ measure_mono (inter_subset_right.trans inter_subset_left)
    have hsing : μ.singularPart ν (A ∩ (S ∩ T)) = 0 := by
      rw [← nonpos_iff_eq_zero]
      exact hT₂ ▸ measure_mono (inter_subset_right.trans inter_subset_right)
    rw [restrict_apply hA, restrict_apply hA, ← add_zero (ν.withDensity f (A ∩ (S ∩ T))), ← hs, ←
      add_apply, add_comm, ← hadd, add_apply, hsing, zero_add]
  have heq' :
    ∀ A : Set α, MeasurableSet A → ν.withDensity f A = (ν.withDensity f).restrict (S ∩ T) A := by
    intro A hA
    have hνfinter : ν.withDensity f (A ∩ (S ∩ T)ᶜ) = 0 := by
      rw [← nonpos_iff_eq_zero]
      exact withDensity_absolutelyContinuous ν f hνinter ▸ measure_mono inter_subset_right
    rw [restrict_apply hA, ← add_zero (ν.withDensity f (A ∩ (S ∩ T))), ← hνfinter, ← diff_eq,
      measure_inter_add_diff _ (hS₁.inter hT₁)]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hadd : Eq (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))) (HAdd.h …
    this : μ.HaveLebesgueDecomposition ν
    hmeas : Measurable (μ.rnDeriv ν)
    hadd' : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (s S) 0
    hS₃ : Eq (ν (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : Eq ((μ.singularPart ν) T) 0
    hT₃ : Eq (ν (HasCompl.compl T)) 0
    hνinter : Eq (ν (HasCompl.compl (Inter.inter S T))) 0
    heq : Eq ((ν.withDensity f).restrict (Inter.inter S T)) ((ν.withDensity (μ.rnD …
    heq' : ∀ (A : Set α), MeasurableSet A → Eq ((ν.withDensity f) A) (((ν.withDens …
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  ext1 A hA
  have hνrn : ν.withDensity (μ.rnDeriv ν) (A ∩ (S ∩ T)ᶜ) = 0 := by
    rw [← nonpos_iff_eq_zero]
    exact
      withDensity_absolutelyContinuous ν (μ.rnDeriv ν) hνinter ▸
        measure_mono inter_subset_right
  rw [heq' A hA, heq, ← add_zero ((ν.withDensity (μ.rnDeriv ν)).restrict (S ∩ T) A), ← hνrn,
    restrict_apply hA, ← diff_eq, measure_inter_add_diff _ (hS₁.inter hT₁)]


theorem eq_withDensity_rnDeriv₀ {s : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f ν) (hs : s ⟂ₘ ν) (hadd : μ = s + ν.withDensity f) :
    ν.withDensity f = ν.withDensity (μ.rnDeriv ν) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f ν
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity f))
    ⊢ Eq (ν.withDensity f) (ν.withDensity (μ.rnDeriv ν))
  -/
  rw [withDensity_congr_ae hf.ae_eq_mk] at hadd ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν s : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f ν
    hs : s.MutuallySingular ν
    hadd : Eq μ (HAdd.hAdd s (ν.withDensity (AEMeasurable.mk f hf)))
    ⊢ Eq (ν.withDensity (AEMeasurable.mk f hf)) (ν.withDensity (μ.rnDeriv ν))
  -/
  exact eq_withDensity_rnDeriv hf.measurable_mk hs hadd
  /-
    🎉 no goals
  -/


theorem eq_rnDeriv₀ [SigmaFinite ν] {s : Measure α} {f : α → ℝ≥0∞}
    (hf : AEMeasurable f ν) (hs : s ⟂ₘ ν) (hadd : μ = s + ν.withDensity f) :
    f =ᵐ[ν] μ.rnDeriv ν :=
  (withDensity_eq_iff_of_sigmaFinite hf (measurable_rnDeriv _ _).aemeasurable).mp
    (eq_withDensity_rnDeriv₀ hf hs hadd)


/-- Given measures `μ` and `ν`, if `s` is a measure mutually singular to `ν` and `f` is a
measurable function such that `μ = s + fν`, then `f = μ.rnDeriv ν`.

This theorem provides the uniqueness of the `rnDeriv` in the Lebesgue decomposition
theorem, while `MeasureTheory.Measure.eq_singularPart` provides the uniqueness of the
`singularPart`. Here, the uniqueness is given in terms of the functions, while the uniqueness in
terms of the functions is given in `eq_withDensity_rnDeriv`. -/
theorem eq_rnDeriv [SigmaFinite ν] {s : Measure α} {f : α → ℝ≥0∞} (hf : Measurable f) (hs : s ⟂ₘ ν)
    (hadd : μ = s + ν.withDensity f) : f =ᵐ[ν] μ.rnDeriv ν :=
  eq_rnDeriv₀ hf.aemeasurable hs hadd


/-- The Radon-Nikodym derivative of `f ν` with respect to `ν` is `f`. -/
theorem rnDeriv_withDensity₀ (ν : Measure α) [SigmaFinite ν] {f : α → ℝ≥0∞}
    (hf : AEMeasurable f ν) :
    (ν.withDensity f).rnDeriv ν =ᵐ[ν] f :=
                                                     /-
                                                       α : Type u_1
                                                       m : MeasurableSpace α
                                                       ν : MeasureTheory.Measure α
                                                       inst✝ : MeasureTheory.SigmaFinite ν
                                                       f : α → ENNReal
                                                       hf : AEMeasurable f ν
                                                       ⊢ Eq (ν.withDensity f) (HAdd.hAdd 0 (ν.withDensity f))
                                                     -/
  have : ν.withDensity f = 0 + ν.withDensity f := by rw [zero_add]
                                                     /-
                                                       🎉 no goals
                                                     -/
  (eq_rnDeriv₀ hf MutuallySingular.zero_left this).symm


/-- The Radon-Nikodym derivative of `f ν` with respect to `ν` is `f`. -/
theorem rnDeriv_withDensity (ν : Measure α) [SigmaFinite ν] {f : α → ℝ≥0∞} (hf : Measurable f) :
    (ν.withDensity f).rnDeriv ν =ᵐ[ν] f :=
  rnDeriv_withDensity₀ ν hf.aemeasurable


lemma rnDeriv_restrict (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] [SigmaFinite ν]
    {s : Set α} (hs : MeasurableSet s) :
    (μ.restrict s).rnDeriv ν =ᵐ[ν] s.indicator (μ.rnDeriv ν) := by
  refine (eq_rnDeriv (s := (μ.restrict s).singularPart ν)
    ((measurable_rnDeriv _ _).indicator hs) (mutuallySingular_singularPart _ _) ?_).symm
  rw [singularPart_restrict _ _ hs, withDensity_indicator hs, ← restrict_withDensity hs,
    ← Measure.restrict_add, ← μ.haveLebesgueDecomposition_add ν]


/-- The Radon-Nikodym derivative of the restriction of a measure to a measurable set is the
indicator function of this set. -/
theorem rnDeriv_restrict_self (ν : Measure α) [SigmaFinite ν] {s : Set α} (hs : MeasurableSet s) :
    (ν.restrict s).rnDeriv ν =ᵐ[ν] s.indicator 1 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.restrict s).rnDeriv ν) (s.indicator 1)
  -/
  rw [← withDensity_indicator_one hs]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (s.indicator 1)).rnDeriv ν …
  -/
  exact rnDeriv_withDensity _ (measurable_one.indicator hs)
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative of the scalar multiple of a measure.
See also `rnDeriv_smul_left'`, which requires sigma-finite `ν` and `μ`. -/
theorem rnDeriv_smul_left (ν μ : Measure α) [IsFiniteMeasure ν]
    [ν.HaveLebesgueDecomposition μ] (r : ℝ≥0) :
    (r • ν).rnDeriv μ =ᵐ[μ] r • ν.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : NNReal
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r ν).rnDeriv μ) (HSMul.hSMul …
  -/
  rw [← withDensity_eq_iff]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ Eq (μ.withDensity ((HSMul.hSMul r ν).rnDeriv μ)) (μ.withDensity (HSMul.hSMul …
    -/
  · simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ Eq (μ.withDensity ((HSMul.hSMul (↑r) ν).rnDeriv μ)) (μ.withDensity (HSMul.hS …
    -/
    rw [withDensity_smul _ (measurable_rnDeriv _ _)]
    suffices (r • ν).singularPart μ + withDensity μ (rnDeriv (r • ν) μ)
        = (r • ν).singularPart μ + r • withDensity μ (rnDeriv ν μ) by
      rwa [Measure.add_right_inj] at this
    rw [← (r • ν).haveLebesgueDecomposition_add μ, singularPart_smul, ← smul_add,
      ← ν.haveLebesgueDecomposition_add μ]
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ AEMeasurable ((HSMul.hSMul r ν).rnDeriv μ) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ AEMeasurable (HSMul.hSMul r (ν.rnDeriv μ)) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable.const_smul _
    /-
      🎉 no goals
    -/
    /-
      case hfi
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      ⊢ Ne (MeasureTheory.lintegral μ fun x => (HSMul.hSMul r ν).rnDeriv μ x) Top.top
    -/
  · exact (lintegral_rnDeriv_lt_top (r • ν) μ).ne
    /-
      🎉 no goals
    -/


/-- Radon-Nikodym derivative of the scalar multiple of a measure.
See also `rnDeriv_smul_left_of_ne_top'`, which requires sigma-finite `ν` and `μ`. -/
theorem rnDeriv_smul_left_of_ne_top (ν μ : Measure α) [IsFiniteMeasure ν]
    [ν.HaveLebesgueDecomposition μ] {r : ℝ≥0∞} (hr : r ≠ ∞) :
    (r • ν).rnDeriv μ =ᵐ[μ] r • ν.rnDeriv μ := by
  have h : (r.toNNReal • ν).rnDeriv μ =ᵐ[μ] r.toNNReal • ν.rnDeriv μ :=
    rnDeriv_smul_left ν μ r.toNNReal
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : ENNReal
    hr : Ne r Top.top
    h : (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r.toNNReal ν).rnDeriv μ) ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r ν).rnDeriv μ) (HSMul.hSMul …
  -/
  simpa [ENNReal.smul_def, ENNReal.coe_toNNReal hr] using h
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative with respect to the scalar multiple of a measure.
See also `rnDeriv_smul_right'`, which requires sigma-finite `ν` and `μ`. -/
theorem rnDeriv_smul_right (ν μ : Measure α) [IsFiniteMeasure ν]
    [ν.HaveLebesgueDecomposition μ] {r : ℝ≥0} (hr : r ≠ 0) :
    ν.rnDeriv (r • μ) =ᵐ[μ] r⁻¹ • ν.rnDeriv μ := by
  refine (absolutelyContinuous_smul <| ENNReal.coe_ne_zero.2 hr).ae_le
    (?_ : ν.rnDeriv (r • μ) =ᵐ[r • μ] r⁻¹ • ν.rnDeriv μ)
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : NNReal
    hr : Ne r 0
    ⊢ (MeasureTheory.ae (HSMul.hSMul r μ)).EventuallyEq (ν.rnDeriv (HSMul.hSMul r  …
  -/
  rw [← withDensity_eq_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : NNReal
    hr : Ne r 0
    ⊢ Eq ((HSMul.hSMul r μ).withDensity (ν.rnDeriv (HSMul.hSMul r μ))) ((HSMul.hSM …
  -/
  rotate_left
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ AEMeasurable (ν.rnDeriv (HSMul.hSMul r μ)) (HSMul.hSMul r μ)
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ AEMeasurable (HSMul.hSMul (Inv.inv r) (ν.rnDeriv μ)) (HSMul.hSMul r μ)
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable.const_smul _
    /-
      🎉 no goals
    -/
    /-
      case hfi
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ Ne (MeasureTheory.lintegral (HSMul.hSMul r μ) fun x => ν.rnDeriv (HSMul.hSMu …
    -/
  · exact (lintegral_rnDeriv_lt_top ν _).ne
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ((HSMul.hSMul r μ).withDensity (ν.rnDeriv (HSMul.hSMul r μ))) ((HSMul.hSM …
    -/
  · simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ((HSMul.hSMul (↑r) μ).withDensity (ν.rnDeriv (HSMul.hSMul (↑r) μ))) ((HSM …
    -/
    rw [withDensity_smul _ (measurable_rnDeriv _ _)]
    suffices ν.singularPart (r • μ) + withDensity (r • μ) (rnDeriv ν (r • μ))
        = ν.singularPart (r • μ) + r⁻¹ • withDensity (r • μ) (rnDeriv ν μ) by
      rwa [add_right_inj] at this
    rw [← ν.haveLebesgueDecomposition_add (r • μ), singularPart_smul_right _ _ _ hr,
      ENNReal.smul_def r, withDensity_smul_measure, ← ENNReal.smul_def, ← smul_assoc,
      smul_eq_mul, inv_mul_cancel₀ hr, one_smul]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : ν.HaveLebesgueDecomposition μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ν (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.rnDeriv μ)))
    -/
    exact ν.haveLebesgueDecomposition_add μ
    /-
      🎉 no goals
    -/


/-- Radon-Nikodym derivative with respect to the scalar multiple of a measure.
See also `rnDeriv_smul_right_of_ne_top'`, which requires sigma-finite `ν` and `μ`. -/
theorem rnDeriv_smul_right_of_ne_top (ν μ : Measure α) [IsFiniteMeasure ν]
    [ν.HaveLebesgueDecomposition μ] {r : ℝ≥0∞} (hr : r ≠ 0) (hr_ne_top : r ≠ ∞) :
    ν.rnDeriv (r • μ) =ᵐ[μ] r⁻¹ • ν.rnDeriv μ := by
  have h : ν.rnDeriv (r.toNNReal • μ) =ᵐ[μ] r.toNNReal⁻¹ • ν.rnDeriv μ := by
    refine rnDeriv_smul_right ν μ ?_
    rw [ne_eq, ENNReal.toNNReal_eq_zero_iff]
    simp [hr, hr_ne_top]
  have : (r.toNNReal)⁻¹ • rnDeriv ν μ = r⁻¹ • rnDeriv ν μ := by
    ext x
    simp only [Pi.smul_apply, ENNReal.smul_def, ne_eq, smul_eq_mul]
    rw [ENNReal.coe_inv, ENNReal.coe_toNNReal hr_ne_top]
    rw [ne_eq, ENNReal.toNNReal_eq_zero_iff]
    simp [hr, hr_ne_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : ENNReal
    hr : Ne r 0
    hr_ne_top : Ne r Top.top
    h : (MeasureTheory.ae μ).EventuallyEq (ν.rnDeriv (HSMul.hSMul r.toNNReal μ)) ( …
    this : Eq (HSMul.hSMul (Inv.inv r.toNNReal) (ν.rnDeriv μ)) (HSMul.hSMul (Inv.i …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (ν.rnDeriv (HSMul.hSMul r μ)) (HSMul.hSMul …
  -/
  simp_rw [this, ENNReal.smul_def, ENNReal.coe_toNNReal hr_ne_top] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : ν.HaveLebesgueDecomposition μ
    r : ENNReal
    hr : Ne r 0
    hr_ne_top : Ne r Top.top
    this : Eq (HSMul.hSMul (Inv.inv r.toNNReal) (ν.rnDeriv μ)) (HSMul.hSMul (Inv.i …
    h : (MeasureTheory.ae μ).EventuallyEq (ν.rnDeriv (HSMul.hSMul r μ)) (HSMul.hSM …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (ν.rnDeriv (HSMul.hSMul r μ)) (HSMul.hSMul …
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative of a sum of two measures.
See also `rnDeriv_add'`, which requires sigma-finite `ν₁`, `ν₂` and `μ`. -/
lemma rnDeriv_add (ν₁ ν₂ μ : Measure α) [IsFiniteMeasure ν₁] [IsFiniteMeasure ν₂]
    [ν₁.HaveLebesgueDecomposition μ] [ν₂.HaveLebesgueDecomposition μ]
    [(ν₁ + ν₂).HaveLebesgueDecomposition μ] :
    (ν₁ + ν₂).rnDeriv μ =ᵐ[μ] ν₁.rnDeriv μ + ν₂.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν₁ ν₂ μ : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.IsFiniteMeasure ν₁
    inst✝³ : MeasureTheory.IsFiniteMeasure ν₂
    inst✝² : ν₁.HaveLebesgueDecomposition μ
    inst✝¹ : ν₂.HaveLebesgueDecomposition μ
    inst✝ : (HAdd.hAdd ν₁ ν₂).HaveLebesgueDecomposition μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HAdd.hAdd ν₁ ν₂).rnDeriv μ) (HAdd.hAdd ( …
  -/
  rw [← withDensity_eq_iff]
  · suffices (ν₁ + ν₂).singularPart μ + μ.withDensity ((ν₁ + ν₂).rnDeriv μ)
        = (ν₁ + ν₂).singularPart μ + μ.withDensity (ν₁.rnDeriv μ + ν₂.rnDeriv μ) by
      rwa [add_right_inj] at this
    rw [← (ν₁ + ν₂).haveLebesgueDecomposition_add μ, singularPart_add,
      withDensity_add_left (measurable_rnDeriv _ _), add_assoc,
      add_comm (ν₂.singularPart μ), add_assoc, add_comm _ (ν₂.singularPart μ),
      ← ν₂.haveLebesgueDecomposition_add μ, ← add_assoc, ← ν₁.haveLebesgueDecomposition_add μ]
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν₁ ν₂ μ : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.IsFiniteMeasure ν₁
      inst✝³ : MeasureTheory.IsFiniteMeasure ν₂
      inst✝² : ν₁.HaveLebesgueDecomposition μ
      inst✝¹ : ν₂.HaveLebesgueDecomposition μ
      inst✝ : (HAdd.hAdd ν₁ ν₂).HaveLebesgueDecomposition μ
      ⊢ AEMeasurable ((HAdd.hAdd ν₁ ν₂).rnDeriv μ) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν₁ ν₂ μ : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.IsFiniteMeasure ν₁
      inst✝³ : MeasureTheory.IsFiniteMeasure ν₂
      inst✝² : ν₁.HaveLebesgueDecomposition μ
      inst✝¹ : ν₂.HaveLebesgueDecomposition μ
      inst✝ : (HAdd.hAdd ν₁ ν₂).HaveLebesgueDecomposition μ
      ⊢ AEMeasurable (HAdd.hAdd (ν₁.rnDeriv μ) (ν₂.rnDeriv μ)) μ
    -/
  · exact ((measurable_rnDeriv _ _).add (measurable_rnDeriv _ _)).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hfi
      α : Type u_1
      m : MeasurableSpace α
      ν₁ ν₂ μ : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.IsFiniteMeasure ν₁
      inst✝³ : MeasureTheory.IsFiniteMeasure ν₂
      inst✝² : ν₁.HaveLebesgueDecomposition μ
      inst✝¹ : ν₂.HaveLebesgueDecomposition μ
      inst✝ : (HAdd.hAdd ν₁ ν₂).HaveLebesgueDecomposition μ
      ⊢ Ne (MeasureTheory.lintegral μ fun x => (HAdd.hAdd ν₁ ν₂).rnDeriv μ x) Top.top
    -/
  · exact (lintegral_rnDeriv_lt_top (ν₁ + ν₂) μ).ne
    /-
      🎉 no goals
    -/


/-- If two finite measures `μ` and `ν` are not mutually singular, there exists some `ε > 0` and
a measurable set `E`, such that `ν(E) > 0` and `E` is positive with respect to `μ - εν`.

This lemma is useful for the Lebesgue decomposition theorem. -/
theorem exists_positive_of_not_mutuallySingular (μ ν : Measure α) [IsFiniteMeasure μ]
    [IsFiniteMeasure ν] (h : ¬ μ ⟂ₘ ν) :
    ∃ ε : ℝ≥0, 0 < ε ∧
      ∃ E : Set α, MeasurableSet E ∧ 0 < ν E
        ∧ ∀ A, MeasurableSet A → ε * ν (A ∩ E) ≤ μ (A ∩ E) := by
  -- for all `n : ℕ`, obtain the Hahn decomposition for `μ - (1 / n) ν`
  have h_decomp (n : ℕ) : ∃ s : Set α, MeasurableSet s
        ∧ (∀ t, MeasurableSet t → ((1 / (n + 1) : ℝ≥0) • ν) (t ∩ s) ≤ μ (t ∩ s))
        ∧ (∀ t, MeasurableSet t → μ (t ∩ sᶜ) ≤ ((1 / (n + 1) : ℝ≥0) • ν) (t ∩ sᶜ)) := by
    obtain ⟨s, hs, hs_le, hs_ge⟩ := hahn_decomposition μ ((1 / (n + 1) : ℝ≥0) • ν)
    refine ⟨s, hs, fun t ht ↦ ?_, fun t ht ↦ ?_⟩
    · exact hs_le (t ∩ s) (ht.inter hs) inter_subset_right
    · exact hs_ge (t ∩ sᶜ) (ht.inter hs.compl) inter_subset_right
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Not (μ.MutuallySingular ν)
    h_decomp : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (And (∀ (t : Set …
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  choose f hf₁ hf₂ hf₃ using h_decomp
  -- set `A` to be the intersection of all the negative parts of obtained Hahn decompositions
  -- and we show that `μ A = 0`
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Not (μ.MutuallySingular ν)
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  let A := ⋂ n, (f n)ᶜ
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Not (μ.MutuallySingular ν)
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  have hAmeas : MeasurableSet A := MeasurableSet.iInter fun n ↦ (hf₁ n).compl
  have hA₂ (n : ℕ) (t : Set α) (ht : MeasurableSet t) :
      μ (t ∩ A) ≤ ((1 / (n + 1) : ℝ≥0) • ν) (t ∩ A) := by
    specialize hf₃ n (t ∩ A) (ht.inter hAmeas)
    have : A ∩ (f n)ᶜ = A := inter_eq_left.mpr (iInter_subset _ n)
    rwa [inter_assoc, this] at hf₃
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Not (μ.MutuallySingular ν)
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  have hA₃ (n : ℕ) : μ A ≤ (1 / (n + 1) : ℝ≥0) * ν A := by simpa using hA₂ n univ .univ
  have hμ : μ A = 0 := by
    lift μ A to ℝ≥0 using measure_ne_top _ _ with μA
    lift ν A to ℝ≥0 using measure_ne_top _ _ with νA
    rw [ENNReal.coe_eq_zero]
    by_cases hb : 0 < νA
    · suffices ∀ b, 0 < b → μA ≤ b by
        by_contra h
        have h' := this (μA / 2) (half_pos (zero_lt_iff.2 h))
        rw [← @Classical.not_not (μA ≤ μA / 2)] at h'
        exact h' (not_le.2 (NNReal.half_lt_self h))
      intro c hc
      have : ∃ n : ℕ, 1 / (n + 1 : ℝ) < c * (νA : ℝ)⁻¹ := by
        refine exists_nat_one_div_lt ?_
        positivity
      rcases this with ⟨n, hn⟩
      have hb₁ : (0 : ℝ) < (νA : ℝ)⁻¹ := by rw [_root_.inv_pos]; exact hb
      have h' : 1 / (↑n + 1) * νA < c := by
        rw [← NNReal.coe_lt_coe, ← mul_lt_mul_right hb₁, NNReal.coe_mul, mul_assoc, ←
          NNReal.coe_inv, ← NNReal.coe_mul, mul_inv_cancel₀, ← NNReal.coe_mul, mul_one,
          NNReal.coe_inv]
        · exact hn
        · exact hb.ne'
      refine le_trans ?_ h'.le
      rw [← ENNReal.coe_le_coe, ENNReal.coe_mul]
      exact hA₃ n
    · rw [not_lt, le_zero_iff] at hb
      specialize hA₃ 0
      simp? [hb] at hA₃ says
        simp only [CharP.cast_eq_zero, zero_add, ne_eq, one_ne_zero, not_false_eq_true, div_self,
          ENNReal.coe_one, hb, ENNReal.coe_zero, mul_zero, nonpos_iff_eq_zero,
          ENNReal.coe_eq_zero] at hA₃
      assumption
  -- since `μ` and `ν` are not mutually singular, `μ A = 0` implies `ν Aᶜ > 0`
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Not (μ.MutuallySingular ν)
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    hA₃ : ∀ (n : Nat), LE.le (μ A) (HMul.hMul (↑(HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)))  …
    hμ : Eq (μ A) 0
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  rw [MutuallySingular] at h; push_neg at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    hA₃ : ∀ (n : Nat), LE.le (μ A) (HMul.hMul (↑(HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)))  …
    hμ : Eq (μ A) 0
    h : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Ne (ν (HasCompl.compl s)) 0
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  have := h _ hAmeas hμ
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    hA₃ : ∀ (n : Nat), LE.le (μ A) (HMul.hMul (↑(HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)))  …
    hμ : Eq (μ A) 0
    h : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Ne (ν (HasCompl.compl s)) 0
    this : Ne (ν (HasCompl.compl A)) 0
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  simp_rw [A, compl_iInter, compl_compl] at this
  -- as `Aᶜ = ⋃ n, f n`, `ν Aᶜ > 0` implies there exists some `n` such that `ν (f n) > 0`
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    hA₃ : ∀ (n : Nat), LE.le (μ A) (HMul.hMul (↑(HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)))  …
    hμ : Eq (μ A) 0
    h : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Ne (ν (HasCompl.compl s)) 0
    this : Ne (ν (Set.iUnion fun i => f i)) 0
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  obtain ⟨n, hn⟩ := exists_measure_pos_of_not_measure_iUnion_null this
  -- thus, choosing `f n` as the set `E` suffices
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le ((HSMul.hSMul (HDiv.hDi …
    hf₃ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t (HasC …
    A : Set α := Set.iInter fun n => HasCompl.compl (f n)
    hAmeas : MeasurableSet A
    hA₂ : ∀ (n : Nat) (t : Set α), MeasurableSet t → LE.le (μ (Inter.inter t A)) ( …
    hA₃ : ∀ (n : Nat), LE.le (μ A) (HMul.hMul (↑(HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)))  …
    hμ : Eq (μ A) 0
    h : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Ne (ν (HasCompl.compl s)) 0
    this : Ne (ν (Set.iUnion fun i => f i)) 0
    n : Nat
    hn : LT.lt 0 (ν (f n))
    ⊢ Exists fun ε => And (LT.lt 0 ε) (Exists fun E => And (MeasurableSet E) (And  …
  -/
  exact ⟨1 / (n + 1), by simp, f n, hf₁ n, hn, hf₂ n⟩
  /-
    🎉 no goals
  -/


/-- Given two measures `μ` and `ν`, `measurableLE μ ν` is the set of measurable
functions `f`, such that, for all measurable sets `A`, `∫⁻ x in A, f x ∂μ ≤ ν A`.

This is useful for the Lebesgue decomposition theorem. -/
def measurableLE (μ ν : Measure α) : Set (α → ℝ≥0∞) :=
  {f | Measurable f ∧ ∀ (A : Set α), MeasurableSet A → (∫⁻ x in A, f x ∂μ) ≤ ν A}


theorem zero_mem_measurableLE : (0 : α → ℝ≥0∞) ∈ measurableLE μ ν :=
                                 /-
                                   α : Type u_1
                                   m : MeasurableSpace α
                                   μ ν : MeasureTheory.Measure α
                                   A : Set α
                                   x✝ : MeasurableSet A
                                   ⊢ LE.le (MeasureTheory.lintegral (μ.restrict A) fun x => 0 x) (ν A)
                                 -/
  ⟨measurable_zero, fun A _ ↦ by simp⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem sup_mem_measurableLE {f g : α → ℝ≥0∞} (hf : f ∈ measurableLE μ ν)
    (hg : g ∈ measurableLE μ ν) : (fun a ↦ f a ⊔ g a) ∈ measurableLE μ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    ⊢ Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE μ ν …
  -/
  refine ⟨Measurable.max hf.1 hg.1, fun A hA ↦ ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict A) fun x => (fun a => Max.max (f  …
  -/
  have h₁ := hA.inter (measurableSet_le hf.1 hg.1)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    h₁ : MeasurableSet (Inter.inter A (setOf fun a => LE.le (f a) (g a)))
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict A) fun x => (fun a => Max.max (f  …
  -/
  have h₂ := hA.inter (measurableSet_lt hg.1 hf.1)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    h₁ : MeasurableSet (Inter.inter A (setOf fun a => LE.le (f a) (g a)))
    h₂ : MeasurableSet (Inter.inter A (setOf fun a => LT.lt (g a) (f a)))
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict A) fun x => (fun a => Max.max (f  …
  -/
  rw [setLIntegral_max hf.1 hg.1]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    h₁ : MeasurableSet (Inter.inter A (setOf fun a => LE.le (f a) (g a)))
    h₂ : MeasurableSet (Inter.inter A (setOf fun a => LT.lt (g a) (f a)))
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict (Inter.inter A (setOf  …
  -/
  refine (add_le_add (hg.2 _ h₁) (hf.2 _ h₂)).trans_eq ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    h₁ : MeasurableSet (Inter.inter A (setOf fun a => LE.le (f a) (g a)))
    h₂ : MeasurableSet (Inter.inter A (setOf fun a => LT.lt (g a) (f a)))
    ⊢ Eq (HAdd.hAdd (ν (Inter.inter A (setOf fun a => LE.le (f a) (g a)))) (ν (Int …
  -/
  simp only [← not_le, ← compl_setOf, ← diff_eq]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    hg : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
    A : Set α
    hA : MeasurableSet A
    h₁ : MeasurableSet (Inter.inter A (setOf fun a => LE.le (f a) (g a)))
    h₂ : MeasurableSet (Inter.inter A (setOf fun a => LT.lt (g a) (f a)))
    ⊢ Eq (HAdd.hAdd (ν (Inter.inter A (setOf fun a => LE.le (f a) (g a)))) (ν (SDi …
  -/
  exact measure_inter_add_diff _ (measurableSet_le hf.1 hg.1)
  /-
    🎉 no goals
  -/


theorem iSup_succ_eq_sup {α} (f : ℕ → α → ℝ≥0∞) (m : ℕ) (a : α) :
    ⨆ (k : ℕ) (_ : k ≤ m + 1), f k a = f m.succ a ⊔ ⨆ (k : ℕ) (_ : k ≤ m), f k a := by
  /-
    α : Sort u_2
    f : Nat → α → ENNReal
    m : Nat
    a : α
    ⊢ Eq (iSup fun k => iSup fun x => f k a) (Max.max (f m.succ a) (iSup fun k =>  …
  -/
  set c := ⨆ (k : ℕ) (_ : k ≤ m + 1), f k a with hc
  /-
    α : Sort u_2
    f : Nat → α → ENNReal
    m : Nat
    a : α
    c : ENNReal := iSup fun k => iSup fun x => f k a
    hc : Eq c (iSup fun k => iSup fun x => f k a)
    ⊢ Eq c (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
  -/
  set d := f m.succ a ⊔ ⨆ (k : ℕ) (_ : k ≤ m), f k a with hd
  /-
    α : Sort u_2
    f : Nat → α → ENNReal
    m : Nat
    a : α
    c : ENNReal := iSup fun k => iSup fun x => f k a
    hc : Eq c (iSup fun k => iSup fun x => f k a)
    d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
    hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
    ⊢ Eq c d
  -/
  rw [le_antisymm_iff, hc, hd]
  /-
    α : Sort u_2
    f : Nat → α → ENNReal
    m : Nat
    a : α
    c : ENNReal := iSup fun k => iSup fun x => f k a
    hc : Eq c (iSup fun k => iSup fun x => f k a)
    d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
    hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
    ⊢ And (LE.le (iSup fun k => iSup fun x => f k a) (Max.max (f m.succ a) (iSup f …
  -/
  constructor
    /-
      case left
      α : Sort u_2
      f : Nat → α → ENNReal
      m : Nat
      a : α
      c : ENNReal := iSup fun k => iSup fun x => f k a
      hc : Eq c (iSup fun k => iSup fun x => f k a)
      d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
      hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      ⊢ LE.le (iSup fun k => iSup fun x => f k a) (Max.max (f m.succ a) (iSup fun k  …
    -/
  · refine iSup₂_le fun n hn ↦ ?_
    /-
      case left
      α : Sort u_2
      f : Nat → α → ENNReal
      m : Nat
      a : α
      c : ENNReal := iSup fun k => iSup fun x => f k a
      hc : Eq c (iSup fun k => iSup fun x => f k a)
      d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
      hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      n : Nat
      hn : LE.le n (HAdd.hAdd m 1)
      ⊢ LE.le (f n a) (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
    -/
    rcases Nat.of_le_succ hn with (h | h)
      /-
        case left.inl
        α : Sort u_2
        f : Nat → α → ENNReal
        m : Nat
        a : α
        c : ENNReal := iSup fun k => iSup fun x => f k a
        hc : Eq c (iSup fun k => iSup fun x => f k a)
        d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
        hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
        n : Nat
        hn : LE.le n (HAdd.hAdd m 1)
        h : LE.le n m
        ⊢ LE.le (f n a) (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      -/
    · exact le_sup_of_le_right (le_iSup₂ (f := fun k (_ : k ≤ m) ↦ f k a) n h)
      /-
        🎉 no goals
      -/
      /-
        case left.inr
        α : Sort u_2
        f : Nat → α → ENNReal
        m : Nat
        a : α
        c : ENNReal := iSup fun k => iSup fun x => f k a
        hc : Eq c (iSup fun k => iSup fun x => f k a)
        d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
        hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
        n : Nat
        hn : LE.le n (HAdd.hAdd m 1)
        h : Eq n m.succ
        ⊢ LE.le (f n a) (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      -/
    · exact h ▸ le_sup_left
      /-
        🎉 no goals
      -/
    /-
      case right
      α : Sort u_2
      f : Nat → α → ENNReal
      m : Nat
      a : α
      c : ENNReal := iSup fun k => iSup fun x => f k a
      hc : Eq c (iSup fun k => iSup fun x => f k a)
      d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
      hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      ⊢ LE.le (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)) (iSup fun k …
    -/
  · refine sup_le ?_ (biSup_mono fun n hn ↦ hn.trans m.le_succ)
    /-
      case right
      α : Sort u_2
      f : Nat → α → ENNReal
      m : Nat
      a : α
      c : ENNReal := iSup fun k => iSup fun x => f k a
      hc : Eq c (iSup fun k => iSup fun x => f k a)
      d : ENNReal := Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a)
      hd : Eq d (Max.max (f m.succ a) (iSup fun k => iSup fun x => f k a))
      ⊢ LE.le (f m.succ a) (iSup fun k => iSup fun x => f k a)
    -/
    exact @le_iSup₂ ℝ≥0∞ ℕ (fun i ↦ i ≤ m + 1) _ _ (m + 1) le_rfl
    /-
      🎉 no goals
    -/


theorem iSup_mem_measurableLE (f : ℕ → α → ℝ≥0∞) (hf : ∀ n, f n ∈ measurableLE μ ν) (n : ℕ) :
    (fun x ↦ ⨆ (k) (_ : k ≤ n), f k x) ∈ measurableLE μ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
    n : Nat
    ⊢ Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE μ ν …
  -/
  induction' n with m hm
    /-
      case zero
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
      ⊢ Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE μ ν …
    -/
  · constructor
      /-
        case zero.left
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        f : Nat → α → ENNReal
        hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
        ⊢ Measurable fun x => iSup fun k => iSup fun x_1 => f k x
      -/
    · simp [(hf 0).1]
      /-
        🎉 no goals
      -/
      /-
        case zero.right
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        f : Nat → α → ENNReal
        hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
        ⊢ ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (μ.restrict  …
      -/
    · intro A hA; simp [(hf 0).2 A hA]
                  /-
                    🎉 no goals
                  -/
  · have :
      (fun a : α ↦ ⨆ (k : ℕ) (_ : k ≤ m + 1), f k a) = fun a ↦
        f m.succ a ⊔ ⨆ (k : ℕ) (_ : k ≤ m), f k a :=
      funext fun _ ↦ iSup_succ_eq_sup _ _ _
    /-
      case succ
      α : Type u_1
      m✝ : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
      m : Nat
      hm : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
      this : Eq (fun a => iSup fun k => iSup fun x => f k a) fun a => Max.max (f m.s …
      ⊢ Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE μ ν …
    -/
    refine ⟨.iSup fun n ↦ Measurable.iSup_Prop _ (hf n).1, fun A hA ↦ ?_⟩
    /-
      case succ
      α : Type u_1
      m✝ : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
      m : Nat
      hm : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE  …
      this : Eq (fun a => iSup fun k => iSup fun x => f k a) fun a => Max.max (f m.s …
      A : Set α
      hA : MeasurableSet A
      ⊢ LE.le (MeasureTheory.lintegral (μ.restrict A) fun x => (fun x => iSup fun k  …
    -/
    rw [this]; exact (sup_mem_measurableLE (hf m.succ) hm).2 A hA
               /-
                 🎉 no goals
               -/


theorem iSup_mem_measurableLE' (f : ℕ → α → ℝ≥0∞) (hf : ∀ n, f n ∈ measurableLE μ ν) (n : ℕ) :
    (⨆ (k) (_ : k ≤ n), f k) ∈ measurableLE μ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
    n : Nat
    ⊢ Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE μ ν …
  -/
  convert iSup_mem_measurableLE f hf n
  /-
    case h.e'_5.h
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition. …
    n : Nat
    x✝ : α
    ⊢ Eq (iSup (fun k => iSup fun x => f k) x✝) (iSup fun k => iSup fun x => f k x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iSup_monotone {α : Type*} (f : ℕ → α → ℝ≥0∞) :
    Monotone fun n x ↦ ⨆ (k) (_ : k ≤ n), f k x :=
  fun _ _ hnm _ ↦ biSup_mono fun _ ↦ ge_trans hnm


theorem iSup_monotone' {α : Type*} (f : ℕ → α → ℝ≥0∞) (x : α) :
    Monotone fun n ↦ ⨆ (k) (_ : k ≤ n), f k x := fun _ _ hnm ↦ iSup_monotone f hnm x


theorem iSup_le_le {α : Type*} (f : ℕ → α → ℝ≥0∞) (n k : ℕ) (hk : k ≤ n) :
    f k ≤ fun x ↦ ⨆ (k) (_ : k ≤ n), f k x :=
  fun x ↦ le_iSup₂ (f := fun k (_ : k ≤ n) ↦ f k x) k hk


/-- `measurableLEEval μ ν` is the set of `∫⁻ x, f x ∂μ` for all `f ∈ measurableLE μ ν`. -/
def measurableLEEval (μ ν : Measure α) : Set ℝ≥0∞ :=
  (fun f : α → ℝ≥0∞ ↦ ∫⁻ x, f x ∂μ) '' measurableLE μ ν


/-- Any pair of finite measures `μ` and `ν`, `HaveLebesgueDecomposition`. That is to say,
there exist a measure `ξ` and a measurable function `f`, such that `ξ` is mutually singular
with respect to `ν` and `μ = ξ + ν.withDensity f`.

This is not an instance since this is also shown for the more general σ-finite measures with
`MeasureTheory.Measure.haveLebesgueDecomposition_of_sigmaFinite`. -/
theorem haveLebesgueDecomposition_of_finiteMeasure [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    HaveLebesgueDecomposition μ ν where
  lebesgue_decomposition := by
    have h := @exists_seq_tendsto_sSup _ _ _ _ _ (measurableLEEval ν μ)
      ⟨0, 0, zero_mem_measurableLE, by simp⟩ (OrderTop.bddAbove _)
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      h : Exists fun u => And (Monotone u) (And (Filter.Tendsto u Filter.atTop (nhds …
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq μ (HA …
    -/
    choose g _ hg₂ f hf₁ hf₂ using h
    -- we set `ξ` to be the supremum of an increasing sequence of functions obtained from above
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      g : Nat → ENNReal
      h✝ : Monotone g
      hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
      f : Nat → α → ENNReal
      hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
      hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq μ (HA …
    -/
    set ξ := ⨆ (n) (k) (_ : k ≤ n), f k with hξ
    -- we see that `ξ` has the largest integral among all functions in `measurableLE`
    have hξ₁ : sSup (measurableLEEval ν μ) = ∫⁻ a, ξ a ∂ν := by
      have := @lintegral_tendsto_of_tendsto_of_monotone _ _ ν (fun n ↦ ⨆ (k) (_ : k ≤ n), f k)
          (⨆ (n) (k) (_ : k ≤ n), f k) ?_ ?_ ?_
      · refine tendsto_nhds_unique ?_ this
        refine tendsto_of_tendsto_of_tendsto_of_le_of_le hg₂ tendsto_const_nhds (fun n ↦ ?_)
          fun n ↦ ?_
        · rw [← hf₂ n]
          apply lintegral_mono
          convert iSup_le_le f n n le_rfl
          simp only [iSup_apply]
        · exact le_sSup ⟨⨆ (k : ℕ) (_ : k ≤ n), f k, iSup_mem_measurableLE' _ hf₁ _, rfl⟩
      · intro n
        refine Measurable.aemeasurable ?_
        convert (iSup_mem_measurableLE _ hf₁ n).1
        simp
      · refine Filter.Eventually.of_forall fun a ↦ ?_
        simp [iSup_monotone' f _]
      · refine Filter.Eventually.of_forall fun a ↦ ?_
        simp [tendsto_atTop_iSup (iSup_monotone' f a)]
    have hξm : Measurable ξ := by
      convert Measurable.iSup fun n ↦ (iSup_mem_measurableLE _ hf₁ n).1
      simp [hξ]
    -- we see that `ξ` has the largest integral among all functions in `measurableLE`
    have hξle A (hA : MeasurableSet A) : ∫⁻ a in A, ξ a ∂ν ≤ μ A := by
        rw [hξ]
        simp_rw [iSup_apply]
        rw [lintegral_iSup (fun n ↦ (iSup_mem_measurableLE _ hf₁ n).1) (iSup_monotone _)]
        exact iSup_le fun n ↦ (iSup_mem_measurableLE _ hf₁ n).2 A hA
    have hle : ν.withDensity ξ ≤ μ := by
      refine le_intro fun B hB _ ↦ ?_
      rw [withDensity_apply _ hB]
      exact hξle B hB
    have : IsFiniteMeasure (ν.withDensity ξ) := by
      refine isFiniteMeasure_withDensity ?_
      have hle' := hle univ
      rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ] at hle'
      exact ne_top_of_le_ne_top (measure_ne_top _ _) hle'
    -- `ξ` is the `f` in the theorem statement and we set `μ₁` to be `μ - ν.withDensity ξ`
    -- since we need `μ₁ + ν.withDensity ξ = μ`
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      g : Nat → ENNReal
      h✝ : Monotone g
      hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
      f : Nat → α → ENNReal
      hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
      hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
      ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
      hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
      hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
      hξm : Measurable ξ
      hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
      hle : LE.le (ν.withDensity ξ) μ
      this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq μ (HA …
    -/
    set μ₁ := μ - ν.withDensity ξ with hμ₁
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      g : Nat → ENNReal
      h✝ : Monotone g
      hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
      f : Nat → α → ENNReal
      hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
      hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
      ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
      hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
      hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
      hξm : Measurable ξ
      hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
      hle : LE.le (ν.withDensity ξ) μ
      this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
      μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
      hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
      ⊢ Exists fun p => And (Measurable p.2) (And (p.1.MutuallySingular ν) (Eq μ (HA …
    -/
    refine ⟨⟨μ₁, ξ⟩, hξm, ?_, ?_⟩
      /-
        case refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        ⊢ { fst := μ₁, snd := ξ }.1.MutuallySingular ν
      -/
    · by_contra h
      -- if they are not mutually singular, then from `exists_positive_of_not_mutuallySingular`,
      -- there exists some `ε > 0` and a measurable set `E`, such that `μ(E) > 0` and `E` is
      -- positive with respect to `ν - εμ`
      /-
        case refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ⊢ False
      -/
      obtain ⟨ε, hε₁, E, hE₁, hE₂, hE₃⟩ := exists_positive_of_not_mutuallySingular μ₁ ν h
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        ⊢ False
      -/
      simp_rw [hμ₁] at hE₃
      -- since `E` is positive, we have `∫⁻ a in A ∩ E, ε + ξ a ∂ν ≤ μ (A ∩ E)` for all `A`
      have hε₂ (A : Set α) (hA : MeasurableSet A) : ∫⁻ a in A ∩ E, ε + ξ a ∂ν ≤ μ (A ∩ E) := by
        specialize hE₃ A hA
        rw [lintegral_add_left measurable_const, lintegral_const, restrict_apply_univ]
        rw [Measure.sub_apply (hA.inter hE₁) hle, withDensity_apply _ (hA.inter hE₁)] at hE₃
        refine add_le_of_le_tsub_right_of_le (hξle _ (hA.inter hE₁)) hE₃
      -- from this, we can show `ξ + ε * E.indicator` is a function in `measurableLE` with
      -- integral greater than `ξ`
      have hξε : (ξ + E.indicator fun _ ↦ (ε : ℝ≥0∞)) ∈ measurableLE ν μ := by
        refine ⟨hξm.add (measurable_const.indicator hE₁), fun A hA ↦ ?_⟩
        have : ∫⁻ a in A, (ξ + E.indicator fun _ ↦ (ε : ℝ≥0∞)) a ∂ν =
            ∫⁻ a in A ∩ E, ε + ξ a ∂ν + ∫⁻ a in A \ E, ξ a ∂ν := by
          simp only [lintegral_add_left measurable_const, lintegral_add_left hξm,
            setLIntegral_const, add_assoc, lintegral_inter_add_diff _ _ hE₁, Pi.add_apply,
            lintegral_indicator hE₁, restrict_apply hE₁]
          rw [inter_comm, add_comm]
        rw [this, ← measure_inter_add_diff A hE₁]
        exact add_le_add (hε₂ A hA) (hξle (A \ E) (hA.diff hE₁))
      have : (∫⁻ a, ξ a + E.indicator (fun _ ↦ (ε : ℝ≥0∞)) a ∂ν) ≤ sSup (measurableLEEval ν μ) :=
        le_sSup ⟨ξ + E.indicator fun _ ↦ (ε : ℝ≥0∞), hξε, rfl⟩
      -- but this contradicts the maximality of `∫⁻ x, ξ x ∂ν`
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this✝ : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        hε₂ : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.restr …
        hξε : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
        this : LE.le (MeasureTheory.lintegral ν fun a => HAdd.hAdd (ξ a) (E.indicator  …
        ⊢ False
      -/
      refine not_lt.2 this ?_
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this✝ : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        hε₂ : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.restr …
        hξε : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
        this : LE.le (MeasureTheory.lintegral ν fun a => HAdd.hAdd (ξ a) (E.indicator  …
        ⊢ LT.lt (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
      -/
      rw [hξ₁, lintegral_add_left hξm, lintegral_indicator hE₁, setLIntegral_const]
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this✝ : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        hε₂ : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.restr …
        hξε : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
        this : LE.le (MeasureTheory.lintegral ν fun a => HAdd.hAdd (ξ a) (E.indicator  …
        ⊢ LT.lt (MeasureTheory.lintegral ν fun a => ξ a) (HAdd.hAdd (MeasureTheory.lin …
      -/
      refine ENNReal.lt_add_right ?_ (ENNReal.mul_pos_iff.2 ⟨ENNReal.coe_pos.2 hε₁, hE₂⟩).ne'
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this✝ : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        hε₂ : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.restr …
        hξε : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
        this : LE.le (MeasureTheory.lintegral ν fun a => HAdd.hAdd (ξ a) (E.indicator  …
        ⊢ Ne (MeasureTheory.lintegral ν fun a => ξ a) Top.top
      -/
      have := measure_ne_top (ν.withDensity ξ) univ
      /-
        case refine_1.intro.intro.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this✝¹ : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        h : Not ({ fst := μ₁, snd := ξ }.1.MutuallySingular ν)
        ε : NNReal
        hε₁ : LT.lt 0 ε
        E : Set α
        hE₁ : MeasurableSet E
        hE₂ : LT.lt 0 (ν E)
        hE₃ : ∀ (A : Set α), MeasurableSet A → LE.le (HMul.hMul (↑ε) (ν (Inter.inter A …
        hε₂ : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.restr …
        hξε : Membership.mem (MeasureTheory.Measure.LebesgueDecomposition.measurableLE …
        this✝ : LE.le (MeasureTheory.lintegral ν fun a => HAdd.hAdd (ξ a) (E.indicator …
        this : Ne ((ν.withDensity ξ) Set.univ) Top.top
        ⊢ Ne (MeasureTheory.lintegral ν fun a => ξ a) Top.top
      -/
      rwa [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ] at this
      /-
        🎉 no goals
      -/
    -- since `ν.withDensity ξ ≤ μ`, it is clear that `μ = μ₁ + ν.withDensity ξ`
      /-
        case refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        ⊢ Eq μ (HAdd.hAdd { fst := μ₁, snd := ξ }.1 (ν.withDensity { fst := μ₁, snd := …
      -/
    · rw [hμ₁]
      /-
        case refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        g : Nat → ENNReal
        h✝ : Monotone g
        hg₂ : Filter.Tendsto g Filter.atTop (nhds (SupSet.sSup (MeasureTheory.Measure. …
        f : Nat → α → ENNReal
        hf₁ : ∀ (n : Nat), Membership.mem (MeasureTheory.Measure.LebesgueDecomposition …
        hf₂ : ∀ (n : Nat), Eq ((fun f => MeasureTheory.lintegral ν fun x => f x) (f n) …
        ξ : α → ENNReal := iSup fun n => iSup fun k => iSup fun x => f k
        hξ : Eq ξ (iSup fun n => iSup fun k => iSup fun x => f k)
        hξ₁ : Eq (SupSet.sSup (MeasureTheory.Measure.LebesgueDecomposition.measurableL …
        hξm : Measurable ξ
        hξle : ∀ (A : Set α), MeasurableSet A → LE.le (MeasureTheory.lintegral (ν.rest …
        hle : LE.le (ν.withDensity ξ) μ
        this : MeasureTheory.IsFiniteMeasure (ν.withDensity ξ)
        μ₁ : MeasureTheory.Measure α := HSub.hSub μ (ν.withDensity ξ)
        hμ₁ : Eq μ₁ (HSub.hSub μ (ν.withDensity ξ))
        ⊢ Eq μ (HAdd.hAdd { fst := HSub.hSub μ (ν.withDensity ξ), snd := ξ }.1 (ν.with …
      -/
      ext1 A hA
      rw [Measure.coe_add, Pi.add_apply, Measure.sub_apply hA hle, add_comm,
        add_tsub_cancel_of_le (hle A)]


/-- If any finite measure has a Lebesgue decomposition with respect to `ν`,
then the same is true for any s-finite measure. -/
theorem HaveLebesgueDecomposition.sfinite_of_isFiniteMeasure [SFinite μ]
    (_h : ∀ (μ : Measure α) [IsFiniteMeasure μ], HaveLebesgueDecomposition μ ν) :
    HaveLebesgueDecomposition μ ν :=
  sum_sfiniteSeq μ ▸ sum_left _


variable (μ ν) in
/-- **The Lebesgue decomposition theorem**:
Any s-finite measure `μ` has Lebesgue decomposition with respect to any σ-finite measure `ν`.
That is to say, there exist a measure `ξ` and a measurable function `f`,
such that `ξ` is mutually singular with respect to `ν` and `μ = ξ + ν.withDensity f` -/
nonrec instance (priority := 100) haveLebesgueDecomposition_of_sigmaFinite
    [SFinite μ] [SigmaFinite ν] : HaveLebesgueDecomposition μ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  wlog hμ : IsFiniteMeasure μ generalizing μ
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this : ∀ (μ : MeasureTheory.Measure α) [inst : MeasureTheory.SFinite μ], Measu …
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ μ.HaveLebesgueDecomposition ν
    -/
  · exact .sfinite_of_isFiniteMeasure fun μ _ ↦ this μ ‹_›
    /-
      🎉 no goals
    -/
  -- Take a disjoint cover that consists of sets of finite measure `ν`.
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  set s : ℕ → Set α := disjointed (spanningSets ν)
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets ν)
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  have hsm : ∀ n, MeasurableSet (s n) := .disjointed <| measurableSet_spanningSets _
  have hs : ∀ n, Fact (ν (s n) < ⊤) := fun n ↦
    ⟨lt_of_le_of_lt (measure_mono <| disjointed_le ..) (measure_spanningSets_lt_top ν n)⟩
  -- Note that the restrictions of `μ` and `ν` to `s n` are finite measures.
  -- Therefore, as we proved above, these restrictions have a Lebesgue decomposition.
  -- Let `ξ n` and `f n` be the singular part and the Radon-Nikodym derivative
  -- of these restrictions.
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets ν)
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ∀ (n : Nat), Fact (LT.lt (ν (s n)) Top.top)
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  set ξ : ℕ → Measure α := fun n : ℕ ↦ singularPart (.restrict μ (s n)) (.restrict ν (s n))
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets ν)
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ∀ (n : Nat), Fact (LT.lt (ν (s n)) Top.top)
    ξ : Nat → MeasureTheory.Measure α := fun n => (μ.restrict (s n)).singularPart  …
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  set f : ℕ → α → ℝ≥0∞ := fun n ↦ (s n).indicator (rnDeriv (.restrict μ (s n)) (.restrict ν (s n)))
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets ν)
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ∀ (n : Nat), Fact (LT.lt (ν (s n)) Top.top)
    ξ : Nat → MeasureTheory.Measure α := fun n => (μ.restrict (s n)).singularPart  …
    f : Nat → α → ENNReal := fun n => (s n).indicator ((μ.restrict (s n)).rnDeriv  …
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  have hfm (n : ℕ) : Measurable (f n) := by measurability
  -- Each `ξ n` is supported on `s n` and is mutually singular with the restriction of `ν` to `s n`.
  -- Therefore, `ξ n` is mutually singular with `ν`, hence their sum is mutually singular with `ν`.
  have hξ : .sum ξ ⟂ₘ ν := by
    refine MutuallySingular.sum_left.2 fun n ↦ ?_
    rw [← ν.restrict_add_restrict_compl (hsm n)]
    refine (mutuallySingular_singularPart ..).add_right (.singularPart ?_ _)
    refine ⟨(s n)ᶜ, (hsm n).compl, ?_⟩
    simp [hsm]
  -- Finally, the sum of all `ξ n` and measure `ν` with the density `∑' n, f n`
  -- is equal to `μ`, thus `(Measure.sum ξ, ∑' n, f n)` is a Lebesgue decomposition for `μ` and `ν`.
  have hadd : .sum ξ + ν.withDensity (∑' n, f n) = μ := calc
    .sum ξ + ν.withDensity (∑' n, f n) = .sum fun n ↦ ξ n + ν.withDensity (f n) := by
      rw [withDensity_tsum hfm, Measure.sum_add_sum]
    _ = .sum fun n ↦ .restrict μ (s n) := by
      simp_rw [ξ, f, withDensity_indicator (hsm _), singularPart_add_rnDeriv]
    _ = μ := sum_restrict_disjointed_spanningSets ..
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    hμ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets ν)
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ∀ (n : Nat), Fact (LT.lt (ν (s n)) Top.top)
    ξ : Nat → MeasureTheory.Measure α := fun n => (μ.restrict (s n)).singularPart  …
    f : Nat → α → ENNReal := fun n => (s n).indicator ((μ.restrict (s n)).rnDeriv  …
    hfm : ∀ (n : Nat), Measurable (f n)
    hξ : (MeasureTheory.Measure.sum ξ).MutuallySingular ν
    hadd : Eq (HAdd.hAdd (MeasureTheory.Measure.sum ξ) (ν.withDensity (tsum fun n  …
    ⊢ μ.HaveLebesgueDecomposition ν
  -/
  exact ⟨⟨(.sum ξ, ∑' n, f n), by measurability, hξ, hadd.symm⟩⟩
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative of the scalar multiple of a measure.
See also `rnDeriv_smul_left`, which has no hypothesis on `μ` but requires finite `ν`. -/
theorem rnDeriv_smul_left' (ν μ : Measure α) [SigmaFinite ν] [SigmaFinite μ] (r : ℝ≥0) :
    (r • ν).rnDeriv μ =ᵐ[μ] r • ν.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite μ
    r : NNReal
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r ν).rnDeriv μ) (HSMul.hSMul …
  -/
  rw [← withDensity_eq_iff_of_sigmaFinite]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      ⊢ Eq (μ.withDensity ((HSMul.hSMul r ν).rnDeriv μ)) (μ.withDensity (HSMul.hSMul …
    -/
  · simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      ⊢ Eq (μ.withDensity ((HSMul.hSMul (↑r) ν).rnDeriv μ)) (μ.withDensity (HSMul.hS …
    -/
    rw [withDensity_smul _ (measurable_rnDeriv _ _)]
    suffices (r • ν).singularPart μ + withDensity μ (rnDeriv (r • ν) μ)
        = (r • ν).singularPart μ + r • withDensity μ (rnDeriv ν μ) by
      rwa [Measure.add_right_inj] at this
    rw [← (r • ν).haveLebesgueDecomposition_add μ, singularPart_smul, ← smul_add,
      ← ν.haveLebesgueDecomposition_add μ]
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      ⊢ AEMeasurable ((HSMul.hSMul r ν).rnDeriv μ) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      ⊢ AEMeasurable (HSMul.hSMul r (ν.rnDeriv μ)) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable.const_smul _
    /-
      🎉 no goals
    -/


/-- Radon-Nikodym derivative of the scalar multiple of a measure.
See also `rnDeriv_smul_left_of_ne_top`, which has no hypothesis on `μ` but requires finite `ν`. -/
theorem rnDeriv_smul_left_of_ne_top' (ν μ : Measure α) [SigmaFinite ν] [SigmaFinite μ]
    {r : ℝ≥0∞} (hr : r ≠ ∞) :
    (r • ν).rnDeriv μ =ᵐ[μ] r • ν.rnDeriv μ := by
  have h : (r.toNNReal • ν).rnDeriv μ =ᵐ[μ] r.toNNReal • ν.rnDeriv μ :=
    rnDeriv_smul_left' ν μ r.toNNReal
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite μ
    r : ENNReal
    hr : Ne r Top.top
    h : (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r.toNNReal ν).rnDeriv μ) ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HSMul.hSMul r ν).rnDeriv μ) (HSMul.hSMul …
  -/
  simpa [ENNReal.smul_def, ENNReal.coe_toNNReal hr] using h
  /-
    🎉 no goals
  -/


/-- Radon-Nikodym derivative with respect to the scalar multiple of a measure.
See also `rnDeriv_smul_right`, which has no hypothesis on `μ` but requires finite `ν`. -/
theorem rnDeriv_smul_right' (ν μ : Measure α) [SigmaFinite ν] [SigmaFinite μ]
    {r : ℝ≥0} (hr : r ≠ 0) :
    ν.rnDeriv (r • μ) =ᵐ[μ] r⁻¹ • ν.rnDeriv μ := by
  refine (absolutelyContinuous_smul <| ENNReal.coe_ne_zero.2 hr).ae_le
    (?_ : ν.rnDeriv (r • μ) =ᵐ[r • μ] r⁻¹ • ν.rnDeriv μ)
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite μ
    r : NNReal
    hr : Ne r 0
    ⊢ (MeasureTheory.ae (HSMul.hSMul r μ)).EventuallyEq (ν.rnDeriv (HSMul.hSMul r  …
  -/
  rw [← withDensity_eq_iff_of_sigmaFinite]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ((HSMul.hSMul r μ).withDensity (ν.rnDeriv (HSMul.hSMul r μ))) ((HSMul.hSM …
    -/
  · simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ((HSMul.hSMul (↑r) μ).withDensity (ν.rnDeriv (HSMul.hSMul (↑r) μ))) ((HSM …
    -/
    rw [withDensity_smul _ (measurable_rnDeriv _ _)]
    suffices ν.singularPart (r • μ) + withDensity (r • μ) (rnDeriv ν (r • μ))
        = ν.singularPart (r • μ) + r⁻¹ • withDensity (r • μ) (rnDeriv ν μ) by
      rwa [add_right_inj] at this
    rw [← ν.haveLebesgueDecomposition_add (r • μ), singularPart_smul_right _ _ _ hr,
      ENNReal.smul_def r, withDensity_smul_measure, ← ENNReal.smul_def, ← smul_assoc,
      smul_eq_mul, inv_mul_cancel₀ hr, one_smul]
    /-
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      hr : Ne r 0
      ⊢ Eq ν (HAdd.hAdd (ν.singularPart μ) (μ.withDensity (ν.rnDeriv μ)))
    -/
    exact ν.haveLebesgueDecomposition_add μ
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      hr : Ne r 0
      ⊢ AEMeasurable (ν.rnDeriv (HSMul.hSMul r μ)) (HSMul.hSMul r μ)
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite μ
      r : NNReal
      hr : Ne r 0
      ⊢ AEMeasurable (HSMul.hSMul (Inv.inv r) (ν.rnDeriv μ)) (HSMul.hSMul r μ)
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable.const_smul _
    /-
      🎉 no goals
    -/


/-- Radon-Nikodym derivative with respect to the scalar multiple of a measure.
See also `rnDeriv_smul_right_of_ne_top`, which has no hypothesis on `μ` but requires finite `ν`. -/
theorem rnDeriv_smul_right_of_ne_top' (ν μ : Measure α) [SigmaFinite ν] [SigmaFinite μ]
    {r : ℝ≥0∞} (hr : r ≠ 0) (hr_ne_top : r ≠ ∞) :
    ν.rnDeriv (r • μ) =ᵐ[μ] r⁻¹ • ν.rnDeriv μ := by
  have h : ν.rnDeriv (r.toNNReal • μ) =ᵐ[μ] r.toNNReal⁻¹ • ν.rnDeriv μ := by
    refine rnDeriv_smul_right' ν μ ?_
    rw [ne_eq, ENNReal.toNNReal_eq_zero_iff]
    simp [hr, hr_ne_top]
  rwa [ENNReal.smul_def, ENNReal.coe_toNNReal hr_ne_top,
    ← ENNReal.toNNReal_inv, ENNReal.smul_def, ENNReal.coe_toNNReal (ENNReal.inv_ne_top.mpr hr)] at h


/-- Radon-Nikodym derivative of a sum of two measures.
See also `rnDeriv_add`, which has no hypothesis on `μ` but requires finite `ν₁` and `ν₂`. -/
lemma rnDeriv_add' (ν₁ ν₂ μ : Measure α) [SigmaFinite ν₁] [SigmaFinite ν₂] [SigmaFinite μ] :
    (ν₁ + ν₂).rnDeriv μ =ᵐ[μ] ν₁.rnDeriv μ + ν₂.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν₁ ν₂ μ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite ν₁
    inst✝¹ : MeasureTheory.SigmaFinite ν₂
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HAdd.hAdd ν₁ ν₂).rnDeriv μ) (HAdd.hAdd ( …
  -/
  rw [← withDensity_eq_iff_of_sigmaFinite]
  · suffices (ν₁ + ν₂).singularPart μ + μ.withDensity ((ν₁ + ν₂).rnDeriv μ)
        = (ν₁ + ν₂).singularPart μ + μ.withDensity (ν₁.rnDeriv μ + ν₂.rnDeriv μ) by
      rwa [add_right_inj] at this
    rw [← (ν₁ + ν₂).haveLebesgueDecomposition_add μ, singularPart_add,
      withDensity_add_left (measurable_rnDeriv _ _), add_assoc,
      add_comm (ν₂.singularPart μ), add_assoc, add_comm _ (ν₂.singularPart μ),
      ← ν₂.haveLebesgueDecomposition_add μ, ← add_assoc, ← ν₁.haveLebesgueDecomposition_add μ]
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      ν₁ ν₂ μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite ν₁
      inst✝¹ : MeasureTheory.SigmaFinite ν₂
      inst✝ : MeasureTheory.SigmaFinite μ
      ⊢ AEMeasurable ((HAdd.hAdd ν₁ ν₂).rnDeriv μ) μ
    -/
  · exact (measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      ν₁ ν₂ μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite ν₁
      inst✝¹ : MeasureTheory.SigmaFinite ν₂
      inst✝ : MeasureTheory.SigmaFinite μ
      ⊢ AEMeasurable (HAdd.hAdd (ν₁.rnDeriv μ) (ν₂.rnDeriv μ)) μ
    -/
  · exact ((measurable_rnDeriv _ _).add (measurable_rnDeriv _ _)).aemeasurable
    /-
      🎉 no goals
    -/


lemma rnDeriv_add_of_mutuallySingular (ν₁ ν₂ μ : Measure α)
    [SigmaFinite ν₁] [SigmaFinite ν₂] [SigmaFinite μ] (h : ν₂ ⟂ₘ μ) :
    (ν₁ + ν₂).rnDeriv μ =ᵐ[μ] ν₁.rnDeriv μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ν₁ ν₂ μ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite ν₁
    inst✝¹ : MeasureTheory.SigmaFinite ν₂
    inst✝ : MeasureTheory.SigmaFinite μ
    h : ν₂.MutuallySingular μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HAdd.hAdd ν₁ ν₂).rnDeriv μ) (ν₁.rnDeriv μ)
  -/
  filter_upwards [rnDeriv_add' ν₁ ν₂ μ, (rnDeriv_eq_zero ν₂ μ).mpr h] with x hx_add hx_zero
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    ν₁ ν₂ μ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite ν₁
    inst✝¹ : MeasureTheory.SigmaFinite ν₂
    inst✝ : MeasureTheory.SigmaFinite μ
    h : ν₂.MutuallySingular μ
    x : α
    hx_add : Eq ((HAdd.hAdd ν₁ ν₂).rnDeriv μ x) (HAdd.hAdd (ν₁.rnDeriv μ) (ν₂.rnDe …
    hx_zero : Eq (ν₂.rnDeriv μ x) (0 x)
    ⊢ Eq ((HAdd.hAdd ν₁ ν₂).rnDeriv μ x) (ν₁.rnDeriv μ x)
  -/
  simp [hx_add, hx_zero]
  /-
    🎉 no goals
  -/


