/-- The binary product of finite measures. -/
noncomputable def prod (μ : FiniteMeasure α) (ν : FiniteMeasure β) : FiniteMeasure (α × β) :=
  ⟨μ.toMeasure.prod ν.toMeasure, inferInstance⟩


@[simp] lemma toMeasure_prod : (μ.prod ν).toMeasure = μ.toMeasure.prod ν.toMeasure := rfl


lemma prod_apply (s : Set (α × β)) (s_mble : MeasurableSet s) :
    μ.prod ν s = ENNReal.toNNReal (∫⁻ x, ν.toMeasure (Prod.mk x ⁻¹' s) ∂μ) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    s : Set (Prod α β)
    s_mble : MeasurableSet s
    ⊢ Eq ((μ.prod ν) s) (MeasureTheory.lintegral ↑μ fun x => ↑ν (Set.preimage (Pro …
  -/
  simp [coeFn_def, Measure.prod_apply s_mble]
  /-
    🎉 no goals
  -/


lemma prod_apply_symm (s : Set (α × β)) (s_mble : MeasurableSet s) :
    μ.prod ν s = ENNReal.toNNReal (∫⁻ y, μ.toMeasure ((fun x ↦ ⟨x, y⟩) ⁻¹' s) ∂ν) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    s : Set (Prod α β)
    s_mble : MeasurableSet s
    ⊢ Eq ((μ.prod ν) s) (MeasureTheory.lintegral ↑ν fun y => ↑μ (Set.preimage (fun …
  -/
  simp [coeFn_def, Measure.prod_apply_symm s_mble]
  /-
    🎉 no goals
  -/


                                                                              /-
                                                                                α : Type u_1
                                                                                inst✝¹ : MeasurableSpace α
                                                                                β : Type u_2
                                                                                inst✝ : MeasurableSpace β
                                                                                μ : MeasureTheory.FiniteMeasure α
                                                                                ν : MeasureTheory.FiniteMeasure β
                                                                                s : Set α
                                                                                t : Set β
                                                                                ⊢ Eq ((μ.prod ν) (SProd.sprod s t)) (HMul.hMul (μ s) (ν t))
                                                                              -/
lemma prod_prod (s : Set α) (t : Set β) : μ.prod ν (s ×ˢ t) = μ s * ν t := by simp [coeFn_def]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp] lemma mass_prod : (μ.prod ν).mass = μ.mass * ν.mass := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq (μ.prod ν).mass (HMul.hMul μ.mass ν.mass)
  -/
  simp only [coeFn_def, mass, univ_prod_univ.symm, toMeasure_prod]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq (((↑μ).prod ↑ν) (SProd.sprod Set.univ Set.univ)).toNNReal (HMul.hMul (↑μ  …
  -/
  rw [← ENNReal.toNNReal_mul]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq (((↑μ).prod ↑ν) (SProd.sprod Set.univ Set.univ)).toNNReal (HMul.hMul (↑μ  …
  -/
  exact congr_arg ENNReal.toNNReal (Measure.prod_prod univ univ)
  /-
    🎉 no goals
  -/


@[simp] lemma zero_prod : (0 : FiniteMeasure α).prod ν = 0 := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq (MeasureTheory.FiniteMeasure.prod 0 ν) 0
  -/
  rw [← mass_zero_iff, mass_prod, zero_mass, zero_mul]
  /-
    🎉 no goals
  -/


@[simp] lemma prod_zero : μ.prod (0 : FiniteMeasure β) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ⊢ Eq (μ.prod 0) 0
  -/
  rw [← mass_zero_iff, mass_prod, zero_mass, mul_zero]
  /-
    🎉 no goals
  -/


                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : MeasurableSpace α
                                                                          β : Type u_2
                                                                          inst✝ : MeasurableSpace β
                                                                          μ : MeasureTheory.FiniteMeasure α
                                                                          ν : MeasureTheory.FiniteMeasure β
                                                                          ⊢ Eq ((μ.prod ν).map Prod.fst) (HSMul.hSMul (ν Set.univ) μ)
                                                                        -/
@[simp] lemma map_fst_prod : (μ.prod ν).map Prod.fst = ν univ • μ := by ext; simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : MeasurableSpace α
                                                                          β : Type u_2
                                                                          inst✝ : MeasurableSpace β
                                                                          μ : MeasureTheory.FiniteMeasure α
                                                                          ν : MeasureTheory.FiniteMeasure β
                                                                          ⊢ Eq ((μ.prod ν).map Prod.snd) (HSMul.hSMul (μ Set.univ) ν)
                                                                        -/
@[simp] lemma map_snd_prod : (μ.prod ν).map Prod.snd = μ univ • ν := by ext; simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


lemma map_prod_map {α' : Type*} [MeasurableSpace α'] {β' : Type*} [MeasurableSpace β']
    {f : α → α'} {g : β → β'} (f_mble : Measurable f) (g_mble : Measurable g) :
    (μ.map f).prod (ν.map g) = (μ.prod ν).map (Prod.map f g) := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ((μ.map f).prod (ν.map g)) ((μ.prod ν).map (Prod.map f g))
  -/
  apply Subtype.ext
  /-
    case a
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ↑((μ.map f).prod (ν.map g)) ↑((μ.prod ν).map (Prod.map f g))
  -/
  simp only [val_eq_toMeasure, toMeasure_prod, toMeasure_map]
  /-
    case a
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ((MeasureTheory.Measure.map f ↑μ).prod (MeasureTheory.Measure.map g ↑ν))  …
  -/
  rw [Measure.map_prod_map _ _ f_mble g_mble]
  /-
    🎉 no goals
  -/


lemma prod_swap : (μ.prod ν).map Prod.swap = ν.prod μ := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq ((μ.prod ν).map Prod.swap) (ν.prod μ)
  -/
  apply Subtype.ext
  /-
    case a
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.FiniteMeasure α
    ν : MeasureTheory.FiniteMeasure β
    ⊢ Eq ↑((μ.prod ν).map Prod.swap) ↑(ν.prod μ)
  -/
  simp [Measure.prod_swap]
  /-
    🎉 no goals
  -/


/-- The binary product of probability measures. -/
noncomputable def prod (μ : ProbabilityMeasure α) (ν : ProbabilityMeasure β) :
    ProbabilityMeasure (α × β) :=
                                    /-
                                      α : Type u_1
                                      inst✝¹ : MeasurableSpace α
                                      β : Type u_2
                                      inst✝ : MeasurableSpace β
                                      μ : MeasureTheory.ProbabilityMeasure α
                                      ν : MeasureTheory.ProbabilityMeasure β
                                      ⊢ MeasureTheory.IsProbabilityMeasure ((↑μ).prod ↑ν)
                                    -/
  ⟨μ.toMeasure.prod ν.toMeasure, by infer_instance⟩
                                    /-
                                      🎉 no goals
                                    -/


lemma prod_apply (s : Set (α × β)) (s_mble : MeasurableSet s) :
    μ.prod ν s = ENNReal.toNNReal (∫⁻ x, ν.toMeasure (Prod.mk x ⁻¹' s) ∂μ) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    s : Set (Prod α β)
    s_mble : MeasurableSet s
    ⊢ Eq ((μ.prod ν) s) (MeasureTheory.lintegral ↑μ fun x => ↑ν (Set.preimage (Pro …
  -/
  simp [coeFn_def, Measure.prod_apply s_mble]
  /-
    🎉 no goals
  -/


lemma prod_apply_symm (s : Set (α × β)) (s_mble : MeasurableSet s) :
    μ.prod ν s = ENNReal.toNNReal (∫⁻ y, μ.toMeasure ((fun x ↦ ⟨x, y⟩) ⁻¹' s) ∂ν) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    s : Set (Prod α β)
    s_mble : MeasurableSet s
    ⊢ Eq ((μ.prod ν) s) (MeasureTheory.lintegral ↑ν fun y => ↑μ (Set.preimage (fun …
  -/
  simp [coeFn_def, Measure.prod_apply_symm s_mble]
  /-
    🎉 no goals
  -/


                                                                              /-
                                                                                α : Type u_1
                                                                                inst✝¹ : MeasurableSpace α
                                                                                β : Type u_2
                                                                                inst✝ : MeasurableSpace β
                                                                                μ : MeasureTheory.ProbabilityMeasure α
                                                                                ν : MeasureTheory.ProbabilityMeasure β
                                                                                s : Set α
                                                                                t : Set β
                                                                                ⊢ Eq ((μ.prod ν) (SProd.sprod s t)) (HMul.hMul (μ s) (ν t))
                                                                              -/
lemma prod_prod (s : Set α) (t : Set β) : μ.prod ν (s ×ˢ t) = μ s * ν t := by simp [coeFn_def]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- The first marginal of a product probability measure is the first probability measure. -/
@[simp] lemma map_fst_prod : (μ.prod ν).map measurable_fst.aemeasurable = μ := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    ⊢ Eq ((μ.prod ν).map ⋯) μ
  -/
  apply Subtype.ext
  simp only [val_eq_to_measure, toMeasure_map, toMeasure_prod, Measure.map_fst_prod,
             measure_univ, one_smul]


/-- The second marginal of a product probability measure is the second probability measure. -/
@[simp] lemma map_snd_prod : (μ.prod ν).map measurable_snd.aemeasurable = ν := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    ⊢ Eq ((μ.prod ν).map ⋯) ν
  -/
  apply Subtype.ext
  simp only [val_eq_to_measure, toMeasure_map, toMeasure_prod, Measure.map_snd_prod,
             measure_univ, one_smul]


lemma map_prod_map {α' : Type*} [MeasurableSpace α'] {β' : Type*} [MeasurableSpace β']
    {f : α → α'} {g : β → β'} (f_mble : Measurable f) (g_mble : Measurable g) :
    (μ.map f_mble.aemeasurable).prod (ν.map g_mble.aemeasurable)
      = (μ.prod ν).map (f_mble.prod_map g_mble).aemeasurable := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ((μ.map ⋯).prod (ν.map ⋯)) ((μ.prod ν).map ⋯)
  -/
  apply Subtype.ext
  /-
    case a
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ↑((μ.map ⋯).prod (ν.map ⋯)) ↑((μ.prod ν).map ⋯)
  -/
  simp only [val_eq_to_measure, toMeasure_prod, toMeasure_map]
  /-
    case a
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_2
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    α' : Type u_3
    inst✝¹ : MeasurableSpace α'
    β' : Type u_4
    inst✝ : MeasurableSpace β'
    f : α → α'
    g : β → β'
    f_mble : Measurable f
    g_mble : Measurable g
    ⊢ Eq ((MeasureTheory.Measure.map f ↑μ).prod (MeasureTheory.Measure.map g ↑ν))  …
  -/
  rw [Measure.map_prod_map _ _ f_mble g_mble]
  /-
    🎉 no goals
  -/


lemma prod_swap : (μ.prod ν).map measurable_swap.aemeasurable = ν.prod μ := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    ⊢ Eq ((μ.prod ν).map ⋯) (ν.prod μ)
  -/
  apply Subtype.ext
  /-
    case a
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.ProbabilityMeasure α
    ν : MeasureTheory.ProbabilityMeasure β
    ⊢ Eq ↑((μ.prod ν).map ⋯) ↑(ν.prod μ)
  -/
  simp [Measure.prod_swap]
  /-
    🎉 no goals
  -/


