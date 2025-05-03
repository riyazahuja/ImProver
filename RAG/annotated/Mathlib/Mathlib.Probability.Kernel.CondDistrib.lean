/-- **Regular conditional probability distribution**: kernel associated with the conditional
expectation of `Y` given `X`.
For almost all `a`, `condDistrib Y X μ` evaluated at `X a` and a measurable set `s` is equal to
the conditional expectation `μ⟦Y ⁻¹' s | mβ.comap X⟧ a`. It also satisfies the equality
`μ[(fun a => f (X a, Y a)) | mβ.comap X] =ᵐ[μ] fun a => ∫ y, f (X a, y) ∂(condDistrib Y X μ (X a))`
for all integrable functions `f`. -/
noncomputable irreducible_def condDistrib {_ : MeasurableSpace α} [MeasurableSpace β] (Y : α → Ω)
    (X : α → β) (μ : Measure α) [IsFiniteMeasure μ] : Kernel β Ω :=
  (μ.map fun a => (X a, Y a)).condKernel


instance [MeasurableSpace β] : IsMarkovKernel (condDistrib Y X μ) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    inst✝ : MeasurableSpace β
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.condDistrib Y X μ)
  -/
  rw [condDistrib]; infer_instance
                    /-
                      🎉 no goals
                    -/


/-- If the singleton `{x}` has non-zero mass for `μ.map X`, then for all `s : Set Ω`,
`condDistrib Y X μ x s = (μ.map X {x})⁻¹ * μ.map (fun a => (X a, Y a)) ({x} ×ˢ s)` . -/
lemma condDistrib_apply_of_ne_zero [MeasurableSingletonClass β]
    (hY : Measurable Y) (x : β) (hX : μ.map X {x} ≠ 0) (s : Set Ω) :
    condDistrib Y X μ x s = (μ.map X {x})⁻¹ * μ.map (fun a => (X a, Y a)) ({x} ×ˢ s) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass β
    hY : Measurable Y
    x : β
    hX : Ne ((MeasureTheory.Measure.map X μ) (Singleton.singleton x)) 0
    s : Set Ω
    ⊢ Eq (((ProbabilityTheory.condDistrib Y X μ) x) s) (HMul.hMul (Inv.inv ((Measu …
  -/
  rw [condDistrib, Measure.condKernel_apply_of_ne_zero _ s]
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      inst✝ : MeasurableSingletonClass β
      hY : Measurable Y
      x : β
      hX : Ne ((MeasureTheory.Measure.map X μ) (Singleton.singleton x)) 0
      s : Set Ω
      ⊢ Eq (HMul.hMul (Inv.inv ((MeasureTheory.Measure.map (fun a => { fst := X a, s …
    -/
  · rw [Measure.fst_map_prod_mk hY]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      inst✝ : MeasurableSingletonClass β
      hY : Measurable Y
      x : β
      hX : Ne ((MeasureTheory.Measure.map X μ) (Singleton.singleton x)) 0
      s : Set Ω
      ⊢ Ne ((MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).fst  …
    -/
  · rwa [Measure.fst_map_prod_mk hY]
    /-
      🎉 no goals
    -/


theorem measurable_condDistrib (hs : MeasurableSet s) :
    Measurable[mβ.comap X] fun a => condDistrib Y X μ (X a) s :=
  (Kernel.measurable_coe _ hs).comp (Measurable.of_comap_le le_rfl)


theorem _root_.MeasureTheory.AEStronglyMeasurable.ae_integrable_condDistrib_map_iff
    (hY : AEMeasurable Y μ) (hf : AEStronglyMeasurable f (μ.map fun a => (X a, Y a))) :
    (∀ᵐ a ∂μ.map X, Integrable (fun ω => f (a, ω)) (condDistrib Y X μ a)) ∧
      Integrable (fun a => ∫ ω, ‖f (a, ω)‖ ∂condDistrib Y X μ a) (μ.map X) ↔
    Integrable f (μ.map fun a => (X a, Y a)) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    hY : AEMeasurable Y μ
    hf : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map (fun a => …
    ⊢ Iff (And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { …
  -/
  rw [condDistrib, ← hf.ae_integrable_condKernel_iff, Measure.fst_map_prod_mk₀ hY]
  /-
    🎉 no goals
  -/


theorem _root_.MeasureTheory.AEStronglyMeasurable.integral_condDistrib_map
    (hY : AEMeasurable Y μ) (hf : AEStronglyMeasurable f (μ.map fun a => (X a, Y a))) :
    AEStronglyMeasurable (fun x => ∫ y, f (x, y) ∂condDistrib Y X μ x) (μ.map X) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    inst✝ : NormedSpace Real F
    hY : AEMeasurable Y μ
    hf : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map (fun a => …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral ((Probab …
  -/
  rw [← Measure.fst_map_prod_mk₀ hY, condDistrib]; exact hf.integral_condKernel
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem _root_.MeasureTheory.AEStronglyMeasurable.integral_condDistrib (hX : AEMeasurable X μ)
    (hY : AEMeasurable Y μ) (hf : AEStronglyMeasurable f (μ.map fun a => (X a, Y a))) :
    AEStronglyMeasurable (fun a => ∫ y, f (X a, y) ∂condDistrib Y X μ (X a)) μ :=
  (hf.integral_condDistrib_map hY).comp_aemeasurable hX


theorem aestronglyMeasurable'_integral_condDistrib (hX : AEMeasurable X μ) (hY : AEMeasurable Y μ)
    (hf : AEStronglyMeasurable f (μ.map fun a => (X a, Y a))) :
    AEStronglyMeasurable' (mβ.comap X) (fun a => ∫ y, f (X a, y) ∂condDistrib Y X μ (X a)) μ :=
  (hf.integral_condDistrib_map hY).comp_ae_measurable' hX


/-- `condDistrib` is a.e. uniquely defined as the kernel satisfying the defining property of
`condKernel`. -/
theorem condDistrib_ae_eq_of_measure_eq_compProd (hX : Measurable X) (hY : Measurable Y)
    (κ : Kernel β Ω) [IsFiniteKernel κ] (hκ : μ.map (fun x => (X x, Y x)) = μ.map X ⊗ₘ κ) :
    ∀ᵐ x ∂μ.map X, κ x = condDistrib Y X μ x := by
  have heq : μ.map X = (μ.map (fun x ↦ (X x, Y x))).fst := by
    ext s hs
    rw [Measure.map_apply hX hs, Measure.fst_apply hs, Measure.map_apply]
    exacts [rfl, Measurable.prod hX hY, measurable_fst hs]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    hX : Measurable X
    hY : Measurable Y
    κ : ProbabilityTheory.Kernel β Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (MeasureTheory.Measure.map (fun x => { fst := X x, snd := Y x }) μ) (( …
    heq : Eq (MeasureTheory.Measure.map X μ) (MeasureTheory.Measure.map (fun x =>  …
    ⊢ Filter.Eventually (fun x => Eq (κ x) ((ProbabilityTheory.condDistrib Y X μ)  …
  -/
  rw [heq, condDistrib]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    hX : Measurable X
    hY : Measurable Y
    κ : ProbabilityTheory.Kernel β Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (MeasureTheory.Measure.map (fun x => { fst := X x, snd := Y x }) μ) (( …
    heq : Eq (MeasureTheory.Measure.map X μ) (MeasureTheory.Measure.map (fun x =>  …
    ⊢ Filter.Eventually (fun x => Eq (κ x) ((MeasureTheory.Measure.map (fun a => { …
  -/
  refine eq_condKernel_of_measure_eq_compProd _ ?_
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    hX : Measurable X
    hY : Measurable Y
    κ : ProbabilityTheory.Kernel β Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (MeasureTheory.Measure.map (fun x => { fst := X x, snd := Y x }) μ) (( …
    heq : Eq (MeasureTheory.Measure.map X μ) (MeasureTheory.Measure.map (fun x =>  …
    ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ) ((Mea …
  -/
  convert hκ
  /-
    case h.e'_3.h.e'_5
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    hX : Measurable X
    hY : Measurable Y
    κ : ProbabilityTheory.Kernel β Ω
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hκ : Eq (MeasureTheory.Measure.map (fun x => { fst := X x, snd := Y x }) μ) (( …
    heq : Eq (MeasureTheory.Measure.map X μ) (MeasureTheory.Measure.map (fun x =>  …
    ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).fst ( …
  -/
  exact heq.symm
  /-
    🎉 no goals
  -/


theorem integrable_toReal_condDistrib (hX : AEMeasurable X μ) (hs : MeasurableSet s) :
    Integrable (fun a => (condDistrib Y X μ (X a) s).toReal) μ := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    s : Set Ω
    hX : AEMeasurable X μ
    hs : MeasurableSet s
    ⊢ MeasureTheory.Integrable (fun a => (((ProbabilityTheory.condDistrib Y X μ) ( …
  -/
  refine integrable_toReal_of_lintegral_ne_top ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : AEMeasurable X μ
      hs : MeasurableSet s
      ⊢ AEMeasurable (fun a => ((ProbabilityTheory.condDistrib Y X μ) (X a)) s) μ
    -/
  · exact Measurable.comp_aemeasurable (Kernel.measurable_coe _ hs) hX
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : AEMeasurable X μ
      hs : MeasurableSet s
      ⊢ Ne (MeasureTheory.lintegral μ fun x => ((ProbabilityTheory.condDistrib Y X μ …
    -/
  · refine ne_of_lt ?_
    calc
      ∫⁻ a, condDistrib Y X μ (X a) s ∂μ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun a => prob_le_one
      _ = μ univ := lintegral_one
      _ < ∞ := measure_lt_top _ _


theorem _root_.MeasureTheory.Integrable.condDistrib_ae_map
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    ∀ᵐ b ∂μ.map X, Integrable (fun ω => f (b, ω)) (condDistrib Y X μ b) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    hY : AEMeasurable Y μ
    hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
    ⊢ Filter.Eventually (fun b => MeasureTheory.Integrable (fun ω => f { fst := b, …
  -/
  rw [condDistrib, ← Measure.fst_map_prod_mk₀ (X := X) hY]; exact hf_int.condKernel_ae
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem _root_.MeasureTheory.Integrable.condDistrib_ae (hX : AEMeasurable X μ)
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    ∀ᵐ a ∂μ, Integrable (fun ω => f (X a, ω)) (condDistrib Y X μ (X a)) :=
  ae_of_ae_map hX (hf_int.condDistrib_ae_map hY)


theorem _root_.MeasureTheory.Integrable.integral_norm_condDistrib_map
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂condDistrib Y X μ x) (μ.map X) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    hY : AEMeasurable Y μ
    hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
    ⊢ MeasureTheory.Integrable (fun x => MeasureTheory.integral ((ProbabilityTheor …
  -/
  rw [condDistrib, ← Measure.fst_map_prod_mk₀ (X := X) hY]; exact hf_int.integral_norm_condKernel
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem _root_.MeasureTheory.Integrable.integral_norm_condDistrib (hX : AEMeasurable X μ)
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun a => ∫ y, ‖f (X a, y)‖ ∂condDistrib Y X μ (X a)) μ :=
  (hf_int.integral_norm_condDistrib_map hY).comp_aemeasurable hX


theorem _root_.MeasureTheory.Integrable.norm_integral_condDistrib_map
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun x => ‖∫ y, f (x, y) ∂condDistrib Y X μ x‖) (μ.map X) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    inst✝ : NormedSpace Real F
    hY : AEMeasurable Y μ
    hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
    ⊢ MeasureTheory.Integrable (fun x => Norm.norm (MeasureTheory.integral ((Proba …
  -/
  rw [condDistrib, ← Measure.fst_map_prod_mk₀ (X := X) hY]; exact hf_int.norm_integral_condKernel
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem _root_.MeasureTheory.Integrable.norm_integral_condDistrib (hX : AEMeasurable X μ)
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun a => ‖∫ y, f (X a, y) ∂condDistrib Y X μ (X a)‖) μ :=
  (hf_int.norm_integral_condDistrib_map hY).comp_aemeasurable hX


theorem _root_.MeasureTheory.Integrable.integral_condDistrib_map
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun x => ∫ y, f (x, y) ∂condDistrib Y X μ x) (μ.map X) :=
  (integrable_norm_iff (hf_int.1.integral_condDistrib_map hY)).mp
    (hf_int.norm_integral_condDistrib_map hY)


theorem _root_.MeasureTheory.Integrable.integral_condDistrib (hX : AEMeasurable X μ)
    (hY : AEMeasurable Y μ) (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    Integrable (fun a => ∫ y, f (X a, y) ∂condDistrib Y X μ (X a)) μ :=
  (hf_int.integral_condDistrib_map hY).comp_aemeasurable hX


theorem setLIntegral_preimage_condDistrib (hX : Measurable X) (hY : AEMeasurable Y μ)
    (hs : MeasurableSet s) (ht : MeasurableSet t) :
    ∫⁻ a in X ⁻¹' t, condDistrib Y X μ (X a) s ∂μ = μ (X ⁻¹' t ∩ Y ⁻¹' s) := by
  -- Porting note: need to massage the LHS integrand into the form accepted by `lintegral_comp`
  -- (`rw` does not see that the two forms are defeq)
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    s : Set Ω
    t : Set β
    hX : Measurable X
    hY : AEMeasurable Y μ
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.preimage X t)) fun a => ((Proba …
  -/
  conv_lhs => arg 2; change (fun a => ((condDistrib Y X μ) a) s) ∘ X
  rw [lintegral_comp (Kernel.measurable_coe _ hs) hX, condDistrib, ← Measure.restrict_map hX ht, ←
    Measure.fst_map_prod_mk₀ hY, Measure.setLIntegral_condKernel_eq_measure_prod ht hs,
    Measure.map_apply_of_aemeasurable (hX.aemeasurable.prod_mk hY) (ht.prod hs), mk_preimage_prod]


@[deprecated (since := "2024-06-29")]
alias set_lintegral_preimage_condDistrib := setLIntegral_preimage_condDistrib


theorem setLIntegral_condDistrib_of_measurableSet (hX : Measurable X) (hY : AEMeasurable Y μ)
    (hs : MeasurableSet s) {t : Set α} (ht : MeasurableSet[mβ.comap X] t) :
    ∫⁻ a in t, condDistrib Y X μ (X a) s ∂μ = μ (t ∩ Y ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    s : Set Ω
    hX : Measurable X
    hY : AEMeasurable Y μ
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => ((ProbabilityTheory.cond …
  -/
  obtain ⟨t', ht', rfl⟩ := ht
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    s : Set Ω
    hX : Measurable X
    hY : AEMeasurable Y μ
    hs : MeasurableSet s
    t' : Set β
    ht' : MeasurableSet t'
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.preimage X t')) fun a => ((Prob …
  -/
  rw [setLIntegral_preimage_condDistrib hX hY hs ht']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condDistrib_of_measurableSet := setLIntegral_condDistrib_of_measurableSet


/-- For almost every `a : α`, the `condDistrib Y X μ` kernel applied to `X a` and a measurable set
`s` is equal to the conditional expectation of the indicator of `Y ⁻¹' s`. -/
theorem condDistrib_ae_eq_condexp (hX : Measurable X) (hY : Measurable Y) (hs : MeasurableSet s) :
    (fun a => (condDistrib Y X μ (X a) s).toReal) =ᵐ[μ] μ⟦Y ⁻¹' s|mβ.comap X⟧ := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    s : Set Ω
    hX : Measurable X
    hY : Measurable Y
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (((ProbabilityTheory.condDistrib …
  -/
  refine ae_eq_condexp_of_forall_setIntegral_eq hX.comap_le ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : Measurable X
      hY : Measurable Y
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable ((Set.preimage Y s).indicator fun ω => 1) μ
    -/
  · exact (integrable_const _).indicator (hY hs)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : Measurable X
      hY : Measurable Y
      hs : MeasurableSet s
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → MeasureTheory.I …
    -/
  · exact fun t _ _ => (integrable_toReal_condDistrib hX.aemeasurable hs).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : Measurable X
      hY : Measurable Y
      hs : MeasurableSet s
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt (μ s_1) Top.top → Eq (MeasureTheo …
    -/
  · intro t ht _
    rw [integral_toReal ((measurable_condDistrib hs).mono hX.comap_le le_rfl).aemeasurable
      (Eventually.of_forall fun ω => measure_lt_top (condDistrib Y X μ (X ω)) _),
      integral_indicator_const _ (hY hs), Measure.restrict_apply (hY hs), smul_eq_mul, mul_one,
      inter_comm, setLIntegral_condDistrib_of_measurableSet hX hY.aemeasurable hs ht]
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : Measurable X
      hY : Measurable Y
      hs : MeasurableSet s
      ⊢ MeasureTheory.AEStronglyMeasurable' (MeasurableSpace.comap X mβ) (fun a => ( …
    -/
  · refine (Measurable.stronglyMeasurable ?_).aeStronglyMeasurable'
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      s : Set Ω
      hX : Measurable X
      hY : Measurable Y
      hs : MeasurableSet s
      ⊢ Measurable fun a => (((ProbabilityTheory.condDistrib Y X μ) (X a)) s).toReal
    -/
    exact @Measurable.ennreal_toReal _ (mβ.comap X) _ (measurable_condDistrib hs)
    /-
      🎉 no goals
    -/


/-- The conditional expectation of a function `f` of the product `(X, Y)` is almost everywhere equal
to the integral of `y ↦ f(X, y)` against the `condDistrib` kernel. -/
theorem condexp_prod_ae_eq_integral_condDistrib' [NormedSpace ℝ F] [CompleteSpace F]
    (hX : Measurable X) (hY : AEMeasurable Y μ)
    (hf_int : Integrable f (μ.map fun a => (X a, Y a))) :
    μ[fun a => f (X a, Y a)|mβ.comap X] =ᵐ[μ] fun a => ∫ y, f (X a,y) ∂condDistrib Y X μ (X a) := by
  have hf_int' : Integrable (fun a => f (X a, Y a)) μ :=
    (integrable_map_measure hf_int.1 (hX.aemeasurable.prod_mk hY)).mp hf_int
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    F : Type u_4
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : NormedAddCommGroup F
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    X : α → β
    Y : α → Ω
    mβ : MeasurableSpace β
    f : Prod β Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hX : Measurable X
    hY : AEMeasurable Y μ
    hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
    hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (MeasurableSpace.co …
  -/
  refine (ae_eq_condexp_of_forall_setIntegral_eq hX.comap_le hf_int' (fun s _ _ => ?_) ?_ ?_).symm
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      s : Set α
      x✝¹ : MeasurableSet s
      x✝ : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.IntegrableOn (fun a => MeasureTheory.integral ((ProbabilityThe …
    -/
  · exact (hf_int.integral_condDistrib hX.aemeasurable hY).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · rintro s ⟨t, ht, rfl⟩ _
    change ∫ a in X ⁻¹' t, ((fun x' => ∫ y, f (x', y) ∂(condDistrib Y X μ) x') ∘ X) a ∂μ =
      ∫ a in X ⁻¹' t, f (X a, Y a) ∂μ
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      t : Set β
      ht : MeasurableSet t
      a✝ : LT.lt (μ (Set.preimage X t)) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.preimage X t)) fun a => Function …
    -/
    simp only [Function.comp_apply]
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      t : Set β
      ht : MeasurableSet t
      a✝ : LT.lt (μ (Set.preimage X t)) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.preimage X t)) fun a => MeasureT …
    -/
    rw [← integral_map hX.aemeasurable (f := fun x' => ∫ y, f (x', y) ∂(condDistrib Y X μ) x')]
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      t : Set β
      ht : MeasurableSet t
      a✝ : LT.lt (μ (Set.preimage X t)) Top.top
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map X (μ.restrict (Set.pre …
    -/
    swap
      /-
        case refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        Ω : Type u_3
        F : Type u_4
        inst✝⁶ : MeasurableSpace Ω
        inst✝⁵ : StandardBorelSpace Ω
        inst✝⁴ : Nonempty Ω
        inst✝³ : NormedAddCommGroup F
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝² : MeasureTheory.IsFiniteMeasure μ
        X : α → β
        Y : α → Ω
        mβ : MeasurableSpace β
        f : Prod β Ω → F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hX : Measurable X
        hY : AEMeasurable Y μ
        hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
        hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
        t : Set β
        ht : MeasurableSet t
        a✝ : LT.lt (μ (Set.preimage X t)) Top.top
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x' => MeasureTheory.integral ((Proba …
      -/
    · rw [← Measure.restrict_map hX ht]
      /-
        case refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        Ω : Type u_3
        F : Type u_4
        inst✝⁶ : MeasurableSpace Ω
        inst✝⁵ : StandardBorelSpace Ω
        inst✝⁴ : Nonempty Ω
        inst✝³ : NormedAddCommGroup F
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝² : MeasureTheory.IsFiniteMeasure μ
        X : α → β
        Y : α → Ω
        mβ : MeasurableSpace β
        f : Prod β Ω → F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hX : Measurable X
        hY : AEMeasurable Y μ
        hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
        hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
        t : Set β
        ht : MeasurableSet t
        a✝ : LT.lt (μ (Set.preimage X t)) Top.top
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x' => MeasureTheory.integral ((Proba …
      -/
      exact (hf_int.1.integral_condDistrib_map hY).restrict
      /-
        🎉 no goals
      -/
    rw [← Measure.restrict_map hX ht, ← Measure.fst_map_prod_mk₀ hY, condDistrib,
      Measure.setIntegral_condKernel_univ_right ht hf_int.integrableOn,
      setIntegral_map (ht.prod MeasurableSet.univ) hf_int.1 (hX.aemeasurable.prod_mk hY),
      mk_preimage_prod, preimage_univ, inter_univ]
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf_int : MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst …
      hf_int' : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      ⊢ MeasureTheory.AEStronglyMeasurable' (MeasurableSpace.comap X mβ) (fun a => M …
    -/
  · exact aestronglyMeasurable'_integral_condDistrib hX.aemeasurable hY hf_int.1
    /-
      🎉 no goals
    -/


/-- The conditional expectation of a function `f` of the product `(X, Y)` is almost everywhere equal
to the integral of `y ↦ f(X, y)` against the `condDistrib` kernel. -/
theorem condexp_prod_ae_eq_integral_condDistrib₀ [NormedSpace ℝ F] [CompleteSpace F]
    (hX : Measurable X) (hY : AEMeasurable Y μ)
    (hf : AEStronglyMeasurable f (μ.map fun a => (X a, Y a)))
    (hf_int : Integrable (fun a => f (X a, Y a)) μ) :
    μ[fun a => f (X a, Y a)|mβ.comap X] =ᵐ[μ] fun a => ∫ y, f (X a, y) ∂condDistrib Y X μ (X a) :=
  haveI hf_int' : Integrable f (μ.map fun a => (X a, Y a)) := by
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map (fun a => …
      hf_int : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      ⊢ MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst := X a …
    -/
    rwa [integrable_map_measure hf (hX.aemeasurable.prod_mk hY)]
    /-
      🎉 no goals
    -/
  condexp_prod_ae_eq_integral_condDistrib' hX hY hf_int'


/-- The conditional expectation of a function `f` of the product `(X, Y)` is almost everywhere equal
to the integral of `y ↦ f(X, y)` against the `condDistrib` kernel. -/
theorem condexp_prod_ae_eq_integral_condDistrib [NormedSpace ℝ F] [CompleteSpace F]
    (hX : Measurable X) (hY : AEMeasurable Y μ) (hf : StronglyMeasurable f)
    (hf_int : Integrable (fun a => f (X a, Y a)) μ) :
    μ[fun a => f (X a, Y a)|mβ.comap X] =ᵐ[μ] fun a => ∫ y, f (X a, y) ∂condDistrib Y X μ (X a) :=
  haveI hf_int' : Integrable f (μ.map fun a => (X a, Y a)) := by
    /-
      α : Type u_1
      β : Type u_2
      Ω : Type u_3
      F : Type u_4
      inst✝⁶ : MeasurableSpace Ω
      inst✝⁵ : StandardBorelSpace Ω
      inst✝⁴ : Nonempty Ω
      inst✝³ : NormedAddCommGroup F
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      X : α → β
      Y : α → Ω
      mβ : MeasurableSpace β
      f : Prod β Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hX : Measurable X
      hY : AEMeasurable Y μ
      hf : MeasureTheory.StronglyMeasurable f
      hf_int : MeasureTheory.Integrable (fun a => f { fst := X a, snd := Y a }) μ
      ⊢ MeasureTheory.Integrable f (MeasureTheory.Measure.map (fun a => { fst := X a …
    -/
    rwa [integrable_map_measure hf.aestronglyMeasurable (hX.aemeasurable.prod_mk hY)]
    /-
      🎉 no goals
    -/
  condexp_prod_ae_eq_integral_condDistrib' hX hY hf_int'


theorem condexp_ae_eq_integral_condDistrib [NormedSpace ℝ F] [CompleteSpace F] (hX : Measurable X)
    (hY : AEMeasurable Y μ) {f : Ω → F} (hf : StronglyMeasurable f)
    (hf_int : Integrable (fun a => f (Y a)) μ) :
    μ[fun a => f (Y a)|mβ.comap X] =ᵐ[μ] fun a => ∫ y, f y ∂condDistrib Y X μ (X a) :=
  condexp_prod_ae_eq_integral_condDistrib hX hY (hf.comp_measurable measurable_snd) hf_int


/-- The conditional expectation of `Y` given `X` is almost everywhere equal to the integral
`∫ y, y ∂(condDistrib Y X μ (X a))`. -/
theorem condexp_ae_eq_integral_condDistrib' {Ω} [NormedAddCommGroup Ω] [NormedSpace ℝ Ω]
    [CompleteSpace Ω] [MeasurableSpace Ω] [BorelSpace Ω] [SecondCountableTopology Ω] {Y : α → Ω}
    (hX : Measurable X) (hY_int : Integrable Y μ) :
    μ[Y|mβ.comap X] =ᵐ[μ] fun a => ∫ y, y ∂condDistrib Y X μ (X a) :=
  condexp_ae_eq_integral_condDistrib hX hY_int.1.aemeasurable stronglyMeasurable_id hY_int


theorem _root_.MeasureTheory.AEStronglyMeasurable.comp_snd_map_prod_mk
    {Ω F} {mΩ : MeasurableSpace Ω} (X : Ω → β) {μ : Measure Ω} [TopologicalSpace F] {f : Ω → F}
    (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun x : β × Ω => f x.2) (μ.map fun ω => (X ω, ω)) := by
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    Ω : Type u_5
    F : Type u_6
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    inst✝ : TopologicalSpace F
    f : Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measure.m …
  -/
  refine ⟨fun x => hf.mk f x.2, hf.stronglyMeasurable_mk.comp_measurable measurable_snd, ?_⟩
  suffices h : Measure.QuasiMeasurePreserving Prod.snd (μ.map fun ω ↦ (X ω, ω)) μ from
    Measure.QuasiMeasurePreserving.ae_eq h hf.ae_eq_mk
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    Ω : Type u_5
    F : Type u_6
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    inst✝ : TopologicalSpace F
    f : Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving Prod.snd (MeasureTheory.Measure …
  -/
  refine ⟨measurable_snd, Measure.AbsolutelyContinuous.mk fun s hs hμs => ?_⟩
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    Ω : Type u_5
    F : Type u_6
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    inst✝ : TopologicalSpace F
    f : Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    s : Set Ω
    hs : MeasurableSet s
    hμs : Eq (μ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map Prod.snd (MeasureTheory.Measure.map (fun ω => …
  -/
  rw [Measure.map_apply _ hs]
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    Ω : Type u_5
    F : Type u_6
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    inst✝ : TopologicalSpace F
    f : Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    s : Set Ω
    hs : MeasurableSet s
    hμs : Eq (μ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map (fun ω => { fst := X ω, snd := ω }) μ) (Set.p …
  -/
  swap; · exact measurable_snd
          /-
            🎉 no goals
          -/
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    Ω : Type u_5
    F : Type u_6
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    inst✝ : TopologicalSpace F
    f : Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    s : Set Ω
    hs : MeasurableSet s
    hμs : Eq (μ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map (fun ω => { fst := X ω, snd := ω }) μ) (Set.p …
  -/
  by_cases hX : AEMeasurable X μ
    /-
      case pos
      β : Type u_2
      mβ : MeasurableSpace β
      Ω : Type u_5
      F : Type u_6
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      inst✝ : TopologicalSpace F
      f : Ω → F
      hf : MeasureTheory.AEStronglyMeasurable f μ
      s : Set Ω
      hs : MeasurableSet s
      hμs : Eq (μ s) 0
      hX : AEMeasurable X μ
      ⊢ Eq ((MeasureTheory.Measure.map (fun ω => { fst := X ω, snd := ω }) μ) (Set.p …
    -/
  · rw [Measure.map_apply_of_aemeasurable]
      /-
        case pos
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : AEMeasurable X μ
        ⊢ Eq (μ (Set.preimage (fun ω => { fst := X ω, snd := ω }) (Set.preimage Prod.s …
      -/
    · rw [← univ_prod, mk_preimage_prod, preimage_univ, univ_inter, preimage_id']
      /-
        case pos
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : AEMeasurable X μ
        ⊢ Eq (μ s) 0
      -/
      exact hμs
      /-
        🎉 no goals
      -/
      /-
        case pos.hf
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : AEMeasurable X μ
        ⊢ AEMeasurable (fun ω => { fst := X ω, snd := ω }) μ
      -/
    · exact hX.prod_mk aemeasurable_id
      /-
        🎉 no goals
      -/
      /-
        case pos.hs
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : AEMeasurable X μ
        ⊢ MeasurableSet (Set.preimage Prod.snd s)
      -/
    · exact measurable_snd hs
      /-
        🎉 no goals
      -/
    /-
      case neg
      β : Type u_2
      mβ : MeasurableSpace β
      Ω : Type u_5
      F : Type u_6
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      inst✝ : TopologicalSpace F
      f : Ω → F
      hf : MeasureTheory.AEStronglyMeasurable f μ
      s : Set Ω
      hs : MeasurableSet s
      hμs : Eq (μ s) 0
      hX : Not (AEMeasurable X μ)
      ⊢ Eq ((MeasureTheory.Measure.map (fun ω => { fst := X ω, snd := ω }) μ) (Set.p …
    -/
  · rw [Measure.map_of_not_aemeasurable]
      /-
        case neg
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : Not (AEMeasurable X μ)
        ⊢ Eq (0 (Set.preimage Prod.snd s)) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        β : Type u_2
        mβ : MeasurableSpace β
        Ω : Type u_5
        F : Type u_6
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        inst✝ : TopologicalSpace F
        f : Ω → F
        hf : MeasureTheory.AEStronglyMeasurable f μ
        s : Set Ω
        hs : MeasurableSet s
        hμs : Eq (μ s) 0
        hX : Not (AEMeasurable X μ)
        ⊢ Not (AEMeasurable (fun ω => { fst := X ω, snd := ω }) μ)
      -/
    · contrapose! hX; exact measurable_fst.comp_aemeasurable hX
                      /-
                        🎉 no goals
                      -/


theorem _root_.MeasureTheory.Integrable.comp_snd_map_prod_mk
    {Ω} {mΩ : MeasurableSpace Ω} (X : Ω → β) {μ : Measure Ω} {f : Ω → F} (hf_int : Integrable f μ) :
    Integrable (fun x : β × Ω => f x.2) (μ.map fun ω => (X ω, ω)) := by
  /-
    β : Type u_2
    F : Type u_4
    inst✝ : NormedAddCommGroup F
    mβ : MeasurableSpace β
    Ω : Type u_5
    mΩ : MeasurableSpace Ω
    X : Ω → β
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
  -/
  by_cases hX : AEMeasurable X μ
    /-
      case pos
      β : Type u_2
      F : Type u_4
      inst✝ : NormedAddCommGroup F
      mβ : MeasurableSpace β
      Ω : Type u_5
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      f : Ω → F
      hf_int : MeasureTheory.Integrable f μ
      hX : AEMeasurable X μ
      ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
    -/
  · have hf := hf_int.1.comp_snd_map_prod_mk X (mΩ := mΩ) (mβ := mβ)
    /-
      case pos
      β : Type u_2
      F : Type u_4
      inst✝ : NormedAddCommGroup F
      mβ : MeasurableSpace β
      Ω : Type u_5
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      f : Ω → F
      hf_int : MeasureTheory.Integrable f μ
      hX : AEMeasurable X μ
      hf : MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measur …
      ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
    -/
    refine ⟨hf, ?_⟩
    /-
      case pos
      β : Type u_2
      F : Type u_4
      inst✝ : NormedAddCommGroup F
      mβ : MeasurableSpace β
      Ω : Type u_5
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      f : Ω → F
      hf_int : MeasureTheory.Integrable f μ
      hX : AEMeasurable X μ
      hf : MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measur …
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => f x.2) (MeasureTheory.Measure.map  …
    -/
    rw [hasFiniteIntegral_iff_nnnorm, lintegral_map' hf.ennnorm (hX.prod_mk aemeasurable_id)]
    /-
      case pos
      β : Type u_2
      F : Type u_4
      inst✝ : NormedAddCommGroup F
      mβ : MeasurableSpace β
      Ω : Type u_5
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      f : Ω → F
      hf_int : MeasureTheory.Integrable f μ
      hX : AEMeasurable X μ
      hf : MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measur …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f { fst := X a, s …
    -/
    exact hf_int.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      β : Type u_2
      F : Type u_4
      inst✝ : NormedAddCommGroup F
      mβ : MeasurableSpace β
      Ω : Type u_5
      mΩ : MeasurableSpace Ω
      X : Ω → β
      μ : MeasureTheory.Measure Ω
      f : Ω → F
      hf_int : MeasureTheory.Integrable f μ
      hX : Not (AEMeasurable X μ)
      ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
    -/
  · rw [Measure.map_of_not_aemeasurable]
      /-
        case neg
        β : Type u_2
        F : Type u_4
        inst✝ : NormedAddCommGroup F
        mβ : MeasurableSpace β
        Ω : Type u_5
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        f : Ω → F
        hf_int : MeasureTheory.Integrable f μ
        hX : Not (AEMeasurable X μ)
        ⊢ MeasureTheory.Integrable (fun x => f x.2) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        β : Type u_2
        F : Type u_4
        inst✝ : NormedAddCommGroup F
        mβ : MeasurableSpace β
        Ω : Type u_5
        mΩ : MeasurableSpace Ω
        X : Ω → β
        μ : MeasureTheory.Measure Ω
        f : Ω → F
        hf_int : MeasureTheory.Integrable f μ
        hX : Not (AEMeasurable X μ)
        ⊢ Not (AEMeasurable (fun ω => { fst := X ω, snd := ω }) μ)
      -/
    · contrapose! hX; exact measurable_fst.comp_aemeasurable hX
                      /-
                        🎉 no goals
                      -/


theorem aestronglyMeasurable_comp_snd_map_prod_mk_iff {Ω F} {_ : MeasurableSpace Ω}
    [TopologicalSpace F] {X : Ω → β} {μ : Measure Ω} (hX : Measurable X) {f : Ω → F} :
    AEStronglyMeasurable (fun x : β × Ω => f x.2) (μ.map fun ω => (X ω, ω)) ↔
    AEStronglyMeasurable f μ :=
  ⟨fun h => h.comp_measurable (hX.prod_mk measurable_id), fun h => h.comp_snd_map_prod_mk X⟩


theorem integrable_comp_snd_map_prod_mk_iff {Ω} {_ : MeasurableSpace Ω} {X : Ω → β} {μ : Measure Ω}
    (hX : Measurable X) {f : Ω → F} :
    Integrable (fun x : β × Ω => f x.2) (μ.map fun ω => (X ω, ω)) ↔ Integrable f μ :=
  ⟨fun h => h.comp_measurable (hX.prod_mk measurable_id), fun h => h.comp_snd_map_prod_mk X⟩


theorem condexp_ae_eq_integral_condDistrib_id [NormedSpace ℝ F] [CompleteSpace F] {X : Ω → β}
    {μ : Measure Ω} [IsFiniteMeasure μ] (hX : Measurable X) {f : Ω → F} (hf_int : Integrable f μ) :
    μ[f|mβ.comap X] =ᵐ[μ] fun a => ∫ y, f y ∂condDistrib id X μ (X a) :=
  condexp_prod_ae_eq_integral_condDistrib' hX aemeasurable_id (hf_int.comp_snd_map_prod_mk X)


