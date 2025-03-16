theorem _root_.MeasureTheory.AEStronglyMeasurable.comp_snd_map_prod_id [TopologicalSpace F]
    (hm : m ≤ mΩ) (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable (fun x : Ω × Ω => f x.2)
      (@Measure.map Ω (Ω × Ω) mΩ (m.prod mΩ) (fun ω => (id ω, id ω)) μ) := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : TopologicalSpace F
    hm : LE.le m mΩ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measure.m …
  -/
  rw [← aestronglyMeasurable_comp_snd_map_prod_mk_iff (measurable_id'' hm)] at hf
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : TopologicalSpace F
    hm : LE.le m mΩ
    hf : MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measur …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measure.m …
  -/
  simp_rw [id] at hf ⊢
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : TopologicalSpace F
    hm : LE.le m mΩ
    hf : MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measur …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f x.2) (MeasureTheory.Measure.m …
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem _root_.MeasureTheory.Integrable.comp_snd_map_prod_id [NormedAddCommGroup F] (hm : m ≤ mΩ)
    (hf : Integrable f μ) : Integrable (fun x : Ω × Ω => f x.2)
      (@Measure.map Ω (Ω × Ω) mΩ (m.prod mΩ) (fun ω => (id ω, id ω)) μ) := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : NormedAddCommGroup F
    hm : LE.le m mΩ
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
  -/
  rw [← integrable_comp_snd_map_prod_mk_iff (measurable_id'' hm)] at hf
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : NormedAddCommGroup F
    hm : LE.le m mΩ
    hf : MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun …
    ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
  -/
  simp_rw [id] at hf ⊢
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → F
    inst✝ : NormedAddCommGroup F
    hm : LE.le m mΩ
    hf : MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun …
    ⊢ MeasureTheory.Integrable (fun x => f x.2) (MeasureTheory.Measure.map (fun ω  …
  -/
  exact hf
  /-
    🎉 no goals
  -/


open Classical in
/-- Kernel associated with the conditional expectation with respect to a σ-algebra. It satisfies
`μ[f | m] =ᵐ[μ] fun ω => ∫ y, f y ∂(condexpKernel μ m ω)`.
It is defined as the conditional distribution of the identity given the identity, where the second
identity is understood as a map from `Ω` with the σ-algebra `mΩ` to `Ω` with σ-algebra `m ⊓ mΩ`.
We use `m ⊓ mΩ` instead of `m` to ensure that it is a sub-σ-algebra of `mΩ`. We then use
`Kernel.comap` to get a kernel from `m` to `mΩ` instead of from `m ⊓ mΩ` to `mΩ`. -/
noncomputable irreducible_def condexpKernel (μ : Measure Ω) [IsFiniteMeasure μ]
    (m : MeasurableSpace Ω) : @Kernel Ω Ω m mΩ :=
  if _h : Nonempty Ω then
    Kernel.comap (@condDistrib Ω Ω Ω mΩ _ _ mΩ (m ⊓ mΩ) id id μ _) id
      (measurable_id'' (inf_le_left : m ⊓ mΩ ≤ m))
  else 0


lemma condexpKernel_eq (μ : Measure Ω) [IsFiniteMeasure μ] [h : Nonempty Ω]
    (m : MeasurableSpace Ω) :
    condexpKernel (mΩ := mΩ) μ m = Kernel.comap (@condDistrib Ω Ω Ω mΩ _ _ mΩ (m ⊓ mΩ) id id μ _) id
      (measurable_id'' (inf_le_left : m ⊓ mΩ ≤ m)) := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : Nonempty Ω
    m : MeasurableSpace Ω
    ⊢ Eq (ProbabilityTheory.condexpKernel μ m) ((ProbabilityTheory.condDistrib id  …
  -/
  simp [condexpKernel, h]
  /-
    🎉 no goals
  -/


lemma condexpKernel_apply_eq_condDistrib [Nonempty Ω] {ω : Ω} :
    condexpKernel μ m ω = @condDistrib Ω Ω Ω mΩ _ _ mΩ (m ⊓ mΩ) id id μ _ (id ω) := by
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : Nonempty Ω
    ω : Ω
    ⊢ Eq ((ProbabilityTheory.condexpKernel μ m) ω) ((ProbabilityTheory.condDistrib …
  -/
  simp [condexpKernel_eq, Kernel.comap_apply]
  /-
    🎉 no goals
  -/


instance : IsMarkovKernel (condexpKernel μ m) := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.condexpKernel μ m)
  -/
  rcases isEmpty_or_nonempty Ω with h | h
    /-
      case inl
      Ω : Type u_1
      F : Type u_2
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : IsEmpty Ω
      ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.condexpKernel μ m)
    -/
  · exact ⟨fun a ↦ (IsEmpty.false a).elim⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      Ω : Type u_1
      F : Type u_2
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : Nonempty Ω
      ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.condexpKernel μ m)
    -/
  · simp [condexpKernel, h]; infer_instance
                             /-
                               🎉 no goals
                             -/


theorem measurable_condexpKernel {s : Set Ω} (hs : MeasurableSet s) :
    Measurable[m] fun ω => condexpKernel μ m ω s := by
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Measurable fun ω => ((ProbabilityTheory.condexpKernel μ m) ω) s
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ Measurable fun ω => ((ProbabilityTheory.condexpKernel μ m) ω) s
  -/
  simp_rw [condexpKernel_apply_eq_condDistrib]
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ Measurable fun ω => ((ProbabilityTheory.condDistrib id id μ) (id ω)) s
  -/
  refine Measurable.mono ?_ (inf_le_left : m ⊓ mΩ ≤ m) le_rfl
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ Measurable fun ω => ((ProbabilityTheory.condDistrib id id μ) (id ω)) s
  -/
  convert measurable_condDistrib (μ := μ) hs
  /-
    case h.e'_3
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ Eq (Min.min m mΩ) (MeasurableSpace.comap id (Min.min m mΩ))
  -/
  rw [MeasurableSpace.comap_id]
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_condexpKernel {s : Set Ω} (hs : MeasurableSet s) :
    StronglyMeasurable[m] fun ω => condexpKernel μ m ω s :=
  Measurable.stronglyMeasurable (measurable_condexpKernel hs)


theorem _root_.MeasureTheory.AEStronglyMeasurable.integral_condexpKernel [NormedSpace ℝ F]
    (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun ω => ∫ y, f y ∂condexpKernel μ m ω) μ := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => MeasureTheory.integral ((Probab …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.AEStronglyMeasurable (fun ω => MeasureTheory.integral ((Probab …
  -/
  simp_rw [condexpKernel_apply_eq_condDistrib]
  exact AEStronglyMeasurable.integral_condDistrib
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf.comp_snd_map_prod_id inf_le_right)


theorem aestronglyMeasurable'_integral_condexpKernel [NormedSpace ℝ F]
    (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable' m (fun ω => ∫ y, f y ∂condexpKernel μ m ω) μ := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (fun ω => MeasureTheory.integral ((Pro …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.AEStronglyMeasurable' m (fun ω => MeasureTheory.integral ((Pro …
  -/
  rw [condexpKernel_eq]
  have h := aestronglyMeasurable'_integral_condDistrib
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf.comp_snd_map_prod_id (inf_le_right : m ⊓ mΩ ≤ mΩ))
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    a✝ : Nontrivial Ω
    h : MeasureTheory.AEStronglyMeasurable' (MeasurableSpace.comap id (Min.min m m …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (fun ω => MeasureTheory.integral (((Pr …
  -/
  rw [MeasurableSpace.comap_id] at h
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    a✝ : Nontrivial Ω
    h : MeasureTheory.AEStronglyMeasurable' (Min.min m mΩ) (fun a => MeasureTheory …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (fun ω => MeasureTheory.integral (((Pr …
  -/
  exact AEStronglyMeasurable'.mono h inf_le_left
  /-
    🎉 no goals
  -/


theorem _root_.MeasureTheory.Integrable.condexpKernel_ae (hf_int : Integrable f μ) :
    ∀ᵐ ω ∂μ, Integrable f (condexpKernel μ m ω) := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : NormedAddCommGroup F
    f : Ω → F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ Filter.Eventually (fun ω => MeasureTheory.Integrable f ((ProbabilityTheory.c …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : NormedAddCommGroup F
    f : Ω → F
    hf_int : MeasureTheory.Integrable f μ
    a✝ : Nontrivial Ω
    ⊢ Filter.Eventually (fun ω => MeasureTheory.Integrable f ((ProbabilityTheory.c …
  -/
  rw [condexpKernel_eq]
  convert Integrable.condDistrib_ae
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf_int.comp_snd_map_prod_id (inf_le_right : m ⊓ mΩ ≤ mΩ)) using 1


theorem _root_.MeasureTheory.Integrable.integral_norm_condexpKernel (hf_int : Integrable f μ) :
    Integrable (fun ω => ∫ y, ‖f y‖ ∂condexpKernel μ m ω) μ := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : NormedAddCommGroup F
    f : Ω → F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun ω => MeasureTheory.integral ((ProbabilityTheor …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : NormedAddCommGroup F
    f : Ω → F
    hf_int : MeasureTheory.Integrable f μ
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.Integrable (fun ω => MeasureTheory.integral ((ProbabilityTheor …
  -/
  rw [condexpKernel_eq]
  convert Integrable.integral_norm_condDistrib
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf_int.comp_snd_map_prod_id (inf_le_right : m ⊓ mΩ ≤ mΩ)) using 1


theorem _root_.MeasureTheory.Integrable.norm_integral_condexpKernel [NormedSpace ℝ F]
    (hf_int : Integrable f μ) :
    Integrable (fun ω => ‖∫ y, f y ∂condexpKernel μ m ω‖) μ := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun ω => Norm.norm (MeasureTheory.integral ((Proba …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf_int : MeasureTheory.Integrable f μ
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.Integrable (fun ω => Norm.norm (MeasureTheory.integral ((Proba …
  -/
  rw [condexpKernel_eq]
  convert Integrable.norm_integral_condDistrib
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf_int.comp_snd_map_prod_id (inf_le_right : m ⊓ mΩ ≤ mΩ)) using 1


theorem _root_.MeasureTheory.Integrable.integral_condexpKernel [NormedSpace ℝ F]
    (hf_int : Integrable f μ) :
    Integrable (fun ω => ∫ y, f y ∂condexpKernel μ m ω) μ := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun ω => MeasureTheory.integral ((ProbabilityTheor …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : NormedAddCommGroup F
    f : Ω → F
    inst✝ : NormedSpace Real F
    hf_int : MeasureTheory.Integrable f μ
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.Integrable (fun ω => MeasureTheory.integral ((ProbabilityTheor …
  -/
  rw [condexpKernel_eq]
  convert Integrable.integral_condDistrib
    (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) aemeasurable_id
    (hf_int.comp_snd_map_prod_id (inf_le_right : m ⊓ mΩ ≤ mΩ)) using 1


theorem integrable_toReal_condexpKernel {s : Set Ω} (hs : MeasurableSet s) :
    Integrable (fun ω => (condexpKernel μ m ω s).toReal) μ := by
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ MeasureTheory.Integrable (fun ω => (((ProbabilityTheory.condexpKernel μ m) ω …
  -/
  nontriviality Ω
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.Integrable (fun ω => (((ProbabilityTheory.condexpKernel μ m) ω …
  -/
  rw [condexpKernel_eq]
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    a✝ : Nontrivial Ω
    ⊢ MeasureTheory.Integrable (fun ω => ((((ProbabilityTheory.condDistrib id id μ …
  -/
  exact integrable_toReal_condDistrib (aemeasurable_id'' μ (inf_le_right : m ⊓ mΩ ≤ mΩ)) hs
  /-
    🎉 no goals
  -/


lemma condexpKernel_ae_eq_condexp' {s : Set Ω} (hs : MeasurableSet s) :
    (fun ω ↦ (condexpKernel μ m ω s).toReal) =ᵐ[μ] μ⟦s | m ⊓ mΩ⟧ := by
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
  -/
  rcases isEmpty_or_nonempty Ω with h | h
    /-
      case inl
      Ω : Type u_1
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      s : Set Ω
      hs : MeasurableSet s
      h : IsEmpty Ω
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
    -/
  · have : μ = 0 := Measure.eq_zero_of_isEmpty μ
    /-
      case inl
      Ω : Type u_1
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      s : Set Ω
      hs : MeasurableSet s
      h : IsEmpty Ω
      this : Eq μ 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
    -/
    simpa [this] using trivial
    /-
      🎉 no goals
    -/
  have h := condDistrib_ae_eq_condexp (μ := μ)
    (measurable_id'' (inf_le_right : m ⊓ mΩ ≤ mΩ)) measurable_id hs
  /-
    case inr
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    h✝ : Nonempty Ω
    h : (MeasureTheory.ae μ).EventuallyEq (fun a => (((ProbabilityTheory.condDistr …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
  -/
  simp only [id_eq, MeasurableSpace.comap_id, preimage_id_eq] at h
  /-
    case inr
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    h✝ : Nonempty Ω
    h : (MeasureTheory.ae μ).EventuallyEq (fun a => (((ProbabilityTheory.condDistr …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
  -/
  simp_rw [condexpKernel_apply_eq_condDistrib]
  /-
    case inr
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    hs : MeasurableSet s
    h✝ : Nonempty Ω
    h : (MeasureTheory.ae μ).EventuallyEq (fun a => (((ProbabilityTheory.condDistr …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condDistrib …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma condexpKernel_ae_eq_condexp
    (hm : m ≤ mΩ) {s : Set Ω} (hs : MeasurableSet s) :
    (fun ω ↦ (condexpKernel μ m ω s).toReal) =ᵐ[μ] μ⟦s | m⟧ :=
                                              /-
                                                Ω : Type u_1
                                                m mΩ : MeasurableSpace Ω
                                                inst✝¹ : StandardBorelSpace Ω
                                                μ : MeasureTheory.Measure Ω
                                                inst✝ : MeasureTheory.IsFiniteMeasure μ
                                                hm : LE.le m mΩ
                                                s : Set Ω
                                                hs : MeasurableSet s
                                                ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ (s …
                                              -/
  (condexpKernel_ae_eq_condexp' hs).trans (by rw [inf_of_le_left hm])
                                              /-
                                                🎉 no goals
                                              -/


lemma condexpKernel_ae_eq_trim_condexp
    (hm : m ≤ mΩ) {s : Set Ω} (hs : MeasurableSet s) :
    (fun ω ↦ (condexpKernel μ m ω s).toReal) =ᵐ[μ.trim hm] μ⟦s | m⟧ := by
  /-
    Ω : Type u_1
    m mΩ : MeasurableSpace Ω
    inst✝¹ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hm : LE.le m mΩ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq (fun ω => (((ProbabilityTheory.c …
  -/
  rw [ae_eq_trim_iff hm _ stronglyMeasurable_condexp]
    /-
      Ω : Type u_1
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hm : LE.le m mΩ
      s : Set Ω
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun ω => (((ProbabilityTheory.condexpKern …
    -/
  · exact condexpKernel_ae_eq_condexp hm hs
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hm : LE.le m mΩ
      s : Set Ω
      hs : MeasurableSet s
      ⊢ MeasureTheory.StronglyMeasurable fun ω => (((ProbabilityTheory.condexpKernel …
    -/
  · refine Measurable.stronglyMeasurable ?_
    /-
      Ω : Type u_1
      m mΩ : MeasurableSpace Ω
      inst✝¹ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hm : LE.le m mΩ
      s : Set Ω
      hs : MeasurableSet s
      ⊢ Measurable fun ω => (((ProbabilityTheory.condexpKernel μ m) ω) s).toReal
    -/
    exact @Measurable.ennreal_toReal _ m _ (measurable_condexpKernel hs)
    /-
      🎉 no goals
    -/


theorem condexp_ae_eq_integral_condexpKernel' [NormedAddCommGroup F] {f : Ω → F}
    [NormedSpace ℝ F] [CompleteSpace F] (hf_int : Integrable f μ) :
    μ[f|m ⊓ mΩ] =ᵐ[μ] fun ω => ∫ y, f y ∂condexpKernel μ m ω := by
  /-
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    inst✝² : NormedAddCommGroup F
    f : Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hf_int : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
  -/
  rcases isEmpty_or_nonempty Ω with h | h
    /-
      case inl
      Ω : Type u_1
      F : Type u_2
      m mΩ : MeasurableSpace Ω
      inst✝⁴ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝³ : MeasureTheory.IsFiniteMeasure μ
      inst✝² : NormedAddCommGroup F
      f : Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hf_int : MeasureTheory.Integrable f μ
      h : IsEmpty Ω
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
    -/
  · have : μ = 0 := Measure.eq_zero_of_isEmpty μ
    /-
      case inl
      Ω : Type u_1
      F : Type u_2
      m mΩ : MeasurableSpace Ω
      inst✝⁴ : StandardBorelSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝³ : MeasureTheory.IsFiniteMeasure μ
      inst✝² : NormedAddCommGroup F
      f : Ω → F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hf_int : MeasureTheory.Integrable f μ
      h : IsEmpty Ω
      this : Eq μ 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
    -/
    simpa [this] using trivial
    /-
      🎉 no goals
    -/
  /-
    case inr
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    inst✝² : NormedAddCommGroup F
    f : Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hf_int : MeasureTheory.Integrable f μ
    h : Nonempty Ω
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
  -/
  have hX : @Measurable Ω Ω mΩ (m ⊓ mΩ) id := measurable_id.mono le_rfl (inf_le_right : m ⊓ mΩ ≤ mΩ)
  /-
    case inr
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    inst✝² : NormedAddCommGroup F
    f : Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hf_int : MeasureTheory.Integrable f μ
    h : Nonempty Ω
    hX : Measurable id
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
  -/
  simp_rw [condexpKernel_apply_eq_condDistrib]
  /-
    case inr
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    inst✝² : NormedAddCommGroup F
    f : Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hf_int : MeasureTheory.Integrable f μ
    h : Nonempty Ω
    hX : Measurable id
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
  -/
  have h := condexp_ae_eq_integral_condDistrib_id hX hf_int
  /-
    case inr
    Ω : Type u_1
    F : Type u_2
    m mΩ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    inst✝² : NormedAddCommGroup F
    f : Ω → F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hf_int : MeasureTheory.Integrable f μ
    h✝ : Nonempty Ω
    hX : Measurable id
    h : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (MeasurableSpace. …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
  -/
  simpa only [MeasurableSpace.comap_id, id_eq] using h
  /-
    🎉 no goals
  -/


/-- The conditional expectation of `f` with respect to a σ-algebra `m` is almost everywhere equal to
the integral `∫ y, f y ∂(condexpKernel μ m ω)`. -/
theorem condexp_ae_eq_integral_condexpKernel [NormedAddCommGroup F] {f : Ω → F}
    [NormedSpace ℝ F] [CompleteSpace F] (hm : m ≤ mΩ) (hf_int : Integrable f μ) :
    μ[f|m] =ᵐ[μ] fun ω => ∫ y, f y ∂condexpKernel μ m ω :=
                                                                 /-
                                                                   Ω : Type u_1
                                                                   F : Type u_2
                                                                   m mΩ : MeasurableSpace Ω
                                                                   inst✝⁴ : StandardBorelSpace Ω
                                                                   μ : MeasureTheory.Measure Ω
                                                                   inst✝³ : MeasureTheory.IsFiniteMeasure μ
                                                                   inst✝² : NormedAddCommGroup F
                                                                   f : Ω → F
                                                                   inst✝¹ : NormedSpace Real F
                                                                   inst✝ : CompleteSpace F
                                                                   hm : LE.le m mΩ
                                                                   hf_int : MeasureTheory.Integrable f μ
                                                                   ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (Min.min m mΩ) μ f) …
                                                                 -/
  ((condexp_ae_eq_integral_condexpKernel' hf_int).symm.trans (by rw [inf_of_le_left hm])).symm
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma condexp_generateFrom_singleton (hs : MeasurableSet s) {f : Ω → F} (hf : Integrable f μ) :
    μ[f | generateFrom {s}] =ᵐ[μ.restrict s] fun _ ↦ ∫ x, f x ∂μ[|s] := by
  /-
    Ω : Type u_1
    F : Type u_2
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    s : Set Ω
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    hs : MeasurableSet s
    f : Ω → F
    hf : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp (Measu …
  -/
  by_cases hμs : μ s = 0
    /-
      case pos
      Ω : Type u_1
      F : Type u_2
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝³ : MeasureTheory.IsFiniteMeasure μ
      s : Set Ω
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hs : MeasurableSet s
      f : Ω → F
      hf : MeasureTheory.Integrable f μ
      hμs : Eq (μ s) 0
      ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp (Measu …
    -/
  · rw [Measure.restrict_eq_zero.2 hμs]
    /-
      case pos
      Ω : Type u_1
      F : Type u_2
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝³ : MeasureTheory.IsFiniteMeasure μ
      s : Set Ω
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      hs : MeasurableSet s
      f : Ω → F
      hf : MeasureTheory.Integrable f μ
      hμs : Eq (μ s) 0
      ⊢ (MeasureTheory.ae 0).EventuallyEq (MeasureTheory.condexp (MeasurableSpace.ge …
    -/
    rfl
    /-
      🎉 no goals
    -/
  refine ae_eq_trans (condexp_restrict_ae_eq_restrict
    (generateFrom_singleton_le hs)
    (measurableSet_generateFrom rfl) hf).symm ?_
  · refine (ae_eq_condexp_of_forall_setIntegral_eq
      (generateFrom_singleton_le hs) hf.restrict ?_ ?_
      stronglyMeasurable_const.aeStronglyMeasurable').symm
      /-
        case neg.refine_1
        Ω : Type u_1
        F : Type u_2
        mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝³ : MeasureTheory.IsFiniteMeasure μ
        s : Set Ω
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hs : MeasurableSet s
        f : Ω → F
        hf : MeasureTheory.Integrable f μ
        hμs : Not (Eq (μ s) 0)
        ⊢ ∀ (s_1 : Set Ω), MeasurableSet s_1 → LT.lt ((μ.restrict s) s_1) Top.top → Me …
      -/
    · rintro t - -
      /-
        case neg.refine_1
        Ω : Type u_1
        F : Type u_2
        mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝³ : MeasureTheory.IsFiniteMeasure μ
        s : Set Ω
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hs : MeasurableSet s
        f : Ω → F
        hf : MeasureTheory.Integrable f μ
        hμs : Not (Eq (μ s) 0)
        t : Set Ω
        ⊢ MeasureTheory.IntegrableOn (fun x => MeasureTheory.integral (ProbabilityTheo …
      -/
      rw [integrableOn_const]
      /-
        case neg.refine_1
        Ω : Type u_1
        F : Type u_2
        mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝³ : MeasureTheory.IsFiniteMeasure μ
        s : Set Ω
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hs : MeasurableSet s
        f : Ω → F
        hf : MeasureTheory.Integrable f μ
        hμs : Not (Eq (μ s) 0)
        t : Set Ω
        ⊢ Or (Eq (MeasureTheory.integral (ProbabilityTheory.cond μ s) fun x => f x) 0) …
      -/
      exact Or.inr <| measure_lt_top (μ.restrict s) t
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        Ω : Type u_1
        F : Type u_2
        mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝³ : MeasureTheory.IsFiniteMeasure μ
        s : Set Ω
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hs : MeasurableSet s
        f : Ω → F
        hf : MeasureTheory.Integrable f μ
        hμs : Not (Eq (μ s) 0)
        ⊢ ∀ (s_1 : Set Ω), MeasurableSet s_1 → LT.lt ((μ.restrict s) s_1) Top.top → Eq …
      -/
    · rintro t ht -
      /-
        case neg.refine_2
        Ω : Type u_1
        F : Type u_2
        mΩ : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝³ : MeasureTheory.IsFiniteMeasure μ
        s : Set Ω
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        inst✝ : CompleteSpace F
        hs : MeasurableSet s
        f : Ω → F
        hf : MeasureTheory.Integrable f μ
        hμs : Not (Eq (μ s) 0)
        t : Set Ω
        ht : MeasurableSet t
        ⊢ Eq (MeasureTheory.integral ((μ.restrict s).restrict t) fun x => MeasureTheor …
      -/
      obtain (h | h | h | h) := measurableSet_generateFrom_singleton_iff.1 ht
        /-
          case neg.refine_2.inl
          Ω : Type u_1
          F : Type u_2
          mΩ : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝³ : MeasureTheory.IsFiniteMeasure μ
          s : Set Ω
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace Real F
          inst✝ : CompleteSpace F
          hs : MeasurableSet s
          f : Ω → F
          hf : MeasureTheory.Integrable f μ
          hμs : Not (Eq (μ s) 0)
          t : Set Ω
          ht : MeasurableSet t
          h : Eq t EmptyCollection.emptyCollection
          ⊢ Eq (MeasureTheory.integral ((μ.restrict s).restrict t) fun x => MeasureTheor …
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
      · simp only [h, cond, integral_smul_measure, ENNReal.toReal_inv, integral_const,
          MeasurableSet.univ, Measure.restrict_apply, univ_inter, Measure.restrict_apply_self]
        /-
          case neg.refine_2.inr.inl
          Ω : Type u_1
          F : Type u_2
          mΩ : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝³ : MeasureTheory.IsFiniteMeasure μ
          s : Set Ω
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace Real F
          inst✝ : CompleteSpace F
          hs : MeasurableSet s
          f : Ω → F
          hf : MeasureTheory.Integrable f μ
          hμs : Not (Eq (μ s) 0)
          t : Set Ω
          ht : MeasurableSet t
          h : Eq t s
          ⊢ Eq (HSMul.hSMul (μ s).toReal (HSMul.hSMul (Inv.inv (μ s).toReal) (MeasureThe …
        -/
        rw [smul_inv_smul₀, Measure.restrict_restrict hs, inter_self]
        /-
          case neg.refine_2.inr.inl.ha
          Ω : Type u_1
          F : Type u_2
          mΩ : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          inst✝³ : MeasureTheory.IsFiniteMeasure μ
          s : Set Ω
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace Real F
          inst✝ : CompleteSpace F
          hs : MeasurableSet s
          f : Ω → F
          hf : MeasureTheory.Integrable f μ
          hμs : Not (Eq (μ s) 0)
          t : Set Ω
          ht : MeasurableSet t
          h : Eq t s
          ⊢ Ne (μ s).toReal 0
        -/
        exact ENNReal.toReal_ne_zero.2 ⟨hμs, measure_ne_top _ _⟩
        /-
          🎉 no goals
        -/
      · simp only [h, integral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter,
          ((Measure.restrict_apply_eq_zero hs.compl).2 <| compl_inter_self s ▸ measure_empty),
          ENNReal.zero_toReal, zero_smul, setIntegral_zero_measure]
      · simp only [h, Measure.restrict_univ, cond, integral_smul_measure, ENNReal.toReal_inv,
          integral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter,
          smul_inv_smul₀ <| ENNReal.toReal_ne_zero.2 ⟨hμs, measure_ne_top _ _⟩]


lemma condexp_set_generateFrom_singleton (hs : MeasurableSet s) (ht : MeasurableSet t) :
    μ⟦t | generateFrom {s}⟧ =ᵐ[μ.restrict s] fun _ ↦ (μ[t|s]).toReal := by
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s t : Set Ω
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp (Measu …
  -/
  rw [← integral_indicator_one ht]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s t : Set Ω
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (MeasureTheory.condexp (Measu …
  -/
  exact condexp_generateFrom_singleton hs <| Integrable.indicator (integrable_const 1) ht
  /-
    🎉 no goals
  -/


lemma condexpKernel_singleton_ae_eq_cond [StandardBorelSpace Ω] (hs : MeasurableSet s)
    (ht : MeasurableSet t) :
    ∀ᵐ ω ∂μ.restrict s,
      condexpKernel μ (generateFrom {s}) ω t = μ[t|s] := by
  have : (fun ω ↦ (condexpKernel μ (generateFrom {s}) ω t).toReal) =ᵐ[μ.restrict s]
      μ⟦t | generateFrom {s}⟧ :=
    ae_restrict_le hs <| condexpKernel_ae_eq_condexp
      (generateFrom_singleton_le hs) ht
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    s t : Set Ω
    inst✝ : StandardBorelSpace Ω
    hs : MeasurableSet s
    ht : MeasurableSet t
    this : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun ω => (((Probability …
    ⊢ Filter.Eventually (fun ω => Eq (((ProbabilityTheory.condexpKernel μ (Measura …
  -/
  filter_upwards [condexp_set_generateFrom_singleton hs ht, this] with ω hω₁ hω₂
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    s t : Set Ω
    inst✝ : StandardBorelSpace Ω
    hs : MeasurableSet s
    ht : MeasurableSet t
    this : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun ω => (((Probability …
    ω : Ω
    hω₁ : Eq (MeasureTheory.condexp (MeasurableSpace.generateFrom (Singleton.singl …
    hω₂ : Eq (((ProbabilityTheory.condexpKernel μ (MeasurableSpace.generateFrom (S …
    ⊢ Eq (((ProbabilityTheory.condexpKernel μ (MeasurableSpace.generateFrom (Singl …
  -/
  rwa [hω₁, ENNReal.toReal_eq_toReal (measure_ne_top _ t) (measure_ne_top _ t)] at hω₂
  /-
    🎉 no goals
  -/


