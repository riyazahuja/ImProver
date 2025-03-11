/-- A map `f : α → α` is said to be pre-ergodic with respect to a measure `μ` if any measurable
strictly invariant set is either almost empty or full. -/
structure PreErgodic (f : α → α) (μ : Measure α := by volume_tac) : Prop where
  aeconst_set ⦃s⦄ : MeasurableSet s → f ⁻¹' s = s → EventuallyConst s (ae μ)


/-- A map `f : α → α` is said to be ergodic with respect to a measure `μ` if it is measure
preserving and pre-ergodic. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure Ergodic (f : α → α) (μ : Measure α := by volume_tac) extends
  MeasurePreserving f μ μ, PreErgodic f μ : Prop


/-- A map `f : α → α` is said to be quasi ergodic with respect to a measure `μ` if it is quasi
measure preserving and pre-ergodic. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure QuasiErgodic (f : α → α) (μ : Measure α := by volume_tac) extends
  QuasiMeasurePreserving f μ μ, PreErgodic f μ : Prop


theorem ae_empty_or_univ (hf : PreErgodic f μ) (hs : MeasurableSet s) (hfs : f ⁻¹' s = s) :
    s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : PreErgodic f μ
    hs : MeasurableSet s
    hfs : Eq (Set.preimage f s) s
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  simpa only [eventuallyConst_set'] using hf.aeconst_set hs hfs
  /-
    🎉 no goals
  -/


theorem measure_self_or_compl_eq_zero (hf : PreErgodic f μ) (hs : MeasurableSet s)
    (hs' : f ⁻¹' s = s) : μ s = 0 ∨ μ sᶜ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : PreErgodic f μ
    hs : MeasurableSet s
    hs' : Eq (Set.preimage f s) s
    ⊢ Or (Eq (μ s) 0) (Eq (μ (HasCompl.compl s)) 0)
  -/
  simpa using hf.ae_empty_or_univ hs hs'
  /-
    🎉 no goals
  -/


theorem ae_mem_or_ae_nmem (hf : PreErgodic f μ) (hsm : MeasurableSet s) (hs : f ⁻¹' s = s) :
    (∀ᵐ x ∂μ, x ∈ s) ∨ ∀ᵐ x ∂μ, x ∉ s :=
  eventuallyConst_set.1 <| hf.aeconst_set hsm hs


/-- On a probability space, the (pre)ergodicity condition is a zero one law. -/
theorem prob_eq_zero_or_one [IsProbabilityMeasure μ] (hf : PreErgodic f μ) (hs : MeasurableSet s)
    (hs' : f ⁻¹' s = s) : μ s = 0 ∨ μ s = 1 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : PreErgodic f μ
    hs : MeasurableSet s
    hs' : Eq (Set.preimage f s) s
    ⊢ Or (Eq (μ s) 0) (Eq (μ s) 1)
  -/
  simpa [hs] using hf.measure_self_or_compl_eq_zero hs hs'
  /-
    🎉 no goals
  -/


theorem of_iterate (n : ℕ) (hf : PreErgodic f^[n] μ) : PreErgodic f μ :=
  ⟨fun _ hs hs' => hf.aeconst_set hs <| IsFixedPt.preimage_iterate hs' n⟩


theorem smul_measure {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (hf : PreErgodic f μ) (c : R) : PreErgodic f (c • μ) where
  aeconst_set _s hs hfs := (hf.aeconst_set hs hfs).anti <| ae_smul_measure_le _


theorem zero_measure (f : α → α) : @PreErgodic α m f 0 where
                          /-
                            α : Type u_1
                            m : MeasurableSpace α
                            f : α → α
                            x✝² : Set α
                            x✝¹ : MeasurableSet x✝²
                            x✝ : Eq (Set.preimage f x✝²) x✝²
                            ⊢ Filter.EventuallyConst x✝² (MeasureTheory.ae 0)
                          -/
  aeconst_set _ _ _ := by simp
                          /-
                            🎉 no goals
                          -/


theorem preErgodic_of_preErgodic_conjugate (hg : MeasurePreserving g μ μ') (hf : PreErgodic f μ)
    {f' : β → β} (h_comm : Semiconj g f f') : PreErgodic f' μ' where
  aeconst_set s hs₀ hs₁ := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      β : Type u_2
      m' : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      g : α → β
      hg : MeasureTheory.MeasurePreserving g μ μ'
      hf : PreErgodic f μ
      f' : β → β
      h_comm : Function.Semiconj g f f'
      s : Set β
      hs₀ : MeasurableSet s
      hs₁ : Eq (Set.preimage f' s) s
      ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ')
    -/
    rw [← hg.aeconst_preimage hs₀.nullMeasurableSet]
    /-
      α : Type u_1
      m : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      β : Type u_2
      m' : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      g : α → β
      hg : MeasureTheory.MeasurePreserving g μ μ'
      hf : PreErgodic f μ
      f' : β → β
      h_comm : Function.Semiconj g f f'
      s : Set β
      hs₀ : MeasurableSet s
      hs₁ : Eq (Set.preimage f' s) s
      ⊢ Filter.EventuallyConst (Set.preimage g s) (MeasureTheory.ae μ)
    -/
    apply hf.aeconst_set (hg.measurable hs₀)
    /-
      α : Type u_1
      m : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      β : Type u_2
      m' : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      g : α → β
      hg : MeasureTheory.MeasurePreserving g μ μ'
      hf : PreErgodic f μ
      f' : β → β
      h_comm : Function.Semiconj g f f'
      s : Set β
      hs₀ : MeasurableSet s
      hs₁ : Eq (Set.preimage f' s) s
      ⊢ Eq (Set.preimage f (Set.preimage g s)) (Set.preimage g s)
    -/
    rw [← preimage_comp, h_comm.comp_eq, preimage_comp, hs₁]
    /-
      🎉 no goals
    -/


theorem preErgodic_conjugate_iff {e : α ≃ᵐ β} (h : MeasurePreserving e μ μ') :
    PreErgodic (e ∘ f ∘ e.symm) μ' ↔ PreErgodic f μ := by
  refine ⟨fun hf => preErgodic_of_preErgodic_conjugate (h.symm e) hf ?_,
      fun hf => preErgodic_of_preErgodic_conjugate h hf ?_⟩
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      β : Type u_2
      m' : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      e : MeasurableEquiv α β
      h : MeasureTheory.MeasurePreserving (⇑e) μ μ'
      hf : PreErgodic (Function.comp (⇑e) (Function.comp f ⇑e.symm)) μ'
      ⊢ Function.Semiconj (⇑e.symm) (Function.comp (⇑e) (Function.comp f ⇑e.symm)) f
    -/
  · simp [Semiconj]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      f : α → α
      μ : MeasureTheory.Measure α
      β : Type u_2
      m' : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      e : MeasurableEquiv α β
      h : MeasureTheory.MeasurePreserving (⇑e) μ μ'
      hf : PreErgodic f μ
      ⊢ Function.Semiconj (⇑e) f (Function.comp (⇑e) (Function.comp f ⇑e.symm))
    -/
  · simp [Semiconj]
    /-
      🎉 no goals
    -/


theorem ergodic_conjugate_iff {e : α ≃ᵐ β} (h : MeasurePreserving e μ μ') :
    Ergodic (e ∘ f ∘ e.symm) μ' ↔ Ergodic f μ := by
  have : MeasurePreserving (e ∘ f ∘ e.symm) μ' μ' ↔ MeasurePreserving f μ μ := by
    rw [h.comp_left_iff, (MeasurePreserving.symm e h).comp_right_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → α
    μ : MeasureTheory.Measure α
    β : Type u_2
    m' : MeasurableSpace β
    μ' : MeasureTheory.Measure β
    e : MeasurableEquiv α β
    h : MeasureTheory.MeasurePreserving (⇑e) μ μ'
    this : Iff (MeasureTheory.MeasurePreserving (Function.comp (⇑e) (Function.comp …
    ⊢ Iff (Ergodic (Function.comp (⇑e) (Function.comp f ⇑e.symm)) μ') (Ergodic f μ)
  -/
  replace h : PreErgodic (e ∘ f ∘ e.symm) μ' ↔ PreErgodic f μ := h.preErgodic_conjugate_iff
  exact ⟨fun hf => { this.mp hf.toMeasurePreserving, h.mp hf.toPreErgodic with },
    fun hf => { this.mpr hf.toMeasurePreserving, h.mpr hf.toPreErgodic with }⟩


theorem aeconst_set₀ (hf : QuasiErgodic f μ) (hsm : NullMeasurableSet s μ) (hs : f ⁻¹' s =ᵐ[μ] s) :
    EventuallyConst s (ae μ) :=
  let ⟨_t, h₀, h₁, h₂⟩ := hf.toQuasiMeasurePreserving.exists_preimage_eq_of_preimage_ae hsm hs
  (hf.aeconst_set h₀ h₂).congr h₁


/-- For a quasi ergodic map, sets that are almost invariant (rather than strictly invariant) are
still either almost empty or full. -/
theorem ae_empty_or_univ₀ (hf : QuasiErgodic f μ) (hsm : NullMeasurableSet s μ)
    (hs : f ⁻¹' s =ᵐ[μ] s) :
    s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ :=
  eventuallyConst_set'.mp <| hf.aeconst_set₀ hsm hs


@[deprecated (since := "2024-07-21")] alias ae_empty_or_univ' := ae_empty_or_univ₀


/-- For a quasi ergodic map, sets that are almost invariant (rather than strictly invariant) are
still either almost empty or full. -/
theorem ae_mem_or_ae_nmem₀ (hf : QuasiErgodic f μ) (hsm : NullMeasurableSet s μ)
    (hs : f ⁻¹' s =ᵐ[μ] s) :
    (∀ᵐ x ∂μ, x ∈ s) ∨ ∀ᵐ x ∂μ, x ∉ s :=
  eventuallyConst_set.mp <| hf.aeconst_set₀ hsm hs


theorem smul_measure {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (hf : QuasiErgodic f μ) (c : R) : QuasiErgodic f (c • μ) :=
  ⟨hf.1.smul_measure _, hf.2.smul_measure _⟩


theorem zero_measure {f : α → α} (hf : Measurable f) : @QuasiErgodic α m f 0 where
  measurable := hf
                             /-
                               α : Type u_1
                               m : MeasurableSpace α
                               f : α → α
                               hf : Measurable f
                               ⊢ (MeasureTheory.Measure.map f 0).AbsolutelyContinuous 0
                             -/
  absolutelyContinuous := by simp
                             /-
                               🎉 no goals
                             -/
  toPreErgodic := .zero_measure f


/-- An ergodic map is quasi ergodic. -/
theorem quasiErgodic (hf : Ergodic f μ) : QuasiErgodic f μ :=
  { hf.toPreErgodic, hf.toMeasurePreserving.quasiMeasurePreserving with }


/-- See also `Ergodic.ae_empty_or_univ_of_preimage_ae_le`. -/
theorem ae_empty_or_univ_of_preimage_ae_le' (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : f ⁻¹' s ≤ᵐ[μ] s) (h_fin : μ s ≠ ∞) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE (Set.preimage f s) s
    h_fin : Ne (μ s) Top.top
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  refine hf.quasiErgodic.ae_empty_or_univ₀ hs ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE (Set.preimage f s) s
    h_fin : Ne (μ s) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
  -/
  refine ae_eq_of_ae_subset_of_measure_ge hs' (hf.measure_preimage hs).ge ?_ h_fin
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE (Set.preimage f s) s
    h_fin : Ne (μ s) Top.top
    ⊢ MeasureTheory.NullMeasurableSet (Set.preimage f s) μ
  -/
  exact hs.preimage hf.quasiMeasurePreserving
  /-
    🎉 no goals
  -/


/-- See also `Ergodic.ae_empty_or_univ_of_ae_le_preimage`. -/
theorem ae_empty_or_univ_of_ae_le_preimage' (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : s ≤ᵐ[μ] f ⁻¹' s) (h_fin : μ s ≠ ∞) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE s (Set.preimage f s)
    h_fin : Ne (μ s) Top.top
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  replace h_fin : μ (f ⁻¹' s) ≠ ∞ := by rwa [hf.measure_preimage hs]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE s (Set.preimage f s)
    h_fin : Ne (μ (Set.preimage f s)) Top.top
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  refine hf.quasiErgodic.ae_empty_or_univ₀ hs ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyLE s (Set.preimage f s)
    h_fin : Ne (μ (Set.preimage f s)) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
  -/
  exact (ae_eq_of_ae_subset_of_measure_ge hs' (hf.measure_preimage hs).le hs h_fin).symm
  /-
    🎉 no goals
  -/


/-- See also `Ergodic.ae_empty_or_univ_of_image_ae_le`. -/
theorem ae_empty_or_univ_of_image_ae_le' (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : f '' s ≤ᵐ[μ] s) (h_fin : μ s ≠ ∞) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ := by
  replace hs' : s ≤ᵐ[μ] f ⁻¹' s :=
    (HasSubset.Subset.eventuallyLE (subset_preimage_image f s)).trans
      (hf.quasiMeasurePreserving.preimage_mono_ae hs')
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    f : α → α
    μ : MeasureTheory.Measure α
    hf : Ergodic f μ
    hs : MeasureTheory.NullMeasurableSet s μ
    h_fin : Ne (μ s) Top.top
    hs' : (MeasureTheory.ae μ).EventuallyLE s (Set.preimage f s)
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  exact ae_empty_or_univ_of_ae_le_preimage' hf hs hs' h_fin
  /-
    🎉 no goals
  -/


theorem smul_measure {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (hf : Ergodic f μ) (c : R) : Ergodic f (c • μ) :=
  ⟨hf.1.smul_measure _, hf.2.smul_measure _⟩


theorem zero_measure {f : α → α} (hf : Measurable f) : @Ergodic α m f 0 where
  measurable := hf
               /-
                 α : Type u_1
                 m : MeasurableSpace α
                 f : α → α
                 hf : Measurable f
                 ⊢ Eq (MeasureTheory.Measure.map f 0) 0
               -/
  map_eq := by simp
               /-
                 🎉 no goals
               -/
  toPreErgodic := .zero_measure f


theorem ae_empty_or_univ_of_preimage_ae_le (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : f ⁻¹' s ≤ᵐ[μ] s) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ :=
  ae_empty_or_univ_of_preimage_ae_le' hf hs hs' <| measure_ne_top μ s


theorem ae_empty_or_univ_of_ae_le_preimage (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : s ≤ᵐ[μ] f ⁻¹' s) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ :=
  ae_empty_or_univ_of_ae_le_preimage' hf hs hs' <| measure_ne_top μ s


theorem ae_empty_or_univ_of_image_ae_le (hf : Ergodic f μ) (hs : NullMeasurableSet s μ)
    (hs' : f '' s ≤ᵐ[μ] s) : s =ᵐ[μ] (∅ : Set α) ∨ s =ᵐ[μ] univ :=
  ae_empty_or_univ_of_image_ae_le' hf hs hs' <| measure_ne_top μ s


