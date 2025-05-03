noncomputable instance : ChartedSpace ℂ ℍ :=
  UpperHalfPlane.isOpenEmbedding_coe.singletonChartedSpace


instance : SmoothManifoldWithCorners 𝓘(ℂ) ℍ :=
  UpperHalfPlane.isOpenEmbedding_coe.singleton_smoothManifoldWithCorners


/-- The inclusion map `ℍ → ℂ` is a smooth map of manifolds. -/
theorem contMDiff_coe : ContMDiff 𝓘(ℂ) 𝓘(ℂ) ⊤ ((↑) : ℍ → ℂ) := fun _ => contMDiffAt_extChartAt


@[deprecated (since := "2024-11-20")] alias smooth_coe := contMDiff_coe


/-- The inclusion map `ℍ → ℂ` is a differentiable map of manifolds. -/
theorem mdifferentiable_coe : MDifferentiable 𝓘(ℂ) 𝓘(ℂ) ((↑) : ℍ → ℂ) :=
                                    /-
                                      ⊢ LE.le 1 Top.top
                                    -/
  contMDiff_coe.mdifferentiable (by simp)
                                    /-
                                      🎉 no goals
                                    -/


lemma contMDiffAt_ofComplex {z : ℂ} (hz : 0 < z.im) :
    ContMDiffAt 𝓘(ℂ) 𝓘(ℂ) ⊤ ofComplex z := by
  /-
    z : Complex
    hz : LT.lt 0 z.im
    ⊢ ContMDiffAt (modelWithCornersSelf Complex Complex) (modelWithCornersSelf Com …
  -/
  rw [contMDiffAt_iff]
  /-
    z : Complex
    hz : LT.lt 0 z.im
    ⊢ And (ContinuousAt (↑UpperHalfPlane.ofComplex) z) (ContDiffWithinAt Complex ( …
  -/
  constructor
  · -- continuity at z
    /-
      case left
      z : Complex
      hz : LT.lt 0 z.im
      ⊢ ContinuousAt (↑UpperHalfPlane.ofComplex) z
    -/
    rw [ContinuousAt, nhds_induced, tendsto_comap_iff]
    /-
      case left
      z : Complex
      hz : LT.lt 0 z.im
      ⊢ Filter.Tendsto (Function.comp Subtype.val ↑UpperHalfPlane.ofComplex) (nhds z …
    -/
    refine Tendsto.congr' (eventuallyEq_coe_comp_ofComplex hz).symm ?_
    /-
      case left
      z : Complex
      hz : LT.lt 0 z.im
      ⊢ Filter.Tendsto id (nhds z) (nhds ↑(↑UpperHalfPlane.ofComplex z))
    -/
    simpa only [ofComplex_apply_of_im_pos hz, Subtype.coe_mk] using tendsto_id
    /-
      🎉 no goals
    -/
  · -- smoothness in local chart
    simp only [extChartAt, PartialHomeomorph.extend, modelWithCornersSelf_partialEquiv,
      PartialEquiv.trans_refl, PartialHomeomorph.toFun_eq_coe, PartialHomeomorph.refl_partialEquiv,
      PartialEquiv.refl_source, PartialHomeomorph.singletonChartedSpace_chartAt_eq,
      PartialEquiv.refl_symm, PartialEquiv.refl_coe, CompTriple.comp_eq, modelWithCornersSelf_coe,
      Set.range_id, id_eq, contDiffWithinAt_univ]
    /-
      case right
      z : Complex
      hz : LT.lt 0 z.im
      ⊢ ContDiffAt Complex (↑Top.top) (Function.comp ↑(chartAt Complex (↑UpperHalfPl …
    -/
    exact contDiffAt_id.congr_of_eventuallyEq (eventuallyEq_coe_comp_ofComplex hz)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-20")] alias smoothAt_ofComplex := contMDiffAt_ofComplex


lemma mdifferentiableAt_ofComplex {z : ℂ} (hz : 0 < z.im) :
    MDifferentiableAt 𝓘(ℂ) 𝓘(ℂ) ofComplex z :=
                                                   /-
                                                     z : Complex
                                                     hz : LT.lt 0 z.im
                                                     ⊢ LE.le 1 Top.top
                                                   -/
  (contMDiffAt_ofComplex hz).mdifferentiableAt (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma mdifferentiableAt_iff {f : ℍ → ℂ} {τ : ℍ} :
    MDifferentiableAt 𝓘(ℂ) 𝓘(ℂ) f τ ↔ DifferentiableAt ℂ (f ∘ ofComplex) ↑τ := by
  /-
    f : UpperHalfPlane → Complex
    τ : UpperHalfPlane
    ⊢ Iff (MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCorn …
  -/
  rw [← mdifferentiableAt_iff_differentiableAt]
  /-
    f : UpperHalfPlane → Complex
    τ : UpperHalfPlane
    ⊢ Iff (MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCorn …
  -/
  refine ⟨fun hf ↦ ?_, fun hf ↦ ?_⟩
    /-
      case refine_1
      f : UpperHalfPlane → Complex
      τ : UpperHalfPlane
      hf : MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCorner …
      ⊢ MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCornersSe …
    -/
  · exact (ofComplex_apply τ ▸ hf).comp _ (mdifferentiableAt_ofComplex τ.im_pos)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      f : UpperHalfPlane → Complex
      τ : UpperHalfPlane
      hf : MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCorner …
      ⊢ MDifferentiableAt (modelWithCornersSelf Complex Complex) (modelWithCornersSe …
    -/
  · simpa only [Function.comp_def, ofComplex_apply] using hf.comp τ (mdifferentiable_coe τ)
    /-
      🎉 no goals
    -/


lemma mdifferentiable_iff {f : ℍ → ℂ} :
    MDifferentiable 𝓘(ℂ) 𝓘(ℂ) f ↔ DifferentiableOn ℂ (f ∘ ofComplex) {z | 0 < z.im} :=
  ⟨fun h z hz ↦ (mdifferentiableAt_iff.mp (h ⟨z, hz⟩)).differentiableWithinAt,
    fun h ⟨z, hz⟩ ↦ mdifferentiableAt_iff.mpr <| (h z hz).differentiableAt
      <| (Complex.continuous_im.isOpen_preimage _ isOpen_Ioi).mem_nhds hz⟩


