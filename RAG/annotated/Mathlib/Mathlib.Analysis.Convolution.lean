theorem convolution_integrand_bound_right_of_le_of_subset {C : ℝ} (hC : ∀ i, ‖g i‖ ≤ C) {x t : G}
    {s u : Set G} (hx : x ∈ s) (hu : -tsupport g + s ⊆ u) :
    ‖L (f t) (g (x - t))‖ ≤ u.indicator (fun t => ‖L‖ * ‖f t‖ * C) t := by
  -- Porting note: had to add `f := _`
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜 E'
    inst✝² : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : AddGroup G
    inst✝ : TopologicalSpace G
    C : Real
    hC : ∀ (i : G), LE.le (Norm.norm (g i)) C
    x t : G
    s u : Set G
    hx : Membership.mem s x
    hu : HasSubset.Subset (HAdd.hAdd (Neg.neg (tsupport g)) s) u
    ⊢ LE.le (Norm.norm ((L (f t)) (g (HSub.hSub x t)))) (u.indicator (fun t => HMu …
  -/
  refine le_indicator (f := fun t ↦ ‖L (f t) (g (x - t))‖) (fun t _ => ?_) (fun t ht => ?_) t
    /-
      case refine_1
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedAddCommGroup E'
      inst✝⁶ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜 E'
      inst✝² : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝¹ : AddGroup G
      inst✝ : TopologicalSpace G
      C : Real
      hC : ∀ (i : G), LE.le (Norm.norm (g i)) C
      x t✝ : G
      s u : Set G
      hx : Membership.mem s x
      hu : HasSubset.Subset (HAdd.hAdd (Neg.neg (tsupport g)) s) u
      t : G
      x✝ : Membership.mem u t
      ⊢ LE.le ((fun t => Norm.norm ((L (f t)) (g (HSub.hSub x t)))) t) (HMul.hMul (H …
    -/
  · apply_rules [L.le_of_opNorm₂_le_of_le, le_rfl]
    /-
      🎉 no goals
    -/
  · have : x - t ∉ support g := by
      refine mt (fun hxt => hu ?_) ht
      refine ⟨_, Set.neg_mem_neg.mpr (subset_closure hxt), _, hx, ?_⟩
      simp only [neg_sub, sub_add_cancel]
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedAddCommGroup E'
      inst✝⁶ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜 E'
      inst✝² : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝¹ : AddGroup G
      inst✝ : TopologicalSpace G
      C : Real
      hC : ∀ (i : G), LE.le (Norm.norm (g i)) C
      x t✝ : G
      s u : Set G
      hx : Membership.mem s x
      hu : HasSubset.Subset (HAdd.hAdd (Neg.neg (tsupport g)) s) u
      t : G
      ht : Not (Membership.mem u t)
      this : Not (Membership.mem (Function.support g) (HSub.hSub x t))
      ⊢ LE.le ((fun t => Norm.norm ((L (f t)) (g (HSub.hSub x t)))) t) 0
    -/
    simp only [nmem_support.mp this, (L _).map_zero, norm_zero, le_rfl]
    /-
      🎉 no goals
    -/


theorem _root_.HasCompactSupport.convolution_integrand_bound_right_of_subset
    (hcg : HasCompactSupport g) (hg : Continuous g)
    {x t : G} {s u : Set G} (hx : x ∈ s) (hu : -tsupport g + s ⊆ u) :
    ‖L (f t) (g (x - t))‖ ≤ u.indicator (fun t => ‖L‖ * ‖f t‖ * ⨆ i, ‖g i‖) t := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜 E'
    inst✝² : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : AddGroup G
    inst✝ : TopologicalSpace G
    hcg : HasCompactSupport g
    hg : Continuous g
    x t : G
    s u : Set G
    hx : Membership.mem s x
    hu : HasSubset.Subset (HAdd.hAdd (Neg.neg (tsupport g)) s) u
    ⊢ LE.le (Norm.norm ((L (f t)) (g (HSub.hSub x t)))) (u.indicator (fun t => HMu …
  -/
  refine convolution_integrand_bound_right_of_le_of_subset _ (fun i => ?_) hx hu
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜 E'
    inst✝² : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : AddGroup G
    inst✝ : TopologicalSpace G
    hcg : HasCompactSupport g
    hg : Continuous g
    x t : G
    s u : Set G
    hx : Membership.mem s x
    hu : HasSubset.Subset (HAdd.hAdd (Neg.neg (tsupport g)) s) u
    i : G
    ⊢ LE.le (Norm.norm (g i)) (iSup fun i => Norm.norm (g i))
  -/
  exact le_ciSup (hg.norm.bddAbove_range_of_hasCompactSupport hcg.norm) _
  /-
    🎉 no goals
  -/


theorem _root_.HasCompactSupport.convolution_integrand_bound_right (hcg : HasCompactSupport g)
    (hg : Continuous g) {x t : G} {s : Set G} (hx : x ∈ s) :
    ‖L (f t) (g (x - t))‖ ≤ (-tsupport g + s).indicator (fun t => ‖L‖ * ‖f t‖ * ⨆ i, ‖g i‖) t :=
  hcg.convolution_integrand_bound_right_of_subset L hg hx Subset.rfl


theorem _root_.Continuous.convolution_integrand_fst [ContinuousSub G] (hg : Continuous g) (t : G) :
    Continuous fun x => L (f t) (g (x - t)) :=
  L.continuous₂.comp₂ continuous_const <| hg.comp <| continuous_id.sub continuous_const


theorem _root_.HasCompactSupport.convolution_integrand_bound_left (hcf : HasCompactSupport f)
    (hf : Continuous f) {x t : G} {s : Set G} (hx : x ∈ s) :
    ‖L (f (x - t)) (g t)‖ ≤
      (-tsupport f + s).indicator (fun t => (‖L‖ * ⨆ i, ‖f i‖) * ‖g t‖) t := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜 E'
    inst✝² : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : AddGroup G
    inst✝ : TopologicalSpace G
    hcf : HasCompactSupport f
    hf : Continuous f
    x t : G
    s : Set G
    hx : Membership.mem s x
    ⊢ LE.le (Norm.norm ((L (f (HSub.hSub x t))) (g t))) ((HAdd.hAdd (Neg.neg (tsup …
  -/
  convert hcf.convolution_integrand_bound_right L.flip hf hx using 1
  /-
    case h.e'_4
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜 E'
    inst✝² : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : AddGroup G
    inst✝ : TopologicalSpace G
    hcf : HasCompactSupport f
    hf : Continuous f
    x t : G
    s : Set G
    hx : Membership.mem s x
    ⊢ Eq ((HAdd.hAdd (Neg.neg (tsupport f)) s).indicator (fun t => HMul.hMul (HMul …
  -/
  simp_rw [L.opNorm_flip, mul_right_comm]
  /-
    🎉 no goals
  -/


/-- The convolution of `f` and `g` exists at `x` when the function `t ↦ L (f t) (g (x - t))` is
integrable. There are various conditions on `f` and `g` to prove this. -/
def ConvolutionExistsAt [Sub G] (f : G → E) (g : G → E') (x : G) (L : E →L[𝕜] E' →L[𝕜] F)
    (μ : Measure G := by volume_tac) : Prop :=
  Integrable (fun t => L (f t) (g (x - t))) μ


/-- The convolution of `f` and `g` exists when the function `t ↦ L (f t) (g (x - t))` is integrable
for all `x : G`. There are various conditions on `f` and `g` to prove this. -/
def ConvolutionExists [Sub G] (f : G → E) (g : G → E') (L : E →L[𝕜] E' →L[𝕜] F)
    (μ : Measure G := by volume_tac) : Prop :=
  ∀ x : G, ConvolutionExistsAt f g x L μ


variable {L} in
theorem ConvolutionExistsAt.integrable [Sub G] {x : G} (h : ConvolutionExistsAt f g x L μ) :
    Integrable (fun t => L (f t) (g (x - t))) μ :=
  h


theorem AEStronglyMeasurable.convolution_integrand' [MeasurableAdd₂ G]
    [MeasurableNeg G] [SFinite ν] (hf : AEStronglyMeasurable f ν)
    (hg : AEStronglyMeasurable g <| map (fun p : G × G => p.1 - p.2) (μ.prod ν)) :
    AEStronglyMeasurable (fun p : G × G => L (f p.2) (g (p.1 - p.2))) (μ.prod ν) :=
  L.aestronglyMeasurable_comp₂ hf.snd <| hg.comp_measurable measurable_sub


theorem AEStronglyMeasurable.convolution_integrand_snd'
    (hf : AEStronglyMeasurable f μ) {x : G}
    (hg : AEStronglyMeasurable g <| map (fun t => x - t) μ) :
    AEStronglyMeasurable (fun t => L (f t) (g (x - t))) μ :=
  L.aestronglyMeasurable_comp₂ hf <| hg.comp_measurable <| measurable_id.const_sub x


theorem AEStronglyMeasurable.convolution_integrand_swap_snd' {x : G}
    (hf : AEStronglyMeasurable f <| map (fun t => x - t) μ) (hg : AEStronglyMeasurable g μ) :
    AEStronglyMeasurable (fun t => L (f (x - t)) (g t)) μ :=
  L.aestronglyMeasurable_comp₂ (hf.comp_measurable <| measurable_id.const_sub x) hg


/-- A sufficient condition to prove that `f ⋆[L, μ] g` exists.
We assume that `f` is integrable on a set `s` and `g` is bounded and ae strongly measurable
on `x₀ - s` (note that both properties hold if `g` is continuous with compact support). -/
theorem _root_.BddAbove.convolutionExistsAt' {x₀ : G} {s : Set G}
    (hbg : BddAbove ((fun i => ‖g i‖) '' ((fun t => -t + x₀) ⁻¹' s))) (hs : MeasurableSet s)
    (h2s : (support fun t => L (f t) (g (x₀ - t))) ⊆ s) (hf : IntegrableOn f s μ)
    (hmg : AEStronglyMeasurable g <| map (fun t => x₀ - t) (μ.restrict s)) :
    ConvolutionExistsAt f g x₀ L μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    s : Set G
    hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
    hs : MeasurableSet s
    h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
    hf : MeasureTheory.IntegrableOn f s μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  rw [ConvolutionExistsAt]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    s : Set G
    hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
    hs : MeasurableSet s
    h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
    hf : MeasureTheory.IntegrableOn f s μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    ⊢ MeasureTheory.Integrable (fun t => (L (f t)) (g (HSub.hSub x₀ t))) μ
  -/
  rw [← integrableOn_iff_integrable_of_support_subset h2s]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    s : Set G
    hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
    hs : MeasurableSet s
    h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
    hf : MeasureTheory.IntegrableOn f s μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    ⊢ MeasureTheory.IntegrableOn (fun t => (L (f t)) (g (HSub.hSub x₀ t))) s μ
  -/
  set s' := (fun t => -t + x₀) ⁻¹' s
  have : ∀ᵐ t : G ∂μ.restrict s,
      ‖L (f t) (g (x₀ - t))‖ ≤ s.indicator (fun t => ‖L‖ * ‖f t‖ * ⨆ i : s', ‖g i‖) t := by
    filter_upwards
    refine le_indicator (fun t ht => ?_) fun t ht => ?_
    · apply_rules [L.le_of_opNorm₂_le_of_le, le_rfl]
      refine (le_ciSup_set hbg <| mem_preimage.mpr ?_)
      rwa [neg_sub, sub_add_cancel]
    · have : t ∉ support fun t => L (f t) (g (x₀ - t)) := mt (fun h => h2s h) ht
      rw [nmem_support.mp this, norm_zero]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    s : Set G
    hs : MeasurableSet s
    h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
    hf : MeasureTheory.IntegrableOn f s μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    s' : Set G := Set.preimage (fun t => HAdd.hAdd (Neg.neg t) x₀) s
    hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) s')
    this : Filter.Eventually (fun t => LE.le (Norm.norm ((L (f t)) (g (HSub.hSub x …
    ⊢ MeasureTheory.IntegrableOn (fun t => (L (f t)) (g (HSub.hSub x₀ t))) s μ
  -/
  refine Integrable.mono' ?_ ?_ this
    /-
      case refine_1
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜 E'
      inst✝⁴ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝³ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝² : AddGroup G
      inst✝¹ : MeasurableAdd G
      inst✝ : MeasurableNeg G
      x₀ : G
      s : Set G
      hs : MeasurableSet s
      h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
      hf : MeasureTheory.IntegrableOn f s μ
      hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
      s' : Set G := Set.preimage (fun t => HAdd.hAdd (Neg.neg t) x₀) s
      hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) s')
      this : Filter.Eventually (fun t => LE.le (Norm.norm ((L (f t)) (g (HSub.hSub x …
      ⊢ MeasureTheory.Integrable (s.indicator fun t => HMul.hMul (HMul.hMul (Norm.no …
    -/
  · rw [integrable_indicator_iff hs]; exact ((hf.norm.const_mul _).mul_const _).integrableOn
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜 E'
      inst✝⁴ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝³ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝² : AddGroup G
      inst✝¹ : MeasurableAdd G
      inst✝ : MeasurableNeg G
      x₀ : G
      s : Set G
      hs : MeasurableSet s
      h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
      hf : MeasureTheory.IntegrableOn f s μ
      hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
      s' : Set G := Set.preimage (fun t => HAdd.hAdd (Neg.neg t) x₀) s
      hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) s')
      this : Filter.Eventually (fun t => LE.le (Norm.norm ((L (f t)) (g (HSub.hSub x …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun t => (L (f t)) (g (HSub.hSub x₀ t))) …
    -/
  · exact hf.aestronglyMeasurable.convolution_integrand_snd' L hmg
    /-
      🎉 no goals
    -/


/-- If `‖f‖ *[μ] ‖g‖` exists, then `f *[L, μ] g` exists. -/
theorem ConvolutionExistsAt.ofNorm' {x₀ : G}
    (h : ConvolutionExistsAt (fun x => ‖f x‖) (fun x => ‖g x‖) x₀ (mul ℝ ℝ) μ)
    (hmf : AEStronglyMeasurable f μ) (hmg : AEStronglyMeasurable g <| map (fun t => x₀ - t) μ) :
    ConvolutionExistsAt f g x₀ L μ := by
  refine (h.const_mul ‖L‖).mono'
    (hmf.convolution_integrand_snd' L hmg) (Eventually.of_forall fun x => ?_)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    h : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (fun x => Nor …
    hmf : MeasureTheory.AEStronglyMeasurable f μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    x : G
    ⊢ LE.le (Norm.norm ((L (f x)) (g (HSub.hSub x₀ x)))) (HMul.hMul (Norm.norm L)  …
  -/
  rw [mul_apply', ← mul_assoc]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : AddGroup G
    inst✝¹ : MeasurableAdd G
    inst✝ : MeasurableNeg G
    x₀ : G
    h : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (fun x => Nor …
    hmf : MeasureTheory.AEStronglyMeasurable f μ
    hmg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t = …
    x : G
    ⊢ LE.le (Norm.norm ((L (f x)) (g (HSub.hSub x₀ x)))) (HMul.hMul (HMul.hMul (No …
  -/
  apply L.le_opNorm₂
  /-
    🎉 no goals
  -/


theorem AEStronglyMeasurable.convolution_integrand_snd (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) (x : G) :
    AEStronglyMeasurable (fun t => L (f t) (g (x - t))) μ :=
  hf.convolution_integrand_snd' L <|
    hg.mono_ac <| (quasiMeasurePreserving_sub_left_of_right_invariant μ x).absolutelyContinuous


theorem AEStronglyMeasurable.convolution_integrand_swap_snd
    (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) (x : G) :
    AEStronglyMeasurable (fun t => L (f (x - t)) (g t)) μ :=
  (hf.mono_ac
        (quasiMeasurePreserving_sub_left_of_right_invariant μ
            x).absolutelyContinuous).convolution_integrand_swap_snd'
    L hg


/-- If `‖f‖ *[μ] ‖g‖` exists, then `f *[L, μ] g` exists. -/
theorem ConvolutionExistsAt.ofNorm {x₀ : G}
    (h : ConvolutionExistsAt (fun x => ‖f x‖) (fun x => ‖g x‖) x₀ (mul ℝ ℝ) μ)
    (hmf : AEStronglyMeasurable f μ) (hmg : AEStronglyMeasurable g μ) :
    ConvolutionExistsAt f g x₀ L μ :=
  h.ofNorm' L hmf <|
    hmg.mono_ac (quasiMeasurePreserving_sub_left_of_right_invariant μ x₀).absolutelyContinuous


theorem AEStronglyMeasurable.convolution_integrand (hf : AEStronglyMeasurable f ν)
    (hg : AEStronglyMeasurable g μ) :
    AEStronglyMeasurable (fun p : G × G => L (f p.2) (g (p.1 - p.2))) (μ.prod ν) :=
  hf.convolution_integrand' L <|
    hg.mono_ac (quasiMeasurePreserving_sub_of_right_invariant μ ν).absolutelyContinuous


theorem Integrable.convolution_integrand (hf : Integrable f ν) (hg : Integrable g μ) :
    Integrable (fun p : G × G => L (f p.2) (g (p.1 - p.2))) (μ.prod ν) := by
  have h_meas : AEStronglyMeasurable (fun p : G × G => L (f p.2) (g (p.1 - p.2))) (μ.prod ν) :=
    hf.aestronglyMeasurable.convolution_integrand L hg.aestronglyMeasurable
  have h2_meas : AEStronglyMeasurable (fun y : G => ∫ x : G, ‖L (f y) (g (x - y))‖ ∂μ) ν :=
    h_meas.prod_swap.norm.integral_prod_right'
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    inst✝⁵ : AddGroup G
    inst✝⁴ : MeasurableAdd₂ G
    inst✝³ : MeasurableNeg G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.SFinite ν
    hf : MeasureTheory.Integrable f ν
    hg : MeasureTheory.Integrable g μ
    h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    ⊢ MeasureTheory.Integrable (fun p => (L (f p.2)) (g (HSub.hSub p.1 p.2))) (μ.p …
  -/
  simp_rw [integrable_prod_iff' h_meas]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    inst✝⁵ : AddGroup G
    inst✝⁴ : MeasurableAdd₂ G
    inst✝³ : MeasurableNeg G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.SFinite ν
    hf : MeasureTheory.Integrable f ν
    hg : MeasureTheory.Integrable g μ
    h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    ⊢ And (Filter.Eventually (fun y => MeasureTheory.Integrable (fun x => (L (f y) …
  -/
  refine ⟨Eventually.of_forall fun t => (L (f t)).integrable_comp (hg.comp_sub_right t), ?_⟩
  refine Integrable.mono' ?_ h2_meas
      (Eventually.of_forall fun t => (?_ : _ ≤ ‖L‖ * ‖f t‖ * ∫ x, ‖g (x - t)‖ ∂μ))
    /-
      case refine_1
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ ν : MeasureTheory.Measure G
      inst✝⁵ : AddGroup G
      inst✝⁴ : MeasurableAdd₂ G
      inst✝³ : MeasurableNeg G
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : μ.IsAddRightInvariant
      inst✝ : MeasureTheory.SFinite ν
      hf : MeasureTheory.Integrable f ν
      hg : MeasureTheory.Integrable g μ
      h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
      h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
      ⊢ MeasureTheory.Integrable (fun t => HMul.hMul (HMul.hMul (Norm.norm L) (Norm. …
    -/
  · simp only [integral_sub_right_eq_self (‖g ·‖)]
    /-
      case refine_1
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ ν : MeasureTheory.Measure G
      inst✝⁵ : AddGroup G
      inst✝⁴ : MeasurableAdd₂ G
      inst✝³ : MeasurableNeg G
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : μ.IsAddRightInvariant
      inst✝ : MeasureTheory.SFinite ν
      hf : MeasureTheory.Integrable f ν
      hg : MeasureTheory.Integrable g μ
      h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
      h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
      ⊢ MeasureTheory.Integrable (fun t => HMul.hMul (HMul.hMul (Norm.norm L) (Norm. …
    -/
    exact (hf.norm.const_mul _).mul_const _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ ν : MeasureTheory.Measure G
      inst✝⁵ : AddGroup G
      inst✝⁴ : MeasurableAdd₂ G
      inst✝³ : MeasurableNeg G
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : μ.IsAddRightInvariant
      inst✝ : MeasureTheory.SFinite ν
      hf : MeasureTheory.Integrable f ν
      hg : MeasureTheory.Integrable g μ
      h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
      h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
      t : G
      ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun x => Norm.norm ((L (f t)) (g  …
    -/
  · simp_rw [← integral_mul_left]
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ ν : MeasureTheory.Measure G
      inst✝⁵ : AddGroup G
      inst✝⁴ : MeasurableAdd₂ G
      inst✝³ : MeasurableNeg G
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : μ.IsAddRightInvariant
      inst✝ : MeasureTheory.SFinite ν
      hf : MeasureTheory.Integrable f ν
      hg : MeasureTheory.Integrable g μ
      h_meas : MeasureTheory.AEStronglyMeasurable (fun p => (L (f p.2)) (g (HSub.hSu …
      h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
      t : G
      ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun x => Norm.norm ((L (f t)) (g  …
    -/
    rw [Real.norm_of_nonneg (by positivity)]
    exact integral_mono_of_nonneg (Eventually.of_forall fun t => norm_nonneg _)
      ((hg.comp_sub_right t).norm.const_mul _) (Eventually.of_forall fun t => L.le_opNorm₂ _ _)


theorem Integrable.ae_convolution_exists (hf : Integrable f ν) (hg : Integrable g μ) :
    ∀ᵐ x ∂μ, ConvolutionExistsAt f g x L ν :=
  ((integrable_prod_iff <|
          hf.aestronglyMeasurable.convolution_integrand L hg.aestronglyMeasurable).mp <|
      hf.convolution_integrand L hg).1


theorem _root_.HasCompactSupport.convolutionExistsAt {x₀ : G}
    (h : HasCompactSupport fun t => L (f t) (g (x₀ - t))) (hf : LocallyIntegrable f μ)
    (hg : Continuous g) : ConvolutionExistsAt f g x₀ L μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    x₀ : G
    h : HasCompactSupport fun t => (L (f t)) (g (HSub.hSub x₀ t))
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  let u := (Homeomorph.neg G).trans (Homeomorph.addRight x₀)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    x₀ : G
    h : HasCompactSupport fun t => (L (f t)) (g (HSub.hSub x₀ t))
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    u : Homeomorph G G := (Homeomorph.neg G).trans (Homeomorph.addRight x₀)
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  let v := (Homeomorph.neg G).trans (Homeomorph.addLeft x₀)
  apply ((u.isCompact_preimage.mpr h).bddAbove_image hg.norm.continuousOn).convolutionExistsAt' L
    isClosed_closure.measurableSet subset_closure (hf.integrableOn_isCompact h)
  have A : AEStronglyMeasurable (g ∘ v)
      (μ.restrict (tsupport fun t : G => L (f t) (g (x₀ - t)))) := by
    apply (hg.comp v.continuous).continuousOn.aestronglyMeasurable_of_isCompact h
    exact (isClosed_tsupport _).measurableSet
  convert ((v.continuous.measurable.measurePreserving
      (μ.restrict (tsupport fun t => L (f t) (g (x₀ - t))))).aestronglyMeasurable_comp_iff
    v.measurableEmbedding).1 A
  /-
    case h.e'_6.h.e'_5.h.h.e
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    x₀ : G
    h : HasCompactSupport fun t => (L (f t)) (g (HSub.hSub x₀ t))
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    u : Homeomorph G G := (Homeomorph.neg G).trans (Homeomorph.addRight x₀)
    v : Homeomorph G G := (Homeomorph.neg G).trans (Homeomorph.addLeft x₀)
    A : MeasureTheory.AEStronglyMeasurable (Function.comp g ⇑v) (μ.restrict (tsupp …
    x✝ : G
    ⊢ Eq (HSub.hSub ↑(toAddUnits x₀)) ⇑v
  -/
  ext x
  simp only [v, Homeomorph.neg, sub_eq_add_neg, val_toAddUnits_apply, Homeomorph.trans_apply,
    Equiv.neg_apply, Equiv.toFun_as_coe, Homeomorph.homeomorph_mk_coe, Equiv.coe_fn_mk,
    Homeomorph.coe_addLeft]


theorem _root_.HasCompactSupport.convolutionExists_right (hcg : HasCompactSupport g)
    (hf : LocallyIntegrable f μ) (hg : Continuous g) : ConvolutionExists f g L μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    ⊢ MeasureTheory.ConvolutionExists f g L μ
  -/
  intro x₀
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  refine HasCompactSupport.convolutionExistsAt L ?_ hf hg
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ HasCompactSupport fun t => (L (f t)) (g (HSub.hSub x₀ t))
  -/
  refine (hcg.comp_homeomorph (Homeomorph.subLeft x₀)).mono ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t)))  …
  -/
  refine fun t => mt fun ht : g (x₀ - t) = 0 => ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ t : G
    ht : Eq (g (HSub.hSub x₀ t)) 0
    ⊢ Eq ((fun t => (L (f t)) (g (HSub.hSub x₀ t))) t) 0
  -/
  simp_rw [ht, (L _).map_zero]
  /-
    🎉 no goals
  -/


theorem _root_.HasCompactSupport.convolutionExists_left_of_continuous_right
    (hcf : HasCompactSupport f) (hf : LocallyIntegrable f μ) (hg : Continuous g) :
    ConvolutionExists f g L μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    ⊢ MeasureTheory.ConvolutionExists f g L μ
  -/
  intro x₀
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  refine HasCompactSupport.convolutionExistsAt L ?_ hf hg
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ HasCompactSupport fun t => (L (f t)) (g (HSub.hSub x₀ t))
  -/
  refine hcf.mono ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ : G
    ⊢ HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t)))  …
  -/
  refine fun t => mt fun ht : f t = 0 => ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    x₀ t : G
    ht : Eq (f t) 0
    ⊢ Eq ((fun t => (L (f t)) (g (HSub.hSub x₀ t))) t) 0
  -/
  simp_rw [ht, L.map_zero₂]
  /-
    🎉 no goals
  -/


/-- A sufficient condition to prove that `f ⋆[L, μ] g` exists.
We assume that the integrand has compact support and `g` is bounded on this support (note that
both properties hold if `g` is continuous with compact support). We also require that `f` is
integrable on the support of the integrand, and that both functions are strongly measurable.

This is a variant of `BddAbove.convolutionExistsAt'` in an abelian group with a left-invariant
measure. This allows us to state the boundedness and measurability of `g` in a more natural way. -/
theorem _root_.BddAbove.convolutionExistsAt [MeasurableAdd₂ G] [SFinite μ] {x₀ : G} {s : Set G}
    (hbg : BddAbove ((fun i => ‖g i‖) '' ((fun t => x₀ - t) ⁻¹' s))) (hs : MeasurableSet s)
    (h2s : (support fun t => L (f t) (g (x₀ - t))) ⊆ s) (hf : IntegrableOn f s μ)
    (hmg : AEStronglyMeasurable g μ) : ConvolutionExistsAt f g x₀ L μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : AddCommGroup G
    inst✝³ : MeasurableNeg G
    inst✝² : μ.IsAddLeftInvariant
    inst✝¹ : MeasurableAdd₂ G
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    s : Set G
    hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
    hs : MeasurableSet s
    h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
    hf : MeasureTheory.IntegrableOn f s μ
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ MeasureTheory.ConvolutionExistsAt f g x₀ L μ
  -/
  refine BddAbove.convolutionExistsAt' L ?_ hs h2s hf ?_
    /-
      case refine_1
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : NormedSpace 𝕜 E'
      inst✝⁶ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁴ : AddCommGroup G
      inst✝³ : MeasurableNeg G
      inst✝² : μ.IsAddLeftInvariant
      inst✝¹ : MeasurableAdd₂ G
      inst✝ : MeasureTheory.SFinite μ
      x₀ : G
      s : Set G
      hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
      hs : MeasurableSet s
      h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
      hf : MeasureTheory.IntegrableOn f s μ
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      ⊢ BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => HAdd. …
    -/
  · simp_rw [← sub_eq_neg_add, hbg]
    /-
      🎉 no goals
    -/
  · have : AEStronglyMeasurable g (map (fun t : G => x₀ - t) μ) :=
      hmg.mono_ac (quasiMeasurePreserving_sub_left_of_right_invariant μ x₀).absolutelyContinuous
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : NormedSpace 𝕜 E'
      inst✝⁶ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁴ : AddCommGroup G
      inst✝³ : MeasurableNeg G
      inst✝² : μ.IsAddLeftInvariant
      inst✝¹ : MeasurableAdd₂ G
      inst✝ : MeasureTheory.SFinite μ
      x₀ : G
      s : Set G
      hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
      hs : MeasurableSet s
      h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
      hf : MeasureTheory.IntegrableOn f s μ
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      this : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t  …
      ⊢ MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t => HS …
    -/
    apply this.mono_measure
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : NormedSpace 𝕜 E'
      inst✝⁶ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁴ : AddCommGroup G
      inst✝³ : MeasurableNeg G
      inst✝² : μ.IsAddLeftInvariant
      inst✝¹ : MeasurableAdd₂ G
      inst✝ : MeasureTheory.SFinite μ
      x₀ : G
      s : Set G
      hbg : BddAbove (Set.image (fun i => Norm.norm (g i)) (Set.preimage (fun t => H …
      hs : MeasurableSet s
      h2s : HasSubset.Subset (Function.support fun t => (L (f t)) (g (HSub.hSub x₀ t …
      hf : MeasureTheory.IntegrableOn f s μ
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      this : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map (fun t  …
      ⊢ LE.le (MeasureTheory.Measure.map (fun t => HSub.hSub x₀ t) (μ.restrict s)) ( …
    -/
    exact map_mono restrict_le_self (measurable_const.sub measurable_id')
    /-
      🎉 no goals
    -/


theorem convolutionExistsAt_flip :
    ConvolutionExistsAt g f x L.flip μ ↔ ConvolutionExistsAt f g x L μ := by
  simp_rw [ConvolutionExistsAt, ← integrable_comp_sub_left (fun t => L (f t) (g (x - t))) x,
    sub_sub_cancel, flip_apply]


theorem ConvolutionExistsAt.integrable_swap (h : ConvolutionExistsAt f g x L μ) :
    Integrable (fun t => L (f (x - t)) (g t)) μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    x : G
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : AddCommGroup G
    inst✝³ : MeasurableNeg G
    inst✝² : μ.IsAddLeftInvariant
    inst✝¹ : MeasurableAdd G
    inst✝ : μ.IsNegInvariant
    h : MeasureTheory.ConvolutionExistsAt f g x L μ
    ⊢ MeasureTheory.Integrable (fun t => (L (f (HSub.hSub x t))) (g t)) μ
  -/
  convert h.comp_sub_left x
  /-
    case h.e'_6.h.h.e'_6.h.e'_1
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    x : G
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : AddCommGroup G
    inst✝³ : MeasurableNeg G
    inst✝² : μ.IsAddLeftInvariant
    inst✝¹ : MeasurableAdd G
    inst✝ : μ.IsNegInvariant
    h : MeasureTheory.ConvolutionExistsAt f g x L μ
    x✝ : G
    ⊢ Eq x✝ (HSub.hSub x (HSub.hSub x x✝))
  -/
  simp_rw [sub_sub_self]
  /-
    🎉 no goals
  -/


theorem convolutionExistsAt_iff_integrable_swap :
    ConvolutionExistsAt f g x L μ ↔ Integrable (fun t => L (f (x - t)) (g t)) μ :=
  convolutionExistsAt_flip.symm


theorem _root_.HasCompactSupport.convolutionExistsLeft
    (hcf : HasCompactSupport f) (hf : Continuous f)
    (hg : LocallyIntegrable g μ) : ConvolutionExists f g L μ := fun x₀ =>
  convolutionExistsAt_flip.mp <| hcf.convolutionExists_right L.flip hg hf x₀


theorem _root_.HasCompactSupport.convolutionExistsRightOfContinuousLeft (hcg : HasCompactSupport g)
    (hf : Continuous f) (hg : LocallyIntegrable g μ) : ConvolutionExists f g L μ := fun x₀ =>
  convolutionExistsAt_flip.mp <| hcg.convolutionExists_left_of_continuous_right L.flip hg hf x₀


/-- The convolution of two functions `f` and `g` with respect to a continuous bilinear map `L` and
measure `μ`. It is defined to be `(f ⋆[L, μ] g) x = ∫ t, L (f t) (g (x - t)) ∂μ`. -/
noncomputable def convolution [Sub G] (f : G → E) (g : G → E') (L : E →L[𝕜] E' →L[𝕜] F)
    (μ : Measure G := by volume_tac) : G → F := fun x =>
  ∫ t, L (f t) (g (x - t)) ∂μ


/-- The convolution of two functions with respect to a bilinear operation `L` and a measure `μ`. -/
scoped[Convolution] notation:67 f " ⋆[" L:67 ", " μ:67 "] " g:66 => convolution f g L μ


/-- The convolution of two functions with respect to a bilinear operation `L` and the volume. -/
scoped[Convolution]
  notation:67 f " ⋆[" L:67 "]" g:66 => convolution f g L MeasureSpace.volume


/-- The convolution of two real-valued functions with respect to volume. -/
scoped[Convolution]
  notation:67 f " ⋆ " g:66 =>
    convolution f g (ContinuousLinearMap.lsmul ℝ ℝ) MeasureSpace.volume


theorem convolution_def [Sub G] : (f ⋆[L, μ] g) x = ∫ t, L (f t) (g (x - t)) ∂μ :=
  rfl


/-- The definition of convolution where the bilinear operator is scalar multiplication.
Note: it often helps the elaborator to give the type of the convolution explicitly. -/
theorem convolution_lsmul [Sub G] {f : G → 𝕜} {g : G → F} :
    (f ⋆[lsmul 𝕜 𝕜, μ] g : G → F) x = ∫ t, f t • g (x - t) ∂μ :=
  rfl


/-- The definition of convolution where the bilinear operator is multiplication. -/
theorem convolution_mul [Sub G] [NormedSpace ℝ 𝕜] {f : G → 𝕜} {g : G → 𝕜} :
    (f ⋆[mul 𝕜 𝕜, μ] g) x = ∫ t, f t * g (x - t) ∂μ :=
  rfl


theorem smul_convolution [SMulCommClass ℝ 𝕜 F] {y : 𝕜} : y • f ⋆[L, μ] g = y • (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : NormedSpace Real F
    inst✝¹ : AddGroup G
    inst✝ : SMulCommClass Real 𝕜 F
    y : 𝕜
    ⊢ Eq (MeasureTheory.convolution (HSMul.hSMul y f) g L μ) (HSMul.hSMul y (Measu …
  -/
  ext; simp only [Pi.smul_apply, convolution_def, ← integral_smul, L.map_smul₂]
       /-
         🎉 no goals
       -/


theorem convolution_smul [SMulCommClass ℝ 𝕜 F] {y : 𝕜} : f ⋆[L, μ] y • g = y • (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : NormedSpace Real F
    inst✝¹ : AddGroup G
    inst✝ : SMulCommClass Real 𝕜 F
    y : 𝕜
    ⊢ Eq (MeasureTheory.convolution f (HSMul.hSMul y g) L μ) (HSMul.hSMul y (Measu …
  -/
  ext; simp only [Pi.smul_apply, convolution_def, ← integral_smul, (L _).map_smul]
       /-
         🎉 no goals
       -/


@[simp]
theorem zero_convolution : 0 ⋆[L, μ] g = 0 := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    ⊢ Eq (MeasureTheory.convolution 0 g L μ) 0
  -/
  ext
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x✝ : G
    ⊢ Eq (MeasureTheory.convolution 0 g L μ x✝) (0 x✝)
  -/
  simp_rw [convolution_def, Pi.zero_apply, L.map_zero₂, integral_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem convolution_zero : f ⋆[L, μ] 0 = 0 := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    ⊢ Eq (MeasureTheory.convolution f 0 L μ) 0
  -/
  ext
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x✝ : G
    ⊢ Eq (MeasureTheory.convolution f 0 L μ x✝) (0 x✝)
  -/
  simp_rw [convolution_def, Pi.zero_apply, (L _).map_zero, integral_zero]
  /-
    🎉 no goals
  -/


theorem ConvolutionExistsAt.distrib_add {x : G} (hfg : ConvolutionExistsAt f g x L μ)
    (hfg' : ConvolutionExistsAt f g' x L μ) :
    (f ⋆[L, μ] (g + g')) x = (f ⋆[L, μ] g) x + (f ⋆[L, μ] g') x := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g g' : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    hfg : MeasureTheory.ConvolutionExistsAt f g x L μ
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x L μ
    ⊢ Eq (MeasureTheory.convolution f (HAdd.hAdd g g') L μ x) (HAdd.hAdd (MeasureT …
  -/
  simp only [convolution_def, (L _).map_add, Pi.add_apply, integral_add hfg hfg']
  /-
    🎉 no goals
  -/


theorem ConvolutionExists.distrib_add (hfg : ConvolutionExists f g L μ)
    (hfg' : ConvolutionExists f g' L μ) : f ⋆[L, μ] (g + g') = f ⋆[L, μ] g + f ⋆[L, μ] g' := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g g' : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    hfg : MeasureTheory.ConvolutionExists f g L μ
    hfg' : MeasureTheory.ConvolutionExists f g' L μ
    ⊢ Eq (MeasureTheory.convolution f (HAdd.hAdd g g') L μ) (HAdd.hAdd (MeasureThe …
  -/
  ext x
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g g' : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    hfg : MeasureTheory.ConvolutionExists f g L μ
    hfg' : MeasureTheory.ConvolutionExists f g' L μ
    x : G
    ⊢ Eq (MeasureTheory.convolution f (HAdd.hAdd g g') L μ x) (HAdd.hAdd (MeasureT …
  -/
  exact (hfg x).distrib_add (hfg' x)
  /-
    🎉 no goals
  -/


theorem ConvolutionExistsAt.add_distrib {x : G} (hfg : ConvolutionExistsAt f g x L μ)
    (hfg' : ConvolutionExistsAt f' g x L μ) :
    ((f + f') ⋆[L, μ] g) x = (f ⋆[L, μ] g) x + (f' ⋆[L, μ] g) x := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f f' : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    hfg : MeasureTheory.ConvolutionExistsAt f g x L μ
    hfg' : MeasureTheory.ConvolutionExistsAt f' g x L μ
    ⊢ Eq (MeasureTheory.convolution (HAdd.hAdd f f') g L μ x) (HAdd.hAdd (MeasureT …
  -/
  simp only [convolution_def, L.map_add₂, Pi.add_apply, integral_add hfg hfg']
  /-
    🎉 no goals
  -/


theorem ConvolutionExists.add_distrib (hfg : ConvolutionExists f g L μ)
    (hfg' : ConvolutionExists f' g L μ) : (f + f') ⋆[L, μ] g = f ⋆[L, μ] g + f' ⋆[L, μ] g := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f f' : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    hfg : MeasureTheory.ConvolutionExists f g L μ
    hfg' : MeasureTheory.ConvolutionExists f' g L μ
    ⊢ Eq (MeasureTheory.convolution (HAdd.hAdd f f') g L μ) (HAdd.hAdd (MeasureThe …
  -/
  ext x
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f f' : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    hfg : MeasureTheory.ConvolutionExists f g L μ
    hfg' : MeasureTheory.ConvolutionExists f' g L μ
    x : G
    ⊢ Eq (MeasureTheory.convolution (HAdd.hAdd f f') g L μ x) (HAdd.hAdd (MeasureT …
  -/
  exact (hfg x).add_distrib (hfg' x)
  /-
    🎉 no goals
  -/


theorem convolution_mono_right {f g g' : G → ℝ} (hfg : ConvolutionExistsAt f g x (lsmul ℝ ℝ) μ)
    (hfg' : ConvolutionExistsAt f g' x (lsmul ℝ ℝ) μ) (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, g x ≤ g' x) :
    (f ⋆[lsmul ℝ ℝ, μ] g) x ≤ (f ⋆[lsmul ℝ ℝ, μ] g') x := by
  /-
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg : MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Real  …
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    ⊢ LE.le (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) μ …
  -/
  apply integral_mono hfg hfg'
  /-
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg : MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Real  …
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    ⊢ LE.le (fun t => ((ContinuousLinearMap.lsmul Real Real) (f t)) (g (HSub.hSub  …
  -/
  simp only [lsmul_apply, Algebra.id.smul_eq_mul]
  /-
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg : MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Real  …
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    ⊢ LE.le (fun t => HMul.hMul (f t) (g (HSub.hSub x t))) fun t => HMul.hMul (f t …
  -/
  intro t
  /-
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg : MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Real  …
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    t : G
    ⊢ LE.le ((fun t => HMul.hMul (f t) (g (HSub.hSub x t))) t) ((fun t => HMul.hMu …
  -/
  apply mul_le_mul_of_nonneg_left (hg _) (hf _)
  /-
    🎉 no goals
  -/


theorem convolution_mono_right_of_nonneg {f g g' : G → ℝ}
    (hfg' : ConvolutionExistsAt f g' x (lsmul ℝ ℝ) μ) (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, g x ≤ g' x)
    (hg' : ∀ x, 0 ≤ g' x) : (f ⋆[lsmul ℝ ℝ, μ] g) x ≤ (f ⋆[lsmul ℝ ℝ, μ] g') x := by
  /-
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    hg' : ∀ (x : G), LE.le 0 (g' x)
    ⊢ LE.le (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) μ …
  -/
  by_cases H : ConvolutionExistsAt f g x (lsmul ℝ ℝ) μ
    /-
      case pos
      G : Type uG
      x : G
      inst✝¹ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝ : AddGroup G
      f g g' : G → Real
      hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
      hf : ∀ (x : G), LE.le 0 (f x)
      hg : ∀ (x : G), LE.le (g x) (g' x)
      hg' : ∀ (x : G), LE.le 0 (g' x)
      H : MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Real Re …
      ⊢ LE.le (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) μ …
    -/
  · exact convolution_mono_right H hfg' hf hg
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    hg' : ∀ (x : G), LE.le 0 (g' x)
    H : Not (MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Re …
    ⊢ LE.le (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) μ …
  -/
  have : (f ⋆[lsmul ℝ ℝ, μ] g) x = 0 := integral_undef H
  /-
    case neg
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    hg' : ∀ (x : G), LE.le 0 (g' x)
    H : Not (MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Re …
    this : Eq (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) …
    ⊢ LE.le (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) μ …
  -/
  rw [this]
  /-
    case neg
    G : Type uG
    x : G
    inst✝¹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : AddGroup G
    f g g' : G → Real
    hfg' : MeasureTheory.ConvolutionExistsAt f g' x (ContinuousLinearMap.lsmul Rea …
    hf : ∀ (x : G), LE.le 0 (f x)
    hg : ∀ (x : G), LE.le (g x) (g' x)
    hg' : ∀ (x : G), LE.le 0 (g' x)
    H : Not (MeasureTheory.ConvolutionExistsAt f g x (ContinuousLinearMap.lsmul Re …
    this : Eq (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul Real Real) …
    ⊢ LE.le 0 (MeasureTheory.convolution f g' (ContinuousLinearMap.lsmul Real Real …
  -/
  exact integral_nonneg fun y => mul_nonneg (hf y) (hg' (x - y))
  /-
    🎉 no goals
  -/


theorem convolution_congr [MeasurableAdd₂ G] [MeasurableNeg G] [SFinite μ]
    [IsAddRightInvariant μ] (h1 : f =ᵐ[μ] f') (h2 : g =ᵐ[μ] g') : f ⋆[L, μ] g = f' ⋆[L, μ] g' := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f f' : G → E
    g g' : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : MeasurableAdd₂ G
    inst✝² : MeasurableNeg G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddRightInvariant
    h1 : (MeasureTheory.ae μ).EventuallyEq f f'
    h2 : (MeasureTheory.ae μ).EventuallyEq g g'
    ⊢ Eq (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f' g' L μ)
  -/
  ext x
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f f' : G → E
    g g' : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : MeasurableAdd₂ G
    inst✝² : MeasurableNeg G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddRightInvariant
    h1 : (MeasureTheory.ae μ).EventuallyEq f f'
    h2 : (MeasureTheory.ae μ).EventuallyEq g g'
    x : G
    ⊢ Eq (MeasureTheory.convolution f g L μ x) (MeasureTheory.convolution f' g' L  …
  -/
  apply integral_congr_ae
  exact (h1.prod_mk <| h2.comp_tendsto
    (quasiMeasurePreserving_sub_left_of_right_invariant μ x).tendsto_ae).fun_comp ↿fun x y ↦ L x y


theorem support_convolution_subset_swap : support (f ⋆[L, μ] g) ⊆ support g + support f := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    ⊢ HasSubset.Subset (Function.support (MeasureTheory.convolution f g L μ)) (HAd …
  -/
  intro x h2x
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    ⊢ Membership.mem (HAdd.hAdd (Function.support g) (Function.support f)) x
  -/
  by_contra hx
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : Not (Membership.mem (HAdd.hAdd (Function.support g) (Function.support f)) …
    ⊢ False
  -/
  apply h2x
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : Not (Membership.mem (HAdd.hAdd (Function.support g) (Function.support f)) …
    ⊢ Eq (MeasureTheory.convolution f g L μ x) 0
  -/
  simp_rw [Set.mem_add, ← exists_and_left, not_exists, not_and_or, nmem_support] at hx
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
    ⊢ Eq (MeasureTheory.convolution f g L μ x) 0
  -/
  rw [convolution_def]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
    ⊢ Eq (MeasureTheory.integral μ fun t => (L (f t)) (g (HSub.hSub x t))) 0
  -/
  convert integral_zero G F using 2
  /-
    case h.e'_2.h.e'_7
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
    ⊢ Eq (fun t => (L (f t)) (g (HSub.hSub x t))) fun x => 0
  -/
  ext t
  /-
    case h.e'_2.h.e'_7.h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : AddGroup G
    x : G
    h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
    hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
    t : G
    ⊢ Eq ((L (f t)) (g (HSub.hSub x t))) 0
  -/
  rcases hx (x - t) t with (h | h | h)
    /-
      case h.e'_2.h.e'_7.h.inl
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜 E'
      inst✝³ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝² : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝¹ : NormedSpace Real F
      inst✝ : AddGroup G
      x : G
      h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
      hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
      t : G
      h : Eq (g (HSub.hSub x t)) 0
      ⊢ Eq ((L (f t)) (g (HSub.hSub x t))) 0
    -/
  · rw [h, (L _).map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_7.h.inr.inl
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜 E'
      inst✝³ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝² : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝¹ : NormedSpace Real F
      inst✝ : AddGroup G
      x : G
      h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
      hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
      t : G
      h : Eq (f t) 0
      ⊢ Eq ((L (f t)) (g (HSub.hSub x t))) 0
    -/
  · rw [h, L.map_zero₂]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_7.h.inr.inr
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜 E'
      inst✝³ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝² : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝¹ : NormedSpace Real F
      inst✝ : AddGroup G
      x : G
      h2x : Membership.mem (Function.support (MeasureTheory.convolution f g L μ)) x
      hx : ∀ (x_1 x_2 : G), Or (Eq (g x_1) 0) (Or (Eq (f x_2) 0) (Not (Eq (HAdd.hAdd …
      t : G
      h : Not (Eq (HAdd.hAdd (HSub.hSub x t) t) x)
      ⊢ Eq ((L (f t)) (g (HSub.hSub x t))) 0
    -/
  · exact (h <| sub_add_cancel x t).elim
    /-
      🎉 no goals
    -/


theorem Integrable.integrable_convolution (hf : Integrable f μ)
    (hg : Integrable g μ) : Integrable (f ⋆[L, μ] g) μ :=
  (hf.convolution_integrand L hg).integral_prod_left


protected theorem _root_.HasCompactSupport.convolution [T2Space G] (hcf : HasCompactSupport f)
    (hcg : HasCompactSupport g) : HasCompactSupport (f ⋆[L, μ] g) :=
  (hcg.isCompact.add hcf).of_isClosed_subset isClosed_closure <|
    closure_minimal
      ((support_convolution_subset_swap L).trans <| add_subset_add subset_closure subset_closure)
      (hcg.isCompact.add hcf).isClosed


/-- The convolution `f * g` is continuous if `f` is locally integrable and `g` is continuous and
compactly supported. Version where `g` depends on an additional parameter in a subset `s` of
a parameter space `P` (and the compact support `k` is independent of the parameter in `s`). -/
theorem continuousOn_convolution_right_with_param {g : P → G → E'} {s : Set P} {k : Set G}
    (hk : IsCompact k) (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0)
    (hf : LocallyIntegrable f μ) (hg : ContinuousOn (↿g) (s ×ˢ univ)) :
    ContinuousOn (fun q : P × G => (f ⋆[L, μ] g q.1) q.2) (s ×ˢ univ) := by
  /- First get rid of the case where the space is not locally compact. Then `g` vanishes everywhere
  and the conclusion is trivial. -/
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ ContinuousOn (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SProd.s …
  -/
  by_cases H : ∀ p ∈ s, ∀ x, g p x = 0
    /-
      case pos
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      P : Type uP
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁵ : NormedSpace Real F
      inst✝⁴ : AddGroup G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalAddGroup G
      inst✝¹ : BorelSpace G
      inst✝ : TopologicalSpace P
      g : P → G → E'
      s : Set P
      k : Set G
      hk : IsCompact k
      hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
      H : ∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0
      ⊢ ContinuousOn (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SProd.s …
    -/
  · apply (continuousOn_const (c := 0)).congr
    /-
      case pos
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      P : Type uP
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁵ : NormedSpace Real F
      inst✝⁴ : AddGroup G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalAddGroup G
      inst✝¹ : BorelSpace G
      inst✝ : TopologicalSpace P
      g : P → G → E'
      s : Set P
      k : Set G
      hk : IsCompact k
      hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
      H : ∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0
      ⊢ Set.EqOn (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (fun x => 0) …
    -/
    rintro ⟨p, x⟩ ⟨hp, -⟩
    /-
      case pos.mk.intro
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      P : Type uP
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁵ : NormedSpace Real F
      inst✝⁴ : AddGroup G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalAddGroup G
      inst✝¹ : BorelSpace G
      inst✝ : TopologicalSpace P
      g : P → G → E'
      s : Set P
      k : Set G
      hk : IsCompact k
      hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
      H : ∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0
      p : P
      x : G
      hp : Membership.mem s { fst := p, snd := x }.1
      ⊢ Eq ((fun q => MeasureTheory.convolution f (g q.1) L μ q.2) { fst := p, snd : …
    -/
    apply integral_eq_zero_of_ae (Eventually.of_forall (fun y ↦ ?_))
    /-
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      P : Type uP
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁶ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁵ : NormedSpace Real F
      inst✝⁴ : AddGroup G
      inst✝³ : TopologicalSpace G
      inst✝² : TopologicalAddGroup G
      inst✝¹ : BorelSpace G
      inst✝ : TopologicalSpace P
      g : P → G → E'
      s : Set P
      k : Set G
      hk : IsCompact k
      hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
      H : ∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0
      p : P
      x : G
      hp : Membership.mem s { fst := p, snd := x }.1
      y : G
      ⊢ Eq ((L (f y)) (g { fst := p, snd := x }.1 (HSub.hSub { fst := p, snd := x }. …
    -/
    simp [H p hp _]
    /-
      🎉 no goals
    -/
  have : LocallyCompactSpace G := by
    push_neg at H
    rcases H with ⟨p, hp, x, hx⟩
    have A : support (g p) ⊆ k := support_subset_iff'.2 (fun y hy ↦ hgs p y hp hy)
    have B : Continuous (g p) := by
      refine hg.comp_continuous (continuous_const.prod_mk continuous_id') fun x => ?_
      simpa only [prod_mk_mem_set_prod_eq, mem_univ, and_true] using hp
    rcases eq_zero_or_locallyCompactSpace_of_support_subset_isCompact_of_addGroup hk A B with H|H
    · simp [H] at hx
    · exact H
  /- Since `G` is locally compact, one may thicken `k` a little bit into a larger compact set
  `(-k) + t`, outside of which all functions that appear in the convolution vanish. Then we can
  apply a continuity statement for integrals depending on a parameter, with respect to
  locally integrable functions and compactly supported continuous functions. -/
  /-
    case neg
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    ⊢ ContinuousOn (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SProd.s …
  -/
  rintro ⟨q₀, x₀⟩ ⟨hq₀, -⟩
  /-
    case neg.mk.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  obtain ⟨t, t_comp, ht⟩ : ∃ t, IsCompact t ∧ t ∈ 𝓝 x₀ := exists_compact_mem_nhds x₀
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  let k' : Set G := (-k) +ᵥ t
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HVAdd.hVAdd (Neg.neg k) t
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  have k'_comp : IsCompact k' := IsCompact.vadd_set hk.neg t_comp
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HVAdd.hVAdd (Neg.neg k) t
    k'_comp : IsCompact k'
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  let g' : (P × G) → G → E' := fun p x ↦ g p.1 (p.2 - x)
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HVAdd.hVAdd (Neg.neg k) t
    k'_comp : IsCompact k'
    g' : Prod P G → G → E' := fun p x => g p.1 (HSub.hSub p.2 x)
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  let s' : Set (P × G) := s ×ˢ t
  have A : ContinuousOn g'.uncurry (s' ×ˢ univ) := by
    have : g'.uncurry = g.uncurry ∘ (fun w ↦ (w.1.1, w.1.2 - w.2)) := by ext y; rfl
    rw [this]
    refine hg.comp (continuous_fst.fst.prod_mk (continuous_fst.snd.sub
      continuous_snd)).continuousOn ?_
    simp +contextual [s', MapsTo]
  have B : ContinuousOn (fun a ↦ ∫ x, L (f x) (g' a x) ∂μ) s' := by
    apply continuousOn_integral_bilinear_of_locally_integrable_of_compact_support L k'_comp A _
      (hf.integrableOn_isCompact k'_comp)
    rintro ⟨p, x⟩ y ⟨hp, hx⟩ hy
    apply hgs p _ hp
    contrapose! hy
    exact ⟨y - x, by simpa using hy, x, hx, by simp⟩
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HVAdd.hVAdd (Neg.neg k) t
    k'_comp : IsCompact k'
    g' : Prod P G → G → E' := fun p x => g p.1 (HSub.hSub p.2 x)
    s' : Set (Prod P G) := SProd.sprod s t
    A : ContinuousOn (Function.uncurry g') (SProd.sprod s' Set.univ)
    B : ContinuousOn (fun a => MeasureTheory.integral μ fun x => (L (f x)) (g' a x …
    ⊢ ContinuousWithinAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (S …
  -/
  apply ContinuousWithinAt.mono_of_mem_nhdsWithin (B (q₀, x₀) ⟨hq₀, mem_of_mem_nhds ht⟩)
  /-
    case neg.mk.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    g : P → G → E'
    s : Set P
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    H : Not (∀ (p : P), Membership.mem s p → ∀ (x : G), Eq (g p x) 0)
    this : LocallyCompactSpace G
    q₀ : P
    x₀ : G
    hq₀ : Membership.mem s { fst := q₀, snd := x₀ }.1
    t : Set G
    t_comp : IsCompact t
    ht : Membership.mem (nhds x₀) t
    k' : Set G := HVAdd.hVAdd (Neg.neg k) t
    k'_comp : IsCompact k'
    g' : Prod P G → G → E' := fun p x => g p.1 (HSub.hSub p.2 x)
    s' : Set (Prod P G) := SProd.sprod s t
    A : ContinuousOn (Function.uncurry g') (SProd.sprod s' Set.univ)
    B : ContinuousOn (fun a => MeasureTheory.integral μ fun x => (L (f x)) (g' a x …
    ⊢ Membership.mem (nhdsWithin { fst := q₀, snd := x₀ } (SProd.sprod s Set.univ) …
  -/
  exact mem_nhdsWithin_prod_iff.2 ⟨s, self_mem_nhdsWithin, t, nhdsWithin_le_nhds ht, Subset.rfl⟩
  /-
    🎉 no goals
  -/


/-- The convolution `f * g` is continuous if `f` is locally integrable and `g` is continuous and
compactly supported. Version where `g` depends on an additional parameter in an open subset `s` of
a parameter space `P` (and the compact support `k` is independent of the parameter in `s`),
given in terms of compositions with an additional continuous map. -/
theorem continuousOn_convolution_right_with_param_comp {s : Set P} {v : P → G}
    (hv : ContinuousOn v s) {g : P → G → E'} {k : Set G} (hk : IsCompact k)
    (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0) (hf : LocallyIntegrable f μ)
    (hg : ContinuousOn (↿g) (s ×ˢ univ)) : ContinuousOn (fun x => (f ⋆[L, μ] g x) (v x)) s := by
  apply
    (continuousOn_convolution_right_with_param L hk hgs hf hg).comp (continuousOn_id.prod hv)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    s : Set P
    v : P → G
    hv : ContinuousOn v s
    g : P → G → E'
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ Set.MapsTo (fun x => { fst := id x, snd := v x }) s (SProd.sprod s Set.univ)
  -/
  intro x hx
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddGroup G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalAddGroup G
    inst✝¹ : BorelSpace G
    inst✝ : TopologicalSpace P
    s : Set P
    v : P → G
    hv : ContinuousOn v s
    g : P → G → E'
    k : Set G
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContinuousOn (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    x : P
    hx : Membership.mem s x
    ⊢ Membership.mem (SProd.sprod s Set.univ) ((fun x => { fst := id x, snd := v x …
  -/
  simp only [hx, prod_mk_mem_set_prod_eq, mem_univ, and_self_iff, _root_.id]
  /-
    🎉 no goals
  -/


/-- The convolution is continuous if one function is locally integrable and the other has compact
support and is continuous. -/
theorem _root_.HasCompactSupport.continuous_convolution_right (hcg : HasCompactSupport g)
    (hf : LocallyIntegrable f μ) (hg : Continuous g) : Continuous (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : NormedSpace Real F
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    ⊢ Continuous (MeasureTheory.convolution f g L μ)
  -/
  rw [continuous_iff_continuousOn_univ]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : NormedSpace Real F
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    ⊢ ContinuousOn (MeasureTheory.convolution f g L μ) Set.univ
  -/
  let g' : G → G → E' := fun _ q => g q
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedSpace 𝕜 E'
    inst✝⁶ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : NormedSpace Real F
    inst✝³ : AddGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : Continuous g
    g' : G → G → E' := fun x q => g q
    ⊢ ContinuousOn (MeasureTheory.convolution f g L μ) Set.univ
  -/
  have : ContinuousOn (↿g') (univ ×ˢ univ) := (hg.comp continuous_snd).continuousOn
  exact continuousOn_convolution_right_with_param_comp L
    (continuous_iff_continuousOn_univ.1 continuous_id) hcg
    (fun p x _ hx => image_eq_zero_of_nmem_tsupport hx) hf this


/-- The convolution is continuous if one function is integrable and the other is bounded and
continuous. -/
theorem _root_.BddAbove.continuous_convolution_right_of_integrable
    [FirstCountableTopology G] [SecondCountableTopologyEither G E']
    (hbg : BddAbove (range fun x => ‖g x‖)) (hf : Integrable f μ) (hg : Continuous g) :
    Continuous (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : NormedSpace Real F
    inst✝⁵ : AddGroup G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalAddGroup G
    inst✝² : BorelSpace G
    inst✝¹ : FirstCountableTopology G
    inst✝ : SecondCountableTopologyEither G E'
    hbg : BddAbove (Set.range fun x => Norm.norm (g x))
    hf : MeasureTheory.Integrable f μ
    hg : Continuous g
    ⊢ Continuous (MeasureTheory.convolution f g L μ)
  -/
  refine continuous_iff_continuousAt.mpr fun x₀ => ?_
  have : ∀ᶠ x in 𝓝 x₀, ∀ᵐ t : G ∂μ, ‖L (f t) (g (x - t))‖ ≤ ‖L‖ * ‖f t‖ * ⨆ i, ‖g i‖ := by
    filter_upwards with x; filter_upwards with t
    apply_rules [L.le_of_opNorm₂_le_of_le, le_rfl, le_ciSup hbg (x - t)]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : NormedSpace Real F
    inst✝⁵ : AddGroup G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalAddGroup G
    inst✝² : BorelSpace G
    inst✝¹ : FirstCountableTopology G
    inst✝ : SecondCountableTopologyEither G E'
    hbg : BddAbove (Set.range fun x => Norm.norm (g x))
    hf : MeasureTheory.Integrable f μ
    hg : Continuous g
    x₀ : G
    this : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm.nor …
    ⊢ ContinuousAt (MeasureTheory.convolution f g L μ) x₀
  -/
  refine continuousAt_of_dominated ?_ this ?_ ?_
  · exact Eventually.of_forall fun x =>
      hf.aestronglyMeasurable.convolution_integrand_snd' L hg.aestronglyMeasurable
    /-
      case refine_2
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedAddCommGroup E'
      inst✝¹² : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹¹ : NontriviallyNormedField 𝕜
      inst✝¹⁰ : NormedSpace 𝕜 E
      inst✝⁹ : NormedSpace 𝕜 E'
      inst✝⁸ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁷ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁶ : NormedSpace Real F
      inst✝⁵ : AddGroup G
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup G
      inst✝² : BorelSpace G
      inst✝¹ : FirstCountableTopology G
      inst✝ : SecondCountableTopologyEither G E'
      hbg : BddAbove (Set.range fun x => Norm.norm (g x))
      hf : MeasureTheory.Integrable f μ
      hg : Continuous g
      x₀ : G
      this : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm.nor …
      ⊢ MeasureTheory.Integrable (fun a => HMul.hMul (HMul.hMul (Norm.norm L) (Norm. …
    -/
  · exact (hf.norm.const_mul _).mul_const _
    /-
      🎉 no goals
    -/
  · exact Eventually.of_forall fun t => (L.continuous₂.comp₂ continuous_const <|
      hg.comp <| continuous_id.sub continuous_const).continuousAt


theorem support_convolution_subset : support (f ⋆[L, μ] g) ⊆ support f + support g :=
  (support_convolution_subset_swap L).trans (add_comm _ _).subset


/-- Commutativity of convolution -/
theorem convolution_flip : g ⋆[L.flip, μ] f = f ⋆[L, μ] g := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddCommGroup G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : μ.IsNegInvariant
    inst✝¹ : MeasurableNeg G
    inst✝ : MeasurableAdd G
    ⊢ Eq (MeasureTheory.convolution g f L.flip μ) (MeasureTheory.convolution f g L …
  -/
  ext1 x
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddCommGroup G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : μ.IsNegInvariant
    inst✝¹ : MeasurableNeg G
    inst✝ : MeasurableAdd G
    x : G
    ⊢ Eq (MeasureTheory.convolution g f L.flip μ x) (MeasureTheory.convolution f g …
  -/
  simp_rw [convolution_def]
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddCommGroup G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : μ.IsNegInvariant
    inst✝¹ : MeasurableNeg G
    inst✝ : MeasurableAdd G
    x : G
    ⊢ Eq (MeasureTheory.integral μ fun t => (L.flip (g t)) (f (HSub.hSub x t))) (M …
  -/
  rw [← integral_sub_left_eq_self _ μ x]
  /-
    case h
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddCommGroup G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : μ.IsNegInvariant
    inst✝¹ : MeasurableNeg G
    inst✝ : MeasurableAdd G
    x : G
    ⊢ Eq (MeasureTheory.integral μ fun x_1 => (L.flip (g (HSub.hSub x x_1))) (f (H …
  -/
  simp_rw [sub_sub_self, flip_apply]
  /-
    🎉 no goals
  -/


/-- The symmetric definition of convolution. -/
theorem convolution_eq_swap : (f ⋆[L, μ] g) x = ∫ t, L (f (x - t)) (g t) ∂μ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    x : G
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : AddCommGroup G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : μ.IsNegInvariant
    inst✝¹ : MeasurableNeg G
    inst✝ : MeasurableAdd G
    ⊢ Eq (MeasureTheory.convolution f g L μ x) (MeasureTheory.integral μ fun t =>  …
  -/
  rw [← convolution_flip]; rfl
                           /-
                             🎉 no goals
                           -/


/-- The symmetric definition of convolution where the bilinear operator is scalar multiplication. -/
theorem convolution_lsmul_swap {f : G → 𝕜} {g : G → F} :
    (f ⋆[lsmul 𝕜 𝕜, μ] g : G → F) x = ∫ t, f (x - t) • g t ∂μ :=
  convolution_eq_swap _


/-- The symmetric definition of convolution where the bilinear operator is multiplication. -/
theorem convolution_mul_swap [NormedSpace ℝ 𝕜] {f : G → 𝕜} {g : G → 𝕜} :
    (f ⋆[mul 𝕜 𝕜, μ] g) x = ∫ t, f (x - t) * g t ∂μ :=
  convolution_eq_swap _


/-- The convolution of two even functions is also even. -/
theorem convolution_neg_of_neg_eq (h1 : ∀ᵐ x ∂μ, f (-x) = f x) (h2 : ∀ᵐ x ∂μ, g (-x) = g x) :
    (f ⋆[L, μ] g) (-x) = (f ⋆[L, μ] g) x :=
  calc
    ∫ t : G, (L (f t)) (g (-x - t)) ∂μ = ∫ t : G, (L (f (-t))) (g (x + t)) ∂μ := by
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedAddCommGroup E'
        inst✝¹¹ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        x : G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NormedSpace 𝕜 E
        inst✝⁸ : NormedSpace 𝕜 E'
        inst✝⁷ : NormedSpace 𝕜 F
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝⁶ : MeasurableSpace G
        μ : MeasureTheory.Measure G
        inst✝⁵ : NormedSpace Real F
        inst✝⁴ : AddCommGroup G
        inst✝³ : μ.IsAddLeftInvariant
        inst✝² : μ.IsNegInvariant
        inst✝¹ : MeasurableNeg G
        inst✝ : MeasurableAdd G
        h1 : Filter.Eventually (fun x => Eq (f (Neg.neg x)) (f x)) (MeasureTheory.ae μ)
        h2 : Filter.Eventually (fun x => Eq (g (Neg.neg x)) (g x)) (MeasureTheory.ae μ)
        ⊢ Eq (MeasureTheory.integral μ fun t => (L (f t)) (g (HSub.hSub (Neg.neg x) t) …
      -/
      apply integral_congr_ae
      /-
        case h
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedAddCommGroup E'
        inst✝¹¹ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        x : G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NormedSpace 𝕜 E
        inst✝⁸ : NormedSpace 𝕜 E'
        inst✝⁷ : NormedSpace 𝕜 F
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝⁶ : MeasurableSpace G
        μ : MeasureTheory.Measure G
        inst✝⁵ : NormedSpace Real F
        inst✝⁴ : AddCommGroup G
        inst✝³ : μ.IsAddLeftInvariant
        inst✝² : μ.IsNegInvariant
        inst✝¹ : MeasurableNeg G
        inst✝ : MeasurableAdd G
        h1 : Filter.Eventually (fun x => Eq (f (Neg.neg x)) (f x)) (MeasureTheory.ae μ)
        h2 : Filter.Eventually (fun x => Eq (g (Neg.neg x)) (g x)) (MeasureTheory.ae μ)
        ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (L (f a)) (g (HSub.hSub (Neg.neg …
      -/
      filter_upwards [h1, (eventually_add_left_iff μ x).2 h2] with t ht h't
      /-
        case h
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedAddCommGroup E'
        inst✝¹¹ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        x : G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NormedSpace 𝕜 E
        inst✝⁸ : NormedSpace 𝕜 E'
        inst✝⁷ : NormedSpace 𝕜 F
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝⁶ : MeasurableSpace G
        μ : MeasureTheory.Measure G
        inst✝⁵ : NormedSpace Real F
        inst✝⁴ : AddCommGroup G
        inst✝³ : μ.IsAddLeftInvariant
        inst✝² : μ.IsNegInvariant
        inst✝¹ : MeasurableNeg G
        inst✝ : MeasurableAdd G
        h1 : Filter.Eventually (fun x => Eq (f (Neg.neg x)) (f x)) (MeasureTheory.ae μ)
        h2 : Filter.Eventually (fun x => Eq (g (Neg.neg x)) (g x)) (MeasureTheory.ae μ)
        t : G
        ht : Eq (f (Neg.neg t)) (f t)
        h't : Eq (g (Neg.neg (HAdd.hAdd x t))) (g (HAdd.hAdd x t))
        ⊢ Eq ((L (f t)) (g (HSub.hSub (Neg.neg x) t))) ((L (f (Neg.neg t))) (g (HAdd.h …
      -/
      simp_rw [ht, ← h't, neg_add']
      /-
        🎉 no goals
      -/
    _ = ∫ t : G, (L (f t)) (g (x - t)) ∂μ := by
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedAddCommGroup E'
        inst✝¹¹ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        x : G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NormedSpace 𝕜 E
        inst✝⁸ : NormedSpace 𝕜 E'
        inst✝⁷ : NormedSpace 𝕜 F
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝⁶ : MeasurableSpace G
        μ : MeasureTheory.Measure G
        inst✝⁵ : NormedSpace Real F
        inst✝⁴ : AddCommGroup G
        inst✝³ : μ.IsAddLeftInvariant
        inst✝² : μ.IsNegInvariant
        inst✝¹ : MeasurableNeg G
        inst✝ : MeasurableAdd G
        h1 : Filter.Eventually (fun x => Eq (f (Neg.neg x)) (f x)) (MeasureTheory.ae μ)
        h2 : Filter.Eventually (fun x => Eq (g (Neg.neg x)) (g x)) (MeasureTheory.ae μ)
        ⊢ Eq (MeasureTheory.integral μ fun t => (L (f (Neg.neg t))) (g (HAdd.hAdd x t) …
      -/
      rw [← integral_neg_eq_self]
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedAddCommGroup E'
        inst✝¹¹ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        x : G
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : NormedSpace 𝕜 E
        inst✝⁸ : NormedSpace 𝕜 E'
        inst✝⁷ : NormedSpace 𝕜 F
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝⁶ : MeasurableSpace G
        μ : MeasureTheory.Measure G
        inst✝⁵ : NormedSpace Real F
        inst✝⁴ : AddCommGroup G
        inst✝³ : μ.IsAddLeftInvariant
        inst✝² : μ.IsNegInvariant
        inst✝¹ : MeasurableNeg G
        inst✝ : MeasurableAdd G
        h1 : Filter.Eventually (fun x => Eq (f (Neg.neg x)) (f x)) (MeasureTheory.ae μ)
        h2 : Filter.Eventually (fun x => Eq (g (Neg.neg x)) (g x)) (MeasureTheory.ae μ)
        ⊢ Eq (MeasureTheory.integral μ fun x_1 => (L (f (Neg.neg (Neg.neg x_1)))) (g ( …
      -/
      simp only [neg_neg, ← sub_eq_add_neg]
      /-
        🎉 no goals
      -/


theorem _root_.HasCompactSupport.continuous_convolution_left
    (hcf : HasCompactSupport f) (hf : Continuous f) (hg : LocallyIntegrable g μ) :
    Continuous (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : NormedSpace Real F
    inst✝⁵ : AddCommGroup G
    inst✝⁴ : μ.IsAddLeftInvariant
    inst✝³ : μ.IsNegInvariant
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : Continuous f
    hg : MeasureTheory.LocallyIntegrable g μ
    ⊢ Continuous (MeasureTheory.convolution f g L μ)
  -/
  rw [← convolution_flip]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : NormedSpace Real F
    inst✝⁵ : AddCommGroup G
    inst✝⁴ : μ.IsAddLeftInvariant
    inst✝³ : μ.IsNegInvariant
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalAddGroup G
    inst✝ : BorelSpace G
    hcf : HasCompactSupport f
    hf : Continuous f
    hg : MeasureTheory.LocallyIntegrable g μ
    ⊢ Continuous (MeasureTheory.convolution g f L.flip μ)
  -/
  exact hcf.continuous_convolution_right L.flip hg hf
  /-
    🎉 no goals
  -/


theorem _root_.BddAbove.continuous_convolution_left_of_integrable
    [FirstCountableTopology G] [SecondCountableTopologyEither G E]
    (hbf : BddAbove (range fun x => ‖f x‖)) (hf : Continuous f) (hg : Integrable g μ) :
    Continuous (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁶ : NormedAddCommGroup E
    inst✝¹⁵ : NormedAddCommGroup E'
    inst✝¹⁴ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : NormedSpace 𝕜 E
    inst✝¹¹ : NormedSpace 𝕜 E'
    inst✝¹⁰ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : AddCommGroup G
    inst✝⁶ : μ.IsAddLeftInvariant
    inst✝⁵ : μ.IsNegInvariant
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalAddGroup G
    inst✝² : BorelSpace G
    inst✝¹ : FirstCountableTopology G
    inst✝ : SecondCountableTopologyEither G E
    hbf : BddAbove (Set.range fun x => Norm.norm (f x))
    hf : Continuous f
    hg : MeasureTheory.Integrable g μ
    ⊢ Continuous (MeasureTheory.convolution f g L μ)
  -/
  rw [← convolution_flip]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁶ : NormedAddCommGroup E
    inst✝¹⁵ : NormedAddCommGroup E'
    inst✝¹⁴ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : NormedSpace 𝕜 E
    inst✝¹¹ : NormedSpace 𝕜 E'
    inst✝¹⁰ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : AddCommGroup G
    inst✝⁶ : μ.IsAddLeftInvariant
    inst✝⁵ : μ.IsNegInvariant
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalAddGroup G
    inst✝² : BorelSpace G
    inst✝¹ : FirstCountableTopology G
    inst✝ : SecondCountableTopologyEither G E
    hbf : BddAbove (Set.range fun x => Norm.norm (f x))
    hf : Continuous f
    hg : MeasureTheory.Integrable g μ
    ⊢ Continuous (MeasureTheory.convolution g f L.flip μ)
  -/
  exact hbf.continuous_convolution_right_of_integrable L.flip hg hf
  /-
    🎉 no goals
  -/


/-- Compute `(f ⋆ g) x₀` if the support of the `f` is within `Metric.ball 0 R`, and `g` is constant
on `Metric.ball x₀ R`.

We can simplify the RHS further if we assume `f` is integrable, but also if `L = (•)` or more
generally if `L` has an `AntilipschitzWith`-condition. -/
theorem convolution_eq_right' {x₀ : G} {R : ℝ} (hf : support f ⊆ ball (0 : G) R)
    (hg : ∀ x ∈ ball x₀ R, g x = g x₀) : (f ⋆[L, μ] g) x₀ = ∫ t, L (f t) (g x₀) ∂μ := by
  have h2 : ∀ t, L (f t) (g (x₀ - t)) = L (f t) (g x₀) := fun t ↦ by
    by_cases ht : t ∈ support f
    · have h2t := hf ht
      rw [mem_ball_zero_iff] at h2t
      specialize hg (x₀ - t)
      rw [sub_eq_add_neg, add_mem_ball_iff_norm, norm_neg, ← sub_eq_add_neg] at hg
      rw [hg h2t]
    · rw [nmem_support] at ht
      simp_rw [ht, L.map_zero₂]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝¹ : NormedSpace Real F
    inst✝ : SeminormedAddCommGroup G
    x₀ : G
    R : Real
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → Eq (g x) (g x₀)
    h2 : ∀ (t : G), Eq ((L (f t)) (g (HSub.hSub x₀ t))) ((L (f t)) (g x₀))
    ⊢ Eq (MeasureTheory.convolution f g L μ x₀) (MeasureTheory.integral μ fun t => …
  -/
  simp_rw [convolution_def, h2]
  /-
    🎉 no goals
  -/


/-- Approximate `(f ⋆ g) x₀` if the support of the `f` is bounded within a ball, and `g` is near
`g x₀` on a ball with the same radius around `x₀`. See `dist_convolution_le` for a special case.

We can simplify the second argument of `dist` further if we add some extra type-classes on `E`
and `𝕜` or if `L` is scalar multiplication. -/
theorem dist_convolution_le' {x₀ : G} {R ε : ℝ} {z₀ : E'} (hε : 0 ≤ ε) (hif : Integrable f μ)
    (hf : support f ⊆ ball (0 : G) R) (hmg : AEStronglyMeasurable g μ)
    (hg : ∀ x ∈ ball x₀ R, dist (g x) z₀ ≤ ε) :
    dist ((f ⋆[L, μ] g : G → F) x₀) (∫ t, L (f t) z₀ ∂μ) ≤ (‖L‖ * ∫ x, ‖f x‖ ∂μ) * ε := by
  have hfg : ConvolutionExistsAt f g x₀ L μ := by
    refine BddAbove.convolutionExistsAt L ?_ Metric.isOpen_ball.measurableSet (Subset.trans ?_ hf)
      hif.integrableOn hmg
    swap; · refine fun t => mt fun ht : f t = 0 => ?_; simp_rw [ht, L.map_zero₂]
    rw [bddAbove_def]
    refine ⟨‖z₀‖ + ε, ?_⟩
    rintro _ ⟨x, hx, rfl⟩
    refine norm_le_norm_add_const_of_dist_le (hg x ?_)
    rwa [mem_ball_iff_norm, norm_sub_rev, ← mem_ball_zero_iff]
  have h2 : ∀ t, dist (L (f t) (g (x₀ - t))) (L (f t) z₀) ≤ ‖L (f t)‖ * ε := by
    intro t; by_cases ht : t ∈ support f
    · have h2t := hf ht
      rw [mem_ball_zero_iff] at h2t
      specialize hg (x₀ - t)
      rw [sub_eq_add_neg, add_mem_ball_iff_norm, norm_neg, ← sub_eq_add_neg] at hg
      refine ((L (f t)).dist_le_opNorm _ _).trans ?_
      exact mul_le_mul_of_nonneg_left (hg h2t) (norm_nonneg _)
    · rw [nmem_support] at ht
      simp_rw [ht, L.map_zero₂, L.map_zero, norm_zero, zero_mul, dist_self]
      rfl
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Dist.dist ((L (f t)) (g (HSub.hSub x₀ t))) ((L (f t)) z …
    ⊢ LE.le (Dist.dist (MeasureTheory.convolution f g L μ x₀) (MeasureTheory.integ …
  -/
  simp_rw [convolution_def]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Dist.dist ((L (f t)) (g (HSub.hSub x₀ t))) ((L (f t)) z …
    ⊢ LE.le (Dist.dist (MeasureTheory.integral μ fun t => (L (f t)) (g (HSub.hSub  …
  -/
  simp_rw [dist_eq_norm] at h2 ⊢
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Norm.norm (HSub.hSub ((L (f t)) (g (HSub.hSub x₀ t))) ( …
    ⊢ LE.le (Norm.norm (HSub.hSub (MeasureTheory.integral μ fun t => (L (f t)) (g  …
  -/
  rw [← integral_sub hfg.integrable]; swap; · exact (L.flip z₀).integrable_comp hif
                                              /-
                                                🎉 no goals
                                              -/
  refine (norm_integral_le_of_norm_le ((L.integrable_comp hif).norm.mul_const ε)
    (Eventually.of_forall h2)).trans ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Norm.norm (HSub.hSub ((L (f t)) (g (HSub.hSub x₀ t))) ( …
    ⊢ LE.le (MeasureTheory.integral μ fun x => HMul.hMul (Norm.norm (L (f x))) ε)  …
  -/
  rw [integral_mul_right]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Norm.norm (HSub.hSub ((L (f t)) (g (HSub.hSub x₀ t))) ( …
    ⊢ LE.le (HMul.hMul (MeasureTheory.integral μ fun a => Norm.norm (L (f a))) ε)  …
  -/
  refine mul_le_mul_of_nonneg_right ?_ hε
  have h3 : ∀ t, ‖L (f t)‖ ≤ ‖L‖ * ‖f t‖ := by
    intro t
    exact L.le_opNorm (f t)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Norm.norm (HSub.hSub ((L (f t)) (g (HSub.hSub x₀ t))) ( …
    h3 : ∀ (t : G), LE.le (Norm.norm (L (f t))) (HMul.hMul (Norm.norm L) (Norm.nor …
    ⊢ LE.le (MeasureTheory.integral μ fun a => Norm.norm (L (f a))) (HMul.hMul (No …
  -/
  refine (integral_mono (L.integrable_comp hif).norm (hif.norm.const_mul _) h3).trans_eq ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace 𝕜 F
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : SecondCountableTopology G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hif : MeasureTheory.Integrable f μ
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hfg : MeasureTheory.ConvolutionExistsAt f g x₀ L μ
    h2 : ∀ (t : G), LE.le (Norm.norm (HSub.hSub ((L (f t)) (g (HSub.hSub x₀ t))) ( …
    h3 : ∀ (t : G), LE.le (Norm.norm (L (f t))) (HMul.hMul (Norm.norm L) (Norm.nor …
    ⊢ Eq (MeasureTheory.integral μ fun a => HMul.hMul (Norm.norm L) (Norm.norm (f  …
  -/
  rw [integral_mul_left]
  /-
    🎉 no goals
  -/


/-- Approximate `f ⋆ g` if the support of the `f` is bounded within a ball, and `g` is near `g x₀`
on a ball with the same radius around `x₀`.

This is a special case of `dist_convolution_le'` where `L` is `(•)`, `f` has integral 1 and `f` is
nonnegative. -/
theorem dist_convolution_le {f : G → ℝ} {x₀ : G} {R ε : ℝ} {z₀ : E'} (hε : 0 ≤ ε)
    (hf : support f ⊆ ball (0 : G) R) (hnf : ∀ x, 0 ≤ f x) (hintf : ∫ x, f x ∂μ = 1)
    (hmg : AEStronglyMeasurable g μ) (hg : ∀ x ∈ ball x₀ R, dist (g x) z₀ ≤ ε) :
    dist ((f ⋆[lsmul ℝ ℝ, μ] g : G → E') x₀) z₀ ≤ ε := by
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    g : G → E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    f : G → Real
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hnf : ∀ (x : G), LE.le 0 (f x)
    hintf : Eq (MeasureTheory.integral μ fun x => f x) 1
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    ⊢ LE.le (Dist.dist (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul R …
  -/
  have hif : Integrable f μ := integrable_of_integral_eq_one hintf
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    g : G → E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    f : G → Real
    x₀ : G
    R ε : Real
    z₀ : E'
    hε : LE.le 0 ε
    hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
    hnf : ∀ (x : G), LE.le 0 (f x)
    hintf : Eq (MeasureTheory.integral μ fun x => f x) 1
    hmg : MeasureTheory.AEStronglyMeasurable g μ
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
    hif : MeasureTheory.Integrable f μ
    ⊢ LE.le (Dist.dist (MeasureTheory.convolution f g (ContinuousLinearMap.lsmul R …
  -/
  convert (dist_convolution_le' (lsmul ℝ ℝ) hε hif hf hmg hg).trans _
    /-
      case h.e'_3.h.e'_4
      G : Type uG
      E' : Type uE'
      inst✝⁸ : NormedAddCommGroup E'
      g : G → E'
      inst✝⁷ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : SecondCountableTopology G
      inst✝³ : μ.IsAddLeftInvariant
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : NormedSpace Real E'
      inst✝ : CompleteSpace E'
      f : G → Real
      x₀ : G
      R ε : Real
      z₀ : E'
      hε : LE.le 0 ε
      hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
      hnf : ∀ (x : G), LE.le 0 (f x)
      hintf : Eq (MeasureTheory.integral μ fun x => f x) 1
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
      hif : MeasureTheory.Integrable f μ
      ⊢ Eq z₀ (MeasureTheory.integral μ fun t => ((ContinuousLinearMap.lsmul Real Re …
    -/
  · simp_rw [lsmul_apply, integral_smul_const, hintf, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      G : Type uG
      E' : Type uE'
      inst✝⁸ : NormedAddCommGroup E'
      g : G → E'
      inst✝⁷ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : SecondCountableTopology G
      inst✝³ : μ.IsAddLeftInvariant
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : NormedSpace Real E'
      inst✝ : CompleteSpace E'
      f : G → Real
      x₀ : G
      R ε : Real
      z₀ : E'
      hε : LE.le 0 ε
      hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
      hnf : ∀ (x : G), LE.le 0 (f x)
      hintf : Eq (MeasureTheory.integral μ fun x => f x) 1
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
      hif : MeasureTheory.Integrable f μ
      ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm (ContinuousLinearMap.lsmul Real Real) …
    -/
  · simp_rw [Real.norm_of_nonneg (hnf _), hintf, mul_one]
    /-
      case convert_2
      G : Type uG
      E' : Type uE'
      inst✝⁸ : NormedAddCommGroup E'
      g : G → E'
      inst✝⁷ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : BorelSpace G
      inst✝⁴ : SecondCountableTopology G
      inst✝³ : μ.IsAddLeftInvariant
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : NormedSpace Real E'
      inst✝ : CompleteSpace E'
      f : G → Real
      x₀ : G
      R ε : Real
      z₀ : E'
      hε : LE.le 0 ε
      hf : HasSubset.Subset (Function.support f) (Metric.ball 0 R)
      hnf : ∀ (x : G), LE.le 0 (f x)
      hintf : Eq (MeasureTheory.integral μ fun x => f x) 1
      hmg : MeasureTheory.AEStronglyMeasurable g μ
      hg : ∀ (x : G), Membership.mem (Metric.ball x₀ R) x → LE.le (Dist.dist (g x) z …
      hif : MeasureTheory.Integrable f μ
      ⊢ LE.le (HMul.hMul (Norm.norm (ContinuousLinearMap.lsmul Real Real)) ε) ε
    -/
    exact (mul_le_mul_of_nonneg_right opNorm_lsmul_le hε).trans_eq (one_mul ε)
    /-
      🎉 no goals
    -/


/-- `(φ i ⋆ g i) (k i)` tends to `z₀` as `i` tends to some filter `l` if
* `φ` is a sequence of nonnegative functions with integral `1` as `i` tends to `l`;
* The support of `φ` tends to small neighborhoods around `(0 : G)` as `i` tends to `l`;
* `g i` is `mu`-a.e. strongly measurable as `i` tends to `l`;
* `g i x` tends to `z₀` as `(i, x)` tends to `l ×ˢ 𝓝 x₀`;
* `k i` tends to `x₀`.

See also `ContDiffBump.convolution_tendsto_right`.
-/
theorem convolution_tendsto_right {ι} {g : ι → G → E'} {l : Filter ι} {x₀ : G} {z₀ : E'}
    {φ : ι → G → ℝ} {k : ι → G} (hnφ : ∀ᶠ i in l, ∀ x, 0 ≤ φ i x)
    (hiφ : ∀ᶠ i in l, ∫ x, φ i x ∂μ = 1)
    -- todo: we could weaken this to "the integral tends to 1"
    (hφ : Tendsto (fun n => support (φ n)) l (𝓝 0).smallSets)
    (hmg : ∀ᶠ i in l, AEStronglyMeasurable (g i) μ) (hcg : Tendsto (uncurry g) (l ×ˢ 𝓝 x₀) (𝓝 z₀))
    (hk : Tendsto k l (𝓝 x₀)) :
    Tendsto (fun i : ι => (φ i ⋆[lsmul ℝ ℝ, μ] g i : G → E') (k i)) l (𝓝 z₀) := by
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hφ : Filter.Tendsto (fun n => Function.support (φ n)) l (nhds 0).smallSets
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hcg : Filter.Tendsto (Function.uncurry g) (SProd.sprod l (nhds x₀)) (nhds z₀)
    hk : Filter.Tendsto k l (nhds x₀)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.convolution (φ i) (g i) (ContinuousLi …
  -/
  simp_rw [tendsto_smallSets_iff] at hφ
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hcg : Filter.Tendsto (Function.uncurry g) (SProd.sprod l (nhds x₀)) (nhds z₀)
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.convolution (φ i) (g i) (ContinuousLi …
  -/
  rw [Metric.tendsto_nhds] at hcg ⊢
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hcg : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist ( …
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (Meas …
  -/
  simp_rw [Metric.eventually_prod_nhds_iff] at hcg
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (Meas …
  -/
  intro ε hε
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  have h2ε : 0 < ε / 3 := div_pos hε (by norm_num)
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  obtain ⟨p, hp, δ, hδ, hgδ⟩ := hcg _ h2ε
  /-
    case intro.intro.intro.intro
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  dsimp only [uncurry] at hgδ
  /-
    case intro.intro.intro.intro
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  have h2k := hk.eventually (ball_mem_nhds x₀ <| half_pos hδ)
  /-
    case intro.intro.intro.intro
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  have h2φ := hφ (ball (0 : G) _) <| ball_mem_nhds _ (half_pos hδ)
  /-
    case intro.intro.intro.intro
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    h2φ : Filter.Eventually (fun x => HasSubset.Subset (Function.support (φ x)) (M …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.convolution (φ x …
  -/
  filter_upwards [hp, h2k, h2φ, hnφ, hiφ, hmg] with i hpi hki hφi hnφi hiφi hmgi
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    h2φ : Filter.Eventually (fun x => HasSubset.Subset (Function.support (φ x)) (M …
    i : ι
    hpi : p i
    hki : LT.lt (Dist.dist (k i) x₀) (HDiv.hDiv δ 2)
    hφi : HasSubset.Subset (Function.support (φ i)) (Metric.ball 0 (HDiv.hDiv δ 2))
    hnφi : ∀ (x : G), LE.le 0 (φ i x)
    hiφi : Eq (MeasureTheory.integral μ fun x => φ i x) 1
    hmgi : MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ LT.lt (Dist.dist (MeasureTheory.convolution (φ i) (g i) (ContinuousLinearMap …
  -/
  have hgi : dist (g i (k i)) z₀ < ε / 3 := hgδ hpi (hki.trans <| half_lt_self hδ)
  have h1 : ∀ x' ∈ ball (k i) (δ / 2), dist (g i x') (g i (k i)) ≤ ε / 3 + ε / 3 := by
    intro x' hx'
    refine (dist_triangle_right _ _ _).trans (add_le_add (hgδ hpi ?_).le hgi.le)
    exact ((dist_triangle _ _ _).trans_lt (add_lt_add hx'.out hki)).trans_eq (add_halves δ)
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    h2φ : Filter.Eventually (fun x => HasSubset.Subset (Function.support (φ x)) (M …
    i : ι
    hpi : p i
    hki : LT.lt (Dist.dist (k i) x₀) (HDiv.hDiv δ 2)
    hφi : HasSubset.Subset (Function.support (φ i)) (Metric.ball 0 (HDiv.hDiv δ 2))
    hnφi : ∀ (x : G), LE.le 0 (φ i x)
    hiφi : Eq (MeasureTheory.integral μ fun x => φ i x) 1
    hmgi : MeasureTheory.AEStronglyMeasurable (g i) μ
    hgi : LT.lt (Dist.dist (g i (k i)) z₀) (HDiv.hDiv ε 3)
    h1 : ∀ (x' : G), Membership.mem (Metric.ball (k i) (HDiv.hDiv δ 2)) x' → LE.le …
    ⊢ LT.lt (Dist.dist (MeasureTheory.convolution (φ i) (g i) (ContinuousLinearMap …
  -/
  have := dist_convolution_le (add_pos h2ε h2ε).le hφi hnφi hiφi hmgi h1
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    h2φ : Filter.Eventually (fun x => HasSubset.Subset (Function.support (φ x)) (M …
    i : ι
    hpi : p i
    hki : LT.lt (Dist.dist (k i) x₀) (HDiv.hDiv δ 2)
    hφi : HasSubset.Subset (Function.support (φ i)) (Metric.ball 0 (HDiv.hDiv δ 2))
    hnφi : ∀ (x : G), LE.le 0 (φ i x)
    hiφi : Eq (MeasureTheory.integral μ fun x => φ i x) 1
    hmgi : MeasureTheory.AEStronglyMeasurable (g i) μ
    hgi : LT.lt (Dist.dist (g i (k i)) z₀) (HDiv.hDiv ε 3)
    h1 : ∀ (x' : G), Membership.mem (Metric.ball (k i) (HDiv.hDiv δ 2)) x' → LE.le …
    this : LE.le (Dist.dist (MeasureTheory.convolution (φ i) (g i) (ContinuousLine …
    ⊢ LT.lt (Dist.dist (MeasureTheory.convolution (φ i) (g i) (ContinuousLinearMap …
  -/
  refine ((dist_triangle _ _ _).trans_lt (add_lt_add_of_le_of_lt this hgi)).trans_eq ?_
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : SeminormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : SecondCountableTopology G
    inst✝³ : μ.IsAddLeftInvariant
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : NormedSpace Real E'
    inst✝ : CompleteSpace E'
    ι : Type u_1
    g : ι → G → E'
    l : Filter ι
    x₀ : G
    z₀ : E'
    φ : ι → G → Real
    k : ι → G
    hnφ : Filter.Eventually (fun i => ∀ (x : G), LE.le 0 (φ i x)) l
    hiφ : Filter.Eventually (fun i => Eq (MeasureTheory.integral μ fun x => φ i x) …
    hmg : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (g i) μ) l
    hk : Filter.Tendsto k l (nhds x₀)
    hφ : ∀ (t : Set G), Membership.mem (nhds 0) t → Filter.Eventually (fun x => Ha …
    hcg : ∀ (ε : Real), GT.gt ε 0 → Exists fun pa => And (Filter.Eventually (fun i …
    ε : Real
    hε : GT.gt ε 0
    h2ε : LT.lt 0 (HDiv.hDiv ε 3)
    p : ι → Prop
    hp : Filter.Eventually (fun i => p i) l
    δ : Real
    hδ : GT.gt δ 0
    hgδ : ∀ ⦃i : ι⦄, p i → ∀ ⦃x : G⦄, LT.lt (Dist.dist x x₀) δ → LT.lt (Dist.dist  …
    h2k : Filter.Eventually (fun x => LT.lt (Dist.dist (k x) x₀) (HDiv.hDiv δ 2)) l
    h2φ : Filter.Eventually (fun x => HasSubset.Subset (Function.support (φ x)) (M …
    i : ι
    hpi : p i
    hki : LT.lt (Dist.dist (k i) x₀) (HDiv.hDiv δ 2)
    hφi : HasSubset.Subset (Function.support (φ i)) (Metric.ball 0 (HDiv.hDiv δ 2))
    hnφi : ∀ (x : G), LE.le 0 (φ i x)
    hiφi : Eq (MeasureTheory.integral μ fun x => φ i x) 1
    hmgi : MeasureTheory.AEStronglyMeasurable (g i) μ
    hgi : LT.lt (Dist.dist (g i (k i)) z₀) (HDiv.hDiv ε 3)
    h1 : ∀ (x' : G), Membership.mem (Metric.ball (k i) (HDiv.hDiv δ 2)) x' → LE.le …
    this : LE.le (Dist.dist (MeasureTheory.convolution (φ i) (g i) (ContinuousLine …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HDiv.hDiv ε 3) (HDiv.hDiv ε 3)) (HDiv.hDiv ε 3)) ε
  -/
  field_simp; ring_nf
              /-
                🎉 no goals
              -/


theorem integral_convolution [MeasurableAdd₂ G] [MeasurableNeg G] [NormedSpace ℝ E]
    [NormedSpace ℝ E'] [CompleteSpace E] [CompleteSpace E'] (hf : Integrable f ν)
    (hg : Integrable g μ) : ∫ x, (f ⋆[L, ν] g) x ∂μ = L (∫ x, f x ∂ν) (∫ x, g x ∂μ) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedAddCommGroup E'
    inst✝¹⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : NormedSpace 𝕜 E
    inst✝¹⁴ : NormedSpace 𝕜 E'
    inst✝¹³ : NormedSpace Real F
    inst✝¹² : NormedSpace 𝕜 F
    inst✝¹¹ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁰ : CompleteSpace F
    inst✝⁹ : AddGroup G
    inst✝⁸ : MeasureTheory.SFinite μ
    inst✝⁷ : MeasureTheory.SFinite ν
    inst✝⁶ : μ.IsAddRightInvariant
    inst✝⁵ : MeasurableAdd₂ G
    inst✝⁴ : MeasurableNeg G
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace E'
    hf : MeasureTheory.Integrable f ν
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.convolution f g L ν x) ( …
  -/
  refine (integral_integral_swap (by apply hf.convolution_integrand L hg)).trans ?_
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedAddCommGroup E'
    inst✝¹⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : NormedSpace 𝕜 E
    inst✝¹⁴ : NormedSpace 𝕜 E'
    inst✝¹³ : NormedSpace Real F
    inst✝¹² : NormedSpace 𝕜 F
    inst✝¹¹ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁰ : CompleteSpace F
    inst✝⁹ : AddGroup G
    inst✝⁸ : MeasureTheory.SFinite μ
    inst✝⁷ : MeasureTheory.SFinite ν
    inst✝⁶ : μ.IsAddRightInvariant
    inst✝⁵ : MeasurableAdd₂ G
    inst✝⁴ : MeasurableNeg G
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace E'
    hf : MeasureTheory.Integrable f ν
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.integral ν fun y => MeasureTheory.integral μ fun x => (L ( …
  -/
  simp_rw [integral_comp_comm _ (hg.comp_sub_right _), integral_sub_right_eq_self]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedAddCommGroup E'
    inst✝¹⁷ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : NormedSpace 𝕜 E
    inst✝¹⁴ : NormedSpace 𝕜 E'
    inst✝¹³ : NormedSpace Real F
    inst✝¹² : NormedSpace 𝕜 F
    inst✝¹¹ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁰ : CompleteSpace F
    inst✝⁹ : AddGroup G
    inst✝⁸ : MeasureTheory.SFinite μ
    inst✝⁷ : MeasureTheory.SFinite ν
    inst✝⁶ : μ.IsAddRightInvariant
    inst✝⁵ : MeasurableAdd₂ G
    inst✝⁴ : MeasurableNeg G
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace E'
    hf : MeasureTheory.Integrable f ν
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.integral ν fun y => (L (f y)) (MeasureTheory.integral μ fu …
  -/
  exact (L.flip (∫ x, g x ∂μ)).integral_comp_comm hf
  /-
    🎉 no goals
  -/


/-- Convolution is associative. This has a weak but inconvenient integrability condition.
See also `MeasureTheory.convolution_assoc`. -/
theorem convolution_assoc' (hL : ∀ (x : E) (y : E') (z : E''), L₂ (L x y) z = L₃ x (L₄ y z))
    {x₀ : G} (hfg : ∀ᵐ y ∂μ, ConvolutionExistsAt f g y L ν)
    (hgk : ∀ᵐ x ∂ν, ConvolutionExistsAt g k x L₄ μ)
    (hi : Integrable (uncurry fun x y => (L₃ (f y)) ((L₄ (g (x - y))) (k (x₀ - x)))) (μ.prod ν)) :
    ((f ⋆[L, ν] g) ⋆[L₂, μ] k) x₀ = (f ⋆[L₃, ν] g ⋆[L₄, μ] k) x₀ :=
  calc
    ((f ⋆[L, ν] g) ⋆[L₂, μ] k) x₀ = ∫ t, L₂ (∫ s, L (f s) (g (t - s)) ∂ν) (k (x₀ - t)) ∂μ := rfl
    _ = ∫ t, ∫ s, L₂ (L (f s) (g (t - s))) (k (x₀ - t)) ∂ν ∂μ :=
      (integral_congr_ae (hfg.mono fun t ht => ((L₂.flip (k (x₀ - t))).integral_comp_comm ht).symm))
                                                                     /-
                                                                       𝕜 : Type u𝕜
                                                                       G : Type uG
                                                                       E : Type uE
                                                                       E' : Type uE'
                                                                       E'' : Type uE''
                                                                       F : Type uF
                                                                       F' : Type uF'
                                                                       F'' : Type uF''
                                                                       inst✝²⁶ : NormedAddCommGroup E
                                                                       inst✝²⁵ : NormedAddCommGroup E'
                                                                       inst✝²⁴ : NormedAddCommGroup E''
                                                                       inst✝²³ : NormedAddCommGroup F
                                                                       f : G → E
                                                                       g : G → E'
                                                                       inst✝²² : RCLike 𝕜
                                                                       inst✝²¹ : NormedSpace 𝕜 E
                                                                       inst✝²⁰ : NormedSpace 𝕜 E'
                                                                       inst✝¹⁹ : NormedSpace 𝕜 E''
                                                                       inst✝¹⁸ : NormedSpace Real F
                                                                       inst✝¹⁷ : NormedSpace 𝕜 F
                                                                       inst✝¹⁶ : MeasurableSpace G
                                                                       μ ν : MeasureTheory.Measure G
                                                                       L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
                                                                       inst✝¹⁵ : CompleteSpace F
                                                                       inst✝¹⁴ : NormedAddCommGroup F'
                                                                       inst✝¹³ : NormedSpace Real F'
                                                                       inst✝¹² : NormedSpace 𝕜 F'
                                                                       inst✝¹¹ : CompleteSpace F'
                                                                       inst✝¹⁰ : NormedAddCommGroup F''
                                                                       inst✝⁹ : NormedSpace Real F''
                                                                       inst✝⁸ : NormedSpace 𝕜 F''
                                                                       inst✝⁷ : CompleteSpace F''
                                                                       k : G → E''
                                                                       L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
                                                                       L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
                                                                       L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
                                                                       inst✝⁶ : AddGroup G
                                                                       inst✝⁵ : MeasureTheory.SFinite μ
                                                                       inst✝⁴ : MeasureTheory.SFinite ν
                                                                       inst✝³ : μ.IsAddRightInvariant
                                                                       inst✝² : MeasurableAdd₂ G
                                                                       inst✝¹ : ν.IsAddRightInvariant
                                                                       inst✝ : MeasurableNeg G
                                                                       hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
                                                                       x₀ : G
                                                                       hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
                                                                       hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
                                                                       hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
                                                                       ⊢ Eq (MeasureTheory.integral μ fun t => MeasureTheory.integral ν fun s => (L₂  …
                                                                     -/
    _ = ∫ t, ∫ s, L₃ (f s) (L₄ (g (t - s)) (k (x₀ - t))) ∂ν ∂μ := by simp_rw [hL]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                                     /-
                                                                       𝕜 : Type u𝕜
                                                                       G : Type uG
                                                                       E : Type uE
                                                                       E' : Type uE'
                                                                       E'' : Type uE''
                                                                       F : Type uF
                                                                       F' : Type uF'
                                                                       F'' : Type uF''
                                                                       inst✝²⁶ : NormedAddCommGroup E
                                                                       inst✝²⁵ : NormedAddCommGroup E'
                                                                       inst✝²⁴ : NormedAddCommGroup E''
                                                                       inst✝²³ : NormedAddCommGroup F
                                                                       f : G → E
                                                                       g : G → E'
                                                                       inst✝²² : RCLike 𝕜
                                                                       inst✝²¹ : NormedSpace 𝕜 E
                                                                       inst✝²⁰ : NormedSpace 𝕜 E'
                                                                       inst✝¹⁹ : NormedSpace 𝕜 E''
                                                                       inst✝¹⁸ : NormedSpace Real F
                                                                       inst✝¹⁷ : NormedSpace 𝕜 F
                                                                       inst✝¹⁶ : MeasurableSpace G
                                                                       μ ν : MeasureTheory.Measure G
                                                                       L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
                                                                       inst✝¹⁵ : CompleteSpace F
                                                                       inst✝¹⁴ : NormedAddCommGroup F'
                                                                       inst✝¹³ : NormedSpace Real F'
                                                                       inst✝¹² : NormedSpace 𝕜 F'
                                                                       inst✝¹¹ : CompleteSpace F'
                                                                       inst✝¹⁰ : NormedAddCommGroup F''
                                                                       inst✝⁹ : NormedSpace Real F''
                                                                       inst✝⁸ : NormedSpace 𝕜 F''
                                                                       inst✝⁷ : CompleteSpace F''
                                                                       k : G → E''
                                                                       L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
                                                                       L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
                                                                       L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
                                                                       inst✝⁶ : AddGroup G
                                                                       inst✝⁵ : MeasureTheory.SFinite μ
                                                                       inst✝⁴ : MeasureTheory.SFinite ν
                                                                       inst✝³ : μ.IsAddRightInvariant
                                                                       inst✝² : MeasurableAdd₂ G
                                                                       inst✝¹ : ν.IsAddRightInvariant
                                                                       inst✝ : MeasurableNeg G
                                                                       hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
                                                                       x₀ : G
                                                                       hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
                                                                       hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
                                                                       hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
                                                                       ⊢ Eq (MeasureTheory.integral μ fun t => MeasureTheory.integral ν fun s => (L₃  …
                                                                     -/
    _ = ∫ s, ∫ t, L₃ (f s) (L₄ (g (t - s)) (k (x₀ - t))) ∂μ ∂ν := by rw [integral_integral_swap hi]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    _ = ∫ s, ∫ u, L₃ (f s) (L₄ (g u) (k (x₀ - s - u))) ∂μ ∂ν := by
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        ⊢ Eq (MeasureTheory.integral ν fun s => MeasureTheory.integral μ fun t => (L₃  …
      -/
      congr; ext t
      /-
        case e_f.h
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        t : G
        ⊢ Eq (MeasureTheory.integral μ fun t_1 => (L₃ (f t)) ((L₄ (g (HSub.hSub t_1 t) …
      -/
      rw [eq_comm, ← integral_sub_right_eq_self _ t]
      /-
        case e_f.h
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        t : G
        ⊢ Eq (MeasureTheory.integral μ fun x => (L₃ (f t)) ((L₄ (g (HSub.hSub x t))) ( …
      -/
      simp_rw [sub_sub_sub_cancel_right]
      /-
        🎉 no goals
      -/
    _ = ∫ s, L₃ (f s) (∫ u, L₄ (g u) (k (x₀ - s - u)) ∂μ) ∂ν := by
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        ⊢ Eq (MeasureTheory.integral ν fun s => MeasureTheory.integral μ fun u => (L₃  …
      -/
      refine integral_congr_ae ?_
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        ⊢ (MeasureTheory.ae ν).EventuallyEq (fun s => MeasureTheory.integral μ fun u = …
      -/
      refine ((quasiMeasurePreserving_sub_left_of_right_invariant ν x₀).ae hgk).mono fun t ht => ?_
      /-
        𝕜 : Type u𝕜
        G : Type uG
        E : Type uE
        E' : Type uE'
        E'' : Type uE''
        F : Type uF
        F' : Type uF'
        F'' : Type uF''
        inst✝²⁶ : NormedAddCommGroup E
        inst✝²⁵ : NormedAddCommGroup E'
        inst✝²⁴ : NormedAddCommGroup E''
        inst✝²³ : NormedAddCommGroup F
        f : G → E
        g : G → E'
        inst✝²² : RCLike 𝕜
        inst✝²¹ : NormedSpace 𝕜 E
        inst✝²⁰ : NormedSpace 𝕜 E'
        inst✝¹⁹ : NormedSpace 𝕜 E''
        inst✝¹⁸ : NormedSpace Real F
        inst✝¹⁷ : NormedSpace 𝕜 F
        inst✝¹⁶ : MeasurableSpace G
        μ ν : MeasureTheory.Measure G
        L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
        inst✝¹⁵ : CompleteSpace F
        inst✝¹⁴ : NormedAddCommGroup F'
        inst✝¹³ : NormedSpace Real F'
        inst✝¹² : NormedSpace 𝕜 F'
        inst✝¹¹ : CompleteSpace F'
        inst✝¹⁰ : NormedAddCommGroup F''
        inst✝⁹ : NormedSpace Real F''
        inst✝⁸ : NormedSpace 𝕜 F''
        inst✝⁷ : CompleteSpace F''
        k : G → E''
        L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
        L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
        inst✝⁶ : AddGroup G
        inst✝⁵ : MeasureTheory.SFinite μ
        inst✝⁴ : MeasureTheory.SFinite ν
        inst✝³ : μ.IsAddRightInvariant
        inst✝² : MeasurableAdd₂ G
        inst✝¹ : ν.IsAddRightInvariant
        inst✝ : MeasurableNeg G
        hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
        x₀ : G
        hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
        hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt g k x L₄ μ …
        hi : MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g  …
        t : G
        ht : MeasureTheory.ConvolutionExistsAt g k (HSub.hSub x₀ t) L₄ μ
        ⊢ Eq ((fun s => MeasureTheory.integral μ fun u => (L₃ (f s)) ((L₄ (g u)) (k (H …
      -/
      exact (L₃ (f t)).integral_comp_comm ht
      /-
        🎉 no goals
      -/
    _ = (f ⋆[L₃, ν] g ⋆[L₄, μ] k) x₀ := rfl


/-- Convolution is associative. This requires that
* all maps are a.e. strongly measurable w.r.t one of the measures
* `f ⋆[L, ν] g` exists almost everywhere
* `‖g‖ ⋆[μ] ‖k‖` exists almost everywhere
* `‖f‖ ⋆[ν] (‖g‖ ⋆[μ] ‖k‖)` exists at `x₀` -/
theorem convolution_assoc (hL : ∀ (x : E) (y : E') (z : E''), L₂ (L x y) z = L₃ x (L₄ y z)) {x₀ : G}
    (hf : AEStronglyMeasurable f ν) (hg : AEStronglyMeasurable g μ) (hk : AEStronglyMeasurable k μ)
    (hfg : ∀ᵐ y ∂μ, ConvolutionExistsAt f g y L ν)
    (hgk : ∀ᵐ x ∂ν, ConvolutionExistsAt (fun x => ‖g x‖) (fun x => ‖k x‖) x (mul ℝ ℝ) μ)
    (hfgk :
      ConvolutionExistsAt (fun x => ‖f x‖) ((fun x => ‖g x‖) ⋆[mul ℝ ℝ, μ] fun x => ‖k x‖) x₀
        (mul ℝ ℝ) ν) :
    ((f ⋆[L, ν] g) ⋆[L₂, μ] k) x₀ = (f ⋆[L₃, ν] g ⋆[L₄, μ] k) x₀ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    ⊢ Eq (MeasureTheory.convolution (MeasureTheory.convolution f g L ν) k L₂ μ x₀) …
  -/
  refine convolution_assoc' L L₂ L₃ L₄ hL hfg (hgk.mono fun x hx => hx.ofNorm L₄ hg hk) ?_
  -- the following is similar to `Integrable.convolution_integrand`
  have h_meas :
    AEStronglyMeasurable (uncurry fun x y => L₃ (f y) (L₄ (g x) (k (x₀ - y - x))))
      (μ.prod ν) := by
    refine L₃.aestronglyMeasurable_comp₂ hf.snd ?_
    refine L₄.aestronglyMeasurable_comp₂ hg.fst ?_
    refine (hk.mono_ac ?_).comp_measurable
      ((measurable_const.sub measurable_snd).sub measurable_fst)
    refine QuasiMeasurePreserving.absolutelyContinuous ?_
    refine QuasiMeasurePreserving.prod_of_left
      ((measurable_const.sub measurable_snd).sub measurable_fst) (Eventually.of_forall fun y => ?_)
    dsimp only
    exact quasiMeasurePreserving_sub_left_of_right_invariant μ _
  have h2_meas :
    AEStronglyMeasurable (fun y => ∫ x, ‖L₃ (f y) (L₄ (g x) (k (x₀ - y - x)))‖ ∂μ) ν :=
    h_meas.prod_swap.norm.integral_prod_right'
  have h3 : map (fun z : G × G => (z.1 - z.2, z.2)) (μ.prod ν) = μ.prod ν :=
    (measurePreserving_sub_prod μ ν).map_eq
  suffices Integrable (uncurry fun x y => L₃ (f y) (L₄ (g x) (k (x₀ - y - x)))) (μ.prod ν) by
    rw [← h3] at this
    convert this.comp_measurable (measurable_sub.prod_mk measurable_snd)
    ext ⟨x, y⟩
    simp (config := { unfoldPartialApp := true }) only [uncurry, Function.comp_apply,
      sub_sub_sub_cancel_right]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    h_meas : MeasureTheory.AEStronglyMeasurable (Function.uncurry fun x y => (L₃ ( …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    h3 : Eq (MeasureTheory.Measure.map (fun z => { fst := HSub.hSub z.1 z.2, snd : …
    ⊢ MeasureTheory.Integrable (Function.uncurry fun x y => (L₃ (f y)) ((L₄ (g x)) …
  -/
  simp_rw [integrable_prod_iff' h_meas]
  refine ⟨((quasiMeasurePreserving_sub_left_of_right_invariant ν x₀).ae hgk).mono fun t ht =>
    (L₃ (f t)).integrable_comp <| ht.ofNorm L₄ hg hk, ?_⟩
  refine (hfgk.const_mul (‖L₃‖ * ‖L₄‖)).mono' h2_meas
    (((quasiMeasurePreserving_sub_left_of_right_invariant ν x₀).ae hgk).mono fun t ht => ?_)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    h_meas : MeasureTheory.AEStronglyMeasurable (Function.uncurry fun x y => (L₃ ( …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    h3 : Eq (MeasureTheory.Measure.map (fun z => { fst := HSub.hSub z.1 z.2, snd : …
    t : G
    ht : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (g x)) (fun x => No …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun x => Norm.norm (Function.uncu …
  -/
  simp_rw [convolution_def, mul_apply', mul_mul_mul_comm ‖L₃‖ ‖L₄‖, ← integral_mul_left]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    h_meas : MeasureTheory.AEStronglyMeasurable (Function.uncurry fun x y => (L₃ ( …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    h3 : Eq (MeasureTheory.Measure.map (fun z => { fst := HSub.hSub z.1 z.2, snd : …
    t : G
    ht : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (g x)) (fun x => No …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun x => Norm.norm (Function.uncu …
  -/
  rw [Real.norm_of_nonneg (by positivity)]
  refine integral_mono_of_nonneg (Eventually.of_forall fun t => norm_nonneg _)
    ((ht.const_mul _).const_mul _) (Eventually.of_forall fun s => ?_)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    h_meas : MeasureTheory.AEStronglyMeasurable (Function.uncurry fun x y => (L₃ ( …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    h3 : Eq (MeasureTheory.Measure.map (fun z => { fst := HSub.hSub z.1 z.2, snd : …
    t : G
    ht : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (g x)) (fun x => No …
    s : G
    ⊢ LE.le ((fun x => Norm.norm (Function.uncurry (fun x y => (L₃ (f y)) ((L₄ (g  …
  -/
  simp only [← mul_assoc ‖L₄‖]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    F' : Type uF'
    F'' : Type uF''
    inst✝²⁶ : NormedAddCommGroup E
    inst✝²⁵ : NormedAddCommGroup E'
    inst✝²⁴ : NormedAddCommGroup E''
    inst✝²³ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝²² : RCLike 𝕜
    inst✝²¹ : NormedSpace 𝕜 E
    inst✝²⁰ : NormedSpace 𝕜 E'
    inst✝¹⁹ : NormedSpace 𝕜 E''
    inst✝¹⁸ : NormedSpace Real F
    inst✝¹⁷ : NormedSpace 𝕜 F
    inst✝¹⁶ : MeasurableSpace G
    μ ν : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹⁵ : CompleteSpace F
    inst✝¹⁴ : NormedAddCommGroup F'
    inst✝¹³ : NormedSpace Real F'
    inst✝¹² : NormedSpace 𝕜 F'
    inst✝¹¹ : CompleteSpace F'
    inst✝¹⁰ : NormedAddCommGroup F''
    inst✝⁹ : NormedSpace Real F''
    inst✝⁸ : NormedSpace 𝕜 F''
    inst✝⁷ : CompleteSpace F''
    k : G → E''
    L₂ : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₃ : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    L₄ : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜) …
    inst✝⁶ : AddGroup G
    inst✝⁵ : MeasureTheory.SFinite μ
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : μ.IsAddRightInvariant
    inst✝² : MeasurableAdd₂ G
    inst✝¹ : ν.IsAddRightInvariant
    inst✝ : MeasurableNeg G
    hL : ∀ (x : E) (y : E') (z : E''), Eq ((L₂ ((L x) y)) z) ((L₃ x) ((L₄ y) z))
    x₀ : G
    hf : MeasureTheory.AEStronglyMeasurable f ν
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hk : MeasureTheory.AEStronglyMeasurable k μ
    hfg : Filter.Eventually (fun y => MeasureTheory.ConvolutionExistsAt f g y L ν) …
    hgk : Filter.Eventually (fun x => MeasureTheory.ConvolutionExistsAt (fun x =>  …
    hfgk : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (f x)) (MeasureTh …
    h_meas : MeasureTheory.AEStronglyMeasurable (Function.uncurry fun x y => (L₃ ( …
    h2_meas : MeasureTheory.AEStronglyMeasurable (fun y => MeasureTheory.integral  …
    h3 : Eq (MeasureTheory.Measure.map (fun z => { fst := HSub.hSub z.1 z.2, snd : …
    t : G
    ht : MeasureTheory.ConvolutionExistsAt (fun x => Norm.norm (g x)) (fun x => No …
    s : G
    ⊢ LE.le (Norm.norm (Function.uncurry (fun x y => (L₃ (f y)) ((L₄ (g x)) (k (HS …
  -/
  apply_rules [ContinuousLinearMap.le_of_opNorm₂_le_of_le, le_rfl]
  /-
    🎉 no goals
  -/


theorem convolution_precompR_apply {g : G → E'' →L[𝕜] E'} (hf : LocallyIntegrable f μ)
    (hcg : HasCompactSupport g) (hg : Continuous g) (x₀ : G) (x : E'') :
    (f ⋆[L.precompR E'', μ] g) x₀ x = (f ⋆[L, μ] fun a => g a x) x₀ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup E''
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 E''
    inst✝⁴ : NormedSpace Real F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : NormedAddCommGroup G
    inst✝ : BorelSpace G
    g : G → ContinuousLinearMap (RingHom.id 𝕜) E'' E'
    hf : MeasureTheory.LocallyIntegrable f μ
    hcg : HasCompactSupport g
    hg : Continuous g
    x₀ : G
    x : E''
    ⊢ Eq ((MeasureTheory.convolution f g (ContinuousLinearMap.precompR E'' L) μ x₀ …
  -/
  have := hcg.convolutionExists_right (L.precompR E'' : _) hf hg x₀
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup E''
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 E''
    inst✝⁴ : NormedSpace Real F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : NormedAddCommGroup G
    inst✝ : BorelSpace G
    g : G → ContinuousLinearMap (RingHom.id 𝕜) E'' E'
    hf : MeasureTheory.LocallyIntegrable f μ
    hcg : HasCompactSupport g
    hg : Continuous g
    x₀ : G
    x : E''
    this : MeasureTheory.ConvolutionExistsAt f g x₀ (ContinuousLinearMap.precompR  …
    ⊢ Eq ((MeasureTheory.convolution f g (ContinuousLinearMap.precompR E'' L) μ x₀ …
  -/
  simp_rw [convolution_def, ContinuousLinearMap.integral_apply this]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    E'' : Type uE''
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup E''
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace 𝕜 E''
    inst✝⁴ : NormedSpace Real F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : NormedAddCommGroup G
    inst✝ : BorelSpace G
    g : G → ContinuousLinearMap (RingHom.id 𝕜) E'' E'
    hf : MeasureTheory.LocallyIntegrable f μ
    hcg : HasCompactSupport g
    hg : Continuous g
    x₀ : G
    x : E''
    this : MeasureTheory.ConvolutionExistsAt f g x₀ (ContinuousLinearMap.precompR  …
    ⊢ Eq (MeasureTheory.integral μ fun x_1 => (((ContinuousLinearMap.precompR E''  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Compute the total derivative of `f ⋆ g` if `g` is `C^1` with compact support and `f` is locally
integrable. To write down the total derivative as a convolution, we use
`ContinuousLinearMap.precompR`. -/
theorem _root_.HasCompactSupport.hasFDerivAt_convolution_right (hcg : HasCompactSupport g)
    (hf : LocallyIntegrable f μ) (hg : ContDiff 𝕜 1 g) (x₀ : G) :
    HasFDerivAt (f ⋆[L, μ] g) ((f ⋆[L.precompR G, μ] fderiv 𝕜 g) x₀) x₀ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  rcases hcg.eq_zero_or_finiteDimensional 𝕜 hg.continuous with (rfl | fin_dim)
    /-
      case inl
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hf : MeasureTheory.LocallyIntegrable f μ
      x₀ : G
      hcg : HasCompactSupport 0
      hg : ContDiff 𝕜 1 0
      ⊢ HasFDerivAt (MeasureTheory.convolution f 0 L μ) (MeasureTheory.convolution f …
    -/
  · have : fderiv 𝕜 (0 : G → E') = 0 := fderiv_const (0 : E')
    /-
      case inl
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hf : MeasureTheory.LocallyIntegrable f μ
      x₀ : G
      hcg : HasCompactSupport 0
      hg : ContDiff 𝕜 1 0
      this : Eq (fderiv 𝕜 0) 0
      ⊢ HasFDerivAt (MeasureTheory.convolution f 0 L μ) (MeasureTheory.convolution f …
    -/
    simp only [this, convolution_zero, Pi.zero_apply]
    /-
      case inl
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hf : MeasureTheory.LocallyIntegrable f μ
      x₀ : G
      hcg : HasCompactSupport 0
      hg : ContDiff 𝕜 1 0
      this : Eq (fderiv 𝕜 0) 0
      ⊢ HasFDerivAt 0 0 x₀
    -/
    exact hasFDerivAt_const (0 : F) x₀
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    fin_dim : FiniteDimensional 𝕜 G
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  have : ProperSpace G := FiniteDimensional.proper_rclike 𝕜 G
  /-
    case inr
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    fin_dim : FiniteDimensional 𝕜 G
    this : ProperSpace G
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  set L' := L.precompR G
  have h1 : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (fun t => L (f t) (g (x - t))) μ :=
    Eventually.of_forall
      (hf.aestronglyMeasurable.convolution_integrand_snd L hg.continuous.aestronglyMeasurable)
  have h2 : ∀ x, AEStronglyMeasurable (fun t => L' (f t) (fderiv 𝕜 g (x - t))) μ :=
    hf.aestronglyMeasurable.convolution_integrand_snd L'
      (hg.continuous_fderiv le_rfl).aestronglyMeasurable
  have h3 : ∀ x t, HasFDerivAt (fun x => g (x - t)) (fderiv 𝕜 g (x - t)) x := fun x t ↦ by
    simpa using
      (hg.differentiable le_rfl).differentiableAt.hasFDerivAt.comp x
        ((hasFDerivAt_id x).sub (hasFDerivAt_const t x))
  /-
    case inr
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    fin_dim : FiniteDimensional 𝕜 G
    this : ProperSpace G
    L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
    h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
    h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  let K' := -tsupport (fderiv 𝕜 g) + closedBall x₀ 1
  /-
    case inr
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    fin_dim : FiniteDimensional 𝕜 G
    this : ProperSpace G
    L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
    h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
    h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
    K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  have hK' : IsCompact K' := (hcg.fderiv 𝕜).neg.add (isCompact_closedBall x₀ 1)
  -- Porting note: was
  -- `refine' hasFDerivAt_integral_of_dominated_of_fderiv_le zero_lt_one h1 _ (h2 x₀) _ _ _`
  -- but it failed; surprisingly, `apply` works
  /-
    case inr
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsAddLeftInvariant
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 1 g
    x₀ : G
    fin_dim : FiniteDimensional 𝕜 G
    this : ProperSpace G
    L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
    h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
    h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
    K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
    hK' : IsCompact K'
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution f …
  -/
  apply hasFDerivAt_integral_of_dominated_of_fderiv_le zero_lt_one h1 _ (h2 x₀)
  · filter_upwards with t x hx using
      (hcg.fderiv 𝕜).convolution_integrand_bound_right L' (hg.continuous_fderiv le_rfl)
        (ball_subset_closedBall hx)
    /-
      case inr.bound_integrable
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hcg : HasCompactSupport g
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContDiff 𝕜 1 g
      x₀ : G
      fin_dim : FiniteDimensional 𝕜 G
      this : ProperSpace G
      L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
      h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
      h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
      h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
      K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
      hK' : IsCompact K'
      ⊢ MeasureTheory.Integrable (fun t => (HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g …
    -/
  · rw [integrable_indicator_iff hK'.measurableSet]
    /-
      case inr.bound_integrable
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hcg : HasCompactSupport g
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContDiff 𝕜 1 g
      x₀ : G
      fin_dim : FiniteDimensional 𝕜 G
      this : ProperSpace G
      L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
      h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
      h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
      h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
      K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
      hK' : IsCompact K'
      ⊢ MeasureTheory.IntegrableOn (fun t => HMul.hMul (HMul.hMul (Norm.norm L') (No …
    -/
    exact ((hf.integrableOn_isCompact hK').norm.const_mul _).mul_const _
    /-
      🎉 no goals
    -/
    /-
      case inr.h_diff
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hcg : HasCompactSupport g
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContDiff 𝕜 1 g
      x₀ : G
      fin_dim : FiniteDimensional 𝕜 G
      this : ProperSpace G
      L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
      h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
      h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
      h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
      K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
      hK' : IsCompact K'
      ⊢ Filter.Eventually (fun a => ∀ (x : G), Membership.mem (Metric.ball x₀ 1) x → …
    -/
  · exact Eventually.of_forall fun t x _ => (L _).hasFDerivAt.comp x (h3 x t)
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u𝕜
      G : Type uG
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedAddCommGroup F
      f : G → E
      g : G → E'
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : NormedSpace 𝕜 E
      inst✝⁸ : NormedSpace 𝕜 E'
      inst✝⁷ : NormedSpace Real F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : BorelSpace G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.IsAddLeftInvariant
      hcg : HasCompactSupport g
      hf : MeasureTheory.LocallyIntegrable f μ
      hg : ContDiff 𝕜 1 g
      x₀ : G
      fin_dim : FiniteDimensional 𝕜 G
      this : ProperSpace G
      L' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
      h1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun t =>  …
      h2 : ∀ (x : G), MeasureTheory.AEStronglyMeasurable (fun t => (L' (f t)) (fderi …
      h3 : ∀ (x t : G), HasFDerivAt (fun x => g (HSub.hSub x t)) (fderiv 𝕜 g (HSub.h …
      K' : Set G := HAdd.hAdd (Neg.neg (tsupport (fderiv 𝕜 g))) (Metric.closedBall x …
      hK' : IsCompact K'
      ⊢ MeasureTheory.Integrable (fun t => (L (f t)) (g (HSub.hSub x₀ t))) μ
    -/
  · exact hcg.convolutionExists_right L hf hg.continuous x₀
    /-
      🎉 no goals
    -/


theorem _root_.HasCompactSupport.hasFDerivAt_convolution_left [IsNegInvariant μ]
    (hcf : HasCompactSupport f) (hf : ContDiff 𝕜 1 f) (hg : LocallyIntegrable g μ) (x₀ : G) :
    HasFDerivAt (f ⋆[L, μ] g) ((fderiv 𝕜 f ⋆[L.precompL G, μ] g) x₀) x₀ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : RCLike 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : NormedAddCommGroup G
    inst✝⁴ : BorelSpace G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    hcf : HasCompactSupport f
    hf : ContDiff 𝕜 1 f
    hg : MeasureTheory.LocallyIntegrable g μ
    x₀ : G
    ⊢ HasFDerivAt (MeasureTheory.convolution f g L μ) (MeasureTheory.convolution ( …
  -/
  simp (config := { singlePass := true }) only [← convolution_flip]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedAddCommGroup E'
    inst✝¹² : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹¹ : RCLike 𝕜
    inst✝¹⁰ : NormedSpace 𝕜 E
    inst✝⁹ : NormedSpace 𝕜 E'
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝⁵ : NormedAddCommGroup G
    inst✝⁴ : BorelSpace G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    hcf : HasCompactSupport f
    hf : ContDiff 𝕜 1 f
    hg : MeasureTheory.LocallyIntegrable g μ
    x₀ : G
    ⊢ HasFDerivAt (MeasureTheory.convolution g f L.flip μ) (MeasureTheory.convolut …
  -/
  exact hcf.hasFDerivAt_convolution_right L.flip hg hf x₀
  /-
    🎉 no goals
  -/


theorem _root_.HasCompactSupport.hasDerivAt_convolution_right (hf : LocallyIntegrable f₀ μ)
    (hcg : HasCompactSupport g₀) (hg : ContDiff 𝕜 1 g₀) (x₀ : 𝕜) :
    HasDerivAt (f₀ ⋆[L, μ] g₀) ((f₀ ⋆[L, μ] deriv g₀) x₀) x₀ := by
  /-
    𝕜 : Type u𝕜
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace Real F
    inst✝² : NormedSpace 𝕜 F
    f₀ : 𝕜 → E
    g₀ : 𝕜 → E'
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    μ : MeasureTheory.Measure 𝕜
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    hf : MeasureTheory.LocallyIntegrable f₀ μ
    hcg : HasCompactSupport g₀
    hg : ContDiff 𝕜 1 g₀
    x₀ : 𝕜
    ⊢ HasDerivAt (MeasureTheory.convolution f₀ g₀ L μ) (MeasureTheory.convolution  …
  -/
  convert (hcg.hasFDerivAt_convolution_right L hf hg x₀).hasDerivAt using 1
  /-
    case h.e'_9
    𝕜 : Type u𝕜
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace Real F
    inst✝² : NormedSpace 𝕜 F
    f₀ : 𝕜 → E
    g₀ : 𝕜 → E'
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    μ : MeasureTheory.Measure 𝕜
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    hf : MeasureTheory.LocallyIntegrable f₀ μ
    hcg : HasCompactSupport g₀
    hg : ContDiff 𝕜 1 g₀
    x₀ : 𝕜
    ⊢ Eq (MeasureTheory.convolution f₀ (deriv g₀) L μ x₀) ((MeasureTheory.convolut …
  -/
  rw [convolution_precompR_apply L hf (hcg.fderiv 𝕜) (hg.continuous_fderiv le_rfl)]
  /-
    case h.e'_9
    𝕜 : Type u𝕜
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : NormedSpace Real F
    inst✝² : NormedSpace 𝕜 F
    f₀ : 𝕜 → E
    g₀ : 𝕜 → E'
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    μ : MeasureTheory.Measure 𝕜
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : MeasureTheory.SFinite μ
    hf : MeasureTheory.LocallyIntegrable f₀ μ
    hcg : HasCompactSupport g₀
    hg : ContDiff 𝕜 1 g₀
    x₀ : 𝕜
    ⊢ Eq (MeasureTheory.convolution f₀ (deriv g₀) L μ x₀) (MeasureTheory.convoluti …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem _root_.HasCompactSupport.hasDerivAt_convolution_left [IsNegInvariant μ]
    (hcf : HasCompactSupport f₀) (hf : ContDiff 𝕜 1 f₀) (hg : LocallyIntegrable g₀ μ) (x₀ : 𝕜) :
    HasDerivAt (f₀ ⋆[L, μ] g₀) ((deriv f₀ ⋆[L, μ] g₀) x₀) x₀ := by
  /-
    𝕜 : Type u𝕜
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace Real F
    inst✝³ : NormedSpace 𝕜 F
    f₀ : 𝕜 → E
    g₀ : 𝕜 → E'
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    μ : MeasureTheory.Measure 𝕜
    inst✝² : μ.IsAddLeftInvariant
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsNegInvariant
    hcf : HasCompactSupport f₀
    hf : ContDiff 𝕜 1 f₀
    hg : MeasureTheory.LocallyIntegrable g₀ μ
    x₀ : 𝕜
    ⊢ HasDerivAt (MeasureTheory.convolution f₀ g₀ L μ) (MeasureTheory.convolution  …
  -/
  simp (config := { singlePass := true }) only [← convolution_flip]
  /-
    𝕜 : Type u𝕜
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 E'
    inst✝⁴ : NormedSpace Real F
    inst✝³ : NormedSpace 𝕜 F
    f₀ : 𝕜 → E
    g₀ : 𝕜 → E'
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    μ : MeasureTheory.Measure 𝕜
    inst✝² : μ.IsAddLeftInvariant
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsNegInvariant
    hcf : HasCompactSupport f₀
    hf : ContDiff 𝕜 1 f₀
    hg : MeasureTheory.LocallyIntegrable g₀ μ
    x₀ : 𝕜
    ⊢ HasDerivAt (MeasureTheory.convolution g₀ f₀ L.flip μ) (MeasureTheory.convolu …
  -/
  exact hcf.hasDerivAt_convolution_right L.flip hg hf x₀
  /-
    🎉 no goals
  -/


/-- The derivative of the convolution `f * g` is given by `f * Dg`, when `f` is locally integrable
and `g` is `C^1` and compactly supported. Version where `g` depends on an additional parameter in an
open subset `s` of a parameter space `P` (and the compact support `k` is independent of the
parameter in `s`). -/
theorem hasFDerivAt_convolution_right_with_param {g : P → G → E'} {s : Set P} {k : Set G}
    (hs : IsOpen s) (hk : IsCompact k) (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0)
    (hf : LocallyIntegrable f μ) (hg : ContDiffOn 𝕜 1 (↿g) (s ×ˢ univ)) (q₀ : P × G)
    (hq₀ : q₀.1 ∈ s) :
    HasFDerivAt (fun q : P × G => (f ⋆[L, μ] g q.1) q.2)
      ((f ⋆[L.precompR (P × G), μ] fun x : G => fderiv 𝕜 (↿g) (q₀.1, x)) q₀.2) q₀ := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 1 (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    q₀ : Prod P G
    hq₀ : Membership.mem s q₀.1
    ⊢ HasFDerivAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (MeasureT …
  -/
  let g' := fderiv 𝕜 ↿g
  have A : ∀ p ∈ s, Continuous (g p) := fun p hp ↦ by
    refine hg.continuousOn.comp_continuous (continuous_const.prod_mk continuous_id') fun x => ?_
    simpa only [prod_mk_mem_set_prod_eq, mem_univ, and_true] using hp
  have A' : ∀ q : P × G, q.1 ∈ s → s ×ˢ univ ∈ 𝓝 q := fun q hq ↦ by
    apply (hs.prod isOpen_univ).mem_nhds
    simpa only [mem_prod, mem_univ, and_true] using hq
  -- The derivative of `g` vanishes away from `k`.
  have g'_zero : ∀ p x, p ∈ s → x ∉ k → g' (p, x) = 0 := by
    intro p x hp hx
    refine (hasFDerivAt_zero_of_eventually_const 0 ?_).fderiv
    have M2 : kᶜ ∈ 𝓝 x := hk.isClosed.isOpen_compl.mem_nhds hx
    have M1 : s ∈ 𝓝 p := hs.mem_nhds hp
    rw [nhds_prod_eq]
    filter_upwards [prod_mem_prod M1 M2]
    rintro ⟨p, y⟩ ⟨hp, hy⟩
    exact hgs p y hp hy
  /- We find a small neighborhood of `{q₀.1} × k` on which the derivative is uniformly bounded. This
    follows from the continuity at all points of the compact set `k`. -/
  obtain ⟨ε, C, εpos, h₀ε, hε⟩ :
      ∃ ε C, 0 < ε ∧ ball q₀.1 ε ⊆ s ∧ ∀ p x, ‖p - q₀.1‖ < ε → ‖g' (p, x)‖ ≤ C := by
    have A : IsCompact ({q₀.1} ×ˢ k) := isCompact_singleton.prod hk
    obtain ⟨t, kt, t_open, ht⟩ : ∃ t, {q₀.1} ×ˢ k ⊆ t ∧ IsOpen t ∧ IsBounded (g' '' t) := by
      have B : ContinuousOn g' (s ×ˢ univ) :=
        hg.continuousOn_fderiv_of_isOpen (hs.prod isOpen_univ) le_rfl
      apply exists_isOpen_isBounded_image_of_isCompact_of_continuousOn A (hs.prod isOpen_univ) _ B
      simp only [prod_subset_prod_iff, hq₀, singleton_subset_iff, subset_univ, and_self_iff,
        true_or]
    obtain ⟨ε, εpos, hε, h'ε⟩ :
      ∃ ε : ℝ, 0 < ε ∧ thickening ε ({q₀.fst} ×ˢ k) ⊆ t ∧ ball q₀.1 ε ⊆ s := by
      obtain ⟨ε, εpos, hε⟩ : ∃ ε : ℝ, 0 < ε ∧ thickening ε (({q₀.fst} : Set P) ×ˢ k) ⊆ t :=
        A.exists_thickening_subset_open t_open kt
      obtain ⟨δ, δpos, hδ⟩ : ∃ δ : ℝ, 0 < δ ∧ ball q₀.1 δ ⊆ s := Metric.isOpen_iff.1 hs _ hq₀
      refine ⟨min ε δ, lt_min εpos δpos, ?_, ?_⟩
      · exact Subset.trans (thickening_mono (min_le_left _ _) _) hε
      · exact Subset.trans (ball_subset_ball (min_le_right _ _)) hδ
    obtain ⟨C, Cpos, hC⟩ : ∃ C, 0 < C ∧ g' '' t ⊆ closedBall 0 C := ht.subset_closedBall_lt 0 0
    refine ⟨ε, C, εpos, h'ε, fun p x hp => ?_⟩
    have hps : p ∈ s := h'ε (mem_ball_iff_norm.2 hp)
    by_cases hx : x ∈ k
    · have H : (p, x) ∈ t := by
        apply hε
        refine mem_thickening_iff.2 ⟨(q₀.1, x), ?_, ?_⟩
        · simp only [hx, singleton_prod, mem_image, Prod.mk.inj_iff, eq_self_iff_true, true_and,
            exists_eq_right]
        · rw [← dist_eq_norm] at hp
          simpa only [Prod.dist_eq, εpos, dist_self, max_lt_iff, and_true] using hp
      have : g' (p, x) ∈ closedBall (0 : P × G →L[𝕜] E') C := hC (mem_image_of_mem _ H)
      rwa [mem_closedBall_zero_iff] at this
    · have : g' (p, x) = 0 := g'_zero _ _ hps hx
      rw [this]
      simpa only [norm_zero] using Cpos.le
  /- Now, we wish to apply a theorem on differentiation of integrals. For this, we need to check
    trivial measurability or integrability assumptions (in `I1`, `I2`, `I3`), as well as a uniform
    integrability assumption over the derivative (in `I4` and `I5`) and pointwise differentiability
    in `I6`. -/
  have I1 :
    ∀ᶠ x : P × G in 𝓝 q₀, AEStronglyMeasurable (fun a : G => L (f a) (g x.1 (x.2 - a))) μ := by
    filter_upwards [A' q₀ hq₀]
    rintro ⟨p, x⟩ ⟨hp, -⟩
    refine (HasCompactSupport.convolutionExists_right L ?_ hf (A _ hp) _).1
    apply hk.of_isClosed_subset (isClosed_tsupport _)
    exact closure_minimal (support_subset_iff'.2 fun z hz => hgs _ _ hp hz) hk.isClosed
  have I2 : Integrable (fun a : G => L (f a) (g q₀.1 (q₀.2 - a))) μ := by
    have M : HasCompactSupport (g q₀.1) := HasCompactSupport.intro hk fun x hx => hgs q₀.1 x hq₀ hx
    apply M.convolutionExists_right L hf (A q₀.1 hq₀) q₀.2
  have I3 : AEStronglyMeasurable (fun a : G => (L (f a)).comp (g' (q₀.fst, q₀.snd - a))) μ := by
    have T : HasCompactSupport fun y => g' (q₀.1, y) :=
      HasCompactSupport.intro hk fun x hx => g'_zero q₀.1 x hq₀ hx
    apply (HasCompactSupport.convolutionExists_right (L.precompR (P × G) : _) T hf _ q₀.2).1
    have : ContinuousOn g' (s ×ˢ univ) :=
      hg.continuousOn_fderiv_of_isOpen (hs.prod isOpen_univ) le_rfl
    apply this.comp_continuous (continuous_const.prod_mk continuous_id')
    intro x
    simpa only [prod_mk_mem_set_prod_eq, mem_univ, and_true] using hq₀
  /-
    case intro.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 1 (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    q₀ : Prod P G
    hq₀ : Membership.mem s q₀.1
    g' : Prod P G → ContinuousLinearMap (RingHom.id 𝕜) (Prod P G) E' := fderiv 𝕜 ( …
    A : ∀ (p : P), Membership.mem s p → Continuous (g p)
    A' : ∀ (q : Prod P G), Membership.mem s q.1 → Membership.mem (nhds q) (SProd.s …
    g'_zero : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → E …
    ε C : Real
    εpos : LT.lt 0 ε
    h₀ε : HasSubset.Subset (Metric.ball q₀.1 ε) s
    hε : ∀ (p : P) (x : G), LT.lt (Norm.norm (HSub.hSub p q₀.1)) ε → LE.le (Norm.n …
    I1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun a =>  …
    I2 : MeasureTheory.Integrable (fun a => (L (f a)) (g q₀.1 (HSub.hSub q₀.2 a))) μ
    I3 : MeasureTheory.AEStronglyMeasurable (fun a => (L (f a)).comp (g' { fst :=  …
    ⊢ HasFDerivAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (MeasureT …
  -/
  set K' := (-k + {q₀.2} : Set G) with K'_def
  /-
    case intro.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 1 (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    q₀ : Prod P G
    hq₀ : Membership.mem s q₀.1
    g' : Prod P G → ContinuousLinearMap (RingHom.id 𝕜) (Prod P G) E' := fderiv 𝕜 ( …
    A : ∀ (p : P), Membership.mem s p → Continuous (g p)
    A' : ∀ (q : Prod P G), Membership.mem s q.1 → Membership.mem (nhds q) (SProd.s …
    g'_zero : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → E …
    ε C : Real
    εpos : LT.lt 0 ε
    h₀ε : HasSubset.Subset (Metric.ball q₀.1 ε) s
    hε : ∀ (p : P) (x : G), LT.lt (Norm.norm (HSub.hSub p q₀.1)) ε → LE.le (Norm.n …
    I1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun a =>  …
    I2 : MeasureTheory.Integrable (fun a => (L (f a)) (g q₀.1 (HSub.hSub q₀.2 a))) μ
    I3 : MeasureTheory.AEStronglyMeasurable (fun a => (L (f a)).comp (g' { fst :=  …
    K' : Set G := HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2)
    K'_def : Eq K' (HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2))
    ⊢ HasFDerivAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (MeasureT …
  -/
  have hK' : IsCompact K' := hk.neg.add isCompact_singleton
  obtain ⟨U, U_open, K'U, hU⟩ : ∃ U, IsOpen U ∧ K' ⊆ U ∧ IntegrableOn f U μ :=
    hf.integrableOn_nhds_isCompact hK'
  obtain ⟨δ, δpos, δε, hδ⟩ : ∃ δ, (0 : ℝ) < δ ∧ δ ≤ ε ∧ K' + ball 0 δ ⊆ U := by
    obtain ⟨V, V_mem, hV⟩ : ∃ V ∈ 𝓝 (0 : G), K' + V ⊆ U :=
      compact_open_separated_add_right hK' U_open K'U
    rcases Metric.mem_nhds_iff.1 V_mem with ⟨δ, δpos, hδ⟩
    refine ⟨min δ ε, lt_min δpos εpos, min_le_right δ ε, ?_⟩
    exact (add_subset_add_left ((ball_subset_ball (min_le_left _ _)).trans hδ)).trans hV
  -- Porting note: added to speed up the line below.
  letI := ContinuousLinearMap.hasOpNorm (𝕜 := 𝕜) (𝕜₂ := 𝕜) (E := E)
    (F := (P × G →L[𝕜] E') →L[𝕜] P × G →L[𝕜] F) (σ₁₂ := RingHom.id 𝕜)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 1 (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    q₀ : Prod P G
    hq₀ : Membership.mem s q₀.1
    g' : Prod P G → ContinuousLinearMap (RingHom.id 𝕜) (Prod P G) E' := fderiv 𝕜 ( …
    A : ∀ (p : P), Membership.mem s p → Continuous (g p)
    A' : ∀ (q : Prod P G), Membership.mem s q.1 → Membership.mem (nhds q) (SProd.s …
    g'_zero : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → E …
    ε C : Real
    εpos : LT.lt 0 ε
    h₀ε : HasSubset.Subset (Metric.ball q₀.1 ε) s
    hε : ∀ (p : P) (x : G), LT.lt (Norm.norm (HSub.hSub p q₀.1)) ε → LE.le (Norm.n …
    I1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun a =>  …
    I2 : MeasureTheory.Integrable (fun a => (L (f a)) (g q₀.1 (HSub.hSub q₀.2 a))) μ
    I3 : MeasureTheory.AEStronglyMeasurable (fun a => (L (f a)).comp (g' { fst :=  …
    K' : Set G := HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2)
    K'_def : Eq K' (HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2))
    hK' : IsCompact K'
    U : Set G
    U_open : IsOpen U
    K'U : HasSubset.Subset K' U
    hU : MeasureTheory.IntegrableOn f U μ
    δ : Real
    δpos : LT.lt 0 δ
    δε : LE.le δ ε
    hδ : HasSubset.Subset (HAdd.hAdd K' (Metric.ball 0 δ)) U
    this : Norm (ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHo …
    ⊢ HasFDerivAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (MeasureT …
  -/
  let bound : G → ℝ := indicator U fun t => ‖(L.precompR (P × G))‖ * ‖f t‖ * C
  have I4 : ∀ᵐ a : G ∂μ, ∀ x : P × G, dist x q₀ < δ →
      ‖L.precompR (P × G) (f a) (g' (x.fst, x.snd - a))‖ ≤ bound a := by
    filter_upwards with a x hx
    rw [Prod.dist_eq, dist_eq_norm, dist_eq_norm] at hx
    have : (-tsupport fun a => g' (x.1, a)) + ball q₀.2 δ ⊆ U := by
      apply Subset.trans _ hδ
      rw [K'_def, add_assoc]
      apply add_subset_add
      · rw [neg_subset_neg]
        refine closure_minimal (support_subset_iff'.2 fun z hz => ?_) hk.isClosed
        apply g'_zero x.1 z (h₀ε _) hz
        rw [mem_ball_iff_norm]
        exact ((le_max_left _ _).trans_lt hx).trans_le δε
      · simp only [add_ball, thickening_singleton, zero_vadd, subset_rfl]
    apply convolution_integrand_bound_right_of_le_of_subset _ _ _ this
    · intro y
      exact hε _ _ (((le_max_left _ _).trans_lt hx).trans_le δε)
    · rw [mem_ball_iff_norm]
      exact (le_max_right _ _).trans_lt hx
  have I5 : Integrable bound μ := by
    rw [integrable_indicator_iff U_open.measurableSet]
    exact (hU.norm.const_mul _).mul_const _
  have I6 : ∀ᵐ a : G ∂μ, ∀ x : P × G, dist x q₀ < δ →
      HasFDerivAt (fun x : P × G => L (f a) (g x.1 (x.2 - a)))
        ((L (f a)).comp (g' (x.fst, x.snd - a))) x := by
    filter_upwards with a x hx
    apply (L _).hasFDerivAt.comp x
    have N : s ×ˢ univ ∈ 𝓝 (x.1, x.2 - a) := by
      apply A'
      apply h₀ε
      rw [Prod.dist_eq] at hx
      exact lt_of_lt_of_le (lt_of_le_of_lt (le_max_left _ _) hx) δε
    have Z := ((hg.differentiableOn le_rfl).differentiableAt N).hasFDerivAt
    have Z' :
        HasFDerivAt (fun x : P × G => (x.1, x.2 - a)) (ContinuousLinearMap.id 𝕜 (P × G)) x := by
      have : (fun x : P × G => (x.1, x.2 - a)) = _root_.id - fun x => (0, a) := by
        ext x <;> simp only [Pi.sub_apply, _root_.id, Prod.fst_sub, sub_zero, Prod.snd_sub]
      rw [this]
      exact (hasFDerivAt_id x).sub_const (0, a)
    exact Z.comp x Z'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 1 (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    q₀ : Prod P G
    hq₀ : Membership.mem s q₀.1
    g' : Prod P G → ContinuousLinearMap (RingHom.id 𝕜) (Prod P G) E' := fderiv 𝕜 ( …
    A : ∀ (p : P), Membership.mem s p → Continuous (g p)
    A' : ∀ (q : Prod P G), Membership.mem s q.1 → Membership.mem (nhds q) (SProd.s …
    g'_zero : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → E …
    ε C : Real
    εpos : LT.lt 0 ε
    h₀ε : HasSubset.Subset (Metric.ball q₀.1 ε) s
    hε : ∀ (p : P) (x : G), LT.lt (Norm.norm (HSub.hSub p q₀.1)) ε → LE.le (Norm.n …
    I1 : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun a =>  …
    I2 : MeasureTheory.Integrable (fun a => (L (f a)) (g q₀.1 (HSub.hSub q₀.2 a))) μ
    I3 : MeasureTheory.AEStronglyMeasurable (fun a => (L (f a)).comp (g' { fst :=  …
    K' : Set G := HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2)
    K'_def : Eq K' (HAdd.hAdd (Neg.neg k) (Singleton.singleton q₀.2))
    hK' : IsCompact K'
    U : Set G
    U_open : IsOpen U
    K'U : HasSubset.Subset K' U
    hU : MeasureTheory.IntegrableOn f U μ
    δ : Real
    δpos : LT.lt 0 δ
    δε : LE.le δ ε
    hδ : HasSubset.Subset (HAdd.hAdd K' (Metric.ball 0 δ)) U
    this : Norm (ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHo …
    bound : G → Real := U.indicator fun t => HMul.hMul (HMul.hMul (Norm.norm (Cont …
    I4 : Filter.Eventually (fun a => ∀ (x : Prod P G), LT.lt (Dist.dist x q₀) δ →  …
    I5 : MeasureTheory.Integrable bound μ
    I6 : Filter.Eventually (fun a => ∀ (x : Prod P G), LT.lt (Dist.dist x q₀) δ →  …
    ⊢ HasFDerivAt (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (MeasureT …
  -/
  exact hasFDerivAt_integral_of_dominated_of_fderiv_le δpos I1 I2 I3 I4 I5 I6
  /-
    🎉 no goals
  -/


/-- The convolution `f * g` is `C^n` when `f` is locally integrable and `g` is `C^n` and compactly
supported. Version where `g` depends on an additional parameter in an open subset `s` of a
parameter space `P` (and the compact support `k` is independent of the parameter in `s`).
In this version, all the types belong to the same universe (to get an induction working in the
proof). Use instead `contDiffOn_convolution_right_with_param`, which removes this restriction. -/
theorem contDiffOn_convolution_right_with_param_aux {G : Type uP} {E' : Type uP} {F : Type uP}
    {P : Type uP} [NormedAddCommGroup E'] [NormedAddCommGroup F] [NormedSpace 𝕜 E']
    [NormedSpace ℝ F] [NormedSpace 𝕜 F] [MeasurableSpace G]
    {μ : Measure G}
    [NormedAddCommGroup G] [BorelSpace G] [NormedSpace 𝕜 G] [NormedAddCommGroup P] [NormedSpace 𝕜 P]
    {f : G → E} {n : ℕ∞} (L : E →L[𝕜] E' →L[𝕜] F) {g : P → G → E'} {s : Set P} {k : Set G}
    (hs : IsOpen s) (hk : IsCompact k) (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0)
    (hf : LocallyIntegrable f μ) (hg : ContDiffOn 𝕜 n (↿g) (s ×ˢ univ)) :
    ContDiffOn 𝕜 n (fun q : P × G => (f ⋆[L, μ] g q.1) q.2) (s ×ˢ univ) := by
  /- We have a formula for the derivation of `f * g`, which is of the same form, thanks to
    `hasFDerivAt_convolution_right_with_param`. Therefore, we can prove the result by induction on
    `n` (but for this we need the spaces at the different steps of the induction to live in the same
    universe, which is why we make the assumption in the lemma that all the relevant spaces
    come from the same universe). -/
  induction n using ENat.nat_induction generalizing g E' F with
  | h0 =>
    rw [WithTop.coe_zero, contDiffOn_zero] at hg ⊢
    exact continuousOn_convolution_right_with_param L hk hgs hf hg
  | hsuc n ih =>
    simp only [Nat.succ_eq_add_one, Nat.cast_add, Nat.cast_one, WithTop.coe_add,
      WithTop.coe_natCast, WithTop.coe_one] at hg ⊢
    let f' : P → G → P × G →L[𝕜] F := fun p a =>
      (f ⋆[L.precompR (P × G), μ] fun x : G => fderiv 𝕜 (uncurry g) (p, x)) a
    have A : ∀ q₀ : P × G, q₀.1 ∈ s →
        HasFDerivAt (fun q : P × G => (f ⋆[L, μ] g q.1) q.2) (f' q₀.1 q₀.2) q₀ :=
      hasFDerivAt_convolution_right_with_param L hs hk hgs hf hg.one_of_succ
    rw [contDiffOn_succ_iff_fderiv_of_isOpen (hs.prod (@isOpen_univ G _))] at hg ⊢
    refine ⟨?_, by simp, ?_⟩
    · rintro ⟨p, x⟩ ⟨hp, -⟩
      exact (A (p, x) hp).differentiableAt.differentiableWithinAt
    · suffices H : ContDiffOn 𝕜 n (↿f') (s ×ˢ univ) by
        apply H.congr
        rintro ⟨p, x⟩ ⟨hp, -⟩
        exact (A (p, x) hp).fderiv
      have B : ∀ (p : P) (x : G), p ∈ s → x ∉ k → fderiv 𝕜 (uncurry g) (p, x) = 0 := by
        intro p x hp hx
        apply (hasFDerivAt_zero_of_eventually_const (0 : E') _).fderiv
        have M2 : kᶜ ∈ 𝓝 x := IsOpen.mem_nhds hk.isClosed.isOpen_compl hx
        have M1 : s ∈ 𝓝 p := hs.mem_nhds hp
        rw [nhds_prod_eq]
        filter_upwards [prod_mem_prod M1 M2]
        rintro ⟨p, y⟩ ⟨hp, hy⟩
        exact hgs p y hp hy
      apply ih (L.precompR (P × G) : _) B
      convert hg.2.2
  | htop ih =>
    rw [contDiffOn_infty] at hg ⊢
    exact fun n ↦ ih n L hgs (hg n)


/-- The convolution `f * g` is `C^n` when `f` is locally integrable and `g` is `C^n` and compactly
supported. Version where `g` depends on an additional parameter in an open subset `s` of a
parameter space `P` (and the compact support `k` is independent of the parameter in `s`). -/
theorem contDiffOn_convolution_right_with_param {f : G → E} {n : ℕ∞} (L : E →L[𝕜] E' →L[𝕜] F)
    {g : P → G → E'} {s : Set P} {k : Set G} (hs : IsOpen s) (hk : IsCompact k)
    (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0) (hf : LocallyIntegrable f μ)
    (hg : ContDiffOn 𝕜 n (↿g) (s ×ˢ univ)) :
    ContDiffOn 𝕜 n (fun q : P × G => (f ⋆[L, μ] g q.1) q.2) (s ×ˢ univ) := by
  /- The result is known when all the universes are the same, from
    `contDiffOn_convolution_right_with_param_aux`. We reduce to this situation by pushing
    everything through `ULift` continuous linear equivalences. -/
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eG : Type max uG uE' uF uP := ULift.{max uE' uF uP} G
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  borelize eG
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eE' : Type max uE' uG uF uP := ULift.{max uG uF uP} E'
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eF : Type max uF uG uE' uP := ULift.{max uG uE' uP} F
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eP : Type max uP uG uE' uF := ULift.{max uG uE' uF} P
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let isoG : eG ≃L[𝕜] G := ContinuousLinearEquiv.ulift
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let isoE' : eE' ≃L[𝕜] E' := ContinuousLinearEquiv.ulift
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let isoF : eF ≃L[𝕜] F := ContinuousLinearEquiv.ulift
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let isoP : eP ≃L[𝕜] P := ContinuousLinearEquiv.ulift
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let ef := f ∘ isoG
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ef : eG → E := Function.comp f ⇑isoG
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eμ : Measure eG := Measure.map isoG.symm μ
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ef : eG → E := Function.comp f ⇑isoG
    eμ : MeasureTheory.Measure eG := MeasureTheory.Measure.map (⇑isoG.symm) μ
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let eg : eP → eG → eE' := fun ep ex => isoE'.symm (g (isoP ep) (isoG ex))
  let eL :=
    ContinuousLinearMap.comp
      ((ContinuousLinearEquiv.arrowCongr isoE' isoF).symm : (E' →L[𝕜] F) →L[𝕜] eE' →L[𝕜] eF) L
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ef : eG → E := Function.comp f ⇑isoG
    eμ : MeasureTheory.Measure eG := MeasureTheory.Measure.map (⇑isoG.symm) μ
    eg : eP → eG → eE' := fun ep ex => isoE'.symm (g (isoP ep) (isoG ex))
    eL : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  let R := fun q : eP × eG => (ef ⋆[eL, eμ] eg q.1) q.2
  have R_contdiff : ContDiffOn 𝕜 n R ((isoP ⁻¹' s) ×ˢ univ) := by
    have hek : IsCompact (isoG ⁻¹' k) := isoG.toHomeomorph.isClosedEmbedding.isCompact_preimage hk
    have hes : IsOpen (isoP ⁻¹' s) := isoP.continuous.isOpen_preimage _ hs
    refine contDiffOn_convolution_right_with_param_aux eL hes hek ?_ ?_ ?_
    · intro p x hp hx
      simp only [eg, (· ∘ ·), ContinuousLinearEquiv.prod_apply, LinearIsometryEquiv.coe_coe,
        ContinuousLinearEquiv.map_eq_zero_iff]
      exact hgs _ _ hp hx
    · exact (locallyIntegrable_map_homeomorph isoG.symm.toHomeomorph).2 hf
    · apply isoE'.symm.contDiff.comp_contDiffOn
      apply hg.comp (isoP.prod isoG).contDiff.contDiffOn
      rintro ⟨p, x⟩ ⟨hp, -⟩
      simpa only [mem_preimage, ContinuousLinearEquiv.prod_apply, prod_mk_mem_set_prod_eq, mem_univ,
        and_true] using hp
  have A : ContDiffOn 𝕜 n (isoF ∘ R ∘ (isoP.prod isoG).symm) (s ×ˢ univ) := by
    apply isoF.contDiff.comp_contDiffOn
    apply R_contdiff.comp (ContinuousLinearEquiv.contDiff _).contDiffOn
    rintro ⟨p, x⟩ ⟨hp, -⟩
    simpa only [mem_preimage, mem_prod, mem_univ, and_true, ContinuousLinearEquiv.prod_symm,
      ContinuousLinearEquiv.prod_apply, ContinuousLinearEquiv.apply_symm_apply] using hp
  have : isoF ∘ R ∘ (isoP.prod isoG).symm = fun q : P × G => (f ⋆[L, μ] g q.1) q.2 := by
    apply funext
    rintro ⟨p, x⟩
    simp only [LinearIsometryEquiv.coe_coe, (· ∘ ·), ContinuousLinearEquiv.prod_symm,
      ContinuousLinearEquiv.prod_apply]
    simp only [R, convolution, coe_comp', ContinuousLinearEquiv.coe_coe, (· ∘ ·)]
    rw [IsClosedEmbedding.integral_map, ← isoF.integral_comp_comm]
    · rfl
    · exact isoG.symm.toHomeomorph.isClosedEmbedding
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ef : eG → E := Function.comp f ⇑isoG
    eμ : MeasureTheory.Measure eG := MeasureTheory.Measure.map (⇑isoG.symm) μ
    eg : eP → eG → eE' := fun ep ex => isoE'.symm (g (isoP ep) (isoG ex))
    eL : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    R : Prod eP eG → eF := fun q => MeasureTheory.convolution ef (eg q.1) eL eμ q.2
    R_contdiff : ContDiffOn 𝕜 (↑n) R (SProd.sprod (Set.preimage (⇑isoP) s) Set.univ)
    A : ContDiffOn 𝕜 (↑n) (Function.comp (⇑isoF) (Function.comp R ⇑(isoP.prod isoG …
    this : Eq (Function.comp (⇑isoF) (Function.comp R ⇑(isoP.prod isoG).symm)) fun …
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  simp_rw [this] at A
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    f : G → E
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    eG : Type (max uG uE' uF uP) := ULift.{max uE' uF uP, uG} G
    this✝¹ : MeasurableSpace eG := borel eG
    this✝ : BorelSpace eG
    eE' : Type (max uE' uG uF uP) := ULift.{max uG uF uP, uE'} E'
    eF : Type (max uF uG uE' uP) := ULift.{max uG uE' uP, uF} F
    eP : Type (max uP uG uE' uF) := ULift.{max uG uE' uF, uP} P
    isoG : ContinuousLinearEquiv (RingHom.id 𝕜) eG G := ContinuousLinearEquiv.ulift
    isoE' : ContinuousLinearEquiv (RingHom.id 𝕜) eE' E' := ContinuousLinearEquiv.u …
    isoF : ContinuousLinearEquiv (RingHom.id 𝕜) eF F := ContinuousLinearEquiv.ulift
    isoP : ContinuousLinearEquiv (RingHom.id 𝕜) eP P := ContinuousLinearEquiv.ulift
    ef : eG → E := Function.comp f ⇑isoG
    eμ : MeasureTheory.Measure eG := MeasureTheory.Measure.map (⇑isoG.symm) μ
    eg : eP → eG → eE' := fun ep ex => isoE'.symm (g (isoP ep) (isoG ex))
    eL : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    R : Prod eP eG → eF := fun q => MeasureTheory.convolution ef (eg q.1) eL eμ q.2
    R_contdiff : ContDiffOn 𝕜 (↑n) R (SProd.sprod (Set.preimage (⇑isoP) s) Set.univ)
    this : Eq (Function.comp (⇑isoF) (Function.comp R ⇑(isoP.prod isoG).symm)) fun …
    A : ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) ( …
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution f (g q.1) L μ q.2) (SP …
  -/
  exact A
  /-
    🎉 no goals
  -/


/-- The convolution `f * g` is `C^n` when `f` is locally integrable and `g` is `C^n` and compactly
supported. Version where `g` depends on an additional parameter in an open subset `s` of a
parameter space `P` (and the compact support `k` is independent of the parameter in `s`),
given in terms of composition with an additional smooth function. -/
theorem contDiffOn_convolution_right_with_param_comp {n : ℕ∞} (L : E →L[𝕜] E' →L[𝕜] F) {s : Set P}
    {v : P → G} (hv : ContDiffOn 𝕜 n v s) {f : G → E} {g : P → G → E'} {k : Set G} (hs : IsOpen s)
    (hk : IsCompact k) (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0) (hf : LocallyIntegrable f μ)
    (hg : ContDiffOn 𝕜 n (↿g) (s ×ˢ univ)) : ContDiffOn 𝕜 n (fun x => (f ⋆[L, μ] g x) (v x)) s := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    s : Set P
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ ContDiffOn 𝕜 (↑n) (fun x => MeasureTheory.convolution f (g x) L μ (v x)) s
  -/
  apply (contDiffOn_convolution_right_with_param L hs hk hgs hf hg).comp (contDiffOn_id.prod hv)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    s : Set P
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ Set.MapsTo (fun x => { fst := id x, snd := v x }) s (SProd.sprod s Set.univ)
  -/
  intro x hx
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup P
    inst✝ : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    n : ENat
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    s : Set P
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    x : P
    hx : Membership.mem s x
    ⊢ Membership.mem (SProd.sprod s Set.univ) ((fun x => { fst := id x, snd := v x …
  -/
  simp only [hx, mem_preimage, prod_mk_mem_set_prod_eq, mem_univ, and_self_iff, _root_.id]
  /-
    🎉 no goals
  -/


/-- The convolution `g * f` is `C^n` when `f` is locally integrable and `g` is `C^n` and compactly
supported. Version where `g` depends on an additional parameter in an open subset `s` of a
parameter space `P` (and the compact support `k` is independent of the parameter in `s`). -/
theorem contDiffOn_convolution_left_with_param [μ.IsAddLeftInvariant] [μ.IsNegInvariant]
    (L : E' →L[𝕜] E →L[𝕜] F) {f : G → E} {n : ℕ∞} {g : P → G → E'} {s : Set P} {k : Set G}
    (hs : IsOpen s) (hk : IsCompact k) (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0)
    (hf : LocallyIntegrable f μ) (hg : ContDiffOn 𝕜 n (↿g) (s ×ˢ univ)) :
    ContDiffOn 𝕜 n (fun q : P × G => (g q.1 ⋆[L, μ] f) q.2) (s ×ˢ univ) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹⁵ : NormedAddCommGroup E
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedAddCommGroup F
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : NormedSpace 𝕜 E
    inst✝¹⁰ : NormedSpace 𝕜 E'
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : NormedAddCommGroup P
    inst✝² : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    L : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜)  …
    f : G → E
    n : ENat
    g : P → G → E'
    s : Set P
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ ContDiffOn 𝕜 (↑n) (fun q => MeasureTheory.convolution (g q.1) f L μ q.2) (SP …
  -/
  simpa only [convolution_flip] using contDiffOn_convolution_right_with_param L.flip hs hk hgs hf hg
  /-
    🎉 no goals
  -/


/-- The convolution `g * f` is `C^n` when `f` is locally integrable and `g` is `C^n` and compactly
supported. Version where `g` depends on an additional parameter in an open subset `s` of a
parameter space `P` (and the compact support `k` is independent of the parameter in `s`),
given in terms of composition with additional smooth functions. -/
theorem contDiffOn_convolution_left_with_param_comp [μ.IsAddLeftInvariant] [μ.IsNegInvariant]
    (L : E' →L[𝕜] E →L[𝕜] F) {s : Set P} {n : ℕ∞} {v : P → G} (hv : ContDiffOn 𝕜 n v s) {f : G → E}
    {g : P → G → E'} {k : Set G} (hs : IsOpen s) (hk : IsCompact k)
    (hgs : ∀ p, ∀ x, p ∈ s → x ∉ k → g p x = 0) (hf : LocallyIntegrable f μ)
    (hg : ContDiffOn 𝕜 n (↿g) (s ×ˢ univ)) : ContDiffOn 𝕜 n (fun x => (g x ⋆[L, μ] f) (v x)) s := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹⁵ : NormedAddCommGroup E
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedAddCommGroup F
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : NormedSpace 𝕜 E
    inst✝¹⁰ : NormedSpace 𝕜 E'
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : NormedAddCommGroup P
    inst✝² : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    L : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜)  …
    s : Set P
    n : ENat
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ ContDiffOn 𝕜 (↑n) (fun x => MeasureTheory.convolution (g x) f L μ (v x)) s
  -/
  apply (contDiffOn_convolution_left_with_param L hs hk hgs hf hg).comp (contDiffOn_id.prod hv)
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹⁵ : NormedAddCommGroup E
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedAddCommGroup F
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : NormedSpace 𝕜 E
    inst✝¹⁰ : NormedSpace 𝕜 E'
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : NormedAddCommGroup P
    inst✝² : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    L : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜)  …
    s : Set P
    n : ENat
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    ⊢ Set.MapsTo (fun x => { fst := id x, snd := v x }) s (SProd.sprod s Set.univ)
  -/
  intro x hx
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    P : Type uP
    inst✝¹⁵ : NormedAddCommGroup E
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedAddCommGroup F
    inst✝¹² : RCLike 𝕜
    inst✝¹¹ : NormedSpace 𝕜 E
    inst✝¹⁰ : NormedSpace 𝕜 E'
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup G
    inst✝⁵ : BorelSpace G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : NormedAddCommGroup P
    inst✝² : NormedSpace 𝕜 P
    μ : MeasureTheory.Measure G
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    L : ContinuousLinearMap (RingHom.id 𝕜) E' (ContinuousLinearMap (RingHom.id 𝕜)  …
    s : Set P
    n : ENat
    v : P → G
    hv : ContDiffOn 𝕜 (↑n) v s
    f : G → E
    g : P → G → E'
    k : Set G
    hs : IsOpen s
    hk : IsCompact k
    hgs : ∀ (p : P) (x : G), Membership.mem s p → Not (Membership.mem k x) → Eq (g …
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiffOn 𝕜 (↑n) (Function.HasUncurry.uncurry g) (SProd.sprod s Set.univ)
    x : P
    hx : Membership.mem s x
    ⊢ Membership.mem (SProd.sprod s Set.univ) ((fun x => { fst := id x, snd := v x …
  -/
  simp only [hx, mem_preimage, prod_mk_mem_set_prod_eq, mem_univ, and_self_iff, _root_.id]
  /-
    🎉 no goals
  -/


theorem _root_.HasCompactSupport.contDiff_convolution_right {n : ℕ∞} (hcg : HasCompactSupport g)
    (hf : LocallyIntegrable f μ) (hg : ContDiff 𝕜 n g) : ContDiff 𝕜 n (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : MeasurableSpace G
    inst✝² : NormedAddCommGroup G
    inst✝¹ : BorelSpace G
    inst✝ : NormedSpace 𝕜 G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    n : ENat
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 (↑n) g
    ⊢ ContDiff 𝕜 (↑n) (MeasureTheory.convolution f g L μ)
  -/
  rcases exists_compact_iff_hasCompactSupport.2 hcg with ⟨k, hk, h'k⟩
  /-
    case intro.intro
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedSpace 𝕜 E'
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : MeasurableSpace G
    inst✝² : NormedAddCommGroup G
    inst✝¹ : BorelSpace G
    inst✝ : NormedSpace 𝕜 G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    n : ENat
    hcg : HasCompactSupport g
    hf : MeasureTheory.LocallyIntegrable f μ
    hg : ContDiff 𝕜 (↑n) g
    k : Set G
    hk : IsCompact k
    h'k : ∀ (x : G), Not (Membership.mem k x) → Eq (g x) 0
    ⊢ ContDiff 𝕜 (↑n) (MeasureTheory.convolution f g L μ)
  -/
  rw [← contDiffOn_univ]
  exact contDiffOn_convolution_right_with_param_comp L contDiffOn_id isOpen_univ hk
    (fun p x _ hx => h'k x hx) hf (hg.comp contDiff_snd).contDiffOn


theorem _root_.HasCompactSupport.contDiff_convolution_left [μ.IsAddLeftInvariant] [μ.IsNegInvariant]
    {n : ℕ∞} (hcf : HasCompactSupport f) (hf : ContDiff 𝕜 n f) (hg : LocallyIntegrable g μ) :
    ContDiff 𝕜 n (f ⋆[L, μ] g) := by
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    n : ENat
    hcf : HasCompactSupport f
    hf : ContDiff 𝕜 (↑n) f
    hg : MeasureTheory.LocallyIntegrable g μ
    ⊢ ContDiff 𝕜 (↑n) (MeasureTheory.convolution f g L μ)
  -/
  rw [← convolution_flip]
  /-
    𝕜 : Type u𝕜
    G : Type uG
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedAddCommGroup F
    f : G → E
    g : G → E'
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : NormedSpace 𝕜 E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : BorelSpace G
    inst✝² : NormedSpace 𝕜 G
    μ : MeasureTheory.Measure G
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) E …
    inst✝¹ : μ.IsAddLeftInvariant
    inst✝ : μ.IsNegInvariant
    n : ENat
    hcf : HasCompactSupport f
    hf : ContDiff 𝕜 (↑n) f
    hg : MeasureTheory.LocallyIntegrable g μ
    ⊢ ContDiff 𝕜 (↑n) (MeasureTheory.convolution g f L.flip μ)
  -/
  exact hcf.contDiff_convolution_right L.flip hg hf
  /-
    🎉 no goals
  -/


/-- The forward convolution of two functions `f` and `g` on `ℝ`, with respect to a continuous
bilinear map `L` and measure `ν`. It is defined to be the function mapping `x` to
`∫ t in 0..x, L (f t) (g (x - t)) ∂ν` if `0 < x`, and 0 otherwise. -/
noncomputable def posConvolution (f : ℝ → E) (g : ℝ → E') (L : E →L[ℝ] E' →L[ℝ] F)
    (ν : Measure ℝ := by volume_tac) : ℝ → F :=
  indicator (Ioi (0 : ℝ)) fun x => ∫ t in (0)..x, L (f t) (g (x - t)) ∂ν


theorem posConvolution_eq_convolution_indicator (f : ℝ → E) (g : ℝ → E') (L : E →L[ℝ] E' →L[ℝ] F)
    (ν : Measure ℝ := by volume_tac) [NoAtoms ν] :
    posConvolution f g L ν = convolution (indicator (Ioi 0) f) (indicator (Ioi 0) g) L ν := by
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ν : autoParam (MeasureTheory.Measure Real) _auto✝
    inst✝ : MeasureTheory.NoAtoms ν
    ⊢ Eq (MeasureTheory.posConvolution f g L ν) (MeasureTheory.convolution ((Set.I …
  -/
  ext1 x
  -- Porting note: was `rw [convolution, posConvolution, indicator]`, now `rw` can't do it
  -- the `rw` unfolded only one `indicator`; now we unfold it everywhere, so we need to adjust
  -- `rw`s below
  /-
    case h
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ν : autoParam (MeasureTheory.Measure Real) _auto✝
    inst✝ : MeasureTheory.NoAtoms ν
    x : Real
    ⊢ Eq (MeasureTheory.posConvolution f g L ν x) (MeasureTheory.convolution ((Set …
  -/
  unfold convolution posConvolution indicator; simp only
  /-
    case h
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real E'
    inst✝¹ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ν : autoParam (MeasureTheory.Measure Real) _auto✝
    inst✝ : MeasureTheory.NoAtoms ν
    x : Real
    ⊢ Eq (ite (Membership.mem (Set.Ioi 0) x) (intervalIntegral (fun t => (L (f t)) …
  -/
  split_ifs with h
  · rw [intervalIntegral.integral_of_le (le_of_lt h), integral_Ioc_eq_integral_Ioo, ←
      integral_indicator (measurableSet_Ioo : MeasurableSet (Ioo 0 x))]
    /-
      case pos
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace Real E'
      inst✝¹ : NormedSpace Real F
      f : Real → E
      g : Real → E'
      L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
      ν : autoParam (MeasureTheory.Measure Real) _auto✝
      inst✝ : MeasureTheory.NoAtoms ν
      x : Real
      h : Membership.mem (Set.Ioi 0) x
      ⊢ Eq (MeasureTheory.integral ν fun x_1 => (Set.Ioo 0 x).indicator (fun t => (L …
    -/
    congr 1 with t : 1
    have : t ≤ 0 ∨ t ∈ Ioo 0 x ∨ x ≤ t := by
      rcases le_or_lt t 0 with (h | h)
      · exact Or.inl h
      · rcases lt_or_le t x with (h' | h')
        exacts [Or.inr (Or.inl ⟨h, h'⟩), Or.inr (Or.inr h')]
    /-
      case pos.e_f.h
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace Real E'
      inst✝¹ : NormedSpace Real F
      f : Real → E
      g : Real → E'
      L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
      ν : autoParam (MeasureTheory.Measure Real) _auto✝
      inst✝ : MeasureTheory.NoAtoms ν
      x : Real
      h : Membership.mem (Set.Ioi 0) x
      t : Real
      this : Or (LE.le t 0) (Or (Membership.mem (Set.Ioo 0 x) t) (LE.le x t))
      ⊢ Eq ((Set.Ioo 0 x).indicator (fun t => (L (f t)) (g (HSub.hSub x t))) t) ((L  …
    -/
    rcases this with (ht | ht | ht)
    · -- Porting note: was
      -- rw [indicator_of_not_mem (not_mem_Ioo_of_le ht), indicator_of_not_mem (not_mem_Ioi.mpr ht),
      --   ContinuousLinearMap.map_zero, ContinuousLinearMap.zero_apply]
      rw [indicator_of_not_mem (not_mem_Ioo_of_le ht), if_neg (not_mem_Ioi.mpr ht),
        ContinuousLinearMap.map_zero, ContinuousLinearMap.zero_apply]
    · -- Porting note: was
      -- rw [indicator_of_mem ht, indicator_of_mem (mem_Ioi.mpr ht.1),
      --     indicator_of_mem (mem_Ioi.mpr <| sub_pos.mpr ht.2)]
      rw [indicator_of_mem ht, if_pos (mem_Ioi.mpr ht.1),
        if_pos (mem_Ioi.mpr <| sub_pos.mpr ht.2)]
    · -- Porting note: was
      -- rw [indicator_of_not_mem (not_mem_Ioo_of_ge ht),
      --     indicator_of_not_mem (not_mem_Ioi.mpr (sub_nonpos_of_le ht)),
      --     ContinuousLinearMap.map_zero]
      rw [indicator_of_not_mem (not_mem_Ioo_of_ge ht),
        if_neg (not_mem_Ioi.mpr (sub_nonpos_of_le ht)), ContinuousLinearMap.map_zero]
    /-
      case neg
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace Real E'
      inst✝¹ : NormedSpace Real F
      f : Real → E
      g : Real → E'
      L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
      ν : autoParam (MeasureTheory.Measure Real) _auto✝
      inst✝ : MeasureTheory.NoAtoms ν
      x : Real
      h : Not (Membership.mem (Set.Ioi 0) x)
      ⊢ Eq 0 (MeasureTheory.integral ν fun t => (L (ite (Membership.mem (Set.Ioi 0)  …
    -/
  · convert (integral_zero ℝ F).symm with t
    /-
      case h.e'_3.h.e'_7.h
      E : Type uE
      E' : Type uE'
      F : Type uF
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace Real E'
      inst✝¹ : NormedSpace Real F
      f : Real → E
      g : Real → E'
      L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
      ν : autoParam (MeasureTheory.Measure Real) _auto✝
      inst✝ : MeasureTheory.NoAtoms ν
      x : Real
      h : Not (Membership.mem (Set.Ioi 0) x)
      t : Real
      ⊢ Eq ((L (ite (Membership.mem (Set.Ioi 0) t) (f t) 0)) (ite (Membership.mem (S …
    -/
    by_cases ht : 0 < t
    · -- Porting note: was
      -- rw [indicator_of_not_mem (_ : x - t ∉ Ioi 0), ContinuousLinearMap.map_zero]
      /-
        case pos
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace Real E'
        inst✝¹ : NormedSpace Real F
        f : Real → E
        g : Real → E'
        L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
        ν : autoParam (MeasureTheory.Measure Real) _auto✝
        inst✝ : MeasureTheory.NoAtoms ν
        x : Real
        h : Not (Membership.mem (Set.Ioi 0) x)
        t : Real
        ht : LT.lt 0 t
        ⊢ Eq ((L (ite (Membership.mem (Set.Ioi 0) t) (f t) 0)) (ite (Membership.mem (S …
      -/
      rw [if_neg (_ : x - t ∉ Ioi 0), ContinuousLinearMap.map_zero]
      /-
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace Real E'
        inst✝¹ : NormedSpace Real F
        f : Real → E
        g : Real → E'
        L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
        ν : autoParam (MeasureTheory.Measure Real) _auto✝
        inst✝ : MeasureTheory.NoAtoms ν
        x : Real
        h : Not (Membership.mem (Set.Ioi 0) x)
        t : Real
        ht : LT.lt 0 t
        ⊢ Not (Membership.mem (Set.Ioi 0) (HSub.hSub x t))
      -/
      rw [not_mem_Ioi] at h ⊢
      /-
        E : Type uE
        E' : Type uE'
        F : Type uF
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace Real E'
        inst✝¹ : NormedSpace Real F
        f : Real → E
        g : Real → E'
        L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
        ν : autoParam (MeasureTheory.Measure Real) _auto✝
        inst✝ : MeasureTheory.NoAtoms ν
        x : Real
        h : LE.le x 0
        t : Real
        ht : LT.lt 0 t
        ⊢ LE.le (HSub.hSub x t) 0
      -/
      exact sub_nonpos.mpr (h.trans ht.le)
      /-
        🎉 no goals
      -/
    · -- Porting note: was
      -- rw [indicator_of_not_mem (mem_Ioi.not.mpr ht), ContinuousLinearMap.map_zero,
      --  ContinuousLinearMap.zero_apply]
      rw [if_neg (mem_Ioi.not.mpr ht), ContinuousLinearMap.map_zero,
        ContinuousLinearMap.zero_apply]


theorem integrable_posConvolution {f : ℝ → E} {g : ℝ → E'} {μ ν : Measure ℝ} [SFinite μ]
    [SFinite ν] [IsAddRightInvariant μ] [NoAtoms ν] (hf : IntegrableOn f (Ioi 0) ν)
    (hg : IntegrableOn g (Ioi 0) μ) (L : E →L[ℝ] E' →L[ℝ] F) :
    Integrable (posConvolution f g L ν) μ := by
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real E'
    inst✝⁴ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    hf : MeasureTheory.IntegrableOn f (Set.Ioi 0) ν
    hg : MeasureTheory.IntegrableOn g (Set.Ioi 0) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ MeasureTheory.Integrable (MeasureTheory.posConvolution f g L ν) μ
  -/
  rw [← integrable_indicator_iff (measurableSet_Ioi : MeasurableSet (Ioi (0 : ℝ)))] at hf hg
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real E'
    inst✝⁴ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    hf : MeasureTheory.Integrable ((Set.Ioi 0).indicator f) ν
    hg : MeasureTheory.Integrable ((Set.Ioi 0).indicator g) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ MeasureTheory.Integrable (MeasureTheory.posConvolution f g L ν) μ
  -/
  rw [posConvolution_eq_convolution_indicator f g L ν]
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup E'
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real E'
    inst✝⁴ : NormedSpace Real F
    f : Real → E
    g : Real → E'
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    hf : MeasureTheory.Integrable ((Set.Ioi 0).indicator f) ν
    hg : MeasureTheory.Integrable ((Set.Ioi 0).indicator g) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ MeasureTheory.Integrable (MeasureTheory.convolution ((Set.Ioi 0).indicator f …
  -/
  exact (hf.convolution_integrand L hg).integral_prod_left
  /-
    🎉 no goals
  -/


/-- The integral over `Ioi 0` of a forward convolution of two functions is equal to the product
of their integrals over this set. (Compare `integral_convolution` for the two-sided convolution.) -/
theorem integral_posConvolution [CompleteSpace E] [CompleteSpace E'] [CompleteSpace F]
    {μ ν : Measure ℝ}
    [SFinite μ] [SFinite ν] [IsAddRightInvariant μ] [NoAtoms ν] {f : ℝ → E} {g : ℝ → E'}
    (hf : IntegrableOn f (Ioi 0) ν) (hg : IntegrableOn g (Ioi 0) μ) (L : E →L[ℝ] E' →L[ℝ] F) :
    ∫ x : ℝ in Ioi 0, ∫ t : ℝ in (0)..x, L (f t) (g (x - t)) ∂ν ∂μ =
      L (∫ x : ℝ in Ioi 0, f x ∂ν) (∫ x : ℝ in Ioi 0, g x ∂μ) := by
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real E
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : CompleteSpace E
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : CompleteSpace F
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    f : Real → E
    g : Real → E'
    hf : MeasureTheory.IntegrableOn f (Set.Ioi 0) ν
    hg : MeasureTheory.IntegrableOn g (Set.Ioi 0) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.Ioi 0)) fun x => intervalIntegra …
  -/
  rw [← integrable_indicator_iff measurableSet_Ioi] at hf hg
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real E
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : CompleteSpace E
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : CompleteSpace F
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    f : Real → E
    g : Real → E'
    hf : MeasureTheory.Integrable ((Set.Ioi 0).indicator f) ν
    hg : MeasureTheory.Integrable ((Set.Ioi 0).indicator g) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.Ioi 0)) fun x => intervalIntegra …
  -/
  simp_rw [← integral_indicator measurableSet_Ioi]
  /-
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real E
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : CompleteSpace E
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : CompleteSpace F
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    f : Real → E
    g : Real → E'
    hf : MeasureTheory.Integrable ((Set.Ioi 0).indicator f) ν
    hg : MeasureTheory.Integrable ((Set.Ioi 0).indicator g) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    ⊢ Eq (MeasureTheory.integral μ fun x => (Set.Ioi 0).indicator (fun x => interv …
  -/
  convert integral_convolution L hf hg using 4 with x
  /-
    case h.e'_2.h.e'_7.h.h.e
    E : Type uE
    E' : Type uE'
    F : Type uF
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real E
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedSpace Real F
    inst✝⁶ : CompleteSpace E
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : CompleteSpace F
    μ ν : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : μ.IsAddRightInvariant
    inst✝ : MeasureTheory.NoAtoms ν
    f : Real → E
    g : Real → E'
    hf : MeasureTheory.Integrable ((Set.Ioi 0).indicator f) ν
    hg : MeasureTheory.Integrable ((Set.Ioi 0).indicator g) μ
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    x : Real
    ⊢ Eq ((Set.Ioi 0).indicator fun x => intervalIntegral (fun t => (L (f t)) (g ( …
  -/
  apply posConvolution_eq_convolution_indicator
  /-
    🎉 no goals
  -/


