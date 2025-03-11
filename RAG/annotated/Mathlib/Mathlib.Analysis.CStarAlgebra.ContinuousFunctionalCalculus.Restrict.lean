/-- The homeomorphism `spectrum S a ≃ₜ spectrum R a` induced by `SpectrumRestricts a f`. -/
def homeomorph {R S A : Type*} [Semifield R] [Semifield S] [Ring A]
    [Algebra R S] [Algebra R A] [Algebra S A] [IsScalarTower R S A] [TopologicalSpace R]
    [TopologicalSpace S] [ContinuousSMul R S] {a : A} {f : C(S, R)} (h : SpectrumRestricts a f) :
    spectrum S a ≃ₜ spectrum R a where
  toFun := MapsTo.restrict f _ _ h.subset_preimage
  invFun := MapsTo.restrict (algebraMap R S) _ _ (image_subset_iff.mp h.algebraMap_image.subset)
  left_inv x := Subtype.ext <| h.rightInvOn x.2
  right_inv x := Subtype.ext <| h.left_inv x
  continuous_toFun := continuous_induced_rng.mpr <| f.continuous.comp continuous_induced_dom
  continuous_invFun := continuous_induced_rng.mpr <|
    continuous_algebraMap R S |>.comp continuous_induced_dom


lemma compactSpace {R S A : Type*} [Semifield R] [Semifield S] [Ring A]
    [Algebra R S] [Algebra R A] [Algebra S A] [IsScalarTower R S A] [TopologicalSpace R]
    [TopologicalSpace S] {a : A} (f : C(S, R)) (h : SpectrumRestricts a f)
    [h_cpct : CompactSpace (spectrum S a)] : CompactSpace (spectrum R a) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : Semifield R
    inst✝⁷ : Semifield S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSpace S
    a : A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_cpct : CompactSpace ↑(spectrum S a)
    ⊢ CompactSpace ↑(spectrum R a)
  -/
  rw [← isCompact_iff_compactSpace] at h_cpct ⊢
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : Semifield R
    inst✝⁷ : Semifield S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSpace S
    a : A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_cpct : IsCompact (spectrum S a)
    ⊢ IsCompact (spectrum R a)
  -/
  exact h.image ▸ h_cpct.image (map_continuous f)
  /-
    🎉 no goals
  -/


/-- If the spectrum of an element restricts to a smaller scalar ring, then a continuous functional
calculus over the larger scalar ring descends to the smaller one. -/
@[simps!]
def starAlgHom {R : Type u} {S : Type v} {A : Type w} [Semifield R]
    [StarRing R] [TopologicalSpace R] [TopologicalSemiring R] [ContinuousStar R] [Semifield S]
    [StarRing S] [TopologicalSpace S] [TopologicalSemiring S] [ContinuousStar S] [Ring A]
    [StarRing A] [Algebra R S] [Algebra R A] [Algebra S A]
    [IsScalarTower R S A] [StarModule R S] [ContinuousSMul R S] {a : A}
    (φ : C(spectrum S a, S) →⋆ₐ[S] A) {f : C(S, R)} (h : SpectrumRestricts a f) :
    C(spectrum R a, R) →⋆ₐ[R] A :=
  (φ.restrictScalars R).comp <|
    (ContinuousMap.compStarAlgHom (spectrum S a) (.ofId R S) (algebraMapCLM R S).continuous).comp <|
      ContinuousMap.compStarAlgHom' R R
        ⟨Subtype.map f h.subset_preimage, (map_continuous f).subtype_map
          fun x (hx : x ∈ spectrum S a) => h.subset_preimage hx⟩


lemma starAlgHom_id {a : A} {φ : C(spectrum S a, S) →⋆ₐ[S] A} {f : C(S, R)}
    (h : SpectrumRestricts a f) (h_id : φ (.restrict (spectrum S a) <| .id S) = a) :
    h.starAlgHom φ (.restrict (spectrum R a) <| .id R) = a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁷ : Semifield R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : MetricSpace R
    inst✝¹⁴ : TopologicalSemiring R
    inst✝¹³ : ContinuousStar R
    inst✝¹² : Semifield S
    inst✝¹¹ : StarRing S
    inst✝¹⁰ : MetricSpace S
    inst✝⁹ : TopologicalSemiring S
    inst✝⁸ : ContinuousStar S
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra S A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : StarAlgHom S (ContinuousMap (↑(spectrum S a)) S) A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMap.restrict (spectrum S a) (ContinuousMap.id S))) a
    ⊢ Eq ((SpectrumRestricts.starAlgHom φ h) (ContinuousMap.restrict (spectrum R a …
  -/
  simp only [SpectrumRestricts.starAlgHom_apply]
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁷ : Semifield R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : MetricSpace R
    inst✝¹⁴ : TopologicalSemiring R
    inst✝¹³ : ContinuousStar R
    inst✝¹² : Semifield S
    inst✝¹¹ : StarRing S
    inst✝¹⁰ : MetricSpace S
    inst✝⁹ : TopologicalSemiring S
    inst✝⁸ : ContinuousStar S
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra S A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : StarAlgHom S (ContinuousMap (↑(spectrum S a)) S) A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMap.restrict (spectrum S a) (ContinuousMap.id S))) a
    ⊢ Eq (φ ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ }.comp ((Con …
  -/
  convert h_id
  /-
    case h.e'_2.h.e'_6
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁷ : Semifield R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : MetricSpace R
    inst✝¹⁴ : TopologicalSemiring R
    inst✝¹³ : ContinuousStar R
    inst✝¹² : Semifield S
    inst✝¹¹ : StarRing S
    inst✝¹⁰ : MetricSpace S
    inst✝⁹ : TopologicalSemiring S
    inst✝⁸ : ContinuousStar S
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra S A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : StarAlgHom S (ContinuousMap (↑(spectrum S a)) S) A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMap.restrict (spectrum S a) (ContinuousMap.id S))) a
    ⊢ Eq ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ }.comp ((Contin …
  -/
  ext x
  /-
    case h.e'_2.h.e'_6.h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁷ : Semifield R
    inst✝¹⁶ : StarRing R
    inst✝¹⁵ : MetricSpace R
    inst✝¹⁴ : TopologicalSemiring R
    inst✝¹³ : ContinuousStar R
    inst✝¹² : Semifield S
    inst✝¹¹ : StarRing S
    inst✝¹⁰ : MetricSpace S
    inst✝⁹ : TopologicalSemiring S
    inst✝⁸ : ContinuousStar S
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra S A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : StarAlgHom S (ContinuousMap (↑(spectrum S a)) S) A
    f : ContinuousMap S R
    h : SpectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMap.restrict (spectrum S a) (ContinuousMap.id S))) a
    x : ↑(spectrum S a)
    ⊢ Eq (({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ }.comp ((Conti …
  -/
  exact h.rightInvOn x.2
  /-
    🎉 no goals
  -/


lemma isClosedEmbedding_starAlgHom {a : A} {φ : C(spectrum S a, S) →⋆ₐ[S] A}
    (hφ : IsClosedEmbedding φ) {f : C(S, R)} (h : SpectrumRestricts a f)
    (halg : IsUniformEmbedding (algebraMap R S)) :
    IsClosedEmbedding (h.starAlgHom φ) :=
  hφ.comp <| IsUniformEmbedding.isClosedEmbedding <| .comp
    (ContinuousMap.isUniformEmbedding_comp _ halg)
    (UniformEquiv.arrowCongr h.homeomorph.symm (.refl _) |>.isUniformEmbedding)


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_starAlgHom := isClosedEmbedding_starAlgHom


/-- Given a `ContinuousFunctionalCalculus S q`. If we form the predicate `p` for `a : A`
characterized by: `q a` and the spectrum of `a` restricts to the scalar subring `R` via
`f : C(S, R)`, then we can get a restricted functional calculus
`ContinuousFunctionalCalculus R p`. -/
protected theorem cfc (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S)) (h0 : p 0)
    (h : ∀ a, p a ↔ q a ∧ SpectrumRestricts a f) :
    ContinuousFunctionalCalculus R p where
  predicate_zero := h0
  spectrum_nonempty a ha := ((h a).mp ha).2.image ▸
    (ContinuousFunctionalCalculus.spectrum_nonempty a ((h a).mp ha).1 |>.image f)
  compactSpace_spectrum a := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁰ : Semifield R
      inst✝¹⁹ : StarRing R
      inst✝¹⁸ : MetricSpace R
      inst✝¹⁷ : TopologicalSemiring R
      inst✝¹⁶ : ContinuousStar R
      inst✝¹⁵ : Semifield S
      inst✝¹⁴ : StarRing S
      inst✝¹³ : MetricSpace S
      inst✝¹² : TopologicalSemiring S
      inst✝¹¹ : ContinuousStar S
      inst✝¹⁰ : Ring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Algebra S A
      inst✝⁷ : Algebra R S
      inst✝⁶ : Algebra R A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : StarModule R S
      inst✝³ : ContinuousSMul R S
      inst✝² : TopologicalSpace A
      inst✝¹ : ContinuousFunctionalCalculus S q
      inst✝ : CompleteSpace R
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ⊢ CompactSpace ↑(spectrum R a)
    -/
    have := ContinuousFunctionalCalculus.compactSpace_spectrum (R := S) a
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁰ : Semifield R
      inst✝¹⁹ : StarRing R
      inst✝¹⁸ : MetricSpace R
      inst✝¹⁷ : TopologicalSemiring R
      inst✝¹⁶ : ContinuousStar R
      inst✝¹⁵ : Semifield S
      inst✝¹⁴ : StarRing S
      inst✝¹³ : MetricSpace S
      inst✝¹² : TopologicalSemiring S
      inst✝¹¹ : ContinuousStar S
      inst✝¹⁰ : Ring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Algebra S A
      inst✝⁷ : Algebra R S
      inst✝⁶ : Algebra R A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : StarModule R S
      inst✝³ : ContinuousSMul R S
      inst✝² : TopologicalSpace A
      inst✝¹ : ContinuousFunctionalCalculus S q
      inst✝ : CompleteSpace R
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      this : CompactSpace ↑(spectrum S a)
      ⊢ CompactSpace ↑(spectrum R a)
    -/
    rw [← isCompact_iff_compactSpace] at this ⊢
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁰ : Semifield R
      inst✝¹⁹ : StarRing R
      inst✝¹⁸ : MetricSpace R
      inst✝¹⁷ : TopologicalSemiring R
      inst✝¹⁶ : ContinuousStar R
      inst✝¹⁵ : Semifield S
      inst✝¹⁴ : StarRing S
      inst✝¹³ : MetricSpace S
      inst✝¹² : TopologicalSemiring S
      inst✝¹¹ : ContinuousStar S
      inst✝¹⁰ : Ring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Algebra S A
      inst✝⁷ : Algebra R S
      inst✝⁶ : Algebra R A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : StarModule R S
      inst✝³ : ContinuousSMul R S
      inst✝² : TopologicalSpace A
      inst✝¹ : ContinuousFunctionalCalculus S q
      inst✝ : CompleteSpace R
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      this : IsCompact (spectrum S a)
      ⊢ IsCompact (spectrum R a)
    -/
    simpa using halg.isClosedEmbedding.isCompact_preimage this
    /-
      🎉 no goals
    -/
  exists_cfc_of_predicate a ha := by
    refine ⟨((h a).mp ha).2.starAlgHom (cfcHom ((h a).mp ha).1 (R := S)),
      ?hom_isClosedEmbedding, ?hom_id, ?hom_map_spectrum, ?predicate_hom⟩
    case hom_isClosedEmbedding =>
      exact ((h a).mp ha).2.isClosedEmbedding_starAlgHom
        (cfcHom_isClosedEmbedding ((h a).mp ha).1) halg
    /-
      case hom_id
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁰ : Semifield R
      inst✝¹⁹ : StarRing R
      inst✝¹⁸ : MetricSpace R
      inst✝¹⁷ : TopologicalSemiring R
      inst✝¹⁶ : ContinuousStar R
      inst✝¹⁵ : Semifield S
      inst✝¹⁴ : StarRing S
      inst✝¹³ : MetricSpace S
      inst✝¹² : TopologicalSemiring S
      inst✝¹¹ : ContinuousStar S
      inst✝¹⁰ : Ring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Algebra S A
      inst✝⁷ : Algebra R S
      inst✝⁶ : Algebra R A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : StarModule R S
      inst✝³ : ContinuousSMul R S
      inst✝² : TopologicalSpace A
      inst✝¹ : ContinuousFunctionalCalculus S q
      inst✝ : CompleteSpace R
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ⊢ Eq ((SpectrumRestricts.starAlgHom (cfcHom ⋯) ⋯) (ContinuousMap.restrict (spe …
    -/
    case hom_id => exact ((h a).mp ha).2.starAlgHom_id <| cfcHom_id ((h a).mp ha).1
    case hom_map_spectrum =>
      intro g
      rw [SpectrumRestricts.starAlgHom_apply]
      simp only [← @spectrum.preimage_algebraMap (R := R) S, cfcHom_map_spectrum]
      ext x
      constructor
      · rintro ⟨y, hy⟩
        have := congr_arg f hy
        simp only [ContinuousMap.coe_mk, ContinuousMap.comp_apply, StarAlgHom.ofId_apply] at this
        rw [((h a).mp ha).2.left_inv _, ((h a).mp ha).2.left_inv _] at this
        exact ⟨_, this⟩
      · rintro ⟨y, rfl⟩
        rw [Set.mem_preimage]
        refine ⟨⟨algebraMap R S y, spectrum.algebraMap_mem S y.prop⟩, ?_⟩
        simp only [ContinuousMap.coe_mk, ContinuousMap.comp_apply, StarAlgHom.ofId_apply]
        congr
        exact Subtype.ext (((h a).mp ha).2.left_inv y)
    case predicate_hom =>
      intro g
      rw [h]
      refine ⟨cfcHom_predicate _ _, ?_⟩
      refine .of_rightInvOn (((h a).mp ha).2.left_inv) fun s hs ↦ ?_
      rw [SpectrumRestricts.starAlgHom_apply, cfcHom_map_spectrum] at hs
      obtain ⟨r, rfl⟩ := hs
      simp [((h a).mp ha).2.left_inv _]


lemma cfcHom_eq_restrict (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S))
    {a : A} (hpa : p a) (hqa : q a) (h : SpectrumRestricts a f) :
    cfcHom hpa = h.starAlgHom (cfcHom hqa) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    p q : A → Prop
    inst✝²² : Semifield R
    inst✝²¹ : StarRing R
    inst✝²⁰ : MetricSpace R
    inst✝¹⁹ : TopologicalSemiring R
    inst✝¹⁸ : ContinuousStar R
    inst✝¹⁷ : Semifield S
    inst✝¹⁶ : StarRing S
    inst✝¹⁵ : MetricSpace S
    inst✝¹⁴ : TopologicalSemiring S
    inst✝¹³ : ContinuousStar S
    inst✝¹² : Ring A
    inst✝¹¹ : StarRing A
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R A
    inst✝⁷ : IsScalarTower R S A
    inst✝⁶ : StarModule R S
    inst✝⁵ : ContinuousSMul R S
    inst✝⁴ : TopologicalSpace A
    inst✝³ : ContinuousFunctionalCalculus S q
    inst✝² : CompleteSpace R
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : ContinuousMap S R
    halg : IsUniformEmbedding ⇑(algebraMap R S)
    a : A
    hpa : p a
    hqa : q a
    h : SpectrumRestricts a ⇑f
    ⊢ Eq (cfcHom hpa) (SpectrumRestricts.starAlgHom (cfcHom hqa) h)
  -/
  apply cfcHom_eq_of_continuous_of_map_id
    /-
      case hφ₁
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      ⊢ Continuous ⇑(SpectrumRestricts.starAlgHom (cfcHom hqa) h)
    -/
  · exact h.isClosedEmbedding_starAlgHom (cfcHom_isClosedEmbedding hqa) halg |>.continuous
    /-
      🎉 no goals
    -/
    /-
      case hφ₂
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      ⊢ Eq ((SpectrumRestricts.starAlgHom (cfcHom hqa) h) (ContinuousMap.restrict (s …
    -/
  · exact h.starAlgHom_id (cfcHom_id hqa)
    /-
      🎉 no goals
    -/


lemma cfc_eq_restrict (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S)) {a : A} (hpa : p a)
    (hqa : q a) (h : SpectrumRestricts a f) (g : R → R) :
    cfc g a = cfc (fun x ↦ algebraMap R S (g (f x))) a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    p q : A → Prop
    inst✝²² : Semifield R
    inst✝²¹ : StarRing R
    inst✝²⁰ : MetricSpace R
    inst✝¹⁹ : TopologicalSemiring R
    inst✝¹⁸ : ContinuousStar R
    inst✝¹⁷ : Semifield S
    inst✝¹⁶ : StarRing S
    inst✝¹⁵ : MetricSpace S
    inst✝¹⁴ : TopologicalSemiring S
    inst✝¹³ : ContinuousStar S
    inst✝¹² : Ring A
    inst✝¹¹ : StarRing A
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R A
    inst✝⁷ : IsScalarTower R S A
    inst✝⁶ : StarModule R S
    inst✝⁵ : ContinuousSMul R S
    inst✝⁴ : TopologicalSpace A
    inst✝³ : ContinuousFunctionalCalculus S q
    inst✝² : CompleteSpace R
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : ContinuousMap S R
    halg : IsUniformEmbedding ⇑(algebraMap R S)
    a : A
    hpa : p a
    hqa : q a
    h : SpectrumRestricts a ⇑f
    g : R → R
    ⊢ Eq (cfc g a) (cfc (fun x => (algebraMap R S) (g (f x))) a)
  -/
  by_cases hg : ContinuousOn g (spectrum R a)
  · rw [cfc_apply g a, cfcHom_eq_restrict f halg hpa hqa h, SpectrumRestricts.starAlgHom_apply,
      cfcHom_eq_cfc_extend 0]
    /-
      case pos
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (spectrum R a)
      ⊢ Eq (cfc (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S), c …
    -/
    apply cfc_congr fun x hx ↦ ?_
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (spectrum R a)
      x : S
      hx : Membership.mem (spectrum S a) x
      ⊢ Eq (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S), contin …
    -/
    lift x to spectrum S a using hx
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (spectrum R a)
      x : Subtype fun x => Membership.mem (spectrum S a) x
      ⊢ Eq (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S), contin …
    -/
    simp [Function.comp, Subtype.val_injective.extend_apply]
    /-
      🎉 no goals
    -/
  · have : ¬ ContinuousOn (fun x ↦ algebraMap R S (g (f x)) : S → S) (spectrum S a) := by
      refine fun hg' ↦ hg ?_
      rw [halg.isEmbedding.continuousOn_iff]
      simpa [halg.isEmbedding.continuousOn_iff, Function.comp_def, h.left_inv _] using
        hg'.comp halg.isEmbedding.continuous.continuousOn (fun _ : R ↦ spectrum.algebraMap_mem S)
    /-
      case neg
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²² : Semifield R
      inst✝²¹ : StarRing R
      inst✝²⁰ : MetricSpace R
      inst✝¹⁹ : TopologicalSemiring R
      inst✝¹⁸ : ContinuousStar R
      inst✝¹⁷ : Semifield S
      inst✝¹⁶ : StarRing S
      inst✝¹⁵ : MetricSpace S
      inst✝¹⁴ : TopologicalSemiring S
      inst✝¹³ : ContinuousStar S
      inst✝¹² : Ring A
      inst✝¹¹ : StarRing A
      inst✝¹⁰ : Algebra S A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : ContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : SpectrumRestricts a ⇑f
      g : R → R
      hg : Not (ContinuousOn g (spectrum R a))
      this : Not (ContinuousOn (fun x => (algebraMap R S) (g (f x))) (spectrum S a))
      ⊢ Eq (cfc g a) (cfc (fun x => (algebraMap R S) (g (f x))) a)
    -/
    rw [cfc_apply_of_not_continuousOn a hg, cfc_apply_of_not_continuousOn a this]
    /-
      🎉 no goals
    -/


local notation "σₙ" => quasispectrum

/-- The homeomorphism `quasispectrum S a ≃ₜ quasispectrum R a` induced by
`QuasispectrumRestricts a f`. -/
def homeomorph {R S A : Type*} [Semifield R] [Field S] [NonUnitalRing A]
    [Algebra R S] [Module R A] [Module S A] [IsScalarTower R S A] [TopologicalSpace R]
    [TopologicalSpace S] [ContinuousSMul R S] [IsScalarTower S A A] [SMulCommClass S A A]
    {a : A} {f : C(S, R)} (h : QuasispectrumRestricts a f) :
    σₙ S a ≃ₜ σₙ R a where
  toFun := MapsTo.restrict f _ _ h.subset_preimage
  invFun := MapsTo.restrict (algebraMap R S) _ _ (image_subset_iff.mp h.algebraMap_image.subset)
  left_inv x := Subtype.ext <| h.rightInvOn x.2
  right_inv x := Subtype.ext <| h.left_inv x
  continuous_toFun := continuous_induced_rng.mpr <| f.continuous.comp continuous_induced_dom
  continuous_invFun := continuous_induced_rng.mpr <|
    continuous_algebraMap R S |>.comp continuous_induced_dom


/-- If the quasispectrum of an element restricts to a smaller scalar ring, then a non-unital
continuous functional calculus over the larger scalar ring descends to the smaller one. -/
@[simps!]
def nonUnitalStarAlgHom {R : Type u} {S : Type v} {A : Type w} [Semifield R]
    [StarRing R] [TopologicalSpace R] [TopologicalSemiring R] [ContinuousStar R] [Field S]
    [StarRing S] [TopologicalSpace S] [TopologicalRing S] [ContinuousStar S] [NonUnitalRing A]
    [StarRing A] [Algebra R S] [Module R A] [Module S A] [IsScalarTower S A A] [SMulCommClass S A A]
    [IsScalarTower R S A] [StarModule R S] [ContinuousSMul R S] {a : A}
    (φ : C(σₙ S a, S)₀ →⋆ₙₐ[S] A) {f : C(S, R)} (h : QuasispectrumRestricts a f) :
    C(σₙ R a, R)₀ →⋆ₙₐ[R] A :=
  (φ.restrictScalars R).comp <|
    (nonUnitalStarAlgHom_postcomp (σₙ S a) (StarAlgHom.ofId R S) (algebraMapCLM R S).continuous)
      |>.comp <| nonUnitalStarAlgHom_precomp R
        ⟨⟨Subtype.map f h.subset_preimage, (map_continuous f).subtype_map
          fun x (hx : x ∈ σₙ S a) => h.subset_preimage hx⟩, Subtype.ext h.map_zero⟩


lemma nonUnitalStarAlgHom_id {a : A} {φ : C(σₙ S a, S)₀ →⋆ₙₐ[S] A} {f : C(S, R)}
    (h : QuasispectrumRestricts a f) (h_id : φ (.id rfl) = a) :
    h.nonUnitalStarAlgHom φ (.id rfl) = a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁹ : Semifield R
    inst✝¹⁸ : StarRing R
    inst✝¹⁷ : MetricSpace R
    inst✝¹⁶ : TopologicalSemiring R
    inst✝¹⁵ : ContinuousStar R
    inst✝¹⁴ : Field S
    inst✝¹³ : StarRing S
    inst✝¹² : MetricSpace S
    inst✝¹¹ : TopologicalRing S
    inst✝¹⁰ : ContinuousStar S
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : Module S A
    inst✝⁶ : IsScalarTower S A A
    inst✝⁵ : SMulCommClass S A A
    inst✝⁴ : Algebra R S
    inst✝³ : Module R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : NonUnitalStarAlgHom S (ContinuousMapZero (↑(quasispectrum S a)) S) A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMapZero.id ⋯)) a
    ⊢ Eq ((QuasispectrumRestricts.nonUnitalStarAlgHom φ h) (ContinuousMapZero.id ⋯ …
  -/
  simp only [QuasispectrumRestricts.nonUnitalStarAlgHom_apply]
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁹ : Semifield R
    inst✝¹⁸ : StarRing R
    inst✝¹⁷ : MetricSpace R
    inst✝¹⁶ : TopologicalSemiring R
    inst✝¹⁵ : ContinuousStar R
    inst✝¹⁴ : Field S
    inst✝¹³ : StarRing S
    inst✝¹² : MetricSpace S
    inst✝¹¹ : TopologicalRing S
    inst✝¹⁰ : ContinuousStar S
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : Module S A
    inst✝⁶ : IsScalarTower S A A
    inst✝⁵ : SMulCommClass S A A
    inst✝⁴ : Algebra R S
    inst✝³ : Module R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : NonUnitalStarAlgHom S (ContinuousMapZero (↑(quasispectrum S a)) S) A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMapZero.id ⋯)) a
    ⊢ Eq (φ ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯, map_zero' : …
  -/
  convert h_id
  /-
    case h.e'_2.h.e'_6
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁹ : Semifield R
    inst✝¹⁸ : StarRing R
    inst✝¹⁷ : MetricSpace R
    inst✝¹⁶ : TopologicalSemiring R
    inst✝¹⁵ : ContinuousStar R
    inst✝¹⁴ : Field S
    inst✝¹³ : StarRing S
    inst✝¹² : MetricSpace S
    inst✝¹¹ : TopologicalRing S
    inst✝¹⁰ : ContinuousStar S
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : Module S A
    inst✝⁶ : IsScalarTower S A A
    inst✝⁵ : SMulCommClass S A A
    inst✝⁴ : Algebra R S
    inst✝³ : Module R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : NonUnitalStarAlgHom S (ContinuousMapZero (↑(quasispectrum S a)) S) A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMapZero.id ⋯)) a
    ⊢ Eq ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯, map_zero' := ⋯ …
  -/
  ext x
  /-
    case h.e'_2.h.e'_6.h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝¹⁹ : Semifield R
    inst✝¹⁸ : StarRing R
    inst✝¹⁷ : MetricSpace R
    inst✝¹⁶ : TopologicalSemiring R
    inst✝¹⁵ : ContinuousStar R
    inst✝¹⁴ : Field S
    inst✝¹³ : StarRing S
    inst✝¹² : MetricSpace S
    inst✝¹¹ : TopologicalRing S
    inst✝¹⁰ : ContinuousStar S
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : Module S A
    inst✝⁶ : IsScalarTower S A A
    inst✝⁵ : SMulCommClass S A A
    inst✝⁴ : Algebra R S
    inst✝³ : Module R A
    inst✝² : IsScalarTower R S A
    inst✝¹ : StarModule R S
    inst✝ : ContinuousSMul R S
    a : A
    φ : NonUnitalStarAlgHom S (ContinuousMapZero (↑(quasispectrum S a)) S) A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_id : Eq (φ (ContinuousMapZero.id ⋯)) a
    x : ↑(quasispectrum S a)
    ⊢ Eq (({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯, map_zero' :=  …
  -/
  exact h.rightInvOn x.2
  /-
    🎉 no goals
  -/


lemma isClosedEmbedding_nonUnitalStarAlgHom {a : A} {φ : C(σₙ S a, S)₀ →⋆ₙₐ[S] A}
    (hφ : IsClosedEmbedding φ) {f : C(S, R)} (h : QuasispectrumRestricts a f)
    (halg : IsUniformEmbedding (algebraMap R S)) :
    IsClosedEmbedding (h.nonUnitalStarAlgHom φ) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝²¹ : Semifield R
    inst✝²⁰ : StarRing R
    inst✝¹⁹ : MetricSpace R
    inst✝¹⁸ : TopologicalSemiring R
    inst✝¹⁷ : ContinuousStar R
    inst✝¹⁶ : Field S
    inst✝¹⁵ : StarRing S
    inst✝¹⁴ : MetricSpace S
    inst✝¹³ : TopologicalRing S
    inst✝¹² : ContinuousStar S
    inst✝¹¹ : NonUnitalRing A
    inst✝¹⁰ : StarRing A
    inst✝⁹ : Module S A
    inst✝⁸ : IsScalarTower S A A
    inst✝⁷ : SMulCommClass S A A
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module R A
    inst✝⁴ : IsScalarTower R S A
    inst✝³ : StarModule R S
    inst✝² : ContinuousSMul R S
    inst✝¹ : TopologicalSpace A
    inst✝ : CompleteSpace R
    a : A
    φ : NonUnitalStarAlgHom S (ContinuousMapZero (↑(quasispectrum S a)) S) A
    hφ : Topology.IsClosedEmbedding ⇑φ
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    halg : IsUniformEmbedding ⇑(algebraMap R S)
    ⊢ Topology.IsClosedEmbedding ⇑(QuasispectrumRestricts.nonUnitalStarAlgHom φ h)
  -/
  have : h.homeomorph.symm 0 = 0 := Subtype.ext (map_zero <| algebraMap _ _)
  refine hφ.comp <| IsUniformEmbedding.isClosedEmbedding <| .comp
    (ContinuousMapZero.isUniformEmbedding_comp _ halg)
    (UniformEquiv.arrowCongrLeft₀ h.homeomorph.symm this |>.isUniformEmbedding)


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_nonUnitalStarAlgHom := isClosedEmbedding_nonUnitalStarAlgHom


/-- Given a `NonUnitalContinuousFunctionalCalculus S q`. If we form the predicate `p` for `a : A`
characterized by: `q a` and the quasispectrum of `a` restricts to the scalar subring `R` via
`f : C(S, R)`, then we can get a restricted functional calculus
`NonUnitalContinuousFunctionalCalculus R p`. -/
protected theorem cfc (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S)) (h0 : p 0)
    (h : ∀ a, p a ↔ q a ∧ QuasispectrumRestricts a f) :
    NonUnitalContinuousFunctionalCalculus R p where
  predicate_zero := h0
  compactSpace_quasispectrum a := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁴ : Semifield R
      inst✝²³ : StarRing R
      inst✝²² : MetricSpace R
      inst✝²¹ : TopologicalSemiring R
      inst✝²⁰ : ContinuousStar R
      inst✝¹⁹ : Field S
      inst✝¹⁸ : StarRing S
      inst✝¹⁷ : MetricSpace S
      inst✝¹⁶ : TopologicalRing S
      inst✝¹⁵ : ContinuousStar S
      inst✝¹⁴ : NonUnitalRing A
      inst✝¹³ : StarRing A
      inst✝¹² : Module S A
      inst✝¹¹ : IsScalarTower S A A
      inst✝¹⁰ : SMulCommClass S A A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : NonUnitalContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ⊢ CompactSpace ↑(quasispectrum R a)
    -/
    have := NonUnitalContinuousFunctionalCalculus.compactSpace_quasispectrum (R := S) a
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁴ : Semifield R
      inst✝²³ : StarRing R
      inst✝²² : MetricSpace R
      inst✝²¹ : TopologicalSemiring R
      inst✝²⁰ : ContinuousStar R
      inst✝¹⁹ : Field S
      inst✝¹⁸ : StarRing S
      inst✝¹⁷ : MetricSpace S
      inst✝¹⁶ : TopologicalRing S
      inst✝¹⁵ : ContinuousStar S
      inst✝¹⁴ : NonUnitalRing A
      inst✝¹³ : StarRing A
      inst✝¹² : Module S A
      inst✝¹¹ : IsScalarTower S A A
      inst✝¹⁰ : SMulCommClass S A A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : NonUnitalContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      this : CompactSpace ↑(quasispectrum S a)
      ⊢ CompactSpace ↑(quasispectrum R a)
    -/
    rw [← isCompact_iff_compactSpace] at this ⊢
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁴ : Semifield R
      inst✝²³ : StarRing R
      inst✝²² : MetricSpace R
      inst✝²¹ : TopologicalSemiring R
      inst✝²⁰ : ContinuousStar R
      inst✝¹⁹ : Field S
      inst✝¹⁸ : StarRing S
      inst✝¹⁷ : MetricSpace S
      inst✝¹⁶ : TopologicalRing S
      inst✝¹⁵ : ContinuousStar S
      inst✝¹⁴ : NonUnitalRing A
      inst✝¹³ : StarRing A
      inst✝¹² : Module S A
      inst✝¹¹ : IsScalarTower S A A
      inst✝¹⁰ : SMulCommClass S A A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : NonUnitalContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      this : IsCompact (quasispectrum S a)
      ⊢ IsCompact (quasispectrum R a)
    -/
    simpa using halg.isClosedEmbedding.isCompact_preimage this
    /-
      🎉 no goals
    -/
  exists_cfc_of_predicate a ha := by
    refine ⟨((h a).mp ha).2.nonUnitalStarAlgHom (cfcₙHom ((h a).mp ha).1 (R := S)),
      ?hom_isClosedEmbedding, ?hom_id, ?hom_map_spectrum, ?predicate_hom⟩
    case hom_isClosedEmbedding =>
      exact ((h a).mp ha).2.isClosedEmbedding_nonUnitalStarAlgHom
        (cfcₙHom_isClosedEmbedding ((h a).mp ha).1) halg
    /-
      case hom_id
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁴ : Semifield R
      inst✝²³ : StarRing R
      inst✝²² : MetricSpace R
      inst✝²¹ : TopologicalSemiring R
      inst✝²⁰ : ContinuousStar R
      inst✝¹⁹ : Field S
      inst✝¹⁸ : StarRing S
      inst✝¹⁷ : MetricSpace S
      inst✝¹⁶ : TopologicalRing S
      inst✝¹⁵ : ContinuousStar S
      inst✝¹⁴ : NonUnitalRing A
      inst✝¹³ : StarRing A
      inst✝¹² : Module S A
      inst✝¹¹ : IsScalarTower S A A
      inst✝¹⁰ : SMulCommClass S A A
      inst✝⁹ : Algebra R S
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : StarModule R S
      inst✝⁵ : ContinuousSMul R S
      inst✝⁴ : TopologicalSpace A
      inst✝³ : NonUnitalContinuousFunctionalCalculus S q
      inst✝² : CompleteSpace R
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ⊢ Eq ((QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom ⋯) ⋯) { toContinuou …
    -/
    case hom_id => exact ((h a).mp ha).2.nonUnitalStarAlgHom_id <| cfcₙHom_id ((h a).mp ha).1
    case hom_map_spectrum =>
      intro g
      rw [nonUnitalStarAlgHom_apply]
      simp only [← @quasispectrum.preimage_algebraMap (R := R) S, cfcₙHom_map_quasispectrum]
      ext x
      constructor
      · rintro ⟨y, hy⟩
        have := congr_arg f hy
        simp only [nonUnitalStarAlgHom_postcomp_apply, NonUnitalStarAlgHom.coe_coe,
          Function.comp_apply, comp_apply, coe_mk, ContinuousMap.coe_mk, StarAlgHom.ofId_apply]
          at this
        rw [((h a).mp ha).2.left_inv _, ((h a).mp ha).2.left_inv _] at this
        exact ⟨_, this⟩
      · rintro ⟨y, rfl⟩
        rw [Set.mem_preimage]
        refine ⟨⟨algebraMap R S y, quasispectrum.algebraMap_mem S y.prop⟩, ?_⟩
        simp only [nonUnitalStarAlgHom_postcomp_apply, NonUnitalStarAlgHom.coe_coe,
          Function.comp_apply, comp_apply, coe_mk, ContinuousMap.coe_mk, StarAlgHom.ofId_apply]
        congr
        exact Subtype.ext (((h a).mp ha).2.left_inv y)
    case predicate_hom =>
      intro g
      rw [h]
      refine ⟨cfcₙHom_predicate _ _, ?_⟩
      refine { rightInvOn := fun s hs ↦ ?_, left_inv := ((h a).mp ha).2.left_inv }
      rw [nonUnitalStarAlgHom_apply,
        cfcₙHom_map_quasispectrum] at hs
      obtain ⟨r, rfl⟩ := hs
      simp [((h a).mp ha).2.left_inv _]


lemma cfcₙHom_eq_restrict (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S)) {a : A}
    (hpa : p a) (hqa : q a) (h : QuasispectrumRestricts a f) :
    cfcₙHom hpa = h.nonUnitalStarAlgHom (cfcₙHom hqa) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    p q : A → Prop
    inst✝²⁶ : Semifield R
    inst✝²⁵ : StarRing R
    inst✝²⁴ : MetricSpace R
    inst✝²³ : TopologicalSemiring R
    inst✝²² : ContinuousStar R
    inst✝²¹ : Field S
    inst✝²⁰ : StarRing S
    inst✝¹⁹ : MetricSpace S
    inst✝¹⁸ : TopologicalRing S
    inst✝¹⁷ : ContinuousStar S
    inst✝¹⁶ : NonUnitalRing A
    inst✝¹⁵ : StarRing A
    inst✝¹⁴ : Module S A
    inst✝¹³ : IsScalarTower S A A
    inst✝¹² : SMulCommClass S A A
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Module R A
    inst✝⁹ : IsScalarTower R S A
    inst✝⁸ : StarModule R S
    inst✝⁷ : ContinuousSMul R S
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
    inst✝⁴ : CompleteSpace R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
    f : ContinuousMap S R
    halg : IsUniformEmbedding ⇑(algebraMap R S)
    a : A
    hpa : p a
    hqa : q a
    h : QuasispectrumRestricts a ⇑f
    ⊢ Eq (cfcₙHom hpa) (QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom hqa) h)
  -/
  apply cfcₙHom_eq_of_continuous_of_map_id
    /-
      case hφ₁
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      ⊢ Continuous ⇑(QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom hqa) h)
    -/
  · exact h.isClosedEmbedding_nonUnitalStarAlgHom (cfcₙHom_isClosedEmbedding hqa) halg |>.continuous
    /-
      🎉 no goals
    -/
    /-
      case hφ₂
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      ⊢ Eq ((QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom hqa) h) { toContinu …
    -/
  · exact h.nonUnitalStarAlgHom_id (cfcₙHom_id hqa)
    /-
      🎉 no goals
    -/


lemma cfcₙ_eq_restrict (f : C(S, R)) (halg : IsUniformEmbedding (algebraMap R S)) {a : A}
    (hpa : p a) (hqa : q a) (h : QuasispectrumRestricts a f) (g : R → R) :
    cfcₙ g a = cfcₙ (fun x ↦ algebraMap R S (g (f x))) a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    p q : A → Prop
    inst✝²⁶ : Semifield R
    inst✝²⁵ : StarRing R
    inst✝²⁴ : MetricSpace R
    inst✝²³ : TopologicalSemiring R
    inst✝²² : ContinuousStar R
    inst✝²¹ : Field S
    inst✝²⁰ : StarRing S
    inst✝¹⁹ : MetricSpace S
    inst✝¹⁸ : TopologicalRing S
    inst✝¹⁷ : ContinuousStar S
    inst✝¹⁶ : NonUnitalRing A
    inst✝¹⁵ : StarRing A
    inst✝¹⁴ : Module S A
    inst✝¹³ : IsScalarTower S A A
    inst✝¹² : SMulCommClass S A A
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Module R A
    inst✝⁹ : IsScalarTower R S A
    inst✝⁸ : StarModule R S
    inst✝⁷ : ContinuousSMul R S
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
    inst✝⁴ : CompleteSpace R
    inst✝³ : IsScalarTower R A A
    inst✝² : SMulCommClass R A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
    f : ContinuousMap S R
    halg : IsUniformEmbedding ⇑(algebraMap R S)
    a : A
    hpa : p a
    hqa : q a
    h : QuasispectrumRestricts a ⇑f
    g : R → R
    ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
  -/
  by_cases hg : ContinuousOn g (σₙ R a) ∧ g 0 = 0
    /-
      case pos
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : And (ContinuousOn g (quasispectrum R a)) (Eq (g 0) 0)
      ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
    -/
  · obtain ⟨hg, hg0⟩ := hg
    rw [cfcₙ_apply g a, cfcₙHom_eq_restrict f halg hpa hqa h, nonUnitalStarAlgHom_apply,
      cfcₙHom_eq_cfcₙ_extend 0]
    /-
      case pos.intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (quasispectrum R a)
      hg0 : Eq (g 0) 0
      ⊢ Eq (cfcₙ (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S),  …
    -/
    apply cfcₙ_congr fun x hx ↦ ?_
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (quasispectrum R a)
      hg0 : Eq (g 0) 0
      x : S
      hx : Membership.mem (quasispectrum S a) x
      ⊢ Eq (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S), contin …
    -/
    lift x to σₙ S a using hx
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : ContinuousOn g (quasispectrum R a)
      hg0 : Eq (g 0) 0
      x : Subtype fun x => Membership.mem (quasispectrum S a) x
      ⊢ Eq (Function.extend Subtype.val (⇑({ toFun := ⇑(StarAlgHom.ofId R S), contin …
    -/
    simp [Function.comp, Subtype.val_injective.extend_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : Not (And (ContinuousOn g (quasispectrum R a)) (Eq (g 0) 0))
      ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
    -/
  · simp only [not_and_or] at hg
    /-
      case neg
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁶ : Semifield R
      inst✝²⁵ : StarRing R
      inst✝²⁴ : MetricSpace R
      inst✝²³ : TopologicalSemiring R
      inst✝²² : ContinuousStar R
      inst✝²¹ : Field S
      inst✝²⁰ : StarRing S
      inst✝¹⁹ : MetricSpace S
      inst✝¹⁸ : TopologicalRing S
      inst✝¹⁷ : ContinuousStar S
      inst✝¹⁶ : NonUnitalRing A
      inst✝¹⁵ : StarRing A
      inst✝¹⁴ : Module S A
      inst✝¹³ : IsScalarTower S A A
      inst✝¹² : SMulCommClass S A A
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Module R A
      inst✝⁹ : IsScalarTower R S A
      inst✝⁸ : StarModule R S
      inst✝⁷ : ContinuousSMul R S
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
      inst✝⁴ : CompleteSpace R
      inst✝³ : IsScalarTower R A A
      inst✝² : SMulCommClass R A A
      inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : IsUniformEmbedding ⇑(algebraMap R S)
      a : A
      hpa : p a
      hqa : q a
      h : QuasispectrumRestricts a ⇑f
      g : R → R
      hg : Or (Not (ContinuousOn g (quasispectrum R a))) (Not (Eq (g 0) 0))
      ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
    -/
    obtain (hg | hg) := hg
    · have : ¬ ContinuousOn (fun x ↦ algebraMap R S (g (f x)) : S → S) (σₙ S a) := by
        refine fun hg' ↦ hg ?_
        rw [halg.isEmbedding.continuousOn_iff]
        simpa [halg.isEmbedding.continuousOn_iff, Function.comp_def, h.left_inv _] using
          hg'.comp halg.isEmbedding.continuous.continuousOn
          (fun _ : R ↦ quasispectrum.algebraMap_mem S)
      /-
        case neg.inl
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁶ : Semifield R
        inst✝²⁵ : StarRing R
        inst✝²⁴ : MetricSpace R
        inst✝²³ : TopologicalSemiring R
        inst✝²² : ContinuousStar R
        inst✝²¹ : Field S
        inst✝²⁰ : StarRing S
        inst✝¹⁹ : MetricSpace S
        inst✝¹⁸ : TopologicalRing S
        inst✝¹⁷ : ContinuousStar S
        inst✝¹⁶ : NonUnitalRing A
        inst✝¹⁵ : StarRing A
        inst✝¹⁴ : Module S A
        inst✝¹³ : IsScalarTower S A A
        inst✝¹² : SMulCommClass S A A
        inst✝¹¹ : Algebra R S
        inst✝¹⁰ : Module R A
        inst✝⁹ : IsScalarTower R S A
        inst✝⁸ : StarModule R S
        inst✝⁷ : ContinuousSMul R S
        inst✝⁶ : TopologicalSpace A
        inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
        inst✝⁴ : CompleteSpace R
        inst✝³ : IsScalarTower R A A
        inst✝² : SMulCommClass R A A
        inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : IsUniformEmbedding ⇑(algebraMap R S)
        a : A
        hpa : p a
        hqa : q a
        h : QuasispectrumRestricts a ⇑f
        g : R → R
        hg : Not (ContinuousOn g (quasispectrum R a))
        this : Not (ContinuousOn (fun x => (algebraMap R S) (g (f x))) (quasispectrum  …
        ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
      -/
      rw [cfcₙ_apply_of_not_continuousOn a hg, cfcₙ_apply_of_not_continuousOn a this]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁶ : Semifield R
        inst✝²⁵ : StarRing R
        inst✝²⁴ : MetricSpace R
        inst✝²³ : TopologicalSemiring R
        inst✝²² : ContinuousStar R
        inst✝²¹ : Field S
        inst✝²⁰ : StarRing S
        inst✝¹⁹ : MetricSpace S
        inst✝¹⁸ : TopologicalRing S
        inst✝¹⁷ : ContinuousStar S
        inst✝¹⁶ : NonUnitalRing A
        inst✝¹⁵ : StarRing A
        inst✝¹⁴ : Module S A
        inst✝¹³ : IsScalarTower S A A
        inst✝¹² : SMulCommClass S A A
        inst✝¹¹ : Algebra R S
        inst✝¹⁰ : Module R A
        inst✝⁹ : IsScalarTower R S A
        inst✝⁸ : StarModule R S
        inst✝⁷ : ContinuousSMul R S
        inst✝⁶ : TopologicalSpace A
        inst✝⁵ : NonUnitalContinuousFunctionalCalculus S q
        inst✝⁴ : CompleteSpace R
        inst✝³ : IsScalarTower R A A
        inst✝² : SMulCommClass R A A
        inst✝¹ : NonUnitalContinuousFunctionalCalculus R p
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : IsUniformEmbedding ⇑(algebraMap R S)
        a : A
        hpa : p a
        hqa : q a
        h : QuasispectrumRestricts a ⇑f
        g : R → R
        hg : Not (Eq (g 0) 0)
        ⊢ Eq (cfcₙ g a) (cfcₙ (fun x => (algebraMap R S) (g (f x))) a)
      -/
    · rw [cfcₙ_apply_of_not_map_zero a hg, cfcₙ_apply_of_not_map_zero a (by simpa [h.map_zero])]
      /-
        🎉 no goals
      -/


