local notation "σₙ" => quasispectrum

local notation "σ" => spectrum


local postfix:max "⁺¹" => Unitization 𝕜


open Unitization in
/--
This is an auxiliary definition used for constructing an instance of the non-unital continuous
functional calculus given a instance of the unital one on the unitization.

This is the natural non-unital star homomorphism obtained from the chain
```lean
calc
  C(σₙ 𝕜 a, 𝕜)₀ →⋆ₙₐ[𝕜] C(σₙ 𝕜 a, 𝕜) := ContinuousMapZero.toContinuousMapHom
  _             ≃⋆[𝕜] C(σ 𝕜 (↑a : A⁺¹), 𝕜) := Homeomorph.compStarAlgEquiv'
  _             →⋆ₐ[𝕜] A⁺¹ := cfcHom
```
This range of this map is contained in the range of `(↑) : A → A⁺¹` (see `cfcₙAux_mem_range_inr`),
and so we may restrict it to `A` to get the necessary homomorphism for the non-unital continuous
functional calculus.
-/
noncomputable def cfcₙAux : C(σₙ 𝕜 a, 𝕜)₀ →⋆ₙₐ[𝕜] A⁺¹ :=
  (cfcHom (R := 𝕜) (hp₁.mpr ha) : C(σ 𝕜 (a : A⁺¹), 𝕜) →⋆ₙₐ[𝕜] A⁺¹) |>.comp
    (Homeomorph.compStarAlgEquiv' 𝕜 𝕜 <| .setCongr <| (quasispectrum_eq_spectrum_inr' 𝕜 𝕜 a).symm)
    |>.comp ContinuousMapZero.toContinuousMapHom


lemma cfcₙAux_id : cfcₙAux hp₁ a ha (ContinuousMapZero.id rfl) = a := cfcHom_id (hp₁.mpr ha)


open Unitization in
lemma isClosedEmbedding_cfcₙAux : IsClosedEmbedding (cfcₙAux hp₁ a ha) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    ⊢ Topology.IsClosedEmbedding ⇑(cfcₙAux ⋯ a ha)
  -/
  simp only [cfcₙAux, NonUnitalStarAlgHom.coe_comp]
  refine ((cfcHom_isClosedEmbedding (hp₁.mpr ha)).comp ?_).comp
    ContinuousMapZero.isClosedEmbedding_toContinuousMap
  let e : C(σₙ 𝕜 a, 𝕜) ≃ₜ C(σ 𝕜 (a : A⁺¹), 𝕜) :=
    (Homeomorph.setCongr (quasispectrum_eq_spectrum_inr' 𝕜 𝕜 a)).arrowCongr (.refl _)
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    e : Homeomorph (ContinuousMap (↑(quasispectrum 𝕜 a)) 𝕜) (ContinuousMap (↑(spec …
    ⊢ Topology.IsClosedEmbedding ⇑↑(Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorph.s …
  -/
  exact e.isClosedEmbedding
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_cfcₙAux := isClosedEmbedding_cfcₙAux


lemma spec_cfcₙAux (f : C(σₙ 𝕜 a, 𝕜)₀) : σ 𝕜 (cfcₙAux hp₁ a ha f) = Set.range f := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    ⊢ Eq (spectrum 𝕜 ((cfcₙAux ⋯ a ha) f)) (Set.range ⇑f)
  -/
  rw [cfcₙAux, NonUnitalStarAlgHom.comp_assoc, NonUnitalStarAlgHom.comp_apply]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    ⊢ Eq (spectrum 𝕜 (↑(cfcHom ⋯) (((↑(Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorp …
  -/
  simp only [NonUnitalStarAlgHom.comp_apply, NonUnitalStarAlgHom.coe_coe]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    ⊢ Eq (spectrum 𝕜 ((cfcHom ⋯) ((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorph.se …
  -/
  rw [cfcHom_map_spectrum (hp₁.mpr ha) (R := 𝕜) _]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    ⊢ Eq (Set.range ⇑((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorph.setCongr ⋯)) ( …
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    x : 𝕜
    ⊢ Iff (Membership.mem (Set.range ⇑((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomor …
  -/
  constructor
  /-
    case h.mp
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NonUnitalNormedRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : NormedSpace 𝕜 A
    inst✝³ : IsScalarTower 𝕜 A A
    inst✝² : SMulCommClass 𝕜 A A
    inst✝¹ : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    x : 𝕜
    ⊢ Membership.mem (Set.range ⇑((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorph.se …
  -/
  all_goals rintro ⟨x, rfl⟩
    /-
      case h.mp.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : NonUnitalNormedRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : NormedSpace 𝕜 A
      inst✝³ : IsScalarTower 𝕜 A A
      inst✝² : SMulCommClass 𝕜 A A
      inst✝¹ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      x : ↑(spectrum 𝕜 ↑a)
      ⊢ Membership.mem (Set.range ⇑f) (((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorp …
    -/
  · exact ⟨⟨x, (Unitization.quasispectrum_eq_spectrum_inr' 𝕜 𝕜 a).symm ▸ x.property⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : NonUnitalNormedRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : NormedSpace 𝕜 A
      inst✝³ : IsScalarTower 𝕜 A A
      inst✝² : SMulCommClass 𝕜 A A
      inst✝¹ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝ : ContinuousFunctionalCalculus 𝕜 p₁
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      x : ↑(quasispectrum 𝕜 a)
      ⊢ Membership.mem (Set.range ⇑((Homeomorph.compStarAlgEquiv' 𝕜 𝕜 (Homeomorph.se …
    -/
  · exact ⟨⟨x, Unitization.quasispectrum_eq_spectrum_inr' 𝕜 𝕜 a ▸ x.property⟩, rfl⟩
    /-
      🎉 no goals
    -/


lemma cfcₙAux_mem_range_inr (f : C(σₙ 𝕜 a, 𝕜)₀) :
    cfcₙAux hp₁ a ha f ∈ NonUnitalStarAlgHom.range (Unitization.inrNonUnitalStarAlgHom 𝕜 A) := by
  have h₁ := (isClosedEmbedding_cfcₙAux hp₁ a ha).continuous.range_subset_closure_image_dense
    (ContinuousMapZero.adjoin_id_dense (s := σₙ 𝕜 a) rfl) ⟨f, rfl⟩
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NonUnitalNormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedSpace 𝕜 A
    inst✝⁴ : IsScalarTower 𝕜 A A
    inst✝³ : SMulCommClass 𝕜 A A
    inst✝² : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
    inst✝ : CompleteSpace A
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
    ⊢ Membership.mem (NonUnitalStarAlgHom.range (Unitization.inrNonUnitalStarAlgHo …
  -/
  rw [← SetLike.mem_coe]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NonUnitalNormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedSpace 𝕜 A
    inst✝⁴ : IsScalarTower 𝕜 A A
    inst✝³ : SMulCommClass 𝕜 A A
    inst✝² : StarModule 𝕜 A
    p : A → Prop
    p₁ : Unitization 𝕜 A → Prop
    hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
    a : A
    ha : p a
    inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
    inst✝ : CompleteSpace A
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
    ⊢ Membership.mem (↑(NonUnitalStarAlgHom.range (Unitization.inrNonUnitalStarAlg …
  -/
  refine closure_minimal ?_ ?_ h₁
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      ⊢ HasSubset.Subset (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgebra.adjoin  …
    -/
  · rw [← NonUnitalStarSubalgebra.coe_map, SetLike.coe_subset_coe, NonUnitalStarSubalgebra.map_le]
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      ⊢ LE.le (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMapZero …
    -/
    apply NonUnitalStarAlgebra.adjoin_le
    /-
      case refine_1.hs
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      ⊢ HasSubset.Subset (Singleton.singleton (ContinuousMapZero.id ⋯)) ↑(NonUnitalS …
    -/
    apply Set.singleton_subset_iff.mpr
    /-
      case refine_1.hs
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      ⊢ Membership.mem (↑(NonUnitalStarSubalgebra.comap (cfcₙAux ⋯ a ha) (NonUnitalS …
    -/
    rw [SetLike.mem_coe, NonUnitalStarSubalgebra.mem_comap, cfcₙAux_id hp₁ a ha]
    /-
      case refine_1.hs
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      ⊢ Membership.mem (NonUnitalStarAlgHom.range (Unitization.inrNonUnitalStarAlgHo …
    -/
    exact ⟨a, rfl⟩
    /-
      🎉 no goals
    -/
  · have : Continuous (Unitization.fst (R := 𝕜) (A := A)) :=
      Unitization.uniformEquivProd.continuous.fst
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      this : Continuous Unitization.fst
      ⊢ IsClosed ↑(NonUnitalStarAlgHom.range (Unitization.inrNonUnitalStarAlgHom 𝕜 A))
    -/
    simp only [NonUnitalStarAlgHom.coe_range]
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      this : Continuous Unitization.fst
      ⊢ IsClosed (Set.range ⇑(Unitization.inrNonUnitalStarAlgHom 𝕜 A))
    -/
    convert IsClosed.preimage this (isClosed_singleton (x := 0))
    /-
      case h.e'_3
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NonUnitalNormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedSpace 𝕜 A
      inst✝⁴ : IsScalarTower 𝕜 A A
      inst✝³ : SMulCommClass 𝕜 A A
      inst✝² : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      a : A
      ha : p a
      inst✝¹ : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝ : CompleteSpace A
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      h₁ : Membership.mem (closure (Set.image ⇑(cfcₙAux ⋯ a ha) ↑(NonUnitalStarAlgeb …
      this : Continuous Unitization.fst
      ⊢ Eq (Set.range ⇑(Unitization.inrNonUnitalStarAlgHom 𝕜 A)) (Set.preimage Uniti …
    -/
    aesop
    /-
      🎉 no goals
    -/


include hp₁ in
open Unitization NonUnitalStarAlgHom in
theorem RCLike.nonUnitalContinuousFunctionalCalculus :
    NonUnitalContinuousFunctionalCalculus 𝕜 (p : A → Prop) where
  predicate_zero := by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NonUnitalNormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedSpace 𝕜 A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      inst✝² : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝¹ : CompleteSpace A
      inst✝ : CStarRing A
      ⊢ p 0
    -/
    rw [← hp₁, Unitization.inr_zero 𝕜]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NonUnitalNormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedSpace 𝕜 A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      inst✝² : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝¹ : CompleteSpace A
      inst✝ : CStarRing A
      ⊢ p₁ 0
    -/
    exact cfc_predicate_zero 𝕜
    /-
      🎉 no goals
    -/
  exists_cfc_of_predicate a ha := by
    let ψ : C(σₙ 𝕜 a, 𝕜)₀ →⋆ₙₐ[𝕜] A := comp (inrRangeEquiv 𝕜 A).symm <|
      codRestrict (cfcₙAux hp₁ a ha) _ (cfcₙAux_mem_range_inr hp₁ a ha)
    have coe_ψ (f : C(σₙ 𝕜 a, 𝕜)₀) : ψ f = cfcₙAux hp₁ a ha f :=
      congr_arg Subtype.val <| (inrRangeEquiv 𝕜 A).apply_symm_apply
        ⟨cfcₙAux hp₁ a ha f, cfcₙAux_mem_range_inr hp₁ a ha f⟩
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NonUnitalNormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedSpace 𝕜 A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      inst✝² : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝¹ : CompleteSpace A
      inst✝ : CStarRing A
      a : A
      ha : p a
      ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜) A := (↑ …
      coe_ψ : ∀ (f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜), Eq (↑(ψ f)) ((cfcₙ …
      ⊢ Exists fun φ => And (Topology.IsClosedEmbedding ⇑φ) (And (Eq (φ { toContinuo …
    -/
    refine ⟨ψ, ?isClosedEmbedding, ?map_id, fun f ↦ ?map_spec, fun f ↦ ?isStarNormal⟩
    case isClosedEmbedding =>
      apply isometry_inr (𝕜 := 𝕜) (A := A) |>.isClosedEmbedding |>.of_comp_iff.mp
      have : inr ∘ ψ = cfcₙAux hp₁ a ha := by ext1; rw [Function.comp_apply, coe_ψ]
      exact this ▸ isClosedEmbedding_cfcₙAux hp₁ a ha
    /-
      case map_id
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NonUnitalNormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedSpace 𝕜 A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      inst✝² : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝¹ : CompleteSpace A
      inst✝ : CStarRing A
      a : A
      ha : p a
      ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜) A := (↑ …
      coe_ψ : ∀ (f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜), Eq (↑(ψ f)) ((cfcₙ …
      ⊢ Eq (ψ { toContinuousMap := ContinuousMap.restrict (quasispectrum 𝕜 a) (Conti …
    -/
    case map_id => exact inr_injective (R := 𝕜) <| coe_ψ _ ▸ cfcₙAux_id hp₁ a ha
    case map_spec =>
      exact quasispectrum_eq_spectrum_inr' 𝕜 𝕜 (ψ f) ▸ coe_ψ _ ▸ spec_cfcₙAux hp₁ a ha f
    /-
      case isStarNormal
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NonUnitalNormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedSpace 𝕜 A
      inst✝⁵ : IsScalarTower 𝕜 A A
      inst✝⁴ : SMulCommClass 𝕜 A A
      inst✝³ : StarModule 𝕜 A
      p : A → Prop
      p₁ : Unitization 𝕜 A → Prop
      hp₁ : ∀ {x : A}, Iff (p₁ ↑x) (p x)
      inst✝² : ContinuousFunctionalCalculus 𝕜 p₁
      inst✝¹ : CompleteSpace A
      inst✝ : CStarRing A
      a : A
      ha : p a
      ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜) A := (↑ …
      coe_ψ : ∀ (f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜), Eq (↑(ψ f)) ((cfcₙ …
      f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
      ⊢ p (ψ f)
    -/
    case isStarNormal => exact hp₁.mp <| coe_ψ _ ▸ cfcHom_predicate (R := 𝕜) (hp₁.mpr ha) _
    /-
      🎉 no goals
    -/


instance IsStarNormal.instContinuousFunctionalCalculus {A : Type*} [CStarAlgebra A] :
    ContinuousFunctionalCalculus ℂ (IsStarNormal : A → Prop) where
  predicate_zero := isStarNormal_zero
  spectrum_nonempty a _ := spectrum.nonempty a
  exists_cfc_of_predicate a ha := by
    refine ⟨(StarAlgebra.elemental ℂ a).subtype.comp <| continuousFunctionalCalculus a,
      ?hom_isClosedEmbedding, ?hom_id, ?hom_map_spectrum, ?predicate_hom⟩
    case hom_isClosedEmbedding =>
      exact Isometry.isClosedEmbedding <|
        isometry_subtype_coe.comp <| StarAlgEquiv.isometry (continuousFunctionalCalculus a)
    /-
      case hom_id
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ Eq (((StarAlgebra.elemental Complex a).subtype.comp ↑(continuousFunctionalCa …
    -/
    case hom_id => exact congr_arg Subtype.val <| continuousFunctionalCalculus_map_id a
    case hom_map_spectrum =>
      intro f
      simp only [StarAlgHom.comp_apply, StarAlgHom.coe_coe, StarSubalgebra.coe_subtype]
      rw [← StarSubalgebra.spectrum_eq (hS := StarAlgebra.elemental.isClosed ℂ a),
        AlgEquiv.spectrum_eq (continuousFunctionalCalculus a), ContinuousMap.spectrum_eq_range]
    /-
      case predicate_hom
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ ∀ (f : ContinuousMap (↑(spectrum Complex a)) Complex), IsStarNormal (((StarA …
    -/
    case predicate_hom => exact fun f ↦ ⟨by rw [← map_star]; exact Commute.all (star f) f |>.map _⟩
    /-
      🎉 no goals
    -/


lemma cfcHom_eq_of_isStarNormal {A : Type*} [CStarAlgebra A] (a : A) [ha : IsStarNormal a] :
    cfcHom ha = (StarAlgebra.elemental ℂ a).subtype.comp (continuousFunctionalCalculus a) := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsStarNormal a
    ⊢ Eq (cfcHom ha) ((StarAlgebra.elemental Complex a).subtype.comp ↑(continuousF …
  -/
  refine cfcHom_eq_of_continuous_of_map_id ha _ ?_ ?_
  · exact continuous_subtype_val.comp <|
      (StarAlgEquiv.isometry (continuousFunctionalCalculus a)).continuous
    /-
      case refine_2
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ Eq (((StarAlgebra.elemental Complex a).subtype.comp ↑(continuousFunctionalCa …
    -/
  · simp [continuousFunctionalCalculus_map_id a]
    /-
      🎉 no goals
    -/


instance IsStarNormal.instNonUnitalContinuousFunctionalCalculus {A : Type*}
    [NonUnitalCStarAlgebra A] : NonUnitalContinuousFunctionalCalculus ℂ (IsStarNormal : A → Prop) :=
  RCLike.nonUnitalContinuousFunctionalCalculus Unitization.isStarNormal_inr


open Unitization CStarAlgebra in
lemma inr_comp_cfcₙHom_eq_cfcₙAux {A : Type*} [NonUnitalCStarAlgebra A] (a : A)
    [ha : IsStarNormal a] : (inrNonUnitalStarAlgHom ℂ A).comp (cfcₙHom ha) =
      cfcₙAux (isStarNormal_inr (R := ℂ) (A := A)) a ha := by
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    a : A
    ha : IsStarNormal a
    ⊢ Eq ((Unitization.inrNonUnitalStarAlgHom Complex A).comp (cfcₙHom ha)) (cfcₙA …
  -/
  have h (a : A) := isStarNormal_inr (R := ℂ) (A := A) (a := a)
  refine @UniqueNonUnitalContinuousFunctionalCalculus.eq_of_continuous_of_map_id
    _ _ _ _ _ _ _ _ _ _ _ inferInstance inferInstance _ (σₙ ℂ a) _ _ rfl _ _ ?_ ?_ ?_
    /-
      case refine_1
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      h : ∀ (a : A), Iff (IsStarNormal ↑a) (IsStarNormal a)
      ⊢ Continuous ⇑((Unitization.inrNonUnitalStarAlgHom Complex A).comp (cfcₙHom ha))
    -/
  · show Continuous (fun f ↦ (cfcₙHom ha f : A⁺¹)); fun_prop
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case refine_2
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      h : ∀ (a : A), Iff (IsStarNormal ↑a) (IsStarNormal a)
      ⊢ Continuous ⇑(cfcₙAux ⋯ a ha)
    -/
  · exact isClosedEmbedding_cfcₙAux @(h) a ha |>.continuous
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      h : ∀ (a : A), Iff (IsStarNormal ↑a) (IsStarNormal a)
      ⊢ Eq (((Unitization.inrNonUnitalStarAlgHom Complex A).comp (cfcₙHom ha)) { toC …
    -/
  · trans (a : A⁺¹)
      /-
        A : Type u_1
        inst✝ : NonUnitalCStarAlgebra A
        a : A
        ha : IsStarNormal a
        h : ∀ (a : A), Iff (IsStarNormal ↑a) (IsStarNormal a)
        ⊢ Eq (((Unitization.inrNonUnitalStarAlgHom Complex A).comp (cfcₙHom ha)) { toC …
      -/
    · congrm(inr $(cfcₙHom_id ha))
      /-
        🎉 no goals
      -/
      /-
        A : Type u_1
        inst✝ : NonUnitalCStarAlgebra A
        a : A
        ha : IsStarNormal a
        h : ∀ (a : A), Iff (IsStarNormal ↑a) (IsStarNormal a)
        ⊢ Eq (↑a) ((cfcₙAux ⋯ a ha) { toContinuousMap := ContinuousMap.restrict (quasi …
      -/
    · exact cfcₙAux_id @(h) a ha |>.symm
      /-
        🎉 no goals
      -/


/-- An element in a non-unital C⋆-algebra is selfadjoint if and only if it is normal and its
quasispectrum is contained in `ℝ`. -/
lemma isSelfAdjoint_iff_isStarNormal_and_quasispectrumRestricts {a : A} :
    IsSelfAdjoint a ↔ IsStarNormal a ∧ QuasispectrumRestricts a Complex.reCLM := by
  /-
    A : Type u_1
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : NonUnitalRing A
    inst✝⁴ : StarRing A
    inst✝³ : Module Complex A
    inst✝² : IsScalarTower Complex A A
    inst✝¹ : SMulCommClass Complex A A
    inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
    a : A
    ⊢ Iff (IsSelfAdjoint a) (And (IsStarNormal a) (QuasispectrumRestricts a ⇑Compl …
  -/
  refine ⟨fun ha ↦ ⟨ha.isStarNormal, ⟨fun x hx ↦ ?_, Complex.ofReal_re⟩⟩, ?_⟩
  · have := eqOn_of_cfcₙ_eq_cfcₙ <|
      (cfcₙ_star (id : ℂ → ℂ) a).symm ▸ (cfcₙ_id ℂ a).symm ▸ ha.star_eq
    /-
      case refine_1
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha : IsSelfAdjoint a
      x : Complex
      hx : Membership.mem (quasispectrum Complex a) x
      this : Set.EqOn (fun x => Star.star (id x)) id (quasispectrum Complex a)
      ⊢ Eq ((algebraMap Real Complex) (Complex.reCLM x)) x
    -/
    exact Complex.conj_eq_iff_re.mp (by simpa using this hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ⊢ And (IsStarNormal a) (QuasispectrumRestricts a ⇑Complex.reCLM) → IsSelfAdjoi …
    -/
  · rintro ⟨ha₁, ha₂⟩
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      ⊢ IsSelfAdjoint a
    -/
    rw [isSelfAdjoint_iff]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (Star.star a) a
    -/
    nth_rw 2 [← cfcₙ_id ℂ a]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (Star.star a) (cfcₙ id a)
    -/
    rw [← cfcₙ_star_id a (R := ℂ)]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (cfcₙ (fun x => Star.star x) a) (cfcₙ id a)
    -/
    refine cfcₙ_congr fun x hx ↦ ?_
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      x : Complex
      hx : Membership.mem (quasispectrum Complex a) x
      ⊢ Eq (Star.star x) (id x)
    -/
    obtain ⟨x, -, rfl⟩ := ha₂.algebraMap_image.symm ▸ hx
    /-
      case refine_2.intro.intro.intro
      A : Type u_1
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : NonUnitalRing A
      inst✝⁴ : StarRing A
      inst✝³ : Module Complex A
      inst✝² : IsScalarTower Complex A A
      inst✝¹ : SMulCommClass Complex A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : QuasispectrumRestricts a ⇑Complex.reCLM
      x : Real
      hx : Membership.mem (quasispectrum Complex a) ((algebraMap Real Complex) x)
      ⊢ Eq (Star.star ((algebraMap Real Complex) x)) (id ((algebraMap Real Complex)  …
    -/
    exact Complex.conj_ofReal _
    /-
      🎉 no goals
    -/


alias ⟨IsSelfAdjoint.quasispectrumRestricts, _⟩ :=
  isSelfAdjoint_iff_isStarNormal_and_quasispectrumRestricts


/-- A normal element whose `ℂ`-quasispectrum is contained in `ℝ` is selfadjoint. -/
lemma QuasispectrumRestricts.isSelfAdjoint (a : A) (ha : QuasispectrumRestricts a Complex.reCLM)
    [IsStarNormal a] : IsSelfAdjoint a :=
  isSelfAdjoint_iff_isStarNormal_and_quasispectrumRestricts.mpr ⟨‹_›, ha⟩


instance IsSelfAdjoint.instNonUnitalContinuousFunctionalCalculus :
    NonUnitalContinuousFunctionalCalculus ℝ (IsSelfAdjoint : A → Prop) :=
  QuasispectrumRestricts.cfc (q := IsStarNormal) (p := IsSelfAdjoint) Complex.reCLM
    Complex.isometry_ofReal.isUniformEmbedding (.zero _)
    (fun _ ↦ isSelfAdjoint_iff_isStarNormal_and_quasispectrumRestricts)


/-- An element in a C⋆-algebra is selfadjoint if and only if it is normal and its spectrum is
contained in `ℝ`. -/
lemma isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts {a : A} :
    IsSelfAdjoint a ↔ IsStarNormal a ∧ SpectrumRestricts a Complex.reCLM := by
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    a : A
    ⊢ Iff (IsSelfAdjoint a) (And (IsStarNormal a) (SpectrumRestricts a ⇑Complex.re …
  -/
  refine ⟨fun ha ↦ ⟨ha.isStarNormal, .of_rightInvOn Complex.ofReal_re fun x hx ↦ ?_⟩, ?_⟩
    /-
      case refine_1
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha : IsSelfAdjoint a
      x : Complex
      hx : Membership.mem (spectrum Complex a) x
      ⊢ Eq ((algebraMap Real Complex) (Complex.reCLM x)) x
    -/
  · have := eqOn_of_cfc_eq_cfc <| (cfc_star (id : ℂ → ℂ) a).symm ▸ (cfc_id ℂ a).symm ▸ ha.star_eq
    /-
      case refine_1
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha : IsSelfAdjoint a
      x : Complex
      hx : Membership.mem (spectrum Complex a) x
      this : Set.EqOn (fun x => Star.star (id x)) id (spectrum Complex a)
      ⊢ Eq ((algebraMap Real Complex) (Complex.reCLM x)) x
    -/
    exact Complex.conj_eq_iff_re.mp (by simpa using this hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ⊢ And (IsStarNormal a) (SpectrumRestricts a ⇑Complex.reCLM) → IsSelfAdjoint a
    -/
  · rintro ⟨ha₁, ha₂⟩
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      ⊢ IsSelfAdjoint a
    -/
    rw [isSelfAdjoint_iff]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (Star.star a) a
    -/
    nth_rw 2 [← cfc_id ℂ a]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (Star.star a) (cfc id a)
    -/
    rw [← cfc_star_id a (R := ℂ)]
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      ⊢ Eq (cfc (fun x => Star.star x) a) (cfc id a)
    -/
    refine cfc_congr fun x hx ↦ ?_
    /-
      case refine_2.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      x : Complex
      hx : Membership.mem (spectrum Complex a) x
      ⊢ Eq (Star.star x) (id x)
    -/
    obtain ⟨x, -, rfl⟩ := ha₂.algebraMap_image.symm ▸ hx
    /-
      case refine_2.intro.intro.intro
      A : Type u_1
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra Complex A
      inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
      a : A
      ha₁ : IsStarNormal a
      ha₂ : SpectrumRestricts a ⇑Complex.reCLM
      x : Real
      hx : Membership.mem (spectrum Complex a) ((algebraMap Real Complex) x)
      ⊢ Eq (Star.star ((algebraMap Real Complex) x)) (id ((algebraMap Real Complex)  …
    -/
    exact Complex.conj_ofReal _
    /-
      🎉 no goals
    -/

-- TODO: REMOVE (duplicate; see comment on `isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts`)

lemma IsSelfAdjoint.spectrumRestricts {a : A} (ha : IsSelfAdjoint a) :
    SpectrumRestricts a Complex.reCLM :=
  isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts.mp ha |>.right

-- TODO: REMOVE (duplicate; see comment on `isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts`)

/-- A normal element whose `ℂ`-spectrum is contained in `ℝ` is selfadjoint. -/
lemma SpectrumRestricts.isSelfAdjoint (a : A) (ha : SpectrumRestricts a Complex.reCLM)
    [IsStarNormal a] : IsSelfAdjoint a :=
  isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts.mpr ⟨‹_›, ha⟩


instance IsSelfAdjoint.instContinuousFunctionalCalculus :
    ContinuousFunctionalCalculus ℝ (IsSelfAdjoint : A → Prop) :=
  SpectrumRestricts.cfc (q := IsStarNormal) (p := IsSelfAdjoint) Complex.reCLM
    Complex.isometry_ofReal.isUniformEmbedding (.zero _)
    (fun _ ↦ isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts)


lemma IsSelfAdjoint.spectrum_nonempty {A : Type*} [Ring A] [StarRing A]
    [TopologicalSpace A] [Algebra ℝ A] [ContinuousFunctionalCalculus ℝ (IsSelfAdjoint : A → Prop)]
    [Nontrivial A] {a : A} (ha : IsSelfAdjoint a) : (σ ℝ a).Nonempty :=
  CFC.spectrum_nonempty ℝ a ha


lemma CFC.exists_sqrt_of_isSelfAdjoint_of_quasispectrumRestricts {A : Type*} [NonUnitalRing A]
    [StarRing A] [TopologicalSpace A] [Module ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A ]
    [NonUnitalContinuousFunctionalCalculus ℝ (IsSelfAdjoint : A → Prop)]
    {a : A} (ha₁ : IsSelfAdjoint a) (ha₂ : QuasispectrumRestricts a ContinuousMap.realToNNReal) :
    ∃ x : A, IsSelfAdjoint x ∧ QuasispectrumRestricts x ContinuousMap.realToNNReal ∧ x * x = a := by
  /-
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Module Real A
    inst✝² : IsScalarTower Real A A
    inst✝¹ : SMulCommClass Real A A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ Exists fun x => And (IsSelfAdjoint x) (And (QuasispectrumRestricts x ⇑Contin …
  -/
  use cfcₙ Real.sqrt a, cfcₙ_predicate Real.sqrt a
  /-
    case right
    A : Type u_1
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Module Real A
    inst✝² : IsScalarTower Real A A
    inst✝¹ : SMulCommClass Real A A
    inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ And (QuasispectrumRestricts (cfcₙ Real.sqrt a) ⇑ContinuousMap.realToNNReal)  …
  -/
  constructor
  · simpa only [QuasispectrumRestricts.nnreal_iff, cfcₙ_map_quasispectrum Real.sqrt a,
      Set.mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
        using fun x _ ↦ Real.sqrt_nonneg x
    /-
      case right.right
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (HMul.hMul (cfcₙ Real.sqrt a) (cfcₙ Real.sqrt a)) a
    -/
  · rw [← cfcₙ_mul ..]
    /-
      case right.right
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (cfcₙ (fun x => HMul.hMul x.sqrt x.sqrt) a) a
    -/
    nth_rw 2 [← cfcₙ_id ℝ a]
    /-
      case right.right
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (cfcₙ (fun x => HMul.hMul x.sqrt x.sqrt) a) (cfcₙ id a)
    -/
    apply cfcₙ_congr fun x hx ↦ ?_
    /-
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : Real
      hx : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul x.sqrt x.sqrt) (id x)
    -/
    rw [QuasispectrumRestricts.nnreal_iff] at ha₂
    /-
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : ∀ (x : Real), Membership.mem (quasispectrum Real a) x → LE.le 0 x
      x : Real
      hx : Membership.mem (quasispectrum Real a) x
      ⊢ Eq (HMul.hMul x.sqrt x.sqrt) (id x)
    -/
    apply ha₂ x at hx
    /-
      A : Type u_1
      inst✝⁶ : NonUnitalRing A
      inst✝⁵ : StarRing A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Module Real A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : ∀ (x : Real), Membership.mem (quasispectrum Real a) x → LE.le 0 x
      x : Real
      hx : LE.le 0 x
      ⊢ Eq (HMul.hMul x.sqrt x.sqrt) (id x)
    -/
    simp [← sq, Real.sq_sqrt hx]
    /-
      🎉 no goals
    -/


lemma nonneg_iff_isSelfAdjoint_and_quasispectrumRestricts {a : A} :
    0 ≤ a ↔ IsSelfAdjoint a ∧ QuasispectrumRestricts a ContinuousMap.realToNNReal := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Module Real A
    inst✝³ : IsScalarTower Real A A
    inst✝² : SMulCommClass Real A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (LE.le 0 a) (And (IsSelfAdjoint a) (QuasispectrumRestricts a ⇑Continuous …
  -/
  refine ⟨fun ha ↦ ⟨.of_nonneg ha, .nnreal_of_nonneg ha⟩, ?_⟩
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Module Real A
    inst✝³ : IsScalarTower Real A A
    inst✝² : SMulCommClass Real A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ And (IsSelfAdjoint a) (QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal) …
  -/
  rintro ⟨ha₁, ha₂⟩
  /-
    case intro
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Module Real A
    inst✝³ : IsScalarTower Real A A
    inst✝² : SMulCommClass Real A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ LE.le 0 a
  -/
  obtain ⟨x, hx, -, rfl⟩ := CFC.exists_sqrt_of_isSelfAdjoint_of_quasispectrumRestricts ha₁ ha₂
  /-
    case intro.intro.intro.intro
    A : Type u_1
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Module Real A
    inst✝³ : IsScalarTower Real A A
    inst✝² : SMulCommClass Real A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    x : A
    hx : IsSelfAdjoint x
    ha₁ : IsSelfAdjoint (HMul.hMul x x)
    ha₂ : QuasispectrumRestricts (HMul.hMul x x) ⇑ContinuousMap.realToNNReal
    ⊢ LE.le 0 (HMul.hMul x x)
  -/
  simpa [sq, hx.star_eq] using star_mul_self_nonneg x
  /-
    🎉 no goals
  -/


open NNReal in
instance Nonneg.instNonUnitalContinuousFunctionalCalculus :
    NonUnitalContinuousFunctionalCalculus ℝ≥0 (fun x : A ↦ 0 ≤ x) :=
  QuasispectrumRestricts.cfc (q := IsSelfAdjoint) ContinuousMap.realToNNReal
    isUniformEmbedding_subtype_val le_rfl
    (fun _ ↦ nonneg_iff_isSelfAdjoint_and_quasispectrumRestricts)


open NNReal in
lemma NNReal.spectrum_nonempty {A : Type*} [Ring A] [StarRing A] [PartialOrder A]
    [TopologicalSpace A] [Algebra ℝ≥0 A] [ContinuousFunctionalCalculus ℝ≥0 (fun x : A ↦ 0 ≤ x)]
    [Nontrivial A] {a : A} (ha : 0 ≤ a) : (spectrum ℝ≥0 a).Nonempty :=
  CFC.spectrum_nonempty ℝ≥0 a ha


lemma CFC.exists_sqrt_of_isSelfAdjoint_of_spectrumRestricts {A : Type*} [Ring A] [StarRing A]
    [TopologicalSpace A] [Algebra ℝ A] [ContinuousFunctionalCalculus ℝ (IsSelfAdjoint : A → Prop)]
    {a : A} (ha₁ : IsSelfAdjoint a) (ha₂ : SpectrumRestricts a ContinuousMap.realToNNReal) :
    ∃ x : A, IsSelfAdjoint x ∧ SpectrumRestricts x ContinuousMap.realToNNReal ∧ x ^ 2 = a := by
  /-
    A : Type u_1
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ Exists fun x => And (IsSelfAdjoint x) (And (SpectrumRestricts x ⇑ContinuousM …
  -/
  use cfc Real.sqrt a, cfc_predicate Real.sqrt a
  /-
    case right
    A : Type u_1
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : TopologicalSpace A
    inst✝¹ : Algebra Real A
    inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ And (SpectrumRestricts (cfc Real.sqrt a) ⇑ContinuousMap.realToNNReal) (Eq (H …
  -/
  constructor
  · simpa only [SpectrumRestricts.nnreal_iff, cfc_map_spectrum Real.sqrt a, Set.mem_image,
      forall_exists_index, and_imp, forall_apply_eq_imp_iff₂] using fun x _ ↦ Real.sqrt_nonneg x
    /-
      case right.right
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (HPow.hPow (cfc Real.sqrt a) 2) a
    -/
  · rw [← cfc_pow ..]
    /-
      case right.right
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (cfc (fun x => HPow.hPow x.sqrt 2) a) a
    -/
    nth_rw 2 [← cfc_id ℝ a]
    /-
      case right.right
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      ⊢ Eq (cfc (fun x => HPow.hPow x.sqrt 2) a) (cfc id a)
    -/
    apply cfc_congr fun x hx ↦ ?_
    /-
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : Real
      hx : Membership.mem (spectrum Real a) x
      ⊢ Eq (HPow.hPow x.sqrt 2) (id x)
    -/
    rw [SpectrumRestricts.nnreal_iff] at ha₂
    /-
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      x : Real
      hx : Membership.mem (spectrum Real a) x
      ⊢ Eq (HPow.hPow x.sqrt 2) (id x)
    -/
    apply ha₂ x at hx
    /-
      A : Type u_1
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : TopologicalSpace A
      inst✝¹ : Algebra Real A
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      a : A
      ha₁ : IsSelfAdjoint a
      ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      x : Real
      hx : LE.le 0 x
      ⊢ Eq (HPow.hPow x.sqrt 2) (id x)
    -/
    simp [Real.sq_sqrt hx]
    /-
      🎉 no goals
    -/


lemma nonneg_iff_isSelfAdjoint_and_spectrumRestricts {a : A} :
    0 ≤ a ↔ IsSelfAdjoint a ∧ SpectrumRestricts a ContinuousMap.realToNNReal := by
  /-
    A : Type u_1
    inst✝⁷ : Ring A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : StarRing A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ Iff (LE.le 0 a) (And (IsSelfAdjoint a) (SpectrumRestricts a ⇑ContinuousMap.r …
  -/
  refine ⟨fun ha ↦ ⟨.of_nonneg ha, .nnreal_of_nonneg ha⟩, ?_⟩
  /-
    A : Type u_1
    inst✝⁷ : Ring A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : StarRing A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ⊢ And (IsSelfAdjoint a) (SpectrumRestricts a ⇑ContinuousMap.realToNNReal) → LE …
  -/
  rintro ⟨ha₁, ha₂⟩
  /-
    case intro
    A : Type u_1
    inst✝⁷ : Ring A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : StarRing A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha₁ : IsSelfAdjoint a
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ LE.le 0 a
  -/
  obtain ⟨x, hx, -, rfl⟩ := CFC.exists_sqrt_of_isSelfAdjoint_of_spectrumRestricts ha₁ ha₂
  /-
    case intro.intro.intro.intro
    A : Type u_1
    inst✝⁷ : Ring A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : StarRing A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : TopologicalSpace A
    inst✝² : Algebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : NonnegSpectrumClass Real A
    x : A
    hx : IsSelfAdjoint x
    ha₁ : IsSelfAdjoint (HPow.hPow x 2)
    ha₂ : SpectrumRestricts (HPow.hPow x 2) ⇑ContinuousMap.realToNNReal
    ⊢ LE.le 0 (HPow.hPow x 2)
  -/
  simpa [sq, hx.star_eq] using star_mul_self_nonneg x
  /-
    🎉 no goals
  -/


open NNReal in
instance Nonneg.instContinuousFunctionalCalculus :
    ContinuousFunctionalCalculus ℝ≥0 (fun x : A ↦ 0 ≤ x) :=
  SpectrumRestricts.cfc (q := IsSelfAdjoint) ContinuousMap.realToNNReal
    isUniformEmbedding_subtype_val le_rfl (fun _ ↦ nonneg_iff_isSelfAdjoint_and_spectrumRestricts)


lemma SpectrumRestricts.nnreal_iff_nnnorm {a : A} {t : ℝ≥0} (ha : IsSelfAdjoint a) (ht : ‖a‖₊ ≤ t) :
    SpectrumRestricts a ContinuousMap.realToNNReal ↔ ‖algebraMap ℝ A t - a‖₊ ≤ t := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    t : NNReal
    ha : IsSelfAdjoint a
    ht : LE.le (NNNorm.nnnorm a) t
    ⊢ Iff (SpectrumRestricts a ⇑ContinuousMap.realToNNReal) (LE.le (NNNorm.nnnorm  …
  -/
  have : IsSelfAdjoint (algebraMap ℝ A t - a) := IsSelfAdjoint.algebraMap A (.all (t : ℝ)) |>.sub ha
  rw [← ENNReal.coe_le_coe, ← IsSelfAdjoint.spectralRadius_eq_nnnorm,
    ← SpectrumRestricts.spectralRadius_eq (f := Complex.reCLM)] at ht ⊢
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      t : NNReal
      ha : IsSelfAdjoint a
      ht : LE.le (spectralRadius Real a) ↑t
      this : IsSelfAdjoint (HSub.hSub ((algebraMap Real A) ↑t) a)
      ⊢ Iff (SpectrumRestricts a ⇑ContinuousMap.realToNNReal) (LE.le (spectralRadius …
    -/
  · exact SpectrumRestricts.nnreal_iff_spectralRadius_le ht
    /-
      🎉 no goals
    -/
  all_goals
    try apply IsSelfAdjoint.spectrumRestricts
    assumption


lemma SpectrumRestricts.nnreal_add {a b : A} (ha₁ : IsSelfAdjoint a)
    (hb₁ : IsSelfAdjoint b) (ha₂ : SpectrumRestricts a ContinuousMap.realToNNReal)
    (hb₂ : SpectrumRestricts b ContinuousMap.realToNNReal) :
    SpectrumRestricts (a + b) ContinuousMap.realToNNReal := by
  rw [SpectrumRestricts.nnreal_iff_nnnorm (ha₁.add hb₁) (nnnorm_add_le a b), NNReal.coe_add,
    map_add, add_sub_add_comm]
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a b : A
    ha₁ : IsSelfAdjoint a
    hb₁ : IsSelfAdjoint b
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    hb₂ : SpectrumRestricts b ⇑ContinuousMap.realToNNReal
    ⊢ LE.le (NNNorm.nnnorm (HAdd.hAdd (HSub.hSub ((algebraMap Real A) ↑(NNNorm.nnn …
  -/
  refine nnnorm_add_le _ _ |>.trans ?_
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a b : A
    ha₁ : IsSelfAdjoint a
    hb₁ : IsSelfAdjoint b
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    hb₂ : SpectrumRestricts b ⇑ContinuousMap.realToNNReal
    ⊢ LE.le (HAdd.hAdd (NNNorm.nnnorm (HSub.hSub ((algebraMap Real A) ↑(NNNorm.nnn …
  -/
  gcongr
  /-
    case h₁
    A : Type u_1
    inst✝ : CStarAlgebra A
    a b : A
    ha₁ : IsSelfAdjoint a
    hb₁ : IsSelfAdjoint b
    ha₂ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    hb₂ : SpectrumRestricts b ⇑ContinuousMap.realToNNReal
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub ((algebraMap Real A) ↑(NNNorm.nnnorm a)) a)) …
  -/
  all_goals rw [← SpectrumRestricts.nnreal_iff_nnnorm] <;> first | rfl | assumption
  /-
    🎉 no goals
  -/


lemma IsSelfAdjoint.sq_spectrumRestricts {a : A} (ha : IsSelfAdjoint a) :
    SpectrumRestricts (a ^ 2) ContinuousMap.realToNNReal := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ⊢ SpectrumRestricts (HPow.hPow a 2) ⇑ContinuousMap.realToNNReal
  -/
  rw [SpectrumRestricts.nnreal_iff, ← cfc_id (R := ℝ) a, ← cfc_pow .., cfc_map_spectrum ..]
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ⊢ ∀ (x : Real), Membership.mem (Set.image (fun x => HPow.hPow (id x) 2) (spect …
  -/
  rintro - ⟨x, -, rfl⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    x : Real
    ⊢ LE.le 0 ((fun x => HPow.hPow (id x) 2) x)
  -/
  exact sq_nonneg x
  /-
    🎉 no goals
  -/


lemma SpectrumRestricts.eq_zero_of_neg {a : A} (ha : IsSelfAdjoint a)
    (ha₁ : SpectrumRestricts a ContinuousMap.realToNNReal)
    (ha₂ : SpectrumRestricts (-a) ContinuousMap.realToNNReal) :
    a = 0 := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ha₂ : SpectrumRestricts (Neg.neg a) ⇑ContinuousMap.realToNNReal
    ⊢ Eq a 0
  -/
  nontriviality A
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ha₂ : SpectrumRestricts (Neg.neg a) ⇑ContinuousMap.realToNNReal
    a✝ : Nontrivial A
    ⊢ Eq a 0
  -/
  rw [SpectrumRestricts.nnreal_iff] at ha₁ ha₂
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real (Neg.neg a)) x → LE.le 0 x
    a✝ : Nontrivial A
    ⊢ Eq a 0
  -/
  apply CFC.eq_zero_of_spectrum_subset_zero (R := ℝ) a
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real (Neg.neg a)) x → LE.le 0 x
    a✝ : Nontrivial A
    ⊢ HasSubset.Subset (spectrum Real a) (Singleton.singleton 0)
  -/
  rw [Set.subset_singleton_iff]
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real (Neg.neg a)) x → LE.le 0 x
    a✝ : Nontrivial A
    ⊢ ∀ (y : Real), Membership.mem (spectrum Real a) y → Eq y 0
  -/
  simp only [← spectrum.neg_eq, Set.mem_neg] at ha₂
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    a✝ : Nontrivial A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) (Neg.neg x) → LE.le 0 x
    ⊢ ∀ (y : Real), Membership.mem (spectrum Real a) y → Eq y 0
  -/
  peel ha₁ with x hx _
  /-
    case h.h
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ha₁ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    a✝ : Nontrivial A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) (Neg.neg x) → LE.le 0 x
    x : Real
    hx : Membership.mem (spectrum Real a) x
    this : LE.le 0 x
    ⊢ Eq x 0
  -/
  linarith [ha₂ (-x) ((neg_neg x).symm ▸ hx)]
  /-
    🎉 no goals
  -/


lemma SpectrumRestricts.smul_of_nonneg {A : Type*} [Ring A] [Algebra ℝ A] {a : A}
    (ha : SpectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ} (hr : 0 ≤ r) :
    SpectrumRestricts (r • a) ContinuousMap.realToNNReal := by
  /-
    A : Type u_2
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : Real
    hr : LE.le 0 r
    ⊢ SpectrumRestricts (HSMul.hSMul r a) ⇑ContinuousMap.realToNNReal
  -/
  rw [SpectrumRestricts.nnreal_iff] at ha ⊢
  /-
    A : Type u_2
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    r : Real
    hr : LE.le 0 r
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real (HSMul.hSMul r a)) x → LE.le 0 x
  -/
  nontriviality A
  /-
    A : Type u_2
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    r : Real
    hr : LE.le 0 r
    a✝ : Nontrivial A
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real (HSMul.hSMul r a)) x → LE.le 0 x
  -/
  intro x hx
  /-
    A : Type u_2
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    r : Real
    hr : LE.le 0 r
    a✝ : Nontrivial A
    x : Real
    hx : Membership.mem (spectrum Real (HSMul.hSMul r a)) x
    ⊢ LE.le 0 x
  -/
  by_cases hr' : r = 0
    /-
      case pos
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      r : Real
      hr : LE.le 0 r
      a✝ : Nontrivial A
      x : Real
      hx : Membership.mem (spectrum Real (HSMul.hSMul r a)) x
      hr' : Eq r 0
      ⊢ LE.le 0 x
    -/
  · simp [hr'] at hx ⊢
    /-
      case pos
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      r : Real
      hr : LE.le 0 r
      a✝ : Nontrivial A
      x : Real
      hr' : Eq r 0
      hx : Eq x 0
      ⊢ LE.le 0 x
    -/
    exact hx.symm.le
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      r : Real
      hr : LE.le 0 r
      a✝ : Nontrivial A
      x : Real
      hx : Membership.mem (spectrum Real (HSMul.hSMul r a)) x
      hr' : Not (Eq r 0)
      ⊢ LE.le 0 x
    -/
  · lift r to ℝˣ using IsUnit.mk0 r hr'
    /-
      case neg.intro
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      a✝ : Nontrivial A
      x : Real
      r : Units Real
      hr : LE.le 0 ↑r
      hx : Membership.mem (spectrum Real (HSMul.hSMul (↑r) a)) x
      hr' : Not (Eq (↑r) 0)
      ⊢ LE.le 0 x
    -/
    rw [← Units.smul_def, spectrum.unit_smul_eq_smul, Set.mem_smul_set_iff_inv_smul_mem] at hx
    /-
      case neg.intro
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      a✝ : Nontrivial A
      x : Real
      r : Units Real
      hr : LE.le 0 ↑r
      hx : Membership.mem (spectrum Real a) (HSMul.hSMul (Inv.inv r) x)
      hr' : Not (Eq (↑r) 0)
      ⊢ LE.le 0 x
    -/
    refine le_of_smul_le_smul_left ?_ (inv_pos.mpr <| lt_of_le_of_ne hr <| ne_comm.mpr hr')
    /-
      case neg.intro
      A : Type u_2
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      ha : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      a✝ : Nontrivial A
      x : Real
      r : Units Real
      hr : LE.le 0 ↑r
      hx : Membership.mem (spectrum Real a) (HSMul.hSMul (Inv.inv r) x)
      hr' : Not (Eq (↑r) 0)
      ⊢ LE.le (HSMul.hSMul (Inv.inv ↑r) 0) (HSMul.hSMul (Inv.inv ↑r) x)
    -/
    simpa [Units.smul_def] using ha _ hx
    /-
      🎉 no goals
    -/


lemma spectrum_star_mul_self_nonneg {b : A} : ∀ x ∈ spectrum ℝ (star b * b), 0 ≤ x := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real (HMul.hMul (Star.star b) b)) x → …
  -/
  set a := star b * b
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  have a_def : a = star b * b := rfl
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  let a_neg : A := cfc (fun x ↦ (- ContinuousMap.id ℝ ⊔ 0) x) a
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  set c := b * a_neg
  have h_eq_a_neg : - (star c * c) = a_neg ^ 3 := by
    simp only [c, a_neg, star_mul]
    rw [← mul_assoc, mul_assoc _ _ b, ← cfc_star, ← cfc_id' ℝ (star b * b), a_def, ← neg_mul]
    rw [← cfc_mul _ _ (star b * b) (by simp; fun_prop), neg_mul]
    simp only [ContinuousMap.coe_neg, ContinuousMap.coe_id, Pi.sup_apply, Pi.neg_apply,
      star_trivial]
    rw [← cfc_mul .., ← cfc_neg .., ← cfc_pow ..]
    congr
    ext x
    by_cases hx : x ≤ 0
    · rw [← neg_nonneg] at hx
      simp [sup_eq_left.mpr hx, pow_succ]
    · rw [not_le, ← neg_neg_iff_pos] at hx
      simp [sup_eq_right.mpr hx.le]
  have h_c_spec₀ : SpectrumRestricts (- (star c * c)) (ContinuousMap.realToNNReal ·) := by
    simp only [SpectrumRestricts.nnreal_iff, h_eq_a_neg]
    rw [← cfc_pow _ _ (ha := .star_mul_self b)]
    simp only [a, cfc_map_spectrum (R := ℝ) (fun x => (-ContinuousMap.id ℝ ⊔ 0) x ^ 3) (star b * b)]
    rintro - ⟨x, -, rfl⟩
    simp
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (Neg.neg (HMul.hMul (Star.star c) c)) (HPow.hPow a_neg 3)
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  have c_eq := star_mul_self_add_self_mul_star c
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (Neg.neg (HMul.hMul (Star.star c) c)) (HPow.hPow a_neg 3)
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HAdd.hAdd (HMul.hMul (Star.star c) c) (HMul.hMul c (Star.star c)))  …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  rw [← eq_sub_iff_add_eq', sub_eq_add_neg, ← sq, ← sq] at c_eq
  have h_c_spec₁ : SpectrumRestricts (c * star c) ContinuousMap.realToNNReal := by
    rw [c_eq]
    refine SpectrumRestricts.nnreal_add ?_ ?_ ?_ h_c_spec₀
    · exact IsSelfAdjoint.smul (by rfl) <| ((ℜ c).prop.pow 2).add ((ℑ c).prop.pow 2)
    · exact (IsSelfAdjoint.star_mul_self c).neg
    · rw [← Nat.cast_smul_eq_nsmul ℝ]
      refine (ℜ c).2.sq_spectrumRestricts.nnreal_add ((ℜ c).2.pow 2) ((ℑ c).2.pow 2)
        (ℑ c).2.sq_spectrumRestricts |>.smul_of_nonneg <| by norm_num
  have h_c_spec₂ : SpectrumRestricts (star c * c) ContinuousMap.realToNNReal := by
    rw [SpectrumRestricts.nnreal_iff] at h_c_spec₁ ⊢
    intro x hx
    replace hx := Set.subset_diff_union _ {(0 : ℝ)} hx
    rw [spectrum.nonzero_mul_eq_swap_mul, Set.diff_union_self, Set.union_singleton,
      Set.mem_insert_iff] at hx
    obtain (rfl | hx) := hx
    exacts [le_rfl, h_c_spec₁ x hx]
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (Neg.neg (HMul.hMul (Star.star c) c)) (HPow.hPow a_neg 3)
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  rw [h_c_spec₂.eq_zero_of_neg (.star_mul_self c) h_c_spec₀, neg_zero] at h_eq_a_neg
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq 0 (HPow.hPow a_neg 3)
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  simp only [a_neg] at h_eq_a_neg
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq 0 (HPow.hPow (cfc (fun x => (Max.max (Neg.neg (ContinuousMap.i …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  rw [← cfc_pow _ _ (ha := .star_mul_self b), ← cfc_zero a (R := ℝ)] at h_eq_a_neg
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
  -/
  intro x hx
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    x : Real
    hx : Membership.mem (spectrum Real a) x
    ⊢ LE.le 0 x
  -/
  by_contra! hx'
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    x : Real
    hx : Membership.mem (spectrum Real a) x
    hx' : LT.lt x 0
    ⊢ False
  -/
  rw [← neg_pos] at hx'
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    x : Real
    hx : Membership.mem (spectrum Real a) x
    hx' : LT.lt 0 (Neg.neg x)
    ⊢ False
  -/
  apply (pow_pos hx' 3).ne
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    x : Real
    hx : Membership.mem (spectrum Real a) x
    hx' : LT.lt 0 (Neg.neg x)
    ⊢ Eq 0 (HPow.hPow (Neg.neg x) 3)
  -/
  have h_eqOn := eqOn_of_cfc_eq_cfc (ha := IsSelfAdjoint.star_mul_self b) h_eq_a_neg
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    b : A
    a : A := HMul.hMul (Star.star b) b
    a_def : Eq a (HMul.hMul (Star.star b) b)
    a_neg : A := cfc (fun x => (Max.max (Neg.neg (ContinuousMap.id Real)) 0) x) a
    c : A := HMul.hMul b a_neg
    h_eq_a_neg : Eq (cfc 0 a) (cfc (fun x => HPow.hPow ((Max.max (Neg.neg (Continu …
    h_c_spec₀ : SpectrumRestricts (Neg.neg (HMul.hMul (Star.star c) c)) fun x => C …
    c_eq : Eq (HMul.hMul c (Star.star c)) (HAdd.hAdd (HSMul.hSMul 2 (HAdd.hAdd (HP …
    h_c_spec₁ : SpectrumRestricts (HMul.hMul c (Star.star c)) ⇑ContinuousMap.realT …
    h_c_spec₂ : SpectrumRestricts (HMul.hMul (Star.star c) c) ⇑ContinuousMap.realT …
    x : Real
    hx : Membership.mem (spectrum Real a) x
    hx' : LT.lt 0 (Neg.neg x)
    h_eqOn : Set.EqOn 0 (fun x => HPow.hPow ((Max.max (Neg.neg (ContinuousMap.id R …
    ⊢ Eq 0 (HPow.hPow (Neg.neg x) 3)
  -/
  simpa [sup_eq_left.mpr hx'.le] using h_eqOn hx
  /-
    🎉 no goals
  -/


lemma IsSelfAdjoint.coe_mem_spectrum_complex {A : Type*} [TopologicalSpace A] [Ring A]
    [StarRing A] [Algebra ℂ A] [ContinuousFunctionalCalculus ℂ (IsStarNormal : A → Prop)]
    {a : A} {x : ℝ} (ha : IsSelfAdjoint a := by cfc_tac) :
    (x : ℂ) ∈ spectrum ℂ a ↔ x ∈ spectrum ℝ a := by
  /-
    A : Type u_2
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    a : A
    x : Real
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Iff (Membership.mem (spectrum Complex a) ↑x) (Membership.mem (spectrum Real  …
  -/
  simp [← ha.spectrumRestricts.algebraMap_image]
  /-
    🎉 no goals
  -/


instance CStarAlgebra.instNonnegSpectrumClass : NonnegSpectrumClass ℝ A :=
  .of_spectrum_nonneg fun a ha ↦ by
    /-
      A : Type u_1
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    -/
    rw [StarOrderedRing.nonneg_iff] at ha
    induction ha using AddSubmonoid.closure_induction with
    | mem x hx =>
      obtain ⟨b, rfl⟩ := hx
      exact spectrum_star_mul_self_nonneg
    | one =>
      nontriviality A
      simp
    | mul x y x_mem y_mem hx hy =>
      rw [← SpectrumRestricts.nnreal_iff] at hx hy ⊢
      rw [← StarOrderedRing.nonneg_iff] at x_mem y_mem
      exact hx.nnreal_add (.of_nonneg x_mem) (.of_nonneg y_mem) hy


open ComplexOrder in
instance CStarAlgebra.instNonnegSpectrumClassComplexUnital : NonnegSpectrumClass ℂ A where
  quasispectrum_nonneg_of_nonneg a ha x := by
    /-
      A : Type u_1
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x : Complex
      ⊢ Membership.mem (quasispectrum Complex a) x → LE.le 0 x
    -/
    rw [mem_quasispectrum_iff]
    /-
      A : Type u_1
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x : Complex
      ⊢ Or (Eq x 0) (Membership.mem (spectrum Complex a) x) → LE.le 0 x
    -/
    refine (Or.elim · ge_of_eq fun hx ↦ ?_)
    /-
      A : Type u_1
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x : Complex
      x✝ : Or (Eq x 0) (Membership.mem (spectrum Complex a) x)
      hx : Membership.mem (spectrum Complex a) x
      ⊢ LE.le 0 x
    -/
    obtain ⟨y, hy, rfl⟩ := (IsSelfAdjoint.of_nonneg ha).spectrumRestricts.algebraMap_image ▸ hx
    /-
      case intro.intro
      A : Type u_1
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      y : Real
      hy : Membership.mem (spectrum Real a) y
      x✝ : Or (Eq ((algebraMap Real Complex) y) 0) (Membership.mem (spectrum Complex …
      hx : Membership.mem (spectrum Complex a) ((algebraMap Real Complex) y)
      ⊢ LE.le 0 ((algebraMap Real Complex) y)
    -/
    simpa using spectrum_nonneg_of_nonneg ha hy
    /-
      🎉 no goals
    -/


/-- The partial order on a unital C⋆-algebra defined by `x ≤ y` if and only if `y - x` is
selfadjoint and has nonnegative spectrum.

This is not declared as an instance because one may already have a partial order with better
definitional properties. However, it can be useful to invoke this as an instance in proofs. -/
@[reducible]
def CStarAlgebra.spectralOrder : PartialOrder A where
  le x y := IsSelfAdjoint (y - x) ∧ SpectrumRestricts (y - x) ContinuousMap.realToNNReal
  le_refl := by
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      ⊢ ∀ (a : A), LE.le a a
    -/
    simp only [sub_self, IsSelfAdjoint.zero, true_and, forall_const]
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      ⊢ SpectrumRestricts 0 ⇑ContinuousMap.realToNNReal
    -/
    rw [SpectrumRestricts.nnreal_iff]
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real 0) x → LE.le 0 x
    -/
    nontriviality A
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      a✝ : Nontrivial A
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real 0) x → LE.le 0 x
    -/
    simp
    /-
      🎉 no goals
    -/
  le_antisymm x y hxy hyx := by
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      x y : A
      hxy : LE.le x y
      hyx : LE.le y x
      ⊢ Eq x y
    -/
    rw [← sub_eq_zero]
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      x y : A
      hxy : LE.le x y
      hyx : LE.le y x
      ⊢ Eq (HSub.hSub x y) 0
    -/
        /-
          A : Type u_1
          inst✝ : CStarAlgebra A
          x y z : A
          hxy : LE.le x y
          hyz : LE.le y z
          ⊢ IsSelfAdjoint (HSub.hSub z x)
        -/
        /-
          🎉 no goals
        -/
    exact hyx.2.eq_zero_of_neg hyx.1 (neg_sub x y ▸ hxy.2)
                                        /-
                                          🎉 no goals
                                        -/
    /-
      🎉 no goals
    -/
  le_trans x y z hxy hyz :=
    ⟨by simpa using hyz.1.add hxy.1, by simpa using hyz.2.nnreal_add hyz.1 hxy.1 hxy.2⟩


/-- The `CStarAlgebra.spectralOrder` on a unital C⋆-algebra is a `StarOrderedRing`. -/
lemma CStarAlgebra.spectralOrderedRing : @StarOrderedRing A _ (CStarAlgebra.spectralOrder A) _ :=
  let _ := CStarAlgebra.spectralOrder A
  { le_iff := by
      /-
        A : Type u_1
        inst✝ : CStarAlgebra A
        x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
        ⊢ ∀ (x y : A), Iff (LE.le x y) (Exists fun p => And (Membership.mem (AddSubmon …
      -/
      intro x y
      /-
        A : Type u_1
        inst✝ : CStarAlgebra A
        x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
        x y : A
        ⊢ Iff (LE.le x y) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
      -/
      constructor
        /-
          case mp
          A : Type u_1
          inst✝ : CStarAlgebra A
          x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
          x y : A
          ⊢ LE.le x y → Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.r …
        -/
      · intro h
        /-
          case mp
          A : Type u_1
          inst✝ : CStarAlgebra A
          x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
          x y : A
          h : LE.le x y
          ⊢ Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s = …
        -/
        obtain ⟨s, hs₁, _, hs₂⟩ := CFC.exists_sqrt_of_isSelfAdjoint_of_spectrumRestricts h.1 h.2
        /-
          case mp.intro.intro.intro
          A : Type u_1
          inst✝ : CStarAlgebra A
          x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
          x y : A
          h : LE.le x y
          s : A
          hs₁ : IsSelfAdjoint s
          left✝ : SpectrumRestricts s ⇑ContinuousMap.realToNNReal
          hs₂ : Eq (HPow.hPow s 2) (HSub.hSub y x)
          ⊢ Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s = …
        -/
        refine ⟨s ^ 2, ?_, by rwa [eq_sub_iff_add_eq', eq_comm] at hs₂⟩
        /-
          case mp.intro.intro.intro
          A : Type u_1
          inst✝ : CStarAlgebra A
          x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
          x y : A
          h : LE.le x y
          s : A
          hs₁ : IsSelfAdjoint s
          left✝ : SpectrumRestricts s ⇑ContinuousMap.realToNNReal
          hs₂ : Eq (HPow.hPow s 2) (HSub.hSub y x)
          ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star.sta …
        -/
        exact AddSubmonoid.subset_closure ⟨s, by simp [hs₁.star_eq, sq]⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr
          A : Type u_1
          inst✝ : CStarAlgebra A
          x✝ : PartialOrder A := CStarAlgebra.spectralOrder A
          x y : A
          ⊢ (Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s  …
        -/
      · rintro ⟨p, hp, rfl⟩
        suffices IsSelfAdjoint p ∧ SpectrumRestricts p ContinuousMap.realToNNReal from
          ⟨by simpa using this.1, by simpa using this.2⟩
        induction hp using AddSubmonoid.closure_induction with
        | mem x hx =>
          obtain ⟨s, rfl⟩ := hx
          refine ⟨IsSelfAdjoint.star_mul_self s, ?_⟩
          rw [SpectrumRestricts.nnreal_iff]
          exact spectrum_star_mul_self_nonneg
        | one =>
          rw [SpectrumRestricts.nnreal_iff]
          nontriviality A
          simp
        | mul x y _ _ hx hy =>
          exact ⟨hx.1.add hy.1, hx.2.nnreal_add hx.1 hy.1 hy.2⟩ }


open scoped CStarAlgebra in
instance CStarAlgebra.instNonnegSpectrumClass' : NonnegSpectrumClass ℝ A where
  quasispectrum_nonneg_of_nonneg a ha := by
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      ⊢ ∀ (x : Real), Membership.mem (quasispectrum Real a) x → LE.le 0 x
    -/
    rw [Unitization.quasispectrum_eq_spectrum_inr' _ ℂ]
    -- should this actually be an instance on the `Unitization`? (probably scoped)
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real ↑a) x → LE.le 0 x
    -/
    let _ := CStarAlgebra.spectralOrder A⁺¹
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real ↑a) x → LE.le 0 x
    -/
    have := CStarAlgebra.spectralOrderedRing A⁺¹
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this : StarOrderedRing (Unitization Complex A)
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real ↑a) x → LE.le 0 x
    -/
    apply spectrum_nonneg_of_nonneg
    /-
      case ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : LE.le 0 a
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this : StarOrderedRing (Unitization Complex A)
      ⊢ LE.le 0 ↑a
    -/
    rw [StarOrderedRing.nonneg_iff] at ha ⊢
    /-
      case ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this : StarOrderedRing (Unitization Complex A)
      ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star.sta …
    -/
    have := AddSubmonoid.mem_map_of_mem (Unitization.inrNonUnitalStarAlgHom ℂ A) ha
    /-
      case ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this✝ : StarOrderedRing (Unitization Complex A)
      this : Membership.mem (AddSubmonoid.map (Unitization.inrNonUnitalStarAlgHom Co …
      ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star.sta …
    -/
    rw [AddMonoidHom.map_mclosure, ← Set.range_comp] at this
    /-
      case ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this✝ : StarOrderedRing (Unitization Complex A)
      this : Membership.mem (AddSubmonoid.closure (Set.range (Function.comp ⇑(Unitiz …
      ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star.sta …
    -/
    apply AddSubmonoid.closure_mono ?_ this
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this✝ : StarOrderedRing (Unitization Complex A)
      this : Membership.mem (AddSubmonoid.closure (Set.range (Function.comp ⇑(Unitiz …
      ⊢ HasSubset.Subset (Set.range (Function.comp ⇑(Unitization.inrNonUnitalStarAlg …
    -/
    rintro _ ⟨s, rfl⟩
    /-
      case intro
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      a : A
      ha : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
      x✝ : PartialOrder (Unitization Complex A) := CStarAlgebra.spectralOrder (Uniti …
      this✝ : StarOrderedRing (Unitization Complex A)
      this : Membership.mem (AddSubmonoid.closure (Set.range (Function.comp ⇑(Unitiz …
      s : A
      ⊢ Membership.mem (Set.range fun s => HMul.hMul (Star.star s) s) (Function.comp …
    -/
    exact ⟨s, by simp⟩
    /-
      🎉 no goals
    -/


lemma cfcHom_real_eq_restrict {a : A} (ha : IsSelfAdjoint a) :
    cfcHom ha =
      ha.spectrumRestricts.starAlgHom (R := ℝ) (S := ℂ)
        (cfcHom ha.isStarNormal) (f := Complex.reCLM) :=
  ha.spectrumRestricts.cfcHom_eq_restrict _ Complex.isometry_ofReal.isUniformEmbedding
    ha ha.isStarNormal


lemma cfc_real_eq_complex {a : A} (f : ℝ → ℝ) (ha : IsSelfAdjoint a := by cfc_tac)  :
    cfc f a = cfc (fun x ↦ f x.re : ℂ → ℂ) a := by
  /-
    A : Type u_1
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra Complex A
    inst✝¹ : ContinuousFunctionalCalculus Complex IsStarNormal
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    a : A
    f : Real → Real
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (cfc f a) (cfc (fun x => ↑(f x.re)) a)
  -/
  replace ha : IsSelfAdjoint a := ha -- hack to avoid issues caused by autoParam
  exact ha.spectrumRestricts.cfc_eq_restrict (f := Complex.reCLM)
    Complex.isometry_ofReal.isUniformEmbedding ha ha.isStarNormal f


lemma cfcₙHom_real_eq_restrict {a : A} (ha : IsSelfAdjoint a) :
    cfcₙHom ha = (ha.quasispectrumRestricts.2).nonUnitalStarAlgHom (cfcₙHom ha.isStarNormal)
      (R := ℝ) (S := ℂ) (f := Complex.reCLM) :=
  ha.quasispectrumRestricts.2.cfcₙHom_eq_restrict _ Complex.isometry_ofReal.isUniformEmbedding
    ha ha.isStarNormal


lemma cfcₙ_real_eq_complex {a : A} (f : ℝ → ℝ) (ha : IsSelfAdjoint a := by cfc_tac)  :
    cfcₙ f a = cfcₙ (fun x ↦ f x.re : ℂ → ℂ) a := by
  /-
    A : Type u_1
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : StarRing A
    inst✝⁴ : Module Complex A
    inst✝³ : IsScalarTower Complex A A
    inst✝² : SMulCommClass Complex A A
    inst✝¹ : NonUnitalContinuousFunctionalCalculus Complex IsStarNormal
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    a : A
    f : Real → Real
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (cfcₙ f a) (cfcₙ (fun x => ↑(f x.re)) a)
  -/
  replace ha : IsSelfAdjoint a := ha -- hack to avoid issues caused by autoParam
  exact ha.quasispectrumRestricts.2.cfcₙ_eq_restrict (f := Complex.reCLM)
    Complex.isometry_ofReal.isUniformEmbedding ha ha.isStarNormal f


lemma cfcHom_nnreal_eq_restrict {a : A} (ha : 0 ≤ a) :
    cfcHom ha = (SpectrumRestricts.nnreal_of_nonneg ha).starAlgHom
      (cfcHom (IsSelfAdjoint.of_nonneg ha)) := by
  /-
    A : Type u_1
    inst✝¹⁰ : TopologicalSpace A
    inst✝⁹ : Ring A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : Algebra Real A
    inst✝⁴ : TopologicalRing A
    inst✝³ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : ContinuousFunctionalCalculus NNReal fun x => LE.le 0 x
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    ha : LE.le 0 a
    ⊢ Eq (cfcHom ha) (SpectrumRestricts.starAlgHom (cfcHom ⋯) ⋯)
  -/
  apply (SpectrumRestricts.nnreal_of_nonneg ha).cfcHom_eq_restrict _ isUniformEmbedding_subtype_val
  /-
    🎉 no goals
  -/


lemma cfc_nnreal_eq_real {a : A} (f : ℝ≥0 → ℝ≥0) (ha : 0 ≤ a := by cfc_tac)  :
    cfc f a = cfc (fun x ↦ f x.toNNReal : ℝ → ℝ) a := by
  /-
    A : Type u_1
    inst✝¹⁰ : TopologicalSpace A
    inst✝⁹ : Ring A
    inst✝⁸ : PartialOrder A
    inst✝⁷ : StarRing A
    inst✝⁶ : StarOrderedRing A
    inst✝⁵ : Algebra Real A
    inst✝⁴ : TopologicalRing A
    inst✝³ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : ContinuousFunctionalCalculus NNReal fun x => LE.le 0 x
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    f : NNReal → NNReal
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfc f a) (cfc (fun x => ↑(f x.toNNReal)) a)
  -/
  replace ha : 0 ≤ a := ha -- hack to avoid issues caused by autoParam
  apply (SpectrumRestricts.nnreal_of_nonneg ha).cfc_eq_restrict _
    isUniformEmbedding_subtype_val ha (.of_nonneg ha)


lemma cfcₙHom_nnreal_eq_restrict {a : A} (ha : 0 ≤ a) :
    cfcₙHom ha = (QuasispectrumRestricts.nnreal_of_nonneg ha).nonUnitalStarAlgHom
      (cfcₙHom (IsSelfAdjoint.of_nonneg ha)) := by
  apply (QuasispectrumRestricts.nnreal_of_nonneg ha).cfcₙHom_eq_restrict _
    isUniformEmbedding_subtype_val


lemma cfcₙ_nnreal_eq_real {a : A} (f : ℝ≥0 → ℝ≥0) (ha : 0 ≤ a := by cfc_tac)  :
    cfcₙ f a = cfcₙ (fun x ↦ f x.toNNReal : ℝ → ℝ) a := by
  /-
    A : Type u_1
    inst✝¹² : TopologicalSpace A
    inst✝¹¹ : NonUnitalRing A
    inst✝¹⁰ : PartialOrder A
    inst✝⁹ : StarRing A
    inst✝⁸ : StarOrderedRing A
    inst✝⁷ : Module Real A
    inst✝⁶ : TopologicalRing A
    inst✝⁵ : IsScalarTower Real A A
    inst✝⁴ : SMulCommClass Real A A
    inst✝³ : NonUnitalContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝² : NonUnitalContinuousFunctionalCalculus NNReal fun x => LE.le 0 x
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    inst✝ : NonnegSpectrumClass Real A
    a : A
    f : NNReal → NNReal
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Eq (cfcₙ f a) (cfcₙ (fun x => ↑(f x.toNNReal)) a)
  -/
  replace ha : 0 ≤ a := ha -- hack to avoid issues caused by autoParam
  apply (QuasispectrumRestricts.nnreal_of_nonneg ha).cfcₙ_eq_restrict _
    isUniformEmbedding_subtype_val ha (.of_nonneg ha)


open scoped NonUnitalContinuousFunctionalCalculus in
/-- This lemma requires a lot from type class synthesis, and so one should instead favor the bespoke
versions for `ℝ≥0`, `ℝ`, and `ℂ`. -/
lemma Unitization.cfcₙ_eq_cfc_inr {R : Type*} [Semifield R] [StarRing R] [MetricSpace R]
    [TopologicalSemiring R] [ContinuousStar R] [Module R A] [IsScalarTower R A A]
    [SMulCommClass R A A] [CompleteSpace R] [Algebra R ℂ] [IsScalarTower R ℂ A]
    {p : A → Prop} {p' : A⁺¹ → Prop} [NonUnitalContinuousFunctionalCalculus R p]
    [ContinuousFunctionalCalculus R p']
    [UniqueNonUnitalContinuousFunctionalCalculus R (Unitization ℂ A)]
    (hp : ∀ {a : A}, p' (a : A⁺¹) ↔ p a) (a : A) (f : R → R) (hf₀ : f 0 = 0 := by cfc_zero_tac) :
    cfcₙ f a = cfc f (a : A⁺¹) := by
  /-
    A : Type u_1
    inst✝¹⁴ : NonUnitalCStarAlgebra A
    R : Type u_2
    inst✝¹³ : Semifield R
    inst✝¹² : StarRing R
    inst✝¹¹ : MetricSpace R
    inst✝¹⁰ : TopologicalSemiring R
    inst✝⁹ : ContinuousStar R
    inst✝⁸ : Module R A
    inst✝⁷ : IsScalarTower R A A
    inst✝⁶ : SMulCommClass R A A
    inst✝⁵ : CompleteSpace R
    inst✝⁴ : Algebra R Complex
    inst✝³ : IsScalarTower R Complex A
    p : A → Prop
    p' : Unitization Complex A → Prop
    inst✝² : NonUnitalContinuousFunctionalCalculus R p
    inst✝¹ : ContinuousFunctionalCalculus R p'
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R (Unitization Complex A)
    hp : ∀ {a : A}, Iff (p' ↑a) (p a)
    a : A
    f : R → R
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ⊢ Eq (↑(cfcₙ f a)) (cfc f ↑a)
  -/
  by_cases h : ContinuousOn f (σₙ R a) ∧ p a
    /-
      case pos
      A : Type u_1
      inst✝¹⁴ : NonUnitalCStarAlgebra A
      R : Type u_2
      inst✝¹³ : Semifield R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R A A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : CompleteSpace R
      inst✝⁴ : Algebra R Complex
      inst✝³ : IsScalarTower R Complex A
      p : A → Prop
      p' : Unitization Complex A → Prop
      inst✝² : NonUnitalContinuousFunctionalCalculus R p
      inst✝¹ : ContinuousFunctionalCalculus R p'
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R (Unitization Complex A)
      hp : ∀ {a : A}, Iff (p' ↑a) (p a)
      a : A
      f : R → R
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      h : And (ContinuousOn f (quasispectrum R a)) (p a)
      ⊢ Eq (↑(cfcₙ f a)) (cfc f ↑a)
    -/
  · obtain ⟨hf, ha⟩ := h
    /-
      case pos.intro
      A : Type u_1
      inst✝¹⁴ : NonUnitalCStarAlgebra A
      R : Type u_2
      inst✝¹³ : Semifield R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R A A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : CompleteSpace R
      inst✝⁴ : Algebra R Complex
      inst✝³ : IsScalarTower R Complex A
      p : A → Prop
      p' : Unitization Complex A → Prop
      inst✝² : NonUnitalContinuousFunctionalCalculus R p
      inst✝¹ : ContinuousFunctionalCalculus R p'
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R (Unitization Complex A)
      hp : ∀ {a : A}, Iff (p' ↑a) (p a)
      a : A
      f : R → R
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      hf : ContinuousOn f (quasispectrum R a)
      ha : p a
      ⊢ Eq (↑(cfcₙ f a)) (cfc f ↑a)
    -/
    rw [← cfcₙ_eq_cfc (quasispectrum_inr_eq R ℂ a ▸ hf)]
    /-
      case pos.intro
      A : Type u_1
      inst✝¹⁴ : NonUnitalCStarAlgebra A
      R : Type u_2
      inst✝¹³ : Semifield R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R A A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : CompleteSpace R
      inst✝⁴ : Algebra R Complex
      inst✝³ : IsScalarTower R Complex A
      p : A → Prop
      p' : Unitization Complex A → Prop
      inst✝² : NonUnitalContinuousFunctionalCalculus R p
      inst✝¹ : ContinuousFunctionalCalculus R p'
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R (Unitization Complex A)
      hp : ∀ {a : A}, Iff (p' ↑a) (p a)
      a : A
      f : R → R
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      hf : ContinuousOn f (quasispectrum R a)
      ha : p a
      ⊢ Eq (↑(cfcₙ f a)) (cfcₙ f ↑a)
    -/
    exact (inrNonUnitalStarAlgHom ℂ A).map_cfcₙ f a
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝¹⁴ : NonUnitalCStarAlgebra A
      R : Type u_2
      inst✝¹³ : Semifield R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : Module R A
      inst✝⁷ : IsScalarTower R A A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : CompleteSpace R
      inst✝⁴ : Algebra R Complex
      inst✝³ : IsScalarTower R Complex A
      p : A → Prop
      p' : Unitization Complex A → Prop
      inst✝² : NonUnitalContinuousFunctionalCalculus R p
      inst✝¹ : ContinuousFunctionalCalculus R p'
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R (Unitization Complex A)
      hp : ∀ {a : A}, Iff (p' ↑a) (p a)
      a : A
      f : R → R
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      h : Not (And (ContinuousOn f (quasispectrum R a)) (p a))
      ⊢ Eq (↑(cfcₙ f a)) (cfc f ↑a)
    -/
  · obtain (hf | ha) := not_and_or.mp h
    · rw [cfcₙ_apply_of_not_continuousOn a hf, inr_zero,
        cfc_apply_of_not_continuousOn _ (quasispectrum_eq_spectrum_inr' R ℂ a ▸ hf)]
    · rw [cfcₙ_apply_of_not_predicate a ha, inr_zero,
        cfc_apply_of_not_predicate _ (not_iff_not.mpr hp |>.mpr ha)]


lemma Unitization.complex_cfcₙ_eq_cfc_inr (a : A) (f : ℂ → ℂ) (hf₀ : f 0 = 0 := by cfc_zero_tac) :
    cfcₙ f a = cfc f (a : A⁺¹) :=
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    a : A
    f : Complex → Complex
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ⊢ Eq (f 0) 0
  -/
  Unitization.cfcₙ_eq_cfc_inr isStarNormal_inr ..
  /-
    🎉 no goals
  -/


/-- note: the version for `ℝ≥0`, `Unization.nnreal_cfcₙ_eq_cfc_inr`, can be found in
`Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Order` -/
lemma Unitization.real_cfcₙ_eq_cfc_inr (a : A) (f : ℝ → ℝ) (hf₀ : f 0 = 0 := by cfc_zero_tac) :
    cfcₙ f a = cfc f (a : A⁺¹) :=
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    a : A
    f : Real → Real
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ⊢ Eq (f 0) 0
  -/
  Unitization.cfcₙ_eq_cfc_inr isSelfAdjoint_inr ..
  /-
    🎉 no goals
  -/


