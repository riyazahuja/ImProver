/--
Implementation: If `π` is a morphism in `TopCat` which is a quotient map, then it is an effective
epimorphism. The theorem `TopCat.effectiveEpi_iff_isQuotientMap` should be used instead of
this definition.
-/
noncomputable
def effectiveEpiStructOfQuotientMap {B X : TopCat.{u}} (π : X ⟶ B) (hπ : IsQuotientMap π) :
    EffectiveEpiStruct π where
  /- `IsQuotientMap.lift` gives the required morphism -/
  desc e h := hπ.lift e fun a b hab ↦
    DFunLike.congr_fun (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩
        /-
          B X : TopCat
          π : Quiver.Hom X B
          hπ : Topology.IsQuotientMap ⇑π
          W✝ : TopCat
          e : Quiver.Hom X W✝
          h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
          a b : ↑X
          hab : Eq (π a) (π b)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => a, continuous_toF …
        -/
    (by ext; exact hab)) a
             /-
               🎉 no goals
             -/
  /- `IsQuotientMap.lift_comp` gives the factorisation -/
  fac e h := (hπ.lift_comp e
    fun a b hab ↦ DFunLike.congr_fun (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩
        /-
          B X : TopCat
          π : Quiver.Hom X B
          hπ : Topology.IsQuotientMap ⇑π
          W✝ : TopCat
          e : Quiver.Hom X W✝
          h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
          a b : ↑X
          hab : Eq (π a) (π b)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => a, continuous_toF …
        -/
    (by ext; exact hab)) a)
             /-
               🎉 no goals
             -/
  /- Uniqueness follows from the fact that `IsQuotientMap.lift` is an equivalence (given by
  `IsQuotientMap.liftEquiv`). -/
  uniq e h g hm := by
    suffices g = hπ.liftEquiv ⟨e,
      fun a b hab ↦ DFunLike.congr_fun
        (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩ (by ext; exact hab))
        a⟩ by assumption
    /-
      B X : TopCat
      π : Quiver.Hom X B
      hπ : Topology.IsQuotientMap ⇑π
      W✝ : TopCat
      e : Quiver.Hom X W✝
      h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      ⊢ Eq g (hπ.liftEquiv ⟨e, ⋯⟩)
    -/
    rw [← Equiv.symm_apply_eq hπ.liftEquiv]
    /-
      B X : TopCat
      π : Quiver.Hom X B
      hπ : Topology.IsQuotientMap ⇑π
      W✝ : TopCat
      e : Quiver.Hom X W✝
      h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      ⊢ Eq (hπ.liftEquiv.symm g) ⟨e, ⋯⟩
    -/
    ext
    /-
      case a.h
      B X : TopCat
      π : Quiver.Hom X B
      hπ : Topology.IsQuotientMap ⇑π
      W✝ : TopCat
      e : Quiver.Hom X W✝
      h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      a✝ : ↑X
      ⊢ Eq (↑(hπ.liftEquiv.symm g) a✝) (↑⟨e, ⋯⟩ a✝)
    -/
    simp only [IsQuotientMap.liftEquiv_symm_apply_coe, ContinuousMap.comp_apply, ← hm]
    /-
      case a.h
      B X : TopCat
      π : Quiver.Hom X B
      hπ : Topology.IsQuotientMap ⇑π
      W✝ : TopCat
      e : Quiver.Hom X W✝
      h : ∀ {Z : TopCat} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      a✝ : ↑X
      ⊢ Eq (g (π a✝)) ((CategoryTheory.CategoryStruct.comp π g) a✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The effective epimorphisms in `TopCat` are precisely the quotient maps. -/
theorem effectiveEpi_iff_isQuotientMap {B X : TopCat.{u}} (π : X ⟶ B) :
    EffectiveEpi π ↔ IsQuotientMap π := by
  /- The backward direction is given by `effectiveEpiStructOfQuotientMap` above. -/
  /-
    B X : TopCat
    π : Quiver.Hom X B
    ⊢ Iff (CategoryTheory.EffectiveEpi π) (Topology.IsQuotientMap ⇑π)
  -/
  refine ⟨fun _ ↦ ?_, fun hπ ↦ ⟨⟨effectiveEpiStructOfQuotientMap π hπ⟩⟩⟩
  /- Since `TopCat` has pullbacks, `π` is in fact a `RegularEpi`. This means that it exhibits `B` as
    a coequalizer of two maps into `X`. It suffices to prove that `π` followed by the isomorphism to
    an arbitrary coequalizer is a quotient map. -/
  /-
    B X : TopCat
    π : Quiver.Hom X B
    x✝ : CategoryTheory.EffectiveEpi π
    ⊢ Topology.IsQuotientMap ⇑π
  -/
  have hπ : RegularEpi π := inferInstance
  /-
    B X : TopCat
    π : Quiver.Hom X B
    x✝ : CategoryTheory.EffectiveEpi π
    hπ : CategoryTheory.RegularEpi π
    ⊢ Topology.IsQuotientMap ⇑π
  -/
  let F := parallelPair hπ.left hπ.right
  /-
    B X : TopCat
    π : Quiver.Hom X B
    x✝ : CategoryTheory.EffectiveEpi π
    hπ : CategoryTheory.RegularEpi π
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
    ⊢ Topology.IsQuotientMap ⇑π
  -/
  let i : B ≅ colimit F := hπ.isColimit.coconePointUniqueUpToIso (colimit.isColimit _)
  suffices IsQuotientMap (homeoOfIso i ∘ π) by
    simpa [← Function.comp_assoc] using (homeoOfIso i).symm.isQuotientMap.comp this
  /-
    B X : TopCat
    π : Quiver.Hom X B
    x✝ : CategoryTheory.EffectiveEpi π
    hπ : CategoryTheory.RegularEpi π
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
    i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
    ⊢ Topology.IsQuotientMap (Function.comp ⇑(TopCat.homeoOfIso i) ⇑π)
  -/
  constructor
  /- Effective epimorphisms are epimorphisms and epimorphisms in `TopCat` are surjective. -/
    /-
      case surjective
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      ⊢ Function.Surjective (Function.comp ⇑(TopCat.homeoOfIso i) ⇑π)
    -/
  · change Function.Surjective (π ≫ i.hom)
    /-
      case surjective
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      ⊢ Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp π i.hom)
    -/
    rw [← epi_iff_surjective]
    /-
      case surjective
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp π i.hom)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /- The key to proving that the coequalizer has the quotient topology is
    `TopCat.coequalizer_isOpen_iff` which characterises the open sets in a coequalizer. -/
    /-
      case eq_coinduced
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      ⊢ Eq (CategoryTheory.Limits.colimit F).topologicalSpace_coe (TopologicalSpace. …
    -/
  · ext U
    have : π ≫ i.hom = colimit.ι F WalkingParallelPair.one := by
      simp [F, i, ← Iso.eq_comp_inv]
    /-
      case eq_coinduced.a.h.a
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      U : Set ↑(CategoryTheory.Limits.colimit F)
      this : Eq (CategoryTheory.CategoryStruct.comp π i.hom) (CategoryTheory.Limits. …
      ⊢ Iff (IsOpen U) (IsOpen U)
    -/
    rw [isOpen_coinduced (f := (homeoOfIso i ∘ π)), coequalizer_isOpen_iff _ U, ← this]
    /-
      case eq_coinduced.a.h.a
      B X : TopCat
      π : Quiver.Hom X B
      x✝ : CategoryTheory.EffectiveEpi π
      hπ : CategoryTheory.RegularEpi π
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair TopCat := …
      i : CategoryTheory.Iso B (CategoryTheory.Limits.colimit F) := CategoryTheory.R …
      U : Set ↑(CategoryTheory.Limits.colimit F)
      this : Eq (CategoryTheory.CategoryStruct.comp π i.hom) (CategoryTheory.Limits. …
      ⊢ Iff (IsOpen (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp π i.hom)) U) …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-22")]
alias effectiveEpi_iff_quotientMap := effectiveEpi_iff_isQuotientMap


