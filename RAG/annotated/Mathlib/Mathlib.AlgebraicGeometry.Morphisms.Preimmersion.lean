/-- A morphism of schemes `f : X ⟶ Y` is a preimmersion if the underlying map of
topological spaces is an embedding and the induced morphisms of stalks are all surjective. -/
@[mk_iff]
class IsPreimmersion {X Y : Scheme} (f : X ⟶ Y) extends SurjectiveOnStalks f : Prop where
  base_embedding : IsEmbedding f.base


lemma Scheme.Hom.isEmbedding {X Y : Scheme} (f : Hom X Y) [IsPreimmersion f] : IsEmbedding f.base :=
  IsPreimmersion.base_embedding


@[deprecated (since := "2024-10-26")]
alias Scheme.Hom.embedding := Scheme.Hom.isEmbedding


lemma isPreimmersion_eq_inf :
    @IsPreimmersion = (@SurjectiveOnStalks ⊓ topologically IsEmbedding : MorphismProperty _) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsPreimmersion) (Min.min (@AlgebraicGeometry.Surjecti …
  -/
  ext
  /-
    case h.h.h.a
    x✝² x✝¹ : AlgebraicGeometry.Scheme
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Iff (AlgebraicGeometry.IsPreimmersion x✝) (Min.min (@AlgebraicGeometry.Surje …
  -/
  rw [isPreimmersion_iff]
  /-
    case h.h.h.a
    x✝² x✝¹ : AlgebraicGeometry.Scheme
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Iff (And (AlgebraicGeometry.SurjectiveOnStalks x✝) (Topology.IsEmbedding ⇑x✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Being surjective on stalks is local at the target. -/
instance isSurjectiveOnStalks_isLocalAtTarget : IsLocalAtTarget
    (stalkwise (Function.Surjective ·)) :=
  stalkwiseIsLocalAtTarget_of_respectsIso RingHom.surjective_respectsIso


instance : IsLocalAtTarget @IsPreimmersion :=
  isPreimmersion_eq_inf ▸ inferInstance


instance (priority := 900) {X Y : Scheme} (f : X ⟶ Y) [IsOpenImmersion f] : IsPreimmersion f where
  base_embedding := f.isOpenEmbedding.isEmbedding
  surj_on_stalks _ := (ConcreteCategory.bijective_of_isIso (C := CommRingCat) _).2


instance : MorphismProperty.IsMultiplicative @IsPreimmersion where
  id_mem _ := inferInstance
  comp_mem f g _ _ := ⟨g.isEmbedding.comp f.isEmbedding⟩


instance comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsPreimmersion f]
    [IsPreimmersion g] : IsPreimmersion (f ≫ g) :=
  MorphismProperty.IsStableUnderComposition.comp_mem f g inferInstance inferInstance


instance (priority := 900) {X Y} (f : X ⟶ Y) [IsPreimmersion f] : Mono f := by
  refine (Scheme.forgetToLocallyRingedSpace ⋙
    LocallyRingedSpace.forgetToSheafedSpace).mono_of_mono_map ?_
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsPreimmersion f
    ⊢ CategoryTheory.Mono ((AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.co …
  -/
  apply SheafedSpace.mono_of_base_injective_of_stalk_epi
    /-
      case h₁
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsPreimmersion f
      ⊢ Function.Injective ⇑((AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.co …
    -/
  · exact f.isEmbedding.injective
    /-
      🎉 no goals
    -/
    /-
      case h₂
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsPreimmersion f
      ⊢ ∀ (x : ↑↑((AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.comp Algebrai …
    -/
  · exact fun x ↦ ConcreteCategory.epi_of_surjective _ (f.stalkMap_surjective x)
    /-
      🎉 no goals
    -/


theorem of_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsPreimmersion g]
    [IsPreimmersion (f ≫ g)] : IsPreimmersion f where
  base_embedding := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsPreimmersion g
      inst✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp f …
      ⊢ Topology.IsEmbedding ⇑f.base
    -/
    have h := (f ≫ g).isEmbedding
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsPreimmersion g
      inst✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp f …
      h : Topology.IsEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g).base
      ⊢ Topology.IsEmbedding ⇑f.base
    -/
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsPreimmersion g
      inst✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp f …
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    rwa [← g.isEmbedding.of_comp_iff]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsPreimmersion g
      inst✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp f …
      x : ↑↑X.toPresheafedSpace
      h : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheor …
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    /-
      🎉 no goals
    -/
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsPreimmersion g
      inst✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp f …
      x : ↑↑X.toPresheafedSpace
      h : Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeometr …
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
  surj_on_stalks x := by
    /-
      🎉 no goals
    -/
    have h := (f ≫ g).stalkMap_surjective x
    rw [Scheme.stalkMap_comp] at h
    exact Function.Surjective.of_comp h


theorem comp_iff {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsPreimmersion g] :
    IsPreimmersion (f ≫ g) ↔ IsPreimmersion f :=
  ⟨fun _ ↦ of_comp f g, fun _ ↦ inferInstance⟩


lemma Spec_map_iff {R S : CommRingCat.{u}} (f : R ⟶ S) :
    IsPreimmersion (Spec.map f) ↔ IsEmbedding (PrimeSpectrum.comap f.hom) ∧
      f.hom.SurjectiveOnStalks := by
  haveI : (RingHom.toMorphismProperty <| fun f ↦ Function.Surjective f).RespectsIso := by
    rw [← RingHom.toMorphismProperty_respectsIso_iff]
    exact RingHom.surjective_respectsIso
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    this : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] f => Fu …
    ⊢ Iff (AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Spec.map f)) (And ( …
  -/
  rw [← HasRingHomProperty.Spec_iff (P := @SurjectiveOnStalks), isPreimmersion_iff, and_comm]
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    this : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] f => Fu …
    ⊢ Iff (And (Topology.IsEmbedding ⇑(AlgebraicGeometry.Spec.map f).base) (Algebr …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mk_Spec_map {R S : CommRingCat.{u}} {f : R ⟶ S}
    (h₁ : IsEmbedding (PrimeSpectrum.comap f.hom)) (h₂ : f.hom.SurjectiveOnStalks) :
    IsPreimmersion (Spec.map f) :=
  (Spec_map_iff f).mpr ⟨h₁, h₂⟩


lemma of_isLocalization {R S : Type u} [CommRing R] (M : Submonoid R) [CommRing S]
    [Algebra R S] [IsLocalization M S] :
    IsPreimmersion (Spec.map (CommRingCat.ofHom <| algebraMap R S)) :=
  IsPreimmersion.mk_Spec_map
    (PrimeSpectrum.localization_comap_isEmbedding (R := R) S M)
    (RingHom.surjectiveOnStalks_of_isLocalization (M := M) S)


open Limits MorphismProperty in
instance : IsStableUnderBaseChange @IsPreimmersion := by
  /-
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.I …
  -/
  refine .mk' fun X Y Z f g _ _ ↦ ?_
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    x✝¹ : CategoryTheory.Limits.HasPullback f g
    x✝ : AlgebraicGeometry.IsPreimmersion g
    ⊢ AlgebraicGeometry.IsPreimmersion (CategoryTheory.Limits.pullback.fst f g)
  -/
  have := pullback_fst (P := @SurjectiveOnStalks) f g inferInstance
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    x✝¹ : CategoryTheory.Limits.HasPullback f g
    x✝ : AlgebraicGeometry.IsPreimmersion g
    this : AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.Limits.pullback.fs …
    ⊢ AlgebraicGeometry.IsPreimmersion (CategoryTheory.Limits.pullback.fst f g)
  -/
  constructor
  let L (x : (pullback f g : _)) : { x : X × Y | f.base x.1 = g.base x.2 } :=
    ⟨⟨(pullback.fst f g).base x, (pullback.snd f g).base x⟩,
    by simp only [Set.mem_setOf, ← Scheme.comp_base_apply, pullback.condition]⟩
  have : IsEmbedding L := IsEmbedding.of_comp (by fun_prop) continuous_subtype_val
    (SurjectiveOnStalks.isEmbedding_pullback f g)
  exact IsEmbedding.subtypeVal.comp ((TopCat.pullbackHomeoPreimage _ f.continuous _
    g.isEmbedding).isEmbedding.comp this)


