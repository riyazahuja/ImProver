/-- A morphism of schemes `f : X ⟶ Y` is locally of finite presentation if for each affine `U ⊆ Y`
and `V ⊆ f ⁻¹' U`, The induced map `Γ(Y, U) ⟶ Γ(X, V)` is of finite presentation. -/
@[mk_iff]
class LocallyOfFinitePresentation : Prop where
  finitePresentation_of_affine_subset :
    ∀ (U : Y.affineOpens) (V : X.affineOpens) (e : V.1 ≤ f ⁻¹ᵁ U.1),
      (f.appLE U V e).hom.FinitePresentation


instance : HasRingHomProperty @LocallyOfFinitePresentation RingHom.FinitePresentation where
  isLocal_ringHomProperty := RingHom.finitePresentation_isLocal
  eq_affineLocally' := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Eq (@AlgebraicGeometry.LocallyOfFinitePresentation) (AlgebraicGeometry.affin …
    -/
    ext X Y f
    /-
      case h
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (AlgebraicGeometry.LocallyOfFinitePresentation f) (AlgebraicGeometry.aff …
    -/
    rw [locallyOfFinitePresentation_iff, affineLocally_iff_affineOpens_le]
    /-
      🎉 no goals
    -/


instance (priority := 900) locallyOfFinitePresentation_of_isOpenImmersion [IsOpenImmersion f] :
    LocallyOfFinitePresentation f :=
  HasRingHomProperty.of_isOpenImmersion
    RingHom.finitePresentation_holdsForLocalizationAway.containsIdentities


instance : MorphismProperty.IsStableUnderComposition @LocallyOfFinitePresentation :=
  HasRingHomProperty.stableUnderComposition RingHom.finitePresentation_stableUnderComposition


instance locallyOfFinitePresentation_comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z)
    [hf : LocallyOfFinitePresentation f] [hg : LocallyOfFinitePresentation g] :
    LocallyOfFinitePresentation (f ≫ g) :=
  MorphismProperty.comp_mem _ f g hf hg


lemma locallyOfFinitePresentation_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @LocallyOfFinitePresentation :=
  HasRingHomProperty.isStableUnderBaseChange RingHom.finitePresentation_isStableUnderBaseChange


