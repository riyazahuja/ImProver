/-- A morphism of schemes `f : X ⟶ Y` is locally of finite type if for each affine `U ⊆ Y` and
`V ⊆ f ⁻¹' U`, The induced map `Γ(Y, U) ⟶ Γ(X, V)` is of finite type.
-/
@[mk_iff]
class LocallyOfFiniteType (f : X ⟶ Y) : Prop where
  finiteType_of_affine_subset :
    ∀ (U : Y.affineOpens) (V : X.affineOpens) (e : V.1 ≤ f ⁻¹ᵁ U.1), (f.appLE U V e).hom.FiniteType


instance : HasRingHomProperty @LocallyOfFiniteType RingHom.FiniteType where
  isLocal_ringHomProperty := RingHom.finiteType_is_local
  eq_affineLocally' := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Eq (@AlgebraicGeometry.LocallyOfFiniteType) (AlgebraicGeometry.affineLocally …
    -/
    ext X Y f
    /-
      case h
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (AlgebraicGeometry.LocallyOfFiniteType f) (AlgebraicGeometry.affineLocal …
    -/
    rw [locallyOfFiniteType_iff, affineLocally_iff_affineOpens_le]
    /-
      🎉 no goals
    -/


instance (priority := 900) locallyOfFiniteType_of_isOpenImmersion [IsOpenImmersion f] :
    LocallyOfFiniteType f :=
  HasRingHomProperty.of_isOpenImmersion
    RingHom.finiteType_holdsForLocalizationAway.containsIdentities


instance : MorphismProperty.IsStableUnderComposition @LocallyOfFiniteType :=
  HasRingHomProperty.stableUnderComposition RingHom.finiteType_stableUnderComposition


instance locallyOfFiniteType_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [hf : LocallyOfFiniteType f] [hg : LocallyOfFiniteType g] : LocallyOfFiniteType (f ≫ g) :=
  MorphismProperty.comp_mem _ f g hf hg


theorem locallyOfFiniteType_of_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [LocallyOfFiniteType (f ≫ g)] : LocallyOfFiniteType f :=
  HasRingHomProperty.of_comp (fun _ _ ↦ RingHom.FiniteType.of_comp_finiteType) ‹_›


instance : MorphismProperty.IsMultiplicative @LocallyOfFiniteType where
  id_mem _ := inferInstance


open scoped TensorProduct in
instance locallyOfFiniteType_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @LocallyOfFiniteType :=
  HasRingHomProperty.isStableUnderBaseChange RingHom.finiteType_isStableUnderBaseChange


