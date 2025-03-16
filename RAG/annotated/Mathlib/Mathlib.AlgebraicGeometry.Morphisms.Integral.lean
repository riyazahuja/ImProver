/-- A morphism of schemes `X ⟶ Y` is finite if
the preimage of any affine open subset of `Y` is affine and the induced ring
hom is finite. -/
@[mk_iff]
class IsIntegralHom {X Y : Scheme} (f : X ⟶ Y) extends IsAffineHom f : Prop where
  integral_app (U : Y.Opens) (hU : IsAffineOpen U) : (f.app U).hom.IsIntegral


instance hasAffineProperty : HasAffineProperty @IsIntegralHom
    fun X _ f _ ↦ IsAffine X ∧ RingHom.IsIntegral (f.app ⊤).hom := by
  /-
    ⊢ AlgebraicGeometry.HasAffineProperty @AlgebraicGeometry.IsIntegralHom fun X x …
  -/
  show HasAffineProperty @IsIntegralHom (affineAnd RingHom.IsIntegral)
  rw [HasAffineProperty.affineAnd_iff _ RingHom.isIntegral_respectsIso
    RingHom.isIntegral_isStableUnderBaseChange.localizationPreserves
    RingHom.isIntegral_ofLocalizationSpan]
  /-
    ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (AlgebraicGeome …
  -/
  simp [isIntegralHom_iff]
  /-
    🎉 no goals
  -/


instance : IsStableUnderComposition @IsIntegralHom :=
  HasAffineProperty.affineAnd_isStableUnderComposition (Q := RingHom.IsIntegral) hasAffineProperty
    RingHom.isIntegral_stableUnderComposition


instance : IsStableUnderBaseChange @IsIntegralHom :=
  HasAffineProperty.affineAnd_isStableUnderBaseChange (Q := RingHom.IsIntegral) hasAffineProperty
    RingHom.isIntegral_respectsIso RingHom.isIntegral_isStableUnderBaseChange


instance : ContainsIdentities @IsIntegralHom :=
                         /-
                           X : AlgebraicGeometry.Scheme
                           x✝¹ : X.Opens
                           x✝ : AlgebraicGeometry.IsAffineOpen x✝¹
                           ⊢ (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.id X) x✝¹). …
                         -/
  ⟨fun X ↦ ⟨fun _ _ ↦ by simpa using RingHom.isIntegral_of_surjective _ (Equiv.refl _).surjective⟩⟩
                         /-
                           🎉 no goals
                         -/


instance : IsMultiplicative @IsIntegralHom where


