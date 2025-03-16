/-- A morphism of schemes `X ⟶ Y` is finite if
the preimage of any affine open subset of `Y` is affine and the induced ring
hom is finite. -/
@[mk_iff]
class IsFinite {X Y : Scheme} (f : X ⟶ Y) extends IsAffineHom f : Prop where
  finite_app (U : Y.Opens) (hU : IsAffineOpen U) : (f.app U).hom.Finite


instance : HasAffineProperty @IsFinite
    (fun X _ f _ ↦ IsAffine X ∧ RingHom.Finite (f.appTop).hom) := by
  /-
    ⊢ AlgebraicGeometry.HasAffineProperty @AlgebraicGeometry.IsFinite fun X x f x_ …
  -/
  show HasAffineProperty @IsFinite (affineAnd RingHom.Finite)
  rw [HasAffineProperty.affineAnd_iff _ RingHom.finite_respectsIso
    RingHom.finite_localizationPreserves RingHom.finite_ofLocalizationSpan]
  /-
    ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (AlgebraicGeome …
  -/
  simp [isFinite_iff]
  /-
    🎉 no goals
  -/


instance : IsStableUnderComposition @IsFinite :=
  HasAffineProperty.affineAnd_isStableUnderComposition inferInstance
    RingHom.finite_stableUnderComposition


instance : IsStableUnderBaseChange @IsFinite :=
  HasAffineProperty.affineAnd_isStableUnderBaseChange inferInstance
    RingHom.finite_respectsIso RingHom.finite_isStableUnderBaseChange


instance : ContainsIdentities @IsFinite :=
  HasAffineProperty.affineAnd_containsIdentities inferInstance
    RingHom.finite_respectsIso RingHom.finite_containsIdentities


instance : IsMultiplicative @IsFinite where


instance (priority := 900) [IsIso f] : IsFinite f := of_isIso @IsFinite f


instance {Z : Scheme.{u}} (g : Y ⟶ Z) [IsFinite f] [IsFinite g] : IsFinite (f ≫ g) :=
  IsStableUnderComposition.comp_mem f g ‹IsFinite f› ‹IsFinite g›


lemma iff_isIntegralHom_and_locallyOfFiniteType :
    IsFinite f ↔ IsIntegralHom f ∧ LocallyOfFiniteType f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.IsFinite f) (And (AlgebraicGeometry.IsIntegralHom f)  …
  -/
  wlog hY : IsAffine Y
  · rw [IsLocalAtTarget.iff_of_openCover (P := @IsFinite) Y.affineCover,
      IsLocalAtTarget.iff_of_openCover (P := @IsIntegralHom) Y.affineCover,
      IsLocalAtTarget.iff_of_openCover (P := @LocallyOfFiniteType) Y.affineCover]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), AlgebraicGeome …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ Iff (∀ (i : Y.affineCover.1), AlgebraicGeometry.IsFinite (AlgebraicGeometry. …
    -/
    simp_rw [this, forall_and]
    /-
      🎉 no goals
    -/
  rw [HasAffineProperty.iff_of_isAffine (P := @IsFinite),
    HasAffineProperty.iff_of_isAffine (P := @IsIntegralHom),
    RingHom.finite_iff_isIntegral_and_finiteType, ← and_assoc]
  refine and_congr_right fun ⟨_, _⟩ ↦
    (HasRingHomProperty.iff_of_isAffine (P := @LocallyOfFiniteType)).symm


lemma eq_inf :
    @IsFinite = (@IsIntegralHom ⊓ @LocallyOfFiniteType : MorphismProperty Scheme) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsFinite) (Min.min @AlgebraicGeometry.IsIntegralHom @ …
  -/
  ext; exact IsFinite.iff_isIntegralHom_and_locallyOfFiniteType _
       /-
         🎉 no goals
       -/


instance (priority := 900) [IsFinite f] : IsIntegralHom f :=
  ((IsFinite.iff_isIntegralHom_and_locallyOfFiniteType f).mp ‹_›).1


instance (priority := 900) [hf : IsFinite f] : LocallyOfFiniteType f :=
  ((IsFinite.iff_isIntegralHom_and_locallyOfFiniteType f).mp ‹_›).2


lemma _root_.AlgebraicGeometry.IsClosedImmersion.iff_isFinite_and_mono :
    IsClosedImmersion f ↔ IsFinite f ∧ Mono f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.IsClosedImmersion f) (And (AlgebraicGeometry.IsFinite …
  -/
  wlog hY : IsAffine Y
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), AlgebraicGeome …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ Iff (AlgebraicGeometry.IsClosedImmersion f) (And (AlgebraicGeometry.IsFinite …
    -/
  · show _ ↔ _ ∧ monomorphisms _ f
    rw [IsLocalAtTarget.iff_of_openCover (P := @IsFinite) Y.affineCover,
      IsLocalAtTarget.iff_of_openCover (P := @IsClosedImmersion) Y.affineCover,
      IsLocalAtTarget.iff_of_openCover (P := monomorphisms _) Y.affineCover]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), AlgebraicGeome …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ Iff (∀ (i : Y.affineCover.1), AlgebraicGeometry.IsClosedImmersion (Algebraic …
    -/
    simp_rw [this, forall_and, monomorphisms]
    /-
      🎉 no goals
    -/
  rw [HasAffineProperty.iff_of_isAffine (P := @IsClosedImmersion),
    HasAffineProperty.iff_of_isAffine (P := @IsFinite),
    RingHom.surjective_iff_epi_and_finite, @and_comm (Epi _), ← and_assoc]
  refine and_congr_right fun ⟨_, _⟩ ↦
    Iff.trans ?_ (arrow_mk_iso_iff (monomorphisms _) (arrowIsoSpecΓOfIsAffine f).symm)
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    x✝ : And (AlgebraicGeometry.IsAffine X) (AlgebraicGeometry.Scheme.Hom.appTop f …
    left✝ : AlgebraicGeometry.IsAffine X
    right✝ : (AlgebraicGeometry.Scheme.Hom.appTop f).hom.Finite
    ⊢ Iff (CategoryTheory.Epi (AlgebraicGeometry.Scheme.Hom.appTop f)) (CategoryTh …
  -/
  trans Mono (f.app ⊤).op
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hY : AlgebraicGeometry.IsAffine Y
      x✝ : And (AlgebraicGeometry.IsAffine X) (AlgebraicGeometry.Scheme.Hom.appTop f …
      left✝ : AlgebraicGeometry.IsAffine X
      right✝ : (AlgebraicGeometry.Scheme.Hom.appTop f).hom.Finite
      ⊢ Iff (CategoryTheory.Epi (AlgebraicGeometry.Scheme.Hom.appTop f)) (CategoryTh …
    -/
  · exact ⟨fun h ↦ inferInstance, fun h ↦ show Epi (f.app ⊤).op.unop by infer_instance⟩
    /-
      🎉 no goals
    -/
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    x✝ : And (AlgebraicGeometry.IsAffine X) (AlgebraicGeometry.Scheme.Hom.appTop f …
    left✝ : AlgebraicGeometry.IsAffine X
    right✝ : (AlgebraicGeometry.Scheme.Hom.appTop f).hom.Finite
    ⊢ Iff (CategoryTheory.Mono (AlgebraicGeometry.Scheme.Hom.app f Top.top).op) (C …
  -/
  exact (Functor.mono_map_iff_mono Scheme.Spec _).symm
  /-
    🎉 no goals
  -/


lemma _root_.AlgebraicGeometry.IsClosedImmersion.eq_isFinite_inf_mono :
    @IsClosedImmersion = (@IsFinite ⊓ monomorphisms Scheme : MorphismProperty _) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsClosedImmersion) (Min.min (@AlgebraicGeometry.IsFin …
  -/
  ext; exact IsClosedImmersion.iff_isFinite_and_mono _
       /-
         🎉 no goals
       -/


instance (priority := 900) {X Y : Scheme} (f : X ⟶ Y) [IsClosedImmersion f] : IsFinite f :=
  ((IsClosedImmersion.iff_isFinite_and_mono f).mp ‹_›).1


