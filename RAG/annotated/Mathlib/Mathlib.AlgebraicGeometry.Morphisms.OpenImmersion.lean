theorem isOpenImmersion_iff_stalk {f : X ⟶ Y} : IsOpenImmersion f ↔
    IsOpenEmbedding f.base ∧ ∀ x, IsIso (f.stalkMap x) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.IsOpenImmersion f) (And (Topology.IsOpenEmbedding ⇑f. …
  -/
  constructor
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ AlgebraicGeometry.IsOpenImmersion f → And (Topology.IsOpenEmbedding ⇑f.base) …
    -/
  · intro h; exact ⟨h.1, inferInstance⟩
             /-
               🎉 no goals
             -/
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ And (Topology.IsOpenEmbedding ⇑f.base) (∀ (x : ↑↑X.toPresheafedSpace), Categ …
    -/
  · rintro ⟨h₁, h₂⟩; exact IsOpenImmersion.of_stalk_iso f h₁
                     /-
                       🎉 no goals
                     -/


theorem isOpenImmersion_eq_inf :
    @IsOpenImmersion = (topologically IsOpenEmbedding) ⊓
      stalkwise (fun f ↦ Function.Bijective f) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsOpenImmersion) (Min.min (AlgebraicGeometry.topologi …
  -/
  ext
  exact isOpenImmersion_iff_stalk.trans
    (and_congr Iff.rfl (forall_congr' fun x ↦ ConcreteCategory.isIso_iff_bijective _))


instance : IsLocalAtTarget (stalkwise (fun f ↦ Function.Bijective f)) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R S} [Co …
  -/
  apply stalkwiseIsLocalAtTarget_of_respectsIso
  /-
    case hP
    X Y : AlgebraicGeometry.Scheme
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => Function.Biject …
  -/
  rw [RingHom.toMorphismProperty_respectsIso_iff]
  /-
    case hP
    X Y : AlgebraicGeometry.Scheme
    ⊢ (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] f => Functio …
  -/
  convert (inferInstanceAs (MorphismProperty.isomorphisms CommRingCat).RespectsIso)
  /-
    case h.e'_3.h
    X Y : AlgebraicGeometry.Scheme
    ⊢ Eq (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] f => Func …
  -/
  ext
  -- Regression in https://github.com/leanprover-community/mathlib4/pull/17583: have to specify C explicitly below.
  /-
    case h.e'_3.h.h
    X Y : AlgebraicGeometry.Scheme
    X✝ Y✝ : CommRingCat
    f✝ : Quiver.Hom X✝ Y✝
    ⊢ Iff (RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] f => Fu …
  -/
  exact (ConcreteCategory.isIso_iff_bijective (C := CommRingCat) _).symm
  /-
    🎉 no goals
  -/


instance isOpenImmersion_isLocalAtTarget : IsLocalAtTarget @IsOpenImmersion :=
  isOpenImmersion_eq_inf ▸ inferInstance


