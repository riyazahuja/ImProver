/--
A morphism of schemes `f : X ⟶ Y` is universally injective if the base change `X ×[Y] Y' ⟶ Y'`
along any morphism `Y' ⟶ Y` is injective (on points).
-/
@[mk_iff]
class UniversallyInjective (f : X ⟶ Y) : Prop where
  universally_injective : universally (topologically (Injective ·)) f


theorem Scheme.Hom.injective (f : X.Hom Y) [UniversallyInjective f] :
    Function.Injective f.base :=
  UniversallyInjective.universally_injective _ _ _ .of_id_snd


theorem universallyInjective_eq :
    @UniversallyInjective = universally (topologically (Injective ·)) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyInjective) (AlgebraicGeometry.topologicall …
  -/
  ext X Y f; rw [universallyInjective_iff]
             /-
               🎉 no goals
             -/


theorem universallyInjective_eq_diagonal :
    @UniversallyInjective = diagonal @Surjective := by
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyInjective) (CategoryTheory.MorphismPropert …
  -/
  apply le_antisymm
    /-
      case a
      ⊢ LE.le (@AlgebraicGeometry.UniversallyInjective) (CategoryTheory.MorphismProp …
    -/
  · intro X Y f hf
    /-
      case a
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.UniversallyInjective f
      ⊢ CategoryTheory.MorphismProperty.diagonal (@AlgebraicGeometry.Surjective) f
    -/
    refine ⟨fun x ↦ ⟨(pullback.fst f f).base x, hf.1 _ _ _ (IsPullback.of_hasPullback f f) ?_⟩⟩
    /-
      case a
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.UniversallyInjective f
      x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
      ⊢ Eq ((CategoryTheory.Limits.pullback.fst f f).base ((CategoryTheory.Limits.pu …
    -/
    rw [← Scheme.comp_base_apply, pullback.diagonal_fst]
    /-
      case a
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.UniversallyInjective f
      x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
      ⊢ Eq ((CategoryTheory.CategoryStruct.id X).base ((CategoryTheory.Limits.pullba …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · rw [← universally_eq_iff.mpr (inferInstanceAs (IsStableUnderBaseChange (diagonal @Surjective))),
      universallyInjective_eq]
    /-
      case a
      ⊢ LE.le (CategoryTheory.MorphismProperty.diagonal @AlgebraicGeometry.Surjectiv …
    -/
    apply universally_mono
    /-
      case a.a
      ⊢ LE.le (CategoryTheory.MorphismProperty.diagonal @AlgebraicGeometry.Surjectiv …
    -/
    intro X Y f hf x₁ x₂ e
    /-
      case a.a
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : CategoryTheory.MorphismProperty.diagonal (@AlgebraicGeometry.Surjective) f
      x₁ x₂ : ↑↑X.toPresheafedSpace
      e : Eq (f.base x₁) (f.base x₂)
      ⊢ Eq x₁ x₂
    -/
    obtain ⟨t, ht₁, ht₂⟩ := Scheme.Pullback.exists_preimage_pullback _ _ e
    /-
      case a.a.intro.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : CategoryTheory.MorphismProperty.diagonal (@AlgebraicGeometry.Surjective) f
      x₁ x₂ : ↑↑X.toPresheafedSpace
      e : Eq (f.base x₁) (f.base x₂)
      t : ↑↑(CategoryTheory.Limits.pullback f f).toPresheafedSpace
      ht₁ : Eq ((CategoryTheory.Limits.pullback.fst f f).base t) x₁
      ht₂ : Eq ((CategoryTheory.Limits.pullback.snd f f).base t) x₂
      ⊢ Eq x₁ x₂
    -/
    obtain ⟨t, rfl⟩ := hf.1 t
    rw [← ht₁, ← ht₂, ← Scheme.comp_base_apply, ← Scheme.comp_base_apply, pullback.diagonal_fst,
      pullback.diagonal_snd]


theorem UniversallyInjective.iff_diagonal :
    UniversallyInjective f ↔ Surjective (pullback.diagonal f) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.UniversallyInjective f) (AlgebraicGeometry.Surjective …
  -/
  rw [universallyInjective_eq_diagonal]; rfl
                                         /-
                                           🎉 no goals
                                         -/


instance (priority := 900) [Mono f] : UniversallyInjective f :=
  have := (pullback.isIso_diagonal_iff f).mpr inferInstance
  (UniversallyInjective.iff_diagonal f).mpr inferInstance


theorem UniversallyInjective.respectsIso : RespectsIso @UniversallyInjective :=
  universallyInjective_eq_diagonal.symm ▸ inferInstance


instance UniversallyInjective.isStableUnderBaseChange :
    IsStableUnderBaseChange @UniversallyInjective :=
  universallyInjective_eq_diagonal.symm ▸ inferInstance


instance universallyInjective_isStableUnderComposition :
    IsStableUnderComposition @UniversallyInjective :=
  universallyInjective_eq ▸ inferInstance


instance : MorphismProperty.IsMultiplicative @UniversallyInjective where
  id_mem _ := inferInstance


instance universallyInjective_isLocalAtTarget : IsLocalAtTarget @UniversallyInjective :=
  universallyInjective_eq_diagonal.symm ▸ inferInstance


