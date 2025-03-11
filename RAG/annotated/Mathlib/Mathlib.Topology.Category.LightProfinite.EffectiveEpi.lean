theorem effectiveEpi_iff_surjective {X Y : LightProfinite.{u}} (f : X ⟶ Y) :
    EffectiveEpi f ↔ Function.Surjective f := by
  /-
    X Y : LightProfinite
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.EffectiveEpi f) (Function.Surjective ⇑f)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ⟨⟨effectiveEpiStruct f h⟩⟩⟩
  /-
    X Y : LightProfinite
    f : Quiver.Hom X Y
    h : CategoryTheory.EffectiveEpi f
    ⊢ Function.Surjective ⇑f
  -/
  rw [← epi_iff_surjective]
  /-
    X Y : LightProfinite
    f : Quiver.Hom X Y
    h : CategoryTheory.EffectiveEpi f
    ⊢ CategoryTheory.Epi f
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Preregular LightProfinite.{u} := by
  /-
    ⊢ CategoryTheory.Preregular LightProfinite
  -/
  apply CompHausLike.preregular
  /-
    case hs
    ⊢ ∀ ⦃X Y : CompHausLike fun X => And (TotallyDisconnectedSpace ↑X) (SecondCoun …
  -/
  intro _ _ f
  /-
    case hs
    X✝ Y✝ : CompHausLike fun X => And (TotallyDisconnectedSpace ↑X) (SecondCountab …
    f : Quiver.Hom X✝ Y✝
    ⊢ CategoryTheory.EffectiveEpi f → Function.Surjective ⇑f
  -/
  exact (effectiveEpi_iff_surjective f).mp
  /-
    🎉 no goals
  -/


