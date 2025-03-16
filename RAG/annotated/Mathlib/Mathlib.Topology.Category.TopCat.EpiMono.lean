theorem epi_iff_surjective {X Y : TopCat.{u}} (f : X ⟶ Y) : Epi f ↔ Function.Surjective f := by
  suffices Epi f ↔ Epi ((forget TopCat).map f) by
    rw [this, CategoryTheory.epi_iff_surjective]
    rfl
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (CategoryTheory.Epi ((CategoryTheory.forget TopCa …
  -/
  constructor
    /-
      case mp
      X Y : TopCat
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → CategoryTheory.Epi ((CategoryTheory.forget TopCat).ma …
    -/
  · intro
    /-
      case mp
      X Y : TopCat
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget TopCat).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget TopCat).map f) → CategoryTheory.E …
    -/
  · apply Functor.epi_of_epi_map
    /-
      🎉 no goals
    -/


theorem mono_iff_injective {X Y : TopCat.{u}} (f : X ⟶ Y) : Mono f ↔ Function.Injective f := by
  suffices Mono f ↔ Mono ((forget TopCat).map f) by
    rw [this, CategoryTheory.mono_iff_injective]
    rfl
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (CategoryTheory.Mono ((CategoryTheory.forget Top …
  -/
  constructor
    /-
      case mp
      X Y : TopCat
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono f → CategoryTheory.Mono ((CategoryTheory.forget TopCat). …
    -/
  · intro
    /-
      case mp
      X Y : TopCat
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget TopCat).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget TopCat).map f) → CategoryTheory. …
    -/
  · apply Functor.mono_of_mono_map
    /-
      🎉 no goals
    -/


