lemma epi_of_surjective (hf : ∀ ⦃X : Cᵒᵖ⦄, Function.Surjective (f.app X)) : Epi f where
  left_cancellation g₁ g₂ hg := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M₁ M₂ : PresheafOfModules R
      f : Quiver.Hom M₁ M₂
      hf : ∀ ⦃X : Opposite C⦄, Function.Surjective ⇑(f.app X).hom
      Z✝ : PresheafOfModules R
      g₁ g₂ : Quiver.Hom M₂ Z✝
      hg : Eq (CategoryTheory.CategoryStruct.comp f g₁) (CategoryTheory.CategoryStru …
      ⊢ Eq g₁ g₂
    -/
    ext X m₂
    /-
      case h.hf.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M₁ M₂ : PresheafOfModules R
      f : Quiver.Hom M₁ M₂
      hf : ∀ ⦃X : Opposite C⦄, Function.Surjective ⇑(f.app X).hom
      Z✝ : PresheafOfModules R
      g₁ g₂ : Quiver.Hom M₂ Z✝
      hg : Eq (CategoryTheory.CategoryStruct.comp f g₁) (CategoryTheory.CategoryStru …
      X : Opposite C
      m₂ : ↑(M₂.obj X)
      ⊢ Eq ((g₁.app X).hom m₂) ((g₂.app X).hom m₂)
    -/
    obtain ⟨m₁, rfl⟩ := hf m₂
    /-
      case h.hf.h.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M₁ M₂ : PresheafOfModules R
      f : Quiver.Hom M₁ M₂
      hf : ∀ ⦃X : Opposite C⦄, Function.Surjective ⇑(f.app X).hom
      Z✝ : PresheafOfModules R
      g₁ g₂ : Quiver.Hom M₂ Z✝
      hg : Eq (CategoryTheory.CategoryStruct.comp f g₁) (CategoryTheory.CategoryStru …
      X : Opposite C
      m₁ : ↑(M₁.obj X)
      ⊢ Eq ((g₁.app X).hom ((f.app X).hom m₁)) ((g₂.app X).hom ((f.app X).hom m₁))
    -/
    exact congr_fun ((evaluation R X ⋙ forget _).congr_map hg) m₁
    /-
      🎉 no goals
    -/


lemma mono_of_injective (hf : ∀ ⦃X : Cᵒᵖ⦄, Function.Injective (f.app X)) : Mono f where
  right_cancellation {M} g₁ g₂ hg := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M₁ M₂ : PresheafOfModules R
      f : Quiver.Hom M₁ M₂
      hf : ∀ ⦃X : Opposite C⦄, Function.Injective ⇑(f.app X).hom
      M : PresheafOfModules R
      g₁ g₂ : Quiver.Hom M M₁
      hg : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
      ⊢ Eq g₁ g₂
    -/
    ext X m
    /-
      case h.hf.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      M₁ M₂ : PresheafOfModules R
      f : Quiver.Hom M₁ M₂
      hf : ∀ ⦃X : Opposite C⦄, Function.Injective ⇑(f.app X).hom
      M : PresheafOfModules R
      g₁ g₂ : Quiver.Hom M M₁
      hg : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
      X : Opposite C
      m : ↑(M.obj X)
      ⊢ Eq ((g₁.app X).hom m) ((g₂.app X).hom m)
    -/
    exact hf (congr_fun ((evaluation R X ⋙ forget _).congr_map hg) m)
    /-
      🎉 no goals
    -/


instance [Epi f] (X : Cᵒᵖ) : Epi (f.app X) :=
  inferInstanceAs (Epi ((evaluation R X).map f))


instance [Mono f] (X : Cᵒᵖ) : Mono (f.app X) :=
  inferInstanceAs (Mono ((evaluation R X).map f))


lemma surjective_of_epi [Epi f] (X : Cᵒᵖ) :
    Function.Surjective (f.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    f : Quiver.Hom M₁ M₂
    inst✝ : CategoryTheory.Epi f
    X : Opposite C
    ⊢ Function.Surjective ⇑(f.app X).hom
  -/
  rw [← ModuleCat.epi_iff_surjective]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    f : Quiver.Hom M₁ M₂
    inst✝ : CategoryTheory.Epi f
    X : Opposite C
    ⊢ CategoryTheory.Epi (f.app X)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma injective_of_mono [Mono f] (X : Cᵒᵖ) :
    Function.Injective (f.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    f : Quiver.Hom M₁ M₂
    inst✝ : CategoryTheory.Mono f
    X : Opposite C
    ⊢ Function.Injective ⇑(f.app X).hom
  -/
  rw [← ModuleCat.mono_iff_injective]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M₁ M₂ : PresheafOfModules R
    f : Quiver.Hom M₁ M₂
    inst✝ : CategoryTheory.Mono f
    X : Opposite C
    ⊢ CategoryTheory.Mono (f.app X)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma epi_iff_surjective :
    Epi f ↔ ∀ ⦃X : Cᵒᵖ⦄, Function.Surjective (f.app X) :=
  ⟨fun _ ↦ surjective_of_epi f, epi_of_surjective⟩


lemma mono_iff_surjective :
    Mono f ↔ ∀ ⦃X : Cᵒᵖ⦄, Function.Injective (f.app X) :=
  ⟨fun _ ↦ injective_of_mono f, mono_of_injective⟩


