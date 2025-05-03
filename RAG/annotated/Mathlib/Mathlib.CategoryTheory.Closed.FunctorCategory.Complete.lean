private abbrev incl : Discrete I ⥤ I := Discrete.functor id


instance : ReflectsIsomorphisms <| (whiskeringLeft _ _ C).obj (incl I) where
  reflects f h := by
    /-
      I : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} I
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      inst✝¹ : CategoryTheory.MonoidalClosed C
      inst✝ : ∀ (F : CategoryTheory.Functor (CategoryTheory.Discrete I) C), (Categor …
      A✝ B✝ : CategoryTheory.Functor I C
      f : Quiver.Hom A✝ B✝
      h : CategoryTheory.IsIso (((CategoryTheory.whiskeringLeft (CategoryTheory.Disc …
      ⊢ CategoryTheory.IsIso f
    -/
    simp only [NatTrans.isIso_iff_isIso_app] at *
    /-
      I : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} I
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      inst✝¹ : CategoryTheory.MonoidalClosed C
      inst✝ : ∀ (F : CategoryTheory.Functor (CategoryTheory.Discrete I) C), (Categor …
      A✝ B✝ : CategoryTheory.Functor I C
      f : Quiver.Hom A✝ B✝
      h : ∀ (X : CategoryTheory.Discrete I), CategoryTheory.IsIso ((((CategoryTheory …
      ⊢ ∀ (X : I), CategoryTheory.IsIso (f.app X)
    -/
    intro X
    /-
      I : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} I
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      inst✝¹ : CategoryTheory.MonoidalClosed C
      inst✝ : ∀ (F : CategoryTheory.Functor (CategoryTheory.Discrete I) C), (Categor …
      A✝ B✝ : CategoryTheory.Functor I C
      f : Quiver.Hom A✝ B✝
      h : ∀ (X : CategoryTheory.Discrete I), CategoryTheory.IsIso ((((CategoryTheory …
      X : I
      ⊢ CategoryTheory.IsIso (f.app X)
    -/
    exact h ⟨X⟩
    /-
      🎉 no goals
    -/


instance : Comonad.PreservesLimitOfIsCoreflexivePair ((whiskeringLeft _ _ C).obj (incl I)) :=
  ⟨inferInstance⟩


instance : ComonadicLeftAdjoint ((whiskeringLeft _ _ C).obj (incl I)) :=
  Comonad.comonadicOfHasPreservesCoreflexiveEqualizersOfReflectsIsomorphisms
    ((incl I).ranAdjunction C)


instance (F : I ⥤ C) : IsLeftAdjoint (tensorLeft (incl I ⋙ F)) :=
  (ihom.adjunction (incl I ⋙ F)).isLeftAdjoint


/-- Auxiliary definition for `functorCategoryMonoidalClosed` -/
def functorCategoryClosed (F : I ⥤ C) : Closed F :=
  have := (ihom.adjunction (incl I ⋙ F)).isLeftAdjoint
  have := isLeftAdjoint_square_lift_comonadic (tensorLeft F) ((whiskeringLeft _ _ C).obj (incl I))
    ((whiskeringLeft _ _ C).obj (incl I)) (tensorLeft (incl I ⋙ F)) (Iso.refl _)
  { rightAdj := (tensorLeft F).rightAdjoint
    adj := Adjunction.ofIsLeftAdjoint (tensorLeft F) }


/--
Assuming the existence of certain limits, functors into a monoidal closed category form a
monoidal closed category.

Note: this is defined completely abstractly, and does not have any good definitional properties.
See the TODO in the module docstring.
-/
def functorCategoryMonoidalClosed : MonoidalClosed (I ⥤ C) where
  closed F := functorCategoryClosed I C F


