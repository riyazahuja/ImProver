/-- The endomorphisms of an object in a bicategory can be considered as a monoidal category. -/
def EndMonoidal (X : C) :=
  X ⟶ X -- deriving Category

-- Porting note: Deriving this fails in the definition above.
-- Adding category instance manually.

instance (X : C) : Category (EndMonoidal X) :=
  show Category (X ⟶ X) from inferInstance


instance (X : C) : Inhabited (EndMonoidal X) :=
  ⟨𝟙 X⟩


attribute [local simp] EndMonoidal in
instance (X : C) : MonoidalCategory (EndMonoidal X) where
  tensorObj f g := f ≫ g
  whiskerLeft {f _ _} η := f ◁ η
  whiskerRight {_ _} η h := η ▷ h
  tensorUnit := 𝟙 _
  associator f g h := α_ f g h
  leftUnitor f := λ_ f
  rightUnitor f := ρ_ f
  tensor_comp := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Bicategory C
      X : C
      ⊢ ∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : CategoryTheory.EndMonoidal X} (f₁ : Quiver.Hom X₁ Y₁) …
    -/
    intros
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Bicategory C
      X : C
      X₁✝ Y₁✝ Z₁✝ X₂✝ Y₂✝ Z₂✝ : CategoryTheory.EndMonoidal X
      f₁✝ : Quiver.Hom X₁✝ Y₁✝
      f₂✝ : Quiver.Hom X₂✝ Y₂✝
      g₁✝ : Quiver.Hom Y₁✝ Z₁✝
      g₂✝ : Quiver.Hom Y₂✝ Z₂✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
    -/
    dsimp
    rw [Bicategory.whiskerLeft_comp, Bicategory.comp_whiskerRight, Category.assoc, Category.assoc,
      Bicategory.whisker_exchange_assoc]


