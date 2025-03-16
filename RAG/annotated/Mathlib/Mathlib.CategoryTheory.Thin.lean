/-- Construct a category instance from a category_struct, using the fact that
    hom spaces are subsingletons to prove the axioms. -/
def thin_category : Category C where


/-- If `C` is a thin category, then `D ⥤ C` is a thin category. -/
instance functor_thin : Quiver.IsThin (D ⥤ C) := fun _ _ =>
                               /-
                                 C : Type u₁
                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                 D : Type u₂
                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                 inst✝ : Quiver.IsThin C
                                 x✝¹ x✝ : CategoryTheory.Functor D C
                                 α β : Quiver.Hom x✝¹ x✝
                                 ⊢ Eq α.app β.app
                               -/
  ⟨fun α β => NatTrans.ext (by subsingleton)⟩
                               /-
                                 🎉 no goals
                               -/


/-- To show `X ≅ Y` in a thin category, it suffices to just give any morphism in each direction. -/
def iso_of_both_ways {X Y : C} (f : X ⟶ Y) (g : Y ⟶ X) :
    X ≅ Y where
  hom := f
  inv := g


instance subsingleton_iso {X Y : C} : Subsingleton (X ≅ Y) :=
  ⟨by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : Quiver.IsThin C
      X Y : C
      ⊢ ∀ (a b : CategoryTheory.Iso X Y), Eq a b
    -/
    intro i₁ i₂
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : Quiver.IsThin C
      X Y : C
      i₁ i₂ : CategoryTheory.Iso X Y
      ⊢ Eq i₁ i₂
    -/
    ext1
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : Quiver.IsThin C
      X Y : C
      i₁ i₂ : CategoryTheory.Iso X Y
      ⊢ Eq i₁.hom i₂.hom
    -/
    subsingleton⟩
    /-
      🎉 no goals
    -/


