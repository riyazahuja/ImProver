instance (C : Type u) [Category.{v} C] [MonoidalCategory C] :
    (coyoneda.obj (op (𝟙_ C))).LaxMonoidal :=
  Functor.LaxMonoidal.ofTensorHom
    (ε' := fun _ => 𝟙 _)
    (μ' := fun X Y p ↦ (λ_ (𝟙_ C)).inv ≫ (p.1 ⊗ p.2))
                      /-
                        C : Type u
                        inst✝¹ : CategoryTheory.Category.{v, u} C
                        inst✝ : CategoryTheory.MonoidalCategory C
                        ⊢ ∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom X' Y'), Eq (CategoryT …
                      -/
    (μ'_natural := by aesop_cat)
                      /-
                        🎉 no goals
                      -/
    (associativity' := fun X Y Z => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y Z : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      ext ⟨⟨f, g⟩, h⟩; dsimp at f g h
      /-
        case h.mk.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y Z : C
        h : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Z
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X
        g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp; simp only [Iso.cancel_iso_inv_left, Category.assoc]
      conv_lhs =>
        rw [← Category.id_comp h, tensor_comp, Category.assoc, associator_naturality,
          ← Category.assoc, unitors_inv_equal, tensorHom_id, triangle_assoc_comp_right_inv]
      /-
        case h.mk.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y Z : C
        h : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Z
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X
        g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      conv_rhs => rw [← Category.id_comp f, tensor_comp]
      /-
        case h.mk.mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X Y Z : C
        h : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Z
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X
        g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp)
      /-
        🎉 no goals
      -/
    (left_unitality' := by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        ⊢ ∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((CategoryTh …
      -/
      intros
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X✝ : C
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((CategoryTheory.coyone …
      -/
      ext ⟨⟨⟩, f⟩; dsimp at f
      /-
        case h.mk.unit
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X✝ : C
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X✝
        ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.leftUnitor ((CategoryTheory.coyon …
      -/
      dsimp
      /-
        case h.mk.unit
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X✝ : C
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X✝
        ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      -/
      simp)
      /-
        🎉 no goals
      -/
    (right_unitality' := fun X => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X : C
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((CategoryTheory.coyon …
      -/
      ext ⟨f, ⟨⟩⟩; dsimp at f
      /-
        case h.mk.unit
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X : C
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X
        ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.rightUnitor ((CategoryTheory.coyo …
      -/
      dsimp
      /-
        case h.mk.unit
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.MonoidalCategory C
        X : C
        f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit X
        ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      -/
      simp [unitors_inv_equal])
      /-
        🎉 no goals
      -/


