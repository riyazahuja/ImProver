instance : (lim (J := J) (C := C)).LaxMonoidal :=
  Functor.LaxMonoidal.ofTensorHom
    (ε' :=
      limit.lift _
        { pt := _
          π := { app := fun _ => 𝟙 _ } })
    (μ' := fun F G ↦
      limit.lift (F ⊗ G)
        { pt := limit F ⊗ limit G
          π :=
            { app := fun j => limit.π F j ⊗ limit.π G j
              naturality := fun j j' f => by
                /-
                  J : Type w
                  inst✝³ : CategoryTheory.SmallCategory J
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                  inst✝ : CategoryTheory.MonoidalCategory C
                  F G : CategoryTheory.Functor J C
                  j j' : J
                  f : Quiver.Hom j j'
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                -/
                dsimp
                /-
                  J : Type w
                  inst✝³ : CategoryTheory.SmallCategory J
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                  inst✝ : CategoryTheory.MonoidalCategory C
                  F G : CategoryTheory.Functor J C
                  j j' : J
                  f : Quiver.Hom j j'
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
                -/
                simp only [Category.id_comp, ← tensor_comp, limit.w] } })
                /-
                  🎉 no goals
                -/
    (μ'_natural := fun f g ↦ limit.hom_ext (fun j ↦ by
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        X✝ Y✝ X'✝ Y'✝ : CategoryTheory.Functor J C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom X'✝ Y'✝
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      simp only [limit.lift_π, Cones.postcompose_obj_π, Monoidal.tensorHom_app, limit.lift_map,
        NatTrans.comp_app, Category.assoc, ← tensor_comp, limMap_π]))
    (associativity' := fun F G H ↦ limit.hom_ext (fun j ↦ by
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F G H : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      simp only [tensorHom_id, limit.lift_map, Category.assoc, limit.lift_π,
        id_tensorHom]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F G H : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      conv_lhs => rw [tensorHom_def, Category.assoc, ← comp_whiskerRight_assoc,
        limit.lift_π, tensor_whiskerLeft, Category.assoc, Category.assoc,
        Iso.inv_hom_id, Category.comp_id,
        ← associator_naturality_right, ← tensorHom_def_assoc]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F G H : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      conv_rhs => rw [tensorHom_def, ← whisker_exchange,
        ← MonoidalCategory.whiskerLeft_comp_assoc, limit.lift_π,
        whisker_exchange, ← associator_naturality_left_assoc]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F G H : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp only
      conv_rhs => rw [tensorHom_def, MonoidalCategory.whiskerLeft_comp,
        ← associator_naturality_middle_assoc,
        ← associator_naturality_right, ← comp_whiskerRight_assoc,
        ← tensorHom_def, ← tensorHom_def_assoc]))
    (left_unitality' := fun F ↦ limit.hom_ext (fun j ↦ by
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp only [tensorHom_id, limit.lift_map, Category.assoc, limit.lift_π]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      simp only [tensorHom_def, id_whiskerLeft, Category.assoc,
        Iso.inv_hom_id, Category.comp_id, ← comp_whiskerRight_assoc]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      erw [limit.lift_π]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rw [id_whiskerRight, Category.id_comp]))
      /-
        🎉 no goals
      -/
    (right_unitality' := fun F ↦ limit.hom_ext (fun j ↦ by
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp only [id_tensorHom, limit.lift_map, Category.assoc, limit.lift_π]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp
      simp only [tensorHom_def, ← whisker_exchange,
        MonoidalCategory.whiskerRight_id, Category.assoc, Iso.inv_hom_id,
        Category.comp_id, ← MonoidalCategory.whiskerLeft_comp_assoc]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      erw [limit.lift_π]
      /-
        J : Type w
        inst✝³ : CategoryTheory.SmallCategory J
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
        inst✝ : CategoryTheory.MonoidalCategory C
        F : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rw [MonoidalCategory.whiskerLeft_id, Category.id_comp]))
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
lemma lim_ε_π (j : J) : ε (lim (J := J) (C := C)) ≫ limit.π _ j = 𝟙 _ :=
  limit.lift_π _ _


@[reassoc (attr := simp)]
lemma lim_μ_π (F G : J ⥤ C) (j : J) : μ lim F G ≫ limit.π _ j = limit.π F j ⊗ limit.π G j :=
  limit.lift_π _ _


