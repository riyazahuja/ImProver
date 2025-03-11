instance functorHasRightDual [RightRigidCategory D] (F : C ⥤ D) : HasRightDual F where
  rightDual :=
    { obj := fun X => (F.obj X)ᘁ
      map := fun f => (F.map (inv f))ᘁ
                                /-
                                  C : Type u_1
                                  D : Type u_2
                                  inst✝³ : CategoryTheory.Groupoid C
                                  inst✝² : CategoryTheory.Category.{?u.43, u_2} D
                                  inst✝¹ : CategoryTheory.MonoidalCategory D
                                  inst✝ : CategoryTheory.RightRigidCategory D
                                  F : CategoryTheory.Functor C D
                                  X✝ Y✝ Z✝ : C
                                  f : Quiver.Hom X✝ Y✝
                                  g : Quiver.Hom Y✝ Z✝
                                  ⊢ Eq ({ obj := fun X => CategoryTheory.HasRightDual.rightDual (F.obj X), map : …
                                -/
      map_comp := fun f g => by simp [comp_rightAdjointMate] }
                                /-
                                  🎉 no goals
                                -/
  exact :=
    { evaluation' :=
        { app := fun _ => ε_ _ _
          naturality := fun X Y f => by
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.43, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.RightRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
            -/
            dsimp
            rw [Category.comp_id, Functor.map_inv, ← id_tensor_comp_tensor_id, Category.assoc,
              id_tensorHom, tensorHom_id,
              rightAdjointMate_comp_evaluation, ← MonoidalCategory.whiskerLeft_comp_assoc,
              IsIso.hom_inv_id, MonoidalCategory.whiskerLeft_id, Category.id_comp] }
      coevaluation' :=
        { app := fun _ => η_ _ _
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.43, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.RightRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
            -/
          naturality := fun X Y f => by
            dsimp
            rw [Functor.map_inv, Category.id_comp, ← id_tensor_comp_tensor_id,
              id_tensorHom, tensorHom_id, ← Category.assoc,
              coevaluation_comp_rightAdjointMate, Category.assoc, ← comp_whiskerRight,
              IsIso.inv_hom_id, id_whiskerRight, Category.comp_id] } }


instance rightRigidFunctorCategory [RightRigidCategory D] : RightRigidCategory (C ⥤ D) where


instance functorHasLeftDual [LeftRigidCategory D] (F : C ⥤ D) : HasLeftDual F where
  leftDual :=
    { obj := fun X => ᘁ(F.obj X)
      map := fun f => ᘁ(F.map (inv f))
                                /-
                                  C : Type u_1
                                  D : Type u_2
                                  inst✝³ : CategoryTheory.Groupoid C
                                  inst✝² : CategoryTheory.Category.{?u.31086, u_2} D
                                  inst✝¹ : CategoryTheory.MonoidalCategory D
                                  inst✝ : CategoryTheory.LeftRigidCategory D
                                  F : CategoryTheory.Functor C D
                                  X✝ Y✝ Z✝ : C
                                  f : Quiver.Hom X✝ Y✝
                                  g : Quiver.Hom Y✝ Z✝
                                  ⊢ Eq ({ obj := fun X => CategoryTheory.HasLeftDual.leftDual (F.obj X), map :=  …
                                -/
      map_comp := fun f g => by simp [comp_leftAdjointMate] }
                                /-
                                  🎉 no goals
                                -/
  exact :=
    { evaluation' :=
        { app := fun _ => ε_ _ _
          naturality := fun X Y f => by
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.31086, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.LeftRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
            -/
            dsimp
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.31086, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.LeftRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
            -/
            simp [tensorHom_def, leftAdjointMate_comp_evaluation] }
            /-
              🎉 no goals
            -/
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.31086, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.LeftRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
            -/
      coevaluation' :=
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Groupoid C
              inst✝² : CategoryTheory.Category.{?u.31086, u_2} D
              inst✝¹ : CategoryTheory.MonoidalCategory D
              inst✝ : CategoryTheory.LeftRigidCategory D
              F : CategoryTheory.Functor C D
              X Y : C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Cat …
            -/
        { app := fun _ => η_ _ _
            /-
              🎉 no goals
            -/
          naturality := fun X Y f => by
            dsimp
            simp [tensorHom_def, coevaluation_comp_leftAdjointMate_assoc] } }


instance leftRigidFunctorCategory [LeftRigidCategory D] : LeftRigidCategory (C ⥤ D) where


instance rigidFunctorCategory [RigidCategory D] : RigidCategory (C ⥤ D) where


