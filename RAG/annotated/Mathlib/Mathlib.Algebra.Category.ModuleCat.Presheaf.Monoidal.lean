instance (X : Cᵒᵖ) : CommRing ((R ⋙ forget₂ _ RingCat).obj X) :=
  inferInstanceAs (CommRing (R.obj X))


/-- Auxiliary definition for `tensorObj`. -/
noncomputable def tensorObjMap {X Y : Cᵒᵖ} (f : X ⟶ Y) : M₁.obj X ⊗ M₂.obj X ⟶
    (ModuleCat.restrictScalars (R.map f).hom).obj (M₁.obj Y ⊗ M₂.obj Y) :=
  ModuleCat.MonoidalCategory.tensorLift (fun m₁ m₂ ↦ M₁.map f m₁ ⊗ₜ M₂.map f m₂)
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.5138, u_1} C
          R : CategoryTheory.Functor (Opposite C) CommRingCat
          M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ ∀ (m₁ m₂ : ↑(M₁.obj X)) (n : ↑(M₂.obj X)), Eq ((fun m₁ m₂ => TensorProduct.t …
        -/
    (by intro m₁ m₁' m₂; dsimp; rw [map_add, TensorProduct.add_tmul])
                                /-
                                  🎉 no goals
                                -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.5138, u_1} C
          R : CategoryTheory.Functor (Opposite C) CommRingCat
          M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ ∀ (a : ↑((R.comp (CategoryTheory.forget₂ CommRingCat RingCat)).obj X)) (m :  …
        -/
    (by intro a m₁ m₂; dsimp; erw [M₁.map_smul]; rfl)
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.5138, u_1} C
          R : CategoryTheory.Functor (Opposite C) CommRingCat
          M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ ∀ (m : ↑(M₁.obj X)) (n₁ n₂ : ↑(M₂.obj X)), Eq ((fun m₁ m₂ => TensorProduct.t …
        -/
    (by intro m₁ m₂ m₂'; dsimp; rw [map_add, TensorProduct.tmul_add])
                                /-
                                  🎉 no goals
                                -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.5138, u_1} C
          R : CategoryTheory.Functor (Opposite C) CommRingCat
          M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ ∀ (a : ↑((R.comp (CategoryTheory.forget₂ CommRingCat RingCat)).obj X)) (m :  …
        -/
    (by intro a m₁ m₂; dsimp; erw [M₂.map_smul, TensorProduct.tmul_smul (r := R.map f a)]; rfl)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


/-- The tensor product of two presheaves of modules. -/
@[simps obj]
noncomputable def tensorObj : PresheafOfModules (R ⋙ forget₂ _ _) where
  obj X := M₁.obj X ⊗ M₂.obj X
  map f := tensorObjMap M₁ M₂ f
  map_id X := ModuleCat.MonoidalCategory.tensor_ext (by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X : Opposite C
      ⊢ ∀ (m : ↑(M₁.obj X)) (n : ↑(M₂.obj X)), Eq (((fun {X Y} f => PresheafOfModule …
    -/
    intro m₁ m₂
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X : Opposite C
      m₁ : ↑(M₁.obj X)
      m₂ : ↑(M₂.obj X)
      ⊢ Eq (((fun {X Y} f => PresheafOfModules.Monoidal.tensorObjMap M₁ M₂ f) (Categ …
    -/
    dsimp [tensorObjMap]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X : Opposite C
      m₁ : ↑(M₁.obj X)
      m₂ : ↑(M₂.obj X)
      ⊢ Eq (TensorProduct.tmul (↑(R.obj X)) ((M₁.map (CategoryTheory.CategoryStruct. …
    -/
    simp)
    /-
      🎉 no goals
    -/
  map_comp f g := ModuleCat.MonoidalCategory.tensor_ext (by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X✝ Y✝ Z✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ ∀ (m : ↑(M₁.obj X✝)) (n : ↑(M₂.obj X✝)), Eq (((fun {X Y} f => PresheafOfModu …
    -/
    intro m₁ m₂
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X✝ Y✝ Z✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      m₁ : ↑(M₁.obj X✝)
      m₂ : ↑(M₂.obj X✝)
      ⊢ Eq (((fun {X Y} f => PresheafOfModules.Monoidal.tensorObjMap M₁ M₂ f) (Categ …
    -/
    dsimp [tensorObjMap]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.49699, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      X✝ Y✝ Z✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      m₁ : ↑(M₁.obj X✝)
      m₂ : ↑(M₂.obj X✝)
      ⊢ Eq (TensorProduct.tmul (↑(R.obj Z✝)) ((M₁.map (CategoryTheory.CategoryStruct …
    -/
    simp)
    /-
      🎉 no goals
    -/


@[simp]
lemma tensorObj_map_tmul {X Y : Cᵒᵖ} (f : X ⟶ Y) (m₁ : M₁.obj X) (m₂ : M₂.obj X) :
    DFunLike.coe (α := (M₁.obj X ⊗ M₂.obj X : _))
      (β := fun _ ↦ (ModuleCat.restrictScalars (R.map f).hom).obj (M₁.obj Y ⊗ M₂.obj Y))
      ((tensorObj M₁ M₂).map f).hom (m₁ ⊗ₜ[R.obj X] m₂) = M₁.map f m₁ ⊗ₜ[R.obj Y] M₂.map f m₂ := rfl


/-- The tensor product of two morphisms of presheaves of modules. -/
@[simps]
noncomputable def tensorHom (f : M₁ ⟶ M₂) (g : M₃ ⟶ M₄) : tensorObj M₁ M₃ ⟶ tensorObj M₂ M₄ where
  app X := f.app X ⊗ g.app X
  naturality {X Y} φ := ModuleCat.MonoidalCategory.tensor_ext (fun m₁ m₃ ↦ by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.113004, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ M₃ M₄ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat Ri …
      f : Quiver.Hom M₁ M₂
      g : Quiver.Hom M₃ M₄
      X Y : Opposite C
      φ : Quiver.Hom X Y
      m₁ : ↑(M₁.obj X)
      m₃ : ↑(M₃.obj X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((PresheafOfModules.Monoidal.tensorO …
    -/
    dsimp
    rw [tensorObj_map_tmul, ModuleCat.MonoidalCategory.tensorHom_tmul, tensorObj_map_tmul,
      naturality_apply, naturality_apply])


open ModuleCat.MonoidalCategory in
noncomputable instance monoidalCategoryStruct :
    MonoidalCategoryStruct (PresheafOfModules.{u} (R ⋙ forget₂ _ _)) where
  tensorObj := tensorObj
  whiskerLeft _ _ _ g := tensorHom (𝟙 _) g
  whiskerRight f _ := tensorHom f (𝟙 _)
  tensorHom := tensorHom
  tensorUnit := unit _
  associator M₁ M₂ M₃ := isoMk (fun _ ↦ α_ _ _ _)
                                                             /-
                                                               C : Type u_1
                                                               inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
                                                               R : CategoryTheory.Functor (Opposite C) CommRingCat
                                                               M₁ M₂ M₃ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingC …
                                                               x✝² x✝¹ : Opposite C
                                                               x✝ : Quiver.Hom x✝² x✝¹
                                                               ⊢ ∀ (m₁ : ↑(M₁.obj x✝²)) (m₂ : ↑(M₂.obj x✝²)) (m₃ : ↑(M₃.obj x✝²)), Eq ((Categ …
                                                             -/
    (fun _ _ _ ↦ ModuleCat.MonoidalCategory.tensor_ext₃' (by intros; rfl))
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  leftUnitor M := Iso.symm (isoMk (fun _ ↦ (λ_ _).symm) (fun X Y f ↦ by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.map f) ((ModuleCat.restrictScalars …
    -/
    ext m
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M.map f) ((ModuleCat.restrictScalar …
    -/
    dsimp [CommRingCat.forgetToRingCat_obj]
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.leftUnitor (M.obj Y)).inv.hom ((M …
    -/
    erw [leftUnitor_inv_apply, leftUnitor_inv_apply, tensorObj_map_tmul, (R.map f).hom.map_one]
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq (TensorProduct.tmul (↑(R.obj Y)) 1 ((M.map f).hom m)) (TensorProduct.tmul …
    -/
    rfl))
    /-
      🎉 no goals
    -/
  rightUnitor M := Iso.symm (isoMk (fun _ ↦ (ρ_ _).symm) (fun X Y f ↦ by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.map f) ((ModuleCat.restrictScalars …
    -/
    ext m
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (M.map f) ((ModuleCat.restrictScalar …
    -/
    dsimp [CommRingCat.forgetToRingCat_obj]
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.rightUnitor (M.obj Y)).inv.hom (( …
    -/
    erw [rightUnitor_inv_apply, rightUnitor_inv_apply, tensorObj_map_tmul, (R.map f).hom.map_one]
    /-
      case hf.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.122690, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X Y : Opposite C
      f : Quiver.Hom X Y
      m : ↑(M.obj X)
      ⊢ Eq (TensorProduct.tmul (↑(R.obj Y)) ((M.map f).hom m) 1) (TensorProduct.tmul …
    -/
    rfl))
    /-
      🎉 no goals
    -/


noncomputable instance monoidalCategory :
    MonoidalCategory (PresheafOfModules.{u} (R ⋙ forget₂ _ _)) where
                          /-
                            C : Type u_1
                            inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                            R : CategoryTheory.Functor (Opposite C) CommRingCat
                            X₁✝ Y₁✝ X₂✝ Y₂✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCa …
                            x✝¹ : Quiver.Hom X₁✝ Y₁✝
                            x✝ : Quiver.Hom X₂✝ Y₂✝
                            ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom x✝¹ x✝) (CategoryTheory. …
                          -/
  tensorHom_def _ _ := by ext1; apply tensorHom_def
                                /-
                                  🎉 no goals
                                -/
                      /-
                        C : Type u_1
                        inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                        R : CategoryTheory.Functor (Opposite C) CommRingCat
                        x✝¹ x✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
                      -/
  tensor_id _ _ := by ext1; apply tensor_id
                            /-
                              🎉 no goals
                            -/
                            /-
                              C : Type u_1
                              inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                              R : CategoryTheory.Functor (Opposite C) CommRingCat
                              X₁✝ Y₁✝ Z₁✝ X₂✝ Y₂✝ Z₂✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ Co …
                              x✝³ : Quiver.Hom X₁✝ Y₁✝
                              x✝² : Quiver.Hom X₂✝ Y₂✝
                              x✝¹ : Quiver.Hom Y₁✝ Z₁✝
                              x✝ : Quiver.Hom Y₂✝ Z₂✝
                              ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
                            -/
  tensor_comp _ _ _ _ := by ext1; apply tensor_comp
                                  /-
                                    🎉 no goals
                                  -/
  whiskerLeft_id M₁ M₂ := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft M₁ (CategoryTheory.Cat …
    -/
    ext1 X
    /-
      case h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      M₁ M₂ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X : Opposite C
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerLeft M₁ (CategoryTheory.Ca …
    -/
    apply MonoidalCategory.whiskerLeft_id (C := ModuleCat (R.obj X))
    /-
      🎉 no goals
    -/
  id_whiskerRight _ _ := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      x✝¹ x✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
    -/
    ext1 X
    /-
      case h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
      R : CategoryTheory.Functor (Opposite C) CommRingCat
      x✝¹ x✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
      X : Opposite C
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Cate …
    -/
    apply MonoidalCategory.id_whiskerRight (C := ModuleCat (R.obj X))
    /-
      🎉 no goals
    -/
                                    /-
                                      C : Type u_1
                                      inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                                      R : CategoryTheory.Functor (Opposite C) CommRingCat
                                      X₁✝ X₂✝ X₃✝ Y₁✝ Y₂✝ Y₃✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ Co …
                                      x✝² : Quiver.Hom X₁✝ Y₁✝
                                      x✝¹ : Quiver.Hom X₂✝ Y₂✝
                                      x✝ : Quiver.Hom X₃✝ Y₃✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                    -/
  associator_naturality _ _ _ := by ext1; apply associator_naturality
                                          /-
                                            🎉 no goals
                                          -/
                                /-
                                  C : Type u_1
                                  inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                                  R : CategoryTheory.Functor (Opposite C) CommRingCat
                                  X✝ Y✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                                  x✝ : Quiver.Hom X✝ Y✝
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                -/
  leftUnitor_naturality _ := by ext1; apply leftUnitor_naturality
                                      /-
                                        🎉 no goals
                                      -/
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                                   R : CategoryTheory.Functor (Opposite C) CommRingCat
                                   X✝ Y✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                                   x✝ : Quiver.Hom X✝ Y✝
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                 -/
  rightUnitor_naturality _ := by ext1; apply rightUnitor_naturality
                                       /-
                                         🎉 no goals
                                       -/
                         /-
                           C : Type u_1
                           inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                           R : CategoryTheory.Functor (Opposite C) CommRingCat
                           x✝³ x✝² x✝¹ x✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat …
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                         -/
  pentagon _ _ _ _ := by ext1; apply pentagon
                               /-
                                 🎉 no goals
                               -/
                     /-
                       C : Type u_1
                       inst✝ : CategoryTheory.Category.{?u.142406, u_1} C
                       R : CategoryTheory.Functor (Opposite C) CommRingCat
                       x✝¹ x✝ : PresheafOfModules (R.comp (CategoryTheory.forget₂ CommRingCat RingCat))
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                     -/
  triangle _ _ := by ext1; apply triangle
                           /-
                             🎉 no goals
                           -/


