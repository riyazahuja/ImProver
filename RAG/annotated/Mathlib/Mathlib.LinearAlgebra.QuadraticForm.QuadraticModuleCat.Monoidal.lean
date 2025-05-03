/-- Auxiliary definition used to build `QuadraticModuleCat.instMonoidalCategory`. -/
@[simps! form]
noncomputable abbrev tensorObj (X Y : QuadraticModuleCat.{u} R) : QuadraticModuleCat.{u} R :=
  of (X.form.tmul Y.form)


/-- Auxiliary definition used to build `QuadraticModuleCat.instMonoidalCategory`.

We want this up front so that we can re-use it to define `whiskerLeft` and `whiskerRight`. -/
noncomputable abbrev tensorHom {W X Y Z : QuadraticModuleCat.{u} R} (f : W ⟶ X) (g : Y ⟶ Z) :
    tensorObj W Y ⟶ tensorObj X Z :=
  ⟨f.toIsometry.tmul g.toIsometry⟩


instance : MonoidalCategoryStruct (QuadraticModuleCat.{u} R) where
  tensorObj := instMonoidalCategory.tensorObj
  whiskerLeft X _ _ f := tensorHom (𝟙 X) f
  whiskerRight {X₁ X₂} (f : X₁ ⟶ X₂) Y := tensorHom f (𝟙 Y)
  tensorHom := tensorHom
  tensorUnit := of (sq (R := R))
  associator X Y Z := ofIso (tensorAssoc X.form Y.form Z.form)
  leftUnitor X := ofIso (tensorLId X.form)
  rightUnitor X := ofIso (tensorRId X.form)


@[simp] theorem toModuleCat_tensor (X Y : QuadraticModuleCat.{u} R) :
    (X ⊗ Y).toModuleCat = X.toModuleCat ⊗ Y.toModuleCat := rfl


theorem forget₂_map_associator_hom (X Y Z : QuadraticModuleCat.{u} R) :
    (forget₂ (QuadraticModuleCat R) (ModuleCat R)).map (α_ X Y Z).hom =
      (α_ X.toModuleCat Y.toModuleCat Z.toModuleCat).hom := rfl


theorem forget₂_map_associator_inv (X Y Z : QuadraticModuleCat.{u} R) :
    (forget₂ (QuadraticModuleCat R) (ModuleCat R)).map (α_ X Y Z).inv =
      (α_ X.toModuleCat Y.toModuleCat Z.toModuleCat).inv := rfl


noncomputable instance instMonoidalCategory : MonoidalCategory (QuadraticModuleCat.{u} R) :=
  Monoidal.induced
    (forget₂ (QuadraticModuleCat R) (ModuleCat R))
    { μIso := fun _ _ => Iso.refl _
      εIso := Iso.refl _
      leftUnitor_eq := fun X => by
        simp only [forget₂_obj, forget₂_map, Iso.refl_symm, Iso.trans_assoc, Iso.trans_hom,
          Iso.refl_hom, MonoidalCategory.tensorIso_hom, MonoidalCategory.tensorHom_id]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom …
        -/
        dsimp only [toModuleCat_tensor, ModuleCat.of_coe]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom …
        -/
        erw [MonoidalCategory.id_whiskerRight]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom …
        -/
        simp
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom …
        -/
        rfl
        /-
          🎉 no goals
        -/
      rightUnitor_eq := fun X => by
        simp only [forget₂_obj, forget₂_map, Iso.refl_symm, Iso.trans_assoc, Iso.trans_hom,
          Iso.refl_hom, MonoidalCategory.tensorIso_hom, MonoidalCategory.id_tensorHom]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X Y Z : QuadraticModuleCat R
          ⊢ Eq ((CategoryTheory.forget₂ (QuadraticModuleCat R) (ModuleCat R)).map (Categ …
        -/
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).ho …
        -/
        dsimp only [toModuleCat_tensor, ModuleCat.of_coe]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).ho …
        -/
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X Y Z : QuadraticModuleCat R
          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.associator X.toModuleCat Y.toModul …
        -/
        erw [MonoidalCategory.whiskerLeft_id]
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).ho …
        -/
        simp
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          X : QuadraticModuleCat R
          ⊢ Eq (ModuleCat.ofHom (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).ho …
        -/
        rfl
        /-
          🎉 no goals
        -/
      associator_eq := fun X Y Z => by
        dsimp only [forget₂_obj, forget₂_map_associator_hom]
        simp only [eqToIso_refl, Iso.refl_trans, Iso.refl_symm, Iso.trans_hom,
          MonoidalCategory.tensorIso_hom, Iso.refl_hom, MonoidalCategory.tensor_id]
        dsimp only [toModuleCat_tensor, ModuleCat.of_coe]
        rw [Category.id_comp, Category.id_comp, Category.comp_id, MonoidalCategory.tensor_id,
          Category.id_comp] }


