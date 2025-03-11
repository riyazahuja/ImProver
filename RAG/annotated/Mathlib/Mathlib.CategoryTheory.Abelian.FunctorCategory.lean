/-- The abelian coimage in a functor category can be calculated componentwise. -/
@[simps!]
def coimageObjIso : (Abelian.coimage α).obj X ≅ Abelian.coimage (α.app X) :=
  PreservesCokernel.iso ((evaluation C D).obj X) _ ≪≫
    cokernel.mapIso _ _ (PreservesKernel.iso ((evaluation C D).obj X) _) (Iso.refl _)
      (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.evaluation C D).obj …
        -/
        dsimp
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.kernel.ι α).a …
        -/
        simp only [Category.comp_id, PreservesKernel.iso_hom]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq ((CategoryTheory.Limits.kernel.ι α).app X) (CategoryTheory.CategoryStruct …
        -/
        exact (kernelComparison_comp_ι _ ((evaluation C D).obj X)).symm)
        /-
          🎉 no goals
        -/


/-- The abelian image in a functor category can be calculated componentwise. -/
@[simps!]
def imageObjIso : (Abelian.image α).obj X ≅ Abelian.image (α.app X) :=
  PreservesKernel.iso ((evaluation C D).obj X) _ ≪≫
    kernel.mapIso _ _ (Iso.refl _) (PreservesCokernel.iso ((evaluation C D).obj X) _)
      (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.evaluation C D).obj …
        -/
        apply (cancel_mono (PreservesCokernel.iso ((evaluation C D).obj X) α).inv).1
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Category.assoc, Iso.hom_inv_id]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.evaluation C D).obj …
        -/
        dsimp
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.cokernel.π α) …
        -/
        simp only [PreservesCokernel.iso_inv, Category.id_comp, Category.comp_id]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          D : Type w
          inst✝¹ : CategoryTheory.Category.{z, w} D
          inst✝ : CategoryTheory.Abelian D
          F G : CategoryTheory.Functor C D
          α : Quiver.Hom F G
          X : C
          ⊢ Eq ((CategoryTheory.Limits.cokernel.π α).app X) (CategoryTheory.CategoryStru …
        -/
        exact (π_comp_cokernelComparison _ ((evaluation C D).obj X)).symm)
        /-
          🎉 no goals
        -/


theorem coimageImageComparison_app :
    coimageImageComparison (α.app X) =
      (coimageObjIso α X).inv ≫ (coimageImageComparison α).app X ≫ (imageObjIso α X).hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    ⊢ Eq (CategoryTheory.Abelian.coimageImageComparison (α.app X)) (CategoryTheory …
  -/
  ext
  /-
    case h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    case h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [imageObjIso, coimageObjIso, cokernel.map]
  simp only [coimage_image_factorisation, PreservesKernel.iso_hom, Category.assoc,
    kernel.lift_ι, Category.comp_id, PreservesCokernel.iso_inv,
    cokernel.π_desc_assoc, Category.id_comp]
  erw [kernelComparison_comp_ι _ ((evaluation C D).obj X),
    π_comp_cokernelComparison_assoc _ ((evaluation C D).obj X)]
  /-
    case h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    ⊢ Eq (α.app X) (CategoryTheory.CategoryStruct.comp (((CategoryTheory.evaluatio …
  -/
  conv_lhs => rw [← coimage_image_factorisation α]
  /-
    case h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.coimage.π α) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coimageImageComparison_app' :
    (coimageImageComparison α).app X =
      (coimageObjIso α X).hom ≫ coimageImageComparison (α.app X) ≫ (imageObjIso α X).inv := by
  simp only [coimageImageComparison_app, Iso.hom_inv_id_assoc, Iso.hom_inv_id, Category.assoc,
    Category.comp_id]


instance functor_category_isIso_coimageImageComparison :
    IsIso (Abelian.coimageImageComparison α) := by
  have : ∀ X : C, IsIso ((Abelian.coimageImageComparison α).app X) := by
    intros
    rw [coimageImageComparison_app']
    infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{z, w} D
    inst✝ : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X : C
    this : ∀ (X : C), CategoryTheory.IsIso ((CategoryTheory.Abelian.coimageImageCo …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison α)
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


noncomputable instance functorCategoryAbelian : Abelian (C ⥤ D) :=
  let _ : HasKernels (C ⥤ D) := inferInstance
  let _ : HasCokernels (C ⥤ D) := inferInstance
  Abelian.ofCoimageImageComparisonIsIso


