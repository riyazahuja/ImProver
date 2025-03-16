/-- If `K₁` and `K₂` are two homological complexes, this is the property that
for all `j`, the coproduct of `K₁ i₁ ⊗ K₂ i₂` for `i₁ + i₂ = j` exists. -/
abbrev HasTensor (K₁ K₂ : HomologicalComplex C c) := HasMapBifunctor K₁ K₂ (curriedTensor C) c


/-- The tensor product of two homological complexes. -/
noncomputable abbrev tensorObj (K₁ K₂ : HomologicalComplex C c) [HasTensor K₁ K₂] :
    HomologicalComplex C c :=
  mapBifunctor K₁ K₂ (curriedTensor C) c


/-- The inclusion `K₁.X i₁ ⊗ K₂.X i₂ ⟶ (tensorObj K₁ K₂).X j` of a summand in
the tensor product of the homological complexes. -/
noncomputable abbrev ιTensorObj (K₁ K₂ : HomologicalComplex C c) [HasTensor K₁ K₂]
    (i₁ i₂ j : I) (h : i₁ + i₂ = j) :
    K₁.X i₁ ⊗ K₂.X i₂ ⟶ (tensorObj K₁ K₂).X j :=
  ιMapBifunctor K₁ K₂ (curriedTensor C) c i₁ i₂ j h


/-- The tensor product of two morphisms of homological complexes. -/
noncomputable abbrev tensorHom {K₁ K₂ L₁ L₂ : HomologicalComplex C c}
    (f : K₁ ⟶ L₁) (g : K₂ ⟶ L₂) [HasTensor K₁ K₂] [HasTensor L₁ L₂] :
    tensorObj K₁ K₂ ⟶ tensorObj L₁ L₂ :=
  mapBifunctorMap f g _ _


/-- Given three homological complexes `K₁`, `K₂`, and `K₃`, this asserts that for
all `j`, the functor `- ⊗ K₃.X i₃` commutes with the coproduct of
the `K₁.X i₁ ⊗ K₂.X i₂` such that `i₁ + i₂ = j`. -/
abbrev HasGoodTensor₁₂ (K₁ K₂ K₃ : HomologicalComplex C c) :=
  HasGoodTrifunctor₁₂Obj (curriedTensor C) (curriedTensor C) K₁ K₂ K₃ c c


/-- Given three homological complexes `K₁`, `K₂`, and `K₃`, this asserts that for
all `j`, the functor `K₁.X i₁` commutes with the coproduct of
the `K₂.X i₂ ⊗ K₃.X i₃` such that `i₂ + i₃ = j`. -/
abbrev HasGoodTensor₂₃ (K₁ K₂ K₃ : HomologicalComplex C c) :=
  HasGoodTrifunctor₂₃Obj (curriedTensor C) (curriedTensor C) K₁ K₂ K₃ c c c


/-- The associator isomorphism for the tensor product of homological complexes. -/
noncomputable abbrev associator (K₁ K₂ K₃ : HomologicalComplex C c)
    [HasTensor K₁ K₂] [HasTensor K₂ K₃]
    [HasTensor (tensorObj K₁ K₂) K₃] [HasTensor K₁ (tensorObj K₂ K₃)]
    [HasGoodTensor₁₂ K₁ K₂ K₃] [HasGoodTensor₂₃ K₁ K₂ K₃] :
    tensorObj (tensorObj K₁ K₂) K₃ ≅ tensorObj K₁ (tensorObj K₂ K₃) :=
  mapBifunctorAssociator (curriedAssociatorNatIso C) K₁ K₂ K₃ c c c


variable (C c) in
/-- The unit of the tensor product of homological complexes. -/
noncomputable abbrev tensorUnit : HomologicalComplex C c := (single C c 0).obj (𝟙_ C)


variable (C c) in
/-- As a graded object, the single complex `(single C c 0).obj (𝟙_ C)` identifies
to the unit `(GradedObject.single₀ I).obj (𝟙_ C)` of the tensor product of graded objects. -/
noncomputable def tensorUnitIso :
    (GradedObject.single₀ I).obj (𝟙_ C) ≅ (tensorUnit C c).X :=
  GradedObject.isoMk _ _ (fun i ↦
    if hi : i = 0 then
      (GradedObject.singleObjApplyIsoOfEq (0 : I) (𝟙_ C) i hi).trans
        (singleObjXIsoOfEq c 0 (𝟙_ C) i hi).symm
    else
      { hom := 0
        inv := 0
        hom_inv_id := (GradedObject.isInitialSingleObjApply 0 (𝟙_ C) i hi).hom_ext _ _
        inv_hom_id := (isZero_single_obj_X c 0 (𝟙_ C) i hi).eq_of_src _ _ })


instance (K₁ K₂ : HomologicalComplex C c) [GradedObject.HasTensor K₁.X K₂.X] :
    HasTensor K₁ K₂ := by
  /-
    C : Type u_1
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.MonoidalCategory C
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝³ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝² : AddMonoid I
    c : ComplexShape I
    inst✝¹ : c.TensorSigns
    K₁ K₂ : HomologicalComplex C c
    inst✝ : CategoryTheory.GradedObject.HasTensor K₁.X K₂.X
    ⊢ K₁.HasTensor K₂
  -/
  assumption
  /-
    🎉 no goals
  -/


instance (K₁ K₂ K₃ : HomologicalComplex C c)
    [GradedObject.HasGoodTensor₁₂Tensor K₁.X K₂.X K₃.X] :
    HasGoodTensor₁₂ K₁ K₂ K₃ :=
  inferInstanceAs (GradedObject.HasGoodTensor₁₂Tensor K₁.X K₂.X K₃.X)


instance (K₁ K₂ K₃ : HomologicalComplex C c)
    [GradedObject.HasGoodTensorTensor₂₃ K₁.X K₂.X K₃.X] :
    HasGoodTensor₂₃ K₁ K₂ K₃ :=
  inferInstanceAs (GradedObject.HasGoodTensorTensor₂₃ K₁.X K₂.X K₃.X)


instance : GradedObject.HasTensor (tensorUnit C c).X K.X :=
  GradedObject.hasTensor_of_iso (tensorUnitIso C c) (Iso.refl _)


instance : HasTensor (tensorUnit C c) K :=
  inferInstanceAs (GradedObject.HasTensor (tensorUnit C c).X K.X)


@[simp]
lemma unit_tensor_d₁ (i₁ i₂ j : I) :
    mapBifunctor.d₁ (tensorUnit C c) K (curriedTensor C) c i₁ i₂ j = 0 := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i₁ i₂ j : I
    ⊢ Eq (HomologicalComplex.mapBifunctor.d₁ (HomologicalComplex.tensorUnit C c) K …
  -/
  by_cases h₁ : c.Rel i₁ (c.next i₁)
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i₁ i₂ j : I
      h₁ : c.Rel i₁ (c.next i₁)
      ⊢ Eq (HomologicalComplex.mapBifunctor.d₁ (HomologicalComplex.tensorUnit C c) K …
    -/
  · by_cases h₂ : ComplexShape.π c c c (c.next i₁, i₂) = j
    · rw [mapBifunctor.d₁_eq _ _ _ _ h₁ _ _ h₂, single_obj_d, Functor.map_zero,
        zero_app, zero_comp, smul_zero]
      /-
        case neg
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁸ : CategoryTheory.MonoidalCategory C
        inst✝⁷ : CategoryTheory.Preadditive C
        inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
        inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
        I : Type u_2
        inst✝³ : AddMonoid I
        c : ComplexShape I
        inst✝² : c.TensorSigns
        K : HomologicalComplex C c
        inst✝¹ : DecidableEq I
        inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
        i₁ i₂ j : I
        h₁ : c.Rel i₁ (c.next i₁)
        h₂ : Not (Eq (c.π c c { fst := c.next i₁, snd := i₂ }) j)
        ⊢ Eq (HomologicalComplex.mapBifunctor.d₁ (HomologicalComplex.tensorUnit C c) K …
      -/
    · rw [mapBifunctor.d₁_eq_zero' _ _ _ _ h₁ _ _ h₂]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i₁ i₂ j : I
      h₁ : Not (c.Rel i₁ (c.next i₁))
      ⊢ Eq (HomologicalComplex.mapBifunctor.d₁ (HomologicalComplex.tensorUnit C c) K …
    -/
  · rw [mapBifunctor.d₁_eq_zero _ _ _ _ _ _ _ h₁]
    /-
      🎉 no goals
    -/


instance : GradedObject.HasTensor K.X (tensorUnit C c).X :=
  GradedObject.hasTensor_of_iso (Iso.refl _) (tensorUnitIso C c)


instance : HasTensor K (tensorUnit C c) :=
  inferInstanceAs (GradedObject.HasTensor K.X (tensorUnit C c).X)


@[simp]
lemma tensor_unit_d₂ (i₁ i₂ j : I) :
    mapBifunctor.d₂ K (tensorUnit C c) (curriedTensor C) c i₁ i₂ j = 0 := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i₁ i₂ j : I
    ⊢ Eq (HomologicalComplex.mapBifunctor.d₂ K (HomologicalComplex.tensorUnit C c) …
  -/
  by_cases h₁ : c.Rel i₂ (c.next i₂)
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i₁ i₂ j : I
      h₁ : c.Rel i₂ (c.next i₂)
      ⊢ Eq (HomologicalComplex.mapBifunctor.d₂ K (HomologicalComplex.tensorUnit C c) …
    -/
  · by_cases h₂ : ComplexShape.π c c c (i₁, c.next i₂) = j
    · rw [mapBifunctor.d₂_eq _ _ _ _ _ h₁ _ h₂, single_obj_d, Functor.map_zero,
        zero_comp, smul_zero]
      /-
        case neg
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁸ : CategoryTheory.MonoidalCategory C
        inst✝⁷ : CategoryTheory.Preadditive C
        inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
        inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
        I : Type u_2
        inst✝³ : AddMonoid I
        c : ComplexShape I
        inst✝² : c.TensorSigns
        K : HomologicalComplex C c
        inst✝¹ : DecidableEq I
        inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
        i₁ i₂ j : I
        h₁ : c.Rel i₂ (c.next i₂)
        h₂ : Not (Eq (c.π c c { fst := i₁, snd := c.next i₂ }) j)
        ⊢ Eq (HomologicalComplex.mapBifunctor.d₂ K (HomologicalComplex.tensorUnit C c) …
      -/
    · rw [mapBifunctor.d₂_eq_zero' _ _ _ _ _ h₁ _ h₂]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i₁ i₂ j : I
      h₁ : Not (c.Rel i₂ (c.next i₂))
      ⊢ Eq (HomologicalComplex.mapBifunctor.d₂ K (HomologicalComplex.tensorUnit C c) …
    -/
  · rw [mapBifunctor.d₂_eq_zero _ _ _ _ _ _ _ h₁]
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `leftUnitor`. -/
noncomputable def leftUnitor' :
    (tensorObj (tensorUnit C c) K).X ≅ K.X :=
  GradedObject.Monoidal.tensorIso ((tensorUnitIso C c).symm) (Iso.refl _) ≪≫
    GradedObject.Monoidal.leftUnitor K.X


lemma leftUnitor'_inv (i : I) :
    (leftUnitor' K).inv i = (λ_ (K.X i)).inv ≫ ((singleObjXSelf c 0 (𝟙_ C)).inv ▷ (K.X i)) ≫
      ιTensorObj (tensorUnit C c) K 0 i i (zero_add i) := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (K.leftUnitor'.inv i) (CategoryTheory.CategoryStruct.comp (CategoryTheory …
  -/
  dsimp [leftUnitor']
  rw [GradedObject.Monoidal.leftUnitor_inv_apply, assoc, assoc, Iso.cancel_iso_inv_left,
    GradedObject.Monoidal.ι_tensorHom]
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [tensorHom_id, ← comp_whiskerRight_assoc]
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  congr 2
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.GradedObject.Monoidal. …
  -/
  rw [← cancel_epi (GradedObject.Monoidal.tensorUnit₀ (I := I)).hom, Iso.hom_inv_id_assoc]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq ((HomologicalComplex.tensorUnitIso C c).hom 0) (CategoryTheory.CategorySt …
  -/
  dsimp [tensorUnitIso]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (dite (Eq 0 0) (fun hi => (CategoryTheory.GradedObject.singleObjApplyIsoO …
  -/
  rw [dif_pos rfl]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq ((CategoryTheory.GradedObject.singleObjApplyIsoOfEq 0 CategoryTheory.Mono …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma leftUnitor'_inv_comm (i j : I) :
    (leftUnitor' K).inv i ≫ (tensorObj (tensorUnit C c) K).d i j =
      K.d i j ≫ (leftUnitor' K).inv j := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i j : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.leftUnitor'.inv i) (((HomologicalC …
  -/
  by_cases hij : c.Rel i j
  · simp only [leftUnitor'_inv, assoc, mapBifunctor.d_eq,
      Preadditive.comp_add, mapBifunctor.ι_D₁, mapBifunctor.ι_D₂,
      unit_tensor_d₁, comp_zero, zero_add]
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [mapBifunctor.d₂_eq _ _ _ _ _ hij _ (by simp)]
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    dsimp
    simp only [ComplexShape.ε_zero, one_smul, ← whisker_exchange_assoc,
      id_whiskerLeft, assoc, Iso.inv_hom_id_assoc]
    /-
      case neg
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.leftUnitor'.inv i) (((HomologicalC …
    -/
  · simp only [shape _ _ _ hij, comp_zero, zero_comp]
    /-
      🎉 no goals
    -/


/-- The left unitor for the tensor product of homological complexes. -/
noncomputable def leftUnitor :
    tensorObj (tensorUnit C c) K ≅ K :=
  Iso.symm (Hom.isoOfComponents (fun i ↦ (GradedObject.eval i).mapIso (leftUnitor' K).symm)
    (fun _ _ _ ↦ leftUnitor'_inv_comm _ _ _))


/-- Auxiliary definition for `rightUnitor`. -/
noncomputable def rightUnitor' :
    (tensorObj K (tensorUnit C c)).X ≅ K.X :=
  GradedObject.Monoidal.tensorIso (Iso.refl _) ((tensorUnitIso C c).symm)  ≪≫
    GradedObject.Monoidal.rightUnitor K.X


lemma rightUnitor'_inv (i : I) :
    (rightUnitor' K).inv i = (ρ_ (K.X i)).inv ≫ ((K.X i) ◁ (singleObjXSelf c 0 (𝟙_ C)).inv) ≫
      ιTensorObj K (tensorUnit C c) i 0 i (add_zero i) := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (K.rightUnitor'.inv i) (CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  dsimp [rightUnitor']
  rw [GradedObject.Monoidal.rightUnitor_inv_apply, assoc, assoc, Iso.cancel_iso_inv_left,
    GradedObject.Monoidal.ι_tensorHom]
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [id_tensorHom, ← MonoidalCategory.whiskerLeft_comp_assoc]
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  congr 2
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.GradedObject.Monoidal. …
  -/
  rw [← cancel_epi (GradedObject.Monoidal.tensorUnit₀ (I := I)).hom, Iso.hom_inv_id_assoc]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq ((HomologicalComplex.tensorUnitIso C c).hom 0) (CategoryTheory.CategorySt …
  -/
  dsimp [tensorUnitIso]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq (dite (Eq 0 0) (fun hi => (CategoryTheory.GradedObject.singleObjApplyIsoO …
  -/
  rw [dif_pos rfl]
  /-
    case e_a.e_f
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i : I
    ⊢ Eq ((CategoryTheory.GradedObject.singleObjApplyIsoOfEq 0 CategoryTheory.Mono …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma rightUnitor'_inv_comm (i j : I) :
    (rightUnitor' K).inv i ≫ (tensorObj K (tensorUnit C c)).d i j =
      K.d i j ≫ (rightUnitor' K).inv j := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁸ : CategoryTheory.MonoidalCategory C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
    inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
    I : Type u_2
    inst✝³ : AddMonoid I
    c : ComplexShape I
    inst✝² : c.TensorSigns
    K : HomologicalComplex C c
    inst✝¹ : DecidableEq I
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    i j : I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.rightUnitor'.inv i) ((K.tensorObj  …
  -/
  by_cases hij : c.Rel i j
  · simp only [rightUnitor'_inv, assoc, mapBifunctor.d_eq,
      Preadditive.comp_add, mapBifunctor.ι_D₁, mapBifunctor.ι_D₂,
      tensor_unit_d₂, comp_zero, add_zero]
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [mapBifunctor.d₁_eq _ _ _ _ hij _ _ (by simp)]
    /-
      case pos
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    dsimp
    simp only [one_smul, whisker_exchange_assoc,
      MonoidalCategory.whiskerRight_id, assoc, Iso.inv_hom_id_assoc]
    /-
      case neg
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁴ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝³ : AddMonoid I
      c : ComplexShape I
      inst✝² : c.TensorSigns
      K : HomologicalComplex C c
      inst✝¹ : DecidableEq I
      inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
      i j : I
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.rightUnitor'.inv i) ((K.tensorObj  …
    -/
  · simp only [shape _ _ _ hij, comp_zero, zero_comp]
    /-
      🎉 no goals
    -/


/-- The right unitor for the tensor product of homological complexes. -/
noncomputable def rightUnitor :
    tensorObj K (tensorUnit C c) ≅ K :=
  Iso.symm (Hom.isoOfComponents (fun i ↦ (GradedObject.eval i).mapIso (rightUnitor' K).symm)
    (fun _ _ _ ↦ rightUnitor'_inv_comm _ _ _))


noncomputable instance monoidalCategoryStruct :
    MonoidalCategoryStruct (HomologicalComplex C c) where
  tensorObj K₁ K₂ := tensorObj K₁ K₂
  whiskerLeft _ _ _ g := tensorHom (𝟙 _) g
  whiskerRight f _ := tensorHom f (𝟙 _)
  tensorHom f g := tensorHom f g
  tensorUnit := tensorUnit C c
  associator K₁ K₂ K₃ := associator K₁ K₂ K₃
  leftUnitor K := leftUnitor K
  rightUnitor K := rightUnitor K


/-- The structure which allows to construct the monoidal category structure
on `HomologicalComplex C c` from the monoidal category structure on
graded objects. -/
noncomputable def Monoidal.inducingFunctorData :
    Monoidal.InducingFunctorData (forget C c) where
  μIso _ _ := Iso.refl _
  εIso := tensorUnitIso C c
  whiskerLeft_eq K₁ K₂ L₂ g := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ L₂ : HomologicalComplex C c
      g : Quiver.Hom K₂ L₂
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp [forget]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ L₂ : HomologicalComplex C c
      g : Quiver.Hom K₂ L₂
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft K₁ g).f (CategoryTheor …
    -/
    erw [comp_id, id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ L₂ : HomologicalComplex C c
      g : Quiver.Hom K₂ L₂
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft K₁ g).f (CategoryTheor …
    -/
    rfl
    /-
      🎉 no goals
    -/
  whiskerRight_eq {K₁ L₁} f K₂ := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      K₂ : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp [forget]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      K₂ : HomologicalComplex C c
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f K₂).f (CategoryTheo …
    -/
    erw [comp_id, id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      K₂ : HomologicalComplex C c
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f K₂).f (CategoryTheo …
    -/
    rfl
    /-
      🎉 no goals
    -/
  tensorHom_eq {K₁ L₁ K₂ L₂} f g := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ K₂ L₂ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      g : Quiver.Hom K₂ L₂
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp [forget]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ K₂ L₂ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      g : Quiver.Hom K₂ L₂
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f g).f (CategoryTheory.C …
    -/
    erw [comp_id, id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ L₁ K₂ L₂ : HomologicalComplex C c
      f : Quiver.Hom K₁ L₁
      g : Quiver.Hom K₂ L₂
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f g).f (CategoryTheory.M …
    -/
    rfl
    /-
      🎉 no goals
    -/
  associator_eq K₁ K₂ K₃ := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ K₃ : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp [forget]
    simp only [tensorHom_id, whiskerRight_tensor, id_whiskerRight,
      id_comp, Iso.inv_hom_id, comp_id, assoc]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ K₃ : HomologicalComplex C c
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.associator K₁ K₂ K₃).hom.f (Catego …
    -/
    erw [id_whiskerRight, id_comp, id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K₁ K₂ K₃ : HomologicalComplex C c
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.associator K₁ K₂ K₃).hom.f (Catego …
    -/
    rfl
    /-
      🎉 no goals
    -/
  leftUnitor_eq K := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    erw [id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    rfl
    /-
      🎉 no goals
    -/
  rightUnitor_eq K := by
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    rw [assoc]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    erw [id_comp]
    /-
      C : Type u_1
      inst✝¹⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      inst✝¹² : CategoryTheory.Preadditive C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹⁰ : (CategoryTheory.MonoidalCategory.curriedTensor C).Additive
      inst✝⁹ : ∀ (X₁ : C), ((CategoryTheory.MonoidalCategory.curriedTensor C).obj X₁ …
      I : Type u_2
      inst✝⁸ : AddMonoid I
      c : ComplexShape I
      inst✝⁷ : c.TensorSigns
      inst✝⁶ : ∀ (X₁ X₂ : CategoryTheory.GradedObject I C), X₁.HasTensor X₂
      inst✝⁵ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝⁴ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
      inst✝³ : ∀ (X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C), X₁.HasTensor₄ObjEx …
      inst✝² : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensor₁₂Ten …
      inst✝¹ : ∀ (X₁ X₂ X₃ : CategoryTheory.GradedObject I C), X₁.HasGoodTensorTenso …
      inst✝ : DecidableEq I
      K : HomologicalComplex C c
      ⊢ Eq ((HomologicalComplex.forget C c).map (CategoryTheory.MonoidalCategoryStru …
    -/
    rfl
    /-
      🎉 no goals
    -/


noncomputable instance monoidalCategory : MonoidalCategory (HomologicalComplex C c) :=
  Monoidal.induced _ (Monoidal.inducingFunctorData C c)


