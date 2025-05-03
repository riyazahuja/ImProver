lemma hasMapBifunctor_flip_iff :
    HasMapBifunctor K₂ K₁ F.flip c ↔ HasMapBifunctor K₁ K₂ F c :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).flip_hasTotal_iff c


instance : HasMapBifunctor K₂ K₁ F.flip c := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹² : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹¹ : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁷ : CategoryTheory.Preadditive D
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : DecidableEq J
    inst✝ : K₁.HasMapBifunctor K₂ F c
    ⊢ K₂.HasMapBifunctor K₁ F.flip c
  -/
  rw [hasMapBifunctor_flip_iff]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹² : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹¹ : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁷ : CategoryTheory.Preadditive D
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : DecidableEq J
    inst✝ : K₁.HasMapBifunctor K₂ F c
    ⊢ K₁.HasMapBifunctor K₂ F c
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `mapBifunctor K₂ K₁ F.flip c ≅ mapBifunctor K₁ K₂ F c`. -/
noncomputable def mapBifunctorFlipIso :
    mapBifunctor K₂ K₁ F.flip c ≅ mapBifunctor K₁ K₂ F c :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).totalFlipIso c


lemma mapBifunctorFlipIso_flip
    [TotalComplexShapeSymmetry c₂ c₁ c] [TotalComplexShapeSymmetrySymmetry c₁ c₂ c] :
    mapBifunctorFlipIso K₂ K₁ F.flip c = (mapBifunctorFlipIso K₁ K₂ F c).symm :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).flip_totalFlipIso c


