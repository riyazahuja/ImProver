/-- Auxiliary definition for `mapBifunctorMapHomotopy₁`. -/
noncomputable def hom₁ (j j' : J) :
    (mapBifunctor K₁ K₂ F c).X j ⟶ (mapBifunctor L₁ L₂ F c).X j' :=
  HomologicalComplex₂.totalDesc _
    (fun i₁ i₂ _ => ComplexShape.ε₁ c₁ c₂ c (c₁.prev i₁, i₂) •
      (F.map (h₁.hom i₁ (c₁.prev i₁))).app (K₂.X i₂) ≫
      (F.obj (L₁.X (c₁.prev i₁))).map (f₂.f i₂) ≫ ιMapBifunctorOrZero L₁ L₂ F c _ _ j')


@[reassoc]
lemma ιMapBifunctor_hom₁ (i₁ i₁' : I₁) (i₂ : I₂) (j j' : J)
    (h : ComplexShape.π c₁ c₂ c (i₁', i₂) = j) (h' : c₁.prev i₁' = i₁) :
    ιMapBifunctor K₁ K₂ F c i₁' i₂ j h ≫ hom₁ h₁ f₂ F c j j' = ComplexShape.ε₁ c₁ c₂ c (i₁, i₂) •
      (F.map (h₁.hom i₁' i₁)).app (K₂.X i₂) ≫ (F.obj (L₁.X i₁)).map (f₂.f i₂) ≫
        ιMapBifunctorOrZero L₁ L₂ F c _ _ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    i₁ i₁' : I₁
    i₂ : I₂
    j j' : J
    h : Eq (c₁.π c₂ c { fst := i₁', snd := i₂ }) j
    h' : Eq (c₁.prev i₁') i₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁' i₂ j h)  …
  -/
  subst h'
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    i₁' : I₁
    i₂ : I₂
    j j' : J
    h : Eq (c₁.π c₂ c { fst := i₁', snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁' i₂ j h)  …
  -/
  simp [hom₁]
  /-
    🎉 no goals
  -/


lemma zero₁ (j j' : J) (h : ¬ c.Rel j' j) :
    hom₁ h₁ f₂ F c j j' = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j j' : J
    h : Not (c.Rel j' j)
    ⊢ Eq (HomologicalComplex.mapBifunctorMapHomotopy.hom₁ h₁ f₂ F c j j') 0
  -/
  ext i₁ i₂ h'
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j j' : J
    h : Not (c.Rel j' j)
    i₁ : I₁
    i₂ : I₂
    h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h')  …
  -/
  dsimp [hom₁]
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j j' : J
    h : Not (c.Rel j' j)
    i₁ : I₁
    i₂ : I₂
    h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h')  …
  -/
  rw [comp_zero, HomologicalComplex₂.ι_totalDesc]
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j j' : J
    h : Not (c.Rel j' j)
    i₁ : I₁
    i₂ : I₂
    h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
  -/
  by_cases h₃ : c₁.Rel (c₁.prev i₁) i₁
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : c₁.Rel (c₁.prev i₁) i₁
      ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
    -/
  · rw [ιMapBifunctorOrZero_eq_zero, comp_zero, comp_zero, smul_zero]
    /-
      case pos.h
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : c₁.Rel (c₁.prev i₁) i₁
      ⊢ Ne (c₁.π c₂ c { fst := c₁.prev i₁, snd := i₂ }) j'
    -/
    intro h₄
    /-
      case pos.h
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : c₁.Rel (c₁.prev i₁) i₁
      h₄ : Eq (c₁.π c₂ c { fst := c₁.prev i₁, snd := i₂ }) j'
      ⊢ False
    -/
    apply h
    /-
      case pos.h
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : c₁.Rel (c₁.prev i₁) i₁
      h₄ : Eq (c₁.π c₂ c { fst := c₁.prev i₁, snd := i₂ }) j'
      ⊢ c.Rel j' j
    -/
    rw [← h', ← h₄]
    /-
      case pos.h
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : c₁.Rel (c₁.prev i₁) i₁
      h₄ : Eq (c₁.π c₂ c { fst := c₁.prev i₁, snd := i₂ }) j'
      ⊢ c.Rel (c₁.π c₂ c { fst := c₁.prev i₁, snd := i₂ }) (c₁.π c₂ c { fst := i₁, s …
    -/
    exact ComplexShape.rel_π₁ c₂ c h₃ i₂
    /-
      🎉 no goals
    -/
    /-
      case neg
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
      ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
    -/
  · dsimp
    /-
      case neg
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j j' : J
      h : Not (c.Rel j' j)
      i₁ : I₁
      i₂ : I₂
      h' : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
      ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
    -/
    rw [h₁.zero _ _ h₃, Functor.map_zero, zero_app, zero_comp, smul_zero]
    /-
      🎉 no goals
    -/


lemma comm₁_aux {i₁ i₁' : I₁} (hi₁ : c₁.Rel i₁ i₁') {i₂ i₂' : I₂} (hi₂ : c₂.Rel i₂ i₂') (j : J)
    (hj : ComplexShape.π c₁ c₂ c (i₁', i₂) = j) :
    ComplexShape.ε₁ c₁ c₂ c (i₁, i₂) • (F.map (h₁.hom i₁' i₁)).app (K₂.X i₂) ≫
      (F.obj (L₁.X i₁)).map (f₂.f i₂) ≫
        (((F.mapBifunctorHomologicalComplex c₁ c₂).obj L₁).obj L₂).d₂ c i₁ i₂ j =
    -(((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).d₂ c i₁' i₂ (c.next j) ≫
      hom₁ h₁ f₂ F c (c.next j) j := by
  have hj' : ComplexShape.π c₁ c₂ c ⟨i₁, i₂'⟩ = j := by
    rw [← hj, ← ComplexShape.next_π₂ c₁ c i₁ hi₂, ComplexShape.next_π₁ c₂ c hi₁ i₂]
  rw [HomologicalComplex₂.d₂_eq _ _ _ hi₂ _ hj', HomologicalComplex₂.d₂_eq _ _ _ hi₂ _
        (by rw [← c.next_eq' (ComplexShape.rel_π₂ c₁ c i₁' hi₂), hj]),
    Linear.comp_units_smul, Linear.comp_units_smul, Linear.units_smul_comp, assoc,
    ιMapBifunctor_hom₁ _ _ _ _ _ _ _ _ _ _ (c₁.prev_eq' hi₁),
    ιMapBifunctorOrZero_eq _ _ _ _ _ _ _ hj',
    Linear.comp_units_smul, smul_smul, smul_smul,
    Functor.mapBifunctorHomologicalComplex_obj_obj_X_d,
    Functor.mapBifunctorHomologicalComplex_obj_obj_X_d,
    NatTrans.naturality_assoc, ComplexShape.ε₁_ε₂ c hi₁ hi₂, neg_mul, Units.neg_smul, neg_inj,
    smul_left_cancel_iff, ← Functor.map_comp_assoc, ← Functor.map_comp_assoc, f₂.comm]


lemma comm₁ (j : J) :
    (mapBifunctorMap f₁ f₂ F c).f j =
    (mapBifunctor K₁ K₂ F c).d j (c.next j) ≫
          mapBifunctorMapHomotopy.hom₁ h₁ f₂ F c (c.next j) j +
        mapBifunctorMapHomotopy.hom₁ h₁ f₂ F c j (c.prev j) ≫
          (mapBifunctor L₁ L₂ F c).d (c.prev j) j +
      (mapBifunctorMap f₁' f₂ F c).f j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j : J
    ⊢ Eq ((HomologicalComplex.mapBifunctorMap f₁ f₂ F c).f j) (HAdd.hAdd (HAdd.hAd …
  -/
  ext i₁ i₂ h
  simp? [HomologicalComplex₂.total_d, h₁.comm i₁, dFrom, fromNext, toPrev, dTo] says
    simp only [ι_mapBifunctorMap, h₁.comm i₁, dNext_eq_dFrom_fromNext, dFrom, fromNext,
      AddMonoidHom.mk'_apply, prevD_eq_toPrev_dTo, toPrev, dTo, Functor.map_add,
      Functor.map_comp, NatTrans.app_add, NatTrans.comp_app,
      Preadditive.add_comp, assoc, HomologicalComplex₂.total_d,
      Functor.mapBifunctorHomologicalComplex_obj_obj_toGradedObject, Preadditive.comp_add,
      HomologicalComplex₂.ι_D₁_assoc, Functor.mapBifunctorHomologicalComplex_obj_obj_X_X,
      HomologicalComplex₂.ι_D₂_assoc, add_left_inj]
  have : ∀ {X Y : D} (a b c d e f : X ⟶ Y), a = c → b = e → f = -d →
      a + b = c + d + (e + f) := by rintro X Y a b _ d _ _ rfl rfl rfl; abel
  /-
    case h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K₁ L₁ : HomologicalComplex C₁ c₁
    f₁ f₁' : Quiver.Hom K₁ L₁
    h₁ : Homotopy f₁ f₁'
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.Additive
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    c : ComplexShape J
    inst✝³ : DecidableEq J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : L₁.HasMapBifunctor L₂ F c
    j : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((F.map (K₁.d i₁ (c₁.next  …
  -/
  apply this
    /-
      case h.a
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (K₁.d i₁ (c₁.next i₁))).app ( …
    -/
  · by_cases h₃ : c₁.Rel i₁ (c₁.next i₁)
    · rw [HomologicalComplex₂.d₁_eq _ _ h₃ _ _ (by rw [← h, ComplexShape.next_π₁ c₂ c h₃]),
        Functor.mapBifunctorHomologicalComplex_obj_obj_d_f, Linear.units_smul_comp, assoc,
        ιMapBifunctor_hom₁ _ _ _ _ i₁ _ _ _ _ _ (c₁.prev_eq' h₃),
        Linear.comp_units_smul, smul_smul, Int.units_mul_self, one_smul,
        ιMapBifunctorOrZero_eq]
    · rw [K₁.shape _ _ h₃, Functor.map_zero, zero_app, zero_comp,
        HomologicalComplex₂.d₁_eq_zero _ _ _ _ _ h₃, zero_comp]
    /-
      case h.a
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (h₁.hom i₁ (c₁.prev i₁))).app …
    -/
  · rw [ιMapBifunctor_hom₁_assoc _ _ _ _ _ _ _ _ _ _ rfl]
    /-
      case h.a
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (h₁.hom i₁ (c₁.prev i₁))).app …
    -/
    by_cases h₃ : c₁.Rel (c₁.prev i₁) i₁
    · rw [ιMapBifunctorOrZero_eq _ _ _ _ _ _ _ (by rw [← ComplexShape.prev_π₁ c₂ c h₃, h]),
        Linear.units_smul_comp, assoc, assoc, HomologicalComplex₂.ι_D₁,
        HomologicalComplex₂.d₁_eq _ _ h₃ _ _ h, Linear.comp_units_smul,
        Linear.comp_units_smul, smul_smul, Int.units_mul_self, one_smul,
        Functor.mapBifunctorHomologicalComplex_obj_obj_d_f, NatTrans.naturality_assoc]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        I₁ : Type u_4
        I₂ : Type u_5
        J : Type u_6
        inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
        inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
        inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
        inst✝⁸ : CategoryTheory.Preadditive C₁
        inst✝⁷ : CategoryTheory.Preadditive C₂
        inst✝⁶ : CategoryTheory.Preadditive D
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K₁ L₁ : HomologicalComplex C₁ c₁
        f₁ f₁' : Quiver.Hom K₁ L₁
        h₁ : Homotopy f₁ f₁'
        K₂ L₂ : HomologicalComplex C₂ c₂
        f₂ : Quiver.Hom K₂ L₂
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝⁵ : F.Additive
        inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        c : ComplexShape J
        inst✝³ : DecidableEq J
        inst✝² : TotalComplexShape c₁ c₂ c
        inst✝¹ : K₁.HasMapBifunctor K₂ F c
        inst✝ : L₁.HasMapBifunctor L₂ F c
        j : J
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
        this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
        h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (h₁.hom i₁ (c₁.prev i₁))).app …
      -/
    · rw [h₁.zero _ _ h₃, Functor.map_zero, zero_app, zero_comp, zero_comp, smul_zero, zero_comp]
      /-
        🎉 no goals
      -/
    /-
      case h.a
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h) ( …
    -/
  · rw [ιMapBifunctor_hom₁_assoc _ _ _ _ _ _ _ _ _ _ rfl]
    /-
      case h.a
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      I₁ : Type u_4
      I₂ : Type u_5
      J : Type u_6
      inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
      inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
      inst✝⁸ : CategoryTheory.Preadditive C₁
      inst✝⁷ : CategoryTheory.Preadditive C₂
      inst✝⁶ : CategoryTheory.Preadditive D
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K₁ L₁ : HomologicalComplex C₁ c₁
      f₁ f₁' : Quiver.Hom K₁ L₁
      h₁ : Homotopy f₁ f₁'
      K₂ L₂ : HomologicalComplex C₂ c₂
      f₂ : Quiver.Hom K₂ L₂
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝⁵ : F.Additive
      inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      c : ComplexShape J
      inst✝³ : DecidableEq J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : K₁.HasMapBifunctor K₂ F c
      inst✝ : L₁.HasMapBifunctor L₂ F c
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁. …
    -/
    by_cases h₃ : c₁.Rel (c₁.prev i₁) i₁
      /-
        case pos
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        I₁ : Type u_4
        I₂ : Type u_5
        J : Type u_6
        inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
        inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
        inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
        inst✝⁸ : CategoryTheory.Preadditive C₁
        inst✝⁷ : CategoryTheory.Preadditive C₂
        inst✝⁶ : CategoryTheory.Preadditive D
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K₁ L₁ : HomologicalComplex C₁ c₁
        f₁ f₁' : Quiver.Hom K₁ L₁
        h₁ : Homotopy f₁ f₁'
        K₂ L₂ : HomologicalComplex C₂ c₂
        f₂ : Quiver.Hom K₂ L₂
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝⁵ : F.Additive
        inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        c : ComplexShape J
        inst✝³ : DecidableEq J
        inst✝² : TotalComplexShape c₁ c₂ c
        inst✝¹ : K₁.HasMapBifunctor K₂ F c
        inst✝ : L₁.HasMapBifunctor L₂ F c
        j : J
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
        this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
        h₃ : c₁.Rel (c₁.prev i₁) i₁
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁. …
      -/
    · dsimp
      rw [Linear.units_smul_comp, assoc, assoc,
        ιMapBifunctorOrZero_eq _ _ _ _ _ _ _ (by rw [← ComplexShape.prev_π₁ c₂ c h₃, h]),
        HomologicalComplex₂.ι_D₂]
      /-
        case pos
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        I₁ : Type u_4
        I₂ : Type u_5
        J : Type u_6
        inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
        inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
        inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
        inst✝⁸ : CategoryTheory.Preadditive C₁
        inst✝⁷ : CategoryTheory.Preadditive C₂
        inst✝⁶ : CategoryTheory.Preadditive D
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K₁ L₁ : HomologicalComplex C₁ c₁
        f₁ f₁' : Quiver.Hom K₁ L₁
        h₁ : Homotopy f₁ f₁'
        K₂ L₂ : HomologicalComplex C₂ c₂
        f₂ : Quiver.Hom K₂ L₂
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝⁵ : F.Additive
        inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        c : ComplexShape J
        inst✝³ : DecidableEq J
        inst✝² : TotalComplexShape c₁ c₂ c
        inst✝¹ : K₁.HasMapBifunctor K₂ F c
        inst✝ : L₁.HasMapBifunctor L₂ F c
        j : J
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
        this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
        h₃ : c₁.Rel (c₁.prev i₁) i₁
        ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
      -/
      by_cases h₄ : c₂.Rel i₂ (c₂.next i₂)
        /-
          case pos
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
          inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
          inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
          inst✝⁸ : CategoryTheory.Preadditive C₁
          inst✝⁷ : CategoryTheory.Preadditive C₂
          inst✝⁶ : CategoryTheory.Preadditive D
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K₁ L₁ : HomologicalComplex C₁ c₁
          f₁ f₁' : Quiver.Hom K₁ L₁
          h₁ : Homotopy f₁ f₁'
          K₂ L₂ : HomologicalComplex C₂ c₂
          f₂ : Quiver.Hom K₂ L₂
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          inst✝⁵ : F.Additive
          inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          c : ComplexShape J
          inst✝³ : DecidableEq J
          inst✝² : TotalComplexShape c₁ c₂ c
          inst✝¹ : K₁.HasMapBifunctor K₂ F c
          inst✝ : L₁.HasMapBifunctor L₂ F c
          j : J
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
          this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
          h₃ : c₁.Rel (c₁.prev i₁) i₁
          h₄ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c { fst := c₁.prev i₁, snd := i₂ }) (CategoryTheor …
        -/
      · exact comm₁_aux h₁ f₂ F c h₃ h₄ j h
        /-
          🎉 no goals
        -/
      · rw [HomologicalComplex₂.d₂_eq_zero _ _ _ _ _ h₄, comp_zero, comp_zero, smul_zero,
          HomologicalComplex₂.d₂_eq_zero _ _ _ _ _ h₄, zero_comp, neg_zero]
    · rw [h₁.zero _ _ h₃, Functor.map_zero, zero_app, zero_comp,
        smul_zero, zero_comp, zero_eq_neg]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        I₁ : Type u_4
        I₂ : Type u_5
        J : Type u_6
        inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
        inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
        inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
        inst✝⁸ : CategoryTheory.Preadditive C₁
        inst✝⁷ : CategoryTheory.Preadditive C₂
        inst✝⁶ : CategoryTheory.Preadditive D
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K₁ L₁ : HomologicalComplex C₁ c₁
        f₁ f₁' : Quiver.Hom K₁ L₁
        h₁ : Homotopy f₁ f₁'
        K₂ L₂ : HomologicalComplex C₂ c₂
        f₂ : Quiver.Hom K₂ L₂
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝⁵ : F.Additive
        inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        c : ComplexShape J
        inst✝³ : DecidableEq J
        inst✝² : TotalComplexShape c₁ c₂ c
        inst✝¹ : K₁.HasMapBifunctor K₂ F c
        inst✝ : L₁.HasMapBifunctor L₂ F c
        j : J
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
        this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
        h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((F.mapBifunctorHomologicalComplex  …
      -/
      by_cases h₄ : c₂.Rel i₂ (c₂.next i₂)
        /-
          case pos
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
          inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
          inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
          inst✝⁸ : CategoryTheory.Preadditive C₁
          inst✝⁷ : CategoryTheory.Preadditive C₂
          inst✝⁶ : CategoryTheory.Preadditive D
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K₁ L₁ : HomologicalComplex C₁ c₁
          f₁ f₁' : Quiver.Hom K₁ L₁
          h₁ : Homotopy f₁ f₁'
          K₂ L₂ : HomologicalComplex C₂ c₂
          f₂ : Quiver.Hom K₂ L₂
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          inst✝⁵ : F.Additive
          inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          c : ComplexShape J
          inst✝³ : DecidableEq J
          inst✝² : TotalComplexShape c₁ c₂ c
          inst✝¹ : K₁.HasMapBifunctor K₂ F c
          inst✝ : L₁.HasMapBifunctor L₂ F c
          j : J
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
          this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
          h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
          h₄ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((F.mapBifunctorHomologicalComplex  …
        -/
      · by_cases h₅ : c.Rel j (c.next j)
        · rw [HomologicalComplex₂.d₂_eq _ _ _ h₄ _ (by rw [← ComplexShape.next_π₂ c₁ c i₁ h₄, h]),
            Linear.units_smul_comp, assoc, Functor.mapBifunctorHomologicalComplex_obj_obj_X_d,
            ιMapBifunctor_hom₁ _ _ _ _ _ _ _ _ _ _ rfl, h₁.zero _ _ h₃,
            Functor.map_zero, zero_app, zero_comp, smul_zero, comp_zero, smul_zero]
          /-
            case neg
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
            inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
            inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
            inst✝⁸ : CategoryTheory.Preadditive C₁
            inst✝⁷ : CategoryTheory.Preadditive C₂
            inst✝⁶ : CategoryTheory.Preadditive D
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K₁ L₁ : HomologicalComplex C₁ c₁
            f₁ f₁' : Quiver.Hom K₁ L₁
            h₁ : Homotopy f₁ f₁'
            K₂ L₂ : HomologicalComplex C₂ c₂
            f₂ : Quiver.Hom K₂ L₂
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            inst✝⁵ : F.Additive
            inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
            c : ComplexShape J
            inst✝³ : DecidableEq J
            inst✝² : TotalComplexShape c₁ c₂ c
            inst✝¹ : K₁.HasMapBifunctor K₂ F c
            inst✝ : L₁.HasMapBifunctor L₂ F c
            j : J
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
            this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
            h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
            h₄ : c₂.Rel i₂ (c₂.next i₂)
            h₅ : Not (c.Rel j (c.next j))
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((F.mapBifunctorHomologicalComplex  …
          -/
        · rw [zero₁ _ _ _ _ _ _ h₅, comp_zero]
          /-
            🎉 no goals
          -/
        /-
          case neg
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          inst✝¹¹ : CategoryTheory.Category.{u_8, u_1} C₁
          inst✝¹⁰ : CategoryTheory.Category.{u_9, u_2} C₂
          inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
          inst✝⁸ : CategoryTheory.Preadditive C₁
          inst✝⁷ : CategoryTheory.Preadditive C₂
          inst✝⁶ : CategoryTheory.Preadditive D
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K₁ L₁ : HomologicalComplex C₁ c₁
          f₁ f₁' : Quiver.Hom K₁ L₁
          h₁ : Homotopy f₁ f₁'
          K₂ L₂ : HomologicalComplex C₂ c₂
          f₂ : Quiver.Hom K₂ L₂
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          inst✝⁵ : F.Additive
          inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          c : ComplexShape J
          inst✝³ : DecidableEq J
          inst✝² : TotalComplexShape c₁ c₂ c
          inst✝¹ : K₁.HasMapBifunctor K₂ F c
          inst✝ : L₁.HasMapBifunctor L₂ F c
          j : J
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
          this : ∀ {X Y : D} (a b c d e f : Quiver.Hom X Y), Eq a c → Eq b e → Eq f (Neg …
          h₃ : Not (c₁.Rel (c₁.prev i₁) i₁)
          h₄ : Not (c₂.Rel i₂ (c₂.next i₂))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((F.mapBifunctorHomologicalComplex  …
        -/
      · rw [HomologicalComplex₂.d₂_eq_zero _ _ _ _ _ h₄, zero_comp]
        /-
          🎉 no goals
        -/


open mapBifunctorMapHomotopy in
/-- The homotopy between `mapBifunctorMap f₁ f₂ F c` and `mapBifunctorMap f₁' f₂ F c` that
is induced by an homotopy between `f₁` and `f₁'`. -/
noncomputable def mapBifunctorMapHomotopy₁ :
    Homotopy (mapBifunctorMap f₁ f₂ F c) (mapBifunctorMap f₁' f₂ F c) where
  hom := hom₁ h₁ f₂ F c
  zero := zero₁ h₁ f₂ F c
  comm := comm₁ h₁ f₂ F c


