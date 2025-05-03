/-- A bicomplex has a total bicomplex if for any `i₁₂ : I₁₂`, the coproduct
of the objects `(K.X i₁).X i₂` such that `ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = i₁₂` exists. -/
abbrev HasTotal := K.toGradedObject.HasMap (ComplexShape.π c₁ c₂ c₁₂)


include e in
variable {K L} in
lemma hasTotal_of_iso [K.HasTotal c₁₂] : L.HasTotal c₁₂ :=
  GradedObject.hasMap_of_iso (GradedObject.isoMk K.toGradedObject L.toGradedObject
    (fun ⟨i₁, i₂⟩ =>
      (HomologicalComplex.eval _ _ i₁ ⋙ HomologicalComplex.eval _ _ i₂).mapIso e)) _


/-- The horizontal differential in the total complex on a given summand. -/
noncomputable def d₁ :
    (K.X i₁).X i₂ ⟶ (K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂)) i₁₂ :=
  ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.d i₁ (c₁.next i₁)).f i₂ ≫
    K.toGradedObject.ιMapObjOrZero (ComplexShape.π c₁ c₂ c₁₂) ⟨_, i₂⟩ i₁₂)


/-- The vertical differential in the total complex on a given summand. -/
noncomputable def d₂ :
    (K.X i₁).X i₂ ⟶ (K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂)) i₁₂ :=
  ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.X i₁).d i₂ (c₂.next i₂) ≫
    K.toGradedObject.ιMapObjOrZero (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁, _⟩ i₁₂)


lemma d₁_eq_zero (h : ¬ c₁.Rel i₁ (c₁.next i₁)) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂) 0
  -/
  dsimp [d₁]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
  -/
  rw [K.shape_f _ _ h, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₂_eq_zero (h : ¬ c₂.Rel i₂ (c₂.next i₂)) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂) 0
  -/
  dsimp [d₂]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
  -/
  rw [HomologicalComplex.shape _ _ _ h, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₁_eq' {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (i₁₂ : I₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.d i₁ i₁').f i₂ ≫
      K.toGradedObject.ιMapObjOrZero (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁', i₂⟩ i₁₂) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ i₁' : I₁
    h : c₁.Rel i₁ i₁'
    i₂ : I₂
    i₁₂ : I₁₂
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  obtain rfl := c₁.next_eq' h
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : c₁.Rel i₁ (c₁.next i₁)
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma d₁_eq {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁', i₂⟩ = i₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.d i₁ i₁').f i₂ ≫
      K.toGradedObject.ιMapObj (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁', i₂⟩ i₁₂ h') := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ i₁' : I₁
    h : c₁.Rel i₁ i₁'
    i₂ : I₂
    i₁₂ : I₁₂
    h' : Eq (c₁.π c₂ c₁₂ { fst := i₁', snd := i₂ }) i₁₂
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  rw [d₁_eq' K c₁₂ h i₂ i₁₂, K.toGradedObject.ιMapObjOrZero_eq]
  /-
    🎉 no goals
  -/


lemma d₂_eq' (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (i₁₂ : I₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.X i₁).d i₂ i₂' ≫
    K.toGradedObject.ιMapObjOrZero (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁, i₂'⟩ i₁₂) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ i₂' : I₂
    h : c₂.Rel i₂ i₂'
    i₁₂ : I₁₂
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  obtain rfl := c₂.next_eq' h
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : c₂.Rel i₂ (c₂.next i₂)
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma d₂_eq (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ = i₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.X i₁).d i₂ i₂' ≫
    K.toGradedObject.ιMapObj (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁, i₂'⟩ i₁₂ h') := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ i₂' : I₂
    h : c₂.Rel i₂ i₂'
    i₁₂ : I₁₂
    h' : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂' }) i₁₂
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂) (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) …
  -/
  rw [d₂_eq' K c₁₂ i₁ h i₁₂, K.toGradedObject.ιMapObjOrZero_eq]
  /-
    🎉 no goals
  -/


lemma d₁_eq_zero' {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁', i₂⟩ ≠ i₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ i₁' : I₁
    h : c₁.Rel i₁ i₁'
    i₂ : I₂
    i₁₂ : I₁₂
    h' : Ne (c₁.π c₂ c₁₂ { fst := i₁', snd := i₂ }) i₁₂
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂) 0
  -/
  rw [totalAux.d₁_eq' K c₁₂ h i₂ i₁₂, K.toGradedObject.ιMapObjOrZero_eq_zero, comp_zero, smul_zero]
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ i₁' : I₁
    h : c₁.Rel i₁ i₁'
    i₂ : I₂
    i₁₂ : I₁₂
    h' : Ne (c₁.π c₂ c₁₂ { fst := i₁', snd := i₂ }) i₁₂
    ⊢ Ne (c₁.π c₂ c₁₂ { fst := i₁', snd := i₂ }) i₁₂
  -/
  exact h'
  /-
    🎉 no goals
  -/


lemma d₂_eq_zero' (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ ≠ i₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ i₂' : I₂
    h : c₂.Rel i₂ i₂'
    i₁₂ : I₁₂
    h' : Ne (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂' }) i₁₂
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂) 0
  -/
  rw [totalAux.d₂_eq' K c₁₂ i₁ h i₁₂, K.toGradedObject.ιMapObjOrZero_eq_zero, comp_zero, smul_zero]
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁ : I₁
    i₂ i₂' : I₂
    h : c₂.Rel i₂ i₂'
    i₁₂ : I₁₂
    h' : Ne (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂' }) i₁₂
    ⊢ Ne (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂' }) i₁₂
  -/
  exact h'
  /-
    🎉 no goals
  -/


/-- The horizontal differential in the total complex. -/
noncomputable def D₁ (i₁₂ i₁₂' : I₁₂) :
    K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂) i₁₂ ⟶
      K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂) i₁₂' :=
  GradedObject.descMapObj _ (ComplexShape.π c₁ c₂ c₁₂)
    (fun ⟨i₁, i₂⟩ _ => K.d₁ c₁₂ i₁ i₂ i₁₂')


/-- The vertical differential in the total complex. -/
noncomputable def D₂ (i₁₂ i₁₂' : I₁₂) :
    K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂) i₁₂ ⟶
      K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂) i₁₂' :=
  GradedObject.descMapObj _ (ComplexShape.π c₁ c₂ c₁₂)
    (fun ⟨i₁, i₂⟩ _ => K.d₂ c₁₂ i₁ i₂ i₁₂')


@[reassoc (attr := simp)]
lemma ιMapObj_D₁ (i₁₂ i₁₂' : I₁₂) (i : I₁ × I₂) (h : ComplexShape.π c₁ c₂ c₁₂ i = i₁₂) :
    K.toGradedObject.ιMapObj (ComplexShape.π c₁ c₂ c₁₂) i i₁₂ h ≫ K.D₁ c₁₂ i₁₂ i₁₂' =
      K.d₁ c₁₂ i.1 i.2 i₁₂' := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    i : Prod I₁ I₂
    h : Eq (c₁.π c₂ c₁₂ i) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
  -/
  simp [D₁]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιMapObj_D₂ (i₁₂ i₁₂' : I₁₂) (i : I₁ × I₂) (h : ComplexShape.π c₁ c₂ c₁₂ i = i₁₂) :
    K.toGradedObject.ιMapObj (ComplexShape.π c₁ c₂ c₁₂) i i₁₂ h ≫ K.D₂ c₁₂ i₁₂ i₁₂' =
      K.d₂ c₁₂ i.1 i.2 i₁₂' := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    i : Prod I₁ I₂
    h : Eq (c₁.π c₂ c₁₂ i) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
  -/
  simp [D₂]
  /-
    🎉 no goals
  -/


lemma D₁_shape (i₁₂ i₁₂' : I₁₂) (h₁₂ : ¬ c₁₂.Rel i₁₂ i₁₂') : K.D₁ c₁₂ i₁₂ i₁₂' = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    ⊢ Eq (K.D₁ c₁₂ i₁₂ i₁₂') 0
  -/
  ext ⟨i₁, i₂⟩ h
  /-
    case hfg.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
  -/
  simp only [totalAux.ιMapObj_D₁, comp_zero]
  /-
    case hfg.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂') 0
  -/
  by_cases h₁ : c₁.Rel i₁ (c₁.next i₁)
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂') 0
    -/
  · rw [K.d₁_eq_zero' c₁₂ h₁ i₂ i₁₂']
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      ⊢ Ne (c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }) i₁₂'
    -/
    intro h₂
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      h₂ : Eq (c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }) i₁₂'
      ⊢ False
    -/
    exact h₁₂ (by simpa only [← h, ← h₂] using ComplexShape.rel_π₁ c₂ c₁₂ h₁ i₂)
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₁ : Not (c₁.Rel i₁ (c₁.next i₁))
      ⊢ Eq (K.d₁ c₁₂ i₁ i₂ i₁₂') 0
    -/
  · exact d₁_eq_zero _ _ _ _ _ h₁
    /-
      🎉 no goals
    -/


lemma D₂_shape (i₁₂ i₁₂' : I₁₂) (h₁₂ : ¬ c₁₂.Rel i₁₂ i₁₂') : K.D₂ c₁₂ i₁₂ i₁₂' = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    ⊢ Eq (K.D₂ c₁₂ i₁₂ i₁₂') 0
  -/
  ext ⟨i₁, i₂⟩ h
  /-
    case hfg.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
  -/
  simp only [totalAux.ιMapObj_D₂, comp_zero]
  /-
    case hfg.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂') 0
  -/
  by_cases h₂ : c₂.Rel i₂ (c₂.next i₂)
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₂ : c₂.Rel i₂ (c₂.next i₂)
      ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂') 0
    -/
  · rw [K.d₂_eq_zero' c₁₂ i₁ h₂ i₁₂']
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₂ : c₂.Rel i₂ (c₂.next i₂)
      ⊢ Ne (c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }) i₁₂'
    -/
    intro h₁
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₂ : c₂.Rel i₂ (c₂.next i₂)
      h₁ : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }) i₁₂'
      ⊢ False
    -/
    exact h₁₂ (by simpa only [← h, ← h₁] using ComplexShape.rel_π₂ c₁ c₁₂ i₁ h₂)
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
      h₂ : Not (c₂.Rel i₂ (c₂.next i₂))
      ⊢ Eq (K.d₂ c₁₂ i₁ i₂ i₁₂') 0
    -/
  · exact d₂_eq_zero _ _ _ _ _ h₂
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma D₁_D₁ (i₁₂ i₁₂' i₁₂'' : I₁₂) : K.D₁ c₁₂ i₁₂ i₁₂' ≫ K.D₁ c₁₂ i₁₂' i₁₂'' = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' i₁₂'' : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
  -/
  by_cases h₁ : c₁₂.Rel i₁₂ i₁₂'
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : c₁₂.Rel i₁₂ i₁₂'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
    -/
  · by_cases h₂ : c₁₂.Rel i₁₂' i₁₂''
      /-
        case pos
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
      -/
    · ext ⟨i₁, i₂⟩ h
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
      -/
      simp only [totalAux.ιMapObj_D₁_assoc, comp_zero]
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
      -/
      by_cases h₃ : c₁.Rel i₁ (c₁.next i₁)
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
      · rw [totalAux.d₁_eq K c₁₂ h₃ i₂ i₁₂']; swap
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₁.Rel i₁ (c₁.next i₁)
            ⊢ Eq (c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }) i₁₂'
          -/
        · rw [← ComplexShape.next_π₁ c₂ c₁₂ h₃ i₂, ← c₁₂.next_eq' h₁, h]
          /-
            🎉 no goals
          -/
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i …
        -/
        simp only [Linear.units_smul_comp, assoc, totalAux.ιMapObj_D₁]
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
        -/
        by_cases h₄ : c₁.Rel (c₁.next i₁) (c₁.next (c₁.next i₁))
        · rw [totalAux.d₁_eq K c₁₂ h₄ i₂ i₁₂'', Linear.comp_units_smul,
            d_f_comp_d_f_assoc, zero_comp, smul_zero, smul_zero]
          rw [← ComplexShape.next_π₁ c₂ c₁₂ h₄, ← ComplexShape.next_π₁ c₂ c₁₂ h₃,
            h, c₁₂.next_eq' h₁, c₁₂.next_eq' h₂]
          /-
            case neg
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₁.Rel i₁ (c₁.next i₁)
            h₄ : Not (c₁.Rel (c₁.next i₁) (c₁.next (c₁.next i₁)))
            ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
          -/
        · rw [K.d₁_eq_zero _ _ _ _ h₄, comp_zero, smul_zero]
          /-
            🎉 no goals
          -/
        /-
          case neg
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
      · rw [K.d₁_eq_zero c₁₂ _ _ _ h₃, zero_comp]
        /-
          🎉 no goals
        -/
      /-
        case neg
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : Not (c₁₂.Rel i₁₂' i₁₂'')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
      -/
    · rw [K.D₁_shape c₁₂ _ _ h₂, comp_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : Not (c₁₂.Rel i₁₂ i₁₂')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
    -/
  · rw [K.D₁_shape c₁₂ _ _ h₁, zero_comp]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma D₂_D₂ (i₁₂ i₁₂' i₁₂'' : I₁₂) : K.D₂ c₁₂ i₁₂ i₁₂' ≫ K.D₂ c₁₂ i₁₂' i₁₂'' = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' i₁₂'' : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
  -/
  by_cases h₁ : c₁₂.Rel i₁₂ i₁₂'
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : c₁₂.Rel i₁₂ i₁₂'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
    -/
  · by_cases h₂ : c₁₂.Rel i₁₂' i₁₂''
      /-
        case pos
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
      -/
    · ext ⟨i₁, i₂⟩ h
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
      -/
      simp only [totalAux.ιMapObj_D₂_assoc, comp_zero]
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₂ c₁₂ i₁₂'  …
      -/
      by_cases h₃ : c₂.Rel i₂ (c₂.next i₂)
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₂ c₁₂ i₁₂'  …
        -/
      · rw [totalAux.d₂_eq K c₁₂ i₁ h₃ i₁₂']; swap
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₂.Rel i₂ (c₂.next i₂)
            ⊢ Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }) i₁₂'
          -/
        · rw [← ComplexShape.next_π₂ c₁ c₁₂ i₁ h₃, ← c₁₂.next_eq' h₁, h]
          /-
            🎉 no goals
          -/
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i …
        -/
        simp only [Linear.units_smul_comp, assoc, totalAux.ιMapObj_D₂]
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
        -/
        by_cases h₄ : c₂.Rel (c₂.next i₂) (c₂.next (c₂.next i₂))
        · rw [totalAux.d₂_eq K c₁₂ i₁ h₄ i₁₂'', Linear.comp_units_smul,
            HomologicalComplex.d_comp_d_assoc, zero_comp, smul_zero, smul_zero]
          rw [← ComplexShape.next_π₂ c₁ c₁₂ i₁ h₄, ← ComplexShape.next_π₂ c₁ c₁₂ i₁ h₃,
            h, c₁₂.next_eq' h₁, c₁₂.next_eq' h₂]
          /-
            case neg
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₂.Rel i₂ (c₂.next i₂)
            h₄ : Not (c₂.Rel (c₂.next i₂) (c₂.next (c₂.next i₂)))
            ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
          -/
        · rw [K.d₂_eq_zero c₁₂ _ _ _ h₄, comp_zero, smul_zero]
          /-
            🎉 no goals
          -/
        /-
          case neg
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : Not (c₂.Rel i₂ (c₂.next i₂))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₂ c₁₂ i₁₂'  …
        -/
      · rw [K.d₂_eq_zero c₁₂ _ _ _ h₃, zero_comp]
        /-
          🎉 no goals
        -/
      /-
        case neg
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : Not (c₁₂.Rel i₁₂' i₁₂'')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
      -/
    · rw [K.D₂_shape c₁₂ _ _ h₂, comp_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : Not (c₁₂.Rel i₁₂ i₁₂')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
    -/
  · rw [K.D₂_shape c₁₂ _ _ h₁, zero_comp]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma D₂_D₁ (i₁₂ i₁₂' i₁₂'' : I₁₂) :
    K.D₂ c₁₂ i₁₂ i₁₂' ≫ K.D₁ c₁₂ i₁₂' i₁₂'' = - K.D₁ c₁₂ i₁₂ i₁₂' ≫ K.D₂ c₁₂ i₁₂' i₁₂'' := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' i₁₂'' : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
  -/
  by_cases h₁ : c₁₂.Rel i₁₂ i₁₂'
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : c₁₂.Rel i₁₂ i₁₂'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
    -/
  · by_cases h₂ : c₁₂.Rel i₁₂' i₁₂''
      /-
        case pos
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
      -/
    · ext ⟨i₁, i₂⟩ h
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
      -/
      simp only [totalAux.ιMapObj_D₂_assoc, comp_neg, totalAux.ιMapObj_D₁_assoc]
      /-
        case pos.hfg.mk
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : c₁₂.Rel i₁₂' i₁₂''
        i₁ : I₁
        i₂ : I₂
        h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
      -/
      by_cases h₃ : c₁.Rel i₁ (c₁.next i₁)
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
      · rw [totalAux.d₁_eq K c₁₂ h₃ i₂ i₁₂']; swap
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₁.Rel i₁ (c₁.next i₁)
            ⊢ Eq (c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }) i₁₂'
          -/
        · rw [← ComplexShape.next_π₁ c₂ c₁₂ h₃ i₂, ← c₁₂.next_eq' h₁, h]
          /-
            🎉 no goals
          -/
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
        simp only [Linear.units_smul_comp, assoc, totalAux.ιMapObj_D₂]
        /-
          case pos
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : c₁.Rel i₁ (c₁.next i₁)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
        by_cases h₄ : c₂.Rel i₂ (c₂.next i₂)
        · have h₅ : ComplexShape.π c₁ c₂ c₁₂ (i₁, c₂.next i₂) = i₁₂' := by
            rw [← c₁₂.next_eq' h₁, ← h, ComplexShape.next_π₂ c₁ c₁₂ i₁ h₄]
          have h₆ : ComplexShape.π c₁ c₂ c₁₂ (c₁.next i₁, c₂.next i₂) = i₁₂'' := by
            rw [← c₁₂.next_eq' h₂, ← ComplexShape.next_π₁ c₂ c₁₂ h₃, h₅]
          simp only [totalAux.d₂_eq K c₁₂ _ h₄ _ h₅, totalAux.d₂_eq K c₁₂ _ h₄ _ h₆,
            Linear.units_smul_comp, assoc, totalAux.ιMapObj_D₁, Linear.comp_units_smul,
            totalAux.d₁_eq K c₁₂ h₃ _ _ h₆, HomologicalComplex.Hom.comm_assoc, smul_smul,
            ComplexShape.ε₂_ε₁ c₁₂ h₃ h₄, neg_mul, Units.neg_smul]
          /-
            case neg
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : c₁.Rel i₁ (c₁.next i₁)
            h₄ : Not (c₂.Rel i₂ (c₂.next i₂))
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
          -/
        · simp only [K.d₂_eq_zero c₁₂ _ _ _ h₄, zero_comp, comp_zero, smul_zero, neg_zero]
          /-
            🎉 no goals
          -/
        /-
          case neg
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
      · rw [K.d₁_eq_zero c₁₂ _ _ _ h₃, zero_comp, neg_zero]
        /-
          case neg
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
          inst✝³ : CategoryTheory.Preadditive C
          I₁ : Type u_2
          I₂ : Type u_3
          I₁₂ : Type u_4
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          c₁₂ : ComplexShape I₁₂
          inst✝² : TotalComplexShape c₁ c₂ c₁₂
          inst✝¹ : DecidableEq I₁₂
          inst✝ : K.HasTotal c₁₂
          i₁₂ i₁₂' i₁₂'' : I₁₂
          h₁ : c₁₂.Rel i₁₂ i₁₂'
          h₂ : c₁₂.Rel i₁₂' i₁₂''
          i₁ : I₁
          i₂ : I₂
          h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
          h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
        -/
        by_cases h₄ : c₂.Rel i₂ (c₂.next i₂)
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
            h₄ : c₂.Rel i₂ (c₂.next i₂)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
          -/
        · rw [totalAux.d₂_eq K c₁₂ i₁ h₄ i₁₂']; swap
            /-
              case pos
              C : Type u_1
              inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
              inst✝³ : CategoryTheory.Preadditive C
              I₁ : Type u_2
              I₂ : Type u_3
              I₁₂ : Type u_4
              c₁ : ComplexShape I₁
              c₂ : ComplexShape I₂
              K : HomologicalComplex₂ C c₁ c₂
              c₁₂ : ComplexShape I₁₂
              inst✝² : TotalComplexShape c₁ c₂ c₁₂
              inst✝¹ : DecidableEq I₁₂
              inst✝ : K.HasTotal c₁₂
              i₁₂ i₁₂' i₁₂'' : I₁₂
              h₁ : c₁₂.Rel i₁₂ i₁₂'
              h₂ : c₁₂.Rel i₁₂' i₁₂''
              i₁ : I₁
              i₂ : I₂
              h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
              h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
              h₄ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }) i₁₂'
            -/
          · rw [← ComplexShape.next_π₂ c₁ c₁₂ i₁ h₄, ← c₁₂.next_eq' h₁, h]
            /-
              🎉 no goals
            -/
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
            h₄ : c₂.Rel i₂ (c₂.next i₂)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i …
          -/
          simp only [Linear.units_smul_comp, assoc, totalAux.ιMapObj_D₁]
          /-
            case pos
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
            h₄ : c₂.Rel i₂ (c₂.next i₂)
            ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂ c₁₂ { fst := i₁, snd := i₂ }) (CategoryTheory.Cate …
          -/
          rw [K.d₁_eq_zero c₁₂ _ _ _ h₃, comp_zero, smul_zero]
          /-
            🎉 no goals
          -/
          /-
            case neg
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
            inst✝³ : CategoryTheory.Preadditive C
            I₁ : Type u_2
            I₂ : Type u_3
            I₁₂ : Type u_4
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            K : HomologicalComplex₂ C c₁ c₂
            c₁₂ : ComplexShape I₁₂
            inst✝² : TotalComplexShape c₁ c₂ c₁₂
            inst✝¹ : DecidableEq I₁₂
            inst✝ : K.HasTotal c₁₂
            i₁₂ i₁₂' i₁₂'' : I₁₂
            h₁ : c₁₂.Rel i₁₂ i₁₂'
            h₂ : c₁₂.Rel i₁₂' i₁₂''
            i₁ : I₁
            i₂ : I₂
            h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
            h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
            h₄ : Not (c₂.Rel i₂ (c₂.next i₂))
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂') (K.D₁ c₁₂ i₁₂'  …
          -/
        · rw [K.d₂_eq_zero c₁₂ _ _ _ h₄, zero_comp]
          /-
            🎉 no goals
          -/
      /-
        case neg
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        I₁ : Type u_2
        I₂ : Type u_3
        I₁₂ : Type u_4
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c₁₂ : ComplexShape I₁₂
        inst✝² : TotalComplexShape c₁ c₂ c₁₂
        inst✝¹ : DecidableEq I₁₂
        inst✝ : K.HasTotal c₁₂
        i₁₂ i₁₂' i₁₂'' : I₁₂
        h₁ : c₁₂.Rel i₁₂ i₁₂'
        h₂ : Not (c₁₂.Rel i₁₂' i₁₂'')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
      -/
    · rw [K.D₁_shape c₁₂ _ _ h₂, K.D₂_shape c₁₂ _ _ h₂, comp_zero, comp_zero, neg_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' i₁₂'' : I₁₂
      h₁ : Not (c₁₂.Rel i₁₂ i₁₂')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₂ c₁₂ i₁₂ i₁₂') (K.D₁ c₁₂ i₁₂' i₁ …
    -/
  · rw [K.D₁_shape c₁₂ _ _ h₁, K.D₂_shape c₁₂ _ _ h₁, zero_comp, zero_comp, neg_zero]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma D₁_D₂ (i₁₂ i₁₂' i₁₂'' : I₁₂) :
                                                                                              /-
                                                                                                C : Type u_1
                                                                                                inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
                                                                                                inst✝³ : CategoryTheory.Preadditive C
                                                                                                I₁ : Type u_2
                                                                                                I₂ : Type u_3
                                                                                                I₁₂ : Type u_4
                                                                                                c₁ : ComplexShape I₁
                                                                                                c₂ : ComplexShape I₂
                                                                                                K : HomologicalComplex₂ C c₁ c₂
                                                                                                c₁₂ : ComplexShape I₁₂
                                                                                                inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                                                                                inst✝¹ : DecidableEq I₁₂
                                                                                                inst✝ : K.HasTotal c₁₂
                                                                                                i₁₂ i₁₂' i₁₂'' : I₁₂
                                                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂' i₁ …
                                                                                              -/
    K.D₁ c₁₂ i₁₂ i₁₂' ≫ K.D₂ c₁₂ i₁₂' i₁₂'' = - K.D₂ c₁₂ i₁₂ i₁₂' ≫ K.D₁ c₁₂ i₁₂' i₁₂'' := by simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


/-- The total complex of a bicomplex. -/
@[simps (config := .lemmasOnly) d]
noncomputable def total : HomologicalComplex C c₁₂ where
  X := K.toGradedObject.mapObj (ComplexShape.π c₁ c₂ c₁₂)
  d i₁₂ i₁₂' := K.D₁ c₁₂ i₁₂ i₁₂' + K.D₂ c₁₂ i₁₂ i₁₂'
  shape i₁₂ i₁₂' h₁₂ := by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.61731, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L M : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      e : CategoryTheory.Iso K L
      ψ : Quiver.Hom L M
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      ⊢ Eq ((fun i₁₂ i₁₂' => HAdd.hAdd (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂ i₁₂')) i₁₂  …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.61731, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L M : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      e : CategoryTheory.Iso K L
      ψ : Quiver.Hom L M
      c₁₂ : ComplexShape I₁₂
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : DecidableEq I₁₂
      inst✝ : K.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      h₁₂ : Not (c₁₂.Rel i₁₂ i₁₂')
      ⊢ Eq (HAdd.hAdd (K.D₁ c₁₂ i₁₂ i₁₂') (K.D₂ c₁₂ i₁₂ i₁₂')) 0
    -/
    rw [K.D₁_shape c₁₂ _ _ h₁₂, K.D₂_shape c₁₂ _ _ h₁₂, zero_add]
    /-
      🎉 no goals
    -/


/-- The inclusion of a summand in the total complex. -/
noncomputable def ιTotal (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂)
    (h : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) = i₁₂) :
    (K.X i₁).X i₂ ⟶ (K.total c₁₂).X i₁₂ :=
  K.toGradedObject.ιMapObj (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁, i₂⟩ i₁₂ h


@[reassoc (attr := simp)]
lemma XXIsoOfEq_hom_ιTotal {x₁ y₁ : I₁} (h₁ : x₁ = y₁) {x₂ y₂ : I₂} (h₂ : x₂ = y₂)
    (i₁₂ : I₁₂) (h : ComplexShape.π c₁ c₂ c₁₂ (y₁, y₂) = i₁₂) :
    (K.XXIsoOfEq _ _ _ h₁ h₂).hom ≫ K.ιTotal c₁₂ y₁ y₂ i₁₂ h =
                                 /-
                                   C : Type u_1
                                   inst✝⁴ : CategoryTheory.Category.{?u.66558, u_1} C
                                   inst✝³ : CategoryTheory.Preadditive C
                                   I₁ : Type u_2
                                   I₂ : Type u_3
                                   I₁₂ : Type u_4
                                   c₁ : ComplexShape I₁
                                   c₂ : ComplexShape I₂
                                   K L M : HomologicalComplex₂ C c₁ c₂
                                   φ : Quiver.Hom K L
                                   e : CategoryTheory.Iso K L
                                   ψ : Quiver.Hom L M
                                   c₁₂ : ComplexShape I₁₂
                                   inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                   inst✝¹ : DecidableEq I₁₂
                                   inst✝ : K.HasTotal c₁₂
                                   x₁ y₁ : I₁
                                   h₁ : Eq x₁ y₁
                                   x₂ y₂ : I₂
                                   h₂ : Eq x₂ y₂
                                   i₁₂ : I₁₂
                                   h : Eq (c₁.π c₂ c₁₂ { fst := y₁, snd := y₂ }) i₁₂
                                   ⊢ Eq (c₁.π c₂ c₁₂ { fst := x₁, snd := x₂ }) i₁₂
                                 -/
      K.ιTotal c₁₂ x₁ x₂ i₁₂ (by rw [h₁, h₂, h]) := by
                                 /-
                                   🎉 no goals
                                 -/
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    x₁ y₁ : I₁
    h₁ : Eq x₁ y₁
    x₂ y₂ : I₂
    h₂ : Eq x₂ y₂
    i₁₂ : I₁₂
    h : Eq (c₁.π c₂ c₁₂ { fst := y₁, snd := y₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C c₁ c …
  -/
  subst h₁ h₂
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    x₁ : I₁
    x₂ : I₂
    i₁₂ : I₁₂
    h : Eq (c₁.π c₂ c₁₂ { fst := x₁, snd := x₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C c₁ c …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma XXIsoOfEq_inv_ιTotal {x₁ y₁ : I₁} (h₁ : x₁ = y₁) {x₂ y₂ : I₂} (h₂ : x₂ = y₂)
    (i₁₂ : I₁₂) (h : ComplexShape.π c₁ c₂ c₁₂ (x₁, x₂) = i₁₂) :
    (K.XXIsoOfEq _ _ _ h₁ h₂).inv ≫ K.ιTotal c₁₂ x₁ x₂ i₁₂ h =
                                 /-
                                   C : Type u_1
                                   inst✝⁴ : CategoryTheory.Category.{?u.68447, u_1} C
                                   inst✝³ : CategoryTheory.Preadditive C
                                   I₁ : Type u_2
                                   I₂ : Type u_3
                                   I₁₂ : Type u_4
                                   c₁ : ComplexShape I₁
                                   c₂ : ComplexShape I₂
                                   K L M : HomologicalComplex₂ C c₁ c₂
                                   φ : Quiver.Hom K L
                                   e : CategoryTheory.Iso K L
                                   ψ : Quiver.Hom L M
                                   c₁₂ : ComplexShape I₁₂
                                   inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                   inst✝¹ : DecidableEq I₁₂
                                   inst✝ : K.HasTotal c₁₂
                                   x₁ y₁ : I₁
                                   h₁ : Eq x₁ y₁
                                   x₂ y₂ : I₂
                                   h₂ : Eq x₂ y₂
                                   i₁₂ : I₁₂
                                   h : Eq (c₁.π c₂ c₁₂ { fst := x₁, snd := x₂ }) i₁₂
                                   ⊢ Eq (c₁.π c₂ c₁₂ { fst := y₁, snd := y₂ }) i₁₂
                                 -/
      K.ιTotal c₁₂ y₁ y₂ i₁₂ (by rw [← h, h₁, h₂]) := by
                                 /-
                                   🎉 no goals
                                 -/
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    x₁ y₁ : I₁
    h₁ : Eq x₁ y₁
    x₂ y₂ : I₂
    h₂ : Eq x₂ y₂
    i₁₂ : I₁₂
    h : Eq (c₁.π c₂ c₁₂ { fst := x₁, snd := x₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C c₁ c …
  -/
  subst h₁ h₂
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    x₁ : I₁
    x₂ : I₂
    i₁₂ : I₁₂
    h : Eq (c₁.π c₂ c₁₂ { fst := x₁, snd := x₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C c₁ c …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The inclusion of a summand in the total complex, or zero if the degrees do not match. -/
noncomputable def ιTotalOrZero (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂) :
    (K.X i₁).X i₂ ⟶ (K.total c₁₂).X i₁₂ :=
  K.toGradedObject.ιMapObjOrZero (ComplexShape.π c₁ c₂ c₁₂) ⟨i₁, i₂⟩ i₁₂


lemma ιTotalOrZero_eq (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂)
    (h : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) = i₁₂) :
    K.ιTotalOrZero c₁₂ i₁ i₂ i₁₂ = K.ιTotal c₁₂ i₁ i₂ i₁₂ h := dif_pos h


lemma ιTotalOrZero_eq_zero (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂)
    (h : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) ≠ i₁₂) :
    K.ιTotalOrZero c₁₂ i₁ i₂ i₁₂ = 0 := dif_neg h


@[reassoc (attr := simp)]
lemma ι_D₁ (i₁₂ i₁₂' : I₁₂) (i₁ : I₁) (i₂ : I₂) (h : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = i₁₂) :
    K.ιTotal c₁₂ i₁ i₂ i₁₂ h ≫ K.D₁ c₁₂ i₁₂ i₁₂' =
      K.d₁ c₁₂ i₁ i₂ i₁₂' := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal c₁₂ i₁ i₂ i₁₂ h) (K.D₁ c₁₂  …
  -/
  apply totalAux.ιMapObj_D₁
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_D₂ (i₁₂ i₁₂' : I₁₂) (i₁ : I₁) (i₂ : I₂)
    (h : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = i₁₂) :
    K.ιTotal c₁₂ i₁ i₂ i₁₂ h ≫ K.D₂ c₁₂ i₁₂ i₁₂' =
      K.d₂ c₁₂ i₁ i₂ i₁₂' := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal c₁₂ i₁ i₂ i₁₂ h) (K.D₂ c₁₂  …
  -/
  apply totalAux.ιMapObj_D₂
  /-
    🎉 no goals
  -/


lemma d₁_eq' {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (i₁₂ : I₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.d i₁ i₁').f i₂ ≫
      K.ιTotalOrZero c₁₂ i₁' i₂ i₁₂) :=
  totalAux.d₁_eq' _ _ h _ _


lemma d₁_eq {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁', i₂⟩ = i₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.d i₁ i₁').f i₂ ≫
      K.ιTotal c₁₂ i₁' i₂ i₁₂ h') :=
  totalAux.d₁_eq _ _ h _ _ _


lemma d₂_eq' (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (i₁₂ : I₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.X i₁).d i₂ i₂' ≫
    K.ιTotalOrZero c₁₂ i₁ i₂' i₁₂) :=
  totalAux.d₂_eq' _ _ _ h _


lemma d₂_eq (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (i₁₂ : I₁₂)
    (h' : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ = i₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ = ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ • ((K.X i₁).d i₂ i₂' ≫
    K.ιTotal c₁₂ i₁ i₂' i₁₂ h') :=
  totalAux.d₂_eq _ _ _ h _ _


/-- Given a bicomplex `K`, this is a constructor for morphisms from `(K.total c₁₂).X i₁₂`. -/
noncomputable def totalDesc : (K.total c₁₂).X i₁₂ ⟶ A :=
  K.toGradedObject.descMapObj _ (fun ⟨i₁, i₂⟩ hi => f i₁ i₂ hi)


@[reassoc (attr := simp)]
lemma ι_totalDesc (i₁ : I₁) (i₂ : I₂) (hi : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) = i₁₂) :
    K.ιTotal c₁₂ i₁ i₂ i₁₂ hi ≫ K.totalDesc f = f i₁ i₂ hi := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    A : C
    i₁₂ : I₁₂
    f : (i₁ : I₁) → (i₂ : I₂) → Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂ → Qu …
    i₁ : I₁
    i₂ : I₂
    hi : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal c₁₂ i₁ i₂ i₁₂ hi) (K.totalD …
  -/
  simp [totalDesc, ιTotal]
  /-
    🎉 no goals
  -/


@[ext]
lemma hom_ext {A : C} {i₁₂ : I₁₂} {f g : (K.total c₁₂).X i₁₂ ⟶ A}
    (h : ∀ (i₁ : I₁) (i₂ : I₂) (hi : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) = i₁₂),
      K.ιTotal c₁₂ i₁ i₂ i₁₂ hi ≫ f = K.ιTotal c₁₂ i₁ i₂ i₁₂ hi ≫ g) : f = g := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    A : C
    i₁₂ : I₁₂
    f g : Quiver.Hom ((K.total c₁₂).X i₁₂) A
    h : ∀ (i₁ : I₁) (i₂ : I₂) (hi : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂) …
    ⊢ Eq f g
  -/
  apply GradedObject.mapObj_ext
  /-
    case hfg
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    A : C
    i₁₂ : I₁₂
    f g : Quiver.Hom ((K.total c₁₂).X i₁₂) A
    h : ∀ (i₁ : I₁) (i₂ : I₂) (hi : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂) …
    ⊢ ∀ (i : Prod I₁ I₂) (hij : Eq (c₁.π c₂ c₁₂ i) i₁₂), Eq (CategoryTheory.Catego …
  -/
  rintro ⟨i₁, i₂⟩ hi
  /-
    case hfg.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    A : C
    i₁₂ : I₁₂
    f g : Quiver.Hom ((K.total c₁₂).X i₁₂) A
    h : ∀ (i₁ : I₁) (i₂ : I₂) (hi : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂) …
    i₁ : I₁
    i₂ : I₂
    hi : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.toGradedObject.ιMapObj (c₁.π c₂ c₁ …
  -/
  exact h i₁ i₂ hi
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma d₁_mapMap (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂) :
    K.d₁ c₁₂ i₁ i₂ i₁₂ ≫ GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂ =
    (φ.f i₁).f i₂ ≫ L.d₁ c₁₂ i₁ i₂ i₁₂ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
  -/
  by_cases h : c₁.Rel i₁ (c₁.next i₁)
    /-
      case pos
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁ : I₁
      i₂ : I₂
      i₁₂ : I₁₂
      h : c₁.Rel i₁ (c₁.next i₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
    -/
  · simp [totalAux.d₁_eq' _ c₁₂ h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁ : I₁
      i₂ : I₂
      i₁₂ : I₁₂
      h : Not (c₁.Rel i₁ (c₁.next i₁))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₁ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
    -/
  · simp [d₁_eq_zero _ c₁₂ i₁ i₂ i₁₂ h]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma d₂_mapMap (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂) :
    K.d₂ c₁₂ i₁ i₂ i₁₂ ≫ GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂ =
    (φ.f i₁).f i₂ ≫ L.d₂ c₁₂ i₁ i₂ i₁₂ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
  -/
  by_cases h : c₂.Rel i₂ (c₂.next i₂)
    /-
      case pos
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁ : I₁
      i₂ : I₂
      i₁₂ : I₁₂
      h : c₂.Rel i₂ (c₂.next i₂)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
    -/
  · simp [totalAux.d₂_eq' _ c₁₂ i₁ h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁ : I₁
      i₂ : I₂
      i₁₂ : I₁₂
      h : Not (c₂.Rel i₂ (c₂.next i₂))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d₂ c₁₂ i₁ i₂ i₁₂) (CategoryTheory. …
    -/
  · simp [d₂_eq_zero _ c₁₂ i₁ i₂ i₁₂ h]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma mapMap_D₁ (i₁₂ i₁₂' : I₁₂) :
    GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂ ≫ L.D₁ c₁₂ i₁₂ i₁₂' =
      K.D₁ c₁₂ i₁₂ i₁₂' ≫ GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂' := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapMap ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mapMap_D₂ (i₁₂ i₁₂' : I₁₂) :
    GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂ ≫ L.D₂ c₁₂ i₁₂ i₁₂' =
      K.D₂ c₁₂ i₁₂ i₁₂' ≫ GradedObject.mapMap (toGradedObjectMap φ) _ i₁₂' := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁₂ i₁₂' : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapMap ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The morphism `K.total c₁₂ ⟶ L.total c₁₂` of homological complexes induced
by a morphism of bicomplexes `K ⟶ L`. -/
noncomputable def map : K.total c₁₂ ⟶ L.total c₁₂ where
  f := GradedObject.mapMap (toGradedObjectMap φ) _
  comm' i₁₂ i₁₂' _ := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.117532, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L M : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      e : CategoryTheory.Iso K L
      ψ : Quiver.Hom L M
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      x✝ : c₁₂.Rel i₁₂ i₁₂'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapMap ( …
    -/
    dsimp [total]
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.117532, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      I₁ : Type u_2
      I₂ : Type u_3
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L M : HomologicalComplex₂ C c₁ c₂
      φ : Quiver.Hom K L
      e : CategoryTheory.Iso K L
      ψ : Quiver.Hom L M
      c₁₂ : ComplexShape I₁₂
      inst✝³ : TotalComplexShape c₁ c₂ c₁₂
      inst✝² : DecidableEq I₁₂
      inst✝¹ : K.HasTotal c₁₂
      inst✝ : L.HasTotal c₁₂
      i₁₂ i₁₂' : I₁₂
      x✝ : c₁₂.Rel i₁₂ i₁₂'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapMap ( …
    -/
    rw [comp_add, add_comp, mapAux.mapMap_D₁, mapAux.mapMap_D₂]
    /-
      🎉 no goals
    -/


@[simp]
lemma forget_map :
    (HomologicalComplex.forget C c₁₂).map (map φ c₁₂) =
      GradedObject.mapMap (toGradedObjectMap φ) _ := rfl


variable (K) in
@[simp]
lemma map_id : map (𝟙 K) c₁₂ = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    ⊢ Eq (HomologicalComplex₂.total.map (CategoryTheory.CategoryStruct.id K) c₁₂)  …
  -/
  apply (HomologicalComplex.forget _ _).map_injective
  /-
    case a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : DecidableEq I₁₂
    inst✝ : K.HasTotal c₁₂
    ⊢ Eq ((HomologicalComplex.forget C c₁₂).map (HomologicalComplex₂.total.map (Ca …
  -/
  apply GradedObject.mapMap_id
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma map_comp : map (φ ≫ ψ) c₁₂ = map φ c₁₂ ≫ map ψ c₁₂ := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L M : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    ψ : Quiver.Hom L M
    c₁₂ : ComplexShape I₁₂
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : DecidableEq I₁₂
    inst✝² : K.HasTotal c₁₂
    inst✝¹ : L.HasTotal c₁₂
    inst✝ : M.HasTotal c₁₂
    ⊢ Eq (HomologicalComplex₂.total.map (CategoryTheory.CategoryStruct.comp φ ψ) c …
  -/
  apply (HomologicalComplex.forget _ _).map_injective
  /-
    case a
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L M : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    ψ : Quiver.Hom L M
    c₁₂ : ComplexShape I₁₂
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : DecidableEq I₁₂
    inst✝² : K.HasTotal c₁₂
    inst✝¹ : L.HasTotal c₁₂
    inst✝ : M.HasTotal c₁₂
    ⊢ Eq ((HomologicalComplex.forget C c₁₂).map (HomologicalComplex₂.total.map (Ca …
  -/
  exact GradedObject.mapMap_comp (toGradedObjectMap φ) (toGradedObjectMap ψ) _
  /-
    🎉 no goals
  -/


/-- The isomorphism `K.total c₁₂ ≅ L.total c₁₂` of homological complexes induced
by an isomorphism of bicomplexes `K ≅ L`. -/
@[simps]
noncomputable def mapIso : K.total c₁₂ ≅ L.total c₁₂ where
  hom := map e.hom _
  inv := map e.inv _
                   /-
                     C : Type u_1
                     inst✝⁶ : CategoryTheory.Category.{?u.131943, u_1} C
                     inst✝⁵ : CategoryTheory.Preadditive C
                     I₁ : Type u_2
                     I₂ : Type u_3
                     I₁₂ : Type u_4
                     c₁ : ComplexShape I₁
                     c₂ : ComplexShape I₂
                     K L M : HomologicalComplex₂ C c₁ c₂
                     φ : Quiver.Hom K L
                     e : CategoryTheory.Iso K L
                     ψ : Quiver.Hom L M
                     c₁₂ : ComplexShape I₁₂
                     inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
                     inst✝³ : DecidableEq I₁₂
                     inst✝² : K.HasTotal c₁₂
                     inst✝¹ : L.HasTotal c₁₂
                     inst✝ : M.HasTotal c₁₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.total.map e.hom  …
                   -/
  hom_inv_id := by rw [← map_comp, e.hom_inv_id, map_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u_1
                     inst✝⁶ : CategoryTheory.Category.{?u.131943, u_1} C
                     inst✝⁵ : CategoryTheory.Preadditive C
                     I₁ : Type u_2
                     I₂ : Type u_3
                     I₁₂ : Type u_4
                     c₁ : ComplexShape I₁
                     c₂ : ComplexShape I₂
                     K L M : HomologicalComplex₂ C c₁ c₂
                     φ : Quiver.Hom K L
                     e : CategoryTheory.Iso K L
                     ψ : Quiver.Hom L M
                     c₁₂ : ComplexShape I₁₂
                     inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
                     inst✝³ : DecidableEq I₁₂
                     inst✝² : K.HasTotal c₁₂
                     inst✝¹ : L.HasTotal c₁₂
                     inst✝ : M.HasTotal c₁₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.total.map e.inv  …
                   -/
  inv_hom_id := by rw [← map_comp, e.inv_hom_id, map_id]
                   /-
                     🎉 no goals
                   -/


@[reassoc (attr := simp)]
lemma ιTotal_map (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂) (h : ComplexShape.π c₁ c₂ c₁₂ (i₁, i₂) = i₁₂) :
    K.ιTotal c₁₂ i₁ i₂ i₁₂ h ≫ (total.map φ c₁₂).f i₁₂ =
      (φ.f i₁).f i₂ ≫ L.ιTotal c₁₂ i₁ i₂ i₁₂ h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    h : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal c₁₂ i₁ i₂ i₁₂ h) ((Homologi …
  -/
  simp [total.map, ιTotal]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιTotalOrZero_map (i₁ : I₁) (i₂ : I₂) (i₁₂ : I₁₂) :
    K.ιTotalOrZero c₁₂ i₁ i₂ i₁₂ ≫ (total.map φ c₁₂).f i₁₂ =
      (φ.f i₁).f i₂ ≫ L.ιTotalOrZero c₁₂ i₁ i₂ i₁₂ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    I₁ : Type u_2
    I₂ : Type u_3
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K L : HomologicalComplex₂ C c₁ c₂
    φ : Quiver.Hom K L
    c₁₂ : ComplexShape I₁₂
    inst✝³ : TotalComplexShape c₁ c₂ c₁₂
    inst✝² : DecidableEq I₁₂
    inst✝¹ : K.HasTotal c₁₂
    inst✝ : L.HasTotal c₁₂
    i₁ : I₁
    i₂ : I₂
    i₁₂ : I₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotalOrZero c₁₂ i₁ i₂ i₁₂) ((Homo …
  -/
  simp [total.map, ιTotalOrZero]
  /-
    🎉 no goals
  -/


/-- The functor which sends a bicomplex to its total complex. -/
@[simps]
noncomputable def totalFunctor :
    HomologicalComplex₂ C c₁ c₂ ⥤ HomologicalComplex C c₁₂ where
  obj K := K.total c₁₂
  map φ := total.map φ c₁₂


