instance [K.HasTotal c] : K.flip.HasTotal c := fun j =>
  hasCoproduct_of_equiv_of_iso (K.toGradedObject.mapObjFun (ComplexShape.π c₁ c₂ c) j) _
    (ComplexShape.symmetryEquiv c₁ c₂ c j) (fun _ => Iso.refl _)


lemma flip_hasTotal_iff : K.flip.HasTotal c ↔ K.HasTotal c := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : TotalComplexShape c₂ c₁ c
    inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
    ⊢ Iff (K.flip.HasTotal c) (K.HasTotal c)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      ⊢ K.flip.HasTotal c → K.HasTotal c
    -/
  · intro
    /-
      case mp
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      a✝ : K.flip.HasTotal c
      ⊢ K.HasTotal c
    -/
    change K.flip.flip.HasTotal c
    /-
      case mp
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      a✝ : K.flip.HasTotal c
      ⊢ K.flip.flip.HasTotal c
    -/
    have := TotalComplexShapeSymmetry.symmetry c₁ c₂ c
    /-
      case mp
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      a✝ : K.flip.HasTotal c
      this : TotalComplexShapeSymmetry c₂ c₁ c
      ⊢ K.flip.flip.HasTotal c
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      ⊢ K.HasTotal c → K.flip.HasTotal c
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c
      inst✝¹ : TotalComplexShape c₂ c₁ c
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c
      a✝ : K.HasTotal c
      ⊢ K.flip.HasTotal c
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `totalFlipIso`. -/
noncomputable def totalFlipIsoX (j : J) : (K.flip.total c).X j ≅ (K.total c).X j where
  hom := K.flip.totalDesc (fun i₂ i₁ h => ComplexShape.σ c₁ c₂ c i₁ i₂ • K.ιTotal c i₁ i₂ j (by
    /-
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{?u.2333, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j : J
      i₂ : I₂
      i₁ : I₁
      h : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    -/
    rw [← ComplexShape.π_symm c₁ c₂ c i₁ i₂, h]))
    /-
      🎉 no goals
    -/
  inv := K.totalDesc (fun i₁ i₂ h => ComplexShape.σ c₁ c₂ c i₁ i₂ • K.flip.ιTotal c i₂ i₁ j (by
    /-
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{?u.2333, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j : J
      i₁ : I₁
      i₂ : I₂
      h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
      ⊢ Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
    -/
    rw [ComplexShape.π_symm c₁ c₂ c i₁ i₂, h]))
    /-
      🎉 no goals
    -/
                   /-
                     C : Type u_1
                     I₁ : Type u_2
                     I₂ : Type u_3
                     J : Type u_4
                     inst✝⁶ : CategoryTheory.Category.{?u.2333, u_1} C
                     inst✝⁵ : CategoryTheory.Preadditive C
                     c₁ : ComplexShape I₁
                     c₂ : ComplexShape I₂
                     K : HomologicalComplex₂ C c₁ c₂
                     c : ComplexShape J
                     inst✝⁴ : TotalComplexShape c₁ c₂ c
                     inst✝³ : TotalComplexShape c₂ c₁ c
                     inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
                     inst✝¹ : K.HasTotal c
                     inst✝ : DecidableEq J
                     j : J
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.totalDesc fun i₂ i₁ h => HSMu …
                   -/
  hom_inv_id := by ext; simp
                        /-
                          🎉 no goals
                        -/
                   /-
                     C : Type u_1
                     I₁ : Type u_2
                     I₂ : Type u_3
                     J : Type u_4
                     inst✝⁶ : CategoryTheory.Category.{?u.2333, u_1} C
                     inst✝⁵ : CategoryTheory.Preadditive C
                     c₁ : ComplexShape I₁
                     c₂ : ComplexShape I₂
                     K : HomologicalComplex₂ C c₁ c₂
                     c : ComplexShape J
                     inst✝⁴ : TotalComplexShape c₁ c₂ c
                     inst✝³ : TotalComplexShape c₂ c₁ c
                     inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
                     inst✝¹ : K.HasTotal c
                     inst✝ : DecidableEq J
                     j : J
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalDesc fun i₁ i₂ h => HSMul.hSM …
                   -/
  inv_hom_id := by ext; simp
                        /-
                          🎉 no goals
                        -/


@[reassoc]
lemma totalFlipIsoX_hom_D₁ (j j' : J) :
    (K.totalFlipIsoX c j).hom ≫ K.D₁ c j j' =
      K.flip.D₂ c j j' ≫ (K.totalFlipIsoX c j').hom := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    j j' : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₁ c j j …
  -/
  by_cases h₀ : c.Rel j j'
    /-
      case pos
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₁ c j j …
    -/
  · ext i₂ i₁ h₁
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.ιTotal c i₂ i₁ j h₁) (Categor …
    -/
    dsimp [totalFlipIsoX]
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.ιTotal c i₂ i₁ j h₁) (Categor …
    -/
    rw [ι_totalDesc_assoc, Linear.units_smul_comp, ι_D₁, ι_D₂_assoc]
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₁ c i₁ i₂ j')) (CategoryTheory.Categor …
    -/
    dsimp
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₁ c i₁ i₂ j')) (CategoryTheory.Categor …
    -/
    by_cases h₂ : c₁.Rel i₁ (c₁.next i₁)
    · have h₃ : ComplexShape.π c₂ c₁ c ⟨i₂, c₁.next i₁⟩ = j' := by
        rw [← ComplexShape.next_π₂ c₂ c i₂ h₂, h₁, c.next_eq' h₀]
      have h₄ : ComplexShape.π c₁ c₂ c ⟨c₁.next i₁, i₂⟩ = j' := by
        rw [← h₃, ComplexShape.π_symm c₁ c₂ c]
      rw [K.d₁_eq _ h₂ _ _ h₄, K.flip.d₂_eq _ _ h₂ _ h₃, Linear.units_smul_comp,
        assoc, ι_totalDesc, Linear.comp_units_smul, smul_smul, smul_smul,
        ComplexShape.σ_ε₁ c₂ c h₂ i₂]
      /-
        case pos
        C : Type u_1
        I₁ : Type u_2
        I₂ : Type u_3
        J : Type u_4
        inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c : ComplexShape J
        inst✝⁴ : TotalComplexShape c₁ c₂ c
        inst✝³ : TotalComplexShape c₂ c₁ c
        inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
        inst✝¹ : K.HasTotal c
        inst✝ : DecidableEq J
        j j' : J
        h₀ : c.Rel j j'
        i₂ : I₂
        i₁ : I₁
        h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
        h₂ : c₁.Rel i₁ (c₁.next i₁)
        h₃ : Eq (c₂.π c₁ c { fst := i₂, snd := c₁.next i₁ }) j'
        h₄ : Eq (c₁.π c₂ c { fst := c₁.next i₁, snd := i₂ }) j'
        ⊢ Eq (HSMul.hSMul (HMul.hMul (c₂.ε₂ c₁ c { fst := i₂, snd := i₁ }) (c₁.σ c₂ c  …
      -/
      dsimp only [flip_X_X, flip_X_d]
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        I₁ : Type u_2
        I₂ : Type u_3
        J : Type u_4
        inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c : ComplexShape J
        inst✝⁴ : TotalComplexShape c₁ c₂ c
        inst✝³ : TotalComplexShape c₂ c₁ c
        inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
        inst✝¹ : K.HasTotal c
        inst✝ : DecidableEq J
        j j' : J
        h₀ : c.Rel j j'
        i₂ : I₂
        i₁ : I₁
        h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
        h₂ : Not (c₁.Rel i₁ (c₁.next i₁))
        ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₁ c i₁ i₂ j')) (CategoryTheory.Categor …
      -/
    · rw [K.d₁_eq_zero _ _ _ _ h₂, K.flip.d₂_eq_zero _ _ _ _ h₂, smul_zero, zero_comp]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : Not (c.Rel j j')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₁ c j j …
    -/
  · rw [K.D₁_shape _ _ _ h₀, K.flip.D₂_shape c _ _ h₀, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma totalFlipIsoX_hom_D₂ (j j' : J) :
    (K.totalFlipIsoX c j).hom ≫ K.D₂ c j j' =
      K.flip.D₁ c j j' ≫ (K.totalFlipIsoX c j').hom := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    j j' : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₂ c j j …
  -/
  by_cases h₀ : c.Rel j j'
    /-
      case pos
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₂ c j j …
    -/
  · ext i₂ i₁ h₁
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.ιTotal c i₂ i₁ j h₁) (Categor …
    -/
    dsimp [totalFlipIsoX]
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.ιTotal c i₂ i₁ j h₁) (Categor …
    -/
    rw [ι_totalDesc_assoc, Linear.units_smul_comp, ι_D₂, ι_D₁_assoc]
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₂ c i₁ i₂ j')) (CategoryTheory.Categor …
    -/
    dsimp
    /-
      case pos.h
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : c.Rel j j'
      i₂ : I₂
      i₁ : I₁
      h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
      ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₂ c i₁ i₂ j')) (CategoryTheory.Categor …
    -/
    by_cases h₂ : c₂.Rel i₂ (c₂.next i₂)
    · have h₃ : ComplexShape.π c₂ c₁ c (ComplexShape.next c₂ i₂, i₁) = j' := by
        rw [← ComplexShape.next_π₁ c₁ c h₂ i₁, h₁, c.next_eq' h₀]
      have h₄ : ComplexShape.π c₁ c₂ c (i₁, ComplexShape.next c₂ i₂) = j' := by
        rw [← h₃, ComplexShape.π_symm c₁ c₂ c]
      rw [K.d₂_eq _ _ h₂ _ h₄, K.flip.d₁_eq _ h₂ _ _ h₃, Linear.units_smul_comp,
        assoc, ι_totalDesc, Linear.comp_units_smul, smul_smul, smul_smul,
        ComplexShape.σ_ε₂ c₁ c i₁ h₂]
      /-
        case pos
        C : Type u_1
        I₁ : Type u_2
        I₂ : Type u_3
        J : Type u_4
        inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c : ComplexShape J
        inst✝⁴ : TotalComplexShape c₁ c₂ c
        inst✝³ : TotalComplexShape c₂ c₁ c
        inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
        inst✝¹ : K.HasTotal c
        inst✝ : DecidableEq J
        j j' : J
        h₀ : c.Rel j j'
        i₂ : I₂
        i₁ : I₁
        h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
        h₂ : c₂.Rel i₂ (c₂.next i₂)
        h₃ : Eq (c₂.π c₁ c { fst := c₂.next i₂, snd := i₁ }) j'
        h₄ : Eq (c₁.π c₂ c { fst := i₁, snd := c₂.next i₂ }) j'
        ⊢ Eq (HSMul.hSMul (HMul.hMul (c₂.ε₁ c₁ c { fst := i₂, snd := i₁ }) (c₁.σ c₂ c  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        I₁ : Type u_2
        I₂ : Type u_3
        J : Type u_4
        inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        c₁ : ComplexShape I₁
        c₂ : ComplexShape I₂
        K : HomologicalComplex₂ C c₁ c₂
        c : ComplexShape J
        inst✝⁴ : TotalComplexShape c₁ c₂ c
        inst✝³ : TotalComplexShape c₂ c₁ c
        inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
        inst✝¹ : K.HasTotal c
        inst✝ : DecidableEq J
        j j' : J
        h₀ : c.Rel j j'
        i₂ : I₂
        i₁ : I₁
        h₁ : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
        h₂ : Not (c₂.Rel i₂ (c₂.next i₂))
        ⊢ Eq (HSMul.hSMul (c₁.σ c₂ c i₁ i₂) (K.d₂ c i₁ i₂ j')) (CategoryTheory.Categor …
      -/
    · rw [K.d₂_eq_zero _ _ _ _ h₂, K.flip.d₁_eq_zero _ _ _ _ h₂, smul_zero, zero_comp]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      h₀ : Not (c.Rel j j')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalFlipIsoX c j).hom (K.D₂ c j j …
    -/
  · rw [K.D₂_shape _ _ _ h₀, K.flip.D₁_shape c _ _ h₀, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


/-- The symmetry isomorphism `K.flip.total c ≅ K.total c` of the total complex of a
bicomplex when we have `[TotalComplexShapeSymmetry c₁ c₂ c]`. -/
noncomputable def totalFlipIso : K.flip.total c ≅ K.total c :=
  HomologicalComplex.Hom.isoOfComponents (K.totalFlipIsoX c) (fun j j' _ => by
    simp only [total_d, Preadditive.comp_add, totalFlipIsoX_hom_D₁,
      totalFlipIsoX_hom_D₂, Preadditive.add_comp]
    /-
      C : Type u_1
      I₁ : Type u_2
      I₂ : Type u_3
      J : Type u_4
      inst✝⁶ : CategoryTheory.Category.{?u.32263, u_1} C
      inst✝⁵ : CategoryTheory.Preadditive C
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      c : ComplexShape J
      inst✝⁴ : TotalComplexShape c₁ c₂ c
      inst✝³ : TotalComplexShape c₂ c₁ c
      inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
      inst✝¹ : K.HasTotal c
      inst✝ : DecidableEq J
      j j' : J
      x✝ : c.Rel j j'
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (K.flip.D₂ c j j') (K.tota …
    -/
    rw [add_comm])
    /-
      🎉 no goals
    -/


@[reassoc]
lemma totalFlipIso_hom_f_D₁ (j j' : J) :
    (K.totalFlipIso c).hom.f j ≫ K.D₁ c j j' =
      K.flip.D₂ c j j' ≫ (K.totalFlipIso c).hom.f j' := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    j j' : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.totalFlipIso c).hom.f j) (K.D₁ c  …
  -/
  apply totalFlipIsoX_hom_D₁
  /-
    🎉 no goals
  -/


@[reassoc]
lemma totalFlipIso_hom_f_D₂ (j j' : J) :
    (K.totalFlipIso c).hom.f j ≫ K.D₂ c j j' =
      K.flip.D₁ c j j' ≫ (K.totalFlipIso c).hom.f j' := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    j j' : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.totalFlipIso c).hom.f j) (K.D₂ c  …
  -/
  apply totalFlipIsoX_hom_D₂
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιTotal_totalFlipIso_f_hom
    (i₁ : I₁) (i₂ : I₂) (j : J) (h : ComplexShape.π c₂ c₁ c (i₂, i₁) = j) :
    K.flip.ιTotal c i₂ i₁ j h ≫ (K.totalFlipIso c).hom.f j =
      ComplexShape.σ c₁ c₂ c i₁ i₂ • K.ιTotal c i₁ i₂ j
            /-
              C : Type u_1
              I₁ : Type u_2
              I₂ : Type u_3
              J : Type u_4
              inst✝⁶ : CategoryTheory.Category.{?u.39224, u_1} C
              inst✝⁵ : CategoryTheory.Preadditive C
              c₁ : ComplexShape I₁
              c₂ : ComplexShape I₂
              K : HomologicalComplex₂ C c₁ c₂
              c : ComplexShape J
              inst✝⁴ : TotalComplexShape c₁ c₂ c
              inst✝³ : TotalComplexShape c₂ c₁ c
              inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
              inst✝¹ : K.HasTotal c
              inst✝ : DecidableEq J
              i₁ : I₁
              i₂ : I₂
              j : J
              h : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
              ⊢ Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
            -/
        (by rw [← ComplexShape.π_symm c₁ c₂ c i₁ i₂, h]) := by
            /-
              🎉 no goals
            -/
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    i₁ : I₁
    i₂ : I₂
    j : J
    h : Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.ιTotal c i₂ i₁ j h) ((K.total …
  -/
  simp [totalFlipIso, totalFlipIsoX]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιTotal_totalFlipIso_f_inv
    (i₁ : I₁) (i₂ : I₂) (j : J) (h : ComplexShape.π c₁ c₂ c (i₁, i₂) = j) :
    K.ιTotal c i₁ i₂ j h ≫ (K.totalFlipIso c).inv.f j =
      ComplexShape.σ c₁ c₂ c i₁ i₂ • K.flip.ιTotal c i₂ i₁ j
            /-
              C : Type u_1
              I₁ : Type u_2
              I₂ : Type u_3
              J : Type u_4
              inst✝⁶ : CategoryTheory.Category.{?u.42972, u_1} C
              inst✝⁵ : CategoryTheory.Preadditive C
              c₁ : ComplexShape I₁
              c₂ : ComplexShape I₂
              K : HomologicalComplex₂ C c₁ c₂
              c : ComplexShape J
              inst✝⁴ : TotalComplexShape c₁ c₂ c
              inst✝³ : TotalComplexShape c₂ c₁ c
              inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
              inst✝¹ : K.HasTotal c
              inst✝ : DecidableEq J
              i₁ : I₁
              i₂ : I₂
              j : J
              h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
              ⊢ Eq (c₂.π c₁ c { fst := i₂, snd := i₁ }) j
            -/
        (by rw [ComplexShape.π_symm c₁ c₂ c i₁ i₂, h]) := by
            /-
              🎉 no goals
            -/
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c
    inst✝³ : TotalComplexShape c₂ c₁ c
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝¹ : K.HasTotal c
    inst✝ : DecidableEq J
    i₁ : I₁
    i₂ : I₂
    j : J
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal c i₁ i₂ j h) ((K.totalFlipI …
  -/
  simp [totalFlipIso, totalFlipIsoX]
  /-
    🎉 no goals
  -/


instance : K.flip.flip.HasTotal c := (inferInstance : K.HasTotal c)


lemma flip_totalFlipIso : K.flip.totalFlipIso c = (K.totalFlipIso c).symm := by
  /-
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁶ : TotalComplexShape c₁ c₂ c
    inst✝⁵ : TotalComplexShape c₂ c₁ c
    inst✝⁴ : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝³ : K.HasTotal c
    inst✝² : DecidableEq J
    inst✝¹ : TotalComplexShapeSymmetry c₂ c₁ c
    inst✝ : TotalComplexShapeSymmetrySymmetry c₁ c₂ c
    ⊢ Eq (K.flip.totalFlipIso c) (K.totalFlipIso c).symm
  -/
  ext j i₁ i₂ h
  /-
    case w.h.h
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁶ : TotalComplexShape c₁ c₂ c
    inst✝⁵ : TotalComplexShape c₂ c₁ c
    inst✝⁴ : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝³ : K.HasTotal c
    inst✝² : DecidableEq J
    inst✝¹ : TotalComplexShapeSymmetry c₂ c₁ c
    inst✝ : TotalComplexShapeSymmetrySymmetry c₁ c₂ c
    j : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.flip.flip.ιTotal c i₁ i₂ j h) ((K. …
  -/
  rw [Iso.symm_hom, ιTotal_totalFlipIso_f_hom]
  /-
    case w.h.h
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁶ : TotalComplexShape c₁ c₂ c
    inst✝⁵ : TotalComplexShape c₂ c₁ c
    inst✝⁴ : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝³ : K.HasTotal c
    inst✝² : DecidableEq J
    inst✝¹ : TotalComplexShapeSymmetry c₂ c₁ c
    inst✝ : TotalComplexShapeSymmetrySymmetry c₁ c₂ c
    j : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (HSMul.hSMul (c₂.σ c₁ c i₂ i₁) (K.flip.ιTotal c i₂ i₁ j ⋯)) (CategoryTheo …
  -/
  dsimp only [flip_flip]
  /-
    case w.h.h
    C : Type u_1
    I₁ : Type u_2
    I₂ : Type u_3
    J : Type u_4
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    c : ComplexShape J
    inst✝⁶ : TotalComplexShape c₁ c₂ c
    inst✝⁵ : TotalComplexShape c₂ c₁ c
    inst✝⁴ : TotalComplexShapeSymmetry c₁ c₂ c
    inst✝³ : K.HasTotal c
    inst✝² : DecidableEq J
    inst✝¹ : TotalComplexShapeSymmetry c₂ c₁ c
    inst✝ : TotalComplexShapeSymmetrySymmetry c₁ c₂ c
    j : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (HSMul.hSMul (c₂.σ c₁ c i₂ i₁) (K.flip.ιTotal c i₂ i₁ j ⋯)) (CategoryTheo …
  -/
  rw [ιTotal_totalFlipIso_f_inv, ComplexShape.σ_symm]
  /-
    🎉 no goals
  -/


