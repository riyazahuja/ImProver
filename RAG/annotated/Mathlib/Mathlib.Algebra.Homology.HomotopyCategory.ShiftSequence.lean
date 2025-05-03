/-- The natural isomorphism `(K⟦n⟧).sc' i j k ≅ K.sc' i' j' k'` when `n + i = i'`,
`n + j = j'` and `n + k = k'`. -/
@[simps!]
def shiftShortComplexFunctor' (n i j k i' j' k' : ℤ)
    (hi : n + i = i') (hj : n + j = j') (hk : n + k = k') :
    (CategoryTheory.shiftFunctor (CochainComplex C ℤ) n) ⋙ shortComplexFunctor' C _ i j k ≅
      shortComplexFunctor' C _ i' j' k' :=
  NatIso.ofComponents (fun K => ShortComplex.isoMk
      (n.negOnePow • ((shiftEval C n i i' hi).app K))
      ((shiftEval C n j j' hj).app K) (n.negOnePow • ((shiftEval C n k k' hk).app K))
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.88, u_1} C
            inst✝ : CategoryTheory.Preadditive C
            n i j k i' j' k' : Int
            hi : Eq (HAdd.hAdd n i) i'
            hj : Eq (HAdd.hAdd n j) j'
            hk : Eq (HAdd.hAdd n k) k'
            K : CochainComplex C Int
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul n.negOnePow ((CochainCom …
          -/
                 /-
                   🎉 no goals
                 -/
      (by dsimp; simp) (by dsimp; simp))
                                  /-
                                    🎉 no goals
                                  -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.88, u_1} C
                    inst✝ : CategoryTheory.Preadditive C
                    n i j k i' j' k' : Int
                    hi : Eq (HAdd.hAdd n i) i'
                    hj : Eq (HAdd.hAdd n j) j'
                    hk : Eq (HAdd.hAdd n k) k'
                    X✝ Y✝ : CochainComplex C Int
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.shiftFunctor (Cocha …
                  -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
      (fun f ↦ by ext <;> dsimp <;> simp)
                                    /-
                                      🎉 no goals
                                    -/


/-- The natural isomorphism `(K⟦n⟧).sc i ≅ K.sc i'` when `n + i = i'`. -/
@[simps!]
noncomputable def shiftShortComplexFunctorIso (n i i' : ℤ) (hi : n + i = i') :
    shiftFunctor C n ⋙ shortComplexFunctor C _ i ≅ shortComplexFunctor C _ i' :=
  shiftShortComplexFunctor' C n _ i _ _ i' _
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.42167, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          n i i' : Int
          hi : Eq (HAdd.hAdd n i) i'
          ⊢ Eq (HAdd.hAdd n ((ComplexShape.up Int).prev i)) ((ComplexShape.up Int).prev  …
        -/
                          /-
                            🎉 no goals
                          -/
    (by simp only [prev]; omega) hi (by simp only [next]; omega)
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma shiftShortComplexFunctorIso_zero_add_hom_app (a : ℤ) (K : CochainComplex C ℤ) :
    (shiftShortComplexFunctorIso C 0 a a (zero_add a)).hom.app K =
      (shortComplexFunctor C (ComplexShape.up ℤ) a).map
        ((shiftFunctorZero (CochainComplex C ℤ) ℤ).hom.app K) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    a : Int
    K : CochainComplex C Int
    ⊢ Eq ((CochainComplex.shiftShortComplexFunctorIso C 0 a a ⋯).hom.app K) ((Homo …
  -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  ext <;> dsimp <;> simp [one_smul, shiftFunctorZero_hom_app_f]
                    /-
                      🎉 no goals
                    -/


lemma shiftShortComplexFunctorIso_add'_hom_app
    (n m mn : ℤ) (hmn : m + n = mn) (a a' a'' : ℤ) (ha' : n + a = a') (ha'' : m + a' = a'')
    (K : CochainComplex C ℤ) :
                                                /-
                                                  C : Type u_1
                                                  inst✝¹ : CategoryTheory.Category.{?u.64770, u_1} C
                                                  inst✝ : CategoryTheory.Preadditive C
                                                  n m mn : Int
                                                  hmn : Eq (HAdd.hAdd m n) mn
                                                  a a' a'' : Int
                                                  ha' : Eq (HAdd.hAdd n a) a'
                                                  ha'' : Eq (HAdd.hAdd m a') a''
                                                  K : CochainComplex C Int
                                                  ⊢ Eq (HAdd.hAdd mn a) a''
                                                -/
    (shiftShortComplexFunctorIso C mn a a'' (by rw [← ha'', ← ha', ← add_assoc, hmn])).hom.app K =
                                                /-
                                                  🎉 no goals
                                                -/
      (shortComplexFunctor C (ComplexShape.up ℤ) a).map
        ((CategoryTheory.shiftFunctorAdd' (CochainComplex C ℤ) m n mn hmn).hom.app K) ≫
        (shiftShortComplexFunctorIso C n a a' ha').hom.app (K⟦m⟧) ≫
        (shiftShortComplexFunctorIso C m a' a'' ha'' ).hom.app K := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    n m mn : Int
    hmn : Eq (HAdd.hAdd m n) mn
    a a' a'' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    K : CochainComplex C Int
    ⊢ Eq ((CochainComplex.shiftShortComplexFunctorIso C mn a a'' ⋯).hom.app K) (Ca …
  -/
  ext <;> dsimp <;> simp only [← hmn, Int.negOnePow_add, shiftFunctorAdd'_hom_app_f',
    XIsoOfEq_shift, Linear.comp_units_smul, Linear.units_smul_comp,
    XIsoOfEq_hom_comp_XIsoOfEq_hom, smul_smul]


variable (C) in
/-- The natural isomorphism `(K⟦n⟧).homology a ≅ K.homology a'`when `n + a = a`. -/
noncomputable def shiftIso (n a a' : ℤ) (ha' : n + a = a') :
    (CategoryTheory.shiftFunctor _ n) ⋙ homologyFunctor C (ComplexShape.up ℤ) a ≅
      homologyFunctor C (ComplexShape.up ℤ) a' :=
  isoWhiskerLeft _ (homologyFunctorIso C (ComplexShape.up ℤ) a) ≪≫
    (Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight (shiftShortComplexFunctorIso C n a a' ha')
      (ShortComplex.homologyFunctor C) ≪≫
    (homologyFunctorIso C (ComplexShape.up ℤ) a').symm


lemma shiftIso_hom_app (n a a' : ℤ) (ha' : n + a = a') (K : CochainComplex C ℤ) :
    (shiftIso C n a a' ha').hom.app K =
      ShortComplex.homologyMap ((shiftShortComplexFunctorIso C n a a' ha').hom.app K) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    n a a' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    K : CochainComplex C Int
    ⊢ Eq ((CochainComplex.ShiftSequence.shiftIso C n a a' ha').hom.app K) (Categor …
  -/
  dsimp [shiftIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    n a a' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    K : CochainComplex C Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
  -/
  erw [id_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


lemma shiftIso_inv_app (n a a' : ℤ) (ha' : n + a = a') (K : CochainComplex C ℤ) :
    (shiftIso C n a a' ha').inv.app K =
      ShortComplex.homologyMap ((shiftShortComplexFunctorIso C n a a' ha').inv.app K) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    n a a' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    K : CochainComplex C Int
    ⊢ Eq ((CochainComplex.ShiftSequence.shiftIso C n a a' ha').inv.app K) (Categor …
  -/
  dsimp [shiftIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    n a a' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    K : CochainComplex C Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [id_comp, comp_id, comp_id]
  /-
    🎉 no goals
  -/


noncomputable instance :
    (homologyFunctor C (ComplexShape.up ℤ) 0).ShiftSequence ℤ where
  sequence n := homologyFunctor C (ComplexShape.up ℤ) n
  isoZero := Iso.refl _
  shiftIso n a a' ha' := ShiftSequence.shiftIso C n a a' ha'
  shiftIso_zero a := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.92449, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      a : Int
      ⊢ Eq ((fun n a a' ha' => CochainComplex.ShiftSequence.shiftIso C n a a' ha') 0 …
    -/
    ext K
    /-
      case w.w.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.92449, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      a : Int
      K : HomologicalComplex C (ComplexShape.up Int)
      ⊢ Eq (((fun n a a' ha' => CochainComplex.ShiftSequence.shiftIso C n a a' ha')  …
    -/
    dsimp [homologyMap]
    simp only [ShiftSequence.shiftIso_hom_app, comp_id,
      shiftShortComplexFunctorIso_zero_add_hom_app]
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.92449, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq ((fun n a a' ha' => CochainComplex.ShiftSequence.shiftIso C n a a' ha') ( …
    -/
    ext K
    /-
      case w.w.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.92449, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      K : HomologicalComplex C (ComplexShape.up Int)
      ⊢ Eq (((fun n a a' ha' => CochainComplex.ShiftSequence.shiftIso C n a a' ha')  …
    -/
    dsimp [homologyMap]
    simp only [ShiftSequence.shiftIso_hom_app, id_comp,
      ← ShortComplex.homologyMap_comp, shiftFunctorAdd'_eq_shiftFunctorAdd,
      shiftShortComplexFunctorIso_add'_hom_app n m _ rfl a a' a'' ha' ha'' K]


lemma quasiIsoAt_shift_iff {K L : CochainComplex C ℤ} (φ : K ⟶ L) (n i j : ℤ) (h : n + i = j) :
    QuasiIsoAt (φ⟦n⟧') i ↔ QuasiIsoAt φ j := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    n i j : Int
    h : Eq (HAdd.hAdd n i) j
    ⊢ Iff (QuasiIsoAt ((CategoryTheory.shiftFunctor (HomologicalComplex C (Complex …
  -/
  simp only [quasiIsoAt_iff_isIso_homologyMap]
  exact (NatIso.isIso_map_iff
    ((homologyFunctor C (ComplexShape.up ℤ) 0).shiftIso n i j h) φ)


lemma quasiIso_shift_iff {K L : CochainComplex C ℤ} (φ : K ⟶ L) (n : ℤ) :
    QuasiIso (φ⟦n⟧') ↔ QuasiIso φ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    n : Int
    ⊢ Iff (QuasiIso ((CategoryTheory.shiftFunctor (HomologicalComplex C (ComplexSh …
  -/
  simp only [quasiIso_iff, fun i ↦ quasiIsoAt_shift_iff φ n i _ rfl]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    n : Int
    ⊢ Iff (∀ (i : Int), QuasiIsoAt φ (HAdd.hAdd n i)) (∀ (i : Int), QuasiIsoAt φ i)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : CochainComplex C Int
      φ : Quiver.Hom K L
      n : Int
      ⊢ (∀ (i : Int), QuasiIsoAt φ (HAdd.hAdd n i)) → ∀ (i : Int), QuasiIsoAt φ i
    -/
  · intro h j
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : CochainComplex C Int
      φ : Quiver.Hom K L
      n : Int
      h : ∀ (i : Int), QuasiIsoAt φ (HAdd.hAdd n i)
      j : Int
      ⊢ QuasiIsoAt φ j
    -/
    obtain ⟨i, rfl⟩ : ∃ i, j = n + i := ⟨j - n, by omega⟩
    /-
      case mp.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : CochainComplex C Int
      φ : Quiver.Hom K L
      n : Int
      h : ∀ (i : Int), QuasiIsoAt φ (HAdd.hAdd n i)
      i : Int
      ⊢ QuasiIsoAt φ (HAdd.hAdd n i)
    -/
    exact h i
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : CochainComplex C Int
      φ : Quiver.Hom K L
      n : Int
      ⊢ (∀ (i : Int), QuasiIsoAt φ i) → ∀ (i : Int), QuasiIsoAt φ (HAdd.hAdd n i)
    -/
  · intro h i
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : CochainComplex C Int
      φ : Quiver.Hom K L
      n : Int
      h : ∀ (i : Int), QuasiIsoAt φ i
      i : Int
      ⊢ QuasiIsoAt φ (HAdd.hAdd n i)
    -/
    exact h (n + i)
    /-
      🎉 no goals
    -/


instance {K L : CochainComplex C ℤ} (φ : K ⟶ L) (n : ℤ) [QuasiIso φ] :
    QuasiIso (φ⟦n⟧') := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    n : Int
    inst✝ : QuasiIso φ
    ⊢ QuasiIso ((CategoryTheory.shiftFunctor (HomologicalComplex C (ComplexShape.u …
  -/
  rw [quasiIso_shift_iff]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    n : Int
    inst✝ : QuasiIso φ
    ⊢ QuasiIso φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : (HomologicalComplex.quasiIso C (ComplexShape.up ℤ)).IsCompatibleWithShift ℤ where
                    /-
                      C : Type u_1
                      inst✝² : CategoryTheory.Category.{u_2, u_1} C
                      inst✝¹ : CategoryTheory.Preadditive C
                      inst✝ : CategoryTheory.CategoryWithHomology C
                      n : Int
                      ⊢ Eq ((HomologicalComplex.quasiIso C (ComplexShape.up Int)).inverseImage (Cate …
                    -/
  condition n := by ext; apply quasiIso_shift_iff
                         /-
                           🎉 no goals
                         -/


variable (C) in
lemma homologyFunctor_shift (n : ℤ) :
    (homologyFunctor C (ComplexShape.up ℤ) 0).shift n =
      homologyFunctor C (ComplexShape.up ℤ) n := rfl


@[reassoc]
lemma liftCycles_shift_homologyπ
    (K : CochainComplex C ℤ) {A : C} {n i : ℤ} (f : A ⟶ (K⟦n⟧).X i) (j : ℤ)
    (hj : (up ℤ).next i = j) (hf : f ≫ (K⟦n⟧).d i j = 0) (i' : ℤ) (hi' : n + i = i') (j' : ℤ)
    (hj' : (up ℤ).next i' = j') :
    (K⟦n⟧).liftCycles f j hj hf ≫ (K⟦n⟧).homologyπ i =
                                                          /-
                                                            C : Type u_1
                                                            inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
                                                            inst✝¹ : CategoryTheory.Preadditive C
                                                            inst✝ : CategoryTheory.CategoryWithHomology C
                                                            K : CochainComplex C Int
                                                            A : C
                                                            n i : Int
                                                            f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
                                                            j : Int
                                                            hj : Eq ((ComplexShape.up Int).next i) j
                                                            hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
                                                            i' : Int
                                                            hi' : Eq (HAdd.hAdd n i) i'
                                                            j' : Int
                                                            hj' : Eq ((ComplexShape.up Int).next i') j'
                                                            ⊢ Eq i' (HAdd.hAdd i n)
                                                          -/
      K.liftCycles (f ≫ (K.shiftFunctorObjXIso n i i' (by omega)).hom) j' hj' (by
                                                          /-
                                                            🎉 no goals
                                                          -/
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hj : Eq ((ComplexShape.up Int).next i) j
          hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
          i' : Int
          hi' : Eq (HAdd.hAdd n i) i'
          j' : Int
          hj' : Eq ((ComplexShape.up Int).next i') j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        simp only [next] at hj hj'
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
          i' : Int
          hi' : Eq (HAdd.hAdd n i) i'
          j' : Int
          hj : Eq (HAdd.hAdd i 1) j
          hj' : Eq (HAdd.hAdd i' 1) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        obtain rfl : i' = i + n := by omega
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
          j' : Int
          hj : Eq (HAdd.hAdd i 1) j
          hi' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
          hj' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        obtain rfl : j' = j + n := by omega
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
          hj : Eq (HAdd.hAdd i 1) j
          hi' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
          hj' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) (HAdd.hAdd j n)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        dsimp at hf ⊢
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hf : Eq (CategoryTheory.CategoryStruct.comp f (HSMul.hSMul n.negOnePow (K.d (H …
          hj : Eq (HAdd.hAdd i 1) j
          hi' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
          hj' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) (HAdd.hAdd j n)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        simp only [Linear.comp_units_smul] at hf
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hj : Eq (HAdd.hAdd i 1) j
          hi' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
          hj' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) (HAdd.hAdd j n)
          hf : Eq (HSMul.hSMul n.negOnePow (CategoryTheory.CategoryStruct.comp f (K.d (H …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        apply (one_smul (M := ℤˣ) _).symm.trans _
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.109057, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.CategoryWithHomology C
          K : CochainComplex C Int
          A : C
          n i : Int
          f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
          j : Int
          hj : Eq (HAdd.hAdd i 1) j
          hi' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
          hj' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) (HAdd.hAdd j n)
          hf : Eq (HSMul.hSMul n.negOnePow (CategoryTheory.CategoryStruct.comp f (K.d (H …
          ⊢ Eq (HSMul.hSMul 1 (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
        -/
        rw [← Int.units_mul_self n.negOnePow, mul_smul, comp_id, hf, smul_zero]) ≫
        /-
          🎉 no goals
        -/
        K.homologyπ i' ≫
          ((HomologicalComplex.homologyFunctor C (up ℤ) 0).shiftIso n i i' hi').inv.app K := by
  simp only [liftCycles, homologyπ,
    shiftFunctorObjXIso, Functor.shiftIso, Functor.ShiftSequence.shiftIso,
    ShiftSequence.shiftIso_inv_app, ShortComplex.homologyπ_naturality,
    ShortComplex.liftCycles_comp_cyclesMap_assoc, shiftShortComplexFunctorIso_inv_app_τ₂,
    assoc, Iso.hom_inv_id, comp_id]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K : CochainComplex C Int
    A : C
    n i : Int
    f : Quiver.Hom A (((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj  …
    j : Int
    hj : Eq ((ComplexShape.up Int).next i) j
    hf : Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.shiftFunctor ( …
    i' : Int
    hi' : Eq (HAdd.hAdd n i) i'
    j' : Int
    hj' : Eq ((ComplexShape.up Int).next i') j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.sc ((CategoryThe …
  -/
  rfl
  /-
    🎉 no goals
  -/


noncomputable instance :
    (homologyFunctor C (ComplexShape.up ℤ) 0).ShiftSequence ℤ :=
  Functor.ShiftSequence.induced (homologyFunctorFactors C (ComplexShape.up ℤ) 0) ℤ
    (homologyFunctor C (ComplexShape.up ℤ))
    (homologyFunctorFactors C (ComplexShape.up ℤ))


lemma homologyShiftIso_hom_app (n a a' : ℤ) (ha' : n + a = a') (K : CochainComplex C ℤ) :
    ((homologyFunctor C (ComplexShape.up ℤ) 0).shiftIso n a a' ha').hom.app
      ((quotient _ _).obj K) =
    (homologyFunctor _ _ a).map (((quotient _ _).commShiftIso n).inv.app K) ≫
      (homologyFunctorFactors _ _ a).hom.app (K⟦n⟧) ≫
      ((HomologicalComplex.homologyFunctor _ _ 0).shiftIso n a a' ha').hom.app K ≫
      (homologyFunctorFactors _ _ a').inv.app K := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    n a a' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    K : CochainComplex C Int
    ⊢ Eq (((HomotopyCategory.homologyFunctor C (ComplexShape.up Int) 0).shiftIso n …
  -/
  apply Functor.ShiftSequence.induced_shiftIso_hom_app_obj
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homologyFunctor_shiftMap
    {K L : CochainComplex C ℤ} {n : ℤ} (f : K ⟶ L⟦n⟧) (a a' : ℤ) (h : n + a = a') :
    (homologyFunctor C (ComplexShape.up ℤ) 0).shiftMap
      ((quotient _ _).map f ≫ ((quotient _ _).commShiftIso n).hom.app _) a a' h =
        (homologyFunctorFactors _ _ a).hom.app K ≫
          (HomologicalComplex.homologyFunctor C (ComplexShape.up ℤ) 0).shiftMap f a a' h ≫
            (homologyFunctorFactors _ _ a').inv.app L := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : CochainComplex C Int
    n : Int
    f : Quiver.Hom K ((CategoryTheory.shiftFunctor (CochainComplex C Int) n).obj L)
    a a' : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq ((HomotopyCategory.homologyFunctor C (ComplexShape.up Int) 0).shiftMap (C …
  -/
  apply Functor.ShiftSequence.induced_shiftMap
  /-
    🎉 no goals
  -/


