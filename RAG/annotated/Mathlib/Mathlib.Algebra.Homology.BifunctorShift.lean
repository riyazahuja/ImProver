/-- The condition that `((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj K₂` has
a total cochain complex. -/
abbrev HasMapBifunctor := HomologicalComplex.HasMapBifunctor K₁ K₂ F (ComplexShape.up ℤ)


/-- Given `K₁ : CochainComplex C₁ ℤ`, `K₂ : CochainComplex C₂ ℤ`,
a bifunctor `F : C₁ ⥤ C₂ ⥤ D`, this `mapBifunctor K₁ K₂ F : CochainComplex D ℤ`
is the total complex of the bicomplex obtained by applying `F` to `K₁` and `K₂`. -/
noncomputable abbrev mapBifunctor [HasMapBifunctor K₁ K₂ F] : CochainComplex D ℤ :=
  HomologicalComplex.mapBifunctor K₁ K₂ F (ComplexShape.up ℤ)


/-- The inclusion of a summand `(F.obj (K₁.X n₁)).obj (K₂.X n₂) ⟶ (mapBifunctor K₁ K₂ F).X n`
of the total cochain complex when `n₁ + n₂ = n`. -/
noncomputable abbrev ιMapBifunctor [HasMapBifunctor K₁ K₂ F] (n₁ n₂ n : ℤ) (h : n₁ + n₂ = n) :
    (F.obj (K₁.X n₁)).obj (K₂.X n₂) ⟶ (mapBifunctor K₁ K₂ F).X n :=
  HomologicalComplex.ιMapBifunctor K₁ K₂ F _ _ _ _ h


/-- Auxiliary definition for `mapBifunctorShift₁Iso`. -/
@[simps! hom_f_f inv_f_f]
def mapBifunctorHomologicalComplexShift₁Iso :
    ((F.mapBifunctorHomologicalComplex _ _).obj (K₁⟦x⟧)).obj K₂ ≅
    (HomologicalComplex₂.shiftFunctor₁ D x).obj
      (((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj K₂) :=
  HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _) (by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      inst✝⁸ : CategoryTheory.Category.{?u.7565, u_1} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.7569, u_2} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.7573, u_3} D
      inst✝⁵ : CategoryTheory.Preadditive C₁
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝³ : CategoryTheory.Preadditive D
      K₁ : CochainComplex C₁ Int
      K₂ : CochainComplex C₂ Int
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝² : F.Additive
      inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
      x : Int
      inst✝ : K₁.HasMapBifunctor K₂ F
      ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
    -/
    intros
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      inst✝⁸ : CategoryTheory.Category.{?u.7565, u_1} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.7569, u_2} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.7573, u_3} D
      inst✝⁵ : CategoryTheory.Preadditive C₁
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝³ : CategoryTheory.Preadditive D
      K₁ : CochainComplex C₁ Int
      K₂ : CochainComplex C₂ Int
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝² : F.Additive
      inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
      x : Int
      inst✝ : K₁.HasMapBifunctor K₂ F
      i✝ j✝ : Int
      a✝ : (ComplexShape.up Int).Rel i✝ j✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x_1 => CategoryTheory.Iso.refl  …
    -/
    ext
    /-
      case h
      C₁ : Type u_1
      C₂ : Type u_2
      D : Type u_3
      inst✝⁸ : CategoryTheory.Category.{?u.7565, u_1} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.7569, u_2} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.7573, u_3} D
      inst✝⁵ : CategoryTheory.Preadditive C₁
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝³ : CategoryTheory.Preadditive D
      K₁ : CochainComplex C₁ Int
      K₂ : CochainComplex C₂ Int
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
      inst✝² : F.Additive
      inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
      x : Int
      inst✝ : K₁.HasMapBifunctor K₂ F
      i✝¹ j✝ : Int
      a✝ : (ComplexShape.up Int).Rel i✝¹ j✝
      i✝ : Int
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun x_1 => CategoryTheory.Iso.refl …
    -/
    dsimp
    simp only [Linear.comp_units_smul, id_comp, Functor.map_units_smul,
      NatTrans.app_units_zsmul, comp_id])


instance : HasMapBifunctor (K₁⟦x⟧) K₂ F :=
  HomologicalComplex₂.hasTotal_of_iso (mapBifunctorHomologicalComplexShift₁Iso K₁ K₂ F x).symm _


/-- The canonical isomorphism `mapBifunctor (K₁⟦x⟧) K₂ F ≅ (mapBifunctor K₁ K₂ F)⟦x⟧`.
This isomorphism does not involve signs. -/
noncomputable def mapBifunctorShift₁Iso :
    mapBifunctor (K₁⟦x⟧) K₂ F ≅ (mapBifunctor K₁ K₂ F)⟦x⟧ :=
  HomologicalComplex₂.total.mapIso (mapBifunctorHomologicalComplexShift₁Iso K₁ K₂ F x) _ ≪≫
    (((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj K₂).totalShift₁Iso x


/-- Auxiliary definition for `mapBifunctorShift₂Iso`. -/
@[simps! hom_f_f inv_f_f]
def mapBifunctorHomologicalComplexShift₂Iso :
    ((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj (K₂⟦y⟧) ≅
    (HomologicalComplex₂.shiftFunctor₂ D y).obj
      (((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj K₂) :=
  HomologicalComplex.Hom.isoOfComponents
               /-
                 C₁ : Type u_1
                 C₂ : Type u_2
                 D : Type u_3
                 inst✝⁸ : CategoryTheory.Category.{?u.32140, u_1} C₁
                 inst✝⁷ : CategoryTheory.Category.{?u.32144, u_2} C₂
                 inst✝⁶ : CategoryTheory.Category.{?u.32148, u_3} D
                 inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
                 inst✝⁴ : CategoryTheory.Preadditive C₂
                 inst✝³ : CategoryTheory.Preadditive D
                 K₁ : CochainComplex C₁ Int
                 K₂ : CochainComplex C₂ Int
                 F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
                 inst✝² : F.PreservesZeroMorphisms
                 inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
                 y : Int
                 inst✝ : K₁.HasMapBifunctor K₂ F
                 i₁ : Int
                 ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
               -/
    (fun i₁ => HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _)) (by
               /-
                 🎉 no goals
               -/
      /-
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{?u.32140, u_1} C₁
        inst✝⁷ : CategoryTheory.Category.{?u.32144, u_2} C₂
        inst✝⁶ : CategoryTheory.Category.{?u.32148, u_3} D
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝⁴ : CategoryTheory.Preadditive C₂
        inst✝³ : CategoryTheory.Preadditive D
        K₁ : CochainComplex C₁ Int
        K₂ : CochainComplex C₂ Int
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝² : F.PreservesZeroMorphisms
        inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        y : Int
        inst✝ : K₁.HasMapBifunctor K₂ F
        ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
      -/
      intros
      /-
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{?u.32140, u_1} C₁
        inst✝⁷ : CategoryTheory.Category.{?u.32144, u_2} C₂
        inst✝⁶ : CategoryTheory.Category.{?u.32148, u_3} D
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝⁴ : CategoryTheory.Preadditive C₂
        inst✝³ : CategoryTheory.Preadditive D
        K₁ : CochainComplex C₁ Int
        K₂ : CochainComplex C₂ Int
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝² : F.PreservesZeroMorphisms
        inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        y : Int
        inst✝ : K₁.HasMapBifunctor K₂ F
        i✝ j✝ : Int
        a✝ : (ComplexShape.up Int).Rel i✝ j✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ => HomologicalComplex.Hom.is …
      -/
      ext
      /-
        case h
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{?u.32140, u_1} C₁
        inst✝⁷ : CategoryTheory.Category.{?u.32144, u_2} C₂
        inst✝⁶ : CategoryTheory.Category.{?u.32148, u_3} D
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝⁴ : CategoryTheory.Preadditive C₂
        inst✝³ : CategoryTheory.Preadditive D
        K₁ : CochainComplex C₁ Int
        K₂ : CochainComplex C₂ Int
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝² : F.PreservesZeroMorphisms
        inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        y : Int
        inst✝ : K₁.HasMapBifunctor K₂ F
        i✝¹ j✝ : Int
        a✝ : (ComplexShape.up Int).Rel i✝¹ j✝
        i✝ : Int
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun i₁ => HomologicalComplex.Hom.i …
      -/
      dsimp
      /-
        case h
        C₁ : Type u_1
        C₂ : Type u_2
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{?u.32140, u_1} C₁
        inst✝⁷ : CategoryTheory.Category.{?u.32144, u_2} C₂
        inst✝⁶ : CategoryTheory.Category.{?u.32148, u_3} D
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝⁴ : CategoryTheory.Preadditive C₂
        inst✝³ : CategoryTheory.Preadditive D
        K₁ : CochainComplex C₁ Int
        K₂ : CochainComplex C₂ Int
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
        inst✝² : F.PreservesZeroMorphisms
        inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        y : Int
        inst✝ : K₁.HasMapBifunctor K₂ F
        i✝¹ j✝ : Int
        a✝ : (ComplexShape.up Int).Rel i✝¹ j✝
        i✝ : Int
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((F …
      -/
      simp only [id_comp, comp_id])
      /-
        🎉 no goals
      -/


instance : HasMapBifunctor K₁ (K₂⟦y⟧) F :=
  HomologicalComplex₂.hasTotal_of_iso (mapBifunctorHomologicalComplexShift₂Iso K₁ K₂ F y).symm _


/-- The canonical isomorphism `mapBifunctor K₁ (K₂⟦y⟧) F ≅ (mapBifunctor K₁ K₂ F)⟦y⟧`.
This isomorphism involves signs: on the summand `(F.obj (K₁.X p)).obj (K₂.X q)`, it is given
by the multiplication by `(p * y).negOnePow`. -/
noncomputable def mapBifunctorShift₂Iso :
    mapBifunctor K₁ (K₂⟦y⟧) F ≅ (mapBifunctor K₁ K₂ F)⟦y⟧ :=
  HomologicalComplex₂.total.mapIso
    (mapBifunctorHomologicalComplexShift₂Iso K₁ K₂ F y) (ComplexShape.up ℤ) ≪≫
    (((F.mapBifunctorHomologicalComplex _ _).obj K₁).obj K₂).totalShift₂Iso y


lemma mapBifunctorShift₁Iso_trans_mapBifunctorShift₂Iso :
    mapBifunctorShift₁Iso K₁ (K₂⟦y⟧) F x ≪≫
      (CategoryTheory.shiftFunctor _ x).mapIso (mapBifunctorShift₂Iso K₁ K₂ F y) =
      (x * y).negOnePow • (mapBifunctorShift₂Iso (K₁⟦x⟧) K₂ F y ≪≫
        (CategoryTheory.shiftFunctor _ y).mapIso (mapBifunctorShift₁Iso K₁ K₂ F x) ≪≫
          (shiftFunctorComm (CochainComplex D ℤ) x y).app _) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    ⊢ Eq ((K₁.mapBifunctorShift₁Iso ((CategoryTheory.shiftFunctor (CochainComplex  …
  -/
  ext1
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    ⊢ Eq ((K₁.mapBifunctorShift₁Iso ((CategoryTheory.shiftFunctor (CochainComplex  …
  -/
  dsimp [mapBifunctorShift₁Iso, mapBifunctorShift₂Iso]
  rw [Functor.map_comp, Functor.map_comp, assoc, assoc, assoc,
    ← HomologicalComplex₂.totalShift₁Iso_hom_naturality_assoc,
    HomologicalComplex₂.totalShift₁Iso_hom_totalShift₂Iso_hom,
    ← HomologicalComplex₂.totalShift₂Iso_hom_naturality_assoc,
    Linear.comp_units_smul, Linear.comp_units_smul,
    smul_left_cancel_iff,
    ← HomologicalComplex₂.total.map_comp_assoc,
    ← HomologicalComplex₂.total.map_comp_assoc,
    ← HomologicalComplex₂.total.map_comp_assoc]
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.total.map (Categ …
  -/
  congr 2
  /-
    case w.e_a.e_φ
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  ext a b
  /-
    case w.e_a.e_φ.h.h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    a b : Int
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
  -/
  dsimp [HomologicalComplex₂.shiftFunctor₁₂CommIso]
  /-
    case w.e_a.e_φ.h.h
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C₁
    inst✝⁴ : CategoryTheory.Preadditive C₂
    inst✝³ : CategoryTheory.Preadditive D
    K₁ : CochainComplex C₁ Int
    K₂ : CochainComplex C₂ Int
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝² : F.Additive
    inst✝¹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    x y : Int
    inst✝ : K₁.HasMapBifunctor K₂ F
    a b : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [id_comp]
  /-
    🎉 no goals
  -/


