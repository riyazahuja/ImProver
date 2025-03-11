/-- When `F : C ⥤ D` is an additive functor, this is
the functor `C ⥤ HomotopyCategory D (ComplexShape.down ℕ)` which
sends `X : C` to `F` applied to a projective resolution of `X`. -/
noncomputable def Functor.leftDerivedToHomotopyCategory (F : C ⥤ D) [F.Additive] :
    C ⥤ HomotopyCategory D (ComplexShape.down ℕ) :=
  projectiveResolutions C ⋙ F.mapHomotopyCategory _


/-- If `P : ProjectiveResolution Z` and `F : C ⥤ D` is an additive functor, this is
an isomorphism between `F.leftDerivedToHomotopyCategory.obj X` and the complex
obtained by applying `F` to `P.complex`. -/
noncomputable def ProjectiveResolution.isoLeftDerivedToHomotopyCategoryObj {X : C}
    (P : ProjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    F.leftDerivedToHomotopyCategory.obj X ≅
      (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).obj P.complex :=
  (F.mapHomotopyCategory _).mapIso P.iso ≪≫
    (F.mapHomotopyCategoryFactors _).app P.complex


@[reassoc]
lemma ProjectiveResolution.isoLeftDerivedToHomotopyCategoryObj_inv_naturality
    {X Y : C} (f : X ⟶ Y) (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f)
    (F : C ⥤ D) [F.Additive] :
    (P.isoLeftDerivedToHomotopyCategoryObj F).inv ≫ F.leftDerivedToHomotopyCategory.map f =
      (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map φ ≫
        (Q.isoLeftDerivedToHomotopyCategoryObj F).inv := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.isoLeftDerivedToHomotopyCategoryOb …
  -/
  dsimp [Functor.leftDerivedToHomotopyCategory, isoLeftDerivedToHomotopyCategoryObj]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, ← Functor.map_comp, iso_inv_naturality f P Q φ comm, Functor.map_comp]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapHomotopyCategoryFactors (Compl …
  -/
  erw [(F.mapHomotopyCategoryFactors (ComplexShape.down ℕ)).inv.naturality_assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapHomotopyCategoryFactors (Compl …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ProjectiveResolution.isoLeftDerivedToHomotopyCategoryObj_hom_naturality
    {X Y : C} (f : X ⟶ Y) (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f)
    (F : C ⥤ D) [F.Additive] :
    F.leftDerivedToHomotopyCategory.map f ≫ (Q.isoLeftDerivedToHomotopyCategoryObj F).hom =
      (P.isoLeftDerivedToHomotopyCategoryObj F).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map φ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X Y : C
      f : Quiver.Hom X Y
      P : CategoryTheory.ProjectiveResolution X
      Q : CategoryTheory.ProjectiveResolution Y
      φ : Quiver.Hom P.complex Q.complex
      comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.leftDerivedToHomotopyCategory.map  …
    -/
    dsimp
    rw [← cancel_epi (P.isoLeftDerivedToHomotopyCategoryObj F).inv, Iso.inv_hom_id_assoc,
      isoLeftDerivedToHomotopyCategoryObj_inv_naturality_assoc f P Q φ comm F,
      Iso.inv_hom_id, comp_id]


/-- The left derived functors of an additive functor. -/
noncomputable def Functor.leftDerived (F : C ⥤ D) [F.Additive] (n : ℕ) : C ⥤ D :=
  F.leftDerivedToHomotopyCategory ⋙ HomotopyCategory.homologyFunctor D _ n


/-- We can compute a left derived functor using a chosen projective resolution. -/
noncomputable def ProjectiveResolution.isoLeftDerivedObj {X : C} (P : ProjectiveResolution X)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (F.leftDerived n).obj X ≅
      (HomologicalComplex.homologyFunctor D _ n).obj
        ((F.mapHomologicalComplex _).obj P.complex) :=
  (HomotopyCategory.homologyFunctor D _ n).mapIso
    (P.isoLeftDerivedToHomotopyCategoryObj F) ≪≫
    (HomotopyCategory.homologyFunctorFactors D (ComplexShape.down ℕ) n).app _


@[reassoc]
lemma ProjectiveResolution.isoLeftDerivedObj_hom_naturality
    {X Y : C} (f : X ⟶ Y) (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (F.leftDerived n).map f ≫ (Q.isoLeftDerivedObj F n).hom =
      (P.isoLeftDerivedObj F n).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map φ := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.leftDerived n).map f) (Q.isoLeftD …
  -/
  dsimp [isoLeftDerivedObj, Functor.leftDerived]
  rw [assoc, ← Functor.map_comp_assoc,
    ProjectiveResolution.isoLeftDerivedToHomotopyCategoryObj_hom_naturality f P Q φ comm F,
    Functor.map_comp, assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  erw [(HomotopyCategory.homologyFunctorFactors D (ComplexShape.down ℕ) n).hom.naturality]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ProjectiveResolution.isoLeftDerivedObj_inv_naturality
    {X Y : C} (f : X ⟶ Y) (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (P.isoLeftDerivedObj F n).inv ≫ (F.leftDerived n).map f =
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map φ ≫
          (Q.isoLeftDerivedObj F n).inv := by
  rw [← cancel_mono (Q.isoLeftDerivedObj F n).hom, assoc, assoc,
    ProjectiveResolution.isoLeftDerivedObj_hom_naturality f P Q φ comm F n,
    Iso.inv_hom_id_assoc, Iso.inv_hom_id, comp_id]


/-- The higher derived functors vanish on projective objects. -/
lemma Functor.isZero_leftDerived_obj_projective_succ
    (F : C ⥤ D) [F.Additive] (n : ℕ) (X : C) [Projective X] :
    IsZero ((F.leftDerived (n + 1)).obj X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.Limits.IsZero ((F.leftDerived (HAdd.hAdd n 1)).obj X)
  -/
  refine IsZero.of_iso ?_ ((ProjectiveResolution.self X).isoLeftDerivedObj F (n + 1))
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.Limits.IsZero ((HomologicalComplex.homologyFunctor D (Complex …
  -/
  erw [← HomologicalComplex.exactAt_iff_isZero_homology]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ ((F.mapHomologicalComplex (ComplexShape.down Nat)).obj (CategoryTheory.Proje …
  -/
  exact ShortComplex.exact_of_isZero_X₂ _ (F.map_isZero (by apply isZero_zero))
  /-
    🎉 no goals
  -/


/-- We can compute a left derived functor on a morphism using a descent of that morphism
to a chain map between chosen projective resolutions.
-/
theorem Functor.leftDerived_map_eq (F : C ⥤ D) [F.Additive] (n : ℕ) {X Y : C} (f : X ⟶ Y)
    {P : ProjectiveResolution X} {Q : ProjectiveResolution Y} (g : P.complex ⟶ Q.complex)
    (w : g ≫ Q.π = P.π ≫ (ChainComplex.single₀ C).map f) :
    (F.leftDerived n).map f =
      (P.isoLeftDerivedObj F n).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map g ≫
          (Q.isoLeftDerivedObj F n).inv := by
  rw [← cancel_mono (Q.isoLeftDerivedObj F n).hom,
    ProjectiveResolution.isoLeftDerivedObj_hom_naturality f P Q g _ F n,
    assoc, assoc, Iso.inv_hom_id, comp_id]
  rw [← HomologicalComplex.comp_f, w, HomologicalComplex.comp_f,
    ChainComplex.single₀_map_f_zero]


/-- The natural transformation
`F.leftDerivedToHomotopyCategory ⟶ G.leftDerivedToHomotopyCategory` induced by
a natural transformation `F ⟶ G` between additive functors. -/
noncomputable def NatTrans.leftDerivedToHomotopyCategory
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) :
    F.leftDerivedToHomotopyCategory ⟶ G.leftDerivedToHomotopyCategory :=
  whiskerLeft _ (NatTrans.mapHomotopyCategory α (ComplexShape.down ℕ))


lemma ProjectiveResolution.leftDerivedToHomotopyCategory_app_eq
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) {X : C} (P : ProjectiveResolution X) :
    (NatTrans.leftDerivedToHomotopyCategory α).app X =
      (P.isoLeftDerivedToHomotopyCategoryObj F).hom ≫
        (HomotopyCategory.quotient _ _).map
          ((NatTrans.mapHomologicalComplex α _).app P.complex) ≫
          (P.isoLeftDerivedToHomotopyCategoryObj G).inv := by
  rw [← cancel_mono (P.isoLeftDerivedToHomotopyCategoryObj G).hom, assoc, assoc,
      Iso.inv_hom_id, comp_id]
  dsimp [isoLeftDerivedToHomotopyCategoryObj, Functor.mapHomotopyCategoryFactors,
    NatTrans.leftDerivedToHomotopyCategory]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  rw [assoc]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  erw [id_comp, comp_id]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  obtain ⟨β, hβ⟩ := (HomotopyCategory.quotient _ _).map_surjective (iso P).hom
  /-
    case intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    β : Quiver.Hom (CategoryTheory.projectiveResolution X).complex P.complex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.down Nat)).map β) P.iso.hom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  rw [← hβ]
  /-
    case intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    β : Quiver.Hom (CategoryTheory.projectiveResolution X).complex P.complex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.down Nat)).map β) P.iso.hom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  dsimp
  /-
    case intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    β : Quiver.Hom (CategoryTheory.projectiveResolution X).complex P.complex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.down Nat)).map β) P.iso.hom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  simp only [← Functor.map_comp, NatTrans.mapHomologicalComplex_naturality]
  /-
    case intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    β : Quiver.Hom (CategoryTheory.projectiveResolution X).complex P.complex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.down Nat)).map β) P.iso.hom
    ⊢ Eq ((HomotopyCategory.quotient D (ComplexShape.down Nat)).map (CategoryTheor …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma NatTrans.leftDerivedToHomotopyCategory_id (F : C ⥤ D) [F.Additive] :
    NatTrans.leftDerivedToHomotopyCategory (𝟙 F) = 𝟙 _ := rfl


@[simp, reassoc]
lemma NatTrans.leftDerivedToHomotopyCategory_comp {F G H : C ⥤ D} (α : F ⟶ G) (β : G ⟶ H)
    [F.Additive] [G.Additive] [H.Additive] :
    NatTrans.leftDerivedToHomotopyCategory (α ≫ β) =
      NatTrans.leftDerivedToHomotopyCategory α ≫
        NatTrans.leftDerivedToHomotopyCategory β := rfl


/-- The natural transformation between left-derived functors induced by a natural transformation. -/
noncomputable def NatTrans.leftDerived
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) (n : ℕ) :
    F.leftDerived n ⟶ G.leftDerived n :=
  whiskerRight (NatTrans.leftDerivedToHomotopyCategory α) _


@[simp]
theorem NatTrans.leftDerived_id (F : C ⥤ D) [F.Additive] (n : ℕ) :
    NatTrans.leftDerived (𝟙 F) n = 𝟙 (F.leftDerived n) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.NatTrans.leftDerived (CategoryTheory.CategoryStruct.id F) …
  -/
  dsimp only [leftDerived]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.whiskerRight (CategoryTheory.NatTrans.leftDerivedToHomoto …
  -/
  simp only [leftDerivedToHomotopyCategory_id, whiskerRight_id']
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.id (F.leftDerivedToHomotopyCategory.comp ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem NatTrans.leftDerived_comp {F G H : C ⥤ D} [F.Additive] [G.Additive] [H.Additive]
    (α : F ⟶ G) (β : G ⟶ H) (n : ℕ) :
    NatTrans.leftDerived (α ≫ β) n = NatTrans.leftDerived α n ≫ NatTrans.leftDerived β n := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁵ : CategoryTheory.Abelian C
    inst✝⁴ : CategoryTheory.HasProjectiveResolutions C
    inst✝³ : CategoryTheory.Abelian D
    F G H : CategoryTheory.Functor C D
    inst✝² : F.Additive
    inst✝¹ : G.Additive
    inst✝ : H.Additive
    α : Quiver.Hom F G
    β : Quiver.Hom G H
    n : Nat
    ⊢ Eq (CategoryTheory.NatTrans.leftDerived (CategoryTheory.CategoryStruct.comp  …
  -/
  simp [NatTrans.leftDerived]
  /-
    🎉 no goals
  -/


/-- A component of the natural transformation between left-derived functors can be computed
using a chosen projective resolution. -/
lemma leftDerived_app_eq
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) {X : C} (P : ProjectiveResolution X)
    (n : ℕ) : (NatTrans.leftDerived α n).app X =
      (P.isoLeftDerivedObj F n).hom ≫
        (HomologicalComplex.homologyFunctor D (ComplexShape.down ℕ) n).map
        ((NatTrans.mapHomologicalComplex α _).app P.complex) ≫
        (P.isoLeftDerivedObj G n).inv := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    n : Nat
    ⊢ Eq ((CategoryTheory.NatTrans.leftDerived α n).app X) (CategoryTheory.Categor …
  -/
  dsimp [NatTrans.leftDerived, isoLeftDerivedObj]
  rw [ProjectiveResolution.leftDerivedToHomotopyCategory_app_eq α P,
    Functor.map_comp, Functor.map_comp, assoc]
  erw [← (HomotopyCategory.homologyFunctorFactors D (ComplexShape.down ℕ) n).hom.naturality_assoc
    ((NatTrans.mapHomologicalComplex α (ComplexShape.down ℕ)).app P.complex)]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.ProjectiveResolution X
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  simp only [Functor.comp_map, Iso.hom_inv_id_app_assoc]
  /-
    🎉 no goals
  -/


/-- If `P : ProjectiveResolution X` and `F` is an additive functor, this is
the canonical morphism from the opcycles in degree `0` of
`(F.mapHomologicalComplex _).obj P.complex` to `F.obj X`. -/
noncomputable def fromLeftDerivedZero' {X : C}
    (P : ProjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    ((F.mapHomologicalComplex _).obj P.complex).opcycles 0 ⟶ F.obj X :=
                                                            /-
                                                              C : Type u
                                                              inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                              D : Type u_1
                                                              inst✝⁴ : CategoryTheory.Category.{?u.69603, u_1} D
                                                              inst✝³ : CategoryTheory.Abelian C
                                                              inst✝² : CategoryTheory.HasProjectiveResolutions C
                                                              inst✝¹ : CategoryTheory.Abelian D
                                                              X : C
                                                              P : CategoryTheory.ProjectiveResolution X
                                                              F : CategoryTheory.Functor C D
                                                              inst✝ : F.Additive
                                                              ⊢ Eq ((ComplexShape.down Nat).prev 0) 1
                                                            -/
  HomologicalComplex.descOpcycles _ (F.map (P.π.f 0)) 1 (by simp) (by
                                                            /-
                                                              🎉 no goals
                                                            -/
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.69603, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X : C
      P : CategoryTheory.ProjectiveResolution X
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.mapHomologicalComplex (ComplexSh …
    -/
    dsimp
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.69603, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X : C
      P : CategoryTheory.ProjectiveResolution X
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (P.complex.d 1 0)) (F.map (P.π …
    -/
    rw [← F.map_comp, complex_d_comp_π_f_zero, F.map_zero])
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pOpcycles_comp_fromLeftDerivedZero' {C} [Category C] [Abelian C] {X : C}
    (P : ProjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    HomologicalComplex.pOpcycles _ _ ≫ P.fromLeftDerivedZero' F = F.map (P.π.f 0) := by
  /-
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D
    inst✝³ : CategoryTheory.Abelian D
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Abelian C
    X : C
    P : CategoryTheory.ProjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.mapHomologicalComplex (ComplexSh …
  -/
  simp [fromLeftDerivedZero']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma fromLeftDerivedZero'_naturality {C} [Category C] [Abelian C] {X Y : C} (f : X ⟶ Y)
    (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f)
    (F : C ⥤ D) [F.Additive] :
    HomologicalComplex.opcyclesMap ((F.mapHomologicalComplex _).map φ) 0 ≫
        Q.fromLeftDerivedZero' F = P.fromLeftDerivedZero' F ≫ F.map f := by
  simp only [← cancel_epi (HomologicalComplex.pOpcycles _ _), ← F.map_comp, comm,
    HomologicalComplex.p_opcyclesMap_assoc, Functor.mapHomologicalComplex_map_f,
    pOpcycles_comp_fromLeftDerivedZero', pOpcycles_comp_fromLeftDerivedZero'_assoc]


instance (F : C ⥤ D) [F.Additive] (X : C) [Projective X] :
    IsIso ((ProjectiveResolution.self X).fromLeftDerivedZero' F) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.IsIso ((CategoryTheory.ProjectiveResolution.self X).fromLeftD …
  -/
  dsimp [ProjectiveResolution.fromLeftDerivedZero']
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.IsIso (((F.mapHomologicalComplex (ComplexShape.down Nat)).obj …
  -/
  rw [ChainComplex.isIso_descOpcycles_iff]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ And (CategoryTheory.ShortComplex.mk (((F.mapHomologicalComplex (ComplexShape …
  -/
  refine ⟨ShortComplex.Splitting.exact ?_, inferInstance⟩
  exact
    { r := 0
      s := 𝟙 _
      f_r := (F.map_isZero (isZero_zero _)).eq_of_src _ _ }


/-- The natural transformation `F.leftDerived 0 ⟶ F`. -/
noncomputable def Functor.fromLeftDerivedZero (F : C ⥤ D) [F.Additive] :
    F.leftDerived 0 ⟶ F where
  app X := (HomotopyCategory.homologyFunctorFactors D (ComplexShape.down ℕ) 0).hom.app _ ≫
      (ChainComplex.isoHomologyι₀ _).hom ≫ (projectiveResolution X).fromLeftDerivedZero' F
  naturality {X Y} f := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96386, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.leftDerived 0).map f) ((fun X =>  …
    -/
    dsimp [leftDerived]
    rw [assoc, assoc, ← ProjectiveResolution.fromLeftDerivedZero'_naturality f
      (projectiveResolution X) (projectiveResolution Y)
      (ProjectiveResolution.lift f _ _) (by simp),
      ← HomologicalComplex.homologyι_naturality_assoc]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96386, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
    -/
    erw [← NatTrans.naturality_assoc]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96386, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasProjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma ProjectiveResolution.fromLeftDerivedZero_eq
    {X : C} (P : ProjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    F.fromLeftDerivedZero.app X = (P.isoLeftDerivedObj F 0).hom ≫
      (ChainComplex.isoHomologyι₀ _).hom ≫
        P.fromLeftDerivedZero' F := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    P : CategoryTheory.ProjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (F.fromLeftDerivedZero.app X) (CategoryTheory.CategoryStruct.comp (P.isoL …
  -/
  dsimp [Functor.fromLeftDerivedZero, isoLeftDerivedObj]
  have h₁ := ProjectiveResolution.fromLeftDerivedZero'_naturality
    (𝟙 X) P (projectiveResolution X) (lift (𝟙 X) _ _) (by simp) F
  have h₂ : (P.isoLeftDerivedToHomotopyCategoryObj F).inv =
    (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map (lift (𝟙 X) _ _) :=
      id_comp _
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    P : CategoryTheory.ProjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.opcyclesMap (( …
    h₂ : Eq (P.isoLeftDerivedToHomotopyCategoryObj F).inv (((F.mapHomologicalCompl …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctorFac …
  -/
  simp only [Functor.map_id, comp_id] at h₁
  rw [assoc, ← cancel_epi ((HomotopyCategory.homologyFunctor _ _ 0).map
      (P.isoLeftDerivedToHomotopyCategoryObj F).inv), ← Functor.map_comp_assoc,
      Iso.inv_hom_id, Functor.map_id, id_comp, ← h₁, h₂,
      ← HomologicalComplex.homologyι_naturality_assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    P : CategoryTheory.ProjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₂ : Eq (P.isoLeftDerivedToHomotopyCategoryObj F).inv (((F.mapHomologicalCompl …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.opcyclesMap (( …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  erw [← NatTrans.naturality_assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasProjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    P : CategoryTheory.ProjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₂ : Eq (P.isoLeftDerivedToHomotopyCategoryObj F).inv (((F.mapHomologicalCompl …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.opcyclesMap (( …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (F : C ⥤ D) [F.Additive] (X : C) [Projective X] :
    IsIso (F.fromLeftDerivedZero.app X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.IsIso (F.fromLeftDerivedZero.app X)
  -/
  rw [(ProjectiveResolution.self X).fromLeftDerivedZero_eq F]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Projective X
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pr …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {X : C} (P : ProjectiveResolution X) :
    IsIso (P.fromLeftDerivedZero' F) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ CategoryTheory.IsIso (P.fromLeftDerivedZero' F)
  -/
  dsimp [ProjectiveResolution.fromLeftDerivedZero']
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ CategoryTheory.IsIso (((F.mapHomologicalComplex (ComplexShape.down Nat)).obj …
  -/
  rw [ChainComplex.isIso_descOpcycles_iff, ShortComplex.exact_and_epi_g_iff_g_is_cokernel]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCof …
  -/
  exact ⟨CokernelCofork.mapIsColimit _ (P.isColimitCokernelCofork) F⟩
  /-
    🎉 no goals
  -/


instance (X : C) : IsIso (F.fromLeftDerivedZero.app X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    X : C
    ⊢ CategoryTheory.IsIso (F.fromLeftDerivedZero.app X)
  -/
  dsimp [Functor.fromLeftDerivedZero]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasProjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    X : C
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((HomotopyCategory. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : IsIso F.fromLeftDerivedZero :=
  NatIso.isIso_of_isIso_app _


/-- The canonical isomorphism `F.leftDerived 0 ≅ F` when `F` is right exact
(i.e. preserves finite colimits). -/
@[simps! hom]
noncomputable def leftDerivedZeroIsoSelf : F.leftDerived 0 ≅ F :=
  (asIso F.fromLeftDerivedZero)


@[reassoc (attr := simp)]
lemma leftDerivedZeroIsoSelf_hom_inv_id :
    F.fromLeftDerivedZero ≫ F.leftDerivedZeroIsoSelf.inv = 𝟙 _ :=
  F.leftDerivedZeroIsoSelf.hom_inv_id


@[reassoc (attr := simp)]
lemma leftDerivedZeroIsoSelf_inv_hom_id :
    F.leftDerivedZeroIsoSelf.inv ≫ F.fromLeftDerivedZero =  𝟙 _ :=
  F.leftDerivedZeroIsoSelf.inv_hom_id


@[reassoc (attr := simp)]
lemma leftDerivedZeroIsoSelf_hom_inv_id_app (X : C) :
    F.fromLeftDerivedZero.app X ≫ F.leftDerivedZeroIsoSelf.inv.app X = 𝟙 _ :=
  F.leftDerivedZeroIsoSelf.hom_inv_id_app X


@[reassoc (attr := simp)]
lemma leftDerivedZeroIsoSelf_inv_hom_id_app (X : C) :
    F.leftDerivedZeroIsoSelf.inv.app X ≫ F.fromLeftDerivedZero.app X = 𝟙 _ :=
  F.leftDerivedZeroIsoSelf.inv_hom_id_app X


