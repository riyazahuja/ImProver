/-- When `F : C ⥤ D` is an additive functor, this is
the functor `C ⥤ HomotopyCategory D (ComplexShape.up ℕ)` which
sends `X : C` to `F` applied to an injective resolution of `X`. -/
noncomputable def Functor.rightDerivedToHomotopyCategory (F : C ⥤ D) [F.Additive] :
    C ⥤ HomotopyCategory D (ComplexShape.up ℕ) :=
  injectiveResolutions C ⋙ F.mapHomotopyCategory _


/-- If `I : InjectiveResolution Z` and `F : C ⥤ D` is an additive functor, this is
an isomorphism between `F.rightDerivedToHomotopyCategory.obj X` and the complex
obtained by applying `F` to `I.cocomplex`. -/
noncomputable def InjectiveResolution.isoRightDerivedToHomotopyCategoryObj {X : C}
    (I : InjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    F.rightDerivedToHomotopyCategory.obj X ≅
      (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).obj I.cocomplex :=
  (F.mapHomotopyCategory _).mapIso I.iso ≪≫
    (F.mapHomotopyCategoryFactors _).app I.cocomplex


@[reassoc]
lemma InjectiveResolution.isoRightDerivedToHomotopyCategoryObj_hom_naturality
    {X Y : C} (f : X ⟶ Y) (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0)
    (F : C ⥤ D) [F.Additive] :
    F.rightDerivedToHomotopyCategory.map f ≫ (J.isoRightDerivedToHomotopyCategoryObj F).hom =
      (I.isoRightDerivedToHomotopyCategoryObj F).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map φ := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.rightDerivedToHomotopyCategory.map …
  -/
  dsimp [Functor.rightDerivedToHomotopyCategory, isoRightDerivedToHomotopyCategoryObj]
  rw [← Functor.map_comp_assoc, iso_hom_naturality f I J φ comm, Functor.map_comp,
    assoc, assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapHomotopyCategory (ComplexShape …
  -/
  erw [(F.mapHomotopyCategoryFactors (ComplexShape.up ℕ)).hom.naturality]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapHomotopyCategory (ComplexShape …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma InjectiveResolution.isoRightDerivedToHomotopyCategoryObj_inv_naturality
    {X Y : C} (f : X ⟶ Y) (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0)
    (F : C ⥤ D) [F.Additive] :
    (I.isoRightDerivedToHomotopyCategoryObj F).inv ≫ F.rightDerivedToHomotopyCategory.map f =
      (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map φ ≫
        (J.isoRightDerivedToHomotopyCategoryObj F).inv := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X Y : C
      f : Quiver.Hom X Y
      I : CategoryTheory.InjectiveResolution X
      J : CategoryTheory.InjectiveResolution Y
      φ : Quiver.Hom I.cocomplex J.cocomplex
      comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (I.isoRightDerivedToHomotopyCategoryO …
    -/
    rw [← cancel_epi (I.isoRightDerivedToHomotopyCategoryObj F).hom, Iso.hom_inv_id_assoc]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X Y : C
      f : Quiver.Hom X Y
      I : CategoryTheory.InjectiveResolution X
      J : CategoryTheory.InjectiveResolution Y
      φ : Quiver.Hom I.cocomplex J.cocomplex
      comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (F.rightDerivedToHomotopyCategory.map f) (CategoryTheory.CategoryStruct.c …
    -/
    dsimp
    rw [← isoRightDerivedToHomotopyCategoryObj_hom_naturality_assoc f I J φ comm F,
      Iso.hom_inv_id, comp_id]


/-- The right derived functors of an additive functor. -/
noncomputable def Functor.rightDerived (F : C ⥤ D) [F.Additive] (n : ℕ) : C ⥤ D :=
  F.rightDerivedToHomotopyCategory ⋙ HomotopyCategory.homologyFunctor D _ n


/-- We can compute a right derived functor using a chosen injective resolution. -/
noncomputable def InjectiveResolution.isoRightDerivedObj {X : C} (I : InjectiveResolution X)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (F.rightDerived n).obj X ≅
      (HomologicalComplex.homologyFunctor D _ n).obj
        ((F.mapHomologicalComplex _).obj I.cocomplex) :=
  (HomotopyCategory.homologyFunctor D _ n).mapIso
    (I.isoRightDerivedToHomotopyCategoryObj F) ≪≫
    (HomotopyCategory.homologyFunctorFactors D (ComplexShape.up ℕ) n).app _


@[reassoc]
lemma InjectiveResolution.isoRightDerivedObj_hom_naturality
    {X Y : C} (f : X ⟶ Y) (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (F.rightDerived n).map f ≫ (J.isoRightDerivedObj F n).hom =
      (I.isoRightDerivedObj F n).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map φ := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.rightDerived n).map f) (J.isoRigh …
  -/
  dsimp [isoRightDerivedObj, Functor.rightDerived]
  rw [assoc, ← Functor.map_comp_assoc,
    InjectiveResolution.isoRightDerivedToHomotopyCategoryObj_hom_naturality f I J φ comm F,
    Functor.map_comp, assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  erw [(HomotopyCategory.homologyFunctorFactors D (ComplexShape.up ℕ) n).hom.naturality]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
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
lemma InjectiveResolution.isoRightDerivedObj_inv_naturality
    {X Y : C} (f : X ⟶ Y) (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0)
    (F : C ⥤ D) [F.Additive] (n : ℕ) :
    (I.isoRightDerivedObj F n).inv ≫ (F.rightDerived n).map f =
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map φ ≫
          (J.isoRightDerivedObj F n).inv := by
  rw [← cancel_mono (J.isoRightDerivedObj F n).hom, assoc, assoc,
    InjectiveResolution.isoRightDerivedObj_hom_naturality f I J φ comm F n,
    Iso.inv_hom_id_assoc, Iso.inv_hom_id, comp_id]


/-- The higher derived functors vanish on injective objects. -/
lemma Functor.isZero_rightDerived_obj_injective_succ
    (F : C ⥤ D) [F.Additive] (n : ℕ) (X : C) [Injective X] :
    IsZero ((F.rightDerived (n+1)).obj X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.Limits.IsZero ((F.rightDerived (HAdd.hAdd n 1)).obj X)
  -/
  refine IsZero.of_iso ?_ ((InjectiveResolution.self X).isoRightDerivedObj F (n + 1))
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.Limits.IsZero ((HomologicalComplex.homologyFunctor D (Complex …
  -/
  erw [← HomologicalComplex.exactAt_iff_isZero_homology]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    n : Nat
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ ((F.mapHomologicalComplex (ComplexShape.up Nat)).obj (CategoryTheory.Injecti …
  -/
  exact ShortComplex.exact_of_isZero_X₂ _ (F.map_isZero (by apply isZero_zero))
  /-
    🎉 no goals
  -/


/-- We can compute a right derived functor on a morphism using a descent of that morphism
to a cochain map between chosen injective resolutions.
-/
theorem Functor.rightDerived_map_eq (F : C ⥤ D) [F.Additive] (n : ℕ) {X Y : C} (f : X ⟶ Y)
    {P : InjectiveResolution X} {Q : InjectiveResolution Y} (g : P.cocomplex ⟶ Q.cocomplex)
    (w : P.ι ≫ g = (CochainComplex.single₀ C).map f ≫ Q.ι) :
    (F.rightDerived n).map f =
      (P.isoRightDerivedObj F n).hom ≫
        (F.mapHomologicalComplex _ ⋙ HomologicalComplex.homologyFunctor _ _ n).map g ≫
          (Q.isoRightDerivedObj F n).inv := by
  rw [← cancel_mono (Q.isoRightDerivedObj F n).hom,
    InjectiveResolution.isoRightDerivedObj_hom_naturality f P Q g _ F n,
    assoc, assoc, Iso.inv_hom_id, comp_id]
  rw [← HomologicalComplex.comp_f, w, HomologicalComplex.comp_f,
    CochainComplex.single₀_map_f_zero]


/-- The natural transformation
`F.rightDerivedToHomotopyCategory ⟶ G.rightDerivedToHomotopyCategory` induced by
a natural transformation `F ⟶ G` between additive functors. -/
noncomputable def NatTrans.rightDerivedToHomotopyCategory
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) :
    F.rightDerivedToHomotopyCategory ⟶ G.rightDerivedToHomotopyCategory :=
  whiskerLeft _ (NatTrans.mapHomotopyCategory α (ComplexShape.up ℕ))


lemma InjectiveResolution.rightDerivedToHomotopyCategory_app_eq
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) {X : C} (P : InjectiveResolution X) :
    (NatTrans.rightDerivedToHomotopyCategory α).app X =
      (P.isoRightDerivedToHomotopyCategoryObj F).hom ≫
        (HomotopyCategory.quotient _ _).map
          ((NatTrans.mapHomologicalComplex α _).app P.cocomplex) ≫
          (P.isoRightDerivedToHomotopyCategoryObj G).inv := by
  rw [← cancel_mono (P.isoRightDerivedToHomotopyCategoryObj G).hom, assoc, assoc,
      Iso.inv_hom_id, comp_id]
  dsimp [isoRightDerivedToHomotopyCategoryObj, Functor.mapHomotopyCategoryFactors,
    NatTrans.rightDerivedToHomotopyCategory]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  rw [assoc]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient D (Comple …
  -/
  erw [id_comp, comp_id]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
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
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    β : Quiver.Hom (CategoryTheory.injectiveResolution X).cocomplex P.cocomplex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.up Nat)).map β) P.iso.hom
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
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    β : Quiver.Hom (CategoryTheory.injectiveResolution X).cocomplex P.cocomplex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.up Nat)).map β) P.iso.hom
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
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    β : Quiver.Hom (CategoryTheory.injectiveResolution X).cocomplex P.cocomplex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.up Nat)).map β) P.iso.hom
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
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    β : Quiver.Hom (CategoryTheory.injectiveResolution X).cocomplex P.cocomplex
    hβ : Eq ((HomotopyCategory.quotient C (ComplexShape.up Nat)).map β) P.iso.hom
    ⊢ Eq ((HomotopyCategory.quotient D (ComplexShape.up Nat)).map (CategoryTheory. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma NatTrans.rightDerivedToHomotopyCategory_id (F : C ⥤ D) [F.Additive] :
    NatTrans.rightDerivedToHomotopyCategory (𝟙 F) = 𝟙 _ := rfl


@[simp, reassoc]
lemma NatTrans.rightDerivedToHomotopyCategory_comp {F G H : C ⥤ D} (α : F ⟶ G) (β : G ⟶ H)
    [F.Additive] [G.Additive] [H.Additive] :
    NatTrans.rightDerivedToHomotopyCategory (α ≫ β) =
      NatTrans.rightDerivedToHomotopyCategory α ≫
        NatTrans.rightDerivedToHomotopyCategory β := rfl


/-- The natural transformation between right-derived functors
induced by a natural transformation. -/
noncomputable def NatTrans.rightDerived
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) (n : ℕ) :
    F.rightDerived n ⟶ G.rightDerived n :=
  whiskerRight (NatTrans.rightDerivedToHomotopyCategory α) _


@[simp]
theorem NatTrans.rightDerived_id (F : C ⥤ D) [F.Additive] (n : ℕ) :
    NatTrans.rightDerived (𝟙 F) n = 𝟙 (F.rightDerived n) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.NatTrans.rightDerived (CategoryTheory.CategoryStruct.id F …
  -/
  dsimp only [rightDerived]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.whiskerRight (CategoryTheory.NatTrans.rightDerivedToHomot …
  -/
  simp only [rightDerivedToHomotopyCategory_id, whiskerRight_id']
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.id (F.rightDerivedToHomotopyCategory.comp  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem NatTrans.rightDerived_comp {F G H : C ⥤ D} [F.Additive] [G.Additive] [H.Additive]
    (α : F ⟶ G) (β : G ⟶ H) (n : ℕ) :
    NatTrans.rightDerived (α ≫ β) n = NatTrans.rightDerived α n ≫ NatTrans.rightDerived β n := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁵ : CategoryTheory.Abelian C
    inst✝⁴ : CategoryTheory.HasInjectiveResolutions C
    inst✝³ : CategoryTheory.Abelian D
    F G H : CategoryTheory.Functor C D
    inst✝² : F.Additive
    inst✝¹ : G.Additive
    inst✝ : H.Additive
    α : Quiver.Hom F G
    β : Quiver.Hom G H
    n : Nat
    ⊢ Eq (CategoryTheory.NatTrans.rightDerived (CategoryTheory.CategoryStruct.comp …
  -/
  simp [NatTrans.rightDerived]
  /-
    🎉 no goals
  -/


/-- A component of the natural transformation between right-derived functors can be computed
using a chosen injective resolution. -/
lemma rightDerived_app_eq
    {F G : C ⥤ D} [F.Additive] [G.Additive] (α : F ⟶ G) {X : C} (P : InjectiveResolution X)
    (n : ℕ) : (NatTrans.rightDerived α n).app X =
      (P.isoRightDerivedObj F n).hom ≫
        (HomologicalComplex.homologyFunctor D (ComplexShape.up ℕ) n).map
        ((NatTrans.mapHomologicalComplex α _).app P.cocomplex) ≫
        (P.isoRightDerivedObj G n).inv := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    n : Nat
    ⊢ Eq ((CategoryTheory.NatTrans.rightDerived α n).app X) (CategoryTheory.Catego …
  -/
  dsimp [NatTrans.rightDerived, isoRightDerivedObj]
  rw [InjectiveResolution.rightDerivedToHomotopyCategory_app_eq α P,
    Functor.map_comp, Functor.map_comp, assoc]
  erw [← (HomotopyCategory.homologyFunctorFactors D (ComplexShape.up ℕ) n).hom.naturality_assoc
    ((NatTrans.mapHomologicalComplex α (ComplexShape.up ℕ)).app P.cocomplex)]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F G : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : G.Additive
    α : Quiver.Hom F G
    X : C
    P : CategoryTheory.InjectiveResolution X
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctor D  …
  -/
  simp only [Functor.comp_map, Iso.hom_inv_id_app_assoc]
  /-
    🎉 no goals
  -/


/-- If `P : InjectiveResolution X` and `F` is an additive functor, this is
the canonical morphism from `F.obj X` to the cycles in degree `0` of
`(F.mapHomologicalComplex _).obj P.cocomplex`. -/
noncomputable def toRightDerivedZero' {X : C}
    (P : InjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    F.obj X ⟶ ((F.mapHomologicalComplex _).obj P.cocomplex).cycles 0 :=
                                                          /-
                                                            C : Type u
                                                            inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                            D : Type u_1
                                                            inst✝⁴ : CategoryTheory.Category.{?u.69757, u_1} D
                                                            inst✝³ : CategoryTheory.Abelian C
                                                            inst✝² : CategoryTheory.HasInjectiveResolutions C
                                                            inst✝¹ : CategoryTheory.Abelian D
                                                            X : C
                                                            P : CategoryTheory.InjectiveResolution X
                                                            F : CategoryTheory.Functor C D
                                                            inst✝ : F.Additive
                                                            ⊢ Eq ((ComplexShape.up Nat).next 0) 1
                                                          -/
  HomologicalComplex.liftCycles _ (F.map (P.ι.f 0)) 1 (by simp) (by
                                                          /-
                                                            🎉 no goals
                                                          -/
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.69757, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      X : C
      P : CategoryTheory.InjectiveResolution X
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (P.ι.f 0)) (((F.mapHomological …
    -/
    dsimp
    rw [← F.map_comp, HomologicalComplex.Hom.comm, HomologicalComplex.single_obj_d,
      zero_comp, F.map_zero])


@[reassoc (attr := simp)]
lemma toRightDerivedZero'_comp_iCycles {C} [Category C] [Abelian C] {X : C}
    (P : InjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    P.toRightDerivedZero' F ≫
      HomologicalComplex.iCycles _ _ = F.map (P.ι.f 0) := by
  /-
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D
    inst✝³ : CategoryTheory.Abelian D
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Abelian C
    X : C
    P : CategoryTheory.InjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.toRightDerivedZero' F) (((F.mapHom …
  -/
  simp [toRightDerivedZero']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma toRightDerivedZero'_naturality {C} [Category C] [Abelian C] {X Y : C} (f : X ⟶ Y)
    (P : InjectiveResolution X) (Q : InjectiveResolution Y)
    (φ : P.cocomplex ⟶ Q.cocomplex) (comm : P.ι.f 0 ≫ φ.f 0 = f ≫ Q.ι.f 0)
    (F : C ⥤ D) [F.Additive] :
    F.map f ≫ Q.toRightDerivedZero' F =
      P.toRightDerivedZero' F ≫
        HomologicalComplex.cyclesMap ((F.mapHomologicalComplex _).map φ) 0 := by
  simp only [← cancel_mono (HomologicalComplex.iCycles _ _),
    Functor.mapHomologicalComplex_obj_X, assoc, toRightDerivedZero'_comp_iCycles,
    CochainComplex.single₀_obj_zero, HomologicalComplex.cyclesMap_i,
    Functor.mapHomologicalComplex_map_f, toRightDerivedZero'_comp_iCycles_assoc,
    ← F.map_comp, comm]


instance (F : C ⥤ D) [F.Additive] (X : C) [Injective X] :
    IsIso ((InjectiveResolution.self X).toRightDerivedZero' F) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.IsIso ((CategoryTheory.InjectiveResolution.self X).toRightDer …
  -/
  dsimp [InjectiveResolution.toRightDerivedZero']
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.IsIso (((F.mapHomologicalComplex (ComplexShape.up Nat)).obj ( …
  -/
  rw [CochainComplex.isIso_liftCycles_iff]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ And (CategoryTheory.ShortComplex.mk (F.map (CategoryTheory.CategoryStruct.id …
  -/
  refine ⟨ShortComplex.Splitting.exact ?_, inferInstance⟩
  exact
    { r := 𝟙 _
      s := 0
      s_g := (F.map_isZero (isZero_zero _)).eq_of_src _ _ }


/-- The natural transformation `F ⟶ F.rightDerived 0`. -/
noncomputable def Functor.toRightDerivedZero (F : C ⥤ D) [F.Additive] :
    F ⟶ F.rightDerived 0 where
  app X := (injectiveResolution X).toRightDerivedZero' F ≫
    (CochainComplex.isoHomologyπ₀ _).hom ≫
      (HomotopyCategory.homologyFunctorFactors D (ComplexShape.up ℕ) 0).inv.app _
  naturality {X Y} f := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96869, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X => CategoryTheory.C …
    -/
    dsimp [rightDerived]
    rw [assoc, assoc, InjectiveResolution.toRightDerivedZero'_naturality_assoc f
      (injectiveResolution X) (injectiveResolution Y)
      (InjectiveResolution.desc f _ _) (by simp),
      ← HomologicalComplex.homologyπ_naturality_assoc]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96869, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolution  …
    -/
    erw [← NatTrans.naturality]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.96869, u_1} D
      inst✝³ : CategoryTheory.Abelian C
      inst✝² : CategoryTheory.HasInjectiveResolutions C
      inst✝¹ : CategoryTheory.Abelian D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolution  …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma InjectiveResolution.toRightDerivedZero_eq
    {X : C} (I : InjectiveResolution X) (F : C ⥤ D) [F.Additive] :
    F.toRightDerivedZero.app X = I.toRightDerivedZero' F ≫
      (CochainComplex.isoHomologyπ₀ _).hom ≫ (I.isoRightDerivedObj F 0).inv := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    I : CategoryTheory.InjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq (F.toRightDerivedZero.app X) (CategoryTheory.CategoryStruct.comp (I.toRig …
  -/
  dsimp [Functor.toRightDerivedZero, isoRightDerivedObj]
  have h₁ := InjectiveResolution.toRightDerivedZero'_naturality
    (𝟙 X) (injectiveResolution X) I (desc (𝟙 X) _ _) (by simp) F
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    I : CategoryTheory.InjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStr …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolution  …
  -/
  simp only [Functor.map_id, id_comp] at h₁
  have h₂ : (I.isoRightDerivedToHomotopyCategoryObj F).hom =
    (F.mapHomologicalComplex _ ⋙ HomotopyCategory.quotient _ _).map (desc (𝟙 X) _ _) :=
    comp_id _
  rw [← cancel_mono ((HomotopyCategory.homologyFunctor _ _ 0).map
      (I.isoRightDerivedToHomotopyCategoryObj F).hom),
    assoc, assoc, assoc, assoc, assoc, ← Functor.map_comp,
    Iso.inv_hom_id, Functor.map_id, comp_id,
    reassoc_of% h₁, h₂, ← HomologicalComplex.homologyπ_naturality_assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    I : CategoryTheory.InjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₁ : Eq (I.toRightDerivedZero' F) (CategoryTheory.CategoryStruct.comp ((Catego …
    h₂ : Eq (I.isoRightDerivedToHomotopyCategoryObj F).hom (((F.mapHomologicalComp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolution  …
  -/
  erw [← NatTrans.naturality]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasInjectiveResolutions C
    inst✝¹ : CategoryTheory.Abelian D
    X : C
    I : CategoryTheory.InjectiveResolution X
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    h₁ : Eq (I.toRightDerivedZero' F) (CategoryTheory.CategoryStruct.comp ((Catego …
    h₂ : Eq (I.isoRightDerivedToHomotopyCategoryObj F).hom (((F.mapHomologicalComp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolution  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (F : C ⥤ D) [F.Additive] (X : C) [Injective X] :
    IsIso (F.toRightDerivedZero.app X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.IsIso (F.toRightDerivedZero.app X)
  -/
  rw [(InjectiveResolution.self X).toRightDerivedZero_eq F]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    X : C
    inst✝ : CategoryTheory.Injective X
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.In …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {X : C} (P : InjectiveResolution X) :
    IsIso (P.toRightDerivedZero' F) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    X : C
    P : CategoryTheory.InjectiveResolution X
    ⊢ CategoryTheory.IsIso (P.toRightDerivedZero' F)
  -/
  dsimp [InjectiveResolution.toRightDerivedZero']
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    X : C
    P : CategoryTheory.InjectiveResolution X
    ⊢ CategoryTheory.IsIso (((F.mapHomologicalComplex (ComplexShape.up Nat)).obj P …
  -/
  rw [CochainComplex.isIso_liftCycles_iff, ShortComplex.exact_and_mono_f_iff_f_is_kernel]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    X : C
    P : CategoryTheory.InjectiveResolution X
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.of …
  -/
  exact ⟨KernelFork.mapIsLimit _ (P.isLimitKernelFork) F⟩
  /-
    🎉 no goals
  -/


instance (X : C) : IsIso (F.toRightDerivedZero.app X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    X : C
    ⊢ CategoryTheory.IsIso (F.toRightDerivedZero.app X)
  -/
  dsimp [Functor.toRightDerivedZero]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} D
    inst✝⁴ : CategoryTheory.Abelian C
    inst✝³ : CategoryTheory.HasInjectiveResolutions C
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    X : C
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.in …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : IsIso F.toRightDerivedZero :=
  NatIso.isIso_of_isIso_app _


/-- The canonical isomorphism `F.rightDerived 0 ≅ F` when `F` is left exact
(i.e. preserves finite limits). -/
@[simps! inv]
noncomputable def rightDerivedZeroIsoSelf : F.rightDerived 0 ≅ F :=
  (asIso F.toRightDerivedZero).symm


@[reassoc (attr := simp)]
lemma rightDerivedZeroIsoSelf_hom_inv_id :
    F.rightDerivedZeroIsoSelf.hom ≫ F.toRightDerivedZero = 𝟙 _ :=
  F.rightDerivedZeroIsoSelf.hom_inv_id


@[reassoc (attr := simp)]
lemma rightDerivedZeroIsoSelf_inv_hom_id :
    F.toRightDerivedZero ≫ F.rightDerivedZeroIsoSelf.hom = 𝟙 _ :=
  F.rightDerivedZeroIsoSelf.inv_hom_id


@[reassoc (attr := simp)]
lemma rightDerivedZeroIsoSelf_hom_inv_id_app (X : C) :
    F.rightDerivedZeroIsoSelf.hom.app X ≫ F.toRightDerivedZero.app X = 𝟙 _ :=
  F.rightDerivedZeroIsoSelf.hom_inv_id_app X


@[reassoc (attr := simp)]
lemma rightDerivedZeroIsoSelf_inv_hom_id_app (X : C) :
    F.toRightDerivedZero.app X ≫ F.rightDerivedZeroIsoSelf.hom.app X = 𝟙 _ :=
  F.rightDerivedZeroIsoSelf.inv_hom_id_app X


