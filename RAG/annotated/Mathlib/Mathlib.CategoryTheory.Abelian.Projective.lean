/-- The preadditive Co-Yoneda functor on `P` preserves homology if `P` is projective. -/
noncomputable instance preservesHomology_preadditiveCoyonedaObj_of_projective
    (P : C) [hP : Projective P] :
    (preadditiveCoyonedaObj (op P)).PreservesHomology := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Projective P
    ⊢ (CategoryTheory.preadditiveCoyonedaObj { unop := P }).PreservesHomology
  -/
  haveI := (projective_iff_preservesEpimorphisms_preadditiveCoyoneda_obj' P).mp hP
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Projective P
    this : (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorp …
    ⊢ (CategoryTheory.preadditiveCoyonedaObj { unop := P }).PreservesHomology
  -/
  haveI := @Functor.preservesEpimorphisms_of_preserves_of_reflects _ _ _ _ _ _ _ _ this _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Projective P
    this✝ : (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimor …
    this : (CategoryTheory.preadditiveCoyonedaObj { unop := P }).PreservesEpimorph …
    ⊢ (CategoryTheory.preadditiveCoyonedaObj { unop := P }).PreservesHomology
  -/
  apply Functor.preservesHomology_of_preservesEpis_and_kernels
  /-
    🎉 no goals
  -/


/-- The preadditive Co-Yoneda functor on `P` preserves finite colimits if `P` is projective. -/
noncomputable instance preservesFiniteColimits_preadditiveCoyonedaObj_of_projective
    (P : C) [hP : Projective P] :
    PreservesFiniteColimits (preadditiveCoyonedaObj (op P)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Projective P
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditiveCoy …
  -/
  apply Functor.preservesFiniteColimits_of_preservesHomology
  /-
    🎉 no goals
  -/


/-- An object is projective if its preadditive Co-Yoneda functor preserves finite colimits. -/
theorem projective_of_preservesFiniteColimits_preadditiveCoyonedaObj (P : C)
    [hP : PreservesFiniteColimits (preadditiveCoyonedaObj (op P))] : Projective P := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    ⊢ CategoryTheory.Projective P
  -/
  rw [projective_iff_preservesEpimorphisms_preadditiveCoyoneda_obj']
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    ⊢ (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorphisms
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    ⊢ ((CategoryTheory.preadditiveCoyonedaObj { unop := P }).comp (CategoryTheory. …
  -/
  have := Functor.preservesHomologyOfExact (preadditiveCoyonedaObj (op P))
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    this : (CategoryTheory.preadditiveCoyonedaObj { unop := P }).PreservesHomology
    ⊢ ((CategoryTheory.preadditiveCoyonedaObj { unop := P }).comp (CategoryTheory. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


