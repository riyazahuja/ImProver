/-- The preadditive Yoneda functor on `J` preserves homology if `J` is injective. -/
instance preservesHomology_preadditiveYonedaObj_of_injective (J : C) [hJ : Injective J] :
    (preadditiveYonedaObj J).PreservesHomology  := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hJ : CategoryTheory.Injective J
    ⊢ (CategoryTheory.preadditiveYonedaObj J).PreservesHomology
  -/
  letI := (injective_iff_preservesEpimorphisms_preadditive_yoneda_obj' J).mp hJ
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hJ : CategoryTheory.Injective J
    this : (CategoryTheory.preadditiveYonedaObj J).PreservesEpimorphisms := (Categ …
    ⊢ (CategoryTheory.preadditiveYonedaObj J).PreservesHomology
  -/
  apply Functor.preservesHomology_of_preservesEpis_and_kernels
  /-
    🎉 no goals
  -/


/-- The preadditive Yoneda functor on `J` preserves colimits if `J` is injective. -/
instance preservesFiniteColimits_preadditiveYonedaObj_of_injective (J : C) [hP : Injective J] :
    PreservesFiniteColimits (preadditiveYonedaObj J) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hP : CategoryTheory.Injective J
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditiveYon …
  -/
  apply Functor.preservesFiniteColimits_of_preservesHomology
  /-
    🎉 no goals
  -/


/-- An object is injective if its preadditive Yoneda functor preserves finite colimits. -/
theorem injective_of_preservesFiniteColimits_preadditiveYonedaObj (J : C)
    [hP : PreservesFiniteColimits (preadditiveYonedaObj J)] : Injective J := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    ⊢ CategoryTheory.Injective J
  -/
  rw [injective_iff_preservesEpimorphisms_preadditive_yoneda_obj']
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    ⊢ (CategoryTheory.preadditiveYonedaObj J).PreservesEpimorphisms
  -/
  have := Functor.preservesHomologyOfExact (preadditiveYonedaObj J)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    J : C
    hP : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.preadditive …
    this : (CategoryTheory.preadditiveYonedaObj J).PreservesHomology
    ⊢ (CategoryTheory.preadditiveYonedaObj J).PreservesEpimorphisms
  -/
  infer_instance
  /-
    🎉 no goals
  -/


