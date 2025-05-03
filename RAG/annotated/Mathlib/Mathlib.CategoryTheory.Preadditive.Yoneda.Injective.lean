theorem injective_iff_preservesEpimorphisms_preadditiveYoneda_obj (J : C) :
    Injective J ↔ (preadditiveYoneda.obj J).PreservesEpimorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    J : C
    ⊢ Iff (CategoryTheory.Injective J) (CategoryTheory.preadditiveYoneda.obj J).Pr …
  -/
  rw [injective_iff_preservesEpimorphisms_yoneda_obj]
  refine
    ⟨fun h : (preadditiveYoneda.obj J ⋙ (forget AddCommGrp)).PreservesEpimorphisms => ?_, ?_⟩
  · exact
      Functor.preservesEpimorphisms_of_preserves_of_reflects (preadditiveYoneda.obj J) (forget _)
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      J : C
      ⊢ (CategoryTheory.preadditiveYoneda.obj J).PreservesEpimorphisms → (CategoryTh …
    -/
  · intro
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      J : C
      a✝ : (CategoryTheory.preadditiveYoneda.obj J).PreservesEpimorphisms
      ⊢ (CategoryTheory.yoneda.obj J).PreservesEpimorphisms
    -/
    exact (inferInstance : (preadditiveYoneda.obj J ⋙ forget _).PreservesEpimorphisms)
    /-
      🎉 no goals
    -/


theorem injective_iff_preservesEpimorphisms_preadditive_yoneda_obj' (J : C) :
    Injective J ↔ (preadditiveYonedaObj J).PreservesEpimorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    J : C
    ⊢ Iff (CategoryTheory.Injective J) (CategoryTheory.preadditiveYonedaObj J).Pre …
  -/
  rw [injective_iff_preservesEpimorphisms_yoneda_obj]
  refine ⟨fun h : (preadditiveYonedaObj J ⋙ (forget <| ModuleCat (End J))).PreservesEpimorphisms =>
    ?_, ?_⟩
  · exact
      Functor.preservesEpimorphisms_of_preserves_of_reflects (preadditiveYonedaObj J) (forget _)
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      J : C
      ⊢ (CategoryTheory.preadditiveYonedaObj J).PreservesEpimorphisms → (CategoryThe …
    -/
  · intro
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      J : C
      a✝ : (CategoryTheory.preadditiveYonedaObj J).PreservesEpimorphisms
      ⊢ (CategoryTheory.yoneda.obj J).PreservesEpimorphisms
    -/
    exact (inferInstance : (preadditiveYonedaObj J ⋙ forget _).PreservesEpimorphisms)
    /-
      🎉 no goals
    -/


