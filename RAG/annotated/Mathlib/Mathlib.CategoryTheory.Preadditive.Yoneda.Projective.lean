theorem projective_iff_preservesEpimorphisms_preadditiveCoyoneda_obj (P : C) :
    Projective P ↔ (preadditiveCoyoneda.obj (op P)).PreservesEpimorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    P : C
    ⊢ Iff (CategoryTheory.Projective P) (CategoryTheory.preadditiveCoyoneda.obj {  …
  -/
  rw [projective_iff_preservesEpimorphisms_coyoneda_obj]
  refine ⟨fun h : (preadditiveCoyoneda.obj (op P) ⋙
      forget AddCommGrp).PreservesEpimorphisms => ?_, ?_⟩
  · exact Functor.preservesEpimorphisms_of_preserves_of_reflects (preadditiveCoyoneda.obj (op P))
        (forget _)
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      P : C
      ⊢ (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorphisms …
    -/
  · intro
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      P : C
      a✝ : (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorphi …
      ⊢ (CategoryTheory.coyoneda.obj { unop := P }).PreservesEpimorphisms
    -/
    exact (inferInstance : (preadditiveCoyoneda.obj (op P) ⋙ forget _).PreservesEpimorphisms)
    /-
      🎉 no goals
    -/


theorem projective_iff_preservesEpimorphisms_preadditiveCoyoneda_obj' (P : C) :
    Projective P ↔ (preadditiveCoyoneda.obj (op P)).PreservesEpimorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    P : C
    ⊢ Iff (CategoryTheory.Projective P) (CategoryTheory.preadditiveCoyoneda.obj {  …
  -/
  rw [projective_iff_preservesEpimorphisms_coyoneda_obj]
  refine ⟨fun h : (preadditiveCoyoneda.obj (op P) ⋙
      forget AddCommGrp).PreservesEpimorphisms => ?_, ?_⟩
  · exact Functor.preservesEpimorphisms_of_preserves_of_reflects (preadditiveCoyoneda.obj (op P))
        (forget _)
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      P : C
      ⊢ (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorphisms …
    -/
  · intro
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      P : C
      a✝ : (CategoryTheory.preadditiveCoyoneda.obj { unop := P }).PreservesEpimorphi …
      ⊢ (CategoryTheory.coyoneda.obj { unop := P }).PreservesEpimorphisms
    -/
    exact (inferInstance : (preadditiveCoyoneda.obj (op P) ⋙ forget _).PreservesEpimorphisms)
    /-
      🎉 no goals
    -/


