/-- The connecting homomorphism
`(singleFunctor C 0).obj S.X₃ ⟶ ((singleFunctor C 0).obj S.X₁)⟦(1 : ℤ)⟧` in the derived
category of `C` when `S` is a short exact short complex in `C`. -/
noncomputable def singleδ : (singleFunctor C 0).obj S.X₃ ⟶
    ((singleFunctor C 0).obj S.X₁)⟦(1 : ℤ)⟧ :=
  (((SingleFunctors.evaluation _ _ 0).mapIso (singleFunctorsPostcompQIso C)).hom.app S.X₃) ≫
    triangleOfSESδ (hS.map_of_exact (HomologicalComplex.single C (ComplexShape.up ℤ) 0)) ≫
    (((SingleFunctors.evaluation _ _ 0).mapIso
      (singleFunctorsPostcompQIso C)).inv.app S.X₁)⟦(1 : ℤ)⟧'


/-- The (distinguished) triangle in the derived category of `C` given by a
short exact short complex in `C`. -/
@[simps!]
noncomputable def singleTriangle : Triangle (DerivedCategory C) :=
  Triangle.mk ((singleFunctor C 0).map S.f)
    ((singleFunctor C 0).map S.g) hS.singleδ


/-- Given a short exact complex `S` in `C` that is short exact (`hS`), this is the
canonical isomorphism between the triangle `hS.singleTriangle` in the derived category
and the triangle attached to the corresponding short exact sequence of cochain complexes
after the application of the single functor. -/
@[simps!]
noncomputable def singleTriangleIso :
    hS.singleTriangle ≅
      triangleOfSES (hS.map_of_exact (HomologicalComplex.single C (ComplexShape.up ℤ) 0)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Iso hS.singleTriangle (DerivedCategory.triangleOfSES ⋯)
  -/
  let e := (SingleFunctors.evaluation _ _ 0).mapIso (singleFunctorsPostcompQIso C)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
    ⊢ CategoryTheory.Iso hS.singleTriangle (DerivedCategory.triangleOfSES ⋯)
  -/
  refine Triangle.isoMk _ _ (e.app S.X₁) (e.app S.X₂) (e.app S.X₃) ?_ ?_ ?_
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp hS.singleTriangle.mor₁ (e.app S.X₂).h …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp hS.singleTriangle.mor₂ (e.app S.X₃).h …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp hS.singleTriangle.mor₃ ((CategoryTheo …
    -/
  · dsimp [singleδ, e]
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc, Category.assoc, ← Functor.map_comp, SingleFunctors.inv_hom_id_hom_app]
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      e : CategoryTheory.Iso ((CategoryTheory.SingleFunctors.evaluation C (DerivedCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((DerivedCategory.singleFunctorsPost …
    -/
    erw [Functor.map_id, comp_id]
    /-
      🎉 no goals
    -/


/-- The distinguished triangle in the derived category of `C` given by a
short exact short complex in `C`. -/
lemma singleTriangle_distinguished :
    hS.singleTriangle ∈ distTriang (DerivedCategory C) :=
  isomorphic_distinguished _ (triangleOfSES_distinguished (hS.map_of_exact
    (HomologicalComplex.single C (ComplexShape.up ℤ) 0))) _ (singleTriangleIso hS)


