/-- The Frobenius morphism for an adjunction `L ⊣ F` at `A` is given by the morphism

    L(FA ⨯ B) ⟶ LFA ⨯ LB ⟶ A ⨯ LB

natural in `B`, where the first morphism is the product comparison and the latter uses the counit
of the adjunction.

We will show that if `C` and `D` are cartesian closed, then this morphism is an isomorphism for all
`A` iff `F` is a cartesian closed functor, i.e. it preserves exponentials.
-/
def frobeniusMorphism (h : L ⊣ F) (A : C) :
    tensorLeft (F.obj A) ⋙ L ⟶ L ⋙ tensorLeft A :=
  prodComparisonNatTrans L (F.obj A) ≫ whiskerLeft _ ((curriedTensor C).map (h.counit.app _))


/-- If `F` is full and faithful and has a left adjoint `L` which preserves binary products, then the
Frobenius morphism is an isomorphism.
-/
instance frobeniusMorphism_iso_of_preserves_binary_products (h : L ⊣ F) (A : C)
    [Limits.PreservesLimitsOfShape (Discrete Limits.WalkingPair) L] [F.Full] [F.Faithful] :
    IsIso (frobeniusMorphism F h A) :=
  suffices ∀ (X : D), IsIso ((frobeniusMorphism F h A).app X) from NatIso.isIso_of_isIso_app _
             /-
               C : Type u
               inst✝⁶ : CategoryTheory.Category.{v, u} C
               D : Type u'
               inst✝⁵ : CategoryTheory.Category.{v, u'} D
               inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
               inst✝³ : CategoryTheory.ChosenFiniteProducts D
               F : CategoryTheory.Functor C D
               L : CategoryTheory.Functor D C
               h : CategoryTheory.Adjunction L F
               A : C
               inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
               inst✝¹ : F.Full
               inst✝ : F.Faithful
               B : D
               ⊢ CategoryTheory.IsIso ((CategoryTheory.frobeniusMorphism F h A).app B)
             -/
  fun B ↦ by dsimp [frobeniusMorphism]; infer_instance
                                        /-
                                          🎉 no goals
                                        -/


/-- The exponential comparison map.
`F` is a cartesian closed functor if this is an iso for all `A`.
-/
def expComparison (A : C) : exp A ⋙ F ⟶ F ⋙ exp (F.obj A) :=
  mateEquiv (exp.adjunction A) (exp.adjunction (F.obj A)) (prodComparisonNatIso F A).inv


theorem expComparison_ev (A B : C) :
    F.obj A ◁ ((expComparison F A).app B) ≫ (exp.ev (F.obj A)).app (F.obj B) =
      inv (prodComparison F _ _) ≫ F.map ((exp.ev _).app _) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  convert mateEquiv_counit _ _ (prodComparisonNatIso F A).inv B using 2
  /-
    case h.e'_3.h.h.e'_6.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    e_1✝ : Eq (Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (F.obj  …
    e_3✝ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorObj (F.obj A) (F.obj (( …
    e_4✝ : Eq (F.obj (CategoryTheory.MonoidalCategoryStruct.tensorObj A ((Category …
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.ChosenFiniteProducts.prodComparison F …
  -/
  apply IsIso.inv_eq_of_hom_inv_id -- Porting note: was `ext`
  simp only [prodComparisonNatTrans_app, prodComparisonNatIso_inv, asIso_inv, NatIso.isIso_inv_app,
    IsIso.hom_inv_id]


theorem coev_expComparison (A B : C) :
    F.map ((exp.coev A).app B) ≫ (expComparison F A).app (A ⊗ B) =
      (exp.coev _).app (F.obj B) ≫ (exp (F.obj A)).map (inv (prodComparison F A B)) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.exp.coev A).a …
  -/
  convert unit_mateEquiv _ _ (prodComparisonNatIso F A).inv B using 3
  /-
    case h.e'_3.h.h.e'_7.h.h.e'_8.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    e_1✝ : Eq (Quiver.Hom (F.obj ((CategoryTheory.Functor.id C).obj B)) ((F.comp ( …
    e_4✝ : Eq (((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).comp (Categ …
    e_5✝ : Eq ((CategoryTheory.exp (F.obj A)).obj (F.obj (CategoryTheory.MonoidalC …
    e_6✝ : Eq ((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).obj (F.obj B …
    e_7✝ : Eq (F.obj (CategoryTheory.MonoidalCategoryStruct.tensorObj A B)) ((((Ca …
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.ChosenFiniteProducts.prodComparison F …
  -/
  apply IsIso.inv_eq_of_hom_inv_id -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): was `ext`
  /-
    case h.e'_3.h.h.e'_7.h.h.e'_8.h.hom_inv_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    e_1✝ : Eq (Quiver.Hom (F.obj ((CategoryTheory.Functor.id C).obj B)) ((F.comp ( …
    e_4✝ : Eq (((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).comp (Categ …
    e_5✝ : Eq ((CategoryTheory.exp (F.obj A)).obj (F.obj (CategoryTheory.MonoidalC …
    e_6✝ : Eq ((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).obj (F.obj B …
    e_7✝ : Eq (F.obj (CategoryTheory.MonoidalCategoryStruct.tensorObj A B)) ((((Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  dsimp
  /-
    case h.e'_3.h.h.e'_7.h.h.e'_8.h.hom_inv_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    e_1✝ : Eq (Quiver.Hom (F.obj ((CategoryTheory.Functor.id C).obj B)) ((F.comp ( …
    e_4✝ : Eq (((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).comp (Categ …
    e_5✝ : Eq ((CategoryTheory.exp (F.obj A)).obj (F.obj (CategoryTheory.MonoidalC …
    e_6✝ : Eq ((CategoryTheory.MonoidalCategory.tensorLeft (F.obj A)).obj (F.obj B …
    e_7✝ : Eq (F.obj (CategoryTheory.MonoidalCategoryStruct.tensorObj A B)) ((((Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem uncurry_expComparison (A B : C) :
    CartesianClosed.uncurry ((expComparison F A).app B) =
      inv (prodComparison F _ _) ≫ F.map ((exp.ev _).app _) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A B : C
    ⊢ Eq (CategoryTheory.CartesianClosed.uncurry ((CategoryTheory.expComparison F  …
  -/
  rw [uncurry_eq, expComparison_ev]
  /-
    🎉 no goals
  -/


/-- The exponential comparison map is natural in `A`. -/
theorem expComparison_whiskerLeft {A A' : C} (f : A' ⟶ A) :
    expComparison F A ≫ whiskerLeft _ (pre (F.map f)) =
      whiskerRight (pre f) _ ≫ expComparison F A' := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.expComparison F A) (C …
  -/
  unfold expComparison pre
  have vcomp1 := mateEquiv_conjugateEquiv_vcomp
    (exp.adjunction A) (exp.adjunction (F.obj A)) (exp.adjunction (F.obj A'))
    ((prodComparisonNatIso F A).inv) (((curriedTensor D).map (F.map f)))
  have vcomp2 := conjugateEquiv_mateEquiv_vcomp
    (exp.adjunction A) (exp.adjunction A') (exp.adjunction (F.obj A'))
    (((curriedTensor C).map f)) ((prodComparisonNatIso F A').inv)
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.mateEquiv (CategoryT …
  -/
  unfold leftAdjointSquareConjugate.vcomp rightAdjointSquareConjugate.vcomp at vcomp1
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.mateEquiv (CategoryT …
  -/
  unfold leftAdjointConjugateSquare.vcomp rightAdjointConjugateSquare.vcomp at vcomp2
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.mateEquiv (CategoryT …
  -/
  rw [← vcomp1, ← vcomp2]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (CategoryThe …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F ((Categ …
  -/
  ext B
  simp only [Functor.comp_obj, tensorLeft_obj, prodComparisonNatIso_inv, asIso_inv,
    NatTrans.comp_app, whiskerLeft_app, curriedTensor_map_app, NatIso.isIso_inv_app,
    whiskerRight_app, IsIso.eq_inv_comp, prodComparisonNatTrans_app]
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  rw [← prodComparison_inv_natural_whiskerRight F f]
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    A A' : C
    f : Quiver.Hom A' A
    vcomp1 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    vcomp2 : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The functor `F` is cartesian closed (ie preserves exponentials) if each natural transformation
`exp_comparison F A` is an isomorphism
-/
class CartesianClosedFunctor : Prop where
  comparison_iso : ∀ A, IsIso (expComparison F A)


theorem frobeniusMorphism_mate (h : L ⊣ F) (A : C) :
    conjugateEquiv (h.comp (exp.adjunction A)) ((exp.adjunction (F.obj A)).comp h)
        (frobeniusMorphism F h A) =
      expComparison F A := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    ⊢ Eq ((CategoryTheory.conjugateEquiv (h.comp (CategoryTheory.exp.adjunction A) …
  -/
  unfold expComparison frobeniusMorphism
  have conjeq := iterated_mateEquiv_conjugateEquiv h h
    (exp.adjunction (F.obj A)) (exp.adjunction A)
    (prodComparisonNatTrans L (F.obj A) ≫ whiskerLeft L ((curriedTensor C).map (h.counit.app A)))
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq ((CategoryTheory.conjugateEquiv (h.comp (CategoryTheory.exp.adjunction A) …
  -/
  rw [← conjeq]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (CategoryThe …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    ⊢ Eq ((CategoryTheory.mateEquiv h h) (CategoryTheory.CategoryStruct.comp (Cate …
  -/
  ext B
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (((CategoryTheory.mateEquiv h h) (CategoryTheory.CategoryStruct.comp (Cat …
  -/
  unfold mateEquiv
  simp only [Functor.comp_obj, tensorLeft_obj, Functor.id_obj, Equiv.coe_fn_mk, whiskerLeft_comp,
    whiskerLeft_twice, whiskerRight_comp, assoc, NatTrans.comp_app, whiskerLeft_app,
    curriedTensor_obj_obj, whiskerRight_app, prodComparisonNatTrans_app, curriedTensor_map_app,
    Functor.comp_map, tensorLeft_map, prodComparisonNatIso_inv, asIso_inv, NatIso.isIso_inv_app]
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  rw [← F.map_comp, ← F.map_comp]
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  simp only [Functor.map_comp]
  /-
    case h.w.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  apply IsIso.eq_inv_of_inv_hom_id
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc]
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  rw [prodComparison_natural_whiskerLeft, prodComparison_natural_whiskerRight_assoc]
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  slice_lhs 2 3 => rw [← prodComparison_comp]
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  simp only [assoc]
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  unfold prodComparison
  have ηlemma : (h.unit.app (F.obj A ⊗ F.obj B) ≫
    lift ((L ⋙ F).map (fst _ _)) ((L ⋙ F).map (snd _ _))) =
      (h.unit.app (F.obj A)) ⊗ (h.unit.app (F.obj B)) := by
    ext <;> simp
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ηlemma : Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.Mo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.MonoidalC …
  -/
  slice_lhs 1 2 => rw [ηlemma]
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ηlemma : Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.Mo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Functor.id_obj, Functor.comp_obj, assoc, ← whisker_exchange, ← tensorHom_def']
  /-
    case h.w.h.inv_hom_id
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    conjeq : Eq ((CategoryTheory.mateEquiv (CategoryTheory.exp.adjunction A) (Cate …
    B : C
    ηlemma : Eq (CategoryTheory.CategoryStruct.comp (h.unit.app (CategoryTheory.Mo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/--
If the exponential comparison transformation (at `A`) is an isomorphism, then the Frobenius morphism
at `A` is an isomorphism.
-/
theorem frobeniusMorphism_iso_of_expComparison_iso (h : L ⊣ F) (A : C)
    [i : IsIso (expComparison F A)] : IsIso (frobeniusMorphism F h A) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    i : CategoryTheory.IsIso (CategoryTheory.expComparison F A)
    ⊢ CategoryTheory.IsIso (CategoryTheory.frobeniusMorphism F h A)
  -/
  rw [← frobeniusMorphism_mate F h] at i
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    i : CategoryTheory.IsIso ((CategoryTheory.conjugateEquiv (h.comp (CategoryTheo …
    ⊢ CategoryTheory.IsIso (CategoryTheory.frobeniusMorphism F h A)
  -/
  exact @conjugateEquiv_of_iso _ _ _ _ _ _ _ _ _ _ _ i
  /-
    🎉 no goals
  -/


/--
If the Frobenius morphism at `A` is an isomorphism, then the exponential comparison transformation
(at `A`) is an isomorphism.
-/
theorem expComparison_iso_of_frobeniusMorphism_iso (h : L ⊣ F) (A : C)
    [i : IsIso (frobeniusMorphism F h A)] : IsIso (expComparison F A) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v, u'} D
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    L : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.CartesianClosed D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    h : CategoryTheory.Adjunction L F
    A : C
    i : CategoryTheory.IsIso (CategoryTheory.frobeniusMorphism F h A)
    ⊢ CategoryTheory.IsIso (CategoryTheory.expComparison F A)
  -/
  rw [← frobeniusMorphism_mate F h]; infer_instance
                                     /-
                                       🎉 no goals
                                     -/


open Limits in
/-- If `F` is full and faithful, and has a left adjoint which preserves binary products, then it is
cartesian closed.

TODO: Show the converse, that if `F` is cartesian closed and its left adjoint preserves binary
products, then it is full and faithful.
-/
theorem cartesianClosedFunctorOfLeftAdjointPreservesBinaryProducts (h : L ⊣ F) [F.Full] [F.Faithful]
    [PreservesLimitsOfShape (Discrete WalkingPair) L] : CartesianClosedFunctor F where
  comparison_iso _ := expComparison_iso_of_frobeniusMorphism_iso F h _


