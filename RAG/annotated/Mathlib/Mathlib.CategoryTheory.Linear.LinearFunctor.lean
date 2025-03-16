/-- An additive functor `F` is `R`-linear provided `F.map` is an `R`-module morphism. -/
class Functor.Linear {C D : Type*} [Category C] [Category D] [Preadditive C] [Preadditive D]
  [Linear R C] [Linear R D] (F : C ⥤ D) [F.Additive] : Prop where
  /-- the functor induces a linear map on morphisms -/
  map_smul : ∀ {X Y : C} (f : X ⟶ Y) (r : R), F.map (r • f) = r • F.map f := by aesop_cat


@[simp]
theorem map_smul {X Y : C} (r : R) (f : X ⟶ Y) : F.map (r • f) = r • F.map f :=
  Functor.Linear.map_smul _ _


@[simp]
theorem map_units_smul {X Y : C} (r : Rˣ) (f : X ⟶ Y) : F.map (r • f) = r • F.map f := by
  /-
    R : Type u_1
    inst✝⁸ : Semiring R
    C : Type u_2
    D : Type u_3
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} D
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Preadditive D
    inst✝³ : CategoryTheory.Linear R C
    inst✝² : CategoryTheory.Linear R D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Additive
    inst✝ : CategoryTheory.Functor.Linear R F
    X Y : C
    r : Units R
    f : Quiver.Hom X Y
    ⊢ Eq (F.map (HSMul.hSMul r f)) (HSMul.hSMul r (F.map f))
  -/
  apply map_smul
  /-
    🎉 no goals
  -/


instance : Linear R (𝟭 C) where


instance {E : Type*} [Category E] [Preadditive E] [CategoryTheory.Linear R E] (G : D ⥤ E)
    [Additive G] [Linear R G] : Linear R (F ⋙ G) where


/-- `F.mapLinearMap` is an `R`-linear map whose underlying function is `F.map`. -/
@[simps]
def mapLinearMap {X Y : C} : (X ⟶ Y) →ₗ[R] F.obj X ⟶ F.obj Y :=
  { F.mapAddHom with map_smul' := fun r f => F.map_smul r f }


theorem coe_mapLinearMap {X Y : C} : ⇑(F.mapLinearMap R : (X ⟶ Y) →ₗ[R] _) = F.map := rfl


instance inducedFunctorLinear : Functor.Linear R (inducedFunctor F) where


instance fullSubcategoryInclusionLinear {C : Type*} [Category C] [Preadditive C]
    [CategoryTheory.Linear R C] (Z : C → Prop) : (fullSubcategoryInclusion Z).Linear R where


instance natLinear : F.Linear ℕ where
  map_smul := F.mapAddHom.map_nsmul


instance intLinear : F.Linear ℤ where
  map_smul f r := F.mapAddHom.map_zsmul f r


instance ratLinear : F.Linear ℚ where
  map_smul f r := F.mapAddHom.toRatLinearMap.map_smul r f


instance inverseLinear (e : C ≌ D) [e.functor.Additive] [e.functor.Linear R] :
  e.inverse.Linear R where
    map_smul r f := by
      /-
        R : Type u_1
        inst✝⁸ : Semiring R
        C : Type u_2
        D : Type u_3
        inst✝⁷ : CategoryTheory.Category.{u_4, u_2} C
        inst✝⁶ : CategoryTheory.Category.{u_5, u_3} D
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Linear R C
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Linear R D
        e : CategoryTheory.Equivalence C D
        inst✝¹ : e.functor.Additive
        inst✝ : CategoryTheory.Functor.Linear R e.functor
        X✝ Y✝ : D
        r : Quiver.Hom X✝ Y✝
        f : R
        ⊢ Eq (e.inverse.map (HSMul.hSMul f r)) (HSMul.hSMul f (e.inverse.map r))
      -/
      apply e.functor.map_injective
      /-
        case a
        R : Type u_1
        inst✝⁸ : Semiring R
        C : Type u_2
        D : Type u_3
        inst✝⁷ : CategoryTheory.Category.{u_4, u_2} C
        inst✝⁶ : CategoryTheory.Category.{u_5, u_3} D
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Linear R C
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Linear R D
        e : CategoryTheory.Equivalence C D
        inst✝¹ : e.functor.Additive
        inst✝ : CategoryTheory.Functor.Linear R e.functor
        X✝ Y✝ : D
        r : Quiver.Hom X✝ Y✝
        f : R
        ⊢ Eq (e.functor.map (e.inverse.map (HSMul.hSMul f r))) (e.functor.map (HSMul.h …
      -/
      simp
      /-
        🎉 no goals
      -/


