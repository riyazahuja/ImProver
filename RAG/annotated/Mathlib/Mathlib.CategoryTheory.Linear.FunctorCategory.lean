instance functorCategoryLinear : Linear R (C ⥤ D) where
  homModule F G :=
    { smul := fun r α =>
        { app := fun X => r • α.app X
          naturality := by
            /-
              R : Type u_1
              inst✝⁴ : Semiring R
              C : Type u_2
              D : Type u_3
              inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
              inst✝² : CategoryTheory.Category.{?u.95, u_3} D
              inst✝¹ : CategoryTheory.Preadditive D
              inst✝ : CategoryTheory.Linear R D
              F G : CategoryTheory.Functor C D
              r : R
              α : Quiver.Hom F G
              ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
            -/
            intros
            /-
              R : Type u_1
              inst✝⁴ : Semiring R
              C : Type u_2
              D : Type u_3
              inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
              inst✝² : CategoryTheory.Category.{?u.95, u_3} D
              inst✝¹ : CategoryTheory.Preadditive D
              inst✝ : CategoryTheory.Linear R D
              F G : CategoryTheory.Functor C D
              r : R
              α : Quiver.Hom F G
              X✝ Y✝ : C
              f✝ : Quiver.Hom X✝ Y✝
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) ((fun X => HSMul.hSMul r ( …
            -/
            rw [comp_smul, smul_comp, α.naturality] }
            /-
              🎉 no goals
            -/
      one_smul := by
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (b : Quiver.Hom F G), Eq (HSMul.hSMul 1 b) b
        -/
        intros
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          b✝ : Quiver.Hom F G
          ⊢ Eq (HSMul.hSMul 1 b✝) b✝
        -/
        ext
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          b✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSMul.hSMul 1 b✝).app x✝) (b✝.app x✝)
        -/
        apply one_smul
        /-
          🎉 no goals
        -/
      zero_smul := by
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (x : Quiver.Hom F G), Eq (HSMul.hSMul 0 x) 0
        -/
        intros
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          x✝ : Quiver.Hom F G
          ⊢ Eq (HSMul.hSMul 0 x✝) 0
        -/
        ext
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
        -/
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          x✝¹ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSMul.hSMul 0 x✝¹).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
        -/
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          a✝ : R
          ⊢ Eq (HSMul.hSMul a✝ 0) 0
        -/
        apply zero_smul
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          a✝ : R
          x✝ : C
          ⊢ Eq ((HSMul.hSMul a✝ 0).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      smul_zero := by
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (r s : R) (x : Quiver.Hom F G), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.h …
        -/
        intros
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          r✝ s✝ : R
          x✝ : Quiver.Hom F G
          ⊢ Eq (HSMul.hSMul (HAdd.hAdd r✝ s✝) x✝) (HAdd.hAdd (HSMul.hSMul r✝ x✝) (HSMul. …
        -/
        ext
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (x y : R) (b : Quiver.Hom F G), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul. …
        -/
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a : R) (x y : Quiver.Hom F G), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.h …
        -/
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          x✝ y✝ : R
          b✝ : Quiver.Hom F G
          ⊢ Eq (HSMul.hSMul (HMul.hMul x✝ y✝) b✝) (HSMul.hSMul x✝ (HSMul.hSMul y✝ b✝))
        -/
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          r✝ s✝ : R
          x✝¹ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSMul.hSMul (HAdd.hAdd r✝ s✝) x✝¹).app x✝) ((HAdd.hAdd (HSMul.hSMul r✝  …
        -/
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          x✝¹ y✝ : R
          b✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSMul.hSMul (HMul.hMul x✝¹ y✝) b✝).app x✝) ((HSMul.hSMul x✝¹ (HSMul.hSM …
        -/
        /-
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          a✝ : R
          x✝ y✝ : Quiver.Hom F G
          ⊢ Eq (HSMul.hSMul a✝ (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (HSMul.hSMul a✝ x✝) (HSMul. …
        -/
        /-
          🎉 no goals
        -/
        apply smul_zero
        /-
          case w.h
          R : Type u_1
          inst✝⁴ : Semiring R
          C : Type u_2
          D : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
          inst✝² : CategoryTheory.Category.{?u.95, u_3} D
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : CategoryTheory.Linear R D
          F G : CategoryTheory.Functor C D
          a✝ : R
          x✝¹ y✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSMul.hSMul a✝ (HAdd.hAdd x✝¹ y✝)).app x✝) ((HAdd.hAdd (HSMul.hSMul a✝  …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      add_smul := by
        intros
        ext
        apply add_smul
      smul_add := by
        intros
        ext
        apply smul_add
      mul_smul := by
        intros
        ext
        apply mul_smul }
  smul_comp := by
    /-
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      ⊢ ∀ (X Y Z : CategoryTheory.Functor C D) (r : R) (f : Quiver.Hom X Y) (g : Qui …
    -/
    intros
    /-
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      X✝ Y✝ Z✝ : CategoryTheory.Functor C D
      r✝ : R
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r✝ f✝) g✝) (HSMul.hSMul  …
    -/
    ext
    /-
      case w.h
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      X✝ Y✝ Z✝ : CategoryTheory.Functor C D
      r✝ : R
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (HSMul.hSMul r✝ f✝) g✝).app x✝) ((HS …
    -/
    apply smul_comp
    /-
      🎉 no goals
    -/
  comp_smul := by
    /-
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      ⊢ ∀ (X Y Z : CategoryTheory.Functor C D) (f : Quiver.Hom X Y) (r : R) (g : Qui …
    -/
    intros
    /-
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      X✝ Y✝ Z✝ : CategoryTheory.Functor C D
      f✝ : Quiver.Hom X✝ Y✝
      r✝ : R
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HSMul.hSMul r✝ g✝)) (HSMul.hSMul  …
    -/
    ext
    /-
      case w.h
      R : Type u_1
      inst✝⁴ : Semiring R
      C : Type u_2
      D : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.91, u_2} C
      inst✝² : CategoryTheory.Category.{?u.95, u_3} D
      inst✝¹ : CategoryTheory.Preadditive D
      inst✝ : CategoryTheory.Linear R D
      X✝ Y✝ Z✝ : CategoryTheory.Functor C D
      f✝ : Quiver.Hom X✝ Y✝
      r✝ : R
      g✝ : Quiver.Hom Y✝ Z✝
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f✝ (HSMul.hSMul r✝ g✝)).app x✝) ((HS …
    -/
    apply comp_smul
    /-
      🎉 no goals
    -/


/-- Application of a natural transformation at a fixed object,
as group homomorphism -/
@[simps]
def appLinearMap (X : C) : (F ⟶ G) →ₗ[R] F.obj X ⟶ G.obj X where
  toFun α := α.app X
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem app_smul (X : C) (r : R) (α : F ⟶ G) : (r • α).app X = r • α.app X :=
  rfl


