instance : SMul R (X ⟶ Y) where
  smul r f := { f := fun n => r • f.f n }


@[simp]
lemma smul_f_apply (r : R) (f : X ⟶ Y) (n : ι) : (r • f).f n = r • f.f n := rfl


@[simp]
lemma units_smul_f_apply (r : Rˣ) (f : X ⟶ Y) (n : ι) : (r • f).f n = r • f.f n := rfl


instance (X Y : HomologicalComplex C c) : Module R (X ⟶ Y) where
                   /-
                     R : Type u_1
                     inst✝⁶ : Semiring R
                     C : Type u_2
                     D : Type u_3
                     inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                     inst✝⁴ : CategoryTheory.Preadditive C
                     inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                     inst✝² : CategoryTheory.Preadditive D
                     inst✝¹ : CategoryTheory.Linear R C
                     inst✝ : CategoryTheory.Linear R D
                     ι : Type u_4
                     c : ComplexShape ι
                     X✝ Y✝ X Y : HomologicalComplex C c
                     a : Quiver.Hom X Y
                     ⊢ Eq (HSMul.hSMul 1 a) a
                   -/
  one_smul a := by aesop_cat
                   /-
                     🎉 no goals
                   -/
                  /-
                    R : Type u_1
                    inst✝⁶ : Semiring R
                    C : Type u_2
                    D : Type u_3
                    inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                    inst✝⁴ : CategoryTheory.Preadditive C
                    inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                    inst✝² : CategoryTheory.Preadditive D
                    inst✝¹ : CategoryTheory.Linear R C
                    inst✝ : CategoryTheory.Linear R D
                    ι : Type u_4
                    c : ComplexShape ι
                    X✝ Y✝ X Y : HomologicalComplex C c
                    ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by aesop_cat
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type u_1
                   inst✝⁶ : Semiring R
                   C : Type u_2
                   D : Type u_3
                   inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                   inst✝⁴ : CategoryTheory.Preadditive C
                   inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                   inst✝² : CategoryTheory.Preadditive D
                   inst✝¹ : CategoryTheory.Linear R C
                   inst✝ : CategoryTheory.Linear R D
                   ι : Type u_4
                   c : ComplexShape ι
                   X✝ Y✝ X Y : HomologicalComplex C c
                   ⊢ ∀ (a : R) (x y : Quiver.Hom X Y), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.h …
                 -/
                       /-
                         R : Type u_1
                         inst✝⁶ : Semiring R
                         C : Type u_2
                         D : Type u_3
                         inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                         inst✝⁴ : CategoryTheory.Preadditive C
                         inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                         inst✝² : CategoryTheory.Preadditive D
                         inst✝¹ : CategoryTheory.Linear R C
                         inst✝ : CategoryTheory.Linear R D
                         ι : Type u_4
                         c : ComplexShape ι
                         X✝ Y✝ X Y : HomologicalComplex C c
                         x✝² x✝¹ : R
                         x✝ : Quiver.Hom X Y
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  smul_add := by aesop_cat
                            /-
                              🎉 no goals
                            -/
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    inst✝⁶ : Semiring R
                    C : Type u_2
                    D : Type u_3
                    inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                    inst✝⁴ : CategoryTheory.Preadditive C
                    inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                    inst✝² : CategoryTheory.Preadditive D
                    inst✝¹ : CategoryTheory.Linear R C
                    inst✝ : CategoryTheory.Linear R D
                    ι : Type u_4
                    c : ComplexShape ι
                    X✝ Y✝ X Y : HomologicalComplex C c
                    ⊢ ∀ (x : Quiver.Hom X Y), Eq (HSMul.hSMul 0 x) 0
                  -/
                       /-
                         R : Type u_1
                         inst✝⁶ : Semiring R
                         C : Type u_2
                         D : Type u_3
                         inst✝⁵ : CategoryTheory.Category.{?u.2877, u_2} C
                         inst✝⁴ : CategoryTheory.Preadditive C
                         inst✝³ : CategoryTheory.Category.{?u.2897, u_3} D
                         inst✝² : CategoryTheory.Preadditive D
                         inst✝¹ : CategoryTheory.Linear R C
                         inst✝ : CategoryTheory.Linear R D
                         ι : Type u_4
                         c : ComplexShape ι
                         X✝ Y✝ X Y : HomologicalComplex C c
                         x✝² x✝¹ : R
                         x✝ : Quiver.Hom X Y
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HSMul.hSMul x✝² x✝) (HSM …
                       -/
  zero_smul := by aesop_cat
                            /-
                              🎉 no goals
                            -/
                  /-
                    🎉 no goals
                  -/
  add_smul _ _ _ := by ext; apply add_smul
  mul_smul _ _ _ := by ext; apply mul_smul


instance : Linear R (HomologicalComplex C c) where


instance CategoryTheory.Functor.mapHomologicalComplex_linear
    (F : C ⥤ D) [F.Additive] [Functor.Linear R F] (c : ComplexShape ι) :
  Functor.Linear R (F.mapHomologicalComplex c) where

