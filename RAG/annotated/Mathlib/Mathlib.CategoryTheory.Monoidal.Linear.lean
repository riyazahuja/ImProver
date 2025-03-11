/-- A category is `MonoidalLinear R` if tensoring is `R`-linear in both factors.
-/
class MonoidalLinear [MonoidalPreadditive C] : Prop where
  whiskerLeft_smul : ∀ (X : C) {Y Z : C} (r : R) (f : Y ⟶ Z) , X ◁ (r • f) = r • (X ◁ f) := by
    aesop_cat
  smul_whiskerRight : ∀ (r : R) {Y Z : C} (f : Y ⟶ Z) (X : C), (r • f) ▷ X = r • (f ▷ X) := by
    aesop_cat


instance tensorLeft_linear (X : C) : (tensorLeft X).Linear R where


instance tensorRight_linear (X : C) : (tensorRight X).Linear R where


instance tensoringLeft_linear (X : C) : ((tensoringLeft C).obj X).Linear R where


instance tensoringRight_linear (X : C) : ((tensoringRight C).obj X).Linear R where


/-- A faithful linear monoidal functor to a linear monoidal category
ensures that the domain is linear monoidal. -/
theorem monoidalLinearOfFaithful {D : Type*} [Category D] [Preadditive D] [Linear R D]
    [MonoidalCategory D] [MonoidalPreadditive D] (F : D ⥤ C) [F.Monoidal] [F.Faithful]
    [F.Additive] [F.Linear R] : MonoidalLinear R D :=
  { whiskerLeft_smul := by
      /-
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        ⊢ ∀ (X : D) {Y Z : D} (r : R) (f : Quiver.Hom Y Z), Eq (CategoryTheory.Monoida …
      -/
      intros X Y Z r f
      /-
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        X Y Z : D
        r : R
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (HSMul.hSMul r f)) ( …
      -/
      apply F.map_injective
      /-
        case a
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        X Y Z : D
        r : R
        f : Quiver.Hom Y Z
        ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (HSMul.hSMul  …
      -/
      rw [Functor.Monoidal.map_whiskerLeft]
      /-
        case a
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        X Y Z : D
        r : R
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
      -/
      simp
      /-
        🎉 no goals
      -/
    smul_whiskerRight := by
      /-
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        ⊢ ∀ (r : R) {Y Z : D} (f : Quiver.Hom Y Z) (X : D), Eq (CategoryTheory.Monoida …
      -/
      intros r X Y f Z
      /-
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        r : R
        X Y : D
        f : Quiver.Hom X Y
        Z : D
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HSMul.hSMul r f) Z)  …
      -/
      apply F.map_injective
      /-
        case a
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        r : R
        X Y : D
        f : Quiver.Hom X Y
        Z : D
        ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HSMul.hSMul r …
      -/
      rw [Functor.Monoidal.map_whiskerRight]
      /-
        case a
        R : Type u_1
        inst✝¹⁵ : Semiring R
        C : Type u_2
        inst✝¹⁴ : CategoryTheory.Category.{u_5, u_2} C
        inst✝¹³ : CategoryTheory.Preadditive C
        inst✝¹² : CategoryTheory.Linear R C
        inst✝¹¹ : CategoryTheory.MonoidalCategory C
        inst✝¹⁰ : CategoryTheory.MonoidalPreadditive C
        inst✝⁹ : CategoryTheory.MonoidalLinear R C
        D : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_3} D
        inst✝⁷ : CategoryTheory.Preadditive D
        inst✝⁶ : CategoryTheory.Linear R D
        inst✝⁵ : CategoryTheory.MonoidalCategory D
        inst✝⁴ : CategoryTheory.MonoidalPreadditive D
        F : CategoryTheory.Functor D C
        inst✝³ : F.Monoidal
        inst✝² : F.Faithful
        inst✝¹ : F.Additive
        inst✝ : CategoryTheory.Functor.Linear R F
        r : R
        X Y : D
        f : Quiver.Hom X Y
        Z : D
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
      -/
      simp }
      /-
        🎉 no goals
      -/


