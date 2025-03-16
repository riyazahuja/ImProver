instance : BraidedCategory (AlgebraCat.{u} R) :=
  braidedCategoryOfFaithful (forget₂ (AlgebraCat R) (ModuleCat R))
    (fun X Y => (Algebra.TensorProduct.comm R X Y).toAlgebraIso)
        /-
          R : Type u
          inst✝ : CommRing R
          ⊢ ∀ (X Y : AlgebraCat R), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheo …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


instance : (forget₂ (AlgebraCat R) (ModuleCat R)).Braided where


instance instSymmetricCategory : SymmetricCategory (AlgebraCat.{u} R) :=
  symmetricCategoryOfFaithful (forget₂ (AlgebraCat R) (ModuleCat R))


