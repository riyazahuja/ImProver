instance : BraidedCategory (QuadraticModuleCat.{u} R) :=
  braidedCategoryOfFaithful (forget₂ (QuadraticModuleCat R) (ModuleCat R))
    (fun X Y => ofIso <| tensorComm X.form Y.form)
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : Invertible 2
          ⊢ ∀ (X Y : QuadraticModuleCat R), Eq (CategoryTheory.CategoryStruct.comp (Cate …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- `forget₂ (QuadraticModuleCat R) (ModuleCat R)` is a braided functor. -/
instance : (forget₂ (QuadraticModuleCat R) (ModuleCat R)).Braided where


instance instSymmetricCategory : SymmetricCategory (QuadraticModuleCat.{u} R) :=
  symmetricCategoryOfFaithful (forget₂ (QuadraticModuleCat R) (ModuleCat R))


