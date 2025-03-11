/--
The morphisms in `C` between the underlying objects of a pair of bimonoids in `C` naturally has a
(set-theoretic) monoid structure. -/
def Conv (M : Comon_ C) (N : Mon_ C) : Type v₁ := M.X ⟶ N.X


instance : One (Conv M N) where
  one := M.counit ≫ N.one


theorem one_eq : (1 : Conv M N) = M.counit ≫ N.one := rfl


instance : Mul (Conv M N) where
  mul := fun f g => M.comul ≫ f ▷ M.X ≫ N.X ◁ g ≫ N.mul


theorem mul_eq (f g : Conv M N) : f * g = M.comul ≫ f ▷ M.X ≫ N.X ◁ g ≫ N.mul := rfl


instance : Monoid (Conv M N) where
                  /-
                    C : Type u₁
                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝ : CategoryTheory.MonoidalCategory C
                    M : Comon_ C
                    N : Mon_ C
                    f : CategoryTheory.Conv M N
                    ⊢ Eq (HMul.hMul 1 f) f
                  -/
  one_mul f := by simp [one_eq, mul_eq, ← whisker_exchange_assoc]
                  /-
                    🎉 no goals
                  -/
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      M : Comon_ C
      N : Mon_ C
      f g h : CategoryTheory.Conv M N
      ⊢ Eq (HMul.hMul (HMul.hMul f g) h) (HMul.hMul f (HMul.hMul g h))
    -/
                  /-
                    C : Type u₁
                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝ : CategoryTheory.MonoidalCategory C
                    M : Comon_ C
                    N : Mon_ C
                    f : CategoryTheory.Conv M N
                    ⊢ Eq (HMul.hMul f 1) f
                  -/
  mul_one f := by simp [one_eq, mul_eq, ← whisker_exchange_assoc]
                  /-
                    🎉 no goals
                  -/
  mul_assoc f g h := by
    simp only [mul_eq]
    simp only [comp_whiskerRight, whisker_assoc, Category.assoc,
      MonoidalCategory.whiskerLeft_comp]
    slice_lhs 7 8 =>
      rw [← whisker_exchange]
    slice_rhs 2 3 =>
      rw [← whisker_exchange]
    slice_rhs 1 2 =>
      rw [M.comul_assoc]
    slice_rhs 3 4 =>
      rw [← associator_naturality_left]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      M : Comon_ C
      N : Mon_ C
      f g h : CategoryTheory.Conv M N
      ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul (CategoryTheory.CategoryStruc …
    -/
    slice_lhs 6 7 =>
    /-
      🎉 no goals
    -/
      rw [← associator_inv_naturality_right]
    slice_lhs 8 9 =>
      rw [N.mul_assoc]
    simp


