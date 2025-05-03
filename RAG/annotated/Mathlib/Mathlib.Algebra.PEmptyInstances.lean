@[to_additive]
instance SemigroupPEmpty : Semigroup PEmpty.{u + 1} where
                /-
                  x x✝ : PEmpty
                  ⊢ PEmpty
                -/
  mul x _ := by cases x
                /-
                  🎉 no goals
                -/
                        /-
                          x y z : PEmpty
                          ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
                        -/
  mul_assoc x y z := by cases x
                        /-
                          🎉 no goals
                        -/

