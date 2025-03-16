instance instMulZeroClass [MulZeroClass M₀] [MulZeroClass N₀] : MulZeroClass (M₀ × N₀) where
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MulZeroClass M₀
                   inst✝ : MulZeroClass N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MulZeroClass M₀
                   inst✝ : MulZeroClass N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/


instance instSemigroupWithZero [SemigroupWithZero M₀] [SemigroupWithZero N₀] :
    SemigroupWithZero (M₀ × N₀) where
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : SemigroupWithZero M₀
                   inst✝ : SemigroupWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : SemigroupWithZero M₀
                   inst✝ : SemigroupWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/


instance instMulZeroOneClass [MulZeroOneClass M₀] [MulZeroOneClass N₀] :
    MulZeroOneClass (M₀ × N₀) where
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MulZeroOneClass M₀
                   inst✝ : MulZeroOneClass N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MulZeroOneClass M₀
                   inst✝ : MulZeroOneClass N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/


instance instMonoidWithZero [MonoidWithZero M₀] [MonoidWithZero N₀] : MonoidWithZero (M₀ × N₀) where
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MonoidWithZero M₀
                   inst✝ : MonoidWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : MonoidWithZero M₀
                   inst✝ : MonoidWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/


instance instCommMonoidWithZero [CommMonoidWithZero M₀] [CommMonoidWithZero N₀] :
    CommMonoidWithZero (M₀ × N₀) where
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : CommMonoidWithZero M₀
                   inst✝ : CommMonoidWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M₀ : Type u_1
                   N₀ : Type u_2
                   inst✝¹ : CommMonoidWithZero M₀
                   inst✝ : CommMonoidWithZero N₀
                   ⊢ ∀ (a : Prod M₀ N₀), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by simp [Prod.mul_def]
                 /-
                   🎉 no goals
                 -/


/-- Multiplication as a multiplicative homomorphism with zero. -/
@[simps]
def mulMonoidWithZeroHom [CommMonoidWithZero M₀] : M₀ × M₀ →*₀ M₀ where
  __ := mulMonoidHom
  map_zero' := mul_zero _


/-- Division as a multiplicative homomorphism with zero. -/
@[simps]
def divMonoidWithZeroHom [CommGroupWithZero M₀] : M₀ × M₀ →*₀ M₀ where
  __ := divMonoidHom
  map_zero' := zero_div _


