instance instMulZeroClass : MulZeroClass ℕ where
  zero_mul := Nat.zero_mul
  mul_zero := Nat.mul_zero


instance instSemigroupWithZero : SemigroupWithZero ℕ where
  __ := instSemigroup
  __ := instMulZeroClass


instance instMonoidWithZero : MonoidWithZero ℕ where
  __ := instMonoid
  __ := instMulZeroClass
  __ := instSemigroupWithZero


instance instCommMonoidWithZero : CommMonoidWithZero ℕ where
  __ := instCommMonoid
  __ := instMonoidWithZero


instance instIsLeftCancelMulZero : IsLeftCancelMulZero ℕ where
  mul_left_cancel_of_ne_zero h := Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero h)


instance instCancelCommMonoidWithZero : CancelCommMonoidWithZero ℕ where
  __ := instCommMonoidWithZero
  __ := instIsLeftCancelMulZero


instance instMulDivCancelClass : MulDivCancelClass ℕ where
  mul_div_cancel _ _b hb := Nat.mul_div_cancel _ (Nat.pos_iff_ne_zero.2 hb)


instance instMulZeroOneClass : MulZeroOneClass ℕ where
  __ := instMulZeroClass
  __ := instMulOneClass


