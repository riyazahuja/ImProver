instance instCommRing : CommRing ℤ where
  __ := instAddCommGroup
  __ := instCommSemigroup
  zero_mul := Int.zero_mul
  mul_zero := Int.mul_zero
  left_distrib := Int.mul_add
  right_distrib := Int.add_mul
  mul_one := Int.mul_one
  one_mul := Int.one_mul
  npow n x := x ^ n
  npow_zero _ := rfl
  npow_succ _ _ := rfl
  natCast := (·)
  natCast_zero := rfl
  natCast_succ _ := rfl
  intCast := (·)
  intCast_ofNat _ := rfl
  intCast_negSucc _ := rfl


instance instCancelCommMonoidWithZero : CancelCommMonoidWithZero ℤ where
  mul_left_cancel_of_ne_zero {_a _b _c} ha := (mul_eq_mul_left_iff ha).1


instance instCharZero : CharZero ℤ where cast_injective _ _ := ofNat.inj


instance instMulDivCancelClass : MulDivCancelClass ℤ where mul_div_cancel _ _ := mul_ediv_cancel _


@[simp, norm_cast]
lemma cast_mul {α : Type*} [NonAssocRing α] : ∀ m n, ((m * n : ℤ) : α) = m * n := fun m => by
  /-
    α : Type u_1
    inst✝ : NonAssocRing α
    m : Int
    ⊢ ∀ (n : Int), Eq (↑(HMul.hMul m n)) (HMul.hMul ↑m ↑n)
  -/
  obtain ⟨m, rfl | rfl⟩ := Int.eq_nat_or_neg m
  · induction m with
    | zero => simp
    | succ m ih => simp_all [add_mul]
  · induction m with
    | zero => simp
    | succ m ih => simp_all [add_mul]


@[simp, norm_cast] lemma cast_pow {R : Type*} [Ring R] (n : ℤ) (m : ℕ) :
    ↑(n ^ m) = (n ^ m : R) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Int
    m : Nat
    ⊢ Eq (↑(HPow.hPow n m)) (HPow.hPow (↑n) m)
  -/
                             /-
                               🎉 no goals
                             -/
  induction' m with m ih <;> simp [_root_.pow_succ, *]
                             /-
                               🎉 no goals
                             -/


instance instCommSemiring : CommSemiring ℤ := inferInstance

instance instSemiring     : Semiring ℤ     := inferInstance

instance instRing         : Ring ℤ         := inferInstance

instance instDistrib      : Distrib ℤ      := inferInstance


