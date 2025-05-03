/-- Left `Mul` by a `k : α` over `[Ring α]` is injective, if `k` is not a zero divisor.
The typeclass that restricts all terms of `α` to have this property is `NoZeroDivisors`. -/
theorem isLeftRegular_of_non_zero_divisor [NonUnitalNonAssocRing α] (k : α)
    (h : ∀ x : α, k * x = 0 → x = 0) : IsLeftRegular k := by
  /-
    α : Type u_1
    inst✝ : NonUnitalNonAssocRing α
    k : α
    h : ∀ (x : α), Eq (HMul.hMul k x) 0 → Eq x 0
    ⊢ IsLeftRegular k
  -/
  refine fun x y (h' : k * x = k * y) => sub_eq_zero.mp (h _ ?_)
  /-
    α : Type u_1
    inst✝ : NonUnitalNonAssocRing α
    k : α
    h : ∀ (x : α), Eq (HMul.hMul k x) 0 → Eq x 0
    x y : α
    h' : Eq (HMul.hMul k x) (HMul.hMul k y)
    ⊢ Eq (HMul.hMul k (HSub.hSub x y)) 0
  -/
  rw [mul_sub, sub_eq_zero, h']
  /-
    🎉 no goals
  -/


/-- Right `Mul` by a `k : α` over `[Ring α]` is injective, if `k` is not a zero divisor.
The typeclass that restricts all terms of `α` to have this property is `NoZeroDivisors`. -/
theorem isRightRegular_of_non_zero_divisor [NonUnitalNonAssocRing α] (k : α)
    (h : ∀ x : α, x * k = 0 → x = 0) : IsRightRegular k := by
  /-
    α : Type u_1
    inst✝ : NonUnitalNonAssocRing α
    k : α
    h : ∀ (x : α), Eq (HMul.hMul x k) 0 → Eq x 0
    ⊢ IsRightRegular k
  -/
  refine fun x y (h' : x * k = y * k) => sub_eq_zero.mp (h _ ?_)
  /-
    α : Type u_1
    inst✝ : NonUnitalNonAssocRing α
    k : α
    h : ∀ (x : α), Eq (HMul.hMul x k) 0 → Eq x 0
    x y : α
    h' : Eq (HMul.hMul x k) (HMul.hMul y k)
    ⊢ Eq (HMul.hMul (HSub.hSub x y) k) 0
  -/
  rw [sub_mul, sub_eq_zero, h']
  /-
    🎉 no goals
  -/


theorem isRegular_of_ne_zero' [NonUnitalNonAssocRing α] [NoZeroDivisors α] {k : α} (hk : k ≠ 0) :
    IsRegular k :=
  ⟨isLeftRegular_of_non_zero_divisor k fun _ h =>
      (NoZeroDivisors.eq_zero_or_eq_zero_of_mul_eq_zero h).resolve_left hk,
    isRightRegular_of_non_zero_divisor k fun _ h =>
      (NoZeroDivisors.eq_zero_or_eq_zero_of_mul_eq_zero h).resolve_right hk⟩


theorem isRegular_iff_ne_zero' [Nontrivial α] [NonUnitalNonAssocRing α] [NoZeroDivisors α]
    {k : α} : IsRegular k ↔ k ≠ 0 :=
  ⟨fun h => by
    /-
      α : Type u_1
      inst✝² : Nontrivial α
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      k : α
      h : IsRegular k
      ⊢ Ne k 0
    -/
    rintro rfl
    /-
      α : Type u_1
      inst✝² : Nontrivial α
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      h : IsRegular 0
      ⊢ False
    -/
    exact not_not.mpr h.left not_isLeftRegular_zero, isRegular_of_ne_zero'⟩
    /-
      🎉 no goals
    -/


/-- A ring with no zero divisors is a `CancelMonoidWithZero`.

Note this is not an instance as it forms a typeclass loop. -/
abbrev NoZeroDivisors.toCancelMonoidWithZero [Ring α] [NoZeroDivisors α] : CancelMonoidWithZero α :=
        /-
          α : Type u_1
          inst✝¹ : Ring α
          inst✝ : NoZeroDivisors α
          ⊢ MonoidWithZero α
        -/
  { (by infer_instance : MonoidWithZero α) with
        /-
          🎉 no goals
        -/
    mul_left_cancel_of_ne_zero := fun ha =>
      @IsRegular.left _ _ _ (isRegular_of_ne_zero' ha) _ _,
    mul_right_cancel_of_ne_zero := fun hb =>
      @IsRegular.right _ _ _ (isRegular_of_ne_zero' hb) _ _ }


/-- A commutative ring with no zero divisors is a `CancelCommMonoidWithZero`.

Note this is not an instance as it forms a typeclass loop. -/
abbrev NoZeroDivisors.toCancelCommMonoidWithZero [CommRing α] [NoZeroDivisors α] :
    CancelCommMonoidWithZero α :=
  { NoZeroDivisors.toCancelMonoidWithZero, ‹CommRing α› with }


instance (priority := 100) IsDomain.toCancelMonoidWithZero [Semiring α] [IsDomain α] :
    CancelMonoidWithZero α :=
  { }


instance (priority := 100) IsDomain.toCancelCommMonoidWithZero : CancelCommMonoidWithZero α :=
  { mul_left_cancel_of_ne_zero := IsLeftCancelMulZero.mul_left_cancel_of_ne_zero }


