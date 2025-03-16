/-- Typeclass for expressing that a type `M₀` with multiplication and a zero satisfies
`0 * a = 0` and `a * 0 = 0` for all `a : M₀`. -/
class MulZeroClass (M₀ : Type u) extends Mul M₀, Zero M₀ where
  /-- Zero is a left absorbing element for multiplication -/
  zero_mul : ∀ a : M₀, 0 * a = 0
  /-- Zero is a right absorbing element for multiplication -/
  mul_zero : ∀ a : M₀, a * 0 = 0


/-- A mixin for left cancellative multiplication by nonzero elements. -/
class IsLeftCancelMulZero (M₀ : Type u) [Mul M₀] [Zero M₀] : Prop where
  /-- Multiplication by a nonzero element is left cancellative. -/
  protected mul_left_cancel_of_ne_zero : ∀ {a b c : M₀}, a ≠ 0 → a * b = a * c → b = c


theorem mul_left_cancel₀ (ha : a ≠ 0) (h : a * b = a * c) : b = c :=
  IsLeftCancelMulZero.mul_left_cancel_of_ne_zero ha h


theorem mul_right_injective₀ (ha : a ≠ 0) : Function.Injective (a * ·) :=
  fun _ _ => mul_left_cancel₀ ha


/-- A mixin for right cancellative multiplication by nonzero elements. -/
class IsRightCancelMulZero (M₀ : Type u) [Mul M₀] [Zero M₀] : Prop where
  /-- Multiplicatin by a nonzero element is right cancellative. -/
  protected mul_right_cancel_of_ne_zero : ∀ {a b c : M₀}, b ≠ 0 → a * b = c * b → a = c


theorem mul_right_cancel₀ (hb : b ≠ 0) (h : a * b = c * b) : a = c :=
  IsRightCancelMulZero.mul_right_cancel_of_ne_zero hb h


theorem mul_left_injective₀ (hb : b ≠ 0) : Function.Injective fun a => a * b :=
  fun _ _ => mul_right_cancel₀ hb


/-- A mixin for cancellative multiplication by nonzero elements. -/
class IsCancelMulZero (M₀ : Type u) [Mul M₀] [Zero M₀]
  extends IsLeftCancelMulZero M₀, IsRightCancelMulZero M₀ : Prop


/-- Predicate typeclass for expressing that `a * b = 0` implies `a = 0` or `b = 0`
for all `a` and `b` of type `G₀`. -/
class NoZeroDivisors (M₀ : Type*) [Mul M₀] [Zero M₀] : Prop where
  /-- For all `a` and `b` of `G₀`, `a * b = 0` implies `a = 0` or `b = 0`. -/
  eq_zero_or_eq_zero_of_mul_eq_zero : ∀ {a b : M₀}, a * b = 0 → a = 0 ∨ b = 0


/-- A type `S₀` is a "semigroup with zero” if it is a semigroup with zero element, and `0` is left
and right absorbing. -/
class SemigroupWithZero (S₀ : Type u) extends Semigroup S₀, MulZeroClass S₀


/-- A typeclass for non-associative monoids with zero elements. -/
class MulZeroOneClass (M₀ : Type u) extends MulOneClass M₀, MulZeroClass M₀


/-- A type `M₀` is a “monoid with zero” if it is a monoid with zero element, and `0` is left
and right absorbing. -/
class MonoidWithZero (M₀ : Type u) extends Monoid M₀, MulZeroOneClass M₀, SemigroupWithZero M₀


/-- A type `M` is a `CancelMonoidWithZero` if it is a monoid with zero element, `0` is left
and right absorbing, and left/right multiplication by a non-zero element is injective. -/
class CancelMonoidWithZero (M₀ : Type*) extends MonoidWithZero M₀, IsCancelMulZero M₀


/-- A type `M` is a commutative “monoid with zero” if it is a commutative monoid with zero
element, and `0` is left and right absorbing. -/
class CommMonoidWithZero (M₀ : Type*) extends CommMonoid M₀, MonoidWithZero M₀


theorem mul_left_inj' (hc : c ≠ 0) : a * c = b * c ↔ a = b :=
  (mul_left_injective₀ hc).eq_iff


theorem mul_right_inj' (ha : a ≠ 0) : a * b = a * c ↔ b = c :=
  (mul_right_injective₀ ha).eq_iff


lemma IsLeftCancelMulZero.to_isRightCancelMulZero [IsLeftCancelMulZero M₀] :
    IsRightCancelMulZero M₀ :=
{ mul_right_cancel_of_ne_zero :=
    fun hb h => mul_left_cancel₀ hb <| (mul_comm _ _).trans (h.trans (mul_comm _ _)) }


lemma IsRightCancelMulZero.to_isLeftCancelMulZero [IsRightCancelMulZero M₀] :
    IsLeftCancelMulZero M₀ :=
{ mul_left_cancel_of_ne_zero :=
    fun hb h => mul_right_cancel₀ hb <| (mul_comm _ _).trans (h.trans (mul_comm _ _)) }


lemma IsLeftCancelMulZero.to_isCancelMulZero [IsLeftCancelMulZero M₀] :
    IsCancelMulZero M₀ :=
{ IsLeftCancelMulZero.to_isRightCancelMulZero with }


lemma IsRightCancelMulZero.to_isCancelMulZero [IsRightCancelMulZero M₀] :
    IsCancelMulZero M₀ :=
{ IsRightCancelMulZero.to_isLeftCancelMulZero with }


/-- A type `M` is a `CancelCommMonoidWithZero` if it is a commutative monoid with zero element,
 `0` is left and right absorbing,
  and left/right multiplication by a non-zero element is injective. -/
class CancelCommMonoidWithZero (M₀ : Type*) extends CommMonoidWithZero M₀, IsLeftCancelMulZero M₀

-- See note [lower cancel priority]

instance (priority := 100) CancelCommMonoidWithZero.toCancelMonoidWithZero
    [CancelCommMonoidWithZero M₀] : CancelMonoidWithZero M₀ :=
{ IsLeftCancelMulZero.to_isCancelMulZero (M₀ := M₀) with }


/-- Prop-valued mixin for a monoid with zero to be equipped with a cancelling division.

The obvious use case is groups with zero, but this condition is also satisfied by `ℕ`, `ℤ` and, more
generally, any euclidean domain. -/
class MulDivCancelClass (M₀ : Type*) [MonoidWithZero M₀] [Div M₀] : Prop where
  protected mul_div_cancel (a b : M₀) : b ≠ 0 → a * b / b = a


@[simp] lemma mul_div_cancel_right₀ (a : M₀) {b : M₀} (hb : b ≠ 0) : a * b / b = a :=
  MulDivCancelClass.mul_div_cancel _ _ hb


@[simp] lemma mul_div_cancel_left₀ (b : M₀) {a : M₀} (ha : a ≠ 0) : a * b / a = b := by
  /-
    M₀ : Type u_1
    inst✝² : CommMonoidWithZero M₀
    inst✝¹ : Div M₀
    inst✝ : MulDivCancelClass M₀
    b a : M₀
    ha : Ne a 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) a) b
  -/
  rw [mul_comm, mul_div_cancel_right₀ _ ha]
  /-
    🎉 no goals
  -/


/-- A type `G₀` is a “group with zero” if it is a monoid with zero element (distinct from `1`)
such that every nonzero element is invertible.
The type is required to come with an “inverse” function, and the inverse of `0` must be `0`.

Examples include division rings and the ordered monoids that are the
target of valuations in general valuation theory. -/
class GroupWithZero (G₀ : Type u) extends MonoidWithZero G₀, DivInvMonoid G₀, Nontrivial G₀ where
  /-- The inverse of `0` in a group with zero is `0`. -/
  protected inv_zero : (0 : G₀)⁻¹ = 0
  /-- Every nonzero element of a group with zero is invertible. -/
  protected mul_inv_cancel (a : G₀) : a ≠ 0 → a * a⁻¹ = 1


@[simp] lemma inv_zero : (0 : G₀)⁻¹ = 0 := GroupWithZero.inv_zero


@[simp] lemma mul_inv_cancel₀ (h : a ≠ 0) : a * a⁻¹ = 1 := GroupWithZero.mul_inv_cancel a h

-- See note [lower instance priority]

instance (priority := 100) GroupWithZero.toMulDivCancelClass : MulDivCancelClass G₀ where
                              /-
                                G₀ : Type u
                                M₀ : Type u_1
                                inst✝ : GroupWithZero G₀
                                a✝ a b : G₀
                                hb : Ne b 0
                                ⊢ Eq (HDiv.hDiv (HMul.hMul a b) b) a
                              -/
  mul_div_cancel a b hb := by rw [div_eq_mul_inv, mul_assoc, mul_inv_cancel₀ hb, mul_one]
                              /-
                                🎉 no goals
                              -/


/-- A type `G₀` is a commutative “group with zero”
if it is a commutative monoid with zero element (distinct from `1`)
such that every nonzero element is invertible.
The type is required to come with an “inverse” function, and the inverse of `0` must be `0`. -/
class CommGroupWithZero (G₀ : Type*) extends CommMonoidWithZero G₀, GroupWithZero G₀


lemma eq_zero_or_one_of_sq_eq_self (hx : x ^ 2 = x) : x = 0 ∨ x = 1 :=
  or_iff_not_imp_left.mpr (mul_left_injective₀ · <| by simpa [sq] using hx)


@[simp]
theorem mul_inv_cancel_right₀ (h : b ≠ 0) (a : G₀) : a * b * b⁻¹ = a :=
  calc
    a * b * b⁻¹ = a * (b * b⁻¹) := mul_assoc _ _ _
                /-
                  G₀ : Type u
                  inst✝ : GroupWithZero G₀
                  b : G₀
                  h : Ne b 0
                  a : G₀
                  ⊢ Eq (HMul.hMul a (HMul.hMul b (Inv.inv b))) a
                -/
    _ = a := by simp [h]
                /-
                  🎉 no goals
                -/


@[simp]
theorem mul_inv_cancel_left₀ (h : a ≠ 0) (b : G₀) : a * (a⁻¹ * b) = b :=
  calc
    a * (a⁻¹ * b) = a * a⁻¹ * b := (mul_assoc _ _ _).symm
                /-
                  G₀ : Type u
                  inst✝ : GroupWithZero G₀
                  a : G₀
                  h : Ne a 0
                  b : G₀
                  ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv a)) b) b
                -/
    _ = b := by simp [h]
                /-
                  🎉 no goals
                -/


theorem mul_eq_zero_of_left {a : M₀} (h : a = 0) (b : M₀) : a * b = 0 := h.symm ▸ zero_mul b


theorem mul_eq_zero_of_right (a : M₀) {b : M₀} (h : b = 0) : a * b = 0 := h.symm ▸ mul_zero a


/-- If `α` has no zero divisors, then the product of two elements equals zero iff one of them
equals zero. -/
@[simp]
theorem mul_eq_zero : a * b = 0 ↔ a = 0 ∨ b = 0 :=
  ⟨eq_zero_or_eq_zero_of_mul_eq_zero,
    fun o => o.elim (fun h => mul_eq_zero_of_left h b) (mul_eq_zero_of_right a)⟩


/-- If `α` has no zero divisors, then the product of two elements equals zero iff one of them
equals zero. -/
@[simp]
                                                      /-
                                                        M₀ : Type u_1
                                                        inst✝¹ : MulZeroClass M₀
                                                        inst✝ : NoZeroDivisors M₀
                                                        a b : M₀
                                                        ⊢ Iff (Eq 0 (HMul.hMul a b)) (Or (Eq a 0) (Eq b 0))
                                                      -/
theorem zero_eq_mul : 0 = a * b ↔ a = 0 ∨ b = 0 := by rw [eq_comm, mul_eq_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If `α` has no zero divisors, then the product of two elements is nonzero iff both of them
are nonzero. -/
theorem mul_ne_zero_iff : a * b ≠ 0 ↔ a ≠ 0 ∧ b ≠ 0 := mul_eq_zero.not.trans not_or


/-- If `α` has no zero divisors, then for elements `a, b : α`, `a * b` equals zero iff so is
`b * a`. -/
theorem mul_eq_zero_comm : a * b = 0 ↔ b * a = 0 :=
  mul_eq_zero.trans <| or_comm.trans mul_eq_zero.symm


/-- If `α` has no zero divisors, then for elements `a, b : α`, `a * b` is nonzero iff so is
`b * a`. -/
theorem mul_ne_zero_comm : a * b ≠ 0 ↔ b * a ≠ 0 := mul_eq_zero_comm.not


                                                   /-
                                                     M₀ : Type u_1
                                                     inst✝¹ : MulZeroClass M₀
                                                     inst✝ : NoZeroDivisors M₀
                                                     a : M₀
                                                     ⊢ Iff (Eq (HMul.hMul a a) 0) (Eq a 0)
                                                   -/
theorem mul_self_eq_zero : a * a = 0 ↔ a = 0 := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                   /-
                                                     M₀ : Type u_1
                                                     inst✝¹ : MulZeroClass M₀
                                                     inst✝ : NoZeroDivisors M₀
                                                     a : M₀
                                                     ⊢ Iff (Eq 0 (HMul.hMul a a)) (Eq a 0)
                                                   -/
theorem zero_eq_mul_self : 0 = a * a ↔ a = 0 := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem mul_self_ne_zero : a * a ≠ 0 ↔ a ≠ 0 := mul_self_eq_zero.not


theorem zero_ne_mul_self : 0 ≠ a * a ↔ a ≠ 0 := zero_eq_mul_self.not


                                                                    /-
                                                                      M₀ : Type u_1
                                                                      inst✝¹ : MulZeroClass M₀
                                                                      inst✝ : NoZeroDivisors M₀
                                                                      a b : M₀
                                                                      ha : Ne a 0
                                                                      ⊢ Iff (Eq (HMul.hMul a b) 0) (Eq b 0)
                                                                    -/
theorem mul_eq_zero_iff_left (ha : a ≠ 0) : a * b = 0 ↔ b = 0 := by simp [ha]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                     /-
                                                                       M₀ : Type u_1
                                                                       inst✝¹ : MulZeroClass M₀
                                                                       inst✝ : NoZeroDivisors M₀
                                                                       a b : M₀
                                                                       hb : Ne b 0
                                                                       ⊢ Iff (Eq (HMul.hMul a b) 0) (Eq a 0)
                                                                     -/
theorem mul_eq_zero_iff_right (hb : b ≠ 0) : a * b = 0 ↔ a = 0 := by simp [hb]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                    /-
                                                                      M₀ : Type u_1
                                                                      inst✝¹ : MulZeroClass M₀
                                                                      inst✝ : NoZeroDivisors M₀
                                                                      a b : M₀
                                                                      ha : Ne a 0
                                                                      ⊢ Iff (Ne (HMul.hMul a b) 0) (Ne b 0)
                                                                    -/
theorem mul_ne_zero_iff_left (ha : a ≠ 0) : a * b ≠ 0 ↔ b ≠ 0 := by simp [ha]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                     /-
                                                                       M₀ : Type u_1
                                                                       inst✝¹ : MulZeroClass M₀
                                                                       inst✝ : NoZeroDivisors M₀
                                                                       a b : M₀
                                                                       hb : Ne b 0
                                                                       ⊢ Iff (Ne (HMul.hMul a b) 0) (Ne a 0)
                                                                     -/
theorem mul_ne_zero_iff_right (hb : b ≠ 0) : a * b ≠ 0 ↔ a ≠ 0 := by simp [hb]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


