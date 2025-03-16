/-- A canonically linear ordered field is a linear ordered field in which `a ≤ b` iff there exists
`c` with `b = a + c`. -/
class CanonicallyLinearOrderedSemifield (α : Type*) extends CanonicallyOrderedCommSemiring α,
  LinearOrderedSemifield α

-- See note [lower instance priority]

instance (priority := 100) CanonicallyLinearOrderedSemifield.toLinearOrderedCommGroupWithZero
    [CanonicallyLinearOrderedSemifield α] : LinearOrderedCommGroupWithZero α :=
  { ‹CanonicallyLinearOrderedSemifield α› with
    mul_le_mul_left := fun _ _ h _ ↦ mul_le_mul_of_nonneg_left h <| zero_le _ }

-- See note [lower instance priority]

instance (priority := 100) CanonicallyLinearOrderedSemifield.toCanonicallyLinearOrderedAddCommMonoid
    [CanonicallyLinearOrderedSemifield α] : CanonicallyLinearOrderedAddCommMonoid α :=
  { ‹CanonicallyLinearOrderedSemifield α› with }

