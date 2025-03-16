/-- A semiring is local if it is nontrivial and `a` or `b` is a unit whenever `a + b = 1`.
Note that `IsLocalRing` is a predicate. -/
class IsLocalRing (R : Type*) [Semiring R] extends Nontrivial R : Prop where
  of_is_unit_or_is_unit_of_add_one ::
  /-- in a local ring `R`, if `a + b = 1`, then either `a` is a unit or `b` is a unit. In another
    word, for every `a : R`, either `a` is a unit or `1 - a` is a unit. -/
  isUnit_or_isUnit_of_add_one {a b : R} (h : a + b = 1) : IsUnit a ∨ IsUnit b


@[deprecated (since := "2024-11-09")] alias LocalRing := IsLocalRing

