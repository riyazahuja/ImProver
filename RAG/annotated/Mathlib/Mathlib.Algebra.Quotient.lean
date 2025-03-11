/-- `HasQuotient A B` is a notation typeclass that allows us to write `A ⧸ b` for `b : B`.
This allows the usual notation for quotients of algebraic structures,
such as groups, modules and rings.

`A` is a parameter, despite being unused in the definition below, so it appears in the notation.
-/
class HasQuotient (A : outParam <| Type u) (B : Type v) where
  /-- auxiliary quotient function, the one used will have `A` explicit -/
  quotient' : B → Type max u v

-- Will be provided by e.g. `Ideal.Quotient.inhabited`

/-- `HasQuotient.Quotient A b` (with notation `A ⧸ b`) is the quotient
 of the type `A` by `b`.

This differs from `HasQuotient.quotient'` in that the `A` argument is
 explicit, which is necessary to make Lean show the notation in the
 goal state.
-/
abbrev HasQuotient.Quotient (A : outParam <| Type u) {B : Type v}
    [HasQuotient A B] (b : B) : Type max u v :=
  HasQuotient.quotient' b


/-- Quotient notation based on the `HasQuotient` typeclass -/
notation:35 G " ⧸ " H:34 => HasQuotient.Quotient G H

