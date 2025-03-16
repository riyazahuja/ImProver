/-- Notation type class for the set product `×ˢ`. -/
class SProd (α : Type u) (β : Type v) (γ : outParam (Type w)) where
  /-- The cartesian product `s ×ˢ t` is the set of `(a, b)` such that `a ∈ s` and `b ∈ t`. -/
  sprod : α → β → γ

-- This notation binds more strongly than (pre)images, unions and intersections.

@[inherit_doc SProd.sprod] infixr:82 " ×ˢ " => SProd.sprod

macro_rules | `($x ×ˢ $y)   => `(fbinop% SProd.sprod $x $y)

