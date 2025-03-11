/-- A notation class for the *positive part* function: `a⁺`. -/
class PosPart (α : Type*) where
  /-- The *positive part* of an element `a`. -/
  posPart : α → α


/-- A notation class for the *positive part* function (multiplicative version): `a⁺ᵐ`. -/
@[to_additive]
class OneLePart (α : Type*) where
  /-- The *positive part* of an element `a`. -/
  oneLePart : α → α


/-- A notation class for the *negative part* function: `a⁻`. -/
class NegPart (α : Type*) where
  /-- The *negative part* of an element `a`. -/
  negPart : α → α


/-- A notation class for the *negative part* function (multiplicative version): `a⁻ᵐ`. -/
@[to_additive]
class LeOnePart (α : Type*) where
  /-- The *negative part* of an element `a`. -/
  leOnePart : α → α


@[inherit_doc] postfix:max "⁺ᵐ " => OneLePart.oneLePart

@[inherit_doc] postfix:max "⁻ᵐ" => LeOnePart.leOnePart

@[inherit_doc] postfix:max "⁺" => PosPart.posPart

@[inherit_doc] postfix:max "⁻" => NegPart.negPart

