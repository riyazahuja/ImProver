variable (S) in
/--
In a commutative semigroup `S`, `Subsemigroup.squareIn S` is the subsemigroup of squares in `S`.
-/
@[to_additive
"In a commutative additive semigroup `S`, the type `AddSubsemigroup.evenIn S`
is the subsemigroup of even elements of `S`."]
def squareIn : Subsemigroup S where
  carrier := {s : S | IsSquare s}
  mul_mem' := IsSquare.mul


@[to_additive (attr := simp)]
theorem mem_squareIn : a ∈ squareIn S ↔ IsSquare a := Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_squareIn : squareIn S = {s : S | IsSquare s} := rfl


variable (M) in
/--
In a commutative monoid `M`, `Submonoid.squareIn M` is the submonoid of squares in `M`.
-/
@[to_additive
"In a commutative additive monoid `M`, the type `AddSubmonoid.evenIn M`
is the submonoid of even elements of `M`."]
def squareIn : Submonoid M where
  __ := Subsemigroup.squareIn M
  one_mem' := IsSquare.one


@[to_additive (attr := simp)]
theorem squareIn_toSubsemigroup : (squareIn M).toSubsemigroup = .squareIn M := rfl


@[to_additive (attr := simp)]
theorem mem_squareIn : a ∈ squareIn M ↔ IsSquare a := Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_squareIn : squareIn M = {s : M | IsSquare s} := rfl


variable (G) in
/--
In an abelian group `G`, `Subgroup.squareIn G` is the subgroup of squares in `G`.
-/
@[to_additive
"In an abelian additive group `G`, the type `AddSubgroup.evenIn G` is
the subgroup of even elements in `G`."]
def squareIn : Subgroup G where
  __ := Submonoid.squareIn G
  inv_mem' := IsSquare.inv


@[to_additive (attr := simp)]
theorem squareIn_toSubmonoid : (squareIn G).toSubmonoid = .squareIn G := rfl


@[to_additive (attr := simp)]
theorem mem_squareIn : a ∈ squareIn G ↔ IsSquare a := Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_squareIn : squareIn G = {s : G | IsSquare s} := rfl


