/--
Given two sets `A` and `B`, `A ↓∩ B` denotes the intersection of `A` and `B` as a set in `Set A`.

The notation is short for `((↑) ⁻¹' B : Set A)`, while giving hints to the elaborator
that both `A` and `B` are terms of `Set α` for the same `α`.
This set is the same as `{x : ↑A | ↑x ∈ B}`.
-/
scoped notation3 A:67 " ↓∩ " B:67 => (Subtype.val ⁻¹' (B : type_of% A) : Set (A : Set _))


/-- Coercion using `(Subtype.val '' ·)` -/
instance {α : Type*} {s : Set α} : CoeHead (Set s) (Set α) := ⟨fun t => (Subtype.val '' t)⟩


open Lean PrettyPrinter Delaborator SubExpr in
/--
If the `Set.Notation` namespace is open, sets of a subtype coerced to the ambient type are
represented with `↑`.
-/
@[scoped delab app.Set.image]
def delab_set_image_subtype : Delab := whenPPOption getPPCoercions do
  let #[α, _, f, _] := (← getExpr).getAppArgs | failure
  guard <| f.isAppOfArity ``Subtype.val 2
  let some _ := α.coeTypeSet? | failure
  let e ← withAppArg delab
  `(↑$e)


