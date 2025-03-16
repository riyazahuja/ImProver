/--
A class expressing a universe inequality. `UnivLE.{u, v}` expresses that `u ≤ v`.

There used to be a stronger definition `∀ α : Type max u v, Small.{v} α` that immediately implies
`Small.{v} ((α : Type u) → (β : Type v))` which is essential for proving that `Type v` has
`Type u`-indexed limits when `u ≤ v`. However the current weaker condition
`∀ α : Type u, Small.{v} α` also implies the same, so we switched to use it for
its simplicity and transitivity.

The strong definition easily implies the weaker definition (see below),
but we can not prove the reverse implication.
This is because in Lean's type theory, while `max u v` is at least at big as `u` and `v`,
it could be bigger than both!
See also `Mathlib.CategoryTheory.UnivLE` for the statement that the stronger definition is
equivalent to `EssSurj (uliftFunctor : Type v ⥤ Type max u v)`.
-/
@[pp_with_univ]
abbrev UnivLE : Prop := ∀ α : Type u, Small.{v} α


theorem univLE_max : UnivLE.{u, max u v} := fun α ↦ small_max.{v} α


theorem Small.trans_univLE (α : Type w) [hα : Small.{u} α] [h : UnivLE.{u, v}] :
    Small.{v} α :=
  let ⟨β, ⟨f⟩⟩ := hα.equiv_small
  let ⟨_, ⟨g⟩⟩ := (h β).equiv_small
  ⟨_, ⟨f.trans g⟩⟩


theorem UnivLE.trans [UnivLE.{u, v}] [UnivLE.{v, w}] : UnivLE.{u, w} :=
  fun α ↦ Small.trans_univLE α

/- This is the crucial instance that subsumes `univLE_max`. -/

instance univLE_of_max [UnivLE.{max u v, v}] : UnivLE.{u, v} := @UnivLE.trans univLE_max ‹_›

/- When `small_Pi` from `Mathlib.Logic.Small.Basic` is imported, we have : -/
-- example (α : Type u) (β : Type v) [UnivLE.{u, v}] : Small.{v} (α → β) := inferInstance


