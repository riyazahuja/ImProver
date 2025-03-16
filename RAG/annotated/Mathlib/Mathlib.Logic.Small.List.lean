instance smallVector {α : Type v} {n : ℕ} [Small.{u} α] : Small.{u} (List.Vector α n) :=
  small_of_injective (Equiv.vectorEquivFin α n).injective


instance smallList {α : Type v} [Small.{u} α] : Small.{u} (List α) := by
  /-
    α : Type v
    inst✝ : Small.{u, v} α
    ⊢ Small.{u, v} (List α)
  -/
  let e : (Σn, List.Vector α n) ≃ List α := Equiv.sigmaFiberEquiv List.length
  /-
    α : Type v
    inst✝ : Small.{u, v} α
    e : Equiv (Sigma fun n => List.Vector α n) (List α) := Equiv.sigmaFiberEquiv L …
    ⊢ Small.{u, v} (List α)
  -/
  exact small_of_surjective e.surjective
  /-
    🎉 no goals
  -/

