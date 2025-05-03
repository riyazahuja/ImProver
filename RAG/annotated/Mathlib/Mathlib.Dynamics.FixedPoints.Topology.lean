/-- If the iterates `f^[n] x` converge to `y` and `f` is continuous at `y`,
then `y` is a fixed point for `f`. -/
theorem isFixedPt_of_tendsto_iterate {x y : α} (hy : Tendsto (fun n => f^[n] x) atTop (𝓝 y))
    (hf : ContinuousAt f y) : IsFixedPt f y := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T2Space α
    f : α → α
    x y : α
    hy : Filter.Tendsto (fun n => Nat.iterate f n x) Filter.atTop (nhds y)
    hf : ContinuousAt f y
    ⊢ Function.IsFixedPt f y
  -/
  refine tendsto_nhds_unique ((tendsto_add_atTop_iff_nat 1).1 ?_) hy
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T2Space α
    f : α → α
    x y : α
    hy : Filter.Tendsto (fun n => Nat.iterate f n x) Filter.atTop (nhds y)
    hf : ContinuousAt f y
    ⊢ Filter.Tendsto (fun n => Nat.iterate f (HAdd.hAdd n 1) x) Filter.atTop (nhds …
  -/
  simp only [iterate_succ' f]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T2Space α
    f : α → α
    x y : α
    hy : Filter.Tendsto (fun n => Nat.iterate f n x) Filter.atTop (nhds y)
    hf : ContinuousAt f y
    ⊢ Filter.Tendsto (fun n => Function.comp f (Nat.iterate f n) x) Filter.atTop ( …
  -/
  exact hf.tendsto.comp hy
  /-
    🎉 no goals
  -/


/-- The set of fixed points of a continuous map is a closed set. -/
theorem isClosed_fixedPoints (hf : Continuous f) : IsClosed (fixedPoints f) :=
  isClosed_eq hf continuous_id

