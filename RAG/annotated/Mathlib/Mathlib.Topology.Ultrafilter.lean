theorem Ultrafilter.clusterPt_iff {f : Ultrafilter X} : ClusterPt x f ↔ ↑f ≤ 𝓝 x :=
  ⟨f.le_of_inf_neBot', fun h => ClusterPt.of_le_nhds h⟩


theorem clusterPt_iff_ultrafilter {f : Filter X} : ClusterPt x f ↔
    ∃ u : Ultrafilter X, u ≤ f ∧ u ≤ 𝓝 x := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ Iff (ClusterPt x f) (Exists fun u => And (LE.le (↑u) f) (LE.le (↑u) (nhds x)))
  -/
  simp_rw [ClusterPt, ← le_inf_iff, exists_ultrafilter_iff, inf_comm]
  /-
    🎉 no goals
  -/


theorem mapClusterPt_iff_ultrafilter :
    MapClusterPt x F u ↔ ∃ U : Ultrafilter α, U ≤ F ∧ Tendsto u U (𝓝 x) := by
  simp_rw [MapClusterPt, ClusterPt, ← Filter.push_pull', map_neBot_iff, tendsto_iff_comap,
    ← le_inf_iff, exists_ultrafilter_iff, inf_comm]


theorem isOpen_iff_ultrafilter :
    IsOpen s ↔ ∀ x ∈ s, ∀ (l : Ultrafilter X), ↑l ≤ 𝓝 x → s ∈ l := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsOpen s) (∀ (x : X), Membership.mem s x → ∀ (l : Ultrafilter X), LE.le …
  -/
  simp_rw [isOpen_iff_mem_nhds, ← mem_iff_ultrafilter]
  /-
    🎉 no goals
  -/


/-- `x` belongs to the closure of `s` if and only if some ultrafilter
  supported on `s` converges to `x`. -/
theorem mem_closure_iff_ultrafilter :
    x ∈ closure s ↔ ∃ u : Ultrafilter X, s ∈ u ∧ ↑u ≤ 𝓝 x := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Membership.mem (closure s) x) (Exists fun u => And (Membership.mem u s) …
  -/
  simp [closure_eq_cluster_pts, ClusterPt, ← exists_ultrafilter_iff, and_comm]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_ultrafilter : IsClosed s ↔
    ∀ x, ∀ u : Ultrafilter X, ↑u ≤ 𝓝 x → s ∈ u → x ∈ s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsClosed s) (∀ (x : X) (u : Ultrafilter X), LE.le (↑u) (nhds x) → Membe …
  -/
  simp [isClosed_iff_clusterPt, ClusterPt, ← exists_ultrafilter_iff]
  /-
    🎉 no goals
  -/


theorem continuousAt_iff_ultrafilter :
    ContinuousAt f x ↔ ∀ g : Ultrafilter X, ↑g ≤ 𝓝 x → Tendsto f g (𝓝 (f x)) :=
  tendsto_iff_ultrafilter f (𝓝 x) (𝓝 (f x))


theorem continuous_iff_ultrafilter :
    Continuous f ↔ ∀ (x) (g : Ultrafilter X), ↑g ≤ 𝓝 x → Tendsto f g (𝓝 (f x)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (Continuous f) (∀ (x : X) (g : Ultrafilter X), LE.le (↑g) (nhds x) → Fil …
  -/
  simp only [continuous_iff_continuousAt, continuousAt_iff_ultrafilter]
  /-
    🎉 no goals
  -/

