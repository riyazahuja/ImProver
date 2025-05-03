/-- An *approximate unit* is a proper filter (i.e., `≠ ⊥`) such that multiplication on the left
(and separately on the right) by `m : α` tends to `𝓝 m` along the filter. -/
structure Filter.IsApproximateUnit {α : Type*} [TopologicalSpace α] [Mul α]
    (l : Filter α) : Prop where
  /-- Multiplication on the left by `m` tends to `𝓝 m` along the filter. -/
  tendsto_mul_left m : Tendsto (m * ·) l (𝓝 m)
  /-- Multiplication on the right by `m` tends to `𝓝 m` along the filter. -/
  tendsto_mul_right m : Tendsto (· * m) l (𝓝 m)
  /-- The filter is not `⊥`. -/
  protected [neBot : NeBot l]


variable (α) in
/-- A unital magma with a topology and bornology has the trivial approximate unit `pure 1`. -/
lemma pure_one : IsApproximateUnit (pure (1 : α))  where
                           /-
                             α : Type u_1
                             inst✝¹ : TopologicalSpace α
                             inst✝ : MulOneClass α
                             m : α
                             ⊢ Filter.Tendsto (fun x => HMul.hMul m x) (Pure.pure 1) (nhds m)
                           -/
  tendsto_mul_left m := by simpa using tendsto_pure_nhds (m * ·) (1 : α)
                           /-
                             🎉 no goals
                           -/
                            /-
                              α : Type u_1
                              inst✝¹ : TopologicalSpace α
                              inst✝ : MulOneClass α
                              m : α
                              ⊢ Filter.Tendsto (fun x => HMul.hMul x m) (Pure.pure 1) (nhds m)
                            -/
  tendsto_mul_right m := by simpa using tendsto_pure_nhds (· * m) (1 : α)
                            /-
                              🎉 no goals
                            -/


set_option linter.unusedVariables false in
/-- If `l` is an approximate unit and `⊥ < l' ≤ l`, then `l'` is also an approximate unit. -/
lemma mono {l l' : Filter α} (hl : l.IsApproximateUnit) (hle : l' ≤ l) [hl' : l'.NeBot] :
    l'.IsApproximateUnit where
  tendsto_mul_left m := hl.tendsto_mul_left m |>.mono_left hle
  tendsto_mul_right m := hl.tendsto_mul_right m |>.mono_left hle


variable (α) in
/-- In a topological unital magma, `𝓝 1` is an approximate unit. -/
lemma nhds_one [ContinuousMul α] : IsApproximateUnit (𝓝 (1 : α)) where
                           /-
                             α : Type u_1
                             inst✝² : TopologicalSpace α
                             inst✝¹ : MulOneClass α
                             inst✝ : ContinuousMul α
                             m : α
                             ⊢ Filter.Tendsto (fun x => HMul.hMul m x) (nhds 1) (nhds m)
                           -/
  tendsto_mul_left m := by simpa using tendsto_id (x := 𝓝 1) |>.const_mul m
                           /-
                             🎉 no goals
                           -/
                            /-
                              α : Type u_1
                              inst✝² : TopologicalSpace α
                              inst✝¹ : MulOneClass α
                              inst✝ : ContinuousMul α
                              m : α
                              ⊢ Filter.Tendsto (fun x => HMul.hMul x m) (nhds 1) (nhds m)
                            -/
  tendsto_mul_right m := by simpa using tendsto_id (x := 𝓝 1) |>.mul_const m
                            /-
                              🎉 no goals
                            -/


/-- In a topological unital magma, `𝓝 1` is the largest approximate unit. -/
lemma iff_neBot_and_le_nhds_one [ContinuousMul α] {l : Filter α} :
    IsApproximateUnit l ↔ l.NeBot ∧ l ≤ 𝓝 1 :=
                          /-
                            α : Type u_1
                            inst✝² : TopologicalSpace α
                            inst✝¹ : MulOneClass α
                            inst✝ : ContinuousMul α
                            l : Filter α
                            hl : l.IsApproximateUnit
                            ⊢ LE.le l (nhds 1)
                          -/
  ⟨fun hl ↦ ⟨hl.neBot, by simpa using hl.tendsto_mul_left 1⟩,
                          /-
                            🎉 no goals
                          -/
    And.elim fun _ hl ↦ nhds_one α |>.mono hl⟩


/-- In a topological unital magma, `𝓝 1` is the largest approximate unit. -/
lemma iff_le_nhds_one [ContinuousMul α] {l : Filter α} [l.NeBot] :
    IsApproximateUnit l ↔ l ≤ 𝓝 1 := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MulOneClass α
    inst✝¹ : ContinuousMul α
    l : Filter α
    inst✝ : l.NeBot
    ⊢ Iff l.IsApproximateUnit (LE.le l (nhds 1))
  -/
  simpa [iff_neBot_and_le_nhds_one] using fun _ ↦ ‹_›
  /-
    🎉 no goals
  -/



