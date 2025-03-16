/-- A σ-compact Hausdorff topological manifold over a finite dimensional real vector space is
metrizable. -/
theorem Manifold.metrizableSpace {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {H : Type*} [TopologicalSpace H] (I : ModelWithCorners ℝ E H)
    (M : Type*) [TopologicalSpace M] [ChartedSpace H M] [SigmaCompactSpace M] [T2Space M] :
    MetrizableSpace M := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    ⊢ TopologicalSpace.MetrizableSpace M
  -/
  haveI := I.locallyCompactSpace; haveI := ChartedSpace.locallyCompactSpace H M
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    ⊢ TopologicalSpace.MetrizableSpace M
  -/
  haveI := I.secondCountableTopology
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    this✝¹ : LocallyCompactSpace H
    this✝ : LocallyCompactSpace M
    this : SecondCountableTopology H
    ⊢ TopologicalSpace.MetrizableSpace M
  -/
  haveI := ChartedSpace.secondCountable_of_sigmaCompact H M
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    this✝² : LocallyCompactSpace H
    this✝¹ : LocallyCompactSpace M
    this✝ : SecondCountableTopology H
    this : SecondCountableTopology M
    ⊢ TopologicalSpace.MetrizableSpace M
  -/
  exact metrizableSpace_of_t3_secondCountable M
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-11-11")] alias ManifoldWithCorners.metrizableSpace :=
  Manifold.metrizableSpace

