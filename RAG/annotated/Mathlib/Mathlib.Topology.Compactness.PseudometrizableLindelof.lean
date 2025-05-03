instance SecondCountableTopology.ofPseudoMetrizableSpaceLindelofSpace [PseudoMetrizableSpace X]
    [LindelofSpace X] : SecondCountableTopology X := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : LindelofSpace X
    ⊢ SecondCountableTopology X
  -/
  letI : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  have h_dense (ε) (hpos : 0 < ε) : ∃ s : Set X, s.Countable ∧ ∀ x, ∃ y ∈ s, dist x y ≤ ε := by
    let U := fun (z : X) ↦ Metric.ball z ε
    obtain ⟨t, hct, huniv⟩ := LindelofSpace.elim_nhds_subcover U
      (fun _ ↦ (Metric.isOpen_ball).mem_nhds (Metric.mem_ball_self hpos))
    refine ⟨t, hct, fun z ↦ ?_⟩
    obtain ⟨y, ht, hzy⟩ : ∃ y ∈ t, z ∈ U y :=
      exists_set_mem_of_union_eq_top t (fun i ↦ U i) huniv z
    exact ⟨y, ht, (Metric.mem_ball.mp hzy).le⟩
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : LindelofSpace X
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    h_dense : ∀ (ε : Real), LT.lt 0 ε → Exists fun s => And s.Countable (∀ (x : X) …
    ⊢ SecondCountableTopology X
  -/
  exact Metric.secondCountable_of_almost_dense_set h_dense
  /-
    🎉 no goals
  -/

