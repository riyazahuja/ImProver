instance : MetricSpace ℕ+ := inferInstanceAs (MetricSpace { n : ℕ // 0 < n })


theorem dist_eq (x y : ℕ+) : dist x y = |(↑x : ℝ) - ↑y| := rfl


@[simp, norm_cast]
theorem dist_coe (x y : ℕ+) : dist (↑x : ℕ) (↑y : ℕ) = dist x y := rfl


theorem isUniformEmbedding_coe : IsUniformEmbedding ((↑) : ℕ+ → ℕ) := isUniformEmbedding_subtype_val


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_coe := isUniformEmbedding_coe


instance : DiscreteTopology ℕ+ := inferInstanceAs (DiscreteTopology { n : ℕ // 0 < n })


instance : ProperSpace ℕ+ where
  isCompact_closedBall n r := by
    /-
      n : PNat
      r : Real
      ⊢ IsCompact (Metric.closedBall n r)
    -/
    change IsCompact (((↑) : ℕ+ → ℕ) ⁻¹' closedBall (↑n : ℕ) r)
    /-
      n : PNat
      r : Real
      ⊢ IsCompact (Set.preimage PNat.val (Metric.closedBall (↑n) r))
    -/
    rw [Nat.closedBall_eq_Icc]
    /-
      n : PNat
      r : Real
      ⊢ IsCompact (Set.preimage PNat.val (Set.Icc (Nat.ceil (HSub.hSub (↑↑n) r)) (Na …
    -/
    exact ((Set.finite_Icc _ _).preimage PNat.coe_injective.injOn).isCompact
    /-
      🎉 no goals
    -/


instance : NoncompactSpace ℕ+ :=
                                 /-
                                   ⊢ (Filter.cocompact PNat).NeBot
                                 -/
  noncompactSpace_of_neBot <| by simp only [Filter.cocompact_eq_cofinite, Filter.cofinite_neBot]
                                 /-
                                   🎉 no goals
                                 -/


