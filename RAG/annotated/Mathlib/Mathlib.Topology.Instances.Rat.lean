instance : MetricSpace ℚ :=
  MetricSpace.induced (↑) Rat.cast_injective Real.metricSpace


theorem dist_eq (x y : ℚ) : dist x y = |(x : ℝ) - y| := rfl


@[norm_cast, simp]
theorem dist_cast (x y : ℚ) : dist (x : ℝ) y = dist x y :=
  rfl


theorem uniformContinuous_coe_real : UniformContinuous ((↑) : ℚ → ℝ) :=
  uniformContinuous_comap


theorem isUniformEmbedding_coe_real : IsUniformEmbedding ((↑) : ℚ → ℝ) :=
  isUniformEmbedding_comap Rat.cast_injective


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_coe_real := isUniformEmbedding_coe_real


theorem isDenseEmbedding_coe_real : IsDenseEmbedding ((↑) : ℚ → ℝ) :=
  isUniformEmbedding_coe_real.isDenseEmbedding Rat.denseRange_cast


@[deprecated (since := "2024-09-30")]
alias denseEmbedding_coe_real := isDenseEmbedding_coe_real


theorem isEmbedding_coe_real : IsEmbedding ((↑) : ℚ → ℝ) :=
  isDenseEmbedding_coe_real.isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_coe_real := isEmbedding_coe_real


theorem continuous_coe_real : Continuous ((↑) : ℚ → ℝ) :=
  uniformContinuous_coe_real.continuous


@[norm_cast, simp]
theorem Nat.dist_cast_rat (x y : ℕ) : dist (x : ℚ) y = dist x y := by
  /-
    x y : Nat
    ⊢ Eq (Dist.dist ↑x ↑y) (Dist.dist x y)
  -/
  rw [← Nat.dist_cast_real, ← Rat.dist_cast]; congr
                                              /-
                                                🎉 no goals
                                              -/


theorem Nat.isUniformEmbedding_coe_rat : IsUniformEmbedding ((↑) : ℕ → ℚ) :=
                                                               /-
                                                                 ⊢ Pairwise fun x y => LE.le 1 (Dist.dist ↑x ↑y)
                                                               -/
  isUniformEmbedding_bot_of_pairwise_le_dist zero_lt_one <| by simpa using Nat.pairwise_one_le_dist
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[deprecated (since := "2024-10-01")]
alias Nat.uniformEmbedding_coe_rat := Nat.isUniformEmbedding_coe_rat


theorem Nat.isClosedEmbedding_coe_rat : IsClosedEmbedding ((↑) : ℕ → ℚ) :=
                                                          /-
                                                            ⊢ Pairwise fun x y => LE.le 1 (Dist.dist ↑x ↑y)
                                                          -/
  isClosedEmbedding_of_pairwise_le_dist zero_lt_one <| by simpa using Nat.pairwise_one_le_dist
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-10-20")]
alias Nat.closedEmbedding_coe_rat := Nat.isClosedEmbedding_coe_rat


@[norm_cast, simp]
theorem Int.dist_cast_rat (x y : ℤ) : dist (x : ℚ) y = dist x y := by
  /-
    x y : Int
    ⊢ Eq (Dist.dist ↑x ↑y) (Dist.dist x y)
  -/
  rw [← Int.dist_cast_real, ← Rat.dist_cast]; congr
                                              /-
                                                🎉 no goals
                                              -/


theorem Int.isUniformEmbedding_coe_rat : IsUniformEmbedding ((↑) : ℤ → ℚ) :=
                                                               /-
                                                                 ⊢ Pairwise fun x y => LE.le 1 (Dist.dist ↑x ↑y)
                                                               -/
  isUniformEmbedding_bot_of_pairwise_le_dist zero_lt_one <| by simpa using Int.pairwise_one_le_dist
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[deprecated (since := "2024-10-01")]
alias Int.uniformEmbedding_coe_rat := Int.isUniformEmbedding_coe_rat


theorem Int.isClosedEmbedding_coe_rat : IsClosedEmbedding ((↑) : ℤ → ℚ) :=
                                                          /-
                                                            ⊢ Pairwise fun x y => LE.le 1 (Dist.dist ↑x ↑y)
                                                          -/
  isClosedEmbedding_of_pairwise_le_dist zero_lt_one <| by simpa using Int.pairwise_one_le_dist
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-10-20")]
alias Int.closedEmbedding_coe_rat := Int.isClosedEmbedding_coe_rat


instance : NoncompactSpace ℚ := Int.isClosedEmbedding_coe_rat.noncompactSpace


theorem uniformContinuous_add : UniformContinuous fun p : ℚ × ℚ => p.1 + p.2 :=
  Rat.isUniformEmbedding_coe_real.isUniformInducing.uniformContinuous_iff.2 <| by
    /-
      ⊢ UniformContinuous (Function.comp Rat.cast fun p => HAdd.hAdd p.1 p.2)
    -/
    simp only [Function.comp_def, Rat.cast_add]
    exact Real.uniformContinuous_add.comp
      (Rat.uniformContinuous_coe_real.prodMap Rat.uniformContinuous_coe_real)


theorem uniformContinuous_neg : UniformContinuous (@Neg.neg ℚ _) :=
  Metric.uniformContinuous_iff.2 fun ε ε0 =>
                            /-
                              ε : Real
                              ε0 : GT.gt ε 0
                              x✝¹ x✝ : Rat
                              h : LT.lt (Dist.dist x✝¹ x✝) ε
                              ⊢ LT.lt (Dist.dist (Neg.neg x✝¹) (Neg.neg x✝)) ε
                            -/
    ⟨_, ε0, fun _ _ h => by simpa only [abs_sub_comm, dist_eq, cast_neg, neg_sub_neg] using h⟩
                            /-
                              🎉 no goals
                            -/


instance : UniformAddGroup ℚ :=
  UniformAddGroup.mk' Rat.uniformContinuous_add Rat.uniformContinuous_neg


instance : TopologicalAddGroup ℚ := inferInstance


instance : OrderTopology ℚ := induced_orderTopology _ Rat.cast_lt exists_rat_btwn


theorem uniformContinuous_abs : UniformContinuous (abs : ℚ → ℚ) :=
  Metric.uniformContinuous_iff.2 fun ε ε0 =>
    ⟨ε, ε0, fun _ _ h =>
                         /-
                           ε : Real
                           ε0 : GT.gt ε 0
                           x✝¹ x✝ : Rat
                           h : LT.lt (Dist.dist x✝¹ x✝) ε
                           ⊢ LE.le (Dist.dist (abs x✝¹) (abs x✝)) (Dist.dist x✝¹ x✝)
                         -/
      lt_of_le_of_lt (by simpa [Rat.dist_eq] using abs_abs_sub_abs_le_abs_sub _ _) h⟩
                         /-
                           🎉 no goals
                         -/


instance : TopologicalRing ℚ := inferInstance


nonrec theorem totallyBounded_Icc (a b : ℚ) : TotallyBounded (Icc a b) := by
  simpa only [preimage_cast_Icc]
    using totallyBounded_preimage Rat.isUniformEmbedding_coe_real.isUniformInducing
      (totallyBounded_Icc (a : ℝ) b)


