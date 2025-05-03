instance : Dist ℤ :=
  ⟨fun x y => dist (x : ℝ) y⟩


theorem dist_eq (x y : ℤ) : dist x y = |(x : ℝ) - y| := rfl


                                                      /-
                                                        m n : Int
                                                        ⊢ Eq (Dist.dist m n) ↑(abs (HSub.hSub m n))
                                                      -/
theorem dist_eq' (m n : ℤ) : dist m n = |m - n| := by rw [dist_eq]; norm_cast
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[norm_cast, simp]
theorem dist_cast_real (x y : ℤ) : dist (x : ℝ) y = dist x y :=
  rfl


theorem pairwise_one_le_dist : Pairwise fun m n : ℤ => 1 ≤ dist m n := by
  /-
    ⊢ Pairwise fun m n => LE.le 1 (Dist.dist m n)
  -/
  intro m n hne
  /-
    m n : Int
    hne : Ne m n
    ⊢ LE.le 1 (Dist.dist m n)
  -/
  rw [dist_eq]; norm_cast; rwa [← zero_add (1 : ℤ), Int.add_one_le_iff, abs_pos, sub_ne_zero]
                           /-
                             🎉 no goals
                           -/


theorem isUniformEmbedding_coe_real : IsUniformEmbedding ((↑) : ℤ → ℝ) :=
  isUniformEmbedding_bot_of_pairwise_le_dist zero_lt_one pairwise_one_le_dist


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_coe_real := isUniformEmbedding_coe_real


theorem isClosedEmbedding_coe_real : IsClosedEmbedding ((↑) : ℤ → ℝ) :=
  isClosedEmbedding_of_pairwise_le_dist zero_lt_one pairwise_one_le_dist


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_coe_real := isClosedEmbedding_coe_real


instance : MetricSpace ℤ := Int.isUniformEmbedding_coe_real.comapMetricSpace _


theorem preimage_ball (x : ℤ) (r : ℝ) : (↑) ⁻¹' ball (x : ℝ) r = ball x r := rfl


theorem preimage_closedBall (x : ℤ) (r : ℝ) : (↑) ⁻¹' closedBall (x : ℝ) r = closedBall x r := rfl


theorem ball_eq_Ioo (x : ℤ) (r : ℝ) : ball x r = Ioo ⌊↑x - r⌋ ⌈↑x + r⌉ := by
  /-
    x : Int
    r : Real
    ⊢ Eq (Metric.ball x r) (Set.Ioo (Int.floor (HSub.hSub (↑x) r)) (Int.ceil (HAdd …
  -/
  rw [← preimage_ball, Real.ball_eq_Ioo, preimage_Ioo]
  /-
    🎉 no goals
  -/


theorem closedBall_eq_Icc (x : ℤ) (r : ℝ) : closedBall x r = Icc ⌈↑x - r⌉ ⌊↑x + r⌋ := by
  /-
    x : Int
    r : Real
    ⊢ Eq (Metric.closedBall x r) (Set.Icc (Int.ceil (HSub.hSub (↑x) r)) (Int.floor …
  -/
  rw [← preimage_closedBall, Real.closedBall_eq_Icc, preimage_Icc]
  /-
    🎉 no goals
  -/


instance : ProperSpace ℤ :=
  ⟨fun x r => by
    /-
      x : Int
      r : Real
      ⊢ IsCompact (Metric.closedBall x r)
    -/
    rw [closedBall_eq_Icc]
    /-
      x : Int
      r : Real
      ⊢ IsCompact (Set.Icc (Int.ceil (HSub.hSub (↑x) r)) (Int.floor (HAdd.hAdd (↑x)  …
    -/
    exact (Set.finite_Icc _ _).isCompact⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cobounded_eq : Bornology.cobounded ℤ = atBot ⊔ atTop := by
  simp_rw [← comap_dist_right_atTop (0 : ℤ), dist_eq', sub_zero,
                                                                /-
                                                                  ⊢ Eq (Filter.comap (fun x => ↑(abs x)) Filter.atTop) (Filter.comap (Function.c …
                                                                -/
    ← comap_abs_atTop, ← @Int.comap_cast_atTop ℝ, comap_comap]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[deprecated (since := "2024-02-07")] alias cocompact_eq := cocompact_eq_atBot_atTop


@[simp]
theorem cofinite_eq : (cofinite : Filter ℤ) = atBot ⊔ atTop := by
  /-
    ⊢ Eq Filter.cofinite (Max.max Filter.atBot Filter.atTop)
  -/
  rw [← cocompact_eq_cofinite, cocompact_eq_atBot_atTop]
  /-
    🎉 no goals
  -/


