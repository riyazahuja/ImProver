theorem Icc_mem_vitaliFamily_at_right {x y : ℝ} (hxy : x < y) :
    Icc x y ∈ (vitaliFamily (volume : Measure ℝ) 1).setsAt x := by
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily MeasureTheory.Measure …
  -/
  rw [Icc_eq_closedBall]
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily MeasureTheory.Measure …
  -/
  refine closedBall_mem_vitaliFamily_of_dist_le_mul _ ?_ (by linarith)
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ LE.le (Dist.dist x (HDiv.hDiv (HAdd.hAdd x y) 2)) (HMul.hMul 1 (HDiv.hDiv (H …
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rw [dist_comm, Real.dist_eq, abs_of_nonneg] <;> linarith
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem tendsto_Icc_vitaliFamily_right (x : ℝ) :
    Tendsto (fun y => Icc x y) (𝓝[>] x) ((vitaliFamily (volume : Measure ℝ) 1).filterAt x) := by
  /-
    x : Real
    ⊢ Filter.Tendsto (fun y => Set.Icc x y) (nhdsWithin x (Set.Ioi x)) ((IsUnifLoc …
  -/
  refine (VitaliFamily.tendsto_filterAt_iff _).2 ⟨?_, ?_⟩
    /-
      case refine_1
      x : Real
      ⊢ Filter.Eventually (fun i => Membership.mem ((IsUnifLocDoublingMeasure.vitali …
    -/
  · filter_upwards [self_mem_nhdsWithin] with y hy using Icc_mem_vitaliFamily_at_right hy
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x : Real
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun i => HasSubset.Subset (Set. …
    -/
  · intro ε εpos
    /-
      case refine_2
      x ε : Real
      εpos : GT.gt ε 0
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Set.Icc x i) (Metric.closedBal …
    -/
    filter_upwards [Icc_mem_nhdsGT <| show x < x + ε by linarith] with y hy
    /-
      case h
      x ε : Real
      εpos : GT.gt ε 0
      y : Real
      hy : Membership.mem (Set.Icc x (HAdd.hAdd x ε)) y
      ⊢ HasSubset.Subset (Set.Icc x y) (Metric.closedBall x ε)
    -/
    rw [closedBall_eq_Icc]
    /-
      case h
      x ε : Real
      εpos : GT.gt ε 0
      y : Real
      hy : Membership.mem (Set.Icc x (HAdd.hAdd x ε)) y
      ⊢ HasSubset.Subset (Set.Icc x y) (Set.Icc (HSub.hSub x ε) (HAdd.hAdd x ε))
    -/
    exact Icc_subset_Icc (by linarith) hy.2
    /-
      🎉 no goals
    -/


theorem Icc_mem_vitaliFamily_at_left {x y : ℝ} (hxy : x < y) :
    Icc x y ∈ (vitaliFamily (volume : Measure ℝ) 1).setsAt y := by
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily MeasureTheory.Measure …
  -/
  rw [Icc_eq_closedBall]
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily MeasureTheory.Measure …
  -/
  refine closedBall_mem_vitaliFamily_of_dist_le_mul _ ?_ (by linarith)
  /-
    x y : Real
    hxy : LT.lt x y
    ⊢ LE.le (Dist.dist y (HDiv.hDiv (HAdd.hAdd x y) 2)) (HMul.hMul 1 (HDiv.hDiv (H …
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rw [Real.dist_eq, abs_of_nonneg] <;> linarith
                                       /-
                                         🎉 no goals
                                       -/


theorem tendsto_Icc_vitaliFamily_left (x : ℝ) :
    Tendsto (fun y => Icc y x) (𝓝[<] x) ((vitaliFamily (volume : Measure ℝ) 1).filterAt x) := by
  /-
    x : Real
    ⊢ Filter.Tendsto (fun y => Set.Icc y x) (nhdsWithin x (Set.Iio x)) ((IsUnifLoc …
  -/
  refine (VitaliFamily.tendsto_filterAt_iff _).2 ⟨?_, ?_⟩
    /-
      case refine_1
      x : Real
      ⊢ Filter.Eventually (fun i => Membership.mem ((IsUnifLocDoublingMeasure.vitali …
    -/
  · filter_upwards [self_mem_nhdsWithin] with y hy using Icc_mem_vitaliFamily_at_left hy
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x : Real
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun i => HasSubset.Subset (Set. …
    -/
  · intro ε εpos
    /-
      case refine_2
      x ε : Real
      εpos : GT.gt ε 0
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Set.Icc i x) (Metric.closedBal …
    -/
    filter_upwards [Icc_mem_nhdsLT <| show x - ε < x by linarith] with y hy
    /-
      case h
      x ε : Real
      εpos : GT.gt ε 0
      y : Real
      hy : Membership.mem (Set.Icc (HSub.hSub x ε) x) y
      ⊢ HasSubset.Subset (Set.Icc y x) (Metric.closedBall x ε)
    -/
    rw [closedBall_eq_Icc]
    /-
      case h
      x ε : Real
      εpos : GT.gt ε 0
      y : Real
      hy : Membership.mem (Set.Icc (HSub.hSub x ε) x) y
      ⊢ HasSubset.Subset (Set.Icc y x) (Set.Icc (HSub.hSub x ε) (HAdd.hAdd x ε))
    -/
    exact Icc_subset_Icc hy.1 (by linarith)
    /-
      🎉 no goals
    -/


