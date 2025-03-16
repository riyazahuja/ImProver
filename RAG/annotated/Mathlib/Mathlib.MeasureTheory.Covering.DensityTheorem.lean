/-- A Vitali family in a space with a uniformly locally doubling measure, designed so that the sets
at `x` contain all `closedBall y r` when `dist x y ≤ K * r`. -/
irreducible_def vitaliFamily (K : ℝ) : VitaliFamily μ := by
  /- the Vitali covering theorem gives a family that works well at small scales, thanks to the
    doubling property. We enlarge this family to add large sets, to make sure that all balls and not
    only small ones belong to the family, for convenience. -/
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    ⊢ VitaliFamily μ
  -/
  let R := scalingScaleOf μ (max (4 * K + 3) 3)
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
    ⊢ VitaliFamily μ
  -/
  have Rpos : 0 < R := scalingScaleOf_pos _ _
  have A : ∀ x : α, ∃ᶠ r in 𝓝[>] (0 : ℝ),
      μ (closedBall x (3 * r)) ≤ scalingConstantOf μ (max (4 * K + 3) 3) * μ (closedBall x r) := by
    intro x
    apply frequently_iff.2 fun {U} hU => ?_
    obtain ⟨ε, εpos, hε⟩ := mem_nhdsGT_iff_exists_Ioc_subset.1 hU
    refine ⟨min ε R, hε ⟨lt_min εpos Rpos, min_le_left _ _⟩, ?_⟩
    exact measure_mul_le_scalingConstantOf_mul μ
      ⟨zero_lt_three, le_max_right _ _⟩ (min_le_right _ _)
  exact (Vitali.vitaliFamily μ (scalingConstantOf μ (max (4 * K + 3) 3)) A).enlarge (R / 4)
    (by linarith)


/-- In the Vitali family `IsUnifLocDoublingMeasure.vitaliFamily K`, the sets based at `x`
contain all balls `closedBall y r` when `dist x y ≤ K * r`. -/
theorem closedBall_mem_vitaliFamily_of_dist_le_mul {K : ℝ} {x y : α} {r : ℝ} (h : dist x y ≤ K * r)
    (rpos : 0 < r) : closedBall y r ∈ (vitaliFamily μ K).setsAt x := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x y : α
    r : Real
    h : LE.le (Dist.dist x y) (HMul.hMul K r)
    rpos : LT.lt 0 r
    ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily μ K).setsAt x) (Metri …
  -/
  let R := scalingScaleOf μ (max (4 * K + 3) 3)
  simp only [vitaliFamily, VitaliFamily.enlarge, Vitali.vitaliFamily, mem_union, mem_setOf_eq,
    isClosed_ball, true_and, (nonempty_ball.2 rpos).mono ball_subset_interior_closedBall,
    measurableSet_closedBall]
  /- The measure is doubling on scales smaller than `R`. Therefore, we treat differently small
    and large balls. For large balls, this follows directly from the enlargement we used in the
    definition. -/
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x y : α
    r : Real
    h : LE.le (Dist.dist x y) (HMul.hMul K r)
    rpos : LT.lt 0 r
    R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
    ⊢ Or (Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric. …
  -/
  by_cases H : closedBall y r ⊆ closedBall x (R / 4)
  /-
    case pos
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x y : α
    r : Real
    h : LE.le (Dist.dist x y) (HMul.hMul K r)
    rpos : LT.lt 0 r
    R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
    H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
    ⊢ Or (Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric. …
  -/
  swap; · exact Or.inr H
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x y : α
    r : Real
    h : LE.le (Dist.dist x y) (HMul.hMul K r)
    rpos : LT.lt 0 r
    R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
    H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
    ⊢ Or (Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric. …
  -/
  left
  /- For small balls, there is the difficulty that `r` could be large but still the ball could be
    small, if the annulus `{y | ε ≤ dist y x ≤ R/4}` is empty. We split between the cases `r ≤ R`
    and `r > R`, and use the doubling for the former and rough estimates for the latter. -/
  /-
    case pos.h
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x y : α
    r : Real
    h : LE.le (Dist.dist x y) (HMul.hMul K r)
    rpos : LT.lt 0 r
    R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
    H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
    ⊢ Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric.clos …
  -/
  rcases le_or_lt r R with (hr | hr)
    /-
      case pos.h.inl
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LE.le r R
      ⊢ Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric.clos …
    -/
  · refine ⟨(K + 1) * r, ?_⟩
    /-
      case pos.h.inl
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LE.le r R
      ⊢ And (HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HMul.hMu …
    -/
    constructor
      /-
        case pos.h.inl.left
        α : Type u_1
        inst✝⁵ : PseudoMetricSpace α
        inst✝⁴ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : IsUnifLocDoublingMeasure μ
        inst✝² : SecondCountableTopology α
        inst✝¹ : BorelSpace α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        K : Real
        x y : α
        r : Real
        h : LE.le (Dist.dist x y) (HMul.hMul K r)
        rpos : LT.lt 0 r
        R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
        H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
        hr : LE.le r R
        ⊢ HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HMul.hMul (HA …
      -/
    · apply closedBall_subset_closedBall'
      /-
        case pos.h.inl.left.h
        α : Type u_1
        inst✝⁵ : PseudoMetricSpace α
        inst✝⁴ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : IsUnifLocDoublingMeasure μ
        inst✝² : SecondCountableTopology α
        inst✝¹ : BorelSpace α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        K : Real
        x y : α
        r : Real
        h : LE.le (Dist.dist x y) (HMul.hMul K r)
        rpos : LT.lt 0 r
        R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
        H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
        hr : LE.le r R
        ⊢ LE.le (HAdd.hAdd r (Dist.dist y x)) (HMul.hMul (HAdd.hAdd K 1) r)
      -/
      rw [dist_comm]
      /-
        case pos.h.inl.left.h
        α : Type u_1
        inst✝⁵ : PseudoMetricSpace α
        inst✝⁴ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : IsUnifLocDoublingMeasure μ
        inst✝² : SecondCountableTopology α
        inst✝¹ : BorelSpace α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        K : Real
        x y : α
        r : Real
        h : LE.le (Dist.dist x y) (HMul.hMul K r)
        rpos : LT.lt 0 r
        R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
        H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
        hr : LE.le r R
        ⊢ LE.le (HAdd.hAdd r (Dist.dist x y)) (HMul.hMul (HAdd.hAdd K 1) r)
      -/
      linarith
      /-
        🎉 no goals
      -/
    · have I1 : closedBall x (3 * ((K + 1) * r)) ⊆ closedBall y ((4 * K + 3) * r) := by
        apply closedBall_subset_closedBall'
        linarith
      have I2 : closedBall y ((4 * K + 3) * r) ⊆ closedBall y (max (4 * K + 3) 3 * r) := by
        apply closedBall_subset_closedBall
        exact mul_le_mul_of_nonneg_right (le_max_left _ _) rpos.le
      /-
        case pos.h.inl.right
        α : Type u_1
        inst✝⁵ : PseudoMetricSpace α
        inst✝⁴ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : IsUnifLocDoublingMeasure μ
        inst✝² : SecondCountableTopology α
        inst✝¹ : BorelSpace α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        K : Real
        x y : α
        r : Real
        h : LE.le (Dist.dist x y) (HMul.hMul K r)
        rpos : LT.lt 0 r
        R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
        H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
        hr : LE.le r R
        I1 : HasSubset.Subset (Metric.closedBall x (HMul.hMul 3 (HMul.hMul (HAdd.hAdd  …
        I2 : HasSubset.Subset (Metric.closedBall y (HMul.hMul (HAdd.hAdd (HMul.hMul 4  …
        ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul 3 (HMul.hMul (HAdd.hAdd K 1) r))))  …
      -/
      apply (measure_mono (I1.trans I2)).trans
      exact measure_mul_le_scalingConstantOf_mul _
        ⟨zero_lt_three.trans_le (le_max_right _ _), le_rfl⟩ hr
    /-
      case pos.h.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LT.lt R r
      ⊢ Exists fun r_1 => And (HasSubset.Subset (Metric.closedBall y r) (Metric.clos …
    -/
  · refine ⟨R / 4, H, ?_⟩
    have : closedBall x (3 * (R / 4)) ⊆ closedBall y r := by
      apply closedBall_subset_closedBall'
      have A : y ∈ closedBall y r := mem_closedBall_self rpos.le
      have B := mem_closedBall'.1 (H A)
      linarith
    /-
      case pos.h.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LT.lt R r
      this : HasSubset.Subset (Metric.closedBall x (HMul.hMul 3 (HDiv.hDiv R 4))) (M …
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul 3 (HDiv.hDiv R 4)))) (HMul.hMul (↑( …
    -/
    apply (measure_mono this).trans _
    /-
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LT.lt R r
      this : HasSubset.Subset (Metric.closedBall x (HMul.hMul 3 (HDiv.hDiv R 4))) (M …
      ⊢ LE.le (μ (Metric.closedBall y r)) (HMul.hMul (↑(IsUnifLocDoublingMeasure.sca …
    -/
    refine le_mul_of_one_le_left (zero_le _) ?_
    /-
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x y : α
      r : Real
      h : LE.le (Dist.dist x y) (HMul.hMul K r)
      rpos : LT.lt 0 r
      R : Real := IsUnifLocDoublingMeasure.scalingScaleOf μ (Max.max (HAdd.hAdd (HMu …
      H : HasSubset.Subset (Metric.closedBall y r) (Metric.closedBall x (HDiv.hDiv R …
      hr : LT.lt R r
      this : HasSubset.Subset (Metric.closedBall x (HMul.hMul 3 (HDiv.hDiv R 4))) (M …
      ⊢ LE.le 1 ↑(IsUnifLocDoublingMeasure.scalingConstantOf μ (Max.max (HAdd.hAdd ( …
    -/
    exact ENNReal.one_le_coe_iff.2 (le_max_right _ _)
    /-
      🎉 no goals
    -/


theorem tendsto_closedBall_filterAt {K : ℝ} {x : α} {ι : Type*} {l : Filter ι} (w : ι → α)
    (δ : ι → ℝ) (δlim : Tendsto δ l (𝓝[>] 0)) (xmem : ∀ᶠ j in l, x ∈ closedBall (w j) (K * δ j)) :
    Tendsto (fun j => closedBall (w j) (δ j)) l ((vitaliFamily μ K).filterAt x) := by
  /-
    α : Type u_1
    inst✝⁵ : PseudoMetricSpace α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : IsUnifLocDoublingMeasure μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    K : Real
    x : α
    ι : Type u_2
    l : Filter ι
    w : ι → α
    δ : ι → Real
    δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
    xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
    ⊢ Filter.Tendsto (fun j => Metric.closedBall (w j) (δ j)) l ((IsUnifLocDoublin …
  -/
  refine (vitaliFamily μ K).tendsto_filterAt_iff.mpr ⟨?_, fun ε hε => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ⊢ Filter.Eventually (fun i => Membership.mem ((IsUnifLocDoublingMeasure.vitali …
    -/
  · filter_upwards [xmem, δlim self_mem_nhdsWithin] with j hj h'j
    /-
      case h
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      j : ι
      hj : Membership.mem (Metric.closedBall (w j) (HMul.hMul K (δ j))) x
      h'j : Membership.mem (Set.preimage δ (Set.Ioi 0)) j
      ⊢ Membership.mem ((IsUnifLocDoublingMeasure.vitaliFamily μ K).setsAt x) (Metri …
    -/
    exact closedBall_mem_vitaliFamily_of_dist_le_mul μ hj h'j
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
    -/
  · rcases l.eq_or_neBot with rfl | h
      /-
        case refine_2.inl
        α : Type u_1
        inst✝⁵ : PseudoMetricSpace α
        inst✝⁴ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : IsUnifLocDoublingMeasure μ
        inst✝² : SecondCountableTopology α
        inst✝¹ : BorelSpace α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        K : Real
        x : α
        ι : Type u_2
        w : ι → α
        δ : ι → Real
        ε : Real
        hε : GT.gt ε 0
        δlim : Filter.Tendsto δ Bot.bot (nhdsWithin 0 (Set.Ioi 0))
        xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
        ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
      -/
    · simp
      /-
        🎉 no goals
      -/
    have hK : 0 ≤ K := by
      rcases (xmem.and (δlim self_mem_nhdsWithin)).exists with ⟨j, hj, h'j⟩
      have : 0 ≤ K * δ j := nonempty_closedBall.1 ⟨x, hj⟩
      exact (mul_nonneg_iff_left_nonneg_of_pos (mem_Ioi.1 h'j)).1 this
    /-
      case refine_2.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      hK : LE.le 0 K
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
    -/
    have δpos := eventually_mem_of_tendsto_nhdsWithin δlim
    /-
      case refine_2.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      δlim : Filter.Tendsto δ l (nhdsWithin 0 (Set.Ioi 0))
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      hK : LE.le 0 K
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
    -/
    replace δlim := tendsto_nhds_of_tendsto_nhdsWithin δlim
    /-
      case refine_2.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      hK : LE.le 0 K
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      δlim : Filter.Tendsto δ l (nhds 0)
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
    -/
    replace hK : 0 < K + 1 := by linarith
    /-
      case refine_2.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      δlim : Filter.Tendsto δ l (nhds 0)
      hK : LT.lt 0 (HAdd.hAdd K 1)
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Metric.closedBall (w i) (δ i)) …
    -/
    apply (((Metric.tendsto_nhds.mp δlim _ (div_pos hε hK)).and δpos).and xmem).mono
    /-
      case refine_2.inr
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      δlim : Filter.Tendsto δ l (nhds 0)
      hK : LT.lt 0 (HAdd.hAdd K 1)
      ⊢ ∀ (x_1 : ι), And (And (LT.lt (Dist.dist (δ x_1) 0) (HDiv.hDiv ε (HAdd.hAdd K …
    -/
    rintro j ⟨⟨hjε, hj₀ : 0 < δ j⟩, hx⟩ y hy
    replace hjε : (K + 1) * δ j < ε := by
      simpa [abs_eq_self.mpr hj₀.le] using (lt_div_iff₀' hK).mp hjε
    /-
      case refine_2.inr.intro.intro
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      δlim : Filter.Tendsto δ l (nhds 0)
      hK : LT.lt 0 (HAdd.hAdd K 1)
      j : ι
      hx : Membership.mem (Metric.closedBall (w j) (HMul.hMul K (δ j))) x
      hj₀ : LT.lt 0 (δ j)
      y : α
      hy : Membership.mem (Metric.closedBall (w j) (δ j)) y
      hjε : LT.lt (HMul.hMul (HAdd.hAdd K 1) (δ j)) ε
      ⊢ Membership.mem (Metric.closedBall x ε) y
    -/
    simp only [mem_closedBall] at hx hy ⊢
    /-
      case refine_2.inr.intro.intro
      α : Type u_1
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : IsUnifLocDoublingMeasure μ
      inst✝² : SecondCountableTopology α
      inst✝¹ : BorelSpace α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      K : Real
      x : α
      ι : Type u_2
      l : Filter ι
      w : ι → α
      δ : ι → Real
      xmem : Filter.Eventually (fun j => Membership.mem (Metric.closedBall (w j) (HM …
      ε : Real
      hε : GT.gt ε 0
      h : l.NeBot
      δpos : Filter.Eventually (fun i => Membership.mem (Set.Ioi 0) (δ i)) l
      δlim : Filter.Tendsto δ l (nhds 0)
      hK : LT.lt 0 (HAdd.hAdd K 1)
      j : ι
      hj₀ : LT.lt 0 (δ j)
      y : α
      hjε : LT.lt (HMul.hMul (HAdd.hAdd K 1) (δ j)) ε
      hx : LE.le (Dist.dist x (w j)) (HMul.hMul K (δ j))
      hy : LE.le (Dist.dist y (w j)) (δ j)
      ⊢ LE.le (Dist.dist y x) ε
    -/
    linarith [dist_triangle_right y x (w j)]
    /-
      🎉 no goals
    -/


/-- A version of **Lebesgue's density theorem** for a sequence of closed balls whose centers are
not required to be fixed.

See also `Besicovitch.ae_tendsto_measure_inter_div`. -/
theorem ae_tendsto_measure_inter_div (S : Set α) (K : ℝ) : ∀ᵐ x ∂μ.restrict S,
    ∀ {ι : Type*} {l : Filter ι} (w : ι → α) (δ : ι → ℝ) (_ : Tendsto δ l (𝓝[>] 0))
      (_ : ∀ᶠ j in l, x ∈ closedBall (w j) (K * δ j)),
      Tendsto (fun j => μ (S ∩ closedBall (w j) (δ j)) / μ (closedBall (w j) (δ j))) l (𝓝 1) := by
  filter_upwards [(vitaliFamily μ K).ae_tendsto_measure_inter_div S] with x hx ι l w δ δlim
    xmem using hx.comp (tendsto_closedBall_filterAt μ _ _ δlim xmem)


/-- A version of **Lebesgue differentiation theorem** for a sequence of closed balls whose
centers are not required to be fixed. -/
theorem ae_tendsto_average_norm_sub {f : α → E} (hf : LocallyIntegrable f μ) (K : ℝ) : ∀ᵐ x ∂μ,
    ∀ {ι : Type*} {l : Filter ι} (w : ι → α) (δ : ι → ℝ) (_ : Tendsto δ l (𝓝[>] 0))
      (_ : ∀ᶠ j in l, x ∈ closedBall (w j) (K * δ j)),
      Tendsto (fun j => ⨍ y in closedBall (w j) (δ j), ‖f y - f x‖ ∂μ) l (𝓝 0) := by
  filter_upwards [(vitaliFamily μ K).ae_tendsto_average_norm_sub hf] with x hx ι l w δ δlim
    xmem using hx.comp (tendsto_closedBall_filterAt μ _ _ δlim xmem)


/-- A version of **Lebesgue differentiation theorem** for a sequence of closed balls whose
centers are not required to be fixed. -/
theorem ae_tendsto_average [NormedSpace ℝ E] [CompleteSpace E]
    {f : α → E} (hf : LocallyIntegrable f μ) (K : ℝ) : ∀ᵐ x ∂μ,
      ∀ {ι : Type*} {l : Filter ι} (w : ι → α) (δ : ι → ℝ) (_ : Tendsto δ l (𝓝[>] 0))
        (_ : ∀ᶠ j in l, x ∈ closedBall (w j) (K * δ j)),
        Tendsto (fun j => ⨍ y in closedBall (w j) (δ j), f y ∂μ) l (𝓝 (f x)) := by
  filter_upwards [(vitaliFamily μ K).ae_tendsto_average hf] with x hx ι l w δ δlim xmem using
    hx.comp (tendsto_closedBall_filterAt μ _ _ δlim xmem)


