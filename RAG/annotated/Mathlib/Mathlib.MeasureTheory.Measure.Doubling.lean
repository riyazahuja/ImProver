/-- A measure `μ` is said to be a uniformly locally doubling measure if there exists a constant `C`
such that for all sufficiently small radii `ε`, and for any centre, the measure of a ball of radius
`2 * ε` is bounded by `C` times the measure of the concentric ball of radius `ε`.

Note: it is important that this definition makes a demand only for sufficiently small `ε`. For
example we want hyperbolic space to carry the instance `IsUnifLocDoublingMeasure volume` but
volumes grow exponentially in hyperbolic space. To be really explicit, consider the hyperbolic plane
of curvature -1, the area of a disc of radius `ε` is `A(ε) = 2π(cosh(ε) - 1)` so
`A(2ε)/A(ε) ~ exp(ε)`. -/
class IsUnifLocDoublingMeasure {α : Type*} [PseudoMetricSpace α] [MeasurableSpace α]
  (μ : Measure α) : Prop where
  exists_measure_closedBall_le_mul'' :
    ∃ C : ℝ≥0, ∀ᶠ ε in 𝓝[>] 0, ∀ x, μ (closedBall x (2 * ε)) ≤ C * μ (closedBall x ε)


theorem exists_measure_closedBall_le_mul :
    ∃ C : ℝ≥0, ∀ᶠ ε in 𝓝[>] 0, ∀ x, μ (closedBall x (2 * ε)) ≤ C * μ (closedBall x ε) :=
  exists_measure_closedBall_le_mul''


/-- A doubling constant for a uniformly locally doubling measure.

See also `IsUnifLocDoublingMeasure.scalingConstantOf`. -/
def doublingConstant : ℝ≥0 :=
  Classical.choose <| exists_measure_closedBall_le_mul μ


theorem exists_measure_closedBall_le_mul' :
    ∀ᶠ ε in 𝓝[>] 0, ∀ x, μ (closedBall x (2 * ε)) ≤ doublingConstant μ * μ (closedBall x ε) :=
  Classical.choose_spec <| exists_measure_closedBall_le_mul μ


theorem exists_eventually_forall_measure_closedBall_le_mul (K : ℝ) :
    ∃ C : ℝ≥0, ∀ᶠ ε in 𝓝[>] 0, ∀ x, ∀ t ≤ K, μ (closedBall x (t * ε)) ≤ C * μ (closedBall x ε) := by
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    ⊢ Exists fun C => Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K  …
  -/
  let C := doublingConstant μ
  have hμ :
    ∀ n : ℕ, ∀ᶠ ε in 𝓝[>] 0, ∀ x,
      μ (closedBall x ((2 : ℝ) ^ n * ε)) ≤ ↑(C ^ n) * μ (closedBall x ε) := by
    intro n
    induction' n with n ih
    · simp
    replace ih := eventually_nhdsGT_zero_mul_left (two_pos : 0 < (2 : ℝ)) ih
    refine (ih.and (exists_measure_closedBall_le_mul' μ)).mono fun ε hε x => ?_
    calc
      μ (closedBall x ((2 : ℝ) ^ (n + 1) * ε)) = μ (closedBall x ((2 : ℝ) ^ n * (2 * ε))) := by
        rw [pow_succ, mul_assoc]
      _ ≤ ↑(C ^ n) * μ (closedBall x (2 * ε)) := hε.1 x
      _ ≤ ↑(C ^ n) * (C * μ (closedBall x ε)) := by gcongr; exact hε.2 x
      _ = ↑(C ^ (n + 1)) * μ (closedBall x ε) := by rw [← mul_assoc, pow_succ, ENNReal.coe_mul]
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
    hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
    ⊢ Exists fun C => Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K  …
  -/
  rcases lt_or_le K 1 with (hK | hK)
    /-
      case inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LT.lt K 1
      ⊢ Exists fun C => Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K  …
    -/
  · refine ⟨1, ?_⟩
    /-
      case inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LT.lt K 1
      ⊢ Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Metr …
    -/
    simp only [ENNReal.coe_one, one_mul]
    /-
      case inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LT.lt K 1
      ⊢ Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Metr …
    -/
    refine eventually_mem_nhdsWithin.mono fun ε hε x t ht ↦ ?_
    /-
      case inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LT.lt K 1
      ε : Real
      hε : Membership.mem (Set.Ioi 0) ε
      x : α
      t : Real
      ht : LE.le t K
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t ε))) (μ (Metric.closedBall x ε))
    -/
    gcongr
    /-
      case inl.h.h
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LT.lt K 1
      ε : Real
      hε : Membership.mem (Set.Ioi 0) ε
      x : α
      t : Real
      ht : LE.le t K
      ⊢ LE.le (HMul.hMul t ε) ε
    -/
    nlinarith [mem_Ioi.mp hε]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LE.le 1 K
      ⊢ Exists fun C => Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K  …
    -/
  · use C ^ ⌈Real.logb 2 K⌉₊
    /-
      case h
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LE.le 1 K
      ⊢ Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Metr …
    -/
    filter_upwards [hμ ⌈Real.logb 2 K⌉₊, eventually_mem_nhdsWithin] with ε hε hε₀ x t ht
    /-
      case h
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LE.le 1 K
      ε : Real
      hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
      hε₀ : Membership.mem (Set.Ioi 0) ε
      x : α
      t : Real
      ht : LE.le t K
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t ε))) (HMul.hMul (↑(HPow.hPow C (N …
    -/
    refine le_trans ?_ (hε x)
    /-
      case h
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
      hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
      hK : LE.le 1 K
      ε : Real
      hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
      hε₀ : Membership.mem (Set.Ioi 0) ε
      x : α
      t : Real
      ht : LE.le t K
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t ε))) (μ (Metric.closedBall x (HMu …
    -/
    gcongr
      /-
        case h.h.h.a0
        α : Type u_1
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : IsUnifLocDoublingMeasure μ
        K : Real
        C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
        hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
        hK : LE.le 1 K
        ε : Real
        hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
        hε₀ : Membership.mem (Set.Ioi 0) ε
        x : α
        t : Real
        ht : LE.le t K
        ⊢ LE.le 0 ε
      -/
    · exact (mem_Ioi.mp hε₀).le
      /-
        🎉 no goals
      -/
      /-
        case h.h.h.h
        α : Type u_1
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : IsUnifLocDoublingMeasure μ
        K : Real
        C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
        hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
        hK : LE.le 1 K
        ε : Real
        hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
        hε₀ : Membership.mem (Set.Ioi 0) ε
        x : α
        t : Real
        ht : LE.le t K
        ⊢ LE.le t (HPow.hPow 2 (Nat.ceil (Real.logb 2 K)))
      -/
    · refine ht.trans ?_
      /-
        case h.h.h.h
        α : Type u_1
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : IsUnifLocDoublingMeasure μ
        K : Real
        C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
        hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
        hK : LE.le 1 K
        ε : Real
        hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
        hε₀ : Membership.mem (Set.Ioi 0) ε
        x : α
        t : Real
        ht : LE.le t K
        ⊢ LE.le K (HPow.hPow 2 (Nat.ceil (Real.logb 2 K)))
      -/
      rw [← Real.rpow_natCast, ← Real.logb_le_iff_le_rpow]
      /-
        case h.h.h.h
        α : Type u_1
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : IsUnifLocDoublingMeasure μ
        K : Real
        C : NNReal := IsUnifLocDoublingMeasure.doublingConstant μ
        hμ : ∀ (n : Nat), Filter.Eventually (fun ε => ∀ (x : α), LE.le (μ (Metric.clos …
        hK : LE.le 1 K
        ε : Real
        hε : ∀ (x : α), LE.le (μ (Metric.closedBall x (HMul.hMul (HPow.hPow 2 (Nat.cei …
        hε₀ : Membership.mem (Set.Ioi 0) ε
        x : α
        t : Real
        ht : LE.le t K
        ⊢ LE.le (Real.logb 2 K) ↑(Nat.ceil (Real.logb 2 K))
      -/
      exacts [Nat.le_ceil _, by norm_num, by linarith]
      /-
        🎉 no goals
      -/


/-- A variant of `IsUnifLocDoublingMeasure.doublingConstant` which allows for scaling the
radius by values other than `2`. -/
def scalingConstantOf (K : ℝ) : ℝ≥0 :=
  max (Classical.choose <| exists_eventually_forall_measure_closedBall_le_mul μ K) 1


@[simp]
theorem one_le_scalingConstantOf (K : ℝ) : 1 ≤ scalingConstantOf μ K :=
  le_max_of_le_right <| le_refl 1


theorem eventually_measure_mul_le_scalingConstantOf_mul (K : ℝ) :
    ∃ R : ℝ,
      0 < R ∧
        ∀ x t r, t ∈ Ioc 0 K → r ≤ R →
          μ (closedBall x (t * r)) ≤ scalingConstantOf μ K * μ (closedBall x r) := by
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α) (t r : Real), Membership.mem (Set …
  -/
  have h := Classical.choose_spec (exists_eventually_forall_measure_closedBall_le_mul μ K)
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α) (t r : Real), Membership.mem (Set …
  -/
  rcases mem_nhdsGT_iff_exists_Ioc_subset.1 h with ⟨R, Rpos, hR⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
    R : Real
    Rpos : Membership.mem (Set.Ioi 0) R
    hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
    ⊢ Exists fun R => And (LT.lt 0 R) (∀ (x : α) (t r : Real), Membership.mem (Set …
  -/
  refine ⟨R, Rpos, fun x t r ht hr => ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
    R : Real
    Rpos : Membership.mem (Set.Ioi 0) R
    hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
    x : α
    t r : Real
    ht : Membership.mem (Set.Ioc 0 K) t
    hr : LE.le r R
    ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t r))) (HMul.hMul (↑(IsUnifLocDoubl …
  -/
  rcases lt_trichotomy r 0 with (rneg | rfl | rpos)
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t r : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le r R
      rneg : LT.lt r 0
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t r))) (HMul.hMul (↑(IsUnifLocDoubl …
    -/
  · have : t * r < 0 := mul_neg_of_pos_of_neg ht.1 rneg
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t r : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le r R
      rneg : LT.lt r 0
      this : LT.lt (HMul.hMul t r) 0
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t r))) (HMul.hMul (↑(IsUnifLocDoubl …
    -/
    simp only [closedBall_eq_empty.2 this, measure_empty, zero_le']
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le 0 R
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t 0))) (HMul.hMul (↑(IsUnifLocDoubl …
    -/
  · simp only [mul_zero, closedBall_zero]
    /-
      case intro.intro.inr.inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le 0 R
      ⊢ LE.le (μ (Metric.closedBall x 0)) (HMul.hMul (↑(IsUnifLocDoublingMeasure.sca …
    -/
    refine le_mul_of_one_le_of_le ?_ le_rfl
    /-
      case intro.intro.inr.inl
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le 0 R
      ⊢ LE.le 1 ↑(IsUnifLocDoublingMeasure.scalingConstantOf μ K)
    -/
    apply ENNReal.one_le_coe_iff.2 (le_max_right _ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t r : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le r R
      rpos : LT.lt 0 r
      ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul t r))) (HMul.hMul (↑(IsUnifLocDoubl …
    -/
  · apply (hR ⟨rpos, hr⟩ x t ht.2).trans
    /-
      case intro.intro.inr.inr
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t r : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le r R
      rpos : LT.lt 0 r
      ⊢ LE.le (HMul.hMul (↑(Classical.choose ⋯)) (μ (Metric.closedBall x r))) (HMul. …
    -/
    gcongr
    /-
      case intro.intro.inr.inr.bc.a
      α : Type u_1
      inst✝² : PseudoMetricSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : IsUnifLocDoublingMeasure μ
      K : Real
      h : Filter.Eventually (fun ε => ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Me …
      R : Real
      Rpos : Membership.mem (Set.Ioi 0) R
      hR : HasSubset.Subset (Set.Ioc 0 R) (setOf fun x => (fun ε => ∀ (x : α) (t : R …
      x : α
      t r : Real
      ht : Membership.mem (Set.Ioc 0 K) t
      hr : LE.le r R
      rpos : LT.lt 0 r
      ⊢ LE.le (Classical.choose ⋯) (IsUnifLocDoublingMeasure.scalingConstantOf μ K)
    -/
    apply le_max_left
    /-
      🎉 no goals
    -/


theorem eventually_measure_le_scaling_constant_mul (K : ℝ) :
    ∀ᶠ r in 𝓝[>] 0, ∀ x, μ (closedBall x (K * r)) ≤ scalingConstantOf μ K * μ (closedBall x r) := by
  filter_upwards [Classical.choose_spec
      (exists_eventually_forall_measure_closedBall_le_mul μ K)] with r hr x
  /-
    case h
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K r : Real
    hr : ∀ (x : α) (t : Real), LE.le t K → LE.le (μ (Metric.closedBall x (HMul.hMu …
    x : α
    ⊢ LE.le (μ (Metric.closedBall x (HMul.hMul K r))) (HMul.hMul (↑(IsUnifLocDoubl …
  -/
  exact (hr x K le_rfl).trans (mul_le_mul_right' (ENNReal.coe_le_coe.2 (le_max_left _ _)) _)
  /-
    🎉 no goals
  -/


theorem eventually_measure_le_scaling_constant_mul' (K : ℝ) (hK : 0 < K) :
    ∀ᶠ r in 𝓝[>] 0, ∀ x,
      μ (closedBall x r) ≤ scalingConstantOf μ K⁻¹ * μ (closedBall x (K * r)) := by
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    hK : LT.lt 0 K
    ⊢ Filter.Eventually (fun r => ∀ (x : α), LE.le (μ (Metric.closedBall x r)) (HM …
  -/
  convert eventually_nhdsGT_zero_mul_left hK (eventually_measure_le_scaling_constant_mul μ K⁻¹)
  /-
    case h.e'_2.h.h.h.e'_3.h.e'_6.h.e'_4
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : IsUnifLocDoublingMeasure μ
    K : Real
    hK : LT.lt 0 K
    x✝ : Real
    a✝ : α
    ⊢ Eq x✝ (HMul.hMul (Inv.inv K) (HMul.hMul K x✝))
  -/
  simp [inv_mul_cancel_left₀ hK.ne']
  /-
    🎉 no goals
  -/


/-- A scale below which the doubling measure `μ` satisfies good rescaling properties when one
multiplies the radius of balls by at most `K`, as stated
in `IsUnifLocDoublingMeasure.measure_mul_le_scalingConstantOf_mul`. -/
def scalingScaleOf (K : ℝ) : ℝ :=
  (eventually_measure_mul_le_scalingConstantOf_mul μ K).choose


theorem scalingScaleOf_pos (K : ℝ) : 0 < scalingScaleOf μ K :=
  (eventually_measure_mul_le_scalingConstantOf_mul μ K).choose_spec.1


theorem measure_mul_le_scalingConstantOf_mul {K : ℝ} {x : α} {t r : ℝ} (ht : t ∈ Ioc 0 K)
    (hr : r ≤ scalingScaleOf μ K) :
    μ (closedBall x (t * r)) ≤ scalingConstantOf μ K * μ (closedBall x r) :=
  (eventually_measure_mul_le_scalingConstantOf_mul μ K).choose_spec.2 x t r ht hr


