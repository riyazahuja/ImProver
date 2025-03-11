/-- If we can fit a small ball inside a set `s` intersected with any neighborhood of `x`, then the
density of `s` near `x` is not `0`.

Along with `aux₁`, this proves that `x` is a Lebesgue point of `s`. This will be used to prove that
the frontier of an order-connected set is null. -/
private lemma aux₀
    (h : ∀ δ, 0 < δ →
      ∃ y, closedBall y (δ / 4) ⊆ closedBall x δ ∧ closedBall y (δ / 4) ⊆ interior s) :
    ¬Tendsto (fun r ↦ volume (closure s ∩ closedBall x r) / volume (closedBall x r)) (𝓝[>] 0)
        (𝓝 0) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    h : ∀ (δ : Real), LT.lt 0 δ → Exists fun y => And (HasSubset.Subset (Metric.cl …
    ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
  -/
  choose f hf₀ hf₁ using h
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
  -/
  intro H
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
    ⊢ False
  -/
  obtain ⟨ε, -, hε', hε₀⟩ := exists_seq_strictAnti_tendsto_nhdsWithin (0 : ℝ)
  refine not_eventually.2
    (Frequently.of_forall fun _ ↦ lt_irrefl <| ENNReal.ofReal <| 4⁻¹ ^ Fintype.card ι)
    ((Filter.Tendsto.eventually_lt (H.comp hε₀) tendsto_const_nhds ?_).mono fun n ↦
      lt_of_le_of_lt ?_)
  on_goal 2 =>
    calc
      ENNReal.ofReal (4⁻¹ ^ Fintype.card ι)
        = volume (closedBall (f (ε n) (hε' n)) (ε n / 4)) / volume (closedBall x (ε n)) := ?_
      _ ≤ volume (closure s ∩ closedBall x (ε n)) / volume (closedBall x (ε n)) := by
        gcongr
        exact subset_inter ((hf₁ _ <| hε' n).trans interior_subset_closure) <| hf₀ _ <| hε' n
    have := hε' n
    rw [Real.volume_pi_closedBall, Real.volume_pi_closedBall, ← ENNReal.ofReal_div_of_pos,
      ← div_pow, mul_div_mul_left _ _ (two_ne_zero' ℝ), div_right_comm, div_self, one_div]
  /-
    case intro.intro.intro.refine_1
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
    ε : Nat → Real
    hε' : ∀ (n : Nat), LT.lt 0 (ε n)
    hε₀ : Filter.Tendsto ε Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    ⊢ LT.lt 0 (ENNReal.ofReal (HPow.hPow (Inv.inv 4) (Fintype.card ι)))
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


/-- If we can fit a small ball inside a set `sᶜ` intersected with any neighborhood of `x`, then the
density of `s` near `x` is not `1`.

Along with `aux₀`, this proves that `x` is a Lebesgue point of `s`. This will be used to prove that
the frontier of an order-connected set is null. -/
private lemma aux₁
    (h : ∀ δ, 0 < δ →
      ∃ y, closedBall y (δ / 4) ⊆ closedBall x δ ∧ closedBall y (δ / 4) ⊆ interior sᶜ) :
    ¬Tendsto (fun r ↦ volume (closure s ∩ closedBall x r) / volume (closedBall x r)) (𝓝[>] 0)
        (𝓝 1) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    h : ∀ (δ : Real), LT.lt 0 δ → Exists fun y => And (HasSubset.Subset (Metric.cl …
    ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
  -/
  choose f hf₀ hf₁ using h
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
  -/
  intro H
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
    ⊢ False
  -/
  obtain ⟨ε, -, hε', hε₀⟩ := exists_seq_strictAnti_tendsto_nhdsWithin (0 : ℝ)
  refine not_eventually.2
      (Frequently.of_forall fun _ ↦ lt_irrefl <| 1 - ENNReal.ofReal (4⁻¹ ^ Fintype.card ι))
      ((Filter.Tendsto.eventually_lt tendsto_const_nhds (H.comp hε₀) <|
            ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero ?_).mono
        fun n ↦ lt_of_le_of_lt' ?_)
  on_goal 2 =>
    calc
      volume (closure s ∩ closedBall x (ε n)) / volume (closedBall x (ε n))
        ≤ volume (closedBall x (ε n) \ closedBall (f (ε n) <| hε' n) (ε n / 4)) /
          volume (closedBall x (ε n)) := by
        gcongr
        rw [diff_eq_compl_inter]
        refine inter_subset_inter_left _ ?_
        rw [subset_compl_comm, ← interior_compl]
        exact hf₁ _ _
      _ = 1 - ENNReal.ofReal (4⁻¹ ^ Fintype.card ι) := ?_
    dsimp only
    have := hε' n
    rw [measure_diff (hf₀ _ _) _ ((Real.volume_pi_closedBall _ _).trans_ne ENNReal.ofReal_ne_top),
      Real.volume_pi_closedBall, Real.volume_pi_closedBall, ENNReal.sub_div fun _ _ ↦ _,
      ENNReal.div_self _ ENNReal.ofReal_ne_top, ← ENNReal.ofReal_div_of_pos, ← div_pow,
      mul_div_mul_left _ _ (two_ne_zero' ℝ), div_right_comm, div_self, one_div]
  /-
    case intro.intro.intro.refine_1
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    f : (δ : Real) → LT.lt 0 δ → ι → Real
    hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
    H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
    ε : Nat → Real
    hε' : ∀ (n : Nat), LT.lt 0 (ε n)
    hε₀ : Filter.Tendsto ε Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    ⊢ Ne (ENNReal.ofReal (HPow.hPow (Inv.inv 4) (Fintype.card ι))) 0
  -/
  all_goals try positivity
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      f : (δ : Real) → LT.lt 0 δ → ι → Real
      hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
      hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
      H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
      ε : Nat → Real
      hε' : ∀ (n : Nat), LT.lt 0 (ε n)
      hε₀ : Filter.Tendsto ε Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      n : Nat
      this : LT.lt 0 (ε n)
      ⊢ LT.lt 0 (ENNReal.ofReal (HPow.hPow (HMul.hMul 2 (HDiv.hDiv (ε n) 4)) (Fintyp …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      f : (δ : Real) → LT.lt 0 δ → ι → Real
      hf₀ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
      hf₁ : ∀ (δ : Real) (a : LT.lt 0 δ), HasSubset.Subset (Metric.closedBall (f δ a …
      H : Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume (Int …
      ε : Nat → Real
      hε' : ∀ (n : Nat), LT.lt 0 (ε n)
      hε₀ : Filter.Tendsto ε Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
      n : Nat
      this : LT.lt 0 (ε n)
      ⊢ MeasureTheory.NullMeasurableSet (Metric.closedBall (f (ε n) ⋯) (HDiv.hDiv (ε …
    -/
  · exact measurableSet_closedBall.nullMeasurableSet
    /-
      🎉 no goals
    -/


theorem IsUpperSet.null_frontier (hs : IsUpperSet s) : volume (frontier s) = 0 := by
  refine measure_mono_null (fun x hx ↦ ?_)
    (Besicovitch.ae_tendsto_measure_inter_div_of_measurableSet _
      (isClosed_closure (s := s)).measurableSet)
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs : IsUpperSet s
    x : ι → Real
    hx : Membership.mem (frontier s) x
    ⊢ Membership.mem (HasCompl.compl (setOf fun x => (fun x => Filter.Tendsto (fun …
  -/
  by_cases h : x ∈ closure s <;>
    simp only [mem_compl_iff, mem_setOf, h, not_false_eq_true, indicator_of_not_mem,
      indicator_of_mem, Pi.one_apply]
    /-
      case pos
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsUpperSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Membership.mem (closure s) x
      ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
    -/
  · refine aux₁ fun _ ↦ hs.compl.exists_subset_ball <| frontier_subset_closure ?_
    /-
      case pos
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsUpperSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Membership.mem (closure s) x
      x✝ : Real
      ⊢ Membership.mem (frontier (HasCompl.compl s)) x
    -/
    rwa [frontier_compl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsUpperSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Not (Membership.mem (closure s) x)
      ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
    -/
  · exact aux₀ fun _ ↦ hs.exists_subset_ball <| frontier_subset_closure hx
    /-
      🎉 no goals
    -/


theorem IsLowerSet.null_frontier (hs : IsLowerSet s) : volume (frontier s) = 0 := by
  refine measure_mono_null (fun x hx ↦ ?_)
    (Besicovitch.ae_tendsto_measure_inter_div_of_measurableSet _
      (isClosed_closure (s := s)).measurableSet)
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs : IsLowerSet s
    x : ι → Real
    hx : Membership.mem (frontier s) x
    ⊢ Membership.mem (HasCompl.compl (setOf fun x => (fun x => Filter.Tendsto (fun …
  -/
  by_cases h : x ∈ closure s <;>
    simp only [mem_compl_iff, mem_setOf, h, not_false_eq_true, indicator_of_not_mem,
      indicator_of_mem, Pi.one_apply]
    /-
      case pos
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsLowerSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Membership.mem (closure s) x
      ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
    -/
  · refine aux₁ fun _ ↦ hs.compl.exists_subset_ball <| frontier_subset_closure ?_
    /-
      case pos
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsLowerSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Membership.mem (closure s) x
      x✝ : Real
      ⊢ Membership.mem (frontier (HasCompl.compl s)) x
    -/
    rwa [frontier_compl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs : IsLowerSet s
      x : ι → Real
      hx : Membership.mem (frontier s) x
      h : Not (Membership.mem (closure s) x)
      ⊢ Not (Filter.Tendsto (fun r => HDiv.hDiv (MeasureTheory.MeasureSpace.volume ( …
    -/
  · exact aux₀ fun _ ↦ hs.exists_subset_ball <| frontier_subset_closure hx
    /-
      🎉 no goals
    -/


theorem Set.OrdConnected.null_frontier (hs : s.OrdConnected) : volume (frontier s) = 0 := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs : s.OrdConnected
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
  -/
  rw [← hs.upperClosure_inter_lowerClosure]
  exact measure_mono_null (frontier_inter_subset _ _) <| measure_union_null
    (measure_inter_null_of_null_left _ (UpperSet.upper _).null_frontier)
    (measure_inter_null_of_null_right _ (LowerSet.lower _).null_frontier)


                                                                             /-
                                                                               ι : Type u_1
                                                                               inst✝ : Fintype ι
                                                                               s : Set (ι → Real)
                                                                               x : ι → Real
                                                                               hs : s.OrdConnected
                                                                               ⊢ MeasureTheory.Measure (ι → Real)
                                                                             -/
protected theorem Set.OrdConnected.nullMeasurableSet (hs : s.OrdConnected) : NullMeasurableSet s :=
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  nullMeasurableSet_of_null_frontier hs.null_frontier


theorem IsAntichain.volume_eq_zero [Nonempty ι] (hs : IsAntichain (· ≤ ·) s) : volume s = 0 := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    s : Set (ι → Real)
    inst✝ : Nonempty ι
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq (MeasureTheory.MeasureSpace.volume s) 0
  -/
  refine measure_mono_null ?_ hs.ordConnected.null_frontier
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    s : Set (ι → Real)
    inst✝ : Nonempty ι
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ HasSubset.Subset s (frontier s)
  -/
  rw [← closure_diff_interior, hs.interior_eq_empty, diff_empty]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    s : Set (ι → Real)
    inst✝ : Nonempty ι
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ HasSubset.Subset s (closure s)
  -/
  exact subset_closure
  /-
    🎉 no goals
  -/

