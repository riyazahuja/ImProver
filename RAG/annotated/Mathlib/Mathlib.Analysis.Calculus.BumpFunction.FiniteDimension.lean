/-- If a set `s` is a neighborhood of `x`, then there exists a smooth function `f` taking
values in `[0, 1]`, supported in `s` and with `f x = 1`. -/
theorem exists_smooth_tsupport_subset {s : Set E} {x : E} (hs : s ∈ 𝓝 x) :
    ∃ f : E → ℝ,
      tsupport f ⊆ s ∧ HasCompactSupport f ∧ ContDiff ℝ ∞ f ∧ range f ⊆ Icc 0 1 ∧ f x = 1 := by
  obtain ⟨d : ℝ, d_pos : 0 < d, hd : Euclidean.closedBall x d ⊆ s⟩ :=
    Euclidean.nhds_basis_closedBall.mem_iff.1 hs
  let c : ContDiffBump (toEuclidean x) :=
    { rIn := d / 2
      rOut := d
      rIn_pos := half_pos d_pos
      rIn_lt_rOut := half_lt_self d_pos }
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    d : Real
    d_pos : LT.lt 0 d
    hd : HasSubset.Subset (Euclidean.closedBall x d) s
    c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport f) s) (And (HasCompactSuppor …
  -/
  let f : E → ℝ := c ∘ toEuclidean
  have f_supp : f.support ⊆ Euclidean.ball x d := by
    intro y hy
    have : toEuclidean y ∈ Function.support c := by
      simpa only [Function.mem_support, Function.comp_apply, Ne] using hy
    rwa [c.support_eq] at this
  have f_tsupp : tsupport f ⊆ Euclidean.closedBall x d := by
    rw [tsupport, ← Euclidean.closure_ball _ d_pos.ne']
    exact closure_mono f_supp
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    d : Real
    d_pos : LT.lt 0 d
    hd : HasSubset.Subset (Euclidean.closedBall x d) s
    c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
    f : E → Real := Function.comp ↑c ⇑toEuclidean
    f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
    f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport f) s) (And (HasCompactSuppor …
  -/
  refine ⟨f, f_tsupp.trans hd, ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ HasCompactSupport f
    -/
  · refine isCompact_of_isClosed_isBounded isClosed_closure ?_
    /-
      case intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ Bornology.IsBounded (tsupport f)
    -/
    have : IsBounded (Euclidean.closedBall x d) := Euclidean.isCompact_closedBall.isBounded
    /-
      case intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      this : Bornology.IsBounded (Euclidean.closedBall x d)
      ⊢ Bornology.IsBounded (tsupport f)
    -/
    refine this.subset (Euclidean.isClosed_closedBall.closure_subset_iff.2 ?_)
    /-
      case intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      this : Bornology.IsBounded (Euclidean.closedBall x d)
      ⊢ HasSubset.Subset (Function.support f) (Euclidean.closedBall x d)
    -/
    exact f_supp.trans Euclidean.ball_subset_closedBall
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ ContDiff Real (↑Top.top) f
    -/
  · apply c.contDiff.comp
    /-
      case intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ ContDiff Real ↑Top.top ⇑toEuclidean
    -/
    exact ContinuousLinearEquiv.contDiff _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    -/
  · rintro t ⟨y, rfl⟩
    /-
      case intro.intro.refine_3.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      y : E
      ⊢ Membership.mem (Set.Icc 0 1) (f y)
    -/
    exact ⟨c.nonneg, c.le_one⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_4
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ Eq (f x) 1
    -/
  · apply c.one_of_mem_closedBall
    /-
      case intro.intro.refine_4
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ Membership.mem (Metric.closedBall (toEuclidean x) c.rIn) (toEuclidean x)
    -/
    apply mem_closedBall_self
    /-
      case intro.intro.refine_4.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      x : E
      hs : Membership.mem (nhds x) s
      d : Real
      d_pos : LT.lt 0 d
      hd : HasSubset.Subset (Euclidean.closedBall x d) s
      c : ContDiffBump (toEuclidean x) := { rIn := HDiv.hDiv d 2, rOut := d, rIn_pos …
      f : E → Real := Function.comp ↑c ⇑toEuclidean
      f_supp : HasSubset.Subset (Function.support f) (Euclidean.ball x d)
      f_tsupp : HasSubset.Subset (tsupport f) (Euclidean.closedBall x d)
      ⊢ LE.le 0 c.rIn
    -/
    exact (half_pos d_pos).le
    /-
      🎉 no goals
    -/


/-- Given an open set `s` in a finite-dimensional real normed vector space, there exists a smooth
function with values in `[0, 1]` whose support is exactly `s`. -/
theorem IsOpen.exists_smooth_support_eq {s : Set E} (hs : IsOpen s) :
    ∃ f : E → ℝ, f.support = s ∧ ContDiff ℝ ∞ f ∧ Set.range f ⊆ Set.Icc 0 1 := by
  /- For any given point `x` in `s`, one can construct a smooth function with support in `s` and
    nonzero at `x`. By second-countability, it follows that we may cover `s` with the supports of
    countably many such functions, say `g i`.
    Then `∑ i, r i • g i` will be the desired function if `r i` is a sequence of positive numbers
    tending quickly enough to zero. Indeed, this ensures that, for any `k ≤ i`, the `k`-th
    derivative of `r i • g i` is bounded by a prescribed (summable) sequence `u i`. From this, the
    summability of the series and of its successive derivatives follows. -/
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  rcases eq_empty_or_nonempty s with (rfl | h's)
  · exact
      ⟨fun _ => 0, Function.support_zero, contDiff_const, by
        simp only [range_const, singleton_subset_iff, left_mem_Icc, zero_le_one]⟩
  /-
    case inr
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  let ι := { f : E → ℝ // f.support ⊆ s ∧ HasCompactSupport f ∧ ContDiff ℝ ∞ f ∧ range f ⊆ Icc 0 1 }
  obtain ⟨T, T_count, hT⟩ : ∃ T : Set ι, T.Countable ∧ ⋃ f ∈ T, support (f : E → ℝ) = s := by
    have : ⋃ f : ι, (f : E → ℝ).support = s := by
      refine Subset.antisymm (iUnion_subset fun f => f.2.1) ?_
      intro x hx
      rcases exists_smooth_tsupport_subset (hs.mem_nhds hx) with ⟨f, hf⟩
      let g : ι := ⟨f, (subset_tsupport f).trans hf.1, hf.2.1, hf.2.2.1, hf.2.2.2.1⟩
      have : x ∈ support (g : E → ℝ) := by
        simp only [g, hf.2.2.2.2, Subtype.coe_mk, mem_support, Ne, one_ne_zero,
          not_false_iff]
      exact mem_iUnion_of_mem _ this
    simp_rw [← this]
    apply isOpen_iUnion_countable
    rintro ⟨f, hf⟩
    exact hf.2.2.1.continuous.isOpen_support
  obtain ⟨g0, hg⟩ : ∃ g0 : ℕ → ι, T = range g0 := by
    apply Countable.exists_eq_range T_count
    rcases eq_empty_or_nonempty T with (rfl | hT)
    · simp only [ι, iUnion_false, iUnion_empty] at hT
      simp only [← hT, mem_empty_iff_false, iUnion_of_empty, iUnion_empty, Set.not_nonempty_empty]
          at h's
    · exact hT
  /-
    case inr.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  let g : ℕ → E → ℝ := fun n => (g0 n).1
  /-
    case inr.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  have g_s : ∀ n, support (g n) ⊆ s := fun n => (g0 n).2.1
  have s_g : ∀ x ∈ s, ∃ n, x ∈ support (g n) := fun x hx ↦ by
    rw [← hT] at hx
    obtain ⟨i, iT, hi⟩ : ∃ i ∈ T, x ∈ support (i : E → ℝ) := by
      simpa only [mem_iUnion, exists_prop] using hx
    rw [hg, mem_range] at iT
    rcases iT with ⟨n, hn⟩
    rw [← hn] at hi
    exact ⟨n, hi⟩
  /-
    case inr.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
    s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  have g_smooth : ∀ n, ContDiff ℝ ∞ (g n) := fun n => (g0 n).2.2.2.1
  /-
    case inr.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
    s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
    g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  have g_comp_supp : ∀ n, HasCompactSupport (g n) := fun n => (g0 n).2.2.1
  /-
    case inr.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
    s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
    g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
    g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  have g_nonneg : ∀ n x, 0 ≤ g n x := fun n x => ((g0 n).2.2.2.2 (mem_range_self x)).1
  obtain ⟨δ, δpos, c, δc, c_lt⟩ :
      ∃ δ : ℕ → ℝ≥0, (∀ i : ℕ, 0 < δ i) ∧ ∃ c : NNReal, HasSum δ c ∧ c < 1 :=
    NNReal.exists_pos_sum_of_countable one_ne_zero ℕ
  have : ∀ n : ℕ, ∃ r : ℝ, 0 < r ∧ ∀ i ≤ n, ∀ x, ‖iteratedFDeriv ℝ i (r • g n) x‖ ≤ δ n := by
    intro n
    have : ∀ i, ∃ R, ∀ x, ‖iteratedFDeriv ℝ i (fun x => g n x) x‖ ≤ R := by
      intro i
      have : BddAbove (range fun x => ‖iteratedFDeriv ℝ i (fun x : E => g n x) x‖) := by
        apply ((g_smooth n).continuous_iteratedFDeriv
          (mod_cast le_top)).norm.bddAbove_range_of_hasCompactSupport
        apply HasCompactSupport.comp_left _ norm_zero
        apply (g_comp_supp n).iteratedFDeriv
      rcases this with ⟨R, hR⟩
      exact ⟨R, fun x => hR (mem_range_self _)⟩
    choose R hR using this
    let M := max (((Finset.range (n + 1)).image R).max' (by simp)) 1
    have δnpos : 0 < δ n := δpos n
    have IR : ∀ i ≤ n, R i ≤ M := by
      intro i hi
      refine le_trans ?_ (le_max_left _ _)
      apply Finset.le_max'
      apply Finset.mem_image_of_mem
      -- Porting note: was
      -- simp only [Finset.mem_range]
      -- linarith
      simpa only [Finset.mem_range, Nat.lt_add_one_iff]
    refine ⟨M⁻¹ * δ n, by positivity, fun i hi x => ?_⟩
    calc
      ‖iteratedFDeriv ℝ i ((M⁻¹ * δ n) • g n) x‖ = ‖(M⁻¹ * δ n) • iteratedFDeriv ℝ i (g n) x‖ := by
        rw [iteratedFDeriv_const_smul_apply]
        exact (g_smooth n).of_le (mod_cast le_top)
      _ = M⁻¹ * δ n * ‖iteratedFDeriv ℝ i (g n) x‖ := by
        rw [norm_smul _ (iteratedFDeriv ℝ i (g n) x), Real.norm_of_nonneg]; positivity
      _ ≤ M⁻¹ * δ n * M := (mul_le_mul_of_nonneg_left ((hR i x).trans (IR i hi)) (by positivity))
      _ = δ n := by field_simp
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
    s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
    g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
    g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
    g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    δc : HasSum δ c
    c_lt : LT.lt c 1
    this : ∀ (n : Nat), Exists fun r => And (LT.lt 0 r) (∀ (i : Nat), LE.le i n →  …
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  choose r rpos hr using this
  have S : ∀ x, Summable fun n => (r n • g n) x := fun x ↦ by
    refine .of_nnnorm_bounded _ δc.summable fun n => ?_
    rw [← NNReal.coe_le_coe, coe_nnnorm]
    simpa only [norm_iteratedFDeriv_zero] using hr n 0 (zero_le n) x
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
    T : Set ι
    T_count : T.Countable
    hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
    g0 : Nat → ι
    hg : Eq T (Set.range g0)
    g : Nat → E → Real := fun n => ↑(g0 n)
    g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
    s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
    g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
    g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
    g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    δc : HasSum δ c
    c_lt : LT.lt c 1
    r : Nat → Real
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
    S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContDiff Real (↑Top.to …
  -/
  refine ⟨fun x => ∑' n, (r n • g n) x, ?_, ?_, ?_⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      ⊢ Eq (Function.support fun x => tsum fun n => HSMul.hSMul (r n) (g n) x) s
    -/
  · apply Subset.antisymm
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₁
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        ⊢ HasSubset.Subset (Function.support fun x => tsum fun n => HSMul.hSMul (r n)  …
      -/
    · intro x hx
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₁
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Membership.mem (Function.support fun x => tsum fun n => HSMul.hSMul (r n) …
        ⊢ Membership.mem s x
      -/
      simp only [Pi.smul_apply, Algebra.id.smul_eq_mul, mem_support, Ne] at hx
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₁
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Not (Eq (tsum fun n => HMul.hMul (r n) (g n x)) 0)
        ⊢ Membership.mem s x
      -/
      contrapose! hx
      have : ∀ n, g n x = 0 := by
        intro n
        contrapose! hx
        exact g_s n hx
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₁
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Not (Membership.mem s x)
        this : ∀ (n : Nat), Eq (g n x) 0
        ⊢ Eq (tsum fun n => HMul.hMul (r n) (g n x)) 0
      -/
      simp only [this, mul_zero, tsum_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₂
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        ⊢ HasSubset.Subset s (Function.support fun x => tsum fun n => HSMul.hSMul (r n …
      -/
    · intro x hx
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₂
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Membership.mem s x
        ⊢ Membership.mem (Function.support fun x => tsum fun n => HSMul.hSMul (r n) (g …
      -/
      obtain ⟨n, hn⟩ : ∃ n, x ∈ support (g n) := s_g x hx
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₂.intro
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Membership.mem s x
        n : Nat
        hn : Membership.mem (Function.support (g n)) x
        ⊢ Membership.mem (Function.support fun x => tsum fun n => HSMul.hSMul (r n) (g …
      -/
      have I : 0 < r n * g n x := mul_pos (rpos n) (lt_of_le_of_ne (g_nonneg n x) (Ne.symm hn))
      /-
        case inr.intro.intro.intro.intro.intro.intro.intro.refine_1.h₂.intro
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        s : Set E
        hs : IsOpen s
        h's : s.Nonempty
        ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
        T : Set ι
        T_count : T.Countable
        hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
        g0 : Nat → ι
        hg : Eq T (Set.range g0)
        g : Nat → E → Real := fun n => ↑(g0 n)
        g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
        s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
        g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
        g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
        g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
        δ : Nat → NNReal
        δpos : ∀ (i : Nat), LT.lt 0 (δ i)
        c : NNReal
        δc : HasSum δ c
        c_lt : LT.lt c 1
        r : Nat → Real
        rpos : ∀ (n : Nat), LT.lt 0 (r n)
        hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
        S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
        x : E
        hx : Membership.mem s x
        n : Nat
        hn : Membership.mem (Function.support (g n)) x
        I : LT.lt 0 (HMul.hMul (r n) (g n x))
        ⊢ Membership.mem (Function.support fun x => tsum fun n => HSMul.hSMul (r n) (g …
      -/
      exact ne_of_gt (tsum_pos (S x) (fun i => mul_nonneg (rpos i).le (g_nonneg i x)) n I)
      /-
        🎉 no goals
      -/
  · refine
      contDiff_tsum_of_eventually (fun n => (g_smooth n).const_smul (r n))
        (fun k _ => (NNReal.hasSum_coe.2 δc).summable) ?_
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      ⊢ ∀ (k : Nat), LE.le (↑k) Top.top → Filter.Eventually (fun i => ∀ (x : E), LE. …
    -/
    intro i _
    simp only [Nat.cofinite_eq_atTop, Pi.smul_apply, Algebra.id.smul_eq_mul,
      Filter.eventually_atTop]
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      i : Nat
      a✝ : LE.le (↑i) Top.top
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → ∀ (x : E), LE.le (Norm.norm (iterat …
    -/
    exact ⟨i, fun n hn x => hr _ _ hn _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_3
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      ⊢ HasSubset.Subset (Set.range fun x => tsum fun n => HSMul.hSMul (r n) (g n) x …
    -/
  · rintro - ⟨y, rfl⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_3.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      ⊢ Membership.mem (Set.Icc 0 1) ((fun x => tsum fun n => HSMul.hSMul (r n) (g n …
    -/
    refine ⟨tsum_nonneg fun n => mul_nonneg (rpos n).le (g_nonneg n y), le_trans ?_ c_lt.le⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_3.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      ⊢ LE.le ((fun x => tsum fun n => HSMul.hSMul (r n) (g n) x) y) ((fun a => ↑a) c)
    -/
    have A : HasSum (fun n => (δ n : ℝ)) c := NNReal.hasSum_coe.2 δc
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_3.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      A : HasSum (fun n => ↑(δ n)) ↑c
      ⊢ LE.le ((fun x => tsum fun n => HSMul.hSMul (r n) (g n) x) y) ((fun a => ↑a) c)
    -/
    simp only [Pi.smul_apply, smul_eq_mul, NNReal.val_eq_coe, ← A.tsum_eq]
    /-
      case inr.intro.intro.intro.intro.intro.intro.intro.refine_3.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      A : HasSum (fun n => ↑(δ n)) ↑c
      ⊢ LE.le (tsum fun n => HMul.hMul (r n) (g n y)) (tsum fun b => ↑(δ b))
    -/
    apply tsum_le_tsum _ (S y) A.summable
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      A : HasSum (fun n => ↑(δ n)) ↑c
      ⊢ ∀ (i : Nat), LE.le (HSMul.hSMul (r i) (g i) y) ↑(δ i)
    -/
    intro n
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      A : HasSum (fun n => ↑(δ n)) ↑c
      n : Nat
      ⊢ LE.le (HSMul.hSMul (r n) (g n) y) ↑(δ n)
    -/
    apply (le_abs_self _).trans
    /-
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      s : Set E
      hs : IsOpen s
      h's : s.Nonempty
      ι : Type (max 0 u_1) := Subtype fun f => And (HasSubset.Subset (Function.suppo …
      T : Set ι
      T_count : T.Countable
      hT : Eq (Set.iUnion fun f => Set.iUnion fun h => Function.support ↑f) s
      g0 : Nat → ι
      hg : Eq T (Set.range g0)
      g : Nat → E → Real := fun n => ↑(g0 n)
      g_s : ∀ (n : Nat), HasSubset.Subset (Function.support (g n)) s
      s_g : ∀ (x : E), Membership.mem s x → Exists fun n => Membership.mem (Function …
      g_smooth : ∀ (n : Nat), ContDiff Real (↑Top.top) (g n)
      g_comp_supp : ∀ (n : Nat), HasCompactSupport (g n)
      g_nonneg : ∀ (n : Nat) (x : E), LE.le 0 (g n x)
      δ : Nat → NNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      c : NNReal
      δc : HasSum δ c
      c_lt : LT.lt c 1
      r : Nat → Real
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      hr : ∀ (n i : Nat), LE.le i n → ∀ (x : E), LE.le (Norm.norm (iteratedFDeriv Re …
      S : ∀ (x : E), Summable fun n => HSMul.hSMul (r n) (g n) x
      y : E
      A : HasSum (fun n => ↑(δ n)) ↑c
      n : Nat
      ⊢ LE.le (abs (HSMul.hSMul (r n) (g n) y)) ↑(δ n)
    -/
    simpa only [norm_iteratedFDeriv_zero] using hr n 0 (zero_le n) y
    /-
      🎉 no goals
    -/


/-- An auxiliary function to construct partitions of unity on finite-dimensional real vector spaces.
It is the characteristic function of the closed unit ball. -/
def φ : E → ℝ :=
  (closedBall (0 : E) 1).indicator fun _ => (1 : ℝ)


theorem u_exists :
    ∃ u : E → ℝ,
      ContDiff ℝ ∞ u ∧ (∀ x, u x ∈ Icc (0 : ℝ) 1) ∧ support u = ball 0 1 ∧ ∀ x, u (-x) = u x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    ⊢ Exists fun u => And (ContDiff Real (↑Top.top) u) (And (∀ (x : E), Membership …
  -/
  have A : IsOpen (ball (0 : E) 1) := isOpen_ball
  obtain ⟨f, f_support, f_smooth, f_range⟩ :
      ∃ f : E → ℝ, f.support = ball (0 : E) 1 ∧ ContDiff ℝ ∞ f ∧ Set.range f ⊆ Set.Icc 0 1 :=
    A.exists_smooth_support_eq
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    A : IsOpen (Metric.ball 0 1)
    f : E → Real
    f_support : Eq (Function.support f) (Metric.ball 0 1)
    f_smooth : ContDiff Real (↑Top.top) f
    f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    ⊢ Exists fun u => And (ContDiff Real (↑Top.top) u) (And (∀ (x : E), Membership …
  -/
  have B : ∀ x, f x ∈ Icc (0 : ℝ) 1 := fun x => f_range (mem_range_self x)
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    A : IsOpen (Metric.ball 0 1)
    f : E → Real
    f_support : Eq (Function.support f) (Metric.ball 0 1)
    f_smooth : ContDiff Real (↑Top.top) f
    f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
    ⊢ Exists fun u => And (ContDiff Real (↑Top.top) u) (And (∀ (x : E), Membership …
  -/
  refine ⟨fun x => (f x + f (-x)) / 2, ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ ContDiff Real ↑Top.top fun x => HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2
    -/
  · exact (f_smooth.add (f_smooth.comp contDiff_neg)).div_const _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ ∀ (x : E), Membership.mem (Set.Icc 0 1) ((fun x => HDiv.hDiv (HAdd.hAdd (f x …
    -/
  · intro x
    /-
      case intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      x : E
      ⊢ Membership.mem (Set.Icc 0 1) ((fun x => HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.n …
    -/
    simp only [mem_Icc]
    /-
      case intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      x : E
      ⊢ And (LE.le 0 (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2)) (LE.le (HDiv.h …
    -/
    constructor
      /-
        case intro.intro.intro.refine_2.left
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        ⊢ LE.le 0 (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2)
      -/
    · linarith [(B x).1, (B (-x)).1]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2.right
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) 1
      -/
    · linarith [(B x).2, (B (-x)).2]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.refine_3
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ Eq (Function.support fun x => HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) …
    -/
  · refine support_eq_iff.2 ⟨fun x hx => ?_, fun x hx => ?_⟩
      /-
        case intro.intro.intro.refine_3.refine_1
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        hx : Membership.mem (Metric.ball 0 1) x
        ⊢ Ne (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) 0
      -/
    · apply ne_of_gt
      have : 0 < f x := by
        apply lt_of_le_of_ne (B x).1 (Ne.symm _)
        rwa [← f_support] at hx
      /-
        case intro.intro.intro.refine_3.refine_1.h
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        hx : Membership.mem (Metric.ball 0 1) x
        this : LT.lt 0 (f x)
        ⊢ LT.lt 0 (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2)
      -/
      linarith [(B (-x)).1]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_3.refine_2
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        hx : Not (Membership.mem (Metric.ball 0 1) x)
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) 0
      -/
    · have I1 : x ∉ support f := by rwa [f_support]
      have I2 : -x ∉ support f := by
        rw [f_support]
        simpa using hx
      /-
        case intro.intro.intro.refine_3.refine_2
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        hx : Not (Membership.mem (Metric.ball 0 1) x)
        I1 : Not (Membership.mem (Function.support f) x)
        I2 : Not (Membership.mem (Function.support f) (Neg.neg x))
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) 0
      -/
      simp only [mem_support, Classical.not_not] at I1 I2
      /-
        case intro.intro.intro.refine_3.refine_2
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : FiniteDimensional Real E
        A : IsOpen (Metric.ball 0 1)
        f : E → Real
        f_support : Eq (Function.support f) (Metric.ball 0 1)
        f_smooth : ContDiff Real (↑Top.top) f
        f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
        B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
        x : E
        hx : Not (Membership.mem (Metric.ball 0 1) x)
        I1 : Eq (f x) 0
        I2 : Eq (f (Neg.neg x)) 0
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) 0
      -/
      simp only [I1, I2, add_zero, zero_div]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.refine_4
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      A : IsOpen (Metric.ball 0 1)
      f : E → Real
      f_support : Eq (Function.support f) (Metric.ball 0 1)
      f_smooth : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      B : ∀ (x : E), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ ∀ (x : E), Eq ((fun x => HDiv.hDiv (HAdd.hAdd (f x) (f (Neg.neg x))) 2) (Neg …
    -/
  · intro x; simp only [add_comm, neg_neg]
             /-
               🎉 no goals
             -/


/-- An auxiliary function to construct partitions of unity on finite-dimensional real vector spaces,
which is smooth, symmetric, and with support equal to the unit ball. -/
def u (x : E) : ℝ :=
  Classical.choose (u_exists E) x


theorem u_smooth : ContDiff ℝ ∞ (u : E → ℝ) :=
  (Classical.choose_spec (u_exists E)).1


theorem u_continuous : Continuous (u : E → ℝ) :=
  (u_smooth E).continuous


theorem u_support : support (u : E → ℝ) = ball 0 1 :=
  (Classical.choose_spec (u_exists E)).2.2.1


theorem u_compact_support : HasCompactSupport (u : E → ℝ) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    ⊢ HasCompactSupport ExistsContDiffBumpBase.u
  -/
  rw [hasCompactSupport_def, u_support, closure_ball (0 : E) one_ne_zero]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    ⊢ IsCompact (Metric.closedBall 0 1)
  -/
  exact isCompact_closedBall _ _
  /-
    🎉 no goals
  -/


theorem u_nonneg (x : E) : 0 ≤ u x :=
  ((Classical.choose_spec (u_exists E)).2.1 x).1


theorem u_le_one (x : E) : u x ≤ 1 :=
  ((Classical.choose_spec (u_exists E)).2.1 x).2


theorem u_neg (x : E) : u (-x) = u x :=
  (Classical.choose_spec (u_exists E)).2.2.2 x


local notation "μ" => MeasureTheory.Measure.addHaar


theorem u_int_pos : 0 < ∫ x : E, u x ∂μ := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    ⊢ LT.lt 0 (MeasureTheory.integral MeasureTheory.Measure.addHaar fun x => Exist …
  -/
  refine (integral_pos_iff_support_of_nonneg u_nonneg ?_).mpr ?_
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      ⊢ MeasureTheory.Integrable ExistsContDiffBumpBase.u MeasureTheory.Measure.addH …
    -/
  · exact (u_continuous E).integrable_of_hasCompactSupport (u_compact_support E)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      ⊢ LT.lt 0 (MeasureTheory.Measure.addHaar (Function.support ExistsContDiffBumpB …
    -/
  · rw [u_support]; exact measure_ball_pos _ _ zero_lt_one
                    /-
                      🎉 no goals
                    -/


/-- An auxiliary function to construct partitions of unity on finite-dimensional real vector spaces,
which is smooth, symmetric, with support equal to the ball of radius `D` and integral `1`. -/
def w (D : ℝ) (x : E) : ℝ :=
  ((∫ x : E, u x ∂μ) * |D| ^ finrank ℝ E)⁻¹ • u (D⁻¹ • x)


theorem w_def (D : ℝ) :
    (w D : E → ℝ) = fun x => ((∫ x : E, u x ∂μ) * |D| ^ finrank ℝ E)⁻¹ • u (D⁻¹ • x) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    ⊢ Eq (ExistsContDiffBumpBase.w D) fun x => HSMul.hSMul (Inv.inv (HMul.hMul (Me …
  -/
  ext1 x; rfl
          /-
            🎉 no goals
          -/


theorem w_nonneg (D : ℝ) (x : E) : 0 ≤ w D x := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ LE.le 0 (ExistsContDiffBumpBase.w D x)
  -/
  apply mul_nonneg _ (u_nonneg _)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ LE.le 0 (Inv.inv (HMul.hMul (MeasureTheory.integral MeasureTheory.Measure.ad …
  -/
  apply inv_nonneg.2
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ LE.le 0 (HMul.hMul (MeasureTheory.integral MeasureTheory.Measure.addHaar fun …
  -/
  apply mul_nonneg (u_int_pos E).le
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ LE.le 0 (HPow.hPow (abs D) (Module.finrank Real E))
  -/
  norm_cast
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ LE.le 0 (HPow.hPow (abs D) (Module.finrank Real E))
  -/
  apply pow_nonneg (abs_nonneg D)
  /-
    🎉 no goals
  -/


theorem w_mul_φ_nonneg (D : ℝ) (x y : E) : 0 ≤ w D y * φ (x - y) :=
                                                  /-
                                                    E : Type u_1
                                                    inst✝⁴ : NormedAddCommGroup E
                                                    inst✝³ : NormedSpace Real E
                                                    inst✝² : FiniteDimensional Real E
                                                    inst✝¹ : MeasurableSpace E
                                                    inst✝ : BorelSpace E
                                                    D : Real
                                                    x y : E
                                                    ⊢ ∀ (a : E), Membership.mem (Metric.closedBall 0 1) a → LE.le 0 1
                                                  -/
  mul_nonneg (w_nonneg D y) (indicator_nonneg (by simp only [zero_le_one, imp_true_iff]) _)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem w_integral {D : ℝ} (Dpos : 0 < D) : ∫ x : E, w D x ∂μ = 1 := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    Dpos : LT.lt 0 D
    ⊢ Eq (MeasureTheory.integral MeasureTheory.Measure.addHaar fun x => ExistsCont …
  -/
  simp_rw [w, integral_smul]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    Dpos : LT.lt 0 D
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (MeasureTheory.integral MeasureTheory.Me …
  -/
  rw [integral_comp_inv_smul_of_nonneg μ (u : E → ℝ) Dpos.le, abs_of_nonneg Dpos.le, mul_comm]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    Dpos : LT.lt 0 D
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HPow.hPow D (Module.finrank Real E)) (M …
  -/
  field_simp [(u_int_pos E).ne']
  /-
    🎉 no goals
  -/


theorem w_support {D : ℝ} (Dpos : 0 < D) : support (w D : E → ℝ) = ball 0 D := by
  have B : D • ball (0 : E) 1 = ball 0 D := by
    rw [smul_unitBall Dpos.ne', Real.norm_of_nonneg Dpos.le]
  have C : D ^ finrank ℝ E ≠ 0 := by
    norm_cast
    exact pow_ne_zero _ Dpos.ne'
  simp only [w_def, Algebra.id.smul_eq_mul, support_mul, support_inv, univ_inter,
    support_comp_inv_smul₀ Dpos.ne', u_support, B, support_const (u_int_pos E).ne', support_const C,
    abs_of_nonneg Dpos.le]


theorem w_compact_support {D : ℝ} (Dpos : 0 < D) : HasCompactSupport (w D : E → ℝ) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    Dpos : LT.lt 0 D
    ⊢ HasCompactSupport (ExistsContDiffBumpBase.w D)
  -/
  rw [hasCompactSupport_def, w_support E Dpos, closure_ball (0 : E) Dpos.ne']
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    Dpos : LT.lt 0 D
    ⊢ IsCompact (Metric.closedBall 0 D)
  -/
  exact isCompact_closedBall _ _
  /-
    🎉 no goals
  -/


/-- An auxiliary function to construct partitions of unity on finite-dimensional real vector spaces.
It is the convolution between a smooth function of integral `1` supported in the ball of radius `D`,
with the indicator function of the closed unit ball. Therefore, it is smooth, equal to `1` on the
ball of radius `1 - D`, with support equal to the ball of radius `1 + D`. -/
def y (D : ℝ) : E → ℝ :=
  w D ⋆[lsmul ℝ ℝ, μ] φ


theorem y_neg (D : ℝ) (x : E) : y D (-x) = y D x := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    ⊢ Eq (ExistsContDiffBumpBase.y D (Neg.neg x)) (ExistsContDiffBumpBase.y D x)
  -/
  apply convolution_neg_of_neg_eq
    /-
      case h1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      ⊢ Filter.Eventually (fun x => Eq (ExistsContDiffBumpBase.w D (Neg.neg x)) (Exi …
    -/
  · filter_upwards with x
    /-
      case h1.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x✝ x : E
      ⊢ Eq (ExistsContDiffBumpBase.w D (Neg.neg x)) (ExistsContDiffBumpBase.w D x)
    -/
    simp only [w_def, Real.rpow_natCast, mul_inv_rev, smul_neg, u_neg, smul_eq_mul, forall_const]
    /-
      🎉 no goals
    -/
    /-
      case h2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      ⊢ Filter.Eventually (fun x => Eq (ExistsContDiffBumpBase.φ (Neg.neg x)) (Exist …
    -/
  · filter_upwards with x
    /-
      case h2.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x✝ x : E
      ⊢ Eq (ExistsContDiffBumpBase.φ (Neg.neg x)) (ExistsContDiffBumpBase.φ x)
    -/
    simp only [φ, indicator, mem_closedBall, dist_zero_right, norm_neg, forall_const]
    /-
      🎉 no goals
    -/


theorem y_eq_one_of_mem_closedBall {D : ℝ} {x : E} (Dpos : 0 < D)
    (hx : x ∈ closedBall (0 : E) (1 - D)) : y D x = 1 := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Membership.mem (Metric.closedBall 0 (HSub.hSub 1 D)) x
    ⊢ Eq (ExistsContDiffBumpBase.y D x) 1
  -/
  change (w D ⋆[lsmul ℝ ℝ, μ] φ) x = 1
  have B : ∀ y : E, y ∈ ball x D → φ y = 1 := by
    have C : ball x D ⊆ ball 0 1 := by
      apply ball_subset_ball'
      simp only [mem_closedBall] at hx
      linarith only [hx]
    intro y hy
    simp only [φ, indicator, mem_closedBall, ite_eq_left_iff, not_le, zero_ne_one]
    intro h'y
    linarith only [mem_ball.1 (C hy), h'y]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Membership.mem (Metric.closedBall 0 (HSub.hSub 1 D)) x
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  have Bx : φ x = 1 := B _ (mem_ball_self Dpos)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Membership.mem (Metric.closedBall 0 (HSub.hSub 1 D)) x
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    Bx : Eq (ExistsContDiffBumpBase.φ x) 1
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  have B' : ∀ y, y ∈ ball x D → φ y = φ x := by rw [Bx]; exact B
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Membership.mem (Metric.closedBall 0 (HSub.hSub 1 D)) x
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    Bx : Eq (ExistsContDiffBumpBase.φ x) 1
    B' : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBas …
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  rw [convolution_eq_right' _ (le_of_eq (w_support E Dpos)) B']
  simp only [lsmul_apply, Algebra.id.smul_eq_mul, integral_mul_right, w_integral E Dpos, Bx,
    one_mul]


theorem y_eq_zero_of_not_mem_ball {D : ℝ} {x : E} (Dpos : 0 < D) (hx : x ∉ ball (0 : E) (1 + D)) :
    y D x = 0 := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Not (Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x)
    ⊢ Eq (ExistsContDiffBumpBase.y D x) 0
  -/
  change (w D ⋆[lsmul ℝ ℝ, μ] φ) x = 0
  have B : ∀ y, y ∈ ball x D → φ y = 0 := by
    intro y hy
    simp only [φ, indicator, mem_closedBall_zero_iff, ite_eq_right_iff, one_ne_zero]
    intro h'y
    have C : ball y D ⊆ ball 0 (1 + D) := by
      apply ball_subset_ball'
      rw [← dist_zero_right] at h'y
      linarith only [h'y]
    exact hx (C (mem_ball_comm.1 hy))
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Not (Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x)
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  have Bx : φ x = 0 := B _ (mem_ball_self Dpos)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Not (Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x)
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    Bx : Eq (ExistsContDiffBumpBase.φ x) 0
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  have B' : ∀ y, y ∈ ball x D → φ y = φ x := by rw [Bx]; exact B
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Not (Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x)
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    Bx : Eq (ExistsContDiffBumpBase.φ x) 0
    B' : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBas …
    ⊢ Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDiffBum …
  -/
  rw [convolution_eq_right' _ (le_of_eq (w_support E Dpos)) B']
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    hx : Not (Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x)
    B : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBase …
    Bx : Eq (ExistsContDiffBumpBase.φ x) 0
    B' : ∀ (y : E), Membership.mem (Metric.ball x D) y → Eq (ExistsContDiffBumpBas …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.Measure.addHaar fun t => ((Continuo …
  -/
  simp only [lsmul_apply, Algebra.id.smul_eq_mul, Bx, mul_zero, integral_const]
  /-
    🎉 no goals
  -/


theorem y_nonneg (D : ℝ) (x : E) : 0 ≤ y D x :=
  integral_nonneg (w_mul_φ_nonneg D x)


theorem y_le_one {D : ℝ} (x : E) (Dpos : 0 < D) : y D x ≤ 1 := by
  have A : (w D ⋆[lsmul ℝ ℝ, μ] φ) x ≤ (w D ⋆[lsmul ℝ ℝ, μ] 1) x := by
    apply
      convolution_mono_right_of_nonneg _ (w_nonneg D) (indicator_le_self' fun x _ => zero_le_one)
        fun _ => zero_le_one
    refine
      (HasCompactSupport.convolutionExistsLeft _ (w_compact_support E Dpos) ?_
          (locallyIntegrable_const (1 : ℝ)) x).integrable
    exact continuous_const.mul ((u_continuous E).comp (continuous_id.const_smul _))
  have B : (w D ⋆[lsmul ℝ ℝ, μ] fun _ => (1 : ℝ)) x = 1 := by
    simp only [convolution, ContinuousLinearMap.map_smul, mul_inv_rev, coe_smul', mul_one,
      lsmul_apply, Algebra.id.smul_eq_mul, integral_mul_left, w_integral E Dpos, Pi.smul_apply]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    A : LE.le (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) ExistsContDi …
    B : Eq (MeasureTheory.convolution (ExistsContDiffBumpBase.w D) (fun x => 1) (C …
    ⊢ LE.le (ExistsContDiffBumpBase.y D x) 1
  -/
  exact A.trans (le_of_eq B)
  /-
    🎉 no goals
  -/


theorem y_pos_of_mem_ball {D : ℝ} {x : E} (Dpos : 0 < D) (D_lt_one : D < 1)
    (hx : x ∈ ball (0 : E) (1 + D)) : 0 < y D x := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    D_lt_one : LT.lt D 1
    hx : Membership.mem (Metric.ball 0 (HAdd.hAdd 1 D)) x
    ⊢ LT.lt 0 (ExistsContDiffBumpBase.y D x)
  -/
  simp only [mem_ball_zero_iff] at hx
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    D : Real
    x : E
    Dpos : LT.lt 0 D
    D_lt_one : LT.lt D 1
    hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
    ⊢ LT.lt 0 (ExistsContDiffBumpBase.y D x)
  -/
  refine (integral_pos_iff_support_of_nonneg (w_mul_φ_nonneg D x) ?_).2 ?_
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      ⊢ MeasureTheory.Integrable (fun i => HMul.hMul (ExistsContDiffBumpBase.w D i)  …
    -/
  · have F_comp : HasCompactSupport (w D) := w_compact_support E Dpos
    have B : LocallyIntegrable (φ : E → ℝ) μ :=
      (locallyIntegrable_const _).indicator measurableSet_closedBall
    have C : Continuous (w D : E → ℝ) :=
      continuous_const.mul ((u_continuous E).comp (continuous_id.const_smul _))
    exact
      (HasCompactSupport.convolutionExistsLeft (lsmul ℝ ℝ : ℝ →L[ℝ] ℝ →L[ℝ] ℝ) F_comp C B
          x).integrable
    /-
      case refine_2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      ⊢ LT.lt 0 (MeasureTheory.Measure.addHaar (Function.support fun i => HMul.hMul  …
    -/
  · set z := (D / (1 + D)) • x with hz
    /-
      case refine_2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      z : E := HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x
      hz : Eq z (HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x)
      ⊢ LT.lt 0 (MeasureTheory.Measure.addHaar (Function.support fun i => HMul.hMul  …
    -/
    have B : 0 < 1 + D := by linarith
    have C : ball z (D * (1 + D - ‖x‖) / (1 + D)) ⊆ support fun y : E => w D y * φ (x - y) := by
      intro y hy
      simp only [support_mul, w_support E Dpos]
      simp only [φ, mem_inter_iff, mem_support, Ne, indicator_apply_eq_zero,
        mem_closedBall_zero_iff, one_ne_zero, not_forall, not_false_iff, exists_prop, and_true]
      constructor
      · apply ball_subset_ball' _ hy
        simp only [hz, norm_smul, abs_of_nonneg Dpos.le, abs_of_nonneg B.le, dist_zero_right,
          Real.norm_eq_abs, abs_div]
        simp only [div_le_iff₀ B, field_simps]
        ring_nf
        rfl
      · have ID : ‖D / (1 + D) - 1‖ = 1 / (1 + D) := by
          rw [Real.norm_of_nonpos]
          · simp only [B.ne', Ne, not_false_iff, mul_one, neg_sub, add_tsub_cancel_right,
              field_simps]
          · simp only [B.ne', Ne, not_false_iff, mul_one, field_simps]
            apply div_nonpos_of_nonpos_of_nonneg _ B.le
            linarith only
        rw [← mem_closedBall_iff_norm']
        apply closedBall_subset_closedBall' _ (ball_subset_closedBall hy)
        rw [← one_smul ℝ x, dist_eq_norm, hz, ← sub_smul, one_smul, norm_smul, ID]
        simp only [B.ne', div_le_iff₀ B, field_simps]
        nlinarith only [hx, D_lt_one]
    /-
      case refine_2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      z : E := HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x
      hz : Eq z (HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x)
      B : LT.lt 0 (HAdd.hAdd 1 D)
      C : HasSubset.Subset (Metric.ball z (HDiv.hDiv (HMul.hMul D (HSub.hSub (HAdd.h …
      ⊢ LT.lt 0 (MeasureTheory.Measure.addHaar (Function.support fun i => HMul.hMul  …
    -/
    apply lt_of_lt_of_le _ (measure_mono C)
    /-
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      z : E := HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x
      hz : Eq z (HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x)
      B : LT.lt 0 (HAdd.hAdd 1 D)
      C : HasSubset.Subset (Metric.ball z (HDiv.hDiv (HMul.hMul D (HSub.hSub (HAdd.h …
      ⊢ LT.lt 0 (MeasureTheory.Measure.addHaar (Metric.ball z (HDiv.hDiv (HMul.hMul  …
    -/
    apply measure_ball_pos
    /-
      case hr
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      D : Real
      x : E
      Dpos : LT.lt 0 D
      D_lt_one : LT.lt D 1
      hx : LT.lt (Norm.norm x) (HAdd.hAdd 1 D)
      z : E := HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x
      hz : Eq z (HSMul.hSMul (HDiv.hDiv D (HAdd.hAdd 1 D)) x)
      B : LT.lt 0 (HAdd.hAdd 1 D)
      C : HasSubset.Subset (Metric.ball z (HDiv.hDiv (HMul.hMul D (HSub.hSub (HAdd.h …
      ⊢ LT.lt 0 (HDiv.hDiv (HMul.hMul D (HSub.hSub (HAdd.hAdd 1 D) (Norm.norm x))) ( …
    -/
    exact div_pos (mul_pos Dpos (by linarith only [hx])) B
    /-
      🎉 no goals
    -/


theorem y_smooth : ContDiffOn ℝ ∞ (uncurry y) (Ioo (0 : ℝ) 1 ×ˢ (univ : Set E)) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    ⊢ ContDiffOn Real (↑Top.top) (Function.uncurry ExistsContDiffBumpBase.y) (SPro …
  -/
  have hs : IsOpen (Ioo (0 : ℝ) (1 : ℝ)) := isOpen_Ioo
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    hs : IsOpen (Set.Ioo 0 1)
    ⊢ ContDiffOn Real (↑Top.top) (Function.uncurry ExistsContDiffBumpBase.y) (SPro …
  -/
  have hk : IsCompact (closedBall (0 : E) 1) := ProperSpace.isCompact_closedBall _ _
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    hs : IsOpen (Set.Ioo 0 1)
    hk : IsCompact (Metric.closedBall 0 1)
    ⊢ ContDiffOn Real (↑Top.top) (Function.uncurry ExistsContDiffBumpBase.y) (SPro …
  -/
  refine contDiffOn_convolution_left_with_param (lsmul ℝ ℝ) hs hk ?_ ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      ⊢ ∀ (p : Real) (x : E), Membership.mem (Set.Ioo 0 1) p → Not (Membership.mem ( …
    -/
  · rintro p x hp hx
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Not (Membership.mem (Metric.closedBall 0 1) x)
      ⊢ Eq (ExistsContDiffBumpBase.w p x) 0
    -/
    simp only [w, mul_inv_rev, Algebra.id.smul_eq_mul, mul_eq_zero, inv_eq_zero]
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Not (Membership.mem (Metric.closedBall 0 1) x)
      ⊢ Or (Or (Eq (HPow.hPow (abs p) (Module.finrank Real E)) 0) (Eq (MeasureTheory …
    -/
    right
    /-
      case refine_1.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Not (Membership.mem (Metric.closedBall 0 1) x)
      ⊢ Eq (ExistsContDiffBumpBase.u (HSMul.hSMul (Inv.inv p) x)) 0
    -/
    contrapose! hx
    /-
      case refine_1.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Ne (ExistsContDiffBumpBase.u (HSMul.hSMul (Inv.inv p) x)) 0
      ⊢ Membership.mem (Metric.closedBall 0 1) x
    -/
    have : p⁻¹ • x ∈ support u := mem_support.2 hx
    simp only [u_support, norm_smul, mem_ball_zero_iff, Real.norm_eq_abs, abs_inv,
      abs_of_nonneg hp.1.le, ← div_eq_inv_mul, div_lt_one hp.1] at this
    /-
      case refine_1.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Ne (ExistsContDiffBumpBase.u (HSMul.hSMul (Inv.inv p) x)) 0
      this : LT.lt (Norm.norm x) p
      ⊢ Membership.mem (Metric.closedBall 0 1) x
    -/
    rw [mem_closedBall_zero_iff]
    /-
      case refine_1.h
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      p : Real
      x : E
      hp : Membership.mem (Set.Ioo 0 1) p
      hx : Ne (ExistsContDiffBumpBase.u (HSMul.hSMul (Inv.inv p) x)) 0
      this : LT.lt (Norm.norm x) p
      ⊢ LE.le (Norm.norm x) 1
    -/
    exact this.le.trans hp.2.le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      ⊢ MeasureTheory.LocallyIntegrable ExistsContDiffBumpBase.φ MeasureTheory.Measu …
    -/
  · exact (locallyIntegrable_const _).indicator measurableSet_closedBall
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      hs : IsOpen (Set.Ioo 0 1)
      hk : IsCompact (Metric.closedBall 0 1)
      ⊢ ContDiffOn Real (↑Top.top) (Function.HasUncurry.uncurry ExistsContDiffBumpBa …
    -/
  · apply ContDiffOn.mul
      /-
        case refine_3.hf
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun x => Inv.inv (HMul.hMul (MeasureTheory.integ …
      -/
    · norm_cast
      refine
        (contDiffOn_const.mul ?_).inv fun x hx =>
          ne_of_gt (mul_pos (u_int_pos E) (pow_pos (abs_pos_of_pos hx.1.1) (finrank ℝ E)))
      /-
        case refine_3.hf
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun x => HPow.hPow (abs x.1) (Module.finrank Rea …
      -/
      apply ContDiffOn.pow
      /-
        case refine_3.hf.hf
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun y => abs y.1) (SProd.sprod (Set.Ioo 0 1) Set …
      -/
      simp_rw [← Real.norm_eq_abs]
      /-
        case refine_3.hf.hf
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun y => Norm.norm y.1) (SProd.sprod (Set.Ioo 0  …
      -/
      apply ContDiffOn.norm ℝ
        /-
          case refine_3.hf.hf.hf
          E : Type u_1
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace Real E
          inst✝² : FiniteDimensional Real E
          inst✝¹ : MeasurableSpace E
          inst✝ : BorelSpace E
          hs : IsOpen (Set.Ioo 0 1)
          hk : IsCompact (Metric.closedBall 0 1)
          ⊢ ContDiffOn Real (↑Top.top) Prod.fst (SProd.sprod (Set.Ioo 0 1) Set.univ)
        -/
      · exact contDiffOn_fst
        /-
          🎉 no goals
        -/
        /-
          case refine_3.hf.hf.h0
          E : Type u_1
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace Real E
          inst✝² : FiniteDimensional Real E
          inst✝¹ : MeasurableSpace E
          inst✝ : BorelSpace E
          hs : IsOpen (Set.Ioo 0 1)
          hk : IsCompact (Metric.closedBall 0 1)
          ⊢ ∀ (x : Prod Real E), Membership.mem (SProd.sprod (Set.Ioo 0 1) Set.univ) x → …
        -/
      · intro x hx; exact ne_of_gt hx.1.1
                    /-
                      🎉 no goals
                    -/
      /-
        case refine_3.hg
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun x => ExistsContDiffBumpBase.u (HSMul.hSMul ( …
      -/
    · apply (u_smooth E).comp_contDiffOn
      /-
        case refine_3.hg
        E : Type u_1
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        hs : IsOpen (Set.Ioo 0 1)
        hk : IsCompact (Metric.closedBall 0 1)
        ⊢ ContDiffOn Real (↑Top.top) (fun x => HSMul.hSMul (Inv.inv x.1) x.2) (SProd.s …
      -/
      exact ContDiffOn.smul (contDiffOn_fst.inv fun x hx => ne_of_gt hx.1.1) contDiffOn_snd
      /-
        🎉 no goals
      -/


theorem y_support {D : ℝ} (Dpos : 0 < D) (D_lt_one : D < 1) :
    support (y D : E → ℝ) = ball (0 : E) (1 + D) :=
  support_eq_iff.2
    ⟨fun _ hx => (y_pos_of_mem_ball Dpos D_lt_one hx).ne', fun _ hx =>
      y_eq_zero_of_not_mem_ball Dpos hx⟩


instance (priority := 100) {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] : HasContDiffBump E := by
  /-
    E✝ : Type u_1
    inst✝⁵ : NormedAddCommGroup E✝
    inst✝⁴ : NormedSpace Real E✝
    inst✝³ : FiniteDimensional Real E✝
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    ⊢ HasContDiffBump E
  -/
  refine ⟨⟨?_⟩⟩
  /-
    E✝ : Type u_1
    inst✝⁵ : NormedAddCommGroup E✝
    inst✝⁴ : NormedSpace Real E✝
    inst✝³ : FiniteDimensional Real E✝
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    ⊢ ContDiffBumpBase E
  -/
  borelize E
  /-
    E✝ : Type u_1
    inst✝⁵ : NormedAddCommGroup E✝
    inst✝⁴ : NormedSpace Real E✝
    inst✝³ : FiniteDimensional Real E✝
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ ContDiffBumpBase E
  -/
  have IR : ∀ R : ℝ, 1 < R → 0 < (R - 1) / (R + 1) := by intro R hR; apply div_pos <;> linarith
  exact
    { toFun := fun R x => if 1 < R then y ((R - 1) / (R + 1)) (((R + 1) / 2)⁻¹ • x) else 0
      mem_Icc := fun R x => by
        simp only [mem_Icc]
        split_ifs with h
        · refine ⟨y_nonneg _ _, y_le_one _ (IR R h)⟩
        · simp only [le_refl, zero_le_one, and_self]
      symmetric := fun R x => by
        simp only
        split_ifs
        · simp only [y_neg, smul_neg]
        · rfl
      smooth := by
        suffices
          ContDiffOn ℝ ∞
            (uncurry y ∘ fun p : ℝ × E => ((p.1 - 1) / (p.1 + 1), ((p.1 + 1) / 2)⁻¹ • p.2))
            (Ioi 1 ×ˢ univ) by
          apply this.congr
          rintro ⟨R, x⟩ ⟨hR : 1 < R, _⟩
          simp only [hR, uncurry_apply_pair, if_true, Function.comp_apply]
        apply (y_smooth E).comp
        · apply ContDiffOn.prod
          · refine
              (contDiffOn_fst.sub contDiffOn_const).div (contDiffOn_fst.add contDiffOn_const) ?_
            rintro ⟨R, x⟩ ⟨hR : 1 < R, _⟩
            apply ne_of_gt
            dsimp only
            linarith
          · apply ContDiffOn.smul _ contDiffOn_snd
            refine ((contDiffOn_fst.add contDiffOn_const).div_const _).inv ?_
            rintro ⟨R, x⟩ ⟨hR : 1 < R, _⟩
            apply ne_of_gt
            dsimp only
            linarith
        · rintro ⟨R, x⟩ ⟨hR : 1 < R, _⟩
          have A : 0 < (R - 1) / (R + 1) := by apply div_pos <;> linarith
          have B : (R - 1) / (R + 1) < 1 := by apply (div_lt_one _).2 <;> linarith
          simp only [mem_preimage, prod_mk_mem_set_prod_eq, mem_Ioo, mem_univ, and_true, A, B]
      eq_one := fun R hR x hx => by
        have A : 0 < R + 1 := by linarith
        simp only [hR, if_true]
        apply y_eq_one_of_mem_closedBall (IR R hR)
        simp only [norm_smul, inv_div, mem_closedBall_zero_iff, Real.norm_eq_abs, abs_div, abs_two,
          abs_of_nonneg A.le]
        calc
          2 / (R + 1) * ‖x‖ ≤ 2 / (R + 1) := mul_le_of_le_one_right (by positivity) hx
          _ = 1 - (R - 1) / (R + 1) := by field_simp; ring
      support := fun R hR => by
        have A : 0 < (R + 1) / 2 := by linarith
        have C : (R - 1) / (R + 1) < 1 := by apply (div_lt_one _).2 <;> linarith
        simp only [hR, if_true, support_comp_inv_smul₀ A.ne', y_support _ (IR R hR) C,
          _root_.smul_ball A.ne', Real.norm_of_nonneg A.le, smul_zero]
        refine congr (congr_arg ball (Eq.refl 0)) ?_
        field_simp; ring }


