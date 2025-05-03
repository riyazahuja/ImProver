/-- A variant of Urysohn's lemma, `ℒ^p` version, for an outer regular measure `μ`:
consider two sets `s ⊆ u` which are respectively closed and open with `μ s < ∞`, and a vector `c`.
Then one may find a continuous function `f` equal to `c` on `s` and to `0` outside of `u`,
bounded by `‖c‖` everywhere, and such that the `ℒ^p` norm of `f - s.indicator (fun y ↦ c)` is
arbitrarily small. Additionally, this function `f` belongs to `ℒ^p`. -/
theorem exists_continuous_eLpNorm_sub_le_of_closed [μ.OuterRegular] (hp : p ≠ ∞) {s u : Set α}
    (s_closed : IsClosed s) (u_open : IsOpen u) (hsu : s ⊆ u) (hs : μ s ≠ ∞) (c : E) {ε : ℝ≥0∞}
    (hε : ε ≠ 0) :
    ∃ f : α → E,
      Continuous f ∧
        eLpNorm (fun x => f x - s.indicator (fun _y => c) x) p μ ≤ ε ∧
          (∀ x, ‖f x‖ ≤ ‖c‖) ∧ Function.support f ⊆ u ∧ Memℒp f p μ := by
  obtain ⟨η, η_pos, hη⟩ :
      ∃ η : ℝ≥0, 0 < η ∧ ∀ s : Set α, μ s ≤ η → eLpNorm (s.indicator fun _x => c) p μ ≤ ε :=
    exists_eLpNorm_indicator_le hp c hε
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ⊢ Exists fun f => And (Continuous f) (And (LE.le (MeasureTheory.eLpNorm (fun x …
  -/
  have ηpos : (0 : ℝ≥0∞) < η := ENNReal.coe_lt_coe.2 η_pos
  obtain ⟨V, sV, V_open, h'V, hV⟩ : ∃ (V : Set α), V ⊇ s ∧ IsOpen V ∧ μ V < ∞ ∧ μ (V \ s) < η :=
    s_closed.measurableSet.exists_isOpen_diff_lt hs ηpos.ne'
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ηpos : LT.lt 0 ↑η
    V : Set α
    sV : Superset V s
    V_open : IsOpen V
    h'V : LT.lt (μ V) Top.top
    hV : LT.lt (μ (SDiff.sdiff V s)) ↑η
    ⊢ Exists fun f => And (Continuous f) (And (LE.le (MeasureTheory.eLpNorm (fun x …
  -/
  let v := u ∩ V
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ηpos : LT.lt 0 ↑η
    V : Set α
    sV : Superset V s
    V_open : IsOpen V
    h'V : LT.lt (μ V) Top.top
    hV : LT.lt (μ (SDiff.sdiff V s)) ↑η
    v : Set α := Inter.inter u V
    ⊢ Exists fun f => And (Continuous f) (And (LE.le (MeasureTheory.eLpNorm (fun x …
  -/
  have hsv : s ⊆ v := subset_inter hsu sV
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ηpos : LT.lt 0 ↑η
    V : Set α
    sV : Superset V s
    V_open : IsOpen V
    h'V : LT.lt (μ V) Top.top
    hV : LT.lt (μ (SDiff.sdiff V s)) ↑η
    v : Set α := Inter.inter u V
    hsv : HasSubset.Subset s v
    ⊢ Exists fun f => And (Continuous f) (And (LE.le (MeasureTheory.eLpNorm (fun x …
  -/
  have hμv : μ v < ∞ := (measure_mono inter_subset_right).trans_lt h'V
  obtain ⟨g, hgv, hgs, hg_range⟩ :=
    exists_continuous_zero_one_of_isClosed (u_open.inter V_open).isClosed_compl s_closed
      (disjoint_compl_left_iff.2 hsv)
  -- Multiply this by `c` to get a continuous approximation to the function `f`; the key point is
  -- that this is pointwise bounded by the indicator of the set `v \ s`, which has small measure.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ηpos : LT.lt 0 ↑η
    V : Set α
    sV : Superset V s
    V_open : IsOpen V
    h'V : LT.lt (μ V) Top.top
    hV : LT.lt (μ (SDiff.sdiff V s)) ↑η
    v : Set α := Inter.inter u V
    hsv : HasSubset.Subset s v
    hμv : LT.lt (μ v) Top.top
    g : ContinuousMap α Real
    hgv : Set.EqOn (⇑g) 0 (HasCompl.compl (Inter.inter u V))
    hgs : Set.EqOn (⇑g) 1 s
    hg_range : ∀ (x : α), Membership.mem (Set.Icc 0 1) (g x)
    ⊢ Exists fun f => And (Continuous f) (And (LE.le (MeasureTheory.eLpNorm (fun x …
  -/
  have g_norm : ∀ x, ‖g x‖ = g x := fun x => by rw [Real.norm_eq_abs, abs_of_nonneg (hg_range x).1]
  have gc_bd0 : ∀ x, ‖g x • c‖ ≤ ‖c‖ := by
    intro x
    simp only [norm_smul, g_norm x]
    apply mul_le_of_le_one_left (norm_nonneg _)
    exact (hg_range x).2
  have gc_bd :
      ∀ x, ‖g x • c - s.indicator (fun _x => c) x‖ ≤ ‖(v \ s).indicator (fun _x => c) x‖ := by
    intro x
    by_cases hv : x ∈ v
    · rw [← Set.diff_union_of_subset hsv] at hv
      cases' hv with hsv hs
      · simpa only [hsv.2, Set.indicator_of_not_mem, not_false_iff, sub_zero, hsv,
          Set.indicator_of_mem] using gc_bd0 x
      · simp [hgs hs, hs]
    · simp [hgv hv, show x ∉ s from fun h => hv (hsv h)]
  have gc_support : (Function.support fun x : α => g x • c) ⊆ v := by
    refine Function.support_subset_iff'.2 fun x hx => ?_
    simp only [hgv hx, Pi.zero_apply, zero_smul]
  have gc_mem : Memℒp (fun x => g x • c) p μ := by
    refine Memℒp.smul_of_top_left (memℒp_top_const _) ?_
    refine ⟨g.continuous.aestronglyMeasurable, ?_⟩
    have : eLpNorm (v.indicator fun _x => (1 : ℝ)) p μ < ⊤ := by
      refine (eLpNorm_indicator_const_le _ _).trans_lt ?_
      simp only [lt_top_iff_ne_top, hμv.ne, nnnorm_one, ENNReal.coe_one, one_div, one_mul, Ne,
        ENNReal.rpow_eq_top_iff, inv_lt_zero, false_and, or_false, not_and, not_lt,
        ENNReal.toReal_nonneg, imp_true_iff]
    refine (eLpNorm_mono fun x => ?_).trans_lt this
    by_cases hx : x ∈ v
    · simp only [hx, abs_of_nonneg (hg_range x).1, (hg_range x).2, Real.norm_eq_abs,
        indicator_of_mem, CStarRing.norm_one]
    · simp only [hgv hx, Pi.zero_apply, Real.norm_eq_abs, abs_zero, abs_nonneg]
  refine
    ⟨fun x => g x • c, g.continuous.smul continuous_const, (eLpNorm_mono gc_bd).trans ?_, gc_bd0,
      gc_support.trans inter_subset_left, gc_mem⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.OuterRegular
    hp : Ne p Top.top
    s u : Set α
    s_closed : IsClosed s
    u_open : IsOpen u
    hsu : HasSubset.Subset s u
    hs : Ne (μ s) Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    η : NNReal
    η_pos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ηpos : LT.lt 0 ↑η
    V : Set α
    sV : Superset V s
    V_open : IsOpen V
    h'V : LT.lt (μ V) Top.top
    hV : LT.lt (μ (SDiff.sdiff V s)) ↑η
    v : Set α := Inter.inter u V
    hsv : HasSubset.Subset s v
    hμv : LT.lt (μ v) Top.top
    g : ContinuousMap α Real
    hgv : Set.EqOn (⇑g) 0 (HasCompl.compl (Inter.inter u V))
    hgs : Set.EqOn (⇑g) 1 s
    hg_range : ∀ (x : α), Membership.mem (Set.Icc 0 1) (g x)
    g_norm : ∀ (x : α), Eq (Norm.norm (g x)) (g x)
    gc_bd0 : ∀ (x : α), LE.le (Norm.norm (HSMul.hSMul (g x) c)) (Norm.norm c)
    gc_bd : ∀ (x : α), LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (g x) c) (s.indica …
    gc_support : HasSubset.Subset (Function.support fun x => HSMul.hSMul (g x) c) v
    gc_mem : MeasureTheory.Memℒp (fun x => HSMul.hSMul (g x) c) p μ
    ⊢ LE.le (MeasureTheory.eLpNorm ((SDiff.sdiff v s).indicator fun _x => c) p μ) ε
  -/
  exact hη _ ((measure_mono (diff_subset_diff inter_subset_right Subset.rfl)).trans hV.le)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias exists_continuous_snorm_sub_le_of_closed := exists_continuous_eLpNorm_sub_le_of_closed


/-- In a locally compact space, any function in `ℒp` can be approximated by compactly supported
continuous functions when `p < ∞`, version in terms of `eLpNorm`. -/
theorem Memℒp.exists_hasCompactSupport_eLpNorm_sub_le
    [R1Space α] [WeaklyLocallyCompactSpace α] [μ.Regular]
    (hp : p ≠ ∞) {f : α → E} (hf : Memℒp f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ g : α → E, HasCompactSupport g ∧ eLpNorm (f - g) p μ ≤ ε ∧ Continuous g ∧ Memℒp g p μ := by
  suffices H :
      ∃ g : α → E, eLpNorm (f - g) p μ ≤ ε ∧ Continuous g ∧ Memℒp g p μ ∧ HasCompactSupport g by
    rcases H with ⟨g, hg, g_cont, g_mem, g_support⟩
    exact ⟨g, g_support, hg, g_cont, g_mem⟩
  -- It suffices to check that the set of functions we consider approximates characteristic
  -- functions, is stable under addition and consists of ae strongly measurable functions.
  -- First check the latter easy facts.
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) ε) (A …
  -/
  apply hf.induction_dense hp _ _ _ _ hε
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : ENNRea …
  -/
  rotate_left
  -- stability under addition
    /-
      α : Type u_1
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : NormalSpace α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : BorelSpace α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝³ : NormedSpace Real E
      inst✝² : R1Space α
      inst✝¹ : WeaklyLocallyCompactSpace α
      inst✝ : μ.Regular
      hp : Ne p Top.top
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      ⊢ ∀ (f g : α → E), And (Continuous f) (And (MeasureTheory.Memℒp f p μ) (HasCom …
    -/
  · rintro f g ⟨f_cont, f_mem, hf⟩ ⟨g_cont, g_mem, hg⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : NormalSpace α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : BorelSpace α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝³ : NormedSpace Real E
      inst✝² : R1Space α
      inst✝¹ : WeaklyLocallyCompactSpace α
      inst✝ : μ.Regular
      hp : Ne p Top.top
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f g : α → E
      f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      hf : HasCompactSupport f
      g_cont : Continuous g
      g_mem : MeasureTheory.Memℒp g p μ
      hg : HasCompactSupport g
      ⊢ And (Continuous (HAdd.hAdd f g)) (And (MeasureTheory.Memℒp (HAdd.hAdd f g) p …
    -/
    exact ⟨f_cont.add g_cont, f_mem.add g_mem, hf.add hg⟩
    /-
      🎉 no goals
    -/
  -- ae strong measurability
    /-
      α : Type u_1
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : NormalSpace α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : BorelSpace α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝³ : NormedSpace Real E
      inst✝² : R1Space α
      inst✝¹ : WeaklyLocallyCompactSpace α
      inst✝ : μ.Regular
      hp : Ne p Top.top
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      ⊢ ∀ (f : α → E), And (Continuous f) (And (MeasureTheory.Memℒp f p μ) (HasCompa …
    -/
  · rintro f ⟨_f_cont, f_mem, _hf⟩
    /-
      case intro.intro
      α : Type u_1
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : NormalSpace α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : BorelSpace α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝³ : NormedSpace Real E
      inst✝² : R1Space α
      inst✝¹ : WeaklyLocallyCompactSpace α
      inst✝ : μ.Regular
      hp : Ne p Top.top
      f✝ : α → E
      hf : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f : α → E
      _f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      _hf : HasCompactSupport f
      ⊢ MeasureTheory.AEStronglyMeasurable f μ
    -/
    exact f_mem.aestronglyMeasurable
    /-
      🎉 no goals
    -/
  -- We are left with approximating characteristic functions.
  -- This follows from `exists_continuous_eLpNorm_sub_le_of_closed`.
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : ENNRea …
  -/
  intro c t ht htμ ε hε
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  rcases exists_Lp_half E μ p hε with ⟨δ, δpos, hδ⟩
  obtain ⟨η, ηpos, hη⟩ :
      ∃ η : ℝ≥0, 0 < η ∧ ∀ s : Set α, μ s ≤ η → eLpNorm (s.indicator fun _x => c) p μ ≤ δ :=
    exists_eLpNorm_indicator_le hp c δpos.ne'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  have hη_pos' : (0 : ℝ≥0∞) < η := ENNReal.coe_pos.2 ηpos
  obtain ⟨s, st, s_compact, s_closed, μs⟩ :
      ∃ s, s ⊆ t ∧ IsCompact s ∧ IsClosed s ∧ μ (t \ s) < η :=
    ht.exists_isCompact_isClosed_diff_lt htμ.ne hη_pos'.ne'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_compact : IsCompact s
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  have hsμ : μ s < ∞ := (measure_mono st).trans_lt htμ
  have I1 : eLpNorm ((s.indicator fun _y => c) - t.indicator fun _y => c) p μ ≤ δ := by
    rw [← eLpNorm_neg, neg_sub, ← indicator_diff st]
    exact hη _ μs.le
  obtain ⟨k, k_compact, sk⟩ : ∃ k : Set α, IsCompact k ∧ s ⊆ interior k :=
    exists_compact_superset s_compact
  rcases exists_continuous_eLpNorm_sub_le_of_closed hp s_closed isOpen_interior sk hsμ.ne c δpos.ne'
    with ⟨f, f_cont, I2, _f_bound, f_support, f_mem⟩
  have I3 : eLpNorm (f - t.indicator fun _y => c) p μ ≤ ε := by
    convert
      (hδ _ _
          (f_mem.aestronglyMeasurable.sub
            (aestronglyMeasurable_const.indicator s_closed.measurableSet))
          ((aestronglyMeasurable_const.indicator s_closed.measurableSet).sub
            (aestronglyMeasurable_const.indicator ht))
          I2 I1).le using 2
    simp only [sub_add_sub_cancel]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_compact : IsCompact s
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    k : Set α
    k_compact : IsCompact k
    sk : HasSubset.Subset s (interior k)
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    _f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_support : HasSubset.Subset (Function.support f) (interior k)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  refine ⟨f, I3, f_cont, f_mem, HasCompactSupport.intro k_compact fun x hx => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_compact : IsCompact s
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    k : Set α
    k_compact : IsCompact k
    sk : HasSubset.Subset s (interior k)
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    _f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_support : HasSubset.Subset (Function.support f) (interior k)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    x : α
    hx : Not (Membership.mem k x)
    ⊢ Eq (f x) 0
  -/
  rw [← Function.nmem_support]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_compact : IsCompact s
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    k : Set α
    k_compact : IsCompact k
    sk : HasSubset.Subset s (interior k)
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    _f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_support : HasSubset.Subset (Function.support f) (interior k)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    x : α
    hx : Not (Membership.mem k x)
    ⊢ Not (Membership.mem (Function.support f) x)
  -/
  contrapose! hx
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_compact : IsCompact s
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    k : Set α
    k_compact : IsCompact k
    sk : HasSubset.Subset s (interior k)
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    _f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_support : HasSubset.Subset (Function.support f) (interior k)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    x : α
    hx : Membership.mem (Function.support f) x
    ⊢ Membership.mem k x
  -/
  exact interior_subset (f_support hx)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.exists_hasCompactSupport_snorm_sub_le := Memℒp.exists_hasCompactSupport_eLpNorm_sub_le


/-- In a locally compact space, any function in `ℒp` can be approximated by compactly supported
continuous functions when `0 < p < ∞`, version in terms of `∫`. -/
theorem Memℒp.exists_hasCompactSupport_integral_rpow_sub_le
    [R1Space α] [WeaklyLocallyCompactSpace α] [μ.Regular]
    {p : ℝ} (hp : 0 < p) {f : α → E} (hf : Memℒp f (ENNReal.ofReal p) μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ g : α → E,
      HasCompactSupport g ∧
        (∫ x, ‖f x - g x‖ ^ p ∂μ) ≤ ε ∧ Continuous g ∧ Memℒp g (ENNReal.ofReal p) μ := by
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.integra …
  -/
  have I : 0 < ε ^ (1 / p) := Real.rpow_pos_of_pos hε _
  have A : ENNReal.ofReal (ε ^ (1 / p)) ≠ 0 := by
    simp only [Ne, ENNReal.ofReal_eq_zero, not_le, I]
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.integra …
  -/
  have B : ENNReal.ofReal p ≠ 0 := by simpa only [Ne, ENNReal.ofReal_eq_zero, not_le] using hp
  rcases hf.exists_hasCompactSupport_eLpNorm_sub_le ENNReal.coe_ne_top A with
    ⟨g, g_support, hg, g_cont, g_mem⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : α → E
    g_support : HasCompactSupport g
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) (↑p.toNNReal) μ) (ENNReal.of …
    g_cont : Continuous g
    g_mem : MeasureTheory.Memℒp g (↑p.toNNReal) μ
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.integra …
  -/
  change eLpNorm _ (ENNReal.ofReal p) _ ≤ _ at hg
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : α → E
    g_support : HasCompactSupport g
    g_cont : Continuous g
    g_mem : MeasureTheory.Memℒp g (↑p.toNNReal) μ
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) (ENNReal.ofReal p) μ) (ENNRe …
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.integra …
  -/
  refine ⟨g, g_support, ?_, g_cont, g_mem⟩
  rwa [(hf.sub g_mem).eLpNorm_eq_integral_rpow_norm B ENNReal.coe_ne_top,
    ENNReal.ofReal_le_ofReal_iff I.le, one_div, ENNReal.toReal_ofReal hp.le,
    Real.rpow_le_rpow_iff _ hε.le (inv_pos.2 hp)] at hg
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : α → E
    g_support : HasCompactSupport g
    g_cont : Continuous g
    g_mem : MeasureTheory.Memℒp g (↑p.toNNReal) μ
    hg : LE.le (HPow.hPow (MeasureTheory.integral μ fun a => HPow.hPow (Norm.norm  …
    ⊢ LE.le 0 (MeasureTheory.integral μ fun a => HPow.hPow (Norm.norm (HSub.hSub f …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- In a locally compact space, any integrable function can be approximated by compactly supported
continuous functions, version in terms of `∫⁻`. -/
theorem Integrable.exists_hasCompactSupport_lintegral_sub_le
    [R1Space α] [WeaklyLocallyCompactSpace α] [μ.Regular]
    {f : α → E} (hf : Integrable f μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ g : α → E,
      HasCompactSupport g ∧ (∫⁻ x, ‖f x - g x‖₊ ∂μ) ≤ ε ∧ Continuous g ∧ Integrable g μ := by
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.lintegr …
  -/
  simp only [← memℒp_one_iff_integrable, ← eLpNorm_one_eq_lintegral_nnnorm] at hf ⊢
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    f : α → E
    ε : ENNReal
    hε : Ne ε 0
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.eLpNorm …
  -/
  exact hf.exists_hasCompactSupport_eLpNorm_sub_le ENNReal.one_ne_top hε
  /-
    🎉 no goals
  -/


/-- In a locally compact space, any integrable function can be approximated by compactly supported
continuous functions, version in terms of `∫`. -/
theorem Integrable.exists_hasCompactSupport_integral_sub_le
    [R1Space α] [WeaklyLocallyCompactSpace α] [μ.Regular]
    {f : α → E} (hf : Integrable f μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ g : α → E, HasCompactSupport g ∧ (∫ x, ‖f x - g x‖ ∂μ) ≤ ε ∧
      Continuous g ∧ Integrable g μ := by
  simp only [← memℒp_one_iff_integrable, ← eLpNorm_one_eq_lintegral_nnnorm, ← ENNReal.ofReal_one]
    at hf ⊢
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝³ : NormedSpace Real E
    inst✝² : R1Space α
    inst✝¹ : WeaklyLocallyCompactSpace α
    inst✝ : μ.Regular
    f : α → E
    ε : Real
    hε : LT.lt 0 ε
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal 1) μ
    ⊢ Exists fun g => And (HasCompactSupport g) (And (LE.le (MeasureTheory.integra …
  -/
  simpa using hf.exists_hasCompactSupport_integral_rpow_sub_le zero_lt_one hε
  /-
    🎉 no goals
  -/


/-- Any function in `ℒp` can be approximated by bounded continuous functions when `p < ∞`,
version in terms of `eLpNorm`. -/
theorem Memℒp.exists_boundedContinuous_eLpNorm_sub_le [μ.WeaklyRegular] (hp : p ≠ ∞) {f : α → E}
    (hf : Memℒp f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ g : α →ᵇ E, eLpNorm (f - (g : α → E)) p μ ≤ ε ∧ Memℒp g p μ := by
  suffices H :
      ∃ g : α → E, eLpNorm (f - g) p μ ≤ ε ∧ Continuous g ∧ Memℒp g p μ ∧ IsBounded (range g) by
    rcases H with ⟨g, hg, g_cont, g_mem, g_bd⟩
    exact ⟨⟨⟨g, g_cont⟩, Metric.isBounded_range_iff.1 g_bd⟩, hg, g_mem⟩
  -- It suffices to check that the set of functions we consider approximates characteristic
  -- functions, is stable under addition and made of ae strongly measurable functions.
  -- First check the latter easy facts.
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) ε) (A …
  -/
  apply hf.induction_dense hp _ _ _ _ hε
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : ENNRea …
  -/
  rotate_left
  -- stability under addition
    /-
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      ⊢ ∀ (f g : α → E), And (Continuous f) (And (MeasureTheory.Memℒp f p μ) (Bornol …
    -/
  · rintro f g ⟨f_cont, f_mem, f_bd⟩ ⟨g_cont, g_mem, g_bd⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f✝ : α → E
      hf : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f g : α → E
      f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      f_bd : Bornology.IsBounded (Set.range f)
      g_cont : Continuous g
      g_mem : MeasureTheory.Memℒp g p μ
      g_bd : Bornology.IsBounded (Set.range g)
      ⊢ And (Continuous (HAdd.hAdd f g)) (And (MeasureTheory.Memℒp (HAdd.hAdd f g) p …
    -/
    refine ⟨f_cont.add g_cont, f_mem.add g_mem, ?_⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f✝ : α → E
      hf : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f g : α → E
      f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      f_bd : Bornology.IsBounded (Set.range f)
      g_cont : Continuous g
      g_mem : MeasureTheory.Memℒp g p μ
      g_bd : Bornology.IsBounded (Set.range g)
      ⊢ Bornology.IsBounded (Set.range (HAdd.hAdd f g))
    -/
    let f' : α →ᵇ E := ⟨⟨f, f_cont⟩, Metric.isBounded_range_iff.1 f_bd⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f✝ : α → E
      hf : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f g : α → E
      f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      f_bd : Bornology.IsBounded (Set.range f)
      g_cont : Continuous g
      g_mem : MeasureTheory.Memℒp g p μ
      g_bd : Bornology.IsBounded (Set.range g)
      f' : BoundedContinuousFunction α E := { toFun := f, continuous_toFun := f_cont …
      ⊢ Bornology.IsBounded (Set.range (HAdd.hAdd f g))
    -/
    let g' : α →ᵇ E := ⟨⟨g, g_cont⟩, Metric.isBounded_range_iff.1 g_bd⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f✝ : α → E
      hf : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      f g : α → E
      f_cont : Continuous f
      f_mem : MeasureTheory.Memℒp f p μ
      f_bd : Bornology.IsBounded (Set.range f)
      g_cont : Continuous g
      g_mem : MeasureTheory.Memℒp g p μ
      g_bd : Bornology.IsBounded (Set.range g)
      f' : BoundedContinuousFunction α E := { toFun := f, continuous_toFun := f_cont …
      g' : BoundedContinuousFunction α E := { toFun := g, continuous_toFun := g_cont …
      ⊢ Bornology.IsBounded (Set.range (HAdd.hAdd f g))
    -/
    exact (f' + g').isBounded_range
    /-
      🎉 no goals
    -/
  -- ae strong measurability
    /-
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : NormalSpace α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : BorelSpace α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.WeaklyRegular
      hp : Ne p Top.top
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      ⊢ ∀ (f : α → E), And (Continuous f) (And (MeasureTheory.Memℒp f p μ) (Bornolog …
    -/
  · exact fun f ⟨_, h, _⟩ => h.aestronglyMeasurable
    /-
      🎉 no goals
    -/
  -- We are left with approximating characteristic functions.
  -- This follows from `exists_continuous_eLpNorm_sub_le_of_closed`.
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : ENNRea …
  -/
  intro c t ht htμ ε hε
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  rcases exists_Lp_half E μ p hε with ⟨δ, δpos, hδ⟩
  obtain ⟨η, ηpos, hη⟩ :
      ∃ η : ℝ≥0, 0 < η ∧ ∀ s : Set α, μ s ≤ η → eLpNorm (s.indicator fun _x => c) p μ ≤ δ :=
    exists_eLpNorm_indicator_le hp c δpos.ne'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  have hη_pos' : (0 : ℝ≥0∞) < η := ENNReal.coe_pos.2 ηpos
  obtain ⟨s, st, s_closed, μs⟩ : ∃ s, s ⊆ t ∧ IsClosed s ∧ μ (t \ s) < η :=
    ht.exists_isClosed_diff_lt htμ.ne hη_pos'.ne'
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  have hsμ : μ s < ∞ := (measure_mono st).trans_lt htμ
  have I1 : eLpNorm ((s.indicator fun _y => c) - t.indicator fun _y => c) p μ ≤ δ := by
    rw [← eLpNorm_neg, neg_sub, ← indicator_diff st]
    exact hη _ μs.le
  rcases exists_continuous_eLpNorm_sub_le_of_closed hp s_closed isOpen_univ (subset_univ _) hsμ.ne c
      δpos.ne' with
    ⟨f, f_cont, I2, f_bound, -, f_mem⟩
  have I3 : eLpNorm (f - t.indicator fun _y => c) p μ ≤ ε := by
    convert
      (hδ _ _
          (f_mem.aestronglyMeasurable.sub
            (aestronglyMeasurable_const.indicator s_closed.measurableSet))
          ((aestronglyMeasurable_const.indicator s_closed.measurableSet).sub
            (aestronglyMeasurable_const.indicator ht))
          I2 I1).le using 2
    simp only [sub_add_sub_cancel]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub g (t.indicator  …
  -/
  refine ⟨f, I3, f_cont, f_mem, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    hp : Ne p Top.top
    f✝ : α → E
    hf : MeasureTheory.Memℒp f✝ p μ
    ε✝ : ENNReal
    hε✝ : Ne ε✝ 0
    c : E
    t : Set α
    ht : MeasurableSet t
    htμ : LT.lt (μ t) Top.top
    ε : ENNReal
    hε : Ne ε 0
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    η : NNReal
    ηpos : LT.lt 0 η
    hη : ∀ (s : Set α), LE.le (μ s) ↑η → LE.le (MeasureTheory.eLpNorm (s.indicator …
    hη_pos' : LT.lt 0 ↑η
    s : Set α
    st : HasSubset.Subset s t
    s_closed : IsClosed s
    μs : LT.lt (μ (SDiff.sdiff t s)) ↑η
    hsμ : LT.lt (μ s) Top.top
    I1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun _y => c) (t.indi …
    f : α → E
    f_cont : Continuous f
    I2 : LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) (s.indicator (fun  …
    f_bound : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm c)
    f_mem : MeasureTheory.Memℒp f p μ
    I3 : LE.le (MeasureTheory.eLpNorm (HSub.hSub f (t.indicator fun _y => c)) p μ) ε
    ⊢ Bornology.IsBounded (Set.range f)
  -/
  exact (BoundedContinuousFunction.ofNormedAddCommGroup f f_cont _ f_bound).isBounded_range
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.exists_boundedContinuous_snorm_sub_le := Memℒp.exists_boundedContinuous_eLpNorm_sub_le


/-- Any function in `ℒp` can be approximated by bounded continuous functions when `0 < p < ∞`,
version in terms of `∫`. -/
theorem Memℒp.exists_boundedContinuous_integral_rpow_sub_le [μ.WeaklyRegular] {p : ℝ} (hp : 0 < p)
    {f : α → E} (hf : Memℒp f (ENNReal.ofReal p) μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ g : α →ᵇ E, (∫ x, ‖f x - g x‖ ^ p ∂μ) ≤ ε ∧ Memℒp g (ENNReal.ofReal p) μ := by
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => HPow.hPow (Nor …
  -/
  have I : 0 < ε ^ (1 / p) := Real.rpow_pos_of_pos hε _
  have A : ENNReal.ofReal (ε ^ (1 / p)) ≠ 0 := by
    simp only [Ne, ENNReal.ofReal_eq_zero, not_le, I]
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => HPow.hPow (Nor …
  -/
  have B : ENNReal.ofReal p ≠ 0 := by simpa only [Ne, ENNReal.ofReal_eq_zero, not_le] using hp
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => HPow.hPow (Nor …
  -/
  rcases hf.exists_boundedContinuous_eLpNorm_sub_le ENNReal.coe_ne_top A with ⟨g, hg, g_mem⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : BoundedContinuousFunction α E
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) (↑p.toNNReal) μ) (ENNReal.o …
    g_mem : MeasureTheory.Memℒp (⇑g) (↑p.toNNReal) μ
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => HPow.hPow (Nor …
  -/
  change eLpNorm _ (ENNReal.ofReal p) _ ≤ _ at hg
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : BoundedContinuousFunction α E
    g_mem : MeasureTheory.Memℒp (⇑g) (↑p.toNNReal) μ
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) (ENNReal.ofReal p) μ) (ENNR …
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => HPow.hPow (Nor …
  -/
  refine ⟨g, ?_, g_mem⟩
  rwa [(hf.sub g_mem).eLpNorm_eq_integral_rpow_norm B ENNReal.coe_ne_top,
    ENNReal.ofReal_le_ofReal_iff I.le, one_div, ENNReal.toReal_ofReal hp.le,
    Real.rpow_le_rpow_iff _ hε.le (inv_pos.2 hp)] at hg
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    p : Real
    hp : LT.lt 0 p
    f : α → E
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    ε : Real
    hε : LT.lt 0 ε
    I : LT.lt 0 (HPow.hPow ε (HDiv.hDiv 1 p))
    A : Ne (ENNReal.ofReal (HPow.hPow ε (HDiv.hDiv 1 p))) 0
    B : Ne (ENNReal.ofReal p) 0
    g : BoundedContinuousFunction α E
    g_mem : MeasureTheory.Memℒp (⇑g) (↑p.toNNReal) μ
    hg : LE.le (HPow.hPow (MeasureTheory.integral μ fun a => HPow.hPow (Norm.norm  …
    ⊢ LE.le 0 (MeasureTheory.integral μ fun a => HPow.hPow (Norm.norm (HSub.hSub f …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- Any integrable function can be approximated by bounded continuous functions,
version in terms of `∫⁻`. -/
theorem Integrable.exists_boundedContinuous_lintegral_sub_le [μ.WeaklyRegular] {f : α → E}
    (hf : Integrable f μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ g : α →ᵇ E, (∫⁻ x, ‖f x - g x‖₊ ∂μ) ≤ ε ∧ Integrable g μ := by
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
  -/
  simp only [← memℒp_one_iff_integrable, ← eLpNorm_one_eq_lintegral_nnnorm] at hf ⊢
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    f : α → E
    ε : ENNReal
    hε : Ne ε 0
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x)  …
  -/
  exact hf.exists_boundedContinuous_eLpNorm_sub_le ENNReal.one_ne_top hε
  /-
    🎉 no goals
  -/


/-- Any integrable function can be approximated by bounded continuous functions,
version in terms of `∫`. -/
theorem Integrable.exists_boundedContinuous_integral_sub_le [μ.WeaklyRegular] {f : α → E}
    (hf : Integrable f μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ g : α →ᵇ E, (∫ x, ‖f x - g x‖ ∂μ) ≤ ε ∧ Integrable g μ := by
  simp only [← memℒp_one_iff_integrable, ← eLpNorm_one_eq_lintegral_nnnorm, ← ENNReal.ofReal_one]
    at hf ⊢
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : NormalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.WeaklyRegular
    f : α → E
    ε : Real
    hε : LT.lt 0 ε
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal 1) μ
    ⊢ Exists fun g => And (LE.le (MeasureTheory.integral μ fun x => Norm.norm (HSu …
  -/
  simpa using hf.exists_boundedContinuous_integral_rpow_sub_le zero_lt_one hε
  /-
    🎉 no goals
  -/


/-- A function in `Lp` can be approximated in `Lp` by continuous functions. -/
theorem boundedContinuousFunction_dense [SecondCountableTopologyEither α E] [Fact (1 ≤ p)]
    (hp : p ≠ ∞) [μ.WeaklyRegular] :
    Dense (boundedContinuousFunction E p μ : Set (Lp E p μ)) := by
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    inst✝ : μ.WeaklyRegular
    ⊢ Dense ↑(MeasureTheory.Lp.boundedContinuousFunction E p μ)
  -/
  intro f
  /-
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    inst✝ : μ.WeaklyRegular
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Membership.mem (closure ↑(MeasureTheory.Lp.boundedContinuousFunction E p μ)) f
  -/
  refine (mem_closure_iff_nhds_basis EMetric.nhds_basis_closed_eball).2 fun ε hε ↦ ?_
  obtain ⟨g, hg, g_mem⟩ :
      ∃ g : α →ᵇ E, eLpNorm ((f : α → E) - (g : α → E)) p μ ≤ ε ∧ Memℒp g p μ :=
    (Lp.memℒp f).exists_boundedContinuous_eLpNorm_sub_le hp hε.ne'
  /-
    case intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    inst✝ : μ.WeaklyRegular
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ε : ENNReal
    hε : LT.lt 0 ε
    g : BoundedContinuousFunction α E
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub ↑↑f ⇑g) p μ) ε
    g_mem : MeasureTheory.Memℒp (⇑g) p μ
    ⊢ Exists fun y => And (Membership.mem (↑(MeasureTheory.Lp.boundedContinuousFun …
  -/
  refine ⟨g_mem.toLp _, ⟨g, rfl⟩, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : NormalSpace α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : BorelSpace α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝³ : NormedSpace Real E
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    inst✝ : μ.WeaklyRegular
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ε : ENNReal
    hε : LT.lt 0 ε
    g : BoundedContinuousFunction α E
    hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub ↑↑f ⇑g) p μ) ε
    g_mem : MeasureTheory.Memℒp (⇑g) p μ
    ⊢ Membership.mem (EMetric.closedBall f ε) (MeasureTheory.Memℒp.toLp (⇑g) g_mem)
  -/
  rwa [EMetric.mem_closedBall', ← Lp.toLp_coeFn f (Lp.memℒp f), Lp.edist_toLp_toLp]
  /-
    🎉 no goals
  -/


/-- A function in `Lp` can be approximated in `Lp` by continuous functions. -/
theorem boundedContinuousFunction_topologicalClosure [SecondCountableTopologyEither α E]
    [Fact (1 ≤ p)] (hp : p ≠ ∞) [μ.WeaklyRegular] :
    (boundedContinuousFunction E p μ).topologicalClosure = ⊤ :=
  SetLike.ext' <| (boundedContinuousFunction_dense E μ hp).closure_eq


theorem toLp_denseRange [μ.WeaklyRegular] [IsFiniteMeasure μ] (hp : p ≠ ∞) :
    DenseRange (toLp p μ 𝕜 : (α →ᵇ E) →L[𝕜] Lp E p μ) := by
  /-
    α : Type u_1
    inst✝¹⁰ : TopologicalSpace α
    inst✝⁹ : NormalSpace α
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : BorelSpace α
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝⁵ : SecondCountableTopologyEither α E
    _i : Fact (LE.le 1 p)
    𝕜 : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedAlgebra Real 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : Ne p Top.top
    ⊢ DenseRange ⇑(BoundedContinuousFunction.toLp p μ 𝕜)
  -/
  haveI : NormedSpace ℝ E := RestrictScalars.normedSpace ℝ 𝕜 E
  simpa only [← range_toLp p μ (𝕜 := 𝕜)]
    using MeasureTheory.Lp.boundedContinuousFunction_dense E μ hp


/-- Continuous functions are dense in `MeasureTheory.Lp`, `1 ≤ p < ∞`. This theorem assumes that
the domain is a compact space because otherwise `ContinuousMap.toLp` is undefined. Use
`BoundedContinuousFunction.toLp_denseRange` if the domain is not a compact space. -/
theorem toLp_denseRange [CompactSpace α] [μ.WeaklyRegular] [IsFiniteMeasure μ] (hp : p ≠ ∞) :
    DenseRange (toLp p μ 𝕜 : C(α, E) →L[𝕜] Lp E p μ) := by
  /-
    α : Type u_1
    inst✝¹¹ : TopologicalSpace α
    inst✝¹⁰ : NormalSpace α
    inst✝⁹ : MeasurableSpace α
    inst✝⁸ : BorelSpace α
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝⁶ : SecondCountableTopologyEither α E
    _i : Fact (LE.le 1 p)
    𝕜 : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : NormedAlgebra Real 𝕜
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompactSpace α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : Ne p Top.top
    ⊢ DenseRange ⇑(ContinuousMap.toLp p μ 𝕜)
  -/
  refine (BoundedContinuousFunction.toLp_denseRange _ _ 𝕜 hp).mono ?_
  /-
    α : Type u_1
    inst✝¹¹ : TopologicalSpace α
    inst✝¹⁰ : NormalSpace α
    inst✝⁹ : MeasurableSpace α
    inst✝⁸ : BorelSpace α
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝⁶ : SecondCountableTopologyEither α E
    _i : Fact (LE.le 1 p)
    𝕜 : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : NormedAlgebra Real 𝕜
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompactSpace α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : Ne p Top.top
    ⊢ HasSubset.Subset (Set.range ⇑(BoundedContinuousFunction.toLp p μ 𝕜)) (Set.ra …
  -/
  refine range_subset_iff.2 fun f ↦ ?_
  /-
    α : Type u_1
    inst✝¹¹ : TopologicalSpace α
    inst✝¹⁰ : NormalSpace α
    inst✝⁹ : MeasurableSpace α
    inst✝⁸ : BorelSpace α
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝⁶ : SecondCountableTopologyEither α E
    _i : Fact (LE.le 1 p)
    𝕜 : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : NormedAlgebra Real 𝕜
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompactSpace α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : Ne p Top.top
    f : BoundedContinuousFunction α E
    ⊢ Membership.mem (Set.range ⇑(ContinuousMap.toLp p μ 𝕜)) ((BoundedContinuousFu …
  -/
  exact ⟨f.toContinuousMap, rfl⟩
  /-
    🎉 no goals
  -/


