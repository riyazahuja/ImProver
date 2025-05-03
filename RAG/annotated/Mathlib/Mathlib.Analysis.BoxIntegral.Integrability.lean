/-- The indicator function of a measurable set is McShane integrable with respect to any
locally-finite measure. -/
theorem hasIntegralIndicatorConst (l : IntegrationParams) (hl : l.bRiemann = false)
    {s : Set (ι → ℝ)} (hs : MeasurableSet s) (I : Box ι) (y : E) (μ : Measure (ι → ℝ))
    [IsLocallyFiniteMeasure μ] :
    HasIntegral.{u, v, v} I l (s.indicator fun _ => y) μ.toBoxAdditive.toSMul
      ((μ (s ∩ I)).toReal • y) := by
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    s : Set (ι → Real)
    hs : MeasurableSet s
    I : BoxIntegral.Box ι
    y : E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ BoxIntegral.HasIntegral I l (s.indicator fun x => y) μ.toBoxAdditive.toSMul  …
  -/
  refine HasIntegral.of_mul ‖y‖ fun ε ε0 => ?_
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    s : Set (ι → Real)
    hs : MeasurableSet s
    I : BoxIntegral.Box ι
    y : E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  lift ε to ℝ≥0 using ε0.le; rw [NNReal.coe_pos] at ε0
  /- First we choose a closed set `F ⊆ s ∩ I.Icc` and an open set `U ⊇ s` such that
    both `(s ∩ I.Icc) \ F` and `U \ s` have measure less than `ε`. -/
  have A : μ (s ∩ Box.Icc I) ≠ ∞ :=
    ((measure_mono Set.inter_subset_right).trans_lt (I.measure_Icc_lt_top μ)).ne
  have B : μ (s ∩ I) ≠ ∞ :=
    ((measure_mono Set.inter_subset_right).trans_lt (I.measure_coe_lt_top μ)).ne
  obtain ⟨F, hFs, hFc, hμF⟩ : ∃ F, F ⊆ s ∩ Box.Icc I ∧ IsClosed F ∧ μ ((s ∩ Box.Icc I) \ F) < ε :=
    (hs.inter I.measurableSet_Icc).exists_isClosed_diff_lt A (ENNReal.coe_pos.2 ε0).ne'
  obtain ⟨U, hsU, hUo, hUt, hμU⟩ :
      ∃ U, s ∩ Box.Icc I ⊆ U ∧ IsOpen U ∧ μ U < ∞ ∧ μ (U \ (s ∩ Box.Icc I)) < ε :=
    (hs.inter I.measurableSet_Icc).exists_isOpen_diff_lt A (ENNReal.coe_pos.2 ε0).ne'
  /- Then we choose `r` so that `closed_ball x (r x) ⊆ U` whenever `x ∈ s ∩ I.Icc` and
    `closed_ball x (r x)` is disjoint with `F` otherwise. -/
  have : ∀ x ∈ s ∩ Box.Icc I, ∃ r : Ioi (0 : ℝ), closedBall x r ⊆ U := fun x hx => by
    rcases nhds_basis_closedBall.mem_iff.1 (hUo.mem_nhds <| hsU hx) with ⟨r, hr₀, hr⟩
    exact ⟨⟨r, hr₀⟩, hr⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    s : Set (ι → Real)
    hs : MeasurableSet s
    I : BoxIntegral.Box ι
    y : E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ε : NNReal
    ε0 : LT.lt 0 ε
    A : Ne (μ (Inter.inter s (BoxIntegral.Box.Icc I))) Top.top
    B : Ne (μ (Inter.inter s ↑I)) Top.top
    F : Set (ι → Real)
    hFs : HasSubset.Subset F (Inter.inter s (BoxIntegral.Box.Icc I))
    hFc : IsClosed F
    hμF : LT.lt (μ (SDiff.sdiff (Inter.inter s (BoxIntegral.Box.Icc I)) F)) ↑ε
    U : Set (ι → Real)
    hsU : HasSubset.Subset (Inter.inter s (BoxIntegral.Box.Icc I)) U
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U (Inter.inter s (BoxIntegral.Box.Icc I)))) ↑ε
    this : ∀ (x : ι → Real), Membership.mem (Inter.inter s (BoxIntegral.Box.Icc I) …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  choose! rs hrsU using this
  have : ∀ x ∈ Box.Icc I \ s, ∃ r : Ioi (0 : ℝ), closedBall x r ⊆ Fᶜ := fun x hx => by
    obtain ⟨r, hr₀, hr⟩ :=
      nhds_basis_closedBall.mem_iff.1 (hFc.isOpen_compl.mem_nhds fun hx' => hx.2 (hFs hx').1)
    exact ⟨⟨r, hr₀⟩, hr⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    s : Set (ι → Real)
    hs : MeasurableSet s
    I : BoxIntegral.Box ι
    y : E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ε : NNReal
    ε0 : LT.lt 0 ε
    A : Ne (μ (Inter.inter s (BoxIntegral.Box.Icc I))) Top.top
    B : Ne (μ (Inter.inter s ↑I)) Top.top
    F : Set (ι → Real)
    hFs : HasSubset.Subset F (Inter.inter s (BoxIntegral.Box.Icc I))
    hFc : IsClosed F
    hμF : LT.lt (μ (SDiff.sdiff (Inter.inter s (BoxIntegral.Box.Icc I)) F)) ↑ε
    U : Set (ι → Real)
    hsU : HasSubset.Subset (Inter.inter s (BoxIntegral.Box.Icc I)) U
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U (Inter.inter s (BoxIntegral.Box.Icc I)))) ↑ε
    rs : (ι → Real) → ↑(Set.Ioi 0)
    hrsU : ∀ (x : ι → Real), Membership.mem (Inter.inter s (BoxIntegral.Box.Icc I) …
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  choose! rs' hrs'F using this
  classical
  set r : (ι → ℝ) → Ioi (0 : ℝ) := s.piecewise rs rs'
  refine ⟨fun _ => r, fun c => l.rCond_of_bRiemann_eq_false hl, fun c π hπ hπp => ?_⟩; rw [mul_comm]
  /- Then the union of boxes `J ∈ π` such that `π.tag ∈ s` includes `F` and is included by `U`,
    hence its measure is `ε`-close to the measure of `s`. -/
  dsimp [integralSum]
  simp only [mem_closedBall, dist_eq_norm, ← indicator_const_smul_apply,
    sum_indicator_eq_sum_filter, ← sum_smul, ← sub_smul, norm_smul, Real.norm_eq_abs, ←
    Prepartition.filter_boxes, ← Prepartition.measure_iUnion_toReal]
  gcongr
  set t := (π.filter (π.tag · ∈ s)).iUnion
  change abs ((μ t).toReal - (μ (s ∩ I)).toReal) ≤ ε
  have htU : t ⊆ U ∩ I := by
    simp only [t, TaggedPrepartition.iUnion_def, iUnion_subset_iff, TaggedPrepartition.mem_filter,
      and_imp]
    refine fun J hJ hJs x hx => ⟨hrsU _ ⟨hJs, π.tag_mem_Icc J⟩ ?_, π.le_of_mem' J hJ hx⟩
    simpa only [r, s.piecewise_eq_of_mem _ _ hJs] using hπ.1 J hJ (Box.coe_subset_Icc hx)
  refine abs_sub_le_iff.2 ⟨?_, ?_⟩
  · refine (ENNReal.le_toReal_sub B).trans (ENNReal.toReal_le_coe_of_le_coe ?_)
    refine (tsub_le_tsub (measure_mono htU) le_rfl).trans (le_measure_diff.trans ?_)
    refine (measure_mono fun x hx => ?_).trans hμU.le
    exact ⟨hx.1.1, fun hx' => hx.2 ⟨hx'.1, hx.1.2⟩⟩
  · have hμt : μ t ≠ ∞ := ((measure_mono (htU.trans inter_subset_left)).trans_lt hUt).ne
    refine (ENNReal.le_toReal_sub hμt).trans (ENNReal.toReal_le_coe_of_le_coe ?_)
    refine le_measure_diff.trans ((measure_mono ?_).trans hμF.le)
    rintro x ⟨⟨hxs, hxI⟩, hxt⟩
    refine ⟨⟨hxs, Box.coe_subset_Icc hxI⟩, fun hxF => hxt ?_⟩
    simp only [t, TaggedPrepartition.iUnion_def, TaggedPrepartition.mem_filter, Set.mem_iUnion]
    rcases hπp x hxI with ⟨J, hJπ, hxJ⟩
    refine ⟨J, ⟨hJπ, ?_⟩, hxJ⟩
    contrapose hxF
    refine hrs'F _ ⟨π.tag_mem_Icc J, hxF⟩ ?_
    simpa only [r, s.piecewise_eq_of_not_mem _ _ hxF] using hπ.1 J hJπ (Box.coe_subset_Icc hxJ)


/-- If `f` is a.e. equal to zero on a rectangular box, then it has McShane integral zero on this
box. -/
theorem HasIntegral.of_aeEq_zero {l : IntegrationParams} {I : Box ι} {f : (ι → ℝ) → E}
    {μ : Measure (ι → ℝ)} [IsLocallyFiniteMeasure μ] (hf : f =ᵐ[μ.restrict I] 0)
    (hl : l.bRiemann = false) : HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul 0 := by
  /- Each set `{x | n < ‖f x‖ ≤ n + 1}`, `n : ℕ`, has measure zero. We cover it by an open set of
    measure less than `ε / 2 ^ n / (n + 1)`. Then the norm of the integral sum is less than `ε`. -/
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f 0
    hl : Eq l.bRiemann Bool.false
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul 0
  -/
  refine hasIntegral_iff.2 fun ε ε0 => ?_
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f 0
    hl : Eq l.bRiemann Bool.false
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  lift ε to ℝ≥0 using ε0.lt.le; rw [gt_iff_lt, NNReal.coe_pos] at ε0
  /-
    case intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f 0
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  rcases NNReal.exists_pos_sum_of_countable ε0.ne' ℕ with ⟨δ, δ0, c, hδc, hcε⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f 0
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  haveI := Fact.mk (I.measure_coe_lt_top μ)
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f 0
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  change μ.restrict I {x | f x ≠ 0} = 0 at hf
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  set N : (ι → ℝ) → ℕ := fun x => ⌈‖f x‖⌉₊
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  have N0 : ∀ {x}, N x = 0 ↔ f x = 0 := by simp [N]
  have : ∀ n, ∃ U, N ⁻¹' {n} ⊆ U ∧ IsOpen U ∧ μ.restrict I U < δ n / n := fun n ↦ by
    refine (N ⁻¹' {n}).exists_isOpen_lt_of_lt _ ?_
    cases' n with n
    · simp [ENNReal.div_zero (ENNReal.coe_pos.2 (δ0 _)).ne']
    · refine (measure_mono_null ?_ hf).le.trans_lt ?_
      · exact fun x hxN hxf => n.succ_ne_zero ((Eq.symm hxN).trans <| N0.2 hxf)
      · simp [(δ0 _).ne']
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    this : ∀ (n : Nat), Exists fun U => And (HasSubset.Subset (Set.preimage N (Sin …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  choose U hNU hUo hμU using this
  have : ∀ x, ∃ r : Ioi (0 : ℝ), closedBall x r ⊆ U (N x) := fun x => by
    obtain ⟨r, hr₀, hr⟩ := nhds_basis_closedBall.mem_iff.1 ((hUo _).mem_nhds (hNU _ rfl))
    exact ⟨⟨r, hr₀⟩, hr⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    this : ∀ (x : ι → Real), Exists fun r => HasSubset.Subset (Metric.closedBall x …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  choose r hrU using this
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  refine ⟨fun _ => r, fun c => l.rCond_of_bRiemann_eq_false hl, fun c π hπ _ => ?_⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f μ.toBoxAdditive.toSMul π) 0) ↑ε
  -/
  rw [dist_eq_norm, sub_zero, ← integralSum_fiberwise fun J => N (π.tag J)]
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    ⊢ LE.le (Norm.norm ((Finset.image (fun J => N (π.tag J)) π.boxes).sum fun y => …
  -/
  refine le_trans ?_ (NNReal.coe_lt_coe.2 hcε).le
  refine (norm_sum_le_of_le _ ?_).trans
    (sum_le_hasSum _ (fun n _ => (δ n).2) (NNReal.hasSum_coe.2 hδc))
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    ⊢ ∀ (b : Nat), Membership.mem (Finset.image (fun J => N (π.tag J)) π.boxes) b  …
  -/
  rintro n -
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    ⊢ LE.le (Norm.norm (BoxIntegral.integralSum f μ.toBoxAdditive.toSMul (π.filter …
  -/
  dsimp [integralSum]
  have : ∀ J ∈ π.filter fun J => N (π.tag J) = n,
      ‖(μ ↑J).toReal • f (π.tag J)‖ ≤ (μ J).toReal * n := fun J hJ ↦ by
    rw [TaggedPrepartition.mem_filter] at hJ
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
    gcongr
    exact hJ.2 ▸ Nat.le_ceil _
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    this : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (N (π.t …
    ⊢ LE.le (Norm.norm ((π.filter fun x => Eq (N (π.tag x)) n).boxes.sum fun J =>  …
  -/
  refine (norm_sum_le_of_le _ this).trans ?_; clear this
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    ⊢ LE.le ((π.filter fun J => Eq (N (π.tag J)) n).boxes.sum fun b => HMul.hMul ( …
  -/
  rw [← sum_mul, ← Prepartition.measure_iUnion_toReal]
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    ⊢ LE.le (HMul.hMul (μ (π.filter fun J => Eq (N (π.tag J)) n).iUnion).toReal ↑n …
  -/
  let m := μ (π.filter fun J => N (π.tag J) = n).iUnion
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    m : ENNReal := μ (π.filter fun J => Eq (N (π.tag J)) n).iUnion
    ⊢ LE.le (HMul.hMul (μ (π.filter fun J => Eq (N (π.tag J)) n).iUnion).toReal ↑n …
  -/
  show m.toReal * ↑n ≤ ↑(δ n)
  have : m < δ n / n := by
    simp only [Measure.restrict_apply (hUo _).measurableSet] at hμU
    refine (measure_mono ?_).trans_lt (hμU _)
    simp only [Set.subset_def, TaggedPrepartition.mem_iUnion, TaggedPrepartition.mem_filter]
    rintro x ⟨J, ⟨hJ, rfl⟩, hx⟩
    exact ⟨hrU _ (hπ.1 _ hJ (Box.coe_subset_Icc hx)), π.le_of_mem' J hJ hx⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    m : ENNReal := μ (π.filter fun J => Eq (N (π.tag J)) n).iUnion
    this : LT.lt m (HDiv.hDiv ↑(δ n) ↑n)
    ⊢ LE.le (HMul.hMul m.toReal ↑n) ↑(δ n)
  -/
  clear_value m
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    m : ENNReal
    this : LT.lt m (HDiv.hDiv ↑(δ n) ↑n)
    ⊢ LE.le (HMul.hMul m.toReal ↑n) ↑(δ n)
  -/
  lift m to ℝ≥0 using ne_top_of_lt this
  rw [ENNReal.coe_toReal, ← NNReal.coe_natCast, ← NNReal.coe_mul, NNReal.coe_le_coe, ←
    ENNReal.coe_le_coe, ENNReal.coe_mul, ENNReal.coe_natCast, mul_comm]
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hl : Eq l.bRiemann Bool.false
    ε : NNReal
    ε0 : LT.lt 0 ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    this✝ : Fact (LT.lt (μ ↑I) Top.top)
    hf : Eq ((μ.restrict ↑I) (setOf fun x => Ne (f x) 0)) 0
    N : (ι → Real) → Nat := fun x => Nat.ceil (Norm.norm (f x))
    N0 : ∀ {x : ι → Real}, Iff (Eq (N x) 0) (Eq (f x) 0)
    U : Nat → Set (ι → Real)
    hNU : ∀ (n : Nat), HasSubset.Subset (Set.preimage N (Singleton.singleton n)) ( …
    hUo : ∀ (n : Nat), IsOpen (U n)
    hμU : ∀ (n : Nat), LT.lt ((μ.restrict ↑I) (U n)) (HDiv.hDiv ↑(δ n) ↑n)
    r : (ι → Real) → ↑(Set.Ioi 0)
    hrU : ∀ (x : ι → Real), HasSubset.Subset (Metric.closedBall x ↑(r x)) (U (N x))
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c ((fun x => r) c) π
    x✝ : π.IsPartition
    n : Nat
    m : NNReal
    this : LT.lt (↑m) (HDiv.hDiv ↑(δ n) ↑n)
    ⊢ LE.le (HMul.hMul ↑n ↑m) ↑(δ n)
  -/
  exact (mul_le_mul_left' this.le _).trans ENNReal.mul_div_le
  /-
    🎉 no goals
  -/


/-- If `f` has integral `y` on a box `I` with respect to a locally finite measure `μ` and `g` is
a.e. equal to `f` on `I`, then `g` has the same integral on `I`. -/
theorem HasIntegral.congr_ae {l : IntegrationParams} {I : Box ι} {y : E} {f g : (ι → ℝ) → E}
    {μ : Measure (ι → ℝ)} [IsLocallyFiniteMeasure μ]
    (hf : HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul y) (hfg : f =ᵐ[μ.restrict I] g)
    (hl : l.bRiemann = false) : HasIntegral.{u, v, v} I l g μ.toBoxAdditive.toSMul y := by
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    y : E
    f g : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    hl : Eq l.bRiemann Bool.false
    ⊢ BoxIntegral.HasIntegral I l g μ.toBoxAdditive.toSMul y
  -/
  have : g - f =ᵐ[μ.restrict I] 0 := hfg.mono fun x hx => sub_eq_zero.2 hx.symm
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    y : E
    f g : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hf : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    hl : Eq l.bRiemann Bool.false
    this : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq (HSub.hSub g f) 0
    ⊢ BoxIntegral.HasIntegral I l g μ.toBoxAdditive.toSMul y
  -/
  simpa using hf.add (HasIntegral.of_aeEq_zero this hl)
  /-
    🎉 no goals
  -/


/-- A simple function is McShane integrable w.r.t. any locally finite measure. -/
theorem hasBoxIntegral (f : SimpleFunc (ι → ℝ) E) (μ : Measure (ι → ℝ)) [IsLocallyFiniteMeasure μ]
    (I : Box ι) (l : IntegrationParams) (hl : l.bRiemann = false) :
    HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul (f.integral (μ.restrict I)) := by
  /-
    ι : Type u
    E : Type v
    inst✝³ : Fintype ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : MeasureTheory.SimpleFunc (ι → Real) E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    ⊢ BoxIntegral.HasIntegral I l (⇑f) μ.toBoxAdditive.toSMul (MeasureTheory.Simpl …
  -/
  induction' f using MeasureTheory.SimpleFunc.induction with y s hs f g _ hfi hgi
  · simpa only [Measure.restrict_apply hs, const_zero, integral_piecewise_zero, integral_const,
      Measure.restrict_apply, MeasurableSet.univ, Set.univ_inter] using
      BoxIntegral.hasIntegralIndicatorConst l hl hs I y μ
    /-
      case h_add
      ι : Type u
      E : Type v
      inst✝³ : Fintype ι
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      f g : MeasureTheory.SimpleFunc (ι → Real) E
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑g)
      hfi : BoxIntegral.HasIntegral I l (⇑f) μ.toBoxAdditive.toSMul (MeasureTheory.S …
      hgi : BoxIntegral.HasIntegral I l (⇑g) μ.toBoxAdditive.toSMul (MeasureTheory.S …
      ⊢ BoxIntegral.HasIntegral I l (⇑(HAdd.hAdd f g)) μ.toBoxAdditive.toSMul (Measu …
    -/
  · borelize E; haveI := Fact.mk (I.measure_coe_lt_top μ)
    /-
      case h_add
      ι : Type u
      E : Type v
      inst✝³ : Fintype ι
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      f g : MeasureTheory.SimpleFunc (ι → Real) E
      a✝ : Disjoint (Function.support ⇑f) (Function.support ⇑g)
      hfi : BoxIntegral.HasIntegral I l (⇑f) μ.toBoxAdditive.toSMul (MeasureTheory.S …
      hgi : BoxIntegral.HasIntegral I l (⇑g) μ.toBoxAdditive.toSMul (MeasureTheory.S …
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      this : Fact (LT.lt (μ ↑I) Top.top)
      ⊢ BoxIntegral.HasIntegral I l (⇑(HAdd.hAdd f g)) μ.toBoxAdditive.toSMul (Measu …
    -/
    rw [integral_add]
    exacts [hfi.add hgi, integrable_iff.2 fun _ _ => measure_lt_top _ _,
      integrable_iff.2 fun _ _ => measure_lt_top _ _]


/-- For a simple function, its McShane (or Henstock, or `⊥`) box integral is equal to its
integral in the sense of `MeasureTheory.SimpleFunc.integral`. -/
theorem box_integral_eq_integral (f : SimpleFunc (ι → ℝ) E) (μ : Measure (ι → ℝ))
    [IsLocallyFiniteMeasure μ] (I : Box ι) (l : IntegrationParams) (hl : l.bRiemann = false) :
    BoxIntegral.integral.{u, v, v} I l f μ.toBoxAdditive.toSMul = f.integral (μ.restrict I) :=
  (f.hasBoxIntegral μ I l hl).integral_eq


/-- If `f : ℝⁿ → E` is Bochner integrable w.r.t. a locally finite measure `μ` on a rectangular box
`I`, then it is McShane integrable on `I` with the same integral. -/
theorem IntegrableOn.hasBoxIntegral [CompleteSpace E] {f : (ι → ℝ) → E} {μ : Measure (ι → ℝ)}
    [IsLocallyFiniteMeasure μ] {I : Box ι} (hf : IntegrableOn f I μ) (l : IntegrationParams)
    (hl : l.bRiemann = false) :
    HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul (∫ x in I, f x ∂μ) := by
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  borelize E
  -- First we replace an `ae_strongly_measurable` function by a measurable one.
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  rcases hf.aestronglyMeasurable with ⟨g, hg, hfg⟩
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  haveI : SeparableSpace (range g ∪ {0} : Set E) := hg.separableSpace_range_union_singleton
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  rw [integral_congr_ae hfg]; have hgi : IntegrableOn g I μ := (integrable_congr hfg).1 hf
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  refine BoxIntegral.HasIntegral.congr_ae ?_ hfg.symm hl
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hf : MeasureTheory.IntegrableOn f (↑I) μ
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae (μ.restrict ↑I)).EventuallyEq f g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    ⊢ BoxIntegral.HasIntegral I l g μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  clear! f
  /- Now consider the sequence of simple functions
    `SimpleFunc.approxOn g hg.measurable (range g ∪ {0}) 0 (by simp)`
    approximating `g`. Recall some properties of this sequence. -/
  set f : ℕ → SimpleFunc (ι → ℝ) E :=
    SimpleFunc.approxOn g hg.measurable (range g ∪ {0}) 0 (by simp)
  have hfi : ∀ n, IntegrableOn (f n) I μ :=
    SimpleFunc.integrable_approxOn_range hg.measurable hgi
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    ⊢ BoxIntegral.HasIntegral I l g μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  have hfi' := fun n => ((f n).hasBoxIntegral μ I l hl).integrable
  have hfg_mono : ∀ (x) {m n}, m ≤ n → ‖f n x - g x‖ ≤ ‖f m x - g x‖ := by
    intro x m n hmn
    rw [← dist_eq_norm, ← dist_eq_norm, dist_nndist, dist_nndist, NNReal.coe_le_coe, ←
      ENNReal.coe_le_coe, ← edist_nndist, ← edist_nndist]
    exact SimpleFunc.edist_approxOn_mono hg.measurable _ x hmn
  /- Now consider `ε > 0`. We need to find `r` such that for any tagged partition subordinate
    to `r`, the integral sum is `(μ I + 1 + 1) * ε`-close to the Bochner integral. -/
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ⊢ BoxIntegral.HasIntegral I l g μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  refine HasIntegral.of_mul ((μ I).toReal + 1 + 1) fun ε ε0 => ?_
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  lift ε to ℝ≥0 using ε0.le; rw [NNReal.coe_pos] at ε0; have ε0' := ENNReal.coe_pos.2 ε0
  -- Choose `N` such that the integral of `‖f N x - g x‖` is less than or equal to `ε`.
  obtain ⟨N₀, hN₀⟩ : ∃ N : ℕ, ∫ x in I, ‖f N x - g x‖ ∂μ ≤ ε := by
    have : Tendsto (fun n => ∫⁻ x in I, ‖f n x - g x‖₊ ∂μ) atTop (𝓝 0) :=
      SimpleFunc.tendsto_approxOn_range_L1_nnnorm hg.measurable hgi
    refine (this.eventually (ge_mem_nhds ε0')).exists.imp fun N hN => ?_
    exact integral_coe_le_of_lintegral_coe_le hN
  -- For each `x`, we choose `Nx x ≥ N₀` such that `dist (f Nx x) (g x) ≤ ε`.
  have : ∀ x, ∃ N₁, N₀ ≤ N₁ ∧ dist (f N₁ x) (g x) ≤ ε := fun x ↦ by
    have : Tendsto (f · x) atTop (𝓝 <| g x) :=
      SimpleFunc.tendsto_approxOn hg.measurable _ (subset_closure (by simp))
    exact ((eventually_ge_atTop N₀).and <| this <| closedBall_mem_nhds _ ε0).exists
  /-
    case intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝² : MeasurableSpace E := borel E
    this✝¹ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    this : ∀ (x : ι → Real), Exists fun N₁ => And (LE.le N₀ N₁) (LE.le (Dist.dist  …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  choose Nx hNx hNxε using this
  -- We also choose a convergent series with `∑' i : ℕ, δ i < ε`.
  /-
    case intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    Nx : (ι → Real) → Nat
    hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
    hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  rcases NNReal.exists_pos_sum_of_countable ε0.ne' ℕ with ⟨δ, δ0, c, hδc, hcε⟩
  /- Since each simple function `fᵢ` is integrable, there exists `rᵢ : ℝⁿ → (0, ∞)` such that
    the integral sum of `f` over any tagged prepartition is `δᵢ`-close to the sum of integrals
    of `fᵢ` over the boxes of this prepartition. For each `x`, we choose `r (Nx x)` as the radius
    at `x`. -/
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    Nx : (ι → Real) → Nat
    hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
    hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  set r : ℝ≥0 → (ι → ℝ) → Ioi (0 : ℝ) := fun c x => (hfi' <| Nx x).convergenceR (δ <| Nx x) c x
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    Nx : (ι → Real) → Nat
    hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
    hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c : NNReal
    hδc : HasSum δ c
    hcε : LT.lt c ε
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  refine ⟨r, fun c => l.rCond_of_bRiemann_eq_false hl, fun c π hπ hπp => ?_⟩
  /- Now we prove the estimate in 3 "jumps": first we replace `g x` in the formula for the
    integral sum by `f (Nx x)`; then we replace each `μ J • f (Nx (π.tag J)) (π.tag J)`
    by the Bochner integral of `f (Nx (π.tag J)) x` over `J`, then we jump to the Bochner
    integral of `g`. -/
  refine (dist_triangle4 _ (∑ J ∈ π.boxes, (μ J).toReal • f (Nx <| π.tag J) (π.tag J))
    (∑ J ∈ π.boxes, ∫ x in J, f (Nx <| π.tag J) x ∂μ) _).trans ?_
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    Nx : (ι → Real) → Nat
    hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
    hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c (r c) π
    hπp : π.IsPartition
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (Dist.dist (BoxIntegral.integralSum g μ.toBoxAdd …
  -/
  rw [add_mul, add_mul, one_mul]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : (ι → Real) → E
    hg : MeasureTheory.StronglyMeasurable g
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
    hgi : MeasureTheory.IntegrableOn g (↑I) μ
    f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
    hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
    hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
    hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
    ε : NNReal
    ε0 : LT.lt 0 ε
    ε0' : LT.lt 0 ↑ε
    N₀ : Nat
    hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
    Nx : (ι → Real) → Nat
    hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
    hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
    δ : Nat → NNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    c✝ : NNReal
    hδc : HasSum δ c✝
    hcε : LT.lt c✝ ε
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
    c : NNReal
    π : BoxIntegral.TaggedPrepartition I
    hπ : l.MemBaseSet I c (r c) π
    hπp : π.IsPartition
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (Dist.dist (BoxIntegral.integralSum g μ.toBoxAdd …
  -/
  refine add_le_add_three ?_ ?_ ?_
  · /- Since each `f (Nx <| π.tag J)` is `ε`-close to `g (π.tag J)`, replacing the latter with
        the former in the formula for the integral sum changes the sum at most by `μ I * ε`. -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      ⊢ LE.le (Dist.dist (BoxIntegral.integralSum g μ.toBoxAdditive.toSMul π) (π.box …
    -/
    rw [← hπp.iUnion_eq, π.measure_iUnion_toReal, sum_mul, integralSum]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      ⊢ LE.le (Dist.dist (π.boxes.sum fun J => (μ.toBoxAdditive.toSMul J) (g (π.tag  …
    -/
    refine dist_sum_sum_le_of_le _ fun J _ => ?_; dsimp
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      J : BoxIntegral.Box ι
      x✝ : Membership.mem π.boxes J
      ⊢ LE.le (Dist.dist (HSMul.hSMul (μ ↑J).toReal (g (π.tag J))) (HSMul.hSMul (μ ↑ …
    -/
    rw [dist_eq_norm, ← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      J : BoxIntegral.Box ι
      x✝ : Membership.mem π.boxes J
      ⊢ LE.le (HMul.hMul (μ ↑J).toReal (Norm.norm (HSub.hSub (g (π.tag J)) ((f (Nx ( …
    -/
    gcongr
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1.h
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      J : BoxIntegral.Box ι
      x✝ : Membership.mem π.boxes J
      ⊢ LE.le (Norm.norm (HSub.hSub (g (π.tag J)) ((f (Nx (π.tag J))) (π.tag J)))) ↑ε
    -/
    rw [← dist_eq_norm']; exact hNxε _
                          /-
                            🎉 no goals
                          -/
  · /- We group the terms of both sums by the values of `Nx (π.tag J)`.
        For each `N`, the sum of Bochner integrals over the boxes is equal
        to the sum of box integrals, and the sum of box integrals is `δᵢ`-close
        to the corresponding integral sum due to the Henstock-Sacks inequality. -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      ⊢ LE.le (Dist.dist (π.boxes.sum fun J => HSMul.hSMul (μ ↑J).toReal ((f (Nx (π. …
    -/
    rw [← π.sum_fiberwise fun J => Nx (π.tag J), ← π.sum_fiberwise fun J => Nx (π.tag J)]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      ⊢ LE.le (Dist.dist ((Finset.image (fun J => Nx (π.tag J)) π.boxes).sum fun y = …
    -/
    refine le_trans ?_ (NNReal.coe_lt_coe.2 hcε).le
    refine
      (dist_sum_sum_le_of_le _ fun n hn => ?_).trans
        (sum_le_hasSum _ (fun n _ => (δ n).2) (NNReal.hasSum_coe.2 hδc))
    have hNxn : ∀ J ∈ π.filter fun J => Nx (π.tag J) = n, Nx (π.tag J) = n := fun J hJ =>
      (π.mem_filter.1 hJ).2
    have hrn : ∀ J ∈ π.filter fun J => Nx (π.tag J) = n,
        r c (π.tag J) = (hfi' n).convergenceR (δ n) c (π.tag J) := fun J hJ ↦ by
      obtain rfl := hNxn J hJ
      rfl
    have :
        l.MemBaseSet I c ((hfi' n).convergenceR (δ n) c) (π.filter fun J => Nx (π.tag J) = n) :=
      (hπ.filter _).mono' _ le_rfl le_rfl fun J hJ => (hrn J hJ).le
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      n : Nat
      hn : Membership.mem (Finset.image (fun J => Nx (π.tag J)) π.boxes) n
      hNxn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π. …
      hrn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π.t …
      this : l.MemBaseSet I c (⋯.convergenceR (↑(δ n)) c) (π.filter fun J => Eq (Nx  …
      ⊢ LE.le (Dist.dist ((π.filter fun J => Eq (Nx (π.tag J)) n).boxes.sum fun J => …
    -/
    convert (hfi' n).dist_integralSum_sum_integral_le_of_memBaseSet (δ0 _) this using 2
      /-
        case h.e'_3.h.e'_3
        ι : Type u
        E : Type v
        inst✝⁴ : Fintype ι
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        μ : MeasureTheory.Measure (ι → Real)
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        I : BoxIntegral.Box ι
        l : BoxIntegral.IntegrationParams
        hl : Eq l.bRiemann Bool.false
        this✝² : MeasurableSpace E := borel E
        this✝¹ : BorelSpace E
        g : (ι → Real) → E
        hg : MeasureTheory.StronglyMeasurable g
        this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
        hgi : MeasureTheory.IntegrableOn g (↑I) μ
        f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
        hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
        hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
        hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
        ε : NNReal
        ε0 : LT.lt 0 ε
        ε0' : LT.lt 0 ↑ε
        N₀ : Nat
        hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
        Nx : (ι → Real) → Nat
        hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
        hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
        δ : Nat → NNReal
        δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
        c✝ : NNReal
        hδc : HasSum δ c✝
        hcε : LT.lt c✝ ε
        r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
        c : NNReal
        π : BoxIntegral.TaggedPrepartition I
        hπ : l.MemBaseSet I c (r c) π
        hπp : π.IsPartition
        n : Nat
        hn : Membership.mem (Finset.image (fun J => Nx (π.tag J)) π.boxes) n
        hNxn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π. …
        hrn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π.t …
        this : l.MemBaseSet I c (⋯.convergenceR (↑(δ n)) c) (π.filter fun J => Eq (Nx  …
        ⊢ Eq ((π.filter fun J => Eq (Nx (π.tag J)) n).boxes.sum fun J => HSMul.hSMul ( …
      -/
    · refine sum_congr rfl fun J hJ => ?_
      /-
        case h.e'_3.h.e'_3
        ι : Type u
        E : Type v
        inst✝⁴ : Fintype ι
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        μ : MeasureTheory.Measure (ι → Real)
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        I : BoxIntegral.Box ι
        l : BoxIntegral.IntegrationParams
        hl : Eq l.bRiemann Bool.false
        this✝² : MeasurableSpace E := borel E
        this✝¹ : BorelSpace E
        g : (ι → Real) → E
        hg : MeasureTheory.StronglyMeasurable g
        this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
        hgi : MeasureTheory.IntegrableOn g (↑I) μ
        f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
        hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
        hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
        hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
        ε : NNReal
        ε0 : LT.lt 0 ε
        ε0' : LT.lt 0 ↑ε
        N₀ : Nat
        hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
        Nx : (ι → Real) → Nat
        hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
        hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
        δ : Nat → NNReal
        δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
        c✝ : NNReal
        hδc : HasSum δ c✝
        hcε : LT.lt c✝ ε
        r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
        c : NNReal
        π : BoxIntegral.TaggedPrepartition I
        hπ : l.MemBaseSet I c (r c) π
        hπp : π.IsPartition
        n : Nat
        hn : Membership.mem (Finset.image (fun J => Nx (π.tag J)) π.boxes) n
        hNxn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π. …
        hrn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π.t …
        this : l.MemBaseSet I c (⋯.convergenceR (↑(δ n)) c) (π.filter fun J => Eq (Nx  …
        J : BoxIntegral.Box ι
        hJ : Membership.mem (π.filter fun J => Eq (Nx (π.tag J)) n).boxes J
        ⊢ Eq (HSMul.hSMul (μ ↑J).toReal ((f (Nx (π.tag J))) (π.tag J))) ((μ.toBoxAddit …
      -/
      simp [hNxn J hJ]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_4
        ι : Type u
        E : Type v
        inst✝⁴ : Fintype ι
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        μ : MeasureTheory.Measure (ι → Real)
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        I : BoxIntegral.Box ι
        l : BoxIntegral.IntegrationParams
        hl : Eq l.bRiemann Bool.false
        this✝² : MeasurableSpace E := borel E
        this✝¹ : BorelSpace E
        g : (ι → Real) → E
        hg : MeasureTheory.StronglyMeasurable g
        this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
        hgi : MeasureTheory.IntegrableOn g (↑I) μ
        f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
        hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
        hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
        hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
        ε : NNReal
        ε0 : LT.lt 0 ε
        ε0' : LT.lt 0 ↑ε
        N₀ : Nat
        hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
        Nx : (ι → Real) → Nat
        hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
        hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
        δ : Nat → NNReal
        δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
        c✝ : NNReal
        hδc : HasSum δ c✝
        hcε : LT.lt c✝ ε
        r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
        c : NNReal
        π : BoxIntegral.TaggedPrepartition I
        hπ : l.MemBaseSet I c (r c) π
        hπp : π.IsPartition
        n : Nat
        hn : Membership.mem (Finset.image (fun J => Nx (π.tag J)) π.boxes) n
        hNxn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π. …
        hrn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π.t …
        this : l.MemBaseSet I c (⋯.convergenceR (↑(δ n)) c) (π.filter fun J => Eq (Nx  …
        ⊢ Eq ((π.filter fun J => Eq (Nx (π.tag J)) n).boxes.sum fun J => MeasureTheory …
      -/
    · refine sum_congr rfl fun J hJ => ?_
      rw [← SimpleFunc.integral_eq_integral, SimpleFunc.box_integral_eq_integral _ _ _ _ hl,
        hNxn J hJ]
      /-
        case h.e'_3.h.e'_4.hfi
        ι : Type u
        E : Type v
        inst✝⁴ : Fintype ι
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        μ : MeasureTheory.Measure (ι → Real)
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        I : BoxIntegral.Box ι
        l : BoxIntegral.IntegrationParams
        hl : Eq l.bRiemann Bool.false
        this✝² : MeasurableSpace E := borel E
        this✝¹ : BorelSpace E
        g : (ι → Real) → E
        hg : MeasureTheory.StronglyMeasurable g
        this✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton …
        hgi : MeasureTheory.IntegrableOn g (↑I) μ
        f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
        hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
        hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
        hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
        ε : NNReal
        ε0 : LT.lt 0 ε
        ε0' : LT.lt 0 ↑ε
        N₀ : Nat
        hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
        Nx : (ι → Real) → Nat
        hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
        hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
        δ : Nat → NNReal
        δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
        c✝ : NNReal
        hδc : HasSum δ c✝
        hcε : LT.lt c✝ ε
        r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
        c : NNReal
        π : BoxIntegral.TaggedPrepartition I
        hπ : l.MemBaseSet I c (r c) π
        hπp : π.IsPartition
        n : Nat
        hn : Membership.mem (Finset.image (fun J => Nx (π.tag J)) π.boxes) n
        hNxn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π. …
        hrn : ∀ (J : BoxIntegral.Box ι), Membership.mem (π.filter fun J => Eq (Nx (π.t …
        this : l.MemBaseSet I c (⋯.convergenceR (↑(δ n)) c) (π.filter fun J => Eq (Nx  …
        J : BoxIntegral.Box ι
        hJ : Membership.mem (π.filter fun J => Eq (Nx (π.tag J)) n).boxes J
        ⊢ MeasureTheory.Integrable (⇑(f (Nx (π.tag J)))) (μ.restrict ↑J)
      -/
      exact (hfi _).mono_set (Prepartition.le_of_mem _ hJ)
      /-
        🎉 no goals
      -/
  · /-  For the last jump, we use the fact that the distance between `f (Nx x) x` and `g x` is less
        than or equal to the distance between `f N₀ x` and `g x` and the integral of
        `‖f N₀ x - g x‖` is less than or equal to `ε`. -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      ⊢ LE.le (Dist.dist (π.boxes.sum fun J => MeasureTheory.integral (μ.restrict ↑J …
    -/
    refine le_trans ?_ hN₀
    have hfi : ∀ (n), ∀ J ∈ π, IntegrableOn (f n) (↑J) μ := fun n J hJ =>
      (hfi n).mono_set (π.le_of_mem' J hJ)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi✝ : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      hfi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory. …
      ⊢ LE.le (Dist.dist (π.boxes.sum fun J => MeasureTheory.integral (μ.restrict ↑J …
    -/
    have hgi : ∀ J ∈ π, IntegrableOn g (↑J) μ := fun J hJ => hgi.mono_set (π.le_of_mem' J hJ)
    have hfgi : ∀ (n), ∀ J ∈ π, IntegrableOn (fun x => ‖f n x - g x‖) J μ := fun n J hJ =>
      ((hfi n J hJ).sub (hgi J hJ)).norm
    rw [← hπp.iUnion_eq, Prepartition.iUnion_def',
      integral_finset_biUnion π.boxes (fun J _ => J.measurableSet_coe) π.pairwiseDisjoint hgi,
      integral_finset_biUnion π.boxes (fun J _ => J.measurableSet_coe) π.pairwiseDisjoint (hfgi _)]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi✝ : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi✝ : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      hfi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory. …
      hgi : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory.Integrable …
      hfgi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory …
      ⊢ LE.le (Dist.dist (π.boxes.sum fun J => MeasureTheory.integral (μ.restrict ↑J …
    -/
    refine dist_sum_sum_le_of_le _ fun J hJ => ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi✝ : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi✝ : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      hfi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory. …
      hgi : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory.Integrable …
      hfgi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π.boxes J
      ⊢ LE.le (Dist.dist (MeasureTheory.integral (μ.restrict ↑J) fun x => (f (Nx (π. …
    -/
    rw [dist_eq_norm, ← integral_sub (hfi _ J hJ) (hgi J hJ)]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi✝ : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi✝ : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      hfi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory. …
      hgi : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory.Integrable …
      hfgi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π.boxes J
      ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict ↑J) fun a => HSub.hSub  …
    -/
    refine norm_integral_le_of_norm_le (hfgi _ J hJ) (Eventually.of_forall fun x => ?_)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      hl : Eq l.bRiemann Bool.false
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      g : (ι → Real) → E
      hg : MeasureTheory.StronglyMeasurable g
      this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range g) (Singleton. …
      hgi✝ : MeasureTheory.IntegrableOn g (↑I) μ
      f : Nat → MeasureTheory.SimpleFunc (ι → Real) E := MeasureTheory.SimpleFunc.ap …
      hfi✝ : ∀ (n : Nat), MeasureTheory.IntegrableOn (⇑(f n)) (↑I) μ
      hfi' : ∀ (n : Nat), BoxIntegral.Integrable I l (⇑(f n)) μ.toBoxAdditive.toSMul
      hfg_mono : ∀ (x : ι → Real) {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hS …
      ε : NNReal
      ε0 : LT.lt 0 ε
      ε0' : LT.lt 0 ↑ε
      N₀ : Nat
      hN₀ : LE.le (MeasureTheory.integral (μ.restrict ↑I) fun x => Norm.norm (HSub.h …
      Nx : (ι → Real) → Nat
      hNx : ∀ (x : ι → Real), LE.le N₀ (Nx x)
      hNxε : ∀ (x : ι → Real), LE.le (Dist.dist ((f (Nx x)) x) (g x)) ↑ε
      δ : Nat → NNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      c✝ : NNReal
      hδc : HasSum δ c✝
      hcε : LT.lt c✝ ε
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0) := fun c x => ⋯.convergenceR (↑(δ (Nx x …
      c : NNReal
      π : BoxIntegral.TaggedPrepartition I
      hπ : l.MemBaseSet I c (r c) π
      hπp : π.IsPartition
      hfi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory. …
      hgi : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory.Integrable …
      hfgi : ∀ (n : Nat) (J : BoxIntegral.Box ι), Membership.mem π J → MeasureTheory …
      J : BoxIntegral.Box ι
      hJ : Membership.mem π.boxes J
      x : ι → Real
      ⊢ LE.le (Norm.norm (HSub.hSub ((f (Nx (π.tag J))) x) (g x))) (Norm.norm (HSub. …
    -/
    exact hfg_mono x (hNx (π.tag J))
    /-
      🎉 no goals
    -/


/-- If `f : ℝⁿ → E` is continuous on a rectangular box `I`, then it is Box integrable on `I`
w.r.t. a locally finite measure `μ` with the same integral. -/
theorem ContinuousOn.hasBoxIntegral [CompleteSpace E] {f : (ι → ℝ) → E} (μ : Measure (ι → ℝ))
    [IsLocallyFiniteMeasure μ] {I : Box ι} (hc : ContinuousOn f (Box.Icc I))
    (l : IntegrationParams) :
    HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul (∫ x in I, f x ∂μ) := by
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    l : BoxIntegral.IntegrationParams
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  obtain ⟨y, hy⟩ := BoxIntegral.integrable_of_continuousOn l hc μ
  /-
    case intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    l : BoxIntegral.IntegrationParams
    y : E
    hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  convert hy
  have : IntegrableOn f I μ :=
    IntegrableOn.mono_set (hc.integrableOn_compact I.isCompact_Icc) Box.coe_subset_Icc
  /-
    case h.e'_13
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    l : BoxIntegral.IntegrationParams
    y : E
    hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    this : MeasureTheory.IntegrableOn f (↑I) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict ↑I) fun x => f x) y
  -/
  exact HasIntegral.unique (IntegrableOn.hasBoxIntegral this ⊥ rfl) (HasIntegral.mono hy bot_le)
  /-
    🎉 no goals
  -/


/-- If `f : ℝⁿ → E` is a.e. continuous and bounded on a rectangular box `I`, then it is Box
    integrable on `I` w.r.t. a locally finite measure `μ` with the same integral. -/
theorem AEContinuous.hasBoxIntegral [CompleteSpace E] {f : (ι → ℝ) → E} (μ : Measure (ι → ℝ))
    [IsLocallyFiniteMeasure μ] {I : Box ι} (hb : ∃ C : ℝ, ∀ x ∈ Box.Icc I, ‖f x‖ ≤ C)
    (hc : ∀ᵐ x ∂μ, ContinuousAt f x) (l : IntegrationParams) :
    HasIntegral.{u, v, v} I l f μ.toBoxAdditive.toSMul (∫ x in I, f x ∂μ) := by
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
    l : BoxIntegral.IntegrationParams
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  obtain ⟨y, hy⟩ := integrable_of_bounded_and_ae_continuous l hb μ hc
  /-
    case intro
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
    l : BoxIntegral.IntegrationParams
    y : E
    hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    ⊢ BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul (MeasureTheory.integral …
  -/
  convert hy
  /-
    case h.e'_13
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
    l : BoxIntegral.IntegrationParams
    y : E
    hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    ⊢ Eq (MeasureTheory.integral (μ.restrict ↑I) fun x => f x) y
  -/
  refine HasIntegral.unique (IntegrableOn.hasBoxIntegral ?_ ⊥ rfl) (HasIntegral.mono hy bot_le)
  /-
    case h.e'_13
    ι : Type u
    E : Type v
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    I : BoxIntegral.Box ι
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
    l : BoxIntegral.IntegrationParams
    y : E
    hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
    ⊢ MeasureTheory.IntegrableOn f (↑I) μ
  -/
  constructor
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict ↑I)
    -/
  · let v := {x : (ι → ℝ) | ContinuousAt f x}
    have : AEStronglyMeasurable f (μ.restrict v) :=
      (continuousOn_of_forall_continuousAt fun _ h ↦ h).aestronglyMeasurable
      (measurableSet_of_continuousAt f)
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict ↑I)
    -/
    refine this.mono_measure (Measure.le_iff.2 fun s hs ↦ ?_)
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ LE.le ((μ.restrict ↑I) s) ((μ.restrict v) s)
    -/
    repeat rw [μ.restrict_apply hs]
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ LE.le (μ (Inter.inter s ↑I)) (μ (Inter.inter s v))
    -/
    apply le_of_le_of_eq <| μ.mono s.inter_subset_left
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ Eq (μ.measureOf s) (μ (Inter.inter s v))
    -/
    refine measure_eq_measure_of_null_diff s.inter_subset_left ?_ |>.symm
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ Eq (μ (SDiff.sdiff s (Inter.inter s v))) 0
    -/
    rw [diff_self_inter, Set.diff_eq]
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ Eq (μ (Inter.inter s (HasCompl.compl v))) 0
    -/
    refine (le_antisymm (zero_le (μ (s ∩ vᶜ))) ?_).symm
    /-
      case h.e'_13.left
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      v : Set (ι → Real) := setOf fun x => ContinuousAt f x
      this : MeasureTheory.AEStronglyMeasurable f (μ.restrict v)
      s : Set (ι → Real)
      hs : MeasurableSet s
      ⊢ LE.le (μ (Inter.inter s (HasCompl.compl v))) 0
    -/
    exact le_trans (μ.mono s.inter_subset_right) (nonpos_iff_eq_zero.2 hc)
    /-
      🎉 no goals
    -/
  · have : IsFiniteMeasure (μ.restrict (Box.Icc I)) :=
      { measure_univ_lt_top := by simp [I.isCompact_Icc.measure_lt_top (μ := μ)] }
    have : IsFiniteMeasure (μ.restrict I) :=
      isFiniteMeasure_of_le (μ.restrict (Box.Icc I))
                            (μ.restrict_mono Box.coe_subset_Icc (le_refl μ))
    /-
      case h.e'_13.right
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      this✝ : MeasureTheory.IsFiniteMeasure (μ.restrict (BoxIntegral.Box.Icc I))
      this : MeasureTheory.IsFiniteMeasure (μ.restrict ↑I)
      ⊢ MeasureTheory.HasFiniteIntegral f (μ.restrict ↑I)
    -/
    obtain ⟨C, hC⟩ := hb
    /-
      case h.e'_13.right.intro
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      this✝ : MeasureTheory.IsFiniteMeasure (μ.restrict (BoxIntegral.Box.Icc I))
      this : MeasureTheory.IsFiniteMeasure (μ.restrict ↑I)
      C : Real
      hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
      ⊢ MeasureTheory.HasFiniteIntegral f (μ.restrict ↑I)
    -/
    refine hasFiniteIntegral_of_bounded (C := C) (Filter.eventually_iff_exists_mem.2 ?_)
    /-
      case h.e'_13.right.intro
      ι : Type u
      E : Type v
      inst✝⁴ : Fintype ι
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      f : (ι → Real) → E
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      I : BoxIntegral.Box ι
      hc : Filter.Eventually (fun x => ContinuousAt f x) (MeasureTheory.ae μ)
      l : BoxIntegral.IntegrationParams
      y : E
      hy : BoxIntegral.HasIntegral I l f μ.toBoxAdditive.toSMul y
      this✝ : MeasureTheory.IsFiniteMeasure (μ.restrict (BoxIntegral.Box.Icc I))
      this : MeasureTheory.IsFiniteMeasure (μ.restrict ↑I)
      C : Real
      hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
      ⊢ Exists fun v => And (Membership.mem (MeasureTheory.ae (μ.restrict ↑I)) v) (∀ …
    -/
    use I, self_mem_ae_restrict I.measurableSet_coe, fun y hy ↦ hC y (I.coe_subset_Icc hy)
    /-
      🎉 no goals
    -/


