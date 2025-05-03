/-- **Blichfeldt's Theorem**. If the volume of the set `s` is larger than the covolume of the
countable subgroup `L` of `E`, then there exist two distinct points `x, y ∈ L` such that `(x + s)`
and `(y + s)` are not disjoint. -/
theorem exists_pair_mem_lattice_not_disjoint_vadd [AddCommGroup L] [Countable L] [AddAction L E]
    [MeasurableSpace L] [MeasurableVAdd L E] [VAddInvariantMeasure L E μ]
    (fund : IsAddFundamentalDomain L F μ) (hS : NullMeasurableSet s μ) (h : μ F < μ s) :
    ∃ x y : L, x ≠ y ∧ ¬Disjoint (x +ᵥ s) (y +ᵥ s) := by
  /-
    E : Type u_1
    L : Type u_2
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : AddCommGroup L
    inst✝⁴ : Countable L
    inst✝³ : AddAction L E
    inst✝² : MeasurableSpace L
    inst✝¹ : MeasurableVAdd L E
    inst✝ : MeasureTheory.VAddInvariantMeasure L E μ
    fund : MeasureTheory.IsAddFundamentalDomain L F μ
    hS : MeasureTheory.NullMeasurableSet s μ
    h : LT.lt (μ F) (μ s)
    ⊢ Exists fun x => Exists fun y => And (Ne x y) (Not (Disjoint (HVAdd.hVAdd x s …
  -/
  contrapose! h
  exact ((fund.measure_eq_tsum _).trans (measure_iUnion₀
    (Pairwise.mono h fun i j hij => (hij.mono inf_le_left inf_le_left).aedisjoint)
      fun _ => (hS.vadd _).inter fund.nullMeasurableSet).symm).trans_le
      (measure_mono <| Set.iUnion_subset fun _ => Set.inter_subset_right)


/-- The **Minkowski Convex Body Theorem**. If `s` is a convex symmetric domain of `E` whose volume
is large enough compared to the covolume of a lattice `L` of `E`, then it contains a non-zero
lattice point of `L`. -/
theorem exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure [NormedAddCommGroup E]
    [NormedSpace ℝ E] [BorelSpace E] [FiniteDimensional ℝ E] [IsAddHaarMeasure μ]
    {L : AddSubgroup E} [Countable L] (fund : IsAddFundamentalDomain L F μ)
    (h_symm : ∀ x ∈ s, -x ∈ s) (h_conv : Convex ℝ s) (h : μ F * 2 ^ finrank ℝ E < μ s) :
    ∃ x ≠ 0, ((x : L) : E) ∈ s := by
  have h_vol : μ F < μ ((2⁻¹ : ℝ) • s) := by
    rw [addHaar_smul_of_nonneg μ (by norm_num : 0 ≤ (2 : ℝ)⁻¹) s, ←
      mul_lt_mul_right (pow_ne_zero (finrank ℝ E) (two_ne_zero' _)) (pow_ne_top two_ne_top),
      mul_right_comm, ofReal_pow (by norm_num : 0 ≤ (2 : ℝ)⁻¹), ofReal_inv_of_pos zero_lt_two]
    norm_num
    rwa [← mul_pow, ENNReal.inv_mul_cancel two_ne_zero two_ne_top, one_pow, one_mul]
  obtain ⟨x, y, hxy, h⟩ :=
    exists_pair_mem_lattice_not_disjoint_vadd fund ((h_conv.smul _).nullMeasurableSet _) h_vol
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝ : Countable (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h✝ : LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_vol : LT.lt (μ F) (μ (HSMul.hSMul (Inv.inv 2) s))
    x y : Subtype fun x => Membership.mem L x
    hxy : Ne x y
    h : Not (Disjoint (HVAdd.hVAdd x (HSMul.hSMul (Inv.inv 2) s)) (HVAdd.hVAdd y ( …
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  obtain ⟨_, ⟨v, hv, rfl⟩, w, hw, hvw⟩ := Set.not_disjoint_iff.mp h
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝ : Countable (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h✝ : LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_vol : LT.lt (μ F) (μ (HSMul.hSMul (Inv.inv 2) s))
    x y : Subtype fun x => Membership.mem L x
    hxy : Ne x y
    h : Not (Disjoint (HVAdd.hVAdd x (HSMul.hSMul (Inv.inv 2) s)) (HVAdd.hVAdd y ( …
    v : E
    hv : Membership.mem (HSMul.hSMul (Inv.inv 2) s) v
    w : E
    hw : Membership.mem (HSMul.hSMul (Inv.inv 2) s) w
    hvw : Eq ((fun x => HVAdd.hVAdd y x) w) ((fun x_1 => HVAdd.hVAdd x x_1) v)
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  refine ⟨x - y, sub_ne_zero.2 hxy, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝ : Countable (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h✝ : LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_vol : LT.lt (μ F) (μ (HSMul.hSMul (Inv.inv 2) s))
    x y : Subtype fun x => Membership.mem L x
    hxy : Ne x y
    h : Not (Disjoint (HVAdd.hVAdd x (HSMul.hSMul (Inv.inv 2) s)) (HVAdd.hVAdd y ( …
    v : E
    hv : Membership.mem (HSMul.hSMul (Inv.inv 2) s) v
    w : E
    hw : Membership.mem (HSMul.hSMul (Inv.inv 2) s) w
    hvw : Eq ((fun x => HVAdd.hVAdd y x) w) ((fun x_1 => HVAdd.hVAdd x x_1) v)
    ⊢ Membership.mem s ↑(HSub.hSub x y)
  -/
  rw [Set.mem_inv_smul_set_iff₀ (two_ne_zero' ℝ)] at hv hw
  simp_rw [AddSubgroup.vadd_def, vadd_eq_add, add_comm _ w, ← sub_eq_sub_iff_add_eq_add, ←
    AddSubgroup.coe_sub] at hvw
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝ : Countable (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h✝ : LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_vol : LT.lt (μ F) (μ (HSMul.hSMul (Inv.inv 2) s))
    x y : Subtype fun x => Membership.mem L x
    hxy : Ne x y
    h : Not (Disjoint (HVAdd.hVAdd x (HSMul.hSMul (Inv.inv 2) s)) (HVAdd.hVAdd y ( …
    v : E
    hv : Membership.mem s (HSMul.hSMul 2 v)
    w : E
    hw : Membership.mem s (HSMul.hSMul 2 w)
    hvw : Eq (HSub.hSub w v) ↑(HSub.hSub x y)
    ⊢ Membership.mem s ↑(HSub.hSub x y)
  -/
  rw [← hvw, ← inv_smul_smul₀ (two_ne_zero' ℝ) (_ - _), smul_sub, sub_eq_add_neg, smul_add]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝ : Countable (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h✝ : LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_vol : LT.lt (μ F) (μ (HSMul.hSMul (Inv.inv 2) s))
    x y : Subtype fun x => Membership.mem L x
    hxy : Ne x y
    h : Not (Disjoint (HVAdd.hVAdd x (HSMul.hSMul (Inv.inv 2) s)) (HVAdd.hVAdd y ( …
    v : E
    hv : Membership.mem s (HSMul.hSMul 2 v)
    w : E
    hw : Membership.mem s (HSMul.hSMul 2 w)
    hvw : Eq (HSub.hSub w v) ↑(HSub.hSub x y)
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul (Inv.inv 2) (HSMul.hSMul 2 w)) (HSM …
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  refine h_conv hw (h_symm _ hv) ?_ ?_ ?_ <;> norm_num
                                              /-
                                                🎉 no goals
                                              -/


/-- The **Minkowski Convex Body Theorem for compact domain**. If `s` is a convex compact symmetric
domain of `E` whose volume is large enough compared to the covolume of a lattice `L` of `E`, then it
contains a non-zero lattice point of `L`. Compared to
`exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure`, this version requires in addition
that `s` is compact and `L` is discrete but provides a weaker inequality rather than a strict
inequality. -/
theorem exists_ne_zero_mem_lattice_of_measure_mul_two_pow_le_measure [NormedAddCommGroup E]
    [NormedSpace ℝ E] [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E] [IsAddHaarMeasure μ]
    {L : AddSubgroup E} [Countable L] [DiscreteTopology L] (fund : IsAddFundamentalDomain L F μ)
    (h_symm : ∀ x ∈ s, -x ∈ s) (h_conv : Convex ℝ s) (h_cpt : IsCompact s)
    (h : μ F * 2 ^ finrank ℝ E ≤ μ s) :
    ∃ x ≠ 0, ((x : L) : E) ∈ s := by
  have h_mes : μ s ≠ 0 := by
    intro hμ
    suffices μ F = 0 from fund.measure_ne_zero (NeZero.ne μ) this
    rw [hμ, le_zero_iff, mul_eq_zero] at h
    exact h.resolve_right <| pow_ne_zero _ two_ne_zero
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  have h_nemp : s.Nonempty := nonempty_of_measure_ne_zero h_mes
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    h_nemp : s.Nonempty
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  let u : ℕ → ℝ≥0 := (exists_seq_strictAnti_tendsto 0).choose
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    h_nemp : s.Nonempty
    u : Nat → NNReal := ⋯.choose
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  let K : ConvexBody E := ⟨s, h_conv, h_cpt, h_nemp⟩
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    h_nemp : s.Nonempty
    u : Nat → NNReal := ⋯.choose
    K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  let S : ℕ → ConvexBody E := fun n => (1 + u n) • K
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    h_nemp : s.Nonempty
    u : Nat → NNReal := ⋯.choose
    K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
    S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  let Z : ℕ → Set E := fun n => (S n) ∩ (L \ {0})
  -- The convex bodies `S n` have volume strictly larger than `μ s` and thus we can apply
  -- `exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure` to them and obtain that
  -- `S n` contains a nonzero point of `L`. Since the intersection of the `S n` is equal to `s`,
  -- it follows that `s` contains a nonzero point of `L`.
  /-
    E : Type u_1
    inst✝⁸ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    F s : Set E
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : BorelSpace E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : Nontrivial E
    inst✝² : μ.IsAddHaarMeasure
    L : AddSubgroup E
    inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
    h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    h_conv : Convex Real s
    h_cpt : IsCompact s
    h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
    h_mes : Ne (μ s) 0
    h_nemp : s.Nonempty
    u : Nat → NNReal := ⋯.choose
    K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
    S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
    Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem s ↑x)
  -/
  have h_zero : 0 ∈ K := K.zero_mem_of_symmetric h_symm
  suffices Set.Nonempty (⋂ n, Z n) by
    erw [← Set.iInter_inter, K.iInter_smul_eq_self h_zero] at this
    · obtain ⟨x, hx⟩ := this
      exact ⟨⟨x, by aesop⟩, by aesop⟩
    · exact (exists_seq_strictAnti_tendsto (0 : ℝ≥0)).choose_spec.2.2
  have h_clos : IsClosed ((L : Set E) \ {0}) := by
    rsuffices ⟨U, hU⟩ : ∃ U : Set E, IsOpen U ∧  U ∩ L = {0}
    · rw [sdiff_eq_sdiff_iff_inf_eq_inf (z := U).mpr (by simp [Set.inter_comm .. ▸ hU.2, zero_mem])]
      exact AddSubgroup.isClosed_of_discrete.sdiff hU.1
    exact isOpen_inter_eq_singleton_of_mem_discrete (zero_mem L)
  refine IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed Z (fun n => ?_)
    (fun n => ?_) ((S 0).isCompact.inter_right h_clos) (fun n => (S n).isClosed.inter h_clos)
    /-
      case refine_1
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ HasSubset.Subset (Z (HAdd.hAdd n 1)) (Z n)
    -/
  · refine Set.inter_subset_inter_left _ (SetLike.coe_subset_coe.mpr ?_)
    /-
      case refine_1
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LE.le (S (HAdd.hAdd n 1)) (S n)
    -/
    refine ConvexBody.smul_le_of_le K h_zero ?_
    /-
      case refine_1
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LE.le (HAdd.hAdd 1 (u (HAdd.hAdd n 1))) (HAdd.hAdd 1 (u n))
    -/
    rw [add_le_add_iff_left]
    /-
      case refine_1
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LE.le (u (HAdd.hAdd n 1)) (u n)
    -/
    exact le_of_lt <| (exists_seq_strictAnti_tendsto (0 : ℝ≥0)).choose_spec.1 (Nat.lt.base n)
    /-
      🎉 no goals
    -/
  · suffices μ F * 2 ^ finrank ℝ E < μ (S n : Set E) by
      have h_symm' : ∀ x ∈ S n, -x ∈ S n := by
        rintro _ ⟨y, hy, rfl⟩
        exact ⟨-y, h_symm _ hy, by simp⟩
      obtain ⟨x, hx_nz, hx_mem⟩ := exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure
        fund h_symm' (S n).convex this
      exact ⟨x, hx_mem, by aesop⟩
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ ↑(S n))
    -/
    refine lt_of_le_of_lt h ?_
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt (μ s) (μ ↑(S n))
    -/
    rw [ConvexBody.coe_smul', NNReal.smul_def, addHaar_smul_of_nonneg _ (NNReal.coe_nonneg _)]
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt (μ s) (HMul.hMul (ENNReal.ofReal (HPow.hPow (↑(HAdd.hAdd 1 (u n))) (Mo …
    -/
    rw [show μ s < _ ↔ 1 * μ s < _ by rw [one_mul]]
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt (HMul.hMul 1 (μ s)) (HMul.hMul (ENNReal.ofReal (HPow.hPow (↑(HAdd.hAdd …
    -/
    refine (mul_lt_mul_right h_mes (ne_of_lt h_cpt.measure_lt_top)).mpr ?_
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt 1 (ENNReal.ofReal (HPow.hPow (↑(HAdd.hAdd 1 (u n))) (Module.finrank Re …
    -/
    rw [ofReal_pow (NNReal.coe_nonneg _)]
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt 1 (HPow.hPow (ENNReal.ofReal ↑(HAdd.hAdd 1 (u n))) (Module.finrank Rea …
    -/
    refine one_lt_pow₀ ?_ (ne_of_gt finrank_pos)
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      F s : Set E
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : Nontrivial E
      inst✝² : μ.IsAddHaarMeasure
      L : AddSubgroup E
      inst✝¹ : Countable (Subtype fun x => Membership.mem L x)
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L …
      h_symm : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      h_conv : Convex Real s
      h_cpt : IsCompact s
      h : LE.le (HMul.hMul (μ F) (HPow.hPow 2 (Module.finrank Real E))) (μ s)
      h_mes : Ne (μ s) 0
      h_nemp : s.Nonempty
      u : Nat → NNReal := ⋯.choose
      K : ConvexBody E := { carrier := s, convex' := h_conv, isCompact' := h_cpt, no …
      S : Nat → ConvexBody E := fun n => HSMul.hSMul (HAdd.hAdd 1 (u n)) K
      Z : Nat → Set E := fun n => Inter.inter (↑(S n)) (SDiff.sdiff (↑L) (Singleton. …
      h_zero : Membership.mem K 0
      h_clos : IsClosed (SDiff.sdiff (↑L) (Singleton.singleton 0))
      n : Nat
      ⊢ LT.lt 1 (ENNReal.ofReal ↑(HAdd.hAdd 1 (u n)))
    -/
    simp [u, K, S, Z, (exists_seq_strictAnti_tendsto (0 : ℝ≥0)).choose_spec.2.1 n]
    /-
      🎉 no goals
    -/


