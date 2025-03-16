/-- A family `𝒜` of sets of a measure space is said to be measure-dense if it contains only
measurable sets and can approximate any measurable set with finite measure, in the sense that
for any measurable set `s` with finite measure the symmetric difference `s ∆ t` can be made
arbitrarily small when `t ∈ 𝒜`. We show below that such a family can be chosen to contain only
sets with finite measures.

The term "measure-dense" is justified by the fact that the approximating condition translates
to the usual notion of density in the metric space made by constant indicators of measurable sets
equipped with the `Lᵖ` norm. -/
structure Measure.MeasureDense (μ : Measure X) (𝒜 : Set (Set X)) : Prop where
  /-- Each set has to be measurable. -/
  measurable : ∀ s ∈ 𝒜, MeasurableSet s
  /-- Any measurable set can be approximated by sets in the family. -/
  approx : ∀ s, MeasurableSet s → μ s ≠ ∞ → ∀ ε : ℝ, 0 < ε → ∃ t ∈ 𝒜, μ (s ∆ t) < ENNReal.ofReal ε


theorem Measure.MeasureDense.nonempty (h𝒜 : μ.MeasureDense 𝒜) : 𝒜.Nonempty := by
  /-
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    ⊢ 𝒜.Nonempty
  -/
  rcases h𝒜.approx ∅ MeasurableSet.empty (by simp) 1 (by norm_num) with ⟨t, ht, -⟩
  /-
    case intro.intro
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    t : Set X
    ht : Membership.mem 𝒜 t
    ⊢ 𝒜.Nonempty
  -/
  exact ⟨t, ht⟩
  /-
    🎉 no goals
  -/


theorem Measure.MeasureDense.nonempty' (h𝒜 : μ.MeasureDense 𝒜) :
    {s | s ∈ 𝒜 ∧ μ s ≠ ∞}.Nonempty := by
  /-
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    ⊢ (setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)).Nonempty
  -/
  rcases h𝒜.approx ∅ MeasurableSet.empty (by simp) 1 (by norm_num) with ⟨t, ht, hμt⟩
  /-
    case intro.intro
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    t : Set X
    ht : Membership.mem 𝒜 t
    hμt : LT.lt (μ (symmDiff EmptyCollection.emptyCollection t)) (ENNReal.ofReal 1)
    ⊢ (setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)).Nonempty
  -/
  refine ⟨t, ht, ?_⟩
  /-
    case intro.intro
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    t : Set X
    ht : Membership.mem 𝒜 t
    hμt : LT.lt (μ (symmDiff EmptyCollection.emptyCollection t)) (ENNReal.ofReal 1)
    ⊢ Ne (μ t) Top.top
  -/
  convert ne_top_of_lt hμt
  /-
    case h.e'_2.h.e'_6
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    t : Set X
    ht : Membership.mem 𝒜 t
    hμt : LT.lt (μ (symmDiff EmptyCollection.emptyCollection t)) (ENNReal.ofReal 1)
    ⊢ Eq t (symmDiff EmptyCollection.emptyCollection t)
  -/
  rw [← bot_eq_empty, bot_symmDiff]
  /-
    🎉 no goals
  -/


/-- The set of measurable sets is measure-dense. -/
theorem measureDense_measurableSet : μ.MeasureDense {s | MeasurableSet s} where
  measurable _ h := h
                                      /-
                                        X : Type u_1
                                        m : MeasurableSpace X
                                        μ : MeasureTheory.Measure X
                                        s : Set X
                                        hs : MeasurableSet s
                                        x✝ : Ne (μ s) Top.top
                                        ε : Real
                                        ε_pos : LT.lt 0 ε
                                        ⊢ LT.lt (μ (symmDiff s s)) (ENNReal.ofReal ε)
                                      -/
  approx s hs _ ε ε_pos := ⟨s, hs, by simpa⟩
                                      /-
                                        🎉 no goals
                                      -/


/-- If a family of sets `𝒜` is measure-dense in `X`, then any measurable set with finite measure
can be approximated by sets in `𝒜` with finite measure. -/
lemma Measure.MeasureDense.fin_meas_approx (h𝒜 : μ.MeasureDense 𝒜) {s : Set X}
    (ms : MeasurableSet s) (hμs : μ s ≠ ∞) (ε : ℝ) (ε_pos : 0 < ε) :
    ∃ t ∈ 𝒜, μ t ≠ ∞ ∧ μ (s ∆ t) < ENNReal.ofReal ε := by
  /-
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    s : Set X
    ms : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : Real
    ε_pos : LT.lt 0 ε
    ⊢ Exists fun t => And (Membership.mem 𝒜 t) (And (Ne (μ t) Top.top) (LT.lt (μ ( …
  -/
  rcases h𝒜.approx s ms hμs ε ε_pos with ⟨t, t_mem, ht⟩
  /-
    case intro.intro
    X : Type u_1
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    s : Set X
    ms : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : Real
    ε_pos : LT.lt 0 ε
    t : Set X
    t_mem : Membership.mem 𝒜 t
    ht : LT.lt (μ (symmDiff s t)) (ENNReal.ofReal ε)
    ⊢ Exists fun t => And (Membership.mem 𝒜 t) (And (Ne (μ t) Top.top) (LT.lt (μ ( …
  -/
  exact ⟨t, t_mem, (measure_ne_top_iff_of_symmDiff <| ne_top_of_lt ht).1 hμs, ht⟩
  /-
    🎉 no goals
  -/


variable (p) in
/-- If `𝒜` is a measure-dense family of sets and `c : E`, then the set of constant indicators
with constant `c` whose underlying set is in `𝒜` is dense in the set of constant indicators
which are in `Lp E p μ` when `1 ≤ p < ∞`. -/
theorem Measure.MeasureDense.indicatorConstLp_subset_closure (h𝒜 : μ.MeasureDense 𝒜) (c : E) :
    {indicatorConstLp p hs hμs c | (s : Set X) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)} ⊆
    closure {indicatorConstLp p (h𝒜.measurable s hs) hμs c |
      (s : Set X) (hs : s ∈ 𝒜) (hμs : μ s ≠ ∞)} := by
  /-
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜 : Set (Set X)
    h𝒜 : μ.MeasureDense 𝒜
    c : E
    ⊢ HasSubset.Subset (setOf fun x => Exists fun s => Exists fun hs => Exists fun …
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      ⊢ HasSubset.Subset (setOf fun x => Exists fun s => Exists fun hs => Exists fun …
    -/
  · refine Subset.trans ?_ subset_closure
    /-
      case inl
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      ⊢ HasSubset.Subset (setOf fun x => Exists fun s => Exists fun hs => Exists fun …
    -/
    rintro - ⟨s, ms, hμs, rfl⟩
    /-
      case inl.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ Membership.mem (setOf fun x => Exists fun s => Exists fun hs => Exists fun h …
    -/
    obtain ⟨t, ht, hμt⟩ := h𝒜.nonempty'
    /-
      case inl.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      t : Set X
      ht : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      ⊢ Membership.mem (setOf fun x => Exists fun s => Exists fun hs => Exists fun h …
    -/
    refine ⟨t, ht, hμt, ?_⟩
    /-
      case inl.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      t : Set X
      ht : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      ⊢ Eq (MeasureTheory.indicatorConstLp p ⋯ hμt 0) (MeasureTheory.indicatorConstL …
    -/
    simp_rw [indicatorConstLp]
    /-
      case inl.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      t : Set X
      ht : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      ⊢ Eq (MeasureTheory.Memℒp.toLp (t.indicator fun x => 0) ⋯) (MeasureTheory.Memℒ …
    -/
    congr
    /-
      case inl.intro.intro.intro.intro.intro.e_f
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      t : Set X
      ht : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      ⊢ Eq (t.indicator fun x => 0) (s.indicator fun x => 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      ⊢ HasSubset.Subset (setOf fun x => Exists fun s => Exists fun hs => Exists fun …
    -/
  · have p_pos : 0 < p := lt_of_lt_of_le (by norm_num) one_le_p.elim
    /-
      case inr
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      p_pos : LT.lt 0 p
      ⊢ HasSubset.Subset (setOf fun x => Exists fun s => Exists fun hs => Exists fun …
    -/
    rintro - ⟨s, ms, hμs, rfl⟩
    /-
      case inr.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      p_pos : LT.lt 0 p
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ Membership.mem (closure (setOf fun x => Exists fun s => Exists fun hs => Exi …
    -/
    refine Metric.mem_closure_iff.2 fun ε hε ↦ ?_
    /-
      case inr.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      p_pos : LT.lt 0 p
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      hε : GT.gt ε 0
      ⊢ Exists fun b => And (Membership.mem (setOf fun x => Exists fun s => Exists f …
    -/
    have aux : 0 < (ε / ‖c‖) ^ p.toReal := rpow_pos_of_pos (div_pos hε (norm_pos_iff.2 hc)) _
    /-
      case inr.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      p_pos : LT.lt 0 p
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      hε : GT.gt ε 0
      aux : LT.lt 0 (HPow.hPow (HDiv.hDiv ε (Norm.norm c)) p.toReal)
      ⊢ Exists fun b => And (Membership.mem (setOf fun x => Exists fun s => Exists f …
    -/
    obtain ⟨t, ht, hμt, hμst⟩ := h𝒜.fin_meas_approx ms hμs ((ε / ‖c‖) ^ p.toReal) aux
    refine ⟨indicatorConstLp p (h𝒜.measurable t ht) hμt c,
      ⟨t, ht, hμt, rfl⟩, ?_⟩
    /-
      case inr.intro.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      c : E
      hc : Ne c 0
      p_pos : LT.lt 0 p
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      hε : GT.gt ε 0
      aux : LT.lt 0 (HPow.hPow (HDiv.hDiv ε (Norm.norm c)) p.toReal)
      t : Set X
      ht : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      hμst : LT.lt (μ (symmDiff s t)) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε (Norm. …
      ⊢ LT.lt (Dist.dist (MeasureTheory.indicatorConstLp p ms hμs c) (MeasureTheory. …
    -/
    rw [dist_indicatorConstLp_eq_norm, norm_indicatorConstLp p_pos.ne.symm p_ne_top.elim]
    calc
      ‖c‖ * (μ (s ∆ t)).toReal ^ (1 / p.toReal)
        < ‖c‖ * (ENNReal.ofReal ((ε / ‖c‖) ^ p.toReal)).toReal ^ (1 / p.toReal) := by
          rw [_root_.mul_lt_mul_left (norm_pos_iff.2 hc)]
          refine Real.rpow_lt_rpow (by simp) ?_
            (one_div_pos.2 <| toReal_pos p_pos.ne.symm p_ne_top.elim)
          rwa [toReal_lt_toReal (measure_symmDiff_ne_top hμs hμt) ofReal_ne_top]
      _ = ε := by
        rw [toReal_ofReal (rpow_nonneg (div_nonneg hε.le (norm_nonneg _)) _),
          one_div, Real.rpow_rpow_inv (div_nonneg hε.le (norm_nonneg _))
            (toReal_pos p_pos.ne.symm p_ne_top.elim).ne.symm,
          mul_div_cancel₀ _ (norm_ne_zero_iff.2 hc)]


/-- If a family of sets `𝒜` is measure-dense in `X`, then it is also the case for the sets in `𝒜`
with finite measure. -/
theorem Measure.MeasureDense.fin_meas (h𝒜 : μ.MeasureDense 𝒜) :
    μ.MeasureDense {s | s ∈ 𝒜 ∧ μ s ≠ ∞} where
  measurable s h := h𝒜.measurable s h.1
  approx s ms hμs ε ε_pos := by
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ Exists fun t => And (Membership.mem (setOf fun s => And (Membership.mem 𝒜 s) …
    -/
    rcases Measure.MeasureDense.fin_meas_approx h𝒜 ms hμs ε ε_pos with ⟨t, t_mem, hμt, hμst⟩
    /-
      case intro.intro.intro
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : μ.MeasureDense 𝒜
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      t : Set X
      t_mem : Membership.mem 𝒜 t
      hμt : Ne (μ t) Top.top
      hμst : LT.lt (μ (symmDiff s t)) (ENNReal.ofReal ε)
      ⊢ Exists fun t => And (Membership.mem (setOf fun s => And (Membership.mem 𝒜 s) …
    -/
    exact ⟨t, ⟨t_mem, hμt⟩, hμst⟩
    /-
      🎉 no goals
    -/


/-- If a measurable space equipped with a finite measure is generated by an algebra of sets, then
this algebra of sets is measure-dense. -/
theorem Measure.MeasureDense.of_generateFrom_isSetAlgebra_finite [IsFiniteMeasure μ]
    (h𝒜 : IsSetAlgebra 𝒜) (hgen : m = MeasurableSpace.generateFrom 𝒜) : μ.MeasureDense 𝒜 where
  measurable s hs := hgen ▸ measurableSet_generateFrom hs
  approx s ms := by
    -- We want to show that any measurable set can be approximated by sets in `𝒜`. To do so, it is
    -- enough to show that such sets constitute a `σ`-algebra containing `𝒜`. This is contained in
    -- the theorem `generateFrom_induction`.
    have : MeasurableSet s ∧ ∀ (ε : ℝ), 0 < ε → ∃ t ∈ 𝒜, (μ (s ∆ t)).toReal < ε := by
      induction s, hgen ▸ ms using generateFrom_induction with
      -- If `t ∈ 𝒜`, then `μ (t ∆ t) = 0` which is less than any `ε > 0`.
      | hC t t_mem _ =>
        exact ⟨hgen ▸ measurableSet_generateFrom t_mem, fun ε ε_pos ↦ ⟨t, t_mem, by simpa⟩⟩
      -- `∅ ∈ 𝒜` and `μ (∅ ∆ ∅) = 0` which is less than any `ε > 0`.
      | empty => exact ⟨MeasurableSet.empty, fun ε ε_pos ↦ ⟨∅, h𝒜.empty_mem, by simpa⟩⟩
      -- If `s` is measurable and `t ∈ 𝒜` such that `μ (s ∆ t) < ε`, then `tᶜ ∈ 𝒜` and
      -- `μ (sᶜ ∆ tᶜ) = μ (s ∆ t) < ε` so `sᶜ` can be approximated.
      | compl t _ ht =>
        refine ⟨ht.1.compl, fun ε ε_pos ↦ ?_⟩
        obtain ⟨u, u_mem, hμtcu⟩ := ht.2 ε ε_pos
        exact ⟨uᶜ, h𝒜.compl_mem u_mem, by rwa [compl_symmDiff_compl]⟩
      -- Let `(fₙ)` be a sequence of measurable sets and `ε > 0`.
      | iUnion f _ hf =>
        refine ⟨MeasurableSet.iUnion (fun n ↦ (hf n).1), fun ε ε_pos ↦ ?_⟩
        -- We have  `μ (⋃ n ≤ N, fₙ) ⟶ μ (⋃ n, fₙ)`.
        have := tendsto_measure_iUnion_accumulate (μ := μ) (f := f)
        rw [← tendsto_toReal_iff (fun _ ↦ measure_ne_top _ _) (measure_ne_top _ _)] at this
        -- So there exists `N` such that `μ (⋃ n, fₙ) - μ (⋃ n ≤ N, fₙ) < ε/2`.
        rcases Metric.tendsto_atTop.1 this (ε / 2) (by linarith [ε_pos]) with ⟨N, hN⟩
        -- For any `n ≤ N` there exists `gₙ ∈ 𝒜` such that `μ (fₙ ∆ gₙ) < ε/(2*(N+1))`.
        choose g g_mem hg using fun n ↦ (hf n).2 (ε / (2 * (N + 1))) (div_pos ε_pos (by linarith))
        -- Therefore we have
        -- `μ ((⋃ n, fₙ) ∆ (⋃ n ≤ N, gₙ))`
        --   `≤ μ ((⋃ n, fₙ) ∆ (⋃ n ≤ N, fₙ)) + μ ((⋃ n ≤ N, fₙ) ∆ (⋃ n ≤ N, gₙ))`
        --   `< ε/2 + ∑ n ≤ N, μ (fₙ ∆ gₙ)` (see `biSup_symmDiff_biSup_le`)
        --   `< ε/2 + (N+1)*ε/(2*(N+1)) = ε/2`.
        refine ⟨⋃ n ∈ Finset.range (N + 1), g n, h𝒜.biUnion_mem _ (fun i _ ↦ g_mem i), ?_⟩
        calc
          (μ ((⋃ n, f n) ∆ (⋃ n ∈ (Finset.range (N + 1)), g n))).toReal
            ≤ (μ ((⋃ n, f n) \ ((⋃ n ∈ (Finset.range (N + 1)), f n)) ∪
              ((⋃ n ∈ (Finset.range (N + 1)), f n) ∆
              (⋃ n ∈ (Finset.range (N + 1)), g ↑n)))).toReal :=
                toReal_mono (measure_ne_top _ _)
                  (measure_mono <| symmDiff_of_ge (iUnion_subset <|
                  fun i ↦ iUnion_subset (fun _ ↦ subset_iUnion f i)) ▸ symmDiff_triangle ..)
          _ ≤ (μ ((⋃ n, f n) \
              ((⋃ n ∈ (Finset.range (N + 1)), f n)))).toReal +
              (μ ((⋃ n ∈ (Finset.range (N + 1)), f n) ∆
              (⋃ n ∈ (Finset.range (N + 1)), g ↑n))).toReal := by
                rw [← toReal_add (measure_ne_top _ _) (measure_ne_top _ _)]
                exact toReal_mono (add_ne_top.2 ⟨measure_ne_top _ _, measure_ne_top _ _⟩) <|
                  measure_union_le _ _
          _ < ε := by
                rw [← add_halves ε]
                apply _root_.add_lt_add
                · rw [measure_diff (h_fin := measure_ne_top _ _),
                    toReal_sub_of_le (ha := measure_ne_top _ _)]
                  · apply lt_of_le_of_lt (sub_le_dist ..)
                    simp only [Finset.mem_range, Nat.lt_add_one_iff]
                    exact (dist_comm (α := ℝ) .. ▸ hN N (le_refl N))
                  · exact measure_mono <| iUnion_subset <|
                      fun i ↦ iUnion_subset fun _ ↦ subset_iUnion f i
                  · exact iUnion_subset <| fun i ↦ iUnion_subset (fun _ ↦ subset_iUnion f i)
                  · exact MeasurableSet.biUnion (countable_coe_iff.1 inferInstance)
                      (fun n _ ↦ (hf n).1.nullMeasurableSet)
                · calc
                    (μ ((⋃ n ∈ (Finset.range (N + 1)), f n) ∆
                    (⋃ n ∈ (Finset.range (N + 1)), g ↑n))).toReal
                      ≤ (μ (⋃ n ∈ (Finset.range (N + 1)), f n ∆ g n)).toReal :=
                          toReal_mono (measure_ne_top _ _) (measure_mono biSup_symmDiff_biSup_le)
                    _ ≤ ∑ n in (Finset.range (N + 1)), (μ (f n ∆ g n)).toReal := by
                          rw [← toReal_sum (fun _ _ ↦ measure_ne_top _ _)]
                          exact toReal_mono (ne_of_lt <| sum_lt_top.2 fun _ _ ↦ measure_lt_top μ _)
                            (measure_biUnion_finset_le _ _)
                    _ < ∑ n in (Finset.range (N + 1)), (ε / (2 * (N + 1))) :=
                          Finset.sum_lt_sum (fun i _ ↦ le_of_lt (hg i)) ⟨0, by simp, hg 0⟩
                    _ ≤ ε / 2 := by
                          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul,
                            Nat.cast_add, Nat.cast_one]
                          rw [div_mul_eq_div_mul_one_div, ← mul_assoc, mul_comm ((N : ℝ) + 1),
                            mul_assoc]
                          exact mul_le_of_le_one_right (by linarith [ε_pos]) <|
                            le_of_eq <| mul_one_div_cancel <| Nat.cast_add_one_ne_zero _
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      this : And (MeasurableSet s) (∀ (ε : Real), LT.lt 0 ε → Exists fun t => And (M …
      ⊢ Ne (μ s) Top.top → ∀ (ε : Real), LT.lt 0 ε → Exists fun t => And (Membership …
    -/
    rintro - ε ε_pos
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      this : And (MeasurableSet s) (∀ (ε : Real), LT.lt 0 ε → Exists fun t => And (M …
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    rcases this.2 ε ε_pos with ⟨t, t_mem, hμst⟩
    /-
      case intro.intro
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      this : And (MeasurableSet s) (∀ (ε : Real), LT.lt 0 ε → Exists fun t => And (M …
      ε : Real
      ε_pos : LT.lt 0 ε
      t : Set X
      t_mem : Membership.mem 𝒜 t
      hμst : LT.lt (μ (symmDiff s t)).toReal ε
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    exact ⟨t, t_mem, (lt_ofReal_iff_toReal_lt (measure_ne_top _ _)).2 hμst⟩
    /-
      🎉 no goals
    -/


/-- If a measure space `X` is generated by an algebra of sets which contains a monotone countable
family of sets with finite measure spanning `X` (thus the measure is `σ`-finite), then this algebra
of sets is measure-dense. -/
theorem Measure.MeasureDense.of_generateFrom_isSetAlgebra_sigmaFinite (h𝒜 : IsSetAlgebra 𝒜)
    (S : μ.FiniteSpanningSetsIn 𝒜) (hgen : m = MeasurableSpace.generateFrom 𝒜) :
    μ.MeasureDense 𝒜 where
  measurable s hs := hgen ▸ measurableSet_generateFrom hs
  approx s ms hμs ε ε_pos := by
    -- We use partial unions of (Sₙ) to get a monotone family spanning `X`.
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      S : μ.FiniteSpanningSetsIn 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    let T := Accumulate S.set
    have T_mem (n) : T n ∈ 𝒜 := by
      simpa using h𝒜.biUnion_mem {k | k ≤ n}.toFinset (fun k _ ↦ S.set_mem k)
    have T_finite (n) : μ (T n) < ∞ := by
      simpa using measure_biUnion_lt_top {k | k ≤ n}.toFinset.finite_toSet
        (fun k _ ↦ S.finite k)
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      S : μ.FiniteSpanningSetsIn 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      T : Nat → Set X := Set.Accumulate S.set
      T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
      T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    have T_spanning : ⋃ n, T n = univ := S.spanning ▸ iUnion_accumulate
    -- We use the fact that we already know this is true for finite measures. As `⋃ n, T n = X`,
    -- we have that `μ ((T n) ∩ s) ⟶ μ s`.
    have mono : Monotone (fun n ↦ (T n) ∩ s) := fun m n hmn ↦ inter_subset_inter_left s
        (biUnion_subset_biUnion_left fun k hkm ↦ Nat.le_trans hkm hmn)
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      S : μ.FiniteSpanningSetsIn 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      T : Nat → Set X := Set.Accumulate S.set
      T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
      T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
      T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
      mono : Monotone fun n => Inter.inter (T n) s
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    have := tendsto_measure_iUnion_atTop (μ := μ) mono
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝒜 : Set (Set X)
      h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
      S : μ.FiniteSpanningSetsIn 𝒜
      hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
      s : Set X
      ms : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : Real
      ε_pos : LT.lt 0 ε
      T : Nat → Set X := Set.Accumulate S.set
      T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
      T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
      T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
      mono : Monotone fun n => Inter.inter (T n) s
      this : Filter.Tendsto (Function.comp ⇑μ fun n => Inter.inter (T n) s) Filter.a …
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
    -/
    rw [← tendsto_toReal_iff] at this
    · -- We can therefore choose `N` such that `μ s - μ ((S N) ∩ s) < ε/2`.
      /-
        X : Type u_1
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝒜 : Set (Set X)
        h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
        S : μ.FiniteSpanningSetsIn 𝒜
        hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
        s : Set X
        ms : MeasurableSet s
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        T : Nat → Set X := Set.Accumulate S.set
        T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
        T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
        T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
        mono : Monotone fun n => Inter.inter (T n) s
        this : Filter.Tendsto (fun n => (Function.comp (⇑μ) (fun n => Inter.inter (T n …
        ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
      -/
      rcases Metric.tendsto_atTop.1 this (ε / 2) (by linarith [ε_pos]) with ⟨N, hN⟩
      /-
        case intro
        X : Type u_1
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝒜 : Set (Set X)
        h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
        S : μ.FiniteSpanningSetsIn 𝒜
        hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
        s : Set X
        ms : MeasurableSet s
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        T : Nat → Set X := Set.Accumulate S.set
        T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
        T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
        T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
        mono : Monotone fun n => Inter.inter (T n) s
        this : Filter.Tendsto (fun n => (Function.comp (⇑μ) (fun n => Inter.inter (T n …
        N : Nat
        hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (Function.comp (⇑μ) (fun n => I …
        ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
      -/
      have : Fact (μ (T N) < ∞) := Fact.mk <| T_finite N
      -- Then we can apply the previous result to the measure `μ ((S N) ∩ •)`.
      -- There exists `t ∈ 𝒜` such that `μ ((S N) ∩ (s ∆ t)) < ε/2`.
      rcases (Measure.MeasureDense.of_generateFrom_isSetAlgebra_finite
        (μ := μ.restrict (T N)) h𝒜 hgen).approx s ms
        (ne_of_lt (lt_of_le_of_lt (μ.restrict_apply_le _ s) hμs.lt_top))
        (ε / 2) (by linarith [ε_pos])
        with ⟨t, t_mem, ht⟩
      -- We can then use `t ∩ (S N)`, because `S N ∈ 𝒜` by hypothesis.
      -- `μ (s ∆ (t ∩ S N))`
      --   `≤ μ (s ∆ (s ∩ S N)) + μ ((s ∩ S N) ∆ (t ∩ S N))`
      --   `= μ s - μ (s ∩ S N) + μ (s ∆ t) ∩ S N) < ε`.
      /-
        case intro.intro.intro
        X : Type u_1
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝒜 : Set (Set X)
        h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
        S : μ.FiniteSpanningSetsIn 𝒜
        hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
        s : Set X
        ms : MeasurableSet s
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        T : Nat → Set X := Set.Accumulate S.set
        T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
        T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
        T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
        mono : Monotone fun n => Inter.inter (T n) s
        this✝ : Filter.Tendsto (fun n => (Function.comp (⇑μ) (fun n => Inter.inter (T  …
        N : Nat
        hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (Function.comp (⇑μ) (fun n => I …
        this : Fact (LT.lt (μ (T N)) Top.top)
        t : Set X
        t_mem : Membership.mem 𝒜 t
        ht : LT.lt ((μ.restrict (T N)) (symmDiff s t)) (ENNReal.ofReal (HDiv.hDiv ε 2))
        ⊢ Exists fun t => And (Membership.mem 𝒜 t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
      -/
      refine ⟨t ∩ T N, h𝒜.inter_mem t_mem (T_mem N), ?_⟩
      calc
        μ (s ∆ (t ∩ T N))
          ≤ μ (s \ (s ∩ T N)) + μ ((s ∆ t) ∩ T N) := by
              rw [← symmDiff_of_le (inter_subset_left ..), symmDiff_comm _ s,
                inter_symmDiff_distrib_right]
              exact measure_symmDiff_le _ _ _
        _ < ENNReal.ofReal (ε / 2) + ENNReal.ofReal (ε / 2) := by
              apply ENNReal.add_lt_add
              · rw [measure_diff
                    (inter_subset_left ..)
                    (ms.inter (hgen ▸ measurableSet_generateFrom (T_mem N))).nullMeasurableSet
                    (ne_top_of_le_ne_top hμs (measure_mono (inter_subset_left ..))),
                  lt_ofReal_iff_toReal_lt (sub_ne_top hμs),
                  toReal_sub_of_le (measure_mono (inter_subset_left ..)) hμs]
                apply lt_of_le_of_lt (sub_le_dist ..)
                nth_rw 1 [← univ_inter s]
                rw [inter_comm s, dist_comm, ← T_spanning, iUnion_inter _ T]
                apply hN N (le_refl _)
              · rwa [← μ.restrict_apply' (hgen ▸ measurableSet_generateFrom (T_mem N))]
        _ = ENNReal.ofReal ε := by
              rw [← ofReal_add (by linarith [ε_pos]) (by linarith [ε_pos]), add_halves]
      /-
        case hf
        X : Type u_1
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝒜 : Set (Set X)
        h𝒜 : MeasureTheory.IsSetAlgebra 𝒜
        S : μ.FiniteSpanningSetsIn 𝒜
        hgen : Eq m (MeasurableSpace.generateFrom 𝒜)
        s : Set X
        ms : MeasurableSet s
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        T : Nat → Set X := Set.Accumulate S.set
        T_mem : ∀ (n : Nat), Membership.mem 𝒜 (T n)
        T_finite : ∀ (n : Nat), LT.lt (μ (T n)) Top.top
        T_spanning : Eq (Set.iUnion fun n => T n) Set.univ
        mono : Monotone fun n => Inter.inter (T n) s
        this : Filter.Tendsto (Function.comp ⇑μ fun n => Inter.inter (T n) s) Filter.a …
        ⊢ ∀ (i : Nat), Ne (Function.comp (⇑μ) (fun n => Inter.inter (T n) s) i) Top.top
      -/
    · exact fun n ↦ ne_top_of_le_ne_top hμs (measure_mono (inter_subset_right ..))
      /-
        🎉 no goals
      -/
    · exact ne_top_of_le_ne_top hμs
        (measure_mono (iUnion_subset (fun i ↦ inter_subset_right ..)))


/-- A measure `μ` is separable if there exists a countable and measure-dense family of sets.

The term "separable" is justified by the fact that the definition translates to the usual notion
of separability in the metric space made by constant indicators equipped with the `Lᵖ` norm. -/
class IsSeparable (μ : Measure X) : Prop where
  exists_countable_measureDense : ∃ 𝒜, 𝒜.Countable ∧ μ.MeasureDense 𝒜


/-- By definition, a separable measure admits a countable and measure-dense family of sets. -/
theorem exists_countable_measureDense [IsSeparable μ] :
    ∃ 𝒜, 𝒜.Countable ∧ μ.MeasureDense 𝒜 :=
  IsSeparable.exists_countable_measureDense


/-- If a measurable space is countably generated and equipped with a `σ`-finite measure, then the
measure is separable. This is not an instance because it is used below to prove the more
general case where `μ` is s-finite. -/
theorem isSeparable_of_sigmaFinite [CountablyGenerated X] [SigmaFinite μ] :
    IsSeparable μ where
  exists_countable_measureDense := by
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SigmaFinite μ
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    have h := countable_countableGeneratingSet (α := X)
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SigmaFinite μ
      h : (MeasurableSpace.countableGeneratingSet X).Countable
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    have hgen := generateFrom_countableGeneratingSet (α := X)
    /-
      X : Type u_1
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SigmaFinite μ
      h : (MeasurableSpace.countableGeneratingSet X).Countable
      hgen : Eq (MeasurableSpace.generateFrom (MeasurableSpace.countableGeneratingSe …
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    let 𝒜 := (countableGeneratingSet X) ∪ {μ.toFiniteSpanningSetsIn.set n | n : ℕ}
    have count_𝒜 : 𝒜.Countable :=
      countable_union.2 ⟨h, countable_iff_exists_subset_range.2
        ⟨μ.toFiniteSpanningSetsIn.set, fun _ hx ↦ hx⟩⟩
    refine ⟨generateSetAlgebra 𝒜, countable_generateSetAlgebra count_𝒜,
      Measure.MeasureDense.of_generateFrom_isSetAlgebra_sigmaFinite isSetAlgebra_generateSetAlgebra
      { set := μ.toFiniteSpanningSetsIn.set
        set_mem := fun n ↦ self_subset_generateSetAlgebra (𝒜 := 𝒜) <| Or.inr ⟨n, rfl⟩
        finite := μ.toFiniteSpanningSetsIn.finite
        spanning := μ.toFiniteSpanningSetsIn.spanning }
      (le_antisymm ?_ (generateFrom_le (fun s hs ↦ ?_)))⟩
      /-
        case refine_1
        X : Type u_1
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : MeasurableSpace.CountablyGenerated X
        inst✝ : MeasureTheory.SigmaFinite μ
        h : (MeasurableSpace.countableGeneratingSet X).Countable
        hgen : Eq (MeasurableSpace.generateFrom (MeasurableSpace.countableGeneratingSe …
        𝒜 : Set (Set X) := Union.union (MeasurableSpace.countableGeneratingSet X) (set …
        count_𝒜 : 𝒜.Countable
        ⊢ LE.le m (MeasurableSpace.generateFrom (MeasureTheory.generateSetAlgebra 𝒜))
      -/
    · rw [← hgen]
      exact generateFrom_mono <| le_trans self_subset_generateSetAlgebra <|
        generateSetAlgebra_mono <| subset_union_left ..
    · induction hs with
      | base t t_mem =>
        rcases t_mem with t_mem | ⟨n, rfl⟩
        · exact hgen ▸ measurableSet_generateFrom t_mem
        · exact μ.toFiniteSpanningSetsIn.set_mem n
      | empty => exact MeasurableSet.empty
      | compl t _ t_mem => exact MeasurableSet.compl t_mem
      | union t u _ _ t_mem u_mem => exact MeasurableSet.union t_mem u_mem


/-- If a measurable space is countably generated and equipped with an s-finite measure, then the
measure is separable. -/
instance [CountablyGenerated X] [SFinite μ] : IsSeparable μ where
  exists_countable_measureDense := by
    /-
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SFinite μ
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    have := isSeparable_of_sigmaFinite (μ := μ.restrict μ.sigmaFiniteSet)
    /-
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜 : Set (Set X)
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SFinite μ
      this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    rcases exists_countable_measureDense (μ := μ.restrict μ.sigmaFiniteSet) with ⟨𝒜, count_𝒜, h𝒜⟩
    /-
      case intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SFinite μ
      this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    let ℬ := {s ∩ μ.sigmaFiniteSet | s ∈ 𝒜}
    /-
      case intro.intro
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasurableSpace.CountablyGenerated X
      inst✝ : MeasureTheory.SFinite μ
      this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
      ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
      ⊢ Exists fun 𝒜 => And 𝒜.Countable (μ.MeasureDense 𝒜)
    -/
    refine ⟨ℬ, count_𝒜.image (fun s ↦ s ∩ μ.sigmaFiniteSet), ?_, ?_⟩
      /-
        case intro.intro.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasurableSpace.CountablyGenerated X
        inst✝ : MeasureTheory.SFinite μ
        this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
        ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
        ⊢ ∀ (s : Set X), Membership.mem ℬ s → MeasurableSet s
      -/
    · rintro - ⟨s, s_mem, rfl⟩
      /-
        case intro.intro.refine_1.intro.intro
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasurableSpace.CountablyGenerated X
        inst✝ : MeasureTheory.SFinite μ
        this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
        ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
        s : Set X
        s_mem : Membership.mem 𝒜 s
        ⊢ MeasurableSet (Inter.inter s μ.sigmaFiniteSet)
      -/
      exact (h𝒜.measurable s s_mem).inter measurableSet_sigmaFiniteSet
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasurableSpace.CountablyGenerated X
        inst✝ : MeasureTheory.SFinite μ
        this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
        ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
        ⊢ ∀ (s : Set X), MeasurableSet s → Ne (μ s) Top.top → ∀ (ε : Real), LT.lt 0 ε  …
      -/
    · intro s ms hμs ε ε_pos
      /-
        case intro.intro.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasurableSpace.CountablyGenerated X
        inst✝ : MeasureTheory.SFinite μ
        this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
        ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
        s : Set X
        ms : MeasurableSet s
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        ⊢ Exists fun t => And (Membership.mem ℬ t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
      -/
      rcases restrict_compl_sigmaFiniteSet_eq_zero_or_top μ s with hs | hs
      · have : (μ.restrict μ.sigmaFiniteSet) s ≠ ∞ :=
          ne_top_of_le_ne_top hμs <| μ.restrict_le_self _
        /-
          case intro.intro.refine_2.inl
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasurableSpace.CountablyGenerated X
          inst✝ : MeasureTheory.SFinite μ
          this✝ : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
          ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
          s : Set X
          ms : MeasurableSet s
          hμs : Ne (μ s) Top.top
          ε : Real
          ε_pos : LT.lt 0 ε
          hs : Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) 0
          this : Ne ((μ.restrict μ.sigmaFiniteSet) s) Top.top
          ⊢ Exists fun t => And (Membership.mem ℬ t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
        -/
        rcases h𝒜.approx s ms this ε ε_pos with ⟨t, t_mem, ht⟩
        /-
          case intro.intro.refine_2.inl.intro.intro
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasurableSpace.CountablyGenerated X
          inst✝ : MeasureTheory.SFinite μ
          this✝ : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
          ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
          s : Set X
          ms : MeasurableSet s
          hμs : Ne (μ s) Top.top
          ε : Real
          ε_pos : LT.lt 0 ε
          hs : Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) 0
          this : Ne ((μ.restrict μ.sigmaFiniteSet) s) Top.top
          t : Set X
          t_mem : Membership.mem 𝒜 t
          ht : LT.lt ((μ.restrict μ.sigmaFiniteSet) (symmDiff s t)) (ENNReal.ofReal ε)
          ⊢ Exists fun t => And (Membership.mem ℬ t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
        -/
        refine ⟨t ∩ μ.sigmaFiniteSet, ⟨t, t_mem, rfl⟩, ?_⟩
        have : μ (s ∆ (t ∩ μ.sigmaFiniteSet) \ μ.sigmaFiniteSet) = 0 := by
          rw [diff_eq_compl_inter, inter_symmDiff_distrib_left, ← ENNReal.bot_eq_zero, eq_bot_iff]
          calc
            μ ((μ.sigmaFiniteSetᶜ ∩ s) ∆ (μ.sigmaFiniteSetᶜ ∩ (t ∩ μ.sigmaFiniteSet)))
              ≤ μ ((μ.sigmaFiniteSetᶜ ∩ s) ∪ (μ.sigmaFiniteSetᶜ ∩ (t ∩ μ.sigmaFiniteSet))) :=
                measure_mono symmDiff_subset_union
            _ ≤ μ (μ.sigmaFiniteSetᶜ ∩ s) + μ (μ.sigmaFiniteSetᶜ ∩ (t ∩ μ.sigmaFiniteSet)) :=
                measure_union_le _ _
            _ = 0 := by
                rw [inter_comm, ← μ.restrict_apply ms, hs, ← inter_assoc, inter_comm,
                  ← inter_assoc, inter_compl_self, empty_inter, measure_empty, zero_add]
        rwa [← measure_inter_add_diff _ measurableSet_sigmaFiniteSet, this, add_zero,
          inter_symmDiff_distrib_right, inter_assoc, inter_self, ← inter_symmDiff_distrib_right,
          ← μ.restrict_apply' measurableSet_sigmaFiniteSet]
        /-
          case intro.intro.refine_2.inr
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasurableSpace.CountablyGenerated X
          inst✝ : MeasureTheory.SFinite μ
          this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
          ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
          s : Set X
          ms : MeasurableSet s
          hμs : Ne (μ s) Top.top
          ε : Real
          ε_pos : LT.lt 0 ε
          hs : Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) Top.top
          ⊢ Exists fun t => And (Membership.mem ℬ t) (LT.lt (μ (symmDiff s t)) (ENNReal. …
        -/
      · refine False.elim <| hμs ?_
        /-
          case intro.intro.refine_2.inr
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasurableSpace.CountablyGenerated X
          inst✝ : MeasureTheory.SFinite μ
          this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
          ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
          s : Set X
          ms : MeasurableSet s
          hμs : Ne (μ s) Top.top
          ε : Real
          ε_pos : LT.lt 0 ε
          hs : Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) Top.top
          ⊢ Eq (μ s) Top.top
        -/
        rw [eq_top_iff, ← hs]
        /-
          case intro.intro.refine_2.inr
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasurableSpace.CountablyGenerated X
          inst✝ : MeasureTheory.SFinite μ
          this : MeasureTheory.IsSeparable (μ.restrict μ.sigmaFiniteSet)
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : (μ.restrict μ.sigmaFiniteSet).MeasureDense 𝒜
          ℬ : Set (Set X) := setOf fun x => Exists fun s => And (Membership.mem 𝒜 s) (Eq …
          s : Set X
          ms : MeasurableSet s
          hμs : Ne (μ s) Top.top
          ε : Real
          ε_pos : LT.lt 0 ε
          hs : Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) Top.top
          ⊢ LE.le ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) (μ s)
        -/
        exact μ.restrict_le_self _
        /-
          🎉 no goals
        -/


/-- If the measure `μ` is separable (in particular if `X` is countably generated and `μ` is
`s`-finite), if `E` is a second-countable `NormedAddCommGroup`, and if `1 ≤ p < +∞`,
then the associated `Lᵖ` space is second-countable. -/
instance Lp.SecondCountableTopology [IsSeparable μ] [TopologicalSpace.SeparableSpace E] :
    SecondCountableTopology (Lp E p μ) := by
  -- It is enough to show that the space is separable, i.e. admits a countable and dense susbet.
  /-
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜 : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    ⊢ _root_.SecondCountableTopology (Subtype fun x => Membership.mem (MeasureTheo …
  -/
  refine @UniformSpace.secondCountable_of_separable _ _ _ ?_
  -- There exists a countable and measure-dense family, and we can keep only the sets with finite
  -- measure while preserving the two properties. This family is denoted `𝒜₀`.
  /-
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜 : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  rcases exists_countable_measureDense (μ := μ) with ⟨𝒜, count_𝒜, h𝒜⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  have h𝒜₀ := Measure.MeasureDense.fin_meas h𝒜
  /-
    case intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    h𝒜₀ : μ.MeasureDense (setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.to …
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  set 𝒜₀ := {s | s ∈ 𝒜 ∧ μ s ≠ ∞}
  /-
    case intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
    h𝒜₀ : μ.MeasureDense 𝒜₀
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  have count_𝒜₀ : 𝒜₀.Countable := count_𝒜.mono fun _ ⟨h, _⟩ ↦ h
  -- `1 ≤ p` so `p ≠ 0`, we prove it now as it is often needed.
  /-
    case intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
    h𝒜₀ : μ.MeasureDense 𝒜₀
    count_𝒜₀ : 𝒜₀.Countable
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  have p_ne_zero : p ≠ 0 := ne_of_gt <| lt_of_lt_of_le (by norm_num) one_le_p.elim
  -- `E` is second-countable, therefore separable and admits a countable and dense subset `u`.
  /-
    case intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
    h𝒜₀ : μ.MeasureDense 𝒜₀
    count_𝒜₀ : 𝒜₀.Countable
    p_ne_zero : Ne p 0
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  rcases exists_countable_dense E with ⟨u, countable_u, dense_u⟩
  -- The countable and dense subset of `Lᵖ` we are going to build is the set of finite sums of
  -- constant indicators with support in `𝒜₀`, and is denoted `D`. To make manipulations easier,
  -- we define the function `key` which given an integer `n` and two families of `n` elements
  -- in `u` and `𝒜₀` associates the corresponding sum.
  let key (n : ℕ) (d : Fin n → u) (s : Fin n → 𝒜₀) : (Lp E p μ) :=
    ∑ i, indicatorConstLp p (h𝒜₀.measurable (s i) (Subtype.mem (s i))) (s i).2.2 (d i : E)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
    h𝒜₀ : μ.MeasureDense 𝒜₀
    count_𝒜₀ : 𝒜₀.Countable
    p_ne_zero : Ne p 0
    u : Set E
    countable_u : u.Countable
    dense_u : Dense u
    key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  let D := {s : Lp E p μ | ∃ n d t, s = key n d t}
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_2
    m : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    p : ENNReal
    one_le_p : Fact (LE.le 1 p)
    p_ne_top : Fact (Ne p Top.top)
    𝒜✝ : Set (Set X)
    inst✝¹ : MeasureTheory.IsSeparable μ
    inst✝ : TopologicalSpace.SeparableSpace E
    𝒜 : Set (Set X)
    count_𝒜 : 𝒜.Countable
    h𝒜 : μ.MeasureDense 𝒜
    𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
    h𝒜₀ : μ.MeasureDense 𝒜₀
    count_𝒜₀ : 𝒜₀.Countable
    p_ne_zero : Ne p 0
    u : Set E
    countable_u : u.Countable
    dense_u : Dense u
    key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
    D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
    ⊢ TopologicalSpace.SeparableSpace (Subtype fun x => Membership.mem (MeasureThe …
  -/
  refine ⟨D, ?_, ?_⟩
  · -- Countability directly follows from countability of `u` and `𝒜₀`. The function `f` below
    -- is the uncurryfied version of `key`, which is easier to manipulate as countability of the
    -- domain is automatically inferred.
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasureTheory.IsSeparable μ
      inst✝ : TopologicalSpace.SeparableSpace E
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : μ.MeasureDense 𝒜
      𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
      h𝒜₀ : μ.MeasureDense 𝒜₀
      count_𝒜₀ : 𝒜₀.Countable
      p_ne_zero : Ne p 0
      u : Set E
      countable_u : u.Countable
      dense_u : Dense u
      key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
      D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
      ⊢ D.Countable
    -/
    let f (nds : Σ n : ℕ, (Fin n → u) × (Fin n → 𝒜₀)) : Lp E p μ := key nds.1 nds.2.1 nds.2.2
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasureTheory.IsSeparable μ
      inst✝ : TopologicalSpace.SeparableSpace E
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : μ.MeasureDense 𝒜
      𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
      h𝒜₀ : μ.MeasureDense 𝒜₀
      count_𝒜₀ : 𝒜₀.Countable
      p_ne_zero : Ne p 0
      u : Set E
      countable_u : u.Countable
      dense_u : Dense u
      key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
      D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
      f : (Sigma fun n => Prod (Fin n → ↑u) (Fin n → ↑𝒜₀)) → Subtype fun x => Member …
      ⊢ D.Countable
    -/
    have := count_𝒜₀.to_subtype
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasureTheory.IsSeparable μ
      inst✝ : TopologicalSpace.SeparableSpace E
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : μ.MeasureDense 𝒜
      𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
      h𝒜₀ : μ.MeasureDense 𝒜₀
      count_𝒜₀ : 𝒜₀.Countable
      p_ne_zero : Ne p 0
      u : Set E
      countable_u : u.Countable
      dense_u : Dense u
      key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
      D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
      f : (Sigma fun n => Prod (Fin n → ↑u) (Fin n → ↑𝒜₀)) → Subtype fun x => Member …
      this : Countable ↑𝒜₀
      ⊢ D.Countable
    -/
    have := countable_u.to_subtype
    have : D ⊆ range f := by
      rintro - ⟨n, d, s, rfl⟩
      use ⟨n, (d, s)⟩
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasureTheory.IsSeparable μ
      inst✝ : TopologicalSpace.SeparableSpace E
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : μ.MeasureDense 𝒜
      𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
      h𝒜₀ : μ.MeasureDense 𝒜₀
      count_𝒜₀ : 𝒜₀.Countable
      p_ne_zero : Ne p 0
      u : Set E
      countable_u : u.Countable
      dense_u : Dense u
      key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
      D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
      f : (Sigma fun n => Prod (Fin n → ↑u) (Fin n → ↑𝒜₀)) → Subtype fun x => Member …
      this✝¹ : Countable ↑𝒜₀
      this✝ : Countable ↑u
      this : HasSubset.Subset D (Set.range f)
      ⊢ D.Countable
    -/
    exact (countable_range f).mono this
    /-
      🎉 no goals
    -/
  · -- Let's turn to the density. Thanks to the density of simple functions in `Lᵖ`, it is enough
    -- to show that the closure of `D` contains constant indicators which are in `Lᵖ` (i. e. the
    -- set has finite measure), is closed by sum and closed.
    -- This is given by `Lp.induction`.
    /-
      case intro.intro.intro.intro.refine_2
      X : Type u_1
      E : Type u_2
      m : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      p : ENNReal
      one_le_p : Fact (LE.le 1 p)
      p_ne_top : Fact (Ne p Top.top)
      𝒜✝ : Set (Set X)
      inst✝¹ : MeasureTheory.IsSeparable μ
      inst✝ : TopologicalSpace.SeparableSpace E
      𝒜 : Set (Set X)
      count_𝒜 : 𝒜.Countable
      h𝒜 : μ.MeasureDense 𝒜
      𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
      h𝒜₀ : μ.MeasureDense 𝒜₀
      count_𝒜₀ : 𝒜₀.Countable
      p_ne_zero : Ne p 0
      u : Set E
      countable_u : u.Countable
      dense_u : Dense u
      key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
      D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
      ⊢ Dense D
    -/
    refine Lp.induction p_ne_top.elim (P := fun f ↦ f ∈ closure D) ?_ ?_ isClosed_closure
      /-
        case intro.intro.intro.intro.refine_2.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        ⊢ ∀ (c : E) {s : Set X} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.top), (f …
      -/
    · intro a s ms hμs
      -- We want to approximate `a • 𝟙ₛ`.
      /-
        case intro.intro.intro.intro.refine_2.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        ⊢ Membership.mem (closure D) ↑(MeasureTheory.Lp.simpleFunc.indicatorConst p ms …
      -/
      apply ne_of_lt at hμs
      /-
        case intro.intro.intro.intro.refine_2.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ⊢ Membership.mem (closure D) ↑(MeasureTheory.Lp.simpleFunc.indicatorConst p ms …
      -/
      rw [SeminormedAddCommGroup.mem_closure_iff]
      /-
        case intro.intro.intro.intro.refine_2.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT.lt (N …
      -/
      intro ε ε_pos
      have μs_pow_nonneg : 0 ≤ (μ s).toReal ^ (1 / p.toReal) :=
        Real.rpow_nonneg ENNReal.toReal_nonneg _
      -- To do so, we first pick `b ∈ u` such that `‖a - b‖ < ε / (3 * (1 + (μ s)^(1/p)))`.
      have approx_a_pos : 0 < ε / (3 * (1 + (μ s).toReal ^ (1 / p.toReal))) :=
        div_pos ε_pos (by linarith [μs_pow_nonneg])
      /-
        case intro.intro.intro.intro.refine_2.refine_1
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        μs_pow_nonneg : LE.le 0 (HPow.hPow (μ s).toReal (HDiv.hDiv 1 p.toReal))
        approx_a_pos : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1 (HPow.hPow (μ s) …
        ⊢ Exists fun b => And (Membership.mem D b) (LT.lt (Norm.norm (HSub.hSub (↑(Mea …
      -/
      have ⟨b, b_mem, hb⟩ := SeminormedAddCommGroup.mem_closure_iff.1 (dense_u a) _ approx_a_pos
      -- Then we pick `t ∈ 𝒜₀` such that `‖b • 𝟙ₛ - b • 𝟙ₜ‖ < ε / 3`.
      rcases SeminormedAddCommGroup.mem_closure_iff.1
        (h𝒜₀.indicatorConstLp_subset_closure p b ⟨s, ms, hμs, rfl⟩)
          (ε / 3) (by linarith [ε_pos]) with ⟨-, ⟨t, ht, hμt, rfl⟩, hst⟩
      /-
        case intro.intro.intro.intro.refine_2.refine_1.intro.intro.intro.intro.intro
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        μs_pow_nonneg : LE.le 0 (HPow.hPow (μ s).toReal (HDiv.hDiv 1 p.toReal))
        approx_a_pos : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1 (HPow.hPow (μ s) …
        b : E
        b_mem : Membership.mem u b
        hb : LT.lt (Norm.norm (HSub.hSub a b)) (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1  …
        t : Set X
        ht : Membership.mem 𝒜₀ t
        hμt : Ne (μ t) Top.top
        hst : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.indicatorConstLp p ms hμs b)  …
        ⊢ Exists fun b => And (Membership.mem D b) (LT.lt (Norm.norm (HSub.hSub (↑(Mea …
      -/
      have mt := h𝒜₀.measurable t ht
      -- We now show that `‖a • 𝟙ₛ - b • 𝟙ₜ‖ₚ < ε`, as follows:
      -- `‖a • 𝟙ₛ - b • 𝟙ₜ‖ₚ`
      --   `= ‖a • 𝟙ₛ - b • 𝟙ₛ + b • 𝟙ₛ - b • 𝟙ₜ‖ₚ`
      --   `≤ ‖a - b‖ * ‖𝟙ₛ‖ₚ + ε / 3`
      --   `= ‖a - b‖ * (μ s)^(1/p) + ε / 3`
      --   `< ε * (μ s)^(1/p) / (3 * (1 + (μ s)^(1/p))) + ε / 3`
      --   `≤ ε / 3 + ε / 3 < ε`.
      refine ⟨indicatorConstLp p mt hμt b,
        ⟨1, fun _ ↦ ⟨b, b_mem⟩, fun _ ↦ ⟨t, ht⟩, by simp [key]⟩, ?_⟩
      rw [Lp.simpleFunc.coe_indicatorConst,
        ← sub_add_sub_cancel _ (indicatorConstLp p ms hμs b), ← add_halves ε]
      /-
        case intro.intro.intro.intro.refine_2.refine_1.intro.intro.intro.intro.intro
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        μs_pow_nonneg : LE.le 0 (HPow.hPow (μ s).toReal (HDiv.hDiv 1 p.toReal))
        approx_a_pos : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1 (HPow.hPow (μ s) …
        b : E
        b_mem : Membership.mem u b
        hb : LT.lt (Norm.norm (HSub.hSub a b)) (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1  …
        t : Set X
        ht : Membership.mem 𝒜₀ t
        hμt : Ne (μ t) Top.top
        hst : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.indicatorConstLp p ms hμs b)  …
        mt : MeasurableSet t
        ⊢ LT.lt (Norm.norm (HAdd.hAdd (HSub.hSub (MeasureTheory.indicatorConstLp p ms  …
      -/
      refine lt_of_le_of_lt (b := ε / 3 + ε / 3) (norm_add_le_of_le ?_ hst.le) (by linarith [ε_pos])
      /-
        case intro.intro.intro.intro.refine_2.refine_1.intro.intro.intro.intro.intro
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        a : E
        s : Set X
        ms : MeasurableSet s
        hμs✝ : LT.lt (μ s) Top.top
        hμs : Ne (μ s) Top.top
        ε : Real
        ε_pos : LT.lt 0 ε
        μs_pow_nonneg : LE.le 0 (HPow.hPow (μ s).toReal (HDiv.hDiv 1 p.toReal))
        approx_a_pos : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1 (HPow.hPow (μ s) …
        b : E
        b_mem : Membership.mem u b
        hb : LT.lt (Norm.norm (HSub.hSub a b)) (HDiv.hDiv ε (HMul.hMul 3 (HAdd.hAdd 1  …
        t : Set X
        ht : Membership.mem 𝒜₀ t
        hμt : Ne (μ t) Top.top
        hst : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.indicatorConstLp p ms hμs b)  …
        mt : MeasurableSet t
        ⊢ LE.le (Norm.norm (HSub.hSub (MeasureTheory.indicatorConstLp p ms ⋯ a) (Measu …
      -/
      rw [indicatorConstLp_sub, norm_indicatorConstLp p_ne_zero p_ne_top.elim]
      calc
        ‖a - b‖ * (μ s).toReal ^ (1 / p.toReal)
          ≤ (ε / (3 * (1 + (μ s).toReal ^ (1 / p.toReal)))) * (μ s).toReal ^ (1 / p.toReal) :=
              mul_le_mul_of_nonneg_right (le_of_lt hb) μs_pow_nonneg
        _ ≤ ε / 3 := by
            rw [← mul_one (ε / 3), div_mul_eq_div_mul_one_div, mul_assoc, one_div_mul_eq_div]
            exact mul_le_mul_of_nonneg_left
              ((div_le_one (by linarith [μs_pow_nonneg])).2 (by linarith))
              (by linarith [ε_pos])
    · -- Now we have to show that the closure of `D` is closed by sum. Let `f` and `g` be two
      -- functions in `Lᵖ` which are also in the closure of `D`.
      /-
        case intro.intro.intro.intro.refine_2.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        ⊢ ∀ ⦃f g : X → E⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.Memℒp g …
      -/
      rintro f g hf hg - f_mem g_mem
      /-
        case intro.intro.intro.intro.refine_2.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        f g : X → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        f_mem : Membership.mem (closure D) (MeasureTheory.Memℒp.toLp f hf)
        g_mem : Membership.mem (closure D) (MeasureTheory.Memℒp.toLp g hg)
        ⊢ Membership.mem (closure D) (HAdd.hAdd (MeasureTheory.Memℒp.toLp f hf) (Measu …
      -/
      rw [SeminormedAddCommGroup.mem_closure_iff] at *
      /-
        case intro.intro.intro.intro.refine_2.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        f g : X → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        f_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        g_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT.lt (N …
      -/
      intro ε ε_pos
      -- For `ε > 0`, there exists `bf, bg ∈ D` such that `‖f - bf‖ₚ < ε/2` and `‖g - bg‖ₚ < ε/2`.
      /-
        case intro.intro.intro.intro.refine_2.refine_2
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        f g : X → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        f_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        g_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        ε : Real
        ε_pos : LT.lt 0 ε
        ⊢ Exists fun b => And (Membership.mem D b) (LT.lt (Norm.norm (HSub.hSub (HAdd. …
      -/
      rcases f_mem (ε / 2) (by linarith [ε_pos]) with ⟨bf, ⟨nf, df, sf, bf_eq⟩, hbf⟩
      /-
        case intro.intro.intro.intro.refine_2.refine_2.intro.intro.intro.intro.intro
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        f g : X → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        f_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        g_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        ε : Real
        ε_pos : LT.lt 0 ε
        bf : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hbf : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.Memℒp.toLp f hf) bf)) (HDiv.h …
        nf : Nat
        df : Fin nf → ↑u
        sf : Fin nf → ↑𝒜₀
        bf_eq : Eq bf (key nf df sf)
        ⊢ Exists fun b => And (Membership.mem D b) (LT.lt (Norm.norm (HSub.hSub (HAdd. …
      -/
      rcases g_mem (ε / 2) (by linarith [ε_pos]) with ⟨bg, ⟨ng, dg, sg, bg_eq⟩, hbg⟩
      -- It is obvious that `D` is closed by sum, it suffices to concatenate the family of
      -- elements of `u` and the family of elements of `𝒜₀`.
      let d := fun i : Fin (nf + ng) ↦ if h : i < nf
        then df (Fin.castLT i h)
        else dg (Fin.subNat nf (Fin.cast (Nat.add_comm ..) i) (le_of_not_gt h))
      let s := fun i : Fin (nf + ng) ↦ if h : i < nf
        then sf (Fin.castLT i h)
        else sg (Fin.subNat nf (Fin.cast (Nat.add_comm ..) i) (le_of_not_gt h))
      -- So we can use `bf + bg`.
      /-
        case intro.intro.intro.intro.refine_2.refine_2.intro.intro.intro.intro.intro.i …
        X : Type u_1
        E : Type u_2
        m : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        μ : MeasureTheory.Measure X
        p : ENNReal
        one_le_p : Fact (LE.le 1 p)
        p_ne_top : Fact (Ne p Top.top)
        𝒜✝ : Set (Set X)
        inst✝¹ : MeasureTheory.IsSeparable μ
        inst✝ : TopologicalSpace.SeparableSpace E
        𝒜 : Set (Set X)
        count_𝒜 : 𝒜.Countable
        h𝒜 : μ.MeasureDense 𝒜
        𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
        h𝒜₀ : μ.MeasureDense 𝒜₀
        count_𝒜₀ : 𝒜₀.Countable
        p_ne_zero : Ne p 0
        u : Set E
        countable_u : u.Countable
        dense_u : Dense u
        key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
        D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
        f g : X → E
        hf : MeasureTheory.Memℒp f p μ
        hg : MeasureTheory.Memℒp g p μ
        f_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        g_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
        ε : Real
        ε_pos : LT.lt 0 ε
        bf : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hbf : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.Memℒp.toLp f hf) bf)) (HDiv.h …
        nf : Nat
        df : Fin nf → ↑u
        sf : Fin nf → ↑𝒜₀
        bf_eq : Eq bf (key nf df sf)
        bg : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hbg : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.Memℒp.toLp g hg) bg)) (HDiv.h …
        ng : Nat
        dg : Fin ng → ↑u
        sg : Fin ng → ↑𝒜₀
        bg_eq : Eq bg (key ng dg sg)
        d : Fin (HAdd.hAdd nf ng) → ↑u := fun i => dite (LT.lt (↑i) nf) (fun h => df ( …
        s : Fin (HAdd.hAdd nf ng) → ↑𝒜₀ := fun i => dite (LT.lt (↑i) nf) (fun h => sf  …
        ⊢ Exists fun b => And (Membership.mem D b) (LT.lt (Norm.norm (HSub.hSub (HAdd. …
      -/
      refine ⟨bf + bg, ⟨nf + ng, d, s, ?_⟩, ?_⟩
        /-
          case intro.intro.intro.intro.refine_2.refine_2.intro.intro.intro.intro.intro.i …
          X : Type u_1
          E : Type u_2
          m : MeasurableSpace X
          inst✝² : NormedAddCommGroup E
          μ : MeasureTheory.Measure X
          p : ENNReal
          one_le_p : Fact (LE.le 1 p)
          p_ne_top : Fact (Ne p Top.top)
          𝒜✝ : Set (Set X)
          inst✝¹ : MeasureTheory.IsSeparable μ
          inst✝ : TopologicalSpace.SeparableSpace E
          𝒜 : Set (Set X)
          count_𝒜 : 𝒜.Countable
          h𝒜 : μ.MeasureDense 𝒜
          𝒜₀ : Set (Set X) := setOf fun s => And (Membership.mem 𝒜 s) (Ne (μ s) Top.top)
          h𝒜₀ : μ.MeasureDense 𝒜₀
          count_𝒜₀ : 𝒜₀.Countable
          p_ne_zero : Ne p 0
          u : Set E
          countable_u : u.Countable
          dense_u : Dense u
          key : (n : Nat) → (Fin n → ↑u) → (Fin n → ↑𝒜₀) → Subtype fun x => Membership.m …
          D : Set (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) := setOf  …
          f g : X → E
          hf : MeasureTheory.Memℒp f p μ
          hg : MeasureTheory.Memℒp g p μ
          f_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
          g_mem : ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem D b) (LT …
          ε : Real
          ε_pos : LT.lt 0 ε
          bf : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
          hbf : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.Memℒp.toLp f hf) bf)) (HDiv.h …
          nf : Nat
          df : Fin nf → ↑u
          sf : Fin nf → ↑𝒜₀
          bf_eq : Eq bf (key nf df sf)
          bg : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
          hbg : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.Memℒp.toLp g hg) bg)) (HDiv.h …
          ng : Nat
          dg : Fin ng → ↑u
          sg : Fin ng → ↑𝒜₀
          bg_eq : Eq bg (key ng dg sg)
          d : Fin (HAdd.hAdd nf ng) → ↑u := fun i => dite (LT.lt (↑i) nf) (fun h => df ( …
          s : Fin (HAdd.hAdd nf ng) → ↑𝒜₀ := fun i => dite (LT.lt (↑i) nf) (fun h => sf  …
          ⊢ Eq (HAdd.hAdd bf bg) (key (HAdd.hAdd nf ng) d s)
        -/
      · simp [key, d, s, Fin.sum_univ_add, bf_eq, bg_eq]
        /-
          🎉 no goals
        -/
      · -- We have
        -- `‖f + g - (bf + bg)‖ₚ`
        --   `≤ ‖f - bf‖ₚ + ‖g - bg‖ₚ`
        --   `< ε/2 + ε/2 = ε`.
        calc
          ‖Memℒp.toLp f hf + Memℒp.toLp g hg - (bf + bg)‖
            = ‖(Memℒp.toLp f hf) - bf + ((Memℒp.toLp g hg) - bg)‖ := by congr; abel
          _ ≤ ‖(Memℒp.toLp f hf) - bf‖ + ‖(Memℒp.toLp g hg) - bg‖ := norm_add_le ..
          _ < ε := by linarith [hbf, hbg]


