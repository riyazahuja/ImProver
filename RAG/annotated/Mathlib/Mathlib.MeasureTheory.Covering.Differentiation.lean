/-- The limit along a Vitali family of `ρ a / μ a` where it makes sense, and garbage otherwise.
Do *not* use this definition: it is only a temporary device to show that this ratio tends almost
everywhere to the Radon-Nikodym derivative. -/
noncomputable def limRatio (ρ : Measure α) (x : α) : ℝ≥0∞ :=
  limUnder (v.filterAt x) fun a => ρ a / μ a


/-- For almost every point `x`, sufficiently small sets in a Vitali family around `x` have positive
measure. (This is a nontrivial result, following from the covering property of Vitali families). -/
theorem ae_eventually_measure_pos [SecondCountableTopology α] :
    ∀ᵐ x ∂μ, ∀ᶠ a in v.filterAt x, 0 < μ a := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝ : SecondCountableTopology α
    ⊢ Filter.Eventually (fun x => Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.fi …
  -/
  set s := {x | ¬∀ᶠ a in v.filterAt x, 0 < μ a} with hs
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝ : SecondCountableTopology α
    s : Set α := setOf fun x => Not (Filter.Eventually (fun a => LT.lt 0 (μ a)) (v …
    hs : Eq s (setOf fun x => Not (Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.f …
    ⊢ Filter.Eventually (fun x => Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.fi …
  -/
  simp (config := { zeta := false }) only [not_lt, not_eventually, nonpos_iff_eq_zero] at hs
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝ : SecondCountableTopology α
    s : Set α := setOf fun x => Not (Filter.Eventually (fun a => LT.lt 0 (μ a)) (v …
    hs : Eq s (setOf fun x => Filter.Frequently (fun x => Eq (μ x) 0) (v.filterAt  …
    ⊢ Filter.Eventually (fun x => Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.fi …
  -/
  change μ s = 0
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝ : SecondCountableTopology α
    s : Set α := setOf fun x => Not (Filter.Eventually (fun a => LT.lt 0 (μ a)) (v …
    hs : Eq s (setOf fun x => Filter.Frequently (fun x => Eq (μ x) 0) (v.filterAt  …
    ⊢ Eq (μ s) 0
  -/
  let f : α → Set (Set α) := fun _ => {a | μ a = 0}
  have h : v.FineSubfamilyOn f s := by
    intro x hx ε εpos
    rw [hs] at hx
    simp only [frequently_filterAt_iff, exists_prop, gt_iff_lt, mem_setOf_eq] at hx
    rcases hx ε εpos with ⟨a, a_sets, ax, μa⟩
    exact ⟨a, ⟨a_sets, μa⟩, ax⟩
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝ : SecondCountableTopology α
    s : Set α := setOf fun x => Not (Filter.Eventually (fun a => LT.lt 0 (μ a)) (v …
    hs : Eq s (setOf fun x => Filter.Frequently (fun x => Eq (μ x) 0) (v.filterAt  …
    f : α → Set (Set α) := fun x => setOf fun a => Eq (μ a) 0
    h : v.FineSubfamilyOn f s
    ⊢ Eq (μ s) 0
  -/
  refine le_antisymm ?_ bot_le
  calc
    μ s ≤ ∑' x : h.index, μ (h.covering x) := h.measure_le_tsum
    _ = ∑' x : h.index, 0 := by congr; ext1 x; exact h.covering_mem x.2
    _ = 0 := by simp only [tsum_zero, add_zero]


/-- For every point `x`, sufficiently small sets in a Vitali family around `x` have finite measure.
(This is a trivial result, following from the fact that the measure is locally finite). -/
theorem eventually_measure_lt_top [IsLocallyFiniteMeasure μ] (x : α) :
    ∀ᶠ a in v.filterAt x, μ a < ∞ :=
  (μ.finiteAt_nhds x).eventually.filter_mono inf_le_left


/-- If two measures `ρ` and `ν` have, at every point of a set `s`, arbitrarily small sets in a
Vitali family satisfying `ρ a ≤ ν a`, then `ρ s ≤ ν s` if `ρ ≪ μ`. -/
theorem measure_le_of_frequently_le [SecondCountableTopology α] [BorelSpace α] {ρ : Measure α}
    (ν : Measure α) [IsLocallyFiniteMeasure ν] (hρ : ρ ≪ μ) (s : Set α)
    (hs : ∀ x ∈ s, ∃ᶠ a in v.filterAt x, ρ a ≤ ν a) : ρ s ≤ ν s := by
  -- this follows from a covering argument using the sets satisfying `ρ a ≤ ν a`.
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    ρ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    ⊢ LE.le (ρ s) (ν s)
  -/
  apply ENNReal.le_of_forall_pos_le_add fun ε εpos _ => ?_
  obtain ⟨U, sU, U_open, νU⟩ : ∃ (U : Set α), s ⊆ U ∧ IsOpen U ∧ ν U ≤ ν s + ε :=
    exists_isOpen_le_add s ν (ENNReal.coe_pos.2 εpos).ne'
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    ρ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    ε : NNReal
    εpos : LT.lt 0 ε
    x✝ : LT.lt (ν s) Top.top
    U : Set α
    sU : HasSubset.Subset s U
    U_open : IsOpen U
    νU : LE.le (ν U) (HAdd.hAdd (ν s) ↑ε)
    ⊢ LE.le (ρ s) (HAdd.hAdd (ν s) ↑ε)
  -/
  let f : α → Set (Set α) := fun _ => {a | ρ a ≤ ν a ∧ a ⊆ U}
  have h : v.FineSubfamilyOn f s := by
    apply v.fineSubfamilyOn_of_frequently f s fun x hx => ?_
    have :=
      (hs x hx).and_eventually
        ((v.eventually_filterAt_mem_setsAt x).and
          (v.eventually_filterAt_subset_of_nhds (U_open.mem_nhds (sU hx))))
    apply Frequently.mono this
    rintro a ⟨ρa, _, aU⟩
    exact ⟨ρa, aU⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    ρ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    ε : NNReal
    εpos : LT.lt 0 ε
    x✝ : LT.lt (ν s) Top.top
    U : Set α
    sU : HasSubset.Subset s U
    U_open : IsOpen U
    νU : LE.le (ν U) (HAdd.hAdd (ν s) ↑ε)
    f : α → Set (Set α) := fun x => setOf fun a => And (LE.le (ρ a) (ν a)) (HasSub …
    h : v.FineSubfamilyOn f s
    ⊢ LE.le (ρ s) (HAdd.hAdd (ν s) ↑ε)
  -/
  haveI : Encodable h.index := h.index_countable.toEncodable
  calc
    ρ s ≤ ∑' x : h.index, ρ (h.covering x) := h.measure_le_tsum_of_absolutelyContinuous hρ
    _ ≤ ∑' x : h.index, ν (h.covering x) := ENNReal.tsum_le_tsum fun x => (h.covering_mem x.2).1
    _ = ν (⋃ x : h.index, h.covering x) := by
      rw [measure_iUnion h.covering_disjoint_subtype fun i => h.measurableSet_u i.2]
    _ ≤ ν U := (measure_mono (iUnion_subset fun i => (h.covering_mem i.2).2))
    _ ≤ ν s + ε := νU


theorem eventually_filterAt_integrableOn (x : α) {f : α → E} (hf : LocallyIntegrable f μ) :
    ∀ᶠ a in v.filterAt x, IntegrableOn f a μ := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    x : α
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ Filter.Eventually (fun a => MeasureTheory.IntegrableOn f a μ) (v.filterAt x)
  -/
  rcases hf x with ⟨w, w_nhds, hw⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    x : α
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    w : Set α
    w_nhds : Membership.mem (nhds x) w
    hw : MeasureTheory.IntegrableOn f w μ
    ⊢ Filter.Eventually (fun a => MeasureTheory.IntegrableOn f a μ) (v.filterAt x)
  -/
  filter_upwards [v.eventually_filterAt_subset_of_nhds w_nhds] with a ha
  /-
    case h
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    x : α
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    w : Set α
    w_nhds : Membership.mem (nhds x) w
    hw : MeasureTheory.IntegrableOn f w μ
    a : Set α
    ha : HasSubset.Subset a w
    ⊢ MeasureTheory.IntegrableOn f a μ
  -/
  exact hw.mono_set ha
  /-
    🎉 no goals
  -/


/-- If a measure `ρ` is singular with respect to `μ`, then for `μ` almost every `x`, the ratio
`ρ a / μ a` tends to zero when `a` shrinks to `x` along the Vitali family. This makes sense
as `μ a` is eventually positive by `ae_eventually_measure_pos`. -/
theorem ae_eventually_measure_zero_of_singular (hρ : ρ ⟂ₘ μ) :
    ∀ᵐ x ∂μ, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 0) := by
  have A : ∀ ε > (0 : ℝ≥0), ∀ᵐ x ∂μ, ∀ᶠ a in v.filterAt x, ρ a < ε * μ a := by
    intro ε εpos
    set s := {x | ¬∀ᶠ a in v.filterAt x, ρ a < ε * μ a} with hs
    change μ s = 0
    obtain ⟨o, _, ρo, μo⟩ : ∃ o : Set α, MeasurableSet o ∧ ρ o = 0 ∧ μ oᶜ = 0 := hρ
    apply le_antisymm _ bot_le
    calc
      μ s ≤ μ (s ∩ o ∪ oᶜ) := by
        conv_lhs => rw [← inter_union_compl s o]
        gcongr
        apply inter_subset_right
      _ ≤ μ (s ∩ o) + μ oᶜ := measure_union_le _ _
      _ = μ (s ∩ o) := by rw [μo, add_zero]
      _ = (ε : ℝ≥0∞)⁻¹ * (ε • μ) (s ∩ o) := by
        simp only [coe_nnreal_smul_apply, ← mul_assoc, mul_comm _ (ε : ℝ≥0∞)]
        rw [ENNReal.mul_inv_cancel (ENNReal.coe_pos.2 εpos).ne' ENNReal.coe_ne_top, one_mul]
      _ ≤ (ε : ℝ≥0∞)⁻¹ * ρ (s ∩ o) := by
        gcongr
        refine v.measure_le_of_frequently_le ρ smul_absolutelyContinuous _ ?_
        intro x hx
        rw [hs] at hx
        simp only [mem_inter_iff, not_lt, not_eventually, mem_setOf_eq] at hx
        exact hx.1
      _ ≤ (ε : ℝ≥0∞)⁻¹ * ρ o := by gcongr; apply inter_subset_right
      _ = 0 := by rw [ρo, mul_zero]
  obtain ⟨u, _, u_pos, u_lim⟩ :
    ∃ u : ℕ → ℝ≥0, StrictAnti u ∧ (∀ n : ℕ, 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ≥0)
  have B : ∀ᵐ x ∂μ, ∀ n, ∀ᶠ a in v.filterAt x, ρ a < u n * μ a :=
    ae_all_iff.2 fun n => A (u n) (u_pos n)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  filter_upwards [B, v.ae_eventually_measure_pos]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    ⊢ ∀ (a : α), (∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul  …
  -/
  intro x hx h'x
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds 0)
  -/
  refine tendsto_order.2 ⟨fun z hz => (ENNReal.not_lt_zero hz).elim, fun z hz => ?_⟩
  obtain ⟨w, w_pos, w_lt⟩ : ∃ w : ℝ≥0, (0 : ℝ≥0∞) < w ∧ (w : ℝ≥0∞) < z :=
    ENNReal.lt_iff_exists_nnreal_btwn.1 hz
  /-
    case h.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    z : ENNReal
    hz : GT.gt z 0
    w : NNReal
    w_pos : LT.lt 0 ↑w
    w_lt : LT.lt (↑w) z
    ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) z) (v.filterAt x)
  -/
  obtain ⟨n, hn⟩ : ∃ n, u n < w := ((tendsto_order.1 u_lim).2 w (ENNReal.coe_pos.1 w_pos)).exists
  /-
    case h.intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    z : ENNReal
    hz : GT.gt z 0
    w : NNReal
    w_pos : LT.lt 0 ↑w
    w_lt : LT.lt (↑w) z
    n : Nat
    hn : LT.lt (u n) w
    ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) z) (v.filterAt x)
  -/
  filter_upwards [hx n, h'x, v.eventually_measure_lt_top x]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    z : ENNReal
    hz : GT.gt z 0
    w : NNReal
    w_pos : LT.lt 0 ↑w
    w_lt : LT.lt (↑w) z
    n : Nat
    hn : LT.lt (u n) w
    ⊢ ∀ (a : Set α), LT.lt (ρ a) (HMul.hMul (↑(u n)) (μ a)) → LT.lt 0 (μ a) → LT.l …
  -/
  intro a ha μa_pos μa_lt_top
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    z : ENNReal
    hz : GT.gt z 0
    w : NNReal
    w_pos : LT.lt 0 ↑w
    w_lt : LT.lt (↑w) z
    n : Nat
    hn : LT.lt (u n) w
    a : Set α
    ha : LT.lt (ρ a) (HMul.hMul (↑(u n)) (μ a))
    μa_pos : LT.lt 0 (μ a)
    μa_lt_top : LT.lt (μ a) Top.top
    ⊢ LT.lt (HDiv.hDiv (ρ a) (μ a)) z
  -/
  rw [ENNReal.div_lt_iff (Or.inl μa_pos.ne') (Or.inl μa_lt_top.ne)]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.MutuallySingular μ
    A : ∀ (ε : NNReal), GT.gt ε 0 → Filter.Eventually (fun x => Filter.Eventually  …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    B : Filter.Eventually (fun x => ∀ (n : Nat), Filter.Eventually (fun a => LT.lt …
    x : α
    hx : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (ρ a) (HMul.hMul (↑(u n))  …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    z : ENNReal
    hz : GT.gt z 0
    w : NNReal
    w_pos : LT.lt 0 ↑w
    w_lt : LT.lt (↑w) z
    n : Nat
    hn : LT.lt (u n) w
    a : Set α
    ha : LT.lt (ρ a) (HMul.hMul (↑(u n)) (μ a))
    μa_pos : LT.lt 0 (μ a)
    μa_lt_top : LT.lt (μ a) Top.top
    ⊢ LT.lt (ρ a) (HMul.hMul z (μ a))
  -/
  exact ha.trans_le (mul_le_mul_right' ((ENNReal.coe_le_coe.2 hn.le).trans w_lt.le) _)
  /-
    🎉 no goals
  -/


/-- A set of points `s` satisfying both `ρ a ≤ c * μ a` and `ρ a ≥ d * μ a` at arbitrarily small
sets in a Vitali family has measure `0` if `c < d`. Indeed, the first inequality should imply
that `ρ s ≤ c * μ s`, and the second one that `ρ s ≥ d * μ s`, a contradiction if `0 < μ s`. -/
theorem null_of_frequently_le_of_frequently_ge {c d : ℝ≥0} (hcd : c < d) (s : Set α)
    (hc : ∀ x ∈ s, ∃ᶠ a in v.filterAt x, ρ a ≤ c * μ a)
    (hd : ∀ x ∈ s, ∃ᶠ a in v.filterAt x, (d : ℝ≥0∞) * μ a ≤ ρ a) : μ s = 0 := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    c d : NNReal
    hcd : LT.lt c d
    s : Set α
    hc : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    hd : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (HMul.h …
    ⊢ Eq (μ s) 0
  -/
  apply measure_null_of_locally_null s fun x _ => ?_
  obtain ⟨o, xo, o_open, μo⟩ : ∃ o : Set α, x ∈ o ∧ IsOpen o ∧ μ o < ∞ :=
    Measure.exists_isOpen_measure_lt_top μ x
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    c d : NNReal
    hcd : LT.lt c d
    s : Set α
    hc : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    hd : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (HMul.h …
    x : α
    x✝ : Membership.mem s x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (Eq (μ u) 0)
  -/
  refine ⟨s ∩ o, inter_mem_nhdsWithin _ (o_open.mem_nhds xo), ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    c d : NNReal
    hcd : LT.lt c d
    s : Set α
    hc : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    hd : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (HMul.h …
    x : α
    x✝ : Membership.mem s x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ Eq (μ (Inter.inter s o)) 0
  -/
  let s' := s ∩ o
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    c d : NNReal
    hcd : LT.lt c d
    s : Set α
    hc : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    hd : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (HMul.h …
    x : α
    x✝ : Membership.mem s x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s' : Set α := Inter.inter s o
    ⊢ Eq (μ (Inter.inter s o)) 0
  -/
  by_contra h
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    c d : NNReal
    hcd : LT.lt c d
    s : Set α
    hc : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (ρ a) ( …
    hd : ∀ (x : α), Membership.mem s x → Filter.Frequently (fun a => LE.le (HMul.h …
    x : α
    x✝ : Membership.mem s x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s' : Set α := Inter.inter s o
    h : Not (Eq (μ (Inter.inter s o)) 0)
    ⊢ False
  -/
  apply lt_irrefl (ρ s')
  calc
    ρ s' ≤ c * μ s' := v.measure_le_of_frequently_le (c • μ) hρ s' fun x hx => hc x hx.1
    _ < d * μ s' := by
      apply (ENNReal.mul_lt_mul_right h _).2 (ENNReal.coe_lt_coe.2 hcd)
      exact (lt_of_le_of_lt (measure_mono inter_subset_right) μo).ne
    _ ≤ ρ s' := v.measure_le_of_frequently_le ρ smul_absolutelyContinuous s' fun x hx ↦ hd x hx.1


/-- If `ρ` is absolutely continuous with respect to `μ`, then for almost every `x`,
the ratio `ρ a / μ a` converges as `a` shrinks to `x` along a Vitali family for `μ`. -/
theorem ae_tendsto_div : ∀ᵐ x ∂μ, ∃ c, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 c) := by
  obtain ⟨w, w_count, w_dense, _, w_top⟩ :
    ∃ w : Set ℝ≥0∞, w.Countable ∧ Dense w ∧ 0 ∉ w ∧ ∞ ∉ w :=
    ENNReal.exists_countable_dense_no_zero_top
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    w : Set ENNReal
    w_count : w.Countable
    w_dense : Dense w
    left✝ : Not (Membership.mem w 0)
    w_top : Not (Membership.mem w Top.top)
    ⊢ Filter.Eventually (fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hD …
  -/
  have I : ∀ x ∈ w, x ≠ ∞ := fun x xs hx => w_top (hx ▸ xs)
  have A : ∀ c ∈ w, ∀ d ∈ w, c < d → ∀ᵐ x ∂μ,
      ¬((∃ᶠ a in v.filterAt x, ρ a / μ a < c) ∧ ∃ᶠ a in v.filterAt x, d < ρ a / μ a) := by
    intro c hc d hd hcd
    lift c to ℝ≥0 using I c hc
    lift d to ℝ≥0 using I d hd
    apply v.null_of_frequently_le_of_frequently_ge hρ (ENNReal.coe_lt_coe.1 hcd)
    · simp only [and_imp, exists_prop, not_frequently, not_and, not_lt, not_le, not_eventually,
        mem_setOf_eq, mem_compl_iff, not_forall]
      intro x h1x _
      apply h1x.mono fun a ha => ?_
      refine (ENNReal.div_le_iff_le_mul ?_ (Or.inr (bot_le.trans_lt ha).ne')).1 ha.le
      simp only [ENNReal.coe_ne_top, Ne, or_true, not_false_iff]
    · simp only [and_imp, exists_prop, not_frequently, not_and, not_lt, not_le, not_eventually,
        mem_setOf_eq, mem_compl_iff, not_forall]
      intro x _ h2x
      apply h2x.mono fun a ha => ?_
      exact ENNReal.mul_le_of_le_div ha.le
  have B : ∀ᵐ x ∂μ, ∀ c ∈ w, ∀ d ∈ w, c < d →
      ¬((∃ᶠ a in v.filterAt x, ρ a / μ a < c) ∧ ∃ᶠ a in v.filterAt x, d < ρ a / μ a) := by
    #adaptation_note /-- 2024-04-23
    The next two lines were previously just `simpa only [ae_ball_iff w_count, ae_all_iff]` -/
    rw [ae_ball_iff w_count]; intro x hx; rw [ae_ball_iff w_count]; revert x
    simpa only [ae_all_iff]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    w : Set ENNReal
    w_count : w.Countable
    w_dense : Dense w
    left✝ : Not (Membership.mem w 0)
    w_top : Not (Membership.mem w Top.top)
    I : ∀ (x : ENNReal), Membership.mem w x → Ne x Top.top
    A : ∀ (c : ENNReal), Membership.mem w c → ∀ (d : ENNReal), Membership.mem w d  …
    B : Filter.Eventually (fun x => ∀ (c : ENNReal), Membership.mem w c → ∀ (d : E …
    ⊢ Filter.Eventually (fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hD …
  -/
  filter_upwards [B]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    w : Set ENNReal
    w_count : w.Countable
    w_dense : Dense w
    left✝ : Not (Membership.mem w 0)
    w_top : Not (Membership.mem w Top.top)
    I : ∀ (x : ENNReal), Membership.mem w x → Ne x Top.top
    A : ∀ (c : ENNReal), Membership.mem w c → ∀ (d : ENNReal), Membership.mem w d  …
    B : Filter.Eventually (fun x => ∀ (c : ENNReal), Membership.mem w c → ∀ (d : E …
    ⊢ ∀ (a : α), (∀ (c : ENNReal), Membership.mem w c → ∀ (d : ENNReal), Membershi …
  -/
  intro x hx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    w : Set ENNReal
    w_count : w.Countable
    w_dense : Dense w
    left✝ : Not (Membership.mem w 0)
    w_top : Not (Membership.mem w Top.top)
    I : ∀ (x : ENNReal), Membership.mem w x → Ne x Top.top
    A : ∀ (c : ENNReal), Membership.mem w c → ∀ (d : ENNReal), Membership.mem w d  …
    B : Filter.Eventually (fun x => ∀ (c : ENNReal), Membership.mem w c → ∀ (d : E …
    x : α
    hx : ∀ (c : ENNReal), Membership.mem w c → ∀ (d : ENNReal), Membership.mem w d …
    ⊢ Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt  …
  -/
  exact tendsto_of_no_upcrossings w_dense hx
  /-
    🎉 no goals
  -/


theorem ae_tendsto_limRatio :
    ∀ᵐ x ∂μ, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatio ρ x)) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  filter_upwards [v.ae_tendsto_div hρ]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ ∀ (a : α), (Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  intro x hx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    hx : Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filter …
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (v.limR …
  -/
  exact tendsto_nhds_limUnder hx
  /-
    🎉 no goals
  -/


/-- Given two thresholds `p < q`, the sets `{x | v.limRatio ρ x < p}`
and `{x | q < v.limRatio ρ x}` are obviously disjoint. The key to proving that `v.limRatio ρ` is
almost everywhere measurable is to show that these sets have measurable supersets which are also
disjoint, up to zero measure. This is the content of this lemma. -/
theorem exists_measurable_supersets_limRatio {p q : ℝ≥0} (hpq : p < q) :
    ∃ a b, MeasurableSet a ∧ MeasurableSet b ∧
      {x | v.limRatio ρ x < p} ⊆ a ∧ {x | (q : ℝ≥0∞) < v.limRatio ρ x} ⊆ b ∧ μ (a ∩ b) = 0 := by
  /- Here is a rough sketch, assuming that the measure is finite and the limit is well defined
    everywhere. Let `u := {x | v.limRatio ρ x < p}` and `w := {x | q < v.limRatio ρ x}`. They
    have measurable supersets `u'` and `w'` of the same measure. We will show that these satisfy
    the conclusion of the theorem, i.e., `μ (u' ∩ w') = 0`. For this, note that
    `ρ (u' ∩ w') = ρ (u ∩ w')` (as `w'` is measurable, see `measure_toMeasurable_add_inter_left`).
    The latter set is included in the set where the limit of the ratios is `< p`, and therefore
    its measure is `≤ p * μ (u ∩ w')`. Using the same trick in the other direction gives that this
    is `p * μ (u' ∩ w')`. We have shown that `ρ (u' ∩ w') ≤ p * μ (u' ∩ w')`. Arguing in the same
    way but using the `w` part gives `q * μ (u' ∩ w') ≤ ρ (u' ∩ w')`. If `μ (u' ∩ w')` were nonzero,
    this would be a contradiction as `p < q`.

    For the rigorous proof, we need to work on a part of the space where the measure is finite
    (provided by `spanningSets (ρ + μ)`) and to restrict to the set where the limit is well defined
    (called `s` below, of full measure). Otherwise, the argument goes through.
    -/
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    ⊢ Exists fun a => Exists fun b => And (MeasurableSet a) (And (MeasurableSet b) …
  -/
  let s := {x | ∃ c, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 c)}
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    ⊢ Exists fun a => Exists fun b => And (MeasurableSet a) (And (MeasurableSet b) …
  -/
  let o : ℕ → Set α := spanningSets (ρ + μ)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
    ⊢ Exists fun a => Exists fun b => And (MeasurableSet a) (And (MeasurableSet b) …
  -/
  let u n := s ∩ {x | v.limRatio ρ x < p} ∩ o n
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
    u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    ⊢ Exists fun a => Exists fun b => And (MeasurableSet a) (And (MeasurableSet b) …
  -/
  let w n := s ∩ {x | (q : ℝ≥0∞) < v.limRatio ρ x} ∩ o n
  -- the supersets are obtained by restricting to the set `s` where the limit is well defined, to
  -- a finite measure part `o n`, taking a measurable superset here, and then taking the union over
  -- `n`.
  refine
    ⟨toMeasurable μ sᶜ ∪ ⋃ n, toMeasurable (ρ + μ) (u n),
      toMeasurable μ sᶜ ∪ ⋃ n, toMeasurable (ρ + μ) (w n), ?_, ?_, ?_, ?_, ?_⟩
  -- check that these sets are measurable supersets as required
  · exact
      (measurableSet_toMeasurable _ _).union
        (MeasurableSet.iUnion fun n => measurableSet_toMeasurable _ _)
  · exact
      (measurableSet_toMeasurable _ _).union
        (MeasurableSet.iUnion fun n => measurableSet_toMeasurable _ _)
    /-
      case refine_3
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      p q : NNReal
      hpq : LT.lt p q
      s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
      o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
      u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      ⊢ HasSubset.Subset (setOf fun x => LT.lt (v.limRatio ρ x) ↑p) (Union.union (Me …
    -/
  · intro x hx
    /-
      case refine_3
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      p q : NNReal
      hpq : LT.lt p q
      s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
      o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
      u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      x : α
      hx : Membership.mem (setOf fun x => LT.lt (v.limRatio ρ x) ↑p) x
      ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
    -/
    by_cases h : x ∈ s
      /-
        case pos
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (v.limRatio ρ x) ↑p) x
        h : Membership.mem s x
        ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
      -/
    · refine Or.inr (mem_iUnion.2 ⟨spanningSetsIndex (ρ + μ) x, ?_⟩)
      /-
        case pos
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (v.limRatio ρ x) ↑p) x
        h : Membership.mem s x
        ⊢ Membership.mem (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (u (MeasureTheory …
      -/
      exact subset_toMeasurable _ _ ⟨⟨h, hx⟩, mem_spanningSetsIndex _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (v.limRatio ρ x) ↑p) x
        h : Not (Membership.mem s x)
        ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
      -/
    · exact Or.inl (subset_toMeasurable μ sᶜ h)
      /-
        🎉 no goals
      -/
    /-
      case refine_4
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      p q : NNReal
      hpq : LT.lt p q
      s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
      o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
      u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      ⊢ HasSubset.Subset (setOf fun x => LT.lt (↑q) (v.limRatio ρ x)) (Union.union ( …
    -/
  · intro x hx
    /-
      case refine_4
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      p q : NNReal
      hpq : LT.lt p q
      s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
      o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
      u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
      x : α
      hx : Membership.mem (setOf fun x => LT.lt (↑q) (v.limRatio ρ x)) x
      ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
    -/
    by_cases h : x ∈ s
      /-
        case pos
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (↑q) (v.limRatio ρ x)) x
        h : Membership.mem s x
        ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
      -/
    · refine Or.inr (mem_iUnion.2 ⟨spanningSetsIndex (ρ + μ) x, ?_⟩)
      /-
        case pos
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (↑q) (v.limRatio ρ x)) x
        h : Membership.mem s x
        ⊢ Membership.mem (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (w (MeasureTheory …
      -/
      exact subset_toMeasurable _ _ ⟨⟨h, hx⟩, mem_spanningSetsIndex _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝⁴ : PseudoMetricSpace α
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        v : VitaliFamily μ
        inst✝³ : SecondCountableTopology α
        inst✝² : BorelSpace α
        inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
        ρ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
        hρ : ρ.AbsolutelyContinuous μ
        p q : NNReal
        hpq : LT.lt p q
        s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
        o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
        u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
        x : α
        hx : Membership.mem (setOf fun x => LT.lt (↑q) (v.limRatio ρ x)) x
        h : Not (Membership.mem s x)
        ⊢ Membership.mem (Union.union (MeasureTheory.toMeasurable μ (HasCompl.compl s) …
      -/
    · exact Or.inl (subset_toMeasurable μ sᶜ h)
      /-
        🎉 no goals
      -/
  -- it remains to check the nontrivial part that these sets have zero measure intersection.
  -- it suffices to do it for fixed `m` and `n`, as one is taking countable unions.
  suffices H : ∀ m n : ℕ, μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) = 0 by
    have A :
      (toMeasurable μ sᶜ ∪ ⋃ n, toMeasurable (ρ + μ) (u n)) ∩
          (toMeasurable μ sᶜ ∪ ⋃ n, toMeasurable (ρ + μ) (w n)) ⊆
        toMeasurable μ sᶜ ∪
          ⋃ (m) (n), toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n) := by
      simp only [inter_union_distrib_left, union_inter_distrib_right, true_and,
        subset_union_left, union_subset_iff, inter_self]
      refine ⟨?_, ?_, ?_⟩
      · exact inter_subset_right.trans subset_union_left
      · exact inter_subset_left.trans subset_union_left
      · simp_rw [iUnion_inter, inter_iUnion]; exact subset_union_right
    refine le_antisymm ((measure_mono A).trans ?_) bot_le
    calc
      μ (toMeasurable μ sᶜ ∪
        ⋃ (m) (n), toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) ≤
          μ (toMeasurable μ sᶜ) +
            μ (⋃ (m) (n), toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) :=
        measure_union_le _ _
      _ = μ (⋃ (m) (n), toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) := by
        have : μ sᶜ = 0 := v.ae_tendsto_div hρ; rw [measure_toMeasurable, this, zero_add]
      _ ≤ ∑' (m) (n), μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) :=
        ((measure_iUnion_le _).trans (ENNReal.tsum_le_tsum fun m => measure_iUnion_le _))
      _ = 0 := by simp only [H, tsum_zero]
  -- now starts the nontrivial part of the argument. We fix `m` and `n`, and show that the
  -- measurable supersets of `u m` and `w n` have zero measure intersection by using the lemmas
  -- `measure_toMeasurable_add_inter_left` (to reduce to `u m` or `w n` instead of the measurable
  -- superset) and `measure_le_of_frequently_le` to compare their measures for `ρ` and `μ`.
  /-
    case refine_5
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
    u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    ⊢ ∀ (m n : Nat), Eq (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ …
  -/
  intro m n
  have I : (ρ + μ) (u m) ≠ ∞ := by
    apply (lt_of_le_of_lt (measure_mono _) (measure_spanningSets_lt_top (ρ + μ) m)).ne
    exact inter_subset_right
  have J : (ρ + μ) (w n) ≠ ∞ := by
    apply (lt_of_le_of_lt (measure_mono _) (measure_spanningSets_lt_top (ρ + μ) n)).ne
    exact inter_subset_right
  have A :
    ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) ≤
      p * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) :=
    calc
      ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) =
          ρ (u m ∩ toMeasurable (ρ + μ) (w n)) :=
        measure_toMeasurable_add_inter_left (measurableSet_toMeasurable _ _) I
      _ ≤ (p • μ) (u m ∩ toMeasurable (ρ + μ) (w n)) := by
        refine v.measure_le_of_frequently_le (p • μ) hρ _ fun x hx => ?_
        have L : Tendsto (fun a : Set α => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatio ρ x)) :=
          tendsto_nhds_limUnder hx.1.1.1
        have I : ∀ᶠ b : Set α in v.filterAt x, ρ b / μ b < p := (tendsto_order.1 L).2 _ hx.1.1.2
        apply I.frequently.mono fun a ha => ?_
        rw [coe_nnreal_smul_apply]
        refine (ENNReal.div_le_iff_le_mul ?_ (Or.inr (bot_le.trans_lt ha).ne')).1 ha.le
        simp only [ENNReal.coe_ne_top, Ne, or_true, not_false_iff]
      _ = p * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) := by
        simp only [coe_nnreal_smul_apply,
          measure_toMeasurable_add_inter_right (measurableSet_toMeasurable _ _) I]
  have B :
    (q : ℝ≥0∞) * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) ≤
      ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) :=
    calc
      (q : ℝ≥0∞) * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) =
          (q : ℝ≥0∞) * μ (toMeasurable (ρ + μ) (u m) ∩ w n) := by
        conv_rhs => rw [inter_comm]
        rw [inter_comm, measure_toMeasurable_add_inter_right (measurableSet_toMeasurable _ _) J]
      _ ≤ ρ (toMeasurable (ρ + μ) (u m) ∩ w n) := by
        rw [← coe_nnreal_smul_apply]
        refine v.measure_le_of_frequently_le _ (.smul_left .rfl _) _ ?_
        intro x hx
        have L : Tendsto (fun a : Set α => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatio ρ x)) :=
          tendsto_nhds_limUnder hx.2.1.1
        have I : ∀ᶠ b : Set α in v.filterAt x, (q : ℝ≥0∞) < ρ b / μ b :=
          (tendsto_order.1 L).1 _ hx.2.1.2
        apply I.frequently.mono fun a ha => ?_
        rw [coe_nnreal_smul_apply]
        exact ENNReal.mul_le_of_le_div ha.le
      _ = ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) := by
        conv_rhs => rw [inter_comm]
        rw [inter_comm]
        exact (measure_toMeasurable_add_inter_left (measurableSet_toMeasurable _ _) J).symm
  /-
    case refine_5
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
    u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    m n : Nat
    I : Ne ((HAdd.hAdd ρ μ) (u m)) Top.top
    J : Ne ((HAdd.hAdd ρ μ) (w n)) Top.top
    A : LE.le (ρ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (u m)) ( …
    B : LE.le (HMul.hMul (↑q) (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hA …
    ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (u m)) (Measu …
  -/
  by_contra h
  /-
    case refine_5
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    s : Set α := setOf fun x => Exists fun c => Filter.Tendsto (fun a => HDiv.hDiv …
    o : Nat → Set α := MeasureTheory.spanningSets (HAdd.hAdd ρ μ)
    u : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    w : Nat → Set α := fun n => Inter.inter (Inter.inter s (setOf fun x => LT.lt ( …
    m n : Nat
    I : Ne ((HAdd.hAdd ρ μ) (u m)) Top.top
    J : Ne ((HAdd.hAdd ρ μ) (w n)) Top.top
    A : LE.le (ρ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (u m)) ( …
    B : LE.le (HMul.hMul (↑q) (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hA …
    h : Not (Eq (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ρ μ) (u m)) …
    ⊢ False
  -/
  apply lt_irrefl (ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)))
  calc
    ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) ≤
        p * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) :=
      A
    _ < q * μ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) := by
      gcongr
      suffices H : (ρ + μ) (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) ≠ ∞ by
        simp only [not_or, ENNReal.add_eq_top, Pi.add_apply, Ne, coe_add] at H
        exact H.2
      apply (lt_of_le_of_lt (measure_mono inter_subset_left) _).ne
      rw [measure_toMeasurable]
      apply lt_of_le_of_lt (measure_mono _) (measure_spanningSets_lt_top (ρ + μ) m)
      exact inter_subset_right
    _ ≤ ρ (toMeasurable (ρ + μ) (u m) ∩ toMeasurable (ρ + μ) (w n)) := B


theorem aemeasurable_limRatio : AEMeasurable (v.limRatio ρ) μ := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ AEMeasurable (v.limRatio ρ) μ
  -/
  apply ENNReal.aemeasurable_of_exist_almost_disjoint_supersets _ _ fun p q hpq => ?_
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p q : NNReal
    hpq : LT.lt p q
    ⊢ Exists fun u => Exists fun v_1 => And (MeasurableSet u) (And (MeasurableSet  …
  -/
  exact v.exists_measurable_supersets_limRatio hρ hpq
  /-
    🎉 no goals
  -/


/-- A measurable version of `v.limRatio ρ`. Do *not* use this definition: it is only a temporary
device to show that `v.limRatio` is almost everywhere equal to the Radon-Nikodym derivative. -/
noncomputable def limRatioMeas : α → ℝ≥0∞ :=
  (v.aemeasurable_limRatio hρ).mk _


theorem limRatioMeas_measurable : Measurable (v.limRatioMeas hρ) :=
  AEMeasurable.measurable_mk _


theorem ae_tendsto_limRatioMeas :
    ∀ᵐ x ∂μ, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatioMeas hρ x)) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  filter_upwards [v.ae_tendsto_limRatio hρ, AEMeasurable.ae_eq_mk (v.aemeasurable_limRatio hρ)]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ ∀ (a : α), Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt a) (n …
  -/
  intro x hx h'x
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (v.l …
    h'x : Eq (v.limRatio ρ x) (AEMeasurable.mk (v.limRatio ρ) ⋯ x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (v.limR …
  -/
  rwa [h'x] at hx
  /-
    🎉 no goals
  -/


/-- If, for all `x` in a set `s`, one has frequently `ρ a / μ a < p`, then `ρ s ≤ p * μ s`, as
proved in `measure_le_of_frequently_le`. Since `ρ a / μ a` tends almost everywhere to
`v.limRatioMeas hρ x`, the same property holds for sets `s` on which `v.limRatioMeas hρ < p`. -/
theorem measure_le_mul_of_subset_limRatioMeas_lt {p : ℝ≥0} {s : Set α}
    (h : s ⊆ {x | v.limRatioMeas hρ x < p}) : ρ s ≤ p * μ s := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    ⊢ LE.le (ρ s) (HMul.hMul (↑p) (μ s))
  -/
  let t := {x : α | Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatioMeas hρ x))}
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    ⊢ LE.le (ρ s) (HMul.hMul (↑p) (μ s))
  -/
  have A : μ tᶜ = 0 := v.ae_tendsto_limRatioMeas hρ
  suffices H : ρ (s ∩ t) ≤ (p • μ) (s ∩ t) by calc
    ρ s = ρ (s ∩ t ∪ s ∩ tᶜ) := by rw [inter_union_compl]
    _ ≤ ρ (s ∩ t) + ρ (s ∩ tᶜ) := measure_union_le _ _
    _ ≤ (p • μ) (s ∩ t) + ρ tᶜ := by gcongr; apply inter_subset_right
    _ ≤ p * μ (s ∩ t) := by simp [(hρ A)]
    _ ≤ p * μ s := by gcongr; apply inter_subset_left
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    ⊢ LE.le (ρ (Inter.inter s t)) ((HSMul.hSMul p μ) (Inter.inter s t))
  -/
  refine v.measure_le_of_frequently_le (p • μ) hρ _ fun x hx => ?_
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    ⊢ Filter.Frequently (fun a => LE.le (ρ a) ((HSMul.hSMul p μ) a)) (v.filterAt x)
  -/
  have I : ∀ᶠ b : Set α in v.filterAt x, ρ b / μ b < p := (tendsto_order.1 hx.2).2 _ (h hx.1)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) ↑p) (v.filterAt x)
    ⊢ Filter.Frequently (fun a => LE.le (ρ a) ((HSMul.hSMul p μ) a)) (v.filterAt x)
  -/
  apply I.frequently.mono fun a ha => ?_
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) ↑p) (v.filterAt x)
    a : Set α
    ha : LT.lt (HDiv.hDiv (ρ a) (μ a)) ↑p
    ⊢ LE.le (ρ a) ((HSMul.hSMul p μ) a)
  -/
  rw [coe_nnreal_smul_apply]
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) ↑p) (v.filterAt x)
    a : Set α
    ha : LT.lt (HDiv.hDiv (ρ a) (μ a)) ↑p
    ⊢ LE.le (ρ a) (HMul.hMul (↑p) (μ a))
  -/
  refine (ENNReal.div_le_iff_le_mul ?_ (Or.inr (bot_le.trans_lt ha).ne')).1 ha.le
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    p : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (v.limRatioMeas hρ x) ↑p)
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun b => LT.lt (HDiv.hDiv (ρ b) (μ b)) ↑p) (v.filterAt x)
    a : Set α
    ha : LT.lt (HDiv.hDiv (ρ a) (μ a)) ↑p
    ⊢ Or (Ne (μ a) 0) (Ne (↑p) Top.top)
  -/
  simp only [ENNReal.coe_ne_top, Ne, or_true, not_false_iff]
  /-
    🎉 no goals
  -/


/-- If, for all `x` in a set `s`, one has frequently `q < ρ a / μ a`, then `q * μ s ≤ ρ s`, as
proved in `measure_le_of_frequently_le`. Since `ρ a / μ a` tends almost everywhere to
`v.limRatioMeas hρ x`, the same property holds for sets `s` on which `q < v.limRatioMeas hρ`. -/
theorem mul_measure_le_of_subset_lt_limRatioMeas {q : ℝ≥0} {s : Set α}
    (h : s ⊆ {x | (q : ℝ≥0∞) < v.limRatioMeas hρ x}) : (q : ℝ≥0∞) * μ s ≤ ρ s := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    ⊢ LE.le (HMul.hMul (↑q) (μ s)) (ρ s)
  -/
  let t := {x : α | Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (v.limRatioMeas hρ x))}
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    ⊢ LE.le (HMul.hMul (↑q) (μ s)) (ρ s)
  -/
  have A : μ tᶜ = 0 := v.ae_tendsto_limRatioMeas hρ
  suffices H : (q • μ) (s ∩ t) ≤ ρ (s ∩ t) by calc
    (q • μ) s = (q • μ) (s ∩ t ∪ s ∩ tᶜ) := by rw [inter_union_compl]
    _ ≤ (q • μ) (s ∩ t) + (q • μ) (s ∩ tᶜ) := measure_union_le _ _
    _ ≤ ρ (s ∩ t) + (q • μ) tᶜ := by gcongr; apply inter_subset_right
    _ = ρ (s ∩ t) := by simp [A]
    _ ≤ ρ s := by gcongr; apply inter_subset_left
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    ⊢ LE.le ((HSMul.hSMul q μ) (Inter.inter s t)) (ρ (Inter.inter s t))
  -/
  refine v.measure_le_of_frequently_le _ (.smul_left .rfl _) _ ?_
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    ⊢ ∀ (x : α), Membership.mem (Inter.inter s t) x → Filter.Frequently (fun a =>  …
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    ⊢ Filter.Frequently (fun a => LE.le ((HSMul.hSMul q μ) a) (ρ a)) (v.filterAt x)
  -/
  have I : ∀ᶠ a in v.filterAt x, (q : ℝ≥0∞) < ρ a / μ a := (tendsto_order.1 hx.2).1 _ (h hx.1)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun a => LT.lt (↑q) (HDiv.hDiv (ρ a) (μ a))) (v.filterA …
    ⊢ Filter.Frequently (fun a => LE.le ((HSMul.hSMul q μ) a) (ρ a)) (v.filterAt x)
  -/
  apply I.frequently.mono fun a ha => ?_
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun a => LT.lt (↑q) (HDiv.hDiv (ρ a) (μ a))) (v.filterA …
    a : Set α
    ha : LT.lt (↑q) (HDiv.hDiv (ρ a) (μ a))
    ⊢ LE.le ((HSMul.hSMul q μ) a) (ρ a)
  -/
  rw [coe_nnreal_smul_apply]
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    q : NNReal
    s : Set α
    h : HasSubset.Subset s (setOf fun x => LT.lt (↑q) (v.limRatioMeas hρ x))
    t : Set α := setOf fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v …
    A : Eq (μ (HasCompl.compl t)) 0
    x : α
    hx : Membership.mem (Inter.inter s t) x
    I : Filter.Eventually (fun a => LT.lt (↑q) (HDiv.hDiv (ρ a) (μ a))) (v.filterA …
    a : Set α
    ha : LT.lt (↑q) (HDiv.hDiv (ρ a) (μ a))
    ⊢ LE.le (HMul.hMul (↑q) (μ a)) (ρ a)
  -/
  exact ENNReal.mul_le_of_le_div ha.le
  /-
    🎉 no goals
  -/


/-- The points with `v.limRatioMeas hρ x = ∞` have measure `0` for `μ`. -/
theorem measure_limRatioMeas_top : μ {x | v.limRatioMeas hρ x = ∞} = 0 := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ Eq (μ (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top)) 0
  -/
  refine measure_null_of_locally_null _ fun x _ => ?_
  obtain ⟨o, xo, o_open, μo⟩ : ∃ o : Set α, x ∈ o ∧ IsOpen o ∧ ρ o < ∞ :=
    Measure.exists_isOpen_measure_lt_top ρ x
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (setOf fun x => Eq (v.limR …
  -/
  let s := {x : α | v.limRatioMeas hρ x = ∞} ∩ o
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) o
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (setOf fun x => Eq (v.limR …
  -/
  refine ⟨s, inter_mem_nhdsWithin _ (o_open.mem_nhds xo), le_antisymm ?_ bot_le⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) o
    ⊢ LE.le (μ s) 0
  -/
  have ρs : ρ s ≠ ∞ := ((measure_mono inter_subset_right).trans_lt μo).ne
  have A : ∀ q : ℝ≥0, 1 ≤ q → μ s ≤ (q : ℝ≥0∞)⁻¹ * ρ s := by
    intro q hq
    rw [mul_comm, ← div_eq_mul_inv, ENNReal.le_div_iff_mul_le _ (Or.inr ρs), mul_comm]
    · apply v.mul_measure_le_of_subset_lt_limRatioMeas hρ
      intro y hy
      have : v.limRatioMeas hρ y = ∞ := hy.1
      simp only [this, ENNReal.coe_lt_top, mem_setOf_eq]
    · simp only [(zero_lt_one.trans_le hq).ne', true_or, ENNReal.coe_eq_zero, Ne,
        not_false_iff]
  have B : Tendsto (fun q : ℝ≥0 => (q : ℝ≥0∞)⁻¹ * ρ s) atTop (𝓝 (∞⁻¹ * ρ s)) := by
    apply ENNReal.Tendsto.mul_const _ (Or.inr ρs)
    exact ENNReal.tendsto_inv_iff.2 (ENNReal.tendsto_coe_nhds_top.2 tendsto_id)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) o
    ρs : Ne (ρ s) Top.top
    A : ∀ (q : NNReal), LE.le 1 q → LE.le (μ s) (HMul.hMul (Inv.inv ↑q) (ρ s))
    B : Filter.Tendsto (fun q => HMul.hMul (Inv.inv ↑q) (ρ s)) Filter.atTop (nhds  …
    ⊢ LE.le (μ s) 0
  -/
  simp only [zero_mul, ENNReal.inv_top] at B
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) o
    ρs : Ne (ρ s) Top.top
    A : ∀ (q : NNReal), LE.le 1 q → LE.le (μ s) (HMul.hMul (Inv.inv ↑q) (ρ s))
    B : Filter.Tendsto (fun q => HMul.hMul (Inv.inv ↑q) (ρ s)) Filter.atTop (nhds 0)
    ⊢ LE.le (μ s) 0
  -/
  apply ge_of_tendsto B
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (ρ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) Top.top) o
    ρs : Ne (ρ s) Top.top
    A : ∀ (q : NNReal), LE.le 1 q → LE.le (μ s) (HMul.hMul (Inv.inv ↑q) (ρ s))
    B : Filter.Tendsto (fun q => HMul.hMul (Inv.inv ↑q) (ρ s)) Filter.atTop (nhds 0)
    ⊢ Filter.Eventually (fun c => LE.le (μ s) (HMul.hMul (Inv.inv ↑c) (ρ s))) Filt …
  -/
  exact eventually_atTop.2 ⟨1, A⟩
  /-
    🎉 no goals
  -/


/-- The points with `v.limRatioMeas hρ x = 0` have measure `0` for `ρ`. -/
theorem measure_limRatioMeas_zero : ρ {x | v.limRatioMeas hρ x = 0} = 0 := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ Eq (ρ (setOf fun x => Eq (v.limRatioMeas hρ x) 0)) 0
  -/
  refine measure_null_of_locally_null _ fun x _ => ?_
  obtain ⟨o, xo, o_open, μo⟩ : ∃ o : Set α, x ∈ o ∧ IsOpen o ∧ μ o < ∞ :=
    Measure.exists_isOpen_measure_lt_top μ x
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (setOf fun x => Eq (v.limR …
  -/
  let s := {x : α | v.limRatioMeas hρ x = 0} ∩ o
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) 0) o
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (setOf fun x => Eq (v.limR …
  -/
  refine ⟨s, inter_mem_nhdsWithin _ (o_open.mem_nhds xo), le_antisymm ?_ bot_le⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) 0) o
    ⊢ LE.le (ρ s) 0
  -/
  have μs : μ s ≠ ∞ := ((measure_mono inter_subset_right).trans_lt μo).ne
  have A : ∀ q : ℝ≥0, 0 < q → ρ s ≤ q * μ s := by
    intro q hq
    apply v.measure_le_mul_of_subset_limRatioMeas_lt hρ
    intro y hy
    have : v.limRatioMeas hρ y = 0 := hy.1
    simp only [this, mem_setOf_eq, hq, ENNReal.coe_pos]
  have B : Tendsto (fun q : ℝ≥0 => (q : ℝ≥0∞) * μ s) (𝓝[>] (0 : ℝ≥0)) (𝓝 ((0 : ℝ≥0) * μ s)) := by
    apply ENNReal.Tendsto.mul_const _ (Or.inr μs)
    rw [ENNReal.tendsto_coe]
    exact nhdsWithin_le_nhds
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) 0) o
    μs : Ne (μ s) Top.top
    A : ∀ (q : NNReal), LT.lt 0 q → LE.le (ρ s) (HMul.hMul (↑q) (μ s))
    B : Filter.Tendsto (fun q => HMul.hMul (↑q) (μ s)) (nhdsWithin 0 (Set.Ioi 0))  …
    ⊢ LE.le (ρ s) 0
  -/
  simp only [zero_mul, ENNReal.coe_zero] at B
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) 0) o
    μs : Ne (μ s) Top.top
    A : ∀ (q : NNReal), LT.lt 0 q → LE.le (ρ s) (HMul.hMul (↑q) (μ s))
    B : Filter.Tendsto (fun q => HMul.hMul (↑q) (μ s)) (nhdsWithin 0 (Set.Ioi 0))  …
    ⊢ LE.le (ρ s) 0
  -/
  apply ge_of_tendsto B
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    x : α
    x✝ : Membership.mem (setOf fun x => Eq (v.limRatioMeas hρ x) 0) x
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    s : Set α := Inter.inter (setOf fun x => Eq (v.limRatioMeas hρ x) 0) o
    μs : Ne (μ s) Top.top
    A : ∀ (q : NNReal), LT.lt 0 q → LE.le (ρ s) (HMul.hMul (↑q) (μ s))
    B : Filter.Tendsto (fun q => HMul.hMul (↑q) (μ s)) (nhdsWithin 0 (Set.Ioi 0))  …
    ⊢ Filter.Eventually (fun c => LE.le (ρ s) (HMul.hMul (↑c) (μ s))) (nhdsWithin  …
  -/
  filter_upwards [self_mem_nhdsWithin] using A
  /-
    🎉 no goals
  -/


/-- As an intermediate step to show that `μ.withDensity (v.limRatioMeas hρ) = ρ`, we show here
that `μ.withDensity (v.limRatioMeas hρ) ≤ t^2 ρ` for any `t > 1`. -/
theorem withDensity_le_mul {s : Set α} (hs : MeasurableSet s) {t : ℝ≥0} (ht : 1 < t) :
    μ.withDensity (v.limRatioMeas hρ) s ≤ (t : ℝ≥0∞) ^ 2 * ρ s := by
  /- We cut `s` into the sets where `v.limRatioMeas hρ = 0`, where `v.limRatioMeas hρ = ∞`, and
    where `v.limRatioMeas hρ ∈ [t^n, t^(n+1))` for `n : ℤ`. The first and second have measure `0`.
    For the latter, since `v.limRatioMeas hρ` fluctuates by at most `t` on this slice, we can use
    `measure_le_mul_of_subset_limRatioMeas_lt` and `mul_measure_le_of_subset_lt_limRatioMeas` to
    show that the two measures are comparable up to `t` (in fact `t^2` for technical reasons of
    strict inequalities). -/
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑t) 2)  …
  -/
  have t_ne_zero' : t ≠ 0 := (zero_lt_one.trans ht).ne'
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑t) 2)  …
  -/
  have t_ne_zero : (t : ℝ≥0∞) ≠ 0 := by simpa only [ENNReal.coe_eq_zero, Ne] using t_ne_zero'
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑t) 2)  …
  -/
  let ν := μ.withDensity (v.limRatioMeas hρ)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ν : MeasureTheory.Measure α := μ.withDensity (v.limRatioMeas hρ)
    ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑t) 2)  …
  -/
  let f := v.limRatioMeas hρ
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ν : MeasureTheory.Measure α := μ.withDensity (v.limRatioMeas hρ)
    f : α → ENNReal := v.limRatioMeas hρ
    ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑t) 2)  …
  -/
  have f_meas : Measurable f := v.limRatioMeas_measurable hρ
  -- Note(kmill): smul elaborator when used for CoeFun fails to get CoeFun instance to trigger
  -- unless you use the `(... :)` notation. Another fix is using `(2 : Nat)`, so this appears
  -- to be an unpleasant interaction with default instances.
  have A : ν (s ∩ f ⁻¹' {0}) ≤ ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' {0}) := by
    apply le_trans _ (zero_le _)
    have M : MeasurableSet (s ∩ f ⁻¹' {0}) := hs.inter (f_meas (measurableSet_singleton _))
    simp only [f, ν, nonpos_iff_eq_zero, M, withDensity_apply, lintegral_eq_zero_iff f_meas]
    apply (ae_restrict_iff' M).2
    exact Eventually.of_forall fun x hx => hx.2
  have B : ν (s ∩ f ⁻¹' {∞}) ≤ ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' {∞}) := by
    apply le_trans (le_of_eq _) (zero_le _)
    apply withDensity_absolutelyContinuous μ _
    rw [← nonpos_iff_eq_zero]
    exact (measure_mono inter_subset_right).trans (v.measure_limRatioMeas_top hρ).le
  have C :
    ∀ n : ℤ,
      ν (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) ≤
        ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) := by
    intro n
    let I := Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))
    have M : MeasurableSet (s ∩ f ⁻¹' I) := hs.inter (f_meas measurableSet_Ico)
    simp only [ν, I, M, withDensity_apply, coe_nnreal_smul_apply]
    calc
      (∫⁻ x in s ∩ f ⁻¹' I, f x ∂μ) ≤ ∫⁻ _ in s ∩ f ⁻¹' I, (t : ℝ≥0∞) ^ (n + 1) ∂μ :=
        lintegral_mono_ae ((ae_restrict_iff' M).2 (Eventually.of_forall fun x hx => hx.2.2.le))
      _ = (t : ℝ≥0∞) ^ (n + 1) * μ (s ∩ f ⁻¹' I) := by
        simp only [lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter]
      _ = (t : ℝ≥0∞) ^ (2 : ℤ) * ((t : ℝ≥0∞) ^ (n - 1) * μ (s ∩ f ⁻¹' I)) := by
        rw [← mul_assoc, ← ENNReal.zpow_add t_ne_zero ENNReal.coe_ne_top]
        congr 2
        abel
      _ ≤ (t : ℝ≥0∞) ^ (2 : ℤ) * ρ (s ∩ f ⁻¹' I) := by
        gcongr
        rw [← ENNReal.coe_zpow (zero_lt_one.trans ht).ne']
        apply v.mul_measure_le_of_subset_lt_limRatioMeas hρ
        intro x hx
        apply lt_of_lt_of_le _ hx.2.1
        rw [← ENNReal.coe_zpow (zero_lt_one.trans ht).ne', ENNReal.coe_lt_coe, sub_eq_add_neg,
          zpow_add₀ t_ne_zero']
        conv_rhs => rw [← mul_one (t ^ n)]
        gcongr
        rw [zpow_neg_one]
        exact inv_lt_one_of_one_lt₀ ht
  calc
    ν s =
      ν (s ∩ f ⁻¹' {0}) + ν (s ∩ f ⁻¹' {∞}) +
        ∑' n : ℤ, ν (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) :=
      measure_eq_measure_preimage_add_measure_tsum_Ico_zpow ν f_meas hs ht
    _ ≤
        ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' {0}) + ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' {∞}) +
          ∑' n : ℤ, ((t : ℝ≥0∞) ^ 2 • ρ :) (s ∩ f ⁻¹' Ico (t ^ n) (t ^ (n + 1))) :=
      (add_le_add (add_le_add A B) (ENNReal.tsum_le_tsum C))
    _ = ((t : ℝ≥0∞) ^ 2 • ρ :) s :=
      (measure_eq_measure_preimage_add_measure_tsum_Ico_zpow ((t : ℝ≥0∞) ^ 2 • ρ) f_meas hs ht).symm


/-- As an intermediate step to show that `μ.withDensity (v.limRatioMeas hρ) = ρ`, we show here
that `ρ ≤ t μ.withDensity (v.limRatioMeas hρ)` for any `t > 1`. -/
theorem le_mul_withDensity {s : Set α} (hs : MeasurableSet s) {t : ℝ≥0} (ht : 1 < t) :
    ρ s ≤ t * μ.withDensity (v.limRatioMeas hρ) s := by
  /- We cut `s` into the sets where `v.limRatioMeas hρ = 0`, where `v.limRatioMeas hρ = ∞`, and
    where `v.limRatioMeas hρ ∈ [t^n, t^(n+1))` for `n : ℤ`. The first and second have measure `0`.
    For the latter, since `v.limRatioMeas hρ` fluctuates by at most `t` on this slice, we can use
    `measure_le_mul_of_subset_limRatioMeas_lt` and `mul_measure_le_of_subset_lt_limRatioMeas` to
    show that the two measures are comparable up to `t`. -/
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    ⊢ LE.le (ρ s) (HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas hρ)) s))
  -/
  have t_ne_zero' : t ≠ 0 := (zero_lt_one.trans ht).ne'
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    ⊢ LE.le (ρ s) (HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas hρ)) s))
  -/
  have t_ne_zero : (t : ℝ≥0∞) ≠ 0 := by simpa only [ENNReal.coe_eq_zero, Ne] using t_ne_zero'
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ⊢ LE.le (ρ s) (HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas hρ)) s))
  -/
  let ν := μ.withDensity (v.limRatioMeas hρ)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ν : MeasureTheory.Measure α := μ.withDensity (v.limRatioMeas hρ)
    ⊢ LE.le (ρ s) (HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas hρ)) s))
  -/
  let f := v.limRatioMeas hρ
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    t_ne_zero' : Ne t 0
    t_ne_zero : Ne (↑t) 0
    ν : MeasureTheory.Measure α := μ.withDensity (v.limRatioMeas hρ)
    f : α → ENNReal := v.limRatioMeas hρ
    ⊢ LE.le (ρ s) (HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas hρ)) s))
  -/
  have f_meas : Measurable f := v.limRatioMeas_measurable hρ
  have A : ρ (s ∩ f ⁻¹' {0}) ≤ (t • ν) (s ∩ f ⁻¹' {0}) := by
    refine le_trans (measure_mono inter_subset_right) (le_trans (le_of_eq ?_) (zero_le _))
    exact v.measure_limRatioMeas_zero hρ
  have B : ρ (s ∩ f ⁻¹' {∞}) ≤ (t • ν) (s ∩ f ⁻¹' {∞}) := by
    apply le_trans (le_of_eq _) (zero_le _)
    apply hρ
    rw [← nonpos_iff_eq_zero]
    exact (measure_mono inter_subset_right).trans (v.measure_limRatioMeas_top hρ).le
  have C :
    ∀ n : ℤ,
      ρ (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) ≤
        (t • ν) (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) := by
    intro n
    let I := Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))
    have M : MeasurableSet (s ∩ f ⁻¹' I) := hs.inter (f_meas measurableSet_Ico)
    simp only [ν, I, M, withDensity_apply, coe_nnreal_smul_apply]
    calc
      ρ (s ∩ f ⁻¹' I) ≤ (t : ℝ≥0∞) ^ (n + 1) * μ (s ∩ f ⁻¹' I) := by
        rw [← ENNReal.coe_zpow t_ne_zero']
        apply v.measure_le_mul_of_subset_limRatioMeas_lt hρ
        intro x hx
        apply hx.2.2.trans_le (le_of_eq _)
        rw [ENNReal.coe_zpow t_ne_zero']
      _ = ∫⁻ _ in s ∩ f ⁻¹' I, (t : ℝ≥0∞) ^ (n + 1) ∂μ := by
        simp only [lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter]
      _ ≤ ∫⁻ x in s ∩ f ⁻¹' I, t * f x ∂μ := by
        apply lintegral_mono_ae ((ae_restrict_iff' M).2 (Eventually.of_forall fun x hx => ?_))
        rw [add_comm, ENNReal.zpow_add t_ne_zero ENNReal.coe_ne_top, zpow_one]
        exact mul_le_mul_left' hx.2.1 _
      _ = t * ∫⁻ x in s ∩ f ⁻¹' I, f x ∂μ := lintegral_const_mul _ f_meas
  calc
    ρ s =
      ρ (s ∩ f ⁻¹' {0}) + ρ (s ∩ f ⁻¹' {∞}) +
        ∑' n : ℤ, ρ (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) :=
      measure_eq_measure_preimage_add_measure_tsum_Ico_zpow ρ f_meas hs ht
    _ ≤
        (t • ν) (s ∩ f ⁻¹' {0}) + (t • ν) (s ∩ f ⁻¹' {∞}) +
          ∑' n : ℤ, (t • ν) (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) :=
      (add_le_add (add_le_add A B) (ENNReal.tsum_le_tsum C))
    _ = (t • ν) s :=
      (measure_eq_measure_preimage_add_measure_tsum_Ico_zpow (t • ν) f_meas hs ht).symm


theorem withDensity_limRatioMeas_eq : μ.withDensity (v.limRatioMeas hρ) = ρ := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    ⊢ Eq (μ.withDensity (v.limRatioMeas hρ)) ρ
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.withDensity (v.limRatioMeas hρ)) s) (ρ s)
  -/
  refine le_antisymm ?_ ?_
  · have : Tendsto (fun t : ℝ≥0 =>
        ((t : ℝ≥0∞) ^ 2 * ρ s : ℝ≥0∞)) (𝓝[>] 1) (𝓝 ((1 : ℝ≥0∞) ^ 2 * ρ s)) := by
      refine ENNReal.Tendsto.mul ?_ ?_ tendsto_const_nhds ?_
      · exact ENNReal.Tendsto.pow (ENNReal.tendsto_coe.2 nhdsWithin_le_nhds)
      · simp only [one_pow, ENNReal.coe_one, true_or, Ne, not_false_iff, one_ne_zero]
      · simp only [one_pow, ENNReal.coe_one, Ne, or_true, ENNReal.one_ne_top, not_false_iff]
    /-
      case h.refine_1
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (HPow.hPow (↑t) 2) (ρ s)) (nhdsWithi …
      ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (ρ s)
    -/
    simp only [one_pow, one_mul, ENNReal.coe_one] at this
    /-
      case h.refine_1
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (HPow.hPow (↑t) 2) (ρ s)) (nhdsWithi …
      ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (ρ s)
    -/
    refine ge_of_tendsto this ?_
    /-
      case h.refine_1
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (HPow.hPow (↑t) 2) (ρ s)) (nhdsWithi …
      ⊢ Filter.Eventually (fun c => LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (H …
    -/
    filter_upwards [self_mem_nhdsWithin] with _ ht
    /-
      case h
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (HPow.hPow (↑t) 2) (ρ s)) (nhdsWithi …
      a✝ : NNReal
      ht : Membership.mem (Set.Ioi 1) a✝
      ⊢ LE.le ((μ.withDensity (v.limRatioMeas hρ)) s) (HMul.hMul (HPow.hPow (↑a✝) 2) …
    -/
    exact v.withDensity_le_mul hρ hs ht
    /-
      🎉 no goals
    -/
  · have :
      Tendsto (fun t : ℝ≥0 => (t : ℝ≥0∞) * μ.withDensity (v.limRatioMeas hρ) s) (𝓝[>] 1)
        (𝓝 ((1 : ℝ≥0∞) * μ.withDensity (v.limRatioMeas hρ) s)) := by
      refine ENNReal.Tendsto.mul_const (ENNReal.tendsto_coe.2 nhdsWithin_le_nhds) ?_
      simp only [ENNReal.coe_one, true_or, Ne, not_false_iff, one_ne_zero]
    /-
      case h.refine_2
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas …
      ⊢ LE.le (ρ s) ((μ.withDensity (v.limRatioMeas hρ)) s)
    -/
    simp only [one_mul, ENNReal.coe_one] at this
    /-
      case h.refine_2
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas …
      ⊢ LE.le (ρ s) ((μ.withDensity (v.limRatioMeas hρ)) s)
    -/
    refine ge_of_tendsto this ?_
    /-
      case h.refine_2
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas …
      ⊢ Filter.Eventually (fun c => LE.le (ρ s) (HMul.hMul (↑c) ((μ.withDensity (v.l …
    -/
    filter_upwards [self_mem_nhdsWithin] with _ ht
    /-
      case h
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      hρ : ρ.AbsolutelyContinuous μ
      s : Set α
      hs : MeasurableSet s
      this : Filter.Tendsto (fun t => HMul.hMul (↑t) ((μ.withDensity (v.limRatioMeas …
      a✝ : NNReal
      ht : Membership.mem (Set.Ioi 1) a✝
      ⊢ LE.le (ρ s) (HMul.hMul (↑a✝) ((μ.withDensity (v.limRatioMeas hρ)) s))
    -/
    exact v.le_mul_withDensity hρ hs ht
    /-
      🎉 no goals
    -/


/-- Weak version of the main theorem on differentiation of measures: given a Vitali family `v`
for a locally finite measure `μ`, and another locally finite measure `ρ`, then for `μ`-almost
every `x` the ratio `ρ a / μ a` converges, when `a` shrinks to `x` along the Vitali family,
towards the Radon-Nikodym derivative of `ρ` with respect to `μ`.

This version assumes that `ρ` is absolutely continuous with respect to `μ`. The general version
without this superfluous assumption is `VitaliFamily.ae_tendsto_rnDeriv`.
-/
theorem ae_tendsto_rnDeriv_of_absolutelyContinuous :
    ∀ᵐ x ∂μ, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (ρ.rnDeriv μ x)) := by
  have A : (μ.withDensity (v.limRatioMeas hρ)).rnDeriv μ =ᵐ[μ] v.limRatioMeas hρ :=
    rnDeriv_withDensity μ (v.limRatioMeas_measurable hρ)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    A : (MeasureTheory.ae μ).EventuallyEq ((μ.withDensity (v.limRatioMeas hρ)).rnD …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  rw [v.withDensity_limRatioMeas_eq hρ] at A
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    A : (MeasureTheory.ae μ).EventuallyEq (ρ.rnDeriv μ) (v.limRatioMeas hρ)
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  filter_upwards [v.ae_tendsto_limRatioMeas hρ, A] with _ _ h'x
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    hρ : ρ.AbsolutelyContinuous μ
    A : (MeasureTheory.ae μ).EventuallyEq (ρ.rnDeriv μ) (v.limRatioMeas hρ)
    a✝¹ : α
    a✝ : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt a✝¹) (nhds (v …
    h'x : Eq (ρ.rnDeriv μ a✝¹) (v.limRatioMeas hρ a✝¹)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt a✝¹) (nhds (ρ.rn …
  -/
  rwa [h'x]
  /-
    🎉 no goals
  -/


/-- Main theorem on differentiation of measures: given a Vitali family `v` for a locally finite
measure `μ`, and another locally finite measure `ρ`, then for `μ`-almost every `x` the
ratio `ρ a / μ a` converges, when `a` shrinks to `x` along the Vitali family, towards the
Radon-Nikodym derivative of `ρ` with respect to `μ`. -/
theorem ae_tendsto_rnDeriv :
    ∀ᵐ x ∂μ, Tendsto (fun a => ρ a / μ a) (v.filterAt x) (𝓝 (ρ.rnDeriv μ x)) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  let t := μ.withDensity (ρ.rnDeriv μ)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  have eq_add : ρ = ρ.singularPart μ + t := haveLebesgueDecomposition_add _ _
  have A : ∀ᵐ x ∂μ, Tendsto (fun a => ρ.singularPart μ a / μ a) (v.filterAt x) (𝓝 0) :=
    v.ae_eventually_measure_zero_of_singular (mutuallySingular_singularPart ρ μ)
  have B : ∀ᵐ x ∂μ, t.rnDeriv μ x = ρ.rnDeriv μ x :=
    rnDeriv_withDensity μ (measurable_rnDeriv ρ μ)
  have C : ∀ᵐ x ∂μ, Tendsto (fun a => t a / μ a) (v.filterAt x) (𝓝 (t.rnDeriv μ x)) :=
    v.ae_tendsto_rnDeriv_of_absolutelyContinuous (withDensity_absolutelyContinuous _ _)
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
    eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
    B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
    C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a))  …
  -/
  filter_upwards [A, B, C] with _ Ax Bx Cx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝³ : SecondCountableTopology α
    inst✝² : BorelSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    ρ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
    eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
    B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
    C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
    a✝ : α
    Ax : Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singularPart μ) a) (μ a)) (v.filte …
    Bx : Eq (t.rnDeriv μ a✝) (ρ.rnDeriv μ a✝)
    Cx : Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a)) (v.filterAt a✝) (nhds (t. …
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt a✝) (nhds (ρ.rnD …
  -/
  convert Ax.add Cx using 1
    /-
      case h.e'_3
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
      eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
      A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
      B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
      C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
      a✝ : α
      Ax : Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singularPart μ) a) (μ a)) (v.filte …
      Bx : Eq (t.rnDeriv μ a✝) (ρ.rnDeriv μ a✝)
      Cx : Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a)) (v.filterAt a✝) (nhds (t. …
      ⊢ Eq (fun a => HDiv.hDiv (ρ a) (μ a)) fun x => HAdd.hAdd (HDiv.hDiv ((ρ.singul …
    -/
  · ext1 a
    /-
      case h.e'_3.h
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
      eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
      A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
      B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
      C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
      a✝ : α
      Ax : Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singularPart μ) a) (μ a)) (v.filte …
      Bx : Eq (t.rnDeriv μ a✝) (ρ.rnDeriv μ a✝)
      Cx : Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a)) (v.filterAt a✝) (nhds (t. …
      a : Set α
      ⊢ Eq (HDiv.hDiv (ρ a) (μ a)) (HAdd.hAdd (HDiv.hDiv ((ρ.singularPart μ) a) (μ a …
    -/
    conv_lhs => rw [eq_add]
    /-
      case h.e'_3.h
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
      eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
      A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
      B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
      C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
      a✝ : α
      Ax : Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singularPart μ) a) (μ a)) (v.filte …
      Bx : Eq (t.rnDeriv μ a✝) (ρ.rnDeriv μ a✝)
      Cx : Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a)) (v.filterAt a✝) (nhds (t. …
      a : Set α
      ⊢ Eq (HDiv.hDiv ((HAdd.hAdd (ρ.singularPart μ) t) a) (μ a)) (HAdd.hAdd (HDiv.h …
    -/
    simp only [Pi.add_apply, coe_add, ENNReal.add_div]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5
      α : Type u_1
      inst✝⁴ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      inst✝³ : SecondCountableTopology α
      inst✝² : BorelSpace α
      inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
      ρ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
      t : MeasureTheory.Measure α := μ.withDensity (ρ.rnDeriv μ)
      eq_add : Eq ρ (HAdd.hAdd (ρ.singularPart μ) t)
      A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singula …
      B : Filter.Eventually (fun x => Eq (t.rnDeriv μ x) (ρ.rnDeriv μ x)) (MeasureTh …
      C : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a) …
      a✝ : α
      Ax : Filter.Tendsto (fun a => HDiv.hDiv ((ρ.singularPart μ) a) (μ a)) (v.filte …
      Bx : Eq (t.rnDeriv μ a✝) (ρ.rnDeriv μ a✝)
      Cx : Filter.Tendsto (fun a => HDiv.hDiv (t a) (μ a)) (v.filterAt a✝) (nhds (t. …
      ⊢ Eq (nhds (ρ.rnDeriv μ a✝)) (nhds (HAdd.hAdd 0 (t.rnDeriv μ a✝)))
    -/
  · simp only [Bx, zero_add]
    /-
      🎉 no goals
    -/


/-- Given a measurable set `s`, then `μ (s ∩ a) / μ a` converges when `a` shrinks to a typical
point `x` along a Vitali family. The limit is `1` for `x ∈ s` and `0` for `x ∉ s`. This shows that
almost every point of `s` is a Lebesgue density point for `s`. A version for non-measurable sets
holds, but it only gives the first conclusion, see `ae_tendsto_measure_inter_div`. -/
theorem ae_tendsto_measure_inter_div_of_measurableSet {s : Set α} (hs : MeasurableSet s) :
    ∀ᵐ x ∂μ, Tendsto (fun a => μ (s ∩ a) / μ a) (v.filterAt x) (𝓝 (s.indicator 1 x)) := by
  haveI : IsLocallyFiniteMeasure (μ.restrict s) :=
    isLocallyFiniteMeasure_of_le restrict_le_self
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    hs : MeasurableSet s
    this : MeasureTheory.IsLocallyFiniteMeasure (μ.restrict s)
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.int …
  -/
  filter_upwards [ae_tendsto_rnDeriv v (μ.restrict s), rnDeriv_restrict_self μ hs]
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    hs : MeasurableSet s
    this : MeasureTheory.IsLocallyFiniteMeasure (μ.restrict s)
    ⊢ ∀ (a : α), Filter.Tendsto (fun a => HDiv.hDiv ((μ.restrict s) a) (μ a)) (v.f …
  -/
  intro x hx h'x
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    hs : MeasurableSet s
    this : MeasureTheory.IsLocallyFiniteMeasure (μ.restrict s)
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv ((μ.restrict s) a) (μ a)) (v.filterAt  …
    h'x : Eq ((μ.restrict s).rnDeriv μ x) (s.indicator 1 x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter s a)) (μ a)) (v.filterAt  …
  -/
  simpa only [h'x, restrict_apply' hs, inter_comm] using hx
  /-
    🎉 no goals
  -/


/-- Given an arbitrary set `s`, then `μ (s ∩ a) / μ a` converges to `1` when `a` shrinks to a
typical point of `s` along a Vitali family. This shows that almost every point of `s` is a
Lebesgue density point for `s`. A stronger version for measurable sets is given
in `ae_tendsto_measure_inter_div_of_measurableSet`. -/
theorem ae_tendsto_measure_inter_div (s : Set α) :
    ∀ᵐ x ∂μ.restrict s, Tendsto (fun a => μ (s ∩ a) / μ a) (v.filterAt x) (𝓝 1) := by
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.int …
  -/
  let t := toMeasurable μ s
  have A :
    ∀ᵐ x ∂μ.restrict s,
      Tendsto (fun a => μ (t ∩ a) / μ a) (v.filterAt x) (𝓝 (t.indicator 1 x)) := by
    apply ae_mono restrict_le_self
    apply ae_tendsto_measure_inter_div_of_measurableSet
    exact measurableSet_toMeasurable _ _
  have B : ∀ᵐ x ∂μ.restrict s, t.indicator 1 x = (1 : ℝ≥0∞) := by
    refine ae_restrict_of_ae_restrict_of_subset (subset_toMeasurable μ s) ?_
    filter_upwards [ae_restrict_mem (measurableSet_toMeasurable μ s)] with _ hx
    simp only [t, hx, Pi.one_apply, indicator_of_mem]
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.int …
  -/
  filter_upwards [A, B] with x hx h'x
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) (v.filter …
    h'x : Eq (t.indicator 1 x) 1
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter s a)) (μ a)) (v.filterAt  …
  -/
  rw [h'x] at hx
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) (v.filter …
    h'x : Eq (t.indicator 1 x) 1
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter s a)) (μ a)) (v.filterAt  …
  -/
  apply hx.congr' _
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) (v.filter …
    h'x : Eq (t.indicator 1 x) 1
    ⊢ (v.filterAt x).EventuallyEq (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) …
  -/
  filter_upwards [v.eventually_filterAt_measurableSet x] with _ ha
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) (v.filter …
    h'x : Eq (t.indicator 1 x) 1
    a✝ : Set α
    ha : MeasurableSet a✝
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter t a✝)) (μ a✝)) (HDiv.hDiv (μ (Inter.inter s a✝ …
  -/
  congr 1
  /-
    case h.e_a
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set α
    t : Set α := MeasureTheory.toMeasurable μ s
    A : Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.i …
    B : Filter.Eventually (fun x => Eq (t.indicator 1 x) 1) (MeasureTheory.ae (μ.r …
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter t a)) (μ a)) (v.filter …
    h'x : Eq (t.indicator 1 x) 1
    a✝ : Set α
    ha : MeasurableSet a✝
    ⊢ Eq (μ (Inter.inter t a✝)) (μ (Inter.inter s a✝))
  -/
  exact measure_toMeasurable_inter_of_sFinite ha _
  /-
    🎉 no goals
  -/


theorem ae_tendsto_lintegral_div' {f : α → ℝ≥0∞} (hf : Measurable f) (h'f : (∫⁻ y, f y ∂μ) ≠ ∞) :
    ∀ᵐ x ∂μ, Tendsto (fun a => (∫⁻ y in a, f y ∂μ) / μ a) (v.filterAt x) (𝓝 (f x)) := by
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  let ρ := μ.withDensity f
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  have : IsFiniteMeasure ρ := isFiniteMeasure_withDensity h'f
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    this : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  filter_upwards [ae_tendsto_rnDeriv v ρ, rnDeriv_withDensity μ hf] with x hx h'x
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    this : MeasureTheory.IsFiniteMeasure ρ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (ρ.r …
    h'x : Eq ((μ.withDensity f).rnDeriv μ x) (f x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  rw [← h'x]
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    this : MeasureTheory.IsFiniteMeasure ρ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (ρ.r …
    h'x : Eq ((μ.withDensity f).rnDeriv μ x) (f x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  apply hx.congr' _
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    this : MeasureTheory.IsFiniteMeasure ρ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (ρ.r …
    h'x : Eq ((μ.withDensity f).rnDeriv μ x) (f x)
    ⊢ (v.filterAt x).EventuallyEq (fun a => HDiv.hDiv (ρ a) (μ a)) fun a => HDiv.h …
  -/
  filter_upwards [v.eventually_filterAt_measurableSet x] with a ha
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : Measurable f
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    ρ : MeasureTheory.Measure α := μ.withDensity f
    this : MeasureTheory.IsFiniteMeasure ρ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) (v.filterAt x) (nhds (ρ.r …
    h'x : Eq ((μ.withDensity f).rnDeriv μ x) (f x)
    a : Set α
    ha : MeasurableSet a
    ⊢ Eq (HDiv.hDiv (ρ a) (μ a)) (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
  -/
  rw [← withDensity_apply f ha]
  /-
    🎉 no goals
  -/


theorem ae_tendsto_lintegral_div {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (h'f : (∫⁻ y, f y ∂μ) ≠ ∞) :
    ∀ᵐ x ∂μ, Tendsto (fun a => (∫⁻ y in a, f y ∂μ) / μ a) (v.filterAt x) (𝓝 (f x)) := by
  have A : (∫⁻ y, hf.mk f y ∂μ) ≠ ∞ := by
    convert h'f using 1
    apply lintegral_congr_ae
    exact hf.ae_eq_mk.symm
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  filter_upwards [v.ae_tendsto_lintegral_div' hf.measurable_mk A, hf.ae_eq_mk] with x hx h'x
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  rw [h'x]
  /-
    case h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  convert hx using 1
  /-
    case h.e'_3
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    ⊢ Eq (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => f y) …
  -/
  ext1 a
  /-
    case h.e'_3.h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    a : Set α
    ⊢ Eq (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => f y) (μ a)) ( …
  -/
  congr 1
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    a : Set α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict a) fun y => f y) (MeasureTheory.lint …
  -/
  apply lintegral_congr_ae
  /-
    case h.e'_3.h.e_a.h
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → ENNReal
    hf : AEMeasurable f μ
    h'f : Ne (MeasureTheory.lintegral μ fun y => f y) Top.top
    A : Ne (MeasureTheory.lintegral μ fun y => AEMeasurable.mk f hf y) Top.top
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (AEMeasurable.mk f hf x)
    a : Set α
    ⊢ (MeasureTheory.ae (μ.restrict a)).EventuallyEq f (AEMeasurable.mk f hf)
  -/
  exact ae_restrict_of_ae hf.ae_eq_mk
  /-
    🎉 no goals
  -/


theorem ae_tendsto_lintegral_nnnorm_sub_div'_of_integrable {f : α → E} (hf : Integrable f μ)
    (h'f : StronglyMeasurable f) :
    ∀ᵐ x ∂μ, Tendsto (fun a => (∫⁻ y in a, ‖f y - f x‖₊ ∂μ) / μ a) (v.filterAt x) (𝓝 0) := by
  /- For every `c`, then `(∫⁻ y in a, ‖f y - c‖₊ ∂μ) / μ a` tends almost everywhere to `‖f x - c‖`.
    We apply this to a countable set of `c` which is dense in the range of `f`, to deduce the
    desired convergence.
    A minor technical inconvenience is that constants are not integrable, so to apply previous
    lemmas we need to replace `c` with the restriction of `c` to a finite measure set `A n` in the
    above sketch. -/
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    h'f : MeasureTheory.StronglyMeasurable f
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  let A := MeasureTheory.Measure.finiteSpanningSetsInOpen' μ
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    h'f : MeasureTheory.StronglyMeasurable f
    A : μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K) := μ.finiteSpanningSetsIn …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  rcases h'f.isSeparable_range with ⟨t, t_count, ht⟩
  have main :
    ∀ᵐ x ∂μ,
      ∀ᵉ (n : ℕ) (c ∈ t),
        Tendsto (fun a => (∫⁻ y in a, ‖f y - (A.set n).indicator (fun _ => c) y‖₊ ∂μ) / μ a)
          (v.filterAt x) (𝓝 ‖f x - (A.set n).indicator (fun _ => c) x‖₊) := by
    #adaptation_note /-- 2024-04-23
    The next two lines were previously just `simp_rw [ae_all_iff, ae_ball_iff t_count]`. -/
    simp_rw [ae_all_iff]
    intro x; rw [ae_ball_iff t_count]; revert x
    intro n c _
    apply ae_tendsto_lintegral_div'
    · refine (h'f.sub ?_).ennnorm
      exact stronglyMeasurable_const.indicator (IsOpen.measurableSet (A.set_mem n))
    · apply ne_of_lt
      calc
        (∫⁻ y, ↑‖f y - (A.set n).indicator (fun _ : α => c) y‖₊ ∂μ) ≤
            ∫⁻ y, ‖f y‖₊ + ‖(A.set n).indicator (fun _ : α => c) y‖₊ ∂μ := by
          apply lintegral_mono
          intro x
          dsimp
          rw [← ENNReal.coe_add]
          exact ENNReal.coe_le_coe.2 (nnnorm_sub_le _ _)
        _ = (∫⁻ y, ‖f y‖₊ ∂μ) + ∫⁻ y, ‖(A.set n).indicator (fun _ : α => c) y‖₊ ∂μ :=
          (lintegral_add_left h'f.ennnorm _)
        _ < ∞ + ∞ :=
          haveI I : Integrable ((A.set n).indicator fun _ : α => c) μ := by
            simp only [integrable_indicator_iff (IsOpen.measurableSet (A.set_mem n)),
              integrableOn_const, A.finite n, or_true]
          ENNReal.add_lt_add hf.2 I.2
  /-
    case intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    h'f : MeasureTheory.StronglyMeasurable f
    A : μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K) := μ.finiteSpanningSetsIn …
    t : Set E
    t_count : t.Countable
    ht : HasSubset.Subset (Set.range f) (closure t)
    main : Filter.Eventually (fun x => ∀ (n : Nat) (c : E), Membership.mem t c → F …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  filter_upwards [main, v.ae_eventually_measure_pos] with x hx h'x
  have M :
    ∀ c ∈ t, Tendsto (fun a => (∫⁻ y in a, ‖f y - c‖₊ ∂μ) / μ a)
      (v.filterAt x) (𝓝 ‖f x - c‖₊) := by
    intro c hc
    obtain ⟨n, xn⟩ : ∃ n, x ∈ A.set n := by simpa [← A.spanning] using mem_univ x
    specialize hx n c hc
    simp only [xn, indicator_of_mem] at hx
    apply hx.congr' _
    filter_upwards [v.eventually_filterAt_subset_of_nhds (IsOpen.mem_nhds (A.set_mem n) xn),
      v.eventually_filterAt_measurableSet x] with a ha h'a
    congr 1
    apply setLIntegral_congr_fun h'a
    filter_upwards with y hy using (by simp only [ha hy, indicator_of_mem])
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    h'f : MeasureTheory.StronglyMeasurable f
    A : μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K) := μ.finiteSpanningSetsIn …
    t : Set E
    t_count : t.Countable
    ht : HasSubset.Subset (Set.range f) (closure t)
    main : Filter.Eventually (fun x => ∀ (n : Nat) (c : E), Membership.mem t c → F …
    x : α
    hx : ∀ (n : Nat) (c : E), Membership.mem t c → Filter.Tendsto (fun a => HDiv.h …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    M : ∀ (c : E), Membership.mem t c → Filter.Tendsto (fun a => HDiv.hDiv (Measur …
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  apply ENNReal.tendsto_nhds_zero.2 fun ε εpos => ?_
  obtain ⟨c, ct, xc⟩ : ∃ c ∈ t, (‖f x - c‖₊ : ℝ≥0∞) < ε / 2 := by
    simp_rw [← edist_eq_coe_nnnorm_sub]
    have : f x ∈ closure t := ht (mem_range_self _)
    exact EMetric.mem_closure_iff.1 this (ε / 2) (ENNReal.half_pos (ne_of_gt εpos))
  filter_upwards [(tendsto_order.1 (M c ct)).2 (ε / 2) xc, h'x, v.eventually_measure_lt_top x] with
    a ha h'a h''a
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    h'f : MeasureTheory.StronglyMeasurable f
    A : μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K) := μ.finiteSpanningSetsIn …
    t : Set E
    t_count : t.Countable
    ht : HasSubset.Subset (Set.range f) (closure t)
    main : Filter.Eventually (fun x => ∀ (n : Nat) (c : E), Membership.mem t c → F …
    x : α
    hx : ∀ (n : Nat) (c : E), Membership.mem t c → Filter.Tendsto (fun a => HDiv.h …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    M : ∀ (c : E), Membership.mem t c → Filter.Tendsto (fun a => HDiv.hDiv (Measur …
    ε : ENNReal
    εpos : GT.gt ε 0
    c : E
    ct : Membership.mem t c
    xc : LT.lt (↑(NNNorm.nnnorm (HSub.hSub (f x) c))) (HDiv.hDiv ε 2)
    a : Set α
    ha : LT.lt (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNor …
    h'a : LT.lt 0 (μ a)
    h''a : LT.lt (μ a) Top.top
    ⊢ LE.le (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNorm.n …
  -/
  apply ENNReal.div_le_of_le_mul
  calc
    (∫⁻ y in a, ‖f y - f x‖₊ ∂μ) ≤ ∫⁻ y in a, ‖f y - c‖₊ + ‖f x - c‖₊ ∂μ := by
      apply lintegral_mono fun x => ?_
      simpa only [← edist_eq_coe_nnnorm_sub] using edist_triangle_right _ _ _
    _ = (∫⁻ y in a, ‖f y - c‖₊ ∂μ) + ∫⁻ _ in a, ‖f x - c‖₊ ∂μ :=
      (lintegral_add_right _ measurable_const)
    _ ≤ ε / 2 * μ a + ε / 2 * μ a := by
      gcongr
      · rw [ENNReal.div_lt_iff (Or.inl h'a.ne') (Or.inl h''a.ne)] at ha
        exact ha.le
      · simp only [lintegral_const, Measure.restrict_apply, MeasurableSet.univ, univ_inter]
        gcongr
    _ = ε * μ a := by rw [← add_mul, ENNReal.add_halves]


theorem ae_tendsto_lintegral_nnnorm_sub_div_of_integrable {f : α → E} (hf : Integrable f μ) :
    ∀ᵐ x ∂μ, Tendsto (fun a => (∫⁻ y in a, ‖f y - f x‖₊ ∂μ) / μ a) (v.filterAt x) (𝓝 0) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  have I : Integrable (hf.1.mk f) μ := hf.congr hf.1.ae_eq_mk
  filter_upwards [v.ae_tendsto_lintegral_nnnorm_sub_div'_of_integrable I hf.1.stronglyMeasurable_mk,
    hf.1.ae_eq_mk] with x hx h'x
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  apply hx.congr _
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    ⊢ ∀ (x_1 : Set α), Eq (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict x_1) fun …
  -/
  intro a
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    a : Set α
    ⊢ Eq (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNorm.nnno …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    a : Set α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNorm.nnnorm (HSub.hS …
  -/
  apply lintegral_congr_ae
  /-
    case e_a.h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    a : Set α
    ⊢ (MeasureTheory.ae (μ.restrict a)).EventuallyEq (fun a => ↑(NNNorm.nnnorm (HS …
  -/
  apply ae_restrict_of_ae
  /-
    case e_a.h.h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    a : Set α
    ⊢ Filter.Eventually (fun x_1 => Eq ((fun a => ↑(NNNorm.nnnorm (HSub.hSub (Meas …
  -/
  filter_upwards [hf.1.ae_eq_mk] with y hy
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.Integrable f μ
    I : MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable.mk f ⋯) μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    h'x : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    a : Set α
    y : α
    hy : Eq (f y) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ y)
    ⊢ Eq ↑(NNNorm.nnnorm (HSub.hSub (MeasureTheory.AEStronglyMeasurable.mk f ⋯ y)  …
  -/
  rw [hy, h'x]
  /-
    🎉 no goals
  -/


theorem ae_tendsto_lintegral_nnnorm_sub_div {f : α → E} (hf : LocallyIntegrable f μ) :
    ∀ᵐ x ∂μ, Tendsto (fun a => (∫⁻ y in a, ‖f y - f x‖₊ ∂μ) / μ a) (v.filterAt x) (𝓝 0) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  rcases hf.exists_nat_integrableOn with ⟨u, u_open, u_univ, hu⟩
  have : ∀ n, ∀ᵐ x ∂μ,
      Tendsto (fun a => (∫⁻ y in a, ‖(u n).indicator f y - (u n).indicator f x‖₊ ∂μ) / μ a)
      (v.filterAt x) (𝓝 0) := by
    intro n
    apply ae_tendsto_lintegral_nnnorm_sub_div_of_integrable
    exact (integrable_indicator_iff (u_open n).measurableSet).2 (hu n)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheor …
  -/
  filter_upwards [ae_all_iff.2 this] with x hx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    x : α
    hx : ∀ (i : Nat), Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral  …
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  obtain ⟨n, hn⟩ : ∃ n, x ∈ u n := by simpa only [← u_univ, mem_iUnion] using mem_univ x
  /-
    case h.intro
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    x : α
    hx : ∀ (i : Nat), Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral  …
    n : Nat
    hn : Membership.mem (u n) x
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) f …
  -/
  apply Tendsto.congr' _ (hx n)
  filter_upwards [v.eventually_filterAt_subset_of_nhds ((u_open n).mem_nhds hn),
    v.eventually_filterAt_measurableSet x] with a ha h'a
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    x : α
    hx : ∀ (i : Nat), Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral  …
    n : Nat
    hn : Membership.mem (u n) x
    a : Set α
    ha : HasSubset.Subset a (u n)
    h'a : MeasurableSet a
    ⊢ Eq (HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNorm.nnno …
  -/
  congr 1
  /-
    case h.e_a
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    x : α
    hx : ∀ (i : Nat), Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral  …
    n : Nat
    hn : Membership.mem (u n) x
    a : Set α
    ha : HasSubset.Subset a (u n)
    h'a : MeasurableSet a
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict a) fun y => ↑(NNNorm.nnnorm (HSub.hS …
  -/
  refine setLIntegral_congr_fun h'a (Eventually.of_forall (fun y hy ↦ ?_))
  /-
    case h.e_a
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set α
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_univ : Eq (Set.iUnion fun n => u n) Set.univ
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (u n) μ
    this : ∀ (n : Nat), Filter.Eventually (fun x => Filter.Tendsto (fun a => HDiv. …
    x : α
    hx : ∀ (i : Nat), Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral  …
    n : Nat
    hn : Membership.mem (u n) x
    a : Set α
    ha : HasSubset.Subset a (u n)
    h'a : MeasurableSet a
    y : α
    hy : Membership.mem a y
    ⊢ Eq ↑(NNNorm.nnnorm (HSub.hSub ((u n).indicator f y) ((u n).indicator f x)))  …
  -/
  rw [indicator_of_mem (ha hy) f, indicator_of_mem hn f]
  /-
    🎉 no goals
  -/


/-- *Lebesgue differentiation theorem*: for almost every point `x`, the
average of `‖f y - f x‖` on `a` tends to `0` as `a` shrinks to `x` along a Vitali family. -/
theorem ae_tendsto_average_norm_sub {f : α → E} (hf : LocallyIntegrable f μ) :
    ∀ᵐ x ∂μ, Tendsto (fun a => ⨍ y in a, ‖f y - f x‖ ∂μ) (v.filterAt x) (𝓝 0) := by
  /-
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => MeasureTheory.average ( …
  -/
  filter_upwards [v.ae_tendsto_lintegral_nnnorm_sub_div hf] with x hx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    ⊢ Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => Norm. …
  -/
  have := (ENNReal.tendsto_toReal ENNReal.zero_ne_top).comp hx
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    this : Filter.Tendsto (Function.comp ENNReal.toReal fun a => HDiv.hDiv (Measur …
    ⊢ Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => Norm. …
  -/
  simp only [ENNReal.zero_toReal] at this
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    this : Filter.Tendsto (Function.comp ENNReal.toReal fun a => HDiv.hDiv (Measur …
    ⊢ Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => Norm. …
  -/
  apply Tendsto.congr' _ this
  filter_upwards [v.eventually_measure_lt_top x, v.eventually_filterAt_integrableOn x hf]
    with a h'a h''a
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    this : Filter.Tendsto (Function.comp ENNReal.toReal fun a => HDiv.hDiv (Measur …
    a : Set α
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    ⊢ Eq (Function.comp ENNReal.toReal (fun a => HDiv.hDiv (MeasureTheory.lintegra …
  -/
  simp only [Function.comp_apply, ENNReal.toReal_div, setAverage_eq, div_eq_inv_mul]
  have A : IntegrableOn (fun y => (‖f y - f x‖₊ : ℝ)) a μ := by
    simp_rw [coe_nnnorm]
    exact (h''a.sub (integrableOn_const.2 (Or.inr h'a))).norm
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    this : Filter.Tendsto (Function.comp ENNReal.toReal fun a => HDiv.hDiv (Measur …
    a : Set α
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    A : MeasureTheory.IntegrableOn (fun y => ↑(NNNorm.nnnorm (HSub.hSub (f y) (f x …
    ⊢ Eq (HMul.hMul (Inv.inv (μ a).toReal) (MeasureTheory.lintegral (μ.restrict a) …
  -/
  rw [lintegral_coe_eq_integral _ A, ENNReal.toReal_ofReal (by positivity)]
  /-
    case h
    α : Type u_1
    inst✝⁴ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : SecondCountableTopology α
    inst✝¹ : BorelSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => HDiv.hDiv (MeasureTheory.lintegral (μ.restrict a …
    this : Filter.Tendsto (Function.comp ENNReal.toReal fun a => HDiv.hDiv (Measur …
    a : Set α
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    A : MeasureTheory.IntegrableOn (fun y => ↑(NNNorm.nnnorm (HSub.hSub (f y) (f x …
    ⊢ Eq (HMul.hMul (Inv.inv (μ a).toReal) (MeasureTheory.integral (μ.restrict a)  …
  -/
  simp only [coe_nnnorm, smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- *Lebesgue differentiation theorem*: for almost every point `x`, the
average of `f` on `a` tends to `f x` as `a` shrinks to `x` along a Vitali family. -/
theorem ae_tendsto_average [NormedSpace ℝ E] [CompleteSpace E] {f : α → E}
    (hf : LocallyIntegrable f μ) :
    ∀ᵐ x ∂μ, Tendsto (fun a => ⨍ y in a, f y ∂μ) (v.filterAt x) (𝓝 (f x)) := by
  /-
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun a => MeasureTheory.average ( …
  -/
  filter_upwards [v.ae_tendsto_average_norm_sub hf, v.ae_eventually_measure_pos] with x hx h'x
  /-
    case h
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    ⊢ Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => f y)  …
  -/
  rw [tendsto_iff_norm_sub_tendsto_zero]
  /-
    case h
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (MeasureTheory.average (μ.rest …
  -/
  refine squeeze_zero' (Eventually.of_forall fun a => norm_nonneg _) ?_ hx
  filter_upwards [h'x, v.eventually_measure_lt_top x, v.eventually_filterAt_integrableOn x hf]
    with a ha h'a h''a
  /-
    case h
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    a : Set α
    ha : LT.lt 0 (μ a)
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    ⊢ LE.le (Norm.norm (HSub.hSub (MeasureTheory.average (μ.restrict a) fun y => f …
  -/
  nth_rw 1 [← setAverage_const ha.ne' h'a.ne (f x)]
  /-
    case h
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    a : Set α
    ha : LT.lt 0 (μ a)
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    ⊢ LE.le (Norm.norm (HSub.hSub (MeasureTheory.average (μ.restrict a) fun y => f …
  -/
  simp_rw [setAverage_eq']
  /-
    case h
    α : Type u_1
    inst✝⁶ : PseudoMetricSpace α
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    v : VitaliFamily μ
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : BorelSpace α
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.LocallyIntegrable f μ
    x : α
    hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
    a : Set α
    ha : LT.lt 0 (μ a)
    h'a : LT.lt (μ a) Top.top
    h''a : MeasureTheory.IntegrableOn f a μ
    ⊢ LE.le (Norm.norm (HSub.hSub (MeasureTheory.integral (HSMul.hSMul (Inv.inv (μ …
  -/
  rw [← integral_sub]
    /-
      case h
      α : Type u_1
      inst✝⁶ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : BorelSpace α
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.LocallyIntegrable f μ
      x : α
      hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
      a : Set α
      ha : LT.lt 0 (μ a)
      h'a : LT.lt (μ a) Top.top
      h''a : MeasureTheory.IntegrableOn f a μ
      ⊢ LE.le (Norm.norm (MeasureTheory.integral (HSMul.hSMul (Inv.inv (μ a)) (μ.res …
    -/
  · exact norm_integral_le_integral_norm _
    /-
      🎉 no goals
    -/
    /-
      case h.hf
      α : Type u_1
      inst✝⁶ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : BorelSpace α
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.LocallyIntegrable f μ
      x : α
      hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
      a : Set α
      ha : LT.lt 0 (μ a)
      h'a : LT.lt (μ a) Top.top
      h''a : MeasureTheory.IntegrableOn f a μ
      ⊢ MeasureTheory.Integrable f (HSMul.hSMul (Inv.inv (μ a)) (μ.restrict a))
    -/
  · exact (integrable_inv_smul_measure ha.ne' h'a.ne).2 h''a
    /-
      🎉 no goals
    -/
    /-
      case h.hg
      α : Type u_1
      inst✝⁶ : PseudoMetricSpace α
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      v : VitaliFamily μ
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : BorelSpace α
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.LocallyIntegrable f μ
      x : α
      hx : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      h'x : Filter.Eventually (fun a => LT.lt 0 (μ a)) (v.filterAt x)
      a : Set α
      ha : LT.lt 0 (μ a)
      h'a : LT.lt (μ a) Top.top
      h''a : MeasureTheory.IntegrableOn f a μ
      ⊢ MeasureTheory.Integrable (fun x_1 => f x) (HSMul.hSMul (Inv.inv (μ a)) (μ.re …
    -/
  · exact (integrable_inv_smul_measure ha.ne' h'a.ne).2 (integrableOn_const.2 (Or.inr h'a))
    /-
      🎉 no goals
    -/


