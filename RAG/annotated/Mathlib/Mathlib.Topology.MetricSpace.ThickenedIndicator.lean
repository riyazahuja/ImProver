/-- The `δ`-thickened indicator of a set `E` is the function that equals `1` on `E`
and `0` outside a `δ`-thickening of `E` and interpolates (continuously) between
these values using `infEdist _ E`.

`thickenedIndicatorAux` is the unbundled `ℝ≥0∞`-valued function. See `thickenedIndicator`
for the (bundled) bounded continuous function with `ℝ≥0`-values. -/
def thickenedIndicatorAux (δ : ℝ) (E : Set α) : α → ℝ≥0∞ :=
  fun x : α => (1 : ℝ≥0∞) - infEdist x E / ENNReal.ofReal δ


theorem continuous_thickenedIndicatorAux {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) :
    Continuous (thickenedIndicatorAux δ E) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    ⊢ Continuous (thickenedIndicatorAux δ E)
  -/
  unfold thickenedIndicatorAux
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    ⊢ Continuous fun x => HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.o …
  -/
  let f := fun x : α => (⟨1, infEdist x E / ENNReal.ofReal δ⟩ : ℝ≥0 × ℝ≥0∞)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    f : α → Prod NNReal ENNReal := fun x => { fst := 1, snd := HDiv.hDiv (EMetric. …
    ⊢ Continuous fun x => HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.o …
  -/
  let sub := fun p : ℝ≥0 × ℝ≥0∞ => (p.1 : ℝ≥0∞) - p.2
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    f : α → Prod NNReal ENNReal := fun x => { fst := 1, snd := HDiv.hDiv (EMetric. …
    sub : Prod NNReal ENNReal → ENNReal := fun p => HSub.hSub (↑p.1) p.2
    ⊢ Continuous fun x => HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.o …
  -/
  rw [show (fun x : α => (1 : ℝ≥0∞) - infEdist x E / ENNReal.ofReal δ) = sub ∘ f by rfl]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    f : α → Prod NNReal ENNReal := fun x => { fst := 1, snd := HDiv.hDiv (EMetric. …
    sub : Prod NNReal ENNReal → ENNReal := fun p => HSub.hSub (↑p.1) p.2
    ⊢ Continuous (Function.comp sub f)
  -/
  apply (@ENNReal.continuous_nnreal_sub 1).comp
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    f : α → Prod NNReal ENNReal := fun x => { fst := 1, snd := HDiv.hDiv (EMetric. …
    sub : Prod NNReal ENNReal → ENNReal := fun p => HSub.hSub (↑p.1) p.2
    ⊢ Continuous fun x => (f x).2
  -/
  apply (ENNReal.continuous_div_const (ENNReal.ofReal δ) _).comp continuous_infEdist
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    f : α → Prod NNReal ENNReal := fun x => { fst := 1, snd := HDiv.hDiv (EMetric. …
    sub : Prod NNReal ENNReal → ENNReal := fun p => HSub.hSub (↑p.1) p.2
    ⊢ Ne (ENNReal.ofReal δ) 0
  -/
  norm_num [δ_pos]
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_le_one (δ : ℝ) (E : Set α) (x : α) :
    thickenedIndicatorAux δ E x ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    ⊢ LE.le (thickenedIndicatorAux δ E x) 1
  -/
  apply @tsub_le_self _ _ _ _ (1 : ℝ≥0∞)
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_lt_top {δ : ℝ} {E : Set α} {x : α} :
    thickenedIndicatorAux δ E x < ∞ :=
  lt_of_le_of_lt (thickenedIndicatorAux_le_one _ _ _) one_lt_top


theorem thickenedIndicatorAux_closure_eq (δ : ℝ) (E : Set α) :
    thickenedIndicatorAux δ (closure E) = thickenedIndicatorAux δ E := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ Eq (thickenedIndicatorAux δ (closure E)) (thickenedIndicatorAux δ E)
  -/
  simp (config := { unfoldPartialApp := true }) only [thickenedIndicatorAux, infEdist_closure]
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_one (δ : ℝ) (E : Set α) {x : α} (x_in_E : x ∈ E) :
    thickenedIndicatorAux δ E x = 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_in_E : Membership.mem E x
    ⊢ Eq (thickenedIndicatorAux δ E x) 1
  -/
  simp [thickenedIndicatorAux, infEdist_zero_of_mem x_in_E, tsub_zero]
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_one_of_mem_closure (δ : ℝ) (E : Set α) {x : α}
    (x_mem : x ∈ closure E) : thickenedIndicatorAux δ E x = 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_mem : Membership.mem (closure E) x
    ⊢ Eq (thickenedIndicatorAux δ E x) 1
  -/
  rw [← thickenedIndicatorAux_closure_eq, thickenedIndicatorAux_one δ (closure E) x_mem]
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_zero {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) {x : α}
    (x_out : x ∉ thickening δ E) : thickenedIndicatorAux δ E x = 0 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : Not (Membership.mem (Metric.thickening δ E) x)
    ⊢ Eq (thickenedIndicatorAux δ E x) 0
  -/
  rw [thickening, mem_setOf_eq, not_lt] at x_out
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x E)
    ⊢ Eq (thickenedIndicatorAux δ E x) 0
  -/
  unfold thickenedIndicatorAux
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x E)
    ⊢ Eq (HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.ofReal δ))) 0
  -/
  apply le_antisymm _ bot_le
  have key := tsub_le_tsub
    (@rfl _ (1 : ℝ≥0∞)).le (ENNReal.div_le_div x_out (@rfl _ (ENNReal.ofReal δ : ℝ≥0∞)).le)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x E)
    key : LE.le (HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.ofReal δ)) …
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.ofReal δ))) Bo …
  -/
  rw [ENNReal.div_self (ne_of_gt (ENNReal.ofReal_pos.mpr δ_pos)) ofReal_ne_top] at key
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x E)
    key : LE.le (HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.ofReal δ)) …
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (EMetric.infEdist x E) (ENNReal.ofReal δ))) Bo …
  -/
  simpa [tsub_self] using key
  /-
    🎉 no goals
  -/


theorem thickenedIndicatorAux_mono {δ₁ δ₂ : ℝ} (hle : δ₁ ≤ δ₂) (E : Set α) :
    thickenedIndicatorAux δ₁ E ≤ thickenedIndicatorAux δ₂ E :=
  fun _ => tsub_le_tsub (@rfl ℝ≥0∞ 1).le (ENNReal.div_le_div rfl.le (ofReal_le_ofReal hle))


theorem indicator_le_thickenedIndicatorAux (δ : ℝ) (E : Set α) :
    (E.indicator fun _ => (1 : ℝ≥0∞)) ≤ thickenedIndicatorAux δ E := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ LE.le (E.indicator fun x => 1) (thickenedIndicatorAux δ E)
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    a : α
    ⊢ LE.le (E.indicator (fun x => 1) a) (thickenedIndicatorAux δ E a)
  -/
  by_cases h : a ∈ E
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      E : Set α
      a : α
      h : Membership.mem E a
      ⊢ LE.le (E.indicator (fun x => 1) a) (thickenedIndicatorAux δ E a)
    -/
  · simp only [h, indicator_of_mem, thickenedIndicatorAux_one δ E h, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      E : Set α
      a : α
      h : Not (Membership.mem E a)
      ⊢ LE.le (E.indicator (fun x => 1) a) (thickenedIndicatorAux δ E a)
    -/
  · simp only [h, indicator_of_not_mem, not_false_iff, zero_le]
    /-
      🎉 no goals
    -/


theorem thickenedIndicatorAux_subset (δ : ℝ) {E₁ E₂ : Set α} (subset : E₁ ⊆ E₂) :
    thickenedIndicatorAux δ E₁ ≤ thickenedIndicatorAux δ E₂ :=
  fun _ => tsub_le_tsub (@rfl ℝ≥0∞ 1).le (ENNReal.div_le_div (infEdist_anti subset) rfl.le)


/-- As the thickening radius δ tends to 0, the δ-thickened indicator of a set E (in α) tends
pointwise (i.e., w.r.t. the product topology on `α → ℝ≥0∞`) to the indicator function of the
closure of E.

This statement is for the unbundled `ℝ≥0∞`-valued functions `thickenedIndicatorAux δ E`, see
`thickenedIndicator_tendsto_indicator_closure` for the version for bundled `ℝ≥0`-valued
bounded continuous functions. -/
theorem thickenedIndicatorAux_tendsto_indicator_closure {δseq : ℕ → ℝ}
    (δseq_lim : Tendsto δseq atTop (𝓝 0)) (E : Set α) :
    Tendsto (fun n => thickenedIndicatorAux (δseq n) E) atTop
      (𝓝 (indicator (closure E) fun _ => (1 : ℝ≥0∞))) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    ⊢ Filter.Tendsto (fun n => thickenedIndicatorAux (δseq n) E) Filter.atTop (nhd …
  -/
  rw [tendsto_pi_nhds]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filt …
  -/
  intro x
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    x : α
    ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
  -/
  by_cases x_mem_closure : x ∈ closure E
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
  · simp_rw [thickenedIndicatorAux_one_of_mem_closure _ E x_mem_closure]
    rw [show (indicator (closure E) fun _ => (1 : ℝ≥0∞)) x = 1 by
        simp only [x_mem_closure, indicator_of_mem]]
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      ⊢ Filter.Tendsto (fun i => 1) Filter.atTop (nhds 1)
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/
  · rw [show (closure E).indicator (fun _ => (1 : ℝ≥0∞)) x = 0 by
        simp only [x_mem_closure, indicator_of_not_mem, not_false_iff]]
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    rcases exists_real_pos_lt_infEdist_of_not_mem_closure x_mem_closure with ⟨ε, ⟨ε_pos, ε_lt⟩⟩
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    rw [Metric.tendsto_nhds] at δseq_lim
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      δseq_lim : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.d …
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    specialize δseq_lim ε ε_pos
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      δseq_lim : Filter.Eventually (fun x => LT.lt (Dist.dist (δseq x) 0) ε) Filter. …
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    simp only [dist_zero_right, Real.norm_eq_abs, eventually_atTop] at δseq_lim
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      δseq_lim : Exists fun a => ∀ (b : Nat), GE.ge b a → LT.lt (abs (δseq b)) ε
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    rcases δseq_lim with ⟨N, hN⟩
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      ⊢ Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x) Filter.atTop (n …
    -/
    apply tendsto_atTop_of_eventually_const (i₀ := N)
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      ⊢ ∀ (i : Nat), GE.ge i N → Eq (thickenedIndicatorAux (δseq i) E x) 0
    -/
    intro n n_large
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      n : Nat
      n_large : GE.ge n N
      ⊢ Eq (thickenedIndicatorAux (δseq n) E x) 0
    -/
    have key : x ∉ thickening ε E := by simpa only [thickening, mem_setOf_eq, not_lt] using ε_lt.le
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      n : Nat
      n_large : GE.ge n N
      key : Not (Membership.mem (Metric.thickening ε E) x)
      ⊢ Eq (thickenedIndicatorAux (δseq n) E x) 0
    -/
    refine le_antisymm ?_ bot_le
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      n : Nat
      n_large : GE.ge n N
      key : Not (Membership.mem (Metric.thickening ε E) x)
      ⊢ LE.le (thickenedIndicatorAux (δseq n) E x) 0
    -/
    apply (thickenedIndicatorAux_mono (lt_of_abs_lt (hN n n_large)).le E x).trans
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δseq : Nat → Real
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ε : Real
      ε_pos : LT.lt 0 ε
      ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LT.lt (abs (δseq b)) ε
      n : Nat
      n_large : GE.ge n N
      key : Not (Membership.mem (Metric.thickening ε E) x)
      ⊢ LE.le (thickenedIndicatorAux ε E x) 0
    -/
    exact (thickenedIndicatorAux_zero ε_pos E key).le
    /-
      🎉 no goals
    -/


/-- The `δ`-thickened indicator of a set `E` is the function that equals `1` on `E`
and `0` outside a `δ`-thickening of `E` and interpolates (continuously) between
these values using `infEdist _ E`.

`thickenedIndicator` is the (bundled) bounded continuous function with `ℝ≥0`-values.
See `thickenedIndicatorAux` for the unbundled `ℝ≥0∞`-valued function. -/
@[simps]
def thickenedIndicator {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) : α →ᵇ ℝ≥0 where
  toFun := fun x : α => (thickenedIndicatorAux δ E x).toNNReal
  continuous_toFun := by
    apply ContinuousOn.comp_continuous continuousOn_toNNReal
      (continuous_thickenedIndicatorAux δ_pos E)
    /-
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      ⊢ ∀ (x : α), Membership.mem (setOf fun a => Ne a Top.top) (thickenedIndicatorA …
    -/
    intro x
    /-
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x : α
      ⊢ Membership.mem (setOf fun a => Ne a Top.top) (thickenedIndicatorAux δ E x)
    -/
    exact (lt_of_le_of_lt (@thickenedIndicatorAux_le_one _ _ δ E x) one_lt_top).ne
    /-
      🎉 no goals
    -/
  map_bounded' := by
    /-
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := fun x => (thickene …
    -/
    use 2
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      ⊢ ∀ (x y : α), LE.le (Dist.dist ({ toFun := fun x => (thickenedIndicatorAux δ  …
    -/
    intro x y
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x y : α
      ⊢ LE.le (Dist.dist ({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNReal …
    -/
    rw [NNReal.dist_eq]
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x y : α
      ⊢ LE.le (abs (HSub.hSub ↑({ toFun := fun x => (thickenedIndicatorAux δ E x).to …
    -/
    apply (abs_sub _ _).trans
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x y : α
      ⊢ LE.le (HAdd.hAdd (abs ↑({ toFun := fun x => (thickenedIndicatorAux δ E x).to …
    -/
    rw [NNReal.abs_eq, NNReal.abs_eq, ← one_add_one_eq_two]
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x y : α
      ⊢ LE.le (HAdd.hAdd ↑({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNRea …
    -/
    have key := @thickenedIndicatorAux_le_one _ _ δ E
    /-
      case h
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      x y : α
      key : ∀ (x : α), LE.le (thickenedIndicatorAux δ E x) 1
      ⊢ LE.le (HAdd.hAdd ↑({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNRea …
    -/
    apply add_le_add <;>
        /-
          case h.h₁
          α : Type u_1
          inst✝ : PseudoEMetricSpace α
          δ : Real
          δ_pos : LT.lt 0 δ
          E : Set α
          x y : α
          key : ∀ (x : α), LE.le (thickenedIndicatorAux δ E x) 1
          ⊢ LE.le (↑({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNReal, continu …
        -/
        /-
          case h.h₁
          α : Type u_1
          inst✝ : PseudoEMetricSpace α
          δ : Real
          δ_pos : LT.lt 0 δ
          E : Set α
          x y : α
          key : ∀ (x : α), LE.le (thickenedIndicatorAux δ E x) 1
          ⊢ LE.le ({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNReal, continuou …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.h₂
          α : Type u_1
          inst✝ : PseudoEMetricSpace α
          δ : Real
          δ_pos : LT.lt 0 δ
          E : Set α
          x y : α
          key : ∀ (x : α), LE.le (thickenedIndicatorAux δ E x) 1
          ⊢ LE.le ({ toFun := fun x => (thickenedIndicatorAux δ E x).toNNReal, continuou …
        -/
        exact (toNNReal_le_toNNReal (lt_of_le_of_lt (key _) one_lt_top).ne one_ne_top).mpr (key _)
        /-
          🎉 no goals
        -/


theorem thickenedIndicator.coeFn_eq_comp {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) :
    ⇑(thickenedIndicator δ_pos E) = ENNReal.toNNReal ∘ thickenedIndicatorAux δ E :=
  rfl


theorem thickenedIndicator_le_one {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) (x : α) :
    thickenedIndicator δ_pos E x ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    ⊢ LE.le ((thickenedIndicator δ_pos E) x) 1
  -/
  rw [thickenedIndicator.coeFn_eq_comp]
  simpa using (toNNReal_le_toNNReal thickenedIndicatorAux_lt_top.ne one_ne_top).mpr
    (thickenedIndicatorAux_le_one δ E x)


theorem thickenedIndicator_one_of_mem_closure {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) {x : α}
    (x_mem : x ∈ closure E) : thickenedIndicator δ_pos E x = 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_mem : Membership.mem (closure E) x
    ⊢ Eq ((thickenedIndicator δ_pos E) x) 1
  -/
  rw [thickenedIndicator_apply, thickenedIndicatorAux_one_of_mem_closure δ E x_mem, one_toNNReal]
  /-
    🎉 no goals
  -/


lemma one_le_thickenedIndicator_apply' {X : Type _} [PseudoEMetricSpace X]
    {δ : ℝ} (δ_pos : 0 < δ) {F : Set X} {x : X} (hxF : x ∈ closure F) :
    1 ≤ thickenedIndicator δ_pos F x := by
  /-
    X : Type u_2
    inst✝ : PseudoEMetricSpace X
    δ : Real
    δ_pos : LT.lt 0 δ
    F : Set X
    x : X
    hxF : Membership.mem (closure F) x
    ⊢ LE.le 1 ((thickenedIndicator δ_pos F) x)
  -/
  rw [thickenedIndicator_one_of_mem_closure δ_pos F hxF]
  /-
    🎉 no goals
  -/


lemma one_le_thickenedIndicator_apply (X : Type _) [PseudoEMetricSpace X]
    {δ : ℝ} (δ_pos : 0 < δ) {F : Set X} {x : X} (hxF : x ∈ F) :
    1 ≤ thickenedIndicator δ_pos F x :=
  one_le_thickenedIndicator_apply' δ_pos (subset_closure hxF)


theorem thickenedIndicator_one {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) {x : α} (x_in_E : x ∈ E) :
    thickenedIndicator δ_pos E x = 1 :=
  thickenedIndicator_one_of_mem_closure _ _ (subset_closure x_in_E)


theorem thickenedIndicator_zero {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) {x : α}
    (x_out : x ∉ thickening δ E) : thickenedIndicator δ_pos E x = 0 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    x : α
    x_out : Not (Membership.mem (Metric.thickening δ E) x)
    ⊢ Eq ((thickenedIndicator δ_pos E) x) 0
  -/
  rw [thickenedIndicator_apply, thickenedIndicatorAux_zero δ_pos E x_out, zero_toNNReal]
  /-
    🎉 no goals
  -/


theorem indicator_le_thickenedIndicator {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) :
    (E.indicator fun _ => (1 : ℝ≥0)) ≤ thickenedIndicator δ_pos E := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    ⊢ LE.le (E.indicator fun x => 1) ⇑(thickenedIndicator δ_pos E)
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    a : α
    ⊢ LE.le (E.indicator (fun x => 1) a) ((thickenedIndicator δ_pos E) a)
  -/
  by_cases h : a ∈ E
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      a : α
      h : Membership.mem E a
      ⊢ LE.le (E.indicator (fun x => 1) a) ((thickenedIndicator δ_pos E) a)
    -/
  · simp only [h, indicator_of_mem, thickenedIndicator_one δ_pos E h, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_pos : LT.lt 0 δ
      E : Set α
      a : α
      h : Not (Membership.mem E a)
      ⊢ LE.le (E.indicator (fun x => 1) a) ((thickenedIndicator δ_pos E) a)
    -/
  · simp only [h, indicator_of_not_mem, not_false_iff, zero_le]
    /-
      🎉 no goals
    -/


theorem thickenedIndicator_mono {δ₁ δ₂ : ℝ} (δ₁_pos : 0 < δ₁) (δ₂_pos : 0 < δ₂) (hle : δ₁ ≤ δ₂)
    (E : Set α) : ⇑(thickenedIndicator δ₁_pos E) ≤ thickenedIndicator δ₂_pos E := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ₁ δ₂ : Real
    δ₁_pos : LT.lt 0 δ₁
    δ₂_pos : LT.lt 0 δ₂
    hle : LE.le δ₁ δ₂
    E : Set α
    ⊢ LE.le ⇑(thickenedIndicator δ₁_pos E) ⇑(thickenedIndicator δ₂_pos E)
  -/
  intro x
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ₁ δ₂ : Real
    δ₁_pos : LT.lt 0 δ₁
    δ₂_pos : LT.lt 0 δ₂
    hle : LE.le δ₁ δ₂
    E : Set α
    x : α
    ⊢ LE.le ((thickenedIndicator δ₁_pos E) x) ((thickenedIndicator δ₂_pos E) x)
  -/
  apply (toNNReal_le_toNNReal thickenedIndicatorAux_lt_top.ne thickenedIndicatorAux_lt_top.ne).mpr
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δ₁ δ₂ : Real
    δ₁_pos : LT.lt 0 δ₁
    δ₂_pos : LT.lt 0 δ₂
    hle : LE.le δ₁ δ₂
    E : Set α
    x : α
    ⊢ LE.le (thickenedIndicatorAux δ₁ E x) (thickenedIndicatorAux δ₂ E x)
  -/
  apply thickenedIndicatorAux_mono hle
  /-
    🎉 no goals
  -/


theorem thickenedIndicator_subset {δ : ℝ} (δ_pos : 0 < δ) {E₁ E₂ : Set α} (subset : E₁ ⊆ E₂) :
    ⇑(thickenedIndicator δ_pos E₁) ≤ thickenedIndicator δ_pos E₂ := fun x =>
  (toNNReal_le_toNNReal thickenedIndicatorAux_lt_top.ne thickenedIndicatorAux_lt_top.ne).mpr
    (thickenedIndicatorAux_subset δ subset x)


/-- As the thickening radius δ tends to 0, the δ-thickened indicator of a set E (in α) tends
pointwise to the indicator function of the closure of E.

Note: This version is for the bundled bounded continuous functions, but the topology is not
the topology on `α →ᵇ ℝ≥0`. Coercions to functions `α → ℝ≥0` are done first, so the topology
instance is the product topology (the topology of pointwise convergence). -/
theorem thickenedIndicator_tendsto_indicator_closure {δseq : ℕ → ℝ} (δseq_pos : ∀ n, 0 < δseq n)
    (δseq_lim : Tendsto δseq atTop (𝓝 0)) (E : Set α) :
    Tendsto (fun n : ℕ => ((↑) : (α →ᵇ ℝ≥0) → α → ℝ≥0) (thickenedIndicator (δseq_pos n) E)) atTop
      (𝓝 (indicator (closure E) fun _ => (1 : ℝ≥0))) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_pos : ∀ (n : Nat), LT.lt 0 (δseq n)
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    ⊢ Filter.Tendsto (fun n => ⇑(thickenedIndicator ⋯ E)) Filter.atTop (nhds ((clo …
  -/
  have key := thickenedIndicatorAux_tendsto_indicator_closure δseq_lim E
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_pos : ∀ (n : Nat), LT.lt 0 (δseq n)
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    key : Filter.Tendsto (fun n => thickenedIndicatorAux (δseq n) E) Filter.atTop  …
    ⊢ Filter.Tendsto (fun n => ⇑(thickenedIndicator ⋯ E)) Filter.atTop (nhds ((clo …
  -/
  rw [tendsto_pi_nhds] at *
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_pos : ∀ (n : Nat), LT.lt 0 (δseq n)
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    key : ∀ (x : α), Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x)  …
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => (thickenedIndicator ⋯ E) x) Filter.atTop …
  -/
  intro x
  rw [show indicator (closure E) (fun _ => (1 : ℝ≥0)) x =
        (indicator (closure E) (fun _ => (1 : ℝ≥0∞)) x).toNNReal
      by refine (congr_fun (comp_indicator_const 1 ENNReal.toNNReal zero_toNNReal) x).symm]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_pos : ∀ (n : Nat), LT.lt 0 (δseq n)
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    key : ∀ (x : α), Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x)  …
    x : α
    ⊢ Filter.Tendsto (fun i => (thickenedIndicator ⋯ E) x) Filter.atTop (nhds ((cl …
  -/
  refine Tendsto.comp (tendsto_toNNReal ?_) (key x)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    δseq : Nat → Real
    δseq_pos : ∀ (n : Nat), LT.lt 0 (δseq n)
    δseq_lim : Filter.Tendsto δseq Filter.atTop (nhds 0)
    E : Set α
    key : ∀ (x : α), Filter.Tendsto (fun i => thickenedIndicatorAux (δseq i) E x)  …
    x : α
    ⊢ Ne ((closure E).indicator (fun x => 1) x) Top.top
  -/
                                     /-
                                       🎉 no goals
                                     -/
  by_cases x_mem : x ∈ closure E <;> simp [x_mem]
                                     /-
                                       🎉 no goals
                                     -/


/-- Pointwise, the multiplicative indicators of δ-thickenings of a set eventually coincide
with the multiplicative indicator of the set as δ>0 tends to zero. -/
@[to_additive "Pointwise, the indicators of δ-thickenings of a set eventually coincide
with the indicator of the set as δ>0 tends to zero."]
lemma mulIndicator_thickening_eventually_eq_mulIndicator_closure (f : α → β) (E : Set α) (x : α) :
    ∀ᶠ δ in 𝓝[>] (0 : ℝ),
      (Metric.thickening δ E).mulIndicator f x = (closure E).mulIndicator f x := by
  /-
    α : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    β : Type u_2
    inst✝ : One β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Eventually (fun δ => Eq ((Metric.thickening δ E).mulIndicator f x) (( …
  -/
  by_cases x_mem_closure : x ∈ closure E
    /-
      case pos
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      ⊢ Filter.Eventually (fun δ => Eq ((Metric.thickening δ E).mulIndicator f x) (( …
    -/
  · filter_upwards [self_mem_nhdsWithin] with δ δ_pos
    /-
      case h
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      δ : Real
      δ_pos : Membership.mem (Set.Ioi 0) δ
      ⊢ Eq ((Metric.thickening δ E).mulIndicator f x) ((closure E).mulIndicator f x)
    -/
    simp only [closure_subset_thickening δ_pos E x_mem_closure, mulIndicator_of_mem, x_mem_closure]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ⊢ Filter.Eventually (fun δ => Eq ((Metric.thickening δ E).mulIndicator f x) (( …
    -/
  · have obs := eventually_not_mem_thickening_of_infEdist_pos x_mem_closure
    filter_upwards [mem_nhdsWithin_of_mem_nhds obs, self_mem_nhdsWithin]
      with δ x_notin_thE _
    /-
      case h
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      obs : Filter.Eventually (fun δ => Not (Membership.mem (Metric.thickening δ E)  …
      δ : Real
      x_notin_thE : Not (Membership.mem (Metric.thickening δ E) x)
      a✝ : Membership.mem (Set.Ioi 0) δ
      ⊢ Eq ((Metric.thickening δ E).mulIndicator f x) ((closure E).mulIndicator f x)
    -/
    simp only [x_notin_thE, not_false_eq_true, mulIndicator_of_not_mem, x_mem_closure]
    /-
      🎉 no goals
    -/


/-- Pointwise, the multiplicative indicators of closed δ-thickenings of a set eventually coincide
with the multiplicative indicator of the set as δ tends to zero. -/
@[to_additive "Pointwise, the indicators of closed δ-thickenings of a set eventually coincide
with the indicator of the set as δ tends to zero."]
lemma mulIndicator_cthickening_eventually_eq_mulIndicator_closure (f : α → β) (E : Set α) (x : α) :
    ∀ᶠ δ in 𝓝 (0 : ℝ),
      (Metric.cthickening δ E).mulIndicator f x = (closure E).mulIndicator f x := by
  /-
    α : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    β : Type u_2
    inst✝ : One β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Eventually (fun δ => Eq ((Metric.cthickening δ E).mulIndicator f x) ( …
  -/
  by_cases x_mem_closure : x ∈ closure E
    /-
      case pos
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      ⊢ Filter.Eventually (fun δ => Eq ((Metric.cthickening δ E).mulIndicator f x) ( …
    -/
  · filter_upwards [univ_mem] with δ _
    /-
      case h
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      δ : Real
      a✝ : Membership.mem Set.univ δ
      ⊢ Eq ((Metric.cthickening δ E).mulIndicator f x) ((closure E).mulIndicator f x)
    -/
    have obs : x ∈ cthickening δ E := closure_subset_cthickening δ E x_mem_closure
    /-
      case h
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Membership.mem (closure E) x
      δ : Real
      a✝ : Membership.mem Set.univ δ
      obs : Membership.mem (Metric.cthickening δ E) x
      ⊢ Eq ((Metric.cthickening δ E).mulIndicator f x) ((closure E).mulIndicator f x)
    -/
    rw [mulIndicator_of_mem obs f, mulIndicator_of_mem x_mem_closure f]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      ⊢ Filter.Eventually (fun δ => Eq ((Metric.cthickening δ E).mulIndicator f x) ( …
    -/
  · filter_upwards [eventually_not_mem_cthickening_of_infEdist_pos x_mem_closure] with δ hδ
    /-
      case h
      α : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      β : Type u_2
      inst✝ : One β
      f : α → β
      E : Set α
      x : α
      x_mem_closure : Not (Membership.mem (closure E) x)
      δ : Real
      hδ : Not (Membership.mem (Metric.cthickening δ E) x)
      ⊢ Eq ((Metric.cthickening δ E).mulIndicator f x) ((closure E).mulIndicator f x)
    -/
    simp only [hδ, not_false_eq_true, mulIndicator_of_not_mem, x_mem_closure]
    /-
      🎉 no goals
    -/


/-- The multiplicative indicators of δ-thickenings of a set tend pointwise to the multiplicative
indicator of the set, as δ>0 tends to zero. -/
@[to_additive "The indicators of δ-thickenings of a set tend pointwise to the indicator of the
set, as δ>0 tends to zero."]
lemma tendsto_mulIndicator_thickening_mulIndicator_closure (f : α → β) (E : Set α) :
    Tendsto (fun δ ↦ (Metric.thickening δ E).mulIndicator f) (𝓝[>] 0)
      (𝓝 ((closure E).mulIndicator f)) := by
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    ⊢ Filter.Tendsto (fun δ => (Metric.thickening δ E).mulIndicator f) (nhdsWithin …
  -/
  rw [tendsto_pi_nhds]
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => (Metric.thickening i E).mulIndicator f x …
  -/
  intro x
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Tendsto (fun i => (Metric.thickening i E).mulIndicator f x) (nhdsWith …
  -/
  rw [tendsto_congr' (mulIndicator_thickening_eventually_eq_mulIndicator_closure f E x)]
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Tendsto (fun x_1 => (closure E).mulIndicator f x) (nhdsWithin 0 (Set. …
  -/
  apply tendsto_const_nhds
  /-
    🎉 no goals
  -/


/-- The multiplicative indicators of closed δ-thickenings of a set tend pointwise to the
multiplicative indicator of the set, as δ tends to zero. -/
@[to_additive "The indicators of closed δ-thickenings of a set tend pointwise to the indicator
of the set, as δ tends to zero."]
lemma tendsto_mulIndicator_cthickening_mulIndicator_closure (f : α → β) (E : Set α) :
    Tendsto (fun δ ↦ (Metric.cthickening δ E).mulIndicator f) (𝓝 0)
      (𝓝 ((closure E).mulIndicator f)) := by
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    ⊢ Filter.Tendsto (fun δ => (Metric.cthickening δ E).mulIndicator f) (nhds 0) ( …
  -/
  rw [tendsto_pi_nhds]
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => (Metric.cthickening i E).mulIndicator f  …
  -/
  intro x
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Tendsto (fun i => (Metric.cthickening i E).mulIndicator f x) (nhds 0) …
  -/
  rw [tendsto_congr' (mulIndicator_cthickening_eventually_eq_mulIndicator_closure f E x)]
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    β : Type u_2
    inst✝¹ : One β
    inst✝ : TopologicalSpace β
    f : α → β
    E : Set α
    x : α
    ⊢ Filter.Tendsto (fun x_1 => (closure E).mulIndicator f x) (nhds 0) (nhds ((cl …
  -/
  apply tendsto_const_nhds
  /-
    🎉 no goals
  -/


