/-- The (open) `δ`-thickening `Metric.thickening δ E` of a subset `E` in a pseudo emetric space
consists of those points that are at distance less than `δ` from some point of `E`. -/
def thickening (δ : ℝ) (E : Set α) : Set α :=
  { x : α | infEdist x E < ENNReal.ofReal δ }


theorem mem_thickening_iff_infEdist_lt : x ∈ thickening δ s ↔ infEdist x s < ENNReal.ofReal δ :=
  Iff.rfl


/-- An exterior point of a subset `E` (i.e., a point outside the closure of `E`) is not in the
(open) `δ`-thickening of `E` for small enough positive `δ`. -/
lemma eventually_not_mem_thickening_of_infEdist_pos {E : Set α} {x : α} (h : x ∉ closure E) :
    ∀ᶠ δ in 𝓝 (0 : ℝ), x ∉ Metric.thickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ⊢ Filter.Eventually (fun δ => Not (Membership.mem (Metric.thickening δ E) x))  …
  -/
  obtain ⟨ε, ⟨ε_pos, ε_lt⟩⟩ := exists_real_pos_lt_infEdist_of_not_mem_closure h
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    ⊢ Filter.Eventually (fun δ => Not (Membership.mem (Metric.thickening δ E) x))  …
  -/
  filter_upwards [eventually_lt_nhds ε_pos] with δ hδ
  /-
    case h
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    δ : Real
    hδ : LT.lt δ ε
    ⊢ Not (Membership.mem (Metric.thickening δ E) x)
  -/
  simp only [thickening, mem_setOf_eq, not_lt]
  /-
    case h
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    δ : Real
    hδ : LT.lt δ ε
    ⊢ LE.le (ENNReal.ofReal δ) (EMetric.infEdist x E)
  -/
  exact (ENNReal.ofReal_le_ofReal hδ.le).trans ε_lt.le
  /-
    🎉 no goals
  -/


/-- The (open) thickening equals the preimage of an open interval under `EMetric.infEdist`. -/
theorem thickening_eq_preimage_infEdist (δ : ℝ) (E : Set α) :
    thickening δ E = (infEdist · E) ⁻¹' Iio (ENNReal.ofReal δ) :=
  rfl


/-- The (open) thickening is an open set. -/
theorem isOpen_thickening {δ : ℝ} {E : Set α} : IsOpen (thickening δ E) :=
  Continuous.isOpen_preimage continuous_infEdist _ isOpen_Iio


/-- The (open) thickening of the empty set is empty. -/
@[simp]
theorem thickening_empty (δ : ℝ) : thickening δ (∅ : Set α) = ∅ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    ⊢ Eq (Metric.thickening δ EmptyCollection.emptyCollection) EmptyCollection.emp …
  -/
  simp only [thickening, setOf_false, infEdist_empty, not_top_lt]
  /-
    🎉 no goals
  -/


theorem thickening_of_nonpos (hδ : δ ≤ 0) (s : Set α) : thickening δ s = ∅ :=
  eq_empty_of_forall_not_mem fun _ => ((ENNReal.ofReal_of_nonpos hδ).trans_le bot_le).not_lt


/-- The (open) thickening `Metric.thickening δ E` of a fixed subset `E` is an increasing function of
the thickening radius `δ`. -/
theorem thickening_mono {δ₁ δ₂ : ℝ} (hle : δ₁ ≤ δ₂) (E : Set α) :
    thickening δ₁ E ⊆ thickening δ₂ E :=
  preimage_mono (Iio_subset_Iio (ENNReal.ofReal_le_ofReal hle))


/-- The (open) thickening `Metric.thickening δ E` with a fixed thickening radius `δ` is
an increasing function of the subset `E`. -/
theorem thickening_subset_of_subset (δ : ℝ) {E₁ E₂ : Set α} (h : E₁ ⊆ E₂) :
    thickening δ E₁ ⊆ thickening δ E₂ := fun _ hx => lt_of_le_of_lt (infEdist_anti h) hx


theorem mem_thickening_iff_exists_edist_lt {δ : ℝ} (E : Set α) (x : α) :
    x ∈ thickening δ E ↔ ∃ z ∈ E, edist x z < ENNReal.ofReal δ :=
  infEdist_lt_iff


/-- The frontier of the (open) thickening of a set is contained in an `EMetric.infEdist` level
set. -/
theorem frontier_thickening_subset (E : Set α) {δ : ℝ} :
    frontier (thickening δ E) ⊆ { x : α | infEdist x E = ENNReal.ofReal δ } :=
  frontier_lt_subset_eq continuous_infEdist continuous_const


theorem frontier_thickening_disjoint (A : Set α) :
    Pairwise (Disjoint on fun r : ℝ => frontier (thickening r A)) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    A : Set α
    ⊢ Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening r A))
  -/
  refine (pairwise_disjoint_on _).2 fun r₁ r₂ hr => ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    A : Set α
    r₁ r₂ : Real
    hr : LT.lt r₁ r₂
    ⊢ Disjoint (frontier (Metric.thickening r₁ A)) (frontier (Metric.thickening r₂ …
  -/
  rcases le_total r₁ 0 with h₁ | h₁
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      A : Set α
      r₁ r₂ : Real
      hr : LT.lt r₁ r₂
      h₁ : LE.le r₁ 0
      ⊢ Disjoint (frontier (Metric.thickening r₁ A)) (frontier (Metric.thickening r₂ …
    -/
  · simp [thickening_of_nonpos h₁]
    /-
      🎉 no goals
    -/
  refine ((disjoint_singleton.2 fun h => hr.ne ?_).preimage _).mono (frontier_thickening_subset _)
    (frontier_thickening_subset _)
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    A : Set α
    r₁ r₂ : Real
    hr : LT.lt r₁ r₂
    h₁ : LE.le 0 r₁
    h : Eq (ENNReal.ofReal r₁) (ENNReal.ofReal r₂)
    ⊢ Eq r₁ r₂
  -/
  apply_fun ENNReal.toReal at h
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    A : Set α
    r₁ r₂ : Real
    hr : LT.lt r₁ r₂
    h₁ : LE.le 0 r₁
    h : Eq (ENNReal.ofReal r₁).toReal (ENNReal.ofReal r₂).toReal
    ⊢ Eq r₁ r₂
  -/
  rwa [ENNReal.toReal_ofReal h₁, ENNReal.toReal_ofReal (h₁.trans hr.le)] at h
  /-
    🎉 no goals
  -/


/-- Any set is contained in the complement of the δ-thickening of the complement of its
δ-thickening. -/
lemma subset_compl_thickening_compl_thickening_self (δ : ℝ) (E : Set α) :
    E ⊆ (thickening δ (thickening δ E)ᶜ)ᶜ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset E (HasCompl.compl (Metric.thickening δ (HasCompl.compl (Met …
  -/
  intro x x_in_E
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_in_E : Membership.mem E x
    ⊢ Membership.mem (HasCompl.compl (Metric.thickening δ (HasCompl.compl (Metric. …
  -/
  simp only [thickening, mem_compl_iff, mem_setOf_eq, not_lt]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_in_E : Membership.mem E x
    ⊢ LE.le (ENNReal.ofReal δ) (EMetric.infEdist x (HasCompl.compl (setOf fun x => …
  -/
  apply EMetric.le_infEdist.mpr fun y hy ↦ ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_in_E : Membership.mem E x
    y : α
    hy : Membership.mem (HasCompl.compl (setOf fun x => LT.lt (EMetric.infEdist x  …
    ⊢ LE.le (ENNReal.ofReal δ) (EDist.edist x y)
  -/
  simp only [mem_compl_iff, mem_setOf_eq, not_lt] at hy
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    x_in_E : Membership.mem E x
    y : α
    hy : LE.le (ENNReal.ofReal δ) (EMetric.infEdist y E)
    ⊢ LE.le (ENNReal.ofReal δ) (EDist.edist x y)
  -/
  simpa only [edist_comm] using le_trans hy <| EMetric.infEdist_le_edist_of_mem x_in_E
  /-
    🎉 no goals
  -/


/-- The δ-thickening of the complement of the δ-thickening of a set is contained in the complement
of the set. -/
lemma thickening_compl_thickening_self_subset_compl (δ : ℝ) (E : Set α) :
    thickening δ (thickening δ E)ᶜ ⊆ Eᶜ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset (Metric.thickening δ (HasCompl.compl (Metric.thickening δ E …
  -/
  apply compl_subset_compl.mp
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset (HasCompl.compl (HasCompl.compl E)) (HasCompl.compl (Metric …
  -/
  simpa only [compl_compl] using subset_compl_thickening_compl_thickening_self δ E
  /-
    🎉 no goals
  -/


theorem mem_thickening_iff_infDist_lt {E : Set X} {x : X} (h : E.Nonempty) :
    x ∈ thickening δ E ↔ infDist x E < δ :=
  lt_ofReal_iff_toReal_lt (infEdist_ne_top h)


/-- A point in a metric space belongs to the (open) `δ`-thickening of a subset `E` if and only if
it is at distance less than `δ` from some point of `E`. -/
theorem mem_thickening_iff {E : Set X} {x : X} : x ∈ thickening δ E ↔ ∃ z ∈ E, dist x z < δ := by
  have key_iff : ∀ z : X, edist x z < ENNReal.ofReal δ ↔ dist x z < δ := fun z ↦ by
    rw [dist_edist, lt_ofReal_iff_toReal_lt (edist_ne_top _ _)]
  /-
    δ : Real
    X : Type u
    inst✝ : PseudoMetricSpace X
    E : Set X
    x : X
    key_iff : ∀ (z : X), Iff (LT.lt (EDist.edist x z) (ENNReal.ofReal δ)) (LT.lt ( …
    ⊢ Iff (Membership.mem (Metric.thickening δ E) x) (Exists fun z => And (Members …
  -/
  simp_rw [mem_thickening_iff_exists_edist_lt, key_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem thickening_singleton (δ : ℝ) (x : X) : thickening δ ({x} : Set X) = ball x δ := by
  /-
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    x : X
    ⊢ Eq (Metric.thickening δ (Singleton.singleton x)) (Metric.ball x δ)
  -/
  ext
  /-
    case h
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    x x✝ : X
    ⊢ Iff (Membership.mem (Metric.thickening δ (Singleton.singleton x)) x✝) (Membe …
  -/
  simp [mem_thickening_iff]
  /-
    🎉 no goals
  -/


theorem ball_subset_thickening {x : X} {E : Set X} (hx : x ∈ E) (δ : ℝ) :
    ball x δ ⊆ thickening δ E :=
                   /-
                     X : Type u
                     inst✝ : PseudoMetricSpace X
                     x : X
                     E : Set X
                     hx : Membership.mem E x
                     δ : Real
                     ⊢ HasSubset.Subset (Metric.ball x δ) (Metric.thickening δ (Singleton.singleton …
                   -/
  Subset.trans (by simp [Subset.rfl]) (thickening_subset_of_subset δ <| singleton_subset_iff.mpr hx)
                   /-
                     🎉 no goals
                   -/


/-- The (open) `δ`-thickening `Metric.thickening δ E` of a subset `E` in a metric space equals the
union of balls of radius `δ` centered at points of `E`. -/
theorem thickening_eq_biUnion_ball {δ : ℝ} {E : Set X} : thickening δ E = ⋃ x ∈ E, ball x δ := by
  /-
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    E : Set X
    ⊢ Eq (Metric.thickening δ E) (Set.iUnion fun x => Set.iUnion fun h => Metric.b …
  -/
  ext x
  /-
    case h
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    E : Set X
    x : X
    ⊢ Iff (Membership.mem (Metric.thickening δ E) x) (Membership.mem (Set.iUnion f …
  -/
  simp only [mem_iUnion₂, exists_prop]
  /-
    case h
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    E : Set X
    x : X
    ⊢ Iff (Membership.mem (Metric.thickening δ E) x) (Exists fun i => And (Members …
  -/
  exact mem_thickening_iff
  /-
    🎉 no goals
  -/


protected theorem _root_.Bornology.IsBounded.thickening {δ : ℝ} {E : Set X} (h : IsBounded E) :
    IsBounded (thickening δ E) := by
  /-
    X : Type u
    inst✝ : PseudoMetricSpace X
    δ : Real
    E : Set X
    h : Bornology.IsBounded E
    ⊢ Bornology.IsBounded (Metric.thickening δ E)
  -/
  rcases E.eq_empty_or_nonempty with rfl | ⟨x, hx⟩
    /-
      case inl
      X : Type u
      inst✝ : PseudoMetricSpace X
      δ : Real
      h : Bornology.IsBounded EmptyCollection.emptyCollection
      ⊢ Bornology.IsBounded (Metric.thickening δ EmptyCollection.emptyCollection)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      X : Type u
      inst✝ : PseudoMetricSpace X
      δ : Real
      E : Set X
      h : Bornology.IsBounded E
      x : X
      hx : Membership.mem E x
      ⊢ Bornology.IsBounded (Metric.thickening δ E)
    -/
  · refine (isBounded_iff_subset_closedBall x).2 ⟨δ + diam E, fun y hy ↦ ?_⟩
    calc
      dist y x ≤ infDist y E + diam E := dist_le_infDist_add_diam (x := y) h hx
      _ ≤ δ + diam E := add_le_add_right ((mem_thickening_iff_infDist_lt ⟨x, hx⟩).1 hy).le _


/-- The closed `δ`-thickening `Metric.cthickening δ E` of a subset `E` in a pseudo emetric space
consists of those points that are at infimum distance at most `δ` from `E`. -/
def cthickening (δ : ℝ) (E : Set α) : Set α :=
  { x : α | infEdist x E ≤ ENNReal.ofReal δ }


@[simp]
theorem mem_cthickening_iff : x ∈ cthickening δ s ↔ infEdist x s ≤ ENNReal.ofReal δ :=
  Iff.rfl


/-- An exterior point of a subset `E` (i.e., a point outside the closure of `E`) is not in the
closed `δ`-thickening of `E` for small enough positive `δ`. -/
lemma eventually_not_mem_cthickening_of_infEdist_pos {E : Set α} {x : α} (h : x ∉ closure E) :
    ∀ᶠ δ in 𝓝 (0 : ℝ), x ∉ Metric.cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ⊢ Filter.Eventually (fun δ => Not (Membership.mem (Metric.cthickening δ E) x)) …
  -/
  obtain ⟨ε, ⟨ε_pos, ε_lt⟩⟩ := exists_real_pos_lt_infEdist_of_not_mem_closure h
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    ⊢ Filter.Eventually (fun δ => Not (Membership.mem (Metric.cthickening δ E) x)) …
  -/
  filter_upwards [eventually_lt_nhds ε_pos] with δ hδ
  /-
    case h
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    δ : Real
    hδ : LT.lt δ ε
    ⊢ Not (Membership.mem (Metric.cthickening δ E) x)
  -/
  simp only [cthickening, mem_setOf_eq, not_le]
  /-
    case h
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    x : α
    h : Not (Membership.mem (closure E) x)
    ε : Real
    ε_pos : LT.lt 0 ε
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    δ : Real
    hδ : LT.lt δ ε
    ⊢ LT.lt (ENNReal.ofReal δ) (EMetric.infEdist x E)
  -/
  exact ((ofReal_lt_ofReal_iff ε_pos).mpr hδ).trans ε_lt
  /-
    🎉 no goals
  -/


theorem mem_cthickening_of_edist_le (x y : α) (δ : ℝ) (E : Set α) (h : y ∈ E)
    (h' : edist x y ≤ ENNReal.ofReal δ) : x ∈ cthickening δ E :=
  (infEdist_le_edist_of_mem h).trans h'


theorem mem_cthickening_of_dist_le {α : Type*} [PseudoMetricSpace α] (x y : α) (δ : ℝ) (E : Set α)
    (h : y ∈ E) (h' : dist x y ≤ δ) : x ∈ cthickening δ E := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x y : α
    δ : Real
    E : Set α
    h : Membership.mem E y
    h' : LE.le (Dist.dist x y) δ
    ⊢ Membership.mem (Metric.cthickening δ E) x
  -/
  apply mem_cthickening_of_edist_le x y δ E h
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x y : α
    δ : Real
    E : Set α
    h : Membership.mem E y
    h' : LE.le (Dist.dist x y) δ
    ⊢ LE.le (EDist.edist x y) (ENNReal.ofReal δ)
  -/
  rw [edist_dist]
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x y : α
    δ : Real
    E : Set α
    h : Membership.mem E y
    h' : LE.le (Dist.dist x y) δ
    ⊢ LE.le (ENNReal.ofReal (Dist.dist x y)) (ENNReal.ofReal δ)
  -/
  exact ENNReal.ofReal_le_ofReal h'
  /-
    🎉 no goals
  -/


theorem cthickening_eq_preimage_infEdist (δ : ℝ) (E : Set α) :
    cthickening δ E = (fun x => infEdist x E) ⁻¹' Iic (ENNReal.ofReal δ) :=
  rfl


/-- The closed thickening is a closed set. -/
theorem isClosed_cthickening {δ : ℝ} {E : Set α} : IsClosed (cthickening δ E) :=
  IsClosed.preimage continuous_infEdist isClosed_Iic


/-- The closed thickening of the empty set is empty. -/
@[simp]
theorem cthickening_empty (δ : ℝ) : cthickening δ (∅ : Set α) = ∅ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    ⊢ Eq (Metric.cthickening δ EmptyCollection.emptyCollection) EmptyCollection.em …
  -/
  simp only [cthickening, ENNReal.ofReal_ne_top, setOf_false, infEdist_empty, top_le_iff]
  /-
    🎉 no goals
  -/


theorem cthickening_of_nonpos {δ : ℝ} (hδ : δ ≤ 0) (E : Set α) : cthickening δ E = closure E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    hδ : LE.le δ 0
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (closure E)
  -/
  ext x
  /-
    case h
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    hδ : LE.le δ 0
    E : Set α
    x : α
    ⊢ Iff (Membership.mem (Metric.cthickening δ E) x) (Membership.mem (closure E) x)
  -/
  simp [mem_closure_iff_infEdist_zero, cthickening, ENNReal.ofReal_eq_zero.2 hδ]
  /-
    🎉 no goals
  -/


/-- The closed thickening with radius zero is the closure of the set. -/
@[simp]
theorem cthickening_zero (E : Set α) : cthickening 0 E = closure E :=
  cthickening_of_nonpos le_rfl E


theorem cthickening_max_zero (δ : ℝ) (E : Set α) : cthickening (max 0 δ) E = cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ Eq (Metric.cthickening (Max.max 0 δ) E) (Metric.cthickening δ E)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total δ 0 <;> simp [cthickening_of_nonpos, *]
                         /-
                           🎉 no goals
                         -/


/-- The closed thickening `Metric.cthickening δ E` of a fixed subset `E` is an increasing function
of the thickening radius `δ`. -/
theorem cthickening_mono {δ₁ δ₂ : ℝ} (hle : δ₁ ≤ δ₂) (E : Set α) :
    cthickening δ₁ E ⊆ cthickening δ₂ E :=
  preimage_mono (Iic_subset_Iic.mpr (ENNReal.ofReal_le_ofReal hle))


@[simp]
theorem cthickening_singleton {α : Type*} [PseudoMetricSpace α] (x : α) {δ : ℝ} (hδ : 0 ≤ δ) :
    cthickening δ ({x} : Set α) = closedBall x δ := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    δ : Real
    hδ : LE.le 0 δ
    ⊢ Eq (Metric.cthickening δ (Singleton.singleton x)) (Metric.closedBall x δ)
  -/
  ext y
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    δ : Real
    hδ : LE.le 0 δ
    y : α
    ⊢ Iff (Membership.mem (Metric.cthickening δ (Singleton.singleton x)) y) (Membe …
  -/
  simp [cthickening, edist_dist, ENNReal.ofReal_le_ofReal_iff hδ]
  /-
    🎉 no goals
  -/


theorem closedBall_subset_cthickening_singleton {α : Type*} [PseudoMetricSpace α] (x : α) (δ : ℝ) :
    closedBall x δ ⊆ cthickening δ ({x} : Set α) := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    δ : Real
    ⊢ HasSubset.Subset (Metric.closedBall x δ) (Metric.cthickening δ (Singleton.si …
  -/
  rcases lt_or_le δ 0 with (hδ | hδ)
    /-
      case inl
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      x : α
      δ : Real
      hδ : LT.lt δ 0
      ⊢ HasSubset.Subset (Metric.closedBall x δ) (Metric.cthickening δ (Singleton.si …
    -/
  · simp only [closedBall_eq_empty.mpr hδ, empty_subset]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      x : α
      δ : Real
      hδ : LE.le 0 δ
      ⊢ HasSubset.Subset (Metric.closedBall x δ) (Metric.cthickening δ (Singleton.si …
    -/
  · simp only [cthickening_singleton x hδ, Subset.rfl]
    /-
      🎉 no goals
    -/


/-- The closed thickening `Metric.cthickening δ E` with a fixed thickening radius `δ` is
an increasing function of the subset `E`. -/
theorem cthickening_subset_of_subset (δ : ℝ) {E₁ E₂ : Set α} (h : E₁ ⊆ E₂) :
    cthickening δ E₁ ⊆ cthickening δ E₂ := fun _ hx => le_trans (infEdist_anti h) hx


theorem cthickening_subset_thickening {δ₁ : ℝ≥0} {δ₂ : ℝ} (hlt : (δ₁ : ℝ) < δ₂) (E : Set α) :
    cthickening δ₁ E ⊆ thickening δ₂ E := fun _ hx =>
  hx.out.trans_lt ((ENNReal.ofReal_lt_ofReal_iff (lt_of_le_of_lt δ₁.prop hlt)).mpr hlt)


/-- The closed thickening `Metric.cthickening δ₁ E` is contained in the open thickening
`Metric.thickening δ₂ E` if the radius of the latter is positive and larger. -/
theorem cthickening_subset_thickening' {δ₁ δ₂ : ℝ} (δ₂_pos : 0 < δ₂) (hlt : δ₁ < δ₂) (E : Set α) :
    cthickening δ₁ E ⊆ thickening δ₂ E := fun _ hx =>
  lt_of_le_of_lt hx.out ((ENNReal.ofReal_lt_ofReal_iff δ₂_pos).mpr hlt)


/-- The open thickening `Metric.thickening δ E` is contained in the closed thickening
`Metric.cthickening δ E` with the same radius. -/
theorem thickening_subset_cthickening (δ : ℝ) (E : Set α) : thickening δ E ⊆ cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset (Metric.thickening δ E) (Metric.cthickening δ E)
  -/
  intro x hx
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    hx : Membership.mem (Metric.thickening δ E) x
    ⊢ Membership.mem (Metric.cthickening δ E) x
  -/
  rw [thickening, mem_setOf_eq] at hx
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    x : α
    hx : LT.lt (EMetric.infEdist x E) (ENNReal.ofReal δ)
    ⊢ Membership.mem (Metric.cthickening δ E) x
  -/
  exact hx.le
  /-
    🎉 no goals
  -/


theorem thickening_subset_cthickening_of_le {δ₁ δ₂ : ℝ} (hle : δ₁ ≤ δ₂) (E : Set α) :
    thickening δ₁ E ⊆ cthickening δ₂ E :=
  (thickening_subset_cthickening δ₁ E).trans (cthickening_mono hle E)


theorem _root_.Bornology.IsBounded.cthickening {α : Type*} [PseudoMetricSpace α] {δ : ℝ} {E : Set α}
    (h : IsBounded E) : IsBounded (cthickening δ E) := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    h : Bornology.IsBounded E
    ⊢ Bornology.IsBounded (Metric.cthickening δ E)
  -/
  have : IsBounded (thickening (max (δ + 1) 1) E) := h.thickening
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    h : Bornology.IsBounded E
    this : Bornology.IsBounded (Metric.thickening (Max.max (HAdd.hAdd δ 1) 1) E)
    ⊢ Bornology.IsBounded (Metric.cthickening δ E)
  -/
  apply this.subset
  exact cthickening_subset_thickening' (zero_lt_one.trans_le (le_max_right _ _))
    ((lt_add_one _).trans_le (le_max_left _ _)) _


protected theorem _root_.IsCompact.cthickening
    {α : Type*} [PseudoMetricSpace α] [ProperSpace α] {s : Set α}
    (hs : IsCompact s) {r : ℝ} : IsCompact (cthickening r s) :=
  isCompact_of_isClosed_isBounded isClosed_cthickening hs.isBounded.cthickening


theorem thickening_subset_interior_cthickening (δ : ℝ) (E : Set α) :
    thickening δ E ⊆ interior (cthickening δ E) :=
  (subset_interior_iff_isOpen.mpr isOpen_thickening).trans
    (interior_mono (thickening_subset_cthickening δ E))


theorem closure_thickening_subset_cthickening (δ : ℝ) (E : Set α) :
    closure (thickening δ E) ⊆ cthickening δ E :=
  (closure_mono (thickening_subset_cthickening δ E)).trans isClosed_cthickening.closure_subset


/-- The closed thickening of a set contains the closure of the set. -/
theorem closure_subset_cthickening (δ : ℝ) (E : Set α) : closure E ⊆ cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset (closure E) (Metric.cthickening δ E)
  -/
  rw [← cthickening_of_nonpos (min_le_right δ 0)]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ HasSubset.Subset (Metric.cthickening (Min.min δ 0) E) (Metric.cthickening δ E)
  -/
  exact cthickening_mono (min_le_left δ 0) E
  /-
    🎉 no goals
  -/


/-- The (open) thickening of a set contains the closure of the set. -/
theorem closure_subset_thickening {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) :
    closure E ⊆ thickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    ⊢ HasSubset.Subset (closure E) (Metric.thickening δ E)
  -/
  rw [← cthickening_zero]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_pos : LT.lt 0 δ
    E : Set α
    ⊢ HasSubset.Subset (Metric.cthickening 0 E) (Metric.thickening δ E)
  -/
  exact cthickening_subset_thickening' δ_pos δ_pos E
  /-
    🎉 no goals
  -/


/-- A set is contained in its own (open) thickening. -/
theorem self_subset_thickening {δ : ℝ} (δ_pos : 0 < δ) (E : Set α) : E ⊆ thickening δ E :=
  (@subset_closure _ E).trans (closure_subset_thickening δ_pos E)


/-- A set is contained in its own closed thickening. -/
theorem self_subset_cthickening {δ : ℝ} (E : Set α) : E ⊆ cthickening δ E :=
  subset_closure.trans (closure_subset_cthickening δ E)


theorem thickening_mem_nhdsSet (E : Set α) {δ : ℝ} (hδ : 0 < δ) : thickening δ E ∈ 𝓝ˢ E :=
  isOpen_thickening.mem_nhdsSet.2 <| self_subset_thickening hδ E


theorem cthickening_mem_nhdsSet (E : Set α) {δ : ℝ} (hδ : 0 < δ) : cthickening δ E ∈ 𝓝ˢ E :=
  mem_of_superset (thickening_mem_nhdsSet E hδ) (thickening_subset_cthickening _ _)


@[simp]
theorem thickening_union (δ : ℝ) (s t : Set α) :
    thickening δ (s ∪ t) = thickening δ s ∪ thickening δ t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s t : Set α
    ⊢ Eq (Metric.thickening δ (Union.union s t)) (Union.union (Metric.thickening δ …
  -/
  simp_rw [thickening, infEdist_union, min_lt_iff, setOf_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem cthickening_union (δ : ℝ) (s t : Set α) :
    cthickening δ (s ∪ t) = cthickening δ s ∪ cthickening δ t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s t : Set α
    ⊢ Eq (Metric.cthickening δ (Union.union s t)) (Union.union (Metric.cthickening …
  -/
  simp_rw [cthickening, infEdist_union, min_le_iff, setOf_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem thickening_iUnion (δ : ℝ) (f : ι → Set α) :
    thickening δ (⋃ i, f i) = ⋃ i, thickening δ (f i) := by
  /-
    ι : Sort u_1
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    f : ι → Set α
    ⊢ Eq (Metric.thickening δ (Set.iUnion fun i => f i)) (Set.iUnion fun i => Metr …
  -/
  simp_rw [thickening, infEdist_iUnion, iInf_lt_iff, setOf_exists]
  /-
    🎉 no goals
  -/


lemma thickening_biUnion {ι : Type*} (δ : ℝ) (f : ι → Set α) (I : Set ι) :
                                                                    /-
                                                                      α : Type u
                                                                      inst✝ : PseudoEMetricSpace α
                                                                      ι : Type u_2
                                                                      δ : Real
                                                                      f : ι → Set α
                                                                      I : Set ι
                                                                      ⊢ Eq (Metric.thickening δ (Set.iUnion fun i => Set.iUnion fun h => f i)) (Set. …
                                                                    -/
    thickening δ (⋃ i ∈ I, f i) = ⋃ i ∈ I, thickening δ (f i) := by simp only [thickening_iUnion]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem ediam_cthickening_le (ε : ℝ≥0) :
    EMetric.diam (cthickening ε s) ≤ EMetric.diam s + 2 * ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ε : NNReal
    ⊢ LE.le (EMetric.diam (Metric.cthickening (↑ε) s)) (HAdd.hAdd (EMetric.diam s) …
  -/
  refine diam_le fun x hx y hy => ENNReal.le_of_forall_pos_le_add fun δ hδ _ => ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ε : NNReal
    x : α
    hx : Membership.mem (Metric.cthickening (↑ε) s) x
    y : α
    hy : Membership.mem (Metric.cthickening (↑ε) s) y
    δ : NNReal
    hδ : LT.lt 0 δ
    x✝ : LT.lt (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2 ↑ε)) Top.top
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2  …
  -/
  rw [mem_cthickening_iff, ENNReal.ofReal_coe_nnreal] at hx hy
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ε : NNReal
    x : α
    hx : LE.le (EMetric.infEdist x s) ↑ε
    y : α
    hy : LE.le (EMetric.infEdist y s) ↑ε
    δ : NNReal
    hδ : LT.lt 0 δ
    x✝ : LT.lt (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2 ↑ε)) Top.top
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2  …
  -/
  have hε : (ε : ℝ≥0∞) < ε + δ := ENNReal.coe_lt_coe.2 (lt_add_of_pos_right _ hδ)
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ε : NNReal
    x : α
    hx : LE.le (EMetric.infEdist x s) ↑ε
    y : α
    hy : LE.le (EMetric.infEdist y s) ↑ε
    δ : NNReal
    hδ : LT.lt 0 δ
    x✝ : LT.lt (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2 ↑ε)) Top.top
    hε : LT.lt (↑ε) (HAdd.hAdd ↑ε ↑δ)
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2  …
  -/
  replace hx := hx.trans_lt hε
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ε : NNReal
    x y : α
    hy : LE.le (EMetric.infEdist y s) ↑ε
    δ : NNReal
    hδ : LT.lt 0 δ
    x✝ : LT.lt (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2 ↑ε)) Top.top
    hε : LT.lt (↑ε) (HAdd.hAdd ↑ε ↑δ)
    hx : LT.lt (EMetric.infEdist x s) (HAdd.hAdd ↑ε ↑δ)
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (HMul.hMul 2  …
  -/
  obtain ⟨x', hx', hxx'⟩ := infEdist_lt_iff.mp hx
  calc
    edist x y ≤ edist x x' + edist y x' := edist_triangle_right _ _ _
    _ ≤ ε + δ + (infEdist y s + EMetric.diam s) :=
      add_le_add hxx'.le (edist_le_infEdist_add_ediam hx')
    _ ≤ ε + δ + (ε + EMetric.diam s) := add_le_add_left (add_le_add_right hy _) _
    _ = _ := by rw [two_mul]; ac_rfl


theorem ediam_thickening_le (ε : ℝ≥0) : EMetric.diam (thickening ε s) ≤ EMetric.diam s + 2 * ε :=
  (EMetric.diam_mono <| thickening_subset_cthickening _ _).trans <| ediam_cthickening_le _


theorem diam_cthickening_le {α : Type*} [PseudoMetricSpace α] (s : Set α) (hε : 0 ≤ ε) :
    diam (cthickening ε s) ≤ diam s + 2 * ε := by
  /-
    ε : Real
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    s : Set α
    hε : LE.le 0 ε
    ⊢ LE.le (Metric.diam (Metric.cthickening ε s)) (HAdd.hAdd (Metric.diam s) (HMu …
  -/
  lift ε to ℝ≥0 using hε
  /-
    case intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    s : Set α
    ε : NNReal
    ⊢ LE.le (Metric.diam (Metric.cthickening (↑ε) s)) (HAdd.hAdd (Metric.diam s) ( …
  -/
  refine (toReal_le_add' (ediam_cthickening_le _) ?_ ?_).trans_eq ?_
    /-
      case intro.refine_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      ε : NNReal
      ⊢ Eq (EMetric.diam s) Top.top → Eq (EMetric.diam (Metric.cthickening (↑ε) s))  …
    -/
  · exact fun h ↦ top_unique <| h ▸ EMetric.diam_mono (self_subset_cthickening _)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      ε : NNReal
      ⊢ Eq (HMul.hMul 2 ↑ε) Top.top → Eq (EMetric.diam (Metric.cthickening (↑ε) s))  …
    -/
  · simp [mul_eq_top]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      ε : NNReal
      ⊢ Eq (HAdd.hAdd (EMetric.diam s).toReal (HMul.hMul 2 ↑ε).toReal) (HAdd.hAdd (M …
    -/
  · simp [diam]
    /-
      🎉 no goals
    -/


theorem diam_thickening_le {α : Type*} [PseudoMetricSpace α] (s : Set α) (hε : 0 ≤ ε) :
    diam (thickening ε s) ≤ diam s + 2 * ε := by
  /-
    ε : Real
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    s : Set α
    hε : LE.le 0 ε
    ⊢ LE.le (Metric.diam (Metric.thickening ε s)) (HAdd.hAdd (Metric.diam s) (HMul …
  -/
  by_cases hs : IsBounded s
  · exact (diam_mono (thickening_subset_cthickening _ _) hs.cthickening).trans
      (diam_cthickening_le _ hε)
  /-
    case neg
    ε : Real
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    s : Set α
    hε : LE.le 0 ε
    hs : Not (Bornology.IsBounded s)
    ⊢ LE.le (Metric.diam (Metric.thickening ε s)) (HAdd.hAdd (Metric.diam s) (HMul …
  -/
  obtain rfl | hε := hε.eq_or_lt
    /-
      case neg.inl
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      hs : Not (Bornology.IsBounded s)
      hε : LE.le 0 0
      ⊢ LE.le (Metric.diam (Metric.thickening 0 s)) (HAdd.hAdd (Metric.diam s) (HMul …
    -/
  · simp [thickening_of_nonpos, diam_nonneg]
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      ε : Real
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      hε✝ : LE.le 0 ε
      hs : Not (Bornology.IsBounded s)
      hε : LT.lt 0 ε
      ⊢ LE.le (Metric.diam (Metric.thickening ε s)) (HAdd.hAdd (Metric.diam s) (HMul …
    -/
  · rw [diam_eq_zero_of_unbounded (mt (IsBounded.subset · <| self_subset_thickening hε _) hs)]
    /-
      case neg.inr
      ε : Real
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      s : Set α
      hε✝ : LE.le 0 ε
      hs : Not (Bornology.IsBounded s)
      hε : LT.lt 0 ε
      ⊢ LE.le 0 (HAdd.hAdd (Metric.diam s) (HMul.hMul 2 ε))
    -/
    positivity
    /-
      🎉 no goals
    -/


@[simp]
theorem thickening_closure : thickening δ (closure s) = thickening δ s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set α
    ⊢ Eq (Metric.thickening δ (closure s)) (Metric.thickening δ s)
  -/
  simp_rw [thickening, infEdist_closure]
  /-
    🎉 no goals
  -/


@[simp]
theorem cthickening_closure : cthickening δ (closure s) = cthickening δ s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set α
    ⊢ Eq (Metric.cthickening δ (closure s)) (Metric.cthickening δ s)
  -/
  simp_rw [cthickening, infEdist_closure]
  /-
    🎉 no goals
  -/


theorem _root_.Disjoint.exists_thickenings (hst : Disjoint s t) (hs : IsCompact s)
    (ht : IsClosed t) :
    ∃ δ, 0 < δ ∧ Disjoint (thickening δ s) (thickening δ t) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    ⊢ Exists fun δ => And (LT.lt 0 δ) (Disjoint (Metric.thickening δ s) (Metric.th …
  -/
  obtain ⟨r, hr, h⟩ := exists_pos_forall_lt_edist hs ht hst
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    ⊢ Exists fun δ => And (LT.lt 0 δ) (Disjoint (Metric.thickening δ s) (Metric.th …
  -/
  refine ⟨r / 2, half_pos (NNReal.coe_pos.2 hr), ?_⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    ⊢ Disjoint (Metric.thickening (HDiv.hDiv (↑r) 2) s) (Metric.thickening (HDiv.h …
  -/
  rw [disjoint_iff_inf_le]
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    ⊢ LE.le (Min.min (Metric.thickening (HDiv.hDiv (↑r) 2) s) (Metric.thickening ( …
  -/
  rintro z ⟨hzs, hzt⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    z : α
    hzs : Membership.mem (Metric.thickening (HDiv.hDiv (↑r) 2) s) z
    hzt : Membership.mem (Metric.thickening (HDiv.hDiv (↑r) 2) t) z
    ⊢ Membership.mem Bot.bot z
  -/
  rw [mem_thickening_iff_exists_edist_lt] at hzs hzt
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    z : α
    hzs : Exists fun z_1 => And (Membership.mem s z_1) (LT.lt (EDist.edist z z_1)  …
    hzt : Exists fun z_1 => And (Membership.mem t z_1) (LT.lt (EDist.edist z z_1)  …
    ⊢ Membership.mem Bot.bot z
  -/
  rw [← NNReal.coe_two, ← NNReal.coe_div, ENNReal.ofReal_coe_nnreal] at hzs hzt
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    z : α
    hzs : Exists fun z_1 => And (Membership.mem s z_1) (LT.lt (EDist.edist z z_1)  …
    hzt : Exists fun z_1 => And (Membership.mem t z_1) (LT.lt (EDist.edist z z_1)  …
    ⊢ Membership.mem Bot.bot z
  -/
  obtain ⟨x, hx, hzx⟩ := hzs
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    z : α
    hzt : Exists fun z_1 => And (Membership.mem t z_1) (LT.lt (EDist.edist z z_1)  …
    x : α
    hx : Membership.mem s x
    hzx : LT.lt (EDist.edist z x) ↑(HDiv.hDiv r 2)
    ⊢ Membership.mem Bot.bot z
  -/
  obtain ⟨y, hy, hzy⟩ := hzt
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LT.lt (↑r) …
    z x : α
    hx : Membership.mem s x
    hzx : LT.lt (EDist.edist z x) ↑(HDiv.hDiv r 2)
    y : α
    hy : Membership.mem t y
    hzy : LT.lt (EDist.edist z y) ↑(HDiv.hDiv r 2)
    ⊢ Membership.mem Bot.bot z
  -/
  refine (h x hx y hy).not_le ?_
  calc
    edist x y ≤ edist z x + edist z y := edist_triangle_left _ _ _
    _ ≤ ↑(r / 2) + ↑(r / 2) := add_le_add hzx.le hzy.le
    _ = r := by rw [← ENNReal.coe_add, add_halves]


theorem _root_.Disjoint.exists_cthickenings (hst : Disjoint s t) (hs : IsCompact s)
    (ht : IsClosed t) :
    ∃ δ, 0 < δ ∧ Disjoint (cthickening δ s) (cthickening δ t) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    ⊢ Exists fun δ => And (LT.lt 0 δ) (Disjoint (Metric.cthickening δ s) (Metric.c …
  -/
  obtain ⟨δ, hδ, h⟩ := hst.exists_thickenings hs ht
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hst : Disjoint s t
    hs : IsCompact s
    ht : IsClosed t
    δ : Real
    hδ : LT.lt 0 δ
    h : Disjoint (Metric.thickening δ s) (Metric.thickening δ t)
    ⊢ Exists fun δ => And (LT.lt 0 δ) (Disjoint (Metric.cthickening δ s) (Metric.c …
  -/
  refine ⟨δ / 2, half_pos hδ, h.mono ?_ ?_⟩ <;>
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      hst : Disjoint s t
      hs : IsCompact s
      ht : IsClosed t
      δ : Real
      hδ : LT.lt 0 δ
      h : Disjoint (Metric.thickening δ s) (Metric.thickening δ t)
      ⊢ LE.le (Metric.cthickening (HDiv.hDiv δ 2) s) (Metric.thickening δ s)
    -/
    /-
      🎉 no goals
    -/
    exact cthickening_subset_thickening' hδ (half_lt_self hδ) _
    /-
      🎉 no goals
    -/


/-- If `s` is compact, `t` is open and `s ⊆ t`, some `cthickening` of `s` is contained in `t`. -/
theorem _root_.IsCompact.exists_cthickening_subset_open (hs : IsCompact s) (ht : IsOpen t)
    (hst : s ⊆ t) :
    ∃ δ, 0 < δ ∧ cthickening δ s ⊆ t :=
  (hst.disjoint_compl_right.exists_cthickenings hs ht.isClosed_compl).imp fun _ h =>
    ⟨h.1, disjoint_compl_right_iff_subset.1 <| h.2.mono_right <| self_subset_cthickening _⟩


theorem _root_.IsCompact.exists_isCompact_cthickening [LocallyCompactSpace α] (hs : IsCompact s) :
    ∃ δ, 0 < δ ∧ IsCompact (cthickening δ s) := by
  /-
    α : Type u
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : LocallyCompactSpace α
    hs : IsCompact s
    ⊢ Exists fun δ => And (LT.lt 0 δ) (IsCompact (Metric.cthickening δ s))
  -/
  rcases exists_compact_superset hs with ⟨K, K_compact, hK⟩
  /-
    case intro.intro
    α : Type u
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : LocallyCompactSpace α
    hs : IsCompact s
    K : Set α
    K_compact : IsCompact K
    hK : HasSubset.Subset s (interior K)
    ⊢ Exists fun δ => And (LT.lt 0 δ) (IsCompact (Metric.cthickening δ s))
  -/
  rcases hs.exists_cthickening_subset_open isOpen_interior hK with ⟨δ, δpos, hδ⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : LocallyCompactSpace α
    hs : IsCompact s
    K : Set α
    K_compact : IsCompact K
    hK : HasSubset.Subset s (interior K)
    δ : Real
    δpos : LT.lt 0 δ
    hδ : HasSubset.Subset (Metric.cthickening δ s) (interior K)
    ⊢ Exists fun δ => And (LT.lt 0 δ) (IsCompact (Metric.cthickening δ s))
  -/
  refine ⟨δ, δpos, ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : LocallyCompactSpace α
    hs : IsCompact s
    K : Set α
    K_compact : IsCompact K
    hK : HasSubset.Subset s (interior K)
    δ : Real
    δpos : LT.lt 0 δ
    hδ : HasSubset.Subset (Metric.cthickening δ s) (interior K)
    ⊢ IsCompact (Metric.cthickening δ s)
  -/
  exact K_compact.of_isClosed_subset isClosed_cthickening (hδ.trans interior_subset)
  /-
    🎉 no goals
  -/


theorem _root_.IsCompact.exists_thickening_subset_open (hs : IsCompact s) (ht : IsOpen t)
    (hst : s ⊆ t) : ∃ δ, 0 < δ ∧ thickening δ s ⊆ t :=
  let ⟨δ, h₀, hδ⟩ := hs.exists_cthickening_subset_open ht hst
  ⟨δ, h₀, (thickening_subset_cthickening _ _).trans hδ⟩


theorem hasBasis_nhdsSet_thickening {K : Set α} (hK : IsCompact K) :
    (𝓝ˢ K).HasBasis (fun δ : ℝ => 0 < δ) fun δ => thickening δ K :=
  (hasBasis_nhdsSet K).to_hasBasis' (fun _U hU => hK.exists_thickening_subset_open hU.1 hU.2)
    fun _ => thickening_mem_nhdsSet K


theorem hasBasis_nhdsSet_cthickening {K : Set α} (hK : IsCompact K) :
    (𝓝ˢ K).HasBasis (fun δ : ℝ => 0 < δ) fun δ => cthickening δ K :=
  (hasBasis_nhdsSet K).to_hasBasis' (fun _U hU => hK.exists_cthickening_subset_open hU.1 hU.2)
    fun _ => cthickening_mem_nhdsSet K


theorem cthickening_eq_iInter_cthickening' {δ : ℝ} (s : Set ℝ) (hsδ : s ⊆ Ioi δ)
    (hs : ∀ ε, δ < ε → (s ∩ Ioc δ ε).Nonempty) (E : Set α) :
    cthickening δ E = ⋂ ε ∈ s, cthickening ε E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set Real
    hsδ : HasSubset.Subset s (Set.Ioi δ)
    hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fun h => Metric. …
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ⊢ HasSubset.Subset (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fu …
    -/
  · exact subset_iInter₂ fun _ hε => cthickening_mono (le_of_lt (hsδ hε)) E
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ⊢ HasSubset.Subset (Set.iInter fun ε => Set.iInter fun h => Metric.cthickening …
    -/
  · unfold cthickening
    /-
      case h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ⊢ HasSubset.Subset (Set.iInter fun ε => Set.iInter fun h => setOf fun x => LE. …
    -/
    intro x hx
    /-
      case h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : Membership.mem (Set.iInter fun ε => Set.iInter fun h => setOf fun x => LE …
      ⊢ Membership.mem (setOf fun x => LE.le (EMetric.infEdist x E) (ENNReal.ofReal  …
    -/
    simp only [mem_iInter, mem_setOf_eq] at *
    /-
      case h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      ⊢ LE.le (EMetric.infEdist x E) (ENNReal.ofReal δ)
    -/
    apply ENNReal.le_of_forall_pos_le_add
    /-
      case h₂.h
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt (ENNReal.ofReal δ) Top.top → LE.le (EMetri …
    -/
    intro η η_pos _
    /-
      case h₂.h
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      η : NNReal
      η_pos : LT.lt 0 η
      a✝ : LT.lt (ENNReal.ofReal δ) Top.top
      ⊢ LE.le (EMetric.infEdist x E) (HAdd.hAdd (ENNReal.ofReal δ) ↑η)
    -/
    rcases hs (δ + η) (lt_add_of_pos_right _ (NNReal.coe_pos.mpr η_pos)) with ⟨ε, ⟨hsε, hε⟩⟩
    /-
      case h₂.h.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      η : NNReal
      η_pos : LT.lt 0 η
      a✝ : LT.lt (ENNReal.ofReal δ) Top.top
      ε : Real
      hsε : Membership.mem s ε
      hε : Membership.mem (Set.Ioc δ (HAdd.hAdd δ ↑η)) ε
      ⊢ LE.le (EMetric.infEdist x E) (HAdd.hAdd (ENNReal.ofReal δ) ↑η)
    -/
    apply ((hx ε hsε).trans (ENNReal.ofReal_le_ofReal hε.2)).trans
    /-
      case h₂.h.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      η : NNReal
      η_pos : LT.lt 0 η
      a✝ : LT.lt (ENNReal.ofReal δ) Top.top
      ε : Real
      hsε : Membership.mem s ε
      hε : Membership.mem (Set.Ioc δ (HAdd.hAdd δ ↑η)) ε
      ⊢ LE.le (ENNReal.ofReal (HAdd.hAdd δ ↑η)) (HAdd.hAdd (ENNReal.ofReal δ) ↑η)
    -/
    rw [ENNReal.coe_nnreal_eq η]
    /-
      case h₂.h.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      x : α
      hx : ∀ (i : Real), Membership.mem s i → LE.le (EMetric.infEdist x E) (ENNReal. …
      η : NNReal
      η_pos : LT.lt 0 η
      a✝ : LT.lt (ENNReal.ofReal δ) Top.top
      ε : Real
      hsε : Membership.mem s ε
      hε : Membership.mem (Set.Ioc δ (HAdd.hAdd δ ↑η)) ε
      ⊢ LE.le (ENNReal.ofReal (HAdd.hAdd δ ↑η)) (HAdd.hAdd (ENNReal.ofReal δ) (ENNRe …
    -/
    exact ENNReal.ofReal_add_le
    /-
      🎉 no goals
    -/


theorem cthickening_eq_iInter_cthickening {δ : ℝ} (E : Set α) :
    cthickening δ E = ⋂ (ε : ℝ) (_ : δ < ε), cthickening ε E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fun x => Metric. …
  -/
  apply cthickening_eq_iInter_cthickening' (Ioi δ) rfl.subset
  /-
    case hs
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ ∀ (ε : Real), LT.lt δ ε → (Inter.inter (Set.Ioi δ) (Set.Ioc δ ε)).Nonempty
  -/
  simp_rw [inter_eq_right.mpr Ioc_subset_Ioi_self]
  /-
    case hs
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ ∀ (ε : Real), LT.lt δ ε → (Set.Ioc δ ε).Nonempty
  -/
  exact fun _ hε => nonempty_Ioc.mpr hε
  /-
    🎉 no goals
  -/


theorem cthickening_eq_iInter_thickening' {δ : ℝ} (δ_nn : 0 ≤ δ) (s : Set ℝ) (hsδ : s ⊆ Ioi δ)
    (hs : ∀ ε, δ < ε → (s ∩ Ioc δ ε).Nonempty) (E : Set α) :
    cthickening δ E = ⋂ ε ∈ s, thickening ε E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_nn : LE.le 0 δ
    s : Set Real
    hsδ : HasSubset.Subset s (Set.Ioi δ)
    hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fun h => Metric. …
  -/
  refine (subset_iInter₂ fun ε hε => ?_).antisymm ?_
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_nn : LE.le 0 δ
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ε : Real
      hε : Membership.mem s ε
      ⊢ HasSubset.Subset (Metric.cthickening δ E) (Metric.thickening ε E)
    -/
  · obtain ⟨ε', -, hε'⟩ := hs ε (hsδ hε)
    /-
      case refine_1.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_nn : LE.le 0 δ
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ε : Real
      hε : Membership.mem s ε
      ε' : Real
      hε' : Membership.mem (Set.Ioc δ ε) ε'
      ⊢ HasSubset.Subset (Metric.cthickening δ E) (Metric.thickening ε E)
    -/
    have ss := cthickening_subset_thickening' (lt_of_le_of_lt δ_nn hε'.1) hε'.1 E
    /-
      case refine_1.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_nn : LE.le 0 δ
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ε : Real
      hε : Membership.mem s ε
      ε' : Real
      hε' : Membership.mem (Set.Ioc δ ε) ε'
      ss : HasSubset.Subset (Metric.cthickening δ E) (Metric.thickening ε' E)
      ⊢ HasSubset.Subset (Metric.cthickening δ E) (Metric.thickening ε E)
    -/
    exact ss.trans (thickening_mono hε'.2 E)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_nn : LE.le 0 δ
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ⊢ HasSubset.Subset (Set.iInter fun i => Set.iInter fun j => Metric.thickening  …
    -/
  · rw [cthickening_eq_iInter_cthickening' s hsδ hs E]
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ : Real
      δ_nn : LE.le 0 δ
      s : Set Real
      hsδ : HasSubset.Subset s (Set.Ioi δ)
      hs : ∀ (ε : Real), LT.lt δ ε → (Inter.inter s (Set.Ioc δ ε)).Nonempty
      E : Set α
      ⊢ HasSubset.Subset (Set.iInter fun i => Set.iInter fun j => Metric.thickening  …
    -/
    exact iInter₂_mono fun ε _ => thickening_subset_cthickening ε E
    /-
      🎉 no goals
    -/


theorem cthickening_eq_iInter_thickening {δ : ℝ} (δ_nn : 0 ≤ δ) (E : Set α) :
    cthickening δ E = ⋂ (ε : ℝ) (_ : δ < ε), thickening ε E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_nn : LE.le 0 δ
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fun x => Metric. …
  -/
  apply cthickening_eq_iInter_thickening' δ_nn (Ioi δ) rfl.subset
  /-
    case hs
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_nn : LE.le 0 δ
    E : Set α
    ⊢ ∀ (ε : Real), LT.lt δ ε → (Inter.inter (Set.Ioi δ) (Set.Ioc δ ε)).Nonempty
  -/
  simp_rw [inter_eq_right.mpr Ioc_subset_Ioi_self]
  /-
    case hs
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    δ_nn : LE.le 0 δ
    E : Set α
    ⊢ ∀ (ε : Real), LT.lt δ ε → (Set.Ioc δ ε).Nonempty
  -/
  exact fun _ hε => nonempty_Ioc.mpr hε
  /-
    🎉 no goals
  -/


theorem cthickening_eq_iInter_thickening'' (δ : ℝ) (E : Set α) :
    cthickening δ E = ⋂ (ε : ℝ) (_ : max 0 δ < ε), thickening ε E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ Eq (Metric.cthickening δ E) (Set.iInter fun ε => Set.iInter fun x => Metric. …
  -/
  rw [← cthickening_max_zero, cthickening_eq_iInter_thickening]
  /-
    case δ_nn
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    E : Set α
    ⊢ LE.le 0 (Max.max 0 δ)
  -/
  exact le_max_left _ _
  /-
    🎉 no goals
  -/


/-- The closure of a set equals the intersection of its closed thickenings of positive radii
accumulating at zero. -/
theorem closure_eq_iInter_cthickening' (E : Set α) (s : Set ℝ)
    (hs : ∀ ε, 0 < ε → (s ∩ Ioc 0 ε).Nonempty) : closure E = ⋂ δ ∈ s, cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening δ …
  -/
  by_cases hs₀ : s ⊆ Ioi 0
    /-
      case pos
      α : Type u
      inst✝ : PseudoEMetricSpace α
      E : Set α
      s : Set Real
      hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
      hs₀ : HasSubset.Subset s (Set.Ioi 0)
      ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening δ …
    -/
  · rw [← cthickening_zero]
    /-
      case pos
      α : Type u
      inst✝ : PseudoEMetricSpace α
      E : Set α
      s : Set Real
      hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
      hs₀ : HasSubset.Subset s (Set.Ioi 0)
      ⊢ Eq (Metric.cthickening 0 E) (Set.iInter fun δ => Set.iInter fun h => Metric. …
    -/
    apply cthickening_eq_iInter_cthickening' _ hs₀ hs
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening δ …
  -/
  obtain ⟨δ, hδs, δ_nonpos⟩ := not_subset.mp hs₀
  /-
    case neg.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
    δ : Real
    hδs : Membership.mem s δ
    δ_nonpos : Not (Membership.mem (Set.Ioi 0) δ)
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening δ …
  -/
  rw [Set.mem_Ioi, not_lt] at δ_nonpos
  /-
    case neg.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
    δ : Real
    hδs : Membership.mem s δ
    δ_nonpos : LE.le δ 0
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening δ …
  -/
  apply Subset.antisymm
    /-
      case neg.intro.intro.h₁
      α : Type u
      inst✝ : PseudoEMetricSpace α
      E : Set α
      s : Set Real
      hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
      hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
      δ : Real
      hδs : Membership.mem s δ
      δ_nonpos : LE.le δ 0
      ⊢ HasSubset.Subset (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric …
    -/
  · exact subset_iInter₂ fun ε _ => closure_subset_cthickening ε E
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      E : Set α
      s : Set Real
      hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
      hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
      δ : Real
      hδs : Membership.mem s δ
      δ_nonpos : LE.le δ 0
      ⊢ HasSubset.Subset (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening …
    -/
  · rw [← cthickening_of_nonpos δ_nonpos E]
    /-
      case neg.intro.intro.h₂
      α : Type u
      inst✝ : PseudoEMetricSpace α
      E : Set α
      s : Set Real
      hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
      hs₀ : Not (HasSubset.Subset s (Set.Ioi 0))
      δ : Real
      hδs : Membership.mem s δ
      δ_nonpos : LE.le δ 0
      ⊢ HasSubset.Subset (Set.iInter fun δ => Set.iInter fun h => Metric.cthickening …
    -/
    exact biInter_subset_of_mem hδs
    /-
      🎉 no goals
    -/


/-- The closure of a set equals the intersection of its closed thickenings of positive radii. -/
theorem closure_eq_iInter_cthickening (E : Set α) :
    closure E = ⋂ (δ : ℝ) (_ : 0 < δ), cthickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun x => Metric.cthickening δ …
  -/
  rw [← cthickening_zero]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    ⊢ Eq (Metric.cthickening 0 E) (Set.iInter fun δ => Set.iInter fun x => Metric. …
  -/
  exact cthickening_eq_iInter_cthickening E
  /-
    🎉 no goals
  -/


/-- The closure of a set equals the intersection of its open thickenings of positive radii
accumulating at zero. -/
theorem closure_eq_iInter_thickening' (E : Set α) (s : Set ℝ) (hs₀ : s ⊆ Ioi 0)
    (hs : ∀ ε, 0 < ε → (s ∩ Ioc 0 ε).Nonempty) : closure E = ⋂ δ ∈ s, thickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs₀ : HasSubset.Subset s (Set.Ioi 0)
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun h => Metric.thickening δ E)
  -/
  rw [← cthickening_zero]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    s : Set Real
    hs₀ : HasSubset.Subset s (Set.Ioi 0)
    hs : ∀ (ε : Real), LT.lt 0 ε → (Inter.inter s (Set.Ioc 0 ε)).Nonempty
    ⊢ Eq (Metric.cthickening 0 E) (Set.iInter fun δ => Set.iInter fun h => Metric. …
  -/
  apply cthickening_eq_iInter_thickening' le_rfl _ hs₀ hs
  /-
    🎉 no goals
  -/


/-- The closure of a set equals the intersection of its (open) thickenings of positive radii. -/
theorem closure_eq_iInter_thickening (E : Set α) :
    closure E = ⋂ (δ : ℝ) (_ : 0 < δ), thickening δ E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    ⊢ Eq (closure E) (Set.iInter fun δ => Set.iInter fun x => Metric.thickening δ E)
  -/
  rw [← cthickening_zero]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    E : Set α
    ⊢ Eq (Metric.cthickening 0 E) (Set.iInter fun δ => Set.iInter fun x => Metric. …
  -/
  exact cthickening_eq_iInter_thickening rfl.ge E
  /-
    🎉 no goals
  -/


/-- The frontier of the closed thickening of a set is contained in an `EMetric.infEdist` level
set. -/
theorem frontier_cthickening_subset (E : Set α) {δ : ℝ} :
    frontier (cthickening δ E) ⊆ { x : α | infEdist x E = ENNReal.ofReal δ } :=
  frontier_le_subset_eq continuous_infEdist continuous_const


/-- The closed ball of radius `δ` centered at a point of `E` is included in the closed
thickening of `E`. -/
theorem closedBall_subset_cthickening {α : Type*} [PseudoMetricSpace α] {x : α} {E : Set α}
    (hx : x ∈ E) (δ : ℝ) : closedBall x δ ⊆ cthickening δ E := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    E : Set α
    hx : Membership.mem E x
    δ : Real
    ⊢ HasSubset.Subset (Metric.closedBall x δ) (Metric.cthickening δ E)
  -/
  refine (closedBall_subset_cthickening_singleton _ _).trans (cthickening_subset_of_subset _ ?_)
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    E : Set α
    hx : Membership.mem E x
    δ : Real
    ⊢ HasSubset.Subset (Singleton.singleton x) E
  -/
  simpa using hx
  /-
    🎉 no goals
  -/


theorem cthickening_subset_iUnion_closedBall_of_lt {α : Type*} [PseudoMetricSpace α] (E : Set α)
    {δ δ' : ℝ} (hδ₀ : 0 < δ') (hδδ' : δ < δ') : cthickening δ E ⊆ ⋃ x ∈ E, closedBall x δ' := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    E : Set α
    δ δ' : Real
    hδ₀ : LT.lt 0 δ'
    hδδ' : LT.lt δ δ'
    ⊢ HasSubset.Subset (Metric.cthickening δ E) (Set.iUnion fun x => Set.iUnion fu …
  -/
  refine (cthickening_subset_thickening' hδ₀ hδδ' E).trans fun x hx => ?_
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    E : Set α
    δ δ' : Real
    hδ₀ : LT.lt 0 δ'
    hδδ' : LT.lt δ δ'
    x : α
    hx : Membership.mem (Metric.thickening δ' E) x
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  obtain ⟨y, hy₁, hy₂⟩ := mem_thickening_iff.mp hx
  /-
    case intro.intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    E : Set α
    δ δ' : Real
    hδ₀ : LT.lt 0 δ'
    hδδ' : LT.lt δ δ'
    x : α
    hx : Membership.mem (Metric.thickening δ' E) x
    y : α
    hy₁ : Membership.mem E y
    hy₂ : LT.lt (Dist.dist x y) δ'
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  exact mem_iUnion₂.mpr ⟨y, hy₁, hy₂.le⟩
  /-
    🎉 no goals
  -/


/-- The closed thickening of a compact set `E` is the union of the balls `Metric.closedBall x δ`
over `x ∈ E`.

See also `Metric.cthickening_eq_biUnion_closedBall`. -/
theorem _root_.IsCompact.cthickening_eq_biUnion_closedBall {α : Type*} [PseudoMetricSpace α]
    {δ : ℝ} {E : Set α} (hE : IsCompact E) (hδ : 0 ≤ δ) :
    cthickening δ E = ⋃ x ∈ E, closedBall x δ := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    hE : IsCompact E
    hδ : LE.le 0 δ
    ⊢ Eq (Metric.cthickening δ E) (Set.iUnion fun x => Set.iUnion fun h => Metric. …
  -/
  rcases eq_empty_or_nonempty E with (rfl | hne)
    /-
      case inl
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      δ : Real
      hδ : LE.le 0 δ
      hE : IsCompact EmptyCollection.emptyCollection
      ⊢ Eq (Metric.cthickening δ EmptyCollection.emptyCollection) (Set.iUnion fun x  …
    -/
  · simp only [cthickening_empty, biUnion_empty]
    /-
      🎉 no goals
    -/
  refine Subset.antisymm (fun x hx ↦ ?_)
    (iUnion₂_subset fun x hx ↦ closedBall_subset_cthickening hx _)
  /-
    case inr
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    hE : IsCompact E
    hδ : LE.le 0 δ
    hne : E.Nonempty
    x : α
    hx : Membership.mem (Metric.cthickening δ E) x
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  obtain ⟨y, yE, hy⟩ : ∃ y ∈ E, infEdist x E = edist x y := hE.exists_infEdist_eq_edist hne _
  /-
    case inr.intro.intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    hE : IsCompact E
    hδ : LE.le 0 δ
    hne : E.Nonempty
    x : α
    hx : Membership.mem (Metric.cthickening δ E) x
    y : α
    yE : Membership.mem E y
    hy : Eq (EMetric.infEdist x E) (EDist.edist x y)
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  have D1 : edist x y ≤ ENNReal.ofReal δ := (le_of_eq hy.symm).trans hx
  have D2 : dist x y ≤ δ := by
    rw [edist_dist] at D1
    exact (ENNReal.ofReal_le_ofReal_iff hδ).1 D1
  /-
    case inr.intro.intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    δ : Real
    E : Set α
    hE : IsCompact E
    hδ : LE.le 0 δ
    hne : E.Nonempty
    x : α
    hx : Membership.mem (Metric.cthickening δ E) x
    y : α
    yE : Membership.mem E y
    hy : Eq (EMetric.infEdist x E) (EDist.edist x y)
    D1 : LE.le (EDist.edist x y) (ENNReal.ofReal δ)
    D2 : LE.le (Dist.dist x y) δ
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  exact mem_biUnion yE D2
  /-
    🎉 no goals
  -/


theorem cthickening_eq_biUnion_closedBall {α : Type*} [PseudoMetricSpace α] [ProperSpace α]
    (E : Set α) (hδ : 0 ≤ δ) : cthickening δ E = ⋃ x ∈ closure E, closedBall x δ := by
  /-
    δ : Real
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    E : Set α
    hδ : LE.le 0 δ
    ⊢ Eq (Metric.cthickening δ E) (Set.iUnion fun x => Set.iUnion fun h => Metric. …
  -/
  rcases eq_empty_or_nonempty E with (rfl | hne)
    /-
      case inl
      δ : Real
      α : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      hδ : LE.le 0 δ
      ⊢ Eq (Metric.cthickening δ EmptyCollection.emptyCollection) (Set.iUnion fun x  …
    -/
  · simp only [cthickening_empty, biUnion_empty, closure_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    δ : Real
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    E : Set α
    hδ : LE.le 0 δ
    hne : E.Nonempty
    ⊢ Eq (Metric.cthickening δ E) (Set.iUnion fun x => Set.iUnion fun h => Metric. …
  -/
  rw [← cthickening_closure]
  refine Subset.antisymm (fun x hx ↦ ?_)
    (iUnion₂_subset fun x hx ↦ closedBall_subset_cthickening hx _)
  obtain ⟨y, yE, hy⟩ : ∃ y ∈ closure E, infDist x (closure E) = dist x y :=
    isClosed_closure.exists_infDist_eq_dist (closure_nonempty_iff.mpr hne) x
  replace hy : dist x y ≤ δ :=
    (ENNReal.ofReal_le_ofReal_iff hδ).mp
      (((congr_arg ENNReal.ofReal hy.symm).le.trans ENNReal.ofReal_toReal_le).trans hx)
  /-
    case inr.intro.intro
    δ : Real
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    E : Set α
    hδ : LE.le 0 δ
    hne : E.Nonempty
    x : α
    hx : Membership.mem (Metric.cthickening δ (closure E)) x
    y : α
    yE : Membership.mem (closure E) y
    hy : LE.le (Dist.dist x y) δ
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
  -/
  exact mem_biUnion yE hy
  /-
    🎉 no goals
  -/


nonrec theorem _root_.IsClosed.cthickening_eq_biUnion_closedBall {α : Type*} [PseudoMetricSpace α]
    [ProperSpace α] {E : Set α} (hE : IsClosed E) (hδ : 0 ≤ δ) :
    cthickening δ E = ⋃ x ∈ E, closedBall x δ := by
  /-
    δ : Real
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    E : Set α
    hE : IsClosed E
    hδ : LE.le 0 δ
    ⊢ Eq (Metric.cthickening δ E) (Set.iUnion fun x => Set.iUnion fun h => Metric. …
  -/
  rw [cthickening_eq_biUnion_closedBall E hδ, hE.closure_eq]
  /-
    🎉 no goals
  -/


/-- For the equality, see `infEdist_cthickening`. -/
theorem infEdist_le_infEdist_cthickening_add :
    infEdist x s ≤ infEdist x (cthickening δ s) + ENNReal.ofReal δ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set α
    x : α
    ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist x (Metric.cthicken …
  -/
  refine le_of_forall_lt' fun r h => ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set α
    x : α
    r : ENNReal
    h : LT.lt (HAdd.hAdd (EMetric.infEdist x (Metric.cthickening δ s)) (ENNReal.of …
    ⊢ LT.lt (EMetric.infEdist x s) r
  -/
  simp_rw [← lt_tsub_iff_right, infEdist_lt_iff, mem_cthickening_iff] at h
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ : Real
    s : Set α
    x : α
    r : ENNReal
    h : Exists fun y => And (LE.le (EMetric.infEdist y s) (ENNReal.ofReal δ)) (LT. …
    ⊢ LT.lt (EMetric.infEdist x s) r
  -/
  obtain ⟨y, hy, hxy⟩ := h
  exact infEdist_le_edist_add_infEdist.trans_lt
    ((ENNReal.add_lt_add_of_lt_of_le (hy.trans_lt ENNReal.ofReal_lt_top).ne hxy hy).trans_eq
      (tsub_add_cancel_of_le <| le_self_add.trans (lt_tsub_iff_left.1 hxy).le))


/-- For the equality, see `infEdist_thickening`. -/
theorem infEdist_le_infEdist_thickening_add :
    infEdist x s ≤ infEdist x (thickening δ s) + ENNReal.ofReal δ :=
  infEdist_le_infEdist_cthickening_add.trans <|
    add_le_add_right (infEdist_anti <| thickening_subset_cthickening _ _) _


/-- For the equality, see `thickening_thickening`. -/
@[simp]
theorem thickening_thickening_subset (ε δ : ℝ) (s : Set α) :
    thickening ε (thickening δ s) ⊆ thickening (ε + δ) s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε δ : Real
    s : Set α
    ⊢ HasSubset.Subset (Metric.thickening ε (Metric.thickening δ s)) (Metric.thick …
  -/
  obtain hε | hε := le_total ε 0
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      ε δ : Real
      s : Set α
      hε : LE.le ε 0
      ⊢ HasSubset.Subset (Metric.thickening ε (Metric.thickening δ s)) (Metric.thick …
    -/
  · simp only [thickening_of_nonpos hε, empty_subset]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε δ : Real
    s : Set α
    hε : LE.le 0 ε
    ⊢ HasSubset.Subset (Metric.thickening ε (Metric.thickening δ s)) (Metric.thick …
  -/
  obtain hδ | hδ := le_total δ 0
    /-
      case inr.inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      ε δ : Real
      s : Set α
      hε : LE.le 0 ε
      hδ : LE.le δ 0
      ⊢ HasSubset.Subset (Metric.thickening ε (Metric.thickening δ s)) (Metric.thick …
    -/
  · simp only [thickening_of_nonpos hδ, thickening_empty, empty_subset]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε δ : Real
    s : Set α
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    ⊢ HasSubset.Subset (Metric.thickening ε (Metric.thickening δ s)) (Metric.thick …
  -/
  intro x
  /-
    case inr.inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε δ : Real
    s : Set α
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    x : α
    ⊢ Membership.mem (Metric.thickening ε (Metric.thickening δ s)) x → Membership. …
  -/
  simp_rw [mem_thickening_iff_exists_edist_lt, ENNReal.ofReal_add hε hδ]
  exact fun ⟨y, ⟨z, hz, hy⟩, hx⟩ =>
    ⟨z, hz, (edist_triangle _ _ _).trans_lt <| ENNReal.add_lt_add hx hy⟩


/-- For the equality, see `thickening_cthickening`. -/
@[simp]
theorem thickening_cthickening_subset (ε : ℝ) (hδ : 0 ≤ δ) (s : Set α) :
    thickening ε (cthickening δ s) ⊆ thickening (ε + δ) s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hδ : LE.le 0 δ
    s : Set α
    ⊢ HasSubset.Subset (Metric.thickening ε (Metric.cthickening δ s)) (Metric.thic …
  -/
  obtain hε | hε := le_total ε 0
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      δ ε : Real
      hδ : LE.le 0 δ
      s : Set α
      hε : LE.le ε 0
      ⊢ HasSubset.Subset (Metric.thickening ε (Metric.cthickening δ s)) (Metric.thic …
    -/
  · simp only [thickening_of_nonpos hε, empty_subset]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hδ : LE.le 0 δ
    s : Set α
    hε : LE.le 0 ε
    ⊢ HasSubset.Subset (Metric.thickening ε (Metric.cthickening δ s)) (Metric.thic …
  -/
  intro x
  simp_rw [mem_thickening_iff_exists_edist_lt, mem_cthickening_iff, ← infEdist_lt_iff,
    ENNReal.ofReal_add hε hδ]
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hδ : LE.le 0 δ
    s : Set α
    hε : LE.le 0 ε
    x : α
    ⊢ (Exists fun z => And (LE.le (EMetric.infEdist z s) (ENNReal.ofReal δ)) (LT.l …
  -/
  rintro ⟨y, hy, hxy⟩
  exact infEdist_le_edist_add_infEdist.trans_lt
    (ENNReal.add_lt_add_of_lt_of_le (hy.trans_lt ENNReal.ofReal_lt_top).ne hxy hy)


/-- For the equality, see `cthickening_thickening`. -/
@[simp]
theorem cthickening_thickening_subset (hε : 0 ≤ ε) (δ : ℝ) (s : Set α) :
    cthickening ε (thickening δ s) ⊆ cthickening (ε + δ) s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε : Real
    hε : LE.le 0 ε
    δ : Real
    s : Set α
    ⊢ HasSubset.Subset (Metric.cthickening ε (Metric.thickening δ s)) (Metric.cthi …
  -/
  obtain hδ | hδ := le_total δ 0
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      ε : Real
      hε : LE.le 0 ε
      δ : Real
      s : Set α
      hδ : LE.le δ 0
      ⊢ HasSubset.Subset (Metric.cthickening ε (Metric.thickening δ s)) (Metric.cthi …
    -/
  · simp only [thickening_of_nonpos hδ, cthickening_empty, empty_subset]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε : Real
    hε : LE.le 0 ε
    δ : Real
    s : Set α
    hδ : LE.le 0 δ
    ⊢ HasSubset.Subset (Metric.cthickening ε (Metric.thickening δ s)) (Metric.cthi …
  -/
  intro x
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε : Real
    hε : LE.le 0 ε
    δ : Real
    s : Set α
    hδ : LE.le 0 δ
    x : α
    ⊢ Membership.mem (Metric.cthickening ε (Metric.thickening δ s)) x → Membership …
  -/
  simp_rw [mem_cthickening_iff, ENNReal.ofReal_add hε hδ]
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    ε : Real
    hε : LE.le 0 ε
    δ : Real
    s : Set α
    hδ : LE.le 0 δ
    x : α
    ⊢ LE.le (EMetric.infEdist x (Metric.thickening δ s)) (ENNReal.ofReal ε) → LE.l …
  -/
  exact fun hx => infEdist_le_infEdist_thickening_add.trans (add_le_add_right hx _)
  /-
    🎉 no goals
  -/


/-- For the equality, see `cthickening_cthickening`. -/
@[simp]
theorem cthickening_cthickening_subset (hε : 0 ≤ ε) (hδ : 0 ≤ δ) (s : Set α) :
    cthickening ε (cthickening δ s) ⊆ cthickening (ε + δ) s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    s : Set α
    ⊢ HasSubset.Subset (Metric.cthickening ε (Metric.cthickening δ s)) (Metric.cth …
  -/
  intro x
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    s : Set α
    x : α
    ⊢ Membership.mem (Metric.cthickening ε (Metric.cthickening δ s)) x → Membershi …
  -/
  simp_rw [mem_cthickening_iff, ENNReal.ofReal_add hε hδ]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    δ ε : Real
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    s : Set α
    x : α
    ⊢ LE.le (EMetric.infEdist x (Metric.cthickening δ s)) (ENNReal.ofReal ε) → LE. …
  -/
  exact fun hx => infEdist_le_infEdist_cthickening_add.trans (add_le_add_right hx _)
  /-
    🎉 no goals
  -/


theorem frontier_cthickening_disjoint (A : Set α) :
    Pairwise (Disjoint on fun r : ℝ≥0 => frontier (cthickening r A)) := fun r₁ r₂ hr =>
                               /-
                                 α : Type u
                                 inst✝ : PseudoEMetricSpace α
                                 A : Set α
                                 r₁ r₂ : NNReal
                                 hr : Ne r₁ r₂
                                 ⊢ Ne (ENNReal.ofReal ↑r₁) (ENNReal.ofReal ↑r₂)
                               -/
  ((disjoint_singleton.2 <| by simpa).preimage _).mono (frontier_cthickening_subset _)
                               /-
                                 🎉 no goals
                               -/
    (frontier_cthickening_subset _)


