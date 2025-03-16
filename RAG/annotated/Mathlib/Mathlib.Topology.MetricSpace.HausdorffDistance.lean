/-- The minimal edistance of a point to a set -/
def infEdist (x : α) (s : Set α) : ℝ≥0∞ :=
  ⨅ y ∈ s, edist x y


@[simp]
theorem infEdist_empty : infEdist x ∅ = ∞ :=
  iInf_emptyset


theorem le_infEdist {d} : d ≤ infEdist x s ↔ ∀ y ∈ s, d ≤ edist x y := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    d : ENNReal
    ⊢ Iff (LE.le d (EMetric.infEdist x s)) (∀ (y : α), Membership.mem s y → LE.le  …
  -/
  simp only [infEdist, le_iInf_iff]
  /-
    🎉 no goals
  -/


/-- The edist to a union is the minimum of the edists -/
@[simp]
theorem infEdist_union : infEdist x (s ∪ t) = infEdist x s ⊓ infEdist x t :=
  iInf_union


@[simp]
theorem infEdist_iUnion (f : ι → Set α) (x : α) : infEdist x (⋃ i, f i) = ⨅ i, infEdist x (f i) :=
  iInf_iUnion f _


lemma infEdist_biUnion {ι : Type*} (f : ι → Set α) (I : Set ι) (x : α) :
                                                                /-
                                                                  α : Type u
                                                                  inst✝ : PseudoEMetricSpace α
                                                                  ι : Type u_2
                                                                  f : ι → Set α
                                                                  I : Set ι
                                                                  x : α
                                                                  ⊢ Eq (EMetric.infEdist x (Set.iUnion fun i => Set.iUnion fun h => f i)) (iInf  …
                                                                -/
    infEdist x (⋃ i ∈ I, f i) = ⨅ i ∈ I, infEdist x (f i) := by simp only [infEdist_iUnion]
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The edist to a singleton is the edistance to the single point of this singleton -/
@[simp]
theorem infEdist_singleton : infEdist x {y} = edist x y :=
  iInf_singleton


/-- The edist to a set is bounded above by the edist to any of its points -/
theorem infEdist_le_edist_of_mem (h : y ∈ s) : infEdist x s ≤ edist x y :=
  iInf₂_le y h


/-- If a point `x` belongs to `s`, then its edist to `s` vanishes -/
theorem infEdist_zero_of_mem (h : x ∈ s) : infEdist x s = 0 :=
  nonpos_iff_eq_zero.1 <| @edist_self _ _ x ▸ infEdist_le_edist_of_mem h


/-- The edist is antitone with respect to inclusion. -/
theorem infEdist_anti (h : s ⊆ t) : infEdist x t ≤ infEdist x s :=
  iInf_le_iInf_of_subset h


/-- The edist to a set is `< r` iff there exists a point in the set at edistance `< r` -/
theorem infEdist_lt_iff {r : ℝ≥0∞} : infEdist x s < r ↔ ∃ y ∈ s, edist x y < r := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    r : ENNReal
    ⊢ Iff (LT.lt (EMetric.infEdist x s) r) (Exists fun y => And (Membership.mem s  …
  -/
  simp_rw [infEdist, iInf_lt_iff, exists_prop]
  /-
    🎉 no goals
  -/


/-- The edist of `x` to `s` is bounded by the sum of the edist of `y` to `s` and
the edist from `x` to `y` -/
theorem infEdist_le_infEdist_add_edist : infEdist x s ≤ infEdist y s + edist x y :=
  calc
    ⨅ z ∈ s, edist x z ≤ ⨅ z ∈ s, edist y z + edist x y :=
      iInf₂_mono fun _ _ => (edist_triangle _ _ _).trans_eq (add_comm _ _)
                                               /-
                                                 α : Type u
                                                 inst✝ : PseudoEMetricSpace α
                                                 x y : α
                                                 s : Set α
                                                 ⊢ Eq (iInf fun z => iInf fun h => HAdd.hAdd (EDist.edist y z) (EDist.edist x y …
                                               -/
    _ = (⨅ z ∈ s, edist y z) + edist x y := by simp only [ENNReal.iInf_add]
                                               /-
                                                 🎉 no goals
                                               -/


theorem infEdist_le_edist_add_infEdist : infEdist x s ≤ edist x y + infEdist y s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    s : Set α
    ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (EDist.edist x y) (EMetric.infEdist  …
  -/
  rw [add_comm]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    s : Set α
    ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist y s) (EDist.edist  …
  -/
  exact infEdist_le_infEdist_add_edist
  /-
    🎉 no goals
  -/


theorem edist_le_infEdist_add_ediam (hy : y ∈ s) : edist x y ≤ infEdist x s + diam s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    s : Set α
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (EMetric.infEdist x s) (EMetric.diam s))
  -/
  simp_rw [infEdist, ENNReal.iInf_add]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    s : Set α
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist x y) (iInf fun i => iInf fun i_1 => HAdd.hAdd (EDist.edis …
  -/
  refine le_iInf₂ fun i hi => ?_
  calc
    edist x y ≤ edist x i + edist i y := edist_triangle _ _ _
    _ ≤ edist x i + diam s := add_le_add le_rfl (edist_le_diam_of_mem hi hy)


/-- The edist to a set depends continuously on the point -/
@[continuity]
theorem continuous_infEdist : Continuous fun x => infEdist x s :=
                                   /-
                                     α : Type u
                                     inst✝ : PseudoEMetricSpace α
                                     s : Set α
                                     ⊢ Ne 1 Top.top
                                   -/
  continuous_of_le_add_edist 1 (by simp) <| by
                                   /-
                                     🎉 no goals
                                   -/
    /-
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s : Set α
      ⊢ ∀ (x y : α), LE.le (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist y s)  …
    -/
    simp only [one_mul, infEdist_le_infEdist_add_edist, forall₂_true_iff]
    /-
      🎉 no goals
    -/


/-- The edist to a set and to its closure coincide -/
theorem infEdist_closure : infEdist x (closure s) = infEdist x s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ⊢ Eq (EMetric.infEdist x (closure s)) (EMetric.infEdist x s)
  -/
  refine le_antisymm (infEdist_anti subset_closure) ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ⊢ LE.le (EMetric.infEdist x s) (EMetric.infEdist x (closure s))
  -/
  refine ENNReal.le_of_forall_pos_le_add fun ε εpos h => ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (EMetric.infEdist x (closure s)) Top.top
    ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist x (closure s)) ↑ε)
  -/
  have ε0 : 0 < (ε / 2 : ℝ≥0∞) := by simpa [pos_iff_ne_zero] using εpos
  have : infEdist x (closure s) < infEdist x (closure s) + ε / 2 :=
    ENNReal.lt_add_right h.ne ε0.ne'
  obtain ⟨y : α, ycs : y ∈ closure s, hy : edist x y < infEdist x (closure s) + ↑ε / 2⟩ :=
    infEdist_lt_iff.mp this
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (EMetric.infEdist x (closure s)) Top.top
    ε0 : LT.lt 0 (HDiv.hDiv (↑ε) 2)
    this : LT.lt (EMetric.infEdist x (closure s)) (HAdd.hAdd (EMetric.infEdist x ( …
    y : α
    ycs : Membership.mem (closure s) y
    hy : LT.lt (EDist.edist x y) (HAdd.hAdd (EMetric.infEdist x (closure s)) (HDiv …
    ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist x (closure s)) ↑ε)
  -/
  obtain ⟨z : α, zs : z ∈ s, dyz : edist y z < ↑ε / 2⟩ := EMetric.mem_closure_iff.1 ycs (ε / 2) ε0
  calc
    infEdist x s ≤ edist x z := infEdist_le_edist_of_mem zs
    _ ≤ edist x y + edist y z := edist_triangle _ _ _
    _ ≤ infEdist x (closure s) + ε / 2 + ε / 2 := add_le_add (le_of_lt hy) (le_of_lt dyz)
    _ = infEdist x (closure s) + ↑ε := by rw [add_assoc, ENNReal.add_halves]


/-- A point belongs to the closure of `s` iff its infimum edistance to this set vanishes -/
theorem mem_closure_iff_infEdist_zero : x ∈ closure s ↔ infEdist x s = 0 :=
  ⟨fun h => by
    /-
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x : α
      s : Set α
      h : Membership.mem (closure s) x
      ⊢ Eq (EMetric.infEdist x s) 0
    -/
    rw [← infEdist_closure]
    /-
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x : α
      s : Set α
      h : Membership.mem (closure s) x
      ⊢ Eq (EMetric.infEdist x (closure s)) 0
    -/
    exact infEdist_zero_of_mem h,
    /-
      🎉 no goals
    -/
   fun h =>
                                                                     /-
                                                                       α : Type u
                                                                       inst✝ : PseudoEMetricSpace α
                                                                       x : α
                                                                       s : Set α
                                                                       h : Eq (EMetric.infEdist x s) 0
                                                                       ε : ENNReal
                                                                       εpos : GT.gt ε 0
                                                                       ⊢ LT.lt (EMetric.infEdist x s) ε
                                                                     -/
    EMetric.mem_closure_iff.2 fun ε εpos => infEdist_lt_iff.mp <| by rwa [h]⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- Given a closed set `s`, a point belongs to `s` iff its infimum edistance to this set vanishes -/
theorem mem_iff_infEdist_zero_of_closed (h : IsClosed s) : x ∈ s ↔ infEdist x s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    h : IsClosed s
    ⊢ Iff (Membership.mem s x) (Eq (EMetric.infEdist x s) 0)
  -/
  rw [← mem_closure_iff_infEdist_zero, h.closure_eq]
  /-
    🎉 no goals
  -/


/-- The infimum edistance of a point to a set is positive if and only if the point is not in the
closure of the set. -/
theorem infEdist_pos_iff_not_mem_closure {x : α} {E : Set α} :
    0 < infEdist x E ↔ x ∉ closure E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    E : Set α
    ⊢ Iff (LT.lt 0 (EMetric.infEdist x E)) (Not (Membership.mem (closure E) x))
  -/
  rw [mem_closure_iff_infEdist_zero, pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem infEdist_closure_pos_iff_not_mem_closure {x : α} {E : Set α} :
    0 < infEdist x (closure E) ↔ x ∉ closure E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    E : Set α
    ⊢ Iff (LT.lt 0 (EMetric.infEdist x (closure E))) (Not (Membership.mem (closure …
  -/
  rw [infEdist_closure, infEdist_pos_iff_not_mem_closure]
  /-
    🎉 no goals
  -/


theorem exists_real_pos_lt_infEdist_of_not_mem_closure {x : α} {E : Set α} (h : x ∉ closure E) :
    ∃ ε : ℝ, 0 < ε ∧ ENNReal.ofReal ε < infEdist x E := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    E : Set α
    h : Not (Membership.mem (closure E) x)
    ⊢ Exists fun ε => And (LT.lt 0 ε) (LT.lt (ENNReal.ofReal ε) (EMetric.infEdist  …
  -/
  rw [← infEdist_pos_iff_not_mem_closure, ENNReal.lt_iff_exists_real_btwn] at h
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    E : Set α
    h : Exists fun r => And (LE.le 0 r) (And (LT.lt 0 (ENNReal.ofReal r)) (LT.lt ( …
    ⊢ Exists fun ε => And (LT.lt 0 ε) (LT.lt (ENNReal.ofReal ε) (EMetric.infEdist  …
  -/
  rcases h with ⟨ε, ⟨_, ⟨ε_pos, ε_lt⟩⟩⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    E : Set α
    ε : Real
    left✝ : LE.le 0 ε
    ε_pos : LT.lt 0 (ENNReal.ofReal ε)
    ε_lt : LT.lt (ENNReal.ofReal ε) (EMetric.infEdist x E)
    ⊢ Exists fun ε => And (LT.lt 0 ε) (LT.lt (ENNReal.ofReal ε) (EMetric.infEdist  …
  -/
  exact ⟨ε, ⟨ENNReal.ofReal_pos.mp ε_pos, ε_lt⟩⟩
  /-
    🎉 no goals
  -/


theorem disjoint_closedBall_of_lt_infEdist {r : ℝ≥0∞} (h : r < infEdist x s) :
    Disjoint (closedBall x r) s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    r : ENNReal
    h : LT.lt r (EMetric.infEdist x s)
    ⊢ Disjoint (EMetric.closedBall x r) s
  -/
  rw [disjoint_left]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    r : ENNReal
    h : LT.lt r (EMetric.infEdist x s)
    ⊢ ∀ ⦃a : α⦄, Membership.mem (EMetric.closedBall x r) a → Not (Membership.mem s …
  -/
  intro y hy h'y
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    r : ENNReal
    h : LT.lt r (EMetric.infEdist x s)
    y : α
    hy : Membership.mem (EMetric.closedBall x r) y
    h'y : Membership.mem s y
    ⊢ False
  -/
  apply lt_irrefl (infEdist x s)
  calc
    infEdist x s ≤ edist x y := infEdist_le_edist_of_mem h'y
    _ ≤ r := by rwa [mem_closedBall, edist_comm] at hy
    _ < infEdist x s := h


/-- The infimum edistance is invariant under isometries -/
theorem infEdist_image (hΦ : Isometry Φ) : infEdist (Φ x) (Φ '' t) = infEdist x t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    x : α
    t : Set α
    Φ : α → β
    hΦ : Isometry Φ
    ⊢ Eq (EMetric.infEdist (Φ x) (Set.image Φ t)) (EMetric.infEdist x t)
  -/
  simp only [infEdist, iInf_image, hΦ.edist_eq]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem infEdist_smul {M} [SMul M α] [IsometricSMul M α] (c : M) (x : α) (s : Set α) :
    infEdist (c • x) (c • s) = infEdist x s :=
  infEdist_image (isometry_smul _ _)


theorem _root_.IsOpen.exists_iUnion_isClosed {U : Set α} (hU : IsOpen U) :
    ∃ F : ℕ → Set α, (∀ n, IsClosed (F n)) ∧ (∀ n, F n ⊆ U) ∧ ⋃ n, F n = U ∧ Monotone F := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    ⊢ Exists fun F => And (∀ (n : Nat), IsClosed (F n)) (And (∀ (n : Nat), HasSubs …
  -/
  obtain ⟨a, a_pos, a_lt_one⟩ : ∃ a : ℝ≥0∞, 0 < a ∧ a < 1 := exists_between zero_lt_one
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    ⊢ Exists fun F => And (∀ (n : Nat), IsClosed (F n)) (And (∀ (n : Nat), HasSubs …
  -/
  let F := fun n : ℕ => (fun x => infEdist x Uᶜ) ⁻¹' Ici (a ^ n)
  have F_subset : ∀ n, F n ⊆ U := fun n x hx ↦ by
    by_contra h
    have : infEdist x Uᶜ ≠ 0 := ((ENNReal.pow_pos a_pos _).trans_le hx).ne'
    exact this (infEdist_zero_of_mem h)
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
    F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
    ⊢ Exists fun F => And (∀ (n : Nat), IsClosed (F n)) (And (∀ (n : Nat), HasSubs …
  -/
  refine ⟨F, fun n => IsClosed.preimage continuous_infEdist isClosed_Ici, F_subset, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      ⊢ Eq (Set.iUnion fun n => F n) U
    -/
  · show ⋃ n, F n = U
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      ⊢ Eq (Set.iUnion fun n => F n) U
    -/
    refine Subset.antisymm (by simp only [iUnion_subset_iff, F_subset, forall_const]) fun x hx => ?_
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      ⊢ Membership.mem (Set.iUnion fun n => F n) x
    -/
    have : ¬x ∈ Uᶜ := by simpa using hx
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      this : Not (Membership.mem (HasCompl.compl U) x)
      ⊢ Membership.mem (Set.iUnion fun n => F n) x
    -/
    rw [mem_iff_infEdist_zero_of_closed hU.isClosed_compl] at this
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      this : Not (Eq (EMetric.infEdist x (HasCompl.compl U)) 0)
      ⊢ Membership.mem (Set.iUnion fun n => F n) x
    -/
    have B : 0 < infEdist x Uᶜ := by simpa [pos_iff_ne_zero] using this
    have : Filter.Tendsto (fun n => a ^ n) atTop (𝓝 0) :=
      ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one a_lt_one
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      this✝ : Not (Eq (EMetric.infEdist x (HasCompl.compl U)) 0)
      B : LT.lt 0 (EMetric.infEdist x (HasCompl.compl U))
      this : Filter.Tendsto (fun n => HPow.hPow a n) Filter.atTop (nhds 0)
      ⊢ Membership.mem (Set.iUnion fun n => F n) x
    -/
    rcases ((tendsto_order.1 this).2 _ B).exists with ⟨n, hn⟩
    /-
      case intro.intro.refine_1.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      this✝ : Not (Eq (EMetric.infEdist x (HasCompl.compl U)) 0)
      B : LT.lt 0 (EMetric.infEdist x (HasCompl.compl U))
      this : Filter.Tendsto (fun n => HPow.hPow a n) Filter.atTop (nhds 0)
      n : Nat
      hn : LT.lt (HPow.hPow a n) (EMetric.infEdist x (HasCompl.compl U))
      ⊢ Membership.mem (Set.iUnion fun n => F n) x
    -/
    simp only [mem_iUnion, mem_Ici, mem_preimage]
    /-
      case intro.intro.refine_1.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      U : Set α
      hU : IsOpen U
      a : ENNReal
      a_pos : LT.lt 0 a
      a_lt_one : LT.lt a 1
      F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
      F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
      x : α
      hx : Membership.mem U x
      this✝ : Not (Eq (EMetric.infEdist x (HasCompl.compl U)) 0)
      B : LT.lt 0 (EMetric.infEdist x (HasCompl.compl U))
      this : Filter.Tendsto (fun n => HPow.hPow a n) Filter.atTop (nhds 0)
      n : Nat
      hn : LT.lt (HPow.hPow a n) (EMetric.infEdist x (HasCompl.compl U))
      ⊢ Exists fun i => Membership.mem (F i) x
    -/
    exact ⟨n, hn.le⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.refine_2
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
    F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
    ⊢ Monotone F
  -/
  show Monotone F
  /-
    case intro.intro.refine_2
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
    F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
    ⊢ Monotone F
  -/
  intro m n hmn x hx
  /-
    case intro.intro.refine_2
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
    F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
    m n : Nat
    hmn : LE.le m n
    x : α
    hx : Membership.mem (F m) x
    ⊢ Membership.mem (F n) x
  -/
  simp only [F, mem_Ici, mem_preimage] at hx ⊢
  /-
    case intro.intro.refine_2
    α : Type u
    inst✝ : PseudoEMetricSpace α
    U : Set α
    hU : IsOpen U
    a : ENNReal
    a_pos : LT.lt 0 a
    a_lt_one : LT.lt a 1
    F : Nat → Set α := fun n => Set.preimage (fun x => EMetric.infEdist x (HasComp …
    F_subset : ∀ (n : Nat), HasSubset.Subset (F n) U
    m n : Nat
    hmn : LE.le m n
    x : α
    hx : LE.le (HPow.hPow a m) (EMetric.infEdist x (HasCompl.compl U))
    ⊢ LE.le (HPow.hPow a n) (EMetric.infEdist x (HasCompl.compl U))
  -/
  apply le_trans (pow_le_pow_right_of_le_one' a_lt_one.le hmn) hx
  /-
    🎉 no goals
  -/


theorem _root_.IsCompact.exists_infEdist_eq_edist (hs : IsCompact s) (hne : s.Nonempty) (x : α) :
    ∃ y ∈ s, infEdist x s = edist x y := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    hne : s.Nonempty
    x : α
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (EMetric.infEdist x s) (EDist.e …
  -/
  have A : Continuous fun y => edist x y := continuous_const.edist continuous_id
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    hne : s.Nonempty
    x : α
    A : Continuous fun y => EDist.edist x y
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (EMetric.infEdist x s) (EDist.e …
  -/
  obtain ⟨y, ys, hy⟩ := hs.exists_isMinOn hne A.continuousOn
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    hne : s.Nonempty
    x : α
    A : Continuous fun y => EDist.edist x y
    y : α
    ys : Membership.mem s y
    hy : IsMinOn (fun y => EDist.edist x y) s y
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (EMetric.infEdist x s) (EDist.e …
  -/
  exact ⟨y, ys, le_antisymm (infEdist_le_edist_of_mem ys) (by rwa [le_infEdist])⟩
  /-
    🎉 no goals
  -/


theorem exists_pos_forall_lt_edist (hs : IsCompact s) (ht : IsClosed t) (hst : Disjoint s t) :
    ∃ r : ℝ≥0, 0 < r ∧ ∀ x ∈ s, ∀ y ∈ t, (r : ℝ≥0∞) < edist x y := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : IsCompact s
    ht : IsClosed t
    hst : Disjoint s t
    ⊢ Exists fun r => And (LT.lt 0 r) (∀ (x : α), Membership.mem s x → ∀ (y : α),  …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hne)
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      t : Set α
      ht : IsClosed t
      hs : IsCompact EmptyCollection.emptyCollection
      hst : Disjoint EmptyCollection.emptyCollection t
      ⊢ Exists fun r => And (LT.lt 0 r) (∀ (x : α), Membership.mem EmptyCollection.e …
    -/
  · use 1
    /-
      case h
      α : Type u
      inst✝ : PseudoEMetricSpace α
      t : Set α
      ht : IsClosed t
      hs : IsCompact EmptyCollection.emptyCollection
      hst : Disjoint EmptyCollection.emptyCollection t
      ⊢ And (LT.lt 0 1) (∀ (x : α), Membership.mem EmptyCollection.emptyCollection x …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : IsCompact s
    ht : IsClosed t
    hst : Disjoint s t
    hne : s.Nonempty
    ⊢ Exists fun r => And (LT.lt 0 r) (∀ (x : α), Membership.mem s x → ∀ (y : α),  …
  -/
  obtain ⟨x, hx, h⟩ := hs.exists_isMinOn hne continuous_infEdist.continuousOn
  have : 0 < infEdist x t :=
    pos_iff_ne_zero.2 fun H => hst.le_bot ⟨hx, (mem_iff_infEdist_zero_of_closed ht).mpr H⟩
  /-
    case inr.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : IsCompact s
    ht : IsClosed t
    hst : Disjoint s t
    hne : s.Nonempty
    x : α
    hx : Membership.mem s x
    h : IsMinOn (fun x => EMetric.infEdist x ?m.35738) s x
    this : LT.lt 0 (EMetric.infEdist x t)
    ⊢ Exists fun r => And (LT.lt 0 r) (∀ (x : α), Membership.mem s x → ∀ (y : α),  …
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 this with ⟨r, h₀, hr⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : IsCompact s
    ht : IsClosed t
    hst : Disjoint s t
    hne : s.Nonempty
    x : α
    hx : Membership.mem s x
    h : IsMinOn (fun x => EMetric.infEdist x ?m.35738) s x
    this : LT.lt 0 (EMetric.infEdist x t)
    r : NNReal
    h₀ : LT.lt 0 ↑r
    hr : LT.lt (↑r) (EMetric.infEdist x t)
    ⊢ Exists fun r => And (LT.lt 0 r) (∀ (x : α), Membership.mem s x → ∀ (y : α),  …
  -/
  exact ⟨r, ENNReal.coe_pos.mp h₀, fun y hy z hz => hr.trans_le <| le_infEdist.1 (h hy) z hz⟩
  /-
    🎉 no goals
  -/


/-- The Hausdorff edistance between two sets is the smallest `r` such that each set
is contained in the `r`-neighborhood of the other one -/
irreducible_def hausdorffEdist {α : Type u} [PseudoEMetricSpace α] (s t : Set α) : ℝ≥0∞ :=
  (⨆ x ∈ s, infEdist x t) ⊔ ⨆ y ∈ t, infEdist y s


/-- The Hausdorff edistance of a set to itself vanishes. -/
@[simp]
theorem hausdorffEdist_self : hausdorffEdist s s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ⊢ Eq (EMetric.hausdorffEdist s s) 0
  -/
  simp only [hausdorffEdist_def, sup_idem, ENNReal.iSup_eq_zero]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ⊢ ∀ (i : α), Membership.mem s i → Eq (EMetric.infEdist i s) 0
  -/
  exact fun x hx => infEdist_zero_of_mem hx
  /-
    🎉 no goals
  -/


/-- The Haudorff edistances of `s` to `t` and of `t` to `s` coincide. -/
theorem hausdorffEdist_comm : hausdorffEdist s t = hausdorffEdist t s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    ⊢ Eq (EMetric.hausdorffEdist s t) (EMetric.hausdorffEdist t s)
  -/
  simp only [hausdorffEdist_def]; apply sup_comm
                                  /-
                                    🎉 no goals
                                  -/


/-- Bounding the Hausdorff edistance by bounding the edistance of any point
in each set to the other set -/
theorem hausdorffEdist_le_of_infEdist {r : ℝ≥0∞} (H1 : ∀ x ∈ s, infEdist x t ≤ r)
    (H2 : ∀ x ∈ t, infEdist x s ≤ r) : hausdorffEdist s t ≤ r := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    r : ENNReal
    H1 : ∀ (x : α), Membership.mem s x → LE.le (EMetric.infEdist x t) r
    H2 : ∀ (x : α), Membership.mem t x → LE.le (EMetric.infEdist x s) r
    ⊢ LE.le (EMetric.hausdorffEdist s t) r
  -/
  simp only [hausdorffEdist_def, sup_le_iff, iSup_le_iff]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    r : ENNReal
    H1 : ∀ (x : α), Membership.mem s x → LE.le (EMetric.infEdist x t) r
    H2 : ∀ (x : α), Membership.mem t x → LE.le (EMetric.infEdist x s) r
    ⊢ And (∀ (i : α), Membership.mem s i → LE.le (EMetric.infEdist i t) r) (∀ (i : …
  -/
  exact ⟨H1, H2⟩
  /-
    🎉 no goals
  -/


/-- Bounding the Hausdorff edistance by exhibiting, for any point in each set,
another point in the other set at controlled distance -/
theorem hausdorffEdist_le_of_mem_edist {r : ℝ≥0∞} (H1 : ∀ x ∈ s, ∃ y ∈ t, edist x y ≤ r)
    (H2 : ∀ x ∈ t, ∃ y ∈ s, edist x y ≤ r) : hausdorffEdist s t ≤ r := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    r : ENNReal
    H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
    H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
    ⊢ LE.le (EMetric.hausdorffEdist s t) r
  -/
  refine hausdorffEdist_le_of_infEdist (fun x xs ↦ ?_) (fun x xt ↦ ?_)
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      r : ENNReal
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xs : Membership.mem s x
      ⊢ LE.le (EMetric.infEdist x t) r
    -/
  · rcases H1 x xs with ⟨y, yt, hy⟩
    /-
      case refine_1.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      r : ENNReal
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      hy : LE.le (EDist.edist x y) r
      ⊢ LE.le (EMetric.infEdist x t) r
    -/
    exact le_trans (infEdist_le_edist_of_mem yt) hy
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      r : ENNReal
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xt : Membership.mem t x
      ⊢ LE.le (EMetric.infEdist x s) r
    -/
  · rcases H2 x xt with ⟨y, ys, hy⟩
    /-
      case refine_2.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      r : ENNReal
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xt : Membership.mem t x
      y : α
      ys : Membership.mem s y
      hy : LE.le (EDist.edist x y) r
      ⊢ LE.le (EMetric.infEdist x s) r
    -/
    exact le_trans (infEdist_le_edist_of_mem ys) hy
    /-
      🎉 no goals
    -/


/-- The distance to a set is controlled by the Hausdorff distance. -/
theorem infEdist_le_hausdorffEdist_of_mem (h : x ∈ s) : infEdist x t ≤ hausdorffEdist s t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s t : Set α
    h : Membership.mem s x
    ⊢ LE.le (EMetric.infEdist x t) (EMetric.hausdorffEdist s t)
  -/
  rw [hausdorffEdist_def]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s t : Set α
    h : Membership.mem s x
    ⊢ LE.le (EMetric.infEdist x t) (Max.max (iSup fun x => iSup fun h => EMetric.i …
  -/
  refine le_trans ?_ le_sup_left
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    s t : Set α
    h : Membership.mem s x
    ⊢ LE.le (EMetric.infEdist x t) (iSup fun x => iSup fun h => EMetric.infEdist x …
  -/
  exact le_iSup₂ (α := ℝ≥0∞) x h
  /-
    🎉 no goals
  -/


/-- If the Hausdorff distance is `< r`, then any point in one of the sets has
a corresponding point at distance `< r` in the other set. -/
theorem exists_edist_lt_of_hausdorffEdist_lt {r : ℝ≥0∞} (h : x ∈ s) (H : hausdorffEdist s t < r) :
    ∃ y ∈ t, edist x y < r :=
  infEdist_lt_iff.mp <|
    calc
      infEdist x t ≤ hausdorffEdist s t := infEdist_le_hausdorffEdist_of_mem h
      _ < r := H


/-- The distance from `x` to `s` or `t` is controlled in terms of the Hausdorff distance
between `s` and `t`. -/
theorem infEdist_le_infEdist_add_hausdorffEdist :
    infEdist x t ≤ infEdist x s + hausdorffEdist s t :=
  ENNReal.le_of_forall_pos_le_add fun ε εpos h => by
    /-
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x : α
      s t : Set α
      ε : NNReal
      εpos : LT.lt 0 ε
      h : LT.lt (HAdd.hAdd (EMetric.infEdist x s) (EMetric.hausdorffEdist s t)) Top. …
      ⊢ LE.le (EMetric.infEdist x t) (HAdd.hAdd (HAdd.hAdd (EMetric.infEdist x s) (E …
    -/
    have ε0 : (ε / 2 : ℝ≥0∞) ≠ 0 := by simpa [pos_iff_ne_zero] using εpos
    have : infEdist x s < infEdist x s + ε / 2 :=
      ENNReal.lt_add_right (ENNReal.add_lt_top.1 h).1.ne ε0
    /-
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x : α
      s t : Set α
      ε : NNReal
      εpos : LT.lt 0 ε
      h : LT.lt (HAdd.hAdd (EMetric.infEdist x s) (EMetric.hausdorffEdist s t)) Top. …
      ε0 : Ne (HDiv.hDiv (↑ε) 2) 0
      this : LT.lt (EMetric.infEdist x s) (HAdd.hAdd (EMetric.infEdist x s) (HDiv.hD …
      ⊢ LE.le (EMetric.infEdist x t) (HAdd.hAdd (HAdd.hAdd (EMetric.infEdist x s) (E …
    -/
    obtain ⟨y : α, ys : y ∈ s, dxy : edist x y < infEdist x s + ↑ε / 2⟩ := infEdist_lt_iff.mp this
    have : hausdorffEdist s t < hausdorffEdist s t + ε / 2 :=
      ENNReal.lt_add_right (ENNReal.add_lt_top.1 h).2.ne ε0
    obtain ⟨z : α, zt : z ∈ t, dyz : edist y z < hausdorffEdist s t + ↑ε / 2⟩ :=
      exists_edist_lt_of_hausdorffEdist_lt ys this
    calc
      infEdist x t ≤ edist x z := infEdist_le_edist_of_mem zt
      _ ≤ edist x y + edist y z := edist_triangle _ _ _
      _ ≤ infEdist x s + ε / 2 + (hausdorffEdist s t + ε / 2) := add_le_add dxy.le dyz.le
      _ = infEdist x s + hausdorffEdist s t + ε := by
        simp [ENNReal.add_halves, add_comm, add_left_comm]


/-- The Hausdorff edistance is invariant under isometries. -/
theorem hausdorffEdist_image (h : Isometry Φ) :
    hausdorffEdist (Φ '' s) (Φ '' t) = hausdorffEdist s t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    s t : Set α
    Φ : α → β
    h : Isometry Φ
    ⊢ Eq (EMetric.hausdorffEdist (Set.image Φ s) (Set.image Φ t)) (EMetric.hausdor …
  -/
  simp only [hausdorffEdist_def, iSup_image, infEdist_image h]
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance is controlled by the diameter of the union. -/
theorem hausdorffEdist_le_ediam (hs : s.Nonempty) (ht : t.Nonempty) :
    hausdorffEdist s t ≤ diam (s ∪ t) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ LE.le (EMetric.hausdorffEdist s t) (EMetric.diam (Union.union s t))
  -/
  rcases hs with ⟨x, xs⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    ht : t.Nonempty
    x : α
    xs : Membership.mem s x
    ⊢ LE.le (EMetric.hausdorffEdist s t) (EMetric.diam (Union.union s t))
  -/
  rcases ht with ⟨y, yt⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    x : α
    xs : Membership.mem s x
    y : α
    yt : Membership.mem t y
    ⊢ LE.le (EMetric.hausdorffEdist s t) (EMetric.diam (Union.union s t))
  -/
  refine hausdorffEdist_le_of_mem_edist ?_ ?_
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      ⊢ ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y) (LE …
    -/
  · intro z hz
    /-
      case intro.intro.refine_1
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      z : α
      hz : Membership.mem s z
      ⊢ Exists fun y => And (Membership.mem t y) (LE.le (EDist.edist z y) (EMetric.d …
    -/
    exact ⟨y, yt, edist_le_diam_of_mem (subset_union_left hz) (subset_union_right yt)⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      ⊢ ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y) (LE …
    -/
  · intro z hz
    /-
      case intro.intro.refine_2
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      z : α
      hz : Membership.mem t z
      ⊢ Exists fun y => And (Membership.mem s y) (LE.le (EDist.edist z y) (EMetric.d …
    -/
    exact ⟨x, xs, edist_le_diam_of_mem (subset_union_right hz) (subset_union_left xs)⟩
    /-
      🎉 no goals
    -/


/-- The Hausdorff distance satisfies the triangle inequality. -/
theorem hausdorffEdist_triangle : hausdorffEdist s u ≤ hausdorffEdist s t + hausdorffEdist t u := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t u : Set α
    ⊢ LE.le (EMetric.hausdorffEdist s u) (HAdd.hAdd (EMetric.hausdorffEdist s t) ( …
  -/
  rw [hausdorffEdist_def]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t u : Set α
    ⊢ LE.le (Max.max (iSup fun x => iSup fun h => EMetric.infEdist x u) (iSup fun  …
  -/
  simp only [sup_le_iff, iSup_le_iff]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t u : Set α
    ⊢ And (∀ (i : α), Membership.mem s i → LE.le (EMetric.infEdist i u) (HAdd.hAdd …
  -/
  constructor
    /-
      case left
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t u : Set α
      ⊢ ∀ (i : α), Membership.mem s i → LE.le (EMetric.infEdist i u) (HAdd.hAdd (EMe …
    -/
  · show ∀ x ∈ s, infEdist x u ≤ hausdorffEdist s t + hausdorffEdist t u
    exact fun x xs =>
      calc
        infEdist x u ≤ infEdist x t + hausdorffEdist t u :=
          infEdist_le_infEdist_add_hausdorffEdist
        _ ≤ hausdorffEdist s t + hausdorffEdist t u :=
          add_le_add_right (infEdist_le_hausdorffEdist_of_mem xs) _
    /-
      case right
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t u : Set α
      ⊢ ∀ (i : α), Membership.mem u i → LE.le (EMetric.infEdist i s) (HAdd.hAdd (EMe …
    -/
  · show ∀ x ∈ u, infEdist x s ≤ hausdorffEdist s t + hausdorffEdist t u
    exact fun x xu =>
      calc
        infEdist x s ≤ infEdist x t + hausdorffEdist t s :=
          infEdist_le_infEdist_add_hausdorffEdist
        _ ≤ hausdorffEdist u t + hausdorffEdist t s :=
          add_le_add_right (infEdist_le_hausdorffEdist_of_mem xu) _
        _ = hausdorffEdist s t + hausdorffEdist t u := by simp [hausdorffEdist_comm, add_comm]


/-- Two sets are at zero Hausdorff edistance if and only if they have the same closure. -/
theorem hausdorffEdist_zero_iff_closure_eq_closure :
    hausdorffEdist s t = 0 ↔ closure s = closure t := by
  simp only [hausdorffEdist_def, ENNReal.sup_eq_zero, ENNReal.iSup_eq_zero, ← subset_def,
    ← mem_closure_iff_infEdist_zero, subset_antisymm_iff, isClosed_closure.closure_subset_iff]


/-- The Hausdorff edistance between a set and its closure vanishes. -/
@[simp]
theorem hausdorffEdist_self_closure : hausdorffEdist s (closure s) = 0 := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ⊢ Eq (EMetric.hausdorffEdist s (closure s)) 0
  -/
  rw [hausdorffEdist_zero_iff_closure_eq_closure, closure_closure]
  /-
    🎉 no goals
  -/


/-- Replacing a set by its closure does not change the Hausdorff edistance. -/
@[simp]
theorem hausdorffEdist_closure₁ : hausdorffEdist (closure s) t = hausdorffEdist s t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    ⊢ Eq (EMetric.hausdorffEdist (closure s) t) (EMetric.hausdorffEdist s t)
  -/
  refine le_antisymm ?_ ?_
  · calc
      _ ≤ hausdorffEdist (closure s) s + hausdorffEdist s t := hausdorffEdist_triangle
      _ = hausdorffEdist s t := by simp [hausdorffEdist_comm]
  · calc
      _ ≤ hausdorffEdist s (closure s) + hausdorffEdist (closure s) t := hausdorffEdist_triangle
      _ = hausdorffEdist (closure s) t := by simp


/-- Replacing a set by its closure does not change the Hausdorff edistance. -/
@[simp]
theorem hausdorffEdist_closure₂ : hausdorffEdist s (closure t) = hausdorffEdist s t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    ⊢ Eq (EMetric.hausdorffEdist s (closure t)) (EMetric.hausdorffEdist s t)
  -/
  simp [@hausdorffEdist_comm _ _ s _]
  /-
    🎉 no goals
  -/


/-- The Hausdorff edistance between sets or their closures is the same. -/
theorem hausdorffEdist_closure : hausdorffEdist (closure s) (closure t) = hausdorffEdist s t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    ⊢ Eq (EMetric.hausdorffEdist (closure s) (closure t)) (EMetric.hausdorffEdist  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Two closed sets are at zero Hausdorff edistance if and only if they coincide. -/
theorem hausdorffEdist_zero_iff_eq_of_closed (hs : IsClosed s) (ht : IsClosed t) :
    hausdorffEdist s t = 0 ↔ s = t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    hs : IsClosed s
    ht : IsClosed t
    ⊢ Iff (Eq (EMetric.hausdorffEdist s t) 0) (Eq s t)
  -/
  rw [hausdorffEdist_zero_iff_closure_eq_closure, hs.closure_eq, ht.closure_eq]
  /-
    🎉 no goals
  -/


/-- The Haudorff edistance to the empty set is infinite. -/
theorem hausdorffEdist_empty (ne : s.Nonempty) : hausdorffEdist s ∅ = ∞ := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ne : s.Nonempty
    ⊢ Eq (EMetric.hausdorffEdist s EmptyCollection.emptyCollection) Top.top
  -/
  rcases ne with ⟨x, xs⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    x : α
    xs : Membership.mem s x
    ⊢ Eq (EMetric.hausdorffEdist s EmptyCollection.emptyCollection) Top.top
  -/
  have : infEdist x ∅ ≤ hausdorffEdist s ∅ := infEdist_le_hausdorffEdist_of_mem xs
  /-
    case intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    x : α
    xs : Membership.mem s x
    this : LE.le (EMetric.infEdist x EmptyCollection.emptyCollection) (EMetric.hau …
    ⊢ Eq (EMetric.hausdorffEdist s EmptyCollection.emptyCollection) Top.top
  -/
  simpa using this
  /-
    🎉 no goals
  -/


/-- If a set is at finite Hausdorff edistance of a nonempty set, it is nonempty. -/
theorem nonempty_of_hausdorffEdist_ne_top (hs : s.Nonempty) (fin : hausdorffEdist s t ≠ ⊤) :
    t.Nonempty :=
  t.eq_empty_or_nonempty.resolve_left fun ht ↦ fin (ht.symm ▸ hausdorffEdist_empty hs)


theorem empty_or_nonempty_of_hausdorffEdist_ne_top (fin : hausdorffEdist s t ≠ ⊤) :
    (s = ∅ ∧ t = ∅) ∨ (s.Nonempty ∧ t.Nonempty) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s t : Set α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
  -/
  rcases s.eq_empty_or_nonempty with hs | hs
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      fin : Ne (EMetric.hausdorffEdist s t) Top.top
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
    -/
  · rcases t.eq_empty_or_nonempty with ht | ht
      /-
        case inl.inl
        α : Type u
        inst✝ : PseudoEMetricSpace α
        s t : Set α
        fin : Ne (EMetric.hausdorffEdist s t) Top.top
        hs : Eq s EmptyCollection.emptyCollection
        ht : Eq t EmptyCollection.emptyCollection
        ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
      -/
    · exact Or.inl ⟨hs, ht⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u
        inst✝ : PseudoEMetricSpace α
        s t : Set α
        fin : Ne (EMetric.hausdorffEdist s t) Top.top
        hs : Eq s EmptyCollection.emptyCollection
        ht : t.Nonempty
        ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
      -/
    · rw [hausdorffEdist_comm] at fin
      /-
        case inl.inr
        α : Type u
        inst✝ : PseudoEMetricSpace α
        s t : Set α
        fin : Ne (EMetric.hausdorffEdist t s) Top.top
        hs : Eq s EmptyCollection.emptyCollection
        ht : t.Nonempty
        ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
      -/
      exact Or.inr ⟨nonempty_of_hausdorffEdist_ne_top ht fin, ht⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u
      inst✝ : PseudoEMetricSpace α
      s t : Set α
      fin : Ne (EMetric.hausdorffEdist s t) Top.top
      hs : s.Nonempty
      ⊢ Or (And (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCo …
    -/
  · exact Or.inr ⟨hs, nonempty_of_hausdorffEdist_ne_top hs fin⟩
    /-
      🎉 no goals
    -/


/-- The minimal distance of a point to a set -/
def infDist (x : α) (s : Set α) : ℝ :=
  ENNReal.toReal (infEdist x s)


theorem infDist_eq_iInf : infDist x s = ⨅ y : s, dist x y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    ⊢ Eq (Metric.infDist x s) (iInf fun y => Dist.dist x ↑y)
  -/
  rw [infDist, infEdist, iInf_subtype', ENNReal.toReal_iInf]
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x : α
      ⊢ Eq (iInf fun i => (EDist.edist x ↑i).toReal) (iInf fun y => Dist.dist x ↑y)
    -/
  · simp only [dist_edist]
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x : α
      ⊢ ∀ (i : Subtype (Membership.mem s)), Ne (EDist.edist x ↑i) Top.top
    -/
  · exact fun _ ↦ edist_ne_top _ _
    /-
      🎉 no goals
    -/


/-- The minimal distance is always nonnegative -/
theorem infDist_nonneg : 0 ≤ infDist x s := toReal_nonneg


/-- The minimal distance to the empty set is 0 (if you want to have the more reasonable
value `∞` instead, use `EMetric.infEdist`, which takes values in `ℝ≥0∞`) -/
@[simp]
                                              /-
                                                α : Type u
                                                inst✝ : PseudoMetricSpace α
                                                x : α
                                                ⊢ Eq (Metric.infDist x EmptyCollection.emptyCollection) 0
                                              -/
theorem infDist_empty : infDist x ∅ = 0 := by simp [infDist]
                                              /-
                                                🎉 no goals
                                              -/


/-- In a metric space, the minimal edistance to a nonempty set is finite. -/
theorem infEdist_ne_top (h : s.Nonempty) : infEdist x s ≠ ⊤ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    h : s.Nonempty
    ⊢ Ne (EMetric.infEdist x s) Top.top
  -/
  rcases h with ⟨y, hy⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    hy : Membership.mem s y
    ⊢ Ne (EMetric.infEdist x s) Top.top
  -/
  exact ne_top_of_le_ne_top (edist_ne_top _ _) (infEdist_le_edist_of_mem hy)
  /-
    🎉 no goals
  -/


@[simp]
theorem infEdist_eq_top_iff : infEdist x s = ∞ ↔ s = ∅ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    ⊢ Iff (Eq (EMetric.infEdist x s) Top.top) (Eq s EmptyCollection.emptyCollection)
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rcases s.eq_empty_or_nonempty with rfl | hs <;> simp [*, Nonempty.ne_empty, infEdist_ne_top]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The minimal distance of a point to a set containing it vanishes. -/
theorem infDist_zero_of_mem (h : x ∈ s) : infDist x s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    h : Membership.mem s x
    ⊢ Eq (Metric.infDist x s) 0
  -/
  simp [infEdist_zero_of_mem h, infDist]
  /-
    🎉 no goals
  -/


/-- The minimal distance to a singleton is the distance to the unique point in this singleton. -/
@[simp]
                                                           /-
                                                             α : Type u
                                                             inst✝ : PseudoMetricSpace α
                                                             x y : α
                                                             ⊢ Eq (Metric.infDist x (Singleton.singleton y)) (Dist.dist x y)
                                                           -/
theorem infDist_singleton : infDist x {y} = dist x y := by simp [infDist, dist_edist]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The minimal distance to a set is bounded by the distance to any point in this set. -/
theorem infDist_le_dist_of_mem (h : y ∈ s) : infDist x s ≤ dist x y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem s y
    ⊢ LE.le (Metric.infDist x s) (Dist.dist x y)
  -/
  rw [dist_edist, infDist]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem s y
    ⊢ LE.le (EMetric.infEdist x s).toReal (EDist.edist x y).toReal
  -/
  exact ENNReal.toReal_mono (edist_ne_top _ _) (infEdist_le_edist_of_mem h)
  /-
    🎉 no goals
  -/


/-- The minimal distance is monotone with respect to inclusion. -/
theorem infDist_le_infDist_of_subset (h : s ⊆ t) (hs : s.Nonempty) : infDist x t ≤ infDist x s :=
  ENNReal.toReal_mono (infEdist_ne_top hs) (infEdist_anti h)


/-- The minimal distance to a set `s` is `< r` iff there exists a point in `s` at distance `< r`. -/
theorem infDist_lt_iff {r : ℝ} (hs : s.Nonempty) : infDist x s < r ↔ ∃ y ∈ s, dist x y < r := by
  simp_rw [infDist, ← ENNReal.lt_ofReal_iff_toReal_lt (infEdist_ne_top hs), infEdist_lt_iff,
    ENNReal.lt_ofReal_iff_toReal_lt (edist_ne_top _ _), ← dist_edist]


/-- The minimal distance from `x` to `s` is bounded by the distance from `y` to `s`, modulo
the distance between `x` and `y`. -/
theorem infDist_le_infDist_add_dist : infDist x s ≤ infDist y s + dist x y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    ⊢ LE.le (Metric.infDist x s) (HAdd.hAdd (Metric.infDist y s) (Dist.dist x y))
  -/
  rw [infDist, infDist, dist_edist]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    ⊢ LE.le (EMetric.infEdist x s).toReal (HAdd.hAdd (EMetric.infEdist y s).toReal …
  -/
  refine ENNReal.toReal_le_add' infEdist_le_infEdist_add_edist ?_ (flip absurd (edist_ne_top _ _))
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    ⊢ Eq (EMetric.infEdist y s) Top.top → Eq (EMetric.infEdist x s) Top.top
  -/
  simp only [infEdist_eq_top_iff, imp_self]
  /-
    🎉 no goals
  -/


theorem not_mem_of_dist_lt_infDist (h : dist x y < infDist x s) : y ∉ s := fun hy =>
  h.not_le <| infDist_le_dist_of_mem hy


theorem disjoint_ball_infDist : Disjoint (ball x (infDist x s)) s :=
  disjoint_left.2 fun _y hy => not_mem_of_dist_lt_infDist <| mem_ball'.1 hy


theorem ball_infDist_subset_compl : ball x (infDist x s) ⊆ sᶜ :=
  (disjoint_ball_infDist (s := s)).subset_compl_right


theorem ball_infDist_compl_subset : ball x (infDist x sᶜ) ⊆ s :=
  ball_infDist_subset_compl.trans_eq (compl_compl s)


theorem disjoint_closedBall_of_lt_infDist {r : ℝ} (h : r < infDist x s) :
    Disjoint (closedBall x r) s :=
  disjoint_ball_infDist.mono_left <| closedBall_subset_ball h


theorem dist_le_infDist_add_diam (hs : IsBounded s) (hy : y ∈ s) :
    dist x y ≤ infDist x s + diam s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    hs : Bornology.IsBounded s
    hy : Membership.mem s y
    ⊢ LE.le (Dist.dist x y) (HAdd.hAdd (Metric.infDist x s) (Metric.diam s))
  -/
  rw [infDist, diam, dist_edist]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    hs : Bornology.IsBounded s
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist x y).toReal (HAdd.hAdd (EMetric.infEdist x s).toReal (EMe …
  -/
  exact toReal_le_add (edist_le_infEdist_add_ediam hy) (infEdist_ne_top ⟨y, hy⟩) hs.ediam_ne_top
  /-
    🎉 no goals
  -/


/-- The minimal distance to a set is Lipschitz in point with constant 1 -/
theorem lipschitz_infDist_pt : LipschitzWith 1 (infDist · s) :=
  LipschitzWith.of_le_add fun _ _ => infDist_le_infDist_add_dist


/-- The minimal distance to a set is uniformly continuous in point -/
theorem uniformContinuous_infDist_pt : UniformContinuous (infDist · s) :=
  (lipschitz_infDist_pt s).uniformContinuous


/-- The minimal distance to a set is continuous in point -/
@[continuity]
theorem continuous_infDist_pt : Continuous (infDist · s) :=
  (uniformContinuous_infDist_pt s).continuous


/-- The minimal distances to a set and its closure coincide. -/
theorem infDist_closure : infDist x (closure s) = infDist x s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    ⊢ Eq (Metric.infDist x (closure s)) (Metric.infDist x s)
  -/
  simp [infDist, infEdist_closure]
  /-
    🎉 no goals
  -/


/-- If a point belongs to the closure of `s`, then its infimum distance to `s` equals zero.
The converse is true provided that `s` is nonempty, see `Metric.mem_closure_iff_infDist_zero`. -/
theorem infDist_zero_of_mem_closure (hx : x ∈ closure s) : infDist x s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    hx : Membership.mem (closure s) x
    ⊢ Eq (Metric.infDist x s) 0
  -/
  rw [← infDist_closure]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    hx : Membership.mem (closure s) x
    ⊢ Eq (Metric.infDist x (closure s)) 0
  -/
  exact infDist_zero_of_mem hx
  /-
    🎉 no goals
  -/


/-- A point belongs to the closure of `s` iff its infimum distance to this set vanishes. -/
theorem mem_closure_iff_infDist_zero (h : s.Nonempty) : x ∈ closure s ↔ infDist x s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    h : s.Nonempty
    ⊢ Iff (Membership.mem (closure s) x) (Eq (Metric.infDist x s) 0)
  -/
  simp [mem_closure_iff_infEdist_zero, infDist, ENNReal.toReal_eq_zero_iff, infEdist_ne_top h]
  /-
    🎉 no goals
  -/


/-- Given a closed set `s`, a point belongs to `s` iff its infimum distance to this set vanishes -/
theorem _root_.IsClosed.mem_iff_infDist_zero (h : IsClosed s) (hs : s.Nonempty) :
                                  /-
                                    α : Type u
                                    inst✝ : PseudoMetricSpace α
                                    s : Set α
                                    x : α
                                    h : IsClosed s
                                    hs : s.Nonempty
                                    ⊢ Iff (Membership.mem s x) (Eq (Metric.infDist x s) 0)
                                  -/
    x ∈ s ↔ infDist x s = 0 := by rw [← mem_closure_iff_infDist_zero hs, h.closure_eq]
                                  /-
                                    🎉 no goals
                                  -/


/-- Given a closed set `s`, a point belongs to `s` iff its infimum distance to this set vanishes. -/
theorem _root_.IsClosed.not_mem_iff_infDist_pos (h : IsClosed s) (hs : s.Nonempty) :
    x ∉ s ↔ 0 < infDist x s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    h : IsClosed s
    hs : s.Nonempty
    ⊢ Iff (Not (Membership.mem s x)) (LT.lt 0 (Metric.infDist x s))
  -/
  simp [h.mem_iff_infDist_zero hs, infDist_nonneg.gt_iff_ne]
  /-
    🎉 no goals
  -/


theorem continuousAt_inv_infDist_pt (h : x ∉ closure s) :
    ContinuousAt (fun x ↦ (infDist x s)⁻¹) x := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    h : Not (Membership.mem (closure s) x)
    ⊢ ContinuousAt (fun x => Inv.inv (Metric.infDist x s)) x
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hs)
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      x : α
      h : Not (Membership.mem (closure EmptyCollection.emptyCollection) x)
      ⊢ ContinuousAt (fun x => Inv.inv (Metric.infDist x EmptyCollection.emptyCollec …
    -/
  · simp only [infDist_empty, continuousAt_const]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x : α
      h : Not (Membership.mem (closure s) x)
      hs : s.Nonempty
      ⊢ ContinuousAt (fun x => Inv.inv (Metric.infDist x s)) x
    -/
  · refine (continuous_infDist_pt s).continuousAt.inv₀ ?_
    /-
      case inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x : α
      h : Not (Membership.mem (closure s) x)
      hs : s.Nonempty
      ⊢ Ne (Metric.infDist x s) 0
    -/
    rwa [Ne, ← mem_closure_iff_infDist_zero hs]
    /-
      🎉 no goals
    -/


/-- The infimum distance is invariant under isometries. -/
theorem infDist_image (hΦ : Isometry Φ) : infDist (Φ x) (Φ '' t) = infDist x t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    t : Set α
    x : α
    Φ : α → β
    hΦ : Isometry Φ
    ⊢ Eq (Metric.infDist (Φ x) (Set.image Φ t)) (Metric.infDist x t)
  -/
  simp [infDist, infEdist_image hΦ]
  /-
    🎉 no goals
  -/


theorem infDist_inter_closedBall_of_mem (h : y ∈ s) :
    infDist x (s ∩ closedBall x (dist y x)) = infDist x s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem s y
    ⊢ Eq (Metric.infDist x (Inter.inter s (Metric.closedBall x (Dist.dist y x))))  …
  -/
  replace h : y ∈ s ∩ closedBall x (dist y x) := ⟨h, mem_closedBall.2 le_rfl⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
    ⊢ Eq (Metric.infDist x (Inter.inter s (Metric.closedBall x (Dist.dist y x))))  …
  -/
  refine le_antisymm ?_ (infDist_le_infDist_of_subset inter_subset_left ⟨y, h⟩)
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
    ⊢ LE.le (Metric.infDist x (Inter.inter s (Metric.closedBall x (Dist.dist y x)) …
  -/
  refine not_lt.1 fun hlt => ?_
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
    hlt : LT.lt (Metric.infDist x s) (Metric.infDist x (Inter.inter s (Metric.clos …
    ⊢ False
  -/
  rcases (infDist_lt_iff ⟨y, h.1⟩).mp hlt with ⟨z, hzs, hz⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
    hlt : LT.lt (Metric.infDist x s) (Metric.infDist x (Inter.inter s (Metric.clos …
    z : α
    hzs : Membership.mem s z
    hz : LT.lt (Dist.dist x z) (Metric.infDist x (Inter.inter s (Metric.closedBall …
    ⊢ False
  -/
  rcases le_or_lt (dist z x) (dist y x) with hle | hlt
    /-
      case intro.intro.inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
      hlt : LT.lt (Metric.infDist x s) (Metric.infDist x (Inter.inter s (Metric.clos …
      z : α
      hzs : Membership.mem s z
      hz : LT.lt (Dist.dist x z) (Metric.infDist x (Inter.inter s (Metric.closedBall …
      hle : LE.le (Dist.dist z x) (Dist.dist y x)
      ⊢ False
    -/
  · exact hz.not_le (infDist_le_dist_of_mem ⟨hzs, hle⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
      hlt✝ : LT.lt (Metric.infDist x s) (Metric.infDist x (Inter.inter s (Metric.clo …
      z : α
      hzs : Membership.mem s z
      hz : LT.lt (Dist.dist x z) (Metric.infDist x (Inter.inter s (Metric.closedBall …
      hlt : LT.lt (Dist.dist y x) (Dist.dist z x)
      ⊢ False
    -/
  · rw [dist_comm z, dist_comm y] at hlt
    /-
      case intro.intro.inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      h : Membership.mem (Inter.inter s (Metric.closedBall x (Dist.dist y x))) y
      hlt✝ : LT.lt (Metric.infDist x s) (Metric.infDist x (Inter.inter s (Metric.clo …
      z : α
      hzs : Membership.mem s z
      hz : LT.lt (Dist.dist x z) (Metric.infDist x (Inter.inter s (Metric.closedBall …
      hlt : LT.lt (Dist.dist x y) (Dist.dist x z)
      ⊢ False
    -/
    exact (hlt.trans hz).not_le (infDist_le_dist_of_mem h)
    /-
      🎉 no goals
    -/


theorem _root_.IsCompact.exists_infDist_eq_dist (h : IsCompact s) (hne : s.Nonempty) (x : α) :
    ∃ y ∈ s, infDist x s = dist x y :=
  let ⟨y, hys, hy⟩ := h.exists_infEdist_eq_edist hne x
              /-
                α : Type u
                inst✝ : PseudoMetricSpace α
                s : Set α
                h : IsCompact s
                hne : s.Nonempty
                x y : α
                hys : Membership.mem s y
                hy : Eq (EMetric.infEdist x s) (EDist.edist x y)
                ⊢ Eq (Metric.infDist x s) (Dist.dist x y)
              -/
  ⟨y, hys, by rw [infDist, dist_edist, hy]⟩
              /-
                🎉 no goals
              -/


theorem _root_.IsClosed.exists_infDist_eq_dist [ProperSpace α] (h : IsClosed s) (hne : s.Nonempty)
    (x : α) : ∃ y ∈ s, infDist x s = dist x y := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    hne : s.Nonempty
    x : α
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x s) (Dist.dist …
  -/
  rcases hne with ⟨z, hz⟩
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x s) (Dist.dist …
  -/
  rw [← infDist_inter_closedBall_of_mem hz]
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x (Inter.inter  …
  -/
  set t := s ∩ closedBall x (dist z x)
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    t : Set α := Inter.inter s (Metric.closedBall x (Dist.dist z x))
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x t) (Dist.dist …
  -/
  have htc : IsCompact t := (isCompact_closedBall x (dist z x)).inter_left h
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    t : Set α := Inter.inter s (Metric.closedBall x (Dist.dist z x))
    htc : IsCompact t
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x t) (Dist.dist …
  -/
  have htne : t.Nonempty := ⟨z, hz, mem_closedBall.2 le_rfl⟩
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    t : Set α := Inter.inter s (Metric.closedBall x (Dist.dist z x))
    htc : IsCompact t
    htne : t.Nonempty
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x t) (Dist.dist …
  -/
  obtain ⟨y, ⟨hys, -⟩, hyd⟩ : ∃ y ∈ t, infDist x t = dist x y := htc.exists_infDist_eq_dist htne x
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    h : IsClosed s
    x z : α
    hz : Membership.mem s z
    t : Set α := Inter.inter s (Metric.closedBall x (Dist.dist z x))
    htc : IsCompact t
    htne : t.Nonempty
    y : α
    hyd : Eq (Metric.infDist x t) (Dist.dist x y)
    hys : Membership.mem s y
    ⊢ Exists fun y => And (Membership.mem s y) (Eq (Metric.infDist x t) (Dist.dist …
  -/
  exact ⟨y, hys, hyd⟩
  /-
    🎉 no goals
  -/


theorem exists_mem_closure_infDist_eq_dist [ProperSpace α] (hne : s.Nonempty) (x : α) :
    ∃ y ∈ closure s, infDist x s = dist x y := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    hne : s.Nonempty
    x : α
    ⊢ Exists fun y => And (Membership.mem (closure s) y) (Eq (Metric.infDist x s)  …
  -/
  simpa only [infDist_closure] using isClosed_closure.exists_infDist_eq_dist hne.closure x
  /-
    🎉 no goals
  -/


/-- The minimal distance of a point to a set as a `ℝ≥0` -/
def infNndist (x : α) (s : Set α) : ℝ≥0 :=
  ENNReal.toNNReal (infEdist x s)


@[simp]
theorem coe_infNndist : (infNndist x s : ℝ) = infDist x s :=
  rfl


/-- The minimal distance to a set (as `ℝ≥0`) is Lipschitz in point with constant 1 -/
theorem lipschitz_infNndist_pt (s : Set α) : LipschitzWith 1 fun x => infNndist x s :=
  LipschitzWith.of_le_add fun _ _ => infDist_le_infDist_add_dist


/-- The minimal distance to a set (as `ℝ≥0`) is uniformly continuous in point -/
theorem uniformContinuous_infNndist_pt (s : Set α) : UniformContinuous fun x => infNndist x s :=
  (lipschitz_infNndist_pt s).uniformContinuous


/-- The minimal distance to a set (as `ℝ≥0`) is continuous in point -/
theorem continuous_infNndist_pt (s : Set α) : Continuous fun x => infNndist x s :=
  (uniformContinuous_infNndist_pt s).continuous


/-- The Hausdorff distance between two sets is the smallest nonnegative `r` such that each set is
included in the `r`-neighborhood of the other. If there is no such `r`, it is defined to
be `0`, arbitrarily. -/
def hausdorffDist (s t : Set α) : ℝ :=
  ENNReal.toReal (hausdorffEdist s t)


/-- The Hausdorff distance is nonnegative. -/
                                                           /-
                                                             α : Type u
                                                             inst✝ : PseudoMetricSpace α
                                                             s t : Set α
                                                             ⊢ LE.le 0 (Metric.hausdorffDist s t)
                                                           -/
theorem hausdorffDist_nonneg : 0 ≤ hausdorffDist s t := by simp [hausdorffDist]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- If two sets are nonempty and bounded in a metric space, they are at finite Hausdorff
edistance. -/
theorem hausdorffEdist_ne_top_of_nonempty_of_bounded (hs : s.Nonempty) (ht : t.Nonempty)
    (bs : IsBounded s) (bt : IsBounded t) : hausdorffEdist s t ≠ ⊤ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    hs : s.Nonempty
    ht : t.Nonempty
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    ⊢ Ne (EMetric.hausdorffEdist s t) Top.top
  -/
  rcases hs with ⟨cs, hcs⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    ht : t.Nonempty
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    cs : α
    hcs : Membership.mem s cs
    ⊢ Ne (EMetric.hausdorffEdist s t) Top.top
  -/
  rcases ht with ⟨ct, hct⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    cs : α
    hcs : Membership.mem s cs
    ct : α
    hct : Membership.mem t ct
    ⊢ Ne (EMetric.hausdorffEdist s t) Top.top
  -/
  rcases bs.subset_closedBall ct with ⟨rs, hrs⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    cs : α
    hcs : Membership.mem s cs
    ct : α
    hct : Membership.mem t ct
    rs : Real
    hrs : HasSubset.Subset s (Metric.closedBall ct rs)
    ⊢ Ne (EMetric.hausdorffEdist s t) Top.top
  -/
  rcases bt.subset_closedBall cs with ⟨rt, hrt⟩
  have : hausdorffEdist s t ≤ ENNReal.ofReal (max rs rt) := by
    apply hausdorffEdist_le_of_mem_edist
    · intro x xs
      exists ct, hct
      have : dist x ct ≤ max rs rt := le_trans (hrs xs) (le_max_left _ _)
      rwa [edist_dist, ENNReal.ofReal_le_ofReal_iff]
      exact le_trans dist_nonneg this
    · intro x xt
      exists cs, hcs
      have : dist x cs ≤ max rs rt := le_trans (hrt xt) (le_max_right _ _)
      rwa [edist_dist, ENNReal.ofReal_le_ofReal_iff]
      exact le_trans dist_nonneg this
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    cs : α
    hcs : Membership.mem s cs
    ct : α
    hct : Membership.mem t ct
    rs : Real
    hrs : HasSubset.Subset s (Metric.closedBall ct rs)
    rt : Real
    hrt : HasSubset.Subset t (Metric.closedBall cs rt)
    this : LE.le (EMetric.hausdorffEdist s t) (ENNReal.ofReal (Max.max rs rt))
    ⊢ Ne (EMetric.hausdorffEdist s t) Top.top
  -/
  exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top this
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance between a set and itself is zero. -/
@[simp]
                                                              /-
                                                                α : Type u
                                                                inst✝ : PseudoMetricSpace α
                                                                s : Set α
                                                                ⊢ Eq (Metric.hausdorffDist s s) 0
                                                              -/
theorem hausdorffDist_self_zero : hausdorffDist s s = 0 := by simp [hausdorffDist]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- The Hausdorff distances from `s` to `t` and from `t` to `s` coincide. -/
theorem hausdorffDist_comm : hausdorffDist s t = hausdorffDist t s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    ⊢ Eq (Metric.hausdorffDist s t) (Metric.hausdorffDist t s)
  -/
  simp [hausdorffDist, hausdorffEdist_comm]
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance to the empty set vanishes (if you want to have the more reasonable
value `∞` instead, use `EMetric.hausdorffEdist`, which takes values in `ℝ≥0∞`). -/
@[simp]
theorem hausdorffDist_empty : hausdorffDist s ∅ = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    ⊢ Eq (Metric.hausdorffDist s EmptyCollection.emptyCollection) 0
  -/
  rcases s.eq_empty_or_nonempty with h | h
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      h : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (Metric.hausdorffDist s EmptyCollection.emptyCollection) 0
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      h : s.Nonempty
      ⊢ Eq (Metric.hausdorffDist s EmptyCollection.emptyCollection) 0
    -/
  · simp [hausdorffDist, hausdorffEdist_empty h]
    /-
      🎉 no goals
    -/


/-- The Hausdorff distance to the empty set vanishes (if you want to have the more reasonable
value `∞` instead, use `EMetric.hausdorffEdist`, which takes values in `ℝ≥0∞`). -/
@[simp]
                                                           /-
                                                             α : Type u
                                                             inst✝ : PseudoMetricSpace α
                                                             s : Set α
                                                             ⊢ Eq (Metric.hausdorffDist EmptyCollection.emptyCollection s) 0
                                                           -/
theorem hausdorffDist_empty' : hausdorffDist ∅ s = 0 := by simp [hausdorffDist_comm]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Bounding the Hausdorff distance by bounding the distance of any point
in each set to the other set -/
theorem hausdorffDist_le_of_infDist {r : ℝ} (hr : 0 ≤ r) (H1 : ∀ x ∈ s, infDist x t ≤ r)
    (H2 : ∀ x ∈ t, infDist x s ≤ r) : hausdorffDist s t ≤ r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    r : Real
    hr : LE.le 0 r
    H1 : ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
    H2 : ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
    ⊢ LE.le (Metric.hausdorffDist s t) r
  -/
  rcases s.eq_empty_or_nonempty with hs | hs
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
      H2 : ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ LE.le (Metric.hausdorffDist s t) r
    -/
  · rwa [hs, hausdorffDist_empty']
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    r : Real
    hr : LE.le 0 r
    H1 : ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
    H2 : ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
    hs : s.Nonempty
    ⊢ LE.le (Metric.hausdorffDist s t) r
  -/
  rcases t.eq_empty_or_nonempty with ht | ht
    /-
      case inr.inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
      H2 : ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
      hs : s.Nonempty
      ht : Eq t EmptyCollection.emptyCollection
      ⊢ LE.le (Metric.hausdorffDist s t) r
    -/
  · rwa [ht, hausdorffDist_empty]
    /-
      🎉 no goals
    -/
  have : hausdorffEdist s t ≤ ENNReal.ofReal r := by
    apply hausdorffEdist_le_of_infEdist _ _
    · simpa only [infDist, ← ENNReal.le_ofReal_iff_toReal_le (infEdist_ne_top ht) hr] using H1
    · simpa only [infDist, ← ENNReal.le_ofReal_iff_toReal_le (infEdist_ne_top hs) hr] using H2
  /-
    case inr.inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    r : Real
    hr : LE.le 0 r
    H1 : ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
    H2 : ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
    hs : s.Nonempty
    ht : t.Nonempty
    this : LE.le (EMetric.hausdorffEdist s t) (ENNReal.ofReal r)
    ⊢ LE.le (Metric.hausdorffDist s t) r
  -/
  exact ENNReal.toReal_le_of_le_ofReal hr this
  /-
    🎉 no goals
  -/


/-- Bounding the Hausdorff distance by exhibiting, for any point in each set,
another point in the other set at controlled distance -/
theorem hausdorffDist_le_of_mem_dist {r : ℝ} (hr : 0 ≤ r) (H1 : ∀ x ∈ s, ∃ y ∈ t, dist x y ≤ r)
    (H2 : ∀ x ∈ t, ∃ y ∈ s, dist x y ≤ r) : hausdorffDist s t ≤ r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    r : Real
    hr : LE.le 0 r
    H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
    H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
    ⊢ LE.le (Metric.hausdorffDist s t) r
  -/
  apply hausdorffDist_le_of_infDist hr
    /-
      case H1
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      ⊢ ∀ (x : α), Membership.mem s x → LE.le (Metric.infDist x t) r
    -/
  · intro x xs
    /-
      case H1
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xs : Membership.mem s x
      ⊢ LE.le (Metric.infDist x t) r
    -/
    rcases H1 x xs with ⟨y, yt, hy⟩
    /-
      case H1.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xs : Membership.mem s x
      y : α
      yt : Membership.mem t y
      hy : LE.le (Dist.dist x y) r
      ⊢ LE.le (Metric.infDist x t) r
    -/
    exact le_trans (infDist_le_dist_of_mem yt) hy
    /-
      🎉 no goals
    -/
    /-
      case H2
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      ⊢ ∀ (x : α), Membership.mem t x → LE.le (Metric.infDist x s) r
    -/
  · intro x xt
    /-
      case H2
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xt : Membership.mem t x
      ⊢ LE.le (Metric.infDist x s) r
    -/
    rcases H2 x xt with ⟨y, ys, hy⟩
    /-
      case H2.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s t : Set α
      r : Real
      hr : LE.le 0 r
      H1 : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem t y)  …
      H2 : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem s y)  …
      x : α
      xt : Membership.mem t x
      y : α
      ys : Membership.mem s y
      hy : LE.le (Dist.dist x y) r
      ⊢ LE.le (Metric.infDist x s) r
    -/
    exact le_trans (infDist_le_dist_of_mem ys) hy
    /-
      🎉 no goals
    -/


/-- The Hausdorff distance is controlled by the diameter of the union. -/
theorem hausdorffDist_le_diam (hs : s.Nonempty) (bs : IsBounded s) (ht : t.Nonempty)
    (bt : IsBounded t) : hausdorffDist s t ≤ diam (s ∪ t) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    hs : s.Nonempty
    bs : Bornology.IsBounded s
    ht : t.Nonempty
    bt : Bornology.IsBounded t
    ⊢ LE.le (Metric.hausdorffDist s t) (Metric.diam (Union.union s t))
  -/
  rcases hs with ⟨x, xs⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    bs : Bornology.IsBounded s
    ht : t.Nonempty
    bt : Bornology.IsBounded t
    x : α
    xs : Membership.mem s x
    ⊢ LE.le (Metric.hausdorffDist s t) (Metric.diam (Union.union s t))
  -/
  rcases ht with ⟨y, yt⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    bs : Bornology.IsBounded s
    bt : Bornology.IsBounded t
    x : α
    xs : Membership.mem s x
    y : α
    yt : Membership.mem t y
    ⊢ LE.le (Metric.hausdorffDist s t) (Metric.diam (Union.union s t))
  -/
  refine hausdorffDist_le_of_mem_dist diam_nonneg ?_ ?_
  · exact fun z hz => ⟨y, yt, dist_le_diam_of_mem (bs.union bt) (subset_union_left hz)
      (subset_union_right yt)⟩
  · exact fun z hz => ⟨x, xs, dist_le_diam_of_mem (bs.union bt) (subset_union_right hz)
      (subset_union_left xs)⟩


/-- The distance to a set is controlled by the Hausdorff distance. -/
theorem infDist_le_hausdorffDist_of_mem (hx : x ∈ s) (fin : hausdorffEdist s t ≠ ⊤) :
    infDist x t ≤ hausdorffDist s t :=
  toReal_mono fin (infEdist_le_hausdorffEdist_of_mem hx)


/-- If the Hausdorff distance is `< r`, any point in one of the sets is at distance
`< r` of a point in the other set. -/
theorem exists_dist_lt_of_hausdorffDist_lt {r : ℝ} (h : x ∈ s) (H : hausdorffDist s t < r)
    (fin : hausdorffEdist s t ≠ ⊤) : ∃ y ∈ t, dist x y < r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    r : Real
    h : Membership.mem s x
    H : LT.lt (Metric.hausdorffDist s t) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt (Dist.dist x y) r)
  -/
  have r0 : 0 < r := lt_of_le_of_lt hausdorffDist_nonneg H
  have : hausdorffEdist s t < ENNReal.ofReal r := by
    rwa [hausdorffDist, ← ENNReal.toReal_ofReal (le_of_lt r0),
      ENNReal.toReal_lt_toReal fin ENNReal.ofReal_ne_top] at H
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    r : Real
    h : Membership.mem s x
    H : LT.lt (Metric.hausdorffDist s t) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    r0 : LT.lt 0 r
    this : LT.lt (EMetric.hausdorffEdist s t) (ENNReal.ofReal r)
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt (Dist.dist x y) r)
  -/
  rcases exists_edist_lt_of_hausdorffEdist_lt h this with ⟨y, hy, yr⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    r : Real
    h : Membership.mem s x
    H : LT.lt (Metric.hausdorffDist s t) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    r0 : LT.lt 0 r
    this : LT.lt (EMetric.hausdorffEdist s t) (ENNReal.ofReal r)
    y : α
    hy : Membership.mem t y
    yr : LT.lt (EDist.edist x y) (ENNReal.ofReal r)
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt (Dist.dist x y) r)
  -/
  rw [edist_dist, ENNReal.ofReal_lt_ofReal_iff r0] at yr
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    r : Real
    h : Membership.mem s x
    H : LT.lt (Metric.hausdorffDist s t) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    r0 : LT.lt 0 r
    this : LT.lt (EMetric.hausdorffEdist s t) (ENNReal.ofReal r)
    y : α
    hy : Membership.mem t y
    yr : LT.lt (Dist.dist x y) r
    ⊢ Exists fun y => And (Membership.mem t y) (LT.lt (Dist.dist x y) r)
  -/
  exact ⟨y, hy, yr⟩
  /-
    🎉 no goals
  -/


/-- If the Hausdorff distance is `< r`, any point in one of the sets is at distance
`< r` of a point in the other set. -/
theorem exists_dist_lt_of_hausdorffDist_lt' {r : ℝ} (h : y ∈ t) (H : hausdorffDist s t < r)
    (fin : hausdorffEdist s t ≠ ⊤) : ∃ x ∈ s, dist x y < r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    y : α
    r : Real
    h : Membership.mem t y
    H : LT.lt (Metric.hausdorffDist s t) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ Exists fun x => And (Membership.mem s x) (LT.lt (Dist.dist x y) r)
  -/
  rw [hausdorffDist_comm] at H
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    y : α
    r : Real
    h : Membership.mem t y
    H : LT.lt (Metric.hausdorffDist t s) r
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ Exists fun x => And (Membership.mem s x) (LT.lt (Dist.dist x y) r)
  -/
  rw [hausdorffEdist_comm] at fin
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    y : α
    r : Real
    h : Membership.mem t y
    H : LT.lt (Metric.hausdorffDist t s) r
    fin : Ne (EMetric.hausdorffEdist t s) Top.top
    ⊢ Exists fun x => And (Membership.mem s x) (LT.lt (Dist.dist x y) r)
  -/
  simpa [dist_comm] using exists_dist_lt_of_hausdorffDist_lt h H fin
  /-
    🎉 no goals
  -/


/-- The infimum distance to `s` and `t` are the same, up to the Hausdorff distance
between `s` and `t` -/
theorem infDist_le_infDist_add_hausdorffDist (fin : hausdorffEdist s t ≠ ⊤) :
    infDist x t ≤ infDist x s + hausdorffDist s t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ LE.le (Metric.infDist x t) (HAdd.hAdd (Metric.infDist x s) (Metric.hausdorff …
  -/
  refine toReal_le_add' infEdist_le_infEdist_add_hausdorffEdist (fun h ↦ ?_) (flip absurd fin)
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    h : Eq (EMetric.infEdist x s) Top.top
    ⊢ Eq (EMetric.infEdist x t) Top.top
  -/
  rw [infEdist_eq_top_iff, ← not_nonempty_iff_eq_empty] at h ⊢
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    h : Not s.Nonempty
    ⊢ Not t.Nonempty
  -/
  rw [hausdorffEdist_comm] at fin
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    fin : Ne (EMetric.hausdorffEdist t s) Top.top
    h : Not s.Nonempty
    ⊢ Not t.Nonempty
  -/
  exact mt (nonempty_of_hausdorffEdist_ne_top · fin) h
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance is invariant under isometries. -/
theorem hausdorffDist_image (h : Isometry Φ) :
    hausdorffDist (Φ '' s) (Φ '' t) = hausdorffDist s t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    s t : Set α
    Φ : α → β
    h : Isometry Φ
    ⊢ Eq (Metric.hausdorffDist (Set.image Φ s) (Set.image Φ t)) (Metric.hausdorffD …
  -/
  simp [hausdorffDist, hausdorffEdist_image h]
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance satisfies the triangle inequality. -/
theorem hausdorffDist_triangle (fin : hausdorffEdist s t ≠ ⊤) :
    hausdorffDist s u ≤ hausdorffDist s t + hausdorffDist t u := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t u : Set α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    ⊢ LE.le (Metric.hausdorffDist s u) (HAdd.hAdd (Metric.hausdorffDist s t) (Metr …
  -/
  refine toReal_le_add' hausdorffEdist_triangle (flip absurd fin) (not_imp_not.1 fun h ↦ ?_)
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t u : Set α
    fin : Ne (EMetric.hausdorffEdist s t) Top.top
    h : Not (Eq (EMetric.hausdorffEdist s u) Top.top)
    ⊢ Not (Eq (EMetric.hausdorffEdist t u) Top.top)
  -/
  rw [hausdorffEdist_comm] at fin
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t u : Set α
    fin : Ne (EMetric.hausdorffEdist t s) Top.top
    h : Not (Eq (EMetric.hausdorffEdist s u) Top.top)
    ⊢ Not (Eq (EMetric.hausdorffEdist t u) Top.top)
  -/
  exact ne_top_of_le_ne_top (add_ne_top.2 ⟨fin, h⟩) hausdorffEdist_triangle
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance satisfies the triangle inequality. -/
theorem hausdorffDist_triangle' (fin : hausdorffEdist t u ≠ ⊤) :
    hausdorffDist s u ≤ hausdorffDist s t + hausdorffDist t u := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t u : Set α
    fin : Ne (EMetric.hausdorffEdist t u) Top.top
    ⊢ LE.le (Metric.hausdorffDist s u) (HAdd.hAdd (Metric.hausdorffDist s t) (Metr …
  -/
  rw [hausdorffEdist_comm] at fin
  have I : hausdorffDist u s ≤ hausdorffDist u t + hausdorffDist t s :=
    hausdorffDist_triangle fin
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t u : Set α
    fin : Ne (EMetric.hausdorffEdist u t) Top.top
    I : LE.le (Metric.hausdorffDist u s) (HAdd.hAdd (Metric.hausdorffDist u t) (Me …
    ⊢ LE.le (Metric.hausdorffDist s u) (HAdd.hAdd (Metric.hausdorffDist s t) (Metr …
  -/
  simpa [add_comm, hausdorffDist_comm] using I
  /-
    🎉 no goals
  -/


/-- The Hausdorff distance between a set and its closure vanishes. -/
@[simp]
                                                                           /-
                                                                             α : Type u
                                                                             inst✝ : PseudoMetricSpace α
                                                                             s : Set α
                                                                             ⊢ Eq (Metric.hausdorffDist s (closure s)) 0
                                                                           -/
theorem hausdorffDist_self_closure : hausdorffDist s (closure s) = 0 := by simp [hausdorffDist]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Replacing a set by its closure does not change the Hausdorff distance. -/
@[simp]
theorem hausdorffDist_closure₁ : hausdorffDist (closure s) t = hausdorffDist s t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    ⊢ Eq (Metric.hausdorffDist (closure s) t) (Metric.hausdorffDist s t)
  -/
  simp [hausdorffDist]
  /-
    🎉 no goals
  -/


/-- Replacing a set by its closure does not change the Hausdorff distance. -/
@[simp]
theorem hausdorffDist_closure₂ : hausdorffDist s (closure t) = hausdorffDist s t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    ⊢ Eq (Metric.hausdorffDist s (closure t)) (Metric.hausdorffDist s t)
  -/
  simp [hausdorffDist]
  /-
    🎉 no goals
  -/


/-- The Hausdorff distances between two sets and their closures coincide. -/
theorem hausdorffDist_closure : hausdorffDist (closure s) (closure t) = hausdorffDist s t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    ⊢ Eq (Metric.hausdorffDist (closure s) (closure t)) (Metric.hausdorffDist s t)
  -/
  simp [hausdorffDist]
  /-
    🎉 no goals
  -/


/-- Two sets are at zero Hausdorff distance if and only if they have the same closures. -/
theorem hausdorffDist_zero_iff_closure_eq_closure (fin : hausdorffEdist s t ≠ ⊤) :
    hausdorffDist s t = 0 ↔ closure s = closure t := by
  simp [← hausdorffEdist_zero_iff_closure_eq_closure, hausdorffDist,
    ENNReal.toReal_eq_zero_iff, fin]


/-- Two closed sets are at zero Hausdorff distance if and only if they coincide. -/
theorem _root_.IsClosed.hausdorffDist_zero_iff_eq (hs : IsClosed s) (ht : IsClosed t)
    (fin : hausdorffEdist s t ≠ ⊤) : hausdorffDist s t = 0 ↔ s = t := by
  simp [← hausdorffEdist_zero_iff_eq_of_closed hs ht, hausdorffDist, ENNReal.toReal_eq_zero_iff,
    fin]


