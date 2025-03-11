/-- On a metric space `X` with a measure `μ`, consider for each `x : X` a family of measurable sets
with nonempty interiors, called `setsAt x`. This family is a Vitali family if it satisfies the
following property: consider a (possibly non-measurable) set `s`, and for any `x` in `s` a
subfamily `f x` of `setsAt x` containing sets of arbitrarily small diameter. Then one can extract
a disjoint subfamily covering almost all `s`.

Vitali families are provided by covering theorems such as the Besicovitch covering theorem or the
Vitali covering theorem. They make it possible to formulate general versions of theorems on
differentiations of measure that apply in both contexts.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure VitaliFamily {m : MeasurableSpace X} (μ : Measure X) where
  /-- Sets of the family "centered" at a given point. -/
  setsAt :  X → Set (Set X)
  /-- All sets of the family are measurable. -/
  measurableSet : ∀ x : X, ∀ s ∈ setsAt x, MeasurableSet s
  /-- All sets of the family have nonempty interior. -/
  nonempty_interior : ∀ x : X, ∀ s ∈ setsAt x, (interior s).Nonempty
  /-- For any closed ball around `x`, there exists a set of the family contained in this ball. -/
  nontrivial : ∀ (x : X), ∀ ε > (0 : ℝ), ∃ s ∈ setsAt x, s ⊆ closedBall x ε
  /-- Consider a (possibly non-measurable) set `s`,
  and for any `x` in `s` a subfamily `f x` of `setsAt x`
  containing sets of arbitrarily small diameter.
  Then one can extract a disjoint subfamily covering almost all `s`. -/
  covering : ∀ (s : Set X) (f : X → Set (Set X)),
    (∀ x ∈ s, f x ⊆ setsAt x) → (∀ x ∈ s, ∀ ε > (0 : ℝ), ∃ t ∈ f x, t ⊆ closedBall x ε) →
    ∃ t : Set (X × Set X), (∀ p ∈ t, p.1 ∈ s) ∧ (t.PairwiseDisjoint fun p ↦ p.2) ∧
      (∀ p ∈ t, p.2 ∈ f p.1) ∧ μ (s \ ⋃ p ∈ t, p.2) = 0


/-- A Vitali family for a measure `μ` is also a Vitali family for any measure absolutely continuous
with respect to `μ`. -/
def mono (v : VitaliFamily μ) (ν : Measure X) (hν : ν ≪ μ) : VitaliFamily ν where
  __ := v
  covering s f h h' :=
    let ⟨t, ts, disj, mem_f, hμ⟩ := v.covering s f h h'
    ⟨t, ts, disj, mem_f, hν hμ⟩


/-- Given a Vitali family `v` for a measure `μ`, a family `f` is a fine subfamily on a set `s` if
every point `x` in `s` belongs to arbitrarily small sets in `v.setsAt x ∩ f x`. This is precisely
the subfamilies for which the Vitali family definition ensures that one can extract a disjoint
covering of almost all `s`. -/
def FineSubfamilyOn (v : VitaliFamily μ) (f : X → Set (Set X)) (s : Set X) : Prop :=
  ∀ x ∈ s, ∀ ε > 0, ∃ t ∈ v.setsAt x ∩ f x, t ⊆ closedBall x ε


theorem exists_disjoint_covering_ae :
    ∃ t : Set (X × Set X),
      (∀ p : X × Set X, p ∈ t → p.1 ∈ s) ∧
      (t.PairwiseDisjoint fun p => p.2) ∧
      (∀ p : X × Set X, p ∈ t → p.2 ∈ v.setsAt p.1 ∩ f p.1) ∧
      μ (s \ ⋃ (p : X × Set X) (_ : p ∈ t), p.2) = 0 :=
  v.covering s (fun x => v.setsAt x ∩ f x) (fun _ _ => inter_subset_left) h


/-- Given `h : v.FineSubfamilyOn f s`, then `h.index` is a set parametrizing a disjoint
covering of almost every `s`. -/
protected def index : Set (X × Set X) :=
  h.exists_disjoint_covering_ae.choose

-- Porting note: Needed to add `(_h : FineSubfamilyOn v f s)`

/-- Given `h : v.FineSubfamilyOn f s`, then `h.covering p` is a set in the family,
for `p ∈ h.index`, such that these sets form a disjoint covering of almost every `s`. -/
@[nolint unusedArguments]
protected def covering (_h : FineSubfamilyOn v f s) : X × Set X → Set X :=
  fun p => p.2


theorem index_subset : ∀ p : X × Set X, p ∈ h.index → p.1 ∈ s :=
  h.exists_disjoint_covering_ae.choose_spec.1


theorem covering_disjoint : h.index.PairwiseDisjoint h.covering :=
  h.exists_disjoint_covering_ae.choose_spec.2.1


theorem covering_disjoint_subtype : Pairwise (Disjoint on fun x : h.index => h.covering x) :=
  (pairwise_subtype_iff_pairwise_set _ _).2 h.covering_disjoint


theorem covering_mem {p : X × Set X} (hp : p ∈ h.index) : h.covering p ∈ f p.1 :=
  (h.exists_disjoint_covering_ae.choose_spec.2.2.1 p hp).2


theorem covering_mem_family {p : X × Set X} (hp : p ∈ h.index) : h.covering p ∈ v.setsAt p.1 :=
  (h.exists_disjoint_covering_ae.choose_spec.2.2.1 p hp).1


theorem measure_diff_biUnion : μ (s \ ⋃ p ∈ h.index, h.covering p) = 0 :=
  h.exists_disjoint_covering_ae.choose_spec.2.2.2


theorem index_countable [SecondCountableTopology X] : h.index.Countable :=
  h.covering_disjoint.countable_of_nonempty_interior fun _ hx =>
    v.nonempty_interior _ _ (h.covering_mem_family hx)


protected theorem measurableSet_u {p : X × Set X} (hp : p ∈ h.index) :
    MeasurableSet (h.covering p) :=
  v.measurableSet p.1 _ (h.covering_mem_family hp)


theorem measure_le_tsum_of_absolutelyContinuous [SecondCountableTopology X] {ρ : Measure X}
    (hρ : ρ ≪ μ) : ρ s ≤ ∑' p : h.index, ρ (h.covering p) :=
  calc
    ρ s ≤ ρ ((s \ ⋃ p ∈ h.index, h.covering p) ∪ ⋃ p ∈ h.index, h.covering p) :=
                       /-
                         X : Type u_1
                         inst✝¹ : PseudoMetricSpace X
                         m0 : MeasurableSpace X
                         μ : MeasureTheory.Measure X
                         v : VitaliFamily μ
                         f : X → Set (Set X)
                         s : Set X
                         h : v.FineSubfamilyOn f s
                         inst✝ : SecondCountableTopology X
                         ρ : MeasureTheory.Measure X
                         hρ : ρ.AbsolutelyContinuous μ
                         ⊢ HasSubset.Subset s (Union.union (SDiff.sdiff s (Set.iUnion fun p => Set.iUni …
                       -/
      measure_mono (by simp only [subset_union_left, diff_union_self])
                       /-
                         🎉 no goals
                       -/
    _ ≤ ρ (s \ ⋃ p ∈ h.index, h.covering p) + ρ (⋃ p ∈ h.index, h.covering p) :=
      (measure_union_le _ _)
    _ = ∑' p : h.index, ρ (h.covering p) := by
      rw [hρ h.measure_diff_biUnion, zero_add,
        measure_biUnion h.index_countable h.covering_disjoint fun x hx => h.measurableSet_u hx]


theorem measure_le_tsum [SecondCountableTopology X] : μ s ≤ ∑' x : h.index, μ (h.covering x) :=
  h.measure_le_tsum_of_absolutelyContinuous Measure.AbsolutelyContinuous.rfl


/-- One can enlarge a Vitali family by adding to the sets `f x` at `x` all sets which are not
contained in a `δ`-neighborhood on `x`. This does not change the local filter at a point, but it
can be convenient to get a nicer global behavior. -/
def enlarge (v : VitaliFamily μ) (δ : ℝ) (δpos : 0 < δ) : VitaliFamily μ where
  setsAt x := v.setsAt x ∪ {s | MeasurableSet s ∧ (interior s).Nonempty ∧ ¬s ⊆ closedBall x δ}
  measurableSet := by
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      ⊢ ∀ (x : X) (s : Set X), Membership.mem ((fun x => Union.union (v.setsAt x) (s …
    -/
    rintro x s (hs | hs)
    /-
      case inl
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      x : X
      s : Set X
      hs : Membership.mem (v.setsAt x) s
      ⊢ MeasurableSet s
    -/
    exacts [v.measurableSet _ _ hs, hs.1]
    /-
      🎉 no goals
    -/
  nonempty_interior := by
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      ⊢ ∀ (x : X) (s : Set X), Membership.mem ((fun x => Union.union (v.setsAt x) (s …
    -/
    rintro x s (hs | hs)
    /-
      case inl
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      x : X
      s : Set X
      hs : Membership.mem (v.setsAt x) s
      ⊢ (interior s).Nonempty
    -/
    exacts [v.nonempty_interior _ _ hs, hs.2.1]
    /-
      🎉 no goals
    -/
  nontrivial := by
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      ⊢ ∀ (x : X) (ε : Real), GT.gt ε 0 → Exists fun s => And (Membership.mem ((fun  …
    -/
    intro x ε εpos
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      x : X
      ε : Real
      εpos : GT.gt ε 0
      ⊢ Exists fun s => And (Membership.mem ((fun x => Union.union (v.setsAt x) (set …
    -/
    rcases v.nontrivial x ε εpos with ⟨s, hs, h's⟩
    /-
      case intro.intro
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      x : X
      ε : Real
      εpos : GT.gt ε 0
      s : Set X
      hs : Membership.mem (v.setsAt x) s
      h's : HasSubset.Subset s (Metric.closedBall x ε)
      ⊢ Exists fun s => And (Membership.mem ((fun x => Union.union (v.setsAt x) (set …
    -/
    exact ⟨s, mem_union_left _ hs, h's⟩
    /-
      🎉 no goals
    -/
  covering := by
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      ⊢ ∀ (s : Set X) (f : X → Set (Set X)), (∀ (x : X), Membership.mem s x → HasSub …
    -/
    intro s f fset ffine
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      s : Set X
      f : X → Set (Set X)
      fset : ∀ (x : X), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Union …
      ffine : ∀ (x : X), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      ⊢ Exists fun t => And (∀ (p : Prod X (Set X)), Membership.mem t p → Membership …
    -/
    let g : X → Set (Set X) := fun x => f x ∩ v.setsAt x
    have : ∀ x ∈ s, ∀ ε : ℝ, ε > 0 → ∃ t ∈ g x, t ⊆ closedBall x ε := by
      intro x hx ε εpos
      obtain ⟨t, tf, ht⟩ : ∃ t ∈ f x, t ⊆ closedBall x (min ε δ) :=
        ffine x hx (min ε δ) (lt_min εpos δpos)
      rcases fset x hx tf with (h't | h't)
      · exact ⟨t, ⟨tf, h't⟩, ht.trans (closedBall_subset_closedBall (min_le_left _ _))⟩
      · refine False.elim (h't.2.2 ?_)
        exact ht.trans (closedBall_subset_closedBall (min_le_right _ _))
    /-
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      s : Set X
      f : X → Set (Set X)
      fset : ∀ (x : X), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Union …
      ffine : ∀ (x : X), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      g : X → Set (Set X) := fun x => Inter.inter (f x) (v.setsAt x)
      this : ∀ (x : X), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t  …
      ⊢ Exists fun t => And (∀ (p : Prod X (Set X)), Membership.mem t p → Membership …
    -/
    rcases v.covering s g (fun x _ => inter_subset_right) this with ⟨t, ts, tdisj, tg, μt⟩
    /-
      case intro.intro.intro.intro
      X : Type u_1
      inst✝ : PseudoMetricSpace X
      m0 : MeasurableSpace X
      μ : MeasureTheory.Measure X
      v : VitaliFamily μ
      δ : Real
      δpos : LT.lt 0 δ
      s : Set X
      f : X → Set (Set X)
      fset : ∀ (x : X), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Union …
      ffine : ∀ (x : X), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      g : X → Set (Set X) := fun x => Inter.inter (f x) (v.setsAt x)
      this : ∀ (x : X), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t  …
      t : Set (Prod X (Set X))
      ts : ∀ (p : Prod X (Set X)), Membership.mem t p → Membership.mem s p.1
      tdisj : t.PairwiseDisjoint fun p => p.2
      tg : ∀ (p : Prod X (Set X)), Membership.mem t p → Membership.mem (g p.1) p.2
      μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun h => p.2))) 0
      ⊢ Exists fun t => And (∀ (p : Prod X (Set X)), Membership.mem t p → Membership …
    -/
    exact ⟨t, ts, tdisj, fun p hp => (tg p hp).1, μt⟩
    /-
      🎉 no goals
    -/


/-- Given a vitali family `v`, then `v.filterAt x` is the filter on `Set X` made of those families
that contain all sets of `v.setsAt x` of a sufficiently small diameter. This filter makes it
possible to express limiting behavior when sets in `v.setsAt x` shrink to `x`. -/
def filterAt (x : X) : Filter (Set X) := (𝓝 x).smallSets ⊓ 𝓟 (v.setsAt x)


theorem _root_.Filter.HasBasis.vitaliFamily {ι : Sort*} {p : ι → Prop} {s : ι → Set X} {x : X}
    (h : (𝓝 x).HasBasis p s) : (v.filterAt x).HasBasis p (fun i ↦ {t ∈ v.setsAt x | t ⊆ s i}) := by
  /-
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set X
    x : X
    h : (nhds x).HasBasis p s
    ⊢ (v.filterAt x).HasBasis p fun i => setOf fun t => And (Membership.mem (v.set …
  -/
  simpa only [← Set.setOf_inter_eq_sep] using h.smallSets.inf_principal _
  /-
    🎉 no goals
  -/


theorem filterAt_basis_closedBall (x : X) :
    (v.filterAt x).HasBasis (0 < ·) ({t ∈ v.setsAt x | t ⊆ closedBall x ·}) :=
  nhds_basis_closedBall.vitaliFamily v


theorem mem_filterAt_iff {x : X} {s : Set (Set X)} :
    s ∈ v.filterAt x ↔ ∃ ε > (0 : ℝ), ∀ t ∈ v.setsAt x, t ⊆ closedBall x ε → t ∈ s := by
  /-
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    x : X
    s : Set (Set X)
    ⊢ Iff (Membership.mem (v.filterAt x) s) (Exists fun ε => And (GT.gt ε 0) (∀ (t …
  -/
  simp only [(v.filterAt_basis_closedBall x).mem_iff, ← and_imp, subset_def, mem_setOf]
  /-
    🎉 no goals
  -/


instance filterAt_neBot (x : X) : (v.filterAt x).NeBot :=
  (v.filterAt_basis_closedBall x).neBot_iff.2 <| v.nontrivial _ _


theorem eventually_filterAt_iff {x : X} {P : Set X → Prop} :
    (∀ᶠ t in v.filterAt x, P t) ↔ ∃ ε > (0 : ℝ), ∀ t ∈ v.setsAt x, t ⊆ closedBall x ε → P t :=
  v.mem_filterAt_iff


theorem tendsto_filterAt_iff {ι : Type*} {l : Filter ι} {f : ι → Set X} {x : X} :
    Tendsto f l (v.filterAt x) ↔
      (∀ᶠ i in l, f i ∈ v.setsAt x) ∧ ∀ ε > (0 : ℝ), ∀ᶠ i in l, f i ⊆ closedBall x ε := by
  simp only [filterAt, tendsto_inf, nhds_basis_closedBall.smallSets.tendsto_right_iff,
    tendsto_principal, and_comm, mem_powerset_iff]


theorem eventually_filterAt_mem_setsAt (x : X) : ∀ᶠ t in v.filterAt x, t ∈ v.setsAt x :=
  (v.tendsto_filterAt_iff.mp tendsto_id).1


theorem eventually_filterAt_subset_closedBall (x : X) {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ t : Set X in v.filterAt x, t ⊆ closedBall x ε :=
  (v.tendsto_filterAt_iff.mp tendsto_id).2 ε hε


theorem eventually_filterAt_measurableSet (x : X) : ∀ᶠ t in v.filterAt x, MeasurableSet t := by
  /-
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    x : X
    ⊢ Filter.Eventually (fun t => MeasurableSet t) (v.filterAt x)
  -/
  filter_upwards [v.eventually_filterAt_mem_setsAt x] with _ ha using v.measurableSet _ _ ha
  /-
    🎉 no goals
  -/


theorem frequently_filterAt_iff {x : X} {P : Set X → Prop} :
    (∃ᶠ t in v.filterAt x, P t) ↔ ∀ ε > (0 : ℝ), ∃ t ∈ v.setsAt x, t ⊆ closedBall x ε ∧ P t := by
  /-
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    x : X
    P : Set X → Prop
    ⊢ Iff (Filter.Frequently (fun t => P t) (v.filterAt x)) (∀ (ε : Real), GT.gt ε …
  -/
  simp only [(v.filterAt_basis_closedBall x).frequently_iff, ← and_assoc, subset_def, mem_setOf]
  /-
    🎉 no goals
  -/


theorem eventually_filterAt_subset_of_nhds {x : X} {o : Set X} (hx : o ∈ 𝓝 x) :
    ∀ᶠ t in v.filterAt x, t ⊆ o :=
  (eventually_smallSets_subset.2 hx).filter_mono inf_le_left


theorem fineSubfamilyOn_of_frequently (v : VitaliFamily μ) (f : X → Set (Set X)) (s : Set X)
    (h : ∀ x ∈ s, ∃ᶠ t in v.filterAt x, t ∈ f x) : v.FineSubfamilyOn f s := by
  /-
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    f : X → Set (Set X)
    s : Set X
    h : ∀ (x : X), Membership.mem s x → Filter.Frequently (fun t => Membership.mem …
    ⊢ v.FineSubfamilyOn f s
  -/
  intro x hx ε εpos
  obtain ⟨t, tv, ht, tf⟩ : ∃ t ∈ v.setsAt x, t ⊆ closedBall x ε ∧ t ∈ f x :=
    v.frequently_filterAt_iff.1 (h x hx) ε εpos
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝ : PseudoMetricSpace X
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    v : VitaliFamily μ
    f : X → Set (Set X)
    s : Set X
    h : ∀ (x : X), Membership.mem s x → Filter.Frequently (fun t => Membership.mem …
    x : X
    hx : Membership.mem s x
    ε : Real
    εpos : GT.gt ε 0
    t : Set X
    tv : Membership.mem (v.setsAt x) t
    ht : HasSubset.Subset t (Metric.closedBall x ε)
    tf : Membership.mem (f x) t
    ⊢ Exists fun t => And (Membership.mem (Inter.inter (v.setsAt x) (f x)) t) (Has …
  -/
  exact ⟨t, ⟨tv, tf⟩, ht⟩
  /-
    🎉 no goals
  -/


