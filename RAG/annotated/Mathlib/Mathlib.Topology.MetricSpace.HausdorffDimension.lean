/-- Hausdorff dimension of a set in an (e)metric space. -/
@[irreducible] noncomputable def dimH (s : Set X) : ℝ≥0∞ := by
  /-
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    s : Set X
    ⊢ ENNReal
  -/
  borelize X; exact ⨆ (d : ℝ≥0) (_ : @hausdorffMeasure X _ _ ⟨rfl⟩ d s = ∞), d
              /-
                🎉 no goals
              -/


/-- Unfold the definition of `dimH` using `[MeasurableSpace X] [BorelSpace X]` from the
environment. -/
theorem dimH_def (s : Set X) : dimH s = ⨆ (d : ℝ≥0) (_ : μH[d] s = ∞), (d : ℝ≥0∞) := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    ⊢ Eq (dimH s) (iSup fun d => iSup fun x => ↑d)
  -/
  borelize X; rw [dimH]
              /-
                🎉 no goals
              -/


theorem hausdorffMeasure_of_lt_dimH {s : Set X} {d : ℝ≥0} (h : ↑d < dimH s) : μH[d] s = ∞ := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : LT.lt (↑d) (dimH s)
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  simp only [dimH_def, lt_iSup_iff] at h
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : Exists fun i => Exists fun i_1 => LT.lt ↑d ↑i
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  rcases h with ⟨d', hsd', hdd'⟩
  /-
    case intro.intro
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d d' : NNReal
    hsd' : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d') s) Top.top
    hdd' : LT.lt ↑d ↑d'
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  rw [ENNReal.coe_lt_coe, ← NNReal.coe_lt_coe] at hdd'
  /-
    case intro.intro
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d d' : NNReal
    hsd' : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d') s) Top.top
    hdd' : LT.lt ↑d ↑d'
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  exact top_unique (hsd' ▸ hausdorffMeasure_mono hdd'.le _)
  /-
    🎉 no goals
  -/


theorem dimH_le {s : Set X} {d : ℝ≥0∞} (H : ∀ d' : ℝ≥0, μH[d'] s = ∞ → ↑d' ≤ d) : dimH s ≤ d :=
  (dimH_def s).trans_le <| iSup₂_le H


theorem dimH_le_of_hausdorffMeasure_ne_top {s : Set X} {d : ℝ≥0} (h : μH[d] s ≠ ∞) : dimH s ≤ d :=
  le_of_not_lt <| mt hausdorffMeasure_of_lt_dimH h


theorem le_dimH_of_hausdorffMeasure_eq_top {s : Set X} {d : ℝ≥0} (h : μH[d] s = ∞) :
    ↑d ≤ dimH s := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
    ⊢ LE.le (↑d) (dimH s)
  -/
  rw [dimH_def]; exact le_iSup₂ (α := ℝ≥0∞) d h
                 /-
                   🎉 no goals
                 -/


theorem hausdorffMeasure_of_dimH_lt {s : Set X} {d : ℝ≥0} (h : dimH s < d) : μH[d] s = 0 := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : LT.lt (dimH s) ↑d
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) 0
  -/
  rw [dimH_def] at h
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : LT.lt (iSup fun d => iSup fun x => ↑d) ↑d
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) 0
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 h with ⟨d', hsd', hd'd⟩
  /-
    case intro.intro
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    d : NNReal
    h : LT.lt (iSup fun d => iSup fun x => ↑d) ↑d
    d' : NNReal
    hsd' : LT.lt (iSup fun d => iSup fun x => ↑d) ↑d'
    hd'd : LT.lt ↑d' ↑d
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) 0
  -/
  rw [ENNReal.coe_lt_coe, ← NNReal.coe_lt_coe] at hd'd
  exact (hausdorffMeasure_zero_or_top hd'd s).resolve_right fun h₂ => hsd'.not_le <|
    le_iSup₂ (α := ℝ≥0∞) d' h₂


theorem measure_zero_of_dimH_lt {μ : Measure X} {d : ℝ≥0} (h : μ ≪ μH[d]) {s : Set X}
    (hd : dimH s < d) : μ s = 0 :=
  h <| hausdorffMeasure_of_dimH_lt hd


theorem le_dimH_of_hausdorffMeasure_ne_zero {s : Set X} {d : ℝ≥0} (h : μH[d] s ≠ 0) : ↑d ≤ dimH s :=
  le_of_not_lt <| mt hausdorffMeasure_of_dimH_lt h


theorem dimH_of_hausdorffMeasure_ne_zero_ne_top {d : ℝ≥0} {s : Set X} (h : μH[d] s ≠ 0)
    (h' : μH[d] s ≠ ∞) : dimH s = d :=
  le_antisymm (dimH_le_of_hausdorffMeasure_ne_top h') (le_dimH_of_hausdorffMeasure_ne_zero h)


@[mono]
theorem dimH_mono {s t : Set X} (h : s ⊆ t) : dimH s ≤ dimH t := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s t : Set X
    h : HasSubset.Subset s t
    ⊢ LE.le (dimH s) (dimH t)
  -/
  borelize X
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s t : Set X
    h : HasSubset.Subset s t
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ LE.le (dimH s) (dimH t)
  -/
  exact dimH_le fun d hd => le_dimH_of_hausdorffMeasure_eq_top <| top_unique <| hd ▸ measure_mono h
  /-
    🎉 no goals
  -/


theorem dimH_subsingleton {s : Set X} (h : s.Subsingleton) : dimH s = 0 := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set X
    h : s.Subsingleton
    ⊢ Eq (dimH s) 0
  -/
  borelize X
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set X
    h : s.Subsingleton
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ Eq (dimH s) 0
  -/
  apply le_antisymm _ (zero_le _)
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set X
    h : s.Subsingleton
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ LE.le (dimH s) 0
  -/
  refine dimH_le_of_hausdorffMeasure_ne_top ?_
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set X
    h : s.Subsingleton
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ Ne ((MeasureTheory.Measure.hausdorffMeasure ↑0) s) Top.top
  -/
  exact ((hausdorffMeasure_le_one_of_subsingleton h le_rfl).trans_lt ENNReal.one_lt_top).ne
  /-
    🎉 no goals
  -/


alias Set.Subsingleton.dimH_zero := dimH_subsingleton


@[simp]
theorem dimH_empty : dimH (∅ : Set X) = 0 :=
  subsingleton_empty.dimH_zero


@[simp]
theorem dimH_singleton (x : X) : dimH ({x} : Set X) = 0 :=
  subsingleton_singleton.dimH_zero


@[simp]
theorem dimH_iUnion {ι : Sort*} [Countable ι] (s : ι → Set X) :
    dimH (⋃ i, s i) = ⨆ i, dimH (s i) := by
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set X
    ⊢ Eq (dimH (Set.iUnion fun i => s i)) (iSup fun i => dimH (s i))
  -/
  borelize X
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set X
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ Eq (dimH (Set.iUnion fun i => s i)) (iSup fun i => dimH (s i))
  -/
  refine le_antisymm (dimH_le fun d hd => ?_) (iSup_le fun i => dimH_mono <| subset_iUnion _ _)
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set X
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.iUnion fun i => s i) …
    ⊢ LE.le (↑d) (iSup fun i => dimH (s i))
  -/
  contrapose! hd
  have : ∀ i, μH[d] (s i) = 0 := fun i =>
    hausdorffMeasure_of_dimH_lt ((le_iSup (fun i => dimH (s i)) i).trans_lt hd)
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set X
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    d : NNReal
    hd : LT.lt (iSup fun i => dimH (s i)) ↑d
    this : ∀ (i : ι), Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (s i)) 0
    ⊢ Ne ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.iUnion fun i => s i)) T …
  -/
  rw [measure_iUnion_null this]
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set X
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    d : NNReal
    hd : LT.lt (iSup fun i => dimH (s i)) ↑d
    this : ∀ (i : ι), Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (s i)) 0
    ⊢ Ne 0 Top.top
  -/
  exact ENNReal.zero_ne_top
  /-
    🎉 no goals
  -/


@[simp]
theorem dimH_bUnion {s : Set ι} (hs : s.Countable) (t : ι → Set X) :
    dimH (⋃ i ∈ s, t i) = ⨆ i ∈ s, dimH (t i) := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set ι
    hs : s.Countable
    t : ι → Set X
    ⊢ Eq (dimH (Set.iUnion fun i => Set.iUnion fun h => t i)) (iSup fun i => iSup  …
  -/
  haveI := hs.toEncodable
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    s : Set ι
    hs : s.Countable
    t : ι → Set X
    this : Encodable ↑s
    ⊢ Eq (dimH (Set.iUnion fun i => Set.iUnion fun h => t i)) (iSup fun i => iSup  …
  -/
  rw [biUnion_eq_iUnion, dimH_iUnion, ← iSup_subtype'']
  /-
    🎉 no goals
  -/


@[simp]
theorem dimH_sUnion {S : Set (Set X)} (hS : S.Countable) : dimH (⋃₀ S) = ⨆ s ∈ S, dimH s := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    S : Set (Set X)
    hS : S.Countable
    ⊢ Eq (dimH S.sUnion) (iSup fun s => iSup fun h => dimH s)
  -/
  rw [sUnion_eq_biUnion, dimH_bUnion hS]
  /-
    🎉 no goals
  -/


@[simp]
theorem dimH_union (s t : Set X) : dimH (s ∪ t) = max (dimH s) (dimH t) := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    s t : Set X
    ⊢ Eq (dimH (Union.union s t)) (Max.max (dimH s) (dimH t))
  -/
  rw [union_eq_iUnion, dimH_iUnion, iSup_bool_eq, cond, cond]
  /-
    🎉 no goals
  -/


theorem dimH_countable {s : Set X} (hs : s.Countable) : dimH s = 0 :=
                              /-
                                X : Type u_2
                                inst✝ : EMetricSpace X
                                s : Set X
                                hs : s.Countable
                                ⊢ Eq (dimH (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton x)) 0
                              -/
  biUnion_of_singleton s ▸ by simp only [dimH_bUnion hs, dimH_singleton, ENNReal.iSup_zero]
                              /-
                                🎉 no goals
                              -/


alias Set.Countable.dimH_zero := dimH_countable


theorem dimH_finite {s : Set X} (hs : s.Finite) : dimH s = 0 :=
  hs.countable.dimH_zero


alias Set.Finite.dimH_zero := dimH_finite


@[simp]
theorem dimH_coe_finset (s : Finset X) : dimH (s : Set X) = 0 :=
  s.finite_toSet.dimH_zero


alias Finset.dimH_zero := dimH_coe_finset


/-- If `r` is less than the Hausdorff dimension of a set `s` in an (extended) metric space with
second countable topology, then there exists a point `x ∈ s` such that every neighborhood
`t` of `x` within `s` has Hausdorff dimension greater than `r`. -/
theorem exists_mem_nhdsWithin_lt_dimH_of_lt_dimH {s : Set X} {r : ℝ≥0∞} (h : r < dimH s) :
    ∃ x ∈ s, ∀ t ∈ 𝓝[s] x, r < dimH t := by
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    inst✝ : SecondCountableTopology X
    s : Set X
    r : ENNReal
    h : LT.lt r (dimH s)
    ⊢ Exists fun x => And (Membership.mem s x) (∀ (t : Set X), Membership.mem (nhd …
  -/
  contrapose! h; choose! t htx htr using h
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    inst✝ : SecondCountableTopology X
    s : Set X
    r : ENNReal
    t : X → Set X
    htx : ∀ (x : X), Membership.mem s x → Membership.mem (nhdsWithin x s) (t x)
    htr : ∀ (x : X), Membership.mem s x → LE.le (dimH (t x)) r
    ⊢ LE.le (dimH s) r
  -/
  rcases countable_cover_nhdsWithin htx with ⟨S, hSs, hSc, hSU⟩
  calc
    dimH s ≤ dimH (⋃ x ∈ S, t x) := dimH_mono hSU
    _ = ⨆ x ∈ S, dimH (t x) := dimH_bUnion hSc _
    _ ≤ r := iSup₂_le fun x hx => htr x <| hSs hx


/-- In an (extended) metric space with second countable topology, the Hausdorff dimension
of a set `s` is the supremum over `x ∈ s` of the limit superiors of `dimH t` along
`(𝓝[s] x).smallSets`. -/
theorem bsupr_limsup_dimH (s : Set X) : ⨆ x ∈ s, limsup dimH (𝓝[s] x).smallSets = dimH s := by
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    inst✝ : SecondCountableTopology X
    s : Set X
    ⊢ Eq (iSup fun x => iSup fun h => Filter.limsup dimH (nhdsWithin x s).smallSet …
  -/
  refine le_antisymm (iSup₂_le fun x _ => ?_) ?_
    /-
      case refine_1
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      x : X
      x✝ : Membership.mem s x
      ⊢ LE.le (Filter.limsup dimH (nhdsWithin x s).smallSets) (dimH s)
    -/
  · refine limsup_le_of_le isCobounded_le_of_bot ?_
    /-
      case refine_1
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      x : X
      x✝ : Membership.mem s x
      ⊢ Filter.Eventually (fun n => LE.le (dimH n) (dimH s)) (nhdsWithin x s).smallS …
    -/
    exact eventually_smallSets.2 ⟨s, self_mem_nhdsWithin, fun t => dimH_mono⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      ⊢ LE.le (dimH s) (iSup fun x => iSup fun h => Filter.limsup dimH (nhdsWithin x …
    -/
  · refine le_of_forall_ge_of_dense fun r hr => ?_
    /-
      case refine_2
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      r : ENNReal
      hr : LT.lt r (dimH s)
      ⊢ LE.le r (iSup fun x => iSup fun h => Filter.limsup dimH (nhdsWithin x s).sma …
    -/
    rcases exists_mem_nhdsWithin_lt_dimH_of_lt_dimH hr with ⟨x, hxs, hxr⟩
    /-
      case refine_2.intro.intro
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      r : ENNReal
      hr : LT.lt r (dimH s)
      x : X
      hxs : Membership.mem s x
      hxr : ∀ (t : Set X), Membership.mem (nhdsWithin x s) t → LT.lt r (dimH t)
      ⊢ LE.le r (iSup fun x => iSup fun h => Filter.limsup dimH (nhdsWithin x s).sma …
    -/
    refine le_iSup₂_of_le x hxs ?_; rw [limsup_eq]; refine le_sInf fun b hb => ?_
    /-
      case refine_2.intro.intro
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      r : ENNReal
      hr : LT.lt r (dimH s)
      x : X
      hxs : Membership.mem s x
      hxr : ∀ (t : Set X), Membership.mem (nhdsWithin x s) t → LT.lt r (dimH t)
      b : ENNReal
      hb : Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le (dimH n) …
      ⊢ LE.le r b
    -/
    rcases eventually_smallSets.1 hb with ⟨t, htx, ht⟩
    /-
      case refine_2.intro.intro.intro.intro
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      r : ENNReal
      hr : LT.lt r (dimH s)
      x : X
      hxs : Membership.mem s x
      hxr : ∀ (t : Set X), Membership.mem (nhdsWithin x s) t → LT.lt r (dimH t)
      b : ENNReal
      hb : Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le (dimH n) …
      t : Set X
      htx : Membership.mem (nhdsWithin x s) t
      ht : ∀ (t_1 : Set X), HasSubset.Subset t_1 t → LE.le (dimH t_1) b
      ⊢ LE.le r b
    -/
    exact (hxr t htx).le.trans (ht t Subset.rfl)
    /-
      🎉 no goals
    -/


/-- In an (extended) metric space with second countable topology, the Hausdorff dimension
of a set `s` is the supremum over all `x` of the limit superiors of `dimH t` along
`(𝓝[s] x).smallSets`. -/
theorem iSup_limsup_dimH (s : Set X) : ⨆ x, limsup dimH (𝓝[s] x).smallSets = dimH s := by
  /-
    X : Type u_2
    inst✝¹ : EMetricSpace X
    inst✝ : SecondCountableTopology X
    s : Set X
    ⊢ Eq (iSup fun x => Filter.limsup dimH (nhdsWithin x s).smallSets) (dimH s)
  -/
  refine le_antisymm (iSup_le fun x => ?_) ?_
    /-
      case refine_1
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      x : X
      ⊢ LE.le (Filter.limsup dimH (nhdsWithin x s).smallSets) (dimH s)
    -/
  · refine limsup_le_of_le isCobounded_le_of_bot ?_
    /-
      case refine_1
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      x : X
      ⊢ Filter.Eventually (fun n => LE.le (dimH n) (dimH s)) (nhdsWithin x s).smallS …
    -/
    exact eventually_smallSets.2 ⟨s, self_mem_nhdsWithin, fun t => dimH_mono⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_2
      inst✝¹ : EMetricSpace X
      inst✝ : SecondCountableTopology X
      s : Set X
      ⊢ LE.le (dimH s) (iSup fun x => Filter.limsup dimH (nhdsWithin x s).smallSets)
    -/
  · rw [← bsupr_limsup_dimH]; exact iSup₂_le_iSup _ _
                              /-
                                🎉 no goals
                              -/


/-- If `f` is a Hölder continuous map with exponent `r > 0`, then `dimH (f '' s) ≤ dimH s / r`. -/
theorem HolderOnWith.dimH_image_le (h : HolderOnWith C r f s) (hr : 0 < r) :
    dimH (f '' s) ≤ dimH s / r := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  borelize X Y
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  refine dimH_le fun d hd => ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.image f s)) Top.top
    ⊢ LE.le (↑d) (HDiv.hDiv (dimH s) ↑r)
  -/
  have := h.hausdorffMeasure_image_le hr d.coe_nonneg
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.image f s)) Top.top
    this : LE.le ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.image f s)) (HM …
    ⊢ LE.le (↑d) (HDiv.hDiv (dimH s) ↑r)
  -/
  rw [hd, ← ENNReal.coe_rpow_of_nonneg _ d.coe_nonneg, top_le_iff] at this
  have Hrd : μH[(r * d : ℝ≥0)] s = ⊤ := by
    contrapose this
    exact ENNReal.mul_ne_top ENNReal.coe_ne_top this
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.image f s)) Top.top
    this : Eq (HMul.hMul (↑(HPow.hPow C ↑d)) ((MeasureTheory.Measure.hausdorffMeas …
    Hrd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑(HMul.hMul r d)) s) Top.top
    ⊢ LE.le (↑d) (HDiv.hDiv (dimH s) ↑r)
  -/
  rw [ENNReal.le_div_iff_mul_le, mul_comm, ← ENNReal.coe_mul]
  exacts [le_dimH_of_hausdorffMeasure_eq_top Hrd, Or.inl (mt ENNReal.coe_eq_zero.1 hr.ne'),
    Or.inl ENNReal.coe_ne_top]


/-- If `f : X → Y` is Hölder continuous with a positive exponent `r`, then the Hausdorff dimension
of the image of a set `s` is at most `dimH s / r`. -/
theorem dimH_image_le (h : HolderWith C r f) (hr : 0 < r) (s : Set X) :
    dimH (f '' s) ≤ dimH s / r :=
  (h.holderOnWith s).dimH_image_le hr


/-- If `f` is a Hölder continuous map with exponent `r > 0`, then the Hausdorff dimension of its
range is at most the Hausdorff dimension of its domain divided by `r`. -/
theorem dimH_range_le (h : HolderWith C r f) (hr : 0 < r) :
    dimH (range f) ≤ dimH (univ : Set X) / r :=
  @image_univ _ _ f ▸ h.dimH_image_le hr univ


/-- If `s` is a set in a space `X` with second countable topology and `f : X → Y` is Hölder
continuous in a neighborhood within `s` of every point `x ∈ s` with the same positive exponent `r`
but possibly different coefficients, then the Hausdorff dimension of the image `f '' s` is at most
the Hausdorff dimension of `s` divided by `r`. -/
theorem dimH_image_le_of_locally_holder_on [SecondCountableTopology X] {r : ℝ≥0} {f : X → Y}
    (hr : 0 < r) {s : Set X} (hf : ∀ x ∈ s, ∃ C : ℝ≥0, ∃ t ∈ 𝓝[s] x, HolderOnWith C r f t) :
    dimH (f '' s) ≤ dimH s / r := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    s : Set X
    hf : ∀ (x : X), Membership.mem s x → Exists fun C => Exists fun t => And (Memb …
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  choose! C t htn hC using hf
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    s : Set X
    C : X → NNReal
    t : X → Set X
    htn : ∀ (x : X), Membership.mem s x → Membership.mem (nhdsWithin x s) (t x)
    hC : ∀ (x : X), Membership.mem s x → HolderOnWith (C x) r f (t x)
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  rcases countable_cover_nhdsWithin htn with ⟨u, hus, huc, huU⟩
  /-
    case intro.intro.intro
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    s : Set X
    C : X → NNReal
    t : X → Set X
    htn : ∀ (x : X), Membership.mem s x → Membership.mem (nhdsWithin x s) (t x)
    hC : ∀ (x : X), Membership.mem s x → HolderOnWith (C x) r f (t x)
    u : Set X
    hus : HasSubset.Subset u s
    huc : u.Countable
    huU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => t x)
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  replace huU := inter_eq_self_of_subset_left huU; rw [inter_iUnion₂] at huU
  /-
    case intro.intro.intro
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    s : Set X
    C : X → NNReal
    t : X → Set X
    htn : ∀ (x : X), Membership.mem s x → Membership.mem (nhdsWithin x s) (t x)
    hC : ∀ (x : X), Membership.mem s x → HolderOnWith (C x) r f (t x)
    u : Set X
    hus : HasSubset.Subset u s
    huc : u.Countable
    huU : Eq (Set.iUnion fun i => Set.iUnion fun j => Inter.inter s (t i)) s
    ⊢ LE.le (dimH (Set.image f s)) (HDiv.hDiv (dimH s) ↑r)
  -/
  rw [← huU, image_iUnion₂, dimH_bUnion huc, dimH_bUnion huc]; simp only [ENNReal.iSup_div]
  /-
    case intro.intro.intro
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    s : Set X
    C : X → NNReal
    t : X → Set X
    htn : ∀ (x : X), Membership.mem s x → Membership.mem (nhdsWithin x s) (t x)
    hC : ∀ (x : X), Membership.mem s x → HolderOnWith (C x) r f (t x)
    u : Set X
    hus : HasSubset.Subset u s
    huc : u.Countable
    huU : Eq (Set.iUnion fun i => Set.iUnion fun j => Inter.inter s (t i)) s
    ⊢ LE.le (iSup fun i => iSup fun h => dimH (Set.image f (Inter.inter s (t i)))) …
  -/
  exact iSup₂_mono fun x hx => ((hC x (hus hx)).mono inter_subset_right).dimH_image_le hr
  /-
    🎉 no goals
  -/


/-- If `f : X → Y` is Hölder continuous in a neighborhood of every point `x : X` with the same
positive exponent `r` but possibly different coefficients, then the Hausdorff dimension of the range
of `f` is at most the Hausdorff dimension of `X` divided by `r`. -/
theorem dimH_range_le_of_locally_holder_on [SecondCountableTopology X] {r : ℝ≥0} {f : X → Y}
    (hr : 0 < r) (hf : ∀ x : X, ∃ C : ℝ≥0, ∃ s ∈ 𝓝 x, HolderOnWith C r f s) :
    dimH (range f) ≤ dimH (univ : Set X) / r := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    ⊢ LE.le (dimH (Set.range f)) (HDiv.hDiv (dimH Set.univ) ↑r)
  -/
  rw [← image_univ]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    ⊢ LE.le (dimH (Set.image f Set.univ)) (HDiv.hDiv (dimH Set.univ) ↑r)
  -/
  refine dimH_image_le_of_locally_holder_on hr fun x _ => ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    r : NNReal
    f : X → Y
    hr : LT.lt 0 r
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    x : X
    x✝ : Membership.mem Set.univ x
    ⊢ Exists fun C => Exists fun t => And (Membership.mem (nhdsWithin x Set.univ)  …
  -/
  simpa only [exists_prop, nhdsWithin_univ] using hf x
  /-
    🎉 no goals
  -/


/-- If `f : X → Y` is Lipschitz continuous on `s`, then `dimH (f '' s) ≤ dimH s`. -/
theorem LipschitzOnWith.dimH_image_le (h : LipschitzOnWith K f s) : dimH (f '' s) ≤ dimH s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    s : Set X
    h : LipschitzOnWith K f s
    ⊢ LE.le (dimH (Set.image f s)) (dimH s)
  -/
  simpa using h.holderOnWith.dimH_image_le zero_lt_one
  /-
    🎉 no goals
  -/


/-- If `f` is a Lipschitz continuous map, then `dimH (f '' s) ≤ dimH s`. -/
theorem dimH_image_le (h : LipschitzWith K f) (s : Set X) : dimH (f '' s) ≤ dimH s :=
  h.lipschitzOnWith.dimH_image_le


/-- If `f` is a Lipschitz continuous map, then the Hausdorff dimension of its range is at most the
Hausdorff dimension of its domain. -/
theorem dimH_range_le (h : LipschitzWith K f) : dimH (range f) ≤ dimH (univ : Set X) :=
  @image_univ _ _ f ▸ h.dimH_image_le univ


/-- If `s` is a set in an extended metric space `X` with second countable topology and `f : X → Y`
is Lipschitz in a neighborhood within `s` of every point `x ∈ s`, then the Hausdorff dimension of
the image `f '' s` is at most the Hausdorff dimension of `s`. -/
theorem dimH_image_le_of_locally_lipschitzOn [SecondCountableTopology X] {f : X → Y} {s : Set X}
    (hf : ∀ x ∈ s, ∃ C : ℝ≥0, ∃ t ∈ 𝓝[s] x, LipschitzOnWith C f t) : dimH (f '' s) ≤ dimH s := by
  have : ∀ x ∈ s, ∃ C : ℝ≥0, ∃ t ∈ 𝓝[s] x, HolderOnWith C 1 f t := by
    simpa only [holderOnWith_one] using hf
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    f : X → Y
    s : Set X
    hf : ∀ (x : X), Membership.mem s x → Exists fun C => Exists fun t => And (Memb …
    this : ∀ (x : X), Membership.mem s x → Exists fun C => Exists fun t => And (Me …
    ⊢ LE.le (dimH (Set.image f s)) (dimH s)
  -/
  simpa only [ENNReal.coe_one, div_one] using dimH_image_le_of_locally_holder_on zero_lt_one this
  /-
    🎉 no goals
  -/


/-- If `f : X → Y` is Lipschitz in a neighborhood of each point `x : X`, then the Hausdorff
dimension of `range f` is at most the Hausdorff dimension of `X`. -/
theorem dimH_range_le_of_locally_lipschitzOn [SecondCountableTopology X] {f : X → Y}
    (hf : ∀ x : X, ∃ C : ℝ≥0, ∃ s ∈ 𝓝 x, LipschitzOnWith C f s) :
    dimH (range f) ≤ dimH (univ : Set X) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    f : X → Y
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    ⊢ LE.le (dimH (Set.range f)) (dimH Set.univ)
  -/
  rw [← image_univ]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    f : X → Y
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    ⊢ LE.le (dimH (Set.image f Set.univ)) (dimH Set.univ)
  -/
  refine dimH_image_le_of_locally_lipschitzOn fun x _ => ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝² : EMetricSpace X
    inst✝¹ : EMetricSpace Y
    inst✝ : SecondCountableTopology X
    f : X → Y
    hf : ∀ (x : X), Exists fun C => Exists fun s => And (Membership.mem (nhds x) s …
    x : X
    x✝ : Membership.mem Set.univ x
    ⊢ Exists fun C => Exists fun t => And (Membership.mem (nhdsWithin x Set.univ)  …
  -/
  simpa only [exists_prop, nhdsWithin_univ] using hf x
  /-
    🎉 no goals
  -/


theorem dimH_preimage_le (hf : AntilipschitzWith K f) (s : Set Y) : dimH (f ⁻¹' s) ≤ dimH s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    ⊢ LE.le (dimH (Set.preimage f s)) (dimH s)
  -/
  borelize X Y
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    ⊢ LE.le (dimH (Set.preimage f s)) (dimH s)
  -/
  refine dimH_le fun d hd => le_dimH_of_hausdorffMeasure_eq_top ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.preimage f s)) Top.top
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  have := hf.hausdorffMeasure_preimage_le d.coe_nonneg s
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.preimage f s)) Top.top
    this : LE.le ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.preimage f s))  …
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  rw [hd, top_le_iff] at this
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.preimage f s)) Top.top
    this : Eq (HMul.hMul (HPow.hPow ↑K ↑d) ((MeasureTheory.Measure.hausdorffMeasur …
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
  -/
  contrapose! this
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    K : NNReal
    f : X → Y
    hf : AntilipschitzWith K f
    s : Set Y
    this✝³ : MeasurableSpace X := borel X
    this✝² : BorelSpace X
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    d : NNReal
    hd : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) (Set.preimage f s)) Top.top
    this : Ne ((MeasureTheory.Measure.hausdorffMeasure ↑d) s) Top.top
    ⊢ Ne (HMul.hMul (HPow.hPow ↑K ↑d) ((MeasureTheory.Measure.hausdorffMeasure ↑d) …
  -/
  exact ENNReal.mul_ne_top (by simp) this
  /-
    🎉 no goals
  -/


theorem le_dimH_image (hf : AntilipschitzWith K f) (s : Set X) : dimH s ≤ dimH (f '' s) :=
  calc
    dimH s ≤ dimH (f ⁻¹' (f '' s)) := dimH_mono (subset_preimage_image _ _)
    _ ≤ dimH (f '' s) := hf.dimH_preimage_le _


theorem Isometry.dimH_image (hf : Isometry f) (s : Set X) : dimH (f '' s) = dimH s :=
  le_antisymm (hf.lipschitz.dimH_image_le _) (hf.antilipschitz.le_dimH_image _)


@[simp]
theorem dimH_image (e : X ≃ᵢ Y) (s : Set X) : dimH (e '' s) = dimH s :=
  e.isometry.dimH_image s


@[simp]
theorem dimH_preimage (e : X ≃ᵢ Y) (s : Set Y) : dimH (e ⁻¹' s) = dimH s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    e : IsometryEquiv X Y
    s : Set Y
    ⊢ Eq (dimH (Set.preimage (⇑e) s)) (dimH s)
  -/
  rw [← e.image_symm, e.symm.dimH_image]
  /-
    🎉 no goals
  -/


theorem dimH_univ (e : X ≃ᵢ Y) : dimH (univ : Set X) = dimH (univ : Set Y) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    e : IsometryEquiv X Y
    ⊢ Eq (dimH Set.univ) (dimH Set.univ)
  -/
  rw [← e.dimH_preimage univ, preimage_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem dimH_image (e : E ≃L[𝕜] F) (s : Set E) : dimH (e '' s) = dimH s :=
  le_antisymm (e.lipschitz.dimH_image_le s) <| by
    /-
      𝕜 : Type u_4
      E : Type u_5
      F : Type u_6
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      s : Set E
      ⊢ LE.le (dimH s) (dimH (Set.image (⇑e) s))
    -/
    simpa only [e.symm_image_image] using e.symm.lipschitz.dimH_image_le (e '' s)
    /-
      🎉 no goals
    -/


@[simp]
theorem dimH_preimage (e : E ≃L[𝕜] F) (s : Set F) : dimH (e ⁻¹' s) = dimH s := by
  /-
    𝕜 : Type u_4
    E : Type u_5
    F : Type u_6
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set F
    ⊢ Eq (dimH (Set.preimage (⇑e) s)) (dimH s)
  -/
  rw [← e.image_symm_eq_preimage, e.symm.dimH_image]
  /-
    🎉 no goals
  -/


theorem dimH_univ (e : E ≃L[𝕜] F) : dimH (univ : Set E) = dimH (univ : Set F) := by
  /-
    𝕜 : Type u_4
    E : Type u_5
    F : Type u_6
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    ⊢ Eq (dimH Set.univ) (dimH Set.univ)
  -/
  rw [← e.dimH_preimage, preimage_univ]
  /-
    🎉 no goals
  -/


theorem dimH_ball_pi (x : ι → ℝ) {r : ℝ} (hr : 0 < r) :
    dimH (Metric.ball x r) = Fintype.card ι := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    x : ι → Real
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (dimH (Metric.ball x r)) ↑(Fintype.card ι)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Type u_1
      inst✝ : Fintype ι
      x : ι → Real
      r : Real
      hr : LT.lt 0 r
      h✝ : IsEmpty ι
      ⊢ Eq (dimH (Metric.ball x r)) ↑(Fintype.card ι)
    -/
  · rwa [dimH_subsingleton, eq_comm, Nat.cast_eq_zero, Fintype.card_eq_zero_iff]
    /-
      case inl
      ι : Type u_1
      inst✝ : Fintype ι
      x : ι → Real
      r : Real
      hr : LT.lt 0 r
      h✝ : IsEmpty ι
      ⊢ (Metric.ball x r).Subsingleton
    -/
    exact fun x _ y _ => Subsingleton.elim x y
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝ : Fintype ι
      x : ι → Real
      r : Real
      hr : LT.lt 0 r
      h✝ : Nonempty ι
      ⊢ Eq (dimH (Metric.ball x r)) ↑(Fintype.card ι)
    -/
  · rw [← ENNReal.coe_natCast]
    have : μH[Fintype.card ι] (Metric.ball x r) = ENNReal.ofReal ((2 * r) ^ Fintype.card ι) := by
      rw [hausdorffMeasure_pi_real, Real.volume_pi_ball _ hr]
    /-
      case inr
      ι : Type u_1
      inst✝ : Fintype ι
      x : ι → Real
      r : Real
      hr : LT.lt 0 r
      h✝ : Nonempty ι
      this : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑(Fintype.card ι)) (Metric. …
      ⊢ Eq (dimH (Metric.ball x r)) ↑↑(Fintype.card ι)
    -/
    refine dimH_of_hausdorffMeasure_ne_zero_ne_top ?_ ?_ <;> rw [NNReal.coe_natCast, this]
      /-
        case inr.refine_1
        ι : Type u_1
        inst✝ : Fintype ι
        x : ι → Real
        r : Real
        hr : LT.lt 0 r
        h✝ : Nonempty ι
        this : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑(Fintype.card ι)) (Metric. …
        ⊢ Ne (ENNReal.ofReal (HPow.hPow (HMul.hMul 2 r) (Fintype.card ι))) 0
      -/
    · simp [pow_pos (mul_pos (zero_lt_two' ℝ) hr)]
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        ι : Type u_1
        inst✝ : Fintype ι
        x : ι → Real
        r : Real
        hr : LT.lt 0 r
        h✝ : Nonempty ι
        this : Eq ((MeasureTheory.Measure.hausdorffMeasure ↑(Fintype.card ι)) (Metric. …
        ⊢ Ne (ENNReal.ofReal (HPow.hPow (HMul.hMul 2 r) (Fintype.card ι))) Top.top
      -/
    · exact ENNReal.ofReal_ne_top
      /-
        🎉 no goals
      -/


theorem dimH_ball_pi_fin {n : ℕ} (x : Fin n → ℝ) {r : ℝ} (hr : 0 < r) :
                                     /-
                                       n : Nat
                                       x : Fin n → Real
                                       r : Real
                                       hr : LT.lt 0 r
                                       ⊢ Eq (dimH (Metric.ball x r)) ↑n
                                     -/
    dimH (Metric.ball x r) = n := by rw [dimH_ball_pi x hr, Fintype.card_fin]
                                     /-
                                       🎉 no goals
                                     -/


theorem dimH_univ_pi (ι : Type*) [Fintype ι] : dimH (univ : Set (ι → ℝ)) = Fintype.card ι := by
  simp only [← Metric.iUnion_ball_nat_succ (0 : ι → ℝ), dimH_iUnion,
    dimH_ball_pi _ (Nat.cast_add_one_pos _), iSup_const]


theorem dimH_univ_pi_fin (n : ℕ) : dimH (univ : Set (Fin n → ℝ)) = n := by
  /-
    n : Nat
    ⊢ Eq (dimH Set.univ) ↑n
  -/
  rw [dimH_univ_pi, Fintype.card_fin]
  /-
    🎉 no goals
  -/


theorem dimH_of_mem_nhds {x : E} {s : Set E} (h : s ∈ 𝓝 x) : dimH s = finrank ℝ E := by
  have e : E ≃L[ℝ] Fin (finrank ℝ E) → ℝ :=
    ContinuousLinearEquiv.ofFinrankEq (Module.finrank_fin_fun ℝ).symm
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
    ⊢ Eq (dimH s) ↑(Module.finrank Real E)
  -/
  rw [← e.dimH_image]
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
    ⊢ Eq (dimH (Set.image (⇑e) s)) ↑(Module.finrank Real E)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      x : E
      s : Set E
      h : Membership.mem (nhds x) s
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LE.le (dimH (Set.image (⇑e) s)) ↑(Module.finrank Real E)
    -/
  · exact (dimH_mono (subset_univ _)).trans_eq (dimH_univ_pi_fin _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      x : E
      s : Set E
      h : Membership.mem (nhds x) s
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LE.le (↑(Module.finrank Real E)) (dimH (Set.image (⇑e) s))
    -/
  · have : e '' s ∈ 𝓝 (e x) := by rw [← e.map_nhds_eq]; exact image_mem_map h
    /-
      case refine_2
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      x : E
      s : Set E
      h : Membership.mem (nhds x) s
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      this : Membership.mem (nhds (e x)) (Set.image (⇑e) s)
      ⊢ LE.le (↑(Module.finrank Real E)) (dimH (Set.image (⇑e) s))
    -/
    rcases Metric.nhds_basis_ball.mem_iff.1 this with ⟨r, hr0, hr⟩
    /-
      case refine_2.intro.intro
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : FiniteDimensional Real E
      x : E
      s : Set E
      h : Membership.mem (nhds x) s
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      this : Membership.mem (nhds (e x)) (Set.image (⇑e) s)
      r : Real
      hr0 : LT.lt 0 r
      hr : HasSubset.Subset (Metric.ball (e x) r) (Set.image (⇑e) s)
      ⊢ LE.le (↑(Module.finrank Real E)) (dimH (Set.image (⇑e) s))
    -/
    simpa only [dimH_ball_pi_fin (e x) hr0] using dimH_mono hr
    /-
      🎉 no goals
    -/


theorem dimH_of_nonempty_interior {s : Set E} (h : (interior s).Nonempty) : dimH s = finrank ℝ E :=
  let ⟨_, hx⟩ := h
  dimH_of_mem_nhds (mem_interior_iff_mem_nhds.1 hx)


theorem dimH_univ_eq_finrank : dimH (univ : Set E) = finrank ℝ E :=
  dimH_of_mem_nhds (@univ_mem _ (𝓝 0))


theorem dimH_univ : dimH (univ : Set ℝ) = 1 := by
  /-
    ⊢ Eq (dimH Set.univ) 1
  -/
  rw [dimH_univ_eq_finrank ℝ, Module.finrank_self, Nat.cast_one]
  /-
    🎉 no goals
  -/


lemma hausdorffMeasure_of_finrank_lt [MeasurableSpace E] [BorelSpace E] {d : ℝ}
    (hd : finrank ℝ E < d) : (μH[d] : Measure E) = 0 := by
  /-
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : Real
    hd : LT.lt (↑(Module.finrank Real E)) d
    ⊢ Eq (MeasureTheory.Measure.hausdorffMeasure d) 0
  -/
  lift d to ℝ≥0 using (Nat.cast_nonneg _).trans hd.le
  /-
    case intro
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : NNReal
    hd : LT.lt ↑(Module.finrank Real E) ↑d
    ⊢ Eq (MeasureTheory.Measure.hausdorffMeasure ↑d) 0
  -/
  rw [← measure_univ_eq_zero]
  /-
    case intro
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : NNReal
    hd : LT.lt ↑(Module.finrank Real E) ↑d
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure ↑d) Set.univ) 0
  -/
  apply hausdorffMeasure_of_dimH_lt
  /-
    case intro.h
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : NNReal
    hd : LT.lt ↑(Module.finrank Real E) ↑d
    ⊢ LT.lt (dimH Set.univ) ↑d
  -/
  rw [dimH_univ_eq_finrank]
  /-
    case intro.h
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : NNReal
    hd : LT.lt ↑(Module.finrank Real E) ↑d
    ⊢ LT.lt ↑(Module.finrank Real E) ↑d
  -/
  exact mod_cast hd
  /-
    🎉 no goals
  -/


theorem dense_compl_of_dimH_lt_finrank {s : Set E} (hs : dimH s < finrank ℝ E) : Dense sᶜ := by
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : LT.lt (dimH s) ↑(Module.finrank Real E)
    ⊢ Dense (HasCompl.compl s)
  -/
  refine fun x => mem_closure_iff_nhds.2 fun t ht => nonempty_iff_ne_empty.2 fun he => hs.not_le ?_
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : LT.lt (dimH s) ↑(Module.finrank Real E)
    x : E
    t : Set E
    ht : Membership.mem (nhds x) t
    he : Eq (Inter.inter t (HasCompl.compl s)) EmptyCollection.emptyCollection
    ⊢ LE.le (↑(Module.finrank Real E)) (dimH s)
  -/
  rw [← diff_eq, diff_eq_empty] at he
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : LT.lt (dimH s) ↑(Module.finrank Real E)
    x : E
    t : Set E
    ht : Membership.mem (nhds x) t
    he : HasSubset.Subset t s
    ⊢ LE.le (↑(Module.finrank Real E)) (dimH s)
  -/
  rw [← Real.dimH_of_mem_nhds ht]
  /-
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    s : Set E
    hs : LT.lt (dimH s) ↑(Module.finrank Real E)
    x : E
    t : Set E
    ht : Membership.mem (nhds x) t
    he : HasSubset.Subset t s
    ⊢ LE.le (dimH t) (dimH s)
  -/
  exact dimH_mono he
  /-
    🎉 no goals
  -/


/-- Let `f` be a function defined on a finite dimensional real normed space. If `f` is `C¹`-smooth
on a convex set `s`, then the Hausdorff dimension of `f '' s` is less than or equal to the Hausdorff
dimension of `s`.

TODO: do we actually need `Convex ℝ s`? -/
theorem ContDiffOn.dimH_image_le {f : E → F} {s t : Set E} (hf : ContDiffOn ℝ 1 f s)
    (hc : Convex ℝ s) (ht : t ⊆ s) : dimH (f '' t) ≤ dimH t :=
  dimH_image_le_of_locally_lipschitzOn fun x hx =>
    let ⟨C, u, hu, hf⟩ := (hf x (ht hx)).exists_lipschitzOnWith hc
    ⟨C, u, nhdsWithin_mono _ ht hu, hf⟩


/-- The Hausdorff dimension of the range of a `C¹`-smooth function defined on a finite dimensional
real normed space is at most the dimension of its domain as a vector space over `ℝ`. -/
theorem ContDiff.dimH_range_le {f : E → F} (h : ContDiff ℝ 1 f) : dimH (range f) ≤ finrank ℝ E :=
  calc
                                            /-
                                              E : Type u_4
                                              F : Type u_5
                                              inst✝⁴ : NormedAddCommGroup E
                                              inst✝³ : NormedSpace Real E
                                              inst✝² : FiniteDimensional Real E
                                              inst✝¹ : NormedAddCommGroup F
                                              inst✝ : NormedSpace Real F
                                              f : E → F
                                              h : ContDiff Real 1 f
                                              ⊢ Eq (dimH (Set.range f)) (dimH (Set.image f Set.univ))
                                            -/
    dimH (range f) = dimH (f '' univ) := by rw [image_univ]
                                            /-
                                              🎉 no goals
                                            -/
    _ ≤ dimH (univ : Set E) := h.contDiffOn.dimH_image_le convex_univ Subset.rfl
    _ = finrank ℝ E := Real.dimH_univ_eq_finrank E


/-- A particular case of Sard's Theorem. Let `f : E → F` be a map between finite dimensional real
vector spaces. Suppose that `f` is `C¹` smooth on a convex set `s` of Hausdorff dimension strictly
less than the dimension of `F`. Then the complement of the image `f '' s` is dense in `F`. -/
theorem ContDiffOn.dense_compl_image_of_dimH_lt_finrank [FiniteDimensional ℝ F] {f : E → F}
    {s t : Set E} (h : ContDiffOn ℝ 1 f s) (hc : Convex ℝ s) (ht : t ⊆ s)
    (htF : dimH t < finrank ℝ F) : Dense (f '' t)ᶜ :=
  dense_compl_of_dimH_lt_finrank <| (h.dimH_image_le hc ht).trans_lt htF


/-- A particular case of Sard's Theorem. If `f` is a `C¹` smooth map from a real vector space to a
real vector space `F` of strictly larger dimension, then the complement of the range of `f` is dense
in `F`. -/
theorem ContDiff.dense_compl_range_of_finrank_lt_finrank [FiniteDimensional ℝ F] {f : E → F}
    (h : ContDiff ℝ 1 f) (hEF : finrank ℝ E < finrank ℝ F) : Dense (range f)ᶜ :=
  dense_compl_of_dimH_lt_finrank <| h.dimH_range_le.trans_lt <| Nat.cast_lt.2 hEF

