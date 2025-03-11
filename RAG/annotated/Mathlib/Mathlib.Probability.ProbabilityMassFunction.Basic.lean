/-- A probability mass function, or discrete probability measures is a function `α → ℝ≥0∞` such
  that the values have (infinite) sum `1`. -/
def PMF.{u} (α : Type u) : Type u :=
  { f : α → ℝ≥0∞ // HasSum f 1 }


instance instFunLike : FunLike (PMF α) α ℝ≥0∞ where
  coe p a := p.1 a
  coe_injective' _ _ h := Subtype.eq h


@[ext]
protected theorem ext {p q : PMF α} (h : ∀ x, p x = q x) : p = q :=
  DFunLike.ext p q h


theorem hasSum_coe_one (p : PMF α) : HasSum p 1 :=
  p.2


@[simp]
theorem tsum_coe (p : PMF α) : ∑' a, p a = 1 :=
  p.hasSum_coe_one.tsum_eq


theorem tsum_coe_ne_top (p : PMF α) : ∑' a, p a ≠ ∞ :=
  p.tsum_coe.symm ▸ ENNReal.one_ne_top


theorem tsum_coe_indicator_ne_top (p : PMF α) (s : Set α) : ∑' a, s.indicator p a ≠ ∞ :=
  ne_of_lt (lt_of_le_of_lt
    (tsum_le_tsum (fun _ => Set.indicator_apply_le fun _ => le_rfl) ENNReal.summable
      ENNReal.summable)
    (lt_of_le_of_ne le_top p.tsum_coe_ne_top))


@[simp]
theorem coe_ne_zero (p : PMF α) : ⇑p ≠ 0 := fun hp =>
  zero_ne_one ((tsum_zero.symm.trans (tsum_congr fun x => symm (congr_fun hp x))).trans p.tsum_coe)


/-- The support of a `PMF` is the set where it is nonzero. -/
def support (p : PMF α) : Set α :=
  Function.support p


@[simp]
theorem mem_support_iff (p : PMF α) (a : α) : a ∈ p.support ↔ p a ≠ 0 := Iff.rfl


@[simp]
theorem support_nonempty (p : PMF α) : p.support.Nonempty :=
  Function.support_nonempty_iff.2 p.coe_ne_zero


@[simp]
theorem support_countable (p : PMF α) : p.support.Countable :=
  Summable.countable_support_ennreal (tsum_coe_ne_top p)


theorem apply_eq_zero_iff (p : PMF α) (a : α) : p a = 0 ↔ a ∉ p.support := by
  /-
    α : Type u_1
    p : PMF α
    a : α
    ⊢ Iff (Eq (p a) 0) (Not (Membership.mem p.support a))
  -/
  rw [mem_support_iff, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem apply_pos_iff (p : PMF α) (a : α) : 0 < p a ↔ a ∈ p.support :=
  pos_iff_ne_zero.trans (p.mem_support_iff a).symm


theorem apply_eq_one_iff (p : PMF α) (a : α) : p a = 1 ↔ p.support = {a} := by
  refine ⟨fun h => Set.Subset.antisymm (fun a' ha' => by_contra fun ha => ?_)
    fun a' ha' => ha'.symm ▸ (p.mem_support_iff a).2 fun ha => zero_ne_one <| ha.symm.trans h,
    fun h => _root_.trans (symm <| tsum_eq_single a
      fun a' ha' => (p.apply_eq_zero_iff a').2 (h.symm ▸ ha')) p.tsum_coe⟩
  /-
    α : Type u_1
    p : PMF α
    a : α
    h : Eq (p a) 1
    a' : α
    ha' : Membership.mem p.support a'
    ha : Not (Membership.mem (Singleton.singleton a) a')
    ⊢ False
  -/
  suffices 1 < ∑' a, p a from ne_of_lt this p.tsum_coe.symm
  have : 0 < ∑' b, ite (b = a) 0 (p b) := lt_of_le_of_ne' zero_le'
    ((tsum_ne_zero_iff ENNReal.summable).2
      ⟨a', ite_ne_left_iff.2 ⟨ha, Ne.symm <| (p.mem_support_iff a').2 ha'⟩⟩)
  calc
    1 = 1 + 0 := (add_zero 1).symm
    _ < p a + ∑' b, ite (b = a) 0 (p b) :=
      (ENNReal.add_lt_add_of_le_of_lt ENNReal.one_ne_top (le_of_eq h.symm) this)
    _ = ite (a = a) (p a) 0 + ∑' b, ite (b = a) 0 (p b) := by rw [eq_self_iff_true, if_true]
    _ = (∑' b, ite (b = a) (p b) 0) + ∑' b, ite (b = a) 0 (p b) := by
      congr
      exact symm (tsum_eq_single a fun b hb => if_neg hb)
    _ = ∑' b, (ite (b = a) (p b) 0 + ite (b = a) 0 (p b)) := ENNReal.tsum_add.symm
    _ = ∑' b, p b := tsum_congr fun b => by split_ifs <;> simp only [zero_add, add_zero, le_rfl]


theorem coe_le_one (p : PMF α) (a : α) : p a ≤ 1 := by
  /-
    α : Type u_1
    p : PMF α
    a : α
    ⊢ LE.le (p a) 1
  -/
  refine hasSum_le (fun b => ?_) (hasSum_ite_eq a (p a)) (hasSum_coe_one p)
  /-
    α : Type u_1
    p : PMF α
    a b : α
    ⊢ LE.le (ite (Eq b a) (p a) 0) (p b)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp only [h, zero_le', le_rfl]
                       /-
                         🎉 no goals
                       -/


theorem apply_ne_top (p : PMF α) (a : α) : p a ≠ ∞ :=
  ne_of_lt (lt_of_le_of_lt (p.coe_le_one a) ENNReal.one_lt_top)


theorem apply_lt_top (p : PMF α) (a : α) : p a < ∞ :=
  lt_of_le_of_ne le_top (p.apply_ne_top a)


/-- Construct an `OuterMeasure` from a `PMF`, by assigning measure to each set `s : Set α` equal
  to the sum of `p x` for each `x ∈ α`. -/
def toOuterMeasure (p : PMF α) : OuterMeasure α :=
  OuterMeasure.sum fun x : α => p x • dirac x


theorem toOuterMeasure_apply : p.toOuterMeasure s = ∑' x, s.indicator p x :=
  tsum_congr fun x => smul_dirac_apply (p x) x s


@[simp]
theorem toOuterMeasure_caratheodory : p.toOuterMeasure.caratheodory = ⊤ := by
  /-
    α : Type u_1
    p : PMF α
    ⊢ Eq p.toOuterMeasure.caratheodory Top.top
  -/
  refine eq_top_iff.2 <| le_trans (le_sInf fun x hx => ?_) (le_sum_caratheodory _)
  /-
    α : Type u_1
    p : PMF α
    x : MeasurableSpace α
    hx : Membership.mem (Set.range fun i => (HSMul.hSMul (p i) (MeasureTheory.Oute …
    ⊢ LE.le Top.top x
  -/
  have ⟨y, hy⟩ := hx
  exact
    ((le_of_eq (dirac_caratheodory y).symm).trans (le_smul_caratheodory _ _)).trans (le_of_eq hy)


@[simp]
theorem toOuterMeasure_apply_finset (s : Finset α) : p.toOuterMeasure s = ∑ x ∈ s, p x := by
  /-
    α : Type u_1
    p : PMF α
    s : Finset α
    ⊢ Eq (p.toOuterMeasure ↑s) (s.sum fun x => p x)
  -/
  refine (toOuterMeasure_apply p s).trans ((tsum_eq_sum (s := s) ?_).trans ?_)
    /-
      case refine_1
      α : Type u_1
      p : PMF α
      s : Finset α
      ⊢ ∀ (b : α), Not (Membership.mem s b) → Eq ((↑s).indicator (⇑p) b) 0
    -/
  · exact fun x hx => Set.indicator_of_not_mem (Finset.mem_coe.not.2 hx) _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      p : PMF α
      s : Finset α
      ⊢ Eq (s.sum fun b => (↑s).indicator (⇑p) b) (s.sum fun x => p x)
    -/
  · exact Finset.sum_congr rfl fun x hx => Set.indicator_of_mem (Finset.mem_coe.2 hx) _
    /-
      🎉 no goals
    -/


theorem toOuterMeasure_apply_singleton (a : α) : p.toOuterMeasure {a} = p a := by
  /-
    α : Type u_1
    p : PMF α
    a : α
    ⊢ Eq (p.toOuterMeasure (Singleton.singleton a)) (p a)
  -/
  refine (p.toOuterMeasure_apply {a}).trans ((tsum_eq_single a fun b hb => ?_).trans ?_)
    /-
      case refine_1
      α : Type u_1
      p : PMF α
      a b : α
      hb : Ne b a
      ⊢ Eq ((Singleton.singleton a).indicator (⇑p) b) 0
    -/
  · exact ite_eq_right_iff.2 fun hb' => False.elim <| hb hb'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      p : PMF α
      a : α
      ⊢ Eq ((Singleton.singleton a).indicator (⇑p) a) (p a)
    -/
  · exact ite_eq_left_iff.2 fun ha' => False.elim <| ha' rfl
    /-
      🎉 no goals
    -/


theorem toOuterMeasure_injective : (toOuterMeasure : PMF α → OuterMeasure α).Injective :=
  fun p q h => PMF.ext fun x => (p.toOuterMeasure_apply_singleton x).symm.trans
    ((congr_fun (congr_arg _ h) _).trans <| q.toOuterMeasure_apply_singleton x)


@[simp]
theorem toOuterMeasure_inj {p q : PMF α} : p.toOuterMeasure = q.toOuterMeasure ↔ p = q :=
  toOuterMeasure_injective.eq_iff


theorem toOuterMeasure_apply_eq_zero_iff : p.toOuterMeasure s = 0 ↔ Disjoint p.support s := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    ⊢ Iff (Eq (p.toOuterMeasure s) 0) (Disjoint p.support s)
  -/
  rw [toOuterMeasure_apply, ENNReal.tsum_eq_zero]
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    ⊢ Iff (∀ (i : α), Eq (s.indicator (⇑p) i) 0) (Disjoint p.support s)
  -/
  exact funext_iff.symm.trans Set.indicator_eq_zero'
  /-
    🎉 no goals
  -/


theorem toOuterMeasure_apply_eq_one_iff : p.toOuterMeasure s = 1 ↔ p.support ⊆ s := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    ⊢ Iff (Eq (p.toOuterMeasure s) 1) (HasSubset.Subset p.support s)
  -/
  refine (p.toOuterMeasure_apply s).symm ▸ ⟨fun h a hap => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      p : PMF α
      s : Set α
      h : Eq (tsum fun x => s.indicator (⇑p) x) 1
      a : α
      hap : Membership.mem p.support a
      ⊢ Membership.mem s a
    -/
  · refine by_contra fun hs => ne_of_lt ?_ (h.trans p.tsum_coe.symm)
    /-
      case refine_1
      α : Type u_1
      p : PMF α
      s : Set α
      h : Eq (tsum fun x => s.indicator (⇑p) x) 1
      a : α
      hap : Membership.mem p.support a
      hs : Not (Membership.mem s a)
      ⊢ LT.lt (tsum fun x => s.indicator (⇑p) x) (tsum fun a => p a)
    -/
    have hs' : s.indicator p a = 0 := Set.indicator_apply_eq_zero.2 fun hs' => False.elim <| hs hs'
    /-
      case refine_1
      α : Type u_1
      p : PMF α
      s : Set α
      h : Eq (tsum fun x => s.indicator (⇑p) x) 1
      a : α
      hap : Membership.mem p.support a
      hs : Not (Membership.mem s a)
      hs' : Eq (s.indicator (⇑p) a) 0
      ⊢ LT.lt (tsum fun x => s.indicator (⇑p) x) (tsum fun a => p a)
    -/
    have hsa : s.indicator p a < p a := hs'.symm ▸ (p.apply_pos_iff a).2 hap
    exact ENNReal.tsum_lt_tsum (p.tsum_coe_indicator_ne_top s)
      (fun x => Set.indicator_apply_le fun _ => le_rfl) hsa
  · suffices ∀ (x) (_ : x ∉ s), p x = 0 from
      _root_.trans (tsum_congr
        fun a => (Set.indicator_apply s p a).trans
          (ite_eq_left_iff.2 <| symm ∘ this a)) p.tsum_coe
    /-
      case refine_2
      α : Type u_1
      p : PMF α
      s : Set α
      h : HasSubset.Subset p.support s
      ⊢ ∀ (x : α), Not (Membership.mem s x) → Eq (p x) 0
    -/
    exact fun a ha => (p.apply_eq_zero_iff a).2 <| Set.not_mem_subset h ha
    /-
      🎉 no goals
    -/


@[simp]
theorem toOuterMeasure_apply_inter_support :
    p.toOuterMeasure (s ∩ p.support) = p.toOuterMeasure s := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    ⊢ Eq (p.toOuterMeasure (Inter.inter s p.support)) (p.toOuterMeasure s)
  -/
  simp only [toOuterMeasure_apply, PMF.support, Set.indicator_inter_support]
  /-
    🎉 no goals
  -/


/-- Slightly stronger than `OuterMeasure.mono` having an intersection with `p.support`. -/
theorem toOuterMeasure_mono {s t : Set α} (h : s ∩ p.support ⊆ t) :
    p.toOuterMeasure s ≤ p.toOuterMeasure t :=
  le_trans (le_of_eq (toOuterMeasure_apply_inter_support p s).symm) (p.toOuterMeasure.mono h)


theorem toOuterMeasure_apply_eq_of_inter_support_eq {s t : Set α}
    (h : s ∩ p.support = t ∩ p.support) : p.toOuterMeasure s = p.toOuterMeasure t :=
  le_antisymm (p.toOuterMeasure_mono (h.symm ▸ Set.inter_subset_left))
    (p.toOuterMeasure_mono (h ▸ Set.inter_subset_left))


@[simp]
theorem toOuterMeasure_apply_fintype [Fintype α] : p.toOuterMeasure s = ∑ x, s.indicator p x :=
  (p.toOuterMeasure_apply s).trans (tsum_eq_sum fun x h => absurd (Finset.mem_univ x) h)


/-- Since every set is Carathéodory-measurable under `PMF.toOuterMeasure`,
  we can further extend this `OuterMeasure` to a `Measure` on `α`. -/
def toMeasure [MeasurableSpace α] (p : PMF α) : Measure α :=
  p.toOuterMeasure.toMeasure ((toOuterMeasure_caratheodory p).symm ▸ le_top)


theorem toOuterMeasure_apply_le_toMeasure_apply : p.toOuterMeasure s ≤ p.toMeasure s :=
  le_toMeasure_apply p.toOuterMeasure _ s


theorem toMeasure_apply_eq_toOuterMeasure_apply (hs : MeasurableSet s) :
    p.toMeasure s = p.toOuterMeasure s :=
  toMeasure_apply p.toOuterMeasure _ hs


theorem toMeasure_apply (hs : MeasurableSet s) : p.toMeasure s = ∑' x, s.indicator p x :=
  (p.toMeasure_apply_eq_toOuterMeasure_apply s hs).trans (p.toOuterMeasure_apply s)


theorem toMeasure_apply_singleton (a : α) (h : MeasurableSet ({a} : Set α)) :
    p.toMeasure {a} = p a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : PMF α
    a : α
    h : MeasurableSet (Singleton.singleton a)
    ⊢ Eq (p.toMeasure (Singleton.singleton a)) (p a)
  -/
  simp [toMeasure_apply_eq_toOuterMeasure_apply _ _ h, toOuterMeasure_apply_singleton]
  /-
    🎉 no goals
  -/


theorem toMeasure_apply_eq_zero_iff (hs : MeasurableSet s) :
    p.toMeasure s = 0 ↔ Disjoint p.support s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : PMF α
    s : Set α
    hs : MeasurableSet s
    ⊢ Iff (Eq (p.toMeasure s) 0) (Disjoint p.support s)
  -/
  rw [toMeasure_apply_eq_toOuterMeasure_apply p s hs, toOuterMeasure_apply_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem toMeasure_apply_eq_one_iff (hs : MeasurableSet s) : p.toMeasure s = 1 ↔ p.support ⊆ s :=
  (p.toMeasure_apply_eq_toOuterMeasure_apply s hs).symm ▸ p.toOuterMeasure_apply_eq_one_iff s


@[simp]
theorem toMeasure_apply_inter_support (hs : MeasurableSet s) (hp : MeasurableSet p.support) :
    p.toMeasure (s ∩ p.support) = p.toMeasure s := by
  simp [p.toMeasure_apply_eq_toOuterMeasure_apply s hs,
    p.toMeasure_apply_eq_toOuterMeasure_apply _ (hs.inter hp)]


@[simp]
theorem restrict_toMeasure_support [MeasurableSingletonClass α] (p : PMF α) :
    Measure.restrict (toMeasure p) (support p) = toMeasure p := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    p : PMF α
    ⊢ Eq (p.toMeasure.restrict p.support) p.toMeasure
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    p : PMF α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((p.toMeasure.restrict p.support) s) (p.toMeasure s)
  -/
  apply (MeasureTheory.Measure.restrict_apply hs).trans
  /-
    case h
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    p : PMF α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (p.toMeasure (Inter.inter s p.support)) (p.toMeasure s)
  -/
  apply toMeasure_apply_inter_support p s hs p.support_countable.measurableSet
  /-
    🎉 no goals
  -/


theorem toMeasure_mono {s t : Set α} (hs : MeasurableSet s) (ht : MeasurableSet t)
    (h : s ∩ p.support ⊆ t) : p.toMeasure s ≤ p.toMeasure t := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : PMF α
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset (Inter.inter s p.support) t
    ⊢ LE.le (p.toMeasure s) (p.toMeasure t)
  -/
  simpa only [p.toMeasure_apply_eq_toOuterMeasure_apply, hs, ht] using toOuterMeasure_mono p h
  /-
    🎉 no goals
  -/


theorem toMeasure_apply_eq_of_inter_support_eq {s t : Set α} (hs : MeasurableSet s)
    (ht : MeasurableSet t) (h : s ∩ p.support = t ∩ p.support) : p.toMeasure s = p.toMeasure t := by
  simpa only [p.toMeasure_apply_eq_toOuterMeasure_apply, hs, ht] using
    toOuterMeasure_apply_eq_of_inter_support_eq p h


theorem toMeasure_injective : (toMeasure : PMF α → Measure α).Injective := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    ⊢ Function.Injective PMF.toMeasure
  -/
  intro p q h
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    p q : PMF α
    h : Eq p.toMeasure q.toMeasure
    ⊢ Eq p q
  -/
  ext x
  rw [← p.toMeasure_apply_singleton x <| measurableSet_singleton x,
    ← q.toMeasure_apply_singleton x <| measurableSet_singleton x, h]


@[simp]
theorem toMeasure_inj {p q : PMF α} : p.toMeasure = q.toMeasure ↔ p = q :=
  toMeasure_injective.eq_iff


@[simp]
theorem toMeasure_apply_finset (s : Finset α) : p.toMeasure s = ∑ x ∈ s, p x :=
  (p.toMeasure_apply_eq_toOuterMeasure_apply s s.measurableSet).trans
    (p.toOuterMeasure_apply_finset s)


theorem toMeasure_apply_of_finite (hs : s.Finite) : p.toMeasure s = ∑' x, s.indicator p x :=
  (p.toMeasure_apply_eq_toOuterMeasure_apply s hs.measurableSet).trans (p.toOuterMeasure_apply s)


@[simp]
theorem toMeasure_apply_fintype [Fintype α] : p.toMeasure s = ∑ x, s.indicator p x :=
  (p.toMeasure_apply_eq_toOuterMeasure_apply s s.toFinite.measurableSet).trans
    (p.toOuterMeasure_apply_fintype s)


/-- Given that `α` is a countable, measurable space with all singleton sets measurable,
we can convert any probability measure into a `PMF`, where the mass of a point
is the measure of the singleton set under the original measure. -/
def toPMF [Countable α] [MeasurableSpace α] [MeasurableSingletonClass α] (μ : Measure α)
    [h : IsProbabilityMeasure μ] : PMF α :=
  ⟨fun x => μ ({x} : Set α),
    ENNReal.summable.hasSum_iff.2
      (_root_.trans
        (symm <|
          (tsum_indicator_apply_singleton μ Set.univ MeasurableSet.univ).symm.trans
            (tsum_congr fun x => congr_fun (Set.indicator_univ _) x))
        h.measure_univ)⟩


theorem toPMF_apply (x : α) : μ.toPMF x = μ {x} := rfl


@[simp]
theorem toPMF_toMeasure : μ.toPMF.toMeasure = μ :=
  Measure.ext fun s hs => by
    /-
      α : Type u_1
      inst✝³ : Countable α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSingletonClass α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (μ.toPMF.toMeasure s) (μ s)
    -/
    rw [μ.toPMF.toMeasure_apply s hs, ← μ.tsum_indicator_apply_singleton s hs]
    /-
      α : Type u_1
      inst✝³ : Countable α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSingletonClass α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (tsum fun x => s.indicator (⇑μ.toPMF) x) (tsum fun x => s.indicator (fun  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The measure associated to a `PMF` by `toMeasure` is a probability measure. -/
instance toMeasure.isProbabilityMeasure [MeasurableSpace α] (p : PMF α) :
    IsProbabilityMeasure p.toMeasure :=
  ⟨by
    simpa only [MeasurableSet.univ, toMeasure_apply_eq_toOuterMeasure_apply, Set.indicator_univ,
      toOuterMeasure_apply, ENNReal.coe_eq_one] using tsum_coe p⟩


@[simp]
theorem toMeasure_toPMF : p.toMeasure.toPMF = p :=
  PMF.ext fun x => by
    /-
      α : Type u_1
      inst✝² : Countable α
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSingletonClass α
      p : PMF α
      x : α
      ⊢ Eq (p.toMeasure.toPMF x) (p x)
    -/
    rw [← p.toMeasure_apply_singleton x (measurableSet_singleton x), p.toMeasure.toPMF_apply]
    /-
      🎉 no goals
    -/


theorem toMeasure_eq_iff_eq_toPMF (μ : Measure α) [IsProbabilityMeasure μ] :
                                        /-
                                          α : Type u_1
                                          inst✝³ : Countable α
                                          inst✝² : MeasurableSpace α
                                          inst✝¹ : MeasurableSingletonClass α
                                          p : PMF α
                                          μ : MeasureTheory.Measure α
                                          inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                          ⊢ Iff (Eq p.toMeasure μ) (Eq p μ.toPMF)
                                        -/
    p.toMeasure = μ ↔ p = μ.toPMF := by rw [← toMeasure_inj, Measure.toPMF_toMeasure]
                                        /-
                                          🎉 no goals
                                        -/


theorem toPMF_eq_iff_toMeasure_eq (μ : Measure α) [IsProbabilityMeasure μ] :
                                        /-
                                          α : Type u_1
                                          inst✝³ : Countable α
                                          inst✝² : MeasurableSpace α
                                          inst✝¹ : MeasurableSingletonClass α
                                          p : PMF α
                                          μ : MeasureTheory.Measure α
                                          inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                          ⊢ Iff (Eq μ.toPMF p) (Eq μ p.toMeasure)
                                        -/
    μ.toPMF = p ↔ μ = p.toMeasure := by rw [← toMeasure_inj, Measure.toPMF_toMeasure]
                                        /-
                                          🎉 no goals
                                        -/


