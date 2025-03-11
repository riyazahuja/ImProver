/-- A measurable function `α → StieltjesFunction` with limits 0 at -∞ and 1 at +∞ gives a measurable
function `α → Measure ℝ` by taking `StieltjesFunction.measure` at each point. -/
lemma StieltjesFunction.measurable_measure {α : Type*} {_ : MeasurableSpace α}
    {f : α → StieltjesFunction} (hf : ∀ q, Measurable fun a ↦ f a q)
    (hf_bot : ∀ a, Tendsto (f a) atBot (𝓝 0))
    (hf_top : ∀ a, Tendsto (f a) atTop (𝓝 1)) :
    Measurable fun a ↦ (f a).measure :=
  have : ∀ a, IsProbabilityMeasure (f a).measure :=
    fun a ↦ (f a).isProbabilityMeasure (hf_bot a) (hf_top a)
  .measure_of_isPiSystem_of_isProbabilityMeasure (borel_eq_generateFrom_Iic ℝ) isPiSystem_Iic <| by
    /-
      α : Type u_1
      x✝ : MeasurableSpace α
      f : α → StieltjesFunction
      hf : ∀ (q : Real), Measurable fun a => ↑(f a) q
      hf_bot : ∀ (a : α), Filter.Tendsto (↑(f a)) Filter.atBot (nhds 0)
      hf_top : ∀ (a : α), Filter.Tendsto (↑(f a)) Filter.atTop (nhds 1)
      this : ∀ (a : α), MeasureTheory.IsProbabilityMeasure (f a).measure
      ⊢ ∀ (s : Set Real), Membership.mem (Set.range Set.Iic) s → Measurable fun a => …
    -/
    simp_rw [forall_mem_range, StieltjesFunction.measure_Iic (f _) (hf_bot _), sub_zero]
    /-
      α : Type u_1
      x✝ : MeasurableSpace α
      f : α → StieltjesFunction
      hf : ∀ (q : Real), Measurable fun a => ↑(f a) q
      hf_bot : ∀ (a : α), Filter.Tendsto (↑(f a)) Filter.atBot (nhds 0)
      hf_top : ∀ (a : α), Filter.Tendsto (↑(f a)) Filter.atTop (nhds 1)
      this : ∀ (a : α), MeasureTheory.IsProbabilityMeasure (f a).measure
      ⊢ ∀ (i : Real), Measurable fun a => ENNReal.ofReal (↑(f a) i)
    -/
    exact fun _ ↦ (hf _).ennreal_ofReal
    /-
      🎉 no goals
    -/


/-- `a : α` is a Stieltjes point for `f : α → ℚ → ℝ` if `f a` is monotone with limit 0 at -∞
and 1 at +∞ and satisfies a continuity property. -/
structure IsRatStieltjesPoint (f : α → ℚ → ℝ) (a : α) : Prop where
  mono : Monotone (f a)
  tendsto_atTop_one : Tendsto (f a) atTop (𝓝 1)
  tendsto_atBot_zero : Tendsto (f a) atBot (𝓝 0)
  iInf_rat_gt_eq : ∀ t : ℚ, ⨅ r : Ioi t, f a r = f a t


lemma isRatStieltjesPoint_unit_prod_iff (f : α → ℚ → ℝ) (a : α) :
    IsRatStieltjesPoint (fun p : Unit × α ↦ f p.2) ((), a)
      ↔ IsRatStieltjesPoint f a := by
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    ⊢ Iff (ProbabilityTheory.IsRatStieltjesPoint (fun p => f p.2) { fst := Unit.un …
  -/
  constructor <;>
    /-
      case mp
      α : Type u_1
      f : α → Rat → Real
      a : α
      ⊢ ProbabilityTheory.IsRatStieltjesPoint (fun p => f p.2) { fst := Unit.unit, s …
    -/
    /-
      🎉 no goals
    -/
    exact fun h ↦ ⟨h.mono, h.tendsto_atTop_one, h.tendsto_atBot_zero, h.iInf_rat_gt_eq⟩
    /-
      🎉 no goals
    -/


lemma measurableSet_isRatStieltjesPoint [MeasurableSpace α] (hf : Measurable f) :
    MeasurableSet {a | IsRatStieltjesPoint f a} := by
  have h1 : MeasurableSet {a | Monotone (f a)} := by
    change MeasurableSet {a | ∀ q r (_ : q ≤ r), f a q ≤ f a r}
    simp_rw [Set.setOf_forall]
    refine MeasurableSet.iInter (fun q ↦ ?_)
    refine MeasurableSet.iInter (fun r ↦ ?_)
    refine MeasurableSet.iInter (fun _ ↦ ?_)
    exact measurableSet_le hf.eval hf.eval
  have h2 : MeasurableSet {a | Tendsto (f a) atTop (𝓝 1)} :=
    measurableSet_tendsto _ (fun q ↦ hf.eval)
  have h3 : MeasurableSet {a | Tendsto (f a) atBot (𝓝 0)} :=
    measurableSet_tendsto _ (fun q ↦ hf.eval)
  have h4 : MeasurableSet {a | ∀ t : ℚ, ⨅ r : Ioi t, f a r = f a t} := by
    rw [Set.setOf_forall]
    refine MeasurableSet.iInter (fun q ↦ ?_)
    exact measurableSet_eq_fun (.iInf fun _ ↦ hf.eval) hf.eval
  suffices {a | IsRatStieltjesPoint f a}
      = ({a | Monotone (f a)} ∩ {a | Tendsto (f a) atTop (𝓝 1)} ∩ {a | Tendsto (f a) atBot (𝓝 0)}
        ∩ {a | ∀ t : ℚ, ⨅ r : Ioi t, f a r = f a t}) by
    rw [this]
    exact (((h1.inter h2).inter h3).inter h4)
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : Measurable f
    h1 : MeasurableSet (setOf fun a => Monotone (f a))
    h2 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atTop (nhds 1))
    h3 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atBot (nhds 0))
    h4 : MeasurableSet (setOf fun a => ∀ (t : Rat), Eq (iInf fun r => f a ↑r) (f a …
    ⊢ Eq (setOf fun a => ProbabilityTheory.IsRatStieltjesPoint f a) (Inter.inter ( …
  -/
  ext a
  /-
    case h
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : Measurable f
    h1 : MeasurableSet (setOf fun a => Monotone (f a))
    h2 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atTop (nhds 1))
    h3 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atBot (nhds 0))
    h4 : MeasurableSet (setOf fun a => ∀ (t : Rat), Eq (iInf fun r => f a ↑r) (f a …
    a : α
    ⊢ Iff (Membership.mem (setOf fun a => ProbabilityTheory.IsRatStieltjesPoint f  …
  -/
  simp only [mem_setOf_eq, mem_inter_iff]
  /-
    case h
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : Measurable f
    h1 : MeasurableSet (setOf fun a => Monotone (f a))
    h2 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atTop (nhds 1))
    h3 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atBot (nhds 0))
    h4 : MeasurableSet (setOf fun a => ∀ (t : Rat), Eq (iInf fun r => f a ↑r) (f a …
    a : α
    ⊢ Iff (ProbabilityTheory.IsRatStieltjesPoint f a) (And (And (And (Monotone (f  …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case h.refine_1
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : Measurable f
      h1 : MeasurableSet (setOf fun a => Monotone (f a))
      h2 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atTop (nhds 1))
      h3 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atBot (nhds 0))
      h4 : MeasurableSet (setOf fun a => ∀ (t : Rat), Eq (iInf fun r => f a ↑r) (f a …
      a : α
      h : ProbabilityTheory.IsRatStieltjesPoint f a
      ⊢ And (And (And (Monotone (f a)) (Filter.Tendsto (f a) Filter.atTop (nhds 1))) …
    -/
  · exact ⟨⟨⟨h.mono, h.tendsto_atTop_one⟩, h.tendsto_atBot_zero⟩, h.iInf_rat_gt_eq⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : Measurable f
      h1 : MeasurableSet (setOf fun a => Monotone (f a))
      h2 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atTop (nhds 1))
      h3 : MeasurableSet (setOf fun a => Filter.Tendsto (f a) Filter.atBot (nhds 0))
      h4 : MeasurableSet (setOf fun a => ∀ (t : Rat), Eq (iInf fun r => f a ↑r) (f a …
      a : α
      h : And (And (And (Monotone (f a)) (Filter.Tendsto (f a) Filter.atTop (nhds 1) …
      ⊢ ProbabilityTheory.IsRatStieltjesPoint f a
    -/
  · exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩
    /-
      🎉 no goals
    -/


lemma IsRatStieltjesPoint.ite {f g : α → ℚ → ℝ} {a : α} (p : α → Prop) [DecidablePred p]
    (hf : p a → IsRatStieltjesPoint f a) (hg : ¬ p a → IsRatStieltjesPoint g a) :
    IsRatStieltjesPoint (fun a ↦ if p a then f a else g a) a where
             /-
               α : Type u_1
               f g : α → Rat → Real
               a : α
               p : α → Prop
               inst✝ : DecidablePred p
               hf : p a → ProbabilityTheory.IsRatStieltjesPoint f a
               hg : Not (p a) → ProbabilityTheory.IsRatStieltjesPoint g a
               ⊢ Monotone (_root_.ite (p a) (f a) (g a))
             -/
  mono := by split_ifs with h; exacts [(hf h).mono, (hg h).mono]
                               /-
                                 🎉 no goals
                               -/
  tendsto_atTop_one := by
    /-
      α : Type u_1
      f g : α → Rat → Real
      a : α
      p : α → Prop
      inst✝ : DecidablePred p
      hf : p a → ProbabilityTheory.IsRatStieltjesPoint f a
      hg : Not (p a) → ProbabilityTheory.IsRatStieltjesPoint g a
      ⊢ Filter.Tendsto (_root_.ite (p a) (f a) (g a)) Filter.atTop (nhds 1)
    -/
    split_ifs with h; exacts [(hf h).tendsto_atTop_one, (hg h).tendsto_atTop_one]
                      /-
                        🎉 no goals
                      -/
  tendsto_atBot_zero := by
    /-
      α : Type u_1
      f g : α → Rat → Real
      a : α
      p : α → Prop
      inst✝ : DecidablePred p
      hf : p a → ProbabilityTheory.IsRatStieltjesPoint f a
      hg : Not (p a) → ProbabilityTheory.IsRatStieltjesPoint g a
      ⊢ Filter.Tendsto (_root_.ite (p a) (f a) (g a)) Filter.atBot (nhds 0)
    -/
    split_ifs with h; exacts [(hf h).tendsto_atBot_zero, (hg h).tendsto_atBot_zero]
                      /-
                        🎉 no goals
                      -/
                       /-
                         α : Type u_1
                         f g : α → Rat → Real
                         a : α
                         p : α → Prop
                         inst✝ : DecidablePred p
                         hf : p a → ProbabilityTheory.IsRatStieltjesPoint f a
                         hg : Not (p a) → ProbabilityTheory.IsRatStieltjesPoint g a
                         ⊢ ∀ (t : Rat), Eq (iInf fun r => _root_.ite (p a) (f a) (g a) ↑r) (_root_.ite  …
                       -/
  iInf_rat_gt_eq := by split_ifs with h; exacts [(hf h).iInf_rat_gt_eq, (hg h).iInf_rat_gt_eq]
                                         /-
                                           🎉 no goals
                                         -/


/-- A function `f : α → ℚ → ℝ` is a (kernel) rational cumulative distribution function if it is
measurable in the first argument and if `f a` satisfies a list of properties for all `a : α`:
monotonicity between 0 at -∞ and 1 at +∞ and a form of continuity.

A function with these properties can be extended to a measurable function `α → StieltjesFunction`.
See `ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunction`.
-/
structure IsMeasurableRatCDF (f : α → ℚ → ℝ) : Prop where
  isRatStieltjesPoint : ∀ a, IsRatStieltjesPoint f a
  measurable : Measurable f


lemma IsMeasurableRatCDF.nonneg {f : α → ℚ → ℝ} (hf : IsMeasurableRatCDF f) (a : α) (q : ℚ) :
    0 ≤ f a q :=
  Monotone.le_of_tendsto (hf.isRatStieltjesPoint a).mono
    (hf.isRatStieltjesPoint a).tendsto_atBot_zero q


lemma IsMeasurableRatCDF.le_one {f : α → ℚ → ℝ} (hf : IsMeasurableRatCDF f) (a : α) (q : ℚ) :
    f a q ≤ 1 :=
  Monotone.ge_of_tendsto (hf.isRatStieltjesPoint a).mono
    (hf.isRatStieltjesPoint a).tendsto_atTop_one q


lemma IsMeasurableRatCDF.tendsto_atTop_one {f : α → ℚ → ℝ} (hf : IsMeasurableRatCDF f) (a : α) :
    Tendsto (f a) atTop (𝓝 1) := (hf.isRatStieltjesPoint a).tendsto_atTop_one


lemma IsMeasurableRatCDF.tendsto_atBot_zero {f : α → ℚ → ℝ} (hf : IsMeasurableRatCDF f) (a : α) :
    Tendsto (f a) atBot (𝓝 0) := (hf.isRatStieltjesPoint a).tendsto_atBot_zero


lemma IsMeasurableRatCDF.iInf_rat_gt_eq {f : α → ℚ → ℝ} (hf : IsMeasurableRatCDF f) (a : α)
    (q : ℚ) :
    ⨅ r : Ioi q, f a r = f a q := (hf.isRatStieltjesPoint a).iInf_rat_gt_eq q


/-- A function with the property `IsMeasurableRatCDF`.
Used in a piecewise construction to convert a function which only satisfies the properties
defining `IsMeasurableRatCDF` on some set into a true `IsMeasurableRatCDF`. -/
def defaultRatCDF (q : ℚ) := if q < 0 then (0 : ℝ) else 1


lemma monotone_defaultRatCDF : Monotone defaultRatCDF := by
  /-
    ⊢ Monotone ProbabilityTheory.defaultRatCDF
  -/
  unfold defaultRatCDF
  /-
    ⊢ Monotone fun q => ite (LT.lt q 0) 0 1
  -/
  intro x y hxy
  /-
    x y : Rat
    hxy : LE.le x y
    ⊢ LE.le ((fun q => ite (LT.lt q 0) 0 1) x) ((fun q => ite (LT.lt q 0) 0 1) y)
  -/
  dsimp only
  /-
    x y : Rat
    hxy : LE.le x y
    ⊢ LE.le (ite (LT.lt x 0) 0 1) (ite (LT.lt y 0) 0 1)
  -/
  split_ifs with h_1 h_2 h_2
  /-
    case pos
    x y : Rat
    hxy : LE.le x y
    h_1 : LT.lt x 0
    h_2 : LT.lt y 0
    ⊢ LE.le 0 0
  -/
  exacts [le_rfl, zero_le_one, absurd (hxy.trans_lt h_2) h_1, le_rfl]
  /-
    🎉 no goals
  -/


lemma defaultRatCDF_nonneg (q : ℚ) : 0 ≤ defaultRatCDF q := by
  /-
    q : Rat
    ⊢ LE.le 0 (ProbabilityTheory.defaultRatCDF q)
  -/
  unfold defaultRatCDF
  /-
    q : Rat
    ⊢ LE.le 0 (ite (LT.lt q 0) 0 1)
  -/
  split_ifs
  /-
    case pos
    q : Rat
    h✝ : LT.lt q 0
    ⊢ LE.le 0 0
  -/
  exacts [le_rfl, zero_le_one]
  /-
    🎉 no goals
  -/


lemma defaultRatCDF_le_one (q : ℚ) : defaultRatCDF q ≤ 1 := by
  /-
    q : Rat
    ⊢ LE.le (ProbabilityTheory.defaultRatCDF q) 1
  -/
  unfold defaultRatCDF
  /-
    q : Rat
    ⊢ LE.le (ite (LT.lt q 0) 0 1) 1
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


lemma tendsto_defaultRatCDF_atTop : Tendsto defaultRatCDF atTop (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto ProbabilityTheory.defaultRatCDF Filter.atTop (nhds 1)
  -/
  refine (tendsto_congr' ?_).mp tendsto_const_nhds
  /-
    ⊢ Filter.atTop.EventuallyEq (fun x => 1) ProbabilityTheory.defaultRatCDF
  -/
  rw [EventuallyEq, eventually_atTop]
  /-
    ⊢ Exists fun a => ∀ (b : Rat), GE.ge b a → Eq 1 (ProbabilityTheory.defaultRatC …
  -/
  exact ⟨0, fun q hq => (if_neg (not_lt.mpr hq)).symm⟩
  /-
    🎉 no goals
  -/


lemma tendsto_defaultRatCDF_atBot : Tendsto defaultRatCDF atBot (𝓝 0) := by
  /-
    ⊢ Filter.Tendsto ProbabilityTheory.defaultRatCDF Filter.atBot (nhds 0)
  -/
  refine (tendsto_congr' ?_).mp tendsto_const_nhds
  /-
    ⊢ Filter.atBot.EventuallyEq (fun x => 0) ProbabilityTheory.defaultRatCDF
  -/
  rw [EventuallyEq, eventually_atBot]
  /-
    ⊢ Exists fun a => ∀ (b : Rat), LE.le b a → Eq 0 (ProbabilityTheory.defaultRatC …
  -/
  refine ⟨-1, fun q hq => (if_pos (hq.trans_lt ?_)).symm⟩
  /-
    q : Rat
    hq : LE.le q (-1)
    ⊢ LT.lt (-1) 0
  -/
  linarith
  /-
    🎉 no goals
  -/


lemma iInf_rat_gt_defaultRatCDF (t : ℚ) :
    ⨅ r : Ioi t, defaultRatCDF r = defaultRatCDF t := by
  /-
    t : Rat
    ⊢ Eq (iInf fun r => ProbabilityTheory.defaultRatCDF ↑r) (ProbabilityTheory.def …
  -/
  simp only [defaultRatCDF]
  have h_bdd : BddBelow (range fun r : ↥(Ioi t) ↦ ite ((r : ℚ) < 0) (0 : ℝ) 1) := by
    refine ⟨0, fun x hx ↦ ?_⟩
    obtain ⟨y, rfl⟩ := mem_range.mpr hx
    dsimp only
    split_ifs
    exacts [le_rfl, zero_le_one]
  /-
    t : Rat
    h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
    ⊢ Eq (iInf fun r => ite (LT.lt (↑r) 0) 0 1) (ite (LT.lt t 0) 0 1)
  -/
  split_ifs with h
    /-
      case pos
      t : Rat
      h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
      h : LT.lt t 0
      ⊢ Eq (iInf fun r => ite (LT.lt (↑r) 0) 0 1) 0
    -/
  · refine le_antisymm ?_ (le_ciInf fun x ↦ ?_)
      /-
        case pos.refine_1
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        ⊢ LE.le (iInf fun r => ite (LT.lt (↑r) 0) 0 1) 0
      -/
    · obtain ⟨q, htq, hq_neg⟩ : ∃ q, t < q ∧ q < 0 := ⟨t / 2, by linarith, by linarith⟩
      /-
        case pos.refine_1.intro.intro
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        q : Rat
        htq : LT.lt t q
        hq_neg : LT.lt q 0
        ⊢ LE.le (iInf fun r => ite (LT.lt (↑r) 0) 0 1) 0
      -/
      refine (ciInf_le h_bdd ⟨q, htq⟩).trans ?_
      /-
        case pos.refine_1.intro.intro
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        q : Rat
        htq : LT.lt t q
        hq_neg : LT.lt q 0
        ⊢ LE.le (ite (LT.lt (↑⟨q, htq⟩) 0) 0 1) 0
      -/
      rw [if_pos]
      /-
        case pos.refine_1.intro.intro.hc
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        q : Rat
        htq : LT.lt t q
        hq_neg : LT.lt q 0
        ⊢ LT.lt (↑⟨q, htq⟩) 0
      -/
      rwa [Subtype.coe_mk]
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        x : ↑(Set.Ioi t)
        ⊢ LE.le 0 (ite (LT.lt (↑x) 0) 0 1)
      -/
    · split_ifs
      /-
        case pos
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LT.lt t 0
        x : ↑(Set.Ioi t)
        h✝ : LT.lt (↑x) 0
        ⊢ LE.le 0 0
      -/
      exacts [le_rfl, zero_le_one]
      /-
        🎉 no goals
      -/
    /-
      case neg
      t : Rat
      h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
      h : Not (LT.lt t 0)
      ⊢ Eq (iInf fun r => ite (LT.lt (↑r) 0) 0 1) 1
    -/
  · refine le_antisymm ?_ ?_
      /-
        case neg.refine_1
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        ⊢ LE.le (iInf fun r => ite (LT.lt (↑r) 0) 0 1) 1
      -/
    · refine (ciInf_le h_bdd ⟨t + 1, lt_add_one t⟩).trans ?_
      /-
        case neg.refine_1
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        ⊢ LE.le (ite (LT.lt (↑⟨HAdd.hAdd t 1, ⋯⟩) 0) 0 1) 1
      -/
      split_ifs
      /-
        case pos
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        h✝ : LT.lt (↑⟨HAdd.hAdd t 1, ⋯⟩) 0
        ⊢ LE.le 0 1
      -/
      exacts [zero_le_one, le_rfl]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        ⊢ LE.le 1 (iInf fun r => ite (LT.lt (↑r) 0) 0 1)
      -/
    · refine le_ciInf fun x ↦ ?_
      /-
        case neg.refine_2
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        x : ↑(Set.Ioi t)
        ⊢ LE.le 1 (ite (LT.lt (↑x) 0) 0 1)
      -/
      rw [if_neg]
      /-
        case neg.refine_2.hnc
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : Not (LT.lt t 0)
        x : ↑(Set.Ioi t)
        ⊢ Not (LT.lt (↑x) 0)
      -/
      rw [not_lt] at h ⊢
      /-
        case neg.refine_2.hnc
        t : Rat
        h_bdd : BddBelow (Set.range fun r => ite (LT.lt (↑r) 0) 0 1)
        h : LE.le 0 t
        x : ↑(Set.Ioi t)
        ⊢ LE.le 0 ↑x
      -/
      exact h.trans (mem_Ioi.mp x.prop).le
      /-
        🎉 no goals
      -/


lemma isRatStieltjesPoint_defaultRatCDF (a : α) :
    IsRatStieltjesPoint (fun (_ : α) ↦ defaultRatCDF) a where
  mono := monotone_defaultRatCDF
  tendsto_atTop_one := tendsto_defaultRatCDF_atTop
  tendsto_atBot_zero := tendsto_defaultRatCDF_atBot
  iInf_rat_gt_eq := iInf_rat_gt_defaultRatCDF


lemma IsMeasurableRatCDF_defaultRatCDF (α : Type*) [MeasurableSpace α] :
    IsMeasurableRatCDF (fun (_ : α) (q : ℚ) ↦ defaultRatCDF q) where
  isRatStieltjesPoint := isRatStieltjesPoint_defaultRatCDF
  measurable := measurable_const


open scoped Classical in
/-- Turn a function `f : α → ℚ → ℝ` into another with the property `IsRatStieltjesPoint f a`
everywhere. At `a` that does not satisfy that property, `f a` is replaced by an arbitrary suitable
function.
Mainly useful when `f` satisfies the property `IsRatStieltjesPoint f a` almost everywhere with
respect to some measure. -/
noncomputable
def toRatCDF (f : α → ℚ → ℝ) : α → ℚ → ℝ := fun a ↦
  if IsRatStieltjesPoint f a then f a else defaultRatCDF


lemma toRatCDF_of_isRatStieltjesPoint {a : α} (h : IsRatStieltjesPoint f a) (q : ℚ) :
    toRatCDF f a q = f a q := by
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    h : ProbabilityTheory.IsRatStieltjesPoint f a
    q : Rat
    ⊢ Eq (ProbabilityTheory.toRatCDF f a q) (f a q)
  -/
  rw [toRatCDF, if_pos h]
  /-
    🎉 no goals
  -/


lemma toRatCDF_unit_prod (a : α) :
    toRatCDF (fun (p : Unit × α) ↦ f p.2) ((), a) = toRatCDF f a := by
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    ⊢ Eq (ProbabilityTheory.toRatCDF (fun p => f p.2) { fst := Unit.unit, snd := a …
  -/
  unfold toRatCDF
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    ⊢ Eq (ite (ProbabilityTheory.IsRatStieltjesPoint (fun p => f p.2) { fst := Uni …
  -/
  rw [isRatStieltjesPoint_unit_prod_iff]
  /-
    🎉 no goals
  -/


lemma measurable_toRatCDF (hf : Measurable f) : Measurable (toRatCDF f) :=
  Measurable.ite (measurableSet_isRatStieltjesPoint hf) hf measurable_const


lemma isMeasurableRatCDF_toRatCDF (hf : Measurable f) :
    IsMeasurableRatCDF (toRatCDF f) where
  isRatStieltjesPoint a := by
    classical
    exact IsRatStieltjesPoint.ite (IsRatStieltjesPoint f) id
      (fun _ ↦ isRatStieltjesPoint_defaultRatCDF a)
  measurable := measurable_toRatCDF hf


/-- Auxiliary definition for `IsMeasurableRatCDF.stieltjesFunction`: turn `f : α → ℚ → ℝ` into
a function `α → ℝ → ℝ` by assigning to `f a x` the infimum of `f a q` over `q : ℚ` with `x < q`. -/
noncomputable irreducible_def IsMeasurableRatCDF.stieltjesFunctionAux (f : α → ℚ → ℝ) :
    α → ℝ → ℝ :=
  fun a x ↦ ⨅ q : { q' : ℚ // x < q' }, f a q


lemma IsMeasurableRatCDF.stieltjesFunctionAux_def' (f : α → ℚ → ℝ) (a : α) :
    IsMeasurableRatCDF.stieltjesFunctionAux f a
      = fun (t : ℝ) ↦ ⨅ r : { r' : ℚ // t < r' }, f a r := by
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    ⊢ Eq (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a) fun t =>  …
  -/
  ext t; exact IsMeasurableRatCDF.stieltjesFunctionAux_def f a t
         /-
           🎉 no goals
         -/


lemma IsMeasurableRatCDF.stieltjesFunctionAux_unit_prod {f : α → ℚ → ℝ} (a : α) :
    IsMeasurableRatCDF.stieltjesFunctionAux (fun (p : Unit × α) ↦ f p.2) ((), a)
      = IsMeasurableRatCDF.stieltjesFunctionAux f a := by
  /-
    α : Type u_1
    f : α → Rat → Real
    a : α
    ⊢ Eq (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux (fun p => f p. …
  -/
  simp_rw [IsMeasurableRatCDF.stieltjesFunctionAux_def']
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.stieltjesFunctionAux_eq (a : α) (r : ℚ) :
    IsMeasurableRatCDF.stieltjesFunctionAux f a r = f a r := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    r : Rat
    ⊢ Eq (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a ↑r) (f a r)
  -/
  rw [← hf.iInf_rat_gt_eq a r, IsMeasurableRatCDF.stieltjesFunctionAux]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    r : Rat
    ⊢ Eq (iInf fun q => f a ↑q) (iInf fun r_1 => f a ↑r_1)
  -/
  refine Equiv.iInf_congr ?_ ?_
  · exact
      { toFun := fun t ↦ ⟨t.1, mod_cast t.2⟩
        invFun := fun t ↦ ⟨t.1, mod_cast t.2⟩
        left_inv := fun t ↦ by simp only [Subtype.coe_eta]
        right_inv := fun t ↦ by simp only [Subtype.coe_eta] }
    /-
      case refine_2
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      r : Rat
      ⊢ ∀ (x : Subtype fun q' => LT.lt ↑r ↑q'), Eq (f a ↑({ toFun := fun t => ⟨↑t, ⋯ …
    -/
  · intro t
    /-
      case refine_2
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      r : Rat
      t : Subtype fun q' => LT.lt ↑r ↑q'
      ⊢ Eq (f a ↑({ toFun := fun t => ⟨↑t, ⋯⟩, invFun := fun t => ⟨↑t, ⋯⟩, left_inv  …
    -/
    simp only [Equiv.coe_fn_mk, Subtype.coe_mk]
    /-
      🎉 no goals
    -/


lemma IsMeasurableRatCDF.stieltjesFunctionAux_nonneg (a : α) (r : ℝ) :
    0 ≤ IsMeasurableRatCDF.stieltjesFunctionAux f a r := by
  have : Nonempty { r' : ℚ // r < ↑r' } := by
    obtain ⟨r, hrx⟩ := exists_rat_gt r
    exact ⟨⟨r, hrx⟩⟩
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    r : Real
    this : Nonempty (Subtype fun r' => LT.lt r ↑r')
    ⊢ LE.le 0 (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a r)
  -/
  rw [IsMeasurableRatCDF.stieltjesFunctionAux_def]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    r : Real
    this : Nonempty (Subtype fun r' => LT.lt r ↑r')
    ⊢ LE.le 0 (iInf fun q => f a ↑q)
  -/
  exact le_ciInf fun r' ↦ hf.nonneg a _
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.monotone_stieltjesFunctionAux (a : α) :
    Monotone (IsMeasurableRatCDF.stieltjesFunctionAux f a) := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    ⊢ Monotone (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a)
  -/
  intro x y hxy
  have : Nonempty { r' : ℚ // y < ↑r' } := by
    obtain ⟨r, hrx⟩ := exists_rat_gt y
    exact ⟨⟨r, hrx⟩⟩
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x y : Real
    hxy : LE.le x y
    this : Nonempty (Subtype fun r' => LT.lt y ↑r')
    ⊢ LE.le (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a x) (Pro …
  -/
  simp_rw [IsMeasurableRatCDF.stieltjesFunctionAux_def]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x y : Real
    hxy : LE.le x y
    this : Nonempty (Subtype fun r' => LT.lt y ↑r')
    ⊢ LE.le (iInf fun q => f a ↑q) (iInf fun q => f a ↑q)
  -/
  refine le_ciInf fun r ↦ (ciInf_le ?_ ?_).trans_eq ?_
    /-
      case refine_1
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      x y : Real
      hxy : LE.le x y
      this : Nonempty (Subtype fun r' => LT.lt y ↑r')
      r : Subtype fun q' => LT.lt y ↑q'
      ⊢ BddBelow (Set.range fun q => f a ↑q)
    -/
  · refine ⟨0, fun z ↦ ?_⟩; rintro ⟨u, rfl⟩; exact hf.nonneg a _
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case refine_2
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      x y : Real
      hxy : LE.le x y
      this : Nonempty (Subtype fun r' => LT.lt y ↑r')
      r : Subtype fun q' => LT.lt y ↑q'
      ⊢ Subtype fun q' => LT.lt x ↑q'
    -/
  · exact ⟨r.1, hxy.trans_lt r.prop⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      x y : Real
      hxy : LE.le x y
      this : Nonempty (Subtype fun r' => LT.lt y ↑r')
      r : Subtype fun q' => LT.lt y ↑q'
      ⊢ Eq (f a ↑⟨↑r, ⋯⟩) (f a ↑r)
    -/
  · rfl
    /-
      🎉 no goals
    -/


lemma IsMeasurableRatCDF.continuousWithinAt_stieltjesFunctionAux_Ici (a : α) (x : ℝ) :
    ContinuousWithinAt (IsMeasurableRatCDF.stieltjesFunctionAux f a) (Ici x) x := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ ContinuousWithinAt (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAu …
  -/
  rw [← continuousWithinAt_Ioi_iff_Ici]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ ContinuousWithinAt (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAu …
  -/
  convert Monotone.tendsto_nhdsGT (monotone_stieltjesFunctionAux hf a) x
  /-
    case a
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ Iff (ContinuousWithinAt (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunct …
  -/
  rw [sInf_image']
  have h' : ⨅ r : Ioi x, stieltjesFunctionAux f a r
      = ⨅ r : { r' : ℚ // x < r' }, stieltjesFunctionAux f a r := by
    refine Real.iInf_Ioi_eq_iInf_rat_gt x ?_ (monotone_stieltjesFunctionAux hf a)
    refine ⟨0, fun z ↦ ?_⟩
    rintro ⟨u, -, rfl⟩
    exact stieltjesFunctionAux_nonneg hf a u
  have h'' :
    ⨅ r : { r' : ℚ // x < r' }, stieltjesFunctionAux f a r =
      ⨅ r : { r' : ℚ // x < r' }, f a r := by
    congr with r
    exact stieltjesFunctionAux_eq hf a r
  /-
    case a
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    h' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionA …
    h'' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunction …
    ⊢ Iff (ContinuousWithinAt (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunct …
  -/
  rw [h', h'', ContinuousWithinAt]
  /-
    case a
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    h' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionA …
    h'' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunction …
    ⊢ Iff (Filter.Tendsto (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionA …
  -/
  congr!
  /-
    case a.a.h.e'_5.h.e'_3
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    h' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionA …
    h'' : Eq (iInf fun r => ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunction …
    ⊢ Eq (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux f a x) (iInf f …
  -/
  rw [stieltjesFunctionAux_def]
  /-
    🎉 no goals
  -/


/-- Extend a function `f : α → ℚ → ℝ` with property `IsMeasurableRatCDF` from `ℚ` to `ℝ`,
to a function `α → StieltjesFunction`. -/
noncomputable def IsMeasurableRatCDF.stieltjesFunction (a : α) : StieltjesFunction where
  toFun := stieltjesFunctionAux f a
  mono' := monotone_stieltjesFunctionAux hf a
  right_continuous' x := continuousWithinAt_stieltjesFunctionAux_Ici hf a x


lemma IsMeasurableRatCDF.stieltjesFunction_eq (a : α) (r : ℚ) : hf.stieltjesFunction a r = f a r :=
  stieltjesFunctionAux_eq hf a r


lemma IsMeasurableRatCDF.stieltjesFunction_nonneg (a : α) (r : ℝ) : 0 ≤ hf.stieltjesFunction a r :=
  stieltjesFunctionAux_nonneg hf a r


lemma IsMeasurableRatCDF.stieltjesFunction_le_one (a : α) (x : ℝ) :
    hf.stieltjesFunction a x ≤ 1 := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ LE.le (↑(hf.stieltjesFunction a) x) 1
  -/
  obtain ⟨r, hrx⟩ := exists_rat_gt x
  /-
    case intro
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    r : Rat
    hrx : LT.lt x ↑r
    ⊢ LE.le (↑(hf.stieltjesFunction a) x) 1
  -/
  rw [← StieltjesFunction.iInf_rat_gt_eq]
  /-
    case intro
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    r : Rat
    hrx : LT.lt x ↑r
    ⊢ LE.le (iInf fun r => ↑(hf.stieltjesFunction a) ↑↑r) 1
  -/
  simp_rw [IsMeasurableRatCDF.stieltjesFunction_eq]
  /-
    case intro
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    r : Rat
    hrx : LT.lt x ↑r
    ⊢ LE.le (iInf fun r => f a ↑r) 1
  -/
  refine ciInf_le_of_le ?_ ?_ (hf.le_one _ _)
    /-
      case intro.refine_1
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      x : Real
      r : Rat
      hrx : LT.lt x ↑r
      ⊢ BddBelow (Set.range fun r => f a ↑r)
    -/
  · refine ⟨0, fun z ↦ ?_⟩; rintro ⟨u, rfl⟩; exact hf.nonneg a _
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case intro.refine_2
      α : Type u_1
      f : α → Rat → Real
      inst✝ : MeasurableSpace α
      hf : ProbabilityTheory.IsMeasurableRatCDF f
      a : α
      x : Real
      r : Rat
      hrx : LT.lt x ↑r
      ⊢ Subtype fun r' => LT.lt x ↑r'
    -/
  · exact ⟨r, hrx⟩
    /-
      🎉 no goals
    -/


lemma IsMeasurableRatCDF.tendsto_stieltjesFunction_atBot (a : α) :
    Tendsto (hf.stieltjesFunction a) atBot (𝓝 0) := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    ⊢ Filter.Tendsto (↑(hf.stieltjesFunction a)) Filter.atBot (nhds 0)
  -/
  have h_exists : ∀ x : ℝ, ∃ q : ℚ, x < q ∧ ↑q < x + 1 := fun x ↦ exists_rat_btwn (lt_add_one x)
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) (HAdd.hA …
    ⊢ Filter.Tendsto (↑(hf.stieltjesFunction a)) Filter.atBot (nhds 0)
  -/
  let qs : ℝ → ℚ := fun x ↦ (h_exists x).choose
  have hqs_tendsto : Tendsto qs atBot atBot := by
    rw [tendsto_atBot_atBot]
    refine fun q ↦ ⟨q - 1, fun y hy ↦ ?_⟩
    have h_le : ↑(qs y) ≤ (q : ℝ) - 1 + 1 :=
      (h_exists y).choose_spec.2.le.trans (add_le_add hy le_rfl)
    rw [sub_add_cancel] at h_le
    exact mod_cast h_le
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    ((hf.tendsto_atBot_zero a).comp hqs_tendsto) (stieltjesFunction_nonneg hf a) fun x ↦ ?_
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) (HAdd.hA …
    qs : Real → Rat := fun x => ⋯.choose
    hqs_tendsto : Filter.Tendsto qs Filter.atBot Filter.atBot
    x : Real
    ⊢ LE.le (↑(hf.stieltjesFunction a) x) (Function.comp (f a) qs x)
  -/
  rw [Function.comp_apply, ← stieltjesFunction_eq hf]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) (HAdd.hA …
    qs : Real → Rat := fun x => ⋯.choose
    hqs_tendsto : Filter.Tendsto qs Filter.atBot Filter.atBot
    x : Real
    ⊢ LE.le (↑(hf.stieltjesFunction a) x) (↑(hf.stieltjesFunction a) ↑(qs x))
  -/
  exact (hf.stieltjesFunction a).mono (h_exists x).choose_spec.1.le
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.tendsto_stieltjesFunction_atTop (a : α) :
    Tendsto (hf.stieltjesFunction a) atTop (𝓝 1) := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    ⊢ Filter.Tendsto (↑(hf.stieltjesFunction a)) Filter.atTop (nhds 1)
  -/
  have h_exists : ∀ x : ℝ, ∃ q : ℚ, x - 1 < q ∧ ↑q < x := fun x ↦ exists_rat_btwn (sub_one_lt x)
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt (HSub.hSub x 1) ↑q) (LT.lt …
    ⊢ Filter.Tendsto (↑(hf.stieltjesFunction a)) Filter.atTop (nhds 1)
  -/
  let qs : ℝ → ℚ := fun x ↦ (h_exists x).choose
  have hqs_tendsto : Tendsto qs atTop atTop := by
    rw [tendsto_atTop_atTop]
    refine fun q ↦ ⟨q + 1, fun y hy ↦ ?_⟩
    have h_le : y - 1 ≤ qs y := (h_exists y).choose_spec.1.le
    rw [sub_le_iff_le_add] at h_le
    exact_mod_cast le_of_add_le_add_right (hy.trans h_le)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le ((hf.tendsto_atTop_one a).comp hqs_tendsto)
      tendsto_const_nhds ?_ (stieltjesFunction_le_one hf a)
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt (HSub.hSub x 1) ↑q) (LT.lt …
    qs : Real → Rat := fun x => ⋯.choose
    hqs_tendsto : Filter.Tendsto qs Filter.atTop Filter.atTop
    ⊢ LE.le (Function.comp (f a) qs) ↑(hf.stieltjesFunction a)
  -/
  intro x
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt (HSub.hSub x 1) ↑q) (LT.lt …
    qs : Real → Rat := fun x => ⋯.choose
    hqs_tendsto : Filter.Tendsto qs Filter.atTop Filter.atTop
    x : Real
    ⊢ LE.le (Function.comp (f a) qs x) (↑(hf.stieltjesFunction a) x)
  -/
  rw [Function.comp_apply, ← stieltjesFunction_eq hf]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    h_exists : ∀ (x : Real), Exists fun q => And (LT.lt (HSub.hSub x 1) ↑q) (LT.lt …
    qs : Real → Rat := fun x => ⋯.choose
    hqs_tendsto : Filter.Tendsto qs Filter.atTop Filter.atTop
    x : Real
    ⊢ LE.le (↑(hf.stieltjesFunction a) ↑(qs x)) (↑(hf.stieltjesFunction a) x)
  -/
  exact (hf.stieltjesFunction a).mono (le_of_lt (h_exists x).choose_spec.2)
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.measurable_stieltjesFunction (x : ℝ) :
    Measurable fun a ↦ hf.stieltjesFunction a x := by
  have : (fun a ↦ hf.stieltjesFunction a x) = fun a ↦ ⨅ r : { r' : ℚ // x < r' }, f a ↑r := by
    ext1 a
    rw [← StieltjesFunction.iInf_rat_gt_eq]
    congr with q
    rw [stieltjesFunction_eq]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    x : Real
    this : Eq (fun a => ↑(hf.stieltjesFunction a) x) fun a => iInf fun r => f a ↑r
    ⊢ Measurable fun a => ↑(hf.stieltjesFunction a) x
  -/
  rw [this]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    x : Real
    this : Eq (fun a => ↑(hf.stieltjesFunction a) x) fun a => iInf fun r => f a ↑r
    ⊢ Measurable fun a => iInf fun r => f a ↑r
  -/
  exact .iInf (fun q ↦ hf.measurable.eval)
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.stronglyMeasurable_stieltjesFunction (x : ℝ) :
    StronglyMeasurable fun a ↦ hf.stieltjesFunction a x :=
  (measurable_stieltjesFunction hf x).stronglyMeasurable


lemma IsMeasurableRatCDF.measure_stieltjesFunction_Iic (a : α) (x : ℝ) :
    (hf.stieltjesFunction a).measure (Iic x) = ENNReal.ofReal (hf.stieltjesFunction a x) := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ Eq ((hf.stieltjesFunction a).measure (Set.Iic x)) (ENNReal.ofReal (↑(hf.stie …
  -/
  rw [← sub_zero (hf.stieltjesFunction a x)]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    x : Real
    ⊢ Eq ((hf.stieltjesFunction a).measure (Set.Iic x)) (ENNReal.ofReal (HSub.hSub …
  -/
  exact (hf.stieltjesFunction a).measure_Iic (tendsto_stieltjesFunction_atBot hf a) _
  /-
    🎉 no goals
  -/


lemma IsMeasurableRatCDF.measure_stieltjesFunction_univ (a : α) :
    (hf.stieltjesFunction a).measure univ = 1 := by
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : ProbabilityTheory.IsMeasurableRatCDF f
    a : α
    ⊢ Eq ((hf.stieltjesFunction a).measure Set.univ) 1
  -/
  rw [← ENNReal.ofReal_one, ← sub_zero (1 : ℝ)]
  exact StieltjesFunction.measure_univ _ (tendsto_stieltjesFunction_atBot hf a)
    (tendsto_stieltjesFunction_atTop hf a)


instance IsMeasurableRatCDF.instIsProbabilityMeasure_stieltjesFunction (a : α) :
    IsProbabilityMeasure (hf.stieltjesFunction a).measure :=
  ⟨measure_stieltjesFunction_univ hf a⟩


lemma IsMeasurableRatCDF.measurable_measure_stieltjesFunction :
    Measurable fun a ↦ (hf.stieltjesFunction a).measure := by
  apply_rules [StieltjesFunction.measurable_measure, measurable_stieltjesFunction,
    tendsto_stieltjesFunction_atBot, tendsto_stieltjesFunction_atTop]


/-- Turn a measurable function `f : α → ℚ → ℝ` into a measurable function `α → StieltjesFunction`.
Composition of `toRatCDF` and `IsMeasurableRatCDF.stieltjesFunction`. -/
noncomputable
def stieltjesOfMeasurableRat (f : α → ℚ → ℝ) (hf : Measurable f) : α → StieltjesFunction :=
  (isMeasurableRatCDF_toRatCDF hf).stieltjesFunction


lemma stieltjesOfMeasurableRat_eq (hf : Measurable f) (a : α) (r : ℚ) :
    stieltjesOfMeasurableRat f hf a r = toRatCDF f a r :=
  IsMeasurableRatCDF.stieltjesFunction_eq _ a r


lemma stieltjesOfMeasurableRat_unit_prod (hf : Measurable f) (a : α) :
    stieltjesOfMeasurableRat (fun (p : Unit × α) ↦ f p.2) (hf.comp measurable_snd) ((), a)
      = stieltjesOfMeasurableRat f hf a := by
  simp_rw [stieltjesOfMeasurableRat,IsMeasurableRatCDF.stieltjesFunction,
    ← IsMeasurableRatCDF.stieltjesFunctionAux_unit_prod a]
  /-
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : Measurable f
    a : α
    ⊢ Eq { toFun := ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux (Pro …
  -/
  congr with x
  /-
    case e_toFun.h
    α : Type u_1
    f : α → Rat → Real
    inst✝ : MeasurableSpace α
    hf : Measurable f
    a : α
    x : Real
    ⊢ Eq (ProbabilityTheory.IsMeasurableRatCDF.stieltjesFunctionAux (ProbabilityTh …
  -/
  congr 1 with p : 1
  cases p with
  | mk _ b => rw [← toRatCDF_unit_prod b]


lemma stieltjesOfMeasurableRat_nonneg (hf : Measurable f) (a : α) (r : ℝ) :
    0 ≤ stieltjesOfMeasurableRat f hf a r := IsMeasurableRatCDF.stieltjesFunction_nonneg _ a r


lemma stieltjesOfMeasurableRat_le_one (hf : Measurable f) (a : α) (x : ℝ) :
    stieltjesOfMeasurableRat f hf a x ≤ 1 := IsMeasurableRatCDF.stieltjesFunction_le_one _ a x


lemma tendsto_stieltjesOfMeasurableRat_atBot (hf : Measurable f) (a : α) :
    Tendsto (stieltjesOfMeasurableRat f hf a) atBot (𝓝 0) :=
  IsMeasurableRatCDF.tendsto_stieltjesFunction_atBot _ a


lemma tendsto_stieltjesOfMeasurableRat_atTop (hf : Measurable f) (a : α) :
    Tendsto (stieltjesOfMeasurableRat f hf a) atTop (𝓝 1) :=
  IsMeasurableRatCDF.tendsto_stieltjesFunction_atTop _ a


lemma measurable_stieltjesOfMeasurableRat (hf : Measurable f) (x : ℝ) :
    Measurable fun a ↦ stieltjesOfMeasurableRat f hf a x :=
  IsMeasurableRatCDF.measurable_stieltjesFunction _ x


lemma stronglyMeasurable_stieltjesOfMeasurableRat (hf : Measurable f) (x : ℝ) :
    StronglyMeasurable fun a ↦ stieltjesOfMeasurableRat f hf a x :=
  IsMeasurableRatCDF.stronglyMeasurable_stieltjesFunction _ x


lemma measure_stieltjesOfMeasurableRat_Iic (hf : Measurable f) (a : α) (x : ℝ) :
    (stieltjesOfMeasurableRat f hf a).measure (Iic x)
      = ENNReal.ofReal (stieltjesOfMeasurableRat f hf a x) :=
  IsMeasurableRatCDF.measure_stieltjesFunction_Iic _ _ _


lemma measure_stieltjesOfMeasurableRat_univ (hf : Measurable f) (a : α) :
    (stieltjesOfMeasurableRat f hf a).measure univ = 1 :=
  IsMeasurableRatCDF.measure_stieltjesFunction_univ _ _


instance instIsProbabilityMeasure_stieltjesOfMeasurableRat
    (hf : Measurable f) (a : α) :
    IsProbabilityMeasure (stieltjesOfMeasurableRat f hf a).measure :=
  IsMeasurableRatCDF.instIsProbabilityMeasure_stieltjesFunction _ _


lemma measurable_measure_stieltjesOfMeasurableRat (hf : Measurable f) :
    Measurable fun a ↦ (stieltjesOfMeasurableRat f hf a).measure :=
  IsMeasurableRatCDF.measurable_measure_stieltjesFunction _


