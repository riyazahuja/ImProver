/-- A sequence of functions `f` is said to converge in measure to some function `g` if for all
`ε > 0`, the measure of the set `{x | ε ≤ dist (f i x) (g x)}` tends to 0 as `i` converges along
some given filter `l`. -/
def TendstoInMeasure [Dist E] {_ : MeasurableSpace α} (μ : Measure α) (f : ι → α → E) (l : Filter ι)
    (g : α → E) : Prop :=
  ∀ ε, 0 < ε → Tendsto (fun i => μ { x | ε ≤ dist (f i x) (g x) }) l (𝓝 0)


theorem tendstoInMeasure_iff_norm [SeminormedAddCommGroup E] {l : Filter ι} {f : ι → α → E}
    {g : α → E} :
    TendstoInMeasure μ f l g ↔
      ∀ ε, 0 < ε → Tendsto (fun i => μ { x | ε ≤ ‖f i x - g x‖ }) l (𝓝 0) := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : SeminormedAddCommGroup E
    l : Filter ι
    f : ι → α → E
    g : α → E
    ⊢ Iff (MeasureTheory.TendstoInMeasure μ f l g) (∀ (ε : Real), LT.lt 0 ε → Filt …
  -/
  simp_rw [TendstoInMeasure, dist_eq_norm]
  /-
    🎉 no goals
  -/


protected theorem congr' (h_left : ∀ᶠ i in l, f i =ᵐ[μ] f' i) (h_right : g =ᵐ[μ] g')
    (h_tendsto : TendstoInMeasure μ f l g) : TendstoInMeasure μ f' l g' := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ⊢ MeasureTheory.TendstoInMeasure μ f' l g'
  -/
  intro ε hε
  suffices
    (fun i => μ { x | ε ≤ dist (f' i x) (g' x) }) =ᶠ[l] fun i => μ { x | ε ≤ dist (f i x) (g x) } by
    rw [tendsto_congr' this]
    exact h_tendsto ε hε
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    ⊢ l.EventuallyEq (fun i => μ (setOf fun x => LE.le ε (Dist.dist (f' i x) (g' x …
  -/
  filter_upwards [h_left] with i h_ae_eq
  /-
    case h
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq (f i) (f' i)
    ⊢ Eq (μ (setOf fun x => LE.le ε (Dist.dist (f' i x) (g' x)))) (μ (setOf fun x  …
  -/
  refine measure_congr ?_
  /-
    case h
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq (f i) (f' i)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun x => LE.le ε (Dist.dist (f' i x …
  -/
  filter_upwards [h_ae_eq, h_right] with x hxf hxg
  /-
    case h
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq (f i) (f' i)
    x : α
    hxf : Eq (f i x) (f' i x)
    hxg : Eq (g x) (g' x)
    ⊢ Eq (setOf (fun x => LE.le ε (Dist.dist (f' i x) (g' x))) x) (setOf (fun x => …
  -/
  rw [eq_iff_iff]
  /-
    case h
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq (f i) (f' i)
    x : α
    hxf : Eq (f i x) (f' i x)
    hxg : Eq (g x) (g' x)
    ⊢ Iff (setOf (fun x => LE.le ε (Dist.dist (f' i x) (g' x))) x) (setOf (fun x = …
  -/
  change ε ≤ dist (f' i x) (g' x) ↔ ε ≤ dist (f i x) (g x)
  /-
    case h
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Dist E
    l : Filter ι
    f f' : ι → α → E
    g g' : α → E
    h_left : Filter.Eventually (fun i => (MeasureTheory.ae μ).EventuallyEq (f i) ( …
    h_right : (MeasureTheory.ae μ).EventuallyEq g g'
    h_tendsto : MeasureTheory.TendstoInMeasure μ f l g
    ε : Real
    hε : LT.lt 0 ε
    i : ι
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq (f i) (f' i)
    x : α
    hxf : Eq (f i x) (f' i x)
    hxg : Eq (g x) (g' x)
    ⊢ Iff (LE.le ε (Dist.dist (f' i x) (g' x))) (LE.le ε (Dist.dist (f i x) (g x)))
  -/
  rw [hxg, hxf]
  /-
    🎉 no goals
  -/


protected theorem congr (h_left : ∀ i, f i =ᵐ[μ] f' i) (h_right : g =ᵐ[μ] g')
    (h_tendsto : TendstoInMeasure μ f l g) : TendstoInMeasure μ f' l g' :=
  TendstoInMeasure.congr' (Eventually.of_forall h_left) h_right h_tendsto


theorem congr_left (h : ∀ i, f i =ᵐ[μ] f' i) (h_tendsto : TendstoInMeasure μ f l g) :
    TendstoInMeasure μ f' l g :=
  h_tendsto.congr h EventuallyEq.rfl


theorem congr_right (h : g =ᵐ[μ] g') (h_tendsto : TendstoInMeasure μ f l g) :
    TendstoInMeasure μ f l g' :=
  h_tendsto.congr (fun _ => EventuallyEq.rfl) h


/-- Auxiliary lemma for `tendstoInMeasure_of_tendsto_ae`. -/
theorem tendstoInMeasure_of_tendsto_ae_of_stronglyMeasurable [IsFiniteMeasure μ]
    (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g)
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) : TendstoInMeasure μ f atTop g := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ MeasureTheory.TendstoInMeasure μ f Filter.atTop g
  -/
  refine fun ε hε => ENNReal.tendsto_atTop_zero.mpr fun δ hδ => ?_
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : ENNReal
    hδ : GT.gt δ 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  by_cases hδi : δ = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : MetricSpace E
      f : Nat → α → E
      g : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : Real
      hε : LT.lt 0 ε
      δ : ENNReal
      hδ : GT.gt δ 0
      hδi : Eq δ Top.top
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
    -/
  · simp only [hδi, imp_true_iff, le_top, exists_const]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : ENNReal
    hδ : GT.gt δ 0
    hδi : Not (Eq δ Top.top)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  lift δ to ℝ≥0 using hδi
  /-
    case neg.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : GT.gt (↑δ) 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  rw [gt_iff_lt, ENNReal.coe_pos, ← NNReal.coe_pos] at hδ
  /-
    case neg.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  obtain ⟨t, _, ht, hunif⟩ := tendstoUniformlyOn_of_ae_tendsto' hf hg hfg hδ
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) (ENNReal.ofReal ↑δ)
    hunif : TendstoUniformlyOn f g Filter.atTop (HasCompl.compl t)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  rw [ENNReal.ofReal_coe_nnreal] at ht
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : TendstoUniformlyOn f g Filter.atTop (HasCompl.compl t)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  rw [Metric.tendstoUniformlyOn_iff] at hunif
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hunif ε hε)
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (μ (setOf fun x => LE.le ε (D …
  -/
  refine ⟨N, fun n hn => ?_⟩
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    ⊢ LE.le (μ (setOf fun x => LE.le ε (Dist.dist (f n x) (g x)))) ↑δ
  -/
  suffices { x : α | ε ≤ dist (f n x) (g x) } ⊆ t from (measure_mono this).trans ht
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    ⊢ HasSubset.Subset (setOf fun x => LE.le ε (Dist.dist (f n x) (g x))) t
  -/
  rw [← Set.compl_subset_compl]
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    ⊢ HasSubset.Subset (HasCompl.compl t) (HasCompl.compl (setOf fun x => LE.le ε  …
  -/
  intro x hx
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    x : α
    hx : Membership.mem (HasCompl.compl t) x
    ⊢ Membership.mem (HasCompl.compl (setOf fun x => LE.le ε (Dist.dist (f n x) (g …
  -/
  rw [Set.mem_compl_iff, Set.nmem_setOf_iff, dist_comm, not_le]
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    δ : NNReal
    hδ : LT.lt 0 ↑δ
    t : Set α
    left✝ : MeasurableSet t
    ht : LE.le (μ t) ↑δ
    hunif : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : α), Membe …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    x : α
    hx : Membership.mem (HasCompl.compl t) x
    ⊢ LT.lt (Dist.dist (g x) (f n x)) ε
  -/
  exact hN n hn x hx
  /-
    🎉 no goals
  -/


/-- Convergence a.e. implies convergence in measure in a finite measure space. -/
theorem tendstoInMeasure_of_tendsto_ae [IsFiniteMeasure μ] (hf : ∀ n, AEStronglyMeasurable (f n) μ)
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) : TendstoInMeasure μ f atTop g := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ MeasureTheory.TendstoInMeasure μ f Filter.atTop g
  -/
  have hg : AEStronglyMeasurable g μ := aestronglyMeasurable_of_tendsto_ae _ hf hfg
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ MeasureTheory.TendstoInMeasure μ f Filter.atTop g
  -/
  refine TendstoInMeasure.congr (fun i => (hf i).ae_eq_mk.symm) hg.ae_eq_mk.symm ?_
  refine tendstoInMeasure_of_tendsto_ae_of_stronglyMeasurable
    (fun i => (hf i).stronglyMeasurable_mk) hg.stronglyMeasurable_mk ?_
  have hf_eq_ae : ∀ᵐ x ∂μ, ∀ n, (hf n).mk (f n) x = f n x :=
    ae_all_iff.mpr fun n => (hf n).ae_eq_mk.symm
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hf_eq_ae : Filter.Eventually (fun x => ∀ (n : Nat), Eq (MeasureTheory.AEStrong …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AEStrongl …
  -/
  filter_upwards [hf_eq_ae, hg.ae_eq_mk, hfg] with x hxf hxg hxfg
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hf_eq_ae : Filter.Eventually (fun x => ∀ (n : Nat), Eq (MeasureTheory.AEStrong …
    x : α
    hxf : ∀ (n : Nat), Eq (MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) (f n x)
    hxg : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    hxfg : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    ⊢ Filter.Tendsto (fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) Fi …
  -/
  rw [← hxg, funext fun n => hxf n]
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MetricSpace E
    f : Nat → α → E
    g : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hf_eq_ae : Filter.Eventually (fun x => ∀ (n : Nat), Eq (MeasureTheory.AEStrong …
    x : α
    hxf : ∀ (n : Nat), Eq (MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) (f n x)
    hxg : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    hxfg : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    ⊢ Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
  -/
  exact hxfg
  /-
    🎉 no goals
  -/


theorem exists_nat_measure_lt_two_inv (hfg : TendstoInMeasure μ f atTop g) (n : ℕ) :
    ∃ N, ∀ m ≥ N, μ { x | (2 : ℝ)⁻¹ ^ n ≤ dist (f m x) (g x) } ≤ (2⁻¹ : ℝ≥0∞) ^ n := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    n : Nat
    ⊢ Exists fun N => ∀ (m_1 : Nat), GE.ge m_1 N → LE.le (μ (setOf fun x => LE.le  …
  -/
  specialize hfg ((2⁻¹ : ℝ) ^ n) (by simp only [Real.rpow_natCast, inv_pos, zero_lt_two, pow_pos])
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    n : Nat
    hfg : Filter.Tendsto (fun i => μ (setOf fun x => LE.le (HPow.hPow (Inv.inv 2)  …
    ⊢ Exists fun N => ∀ (m_1 : Nat), GE.ge m_1 N → LE.le (μ (setOf fun x => LE.le  …
  -/
  rw [ENNReal.tendsto_atTop_zero] at hfg
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    n : Nat
    hfg : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n_1 : Nat), GE.ge n_1 N  …
    ⊢ Exists fun N => ∀ (m_1 : Nat), GE.ge m_1 N → LE.le (μ (setOf fun x => LE.le  …
  -/
  exact hfg ((2 : ℝ≥0∞)⁻¹ ^ n) (pos_iff_ne_zero.mpr fun h_zero => by simpa using pow_eq_zero h_zero)
  /-
    🎉 no goals
  -/


/-- Given a sequence of functions `f` which converges in measure to `g`,
`seqTendstoAeSeqAux` is a sequence such that
`∀ m ≥ seqTendstoAeSeqAux n, μ {x | 2⁻¹ ^ n ≤ dist (f m x) (g x)} ≤ 2⁻¹ ^ n`. -/
noncomputable def seqTendstoAeSeqAux (hfg : TendstoInMeasure μ f atTop g) (n : ℕ) :=
  Classical.choose (exists_nat_measure_lt_two_inv hfg n)


/-- Transformation of `seqTendstoAeSeqAux` to makes sure it is strictly monotone. -/
noncomputable def seqTendstoAeSeq (hfg : TendstoInMeasure μ f atTop g) : ℕ → ℕ
  | 0 => seqTendstoAeSeqAux hfg 0
  | n + 1 => max (seqTendstoAeSeqAux hfg (n + 1)) (seqTendstoAeSeq hfg n + 1)


theorem seqTendstoAeSeq_succ (hfg : TendstoInMeasure μ f atTop g) {n : ℕ} :
    seqTendstoAeSeq hfg (n + 1) =
      max (seqTendstoAeSeqAux hfg (n + 1)) (seqTendstoAeSeq hfg n + 1) := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    n : Nat
    ⊢ Eq (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg (HAdd.hAdd n 1)) (M …
  -/
  rw [seqTendstoAeSeq]
  /-
    🎉 no goals
  -/


theorem seqTendstoAeSeq_spec (hfg : TendstoInMeasure μ f atTop g) (n k : ℕ)
    (hn : seqTendstoAeSeq hfg n ≤ k) :
    μ { x | (2 : ℝ)⁻¹ ^ n ≤ dist (f k x) (g x) } ≤ (2 : ℝ≥0∞)⁻¹ ^ n := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    n k : Nat
    hn : LE.le (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg n) k
    ⊢ LE.le (μ (setOf fun x => LE.le (HPow.hPow (Inv.inv 2) n) (Dist.dist (f k x)  …
  -/
  cases n
    /-
      case zero
      α : Type u_1
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MetricSpace E
      f : Nat → α → E
      g : α → E
      hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
      k : Nat
      hn : LE.le (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg 0) k
      ⊢ LE.le (μ (setOf fun x => LE.le (HPow.hPow (Inv.inv 2) 0) (Dist.dist (f k x)  …
    -/
  · exact Classical.choose_spec (exists_nat_measure_lt_two_inv hfg 0) k hn
    /-
      🎉 no goals
    -/
  · exact Classical.choose_spec
      (exists_nat_measure_lt_two_inv hfg _) _ (le_trans (le_max_left _ _) hn)


theorem seqTendstoAeSeq_strictMono (hfg : TendstoInMeasure μ f atTop g) :
    StrictMono (seqTendstoAeSeq hfg) := by
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    ⊢ StrictMono (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg)
  -/
  refine strictMono_nat_of_lt_succ fun n => ?_
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    n : Nat
    ⊢ LT.lt (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg n) (MeasureTheor …
  -/
  rw [seqTendstoAeSeq_succ]
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    n : Nat
    ⊢ LT.lt (MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg n) (Max.max (Mea …
  -/
  exact lt_of_lt_of_le (lt_add_one <| seqTendstoAeSeq hfg n) (le_max_right _ _)
  /-
    🎉 no goals
  -/


/-- If `f` is a sequence of functions which converges in measure to `g`, then there exists a
subsequence of `f` which converges a.e. to `g`. -/
theorem TendstoInMeasure.exists_seq_tendsto_ae (hfg : TendstoInMeasure μ f atTop g) :
    ∃ ns : ℕ → ℕ, StrictMono ns ∧ ∀ᵐ x ∂μ, Tendsto (fun i => f (ns i) x) atTop (𝓝 (g x)) := by
  /- Since `f` tends to `g` in measure, it has a subsequence `k ↦ f (ns k)` such that
    `μ {|f (ns k) - g| ≥ 2⁻ᵏ} ≤ 2⁻ᵏ` for all `k`. Defining
    `s := ⋂ k, ⋃ i ≥ k, {|f (ns k) - g| ≥ 2⁻ᵏ}`, we see that `μ s = 0` by the
    first Borel-Cantelli lemma.

    On the other hand, as `s` is precisely the set for which `f (ns k)`
    doesn't converge to `g`, `f (ns k)` converges almost everywhere to `g` as required. -/
  have h_lt_ε_real : ∀ (ε : ℝ) (_ : 0 < ε), ∃ k : ℕ, 2 * (2 : ℝ)⁻¹ ^ k < ε := by
    intro ε hε
    obtain ⟨k, h_k⟩ : ∃ k : ℕ, (2 : ℝ)⁻¹ ^ k < ε := exists_pow_lt_of_lt_one hε (by norm_num)
    refine ⟨k + 1, (le_of_eq ?_).trans_lt h_k⟩
    rw [pow_add]; ring
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ⊢ Exists fun ns => And (StrictMono ns) (Filter.Eventually (fun x => Filter.Ten …
  -/
  set ns := ExistsSeqTendstoAe.seqTendstoAeSeq hfg
  /-
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    ⊢ Exists fun ns => And (StrictMono ns) (Filter.Eventually (fun x => Filter.Ten …
  -/
  use ns
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    ⊢ And (StrictMono ns) (Filter.Eventually (fun x => Filter.Tendsto (fun i => f  …
  -/
  let S := fun k => { x | (2 : ℝ)⁻¹ ^ k ≤ dist (f (ns k) x) (g x) }
  have hμS_le : ∀ k, μ (S k) ≤ (2 : ℝ≥0∞)⁻¹ ^ k :=
    fun k => ExistsSeqTendstoAe.seqTendstoAeSeq_spec hfg k (ns k) le_rfl
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    S : Nat → Set α := fun k => setOf fun x => LE.le (HPow.hPow (Inv.inv 2) k) (Di …
    hμS_le : ∀ (k : Nat), LE.le (μ (S k)) (HPow.hPow (Inv.inv 2) k)
    ⊢ And (StrictMono ns) (Filter.Eventually (fun x => Filter.Tendsto (fun i => f  …
  -/
  set s := Filter.atTop.limsup S with hs
  have hμs : μ s = 0 := by
    refine measure_limsup_atTop_eq_zero (ne_top_of_le_ne_top ?_ (ENNReal.tsum_le_tsum hμS_le))
    simpa only [ENNReal.tsum_geometric, ENNReal.one_sub_inv_two, inv_inv] using ENNReal.two_ne_top
  have h_tendsto : ∀ x ∈ sᶜ, Tendsto (fun i => f (ns i) x) atTop (𝓝 (g x)) := by
    refine fun x hx => Metric.tendsto_atTop.mpr fun ε hε => ?_
    rw [hs, limsup_eq_iInf_iSup_of_nat] at hx
    simp only [S, Set.iSup_eq_iUnion, Set.iInf_eq_iInter, Set.compl_iInter, Set.compl_iUnion,
      Set.mem_iUnion, Set.mem_iInter, Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hx
    obtain ⟨N, hNx⟩ := hx
    obtain ⟨k, hk_lt_ε⟩ := h_lt_ε_real ε hε
    refine ⟨max N (k - 1), fun n hn_ge => lt_of_le_of_lt ?_ hk_lt_ε⟩
    specialize hNx n ((le_max_left _ _).trans hn_ge)
    have h_inv_n_le_k : (2 : ℝ)⁻¹ ^ n ≤ 2 * (2 : ℝ)⁻¹ ^ k := by
      rw [mul_comm, ← inv_mul_le_iff₀' (zero_lt_two' ℝ)]
      conv_lhs =>
        congr
        rw [← pow_one (2 : ℝ)⁻¹]
      rw [← pow_add, add_comm]
      exact pow_le_pow_of_le_one (one_div (2 : ℝ) ▸ one_half_pos.le)
        (inv_le_one_of_one_le₀ one_le_two)
        ((le_tsub_add.trans (add_le_add_right (le_max_right _ _) 1)).trans
          (add_le_add_right hn_ge 1))
    exact le_trans hNx.le h_inv_n_le_k
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    S : Nat → Set α := fun k => setOf fun x => LE.le (HPow.hPow (Inv.inv 2) k) (Di …
    hμS_le : ∀ (k : Nat), LE.le (μ (S k)) (HPow.hPow (Inv.inv 2) k)
    s : Set α := Filter.limsup S Filter.atTop
    hs : Eq s (Filter.limsup S Filter.atTop)
    hμs : Eq (μ s) 0
    h_tendsto : ∀ (x : α), Membership.mem (HasCompl.compl s) x → Filter.Tendsto (f …
    ⊢ And (StrictMono ns) (Filter.Eventually (fun x => Filter.Tendsto (fun i => f  …
  -/
  rw [ae_iff]
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    S : Nat → Set α := fun k => setOf fun x => LE.le (HPow.hPow (Inv.inv 2) k) (Di …
    hμS_le : ∀ (k : Nat), LE.le (μ (S k)) (HPow.hPow (Inv.inv 2) k)
    s : Set α := Filter.limsup S Filter.atTop
    hs : Eq s (Filter.limsup S Filter.atTop)
    hμs : Eq (μ s) 0
    h_tendsto : ∀ (x : α), Membership.mem (HasCompl.compl s) x → Filter.Tendsto (f …
    ⊢ And (StrictMono ns) (Eq (μ (setOf fun a => Not (Filter.Tendsto (fun i => f ( …
  -/
  refine ⟨ExistsSeqTendstoAe.seqTendstoAeSeq_strictMono hfg, measure_mono_null (fun x => ?_) hμs⟩
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    S : Nat → Set α := fun k => setOf fun x => LE.le (HPow.hPow (Inv.inv 2) k) (Di …
    hμS_le : ∀ (k : Nat), LE.le (μ (S k)) (HPow.hPow (Inv.inv 2) k)
    s : Set α := Filter.limsup S Filter.atTop
    hs : Eq s (Filter.limsup S Filter.atTop)
    hμs : Eq (μ s) 0
    h_tendsto : ∀ (x : α), Membership.mem (HasCompl.compl s) x → Filter.Tendsto (f …
    x : α
    ⊢ Membership.mem (setOf fun a => Not (Filter.Tendsto (fun i => f (ns i) a) Fil …
  -/
  rw [Set.mem_setOf_eq, ← @Classical.not_not (x ∈ s), not_imp_not]
  /-
    case h
    α : Type u_1
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MetricSpace E
    f : Nat → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    h_lt_ε_real : ∀ (ε : Real), LT.lt 0 ε → Exists fun k => LT.lt (HMul.hMul 2 (HP …
    ns : Nat → Nat := MeasureTheory.ExistsSeqTendstoAe.seqTendstoAeSeq hfg
    S : Nat → Set α := fun k => setOf fun x => LE.le (HPow.hPow (Inv.inv 2) k) (Di …
    hμS_le : ∀ (k : Nat), LE.le (μ (S k)) (HPow.hPow (Inv.inv 2) k)
    s : Set α := Filter.limsup S Filter.atTop
    hs : Eq s (Filter.limsup S Filter.atTop)
    hμs : Eq (μ s) 0
    h_tendsto : ∀ (x : α), Membership.mem (HasCompl.compl s) x → Filter.Tendsto (f …
    x : α
    ⊢ Not (Membership.mem s x) → Filter.Tendsto (fun i => f (ns i) x) Filter.atTop …
  -/
  exact h_tendsto x
  /-
    🎉 no goals
  -/


theorem TendstoInMeasure.exists_seq_tendstoInMeasure_atTop {u : Filter ι} [NeBot u]
    [IsCountablyGenerated u] {f : ι → α → E} {g : α → E} (hfg : TendstoInMeasure μ f u g) :
    ∃ ns : ℕ → ι, TendstoInMeasure μ (fun n => f (ns n)) atTop g := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MetricSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f u g
    ⊢ Exists fun ns => MeasureTheory.TendstoInMeasure μ (fun n => f (ns n)) Filter …
  -/
  obtain ⟨ns, h_tendsto_ns⟩ : ∃ ns : ℕ → ι, Tendsto ns atTop u := exists_seq_tendsto u
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MetricSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f u g
    ns : Nat → ι
    h_tendsto_ns : Filter.Tendsto ns Filter.atTop u
    ⊢ Exists fun ns => MeasureTheory.TendstoInMeasure μ (fun n => f (ns n)) Filter …
  -/
  exact ⟨ns, fun ε hε => (hfg ε hε).comp h_tendsto_ns⟩
  /-
    🎉 no goals
  -/


theorem TendstoInMeasure.exists_seq_tendsto_ae' {u : Filter ι} [NeBot u] [IsCountablyGenerated u]
    {f : ι → α → E} {g : α → E} (hfg : TendstoInMeasure μ f u g) :
    ∃ ns : ℕ → ι, ∀ᵐ x ∂μ, Tendsto (fun i => f (ns i) x) atTop (𝓝 (g x)) := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MetricSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f u g
    ⊢ Exists fun ns => Filter.Eventually (fun x => Filter.Tendsto (fun i => f (ns  …
  -/
  obtain ⟨ms, hms⟩ := hfg.exists_seq_tendstoInMeasure_atTop
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MetricSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f u g
    ms : Nat → ι
    hms : MeasureTheory.TendstoInMeasure μ (fun n => f (ms n)) Filter.atTop g
    ⊢ Exists fun ns => Filter.Eventually (fun x => Filter.Tendsto (fun i => f (ns  …
  -/
  obtain ⟨ns, -, hns⟩ := hms.exists_seq_tendsto_ae
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MetricSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hfg : MeasureTheory.TendstoInMeasure μ f u g
    ms : Nat → ι
    hms : MeasureTheory.TendstoInMeasure μ (fun n => f (ms n)) Filter.atTop g
    ns : Nat → Nat
    hns : Filter.Eventually (fun x => Filter.Tendsto (fun i => f (ms (ns i)) x) Fi …
    ⊢ Exists fun ns => Filter.Eventually (fun x => Filter.Tendsto (fun i => f (ns  …
  -/
  exact ⟨ms ∘ ns, hns⟩
  /-
    🎉 no goals
  -/


theorem TendstoInMeasure.aemeasurable {u : Filter ι} [NeBot u] [IsCountablyGenerated u]
    {f : ι → α → E} {g : α → E} (hf : ∀ n, AEMeasurable (f n) μ)
    (h_tendsto : TendstoInMeasure μ f u g) : AEMeasurable g μ := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace E
    inst✝³ : NormedAddCommGroup E
    inst✝² : BorelSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hf : ∀ (n : ι), AEMeasurable (f n) μ
    h_tendsto : MeasureTheory.TendstoInMeasure μ f u g
    ⊢ AEMeasurable g μ
  -/
  obtain ⟨ns, hns⟩ := h_tendsto.exists_seq_tendsto_ae'
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace E
    inst✝³ : NormedAddCommGroup E
    inst✝² : BorelSpace E
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → E
    g : α → E
    hf : ∀ (n : ι), AEMeasurable (f n) μ
    h_tendsto : MeasureTheory.TendstoInMeasure μ f u g
    ns : Nat → ι
    hns : Filter.Eventually (fun x => Filter.Tendsto (fun i => f (ns i) x) Filter. …
    ⊢ AEMeasurable g μ
  -/
  exact aemeasurable_of_tendsto_metrizable_ae atTop (fun n => hf (ns n)) hns
  /-
    🎉 no goals
  -/


/-- This lemma is superseded by `MeasureTheory.tendstoInMeasure_of_tendsto_eLpNorm` where we
allow `p = ∞` and only require `AEStronglyMeasurable`. -/
theorem tendstoInMeasure_of_tendsto_eLpNorm_of_stronglyMeasurable (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g)
    {l : Filter ι} (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) l (𝓝 0)) :
    TendstoInMeasure μ f l g := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    ⊢ MeasureTheory.TendstoInMeasure μ f l g
  -/
  intro ε hε
  replace hfg := ENNReal.Tendsto.const_mul
    (Tendsto.ennrpow_const p.toReal hfg) (Or.inr <| @ENNReal.ofReal_ne_top (1 / ε ^ p.toReal))
  simp only [mul_zero,
    ENNReal.zero_rpow_of_pos (ENNReal.toReal_pos hp_ne_zero hp_ne_top)] at hfg
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    l : Filter ι
    ε : Real
    hε : LT.lt 0 ε
    hfg : Filter.Tendsto (fun b => HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hP …
    ⊢ Filter.Tendsto (fun i => μ (setOf fun x => LE.le ε (Dist.dist (f i x) (g x)) …
  -/
  rw [ENNReal.tendsto_nhds_zero] at hfg ⊢
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    l : Filter ι
    ε : Real
    hε : LT.lt 0 ε
    hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
    ⊢ ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (μ (setOf …
  -/
  intro δ hδ
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    l : Filter ι
    ε : Real
    hε : LT.lt 0 ε
    hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
    δ : ENNReal
    hδ : GT.gt δ 0
    ⊢ Filter.Eventually (fun x => LE.le (μ (setOf fun x_1 => LE.le ε (Dist.dist (f …
  -/
  refine (hfg δ hδ).mono fun n hn => ?_
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    l : Filter ι
    ε : Real
    hε : LT.lt 0 ε
    hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
    δ : ENNReal
    hδ : GT.gt δ 0
    n : ι
    hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
    ⊢ LE.le (μ (setOf fun x => LE.le ε (Dist.dist (f n x) (g x)))) δ
  -/
  refine le_trans ?_ hn
  rw [ENNReal.ofReal_div_of_pos (Real.rpow_pos_of_pos hε _), ENNReal.ofReal_one, mul_comm,
    mul_one_div, ENNReal.le_div_iff_mul_le _ (Or.inl ENNReal.ofReal_ne_top), mul_comm]
    /-
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      l : Filter ι
      ε : Real
      hε : LT.lt 0 ε
      hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
      δ : ENNReal
      hδ : GT.gt δ 0
      n : ι
      hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
      ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HPow.hPow ε p.toReal)) (μ (setOf fun x =>  …
    -/
  · rw [← ENNReal.ofReal_rpow_of_pos hε]
    convert mul_meas_ge_le_pow_eLpNorm' μ hp_ne_zero hp_ne_top ((hf n).sub hg).aestronglyMeasurable
        (ENNReal.ofReal ε)
    /-
      case h.e'_3.h.e'_6.h.e'_6.h.e'_2.h.a
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      l : Filter ι
      ε : Real
      hε : LT.lt 0 ε
      hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
      δ : ENNReal
      hδ : GT.gt δ 0
      n : ι
      hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
      x✝ : α
      ⊢ Iff (LE.le ε (Dist.dist (f n x✝) (g x✝))) (LE.le (ENNReal.ofReal ε) ↑(NNNorm …
    -/
    rw [dist_eq_norm, ← ENNReal.ofReal_le_ofReal_iff (norm_nonneg _), ofReal_norm_eq_coe_nnnorm]
    /-
      case h.e'_3.h.e'_6.h.e'_6.h.e'_2.h.a
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      l : Filter ι
      ε : Real
      hε : LT.lt 0 ε
      hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
      δ : ENNReal
      hδ : GT.gt δ 0
      n : ι
      hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
      x✝ : α
      ⊢ Iff (LE.le (ENNReal.ofReal ε) ↑(NNNorm.nnnorm (HSub.hSub (f n x✝) (g x✝))))  …
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      l : Filter ι
      ε : Real
      hε : LT.lt 0 ε
      hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
      δ : ENNReal
      hδ : GT.gt δ 0
      n : ι
      hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
      ⊢ Or (Ne (ENNReal.ofReal (HPow.hPow ε p.toReal)) 0) (Ne (HPow.hPow (MeasureThe …
    -/
  · rw [Ne, ENNReal.ofReal_eq_zero, not_le]
    /-
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      hp_ne_zero : Ne p 0
      hp_ne_top : Ne p Top.top
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      l : Filter ι
      ε : Real
      hε : LT.lt 0 ε
      hfg : ∀ (ε_1 : ENNReal), GT.gt ε_1 0 → Filter.Eventually (fun x => LE.le (HMul …
      δ : ENNReal
      hδ : GT.gt δ 0
      n : ι
      hn : LE.le (HMul.hMul (ENNReal.ofReal (HDiv.hDiv 1 (HPow.hPow ε p.toReal))) (H …
      ⊢ Or (LT.lt 0 (HPow.hPow ε p.toReal)) (Ne (HPow.hPow (MeasureTheory.eLpNorm (H …
    -/
    exact Or.inl (Real.rpow_pos_of_pos hε _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias tendstoInMeasure_of_tendsto_snorm_of_stronglyMeasurable :=
  tendstoInMeasure_of_tendsto_eLpNorm_of_stronglyMeasurable


/-- This lemma is superseded by `MeasureTheory.tendstoInMeasure_of_tendsto_eLpNorm` where we
allow `p = ∞`. -/
theorem tendstoInMeasure_of_tendsto_eLpNorm_of_ne_top (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hg : AEStronglyMeasurable g μ) {l : Filter ι}
    (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) l (𝓝 0)) : TendstoInMeasure μ f l g := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    ⊢ MeasureTheory.TendstoInMeasure μ f l g
  -/
  refine TendstoInMeasure.congr (fun i => (hf i).ae_eq_mk.symm) hg.ae_eq_mk.symm ?_
  refine tendstoInMeasure_of_tendsto_eLpNorm_of_stronglyMeasurable
    hp_ne_zero hp_ne_top (fun i => (hf i).stronglyMeasurable_mk) hg.stronglyMeasurable_mk ?_
  have : (fun n => eLpNorm ((hf n).mk (f n) - hg.mk g) p μ) = fun n => eLpNorm (f n - g) p μ := by
    ext1 n; refine eLpNorm_congr_ae (EventuallyEq.sub (hf n).ae_eq_mk.symm hg.ae_eq_mk.symm)
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    this : Eq (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory.AEStrongly …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory.AES …
  -/
  rw [this]
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    this : Eq (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory.AEStrongly …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) l (n …
  -/
  exact hfg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias tendstoInMeasure_of_tendsto_snorm_of_ne_top := tendstoInMeasure_of_tendsto_eLpNorm_of_ne_top


/-- See also `MeasureTheory.tendstoInMeasure_of_tendsto_eLpNorm` which work for general
Lp-convergence for all `p ≠ 0`. -/
theorem tendstoInMeasure_of_tendsto_eLpNorm_top {E} [NormedAddCommGroup E] {f : ι → α → E}
    {g : α → E} {l : Filter ι} (hfg : Tendsto (fun n => eLpNorm (f n - g) ∞ μ) l (𝓝 0)) :
    TendstoInMeasure μ f l g := by
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) Top.t …
    ⊢ MeasureTheory.TendstoInMeasure μ f l g
  -/
  intro δ hδ
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) Top.t …
    δ : Real
    hδ : LT.lt 0 δ
    ⊢ Filter.Tendsto (fun i => μ (setOf fun x => LE.le δ (Dist.dist (f i x) (g x)) …
  -/
  simp only [eLpNorm_exponent_top, eLpNormEssSup] at hfg
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    hfg : Filter.Tendsto (fun n => essSup (fun x => ENorm.enorm (HSub.hSub (f n) g …
    ⊢ Filter.Tendsto (fun i => μ (setOf fun x => LE.le δ (Dist.dist (f i x) (g x)) …
  -/
  rw [ENNReal.tendsto_nhds_zero] at hfg ⊢
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    hfg : ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun x => LE.le (essSup ( …
    ⊢ ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun x => LE.le (μ (setOf fun …
  -/
  intro ε hε
  specialize hfg (ENNReal.ofReal δ / 2)
      (ENNReal.div_pos_iff.2 ⟨(ENNReal.ofReal_pos.2 hδ).ne.symm, ENNReal.two_ne_top⟩)
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    ε : ENNReal
    hε : GT.gt ε 0
    hfg : Filter.Eventually (fun x => LE.le (essSup (fun x_1 => ENorm.enorm (HSub. …
    ⊢ Filter.Eventually (fun x => LE.le (μ (setOf fun x_1 => LE.le δ (Dist.dist (f …
  -/
  refine hfg.mono fun n hn => ?_
  simp only [gt_iff_lt, zero_tsub, zero_le, zero_add, Set.mem_Icc,
    Pi.sub_apply] at *
  have : essSup (fun x : α => (‖f n x - g x‖₊ : ℝ≥0∞)) μ < ENNReal.ofReal δ :=
    lt_of_le_of_lt hn
      (ENNReal.half_lt_self (ENNReal.ofReal_pos.2 hδ).ne.symm ENNReal.ofReal_lt_top.ne)
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    ε : ENNReal
    hfg : Filter.Eventually (fun x => LE.le (essSup (fun x_1 => ENorm.enorm (HSub. …
    n : ι
    hn : LE.le (essSup (fun x => ENorm.enorm (HSub.hSub (f n x) (g x))) μ) (HDiv.h …
    hε : LT.lt 0 ε
    this : LT.lt (essSup (fun x => ↑(NNNorm.nnnorm (HSub.hSub (f n x) (g x)))) μ)  …
    ⊢ LE.le (μ (setOf fun x => LE.le δ (Dist.dist (f n x) (g x)))) ε
  -/
  refine ((le_of_eq ?_).trans (ae_lt_of_essSup_lt this).le).trans hε.le
  /-
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    ε : ENNReal
    hfg : Filter.Eventually (fun x => LE.le (essSup (fun x_1 => ENorm.enorm (HSub. …
    n : ι
    hn : LE.le (essSup (fun x => ENorm.enorm (HSub.hSub (f n x) (g x))) μ) (HDiv.h …
    hε : LT.lt 0 ε
    this : LT.lt (essSup (fun x => ↑(NNNorm.nnnorm (HSub.hSub (f n x) (g x)))) μ)  …
    ⊢ Eq (μ (setOf fun x => LE.le δ (Dist.dist (f n x) (g x)))) (μ (HasCompl.compl …
  -/
  congr with x
  simp only [ENNReal.ofReal_le_iff_le_toReal ENNReal.coe_lt_top.ne, ENNReal.coe_toReal, not_lt,
    coe_nnnorm, Set.mem_setOf_eq, Set.mem_compl_iff]
  /-
    case h.e_6.h.h
    α : Type u_1
    ι : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝ : NormedAddCommGroup E
    f : ι → α → E
    g : α → E
    l : Filter ι
    δ : Real
    hδ : LT.lt 0 δ
    ε : ENNReal
    hfg : Filter.Eventually (fun x => LE.le (essSup (fun x_1 => ENorm.enorm (HSub. …
    n : ι
    hn : LE.le (essSup (fun x => ENorm.enorm (HSub.hSub (f n x) (g x))) μ) (HDiv.h …
    hε : LT.lt 0 ε
    this : LT.lt (essSup (fun x => ↑(NNNorm.nnnorm (HSub.hSub (f n x) (g x)))) μ)  …
    x : α
    ⊢ Iff (LE.le δ (Dist.dist (f n x) (g x))) (LE.le δ (Norm.norm (HSub.hSub (f n  …
  -/
  rw [← dist_eq_norm (f n x) (g x)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias tendstoInMeasure_of_tendsto_snorm_top := tendstoInMeasure_of_tendsto_eLpNorm_top


/-- Convergence in Lp implies convergence in measure. -/
theorem tendstoInMeasure_of_tendsto_eLpNorm {l : Filter ι} (hp_ne_zero : p ≠ 0)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hg : AEStronglyMeasurable g μ)
    (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) l (𝓝 0)) : TendstoInMeasure μ f l g := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    f : ι → α → E
    g : α → E
    l : Filter ι
    hp_ne_zero : Ne p 0
    hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    ⊢ MeasureTheory.TendstoInMeasure μ f l g
  -/
  by_cases hp_ne_top : p = ∞
    /-
      case pos
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      l : Filter ι
      hp_ne_zero : Ne p 0
      hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      hp_ne_top : Eq p Top.top
      ⊢ MeasureTheory.TendstoInMeasure μ f l g
    -/
  · subst hp_ne_top
    /-
      case pos
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : ι → α → E
      g : α → E
      l : Filter ι
      hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      hp_ne_zero : Ne Top.top 0
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) Top.t …
      ⊢ MeasureTheory.TendstoInMeasure μ f l g
    -/
    exact tendstoInMeasure_of_tendsto_eLpNorm_top hfg
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      f : ι → α → E
      g : α → E
      l : Filter ι
      hp_ne_zero : Ne p 0
      hf : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (f n) μ
      hg : MeasureTheory.AEStronglyMeasurable g μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      hp_ne_top : Not (Eq p Top.top)
      ⊢ MeasureTheory.TendstoInMeasure μ f l g
    -/
  · exact tendstoInMeasure_of_tendsto_eLpNorm_of_ne_top hp_ne_zero hp_ne_top hf hg hfg
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias tendstoInMeasure_of_tendsto_snorm := tendstoInMeasure_of_tendsto_eLpNorm


/-- Convergence in Lp implies convergence in measure. -/
theorem tendstoInMeasure_of_tendsto_Lp [hp : Fact (1 ≤ p)] {f : ι → Lp E p μ} {g : Lp E p μ}
    {l : Filter ι} (hfg : Tendsto f l (𝓝 g)) : TendstoInMeasure μ (fun n => f n) l g :=
  tendstoInMeasure_of_tendsto_eLpNorm (zero_lt_one.trans_le hp.elim).ne.symm
    (fun _ => Lp.aestronglyMeasurable _) (Lp.aestronglyMeasurable _)
    ((Lp.tendsto_Lp_iff_tendsto_ℒp' _ _).mp hfg)


