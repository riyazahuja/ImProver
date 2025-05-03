/-- Uniform integrability in the measure theory sense.

A sequence of functions `f` is said to be uniformly integrable if for all `ε > 0`, there exists
some `δ > 0` such that for all sets `s` with measure less than `δ`, the Lp-norm of `f i`
restricted on `s` is less than `ε`.

Uniform integrability is also known as uniformly absolutely continuous integrals. -/
def UnifIntegrable {_ : MeasurableSpace α} (f : ι → α → β) (p : ℝ≥0∞) (μ : Measure α) : Prop :=
  ∀ ⦃ε : ℝ⦄ (_ : 0 < ε), ∃ (δ : ℝ) (_ : 0 < δ), ∀ i s,
    MeasurableSet s → μ s ≤ ENNReal.ofReal δ → eLpNorm (s.indicator (f i)) p μ ≤ ENNReal.ofReal ε


/-- In probability theory, a family of measurable functions is uniformly integrable if it is
uniformly integrable in the measure theory sense and is uniformly bounded. -/
def UniformIntegrable {_ : MeasurableSpace α} (f : ι → α → β) (p : ℝ≥0∞) (μ : Measure α) : Prop :=
  (∀ i, AEStronglyMeasurable (f i) μ) ∧ UnifIntegrable f p μ ∧ ∃ C : ℝ≥0, ∀ i, eLpNorm (f i) p μ ≤ C


protected theorem aeStronglyMeasurable {f : ι → α → β} {p : ℝ≥0∞} (hf : UniformIntegrable f p μ)
    (i : ι) : AEStronglyMeasurable (f i) μ :=
  hf.1 i


protected theorem unifIntegrable {f : ι → α → β} {p : ℝ≥0∞} (hf : UniformIntegrable f p μ) :
    UnifIntegrable f p μ :=
  hf.2.1


protected theorem memℒp {f : ι → α → β} {p : ℝ≥0∞} (hf : UniformIntegrable f p μ) (i : ι) :
    Memℒp (f i) p μ :=
  ⟨hf.1 i,
    let ⟨_, _, hC⟩ := hf.2
    lt_of_le_of_lt (hC i) ENNReal.coe_lt_top⟩


protected theorem add (hf : UnifIntegrable f p μ) (hg : UnifIntegrable g p μ) (hp : 1 ≤ p)
    (hf_meas : ∀ i, AEStronglyMeasurable (f i) μ) (hg_meas : ∀ i, AEStronglyMeasurable (g i) μ) :
    UnifIntegrable (f + g) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ MeasureTheory.UnifIntegrable (HAdd.hAdd f g) p μ
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  have hε2 : 0 < ε / 2 := half_pos hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ₁, hδ₁_pos, hfδ₁⟩ := hf hε2
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    δ₁ : Real
    hδ₁_pos : LT.lt 0 δ₁
    hfδ₁ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₁ …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ₂, hδ₂_pos, hgδ₂⟩ := hg hε2
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    δ₁ : Real
    hδ₁_pos : LT.lt 0 δ₁
    hfδ₁ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₁ …
    δ₂ : Real
    hδ₂_pos : LT.lt 0 δ₂
    hgδ₂ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂ …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  refine ⟨min δ₁ δ₂, lt_min hδ₁_pos hδ₂_pos, fun i s hs hμs => ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    δ₁ : Real
    hδ₁_pos : LT.lt 0 δ₁
    hfδ₁ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₁ …
    δ₂ : Real
    hδ₂_pos : LT.lt 0 δ₂
    hgδ₂ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂ …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (HAdd.hAdd f g i)) p μ) (ENNReal.o …
  -/
  simp_rw [Pi.add_apply, Set.indicator_add']
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    δ₁ : Real
    hδ₁_pos : LT.lt 0 δ₁
    hfδ₁ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₁ …
    δ₂ : Real
    hδ₂_pos : LT.lt 0 δ₂
    hgδ₂ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂ …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
    ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd (s.indicator (f i)) (s.indicator (g  …
  -/
  refine (eLpNorm_add_le ((hf_meas i).indicator hs) ((hg_meas i).indicator hs) hp).trans ?_
  have hε_halves : ENNReal.ofReal ε = ENNReal.ofReal (ε / 2) + ENNReal.ofReal (ε / 2) := by
    rw [← ENNReal.ofReal_add hε2.le hε2.le, add_halves]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : Real
    hε : LT.lt 0 ε
    hε2 : LT.lt 0 (HDiv.hDiv ε 2)
    δ₁ : Real
    hδ₁_pos : LT.lt 0 δ₁
    hfδ₁ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₁ …
    δ₂ : Real
    hδ₂_pos : LT.lt 0 δ₂
    hgδ₂ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂ …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
    hε_halves : Eq (ENNReal.ofReal ε) (HAdd.hAdd (ENNReal.ofReal (HDiv.hDiv ε 2))  …
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (MeasureThe …
  -/
  rw [hε_halves]
  exact add_le_add (hfδ₁ i s hs (hμs.trans (ENNReal.ofReal_le_ofReal (min_le_left _ _))))
    (hgδ₂ i s hs (hμs.trans (ENNReal.ofReal_le_ofReal (min_le_right _ _))))


protected theorem neg (hf : UnifIntegrable f p μ) : UnifIntegrable (-f) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    ⊢ MeasureTheory.UnifIntegrable (Neg.neg f) p μ
  -/
  simp_rw [UnifIntegrable, Pi.neg_apply, Set.indicator_neg', eLpNorm_neg]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    ⊢ ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun δ => Exists fun h => ∀ (i : ι) (s : Set …
  -/
  exact hf
  /-
    🎉 no goals
  -/


protected theorem sub (hf : UnifIntegrable f p μ) (hg : UnifIntegrable g p μ) (hp : 1 ≤ p)
    (hf_meas : ∀ i, AEStronglyMeasurable (f i) μ) (hg_meas : ∀ i, AEStronglyMeasurable (g i) μ) :
    UnifIntegrable (f - g) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ MeasureTheory.UnifIntegrable (HSub.hSub f g) p μ
  -/
  rw [sub_eq_add_neg]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hg : MeasureTheory.UnifIntegrable g p μ
    hp : LE.le 1 p
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ MeasureTheory.UnifIntegrable (HAdd.hAdd f (Neg.neg g)) p μ
  -/
  exact hf.add hg.neg hp hf_meas fun i => (hg_meas i).neg
  /-
    🎉 no goals
  -/


protected theorem ae_eq (hf : UnifIntegrable f p μ) (hfg : ∀ n, f n =ᵐ[μ] g n) :
    UnifIntegrable g p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ⊢ MeasureTheory.UnifIntegrable g p μ
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ, hδ_pos, hfδ⟩ := hf hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hfδ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ)  …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  refine ⟨δ, hδ_pos, fun n s hs hμs => (le_of_eq <| eLpNorm_congr_ae ?_).trans (hfδ n s hs hμs)⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hfδ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ)  …
    n : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator (g n)) (s.indicator (f n))
  -/
  filter_upwards [hfg n] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hfδ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ)  …
    n : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    x : α
    hx : Eq (f n x) (g n x)
    ⊢ Eq (s.indicator (g n) x) (s.indicator (f n) x)
  -/
  simp_rw [Set.indicator_apply, hx]
  /-
    🎉 no goals
  -/


/-- Uniform integrability is preserved by restriction of the functions to a set. -/
protected theorem indicator (hf : UnifIntegrable f p μ) (E : Set α) :
    UnifIntegrable (fun i => E.indicator (f i)) p μ := fun ε hε ↦ by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    E : Set α
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ, hδ_pos, hε⟩ := hf hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    E : Set α
    ε : Real
    hε✝ : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hε : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  refine ⟨δ, hδ_pos, fun i s hs hμs ↦ ?_⟩
  calc
    eLpNorm (s.indicator (E.indicator (f i))) p μ
      = eLpNorm (E.indicator (s.indicator (f i))) p μ := by
      simp only [indicator_indicator, inter_comm]
    _ ≤ eLpNorm (s.indicator (f i)) p μ := eLpNorm_indicator_le _
    _ ≤ ENNReal.ofReal ε := hε _ _ hs hμs


/-- Uniform integrability is preserved by restriction of the measure to a set. -/
protected theorem restrict (hf : UnifIntegrable f p μ) (E : Set α) :
    UnifIntegrable f p (μ.restrict E) := fun ε hε ↦ by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    E : Set α
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ, hδ_pos, hδε⟩ := hf hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    E : Set α
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hδε : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ)  …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  refine ⟨δ, hδ_pos, fun i s hs hμs ↦ ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifIntegrable f p μ
    E : Set α
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hδε : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ)  …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le ((μ.restrict E) s) (ENNReal.ofReal δ)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p (μ.restrict E)) (ENNReal. …
  -/
  rw [μ.restrict_apply hs, ← measure_toMeasurable] at hμs
  calc
    eLpNorm (indicator s (f i)) p (μ.restrict E) = eLpNorm (f i) p (μ.restrict (s ∩ E)) := by
      rw [eLpNorm_indicator_eq_eLpNorm_restrict hs, μ.restrict_restrict hs]
    _ ≤ eLpNorm (f i) p (μ.restrict (toMeasurable μ (s ∩ E))) :=
      eLpNorm_mono_measure _ <| Measure.restrict_mono (subset_toMeasurable _ _) le_rfl
    _ = eLpNorm (indicator (toMeasurable μ (s ∩ E)) (f i)) p μ :=
      (eLpNorm_indicator_eq_eLpNorm_restrict (measurableSet_toMeasurable _ _)).symm
    _ ≤ ENNReal.ofReal ε := hδε i _ (measurableSet_toMeasurable _ _) hμs


theorem unifIntegrable_zero_meas [MeasurableSpace α] {p : ℝ≥0∞} {f : ι → α → β} :
    UnifIntegrable f p (0 : Measure α) :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              ι : Type u_3
                                              inst✝¹ : NormedAddCommGroup β
                                              inst✝ : MeasurableSpace α
                                              p : ENNReal
                                              f : ι → α → β
                                              ε : Real
                                              x✝² : LT.lt 0 ε
                                              i : ι
                                              s : Set α
                                              x✝¹ : MeasurableSet s
                                              x✝ : LE.le (0 s) (ENNReal.ofReal 1)
                                              ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p 0) (ENNReal.ofReal ε)
                                            -/
  fun ε _ => ⟨1, one_pos, fun i s _ _ => by simp⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem unifIntegrable_congr_ae {p : ℝ≥0∞} {f g : ι → α → β} (hfg : ∀ n, f n =ᵐ[μ] g n) :
    UnifIntegrable f p μ ↔ UnifIntegrable g p μ :=
  ⟨fun hf => hf.ae_eq hfg, fun hg => hg.ae_eq fun n => (hfg n).symm⟩


theorem tendsto_indicator_ge (f : α → β) (x : α) :
    Tendsto (fun M : ℕ => { x | (M : ℝ) ≤ ‖f x‖₊ }.indicator f x) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    ⊢ Filter.Tendsto (fun M => (setOf fun x => LE.le ↑M ↑(NNNorm.nnnorm (f x))).in …
  -/
  refine tendsto_atTop_of_eventually_const (i₀ := Nat.ceil (‖f x‖₊ : ℝ) + 1) fun n hn => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    n : Nat
    hn : GE.ge n (HAdd.hAdd (Nat.ceil ↑(NNNorm.nnnorm (f x))) 1)
    ⊢ Eq ((setOf fun x => LE.le ↑n ↑(NNNorm.nnnorm (f x))).indicator f x) 0
  -/
  rw [Set.indicator_of_not_mem]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    n : Nat
    hn : GE.ge n (HAdd.hAdd (Nat.ceil ↑(NNNorm.nnnorm (f x))) 1)
    ⊢ Not (Membership.mem (setOf fun x => LE.le ↑n ↑(NNNorm.nnnorm (f x))) x)
  -/
  simp only [not_le, Set.mem_setOf_eq]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    n : Nat
    hn : GE.ge n (HAdd.hAdd (Nat.ceil ↑(NNNorm.nnnorm (f x))) 1)
    ⊢ LT.lt ↑(NNNorm.nnnorm (f x)) ↑n
  -/
  refine lt_of_le_of_lt (Nat.le_ceil _) ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    n : Nat
    hn : GE.ge n (HAdd.hAdd (Nat.ceil ↑(NNNorm.nnnorm (f x))) 1)
    ⊢ LT.lt ↑(Nat.ceil ↑(NNNorm.nnnorm (f x))) ↑n
  -/
  refine lt_of_lt_of_le (lt_add_one _) ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    f : α → β
    x : α
    n : Nat
    hn : GE.ge n (HAdd.hAdd (Nat.ceil ↑(NNNorm.nnnorm (f x))) 1)
    ⊢ LE.le (HAdd.hAdd (↑(Nat.ceil ↑(NNNorm.nnnorm (f x)))) 1) ↑n
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- This lemma is weaker than `MeasureTheory.Memℒp.integral_indicator_norm_ge_nonneg_le`
as the latter provides `0 ≤ M` and does not require the measurability of `f`. -/
theorem Memℒp.integral_indicator_norm_ge_le (hf : Memℒp f 1 μ) (hmeas : StronglyMeasurable f)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℝ, (∫⁻ x, ‖{ x | M ≤ ‖f x‖₊ }.indicator f x‖₊ ∂μ) ≤ ENNReal.ofReal ε := by
  have htendsto :
      ∀ᵐ x ∂μ, Tendsto (fun M : ℕ => { x | (M : ℝ) ≤ ‖f x‖₊ }.indicator f x) atTop (𝓝 0) :=
    univ_mem' (id fun x => tendsto_indicator_ge f x)
  have hmeas : ∀ M : ℕ, AEStronglyMeasurable ({ x | (M : ℝ) ≤ ‖f x‖₊ }.indicator f) μ := by
    intro M
    apply hf.1.indicator
    apply StronglyMeasurable.measurableSet_le stronglyMeasurable_const
      hmeas.nnnorm.measurable.coe_nnreal_real.stronglyMeasurable
  have hbound : HasFiniteIntegral (fun x => ‖f x‖) μ := by
    rw [memℒp_one_iff_integrable] at hf
    exact hf.norm.2
  have : Tendsto (fun n : ℕ ↦ ∫⁻ a, ENNReal.ofReal ‖{ x | n ≤ ‖f x‖₊ }.indicator f a - 0‖ ∂μ)
      atTop (𝓝 0) := by
    refine tendsto_lintegral_norm_of_dominated_convergence hmeas hbound ?_ htendsto
    refine fun n => univ_mem' (id fun x => ?_)
    by_cases hx : (n : ℝ) ≤ ‖f x‖
    · dsimp
      rwa [Set.indicator_of_mem]
    · dsimp
      rw [Set.indicator_of_not_mem, norm_zero]
      · exact norm_nonneg _
      · assumption
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => ENNReal.ofR …
    ⊢ Exists fun M => LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (( …
  -/
  rw [ENNReal.tendsto_atTop_zero] at this
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → L …
    ⊢ Exists fun M => LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (( …
  -/
  obtain ⟨M, hM⟩ := this (ENNReal.ofReal ε) (ENNReal.ofReal_pos.2 hε)
  simp only [zero_tsub, zero_le, sub_zero, zero_add, coe_nnnorm,
    Set.mem_Icc] at hM
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → L …
    M : Nat
    hM : ∀ (n : Nat), GE.ge n M → LE.le (MeasureTheory.lintegral μ fun a => ENNRea …
    ⊢ Exists fun M => LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (( …
  -/
  refine ⟨M, ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → L …
    M : Nat
    hM : ∀ (n : Nat), GE.ge n M → LE.le (MeasureTheory.lintegral μ fun a => ENNRea …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x => L …
  -/
  convert hM M le_rfl
  /-
    case h.e'_3.h.e'_4.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → L …
    M : Nat
    hM : ∀ (n : Nat), GE.ge n M → LE.le (MeasureTheory.lintegral μ fun a => ENNRea …
    x✝ : α
    ⊢ Eq (↑(NNNorm.nnnorm ((setOf fun x => LE.le ↑M ↑(NNNorm.nnnorm (f x))).indica …
  -/
  simp only [coe_nnnorm, ENNReal.ofReal_eq_coe_nnreal (norm_nonneg _)]
  /-
    case h.e'_3.h.e'_4.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hmeas✝ : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    htendsto : Filter.Eventually (fun x => Filter.Tendsto (fun M => (setOf fun x = …
    hmeas : ∀ (M : Nat), MeasureTheory.AEStronglyMeasurable ((setOf fun x => LE.le …
    hbound : MeasureTheory.HasFiniteIntegral (fun x => Norm.norm (f x)) μ
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → L …
    M : Nat
    hM : ∀ (n : Nat), GE.ge n M → LE.le (MeasureTheory.lintegral μ fun a => ENNRea …
    x✝ : α
    ⊢ Eq ↑(NNNorm.nnnorm ((setOf fun x => LE.le (↑M) (Norm.norm (f x))).indicator  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- This lemma is superseded by `MeasureTheory.Memℒp.integral_indicator_norm_ge_nonneg_le`
which does not require measurability. -/
theorem Memℒp.integral_indicator_norm_ge_nonneg_le_of_meas (hf : Memℒp f 1 μ)
    (hmeas : StronglyMeasurable f) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℝ, 0 ≤ M ∧ (∫⁻ x, ‖{ x | M ≤ ‖f x‖₊ }.indicator f x‖₊ ∂μ) ≤ ENNReal.ofReal ε :=
  let ⟨M, hM⟩ := hf.integral_indicator_norm_ge_le hmeas hε
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   m : MeasurableSpace α
                                   μ : MeasureTheory.Measure α
                                   inst✝ : NormedAddCommGroup β
                                   f : α → β
                                   hf : MeasureTheory.Memℒp f 1 μ
                                   hmeas : MeasureTheory.StronglyMeasurable f
                                   ε : Real
                                   hε : LT.lt 0 ε
                                   M : Real
                                   hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
                                   ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x => L …
                                 -/
  ⟨max M 0, le_max_right _ _, by simpa⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem Memℒp.integral_indicator_norm_ge_nonneg_le (hf : Memℒp f 1 μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℝ, 0 ≤ M ∧ (∫⁻ x, ‖{ x | M ≤ ‖f x‖₊ }.indicator f x‖₊ ∂μ) ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun M => And (LE.le 0 M) (LE.le (MeasureTheory.lintegral μ fun x => ↑ …
  -/
  have hf_mk : Memℒp (hf.1.mk f) 1 μ := (memℒp_congr_ae hf.1.ae_eq_mk).mp hf
  obtain ⟨M, hM_pos, hfM⟩ :=
    hf_mk.integral_indicator_norm_ge_nonneg_le_of_meas hf.1.stronglyMeasurable_mk hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hf_mk : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk f ⋯) 1 μ
    M : Real
    hM_pos : LE.le 0 M
    hfM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x  …
    ⊢ Exists fun M => And (LE.le 0 M) (LE.le (MeasureTheory.lintegral μ fun x => ↑ …
  -/
  refine ⟨M, hM_pos, (le_of_eq ?_).trans hfM⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hf_mk : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk f ⋯) 1 μ
    M : Real
    hM_pos : LE.le 0 M
    hfM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x  …
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x => LE.l …
  -/
  refine lintegral_congr_ae ?_
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hf_mk : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk f ⋯) 1 μ
    M : Real
    hM_pos : LE.le 0 M
    hfM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ↑(NNNorm.nnnorm ((setOf fun x => …
  -/
  filter_upwards [hf.1.ae_eq_mk] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ε : Real
    hε : LT.lt 0 ε
    hf_mk : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk f ⋯) 1 μ
    M : Real
    hM_pos : LE.le 0 M
    hfM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x  …
    x : α
    hx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
    ⊢ Eq ↑(NNNorm.nnnorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))).indicato …
  -/
  simp only [Set.indicator_apply, coe_nnnorm, Set.mem_setOf_eq, ENNReal.coe_inj, hx.symm]
  /-
    🎉 no goals
  -/


theorem Memℒp.eLpNormEssSup_indicator_norm_ge_eq_zero (hf : Memℒp f ∞ μ)
    (hmeas : StronglyMeasurable f) :
    ∃ M : ℝ, eLpNormEssSup ({ x | M ≤ ‖f x‖₊ }.indicator f) μ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f Top.top μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ⊢ Exists fun M => Eq (MeasureTheory.eLpNormEssSup ((setOf fun x => LE.le M ↑(N …
  -/
  have hbdd : eLpNormEssSup f μ < ∞ := hf.eLpNorm_lt_top
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f Top.top μ
    hmeas : MeasureTheory.StronglyMeasurable f
    hbdd : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
    ⊢ Exists fun M => Eq (MeasureTheory.eLpNormEssSup ((setOf fun x => LE.le M ↑(N …
  -/
  refine ⟨(eLpNorm f ∞ μ + 1).toReal, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f Top.top μ
    hmeas : MeasureTheory.StronglyMeasurable f
    hbdd : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
    ⊢ Eq (MeasureTheory.eLpNormEssSup ((setOf fun x => LE.le (HAdd.hAdd (MeasureTh …
  -/
  rw [eLpNormEssSup_indicator_eq_eLpNormEssSup_restrict]
  · have : μ.restrict { x : α | (eLpNorm f ⊤ μ + 1).toReal ≤ ‖f x‖₊ } = 0 := by
      simp only [coe_nnnorm, eLpNorm_exponent_top, Measure.restrict_eq_zero]
      have : { x : α | (eLpNormEssSup f μ + 1).toReal ≤ ‖f x‖ } ⊆
          { x : α | eLpNormEssSup f μ < ‖f x‖₊ } := by
        intro x hx
        rw [Set.mem_setOf_eq, ← ENNReal.toReal_lt_toReal hbdd.ne ENNReal.coe_lt_top.ne,
          ENNReal.coe_toReal, coe_nnnorm]
        refine lt_of_lt_of_le ?_ hx
        rw [ENNReal.toReal_lt_toReal hbdd.ne]
        · exact ENNReal.lt_add_right hbdd.ne one_ne_zero
        · exact (ENNReal.add_lt_top.2 ⟨hbdd, ENNReal.one_lt_top⟩).ne
      rw [← nonpos_iff_eq_zero]
      refine (measure_mono this).trans ?_
      have hle := coe_nnnorm_ae_le_eLpNormEssSup f μ
      simp_rw [ae_iff, not_le] at hle
      exact nonpos_iff_eq_zero.2 hle
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      hf : MeasureTheory.Memℒp f Top.top μ
      hmeas : MeasureTheory.StronglyMeasurable f
      hbdd : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
      this : Eq (μ.restrict (setOf fun x => LE.le (HAdd.hAdd (MeasureTheory.eLpNorm  …
      ⊢ Eq (MeasureTheory.eLpNormEssSup f (μ.restrict (setOf fun x => LE.le (HAdd.hA …
    -/
    rw [this, eLpNormEssSup_measure_zero]
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f Top.top μ
    hmeas : MeasureTheory.StronglyMeasurable f
    hbdd : LT.lt (MeasureTheory.eLpNormEssSup f μ) Top.top
    ⊢ MeasurableSet (setOf fun x => LE.le (HAdd.hAdd (MeasureTheory.eLpNorm f Top. …
  -/
  exact measurableSet_le measurable_const hmeas.nnnorm.measurable.subtype_coe
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snormEssSup_indicator_norm_ge_eq_zero := Memℒp.eLpNormEssSup_indicator_norm_ge_eq_zero

/- This lemma is slightly weaker than `MeasureTheory.Memℒp.eLpNorm_indicator_norm_ge_pos_le` as the
latter provides `0 < M`. -/

theorem Memℒp.eLpNorm_indicator_norm_ge_le (hf : Memℒp f p μ) (hmeas : StronglyMeasurable f) {ε : ℝ}
    (hε : 0 < ε) : ∃ M : ℝ, eLpNorm ({ x | M ≤ ‖f x‖₊ }.indicator f) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
  -/
  by_cases hp_ne_zero : p = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Eq p 0
      ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
    -/
  · refine ⟨1, hp_ne_zero.symm ▸ ?_⟩
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le 1 ↑(NNNorm.nnnorm (f x)) …
    -/
    simp [eLpNorm_exponent_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
  -/
  by_cases hp_ne_top : p = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Not (Eq p 0)
      hp_ne_top : Eq p Top.top
      ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
    -/
  · subst hp_ne_top
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hf : MeasureTheory.Memℒp f Top.top μ
      hp_ne_zero : Not (Eq Top.top 0)
      ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
    -/
    obtain ⟨M, hM⟩ := hf.eLpNormEssSup_indicator_norm_ge_eq_zero hmeas
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hf : MeasureTheory.Memℒp f Top.top μ
      hp_ne_zero : Not (Eq Top.top 0)
      M : Real
      hM : Eq (MeasureTheory.eLpNormEssSup ((setOf fun x => LE.le M ↑(NNNorm.nnnorm  …
      ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
    -/
    refine ⟨M, ?_⟩
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hf : MeasureTheory.Memℒp f Top.top μ
      hp_ne_zero : Not (Eq Top.top 0)
      M : Real
      hM : Eq (MeasureTheory.eLpNormEssSup ((setOf fun x => LE.le M ↑(NNNorm.nnnorm  …
      ⊢ LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x)) …
    -/
    simp only [eLpNorm_exponent_top, hM, zero_le]
    /-
      🎉 no goals
    -/
  obtain ⟨M, hM', hM⟩ := Memℒp.integral_indicator_norm_ge_nonneg_le
    (μ := μ) (hf.norm_rpow hp_ne_zero hp_ne_top) (Real.rpow_pos_of_pos hε p.toReal)
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    ⊢ Exists fun M => LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNo …
  -/
  refine ⟨M ^ (1 / p.toReal), ?_⟩
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    ⊢ LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv  …
  -/
  rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_ne_zero hp_ne_top, ← ENNReal.rpow_one (ENNReal.ofReal ε)]
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnn …
  -/
  conv_rhs => rw [← mul_one_div_cancel (ENNReal.toReal_pos hp_ne_zero hp_ne_top).ne.symm]
  rw [ENNReal.rpow_mul,
    ENNReal.rpow_le_rpow_iff (one_div_pos.2 <| ENNReal.toReal_pos hp_ne_zero hp_ne_top),
    ENNReal.ofReal_rpow_of_pos hε]
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm ((setOf …
  -/
  convert hM
  /-
    case h.e'_3.h.e'_4.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    x✝ : α
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm ((setOf fun x => LE.le (HPow.hPow M (HDiv.hDi …
  -/
  rename_i x
  rw [← ENNReal.coe_rpow_of_nonneg _ ENNReal.toReal_nonneg, nnnorm_indicator_eq_indicator_nnnorm,
    nnnorm_indicator_eq_indicator_nnnorm]
  have hiff : M ^ (1 / p.toReal) ≤ ‖f x‖₊ ↔ M ≤ ‖‖f x‖ ^ p.toReal‖₊ := by
    rw [coe_nnnorm, coe_nnnorm, Real.norm_rpow_of_nonneg (norm_nonneg _), norm_norm,
      ← Real.rpow_le_rpow_iff hM' (Real.rpow_nonneg (norm_nonneg _) _)
        (one_div_pos.2 <| ENNReal.toReal_pos hp_ne_zero hp_ne_top), ← Real.rpow_mul (norm_nonneg _),
      mul_one_div_cancel (ENNReal.toReal_pos hp_ne_zero hp_ne_top).ne.symm, Real.rpow_one]
  /-
    case h.e'_3.h.e'_4.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    hp_ne_zero : Not (Eq p 0)
    hp_ne_top : Not (Eq p Top.top)
    M : Real
    hM' : LE.le 0 M
    hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
    x : α
    hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
    ⊢ Eq ↑(HPow.hPow ((setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑ …
  -/
  by_cases hx : x ∈ { x : α | M ^ (1 / p.toReal) ≤ ‖f x‖₊ }
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Not (Eq p 0)
      hp_ne_top : Not (Eq p Top.top)
      M : Real
      hM' : LE.le 0 M
      hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
      x : α
      hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
      hx : Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) …
      ⊢ Eq ↑(HPow.hPow ((setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑ …
    -/
  · rw [Set.indicator_of_mem hx, Set.indicator_of_mem, Real.nnnorm_of_nonneg]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        hp_ne_zero : Not (Eq p 0)
        hp_ne_top : Not (Eq p Top.top)
        M : Real
        hM' : LE.le 0 M
        hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
        x : α
        hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
        hx : Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) …
        ⊢ Eq ↑(HPow.hPow (NNNorm.nnnorm (f x)) p.toReal) ↑⟨HPow.hPow (Norm.norm (f x)) …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Not (Eq p 0)
      hp_ne_top : Not (Eq p Top.top)
      M : Real
      hM' : LE.le 0 M
      hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
      x : α
      hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
      hx : Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) …
      ⊢ Membership.mem (setOf fun x => LE.le M ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm …
    -/
    rw [Set.mem_setOf_eq]
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Not (Eq p 0)
      hp_ne_top : Not (Eq p Top.top)
      M : Real
      hM' : LE.le 0 M
      hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
      x : α
      hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
      hx : Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) …
      ⊢ LE.le M ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) p.toReal))
    -/
    rwa [← hiff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      hp_ne_zero : Not (Eq p 0)
      hp_ne_top : Not (Eq p Top.top)
      M : Real
      hM' : LE.le 0 M
      hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
      x : α
      hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
      hx : Not (Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toR …
      ⊢ Eq ↑(HPow.hPow ((setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑ …
    -/
  · rw [Set.indicator_of_not_mem hx, Set.indicator_of_not_mem]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        hp_ne_zero : Not (Eq p 0)
        hp_ne_top : Not (Eq p Top.top)
        M : Real
        hM' : LE.le 0 M
        hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
        x : α
        hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
        hx : Not (Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toR …
        ⊢ Eq ↑(HPow.hPow 0 p.toReal) ↑0
      -/
    · simp [(ENNReal.toReal_pos hp_ne_zero hp_ne_top).ne.symm]
      /-
        🎉 no goals
      -/
      /-
        case neg.h
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        hp_ne_zero : Not (Eq p 0)
        hp_ne_top : Not (Eq p Top.top)
        M : Real
        hM' : LE.le 0 M
        hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
        x : α
        hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
        hx : Not (Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toR …
        ⊢ Not (Membership.mem (setOf fun x => LE.le M ↑(NNNorm.nnnorm (HPow.hPow (Norm …
      -/
    · rw [Set.mem_setOf_eq]
      /-
        case neg.h
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        hp_ne_zero : Not (Eq p 0)
        hp_ne_top : Not (Eq p Top.top)
        M : Real
        hM' : LE.le 0 M
        hM : LE.le (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((setOf fun x = …
        x : α
        hiff : Iff (LE.le (HPow.hPow M (HDiv.hDiv 1 p.toReal)) ↑(NNNorm.nnnorm (f x))) …
        hx : Not (Membership.mem (setOf fun x => LE.le (HPow.hPow M (HDiv.hDiv 1 p.toR …
        ⊢ Not (LE.le M ↑(NNNorm.nnnorm (HPow.hPow (Norm.norm (f x)) p.toReal)))
      -/
      rwa [← hiff]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_indicator_norm_ge_le := Memℒp.eLpNorm_indicator_norm_ge_le


/-- This lemma implies that a single function is uniformly integrable (in the probability sense). -/
theorem Memℒp.eLpNorm_indicator_norm_ge_pos_le (hf : Memℒp f p μ) (hmeas : StronglyMeasurable f)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℝ, 0 < M ∧ eLpNorm ({ x | M ≤ ‖f x‖₊ }.indicator f) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun M => And (LT.lt 0 M) (LE.le (MeasureTheory.eLpNorm ((setOf fun x  …
  -/
  obtain ⟨M, hM⟩ := hf.eLpNorm_indicator_norm_ge_le hmeas hε
  refine
    ⟨max M 1, lt_of_lt_of_le zero_lt_one (le_max_right _ _), le_trans (eLpNorm_mono fun x => ?_) hM⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    x : α
    ⊢ LE.le (Norm.norm ((setOf fun x => LE.le (Max.max M 1) ↑(NNNorm.nnnorm (f x)) …
  -/
  rw [norm_indicator_eq_indicator_norm, norm_indicator_eq_indicator_norm]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    x : α
    ⊢ LE.le ((setOf fun x => LE.le (Max.max M 1) ↑(NNNorm.nnnorm (f x))).indicator …
  -/
  refine Set.indicator_le_indicator_of_subset (fun x hx => ?_) (fun x => norm_nonneg (f x)) x
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    x✝ x : α
    hx : Membership.mem (setOf fun x => LE.le (Max.max M 1) ↑(NNNorm.nnnorm (f x)) …
    ⊢ Membership.mem (setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))) x
  -/
  rw [Set.mem_setOf_eq] at hx -- removing the `rw` breaks the proof!
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    x✝ x : α
    hx : LE.le (Max.max M 1) ↑(NNNorm.nnnorm (f x))
    ⊢ Membership.mem (setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))) x
  -/
  exact (max_le_iff.1 hx).1
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_indicator_norm_ge_pos_le := Memℒp.eLpNorm_indicator_norm_ge_pos_le


theorem eLpNorm_indicator_le_of_bound {f : α → β} (hp_top : p ≠ ∞) {ε : ℝ} (hε : 0 < ε) {M : ℝ}
    (hf : ∀ x, ‖f x‖ < M) :
    ∃ (δ : ℝ) (_ : 0 < δ), ∀ s, MeasurableSet s →
      μ s ≤ ENNReal.ofReal δ → eLpNorm (s.indicator f) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  by_cases hM : M ≤ 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LE.le M 0
      ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
    -/
  · refine ⟨1, zero_lt_one, fun s _ _ => ?_⟩
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LE.le M 0
      s : Set α
      x✝¹ : MeasurableSet s
      x✝ : LE.le (μ s) (ENNReal.ofReal 1)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (ENNReal.ofReal ε)
    -/
    rw [(_ : f = 0)]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_top : Ne p Top.top
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
        hM : LE.le M 0
        s : Set α
        x✝¹ : MeasurableSet s
        x✝ : LE.le (μ s) (ENNReal.ofReal 1)
        ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator 0) p μ) (ENNReal.ofReal ε)
      -/
    · simp [hε.le]
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_top : Ne p Top.top
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
        hM : LE.le M 0
        s : Set α
        x✝¹ : MeasurableSet s
        x✝ : LE.le (μ s) (ENNReal.ofReal 1)
        ⊢ Eq f 0
      -/
    · ext x
      /-
        case h
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_top : Ne p Top.top
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
        hM : LE.le M 0
        s : Set α
        x✝¹ : MeasurableSet s
        x✝ : LE.le (μ s) (ENNReal.ofReal 1)
        x : α
        ⊢ Eq (f x) (0 x)
      -/
      rw [Pi.zero_apply, ← norm_le_zero_iff]
      /-
        case h
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_top : Ne p Top.top
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
        hM : LE.le M 0
        s : Set α
        x✝¹ : MeasurableSet s
        x✝ : LE.le (μ s) (ENNReal.ofReal 1)
        x : α
        ⊢ LE.le (Norm.norm (f x)) 0
      -/
      exact (lt_of_lt_of_le (hf x) hM).le
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    hM : Not (LE.le M 0)
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  rw [not_le] at hM
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    hM : LT.lt 0 M
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  refine ⟨(ε / M) ^ p.toReal, Real.rpow_pos_of_pos (div_pos hε hM) _, fun s hs hμ => ?_⟩
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    hM : LT.lt 0 M
    s : Set α
    hs : MeasurableSet s
    hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (ENNReal.ofReal ε)
  -/
  by_cases hp : p = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LT.lt 0 M
      s : Set α
      hs : MeasurableSet s
      hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
      hp : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (ENNReal.ofReal ε)
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    hM : LT.lt 0 M
    s : Set α
    hs : MeasurableSet s
    hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
    hp : Not (Eq p 0)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (ENNReal.ofReal ε)
  -/
  rw [eLpNorm_indicator_eq_eLpNorm_restrict hs]
  have haebdd : ∀ᵐ x ∂μ.restrict s, ‖f x‖ ≤ M := by
    filter_upwards
    exact fun x => (hf x).le
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
    hM : LT.lt 0 M
    s : Set α
    hs : MeasurableSet s
    hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
    hp : Not (Eq p 0)
    haebdd : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) M) (MeasureTheory …
    ⊢ LE.le (MeasureTheory.eLpNorm f p (μ.restrict s)) (ENNReal.ofReal ε)
  -/
  refine le_trans (eLpNorm_le_of_ae_bound haebdd) ?_
  rw [Measure.restrict_apply MeasurableSet.univ, Set.univ_inter,
    ← ENNReal.le_div_iff_mul_le (Or.inl _) (Or.inl ENNReal.ofReal_ne_top)]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LT.lt 0 M
      s : Set α
      hs : MeasurableSet s
      hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
      hp : Not (Eq p 0)
      haebdd : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) M) (MeasureTheory …
      ⊢ LE.le (HPow.hPow (μ s) (Inv.inv p.toReal)) (HDiv.hDiv (ENNReal.ofReal ε) (EN …
    -/
  · rw [ENNReal.rpow_inv_le_iff (ENNReal.toReal_pos hp hp_top)]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LT.lt 0 M
      s : Set α
      hs : MeasurableSet s
      hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
      hp : Not (Eq p 0)
      haebdd : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) M) (MeasureTheory …
      ⊢ LE.le (μ s) (HPow.hPow (HDiv.hDiv (ENNReal.ofReal ε) (ENNReal.ofReal M)) p.t …
    -/
    refine le_trans hμ ?_
    rw [← ENNReal.ofReal_rpow_of_pos (div_pos hε hM),
      ENNReal.rpow_le_rpow_iff (ENNReal.toReal_pos hp hp_top), ENNReal.ofReal_div_of_pos hM]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_top : Ne p Top.top
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hf : ∀ (x : α), LT.lt (Norm.norm (f x)) M
      hM : LT.lt 0 M
      s : Set α
      hs : MeasurableSet s
      hμ : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε M) p.toReal))
      hp : Not (Eq p 0)
      haebdd : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) M) (MeasureTheory …
      ⊢ Ne (ENNReal.ofReal M) 0
    -/
  · simpa only [ENNReal.ofReal_eq_zero, not_le, Ne]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_le_of_bound := eLpNorm_indicator_le_of_bound


/-- Auxiliary lemma for `MeasureTheory.Memℒp.eLpNorm_indicator_le`. -/
theorem Memℒp.eLpNorm_indicator_le' (hp_one : 1 ≤ p) (hp_top : p ≠ ∞) (hf : Memℒp f p μ)
    (hmeas : StronglyMeasurable f) {ε : ℝ} (hε : 0 < ε) :
    ∃ (δ : ℝ) (_ : 0 < δ), ∀ s, MeasurableSet s → μ s ≤ ENNReal.ofReal δ →
      eLpNorm (s.indicator f) p μ ≤ 2 * ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  obtain ⟨M, hMpos, hM⟩ := hf.eLpNorm_indicator_norm_ge_pos_le hmeas hε
  obtain ⟨δ, hδpos, hδ⟩ :=
    eLpNorm_indicator_le_of_bound (f := { x | ‖f x‖ < M }.indicator f) hp_top hε (by
      intro x
      rw [norm_indicator_eq_indicator_norm, Set.indicator_apply]
      · split_ifs with h
        exacts [h, hMpos])
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hMpos : LT.lt 0 M
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  refine ⟨δ, hδpos, fun s hs hμs => ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    M : Real
    hMpos : LT.lt 0 M
    hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (HMul.hMul 2 (ENNReal.ofRe …
  -/
  rw [(_ : f = { x : α | M ≤ ‖f x‖₊ }.indicator f + { x : α | ‖f x‖ < M }.indicator f)]
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hMpos : LT.lt 0 M
      hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (HAdd.hAdd ((setOf fun x => LE.le  …
    -/
  · rw [eLpNorm_indicator_eq_eLpNorm_restrict hs]
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hMpos : LT.lt 0 M
      hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd ((setOf fun x => LE.le M ↑(NNNorm.nn …
    -/
    refine le_trans (eLpNorm_add_le ?_ ?_ hp_one) ?_
    · exact StronglyMeasurable.aestronglyMeasurable
        (hmeas.indicator (measurableSet_le measurable_const hmeas.nnnorm.measurable.subtype_coe))
    · exact StronglyMeasurable.aestronglyMeasurable
        (hmeas.indicator (measurableSet_lt hmeas.nnnorm.measurable.subtype_coe measurable_const))
      /-
        case intro.intro.intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        ⊢ LE.le (HAdd.hAdd (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nn …
      -/
    · rw [two_mul]
      /-
        case intro.intro.intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        ⊢ LE.le (HAdd.hAdd (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nn …
      -/
      refine add_le_add (le_trans (eLpNorm_mono_measure _ Measure.restrict_le_self) hM) ?_
      /-
        case intro.intro.intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        ⊢ LE.le (MeasureTheory.eLpNorm ((setOf fun x => LT.lt (Norm.norm (f x)) M).ind …
      -/
      rw [← eLpNorm_indicator_eq_eLpNorm_restrict hs]
      /-
        case intro.intro.intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (?m.97585 s) (ENNReal.ofReal δ) →  …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator ((setOf fun x => LT.lt (Norm.norm  …
      -/
      exact hδ s hs hμs
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hMpos : LT.lt 0 M
      hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ Eq f (HAdd.hAdd ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))).indicator f …
    -/
  · ext x
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      M : Real
      hMpos : LT.lt 0 M
      hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      x : α
      ⊢ Eq (f x) (HAdd.hAdd ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))).indicat …
    -/
    by_cases hx : M ≤ ‖f x‖
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        x : α
        hx : LE.le M (Norm.norm (f x))
        ⊢ Eq (f x) (HAdd.hAdd ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))).indicat …
      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    · rw [Pi.add_apply, Set.indicator_of_mem, Set.indicator_of_not_mem, add_zero] <;> simpa
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        f : α → β
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        hf : MeasureTheory.Memℒp f p μ
        hmeas : MeasureTheory.StronglyMeasurable f
        ε : Real
        hε : LT.lt 0 ε
        M : Real
        hMpos : LT.lt 0 M
        hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
        δ : Real
        hδpos : LT.lt 0 δ
        hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal δ)
        x : α
        hx : Not (LE.le M (Norm.norm (f x)))
        ⊢ Eq (f x) (HAdd.hAdd ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f x))).indicat …
      -/
    · rw [Pi.add_apply, Set.indicator_of_not_mem, Set.indicator_of_mem, zero_add] <;>
        /-
          case neg.h
          α : Type u_1
          β : Type u_2
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          inst✝ : NormedAddCommGroup β
          p : ENNReal
          f : α → β
          hp_one : LE.le 1 p
          hp_top : Ne p Top.top
          hf : MeasureTheory.Memℒp f p μ
          hmeas : MeasureTheory.StronglyMeasurable f
          ε : Real
          hε : LT.lt 0 ε
          M : Real
          hMpos : LT.lt 0 M
          hM : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le M ↑(NNNorm.nnnorm (f  …
          δ : Real
          hδpos : LT.lt 0 δ
          hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
          s : Set α
          hs : MeasurableSet s
          hμs : LE.le (μ s) (ENNReal.ofReal δ)
          x : α
          hx : Not (LE.le M (Norm.norm (f x)))
          ⊢ Membership.mem (setOf fun x => LT.lt (Norm.norm (f x)) M) x
        -/
        /-
          🎉 no goals
        -/
        simpa using hx
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_indicator_le' := Memℒp.eLpNorm_indicator_le'


/-- This lemma is superseded by `MeasureTheory.Memℒp.eLpNorm_indicator_le` which does not require
measurability on `f`. -/
theorem Memℒp.eLpNorm_indicator_le_of_meas (hp_one : 1 ≤ p) (hp_top : p ≠ ∞) (hf : Memℒp f p μ)
    (hmeas : StronglyMeasurable f) {ε : ℝ} (hε : 0 < ε) :
    ∃ (δ : ℝ) (_ : 0 < δ), ∀ s, MeasurableSet s → μ s ≤ ENNReal.ofReal δ →
      eLpNorm (s.indicator f) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  obtain ⟨δ, hδpos, hδ⟩ := hf.eLpNorm_indicator_le' hp_one hp_top hmeas (half_pos hε)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    hmeas : MeasureTheory.StronglyMeasurable f
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  refine ⟨δ, hδpos, fun s hs hμs => le_trans (hδ s hs hμs) ?_⟩
  rw [ENNReal.ofReal_div_of_pos zero_lt_two, (by norm_num : ENNReal.ofReal 2 = 2),
      ENNReal.mul_div_cancel] <;>
    /-
      case intro.intro.ha₀
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : α → β
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      hmeas : MeasureTheory.StronglyMeasurable f
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ Ne 2 0
    -/
    /-
      🎉 no goals
    -/
    norm_num
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_indicator_le_of_meas := Memℒp.eLpNorm_indicator_le_of_meas


theorem Memℒp.eLpNorm_indicator_le (hp_one : 1 ≤ p) (hp_top : p ≠ ∞) (hf : Memℒp f p μ) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ (δ : ℝ) (_ : 0 < δ), ∀ s, MeasurableSet s → μ s ≤ ENNReal.ofReal δ →
      eLpNorm (s.indicator f) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  have hℒp := hf
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  obtain ⟨⟨f', hf', heq⟩, _⟩ := hf
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    right✝ : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    f' : α → β
    hf' : MeasureTheory.StronglyMeasurable f'
    heq : (MeasureTheory.ae μ).EventuallyEq f f'
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  obtain ⟨δ, hδpos, hδ⟩ := (hℒp.ae_eq heq).eLpNorm_indicator_le_of_meas hp_one hp_top hf' hε
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    right✝ : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    f' : α → β
    hf' : MeasureTheory.StronglyMeasurable f'
    heq : (MeasureTheory.ae μ).EventuallyEq f f'
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
    ⊢ Exists fun δ => Exists fun x => ∀ (s : Set α), MeasurableSet s → LE.le (μ s) …
  -/
  refine ⟨δ, hδpos, fun s hs hμs => ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    right✝ : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    f' : α → β
    hf' : MeasureTheory.StronglyMeasurable f'
    heq : (MeasureTheory.ae μ).EventuallyEq f f'
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (ENNReal.ofReal ε)
  -/
  convert hδ s hs hμs using 1
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    right✝ : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    f' : α → β
    hf' : MeasureTheory.StronglyMeasurable f'
    heq : (MeasureTheory.ae μ).EventuallyEq f f'
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm (s.ind …
  -/
  rw [eLpNorm_indicator_eq_eLpNorm_restrict hs, eLpNorm_indicator_eq_eLpNorm_restrict hs]
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : α → β
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    ε : Real
    hε : LT.lt 0 ε
    hℒp : MeasureTheory.Memℒp f p μ
    right✝ : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    f' : α → β
    hf' : MeasureTheory.StronglyMeasurable f'
    heq : (MeasureTheory.ae μ).EventuallyEq f f'
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.restrict s)) (MeasureTheory.eLpNorm f' p (μ …
  -/
  exact eLpNorm_congr_ae heq.restrict
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_indicator_le := Memℒp.eLpNorm_indicator_le


/-- A constant function is uniformly integrable. -/
theorem unifIntegrable_const {g : α → β} (hp : 1 ≤ p) (hp_ne_top : p ≠ ∞) (hg : Memℒp g p μ) :
    UnifIntegrable (fun _ : ι => g) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    g : α → β
    hp : LE.le 1 p
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ⊢ MeasureTheory.UnifIntegrable (fun x => g) p μ
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    g : α → β
    hp : LE.le 1 p
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ, hδ_pos, hgδ⟩ := hg.eLpNorm_indicator_le hp hp_ne_top hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    g : α → β
    hp : LE.le 1 p
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ε : Real
    hε : LT.lt 0 ε
    δ : Real
    hδ_pos : LT.lt 0 δ
    hgδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le  …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  exact ⟨δ, hδ_pos, fun _ => hgδ⟩
  /-
    🎉 no goals
  -/


/-- A single function is uniformly integrable. -/
theorem unifIntegrable_subsingleton [Subsingleton ι] (hp_one : 1 ≤ p) (hp_top : p ≠ ∞)
    {f : ι → α → β} (hf : ∀ i, Memℒp (f i) p μ) : UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Subsingleton ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Subsingleton ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  by_cases hι : Nonempty ι
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : Subsingleton ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      hι : Nonempty ι
      ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
    -/
  · cases' hι with i
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : Subsingleton ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      i : ι
      ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
    -/
    obtain ⟨δ, hδpos, hδ⟩ := (hf i).eLpNorm_indicator_le hp_one hp_top hε
    /-
      case pos.intro.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : Subsingleton ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      i : ι
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
      ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
    -/
    refine ⟨δ, hδpos, fun j s hs hμs => ?_⟩
    /-
      case pos.intro.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : Subsingleton ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      i : ι
      δ : Real
      hδpos : LT.lt 0 δ
      hδ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → LE.le ( …
      j : ι
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f j)) p μ) (ENNReal.ofReal ε)
    -/
    convert hδ s hs hμs
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : Subsingleton ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      hι : Not (Nonempty ι)
      ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
    -/
  · exact ⟨1, zero_lt_one, fun i => False.elim <| hι <| Nonempty.intro i⟩
    /-
      🎉 no goals
    -/


/-- This lemma is less general than `MeasureTheory.unifIntegrable_finite` which applies to
all sequences indexed by a finite type. -/
theorem unifIntegrable_fin (hp_one : 1 ≤ p) (hp_top : p ≠ ∞) {n : ℕ} {f : Fin n → α → β}
    (hf : ∀ i, Memℒp (f i) p μ) : UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    f : Fin n → α → β
    hf : ∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  revert f
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    ⊢ ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Meas …
  -/
  induction' n with n h
    /-
      case zero
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      ⊢ ∀ {f : Fin 0 → α → β}, (∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ) → Meas …
    -/
  · intro f hf
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added this instance
    /-
      case zero
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : Fin 0 → α → β
      hf : ∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ
      ⊢ MeasureTheory.UnifIntegrable f p μ
    -/
    have : Subsingleton (Fin Nat.zero) := subsingleton_fin_zero
    /-
      case zero
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      f : Fin 0 → α → β
      hf : ∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ
      this : Subsingleton (Fin Nat.zero)
      ⊢ MeasureTheory.UnifIntegrable f p μ
    -/
    exact unifIntegrable_subsingleton hp_one hp_top hf
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    ⊢ ∀ {f : Fin (HAdd.hAdd n 1) → α → β}, (∀ (i : Fin (HAdd.hAdd n 1)), MeasureTh …
  -/
  intro f hfLp ε hε
  /-
    case succ
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Fin (HAdd.hAdd n 1)) (s : Set α), Mea …
  -/
  let g : Fin n → α → β := fun k => f k
  /-
    case succ
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := fun k => f ↑↑k
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Fin (HAdd.hAdd n 1)) (s : Set α), Mea …
  -/
  have hgLp : ∀ i, Memℒp (g i) p μ := fun i => hfLp i
  /-
    case succ
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Fin (HAdd.hAdd n 1)) (s : Set α), Mea …
  -/
  obtain ⟨δ₁, hδ₁pos, hδ₁⟩ := h hgLp hε
  /-
    case succ.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ₁ : Real
    hδ₁pos : LT.lt 0 δ₁
    hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Fin (HAdd.hAdd n 1)) (s : Set α), Mea …
  -/
  obtain ⟨δ₂, hδ₂pos, hδ₂⟩ := (hfLp n).eLpNorm_indicator_le hp_one hp_top hε
  /-
    case succ.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ₁ : Real
    hδ₁pos : LT.lt 0 δ₁
    hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
    δ₂ : Real
    hδ₂pos : LT.lt 0 δ₂
    hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Fin (HAdd.hAdd n 1)) (s : Set α), Mea …
  -/
  refine ⟨min δ₁ δ₂, lt_min hδ₁pos hδ₂pos, fun i s hs hμs => ?_⟩
  /-
    case succ.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ₁ : Real
    hδ₁pos : LT.lt 0 δ₁
    hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
    δ₂ : Real
    hδ₂pos : LT.lt 0 δ₂
    hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
    i : Fin (HAdd.hAdd n 1)
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
  -/
  by_cases hi : i.val < n
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      n : Nat
      h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
      f : Fin (HAdd.hAdd n 1) → α → β
      hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      g : Fin n → α → β := fun k => f ↑↑k
      hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
      δ₁ : Real
      hδ₁pos : LT.lt 0 δ₁
      hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
      δ₂ : Real
      hδ₂pos : LT.lt 0 δ₂
      hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
      i : Fin (HAdd.hAdd n 1)
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
      hi : LT.lt (↑i) n
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
    -/
  · rw [(_ : f i = g ⟨i.val, hi⟩)]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : LT.lt (↑i) n
        ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (g ⟨↑i, hi⟩)) p μ) (ENNReal.ofReal …
      -/
    · exact hδ₁ _ s hs (le_trans hμs <| ENNReal.ofReal_le_ofReal <| min_le_left _ _)
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : LT.lt (↑i) n
        ⊢ Eq (f i) (g ⟨↑i, hi⟩)
      -/
    · simp [g]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      n : Nat
      h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
      f : Fin (HAdd.hAdd n 1) → α → β
      hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
      ε : Real
      hε : LT.lt 0 ε
      g : Fin n → α → β := fun k => f ↑↑k
      hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
      δ₁ : Real
      hδ₁pos : LT.lt 0 δ₁
      hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
      δ₂ : Real
      hδ₂pos : LT.lt 0 δ₂
      hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
      i : Fin (HAdd.hAdd n 1)
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
      hi : Not (LT.lt (↑i) n)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
    -/
  · rw [(_ : i = n)]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : Not (LT.lt (↑i) n)
        ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f ↑n)) p μ) (ENNReal.ofReal ε)
      -/
    · exact hδ₂ _ hs (le_trans hμs <| ENNReal.ofReal_le_ofReal <| min_le_right _ _)
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : Not (LT.lt (↑i) n)
        ⊢ Eq i ↑n
      -/
    · have hi' := Fin.is_lt i
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : Not (LT.lt (↑i) n)
        hi' : LT.lt (↑i) (HAdd.hAdd n 1)
        ⊢ Eq i ↑n
      -/
      rw [Nat.lt_succ_iff] at hi'
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : Not (LT.lt (↑i) n)
        hi' : LE.le (↑i) n
        ⊢ Eq i ↑n
      -/
      rw [not_lt] at hi
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : Real
        hε : LT.lt 0 ε
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        δ₁ : Real
        hδ₁pos : LT.lt 0 δ₁
        hδ₁ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
        δ₂ : Real
        hδ₂pos : LT.lt 0 δ₂
        hδ₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → LE.le …
        i : Fin (HAdd.hAdd n 1)
        s : Set α
        hs : MeasurableSet s
        hμs : LE.le (μ s) (ENNReal.ofReal (Min.min δ₁ δ₂))
        hi : LE.le n ↑i
        hi' : LE.le (↑i) n
        ⊢ Eq i ↑n
      -/
      simp [← le_antisymm hi' hi]
      /-
        🎉 no goals
      -/


/-- A finite sequence of Lp functions is uniformly integrable. -/
theorem unifIntegrable_finite [Finite ι] (hp_one : 1 ≤ p) (hp_top : p ≠ ∞) {f : ι → α → β}
    (hf : ∀ i, Memℒp (f i) p μ) : UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  obtain ⟨n, hn⟩ := Finite.exists_equiv_fin ι
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  intro ε hε
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  let g : Fin n → α → β := f ∘ hn.some.symm
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  have hg : ∀ i, Memℒp (g i) p μ := fun _ => hf _
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨δ, hδpos, hδ⟩ := unifIntegrable_fin hp_one hp_top hg hε
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal  …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  refine ⟨δ, hδpos, fun i s hs hμs => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (i : Fin n) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal  …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
  -/
  specialize hδ (hn.some i) s hs hμs
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ : Real
    hδpos : LT.lt 0 δ
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    hδ : LE.le (MeasureTheory.eLpNorm (s.indicator (g (hn.some i))) p μ) (ENNReal. …
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
  -/
  simp_rw [g, Function.comp_apply, Equiv.symm_apply_apply] at hδ
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ε : Real
    hε : LT.lt 0 ε
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    δ : Real
    hδpos : LT.lt 0 δ
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ)
    hδ : LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem eLpNorm_sub_le_of_dist_bdd (μ : Measure α)
    {p : ℝ≥0∞} (hp' : p ≠ ∞) {s : Set α} (hs : MeasurableSet[m] s)
    {f g : α → β} {c : ℝ} (hc : 0 ≤ c) (hf : ∀ x ∈ s, dist (f x) (g x) ≤ c) :
    eLpNorm (s.indicator (f - g)) p μ ≤ ENNReal.ofReal c * μ s ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp' : Ne p Top.top
    s : Set α
    hs : MeasurableSet s
    f g : α → β
    c : Real
    hc : LE.le 0 c
    hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (HSub.hSub f g)) p μ) (HMul.hMul ( …
  -/
  by_cases hp : p = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp' : Ne p Top.top
      s : Set α
      hs : MeasurableSet s
      f g : α → β
      c : Real
      hc : LE.le 0 c
      hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
      hp : Eq p 0
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (HSub.hSub f g)) p μ) (HMul.hMul ( …
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  have : ∀ x, ‖s.indicator (f - g) x‖ ≤ ‖s.indicator (fun _ => c) x‖ := by
    intro x
    by_cases hx : x ∈ s
    · rw [Set.indicator_of_mem hx, Set.indicator_of_mem hx, Pi.sub_apply, ← dist_eq_norm,
        Real.norm_eq_abs, abs_of_nonneg hc]
      exact hf x hx
    · simp [Set.indicator_of_not_mem hx]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp' : Ne p Top.top
    s : Set α
    hs : MeasurableSet s
    f g : α → β
    c : Real
    hc : LE.le 0 c
    hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
    hp : Not (Eq p 0)
    this : ∀ (x : α), LE.le (Norm.norm (s.indicator (HSub.hSub f g) x)) (Norm.norm …
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (HSub.hSub f g)) p μ) (HMul.hMul ( …
  -/
  refine le_trans (eLpNorm_mono this) ?_
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp' : Ne p Top.top
    s : Set α
    hs : MeasurableSet s
    f g : α → β
    c : Real
    hc : LE.le 0 c
    hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
    hp : Not (Eq p 0)
    this : ∀ (x : α), LE.le (Norm.norm (s.indicator (HSub.hSub f g) x)) (Norm.norm …
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (ENNRe …
  -/
  rw [eLpNorm_indicator_const hs hp hp']
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp' : Ne p Top.top
    s : Set α
    hs : MeasurableSet s
    f g : α → β
    c : Real
    hc : LE.le 0 c
    hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
    hp : Not (Eq p 0)
    this : ∀ (x : α), LE.le (Norm.norm (s.indicator (HSub.hSub f g) x)) (Norm.norm …
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ s) (HDiv.hDiv 1 p.toReal …
  -/
  refine mul_le_mul_right' (le_of_eq ?_) _
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp' : Ne p Top.top
    s : Set α
    hs : MeasurableSet s
    f g : α → β
    c : Real
    hc : LE.le 0 c
    hf : ∀ (x : α), Membership.mem s x → LE.le (Dist.dist (f x) (g x)) c
    hp : Not (Eq p 0)
    this : ∀ (x : α), LE.le (Norm.norm (s.indicator (HSub.hSub f g) x)) (Norm.norm …
    ⊢ Eq (↑(NNNorm.nnnorm c)) (ENNReal.ofReal c)
  -/
  rw [← ofReal_norm_eq_coe_nnnorm, Real.norm_eq_abs, abs_of_nonneg hc]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_sub_le_of_dist_bdd := eLpNorm_sub_le_of_dist_bdd


/-- A sequence of uniformly integrable functions which converges μ-a.e. converges in Lp. -/
theorem tendsto_Lp_finite_of_tendsto_ae_of_meas [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    {f : ℕ → α → β} {g : α → β} (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g)
    (hg' : Memℒp g p μ) (hui : UnifIntegrable f p μ)
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  rw [ENNReal.tendsto_atTop_zero]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le  …
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  by_cases h : ε < ∞; swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : Nat → α → β
      g : α → β
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hg' : MeasureTheory.Memℒp g p μ
      hui : MeasureTheory.UnifIntegrable f p μ
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : ENNReal
      hε : GT.gt ε 0
      h : Not (LT.lt ε Top.top)
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
    -/
  · rw [not_lt, top_le_iff] at h
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : Nat → α → β
      g : α → β
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hg' : MeasureTheory.Memℒp g p μ
      hui : MeasureTheory.UnifIntegrable f p μ
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : ENNReal
      hε : GT.gt ε 0
      h : Eq ε Top.top
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
    -/
    exact ⟨0, fun n _ => by simp [h]⟩
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : Nat → α → β
      g : α → β
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hg' : MeasureTheory.Memℒp g p μ
      hui : MeasureTheory.UnifIntegrable f p μ
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : ENNReal
      hε : GT.gt ε 0
      h : LT.lt ε Top.top
      hμ : Eq μ 0
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
    -/
  · exact ⟨0, fun n _ => by simp [hμ]⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hε' : 0 < ε.toReal / 3 := div_pos (ENNReal.toReal_pos hε.ne' h.ne) (by norm_num)
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hdivp : 0 ≤ 1 / p.toReal := by positivity
  have hpow : 0 < measureUnivNNReal μ ^ (1 / p.toReal) :=
    Real.rpow_pos_of_pos (measureUnivNNReal_pos hμ) _
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨δ₁, hδ₁, heLpNorm₁⟩ := hui hε'
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨δ₂, hδ₂, heLpNorm₂⟩ := hg'.eLpNorm_indicator_le hp hp' hε'
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨t, htm, ht₁, ht₂⟩ := tendstoUniformlyOn_of_ae_tendsto' hf hg hfg (lt_min hδ₁ hδ₂)
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    ht₂ : TendstoUniformlyOn f g Filter.atTop (HasCompl.compl t)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  rw [Metric.tendstoUniformlyOn_iff] at ht₂
  specialize ht₂ (ε.toReal / (3 * measureUnivNNReal μ ^ (1 / p.toReal)))
    (div_pos (ENNReal.toReal_pos (gt_iff_lt.1 hε).ne.symm h.ne) (mul_pos (by norm_num) hpow))
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    ht₂ : Filter.Eventually (fun n => ∀ (x : α), Membership.mem (HasCompl.compl t) …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨N, hN⟩ := eventually_atTop.1 ht₂; clear ht₂
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  refine ⟨N, fun n hn => ?_⟩
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) ε
  -/
  rw [← t.indicator_self_add_compl (f n - g)]
  refine le_trans (eLpNorm_add_le (((hf n).sub hg).indicator htm).aestronglyMeasurable
    (((hf n).sub hg).indicator htm.compl).aestronglyMeasurable hp) ?_
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.eLpNorm (t.indicator (HSub.hSub (f n) g)) p  …
  -/
  rw [sub_eq_add_neg, Set.indicator_add' t, Set.indicator_neg']
  refine le_trans (add_le_add_right (eLpNorm_add_le ((hf n).indicator htm).aestronglyMeasurable
    (hg.indicator htm).neg.aestronglyMeasurable hp) _) ?_
  have hnf : eLpNorm (t.indicator (f n)) p μ ≤ ENNReal.ofReal (ε.toReal / 3) := by
    refine heLpNorm₁ n t htm (le_trans ht₁ ?_)
    rw [ENNReal.ofReal_le_ofReal_iff hδ₁.le]
    exact min_le_left _ _
  have hng : eLpNorm (t.indicator g) p μ ≤ ENNReal.ofReal (ε.toReal / 3) := by
    refine heLpNorm₂ t htm (le_trans ht₁ ?_)
    rw [ENNReal.ofReal_le_ofReal_iff hδ₂.le]
    exact min_le_right _ _
  have hlt : eLpNorm (tᶜ.indicator (f n - g)) p μ ≤ ENNReal.ofReal (ε.toReal / 3) := by
    specialize hN n hn
    have : 0 ≤ ε.toReal / (3 * measureUnivNNReal μ ^ (1 / p.toReal)) := by positivity
    have := eLpNorm_sub_le_of_dist_bdd μ hp' htm.compl this fun x hx =>
      (dist_comm (g x) (f n x) ▸ (hN x hx).le :
        dist (f n x) (g x) ≤ ε.toReal / (3 * measureUnivNNReal μ ^ (1 / p.toReal)))
    refine le_trans this ?_
    rw [div_mul_eq_div_mul_one_div, ← ENNReal.ofReal_toReal (measure_lt_top μ tᶜ).ne,
      ENNReal.ofReal_rpow_of_nonneg ENNReal.toReal_nonneg hdivp, ← ENNReal.ofReal_mul, mul_assoc]
    · refine ENNReal.ofReal_le_ofReal (mul_le_of_le_one_right hε'.le ?_)
      rw [mul_comm, mul_one_div, div_le_one]
      · refine Real.rpow_le_rpow ENNReal.toReal_nonneg
          (ENNReal.toReal_le_of_le_ofReal (measureUnivNNReal_pos hμ).le ?_) hdivp
        rw [ENNReal.ofReal_coe_nnreal, coe_measureUnivNNReal]
        exact measure_mono (Set.subset_univ _)
      · exact Real.rpow_pos_of_pos (measureUnivNNReal_pos hμ) _
    · positivity
  have : ENNReal.ofReal (ε.toReal / 3) = ε / 3 := by
    rw [ENNReal.ofReal_div_of_pos (show (0 : ℝ) < 3 by norm_num), ENNReal.ofReal_toReal h.ne]
    simp
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    hnf : LE.le (MeasureTheory.eLpNorm (t.indicator (f n)) p μ) (ENNReal.ofReal (H …
    hng : LE.le (MeasureTheory.eLpNorm (t.indicator g) p μ) (ENNReal.ofReal (HDiv. …
    hlt : LE.le (MeasureTheory.eLpNorm ((HasCompl.compl t).indicator (HSub.hSub (f …
    this : Eq (ENNReal.ofReal (HDiv.hDiv ε.toReal 3)) (HDiv.hDiv ε 3)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.eLpNorm (t.indicator (f n)) p μ)  …
  -/
  rw [this] at hnf hng hlt
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    hnf : LE.le (MeasureTheory.eLpNorm (t.indicator (f n)) p μ) (HDiv.hDiv ε 3)
    hng : LE.le (MeasureTheory.eLpNorm (t.indicator g) p μ) (HDiv.hDiv ε 3)
    hlt : LE.le (MeasureTheory.eLpNorm ((HasCompl.compl t).indicator (HSub.hSub (f …
    this : Eq (ENNReal.ofReal (HDiv.hDiv ε.toReal 3)) (HDiv.hDiv ε 3)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.eLpNorm (t.indicator (f n)) p μ)  …
  -/
  rw [eLpNorm_neg, ← ENNReal.add_thirds ε, ← sub_eq_add_neg]
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    h : LT.lt ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε.toReal 3)
    hdivp : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hpow : LT.lt 0 (HPow.hPow (MeasureTheory.measureUnivNNReal μ) (HDiv.hDiv 1 p.t …
    δ₁ : Real
    hδ₁ : LT.lt 0 δ₁
    heLpNorm₁ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.of …
    δ₂ : Real
    hδ₂ : LT.lt 0 δ₂
    heLpNorm₂ : ∀ (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ₂) → …
    t : Set α
    htm : MeasurableSet t
    ht₁ : LE.le (μ t) (ENNReal.ofReal (Min.min δ₁ δ₂))
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x : α), Membership.mem (HasCompl.compl t) x → …
    n : Nat
    hn : GE.ge n N
    hnf : LE.le (MeasureTheory.eLpNorm (t.indicator (f n)) p μ) (HDiv.hDiv ε 3)
    hng : LE.le (MeasureTheory.eLpNorm (t.indicator g) p μ) (HDiv.hDiv ε 3)
    hlt : LE.le (MeasureTheory.eLpNorm ((HasCompl.compl t).indicator (HSub.hSub (f …
    this : Eq (ENNReal.ofReal (HDiv.hDiv ε.toReal 3)) (HDiv.hDiv ε 3)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.eLpNorm (t.indicator (f n)) p μ)  …
  -/
  exact add_le_add_three hnf hng hlt
  /-
    🎉 no goals
  -/


/-- A sequence of uniformly integrable functions which converges μ-a.e. converges in Lp. -/
theorem tendsto_Lp_finite_of_tendsto_ae [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    {f : ℕ → α → β} {g : α → β} (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hg : Memℒp g p μ)
    (hui : UnifIntegrable f p μ) (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  have : ∀ n, eLpNorm (f n - g) p μ = eLpNorm ((hf n).mk (f n) - hg.1.mk g) p μ :=
    fun n => eLpNorm_congr_ae ((hf n).ae_eq_mk.sub hg.1.ae_eq_mk)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  simp_rw [this]
  refine tendsto_Lp_finite_of_tendsto_ae_of_meas hp hp' (fun n => (hf n).stronglyMeasurable_mk)
    hg.1.stronglyMeasurable_mk (hg.ae_eq hg.1.ae_eq_mk) (hui.ae_eq fun n => (hf n).ae_eq_mk) ?_
  have h_ae_forall_eq : ∀ᵐ x ∂μ, ∀ n, f n x = (hf n).mk (f n) x := by
    rw [ae_all_iff]
    exact fun n => (hf n).ae_eq_mk
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    h_ae_forall_eq : Filter.Eventually (fun x => ∀ (n : Nat), Eq (f n x) (MeasureT …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AEStrongl …
  -/
  filter_upwards [hfg, h_ae_forall_eq, hg.1.ae_eq_mk] with x hx_tendsto hxf_eq hxg_eq
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    h_ae_forall_eq : Filter.Eventually (fun x => ∀ (n : Nat), Eq (f n x) (MeasureT …
    x : α
    hx_tendsto : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    hxf_eq : ∀ (n : Nat), Eq (f n x) (MeasureTheory.AEStronglyMeasurable.mk (f n)  …
    hxg_eq : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g ⋯ x)
    ⊢ Filter.Tendsto (fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) Fi …
  -/
  rw [← hxg_eq]
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    h_ae_forall_eq : Filter.Eventually (fun x => ∀ (n : Nat), Eq (f n x) (MeasureT …
    x : α
    hx_tendsto : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    hxf_eq : ∀ (n : Nat), Eq (f n x) (MeasureTheory.AEStronglyMeasurable.mk (f n)  …
    hxg_eq : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g ⋯ x)
    ⊢ Filter.Tendsto (fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) Fi …
  -/
  convert hx_tendsto using 1
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    h_ae_forall_eq : Filter.Eventually (fun x => ∀ (n : Nat), Eq (f n x) (MeasureT …
    x : α
    hx_tendsto : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    hxf_eq : ∀ (n : Nat), Eq (f n x) (MeasureTheory.AEStronglyMeasurable.mk (f n)  …
    hxg_eq : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g ⋯ x)
    ⊢ Eq (fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) fun n => f n x
  -/
  ext1 n
  /-
    case h.e'_3.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    this : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measur …
    h_ae_forall_eq : Filter.Eventually (fun x => ∀ (n : Nat), Eq (f n x) (MeasureT …
    x : α
    hx_tendsto : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    hxf_eq : ∀ (n : Nat), Eq (f n x) (MeasureTheory.AEStronglyMeasurable.mk (f n)  …
    hxg_eq : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g ⋯ x)
    n : Nat
    ⊢ Eq (MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯ x) (f n x)
  -/
  exact (hxf_eq n).symm
  /-
    🎉 no goals
  -/


theorem unifIntegrable_of_tendsto_Lp_zero (hp : 1 ≤ p) (hp' : p ≠ ∞) (hf : ∀ n, Memℒp (f n) p μ)
    (hf_tendsto : Tendsto (fun n => eLpNorm (f n) p μ) atTop (𝓝 0)) : UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (f n) p μ) Filter. …
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (f n) p μ) Filter. …
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  rw [ENNReal.tendsto_atTop_zero] at hf_tendsto
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  obtain ⟨N, hN⟩ := hf_tendsto (ENNReal.ofReal ε) (by simpa)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  let F : Fin N → α → β := fun n => f n
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
    F : Fin N → α → β := fun n => f ↑n
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  have hF : ∀ n, Memℒp (F n) p μ := fun n => hf n
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  obtain ⟨δ₁, hδpos₁, hδ₁⟩ := unifIntegrable_fin hp hp' hF hε
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    δ₁ : Real
    hδpos₁ : LT.lt 0 δ₁
    hδ₁ : ∀ (i : Fin N) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
    ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
  -/
  refine ⟨δ₁, hδpos₁, fun n s hs hμs => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : Real
    hε : LT.lt 0 ε
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    δ₁ : Real
    hδpos₁ : LT.lt 0 δ₁
    hδ₁ : ∀ (i : Fin N) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
    n : Nat
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal δ₁)
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f n)) p μ) (ENNReal.ofReal ε)
  -/
  by_cases hn : n < N
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : Nat → α → β
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
      ε : Real
      hε : LT.lt 0 ε
      N : Nat
      hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
      F : Fin N → α → β := fun n => f ↑n
      hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
      δ₁ : Real
      hδpos₁ : LT.lt 0 δ₁
      hδ₁ : ∀ (i : Fin N) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ₁)
      hn : LT.lt n N
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f n)) p μ) (ENNReal.ofReal ε)
    -/
  · exact hδ₁ ⟨n, hn⟩ s hs hμs
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : Nat → α → β
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
      ε : Real
      hε : LT.lt 0 ε
      N : Nat
      hN : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) (ENNReal …
      F : Fin N → α → β := fun n => f ↑n
      hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
      δ₁ : Real
      hδpos₁ : LT.lt 0 δ₁
      hδ₁ : ∀ (i : Fin N) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal δ₁)
      hn : Not (LT.lt n N)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f n)) p μ) (ENNReal.ofReal ε)
    -/
  · exact (eLpNorm_indicator_le _).trans (hN n (not_lt.1 hn))
    /-
      🎉 no goals
    -/


/-- Convergence in Lp implies uniform integrability. -/
theorem unifIntegrable_of_tendsto_Lp (hp : 1 ≤ p) (hp' : p ≠ ∞) (hf : ∀ n, Memℒp (f n) p μ)
    (hg : Memℒp g p μ) (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0)) :
    UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hg : MeasureTheory.Memℒp g p μ
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  have : f = (fun _ => g) + fun n => f n - g := by ext1 n; simp
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hg : MeasureTheory.Memℒp g p μ
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  rw [this]
  refine UnifIntegrable.add ?_ ?_ hp (fun _ => hg.aestronglyMeasurable)
      fun n => (hf n).1.sub hg.aestronglyMeasurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : Nat → α → β
      g : α → β
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hg : MeasureTheory.Memℒp g p μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
      ⊢ MeasureTheory.UnifIntegrable (fun x => g) p μ
    -/
  · exact unifIntegrable_const hp hp' hg
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : Nat → α → β
      g : α → β
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hg : MeasureTheory.Memℒp g p μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
      ⊢ MeasureTheory.UnifIntegrable (fun n => HSub.hSub (f n) g) p μ
    -/
  · exact unifIntegrable_of_tendsto_Lp_zero hp hp' (fun n => (hf n).sub hg) hfg
    /-
      🎉 no goals
    -/


/-- Forward direction of Vitali's convergence theorem: if `f` is a sequence of uniformly integrable
functions that converge in measure to some function `g` in a finite measure space, then `f`
converge in Lp to `g`. -/
theorem tendsto_Lp_finite_of_tendstoInMeasure [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hg : Memℒp g p μ) (hui : UnifIntegrable f p μ)
    (hfg : TendstoInMeasure μ f atTop g) : Tendsto (fun n ↦ eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    g : α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  refine tendsto_of_subseq_tendsto fun ns hns => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : Nat → α → β
    g : α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    ns : Nat → Nat
    hns : Filter.Tendsto ns Filter.atTop Filter.atTop
    ⊢ Exists fun ms => Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ( …
  -/
  obtain ⟨ms, _, hms'⟩ := TendstoInMeasure.exists_seq_tendsto_ae fun ε hε => (hfg ε hε).comp hns
  exact ⟨ms,
    tendsto_Lp_finite_of_tendsto_ae hp hp' (fun _ => hf _) hg (fun ε hε =>
      let ⟨δ, hδ, hδ'⟩ := hui hε
      ⟨δ, hδ, fun i s hs hμs => hδ' _ s hs hμs⟩)
      hms'⟩


/-- **Vitali's convergence theorem**: A sequence of functions `f` converges to `g` in Lp if and
only if it is uniformly integrable and converges to `g` in measure. -/
theorem tendstoInMeasure_iff_tendsto_Lp_finite [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ n, Memℒp (f n) p μ) (hg : Memℒp g p μ) :
    TendstoInMeasure μ f atTop g ∧ UnifIntegrable f p μ ↔
      Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) :=
  ⟨fun h => tendsto_Lp_finite_of_tendstoInMeasure hp hp' (fun n => (hf n).1) hg h.2 h.1, fun h =>
    ⟨tendstoInMeasure_of_tendsto_eLpNorm (lt_of_lt_of_le zero_lt_one hp).ne.symm
        (fun n => (hf n).aestronglyMeasurable) hg.aestronglyMeasurable h,
      unifIntegrable_of_tendsto_Lp hp hp' hf hg h⟩⟩


/-- This lemma is superseded by `unifIntegrable_of` which do not require `C` to be positive. -/
theorem unifIntegrable_of' (hp : 1 ≤ p) (hp' : p ≠ ∞) {f : ι → α → β}
    (hf : ∀ i, StronglyMeasurable (f i))
    (h : ∀ ε : ℝ, 0 < ε → ∃ C : ℝ≥0, 0 < C ∧
      ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε) :
    UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  have hpzero := (lt_of_lt_of_le zero_lt_one hp).ne.symm
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
    hpzero : Ne p 0
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  by_cases hμ : μ Set.univ = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
      hpzero : Ne p 0
      hμ : Eq (μ Set.univ) 0
      ⊢ MeasureTheory.UnifIntegrable f p μ
    -/
  · rw [Measure.measure_univ_eq_zero] at hμ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
      hpzero : Ne p 0
      hμ : Eq μ 0
      ⊢ MeasureTheory.UnifIntegrable f p μ
    -/
    exact hμ.symm ▸ unifIntegrable_zero_meas
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
    hpzero : Ne p 0
    hμ : Not (Eq (μ Set.univ) 0)
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  intro ε hε
  /-
    case neg
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
    hpzero : Ne p 0
    hμ : Not (Eq (μ Set.univ) 0)
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => Exists fun x => ∀ (i : ι) (s : Set α), MeasurableSet s → LE. …
  -/
  obtain ⟨C, hCpos, hC⟩ := h (ε / 2) (half_pos hε)
  refine ⟨(ε / (2 * C)) ^ ENNReal.toReal p,
    Real.rpow_pos_of_pos (div_pos hε (mul_pos two_pos (NNReal.coe_pos.2 hCpos))) _,
    fun i s hs hμs => ?_⟩
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
    hpzero : Ne p 0
    hμ : Not (Eq (μ Set.univ) 0)
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hCpos : LT.lt 0 C
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    i : ι
    s : Set α
    hs : MeasurableSet s
    hμs : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε (HMul.hMul 2 ↑C)) p. …
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) (ENNReal.ofReal ε)
  -/
  by_cases hμs' : μ s = 0
  · rw [(eLpNorm_eq_zero_iff ((hf i).indicator hs).aestronglyMeasurable hpzero).2
        (indicator_meas_zero hμs')]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.l …
      hpzero : Ne p 0
      hμ : Not (Eq (μ Set.univ) 0)
      ε : Real
      hε : LT.lt 0 ε
      C : NNReal
      hCpos : LT.lt 0 C
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      s : Set α
      hs : MeasurableSet s
      hμs : LE.le (μ s) (ENNReal.ofReal (HPow.hPow (HDiv.hDiv ε (HMul.hMul 2 ↑C)) p. …
      hμs' : Eq (μ s) 0
      ⊢ LE.le 0 (ENNReal.ofReal ε)
    -/
    norm_num
    /-
      🎉 no goals
    -/
  calc
    eLpNorm (Set.indicator s (f i)) p μ ≤
        eLpNorm (Set.indicator (s ∩ { x | C ≤ ‖f i x‖₊ }) (f i)) p μ +
          eLpNorm (Set.indicator (s ∩ { x | ‖f i x‖₊ < C }) (f i)) p μ := by
      refine le_trans (Eq.le ?_) (eLpNorm_add_le
        (StronglyMeasurable.aestronglyMeasurable
          ((hf i).indicator (hs.inter (stronglyMeasurable_const.measurableSet_le (hf i).nnnorm))))
        (StronglyMeasurable.aestronglyMeasurable
          ((hf i).indicator (hs.inter ((hf i).nnnorm.measurableSet_lt stronglyMeasurable_const))))
        hp)
      congr
      change _ = fun x => (s ∩ { x : α | C ≤ ‖f i x‖₊ }).indicator (f i) x +
        (s ∩ { x : α | ‖f i x‖₊ < C }).indicator (f i) x
      rw [← Set.indicator_union_of_disjoint]
      · rw [← Set.inter_union_distrib_left, (by ext; simp [le_or_lt] :
            { x : α | C ≤ ‖f i x‖₊ } ∪ { x : α | ‖f i x‖₊ < C } = Set.univ),
          Set.inter_univ]
      · refine (Disjoint.inf_right' _ ?_).inf_left' _
        rw [disjoint_iff_inf_le]
        rintro x ⟨hx₁, hx₂⟩
        rw [Set.mem_setOf_eq] at hx₁ hx₂
        exact False.elim (hx₂.ne (eq_of_le_of_not_lt hx₁ (not_lt.2 hx₂.le)).symm)
    _ ≤ eLpNorm (Set.indicator { x | C ≤ ‖f i x‖₊ } (f i)) p μ +
        (C : ℝ≥0∞) * μ s ^ (1 / ENNReal.toReal p) := by
      refine add_le_add
        (eLpNorm_mono fun x => norm_indicator_le_of_subset Set.inter_subset_right _ _) ?_
      rw [← Set.indicator_indicator]
      rw [eLpNorm_indicator_eq_eLpNorm_restrict hs]
      have : ∀ᵐ x ∂μ.restrict s, ‖{ x : α | ‖f i x‖₊ < C }.indicator (f i) x‖ ≤ C := by
        filter_upwards
        simp_rw [norm_indicator_eq_indicator_norm]
        exact Set.indicator_le' (fun x (hx : _ < _) => hx.le) fun _ _ => NNReal.coe_nonneg _
      refine le_trans (eLpNorm_le_of_ae_bound this) ?_
      rw [mul_comm, Measure.restrict_apply' hs, Set.univ_inter, ENNReal.ofReal_coe_nnreal, one_div]
    _ ≤ ENNReal.ofReal (ε / 2) + C * ENNReal.ofReal (ε / (2 * C)) := by
      refine add_le_add (hC i) (mul_le_mul_left' ?_ _)
      rwa [one_div, ENNReal.rpow_inv_le_iff (ENNReal.toReal_pos hpzero hp'),
        ENNReal.ofReal_rpow_of_pos (div_pos hε (mul_pos two_pos (NNReal.coe_pos.2 hCpos)))]
    _ ≤ ENNReal.ofReal (ε / 2) + ENNReal.ofReal (ε / 2) := by
      refine add_le_add_left ?_ _
      rw [← ENNReal.ofReal_coe_nnreal, ← ENNReal.ofReal_mul (NNReal.coe_nonneg _), ← div_div,
        mul_div_cancel₀ _ (NNReal.coe_pos.2 hCpos).ne.symm]
    _ ≤ ENNReal.ofReal ε := by
      rw [← ENNReal.ofReal_add (half_pos hε).le (half_pos hε).le, add_halves]


theorem unifIntegrable_of (hp : 1 ≤ p) (hp' : p ≠ ∞) {f : ι → α → β}
    (hf : ∀ i, AEStronglyMeasurable (f i) μ)
    (h : ∀ ε : ℝ, 0 < ε → ∃ C : ℝ≥0,
      ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε) :
    UnifIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    ⊢ MeasureTheory.UnifIntegrable f p μ
  -/
  set g : ι → α → β := fun i => (hf i).choose
  refine
    (unifIntegrable_of' hp hp' (fun i => (Exists.choose_spec <| hf i).1) fun ε hε => ?_).ae_eq
      fun i => (Exists.choose_spec <| hf i).2.symm
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((s …
  -/
  obtain ⟨C, hC⟩ := h ε hε
  have hCg : ∀ i, eLpNorm ({ x | C ≤ ‖g i x‖₊ }.indicator (g i)) p μ ≤ ENNReal.ofReal ε := by
    intro i
    refine le_trans (le_of_eq <| eLpNorm_congr_ae ?_) (hC i)
    filter_upwards [(Exists.choose_spec <| hf i).2] with x hx
    by_cases hfx : x ∈ { x | C ≤ ‖f i x‖₊ }
    · rw [Set.indicator_of_mem hfx, Set.indicator_of_mem, hx]
      rwa [Set.mem_setOf, hx] at hfx
    · rw [Set.indicator_of_not_mem hfx, Set.indicator_of_not_mem]
      rwa [Set.mem_setOf, hx] at hfx
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    hCg : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm …
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((s …
  -/
  refine ⟨max C 1, lt_max_of_lt_right one_pos, fun i => le_trans (eLpNorm_mono fun x => ?_) (hCg i)⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    hCg : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm …
    i : ι
    x : α
    ⊢ LE.le (Norm.norm ((setOf fun x => LE.le (Max.max C 1) (NNNorm.nnnorm (Exists …
  -/
  rw [norm_indicator_eq_indicator_norm, norm_indicator_eq_indicator_norm]
  exact Set.indicator_le_indicator_of_subset
    (fun x hx => Set.mem_setOf_eq ▸ le_trans (le_max_left _ _) hx) (fun _ => norm_nonneg _) _


theorem uniformIntegrable_zero_meas [MeasurableSpace α] : UniformIntegrable f p (0 : Measure α) :=
  ⟨fun _ => aestronglyMeasurable_zero_measure _, unifIntegrable_zero_meas, 0,
    fun _ => eLpNorm_measure_zero.le⟩


theorem UniformIntegrable.ae_eq {g : ι → α → β} (hf : UniformIntegrable f p μ)
    (hfg : ∀ n, f n =ᵐ[μ] g n) : UniformIntegrable g p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f g : ι → α → β
    hf : MeasureTheory.UniformIntegrable f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ⊢ MeasureTheory.UniformIntegrable g p μ
  -/
  obtain ⟨hfm, hunif, C, hC⟩ := hf
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f g : ι → α → β
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hunif : MeasureTheory.UnifIntegrable f p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    ⊢ MeasureTheory.UniformIntegrable g p μ
  -/
  refine ⟨fun i => (hfm i).congr (hfg i), (unifIntegrable_congr_ae hfg).1 hunif, C, fun i => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f g : ι → α → β
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hunif : MeasureTheory.UnifIntegrable f p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    i : ι
    ⊢ LE.le (MeasureTheory.eLpNorm (g i) p μ) ↑C
  -/
  rw [← eLpNorm_congr_ae (hfg i)]
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f g : ι → α → β
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hunif : MeasureTheory.UnifIntegrable f p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    i : ι
    ⊢ LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
  -/
  exact hC i
  /-
    🎉 no goals
  -/


theorem uniformIntegrable_congr_ae {g : ι → α → β} (hfg : ∀ n, f n =ᵐ[μ] g n) :
    UniformIntegrable f p μ ↔ UniformIntegrable g p μ :=
  ⟨fun h => h.ae_eq hfg, fun h => h.ae_eq fun i => (hfg i).symm⟩


/-- A finite sequence of Lp functions is uniformly integrable in the probability sense. -/
theorem uniformIntegrable_finite [Finite ι] (hp_one : 1 ≤ p) (hp_top : p ≠ ∞)
    (hf : ∀ i, Memℒp (f i) p μ) : UniformIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  cases nonempty_fintype ι
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    val✝ : Fintype ι
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  refine ⟨fun n => (hf n).1, unifIntegrable_finite hp_one hp_top hf, ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : Finite ι
    hp_one : LE.le 1 p
    hp_top : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    val✝ : Fintype ι
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
  -/
  by_cases hι : Nonempty ι
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : Finite ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      val✝ : Fintype ι
      hι : Nonempty ι
      ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    -/
  · choose _ hf using hf
    set C := (Finset.univ.image fun i : ι => eLpNorm (f i) p μ).max'
      ⟨eLpNorm (f hι.some) p μ, Finset.mem_image.2 ⟨hι.some, Finset.mem_univ _, rfl⟩⟩
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : Finite ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      val✝ : Fintype ι
      hι : Nonempty ι
      h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
      C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
      ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    -/
    refine ⟨C.toNNReal, fun i => ?_⟩
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : Finite ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      val✝ : Fintype ι
      hι : Nonempty ι
      h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
      C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
      i : ι
      ⊢ LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C.toNNReal
    -/
    rw [ENNReal.coe_toNNReal]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup β
        p : ENNReal
        f : ι → α → β
        inst✝ : Finite ι
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        val✝ : Fintype ι
        hι : Nonempty ι
        h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
        C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
        i : ι
        ⊢ LE.le (MeasureTheory.eLpNorm (f i) p μ) C
      -/
    · exact Finset.le_max' (α := ℝ≥0∞) _ _ (Finset.mem_image.2 ⟨i, Finset.mem_univ _, rfl⟩)
      /-
        🎉 no goals
      -/
      /-
        case pos
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup β
        p : ENNReal
        f : ι → α → β
        inst✝ : Finite ι
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        val✝ : Fintype ι
        hι : Nonempty ι
        h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
        C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
        i : ι
        ⊢ Ne C Top.top
      -/
    · refine ne_of_lt ((Finset.max'_lt_iff _ _).2 fun y hy => ?_)
      /-
        case pos
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup β
        p : ENNReal
        f : ι → α → β
        inst✝ : Finite ι
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        val✝ : Fintype ι
        hι : Nonempty ι
        h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
        C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
        i : ι
        y : ENNReal
        hy : Membership.mem (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) F …
        ⊢ LT.lt y Top.top
      -/
      rw [Finset.mem_image] at hy
      /-
        case pos
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup β
        p : ENNReal
        f : ι → α → β
        inst✝ : Finite ι
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        val✝ : Fintype ι
        hι : Nonempty ι
        h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
        C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
        i : ι
        y : ENNReal
        hy : Exists fun a => And (Membership.mem Finset.univ a) (Eq (MeasureTheory.eLp …
        ⊢ LT.lt y Top.top
      -/
      obtain ⟨i, -, rfl⟩ := hy
      /-
        case pos.intro.intro
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup β
        p : ENNReal
        f : ι → α → β
        inst✝ : Finite ι
        hp_one : LE.le 1 p
        hp_top : Ne p Top.top
        val✝ : Fintype ι
        hι : Nonempty ι
        h✝ : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf : ∀ (i : ι), LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
        C : ENNReal := (Finset.image (fun i => MeasureTheory.eLpNorm (f i) p μ) Finset …
        i✝ i : ι
        ⊢ LT.lt (MeasureTheory.eLpNorm (f i) p μ) Top.top
      -/
      exact hf i
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : Finite ι
      hp_one : LE.le 1 p
      hp_top : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      val✝ : Fintype ι
      hι : Not (Nonempty ι)
      ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    -/
  · exact ⟨0, fun i => False.elim <| hι <| Nonempty.intro i⟩
    /-
      🎉 no goals
    -/


/-- A single function is uniformly integrable in the probability sense. -/
theorem uniformIntegrable_subsingleton [Subsingleton ι] (hp_one : 1 ≤ p) (hp_top : p ≠ ∞)
    (hf : ∀ i, Memℒp (f i) p μ) : UniformIntegrable f p μ :=
  uniformIntegrable_finite hp_one hp_top hf


/-- A constant sequence of functions is uniformly integrable in the probability sense. -/
theorem uniformIntegrable_const {g : α → β} (hp : 1 ≤ p) (hp_ne_top : p ≠ ∞) (hg : Memℒp g p μ) :
    UniformIntegrable (fun _ : ι => g) p μ :=
  ⟨fun _ => hg.1, unifIntegrable_const hp hp_ne_top hg,
    ⟨(eLpNorm g p μ).toNNReal, fun _ => le_of_eq (ENNReal.coe_toNNReal hg.2.ne).symm⟩⟩


/-- This lemma is superseded by `uniformIntegrable_of` which only requires
`AEStronglyMeasurable`. -/
theorem uniformIntegrable_of' [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ i, StronglyMeasurable (f i))
    (h : ∀ ε : ℝ, 0 < ε → ∃ C : ℝ≥0,
      ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε) :
    UniformIntegrable f p μ := by
  refine ⟨fun i => (hf i).aestronglyMeasurable,
    unifIntegrable_of hp hp' (fun i => (hf i).aestronglyMeasurable) h, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
  -/
  obtain ⟨C, hC⟩ := h 1 one_pos
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
  -/
  refine ⟨((C : ℝ≥0∞) * μ Set.univ ^ p.toReal⁻¹ + 1).toNNReal, fun i => ?_⟩
  calc
    eLpNorm (f i) p μ ≤
        eLpNorm ({ x : α | ‖f i x‖₊ < C }.indicator (f i)) p μ +
          eLpNorm ({ x : α | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ := by
      refine le_trans (eLpNorm_mono fun x => ?_) (eLpNorm_add_le
        (StronglyMeasurable.aestronglyMeasurable
          ((hf i).indicator ((hf i).nnnorm.measurableSet_lt stronglyMeasurable_const)))
        (StronglyMeasurable.aestronglyMeasurable
          ((hf i).indicator (stronglyMeasurable_const.measurableSet_le (hf i).nnnorm))) hp)
      rw [Pi.add_apply, Set.indicator_apply]
      split_ifs with hx
      · rw [Set.indicator_of_not_mem, add_zero]
        simpa using hx
      · rw [Set.indicator_of_mem, zero_add]
        simpa using hx
    _ ≤ (C : ℝ≥0∞) * μ Set.univ ^ p.toReal⁻¹ + 1 := by
      have : ∀ᵐ x ∂μ, ‖{ x : α | ‖f i x‖₊ < C }.indicator (f i) x‖₊ ≤ C := by
        filter_upwards
        simp_rw [nnnorm_indicator_eq_indicator_nnnorm]
        exact Set.indicator_le fun x (hx : _ < _) => hx.le
      refine add_le_add (le_trans (eLpNorm_le_of_ae_bound this) ?_) (ENNReal.ofReal_one ▸ hC i)
      simp_rw [NNReal.val_eq_coe, ENNReal.ofReal_coe_nnreal, mul_comm]
      exact le_rfl
    _ = ((C : ℝ≥0∞) * μ Set.univ ^ p.toReal⁻¹ + 1 : ℝ≥0∞).toNNReal := by
      rw [ENNReal.coe_toNNReal]
      exact ENNReal.add_ne_top.2
        ⟨ENNReal.mul_ne_top ENNReal.coe_ne_top (ENNReal.rpow_ne_top_of_nonneg
          (inv_nonneg.2 ENNReal.toReal_nonneg) (measure_lt_top _ _).ne),
        ENNReal.one_ne_top⟩


/-- A sequence of functions `(fₙ)` is uniformly integrable in the probability sense if for all
`ε > 0`, there exists some `C` such that `∫ x in {|fₙ| ≥ C}, fₙ x ∂μ ≤ ε` for all `n`. -/
theorem uniformIntegrable_of [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ i, AEStronglyMeasurable (f i) μ)
    (h : ∀ ε : ℝ, 0 < ε → ∃ C : ℝ≥0,
      ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε) :
    UniformIntegrable f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  set g : ι → α → β := fun i => (hf i).choose
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  have hgmeas : ∀ i, StronglyMeasurable (g i) := fun i => (Exists.choose_spec <| hf i).1
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  have hgeq : ∀ i, g i =ᵐ[μ] f i := fun i => (Exists.choose_spec <| hf i).2.symm
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  refine (uniformIntegrable_of' hp hp' hgmeas fun ε hε => ?_).ae_eq hgeq
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨C, hC⟩ := h ε hε
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  refine ⟨C, fun i => le_trans (le_of_eq <| eLpNorm_congr_ae ?_) (hC i)⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    i : ι
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((setOf fun x => LE.le C (NNNorm.nnnorm (g …
  -/
  filter_upwards [(Exists.choose_spec <| hf i).2] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ε : Real
    hε : LT.lt 0 ε
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    i : ι
    x : α
    hx : Eq (f i x) (Exists.choose ⋯ x)
    ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (g i x))).indicator (g i) x) ((se …
  -/
  by_cases hfx : x ∈ { x | C ≤ ‖f i x‖₊ }
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
      ε : Real
      hε : LT.lt 0 ε
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x
      ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (g i x))).indicator (g i) x) ((se …
    -/
  · rw [Set.indicator_of_mem hfx, Set.indicator_of_mem, hx]
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
      ε : Real
      hε : LT.lt 0 ε
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x
      ⊢ Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (g i x))) x
    -/
    rwa [Set.mem_setOf, hx] at hfx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
      ε : Real
      hε : LT.lt 0 ε
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x)
      ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (g i x))).indicator (g i) x) ((se …
    -/
  · rw [Set.indicator_of_not_mem hfx, Set.indicator_of_not_mem]
    /-
      case neg.h
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hp : LE.le 1 p
      hp' : Ne p Top.top
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      h : ∀ (ε : Real), LT.lt 0 ε → Exists fun C => ∀ (i : ι), LE.le (MeasureTheory. …
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
      ε : Real
      hε : LT.lt 0 ε
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x)
      ⊢ Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (g i x))) x)
    -/
    rwa [Set.mem_setOf, hx] at hfx
    /-
      🎉 no goals
    -/


/-- This lemma is superseded by `UniformIntegrable.spec` which does not require measurability. -/
theorem UniformIntegrable.spec' (hp : p ≠ 0) (hp' : p ≠ ∞) (hf : ∀ i, StronglyMeasurable (f i))
    (hfu : UniformIntegrable f p μ) {ε : ℝ} (hε : 0 < ε) :
    ∃ C : ℝ≥0, ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨-, hfu, M, hM⟩ := hfu
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    ε : Real
    hε : LT.lt 0 ε
    hfu : MeasureTheory.UnifIntegrable f p μ
    M : NNReal
    hM : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑M
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨δ, hδpos, hδ⟩ := hfu hε
  obtain ⟨C, hC⟩ : ∃ C : ℝ≥0, ∀ i, μ { x | C ≤ ‖f i x‖₊ } ≤ ENNReal.ofReal δ := by
    by_contra hcon; push_neg at hcon
    choose ℐ hℐ using hcon
    lift δ to ℝ≥0 using hδpos.le
    have : ∀ C : ℝ≥0, C • (δ : ℝ≥0∞) ^ (1 / p.toReal) ≤ eLpNorm (f (ℐ C)) p μ := by
      intro C
      calc
        C • (δ : ℝ≥0∞) ^ (1 / p.toReal) ≤ C • μ { x | C ≤ ‖f (ℐ C) x‖₊ } ^ (1 / p.toReal) := by
          rw [ENNReal.smul_def, ENNReal.smul_def, smul_eq_mul, smul_eq_mul]
          simp_rw [ENNReal.ofReal_coe_nnreal] at hℐ
          refine mul_le_mul' le_rfl
            (ENNReal.rpow_le_rpow (hℐ C).le (one_div_nonneg.2 ENNReal.toReal_nonneg))
        _ ≤ eLpNorm ({ x | C ≤ ‖f (ℐ C) x‖₊ }.indicator (f (ℐ C))) p μ := by
          refine le_eLpNorm_of_bddBelow hp hp' _
            (measurableSet_le measurable_const (hf _).nnnorm.measurable)
            (Eventually.of_forall fun x hx => ?_)
          rwa [nnnorm_indicator_eq_indicator_nnnorm, Set.indicator_of_mem hx]
        _ ≤ eLpNorm (f (ℐ C)) p μ := eLpNorm_indicator_le _
    specialize this (2 * max M 1 * δ⁻¹ ^ (1 / p.toReal))
    rw [← ENNReal.coe_rpow_of_nonneg _ (one_div_nonneg.2 ENNReal.toReal_nonneg), ← ENNReal.coe_smul,
      smul_eq_mul, mul_assoc, NNReal.inv_rpow,
      inv_mul_cancel₀ (NNReal.rpow_pos (NNReal.coe_pos.1 hδpos)).ne.symm, mul_one, ENNReal.coe_mul,
      ← NNReal.inv_rpow] at this
    refine (lt_of_le_of_lt (le_trans
      (hM <| ℐ <| 2 * max M 1 * δ⁻¹ ^ (1 / p.toReal)) (le_max_left (M : ℝ≥0∞) 1))
        (lt_of_lt_of_le ?_ this)).ne rfl
    rw [← ENNReal.coe_one, ← ENNReal.coe_max, ← ENNReal.coe_mul, ENNReal.coe_lt_coe]
    exact lt_two_mul_self (lt_max_of_lt_right one_pos)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    ε : Real
    hε : LT.lt 0 ε
    hfu : MeasureTheory.UnifIntegrable f p μ
    M : NNReal
    hM : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑M
    δ : Real
    hδpos : LT.lt 0 δ
    hδ : ∀ (i : ι) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ) → …
    C : NNReal
    hC : ∀ (i : ι), LE.le (μ (setOf fun x => LE.le C (NNNorm.nnnorm (f i x)))) (EN …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  exact ⟨C, fun i => hδ i _ (measurableSet_le measurable_const (hf i).nnnorm.measurable) (hC i)⟩
  /-
    🎉 no goals
  -/


theorem UniformIntegrable.spec (hp : p ≠ 0) (hp' : p ≠ ∞) (hfu : UniformIntegrable f p μ) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ C : ℝ≥0, ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  set g : ι → α → β := fun i => (hfu.1 i).choose
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  have hgmeas : ∀ i, StronglyMeasurable (g i) := fun i => (Exists.choose_spec <| hfu.1 i).1
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  have hgunif : UniformIntegrable g p μ := hfu.ae_eq fun i => (Exists.choose_spec <| hfu.1 i).2
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgunif : MeasureTheory.UniformIntegrable g p μ
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨C, hC⟩ := hgunif.spec' hp hp' hgmeas hε
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgunif : MeasureTheory.UniformIntegrable g p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  refine ⟨C, fun i => le_trans (le_of_eq <| eLpNorm_congr_ae ?_) (hC i)⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgunif : MeasureTheory.UniformIntegrable g p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    i : ι
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((setOf fun x => LE.le C (NNNorm.nnnorm (f …
  -/
  filter_upwards [(Exists.choose_spec <| hfu.1 i).2] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    f : ι → α → β
    hp : Ne p 0
    hp' : Ne p Top.top
    hfu : MeasureTheory.UniformIntegrable f p μ
    ε : Real
    hε : LT.lt 0 ε
    g : ι → α → β := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgunif : MeasureTheory.UniformIntegrable g p μ
    C : NNReal
    hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
    i : ι
    x : α
    hx : Eq (f i x) (Exists.choose ⋯ x)
    ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (f i x))).indicator (f i) x) ((se …
  -/
  by_cases hfx : x ∈ { x | C ≤ ‖f i x‖₊ }
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      hp : Ne p 0
      hp' : Ne p Top.top
      hfu : MeasureTheory.UniformIntegrable f p μ
      ε : Real
      hε : LT.lt 0 ε
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgunif : MeasureTheory.UniformIntegrable g p μ
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x
      ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (f i x))).indicator (f i) x) ((se …
    -/
  · rw [Set.indicator_of_mem hfx, Set.indicator_of_mem, hx]
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      hp : Ne p 0
      hp' : Ne p Top.top
      hfu : MeasureTheory.UniformIntegrable f p μ
      ε : Real
      hε : LT.lt 0 ε
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgunif : MeasureTheory.UniformIntegrable g p μ
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x
      ⊢ Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (g i x))) x
    -/
    rwa [Set.mem_setOf, hx] at hfx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      hp : Ne p 0
      hp' : Ne p Top.top
      hfu : MeasureTheory.UniformIntegrable f p μ
      ε : Real
      hε : LT.lt 0 ε
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgunif : MeasureTheory.UniformIntegrable g p μ
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x)
      ⊢ Eq ((setOf fun x => LE.le C (NNNorm.nnnorm (f i x))).indicator (f i) x) ((se …
    -/
  · rw [Set.indicator_of_not_mem hfx, Set.indicator_of_not_mem]
    /-
      case neg.h
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      f : ι → α → β
      hp : Ne p 0
      hp' : Ne p Top.top
      hfu : MeasureTheory.UniformIntegrable f p μ
      ε : Real
      hε : LT.lt 0 ε
      g : ι → α → β := fun i => Exists.choose ⋯
      hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
      hgunif : MeasureTheory.UniformIntegrable g p μ
      C : NNReal
      hC : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C (NNNorm. …
      i : ι
      x : α
      hx : Eq (f i x) (Exists.choose ⋯ x)
      hfx : Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (f i x))) x)
      ⊢ Not (Membership.mem (setOf fun x => LE.le C (NNNorm.nnnorm (g i x))) x)
    -/
    rwa [Set.mem_setOf, hx] at hfx
    /-
      🎉 no goals
    -/


/-- The definition of uniform integrable in mathlib is equivalent to the definition commonly
found in literature. -/
theorem uniformIntegrable_iff [IsFiniteMeasure μ] (hp : 1 ≤ p) (hp' : p ≠ ∞) :
    UniformIntegrable f p μ ↔
      (∀ i, AEStronglyMeasurable (f i) μ) ∧
        ∀ ε : ℝ, 0 < ε → ∃ C : ℝ≥0,
          ∀ i, eLpNorm ({ x | C ≤ ‖f i x‖₊ }.indicator (f i)) p μ ≤ ENNReal.ofReal ε :=
  ⟨fun h => ⟨h.1, fun _ => h.spec (lt_of_lt_of_le zero_lt_one hp).ne.symm hp'⟩,
    fun h => uniformIntegrable_of hp hp' h.1 h.2⟩


/-- The averaging of a uniformly integrable sequence is also uniformly integrable. -/
theorem uniformIntegrable_average
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (hp : 1 ≤ p) {f : ℕ → α → E} (hf : UniformIntegrable f p μ) :
    UniformIntegrable (fun (n : ℕ) => (n : ℝ)⁻¹ • (∑ i ∈ Finset.range n, f i)) p μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : MeasureTheory.UniformIntegrable f p μ
    ⊢ MeasureTheory.UniformIntegrable (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset. …
  -/
  obtain ⟨hf₁, hf₂, hf₃⟩ := hf
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hp : LE.le 1 p
    f : Nat → α → E
    hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
    hf₂ : MeasureTheory.UnifIntegrable f p μ
    hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
    ⊢ MeasureTheory.UniformIntegrable (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset. …
  -/
  refine ⟨fun n => ?_, fun ε hε => ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable ((fun n => HSMul.hSMul (Inv.inv ↑n) ((Fin …
    -/
  · exact (Finset.aestronglyMeasurable_sum' _ fun i _ => hf₁ i).const_smul _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
    -/
  · obtain ⟨δ, hδ₁, hδ₂⟩ := hf₂ hε
    /-
      case intro.intro.refine_2.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      ⊢ Exists fun δ => Exists fun x => ∀ (i : Nat) (s : Set α), MeasurableSet s → L …
    -/
    refine ⟨δ, hδ₁, fun n s hs hle => ?_⟩
    /-
      case intro.intro.refine_2.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hle : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator ((fun n => HSMul.hSMul (Inv.inv ↑n …
    -/
    simp_rw [Finset.smul_sum, Finset.indicator_sum]
    /-
      case intro.intro.refine_2.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hle : LE.le (μ s) (ENNReal.ofReal δ)
      ⊢ LE.le (MeasureTheory.eLpNorm ((Finset.range n).sum fun i => s.indicator (HSM …
    -/
    refine le_trans (eLpNorm_sum_le (fun i _ => ((hf₁ i).const_smul _).indicator hs) hp) ?_
    have : ∀ i, s.indicator ((n : ℝ) ⁻¹ • f i) = (↑n : ℝ)⁻¹ • s.indicator (f i) :=
      fun i ↦ indicator_const_smul _ _ _
    /-
      case intro.intro.refine_2.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hle : LE.le (μ s) (ENNReal.ofReal δ)
      this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
      ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (s.indicator (HSM …
    -/
    simp_rw [this, eLpNorm_const_smul, ← Finset.mul_sum, nnnorm_inv, Real.nnnorm_natCast]
    /-
      case intro.intro.refine_2.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hle : LE.le (μ s) (ENNReal.ofReal δ)
      this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
      ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
    -/
    by_cases hn : (↑(↑n : ℝ≥0)⁻¹ : ℝ≥0∞) = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Eq (↑(Inv.inv ↑n)) 0
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
      -/
    · simp only [hn, zero_mul, zero_le]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₁ : LT.lt 0 δ
      hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
      n : Nat
      s : Set α
      hs : MeasurableSet s
      hle : LE.le (μ s) (ENNReal.ofReal δ)
      this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
      hn : Not (Eq (↑(Inv.inv ↑n)) 0)
      ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
    -/
    refine le_trans ?_ (?_ : ↑(↑n : ℝ≥0)⁻¹ * n • ENNReal.ofReal ε ≤ ENNReal.ofReal ε)
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
      -/
    · refine (ENNReal.mul_le_mul_left hn ENNReal.coe_ne_top).2 ?_
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (s.indicator (f i …
      -/
      conv_rhs => rw [← Finset.card_range n]
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (s.indicator (f i …
      -/
      exact Finset.sum_le_card_nsmul _ _ _ fun i _ => hδ₂ _ _ hs hle
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) (HSMul.hSMul n (ENNReal.ofReal ε))) (ENNRea …
      -/
    · simp only [ENNReal.coe_eq_zero, inv_eq_zero, Nat.cast_eq_zero] at hn
      rw [nsmul_eq_mul, ← mul_assoc, ENNReal.coe_inv, ENNReal.coe_natCast,
        ENNReal.inv_mul_cancel _ (ENNReal.natCast_ne_top _), one_mul]
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ₁ : LT.lt 0 δ
        hδ₂ : ∀ (i : Nat) (s : Set α), MeasurableSet s → LE.le (μ s) (ENNReal.ofReal δ …
        n : Nat
        s : Set α
        hs : MeasurableSet s
        hle : LE.le (μ s) (ENNReal.ofReal δ)
        this : ∀ (i : Nat), Eq (s.indicator (HSMul.hSMul (Inv.inv ↑n) (f i))) (HSMul.h …
        hn : Not (Eq n 0)
        ⊢ Ne (↑n) 0
      -/
      all_goals simpa only [Ne, Nat.cast_eq_zero]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      hf₃ : Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ⊢ Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((fun n => HSMul.h …
    -/
  · obtain ⟨C, hC⟩ := hf₃
    /-
      case intro.intro.refine_3.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      C : NNReal
      hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ⊢ Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((fun n => HSMul.h …
    -/
    simp_rw [Finset.smul_sum]
    /-
      case intro.intro.refine_3.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      C : NNReal
      hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      ⊢ Exists fun C => ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((Finset.range i). …
    -/
    refine ⟨C, fun n => (eLpNorm_sum_le (fun i _ => (hf₁ i).const_smul _) hp).trans ?_⟩
    /-
      case intro.intro.refine_3.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      C : NNReal
      hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      n : Nat
      ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (HSMul.hSMul (Inv …
    -/
    simp_rw [eLpNorm_const_smul, ← Finset.mul_sum, nnnorm_inv, Real.nnnorm_natCast]
    /-
      case intro.intro.refine_3.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      C : NNReal
      hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      n : Nat
      ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
    -/
    by_cases hn : (↑(↑n : ℝ≥0)⁻¹ : ℝ≥0∞) = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Eq (↑(Inv.inv ↑n)) 0
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
      -/
    · simp only [hn, zero_mul, zero_le]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hp : LE.le 1 p
      f : Nat → α → E
      hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf₂ : MeasureTheory.UnifIntegrable f p μ
      C : NNReal
      hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
      n : Nat
      hn : Not (Eq (↑(Inv.inv ↑n)) 0)
      ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
    -/
    refine le_trans ?_ (?_ : ↑(↑n : ℝ≥0)⁻¹ * (n • C : ℝ≥0∞) ≤ C)
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) ((Finset.range n).sum fun i => MeasureTheor …
      -/
    · refine (ENNReal.mul_le_mul_left hn ENNReal.coe_ne_top).2 ?_
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (f i) p μ) (HSMul …
      -/
      conv_rhs => rw [← Finset.card_range n]
      /-
        case neg.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le ((Finset.range n).sum fun i => MeasureTheory.eLpNorm (f i) p μ) (HSMul …
      -/
      exact Finset.sum_le_card_nsmul _ _ _ fun i _ => hC i
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Not (Eq (↑(Inv.inv ↑n)) 0)
        ⊢ LE.le (HMul.hMul (↑(Inv.inv ↑n)) (HSMul.hSMul n ↑C)) ↑C
      -/
    · simp only [ENNReal.coe_eq_zero, inv_eq_zero, Nat.cast_eq_zero] at hn
      rw [nsmul_eq_mul, ← mul_assoc, ENNReal.coe_inv, ENNReal.coe_natCast,
        ENNReal.inv_mul_cancel _ (ENNReal.natCast_ne_top _), one_mul]
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        hp : LE.le 1 p
        f : Nat → α → E
        hf₁ : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (f i) μ
        hf₂ : MeasureTheory.UnifIntegrable f p μ
        C : NNReal
        hC : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm (f i) p μ) ↑C
        n : Nat
        hn : Not (Eq n 0)
        ⊢ Ne (↑n) 0
      -/
      all_goals simpa only [Ne, Nat.cast_eq_zero]
      /-
        🎉 no goals
      -/


/-- The averaging of a uniformly integrable real-valued sequence is also uniformly integrable. -/
theorem uniformIntegrable_average_real (hp : 1 ≤ p) {f : ℕ → α → ℝ} (hf : UniformIntegrable f p μ) :
    UniformIntegrable (fun n => (∑ i ∈ Finset.range n, f i) / (n : α → ℝ)) p μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    f : Nat → α → Real
    hf : MeasureTheory.UniformIntegrable f p μ
    ⊢ MeasureTheory.UniformIntegrable (fun n => HDiv.hDiv ((Finset.range n).sum fu …
  -/
  convert uniformIntegrable_average hp hf using 2 with n
  /-
    case h.e'_6.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    f : Nat → α → Real
    hf : MeasureTheory.UniformIntegrable f p μ
    n : Nat
    ⊢ Eq (HDiv.hDiv ((Finset.range n).sum fun i => f i) ↑n) (HSMul.hSMul (Inv.inv  …
  -/
  ext x
  /-
    case h.e'_6.h.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    f : Nat → α → Real
    hf : MeasureTheory.UniformIntegrable f p μ
    n : Nat
    x : α
    ⊢ Eq (HDiv.hDiv ((Finset.range n).sum fun i => f i) (↑n) x) (HSMul.hSMul (Inv. …
  -/
  simp [div_eq_inv_mul]
  /-
    🎉 no goals
  -/


