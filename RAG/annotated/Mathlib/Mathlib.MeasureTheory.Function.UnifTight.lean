/-- A sequence of functions `f` is uniformly tight in `L^p` if for all `ε > 0`, there
exists some measurable set `s` with finite measure such that the Lp-norm of
`f i` restricted to `sᶜ` is smaller than `ε` for all `i`. -/
def UnifTight {_ : MeasurableSpace α} (f : ι → α → β) (p : ℝ≥0∞) (μ : Measure α) : Prop :=
  ∀ ⦃ε : ℝ≥0⦄, 0 < ε → ∃ s : Set α, μ s ≠ ∞ ∧ ∀ i, eLpNorm (sᶜ.indicator (f i)) p μ ≤ ε


theorem unifTight_iff_ennreal {_ : MeasurableSpace α} (f : ι → α → β) (p : ℝ≥0∞) (μ : Measure α) :
    UnifTight f p μ ↔ ∀ ⦃ε : ℝ≥0∞⦄, 0 < ε → ∃ s : Set α,
      μ s ≠ ∞ ∧ ∀ i, eLpNorm (sᶜ.indicator (f i)) p μ ≤ ε := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.UnifTight f p μ) (∀ ⦃ε : ENNReal⦄, LT.lt 0 ε → Exists fun …
  -/
  simp only [ENNReal.forall_ennreal, ENNReal.coe_pos]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.UnifTight f p μ) (And (∀ (r : NNReal), LT.lt 0 r → Exists …
  -/
  refine (and_iff_left ?_).symm
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ LT.lt 0 Top.top → Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le ( …
  -/
  simp only [zero_lt_top, le_top, implies_true, and_true, true_implies]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Exists fun s => Ne (μ s) Top.top
  -/
  use ∅; simpa only [measure_empty] using zero_ne_top
         /-
           🎉 no goals
         -/


theorem unifTight_iff_real {_ : MeasurableSpace α} (f : ι → α → β) (p : ℝ≥0∞) (μ : Measure α) :
    UnifTight f p μ ↔ ∀ ⦃ε : ℝ⦄, 0 < ε → ∃ s : Set α,
      μ s ≠ ∞ ∧ ∀ i, eLpNorm (sᶜ.indicator (f i)) p μ ≤ .ofReal ε := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.UnifTight f p μ) (∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun s  …
  -/
  refine ⟨fun hut rε hrε ↦ hut (Real.toNNReal_pos.mpr hrε), fun hut ε hε ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    hut : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun s => And (Ne (μ s) Top.top) (∀ (i : …
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  obtain ⟨s, hμs, hfε⟩ := hut hε
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    hut : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun s => And (Ne (μ s) Top.top) (∀ (i : …
    ε : NNReal
    hε : LT.lt 0 ε
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  use s, hμs; intro i
  /-
    case right
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedAddCommGroup β
    x✝ : MeasurableSpace α
    f : ι → α → β
    p : ENNReal
    μ : MeasureTheory.Measure α
    hut : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun s => And (Ne (μ s) Top.top) (∀ (i : …
    ε : NNReal
    hε : LT.lt 0 ε
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    i : ι
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f i)) p μ) ↑ε
  -/
  exact (hfε i).trans_eq (ofReal_coe_nnreal (p := ε))
  /-
    🎉 no goals
  -/


theorem eventually_cofinite_indicator (hf : UnifTight f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∀ᶠ s in μ.cofinite.smallSets, ∀ i, eLpNorm (s.indicator (f i)) p μ ≤ ε := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Filter.Eventually (fun s => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indic …
  -/
  by_cases hε_top : ε = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      ε : ENNReal
      hε : Ne ε 0
      hε_top : Eq ε Top.top
      ⊢ Filter.Eventually (fun s => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indic …
    -/
  · subst hε_top; simp
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
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    ε : ENNReal
    hε : Ne ε 0
    hε_top : Not (Eq ε Top.top)
    ⊢ Filter.Eventually (fun s => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indic …
  -/
  rcases hf (pos_iff_ne_zero.2 (toNNReal_ne_zero.mpr ⟨hε,hε_top⟩)) with ⟨s, hμs, hfs⟩
  /-
    case neg.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    ε : ENNReal
    hε : Ne ε 0
    hε_top : Not (Eq ε Top.top)
    s : Set α
    hμs : Ne (μ s) Top.top
    hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    ⊢ Filter.Eventually (fun s => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indic …
  -/
  refine (eventually_smallSets' ?_).2 ⟨sᶜ, ?_, fun i ↦ (coe_toNNReal hε_top) ▸ hfs i⟩
    /-
      case neg.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      ε : ENNReal
      hε : Ne ε 0
      hε_top : Not (Eq ε Top.top)
      s : Set α
      hμs : Ne (μ s) Top.top
      hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
      ⊢ ∀ ⦃s t : Set α⦄, HasSubset.Subset s t → (∀ (i : ι), LE.le (MeasureTheory.eLp …
    -/
  · intro s t hst ht i
    /-
      case neg.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      ε : ENNReal
      hε : Ne ε 0
      hε_top : Not (Eq ε Top.top)
      s✝ : Set α
      hμs : Ne (μ s✝) Top.top
      hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s✝).indicator ( …
      s t : Set α
      hst : HasSubset.Subset s t
      ht : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (t.indicator (f i)) p μ) ε
      i : ι
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) ε
    -/
    exact (eLpNorm_mono <| norm_indicator_le_of_subset hst _).trans (ht i)
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      ε : ENNReal
      hε : Ne ε 0
      hε_top : Not (Eq ε Top.top)
      s : Set α
      hμs : Ne (μ s) Top.top
      hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
      ⊢ Membership.mem μ.cofinite (HasCompl.compl s)
    -/
  · rwa [Measure.compl_mem_cofinite, lt_top_iff_ne_top]
    /-
      🎉 no goals
    -/


protected theorem exists_measurableSet_indicator (hf : UnifTight f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ s, MeasurableSet s ∧ μ s < ∞ ∧ ∀ i, eLpNorm (sᶜ.indicator (f i)) p μ ≤ ε :=
  let ⟨s, hμs, hsm, hfs⟩ := (hf.eventually_cofinite_indicator hε).exists_measurable_mem_of_smallSets
                          /-
                            α : Type u_1
                            β : Type u_2
                            ι : Type u_3
                            m : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            inst✝ : NormedAddCommGroup β
                            f : ι → α → β
                            p : ENNReal
                            hf : MeasureTheory.UnifTight f p μ
                            ε : ENNReal
                            hε : Ne ε 0
                            s : Set α
                            hμs : Membership.mem μ.cofinite s
                            hsm : MeasurableSet s
                            hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) ε
                            ⊢ ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (HasCompl.compl s)) …
                          -/
  ⟨sᶜ, hsm.compl, hμs, by rwa [compl_compl s]⟩
                          /-
                            🎉 no goals
                          -/


protected theorem add (hf : UnifTight f p μ) (hg : UnifTight g p μ)
    (hf_meas : ∀ i, AEStronglyMeasurable (f i) μ) (hg_meas : ∀ i, AEStronglyMeasurable (g i) μ) :
    UnifTight (f + g) p μ := fun ε hε ↦ by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  rcases exists_Lp_half β μ p (coe_ne_zero.mpr hε.ne') with ⟨η, hη_pos, hη⟩
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
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : NNReal
    hε : LT.lt 0 ε
    η : ENNReal
    hη_pos : LT.lt 0 η
    hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  by_cases hη_top : η = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f g : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      hg : MeasureTheory.UnifTight g p μ
      hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
      ε : NNReal
      hε : LT.lt 0 ε
      η : ENNReal
      hη_pos : LT.lt 0 η
      hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      hη_top : Eq η Top.top
      ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
    -/
  · replace hη := hη_top ▸ hη
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f g : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      hg : MeasureTheory.UnifTight g p μ
      hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
      ε : NNReal
      hε : LT.lt 0 ε
      η : ENNReal
      hη_pos : LT.lt 0 η
      hη_top : Eq η Top.top
      hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
    -/
    refine ⟨∅, (by measurability), fun i ↦ ?_⟩
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f g : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      hg : MeasureTheory.UnifTight g p μ
      hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
      ε : NNReal
      hε : LT.lt 0 ε
      η : ENNReal
      hη_pos : LT.lt 0 η
      hη_top : Eq η Top.top
      hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      i : ι
      ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl EmptyCollection.emptyCollectio …
    -/
    simp only [compl_empty, indicator_univ, Pi.add_apply]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f g : ι → α → β
      p : ENNReal
      hf : MeasureTheory.UnifTight f p μ
      hg : MeasureTheory.UnifTight g p μ
      hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
      ε : NNReal
      hε : LT.lt 0 ε
      η : ENNReal
      hη_pos : LT.lt 0 η
      hη_top : Eq η Top.top
      hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      i : ι
      ⊢ LE.le (MeasureTheory.eLpNorm (HAdd.hAdd (f i) (g i)) p μ) ↑ε
    -/
    exact (hη (f i) (g i) (hf_meas i) (hg_meas i) le_top le_top).le
    /-
      🎉 no goals
    -/
  obtain ⟨s, hμs, hsm, hfs, hgs⟩ :
      ∃ s ∈ μ.cofinite, MeasurableSet s ∧
        (∀ i, eLpNorm (s.indicator (f i)) p μ ≤ η) ∧
        (∀ i, eLpNorm (s.indicator (g i)) p μ ≤ η) :=
    ((hf.eventually_cofinite_indicator hη_pos.ne').and
      (hg.eventually_cofinite_indicator hη_pos.ne')).exists_measurable_mem_of_smallSets
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : NNReal
    hε : LT.lt 0 ε
    η : ENNReal
    hη_pos : LT.lt 0 η
    hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    hη_top : Not (Eq η Top.top)
    s : Set α
    hμs : Membership.mem μ.cofinite s
    hsm : MeasurableSet s
    hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) η
    hgs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indicator (g i)) p μ) η
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  refine ⟨sᶜ, ne_of_lt hμs, fun i ↦ ?_⟩
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ε : NNReal
    hε : LT.lt 0 ε
    η : ENNReal
    hη_pos : LT.lt 0 η
    hη : ∀ (f g : α → β), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
    hη_top : Not (Eq η Top.top)
    s : Set α
    hμs : Membership.mem μ.cofinite s
    hsm : MeasurableSet s
    hfs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indicator (f i)) p μ) η
    hgs : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm (s.indicator (g i)) p μ) η
    i : ι
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (HasCompl.compl s)).indicator  …
  -/
  have η_cast : ↑η.toNNReal = η := coe_toNNReal hη_top
  calc
    eLpNorm (indicator sᶜᶜ (f i + g i)) p μ
      = eLpNorm (indicator s (f i) + indicator s (g i)) p μ := by rw [compl_compl, indicator_add']
    _ ≤ ε := le_of_lt <|
      hη _ _ ((hf_meas i).indicator hsm) ((hg_meas i).indicator hsm)
        (η_cast ▸ hfs i) (η_cast ▸ hgs i)


protected theorem neg (hf : UnifTight f p μ) : UnifTight (-f) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    ⊢ MeasureTheory.UnifTight (Neg.neg f) p μ
  -/
  simp_rw [UnifTight, Pi.neg_apply, Set.indicator_neg', eLpNorm_neg]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    ⊢ ∀ ⦃ε : NNReal⦄, LT.lt 0 ε → Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι …
  -/
  exact hf
  /-
    🎉 no goals
  -/


protected theorem sub (hf : UnifTight f p μ) (hg : UnifTight g p μ)
    (hf_meas : ∀ i, AEStronglyMeasurable (f i) μ) (hg_meas : ∀ i, AEStronglyMeasurable (g i) μ) :
    UnifTight (f - g) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ MeasureTheory.UnifTight (HSub.hSub f g) p μ
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
    hf : MeasureTheory.UnifTight f p μ
    hg : MeasureTheory.UnifTight g p μ
    hf_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hg_meas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (g i) μ
    ⊢ MeasureTheory.UnifTight (HAdd.hAdd f (Neg.neg g)) p μ
  -/
  exact hf.add hg.neg hf_meas fun i => (hg_meas i).neg
  /-
    🎉 no goals
  -/


protected theorem aeeq (hf : UnifTight f p μ) (hfg : ∀ n, f n =ᵐ[μ] g n) :
    UnifTight g p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : ι → α → β
    p : ENNReal
    hf : MeasureTheory.UnifTight f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ⊢ MeasureTheory.UnifTight g p μ
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
    hf : MeasureTheory.UnifTight f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  obtain ⟨s, hμs, hfε⟩ := hf hε
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
    hf : MeasureTheory.UnifTight f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : NNReal
    hε : LT.lt 0 ε
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  refine ⟨s, hμs, fun n => (le_of_eq <| eLpNorm_congr_ae ?_).trans (hfε n)⟩
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
    hf : MeasureTheory.UnifTight f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : NNReal
    hε : LT.lt 0 ε
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    n : ι
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((HasCompl.compl s).indicator (g n)) ((Has …
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
    hf : MeasureTheory.UnifTight f p μ
    hfg : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyEq (f n) (g n)
    ε : NNReal
    hε : LT.lt 0 ε
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f …
    n : ι
    x : α
    hx : Eq (f n x) (g n x)
    ⊢ Eq ((HasCompl.compl s).indicator (g n) x) ((HasCompl.compl s).indicator (f n …
  -/
  simp only [indicator, mem_compl_iff, ite_not, hx]
  /-
    🎉 no goals
  -/


/-- If two functions agree a.e., then one is tight iff the other is tight. -/
theorem unifTight_congr_ae {g : ι → α → β} (hfg : ∀ n, f n =ᵐ[μ] g n) :
    UnifTight f p μ ↔ UnifTight g p μ :=
  ⟨fun h => h.aeeq hfg, fun h => h.aeeq fun i => (hfg i).symm⟩


/-- A constant sequence is tight. -/
theorem unifTight_const {g : α → β} (hp_ne_top : p ≠ ∞) (hg : Memℒp g p μ) :
    UnifTight (fun _ : ι => g) p μ := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    g : α → β
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ⊢ MeasureTheory.UnifTight (fun x => g) p μ
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
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  by_cases hε_top : ε = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      g : α → β
      hp_ne_top : Ne p Top.top
      hg : MeasureTheory.Memℒp g p μ
      ε : NNReal
      hε : LT.lt 0 ε
      hε_top : Eq (↑ε) Top.top
      ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
    -/
  · exact ⟨∅, (by measurability), fun _ => hε_top.symm ▸ le_top⟩
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
    g : α → β
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  obtain ⟨s, _, hμs, hgε⟩ := hg.exists_eLpNorm_indicator_compl_lt hp_ne_top (coe_ne_zero.mpr hε.ne')
  /-
    case neg.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    g : α → β
    hp_ne_top : Ne p Top.top
    hg : MeasureTheory.Memℒp g p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    s : Set α
    left✝ : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator g) p μ) ↑ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  exact ⟨s, ne_of_lt hμs, fun _ => hgε.le⟩
  /-
    🎉 no goals
  -/


/-- A single function is tight. -/
theorem unifTight_of_subsingleton [Subsingleton ι] (hp_top : p ≠ ∞)
    {f : ι → α → β} (hf : ∀ i, Memℒp (f i) p μ) : UnifTight f p μ := fun ε hε ↦ by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Subsingleton ι
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  by_cases hε_top : ε = ∞
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
      hp_top : Ne p Top.top
      f : ι → α → β
      hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
      ε : NNReal
      hε : LT.lt 0 ε
      hε_top : Eq (↑ε) Top.top
      ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
    -/
  · exact ⟨∅, by measurability, fun _ => hε_top.symm ▸ le_top⟩
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    hι : Nonempty ι
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  case neg => exact ⟨∅, (by measurability), fun i => False.elim <| hι <| Nonempty.intro i⟩
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    hι : Nonempty ι
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  cases' hι with i
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    i : ι
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  obtain ⟨s, _, hμs, hfε⟩ := (hf i).exists_eLpNorm_indicator_compl_lt hp_top (coe_ne_zero.2 hε.ne')
  /-
    case pos.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Subsingleton ι
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    i : ι
    s : Set α
    left✝ : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f i)) p μ) ↑ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  refine ⟨s, ne_of_lt hμs, fun j => ?_⟩
  /-
    case pos.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Subsingleton ι
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    i : ι
    s : Set α
    left✝ : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f i)) p μ) ↑ε
    j : ι
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f j)) p μ) ↑ε
  -/
  convert hfε.le
  /-
    🎉 no goals
  -/


/-- This lemma is less general than `MeasureTheory.unifTight_finite` which applies to
all sequences indexed by a finite type. -/
private theorem unifTight_fin (hp_top : p ≠ ∞) {n : ℕ} {f : Fin n → α → β}
    (hf : ∀ i, Memℒp (f i) p μ) : UnifTight f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_top : Ne p Top.top
    n : Nat
    f : Fin n → α → β
    hf : ∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ
    ⊢ MeasureTheory.UnifTight f p μ
  -/
  revert f
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
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
      hp_top : Ne p Top.top
      ⊢ ∀ {f : Fin 0 → α → β}, (∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ) → Meas …
    -/
  · intro f hf
    /-
      case zero
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_top : Ne p Top.top
      f : Fin 0 → α → β
      hf : ∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ
      ⊢ MeasureTheory.UnifTight f p μ
    -/
    have : Subsingleton (Fin Nat.zero) := subsingleton_fin_zero -- Porting note: Added this instance
    /-
      case zero
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_top : Ne p Top.top
      f : Fin 0 → α → β
      hf : ∀ (i : Fin 0), MeasureTheory.Memℒp (f i) p μ
      this : Subsingleton (Fin Nat.zero)
      ⊢ MeasureTheory.UnifTight f p μ
    -/
    exact unifTight_of_subsingleton hp_top hf
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
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
  -/
  by_cases hε_top : ε = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      p : ENNReal
      hp_top : Ne p Top.top
      n : Nat
      h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
      f : Fin (HAdd.hAdd n 1) → α → β
      hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
      ε : NNReal
      hε : LT.lt 0 ε
      hε_top : Eq (↑ε) Top.top
      ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
    -/
  · exact ⟨∅, (by measurability), fun _ => hε_top.symm ▸ le_top⟩
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
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
  -/
  let g : Fin n → α → β := fun k => f k
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    g : Fin n → α → β := fun k => f ↑↑k
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
  -/
  have hgLp : ∀ i, Memℒp (g i) p μ := fun i => hfLp i
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
  -/
  obtain ⟨S, hμS, hFε⟩ := h hgLp hε
  obtain ⟨s, _, hμs, hfε⟩ :=
    (hfLp n).exists_eLpNorm_indicator_compl_lt hp_top (coe_ne_zero.2 hε.ne')
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    S : Set α
    hμS : Ne (μ S) Top.top
    hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
    s : Set α
    left✝ : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Fin (HAdd.hAdd n 1)), LE.le ( …
  -/
  refine ⟨s ∪ S, (by measurability), fun i => ?_⟩
  /-
    case neg.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    p : ENNReal
    hp_top : Ne p Top.top
    n : Nat
    h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
    f : Fin (HAdd.hAdd n 1) → α → β
    hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    hε_top : Not (Eq (↑ε) Top.top)
    g : Fin n → α → β := fun k => f ↑↑k
    hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    S : Set α
    hμS : Ne (μ S) Top.top
    hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
    s : Set α
    left✝ : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
    i : Fin (HAdd.hAdd n 1)
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (Union.union s S)).indicator ( …
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
      hp_top : Ne p Top.top
      n : Nat
      h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
      f : Fin (HAdd.hAdd n 1) → α → β
      hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
      ε : NNReal
      hε : LT.lt 0 ε
      hε_top : Not (Eq (↑ε) Top.top)
      g : Fin n → α → β := fun k => f ↑↑k
      hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
      S : Set α
      hμS : Ne (μ S) Top.top
      hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
      s : Set α
      left✝ : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
      i : Fin (HAdd.hAdd n 1)
      hi : LT.lt (↑i) n
      ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (Union.union s S)).indicator ( …
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : LT.lt (↑i) n
        ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (Union.union s S)).indicator ( …
      -/
    · rw [compl_union, ← indicator_indicator]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : LT.lt (↑i) n
        ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator ((HasCompl.compl  …
      -/
      apply (eLpNorm_indicator_le _).trans
      /-
        case pos
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : LT.lt (↑i) n
        ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicator (g ⟨↑i, hi⟩)) p μ …
      -/
      exact hFε (Fin.castLT i hi)
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : LT.lt (↑i) n
        ⊢ Eq (f i) (g ⟨↑i, hi⟩)
      -/
    · simp only [Fin.coe_eq_castSucc, Fin.castSucc_mk, g]
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
      hp_top : Ne p Top.top
      n : Nat
      h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
      f : Fin (HAdd.hAdd n 1) → α → β
      hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
      ε : NNReal
      hε : LT.lt 0 ε
      hε_top : Not (Eq (↑ε) Top.top)
      g : Fin n → α → β := fun k => f ↑↑k
      hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
      S : Set α
      hμS : Ne (μ S) Top.top
      hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
      s : Set α
      left✝ : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
      i : Fin (HAdd.hAdd n 1)
      hi : Not (LT.lt (↑i) n)
      ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (Union.union s S)).indicator ( …
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : Not (LT.lt (↑i) n)
        ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl (Union.union s S)).indicator ( …
      -/
    · rw [compl_union, inter_comm, ← indicator_indicator]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        p : ENNReal
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : Not (LT.lt (↑i) n)
        ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicator ((HasCompl.compl  …
      -/
      exact (eLpNorm_indicator_le _).trans hfε.le
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
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
        hp_top : Ne p Top.top
        n : Nat
        h : ∀ {f : Fin n → α → β}, (∀ (i : Fin n), MeasureTheory.Memℒp (f i) p μ) → Me …
        f : Fin (HAdd.hAdd n 1) → α → β
        hfLp : ∀ (i : Fin (HAdd.hAdd n 1)), MeasureTheory.Memℒp (f i) p μ
        ε : NNReal
        hε : LT.lt 0 ε
        hε_top : Not (Eq (↑ε) Top.top)
        g : Fin n → α → β := fun k => f ↑↑k
        hgLp : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
        S : Set α
        hμS : Ne (μ S) Top.top
        hFε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl S).indicato …
        s : Set α
        left✝ : MeasurableSet s
        hμs : LT.lt (μ s) Top.top
        hfε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f ↑n)) p μ) ↑ε
        i : Fin (HAdd.hAdd n 1)
        hi : LE.le n ↑i
        hi' : LE.le (↑i) n
        ⊢ Eq i ↑n
      -/
      simp [← le_antisymm hi' hi]
      /-
        🎉 no goals
      -/


/-- A finite sequence of Lp functions is uniformly tight. -/
theorem unifTight_finite [Finite ι] (hp_top : p ≠ ∞) {f : ι → α → β}
    (hf : ∀ i, Memℒp (f i) p μ) : UnifTight f p μ := fun ε hε ↦ by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    p : ENNReal
    inst✝ : Finite ι
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  set g : Fin n → α → β := f ∘ hn.some.symm
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  obtain ⟨s, hμs, hfε⟩ := unifTight_fin hp_top hg hε
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : ι), LE.le (MeasureTheory.eLpN …
  -/
  refine ⟨s, hμs, fun i => ?_⟩
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
    hp_top : Ne p Top.top
    f : ι → α → β
    hf : ∀ (i : ι), MeasureTheory.Memℒp (f i) p μ
    ε : NNReal
    hε : LT.lt 0 ε
    n : Nat
    hn : Nonempty (Equiv ι (Fin n))
    g : Fin n → α → β := Function.comp f ⇑hn.some.symm
    hg : ∀ (i : Fin n), MeasureTheory.Memℒp (g i) p μ
    s : Set α
    hμs : Ne (μ s) Top.top
    hfε : ∀ (i : Fin n), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
    i : ι
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f i)) p μ) ↑ε
  -/
  simpa only [g, Function.comp_apply, Equiv.symm_apply_apply] using hfε (hn.some i)
  /-
    🎉 no goals
  -/


/-- Intermediate lemma for `unifTight_of_tendsto_Lp`. -/
private theorem unifTight_of_tendsto_Lp_zero (hp' : p ≠ ∞) (hf : ∀ n, Memℒp (f n) p μ)
    (hf_tendsto : Tendsto (fun n ↦ eLpNorm (f n) p μ) atTop (𝓝 0)) : UnifTight f p μ := fun ε hε ↦by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (f n) p μ) Filter. …
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  rw [ENNReal.tendsto_atTop_zero] at hf_tendsto
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  obtain ⟨N, hNε⟩ := hf_tendsto ε (by simpa only [gt_iff_lt, ENNReal.coe_pos])
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    N : Nat
    hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  let F : Fin N → α → β := fun n => f n
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    N : Nat
    hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
    F : Fin N → α → β := fun n => f ↑n
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  have hF : ∀ n, Memℒp (F n) p μ := fun n => hf n
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    N : Nat
    hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  obtain ⟨s, hμs, hFε⟩ := unifTight_fin hp' hF hε
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    N : Nat
    hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    s : Set α
    hμs : Ne (μ s) Top.top
    hFε : ∀ (i : Fin N), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
    ⊢ Exists fun s => And (Ne (μ s) Top.top) (∀ (i : Nat), LE.le (MeasureTheory.eL …
  -/
  refine ⟨s, hμs, fun n => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
    ε : NNReal
    hε : LT.lt 0 ε
    N : Nat
    hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
    F : Fin N → α → β := fun n => f ↑n
    hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
    s : Set α
    hμs : Ne (μ s) Top.top
    hFε : ∀ (i : Fin N), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
    n : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f n)) p μ) ↑ε
  -/
  by_cases hn : n < N
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : Nat → α → β
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
      ε : NNReal
      hε : LT.lt 0 ε
      N : Nat
      hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
      F : Fin N → α → β := fun n => f ↑n
      hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
      s : Set α
      hμs : Ne (μ s) Top.top
      hFε : ∀ (i : Fin N), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
      n : Nat
      hn : LT.lt n N
      ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f n)) p μ) ↑ε
    -/
  · exact hFε ⟨n, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : Nat → α → β
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hf_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n …
      ε : NNReal
      hε : LT.lt 0 ε
      N : Nat
      hNε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑ε
      F : Fin N → α → β := fun n => f ↑n
      hF : ∀ (n : Fin N), MeasureTheory.Memℒp (F n) p μ
      s : Set α
      hμs : Ne (μ s) Top.top
      hFε : ∀ (i : Fin N), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicato …
      n : Nat
      hn : Not (LT.lt n N)
      ⊢ LE.le (MeasureTheory.eLpNorm ((HasCompl.compl s).indicator (f n)) p μ) ↑ε
    -/
  · exact (eLpNorm_indicator_le _).trans (hNε n (not_lt.mp hn))
    /-
      🎉 no goals
    -/


/-- Convergence in Lp implies uniform tightness. -/
private theorem unifTight_of_tendsto_Lp (hp' : p ≠ ∞) (hf : ∀ n, Memℒp (f n) p μ)
    (hg : Memℒp g p μ) (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0)) :
    UnifTight f p μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hg : MeasureTheory.Memℒp g p μ
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    ⊢ MeasureTheory.UnifTight f p μ
  -/
  have : f = (fun _ => g) + fun n => f n - g := by ext1 n; simp
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    hg : MeasureTheory.Memℒp g p μ
    hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
    this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
    ⊢ MeasureTheory.UnifTight f p μ
  -/
  rw [this]
  refine UnifTight.add ?_ ?_ (fun _ => hg.aestronglyMeasurable)
      fun n => (hf n).1.sub hg.aestronglyMeasurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : Nat → α → β
      g : α → β
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hg : MeasureTheory.Memℒp g p μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
      ⊢ MeasureTheory.UnifTight (fun x => g) p μ
    -/
  · exact unifTight_const hp' hg
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : Nat → α → β
      g : α → β
      hp' : Ne p Top.top
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      hg : MeasureTheory.Memℒp g p μ
      hfg : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ)  …
      this : Eq f (HAdd.hAdd (fun x => g) fun n => HSub.hSub (f n) g)
      ⊢ MeasureTheory.UnifTight (fun n => HSub.hSub (f n) g) p μ
    -/
  · exact unifTight_of_tendsto_Lp_zero hp' (fun n => (hf n).sub hg) hfg
    /-
      🎉 no goals
    -/

/- Next we deal with the forward direction. The `Memℒp` and `TendstoInMeasure` hypotheses
are unwrapped and strengthened (by known lemmas) to also have the `StronglyMeasurable`
and a.e. convergence hypotheses. The bulk of the proof is done under these stronger hypotheses.-/


/-- Bulk of the proof under strengthened hypotheses. Invoked from `tendsto_Lp_of_tendsto_ae`. -/
private theorem tendsto_Lp_of_tendsto_ae_of_meas (hp : 1 ≤ p) (hp' : p ≠ ∞)
    {f : ℕ → α → β} {g : α → β} (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g)
    (hg' : Memℒp g p μ) (hui : UnifIntegrable f p μ) (hut : UnifTight f p μ)
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  rw [ENNReal.tendsto_atTop_zero]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le  …
  -/
  intro ε hε
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  by_cases hfinε : ε ≠ ∞; swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : Nat → α → β
      g : α → β
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hg' : MeasureTheory.Memℒp g p μ
      hui : MeasureTheory.UnifIntegrable f p μ
      hut : MeasureTheory.UnifTight f p μ
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : ENNReal
      hε : GT.gt ε 0
      hfinε : Not (Ne ε Top.top)
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
    -/
  · rw [not_ne_iff.mp hfinε]; exact ⟨0, fun n _ => le_top⟩
                              /-
                                🎉 no goals
                              -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      f : Nat → α → β
      g : α → β
      hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hg' : MeasureTheory.Memℒp g p μ
      hui : MeasureTheory.UnifIntegrable f p μ
      hut : MeasureTheory.UnifTight f p μ
      hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
      ε : ENNReal
      hε : GT.gt ε 0
      hfinε : Ne ε Top.top
      hμ : Eq μ 0
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
    -/
  · rw [hμ]; use 0; intro n _; rw [eLpNorm_measure_zero]; exact zero_le ε
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hε' : 0 < ε / 3 := ENNReal.div_pos hε.ne' (coe_ne_top)
  -- use tightness to divide the domain into interior and exterior
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨Eg, hmEg, hμEg, hgε⟩ := Memℒp.exists_eLpNorm_indicator_compl_lt hp' hg' hε'.ne' --hrε'
  /-
    case neg.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨Ef, hmEf, hμEf, hfε⟩ := hut.exists_measurableSet_indicator hε'.ne'
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hmE := hmEf.union hmEg
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    hmE : MeasurableSet (Union.union Ef Eg)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hfmE := (measure_union_le Ef Eg).trans_lt (add_lt_top.mpr ⟨hμEf, hμEg⟩)
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    hmE : MeasurableSet (Union.union Ef Eg)
    hfmE : LT.lt (μ (Union.union Ef Eg)) Top.top
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  set E : Set α := Ef ∪ Eg
  -- use uniform integrability to get control on the limit over E
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hgE' := Memℒp.restrict E hg'
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have huiE := hui.restrict  E
  have hfgE : (∀ᵐ x ∂(μ.restrict E), Tendsto (fun n => f n x) atTop (𝓝 (g x))) :=
    ae_restrict_of_ae hfg
  -- `tendsto_Lp_of_tendsto_ae_of_meas` needs to
  -- synthesize an argument `[IsFiniteMeasure (μ.restrict E)]`.
  -- It is enough to have in the context a term of `Fact (μ E < ∞)`, which is our `ffmE` below,
  -- which is automatically fed into `Restrict.isFiniteInstance`.
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have ffmE : Fact _ := { out := hfmE }
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  have hInner := tendsto_Lp_finite_of_tendsto_ae_of_meas hp hp' hf hg hgE' huiE hfgE
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p  …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  rw [ENNReal.tendsto_atTop_zero] at hInner
  -- get a sufficiently large N for given ε, and consider any n ≥ N
  /-
    case neg.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  obtain ⟨N, hfngε⟩ := hInner (ε / 3) hε'
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → …
    N : Nat
    hfngε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  use N; intro n hn
  -- get interior estimates
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → …
    N : Nat
    hfngε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) …
    n : Nat
    hn : GE.ge n N
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) ε
  -/
  have hmfngE : AEStronglyMeasurable _ μ := (((hf n).sub hg).indicator hmE).aestronglyMeasurable
  have hfngEε := calc
    eLpNorm (E.indicator (f n - g)) p μ
      = eLpNorm (f n - g) p (μ.restrict E) := eLpNorm_indicator_eq_eLpNorm_restrict hmE
    _ ≤ ε / 3                            := hfngε n hn
  -- get exterior estimates
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → …
    N : Nat
    hfngε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) …
    n : Nat
    hn : GE.ge n N
    hmfngE : MeasureTheory.AEStronglyMeasurable (E.indicator (HSub.hSub (f n) g)) μ
    hfngEε : LE.le (MeasureTheory.eLpNorm (E.indicator (HSub.hSub (f n) g)) p μ) ( …
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) ε
  -/
  have hmgEc : AEStronglyMeasurable _ μ := (hg.indicator hmE.compl).aestronglyMeasurable
  have hgEcε := calc
    eLpNorm (Eᶜ.indicator g) p μ
      ≤ eLpNorm (Efᶜ.indicator (Egᶜ.indicator g)) p μ := by
        unfold E; rw [compl_union, ← indicator_indicator]
    _ ≤ eLpNorm (Egᶜ.indicator g) p μ := eLpNorm_indicator_le _
    _ ≤ ε / 3 := hgε.le
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : ENNReal
    hε : GT.gt ε 0
    hfinε : Ne ε Top.top
    hμ : Not (Eq μ 0)
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    Eg : Set α
    hmEg : MeasurableSet Eg
    hμEg : LT.lt (μ Eg) Top.top
    hgε : LT.lt (MeasureTheory.eLpNorm ((HasCompl.compl Eg).indicator g) p μ) (HDi …
    Ef : Set α
    hmEf : MeasurableSet Ef
    hμEf : LT.lt (μ Ef) Top.top
    hfε : ∀ (i : Nat), LE.le (MeasureTheory.eLpNorm ((HasCompl.compl Ef).indicator …
    E : Set α := Union.union Ef Eg
    hmE : MeasurableSet E
    hfmE : LT.lt (μ E) Top.top
    hgE' : MeasureTheory.Memℒp g p (μ.restrict E)
    huiE : MeasureTheory.UnifIntegrable f p (μ.restrict E)
    hfgE : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ffmE : Fact (LT.lt (μ E) Top.top)
    hInner : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → …
    N : Nat
    hfngε : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) …
    n : Nat
    hn : GE.ge n N
    hmfngE : MeasureTheory.AEStronglyMeasurable (E.indicator (HSub.hSub (f n) g)) μ
    hfngEε : LE.le (MeasureTheory.eLpNorm (E.indicator (HSub.hSub (f n) g)) p μ) ( …
    hmgEc : MeasureTheory.AEStronglyMeasurable ((HasCompl.compl E).indicator g) μ
    hgEcε : LE.le (MeasureTheory.eLpNorm ((HasCompl.compl E).indicator g) p μ) (HD …
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) ε
  -/
  have hmfnEc : AEStronglyMeasurable _ μ := ((hf n).indicator hmE.compl).aestronglyMeasurable
  have hfnEcε : eLpNorm (Eᶜ.indicator (f n)) p μ ≤ ε / 3 := calc
    eLpNorm (Eᶜ.indicator (f n)) p μ
      ≤ eLpNorm (Egᶜ.indicator (Efᶜ.indicator (f n))) p μ := by
        unfold E; rw [compl_union, inter_comm, ← indicator_indicator]
    _ ≤ eLpNorm (Efᶜ.indicator (f n)) p μ := eLpNorm_indicator_le _
    _ ≤ ε / 3 := hfε n
  have hmfngEc : AEStronglyMeasurable _ μ :=
    (((hf n).sub hg).indicator hmE.compl).aestronglyMeasurable
  have hfngEcε := calc
    eLpNorm (Eᶜ.indicator (f n - g)) p μ
      = eLpNorm (Eᶜ.indicator (f n) - Eᶜ.indicator g) p μ := by
        rw [(Eᶜ.indicator_sub' _ _)]
    _ ≤ eLpNorm (Eᶜ.indicator (f n)) p μ + eLpNorm (Eᶜ.indicator g) p μ := by
        apply eLpNorm_sub_le (by assumption) (by assumption) hp
    _ ≤ ε / 3 + ε / 3 := add_le_add hfnEcε hgEcε
  -- finally, combine interior and exterior estimates
  calc
    eLpNorm (f n - g) p μ
      = eLpNorm (Eᶜ.indicator (f n - g) + E.indicator (f n - g)) p μ := by
        congr; exact (E.indicator_compl_add_self _).symm
    _ ≤ eLpNorm (indicator Eᶜ (f n - g)) p μ + eLpNorm (indicator E (f n - g)) p μ := by
        apply eLpNorm_add_le (by assumption) (by assumption) hp
    _ ≤ (ε / 3 + ε / 3) + ε / 3 := add_le_add hfngEcε hfngEε
    _ = ε := by simp only [ENNReal.add_thirds] --ENNReal.add_thirds ε


/-- Lemma used in `tendsto_Lp_of_tendsto_ae`. -/
private theorem ae_tendsto_ae_congr {f f' : ℕ → α → β} {g g' : α → β}
    (hff' : ∀ (n : ℕ), f n =ᵐ[μ] f' n) (hgg' : g =ᵐ[μ] g')
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    ∀ᵐ x ∂μ, Tendsto (fun n => f' n x) atTop (𝓝 (g' x)) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    f f' : Nat → α → β
    g g' : α → β
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (f' n)
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => f' n x) Filter.atTop (n …
  -/
  have hff'' := eventually_countable_forall.mpr hff'
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    f f' : Nat → α → β
    g g' : α → β
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (f' n)
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hff'' : Filter.Eventually (fun x => ∀ (i : Nat), Eq (f i x) (f' i x)) (Measure …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => f' n x) Filter.atTop (n …
  -/
  filter_upwards [hff'', hgg', hfg] with x hff'x hgg'x hfgx
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    f f' : Nat → α → β
    g g' : α → β
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (f' n)
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hff'' : Filter.Eventually (fun x => ∀ (i : Nat), Eq (f i x) (f' i x)) (Measure …
    x : α
    hff'x : ∀ (i : Nat), Eq (f i x) (f' i x)
    hgg'x : Eq (g x) (g' x)
    hfgx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    ⊢ Filter.Tendsto (fun n => f' n x) Filter.atTop (nhds (g' x))
  -/
  apply Tendsto.congr hff'x
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    f f' : Nat → α → β
    g g' : α → β
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (f' n)
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hff'' : Filter.Eventually (fun x => ∀ (i : Nat), Eq (f i x) (f' i x)) (Measure …
    x : α
    hff'x : ∀ (i : Nat), Eq (f i x) (f' i x)
    hgg'x : Eq (g x) (g' x)
    hfgx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    ⊢ Filter.Tendsto (fun x_1 => f x_1 x) Filter.atTop (nhds (g' x))
  -/
  rw [← hgg'x]; exact hfgx
                /-
                  🎉 no goals
                -/


/-- Forward direction of Vitali's convergnece theorem, with a.e. instead of InMeasure convergence.-/
theorem tendsto_Lp_of_tendsto_ae (hp : 1 ≤ p) (hp' : p ≠ ∞)
    {f : ℕ → α → β} {g : α → β} (haef : ∀ n, AEStronglyMeasurable (f n) μ)
    (hg' : Memℒp g p μ) (hui : UnifIntegrable f p μ) (hut : UnifTight f p μ)
    (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  -- come up with an a.e. equal strongly measurable replacement `f` for `g`
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hf := fun n => (haef n).stronglyMeasurable_mk
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hff' := fun n => (haef n).ae_eq_mk (μ := μ)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hui' := hui.ae_eq hff'
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hut' := hut.aeeq hff'
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    hut' : MeasureTheory.UnifTight (fun n => MeasureTheory.AEStronglyMeasurable.mk …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hg := hg'.aestronglyMeasurable.stronglyMeasurable_mk
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    hut' : MeasureTheory.UnifTight (fun n => MeasureTheory.AEStronglyMeasurable.mk …
    hg : MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk g …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hgg' := hg'.aestronglyMeasurable.ae_eq_mk (μ := μ)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    hut' : MeasureTheory.UnifTight (fun n => MeasureTheory.AEStronglyMeasurable.mk …
    hg : MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk g …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g (MeasureTheory.AEStronglyMeasurable …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hg'' := hg'.ae_eq hgg'
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    hut' : MeasureTheory.UnifTight (fun n => MeasureTheory.AEStronglyMeasurable.mk …
    hg : MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk g …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g (MeasureTheory.AEStronglyMeasurable …
    hg'' : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk g ⋯) p μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have haefg' := ae_tendsto_ae_congr hff' hgg' hfg
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hui' : MeasureTheory.UnifIntegrable (fun n => MeasureTheory.AEStronglyMeasurab …
    hut' : MeasureTheory.UnifTight (fun n => MeasureTheory.AEStronglyMeasurable.mk …
    hg : MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk g …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g (MeasureTheory.AEStronglyMeasurable …
    hg'' : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk g ⋯) p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  set f' := fun n => (haef n).mk (μ := μ)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    hg : MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk g …
    hgg' : (MeasureTheory.ae μ).EventuallyEq g (MeasureTheory.AEStronglyMeasurable …
    hg'' : MeasureTheory.Memℒp (MeasureTheory.AEStronglyMeasurable.mk g ⋯) p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    f' : Nat → α → β := fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯
    hui' : MeasureTheory.UnifIntegrable f' p μ
    hut' : MeasureTheory.UnifTight f' p μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  set g' := hg'.aestronglyMeasurable.mk (μ := μ)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    f' : Nat → α → β := fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯
    hui' : MeasureTheory.UnifIntegrable f' p μ
    hut' : MeasureTheory.UnifTight f' p μ
    g' : α → β := MeasureTheory.AEStronglyMeasurable.mk g ⋯
    hg : MeasureTheory.StronglyMeasurable g'
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'' : MeasureTheory.Memℒp g' p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have haefg (n : ℕ) : f n - g =ᵐ[μ] f' n - g' := (hff' n).sub hgg'
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    f' : Nat → α → β := fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯
    hui' : MeasureTheory.UnifIntegrable f' p μ
    hut' : MeasureTheory.UnifTight f' p μ
    g' : α → β := MeasureTheory.AEStronglyMeasurable.mk g ⋯
    hg : MeasureTheory.StronglyMeasurable g'
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'' : MeasureTheory.Memℒp g' p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    haefg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (HSub.hSub (f n) g) (HS …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  have hsnfg (n : ℕ) := eLpNorm_congr_ae (p := p) (haefg n)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    f' : Nat → α → β := fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯
    hui' : MeasureTheory.UnifIntegrable f' p μ
    hut' : MeasureTheory.UnifTight f' p μ
    g' : α → β := MeasureTheory.AEStronglyMeasurable.mk g ⋯
    hg : MeasureTheory.StronglyMeasurable g'
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'' : MeasureTheory.Memℒp g' p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    haefg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (HSub.hSub (f n) g) (HS …
    hsnfg : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measu …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  apply Filter.Tendsto.congr (fun n => (hsnfg n).symm)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    f : Nat → α → β
    g : α → β
    haef : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg' : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    hf : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMe …
    hff' : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (f n) (MeasureTheory.AES …
    f' : Nat → α → β := fun n => MeasureTheory.AEStronglyMeasurable.mk (f n) ⋯
    hui' : MeasureTheory.UnifIntegrable f' p μ
    hut' : MeasureTheory.UnifTight f' p μ
    g' : α → β := MeasureTheory.AEStronglyMeasurable.mk g ⋯
    hg : MeasureTheory.StronglyMeasurable g'
    hgg' : (MeasureTheory.ae μ).EventuallyEq g g'
    hg'' : MeasureTheory.Memℒp g' p μ
    haefg' : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.AE …
    haefg : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (HSub.hSub (f n) g) (HS …
    hsnfg : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) (Measu …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f' n) g') p μ) Fi …
  -/
  exact tendsto_Lp_of_tendsto_ae_of_meas hp hp' hf hg hg'' hui' hut' haefg'
  /-
    🎉 no goals
  -/


/-- Forward direction of Vitali's convergence theorem:
if `f` is a sequence of uniformly integrable, uniformly tight functions that converge in
measure to some function `g` in a finite measure space, then `f` converge in Lp to `g`. -/
theorem tendsto_Lp_of_tendstoInMeasure (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hg : Memℒp g p μ)
    (hui : UnifIntegrable f p μ) (hut : UnifTight f p μ)
    (hfg : TendstoInMeasure μ f atTop g) : Tendsto (fun n ↦ eLpNorm (f n - g) p μ) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) p μ) Filt …
  -/
  refine tendsto_of_subseq_tendsto fun ns hns => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : NormedAddCommGroup β
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : Nat → α → β
    g : α → β
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hg : MeasureTheory.Memℒp g p μ
    hui : MeasureTheory.UnifIntegrable f p μ
    hut : MeasureTheory.UnifTight f p μ
    hfg : MeasureTheory.TendstoInMeasure μ f Filter.atTop g
    ns : Nat → Nat
    hns : Filter.Tendsto ns Filter.atTop Filter.atTop
    ⊢ Exists fun ms => Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ( …
  -/
  obtain ⟨ms, _, hms'⟩ := TendstoInMeasure.exists_seq_tendsto_ae fun ε hε => (hfg ε hε).comp hns
  exact ⟨ms,
    tendsto_Lp_of_tendsto_ae hp hp' (fun _ => hf _) hg
      (fun ε hε => -- `UnifIntegrable` on a subsequence
        let ⟨δ, hδ, hδ'⟩ := hui hε
        ⟨δ, hδ, fun i s hs hμs => hδ' _ s hs hμs⟩)
      (fun ε hε => -- `UnifTight` on a subsequence
        let ⟨s, hμs, hfε⟩ := hut hε
        ⟨s, hμs, fun i => hfε _⟩)
      hms'⟩


/-- **Vitali's convergence theorem** (non-finite measure version).

A sequence of functions `f` converges to `g` in Lp
if and only if it is uniformly integrable, uniformly tight and converges to `g` in measure. -/
theorem tendstoInMeasure_iff_tendsto_Lp (hp : 1 ≤ p) (hp' : p ≠ ∞)
    (hf : ∀ n, Memℒp (f n) p μ) (hg : Memℒp g p μ) :
    TendstoInMeasure μ f atTop g ∧ UnifIntegrable f p μ ∧ UnifTight f p μ
      ↔ Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (𝓝 0) where
  mp h := tendsto_Lp_of_tendstoInMeasure hp hp' (fun n => (hf n).1) hg h.2.1 h.2.2 h.1
  mpr h := ⟨tendstoInMeasure_of_tendsto_eLpNorm (lt_of_lt_of_le zero_lt_one hp).ne'
        (fun n => (hf n).aestronglyMeasurable) hg.aestronglyMeasurable h,
      unifIntegrable_of_tendsto_Lp hp hp' hf hg h,
      unifTight_of_tendsto_Lp hp' hf hg h⟩


