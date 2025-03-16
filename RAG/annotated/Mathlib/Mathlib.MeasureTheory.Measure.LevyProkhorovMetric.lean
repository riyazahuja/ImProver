/-- The Lévy-Prokhorov edistance between measures:
`d(μ,ν) = inf {r ≥ 0 | ∀ B, μ B ≤ ν Bᵣ + r ∧ ν B ≤ μ Bᵣ + r}`. -/
noncomputable def levyProkhorovEDist (μ ν : Measure Ω) : ℝ≥0∞ :=
  sInf {ε | ∀ B, MeasurableSet B →
            μ B ≤ ν (thickening ε.toReal B) + ε ∧ ν B ≤ μ (thickening ε.toReal B) + ε}

/- This result is not placed in earlier more generic files, since it is rather specialized;
it mixes measure and metric in a very particular way. -/

lemma meas_le_of_le_of_forall_le_meas_thickening_add {ε₁ ε₂ : ℝ≥0∞} (μ ν : Measure Ω)
    (h_le : ε₁ ≤ ε₂) {B : Set Ω} (hε₁ : μ B ≤ ν (thickening ε₁.toReal B) + ε₁) :
    μ B ≤ ν (thickening ε₂.toReal B) + ε₂ := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    ε₁ ε₂ : ENNReal
    μ ν : MeasureTheory.Measure Ω
    h_le : LE.le ε₁ ε₂
    B : Set Ω
    hε₁ : LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε₁.toReal B)) ε₁)
    ⊢ LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε₂.toReal B)) ε₂)
  -/
  by_cases ε_top : ε₂ = ∞
  · simp only [ne_eq, FiniteMeasure.ennreal_coeFn_eq_coeFn_toMeasure, ε_top, top_toReal,
                add_top, le_top]
  /-
    case neg
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    ε₁ ε₂ : ENNReal
    μ ν : MeasureTheory.Measure Ω
    h_le : LE.le ε₁ ε₂
    B : Set Ω
    hε₁ : LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε₁.toReal B)) ε₁)
    ε_top : Not (Eq ε₂ Top.top)
    ⊢ LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε₂.toReal B)) ε₂)
  -/
  apply hε₁.trans (add_le_add ?_ h_le)
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    ε₁ ε₂ : ENNReal
    μ ν : MeasureTheory.Measure Ω
    h_le : LE.le ε₁ ε₂
    B : Set Ω
    hε₁ : LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε₁.toReal B)) ε₁)
    ε_top : Not (Eq ε₂ Top.top)
    ⊢ LE.le (ν (Metric.thickening ε₁.toReal B)) (ν (Metric.thickening ε₂.toReal B))
  -/
  exact measure_mono (μ := ν) (thickening_mono (toReal_mono ε_top h_le) B)
  /-
    🎉 no goals
  -/


lemma left_measure_le_of_levyProkhorovEDist_lt {μ ν : Measure Ω} {c : ℝ≥0∞}
    (h : levyProkhorovEDist μ ν < c) {B : Set Ω} (B_mble : MeasurableSet B) :
    μ B ≤ ν (thickening c.toReal B) + c := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    c : ENNReal
    h : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) c
    B : Set Ω
    B_mble : MeasurableSet B
    ⊢ LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening c.toReal B)) c)
  -/
  obtain ⟨c', ⟨hc', lt_c⟩⟩ := sInf_lt_iff.mp h
  /-
    case intro.intro
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    c : ENNReal
    h : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) c
    B : Set Ω
    B_mble : MeasurableSet B
    c' : ENNReal
    hc' : Membership.mem (setOf fun ε => ∀ (B : Set Ω), MeasurableSet B → And (LE. …
    lt_c : LT.lt c' c
    ⊢ LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening c.toReal B)) c)
  -/
  exact meas_le_of_le_of_forall_le_meas_thickening_add μ ν lt_c.le (hc' B B_mble).1
  /-
    🎉 no goals
  -/


lemma right_measure_le_of_levyProkhorovEDist_lt {μ ν : Measure Ω} {c : ℝ≥0∞}
    (h : levyProkhorovEDist μ ν < c) {B : Set Ω} (B_mble : MeasurableSet B) :
    ν B ≤ μ (thickening c.toReal B) + c := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    c : ENNReal
    h : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) c
    B : Set Ω
    B_mble : MeasurableSet B
    ⊢ LE.le (ν B) (HAdd.hAdd (μ (Metric.thickening c.toReal B)) c)
  -/
  obtain ⟨c', ⟨hc', lt_c⟩⟩ := sInf_lt_iff.mp h
  /-
    case intro.intro
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    c : ENNReal
    h : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) c
    B : Set Ω
    B_mble : MeasurableSet B
    c' : ENNReal
    hc' : Membership.mem (setOf fun ε => ∀ (B : Set Ω), MeasurableSet B → And (LE. …
    lt_c : LT.lt c' c
    ⊢ LE.le (ν B) (HAdd.hAdd (μ (Metric.thickening c.toReal B)) c)
  -/
  exact meas_le_of_le_of_forall_le_meas_thickening_add ν μ lt_c.le (hc' B B_mble).2
  /-
    🎉 no goals
  -/


/-- A general sufficient condition for bounding `levyProkhorovEDist` from above. -/
lemma levyProkhorovEDist_le_of_forall_add_pos_le (μ ν : Measure Ω) (δ : ℝ≥0∞)
    (h : ∀ ε B, 0 < ε → ε < ∞ → MeasurableSet B →
      μ B ≤ ν (thickening (δ + ε).toReal B) + δ + ε ∧
      ν B ≤ μ (thickening (δ + ε).toReal B) + δ + ε) :
    levyProkhorovEDist μ ν ≤ δ := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) δ
  -/
  apply ENNReal.le_of_forall_pos_le_add
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt δ Top.top → LE.le (MeasureTheory.levyProkh …
  -/
  intro ε hε _
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ε : NNReal
    hε : LT.lt 0 ε
    a✝ : LT.lt δ Top.top
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) (HAdd.hAdd δ ↑ε)
  -/
  by_cases ε_top : ε = ∞
    /-
      case pos
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : PseudoEMetricSpace Ω
      μ ν : MeasureTheory.Measure Ω
      δ : ENNReal
      h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
      ε : NNReal
      hε : LT.lt 0 ε
      a✝ : LT.lt δ Top.top
      ε_top : Eq (↑ε) Top.top
      ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) (HAdd.hAdd δ ↑ε)
    -/
  · simp only [ε_top, add_top, le_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ε : NNReal
    hε : LT.lt 0 ε
    a✝ : LT.lt δ Top.top
    ε_top : Not (Eq (↑ε) Top.top)
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) (HAdd.hAdd δ ↑ε)
  -/
  apply sInf_le
  /-
    case neg.a
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ε : NNReal
    hε : LT.lt 0 ε
    a✝ : LT.lt δ Top.top
    ε_top : Not (Eq (↑ε) Top.top)
    ⊢ Membership.mem (setOf fun ε => ∀ (B : Set Ω), MeasurableSet B → And (LE.le ( …
  -/
  intro B B_mble
  /-
    case neg.a
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B …
    ε : NNReal
    hε : LT.lt 0 ε
    a✝ : LT.lt δ Top.top
    ε_top : Not (Eq (↑ε) Top.top)
    B : Set Ω
    B_mble : MeasurableSet B
    ⊢ And (LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening (HAdd.hAdd δ ↑ε).toReal B) …
  -/
  simpa only [add_assoc] using h ε B (coe_pos.mpr hε) coe_lt_top B_mble
  /-
    🎉 no goals
  -/


/-- A simple general sufficient condition for bounding `levyProkhorovEDist` from above. -/
lemma levyProkhorovEDist_le_of_forall (μ ν : Measure Ω) (δ : ℝ≥0∞)
    (h : ∀ ε B, δ < ε → ε < ∞ → MeasurableSet B →
        μ B ≤ ν (thickening ε.toReal B) + ε ∧ ν B ≤ μ (thickening ε.toReal B) + ε) :
    levyProkhorovEDist μ ν ≤ δ := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) δ
  -/
  by_cases δ_top : δ = ∞
    /-
      case pos
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : PseudoEMetricSpace Ω
      μ ν : MeasureTheory.Measure Ω
      δ : ENNReal
      h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
      δ_top : Eq δ Top.top
      ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) δ
    -/
  · simp only [δ_top, add_top, le_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    δ_top : Not (Eq δ Top.top)
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) δ
  -/
  apply levyProkhorovEDist_le_of_forall_add_pos_le
  /-
    case neg.h
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    δ_top : Not (Eq δ Top.top)
    ⊢ ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B → …
  -/
  intro x B x_pos x_lt_top B_mble
  simpa only [← add_assoc] using h (δ + x) B (ENNReal.lt_add_right δ_top x_pos.ne.symm)
    (by simp only [add_lt_top, Ne.lt_top δ_top, x_lt_top, and_self]) B_mble


lemma levyProkhorovEDist_le_max_measure_univ (μ ν : Measure Ω) :
    levyProkhorovEDist μ ν ≤ max (μ univ) (ν univ) := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) (Max.max (μ Set.univ) (ν Set.un …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  refine sInf_le fun B _ ↦ ⟨?_, ?_⟩ <;> apply le_add_left <;> simp [measure_mono]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma levyProkhorovEDist_lt_top (μ ν : Measure Ω) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    levyProkhorovEDist μ ν < ∞ :=
                                                              /-
                                                                Ω : Type u_1
                                                                inst✝³ : MeasurableSpace Ω
                                                                inst✝² : PseudoEMetricSpace Ω
                                                                μ ν : MeasureTheory.Measure Ω
                                                                inst✝¹ : MeasureTheory.IsFiniteMeasure μ
                                                                inst✝ : MeasureTheory.IsFiniteMeasure ν
                                                                ⊢ LT.lt (Max.max (μ Set.univ) (ν Set.univ)) Top.top
                                                              -/
  (levyProkhorovEDist_le_max_measure_univ μ ν).trans_lt <| by simp [measure_lt_top]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma levyProkhorovEDist_ne_top (μ ν : Measure Ω) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    levyProkhorovEDist μ ν ≠ ∞ := (levyProkhorovEDist_lt_top μ ν).ne


lemma levyProkhorovEDist_self (μ : Measure Ω) :
    levyProkhorovEDist μ μ = 0 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.levyProkhorovEDist μ μ) 0
  -/
  rw [← nonpos_iff_eq_zero, ← csInf_Ioo zero_lt_top]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ : MeasureTheory.Measure Ω
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ μ) (InfSet.sInf (Set.Ioo 0 Top.top))
  -/
  refine sInf_le_sInf fun ε ⟨hε₀, hε_top⟩ B _ ↦ and_self_iff.2 ?_
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ : MeasureTheory.Measure Ω
    ε : ENNReal
    x✝¹ : Membership.mem (Set.Ioo 0 Top.top) ε
    B : Set Ω
    x✝ : MeasurableSet B
    hε₀ : LT.lt 0 ε
    hε_top : LT.lt ε Top.top
    ⊢ LE.le (μ B) (HAdd.hAdd (μ (Metric.thickening ε.toReal B)) ε)
  -/
  refine le_add_right <| measure_mono <| self_subset_thickening ?_ _
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ : MeasureTheory.Measure Ω
    ε : ENNReal
    x✝¹ : Membership.mem (Set.Ioo 0 Top.top) ε
    B : Set Ω
    x✝ : MeasurableSet B
    hε₀ : LT.lt 0 ε
    hε_top : LT.lt ε Top.top
    ⊢ LT.lt 0 ε.toReal
  -/
  exact ENNReal.toReal_pos hε₀.ne' hε_top.ne
  /-
    🎉 no goals
  -/


lemma levyProkhorovEDist_comm (μ ν : Measure Ω) :
    levyProkhorovEDist μ ν = levyProkhorovEDist ν μ := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.levyProkhorovEDist μ ν) (MeasureTheory.levyProkhorovEDist  …
  -/
  simp only [levyProkhorovEDist, and_comm]
  /-
    🎉 no goals
  -/


lemma levyProkhorovEDist_triangle [OpensMeasurableSpace Ω] (μ ν κ : Measure Ω) :
    levyProkhorovEDist μ κ ≤ levyProkhorovEDist μ ν + levyProkhorovEDist ν κ := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ κ) (HAdd.hAdd (MeasureTheory.levyP …
  -/
  by_cases LPμν_finite : levyProkhorovEDist μ ν = ∞
    /-
      case pos
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoEMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ ν κ : MeasureTheory.Measure Ω
      LPμν_finite : Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top
      ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ κ) (HAdd.hAdd (MeasureTheory.levyP …
    -/
  · simp [LPμν_finite]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ κ) (HAdd.hAdd (MeasureTheory.levyP …
  -/
  by_cases LPνκ_finite : levyProkhorovEDist ν κ = ∞
    /-
      case pos
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoEMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μ ν κ : MeasureTheory.Measure Ω
      LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
      LPνκ_finite : Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top
      ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ κ) (HAdd.hAdd (MeasureTheory.levyP …
    -/
  · simp [LPνκ_finite]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ κ) (HAdd.hAdd (MeasureTheory.levyP …
  -/
  apply levyProkhorovEDist_le_of_forall_add_pos_le
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ⊢ ∀ (ε : ENNReal) (B : Set Ω), LT.lt 0 ε → LT.lt ε Top.top → MeasurableSet B → …
  -/
  intro ε B ε_pos ε_lt_top B_mble
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  have half_ε_pos : 0 < ε / 2 := ENNReal.div_pos ε_pos.ne' two_ne_top
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  have half_ε_lt_top : ε / 2 < ∞ := ENNReal.div_lt_top ε_lt_top.ne two_ne_zero
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  let r := levyProkhorovEDist μ ν + ε / 2
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    r : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (HDiv.hDiv ε 2)
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  let s := levyProkhorovEDist ν κ + ε / 2
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    r : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (HDiv.hDiv ε 2)
    s : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist ν κ) (HDiv.hDiv ε 2)
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  have lt_r : levyProkhorovEDist μ ν < r := lt_add_right LPμν_finite half_ε_pos.ne'
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    r : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (HDiv.hDiv ε 2)
    s : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist ν κ) (HDiv.hDiv ε 2)
    lt_r : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) r
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  have lt_s : levyProkhorovEDist ν κ < s := lt_add_right LPνκ_finite half_ε_pos.ne'
  have hs_add_r : s + r = levyProkhorovEDist μ ν + levyProkhorovEDist ν κ + ε := by
    simp_rw [s, r, add_assoc, add_comm (ε / 2), add_assoc, ENNReal.add_halves, ← add_assoc,
      add_comm (levyProkhorovEDist μ ν)]
  have hs_add_r' : s.toReal + r.toReal
      = (levyProkhorovEDist μ ν + levyProkhorovEDist ν κ + ε).toReal := by
    rw [← hs_add_r, ← ENNReal.toReal_add]
    · exact ENNReal.add_ne_top.mpr ⟨LPνκ_finite, half_ε_lt_top.ne⟩
    · exact ENNReal.add_ne_top.mpr ⟨LPμν_finite, half_ε_lt_top.ne⟩
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    r : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (HDiv.hDiv ε 2)
    s : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist ν κ) (HDiv.hDiv ε 2)
    lt_r : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) r
    lt_s : LT.lt (MeasureTheory.levyProkhorovEDist ν κ) s
    hs_add_r : Eq (HAdd.hAdd s r) (HAdd.hAdd (HAdd.hAdd (MeasureTheory.levyProkhor …
    hs_add_r' : Eq (HAdd.hAdd s.toReal r.toReal) (HAdd.hAdd (HAdd.hAdd (MeasureThe …
    ⊢ And (LE.le (μ B) (HAdd.hAdd (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd (HAd …
  -/
  rw [← hs_add_r', add_assoc, ← hs_add_r, add_assoc _ _ ε, ← hs_add_r]
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    LPμν_finite : Not (Eq (MeasureTheory.levyProkhorovEDist μ ν) Top.top)
    LPνκ_finite : Not (Eq (MeasureTheory.levyProkhorovEDist ν κ) Top.top)
    ε : ENNReal
    B : Set Ω
    ε_pos : LT.lt 0 ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    half_ε_lt_top : LT.lt (HDiv.hDiv ε 2) Top.top
    r : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (HDiv.hDiv ε 2)
    s : ENNReal := HAdd.hAdd (MeasureTheory.levyProkhorovEDist ν κ) (HDiv.hDiv ε 2)
    lt_r : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) r
    lt_s : LT.lt (MeasureTheory.levyProkhorovEDist ν κ) s
    hs_add_r : Eq (HAdd.hAdd s r) (HAdd.hAdd (HAdd.hAdd (MeasureTheory.levyProkhor …
    hs_add_r' : Eq (HAdd.hAdd s.toReal r.toReal) (HAdd.hAdd (HAdd.hAdd (MeasureThe …
    ⊢ And (LE.le (μ B) (HAdd.hAdd (κ (Metric.thickening (HAdd.hAdd s.toReal r.toRe …
  -/
  refine ⟨?_, ?_⟩
  · calc μ B ≤ ν (thickening r.toReal B) + r :=
      left_measure_le_of_levyProkhorovEDist_lt lt_r B_mble
    _ ≤ κ (thickening s.toReal (thickening r.toReal B)) + s + r :=
      add_le_add_right
        (left_measure_le_of_levyProkhorovEDist_lt lt_s isOpen_thickening.measurableSet) _
    _ = κ (thickening s.toReal (thickening r.toReal B)) + (s + r) := add_assoc _ _ _
    _ ≤ κ (thickening (s.toReal + r.toReal) B) + (s + r) :=
      add_le_add_right (measure_mono (thickening_thickening_subset _ _ _)) _
  · calc κ B ≤ ν (thickening s.toReal B) + s :=
      right_measure_le_of_levyProkhorovEDist_lt lt_s B_mble
    _ ≤ μ (thickening r.toReal (thickening s.toReal B)) + r + s :=
      add_le_add_right
        (right_measure_le_of_levyProkhorovEDist_lt lt_r isOpen_thickening.measurableSet) s
    _ = μ (thickening r.toReal (thickening s.toReal B)) + (s + r) := by rw [add_assoc, add_comm r]
    _ ≤ μ (thickening (r.toReal + s.toReal) B) + (s + r) :=
      add_le_add_right (measure_mono (thickening_thickening_subset _ _ _)) _
    _ = μ (thickening (s.toReal + r.toReal) B) + (s + r) := by rw [add_comm r.toReal]


/-- The Lévy-Prokhorov distance between finite measures:
`d(μ,ν) = inf {r ≥ 0 | ∀ B, μ B ≤ ν Bᵣ + r ∧ ν B ≤ μ Bᵣ + r}`. -/
noncomputable def levyProkhorovDist (μ ν : Measure Ω) : ℝ :=
  (levyProkhorovEDist μ ν).toReal


lemma levyProkhorovDist_self (μ : Measure Ω) :
    levyProkhorovDist μ μ = 0 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.levyProkhorovDist μ μ) 0
  -/
  simp only [levyProkhorovDist, levyProkhorovEDist_self, zero_toReal]
  /-
    🎉 no goals
  -/


lemma levyProkhorovDist_comm (μ ν : Measure Ω) :
    levyProkhorovDist μ ν = levyProkhorovDist ν μ := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : PseudoEMetricSpace Ω
    μ ν : MeasureTheory.Measure Ω
    ⊢ Eq (MeasureTheory.levyProkhorovDist μ ν) (MeasureTheory.levyProkhorovDist ν μ)
  -/
  simp only [levyProkhorovDist, levyProkhorovEDist_comm]
  /-
    🎉 no goals
  -/


lemma levyProkhorovDist_triangle [OpensMeasurableSpace Ω] (μ ν κ : Measure Ω)
    [IsFiniteMeasure μ] [IsFiniteMeasure ν] [IsFiniteMeasure κ] :
    levyProkhorovDist μ κ ≤ levyProkhorovDist μ ν + levyProkhorovDist ν κ := by
  /-
    Ω : Type u_1
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : PseudoEMetricSpace Ω
    inst✝³ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : MeasureTheory.IsFiniteMeasure κ
    ⊢ LE.le (MeasureTheory.levyProkhorovDist μ κ) (HAdd.hAdd (MeasureTheory.levyPr …
  -/
  have dμν_finite := (levyProkhorovEDist_lt_top μ ν).ne
  /-
    Ω : Type u_1
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : PseudoEMetricSpace Ω
    inst✝³ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : MeasureTheory.IsFiniteMeasure κ
    dμν_finite : Ne (MeasureTheory.levyProkhorovEDist μ ν) Top.top
    ⊢ LE.le (MeasureTheory.levyProkhorovDist μ κ) (HAdd.hAdd (MeasureTheory.levyPr …
  -/
  have dνκ_finite := (levyProkhorovEDist_lt_top ν κ).ne
  /-
    Ω : Type u_1
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : PseudoEMetricSpace Ω
    inst✝³ : OpensMeasurableSpace Ω
    μ ν κ : MeasureTheory.Measure Ω
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : MeasureTheory.IsFiniteMeasure ν
    inst✝ : MeasureTheory.IsFiniteMeasure κ
    dμν_finite : Ne (MeasureTheory.levyProkhorovEDist μ ν) Top.top
    dνκ_finite : Ne (MeasureTheory.levyProkhorovEDist ν κ) Top.top
    ⊢ LE.le (MeasureTheory.levyProkhorovDist μ κ) (HAdd.hAdd (MeasureTheory.levyPr …
  -/
  convert ENNReal.toReal_mono ?_ <| levyProkhorovEDist_triangle μ ν κ
    /-
      case h.e'_4
      Ω : Type u_1
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : PseudoEMetricSpace Ω
      inst✝³ : OpensMeasurableSpace Ω
      μ ν κ : MeasureTheory.Measure Ω
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : MeasureTheory.IsFiniteMeasure κ
      dμν_finite : Ne (MeasureTheory.levyProkhorovEDist μ ν) Top.top
      dνκ_finite : Ne (MeasureTheory.levyProkhorovEDist ν κ) Top.top
      ⊢ Eq (HAdd.hAdd (MeasureTheory.levyProkhorovDist μ ν) (MeasureTheory.levyProkh …
    -/
  · simp only [levyProkhorovDist, ENNReal.toReal_add dμν_finite dνκ_finite]
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝⁵ : MeasurableSpace Ω
      inst✝⁴ : PseudoEMetricSpace Ω
      inst✝³ : OpensMeasurableSpace Ω
      μ ν κ : MeasureTheory.Measure Ω
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      inst✝¹ : MeasureTheory.IsFiniteMeasure ν
      inst✝ : MeasureTheory.IsFiniteMeasure κ
      dμν_finite : Ne (MeasureTheory.levyProkhorovEDist μ ν) Top.top
      dνκ_finite : Ne (MeasureTheory.levyProkhorovEDist ν κ) Top.top
      ⊢ Ne (HAdd.hAdd (MeasureTheory.levyProkhorovEDist μ ν) (MeasureTheory.levyProk …
    -/
  · exact ENNReal.add_ne_top.mpr ⟨dμν_finite, dνκ_finite⟩
    /-
      🎉 no goals
    -/


/-- A type synonym, to be used for `Measure α`, `FiniteMeasure α`, or `ProbabilityMeasure α`,
when they are to be equipped with the Lévy-Prokhorov distance. -/
def LevyProkhorov (α : Type*) := α


/-- The "identity" equivalence between the type synonym `LevyProkhorov α` and `α`. -/
def LevyProkhorov.equiv (α : Type*) : LevyProkhorov α ≃ α := Equiv.refl _


/-- The Lévy-Prokhorov distance `levyProkhorovEDist` makes `Measure Ω` a pseudoemetric
space. The instance is recorded on the type synonym `LevyProkhorov (Measure Ω) := Measure Ω`. -/
noncomputable instance : PseudoEMetricSpace (LevyProkhorov (Measure Ω)) where
  edist := levyProkhorovEDist
  edist_self := levyProkhorovEDist_self
  edist_comm := levyProkhorovEDist_comm
  edist_triangle := levyProkhorovEDist_triangle


/-- The Lévy-Prokhorov distance `levyProkhorovDist` makes `FiniteMeasure Ω` a pseudometric
space. The instance is recorded on the type synonym
`LevyProkhorov (FiniteMeasure Ω) := FiniteMeasure Ω`. -/
noncomputable instance levyProkhorovDist_pseudoMetricSpace_finiteMeasure :
    PseudoMetricSpace (LevyProkhorov (FiniteMeasure Ω)) where
  dist μ ν := levyProkhorovDist μ.toMeasure ν.toMeasure
  dist_self _ := levyProkhorovDist_self _
  dist_comm _ _ := levyProkhorovDist_comm _ _
  dist_triangle _ _ _ := levyProkhorovDist_triangle _ _ _
                       /-
                         Ω : Type u_1
                         inst✝² : MeasurableSpace Ω
                         inst✝¹ : PseudoEMetricSpace Ω
                         inst✝ : OpensMeasurableSpace Ω
                         μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.FiniteMeasure Ω)
                         ⊢ Eq ((fun x y => ↑⟨MeasureTheory.levyProkhorovDist ↑x ↑y, ⋯⟩) μ ν) (ENNReal.o …
                       -/
  edist_dist μ ν := by simp [← ENNReal.ofReal_coe_nnreal]
                       /-
                         🎉 no goals
                       -/


lemma measure_le_measure_closure_of_levyProkhorovEDist_eq_zero {μ ν : Measure Ω}
    (hLP : levyProkhorovEDist μ ν = 0) {s : Set Ω} (s_mble : MeasurableSet s)
    (h_finite : ∃ δ > 0, ν (thickening δ s) ≠ ∞) :
    μ s ≤ ν (closure s) := by
  have key : Tendsto (fun ε ↦ ν (thickening ε.toReal s)) (𝓝[>] (0 : ℝ≥0∞)) (𝓝 (ν (closure s))) := by
    have aux : Tendsto ENNReal.toReal (𝓝[>] 0) (𝓝[>] 0) := by
      apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within (s := Ioi 0) ENNReal.toReal
      · exact tendsto_nhdsWithin_of_tendsto_nhds (continuousAt_toReal zero_ne_top).tendsto
      · filter_upwards [Ioo_mem_nhdsGT zero_lt_one] with x hx
        exact toReal_pos hx.1.ne.symm <| ne_top_of_lt hx.2
    exact (tendsto_measure_thickening h_finite).comp aux
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_mble : MeasurableSet s
    h_finite : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top …
    key : Filter.Tendsto (fun ε => ν (Metric.thickening ε.toReal s)) (nhdsWithin 0 …
    ⊢ LE.le (μ s) (ν (closure s))
  -/
  have obs := Tendsto.add key (tendsto_nhdsWithin_of_tendsto_nhds tendsto_id)
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_mble : MeasurableSet s
    h_finite : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top …
    key : Filter.Tendsto (fun ε => ν (Metric.thickening ε.toReal s)) (nhdsWithin 0 …
    obs : Filter.Tendsto (fun x => HAdd.hAdd (ν (Metric.thickening x.toReal s)) (i …
    ⊢ LE.le (μ s) (ν (closure s))
  -/
  simp only [id_eq, add_zero] at obs
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_mble : MeasurableSet s
    h_finite : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top …
    key : Filter.Tendsto (fun ε => ν (Metric.thickening ε.toReal s)) (nhdsWithin 0 …
    obs : Filter.Tendsto (fun x => HAdd.hAdd (ν (Metric.thickening x.toReal s)) x) …
    ⊢ LE.le (μ s) (ν (closure s))
  -/
  apply ge_of_tendsto (b := μ s) obs
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_mble : MeasurableSet s
    h_finite : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top …
    key : Filter.Tendsto (fun ε => ν (Metric.thickening ε.toReal s)) (nhdsWithin 0 …
    obs : Filter.Tendsto (fun x => HAdd.hAdd (ν (Metric.thickening x.toReal s)) x) …
    ⊢ Filter.Eventually (fun c => LE.le (μ s) (HAdd.hAdd (ν (Metric.thickening c.t …
  -/
  filter_upwards [self_mem_nhdsWithin] with ε ε_pos
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_mble : MeasurableSet s
    h_finite : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top …
    key : Filter.Tendsto (fun ε => ν (Metric.thickening ε.toReal s)) (nhdsWithin 0 …
    obs : Filter.Tendsto (fun x => HAdd.hAdd (ν (Metric.thickening x.toReal s)) x) …
    ε : ENNReal
    ε_pos : Membership.mem (Set.Ioi 0) ε
    ⊢ LE.le (μ s) (HAdd.hAdd (ν (Metric.thickening ε.toReal s)) ε)
  -/
  exact left_measure_le_of_levyProkhorovEDist_lt (B_mble := s_mble) (hLP ▸ ε_pos)
  /-
    🎉 no goals
  -/


/-- Two measures at vanishing Lévy-Prokhorov distance from each other assign the same values to all
closed sets. -/
lemma measure_eq_measure_of_levyProkhorovEDist_eq_zero_of_isClosed {μ ν : Measure Ω}
    (hLP : levyProkhorovEDist μ ν = 0) {s : Set Ω} (s_closed : IsClosed s)
    (hμs : ∃ δ > 0, μ (thickening δ s) ≠ ∞) (hνs : ∃ δ > 0, ν (thickening δ s) ≠ ∞) :
    μ s = ν s := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoEMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    hLP : Eq (MeasureTheory.levyProkhorovEDist μ ν) 0
    s : Set Ω
    s_closed : IsClosed s
    hμs : Exists fun δ => And (GT.gt δ 0) (Ne (μ (Metric.thickening δ s)) Top.top)
    hνs : Exists fun δ => And (GT.gt δ 0) (Ne (ν (Metric.thickening δ s)) Top.top)
    ⊢ Eq (μ s) (ν s)
  -/
  apply le_antisymm
  · exact measure_le_measure_closure_of_levyProkhorovEDist_eq_zero
      hLP s_closed.measurableSet hνs |>.trans <|
      le_of_eq (congr_arg _ s_closed.closure_eq)
  · exact measure_le_measure_closure_of_levyProkhorovEDist_eq_zero
      (levyProkhorovEDist_comm μ ν ▸ hLP) s_closed.measurableSet hμs |>.trans <|
      le_of_eq (congr_arg _ s_closed.closure_eq)


/-- The Lévy-Prokhorov distance `levyProkhorovDist` makes `ProbabilityMeasure Ω` a pseudometric
space. The instance is recorded on the type synonym
`LevyProkhorov (ProbabilityMeasure Ω) := ProbabilityMeasure Ω`.

Note: For this pseudometric to give the topology of convergence in distribution, one must
furthermore assume that `Ω` is separable. -/
noncomputable instance levyProkhorovDist_pseudoMetricSpace_probabilityMeasure :
    PseudoMetricSpace (LevyProkhorov (ProbabilityMeasure Ω)) where
  dist μ ν := levyProkhorovDist μ.toMeasure ν.toMeasure
  dist_self _ := levyProkhorovDist_self _
  dist_comm _ _ := levyProkhorovDist_comm _ _
  dist_triangle _ _ _ := levyProkhorovDist_triangle _ _ _
                       /-
                         Ω : Type u_1
                         inst✝² : MeasurableSpace Ω
                         inst✝¹ : PseudoEMetricSpace Ω
                         inst✝ : OpensMeasurableSpace Ω
                         μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
                         ⊢ Eq ((fun x y => ↑⟨MeasureTheory.levyProkhorovDist ↑x ↑y, ⋯⟩) μ ν) (ENNReal.o …
                       -/
  edist_dist μ ν := by simp [← ENNReal.ofReal_coe_nnreal]
                       /-
                         🎉 no goals
                       -/


lemma LevyProkhorov.dist_def (μ ν : LevyProkhorov (ProbabilityMeasure Ω)) :
    dist μ ν = levyProkhorovDist μ.toMeasure ν.toMeasure := rfl


/-- If `Ω` is a Borel space, then the Lévy-Prokhorov distance `levyProkhorovDist` makes
`ProbabilityMeasure Ω` a metric space. The instance is recorded on the type synonym
`LevyProkhorov (ProbabilityMeasure Ω) := ProbabilityMeasure Ω`.

Note: For this metric to give the topology of convergence in distribution, one must
furthermore assume that `Ω` is separable. -/
noncomputable instance levyProkhorovDist_metricSpace_probabilityMeasure [BorelSpace Ω] :
    MetricSpace (LevyProkhorov (ProbabilityMeasure Ω)) where
  eq_of_dist_eq_zero := by
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : PseudoEMetricSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : BorelSpace Ω
      ⊢ ∀ {x y : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)},  …
    -/
    intro μ ν h
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : PseudoEMetricSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : BorelSpace Ω
      μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      h : Eq (Dist.dist μ ν) 0
      ⊢ Eq μ ν
    -/
    apply (LevyProkhorov.equiv _).injective
    /-
      case a
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : PseudoEMetricSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : BorelSpace Ω
      μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      h : Eq (Dist.dist μ ν) 0
      ⊢ Eq ((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMeasure Ω)) …
    -/
    apply ProbabilityMeasure.toMeasure_injective
    /-
      case a.a
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : PseudoEMetricSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : BorelSpace Ω
      μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      h : Eq (Dist.dist μ ν) 0
      ⊢ Eq ↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMeasure Ω) …
    -/
    apply ext_of_generate_finite _ ?_ isPiSystem_isClosed ?_ (by simp)
      /-
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : PseudoEMetricSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : BorelSpace Ω
        μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        h : Eq (Dist.dist μ ν) 0
        ⊢ Eq inst✝³ (MeasurableSpace.generateFrom (setOf fun s => IsClosed s))
      -/
    · rw [BorelSpace.measurable_eq (α := Ω), borel_eq_generateFrom_isClosed]
      /-
        🎉 no goals
      -/
      /-
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : PseudoEMetricSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : BorelSpace Ω
        μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        h : Eq (Dist.dist μ ν) 0
        ⊢ ∀ (s : Set Ω), Membership.mem (setOf fun s => IsClosed s) s → Eq (↑((Measure …
      -/
    · intro A A_closed
      /-
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : PseudoEMetricSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : BorelSpace Ω
        μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        h : Eq (Dist.dist μ ν) 0
        A : Set Ω
        A_closed : Membership.mem (setOf fun s => IsClosed s) A
        ⊢ Eq (↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMeasure Ω …
      -/
      apply measure_eq_measure_of_levyProkhorovEDist_eq_zero_of_isClosed
      · simpa only [levyProkhorovEDist_ne_top μ.toMeasure ν.toMeasure, mem_setOf_eq,
                    or_false, ne_eq, zero_ne_top, not_false_eq_true, zero_toReal]
          using (toReal_eq_zero_iff _).mp h
        /-
          case s_closed
          Ω : Type u_1
          inst✝³ : MeasurableSpace Ω
          inst✝² : PseudoEMetricSpace Ω
          inst✝¹ : OpensMeasurableSpace Ω
          inst✝ : BorelSpace Ω
          μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          h : Eq (Dist.dist μ ν) 0
          A : Set Ω
          A_closed : Membership.mem (setOf fun s => IsClosed s) A
          ⊢ IsClosed A
        -/
      · exact A_closed
        /-
          🎉 no goals
        -/
        /-
          case hμs
          Ω : Type u_1
          inst✝³ : MeasurableSpace Ω
          inst✝² : PseudoEMetricSpace Ω
          inst✝¹ : OpensMeasurableSpace Ω
          inst✝ : BorelSpace Ω
          μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          h : Eq (Dist.dist μ ν) 0
          A : Set Ω
          A_closed : Membership.mem (setOf fun s => IsClosed s) A
          ⊢ Exists fun δ => And (GT.gt δ 0) (Ne (↑((MeasureTheory.LevyProkhorov.equiv (M …
        -/
      · exact ⟨1, Real.zero_lt_one, measure_ne_top _ _⟩
        /-
          🎉 no goals
        -/
        /-
          case hνs
          Ω : Type u_1
          inst✝³ : MeasurableSpace Ω
          inst✝² : PseudoEMetricSpace Ω
          inst✝¹ : OpensMeasurableSpace Ω
          inst✝ : BorelSpace Ω
          μ ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          h : Eq (Dist.dist μ ν) 0
          A : Set Ω
          A_closed : Membership.mem (setOf fun s => IsClosed s) A
          ⊢ Exists fun δ => And (GT.gt δ 0) (Ne (↑((MeasureTheory.LevyProkhorov.equiv (M …
        -/
      · exact ⟨1, Real.zero_lt_one, measure_ne_top _ _⟩
        /-
          🎉 no goals
        -/


/-- A simple sufficient condition for bounding `levyProkhorovEDist` between probability measures
from above. The condition involves only one of two natural bounds, the other bound is for free. -/
lemma levyProkhorovEDist_le_of_forall_le
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] (δ : ℝ≥0∞)
    (h : ∀ ε B, δ < ε → ε < ∞ → MeasurableSet B → μ B ≤ ν (thickening ε.toReal B) + ε) :
    levyProkhorovEDist μ ν ≤ δ := by
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) δ
  -/
  apply levyProkhorovEDist_le_of_forall μ ν δ
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ⊢ ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B → …
  -/
  intro ε B ε_gt ε_lt_top B_mble
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    ⊢ And (LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε.toReal B)) ε)) (LE.le (ν …
  -/
  refine ⟨h ε B ε_gt ε_lt_top B_mble, ?_⟩
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    ⊢ LE.le (ν B) (HAdd.hAdd (μ (Metric.thickening ε.toReal B)) ε)
  -/
  have B_subset := subset_compl_thickening_compl_thickening_self ε.toReal B
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    ⊢ LE.le (ν B) (HAdd.hAdd (μ (Metric.thickening ε.toReal B)) ε)
  -/
  apply (measure_mono (μ := ν) B_subset).trans
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    ⊢ LE.le (ν (HasCompl.compl (Metric.thickening ε.toReal (HasCompl.compl (Metric …
  -/
  rw [prob_compl_eq_one_sub isOpen_thickening.measurableSet]
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  have Tc_mble := (isOpen_thickening (δ := ε.toReal) (E := B)).isClosed_compl.measurableSet
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : ENNReal
    h : ∀ (ε : ENNReal) (B : Set Ω), LT.lt δ ε → LT.lt ε Top.top → MeasurableSet B …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  specialize h ε (thickening ε.toReal B)ᶜ ε_gt ε_lt_top Tc_mble
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    h : LE.le (μ (HasCompl.compl (Metric.thickening ε.toReal B))) (HAdd.hAdd (ν (M …
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  rw [prob_compl_eq_one_sub isOpen_thickening.measurableSet] at h
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    h : LE.le (HSub.hSub 1 (μ (Metric.thickening ε.toReal B))) (HAdd.hAdd (ν (Metr …
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  have almost := add_le_add (c := μ (thickening ε.toReal B)) h rfl.le
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    h : LE.le (HSub.hSub 1 (μ (Metric.thickening ε.toReal B))) (HAdd.hAdd (ν (Metr …
    almost : LE.le (HAdd.hAdd (HSub.hSub 1 (μ (Metric.thickening ε.toReal B))) (μ  …
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  rw [tsub_add_cancel_of_le prob_le_one, add_assoc] at almost
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    h : LE.le (HSub.hSub 1 (μ (Metric.thickening ε.toReal B))) (HAdd.hAdd (ν (Metr …
    almost : LE.le 1 (HAdd.hAdd (ν (Metric.thickening ε.toReal (HasCompl.compl (Me …
    ⊢ LE.le (HSub.hSub 1 (ν (Metric.thickening ε.toReal (HasCompl.compl (Metric.th …
  -/
  apply (tsub_le_tsub_right almost _).trans
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt δ ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    B_subset : HasSubset.Subset B (HasCompl.compl (Metric.thickening ε.toReal (Has …
    Tc_mble : MeasurableSet (HasCompl.compl (Metric.thickening ε.toReal B))
    h : LE.le (HSub.hSub 1 (μ (Metric.thickening ε.toReal B))) (HAdd.hAdd (ν (Metr …
    almost : LE.le 1 (HAdd.hAdd (ν (Metric.thickening ε.toReal (HasCompl.compl (Me …
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (ν (Metric.thickening ε.toReal (HasCompl.compl ( …
  -/
  rw [ENNReal.add_sub_cancel_left (measure_ne_top ν _), add_comm ε]
  /-
    🎉 no goals
  -/


/-- A simple sufficient condition for bounding `levyProkhorovDist` between probability measures
from above. The condition involves only one of two natural bounds, the other bound is for free. -/
lemma levyProkhorovDist_le_of_forall_le
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] {δ : ℝ} (δ_nn : 0 ≤ δ)
    (h : ∀ ε B, δ < ε → MeasurableSet B → μ B ≤ ν (thickening ε B) + ENNReal.ofReal ε) :
    levyProkhorovDist μ ν ≤ δ := by
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : Real
    δ_nn : LE.le 0 δ
    h : ∀ (ε : Real) (B : Set Ω), LT.lt δ ε → MeasurableSet B → LE.le (μ B) (HAdd. …
    ⊢ LE.le (MeasureTheory.levyProkhorovDist μ ν) δ
  -/
  apply toReal_le_of_le_ofReal δ_nn
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : Real
    δ_nn : LE.le 0 δ
    h : ∀ (ε : Real) (B : Set Ω), LT.lt δ ε → MeasurableSet B → LE.le (μ B) (HAdd. …
    ⊢ LE.le (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal δ)
  -/
  apply levyProkhorovEDist_le_of_forall_le
  /-
    case h
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : Real
    δ_nn : LE.le 0 δ
    h : ∀ (ε : Real) (B : Set Ω), LT.lt δ ε → MeasurableSet B → LE.le (μ B) (HAdd. …
    ⊢ ∀ (ε : ENNReal) (B : Set Ω), LT.lt (ENNReal.ofReal δ) ε → LT.lt ε Top.top →  …
  -/
  intro ε B ε_gt ε_lt_top B_mble
  have ε_gt' : δ < ε.toReal := by
    refine (ofReal_lt_ofReal_iff ?_).mp ?_
    · exact ENNReal.toReal_pos (ne_zero_of_lt ε_gt) ε_lt_top.ne
    · simpa [ofReal_toReal_eq_iff.mpr ε_lt_top.ne] using ε_gt
  /-
    case h
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : Real
    δ_nn : LE.le 0 δ
    h : ∀ (ε : Real) (B : Set Ω), LT.lt δ ε → MeasurableSet B → LE.le (μ B) (HAdd. …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt (ENNReal.ofReal δ) ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    ε_gt' : LT.lt δ ε.toReal
    ⊢ LE.le (μ B) (HAdd.hAdd (ν (Metric.thickening ε.toReal B)) ε)
  -/
  convert h ε.toReal B ε_gt' B_mble
  /-
    case h.e'_4.h.e'_6
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    δ : Real
    δ_nn : LE.le 0 δ
    h : ∀ (ε : Real) (B : Set Ω), LT.lt δ ε → MeasurableSet B → LE.le (μ B) (HAdd. …
    ε : ENNReal
    B : Set Ω
    ε_gt : LT.lt (ENNReal.ofReal δ) ε
    ε_lt_top : LT.lt ε Top.top
    B_mble : MeasurableSet B
    ε_gt' : LT.lt δ ε.toReal
    ⊢ Eq ε (ENNReal.ofReal ε.toReal)
  -/
  exact (ENNReal.ofReal_toReal ε_lt_top.ne).symm
  /-
    🎉 no goals
  -/


/-- A version of the layer cake formula for bounded continuous functions which have finite integral:
∫ f dμ = ∫ t in (0, ‖f‖], μ {x | f(x) ≥ t} dt. -/
lemma BoundedContinuousFunction.integral_eq_integral_meas_le_of_hasFiniteIntegral
    {α : Type*} [MeasurableSpace α] [TopologicalSpace α] [OpensMeasurableSpace α]
    (f : α →ᵇ ℝ) (μ : Measure α) (f_nn : 0 ≤ᵐ[μ] f) (hf : HasFiniteIntegral f μ) :
    ∫ ω, f ω ∂μ = ∫ t in Ioc 0 ‖f‖, ENNReal.toReal (μ {a : α | t ≤ f a}) := by
  /-
    α : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    f : BoundedContinuousFunction α Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
    hf : MeasureTheory.HasFiniteIntegral (⇑f) μ
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  rw [Integrable.integral_eq_integral_Ioc_meas_le (M := ‖f‖) ?_ f_nn ?_]
    /-
      α : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      inst✝ : OpensMeasurableSpace α
      f : BoundedContinuousFunction α Real
      μ : MeasureTheory.Measure α
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
      hf : MeasureTheory.HasFiniteIntegral (⇑f) μ
      ⊢ MeasureTheory.Integrable (⇑f) μ
    -/
  · refine ⟨f.continuous.measurable.aestronglyMeasurable, hf⟩
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      inst✝ : OpensMeasurableSpace α
      f : BoundedContinuousFunction α Real
      μ : MeasureTheory.Measure α
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
      hf : MeasureTheory.HasFiniteIntegral (⇑f) μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE ⇑f fun x => Norm.norm f
    -/
  · exact Eventually.of_forall (fun x ↦ BoundedContinuousFunction.apply_le_norm f x)
    /-
      🎉 no goals
    -/


/-- A version of the layer cake formula for bounded continuous functions and finite measures:
∫ f dμ = ∫ t in (0, ‖f‖], μ {x | f(x) ≥ t} dt. -/
lemma BoundedContinuousFunction.integral_eq_integral_meas_le
    {α : Type*} [MeasurableSpace α] [TopologicalSpace α] [OpensMeasurableSpace α]
    (f : α →ᵇ ℝ) (μ : Measure α) [IsFiniteMeasure μ] (f_nn : 0 ≤ᵐ[μ] f) :
    ∫ ω, f ω ∂μ = ∫ t in Ioc 0 ‖f‖, ENNReal.toReal (μ {a : α | t ≤ f a}) :=
  integral_eq_integral_meas_le_of_hasFiniteIntegral _ _ f_nn (f.integrable μ).2


/-- Assuming `levyProkhorovEDist μ ν < ε`, we can bound `∫ f ∂μ` in terms of
`∫ t in (0, ‖f‖], ν (thickening ε {x | f(x) ≥ t}) dt` and `‖f‖`. -/
lemma BoundedContinuousFunction.integral_le_of_levyProkhorovEDist_lt (μ ν : Measure Ω)
    [IsFiniteMeasure μ] [IsFiniteMeasure ν] {ε : ℝ} (ε_pos : 0 < ε)
    (hμν : levyProkhorovEDist μ ν < ENNReal.ofReal ε) (f : Ω →ᵇ ℝ) (f_nn : 0 ≤ᵐ[μ] f) :
    ∫ ω, f ω ∂μ
      ≤ (∫ t in Ioc 0 ‖f‖, ENNReal.toReal (ν (thickening ε {a | t ≤ f a}))) + ε * ‖f‖ := by
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ε : Real
    ε_pos : LT.lt 0 ε
    hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
    f : BoundedContinuousFunction Ω Real
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
    ⊢ LE.le (MeasureTheory.integral μ fun ω => f ω) (HAdd.hAdd (MeasureTheory.inte …
  -/
  rw [BoundedContinuousFunction.integral_eq_integral_meas_le f μ f_nn]
  have key : (fun (t : ℝ) ↦ ENNReal.toReal (μ {a | t ≤ f a}))
              ≤ (fun (t : ℝ) ↦ ENNReal.toReal (ν (thickening ε {a | t ≤ f a})) + ε) := by
    intro t
    convert ENNReal.toReal_mono ?_ <| left_measure_le_of_levyProkhorovEDist_lt hμν
      (B := {a | t ≤ f a}) (f.continuous.measurable measurableSet_Ici)
    · rw [ENNReal.toReal_add (measure_ne_top ν _) ofReal_ne_top, ENNReal.toReal_ofReal ε_pos.le]
    · exact ENNReal.add_ne_top.mpr ⟨measure_ne_top ν _, ofReal_ne_top⟩
  have intble₁ : IntegrableOn (fun t ↦ ENNReal.toReal (μ {a | t ≤ f a})) (Ioc 0 ‖f‖) := by
    apply Measure.integrableOn_of_bounded (M := ENNReal.toReal (μ univ)) measure_Ioc_lt_top.ne
    · apply (Measurable.ennreal_toReal (Antitone.measurable ?_)).aestronglyMeasurable
      exact fun _ _ hst ↦ measure_mono (fun _ h ↦ hst.trans h)
    · apply Eventually.of_forall <| fun t ↦ ?_
      simp only [Real.norm_eq_abs, abs_toReal]
      exact ENNReal.toReal_mono (measure_ne_top _ _) <| measure_mono (subset_univ _)
  have intble₂ : IntegrableOn
                  (fun t ↦ ENNReal.toReal (ν (thickening ε {a | t ≤ f a}))) (Ioc 0 ‖f‖) := by
    apply Measure.integrableOn_of_bounded (M := ENNReal.toReal (ν univ)) measure_Ioc_lt_top.ne
    · apply (Measurable.ennreal_toReal (Antitone.measurable ?_)).aestronglyMeasurable
      exact fun _ _ hst ↦ measure_mono <| thickening_subset_of_subset ε (fun _ h ↦ hst.trans h)
    · apply Eventually.of_forall <| fun t ↦ ?_
      simp only [Real.norm_eq_abs, abs_toReal]
      exact ENNReal.toReal_mono (measure_ne_top _ _) <| measure_mono (subset_univ _)
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ε : Real
    ε_pos : LT.lt 0 ε
    hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
    f : BoundedContinuousFunction Ω Real
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
    key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
    intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
    intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
    ⊢ LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (S …
  -/
  apply le_trans (setIntegral_mono (s := Ioc 0 ‖f‖) ?_ ?_ key)
    /-
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      ε : Real
      ε_pos : LT.lt 0 ε
      hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
      f : BoundedContinuousFunction Ω Real
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
      key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
      intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
      intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
      ⊢ LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (S …
    -/
  · rw [integral_add]
      /-
        Ω : Type u_1
        inst✝⁴ : MeasurableSpace Ω
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : OpensMeasurableSpace Ω
        μ ν : MeasureTheory.Measure Ω
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        ε : Real
        ε_pos : LT.lt 0 ε
        hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
        f : BoundedContinuousFunction Ω Real
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
        key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
        intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
        intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
        ⊢ LE.le (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
      -/
    · apply add_le_add_left
      simp only [integral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter,
                  Real.volume_Ioc, sub_zero, norm_nonneg, toReal_ofReal, smul_eq_mul,
                  (mul_comm _ ε).le]
      /-
        case hf
        Ω : Type u_1
        inst✝⁴ : MeasurableSpace Ω
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : OpensMeasurableSpace Ω
        μ ν : MeasureTheory.Measure Ω
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        ε : Real
        ε_pos : LT.lt 0 ε
        hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
        f : BoundedContinuousFunction Ω Real
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
        key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
        intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
        intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
        ⊢ MeasureTheory.Integrable (fun x => (ν (Metric.thickening ε (setOf fun a => L …
      -/
    · exact intble₂
      /-
        🎉 no goals
      -/
      /-
        case hg
        Ω : Type u_1
        inst✝⁴ : MeasurableSpace Ω
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : OpensMeasurableSpace Ω
        μ ν : MeasureTheory.Measure Ω
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : MeasureTheory.IsFiniteMeasure ν
        ε : Real
        ε_pos : LT.lt 0 ε
        hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
        f : BoundedContinuousFunction Ω Real
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
        key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
        intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
        intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
        ⊢ MeasureTheory.Integrable (fun x => ε) (MeasureTheory.MeasureSpace.volume.res …
      -/
    · exact integrable_const ε
      /-
        🎉 no goals
      -/
    /-
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      ε : Real
      ε_pos : LT.lt 0 ε
      hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
      f : BoundedContinuousFunction Ω Real
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
      key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
      intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
      intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
      ⊢ MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a))).toRe …
    -/
  · exact intble₁
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      ε : Real
      ε_pos : LT.lt 0 ε
      hμν : LT.lt (MeasureTheory.levyProkhorovEDist μ ν) (ENNReal.ofReal ε)
      f : BoundedContinuousFunction Ω Real
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
      key : LE.le (fun t => (μ (setOf fun a => LE.le t (f a))).toReal) fun t => HAdd …
      intble₁ : MeasureTheory.IntegrableOn (fun t => (μ (setOf fun a => LE.le t (f a …
      intble₂ : MeasureTheory.IntegrableOn (fun t => (ν (Metric.thickening ε (setOf  …
      ⊢ MeasureTheory.IntegrableOn (fun t => HAdd.hAdd (ν (Metric.thickening ε (setO …
    -/
  · exact intble₂.add <| integrable_const ε
    /-
      🎉 no goals
    -/


/-- A monotone decreasing convergence lemma for integrals of measures of thickenings:
`∫ t in (0, ‖f‖], μ (thickening ε {x | f(x) ≥ t}) dt` tends to
`∫ t in (0, ‖f‖], μ {x | f(x) ≥ t} dt` as `ε → 0`. -/
lemma tendsto_integral_meas_thickening_le (f : Ω →ᵇ ℝ)
    {A : Set ℝ} (A_finmeas : volume A ≠ ∞) (μ : ProbabilityMeasure Ω) :
    Tendsto (fun ε ↦ ∫ t in A, ENNReal.toReal (μ (thickening ε {a | t ≤ f a}))) (𝓝[>] (0 : ℝ))
      (𝓝 (∫ t in A, ENNReal.toReal (μ {a | t ≤ f a}))) := by
  apply tendsto_integral_filter_of_dominated_convergence (G := ℝ) (μ := volume.restrict A)
        (F := fun ε t ↦ (μ (thickening ε {a | t ≤ f a}))) (f := fun t ↦ (μ {a | t ≤ f a})) 1
    /-
      case hF_meas
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fun t => ↑(μ …
    -/
  · apply Eventually.of_forall fun n ↦ Measurable.aestronglyMeasurable ?_
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      n : Real
      ⊢ Measurable fun t => ↑(μ (Metric.thickening n (setOf fun a => LE.le t (f a))))
    -/
    simp only [measurable_coe_nnreal_real_iff]
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      n : Real
      ⊢ Measurable fun x => μ (Metric.thickening n (setOf fun a => LE.le x (f a)))
    -/
    apply measurable_toNNReal.comp <| Antitone.measurable (fun s t hst ↦ ?_)
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      n s t : Real
      hst : LE.le s t
      ⊢ LE.le (↑μ (Metric.thickening n (setOf fun a => LE.le t (f a)))) (↑μ (Metric. …
    -/
    exact measure_mono <| thickening_subset_of_subset _ <| fun ω h ↦ hst.trans h
    /-
      🎉 no goals
    -/
    /-
      case h_bound
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm ↑(μ …
    -/
  · apply Eventually.of_forall (fun i ↦ ?_)
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      i : Real
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm ↑(μ (Metric.thickening i (setOf …
    -/
    apply Eventually.of_forall (fun t ↦ ?_)
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      i t : Real
      ⊢ LE.le (Norm.norm ↑(μ (Metric.thickening i (setOf fun a => LE.le t (f a)))))  …
    -/
    simp only [Real.norm_eq_abs, NNReal.abs_eq, Pi.one_apply]
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      i t : Real
      ⊢ LE.le (↑(μ (Metric.thickening i (setOf fun a => LE.le t (f a))))) 1
    -/
    exact ENNReal.toReal_mono one_ne_top prob_le_one
    /-
      🎉 no goals
    -/
    /-
      case bound_integrable
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      ⊢ MeasureTheory.Integrable 1 (MeasureTheory.MeasureSpace.volume.restrict A)
    -/
  · have aux : IsFiniteMeasure (volume.restrict A) := ⟨by simp [lt_top_iff_ne_top, A_finmeas]⟩
    /-
      case bound_integrable
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      aux : MeasureTheory.IsFiniteMeasure (MeasureTheory.MeasureSpace.volume.restric …
      ⊢ MeasureTheory.Integrable 1 (MeasureTheory.MeasureSpace.volume.restrict A)
    -/
    apply integrable_const
    /-
      🎉 no goals
    -/
    /-
      case h_lim
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => ↑(μ (Metric.thickening  …
    -/
  · apply Eventually.of_forall (fun t ↦ ?_)
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      t : Real
      ⊢ Filter.Tendsto (fun n => ↑(μ (Metric.thickening n (setOf fun a => LE.le t (f …
    -/
    simp only [NNReal.tendsto_coe]
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      f : BoundedContinuousFunction Ω Real
      A : Set Real
      A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
      μ : MeasureTheory.ProbabilityMeasure Ω
      t : Real
      ⊢ Filter.Tendsto (fun a => μ (Metric.thickening a (setOf fun a => LE.le t (f a …
    -/
    apply (ENNReal.tendsto_toNNReal _).comp
      /-
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        f : BoundedContinuousFunction Ω Real
        A : Set Real
        A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
        μ : MeasureTheory.ProbabilityMeasure Ω
        t : Real
        ⊢ Filter.Tendsto (fun a => ↑μ (Metric.thickening a (setOf fun a => LE.le t (f  …
      -/
    · apply tendsto_measure_thickening_of_isClosed ?_ ?_
        /-
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          f : BoundedContinuousFunction Ω Real
          A : Set Real
          A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
          μ : MeasureTheory.ProbabilityMeasure Ω
          t : Real
          ⊢ Exists fun R => And (GT.gt R 0) (Ne (↑μ (Metric.thickening R (setOf fun a => …
        -/
      · exact ⟨1, ⟨Real.zero_lt_one, measure_ne_top _ _⟩⟩
        /-
          🎉 no goals
        -/
        /-
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          f : BoundedContinuousFunction Ω Real
          A : Set Real
          A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
          μ : MeasureTheory.ProbabilityMeasure Ω
          t : Real
          ⊢ IsClosed (setOf fun a => LE.le t (f a))
        -/
      · exact isClosed_le continuous_const f.continuous
        /-
          🎉 no goals
        -/
      /-
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        f : BoundedContinuousFunction Ω Real
        A : Set Real
        A_finmeas : Ne (MeasureTheory.MeasureSpace.volume A) Top.top
        μ : MeasureTheory.ProbabilityMeasure Ω
        t : Real
        ⊢ Ne (↑μ (setOf fun a => LE.le t (f a))) Top.top
      -/
    · exact measure_ne_top _ _
      /-
        🎉 no goals
      -/


/-- The identity map `LevyProkhorov (ProbabilityMeasure Ω) → ProbabilityMeasure Ω` is continuous. -/
lemma LevyProkhorov.continuous_equiv_probabilityMeasure :
    Continuous (LevyProkhorov.equiv (α := ProbabilityMeasure Ω)) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    ⊢ Continuous ⇑(MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMea …
  -/
  refine SeqContinuous.continuous ?_
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    ⊢ SeqContinuous ⇑(MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probability …
  -/
  intro μs ν hμs
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    ⊢ Filter.Tendsto (Function.comp (⇑(MeasureTheory.LevyProkhorov.equiv (MeasureT …
  -/
  set P := LevyProkhorov.equiv _ ν -- more palatable notation
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    ⊢ Filter.Tendsto (Function.comp (⇑(MeasureTheory.LevyProkhorov.equiv (MeasureT …
  -/
  set Ps := fun n ↦ LevyProkhorov.equiv _ (μs n) -- more palatable notation
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    ⊢ Filter.Tendsto (Function.comp (⇑(MeasureTheory.LevyProkhorov.equiv (MeasureT …
  -/
  rw [ProbabilityMeasure.tendsto_iff_forall_integral_tendsto]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    ⊢ ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => MeasureTh …
  -/
  refine fun f ↦ tendsto_integral_of_forall_limsup_integral_le_integral ?_ f
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f : BoundedContinuousFunction Ω Real
    ⊢ ∀ (f : BoundedContinuousFunction Ω Real), LE.le 0 f → LE.le (Filter.limsup ( …
  -/
  intro f f_nn
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f✝ f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
  -/
  by_cases f_zero : ‖f‖ = 0
    /-
      case pos
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Eq (Norm.norm f) 0
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
    -/
  · simp only [norm_eq_zero] at f_zero
    /-
      case pos
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Eq f 0
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
    -/
    simp [f_zero, limsup_const]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f✝ f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    f_zero : Not (Eq (Norm.norm f) 0)
    ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
  -/
  have norm_f_pos : 0 < ‖f‖ := lt_of_le_of_ne (norm_nonneg _) (fun a => f_zero a.symm)
  /-
    case neg
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f✝ f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    f_zero : Not (Eq (Norm.norm f) 0)
    norm_f_pos : LT.lt 0 (Norm.norm f)
    ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
  -/
  apply _root_.le_of_forall_pos_le_add
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f✝ f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    f_zero : Not (Eq (Norm.norm f) 0)
    norm_f_pos : LT.lt 0 (Norm.norm f)
    ⊢ ∀ (ε : Real), LT.lt 0 ε → LE.le (Filter.limsup (fun i => MeasureTheory.integ …
  -/
  intro δ δ_pos
  /-
    case neg.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : PseudoMetricSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
    hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
    P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
    Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
    f✝ f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    f_zero : Not (Eq (Norm.norm f) 0)
    norm_f_pos : LT.lt 0 (Norm.norm f)
    δ : Real
    δ_pos : LT.lt 0 δ
    ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral ↑(Function.comp (⇑(Mea …
  -/
  apply limsup_le_of_le ?_
    /-
      case neg.h
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      ⊢ Filter.Eventually (fun n => LE.le (MeasureTheory.integral ↑(Function.comp (⇑ …
    -/
  · obtain ⟨εs, ⟨_, ⟨εs_pos, εs_lim⟩⟩⟩ := exists_seq_strictAnti_tendsto (0 : ℝ)
    /-
      case neg.h.intro.intro.intro
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ⊢ Filter.Eventually (fun n => LE.le (MeasureTheory.integral ↑(Function.comp (⇑ …
    -/
    have ε_of_room := Tendsto.add (tendsto_iff_dist_tendsto_zero.mp hμs) εs_lim
    have ε_of_room' : Tendsto (fun n ↦ dist (μs n) ν + εs n) atTop (𝓝[>] 0) := by
      rw [tendsto_nhdsWithin_iff]
      refine ⟨by simpa using ε_of_room, Eventually.of_forall fun n ↦ ?_⟩
      · rw [mem_Ioi]
        linarith [εs_pos n, dist_nonneg (x := μs n) (y := ν)]
    /-
      case neg.h.intro.intro.intro
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
      ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
      ⊢ Filter.Eventually (fun n => LE.le (MeasureTheory.integral ↑(Function.comp (⇑ …
    -/
    rw [add_zero] at ε_of_room
    /-
      case neg.h.intro.intro.intro
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
      ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
      ⊢ Filter.Eventually (fun n => LE.le (MeasureTheory.integral ↑(Function.comp (⇑ …
    -/
    have key := (tendsto_integral_meas_thickening_le f (A := Ioc 0 ‖f‖) (by simp) P).comp ε_of_room'
    /-
      case neg.h.intro.intro.intro
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
      ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
      key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
      ⊢ Filter.Eventually (fun n => LE.le (MeasureTheory.integral ↑(Function.comp (⇑ …
    -/
    have aux : ∀ (z : ℝ), Iio (z + δ/2) ∈ 𝓝 z := fun z ↦ Iio_mem_nhds (by linarith)
    filter_upwards [key (aux _), ε_of_room <| Iio_mem_nhds <| half_pos <|
                      mul_pos (inv_pos.mpr norm_f_pos) δ_pos]
      with n hn hn'
    /-
      case h
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
      ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
      key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
      aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
      n : Nat
      hn : Membership.mem (Set.preimage (Function.comp (fun ε => MeasureTheory.integ …
      hn' : Membership.mem (Set.preimage (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (ε …
      ⊢ LE.le (MeasureTheory.integral ↑(Function.comp (⇑(MeasureTheory.LevyProkhorov …
    -/
    simp only [mem_preimage, mem_Iio] at *
    /-
      case h
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      εs : Nat → Real
      left✝ : StrictAnti εs
      εs_pos : ∀ (n : Nat), LT.lt 0 (εs n)
      εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
      ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
      ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
      key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
      aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
      n : Nat
      hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
      hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
      ⊢ LE.le (MeasureTheory.integral ↑(Function.comp (⇑(MeasureTheory.LevyProkhorov …
    -/
    specialize εs_pos n
    have bound := BoundedContinuousFunction.integral_le_of_levyProkhorovEDist_lt
                    (Ps n) P (ε := dist (μs n) ν + εs n) ?_ ?_ f ?_
      /-
        case h.refine_4
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
        ⊢ LE.le (MeasureTheory.integral ↑(Function.comp (⇑(MeasureTheory.LevyProkhorov …
      -/
    · refine bound.trans ?_
      /-
        case h.refine_4
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
        ⊢ LE.le (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
      -/
      apply (add_le_add_right hn.le _).trans
      /-
        case h.refine_4
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSp …
      -/
      rw [BoundedContinuousFunction.integral_eq_integral_meas_le]
        /-
          case h.refine_4
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
          P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
          Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
          f✝ f : BoundedContinuousFunction Ω Real
          f_nn : LE.le 0 f
          f_zero : Not (Eq (Norm.norm f) 0)
          norm_f_pos : LT.lt 0 (Norm.norm f)
          δ : Real
          δ_pos : LT.lt 0 δ
          εs : Nat → Real
          left✝ : StrictAnti εs
          εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
          ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
          ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
          key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
          aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
          n : Nat
          hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
          hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
          εs_pos : LT.lt 0 (εs n)
          bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
          ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSp …
        -/
      · simp only [ProbabilityMeasure.ennreal_coeFn_eq_coeFn_toMeasure]
        /-
          case h.refine_4
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
          P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
          Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
          f✝ f : BoundedContinuousFunction Ω Real
          f_nn : LE.le 0 f
          f_zero : Not (Eq (Norm.norm f) 0)
          norm_f_pos : LT.lt 0 (Norm.norm f)
          δ : Real
          δ_pos : LT.lt 0 δ
          εs : Nat → Real
          left✝ : StrictAnti εs
          εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
          ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
          ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
          key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
          aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
          n : Nat
          hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
          hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
          εs_pos : LT.lt 0 (εs n)
          bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
          ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSp …
        -/
        rw [add_assoc, mul_comm]
        /-
          case h.refine_4
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
          P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
          Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
          f✝ f : BoundedContinuousFunction Ω Real
          f_nn : LE.le 0 f
          f_zero : Not (Eq (Norm.norm f) 0)
          norm_f_pos : LT.lt 0 (Norm.norm f)
          δ : Real
          δ_pos : LT.lt 0 δ
          εs : Nat → Real
          left✝ : StrictAnti εs
          εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
          ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
          ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
          key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
          aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
          n : Nat
          hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
          hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
          εs_pos : LT.lt 0 (εs n)
          bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
          ⊢ LE.le (HAdd.hAdd (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
        -/
        gcongr
        calc
          δ / 2 + ‖f‖ * (dist (μs n) ν + εs n)
          _ ≤ δ / 2 + ‖f‖ * (‖f‖⁻¹ * δ / 2) := by gcongr
          _ = δ := by field_simp; ring
        /-
          case h.refine_4.f_nn
          Ω : Type u_1
          inst✝² : MeasurableSpace Ω
          inst✝¹ : PseudoMetricSpace Ω
          inst✝ : OpensMeasurableSpace Ω
          μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
          hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
          P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
          Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
          f✝ f : BoundedContinuousFunction Ω Real
          f_nn : LE.le 0 f
          f_zero : Not (Eq (Norm.norm f) 0)
          norm_f_pos : LT.lt 0 (Norm.norm f)
          δ : Real
          δ_pos : LT.lt 0 δ
          εs : Nat → Real
          left✝ : StrictAnti εs
          εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
          ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
          ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
          key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
          aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
          n : Nat
          hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
          hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
          εs_pos : LT.lt 0 (εs n)
          bound : LE.le (MeasureTheory.integral ↑(Ps n) fun ω => f ω) (HAdd.hAdd (Measur …
          ⊢ (MeasureTheory.ae ↑P).EventuallyLE 0 ⇑f
        -/
      · exact Eventually.of_forall f_nn
        /-
          🎉 no goals
        -/
      /-
        case h.refine_1
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        ⊢ LT.lt 0 (HAdd.hAdd (Dist.dist (μs n) ν) (εs n))
      -/
    · positivity
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        ⊢ LT.lt (MeasureTheory.levyProkhorovEDist ↑(Ps n) ↑P) (ENNReal.ofReal (HAdd.hA …
      -/
    · rw [ENNReal.ofReal_add (by positivity) (by positivity), ← add_zero (levyProkhorovEDist _ _)]
      apply ENNReal.add_lt_add_of_le_of_lt (levyProkhorovEDist_ne_top _ _)
            (le_of_eq ?_) (ofReal_pos.mpr εs_pos)
      /-
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        ⊢ Eq (MeasureTheory.levyProkhorovEDist ↑(Ps n) ↑P) (ENNReal.ofReal (Dist.dist  …
      -/
      rw [LevyProkhorov.dist_def, levyProkhorovDist, ofReal_toReal (levyProkhorovEDist_ne_top _ _)]
      /-
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        ⊢ Eq (MeasureTheory.levyProkhorovEDist ↑(Ps n) ↑P) (MeasureTheory.levyProkhoro …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3
        Ω : Type u_1
        inst✝² : MeasurableSpace Ω
        inst✝¹ : PseudoMetricSpace Ω
        inst✝ : OpensMeasurableSpace Ω
        μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
        hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
        P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
        Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
        f✝ f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        f_zero : Not (Eq (Norm.norm f) 0)
        norm_f_pos : LT.lt 0 (Norm.norm f)
        δ : Real
        δ_pos : LT.lt 0 δ
        εs : Nat → Real
        left✝ : StrictAnti εs
        εs_lim : Filter.Tendsto εs Filter.atTop (nhds 0)
        ε_of_room : Filter.Tendsto (fun x => HAdd.hAdd (Dist.dist (μs x) ν) (εs x)) Fi …
        ε_of_room' : Filter.Tendsto (fun n => HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) F …
        key : Filter.Tendsto (Function.comp (fun ε => MeasureTheory.integral (MeasureT …
        aux : ∀ (z : Real), Membership.mem (nhds z) (Set.Iio (HAdd.hAdd z (HDiv.hDiv δ …
        n : Nat
        hn : LT.lt (Function.comp (fun ε => MeasureTheory.integral (MeasureTheory.Meas …
        hn' : LT.lt (HAdd.hAdd (Dist.dist (μs n) ν) (εs n)) (HDiv.hDiv (HMul.hMul (Inv …
        εs_pos : LT.lt 0 (εs n)
        ⊢ (MeasureTheory.ae ↑(Ps n)).EventuallyLE 0 ⇑f
      -/
    · exact Eventually.of_forall f_nn
      /-
        🎉 no goals
      -/
  · simp only [IsCoboundedUnder, IsCobounded, eventually_map, eventually_atTop,
               forall_exists_index]
    /-
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : PseudoMetricSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      μs : Nat → MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      ν : MeasureTheory.LevyProkhorov (MeasureTheory.ProbabilityMeasure Ω)
      hμs : Filter.Tendsto μs Filter.atTop (nhds ν)
      P : MeasureTheory.ProbabilityMeasure Ω := (MeasureTheory.LevyProkhorov.equiv ( …
      Ps : Nat → MeasureTheory.ProbabilityMeasure Ω := fun n => (MeasureTheory.LevyP …
      f✝ f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      f_zero : Not (Eq (Norm.norm f) 0)
      norm_f_pos : LT.lt 0 (Norm.norm f)
      δ : Real
      δ_pos : LT.lt 0 δ
      ⊢ Exists fun b => ∀ (a : Real) (x : Nat), (∀ (b : Nat), GE.ge b x → LE.le (Mea …
    -/
    refine ⟨0, fun a i hia ↦ le_trans (integral_nonneg f_nn) (hia i le_rfl)⟩
    /-
      🎉 no goals
    -/


/-- The topology of the Lévy-Prokhorov metric is at least as fine as the topology of convergence in
distribution. -/
theorem levyProkhorov_le_convergenceInDistribution :
    TopologicalSpace.coinduced (LevyProkhorov.equiv (α := ProbabilityMeasure Ω)) inferInstance
      ≤ (inferInstance : TopologicalSpace (ProbabilityMeasure Ω)) :=
  (LevyProkhorov.continuous_equiv_probabilityMeasure).coinduced_le


lemma ProbabilityMeasure.toMeasure_add_pos_gt_mem_nhds (P : ProbabilityMeasure Ω)
    {G : Set Ω} (G_open : IsOpen G) {ε : ℝ≥0∞} (ε_pos : 0 < ε) :
    {Q | P.toMeasure G < Q.toMeasure G + ε} ∈ 𝓝 P := by
  /-
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    ⊢ Membership.mem (nhds P) (setOf fun Q => LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε))
  -/
  by_cases easy : P.toMeasure G < ε
    /-
      case pos
      Ω : Type u_1
      inst✝² : PseudoMetricSpace Ω
      inst✝¹ : MeasurableSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      P : MeasureTheory.ProbabilityMeasure Ω
      G : Set Ω
      G_open : IsOpen G
      ε : ENNReal
      ε_pos : LT.lt 0 ε
      easy : LT.lt (↑P G) ε
      ⊢ Membership.mem (nhds P) (setOf fun Q => LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε))
    -/
  · exact Eventually.of_forall (fun _ ↦ lt_of_lt_of_le easy le_add_self)
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    easy : Not (LT.lt (↑P G) ε)
    ⊢ Membership.mem (nhds P) (setOf fun Q => LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε))
  -/
  by_cases ε_top : ε = ∞
    /-
      case pos
      Ω : Type u_1
      inst✝² : PseudoMetricSpace Ω
      inst✝¹ : MeasurableSpace Ω
      inst✝ : OpensMeasurableSpace Ω
      P : MeasureTheory.ProbabilityMeasure Ω
      G : Set Ω
      G_open : IsOpen G
      ε : ENNReal
      ε_pos : LT.lt 0 ε
      easy : Not (LT.lt (↑P G) ε)
      ε_top : Eq ε Top.top
      ⊢ Membership.mem (nhds P) (setOf fun Q => LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε))
    -/
  · simp [ε_top, measure_lt_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    easy : Not (LT.lt (↑P G) ε)
    ε_top : Not (Eq ε Top.top)
    ⊢ Membership.mem (nhds P) (setOf fun Q => LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε))
  -/
  simp only [not_lt] at easy
  have aux : P.toMeasure G - ε < liminf (fun Q ↦ Q.toMeasure G) (𝓝 P) := by
    apply lt_of_lt_of_le (ENNReal.sub_lt_self (measure_lt_top _ _).ne _ _)
        <| ProbabilityMeasure.le_liminf_measure_open_of_tendsto tendsto_id G_open
    · exact (lt_of_lt_of_le ε_pos easy).ne.symm
    · exact ε_pos.ne.symm
  filter_upwards [gt_mem_sets_of_limsInf_gt (α := ℝ≥0∞) isBounded_ge_of_bot
      (show P.toMeasure G - ε < limsInf ((𝓝 P).map (fun Q ↦ Q.toMeasure G)) from aux)] with Q hQ
  /-
    case h
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    ε_top : Not (Eq ε Top.top)
    easy : LE.le ε (↑P G)
    aux : LT.lt (HSub.hSub (↑P G) ε) (Filter.liminf (fun Q => ↑Q G) (nhds P))
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : Membership.mem (Set.preimage (fun Q => ↑Q G) (setOf fun x => LT.lt (HSub. …
    ⊢ LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε)
  -/
  simp only [preimage_setOf_eq, mem_setOf_eq] at hQ
  /-
    case h
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    ε_top : Not (Eq ε Top.top)
    easy : LE.le ε (↑P G)
    aux : LT.lt (HSub.hSub (↑P G) ε) (Filter.liminf (fun Q => ↑Q G) (nhds P))
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : LT.lt (HSub.hSub (↑P G) ε) (↑Q G)
    ⊢ LT.lt (↑P G) (HAdd.hAdd (↑Q G) ε)
  -/
  convert ENNReal.add_lt_add_right ε_top hQ
  /-
    case h.e'_3
    Ω : Type u_1
    inst✝² : PseudoMetricSpace Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    G : Set Ω
    G_open : IsOpen G
    ε : ENNReal
    ε_pos : LT.lt 0 ε
    ε_top : Not (Eq ε Top.top)
    easy : LE.le ε (↑P G)
    aux : LT.lt (HSub.hSub (↑P G) ε) (Filter.liminf (fun Q => ↑Q G) (nhds P))
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : LT.lt (HSub.hSub (↑P G) ε) (↑Q G)
    ⊢ Eq (↑P G) (HAdd.hAdd (HSub.hSub (↑P G) ε) ε)
  -/
  exact (tsub_add_cancel_of_le easy).symm
  /-
    🎉 no goals
  -/


variable (Ω) in
/-- In a separable pseudometric space, for any ε > 0 there exists a countable collection of
disjoint Borel measurable subsets of diameter at most ε that cover the whole space. -/
lemma SeparableSpace.exists_measurable_partition_diam_le {ε : ℝ} (ε_pos : 0 < ε) :
    ∃ (As : ℕ → Set Ω), (∀ n, MeasurableSet (As n)) ∧ (∀ n, Bornology.IsBounded (As n)) ∧
        (∀ n, diam (As n) ≤ ε) ∧ (⋃ n, As n = univ) ∧
        (Pairwise (fun (n m : ℕ) ↦ Disjoint (As n) (As m))) := by
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  by_cases X_emp : IsEmpty Ω
  · refine ⟨fun _ ↦ ∅, fun _ ↦ MeasurableSet.empty, fun _ ↦ Bornology.isBounded_empty, ?_, ?_,
            fun _ _ _ ↦ disjoint_of_subsingleton⟩
      /-
        case pos.refine_1
        Ω : Type u_1
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : MeasurableSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : TopologicalSpace.SeparableSpace Ω
        ε : Real
        ε_pos : LT.lt 0 ε
        X_emp : IsEmpty Ω
        ⊢ ∀ (n : Nat), LE.le (Metric.diam ((fun x => EmptyCollection.emptyCollection)  …
      -/
    · intro n
      /-
        case pos.refine_1
        Ω : Type u_1
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : MeasurableSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : TopologicalSpace.SeparableSpace Ω
        ε : Real
        ε_pos : LT.lt 0 ε
        X_emp : IsEmpty Ω
        n : Nat
        ⊢ LE.le (Metric.diam ((fun x => EmptyCollection.emptyCollection) n)) ε
      -/
      simpa only [diam_empty] using LT.lt.le ε_pos
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        Ω : Type u_1
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : MeasurableSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : TopologicalSpace.SeparableSpace Ω
        ε : Real
        ε_pos : LT.lt 0 ε
        X_emp : IsEmpty Ω
        ⊢ Eq (Set.iUnion fun n => (fun x => EmptyCollection.emptyCollection) n) Set.univ
      -/
    · simp only [iUnion_empty]
      /-
        case pos.refine_2
        Ω : Type u_1
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : MeasurableSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : TopologicalSpace.SeparableSpace Ω
        ε : Real
        ε_pos : LT.lt 0 ε
        X_emp : IsEmpty Ω
        ⊢ Eq EmptyCollection.emptyCollection Set.univ
      -/
      apply Eq.symm
      /-
        case pos.refine_2.h
        Ω : Type u_1
        inst✝³ : PseudoMetricSpace Ω
        inst✝² : MeasurableSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        inst✝ : TopologicalSpace.SeparableSpace Ω
        ε : Real
        ε_pos : LT.lt 0 ε
        X_emp : IsEmpty Ω
        ⊢ Eq Set.univ EmptyCollection.emptyCollection
      -/
      simp only [univ_eq_empty_iff, X_emp]
      /-
        🎉 no goals
      -/
  /-
    case neg
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Not (IsEmpty Ω)
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  rw [not_isEmpty_iff] at X_emp
  /-
    case neg
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Nonempty Ω
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  obtain ⟨xs, xs_dense⟩ := exists_dense_seq Ω
  /-
    case neg.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Nonempty Ω
    xs : Nat → Ω
    xs_dense : DenseRange xs
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  have half_ε_pos : 0 < ε / 2 := half_pos ε_pos
  /-
    case neg.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Nonempty Ω
    xs : Nat → Ω
    xs_dense : DenseRange xs
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  set Bs := fun n ↦ Metric.ball (xs n) (ε/2)
  /-
    case neg.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Nonempty Ω
    xs : Nat → Ω
    xs_dense : DenseRange xs
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  set As := disjointed Bs
  /-
    case neg.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ε : Real
    ε_pos : LT.lt 0 ε
    X_emp : Nonempty Ω
    xs : Nat → Ω
    xs_dense : DenseRange xs
    half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
    Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
    As : Nat → Set Ω := disjointed Bs
    ⊢ Exists fun As => And (∀ (n : Nat), MeasurableSet (As n)) (And (∀ (n : Nat),  …
  -/
  refine ⟨As, ?_, ?_, ?_, ?_, ?_⟩
    /-
      case neg.intro.refine_1
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      ⊢ ∀ (n : Nat), MeasurableSet (As n)
    -/
  · exact MeasurableSet.disjointed (fun n ↦ measurableSet_ball)
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.refine_2
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      ⊢ ∀ (n : Nat), Bornology.IsBounded (As n)
    -/
  · exact fun n ↦ Bornology.IsBounded.subset isBounded_ball <| disjointed_subset Bs n
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.refine_3
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      ⊢ ∀ (n : Nat), LE.le (Metric.diam (As n)) ε
    -/
  · intro n
    /-
      case neg.intro.refine_3
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      n : Nat
      ⊢ LE.le (Metric.diam (As n)) ε
    -/
    apply (diam_mono (disjointed_subset Bs n) isBounded_ball).trans
    /-
      case neg.intro.refine_3
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      n : Nat
      ⊢ LE.le (Metric.diam (Bs n)) ε
    -/
    convert diam_ball half_ε_pos.le
    /-
      case h.e'_4
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      n : Nat
      ⊢ Eq ε (HMul.hMul 2 (HDiv.hDiv ε 2))
    -/
    ring
    /-
      🎉 no goals
    -/
  · have aux : ⋃ n, Bs n = univ := by
      convert DenseRange.iUnion_uniformity_ball xs_dense <| Metric.dist_mem_uniformity half_ε_pos
      exact (ball_eq_ball' _ _).symm
    /-
      case neg.intro.refine_4
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      aux : Eq (Set.iUnion fun n => Bs n) Set.univ
      ⊢ Eq (Set.iUnion fun n => As n) Set.univ
    -/
    simpa only [← aux] using iUnion_disjointed
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.refine_5
      Ω : Type u_1
      inst✝³ : PseudoMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      inst✝ : TopologicalSpace.SeparableSpace Ω
      ε : Real
      ε_pos : LT.lt 0 ε
      X_emp : Nonempty Ω
      xs : Nat → Ω
      xs_dense : DenseRange xs
      half_ε_pos : LT.lt 0 (HDiv.hDiv ε 2)
      Bs : Nat → Set Ω := fun n => Metric.ball (xs n) (HDiv.hDiv ε 2)
      As : Nat → Set Ω := disjointed Bs
      ⊢ Pairwise fun n m => Disjoint (As n) (As m)
    -/
  · exact disjoint_disjointed Bs
    /-
      🎉 no goals
    -/


lemma LevyProkhorov.continuous_equiv_symm_probabilityMeasure :
    Continuous (LevyProkhorov.equiv (α := ProbabilityMeasure Ω)).symm := by
  -- We check continuity of `id : ProbabilityMeasure Ω → LevyProkhorov (ProbabilityMeasure Ω)` at
  -- each point `P : ProbabilityMeasure Ω`.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ⊢ Continuous ⇑(MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMea …
  -/
  rw [continuous_iff_continuousAt]
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    ⊢ ∀ (x : MeasureTheory.ProbabilityMeasure Ω), ContinuousAt (⇑(MeasureTheory.Le …
  -/
  intro P
  -- To check continuity, fix `ε > 0`. To leave some wiggle room, be ready to use `ε/3 > 0` instead.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ⊢ ContinuousAt (⇑(MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probability …
  -/
  rw [continuousAt_iff']
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist ((Mea …
  -/
  intro ε ε_pos
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.e …
  -/
  have third_ε_pos : 0 < ε / 3 := by linarith
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.e …
  -/
  have third_ε_pos' : 0 < ENNReal.ofReal (ε / 3) := ofReal_pos.mpr third_ε_pos
  -- First use separability to choose a countable partition of `Ω` into measurable
  -- subsets `Es n ⊆ Ω` of small diameter, `diam (Es n) < ε/3`.
  obtain ⟨Es, Es_mble, Es_bdd, Es_diam, Es_cover, Es_disjoint⟩ :=
    SeparableSpace.exists_measurable_partition_diam_le Ω third_ε_pos
  -- Instead of the whole space `Ω = ⋃ n ∈ ℕ, Es n`, focus on a large but finite
  -- union `⋃ n < N, Es n`, chosen in such a way that the complement has small `P`-mass,
  -- `P (⋃ n < N, Es n)ᶜ < ε/3`.
  obtain ⟨N, hN⟩ : ∃ N, P.toMeasure (⋃ j ∈ Iio N, Es j)ᶜ < ENNReal.ofReal (ε/3) := by
    have exhaust := @tendsto_measure_biUnion_Ici_zero_of_pairwise_disjoint Ω _ P.toMeasure _
                    Es (fun n ↦ (Es_mble n).nullMeasurableSet) Es_disjoint
    simp only [tendsto_atTop_nhds, Function.comp_apply] at exhaust
    obtain ⟨N, hN⟩ := exhaust (Iio (ENNReal.ofReal (ε / 3))) third_ε_pos' isOpen_Iio
    refine ⟨N, ?_⟩
    have rewr : ⋃ i, ⋃ (_ : N ≤ i), Es i = (⋃ i, ⋃ (_ : i < N), Es i)ᶜ := by
      simpa only [mem_Iio, compl_Iio, mem_Ici] using
        (biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ Es_cover Es_disjoint (Iio N)).symm
    simpa only [mem_Iio, ← rewr, gt_iff_lt] using hN N le_rfl
  -- With the finite `N` fixed above, consider the finite collection of open sets of the form
  -- `Gs J = thickening (ε/3) (⋃ j ∈ J, Es j)`, where `J ⊆ {0, 1, ..., N-1}`.
  /-
    case intro.intro.intro.intro.intro.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.e …
  -/
  have Js_finite : Set.Finite {J | J ⊆ Iio N} := Finite.finite_subsets <| finite_Iio N
  /-
    case intro.intro.intro.intro.intro.intro
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.e …
  -/
  set Gs := (fun (J : Set ℕ) ↦ thickening (ε/3) (⋃ j ∈ J, Es j)) '' {J | J ⊆ Iio N}
  have Gs_open : ∀ (J : Set ℕ), IsOpen (thickening (ε/3) (⋃ j ∈ J, Es j)) :=
    fun J ↦ isOpen_thickening
  -- Any open set `G ⊆ Ω` determines a neighborhood of `P` consisting of those `Q` that
  -- satisfy `P G < Q G + ε/3`.
  have mem_nhds_P (G : Set Ω) (G_open : IsOpen G) :
      {Q | P.toMeasure G < Q.toMeasure G + ENNReal.ofReal (ε/3)} ∈ 𝓝 P :=
    P.toMeasure_add_pos_gt_mem_nhds G_open third_ε_pos'
  -- Assume that `Q` is in the neighborhood of `P` such that for each `J ⊆ {0, 1, ..., N-1}`
  -- we have `P (Gs J) < Q (Gs J) + ε/3`.
  filter_upwards [(Finset.iInter_mem_sets Js_finite.toFinset).mpr <|
                    fun J _ ↦ mem_nhds_P _ (Gs_open J)] with Q hQ
  /-
    case h
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : Membership.mem (Set.iInter fun i => Set.iInter fun h => setOf fun Q => LT …
    ⊢ LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probabil …
  -/
  simp only [Finite.mem_toFinset, mem_setOf_eq, thickening_iUnion, mem_iInter] at hQ
  -- Note that in order to show that the Lévy-Prokhorov distance `LPdist P Q` is small (`≤ 2*ε/3`),
  -- it suffices to show that for arbitrary subsets `B ⊆ Ω`, the measure `P B` is bounded above up
  -- to a small error by the `Q`-measure of a small thickening of `B`.
  /-
    case h
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    ⊢ LT.lt (Dist.dist ((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probabil …
  -/
  apply lt_of_le_of_lt ?_ (show 2*(ε/3) < ε by linarith)
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    ⊢ LE.le (Dist.dist ((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probabil …
  -/
  rw [dist_comm]
  -- Fix an arbitrary set `B ⊆ Ω`, and an arbitrary `δ > 2*ε/3` to gain some room for error
  -- and for thickening.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    ⊢ LE.le (Dist.dist ((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probabil …
  -/
  apply levyProkhorovDist_le_of_forall_le _ _ (by linarith) (fun δ B δ_gt _ ↦ ?_)
  -- Let `JB ⊆ {0, 1, ..., N-1}` consist of those indices `j` such that `B` intersects `Es j`.
  -- Then the open set `Gs JB` approximates `B` rather well:
  -- except for what happens in the small complement `(⋃ n < N, Es n)ᶜ`, the set `B` is
  -- contained in `Gs JB`, and conversely `Gs JB` only contains points within `δ` from `B`.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    ⊢ LE.le (↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMeasur …
  -/
  set JB := {i | B ∩ Es i ≠ ∅ ∧ i ∈ Iio N}
  have B_subset : B ⊆ (⋃ i ∈ JB, thickening (ε/3) (Es i)) ∪ (⋃ j ∈ Iio N, Es j)ᶜ := by
    suffices B ⊆ (⋃ i ∈ JB, thickening (ε/3) (Es i)) ∪ (⋃ j ∈ Ici N, Es j) by
      refine this.trans <| union_subset_union le_rfl ?_
      intro ω hω
      simp only [mem_Ici, mem_iUnion, exists_prop] at hω
      obtain ⟨i, i_large, ω_in_Esi⟩ := hω
      by_contra con
      simp only [mem_Iio, compl_iUnion, mem_iInter, mem_compl_iff, not_forall, not_not,
                  exists_prop] at con
      obtain ⟨j, j_small, ω_in_Esj⟩ := con
      exact disjoint_left.mp (Es_disjoint (show j ≠ i by omega)) ω_in_Esj ω_in_Esi
    intro ω ω_in_B
    obtain ⟨i, hi⟩ := show ∃ n, ω ∈ Es n by simp only [← mem_iUnion, Es_cover, mem_univ]
    simp only [mem_Ici, mem_union, mem_iUnion, exists_prop]
    by_cases i_small : i ∈ Iio N
    · refine Or.inl ⟨i, ?_, self_subset_thickening third_ε_pos _ hi⟩
      simp only [mem_Iio, mem_setOf_eq, JB]
      refine ⟨nonempty_iff_ne_empty.mp <| Set.nonempty_of_mem <| mem_inter ω_in_B hi, i_small⟩
    · exact Or.inr ⟨i, by simpa only [mem_Iio, not_lt] using i_small, hi⟩
  have subset_thickB : ⋃ i ∈ JB, thickening (ε / 3) (Es i) ⊆ thickening δ B := by
    intro ω ω_in_U
    simp only [mem_setOf_eq, mem_iUnion, exists_prop] at ω_in_U
    obtain ⟨k, ⟨B_intersects, _⟩, ω_in_thEk⟩ := ω_in_U
    rw [mem_thickening_iff] at ω_in_thEk ⊢
    obtain ⟨w, w_in_Ek, w_near⟩ := ω_in_thEk
    obtain ⟨z, ⟨z_in_B, z_in_Ek⟩⟩ := nonempty_iff_ne_empty.mpr B_intersects
    refine ⟨z, z_in_B, lt_of_le_of_lt (dist_triangle ω w z) ?_⟩
    apply lt_of_le_of_lt (add_le_add w_near.le <|
            (dist_le_diam_of_mem (Es_bdd k) w_in_Ek z_in_Ek).trans <| Es_diam k)
    linarith
  -- We use the resulting upper bound `P B ≤ P (Gs JB) + P (small complement)`.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    ⊢ LE.le (↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.ProbabilityMeasur …
  -/
  apply (measure_mono B_subset).trans ((measure_union_le _ _).trans ?_)
  -- From the choice of `Q` in a suitable neighborhood, we have `P (Gs JB) < Q (Gs JB) + ε/3`.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    hQ : ∀ (i : Set Nat), HasSubset.Subset i (Set.Iio N) → LT.lt (↑P (Set.iUnion f …
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    ⊢ LE.le (HAdd.hAdd (↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probab …
  -/
  specialize hQ _ (show JB ⊆ Iio N from fun _ h ↦ h.2)
  -- Now it remains to add the pieces and use the above estimates.
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    hQ : LT.lt (↑P (Set.iUnion fun i => Set.iUnion fun i_1 => Metric.thickening (H …
    ⊢ LE.le (HAdd.hAdd (↑((MeasureTheory.LevyProkhorov.equiv (MeasureTheory.Probab …
  -/
  apply (add_le_add hQ.le hN.le).trans
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    hQ : LT.lt (↑P (Set.iUnion fun i => Set.iUnion fun i_1 => Metric.thickening (H …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (↑Q (Set.iUnion fun i => Set.iUnion fun i_1 => M …
  -/
  rw [add_assoc, ← ENNReal.ofReal_add third_ε_pos.le third_ε_pos.le, ← two_mul]
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    hQ : LT.lt (↑P (Set.iUnion fun i => Set.iUnion fun i_1 => Metric.thickening (H …
    ⊢ LE.le (HAdd.hAdd (↑Q (Set.iUnion fun i => Set.iUnion fun i_1 => Metric.thick …
  -/
  apply add_le_add (measure_mono subset_thickB) (ofReal_le_ofReal _)
  /-
    Ω : Type u_1
    inst✝³ : PseudoMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : TopologicalSpace.SeparableSpace Ω
    P : MeasureTheory.ProbabilityMeasure Ω
    ε : Real
    ε_pos : GT.gt ε 0
    third_ε_pos : LT.lt 0 (HDiv.hDiv ε 3)
    third_ε_pos' : LT.lt 0 (ENNReal.ofReal (HDiv.hDiv ε 3))
    Es : Nat → Set Ω
    Es_mble : ∀ (n : Nat), MeasurableSet (Es n)
    Es_bdd : ∀ (n : Nat), Bornology.IsBounded (Es n)
    Es_diam : ∀ (n : Nat), LE.le (Metric.diam (Es n)) (HDiv.hDiv ε 3)
    Es_cover : Eq (Set.iUnion fun n => Es n) Set.univ
    Es_disjoint : Pairwise fun n m => Disjoint (Es n) (Es m)
    N : Nat
    hN : LT.lt (↑P (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun h => Es j)) …
    Js_finite : (setOf fun J => HasSubset.Subset J (Set.Iio N)).Finite
    Gs : Set (Set Ω) := Set.image (fun J => Metric.thickening (HDiv.hDiv ε 3) (Set …
    Gs_open : ∀ (J : Set Nat), IsOpen (Metric.thickening (HDiv.hDiv ε 3) (Set.iUni …
    mem_nhds_P : ∀ (G : Set Ω), IsOpen G → Membership.mem (nhds P) (setOf fun Q => …
    Q : MeasureTheory.ProbabilityMeasure Ω
    δ : Real
    B : Set Ω
    δ_gt : LT.lt (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
    x✝ : MeasurableSet B
    JB : Set Nat := setOf fun i => And (Ne (Inter.inter B (Es i)) EmptyCollection. …
    B_subset : HasSubset.Subset B (Union.union (Set.iUnion fun i => Set.iUnion fun …
    subset_thickB : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Metr …
    hQ : LT.lt (↑P (Set.iUnion fun i => Set.iUnion fun i_1 => Metric.thickening (H …
    ⊢ LE.le (HMul.hMul 2 (HDiv.hDiv ε 3)) δ
  -/
  exact δ_gt.le
  /-
    🎉 no goals
  -/


/-- The topology of the Lévy-Prokhorov metric on probability measures on a separable space
coincides with the topology of convergence in distribution. -/
theorem levyProkhorov_eq_convergenceInDistribution :
    (inferInstance : TopologicalSpace (ProbabilityMeasure Ω))
      = TopologicalSpace.coinduced (LevyProkhorov.equiv _) inferInstance :=
  le_antisymm (LevyProkhorov.continuous_equiv_symm_probabilityMeasure (Ω := Ω)).coinduced_le
              levyProkhorov_le_convergenceInDistribution


/-- The identity map is a homeomorphism from `ProbabilityMeasure Ω` with the topology of
convergence in distribution to `ProbabilityMeasure Ω` with the Lévy-Prokhorov (pseudo)metric. -/
def homeomorph_probabilityMeasure_levyProkhorov :
    ProbabilityMeasure Ω ≃ₜ LevyProkhorov (ProbabilityMeasure Ω) where
  toFun := LevyProkhorov.equiv _
  invFun := (LevyProkhorov.equiv _).symm
  left_inv := congrFun rfl
  right_inv := congrFun rfl
  continuous_toFun := LevyProkhorov.continuous_equiv_symm_probabilityMeasure
  continuous_invFun := LevyProkhorov.continuous_equiv_probabilityMeasure


/-- The topology of convergence in distribution on a separable space is pseudo-metrizable. -/
instance (X : Type*) [TopologicalSpace X] [PseudoMetrizableSpace X] [SeparableSpace X]
    [MeasurableSpace X] [OpensMeasurableSpace X] :
    PseudoMetrizableSpace (ProbabilityMeasure X) :=
  letI : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  (homeomorph_probabilityMeasure_levyProkhorov (Ω := X)).isInducing.pseudoMetrizableSpace


/-- The topology of convergence in distribution on a separable Borel space is metrizable. -/
instance instMetrizableSpaceProbabilityMeasure (X : Type*) [TopologicalSpace X]
    [PseudoMetrizableSpace X] [SeparableSpace X] [MeasurableSpace X] [BorelSpace X] :
    MetrizableSpace (ProbabilityMeasure X) := by
  /-
    Ω : Type u_1
    inst✝⁸ : PseudoMetricSpace Ω
    inst✝⁷ : MeasurableSpace Ω
    inst✝⁶ : OpensMeasurableSpace Ω
    inst✝⁵ : TopologicalSpace.SeparableSpace Ω
    X : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    ⊢ TopologicalSpace.MetrizableSpace (MeasureTheory.ProbabilityMeasure X)
  -/
  letI : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  /-
    Ω : Type u_1
    inst✝⁸ : PseudoMetricSpace Ω
    inst✝⁷ : MeasurableSpace Ω
    inst✝⁶ : OpensMeasurableSpace Ω
    inst✝⁵ : TopologicalSpace.SeparableSpace Ω
    X : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ TopologicalSpace.MetrizableSpace (MeasureTheory.ProbabilityMeasure X)
  -/
  exact homeomorph_probabilityMeasure_levyProkhorov.isEmbedding.metrizableSpace
  /-
    🎉 no goals
  -/


