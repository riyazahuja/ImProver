local infixr:25 " →ₛ " => SimpleFunc


/-- Given a simple function `f` with values in `ℝ≥0`, there exists a lower semicontinuous
function `g ≥ f` with integral arbitrarily close to that of `f`. Formulation in terms of
`lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem SimpleFunc.exists_le_lowerSemicontinuous_lintegral_ge (f : α →ₛ ℝ≥0) {ε : ℝ≥0∞}
    (ε0 : ε ≠ 0) :
    ∃ g : α → ℝ≥0, (∀ x, f x ≤ g x) ∧ LowerSemicontinuous g ∧
      (∫⁻ x, g x ∂μ) ≤ (∫⁻ x, f x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : MeasureTheory.SimpleFunc α NNReal
    ε : ENNReal
    ε0 : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LE.le (f x) (g x)) (And (LowerSemicontinuous …
  -/
  induction' f using MeasureTheory.SimpleFunc.induction with c s hs f₁ f₂ _ h₁ h₂ generalizing ε
    /-
      case h_ind
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      ε : ENNReal
      ε0 : Ne ε 0
      ⊢ Exists fun g => And (∀ (x : α), LE.le ((MeasureTheory.SimpleFunc.piecewise s …
    -/
  · let f := SimpleFunc.piecewise s hs (SimpleFunc.const α c) (SimpleFunc.const α 0)
    /-
      case h_ind
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      ε : ENNReal
      ε0 : Ne ε 0
      f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
      ⊢ Exists fun g => And (∀ (x : α), LE.le ((MeasureTheory.SimpleFunc.piecewise s …
    -/
    by_cases h : ∫⁻ x, f x ∂μ = ⊤
    · refine
        ⟨fun _ => c, fun x => ?_, lowerSemicontinuous_const, by
          simp only [f, _root_.top_add, le_top, h]⟩
      simp only [SimpleFunc.coe_const, SimpleFunc.const_zero, SimpleFunc.coe_zero,
        Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise]
      /-
        case pos
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        ε0 : Ne ε 0
        f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
        h : Eq (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        x : α
        ⊢ LE.le (s.indicator (Function.const α c) x) c
      -/
      exact Set.indicator_le_self _ _ _
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      ε : ENNReal
      ε0 : Ne ε 0
      f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
      h : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top)
      ⊢ Exists fun g => And (∀ (x : α), LE.le ((MeasureTheory.SimpleFunc.piecewise s …
    -/
    by_cases hc : c = 0
      /-
        case pos
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        ε0 : Ne ε 0
        f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
        h : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top)
        hc : Eq c 0
        ⊢ Exists fun g => And (∀ (x : α), LE.le ((MeasureTheory.SimpleFunc.piecewise s …
      -/
    · refine ⟨fun _ => 0, ?_, lowerSemicontinuous_const, ?_⟩
      · classical
        simp only [hc, Set.indicator_zero', Pi.zero_apply, SimpleFunc.const_zero, imp_true_iff,
          eq_self_iff_true, SimpleFunc.coe_zero, Set.piecewise_eq_indicator,
          SimpleFunc.coe_piecewise, le_zero_iff]
        /-
          case pos.refine_2
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : MeasurableSpace α
          inst✝¹ : BorelSpace α
          μ : MeasureTheory.Measure α
          inst✝ : μ.WeaklyRegular
          c : NNReal
          s : Set α
          hs : MeasurableSet s
          ε : ENNReal
          ε0 : Ne ε 0
          f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
          h : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top)
          hc : Eq c 0
          ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑((fun x => 0) x)) (HAdd.hAdd (Mea …
        -/
      · simp only [lintegral_const, zero_mul, zero_le, ENNReal.coe_zero]
        /-
          🎉 no goals
        -/
    have ne_top : μ s ≠ ⊤ := by
      classical
      simpa [f, hs, hc, lt_top_iff_ne_top, SimpleFunc.coe_const,
        Function.const_apply, lintegral_const, ENNReal.coe_indicator, Set.univ_inter,
        ENNReal.coe_ne_top, MeasurableSet.univ, ENNReal.mul_eq_top, SimpleFunc.const_zero,
        lintegral_indicator, ENNReal.coe_eq_zero, Ne, not_false_iff,
        SimpleFunc.coe_zero, Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise,
        restrict_apply] using h
    have : μ s < μ s + ε / c := by
      have : (0 : ℝ≥0∞) < ε / c := ENNReal.div_pos_iff.2 ⟨ε0, ENNReal.coe_ne_top⟩
      simpa using ENNReal.add_lt_add_left ne_top this
    obtain ⟨u, su, u_open, μu⟩ : ∃ (u : _), u ⊇ s ∧ IsOpen u ∧ μ u < μ s + ε / c :=
      s.exists_isOpen_lt_of_lt _ this
    refine ⟨Set.indicator u fun _ => c,
            fun x => ?_, u_open.lowerSemicontinuous_indicator (zero_le _), ?_⟩
    · simp only [SimpleFunc.coe_const, SimpleFunc.const_zero, SimpleFunc.coe_zero,
        Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise]
      /-
        case neg.intro.intro.intro.refine_1
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        ε0 : Ne ε 0
        f : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.piecewise s  …
        h : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top)
        hc : Not (Eq c 0)
        ne_top : Ne (μ s) Top.top
        this : LT.lt (μ s) (HAdd.hAdd (μ s) (HDiv.hDiv ε ↑c))
        u : Set α
        su : Superset u s
        u_open : IsOpen u
        μu : LT.lt (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε ↑c))
        x : α
        ⊢ LE.le (s.indicator (Function.const α c) x) (u.indicator (fun x => c) x)
      -/
      exact Set.indicator_le_indicator_of_subset su (fun x => zero_le _) _
      /-
        🎉 no goals
      -/
    · suffices (c : ℝ≥0∞) * μ u ≤ c * μ s + ε by
        classical
        simpa only [ENNReal.coe_indicator, u_open.measurableSet, lintegral_indicator,
          lintegral_const, MeasurableSet.univ, Measure.restrict_apply, Set.univ_inter, const_zero,
          coe_piecewise, coe_const, coe_zero, Set.piecewise_eq_indicator, Function.const_apply, hs]
      calc
        (c : ℝ≥0∞) * μ u ≤ c * (μ s + ε / c) := mul_le_mul_left' μu.le _
        _ = c * μ s + ε := by
          simp_rw [mul_add]
          rw [ENNReal.mul_div_cancel _ ENNReal.coe_ne_top]
          simpa using hc

    /-
      case h_add
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      ⊢ Exists fun g => And (∀ (x : α), LE.le ((HAdd.hAdd f₁ f₂) x) (g x)) (And (Low …
    -/
  · rcases h₁ (ENNReal.half_pos ε0).ne' with ⟨g₁, f₁_le_g₁, g₁cont, g₁int⟩
    /-
      case h_add.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (f₁ x) (g₁ x)
      g₁cont : LowerSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (HAdd.hAdd (Measure …
      ⊢ Exists fun g => And (∀ (x : α), LE.le ((HAdd.hAdd f₁ f₂) x) (g x)) (And (Low …
    -/
    rcases h₂ (ENNReal.half_pos ε0).ne' with ⟨g₂, f₂_le_g₂, g₂cont, g₂int⟩
    refine
      ⟨fun x => g₁ x + g₂ x, fun x => add_le_add (f₁_le_g₁ x) (f₂_le_g₂ x), g₁cont.add g₂cont, ?_⟩
    /-
      case h_add.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (f₁ x) (g₁ x)
      g₁cont : LowerSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (f₂ x) (g₂ x)
      g₂cont : LowerSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) (HAdd.hAdd (Measure …
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑((fun x => HAdd.hAdd (g₁ x) (g₂ x …
    -/
    simp only [SimpleFunc.coe_add, ENNReal.coe_add, Pi.add_apply]
    rw [lintegral_add_left f₁.measurable.coe_nnreal_ennreal,
      lintegral_add_left g₁cont.measurable.coe_nnreal_ennreal]
    /-
      case h_add.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (f₁ x) (g₁ x)
      g₁cont : LowerSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (f₂ x) (g₂ x)
      g₂cont : LowerSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) (HAdd.hAdd (Measure …
      ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(g₁ a)) (MeasureTheory …
    -/
    convert add_le_add g₁int g₂int using 1
    /-
      case h.e'_4
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (f₁ x) (g₁ x)
      g₁cont : LowerSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (f₂ x) (g₂ x)
      g₂cont : LowerSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) (HAdd.hAdd (Measure …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(f₁ a)) (Measu …
    -/
    conv_lhs => rw [← ENNReal.add_halves ε]
    /-
      case h.e'_4
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₁ x) (g …
      h₂ : ∀ {ε : ENNReal}, Ne ε 0 → Exists fun g => And (∀ (x : α), LE.le (f₂ x) (g …
      ε : ENNReal
      ε0 : Ne ε 0
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (f₁ x) (g₁ x)
      g₁cont : LowerSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (f₂ x) (g₂ x)
      g₂cont : LowerSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) (HAdd.hAdd (Measure …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(f₁ a)) (Measu …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/

-- Porting note: errors with
-- `ambiguous identifier 'eapproxDiff', possible interpretations:`
-- `[SimpleFunc.eapproxDiff, SimpleFunc.eapproxDiff]`
-- open SimpleFunc (eapproxDiff tsum_eapproxDiff)


/-- Given a measurable function `f` with values in `ℝ≥0`, there exists a lower semicontinuous
function `g ≥ f` with integral arbitrarily close to that of `f`. Formulation in terms of
`lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_le_lowerSemicontinuous_lintegral_ge (f : α → ℝ≥0∞) (hf : Measurable f) {ε : ℝ≥0∞}
    (εpos : ε ≠ 0) :
    ∃ g : α → ℝ≥0∞,
      (∀ x, f x ≤ g x) ∧ LowerSemicontinuous g ∧ (∫⁻ x, g x ∂μ) ≤ (∫⁻ x, f x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → ENNReal
    hf : Measurable f
    ε : ENNReal
    εpos : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LE.le (f x) (g x)) (And (LowerSemicontinuous …
  -/
  rcases ENNReal.exists_pos_sum_of_countable' εpos ℕ with ⟨δ, δpos, hδ⟩
  have :
    ∀ n,
      ∃ g : α → ℝ≥0,
        (∀ x, SimpleFunc.eapproxDiff f n x ≤ g x) ∧
          LowerSemicontinuous g ∧
            (∫⁻ x, g x ∂μ) ≤ (∫⁻ x, SimpleFunc.eapproxDiff f n x ∂μ) + δ n :=
    fun n =>
    SimpleFunc.exists_le_lowerSemicontinuous_lintegral_ge μ (SimpleFunc.eapproxDiff f n)
      (δpos n).ne'
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → ENNReal
    hf : Measurable f
    ε : ENNReal
    εpos : Ne ε 0
    δ : Nat → ENNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    hδ : LT.lt (tsum fun i => δ i) ε
    this : ∀ (n : Nat), Exists fun g => And (∀ (x : α), LE.le ((MeasureTheory.Simp …
    ⊢ Exists fun g => And (∀ (x : α), LE.le (f x) (g x)) (And (LowerSemicontinuous …
  -/
  choose g f_le_g gcont hg using this
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → ENNReal
    hf : Measurable f
    ε : ENNReal
    εpos : Ne ε 0
    δ : Nat → ENNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    hδ : LT.lt (tsum fun i => δ i) ε
    g : Nat → α → NNReal
    f_le_g : ∀ (n : Nat) (x : α), LE.le ((MeasureTheory.SimpleFunc.eapproxDiff f n …
    gcont : ∀ (n : Nat), LowerSemicontinuous (g n)
    hg : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun x => ↑(g n x)) (HAdd.hA …
    ⊢ Exists fun g => And (∀ (x : α), LE.le (f x) (g x)) (And (LowerSemicontinuous …
  -/
  refine ⟨fun x => ∑' n, g n x, fun x => ?_, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f : α → ENNReal
      hf : Measurable f
      ε : ENNReal
      εpos : Ne ε 0
      δ : Nat → ENNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      hδ : LT.lt (tsum fun i => δ i) ε
      g : Nat → α → NNReal
      f_le_g : ∀ (n : Nat) (x : α), LE.le ((MeasureTheory.SimpleFunc.eapproxDiff f n …
      gcont : ∀ (n : Nat), LowerSemicontinuous (g n)
      hg : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun x => ↑(g n x)) (HAdd.hA …
      x : α
      ⊢ LE.le (f x) ((fun x => tsum fun n => ↑(g n x)) x)
    -/
  · rw [← SimpleFunc.tsum_eapproxDiff f hf]
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f : α → ENNReal
      hf : Measurable f
      ε : ENNReal
      εpos : Ne ε 0
      δ : Nat → ENNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      hδ : LT.lt (tsum fun i => δ i) ε
      g : Nat → α → NNReal
      f_le_g : ∀ (n : Nat) (x : α), LE.le ((MeasureTheory.SimpleFunc.eapproxDiff f n …
      gcont : ∀ (n : Nat), LowerSemicontinuous (g n)
      hg : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun x => ↑(g n x)) (HAdd.hA …
      x : α
      ⊢ LE.le (tsum fun n => ↑((MeasureTheory.SimpleFunc.eapproxDiff f n) x)) ((fun  …
    -/
    exact ENNReal.tsum_le_tsum fun n => ENNReal.coe_le_coe.2 (f_le_g n x)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f : α → ENNReal
      hf : Measurable f
      ε : ENNReal
      εpos : Ne ε 0
      δ : Nat → ENNReal
      δpos : ∀ (i : Nat), LT.lt 0 (δ i)
      hδ : LT.lt (tsum fun i => δ i) ε
      g : Nat → α → NNReal
      f_le_g : ∀ (n : Nat) (x : α), LE.le ((MeasureTheory.SimpleFunc.eapproxDiff f n …
      gcont : ∀ (n : Nat), LowerSemicontinuous (g n)
      hg : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun x => ↑(g n x)) (HAdd.hA …
      ⊢ LowerSemicontinuous fun x => tsum fun n => ↑(g n x)
    -/
  · refine lowerSemicontinuous_tsum fun n => ?_
    exact
      ENNReal.continuous_coe.comp_lowerSemicontinuous (gcont n) fun x y hxy =>
        ENNReal.coe_le_coe.2 hxy
  · calc
      ∫⁻ x, ∑' n : ℕ, g n x ∂μ = ∑' n, ∫⁻ x, g n x ∂μ := by
        rw [lintegral_tsum fun n => (gcont n).measurable.coe_nnreal_ennreal.aemeasurable]
      _ ≤ ∑' n, ((∫⁻ x, SimpleFunc.eapproxDiff f n x ∂μ) + δ n) := ENNReal.tsum_le_tsum hg
      _ = ∑' n, ∫⁻ x, SimpleFunc.eapproxDiff f n x ∂μ + ∑' n, δ n := ENNReal.tsum_add
      _ ≤ (∫⁻ x : α, f x ∂μ) + ε := by
        refine add_le_add ?_ hδ.le
        rw [← lintegral_tsum]
        · simp_rw [SimpleFunc.tsum_eapproxDiff f hf, le_refl]
        · intro n; exact (SimpleFunc.measurable _).coe_nnreal_ennreal.aemeasurable


/-- Given a measurable function `f` with values in `ℝ≥0` in a sigma-finite space, there exists a
lower semicontinuous function `g > f` with integral arbitrarily close to that of `f`.
Formulation in terms of `lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_lt_lowerSemicontinuous_lintegral_ge [SigmaFinite μ] (f : α → ℝ≥0)
    (fmeas : Measurable f) {ε : ℝ≥0∞} (ε0 : ε ≠ 0) :
    ∃ g : α → ℝ≥0∞,
      (∀ x, (f x : ℝ≥0∞) < g x) ∧ LowerSemicontinuous g ∧ (∫⁻ x, g x ∂μ) ≤ (∫⁻ x, f x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : Measurable f
    ε : ENNReal
    ε0 : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have : ε / 2 ≠ 0 := (ENNReal.half_pos ε0).ne'
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : Measurable f
    ε : ENNReal
    ε0 : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  rcases exists_pos_lintegral_lt_of_sigmaFinite μ this with ⟨w, wpos, wmeas, wint⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : Measurable f
    ε : ENNReal
    ε0 : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    w : α → NNReal
    wpos : ∀ (x : α), LT.lt 0 (w x)
    wmeas : Measurable w
    wint : LT.lt (MeasureTheory.lintegral μ fun x => ↑(w x)) (HDiv.hDiv ε 2)
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  let f' x := ((f x + w x : ℝ≥0) : ℝ≥0∞)
  rcases exists_le_lowerSemicontinuous_lintegral_ge μ f' (fmeas.add wmeas).coe_nnreal_ennreal
      this with
    ⟨g, le_g, gcont, gint⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : Measurable f
    ε : ENNReal
    ε0 : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    w : α → NNReal
    wpos : ∀ (x : α), LT.lt 0 (w x)
    wmeas : Measurable w
    wint : LT.lt (MeasureTheory.lintegral μ fun x => ↑(w x)) (HDiv.hDiv ε 2)
    f' : α → ENNReal := fun x => ↑(HAdd.hAdd (f x) (w x))
    g : α → ENNReal
    le_g : ∀ (x : α), LE.le (f' x) (g x)
    gcont : LowerSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  refine ⟨g, fun x => ?_, gcont, ?_⟩
  · calc
      (f x : ℝ≥0∞) < f' x := by
        simpa only [← ENNReal.coe_lt_coe, add_zero] using add_lt_add_left (wpos x) (f x)
      _ ≤ g x := le_g x
  · calc
      (∫⁻ x : α, g x ∂μ) ≤ (∫⁻ x : α, f x + w x ∂μ) + ε / 2 := gint
      _ = ((∫⁻ x : α, f x ∂μ) + ∫⁻ x : α, w x ∂μ) + ε / 2 := by
        rw [lintegral_add_right _ wmeas.coe_nnreal_ennreal]
      _ ≤ (∫⁻ x : α, f x ∂μ) + ε / 2 + ε / 2 := add_le_add_right (add_le_add_left wint.le _) _
      _ = (∫⁻ x : α, f x ∂μ) + ε := by rw [add_assoc, ENNReal.add_halves]


/-- Given an almost everywhere measurable function `f` with values in `ℝ≥0` in a sigma-finite space,
there exists a lower semicontinuous function `g > f` with integral arbitrarily close to that of `f`.
Formulation in terms of `lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_lt_lowerSemicontinuous_lintegral_ge_of_aemeasurable [SigmaFinite μ] (f : α → ℝ≥0)
    (fmeas : AEMeasurable f μ) {ε : ℝ≥0∞} (ε0 : ε ≠ 0) :
    ∃ g : α → ℝ≥0∞,
      (∀ x, (f x : ℝ≥0∞) < g x) ∧ LowerSemicontinuous g ∧ (∫⁻ x, g x ∂μ) ≤ (∫⁻ x, f x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : AEMeasurable f μ
    ε : ENNReal
    ε0 : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have : ε / 2 ≠ 0 := (ENNReal.half_pos ε0).ne'
  rcases exists_lt_lowerSemicontinuous_lintegral_ge μ (fmeas.mk f) fmeas.measurable_mk this with
    ⟨g0, f_lt_g0, g0_cont, g0_int⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : AEMeasurable f μ
    ε : ENNReal
    ε0 : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    g0 : α → ENNReal
    f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
    g0_cont : LowerSemicontinuous g0
    g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  rcases exists_measurable_superset_of_null fmeas.ae_eq_mk with ⟨s, hs, smeas, μs⟩
  rcases exists_le_lowerSemicontinuous_lintegral_ge μ (s.indicator fun _x => ∞)
      (measurable_const.indicator smeas) this with
    ⟨g1, le_g1, g1_cont, g1_int⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fmeas : AEMeasurable f μ
    ε : ENNReal
    ε0 : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    g0 : α → ENNReal
    f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
    g0_cont : LowerSemicontinuous g0
    g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
    s : Set α
    hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
    smeas : MeasurableSet s
    μs : Eq (μ s) 0
    g1 : α → ENNReal
    le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
    g1_cont : LowerSemicontinuous g1
    g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  refine ⟨fun x => g0 x + g1 x, fun x => ?_, g0_cont.add g1_cont, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      fmeas : AEMeasurable f μ
      ε : ENNReal
      ε0 : Ne ε 0
      this : Ne (HDiv.hDiv ε 2) 0
      g0 : α → ENNReal
      f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
      g0_cont : LowerSemicontinuous g0
      g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
      s : Set α
      hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
      smeas : MeasurableSet s
      μs : Eq (μ s) 0
      g1 : α → ENNReal
      le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
      g1_cont : LowerSemicontinuous g1
      g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
      x : α
      ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
    -/
  · by_cases h : x ∈ s
      /-
        case pos
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Membership.mem s x
        ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
    · have := le_g1 x
      /-
        case pos
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this✝ : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Membership.mem s x
        this : LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
      simp only [h, Set.indicator_of_mem, top_le_iff] at this
      /-
        case pos
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this✝ : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Membership.mem s x
        this : Eq (g1 x) Top.top
        ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
      simp [this]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Not (Membership.mem s x)
        ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
    · have : f x = fmeas.mk f x := by rw [Set.compl_subset_comm] at hs; exact hs h
      /-
        case neg
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this✝ : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Not (Membership.mem s x)
        this : Eq (f x) (AEMeasurable.mk f fmeas x)
        ⊢ LT.lt (↑(f x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
      rw [this]
      /-
        case neg
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fmeas : AEMeasurable f μ
        ε : ENNReal
        ε0 : Ne ε 0
        this✝ : Ne (HDiv.hDiv ε 2) 0
        g0 : α → ENNReal
        f_lt_g0 : ∀ (x : α), LT.lt (↑(AEMeasurable.mk f fmeas x)) (g0 x)
        g0_cont : LowerSemicontinuous g0
        g0_int : LE.le (MeasureTheory.lintegral μ fun x => g0 x) (HAdd.hAdd (MeasureTh …
        s : Set α
        hs : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq (f x) (AEMe …
        smeas : MeasurableSet s
        μs : Eq (μ s) 0
        g1 : α → ENNReal
        le_g1 : ∀ (x : α), LE.le (s.indicator (fun _x => Top.top) x) (g1 x)
        g1_cont : LowerSemicontinuous g1
        g1_int : LE.le (MeasureTheory.lintegral μ fun x => g1 x) (HAdd.hAdd (MeasureTh …
        x : α
        h : Not (Membership.mem s x)
        this : Eq (f x) (AEMeasurable.mk f fmeas x)
        ⊢ LT.lt (↑(AEMeasurable.mk f fmeas x)) ((fun x => HAdd.hAdd (g0 x) (g1 x)) x)
      -/
      exact (f_lt_g0 x).trans_le le_self_add
      /-
        🎉 no goals
      -/
  · calc
      ∫⁻ x, g0 x + g1 x ∂μ = (∫⁻ x, g0 x ∂μ) + ∫⁻ x, g1 x ∂μ :=
        lintegral_add_left g0_cont.measurable _
      _ ≤ (∫⁻ x, f x ∂μ) + ε / 2 + (0 + ε / 2) := by
        refine add_le_add ?_ ?_
        · convert g0_int using 2
          exact lintegral_congr_ae (fmeas.ae_eq_mk.fun_comp _)
        · convert g1_int
          simp only [smeas, μs, lintegral_const, Set.univ_inter, MeasurableSet.univ,
            lintegral_indicator, mul_zero, restrict_apply]
      _ = (∫⁻ x, f x ∂μ) + ε := by simp only [add_assoc, ENNReal.add_halves, zero_add]


/-- Given an integrable function `f` with values in `ℝ≥0` in a sigma-finite space, there exists a
lower semicontinuous function `g > f` with integral arbitrarily close to that of `f`.
Formulation in terms of `integral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_lt_lowerSemicontinuous_integral_gt_nnreal [SigmaFinite μ] (f : α → ℝ≥0)
    (fint : Integrable (fun x => (f x : ℝ)) μ) {ε : ℝ} (εpos : 0 < ε) :
    ∃ g : α → ℝ≥0∞,
      (∀ x, (f x : ℝ≥0∞) < g x) ∧
      LowerSemicontinuous g ∧
      (∀ᵐ x ∂μ, g x < ⊤) ∧
      Integrable (fun x => (g x).toReal) μ ∧ (∫ x, (g x).toReal ∂μ) < (∫ x, ↑(f x) ∂μ) + ε := by
  have fmeas : AEMeasurable f μ := by
    convert fint.aestronglyMeasurable.real_toNNReal.aemeasurable
    simp only [Real.toNNReal_coe]
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : Real
    εpos : LT.lt 0 ε
    fmeas : AEMeasurable f μ
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  lift ε to ℝ≥0 using εpos.le
  /-
    case intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    fmeas : AEMeasurable f μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  obtain ⟨δ, δpos, hδε⟩ : ∃ δ : ℝ≥0, 0 < δ ∧ δ < ε := exists_between εpos
  have int_f_ne_top : (∫⁻ a : α, f a ∂μ) ≠ ∞ :=
    (hasFiniteIntegral_iff_ofNNReal.1 fint.hasFiniteIntegral).ne
  rcases exists_lt_lowerSemicontinuous_lintegral_ge_of_aemeasurable μ f fmeas
      (ENNReal.coe_ne_zero.2 δpos.ne') with
    ⟨g, f_lt_g, gcont, gint⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    fmeas : AEMeasurable f μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    δ : NNReal
    δpos : LT.lt 0 δ
    hδε : LT.lt δ ε
    int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
    g : α → ENNReal
    f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
    gcont : LowerSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have gint_ne : (∫⁻ x : α, g x ∂μ) ≠ ∞ := ne_top_of_le_ne_top (by simpa) gint
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    fmeas : AEMeasurable f μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    δ : NNReal
    δpos : LT.lt 0 δ
    hδε : LT.lt δ ε
    int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
    g : α → ENNReal
    f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
    gcont : LowerSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
    gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have g_lt_top : ∀ᵐ x : α ∂μ, g x < ∞ := ae_lt_top gcont.measurable gint_ne
  have Ig : (∫⁻ a : α, ENNReal.ofReal (g a).toReal ∂μ) = ∫⁻ a : α, g a ∂μ := by
    apply lintegral_congr_ae
    filter_upwards [g_lt_top] with _ hx
    simp only [hx.ne, ENNReal.ofReal_toReal, Ne, not_false_iff]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    fmeas : AEMeasurable f μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    δ : NNReal
    δpos : LT.lt 0 δ
    hδε : LT.lt δ ε
    int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
    g : α → ENNReal
    f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
    gcont : LowerSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
    gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
    Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  refine ⟨g, f_lt_g, gcont, g_lt_top, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      fmeas : AEMeasurable f μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      δ : NNReal
      δpos : LT.lt 0 δ
      hδε : LT.lt δ ε
      int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
      g : α → ENNReal
      f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
      gcont : LowerSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
      gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
      ⊢ MeasureTheory.Integrable (fun x => (g x).toReal) μ
    -/
  · refine ⟨gcont.measurable.ennreal_toReal.aemeasurable.aestronglyMeasurable, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      fmeas : AEMeasurable f μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      δ : NNReal
      δpos : LT.lt 0 δ
      hδε : LT.lt δ ε
      int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
      g : α → ENNReal
      f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
      gcont : LowerSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
      gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => (g x).toReal) μ
    -/
    simp only [hasFiniteIntegral_iff_norm, Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      fmeas : AEMeasurable f μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      δ : NNReal
      δpos : LT.lt 0 δ
      hδε : LT.lt δ ε
      int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
      g : α → ENNReal
      f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
      gcont : LowerSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
      gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) Top.top
    -/
    convert gint_ne.lt_top using 1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      fmeas : AEMeasurable f μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      δ : NNReal
      δpos : LT.lt 0 δ
      hδε : LT.lt δ ε
      int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
      g : α → ENNReal
      f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
      gcont : LowerSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
      gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
      ⊢ LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (MeasureTh …
    -/
  · rw [integral_eq_lintegral_of_nonneg_ae, integral_eq_lintegral_of_nonneg_ae]
    · calc
        ENNReal.toReal (∫⁻ a : α, ENNReal.ofReal (g a).toReal ∂μ) =
            ENNReal.toReal (∫⁻ a : α, g a ∂μ) := by congr 1
        _ ≤ ENNReal.toReal ((∫⁻ a : α, f a ∂μ) + δ) := by
          apply ENNReal.toReal_mono _ gint
          simpa using int_f_ne_top
        _ = ENNReal.toReal (∫⁻ a : α, f a ∂μ) + δ := by
          rw [ENNReal.toReal_add int_f_ne_top ENNReal.coe_ne_top, ENNReal.coe_toReal]
        _ < ENNReal.toReal (∫⁻ a : α, f a ∂μ) + ε := add_lt_add_left hδε _
        _ = (∫⁻ a : α, ENNReal.ofReal ↑(f a) ∂μ).toReal + ε := by simp

      /-
        case intro.intro.intro.intro.intro.intro.refine_2.hf
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        fmeas : AEMeasurable f μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        δ : NNReal
        δpos : LT.lt 0 δ
        hδε : LT.lt δ ε
        int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
        g : α → ENNReal
        f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
        gcont : LowerSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
        gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
        g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
        Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => ↑(f x)
      -/
    · apply Filter.Eventually.of_forall fun x => _; simp
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.hfm
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        fmeas : AEMeasurable f μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        δ : NNReal
        δpos : LT.lt 0 δ
        hδε : LT.lt δ ε
        int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
        g : α → ENNReal
        f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
        gcont : LowerSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
        gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
        g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
        Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x => ↑(f x)) μ
      -/
    · exact fmeas.coe_nnreal_real.aestronglyMeasurable
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.hf
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        fmeas : AEMeasurable f μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        δ : NNReal
        δpos : LT.lt 0 δ
        hδε : LT.lt δ ε
        int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
        g : α → ENNReal
        f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
        gcont : LowerSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
        gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
        g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
        Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => (g x).toReal
      -/
    · apply Filter.Eventually.of_forall fun x => _; simp
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.hfm
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝¹ : μ.WeaklyRegular
        inst✝ : MeasureTheory.SigmaFinite μ
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        fmeas : AEMeasurable f μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        δ : NNReal
        δpos : LT.lt 0 δ
        hδε : LT.lt δ ε
        int_f_ne_top : Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
        g : α → ENNReal
        f_lt_g : ∀ (x : α), LT.lt (↑(f x)) (g x)
        gcont : LowerSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheor …
        gint_ne : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
        g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
        Ig : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (g a).toReal) (Meas …
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (g x).toReal) μ
      -/
    · apply gcont.measurable.ennreal_toReal.aemeasurable.aestronglyMeasurable
      /-
        🎉 no goals
      -/


/-- Given a simple function `f` with values in `ℝ≥0`, there exists an upper semicontinuous
function `g ≤ f` with integral arbitrarily close to that of `f`. Formulation in terms of
`lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem SimpleFunc.exists_upperSemicontinuous_le_lintegral_le (f : α →ₛ ℝ≥0)
    (int_f : (∫⁻ x, f x ∂μ) ≠ ∞) {ε : ℝ≥0∞} (ε0 : ε ≠ 0) :
    ∃ g : α → ℝ≥0, (∀ x, g x ≤ f x) ∧ UpperSemicontinuous g ∧
      (∫⁻ x, f x ∂μ) ≤ (∫⁻ x, g x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : MeasureTheory.SimpleFunc α NNReal
    int_f : Ne (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
    ε : ENNReal
    ε0 : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  induction' f using MeasureTheory.SimpleFunc.induction with c s hs f₁ f₂ _ h₁ h₂ generalizing ε
    /-
      case h_ind
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((MeasureTheory.SimpleFunc.pie …
      ε : ENNReal
      ε0 : Ne ε 0
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((MeasureTheory.SimpleFunc.piece …
    -/
  · by_cases hc : c = 0
      /-
        case pos
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((MeasureTheory.SimpleFunc.pie …
        ε : ENNReal
        ε0 : Ne ε 0
        hc : Eq c 0
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((MeasureTheory.SimpleFunc.piece …
      -/
    · refine ⟨fun _ => 0, ?_, upperSemicontinuous_const, ?_⟩
      · classical
        simp only [hc, Set.indicator_zero', Pi.zero_apply, SimpleFunc.const_zero, imp_true_iff,
          eq_self_iff_true, SimpleFunc.coe_zero, Set.piecewise_eq_indicator,
          SimpleFunc.coe_piecewise, le_zero_iff]
      · classical
        simp only [hc, Set.indicator_zero', lintegral_const, zero_mul, Pi.zero_apply,
          SimpleFunc.const_zero, zero_add, zero_le', SimpleFunc.coe_zero,
          Set.piecewise_eq_indicator, ENNReal.coe_zero, SimpleFunc.coe_piecewise, zero_le]
    have μs_lt_top : μ s < ∞ := by
      classical
      simpa only [hs, hc, lt_top_iff_ne_top, true_and, SimpleFunc.coe_const, or_false,
        lintegral_const, ENNReal.coe_indicator, Set.univ_inter, ENNReal.coe_ne_top,
        Measure.restrict_apply MeasurableSet.univ, ENNReal.mul_eq_top, SimpleFunc.const_zero,
        Function.const_apply, lintegral_indicator, ENNReal.coe_eq_zero, Ne, not_false_iff,
        SimpleFunc.coe_zero, Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise,
        false_and] using int_f
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((MeasureTheory.SimpleFunc.pie …
      ε : ENNReal
      ε0 : Ne ε 0
      hc : Not (Eq c 0)
      μs_lt_top : LT.lt (μ s) Top.top
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((MeasureTheory.SimpleFunc.piece …
    -/
    have : (0 : ℝ≥0∞) < ε / c := ENNReal.div_pos_iff.2 ⟨ε0, ENNReal.coe_ne_top⟩
    obtain ⟨F, Fs, F_closed, μF⟩ : ∃ (F : _), F ⊆ s ∧ IsClosed F ∧ μ s < μ F + ε / c :=
      hs.exists_isClosed_lt_add μs_lt_top.ne this.ne'
    refine
      ⟨Set.indicator F fun _ => c, fun x => ?_, F_closed.upperSemicontinuous_indicator (zero_le _),
        ?_⟩
    · simp only [SimpleFunc.coe_const, SimpleFunc.const_zero, SimpleFunc.coe_zero,
        Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise]
      /-
        case neg.intro.intro.intro.refine_1
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((MeasureTheory.SimpleFunc.pie …
        ε : ENNReal
        ε0 : Ne ε 0
        hc : Not (Eq c 0)
        μs_lt_top : LT.lt (μ s) Top.top
        this : LT.lt 0 (HDiv.hDiv ε ↑c)
        F : Set α
        Fs : HasSubset.Subset F s
        F_closed : IsClosed F
        μF : LT.lt (μ s) (HAdd.hAdd (μ F) (HDiv.hDiv ε ↑c))
        x : α
        ⊢ LE.le (F.indicator (fun x => c) x) (s.indicator (Function.const α c) x)
      -/
      exact Set.indicator_le_indicator_of_subset Fs (fun x => zero_le _) _
      /-
        🎉 no goals
      -/
    · suffices (c : ℝ≥0∞) * μ s ≤ c * μ F + ε by
        classical
        simpa only [hs, F_closed.measurableSet, SimpleFunc.coe_const, Function.const_apply,
          lintegral_const, ENNReal.coe_indicator, Set.univ_inter, MeasurableSet.univ,
          SimpleFunc.const_zero, lintegral_indicator, SimpleFunc.coe_zero,
          Set.piecewise_eq_indicator, SimpleFunc.coe_piecewise, Measure.restrict_apply]
      calc
        (c : ℝ≥0∞) * μ s ≤ c * (μ F + ε / c) := mul_le_mul_left' μF.le _
        _ = c * μ F + ε := by
          simp_rw [mul_add]
          rw [ENNReal.mul_div_cancel _ ENNReal.coe_ne_top]
          simpa using hc
  · have A : ((∫⁻ x : α, f₁ x ∂μ) + ∫⁻ x : α, f₂ x ∂μ) ≠ ⊤ := by
      rwa [← lintegral_add_left f₁.measurable.coe_nnreal_ennreal]
    rcases h₁ (ENNReal.add_ne_top.1 A).1 (ENNReal.half_pos ε0).ne' with
      ⟨g₁, f₁_le_g₁, g₁cont, g₁int⟩
    rcases h₂ (ENNReal.add_ne_top.1 A).2 (ENNReal.half_pos ε0).ne' with
      ⟨g₂, f₂_le_g₂, g₂cont, g₂int⟩
    refine
      ⟨fun x => g₁ x + g₂ x, fun x => add_le_add (f₁_le_g₁ x) (f₂_le_g₂ x), g₁cont.add g₂cont, ?_⟩
    /-
      case h_add.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) Top.top → ∀ {ε : ENNReal} …
      h₂ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) Top.top → ∀ {ε : ENNReal} …
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd f₁ f₂) x)) Top.top
      ε : ENNReal
      ε0 : Ne ε 0
      A : Ne (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureTheory. …
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁cont : UpperSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (g₂ x) (f₂ x)
      g₂cont : UpperSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) (HAdd.hAdd (Measure …
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd f₁ f₂) x)) (HAdd.hAdd …
    -/
    simp only [SimpleFunc.coe_add, ENNReal.coe_add, Pi.add_apply]
    rw [lintegral_add_left f₁.measurable.coe_nnreal_ennreal,
      lintegral_add_left g₁cont.measurable.coe_nnreal_ennreal]
    /-
      case h_add.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) Top.top → ∀ {ε : ENNReal} …
      h₂ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) Top.top → ∀ {ε : ENNReal} …
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd f₁ f₂) x)) Top.top
      ε : ENNReal
      ε0 : Ne ε 0
      A : Ne (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureTheory. …
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁cont : UpperSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (g₂ x) (f₂ x)
      g₂cont : UpperSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) (HAdd.hAdd (Measure …
      ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(f₁ a)) (MeasureTheory …
    -/
    convert add_le_add g₁int g₂int using 1
    /-
      case h.e'_4
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) Top.top → ∀ {ε : ENNReal} …
      h₂ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) Top.top → ∀ {ε : ENNReal} …
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd f₁ f₂) x)) Top.top
      ε : ENNReal
      ε0 : Ne ε 0
      A : Ne (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureTheory. …
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁cont : UpperSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (g₂ x) (f₂ x)
      g₂cont : UpperSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) (HAdd.hAdd (Measure …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(g₁ a)) (Measu …
    -/
    conv_lhs => rw [← ENNReal.add_halves ε]
    /-
      case h.e'_4
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) Top.top → ∀ {ε : ENNReal} …
      h₂ : Ne (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) Top.top → ∀ {ε : ENNReal} …
      int_f : Ne (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd f₁ f₂) x)) Top.top
      ε : ENNReal
      ε0 : Ne ε 0
      A : Ne (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureTheory. …
      g₁ : α → NNReal
      f₁_le_g₁ : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁cont : UpperSemicontinuous g₁
      g₁int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (HAdd.hAdd (Measure …
      g₂ : α → NNReal
      f₂_le_g₂ : ∀ (x : α), LE.le (g₂ x) (f₂ x)
      g₂cont : UpperSemicontinuous g₂
      g₂int : LE.le (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) (HAdd.hAdd (Measure …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ↑(g₁ a)) (Measu …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- Given an integrable function `f` with values in `ℝ≥0`, there exists an upper semicontinuous
function `g ≤ f` with integral arbitrarily close to that of `f`. Formulation in terms of
`lintegral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_upperSemicontinuous_le_lintegral_le (f : α → ℝ≥0) (int_f : (∫⁻ x, f x ∂μ) ≠ ∞)
    {ε : ℝ≥0∞} (ε0 : ε ≠ 0) :
    ∃ g : α → ℝ≥0, (∀ x, g x ≤ f x) ∧ UpperSemicontinuous g ∧
      (∫⁻ x, f x ∂μ) ≤ (∫⁻ x, g x ∂μ) + ε := by
  obtain ⟨fs, fs_le_f, int_fs⟩ :
    ∃ fs : α →ₛ ℝ≥0, (∀ x, fs x ≤ f x) ∧ (∫⁻ x, f x ∂μ) ≤ (∫⁻ x, fs x ∂μ) + ε / 2 := by
    -- Porting note: need to name identifier (not `this`), because `conv_rhs at this` errors
    have aux := ENNReal.lt_add_right int_f (ENNReal.half_pos ε0).ne'
    conv_rhs at aux => rw [lintegral_eq_nnreal (fun x => (f x : ℝ≥0∞)) μ]
    erw [ENNReal.biSup_add] at aux <;> [skip; exact ⟨0, fun x => by simp⟩]
    simp only [lt_iSup_iff] at aux
    rcases aux with ⟨fs, fs_le_f, int_fs⟩
    refine ⟨fs, fun x => by simpa only [ENNReal.coe_le_coe] using fs_le_f x, ?_⟩
    convert int_fs.le
    rw [← SimpleFunc.lintegral_eq_lintegral]
    simp only [SimpleFunc.coe_map, Function.comp_apply]
  have int_fs_lt_top : (∫⁻ x, fs x ∂μ) ≠ ∞ := by
    refine ne_top_of_le_ne_top int_f (lintegral_mono fun x => ?_)
    simpa only [ENNReal.coe_le_coe] using fs_le_f x
  obtain ⟨g, g_le_fs, gcont, gint⟩ :
    ∃ g : α → ℝ≥0,
      (∀ x, g x ≤ fs x) ∧ UpperSemicontinuous g ∧ (∫⁻ x, fs x ∂μ) ≤ (∫⁻ x, g x ∂μ) + ε / 2 :=
    fs.exists_upperSemicontinuous_le_lintegral_le int_fs_lt_top (ENNReal.half_pos ε0).ne'
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    int_f : Ne (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
    ε : ENNReal
    ε0 : Ne ε 0
    fs : MeasureTheory.SimpleFunc α NNReal
    fs_le_f : ∀ (x : α), LE.le (fs x) (f x)
    int_fs : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (Measure …
    int_fs_lt_top : Ne (MeasureTheory.lintegral μ fun x => ↑(fs x)) Top.top
    g : α → NNReal
    g_le_fs : ∀ (x : α), LE.le (g x) (fs x)
    gcont : UpperSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(fs x)) (HAdd.hAdd (MeasureT …
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  refine ⟨g, fun x => (g_le_fs x).trans (fs_le_f x), gcont, ?_⟩
  calc
    (∫⁻ x, f x ∂μ) ≤ (∫⁻ x, fs x ∂μ) + ε / 2 := int_fs
    _ ≤ (∫⁻ x, g x ∂μ) + ε / 2 + ε / 2 := add_le_add gint le_rfl
    _ = (∫⁻ x, g x ∂μ) + ε := by rw [add_assoc, ENNReal.add_halves]


/-- Given an integrable function `f` with values in `ℝ≥0`, there exists an upper semicontinuous
function `g ≤ f` with integral arbitrarily close to that of `f`. Formulation in terms of
`integral`.
Auxiliary lemma for Vitali-Carathéodory theorem `exists_lt_lower_semicontinuous_integral_lt`. -/
theorem exists_upperSemicontinuous_le_integral_le (f : α → ℝ≥0)
    (fint : Integrable (fun x => (f x : ℝ)) μ) {ε : ℝ} (εpos : 0 < ε) :
    ∃ g : α → ℝ≥0,
      (∀ x, g x ≤ f x) ∧
      UpperSemicontinuous g ∧
      Integrable (fun x => (g x : ℝ)) μ ∧ (∫ x, (f x : ℝ) ∂μ) - ε ≤ ∫ x, ↑(g x) ∂μ := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  lift ε to ℝ≥0 using εpos.le
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  rw [NNReal.coe_pos, ← ENNReal.coe_pos] at εpos
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  have If : (∫⁻ x, f x ∂μ) < ∞ := hasFiniteIntegral_iff_ofNNReal.1 fint.hasFiniteIntegral
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  rcases exists_upperSemicontinuous_le_lintegral_le f If.ne εpos.ne' with ⟨g, gf, gcont, gint⟩
  have Ig : (∫⁻ x, g x ∂μ) < ∞ := by
    refine lt_of_le_of_lt (lintegral_mono fun x => ?_) If
    simpa using gf x
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.WeaklyRegular
    f : α → NNReal
    fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ε : NNReal
    εpos : LT.lt 0 ↑ε
    If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
    g : α → NNReal
    gf : ∀ (x : α), LE.le (g x) (f x)
    gcont : UpperSemicontinuous g
    gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
    Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (UpperSemicontinuous …
  -/
  refine ⟨g, gf, gcont, ?_, ?_⟩
  · refine
      Integrable.mono fint gcont.measurable.coe_nnreal_real.aemeasurable.aestronglyMeasurable ?_
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
      g : α → NNReal
      gf : ∀ (x : α), LE.le (g x) (f x)
      gcont : UpperSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
      Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm ↑(g a)) (Norm.norm ↑(f a))) (Me …
    -/
    exact Filter.Eventually.of_forall fun x => by simp [gf x]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : MeasurableSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.WeaklyRegular
      f : α → NNReal
      fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
      ε : NNReal
      εpos : LT.lt 0 ↑ε
      If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
      g : α → NNReal
      gf : ∀ (x : α), LE.le (g x) (f x)
      gcont : UpperSemicontinuous g
      gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
      Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
      ⊢ LE.le (HSub.hSub (MeasureTheory.integral μ fun x => ↑(f x)) ↑ε) (MeasureTheo …
    -/
  · rw [integral_eq_lintegral_of_nonneg_ae, integral_eq_lintegral_of_nonneg_ae]
      /-
        case intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ LE.le (HSub.hSub (MeasureTheory.lintegral μ fun a => ENNReal.ofReal ↑(f a)). …
      -/
    · rw [sub_le_iff_le_add]
      /-
        case intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal ↑(f a)).toReal (HAd …
      -/
      convert ENNReal.toReal_mono _ gint
        /-
          case h.e'_3.h.e'_1.h.e'_4.h
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : MeasurableSpace α
          inst✝¹ : BorelSpace α
          μ : MeasureTheory.Measure α
          inst✝ : μ.WeaklyRegular
          f : α → NNReal
          fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
          ε : NNReal
          εpos : LT.lt 0 ↑ε
          If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
          g : α → NNReal
          gf : ∀ (x : α), LE.le (g x) (f x)
          gcont : UpperSemicontinuous g
          gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
          Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
          x✝ : α
          ⊢ Eq (ENNReal.ofReal ↑(f x✝)) ↑(f x✝)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case h.e'_4
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : MeasurableSpace α
          inst✝¹ : BorelSpace α
          μ : MeasureTheory.Measure α
          inst✝ : μ.WeaklyRegular
          f : α → NNReal
          fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
          ε : NNReal
          εpos : LT.lt 0 ↑ε
          If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
          g : α → NNReal
          gf : ∀ (x : α), LE.le (g x) (f x)
          gcont : UpperSemicontinuous g
          gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
          Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
          ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ENNReal.ofReal ↑(g a)).toR …
        -/
      · rw [ENNReal.toReal_add Ig.ne ENNReal.coe_ne_top]; simp
                                                          /-
                                                            🎉 no goals
                                                          -/
        /-
          case intro.intro.intro.intro.refine_2
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : MeasurableSpace α
          inst✝¹ : BorelSpace α
          μ : MeasureTheory.Measure α
          inst✝ : μ.WeaklyRegular
          f : α → NNReal
          fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
          ε : NNReal
          εpos : LT.lt 0 ↑ε
          If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
          g : α → NNReal
          gf : ∀ (x : α), LE.le (g x) (f x)
          gcont : UpperSemicontinuous g
          gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
          Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
          ⊢ Ne (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(g x)) ↑ε) Top.top
        -/
      · simpa using Ig.ne
        /-
          🎉 no goals
        -/
      /-
        case intro.intro.intro.intro.refine_2.hf
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => ↑(g x)
      -/
    · apply Filter.Eventually.of_forall; simp
                                         /-
                                           🎉 no goals
                                         -/
      /-
        case intro.intro.intro.intro.refine_2.hfm
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x => ↑(g x)) μ
      -/
    · exact gcont.measurable.coe_nnreal_real.aemeasurable.aestronglyMeasurable
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2.hf
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => ↑(f x)
      -/
    · apply Filter.Eventually.of_forall; simp
                                         /-
                                           🎉 no goals
                                         -/
      /-
        case intro.intro.intro.intro.refine_2.hfm
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : MeasurableSpace α
        inst✝¹ : BorelSpace α
        μ : MeasureTheory.Measure α
        inst✝ : μ.WeaklyRegular
        f : α → NNReal
        fint : MeasureTheory.Integrable (fun x => ↑(f x)) μ
        ε : NNReal
        εpos : LT.lt 0 ↑ε
        If : LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
        g : α → NNReal
        gf : ∀ (x : α), LE.le (g x) (f x)
        gcont : UpperSemicontinuous g
        gint : LE.le (MeasureTheory.lintegral μ fun x => ↑(f x)) (HAdd.hAdd (MeasureTh …
        Ig : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        ⊢ MeasureTheory.AEStronglyMeasurable (fun x => ↑(f x)) μ
      -/
    · exact fint.aestronglyMeasurable
      /-
        🎉 no goals
      -/


/-- **Vitali-Carathéodory Theorem**: given an integrable real function `f`, there exists an
integrable function `g > f` which is lower semicontinuous, with integral arbitrarily close
to that of `f`. This function has to be `EReal`-valued in general. -/
theorem exists_lt_lowerSemicontinuous_integral_lt [SigmaFinite μ] (f : α → ℝ) (hf : Integrable f μ)
    {ε : ℝ} (εpos : 0 < ε) :
    ∃ g : α → EReal,
      (∀ x, (f x : EReal) < g x) ∧
      LowerSemicontinuous g ∧
      Integrable (fun x => EReal.toReal (g x)) μ ∧
      (∀ᵐ x ∂μ, g x < ⊤) ∧ (∫ x, EReal.toReal (g x) ∂μ) < (∫ x, f x ∂μ) + ε := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  let δ : ℝ≥0 := ⟨ε / 2, (half_pos εpos).le⟩
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have δpos : 0 < δ := half_pos εpos
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  let fp : α → ℝ≥0 := fun x => Real.toNNReal (f x)
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    fp : α → NNReal := fun x => (f x).toNNReal
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have int_fp : Integrable (fun x => (fp x : ℝ)) μ := hf.real_toNNReal
  rcases exists_lt_lowerSemicontinuous_integral_gt_nnreal fp int_fp δpos with
    ⟨gp, fp_lt_gp, gpcont, gp_lt_top, gp_integrable, gpint⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    fp : α → NNReal := fun x => (f x).toNNReal
    int_fp : MeasureTheory.Integrable (fun x => ↑(fp x)) μ
    gp : α → ENNReal
    fp_lt_gp : ∀ (x : α), LT.lt (↑(fp x)) (gp x)
    gpcont : LowerSemicontinuous gp
    gp_lt_top : Filter.Eventually (fun x => LT.lt (gp x) Top.top) (MeasureTheory.a …
    gp_integrable : MeasureTheory.Integrable (fun x => (gp x).toReal) μ
    gpint : LT.lt (MeasureTheory.integral μ fun x => (gp x).toReal) (HAdd.hAdd (Me …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  let fm : α → ℝ≥0 := fun x => Real.toNNReal (-f x)
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    fp : α → NNReal := fun x => (f x).toNNReal
    int_fp : MeasureTheory.Integrable (fun x => ↑(fp x)) μ
    gp : α → ENNReal
    fp_lt_gp : ∀ (x : α), LT.lt (↑(fp x)) (gp x)
    gpcont : LowerSemicontinuous gp
    gp_lt_top : Filter.Eventually (fun x => LT.lt (gp x) Top.top) (MeasureTheory.a …
    gp_integrable : MeasureTheory.Integrable (fun x => (gp x).toReal) μ
    gpint : LT.lt (MeasureTheory.integral μ fun x => (gp x).toReal) (HAdd.hAdd (Me …
    fm : α → NNReal := fun x => (Neg.neg (f x)).toNNReal
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  have int_fm : Integrable (fun x => (fm x : ℝ)) μ := hf.neg.real_toNNReal
  rcases exists_upperSemicontinuous_le_integral_le fm int_fm δpos with
    ⟨gm, gm_le_fm, gmcont, gm_integrable, gmint⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    fp : α → NNReal := fun x => (f x).toNNReal
    int_fp : MeasureTheory.Integrable (fun x => ↑(fp x)) μ
    gp : α → ENNReal
    fp_lt_gp : ∀ (x : α), LT.lt (↑(fp x)) (gp x)
    gpcont : LowerSemicontinuous gp
    gp_lt_top : Filter.Eventually (fun x => LT.lt (gp x) Top.top) (MeasureTheory.a …
    gp_integrable : MeasureTheory.Integrable (fun x => (gp x).toReal) μ
    gpint : LT.lt (MeasureTheory.integral μ fun x => (gp x).toReal) (HAdd.hAdd (Me …
    fm : α → NNReal := fun x => (Neg.neg (f x)).toNNReal
    int_fm : MeasureTheory.Integrable (fun x => ↑(fm x)) μ
    gm : α → NNReal
    gm_le_fm : ∀ (x : α), LE.le (gm x) (fm x)
    gmcont : UpperSemicontinuous gm
    gm_integrable : MeasureTheory.Integrable (fun x => ↑(gm x)) μ
    gmint : LE.le (HSub.hSub (MeasureTheory.integral μ fun x => ↑(fm x)) ((fun a = …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  let g : α → EReal := fun x => (gp x : EReal) - gm x
  have ae_g : ∀ᵐ x ∂μ, (g x).toReal = (gp x : EReal).toReal - (gm x : EReal).toReal := by
    filter_upwards [gp_lt_top] with _ hx
    rw [EReal.toReal_sub] <;> simp [hx.ne]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    δ : NNReal := ⟨HDiv.hDiv ε 2, ⋯⟩
    δpos : LT.lt 0 δ
    fp : α → NNReal := fun x => (f x).toNNReal
    int_fp : MeasureTheory.Integrable (fun x => ↑(fp x)) μ
    gp : α → ENNReal
    fp_lt_gp : ∀ (x : α), LT.lt (↑(fp x)) (gp x)
    gpcont : LowerSemicontinuous gp
    gp_lt_top : Filter.Eventually (fun x => LT.lt (gp x) Top.top) (MeasureTheory.a …
    gp_integrable : MeasureTheory.Integrable (fun x => (gp x).toReal) μ
    gpint : LT.lt (MeasureTheory.integral μ fun x => (gp x).toReal) (HAdd.hAdd (Me …
    fm : α → NNReal := fun x => (Neg.neg (f x)).toNNReal
    int_fm : MeasureTheory.Integrable (fun x => ↑(fm x)) μ
    gm : α → NNReal
    gm_le_fm : ∀ (x : α), LE.le (gm x) (fm x)
    gmcont : UpperSemicontinuous gm
    gm_integrable : MeasureTheory.Integrable (fun x => ↑(gm x)) μ
    gmint : LE.le (HSub.hSub (MeasureTheory.integral μ fun x => ↑(fm x)) ((fun a = …
    g : α → EReal := fun x => HSub.hSub ↑(gp x) ↑↑(gm x)
    ae_g : Filter.Eventually (fun x => Eq (g x).toReal (HSub.hSub (↑(gp x)).toReal …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (↑(f x)) (g x)) (And (LowerSemicontinu …
  -/
  refine ⟨g, ?lt, ?lsc, ?int, ?aelt, ?intlt⟩
  case int =>
    show Integrable (fun x => EReal.toReal (g x)) μ
    rw [integrable_congr ae_g]
    convert gp_integrable.sub gm_integrable
    simp
  case intlt =>
    show (∫ x : α, (g x).toReal ∂μ) < (∫ x : α, f x ∂μ) + ε
    exact
      calc
        (∫ x : α, (g x).toReal ∂μ) = ∫ x : α, EReal.toReal (gp x) - EReal.toReal (gm x) ∂μ :=
          integral_congr_ae ae_g
        _ = (∫ x : α, EReal.toReal (gp x) ∂μ) - ∫ x : α, ↑(gm x) ∂μ := by
          simp only [EReal.toReal_coe_ennreal, ENNReal.coe_toReal]
          exact integral_sub gp_integrable gm_integrable
        _ < (∫ x : α, ↑(fp x) ∂μ) + ↑δ - ∫ x : α, ↑(gm x) ∂μ := by
          apply sub_lt_sub_right
          convert gpint
          simp only [EReal.toReal_coe_ennreal]
        _ ≤ (∫ x : α, ↑(fp x) ∂μ) + ↑δ - ((∫ x : α, ↑(fm x) ∂μ) - δ) := sub_le_sub_left gmint _
        _ = (∫ x : α, f x ∂μ) + 2 * δ := by
          simp_rw [integral_eq_integral_pos_part_sub_integral_neg_part hf]; ring
        _ = (∫ x : α, f x ∂μ) + ε := by congr 1; field_simp [δ, mul_comm]
  case aelt =>
    show ∀ᵐ x : α ∂μ, g x < ⊤
    filter_upwards [gp_lt_top] with ?_ hx
    simp only [g, sub_eq_add_neg, Ne, (EReal.add_lt_top _ _).ne, lt_top_iff_ne_top,
      lt_top_iff_ne_top.1 hx, EReal.coe_ennreal_eq_top_iff, not_false_iff, EReal.neg_eq_top_iff,
      EReal.coe_ennreal_ne_bot]
  case lt =>
    show ∀ x, (f x : EReal) < g x
    intro x
    rw [EReal.coe_real_ereal_eq_coe_toNNReal_sub_coe_toNNReal (f x)]
    refine EReal.sub_lt_sub_of_lt_of_le ?_ ?_ ?_ ?_
    · simp only [EReal.coe_ennreal_lt_coe_ennreal_iff]; exact fp_lt_gp x
    · simp only [ENNReal.coe_le_coe, EReal.coe_ennreal_le_coe_ennreal_iff]
      exact gm_le_fm x
    · simp only [EReal.coe_ennreal_ne_bot, Ne, not_false_iff]
    · simp only [EReal.coe_nnreal_ne_top, Ne, not_false_iff]
  case lsc =>
    show LowerSemicontinuous g
    apply LowerSemicontinuous.add'
    · exact continuous_coe_ennreal_ereal.comp_lowerSemicontinuous gpcont fun x y hxy =>
          EReal.coe_ennreal_le_coe_ennreal_iff.2 hxy
    · apply continuous_neg.comp_upperSemicontinuous_antitone _ fun x y hxy =>
          EReal.neg_le_neg_iff.2 hxy
      dsimp
      apply continuous_coe_ennreal_ereal.comp_upperSemicontinuous _ fun x y hxy =>
          EReal.coe_ennreal_le_coe_ennreal_iff.2 hxy
      exact ENNReal.continuous_coe.comp_upperSemicontinuous gmcont fun x y hxy =>
          ENNReal.coe_le_coe.2 hxy
    · intro x
      exact EReal.continuousAt_add (by simp) (by simp)


/-- **Vitali-Carathéodory Theorem**: given an integrable real function `f`, there exists an
integrable function `g < f` which is upper semicontinuous, with integral arbitrarily close to that
of `f`. This function has to be `EReal`-valued in general. -/
theorem exists_upperSemicontinuous_lt_integral_gt [SigmaFinite μ] (f : α → ℝ) (hf : Integrable f μ)
    {ε : ℝ} (εpos : 0 < ε) :
    ∃ g : α → EReal,
      (∀ x, (g x : EReal) < f x) ∧
      UpperSemicontinuous g ∧
      Integrable (fun x => EReal.toReal (g x)) μ ∧
      (∀ᵐ x ∂μ, ⊥ < g x) ∧ (∫ x, f x ∂μ) < (∫ x, EReal.toReal (g x) ∂μ) + ε := by
  rcases exists_lt_lowerSemicontinuous_integral_lt (fun x => -f x) hf.neg εpos with
    ⟨g, g_lt_f, gcont, g_integrable, g_lt_top, gint⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : μ.WeaklyRegular
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    εpos : LT.lt 0 ε
    g : α → EReal
    g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
    gcont : LowerSemicontinuous g
    g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
    g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
    gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
    ⊢ Exists fun g => And (∀ (x : α), LT.lt (g x) ↑(f x)) (And (UpperSemicontinuou …
  -/
  refine ⟨fun x => -g x, ?_, ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
      ⊢ ∀ (x : α), LT.lt ((fun x => Neg.neg (g x)) x) ↑(f x)
    -/
  · exact fun x => EReal.neg_lt_comm.1 (by simpa only [EReal.coe_neg] using g_lt_f x)
    /-
      🎉 no goals
    -/
  · exact
      continuous_neg.comp_lowerSemicontinuous_antitone gcont fun x y hxy =>
        EReal.neg_le_neg_iff.2 hxy
    /-
      case intro.intro.intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
      ⊢ MeasureTheory.Integrable (fun x => ((fun x => Neg.neg (g x)) x).toReal) μ
    -/
  · convert g_integrable.neg
    /-
      case h.e'_6.h
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
      x✝ : α
      ⊢ Eq ((fun x => Neg.neg (g x)) x✝).toReal (Neg.neg (fun x => (g x).toReal) x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_4
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
      ⊢ Filter.Eventually (fun x => LT.lt Bot.bot ((fun x => Neg.neg (g x)) x)) (Mea …
    -/
  · simpa [bot_lt_iff_ne_bot, lt_top_iff_ne_top] using g_lt_top
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_5
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (MeasureTheory.integral μ fun x => (g x).toReal) (HAdd.hAdd (Meas …
      ⊢ LT.lt (MeasureTheory.integral μ fun x => f x) (HAdd.hAdd (MeasureTheory.inte …
    -/
  · simp_rw [integral_neg, lt_neg_add_iff_add_lt] at gint
    /-
      case intro.intro.intro.intro.intro.refine_5
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (HAdd.hAdd (MeasureTheory.integral μ fun a => f a) (MeasureTheory …
      ⊢ LT.lt (MeasureTheory.integral μ fun x => f x) (HAdd.hAdd (MeasureTheory.inte …
    -/
    rw [add_comm] at gint
    /-
      case intro.intro.intro.intro.intro.refine_5
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : μ.WeaklyRegular
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ε : Real
      εpos : LT.lt 0 ε
      g : α → EReal
      g_lt_f : ∀ (x : α), LT.lt (↑(Neg.neg (f x))) (g x)
      gcont : LowerSemicontinuous g
      g_integrable : MeasureTheory.Integrable (fun x => (g x).toReal) μ
      g_lt_top : Filter.Eventually (fun x => LT.lt (g x) Top.top) (MeasureTheory.ae μ)
      gint : LT.lt (HAdd.hAdd (MeasureTheory.integral μ fun x => (g x).toReal) (Meas …
      ⊢ LT.lt (MeasureTheory.integral μ fun x => f x) (HAdd.hAdd (MeasureTheory.inte …
    -/
    simpa [integral_neg] using gint
    /-
      🎉 no goals
    -/


