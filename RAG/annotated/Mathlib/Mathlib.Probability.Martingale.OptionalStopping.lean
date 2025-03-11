/-- Given a submartingale `f` and bounded stopping times `τ` and `π` such that `τ ≤ π`, the
expectation of `stoppedValue f τ` is less than or equal to the expectation of `stoppedValue f π`.
This is the forward direction of the optional stopping theorem. -/
theorem Submartingale.expected_stoppedValue_mono [SigmaFiniteFiltration μ 𝒢]
    (hf : Submartingale f 𝒢 μ) (hτ : IsStoppingTime 𝒢 τ) (hπ : IsStoppingTime 𝒢 π) (hle : τ ≤ π)
    {N : ℕ} (hbdd : ∀ ω, π ω ≤ N) : μ[stoppedValue f τ] ≤ μ[stoppedValue f π] := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    τ π : Ω → Nat
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
    hf : MeasureTheory.Submartingale f 𝒢 μ
    hτ : MeasureTheory.IsStoppingTime 𝒢 τ
    hπ : MeasureTheory.IsStoppingTime 𝒢 π
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue f τ x) ( …
  -/
  rw [← sub_nonneg, ← integral_sub', stoppedValue_sub_eq_sum' hle hbdd]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ π : Ω → Nat
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      hf : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hle : LE.le τ π
      N : Nat
      hbdd : ∀ (ω : Ω), LE.le (π ω) N
      ⊢ LE.le 0 (MeasureTheory.integral μ fun a => (fun ω => (Finset.range (HAdd.hAd …
    -/
  · simp only [Finset.sum_apply]
    have : ∀ i, MeasurableSet[𝒢 i] {ω : Ω | τ ω ≤ i ∧ i < π ω} := by
      intro i
      refine (hτ i).inter ?_
      convert (hπ i).compl using 1
      ext x
      simp; rfl
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ π : Ω → Nat
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      hf : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hle : LE.le τ π
      N : Nat
      hbdd : ∀ (ω : Ω), LE.le (π ω) N
      this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
      ⊢ LE.le 0 (MeasureTheory.integral μ fun a => (Finset.range (HAdd.hAdd N 1)).su …
    -/
    rw [integral_finset_sum]
      /-
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        𝒢 : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        τ π : Ω → Nat
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
        hf : MeasureTheory.Submartingale f 𝒢 μ
        hτ : MeasureTheory.IsStoppingTime 𝒢 τ
        hπ : MeasureTheory.IsStoppingTime 𝒢 π
        hle : LE.le τ π
        N : Nat
        hbdd : ∀ (ω : Ω), LE.le (π ω) N
        this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
        ⊢ LE.le 0 ((Finset.range (HAdd.hAdd N 1)).sum fun i => MeasureTheory.integral  …
      -/
    · refine Finset.sum_nonneg fun i _ => ?_
      /-
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        𝒢 : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        τ π : Ω → Nat
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
        hf : MeasureTheory.Submartingale f 𝒢 μ
        hτ : MeasureTheory.IsStoppingTime 𝒢 τ
        hπ : MeasureTheory.IsStoppingTime 𝒢 π
        hle : LE.le τ π
        N : Nat
        hbdd : ∀ (ω : Ω), LE.le (π ω) N
        this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd N 1)) i
        ⊢ LE.le 0 (MeasureTheory.integral μ fun a => (setOf fun ω => And (LE.le (τ ω)  …
      -/
      rw [integral_indicator (𝒢.le _ _ (this _)), integral_sub', sub_nonneg]
        /-
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          𝒢 : MeasureTheory.Filtration Nat m0
          f : Nat → Ω → Real
          τ π : Ω → Nat
          inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
          hf : MeasureTheory.Submartingale f 𝒢 μ
          hτ : MeasureTheory.IsStoppingTime 𝒢 τ
          hπ : MeasureTheory.IsStoppingTime 𝒢 π
          hle : LE.le τ π
          N : Nat
          hbdd : ∀ (ω : Ω), LE.le (π ω) N
          this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
          i : Nat
          x✝ : Membership.mem (Finset.range (HAdd.hAdd N 1)) i
          ⊢ LE.le (MeasureTheory.integral (μ.restrict (setOf fun ω => And (LE.le (τ ω) i …
        -/
      · exact hf.setIntegral_le (Nat.le_succ i) (this _)
        /-
          🎉 no goals
        -/
        /-
          case hf
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          𝒢 : MeasureTheory.Filtration Nat m0
          f : Nat → Ω → Real
          τ π : Ω → Nat
          inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
          hf : MeasureTheory.Submartingale f 𝒢 μ
          hτ : MeasureTheory.IsStoppingTime 𝒢 τ
          hπ : MeasureTheory.IsStoppingTime 𝒢 π
          hle : LE.le τ π
          N : Nat
          hbdd : ∀ (ω : Ω), LE.le (π ω) N
          this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
          i : Nat
          x✝ : Membership.mem (Finset.range (HAdd.hAdd N 1)) i
          ⊢ MeasureTheory.Integrable (f (HAdd.hAdd i 1)) (μ.restrict (setOf fun ω => And …
        -/
      · exact (hf.integrable _).integrableOn
        /-
          🎉 no goals
        -/
        /-
          case hg
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          𝒢 : MeasureTheory.Filtration Nat m0
          f : Nat → Ω → Real
          τ π : Ω → Nat
          inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
          hf : MeasureTheory.Submartingale f 𝒢 μ
          hτ : MeasureTheory.IsStoppingTime 𝒢 τ
          hπ : MeasureTheory.IsStoppingTime 𝒢 π
          hle : LE.le τ π
          N : Nat
          hbdd : ∀ (ω : Ω), LE.le (π ω) N
          this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
          i : Nat
          x✝ : Membership.mem (Finset.range (HAdd.hAdd N 1)) i
          ⊢ MeasureTheory.Integrable (f i) (μ.restrict (setOf fun ω => And (LE.le (τ ω)  …
        -/
      · exact (hf.integrable _).integrableOn
        /-
          🎉 no goals
        -/
    /-
      case hf
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ π : Ω → Nat
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      hf : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hle : LE.le τ π
      N : Nat
      hbdd : ∀ (ω : Ω), LE.le (π ω) N
      this : ∀ (i : Nat), MeasurableSet (setOf fun ω => And (LE.le (τ ω) i) (LT.lt i …
      ⊢ ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd N 1)) i → MeasureTheory …
    -/
    intro i _
    exact Integrable.indicator (Integrable.sub (hf.integrable _) (hf.integrable _))
      (𝒢.le _ _ (this _))
    /-
      case hf
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ π : Ω → Nat
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      hf : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hle : LE.le τ π
      N : Nat
      hbdd : ∀ (ω : Ω), LE.le (π ω) N
      ⊢ MeasureTheory.Integrable (MeasureTheory.stoppedValue f π) μ
    -/
  · exact hf.integrable_stoppedValue hπ hbdd
    /-
      🎉 no goals
    -/
    /-
      case hg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ π : Ω → Nat
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ 𝒢
      hf : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hle : LE.le τ π
      N : Nat
      hbdd : ∀ (ω : Ω), LE.le (π ω) N
      ⊢ MeasureTheory.Integrable (MeasureTheory.stoppedValue f τ) μ
    -/
  · exact hf.integrable_stoppedValue hτ fun ω => le_trans (hle ω) (hbdd ω)
    /-
      🎉 no goals
    -/


/-- The converse direction of the optional stopping theorem, i.e. an adapted integrable process `f`
is a submartingale if for all bounded stopping times `τ` and `π` such that `τ ≤ π`, the
stopped value of `f` at `τ` has expectation smaller than its stopped value at `π`. -/
theorem submartingale_of_expected_stoppedValue_mono [IsFiniteMeasure μ] (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ) (hf : ∀ τ π : Ω → ℕ, IsStoppingTime 𝒢 τ → IsStoppingTime 𝒢 π →
      τ ≤ π → (∃ N, ∀ ω, π ω ≤ N) → μ[stoppedValue f τ] ≤ μ[stoppedValue f π]) :
    Submartingale f 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hadp : MeasureTheory.Adapted 𝒢 f
    hint : ∀ (i : Nat), MeasureTheory.Integrable (f i) μ
    hf : ∀ (τ π : Ω → Nat), MeasureTheory.IsStoppingTime 𝒢 τ → MeasureTheory.IsSto …
    ⊢ MeasureTheory.Submartingale f 𝒢 μ
  -/
  refine submartingale_of_setIntegral_le hadp hint fun i j hij s hs => ?_
  classical
  specialize hf (s.piecewise (fun _ => i) fun _ => j) _ (isStoppingTime_piecewise_const hij hs)
    (isStoppingTime_const 𝒢 j) (fun x => (ite_le_sup _ _ (x ∈ s)).trans (max_eq_right hij).le)
    ⟨j, fun _ => le_rfl⟩
  rwa [stoppedValue_const, stoppedValue_piecewise_const,
    integral_piecewise (𝒢.le _ _ hs) (hint _).integrableOn (hint _).integrableOn, ←
    integral_add_compl (𝒢.le _ _ hs) (hint j), add_le_add_iff_right] at hf


/-- **The optional stopping theorem** (fair game theorem): an adapted integrable process `f`
is a submartingale if and only if for all bounded stopping times `τ` and `π` such that `τ ≤ π`, the
stopped value of `f` at `τ` has expectation smaller than its stopped value at `π`. -/
theorem submartingale_iff_expected_stoppedValue_mono [IsFiniteMeasure μ] (hadp : Adapted 𝒢 f)
    (hint : ∀ i, Integrable (f i) μ) :
    Submartingale f 𝒢 μ ↔ ∀ τ π : Ω → ℕ, IsStoppingTime 𝒢 τ → IsStoppingTime 𝒢 π →
      τ ≤ π → (∃ N, ∀ x, π x ≤ N) → μ[stoppedValue f τ] ≤ μ[stoppedValue f π] :=
  ⟨fun hf _ _ hτ hπ hle ⟨_, hN⟩ => hf.expected_stoppedValue_mono hτ hπ hle hN,
    submartingale_of_expected_stoppedValue_mono hadp hint⟩


/-- The stopped process of a submartingale with respect to a stopping time is a submartingale. -/
protected theorem Submartingale.stoppedProcess [IsFiniteMeasure μ] (h : Submartingale f 𝒢 μ)
    (hτ : IsStoppingTime 𝒢 τ) : Submartingale (stoppedProcess f τ) 𝒢 μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    τ : Ω → Nat
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : MeasureTheory.Submartingale f 𝒢 μ
    hτ : MeasureTheory.IsStoppingTime 𝒢 τ
    ⊢ MeasureTheory.Submartingale (MeasureTheory.stoppedProcess f τ) 𝒢 μ
  -/
  rw [submartingale_iff_expected_stoppedValue_mono]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ : Ω → Nat
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      ⊢ ∀ (τ_1 π : Ω → Nat), MeasureTheory.IsStoppingTime 𝒢 τ_1 → MeasureTheory.IsSt …
    -/
  · intro σ π hσ hπ hσ_le_π hπ_bdd
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ : Ω → Nat
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime 𝒢 σ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hσ_le_π : LE.le σ π
      hπ_bdd : Exists fun N => ∀ (x : Ω), LE.le (π x) N
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue (Measure …
    -/
    simp_rw [stoppedValue_stoppedProcess]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ : Ω → Nat
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime 𝒢 σ
      hπ : MeasureTheory.IsStoppingTime 𝒢 π
      hσ_le_π : LE.le σ π
      hπ_bdd : Exists fun N => ∀ (x : Ω), LE.le (π x) N
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue f (fun ω …
    -/
    obtain ⟨n, hπ_le_n⟩ := hπ_bdd
    exact h.expected_stoppedValue_mono (hσ.min hτ) (hπ.min hτ)
      (fun ω => min_le_min (hσ_le_π ω) le_rfl) fun ω => (min_le_left _ _).trans (hπ_le_n ω)
    /-
      case hadp
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      τ : Ω → Nat
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : MeasureTheory.Submartingale f 𝒢 μ
      hτ : MeasureTheory.IsStoppingTime 𝒢 τ
      ⊢ MeasureTheory.Adapted 𝒢 (MeasureTheory.stoppedProcess f τ)
    -/
  · exact Adapted.stoppedProcess_of_discrete h.adapted hτ
    /-
      🎉 no goals
    -/
  · exact fun i =>
      h.integrable_stoppedValue ((isStoppingTime_const _ i).min hτ) fun ω => min_le_left _ _


theorem smul_le_stoppedValue_hitting [IsFiniteMeasure μ] (hsub : Submartingale f 𝒢 μ) {ε : ℝ≥0}
    (n : ℕ) : ε • μ {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω} ≤
    ENNReal.ofReal (∫ ω in {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω},
      stoppedValue f (hitting f {y : ℝ | ↑ε ≤ y} 0 n) ω ∂μ) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hsub : MeasureTheory.Submartingale f 𝒢 μ
    ε : NNReal
    n : Nat
    ⊢ LE.le (HSMul.hSMul ε (μ (setOf fun ω => LE.le (↑ε) ((Finset.range (HAdd.hAdd …
  -/
  have hn : Set.Icc 0 n = {k | k ≤ n} := by ext x; simp
  have : ∀ ω, ((ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω) →
      (ε : ℝ) ≤ stoppedValue f (hitting f {y : ℝ | ↑ε ≤ y} 0 n) ω := by
    intro x hx
    simp_rw [le_sup'_iff, mem_range, Nat.lt_succ_iff] at hx
    refine stoppedValue_hitting_mem ?_
    simp only [Set.mem_setOf_eq, exists_prop, hn]
    exact
      let ⟨j, hj₁, hj₂⟩ := hx
      ⟨j, hj₁, hj₂⟩
  have h := setIntegral_ge_of_const_le (measurableSet_le measurable_const
    (Finset.measurable_range_sup'' fun n _ => (hsub.stronglyMeasurable n).measurable.le (𝒢.le n)))
      (measure_ne_top _ _) this (Integrable.integrableOn (hsub.integrable_stoppedValue
        (hitting_isStoppingTime hsub.adapted measurableSet_Ici) hitting_le))
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    𝒢 : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hsub : MeasureTheory.Submartingale f 𝒢 μ
    ε : NNReal
    n : Nat
    hn : Eq (Set.Icc 0 n) (setOf fun k => LE.le k n)
    this : ∀ (ω : Ω), LE.le (↑ε) ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ fun k => f …
    h : LE.le (HMul.hMul (↑ε) (μ (setOf fun a => LE.le (↑ε) ((Finset.range (HAdd.h …
    ⊢ LE.le (HSMul.hSMul ε (μ (setOf fun ω => LE.le (↑ε) ((Finset.range (HAdd.hAdd …
  -/
  rw [ENNReal.le_ofReal_iff_toReal_le, ENNReal.toReal_smul]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hsub : MeasureTheory.Submartingale f 𝒢 μ
      ε : NNReal
      n : Nat
      hn : Eq (Set.Icc 0 n) (setOf fun k => LE.le k n)
      this : ∀ (ω : Ω), LE.le (↑ε) ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ fun k => f …
      h : LE.le (HMul.hMul (↑ε) (μ (setOf fun a => LE.le (↑ε) ((Finset.range (HAdd.h …
      ⊢ LE.le (HSMul.hSMul ε (μ (setOf fun ω => LE.le (↑ε) ((Finset.range (HAdd.hAdd …
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case ha
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hsub : MeasureTheory.Submartingale f 𝒢 μ
      ε : NNReal
      n : Nat
      hn : Eq (Set.Icc 0 n) (setOf fun k => LE.le k n)
      this : ∀ (ω : Ω), LE.le (↑ε) ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ fun k => f …
      h : LE.le (HMul.hMul (↑ε) (μ (setOf fun a => LE.le (↑ε) ((Finset.range (HAdd.h …
      ⊢ Ne (HSMul.hSMul ε (μ (setOf fun ω => LE.le (↑ε) ((Finset.range (HAdd.hAdd n  …
    -/
  · exact ENNReal.mul_ne_top (by simp) (measure_ne_top _ _)
    /-
      🎉 no goals
    -/
    /-
      case hb
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      𝒢 : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hsub : MeasureTheory.Submartingale f 𝒢 μ
      ε : NNReal
      n : Nat
      hn : Eq (Set.Icc 0 n) (setOf fun k => LE.le k n)
      this : ∀ (ω : Ω), LE.le (↑ε) ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ fun k => f …
      h : LE.le (HMul.hMul (↑ε) (μ (setOf fun a => LE.le (↑ε) ((Finset.range (HAdd.h …
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (setOf fun ω => LE.le (↑ε) ((Fin …
    -/
  · exact le_trans (mul_nonneg ε.coe_nonneg ENNReal.toReal_nonneg) h
    /-
      🎉 no goals
    -/


/-- **Doob's maximal inequality**: Given a non-negative submartingale `f`, for all `ε : ℝ≥0`,
we have `ε • μ {ε ≤ f* n} ≤ ∫ ω in {ε ≤ f* n}, f n` where `f* n ω = max_{k ≤ n}, f k ω`.

In some literature, the Doob's maximal inequality refers to what we call Doob's Lp inequality
(which is a corollary of this lemma and will be proved in an upcoming PR). -/
theorem maximal_ineq [IsFiniteMeasure μ] (hsub : Submartingale f 𝒢 μ) (hnonneg : 0 ≤ f) {ε : ℝ≥0}
    (n : ℕ) : ε • μ {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω} ≤
    ENNReal.ofReal (∫ ω in {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω},
      f n ω ∂μ) := by
  suffices ε • μ {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω} +
      ENNReal.ofReal
          (∫ ω in {ω | ((range (n + 1)).sup' nonempty_range_succ fun k => f k ω) < ε}, f n ω ∂μ) ≤
      ENNReal.ofReal (μ[f n]) by
    have hadd : ENNReal.ofReal (∫ ω, f n ω ∂μ) =
      ENNReal.ofReal
        (∫ ω in {ω | ↑ε ≤ (range (n+1)).sup' nonempty_range_succ fun k => f k ω}, f n ω ∂μ) +
      ENNReal.ofReal
        (∫ ω in {ω | ((range (n+1)).sup' nonempty_range_succ fun k => f k ω) < ↑ε}, f n ω ∂μ) := by
      rw [← ENNReal.ofReal_add, ← setIntegral_union]
      · rw [← setIntegral_univ]
        convert rfl
        ext ω
        change (ε : ℝ) ≤ _ ∨ _ < (ε : ℝ) ↔ _
        simp only [le_or_lt, Set.mem_univ]
      · rw [disjoint_iff_inf_le]
        rintro ω ⟨hω₁, hω₂⟩
        change (ε : ℝ) ≤ _ at hω₁
        change _ < (ε : ℝ) at hω₂
        exact (not_le.2 hω₂) hω₁
      · exact measurableSet_lt (Finset.measurable_range_sup'' fun n _ =>
          (hsub.stronglyMeasurable n).measurable.le (𝒢.le n)) measurable_const
      exacts [(hsub.integrable _).integrableOn, (hsub.integrable _).integrableOn,
        integral_nonneg (hnonneg _), integral_nonneg (hnonneg _)]
    rwa [hadd, ENNReal.add_le_add_iff_right ENNReal.ofReal_ne_top] at this
  calc
    ε • μ {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω} +
        ENNReal.ofReal
          (∫ ω in {ω | ((range (n + 1)).sup' nonempty_range_succ fun k => f k ω) < ε}, f n ω ∂μ) ≤
        ENNReal.ofReal
          (∫ ω in {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_succ fun k => f k ω},
            stoppedValue f (hitting f {y : ℝ | ↑ε ≤ y} 0 n) ω ∂μ) +
        ENNReal.ofReal
          (∫ ω in {ω | ((range (n + 1)).sup' nonempty_range_succ fun k => f k ω) < ε},
            stoppedValue f (hitting f {y : ℝ | ↑ε ≤ y} 0 n) ω ∂μ) := by
      refine add_le_add (smul_le_stoppedValue_hitting hsub _)
        (ENNReal.ofReal_le_ofReal (setIntegral_mono_on (hsub.integrable n).integrableOn
          (Integrable.integrableOn (hsub.integrable_stoppedValue
            (hitting_isStoppingTime hsub.adapted measurableSet_Ici) hitting_le))
              (measurableSet_lt (Finset.measurable_range_sup'' fun n _ =>
                (hsub.stronglyMeasurable n).measurable.le (𝒢.le n)) measurable_const) ?_))
      intro ω hω
      rw [Set.mem_setOf_eq] at hω
      have : hitting f {y : ℝ | ↑ε ≤ y} 0 n ω = n := by
        classical simp only [hitting, Set.mem_setOf_eq, exists_prop, Pi.natCast_def, Nat.cast_id,
          ite_eq_right_iff, forall_exists_index, and_imp]
        intro m hm hεm
        exact False.elim
          ((not_le.2 hω) ((le_sup'_iff _).2 ⟨m, mem_range.2 (Nat.lt_succ_of_le hm.2), hεm⟩))
      simp_rw [stoppedValue, this, le_rfl]
    _ = ENNReal.ofReal (∫ ω, stoppedValue f (hitting f {y : ℝ | ↑ε ≤ y} 0 n) ω ∂μ) := by
      rw [← ENNReal.ofReal_add, ← setIntegral_union]
      · rw [← setIntegral_univ (μ := μ)]
        convert rfl
        ext ω
        change _ ↔ (ε : ℝ) ≤ _ ∨ _ < (ε : ℝ)
        simp only [le_or_lt, Set.mem_univ]
      · rw [disjoint_iff_inf_le]
        rintro ω ⟨hω₁, hω₂⟩
        change (ε : ℝ) ≤ _ at hω₁
        change _ < (ε : ℝ) at hω₂
        exact (not_le.2 hω₂) hω₁
      · exact measurableSet_lt (Finset.measurable_range_sup'' fun n _ =>
          (hsub.stronglyMeasurable n).measurable.le (𝒢.le n)) measurable_const
      · exact Integrable.integrableOn (hsub.integrable_stoppedValue
          (hitting_isStoppingTime hsub.adapted measurableSet_Ici) hitting_le)
      · exact Integrable.integrableOn (hsub.integrable_stoppedValue
          (hitting_isStoppingTime hsub.adapted measurableSet_Ici) hitting_le)
      exacts [integral_nonneg fun x => hnonneg _ _, integral_nonneg fun x => hnonneg _ _]
    _ ≤ ENNReal.ofReal (μ[f n]) := by
      refine ENNReal.ofReal_le_ofReal ?_
      rw [← stoppedValue_const f n]
      exact hsub.expected_stoppedValue_mono (hitting_isStoppingTime hsub.adapted measurableSet_Ici)
        (isStoppingTime_const _ _) (fun ω => hitting_le ω) (fun _ => le_refl n)


