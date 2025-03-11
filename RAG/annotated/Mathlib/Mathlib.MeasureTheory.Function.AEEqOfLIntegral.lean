theorem ae_const_le_iff_forall_lt_measure_zero {β} [LinearOrder β] [TopologicalSpace β]
    [OrderTopology β] [FirstCountableTopology β] (f : α → β) (c : β) :
    (∀ᵐ x ∂μ, c ≤ f x) ↔ ∀ b < c, μ {x | f x ≤ b} = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    ⊢ Iff (Filter.Eventually (fun x => LE.le c (f x)) (MeasureTheory.ae μ)) (∀ (b  …
  -/
  rw [ae_iff]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    ⊢ Iff (Eq (μ (setOf fun a => Not (LE.le c (f a)))) 0) (∀ (b : β), LT.lt b c →  …
  -/
  push_neg
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    ⊢ Iff (Eq (μ (setOf fun a => LT.lt (f a) c)) 0) (∀ (b : β), LT.lt b c → Eq (μ  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_2
      inst✝³ : LinearOrder β
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology β
      inst✝ : FirstCountableTopology β
      f : α → β
      c : β
      ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0 → ∀ (b : β), LT.lt b c → Eq (μ (setO …
    -/
  · intro h b hb
    /-
      case mp
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_2
      inst✝³ : LinearOrder β
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology β
      inst✝ : FirstCountableTopology β
      f : α → β
      c : β
      h : Eq (μ (setOf fun a => LT.lt (f a) c)) 0
      b : β
      hb : LT.lt b c
      ⊢ Eq (μ (setOf fun x => LE.le (f x) b)) 0
    -/
    exact measure_mono_null (fun y hy => (lt_of_le_of_lt hy hb : _)) h
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    ⊢ (∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0) → Eq (μ (se …
  -/
  intro hc
  /-
    case mpr
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
  -/
  by_cases h : ∀ b, c ≤ b
  · have : {a : α | f a < c} = ∅ := by
      apply Set.eq_empty_iff_forall_not_mem.2 fun x hx => ?_
      exact (lt_irrefl _ (lt_of_lt_of_le hx (h (f x)))).elim
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_2
      inst✝³ : LinearOrder β
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology β
      inst✝ : FirstCountableTopology β
      f : α → β
      c : β
      hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
      h : ∀ (b : β), LE.le c b
      this : Eq (setOf fun a => LT.lt (f a) c) EmptyCollection.emptyCollection
      ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    h : Not (∀ (b : β), LE.le c b)
    ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
  -/
  by_cases H : ¬IsLUB (Set.Iio c) c
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_2
      inst✝³ : LinearOrder β
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology β
      inst✝ : FirstCountableTopology β
      f : α → β
      c : β
      hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
      h : Not (∀ (b : β), LE.le c b)
      H : Not (IsLUB (Set.Iio c) c)
      ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
    -/
  · have : c ∈ upperBounds (Set.Iio c) := fun y hy => le_of_lt hy
    obtain ⟨b, b_up, bc⟩ : ∃ b : β, b ∈ upperBounds (Set.Iio c) ∧ b < c := by
      simpa [IsLUB, IsLeast, this, lowerBounds] using H
    /-
      case pos.intro.intro
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_2
      inst✝³ : LinearOrder β
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology β
      inst✝ : FirstCountableTopology β
      f : α → β
      c : β
      hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
      h : Not (∀ (b : β), LE.le c b)
      H : Not (IsLUB (Set.Iio c) c)
      this : Membership.mem (upperBounds (Set.Iio c)) c
      b : β
      b_up : Membership.mem (upperBounds (Set.Iio c)) b
      bc : LT.lt b c
      ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
    -/
    exact measure_mono_null (fun x hx => b_up hx) (hc b bc)
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    h : Not (∀ (b : β), LE.le c b)
    H : Not (Not (IsLUB (Set.Iio c) c))
    ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
  -/
  push_neg at H h
  obtain ⟨u, _, u_lt, u_lim, -⟩ :
    ∃ u : ℕ → β,
      StrictMono u ∧ (∀ n : ℕ, u n < c) ∧ Tendsto u atTop (𝓝 c) ∧ ∀ n : ℕ, u n ∈ Set.Iio c :=
    H.exists_seq_strictMono_tendsto_of_not_mem (lt_irrefl c) h
  have h_Union : {x | f x < c} = ⋃ n : ℕ, {x | f x ≤ u n} := by
    ext1 x
    simp_rw [Set.mem_iUnion, Set.mem_setOf_eq]
    constructor <;> intro h
    · obtain ⟨n, hn⟩ := ((tendsto_order.1 u_lim).1 _ h).exists; exact ⟨n, hn.le⟩
    · obtain ⟨n, hn⟩ := h; exact hn.trans_lt (u_lt _)
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    H : IsLUB (Set.Iio c) c
    h : Exists fun b => LT.lt b c
    u : Nat → β
    left✝ : StrictMono u
    u_lt : ∀ (n : Nat), LT.lt (u n) c
    u_lim : Filter.Tendsto u Filter.atTop (nhds c)
    h_Union : Eq (setOf fun x => LT.lt (f x) c) (Set.iUnion fun n => setOf fun x = …
    ⊢ Eq (μ (setOf fun a => LT.lt (f a) c)) 0
  -/
  rw [h_Union, measure_iUnion_null_iff]
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    H : IsLUB (Set.Iio c) c
    h : Exists fun b => LT.lt b c
    u : Nat → β
    left✝ : StrictMono u
    u_lt : ∀ (n : Nat), LT.lt (u n) c
    u_lim : Filter.Tendsto u Filter.atTop (nhds c)
    h_Union : Eq (setOf fun x => LT.lt (f x) c) (Set.iUnion fun n => setOf fun x = …
    ⊢ ∀ (i : Nat), Eq (μ (setOf fun x => LE.le (f x) (u i))) 0
  -/
  intro n
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : FirstCountableTopology β
    f : α → β
    c : β
    hc : ∀ (b : β), LT.lt b c → Eq (μ (setOf fun x => LE.le (f x) b)) 0
    H : IsLUB (Set.Iio c) c
    h : Exists fun b => LT.lt b c
    u : Nat → β
    left✝ : StrictMono u
    u_lt : ∀ (n : Nat), LT.lt (u n) c
    u_lim : Filter.Tendsto u Filter.atTop (nhds c)
    h_Union : Eq (setOf fun x => LT.lt (f x) c) (Set.iUnion fun n => setOf fun x = …
    n : Nat
    ⊢ Eq (μ (setOf fun x => LE.le (f x) (u n))) 0
  -/
  exact hc _ (u_lt n)
  /-
    🎉 no goals
  -/


theorem ae_le_of_forall_setLIntegral_le_of_sigmaFinite₀ [SigmaFinite μ]
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in s, g x ∂μ) :
    f ≤ᵐ[μ] g := by
  have A : ∀ (ε N : ℝ≥0) (p : ℕ), 0 < ε →
      μ ({x | g x + ε ≤ f x ∧ g x ≤ N} ∩ spanningSets μ p) = 0 := by
    intro ε N p εpos
    let s := {x | g x + ε ≤ f x ∧ g x ≤ N} ∩ spanningSets μ p
    have s_lt_top : μ s < ∞ :=
      (measure_mono (Set.inter_subset_right)).trans_lt (measure_spanningSets_lt_top μ p)
    have A : (∫⁻ x in s, g x ∂μ) + ε * μ s ≤ (∫⁻ x in s, g x ∂μ) + 0 :=
      calc
        (∫⁻ x in s, g x ∂μ) + ε * μ s = (∫⁻ x in s, g x ∂μ) + ∫⁻ _ in s, ε ∂μ := by
          simp only [lintegral_const, Set.univ_inter, MeasurableSet.univ, Measure.restrict_apply]
        _ = ∫⁻ x in s, g x + ε ∂μ := (lintegral_add_right _ measurable_const).symm
        _ ≤ ∫⁻ x in s, f x ∂μ :=
          setLIntegral_mono_ae hf.restrict <| ae_of_all _ fun x hx => hx.1.1
        _ ≤ (∫⁻ x in s, g x ∂μ) + 0 := by
          rw [add_zero, ← Measure.restrict_toMeasurable s_lt_top.ne]
          refine h _ (measurableSet_toMeasurable ..) ?_
          rwa [measure_toMeasurable]
    have B : (∫⁻ x in s, g x ∂μ) ≠ ∞ :=
      (setLIntegral_lt_top_of_le_nnreal s_lt_top.ne ⟨N, fun _ h ↦ h.1.2⟩).ne
    have : (ε : ℝ≥0∞) * μ s ≤ 0 := ENNReal.le_of_add_le_add_left B A
    simpa only [ENNReal.coe_eq_zero, nonpos_iff_eq_zero, mul_eq_zero, εpos.ne', false_or]
  obtain ⟨u, _, u_pos, u_lim⟩ :
    ∃ u : ℕ → ℝ≥0, StrictAnti u ∧ (∀ n, 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ≥0)
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → ENNReal
    hf : AEMeasurable f μ
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureTheor …
    A : ∀ (ε N : NNReal) (p : Nat), LT.lt 0 ε → Eq (μ (Inter.inter (setOf fun x => …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ (MeasureTheory.ae μ).EventuallyLE f g
  -/
  let s := fun n : ℕ => {x | g x + u n ≤ f x ∧ g x ≤ (n : ℝ≥0)} ∩ spanningSets μ n
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → ENNReal
    hf : AEMeasurable f μ
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureTheor …
    A : ∀ (ε N : NNReal) (p : Nat), LT.lt 0 ε → Eq (μ (Inter.inter (setOf fun x => …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    s : Nat → Set α := fun n => Inter.inter (setOf fun x => And (LE.le (HAdd.hAdd  …
    ⊢ (MeasureTheory.ae μ).EventuallyLE f g
  -/
  have μs : ∀ n, μ (s n) = 0 := fun n => A _ _ _ (u_pos n)
  have B : {x | f x ≤ g x}ᶜ ⊆ ⋃ n, s n := by
    intro x hx
    simp only [Set.mem_compl_iff, Set.mem_setOf, not_le] at hx
    have L1 : ∀ᶠ n in atTop, g x + u n ≤ f x := by
      have : Tendsto (fun n => g x + u n) atTop (𝓝 (g x + (0 : ℝ≥0))) :=
        tendsto_const_nhds.add (ENNReal.tendsto_coe.2 u_lim)
      simp only [ENNReal.coe_zero, add_zero] at this
      exact this.eventually_le_const hx
    have L2 : ∀ᶠ n : ℕ in (atTop : Filter ℕ), g x ≤ (n : ℝ≥0) :=
      have : Tendsto (fun n : ℕ => ((n : ℝ≥0) : ℝ≥0∞)) atTop (𝓝 ∞) := by
        simp only [ENNReal.coe_natCast]
        exact ENNReal.tendsto_nat_nhds_top
      this.eventually_const_le (hx.trans_le le_top)
    apply Set.mem_iUnion.2
    exact ((L1.and L2).and (eventually_mem_spanningSets μ x)).exists
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → ENNReal
    hf : AEMeasurable f μ
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureTheor …
    A : ∀ (ε N : NNReal) (p : Nat), LT.lt 0 ε → Eq (μ (Inter.inter (setOf fun x => …
    u : Nat → NNReal
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    s : Nat → Set α := fun n => Inter.inter (setOf fun x => And (LE.le (HAdd.hAdd  …
    μs : ∀ (n : Nat), Eq (μ (s n)) 0
    B : HasSubset.Subset (HasCompl.compl (setOf fun x => LE.le (f x) (g x))) (Set. …
    ⊢ (MeasureTheory.ae μ).EventuallyLE f g
  -/
  refine le_antisymm ?_ bot_le
  calc
    μ {x : α | (fun x : α => f x ≤ g x) x}ᶜ ≤ μ (⋃ n, s n) := measure_mono B
    _ ≤ ∑' n, μ (s n) := measure_iUnion_le _
    _ = 0 := by simp only [μs, tsum_zero]


@[deprecated (since := "2024-06-29")]
alias ae_le_of_forall_set_lintegral_le_of_sigmaFinite₀ :=
  ae_le_of_forall_setLIntegral_le_of_sigmaFinite₀


theorem ae_le_of_forall_setLIntegral_le_of_sigmaFinite [SigmaFinite μ] {f g : α → ℝ≥0∞}
    (hf : Measurable f)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → (∫⁻ x in s, f x ∂μ) ≤ ∫⁻ x in s, g x ∂μ) : f ≤ᵐ[μ] g :=
  ae_le_of_forall_setLIntegral_le_of_sigmaFinite₀ hf.aemeasurable h


@[deprecated (since := "2024-06-29")]
alias ae_le_of_forall_set_lintegral_le_of_sigmaFinite :=
  ae_le_of_forall_setLIntegral_le_of_sigmaFinite


theorem ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite₀ [SigmaFinite μ]
    {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → ∫⁻ x in s, f x ∂μ = ∫⁻ x in s, g x ∂μ) : f =ᵐ[μ] g := by
  have A : f ≤ᵐ[μ] g :=
    ae_le_of_forall_setLIntegral_le_of_sigmaFinite₀ hf fun s hs h's => le_of_eq (h s hs h's)
  have B : g ≤ᵐ[μ] f :=
    ae_le_of_forall_setLIntegral_le_of_sigmaFinite₀ hg fun s hs h's => ge_of_eq (h s hs h's)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.l …
    A : (MeasureTheory.ae μ).EventuallyLE f g
    B : (MeasureTheory.ae μ).EventuallyLE g f
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  filter_upwards [A, B] with x using le_antisymm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias ae_eq_of_forall_set_lintegral_eq_of_sigmaFinite₀ :=
  ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite₀


theorem ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite [SigmaFinite μ] {f g : α → ℝ≥0∞}
    (hf : Measurable f) (hg : Measurable g)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → ∫⁻ x in s, f x ∂μ = ∫⁻ x in s, g x ∂μ) : f =ᵐ[μ] g :=
  ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite₀ hf.aemeasurable hg.aemeasurable h


@[deprecated (since := "2024-06-29")]
alias ae_eq_of_forall_set_lintegral_eq_of_sigmaFinite :=
  ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite


theorem AEMeasurable.ae_eq_of_forall_setLIntegral_eq {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) (hfi : ∫⁻ x, f x ∂μ ≠ ∞) (hgi : ∫⁻ x, g x ∂μ ≠ ∞)
    (hfg : ∀ ⦃s⦄, MeasurableSet s → μ s < ∞ → ∫⁻ x in s, f x ∂μ = ∫⁻ x in s, g x ∂μ) :
    f =ᵐ[μ] g := by
  have hf' : AEFinStronglyMeasurable f μ :=
    ENNReal.aefinStronglyMeasurable_of_aemeasurable hfi hf
  have hg' : AEFinStronglyMeasurable g μ :=
    ENNReal.aefinStronglyMeasurable_of_aemeasurable hgi hg
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  let s := hf'.sigmaFiniteSet
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  let t := hg'.sigmaFiniteSet
  suffices f =ᵐ[μ.restrict (s ∪ t)] g by
    refine ae_of_ae_restrict_of_ae_restrict_compl _ this ?_
    simp only [Set.compl_union]
    have h1 : f =ᵐ[μ.restrict sᶜ] 0 := hf'.ae_eq_zero_compl
    have h2 : g =ᵐ[μ.restrict tᶜ] 0 := hg'.ae_eq_zero_compl
    rw [ae_restrict_iff' (hf'.measurableSet.compl.inter hg'.measurableSet.compl)]
    rw [EventuallyEq, ae_restrict_iff' hf'.measurableSet.compl] at h1
    rw [EventuallyEq, ae_restrict_iff' hg'.measurableSet.compl] at h2
    filter_upwards [h1, h2] with x h1 h2 hx
    rw [h1 (Set.inter_subset_left hx), h2 (Set.inter_subset_right hx)]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    t : Set α := hg'.sigmaFiniteSet
    ⊢ (MeasureTheory.ae (μ.restrict (Union.union s t))).EventuallyEq f g
  -/
  have := hf'.sigmaFinite_restrict
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    t : Set α := hg'.sigmaFiniteSet
    this : MeasureTheory.SigmaFinite (μ.restrict hf'.sigmaFiniteSet)
    ⊢ (MeasureTheory.ae (μ.restrict (Union.union s t))).EventuallyEq f g
  -/
  have := hg'.sigmaFinite_restrict
  refine ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite₀ hf.restrict hg.restrict
    fun u hu huμ ↦ ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    t : Set α := hg'.sigmaFiniteSet
    this✝ : MeasureTheory.SigmaFinite (μ.restrict hf'.sigmaFiniteSet)
    this : MeasureTheory.SigmaFinite (μ.restrict hg'.sigmaFiniteSet)
    u : Set α
    hu : MeasurableSet u
    huμ : LT.lt ((μ.restrict (Union.union s t)) u) Top.top
    ⊢ Eq (MeasureTheory.lintegral ((μ.restrict (Union.union s t)).restrict u) fun  …
  -/
  rw [Measure.restrict_restrict hu]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    t : Set α := hg'.sigmaFiniteSet
    this✝ : MeasureTheory.SigmaFinite (μ.restrict hf'.sigmaFiniteSet)
    this : MeasureTheory.SigmaFinite (μ.restrict hg'.sigmaFiniteSet)
    u : Set α
    hu : MeasurableSet u
    huμ : LT.lt ((μ.restrict (Union.union s t)) u) Top.top
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Inter.inter u (Union.union s t))) f …
  -/
  rw [Measure.restrict_apply hu] at huμ
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hgi : Ne (MeasureTheory.lintegral μ fun x => g x) Top.top
    hfg : ∀ ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hf' : MeasureTheory.AEFinStronglyMeasurable f μ
    hg' : MeasureTheory.AEFinStronglyMeasurable g μ
    s : Set α := hf'.sigmaFiniteSet
    t : Set α := hg'.sigmaFiniteSet
    this✝ : MeasureTheory.SigmaFinite (μ.restrict hf'.sigmaFiniteSet)
    this : MeasureTheory.SigmaFinite (μ.restrict hg'.sigmaFiniteSet)
    u : Set α
    hu : MeasurableSet u
    huμ : LT.lt (μ (Inter.inter u (Union.union s t))) Top.top
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Inter.inter u (Union.union s t))) f …
  -/
  exact hfg (hu.inter (hf'.measurableSet.union hg'.measurableSet)) huμ
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias AEMeasurable.ae_eq_of_forall_set_lintegral_eq := AEMeasurable.ae_eq_of_forall_setLIntegral_eq


theorem withDensity_eq_iff_of_sigmaFinite [SigmaFinite μ] {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) : μ.withDensity f = μ.withDensity g ↔ f =ᵐ[μ] g :=
  ⟨fun hfg ↦ by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hfg : Eq (μ.withDensity f) (μ.withDensity g)
      ⊢ (MeasureTheory.ae μ).EventuallyEq f g
    -/
    refine ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite₀ hf hg fun s hs _ ↦ ?_
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hfg : Eq (μ.withDensity f) (μ.withDensity g)
      s : Set α
      hs : MeasurableSet s
      x✝ : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.lint …
    -/
    rw [← withDensity_apply f hs, ← withDensity_apply g hs, ← hfg], withDensity_congr_ae⟩
    /-
      🎉 no goals
    -/


theorem withDensity_eq_iff {f g : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) (hfi : ∫⁻ x, f x ∂μ ≠ ∞) :
    μ.withDensity f = μ.withDensity g ↔ f =ᵐ[μ] g :=
  ⟨fun hfg ↦ by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      hfg : Eq (μ.withDensity f) (μ.withDensity g)
      ⊢ (MeasureTheory.ae μ).EventuallyEq f g
    -/
    refine AEMeasurable.ae_eq_of_forall_setLIntegral_eq hf hg hfi ?_ fun s hs _ ↦ ?_
    · rwa [← setLIntegral_univ, ← withDensity_apply g MeasurableSet.univ, ← hfg,
        withDensity_apply f MeasurableSet.univ, setLIntegral_univ]
      /-
        case refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : AEMeasurable f μ
        hg : AEMeasurable g μ
        hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
        hfg : Eq (μ.withDensity f) (μ.withDensity g)
        s : Set α
        hs : MeasurableSet s
        x✝ : LT.lt (μ s) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.lint …
      -/
    · rw [← withDensity_apply f hs, ← withDensity_apply g hs, ← hfg], withDensity_congr_ae⟩
      /-
        🎉 no goals
      -/


