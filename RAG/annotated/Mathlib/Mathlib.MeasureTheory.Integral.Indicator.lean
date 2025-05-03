/-- If the indicators of measurable sets `Aᵢ` tend pointwise to the indicator of a set `A`
and we eventually have `Aᵢ ⊆ B` for some set `B` of finite measure, then the measures of `Aᵢ`
tend to the measure of `A`. -/
lemma tendsto_measure_of_tendsto_indicator {μ : Measure α}
    (As_mble : ∀ i, MeasurableSet (As i)) {B : Set α} (B_mble : MeasurableSet B)
    (B_finmeas : μ B ≠ ∞) (As_le_B : ∀ᶠ i in L, As i ⊆ B)
    (h_lim : ∀ x, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    Tendsto (fun i ↦ μ (As i)) L (𝓝 (μ A)) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    A : Set α
    ι : Type u_2
    L : Filter ι
    inst✝ : L.IsCountablyGenerated
    As : ι → Set α
    μ : MeasureTheory.Measure α
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    B : Set α
    B_mble : MeasurableSet B
    B_finmeas : Ne (μ B) Top.top
    As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    ⊢ Filter.Tendsto (fun i => μ (As i)) L (nhds (μ A))
  -/
  rcases L.eq_or_neBot with rfl | _
    /-
      case inl
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_2
      As : ι → Set α
      μ : MeasureTheory.Measure α
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      inst✝ : Bot.bot.IsCountablyGenerated
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) Bot.bot
      h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
      ⊢ Filter.Tendsto (fun i => μ (As i)) Bot.bot (nhds (μ A))
    -/
  · exact tendsto_bot
    /-
      🎉 no goals
    -/
  apply tendsto_measure_of_ae_tendsto_indicator L ?_ As_mble B_mble B_finmeas As_le_B
        (ae_of_all μ h_lim)
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    A : Set α
    ι : Type u_2
    L : Filter ι
    inst✝ : L.IsCountablyGenerated
    As : ι → Set α
    μ : MeasureTheory.Measure α
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    B : Set α
    B_mble : MeasurableSet B
    B_finmeas : Ne (μ B) Top.top
    As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    h✝ : L.NeBot
    ⊢ MeasurableSet A
  -/
  exact measurableSet_of_tendsto_indicator L As_mble h_lim
  /-
    🎉 no goals
  -/


/-- If `μ` is a finite measure and the indicators of measurable sets `Aᵢ` tend pointwise to
the indicator of a set `A`, then the measures `μ Aᵢ` tend to the measure `μ A`. -/
lemma tendsto_measure_of_tendsto_indicator_of_isFiniteMeasure
    (μ : Measure α) [IsFiniteMeasure μ] (As_mble : ∀ i, MeasurableSet (As i))
    (h_lim : ∀ x, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    Tendsto (fun i ↦ μ (As i)) L (𝓝 (μ A)) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    ⊢ Filter.Tendsto (fun i => μ (As i)) L (nhds (μ A))
  -/
  rcases L.eq_or_neBot with rfl | _
    /-
      case inl
      α : Type u_1
      inst✝² : MeasurableSpace α
      A : Set α
      ι : Type u_2
      As : ι → Set α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      inst✝ : Bot.bot.IsCountablyGenerated
      h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
      ⊢ Filter.Tendsto (fun i => μ (As i)) Bot.bot (nhds (μ A))
    -/
  · exact tendsto_bot
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    h✝ : L.NeBot
    ⊢ Filter.Tendsto (fun i => μ (As i)) L (nhds (μ A))
  -/
  apply tendsto_measure_of_ae_tendsto_indicator_of_isFiniteMeasure L ?_ As_mble (ae_of_all μ h_lim)
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    h✝ : L.NeBot
    ⊢ MeasurableSet A
  -/
  exact measurableSet_of_tendsto_indicator L As_mble h_lim
  /-
    🎉 no goals
  -/


