/-- Measurability structure on `Measure`: Measures are measurable w.r.t. all projections -/
instance instMeasurableSpace : MeasurableSpace (Measure α) :=
  ⨆ (s : Set α) (_ : MeasurableSet s), (borel ℝ≥0∞).comap fun μ => μ s


theorem measurable_coe {s : Set α} (hs : MeasurableSet s) : Measurable fun μ : Measure α => μ s :=
  Measurable.of_comap_le <| le_iSup_of_le s <| le_iSup_of_le hs <| le_rfl


theorem measurable_of_measurable_coe (f : β → Measure α)
    (h : ∀ (s : Set α), MeasurableSet s → Measurable fun b => f b s) : Measurable f :=
  Measurable.of_le_map <|
    iSup₂_le fun s hs =>
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝¹ : MeasurableSpace α
                                                    inst✝ : MeasurableSpace β
                                                    f : β → MeasureTheory.Measure α
                                                    h : ∀ (s : Set α), MeasurableSet s → Measurable fun b => (f b) s
                                                    s : Set α
                                                    hs : MeasurableSet s
                                                    ⊢ LE.le (borel ENNReal) (MeasurableSpace.map (fun μ => μ s) (MeasurableSpace.m …
                                                  -/
      MeasurableSpace.comap_le_iff_le_map.2 <| by rw [MeasurableSpace.map_comp]; exact h s hs
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


instance instMeasurableAdd₂ {α : Type*} {m : MeasurableSpace α} : MeasurableAdd₂ (Measure α) := by
  /-
    α✝ : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α✝
    inst✝ : MeasurableSpace β
    α : Type u_3
    m : MeasurableSpace α
    ⊢ MeasurableAdd₂ (MeasureTheory.Measure α)
  -/
  refine ⟨Measure.measurable_of_measurable_coe _ fun s hs => ?_⟩
  /-
    α✝ : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α✝
    inst✝ : MeasurableSpace β
    α : Type u_3
    m : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Measurable fun b => (HAdd.hAdd b.1 b.2) s
  -/
  simp_rw [Measure.coe_add, Pi.add_apply]
  /-
    α✝ : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α✝
    inst✝ : MeasurableSpace β
    α : Type u_3
    m : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Measurable fun b => HAdd.hAdd (b.1 s) (b.2 s)
  -/
  refine Measurable.add ?_ ?_
    /-
      case refine_1
      α✝ : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α✝
      inst✝ : MeasurableSpace β
      α : Type u_3
      m : MeasurableSpace α
      s : Set α
      hs : MeasurableSet s
      ⊢ Measurable fun b => b.1 s
    -/
  · exact (Measure.measurable_coe hs).comp measurable_fst
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α✝ : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α✝
      inst✝ : MeasurableSpace β
      α : Type u_3
      m : MeasurableSpace α
      s : Set α
      hs : MeasurableSet s
      ⊢ Measurable fun b => b.2 s
    -/
  · exact (Measure.measurable_coe hs).comp measurable_snd
    /-
      🎉 no goals
    -/


theorem measurable_measure {μ : α → Measure β} :
    Measurable μ ↔ ∀ (s : Set β), MeasurableSet s → Measurable fun b => μ b s :=
  ⟨fun hμ _s hs => (measurable_coe hs).comp hμ, measurable_of_measurable_coe μ⟩


theorem _root_.Measurable.measure_of_isPiSystem {μ : α → Measure β} [∀ a, IsFiniteMeasure (μ a)]
    {S : Set (Set β)} (hgen : ‹MeasurableSpace β› = .generateFrom S) (hpi : IsPiSystem S)
    (h_basic : ∀ s ∈ S, Measurable fun a ↦ μ a s) (h_univ : Measurable fun a ↦ μ a univ) :
    Measurable μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : α → MeasureTheory.Measure β
    inst✝ : ∀ (a : α), MeasureTheory.IsFiniteMeasure (μ a)
    S : Set (Set β)
    hgen : Eq inst✝¹ (MeasurableSpace.generateFrom S)
    hpi : IsPiSystem S
    h_basic : ∀ (s : Set β), Membership.mem S s → Measurable fun a => (μ a) s
    h_univ : Measurable fun a => (μ a) Set.univ
    ⊢ Measurable μ
  -/
  rw [measurable_measure]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : α → MeasureTheory.Measure β
    inst✝ : ∀ (a : α), MeasureTheory.IsFiniteMeasure (μ a)
    S : Set (Set β)
    hgen : Eq inst✝¹ (MeasurableSpace.generateFrom S)
    hpi : IsPiSystem S
    h_basic : ∀ (s : Set β), Membership.mem S s → Measurable fun a => (μ a) s
    h_univ : Measurable fun a => (μ a) Set.univ
    ⊢ ∀ (s : Set β), MeasurableSet s → Measurable fun b => (μ b) s
  -/
  intro s hs
  induction s, hs using MeasurableSpace.induction_on_inter hgen hpi with
  | empty => simp
  | basic s hs => exact h_basic s hs
  | compl s hsm ihs =>
    simp only [measure_compl hsm (measure_ne_top _ _)]
    exact h_univ.sub ihs
  | iUnion f hfd hfm ihf =>
    simpa only [measure_iUnion hfd hfm] using .ennreal_tsum ihf


theorem _root_.Measurable.measure_of_isPiSystem_of_isProbabilityMeasure {μ : α → Measure β}
    [∀ a, IsProbabilityMeasure (μ a)]
    {S : Set (Set β)} (hgen : ‹MeasurableSpace β› = .generateFrom S) (hpi : IsPiSystem S)
    (h_basic : ∀ s ∈ S, Measurable fun a ↦ μ a s) : Measurable μ :=
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  inst✝² : MeasurableSpace α
                                                  inst✝¹ : MeasurableSpace β
                                                  μ : α → MeasureTheory.Measure β
                                                  inst✝ : ∀ (a : α), MeasureTheory.IsProbabilityMeasure (μ a)
                                                  S : Set (Set β)
                                                  hgen : Eq inst✝¹ (MeasurableSpace.generateFrom S)
                                                  hpi : IsPiSystem S
                                                  h_basic : ∀ (s : Set β), Membership.mem S s → Measurable fun a => (μ a) s
                                                  ⊢ Measurable fun a => (μ a) Set.univ
                                                -/
  .measure_of_isPiSystem hgen hpi h_basic <| by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem measurable_map (f : α → β) (hf : Measurable f) :
    Measurable fun μ : Measure α => map f μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    ⊢ Measurable fun μ => MeasureTheory.Measure.map f μ
  -/
  refine measurable_of_measurable_coe _ fun s hs => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun b => (MeasureTheory.Measure.map f b) s
  -/
  simp_rw [map_apply hf hs]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable fun b => b (Set.preimage f s)
  -/
  exact measurable_coe (hf hs)
  /-
    🎉 no goals
  -/


theorem measurable_dirac : Measurable (Measure.dirac : α → Measure α) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Measurable MeasureTheory.Measure.dirac
  -/
  refine measurable_of_measurable_coe _ fun s hs => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Measurable fun b => (MeasureTheory.Measure.dirac b) s
  -/
  simp_rw [dirac_apply' _ hs]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Measurable fun b => s.indicator 1 b
  -/
  exact measurable_one.indicator hs
  /-
    🎉 no goals
  -/


theorem measurable_lintegral {f : α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun μ : Measure α => ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Measurable fun μ => MeasureTheory.lintegral μ fun x => f x
  -/
  simp only [lintegral_eq_iSup_eapprox_lintegral, hf, SimpleFunc.lintegral]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Measurable fun μ => iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n).ran …
  -/
  refine .iSup fun n => Finset.measurable_sum _ fun i _ => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    n : Nat
    i : ENNReal
    x✝ : Membership.mem (MeasureTheory.SimpleFunc.eapprox f n).range i
    ⊢ Measurable fun μ => HMul.hMul i (μ (Set.preimage (⇑(MeasureTheory.SimpleFunc …
  -/
  refine Measurable.const_mul ?_ _
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    n : Nat
    i : ENNReal
    x✝ : Membership.mem (MeasureTheory.SimpleFunc.eapprox f n).range i
    ⊢ Measurable fun μ => μ (Set.preimage (⇑(MeasureTheory.SimpleFunc.eapprox f n) …
  -/
  exact measurable_coe ((SimpleFunc.eapprox f n).measurableSet_preimage _)
  /-
    🎉 no goals
  -/


/-- Monadic join on `Measure` in the category of measurable spaces and measurable
functions. -/
def join (m : Measure (Measure α)) : Measure α :=
  Measure.ofMeasurable (fun s _ => ∫⁻ μ, μ s ∂m)
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : MeasurableSpace α
          inst✝ : MeasurableSpace β
          m : MeasureTheory.Measure (MeasureTheory.Measure α)
          ⊢ Eq ((fun s x => MeasureTheory.lintegral m fun μ => μ s) EmptyCollection.empt …
        -/
    (by simp only [measure_empty, lintegral_const, zero_mul])
        /-
          🎉 no goals
        -/
    (by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        m : MeasureTheory.Measure (MeasureTheory.Measure α)
        ⊢ ∀ ⦃f : Nat → Set α⦄ (h : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Functi …
      -/
      intro f hf h
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        m : MeasureTheory.Measure (MeasureTheory.Measure α)
        f : Nat → Set α
        hf : ∀ (i : Nat), MeasurableSet (f i)
        h : Pairwise (Function.onFun Disjoint f)
        ⊢ Eq ((fun s x => MeasureTheory.lintegral m fun μ => μ s) (Set.iUnion fun i => …
      -/
      simp_rw [measure_iUnion h hf]
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        m : MeasureTheory.Measure (MeasureTheory.Measure α)
        f : Nat → Set α
        hf : ∀ (i : Nat), MeasurableSet (f i)
        h : Pairwise (Function.onFun Disjoint f)
        ⊢ Eq (MeasureTheory.lintegral m fun μ => tsum fun i => μ (f i)) (tsum fun i => …
      -/
      apply lintegral_tsum
      /-
        case hf
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        m : MeasureTheory.Measure (MeasureTheory.Measure α)
        f : Nat → Set α
        hf : ∀ (i : Nat), MeasurableSet (f i)
        h : Pairwise (Function.onFun Disjoint f)
        ⊢ ∀ (i : Nat), AEMeasurable (fun a => a (f i)) m
      -/
      intro i; exact (measurable_coe (hf i)).aemeasurable)
               /-
                 🎉 no goals
               -/


@[simp]
theorem join_apply {m : Measure (Measure α)} {s : Set α} (hs : MeasurableSet s) :
    join m s = ∫⁻ μ, μ s ∂m :=
  Measure.ofMeasurable_apply s hs


@[simp]
theorem join_zero : (0 : Measure (Measure α)).join = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Eq (MeasureTheory.Measure.join 0) 0
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.join 0) s) (0 s)
  -/
  simp only [hs, join_apply, lintegral_zero_measure, coe_zero, Pi.zero_apply]
  /-
    🎉 no goals
  -/


theorem measurable_join : Measurable (join : Measure (Measure α) → Measure α) :=
  measurable_of_measurable_coe _ fun s hs => by
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : Set α
      hs : MeasurableSet s
      ⊢ Measurable fun b => b.join s
    -/
    simp only [join_apply hs]; exact measurable_lintegral (measurable_coe hs)
                               /-
                                 🎉 no goals
                               -/


theorem lintegral_join {m : Measure (Measure α)} {f : α → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ x, f x ∂join m = ∫⁻ μ, ∫⁻ x, f x ∂μ ∂m := by
  simp_rw [lintegral_eq_iSup_eapprox_lintegral hf, SimpleFunc.lintegral,
    join_apply (SimpleFunc.measurableSet_preimage _ _)]
  suffices
    ∀ (s : ℕ → Finset ℝ≥0∞) (f : ℕ → ℝ≥0∞ → Measure α → ℝ≥0∞), (∀ n r, Measurable (f n r)) →
      Monotone (fun n μ => ∑ r ∈ s n, r * f n r μ) →
      ⨆ n, ∑ r ∈ s n, r * ∫⁻ μ, f n r μ ∂m = ∫⁻ μ, ⨆ n, ∑ r ∈ s n, r * f n r μ ∂m by
    refine
      this (fun n => SimpleFunc.range (SimpleFunc.eapprox f n))
        (fun n r μ => μ (SimpleFunc.eapprox f n ⁻¹' {r})) ?_ ?_
    · exact fun n r => measurable_coe (SimpleFunc.measurableSet_preimage _ _)
    · exact fun n m h μ => SimpleFunc.lintegral_mono (SimpleFunc.monotone_eapprox _ h) le_rfl
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f : α → ENNReal
    hf : Measurable f
    ⊢ ∀ (s : Nat → Finset ENNReal) (f : Nat → ENNReal → MeasureTheory.Measure α →  …
  -/
  intro s f hf hm
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f✝ : α → ENNReal
    hf✝ : Measurable f✝
    s : Nat → Finset ENNReal
    f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
    hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
    hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    ⊢ Eq (iSup fun n => (s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m  …
  -/
  rw [lintegral_iSup _ hm]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f✝ : α → ENNReal
    hf✝ : Measurable f✝
    s : Nat → Finset ENNReal
    f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
    hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
    hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    ⊢ Eq (iSup fun n => (s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m  …
  -/
  swap
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.Measure (MeasureTheory.Measure α)
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      s : Nat → Finset ENNReal
      f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
      hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
      hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
      ⊢ ∀ (n : Nat), Measurable fun μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    -/
  · exact fun n => Finset.measurable_sum _ fun r _ => (hf _ _).const_mul _
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f✝ : α → ENNReal
    hf✝ : Measurable f✝
    s : Nat → Finset ENNReal
    f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
    hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
    hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    ⊢ Eq (iSup fun n => (s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m  …
  -/
  congr
  /-
    case e_s
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f✝ : α → ENNReal
    hf✝ : Measurable f✝
    s : Nat → Finset ENNReal
    f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
    hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
    hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    ⊢ Eq (fun n => (s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m fun μ …
  -/
  funext n
  /-
    case e_s.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure (MeasureTheory.Measure α)
    f✝ : α → ENNReal
    hf✝ : Measurable f✝
    s : Nat → Finset ENNReal
    f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
    hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
    hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
    n : Nat
    ⊢ Eq ((s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m fun μ => f n r …
  -/
  rw [lintegral_finset_sum (s n)]
    /-
      case e_s.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.Measure (MeasureTheory.Measure α)
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      s : Nat → Finset ENNReal
      f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
      hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
      hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
      n : Nat
      ⊢ Eq ((s n).sum fun r => HMul.hMul r (MeasureTheory.lintegral m fun μ => f n r …
    -/
  · simp_rw [lintegral_const_mul _ (hf _ _)]
    /-
      🎉 no goals
    -/
    /-
      case e_s.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.Measure (MeasureTheory.Measure α)
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      s : Nat → Finset ENNReal
      f : Nat → ENNReal → MeasureTheory.Measure α → ENNReal
      hf : ∀ (n : Nat) (r : ENNReal), Measurable (f n r)
      hm : Monotone fun n μ => (s n).sum fun r => HMul.hMul r (f n r μ)
      n : Nat
      ⊢ ∀ (b : ENNReal), Membership.mem (s n) b → Measurable fun a => HMul.hMul b (f …
    -/
  · exact fun r _ => (hf _ _).const_mul _
    /-
      🎉 no goals
    -/


/-- Monadic bind on `Measure`, only works in the category of measurable spaces and measurable
functions. When the function `f` is not measurable the result is not well defined. -/
def bind (m : Measure α) (f : α → Measure β) : Measure β :=
  join (map f m)


@[simp]
                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  inst✝¹ : MeasurableSpace α
                                                                  inst✝ : MeasurableSpace β
                                                                  f : α → MeasureTheory.Measure β
                                                                  ⊢ Eq (MeasureTheory.Measure.bind 0 f) 0
                                                                -/
theorem bind_zero_left (f : α → Measure β) : bind 0 f = 0 := by simp [bind]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem bind_zero_right (m : Measure α) : bind m (0 : α → Measure β) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    ⊢ Eq (m.bind 0) 0
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((m.bind 0) s) (0 s)
  -/
  simp only [bind, hs, join_apply, coe_zero, Pi.zero_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map 0 m) fun μ => μ s) 0
  -/
  rw [lintegral_map (measurable_coe hs) measurable_zero]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => (0 a) s) 0
  -/
  simp only [Pi.zero_apply, coe_zero, lintegral_const, zero_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem bind_zero_right' (m : Measure α) : bind m (fun _ => 0 : α → Measure β) = 0 :=
  bind_zero_right m


@[simp]
theorem bind_apply {m : Measure α} {f : α → Measure β} {s : Set β} (hs : MeasurableSet s)
    (hf : Measurable f) : bind m f s = ∫⁻ a, f a s ∂m := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → MeasureTheory.Measure β
    s : Set β
    hs : MeasurableSet s
    hf : Measurable f
    ⊢ Eq ((m.bind f) s) (MeasureTheory.lintegral m fun a => (f a) s)
  -/
  rw [bind, join_apply hs, lintegral_map (measurable_coe hs) hf]
  /-
    🎉 no goals
  -/


@[simp]
lemma bind_const {m : Measure α} {ν : Measure β} : m.bind (fun _ ↦ ν) = m Set.univ • ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    ⊢ Eq (m.bind fun x => ν) (HSMul.hSMul (m Set.univ) ν)
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((m.bind fun x => ν) s) ((HSMul.hSMul (m Set.univ) ν) s)
  -/
  rw [bind_apply hs measurable_const, lintegral_const, smul_apply, smul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


theorem measurable_bind' {g : α → Measure β} (hg : Measurable g) : Measurable fun m => bind m g :=
  measurable_join.comp (measurable_map _ hg)


theorem lintegral_bind {m : Measure α} {μ : α → Measure β} {f : β → ℝ≥0∞} (hμ : Measurable μ)
    (hf : Measurable f) : ∫⁻ x, f x ∂bind m μ = ∫⁻ a, ∫⁻ x, f x ∂μ a ∂m :=
  (lintegral_join hf).trans (lintegral_map (measurable_lintegral hf) hμ)


theorem bind_bind {γ} [MeasurableSpace γ] {m : Measure α} {f : α → Measure β} {g : β → Measure γ}
    (hf : Measurable f) (hg : Measurable g) : bind (bind m f) g = bind m fun a => bind (f a) g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    γ : Type u_3
    inst✝ : MeasurableSpace γ
    m : MeasureTheory.Measure α
    f : α → MeasureTheory.Measure β
    g : β → MeasureTheory.Measure γ
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq ((m.bind f).bind g) (m.bind fun a => (f a).bind g)
  -/
  ext1 s hs
  erw [bind_apply hs hg, bind_apply hs ((measurable_bind' hg).comp hf),
    lintegral_bind hf ((measurable_coe hs).comp hg)]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    γ : Type u_3
    inst✝ : MeasurableSpace γ
    m : MeasureTheory.Measure α
    f : α → MeasureTheory.Measure β
    g : β → MeasureTheory.Measure γ
    hf : Measurable f
    hg : Measurable g
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => MeasureTheory.lintegral (f a) fun x = …
  -/
  conv_rhs => enter [2, a]; erw [bind_apply hs hg]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    γ : Type u_3
    inst✝ : MeasurableSpace γ
    m : MeasureTheory.Measure α
    f : α → MeasureTheory.Measure β
    g : β → MeasureTheory.Measure γ
    hf : Measurable f
    hg : Measurable g
    s : Set γ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => MeasureTheory.lintegral (f a) fun x = …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dirac_bind {f : α → Measure β} (hf : Measurable f) (a : α) : bind (dirac a) f = f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → MeasureTheory.Measure β
    hf : Measurable f
    a : α
    ⊢ Eq ((MeasureTheory.Measure.dirac a).bind f) (f a)
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → MeasureTheory.Measure β
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.Measure.dirac a).bind f) s) ((f a) s)
  -/
  erw [bind_apply hs hf, lintegral_dirac' a ((measurable_coe hs).comp hf)]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → MeasureTheory.Measure β
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (Function.comp (fun μ => μ s) f a) ((f a) s)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem bind_dirac {m : Measure α} : bind m dirac = m := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.Measure α
    ⊢ Eq (m.bind MeasureTheory.Measure.dirac) m
  -/
  ext1 s hs
  simp only [bind_apply hs measurable_dirac, dirac_apply' _ hs, lintegral_indicator hs,
    Pi.one_apply, lintegral_one, restrict_apply, MeasurableSet.univ, univ_inter]


@[simp]
lemma bind_dirac_eq_map (m : Measure α) {f : α → β} (hf : Measurable f) :
    m.bind (fun x ↦ Measure.dirac (f x)) = m.map f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    ⊢ Eq (m.bind fun x => MeasureTheory.Measure.dirac (f x)) (MeasureTheory.Measur …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((m.bind fun x => MeasureTheory.Measure.dirac (f x)) s) ((MeasureTheory.M …
  -/
  rw [bind_apply hs]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => (MeasureTheory.Measure.dirac (f a)) s …
  -/
  swap; · exact measurable_dirac.comp hf
          /-
            🎉 no goals
          -/
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => (MeasureTheory.Measure.dirac (f a)) s …
  -/
  simp_rw [dirac_apply' _ hs]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun a => s.indicator 1 (f a)) ((MeasureTheory. …
  -/
  rw [← lintegral_map _ hf, lintegral_indicator_one hs]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    m : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Measurable (s.indicator 1)
  -/
  exact measurable_const.indicator hs
  /-
    🎉 no goals
  -/


                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝ : MeasurableSpace α
                                                                            μ : MeasureTheory.Measure (MeasureTheory.Measure α)
                                                                            ⊢ Eq μ.join (μ.bind id)
                                                                          -/
theorem join_eq_bind (μ : Measure (Measure α)) : join μ = bind μ id := by rw [bind, map_id]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem join_map_map {f : α → β} (hf : Measurable f) (μ : Measure (Measure α)) :
    join (map (map f) μ) = map f (join μ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    μ : MeasureTheory.Measure (MeasureTheory.Measure α)
    ⊢ Eq (MeasureTheory.Measure.map (MeasureTheory.Measure.map f) μ).join (Measure …
  -/
  ext1 s hs
  rw [join_apply hs, map_apply hf hs, join_apply (hf hs),
    lintegral_map (measurable_coe hs) (measurable_map f hf)]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    μ : MeasureTheory.Measure (MeasureTheory.Measure α)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral μ fun a => (MeasureTheory.Measure.map f a) s) (M …
  -/
  simp_rw [map_apply hf hs]
  /-
    🎉 no goals
  -/


theorem join_map_join (μ : Measure (Measure (Measure α))) : join (map join μ) = join (join μ) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure (MeasureTheory.Measure (MeasureTheory.Measure α))
    ⊢ Eq (MeasureTheory.Measure.map MeasureTheory.Measure.join μ).join μ.join.join
  -/
  show bind μ join = join (join μ)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure (MeasureTheory.Measure (MeasureTheory.Measure α))
    ⊢ Eq (μ.bind MeasureTheory.Measure.join) μ.join.join
  -/
  rw [join_eq_bind, join_eq_bind, bind_bind measurable_id measurable_id]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure (MeasureTheory.Measure (MeasureTheory.Measure α))
    ⊢ Eq (μ.bind MeasureTheory.Measure.join) (μ.bind fun a => (id a).bind id)
  -/
  apply congr_arg (bind μ)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure (MeasureTheory.Measure (MeasureTheory.Measure α))
    ⊢ Eq MeasureTheory.Measure.join fun a => (id a).bind id
  -/
  funext ν
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure (MeasureTheory.Measure (MeasureTheory.Measure α))
    ν : MeasureTheory.Measure (MeasureTheory.Measure α)
    ⊢ Eq ν.join ((id ν).bind id)
  -/
  exact join_eq_bind ν
  /-
    🎉 no goals
  -/


theorem join_map_dirac (μ : Measure α) : join (map dirac μ) = μ := bind_dirac


theorem join_dirac (μ : Measure α) : join (dirac μ) = μ :=
  (join_eq_bind (dirac μ)).trans (dirac_bind measurable_id _)


