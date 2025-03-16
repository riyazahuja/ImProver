/-- Essential supremum of `f` with respect to measure `μ`: the smallest `c : β` such that
`f x ≤ c` a.e. -/
def essSup {_ : MeasurableSpace α} (f : α → β) (μ : Measure α) :=
  (ae μ).limsup f


/-- Essential infimum of `f` with respect to measure `μ`: the greatest `c : β` such that
`c ≤ f x` a.e. -/
def essInf {_ : MeasurableSpace α} (f : α → β) (μ : Measure α) :=
  (ae μ).liminf f


theorem essSup_congr_ae {f g : α → β} (hfg : f =ᵐ[μ] g) : essSup f μ = essSup g μ :=
  limsup_congr hfg


theorem essInf_congr_ae {f g : α → β} (hfg : f =ᵐ[μ] g) : essInf f μ = essInf g μ :=
  @essSup_congr_ae α βᵒᵈ _ _ _ _ _ hfg


@[simp]
theorem essSup_const' [NeZero μ] (c : β) : essSup (fun _ : α => c) μ = c :=
  limsup_const _


@[simp]
theorem essInf_const' [NeZero μ] (c : β) : essInf (fun _ : α => c) μ = c :=
  liminf_const _


theorem essSup_const (c : β) (hμ : μ ≠ 0) : essSup (fun _ : α => c) μ = c :=
  have := NeZero.mk hμ; essSup_const' _


theorem essInf_const (c : β) (hμ : μ ≠ 0) : essInf (fun _ : α => c) μ = c :=
  have := NeZero.mk hμ; essInf_const' _


@[simp]
lemma essSup_smul_measure (hc : c ≠ 0) (f : α → β) : essSup f (c • μ) = essSup f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : ConditionallyCompleteLattice β
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : SMulWithZero R ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : NoZeroSMulDivisors R ENNReal
    c : R
    hc : Ne c 0
    f : α → β
    ⊢ Eq (essSup f (HSMul.hSMul c μ)) (essSup f μ)
  -/
  simp_rw [essSup, Measure.ae_smul_measure_eq hc]
  /-
    🎉 no goals
  -/


lemma essSup_eq_ciSup (hμ : ∀ a, μ {a} ≠ 0) (hf : BddAbove (Set.range f)) :
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  m : MeasurableSpace α
                                  μ : MeasureTheory.Measure α
                                  inst✝¹ : ConditionallyCompleteLattice β
                                  f : α → β
                                  inst✝ : Nonempty α
                                  hμ : ∀ (a : α), Ne (μ (Singleton.singleton a)) 0
                                  hf : BddAbove (Set.range f)
                                  ⊢ Eq (essSup f μ) (iSup fun a => f a)
                                -/
    essSup f μ = ⨆ a, f a := by rw [essSup, ae_eq_top.2 hμ, limsup_top_eq_ciSup hf]
                                /-
                                  🎉 no goals
                                -/


lemma essInf_eq_ciInf (hμ : ∀ a, μ {a} ≠ 0) (hf : BddBelow (Set.range f)) :
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  m : MeasurableSpace α
                                  μ : MeasureTheory.Measure α
                                  inst✝¹ : ConditionallyCompleteLattice β
                                  f : α → β
                                  inst✝ : Nonempty α
                                  hμ : ∀ (a : α), Ne (μ (Singleton.singleton a)) 0
                                  hf : BddBelow (Set.range f)
                                  ⊢ Eq (essInf f μ) (iInf fun a => f a)
                                -/
    essInf f μ = ⨅ a, f a := by rw [essInf, ae_eq_top.2 hμ, liminf_top_eq_ciInf hf]
                                /-
                                  🎉 no goals
                                -/


@[simp] lemma essSup_count_eq_ciSup (hf : BddAbove (Set.range f)) :
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        m : MeasurableSpace α
                                                        inst✝² : ConditionallyCompleteLattice β
                                                        f : α → β
                                                        inst✝¹ : Nonempty α
                                                        inst✝ : MeasurableSingletonClass α
                                                        hf : BddAbove (Set.range f)
                                                        ⊢ ∀ (a : α), Ne (MeasureTheory.Measure.count (Singleton.singleton a)) 0
                                                      -/
    essSup f .count = ⨆ a, f a := essSup_eq_ciSup (by simp) hf
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] lemma essInf_count_eq_ciInf (hf : BddBelow (Set.range f)) :
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        m : MeasurableSpace α
                                                        inst✝² : ConditionallyCompleteLattice β
                                                        f : α → β
                                                        inst✝¹ : Nonempty α
                                                        inst✝ : MeasurableSingletonClass α
                                                        hf : BddBelow (Set.range f)
                                                        ⊢ ∀ (a : α), Ne (MeasureTheory.Measure.count (Singleton.singleton a)) 0
                                                      -/
    essInf f .count = ⨅ a, f a := essInf_eq_ciInf (by simp) hf
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] lemma essSup_uniformOn_eq_ciSup [Finite α] (hf : BddAbove (Set.range f)) :
    essSup f (uniformOn univ) = ⨆ a, f a :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        m : MeasurableSpace α
                        inst✝³ : ConditionallyCompleteLattice β
                        f : α → β
                        inst✝² : Nonempty α
                        inst✝¹ : MeasurableSingletonClass α
                        inst✝ : Finite α
                        hf : BddAbove (Set.range f)
                        ⊢ ∀ (a : α), Ne ((ProbabilityTheory.uniformOn Set.univ) (Singleton.singleton a …
                      -/
  essSup_eq_ciSup (by simp [uniformOn, cond_apply, Set.finite_univ]) hf
                      /-
                        🎉 no goals
                      -/


@[simp] lemma essInf_cond_count_eq_ciInf [Finite α] (hf : BddBelow (Set.range f)) :
    essInf f (uniformOn univ) = ⨅ a, f a :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        m : MeasurableSpace α
                        inst✝³ : ConditionallyCompleteLattice β
                        f : α → β
                        inst✝² : Nonempty α
                        inst✝¹ : MeasurableSingletonClass α
                        inst✝ : Finite α
                        hf : BddBelow (Set.range f)
                        ⊢ ∀ (a : α), Ne ((ProbabilityTheory.uniformOn Set.univ) (Singleton.singleton a …
                      -/
  essInf_eq_ciInf (by simp [uniformOn, cond_apply, Set.finite_univ]) hf
                      /-
                        🎉 no goals
                      -/


theorem essSup_eq_sInf {m : MeasurableSpace α} (μ : Measure α) (f : α → β) :
    essSup f μ = sInf { a | μ { x | a < f x } = 0 } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Eq (essSup f μ) (InfSet.sInf (setOf fun a => Eq (μ (setOf fun x => LT.lt a ( …
  -/
  dsimp [essSup, limsup, limsSup]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Eq (InfSet.sInf (setOf fun a => Filter.Eventually (fun n => LE.le n a) (Filt …
  -/
  simp only [eventually_map, ae_iff, not_le]
  /-
    🎉 no goals
  -/


theorem essInf_eq_sSup {m : MeasurableSpace α} (μ : Measure α) (f : α → β) :
    essInf f μ = sSup { a | μ { x | f x < a } = 0 } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Eq (essInf f μ) (SupSet.sSup (setOf fun a => Eq (μ (setOf fun x => LT.lt (f  …
  -/
  dsimp [essInf, liminf, limsInf]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Eq (SupSet.sSup (setOf fun a => Filter.Eventually (fun n => LE.le a n) (Filt …
  -/
  simp only [eventually_map, ae_iff, not_le]
  /-
    🎉 no goals
  -/


theorem ae_lt_of_essSup_lt (hx : essSup f μ < x)
    (hf : IsBoundedUnder (· ≤ ·) (ae μ) f := by isBoundedDefault) :
    ∀ᵐ y ∂μ, f y < x :=
  eventually_lt_of_limsup_lt hx hf


theorem ae_lt_of_lt_essInf (hx : x < essInf f μ)
    (hf : IsBoundedUnder (· ≥ ·) (ae μ) f := by isBoundedDefault) :
    ∀ᵐ y ∂μ, x < f y :=
  eventually_lt_of_lt_liminf hx hf


theorem ae_le_essSup
    (hf : IsBoundedUnder (· ≤ ·) (ae μ) f := by isBoundedDefault) :
    ∀ᵐ y ∂μ, f y ≤ essSup f μ :=
  eventually_le_limsup hf


theorem ae_essInf_le
    (hf : IsBoundedUnder (· ≥ ·) (ae μ) f := by isBoundedDefault) :
    ∀ᵐ y ∂μ, essInf f μ ≤ f y :=
  eventually_liminf_le hf


theorem meas_essSup_lt
    (hf : IsBoundedUnder (· ≤ ·) (ae μ) f := by isBoundedDefault) :
    μ { y | essSup f μ < f y } = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : ConditionallyCompleteLinearOrder β
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : FirstCountableTopology β
    inst✝ : OrderTopology β
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheor …
    ⊢ Eq (μ (setOf fun y => LT.lt (essSup f μ) (f y))) 0
  -/
  simp_rw [← not_le]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : ConditionallyCompleteLinearOrder β
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : FirstCountableTopology β
    inst✝ : OrderTopology β
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheor …
    ⊢ Eq (μ (setOf fun y => Not (LE.le (f y) (essSup f μ)))) 0
  -/
  exact ae_le_essSup hf
  /-
    🎉 no goals
  -/


theorem meas_lt_essInf
    (hf : IsBoundedUnder (· ≥ ·) (ae μ) f := by isBoundedDefault) :
    μ { y | f y < essInf f μ } = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : ConditionallyCompleteLinearOrder β
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : FirstCountableTopology β
    inst✝ : OrderTopology β
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) (MeasureTheor …
    ⊢ Eq (μ (setOf fun y => LT.lt (f y) (essInf f μ))) 0
  -/
  simp_rw [← not_le]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : ConditionallyCompleteLinearOrder β
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : FirstCountableTopology β
    inst✝ : OrderTopology β
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) (MeasureTheor …
    ⊢ Eq (μ (setOf fun y => Not (LE.le (essInf f μ) (f y)))) 0
  -/
  exact ae_essInf_le hf
  /-
    🎉 no goals
  -/


@[simp]
theorem essSup_measure_zero {m : MeasurableSpace α} {f : α → β} : essSup f (0 : Measure α) = ⊥ :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝ : CompleteLattice β
                               m : MeasurableSpace α
                               f : α → β
                               ⊢ Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le n a) (Filte …
                             -/
  le_bot_iff.mp (sInf_le (by simp [Set.mem_setOf_eq, EventuallyLE, ae_iff]))
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem essInf_measure_zero {_ : MeasurableSpace α} {f : α → β} : essInf f (0 : Measure α) = ⊤ :=
  @essSup_measure_zero α βᵒᵈ _ _ _


theorem essSup_mono_ae {f g : α → β} (hfg : f ≤ᵐ[μ] g) : essSup f μ ≤ essSup g μ :=
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f g : α → β
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) f
  -/
  /-
    🎉 no goals
  -/
  limsup_le_limsup hfg
  /-
    🎉 no goals
  -/


theorem essInf_mono_ae {f g : α → β} (hfg : f ≤ᵐ[μ] g) : essInf f μ ≤ essInf g μ :=
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f g : α → β
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) (MeasureTheory.ae μ) f
  -/
  /-
    🎉 no goals
  -/
  liminf_le_liminf hfg
  /-
    🎉 no goals
  -/


theorem essSup_le_of_ae_le {f : α → β} (c : β) (hf : f ≤ᵐ[μ] fun _ => c) : essSup f μ ≤ c :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        m : MeasurableSpace α
                        μ : MeasureTheory.Measure α
                        inst✝ : CompleteLattice β
                        f : α → β
                        c : β
                        hf : (MeasureTheory.ae μ).EventuallyLE f fun x => c
                        ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) f
                      -/
  limsup_le_of_le (by isBoundedDefault) hf
                      /-
                        🎉 no goals
                      -/


theorem le_essInf_of_ae_le {f : α → β} (c : β) (hf : (fun _ => c) ≤ᵐ[μ] f) : c ≤ essInf f μ :=
  @essSup_le_of_ae_le α βᵒᵈ _ _ _ _ c hf


theorem essSup_const_bot : essSup (fun _ : α => (⊥ : β)) μ = (⊥ : β) :=
  limsup_const_bot


theorem essInf_const_top : essInf (fun _ : α => (⊤ : β)) μ = (⊤ : β) :=
  liminf_const_top


theorem OrderIso.essSup_apply {m : MeasurableSpace α} {γ} [CompleteLattice γ] (f : α → β)
    (μ : Measure α) (g : β ≃o γ) : g (essSup f μ) = essSup (fun x => g (f x)) μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice β
    m : MeasurableSpace α
    γ : Type u_3
    inst✝ : CompleteLattice γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : OrderIso β γ
    ⊢ Eq (g (essSup f μ)) (essSup (fun x => g (f x)) μ)
  -/
  refine OrderIso.limsup_apply g ?_ ?_ ?_ ?_
  /-
    case refine_1
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteLattice β
    m : MeasurableSpace α
    γ : Type u_3
    inst✝ : CompleteLattice γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : OrderIso β γ
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) f
  -/
  all_goals isBoundedDefault
  /-
    🎉 no goals
  -/


theorem OrderIso.essInf_apply {_ : MeasurableSpace α} {γ} [CompleteLattice γ] (f : α → β)
    (μ : Measure α) (g : β ≃o γ) : g (essInf f μ) = essInf (fun x => g (f x)) μ :=
  @OrderIso.essSup_apply α βᵒᵈ _ _ γᵒᵈ _ _ _ g.dual


theorem essSup_mono_measure {f : α → β} (hμν : ν ≪ μ) : essSup f ν ≤ essSup f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f : α → β
    hμν : ν.AbsolutelyContinuous μ
    ⊢ LE.le (essSup f ν) (essSup f μ)
  -/
  refine limsup_le_limsup_of_le (Measure.ae_le_iff_absolutelyContinuous.mpr hμν) ?_ ?_
  /-
    case refine_1
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f : α → β
    hμν : ν.AbsolutelyContinuous μ
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae ν) f
  -/
  all_goals isBoundedDefault
  /-
    🎉 no goals
  -/


theorem essSup_mono_measure' {α : Type*} {β : Type*} {_ : MeasurableSpace α}
    {μ ν : MeasureTheory.Measure α} [CompleteLattice β] {f : α → β} (hμν : ν ≤ μ) :
    essSup f ν ≤ essSup f μ :=
  essSup_mono_measure (Measure.absolutelyContinuous_of_le hμν)


theorem essInf_antitone_measure {f : α → β} (hμν : μ ≪ ν) : essInf f ν ≤ essInf f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f : α → β
    hμν : μ.AbsolutelyContinuous ν
    ⊢ LE.le (essInf f ν) (essInf f μ)
  -/
  refine liminf_le_liminf_of_le (Measure.ae_le_iff_absolutelyContinuous.mpr hμν) ?_ ?_
  /-
    case refine_1
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    f : α → β
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) (MeasureTheory.ae ν) f
  -/
  all_goals isBoundedDefault
  /-
    🎉 no goals
  -/


lemma essSup_eq_iSup (hμ : ∀ a, μ {a} ≠ 0) (f : α → β) : essSup f μ = ⨆ i, f i := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    hμ : ∀ (a : α), Ne (μ (Singleton.singleton a)) 0
    f : α → β
    ⊢ Eq (essSup f μ) (iSup fun i => f i)
  -/
  rw [essSup, ae_eq_top.2 hμ, limsup_top_eq_iSup]
  /-
    🎉 no goals
  -/


lemma essInf_eq_iInf (hμ : ∀ a, μ {a} ≠ 0) (f : α → β) : essInf f μ = ⨅ i, f i := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    hμ : ∀ (a : α), Ne (μ (Singleton.singleton a)) 0
    f : α → β
    ⊢ Eq (essInf f μ) (iInf fun i => f i)
  -/
  rw [essInf, ae_eq_top.2 hμ, liminf_top_eq_iInf]
  /-
    🎉 no goals
  -/


@[simp] lemma essSup_count [MeasurableSingletonClass α] (f : α → β) : essSup f .count = ⨆ i, f i :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       m : MeasurableSpace α
                       inst✝¹ : CompleteLattice β
                       inst✝ : MeasurableSingletonClass α
                       f : α → β
                       ⊢ ∀ (a : α), Ne (MeasureTheory.Measure.count (Singleton.singleton a)) 0
                     -/
  essSup_eq_iSup (by simp) _
                     /-
                       🎉 no goals
                     -/


@[simp] lemma essInf_count [MeasurableSingletonClass α] (f : α → β) : essInf f .count = ⨅ i, f i :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       m : MeasurableSpace α
                       inst✝¹ : CompleteLattice β
                       inst✝ : MeasurableSingletonClass α
                       f : α → β
                       ⊢ ∀ (a : α), Ne (MeasureTheory.Measure.count (Singleton.singleton a)) 0
                     -/
  essInf_eq_iInf (by simp) _
                     /-
                       🎉 no goals
                     -/


theorem essSup_comp_le_essSup_map_measure (hf : AEMeasurable f μ) :
    essSup (g ∘ f) μ ≤ essSup g (Measure.map f μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : AEMeasurable f μ
    ⊢ LE.le (essSup (Function.comp g f) μ) (essSup g (MeasureTheory.Measure.map f  …
  -/
  refine limsSup_le_limsSup_of_le ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : AEMeasurable f μ
    ⊢ LE.le (Filter.map (Function.comp g f) (MeasureTheory.ae μ)) (Filter.map g (M …
  -/
  rw [← Filter.map_map]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : AEMeasurable f μ
    ⊢ LE.le (Filter.map g (Filter.map f (MeasureTheory.ae μ))) (Filter.map g (Meas …
  -/
  exact Filter.map_mono (Measure.tendsto_ae_map hf)
  /-
    🎉 no goals
  -/


theorem MeasurableEmbedding.essSup_map_measure (hf : MeasurableEmbedding f) :
    essSup g (Measure.map f μ) = essSup (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : MeasurableEmbedding f
    ⊢ Eq (essSup g (MeasureTheory.Measure.map f μ)) (essSup (Function.comp g f) μ)
  -/
  refine le_antisymm ?_ (essSup_comp_le_essSup_map_measure hf.measurable.aemeasurable)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : MeasurableEmbedding f
    ⊢ LE.le (essSup g (MeasureTheory.Measure.map f μ)) (essSup (Function.comp g f) …
  -/
  refine limsSup_le_limsSup (by isBoundedDefault) (by isBoundedDefault) (fun c h_le => ?_)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : MeasurableEmbedding f
    c : β
    h_le : Filter.Eventually (fun n => LE.le n c) (Filter.map (Function.comp g f)  …
    ⊢ Filter.Eventually (fun n => LE.le n c) (Filter.map g (MeasureTheory.ae (Meas …
  -/
  rw [eventually_map] at h_le ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    hf : MeasurableEmbedding f
    c : β
    h_le : Filter.Eventually (fun a => LE.le (Function.comp g f a) c) (MeasureTheo …
    ⊢ Filter.Eventually (fun a => LE.le (g a) c) (MeasureTheory.ae (MeasureTheory. …
  -/
  exact hf.ae_map_iff.mpr h_le
  /-
    🎉 no goals
  -/


theorem essSup_map_measure_of_measurable (hg : Measurable g) (hf : AEMeasurable f μ) :
    essSup g (Measure.map f μ) = essSup (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : Measurable g
    hf : AEMeasurable f μ
    ⊢ Eq (essSup g (MeasureTheory.Measure.map f μ)) (essSup (Function.comp g f) μ)
  -/
  refine le_antisymm ?_ (essSup_comp_le_essSup_map_measure hf)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : Measurable g
    hf : AEMeasurable f μ
    ⊢ LE.le (essSup g (MeasureTheory.Measure.map f μ)) (essSup (Function.comp g f) …
  -/
  refine limsSup_le_limsSup (by isBoundedDefault) (by isBoundedDefault) (fun c h_le => ?_)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : Measurable g
    hf : AEMeasurable f μ
    c : β
    h_le : Filter.Eventually (fun n => LE.le n c) (Filter.map (Function.comp g f)  …
    ⊢ Filter.Eventually (fun n => LE.le n c) (Filter.map g (MeasureTheory.ae (Meas …
  -/
  rw [eventually_map] at h_le ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : Measurable g
    hf : AEMeasurable f μ
    c : β
    h_le : Filter.Eventually (fun a => LE.le (Function.comp g f a) c) (MeasureTheo …
    ⊢ Filter.Eventually (fun a => LE.le (g a) c) (MeasureTheory.ae (MeasureTheory. …
  -/
  rw [ae_map_iff hf (measurableSet_le hg measurable_const)]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : Measurable g
    hf : AEMeasurable f μ
    c : β
    h_le : Filter.Eventually (fun a => LE.le (Function.comp g f a) c) (MeasureTheo …
    ⊢ Filter.Eventually (fun x => LE.le (g (f x)) c) (MeasureTheory.ae μ)
  -/
  exact h_le
  /-
    🎉 no goals
  -/


theorem essSup_map_measure (hg : AEMeasurable g (Measure.map f μ)) (hf : AEMeasurable f μ) :
    essSup g (Measure.map f μ) = essSup (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Eq (essSup g (MeasureTheory.Measure.map f μ)) (essSup (Function.comp g f) μ)
  -/
  rw [essSup_congr_ae hg.ae_eq_mk, essSup_map_measure_of_measurable hg.measurable_mk hf]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Eq (essSup (Function.comp (AEMeasurable.mk g hg) f) μ) (essSup (Function.com …
  -/
  refine essSup_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Function.comp (AEMeasurable.mk g hg) f) ( …
  -/
  have h_eq := ae_of_ae_map hf hg.ae_eq_mk
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    h_eq : Filter.Eventually (fun x => Eq (g (f x)) (AEMeasurable.mk g hg (f x)))  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Function.comp (AEMeasurable.mk g hg) f) ( …
  -/
  rw [← EventuallyEq] at h_eq
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : CompleteLattice β
    γ : Type u_3
    mγ : MeasurableSpace γ
    f : α → γ
    g : γ → β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : OrderClosedTopology β
    inst✝ : OpensMeasurableSpace β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    h_eq : (MeasureTheory.ae μ).EventuallyEq (fun x => g (f x)) fun x => AEMeasura …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Function.comp (AEMeasurable.mk g hg) f) ( …
  -/
  exact h_eq.symm
  /-
    🎉 no goals
  -/


lemma essSup_piecewise {s : Set α} [DecidablePred (· ∈ s)] {g} (hs : MeasurableSet s) :
    essSup (s.piecewise f g) μ = max (essSup f (μ.restrict s)) (essSup g (μ.restrict sᶜ)) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : α → ENNReal
    hs : MeasurableSet s
    ⊢ Eq (essSup (s.piecewise f g) μ) (Max.max (essSup f (μ.restrict s)) (essSup g …
  -/
  simp only [essSup, limsup_piecewise, blimsup_eq_limsup, ae_restrict_eq, hs, hs.compl]; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem essSup_indicator_eq_essSup_restrict {s : Set α} {f : α → ℝ≥0∞} (hs : MeasurableSet s) :
    essSup (s.indicator f) μ = essSup f (μ.restrict s) := by
  classical
  simp only [← piecewise_eq_indicator, essSup_piecewise hs, max_eq_left_iff]
  exact limsup_const_bot.trans_le (zero_le _)


theorem ae_le_essSup (f : α → ℝ≥0∞) : ∀ᵐ y ∂μ, f y ≤ essSup f μ :=
  eventually_le_limsup f


@[simp]
theorem essSup_eq_zero_iff : essSup f μ = 0 ↔ f =ᵐ[μ] 0 :=
  limsup_eq_zero_iff


theorem essSup_const_mul {a : ℝ≥0∞} : essSup (fun x : α => a * f x) μ = a * essSup f μ :=
  limsup_const_mul


theorem essSup_mul_le (f g : α → ℝ≥0∞) : essSup (f * g) μ ≤ essSup f μ * essSup g μ :=
  limsup_mul_le f g


theorem essSup_add_le (f g : α → ℝ≥0∞) : essSup (f + g) μ ≤ essSup f μ + essSup g μ :=
  limsup_add_le f g


theorem essSup_liminf_le {ι} [Countable ι] [LinearOrder ι] (f : ι → α → ℝ≥0∞) :
    essSup (fun x => atTop.liminf fun n => f n x) μ ≤
      atTop.liminf fun n => essSup (fun x => f n x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_3
    inst✝¹ : Countable ι
    inst✝ : LinearOrder ι
    f : ι → α → ENNReal
    ⊢ LE.le (essSup (fun x => Filter.liminf (fun n => f n x) Filter.atTop) μ) (Fil …
  -/
  simp_rw [essSup]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_3
    inst✝¹ : Countable ι
    inst✝ : LinearOrder ι
    f : ι → α → ENNReal
    ⊢ LE.le (Filter.limsup (fun x => Filter.liminf (fun n => f n x) Filter.atTop)  …
  -/
  exact ENNReal.limsup_liminf_le_liminf_limsup fun a b => f b a
  /-
    🎉 no goals
  -/


theorem coe_essSup {f : α → ℝ≥0} (hf : IsBoundedUnder (· ≤ ·) (ae μ) f) :
    ((essSup f μ : ℝ≥0) : ℝ≥0∞) = essSup (fun x => (f x : ℝ≥0∞)) μ :=
  (ENNReal.coe_sInf <| hf).trans <|
    eq_of_forall_le_iff fun r => by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → NNReal
        hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (MeasureTheory.ae μ) f
        r : ENNReal
        ⊢ Iff (LE.le r (iInf fun a => iInf fun h => ↑a)) (LE.le r (essSup (fun x => ↑( …
      -/
      simp [essSup, limsup, limsSup, eventually_map, ENNReal.forall_ennreal]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma essSup_restrict_eq_of_support_subset {s : Set α} {f : α → ℝ≥0∞} (hsf : f.support ⊆ s) :
    essSup f (μ.restrict s) = essSup f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ Eq (essSup f (μ.restrict s)) (essSup f μ)
  -/
  apply le_antisymm (essSup_mono_measure' Measure.restrict_le_self)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ LE.le (essSup f μ) (essSup f (μ.restrict s))
  -/
  apply le_of_forall_lt (fun c hc ↦ ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    ⊢ LT.lt c (essSup f (μ.restrict s))
  -/
  obtain ⟨d, cd, hd⟩ : ∃ d, c < d ∧ d < essSup f μ := exists_between hc
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    ⊢ LT.lt c (essSup f (μ.restrict s))
  -/
  let t := {x | d < f x}
  have A : 0 < (μ.restrict t) t := by
    simp only [Measure.restrict_apply_self]
    rw [essSup_eq_sInf] at hd
    have : d ∉ {a | μ {x | a < f x} = 0} := not_mem_of_lt_csInf hd (OrderBot.bddBelow _)
    exact bot_lt_iff_ne_bot.2 this
  have B : 0 < (μ.restrict s) t := by
    have : μ.restrict t ≤ μ.restrict s := by
      apply Measure.restrict_mono _ le_rfl
      apply subset_trans _ hsf
      intro x (hx : d < f x)
      exact (lt_of_le_of_lt bot_le hx).ne'
    exact lt_of_lt_of_le A (this _)
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    t : Set α := setOf fun x => LT.lt d (f x)
    A : LT.lt 0 ((μ.restrict t) t)
    B : LT.lt 0 ((μ.restrict s) t)
    ⊢ LT.lt c (essSup f (μ.restrict s))
  -/
  apply cd.trans_le
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    t : Set α := setOf fun x => LT.lt d (f x)
    A : LT.lt 0 ((μ.restrict t) t)
    B : LT.lt 0 ((μ.restrict s) t)
    ⊢ LE.le d (essSup f (μ.restrict s))
  -/
  rw [essSup_eq_sInf]
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    t : Set α := setOf fun x => LT.lt d (f x)
    A : LT.lt 0 ((μ.restrict t) t)
    B : LT.lt 0 ((μ.restrict s) t)
    ⊢ LE.le d (InfSet.sInf (setOf fun a => Eq ((μ.restrict s) (setOf fun x => LT.l …
  -/
  apply le_sInf (fun b hb ↦ ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    t : Set α := setOf fun x => LT.lt d (f x)
    A : LT.lt 0 ((μ.restrict t) t)
    B : LT.lt 0 ((μ.restrict s) t)
    b : ENNReal
    hb : Membership.mem (setOf fun a => Eq ((μ.restrict s) (setOf fun x => LT.lt a …
    ⊢ LE.le d b
  -/
  contrapose! hb
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    c : ENNReal
    hc : LT.lt c (essSup f μ)
    d : ENNReal
    cd : LT.lt c d
    hd : LT.lt d (essSup f μ)
    t : Set α := setOf fun x => LT.lt d (f x)
    A : LT.lt 0 ((μ.restrict t) t)
    B : LT.lt 0 ((μ.restrict s) t)
    b : ENNReal
    hb : LT.lt b d
    ⊢ Not (Membership.mem (setOf fun a => Eq ((μ.restrict s) (setOf fun x => LT.lt …
  -/
  exact ne_of_gt (B.trans_le (measure_mono (fun x hx ↦ hb.trans hx)))
  /-
    🎉 no goals
  -/


