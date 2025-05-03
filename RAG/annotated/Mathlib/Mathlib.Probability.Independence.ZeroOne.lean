theorem Kernel.measure_eq_zero_or_one_or_top_of_indepSet_self {t : Set Ω}
    (h_indep : Kernel.IndepSet t t κ μα) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 ∨ κ a t = ∞ := by
  specialize h_indep t t (measurableSet_generateFrom (Set.mem_singleton t))
    (measurableSet_generateFrom (Set.mem_singleton t))
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    t : Set Ω
    h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
    ⊢ Filter.Eventually (fun a => Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ …
  -/
  filter_upwards [h_indep] with a ha
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    t : Set Ω
    h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
    a : α
    ha : Eq ((κ a) (Inter.inter t t)) (HMul.hMul ((κ a) t) ((κ a) t))
    ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
  -/
  by_cases h0 : κ a t = 0
    /-
      case pos
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      t : Set Ω
      h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
      a : α
      ha : Eq ((κ a) (Inter.inter t t)) (HMul.hMul ((κ a) t) ((κ a) t))
      h0 : Eq ((κ a) t) 0
      ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
    -/
  · exact Or.inl h0
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    t : Set Ω
    h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
    a : α
    ha : Eq ((κ a) (Inter.inter t t)) (HMul.hMul ((κ a) t) ((κ a) t))
    h0 : Not (Eq ((κ a) t) 0)
    ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
  -/
  by_cases h_top : κ a t = ∞
    /-
      case pos
      α : Type u_1
      Ω : Type u_2
      _mα : MeasurableSpace α
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      t : Set Ω
      h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
      a : α
      ha : Eq ((κ a) (Inter.inter t t)) (HMul.hMul ((κ a) t) ((κ a) t))
      h0 : Not (Eq ((κ a) t) 0)
      h_top : Eq ((κ a) t) Top.top
      ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
    -/
  · exact Or.inr (Or.inr h_top)
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    t : Set Ω
    h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
    a : α
    ha : Eq ((κ a) (Inter.inter t t)) (HMul.hMul ((κ a) t) ((κ a) t))
    h0 : Not (Eq ((κ a) t) 0)
    h_top : Not (Eq ((κ a) t) Top.top)
    ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
  -/
  rw [← one_mul (κ a (t ∩ t)), Set.inter_self, ENNReal.mul_eq_mul_right h0 h_top] at ha
  /-
    case neg
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    t : Set Ω
    h_indep : Filter.Eventually (fun a => Eq ((κ a) (Inter.inter t t)) (HMul.hMul  …
    a : α
    ha : Eq 1 ((κ a) t)
    h0 : Not (Eq ((κ a) t) 0)
    h_top : Not (Eq ((κ a) t) Top.top)
    ⊢ Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
  -/
  exact Or.inr (Or.inl ha.symm)
  /-
    🎉 no goals
  -/


theorem measure_eq_zero_or_one_or_top_of_indepSet_self {t : Set Ω}
    (h_indep : IndepSet t t μ) : μ t = 0 ∨ μ t = 1 ∨ μ t = ∞ := by
  simpa only [ae_dirac_eq, Filter.eventually_pure]
    using Kernel.measure_eq_zero_or_one_or_top_of_indepSet_self h_indep


theorem Kernel.measure_eq_zero_or_one_of_indepSet_self' (h : ∀ᵐ a ∂μα, IsFiniteMeasure (κ a))
    {t : Set Ω} (h_indep : IndepSet t t κ μα) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 := by
  /-
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => MeasureTheory.IsFiniteMeasure (κ a)) (MeasureT …
    t : Set Ω
    h_indep : ProbabilityTheory.Kernel.IndepSet t t κ μα
    ⊢ Filter.Eventually (fun a => Or (Eq ((κ a) t) 0) (Eq ((κ a) t) 1)) (MeasureTh …
  -/
  filter_upwards [measure_eq_zero_or_one_or_top_of_indepSet_self h_indep, h] with a h_0_1_top h'
  /-
    case h
    α : Type u_1
    Ω : Type u_2
    _mα : MeasurableSpace α
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => MeasureTheory.IsFiniteMeasure (κ a)) (MeasureT …
    t : Set Ω
    h_indep : ProbabilityTheory.Kernel.IndepSet t t κ μα
    a : α
    h_0_1_top : Or (Eq ((κ a) t) 0) (Or (Eq ((κ a) t) 1) (Eq ((κ a) t) Top.top))
    h' : MeasureTheory.IsFiniteMeasure (κ a)
    ⊢ Or (Eq ((κ a) t) 0) (Eq ((κ a) t) 1)
  -/
  simpa only [measure_ne_top (κ a), or_false] using h_0_1_top
  /-
    🎉 no goals
  -/


theorem Kernel.measure_eq_zero_or_one_of_indepSet_self [h : ∀ a, IsFiniteMeasure (κ a)] {t : Set Ω}
    (h_indep : IndepSet t t κ μα) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 :=
  Kernel.measure_eq_zero_or_one_of_indepSet_self' (ae_of_all μα h) h_indep


theorem measure_eq_zero_or_one_of_indepSet_self [IsFiniteMeasure μ] {t : Set Ω}
    (h_indep : IndepSet t t μ) : μ t = 0 ∨ μ t = 1 := by
  simpa only [ae_dirac_eq, Filter.eventually_pure]
    using Kernel.measure_eq_zero_or_one_of_indepSet_self h_indep


theorem condexp_eq_zero_or_one_of_condIndepSet_self
    [StandardBorelSpace Ω]
    (hm : m ≤ m0) [hμ : IsFiniteMeasure μ] {t : Set Ω} (ht : MeasurableSet t)
    (h_indep : CondIndepSet m hm t t μ) :
    ∀ᵐ ω ∂μ, (μ⟦t | m⟧) ω = 0 ∨ (μ⟦t | m⟧) ω = 1 := by
  -- TODO: Why is not inferred?
  /-
    Ω : Type u_2
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : StandardBorelSpace Ω
    hm : LE.le m m0
    hμ : MeasureTheory.IsFiniteMeasure μ
    t : Set Ω
    ht : MeasurableSet t
    h_indep : ProbabilityTheory.CondIndepSet m hm t t μ
    ⊢ Filter.Eventually (fun ω => Or (Eq (MeasureTheory.condexp m μ (t.indicator f …
  -/
  have (a) : IsFiniteMeasure (condexpKernel μ m a) := inferInstance
  /-
    Ω : Type u_2
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : StandardBorelSpace Ω
    hm : LE.le m m0
    hμ : MeasureTheory.IsFiniteMeasure μ
    t : Set Ω
    ht : MeasurableSet t
    h_indep : ProbabilityTheory.CondIndepSet m hm t t μ
    this : ∀ (a : Ω), MeasureTheory.IsFiniteMeasure ((ProbabilityTheory.condexpKer …
    ⊢ Filter.Eventually (fun ω => Or (Eq (MeasureTheory.condexp m μ (t.indicator f …
  -/
  have h := ae_of_ae_trim hm (Kernel.measure_eq_zero_or_one_of_indepSet_self h_indep)
  /-
    Ω : Type u_2
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : StandardBorelSpace Ω
    hm : LE.le m m0
    hμ : MeasureTheory.IsFiniteMeasure μ
    t : Set Ω
    ht : MeasurableSet t
    h_indep : ProbabilityTheory.CondIndepSet m hm t t μ
    this : ∀ (a : Ω), MeasureTheory.IsFiniteMeasure ((ProbabilityTheory.condexpKer …
    h : Filter.Eventually (fun x => Or (Eq (((ProbabilityTheory.condexpKernel μ m) …
    ⊢ Filter.Eventually (fun ω => Or (Eq (MeasureTheory.condexp m μ (t.indicator f …
  -/
  filter_upwards [condexpKernel_ae_eq_condexp hm ht, h] with ω hω_eq hω
  /-
    case h
    Ω : Type u_2
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : StandardBorelSpace Ω
    hm : LE.le m m0
    hμ : MeasureTheory.IsFiniteMeasure μ
    t : Set Ω
    ht : MeasurableSet t
    h_indep : ProbabilityTheory.CondIndepSet m hm t t μ
    this : ∀ (a : Ω), MeasureTheory.IsFiniteMeasure ((ProbabilityTheory.condexpKer …
    h : Filter.Eventually (fun x => Or (Eq (((ProbabilityTheory.condexpKernel μ m) …
    ω : Ω
    hω_eq : Eq (((ProbabilityTheory.condexpKernel μ m) ω) t).toReal (MeasureTheory …
    hω : Or (Eq (((ProbabilityTheory.condexpKernel μ m) ω) t) 0) (Eq (((Probabilit …
    ⊢ Or (Eq (MeasureTheory.condexp m μ (t.indicator fun ω => 1) ω) 0) (Eq (Measur …
  -/
  rw [← hω_eq, ENNReal.toReal_eq_zero_iff, ENNReal.toReal_eq_one_iff]
  cases hω with
  | inl h => exact Or.inl (Or.inl h)
  | inr h => exact Or.inr h


theorem Kernel.indep_biSup_compl (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα) (t : Set ι) :
    Indep (⨆ n ∈ t, s n) (⨆ n ∈ tᶜ, s n) κ μα :=
  indep_iSup_of_disjoint h_le h_indep disjoint_compl_right


theorem indep_biSup_compl (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) (t : Set ι) :
    Indep (⨆ n ∈ t, s n) (⨆ n ∈ tᶜ, s n) μ :=
  Kernel.indep_biSup_compl h_le h_indep t


theorem condIndep_biSup_compl [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) (t : Set ι) :
    CondIndep m (⨆ n ∈ t, s n) (⨆ n ∈ tᶜ, s n) hm μ :=
  Kernel.indep_biSup_compl h_le h_indep t


theorem Kernel.indep_biSup_limsup (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα)
    (hf : ∀ t, p t → tᶜ ∈ f) {t : Set ι} (ht : p t) :
    Indep (⨆ n ∈ t, s n) (limsup s f) κ μα := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    p : Set ι → Prop
    f : Filter ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    t : Set ι
    ht : p t
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun n => iSup fun h => s n) (Filter.lim …
  -/
  refine indep_of_indep_of_le_right (indep_biSup_compl h_le h_indep t) ?_
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    p : Set ι → Prop
    f : Filter ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    t : Set ι
    ht : p t
    ⊢ LE.le (Filter.limsup s f) (iSup fun n => iSup fun h => s n)
  -/
  refine limsSup_le_of_le (by isBoundedDefault) ?_
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    p : Set ι → Prop
    f : Filter ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    t : Set ι
    ht : p t
    ⊢ Filter.Eventually (fun n => LE.le n (iSup fun n => iSup fun h => s n)) (Filt …
  -/
  simp only [Set.mem_compl_iff, eventually_map]
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    p : Set ι → Prop
    f : Filter ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    t : Set ι
    ht : p t
    ⊢ Filter.Eventually (fun a => LE.le (s a) (iSup fun n => iSup fun x => s n)) f
  -/
  exact eventually_of_mem (hf t ht) le_iSup₂
  /-
    🎉 no goals
  -/


theorem indep_biSup_limsup
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    {t : Set ι} (ht : p t) :
    Indep (⨆ n ∈ t, s n) (limsup s f) μ :=
  Kernel.indep_biSup_limsup h_le h_indep hf ht


theorem condIndep_biSup_limsup [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    {t : Set ι} (ht : p t) :
    CondIndep m (⨆ n ∈ t, s n) (limsup s f) hm μ :=
  Kernel.indep_biSup_limsup h_le h_indep hf ht


theorem Kernel.indep_iSup_directed_limsup (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) :
    Indep (⨆ a, ⨆ n ∈ ns a, s n) (limsup s f) κ μα := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun a => iSup fun n => iSup fun h => s  …
  -/
  rcases eq_or_ne μα 0 with rfl | hμ
    /-
      case inl
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      h_indep : ProbabilityTheory.Kernel.iIndep s κ 0
      ⊢ ProbabilityTheory.Kernel.Indep (iSup fun a => iSup fun n => iSup fun h => s  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  obtain ⟨η, η_eq, hη⟩ : ∃ (η : Kernel α Ω), κ =ᵐ[μα] η ∧ IsMarkovKernel η :=
    exists_ae_eq_isMarkovKernel h_indep.ae_isProbabilityMeasure hμ
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hμ : Ne μα 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun a => iSup fun n => iSup fun h => s  …
  -/
  replace h_indep := h_indep.congr η_eq
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hμ : Ne μα 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    h_indep : ProbabilityTheory.Kernel.iIndep s η μα
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun a => iSup fun n => iSup fun h => s  …
  -/
  apply Indep.congr (Filter.EventuallyEq.symm η_eq)
  /-
    case inr.intro.intro
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hμ : Ne μα 0
    η : ProbabilityTheory.Kernel α Ω
    η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
    hη : ProbabilityTheory.IsMarkovKernel η
    h_indep : ProbabilityTheory.Kernel.iIndep s η μα
    ⊢ ProbabilityTheory.Kernel.Indep (iSup fun a => iSup fun n => iSup fun h => s  …
  -/
  apply indep_iSup_of_directed_le
    /-
      case inr.intro.intro.h_indep
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      ⊢ ∀ (i : β), ProbabilityTheory.Kernel.Indep (iSup fun n => iSup fun h => s n)  …
    -/
  · exact fun a => indep_biSup_limsup h_le h_indep hf (hnsp a)
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.h_le
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      ⊢ ∀ (i : β), LE.le (iSup fun n => iSup fun h => s n) m0
    -/
  · exact fun a => iSup₂_le fun n _ => h_le n
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.h_le'
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      ⊢ LE.le (Filter.limsup s f) m0
    -/
  · exact limsup_le_iSup.trans (iSup_le h_le)
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.hm
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      ⊢ Directed (fun x1 x2 => LE.le x1 x2) fun i => iSup fun n => iSup fun h => s n
    -/
  · intro a b
    /-
      case inr.intro.intro.hm
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      a b : β
      ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) ((fun i => iSup fun n => iSu …
    -/
    obtain ⟨c, hc⟩ := hns a b
    /-
      case inr.intro.intro.hm.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      β : Type u_4
      p : Set ι → Prop
      f : Filter ι
      ns : β → Set ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
      hns : Directed (fun x1 x2 => LE.le x1 x2) ns
      hnsp : ∀ (a : β), p (ns a)
      hμ : Ne μα 0
      η : ProbabilityTheory.Kernel α Ω
      η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
      hη : ProbabilityTheory.IsMarkovKernel η
      h_indep : ProbabilityTheory.Kernel.iIndep s η μα
      a b c : β
      hc : And ((fun x1 x2 => LE.le x1 x2) (ns a) (ns c)) ((fun x1 x2 => LE.le x1 x2 …
      ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) ((fun i => iSup fun n => iSu …
    -/
    refine ⟨c, ?_, ?_⟩ <;> refine iSup_mono fun n => iSup_mono' fun hn => ⟨?_, le_rfl⟩
      /-
        case inr.intro.intro.hm.intro.refine_1
        α : Type u_1
        Ω : Type u_2
        ι : Type u_3
        _mα : MeasurableSpace α
        s : ι → MeasurableSpace Ω
        m0 : MeasurableSpace Ω
        κ : ProbabilityTheory.Kernel α Ω
        μα : MeasureTheory.Measure α
        β : Type u_4
        p : Set ι → Prop
        f : Filter ι
        ns : β → Set ι
        h_le : ∀ (n : ι), LE.le (s n) m0
        hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
        hns : Directed (fun x1 x2 => LE.le x1 x2) ns
        hnsp : ∀ (a : β), p (ns a)
        hμ : Ne μα 0
        η : ProbabilityTheory.Kernel α Ω
        η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
        hη : ProbabilityTheory.IsMarkovKernel η
        h_indep : ProbabilityTheory.Kernel.iIndep s η μα
        a b c : β
        hc : And ((fun x1 x2 => LE.le x1 x2) (ns a) (ns c)) ((fun x1 x2 => LE.le x1 x2 …
        n : ι
        hn : Membership.mem (ns a) n
        ⊢ Membership.mem (ns c) n
      -/
    · exact hc.1 hn
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.hm.intro.refine_2
        α : Type u_1
        Ω : Type u_2
        ι : Type u_3
        _mα : MeasurableSpace α
        s : ι → MeasurableSpace Ω
        m0 : MeasurableSpace Ω
        κ : ProbabilityTheory.Kernel α Ω
        μα : MeasureTheory.Measure α
        β : Type u_4
        p : Set ι → Prop
        f : Filter ι
        ns : β → Set ι
        h_le : ∀ (n : ι), LE.le (s n) m0
        hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
        hns : Directed (fun x1 x2 => LE.le x1 x2) ns
        hnsp : ∀ (a : β), p (ns a)
        hμ : Ne μα 0
        η : ProbabilityTheory.Kernel α Ω
        η_eq : (MeasureTheory.ae μα).EventuallyEq ⇑κ ⇑η
        hη : ProbabilityTheory.IsMarkovKernel η
        h_indep : ProbabilityTheory.Kernel.iIndep s η μα
        a b c : β
        hc : And ((fun x1 x2 => LE.le x1 x2) (ns a) (ns c)) ((fun x1 x2 => LE.le x1 x2 …
        n : ι
        hn : Membership.mem (ns b) n
        ⊢ Membership.mem (ns c) n
      -/
    · exact hc.2 hn
      /-
        🎉 no goals
      -/


theorem indep_iSup_directed_limsup
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) :
    Indep (⨆ a, ⨆ n ∈ ns a, s n) (limsup s f) μ :=
  Kernel.indep_iSup_directed_limsup h_le h_indep hf hns hnsp


theorem condIndep_iSup_directed_limsup [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) :
    CondIndep m (⨆ a, ⨆ n ∈ ns a, s n) (limsup s f) hm μ :=
  Kernel.indep_iSup_directed_limsup h_le h_indep hf hns hnsp


theorem Kernel.indep_iSup_limsup (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα)
    (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    Indep (⨆ n, s n) (limsup s f) κ μα := by
  suffices (⨆ a, ⨆ n ∈ ns a, s n) = ⨆ n, s n by
    rw [← this]
    exact indep_iSup_directed_limsup h_le h_indep hf hns hnsp
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    ⊢ Eq (iSup fun a => iSup fun n => iSup fun h => s n) (iSup fun n => s n)
  -/
  rw [iSup_comm]
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    ⊢ Eq (iSup fun j => iSup fun i => iSup fun h => s j) (iSup fun n => s n)
  -/
  refine iSup_congr fun n => ?_
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    n : ι
    ⊢ Eq (iSup fun i => iSup fun h => s n) (s n)
  -/
  have h : ⨆ (i : β) (_ : n ∈ ns i), s n = ⨆ _ : ∃ i, n ∈ ns i, s n := by rw [iSup_exists]
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    n : ι
    h : Eq (iSup fun i => iSup fun x => s n) (iSup fun x => s n)
    ⊢ Eq (iSup fun i => iSup fun h => s n) (s n)
  -/
  haveI : Nonempty (∃ i : β, n ∈ ns i) := ⟨hns_univ n⟩
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    n : ι
    h : Eq (iSup fun i => iSup fun x => s n) (iSup fun x => s n)
    this : Nonempty (Exists fun i => Membership.mem (ns i) n)
    ⊢ Eq (iSup fun i => iSup fun h => s n) (s n)
  -/
  rw [h, iSup_const]
  /-
    🎉 no goals
  -/


theorem indep_iSup_limsup
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    Indep (⨆ n, s n) (limsup s f) μ :=
  Kernel.indep_iSup_limsup h_le h_indep hf hns hnsp hns_univ


theorem condIndep_iSup_limsup [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    CondIndep m (⨆ n, s n) (limsup s f) hm μ :=
  Kernel.indep_iSup_limsup h_le h_indep hf hns hnsp hns_univ


theorem Kernel.indep_limsup_self (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα)
    (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    Indep (limsup s f) (limsup s f) κ μα :=
  indep_of_indep_of_le_left (indep_iSup_limsup h_le h_indep hf hns hnsp hns_univ) limsup_le_iSup


theorem indep_limsup_self
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    Indep (limsup s f) (limsup s f) μ :=
  Kernel.indep_limsup_self h_le h_indep hf hns hnsp hns_univ


theorem condIndep_limsup_self [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) (hf : ∀ t, p t → tᶜ ∈ f)
    (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a)) (hns_univ : ∀ n, ∃ a, n ∈ ns a) :
    CondIndep m (limsup s f) (limsup s f) hm μ :=
  Kernel.indep_limsup_self h_le h_indep hf hns hnsp hns_univ


theorem Kernel.measure_zero_or_one_of_measurableSet_limsup (h_le : ∀ n, s n ≤ m0)
    (h_indep : iIndep s κ μα)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a))
    (hns_univ : ∀ n, ∃ a, n ∈ ns a) {t : Set Ω} (ht_tail : MeasurableSet[limsup s f] t) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 := by
  apply measure_eq_zero_or_one_of_indepSet_self' ?_
    ((indep_limsup_self h_le h_indep hf hns hnsp hns_univ).indepSet_of_measurableSet ht_tail
      ht_tail)
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    t : Set Ω
    ht_tail : MeasurableSet t
    ⊢ Filter.Eventually (fun a => MeasureTheory.IsFiniteMeasure (κ a)) (MeasureThe …
  -/
  filter_upwards [h_indep.ae_isProbabilityMeasure] with a ha using by infer_instance
  /-
    🎉 no goals
  -/


theorem measure_zero_or_one_of_measurableSet_limsup
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a))
    (hns_univ : ∀ n, ∃ a, n ∈ ns a) {t : Set Ω} (ht_tail : MeasurableSet[limsup s f] t) :
    μ t = 0 ∨ μ t = 1 := by
  simpa only [ae_dirac_eq, Filter.eventually_pure]
    using Kernel.measure_zero_or_one_of_measurableSet_limsup h_le h_indep hf hns hnsp hns_univ
      ht_tail


theorem condexp_zero_or_one_of_measurableSet_limsup [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ)
    (hf : ∀ t, p t → tᶜ ∈ f) (hns : Directed (· ≤ ·) ns) (hnsp : ∀ a, p (ns a))
    (hns_univ : ∀ n, ∃ a, n ∈ ns a) {t : Set Ω} (ht_tail : MeasurableSet[limsup s f] t) :
    ∀ᵐ ω ∂μ, (μ⟦t | m⟧) ω = 0 ∨ (μ⟦t | m⟧) ω = 1 := by
  have h := ae_of_ae_trim hm
    (Kernel.measure_zero_or_one_of_measurableSet_limsup h_le h_indep hf hns hnsp hns_univ ht_tail)
  /-
    Ω : Type u_2
    ι : Type u_3
    s : ι → MeasurableSpace Ω
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    inst✝¹ : StandardBorelSpace Ω
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.iCondIndep m hm s μ
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    t : Set Ω
    ht_tail : MeasurableSet t
    h : Filter.Eventually (fun x => Or (Eq (((ProbabilityTheory.condexpKernel μ m) …
    ⊢ Filter.Eventually (fun ω => Or (Eq (MeasureTheory.condexp m μ (t.indicator f …
  -/
  have ht : MeasurableSet t := limsup_le_iSup.trans (iSup_le h_le) t ht_tail
  /-
    Ω : Type u_2
    ι : Type u_3
    s : ι → MeasurableSpace Ω
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    inst✝¹ : StandardBorelSpace Ω
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.iCondIndep m hm s μ
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    t : Set Ω
    ht_tail : MeasurableSet t
    h : Filter.Eventually (fun x => Or (Eq (((ProbabilityTheory.condexpKernel μ m) …
    ht : MeasurableSet t
    ⊢ Filter.Eventually (fun ω => Or (Eq (MeasureTheory.condexp m μ (t.indicator f …
  -/
  filter_upwards [condexpKernel_ae_eq_condexp hm ht, h] with ω hω_eq hω
  /-
    case h
    Ω : Type u_2
    ι : Type u_3
    s : ι → MeasurableSpace Ω
    m m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    β : Type u_4
    p : Set ι → Prop
    f : Filter ι
    ns : β → Set ι
    inst✝¹ : StandardBorelSpace Ω
    hm : LE.le m m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.iCondIndep m hm s μ
    hf : ∀ (t : Set ι), p t → Membership.mem f (HasCompl.compl t)
    hns : Directed (fun x1 x2 => LE.le x1 x2) ns
    hnsp : ∀ (a : β), p (ns a)
    hns_univ : ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    t : Set Ω
    ht_tail : MeasurableSet t
    h : Filter.Eventually (fun x => Or (Eq (((ProbabilityTheory.condexpKernel μ m) …
    ht : MeasurableSet t
    ω : Ω
    hω_eq : Eq (((ProbabilityTheory.condexpKernel μ m) ω) t).toReal (MeasureTheory …
    hω : Or (Eq (((ProbabilityTheory.condexpKernel μ m) ω) t) 0) (Eq (((Probabilit …
    ⊢ Or (Eq (MeasureTheory.condexp m μ (t.indicator fun ω => 1) ω) 0) (Eq (Measur …
  -/
  rw [← hω_eq, ENNReal.toReal_eq_zero_iff, ENNReal.toReal_eq_one_iff]
  cases hω with
  | inl h => exact Or.inl (Or.inl h)
  | inr h => exact Or.inr h


theorem Kernel.indep_limsup_atTop_self (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα) :
    Indep (limsup s atTop) (limsup s atTop) κ μα := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeSup ι
    inst✝¹ : NoMaxOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atTop) (Filter.limsup …
  -/
  let ns : ι → Set ι := Set.Iic
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeSup ι
    inst✝¹ : NoMaxOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ns : ι → Set ι := Set.Iic
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atTop) (Filter.limsup …
  -/
  have hnsp : ∀ i, BddAbove (ns i) := fun i => bddAbove_Iic
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeSup ι
    inst✝¹ : NoMaxOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ns : ι → Set ι := Set.Iic
    hnsp : ∀ (i : ι), BddAbove (ns i)
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atTop) (Filter.limsup …
  -/
  refine indep_limsup_self h_le h_indep ?_ ?_ hnsp ?_
    /-
      case refine_1
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      ⊢ ∀ (t : Set ι), BddAbove t → Membership.mem Filter.atTop (HasCompl.compl t)
    -/
  · simp only [mem_atTop_sets, Set.mem_compl_iff, BddAbove, upperBounds, Set.Nonempty]
    /-
      case refine_1
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      ⊢ ∀ (t : Set ι), (Exists fun x => Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Me …
    -/
    rintro t ⟨a, ha⟩
    /-
      case refine_1.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le a x) a
      ⊢ Exists fun a => ∀ (b : ι), GE.ge b a → Not (Membership.mem t b)
    -/
    obtain ⟨b, hb⟩ : ∃ b, a < b := exists_gt a
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le a x) a
      b : ι
      hb : LT.lt a b
      ⊢ Exists fun a => ∀ (b : ι), GE.ge b a → Not (Membership.mem t b)
    -/
    refine ⟨b, fun c hc hct => ?_⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le a x) a
      b : ι
      hb : LT.lt a b
      c : ι
      hc : GE.ge c b
      hct : Membership.mem t c
      ⊢ False
    -/
    suffices ∀ i ∈ t, i < c from lt_irrefl c (this c hct)
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le a x) a
      b : ι
      hb : LT.lt a b
      c : ι
      hc : GE.ge c b
      hct : Membership.mem t c
      ⊢ ∀ (i : ι), Membership.mem t i → LT.lt i c
    -/
    exact fun i hi => (ha hi).trans_lt (hb.trans_le hc)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      ⊢ Directed (fun x1 x2 => LE.le x1 x2) ns
    -/
  · exact Monotone.directed_le fun i j hij k hki => le_trans hki hij
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeSup ι
      inst✝¹ : NoMaxOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Iic
      hnsp : ∀ (i : ι), BddAbove (ns i)
      ⊢ ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    -/
  · exact fun n => ⟨n, le_rfl⟩
    /-
      🎉 no goals
    -/


theorem indep_limsup_atTop_self (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) :
    Indep (limsup s atTop) (limsup s atTop) μ :=
  Kernel.indep_limsup_atTop_self h_le h_indep


theorem condIndep_limsup_atTop_self [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) :
    CondIndep m (limsup s atTop) (limsup s atTop) hm μ :=
  Kernel.indep_limsup_atTop_self h_le h_indep


theorem Kernel.measure_zero_or_one_of_measurableSet_limsup_atTop (h_le : ∀ n, s n ≤ m0)
    (h_indep : iIndep s κ μα) {t : Set Ω} (ht_tail : MeasurableSet[limsup s atTop] t) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 := by
  apply measure_eq_zero_or_one_of_indepSet_self' ?_
    ((indep_limsup_atTop_self h_le h_indep).indepSet_of_measurableSet ht_tail ht_tail)
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeSup ι
    inst✝¹ : NoMaxOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    t : Set Ω
    ht_tail : MeasurableSet t
    ⊢ Filter.Eventually (fun a => MeasureTheory.IsFiniteMeasure (κ a)) (MeasureThe …
  -/
  filter_upwards [h_indep.ae_isProbabilityMeasure] with a ha using by infer_instance
  /-
    🎉 no goals
  -/


/-- **Kolmogorov's 0-1 law** : any event in the tail σ-algebra of an independent sequence of
sub-σ-algebras has probability 0 or 1.
The tail σ-algebra `limsup s atTop` is the same as `⋂ n, ⋃ i ≥ n, s i`. -/
theorem measure_zero_or_one_of_measurableSet_limsup_atTop
    (h_le : ∀ n, s n ≤ m0)
    (h_indep : iIndep s μ) {t : Set Ω} (ht_tail : MeasurableSet[limsup s atTop] t) :
    μ t = 0 ∨ μ t = 1 := by
  simpa only [ae_dirac_eq, Filter.eventually_pure]
    using Kernel.measure_zero_or_one_of_measurableSet_limsup_atTop h_le h_indep ht_tail


theorem condexp_zero_or_one_of_measurableSet_limsup_atTop [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ] (h_le : ∀ n, s n ≤ m0)
    (h_indep : iCondIndep m hm s μ) {t : Set Ω} (ht_tail : MeasurableSet[limsup s atTop] t) :
    ∀ᵐ ω ∂μ, (μ⟦t | m⟧) ω = 0 ∨ (μ⟦t | m⟧) ω = 1 :=
  condexp_eq_zero_or_one_of_condIndepSet_self hm (limsup_le_iSup.trans (iSup_le h_le) t ht_tail)
    ((condIndep_limsup_atTop_self hm h_le h_indep).condIndepSet_of_measurableSet ht_tail ht_tail)


theorem Kernel.indep_limsup_atBot_self (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s κ μα) :
    Indep (limsup s atBot) (limsup s atBot) κ μα := by
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeInf ι
    inst✝¹ : NoMinOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atBot) (Filter.limsup …
  -/
  let ns : ι → Set ι := Set.Ici
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeInf ι
    inst✝¹ : NoMinOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ns : ι → Set ι := Set.Ici
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atBot) (Filter.limsup …
  -/
  have hnsp : ∀ i, BddBelow (ns i) := fun i => bddBelow_Ici
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeInf ι
    inst✝¹ : NoMinOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    ns : ι → Set ι := Set.Ici
    hnsp : ∀ (i : ι), BddBelow (ns i)
    ⊢ ProbabilityTheory.Kernel.Indep (Filter.limsup s Filter.atBot) (Filter.limsup …
  -/
  refine indep_limsup_self h_le h_indep ?_ ?_ hnsp ?_
    /-
      case refine_1
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      ⊢ ∀ (t : Set ι), BddBelow t → Membership.mem Filter.atBot (HasCompl.compl t)
    -/
  · simp only [mem_atBot_sets, Set.mem_compl_iff, BddBelow, lowerBounds, Set.Nonempty]
    /-
      case refine_1
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      ⊢ ∀ (t : Set ι), (Exists fun x => Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Me …
    -/
    rintro t ⟨a, ha⟩
    /-
      case refine_1.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le x a) a
      ⊢ Exists fun a => ∀ (b : ι), LE.le b a → Not (Membership.mem t b)
    -/
    obtain ⟨b, hb⟩ : ∃ b, b < a := exists_lt a
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le x a) a
      b : ι
      hb : LT.lt b a
      ⊢ Exists fun a => ∀ (b : ι), LE.le b a → Not (Membership.mem t b)
    -/
    refine ⟨b, fun c hc hct => ?_⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le x a) a
      b : ι
      hb : LT.lt b a
      c : ι
      hc : LE.le c b
      hct : Membership.mem t c
      ⊢ False
    -/
    suffices ∀ i ∈ t, c < i from lt_irrefl c (this c hct)
    /-
      case refine_1.intro.intro
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      t : Set ι
      a : ι
      ha : Membership.mem (setOf fun x => ∀ ⦃a : ι⦄, Membership.mem t a → LE.le x a) a
      b : ι
      hb : LT.lt b a
      c : ι
      hc : LE.le c b
      hct : Membership.mem t c
      ⊢ ∀ (i : ι), Membership.mem t i → LT.lt c i
    -/
    exact fun i hi => hc.trans_lt (hb.trans_le (ha hi))
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      ⊢ Directed (fun x1 x2 => LE.le x1 x2) ns
    -/
  · exact Antitone.directed_le fun _ _ ↦ Set.Ici_subset_Ici.2
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      Ω : Type u_2
      ι : Type u_3
      _mα : MeasurableSpace α
      s : ι → MeasurableSpace Ω
      m0 : MeasurableSpace Ω
      κ : ProbabilityTheory.Kernel α Ω
      μα : MeasureTheory.Measure α
      inst✝² : SemilatticeInf ι
      inst✝¹ : NoMinOrder ι
      inst✝ : Nonempty ι
      h_le : ∀ (n : ι), LE.le (s n) m0
      h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
      ns : ι → Set ι := Set.Ici
      hnsp : ∀ (i : ι), BddBelow (ns i)
      ⊢ ∀ (n : ι), Exists fun a => Membership.mem (ns a) n
    -/
  · exact fun n => ⟨n, le_rfl⟩
    /-
      🎉 no goals
    -/


theorem indep_limsup_atBot_self
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) :
    Indep (limsup s atBot) (limsup s atBot) μ :=
  Kernel.indep_limsup_atBot_self h_le h_indep


theorem condIndep_limsup_atBot_self [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ]
    (h_le : ∀ n, s n ≤ m0) (h_indep : iCondIndep m hm s μ) :
    CondIndep m (limsup s atBot) (limsup s atBot) hm μ :=
  Kernel.indep_limsup_atBot_self h_le h_indep


/-- **Kolmogorov's 0-1 law**, kernel version: any event in the tail σ-algebra of an independent
sequence of sub-σ-algebras has probability 0 or 1 almost surely. -/
theorem Kernel.measure_zero_or_one_of_measurableSet_limsup_atBot (h_le : ∀ n, s n ≤ m0)
    (h_indep : iIndep s κ μα) {t : Set Ω} (ht_tail : MeasurableSet[limsup s atBot] t) :
    ∀ᵐ a ∂μα, κ a t = 0 ∨ κ a t = 1 := by
  apply measure_eq_zero_or_one_of_indepSet_self' ?_
    ((indep_limsup_atBot_self h_le h_indep).indepSet_of_measurableSet ht_tail ht_tail)
  /-
    α : Type u_1
    Ω : Type u_2
    ι : Type u_3
    _mα : MeasurableSpace α
    s : ι → MeasurableSpace Ω
    m0 : MeasurableSpace Ω
    κ : ProbabilityTheory.Kernel α Ω
    μα : MeasureTheory.Measure α
    inst✝² : SemilatticeInf ι
    inst✝¹ : NoMinOrder ι
    inst✝ : Nonempty ι
    h_le : ∀ (n : ι), LE.le (s n) m0
    h_indep : ProbabilityTheory.Kernel.iIndep s κ μα
    t : Set Ω
    ht_tail : MeasurableSet t
    ⊢ Filter.Eventually (fun a => MeasureTheory.IsFiniteMeasure (κ a)) (MeasureThe …
  -/
  filter_upwards [h_indep.ae_isProbabilityMeasure] with a ha using by infer_instance
  /-
    🎉 no goals
  -/


/-- **Kolmogorov's 0-1 law** : any event in the tail σ-algebra of an independent sequence of
sub-σ-algebras has probability 0 or 1. -/
theorem measure_zero_or_one_of_measurableSet_limsup_atBot
    (h_le : ∀ n, s n ≤ m0) (h_indep : iIndep s μ) {t : Set Ω}
    (ht_tail : MeasurableSet[limsup s atBot] t) :
    μ t = 0 ∨ μ t = 1 := by
  simpa only [ae_dirac_eq, Filter.eventually_pure]
    using Kernel.measure_zero_or_one_of_measurableSet_limsup_atBot h_le h_indep ht_tail


/-- **Kolmogorov's 0-1 law**, conditional version: any event in the tail σ-algebra of a
conditionally independent sequence of sub-σ-algebras has conditional probability 0 or 1. -/
theorem condexp_zero_or_one_of_measurableSet_limsup_atBot [StandardBorelSpace Ω]
    (hm : m ≤ m0) [IsFiniteMeasure μ] (h_le : ∀ n, s n ≤ m0)
    (h_indep : iCondIndep m hm s μ) {t : Set Ω} (ht_tail : MeasurableSet[limsup s atBot] t) :
    ∀ᵐ ω ∂μ, (μ⟦t | m⟧) ω = 0 ∨ (μ⟦t | m⟧) ω = 1 :=
  condexp_eq_zero_or_one_of_condIndepSet_self hm (limsup_le_iSup.trans (iSup_le h_le) t ht_tail)
    ((condIndep_limsup_atBot_self hm h_le h_indep).condIndepSet_of_measurableSet ht_tail ht_tail)


