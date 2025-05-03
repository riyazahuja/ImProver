@[nontriviality, measurability]
theorem Subsingleton.aemeasurable [Subsingleton α] : AEMeasurable f μ :=
  Subsingleton.measurable.aemeasurable


@[nontriviality, measurability]
theorem aemeasurable_of_subsingleton_codomain [Subsingleton β] : AEMeasurable f μ :=
  (measurable_of_subsingleton_codomain f).aemeasurable


@[simp, measurability]
theorem aemeasurable_zero_measure : AEMeasurable f (0 : Measure α) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    ⊢ AEMeasurable f 0
  -/
  nontriviality α; inhabit α
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    ⊢ AEMeasurable f 0
  -/
  exact ⟨fun _ => f default, measurable_const, rfl⟩
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem aemeasurable_id'' (μ : Measure α) {m : MeasurableSpace α} (hm : m ≤ m0) :
    @AEMeasurable α α m m0 id μ :=
  @Measurable.aemeasurable α α m0 m id μ (measurable_id'' hm)


lemma aemeasurable_of_map_neZero {μ : Measure α}
    {f : α → β} (h : NeZero (μ.map f)) :
    AEMeasurable f μ := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    h : NeZero (MeasureTheory.Measure.map f μ)
    ⊢ AEMeasurable f μ
  -/
  by_contra h'
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    h : NeZero (MeasureTheory.Measure.map f μ)
    h' : Not (AEMeasurable f μ)
    ⊢ False
  -/
  simp [h'] at h
  /-
    🎉 no goals
  -/


lemma mono_ac (hf : AEMeasurable f ν) (hμν : μ ≪ ν) : AEMeasurable f μ :=
  ⟨hf.mk f, hf.measurable_mk, hμν.ae_le hf.ae_eq_mk⟩


theorem mono_measure (h : AEMeasurable f μ) (h' : ν ≤ μ) : AEMeasurable f ν :=
  mono_ac h h'.absolutelyContinuous


theorem mono_set {s t} (h : s ⊆ t) (ht : AEMeasurable f (μ.restrict t)) :
    AEMeasurable f (μ.restrict s) :=
  ht.mono_measure (restrict_mono h le_rfl)


@[fun_prop]
protected theorem mono' (h : AEMeasurable f μ) (h' : ν ≪ μ) : AEMeasurable f ν :=
  ⟨h.mk f, h.measurable_mk, h' h.ae_eq_mk⟩


theorem ae_mem_imp_eq_mk {s} (h : AEMeasurable f (μ.restrict s)) :
    ∀ᵐ x ∂μ, x ∈ s → f x = h.mk f x :=
  ae_imp_of_ae_restrict h.ae_eq_mk


theorem ae_inf_principal_eq_mk {s} (h : AEMeasurable f (μ.restrict s)) : f =ᶠ[ae μ ⊓ 𝓟 s] h.mk f :=
  le_ae_restrict h.ae_eq_mk


@[measurability]
theorem sum_measure [Countable ι] {μ : ι → Measure α} (h : ∀ i, AEMeasurable f (μ i)) :
    AEMeasurable f (sum μ) := by
  classical
  nontriviality β
  inhabit β
  set s : ι → Set α := fun i => toMeasurable (μ i) { x | f x ≠ (h i).mk f x }
  have hsμ : ∀ i, μ i (s i) = 0 := by
    intro i
    rw [measure_toMeasurable]
    exact (h i).ae_eq_mk
  have hsm : MeasurableSet (⋂ i, s i) :=
    MeasurableSet.iInter fun i => measurableSet_toMeasurable _ _
  have hs : ∀ i x, x ∉ s i → f x = (h i).mk f x := by
    intro i x hx
    contrapose! hx
    exact subset_toMeasurable _ _ hx
  set g : α → β := (⋂ i, s i).piecewise (const α default) f
  refine ⟨g, measurable_of_restrict_of_restrict_compl hsm ?_ ?_, ae_sum_iff.mpr fun i => ?_⟩
  · rw [restrict_piecewise]
    simp only [s, Set.restrict, const]
    exact measurable_const
  · rw [restrict_piecewise_compl, compl_iInter]
    intro t ht
    refine ⟨⋃ i, (h i).mk f ⁻¹' t ∩ (s i)ᶜ, MeasurableSet.iUnion fun i ↦
      (measurable_mk _ ht).inter (measurableSet_toMeasurable _ _).compl, ?_⟩
    ext ⟨x, hx⟩
    simp only [mem_preimage, mem_iUnion, Subtype.coe_mk, Set.restrict, mem_inter_iff,
      mem_compl_iff] at hx ⊢
    constructor
    · rintro ⟨i, hxt, hxs⟩
      rwa [hs _ _ hxs]
    · rcases hx with ⟨i, hi⟩
      rw [hs _ _ hi]
      exact fun h => ⟨i, h, hi⟩
  · refine measure_mono_null (fun x (hx : f x ≠ g x) => ?_) (hsμ i)
    contrapose! hx
    refine (piecewise_eq_of_not_mem _ _ _ ?_).symm
    exact fun h => hx (mem_iInter.1 h i)


@[simp]
theorem _root_.aemeasurable_sum_measure_iff [Countable ι] {μ : ι → Measure α} :
    AEMeasurable f (sum μ) ↔ ∀ i, AEMeasurable f (μ i) :=
  ⟨fun h _ => h.mono_measure (le_sum _ _), sum_measure⟩


@[simp]
theorem _root_.aemeasurable_add_measure_iff :
    AEMeasurable f (μ + ν) ↔ AEMeasurable f μ ∧ AEMeasurable f ν := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ ν : MeasureTheory.Measure α
    ⊢ Iff (AEMeasurable f (HAdd.hAdd μ ν)) (And (AEMeasurable f μ) (AEMeasurable f …
  -/
  rw [← sum_cond, aemeasurable_sum_measure_iff, Bool.forall_bool, and_comm]
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ ν : MeasureTheory.Measure α
    ⊢ Iff (And (AEMeasurable f (cond Bool.true μ ν)) (AEMeasurable f (cond Bool.fa …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[measurability]
theorem add_measure {f : α → β} (hμ : AEMeasurable f μ) (hν : AEMeasurable f ν) :
    AEMeasurable f (μ + ν) :=
  aemeasurable_add_measure_iff.2 ⟨hμ, hν⟩


@[measurability]
protected theorem iUnion [Countable ι] {s : ι → Set α}
    (h : ∀ i, AEMeasurable f (μ.restrict (s i))) : AEMeasurable f (μ.restrict (⋃ i, s i)) :=
  (sum_measure h).mono_measure <| restrict_iUnion_le


@[simp]
theorem _root_.aemeasurable_iUnion_iff [Countable ι] {s : ι → Set α} :
    AEMeasurable f (μ.restrict (⋃ i, s i)) ↔ ∀ i, AEMeasurable f (μ.restrict (s i)) :=
  ⟨fun h _ => h.mono_measure <| restrict_mono (subset_iUnion _ _) le_rfl, AEMeasurable.iUnion⟩


@[simp]
theorem _root_.aemeasurable_union_iff {s t : Set α} :
    AEMeasurable f (μ.restrict (s ∪ t)) ↔
      AEMeasurable f (μ.restrict s) ∧ AEMeasurable f (μ.restrict t) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    s t : Set α
    ⊢ Iff (AEMeasurable f (μ.restrict (Union.union s t))) (And (AEMeasurable f (μ. …
  -/
  simp only [union_eq_iUnion, aemeasurable_iUnion_iff, Bool.forall_bool, cond, and_comm]
  /-
    🎉 no goals
  -/


@[measurability]
theorem smul_measure [Monoid R] [DistribMulAction R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (h : AEMeasurable f μ) (c : R) : AEMeasurable f (c • μ) :=
  ⟨h.mk f, h.measurable_mk, ae_smul_measure h.ae_eq_mk c⟩


theorem comp_aemeasurable {f : α → δ} {g : δ → β} (hg : AEMeasurable g (μ.map f))
    (hf : AEMeasurable f μ) : AEMeasurable (g ∘ f) μ :=
  ⟨hg.mk g ∘ hf.mk f, hg.measurable_mk.comp hf.measurable_mk,
    (ae_eq_comp hf hg.ae_eq_mk).trans (hf.ae_eq_mk.fun_comp (mk g hg))⟩


@[fun_prop]
theorem comp_aemeasurable' {f : α → δ} {g : δ → β} (hg : AEMeasurable g (μ.map f))
    (hf : AEMeasurable f μ) : AEMeasurable (fun x ↦ g (f x)) μ := comp_aemeasurable hg hf


theorem comp_measurable {f : α → δ} {g : δ → β} (hg : AEMeasurable g (μ.map f))
    (hf : Measurable f) : AEMeasurable (g ∘ f) μ :=
  hg.comp_aemeasurable hf.aemeasurable


theorem comp_quasiMeasurePreserving {ν : Measure δ} {f : α → δ} {g : δ → β} (hg : AEMeasurable g ν)
    (hf : QuasiMeasurePreserving f μ ν) : AEMeasurable (g ∘ f) μ :=
  (hg.mono' hf.absolutelyContinuous).comp_measurable hf.measurable


theorem map_map_of_aemeasurable {g : β → γ} {f : α → β} (hg : AEMeasurable g (Measure.map f μ))
    (hf : AEMeasurable f μ) : (μ.map f).map g = μ.map (g ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    g : β → γ
    f : α → β
    hg : AEMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.Measure.map g (MeasureTheory.Measure.map f μ)) (MeasureThe …
  -/
  ext1 s hs
  rw [map_apply_of_aemeasurable hg hs, map_apply₀ hf (hg.nullMeasurable hs),
    map_apply_of_aemeasurable (hg.comp_aemeasurable hf) hs, preimage_comp]


@[fun_prop, measurability]
theorem prod_mk {f : α → β} {g : α → γ} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (fun x => (f x, g x)) μ :=
  ⟨fun a => (hf.mk f a, hg.mk g a), hf.measurable_mk.prod_mk hg.measurable_mk,
    EventuallyEq.prod_mk hf.ae_eq_mk hg.ae_eq_mk⟩


theorem exists_ae_eq_range_subset (H : AEMeasurable f μ) {t : Set β} (ht : ∀ᵐ x ∂μ, f x ∈ t)
    (h₀ : t.Nonempty) : ∃ g, Measurable g ∧ range g ⊆ t ∧ f =ᵐ[μ] g := by
  classical
  let s : Set α := toMeasurable μ { x | f x = H.mk f x ∧ f x ∈ t }ᶜ
  let g : α → β := piecewise s (fun _ => h₀.some) (H.mk f)
  refine ⟨g, ?_, ?_, ?_⟩
  · exact Measurable.piecewise (measurableSet_toMeasurable _ _) measurable_const H.measurable_mk
  · rintro _ ⟨x, rfl⟩
    by_cases hx : x ∈ s
    · simpa [g, hx] using h₀.some_mem
    · simp only [g, hx, piecewise_eq_of_not_mem, not_false_iff]
      contrapose! hx
      apply subset_toMeasurable
      simp +contextual only [hx, mem_compl_iff, mem_setOf_eq, not_and,
        not_false_iff, imp_true_iff]
  · have A : μ (toMeasurable μ { x | f x = H.mk f x ∧ f x ∈ t }ᶜ) = 0 := by
      rw [measure_toMeasurable, ← compl_mem_ae_iff, compl_compl]
      exact H.ae_eq_mk.and ht
    filter_upwards [compl_mem_ae_iff.2 A] with x hx
    rw [mem_compl_iff] at hx
    simp only [s, g, hx, piecewise_eq_of_not_mem, not_false_iff]
    contrapose! hx
    apply subset_toMeasurable
    simp only [hx, mem_compl_iff, mem_setOf_eq, false_and, not_false_iff]


theorem exists_measurable_nonneg {β} [Preorder β] [Zero β] {mβ : MeasurableSpace β} {f : α → β}
    (hf : AEMeasurable f μ) (f_nn : ∀ᵐ t ∂μ, 0 ≤ f t) : ∃ g, Measurable g ∧ 0 ≤ g ∧ f =ᵐ[μ] g := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝¹ : Preorder β
    inst✝ : Zero β
    mβ : MeasurableSpace β
    f : α → β
    hf : AEMeasurable f μ
    f_nn : Filter.Eventually (fun t => LE.le 0 (f t)) (MeasureTheory.ae μ)
    ⊢ Exists fun g => And (Measurable g) (And (LE.le 0 g) ((MeasureTheory.ae μ).Ev …
  -/
  obtain ⟨G, hG_meas, hG_mem, hG_ae_eq⟩ := hf.exists_ae_eq_range_subset f_nn ⟨0, le_rfl⟩
  /-
    case intro.intro.intro
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝¹ : Preorder β
    inst✝ : Zero β
    mβ : MeasurableSpace β
    f : α → β
    hf : AEMeasurable f μ
    f_nn : Filter.Eventually (fun t => LE.le 0 (f t)) (MeasureTheory.ae μ)
    G : α → β
    hG_meas : Measurable G
    hG_mem : HasSubset.Subset (Set.range G) (Preorder.toLE.1 0)
    hG_ae_eq : (MeasureTheory.ae μ).EventuallyEq f G
    ⊢ Exists fun g => And (Measurable g) (And (LE.le 0 g) ((MeasureTheory.ae μ).Ev …
  -/
  exact ⟨G, hG_meas, fun x => hG_mem (mem_range_self x), hG_ae_eq⟩
  /-
    🎉 no goals
  -/


theorem subtype_mk (h : AEMeasurable f μ) {s : Set β} {hfs : ∀ x, f x ∈ s} :
    AEMeasurable (codRestrict f s hfs) μ := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    h : AEMeasurable f μ
    s : Set β
    hfs : ∀ (x : α), Membership.mem s (f x)
    ⊢ AEMeasurable (Set.codRestrict f s hfs) μ
  -/
  nontriviality α; inhabit α
  obtain ⟨g, g_meas, hg, fg⟩ : ∃ g : α → β, Measurable g ∧ range g ⊆ s ∧ f =ᵐ[μ] g :=
    h.exists_ae_eq_range_subset (Eventually.of_forall hfs) ⟨_, hfs default⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    h : AEMeasurable f μ
    s : Set β
    hfs : ∀ (x : α), Membership.mem s (f x)
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    g : α → β
    g_meas : Measurable g
    hg : HasSubset.Subset (Set.range g) s
    fg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ AEMeasurable (Set.codRestrict f s hfs) μ
  -/
  refine ⟨codRestrict g s fun x => hg (mem_range_self _), Measurable.subtype_mk g_meas, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    h : AEMeasurable f μ
    s : Set β
    hfs : ∀ (x : α), Membership.mem s (f x)
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    g : α → β
    g_meas : Measurable g
    hg : HasSubset.Subset (Set.range g) s
    fg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.codRestrict f s hfs) (Set.codRestrict …
  -/
  filter_upwards [fg] with x hx
  /-
    case h
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    h : AEMeasurable f μ
    s : Set β
    hfs : ∀ (x : α), Membership.mem s (f x)
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    g : α → β
    g_meas : Measurable g
    hg : HasSubset.Subset (Set.range g) s
    fg : (MeasureTheory.ae μ).EventuallyEq f g
    x : α
    hx : Eq (f x) (g x)
    ⊢ Eq (Set.codRestrict f s hfs x) (Set.codRestrict g s ⋯ x)
  -/
  simpa [Subtype.ext_iff]
  /-
    🎉 no goals
  -/


theorem aemeasurable_const' (h : ∀ᵐ (x) (y) ∂μ, f x = f y) : AEMeasurable f μ := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    h : Filter.Eventually (fun x => Filter.Eventually (fun y => Eq (f x) (f y)) (M …
    ⊢ AEMeasurable f μ
  -/
  rcases eq_or_ne μ 0 with (rfl | hμ)
    /-
      case inl
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : α → β
      h : Filter.Eventually (fun x => Filter.Eventually (fun y => Eq (f x) (f y)) (M …
      ⊢ AEMeasurable f 0
    -/
  · exact aemeasurable_zero_measure
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      h : Filter.Eventually (fun x => Filter.Eventually (fun y => Eq (f x) (f y)) (M …
      hμ : Ne μ 0
      ⊢ AEMeasurable f μ
    -/
  · haveI := ae_neBot.2 hμ
    /-
      case inr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      h : Filter.Eventually (fun x => Filter.Eventually (fun y => Eq (f x) (f y)) (M …
      hμ : Ne μ 0
      this : (MeasureTheory.ae μ).NeBot
      ⊢ AEMeasurable f μ
    -/
    rcases h.exists with ⟨x, hx⟩
    /-
      case inr.intro
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      h : Filter.Eventually (fun x => Filter.Eventually (fun y => Eq (f x) (f y)) (M …
      hμ : Ne μ 0
      this : (MeasureTheory.ae μ).NeBot
      x : α
      hx : Filter.Eventually (fun y => Eq (f x) (f y)) (MeasureTheory.ae μ)
      ⊢ AEMeasurable f μ
    -/
    exact ⟨const α (f x), measurable_const, EventuallyEq.symm hx⟩
    /-
      🎉 no goals
    -/


theorem aemeasurable_uIoc_iff [LinearOrder α] {f : α → β} {a b : α} :
    (AEMeasurable f <| μ.restrict <| Ι a b) ↔
      (AEMeasurable f <| μ.restrict <| Ioc a b) ∧ (AEMeasurable f <| μ.restrict <| Ioc b a) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder α
    f : α → β
    a b : α
    ⊢ Iff (AEMeasurable f (μ.restrict (Set.uIoc a b))) (And (AEMeasurable f (μ.res …
  -/
  rw [uIoc_eq_union, aemeasurable_union_iff]
  /-
    🎉 no goals
  -/


theorem aemeasurable_iff_measurable [μ.IsComplete] : AEMeasurable f μ ↔ Measurable f :=
  ⟨fun h => h.nullMeasurable.measurable_of_complete, fun h => h.aemeasurable⟩


theorem MeasurableEmbedding.aemeasurable_map_iff {g : β → γ} (hf : MeasurableEmbedding f) :
    AEMeasurable g (μ.map f) ↔ AEMeasurable (g ∘ f) μ := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : β → γ
    hf : MeasurableEmbedding f
    ⊢ Iff (AEMeasurable g (MeasureTheory.Measure.map f μ)) (AEMeasurable (Function …
  -/
  refine ⟨fun H => H.comp_measurable hf.measurable, ?_⟩
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : β → γ
    hf : MeasurableEmbedding f
    ⊢ AEMeasurable (Function.comp g f) μ → AEMeasurable g (MeasureTheory.Measure.m …
  -/
  rintro ⟨g₁, hgm₁, heq⟩
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : β → γ
    hf : MeasurableEmbedding f
    g₁ : α → γ
    hgm₁ : Measurable g₁
    heq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g₁
    ⊢ AEMeasurable g (MeasureTheory.Measure.map f μ)
  -/
  rcases hf.exists_measurable_extend hgm₁ fun x => ⟨g x⟩ with ⟨g₂, hgm₂, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    μ : MeasureTheory.Measure α
    g : β → γ
    hf : MeasurableEmbedding f
    g₂ : β → γ
    hgm₂ : Measurable g₂
    hgm₁ : Measurable (Function.comp g₂ f)
    heq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) (Function.comp g₂ f)
    ⊢ AEMeasurable g (MeasureTheory.Measure.map f μ)
  -/
  exact ⟨g₂, hgm₂, hf.ae_map_iff.2 heq⟩
  /-
    🎉 no goals
  -/


theorem MeasurableEmbedding.aemeasurable_comp_iff {g : β → γ} (hg : MeasurableEmbedding g)
    {μ : Measure α} : AEMeasurable (g ∘ f) μ ↔ AEMeasurable f μ := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    g : β → γ
    hg : MeasurableEmbedding g
    μ : MeasureTheory.Measure α
    ⊢ Iff (AEMeasurable (Function.comp g f) μ) (AEMeasurable f μ)
  -/
  refine ⟨fun H => ?_, hg.measurable.comp_aemeasurable⟩
  suffices AEMeasurable ((rangeSplitting g ∘ rangeFactorization g) ∘ f) μ by
    rwa [(rightInverse_rangeSplitting hg.injective).comp_eq_id] at this
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    g : β → γ
    hg : MeasurableEmbedding g
    μ : MeasureTheory.Measure α
    H : AEMeasurable (Function.comp g f) μ
    ⊢ AEMeasurable (Function.comp (Function.comp (Set.rangeSplitting g) (Set.range …
  -/
  exact hg.measurable_rangeSplitting.comp_aemeasurable H.subtype_mk
  /-
    🎉 no goals
  -/


theorem aemeasurable_restrict_iff_comap_subtype {s : Set α} (hs : MeasurableSet s) {μ : Measure α}
    {f : α → β} : AEMeasurable f (μ.restrict s) ↔ AEMeasurable (f ∘ (↑) : s → β) (comap (↑) μ) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    s : Set α
    hs : MeasurableSet s
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Iff (AEMeasurable f (μ.restrict s)) (AEMeasurable (Function.comp f Subtype.v …
  -/
  rw [← map_comap_subtype_coe hs, (MeasurableEmbedding.subtype_coe hs).aemeasurable_map_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem aemeasurable_one [One β] : AEMeasurable (fun _ : α => (1 : β)) μ :=
  measurable_one.aemeasurable


@[simp]
theorem aemeasurable_smul_measure_iff {c : ℝ≥0∞} (hc : c ≠ 0) :
    AEMeasurable f (c • μ) ↔ AEMeasurable f μ :=
  ⟨fun h => ⟨h.mk f, h.measurable_mk, (ae_smul_measure_iff hc).1 h.ae_eq_mk⟩, fun h =>
    ⟨h.mk f, h.measurable_mk, (ae_smul_measure_iff hc).2 h.ae_eq_mk⟩⟩


theorem aemeasurable_of_aemeasurable_trim {α} {m m0 : MeasurableSpace α} {μ : Measure α}
    (hm : m ≤ m0) {f : α → β} (hf : AEMeasurable f (μ.trim hm)) : AEMeasurable f μ :=
  ⟨hf.mk f, Measurable.mono hf.measurable_mk hm le_rfl, ae_eq_of_ae_eq_trim hf.ae_eq_mk⟩


theorem aemeasurable_restrict_of_measurable_subtype {s : Set α} (hs : MeasurableSet s)
    (hf : Measurable fun x : s => f x) : AEMeasurable f (μ.restrict s) :=
  (aemeasurable_restrict_iff_comap_subtype hs).2 hf.aemeasurable


theorem aemeasurable_map_equiv_iff (e : α ≃ᵐ β) {f : β → γ} :
    AEMeasurable f (μ.map e) ↔ AEMeasurable (f ∘ e) μ :=
  e.measurableEmbedding.aemeasurable_map_iff


theorem AEMeasurable.restrict (hfm : AEMeasurable f μ) {s} : AEMeasurable f (μ.restrict s) :=
  ⟨AEMeasurable.mk f hfm, hfm.measurable_mk, ae_restrict_of_ae hfm.ae_eq_mk⟩


theorem aemeasurable_Ioi_of_forall_Ioc {β} {mβ : MeasurableSpace β} [LinearOrder α]
    [(atTop : Filter α).IsCountablyGenerated] {x : α} {g : α → β}
    (g_meas : ∀ t > x, AEMeasurable g (μ.restrict (Ioc x t))) :
    AEMeasurable g (μ.restrict (Ioi x)) := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    mβ : MeasurableSpace β
    inst✝¹ : LinearOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    x : α
    g : α → β
    g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
    ⊢ AEMeasurable g (μ.restrict (Set.Ioi x))
  -/
  haveI : Nonempty α := ⟨x⟩
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    mβ : MeasurableSpace β
    inst✝¹ : LinearOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    x : α
    g : α → β
    g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
    this : Nonempty α
    ⊢ AEMeasurable g (μ.restrict (Set.Ioi x))
  -/
  obtain ⟨u, hu_tendsto⟩ := exists_seq_tendsto (atTop : Filter α)
  have Ioi_eq_iUnion : Ioi x = ⋃ n : ℕ, Ioc x (u n) := by
    rw [iUnion_Ioc_eq_Ioi_self_iff.mpr _]
    exact fun y _ => (hu_tendsto.eventually (eventually_ge_atTop y)).exists
  /-
    case intro
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    mβ : MeasurableSpace β
    inst✝¹ : LinearOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    x : α
    g : α → β
    g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
    this : Nonempty α
    u : Nat → α
    hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
    Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
    ⊢ AEMeasurable g (μ.restrict (Set.Ioi x))
  -/
  rw [Ioi_eq_iUnion, aemeasurable_iUnion_iff]
  /-
    case intro
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    mβ : MeasurableSpace β
    inst✝¹ : LinearOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    x : α
    g : α → β
    g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
    this : Nonempty α
    u : Nat → α
    hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
    Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
    ⊢ ∀ (i : Nat), AEMeasurable g (μ.restrict (Set.Ioc x (u i)))
  -/
  intro n
  /-
    case intro
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    mβ : MeasurableSpace β
    inst✝¹ : LinearOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    x : α
    g : α → β
    g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
    this : Nonempty α
    u : Nat → α
    hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
    Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
    n : Nat
    ⊢ AEMeasurable g (μ.restrict (Set.Ioc x (u n)))
  -/
  cases' lt_or_le x (u n) with h h
    /-
      case intro.inl
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      mβ : MeasurableSpace β
      inst✝¹ : LinearOrder α
      inst✝ : Filter.atTop.IsCountablyGenerated
      x : α
      g : α → β
      g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
      this : Nonempty α
      u : Nat → α
      hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
      Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
      n : Nat
      h : LT.lt x (u n)
      ⊢ AEMeasurable g (μ.restrict (Set.Ioc x (u n)))
    -/
  · exact g_meas (u n) h
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      mβ : MeasurableSpace β
      inst✝¹ : LinearOrder α
      inst✝ : Filter.atTop.IsCountablyGenerated
      x : α
      g : α → β
      g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
      this : Nonempty α
      u : Nat → α
      hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
      Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
      n : Nat
      h : LE.le (u n) x
      ⊢ AEMeasurable g (μ.restrict (Set.Ioc x (u n)))
    -/
  · rw [Ioc_eq_empty (not_lt.mpr h), Measure.restrict_empty]
    /-
      case intro.inr
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      mβ : MeasurableSpace β
      inst✝¹ : LinearOrder α
      inst✝ : Filter.atTop.IsCountablyGenerated
      x : α
      g : α → β
      g_meas : ∀ (t : α), GT.gt t x → AEMeasurable g (μ.restrict (Set.Ioc x t))
      this : Nonempty α
      u : Nat → α
      hu_tendsto : Filter.Tendsto u Filter.atTop Filter.atTop
      Ioi_eq_iUnion : Eq (Set.Ioi x) (Set.iUnion fun n => Set.Ioc x (u n))
      n : Nat
      h : LE.le (u n) x
      ⊢ AEMeasurable g 0
    -/
    exact aemeasurable_zero_measure
    /-
      🎉 no goals
    -/


theorem aemeasurable_indicator_iff {s} (hs : MeasurableSet s) :
    AEMeasurable (indicator s f) μ ↔ AEMeasurable f (μ.restrict s) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    inst✝ : Zero β
    s : Set α
    hs : MeasurableSet s
    ⊢ Iff (AEMeasurable (s.indicator f) μ) (AEMeasurable f (μ.restrict s))
  -/
  constructor
    /-
      case mp
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      ⊢ AEMeasurable (s.indicator f) μ → AEMeasurable f (μ.restrict s)
    -/
  · intro h
    /-
      case mp
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : AEMeasurable (s.indicator f) μ
      ⊢ AEMeasurable f (μ.restrict s)
    -/
    exact (h.mono_measure Measure.restrict_le_self).congr (indicator_ae_eq_restrict hs)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      ⊢ AEMeasurable f (μ.restrict s) → AEMeasurable (s.indicator f) μ
    -/
  · intro h
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : AEMeasurable f (μ.restrict s)
      ⊢ AEMeasurable (s.indicator f) μ
    -/
    refine ⟨indicator s (h.mk f), h.measurable_mk.indicator hs, ?_⟩
    have A : s.indicator f =ᵐ[μ.restrict s] s.indicator (AEMeasurable.mk f h) :=
      (indicator_ae_eq_restrict hs).trans (h.ae_eq_mk.trans <| (indicator_ae_eq_restrict hs).symm)
    have B : s.indicator f =ᵐ[μ.restrict sᶜ] s.indicator (AEMeasurable.mk f h) :=
      (indicator_ae_eq_restrict_compl hs).trans (indicator_ae_eq_restrict_compl hs).symm
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure α
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : AEMeasurable f (μ.restrict s)
      A : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (s.indicator f) (s.indicato …
      B : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq (s.indicat …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) (s.indicator (AEMeasurable …
    -/
    exact ae_of_ae_restrict_of_ae_restrict_compl _ A B
    /-
      🎉 no goals
    -/


theorem aemeasurable_indicator_iff₀ {s} (hs : NullMeasurableSet s μ) :
    AEMeasurable (indicator s f) μ ↔ AEMeasurable f (μ.restrict s) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure α
    inst✝ : Zero β
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Iff (AEMeasurable (s.indicator f) μ) (AEMeasurable f (μ.restrict s))
  -/
  rcases hs with ⟨t, ht, hst⟩
  rw [← aemeasurable_congr (indicator_ae_eq_of_ae_eq_set hst.symm), aemeasurable_indicator_iff ht,
      restrict_congr_set hst]


/-- A characterization of the a.e.-measurability of the indicator function which takes a constant
value `b` on a set `A` and `0` elsewhere. -/
lemma aemeasurable_indicator_const_iff {s} [MeasurableSingletonClass β] (b : β) [NeZero b] :
    AEMeasurable (s.indicator (fun _ ↦ b)) μ ↔ NullMeasurableSet s μ := by
  classical
  constructor <;> intro h
  · convert h.nullMeasurable (MeasurableSet.singleton (0 : β)).compl
    rw [indicator_const_preimage_eq_union s {0}ᶜ b]
    simp [NeZero.ne b]
  · exact (aemeasurable_indicator_iff₀ h).mpr aemeasurable_const


@[measurability]
theorem AEMeasurable.indicator (hfm : AEMeasurable f μ) {s} (hs : MeasurableSet s) :
    AEMeasurable (s.indicator f) μ :=
  (aemeasurable_indicator_iff hs).mpr hfm.restrict


theorem AEMeasurable.indicator₀ (hfm : AEMeasurable f μ) {s} (hs : NullMeasurableSet s μ) :
    AEMeasurable (s.indicator f) μ :=
  (aemeasurable_indicator_iff₀ hs).mpr hfm.restrict


theorem MeasureTheory.Measure.restrict_map_of_aemeasurable {f : α → δ} (hf : AEMeasurable f μ)
    {s : Set δ} (hs : MeasurableSet s) : (μ.map f).restrict s = (μ.restrict <| f ⁻¹' s).map f :=
  calc
    (μ.map f).restrict s = (μ.map (hf.mk f)).restrict s := by
      /-
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq ((MeasureTheory.Measure.map f μ).restrict s) ((MeasureTheory.Measure.map  …
      -/
      congr 1
      /-
        case e_μ
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map (AEMeasurable. …
      -/
      apply Measure.map_congr hf.ae_eq_mk
      /-
        🎉 no goals
      -/
    _ = (μ.restrict <| hf.mk f ⁻¹' s).map (hf.mk f) := Measure.restrict_map hf.measurable_mk hs
    _ = (μ.restrict <| hf.mk f ⁻¹' s).map f :=
      (Measure.map_congr (ae_restrict_of_ae hf.ae_eq_mk.symm))
    _ = (μ.restrict <| f ⁻¹' s).map f := by
      /-
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq (MeasureTheory.Measure.map f (μ.restrict (Set.preimage (AEMeasurable.mk f …
      -/
      apply congr_arg
      /-
        case h
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        ⊢ Eq (μ.restrict (Set.preimage (AEMeasurable.mk f hf) s)) (μ.restrict (Set.pre …
      -/
      ext1 t ht
      /-
        case h.h
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        t : Set α
        ht : MeasurableSet t
        ⊢ Eq ((μ.restrict (Set.preimage (AEMeasurable.mk f hf) s)) t) ((μ.restrict (Se …
      -/
      simp only [ht, Measure.restrict_apply]
      /-
        case h.h
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        t : Set α
        ht : MeasurableSet t
        ⊢ Eq (μ (Inter.inter t (Set.preimage (AEMeasurable.mk f hf) s))) (μ (Inter.int …
      -/
      apply measure_congr
      /-
        case h.h.H
        α : Type u_2
        δ : Type u_5
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        f : α → δ
        hf : AEMeasurable f μ
        s : Set δ
        hs : MeasurableSet s
        t : Set α
        ht : MeasurableSet t
        ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter t (Set.preimage (AEMeasurable …
      -/
      apply (EventuallyEq.refl _ _).inter (hf.ae_eq_mk.symm.preimage s)
      /-
        🎉 no goals
      -/


theorem MeasureTheory.Measure.map_mono_of_aemeasurable {f : α → δ} (h : μ ≤ ν)
    (hf : AEMeasurable f ν) : μ.map f ≤ ν.map f :=
                         /-
                           α : Type u_2
                           δ : Type u_5
                           m0 : MeasurableSpace α
                           inst✝ : MeasurableSpace δ
                           μ ν : MeasureTheory.Measure α
                           f : α → δ
                           h : LE.le μ ν
                           hf : AEMeasurable f ν
                           s : Set δ
                           hs : MeasurableSet s
                           ⊢ LE.le ((MeasureTheory.Measure.map f μ) s) ((MeasureTheory.Measure.map f ν) s)
                         -/
  le_iff.2 fun s hs ↦ by simpa [hf, hs, hf.mono_measure h] using h (f ⁻¹' s)
                         /-
                           🎉 no goals
                         -/


/-- If the `σ`-algebra of the codomain of a null measurable function is countably generated,
then the function is a.e.-measurable. -/
lemma MeasureTheory.NullMeasurable.aemeasurable {f : α → β}
    [hc : MeasurableSpace.CountablyGenerated β] (h : NullMeasurable f μ) : AEMeasurable f μ := by
  classical
  nontriviality β; inhabit β
  rcases hc.1 with ⟨S, hSc, rfl⟩
  choose! T hTf hTm hTeq using fun s hs ↦ (h <| .basic s hs).exists_measurable_subset_ae_eq
  choose! U hUf hUm hUeq using fun s hs ↦ (h <| .basic s hs).exists_measurable_superset_ae_eq
  set v := ⋃ s ∈ S, U s \ T s
  have hvm : MeasurableSet v := .biUnion hSc fun s hs ↦ (hUm s hs).diff (hTm s hs)
  have hvμ : μ v = 0 := (measure_biUnion_null_iff hSc).2 fun s hs ↦ ae_le_set.1 <|
    ((hUeq s hs).trans (hTeq s hs).symm).le
  refine ⟨v.piecewise (fun _ ↦ default) f, ?_, measure_mono_null (fun x ↦
    not_imp_comm.2 fun hxv ↦ (piecewise_eq_of_not_mem _ _ _ hxv).symm) hvμ⟩
  refine measurable_of_restrict_of_restrict_compl hvm ?_ ?_
  · rw [restrict_piecewise]
    apply measurable_const
  · rw [restrict_piecewise_compl, restrict_eq]
    refine measurable_generateFrom fun s hs ↦ .of_subtype_image ?_
    rw [preimage_comp, Subtype.image_preimage_coe]
    convert (hTm s hs).diff hvm using 1
    rw [inter_comm]
    refine Set.ext fun x ↦ and_congr_left fun hxv ↦ ⟨fun hx ↦ ?_, fun hx ↦ hTf s hs hx⟩
    exact by_contra fun hx' ↦ hxv <| mem_biUnion hs ⟨hUf s hs hx, hx'⟩


/-- Let `f : α → β` be a null measurable function
such that a.e. all values of `f` belong to a set `t`
such that the restriction of the `σ`-algebra in the codomain to `t` is countably generated,
then `f` is a.e.-measurable. -/
lemma MeasureTheory.NullMeasurable.aemeasurable_of_aerange {f : α → β} {t : Set β}
    [MeasurableSpace.CountablyGenerated t] (h : NullMeasurable f μ) (hft : ∀ᵐ x ∂μ, f x ∈ t) :
    AEMeasurable f μ := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    t : Set β
    inst✝ : MeasurableSpace.CountablyGenerated ↑t
    h : MeasureTheory.NullMeasurable f μ
    hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ AEMeasurable f μ
  -/
  rcases eq_empty_or_nonempty t with rfl | hne
    /-
      case inl
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      h : MeasureTheory.NullMeasurable f μ
      inst✝ : MeasurableSpace.CountablyGenerated ↑EmptyCollection.emptyCollection
      hft : Filter.Eventually (fun x => Membership.mem EmptyCollection.emptyCollecti …
      ⊢ AEMeasurable f μ
    -/
  · obtain rfl : μ = 0 := by simpa using hft
    /-
      case inl
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      f : α → β
      inst✝ : MeasurableSpace.CountablyGenerated ↑EmptyCollection.emptyCollection
      h : MeasureTheory.NullMeasurable f 0
      hft : Filter.Eventually (fun x => Membership.mem EmptyCollection.emptyCollecti …
      ⊢ AEMeasurable f 0
    -/
    apply aemeasurable_zero_measure
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      t : Set β
      inst✝ : MeasurableSpace.CountablyGenerated ↑t
      h : MeasureTheory.NullMeasurable f μ
      hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
      hne : t.Nonempty
      ⊢ AEMeasurable f μ
    -/
  · rw [← μ.ae_completion] at hft
    obtain ⟨f', hf'm, hf't, hff'⟩ :
        ∃ f' : α → β, NullMeasurable f' μ ∧ range f' ⊆ t ∧ f =ᵐ[μ] f' :=
      h.measurable'.aemeasurable.exists_ae_eq_range_subset hft hne
    /-
      case inr.intro.intro.intro
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      t : Set β
      inst✝ : MeasurableSpace.CountablyGenerated ↑t
      h : MeasureTheory.NullMeasurable f μ
      hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ. …
      hne : t.Nonempty
      f' : α → β
      hf'm : MeasureTheory.NullMeasurable f' μ
      hf't : HasSubset.Subset (Set.range f') t
      hff' : (MeasureTheory.ae μ).EventuallyEq f f'
      ⊢ AEMeasurable f μ
    -/
    rw [range_subset_iff] at hf't
    /-
      case inr.intro.intro.intro
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      t : Set β
      inst✝ : MeasurableSpace.CountablyGenerated ↑t
      h : MeasureTheory.NullMeasurable f μ
      hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ. …
      hne : t.Nonempty
      f' : α → β
      hf'm : MeasureTheory.NullMeasurable f' μ
      hf't : ∀ (y : α), Membership.mem t (f' y)
      hff' : (MeasureTheory.ae μ).EventuallyEq f f'
      ⊢ AEMeasurable f μ
    -/
    lift f' to α → t using hf't
    /-
      case inr.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      t : Set β
      inst✝ : MeasurableSpace.CountablyGenerated ↑t
      h : MeasureTheory.NullMeasurable f μ
      hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ. …
      hne : t.Nonempty
      f' : α → ↑t
      hf'm : MeasureTheory.NullMeasurable (fun i => ↑(f' i)) μ
      hff' : (MeasureTheory.ae μ).EventuallyEq f fun i => ↑(f' i)
      ⊢ AEMeasurable f μ
    -/
    replace hf'm : NullMeasurable f' μ := hf'm.measurable'.subtype_mk
    /-
      case inr.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      t : Set β
      inst✝ : MeasurableSpace.CountablyGenerated ↑t
      h : MeasureTheory.NullMeasurable f μ
      hft : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ. …
      hne : t.Nonempty
      f' : α → ↑t
      hff' : (MeasureTheory.ae μ).EventuallyEq f fun i => ↑(f' i)
      hf'm : MeasureTheory.NullMeasurable f' μ
      ⊢ AEMeasurable f μ
    -/
    exact (measurable_subtype_coe.comp_aemeasurable hf'm.aemeasurable).congr hff'.symm
    /-
      🎉 no goals
    -/


lemma map_sum {ι : Type*} {m : ι → Measure α} {f : α → β} (hf : AEMeasurable f (Measure.sum m)) :
    Measure.map f (Measure.sum m) = Measure.sum (fun i ↦ Measure.map f (m i)) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_7
    m : ι → MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f (MeasureTheory.Measure.sum m)
    ⊢ Eq (MeasureTheory.Measure.map f (MeasureTheory.Measure.sum m)) (MeasureTheor …
  -/
  ext s hs
  /-
    case h
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_7
    m : ι → MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f (MeasureTheory.Measure.sum m)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map f (MeasureTheory.Measure.sum m)) s) ((Measure …
  -/
  rw [map_apply_of_aemeasurable hf hs, sum_apply₀ _ (hf.nullMeasurable hs), sum_apply _ hs]
  /-
    case h
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_7
    m : ι → MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f (MeasureTheory.Measure.sum m)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (tsum fun i => (m i) (Set.preimage f s)) (tsum fun i => (MeasureTheory.Me …
  -/
  have M i : AEMeasurable f (m i) := hf.mono_measure (le_sum m i)
  /-
    case h
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_7
    m : ι → MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f (MeasureTheory.Measure.sum m)
    s : Set β
    hs : MeasurableSet s
    M : ∀ (i : ι), AEMeasurable f (m i)
    ⊢ Eq (tsum fun i => (m i) (Set.preimage f s)) (tsum fun i => (MeasureTheory.Me …
  -/
  simp_rw [map_apply_of_aemeasurable (M _) hs]
  /-
    🎉 no goals
  -/


instance (μ : Measure α) (f : α → β) [SFinite μ] : SFinite (μ.map f) := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    δ : Type u_5
    R : Type u_6
    m0 : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    inst✝¹ : MeasurableSpace δ
    f✝ g : α → β
    μ✝ ν μ : MeasureTheory.Measure α
    f : α → β
    inst✝ : MeasureTheory.SFinite μ
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map f μ)
  -/
  by_cases H : AEMeasurable f μ
    /-
      case pos
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      R : Type u_6
      m0 : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      inst✝¹ : MeasurableSpace δ
      f✝ g : α → β
      μ✝ ν μ : MeasureTheory.Measure α
      f : α → β
      inst✝ : MeasureTheory.SFinite μ
      H : AEMeasurable f μ
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map f μ)
    -/
  · rw [← sum_sfiniteSeq μ] at H ⊢
    /-
      case pos
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      R : Type u_6
      m0 : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      inst✝¹ : MeasurableSpace δ
      f✝ g : α → β
      μ✝ ν μ : MeasureTheory.Measure α
      f : α → β
      inst✝ : MeasureTheory.SFinite μ
      H : AEMeasurable f (MeasureTheory.Measure.sum (MeasureTheory.sfiniteSeq μ))
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map f (MeasureTheory.Measure.su …
    -/
    rw [map_sum H]
    /-
      case pos
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      R : Type u_6
      m0 : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      inst✝¹ : MeasurableSpace δ
      f✝ g : α → β
      μ✝ ν μ : MeasureTheory.Measure α
      f : α → β
      inst✝ : MeasureTheory.SFinite μ
      H : AEMeasurable f (MeasureTheory.Measure.sum (MeasureTheory.sfiniteSeq μ))
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun i => MeasureTheory.Meas …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      R : Type u_6
      m0 : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      inst✝¹ : MeasurableSpace δ
      f✝ g : α → β
      μ✝ ν μ : MeasureTheory.Measure α
      f : α → β
      inst✝ : MeasureTheory.SFinite μ
      H : Not (AEMeasurable f μ)
      ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map f μ)
    -/
  · rw [map_of_not_aemeasurable H]
    /-
      case neg
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      R : Type u_6
      m0 : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      inst✝¹ : MeasurableSpace δ
      f✝ g : α → β
      μ✝ ν μ : MeasureTheory.Measure α
      f : α → β
      inst✝ : MeasureTheory.SFinite μ
      H : Not (AEMeasurable f μ)
      ⊢ MeasureTheory.SFinite 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/


