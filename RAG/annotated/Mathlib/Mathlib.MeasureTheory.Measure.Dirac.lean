/-- The dirac measure. -/
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        δ : Type u_3
                                                                        inst✝¹ : MeasurableSpace α
                                                                        inst✝ : MeasurableSpace β
                                                                        s : Set α
                                                                        a✝ a : α
                                                                        ⊢ LE.le inst✝¹ (MeasureTheory.OuterMeasure.dirac a).caratheodory
                                                                      -/
def dirac (a : α) : Measure α := (OuterMeasure.dirac a).toMeasure (by simp)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance : MeasureSpace PUnit :=
  ⟨dirac PUnit.unit⟩


theorem le_dirac_apply {a} : s.indicator 1 a ≤ dirac a s :=
  OuterMeasure.dirac_apply a s ▸ le_toMeasure_apply _ _ _


@[simp]
theorem dirac_apply' (a : α) (hs : MeasurableSet s) : dirac a s = s.indicator 1 a :=
  toMeasure_apply _ _ hs


@[simp]
theorem dirac_apply_of_mem {a : α} (h : a ∈ s) : dirac a s = 1 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    a : α
    h : Membership.mem s a
    ⊢ Eq ((MeasureTheory.Measure.dirac a) s) 1
  -/
  have : ∀ t : Set α, a ∈ t → t.indicator (1 : α → ℝ≥0∞) a = 1 := fun t ht => indicator_of_mem ht 1
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    a : α
    h : Membership.mem s a
    this : ∀ (t : Set α), Membership.mem t a → Eq (t.indicator 1 a) 1
    ⊢ Eq ((MeasureTheory.Measure.dirac a) s) 1
  -/
  refine le_antisymm (this univ trivial ▸ ?_) (this s h ▸ le_dirac_apply)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    a : α
    h : Membership.mem s a
    this : ∀ (t : Set α), Membership.mem t a → Eq (t.indicator 1 a) 1
    ⊢ LE.le ((MeasureTheory.Measure.dirac a) s) (Set.univ.indicator 1 a)
  -/
  rw [← dirac_apply' a MeasurableSet.univ]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    a : α
    h : Membership.mem s a
    this : ∀ (t : Set α), Membership.mem t a → Eq (t.indicator 1 a) 1
    ⊢ LE.le ((MeasureTheory.Measure.dirac a) s) ((MeasureTheory.Measure.dirac a) S …
  -/
  exact measure_mono (subset_univ s)
  /-
    🎉 no goals
  -/


@[simp]
theorem dirac_apply [MeasurableSingletonClass α] (a : α) (s : Set α) :
    dirac a s = s.indicator 1 a := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α
    s : Set α
    ⊢ Eq ((MeasureTheory.Measure.dirac a) s) (s.indicator 1 a)
  -/
  by_cases h : a ∈ s; · rw [dirac_apply_of_mem h, indicator_of_mem h, Pi.one_apply]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    ⊢ Eq ((MeasureTheory.Measure.dirac a) s) (s.indicator 1 a)
  -/
  rw [indicator_of_not_mem h, ← nonpos_iff_eq_zero]
  calc
    dirac a s ≤ dirac a {a}ᶜ := measure_mono (subset_compl_comm.1 <| singleton_subset_iff.2 h)
    _ = 0 := by simp [dirac_apply' _ (measurableSet_singleton _).compl]


@[simp] lemma dirac_ne_zero : dirac a ≠ 0 :=
             /-
               α : Type u_1
               inst✝ : MeasurableSpace α
               a : α
               h : Eq (MeasureTheory.Measure.dirac a) 0
               ⊢ False
             -/
  fun h ↦ by simpa [h] using dirac_apply_of_mem (mem_univ a)
             /-
               🎉 no goals
             -/


theorem map_dirac {f : α → β} (hf : Measurable f) (a : α) : (dirac a).map f = dirac (f a) := by
  classical
  exact ext fun s hs => by simp [hs, map_apply hf hs, hf hs, indicator_apply]


lemma map_const (μ : Measure α) (c : β) : μ.map (fun _ ↦ c) = (μ Set.univ) • dirac c := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    c : β
    ⊢ Eq (MeasureTheory.Measure.map (fun x => c) μ) (HSMul.hSMul (μ Set.univ) (Mea …
  -/
  ext s hs
  simp only [aemeasurable_const, measurable_const, Measure.coe_smul, Pi.smul_apply,
    dirac_apply' _ hs, smul_eq_mul]
  classical
  rw [Measure.map_apply measurable_const hs, Set.preimage_const]
  by_cases hsc : c ∈ s
  · rw [(Set.indicator_eq_one_iff_mem _).mpr hsc, mul_one, if_pos hsc]
  · rw [if_neg hsc, (Set.indicator_eq_zero_iff_not_mem _).mpr hsc, measure_empty, mul_zero]


@[simp]
theorem restrict_singleton (μ : Measure α) (a : α) : μ.restrict {a} = μ {a} • dirac a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Eq (μ.restrict (Singleton.singleton a)) (HSMul.hSMul (μ (Singleton.singleton …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    a : α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.restrict (Singleton.singleton a)) s) ((HSMul.hSMul (μ (Singleton.sing …
  -/
  by_cases ha : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      a : α
      s : Set α
      hs : MeasurableSet s
      ha : Membership.mem s a
      ⊢ Eq ((μ.restrict (Singleton.singleton a)) s) ((HSMul.hSMul (μ (Singleton.sing …
    -/
  · have : s ∩ {a} = {a} := by simpa
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      a : α
      s : Set α
      hs : MeasurableSet s
      ha : Membership.mem s a
      this : Eq (Inter.inter s (Singleton.singleton a)) (Singleton.singleton a)
      ⊢ Eq ((μ.restrict (Singleton.singleton a)) s) ((HSMul.hSMul (μ (Singleton.sing …
    -/
    simp [*]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      a : α
      s : Set α
      hs : MeasurableSet s
      ha : Not (Membership.mem s a)
      ⊢ Eq ((μ.restrict (Singleton.singleton a)) s) ((HSMul.hSMul (μ (Singleton.sing …
    -/
  · have : s ∩ {a} = ∅ := inter_singleton_eq_empty.2 ha
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      a : α
      s : Set α
      hs : MeasurableSet s
      ha : Not (Membership.mem s a)
      this : Eq (Inter.inter s (Singleton.singleton a)) EmptyCollection.emptyCollect …
      ⊢ Eq ((μ.restrict (Singleton.singleton a)) s) ((HSMul.hSMul (μ (Singleton.sing …
    -/
    simp [*]
    /-
      🎉 no goals
    -/


/-- Two measures on a countable space are equal if they agree on singletons. -/
theorem ext_of_singleton [Countable α] {μ ν : Measure α} (h : ∀ a, μ {a} = ν {a}) : μ = ν :=
                                                        /-
                                                          α : Type u_1
                                                          inst✝¹ : MeasurableSpace α
                                                          inst✝ : Countable α
                                                          μ ν : MeasureTheory.Measure α
                                                          h : ∀ (a : α), Eq (μ (Singleton.singleton a)) (ν (Singleton.singleton a))
                                                          ⊢ Eq (Set.range Singleton.singleton).sUnion Set.univ
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  ext_of_sUnion_eq_univ (countable_range singleton) (by aesop) (by aesop)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Two measures on a countable space are equal if and only if they agree on singletons. -/
theorem ext_iff_singleton [Countable α] {μ ν : Measure α} : μ = ν ↔ ∀ a, μ {a} = ν {a} :=
  ⟨fun h _ ↦ h ▸ rfl, ext_of_singleton⟩


/-- If `f` is a map with countable codomain, then `μ.map f` is a sum of Dirac measures. -/
theorem map_eq_sum [Countable β] [MeasurableSingletonClass β] (μ : Measure α) (f : α → β)
    (hf : Measurable f) : μ.map f = sum fun b : β => μ (f ⁻¹' {b}) • dirac b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : Countable β
    inst✝ : MeasurableSingletonClass β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.sum fun b => HSMul …
  -/
  ext s
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : Countable β
    inst✝ : MeasurableSingletonClass β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    s : Set β
    a✝ : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map f μ) s) ((MeasureTheory.Measure.sum fun b =>  …
  -/
  have : ∀ y ∈ s, MeasurableSet (f ⁻¹' {y}) := fun y _ => hf (measurableSet_singleton _)
  simp [← tsum_measure_preimage_singleton (to_countable s) this, *,
    tsum_subtype s fun b => μ (f ⁻¹' {b}), ← indicator_mul_right s fun b => μ (f ⁻¹' {b})]


/-- A measure on a countable type is a sum of Dirac measures. -/
@[simp]
theorem sum_smul_dirac [Countable α] [MeasurableSingletonClass α] (μ : Measure α) :
                                             /-
                                               α : Type u_1
                                               inst✝² : MeasurableSpace α
                                               inst✝¹ : Countable α
                                               inst✝ : MeasurableSingletonClass α
                                               μ : MeasureTheory.Measure α
                                               ⊢ Eq (MeasureTheory.Measure.sum fun a => HSMul.hSMul (μ (Singleton.singleton a …
                                             -/
    (sum fun a => μ {a} • dirac a) = μ := by simpa using (map_eq_sum μ id measurable_id).symm
                                             /-
                                               🎉 no goals
                                             -/


/-- Given that `α` is a countable, measurable space with all singleton sets measurable,
write the measure of a set `s` as the sum of the measure of `{x}` for all `x ∈ s`. -/
theorem tsum_indicator_apply_singleton [Countable α] [MeasurableSingletonClass α] (μ : Measure α)
    (s : Set α) (hs : MeasurableSet s) : (∑' x : α, s.indicator (fun x => μ {x}) x) = μ s := by
  classical
  calc
    (∑' x : α, s.indicator (fun x => μ {x}) x) =
      Measure.sum (fun a => μ {a} • Measure.dirac a) s := by
      simp only [Measure.sum_apply _ hs, Measure.smul_apply, smul_eq_mul, Measure.dirac_apply,
        Set.indicator_apply, mul_ite, Pi.one_apply, mul_one, mul_zero]
    _ = μ s := by rw [μ.sum_smul_dirac]


theorem mem_ae_dirac_iff {a : α} (hs : MeasurableSet s) : s ∈ ae (dirac a) ↔ a ∈ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    a : α
    hs : MeasurableSet s
    ⊢ Iff (Membership.mem (MeasureTheory.ae (MeasureTheory.Measure.dirac a)) s) (M …
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases a ∈ s <;> simp [mem_ae_iff, dirac_apply', hs.compl, indicator_apply, *]
                     /-
                       🎉 no goals
                     -/


theorem ae_dirac_iff {a : α} {p : α → Prop} (hp : MeasurableSet { x | p x }) :
    (∀ᵐ x ∂dirac a, p x) ↔ p a :=
  mem_ae_dirac_iff hp


@[simp]
theorem ae_dirac_eq [MeasurableSingletonClass α] (a : α) : ae (dirac a) = pure a := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α
    ⊢ Eq (MeasureTheory.ae (MeasureTheory.Measure.dirac a)) (Pure.pure a)
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α
    s : Set α
    ⊢ Iff (Membership.mem (MeasureTheory.ae (MeasureTheory.Measure.dirac a)) s) (M …
  -/
  simp [mem_ae_iff, imp_false]
  /-
    🎉 no goals
  -/


theorem ae_eq_dirac' [MeasurableSingletonClass β] {a : α} {f : α → β} (hf : Measurable f) :
    f =ᵐ[dirac a] const α (f a) :=
  (ae_dirac_iff <| show MeasurableSet (f ⁻¹' {f a}) from hf <| measurableSet_singleton _).2 rfl


theorem ae_eq_dirac [MeasurableSingletonClass α] {a : α} (f : α → δ) :
                                      /-
                                        α : Type u_1
                                        δ : Type u_3
                                        inst✝¹ : MeasurableSpace α
                                        inst✝ : MeasurableSingletonClass α
                                        a : α
                                        f : α → δ
                                        ⊢ (MeasureTheory.ae (MeasureTheory.Measure.dirac a)).EventuallyEq f (Function. …
                                      -/
    f =ᵐ[dirac a] const α (f a) := by simp [Filter.EventuallyEq]
                                      /-
                                        🎉 no goals
                                      -/


instance Measure.dirac.isProbabilityMeasure {x : α} : IsProbabilityMeasure (dirac x) :=
  ⟨dirac_apply_of_mem <| mem_univ x⟩


instance Measure.dirac.instIsFiniteMeasure {a : α} : IsFiniteMeasure (dirac a) := inferInstance

instance Measure.dirac.instSigmaFinite {a : α} : SigmaFinite (dirac a) := inferInstance


theorem restrict_dirac' (hs : MeasurableSet s) [Decidable (a ∈ s)] :
    (Measure.dirac a).restrict s = if a ∈ s then Measure.dirac a else 0 := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    s : Set α
    a : α
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) (ite (Membership.mem s a) (M …
  -/
  split_ifs with has
    /-
      case pos
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      s : Set α
      a : α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      has : Membership.mem s a
      ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) (MeasureTheory.Measure.dirac …
    -/
  · apply restrict_eq_self_of_ae_mem
    /-
      case pos.hs
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      s : Set α
      a : α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      has : Membership.mem s a
      ⊢ Filter.Eventually (fun x => Membership.mem s x) (MeasureTheory.ae (MeasureTh …
    -/
                          /-
                            🎉 no goals
                          -/
    rw [ae_dirac_iff] <;> assumption
                          /-
                            🎉 no goals
                          -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      s : Set α
      a : α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      has : Not (Membership.mem s a)
      ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) 0
    -/
  · rw [restrict_eq_zero, dirac_apply' _ hs, indicator_of_not_mem has]
    /-
      🎉 no goals
    -/


theorem restrict_dirac [MeasurableSingletonClass α] [Decidable (a ∈ s)] :
    (Measure.dirac a).restrict s = if a ∈ s then Measure.dirac a else 0 := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    s : Set α
    a : α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) (ite (Membership.mem s a) (M …
  -/
  split_ifs with has
    /-
      case pos
      α : Type u_1
      inst✝² : MeasurableSpace α
      s : Set α
      a : α
      inst✝¹ : MeasurableSingletonClass α
      inst✝ : Decidable (Membership.mem s a)
      has : Membership.mem s a
      ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) (MeasureTheory.Measure.dirac …
    -/
  · apply restrict_eq_self_of_ae_mem
    /-
      case pos.hs
      α : Type u_1
      inst✝² : MeasurableSpace α
      s : Set α
      a : α
      inst✝¹ : MeasurableSingletonClass α
      inst✝ : Decidable (Membership.mem s a)
      has : Membership.mem s a
      ⊢ Filter.Eventually (fun x => Membership.mem s x) (MeasureTheory.ae (MeasureTh …
    -/
    rwa [ae_dirac_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : MeasurableSpace α
      s : Set α
      a : α
      inst✝¹ : MeasurableSingletonClass α
      inst✝ : Decidable (Membership.mem s a)
      has : Not (Membership.mem s a)
      ⊢ Eq ((MeasureTheory.Measure.dirac a).restrict s) 0
    -/
  · rw [restrict_eq_zero, dirac_apply, indicator_of_not_mem has]
    /-
      🎉 no goals
    -/


lemma mutuallySingular_dirac [MeasurableSingletonClass α] (x : α) (μ : Measure α) [NoAtoms μ] :
    Measure.dirac x ⟂ₘ μ :=
                                               /-
                                                 α : Type u_1
                                                 inst✝² : MeasurableSpace α
                                                 inst✝¹ : MeasurableSingletonClass α
                                                 x : α
                                                 μ : MeasureTheory.Measure α
                                                 inst✝ : MeasureTheory.NoAtoms μ
                                                 ⊢ Eq ((MeasureTheory.Measure.dirac x) (HasCompl.compl (Singleton.singleton x)) …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  ⟨{x}ᶜ, (MeasurableSet.singleton x).compl, by simp, by simp⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Dirac delta measures at two points are equal if every measurable set contains either both or
neither of the points. -/
lemma dirac_eq_dirac_iff_forall_mem_iff_mem {x y : α} :
    Measure.dirac x = Measure.dirac y ↔ ∀ A, MeasurableSet A → (x ∈ A ↔ y ∈ A) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    x y : α
    ⊢ Iff (Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)) (∀  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      ⊢ Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y) → ∀ (A :  …
    -/
  · intro h A A_mble
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
      A : Set α
      A_mble : MeasurableSet A
      ⊢ Iff (Membership.mem A x) (Membership.mem A y)
    -/
    have obs := congr_arg (fun μ ↦ μ A) h
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
      A : Set α
      A_mble : MeasurableSet A
      obs : Eq ((fun μ => μ A) (MeasureTheory.Measure.dirac x)) ((fun μ => μ A) (Mea …
      ⊢ Iff (Membership.mem A x) (Membership.mem A y)
    -/
    simp only [Measure.dirac_apply' _ A_mble] at obs
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
      A : Set α
      A_mble : MeasurableSet A
      obs : Eq (A.indicator 1 x) (A.indicator 1 y)
      ⊢ Iff (Membership.mem A x) (Membership.mem A y)
    -/
    by_cases x_in_A : x ∈ A
    · simpa only [x_in_A, indicator_of_mem, Pi.one_apply, true_iff, Eq.comm (a := (1 : ℝ≥0∞)),
                  indicator_eq_one_iff_mem] using obs
    · simpa only [x_in_A, indicator_of_not_mem, Eq.comm (a := (0 : ℝ≥0∞)), indicator_apply_eq_zero,
                  false_iff, not_false_eq_true, Pi.one_apply, one_ne_zero, imp_false] using obs
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      ⊢ (∀ (A : Set α), MeasurableSet A → Iff (Membership.mem A x) (Membership.mem A …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : ∀ (A : Set α), MeasurableSet A → Iff (Membership.mem A x) (Membership.mem  …
      ⊢ Eq (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
    -/
    ext A A_mble
    /-
      case mpr.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : ∀ (A : Set α), MeasurableSet A → Iff (Membership.mem A x) (Membership.mem  …
      A : Set α
      A_mble : MeasurableSet A
      ⊢ Eq ((MeasureTheory.Measure.dirac x) A) ((MeasureTheory.Measure.dirac y) A)
    -/
    by_cases x_in_A : x ∈ A
    · simp only [Measure.dirac_apply' _ A_mble, x_in_A, indicator_of_mem, Pi.one_apply,
                 (h A A_mble).mp x_in_A]
      /-
        case neg
        α : Type u_1
        inst✝ : MeasurableSpace α
        x y : α
        h : ∀ (A : Set α), MeasurableSet A → Iff (Membership.mem A x) (Membership.mem  …
        A : Set α
        A_mble : MeasurableSet A
        x_in_A : Not (Membership.mem A x)
        ⊢ Eq ((MeasureTheory.Measure.dirac x) A) ((MeasureTheory.Measure.dirac y) A)
      -/
    · have y_notin_A : y ∉ A := by simp_all only [false_iff, not_false_eq_true]
      simp only [Measure.dirac_apply' _ A_mble, x_in_A, y_notin_A,
                 not_false_eq_true, indicator_of_not_mem]


/-- Dirac delta measures at two points are different if and only if there is a measurable set
containing one of the points but not the other. -/
lemma dirac_ne_dirac_iff_exists_measurableSet {x y : α} :
    Measure.dirac x ≠ Measure.dirac y ↔ ∃ A, MeasurableSet A ∧ x ∈ A ∧ y ∉ A := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    x y : α
    ⊢ Iff (Ne (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)) (Ex …
  -/
  apply not_iff_not.mp
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    x y : α
    ⊢ Iff (Not (Ne (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y) …
  -/
  simp only [ne_eq, not_not, not_exists, not_and, dirac_eq_dirac_iff_forall_mem_iff_mem]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    x y : α
    ⊢ Iff (∀ (A : Set α), MeasurableSet A → Iff (Membership.mem A x) (Membership.m …
  -/
  refine ⟨fun h A A_mble ↦ by simp only [h A A_mble, imp_self], fun h A A_mble ↦ ?_⟩
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    x y : α
    h : ∀ (x_1 : Set α), MeasurableSet x_1 → Membership.mem x_1 x → Membership.mem …
    A : Set α
    A_mble : MeasurableSet A
    ⊢ Iff (Membership.mem A x) (Membership.mem A y)
  -/
  by_cases x_in_A : x ∈ A
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : ∀ (x_1 : Set α), MeasurableSet x_1 → Membership.mem x_1 x → Membership.mem …
      A : Set α
      A_mble : MeasurableSet A
      x_in_A : Membership.mem A x
      ⊢ Iff (Membership.mem A x) (Membership.mem A y)
    -/
  · simp only [x_in_A, h A A_mble x_in_A]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      x y : α
      h : ∀ (x_1 : Set α), MeasurableSet x_1 → Membership.mem x_1 x → Membership.mem …
      A : Set α
      A_mble : MeasurableSet A
      x_in_A : Not (Membership.mem A x)
      ⊢ Iff (Membership.mem A x) (Membership.mem A y)
    -/
  · simpa only [x_in_A, false_iff] using h Aᶜ (MeasurableSet.compl_iff.mpr A_mble) x_in_A
    /-
      🎉 no goals
    -/


/-- Dirac delta measures at two different points are different, assuming the measurable space
separates points. -/
lemma dirac_ne_dirac [SeparatesPoints α] {x y : α} (x_ne_y : x ≠ y) :
    Measure.dirac x ≠ Measure.dirac y := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace.SeparatesPoints α
    x y : α
    x_ne_y : Ne x y
    ⊢ Ne (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
  -/
  obtain ⟨A, A_mble, x_in_A, y_notin_A⟩ := exists_measurableSet_of_ne x_ne_y
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace.SeparatesPoints α
    x y : α
    x_ne_y : Ne x y
    A : Set α
    A_mble : MeasurableSet A
    x_in_A : Membership.mem A x
    y_notin_A : Not (Membership.mem A y)
    ⊢ Ne (MeasureTheory.Measure.dirac x) (MeasureTheory.Measure.dirac y)
  -/
  exact dirac_ne_dirac_iff_exists_measurableSet.mpr ⟨A, A_mble, x_in_A, y_notin_A⟩
  /-
    🎉 no goals
  -/


/-- Dirac delta measures at two points are different if and only if the two points are different,
assuming the measurable space separates points. -/
lemma dirac_ne_dirac_iff [SeparatesPoints α] {x y : α} :
    Measure.dirac x ≠ Measure.dirac y ↔ x ≠ y :=
  ⟨fun h x_eq_y ↦ h <| congrArg dirac x_eq_y, fun h ↦ dirac_ne_dirac h⟩


/-- Dirac delta measures at two points are equal if and only if the two points are equal,
assuming the measurable space separates points. -/
lemma dirac_eq_dirac_iff [SeparatesPoints α] {x y : α} :
    Measure.dirac x = Measure.dirac y ↔ x = y := not_iff_not.mp dirac_ne_dirac_iff


/-- The assignment `x ↦ dirac x` is injective, assuming the measurable space separates points. -/
lemma injective_dirac [SeparatesPoints α] :
                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝¹ : MeasurableSpace α
                                                                        inst✝ : MeasurableSpace.SeparatesPoints α
                                                                        x y : α
                                                                        x_ne_y : Eq ((fun x => MeasureTheory.Measure.dirac x) x) ((fun x => MeasureThe …
                                                                        ⊢ Eq x y
                                                                      -/
    Function.Injective (fun (x : α) ↦ dirac x) := fun x y x_ne_y ↦ by rwa [← dirac_eq_dirac_iff]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


