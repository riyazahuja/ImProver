/-- A function `f` is strongly measurable at a filter `l` w.r.t. a measure `μ` if it is
ae strongly measurable w.r.t. `μ.restrict s` for some `s ∈ l`. -/
def StronglyMeasurableAtFilter (f : α → β) (l : Filter α) (μ : Measure α := by volume_tac) :=
  ∃ s ∈ l, AEStronglyMeasurable f (μ.restrict s)


@[simp]
theorem stronglyMeasurableAt_bot {f : α → β} : StronglyMeasurableAtFilter f ⊥ μ :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝¹ : MeasurableSpace α
                    inst✝ : TopologicalSpace β
                    μ : MeasureTheory.Measure α
                    f : α → β
                    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict EmptyCollection.emptyCollec …
                  -/
  ⟨∅, mem_bot, by simp⟩
                  /-
                    🎉 no goals
                  -/


protected theorem StronglyMeasurableAtFilter.eventually (h : StronglyMeasurableAtFilter f l μ) :
    ∀ᶠ s in l.smallSets, AEStronglyMeasurable f (μ.restrict s) :=
  (eventually_smallSets' fun _ _ => AEStronglyMeasurable.mono_set).2 h


protected theorem StronglyMeasurableAtFilter.filter_mono (h : StronglyMeasurableAtFilter f l μ)
    (h' : l' ≤ l) : StronglyMeasurableAtFilter f l' μ :=
  let ⟨s, hsl, hs⟩ := h
  ⟨s, h' hsl, hs⟩


protected theorem MeasureTheory.AEStronglyMeasurable.stronglyMeasurableAtFilter
    (h : AEStronglyMeasurable f μ) : StronglyMeasurableAtFilter f l μ :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : MeasurableSpace α
                        inst✝ : TopologicalSpace β
                        l : Filter α
                        f : α → β
                        μ : MeasureTheory.Measure α
                        h : MeasureTheory.AEStronglyMeasurable f μ
                        ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict Set.univ)
                      -/
  ⟨univ, univ_mem, by rwa [Measure.restrict_univ]⟩
                      /-
                        🎉 no goals
                      -/


theorem AeStronglyMeasurable.stronglyMeasurableAtFilter_of_mem {s}
    (h : AEStronglyMeasurable f (μ.restrict s)) (hl : s ∈ l) : StronglyMeasurableAtFilter f l μ :=
  ⟨s, hl, h⟩


protected theorem MeasureTheory.StronglyMeasurable.stronglyMeasurableAtFilter
    (h : StronglyMeasurable f) : StronglyMeasurableAtFilter f l μ :=
  h.aestronglyMeasurable.stronglyMeasurableAtFilter


theorem hasFiniteIntegral_restrict_of_bounded [NormedAddCommGroup E] {f : α → E} {s : Set α}
    {μ : Measure α} {C} (hs : μ s < ∞) (hf : ∀ᵐ x ∂μ.restrict s, ‖f x‖ ≤ C) :
    HasFiniteIntegral f (μ.restrict s) :=
                                                /-
                                                  α : Type u_1
                                                  E : Type u_4
                                                  inst✝¹ : MeasurableSpace α
                                                  inst✝ : NormedAddCommGroup E
                                                  f : α → E
                                                  s : Set α
                                                  μ : MeasureTheory.Measure α
                                                  C : Real
                                                  hs : LT.lt (μ s) Top.top
                                                  hf : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae  …
                                                  ⊢ LT.lt ((μ.restrict s) Set.univ) Top.top
                                                -/
  haveI : IsFiniteMeasure (μ.restrict s) := ⟨by rwa [Measure.restrict_apply_univ]⟩
                                                /-
                                                  🎉 no goals
                                                -/
  hasFiniteIntegral_of_bounded hf


/-- A function is `IntegrableOn` a set `s` if it is almost everywhere strongly measurable on `s`
and if the integral of its pointwise norm over `s` is less than infinity. -/
def IntegrableOn (f : α → ε) (s : Set α) (μ : Measure α := by volume_tac) : Prop :=
  Integrable f (μ.restrict s)


theorem IntegrableOn.integrable (h : IntegrableOn f s μ) : Integrable f (μ.restrict s) :=
  h


@[simp]
                                                      /-
                                                        α : Type u_1
                                                        E : Type u_4
                                                        inst✝¹ : MeasurableSpace α
                                                        inst✝ : NormedAddCommGroup E
                                                        f : α → E
                                                        μ : MeasureTheory.Measure α
                                                        ⊢ MeasureTheory.IntegrableOn f EmptyCollection.emptyCollection μ
                                                      -/
theorem integrableOn_empty : IntegrableOn f ∅ μ := by simp [IntegrableOn, integrable_zero_measure]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem integrableOn_univ : IntegrableOn f univ μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.IntegrableOn f Set.univ μ) (MeasureTheory.Integrable f μ)
  -/
  rw [IntegrableOn, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


theorem integrableOn_zero : IntegrableOn (fun _ => (0 : E)) s μ :=
  integrable_zero _ _ _


@[simp]
theorem integrableOn_const {C : E} : IntegrableOn (fun _ => C) s μ ↔ C = 0 ∨ μ s < ∞ :=
                                   /-
                                     α : Type u_1
                                     E : Type u_4
                                     inst✝¹ : MeasurableSpace α
                                     inst✝ : NormedAddCommGroup E
                                     s : Set α
                                     μ : MeasureTheory.Measure α
                                     C : E
                                     ⊢ Iff (Or (Eq C 0) (LT.lt ((μ.restrict s) Set.univ) Top.top)) (Or (Eq C 0) (LT …
                                   -/
  integrable_const_iff.trans <| by rw [Measure.restrict_apply_univ]
                                   /-
                                     🎉 no goals
                                   -/


theorem IntegrableOn.mono (h : IntegrableOn f t ν) (hs : s ⊆ t) (hμ : μ ≤ ν) : IntegrableOn f s μ :=
  h.mono_measure <| Measure.restrict_mono hs hμ


theorem IntegrableOn.mono_set (h : IntegrableOn f t μ) (hst : s ⊆ t) : IntegrableOn f s μ :=
  h.mono hst le_rfl


theorem IntegrableOn.mono_measure (h : IntegrableOn f s ν) (hμ : μ ≤ ν) : IntegrableOn f s μ :=
  h.mono (Subset.refl _) hμ


theorem IntegrableOn.mono_set_ae (h : IntegrableOn f t μ) (hst : s ≤ᵐ[μ] t) : IntegrableOn f s μ :=
  h.integrable.mono_measure <| Measure.restrict_mono_ae hst


theorem IntegrableOn.congr_set_ae (h : IntegrableOn f t μ) (hst : s =ᵐ[μ] t) : IntegrableOn f s μ :=
  h.mono_set_ae hst.le


theorem IntegrableOn.congr_fun_ae (h : IntegrableOn f s μ) (hst : f =ᵐ[μ.restrict s] g) :
    IntegrableOn g s μ :=
  Integrable.congr h hst


theorem integrableOn_congr_fun_ae (hst : f =ᵐ[μ.restrict s] g) :
    IntegrableOn f s μ ↔ IntegrableOn g s μ :=
  ⟨fun h => h.congr_fun_ae hst, fun h => h.congr_fun_ae hst.symm⟩


theorem IntegrableOn.congr_fun (h : IntegrableOn f s μ) (hst : EqOn f g s) (hs : MeasurableSet s) :
    IntegrableOn g s μ :=
  h.congr_fun_ae ((ae_restrict_iff' hs).2 (Eventually.of_forall hst))


theorem integrableOn_congr_fun (hst : EqOn f g s) (hs : MeasurableSet s) :
    IntegrableOn f s μ ↔ IntegrableOn g s μ :=
  ⟨fun h => h.congr_fun hst hs, fun h => h.congr_fun hst.symm hs⟩


theorem Integrable.integrableOn (h : Integrable f μ) : IntegrableOn f s μ := h.restrict


theorem IntegrableOn.restrict (h : IntegrableOn f s μ) (hs : MeasurableSet s) :
    IntegrableOn f s (μ.restrict t) := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.IntegrableOn f s μ
    hs : MeasurableSet s
    ⊢ MeasureTheory.IntegrableOn f s (μ.restrict t)
  -/
  rw [IntegrableOn, Measure.restrict_restrict hs]; exact h.mono_set inter_subset_left
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem IntegrableOn.inter_of_restrict (h : IntegrableOn f s (μ.restrict t)) :
    IntegrableOn f (s ∩ t) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.IntegrableOn f s (μ.restrict t)
    ⊢ MeasureTheory.IntegrableOn f (Inter.inter s t) μ
  -/
  have := h.mono_set (inter_subset_left (t := t))
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    h : MeasureTheory.IntegrableOn f s (μ.restrict t)
    this : MeasureTheory.IntegrableOn f (Inter.inter s t) (μ.restrict t)
    ⊢ MeasureTheory.IntegrableOn f (Inter.inter s t) μ
  -/
  rwa [IntegrableOn, μ.restrict_restrict_of_subset inter_subset_right] at this
  /-
    🎉 no goals
  -/


lemma Integrable.piecewise [DecidablePred (· ∈ s)]
    (hs : MeasurableSet s) (hf : IntegrableOn f s μ) (hg : IntegrableOn g sᶜ μ) :
    Integrable (s.piecewise f g) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f g : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.IntegrableOn f s μ
    hg : MeasureTheory.IntegrableOn g (HasCompl.compl s) μ
    ⊢ MeasureTheory.Integrable (s.piecewise f g) μ
  -/
  rw [IntegrableOn] at hf hg
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f g : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.Integrable f (μ.restrict s)
    hg : MeasureTheory.Integrable g (μ.restrict (HasCompl.compl s))
    ⊢ MeasureTheory.Integrable (s.piecewise f g) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf hg ⊢
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f g : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.Memℒp f 1 (μ.restrict s)
    hg : MeasureTheory.Memℒp g 1 (μ.restrict (HasCompl.compl s))
    ⊢ MeasureTheory.Memℒp (s.piecewise f g) 1 μ
  -/
  exact Memℒp.piecewise hs hf hg
  /-
    🎉 no goals
  -/


theorem IntegrableOn.left_of_union (h : IntegrableOn f (s ∪ t) μ) : IntegrableOn f s μ :=
  h.mono_set subset_union_left


theorem IntegrableOn.right_of_union (h : IntegrableOn f (s ∪ t) μ) : IntegrableOn f t μ :=
  h.mono_set subset_union_right


theorem IntegrableOn.union (hs : IntegrableOn f s μ) (ht : IntegrableOn f t μ) :
    IntegrableOn f (s ∪ t) μ :=
  (hs.add_measure ht).mono_measure <| Measure.restrict_union_le _ _


@[simp]
theorem integrableOn_union : IntegrableOn f (s ∪ t) μ ↔ IntegrableOn f s μ ∧ IntegrableOn f t μ :=
  ⟨fun h => ⟨h.left_of_union, h.right_of_union⟩, fun h => h.1.union h.2⟩


@[simp]
theorem integrableOn_singleton_iff {x : α} [MeasurableSingletonClass α] :
    IntegrableOn f {x} μ ↔ f x = 0 ∨ μ {x} < ∞ := by
  have : f =ᵐ[μ.restrict {x}] fun _ => f x := by
    filter_upwards [ae_restrict_mem (measurableSet_singleton x)] with _ ha
    simp only [mem_singleton_iff.1 ha]
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    x : α
    inst✝ : MeasurableSingletonClass α
    this : (MeasureTheory.ae (μ.restrict (Singleton.singleton x))).EventuallyEq f  …
    ⊢ Iff (MeasureTheory.IntegrableOn f (Singleton.singleton x) μ) (Or (Eq (f x) 0 …
  -/
  rw [IntegrableOn, integrable_congr this, integrable_const_iff]
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    x : α
    inst✝ : MeasurableSingletonClass α
    this : (MeasureTheory.ae (μ.restrict (Singleton.singleton x))).EventuallyEq f  …
    ⊢ Iff (Or (Eq (f x) 0) (LT.lt ((μ.restrict (Singleton.singleton x)) Set.univ)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem integrableOn_finite_biUnion {s : Set β} (hs : s.Finite) {t : β → Set α} :
    IntegrableOn f (⋃ i ∈ s, t i) μ ↔ ∀ i ∈ s, IntegrableOn f (t i) μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    s : Set β
    hs : s.Finite
    t : β → Set α
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.iUnion fun i => Set.iUnion fun h => t …
  -/
  refine hs.induction_on ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      f : α → E
      μ : MeasureTheory.Measure α
      s : Set β
      hs : s.Finite
      t : β → Set α
      ⊢ Iff (MeasureTheory.IntegrableOn f (Set.iUnion fun i => Set.iUnion fun h => t …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      f : α → E
      μ : MeasureTheory.Measure α
      s : Set β
      hs : s.Finite
      t : β → Set α
      ⊢ ∀ {a : β} {s : Set β}, Not (Membership.mem s a) → s.Finite → Iff (MeasureThe …
    -/
  · intro a s _ _ hf; simp [hf, or_imp, forall_and]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem integrableOn_finset_iUnion {s : Finset β} {t : β → Set α} :
    IntegrableOn f (⋃ i ∈ s, t i) μ ↔ ∀ i ∈ s, IntegrableOn f (t i) μ :=
  integrableOn_finite_biUnion s.finite_toSet


@[simp]
theorem integrableOn_finite_iUnion [Finite β] {t : β → Set α} :
    IntegrableOn f (⋃ i, t i) μ ↔ ∀ i, IntegrableOn f (t i) μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    inst✝ : Finite β
    t : β → Set α
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.iUnion fun i => t i) μ) (∀ (i : β), M …
  -/
  cases nonempty_fintype β
  /-
    case intro
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    inst✝ : Finite β
    t : β → Set α
    val✝ : Fintype β
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.iUnion fun i => t i) μ) (∀ (i : β), M …
  -/
  simpa using @integrableOn_finset_iUnion _ _ _ _ _ f μ Finset.univ t
  /-
    🎉 no goals
  -/


lemma IntegrableOn.finset [MeasurableSingletonClass α] {μ : Measure α} [IsFiniteMeasure μ]
    {s : Finset α} {f : α → E} : IntegrableOn f s μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Finset α
    f : α → E
    ⊢ MeasureTheory.IntegrableOn f (↑s) μ
  -/
  rw [← s.toSet.biUnion_of_singleton]
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Finset α
    f : α → E
    ⊢ MeasureTheory.IntegrableOn f (Set.iUnion fun x => Set.iUnion fun h => Single …
  -/
  simp [integrableOn_finset_iUnion, measure_lt_top]
  /-
    🎉 no goals
  -/


lemma IntegrableOn.of_finite [MeasurableSingletonClass α] {μ : Measure α} [IsFiniteMeasure μ]
    {s : Set α} (hs : s.Finite) {f : α → E} : IntegrableOn f s μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set α
    hs : s.Finite
    f : α → E
    ⊢ MeasureTheory.IntegrableOn f s μ
  -/
  simpa using IntegrableOn.finset (s := hs.toFinset)
  /-
    🎉 no goals
  -/


theorem IntegrableOn.add_measure (hμ : IntegrableOn f s μ) (hν : IntegrableOn f s ν) :
    IntegrableOn f s (μ + ν) := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ ν : MeasureTheory.Measure α
    hμ : MeasureTheory.IntegrableOn f s μ
    hν : MeasureTheory.IntegrableOn f s ν
    ⊢ MeasureTheory.IntegrableOn f s (HAdd.hAdd μ ν)
  -/
  delta IntegrableOn; rw [Measure.restrict_add]; exact hμ.integrable.add_measure hν
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem integrableOn_add_measure :
    IntegrableOn f s (μ + ν) ↔ IntegrableOn f s μ ∧ IntegrableOn f s ν :=
  ⟨fun h =>
    ⟨h.mono_measure (Measure.le_add_right le_rfl), h.mono_measure (Measure.le_add_left le_rfl)⟩,
    fun h => h.1.add_measure h.2⟩


theorem _root_.MeasurableEmbedding.integrableOn_map_iff [MeasurableSpace β] {e : α → β}
    (he : MeasurableEmbedding e) {f : β → E} {μ : Measure α} {s : Set β} :
    IntegrableOn f s (μ.map e) ↔ IntegrableOn (f ∘ e) (e ⁻¹' s) μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasurableSpace β
    e : α → β
    he : MeasurableEmbedding e
    f : β → E
    μ : MeasureTheory.Measure α
    s : Set β
    ⊢ Iff (MeasureTheory.IntegrableOn f s (MeasureTheory.Measure.map e μ)) (Measur …
  -/
  simp_rw [IntegrableOn, he.restrict_map, he.integrable_map_iff]
  /-
    🎉 no goals
  -/


theorem _root_.MeasurableEmbedding.integrableOn_iff_comap [MeasurableSpace β] {e : α → β}
    (he : MeasurableEmbedding e) {f : β → E} {μ : Measure β} {s : Set β} (hs : s ⊆ range e) :
    IntegrableOn f s μ ↔ IntegrableOn (f ∘ e) (e ⁻¹' s) (μ.comap e) := by
  simp_rw [← he.integrableOn_map_iff, he.map_comap, IntegrableOn,
    Measure.restrict_restrict_of_subset hs]


theorem integrableOn_map_equiv [MeasurableSpace β] (e : α ≃ᵐ β) {f : β → E} {μ : Measure α}
    {s : Set β} : IntegrableOn f s (μ.map e) ↔ IntegrableOn (f ∘ e) (e ⁻¹' s) μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    f : β → E
    μ : MeasureTheory.Measure α
    s : Set β
    ⊢ Iff (MeasureTheory.IntegrableOn f s (MeasureTheory.Measure.map (⇑e) μ)) (Mea …
  -/
  simp only [IntegrableOn, e.restrict_map, integrable_map_equiv e]
  /-
    🎉 no goals
  -/


theorem MeasurePreserving.integrableOn_comp_preimage [MeasurableSpace β] {e : α → β} {ν}
    (h₁ : MeasurePreserving e μ ν) (h₂ : MeasurableEmbedding e) {f : β → E} {s : Set β} :
    IntegrableOn (f ∘ e) (e ⁻¹' s) μ ↔ IntegrableOn f s ν :=
  (h₁.restrict_preimage_emb h₂ s).integrable_comp_emb h₂


theorem MeasurePreserving.integrableOn_image [MeasurableSpace β] {e : α → β} {ν}
    (h₁ : MeasurePreserving e μ ν) (h₂ : MeasurableEmbedding e) {f : β → E} {s : Set α} :
    IntegrableOn f (e '' s) ν ↔ IntegrableOn (f ∘ e) s μ :=
  ((h₁.restrict_image_emb h₂ s).integrable_comp_emb h₂).symm


theorem integrable_indicator_iff (hs : MeasurableSet s) :
    Integrable (indicator s f) μ ↔ IntegrableOn f s μ := by
  simp_rw [IntegrableOn, Integrable, hasFiniteIntegral_iff_nnnorm,
    nnnorm_indicator_eq_indicator_nnnorm, ENNReal.coe_indicator, lintegral_indicator hs,
    aestronglyMeasurable_indicator_iff hs]


theorem IntegrableOn.integrable_indicator (h : IntegrableOn f s μ) (hs : MeasurableSet s) :
    Integrable (indicator s f) μ :=
  (integrable_indicator_iff hs).2 h


theorem Integrable.indicator (h : Integrable f μ) (hs : MeasurableSet s) :
    Integrable (indicator s f) μ :=
  h.integrableOn.integrable_indicator hs


theorem IntegrableOn.indicator (h : IntegrableOn f s μ) (ht : MeasurableSet t) :
    IntegrableOn (indicator t f) s μ :=
  Integrable.indicator h ht


theorem integrable_indicatorConstLp {E} [NormedAddCommGroup E] {p : ℝ≥0∞} {s : Set α}
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) :
    Integrable (indicatorConstLp p hs hμs c) μ := by
  rw [integrable_congr indicatorConstLp_coeFn, integrable_indicator_iff hs, IntegrableOn,
    integrable_const_iff, lt_top_iff_ne_top]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Or (Eq c 0) (Ne ((μ.restrict s) Set.univ) Top.top)
  -/
  right
  /-
    case h
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Ne ((μ.restrict s) Set.univ) Top.top
  -/
  simpa only [Set.univ_inter, MeasurableSet.univ, Measure.restrict_apply] using hμs
  /-
    🎉 no goals
  -/


/-- If a function is integrable on a set `s` and nonzero there, then the measurable hull of `s` is
well behaved: the restriction of the measure to `toMeasurable μ s` coincides with its restriction
to `s`. -/
theorem IntegrableOn.restrict_toMeasurable (hf : IntegrableOn f s μ) (h's : ∀ x ∈ s, f x ≠ 0) :
    μ.restrict (toMeasurable μ s) = μ.restrict s := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    ⊢ Eq (μ.restrict (MeasureTheory.toMeasurable μ s)) (μ.restrict s)
  -/
  rcases exists_seq_strictAnti_tendsto (0 : ℝ) with ⟨u, _, u_pos, u_lim⟩
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ Eq (μ.restrict (MeasureTheory.toMeasurable μ s)) (μ.restrict s)
  -/
  let v n := toMeasurable (μ.restrict s) { x | u n ≤ ‖f x‖ }
  have A : ∀ n, μ (s ∩ v n) ≠ ∞ := by
    intro n
    rw [inter_comm, ← Measure.restrict_apply (measurableSet_toMeasurable _ _),
      measure_toMeasurable]
    exact (hf.measure_norm_ge_lt_top (u_pos n)).ne
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set α := fun n => MeasureTheory.toMeasurable (μ.restrict s) (setOf f …
    A : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
    ⊢ Eq (μ.restrict (MeasureTheory.toMeasurable μ s)) (μ.restrict s)
  -/
  apply Measure.restrict_toMeasurable_of_cover _ A
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set α := fun n => MeasureTheory.toMeasurable (μ.restrict s) (setOf f …
    A : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
    ⊢ HasSubset.Subset s (Set.iUnion fun n => v n)
  -/
  intro x hx
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set α := fun n => MeasureTheory.toMeasurable (μ.restrict s) (setOf f …
    A : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem (Set.iUnion fun n => v n) x
  -/
  have : 0 < ‖f x‖ := by simp only [h's x hx, norm_pos_iff, Ne, not_false_iff]
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set α := fun n => MeasureTheory.toMeasurable (μ.restrict s) (setOf f …
    A : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
    x : α
    hx : Membership.mem s x
    this : LT.lt 0 (Norm.norm (f x))
    ⊢ Membership.mem (Set.iUnion fun n => v n) x
  -/
  obtain ⟨n, hn⟩ : ∃ n, u n < ‖f x‖ := ((tendsto_order.1 u_lim).2 _ this).exists
  /-
    case intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h's : ∀ (x : α), Membership.mem s x → Ne (f x) 0
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set α := fun n => MeasureTheory.toMeasurable (μ.restrict s) (setOf f …
    A : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
    x : α
    hx : Membership.mem s x
    this : LT.lt 0 (Norm.norm (f x))
    n : Nat
    hn : LT.lt (u n) (Norm.norm (f x))
    ⊢ Membership.mem (Set.iUnion fun n => v n) x
  -/
  exact mem_iUnion.2 ⟨n, subset_toMeasurable _ _ hn.le⟩
  /-
    🎉 no goals
  -/


/-- If a function is integrable on a set `s`, and vanishes on `t \ s`, then it is integrable on `t`
if `t` is null-measurable. -/
theorem IntegrableOn.of_ae_diff_eq_zero (hf : IntegrableOn f s μ) (ht : NullMeasurableSet t μ)
    (h't : ∀ᵐ x ∂μ, x ∈ t \ s → f x = 0) : IntegrableOn f t μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    ⊢ MeasureTheory.IntegrableOn f t μ
  -/
  let u := { x ∈ s | f x ≠ 0 }
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    u : Set α := setOf fun x => And (Membership.mem s x) (Ne (f x) 0)
    ⊢ MeasureTheory.IntegrableOn f t μ
  -/
  have hu : IntegrableOn f u μ := hf.mono_set fun x hx => hx.1
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    u : Set α := setOf fun x => And (Membership.mem s x) (Ne (f x) 0)
    hu : MeasureTheory.IntegrableOn f u μ
    ⊢ MeasureTheory.IntegrableOn f t μ
  -/
  let v := toMeasurable μ u
  have A : IntegrableOn f v μ := by
    rw [IntegrableOn, hu.restrict_toMeasurable]
    · exact hu
    · intro x hx; exact hx.2
  have B : IntegrableOn f (t \ v) μ := by
    apply integrableOn_zero.congr
    filter_upwards [ae_restrict_of_ae h't,
      ae_restrict_mem₀ (ht.diff (measurableSet_toMeasurable μ u).nullMeasurableSet)] with x hxt hx
    by_cases h'x : x ∈ s
    · by_contra H
      exact hx.2 (subset_toMeasurable μ u ⟨h'x, Ne.symm H⟩)
    · exact (hxt ⟨hx.1, h'x⟩).symm
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    u : Set α := setOf fun x => And (Membership.mem s x) (Ne (f x) 0)
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := MeasureTheory.toMeasurable μ u
    A : MeasureTheory.IntegrableOn f v μ
    B : MeasureTheory.IntegrableOn f (SDiff.sdiff t v) μ
    ⊢ MeasureTheory.IntegrableOn f t μ
  -/
  apply (A.union B).mono_set _
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    u : Set α := setOf fun x => And (Membership.mem s x) (Ne (f x) 0)
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := MeasureTheory.toMeasurable μ u
    A : MeasureTheory.IntegrableOn f v μ
    B : MeasureTheory.IntegrableOn f (SDiff.sdiff t v) μ
    ⊢ HasSubset.Subset t (Union.union v (SDiff.sdiff t v))
  -/
  rw [union_diff_self]
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s t : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    u : Set α := setOf fun x => And (Membership.mem s x) (Ne (f x) 0)
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := MeasureTheory.toMeasurable μ u
    A : MeasureTheory.IntegrableOn f v μ
    B : MeasureTheory.IntegrableOn f (SDiff.sdiff t v) μ
    ⊢ HasSubset.Subset t (Union.union v t)
  -/
  exact subset_union_right
  /-
    🎉 no goals
  -/


/-- If a function is integrable on a set `s`, and vanishes on `t \ s`, then it is integrable on `t`
if `t` is measurable. -/
theorem IntegrableOn.of_forall_diff_eq_zero (hf : IntegrableOn f s μ) (ht : MeasurableSet t)
    (h't : ∀ x ∈ t \ s, f x = 0) : IntegrableOn f t μ :=
  hf.of_ae_diff_eq_zero ht.nullMeasurableSet (Eventually.of_forall h't)


/-- If a function is integrable on a set `s` and vanishes almost everywhere on its complement,
then it is integrable. -/
theorem IntegrableOn.integrable_of_ae_not_mem_eq_zero (hf : IntegrableOn f s μ)
    (h't : ∀ᵐ x ∂μ, x ∉ s → f x = 0) : Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h't : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Meas …
    ⊢ MeasureTheory.Integrable f μ
  -/
  rw [← integrableOn_univ]
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h't : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Meas …
    ⊢ MeasureTheory.IntegrableOn f Set.univ μ
  -/
  apply hf.of_ae_diff_eq_zero nullMeasurableSet_univ
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.IntegrableOn f s μ
    h't : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Meas …
    ⊢ Filter.Eventually (fun x => Membership.mem (SDiff.sdiff Set.univ s) x → Eq ( …
  -/
  filter_upwards [h't] with x hx h'x using hx h'x.2
  /-
    🎉 no goals
  -/


/-- If a function is integrable on a set `s` and vanishes everywhere on its complement,
then it is integrable. -/
theorem IntegrableOn.integrable_of_forall_not_mem_eq_zero (hf : IntegrableOn f s μ)
    (h't : ∀ x, x ∉ s → f x = 0) : Integrable f μ :=
  hf.integrable_of_ae_not_mem_eq_zero (Eventually.of_forall fun x hx => h't x hx)


theorem integrableOn_iff_integrable_of_support_subset (h1s : support f ⊆ s) :
    IntegrableOn f s μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    h1s : HasSubset.Subset (Function.support f) s
    ⊢ Iff (MeasureTheory.IntegrableOn f s μ) (MeasureTheory.Integrable f μ)
  -/
  refine ⟨fun h => ?_, fun h => h.integrableOn⟩
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    h1s : HasSubset.Subset (Function.support f) s
    h : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine h.integrable_of_forall_not_mem_eq_zero fun x hx => ?_
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    h1s : HasSubset.Subset (Function.support f) s
    h : MeasureTheory.IntegrableOn f s μ
    x : α
    hx : Not (Membership.mem s x)
    ⊢ Eq (f x) 0
  -/
  contrapose! hx
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    μ : MeasureTheory.Measure α
    h1s : HasSubset.Subset (Function.support f) s
    h : MeasureTheory.IntegrableOn f s μ
    x : α
    hx : Ne (f x) 0
    ⊢ Membership.mem s x
  -/
  exact h1s (mem_support.2 hx)
  /-
    🎉 no goals
  -/


theorem integrableOn_Lp_of_measure_ne_top {E} [NormedAddCommGroup E] {p : ℝ≥0∞} {s : Set α}
    (f : Lp E p μ) (hp : 1 ≤ p) (hμs : μ s ≠ ∞) : IntegrableOn f s μ := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hp : LE.le 1 p
    hμs : Ne (μ s) Top.top
    ⊢ MeasureTheory.IntegrableOn (↑↑f) s μ
  -/
  refine memℒp_one_iff_integrable.mp ?_
  have hμ_restrict_univ : (μ.restrict s) Set.univ < ∞ := by
    simpa only [Set.univ_inter, MeasurableSet.univ, Measure.restrict_apply, lt_top_iff_ne_top]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hp : LE.le 1 p
    hμs : Ne (μ s) Top.top
    hμ_restrict_univ : LT.lt ((μ.restrict s) Set.univ) Top.top
    ⊢ MeasureTheory.Memℒp (↑↑f) 1 (μ.restrict s)
  -/
  haveI hμ_finite : IsFiniteMeasure (μ.restrict s) := ⟨hμ_restrict_univ⟩
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hp : LE.le 1 p
    hμs : Ne (μ s) Top.top
    hμ_restrict_univ : LT.lt ((μ.restrict s) Set.univ) Top.top
    hμ_finite : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ MeasureTheory.Memℒp (↑↑f) 1 (μ.restrict s)
  -/
  exact ((Lp.memℒp _).restrict s).memℒp_of_exponent_le hp
  /-
    🎉 no goals
  -/


theorem Integrable.lintegral_lt_top {f : α → ℝ} (hf : Integrable f μ) :
    (∫⁻ x, ENNReal.ofReal (f x) ∂μ) < ∞ :=
  calc
    (∫⁻ x, ENNReal.ofReal (f x) ∂μ) ≤ ∫⁻ x, ↑‖f x‖₊ ∂μ := lintegral_ofReal_le_lintegral_nnnorm f
    _ < ∞ := hf.2


theorem IntegrableOn.setLIntegral_lt_top {f : α → ℝ} {s : Set α} (hf : IntegrableOn f s μ) :
    (∫⁻ x in s, ENNReal.ofReal (f x) ∂μ) < ∞ :=
  Integrable.lintegral_lt_top hf


@[deprecated (since := "2024-06-29")]
alias IntegrableOn.set_lintegral_lt_top := IntegrableOn.setLIntegral_lt_top


/-- We say that a function `f` is *integrable at filter* `l` if it is integrable on some
set `s ∈ l`. Equivalently, it is eventually integrable on `s` in `l.smallSets`. -/
def IntegrableAtFilter (f : α → ε) (l : Filter α) (μ : Measure α := by volume_tac) :=
  ∃ s ∈ l, IntegrableOn f s μ


theorem _root_.MeasurableEmbedding.integrableAtFilter_map_iff [MeasurableSpace β] {e : α → β}
    (he : MeasurableEmbedding e) {f : β → E} :
    IntegrableAtFilter f (l.map e) (μ.map e) ↔ IntegrableAtFilter (f ∘ e) l μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    inst✝ : MeasurableSpace β
    e : α → β
    he : MeasurableEmbedding e
    f : β → E
    ⊢ Iff (MeasureTheory.IntegrableAtFilter f (Filter.map e l) (MeasureTheory.Meas …
  -/
  simp_rw [IntegrableAtFilter, he.integrableOn_map_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    inst✝ : MeasurableSpace β
    e : α → β
    he : MeasurableEmbedding e
    f : β → E
    ⊢ Iff (Exists fun s => And (Membership.mem (Filter.map e l) s) (MeasureTheory. …
  -/
  constructor <;> rintro ⟨s, hs⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      l : Filter α
      inst✝ : MeasurableSpace β
      e : α → β
      he : MeasurableEmbedding e
      f : β → E
      s : Set β
      hs : And (Membership.mem (Filter.map e l) s) (MeasureTheory.IntegrableOn (Func …
      ⊢ Exists fun s => And (Membership.mem l s) (MeasureTheory.IntegrableOn (Functi …
    -/
  · exact ⟨_, hs⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      l : Filter α
      inst✝ : MeasurableSpace β
      e : α → β
      he : MeasurableEmbedding e
      f : β → E
      s : Set α
      hs : And (Membership.mem l s) (MeasureTheory.IntegrableOn (Function.comp f e)  …
      ⊢ Exists fun s => And (Membership.mem (Filter.map e l) s) (MeasureTheory.Integ …
    -/
  · exact ⟨e '' s, by rwa [mem_map, he.injective.preimage_image]⟩
    /-
      🎉 no goals
    -/


theorem _root_.MeasurableEmbedding.integrableAtFilter_iff_comap [MeasurableSpace β] {e : α → β}
    (he : MeasurableEmbedding e) {f : β → E} {μ : Measure β} :
    IntegrableAtFilter f (l.map e) μ ↔ IntegrableAtFilter (f ∘ e) l (μ.comap e) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    l : Filter α
    inst✝ : MeasurableSpace β
    e : α → β
    he : MeasurableEmbedding e
    f : β → E
    μ : MeasureTheory.Measure β
    ⊢ Iff (MeasureTheory.IntegrableAtFilter f (Filter.map e l) μ) (MeasureTheory.I …
  -/
  simp_rw [← he.integrableAtFilter_map_iff, IntegrableAtFilter, he.map_comap]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    l : Filter α
    inst✝ : MeasurableSpace β
    e : α → β
    he : MeasurableEmbedding e
    f : β → E
    μ : MeasureTheory.Measure β
    ⊢ Iff (Exists fun s => And (Membership.mem (Filter.map e l) s) (MeasureTheory. …
  -/
  constructor <;> rintro ⟨s, hs, int⟩
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      l : Filter α
      inst✝ : MeasurableSpace β
      e : α → β
      he : MeasurableEmbedding e
      f : β → E
      μ : MeasureTheory.Measure β
      s : Set β
      hs : Membership.mem (Filter.map e l) s
      int : MeasureTheory.IntegrableOn f s μ
      ⊢ Exists fun s => And (Membership.mem (Filter.map e l) s) (MeasureTheory.Integ …
    -/
  · exact ⟨s, hs, int.mono_measure <| μ.restrict_le_self⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      E : Type u_4
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      l : Filter α
      inst✝ : MeasurableSpace β
      e : α → β
      he : MeasurableEmbedding e
      f : β → E
      μ : MeasureTheory.Measure β
      s : Set β
      hs : Membership.mem (Filter.map e l) s
      int : MeasureTheory.IntegrableOn f s (μ.restrict (Set.range e))
      ⊢ Exists fun s => And (Membership.mem (Filter.map e l) s) (MeasureTheory.Integ …
    -/
  · exact ⟨_, inter_mem hs range_mem_map, int.inter_of_restrict⟩
    /-
      🎉 no goals
    -/


theorem Integrable.integrableAtFilter (h : Integrable f μ) (l : Filter α) :
    IntegrableAtFilter f l μ :=
  ⟨univ, Filter.univ_mem, integrableOn_univ.2 h⟩


protected theorem IntegrableAtFilter.eventually (h : IntegrableAtFilter f l μ) :
    ∀ᶠ s in l.smallSets, IntegrableOn f s μ :=
  Iff.mpr (eventually_smallSets' fun _s _t hst ht => ht.mono_set hst) h


protected theorem IntegrableAtFilter.add {f g : α → E}
    (hf : IntegrableAtFilter f l μ) (hg : IntegrableAtFilter g l μ) :
    IntegrableAtFilter (f + g) l μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    hf : MeasureTheory.IntegrableAtFilter f l μ
    hg : MeasureTheory.IntegrableAtFilter g l μ
    ⊢ MeasureTheory.IntegrableAtFilter (HAdd.hAdd f g) l μ
  -/
  rcases hf with ⟨s, sl, hs⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    hg : MeasureTheory.IntegrableAtFilter g l μ
    s : Set α
    sl : Membership.mem l s
    hs : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.IntegrableAtFilter (HAdd.hAdd f g) l μ
  -/
  rcases hg with ⟨t, tl, ht⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    s : Set α
    sl : Membership.mem l s
    hs : MeasureTheory.IntegrableOn f s μ
    t : Set α
    tl : Membership.mem l t
    ht : MeasureTheory.IntegrableOn g t μ
    ⊢ MeasureTheory.IntegrableAtFilter (HAdd.hAdd f g) l μ
  -/
  refine ⟨s ∩ t, inter_mem sl tl, ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    s : Set α
    sl : Membership.mem l s
    hs : MeasureTheory.IntegrableOn f s μ
    t : Set α
    tl : Membership.mem l t
    ht : MeasureTheory.IntegrableOn g t μ
    ⊢ MeasureTheory.IntegrableOn (HAdd.hAdd f g) (Inter.inter s t) μ
  -/
  exact (hs.mono_set inter_subset_left).add (ht.mono_set inter_subset_right)
  /-
    🎉 no goals
  -/


protected theorem IntegrableAtFilter.neg {f : α → E} (hf : IntegrableAtFilter f l μ) :
    IntegrableAtFilter (-f) l μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f : α → E
    hf : MeasureTheory.IntegrableAtFilter f l μ
    ⊢ MeasureTheory.IntegrableAtFilter (Neg.neg f) l μ
  -/
  rcases hf with ⟨s, sl, hs⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f : α → E
    s : Set α
    sl : Membership.mem l s
    hs : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.IntegrableAtFilter (Neg.neg f) l μ
  -/
  exact ⟨s, sl, hs.neg⟩
  /-
    🎉 no goals
  -/


protected theorem IntegrableAtFilter.sub {f g : α → E}
    (hf : IntegrableAtFilter f l μ) (hg : IntegrableAtFilter g l μ) :
    IntegrableAtFilter (f - g) l μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    hf : MeasureTheory.IntegrableAtFilter f l μ
    hg : MeasureTheory.IntegrableAtFilter g l μ
    ⊢ MeasureTheory.IntegrableAtFilter (HSub.hSub f g) l μ
  -/
  rw [sub_eq_add_neg]
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    f g : α → E
    hf : MeasureTheory.IntegrableAtFilter f l μ
    hg : MeasureTheory.IntegrableAtFilter g l μ
    ⊢ MeasureTheory.IntegrableAtFilter (HAdd.hAdd f (Neg.neg g)) l μ
  -/
  exact hf.add hg.neg
  /-
    🎉 no goals
  -/


protected theorem IntegrableAtFilter.smul {𝕜 : Type*} [NormedAddCommGroup 𝕜] [SMulZeroClass 𝕜 E]
    [BoundedSMul 𝕜 E] {f : α → E} (hf : IntegrableAtFilter f l μ) (c : 𝕜) :
    IntegrableAtFilter (c • f) l μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝⁴ : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    𝕜 : Type u_6
    inst✝² : NormedAddCommGroup 𝕜
    inst✝¹ : SMulZeroClass 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → E
    hf : MeasureTheory.IntegrableAtFilter f l μ
    c : 𝕜
    ⊢ MeasureTheory.IntegrableAtFilter (HSMul.hSMul c f) l μ
  -/
  rcases hf with ⟨s, sl, hs⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_4
    inst✝⁴ : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    l : Filter α
    𝕜 : Type u_6
    inst✝² : NormedAddCommGroup 𝕜
    inst✝¹ : SMulZeroClass 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → E
    c : 𝕜
    s : Set α
    sl : Membership.mem l s
    hs : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.IntegrableAtFilter (HSMul.hSMul c f) l μ
  -/
  exact ⟨s, sl, hs.smul c⟩
  /-
    🎉 no goals
  -/


protected theorem IntegrableAtFilter.norm (hf : IntegrableAtFilter f l μ) :
    IntegrableAtFilter (fun x => ‖f x‖) l μ :=
  Exists.casesOn hf fun s hs ↦ ⟨s, hs.1, hs.2.norm⟩


theorem IntegrableAtFilter.filter_mono (hl : l ≤ l') (hl' : IntegrableAtFilter f l' μ) :
    IntegrableAtFilter f l μ :=
  let ⟨s, hs, hsf⟩ := hl'
  ⟨s, hl hs, hsf⟩


theorem IntegrableAtFilter.inf_of_left (hl : IntegrableAtFilter f l μ) :
    IntegrableAtFilter f (l ⊓ l') μ :=
  hl.filter_mono inf_le_left


theorem IntegrableAtFilter.inf_of_right (hl : IntegrableAtFilter f l μ) :
    IntegrableAtFilter f (l' ⊓ l) μ :=
  hl.filter_mono inf_le_right


@[simp]
theorem IntegrableAtFilter.inf_ae_iff {l : Filter α} :
    IntegrableAtFilter f (l ⊓ ae μ) μ ↔ IntegrableAtFilter f l μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    ⊢ Iff (MeasureTheory.IntegrableAtFilter f (Min.min l (MeasureTheory.ae μ)) μ)  …
  -/
  refine ⟨?_, fun h ↦ h.filter_mono inf_le_left⟩
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    ⊢ MeasureTheory.IntegrableAtFilter f (Min.min l (MeasureTheory.ae μ)) μ → Meas …
  -/
  rintro ⟨s, ⟨t, ht, u, hu, rfl⟩, hf⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    t : Set α
    ht : Membership.mem l t
    u : Set α
    hu : Membership.mem (MeasureTheory.ae μ) u
    hf : MeasureTheory.IntegrableOn f (Inter.inter t u) μ
    ⊢ MeasureTheory.IntegrableAtFilter f l μ
  -/
  refine ⟨t, ht, hf.congr_set_ae <| eventuallyEq_set.2 ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    t : Set α
    ht : Membership.mem l t
    u : Set α
    hu : Membership.mem (MeasureTheory.ae μ) u
    hf : MeasureTheory.IntegrableOn f (Inter.inter t u) μ
    ⊢ Filter.Eventually (fun x => Iff (Membership.mem t x) (Membership.mem (Inter. …
  -/
  filter_upwards [hu] with x hx using (and_iff_left hx).symm
  /-
    🎉 no goals
  -/


alias ⟨IntegrableAtFilter.of_inf_ae, _⟩ := IntegrableAtFilter.inf_ae_iff


@[simp]
theorem integrableAtFilter_top : IntegrableAtFilter f ⊤ μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.IntegrableAtFilter f Top.top μ) (MeasureTheory.Integrable …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.integrableAtFilter ⊤⟩
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    h : MeasureTheory.IntegrableAtFilter f Top.top μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  obtain ⟨s, hsf, hs⟩ := h
  /-
    case intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    s : Set α
    hsf : Membership.mem Top.top s
    hs : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  exact (integrableOn_iff_integrable_of_support_subset fun _ _ ↦ hsf _).mp hs
  /-
    🎉 no goals
  -/


theorem IntegrableAtFilter.sup_iff {l l' : Filter α} :
    IntegrableAtFilter f (l ⊔ l') μ ↔ IntegrableAtFilter f l μ ∧ IntegrableAtFilter f l' μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l l' : Filter α
    ⊢ Iff (MeasureTheory.IntegrableAtFilter f (Max.max l l') μ) (And (MeasureTheor …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      f : α → E
      μ : MeasureTheory.Measure α
      l l' : Filter α
      ⊢ MeasureTheory.IntegrableAtFilter f (Max.max l l') μ → And (MeasureTheory.Int …
    -/
  · exact fun h => ⟨h.filter_mono le_sup_left, h.filter_mono le_sup_right⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      f : α → E
      μ : MeasureTheory.Measure α
      l l' : Filter α
      ⊢ And (MeasureTheory.IntegrableAtFilter f l μ) (MeasureTheory.IntegrableAtFilt …
    -/
  · exact fun ⟨⟨s, hsl, hs⟩, ⟨t, htl, ht⟩⟩ ↦ ⟨s ∪ t, union_mem_sup hsl htl, hs.union ht⟩
    /-
      🎉 no goals
    -/


/-- If `μ` is a measure finite at filter `l` and `f` is a function such that its norm is bounded
above at `l`, then `f` is integrable at `l`. -/
theorem Measure.FiniteAtFilter.integrableAtFilter {l : Filter α} [IsMeasurablyGenerated l]
    (hfm : StronglyMeasurableAtFilter f l μ) (hμ : μ.FiniteAtFilter l)
    (hf : l.IsBoundedUnder (· ≤ ·) (norm ∘ f)) : IntegrableAtFilter f l μ := by
  obtain ⟨C, hC⟩ : ∃ C, ∀ᶠ s in l.smallSets, ∀ x ∈ s, ‖f x‖ ≤ C :=
    hf.imp fun C hC => eventually_smallSets.2 ⟨_, hC, fun t => id⟩
  rcases (hfm.eventually.and (hμ.eventually.and hC)).exists_measurable_mem_of_smallSets with
    ⟨s, hsl, hsm, hfm, hμ, hC⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    inst✝ : l.IsMeasurablyGenerated
    hfm✝ : StronglyMeasurableAtFilter f l μ
    hμ✝ : μ.FiniteAtFilter l
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    C : Real
    hC✝ : Filter.Eventually (fun s => ∀ (x : α), Membership.mem s x → LE.le (Norm. …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hμ : LT.lt (μ s) Top.top
    hC : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) C
    ⊢ MeasureTheory.IntegrableAtFilter f l μ
  -/
  refine ⟨s, hsl, ⟨hfm, hasFiniteIntegral_restrict_of_bounded hμ (C := C) ?_⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    inst✝ : l.IsMeasurablyGenerated
    hfm✝ : StronglyMeasurableAtFilter f l μ
    hμ✝ : μ.FiniteAtFilter l
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    C : Real
    hC✝ : Filter.Eventually (fun s => ∀ (x : α), Membership.mem s x → LE.le (Norm. …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hμ : LT.lt (μ s) Top.top
    hC : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) C
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae (μ. …
  -/
  rw [ae_restrict_eq hsm, eventually_inf_principal]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    inst✝ : l.IsMeasurablyGenerated
    hfm✝ : StronglyMeasurableAtFilter f l μ
    hμ✝ : μ.FiniteAtFilter l
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    C : Real
    hC✝ : Filter.Eventually (fun s => ∀ (x : α), Membership.mem s x → LE.le (Norm. …
    s : Set α
    hsl : Membership.mem l s
    hsm : MeasurableSet s
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hμ : LT.lt (μ s) Top.top
    hC : ∀ (x : α), Membership.mem s x → LE.le (Norm.norm (f x)) C
    ⊢ Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x)) C)  …
  -/
  exact Eventually.of_forall hC
  /-
    🎉 no goals
  -/


theorem Measure.FiniteAtFilter.integrableAtFilter_of_tendsto_ae {l : Filter α}
    [IsMeasurablyGenerated l] (hfm : StronglyMeasurableAtFilter f l μ) (hμ : μ.FiniteAtFilter l) {b}
    (hf : Tendsto f (l ⊓ ae μ) (𝓝 b)) : IntegrableAtFilter f l μ :=
  (hμ.inf_of_left.integrableAtFilter (hfm.filter_mono inf_le_left)
      hf.norm.isBoundedUnder_le).of_inf_ae


alias _root_.Filter.Tendsto.integrableAtFilter_ae :=
  Measure.FiniteAtFilter.integrableAtFilter_of_tendsto_ae


theorem Measure.FiniteAtFilter.integrableAtFilter_of_tendsto {l : Filter α}
    [IsMeasurablyGenerated l] (hfm : StronglyMeasurableAtFilter f l μ) (hμ : μ.FiniteAtFilter l) {b}
    (hf : Tendsto f l (𝓝 b)) : IntegrableAtFilter f l μ :=
  hμ.integrableAtFilter hfm hf.norm.isBoundedUnder_le


alias _root_.Filter.Tendsto.integrableAtFilter :=
  Measure.FiniteAtFilter.integrableAtFilter_of_tendsto


lemma Measure.integrableOn_of_bounded (s_finite : μ s ≠ ∞) (f_mble : AEStronglyMeasurable f μ)
    {M : ℝ} (f_bdd : ∀ᵐ a ∂(μ.restrict s), ‖f a‖ ≤ M) :
    IntegrableOn f s μ :=
  ⟨f_mble.restrict, hasFiniteIntegral_restrict_of_bounded (C := M) s_finite.lt_top f_bdd⟩


theorem integrable_add_of_disjoint {f g : α → E} (h : Disjoint (support f) (support g))
    (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    Integrable (f + g) μ ↔ Integrable f μ ∧ Integrable g μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f g : α → E
    h : Disjoint (Function.support f) (Function.support g)
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ Iff (MeasureTheory.Integrable (HAdd.hAdd f g) μ) (And (MeasureTheory.Integra …
  -/
  refine ⟨fun hfg => ⟨?_, ?_⟩, fun h => h.1.add h.2⟩
    /-
      case refine_1
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      h : Disjoint (Function.support f) (Function.support g)
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.StronglyMeasurable g
      hfg : MeasureTheory.Integrable (HAdd.hAdd f g) μ
      ⊢ MeasureTheory.Integrable f μ
    -/
  · rw [← indicator_add_eq_left h]; exact hfg.indicator hf.measurableSet_support
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      f g : α → E
      h : Disjoint (Function.support f) (Function.support g)
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.StronglyMeasurable g
      hfg : MeasureTheory.Integrable (HAdd.hAdd f g) μ
      ⊢ MeasureTheory.Integrable g μ
    -/
  · rw [← indicator_add_eq_right h]; exact hfg.indicator hg.measurableSet_support
                                     /-
                                       🎉 no goals
                                     -/


/-- If a function converges along a filter to a limit `a`, is integrable along this filter, and
all elements of the filter have infinite measure, then the limit has to vanish. -/
lemma IntegrableAtFilter.eq_zero_of_tendsto
    (h : IntegrableAtFilter f l μ) (h' : ∀ s ∈ l, μ s = ∞) {a : E}
    (hf : Tendsto f l (𝓝 a)) : a = 0 := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h : MeasureTheory.IntegrableAtFilter f l μ
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    ⊢ Eq a 0
  -/
  by_contra H
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h : MeasureTheory.IntegrableAtFilter f l μ
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ⊢ False
  -/
  obtain ⟨ε, εpos, hε⟩ : ∃ (ε : ℝ), 0 < ε ∧ ε < ‖a‖ := exists_between (norm_pos_iff.mpr H)
  /-
    case intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h : MeasureTheory.IntegrableAtFilter f l μ
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    ⊢ False
  -/
  rcases h with ⟨u, ul, hu⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    u : Set α
    ul : Membership.mem l u
    hu : MeasureTheory.IntegrableOn f u μ
    ⊢ False
  -/
  let v := u ∩ {b | ε < ‖f b‖}
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    u : Set α
    ul : Membership.mem l u
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := Inter.inter u (setOf fun b => LT.lt ε (Norm.norm (f b)))
    ⊢ False
  -/
  have hv : IntegrableOn f v μ := hu.mono_set inter_subset_left
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    u : Set α
    ul : Membership.mem l u
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := Inter.inter u (setOf fun b => LT.lt ε (Norm.norm (f b)))
    hv : MeasureTheory.IntegrableOn f v μ
    ⊢ False
  -/
  have vl : v ∈ l := inter_mem ul ((tendsto_order.1 hf.norm).1 _ hε)
  have : μ.restrict v v < ∞ := lt_of_le_of_lt (measure_mono inter_subset_right)
    (Integrable.measure_gt_lt_top hv.norm εpos)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    u : Set α
    ul : Membership.mem l u
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := Inter.inter u (setOf fun b => LT.lt ε (Norm.norm (f b)))
    hv : MeasureTheory.IntegrableOn f v μ
    vl : Membership.mem l v
    this : LT.lt ((μ.restrict v) v) Top.top
    ⊢ False
  -/
  have : μ v ≠ ∞ := ne_of_lt (by simpa only [Measure.restrict_apply_self])
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    f : α → E
    μ : MeasureTheory.Measure α
    l : Filter α
    h' : ∀ (s : Set α), Membership.mem l s → Eq (μ s) Top.top
    a : E
    hf : Filter.Tendsto f l (nhds a)
    H : Not (Eq a 0)
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt ε (Norm.norm a)
    u : Set α
    ul : Membership.mem l u
    hu : MeasureTheory.IntegrableOn f u μ
    v : Set α := Inter.inter u (setOf fun b => LT.lt ε (Norm.norm (f b)))
    hv : MeasureTheory.IntegrableOn f v μ
    vl : Membership.mem l v
    this✝ : LT.lt ((μ.restrict v) v) Top.top
    this : Ne (μ v) Top.top
    ⊢ False
  -/
  exact this (h' v vl)
  /-
    🎉 no goals
  -/


/-- A function which is continuous on a set `s` is almost everywhere measurable with respect to
`μ.restrict s`. -/
theorem ContinuousOn.aemeasurable [TopologicalSpace α] [OpensMeasurableSpace α] [MeasurableSpace β]
    [TopologicalSpace β] [BorelSpace β] {f : α → β} {s : Set α} {μ : Measure α}
    (hf : ContinuousOn f s) (hs : MeasurableSet s) : AEMeasurable f (μ.restrict s) := by
  classical
  nontriviality α; inhabit α
  have : (Set.piecewise s f fun _ => f default) =ᵐ[μ.restrict s] f := piecewise_ae_eq_restrict hs
  refine ⟨Set.piecewise s f fun _ => f default, ?_, this.symm⟩
  apply measurable_of_isOpen
  intro t ht
  obtain ⟨u, u_open, hu⟩ : ∃ u : Set α, IsOpen u ∧ f ⁻¹' t ∩ s = u ∩ s :=
    _root_.continuousOn_iff'.1 hf t ht
  rw [piecewise_preimage, Set.ite, hu]
  exact (u_open.measurableSet.inter hs).union ((measurable_const ht.measurableSet).diff hs)


/-- A function which is continuous on a separable set `s` is almost everywhere strongly measurable
with respect to `μ.restrict s`. -/
theorem ContinuousOn.aestronglyMeasurable_of_isSeparable [TopologicalSpace α]
    [PseudoMetrizableSpace α] [OpensMeasurableSpace α] [TopologicalSpace β]
    [PseudoMetrizableSpace β] {f : α → β} {s : Set α} {μ : Measure α} (hf : ContinuousOn f s)
    (hs : MeasurableSet s) (h's : TopologicalSpace.IsSeparable s) :
    AEStronglyMeasurable f (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    h's : TopologicalSpace.IsSeparable s
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  letI := pseudoMetrizableSpacePseudoMetric α
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    h's : TopologicalSpace.IsSeparable s
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    h's : TopologicalSpace.IsSeparable s
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  rw [aestronglyMeasurable_iff_aemeasurable_separable]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    h's : TopologicalSpace.IsSeparable s
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ And (AEMeasurable f (μ.restrict s)) (Exists fun t => And (TopologicalSpace.I …
  -/
  refine ⟨hf.aemeasurable hs, f '' s, hf.isSeparable_image h's, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace.PseudoMetrizableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    h's : TopologicalSpace.IsSeparable s
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.image f s) (f x)) (MeasureTh …
  -/
  exact mem_of_superset (self_mem_ae_restrict hs) (subset_preimage_image _ _)
  /-
    🎉 no goals
  -/


/-- A function which is continuous on a set `s` is almost everywhere strongly measurable with
respect to `μ.restrict s` when either the source space or the target space is second-countable. -/
theorem ContinuousOn.aestronglyMeasurable [TopologicalSpace α] [TopologicalSpace β]
    [h : SecondCountableTopologyEither α β] [OpensMeasurableSpace α] [PseudoMetrizableSpace β]
    {f : α → β} {s : Set α} {μ : Measure α} (hf : ContinuousOn f s) (hs : MeasurableSet s) :
    AEStronglyMeasurable f (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    h : SecondCountableTopologyEither α β
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  borelize β
  refine
    aestronglyMeasurable_iff_aemeasurable_separable.2
      ⟨hf.aemeasurable hs, f '' s, ?_,
        mem_of_superset (self_mem_ae_restrict hs) (subset_preimage_image _ _)⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    h : SecondCountableTopologyEither α β
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : MeasurableSet s
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ TopologicalSpace.IsSeparable (Set.image f s)
  -/
  cases h.out
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      h : SecondCountableTopologyEither α β
      inst✝¹ : OpensMeasurableSpace α
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      f : α → β
      s : Set α
      μ : MeasureTheory.Measure α
      hf : ContinuousOn f s
      hs : MeasurableSet s
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology α
      ⊢ TopologicalSpace.IsSeparable (Set.image f s)
    -/
  · rw [image_eq_range]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      h : SecondCountableTopologyEither α β
      inst✝¹ : OpensMeasurableSpace α
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      f : α → β
      s : Set α
      μ : MeasureTheory.Measure α
      hf : ContinuousOn f s
      hs : MeasurableSet s
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology α
      ⊢ TopologicalSpace.IsSeparable (Set.range fun x => f ↑x)
    -/
    exact isSeparable_range <| continuousOn_iff_continuous_restrict.1 hf
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      h : SecondCountableTopologyEither α β
      inst✝¹ : OpensMeasurableSpace α
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      f : α → β
      s : Set α
      μ : MeasureTheory.Measure α
      hf : ContinuousOn f s
      hs : MeasurableSet s
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology β
      ⊢ TopologicalSpace.IsSeparable (Set.image f s)
    -/
  · exact .of_separableSpace _
    /-
      🎉 no goals
    -/


/-- A function which is continuous on a compact set `s` is almost everywhere strongly measurable
with respect to `μ.restrict s`. -/
theorem ContinuousOn.aestronglyMeasurable_of_isCompact [TopologicalSpace α] [OpensMeasurableSpace α]
    [TopologicalSpace β] [PseudoMetrizableSpace β] {f : α → β} {s : Set α} {μ : Measure α}
    (hf : ContinuousOn f s) (hs : IsCompact s) (h's : MeasurableSet s) :
    AEStronglyMeasurable f (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : IsCompact s
    h's : MeasurableSet s
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  letI := pseudoMetrizableSpacePseudoMetric β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : IsCompact s
    h's : MeasurableSet s
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : IsCompact s
    h's : MeasurableSet s
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  rw [aestronglyMeasurable_iff_aemeasurable_separable]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    s : Set α
    μ : MeasureTheory.Measure α
    hf : ContinuousOn f s
    hs : IsCompact s
    h's : MeasurableSet s
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ And (AEMeasurable f (μ.restrict s)) (Exists fun t => And (TopologicalSpace.I …
  -/
  refine ⟨hf.aemeasurable h's, f '' s, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      f : α → β
      s : Set α
      μ : MeasureTheory.Measure α
      hf : ContinuousOn f s
      hs : IsCompact s
      h's : MeasurableSet s
      this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ TopologicalSpace.IsSeparable (Set.image f s)
    -/
  · exact (hs.image_of_continuousOn hf).isSeparable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      f : α → β
      s : Set α
      μ : MeasureTheory.Measure α
      hf : ContinuousOn f s
      hs : IsCompact s
      h's : MeasurableSet s
      this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.image f s) (f x)) (MeasureTh …
    -/
  · exact mem_of_superset (self_mem_ae_restrict h's) (subset_preimage_image _ _)
    /-
      🎉 no goals
    -/


theorem ContinuousOn.integrableAt_nhdsWithin_of_isSeparable [TopologicalSpace α]
    [PseudoMetrizableSpace α] [OpensMeasurableSpace α] {μ : Measure α} [IsLocallyFiniteMeasure μ]
    {a : α} {t : Set α} {f : α → E} (hft : ContinuousOn f t) (ht : MeasurableSet t)
    (h't : TopologicalSpace.IsSeparable t) (ha : a ∈ t) : IntegrableAtFilter f (𝓝[t] a) μ :=
  haveI : (𝓝[t] a).IsMeasurablyGenerated := ht.nhdsWithin_isMeasurablyGenerated _
  (hft a ha).integrableAtFilter
    ⟨_, self_mem_nhdsWithin, hft.aestronglyMeasurable_of_isSeparable ht h't⟩
    (μ.finiteAt_nhdsWithin _ _)


theorem ContinuousOn.integrableAt_nhdsWithin [TopologicalSpace α]
    [SecondCountableTopologyEither α E] [OpensMeasurableSpace α] {μ : Measure α}
    [IsLocallyFiniteMeasure μ] {a : α} {t : Set α} {f : α → E} (hft : ContinuousOn f t)
    (ht : MeasurableSet t) (ha : a ∈ t) : IntegrableAtFilter f (𝓝[t] a) μ :=
  haveI : (𝓝[t] a).IsMeasurablyGenerated := ht.nhdsWithin_isMeasurablyGenerated _
  (hft a ha).integrableAtFilter ⟨_, self_mem_nhdsWithin, hft.aestronglyMeasurable ht⟩
    (μ.finiteAt_nhdsWithin _ _)


theorem Continuous.integrableAt_nhds [TopologicalSpace α] [SecondCountableTopologyEither α E]
    [OpensMeasurableSpace α] {μ : Measure α} [IsLocallyFiniteMeasure μ] {f : α → E}
    (hf : Continuous f) (a : α) : IntegrableAtFilter f (𝓝 a) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : Continuous f
    a : α
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds a) μ
  -/
  rw [← nhdsWithin_univ]
  /-
    α : Type u_1
    E : Type u_4
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : α → E
    hf : Continuous f
    a : α
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin a Set.univ) μ
  -/
  exact hf.continuousOn.integrableAt_nhdsWithin MeasurableSet.univ (mem_univ a)
  /-
    🎉 no goals
  -/


/-- If a function is continuous on an open set `s`, then it is strongly measurable at the filter
`𝓝 x` for all `x ∈ s` if either the source space or the target space is second-countable. -/
theorem ContinuousOn.stronglyMeasurableAtFilter [TopologicalSpace α] [OpensMeasurableSpace α]
    [TopologicalSpace β] [PseudoMetrizableSpace β] [SecondCountableTopologyEither α β] {f : α → β}
    {s : Set α} {μ : Measure α} (hs : IsOpen s) (hf : ContinuousOn f s) :
    ∀ x ∈ s, StronglyMeasurableAtFilter f (𝓝 x) μ := fun _x hx =>
  ⟨s, IsOpen.mem_nhds hs hx, hf.aestronglyMeasurable hs.measurableSet⟩


theorem ContinuousAt.stronglyMeasurableAtFilter [TopologicalSpace α] [OpensMeasurableSpace α]
    [SecondCountableTopologyEither α E] {f : α → E} {s : Set α} {μ : Measure α} (hs : IsOpen s)
    (hf : ∀ x ∈ s, ContinuousAt f x) : ∀ x ∈ s, StronglyMeasurableAtFilter f (𝓝 x) μ :=
  ContinuousOn.stronglyMeasurableAtFilter hs <| continuousOn_of_forall_continuousAt hf


theorem Continuous.stronglyMeasurableAtFilter [TopologicalSpace α] [OpensMeasurableSpace α]
    [TopologicalSpace β] [PseudoMetrizableSpace β] [SecondCountableTopologyEither α β] {f : α → β}
    (hf : Continuous f) (μ : Measure α) (l : Filter α) : StronglyMeasurableAtFilter f l μ :=
  hf.stronglyMeasurable.stronglyMeasurableAtFilter


/-- If a function is continuous on a measurable set `s`, then it is measurable at the filter
  `𝓝[s] x` for all `x`. -/
theorem ContinuousOn.stronglyMeasurableAtFilter_nhdsWithin {α β : Type*} [MeasurableSpace α]
    [TopologicalSpace α] [OpensMeasurableSpace α] [TopologicalSpace β] [PseudoMetrizableSpace β]
    [SecondCountableTopologyEither α β] {f : α → β} {s : Set α} {μ : Measure α}
    (hf : ContinuousOn f s) (hs : MeasurableSet s) (x : α) :
    StronglyMeasurableAtFilter f (𝓝[s] x) μ :=
  ⟨s, self_mem_nhdsWithin, hf.aestronglyMeasurable hs⟩


theorem integrableOn_Icc_iff_integrableOn_Ioc' (ha : μ {a} ≠ ∞) :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ioc a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : PartialOrder α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    ha : Ne (μ (Singleton.singleton a)) Top.top
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
  -/
  by_cases hab : a ≤ b
  · rw [← Ioc_union_left hab, integrableOn_union,
      eq_true (integrableOn_singleton_iff.mpr <| Or.inr ha.lt_top), and_true]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      ha : Ne (μ (Singleton.singleton a)) Top.top
      hab : Not (LE.le a b)
      ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
    -/
  · rw [Icc_eq_empty hab, Ioc_eq_empty]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      ha : Ne (μ (Singleton.singleton a)) Top.top
      hab : Not (LE.le a b)
      ⊢ Not (LT.lt a b)
    -/
    contrapose! hab
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      ha : Ne (μ (Singleton.singleton a)) Top.top
      hab : LT.lt a b
      ⊢ LE.le a b
    -/
    exact hab.le
    /-
      🎉 no goals
    -/


theorem integrableOn_Icc_iff_integrableOn_Ico' (hb : μ {b} ≠ ∞) :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ico a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : PartialOrder α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    hb : Ne (μ (Singleton.singleton b)) Top.top
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
  -/
  by_cases hab : a ≤ b
  · rw [← Ico_union_right hab, integrableOn_union,
      eq_true (integrableOn_singleton_iff.mpr <| Or.inr hb.lt_top), and_true]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      hb : Ne (μ (Singleton.singleton b)) Top.top
      hab : Not (LE.le a b)
      ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
    -/
  · rw [Icc_eq_empty hab, Ico_eq_empty]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      hb : Ne (μ (Singleton.singleton b)) Top.top
      hab : Not (LE.le a b)
      ⊢ Not (LT.lt a b)
    -/
    contrapose! hab
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      hb : Ne (μ (Singleton.singleton b)) Top.top
      hab : LT.lt a b
      ⊢ LE.le a b
    -/
    exact hab.le
    /-
      🎉 no goals
    -/


theorem integrableOn_Ico_iff_integrableOn_Ioo' (ha : μ {a} ≠ ∞) :
    IntegrableOn f (Ico a b) μ ↔ IntegrableOn f (Ioo a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : PartialOrder α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    ha : Ne (μ (Singleton.singleton a)) Top.top
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Ico a b) μ) (MeasureTheory.Integrable …
  -/
  by_cases hab : a < b
  · rw [← Ioo_union_left hab, integrableOn_union,
      eq_true (integrableOn_singleton_iff.mpr <| Or.inr ha.lt_top), and_true]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      ha : Ne (μ (Singleton.singleton a)) Top.top
      hab : Not (LT.lt a b)
      ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Ico a b) μ) (MeasureTheory.Integrable …
    -/
  · rw [Ioo_eq_empty hab, Ico_eq_empty hab]
    /-
      🎉 no goals
    -/


theorem integrableOn_Ioc_iff_integrableOn_Ioo' (hb : μ {b} ≠ ∞) :
    IntegrableOn f (Ioc a b) μ ↔ IntegrableOn f (Ioo a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : PartialOrder α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    hb : Ne (μ (Singleton.singleton b)) Top.top
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Ioc a b) μ) (MeasureTheory.Integrable …
  -/
  by_cases hab : a < b
  · rw [← Ioo_union_right hab, integrableOn_union,
      eq_true (integrableOn_singleton_iff.mpr <| Or.inr hb.lt_top), and_true]
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝³ : MeasurableSpace α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : PartialOrder α
      inst✝ : MeasurableSingletonClass α
      f : α → E
      μ : MeasureTheory.Measure α
      a b : α
      hb : Ne (μ (Singleton.singleton b)) Top.top
      hab : Not (LT.lt a b)
      ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Ioc a b) μ) (MeasureTheory.Integrable …
    -/
  · rw [Ioo_eq_empty hab, Ioc_eq_empty hab]
    /-
      🎉 no goals
    -/


theorem integrableOn_Icc_iff_integrableOn_Ioo' (ha : μ {a} ≠ ∞) (hb : μ {b} ≠ ∞) :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ioo a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : PartialOrder α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    ha : Ne (μ (Singleton.singleton a)) Top.top
    hb : Ne (μ (Singleton.singleton b)) Top.top
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
  -/
  rw [integrableOn_Icc_iff_integrableOn_Ioc' ha, integrableOn_Ioc_iff_integrableOn_Ioo' hb]
  /-
    🎉 no goals
  -/


theorem integrableOn_Ici_iff_integrableOn_Ioi' (hb : μ {b} ≠ ∞) :
    IntegrableOn f (Ici b) μ ↔ IntegrableOn f (Ioi b) μ := by
  rw [← Ioi_union_left, integrableOn_union,
    eq_true (integrableOn_singleton_iff.mpr <| Or.inr hb.lt_top), and_true]


theorem integrableOn_Iic_iff_integrableOn_Iio' (hb : μ {b} ≠ ∞) :
    IntegrableOn f (Iic b) μ ↔ IntegrableOn f (Iio b) μ := by
  rw [← Iio_union_right, integrableOn_union,
    eq_true (integrableOn_singleton_iff.mpr <| Or.inr hb.lt_top), and_true]


theorem integrableOn_Icc_iff_integrableOn_Ioc :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ioc a b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               a b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton a)) Top.top
                                             -/
  integrableOn_Icc_iff_integrableOn_Ioc' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem integrableOn_Icc_iff_integrableOn_Ico :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ico a b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               a b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton b)) Top.top
                                             -/
  integrableOn_Icc_iff_integrableOn_Ico' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem integrableOn_Ico_iff_integrableOn_Ioo :
    IntegrableOn f (Ico a b) μ ↔ IntegrableOn f (Ioo a b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               a b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton a)) Top.top
                                             -/
  integrableOn_Ico_iff_integrableOn_Ioo' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem integrableOn_Ioc_iff_integrableOn_Ioo :
    IntegrableOn f (Ioc a b) μ ↔ IntegrableOn f (Ioo a b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               a b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton b)) Top.top
                                             -/
  integrableOn_Ioc_iff_integrableOn_Ioo' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem integrableOn_Icc_iff_integrableOn_Ioo :
    IntegrableOn f (Icc a b) μ ↔ IntegrableOn f (Ioo a b) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝⁴ : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : PartialOrder α
    inst✝¹ : MeasurableSingletonClass α
    f : α → E
    μ : MeasureTheory.Measure α
    a b : α
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Icc a b) μ) (MeasureTheory.Integrable …
  -/
  rw [integrableOn_Icc_iff_integrableOn_Ioc, integrableOn_Ioc_iff_integrableOn_Ioo]
  /-
    🎉 no goals
  -/


theorem integrableOn_Ici_iff_integrableOn_Ioi :
    IntegrableOn f (Ici b) μ ↔ IntegrableOn f (Ioi b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton b)) Top.top
                                             -/
  integrableOn_Ici_iff_integrableOn_Ioi' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem integrableOn_Iic_iff_integrableOn_Iio :
    IntegrableOn f (Iic b) μ ↔ IntegrableOn f (Iio b) μ :=
                                             /-
                                               α : Type u_1
                                               E : Type u_4
                                               inst✝⁴ : MeasurableSpace α
                                               inst✝³ : NormedAddCommGroup E
                                               inst✝² : PartialOrder α
                                               inst✝¹ : MeasurableSingletonClass α
                                               f : α → E
                                               μ : MeasureTheory.Measure α
                                               b : α
                                               inst✝ : MeasureTheory.NoAtoms μ
                                               ⊢ Ne (μ (Singleton.singleton b)) Top.top
                                             -/
  integrableOn_Iic_iff_integrableOn_Iio' (by rw [measure_singleton]; exact ENNReal.zero_ne_top)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


