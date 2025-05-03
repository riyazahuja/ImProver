/-- If `ν` is a finite measure, and `s ⊆ α × β` is measurable, then `x ↦ ν { y | (x, y) ∈ s }` is
  a measurable function. `measurable_measure_prod_mk_left` is strictly more general. -/
theorem measurable_measure_prod_mk_left_finite [IsFiniteMeasure ν] {s : Set (α × β)}
    (hs : MeasurableSet s) : Measurable fun x => ν (Prod.mk x ⁻¹' s) := by
  induction s, hs using induction_on_inter generateFrom_prod.symm isPiSystem_prod with
  | empty => simp
  | basic s hs =>
    obtain ⟨s, hs, t, -, rfl⟩ := hs
    classical simpa only [mk_preimage_prod_right_eq_if, measure_if]
      using measurable_const.indicator hs
  | compl s hs ihs =>
    simp_rw [preimage_compl, measure_compl (measurable_prod_mk_left hs) (measure_ne_top ν _)]
    exact ihs.const_sub _
  | iUnion f hfd hfm ihf =>
    have (a : α) : ν (Prod.mk a ⁻¹' ⋃ i, f i) = ∑' i, ν (Prod.mk a ⁻¹' f i) := by
      rw [preimage_iUnion, measure_iUnion]
      exacts [hfd.mono fun _ _ ↦ .preimage _, fun i ↦ measurable_prod_mk_left (hfm i)]
    simpa only [this] using Measurable.ennreal_tsum ihf


/-- If `ν` is an s-finite measure, and `s ⊆ α × β` is measurable, then `x ↦ ν { y | (x, y) ∈ s }`
  is a measurable function. -/
theorem measurable_measure_prod_mk_left [SFinite ν] {s : Set (α × β)} (hs : MeasurableSet s) :
    Measurable fun x => ν (Prod.mk x ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun x => ν (Set.preimage (Prod.mk x) s)
  -/
  rw [← sum_sfiniteSeq ν]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun x => (MeasureTheory.Measure.sum (MeasureTheory.sfiniteSeq ν)) …
  -/
  simp_rw [Measure.sum_apply_of_countable]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun x => tsum fun i => (MeasureTheory.sfiniteSeq ν i) (Set.preima …
  -/
  exact Measurable.ennreal_tsum (fun i ↦ measurable_measure_prod_mk_left_finite hs)
  /-
    🎉 no goals
  -/


/-- If `μ` is a σ-finite measure, and `s ⊆ α × β` is measurable, then `y ↦ μ { x | (x, y) ∈ s }` is
  a measurable function. -/
theorem measurable_measure_prod_mk_right {μ : Measure α} [SFinite μ] {s : Set (α × β)}
    (hs : MeasurableSet s) : Measurable fun y => μ ((fun x => (x, y)) ⁻¹' s) :=
  measurable_measure_prod_mk_left (measurableSet_swap_iff.mpr hs)


theorem Measurable.map_prod_mk_left [SFinite ν] :
    Measurable fun x : α => map (Prod.mk x) ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Measurable fun x => MeasureTheory.Measure.map (Prod.mk x) ν
  -/
  apply measurable_of_measurable_coe; intro s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun b => (MeasureTheory.Measure.map (Prod.mk b) ν) s
  -/
  simp_rw [map_apply measurable_prod_mk_left hs]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun b => ν (Set.preimage (Prod.mk b) s)
  -/
  exact measurable_measure_prod_mk_left hs
  /-
    🎉 no goals
  -/


theorem Measurable.map_prod_mk_right {μ : Measure α} [SFinite μ] :
    Measurable fun y : β => map (fun x : α => (x, y)) μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Measurable fun y => MeasureTheory.Measure.map (fun x => { fst := x, snd := y …
  -/
  apply measurable_of_measurable_coe; intro s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun b => (MeasureTheory.Measure.map (fun x => { fst := x, snd :=  …
  -/
  simp_rw [map_apply measurable_prod_mk_right hs]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Measurable fun b => μ (Set.preimage (fun x => { fst := x, snd := b }) s)
  -/
  exact measurable_measure_prod_mk_right hs
  /-
    🎉 no goals
  -/


/-- The Lebesgue integral is measurable. This shows that the integrand of (the right-hand-side of)
  Tonelli's theorem is measurable. -/
theorem Measurable.lintegral_prod_right' [SFinite ν] :
    ∀ {f : α × β → ℝ≥0∞}, Measurable f → Measurable fun x => ∫⁻ y, f (x, y) ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ ∀ {f : Prod α β → ENNReal}, Measurable f → Measurable fun x => MeasureTheory …
  -/
  have m := @measurable_prod_mk_left
  refine Measurable.ennreal_induction (P := fun f => Measurable fun (x : α) => ∫⁻ y, f (x, y) ∂ν)
    ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      ⊢ ∀ (c : ENNReal) ⦃s : Set (Prod α β)⦄, MeasurableSet s → (fun f => Measurable …
    -/
  · intro c s hs
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      c : ENNReal
      s : Set (Prod α β)
      hs : MeasurableSet s
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => s.indicator (fun x => …
    -/
    simp only [← indicator_comp_right]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      c : ENNReal
      s : Set (Prod α β)
      hs : MeasurableSet s
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => (Set.preimage (Prod.m …
    -/
    suffices Measurable fun x => c * ν (Prod.mk x ⁻¹' s) by simpa [lintegral_indicator (m hs)]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      c : ENNReal
      s : Set (Prod α β)
      hs : MeasurableSet s
      ⊢ Measurable fun x => HMul.hMul c (ν (Set.preimage (Prod.mk x) s))
    -/
    exact (measurable_measure_prod_mk_left hs).const_mul _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      ⊢ ∀ ⦃f g : Prod α β → ENNReal⦄, Disjoint (Function.support f) (Function.suppor …
    -/
  · rintro f g - hf - h2f h2g
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      f g : Prod α β → ENNReal
      hf : Measurable f
      h2f : Measurable fun x => MeasureTheory.lintegral ν fun y => f { fst := x, snd …
      h2g : Measurable fun x => MeasureTheory.lintegral ν fun y => g { fst := x, snd …
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => HAdd.hAdd f g { fst : …
    -/
    simp only [Pi.add_apply]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.7455} {β : Type ?u.7454} {m : MeasurableSpace α} {mβ : Meas …
      f g : Prod α β → ENNReal
      hf : Measurable f
      h2f : Measurable fun x => MeasureTheory.lintegral ν fun y => f { fst := x, snd …
      h2g : Measurable fun x => MeasureTheory.lintegral ν fun y => g { fst := x, snd …
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => HAdd.hAdd (f { fst := …
    -/
    conv => enter [1, x]; erw [lintegral_add_left (hf.comp m)]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f g : Prod α β → ENNReal
      hf : Measurable f
      h2f : Measurable fun x => MeasureTheory.lintegral ν fun y => f { fst := x, snd …
      h2g : Measurable fun x => MeasureTheory.lintegral ν fun y => g { fst := x, snd …
      ⊢ Measurable fun x => HAdd.hAdd (MeasureTheory.lintegral ν fun a => Function.c …
    -/
    exact h2f.add h2g
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      ⊢ ∀ ⦃f : Nat → Prod α β → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
    -/
  · intro f hf h2f h3f
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Measurable fun x => MeasureTheory.lintegral ν fun …
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => (fun x => iSup fun n  …
    -/
    have : ∀ x, Monotone fun n y => f n (x, y) := fun x i j hij y => h2f hij (x, y)
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Measurable fun x => MeasureTheory.lintegral ν fun …
      this : ∀ (x : α), Monotone fun n y => f n { fst := x, snd := y }
      ⊢ Measurable fun x => MeasureTheory.lintegral ν fun y => (fun x => iSup fun n  …
    -/
    conv => enter [1, x]; erw [lintegral_iSup (fun n => (hf n).comp m) (this x)]
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Measurable fun x => MeasureTheory.lintegral ν fun …
      this : ∀ (x : α), Monotone fun n y => f n { fst := x, snd := y }
      ⊢ Measurable fun x => iSup fun n => MeasureTheory.lintegral ν fun a => Functio …
    -/
    exact .iSup h3f
    /-
      🎉 no goals
    -/


/-- The Lebesgue integral is measurable. This shows that the integrand of (the right-hand-side of)
  Tonelli's theorem is measurable.
  This version has the argument `f` in curried form. -/
theorem Measurable.lintegral_prod_right [SFinite ν] {f : α → β → ℝ≥0∞}
    (hf : Measurable (uncurry f)) : Measurable fun x => ∫⁻ y, f x y ∂ν :=
  hf.lintegral_prod_right'


/-- The Lebesgue integral is measurable. This shows that the integrand of (the right-hand-side of)
  the symmetric version of Tonelli's theorem is measurable. -/
theorem Measurable.lintegral_prod_left' [SFinite μ] {f : α × β → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun y => ∫⁻ x, f (x, y) ∂μ :=
  (measurable_swap_iff.mpr hf).lintegral_prod_right'


/-- The Lebesgue integral is measurable. This shows that the integrand of (the right-hand-side of)
  the symmetric version of Tonelli's theorem is measurable.
  This version has the argument `f` in curried form. -/
theorem Measurable.lintegral_prod_left [SFinite μ] {f : α → β → ℝ≥0∞}
    (hf : Measurable (uncurry f)) : Measurable fun y => ∫⁻ x, f x y ∂μ :=
  hf.lintegral_prod_left'


/-- The binary product of measures. They are defined for arbitrary measures, but we basically
  prove all properties under the assumption that at least one of them is s-finite. -/
protected irreducible_def prod (μ : Measure α) (ν : Measure β) : Measure (α × β) :=
  bind μ fun x : α => map (Prod.mk x) ν


instance prod.measureSpace {α β} [MeasureSpace α] [MeasureSpace β] : MeasureSpace (α × β) where
  volume := volume.prod volume


theorem volume_eq_prod (α β) [MeasureSpace α] [MeasureSpace β] :
    (volume : Measure (α × β)) = (volume : Measure α).prod (volume : Measure β) :=
  rfl


theorem prod_apply {s : Set (α × β)} (hs : MeasurableSet s) :
    μ.prod ν s = ∫⁻ x, ν (Prod.mk x ⁻¹' s) ∂μ := by
  simp_rw [Measure.prod, bind_apply hs (Measurable.map_prod_mk_left (ν := ν)),
    map_apply measurable_prod_mk_left hs]


/-- The product measure of the product of two sets is the product of their measures. Note that we
do not need the sets to be measurable. -/
@[simp]
theorem prod_prod (s : Set α) (t : Set β) : μ.prod ν (s ×ˢ t) = μ s * ν t := by
  classical
  apply le_antisymm
  · set S := toMeasurable μ s
    set T := toMeasurable ν t
    have hSTm : MeasurableSet (S ×ˢ T) :=
      (measurableSet_toMeasurable _ _).prod (measurableSet_toMeasurable _ _)
    calc
      μ.prod ν (s ×ˢ t) ≤ μ.prod ν (S ×ˢ T) := by gcongr <;> apply subset_toMeasurable
      _ = μ S * ν T := by
        rw [prod_apply hSTm]
        simp_rw [S, mk_preimage_prod_right_eq_if, measure_if,
          lintegral_indicator (measurableSet_toMeasurable _ _), lintegral_const,
          restrict_apply_univ, mul_comm]
      _ = μ s * ν t := by rw [measure_toMeasurable, measure_toMeasurable]
  · -- Formalization is based on https://mathoverflow.net/a/254134/136589
    set ST := toMeasurable (μ.prod ν) (s ×ˢ t)
    have hSTm : MeasurableSet ST := measurableSet_toMeasurable _ _
    have hST : s ×ˢ t ⊆ ST := subset_toMeasurable _ _
    set f : α → ℝ≥0∞ := fun x => ν (Prod.mk x ⁻¹' ST)
    have hfm : Measurable f := measurable_measure_prod_mk_left hSTm
    set s' : Set α := { x | ν t ≤ f x }
    have hss' : s ⊆ s' := fun x hx => measure_mono fun y hy => hST <| mk_mem_prod hx hy
    calc
      μ s * ν t ≤ μ s' * ν t := by gcongr
      _ = ∫⁻ _ in s', ν t ∂μ := by rw [setLIntegral_const, mul_comm]
      _ ≤ ∫⁻ x in s', f x ∂μ := setLIntegral_mono hfm fun x => id
      _ ≤ ∫⁻ x, f x ∂μ := lintegral_mono' restrict_le_self le_rfl
      _ = μ.prod ν ST := (prod_apply hSTm).symm
      _ = μ.prod ν (s ×ˢ t) := measure_toMeasurable _


@[simp] lemma map_fst_prod : Measure.map Prod.fst (μ.prod ν) = (ν univ) • μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq (MeasureTheory.Measure.map Prod.fst (μ.prod ν)) (HSMul.hSMul (ν Set.univ) …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map Prod.fst (μ.prod ν)) s) ((HSMul.hSMul (ν Set. …
  -/
  simp [Measure.map_apply measurable_fst hs, ← prod_univ, mul_comm]
  /-
    🎉 no goals
  -/


@[simp] lemma map_snd_prod : Measure.map Prod.snd (μ.prod ν) = (μ univ) • ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq (MeasureTheory.Measure.map Prod.snd (μ.prod ν)) (HSMul.hSMul (μ Set.univ) …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map Prod.snd (μ.prod ν)) s) ((HSMul.hSMul (μ Set. …
  -/
  simp [Measure.map_apply measurable_snd hs, ← univ_prod]
  /-
    🎉 no goals
  -/


instance prod.instIsOpenPosMeasure {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    {m : MeasurableSpace X} {μ : Measure X} [IsOpenPosMeasure μ] {m' : MeasurableSpace Y}
    {ν : Measure Y} [IsOpenPosMeasure ν] [SFinite ν] : IsOpenPosMeasure (μ.prod ν) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    ⊢ (μ.prod ν).IsOpenPosMeasure
  -/
  constructor
  /-
    case open_pos
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    ⊢ ∀ (U : Set (Prod X Y)), IsOpen U → U.Nonempty → Ne ((μ.prod ν) U) 0
  -/
  rintro U U_open ⟨⟨x, y⟩, hxy⟩
  /-
    case open_pos.intro.mk
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    U : Set (Prod X Y)
    U_open : IsOpen U
    x : X
    y : Y
    hxy : Membership.mem U { fst := x, snd := y }
    ⊢ Ne ((μ.prod ν) U) 0
  -/
  rcases isOpen_prod_iff.1 U_open x y hxy with ⟨u, v, u_open, v_open, xu, yv, huv⟩
  /-
    case open_pos.intro.mk.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    U : Set (Prod X Y)
    U_open : IsOpen U
    x : X
    y : Y
    hxy : Membership.mem U { fst := x, snd := y }
    u : Set X
    v : Set Y
    u_open : IsOpen u
    v_open : IsOpen v
    xu : Membership.mem u x
    yv : Membership.mem v y
    huv : HasSubset.Subset (SProd.sprod u v) U
    ⊢ Ne ((μ.prod ν) U) 0
  -/
  refine ne_of_gt (lt_of_lt_of_le ?_ (measure_mono huv))
  /-
    case open_pos.intro.mk.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    U : Set (Prod X Y)
    U_open : IsOpen U
    x : X
    y : Y
    hxy : Membership.mem U { fst := x, snd := y }
    u : Set X
    v : Set Y
    u_open : IsOpen u
    v_open : IsOpen v
    xu : Membership.mem u x
    yv : Membership.mem v y
    huv : HasSubset.Subset (SProd.sprod u v) U
    ⊢ LT.lt 0 ((μ.prod ν) (SProd.sprod u v))
  -/
  simp only [prod_prod, CanonicallyOrderedCommSemiring.mul_pos]
  /-
    case open_pos.intro.mk.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    X : Type u_4
    Y : Type u_5
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    m : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : μ.IsOpenPosMeasure
    m' : MeasurableSpace Y
    ν : MeasureTheory.Measure Y
    inst✝¹ : ν.IsOpenPosMeasure
    inst✝ : MeasureTheory.SFinite ν
    U : Set (Prod X Y)
    U_open : IsOpen U
    x : X
    y : Y
    hxy : Membership.mem U { fst := x, snd := y }
    u : Set X
    v : Set Y
    u_open : IsOpen u
    v_open : IsOpen v
    xu : Membership.mem u x
    yv : Membership.mem v y
    huv : HasSubset.Subset (SProd.sprod u v) U
    ⊢ And (LT.lt 0 (μ u)) (LT.lt 0 (ν v))
  -/
  constructor
    /-
      case open_pos.intro.mk.intro.intro.intro.intro.intro.intro.left
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁸ : MeasurableSpace α
      inst✝⁷ : MeasurableSpace β
      inst✝⁶ : MeasurableSpace γ
      μ✝ μ' : MeasureTheory.Measure α
      ν✝ ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝⁵ : MeasureTheory.SFinite ν✝
      X : Type u_4
      Y : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝² : μ.IsOpenPosMeasure
      m' : MeasurableSpace Y
      ν : MeasureTheory.Measure Y
      inst✝¹ : ν.IsOpenPosMeasure
      inst✝ : MeasureTheory.SFinite ν
      U : Set (Prod X Y)
      U_open : IsOpen U
      x : X
      y : Y
      hxy : Membership.mem U { fst := x, snd := y }
      u : Set X
      v : Set Y
      u_open : IsOpen u
      v_open : IsOpen v
      xu : Membership.mem u x
      yv : Membership.mem v y
      huv : HasSubset.Subset (SProd.sprod u v) U
      ⊢ LT.lt 0 (μ u)
    -/
  · exact u_open.measure_pos μ ⟨x, xu⟩
    /-
      🎉 no goals
    -/
    /-
      case open_pos.intro.mk.intro.intro.intro.intro.intro.intro.right
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁸ : MeasurableSpace α
      inst✝⁷ : MeasurableSpace β
      inst✝⁶ : MeasurableSpace γ
      μ✝ μ' : MeasureTheory.Measure α
      ν✝ ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝⁵ : MeasureTheory.SFinite ν✝
      X : Type u_4
      Y : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      m : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝² : μ.IsOpenPosMeasure
      m' : MeasurableSpace Y
      ν : MeasureTheory.Measure Y
      inst✝¹ : ν.IsOpenPosMeasure
      inst✝ : MeasureTheory.SFinite ν
      U : Set (Prod X Y)
      U_open : IsOpen U
      x : X
      y : Y
      hxy : Membership.mem U { fst := x, snd := y }
      u : Set X
      v : Set Y
      u_open : IsOpen u
      v_open : IsOpen v
      xu : Membership.mem u x
      yv : Membership.mem v y
      huv : HasSubset.Subset (SProd.sprod u v) U
      ⊢ LT.lt 0 (ν v)
    -/
  · exact v_open.measure_pos ν ⟨y, yv⟩
    /-
      🎉 no goals
    -/


instance {X Y : Type*}
    [TopologicalSpace X] [MeasureSpace X] [IsOpenPosMeasure (volume : Measure X)]
    [TopologicalSpace Y] [MeasureSpace Y] [IsOpenPosMeasure (volume : Measure Y)]
    [SFinite (volume : Measure Y)] : IsOpenPosMeasure (volume : Measure (X × Y)) :=
  prod.instIsOpenPosMeasure


instance prod.instIsFiniteMeasure {α β : Type*} {mα : MeasurableSpace α} {mβ : MeasurableSpace β}
    (μ : Measure α) (ν : Measure β) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    IsFiniteMeasure (μ.prod ν) := by
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α✝
    inst✝⁴ : MeasurableSpace β✝
    inst✝³ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ MeasureTheory.IsFiniteMeasure (μ.prod ν)
  -/
  constructor
  /-
    case measure_univ_lt_top
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α✝
    inst✝⁴ : MeasurableSpace β✝
    inst✝³ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ LT.lt ((μ.prod ν) Set.univ) Top.top
  -/
  rw [← univ_prod_univ, prod_prod]
  /-
    case measure_univ_lt_top
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α✝
    inst✝⁴ : MeasurableSpace β✝
    inst✝³ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ LT.lt (HMul.hMul (μ Set.univ) (ν Set.univ)) Top.top
  -/
  exact mul_lt_top (measure_lt_top _ _) (measure_lt_top _ _)
  /-
    🎉 no goals
  -/


instance {α β : Type*} [MeasureSpace α] [MeasureSpace β] [IsFiniteMeasure (volume : Measure α)]
    [IsFiniteMeasure (volume : Measure β)] : IsFiniteMeasure (volume : Measure (α × β)) :=
  prod.instIsFiniteMeasure _ _


instance prod.instIsProbabilityMeasure {α β : Type*} {mα : MeasurableSpace α}
    {mβ : MeasurableSpace β} (μ : Measure α) (ν : Measure β) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] : IsProbabilityMeasure (μ.prod ν) :=
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ : Type u_3
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ
        μ✝ μ' : MeasureTheory.Measure α✝
        ν✝ ν' : MeasureTheory.Measure β✝
        τ : MeasureTheory.Measure γ
        inst✝² : MeasureTheory.SFinite ν✝
        α : Type u_4
        β : Type u_5
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        μ : MeasureTheory.Measure α
        ν : MeasureTheory.Measure β
        inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
        inst✝ : MeasureTheory.IsProbabilityMeasure ν
        ⊢ Eq ((μ.prod ν) Set.univ) 1
      -/
  ⟨by rw [← univ_prod_univ, prod_prod, measure_univ, measure_univ, mul_one]⟩
      /-
        🎉 no goals
      -/


instance {α β : Type*} [MeasureSpace α] [MeasureSpace β]
    [IsProbabilityMeasure (volume : Measure α)] [IsProbabilityMeasure (volume : Measure β)] :
    IsProbabilityMeasure (volume : Measure (α × β)) :=
  prod.instIsProbabilityMeasure _ _


instance prod.instIsFiniteMeasureOnCompacts {α β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    {mα : MeasurableSpace α} {mβ : MeasurableSpace β} (μ : Measure α) (ν : Measure β)
    [IsFiniteMeasureOnCompacts μ] [IsFiniteMeasureOnCompacts ν] [SFinite ν] :
    IsFiniteMeasureOnCompacts (μ.prod ν) := by
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α✝
    inst✝⁷ : MeasurableSpace β✝
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝ : MeasureTheory.SFinite ν
    ⊢ MeasureTheory.IsFiniteMeasureOnCompacts (μ.prod ν)
  -/
  refine ⟨fun K hK => ?_⟩
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α✝
    inst✝⁷ : MeasurableSpace β✝
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝ : MeasureTheory.SFinite ν
    K : Set (Prod α β)
    hK : IsCompact K
    ⊢ LT.lt ((μ.prod ν) K) Top.top
  -/
  set L := (Prod.fst '' K) ×ˢ (Prod.snd '' K) with hL
  have : K ⊆ L := by
    rintro ⟨x, y⟩ hxy
    simp only [L, prod_mk_mem_set_prod_eq, mem_image, Prod.exists, exists_and_right,
      exists_eq_right]
    exact ⟨⟨y, hxy⟩, ⟨x, hxy⟩⟩
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α✝
    inst✝⁷ : MeasurableSpace β✝
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝ : MeasureTheory.SFinite ν
    K : Set (Prod α β)
    hK : IsCompact K
    L : Set (Prod α β) := SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K)
    hL : Eq L (SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K))
    this : HasSubset.Subset K L
    ⊢ LT.lt ((μ.prod ν) K) Top.top
  -/
  apply lt_of_le_of_lt (measure_mono this)
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α✝
    inst✝⁷ : MeasurableSpace β✝
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝ : MeasureTheory.SFinite ν
    K : Set (Prod α β)
    hK : IsCompact K
    L : Set (Prod α β) := SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K)
    hL : Eq L (SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K))
    this : HasSubset.Subset K L
    ⊢ LT.lt ((μ.prod ν) L) Top.top
  -/
  rw [hL, prod_prod]
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α✝
    inst✝⁷ : MeasurableSpace β✝
    inst✝⁶ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝⁵ : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    inst✝ : MeasureTheory.SFinite ν
    K : Set (Prod α β)
    hK : IsCompact K
    L : Set (Prod α β) := SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K)
    hL : Eq L (SProd.sprod (Set.image Prod.fst K) (Set.image Prod.snd K))
    this : HasSubset.Subset K L
    ⊢ LT.lt (HMul.hMul (μ (Set.image Prod.fst K)) (ν (Set.image Prod.snd K))) Top. …
  -/
  exact mul_lt_top (hK.image continuous_fst).measure_lt_top (hK.image continuous_snd).measure_lt_top
  /-
    🎉 no goals
  -/


instance {X Y : Type*}
    [TopologicalSpace X] [MeasureSpace X] [IsFiniteMeasureOnCompacts (volume : Measure X)]
    [TopologicalSpace Y] [MeasureSpace Y] [IsFiniteMeasureOnCompacts (volume : Measure Y)]
    [SFinite (volume : Measure Y)] : IsFiniteMeasureOnCompacts (volume : Measure (X × Y)) :=
  prod.instIsFiniteMeasureOnCompacts _ _


instance prod.instNoAtoms_fst [NoAtoms μ] :
    NoAtoms (Measure.prod μ ν) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ MeasureTheory.NoAtoms (μ.prod ν)
  -/
  refine NoAtoms.mk (fun x => ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.NoAtoms μ
    x : Prod α β
    ⊢ Eq ((μ.prod ν) (Singleton.singleton x)) 0
  -/
  rw [← Set.singleton_prod_singleton, Measure.prod_prod, measure_singleton, zero_mul]
  /-
    🎉 no goals
  -/


instance prod.instNoAtoms_snd [NoAtoms ν] :
    NoAtoms (Measure.prod μ ν) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.NoAtoms ν
    ⊢ MeasureTheory.NoAtoms (μ.prod ν)
  -/
  refine NoAtoms.mk (fun x => ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.NoAtoms ν
    x : Prod α β
    ⊢ Eq ((μ.prod ν) (Singleton.singleton x)) 0
  -/
  rw [← Set.singleton_prod_singleton, Measure.prod_prod, measure_singleton (μ := ν), mul_zero]
  /-
    🎉 no goals
  -/


theorem ae_measure_lt_top {s : Set (α × β)} (hs : MeasurableSet s) (h2s : (μ.prod ν) s ≠ ∞) :
    ∀ᵐ x ∂μ, ν (Prod.mk x ⁻¹' s) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (ν (Set.preimage (Prod.mk x) s)) Top.top)  …
  -/
  rw [prod_apply hs] at h2s
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne (MeasureTheory.lintegral μ fun x => ν (Set.preimage (Prod.mk x) s)) T …
    ⊢ Filter.Eventually (fun x => LT.lt (ν (Set.preimage (Prod.mk x) s)) Top.top)  …
  -/
  exact ae_lt_top (measurable_measure_prod_mk_left hs) h2s
  /-
    🎉 no goals
  -/


/-- Note: the assumption `hs` cannot be dropped. For a counterexample, see
  Walter Rudin *Real and Complex Analysis*, example (c) in section 8.9. -/
theorem measure_prod_null {s : Set (α × β)} (hs : MeasurableSet s) :
    μ.prod ν s = 0 ↔ (fun x => ν (Prod.mk x ⁻¹' s)) =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Iff (Eq ((μ.prod ν) s) 0) ((MeasureTheory.ae μ).EventuallyEq (fun x => ν (Se …
  -/
  rw [prod_apply hs, lintegral_eq_zero_iff (measurable_measure_prod_mk_left hs)]
  /-
    🎉 no goals
  -/


/-- Note: the converse is not true without assuming that `s` is measurable. For a counterexample,
  see Walter Rudin *Real and Complex Analysis*, example (c) in section 8.9. -/
theorem measure_ae_null_of_prod_null {s : Set (α × β)} (h : μ.prod ν s = 0) :
    (fun x => ν (Prod.mk x ⁻¹' s)) =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    h : Eq ((μ.prod ν) s) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ν (Set.preimage (Prod.mk x) s)) 0
  -/
  obtain ⟨t, hst, mt, ht⟩ := exists_measurable_superset_of_null h
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    h : Eq ((μ.prod ν) s) 0
    t : Set (Prod α β)
    hst : HasSubset.Subset s t
    mt : MeasurableSet t
    ht : Eq ((μ.prod ν) t) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ν (Set.preimage (Prod.mk x) s)) 0
  -/
  rw [measure_prod_null mt] at ht
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    h : Eq ((μ.prod ν) s) 0
    t : Set (Prod α β)
    hst : HasSubset.Subset s t
    mt : MeasurableSet t
    ht : (MeasureTheory.ae μ).EventuallyEq (fun x => ν (Set.preimage (Prod.mk x) t …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ν (Set.preimage (Prod.mk x) s)) 0
  -/
  rw [eventuallyLE_antisymm_iff]
  exact
    ⟨EventuallyLE.trans_eq (Eventually.of_forall fun x => (measure_mono (preimage_mono hst) : _))
        ht,
      Eventually.of_forall fun x => zero_le _⟩


theorem AbsolutelyContinuous.prod [SFinite ν'] (h1 : μ ≪ μ') (h2 : ν ≪ ν') :
    μ.prod ν ≪ μ'.prod ν' := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ν'
    h1 : μ.AbsolutelyContinuous μ'
    h2 : ν.AbsolutelyContinuous ν'
    ⊢ (μ.prod ν).AbsolutelyContinuous (μ'.prod ν')
  -/
  refine AbsolutelyContinuous.mk fun s hs h2s => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ν'
    h1 : μ.AbsolutelyContinuous μ'
    h2 : ν.AbsolutelyContinuous ν'
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Eq ((μ'.prod ν') s) 0
    ⊢ Eq ((μ.prod ν) s) 0
  -/
  rw [measure_prod_null hs] at h2s ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ν'
    h1 : μ.AbsolutelyContinuous μ'
    h2 : ν.AbsolutelyContinuous ν'
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : (MeasureTheory.ae μ').EventuallyEq (fun x => ν' (Set.preimage (Prod.mk x …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ν (Set.preimage (Prod.mk x) s)) 0
  -/
  exact (h2s.filter_mono h1.ae_le).mono fun _ h => h2 h
  /-
    🎉 no goals
  -/


/-- Note: the converse is not true. For a counterexample, see
  Walter Rudin *Real and Complex Analysis*, example (c) in section 8.9. It is true if the set is
  measurable, see `ae_prod_mem_iff_ae_ae_mem`. -/
theorem ae_ae_of_ae_prod {p : α × β → Prop} (h : ∀ᵐ z ∂μ.prod ν, p z) :
    ∀ᵐ x ∂μ, ∀ᵐ y ∂ν, p (x, y) :=
  measure_ae_null_of_prod_null h


theorem ae_ae_eq_curry_of_prod {γ : Type*} {f g : α × β → γ} (h : f =ᵐ[μ.prod ν] g) :
    ∀ᵐ x ∂μ, curry f x =ᵐ[ν] curry g x :=
  ae_ae_of_ae_prod h


theorem ae_ae_eq_of_ae_eq_uncurry {γ : Type*} {f g : α → β → γ}
    (h : uncurry f =ᵐ[μ.prod ν] uncurry g) : ∀ᵐ x ∂μ, f x =ᵐ[ν] g x :=
  ae_ae_eq_curry_of_prod h


theorem ae_prod_iff_ae_ae {p : α × β → Prop} (hp : MeasurableSet {x | p x}) :
    (∀ᵐ z ∂μ.prod ν, p z) ↔ ∀ᵐ x ∂μ, ∀ᵐ y ∂ν, p (x, y) :=
  measure_prod_null hp.compl


theorem ae_prod_mem_iff_ae_ae_mem {s : Set (α × β)} (hs : MeasurableSet s) :
    (∀ᵐ z ∂μ.prod ν, z ∈ s) ↔ ∀ᵐ x ∂μ, ∀ᵐ y ∂ν, (x, y) ∈ s :=
  measure_prod_null hs.compl


theorem quasiMeasurePreserving_fst : QuasiMeasurePreserving Prod.fst (μ.prod ν) μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving Prod.fst (μ.prod ν) μ
  -/
  refine ⟨measurable_fst, AbsolutelyContinuous.mk fun s hs h2s => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    hs : MeasurableSet s
    h2s : Eq (μ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map Prod.fst (μ.prod ν)) s) 0
  -/
  rw [map_apply measurable_fst hs, ← prod_univ, prod_prod, h2s, zero_mul]
  /-
    🎉 no goals
  -/


theorem quasiMeasurePreserving_snd : QuasiMeasurePreserving Prod.snd (μ.prod ν) ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving Prod.snd (μ.prod ν) ν
  -/
  refine ⟨measurable_snd, AbsolutelyContinuous.mk fun s hs h2s => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set β
    hs : MeasurableSet s
    h2s : Eq (ν s) 0
    ⊢ Eq ((MeasureTheory.Measure.map Prod.snd (μ.prod ν)) s) 0
  -/
  rw [map_apply measurable_snd hs, ← univ_prod, prod_prod, h2s, mul_zero]
  /-
    🎉 no goals
  -/


lemma set_prod_ae_eq {s s' : Set α} {t t' : Set β} (hs : s =ᵐ[μ] s') (ht : t =ᵐ[ν] t') :
    (s ×ˢ t : Set (α × β)) =ᵐ[μ.prod ν] (s' ×ˢ t' : Set (α × β)) :=
  (quasiMeasurePreserving_fst.preimage_ae_eq hs).inter
    (quasiMeasurePreserving_snd.preimage_ae_eq ht)


lemma measure_prod_compl_eq_zero {s : Set α} {t : Set β}
    (s_ae_univ : μ sᶜ = 0) (t_ae_univ : ν tᶜ = 0) :
    μ.prod ν (s ×ˢ t)ᶜ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    t : Set β
    s_ae_univ : Eq (μ (HasCompl.compl s)) 0
    t_ae_univ : Eq (ν (HasCompl.compl t)) 0
    ⊢ Eq ((μ.prod ν) (HasCompl.compl (SProd.sprod s t))) 0
  -/
  rw [Set.compl_prod_eq_union, measure_union_null_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    t : Set β
    s_ae_univ : Eq (μ (HasCompl.compl s)) 0
    t_ae_univ : Eq (ν (HasCompl.compl t)) 0
    ⊢ And (Eq ((μ.prod ν) (SProd.sprod (HasCompl.compl s) Set.univ)) 0) (Eq ((μ.pr …
  -/
  simp [s_ae_univ, t_ae_univ]
  /-
    🎉 no goals
  -/


lemma _root_.MeasureTheory.NullMeasurableSet.prod {s : Set α} {t : Set β}
    (s_mble : NullMeasurableSet s μ) (t_mble : NullMeasurableSet t ν) :
    NullMeasurableSet (s ×ˢ t) (μ.prod ν) :=
  let ⟨s₀, mble_s₀, s_aeeq_s₀⟩ := s_mble
  let ⟨t₀, mble_t₀, t_aeeq_t₀⟩ := t_mble
  ⟨s₀ ×ˢ t₀, ⟨mble_s₀.prod mble_t₀, set_prod_ae_eq s_aeeq_s₀ t_aeeq_t₀⟩⟩


/-- If `s ×ˢ t` is a null measurable set and `μ s ≠ 0`, then `t` is a null measurable set. -/
lemma _root_.MeasureTheory.NullMeasurableSet.right_of_prod {s : Set α} {t : Set β}
    (h : NullMeasurableSet (s ×ˢ t) (μ.prod ν)) (hs : μ s ≠ 0) : NullMeasurableSet t ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    t : Set β
    h : MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)
    hs : Ne (μ s) 0
    ⊢ MeasureTheory.NullMeasurableSet t ν
  -/
  rcases h with ⟨u, hum, hu⟩
  obtain ⟨x, hxs, hx⟩ : ∃ x ∈ s, (Prod.mk x ⁻¹' (s ×ˢ t)) =ᵐ[ν] (Prod.mk x ⁻¹' u) :=
    ((frequently_ae_iff.2 hs).and_eventually (ae_ae_eq_curry_of_prod hu)).exists
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    t : Set β
    hs : Ne (μ s) 0
    u : Set (Prod α β)
    hum : MeasurableSet u
    hu : (MeasureTheory.ae (μ.prod ν)).EventuallyEq (SProd.sprod s t) u
    x : α
    hxs : Membership.mem s x
    hx : (MeasureTheory.ae ν).EventuallyEq (Set.preimage (Prod.mk x) (SProd.sprod  …
    ⊢ MeasureTheory.NullMeasurableSet t ν
  -/
  refine ⟨Prod.mk x ⁻¹' u, measurable_prod_mk_left hum, ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    t : Set β
    hs : Ne (μ s) 0
    u : Set (Prod α β)
    hum : MeasurableSet u
    hu : (MeasureTheory.ae (μ.prod ν)).EventuallyEq (SProd.sprod s t) u
    x : α
    hxs : Membership.mem s x
    hx : (MeasureTheory.ae ν).EventuallyEq (Set.preimage (Prod.mk x) (SProd.sprod  …
    ⊢ (MeasureTheory.ae ν).EventuallyEq t (Set.preimage (Prod.mk x) u)
  -/
  rwa [mk_preimage_prod_right hxs] at hx
  /-
    🎉 no goals
  -/


/-- If `Prod.snd ⁻¹' t` is a null measurable set and `μ ≠ 0`, then `t` is a null measurable set. -/
lemma _root_.MeasureTheory.NullMeasurableSet.of_preimage_snd [NeZero μ] {t : Set β}
    (h : NullMeasurableSet (Prod.snd ⁻¹' t) (μ.prod ν)) : NullMeasurableSet t ν :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝³ : MeasurableSpace α
                       inst✝² : MeasurableSpace β
                       μ : MeasureTheory.Measure α
                       ν : MeasureTheory.Measure β
                       inst✝¹ : MeasureTheory.SFinite ν
                       inst✝ : NeZero μ
                       t : Set β
                       h : MeasureTheory.NullMeasurableSet (Set.preimage Prod.snd t) (μ.prod ν)
                       ⊢ MeasureTheory.NullMeasurableSet (SProd.sprod Set.univ t) (μ.prod ν)
                     -/
  .right_of_prod (by rwa [univ_prod]) (NeZero.ne (μ univ))
                     /-
                       🎉 no goals
                     -/


/-- `Prod.snd ⁻¹' t` is null measurable w.r.t. `μ.prod ν` iff `t` is null measurable w.r.t. `ν`
provided that `μ ≠ 0`. -/
lemma nullMeasurableSet_preimage_snd [NeZero μ] {t : Set β} :
    NullMeasurableSet (Prod.snd ⁻¹' t) (μ.prod ν) ↔ NullMeasurableSet t ν :=
  ⟨.of_preimage_snd, (.preimage · quasiMeasurePreserving_snd)⟩


lemma nullMeasurable_comp_snd [NeZero μ] {f : β → γ} :
    NullMeasurable (f ∘ Prod.snd) (μ.prod ν) ↔ NullMeasurable f ν :=
  forall₂_congr fun s _ ↦ nullMeasurableSet_preimage_snd (t := f ⁻¹' s)


/-- `μ.prod ν` has finite spanning sets in rectangles of finite spanning sets. -/
noncomputable def FiniteSpanningSetsIn.prod {ν : Measure β} {C : Set (Set α)} {D : Set (Set β)}
    (hμ : μ.FiniteSpanningSetsIn C) (hν : ν.FiniteSpanningSetsIn D) :
    (μ.prod ν).FiniteSpanningSetsIn (image2 (· ×ˢ ·) C D) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν✝ ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝ : MeasureTheory.SFinite ν✝
    ν : MeasureTheory.Measure β
    C : Set (Set α)
    D : Set (Set β)
    hμ : μ.FiniteSpanningSetsIn C
    hν : ν.FiniteSpanningSetsIn D
    ⊢ (μ.prod ν).FiniteSpanningSetsIn (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) …
  -/
  haveI := hν.sigmaFinite
  refine
    ⟨fun n => hμ.set n.unpair.1 ×ˢ hν.set n.unpair.2, fun n =>
      mem_image2_of_mem (hμ.set_mem _) (hν.set_mem _), fun n => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν✝ ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝ : MeasureTheory.SFinite ν✝
      ν : MeasureTheory.Measure β
      C : Set (Set α)
      D : Set (Set β)
      hμ : μ.FiniteSpanningSetsIn C
      hν : ν.FiniteSpanningSetsIn D
      this : MeasureTheory.SigmaFinite ν
      n : Nat
      ⊢ LT.lt ((μ.prod ν) ((fun n => SProd.sprod (hμ.set (Nat.unpair n).1) (hν.set ( …
    -/
  · rw [prod_prod]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν✝ ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝ : MeasureTheory.SFinite ν✝
      ν : MeasureTheory.Measure β
      C : Set (Set α)
      D : Set (Set β)
      hμ : μ.FiniteSpanningSetsIn C
      hν : ν.FiniteSpanningSetsIn D
      this : MeasureTheory.SigmaFinite ν
      n : Nat
      ⊢ LT.lt (HMul.hMul (μ (hμ.set (Nat.unpair n).1)) (ν (hν.set (Nat.unpair n).2)) …
    -/
    exact mul_lt_top (hμ.finite _) (hν.finite _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν✝ ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝ : MeasureTheory.SFinite ν✝
      ν : MeasureTheory.Measure β
      C : Set (Set α)
      D : Set (Set β)
      hμ : μ.FiniteSpanningSetsIn C
      hν : ν.FiniteSpanningSetsIn D
      this : MeasureTheory.SigmaFinite ν
      ⊢ Eq (Set.iUnion fun i => (fun n => SProd.sprod (hμ.set (Nat.unpair n).1) (hν. …
    -/
  · simp_rw [iUnion_unpair_prod, hμ.spanning, hν.spanning, univ_prod_univ]
    /-
      🎉 no goals
    -/


lemma prod_sum_left {ι : Type*} (m : ι → Measure α) (μ : Measure β) [SFinite μ] :
    (Measure.sum m).prod μ = Measure.sum (fun i ↦ (m i).prod μ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ι : Type u_4
    m : ι → MeasureTheory.Measure α
    μ : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq ((MeasureTheory.Measure.sum m).prod μ) (MeasureTheory.Measure.sum fun i = …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    ι : Type u_4
    m : ι → MeasureTheory.Measure α
    μ : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.Measure.sum m).prod μ) s) ((MeasureTheory.Measure.sum fu …
  -/
  simp only [prod_apply hs, lintegral_sum_measure, hs, sum_apply, ENNReal.tsum_prod']
  /-
    🎉 no goals
  -/


lemma prod_sum_right {ι' : Type*} [Countable ι'] (m : Measure α) (m' : ι' → Measure β)
    [∀ n, SFinite (m' n)] :
    m.prod (Measure.sum m') = Measure.sum (fun p ↦ m.prod (m' p)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι' : Type u_4
    inst✝¹ : Countable ι'
    m : MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    ⊢ Eq (m.prod (MeasureTheory.Measure.sum m')) (MeasureTheory.Measure.sum fun p  …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι' : Type u_4
    inst✝¹ : Countable ι'
    m : MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq ((m.prod (MeasureTheory.Measure.sum m')) s) ((MeasureTheory.Measure.sum f …
  -/
  simp only [prod_apply hs, lintegral_sum_measure, hs, sum_apply, ENNReal.tsum_prod']
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι' : Type u_4
    inst✝¹ : Countable ι'
    m : MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral m fun x => (MeasureTheory.Measure.sum m') (Set.p …
  -/
  have M : ∀ x, MeasurableSet (Prod.mk x ⁻¹' s) := fun x => measurable_prod_mk_left hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι' : Type u_4
    inst✝¹ : Countable ι'
    m : MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    s : Set (Prod α β)
    hs : MeasurableSet s
    M : ∀ (x : α), MeasurableSet (Set.preimage (Prod.mk x) s)
    ⊢ Eq (MeasureTheory.lintegral m fun x => (MeasureTheory.Measure.sum m') (Set.p …
  -/
  simp_rw [Measure.sum_apply _ (M _)]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι' : Type u_4
    inst✝¹ : Countable ι'
    m : MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    s : Set (Prod α β)
    hs : MeasurableSet s
    M : ∀ (x : α), MeasurableSet (Set.preimage (Prod.mk x) s)
    ⊢ Eq (MeasureTheory.lintegral m fun x => tsum fun i => (m' i) (Set.preimage (P …
  -/
  rw [lintegral_tsum (fun i ↦ (measurable_measure_prod_mk_left hs).aemeasurable)]
  /-
    🎉 no goals
  -/


lemma prod_sum {ι ι' : Type*} [Countable ι'] (m : ι → Measure α) (m' : ι' → Measure β)
    [∀ n, SFinite (m' n)] :
    (Measure.sum m).prod (Measure.sum m') =
      Measure.sum (fun (p : ι × ι') ↦ (m p.1).prod (m' p.2)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ι : Type u_4
    ι' : Type u_5
    inst✝¹ : Countable ι'
    m : ι → MeasureTheory.Measure α
    m' : ι' → MeasureTheory.Measure β
    inst✝ : ∀ (n : ι'), MeasureTheory.SFinite (m' n)
    ⊢ Eq ((MeasureTheory.Measure.sum m).prod (MeasureTheory.Measure.sum m')) (Meas …
  -/
  simp_rw [prod_sum_left, prod_sum_right, sum_sum]
  /-
    🎉 no goals
  -/


instance prod.instSigmaFinite {α β : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SigmaFinite μ] {_ : MeasurableSpace β} {ν : Measure β} [SigmaFinite ν] :
    SigmaFinite (μ.prod ν) :=
  (μ.toFiniteSpanningSetsIn.prod ν.toFiniteSpanningSetsIn).sigmaFinite


instance prod.instSFinite {α β : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SFinite μ] {_ : MeasurableSpace β} {ν : Measure β} [SFinite ν] :
    SFinite (μ.prod ν) := by
  have : μ.prod ν =
      Measure.sum (fun (p : ℕ × ℕ) ↦ (sfiniteSeq μ p.1).prod (sfiniteSeq ν p.2)) := by
    conv_lhs => rw [← sum_sfiniteSeq μ, ← sum_sfiniteSeq ν]
    apply prod_sum
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α✝
    inst✝⁴ : MeasurableSpace β✝
    inst✝³ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    x✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    this : Eq (μ.prod ν) (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfinit …
    ⊢ MeasureTheory.SFinite (μ.prod ν)
  -/
  rw [this]
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α✝
    inst✝⁴ : MeasurableSpace β✝
    inst✝³ : MeasurableSpace γ
    μ✝ μ' : MeasureTheory.Measure α✝
    ν✝ ν' : MeasureTheory.Measure β✝
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν✝
    α : Type u_4
    β : Type u_5
    x✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    this : Eq (μ.prod ν) (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfinit …
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfi …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {α β} [MeasureSpace α] [SigmaFinite (volume : Measure α)]
    [MeasureSpace β] [SigmaFinite (volume : Measure β)] : SigmaFinite (volume : Measure (α × β)) :=
  prod.instSigmaFinite


instance {α β} [MeasureSpace α] [SFinite (volume : Measure α)]
    [MeasureSpace β] [SFinite (volume : Measure β)] : SFinite (volume : Measure (α × β)) :=
  prod.instSFinite


/-- A measure on a product space equals the product measure if they are equal on rectangles
  with as sides sets that generate the corresponding σ-algebras. -/
theorem prod_eq_generateFrom {μ : Measure α} {ν : Measure β} {C : Set (Set α)} {D : Set (Set β)}
    (hC : generateFrom C = ‹_›) (hD : generateFrom D = ‹_›) (h2C : IsPiSystem C)
    (h2D : IsPiSystem D) (h3C : μ.FiniteSpanningSetsIn C) (h3D : ν.FiniteSpanningSetsIn D)
    {μν : Measure (α × β)} (h₁ : ∀ s ∈ C, ∀ t ∈ D, μν (s ×ˢ t) = μ s * ν t) : μ.prod ν = μν := by
  refine
    (h3C.prod h3D).ext
      (generateFrom_eq_prod hC hD h3C.isCountablySpanning h3D.isCountablySpanning).symm
      (h2C.prod h2D) ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    C : Set (Set α)
    D : Set (Set β)
    hC : Eq (MeasurableSpace.generateFrom C) inst✝¹
    hD : Eq (MeasurableSpace.generateFrom D) inst✝
    h2C : IsPiSystem C
    h2D : IsPiSystem D
    h3C : μ.FiniteSpanningSetsIn C
    h3D : ν.FiniteSpanningSetsIn D
    μν : MeasureTheory.Measure (Prod α β)
    h₁ : ∀ (s : Set α), Membership.mem C s → ∀ (t : Set β), Membership.mem D t → E …
    ⊢ ∀ (s : Set (Prod α β)), Membership.mem (Set.image2 (fun x1 x2 => SProd.sprod …
  -/
  rintro _ ⟨s, hs, t, ht, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    C : Set (Set α)
    D : Set (Set β)
    hC : Eq (MeasurableSpace.generateFrom C) inst✝¹
    hD : Eq (MeasurableSpace.generateFrom D) inst✝
    h2C : IsPiSystem C
    h2D : IsPiSystem D
    h3C : μ.FiniteSpanningSetsIn C
    h3D : ν.FiniteSpanningSetsIn D
    μν : MeasureTheory.Measure (Prod α β)
    h₁ : ∀ (s : Set α), Membership.mem C s → ∀ (t : Set β), Membership.mem D t → E …
    s : Set α
    hs : Membership.mem C s
    t : Set β
    ht : Membership.mem D t
    ⊢ Eq ((μ.prod ν) ((fun x1 x2 => SProd.sprod x1 x2) s t)) (μν ((fun x1 x2 => SP …
  -/
  haveI := h3D.sigmaFinite
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    C : Set (Set α)
    D : Set (Set β)
    hC : Eq (MeasurableSpace.generateFrom C) inst✝¹
    hD : Eq (MeasurableSpace.generateFrom D) inst✝
    h2C : IsPiSystem C
    h2D : IsPiSystem D
    h3C : μ.FiniteSpanningSetsIn C
    h3D : ν.FiniteSpanningSetsIn D
    μν : MeasureTheory.Measure (Prod α β)
    h₁ : ∀ (s : Set α), Membership.mem C s → ∀ (t : Set β), Membership.mem D t → E …
    s : Set α
    hs : Membership.mem C s
    t : Set β
    ht : Membership.mem D t
    this : MeasureTheory.SigmaFinite ν
    ⊢ Eq ((μ.prod ν) ((fun x1 x2 => SProd.sprod x1 x2) s t)) (μν ((fun x1 x2 => SP …
  -/
  rw [h₁ s hs t ht, prod_prod]
  /-
    🎉 no goals
  -/

/- Note that the next theorem is not true for s-finite measures: let `μ = ν = ∞ • Leb` on `[0,1]`
(they are  s-finite as countable sums of the finite Lebesgue measure), and let `μν = μ.prod ν + λ`
where `λ` is Lebesgue measure on the diagonal. Then both measures give infinite mass to rectangles
`s × t` whose sides have positive Lebesgue measure, and `0` measure when one of the sides has zero
Lebesgue measure. And yet they do not coincide, as the first one gives zero mass to the diagonal,
and the second one gives mass one.
-/

/-- A measure on a product space equals the product measure of sigma-finite measures if they are
equal on rectangles. -/
theorem prod_eq {μ : Measure α} [SigmaFinite μ] {ν : Measure β} [SigmaFinite ν]
    {μν : Measure (α × β)}
    (h : ∀ s t, MeasurableSet s → MeasurableSet t → μν (s ×ˢ t) = μ s * ν t) : μ.prod ν = μν :=
  prod_eq_generateFrom generateFrom_measurableSet generateFrom_measurableSet
    isPiSystem_measurableSet isPiSystem_measurableSet μ.toFiniteSpanningSetsIn
    ν.toFiniteSpanningSetsIn fun s hs t ht => h s t hs ht


theorem prod_swap : map Prod.swap (μ.prod ν) = ν.prod μ := by
  have : sum (fun (i : ℕ × ℕ) ↦ map Prod.swap ((sfiniteSeq μ i.1).prod (sfiniteSeq ν i.2)))
       = sum (fun (i : ℕ × ℕ) ↦ map Prod.swap ((sfiniteSeq μ i.2).prod (sfiniteSeq ν i.1))) := by
    ext s hs
    rw [sum_apply _ hs, sum_apply _ hs]
    exact ((Equiv.prodComm ℕ ℕ).tsum_eq _).symm
  rw [← sum_sfiniteSeq μ, ← sum_sfiniteSeq ν, prod_sum, prod_sum,
    map_sum measurable_swap.aemeasurable, this]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    this : Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.s …
    ⊢ Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.swap ( …
  -/
  congr 1
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    this : Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.s …
    ⊢ Eq (fun i => MeasureTheory.Measure.map Prod.swap ((MeasureTheory.sfiniteSeq  …
  -/
  ext1 i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    this : Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.s …
    i : Prod Nat Nat
    ⊢ Eq (MeasureTheory.Measure.map Prod.swap ((MeasureTheory.sfiniteSeq μ i.2).pr …
  -/
  refine (prod_eq ?_).symm
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    this : Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.s …
    i : Prod Nat Nat
    ⊢ ∀ (s : Set β) (t : Set α), MeasurableSet s → MeasurableSet t → Eq ((MeasureT …
  -/
  intro s t hs ht
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    this : Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map Prod.s …
    i : Prod Nat Nat
    s : Set β
    t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq ((MeasureTheory.Measure.map Prod.swap ((MeasureTheory.sfiniteSeq μ i.2).p …
  -/
  simp_rw [map_apply measurable_swap (hs.prod ht), preimage_swap_prod, prod_prod, mul_comm]
  /-
    🎉 no goals
  -/


theorem measurePreserving_swap : MeasurePreserving Prod.swap (μ.prod ν) (ν.prod μ) :=
  ⟨measurable_swap, prod_swap⟩


theorem prod_apply_symm {s : Set (α × β)} (hs : MeasurableSet s) :
    μ.prod ν s = ∫⁻ y, μ ((fun x => (x, y)) ⁻¹' s) ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq ((μ.prod ν) s) (MeasureTheory.lintegral ν fun y => μ (Set.preimage (fun x …
  -/
  rw [← prod_swap, map_apply measurable_swap hs, prod_apply (measurable_swap hs)]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ν fun x => μ (Set.preimage (Prod.mk x) (Set.prei …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ae_ae_comm {p : α → β → Prop} (h : MeasurableSet {x : α × β | p x.1 x.2}) :
    (∀ᵐ x ∂μ, ∀ᵐ y ∂ν, p x y) ↔ ∀ᵐ y ∂ν, ∀ᵐ x ∂μ, p x y := calc
  _ ↔ ∀ᵐ x ∂μ.prod ν, p x.1 x.2 := .symm <| ae_prod_iff_ae_ae h
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝³ : MeasurableSpace α
                                        inst✝² : MeasurableSpace β
                                        μ : MeasureTheory.Measure α
                                        ν : MeasureTheory.Measure β
                                        inst✝¹ : MeasureTheory.SFinite ν
                                        inst✝ : MeasureTheory.SFinite μ
                                        p : α → β → Prop
                                        h : MeasurableSet (setOf fun x => p x.1 x.2)
                                        ⊢ Iff (Filter.Eventually (fun x => p x.1 x.2) (MeasureTheory.ae (μ.prod ν))) ( …
                                      -/
  _ ↔ ∀ᵐ x ∂ν.prod μ, p x.2 x.1 := by rw [← prod_swap, ae_map_iff (by fun_prop) h]; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  _ ↔ ∀ᵐ y ∂ν, ∀ᵐ x ∂μ, p x y := ae_prod_iff_ae_ae <| measurable_swap h


/-- If `s ×ˢ t` is a null measurable set and `ν t ≠ 0`, then `s` is a null measurable set. -/
lemma _root_.MeasureTheory.NullMeasurableSet.left_of_prod {s : Set α} {t : Set β}
    (h : NullMeasurableSet (s ×ˢ t) (μ.prod ν)) (ht : ν t ≠ 0) : NullMeasurableSet s μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    h : MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)
    ht : Ne (ν t) 0
    ⊢ MeasureTheory.NullMeasurableSet s μ
  -/
  refine .right_of_prod ?_ ht
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    h : MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)
    ht : Ne (ν t) 0
    ⊢ MeasureTheory.NullMeasurableSet (SProd.sprod t s) (ν.prod μ)
  -/
  rw [← preimage_swap_prod]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    h : MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)
    ht : Ne (ν t) 0
    ⊢ MeasureTheory.NullMeasurableSet (Set.preimage Prod.swap (SProd.sprod s t)) ( …
  -/
  exact h.preimage measurePreserving_swap.quasiMeasurePreserving
  /-
    🎉 no goals
  -/


/-- If `Prod.fst ⁻¹' s` is a null measurable set and `ν ≠ 0`, then `s` is a null measurable set. -/
lemma _root_.MeasureTheory.NullMeasurableSet.of_preimage_fst [NeZero ν] {s : Set α}
    (h : NullMeasurableSet (Prod.fst ⁻¹' s) (μ.prod ν)) : NullMeasurableSet s μ :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      inst✝⁴ : MeasurableSpace α
                      inst✝³ : MeasurableSpace β
                      μ : MeasureTheory.Measure α
                      ν : MeasureTheory.Measure β
                      inst✝² : MeasureTheory.SFinite ν
                      inst✝¹ : MeasureTheory.SFinite μ
                      inst✝ : NeZero ν
                      s : Set α
                      h : MeasureTheory.NullMeasurableSet (Set.preimage Prod.fst s) (μ.prod ν)
                      ⊢ MeasureTheory.NullMeasurableSet (SProd.sprod s Set.univ) (μ.prod ν)
                    -/
  .left_of_prod (by rwa [prod_univ]) (NeZero.ne (ν univ))
                    /-
                      🎉 no goals
                    -/


/-- `Prod.fst ⁻¹' s` is null measurable w.r.t. `μ.prod ν` iff `s` is null measurable w.r.t. `μ`
provided that `ν ≠ 0`. -/
lemma nullMeasurableSet_preimage_fst [NeZero ν] {s : Set α} :
    NullMeasurableSet (Prod.fst ⁻¹' s) (μ.prod ν) ↔ NullMeasurableSet s μ :=
  ⟨.of_preimage_fst, (.preimage · quasiMeasurePreserving_fst)⟩


lemma nullMeasurable_comp_fst [NeZero ν] {f : α → γ} :
    NullMeasurable (f ∘ Prod.fst) (μ.prod ν) ↔ NullMeasurable f μ :=
  forall₂_congr fun s _ ↦ nullMeasurableSet_preimage_fst (s := f ⁻¹' s)


/-- The product of two non-null sets is null measurable
if and only if both of them are null measurable. -/
lemma nullMeasurableSet_prod_of_ne_zero {s : Set α} {t : Set β} (hs : μ s ≠ 0) (ht : ν t ≠ 0) :
    NullMeasurableSet (s ×ˢ t) (μ.prod ν) ↔ NullMeasurableSet s μ ∧ NullMeasurableSet t ν :=
  ⟨fun h ↦ ⟨h.left_of_prod ht, h.right_of_prod hs⟩, fun ⟨hs, ht⟩ ↦ hs.prod ht⟩


/-- The product of two sets is null measurable
if and only if both of them are null measurable or one of them has measure zero. -/
lemma nullMeasurableSet_prod {s : Set α} {t : Set β} :
    NullMeasurableSet (s ×ˢ t) (μ.prod ν) ↔
      NullMeasurableSet s μ ∧ NullMeasurableSet t ν ∨ μ s = 0 ∨ ν t = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    ⊢ Iff (MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)) (Or (And  …
  -/
  rcases eq_or_ne (μ s) 0 with hs | hs; · simp [NullMeasurableSet.of_null, *]
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    hs : Ne (μ s) 0
    ⊢ Iff (MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)) (Or (And  …
  -/
  rcases eq_or_ne (ν t) 0 with ht | ht; · simp [NullMeasurableSet.of_null, *]
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    hs : Ne (μ s) 0
    ht : Ne (ν t) 0
    ⊢ Iff (MeasureTheory.NullMeasurableSet (SProd.sprod s t) (μ.prod ν)) (Or (And  …
  -/
  simp [*, nullMeasurableSet_prod_of_ne_zero]
  /-
    🎉 no goals
  -/


theorem prodAssoc_prod [SFinite τ] :
    map MeasurableEquiv.prodAssoc ((μ.prod ν).prod τ) = μ.prod (ν.prod τ) := by
  have : sum (fun (p : ℕ × ℕ × ℕ) ↦
        (sfiniteSeq μ p.1).prod ((sfiniteSeq ν p.2.1).prod (sfiniteSeq τ p.2.2)))
      = sum (fun (p : (ℕ × ℕ) × ℕ) ↦
        (sfiniteSeq μ p.1.1).prod ((sfiniteSeq ν p.1.2).prod (sfiniteSeq τ p.2))) := by
    ext s hs
    rw [sum_apply _ hs, sum_apply _ hs, ← (Equiv.prodAssoc _ _ _).tsum_eq]
    simp only [Equiv.prodAssoc_apply]
  rw [← sum_sfiniteSeq μ, ← sum_sfiniteSeq ν, ← sum_sfiniteSeq τ, prod_sum, prod_sum,
    map_sum MeasurableEquiv.prodAssoc.measurable.aemeasurable, prod_sum, prod_sum, this]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite τ
    this : Eq (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfiniteSeq μ p.1) …
    ⊢ Eq (MeasureTheory.Measure.sum fun i => MeasureTheory.Measure.map (⇑Measurabl …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite τ
    this : Eq (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfiniteSeq μ p.1) …
    ⊢ Eq (fun i => MeasureTheory.Measure.map (⇑MeasurableEquiv.prodAssoc) (((Measu …
  -/
  ext1 i
  refine (prod_eq_generateFrom generateFrom_measurableSet generateFrom_prod
    isPiSystem_measurableSet isPiSystem_prod ((sfiniteSeq μ i.1.1)).toFiniteSpanningSetsIn
    ((sfiniteSeq ν i.1.2).toFiniteSpanningSetsIn.prod (sfiniteSeq τ i.2).toFiniteSpanningSetsIn)
      ?_).symm
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite τ
    this : Eq (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfiniteSeq μ p.1) …
    i : Prod (Prod Nat Nat) Nat
    ⊢ ∀ (s : Set α), Membership.mem (setOf fun s => MeasurableSet s) s → ∀ (t : Se …
  -/
  rintro s hs _ ⟨t, ht, u, hu, rfl⟩; rw [mem_setOf_eq] at hs ht hu
  simp_rw [map_apply (MeasurableEquiv.measurable _) (hs.prod (ht.prod hu)),
    MeasurableEquiv.prodAssoc, MeasurableEquiv.coe_mk, Equiv.prod_assoc_preimage, prod_prod,
    mul_assoc]


theorem prod_restrict (s : Set α) (t : Set β) :
    (μ.restrict s).prod (ν.restrict t) = (μ.prod ν).restrict (s ×ˢ t) := by
  rw [← sum_sfiniteSeq μ, ← sum_sfiniteSeq ν, restrict_sum_of_countable, restrict_sum_of_countable,
    prod_sum, prod_sum, restrict_sum_of_countable]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    ⊢ Eq (MeasureTheory.Measure.sum fun p => ((MeasureTheory.sfiniteSeq μ p.1).res …
  -/
  congr 1
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    ⊢ Eq (fun p => ((MeasureTheory.sfiniteSeq μ p.1).restrict s).prod ((MeasureThe …
  -/
  ext1 i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    t : Set β
    i : Prod Nat Nat
    ⊢ Eq (((MeasureTheory.sfiniteSeq μ i.1).restrict s).prod ((MeasureTheory.sfini …
  -/
  refine prod_eq fun s' t' hs' ht' => ?_
  rw [restrict_apply (hs'.prod ht'), prod_inter_prod, prod_prod, restrict_apply hs',
    restrict_apply ht']


theorem restrict_prod_eq_prod_univ (s : Set α) :
    (μ.restrict s).prod ν = (μ.prod ν).restrict (s ×ˢ univ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    ⊢ Eq ((μ.restrict s).prod ν) ((μ.prod ν).restrict (SProd.sprod s Set.univ))
  -/
  have : ν = ν.restrict Set.univ := Measure.restrict_univ.symm
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    this : Eq ν (ν.restrict Set.univ)
    ⊢ Eq ((μ.restrict s).prod ν) ((μ.prod ν).restrict (SProd.sprod s Set.univ))
  -/
  rw [this, Measure.prod_restrict, ← this]
  /-
    🎉 no goals
  -/


theorem prod_dirac (y : β) : μ.prod (dirac y) = map (fun x => (x, y)) μ := by
  classical
  rw [← sum_sfiniteSeq μ, prod_sum_left, map_sum measurable_prod_mk_right.aemeasurable]
  congr
  ext1 i
  refine prod_eq fun s t hs ht => ?_
  simp_rw [map_apply measurable_prod_mk_right (hs.prod ht), mk_preimage_prod_left_eq_if, measure_if,
    dirac_apply' _ ht, ← indicator_mul_right _ fun _ => sfiniteSeq μ i s, Pi.one_apply, mul_one]


theorem dirac_prod (x : α) : (dirac x).prod ν = map (Prod.mk x) ν := by
  classical
  rw [← sum_sfiniteSeq ν, prod_sum_right, map_sum measurable_prod_mk_left.aemeasurable]
  congr
  ext1 i
  refine prod_eq fun s t hs ht => ?_
  simp_rw [map_apply measurable_prod_mk_left (hs.prod ht), mk_preimage_prod_right_eq_if, measure_if,
    dirac_apply' _ hs, ← indicator_mul_left _ _ fun _ => sfiniteSeq ν i t, Pi.one_apply, one_mul]


theorem dirac_prod_dirac {x : α} {y : β} : (dirac x).prod (dirac y) = dirac (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    x : α
    y : β
    ⊢ Eq ((MeasureTheory.Measure.dirac x).prod (MeasureTheory.Measure.dirac y)) (M …
  -/
  rw [prod_dirac, map_dirac measurable_prod_mk_right]
  /-
    🎉 no goals
  -/


theorem prod_add (ν' : Measure β) [SFinite ν'] : μ.prod (ν + ν') = μ.prod ν + μ.prod ν' := by
  simp_rw [← sum_sfiniteSeq ν, ← sum_sfiniteSeq ν', sum_add_sum, ← sum_sfiniteSeq μ, prod_sum,
    sum_add_sum]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    ν' : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν'
    ⊢ Eq (MeasureTheory.Measure.sum fun p => (MeasureTheory.sfiniteSeq μ p.1).prod …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    ν' : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν'
    ⊢ Eq (fun p => (MeasureTheory.sfiniteSeq μ p.1).prod (HAdd.hAdd (MeasureTheory …
  -/
  ext1 i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    ν' : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν'
    i : Prod Nat Nat
    ⊢ Eq ((MeasureTheory.sfiniteSeq μ i.1).prod (HAdd.hAdd (MeasureTheory.sfiniteS …
  -/
  refine prod_eq fun s t _ _ => ?_
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    ν' : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν'
    i : Prod Nat Nat
    s : Set α
    t : Set β
    x✝¹ : MeasurableSet s
    x✝ : MeasurableSet t
    ⊢ Eq ((HAdd.hAdd ((MeasureTheory.sfiniteSeq μ i.1).prod (MeasureTheory.sfinite …
  -/
  simp_rw [add_apply, prod_prod, left_distrib]
  /-
    🎉 no goals
  -/


theorem add_prod (μ' : Measure α) [SFinite μ'] : (μ + μ').prod ν = μ.prod ν + μ'.prod ν := by
  simp_rw [← sum_sfiniteSeq μ, ← sum_sfiniteSeq μ', sum_add_sum, ← sum_sfiniteSeq ν, prod_sum,
    sum_add_sum]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    μ' : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ'
    ⊢ Eq (MeasureTheory.Measure.sum fun p => (HAdd.hAdd (MeasureTheory.sfiniteSeq  …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    μ' : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ'
    ⊢ Eq (fun p => (HAdd.hAdd (MeasureTheory.sfiniteSeq μ p.1) (MeasureTheory.sfin …
  -/
  ext1 i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    μ' : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ'
    i : Prod Nat Nat
    ⊢ Eq ((HAdd.hAdd (MeasureTheory.sfiniteSeq μ i.1) (MeasureTheory.sfiniteSeq μ' …
  -/
  refine prod_eq fun s t _ _ => ?_
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    μ' : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ'
    i : Prod Nat Nat
    s : Set α
    t : Set β
    x✝¹ : MeasurableSet s
    x✝ : MeasurableSet t
    ⊢ Eq ((HAdd.hAdd ((MeasureTheory.sfiniteSeq μ i.1).prod (MeasureTheory.sfinite …
  -/
  simp_rw [add_apply, prod_prod, right_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_prod (ν : Measure β) : (0 : Measure α).prod ν = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.Measure.prod 0 ν) 0
  -/
  rw [Measure.prod]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.Measure.bind 0 fun x => MeasureTheory.Measure.map (Prod.mk …
  -/
  exact bind_zero_left _
  /-
    🎉 no goals
  -/


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝¹ : MeasurableSpace α
                                                                       inst✝ : MeasurableSpace β
                                                                       μ : MeasureTheory.Measure α
                                                                       ⊢ Eq (μ.prod 0) 0
                                                                     -/
theorem prod_zero (μ : Measure α) : μ.prod (0 : Measure β) = 0 := by simp [Measure.prod]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem map_prod_map {δ} [MeasurableSpace δ] {f : α → β} {g : γ → δ} (μa : Measure α)
    (μc : Measure γ) [SFinite μa] [SFinite μc] (hf : Measurable f) (hg : Measurable g) :
    (map f μa).prod (map g μc) = map (Prod.map f g) (μa.prod μc) := by
  simp_rw [← sum_sfiniteSeq μa, ← sum_sfiniteSeq μc, map_sum hf.aemeasurable,
    map_sum hg.aemeasurable, prod_sum, map_sum (hf.prod_map hg).aemeasurable]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    f : α → β
    g : γ → δ
    μa : MeasureTheory.Measure α
    μc : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (MeasureTheory.Measure.sum fun p => (MeasureTheory.Measure.map f (Measure …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    f : α → β
    g : γ → δ
    μa : MeasureTheory.Measure α
    μc : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (fun p => (MeasureTheory.Measure.map f (MeasureTheory.sfiniteSeq μa p.1)) …
  -/
  ext1 i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    f : α → β
    g : γ → δ
    μa : MeasureTheory.Measure α
    μc : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    hf : Measurable f
    hg : Measurable g
    i : Prod Nat Nat
    ⊢ Eq ((MeasureTheory.Measure.map f (MeasureTheory.sfiniteSeq μa i.1)).prod (Me …
  -/
  refine prod_eq fun s t hs ht => ?_
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    f : α → β
    g : γ → δ
    μa : MeasureTheory.Measure α
    μc : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    hf : Measurable f
    hg : Measurable g
    i : Prod Nat Nat
    s : Set β
    t : Set δ
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq ((MeasureTheory.Measure.map (Prod.map f g) ((MeasureTheory.sfiniteSeq μa  …
  -/
  rw [map_apply (hf.prod_map hg) (hs.prod ht), map_apply hf hs, map_apply hg ht]
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    f : α → β
    g : γ → δ
    μa : MeasureTheory.Measure α
    μc : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    hf : Measurable f
    hg : Measurable g
    i : Prod Nat Nat
    s : Set β
    t : Set δ
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq (((MeasureTheory.sfiniteSeq μa i.1).prod (MeasureTheory.sfiniteSeq μc i.2 …
  -/
  exact prod_prod (f ⁻¹' s) (g ⁻¹' t)
  /-
    🎉 no goals
  -/


/-- Let `f : α → β` be a measure preserving map.
For a.e. all `a`, let `g a : γ → δ` be a measure preserving map.
Also suppose that `g` is measurable as a function of two arguments.
Then the map `fun (a, c) ↦ (f a, g a c)` is a measure preserving map
for the product measures on `α × γ` and `β × δ`.

Some authors call a map of the form `fun (a, c) ↦ (f a, g a c)` a *skew product* over `f`,
thus the choice of a name.
-/
theorem skew_product [SFinite μa] [SFinite μc] {f : α → β} (hf : MeasurePreserving f μa μb)
    {g : α → γ → δ} (hgm : Measurable (uncurry g)) (hg : ∀ᵐ a ∂μa, map (g a) μc = μd) :
    MeasurePreserving (fun p : α × γ => (f p.1, g p.1 p.2)) (μa.prod μc) (μb.prod μd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    ⊢ MeasureTheory.MeasurePreserving (fun p => { fst := f p.1, snd := g p.1 p.2 } …
  -/
  have : Measurable fun p : α × γ => (f p.1, g p.1 p.2) := (hf.1.comp measurable_fst).prod_mk hgm
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
    ⊢ MeasureTheory.MeasurePreserving (fun p => { fst := f p.1, snd := g p.1 p.2 } …
  -/
  use this
  /- if `μa = 0`, then the lemma is trivial, otherwise we can use `hg`
    to deduce `SFinite μd`. -/
  /-
    case map_eq
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
    ⊢ Eq (MeasureTheory.Measure.map (fun p => { fst := f p.1, snd := g p.1 p.2 })  …
  -/
  rcases eq_zero_or_neZero μa with rfl | _
    /-
      case map_eq.inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      δ : Type u_4
      inst✝² : MeasurableSpace δ
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      μd : MeasureTheory.Measure δ
      inst✝¹ : MeasureTheory.SFinite μc
      f : α → β
      g : α → γ → δ
      hgm : Measurable (Function.uncurry g)
      this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
      inst✝ : MeasureTheory.SFinite 0
      hf : MeasureTheory.MeasurePreserving f 0 μb
      hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
      ⊢ Eq (MeasureTheory.Measure.map (fun p => { fst := f p.1, snd := g p.1 p.2 })  …
    -/
  · simp [← hf.map_eq]
    /-
      🎉 no goals
    -/
  have sf : SFinite μd := by
    obtain ⟨a, ha⟩ : ∃ a, map (g a) μc = μd := hg.exists
    rw [← ha]
    infer_instance
  -- Thus we can use the integral formula for the product measure, and compute things explicitly
  /-
    case map_eq.inr
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
    h✝ : NeZero μa
    sf : MeasureTheory.SFinite μd
    ⊢ Eq (MeasureTheory.Measure.map (fun p => { fst := f p.1, snd := g p.1 p.2 })  …
  -/
  ext s hs
  rw [map_apply this hs, prod_apply (this hs), prod_apply hs,
    ← hf.lintegral_comp (measurable_measure_prod_mk_left hs)]
  /-
    case map_eq.inr.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
    h✝ : NeZero μa
    sf : MeasureTheory.SFinite μd
    s : Set (Prod β δ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral μa fun x => μc (Set.preimage (Prod.mk x) (Set.pr …
  -/
  apply lintegral_congr_ae
  /-
    case map_eq.inr.h.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    δ : Type u_4
    inst✝² : MeasurableSpace δ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    μd : MeasureTheory.Measure δ
    inst✝¹ : MeasureTheory.SFinite μa
    inst✝ : MeasureTheory.SFinite μc
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    g : α → γ → δ
    hgm : Measurable (Function.uncurry g)
    hg : Filter.Eventually (fun a => Eq (MeasureTheory.Measure.map (g a) μc) μd) ( …
    this : Measurable fun p => { fst := f p.1, snd := g p.1 p.2 }
    h✝ : NeZero μa
    sf : MeasureTheory.SFinite μd
    s : Set (Prod β δ)
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae μa).EventuallyEq (fun a => μc (Set.preimage (Prod.mk a) (S …
  -/
  filter_upwards [hg] with a ha
  rw [← ha, map_apply hgm.of_uncurry_left (measurable_prod_mk_left hs), preimage_preimage,
    preimage_preimage]


/-- If `f : α → β` sends the measure `μa` to `μb` and `g : γ → δ` sends the measure `μc` to `μd`,
then `Prod.map f g` sends `μa.prod μc` to `μb.prod μd`. -/
protected theorem prod [SFinite μa] [SFinite μc] {f : α → β} {g : γ → δ}
    (hf : MeasurePreserving f μa μb) (hg : MeasurePreserving g μc μd) :
    MeasurePreserving (Prod.map f g) (μa.prod μc) (μb.prod μd) :=
  have : Measurable (uncurry fun _ : α => g) := hg.1.comp measurable_snd
  hf.skew_product this <| ae_of_all _ fun _ => hg.map_eq


theorem prod_of_right {f : α × β → γ} {μ : Measure α} {ν : Measure β} {τ : Measure γ}
    (hf : Measurable f) [SFinite ν]
    (h2f : ∀ᵐ x ∂μ, QuasiMeasurePreserving (fun y => f (x, y)) ν τ) :
    QuasiMeasurePreserving f (μ.prod ν) τ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : MeasurableSpace γ
    f : Prod α β → γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    hf : Measurable f
    inst✝ : MeasureTheory.SFinite ν
    h2f : Filter.Eventually (fun x => MeasureTheory.Measure.QuasiMeasurePreserving …
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving f (μ.prod ν) τ
  -/
  refine ⟨hf, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : MeasurableSpace γ
    f : Prod α β → γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    hf : Measurable f
    inst✝ : MeasureTheory.SFinite ν
    h2f : Filter.Eventually (fun x => MeasureTheory.Measure.QuasiMeasurePreserving …
    ⊢ (MeasureTheory.Measure.map f (μ.prod ν)).AbsolutelyContinuous τ
  -/
  refine AbsolutelyContinuous.mk fun s hs h2s => ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : MeasurableSpace γ
    f : Prod α β → γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    hf : Measurable f
    inst✝ : MeasureTheory.SFinite ν
    h2f : Filter.Eventually (fun x => MeasureTheory.Measure.QuasiMeasurePreserving …
    s : Set γ
    hs : MeasurableSet s
    h2s : Eq (τ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map f (μ.prod ν)) s) 0
  -/
  rw [map_apply hf hs, prod_apply (hf hs)]; simp_rw [preimage_preimage]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : MeasurableSpace γ
    f : Prod α β → γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    hf : Measurable f
    inst✝ : MeasureTheory.SFinite ν
    h2f : Filter.Eventually (fun x => MeasureTheory.Measure.QuasiMeasurePreserving …
    s : Set γ
    hs : MeasurableSet s
    h2s : Eq (τ s) 0
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ν (Set.preimage (fun x_1 => f { fst : …
  -/
  rw [lintegral_congr_ae (h2f.mono fun x hx => hx.preimage_null h2s), lintegral_zero]
  /-
    🎉 no goals
  -/


theorem prod_of_left {α β γ} [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace γ]
    {f : α × β → γ} {μ : Measure α} {ν : Measure β} {τ : Measure γ} (hf : Measurable f)
    [SFinite μ] [SFinite ν]
    (h2f : ∀ᵐ y ∂ν, QuasiMeasurePreserving (fun x => f (x, y)) μ τ) :
    QuasiMeasurePreserving f (μ.prod ν) τ := by
  /-
    α : Type u_4
    β : Type u_5
    γ : Type u_6
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    f : Prod α β → γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    hf : Measurable f
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h2f : Filter.Eventually (fun y => MeasureTheory.Measure.QuasiMeasurePreserving …
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving f (μ.prod ν) τ
  -/
  rw [← prod_swap]
  convert (QuasiMeasurePreserving.prod_of_right (hf.comp measurable_swap) h2f).comp
      ((measurable_swap.measurePreserving (ν.prod μ)).symm
          MeasurableEquiv.prodComm).quasiMeasurePreserving


theorem AEMeasurable.prod_swap [SFinite μ] [SFinite ν] {f : β × α → γ}
    (hf : AEMeasurable f (ν.prod μ)) : AEMeasurable (fun z : α × β => f z.swap) (μ.prod ν) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    f : Prod β α → γ
    hf : AEMeasurable f (ν.prod μ)
    ⊢ AEMeasurable (fun z => f z.swap) (μ.prod ν)
  -/
  rw [← Measure.prod_swap] at hf
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    f : Prod β α → γ
    hf : AEMeasurable f (MeasureTheory.Measure.map Prod.swap (μ.prod ν))
    ⊢ AEMeasurable (fun z => f z.swap) (μ.prod ν)
  -/
  exact hf.comp_measurable measurable_swap
  /-
    🎉 no goals
  -/

-- TODO: make this theorem usable with `fun_prop`

theorem AEMeasurable.fst [SFinite ν] {f : α → γ} (hf : AEMeasurable f μ) :
    AEMeasurable (fun z : α × β => f z.1) (μ.prod ν) :=
  hf.comp_quasiMeasurePreserving quasiMeasurePreserving_fst

-- TODO: make this theorem usable with `fun_prop`

theorem AEMeasurable.snd [SFinite ν] {f : β → γ} (hf : AEMeasurable f ν) :
    AEMeasurable (fun z : α × β => f z.2) (μ.prod ν) :=
  hf.comp_quasiMeasurePreserving quasiMeasurePreserving_snd


theorem lintegral_prod_swap [SFinite μ] (f : α × β → ℝ≥0∞) :
    ∫⁻ z, f z.swap ∂ν.prod μ = ∫⁻ z, f z ∂μ.prod ν :=
  measurePreserving_swap.lintegral_comp_emb MeasurableEquiv.prodComm.measurableEmbedding f


/-- **Tonelli's Theorem**: For `ℝ≥0∞`-valued measurable functions on `α × β`,
  the integral of `f` is equal to the iterated integral. -/
theorem lintegral_prod_of_measurable :
    ∀ (f : α × β → ℝ≥0∞), Measurable f → ∫⁻ z, f z ∂μ.prod ν = ∫⁻ x, ∫⁻ y, f (x, y) ∂ν ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    ⊢ ∀ (f : Prod α β → ENNReal), Measurable f → Eq (MeasureTheory.lintegral (μ.pr …
  -/
  have m := @measurable_prod_mk_left
  refine Measurable.ennreal_induction
    (P := fun f => ∫⁻ z, f z ∂μ.prod ν = ∫⁻ x, ∫⁻ y, f (x, y) ∂ν ∂μ) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type ?u.164136} {β : Type ?u.164135} {m : MeasurableSpace α} {mβ :  …
      ⊢ ∀ (c : ENNReal) ⦃s : Set (Prod α β)⦄, MeasurableSet s → (fun f => Eq (Measur …
    -/
  · intro c s hs
    conv_rhs =>
      enter [2, x, 2, y]
      rw [← indicator_comp_right, const_def, const_comp, ← const_def]
    conv_rhs =>
      enter [2, x]
      rw [lintegral_indicator (m (x := x) hs), lintegral_const,
        Measure.restrict_apply MeasurableSet.univ, univ_inter]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      c : ENNReal
      s : Set (Prod α β)
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => s.indicator (fun x => c) z)  …
    -/
    simp [hs, lintegral_const_mul, measurable_measure_prod_mk_left (ν := ν) hs, prod_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      ⊢ ∀ ⦃f g : Prod α β → ENNReal⦄, Disjoint (Function.support f) (Function.suppor …
    -/
  · rintro f g - hf _ h2f h2g
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f g : Prod α β → ENNReal
      hf : Measurable f
      a✝ : Measurable g
      h2f : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lint …
      h2g : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => g z) (MeasureTheory.lint …
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => HAdd.hAdd f g z) (MeasureThe …
    -/
    simp only [Pi.add_apply]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f g : Prod α β → ENNReal
      hf : Measurable f
      a✝ : Measurable g
      h2f : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lint …
      h2g : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => g z) (MeasureTheory.lint …
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => HAdd.hAdd (f z) (g z)) (Meas …
    -/
    conv_lhs => rw [lintegral_add_left hf]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f g : Prod α β → ENNReal
      hf : Measurable f
      a✝ : Measurable g
      h2f : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lint …
      h2g : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => g z) (MeasureTheory.lint …
      ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.prod ν) fun a => f a) (MeasureTheo …
    -/
    conv_rhs => enter [2, x]; erw [lintegral_add_left (hf.comp (m (x := x)))]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f g : Prod α β → ENNReal
      hf : Measurable f
      a✝ : Measurable g
      h2f : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lint …
      h2g : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => g z) (MeasureTheory.lint …
      ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.prod ν) fun a => f a) (MeasureTheo …
    -/
    simp [lintegral_add_left, Measurable.lintegral_prod_right', hf, h2f, h2g]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      ⊢ ∀ ⦃f : Nat → Prod α β → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
    -/
  · intro f hf h2f h3f
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f …
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => (fun x => iSup fun n => f n  …
    -/
    have kf : ∀ x n, Measurable fun y => f n (x, y) := fun x n => (hf n).comp m
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f …
      kf : ∀ (x : α) (n : Nat), Measurable fun y => f n { fst := x, snd := y }
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => (fun x => iSup fun n => f n  …
    -/
    have k2f : ∀ x, Monotone fun n y => f n (x, y) := fun x i j hij y => h2f hij (x, y)
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f …
      kf : ∀ (x : α) (n : Nat), Measurable fun y => f n { fst := x, snd := y }
      k2f : ∀ (x : α), Monotone fun n y => f n { fst := x, snd := y }
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => (fun x => iSup fun n => f n  …
    -/
    have lf : ∀ n, Measurable fun x => ∫⁻ y, f n (x, y) ∂ν := fun n => (hf n).lintegral_prod_right'
    have l2f : Monotone fun n x => ∫⁻ y, f n (x, y) ∂ν := fun i j hij x =>
      lintegral_mono (k2f x hij)
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝ : MeasureTheory.SFinite ν
      m : ∀ {α : Type u_1} {β : Type u_2} {m : MeasurableSpace α} {mβ : MeasurableSp …
      f : Nat → Prod α β → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      h2f : Monotone f
      h3f : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f …
      kf : ∀ (x : α) (n : Nat), Measurable fun y => f n { fst := x, snd := y }
      k2f : ∀ (x : α), Monotone fun n y => f n { fst := x, snd := y }
      lf : ∀ (n : Nat), Measurable fun x => MeasureTheory.lintegral ν fun y => f n { …
      l2f : Monotone fun n x => MeasureTheory.lintegral ν fun y => f n { fst := x, s …
      ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => (fun x => iSup fun n => f n  …
    -/
    simp only [lintegral_iSup hf h2f, lintegral_iSup (kf _), k2f, lintegral_iSup lf l2f, h3f]
    /-
      🎉 no goals
    -/


/-- **Tonelli's Theorem**: For `ℝ≥0∞`-valued almost everywhere measurable functions on `α × β`,
  the integral of `f` is equal to the iterated integral. -/
theorem lintegral_prod (f : α × β → ℝ≥0∞) (hf : AEMeasurable f (μ.prod ν)) :
    ∫⁻ z, f z ∂μ.prod ν = ∫⁻ x, ∫⁻ y, f (x, y) ∂ν ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → ENNReal
    hf : AEMeasurable f (μ.prod ν)
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lintegra …
  -/
  have A : ∫⁻ z, f z ∂μ.prod ν = ∫⁻ z, hf.mk f z ∂μ.prod ν := lintegral_congr_ae hf.ae_eq_mk
  have B : (∫⁻ x, ∫⁻ y, f (x, y) ∂ν ∂μ) = ∫⁻ x, ∫⁻ y, hf.mk f (x, y) ∂ν ∂μ := by
    apply lintegral_congr_ae
    filter_upwards [ae_ae_of_ae_prod hf.ae_eq_mk] with _ ha using lintegral_congr_ae ha
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → ENNReal
    hf : AEMeasurable f (μ.prod ν)
    A : Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.linteg …
    B : Eq (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun y =>  …
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lintegra …
  -/
  rw [A, B, lintegral_prod_of_measurable _ hf.measurable_mk]
  /-
    🎉 no goals
  -/


/-- The symmetric version of Tonelli's Theorem: For `ℝ≥0∞`-valued almost everywhere measurable
functions on `α × β`, the integral of `f` is equal to the iterated integral, in reverse order. -/
theorem lintegral_prod_symm [SFinite μ] (f : α × β → ℝ≥0∞) (hf : AEMeasurable f (μ.prod ν)) :
    ∫⁻ z, f z ∂μ.prod ν = ∫⁻ y, ∫⁻ x, f (x, y) ∂μ ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → ENNReal
    hf : AEMeasurable f (μ.prod ν)
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f z) (MeasureTheory.lintegra …
  -/
  simp_rw [← lintegral_prod_swap f]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → ENNReal
    hf : AEMeasurable f (μ.prod ν)
    ⊢ Eq (MeasureTheory.lintegral (ν.prod μ) fun z => f z.swap) (MeasureTheory.lin …
  -/
  exact lintegral_prod _ hf.prod_swap
  /-
    🎉 no goals
  -/


/-- The symmetric version of Tonelli's Theorem: For `ℝ≥0∞`-valued measurable
functions on `α × β`, the integral of `f` is equal to the iterated integral, in reverse order. -/
theorem lintegral_prod_symm' [SFinite μ] (f : α × β → ℝ≥0∞) (hf : Measurable f) :
    ∫⁻ z, f z ∂μ.prod ν = ∫⁻ y, ∫⁻ x, f (x, y) ∂μ ∂ν :=
  lintegral_prod_symm f hf.aemeasurable


/-- The reversed version of **Tonelli's Theorem**. In this version `f` is in curried form, which
makes it easier for the elaborator to figure out `f` automatically. -/
theorem lintegral_lintegral ⦃f : α → β → ℝ≥0∞⦄ (hf : AEMeasurable (uncurry f) (μ.prod ν)) :
    ∫⁻ x, ∫⁻ y, f x y ∂ν ∂μ = ∫⁻ z, f z.1 z.2 ∂μ.prod ν :=
  (lintegral_prod _ hf).symm


/-- The reversed version of **Tonelli's Theorem** (symmetric version). In this version `f` is in
curried form, which makes it easier for the elaborator to figure out `f` automatically. -/
theorem lintegral_lintegral_symm [SFinite μ] ⦃f : α → β → ℝ≥0∞⦄
    (hf : AEMeasurable (uncurry f) (μ.prod ν)) :
    ∫⁻ x, ∫⁻ y, f x y ∂ν ∂μ = ∫⁻ z, f z.2 z.1 ∂ν.prod μ :=
  (lintegral_prod_symm _ hf.prod_swap).symm


/-- Change the order of Lebesgue integration. -/
theorem lintegral_lintegral_swap [SFinite μ] ⦃f : α → β → ℝ≥0∞⦄
    (hf : AEMeasurable (uncurry f) (μ.prod ν)) :
    ∫⁻ x, ∫⁻ y, f x y ∂ν ∂μ = ∫⁻ y, ∫⁻ x, f x y ∂μ ∂ν :=
  (lintegral_lintegral hf).trans (lintegral_prod_symm _ hf)


theorem lintegral_prod_mul {f : α → ℝ≥0∞} {g : β → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g ν) : ∫⁻ z, f z.1 * g z.2 ∂μ.prod ν = (∫⁻ x, f x ∂μ) * ∫⁻ y, g y ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    f : α → ENNReal
    g : β → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g ν
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => HMul.hMul (f z.1) (g z.2)) ( …
  -/
  simp [lintegral_prod _ (hf.fst.mul hg.snd), lintegral_lintegral_mul hf hg]
  /-
    🎉 no goals
  -/


/-- Marginal measure on `α` obtained from a measure `ρ` on `α × β`, defined by `ρ.map Prod.fst`. -/
noncomputable def fst (ρ : Measure (α × β)) : Measure α :=
  ρ.map Prod.fst


theorem fst_apply {s : Set α} (hs : MeasurableSet s) : ρ.fst s = ρ (Prod.fst ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (ρ.fst s) (ρ (Set.preimage Prod.fst s))
  -/
  rw [fst, Measure.map_apply measurable_fst hs]
  /-
    🎉 no goals
  -/


                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝¹ : MeasurableSpace α
                                               inst✝ : MeasurableSpace β
                                               ρ : MeasureTheory.Measure (Prod α β)
                                               ⊢ Eq (ρ.fst Set.univ) (ρ Set.univ)
                                             -/
theorem fst_univ : ρ.fst univ = ρ univ := by rw [fst_apply MeasurableSet.univ, preimage_univ]
                                             /-
                                               🎉 no goals
                                             -/


                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 inst✝¹ : MeasurableSpace α
                                                                 inst✝ : MeasurableSpace β
                                                                 ⊢ Eq (MeasureTheory.Measure.fst 0) 0
                                                               -/
@[simp] theorem fst_zero : fst (0 : Measure (α × β)) = 0 := by simp [fst]
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance [SFinite ρ] : SFinite ρ.fst := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ MeasureTheory.SFinite ρ.fst
  -/
  rw [fst]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map Prod.fst ρ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance fst.instIsFiniteMeasure [IsFiniteMeasure ρ] : IsFiniteMeasure ρ.fst := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ MeasureTheory.IsFiniteMeasure ρ.fst
  -/
  rw [fst]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map Prod.fst ρ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance fst.instIsProbabilityMeasure [IsProbabilityMeasure ρ] : IsProbabilityMeasure ρ.fst where
  measure_univ := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite ν
      ρ : MeasureTheory.Measure (Prod α β)
      inst✝ : MeasureTheory.IsProbabilityMeasure ρ
      ⊢ Eq (ρ.fst Set.univ) 1
    -/
    rw [fst_univ]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite ν
      ρ : MeasureTheory.Measure (Prod α β)
      inst✝ : MeasureTheory.IsProbabilityMeasure ρ
      ⊢ Eq (ρ Set.univ) 1
    -/
    exact measure_univ
    /-
      🎉 no goals
    -/


@[simp]
lemma fst_prod [IsProbabilityMeasure ν] : (μ.prod ν).fst = μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    ⊢ Eq (μ.prod ν).fst μ
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.prod ν).fst s) (μ s)
  -/
  rw [fst_apply hs, ← prod_univ, prod_prod, measure_univ, mul_one]
  /-
    🎉 no goals
  -/


theorem fst_map_prod_mk₀ {X : α → β} {Y : α → γ} {μ : Measure α}
    (hY : AEMeasurable Y μ) : (μ.map fun a => (X a, Y a)).fst = μ.map X := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    X : α → β
    Y : α → γ
    μ : MeasureTheory.Measure α
    hY : AEMeasurable Y μ
    ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).fst ( …
  -/
  by_cases hX : AEMeasurable X μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      X : α → β
      Y : α → γ
      μ : MeasureTheory.Measure α
      hY : AEMeasurable Y μ
      hX : AEMeasurable X μ
      ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).fst ( …
    -/
  · ext1 s hs
    rw [Measure.fst_apply hs, Measure.map_apply_of_aemeasurable (hX.prod_mk hY) (measurable_fst hs),
      Measure.map_apply_of_aemeasurable hX hs, ← prod_univ, mk_preimage_prod, preimage_univ,
      inter_univ]
  · have : ¬AEMeasurable (fun x ↦ (X x, Y x)) μ := by
      contrapose! hX; exact measurable_fst.comp_aemeasurable hX
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      X : α → β
      Y : α → γ
      μ : MeasureTheory.Measure α
      hY : AEMeasurable Y μ
      hX : Not (AEMeasurable X μ)
      this : Not (AEMeasurable (fun x => { fst := X x, snd := Y x }) μ)
      ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).fst ( …
    -/
    simp [map_of_not_aemeasurable, hX, this]
    /-
      🎉 no goals
    -/


theorem fst_map_prod_mk {X : α → β} {Y : α → γ} {μ : Measure α}
    (hY : Measurable Y) : (μ.map fun a => (X a, Y a)).fst = μ.map X :=
  fst_map_prod_mk₀ hY.aemeasurable


@[simp]
lemma fst_add {μ ν : Measure (α × β)} : (μ + ν).fst = μ.fst + ν.fst := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (HAdd.hAdd μ ν).fst (HAdd.hAdd μ.fst ν.fst)
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν : MeasureTheory.Measure (Prod α β)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((HAdd.hAdd μ ν).fst s) ((HAdd.hAdd μ.fst ν.fst) s)
  -/
  simp_rw [coe_add, Pi.add_apply, fst_apply hs, coe_add, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma fst_sum {ι : Type*} (μ : ι → Measure (α × β)) : (sum μ).fst = sum (fun n ↦ (μ n).fst) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_4
    μ : ι → MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.sum μ).fst (MeasureTheory.Measure.sum fun n => (μ  …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_4
    μ : ι → MeasureTheory.Measure (Prod α β)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum μ).fst s) ((MeasureTheory.Measure.sum fun n = …
  -/
  rw [fst_apply hs, sum_apply, sum_apply _ hs]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      ι : Type u_4
      μ : ι → MeasureTheory.Measure (Prod α β)
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (tsum fun i => (μ i) (Set.preimage Prod.fst s)) (tsum fun i => (μ i).fst s)
    -/
  · simp_rw [fst_apply hs]
    /-
      🎉 no goals
    -/
    /-
      case h.hs
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      ι : Type u_4
      μ : ι → MeasureTheory.Measure (Prod α β)
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.preimage Prod.fst s)
    -/
  · exact measurable_fst hs
    /-
      🎉 no goals
    -/


@[gcongr]
theorem fst_mono {μ : Measure (α × β)} (h : ρ ≤ μ) : ρ.fst ≤ μ.fst := map_mono h measurable_fst


/-- Marginal measure on `β` obtained from a measure on `ρ` `α × β`, defined by `ρ.map Prod.snd`. -/
noncomputable def snd (ρ : Measure (α × β)) : Measure β :=
  ρ.map Prod.snd


theorem snd_apply {s : Set β} (hs : MeasurableSet s) : ρ.snd s = ρ (Prod.snd ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (ρ.snd s) (ρ (Set.preimage Prod.snd s))
  -/
  rw [snd, Measure.map_apply measurable_snd hs]
  /-
    🎉 no goals
  -/


                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝¹ : MeasurableSpace α
                                               inst✝ : MeasurableSpace β
                                               ρ : MeasureTheory.Measure (Prod α β)
                                               ⊢ Eq (ρ.snd Set.univ) (ρ Set.univ)
                                             -/
theorem snd_univ : ρ.snd univ = ρ univ := by rw [snd_apply MeasurableSet.univ, preimage_univ]
                                             /-
                                               🎉 no goals
                                             -/


                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 inst✝¹ : MeasurableSpace α
                                                                 inst✝ : MeasurableSpace β
                                                                 ⊢ Eq (MeasureTheory.Measure.snd 0) 0
                                                               -/
@[simp] theorem snd_zero : snd (0 : Measure (α × β)) = 0 := by simp [snd]
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance [SFinite ρ] : SFinite ρ.snd := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ MeasureTheory.SFinite ρ.snd
  -/
  rw [snd]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.map Prod.snd ρ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance snd.instIsFiniteMeasure [IsFiniteMeasure ρ] : IsFiniteMeasure ρ.snd := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ MeasureTheory.IsFiniteMeasure ρ.snd
  -/
  rw [snd]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ μ' : MeasureTheory.Measure α
    ν ν' : MeasureTheory.Measure β
    τ : MeasureTheory.Measure γ
    inst✝¹ : MeasureTheory.SFinite ν
    ρ : MeasureTheory.Measure (Prod α β)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map Prod.snd ρ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance snd.instIsProbabilityMeasure [IsProbabilityMeasure ρ] : IsProbabilityMeasure ρ.snd where
  measure_univ := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite ν
      ρ : MeasureTheory.Measure (Prod α β)
      inst✝ : MeasureTheory.IsProbabilityMeasure ρ
      ⊢ Eq (ρ.snd Set.univ) 1
    -/
    rw [snd_univ]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite ν
      ρ : MeasureTheory.Measure (Prod α β)
      inst✝ : MeasureTheory.IsProbabilityMeasure ρ
      ⊢ Eq (ρ Set.univ) 1
    -/
    exact measure_univ
    /-
      🎉 no goals
    -/


@[simp]
lemma snd_prod [IsProbabilityMeasure μ] : (μ.prod ν).snd = ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (μ.prod ν).snd ν
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((μ.prod ν).snd s) (ν s)
  -/
  rw [snd_apply hs, ← univ_prod, prod_prod, measure_univ, one_mul]
  /-
    🎉 no goals
  -/


theorem snd_map_prod_mk₀ {X : α → β} {Y : α → γ} {μ : Measure α} (hX : AEMeasurable X μ) :
    (μ.map fun a => (X a, Y a)).snd = μ.map Y := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    X : α → β
    Y : α → γ
    μ : MeasureTheory.Measure α
    hX : AEMeasurable X μ
    ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).snd ( …
  -/
  by_cases hY : AEMeasurable Y μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      X : α → β
      Y : α → γ
      μ : MeasureTheory.Measure α
      hX : AEMeasurable X μ
      hY : AEMeasurable Y μ
      ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).snd ( …
    -/
  · ext1 s hs
    rw [Measure.snd_apply hs, Measure.map_apply_of_aemeasurable (hX.prod_mk hY) (measurable_snd hs),
      Measure.map_apply_of_aemeasurable hY hs, ← univ_prod, mk_preimage_prod, preimage_univ,
      univ_inter]
  · have : ¬AEMeasurable (fun x ↦ (X x, Y x)) μ := by
      contrapose! hY; exact measurable_snd.comp_aemeasurable hY
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      X : α → β
      Y : α → γ
      μ : MeasureTheory.Measure α
      hX : AEMeasurable X μ
      hY : Not (AEMeasurable Y μ)
      this : Not (AEMeasurable (fun x => { fst := X x, snd := Y x }) μ)
      ⊢ Eq (MeasureTheory.Measure.map (fun a => { fst := X a, snd := Y a }) μ).snd ( …
    -/
    simp [map_of_not_aemeasurable, hY, this]
    /-
      🎉 no goals
    -/


theorem snd_map_prod_mk {X : α → β} {Y : α → γ} {μ : Measure α} (hX : Measurable X) :
    (μ.map fun a => (X a, Y a)).snd = μ.map Y :=
  snd_map_prod_mk₀ hX.aemeasurable


@[simp]
lemma snd_add {μ ν : Measure (α × β)} : (μ + ν).snd = μ.snd + ν.snd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (HAdd.hAdd μ ν).snd (HAdd.hAdd μ.snd ν.snd)
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν : MeasureTheory.Measure (Prod α β)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((HAdd.hAdd μ ν).snd s) ((HAdd.hAdd μ.snd ν.snd) s)
  -/
  simp_rw [coe_add, Pi.add_apply, snd_apply hs, coe_add, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma snd_sum {ι : Type*} (μ : ι → Measure (α × β)) : (sum μ).snd = sum (fun n ↦ (μ n).snd) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_4
    μ : ι → MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.sum μ).snd (MeasureTheory.Measure.sum fun n => (μ  …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ι : Type u_4
    μ : ι → MeasureTheory.Measure (Prod α β)
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum μ).snd s) ((MeasureTheory.Measure.sum fun n = …
  -/
  rw [snd_apply hs, sum_apply, sum_apply _ hs]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      ι : Type u_4
      μ : ι → MeasureTheory.Measure (Prod α β)
      s : Set β
      hs : MeasurableSet s
      ⊢ Eq (tsum fun i => (μ i) (Set.preimage Prod.snd s)) (tsum fun i => (μ i).snd s)
    -/
  · simp_rw [snd_apply hs]
    /-
      🎉 no goals
    -/
    /-
      case h.hs
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      ι : Type u_4
      μ : ι → MeasureTheory.Measure (Prod α β)
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.preimage Prod.snd s)
    -/
  · exact measurable_snd hs
    /-
      🎉 no goals
    -/


@[gcongr]
theorem snd_mono {μ : Measure (α × β)} (h : ρ ≤ μ) : ρ.snd ≤ μ.snd := map_mono h measurable_snd


@[simp] lemma fst_map_swap : (ρ.map Prod.swap).fst = ρ.snd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.map Prod.swap ρ).fst ρ.snd
  -/
  rw [Measure.fst, Measure.map_map measurable_fst measurable_swap]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp Prod.fst Prod.swap) ρ) ρ.snd
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma snd_map_swap : (ρ.map Prod.swap).snd = ρ.fst := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.map Prod.swap ρ).snd ρ.fst
  -/
  rw [Measure.snd, Measure.map_map measurable_snd measurable_swap]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ρ : MeasureTheory.Measure (Prod α β)
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp Prod.snd Prod.swap) ρ) ρ.fst
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The measurable equiv induced by the equiv `(α × β) × γ ≃ α × (β × γ)` is measure preserving. -/
theorem _root_.MeasureTheory.measurePreserving_prodAssoc (μa : Measure α) (μb : Measure β)
    (μc : Measure γ) [SFinite μb] [SFinite μc] :
    MeasurePreserving (MeasurableEquiv.prodAssoc : (α × β) × γ ≃ᵐ α × β × γ)
      ((μa.prod μb).prod μc) (μa.prod (μb.prod μc)) where
  measurable := MeasurableEquiv.prodAssoc.measurable
  map_eq := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite μb
      inst✝ : MeasureTheory.SFinite μc
      ⊢ Eq (MeasureTheory.Measure.map (⇑MeasurableEquiv.prodAssoc) ((μa.prod μb).pro …
    -/
    ext s hs
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite μb
      inst✝ : MeasureTheory.SFinite μc
      s : Set (Prod α (Prod β γ))
      hs : MeasurableSet s
      ⊢ Eq ((MeasureTheory.Measure.map (⇑MeasurableEquiv.prodAssoc) ((μa.prod μb).pr …
    -/
    have A (x : α) : MeasurableSet (Prod.mk x ⁻¹' s) := measurable_prod_mk_left hs
    have B : MeasurableSet (MeasurableEquiv.prodAssoc ⁻¹' s) :=
      MeasurableEquiv.prodAssoc.measurable hs
    simp_rw [map_apply MeasurableEquiv.prodAssoc.measurable hs, prod_apply hs, prod_apply (A _),
      prod_apply B, lintegral_prod _ (measurable_measure_prod_mk_left B).aemeasurable]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      inst✝¹ : MeasureTheory.SFinite μb
      inst✝ : MeasureTheory.SFinite μc
      s : Set (Prod α (Prod β γ))
      hs : MeasurableSet s
      A : ∀ (x : α), MeasurableSet (Set.preimage (Prod.mk x) s)
      B : MeasurableSet (Set.preimage (⇑MeasurableEquiv.prodAssoc) s)
      ⊢ Eq (MeasureTheory.lintegral μa fun x => MeasureTheory.lintegral μb fun y =>  …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem _root_.MeasureTheory.volume_preserving_prodAssoc {α₁ β₁ γ₁ : Type*} [MeasureSpace α₁]
    [MeasureSpace β₁] [MeasureSpace γ₁] [SFinite (volume : Measure β₁)]
    [SFinite (volume : Measure γ₁)] :
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁸ : MeasurableSpace α
      inst✝⁷ : MeasurableSpace β
      inst✝⁶ : MeasurableSpace γ
      μ μ' : MeasureTheory.Measure α
      ν ν' : MeasureTheory.Measure β
      τ : MeasureTheory.Measure γ
      inst✝⁵ : MeasureTheory.SFinite ν
      α₁ : Type u_4
      β₁ : Type u_5
      γ₁ : Type u_6
      inst✝⁴ : MeasureTheory.MeasureSpace α₁
      inst✝³ : MeasureTheory.MeasureSpace β₁
      inst✝² : MeasureTheory.MeasureSpace γ₁
      inst✝¹ : MeasureTheory.SFinite MeasureTheory.MeasureSpace.volume
      inst✝ : MeasureTheory.SFinite MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.Measure (Prod (Prod α₁ β₁) γ₁)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (MeasurableEquiv.prodAssoc : (α₁ × β₁) × γ₁ ≃ᵐ α₁ × β₁ × γ₁) :=
    /-
      🎉 no goals
    -/
  MeasureTheory.measurePreserving_prodAssoc volume volume volume


