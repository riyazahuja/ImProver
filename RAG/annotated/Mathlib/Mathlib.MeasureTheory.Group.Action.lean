@[to_additive]
instance zero [MeasurableSpace α] [SMul M α] : SMulInvariantMeasure M α (0 : Measure α) :=
  ⟨fun _ _ _ => rfl⟩


@[to_additive]
instance add [SMulInvariantMeasure M α μ] [SMulInvariantMeasure M α ν] :
    SMulInvariantMeasure M α (μ + ν) :=
  ⟨fun c _s hs =>
    show _ + _ = _ + _ from
      congr_arg₂ (· + ·) (measure_preimage_smul c hs) (measure_preimage_smul c hs)⟩


@[to_additive]
instance smul [SMulInvariantMeasure M α μ] (c : ℝ≥0∞) : SMulInvariantMeasure M α (c • μ) :=
  ⟨fun a _s hs => show c • _ = c • _ from congr_arg (c • ·) (measure_preimage_smul a hs)⟩


@[to_additive]
instance smul_nnreal [SMulInvariantMeasure M α μ] (c : ℝ≥0) : SMulInvariantMeasure M α (c • μ) :=
  SMulInvariantMeasure.smul c


/-- See also `measure_preimage_smul_of_nullMeasurableSet` and `measure_preimage_smul`. -/
@[to_additive "See also `measure_preimage_smul_of_nullMeasurableSet` and `measure_preimage_smul`."]
theorem measure_preimage_smul_le (c : G) (s : Set α) : μ ((c • ·) ⁻¹' s) ≤ μ s :=
  (outerMeasure_le_iff (m := .map (c • ·) μ.1)).2
    (fun _s hs ↦ (SMulInvariantMeasure.measure_preimage_smul _ hs).le) _


/-- See also `smul_ae`. -/
@[to_additive "See also `vadd_ae`."]
theorem tendsto_smul_ae (c : G) : Filter.Tendsto (c • ·) (ae μ) (ae μ) := fun _s hs ↦
  eq_bot_mono (measure_preimage_smul_le μ c _) hs


@[to_additive]
theorem measure_preimage_smul_null (h : μ s = 0) (c : G) : μ ((c • ·) ⁻¹' s) = 0 :=
  eq_bot_mono (measure_preimage_smul_le μ c _) h


@[to_additive]
theorem measure_preimage_smul_of_nullMeasurableSet (hs : NullMeasurableSet s μ) (c : G) :
    μ ((c • ·) ⁻¹' s) = μ s := by
  rw [← measure_toMeasurable s,
    ← SMulInvariantMeasure.measure_preimage_smul c (measurableSet_toMeasurable μ s)]
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝¹ : SMul G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    c : G
    ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul c x) s)) (μ (Set.preimage (fun x = …
  -/
  exact measure_congr (tendsto_smul_ae μ c hs.toMeasurable_ae_eq) |>.symm
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem measure_preimage_smul (c : G) (s : Set α) : μ ((c • ·) ⁻¹' s) = μ s :=
  (measure_preimage_smul_le μ c s).antisymm <| by
    /-
      G : Type u
      α : Type w
      m : MeasurableSpace α
      inst✝² : Group G
      inst✝¹ : MulAction G α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
      c : G
      s : Set α
      ⊢ LE.le (μ s) (μ (Set.preimage (fun x => HSMul.hSMul c x) s))
    -/
    simpa [preimage_preimage] using measure_preimage_smul_le μ c⁻¹ ((c • ·) ⁻¹' s)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem measure_smul (c : G) (s : Set α) : μ (c • s) = μ s := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s : Set α
    ⊢ Eq (μ (HSMul.hSMul c s)) (μ s)
  -/
  simpa only [preimage_smul_inv] using measure_preimage_smul μ c⁻¹ s
  /-
    🎉 no goals
  -/


@[to_additive]
theorem measure_smul_eq_zero_iff {s} (c : G) : μ (c • s) = 0 ↔ μ s = 0 := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s : Set α
    c : G
    ⊢ Iff (Eq (μ (HSMul.hSMul c s)) 0) (Eq (μ s) 0)
  -/
  rw [measure_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem measure_smul_null {s} (h : μ s = 0) (c : G) : μ (c • s) = 0 :=
  (measure_smul_eq_zero_iff _).2 h


@[to_additive (attr := simp)]
theorem smul_mem_ae (c : G) {s : Set α} : c • s ∈ ae μ ↔ s ∈ ae μ := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s : Set α
    ⊢ Iff (Membership.mem (MeasureTheory.ae μ) (HSMul.hSMul c s)) (Membership.mem  …
  -/
  simp only [mem_ae_iff, ← smul_set_compl, measure_smul_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem smul_ae (c : G) : c • ae μ = ae μ := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    ⊢ Eq (HSMul.hSMul c (MeasureTheory.ae μ)) (MeasureTheory.ae μ)
  -/
  ext s
  /-
    case h
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s : Set α
    ⊢ Iff (Membership.mem (HSMul.hSMul c (MeasureTheory.ae μ)) s) (Membership.mem  …
  -/
  simp only [mem_smul_filter, preimage_smul, smul_mem_ae]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem eventuallyConst_smul_set_ae (c : G) {s : Set α} :
    EventuallyConst (c • s : Set α) (ae μ) ↔ EventuallyConst s (ae μ) := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s : Set α
    ⊢ Iff (Filter.EventuallyConst (HSMul.hSMul c s) (MeasureTheory.ae μ)) (Filter. …
  -/
  rw [← preimage_smul_inv, eventuallyConst_preimage, Filter.map_smul, smul_ae]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem smul_set_ae_le (c : G) {s t : Set α} : c • s ≤ᵐ[μ] c • t ↔ s ≤ᵐ[μ] t := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE (HSMul.hSMul c s) (HSMul.hSMul c t))  …
  -/
  simp only [ae_le_set, ← smul_set_sdiff, measure_smul_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem smul_set_ae_eq (c : G) {s t : Set α} : c • s =ᵐ[μ] c • t ↔ s =ᵐ[μ] t := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    c : G
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul c s) (HSMul.hSMul c t))  …
  -/
  simp only [Filter.eventuallyLE_antisymm_iff, smul_set_ae_le]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem measurePreserving_smul : MeasurePreserving (c • ·) μ μ :=
  { measurable := measurable_const_smul c
    map_eq := by
      /-
        M : Type v
        α : Type w
        m : MeasurableSpace α
        inst✝³ : MeasurableSpace M
        inst✝² : SMul M α
        inst✝¹ : MeasurableSMul M α
        c : M
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
        ⊢ Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul c x) μ) μ
      -/
      ext1 s hs
      /-
        case h
        M : Type v
        α : Type w
        m : MeasurableSpace α
        inst✝³ : MeasurableSpace M
        inst✝² : SMul M α
        inst✝¹ : MeasurableSMul M α
        c : M
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
        s : Set α
        hs : MeasurableSet s
        ⊢ Eq ((MeasureTheory.Measure.map (fun x => HSMul.hSMul c x) μ) s) (μ s)
      -/
      rw [map_apply (measurable_const_smul c) hs]
      /-
        case h
        M : Type v
        α : Type w
        m : MeasurableSpace α
        inst✝³ : MeasurableSpace M
        inst✝² : SMul M α
        inst✝¹ : MeasurableSMul M α
        c : M
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
        s : Set α
        hs : MeasurableSet s
        ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul c x) s)) (μ s)
      -/
      exact SMulInvariantMeasure.measure_preimage_smul c hs }
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem map_smul : map (c • ·) μ = μ :=
  (measurePreserving_smul c μ).map_eq


@[to_additive]
theorem MeasurePreserving.smulInvariantMeasure_iterateMulAct
    {f : α → α} {_ : MeasurableSpace α} {μ : Measure α} (hf : MeasurePreserving f μ μ) :
    SMulInvariantMeasure (IterateMulAct f) α μ :=
  ⟨fun n _s hs ↦ (hf.iterate n.val).measure_preimage hs.nullMeasurableSet⟩


@[to_additive]
theorem smulInvariantMeasure_iterateMulAct
    {f : α → α} {_ : MeasurableSpace α} {μ : Measure α} (hf : Measurable f) :
    SMulInvariantMeasure (IterateMulAct f) α μ ↔ MeasurePreserving f μ μ :=
  ⟨fun _ ↦
    have := hf.measurableSMul₂_iterateMulAct
    measurePreserving_smul (IterateMulAct.mk (f := f) 1) μ,
    MeasurePreserving.smulInvariantMeasure_iterateMulAct⟩


@[to_additive]
theorem smulInvariantMeasure_map [SMul M α] [SMul M β]
    [MeasurableSMul M β]
    (μ : Measure α) [SMulInvariantMeasure M α μ] (f : α → β)
    (hsmul : ∀ (m : M) a, f (m • a) = m • f a) (hf : Measurable f) :
    SMulInvariantMeasure M β (map f μ) where
  measure_preimage_smul m S hS := calc
    map f μ ((m • ·) ⁻¹' S)
    _ = μ (f ⁻¹' ((m • ·) ⁻¹' S)) := map_apply hf <| hS.preimage (measurable_const_smul _)
                                  /-
                                    M : Type uM
                                    α : Type uα
                                    β : Type uβ
                                    inst✝⁶ : MeasurableSpace M
                                    inst✝⁵ : MeasurableSpace α
                                    inst✝⁴ : MeasurableSpace β
                                    inst✝³ : SMul M α
                                    inst✝² : SMul M β
                                    inst✝¹ : MeasurableSMul M β
                                    μ : MeasureTheory.Measure α
                                    inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
                                    f : α → β
                                    hsmul : ∀ (m : M) (a : α), Eq (f (HSMul.hSMul m a)) (HSMul.hSMul m (f a))
                                    hf : Measurable f
                                    m : M
                                    S : Set β
                                    hS : MeasurableSet S
                                    ⊢ Eq (μ (Set.preimage f (Set.preimage (fun x => HSMul.hSMul m x) S))) (μ (Set. …
                                  -/
    _ = μ ((m • f ·) ⁻¹' S) := by rw [preimage_preimage]
                                  /-
                                    🎉 no goals
                                  -/
                                     /-
                                       M : Type uM
                                       α : Type uα
                                       β : Type uβ
                                       inst✝⁶ : MeasurableSpace M
                                       inst✝⁵ : MeasurableSpace α
                                       inst✝⁴ : MeasurableSpace β
                                       inst✝³ : SMul M α
                                       inst✝² : SMul M β
                                       inst✝¹ : MeasurableSMul M β
                                       μ : MeasureTheory.Measure α
                                       inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
                                       f : α → β
                                       hsmul : ∀ (m : M) (a : α), Eq (f (HSMul.hSMul m a)) (HSMul.hSMul m (f a))
                                       hf : Measurable f
                                       m : M
                                       S : Set β
                                       hS : MeasurableSet S
                                       ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul m (f x)) S)) (μ (Set.preimage (fun …
                                     -/
    _ = μ ((f <| m • ·) ⁻¹' S) := by simp_rw [hsmul]
                                     /-
                                       🎉 no goals
                                     -/
                                        /-
                                          M : Type uM
                                          α : Type uα
                                          β : Type uβ
                                          inst✝⁶ : MeasurableSpace M
                                          inst✝⁵ : MeasurableSpace α
                                          inst✝⁴ : MeasurableSpace β
                                          inst✝³ : SMul M α
                                          inst✝² : SMul M β
                                          inst✝¹ : MeasurableSMul M β
                                          μ : MeasureTheory.Measure α
                                          inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
                                          f : α → β
                                          hsmul : ∀ (m : M) (a : α), Eq (f (HSMul.hSMul m a)) (HSMul.hSMul m (f a))
                                          hf : Measurable f
                                          m : M
                                          S : Set β
                                          hS : MeasurableSet S
                                          ⊢ Eq (μ (Set.preimage (fun x => f (HSMul.hSMul m x)) S)) (μ (Set.preimage (fun …
                                        -/
    _ = μ ((m • ·) ⁻¹' (f ⁻¹' S)) := by rw [← preimage_preimage]
                                        /-
                                          🎉 no goals
                                        -/
                          /-
                            M : Type uM
                            α : Type uα
                            β : Type uβ
                            inst✝⁶ : MeasurableSpace M
                            inst✝⁵ : MeasurableSpace α
                            inst✝⁴ : MeasurableSpace β
                            inst✝³ : SMul M α
                            inst✝² : SMul M β
                            inst✝¹ : MeasurableSMul M β
                            μ : MeasureTheory.Measure α
                            inst✝ : MeasureTheory.SMulInvariantMeasure M α μ
                            f : α → β
                            hsmul : ∀ (m : M) (a : α), Eq (f (HSMul.hSMul m a)) (HSMul.hSMul m (f a))
                            hf : Measurable f
                            m : M
                            S : Set β
                            hS : MeasurableSet S
                            ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul m x) (Set.preimage f S))) (μ (Set. …
                          -/
    _ = μ (f ⁻¹' S) := by rw [SMulInvariantMeasure.measure_preimage_smul m (hS.preimage hf)]
                          /-
                            🎉 no goals
                          -/
    _ = map f μ S := (map_apply hf hS).symm


@[to_additive]
instance smulInvariantMeasure_map_smul [SMul M α] [SMul N α] [SMulCommClass N M α]
    [MeasurableSMul M α] [MeasurableSMul N α]
    (μ : Measure α) [SMulInvariantMeasure M α μ] (n : N) :
    SMulInvariantMeasure M α (map (n • ·) μ) :=
  smulInvariantMeasure_map μ _ (smul_comm n) <| measurable_const_smul _


variable [MeasurableSpace G] [MeasurableSMul G α] in
/-- Equivalent definitions of a measure invariant under a multiplicative action of a group.

- 0: `SMulInvariantMeasure G α μ`;

- 1: for every `c : G` and a measurable set `s`, the measure of the preimage of `s` under scalar
     multiplication by `c` is equal to the measure of `s`;

- 2: for every `c : G` and a measurable set `s`, the measure of the image `c • s` of `s` under
     scalar multiplication by `c` is equal to the measure of `s`;

- 3, 4: properties 2, 3 for any set, including non-measurable ones;

- 5: for any `c : G`, scalar multiplication by `c` maps `μ` to `μ`;

- 6: for any `c : G`, scalar multiplication by `c` is a measure preserving map. -/
@[to_additive]
theorem smulInvariantMeasure_tfae :
    List.TFAE
      [SMulInvariantMeasure G α μ,
        ∀ (c : G) (s), MeasurableSet s → μ ((c • ·) ⁻¹' s) = μ s,
        ∀ (c : G) (s), MeasurableSet s → μ (c • s) = μ s,
        ∀ (c : G) (s), μ ((c • ·) ⁻¹' s) = μ s,
        ∀ (c : G) (s), μ (c • s) = μ s,
        ∀ c : G, Measure.map (c • ·) μ = μ,
        ∀ c : G, MeasurePreserving (c • ·) μ μ] := by
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_have 1 ↔ 2 := ⟨fun h => h.1, fun h => ⟨h⟩⟩
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    tfae_1_iff_2 : Iff (MeasureTheory.SMulInvariantMeasure G α μ) (∀ (c : G) (s :  …
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_have 1 → 6 := fun h c => (measurePreserving_smul c μ).map_eq
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    tfae_1_iff_2 : Iff (MeasureTheory.SMulInvariantMeasure G α μ) (∀ (c : G) (s :  …
    tfae_1_to_6 : MeasureTheory.SMulInvariantMeasure G α μ → ∀ (c : G), Eq (Measur …
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_have 6 → 7 := fun H c => ⟨measurable_const_smul c, H c⟩
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    tfae_1_iff_2 : Iff (MeasureTheory.SMulInvariantMeasure G α μ) (∀ (c : G) (s :  …
    tfae_1_to_6 : MeasureTheory.SMulInvariantMeasure G α μ → ∀ (c : G), Eq (Measur …
    tfae_6_to_7 : (∀ (c : G), Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul  …
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_have 7 → 4 := fun H c => (H c).measure_preimage_emb (measurableEmbedding_const_smul c)
  tfae_have 4 → 5
  | H, c, s => by
    rw [← preimage_smul_inv]
    apply H
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    tfae_1_iff_2 : Iff (MeasureTheory.SMulInvariantMeasure G α μ) (∀ (c : G) (s :  …
    tfae_1_to_6 : MeasureTheory.SMulInvariantMeasure G α μ → ∀ (c : G), Eq (Measur …
    tfae_6_to_7 : (∀ (c : G), Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul  …
    tfae_7_to_4 : (∀ (c : G), MeasureTheory.MeasurePreserving (fun x => HSMul.hSMu …
    tfae_4_to_5 : (∀ (c : G) (s : Set α), Eq (μ (Set.preimage (fun x => HSMul.hSMu …
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_have 5 → 3 := fun H c s _ => H c s
  tfae_have 3 → 2
  | H, c, s, hs => by
    rw [preimage_smul]
    exact H c⁻¹ s hs
  /-
    G : Type u
    α : Type w
    m : MeasurableSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    tfae_1_iff_2 : Iff (MeasureTheory.SMulInvariantMeasure G α μ) (∀ (c : G) (s :  …
    tfae_1_to_6 : MeasureTheory.SMulInvariantMeasure G α μ → ∀ (c : G), Eq (Measur …
    tfae_6_to_7 : (∀ (c : G), Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul  …
    tfae_7_to_4 : (∀ (c : G), MeasureTheory.MeasurePreserving (fun x => HSMul.hSMu …
    tfae_4_to_5 : (∀ (c : G) (s : Set α), Eq (μ (Set.preimage (fun x => HSMul.hSMu …
    tfae_5_to_3 : (∀ (c : G) (s : Set α), Eq (μ (HSMul.hSMul c s)) (μ s)) → ∀ (c : …
    tfae_3_to_2 : (∀ (c : G) (s : Set α), MeasurableSet s → Eq (μ (HSMul.hSMul c s …
    ⊢ (List.cons (MeasureTheory.SMulInvariantMeasure G α μ) (List.cons (∀ (c : G)  …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


variable [MeasurableSpace G] [MeasurableSMul G α] in
@[to_additive]
theorem NullMeasurableSet.smul {s} (hs : NullMeasurableSet s μ) (c : G) :
    NullMeasurableSet (c • s) μ := by
  simpa only [← preimage_smul_inv] using
    hs.preimage (measurePreserving_smul _ _).quasiMeasurePreserving


include G in
/-- If measure `μ` is invariant under a group action and is nonzero on a compact set `K`, then it is
positive on any nonempty open set. In case of a regular measure, one can assume `μ ≠ 0` instead of
`μ K ≠ 0`, see `MeasureTheory.measure_isOpen_pos_of_smulInvariant_of_ne_zero`. -/
@[to_additive]
theorem measure_isOpen_pos_of_smulInvariant_of_compact_ne_zero (hK : IsCompact K) (hμK : μ K ≠ 0)
    (hU : IsOpen U) (hne : U.Nonempty) : 0 < μ U :=
  let ⟨t, ht⟩ := hK.exists_finite_cover_smul G hU hne
  pos_iff_ne_zero.2 fun hμU =>
    hμK <|
      measure_mono_null ht <|
                                                                     /-
                                                                       G : Type u
                                                                       α : Type w
                                                                       m : MeasurableSpace α
                                                                       inst✝⁵ : Group G
                                                                       inst✝⁴ : MulAction G α
                                                                       μ : MeasureTheory.Measure α
                                                                       inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
                                                                       inst✝² : TopologicalSpace α
                                                                       inst✝¹ : ContinuousConstSMul G α
                                                                       inst✝ : MulAction.IsMinimal G α
                                                                       K U : Set α
                                                                       hK : IsCompact K
                                                                       hμK : Ne (μ K) 0
                                                                       hU : IsOpen U
                                                                       hne : U.Nonempty
                                                                       t : Finset G
                                                                       ht : HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h => HSMul.hSMul g …
                                                                       hμU : Eq (μ U) 0
                                                                       x✝¹ : G
                                                                       x✝ : Membership.mem (↑t) x✝¹
                                                                       ⊢ Eq (μ (HSMul.hSMul x✝¹ U)) 0
                                                                     -/
        (measure_biUnion_null_iff t.countable_toSet).2 fun _ _ => by rwa [measure_smul]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[to_additive]
theorem isLocallyFiniteMeasure_of_smulInvariant (hU : IsOpen U) (hne : U.Nonempty) (hμU : μ U ≠ ∞) :
    IsLocallyFiniteMeasure μ :=
  ⟨fun x =>
    let ⟨g, hg⟩ := hU.exists_smul_mem G x hne
    ⟨(g • ·) ⁻¹' U, (hU.preimage (continuous_id.const_smul _)).mem_nhds hg,
                      /-
                        G : Type u
                        α : Type w
                        m : MeasurableSpace α
                        inst✝⁵ : Group G
                        inst✝⁴ : MulAction G α
                        μ : MeasureTheory.Measure α
                        inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
                        inst✝² : TopologicalSpace α
                        inst✝¹ : ContinuousConstSMul G α
                        inst✝ : MulAction.IsMinimal G α
                        U : Set α
                        hU : IsOpen U
                        hne : U.Nonempty
                        hμU : Ne (μ U) Top.top
                        x : α
                        g : G
                        hg : Membership.mem U (HSMul.hSMul g x)
                        ⊢ Ne (μ (Set.preimage (fun x => HSMul.hSMul g x) U)) Top.top
                      -/
      Ne.lt_top <| by rwa [measure_preimage_smul]⟩⟩
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem measure_isOpen_pos_of_smulInvariant_of_ne_zero (hμ : μ ≠ 0) (hU : IsOpen U)
    (hne : U.Nonempty) : 0 < μ U :=
  let ⟨_K, hK, hμK⟩ := Regular.exists_isCompact_not_null.mpr hμ
  measure_isOpen_pos_of_smulInvariant_of_compact_ne_zero G hK hμK hU hne


@[to_additive]
theorem measure_pos_iff_nonempty_of_smulInvariant (hμ : μ ≠ 0) (hU : IsOpen U) :
    0 < μ U ↔ U.Nonempty :=
  ⟨fun h => nonempty_of_measure_ne_zero h.ne',
    measure_isOpen_pos_of_smulInvariant_of_ne_zero G hμ hU⟩


@[to_additive]
theorem measure_eq_zero_iff_eq_empty_of_smulInvariant (hμ : μ ≠ 0) (hU : IsOpen U) :
    μ U = 0 ↔ U = ∅ := by
  rw [← not_iff_not, ← Ne, ← pos_iff_ne_zero,
    measure_pos_iff_nonempty_of_smulInvariant G hμ hU, nonempty_iff_ne_empty]


