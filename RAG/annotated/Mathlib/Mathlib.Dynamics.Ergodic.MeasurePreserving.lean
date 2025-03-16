/-- `f` is a measure preserving map w.r.t. measures `μa` and `μb` if `f` is measurable
and `map f μa = μb`. -/
structure MeasurePreserving (f : α → β)
  (μa : Measure α := by volume_tac) (μb : Measure β := by volume_tac) : Prop where
  protected measurable : Measurable f
  protected map_eq : map f μa = μb


protected theorem _root_.Measurable.measurePreserving
    {f : α → β} (h : Measurable f) (μa : Measure α) : MeasurePreserving f μa (map f μa) :=
  ⟨h, rfl⟩


protected theorem id (μ : Measure α) : MeasurePreserving id μ μ :=
  ⟨measurable_id, map_id⟩


protected theorem aemeasurable {f : α → β} (hf : MeasurePreserving f μa μb) : AEMeasurable f μa :=
  hf.1.aemeasurable


@[nontriviality]
theorem of_isEmpty [IsEmpty β] (f : α → β) (μa : Measure α) (μb : Measure β) :
    MeasurePreserving f μa μb :=
  ⟨measurable_of_subsingleton_codomain _, Subsingleton.elim _ _⟩


theorem symm (e : α ≃ᵐ β) {μa : Measure α} {μb : Measure β} (h : MeasurePreserving e μa μb) :
    MeasurePreserving e.symm μb μa :=
  ⟨e.symm.measurable, by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      e : MeasurableEquiv α β
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      h : MeasureTheory.MeasurePreserving (⇑e) μa μb
      ⊢ Eq (MeasureTheory.Measure.map (⇑e.symm) μb) μa
    -/
    rw [← h.map_eq, map_map e.symm.measurable e.measurable, e.symm_comp_self, map_id]⟩
    /-
      🎉 no goals
    -/


theorem restrict_preimage {f : α → β} (hf : MeasurePreserving f μa μb) {s : Set β}
    (hs : MeasurableSet s) : MeasurePreserving f (μa.restrict (f ⁻¹' s)) (μb.restrict s) :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝¹ : MeasurableSpace α
                       inst✝ : MeasurableSpace β
                       μa : MeasureTheory.Measure α
                       μb : MeasureTheory.Measure β
                       f : α → β
                       hf : MeasureTheory.MeasurePreserving f μa μb
                       s : Set β
                       hs : MeasurableSet s
                       ⊢ Eq (MeasureTheory.Measure.map f (μa.restrict (Set.preimage f s))) (μb.restri …
                     -/
  ⟨hf.measurable, by rw [← hf.map_eq, restrict_map hf.measurable hs]⟩
                     /-
                       🎉 no goals
                     -/


theorem restrict_preimage_emb {f : α → β} (hf : MeasurePreserving f μa μb)
    (h₂ : MeasurableEmbedding f) (s : Set β) :
    MeasurePreserving f (μa.restrict (f ⁻¹' s)) (μb.restrict s) :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝¹ : MeasurableSpace α
                       inst✝ : MeasurableSpace β
                       μa : MeasureTheory.Measure α
                       μb : MeasureTheory.Measure β
                       f : α → β
                       hf : MeasureTheory.MeasurePreserving f μa μb
                       h₂ : MeasurableEmbedding f
                       s : Set β
                       ⊢ Eq (MeasureTheory.Measure.map f (μa.restrict (Set.preimage f s))) (μb.restri …
                     -/
  ⟨hf.measurable, by rw [← hf.map_eq, h₂.restrict_map]⟩
                     /-
                       🎉 no goals
                     -/


theorem restrict_image_emb {f : α → β} (hf : MeasurePreserving f μa μb) (h₂ : MeasurableEmbedding f)
    (s : Set α) : MeasurePreserving f (μa.restrict s) (μb.restrict (f '' s)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    h₂ : MeasurableEmbedding f
    s : Set α
    ⊢ MeasureTheory.MeasurePreserving f (μa.restrict s) (μb.restrict (Set.image f  …
  -/
  simpa only [Set.preimage_image_eq _ h₂.injective] using hf.restrict_preimage_emb h₂ (f '' s)
  /-
    🎉 no goals
  -/


theorem aemeasurable_comp_iff {f : α → β} (hf : MeasurePreserving f μa μb)
    (h₂ : MeasurableEmbedding f) {g : β → γ} : AEMeasurable (g ∘ f) μa ↔ AEMeasurable g μb := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    h₂ : MeasurableEmbedding f
    g : β → γ
    ⊢ Iff (AEMeasurable (Function.comp g f) μa) (AEMeasurable g μb)
  -/
  rw [← hf.map_eq, h₂.aemeasurable_map_iff]
  /-
    🎉 no goals
  -/


protected theorem quasiMeasurePreserving {f : α → β} (hf : MeasurePreserving f μa μb) :
    QuasiMeasurePreserving f μa μb :=
  ⟨hf.1, hf.2.absolutelyContinuous⟩


protected theorem comp {g : β → γ} {f : α → β} (hg : MeasurePreserving g μb μc)
    (hf : MeasurePreserving f μa μb) : MeasurePreserving (g ∘ f) μa μc :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        inst✝² : MeasurableSpace α
                        inst✝¹ : MeasurableSpace β
                        inst✝ : MeasurableSpace γ
                        μa : MeasureTheory.Measure α
                        μb : MeasureTheory.Measure β
                        μc : MeasureTheory.Measure γ
                        g : β → γ
                        f : α → β
                        hg : MeasureTheory.MeasurePreserving g μb μc
                        hf : MeasureTheory.MeasurePreserving f μa μb
                        ⊢ Eq (MeasureTheory.Measure.map (Function.comp g f) μa) μc
                      -/
  ⟨hg.1.comp hf.1, by rw [← map_map hg.1 hf.1, hf.2, hg.2]⟩
                      /-
                        🎉 no goals
                      -/


/-- An alias of `MeasureTheory.MeasurePreserving.comp` with a convenient defeq and argument order
for `MeasurableEquiv` -/
protected theorem trans {e : α ≃ᵐ β} {e' : β ≃ᵐ γ}
    {μa : Measure α} {μb : Measure β} {μc : Measure γ}
    (h : MeasurePreserving e μa μb) (h' : MeasurePreserving e' μb μc) :
    MeasurePreserving (e.trans e') μa μc :=
  h'.comp h


protected theorem comp_left_iff {g : α → β} {e : β ≃ᵐ γ} (h : MeasurePreserving e μb μc) :
    MeasurePreserving (e ∘ g) μa μc ↔ MeasurePreserving g μa μb := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv β γ
    h : MeasureTheory.MeasurePreserving (⇑e) μb μc
    ⊢ Iff (MeasureTheory.MeasurePreserving (Function.comp (⇑e) g) μa μc) (MeasureT …
  -/
  refine ⟨fun hg => ?_, fun hg => h.comp hg⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv β γ
    h : MeasureTheory.MeasurePreserving (⇑e) μb μc
    hg : MeasureTheory.MeasurePreserving (Function.comp (⇑e) g) μa μc
    ⊢ MeasureTheory.MeasurePreserving g μa μb
  -/
  convert (MeasurePreserving.symm e h).comp hg
  /-
    case h.e'_5
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv β γ
    h : MeasureTheory.MeasurePreserving (⇑e) μb μc
    hg : MeasureTheory.MeasurePreserving (Function.comp (⇑e) g) μa μc
    ⊢ Eq g (Function.comp (⇑e.symm) (Function.comp (⇑e) g))
  -/
  simp [← Function.comp_assoc e.symm e g]
  /-
    🎉 no goals
  -/


protected theorem comp_right_iff {g : α → β} {e : γ ≃ᵐ α} (h : MeasurePreserving e μc μa) :
    MeasurePreserving (g ∘ e) μc μb ↔ MeasurePreserving g μa μb := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv γ α
    h : MeasureTheory.MeasurePreserving (⇑e) μc μa
    ⊢ Iff (MeasureTheory.MeasurePreserving (Function.comp g ⇑e) μc μb) (MeasureThe …
  -/
  refine ⟨fun hg => ?_, fun hg => hg.comp h⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv γ α
    h : MeasureTheory.MeasurePreserving (⇑e) μc μa
    hg : MeasureTheory.MeasurePreserving (Function.comp g ⇑e) μc μb
    ⊢ MeasureTheory.MeasurePreserving g μa μb
  -/
  convert hg.comp (MeasurePreserving.symm e h)
  /-
    case h.e'_5
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    μc : MeasureTheory.Measure γ
    g : α → β
    e : MeasurableEquiv γ α
    h : MeasureTheory.MeasurePreserving (⇑e) μc μa
    hg : MeasureTheory.MeasurePreserving (Function.comp g ⇑e) μc μb
    ⊢ Eq g (Function.comp (Function.comp g ⇑e) ⇑e.symm)
  -/
  simp [Function.comp_assoc g e e.symm]
  /-
    🎉 no goals
  -/


protected theorem sigmaFinite {f : α → β} (hf : MeasurePreserving f μa μb) [SigmaFinite μb] :
    SigmaFinite μa :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              inst✝² : MeasurableSpace α
                                              inst✝¹ : MeasurableSpace β
                                              μa : MeasureTheory.Measure α
                                              μb : MeasureTheory.Measure β
                                              f : α → β
                                              hf : MeasureTheory.MeasurePreserving f μa μb
                                              inst✝ : MeasureTheory.SigmaFinite μb
                                              ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f μa)
                                            -/
  SigmaFinite.of_map μa hf.aemeasurable (by rwa [hf.map_eq])
                                            /-
                                              🎉 no goals
                                            -/


theorem measure_preimage {f : α → β} (hf : MeasurePreserving f μa μb) {s : Set β}
    (hs : NullMeasurableSet s μb) : μa (f ⁻¹' s) = μb s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    s : Set β
    hs : MeasureTheory.NullMeasurableSet s μb
    ⊢ Eq (μa (Set.preimage f s)) (μb s)
  -/
  rw [← hf.map_eq] at hs ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    s : Set β
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.map f μa)
    ⊢ Eq (μa (Set.preimage f s)) ((MeasureTheory.Measure.map f μa) s)
  -/
  rw [map_apply₀ hf.1.aemeasurable hs]
  /-
    🎉 no goals
  -/


theorem measure_preimage_emb {f : α → β} (hf : MeasurePreserving f μa μb)
    (hfe : MeasurableEmbedding f) (s : Set β) : μa (f ⁻¹' s) = μb s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μa : MeasureTheory.Measure α
    μb : MeasureTheory.Measure β
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μa μb
    hfe : MeasurableEmbedding f
    s : Set β
    ⊢ Eq (μa (Set.preimage f s)) (μb s)
  -/
  rw [← hf.map_eq, hfe.map_apply]
  /-
    🎉 no goals
  -/


theorem measure_preimage_equiv {f : α ≃ᵐ β} (hf : MeasurePreserving f μa μb) (s : Set β) :
    μa (f ⁻¹' s) = μb s :=
  measure_preimage_emb hf f.measurableEmbedding s


theorem aeconst_comp [MeasurableSingletonClass γ] {f : α → β} (hf : MeasurePreserving f μa μb)
    {g : β → γ} (hg : NullMeasurable g μb) :
    Filter.EventuallyConst (g ∘ f) (ae μa) ↔ Filter.EventuallyConst g (ae μb) :=
  exists_congr fun s ↦ and_congr_left fun hs ↦ by
    simp only [Filter.mem_map, mem_ae_iff, ← hf.measure_preimage (hg hs.measurableSet).compl,
      preimage_comp, preimage_compl]


theorem aeconst_preimage {f : α → β} (hf : MeasurePreserving f μa μb) {s : Set β}
    (hs : NullMeasurableSet s μb) :
    Filter.EventuallyConst (f ⁻¹' s) (ae μa) ↔ Filter.EventuallyConst s (ae μb) :=
  aeconst_comp hf hs.mem


theorem add_measure {f μa' μb'} (hf : MeasurePreserving f μa μb)
    (hf' : MeasurePreserving f μa' μb') : MeasurePreserving f (μa + μa') (μb + μb') where
  measurable := hf.measurable
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝¹ : MeasurableSpace α
                 inst✝ : MeasurableSpace β
                 μa : MeasureTheory.Measure α
                 μb : MeasureTheory.Measure β
                 f : α → β
                 μa' : MeasureTheory.Measure α
                 μb' : MeasureTheory.Measure β
                 hf : MeasureTheory.MeasurePreserving f μa μb
                 hf' : MeasureTheory.MeasurePreserving f μa' μb'
                 ⊢ Eq (MeasureTheory.Measure.map f (HAdd.hAdd μa μa')) (HAdd.hAdd μb μb')
               -/
  map_eq := by rw [Measure.map_add _ _ hf.measurable, hf.map_eq, hf'.map_eq]
               /-
                 🎉 no goals
               -/


theorem smul_measure {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] {f : α → β}
    (hf : MeasurePreserving f μa μb) (c : R) : MeasurePreserving f (c • μa) (c • μb) where
  measurable := hf.measurable
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝³ : MeasurableSpace α
                 inst✝² : MeasurableSpace β
                 μa : MeasureTheory.Measure α
                 μb : MeasureTheory.Measure β
                 R : Type u_5
                 inst✝¹ : SMul R ENNReal
                 inst✝ : IsScalarTower R ENNReal ENNReal
                 f : α → β
                 hf : MeasureTheory.MeasurePreserving f μa μb
                 c : R
                 ⊢ Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μa)) (HSMul.hSMul c μb)
               -/
  map_eq := by rw [Measure.map_smul, hf.map_eq]
               /-
                 🎉 no goals
               -/


protected theorem iterate (hf : MeasurePreserving f μ μ) :
    ∀ n, MeasurePreserving f^[n] μ μ
  | 0 => .id μ
  | n + 1 => (MeasurePreserving.iterate hf n).comp hf


open scoped symmDiff in
lemma measure_symmDiff_preimage_iterate_le
    (hf : MeasurePreserving f μ μ) (hs : NullMeasurableSet s μ) (n : ℕ) :
    μ (s ∆ (f^[n] ⁻¹' s)) ≤ n • μ (s ∆ (f ⁻¹' s)) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ⊢ LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f n) s))) (HSMul.hSMul n (μ  …
  -/
  induction' n with n ih; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ih : LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f n) s))) (HSMul.hSMul n  …
    ⊢ LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f (HAdd.hAdd n 1)) s))) (HSM …
  -/
  simp only [add_smul, one_smul, ← n.add_one]
  /-
    case succ
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    ih : LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f n) s))) (HSMul.hSMul n  …
    ⊢ LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f (HAdd.hAdd n 1)) s))) (HAd …
  -/
  refine le_trans (measure_symmDiff_le s (f^[n] ⁻¹' s) (f^[n+1] ⁻¹' s)) (add_le_add ih ?_)
  replace hs : NullMeasurableSet (s ∆ (f ⁻¹' s)) μ :=
    hs.symmDiff <| hs.preimage hf.quasiMeasurePreserving
  /-
    case succ
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    n : Nat
    ih : LE.le (μ (symmDiff s (Set.preimage (Nat.iterate f n) s))) (HSMul.hSMul n  …
    hs : MeasureTheory.NullMeasurableSet (symmDiff s (Set.preimage f s)) μ
    ⊢ LE.le (μ (symmDiff (Set.preimage (Nat.iterate f n) s) (Set.preimage (Nat.ite …
  -/
  rw [iterate_succ', preimage_comp, ← preimage_symmDiff, (hf.iterate n).measure_preimage hs]
  /-
    🎉 no goals
  -/


/-- If `μ univ < n * μ s` and `f` is a map preserving measure `μ`,
then for some `x ∈ s` and `0 < m < n`, `f^[m] x ∈ s`. -/
theorem exists_mem_iterate_mem_of_measure_univ_lt_mul_measure (hf : MeasurePreserving f μ μ)
    (hs : NullMeasurableSet s μ) {n : ℕ} (hvol : μ (Set.univ : Set α) < n * μ s) :
    ∃ x ∈ s, ∃ m ∈ Set.Ioo 0 n, f^[m] x ∈ s := by
  have A : ∀ m, NullMeasurableSet (f^[m] ⁻¹' s) μ := fun m ↦
    hs.preimage (hf.iterate m).quasiMeasurePreserving
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
    A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Membership.me …
  -/
  have B : ∀ m, μ (f^[m] ⁻¹' s) = μ s := fun m ↦ (hf.iterate m).measure_preimage hs
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
    A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
    B : ∀ (m : Nat), Eq (μ (Set.preimage (Nat.iterate f m) s)) (μ s)
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Membership.me …
  -/
  have : μ (univ : Set α) < ∑ m ∈ Finset.range n, μ (f^[m] ⁻¹' s) := by simpa [B]
  obtain ⟨i, hi, j, hj, hij, x, hxi : f^[i] x ∈ s, hxj : f^[j] x ∈ s⟩ :
      ∃ i < n, ∃ j < n, i ≠ j ∧ (f^[i] ⁻¹' s ∩ f^[j] ⁻¹' s).Nonempty := by
    simpa using exists_nonempty_inter_of_measure_univ_lt_sum_measure μ (fun m _ ↦ A m) this
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
    A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
    B : ∀ (m : Nat), Eq (μ (Set.preimage (Nat.iterate f m) s)) (μ s)
    this : LT.lt (μ Set.univ) ((Finset.range n).sum fun m => μ (Set.preimage (Nat. …
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j n
    hij : Ne i j
    x : α
    hxi : Membership.mem s (Nat.iterate f i x)
    hxj : Membership.mem s (Nat.iterate f j x)
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Membership.me …
  -/
  wlog hlt : i < j generalizing i j
    /-
      case intro.intro.intro.intro.intro.intro.intro.inr
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → α
      s : Set α
      hf : MeasureTheory.MeasurePreserving f μ μ
      hs : MeasureTheory.NullMeasurableSet s μ
      n : Nat
      hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
      A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
      B : ∀ (m : Nat), Eq (μ (Set.preimage (Nat.iterate f m) s)) (μ s)
      this✝ : LT.lt (μ Set.univ) ((Finset.range n).sum fun m => μ (Set.preimage (Nat …
      i : Nat
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j n
      hij : Ne i j
      x : α
      hxi : Membership.mem s (Nat.iterate f i x)
      hxj : Membership.mem s (Nat.iterate f j x)
      this : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → Ne i j → Membership.m …
      hlt : Not (LT.lt i j)
      ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Membership.me …
    -/
  · exact this j hj i hi hij.symm hxj hxi (hij.lt_or_lt.resolve_left hlt)
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
    A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
    B : ∀ (m : Nat), Eq (μ (Set.preimage (Nat.iterate f m) s)) (μ s)
    this : LT.lt (μ Set.univ) ((Finset.range n).sum fun m => μ (Set.preimage (Nat. …
    x : α
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j n
    hij : Ne i j
    hxi : Membership.mem s (Nat.iterate f i x)
    hxj : Membership.mem s (Nat.iterate f j x)
    hlt : LT.lt i j
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Membership.me …
  -/
  refine ⟨f^[i] x, hxi, j - i, ⟨tsub_pos_of_lt hlt, lt_of_le_of_lt (j.sub_le i) hj⟩, ?_⟩
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    n : Nat
    hvol : LT.lt (μ Set.univ) (HMul.hMul (↑n) (μ s))
    A : ∀ (m : Nat), MeasureTheory.NullMeasurableSet (Set.preimage (Nat.iterate f  …
    B : ∀ (m : Nat), Eq (μ (Set.preimage (Nat.iterate f m) s)) (μ s)
    this : LT.lt (μ Set.univ) ((Finset.range n).sum fun m => μ (Set.preimage (Nat. …
    x : α
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j n
    hij : Ne i j
    hxi : Membership.mem s (Nat.iterate f i x)
    hxj : Membership.mem s (Nat.iterate f j x)
    hlt : LT.lt i j
    ⊢ Membership.mem s (Nat.iterate f (HSub.hSub j i) (Nat.iterate f i x))
  -/
  rwa [← iterate_add_apply, tsub_add_cancel_of_le hlt.le]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-12")]
alias exists_mem_iterate_mem_of_volume_lt_mul_volume :=
  exists_mem_iterate_mem_of_measure_univ_lt_mul_measure


/-- A self-map preserving a finite measure is conservative: if `μ s ≠ 0`, then at least one point
`x ∈ s` comes back to `s` under iterations of `f`. Actually, a.e. point of `s` comes back to `s`
infinitely many times, see `MeasureTheory.MeasurePreserving.conservative` and theorems about
`MeasureTheory.Conservative`. -/
theorem exists_mem_iterate_mem [IsFiniteMeasure μ] (hf : MeasurePreserving f μ μ)
    (hs : NullMeasurableSet s μ) (hs' : μ s ≠ 0) : ∃ x ∈ s, ∃ m ≠ 0, f^[m] x ∈ s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : Ne (μ s) 0
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rcases ENNReal.exists_nat_mul_gt hs' (measure_ne_top μ (Set.univ : Set α)) with ⟨N, hN⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : Ne (μ s) 0
    N : Nat
    hN : LT.lt (μ Set.univ) (HMul.hMul (↑N) (μ s))
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  rcases hf.exists_mem_iterate_mem_of_measure_univ_lt_mul_measure hs hN with ⟨x, hx, m, hm, hmx⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.MeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : Ne (μ s) 0
    N : Nat
    hN : LT.lt (μ Set.univ) (HMul.hMul (↑N) (μ s))
    x : α
    hx : Membership.mem s x
    m : Nat
    hm : Membership.mem (Set.Ioo 0 N) m
    hmx : Membership.mem s (Nat.iterate f m x)
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun m => And (Ne m 0) (Memb …
  -/
  exact ⟨x, hx, m, hm.1.ne', hmx⟩
  /-
    🎉 no goals
  -/


theorem measurePreserving_symm (μ : Measure α) (e : α ≃ᵐ β) :
    MeasurePreserving e.symm (map e μ) μ :=
  (e.measurable.measurePreserving μ).symm _


