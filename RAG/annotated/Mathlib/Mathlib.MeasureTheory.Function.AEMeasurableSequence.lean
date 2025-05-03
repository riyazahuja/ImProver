/-- If we have the additional hypothesis `∀ᵐ x ∂μ, p x (fun n ↦ f n x)`, this is a measurable set
whose complement has measure 0 such that for all `x ∈ aeSeqSet`, `f i x` is equal to
`(hf i).mk (f i) x` for all `i` and we have the pointwise property `p x (fun n ↦ f n x)`. -/
def aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) (p : α → (ι → β) → Prop) : Set α :=
  (toMeasurable μ { x | (∀ i, f i x = (hf i).mk (f i) x) ∧ p x fun n => f n x }ᶜ)ᶜ


open Classical in
/-- A sequence of measurable functions that are equal to `f` and verify property `p` on the
measurable set `aeSeqSet hf p`. -/
noncomputable def aeSeq (hf : ∀ i, AEMeasurable (f i) μ) (p : α → (ι → β) → Prop) : ι → α → β :=
  fun i x => ite (x ∈ aeSeqSet hf p) ((hf i).mk (f i) x) (⟨f i x⟩ : Nonempty β).some


theorem mk_eq_fun_of_mem_aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) {x : α} (hx : x ∈ aeSeqSet hf p)
    (i : ι) : (hf i).mk (f i) x = f i x :=
  haveI h_ss : aeSeqSet hf p ⊆ { x | ∀ i, f i x = (hf i).mk (f i) x } := by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : ι → α → β
      μ : MeasureTheory.Measure α
      p : α → (ι → β) → Prop
      hf : ∀ (i : ι), AEMeasurable (f i) μ
      x : α
      hx : Membership.mem (aeSeqSet hf p) x
      i : ι
      ⊢ HasSubset.Subset (aeSeqSet hf p) (setOf fun x => ∀ (i : ι), Eq (f i x) (AEMe …
    -/
    rw [aeSeqSet, ← compl_compl { x | ∀ i, f i x = (hf i).mk (f i) x }, Set.compl_subset_compl]
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : ι → α → β
      μ : MeasureTheory.Measure α
      p : α → (ι → β) → Prop
      hf : ∀ (i : ι), AEMeasurable (f i) μ
      x : α
      hx : Membership.mem (aeSeqSet hf p) x
      i : ι
      ⊢ HasSubset.Subset (HasCompl.compl (setOf fun x => ∀ (i : ι), Eq (f i x) (AEMe …
    -/
    refine Set.Subset.trans (Set.compl_subset_compl.mpr fun x h => ?_) (subset_toMeasurable _ _)
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f : ι → α → β
      μ : MeasureTheory.Measure α
      p : α → (ι → β) → Prop
      hf : ∀ (i : ι), AEMeasurable (f i) μ
      x✝ : α
      hx : Membership.mem (aeSeqSet hf p) x✝
      i : ι
      x : α
      h : Membership.mem (setOf fun x => And (∀ (i : ι), Eq (f i x) (AEMeasurable.mk …
      ⊢ Membership.mem (setOf fun x => ∀ (i : ι), Eq (f i x) (AEMeasurable.mk (f i)  …
    -/
    exact h.1
    /-
      🎉 no goals
    -/
  (h_ss hx i).symm


theorem aeSeq_eq_mk_of_mem_aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) {x : α}
    (hx : x ∈ aeSeqSet hf p) (i : ι) : aeSeq hf p i x = (hf i).mk (f i) x := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    i : ι
    ⊢ Eq (aeSeq hf p i x) (AEMeasurable.mk (f i) ⋯ x)
  -/
  simp only [aeSeq, hx, if_true]
  /-
    🎉 no goals
  -/


theorem aeSeq_eq_fun_of_mem_aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) {x : α}
    (hx : x ∈ aeSeqSet hf p) (i : ι) : aeSeq hf p i x = f i x := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    i : ι
    ⊢ Eq (aeSeq hf p i x) (f i x)
  -/
  simp only [aeSeq_eq_mk_of_mem_aeSeqSet hf hx i, mk_eq_fun_of_mem_aeSeqSet hf hx i]
  /-
    🎉 no goals
  -/


theorem prop_of_mem_aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) {x : α} (hx : x ∈ aeSeqSet hf p) :
    p x fun n => aeSeq hf p n x := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    ⊢ p x fun n => aeSeq hf p n x
  -/
  simp only [aeSeq, hx, if_true]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    ⊢ p x fun n => AEMeasurable.mk (f n) ⋯ x
  -/
  rw [funext fun n => mk_eq_fun_of_mem_aeSeqSet hf hx n]
  have h_ss : aeSeqSet hf p ⊆ { x | p x fun n => f n x } := by
    rw [← compl_compl { x | p x fun n => f n x }, aeSeqSet, Set.compl_subset_compl]
    refine Set.Subset.trans (Set.compl_subset_compl.mpr ?_) (subset_toMeasurable _ _)
    exact fun x hx => hx.2
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    h_ss : HasSubset.Subset (aeSeqSet hf p) (setOf fun x => p x fun n => f n x)
    ⊢ p x fun n => f n x
  -/
  have hx' := Set.mem_of_subset_of_mem h_ss hx
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    h_ss : HasSubset.Subset (aeSeqSet hf p) (setOf fun x => p x fun n => f n x)
    hx' : Membership.mem (setOf fun x => p x fun n => f n x) x
    ⊢ p x fun n => f n x
  -/
  exact hx'
  /-
    🎉 no goals
  -/


theorem fun_prop_of_mem_aeSeqSet (hf : ∀ i, AEMeasurable (f i) μ) {x : α} (hx : x ∈ aeSeqSet hf p) :
    p x fun n => f n x := by
  have h_eq : (fun n => f n x) = fun n => aeSeq hf p n x :=
    funext fun n => (aeSeq_eq_fun_of_mem_aeSeqSet hf hx n).symm
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    h_eq : Eq (fun n => f n x) fun n => aeSeq hf p n x
    ⊢ p x fun n => f n x
  -/
  rw [h_eq]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    x : α
    hx : Membership.mem (aeSeqSet hf p) x
    h_eq : Eq (fun n => f n x) fun n => aeSeq hf p n x
    ⊢ p x fun n => aeSeq hf p n x
  -/
  exact prop_of_mem_aeSeqSet hf hx
  /-
    🎉 no goals
  -/


theorem aeSeqSet_measurableSet {hf : ∀ i, AEMeasurable (f i) μ} : MeasurableSet (aeSeqSet hf p) :=
  (measurableSet_toMeasurable _ _).compl


theorem measurable (hf : ∀ i, AEMeasurable (f i) μ) (p : α → (ι → β) → Prop) (i : ι) :
    Measurable (aeSeq hf p i) :=
  Measurable.ite aeSeqSet_measurableSet (hf i).measurable_mk <| measurable_const' fun _ _ => rfl


theorem measure_compl_aeSeqSet_eq_zero [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n => f n x) : μ (aeSeqSet hf p)ᶜ = 0 := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    ⊢ Eq (μ (HasCompl.compl (aeSeqSet hf p))) 0
  -/
  rw [aeSeqSet, compl_compl, measure_toMeasurable]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    ⊢ Eq (μ (HasCompl.compl (setOf fun x => And (∀ (i : ι), Eq (f i x) (AEMeasurab …
  -/
  have hf_eq := fun i => (hf i).ae_eq_mk
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    hf_eq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (f i) (AEMeasurable.mk (f …
    ⊢ Eq (μ (HasCompl.compl (setOf fun x => And (∀ (i : ι), Eq (f i x) (AEMeasurab …
  -/
  simp_rw [Filter.EventuallyEq, ← ae_all_iff] at hf_eq
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    hf_eq : Filter.Eventually (fun a => ∀ (i : ι), Eq (f i a) (AEMeasurable.mk (f  …
    ⊢ Eq (μ (HasCompl.compl (setOf fun x => And (∀ (i : ι), Eq (f i x) (AEMeasurab …
  -/
  exact Filter.Eventually.and hf_eq hp
  /-
    🎉 no goals
  -/


theorem aeSeq_eq_mk_ae [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n => f n x) : ∀ᵐ a : α ∂μ, ∀ i : ι, aeSeq hf p i a = (hf i).mk (f i) a :=
  have h_ss : aeSeqSet hf p ⊆ { a : α | ∀ i, aeSeq hf p i a = (hf i).mk (f i) a } := fun x hx i =>
       /-
         ι : Sort u_1
         α : Type u_2
         β : Type u_3
         inst✝² : MeasurableSpace α
         inst✝¹ : MeasurableSpace β
         f : ι → α → β
         μ : MeasureTheory.Measure α
         p : α → (ι → β) → Prop
         inst✝ : Countable ι
         hf : ∀ (i : ι), AEMeasurable (f i) μ
         hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
         x : α
         hx : Membership.mem (aeSeqSet hf p) x
         i : ι
         ⊢ Eq (aeSeq hf p i x) (AEMeasurable.mk (f i) ⋯ x)
       -/
    by simp only [aeSeq, hx, if_true]
       /-
         🎉 no goals
       -/
  (ae_iff.2 (measure_compl_aeSeqSet_eq_zero hf hp)).mono h_ss


theorem aeSeq_eq_fun_ae [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n => f n x) : ∀ᵐ a : α ∂μ, ∀ i : ι, aeSeq hf p i a = f i a :=
  haveI h_ss : { a : α | ¬∀ i : ι, aeSeq hf p i a = f i a } ⊆ (aeSeqSet hf p)ᶜ := fun _ =>
    mt fun hx i => aeSeq_eq_fun_of_mem_aeSeqSet hf hx i
  measure_mono_null h_ss (measure_compl_aeSeqSet_eq_zero hf hp)


theorem aeSeq_n_eq_fun_n_ae [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n => f n x) (n : ι) : aeSeq hf p n =ᵐ[μ] f n :=
  ae_all_iff.mp (aeSeq_eq_fun_ae hf hp) n


theorem iSup [CompleteLattice β] [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n => f n x) : ⨆ n, aeSeq hf p n =ᵐ[μ] ⨆ n, f n := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝¹ : CompleteLattice β
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (_root_.iSup fun n => aeSeq hf p n) (_root …
  -/
  simp_rw [Filter.EventuallyEq, ae_iff, iSup_apply]
  have h_ss : aeSeqSet hf p ⊆ { a : α | ⨆ i : ι, aeSeq hf p i a = ⨆ i : ι, f i a } := by
    intro x hx
    congr
    exact funext fun i => aeSeq_eq_fun_of_mem_aeSeqSet hf hx i
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    f : ι → α → β
    μ : MeasureTheory.Measure α
    p : α → (ι → β) → Prop
    inst✝¹ : CompleteLattice β
    inst✝ : Countable ι
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hp : Filter.Eventually (fun x => p x fun n => f n x) (MeasureTheory.ae μ)
    h_ss : HasSubset.Subset (aeSeqSet hf p) (setOf fun a => Eq (_root_.iSup fun i  …
    ⊢ Eq (μ (setOf fun a => Not (Eq (_root_.iSup fun i => aeSeq hf p i a) (_root_. …
  -/
  exact measure_mono_null (Set.compl_subset_compl.mpr h_ss) (measure_compl_aeSeqSet_eq_zero hf hp)
  /-
    🎉 no goals
  -/


theorem iInf [CompleteLattice β] [Countable ι] (hf : ∀ i, AEMeasurable (f i) μ)
    (hp : ∀ᵐ x ∂μ, p x fun n ↦ f n x) : ⨅ n, aeSeq hf p n =ᵐ[μ] ⨅ n, f n :=
  iSup (β := βᵒᵈ) hf hp


