@[to_additive]
instance [μ.IsMulLeftInvariant] : ErgodicSMul G G μ := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulLeftInvariant
    ⊢ ErgodicSMul G G μ
  -/
  refine ⟨fun {s} hsm hs ↦ ?_⟩
  suffices (∃ᵐ x ∂μ, x ∈ s) → ∀ᵐ x ∂μ, x ∈ s by
    simp only [eventuallyConst_set, ← not_frequently]
    exact or_not_of_imp this
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulLeftInvariant
    s : Set G
    hsm : MeasurableSet s
    hs : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
    ⊢ Filter.Frequently (fun x => Membership.mem s x) (MeasureTheory.ae μ) → Filte …
  -/
  intro hμs
  obtain ⟨a, has, ha⟩ : ∃ a ∈ s, ∀ᵐ b ∂μ, (b * a ∈ s ↔ a ∈ s) := by
    refine (hμs.and_eventually ?_).exists
    rw [ae_ae_comm]
    · exact ae_of_all _ fun b ↦ (hs b).mem_iff
    · exact ((hsm.preimage <| measurable_snd.mul measurable_fst).mem.iff
        (hsm.preimage measurable_fst).mem).setOf
  /-
    case intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulLeftInvariant
    s : Set G
    hsm : MeasurableSet s
    hs : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
    hμs : Filter.Frequently (fun x => Membership.mem s x) (MeasureTheory.ae μ)
    a : G
    has : Membership.mem s a
    ha : Filter.Eventually (fun b => Iff (Membership.mem s (HMul.hMul b a)) (Membe …
    ⊢ Filter.Eventually (fun x => Membership.mem s x) (MeasureTheory.ae μ)
  -/
  simpa [has] using (MeasureTheory.quasiMeasurePreserving_mul_right μ a⁻¹).ae ha
  /-
    🎉 no goals
  -/


@[to_additive]
instance [μ.IsMulRightInvariant] : ErgodicSMul Gᵐᵒᵖ G μ := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulRightInvariant
    ⊢ ErgodicSMul (MulOpposite G) G μ
  -/
  refine ⟨fun {s} hsm hs ↦ ?_⟩
  suffices (∃ᵐ x ∂μ, x ∈ s) → ∀ᵐ x ∂μ, x ∈ s by
    simp only [eventuallyConst_set, ← not_frequently]
    exact or_not_of_imp this
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulRightInvariant
    s : Set G
    hsm : MeasurableSet s
    hs : ∀ (g : MulOpposite G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (f …
    ⊢ Filter.Frequently (fun x => Membership.mem s x) (MeasureTheory.ae μ) → Filte …
  -/
  intro hμs
  obtain ⟨a, has, ha⟩ : ∃ a ∈ s, ∀ᵐ b ∂μ, (a * b ∈ s ↔ a ∈ s) := by
    refine (hμs.and_eventually ?_).exists
    rw [ae_ae_comm]
    · exact ae_of_all _ fun b ↦ (hs ⟨b⟩).mem_iff
    · exact ((hsm.preimage <| measurable_fst.mul measurable_snd).mem.iff
        (hsm.preimage measurable_fst).mem).setOf
  /-
    case intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableMul₂ G
    inst✝² : MeasurableInv G
    μ : MeasureTheory.Measure G
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.IsMulRightInvariant
    s : Set G
    hsm : MeasurableSet s
    hs : ∀ (g : MulOpposite G), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (f …
    hμs : Filter.Frequently (fun x => Membership.mem s x) (MeasureTheory.ae μ)
    a : G
    has : Membership.mem s a
    ha : Filter.Eventually (fun b => Iff (Membership.mem s (HMul.hMul a b)) (Membe …
    ⊢ Filter.Eventually (fun x => Membership.mem s x) (MeasureTheory.ae μ)
  -/
  simpa [has] using (quasiMeasurePreserving_mul_left μ a⁻¹).ae ha
  /-
    🎉 no goals
  -/

