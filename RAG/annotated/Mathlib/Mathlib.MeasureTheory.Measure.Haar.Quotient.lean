/-- Measurability of the action of the topological group `G` on the left-coset space `G / Γ`. -/
@[to_additive "Measurability of the action of the additive topological group `G` on the left-coset
  space `G / Γ`."]
instance QuotientGroup.measurableSMul {G : Type*} [Group G] {Γ : Subgroup G} [MeasurableSpace G]
    [TopologicalSpace G] [TopologicalGroup G] [BorelSpace G] [BorelSpace (G ⧸ Γ)] :
    MeasurableSMul G (G ⧸ Γ) where
  measurable_const_smul g := (continuous_const_smul g).measurable
  measurable_smul_const _ := (continuous_id.smul continuous_const).measurable


/-- Given a subgroup `Γ` of a topological group `G` with measure `ν`, and a measure 'μ' on the
  quotient `G ⧸ Γ` satisfying `QuotientMeasureEqMeasurePreimage`, the restriction
  of `ν` to a fundamental domain is measure-preserving with respect to `μ`. -/
@[to_additive]
theorem measurePreserving_quotientGroup_mk_of_QuotientMeasureEqMeasurePreimage
    {𝓕 : Set G} (h𝓕 : IsFundamentalDomain Γ.op 𝓕 ν) (μ : Measure (G ⧸ Γ))
    [QuotientMeasureEqMeasurePreimage ν μ] :
    MeasurePreserving (@QuotientGroup.mk G _ Γ) (ν.restrict 𝓕) μ :=
  h𝓕.measurePreserving_quotient_mk μ


local notation "π" => @QuotientGroup.mk G _ Γ


/-- If `μ` satisfies `QuotientMeasureEqMeasurePreimage` relative to a both left- and right-
  invariant measure `ν` on `G`, then it is a `G` invariant measure on `G ⧸ Γ`. -/
@[to_additive]
lemma MeasureTheory.QuotientMeasureEqMeasurePreimage.smulInvariantMeasure_quotient
    [IsMulLeftInvariant ν] [hasFun : HasFundamentalDomain Γ.op G ν] :
    SMulInvariantMeasure G (G ⧸ Γ) μ where
  measure_preimage_smul g A hA := by
    /-
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul g x) A)) (μ A)
    -/
    have meas_π : Measurable π := continuous_quotient_mk'.measurable
    /-
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul g x) A)) (μ A)
    -/
    obtain ⟨𝓕, h𝓕⟩ := hasFun.ExistsIsFundamentalDomain
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul g x) A)) (μ A)
    -/
    have h𝓕_translate_fundom : IsFundamentalDomain Γ.op (g • 𝓕) ν := h𝓕.smul_of_comm g
    -- TODO: why `rw` fails with both of these rewrites?
    erw [h𝓕.projection_respects_measure_apply (μ := μ)
      (meas_π (measurableSet_preimage (measurable_const_smul g) hA)),
      h𝓕_translate_fundom.projection_respects_measure_apply (μ := μ) hA]
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_translate_fundom : MeasureTheory.IsFundamentalDomain (Subtype fun x => Memb …
      ⊢ Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype f …
    -/
    change ν ((π ⁻¹' _) ∩ _) = ν ((π ⁻¹' _) ∩ _)
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_translate_fundom : MeasureTheory.IsFundamentalDomain (Subtype fun x => Memb …
      ⊢ Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.preimage (fun x => HS …
    -/
    set π_preA := π ⁻¹' A
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_translate_fundom : MeasureTheory.IsFundamentalDomain (Subtype fun x => Memb …
      π_preA : Set G := Set.preimage QuotientGroup.mk A
      ⊢ Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.preimage (fun x => HS …
    -/
    have : π ⁻¹' ((fun x : G ⧸ Γ => g • x) ⁻¹' A) = (g * ·) ⁻¹' π_preA := by ext1; simp [π_preA]
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_translate_fundom : MeasureTheory.IsFundamentalDomain (Subtype fun x => Memb …
      π_preA : Set G := Set.preimage QuotientGroup.mk A
      this : Eq (Set.preimage QuotientGroup.mk (Set.preimage (fun x => HSMul.hSMul g …
      ⊢ Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.preimage (fun x => HS …
    -/
    rw [this]
    have : ν ((g * ·) ⁻¹' π_preA ∩ 𝓕) = ν (π_preA ∩ (g⁻¹ * ·) ⁻¹' 𝓕) := by
      trans ν ((g * ·) ⁻¹' (π_preA ∩ (g⁻¹ * ·) ⁻¹' 𝓕))
      · rw [preimage_inter]
        congr 2
        simp [Set.preimage]
      rw [measure_preimage_mul]
    /-
      case intro
      G : Type u_1
      inst✝⁹ : Group G
      inst✝⁸ : MeasurableSpace G
      ν : MeasureTheory.Measure G
      Γ : Subgroup G
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁷ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      inst✝³ : PolishSpace G
      inst✝² : T2Space (HasQuotient.Quotient G Γ)
      inst✝¹ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      g : G
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      meas_π : Measurable QuotientGroup.mk
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_translate_fundom : MeasureTheory.IsFundamentalDomain (Subtype fun x => Memb …
      π_preA : Set G := Set.preimage QuotientGroup.mk A
      this✝ : Eq (Set.preimage QuotientGroup.mk (Set.preimage (fun x => HSMul.hSMul  …
      this : Eq (ν (Inter.inter (Set.preimage (fun x => HMul.hMul g x) π_preA) 𝓕)) ( …
      ⊢ Eq (ν (Inter.inter (Set.preimage (fun x => HMul.hMul g x) π_preA) 𝓕)) (ν (In …
    -/
    rw [this, ← preimage_smul_inv]; rfl
                                    /-
                                      🎉 no goals
                                    -/


/-- If `μ` on `G ⧸ Γ` satisfies `QuotientMeasureEqMeasurePreimage` relative to a both left- and
  right-invariant measure on `G` and `Γ` is a normal subgroup, then `μ` is a left-invariant
  measure. -/
@[to_additive "If `μ` on `G ⧸ Γ` satisfies `AddQuotientMeasureEqMeasurePreimage` relative to a both
  left- and right-invariant measure on `G` and `Γ` is a normal subgroup, then `μ` is a
  left-invariant measure."]
lemma MeasureTheory.QuotientMeasureEqMeasurePreimage.mulInvariantMeasure_quotient
    [hasFun : HasFundamentalDomain Γ.op G ν] [QuotientMeasureEqMeasurePreimage ν μ] :
    μ.IsMulLeftInvariant where
  map_mul_left_eq_self x := by
    /-
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : PolishSpace G
      Γ : Subgroup G
      inst✝⁴ : Γ.Normal
      inst✝³ : T2Space (HasQuotient.Quotient G Γ)
      inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝¹ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      x : HasQuotient.Quotient G Γ
      ⊢ Eq (MeasureTheory.Measure.map (fun x_1 => HMul.hMul x x_1) μ) μ
    -/
    ext A hA
    /-
      case h
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : PolishSpace G
      Γ : Subgroup G
      inst✝⁴ : Γ.Normal
      inst✝³ : T2Space (HasQuotient.Quotient G Γ)
      inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝¹ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      x : HasQuotient.Quotient G Γ
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      ⊢ Eq ((MeasureTheory.Measure.map (fun x_1 => HMul.hMul x x_1) μ) A) (μ A)
    -/
    obtain ⟨x₁, h⟩ := @Quotient.exists_rep _ (QuotientGroup.leftRel Γ) x
    /-
      case h.intro
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : PolishSpace G
      Γ : Subgroup G
      inst✝⁴ : Γ.Normal
      inst✝³ : T2Space (HasQuotient.Quotient G Γ)
      inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝¹ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      x : HasQuotient.Quotient G Γ
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      x₁ : G
      h : Eq (Quotient.mk (QuotientGroup.leftRel Γ) x₁) x
      ⊢ Eq ((MeasureTheory.Measure.map (fun x_1 => HMul.hMul x x_1) μ) A) (μ A)
    -/
    convert measure_preimage_smul μ x₁ A using 1
      /-
        case h.e'_2
        G : Type u_1
        inst✝¹⁰ : Group G
        inst✝⁹ : MeasurableSpace G
        inst✝⁸ : TopologicalSpace G
        inst✝⁷ : TopologicalGroup G
        inst✝⁶ : BorelSpace G
        inst✝⁵ : PolishSpace G
        Γ : Subgroup G
        inst✝⁴ : Γ.Normal
        inst✝³ : T2Space (HasQuotient.Quotient G Γ)
        inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        ν : MeasureTheory.Measure G
        inst✝¹ : ν.IsMulLeftInvariant
        hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
        inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
        x : HasQuotient.Quotient G Γ
        A : Set (HasQuotient.Quotient G Γ)
        hA : MeasurableSet A
        x₁ : G
        h : Eq (Quotient.mk (QuotientGroup.leftRel Γ) x₁) x
        ⊢ Eq ((MeasureTheory.Measure.map (fun x_1 => HMul.hMul x x_1) μ) A) (μ (Set.pr …
      -/
    · rw [← h, Measure.map_apply (measurable_const_mul _) hA]
      /-
        case h.e'_2
        G : Type u_1
        inst✝¹⁰ : Group G
        inst✝⁹ : MeasurableSpace G
        inst✝⁸ : TopologicalSpace G
        inst✝⁷ : TopologicalGroup G
        inst✝⁶ : BorelSpace G
        inst✝⁵ : PolishSpace G
        Γ : Subgroup G
        inst✝⁴ : Γ.Normal
        inst✝³ : T2Space (HasQuotient.Quotient G Γ)
        inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        ν : MeasureTheory.Measure G
        inst✝¹ : ν.IsMulLeftInvariant
        hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
        inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
        x : HasQuotient.Quotient G Γ
        A : Set (HasQuotient.Quotient G Γ)
        hA : MeasurableSet A
        x₁ : G
        h : Eq (Quotient.mk (QuotientGroup.leftRel Γ) x₁) x
        ⊢ Eq (μ (Set.preimage (fun x => HMul.hMul (Quotient.mk (QuotientGroup.leftRel  …
      -/
      simp [← MulAction.Quotient.coe_smul_out, ← Quotient.mk''_eq_mk]
      /-
        🎉 no goals
      -/
    /-
      case h.intro
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      inst✝⁵ : PolishSpace G
      Γ : Subgroup G
      inst✝⁴ : Γ.Normal
      inst✝³ : T2Space (HasQuotient.Quotient G Γ)
      inst✝² : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝¹ : ν.IsMulLeftInvariant
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      x : HasQuotient.Quotient G Γ
      A : Set (HasQuotient.Quotient G Γ)
      hA : MeasurableSet A
      x₁ : G
      h : Eq (Quotient.mk (QuotientGroup.leftRel Γ) x₁) x
      ⊢ MeasureTheory.SMulInvariantMeasure G (HasQuotient.Quotient G Γ) μ
    -/
    exact smulInvariantMeasure_quotient ν
    /-
      🎉 no goals
    -/


/-- Assume that a measure `μ` is `IsMulLeftInvariant`, that the action of `Γ` on `G` has a
measurable fundamental domain `s` with positive finite volume, and that there is a single measurable
set `V ⊆ G ⧸ Γ` along which the pullback of `μ` and `ν` agree (so the scaling is right). Then
`μ` satisfies `QuotientMeasureEqMeasurePreimage`. The main tool of the proof is the uniqueness of
left invariant measures, if normalized by a single positive finite-measured set. -/
@[to_additive
"Assume that a measure `μ` is `IsAddLeftInvariant`, that the action of `Γ` on `G` has a
measurable fundamental domain `s` with positive finite volume, and that there is a single measurable
set `V ⊆ G ⧸ Γ` along which the pullback of `μ` and `ν` agree (so the scaling is right). Then
`μ` satisfies `AddQuotientMeasureEqMeasurePreimage`. The main tool of the proof is the uniqueness of
left invariant measures, if normalized by a single positive finite-measured set."]
theorem MeasureTheory.Measure.IsMulLeftInvariant.quotientMeasureEqMeasurePreimage_of_set {s : Set G}
    (fund_dom_s : IsFundamentalDomain Γ.op s ν) {V : Set (G ⧸ Γ)}
    (meas_V : MeasurableSet V) (neZeroV : μ V ≠ 0) (hV : μ V = ν (π ⁻¹' V ∩ s))
    (neTopV : μ V ≠ ⊤) : QuotientMeasureEqMeasurePreimage ν μ := by
  /-
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  apply fund_dom_s.quotientMeasureEqMeasurePreimage
  /-
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    ⊢ Eq μ (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel (Subtype fu …
  -/
  ext U _
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    ⊢ Eq (μ U) ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel (Subty …
  -/
  have meas_π : Measurable (QuotientGroup.mk : G → G ⧸ Γ) := continuous_quotient_mk'.measurable
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    ⊢ Eq (μ U) ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel (Subty …
  -/
  let μ' : Measure (G ⧸ Γ) := (ν.restrict s).map π
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    ⊢ Eq (μ U) ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel (Subty …
  -/
  haveI has_fund : HasFundamentalDomain Γ.op G ν := ⟨⟨s, fund_dom_s⟩⟩
  have i : QuotientMeasureEqMeasurePreimage ν μ' :=
    fund_dom_s.quotientMeasureEqMeasurePreimage_quotientMeasure
  have : μ'.IsMulLeftInvariant :=
    MeasureTheory.QuotientMeasureEqMeasurePreimage.mulInvariantMeasure_quotient ν
  suffices μ = μ' by
    rw [this]
    rfl
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this : μ'.IsMulLeftInvariant
    ⊢ Eq μ μ'
  -/
  have : SigmaFinite μ' := i.sigmaFiniteQuotient
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this✝ : μ'.IsMulLeftInvariant
    this : MeasureTheory.SigmaFinite μ'
    ⊢ Eq μ μ'
  -/
  rw [measure_eq_div_smul μ' μ neZeroV neTopV, hV]
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this✝ : μ'.IsMulLeftInvariant
    this : MeasureTheory.SigmaFinite μ'
    ⊢ Eq μ (HSMul.hSMul (HDiv.hDiv (μ' V) (ν (Inter.inter (Set.preimage QuotientGr …
  -/
  symm
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this✝ : μ'.IsMulLeftInvariant
    this : MeasureTheory.SigmaFinite μ'
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv (μ' V) (ν (Inter.inter (Set.preimage QuotientGrou …
  -/
  suffices (μ' V / ν (QuotientGroup.mk ⁻¹' V ∩ s)) = 1 by rw [this, one_smul]
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this✝ : μ'.IsMulLeftInvariant
    this : MeasureTheory.SigmaFinite μ'
    ⊢ Eq (HDiv.hDiv (μ' V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))) 1
  -/
  rw [Measure.map_apply meas_π meas_V, Measure.restrict_apply]
    /-
      case h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁵ : ν.IsMulLeftInvariant
      inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      s : Set G
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      V : Set (HasQuotient.Quotient G Γ)
      meas_V : MeasurableSet V
      neZeroV : Ne (μ V) 0
      hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
      neTopV : Ne (μ V) Top.top
      U : Set (HasQuotient.Quotient G Γ)
      a✝ : MeasurableSet U
      meas_π : Measurable QuotientGroup.mk
      μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
      has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
      i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
      this✝ : μ'.IsMulLeftInvariant
      this : MeasureTheory.SigmaFinite μ'
      ⊢ Eq (HDiv.hDiv (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s)) (ν (Inte …
    -/
  · convert ENNReal.div_self ..
      /-
        case h.convert_2
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        ν : MeasureTheory.Measure G
        inst✝⁵ : ν.IsMulLeftInvariant
        inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        s : Set G
        fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
        V : Set (HasQuotient.Quotient G Γ)
        meas_V : MeasurableSet V
        neZeroV : Ne (μ V) 0
        hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
        neTopV : Ne (μ V) Top.top
        U : Set (HasQuotient.Quotient G Γ)
        a✝ : MeasurableSet U
        meas_π : Measurable QuotientGroup.mk
        μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
        has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
        i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
        this✝ : μ'.IsMulLeftInvariant
        this : MeasureTheory.SigmaFinite μ'
        ⊢ Ne (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s)) 0
      -/
    · exact trans hV.symm neZeroV
      /-
        🎉 no goals
      -/
      /-
        case h.convert_3
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        ν : MeasureTheory.Measure G
        inst✝⁵ : ν.IsMulLeftInvariant
        inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        s : Set G
        fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
        V : Set (HasQuotient.Quotient G Γ)
        meas_V : MeasurableSet V
        neZeroV : Ne (μ V) 0
        hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
        neTopV : Ne (μ V) Top.top
        U : Set (HasQuotient.Quotient G Γ)
        a✝ : MeasurableSet U
        meas_π : Measurable QuotientGroup.mk
        μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
        has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
        i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
        this✝ : μ'.IsMulLeftInvariant
        this : MeasureTheory.SigmaFinite μ'
        ⊢ Ne (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s)) Top.top
      -/
    · exact trans hV.symm neTopV
      /-
        🎉 no goals
      -/
  /-
    case h
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁵ : ν.IsMulLeftInvariant
    inst✝⁴ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : MeasureTheory.SigmaFinite ν
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    V : Set (HasQuotient.Quotient G Γ)
    meas_V : MeasurableSet V
    neZeroV : Ne (μ V) 0
    hV : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) s))
    neTopV : Ne (μ V) Top.top
    U : Set (HasQuotient.Quotient G Γ)
    a✝ : MeasurableSet U
    meas_π : Measurable QuotientGroup.mk
    μ' : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := MeasureTheory.Measure …
    has_fund : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem …
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    this✝ : μ'.IsMulLeftInvariant
    this : MeasureTheory.SigmaFinite μ'
    ⊢ MeasurableSet (Set.preimage QuotientGroup.mk V)
  -/
  exact measurableSet_quotient.mp meas_V
  /-
    🎉 no goals
  -/


/-- If a measure `μ` is left-invariant and satisfies the right scaling condition, then it
  satisfies `QuotientMeasureEqMeasurePreimage`. -/
@[to_additive "If a measure `μ` is
left-invariant and satisfies the right scaling condition, then it satisfies
`AddQuotientMeasureEqMeasurePreimage`."]
theorem MeasureTheory.leftInvariantIsQuotientMeasureEqMeasurePreimage [IsFiniteMeasure μ]
    [hasFun : HasFundamentalDomain Γ.op G ν]
    (h : covolume Γ.op G ν = μ univ) : QuotientMeasureEqMeasurePreimage ν μ := by
  /-
    G : Type u_1
    inst✝¹⁵ : Group G
    inst✝¹⁴ : MeasurableSpace G
    inst✝¹³ : TopologicalSpace G
    inst✝¹² : TopologicalGroup G
    inst✝¹¹ : BorelSpace G
    inst✝¹⁰ : PolishSpace G
    Γ : Subgroup G
    inst✝⁹ : Γ.Normal
    inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁶ : ν.IsMulLeftInvariant
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : ν.IsMulRightInvariant
    inst✝³ : MeasureTheory.SigmaFinite ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
    h : Eq (MeasureTheory.covolume (Subtype fun x => Membership.mem Γ.op x) G ν) ( …
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  obtain ⟨s, fund_dom_s⟩ := hasFun.ExistsIsFundamentalDomain
  /-
    case intro
    G : Type u_1
    inst✝¹⁵ : Group G
    inst✝¹⁴ : MeasurableSpace G
    inst✝¹³ : TopologicalSpace G
    inst✝¹² : TopologicalGroup G
    inst✝¹¹ : BorelSpace G
    inst✝¹⁰ : PolishSpace G
    Γ : Subgroup G
    inst✝⁹ : Γ.Normal
    inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁶ : ν.IsMulLeftInvariant
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : ν.IsMulRightInvariant
    inst✝³ : MeasureTheory.SigmaFinite ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
    h : Eq (MeasureTheory.covolume (Subtype fun x => Membership.mem Γ.op x) G ν) ( …
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  have finiteCovol : μ univ < ⊤ := measure_lt_top μ univ
  /-
    case intro
    G : Type u_1
    inst✝¹⁵ : Group G
    inst✝¹⁴ : MeasurableSpace G
    inst✝¹³ : TopologicalSpace G
    inst✝¹² : TopologicalGroup G
    inst✝¹¹ : BorelSpace G
    inst✝¹⁰ : PolishSpace G
    Γ : Subgroup G
    inst✝⁹ : Γ.Normal
    inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁶ : ν.IsMulLeftInvariant
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : ν.IsMulRightInvariant
    inst✝³ : MeasureTheory.SigmaFinite ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
    h : Eq (MeasureTheory.covolume (Subtype fun x => Membership.mem Γ.op x) G ν) ( …
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    finiteCovol : LT.lt (μ Set.univ) Top.top
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  rw [fund_dom_s.covolume_eq_volume] at h
  /-
    case intro
    G : Type u_1
    inst✝¹⁵ : Group G
    inst✝¹⁴ : MeasurableSpace G
    inst✝¹³ : TopologicalSpace G
    inst✝¹² : TopologicalGroup G
    inst✝¹¹ : BorelSpace G
    inst✝¹⁰ : PolishSpace G
    Γ : Subgroup G
    inst✝⁹ : Γ.Normal
    inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    ν : MeasureTheory.Measure G
    inst✝⁶ : ν.IsMulLeftInvariant
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : ν.IsMulRightInvariant
    inst✝³ : MeasureTheory.SigmaFinite ν
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
    s : Set G
    h : Eq (ν s) (μ Set.univ)
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    finiteCovol : LT.lt (μ Set.univ) Top.top
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  by_cases meas_s_ne_zero : ν s = 0
    /-
      case pos
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Eq (ν s) 0
      ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    -/
  · convert fund_dom_s.quotientMeasureEqMeasurePreimage_of_zero meas_s_ne_zero
    /-
      case h.e'_7.h
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Eq (ν s) 0
      e_4✝ : Eq (MulAction.instMulAction Γ.op) Subgroup.instMulAction
      ⊢ Eq μ 0
    -/
    rw [← @measure_univ_eq_zero, ← h, meas_s_ne_zero]
    /-
      🎉 no goals
    -/
  apply IsMulLeftInvariant.quotientMeasureEqMeasurePreimage_of_set (fund_dom_s := fund_dom_s)
    (meas_V := MeasurableSet.univ)
    /-
      case neg.neZeroV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Ne (μ Set.univ) 0
    -/
  · rw [← h]
    /-
      case neg.neZeroV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Ne (ν s) 0
    -/
    exact meas_s_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case neg.hV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Eq (μ Set.univ) (ν (Inter.inter (Set.preimage QuotientGroup.mk Set.univ) s))
    -/
  · rw [← h]
    /-
      case neg.hV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Eq (ν s) (ν (Inter.inter (Set.preimage QuotientGroup.mk Set.univ) s))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg.neTopV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Ne (μ Set.univ) Top.top
    -/
  · rw [← h]
    /-
      case neg.neTopV
      G : Type u_1
      inst✝¹⁵ : Group G
      inst✝¹⁴ : MeasurableSpace G
      inst✝¹³ : TopologicalSpace G
      inst✝¹² : TopologicalGroup G
      inst✝¹¹ : BorelSpace G
      inst✝¹⁰ : PolishSpace G
      Γ : Subgroup G
      inst✝⁹ : Γ.Normal
      inst✝⁸ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁷ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      ν : MeasureTheory.Measure G
      inst✝⁶ : ν.IsMulLeftInvariant
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : ν.IsMulRightInvariant
      inst✝³ : MeasureTheory.SigmaFinite ν
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ …
      s : Set G
      h : Eq (ν s) (μ Set.univ)
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      finiteCovol : LT.lt (μ Set.univ) Top.top
      meas_s_ne_zero : Not (Eq (ν s) 0)
      ⊢ Ne (ν s) Top.top
    -/
    convert finiteCovol.ne
    /-
      🎉 no goals
    -/


/-- If a measure `μ` on the quotient `G ⧸ Γ` of a group `G` by a discrete normal subgroup `Γ` having
fundamental domain, satisfies `QuotientMeasureEqMeasurePreimage` relative to a standardized choice
of Haar measure on `G`, and assuming `μ` is finite, then `μ` is itself Haar.
TODO: Is it possible to drop the assumption that `μ` is finite? -/
@[to_additive "If a measure `μ` on the quotient `G ⧸ Γ` of an additive group `G` by a discrete
normal subgroup `Γ` having fundamental domain, satisfies `AddQuotientMeasureEqMeasurePreimage`
relative to a standardized choice of Haar measure on `G`, and assuming `μ` is finite, then `μ` is
itself Haar."]
theorem MeasureTheory.QuotientMeasureEqMeasurePreimage.haarMeasure_quotient [LocallyCompactSpace G]
    [QuotientMeasureEqMeasurePreimage ν μ] [i : HasFundamentalDomain Γ.op G ν]
    [IsFiniteMeasure μ] : IsHaarMeasure μ := by
  /-
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    i : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.IsHaarMeasure
  -/
  obtain ⟨K⟩ := PositiveCompacts.nonempty' (α := G)
  let K' : PositiveCompacts (G ⧸ Γ) :=
    K.map π QuotientGroup.continuous_mk QuotientGroup.isOpenMap_coe
  haveI : IsMulLeftInvariant μ :=
    MeasureTheory.QuotientMeasureEqMeasurePreimage.mulInvariantMeasure_quotient ν
  /-
    case intro
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    i : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    K : TopologicalSpace.PositiveCompacts G
    K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
    this : μ.IsMulLeftInvariant
    ⊢ μ.IsHaarMeasure
  -/
  rw [haarMeasure_unique μ K']
  have finiteCovol : covolume Γ.op G ν ≠ ⊤ :=
    ne_top_of_lt <| QuotientMeasureEqMeasurePreimage.covolume_ne_top μ (ν := ν)
  /-
    case intro
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    i : MeasureTheory.HasFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    K : TopologicalSpace.PositiveCompacts G
    K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
    this : μ.IsMulLeftInvariant
    finiteCovol : Ne (MeasureTheory.covolume (Subtype fun x => Membership.mem Γ.op …
    ⊢ (HSMul.hSMul (μ ↑K') (MeasureTheory.Measure.haarMeasure K')).IsHaarMeasure
  -/
  obtain ⟨s, fund_dom_s⟩ := i
  /-
    case intro.mk.intro
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    K : TopologicalSpace.PositiveCompacts G
    K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
    this : μ.IsMulLeftInvariant
    finiteCovol : Ne (MeasureTheory.covolume (Subtype fun x => Membership.mem Γ.op …
    s : Set G
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    ⊢ (HSMul.hSMul (μ ↑K') (MeasureTheory.Measure.haarMeasure K')).IsHaarMeasure
  -/
  rw [fund_dom_s.covolume_eq_volume] at finiteCovol
  -- TODO: why `rw` fails?
  /-
    case intro.mk.intro
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    K : TopologicalSpace.PositiveCompacts G
    K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
    this : μ.IsMulLeftInvariant
    s : Set G
    finiteCovol : Ne (ν s) Top.top
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    ⊢ (HSMul.hSMul (μ ↑K') (MeasureTheory.Measure.haarMeasure K')).IsHaarMeasure
  -/
  erw [fund_dom_s.projection_respects_measure_apply μ K'.isCompact.measurableSet]
  /-
    case intro.mk.intro
    G : Type u_1
    inst✝¹⁴ : Group G
    inst✝¹³ : MeasurableSpace G
    inst✝¹² : TopologicalSpace G
    inst✝¹¹ : TopologicalGroup G
    inst✝¹⁰ : BorelSpace G
    inst✝⁹ : PolishSpace G
    Γ : Subgroup G
    inst✝⁸ : Γ.Normal
    inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝⁴ : ν.IsHaarMeasure
    inst✝³ : ν.IsMulRightInvariant
    inst✝² : LocallyCompactSpace G
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    K : TopologicalSpace.PositiveCompacts G
    K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
    this : μ.IsMulLeftInvariant
    s : Set G
    finiteCovol : Ne (ν s) Top.top
    fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
    ⊢ (HSMul.hSMul (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel  …
  -/
  apply IsHaarMeasure.smul
    /-
      case intro.mk.intro.cpos
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ Ne (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype f …
    -/
  · intro h
    /-
      case intro.mk.intro.cpos
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      ⊢ False
    -/
    haveI i' : IsOpenPosMeasure (ν : Measure G) := inferInstance
    /-
      case intro.mk.intro.cpos
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ False
    -/
    apply IsOpenPosMeasure.open_pos (interior K) (μ := ν) (self := i')
      /-
        case intro.mk.intro.cpos.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : LocallyCompactSpace G
        inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        K : TopologicalSpace.PositiveCompacts G
        K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
        this : μ.IsMulLeftInvariant
        s : Set G
        finiteCovol : Ne (ν s) Top.top
        fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
        h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
        i' : ν.IsOpenPosMeasure
        ⊢ IsOpen (interior ↑K)
      -/
    · exact isOpen_interior
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.intro.cpos.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : LocallyCompactSpace G
        inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        K : TopologicalSpace.PositiveCompacts G
        K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
        this : μ.IsMulLeftInvariant
        s : Set G
        finiteCovol : Ne (ν s) Top.top
        fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
        h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
        i' : ν.IsOpenPosMeasure
        ⊢ (interior ↑K).Nonempty
      -/
    · exact K.interior_nonempty
      /-
        🎉 no goals
      -/
    rw [← le_zero_iff,
      ← fund_dom_s.measure_zero_of_invariant _ (fun g ↦ QuotientGroup.sound _ _ g) h]
    /-
      case intro.mk.intro.cpos.a
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ LE.le (ν (interior ↑K)) (ν (Set.preimage ⇑(QuotientGroup.mk' Γ) ↑K'))
    -/
    apply measure_mono
    /-
      case intro.mk.intro.cpos.a.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ HasSubset.Subset (interior ↑K) (Set.preimage ⇑(QuotientGroup.mk' Γ) ↑K')
    -/
    refine interior_subset.trans ?_
    /-
      case intro.mk.intro.cpos.a.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ HasSubset.Subset (↑K) (Set.preimage ⇑(QuotientGroup.mk' Γ) ↑K')
    -/
    rw [QuotientGroup.coe_mk']
    /-
      case intro.mk.intro.cpos.a.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ HasSubset.Subset (↑K) (Set.preimage QuotientGroup.mk ↑K')
    -/
    show (K : Set G) ⊆ π ⁻¹' (π '' K)
    /-
      case intro.mk.intro.cpos.a.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      h : Eq (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype …
      i' : ν.IsOpenPosMeasure
      ⊢ HasSubset.Subset (↑K) (Set.preimage QuotientGroup.mk (Set.image QuotientGrou …
    -/
    exact subset_preimage_image π K
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.ctop
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ Ne (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel (Subtype f …
    -/
  · show ν (π ⁻¹' (π '' K) ∩ s) ≠ ⊤
    /-
      case intro.mk.intro.ctop
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ Ne (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.image QuotientGroup.m …
    -/
    apply ne_of_lt
    /-
      case intro.mk.intro.ctop.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ LT.lt (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.image QuotientGrou …
    -/
    refine lt_of_le_of_lt ?_ finiteCovol.lt_top
    /-
      case intro.mk.intro.ctop.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ LE.le (ν (Inter.inter (Set.preimage QuotientGroup.mk (Set.image QuotientGrou …
    -/
    apply measure_mono
    /-
      case intro.mk.intro.ctop.h.h
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : LocallyCompactSpace G
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      K : TopologicalSpace.PositiveCompacts G
      K' : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ) := Topologic …
      this : μ.IsMulLeftInvariant
      s : Set G
      finiteCovol : Ne (ν s) Top.top
      fund_dom_s : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.me …
      ⊢ HasSubset.Subset (Inter.inter (Set.preimage QuotientGroup.mk (Set.image Quot …
    -/
    exact inter_subset_right
    /-
      🎉 no goals
    -/


/-- Given a normal subgroup `Γ` of a topological group `G` with Haar measure `μ`, which is also
  right-invariant, and a finite volume fundamental domain `𝓕`, the quotient map to `G ⧸ Γ`,
  properly normalized, satisfies `QuotientMeasureEqMeasurePreimage`. -/
@[to_additive "Given a normal
subgroup `Γ` of an additive topological group `G` with Haar measure `μ`, which is also
right-invariant, and a finite volume fundamental domain `𝓕`, the quotient map to `G ⧸ Γ`,
properly normalized, satisfies `AddQuotientMeasureEqMeasurePreimage`."]
theorem IsFundamentalDomain.QuotientMeasureEqMeasurePreimage_HaarMeasure {𝓕 : Set G}
    (h𝓕 : IsFundamentalDomain Γ.op 𝓕 ν) [IsMulLeftInvariant μ] [SigmaFinite μ]
    {V : Set (G ⧸ Γ)} (hV : (interior V).Nonempty) (meas_V : MeasurableSet V)
    (hμK : μ V = ν ((π ⁻¹' V) ∩ 𝓕)) (neTopV : μ V ≠ ⊤) :
    QuotientMeasureEqMeasurePreimage ν μ := by
  apply IsMulLeftInvariant.quotientMeasureEqMeasurePreimage_of_set (fund_dom_s := h𝓕)
    (meas_V := meas_V)
    /-
      case neZeroV
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      V : Set (HasQuotient.Quotient G Γ)
      hV : (interior V).Nonempty
      meas_V : MeasurableSet V
      hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
      neTopV : Ne (μ V) Top.top
      ⊢ Ne (μ V) 0
    -/
  · rw [hμK]
    /-
      case neZeroV
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      V : Set (HasQuotient.Quotient G Γ)
      hV : (interior V).Nonempty
      meas_V : MeasurableSet V
      hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
      neTopV : Ne (μ V) Top.top
      ⊢ Ne (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
    -/
    intro c_eq_zero
    /-
      case neZeroV
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      V : Set (HasQuotient.Quotient G Γ)
      hV : (interior V).Nonempty
      meas_V : MeasurableSet V
      hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
      neTopV : Ne (μ V) Top.top
      c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
      ⊢ False
    -/
    apply IsOpenPosMeasure.open_pos (interior (π ⁻¹' V)) (μ := ν)
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ IsOpen (interior (Set.preimage QuotientGroup.mk V))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ (interior (Set.preimage QuotientGroup.mk V)).Nonempty
      -/
    · apply Set.Nonempty.mono (preimage_interior_subset_interior_preimage continuous_coinduced_rng)
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ (Set.preimage QuotientGroup.mk (interior V)).Nonempty
      -/
      apply hV.preimage'
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ HasSubset.Subset (interior V) (Set.range QuotientGroup.mk)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ Eq (ν (interior (Set.preimage QuotientGroup.mk V))) 0
      -/
    · apply measure_mono_null (h := interior_subset)
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ Eq (ν (Set.preimage QuotientGroup.mk V)) 0
      -/
      apply h𝓕.measure_zero_of_invariant (ht := fun g ↦ QuotientGroup.sound _ _ g)
      /-
        case neZeroV.a
        G : Type u_1
        inst✝¹⁴ : Group G
        inst✝¹³ : MeasurableSpace G
        inst✝¹² : TopologicalSpace G
        inst✝¹¹ : TopologicalGroup G
        inst✝¹⁰ : BorelSpace G
        inst✝⁹ : PolishSpace G
        Γ : Subgroup G
        inst✝⁸ : Γ.Normal
        inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
        inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
        μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
        inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
        ν : MeasureTheory.Measure G
        inst✝⁴ : ν.IsHaarMeasure
        inst✝³ : ν.IsMulRightInvariant
        inst✝² : MeasureTheory.SigmaFinite ν
        𝓕 : Set G
        h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
        inst✝¹ : μ.IsMulLeftInvariant
        inst✝ : MeasureTheory.SigmaFinite μ
        V : Set (HasQuotient.Quotient G Γ)
        hV : (interior V).Nonempty
        meas_V : MeasurableSet V
        hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
        neTopV : Ne (μ V) Top.top
        c_eq_zero : Eq (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕)) 0
        ⊢ Eq (ν (Inter.inter (Set.preimage (⇑(QuotientGroup.mk' Γ)) V) 𝓕)) 0
      -/
      exact c_eq_zero
      /-
        🎉 no goals
      -/
    /-
      case hV
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      V : Set (HasQuotient.Quotient G Γ)
      hV : (interior V).Nonempty
      meas_V : MeasurableSet V
      hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
      neTopV : Ne (μ V) Top.top
      ⊢ Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
    -/
  · exact hμK
    /-
      🎉 no goals
    -/
    /-
      case neTopV
      G : Type u_1
      inst✝¹⁴ : Group G
      inst✝¹³ : MeasurableSpace G
      inst✝¹² : TopologicalSpace G
      inst✝¹¹ : TopologicalGroup G
      inst✝¹⁰ : BorelSpace G
      inst✝⁹ : PolishSpace G
      Γ : Subgroup G
      inst✝⁸ : Γ.Normal
      inst✝⁷ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁶ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ)
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝⁴ : ν.IsHaarMeasure
      inst✝³ : ν.IsMulRightInvariant
      inst✝² : MeasureTheory.SigmaFinite ν
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝¹ : μ.IsMulLeftInvariant
      inst✝ : MeasureTheory.SigmaFinite μ
      V : Set (HasQuotient.Quotient G Γ)
      hV : (interior V).Nonempty
      meas_V : MeasurableSet V
      hμK : Eq (μ V) (ν (Inter.inter (Set.preimage QuotientGroup.mk V) 𝓕))
      neTopV : Ne (μ V) Top.top
      ⊢ Ne (μ V) Top.top
    -/
  · exact neTopV
    /-
      🎉 no goals
    -/


/-- Given a normal subgroup `Γ` of a topological group `G` with Haar measure `μ`, which is also
  right-invariant, and a finite volume fundamental domain `𝓕`, the quotient map to `G ⧸ Γ`,
  properly normalized, satisfies `QuotientMeasureEqMeasurePreimage`. -/
@[to_additive "Given a
normal subgroup `Γ` of an additive topological group `G` with Haar measure `μ`, which is also
right-invariant, and a finite volume fundamental domain `𝓕`, the quotient map to `G ⧸ Γ`,
properly normalized, satisfies `AddQuotientMeasureEqMeasurePreimage`."]
theorem IsFundamentalDomain.QuotientMeasureEqMeasurePreimage_smulHaarMeasure {𝓕 : Set G}
    (h𝓕 : IsFundamentalDomain Γ.op 𝓕 ν) (h𝓕_finite : ν 𝓕 ≠ ⊤) :
    QuotientMeasureEqMeasurePreimage ν
      ((ν ((π ⁻¹' (K : Set (G ⧸ Γ))) ∩ 𝓕)) • haarMeasure K) := by
  /-
    G : Type u_1
    inst✝¹² : Group G
    inst✝¹¹ : MeasurableSpace G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : TopologicalGroup G
    inst✝⁸ : BorelSpace G
    inst✝⁷ : PolishSpace G
    Γ : Subgroup G
    inst✝⁶ : Γ.Normal
    inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : MeasureTheory.SigmaFinite ν
    K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    h𝓕_finite : Ne (ν 𝓕) Top.top
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν (HSMul.hSMul (ν (Inter.inte …
  -/
  set c := ν ((π ⁻¹' (K : Set (G ⧸ Γ))) ∩ 𝓕)
  have c_ne_top : c ≠ ∞ := by
    contrapose! h𝓕_finite
    have : c ≤ ν 𝓕 := measure_mono (Set.inter_subset_right)
    rw [h𝓕_finite] at this
    exact top_unique this
  /-
    G : Type u_1
    inst✝¹² : Group G
    inst✝¹¹ : MeasurableSpace G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : TopologicalGroup G
    inst✝⁸ : BorelSpace G
    inst✝⁷ : PolishSpace G
    Γ : Subgroup G
    inst✝⁶ : Γ.Normal
    inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : MeasureTheory.SigmaFinite ν
    K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    h𝓕_finite : Ne (ν 𝓕) Top.top
    c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
    c_ne_top : Ne c Top.top
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν (HSMul.hSMul c (MeasureTheo …
  -/
  set μ := c • haarMeasure K
  /-
    G : Type u_1
    inst✝¹² : Group G
    inst✝¹¹ : MeasurableSpace G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : TopologicalGroup G
    inst✝⁸ : BorelSpace G
    inst✝⁷ : PolishSpace G
    Γ : Subgroup G
    inst✝⁶ : Γ.Normal
    inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
    inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    ν : MeasureTheory.Measure G
    inst✝² : ν.IsHaarMeasure
    inst✝¹ : ν.IsMulRightInvariant
    inst✝ : MeasureTheory.SigmaFinite ν
    K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    h𝓕_finite : Ne (ν 𝓕) Top.top
    c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
    c_ne_top : Ne c Top.top
    μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := HSMul.hSMul c (Measure …
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  have hμK : μ K = c := by simp [μ, haarMeasure_self]
  haveI : SigmaFinite μ := by
    clear_value c
    lift c to NNReal using c_ne_top
    exact SMul.sigmaFinite c
  apply IsFundamentalDomain.QuotientMeasureEqMeasurePreimage_HaarMeasure (h𝓕 := h𝓕)
    (meas_V := K.isCompact.measurableSet) (μ := μ)
    /-
      case hV
      G : Type u_1
      inst✝¹² : Group G
      inst✝¹¹ : MeasurableSpace G
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : TopologicalGroup G
      inst✝⁸ : BorelSpace G
      inst✝⁷ : PolishSpace G
      Γ : Subgroup G
      inst✝⁶ : Γ.Normal
      inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝² : ν.IsHaarMeasure
      inst✝¹ : ν.IsMulRightInvariant
      inst✝ : MeasureTheory.SigmaFinite ν
      K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_finite : Ne (ν 𝓕) Top.top
      c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
      c_ne_top : Ne c Top.top
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := HSMul.hSMul c (Measure …
      hμK : Eq (μ ↑K) c
      this : MeasureTheory.SigmaFinite μ
      ⊢ (interior ↑K).Nonempty
    -/
  · exact K.interior_nonempty
    /-
      🎉 no goals
    -/
    /-
      case hμK
      G : Type u_1
      inst✝¹² : Group G
      inst✝¹¹ : MeasurableSpace G
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : TopologicalGroup G
      inst✝⁸ : BorelSpace G
      inst✝⁷ : PolishSpace G
      Γ : Subgroup G
      inst✝⁶ : Γ.Normal
      inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝² : ν.IsHaarMeasure
      inst✝¹ : ν.IsMulRightInvariant
      inst✝ : MeasureTheory.SigmaFinite ν
      K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_finite : Ne (ν 𝓕) Top.top
      c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
      c_ne_top : Ne c Top.top
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := HSMul.hSMul c (Measure …
      hμK : Eq (μ ↑K) c
      this : MeasureTheory.SigmaFinite μ
      ⊢ Eq (μ ↑K) (ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕))
    -/
  · exact hμK
    /-
      🎉 no goals
    -/
    /-
      case neTopV
      G : Type u_1
      inst✝¹² : Group G
      inst✝¹¹ : MeasurableSpace G
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : TopologicalGroup G
      inst✝⁸ : BorelSpace G
      inst✝⁷ : PolishSpace G
      Γ : Subgroup G
      inst✝⁶ : Γ.Normal
      inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝² : ν.IsHaarMeasure
      inst✝¹ : ν.IsMulRightInvariant
      inst✝ : MeasureTheory.SigmaFinite ν
      K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_finite : Ne (ν 𝓕) Top.top
      c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
      c_ne_top : Ne c Top.top
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := HSMul.hSMul c (Measure …
      hμK : Eq (μ ↑K) c
      this : MeasureTheory.SigmaFinite μ
      ⊢ Ne (μ ↑K) Top.top
    -/
  · rw [hμK]
    /-
      case neTopV
      G : Type u_1
      inst✝¹² : Group G
      inst✝¹¹ : MeasurableSpace G
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : TopologicalGroup G
      inst✝⁸ : BorelSpace G
      inst✝⁷ : PolishSpace G
      Γ : Subgroup G
      inst✝⁶ : Γ.Normal
      inst✝⁵ : T2Space (HasQuotient.Quotient G Γ)
      inst✝⁴ : SecondCountableTopology (HasQuotient.Quotient G Γ)
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      ν : MeasureTheory.Measure G
      inst✝² : ν.IsHaarMeasure
      inst✝¹ : ν.IsMulRightInvariant
      inst✝ : MeasureTheory.SigmaFinite ν
      K : TopologicalSpace.PositiveCompacts (HasQuotient.Quotient G Γ)
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      h𝓕_finite : Ne (ν 𝓕) Top.top
      c : ENNReal := ν (Inter.inter (Set.preimage QuotientGroup.mk ↑K) 𝓕)
      c_ne_top : Ne c Top.top
      μ : MeasureTheory.Measure (HasQuotient.Quotient G Γ) := HSMul.hSMul c (Measure …
      hμK : Eq (μ ↑K) c
      this : MeasureTheory.SigmaFinite μ
      ⊢ Ne c Top.top
    -/
    exact c_ne_top
    /-
      🎉 no goals
    -/


local notation "μ_𝓕" => Measure.map (@QuotientGroup.mk G _ Γ) (μ.restrict 𝓕)


/-- The `essSup` of a function `g` on the quotient space `G ⧸ Γ` with respect to the pushforward
  of the restriction, `μ_𝓕`, of a right-invariant measure `μ` to a fundamental domain `𝓕`, is the
  same as the `essSup` of `g`'s lift to the universal cover `G` with respect to `μ`. -/
@[to_additive "The `essSup` of a function `g` on the additive quotient space `G ⧸ Γ` with respect
  to the pushforward of the restriction, `μ_𝓕`, of a right-invariant measure `μ` to a fundamental
  domain `𝓕`, is the same as the `essSup` of `g`'s lift to the universal cover `G` with respect
  to `μ`."]
lemma essSup_comp_quotientGroup_mk [μ.IsMulRightInvariant] {g : G ⧸ Γ → ℝ≥0∞}
    (g_ae_measurable : AEMeasurable g μ_𝓕) : essSup g μ_𝓕 = essSup (fun (x : G) ↦ g x) μ := by
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    ⊢ Eq (essSup g (MeasureTheory.Measure.map QuotientGroup.mk (μ.restrict 𝓕))) (e …
  -/
  have hπ : Measurable (QuotientGroup.mk : G → G ⧸ Γ) := continuous_quotient_mk'.measurable
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    ⊢ Eq (essSup g (MeasureTheory.Measure.map QuotientGroup.mk (μ.restrict 𝓕))) (e …
  -/
  rw [essSup_map_measure g_ae_measurable hπ.aemeasurable]
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    ⊢ Eq (essSup (Function.comp g QuotientGroup.mk) (μ.restrict 𝓕)) (essSup (fun x …
  -/
  refine h𝓕.essSup_measure_restrict ?_
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    ⊢ ∀ (γ : Subtype fun x => Membership.mem Γ.op x) (x : G), Eq (Function.comp g  …
  -/
  intro ⟨γ, hγ⟩ x
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    γ : MulOpposite G
    hγ : Membership.mem Γ.op γ
    x : G
    ⊢ Eq (Function.comp g QuotientGroup.mk (HSMul.hSMul ⟨γ, hγ⟩ x)) (Function.comp …
  -/
  dsimp
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    γ : MulOpposite G
    hγ : Membership.mem Γ.op γ
    x : G
    ⊢ Eq (g ↑(HMul.hMul x (MulOpposite.unop γ))) (g ↑x)
  -/
  congr 1
  /-
    case e_a
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    g : HasQuotient.Quotient G Γ → ENNReal
    g_ae_measurable : AEMeasurable g (MeasureTheory.Measure.map QuotientGroup.mk ( …
    hπ : Measurable QuotientGroup.mk
    γ : MulOpposite G
    hγ : Membership.mem Γ.op γ
    x : G
    ⊢ Eq ↑(HMul.hMul x (MulOpposite.unop γ)) ↑x
  -/
  exact QuotientGroup.mk_mul_of_mem x hγ
  /-
    🎉 no goals
  -/


/-- Given a quotient space `G ⧸ Γ` where `Γ` is `Countable`, and the restriction,
  `μ_𝓕`, of a right-invariant measure `μ` on `G` to a fundamental domain `𝓕`, a set
  in the quotient which has `μ_𝓕`-measure zero, also has measure zero under the
  folding of `μ` under the quotient. Note that, if `Γ` is infinite, then the folded map
  will take the value `∞` on any open set in the quotient! -/
@[to_additive "Given an additive quotient space `G ⧸ Γ` where `Γ` is `Countable`, and the
  restriction, `μ_𝓕`, of a right-invariant measure `μ` on `G` to a fundamental domain `𝓕`, a set
  in the quotient which has `μ_𝓕`-measure zero, also has measure zero under the
  folding of `μ` under the quotient. Note that, if `Γ` is infinite, then the folded map
  will take the value `∞` on any open set in the quotient!"]
lemma _root_.MeasureTheory.IsFundamentalDomain.absolutelyContinuous_map
    [μ.IsMulRightInvariant] :
    map (QuotientGroup.mk : G → G ⧸ Γ) μ ≪ map (QuotientGroup.mk : G → G ⧸ Γ) (μ.restrict 𝓕) := by
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    ⊢ (MeasureTheory.Measure.map QuotientGroup.mk μ).AbsolutelyContinuous (Measure …
  -/
  set π : G → G ⧸ Γ := QuotientGroup.mk
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    ⊢ (MeasureTheory.Measure.map π μ).AbsolutelyContinuous (MeasureTheory.Measure. …
  -/
  have meas_π : Measurable π := continuous_quotient_mk'.measurable
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    ⊢ (MeasureTheory.Measure.map π μ).AbsolutelyContinuous (MeasureTheory.Measure. …
  -/
  apply AbsolutelyContinuous.mk
  /-
    case h
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    ⊢ ∀ ⦃s : Set (HasQuotient.Quotient G Γ)⦄, MeasurableSet s → Eq ((MeasureTheory …
  -/
  intro s s_meas hs
  /-
    case h
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    s : Set (HasQuotient.Quotient G Γ)
    s_meas : MeasurableSet s
    hs : Eq ((MeasureTheory.Measure.map π (μ.restrict 𝓕)) s) 0
    ⊢ Eq ((MeasureTheory.Measure.map π μ) s) 0
  -/
  rw [map_apply meas_π s_meas] at hs ⊢
  /-
    case h
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    s : Set (HasQuotient.Quotient G Γ)
    s_meas : MeasurableSet s
    hs : Eq ((μ.restrict 𝓕) (Set.preimage π s)) 0
    ⊢ Eq (μ (Set.preimage π s)) 0
  -/
  rw [Measure.restrict_apply] at hs
    /-
      case h
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      ⊢ Eq (μ (Set.preimage π s)) 0
    -/
  · apply h𝓕.measure_zero_of_invariant _ _ hs
    /-
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      ⊢ ∀ (g : Subtype fun x => Membership.mem Γ.op x), Eq (HSMul.hSMul g (Set.preim …
    -/
    intro γ
    /-
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      γ : Subtype fun x => Membership.mem Γ.op x
      ⊢ Eq (HSMul.hSMul γ (Set.preimage π s)) (Set.preimage π s)
    -/
    ext g
    /-
      case h
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      γ : Subtype fun x => Membership.mem Γ.op x
      g : G
      ⊢ Iff (Membership.mem (HSMul.hSMul γ (Set.preimage π s)) g) (Membership.mem (S …
    -/
    rw [Set.mem_smul_set_iff_inv_smul_mem, mem_preimage, mem_preimage]
    /-
      case h
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      γ : Subtype fun x => Membership.mem Γ.op x
      g : G
      ⊢ Iff (Membership.mem s (π (HSMul.hSMul (Inv.inv γ) g))) (Membership.mem s (π  …
    -/
    congr! 1
    /-
      case h.a.h.e'_5
      G : Type u_1
      inst✝⁸ : Group G
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : TopologicalSpace G
      inst✝⁵ : TopologicalGroup G
      inst✝⁴ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
      inst✝ : μ.IsMulRightInvariant
      π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
      meas_π : Measurable π
      s : Set (HasQuotient.Quotient G Γ)
      s_meas : MeasurableSet s
      hs : Eq (μ (Inter.inter (Set.preimage π s) 𝓕)) 0
      γ : Subtype fun x => Membership.mem Γ.op x
      g : G
      ⊢ Eq (π (HSMul.hSMul (Inv.inv γ) g)) (π g)
    -/
    convert QuotientGroup.mk_mul_of_mem g (γ⁻¹).2 using 1
    /-
      🎉 no goals
    -/
  /-
    case h
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalGroup G
    inst✝⁴ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝³ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝² : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝¹ : BorelSpace (HasQuotient.Quotient G Γ)
    inst✝ : μ.IsMulRightInvariant
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    s : Set (HasQuotient.Quotient G Γ)
    s_meas : MeasurableSet s
    hs : Eq ((μ.restrict 𝓕) (Set.preimage π s)) 0
    ⊢ MeasurableSet (Set.preimage π s)
  -/
  exact MeasurableSet.preimage s_meas meas_π
  /-
    🎉 no goals
  -/


/-- This is a simple version of the **Unfolding Trick**: Given a subgroup `Γ` of a group `G`, the
  integral of a function `f` on `G` with respect to a right-invariant measure `μ` is equal to the
  integral over the quotient `G ⧸ Γ` of the automorphization of `f`. -/
@[to_additive "This is a simple version of the **Unfolding Trick**: Given a subgroup `Γ` of an
  additive group `G`, the integral of a function `f` on `G` with respect to a right-invariant
  measure `μ` is equal to the integral over the quotient `G ⧸ Γ` of the automorphization of `f`."]
lemma QuotientGroup.integral_eq_integral_automorphize {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [μ.IsMulRightInvariant] {f : G → E}
    (hf₁ : Integrable f μ) (hf₂ : AEStronglyMeasurable (automorphize f) μ_𝓕) :
    ∫ x : G, f x ∂μ = ∫ x : G ⧸ Γ, automorphize f x ∂μ_𝓕 := by
  calc ∫ x : G, f x ∂μ = ∑' γ : Γ.op, ∫ x in 𝓕, f (γ • x) ∂μ :=
    h𝓕.integral_eq_tsum'' f hf₁
    _ = ∫ x in 𝓕, ∑' γ : Γ.op, f (γ • x) ∂μ := ?_
    _ = ∫ x : G ⧸ Γ, automorphize f x ∂μ_𝓕 :=
      (integral_map continuous_quotient_mk'.aemeasurable hf₂).symm
  /-
    G : Type u_1
    inst✝¹⁰ : Group G
    inst✝⁹ : MeasurableSpace G
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : μ.IsMulRightInvariant
    f : G → E
    hf₁ : MeasureTheory.Integrable f μ
    hf₂ : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphize f) (Measu …
    ⊢ Eq (tsum fun γ => MeasureTheory.integral (μ.restrict 𝓕) fun x => f (HSMul.hS …
  -/
  rw [integral_tsum]
  · exact fun i ↦ (hf₁.1.comp_quasiMeasurePreserving
      (measurePreserving_smul i μ).quasiMeasurePreserving).restrict
    /-
      case hf'
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.IsMulRightInvariant
      f : G → E
      hf₁ : MeasureTheory.Integrable f μ
      hf₂ : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphize f) (Measu …
      ⊢ Ne (tsum fun i => MeasureTheory.lintegral (μ.restrict 𝓕) fun a => ↑(NNNorm.n …
    -/
  · rw [← h𝓕.lintegral_eq_tsum'' (fun x ↦ ‖f x‖₊)]
    /-
      case hf'
      G : Type u_1
      inst✝¹⁰ : Group G
      inst✝⁹ : MeasurableSpace G
      inst✝⁸ : TopologicalSpace G
      inst✝⁷ : TopologicalGroup G
      inst✝⁶ : BorelSpace G
      μ : MeasureTheory.Measure G
      Γ : Subgroup G
      𝓕 : Set G
      h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
      inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
      inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
      inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : μ.IsMulRightInvariant
      f : G → E
      hf₁ : MeasureTheory.Integrable f μ
      hf₂ : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphize f) (Measu …
      ⊢ Ne (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (f x))) Top.top
    -/
    exact ne_of_lt hf₁.2
    /-
      🎉 no goals
    -/


/-- This is the **Unfolding Trick**: Given a subgroup `Γ` of a group `G`, the integral of a
  function `f` on `G` times the lift to `G` of a function `g` on the quotient `G ⧸ Γ` with respect
  to a right-invariant measure `μ` on `G`, is equal to the integral over the quotient of the
  automorphization of `f` times `g`. -/
lemma QuotientGroup.integral_mul_eq_integral_automorphize_mul {K : Type*} [NormedField K]
    [NormedSpace ℝ K] [μ.IsMulRightInvariant] {f : G → K}
    (f_ℒ_1 : Integrable f μ) {g : G ⧸ Γ → K} (hg : AEStronglyMeasurable g μ_𝓕)
    (g_ℒ_infinity : essSup (fun x ↦ ↑‖g x‖₊) μ_𝓕 ≠ ∞)
    (F_ae_measurable : AEStronglyMeasurable (QuotientGroup.automorphize f) μ_𝓕) :
    ∫ x : G, g (x : G ⧸ Γ) * (f x) ∂μ
      = ∫ x : G ⧸ Γ, g x * (QuotientGroup.automorphize f x) ∂μ_𝓕 := by
  /-
    G : Type u_1
    inst✝¹⁰ : Group G
    inst✝⁹ : MeasurableSpace G
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ.IsMulRightInvariant
    f : G → K
    f_ℒ_1 : MeasureTheory.Integrable f μ
    g : HasQuotient.Quotient G Γ → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientG …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphi …
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul (g ↑x) (f x)) (MeasureTheory …
  -/
  let π : G → G ⧸ Γ := QuotientGroup.mk
  /-
    G : Type u_1
    inst✝¹⁰ : Group G
    inst✝⁹ : MeasurableSpace G
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ.IsMulRightInvariant
    f : G → K
    f_ℒ_1 : MeasureTheory.Integrable f μ
    g : HasQuotient.Quotient G Γ → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientG …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphi …
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul (g ↑x) (f x)) (MeasureTheory …
  -/
  have meas_π : Measurable π := continuous_quotient_mk'.measurable
  have H₀ : QuotientGroup.automorphize ((g ∘ π) * f) = g * (QuotientGroup.automorphize f) := by
    exact QuotientGroup.automorphize_smul_left f g
  calc ∫ (x : G), g (π x) * (f x) ∂μ =
        ∫ (x : G ⧸ Γ), QuotientGroup.automorphize ((g ∘ π) * f) x ∂μ_𝓕 := ?_
    _ = ∫ (x : G ⧸ Γ), g x * (QuotientGroup.automorphize f x) ∂μ_𝓕 := by simp [H₀]
  have H₁ : Integrable ((g ∘ π) * f) μ := by
    have : AEStronglyMeasurable (fun (x : G) ↦ g (x : (G ⧸ Γ))) μ :=
      (hg.mono_ac h𝓕.absolutelyContinuous_map).comp_measurable meas_π
    refine Integrable.essSup_smul f_ℒ_1 this ?_
    have hg' : AEStronglyMeasurable (fun x ↦ (‖g x‖₊ : ℝ≥0∞)) μ_𝓕 :=
      (ENNReal.continuous_coe.comp continuous_nnnorm).comp_aestronglyMeasurable hg
    rw [← essSup_comp_quotientGroup_mk h𝓕 hg'.aemeasurable]
    exact g_ℒ_infinity
  have H₂ : AEStronglyMeasurable (QuotientGroup.automorphize ((g ∘ π) * f)) μ_𝓕 := by
    simp_rw [H₀]
    exact hg.mul F_ae_measurable
  /-
    G : Type u_1
    inst✝¹⁰ : Group G
    inst✝⁹ : MeasurableSpace G
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : TopologicalGroup G
    inst✝⁶ : BorelSpace G
    μ : MeasureTheory.Measure G
    Γ : Subgroup G
    𝓕 : Set G
    h𝓕 : MeasureTheory.IsFundamentalDomain (Subtype fun x => Membership.mem Γ.op x …
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G Γ)
    inst✝³ : BorelSpace (HasQuotient.Quotient G Γ)
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ.IsMulRightInvariant
    f : G → K
    f_ℒ_1 : MeasureTheory.Integrable f μ
    g : HasQuotient.Quotient G Γ → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientG …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphi …
    π : G → HasQuotient.Quotient G Γ := QuotientGroup.mk
    meas_π : Measurable π
    H₀ : Eq (QuotientGroup.automorphize (HMul.hMul (Function.comp g π) f)) (HMul.h …
    H₁ : MeasureTheory.Integrable (HMul.hMul (Function.comp g π) f) μ
    H₂ : MeasureTheory.AEStronglyMeasurable (QuotientGroup.automorphize (HMul.hMul …
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul (g (π x)) (f x)) (MeasureThe …
  -/
  apply QuotientGroup.integral_eq_integral_automorphize h𝓕 H₁ H₂
  /-
    🎉 no goals
  -/


local notation "μ_𝓕" => Measure.map (@QuotientAddGroup.mk G' _ Γ') (μ'.restrict 𝓕')


/-- This is the **Unfolding Trick**: Given an additive subgroup `Γ'` of an additive group `G'`, the
  integral of a function `f` on `G'` times the lift to `G'` of a function `g` on the quotient
  `G' ⧸ Γ'` with respect to a right-invariant measure `μ` on `G'`, is equal to the integral over
  the quotient of the automorphization of `f` times `g`. -/
lemma QuotientAddGroup.integral_mul_eq_integral_automorphize_mul {K : Type*} [NormedField K]
    [NormedSpace ℝ K] [μ'.IsAddRightInvariant] {f : G' → K}
    (f_ℒ_1 : Integrable f μ') {g : G' ⧸ Γ' → K} (hg : AEStronglyMeasurable g μ_𝓕)
    (g_ℒ_infinity : essSup (fun x ↦ (‖g x‖₊ : ℝ≥0∞)) μ_𝓕 ≠ ∞)
    (F_ae_measurable : AEStronglyMeasurable (QuotientAddGroup.automorphize f) μ_𝓕)
    (h𝓕 : IsAddFundamentalDomain Γ'.op 𝓕' μ') :
    ∫ x : G', g (x : G' ⧸ Γ') * (f x) ∂μ'
      = ∫ x : G' ⧸ Γ', g x * (QuotientAddGroup.automorphize f x) ∂μ_𝓕 := by
  /-
    G' : Type u_1
    inst✝¹⁰ : AddGroup G'
    inst✝⁹ : MeasurableSpace G'
    inst✝⁸ : TopologicalSpace G'
    inst✝⁷ : TopologicalAddGroup G'
    inst✝⁶ : BorelSpace G'
    μ' : MeasureTheory.Measure G'
    Γ' : AddSubgroup G'
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ' x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G' Γ')
    inst✝³ : BorelSpace (HasQuotient.Quotient G' Γ')
    𝓕' : Set G'
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ'.IsAddRightInvariant
    f : G' → K
    f_ℒ_1 : MeasureTheory.Integrable f μ'
    g : HasQuotient.Quotient G' Γ' → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientA …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientAddGroup.automor …
    h𝓕 : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem Γ'. …
    ⊢ Eq (MeasureTheory.integral μ' fun x => HMul.hMul (g ↑x) (f x)) (MeasureTheor …
  -/
  let π : G' → G' ⧸ Γ' := QuotientAddGroup.mk
  /-
    G' : Type u_1
    inst✝¹⁰ : AddGroup G'
    inst✝⁹ : MeasurableSpace G'
    inst✝⁸ : TopologicalSpace G'
    inst✝⁷ : TopologicalAddGroup G'
    inst✝⁶ : BorelSpace G'
    μ' : MeasureTheory.Measure G'
    Γ' : AddSubgroup G'
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ' x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G' Γ')
    inst✝³ : BorelSpace (HasQuotient.Quotient G' Γ')
    𝓕' : Set G'
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ'.IsAddRightInvariant
    f : G' → K
    f_ℒ_1 : MeasureTheory.Integrable f μ'
    g : HasQuotient.Quotient G' Γ' → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientA …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientAddGroup.automor …
    h𝓕 : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem Γ'. …
    π : G' → HasQuotient.Quotient G' Γ' := QuotientAddGroup.mk
    ⊢ Eq (MeasureTheory.integral μ' fun x => HMul.hMul (g ↑x) (f x)) (MeasureTheor …
  -/
  have meas_π : Measurable π := continuous_quotient_mk'.measurable
  have H₀ : QuotientAddGroup.automorphize ((g ∘ π) * f) = g * (QuotientAddGroup.automorphize f) :=
    by exact QuotientAddGroup.automorphize_smul_left f g
  calc ∫ (x : G'), g (π x) * f x ∂μ' =
    ∫ (x : G' ⧸ Γ'), QuotientAddGroup.automorphize ((g ∘ π) * f) x ∂μ_𝓕 := ?_
    _ = ∫ (x : G' ⧸ Γ'), g x * (QuotientAddGroup.automorphize f x) ∂μ_𝓕 := by simp [H₀]
  have H₁ : Integrable ((g ∘ π) * f) μ' := by
    have : AEStronglyMeasurable (fun (x : G') ↦ g (x : (G' ⧸ Γ'))) μ' :=
      (hg.mono_ac h𝓕.absolutelyContinuous_map).comp_measurable meas_π
    refine Integrable.essSup_smul f_ℒ_1 this ?_
    have hg' : AEStronglyMeasurable (fun x ↦ (‖g x‖₊ : ℝ≥0∞)) μ_𝓕 :=
      (ENNReal.continuous_coe.comp continuous_nnnorm).comp_aestronglyMeasurable hg
    rw [← essSup_comp_quotientAddGroup_mk h𝓕 hg'.aemeasurable]
    exact g_ℒ_infinity
  have H₂ : AEStronglyMeasurable (QuotientAddGroup.automorphize ((g ∘ π) * f)) μ_𝓕 := by
    simp_rw [H₀]
    exact hg.mul F_ae_measurable
  /-
    G' : Type u_1
    inst✝¹⁰ : AddGroup G'
    inst✝⁹ : MeasurableSpace G'
    inst✝⁸ : TopologicalSpace G'
    inst✝⁷ : TopologicalAddGroup G'
    inst✝⁶ : BorelSpace G'
    μ' : MeasureTheory.Measure G'
    Γ' : AddSubgroup G'
    inst✝⁵ : Countable (Subtype fun x => Membership.mem Γ' x)
    inst✝⁴ : MeasurableSpace (HasQuotient.Quotient G' Γ')
    inst✝³ : BorelSpace (HasQuotient.Quotient G' Γ')
    𝓕' : Set G'
    K : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedSpace Real K
    inst✝ : μ'.IsAddRightInvariant
    f : G' → K
    f_ℒ_1 : MeasureTheory.Integrable f μ'
    g : HasQuotient.Quotient G' Γ' → K
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map QuotientA …
    g_ℒ_infinity : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) (MeasureTheory.Mea …
    F_ae_measurable : MeasureTheory.AEStronglyMeasurable (QuotientAddGroup.automor …
    h𝓕 : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem Γ'. …
    π : G' → HasQuotient.Quotient G' Γ' := QuotientAddGroup.mk
    meas_π : Measurable π
    H₀ : Eq (QuotientAddGroup.automorphize (HMul.hMul (Function.comp g π) f)) (HMu …
    H₁ : MeasureTheory.Integrable (HMul.hMul (Function.comp g π) f) μ'
    H₂ : MeasureTheory.AEStronglyMeasurable (QuotientAddGroup.automorphize (HMul.h …
    ⊢ Eq (MeasureTheory.integral μ' fun x => HMul.hMul (g (π x)) (f x)) (MeasureTh …
  -/
  apply QuotientAddGroup.integral_eq_integral_automorphize h𝓕 H₁ H₂
  /-
    🎉 no goals
  -/


