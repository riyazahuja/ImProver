@[to_additive]
instance instTopologicalSpace (N : Subgroup G) : TopologicalSpace (G ⧸ N) :=
  instTopologicalSpaceQuotient


@[to_additive]
instance [CompactSpace G] (N : Subgroup G) : CompactSpace (G ⧸ N) :=
  Quotient.compactSpace


@[to_additive]
theorem isQuotientMap_mk (N : Subgroup G) : IsQuotientMap (mk : G → G ⧸ N) :=
  isQuotientMap_quot_mk


@[deprecated (since := "2024-10-22")]
alias quotientMap_mk := isQuotientMap_mk


@[to_additive]
theorem continuous_mk {N : Subgroup G} : Continuous (mk : G → G ⧸ N) :=
  continuous_quot_mk


@[to_additive]
theorem isOpenMap_coe : IsOpenMap ((↑) : G → G ⧸ N) := isOpenMap_quotient_mk'_mul


@[to_additive]
theorem isOpenQuotientMap_mk : IsOpenQuotientMap (mk : G → G ⧸ N) :=
  MulAction.isOpenQuotientMap_quotientMk


@[to_additive (attr := simp)]
theorem dense_preimage_mk {s : Set (G ⧸ N)} : Dense ((↑) ⁻¹' s : Set G) ↔ Dense s :=
  isOpenQuotientMap_mk.dense_preimage_iff


@[to_additive]
theorem dense_image_mk {s : Set G} :
    Dense (mk '' s : Set (G ⧸ N)) ↔ Dense (s * (N : Set G)) := by
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    N : Subgroup G
    s : Set G
    ⊢ Iff (Dense (Set.image QuotientGroup.mk s)) (Dense (HMul.hMul s ↑N))
  -/
  rw [← dense_preimage_mk, preimage_image_mk_eq_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
instance instContinuousSMul : ContinuousSMul G (G ⧸ N) where
  continuous_smul := by
    /-
      G : Type u_1
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : ContinuousMul G
      N : Subgroup G
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    rw [← (IsOpenQuotientMap.id.prodMap isOpenQuotientMap_mk).continuous_comp_iff]
    /-
      G : Type u_1
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : ContinuousMul G
      N : Subgroup G
      ⊢ Continuous (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id Quotie …
    -/
    exact continuous_mk.comp continuous_mul
    /-
      🎉 no goals
    -/


@[to_additive]
instance instContinuousConstSMul : ContinuousConstSMul G (G ⧸ N) := inferInstance


/-- A quotient of a locally compact group is locally compact. -/
@[to_additive]
instance instLocallyCompactSpace [LocallyCompactSpace G] (N : Subgroup G) :
    LocallyCompactSpace (G ⧸ N) :=
  QuotientGroup.isOpenQuotientMap_mk.locallyCompactSpace


@[to_additive (attr := deprecated "No deprecation message was provided." (since := "2024-10-05"))]
theorem continuous_smul₁ (x : G ⧸ N) : Continuous fun g : G => g • x :=
  continuous_id.smul continuous_const


/-- Neighborhoods in the quotient are precisely the map of neighborhoods in the prequotient. -/
@[to_additive
  "Neighborhoods in the quotient are precisely the map of neighborhoods in the prequotient."]
theorem nhds_eq (x : G) : 𝓝 (x : G ⧸ N) = Filter.map (↑) (𝓝 x) :=
  (isOpenQuotientMap_mk.map_nhds_eq _).symm


@[to_additive]
instance instFirstCountableTopology [FirstCountableTopology G] :
    FirstCountableTopology (G ⧸ N) where
  nhds_generated_countable := mk_surjective.forall.2 fun x ↦ nhds_eq N x ▸ inferInstance


/-- The quotient of a second countable topological group by a subgroup is second countable. -/
@[to_additive
  "The quotient of a second countable additive topological group by a subgroup is second
  countable."]
instance instSecondCountableTopology [SecondCountableTopology G] :
    SecondCountableTopology (G ⧸ N) :=
  ContinuousConstSMul.secondCountableTopology


@[to_additive (attr := deprecated "No deprecation message was provided." (since := "2024-08-05"))]
theorem nhds_one_isCountablyGenerated [FirstCountableTopology G] [N.Normal] :
    (𝓝 (1 : G ⧸ N)).IsCountablyGenerated :=
  inferInstance


@[to_additive]
instance instTopologicalGroup [N.Normal] : TopologicalGroup (G ⧸ N) where
  continuous_mul := by
    /-
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      N : Subgroup G
      inst✝ : N.Normal
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    rw [← (isOpenQuotientMap_mk.prodMap isOpenQuotientMap_mk).continuous_comp_iff]
    /-
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      N : Subgroup G
      inst✝ : N.Normal
      ⊢ Continuous (Function.comp (fun p => HMul.hMul p.1 p.2) (Prod.map QuotientGro …
    -/
    exact continuous_mk.comp continuous_mul
    /-
      🎉 no goals
    -/
  continuous_inv := continuous_inv.quotient_map' _


@[to_additive (attr := deprecated "No deprecation message was provided." (since := "2024-08-05"))]
theorem _root_.topologicalGroup_quotient [N.Normal] : TopologicalGroup (G ⧸ N) :=
  instTopologicalGroup N


@[to_additive]
theorem isClosedMap_coe {H : Subgroup G} (hH : IsCompact (H : Set G)) :
    IsClosedMap ((↑) : G → G ⧸ H) := by
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : Subgroup G
    hH : IsCompact ↑H
    ⊢ IsClosedMap QuotientGroup.mk
  -/
  intro t ht
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : Subgroup G
    hH : IsCompact ↑H
    t : Set G
    ht : IsClosed t
    ⊢ IsClosed (Set.image QuotientGroup.mk t)
  -/
  rw [← (isQuotientMap_mk H).isClosed_preimage, preimage_image_mk_eq_mul]
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : Subgroup G
    hH : IsCompact ↑H
    t : Set G
    ht : IsClosed t
    ⊢ IsClosed (HMul.hMul t ↑H)
  -/
  exact ht.mul_right_of_isCompact hH
  /-
    🎉 no goals
  -/


@[to_additive]
instance instT3Space [N.Normal] [hN : IsClosed (N : Set G)] : T3Space (G ⧸ N) := by
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    N : Subgroup G
    inst✝ : N.Normal
    hN : IsClosed ↑N
    ⊢ T3Space (HasQuotient.Quotient G N)
  -/
  rw [← QuotientGroup.ker_mk' N] at hN
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    N : Subgroup G
    inst✝ : N.Normal
    hN : IsClosed ↑(QuotientGroup.mk' N).ker
    ⊢ T3Space (HasQuotient.Quotient G N)
  -/
  haveI := TopologicalGroup.t1Space (G ⧸ N) ((isQuotientMap_mk N).isClosed_preimage.mp hN)
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    N : Subgroup G
    inst✝ : N.Normal
    hN : IsClosed ↑(QuotientGroup.mk' N).ker
    this : T1Space (HasQuotient.Quotient G N)
    ⊢ T3Space (HasQuotient.Quotient G N)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


