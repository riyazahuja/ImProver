/-- Put the same topological space structure on the opposite monoid as on the original space. -/
@[to_additive "Put the same topological space structure on the opposite monoid as on the original
space."]
instance instTopologicalSpaceMulOpposite [TopologicalSpace M] : TopologicalSpace Mᵐᵒᵖ :=
  TopologicalSpace.induced (unop : Mᵐᵒᵖ → M) ‹_›


@[to_additive (attr := continuity)]
theorem continuous_unop : Continuous (unop : Mᵐᵒᵖ → M) :=
  continuous_induced_dom


@[to_additive (attr := continuity)]
theorem continuous_op : Continuous (op : M → Mᵐᵒᵖ) :=
  continuous_induced_rng.2 continuous_id


/-- `MulOpposite.op` as a homeomorphism. -/
@[to_additive (attr := simps!) "`AddOpposite.op` as a homeomorphism."]
def opHomeomorph : M ≃ₜ Mᵐᵒᵖ where
  toEquiv := opEquiv
  continuous_toFun := continuous_op
  continuous_invFun := continuous_unop


@[to_additive]
instance instT2Space [T2Space M] : T2Space Mᵐᵒᵖ := opHomeomorph.t2Space


@[to_additive]
instance instDiscreteTopology [DiscreteTopology M] : DiscreteTopology Mᵐᵒᵖ :=
  opHomeomorph.symm.isEmbedding.discreteTopology


@[to_additive (attr := simp)]
theorem map_op_nhds (x : M) : map (op : M → Mᵐᵒᵖ) (𝓝 x) = 𝓝 (op x) :=
  opHomeomorph.map_nhds_eq x


@[to_additive (attr := simp)]
theorem map_unop_nhds (x : Mᵐᵒᵖ) : map (unop : Mᵐᵒᵖ → M) (𝓝 x) = 𝓝 (unop x) :=
  opHomeomorph.symm.map_nhds_eq x


@[to_additive (attr := simp)]
theorem comap_op_nhds (x : Mᵐᵒᵖ) : comap (op : M → Mᵐᵒᵖ) (𝓝 x) = 𝓝 (unop x) :=
  opHomeomorph.comap_nhds_eq x


@[to_additive (attr := simp)]
theorem comap_unop_nhds (x : M) : comap (unop : Mᵐᵒᵖ → M) (𝓝 x) = 𝓝 (op x) :=
  opHomeomorph.symm.comap_nhds_eq x


/-- The units of a monoid are equipped with a topology, via the embedding into `M × M`. -/
@[to_additive "The additive units of a monoid are equipped with a topology, via the embedding into
`M × M`."]
instance instTopologicalSpaceUnits : TopologicalSpace Mˣ :=
  TopologicalSpace.induced (embedProduct M) inferInstance


@[to_additive]
theorem isInducing_embedProduct : IsInducing (embedProduct M) := ⟨rfl⟩


@[deprecated (since := "2024-10-28")] alias inducing_embedProduct := isInducing_embedProduct


@[to_additive]
theorem isEmbedding_embedProduct : IsEmbedding (embedProduct M) :=
  ⟨isInducing_embedProduct, embedProduct_injective M⟩


@[deprecated (since := "2024-10-26")]
alias embedding_embedProduct := isEmbedding_embedProduct


@[to_additive]
instance instT2Space [T2Space M] : T2Space Mˣ := isEmbedding_embedProduct.t2Space


@[to_additive]
instance instDiscreteTopology [DiscreteTopology M] : DiscreteTopology Mˣ :=
  isEmbedding_embedProduct.discreteTopology


@[to_additive] lemma topology_eq_inf :
    instTopologicalSpaceUnits =
      .induced (val : Mˣ → M) ‹_› ⊓ .induced (fun u ↦ ↑u⁻¹ : Mˣ → M) ‹_› := by
  simp only [isInducing_embedProduct.1, instTopologicalSpaceProd, induced_inf,
                                                       /-
                                                         M : Type u_1
                                                         inst✝¹ : TopologicalSpace M
                                                         inst✝ : Monoid M
                                                         ⊢ Eq (Min.min (TopologicalSpace.induced (Function.comp Prod.fst ⇑(Units.embedP …
                                                       -/
    instTopologicalSpaceMulOpposite, induced_compose]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- An auxiliary lemma that can be used to prove that coercion `Mˣ → M` is a topological embedding.
Use `Units.isEmbedding_val₀`, `Units.isEmbedding_val`, or `toUnits_homeomorph` instead. -/
@[to_additive "An auxiliary lemma that can be used to prove that coercion `AddUnits M → M` is a
topological embedding. Use `AddUnits.isEmbedding_val` or `toAddUnits_homeomorph` instead."]
lemma isEmbedding_val_mk' {M : Type*} [Monoid M] [TopologicalSpace M] {f : M → M}
    (hc : ContinuousOn f {x : M | IsUnit x}) (hf : ∀ u : Mˣ, f u.1 = ↑u⁻¹) :
    IsEmbedding (val : Mˣ → M) := by
  /-
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : TopologicalSpace M
    f : M → M
    hc : ContinuousOn f (setOf fun x => IsUnit x)
    hf : ∀ (u : Units M), Eq (f ↑u) ↑(Inv.inv u)
    ⊢ Topology.IsEmbedding Units.val
  -/
  refine ⟨⟨?_⟩, ext⟩
  rw [topology_eq_inf, inf_eq_left, ← continuous_iff_le_induced,
    @continuous_iff_continuousAt _ _ (.induced _ _)]
  /-
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : TopologicalSpace M
    f : M → M
    hc : ContinuousOn f (setOf fun x => IsUnit x)
    hf : ∀ (u : Units M), Eq (f ↑u) ↑(Inv.inv u)
    ⊢ ∀ (x : Units M), ContinuousAt (fun u => ↑(Inv.inv u)) x
  -/
  intros u s hs
  /-
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : TopologicalSpace M
    f : M → M
    hc : ContinuousOn f (setOf fun x => IsUnit x)
    hf : ∀ (u : Units M), Eq (f ↑u) ↑(Inv.inv u)
    u : Units M
    s : Set M
    hs : Membership.mem (nhds ((fun u => ↑(Inv.inv u)) u)) s
    ⊢ Membership.mem (Filter.map (fun u => ↑(Inv.inv u)) (nhds u)) s
  -/
  simp only [← hf, nhds_induced, Filter.mem_map] at hs ⊢
  /-
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : TopologicalSpace M
    f : M → M
    hc : ContinuousOn f (setOf fun x => IsUnit x)
    hf : ∀ (u : Units M), Eq (f ↑u) ↑(Inv.inv u)
    u : Units M
    s : Set M
    hs : Membership.mem (nhds (f ↑u)) s
    ⊢ Membership.mem (Filter.comap Units.val (nhds ↑u)) (Set.preimage (fun u => f  …
  -/
  exact ⟨_, mem_inf_principal.1 (hc u u.isUnit hs), fun u' hu' ↦ hu' u'.isUnit⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias embedding_val_mk' := isEmbedding_val_mk'


/-- An auxiliary lemma that can be used to prove that coercion `Mˣ → M` is a topological embedding.
Use `Units.isEmbedding_val₀`, `Units.isEmbedding_val`, or `toUnits_homeomorph` instead. -/
@[to_additive "An auxiliary lemma that can be used to prove that coercion `AddUnits M → M` is a
topological embedding. Use `AddUnits.isEmbedding_val` or `toAddUnits_homeomorph` instead."]
lemma embedding_val_mk {M : Type*} [DivisionMonoid M] [TopologicalSpace M]
    (h : ContinuousOn Inv.inv {x : M | IsUnit x}) : IsEmbedding (val : Mˣ → M) :=
  isEmbedding_val_mk' h fun u ↦ (val_inv_eq_inv_val u).symm


@[to_additive]
theorem continuous_embedProduct : Continuous (embedProduct M) :=
  continuous_induced_dom


@[to_additive]
theorem continuous_val : Continuous ((↑) : Mˣ → M) :=
  (@continuous_embedProduct M _ _).fst


@[to_additive]
protected theorem continuous_iff {f : X → Mˣ} :
    Continuous f ↔ Continuous (val ∘ f) ∧ Continuous (fun x => ↑(f x)⁻¹ : X → M) := by
  simp only [isInducing_embedProduct.continuous_iff, embedProduct_apply, Function.comp_def,
    continuous_prod_mk, opHomeomorph.symm.isInducing.continuous_iff, opHomeomorph_symm_apply,
    unop_op]


@[to_additive]
theorem continuous_coe_inv : Continuous (fun u => ↑u⁻¹ : Mˣ → M) :=
  (Units.continuous_iff.1 continuous_id).2


