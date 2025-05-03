/-- Let `E` be a topological vector space over a normed field `𝕜`, let `α` be any type.
Let `H` be a submodule of `α →ᵤ E` such that the range of each `f ∈ H` is von Neumann bounded.
Then `H` is a topological vector space over `𝕜`,
i.e., the pointwise scalar multiplication is continuous in both variables.

For convenience we require that `H` is a vector space over `𝕜`
with a topology induced by `UniformFun.ofFun ∘ φ`, where `φ : H →ₗ[𝕜] (α → E)`. -/
lemma UniformFun.continuousSMul_induced_of_range_bounded (φ : hom)
    (hφ : IsInducing (ofFun ∘ φ)) (h : ∀ u : H, Bornology.IsVonNBounded 𝕜 (Set.range (φ u))) :
    ContinuousSMul 𝕜 H := by
  have : TopologicalAddGroup H :=
    let ofFun' : (α → E) →+ (α →ᵤ E) := AddMonoidHom.id _
    IsInducing.topologicalAddGroup (ofFun'.comp (φ : H →+ (α → E))) hφ
  have hb : (𝓝 (0 : H)).HasBasis (· ∈ 𝓝 (0 : E)) fun V ↦ {u | ∀ x, φ u x ∈ V} := by
    simp only [hφ.nhds_eq_comap, Function.comp_apply, map_zero]
    exact UniformFun.hasBasis_nhds_zero.comap _
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝¹⁰ : NormedField 𝕜
    inst✝⁹ : AddCommGroup H
    inst✝⁸ : Module 𝕜 H
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
    h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
    this : TopologicalAddGroup H
    hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
    ⊢ ContinuousSMul 𝕜 H
  -/
  apply ContinuousSMul.of_basis_zero hb
    /-
      case hsmul
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      ⊢ ∀ {i : Set E}, Membership.mem (nhds 0) i → Exists fun V => And (Membership.m …
    -/
  · intro U hU
    have : Tendsto (fun x : 𝕜 × E ↦ x.1 • x.2) (𝓝 0) (𝓝 0) :=
      continuous_smul.tendsto' _ _ (zero_smul _ _)
    rcases ((Filter.basis_sets _).prod_nhds (Filter.basis_sets _)).tendsto_left_iff.1 this U hU
      with ⟨⟨V, W⟩, ⟨hV, hW⟩, hVW⟩
    /-
      case hsmul.intro.mk.intro.intro
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this✝ : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      U : Set E
      hU : Membership.mem (nhds 0) U
      this : Filter.Tendsto (fun x => HSMul.hSMul x.1 x.2) (nhds 0) (nhds 0)
      V : Set 𝕜
      W : Set E
      hVW : Set.MapsTo (fun x => HSMul.hSMul x.1 x.2) (SProd.sprod (id { fst := V, s …
      hV : Membership.mem (nhds 0) { fst := V, snd := W }.1
      hW : Membership.mem (nhds 0) { fst := V, snd := W }.2
      ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun j => And (Member …
    -/
    refine ⟨V, hV, W, hW, Set.smul_subset_iff.2 fun a ha u hu x ↦ ?_⟩
    /-
      case hsmul.intro.mk.intro.intro
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this✝ : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      U : Set E
      hU : Membership.mem (nhds 0) U
      this : Filter.Tendsto (fun x => HSMul.hSMul x.1 x.2) (nhds 0) (nhds 0)
      V : Set 𝕜
      W : Set E
      hVW : Set.MapsTo (fun x => HSMul.hSMul x.1 x.2) (SProd.sprod (id { fst := V, s …
      hV : Membership.mem (nhds 0) { fst := V, snd := W }.1
      hW : Membership.mem (nhds 0) { fst := V, snd := W }.2
      a : 𝕜
      ha : Membership.mem V a
      u : H
      hu : Membership.mem (setOf fun u => ∀ (x : α), Membership.mem W (φ u x)) u
      x : α
      ⊢ Membership.mem U (φ (HSMul.hSMul a u) x)
    -/
    rw [map_smul]
    /-
      case hsmul.intro.mk.intro.intro
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this✝ : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      U : Set E
      hU : Membership.mem (nhds 0) U
      this : Filter.Tendsto (fun x => HSMul.hSMul x.1 x.2) (nhds 0) (nhds 0)
      V : Set 𝕜
      W : Set E
      hVW : Set.MapsTo (fun x => HSMul.hSMul x.1 x.2) (SProd.sprod (id { fst := V, s …
      hV : Membership.mem (nhds 0) { fst := V, snd := W }.1
      hW : Membership.mem (nhds 0) { fst := V, snd := W }.2
      a : 𝕜
      ha : Membership.mem V a
      u : H
      hu : Membership.mem (setOf fun u => ∀ (x : α), Membership.mem W (φ u x)) u
      x : α
      ⊢ Membership.mem U (HSMul.hSMul a (φ u) x)
    -/
    exact hVW (Set.mk_mem_prod ha (hu x))
    /-
      🎉 no goals
    -/
    /-
      case hsmul_left
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      ⊢ ∀ (x₀ : 𝕜) {i : Set E}, Membership.mem (nhds 0) i → Exists fun j => And (Mem …
    -/
  · intro c U hU
    have : Tendsto (c • · : E → E) (𝓝 0) (𝓝 0) :=
      (continuous_const_smul c).tendsto' _ _ (smul_zero _)
    /-
      case hsmul_left
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this✝ : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      c : 𝕜
      U : Set E
      hU : Membership.mem (nhds 0) U
      this : Filter.Tendsto (fun x => HSMul.hSMul c x) (nhds 0) (nhds 0)
      ⊢ Exists fun j => And (Membership.mem (nhds 0) j) (Set.MapsTo (fun x => HSMul. …
    -/
    refine ⟨_, this hU, fun u hu x ↦ ?_⟩
    /-
      case hsmul_left
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this✝ : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      c : 𝕜
      U : Set E
      hU : Membership.mem (nhds 0) U
      this : Filter.Tendsto (fun x => HSMul.hSMul c x) (nhds 0) (nhds 0)
      u : H
      hu : Membership.mem (setOf fun u => ∀ (x : α), Membership.mem (Set.preimage (f …
      x : α
      ⊢ Membership.mem U (φ ((fun x => HSMul.hSMul c x) u) x)
    -/
    simpa only [map_smul] using hu x
    /-
      🎉 no goals
    -/
    /-
      case hsmul_right
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      ⊢ ∀ (m₀ : H) {i : Set E}, Membership.mem (nhds 0) i → Filter.Eventually (fun x …
    -/
  · intro u U hU
    /-
      case hsmul_right
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      u : H
      U : Set E
      hU : Membership.mem (nhds 0) U
      ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun u => ∀ (x : α), Member …
    -/
    simp only [Set.mem_setOf_eq, map_smul, Pi.smul_apply]
    /-
      case hsmul_right
      𝕜 : Type u_1
      α : Type u_2
      E : Type u_3
      H : Type u_4
      hom : Type u_5
      inst✝¹⁰ : NormedField 𝕜
      inst✝⁹ : AddCommGroup H
      inst✝⁸ : Module 𝕜 H
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module 𝕜 E
      inst✝⁵ : TopologicalSpace H
      inst✝⁴ : UniformSpace E
      inst✝³ : UniformAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : FunLike hom H (α → E)
      inst✝ : LinearMapClass hom 𝕜 H (α → E)
      φ : hom
      hφ : Topology.IsInducing (Function.comp ⇑UniformFun.ofFun ⇑φ)
      h : ∀ (u : H), Bornology.IsVonNBounded 𝕜 (Set.range (φ u))
      this : TopologicalAddGroup H
      hb : (nhds 0).HasBasis (fun x => Membership.mem (nhds 0) x) fun V => setOf fun …
      u : H
      U : Set E
      hU : Membership.mem (nhds 0) U
      ⊢ Filter.Eventually (fun x => ∀ (x_1 : α), Membership.mem U (HSMul.hSMul x (φ  …
    -/
    simpa only [Set.mapsTo_range_iff] using (h u hU).eventually_nhds_zero (mem_of_mem_nhds hU)
    /-
      🎉 no goals
    -/


/-- Let `E` be a TVS, `𝔖 : Set (Set α)` and `H` a submodule of `α →ᵤ[𝔖] E`. If the image of any
`S ∈ 𝔖` by any `u ∈ H` is bounded (in the sense of `Bornology.IsVonNBounded`), then `H`,
equipped with the topology of `𝔖`-convergence, is a TVS.

For convenience, we don't literally ask for `H : Submodule (α →ᵤ[𝔖] E)`. Instead, we prove the
result for any vector space `H` equipped with a linear inducing to `α →ᵤ[𝔖] E`, which is often
easier to use. We also state the `Submodule` version as
`UniformOnFun.continuousSMul_submodule_of_image_bounded`. -/
lemma UniformOnFun.continuousSMul_induced_of_image_bounded (φ : hom) (hφ : IsInducing (ofFun 𝔖 ∘ φ))
    (h : ∀ u : H, ∀ s ∈ 𝔖, Bornology.IsVonNBounded 𝕜 ((φ u : α → E) '' s)) :
    ContinuousSMul 𝕜 H := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝¹⁰ : NormedField 𝕜
    inst✝⁹ : AddCommGroup H
    inst✝⁸ : Module 𝕜 H
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    𝔖 : Set (Set α)
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    hφ : Topology.IsInducing (Function.comp ⇑(UniformOnFun.ofFun 𝔖) ⇑φ)
    h : ∀ (u : H) (s : Set α), Membership.mem 𝔖 s → Bornology.IsVonNBounded 𝕜 (Set …
    ⊢ ContinuousSMul 𝕜 H
  -/
  obtain rfl := hφ.eq_induced; clear hφ
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup H
    inst✝⁷ : Module 𝕜 H
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    𝔖 : Set (Set α)
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    h : ∀ (u : H) (s : Set α), Membership.mem 𝔖 s → Bornology.IsVonNBounded 𝕜 (Set …
    ⊢ ContinuousSMul 𝕜 H
  -/
  simp only [induced_iInf, UniformOnFun.topologicalSpace_eq, induced_compose]
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup H
    inst✝⁷ : Module 𝕜 H
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    𝔖 : Set (Set α)
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    h : ∀ (u : H) (s : Set α), Membership.mem 𝔖 s → Bornology.IsVonNBounded 𝕜 (Set …
    ⊢ ContinuousSMul 𝕜 H
  -/
  refine continuousSMul_iInf fun s ↦ continuousSMul_iInf fun hs ↦ ?_
  letI : TopologicalSpace H :=
    .induced (UniformFun.ofFun ∘ s.restrict ∘ φ) (UniformFun.topologicalSpace s E)
  set φ' : H →ₗ[𝕜] (s → E) :=
    { toFun := s.restrict ∘ φ,
      map_smul' := fun c x ↦ by exact congr_arg s.restrict (map_smul φ c x),
      map_add' := fun x y ↦ by exact congr_arg s.restrict (map_add φ x y) }
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup H
    inst✝⁷ : Module 𝕜 H
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    𝔖 : Set (Set α)
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    h : ∀ (u : H) (s : Set α), Membership.mem 𝔖 s → Bornology.IsVonNBounded 𝕜 (Set …
    s : Set α
    hs : Membership.mem 𝔖 s
    this : TopologicalSpace H := TopologicalSpace.induced (Function.comp (⇑Uniform …
    φ' : LinearMap (RingHom.id 𝕜) H (↑s → E) := { toFun := Function.comp s.restric …
    ⊢ ContinuousSMul 𝕜 H
  -/
  refine UniformFun.continuousSMul_induced_of_range_bounded 𝕜 s E H φ' ⟨rfl⟩ fun u ↦ ?_
  /-
    𝕜 : Type u_1
    α : Type u_2
    E : Type u_3
    H : Type u_4
    hom : Type u_5
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup H
    inst✝⁷ : Module 𝕜 H
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : UniformSpace E
    inst✝³ : UniformAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    𝔖 : Set (Set α)
    inst✝¹ : FunLike hom H (α → E)
    inst✝ : LinearMapClass hom 𝕜 H (α → E)
    φ : hom
    h : ∀ (u : H) (s : Set α), Membership.mem 𝔖 s → Bornology.IsVonNBounded 𝕜 (Set …
    s : Set α
    hs : Membership.mem 𝔖 s
    this : TopologicalSpace H := TopologicalSpace.induced (Function.comp (⇑Uniform …
    φ' : LinearMap (RingHom.id 𝕜) H (↑s → E) := { toFun := Function.comp s.restric …
    u : H
    ⊢ Bornology.IsVonNBounded 𝕜 (Set.range (φ' u))
  -/
  simpa only [Set.image_eq_range] using h u s hs
  /-
    🎉 no goals
  -/


/-- Let `E` be a TVS, `𝔖 : Set (Set α)` and `H` a submodule of `α →ᵤ[𝔖] E`. If the image of any
`S ∈ 𝔖` by any `u ∈ H` is bounded (in the sense of `Bornology.IsVonNBounded`), then `H`,
equipped with the topology of `𝔖`-convergence, is a TVS.

If you have a hard time using this lemma, try the one above instead. -/
theorem UniformOnFun.continuousSMul_submodule_of_image_bounded (H : Submodule 𝕜 (α →ᵤ[𝔖] E))
    (h : ∀ u ∈ H, ∀ s ∈ 𝔖, Bornology.IsVonNBounded 𝕜 (u '' s)) :
    @ContinuousSMul 𝕜 H _ _ ((UniformOnFun.topologicalSpace α E 𝔖).induced ((↑) : H → α →ᵤ[𝔖] E)) :=
  UniformOnFun.continuousSMul_induced_of_image_bounded 𝕜 α E H
    (LinearMap.id.domRestrict H : H →ₗ[𝕜] α → E) IsInducing.subtypeVal fun ⟨u, hu⟩ => h u hu


