@[to_additive]
instance Pi.instUniformGroup {ι : Type*} {G : ι → Type*} [∀ i, UniformSpace (G i)]
    [∀ i, Group (G i)] [∀ i, UniformGroup (G i)] : UniformGroup (∀ i, G i) where
  uniformContinuous_div := uniformContinuous_pi.mpr fun i ↦
    (uniformContinuous_proj G i).comp uniformContinuous_fst |>.div <|
      (uniformContinuous_proj G i).comp uniformContinuous_snd


@[to_additive]
theorem isUniformEmbedding_translate_mul (a : α) : IsUniformEmbedding fun x : α => x * a :=
  { comap_uniformity := by
      /-
        α : Type u_1
        inst✝² : UniformSpace α
        inst✝¹ : Group α
        inst✝ : UniformGroup α
        a : α
        ⊢ Eq (Filter.comap (fun x => { fst := HMul.hMul x.1 a, snd := HMul.hMul x.2 a  …
      -/
      nth_rw 1 [← uniformity_translate_mul a, comap_map]
      /-
        α : Type u_1
        inst✝² : UniformSpace α
        inst✝¹ : Group α
        inst✝ : UniformGroup α
        a : α
        ⊢ Function.Injective fun x => { fst := HMul.hMul x.1 a, snd := HMul.hMul x.2 a }
      -/
      rintro ⟨p₁, p₂⟩ ⟨q₁, q₂⟩
      /-
        case mk.mk
        α : Type u_1
        inst✝² : UniformSpace α
        inst✝¹ : Group α
        inst✝ : UniformGroup α
        a p₁ p₂ q₁ q₂ : α
        ⊢ Eq ((fun x => { fst := HMul.hMul x.1 a, snd := HMul.hMul x.2 a }) { fst := p …
      -/
      simp only [Prod.mk.injEq, mul_left_inj, imp_self]
      /-
        🎉 no goals
      -/
    injective := mul_left_injective a }


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_translate_mul := isUniformEmbedding_translate_mul


@[to_additive]
lemma IsUniformInducing.uniformGroup {γ : Type*} [Group γ] [UniformSpace γ] [UniformGroup γ]
    [UniformSpace β] {F : Type*} [FunLike F β γ] [MonoidHomClass F β γ]
    (f : F) (hf : IsUniformInducing f) :
    UniformGroup β where
  uniformContinuous_div := by
    /-
      β : Type u_2
      inst✝⁶ : Group β
      γ : Type u_3
      inst✝⁵ : Group γ
      inst✝⁴ : UniformSpace γ
      inst✝³ : UniformGroup γ
      inst✝² : UniformSpace β
      F : Type u_4
      inst✝¹ : FunLike F β γ
      inst✝ : MonoidHomClass F β γ
      f : F
      hf : IsUniformInducing ⇑f
      ⊢ UniformContinuous fun p => HDiv.hDiv p.1 p.2
    -/
    simp_rw [hf.uniformContinuous_iff, Function.comp_def, map_div]
    /-
      β : Type u_2
      inst✝⁶ : Group β
      γ : Type u_3
      inst✝⁵ : Group γ
      inst✝⁴ : UniformSpace γ
      inst✝³ : UniformGroup γ
      inst✝² : UniformSpace β
      F : Type u_4
      inst✝¹ : FunLike F β γ
      inst✝ : MonoidHomClass F β γ
      f : F
      hf : IsUniformInducing ⇑f
      ⊢ UniformContinuous fun x => HDiv.hDiv (f x.1) (f x.2)
    -/
    exact uniformContinuous_div.comp (hf.uniformContinuous.prodMap hf.uniformContinuous)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformGroup := IsUniformInducing.uniformGroup


@[to_additive]
protected theorem UniformGroup.comap {γ : Type*} [Group γ] {u : UniformSpace γ} [UniformGroup γ]
    {F : Type*} [FunLike F β γ] [MonoidHomClass F β γ] (f : F) : @UniformGroup β (u.comap f) _ :=
  letI : UniformSpace β := u.comap f; IsUniformInducing.uniformGroup f ⟨rfl⟩


@[to_additive]
instance uniformGroup (S : Subgroup α) : UniformGroup S := .comap S.subtype


@[to_additive]
theorem CauchySeq.mul {ι : Type*} [Preorder ι] {u v : ι → α} (hu : CauchySeq u)
    (hv : CauchySeq v) : CauchySeq (u * v) :=
  uniformContinuous_mul.comp_cauchySeq (hu.prod hv)


@[to_additive]
theorem CauchySeq.mul_const {ι : Type*} [Preorder ι] {u : ι → α} {x : α} (hu : CauchySeq u) :
    CauchySeq fun n => u n * x :=
  (uniformContinuous_id.mul uniformContinuous_const).comp_cauchySeq hu


@[to_additive]
theorem CauchySeq.const_mul {ι : Type*} [Preorder ι] {u : ι → α} {x : α} (hu : CauchySeq u) :
    CauchySeq fun n => x * u n :=
  (uniformContinuous_const.mul uniformContinuous_id).comp_cauchySeq hu


@[to_additive]
theorem CauchySeq.inv {ι : Type*} [Preorder ι] {u : ι → α} (h : CauchySeq u) :
    CauchySeq u⁻¹ :=
  uniformContinuous_inv.comp_cauchySeq h


@[to_additive]
theorem totallyBounded_iff_subset_finite_iUnion_nhds_one {s : Set α} :
    TotallyBounded s ↔ ∀ U ∈ 𝓝 (1 : α), ∃ t : Set α, t.Finite ∧ s ⊆ ⋃ y ∈ t, y • U :=
  (𝓝 (1 : α)).basis_sets.uniformity_of_nhds_one_inv_mul_swapped.totallyBounded_iff.trans <| by
    /-
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set α
      ⊢ Iff (∀ (i : Set α), Membership.mem (nhds 1) i → Exists fun t => And t.Finite …
    -/
    simp [← preimage_smul_inv, preimage]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem totallyBounded_inv {s : Set α} (hs : TotallyBounded s) : TotallyBounded (s⁻¹) := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    s : Set α
    hs : TotallyBounded s
    ⊢ TotallyBounded (Inv.inv s)
  -/
  convert TotallyBounded.image hs uniformContinuous_inv
  /-
    case h.e'_3.h.e
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    s : Set α
    hs : TotallyBounded s
    ⊢ Eq Inv.inv (Set.image fun x => Inv.inv x)
  -/
  aesop
  /-
    🎉 no goals
  -/


@[to_additive]
theorem TendstoUniformlyOnFilter.mul (hf : TendstoUniformlyOnFilter f g l l')
    (hf' : TendstoUniformlyOnFilter f' g' l l') : TendstoUniformlyOnFilter (f * f') (g * g') l l' :=
  fun u hu =>
  ((uniformContinuous_mul.comp_tendstoUniformlyOnFilter (hf.prod hf')) u hu).diag_of_prod_left


@[to_additive]
theorem TendstoUniformlyOnFilter.div (hf : TendstoUniformlyOnFilter f g l l')
    (hf' : TendstoUniformlyOnFilter f' g' l l') : TendstoUniformlyOnFilter (f / f') (g / g') l l' :=
  fun u hu =>
  ((uniformContinuous_div.comp_tendstoUniformlyOnFilter (hf.prod hf')) u hu).diag_of_prod_left


@[to_additive]
theorem TendstoUniformlyOn.mul (hf : TendstoUniformlyOn f g l s)
    (hf' : TendstoUniformlyOn f' g' l s) : TendstoUniformlyOn (f * f') (g * g') l s := fun u hu =>
  ((uniformContinuous_mul.comp_tendstoUniformlyOn (hf.prod hf')) u hu).diag_of_prod


@[to_additive]
theorem TendstoUniformlyOn.div (hf : TendstoUniformlyOn f g l s)
    (hf' : TendstoUniformlyOn f' g' l s) : TendstoUniformlyOn (f / f') (g / g') l s := fun u hu =>
  ((uniformContinuous_div.comp_tendstoUniformlyOn (hf.prod hf')) u hu).diag_of_prod


@[to_additive]
theorem TendstoUniformly.mul (hf : TendstoUniformly f g l) (hf' : TendstoUniformly f' g' l) :
    TendstoUniformly (f * f') (g * g') l := fun u hu =>
  ((uniformContinuous_mul.comp_tendstoUniformly (hf.prod hf')) u hu).diag_of_prod


@[to_additive]
theorem TendstoUniformly.div (hf : TendstoUniformly f g l) (hf' : TendstoUniformly f' g' l) :
    TendstoUniformly (f / f') (g / g') l := fun u hu =>
  ((uniformContinuous_div.comp_tendstoUniformly (hf.prod hf')) u hu).diag_of_prod


@[to_additive]
theorem UniformCauchySeqOn.mul (hf : UniformCauchySeqOn f l s) (hf' : UniformCauchySeqOn f' l s) :
    UniformCauchySeqOn (f * f') l s := fun u hu => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Type u_3
    l : Filter ι
    f f' : ι → β → α
    s : Set β
    hf : UniformCauchySeqOn f l s
    hf' : UniformCauchySeqOn f' l s
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ⊢ Filter.Eventually (fun m => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  simpa using (uniformContinuous_mul.comp_uniformCauchySeqOn (hf.prod' hf')) u hu
  /-
    🎉 no goals
  -/


@[to_additive]
theorem UniformCauchySeqOn.div (hf : UniformCauchySeqOn f l s) (hf' : UniformCauchySeqOn f' l s) :
    UniformCauchySeqOn (f / f') l s := fun u hu => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Type u_3
    l : Filter ι
    f f' : ι → β → α
    s : Set β
    hf : UniformCauchySeqOn f l s
    hf' : UniformCauchySeqOn f' l s
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ⊢ Filter.Eventually (fun m => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  simpa using (uniformContinuous_div.comp_uniformCauchySeqOn (hf.prod' hf')) u hu
  /-
    🎉 no goals
  -/


@[to_additive]
theorem topologicalGroup_is_uniform_of_compactSpace [CompactSpace G] : UniformGroup G :=
  ⟨by
    /-
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      ⊢ UniformContinuous fun p => HDiv.hDiv p.1 p.2
    -/
    apply CompactSpace.uniformContinuous_of_continuous
    /-
      case h
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      ⊢ Continuous fun p => HDiv.hDiv p.1 p.2
    -/
    exact continuous_div'⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance Subgroup.isClosed_of_discrete [T2Space G] {H : Subgroup G} [DiscreteTopology H] :
    IsClosed (H : Set G) := by
  obtain ⟨V, V_in, VH⟩ : ∃ (V : Set G), V ∈ 𝓝 (1 : G) ∧ V ∩ (H : Set G) = {1} :=
    nhds_inter_eq_singleton_of_mem_discrete H.one_mem
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    ⊢ IsClosed ↑H
  -/
  have : (fun p : G × G => p.2 / p.1) ⁻¹' V ∈ 𝓤 G := preimage_mem_comap V_in
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    ⊢ IsClosed ↑H
  -/
  apply isClosed_of_spaced_out this
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    ⊢ (↑H).Pairwise fun x y => Not (Membership.mem (Set.preimage (fun p => HDiv.hD …
  -/
  intro h h_in h' h'_in
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    h : G
    h_in : Membership.mem (↑H) h
    h' : G
    h'_in : Membership.mem (↑H) h'
    ⊢ Ne h h' → (fun x y => Not (Membership.mem (Set.preimage (fun p => HDiv.hDiv  …
  -/
  contrapose!
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    h : G
    h_in : Membership.mem (↑H) h
    h' : G
    h'_in : Membership.mem (↑H) h'
    ⊢ Not (Not (Membership.mem (Set.preimage (fun p => HDiv.hDiv p.2 p.1) V) { fst …
  -/
  simp only [Set.mem_preimage, not_not]
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    h : G
    h_in : Membership.mem (↑H) h
    h' : G
    h'_in : Membership.mem (↑H) h'
    ⊢ Membership.mem V (HDiv.hDiv h' h) → Eq h h'
  -/
  rintro (hyp : h' / h ∈ V)
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p.1 …
    h : G
    h_in : Membership.mem (↑H) h
    h' : G
    h'_in : Membership.mem (↑H) h'
    hyp : Membership.mem V (HDiv.hDiv h' h)
    ⊢ Eq h h'
  -/
  have : h' / h ∈ ({1} : Set G) := VH ▸ Set.mem_inter hyp (H.div_mem h'_in h_in)
  /-
    case intro.intro
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Subgroup G
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem H x)
    V : Set G
    V_in : Membership.mem (nhds 1) V
    VH : Eq (Inter.inter V ↑H) (Singleton.singleton 1)
    this✝ : Membership.mem (uniformity G) (Set.preimage (fun p => HDiv.hDiv p.2 p. …
    h : G
    h_in : Membership.mem (↑H) h
    h' : G
    h'_in : Membership.mem (↑H) h'
    hyp : Membership.mem V (HDiv.hDiv h' h)
    this : Membership.mem (Singleton.singleton 1) (HDiv.hDiv h' h)
    ⊢ Eq h h'
  -/
  exact (eq_of_div_eq_one this).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Subgroup.tendsto_coe_cofinite_of_discrete [T2Space G] (H : Subgroup G) [DiscreteTopology H] :
    Tendsto ((↑) : H → G) cofinite (cocompact _) :=
  IsClosed.tendsto_coe_cofinite_of_discreteTopology inferInstance inferInstance


@[to_additive]
lemma MonoidHom.tendsto_coe_cofinite_of_discrete [T2Space G] {H : Type*} [Group H] {f : H →* G}
    (hf : Function.Injective f) (hf' : DiscreteTopology f.range) :
    Tendsto f cofinite (cocompact _) := by
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom H G
    hf : Function.Injective ⇑f
    hf' : DiscreteTopology (Subtype fun x => Membership.mem f.range x)
    ⊢ Filter.Tendsto (⇑f) Filter.cofinite (Filter.cocompact G)
  -/
  replace hf : Function.Injective f.rangeRestrict := by simpa
  /-
    G : Type u_1
    inst✝⁴ : Group G
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : T2Space G
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom H G
    hf' : DiscreteTopology (Subtype fun x => Membership.mem f.range x)
    hf : Function.Injective ⇑f.rangeRestrict
    ⊢ Filter.Tendsto (⇑f) Filter.cofinite (Filter.cocompact G)
  -/
  exact f.range.tendsto_coe_cofinite_of_discrete.comp hf.tendsto_cofinite
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tendstoUniformly_iff (F : ι → α → G) (f : α → G) (p : Filter ι)
    (hu : TopologicalGroup.toUniformSpace G = u) :
    TendstoUniformly F f p ↔ ∀ u ∈ 𝓝 (1 : G), ∀ᶠ i in p, ∀ a, F i a / f a ∈ u :=
  hu ▸ ⟨fun h u hu => h _ ⟨u, hu, fun _ => id⟩,
    fun h _ ⟨u, hu, hv⟩ => mem_of_superset (h u hu) fun _ hi a => hv (hi a)⟩


@[to_additive]
theorem tendstoUniformlyOn_iff (F : ι → α → G) (f : α → G) (p : Filter ι) (s : Set α)
    (hu : TopologicalGroup.toUniformSpace G = u) :
    TendstoUniformlyOn F f p s ↔ ∀ u ∈ 𝓝 (1 : G), ∀ᶠ i in p, ∀ a ∈ s, F i a / f a ∈ u :=
  hu ▸ ⟨fun h u hu => h _ ⟨u, hu, fun _ => id⟩,
    fun h _ ⟨u, hu, hv⟩ => mem_of_superset (h u hu) fun _ hi a ha => hv (hi a ha)⟩


@[to_additive]
theorem tendstoLocallyUniformly_iff [TopologicalSpace α] (F : ι → α → G) (f : α → G)
    (p : Filter ι) (hu : TopologicalGroup.toUniformSpace G = u) :
    TendstoLocallyUniformly F f p ↔
      ∀ u ∈ 𝓝 (1 : G), ∀ (x : α), ∃ t ∈ 𝓝 x, ∀ᶠ i in p, ∀ a ∈ t, F i a / f a ∈ u :=
  hu ▸ ⟨fun h u hu => h _ ⟨u, hu, fun _ => id⟩, fun h _ ⟨u, hu, hv⟩ x =>
    Exists.imp (fun _ ⟨h, hp⟩ => ⟨h, mem_of_superset hp fun _ hi a ha => hv (hi a ha)⟩)
      (h u hu x)⟩


@[to_additive]
theorem tendstoLocallyUniformlyOn_iff [TopologicalSpace α] (F : ι → α → G) (f : α → G)
    (p : Filter ι) (s : Set α) (hu : TopologicalGroup.toUniformSpace G = u) :
    TendstoLocallyUniformlyOn F f p s ↔
      ∀ u ∈ 𝓝 (1 : G), ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, ∀ᶠ i in p, ∀ a ∈ t, F i a / f a ∈ u :=
  hu ▸ ⟨fun h u hu => h _ ⟨u, hu, fun _ => id⟩, fun h _ ⟨u, hu, hv⟩ x =>
    (Exists.imp fun _ ⟨h, hp⟩ => ⟨h, mem_of_superset hp fun _ hi a ha => hv (hi a ha)⟩) ∘
      h u hu x⟩


include W'_nhd in
private theorem extend_Z_bilin_aux (x₀ : α) (y₁ : δ) : ∃ U₂ ∈ comap e (𝓝 x₀), ∀ x ∈ U₂, ∀ x' ∈ U₂,
    (fun p : β × δ => φ p.1 p.2) (x' - x, y₁) ∈ W' := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  let Nx := 𝓝 x₀
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  let ee := fun u : β × β => (e u.1, e u.2)
  have lim1 : Tendsto (fun a : β × β => (a.2 - a.1, y₁))
      (comap e Nx ×ˢ comap e Nx) (𝓝 (0, y₁)) := by
    have := Tendsto.prod_mk (tendsto_sub_comap_self de x₀)
      (tendsto_const_nhds : Tendsto (fun _ : β × β => y₁) (comap ee <| 𝓝 (x₀, x₀)) (𝓝 y₁))
    rw [nhds_prod_eq, prod_comap_comap_eq, ← nhds_prod_eq]
    exact (this : _)
  have lim2 : Tendsto (fun p : β × δ => φ p.1 p.2) (𝓝 (0, y₁)) (𝓝 0) := by
    simpa using hφ.tendsto (0, y₁)
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  have lim := lim2.comp lim1
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : Filter.Tendsto (Function.comp (fun p => (φ p.1) p.2) fun a => { fst := H …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  rw [tendsto_prod_self_iff] at lim
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : ∀ (W : Set G), Membership.mem (nhds 0) W → Exists fun U => And (Membersh …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  simp_rw [forall_mem_comm]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : ∀ (W : Set G), Membership.mem (nhds 0) W → Exists fun U => And (Membersh …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (a …
  -/
  exact lim W' W'_nhd
  /-
    🎉 no goals
  -/


include df W'_nhd in
private theorem extend_Z_bilin_key (x₀ : α) (y₀ : γ) : ∃ U ∈ comap e (𝓝 x₀), ∃ V ∈ comap f (𝓝 y₀),
    ∀ x ∈ U, ∀ x' ∈ U, ∀ (y) (_ : y ∈ V) (y') (_ : y' ∈ V),
    (fun p : β × δ => φ p.1 p.2) (x', y') - (fun p : β × δ => φ p.1 p.2) (x, y) ∈ W' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  let ee := fun u : β × β => (e u.1, e u.2)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  let ff := fun u : δ × δ => (f u.1, f u.2)
  have lim_φ : Filter.Tendsto (fun p : β × δ => φ p.1 p.2) (𝓝 (0, 0)) (𝓝 0) := by
    simpa using hφ.tendsto (0, 0)
  have lim_φ_sub_sub :
    Tendsto (fun p : (β × β) × δ × δ => (fun p : β × δ => φ p.1 p.2) (p.1.2 - p.1.1, p.2.2 - p.2.1))
      ((comap ee <| 𝓝 (x₀, x₀)) ×ˢ (comap ff <| 𝓝 (y₀, y₀))) (𝓝 0) := by
    have lim_sub_sub :
      Tendsto (fun p : (β × β) × δ × δ => (p.1.2 - p.1.1, p.2.2 - p.2.1))
        (comap ee (𝓝 (x₀, x₀)) ×ˢ comap ff (𝓝 (y₀, y₀))) (𝓝 0 ×ˢ 𝓝 0) := by
      have := Filter.prod_mono (tendsto_sub_comap_self de x₀) (tendsto_sub_comap_self df y₀)
      rwa [prod_map_map_eq] at this
    rw [← nhds_prod_eq] at lim_sub_sub
    exact Tendsto.comp lim_φ lim_sub_sub
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases exists_nhds_zero_quarter W'_nhd with ⟨W, W_nhd, W4⟩
  have :
    ∃ U₁ ∈ comap e (𝓝 x₀), ∃ V₁ ∈ comap f (𝓝 y₀), ∀ (x) (_ : x ∈ U₁) (x') (_ : x' ∈ U₁),
      ∀ (y) (_ : y ∈ V₁) (y') (_ : y' ∈ V₁), (fun p : β × δ => φ p.1 p.2) (x' - x, y' - y) ∈ W := by
    rcases tendsto_prod_iff.1 lim_φ_sub_sub W W_nhd with ⟨U, U_in, V, V_in, H⟩
    rw [nhds_prod_eq, ← prod_comap_comap_eq, mem_prod_same_iff] at U_in V_in
    rcases U_in with ⟨U₁, U₁_in, HU₁⟩
    rcases V_in with ⟨V₁, V₁_in, HV₁⟩
    exists U₁, U₁_in, V₁, V₁_in
    intro x x_in x' x'_in y y_in y' y'_in
    exact H _ _ (HU₁ (mk_mem_prod x_in x'_in)) (HV₁ (mk_mem_prod y_in y'_in))
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    this : Exists fun U₁ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁)  …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases this with ⟨U₁, U₁_nhd, V₁, V₁_nhd, H⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  obtain ⟨x₁, x₁_in⟩ : U₁.Nonempty := (de.comap_nhds_neBot _).nonempty_of_mem U₁_nhd
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  obtain ⟨y₁, y₁_in⟩ : V₁.Nonempty := (df.comap_nhds_neBot _).nonempty_of_mem V₁_nhd
  have cont_flip : Continuous fun p : δ × β => φ.flip p.1 p.2 := by
    show Continuous ((fun p : β × δ => φ p.1 p.2) ∘ Prod.swap)
    exact hφ.comp continuous_swap
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases extend_Z_bilin_aux de hφ W_nhd x₀ y₁ with ⟨U₂, U₂_nhd, HU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases extend_Z_bilin_aux df cont_flip W_nhd y₀ x₁ with ⟨V₂, V₂_nhd, HV⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  exists U₁ ∩ U₂, inter_mem U₁_nhd U₂_nhd, V₁ ∩ V₂, inter_mem V₁_nhd V₂_nhd
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    ⊢ ∀ (x : β), Membership.mem (Inter.inter U₁ U₂) x → ∀ (x' : β), Membership.mem …
  -/
  rintro x ⟨xU₁, xU₂⟩ x' ⟨x'U₁, x'U₂⟩ y ⟨yV₁, yV₂⟩ y' ⟨y'V₁, y'V₂⟩
  have key_formula : φ x' y' - φ x y
    = φ (x' - x) y₁ + φ (x' - x) (y' - y₁) + φ x₁ (y' - y) + φ (x - x₁) (y' - y) := by simp; abel
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    ⊢ Membership.mem W' (HSub.hSub ((fun p => (φ p.1) p.2) { fst := x', snd := y'  …
  -/
  rw [key_formula]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₁ := HU x xU₂ x' x'U₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₂ := H x xU₁ x' x'U₁ y₁ y₁_in y' y'V₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₃ := HV y yV₂ y' y'V₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₃ : Membership.mem W ((fun p => (φ.flip p.1) p.2) { fst := HSub.hSub y' y, sn …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₄ := H x₁ x₁_in x xU₁ y yV₁ y' y'V₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₃ : Membership.mem W ((fun p => (φ.flip p.1) p.2) { fst := HSub.hSub y' y, sn …
    h₄ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x x₁, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  exact W4 h₁ h₂ h₃ h₄
  /-
    🎉 no goals
  -/


/-- Bourbaki GT III.6.5 Theorem I:
ℤ-bilinear continuous maps from dense images into a complete Hausdorff group extend by continuity.
Note: Bourbaki assumes that α and β are also complete Hausdorff, but this is not necessary. -/
theorem extend_Z_bilin : Continuous (extend (de.prodMap df) (fun p : β × δ => φ p.1 p.2)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹⁴ : TopologicalSpace α
    inst✝¹³ : AddCommGroup α
    inst✝¹² : TopologicalAddGroup α
    inst✝¹¹ : TopologicalSpace β
    inst✝¹⁰ : AddCommGroup β
    inst✝⁹ : TopologicalSpace γ
    inst✝⁸ : AddCommGroup γ
    inst✝⁷ : TopologicalAddGroup γ
    inst✝⁶ : TopologicalSpace δ
    inst✝⁵ : AddCommGroup δ
    inst✝⁴ : UniformSpace G
    inst✝³ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    inst✝² : UniformAddGroup G
    inst✝¹ : T0Space G
    inst✝ : CompleteSpace G
    ⊢ Continuous (⋯.extend fun p => (φ p.1) p.2)
  -/
  refine continuous_extend_of_cauchy _ ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹⁴ : TopologicalSpace α
    inst✝¹³ : AddCommGroup α
    inst✝¹² : TopologicalAddGroup α
    inst✝¹¹ : TopologicalSpace β
    inst✝¹⁰ : AddCommGroup β
    inst✝⁹ : TopologicalSpace γ
    inst✝⁸ : AddCommGroup γ
    inst✝⁷ : TopologicalAddGroup γ
    inst✝⁶ : TopologicalSpace δ
    inst✝⁵ : AddCommGroup δ
    inst✝⁴ : UniformSpace G
    inst✝³ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    inst✝² : UniformAddGroup G
    inst✝¹ : T0Space G
    inst✝ : CompleteSpace G
    ⊢ ∀ (b : Prod α γ), Cauchy (Filter.map (fun p => (φ p.1) p.2) (Filter.comap (P …
  -/
  rintro ⟨x₀, y₀⟩
  /-
    case mk
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹⁴ : TopologicalSpace α
    inst✝¹³ : AddCommGroup α
    inst✝¹² : TopologicalAddGroup α
    inst✝¹¹ : TopologicalSpace β
    inst✝¹⁰ : AddCommGroup β
    inst✝⁹ : TopologicalSpace γ
    inst✝⁸ : AddCommGroup γ
    inst✝⁷ : TopologicalAddGroup γ
    inst✝⁶ : TopologicalSpace δ
    inst✝⁵ : AddCommGroup δ
    inst✝⁴ : UniformSpace G
    inst✝³ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    inst✝² : UniformAddGroup G
    inst✝¹ : T0Space G
    inst✝ : CompleteSpace G
    x₀ : α
    y₀ : γ
    ⊢ Cauchy (Filter.map (fun p => (φ p.1) p.2) (Filter.comap (Prod.map ⇑e ⇑f) (nh …
  -/
  constructor
    /-
      case mk.left
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      ⊢ (Filter.map (fun p => (φ p.1) p.2) (Filter.comap (Prod.map ⇑e ⇑f) (nhds { fs …
    -/
  · apply NeBot.map
    /-
      case mk.left.hf
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      ⊢ (Filter.comap (Prod.map ⇑e ⇑f) (nhds { fst := x₀, snd := y₀ })).NeBot
    -/
    apply comap_neBot
    /-
      case mk.left.hf.hm
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      ⊢ ∀ (t : Set (Prod α γ)), Membership.mem (nhds { fst := x₀, snd := y₀ }) t → E …
    -/
    intro U h
    /-
      case mk.left.hf.hm
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      U : Set (Prod α γ)
      h : Membership.mem (nhds { fst := x₀, snd := y₀ }) U
      ⊢ Exists fun a => Membership.mem U (Prod.map (⇑e) (⇑f) a)
    -/
    rcases mem_closure_iff_nhds.1 ((de.prodMap df).dense (x₀, y₀)) U h with ⟨x, x_in, ⟨z, z_x⟩⟩
    /-
      case mk.left.hf.hm.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      U : Set (Prod α γ)
      h : Membership.mem (nhds { fst := x₀, snd := y₀ }) U
      x : Prod α γ
      x_in : Membership.mem U x
      z : Prod β δ
      z_x : Eq (Prod.map (⇑e) (⇑f) z) x
      ⊢ Exists fun a => Membership.mem U (Prod.map (⇑e) (⇑f) a)
    -/
    exists z
    /-
      case mk.left.hf.hm.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      U : Set (Prod α γ)
      h : Membership.mem (nhds { fst := x₀, snd := y₀ }) U
      x : Prod α γ
      x_in : Membership.mem U x
      z : Prod β δ
      z_x : Eq (Prod.map (⇑e) (⇑f) z) x
      ⊢ Membership.mem U (Prod.map (⇑e) (⇑f) z)
    -/
    aesop
    /-
      🎉 no goals
    -/
  · suffices map (fun p : (β × δ) × β × δ => (fun p : β × δ => φ p.1 p.2) p.2 -
      (fun p : β × δ => φ p.1 p.2) p.1)
        (comap (fun p : (β × δ) × β × δ => ((e p.1.1, f p.1.2), (e p.2.1, f p.2.2)))
        (𝓝 (x₀, y₀) ×ˢ 𝓝 (x₀, y₀))) ≤ 𝓝 0 by
      rwa [uniformity_eq_comap_nhds_zero G, prod_map_map_eq, ← map_le_iff_le_comap, Filter.map_map,
        prod_comap_comap_eq]
    /-
      case mk.right
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      ⊢ LE.le (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2) ((fun p = …
    -/
    intro W' W'_nhd
    /-
      case mk.right
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    have key := extend_Z_bilin_key de df hφ W'_nhd x₀ y₀
    /-
      case mk.right
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      key : Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Ex …
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rcases key with ⟨U, U_nhd, V, V_nhd, h⟩
    /-
      case mk.right.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      U_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U
      V : Set δ
      V_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rw [mem_comap] at U_nhd
    /-
      case mk.right.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      U_nhd : Exists fun t => And (Membership.mem (nhds x₀) t) (HasSubset.Subset (Se …
      V : Set δ
      V_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rcases U_nhd with ⟨U', U'_nhd, U'_sub⟩
    /-
      case mk.right.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      V_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rw [mem_comap] at V_nhd
    /-
      case mk.right.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      V_nhd : Exists fun t => And (Membership.mem (nhds y₀) t) (HasSubset.Subset (Se …
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rcases V_nhd with ⟨V', V'_nhd, V'_sub⟩
    /-
      case mk.right.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      V' : Set γ
      V'_nhd : Membership.mem (nhds y₀) V'
      V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
      ⊢ Membership.mem (Filter.map (fun p => HSub.hSub ((fun p => (φ p.1) p.2) p.2)  …
    -/
    rw [mem_map, mem_comap, nhds_prod_eq]
    /-
      case mk.right.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      V' : Set γ
      V'_nhd : Membership.mem (nhds y₀) V'
      V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
      ⊢ Exists fun t => And (Membership.mem (SProd.sprod (SProd.sprod (nhds x₀) (nhd …
    -/
    exists (U' ×ˢ V') ×ˢ U' ×ˢ V'
    /-
      case mk.right.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      V' : Set γ
      V'_nhd : Membership.mem (nhds y₀) V'
      V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
      ⊢ And (Membership.mem (SProd.sprod (SProd.sprod (nhds x₀) (nhds y₀)) (SProd.sp …
    -/
    rw [mem_prod_same_iff]
    /-
      case mk.right.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      V' : Set γ
      V'_nhd : Membership.mem (nhds y₀) V'
      V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
      ⊢ And (Exists fun t => And (Membership.mem (SProd.sprod (nhds x₀) (nhds y₀)) t …
    -/
    simp only [exists_prop]
    /-
      case mk.right.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      G : Type u_5
      inst✝¹⁴ : TopologicalSpace α
      inst✝¹³ : AddCommGroup α
      inst✝¹² : TopologicalAddGroup α
      inst✝¹¹ : TopologicalSpace β
      inst✝¹⁰ : AddCommGroup β
      inst✝⁹ : TopologicalSpace γ
      inst✝⁸ : AddCommGroup γ
      inst✝⁷ : TopologicalAddGroup γ
      inst✝⁶ : TopologicalSpace δ
      inst✝⁵ : AddCommGroup δ
      inst✝⁴ : UniformSpace G
      inst✝³ : AddCommGroup G
      e : AddMonoidHom β α
      de : IsDenseInducing ⇑e
      f : AddMonoidHom δ γ
      df : IsDenseInducing ⇑f
      φ : AddMonoidHom β (AddMonoidHom δ G)
      hφ : Continuous fun p => (φ p.1) p.2
      inst✝² : UniformAddGroup G
      inst✝¹ : T0Space G
      inst✝ : CompleteSpace G
      x₀ : α
      y₀ : γ
      W' : Set G
      W'_nhd : Membership.mem (nhds 0) W'
      U : Set β
      V : Set δ
      h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
      U' : Set α
      U'_nhd : Membership.mem (nhds x₀) U'
      U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
      V' : Set γ
      V'_nhd : Membership.mem (nhds y₀) V'
      V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
      ⊢ And (Exists fun t => And (Membership.mem (SProd.sprod (nhds x₀) (nhds y₀)) t …
    -/
    constructor
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        ⊢ Exists fun t => And (Membership.mem (SProd.sprod (nhds x₀) (nhds y₀)) t) (Ha …
      -/
    · have := prod_mem_prod U'_nhd V'_nhd
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        this : Membership.mem (SProd.sprod (nhds x₀) (nhds y₀)) (SProd.sprod U' V')
        ⊢ Exists fun t => And (Membership.mem (SProd.sprod (nhds x₀) (nhds y₀)) t) (Ha …
      -/
      tauto
      /-
        🎉 no goals
      -/
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        ⊢ HasSubset.Subset (Set.preimage (fun p => { fst := { fst := e p.1.1, snd := f …
      -/
    · intro p h'
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        p : Prod (Prod β δ) (Prod β δ)
        h' : Membership.mem (Set.preimage (fun p => { fst := { fst := e p.1.1, snd :=  …
        ⊢ Membership.mem (Set.preimage (fun p => HSub.hSub ((φ p.2.1) p.2.2) ((φ p.1.1 …
      -/
      simp only [Set.mem_preimage, Set.prod_mk_mem_set_prod_eq] at h'
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.right
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        p : Prod (Prod β δ) (Prod β δ)
        h' : And (And (Membership.mem U' (e p.1.1)) (Membership.mem V' (f p.1.2))) (An …
        ⊢ Membership.mem (Set.preimage (fun p => HSub.hSub ((φ p.2.1) p.2.2) ((φ p.1.1 …
      -/
      rcases p with ⟨⟨x, y⟩, ⟨x', y'⟩⟩
      /-
        case mk.right.intro.intro.intro.intro.intro.intro.intro.intro.right.mk.mk.mk
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        G : Type u_5
        inst✝¹⁴ : TopologicalSpace α
        inst✝¹³ : AddCommGroup α
        inst✝¹² : TopologicalAddGroup α
        inst✝¹¹ : TopologicalSpace β
        inst✝¹⁰ : AddCommGroup β
        inst✝⁹ : TopologicalSpace γ
        inst✝⁸ : AddCommGroup γ
        inst✝⁷ : TopologicalAddGroup γ
        inst✝⁶ : TopologicalSpace δ
        inst✝⁵ : AddCommGroup δ
        inst✝⁴ : UniformSpace G
        inst✝³ : AddCommGroup G
        e : AddMonoidHom β α
        de : IsDenseInducing ⇑e
        f : AddMonoidHom δ γ
        df : IsDenseInducing ⇑f
        φ : AddMonoidHom β (AddMonoidHom δ G)
        hφ : Continuous fun p => (φ p.1) p.2
        inst✝² : UniformAddGroup G
        inst✝¹ : T0Space G
        inst✝ : CompleteSpace G
        x₀ : α
        y₀ : γ
        W' : Set G
        W'_nhd : Membership.mem (nhds 0) W'
        U : Set β
        V : Set δ
        h : ∀ (x : β), Membership.mem U x → ∀ (x' : β), Membership.mem U x' → ∀ (y : δ …
        U' : Set α
        U'_nhd : Membership.mem (nhds x₀) U'
        U'_sub : HasSubset.Subset (Set.preimage (⇑e) U') U
        V' : Set γ
        V'_nhd : Membership.mem (nhds y₀) V'
        V'_sub : HasSubset.Subset (Set.preimage (⇑f) V') V
        x : β
        y : δ
        x' : β
        y' : δ
        h' : And (And (Membership.mem U' (e { fst := { fst := x, snd := y }, snd := {  …
        ⊢ Membership.mem (Set.preimage (fun p => HSub.hSub ((φ p.2.1) p.2.2) ((φ p.1.1 …
      -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
      apply h <;> tauto
                  /-
                    🎉 no goals
                  -/


open Classical in
/-- The quotient `G ⧸ N` of a complete first countable topological group `G` by a normal subgroup
is itself complete. [N. Bourbaki, *General Topology*, IX.3.1 Proposition 4][bourbaki1966b]

Because a topological group is not equipped with a `UniformSpace` instance by default, we must
explicitly provide it in order to consider completeness. See `QuotientGroup.completeSpace` for a
version in which `G` is already equipped with a uniform structure. -/
@[to_additive "The quotient `G ⧸ N` of a complete first countable topological additive group
`G` by a normal additive subgroup is itself complete. Consequently, quotients of Banach spaces by
subspaces are complete. [N. Bourbaki, *General Topology*, IX.3.1 Proposition 4][bourbaki1966b]

Because an additive topological group is not equipped with a `UniformSpace` instance by default,
we must explicitly provide it in order to consider completeness. See
`QuotientAddGroup.completeSpace` for a version in which `G` is already equipped with a uniform
structure."]
instance QuotientGroup.completeSpace' (G : Type u) [Group G] [TopologicalSpace G]
    [TopologicalGroup G] [FirstCountableTopology G] (N : Subgroup G) [N.Normal]
    [@CompleteSpace G (TopologicalGroup.toUniformSpace G)] :
    @CompleteSpace (G ⧸ N) (TopologicalGroup.toUniformSpace (G ⧸ N)) := by
  /- Since `G ⧸ N` is a topological group it is a uniform space, and since `G` is first countable
    the uniformities of both `G` and `G ⧸ N` are countably generated. Moreover, we may choose a
    sequential antitone neighborhood basis `u` for `𝓝 (1 : G)` so that `(u (n + 1)) ^ 2 ⊆ u n`, and
    this descends to an antitone neighborhood basis `v` for `𝓝 (1 : G ⧸ N)`. Since `𝓤 (G ⧸ N)` is
    countably generated, it suffices to show any Cauchy sequence `x` converges. -/
  /-
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  letI : UniformSpace (G ⧸ N) := TopologicalGroup.toUniformSpace (G ⧸ N)
  /-
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniformSp …
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  letI : UniformSpace G := TopologicalGroup.toUniformSpace G
  /-
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniformS …
    this : UniformSpace G := TopologicalGroup.toUniformSpace G
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  haveI : (𝓤 (G ⧸ N)).IsCountablyGenerated := comap.isCountablyGenerated _ _
  /-
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  obtain ⟨u, hu, u_mul⟩ := TopologicalGroup.exists_antitone_basis_nhds_one G
  /-
    case intro.intro
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  obtain ⟨hv, v_anti⟩ := hu.map ((↑) : G → G ⧸ N)
  /-
    case intro.intro.mk
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (Filter.map QuotientGroup.mk (nhds 1)).HasBasis (fun x => True) fun x =>  …
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  rw [← QuotientGroup.nhds_eq N 1, QuotientGroup.mk_one] at hv
  /-
    case intro.intro.mk
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (nhds 1).HasBasis (fun x => True) fun x => Set.image QuotientGroup.mk (u x)
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  refine UniformSpace.complete_of_cauchySeq_tendsto fun x hx => ?_
  /- Given `n : ℕ`, for sufficiently large `a b : ℕ`, given any lift of `x b`, we can find a lift
    of `x a` such that the quotient of the lifts lies in `u n`. -/
  have key₀ : ∀ i j : ℕ, ∃ M : ℕ, j < M ∧ ∀ a b : ℕ, M ≤ a → M ≤ b →
      ∀ g : G, x b = g → ∃ g' : G, g / g' ∈ u i ∧ x a = g' := by
    have h𝓤GN : (𝓤 (G ⧸ N)).HasBasis (fun _ ↦ True) fun i ↦ { x | x.snd / x.fst ∈ (↑) '' u i } := by
      simpa [uniformity_eq_comap_nhds_one'] using hv.comap _
    rw [h𝓤GN.cauchySeq_iff] at hx
    simp only [mem_setOf_eq, forall_true_left, mem_image] at hx
    intro i j
    rcases hx i with ⟨M, hM⟩
    refine ⟨max j M + 1, (le_max_left _ _).trans_lt (lt_add_one _), fun a b ha hb g hg => ?_⟩
    obtain ⟨y, y_mem, hy⟩ :=
      hM a (((le_max_right j _).trans (lt_add_one _).le).trans ha) b
        (((le_max_right j _).trans (lt_add_one _).le).trans hb)
    refine
      ⟨y⁻¹ * g, by
        simpa only [div_eq_mul_inv, mul_inv_rev, inv_inv, mul_inv_cancel_left] using y_mem, ?_⟩
    rw [QuotientGroup.mk_mul, QuotientGroup.mk_inv, hy, hg, inv_div, div_mul_cancel]
  /- Inductively construct a subsequence `φ : ℕ → ℕ` using `key₀` so that if `a b : ℕ` exceed
    `φ (n + 1)`, then we may find lifts whose quotients lie within `u n`. -/
  /-
    case intro.intro.mk
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (nhds 1).HasBasis (fun x => True) fun x => Set.image QuotientGroup.mk (u x)
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    x : Nat → HasQuotient.Quotient G N
    hx : CauchySeq x
    key₀ : ∀ (i j : Nat), Exists fun M => And (LT.lt j M) (∀ (a b : Nat), LE.le M  …
    ⊢ Exists fun a => Filter.Tendsto x Filter.atTop (nhds a)
  -/
  set φ : ℕ → ℕ := fun n => Nat.recOn n (choose <| key₀ 0 0) fun k yk => choose <| key₀ (k + 1) yk
  have hφ :
    ∀ n : ℕ,
      φ n < φ (n + 1) ∧
        ∀ a b : ℕ,
          φ (n + 1) ≤ a →
            φ (n + 1) ≤ b → ∀ g : G, x b = g → ∃ g' : G, g / g' ∈ u (n + 1) ∧ x a = g' :=
    fun n => choose_spec (key₀ (n + 1) (φ n))
  /- Inductively construct a sequence `x' n : G` of lifts of `x (φ (n + 1))` such that quotients of
    successive terms lie in `x' n / x' (n + 1) ∈ u (n + 1)`. We actually need the proofs that each
    term is a lift to construct the next term, so we use a Σ-type. -/
  set x' : ∀ n, PSigma fun g : G => x (φ (n + 1)) = g := fun n =>
    Nat.recOn n
      ⟨choose (QuotientGroup.mk_surjective (x (φ 1))),
        (choose_spec (QuotientGroup.mk_surjective (x (φ 1)))).symm⟩
      fun k hk =>
      ⟨choose <| (hφ k).2 _ _ (hφ (k + 1)).1.le le_rfl hk.fst hk.snd,
        (choose_spec <| (hφ k).2 _ _ (hφ (k + 1)).1.le le_rfl hk.fst hk.snd).2⟩
  have hx' : ∀ n : ℕ, (x' n).fst / (x' (n + 1)).fst ∈ u (n + 1) := fun n =>
    (choose_spec <| (hφ n).2 _ _ (hφ (n + 1)).1.le le_rfl (x' n).fst (x' n).snd).1
  /- The sequence `x'` is Cauchy. This is where we exploit the condition on `u`. The key idea
    is to show by decreasing induction that `x' m / x' n ∈ u m` if `m ≤ n`. -/
  have x'_cauchy : CauchySeq fun n => (x' n).fst := by
    have h𝓤G : (𝓤 G).HasBasis (fun _ => True) fun i => { x | x.snd / x.fst ∈ u i } := by
      simpa [uniformity_eq_comap_nhds_one'] using hu.toHasBasis.comap _
    rw [h𝓤G.cauchySeq_iff']
    simp only [mem_setOf_eq, forall_true_left]
    exact fun m =>
      ⟨m, fun n hmn =>
        Nat.decreasingInduction'
          (fun k _ _ hk => u_mul k ⟨_, hx' k, _, hk, div_mul_div_cancel _ _ _⟩) hmn
          (by simpa only [div_self'] using mem_of_mem_nhds (hu.mem _))⟩
  /- Since `G` is complete, `x'` converges to some `x₀`, and so the image of this sequence under
    the quotient map converges to `↑x₀`. The image of `x'` is a convergent subsequence of `x`, and
    since `x` is Cauchy, this implies it converges. -/
  /-
    case intro.intro.mk
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (nhds 1).HasBasis (fun x => True) fun x => Set.image QuotientGroup.mk (u x)
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    x : Nat → HasQuotient.Quotient G N
    hx : CauchySeq x
    key₀ : ∀ (i j : Nat), Exists fun M => And (LT.lt j M) (∀ (a b : Nat), LE.le M  …
    φ : Nat → Nat := fun n => Nat.recOn n (Classical.choose ⋯) fun k yk => Classic …
    hφ : ∀ (n : Nat), And (LT.lt (φ n) (φ (HAdd.hAdd n 1))) (∀ (a b : Nat), LE.le  …
    x' : (n : Nat) → PSigma fun g => Eq (x (φ (HAdd.hAdd n 1))) ↑g := fun n => Nat …
    hx' : ∀ (n : Nat), Membership.mem (u (HAdd.hAdd n 1)) (HDiv.hDiv (x' n).fst (x …
    x'_cauchy : CauchySeq fun n => (x' n).fst
    ⊢ Exists fun a => Filter.Tendsto x Filter.atTop (nhds a)
  -/
  rcases cauchySeq_tendsto_of_complete x'_cauchy with ⟨x₀, hx₀⟩
  refine
    ⟨↑x₀,
      tendsto_nhds_of_cauchySeq_of_subseq hx
        (strictMono_nat_of_lt_succ fun n => (hφ (n + 1)).1).tendsto_atTop ?_⟩
  /-
    case intro.intro.mk.intro
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (nhds 1).HasBasis (fun x => True) fun x => Set.image QuotientGroup.mk (u x)
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    x : Nat → HasQuotient.Quotient G N
    hx : CauchySeq x
    key₀ : ∀ (i j : Nat), Exists fun M => And (LT.lt j M) (∀ (a b : Nat), LE.le M  …
    φ : Nat → Nat := fun n => Nat.recOn n (Classical.choose ⋯) fun k yk => Classic …
    hφ : ∀ (n : Nat), And (LT.lt (φ n) (φ (HAdd.hAdd n 1))) (∀ (a b : Nat), LE.le  …
    x' : (n : Nat) → PSigma fun g => Eq (x (φ (HAdd.hAdd n 1))) ↑g := fun n => Nat …
    hx' : ∀ (n : Nat), Membership.mem (u (HAdd.hAdd n 1)) (HDiv.hDiv (x' n).fst (x …
    x'_cauchy : CauchySeq fun n => (x' n).fst
    x₀ : G
    hx₀ : Filter.Tendsto (fun n => (x' n).fst) Filter.atTop (nhds x₀)
    ⊢ Filter.Tendsto (Function.comp x fun n => φ (HAdd.hAdd n 1)) Filter.atTop (nh …
  -/
  convert ((continuous_coinduced_rng : Continuous ((↑) : G → G ⧸ N)).tendsto x₀).comp hx₀
  /-
    case h.e'_3
    G : Type u
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : FirstCountableTopology G
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : CompleteSpace G
    this✝¹ : UniformSpace (HasQuotient.Quotient G N) := TopologicalGroup.toUniform …
    this✝ : UniformSpace G := TopologicalGroup.toUniformSpace G
    this : (uniformity (HasQuotient.Quotient G N)).IsCountablyGenerated
    u : Nat → Set G
    hu : (nhds 1).HasAntitoneBasis u
    u_mul : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (u (HAdd.hAdd n 1)) (u (HAdd. …
    hv : (nhds 1).HasBasis (fun x => True) fun x => Set.image QuotientGroup.mk (u x)
    v_anti : Antitone fun x => Set.image QuotientGroup.mk (u x)
    x : Nat → HasQuotient.Quotient G N
    hx : CauchySeq x
    key₀ : ∀ (i j : Nat), Exists fun M => And (LT.lt j M) (∀ (a b : Nat), LE.le M  …
    φ : Nat → Nat := fun n => Nat.recOn n (Classical.choose ⋯) fun k yk => Classic …
    hφ : ∀ (n : Nat), And (LT.lt (φ n) (φ (HAdd.hAdd n 1))) (∀ (a b : Nat), LE.le  …
    x' : (n : Nat) → PSigma fun g => Eq (x (φ (HAdd.hAdd n 1))) ↑g := fun n => Nat …
    hx' : ∀ (n : Nat), Membership.mem (u (HAdd.hAdd n 1)) (HDiv.hDiv (x' n).fst (x …
    x'_cauchy : CauchySeq fun n => (x' n).fst
    x₀ : G
    hx₀ : Filter.Tendsto (fun n => (x' n).fst) Filter.atTop (nhds x₀)
    ⊢ Eq (Function.comp x fun n => φ (HAdd.hAdd n 1)) (Function.comp QuotientGroup …
  -/
  exact funext fun n => (x' n).snd
  /-
    🎉 no goals
  -/


/-- The quotient `G ⧸ N` of a complete first countable uniform group `G` by a normal subgroup
is itself complete. In contrast to `QuotientGroup.completeSpace'`, in this version `G` is
already equipped with a uniform structure.
[N. Bourbaki, *General Topology*, IX.3.1 Proposition 4][bourbaki1966b]

Even though `G` is equipped with a uniform structure, the quotient `G ⧸ N` does not inherit a
uniform structure, so it is still provided manually via `TopologicalGroup.toUniformSpace`.
In the most common use cases, this coincides (definitionally) with the uniform structure on the
quotient obtained via other means. -/
@[to_additive "The quotient `G ⧸ N` of a complete first countable uniform additive group
`G` by a normal additive subgroup is itself complete. Consequently, quotients of Banach spaces by
subspaces are complete. In contrast to `QuotientAddGroup.completeSpace'`, in this version
`G` is already equipped with a uniform structure.
[N. Bourbaki, *General Topology*, IX.3.1 Proposition 4][bourbaki1966b]

Even though `G` is equipped with a uniform structure, the quotient `G ⧸ N` does not inherit a
uniform structure, so it is still provided manually via `TopologicalAddGroup.toUniformSpace`.
In the most common use case ─ quotients of normed additive commutative groups by subgroups ─
significant care was taken so that the uniform structure inherent in that setting coincides
(definitionally) with the uniform structure provided here."]
instance QuotientGroup.completeSpace (G : Type u) [Group G] [us : UniformSpace G] [UniformGroup G]
    [FirstCountableTopology G] (N : Subgroup G) [N.Normal] [hG : CompleteSpace G] :
    @CompleteSpace (G ⧸ N) (TopologicalGroup.toUniformSpace (G ⧸ N)) := by
  /-
    G : Type u
    inst✝³ : Group G
    us : UniformSpace G
    inst✝² : UniformGroup G
    inst✝¹ : FirstCountableTopology G
    N : Subgroup G
    inst✝ : N.Normal
    hG : CompleteSpace G
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  rw [← @UniformGroup.toUniformSpace_eq _ us _ _] at hG
  /-
    G : Type u
    inst✝³ : Group G
    us : UniformSpace G
    inst✝² : UniformGroup G
    inst✝¹ : FirstCountableTopology G
    N : Subgroup G
    inst✝ : N.Normal
    hG : CompleteSpace G
    ⊢ CompleteSpace (HasQuotient.Quotient G N)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


