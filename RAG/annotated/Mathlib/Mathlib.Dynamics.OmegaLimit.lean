/-- The ω-limit of a set `s` under `ϕ` with respect to a filter `f` is `⋂ u ∈ f, cl (ϕ u s)`. -/
def omegaLimit [TopologicalSpace β] (f : Filter τ) (ϕ : τ → α → β) (s : Set α) : Set β :=
  ⋂ u ∈ f, closure (image2 ϕ u s)


@[inherit_doc]
scoped[omegaLimit] notation "ω" => omegaLimit


/-- The ω-limit w.r.t. `Filter.atTop`. -/
scoped[omegaLimit] notation "ω⁺" => omegaLimit Filter.atTop


/-- The ω-limit w.r.t. `Filter.atBot`. -/
scoped[omegaLimit] notation "ω⁻" => omegaLimit Filter.atBot


theorem omegaLimit_def : ω f ϕ s = ⋂ u ∈ f, closure (image2 ϕ u s) := rfl


theorem omegaLimit_subset_of_tendsto {m : τ → τ} {f₁ f₂ : Filter τ} (hf : Tendsto m f₁ f₂) :
    ω f₁ (fun t x ↦ ϕ (m t) x) s ⊆ ω f₂ ϕ s := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    ϕ : τ → α → β
    s : Set α
    m : τ → τ
    f₁ f₂ : Filter τ
    hf : Filter.Tendsto m f₁ f₂
    ⊢ HasSubset.Subset (omegaLimit f₁ (fun t x => ϕ (m t) x) s) (omegaLimit f₂ ϕ s)
  -/
  refine iInter₂_mono' fun u hu ↦ ⟨m ⁻¹' u, tendsto_def.mp hf _ hu, ?_⟩
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    ϕ : τ → α → β
    s : Set α
    m : τ → τ
    f₁ f₂ : Filter τ
    hf : Filter.Tendsto m f₁ f₂
    u : Set τ
    hu : Membership.mem f₂ u
    ⊢ HasSubset.Subset (closure (Set.image2 (fun t x => ϕ (m t) x) (Set.preimage m …
  -/
  rw [← image2_image_left]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    ϕ : τ → α → β
    s : Set α
    m : τ → τ
    f₁ f₂ : Filter τ
    hf : Filter.Tendsto m f₁ f₂
    u : Set τ
    hu : Membership.mem f₂ u
    ⊢ HasSubset.Subset (closure (Set.image2 ϕ (Set.image m (Set.preimage m u)) s)) …
  -/
  exact closure_mono (image2_subset (image_preimage_subset _ _) Subset.rfl)
  /-
    🎉 no goals
  -/


theorem omegaLimit_mono_left {f₁ f₂ : Filter τ} (hf : f₁ ≤ f₂) : ω f₁ ϕ s ⊆ ω f₂ ϕ s :=
  omegaLimit_subset_of_tendsto ϕ s (tendsto_id'.2 hf)


theorem omegaLimit_mono_right {s₁ s₂ : Set α} (hs : s₁ ⊆ s₂) : ω f ϕ s₁ ⊆ ω f ϕ s₂ :=
  iInter₂_mono fun _u _hu ↦ closure_mono (image2_subset Subset.rfl hs)


theorem isClosed_omegaLimit : IsClosed (ω f ϕ s) :=
  isClosed_iInter fun _u ↦ isClosed_iInter fun _hu ↦ isClosed_closure


theorem mapsTo_omegaLimit' {α' β' : Type*} [TopologicalSpace β'] {f : Filter τ} {ϕ : τ → α → β}
    {ϕ' : τ → α' → β'} {ga : α → α'} {s' : Set α'} (hs : MapsTo ga s s') {gb : β → β'}
    (hg : ∀ᶠ t in f, EqOn (gb ∘ ϕ t) (ϕ' t ∘ ga) s) (hgc : Continuous gb) :
    MapsTo gb (ω f ϕ s) (ω f ϕ' s') := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    s : Set α
    α' : Type u_5
    β' : Type u_6
    inst✝ : TopologicalSpace β'
    f : Filter τ
    ϕ : τ → α → β
    ϕ' : τ → α' → β'
    ga : α → α'
    s' : Set α'
    hs : Set.MapsTo ga s s'
    gb : β → β'
    hg : Filter.Eventually (fun t => Set.EqOn (Function.comp gb (ϕ t)) (Function.c …
    hgc : Continuous gb
    ⊢ Set.MapsTo gb (omegaLimit f ϕ s) (omegaLimit f ϕ' s')
  -/
  simp only [omegaLimit_def, mem_iInter, MapsTo]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    s : Set α
    α' : Type u_5
    β' : Type u_6
    inst✝ : TopologicalSpace β'
    f : Filter τ
    ϕ : τ → α → β
    ϕ' : τ → α' → β'
    ga : α → α'
    s' : Set α'
    hs : Set.MapsTo ga s s'
    gb : β → β'
    hg : Filter.Eventually (fun t => Set.EqOn (Function.comp gb (ϕ t)) (Function.c …
    hgc : Continuous gb
    ⊢ ∀ ⦃x : β⦄, (∀ (i : Set τ), Membership.mem f i → Membership.mem (closure (Set …
  -/
  intro y hy u hu
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    s : Set α
    α' : Type u_5
    β' : Type u_6
    inst✝ : TopologicalSpace β'
    f : Filter τ
    ϕ : τ → α → β
    ϕ' : τ → α' → β'
    ga : α → α'
    s' : Set α'
    hs : Set.MapsTo ga s s'
    gb : β → β'
    hg : Filter.Eventually (fun t => Set.EqOn (Function.comp gb (ϕ t)) (Function.c …
    hgc : Continuous gb
    y : β
    hy : ∀ (i : Set τ), Membership.mem f i → Membership.mem (closure (Set.image2 ϕ …
    u : Set τ
    hu : Membership.mem f u
    ⊢ Membership.mem (closure (Set.image2 ϕ' u s')) (gb y)
  -/
  refine map_mem_closure hgc (hy _ (inter_mem hu hg)) (forall_mem_image2.2 fun t ht x hx ↦ ?_)
  calc
    ϕ' t (ga x) ∈ image2 ϕ' u s' := mem_image2_of_mem ht.1 (hs hx)
    _ = gb (ϕ t x) := ht.2 hx |>.symm


theorem mapsTo_omegaLimit {α' β' : Type*} [TopologicalSpace β'] {f : Filter τ} {ϕ : τ → α → β}
    {ϕ' : τ → α' → β'} {ga : α → α'} {s' : Set α'} (hs : MapsTo ga s s') {gb : β → β'}
    (hg : ∀ t x, gb (ϕ t x) = ϕ' t (ga x)) (hgc : Continuous gb) :
    MapsTo gb (ω f ϕ s) (ω f ϕ' s') :=
  mapsTo_omegaLimit' _ hs (Eventually.of_forall fun t x _hx ↦ hg t x) hgc


theorem omegaLimit_image_eq {α' : Type*} (ϕ : τ → α' → β) (f : Filter τ) (g : α → α') :
                                                       /-
                                                         τ : Type u_1
                                                         α : Type u_2
                                                         β : Type u_3
                                                         inst✝ : TopologicalSpace β
                                                         s : Set α
                                                         α' : Type u_5
                                                         ϕ : τ → α' → β
                                                         f : Filter τ
                                                         g : α → α'
                                                         ⊢ Eq (omegaLimit f ϕ (Set.image g s)) (omegaLimit f (fun t x => ϕ t (g x)) s)
                                                       -/
    ω f ϕ (g '' s) = ω f (fun t x ↦ ϕ t (g x)) s := by simp only [omegaLimit, image2_image_right]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem omegaLimit_preimage_subset {α' : Type*} (ϕ : τ → α' → β) (s : Set α') (f : Filter τ)
    (g : α → α') : ω f (fun t x ↦ ϕ t (g x)) (g ⁻¹' s) ⊆ ω f ϕ s :=
  mapsTo_omegaLimit _ (mapsTo_preimage _ _) (fun _t _x ↦ rfl) continuous_id


/-- An element `y` is in the ω-limit set of `s` w.r.t. `f` if the
    preimages of an arbitrary neighbourhood of `y` frequently
    (w.r.t. `f`) intersects of `s`. -/
theorem mem_omegaLimit_iff_frequently (y : β) :
    y ∈ ω f ϕ s ↔ ∀ n ∈ 𝓝 y, ∃ᶠ t in f, (s ∩ ϕ t ⁻¹' n).Nonempty := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    y : β
    ⊢ Iff (Membership.mem (omegaLimit f ϕ s) y) (∀ (n : Set β), Membership.mem (nh …
  -/
  simp_rw [frequently_iff, omegaLimit_def, mem_iInter, mem_closure_iff_nhds]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    y : β
    ⊢ Iff (∀ (i : Set τ), Membership.mem f i → ∀ (t : Set β), Membership.mem (nhds …
  -/
  constructor
    /-
      case mp
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      ⊢ (∀ (i : Set τ), Membership.mem f i → ∀ (t : Set β), Membership.mem (nhds y)  …
    -/
  · intro h _ hn _ hu
    /-
      case mp
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      h : ∀ (i : Set τ), Membership.mem f i → ∀ (t : Set β), Membership.mem (nhds y) …
      n✝ : Set β
      hn : Membership.mem (nhds y) n✝
      U✝ : Set τ
      hu : Membership.mem f U✝
      ⊢ Exists fun x => And (Membership.mem U✝ x) (Inter.inter s (Set.preimage (ϕ x) …
    -/
    rcases h _ hu _ hn with ⟨_, _, _, ht, _, hx, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      h : ∀ (i : Set τ), Membership.mem f i → ∀ (t : Set β), Membership.mem (nhds y) …
      n✝ : Set β
      hn : Membership.mem (nhds y) n✝
      U✝ : Set τ
      hu : Membership.mem f U✝
      w✝¹ : τ
      ht : Membership.mem U✝ w✝¹
      w✝ : α
      hx : Membership.mem s w✝
      left✝ : Membership.mem n✝ (ϕ w✝¹ w✝)
      ⊢ Exists fun x => And (Membership.mem U✝ x) (Inter.inter s (Set.preimage (ϕ x) …
    -/
    exact ⟨_, ht, _, hx, by rwa [mem_preimage]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      ⊢ (∀ (n : Set β), Membership.mem (nhds y) n → ∀ {U : Set τ}, Membership.mem f  …
    -/
  · intro h _ hu _ hn
    /-
      case mpr
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      h : ∀ (n : Set β), Membership.mem (nhds y) n → ∀ {U : Set τ}, Membership.mem f …
      i✝ : Set τ
      hu : Membership.mem f i✝
      t✝ : Set β
      hn : Membership.mem (nhds y) t✝
      ⊢ (Inter.inter t✝ (Set.image2 ϕ i✝ s)).Nonempty
    -/
    rcases h _ hn hu with ⟨_, ht, _, hx, hϕtx⟩
    /-
      case mpr.intro.intro.intro.intro
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      y : β
      h : ∀ (n : Set β), Membership.mem (nhds y) n → ∀ {U : Set τ}, Membership.mem f …
      i✝ : Set τ
      hu : Membership.mem f i✝
      t✝ : Set β
      hn : Membership.mem (nhds y) t✝
      w✝¹ : τ
      ht : Membership.mem i✝ w✝¹
      w✝ : α
      hx : Membership.mem s w✝
      hϕtx : Membership.mem (Set.preimage (ϕ w✝¹) t✝) w✝
      ⊢ (Inter.inter t✝ (Set.image2 ϕ i✝ s)).Nonempty
    -/
    exact ⟨_, hϕtx, _, ht, _, hx, rfl⟩
    /-
      🎉 no goals
    -/


/-- An element `y` is in the ω-limit set of `s` w.r.t. `f` if the
    forward images of `s` frequently (w.r.t. `f`) intersect arbitrary
    neighbourhoods of `y`. -/
theorem mem_omegaLimit_iff_frequently₂ (y : β) :
    y ∈ ω f ϕ s ↔ ∀ n ∈ 𝓝 y, ∃ᶠ t in f, (ϕ t '' s ∩ n).Nonempty := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    y : β
    ⊢ Iff (Membership.mem (omegaLimit f ϕ s) y) (∀ (n : Set β), Membership.mem (nh …
  -/
  simp_rw [mem_omegaLimit_iff_frequently, image_inter_nonempty_iff]
  /-
    🎉 no goals
  -/


/-- An element `y` is in the ω-limit of `x` w.r.t. `f` if the forward
    images of `x` frequently (w.r.t. `f`) falls within an arbitrary
    neighbourhood of `y`. -/
theorem mem_omegaLimit_singleton_iff_map_cluster_point (x : α) (y : β) :
    y ∈ ω f ϕ {x} ↔ MapClusterPt y f fun t ↦ ϕ t x := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    x : α
    y : β
    ⊢ Iff (Membership.mem (omegaLimit f ϕ (Singleton.singleton x)) y) (MapClusterP …
  -/
  simp_rw [mem_omegaLimit_iff_frequently, mapClusterPt_iff, singleton_inter_nonempty, mem_preimage]
  /-
    🎉 no goals
  -/


theorem omegaLimit_inter : ω f ϕ (s₁ ∩ s₂) ⊆ ω f ϕ s₁ ∩ ω f ϕ s₂ :=
  subset_inter (omegaLimit_mono_right _ _ inter_subset_left)
    (omegaLimit_mono_right _ _ inter_subset_right)


theorem omegaLimit_iInter (p : ι → Set α) : ω f ϕ (⋂ i, p i) ⊆ ⋂ i, ω f ϕ (p i) :=
  subset_iInter fun _i ↦ omegaLimit_mono_right _ _ (iInter_subset _ _)


theorem omegaLimit_union : ω f ϕ (s₁ ∪ s₂) = ω f ϕ s₁ ∪ ω f ϕ s₂ := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s₁ s₂ : Set α
    ⊢ Eq (omegaLimit f ϕ (Union.union s₁ s₂)) (Union.union (omegaLimit f ϕ s₁) (om …
  -/
  ext y; constructor
  · simp only [mem_union, mem_omegaLimit_iff_frequently, union_inter_distrib_right, union_nonempty,
      frequently_or_distrib]
    /-
      case h.mp
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s₁ s₂ : Set α
      y : β
      ⊢ (∀ (n : Set β), Membership.mem (nhds y) n → Or (Filter.Frequently (fun x =>  …
    -/
    contrapose!
    /-
      case h.mp
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s₁ s₂ : Set α
      y : β
      ⊢ And (Exists fun n => And (Membership.mem (nhds y) n) (Not (Filter.Frequently …
    -/
    simp only [not_frequently, not_nonempty_iff_eq_empty, ← subset_empty_iff]
    /-
      case h.mp
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s₁ s₂ : Set α
      y : β
      ⊢ And (Exists fun n => And (Membership.mem (nhds y) n) (Filter.Eventually (fun …
    -/
    rintro ⟨⟨n₁, hn₁, h₁⟩, ⟨n₂, hn₂, h₂⟩⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s₁ s₂ : Set α
      y : β
      n₁ : Set β
      hn₁ : Membership.mem (nhds y) n₁
      h₁ : Filter.Eventually (fun x => HasSubset.Subset (Inter.inter s₁ (Set.preimag …
      n₂ : Set β
      hn₂ : Membership.mem (nhds y) n₂
      h₂ : Filter.Eventually (fun x => HasSubset.Subset (Inter.inter s₂ (Set.preimag …
      ⊢ Exists fun n => And (Membership.mem (nhds y) n) (And (Filter.Eventually (fun …
    -/
    refine ⟨n₁ ∩ n₂, inter_mem hn₁ hn₂, h₁.mono fun t ↦ ?_, h₂.mono fun t ↦ ?_⟩
    exacts [Subset.trans <| inter_subset_inter_right _ <| preimage_mono inter_subset_left,
      Subset.trans <| inter_subset_inter_right _ <| preimage_mono inter_subset_right]
    /-
      case h.mpr
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s₁ s₂ : Set α
      y : β
      ⊢ Membership.mem (Union.union (omegaLimit f ϕ s₁) (omegaLimit f ϕ s₂)) y → Mem …
    -/
  · rintro (hy | hy)
    exacts [omegaLimit_mono_right _ _ subset_union_left hy,
      omegaLimit_mono_right _ _ subset_union_right hy]


theorem omegaLimit_iUnion (p : ι → Set α) : ⋃ i, ω f ϕ (p i) ⊆ ω f ϕ (⋃ i, p i) := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    ι : Type u_4
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    p : ι → Set α
    ⊢ HasSubset.Subset (Set.iUnion fun i => omegaLimit f ϕ (p i)) (omegaLimit f ϕ  …
  -/
  rw [iUnion_subset_iff]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    ι : Type u_4
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    p : ι → Set α
    ⊢ ∀ (i : ι), HasSubset.Subset (omegaLimit f ϕ (p i)) (omegaLimit f ϕ (Set.iUni …
  -/
  exact fun i ↦ omegaLimit_mono_right _ _ (subset_iUnion _ _)
  /-
    🎉 no goals
  -/


theorem omegaLimit_eq_iInter : ω f ϕ s = ⋂ u : ↥f.sets, closure (image2 ϕ u s) :=
  biInter_eq_iInter _ _


theorem omegaLimit_eq_biInter_inter {v : Set τ} (hv : v ∈ f) :
    ω f ϕ s = ⋂ u ∈ f, closure (image2 ϕ (u ∩ v) s) :=
  Subset.antisymm (iInter₂_mono' fun u hu ↦ ⟨u ∩ v, inter_mem hu hv, Subset.rfl⟩)
    (iInter₂_mono fun _u _hu ↦ closure_mono <| image2_subset inter_subset_left Subset.rfl)


theorem omegaLimit_eq_iInter_inter {v : Set τ} (hv : v ∈ f) :
    ω f ϕ s = ⋂ u : ↥f.sets, closure (image2 ϕ (u ∩ v) s) := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    v : Set τ
    hv : Membership.mem f v
    ⊢ Eq (omegaLimit f ϕ s) (Set.iInter fun u => closure (Set.image2 ϕ (Inter.inte …
  -/
  rw [omegaLimit_eq_biInter_inter _ _ _ hv]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    v : Set τ
    hv : Membership.mem f v
    ⊢ Eq (Set.iInter fun u => Set.iInter fun h => closure (Set.image2 ϕ (Inter.int …
  -/
  apply biInter_eq_iInter
  /-
    🎉 no goals
  -/


theorem omegaLimit_subset_closure_fw_image {u : Set τ} (hu : u ∈ f) :
    ω f ϕ s ⊆ closure (image2 ϕ u s) := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    u : Set τ
    hu : Membership.mem f u
    ⊢ HasSubset.Subset (omegaLimit f ϕ s) (closure (Set.image2 ϕ u s))
  -/
  rw [omegaLimit_eq_iInter]
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    u : Set τ
    hu : Membership.mem f u
    ⊢ HasSubset.Subset (Set.iInter fun u => closure (Set.image2 ϕ (↑u) s)) (closur …
  -/
  intro _ hx
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    u : Set τ
    hu : Membership.mem f u
    a✝ : β
    hx : Membership.mem (Set.iInter fun u => closure (Set.image2 ϕ (↑u) s)) a✝
    ⊢ Membership.mem (closure (Set.image2 ϕ u s)) a✝
  -/
  rw [mem_iInter] at hx
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    u : Set τ
    hu : Membership.mem f u
    a✝ : β
    hx : ∀ (i : ↑f.sets), Membership.mem (closure (Set.image2 ϕ (↑i) s)) a✝
    ⊢ Membership.mem (closure (Set.image2 ϕ u s)) a✝
  -/
  exact hx ⟨u, hu⟩
  /-
    🎉 no goals
  -/

-- An instance with better keys

instance : Inhabited f.sets := Filter.inhabitedMem


/-- A set is eventually carried into any open neighbourhood of its ω-limit:
if `c` is a compact set such that `closure {ϕ t x | t ∈ v, x ∈ s} ⊆ c` for some `v ∈ f`
and `n` is an open neighbourhood of `ω f ϕ s`, then for some `u ∈ f` we have
`closure {ϕ t x | t ∈ u, x ∈ s} ⊆ n`. -/
theorem eventually_closure_subset_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset' {c : Set β}
    (hc₁ : IsCompact c) (hc₂ : ∃ v ∈ f, closure (image2 ϕ v s) ⊆ c) {n : Set β} (hn₁ : IsOpen n)
    (hn₂ : ω f ϕ s ⊆ n) : ∃ u ∈ f, closure (image2 ϕ u s) ⊆ n := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    hc₂ : Exists fun v => And (Membership.mem f v) (HasSubset.Subset (closure (Set …
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  rcases hc₂ with ⟨v, hv₁, hv₂⟩
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  let k := closure (image2 ϕ v s)
  have hk : IsCompact (k \ n) :=
    (hc₁.of_isClosed_subset isClosed_closure hv₂).diff hn₁
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  let j u := (closure (image2 ϕ (u ∩ v) s))ᶜ
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  have hj₁ : ∀ u ∈ f, IsOpen (j u) := fun _ _ ↦ isOpen_compl_iff.mpr isClosed_closure
  have hj₂ : k \ n ⊆ ⋃ u ∈ f, j u := by
    have : ⋃ u ∈ f, j u = ⋃ u : (↥f.sets), j u := biUnion_eq_iUnion _ _
    rw [this, diff_subset_comm, diff_iUnion]
    rw [omegaLimit_eq_iInter_inter _ _ _ hv₁] at hn₂
    simp_rw [j, diff_compl]
    rw [← inter_iInter]
    exact Subset.trans inter_subset_right hn₂
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    hj₁ : ∀ (u : Set τ), Membership.mem f u → IsOpen (j u)
    hj₂ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun u => Set.iUnion fun h …
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  rcases hk.elim_finite_subcover_image hj₁ hj₂ with ⟨g, hg₁ : ∀ u ∈ g, u ∈ f, hg₂, hg₃⟩
  /-
    case intro.intro.intro.intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    hj₁ : ∀ (u : Set τ), Membership.mem f u → IsOpen (j u)
    hj₂ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun u => Set.iUnion fun h …
    g : Set (Set τ)
    hg₁ : ∀ (u : Set τ), Membership.mem g u → Membership.mem f u
    hg₂ : g.Finite
    hg₃ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun i => Set.iUnion fun h …
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  let w := (⋂ u ∈ g, u) ∩ v
  /-
    case intro.intro.intro.intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    hj₁ : ∀ (u : Set τ), Membership.mem f u → IsOpen (j u)
    hj₂ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun u => Set.iUnion fun h …
    g : Set (Set τ)
    hg₁ : ∀ (u : Set τ), Membership.mem g u → Membership.mem f u
    hg₂ : g.Finite
    hg₃ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun i => Set.iUnion fun h …
    w : Set τ := Inter.inter (Set.iInter fun u => Set.iInter fun h => u) v
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  have hw₂ : w ∈ f := by simpa [w, *]
  have hw₃ : k \ n ⊆ (closure (image2 ϕ w s))ᶜ := by
    apply Subset.trans hg₃
    simp only [j, iUnion_subset_iff, compl_subset_compl]
    intros u hu
    mono
    refine iInter_subset_of_subset u (iInter_subset_of_subset hu ?_)
    all_goals exact Subset.rfl
  have hw₄ : kᶜ ⊆ (closure (image2 ϕ w s))ᶜ := by
    simp only [compl_subset_compl]
    exact closure_mono (image2_subset inter_subset_right Subset.rfl)
  /-
    case intro.intro.intro.intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    hj₁ : ∀ (u : Set τ), Membership.mem f u → IsOpen (j u)
    hj₂ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun u => Set.iUnion fun h …
    g : Set (Set τ)
    hg₁ : ∀ (u : Set τ), Membership.mem g u → Membership.mem f u
    hg₂ : g.Finite
    hg₃ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun i => Set.iUnion fun h …
    w : Set τ := Inter.inter (Set.iInter fun u => Set.iInter fun h => u) v
    hw₂ : Membership.mem f w
    hw₃ : HasSubset.Subset (SDiff.sdiff k n) (HasCompl.compl (closure (Set.image2  …
    hw₄ : HasSubset.Subset (HasCompl.compl k) (HasCompl.compl (closure (Set.image2 …
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  have hnc : nᶜ ⊆ k \ n ∪ kᶜ := by rw [union_comm, ← inter_subset, diff_eq, inter_comm]
  have hw : closure (image2 ϕ w s) ⊆ n :=
    compl_subset_compl.mp (Subset.trans hnc (union_subset hw₃ hw₄))
  /-
    case intro.intro.intro.intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    c : Set β
    hc₁ : IsCompact c
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    k : Set β := closure (Set.image2 ϕ v s)
    hk : IsCompact (SDiff.sdiff k n)
    j : Set τ → Set β := fun u => HasCompl.compl (closure (Set.image2 ϕ (Inter.int …
    hj₁ : ∀ (u : Set τ), Membership.mem f u → IsOpen (j u)
    hj₂ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun u => Set.iUnion fun h …
    g : Set (Set τ)
    hg₁ : ∀ (u : Set τ), Membership.mem g u → Membership.mem f u
    hg₂ : g.Finite
    hg₃ : HasSubset.Subset (SDiff.sdiff k n) (Set.iUnion fun i => Set.iUnion fun h …
    w : Set τ := Inter.inter (Set.iInter fun u => Set.iInter fun h => u) v
    hw₂ : Membership.mem f w
    hw₃ : HasSubset.Subset (SDiff.sdiff k n) (HasCompl.compl (closure (Set.image2  …
    hw₄ : HasSubset.Subset (HasCompl.compl k) (HasCompl.compl (closure (Set.image2 …
    hnc : HasSubset.Subset (HasCompl.compl n) (Union.union (SDiff.sdiff k n) (HasC …
    hw : HasSubset.Subset (closure (Set.image2 ϕ w s)) n
    ⊢ Exists fun u => And (Membership.mem f u) (HasSubset.Subset (closure (Set.ima …
  -/
  exact ⟨_, hw₂, hw⟩
  /-
    🎉 no goals
  -/


/-- A set is eventually carried into any open neighbourhood of its ω-limit:
if `c` is a compact set such that `closure {ϕ t x | t ∈ v, x ∈ s} ⊆ c` for some `v ∈ f`
and `n` is an open neighbourhood of `ω f ϕ s`, then for some `u ∈ f` we have
`closure {ϕ t x | t ∈ u, x ∈ s} ⊆ n`. -/
theorem eventually_closure_subset_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset [T2Space β]
    {c : Set β} (hc₁ : IsCompact c) (hc₂ : ∀ᶠ t in f, MapsTo (ϕ t) s c) {n : Set β} (hn₁ : IsOpen n)
    (hn₂ : ω f ϕ s ⊆ n) : ∃ u ∈ f, closure (image2 ϕ u s) ⊆ n :=
  eventually_closure_subset_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset' f ϕ _ hc₁
    ⟨_, hc₂, closure_minimal (image2_subset_iff.2 fun _t ↦ id) hc₁.isClosed⟩ hn₁ hn₂


theorem eventually_mapsTo_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset [T2Space β]
    {c : Set β} (hc₁ : IsCompact c) (hc₂ : ∀ᶠ t in f, MapsTo (ϕ t) s c) {n : Set β} (hn₁ : IsOpen n)
    (hn₂ : ω f ϕ s ⊆ n) : ∀ᶠ t in f, MapsTo (ϕ t) s n := by
  rcases eventually_closure_subset_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset f ϕ s hc₁
      hc₂ hn₁ hn₂ with
    ⟨u, hu_mem, hu⟩
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : T2Space β
    c : Set β
    hc₁ : IsCompact c
    hc₂ : Filter.Eventually (fun t => Set.MapsTo (ϕ t) s c) f
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    u : Set τ
    hu_mem : Membership.mem f u
    hu : HasSubset.Subset (closure (Set.image2 ϕ u s)) n
    ⊢ Filter.Eventually (fun t => Set.MapsTo (ϕ t) s n) f
  -/
  refine mem_of_superset hu_mem fun t ht x hx ↦ ?_
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : T2Space β
    c : Set β
    hc₁ : IsCompact c
    hc₂ : Filter.Eventually (fun t => Set.MapsTo (ϕ t) s c) f
    n : Set β
    hn₁ : IsOpen n
    hn₂ : HasSubset.Subset (omegaLimit f ϕ s) n
    u : Set τ
    hu_mem : Membership.mem f u
    hu : HasSubset.Subset (closure (Set.image2 ϕ u s)) n
    t : τ
    ht : Membership.mem u t
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem n (ϕ t x)
  -/
  exact hu (subset_closure <| mem_image2_of_mem ht hx)
  /-
    🎉 no goals
  -/


theorem eventually_closure_subset_of_isOpen_of_omegaLimit_subset [CompactSpace β] {v : Set β}
    (hv₁ : IsOpen v) (hv₂ : ω f ϕ s ⊆ v) : ∃ u ∈ f, closure (image2 ϕ u s) ⊆ v :=
  eventually_closure_subset_of_isCompact_absorbing_of_isOpen_of_omegaLimit_subset' _ _ _
    isCompact_univ ⟨univ, univ_mem, subset_univ _⟩ hv₁ hv₂


theorem eventually_mapsTo_of_isOpen_of_omegaLimit_subset [CompactSpace β] {v : Set β}
    (hv₁ : IsOpen v) (hv₂ : ω f ϕ s ⊆ v) : ∀ᶠ t in f, MapsTo (ϕ t) s v := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : CompactSpace β
    v : Set β
    hv₁ : IsOpen v
    hv₂ : HasSubset.Subset (omegaLimit f ϕ s) v
    ⊢ Filter.Eventually (fun t => Set.MapsTo (ϕ t) s v) f
  -/
  rcases eventually_closure_subset_of_isOpen_of_omegaLimit_subset f ϕ s hv₁ hv₂ with ⟨u, hu_mem, hu⟩
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : CompactSpace β
    v : Set β
    hv₁ : IsOpen v
    hv₂ : HasSubset.Subset (omegaLimit f ϕ s) v
    u : Set τ
    hu_mem : Membership.mem f u
    hu : HasSubset.Subset (closure (Set.image2 ϕ u s)) v
    ⊢ Filter.Eventually (fun t => Set.MapsTo (ϕ t) s v) f
  -/
  refine mem_of_superset hu_mem fun t ht x hx ↦ ?_
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : CompactSpace β
    v : Set β
    hv₁ : IsOpen v
    hv₂ : HasSubset.Subset (omegaLimit f ϕ s) v
    u : Set τ
    hu_mem : Membership.mem f u
    hu : HasSubset.Subset (closure (Set.image2 ϕ u s)) v
    t : τ
    ht : Membership.mem u t
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem v (ϕ t x)
  -/
  exact hu (subset_closure <| mem_image2_of_mem ht hx)
  /-
    🎉 no goals
  -/


/-- The ω-limit of a nonempty set w.r.t. a nontrivial filter is nonempty. -/
theorem nonempty_omegaLimit_of_isCompact_absorbing [NeBot f] {c : Set β} (hc₁ : IsCompact c)
    (hc₂ : ∃ v ∈ f, closure (image2 ϕ v s) ⊆ c) (hs : s.Nonempty) : (ω f ϕ s).Nonempty := by
  /-
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : f.NeBot
    c : Set β
    hc₁ : IsCompact c
    hc₂ : Exists fun v => And (Membership.mem f v) (HasSubset.Subset (closure (Set …
    hs : s.Nonempty
    ⊢ (omegaLimit f ϕ s).Nonempty
  -/
  rcases hc₂ with ⟨v, hv₁, hv₂⟩
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : f.NeBot
    c : Set β
    hc₁ : IsCompact c
    hs : s.Nonempty
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    ⊢ (omegaLimit f ϕ s).Nonempty
  -/
  rw [omegaLimit_eq_iInter_inter _ _ _ hv₁]
  /-
    case intro.intro
    τ : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    f : Filter τ
    ϕ : τ → α → β
    s : Set α
    inst✝ : f.NeBot
    c : Set β
    hc₁ : IsCompact c
    hs : s.Nonempty
    v : Set τ
    hv₁ : Membership.mem f v
    hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
    ⊢ (Set.iInter fun u => closure (Set.image2 ϕ (Inter.inter (↑u) v) s)).Nonempty
  -/
  apply IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
    /-
      case intro.intro.htd
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      ⊢ Directed (fun x1 x2 => Superset x1 x2) fun i => closure (Set.image2 ϕ (Inter …
    -/
  · rintro ⟨u₁, hu₁⟩ ⟨u₂, hu₂⟩
    /-
      case intro.intro.htd.mk.mk
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      u₁ : Set τ
      hu₁ : Membership.mem f.sets u₁
      u₂ : Set τ
      hu₂ : Membership.mem f.sets u₂
      ⊢ Exists fun z => And ((fun x1 x2 => Superset x1 x2) ((fun i => closure (Set.i …
    -/
    use ⟨u₁ ∩ u₂, inter_mem hu₁ hu₂⟩
    /-
      case h
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      u₁ : Set τ
      hu₁ : Membership.mem f.sets u₁
      u₂ : Set τ
      hu₂ : Membership.mem f.sets u₂
      ⊢ And ((fun x1 x2 => Superset x1 x2) ((fun i => closure (Set.image2 ϕ (Inter.i …
    -/
    constructor
    /-
      case h.left
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      u₁ : Set τ
      hu₁ : Membership.mem f.sets u₁
      u₂ : Set τ
      hu₂ : Membership.mem f.sets u₂
      ⊢ (fun x1 x2 => Superset x1 x2) ((fun i => closure (Set.image2 ϕ (Inter.inter  …
    -/
    all_goals exact closure_mono (image2_subset (inter_subset_inter_left _ (by simp)) Subset.rfl)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.htn
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      ⊢ ∀ (i : ↑f.sets), (closure (Set.image2 ϕ (Inter.inter (↑i) v) s)).Nonempty
    -/
  · intro u
    have hn : (image2 ϕ (u ∩ v) s).Nonempty :=
      Nonempty.image2 (Filter.nonempty_of_mem (inter_mem u.prop hv₁)) hs
    /-
      case intro.intro.htn
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      u : ↑f.sets
      hn : (Set.image2 ϕ (Inter.inter (↑u) v) s).Nonempty
      ⊢ (closure (Set.image2 ϕ (Inter.inter (↑u) v) s)).Nonempty
    -/
    exact hn.mono subset_closure
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.htc
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      ⊢ ∀ (i : ↑f.sets), IsCompact (closure (Set.image2 ϕ (Inter.inter (↑i) v) s))
    -/
  · intro
    /-
      case intro.intro.htc
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      i✝ : ↑f.sets
      ⊢ IsCompact (closure (Set.image2 ϕ (Inter.inter (↑i✝) v) s))
    -/
    apply hc₁.of_isClosed_subset isClosed_closure
    calc
      _ ⊆ closure (image2 ϕ v s) := closure_mono (image2_subset inter_subset_right Subset.rfl)
      _ ⊆ c := hv₂
    /-
      case intro.intro.htcl
      τ : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      f : Filter τ
      ϕ : τ → α → β
      s : Set α
      inst✝ : f.NeBot
      c : Set β
      hc₁ : IsCompact c
      hs : s.Nonempty
      v : Set τ
      hv₁ : Membership.mem f v
      hv₂ : HasSubset.Subset (closure (Set.image2 ϕ v s)) c
      ⊢ ∀ (i : ↑f.sets), IsClosed (closure (Set.image2 ϕ (Inter.inter (↑i) v) s))
    -/
  · exact fun _ ↦ isClosed_closure
    /-
      🎉 no goals
    -/


theorem nonempty_omegaLimit [CompactSpace β] [NeBot f] (hs : s.Nonempty) : (ω f ϕ s).Nonempty :=
  nonempty_omegaLimit_of_isCompact_absorbing _ _ _ isCompact_univ ⟨univ, univ_mem, subset_univ _⟩ hs


theorem isInvariant_omegaLimit (hf : ∀ t, Tendsto (t + ·) f f) : IsInvariant ϕ (ω f ϕ s) := by
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddMonoid τ
    inst✝¹ : ContinuousAdd τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    ⊢ IsInvariant ϕ.toFun (omegaLimit f ϕ.toFun s)
  -/
  refine fun t ↦ MapsTo.mono_right ?_ (omegaLimit_subset_of_tendsto ϕ s (hf t))
  exact
    mapsTo_omegaLimit _ (mapsTo_id _) (fun t' x ↦ (ϕ.map_add _ _ _).symm)
      (continuous_const.flow ϕ continuous_id)


theorem omegaLimit_image_subset (t : τ) (ht : Tendsto (· + t) f f) :
    ω f ϕ (ϕ t '' s) ⊆ ω f ϕ s := by
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddMonoid τ
    inst✝¹ : ContinuousAdd τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    t : τ
    ht : Filter.Tendsto (fun x => HAdd.hAdd x t) f f
    ⊢ HasSubset.Subset (omegaLimit f ϕ.toFun (Set.image (ϕ.toFun t) s)) (omegaLimi …
  -/
  simp only [omegaLimit_image_eq, ← map_add]
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddMonoid τ
    inst✝¹ : ContinuousAdd τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    t : τ
    ht : Filter.Tendsto (fun x => HAdd.hAdd x t) f f
    ⊢ HasSubset.Subset (omegaLimit f (fun t_1 x => ϕ.toFun (HAdd.hAdd t_1 t) x) s) …
  -/
  exact omegaLimit_subset_of_tendsto ϕ s ht
  /-
    🎉 no goals
  -/


/-- the ω-limit of a forward image of `s` is the same as the ω-limit of `s`. -/
@[simp]
theorem omegaLimit_image_eq (hf : ∀ t, Tendsto (· + t) f f) (t : τ) : ω f ϕ (ϕ t '' s) = ω f ϕ s :=
  Subset.antisymm (omegaLimit_image_subset _ _ _ _ (hf t)) <|
    calc
                                                   /-
                                                     τ : Type u_1
                                                     inst✝³ : TopologicalSpace τ
                                                     inst✝² : AddCommGroup τ
                                                     inst✝¹ : TopologicalAddGroup τ
                                                     α : Type u_2
                                                     inst✝ : TopologicalSpace α
                                                     f : Filter τ
                                                     ϕ : Flow τ α
                                                     s : Set α
                                                     hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd x t) f f
                                                     t : τ
                                                     ⊢ Eq (omegaLimit f ϕ.toFun s) (omegaLimit f ϕ.toFun (Set.image (ϕ.toFun (Neg.n …
                                                   -/
      ω f ϕ s = ω f ϕ (ϕ (-t) '' (ϕ t '' s)) := by simp [image_image, ← map_add]
                                                   /-
                                                     🎉 no goals
                                                   -/
      _ ⊆ ω f ϕ (ϕ t '' s) := omegaLimit_image_subset _ _ _ _ (hf _)


theorem omegaLimit_omegaLimit (hf : ∀ t, Tendsto (t + ·) f f) : ω f ϕ (ω f ϕ s) ⊆ ω f ϕ s := by
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    ⊢ HasSubset.Subset (omegaLimit f ϕ.toFun (omegaLimit f ϕ.toFun s)) (omegaLimit …
  -/
  simp only [subset_def, mem_omegaLimit_iff_frequently₂, frequently_iff]
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    ⊢ ∀ (x : α), (∀ (n : Set α), Membership.mem (nhds x) n → ∀ {U : Set τ}, Member …
  -/
  intro _ h
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    x✝ : α
    h : ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem  …
    ⊢ ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem f  …
  -/
  rintro n hn u hu
  /-
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    x✝ : α
    h : ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem  …
    n : Set α
    hn : Membership.mem (nhds x✝) n
    u : Set τ
    hu : Membership.mem f u
    ⊢ Exists fun x => And (Membership.mem u x) (Inter.inter (Set.image (ϕ.toFun x) …
  -/
  rcases mem_nhds_iff.mp hn with ⟨o, ho₁, ho₂, ho₃⟩
  /-
    case intro.intro.intro
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    x✝ : α
    h : ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem  …
    n : Set α
    hn : Membership.mem (nhds x✝) n
    u : Set τ
    hu : Membership.mem f u
    o : Set α
    ho₁ : HasSubset.Subset o n
    ho₂ : IsOpen o
    ho₃ : Membership.mem o x✝
    ⊢ Exists fun x => And (Membership.mem u x) (Inter.inter (Set.image (ϕ.toFun x) …
  -/
  rcases h o (IsOpen.mem_nhds ho₂ ho₃) hu with ⟨t, _ht₁, ht₂⟩
  have l₁ : (ω f ϕ s ∩ o).Nonempty :=
    ht₂.mono
      (inter_subset_inter_left _
        ((isInvariant_iff_image _ _).mp (isInvariant_omegaLimit _ _ _ hf) _))
  have l₂ : (closure (image2 ϕ u s) ∩ o).Nonempty :=
    l₁.mono fun b hb ↦ ⟨omegaLimit_subset_closure_fw_image _ _ _ hu hb.1, hb.2⟩
  have l₃ : (o ∩ image2 ϕ u s).Nonempty := by
    rcases l₂ with ⟨b, hb₁, hb₂⟩
    exact mem_closure_iff_nhds.mp hb₁ o (IsOpen.mem_nhds ho₂ hb₂)
  /-
    case intro.intro.intro.intro.intro
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    x✝ : α
    h : ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem  …
    n : Set α
    hn : Membership.mem (nhds x✝) n
    u : Set τ
    hu : Membership.mem f u
    o : Set α
    ho₁ : HasSubset.Subset o n
    ho₂ : IsOpen o
    ho₃ : Membership.mem o x✝
    t : τ
    _ht₁ : Membership.mem u t
    ht₂ : (Inter.inter (Set.image (ϕ.toFun t) (omegaLimit f ϕ.toFun s)) o).Nonempty
    l₁ : (Inter.inter (omegaLimit f ϕ.toFun s) o).Nonempty
    l₂ : (Inter.inter (closure (Set.image2 ϕ.toFun u s)) o).Nonempty
    l₃ : (Inter.inter o (Set.image2 ϕ.toFun u s)).Nonempty
    ⊢ Exists fun x => And (Membership.mem u x) (Inter.inter (Set.image (ϕ.toFun x) …
  -/
  rcases l₃ with ⟨ϕra, ho, ⟨_, hr, _, ha, hϕra⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    τ : Type u_1
    inst✝³ : TopologicalSpace τ
    inst✝² : AddCommGroup τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : Filter τ
    ϕ : Flow τ α
    s : Set α
    hf : ∀ (t : τ), Filter.Tendsto (fun x => HAdd.hAdd t x) f f
    x✝ : α
    h : ∀ (n : Set α), Membership.mem (nhds x✝) n → ∀ {U : Set τ}, Membership.mem  …
    n : Set α
    hn : Membership.mem (nhds x✝) n
    u : Set τ
    hu : Membership.mem f u
    o : Set α
    ho₁ : HasSubset.Subset o n
    ho₂ : IsOpen o
    ho₃ : Membership.mem o x✝
    t : τ
    _ht₁ : Membership.mem u t
    ht₂ : (Inter.inter (Set.image (ϕ.toFun t) (omegaLimit f ϕ.toFun s)) o).Nonempty
    l₁ : (Inter.inter (omegaLimit f ϕ.toFun s) o).Nonempty
    l₂ : (Inter.inter (closure (Set.image2 ϕ.toFun u s)) o).Nonempty
    ϕra : α
    ho : Membership.mem o ϕra
    w✝¹ : τ
    hr : Membership.mem u w✝¹
    w✝ : α
    ha : Membership.mem s w✝
    hϕra : Eq (ϕ.toFun w✝¹ w✝) ϕra
    ⊢ Exists fun x => And (Membership.mem u x) (Inter.inter (Set.image (ϕ.toFun x) …
  -/
  exact ⟨_, hr, ϕra, ⟨_, ha, hϕra⟩, ho₁ ho⟩
  /-
    🎉 no goals
  -/


