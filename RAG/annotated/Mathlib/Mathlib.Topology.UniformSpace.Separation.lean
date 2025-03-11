instance (priority := 100) UniformSpace.to_regularSpace : RegularSpace α :=
  .of_hasBasis
    (fun _ ↦ nhds_basis_uniformity' uniformity_hasBasis_closed)
    fun a _V hV ↦ isClosed_ball a hV.2


theorem Filter.HasBasis.specializes_iff_uniformity {ι : Sort*} {p : ι → Prop} {s : ι → Set (α × α)}
    (h : (𝓤 α).HasBasis p s) {x y : α} : x ⤳ y ↔ ∀ i, p i → (x, y) ∈ s i :=
  (nhds_basis_uniformity h).specializes_iff


theorem Filter.HasBasis.inseparable_iff_uniformity {ι : Sort*} {p : ι → Prop} {s : ι → Set (α × α)}
    (h : (𝓤 α).HasBasis p s) {x y : α} : Inseparable x y ↔ ∀ i, p i → (x, y) ∈ s i :=
  specializes_iff_inseparable.symm.trans h.specializes_iff_uniformity


theorem inseparable_iff_ker_uniformity {x y : α} : Inseparable x y ↔ (x, y) ∈ (𝓤 α).ker :=
  (𝓤 α).basis_sets.inseparable_iff_uniformity


protected theorem Inseparable.nhds_le_uniformity {x y : α} (h : Inseparable x y) :
    𝓝 (x, y) ≤ 𝓤 α := by
  /-
    α : Type u
    inst✝ : UniformSpace α
    x y : α
    h : Inseparable x y
    ⊢ LE.le (nhds { fst := x, snd := y }) (uniformity α)
  -/
  rw [h.prod rfl]
  /-
    α : Type u
    inst✝ : UniformSpace α
    x y : α
    h : Inseparable x y
    ⊢ LE.le (nhds { fst := y, snd := y }) (uniformity α)
  -/
  apply nhds_le_uniformity
  /-
    🎉 no goals
  -/


theorem inseparable_iff_clusterPt_uniformity {x y : α} :
    Inseparable x y ↔ ClusterPt (x, y) (𝓤 α) := by
  /-
    α : Type u
    inst✝ : UniformSpace α
    x y : α
    ⊢ Iff (Inseparable x y) (ClusterPt { fst := x, snd := y } (uniformity α))
  -/
  refine ⟨fun h ↦ .of_nhds_le h.nhds_le_uniformity, fun h ↦ ?_⟩
  /-
    α : Type u
    inst✝ : UniformSpace α
    x y : α
    h : ClusterPt { fst := x, snd := y } (uniformity α)
    ⊢ Inseparable x y
  -/
  simp_rw [uniformity_hasBasis_closed.inseparable_iff_uniformity, isClosed_iff_clusterPt]
  /-
    α : Type u
    inst✝ : UniformSpace α
    x y : α
    h : ClusterPt { fst := x, snd := y } (uniformity α)
    ⊢ ∀ (i : Set (Prod α α)), And (Membership.mem (uniformity α) i) (∀ (a : Prod α …
  -/
  exact fun U ⟨hU, hUc⟩ ↦ hUc _ <| h.mono <| le_principal_iff.2 hU
  /-
    🎉 no goals
  -/


theorem t0Space_iff_uniformity :
    T0Space α ↔ ∀ x y, (∀ r ∈ 𝓤 α, (x, y) ∈ r) → x = y := by
  /-
    α : Type u
    inst✝ : UniformSpace α
    ⊢ Iff (T0Space α) (∀ (x y : α), (∀ (r : Set (Prod α α)), Membership.mem (unifo …
  -/
  simp only [t0Space_iff_inseparable, inseparable_iff_ker_uniformity, mem_ker, id]
  /-
    🎉 no goals
  -/


theorem t0Space_iff_uniformity' :
    T0Space α ↔ Pairwise fun x y ↦ ∃ r ∈ 𝓤 α, (x, y) ∉ r := by
  /-
    α : Type u
    inst✝ : UniformSpace α
    ⊢ Iff (T0Space α) (Pairwise fun x y => Exists fun r => And (Membership.mem (un …
  -/
  simp [t0Space_iff_not_inseparable, inseparable_iff_ker_uniformity]
  /-
    🎉 no goals
  -/


theorem t0Space_iff_ker_uniformity : T0Space α ↔ (𝓤 α).ker = diagonal α := by
  simp_rw [t0Space_iff_uniformity, subset_antisymm_iff, diagonal_subset_iff, subset_def,
    Prod.forall, Filter.mem_ker, mem_diagonal_iff, iff_self_and]
  /-
    α : Type u
    inst✝ : UniformSpace α
    ⊢ (∀ (a b : α), (∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Mem …
  -/
  exact fun _ x s hs ↦ refl_mem_uniformity hs
  /-
    🎉 no goals
  -/


theorem eq_of_uniformity {α : Type*} [UniformSpace α] [T0Space α] {x y : α}
    (h : ∀ {V}, V ∈ 𝓤 α → (x, y) ∈ V) : x = y :=
  t0Space_iff_uniformity.mp ‹T0Space α› x y @h


theorem eq_of_uniformity_basis {α : Type*} [UniformSpace α] [T0Space α] {ι : Sort*}
    {p : ι → Prop} {s : ι → Set (α × α)} (hs : (𝓤 α).HasBasis p s) {x y : α}
    (h : ∀ {i}, p i → (x, y) ∈ s i) : x = y :=
  (hs.inseparable_iff_uniformity.2 @h).eq


theorem eq_of_forall_symmetric {α : Type*} [UniformSpace α] [T0Space α] {x y : α}
    (h : ∀ {V}, V ∈ 𝓤 α → SymmetricRel V → (x, y) ∈ V) : x = y :=
                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : UniformSpace α
                                                  inst✝ : T0Space α
                                                  x y : α
                                                  h : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → SymmetricRel V → …
                                                  ⊢ ∀ {i : Set (Prod α α)}, And (Membership.mem (uniformity α) i) (SymmetricRel  …
                                                -/
  eq_of_uniformity_basis hasBasis_symmetric (by simpa)
                                                /-
                                                  🎉 no goals
                                                -/


theorem eq_of_clusterPt_uniformity [T0Space α] {x y : α} (h : ClusterPt (x, y) (𝓤 α)) : x = y :=
  (inseparable_iff_clusterPt_uniformity.2 h).eq


theorem Filter.Tendsto.inseparable_iff_uniformity {β} {l : Filter β} [NeBot l] {f g : β → α}
    {a b : α} (ha : Tendsto f l (𝓝 a)) (hb : Tendsto g l (𝓝 b)) :
    Inseparable a b ↔ Tendsto (fun x ↦ (f x, g x)) l (𝓤 α) := by
  /-
    α : Type u
    inst✝¹ : UniformSpace α
    β : Type u_1
    l : Filter β
    inst✝ : l.NeBot
    f g : β → α
    a b : α
    ha : Filter.Tendsto f l (nhds a)
    hb : Filter.Tendsto g l (nhds b)
    ⊢ Iff (Inseparable a b) (Filter.Tendsto (fun x => { fst := f x, snd := g x })  …
  -/
  refine ⟨fun h ↦ (ha.prod_mk_nhds hb).mono_right h.nhds_le_uniformity, fun h ↦ ?_⟩
  /-
    α : Type u
    inst✝¹ : UniformSpace α
    β : Type u_1
    l : Filter β
    inst✝ : l.NeBot
    f g : β → α
    a b : α
    ha : Filter.Tendsto f l (nhds a)
    hb : Filter.Tendsto g l (nhds b)
    h : Filter.Tendsto (fun x => { fst := f x, snd := g x }) l (uniformity α)
    ⊢ Inseparable a b
  -/
  rw [inseparable_iff_clusterPt_uniformity]
  /-
    α : Type u
    inst✝¹ : UniformSpace α
    β : Type u_1
    l : Filter β
    inst✝ : l.NeBot
    f g : β → α
    a b : α
    ha : Filter.Tendsto f l (nhds a)
    hb : Filter.Tendsto g l (nhds b)
    h : Filter.Tendsto (fun x => { fst := f x, snd := g x }) l (uniformity α)
    ⊢ ClusterPt { fst := a, snd := b } (uniformity α)
  -/
  exact (ClusterPt.of_le_nhds (ha.prod_mk_nhds hb)).mono h
  /-
    🎉 no goals
  -/


theorem isClosed_of_spaced_out [T0Space α] {V₀ : Set (α × α)} (V₀_in : V₀ ∈ 𝓤 α) {s : Set α}
    (hs : s.Pairwise fun x y => (x, y) ∉ V₀) : IsClosed s := by
  /-
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    ⊢ IsClosed s
  -/
  rcases comp_symm_mem_uniformity_sets V₀_in with ⟨V₁, V₁_in, V₁_symm, h_comp⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    ⊢ IsClosed s
  -/
  apply isClosed_of_closure_subset
  /-
    case intro.intro.intro.h
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    ⊢ HasSubset.Subset (closure s) s
  -/
  intro x hx
  /-
    case intro.intro.intro.h
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : Membership.mem (closure s) x
    ⊢ Membership.mem s x
  -/
  rw [mem_closure_iff_ball] at hx
  /-
    case intro.intro.intro.h
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    ⊢ Membership.mem s x
  -/
  rcases hx V₁_in with ⟨y, hy, hy'⟩
  /-
    case intro.intro.intro.h.intro.intro
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    y : α
    hy : Membership.mem (UniformSpace.ball x V₁) y
    hy' : Membership.mem s y
    ⊢ Membership.mem s x
  -/
  suffices x = y by rwa [this]
  /-
    case intro.intro.intro.h.intro.intro
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    y : α
    hy : Membership.mem (UniformSpace.ball x V₁) y
    hy' : Membership.mem s y
    ⊢ Eq x y
  -/
  apply eq_of_forall_symmetric
  /-
    case intro.intro.intro.h.intro.intro.h
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    y : α
    hy : Membership.mem (UniformSpace.ball x V₁) y
    hy' : Membership.mem s y
    ⊢ ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → SymmetricRel V → M …
  -/
  intro V V_in _
  /-
    case intro.intro.intro.h.intro.intro.h
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    y : α
    hy : Membership.mem (UniformSpace.ball x V₁) y
    hy' : Membership.mem s y
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    a✝ : SymmetricRel V
    ⊢ Membership.mem V { fst := x, snd := y }
  -/
  rcases hx (inter_mem V₁_in V_in) with ⟨z, hz, hz'⟩
  obtain rfl : z = y := by
    by_contra hzy
    exact hs hz' hy' hzy (h_comp <| mem_comp_of_mem_ball V₁_symm (ball_inter_left x _ _ hz) hy)
  /-
    case intro.intro.intro.h.intro.intro.h.intro.intro
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    V₀ : Set (Prod α α)
    V₀_in : Membership.mem (uniformity α) V₀
    s : Set α
    hs : s.Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd := y })
    V₁ : Set (Prod α α)
    V₁_in : Membership.mem (uniformity α) V₁
    V₁_symm : SymmetricRel V₁
    h_comp : HasSubset.Subset (compRel V₁ V₁) V₀
    x : α
    hx : ∀ {V : Set (Prod α α)}, Membership.mem (uniformity α) V → (Inter.inter (U …
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    a✝ : SymmetricRel V
    z : α
    hz : Membership.mem (UniformSpace.ball x (Inter.inter V₁ V)) z
    hz' : Membership.mem s z
    hy : Membership.mem (UniformSpace.ball x V₁) z
    hy' : Membership.mem s z
    ⊢ Membership.mem V { fst := x, snd := z }
  -/
  exact ball_inter_right x _ _ hz
  /-
    🎉 no goals
  -/


theorem isClosed_range_of_spaced_out {ι} [T0Space α] {V₀ : Set (α × α)} (V₀_in : V₀ ∈ 𝓤 α)
    {f : ι → α} (hf : Pairwise fun x y => (f x, f y) ∉ V₀) : IsClosed (range f) :=
  isClosed_of_spaced_out V₀_in <| by
    /-
      α : Type u
      inst✝¹ : UniformSpace α
      ι : Type u_1
      inst✝ : T0Space α
      V₀ : Set (Prod α α)
      V₀_in : Membership.mem (uniformity α) V₀
      f : ι → α
      hf : Pairwise fun x y => Not (Membership.mem V₀ { fst := f x, snd := f y })
      ⊢ (Set.range f).Pairwise fun x y => Not (Membership.mem V₀ { fst := x, snd :=  …
    -/
    rintro _ ⟨x, rfl⟩ _ ⟨y, rfl⟩ h
    /-
      case intro.intro
      α : Type u
      inst✝¹ : UniformSpace α
      ι : Type u_1
      inst✝ : T0Space α
      V₀ : Set (Prod α α)
      V₀_in : Membership.mem (uniformity α) V₀
      f : ι → α
      hf : Pairwise fun x y => Not (Membership.mem V₀ { fst := f x, snd := f y })
      x y : ι
      h : Ne (f x) (f y)
      ⊢ Not (Membership.mem V₀ { fst := f x, snd := f y })
    -/
    exact hf (ne_of_apply_ne f h)
    /-
      🎉 no goals
    -/


theorem comap_map_mk_uniformity : comap (Prod.map mk mk) (map (Prod.map mk mk) (𝓤 α)) = 𝓤 α := by
  /-
    α : Type u
    inst✝ : UniformSpace α
    ⊢ Eq (Filter.comap (Prod.map SeparationQuotient.mk SeparationQuotient.mk) (Fil …
  -/
  refine le_antisymm ?_ le_comap_map
  /-
    α : Type u
    inst✝ : UniformSpace α
    ⊢ LE.le (Filter.comap (Prod.map SeparationQuotient.mk SeparationQuotient.mk) ( …
  -/
  refine ((((𝓤 α).basis_sets.map _).comap _).le_basis_iff uniformity_hasBasis_open).2 fun U hU ↦ ?_
  /-
    α : Type u
    inst✝ : UniformSpace α
    U : Set (Prod α α)
    hU : And (Membership.mem (uniformity α) U) (IsOpen U)
    ⊢ Exists fun i => And (Membership.mem (uniformity α) i) (HasSubset.Subset (Set …
  -/
  refine ⟨U, hU.1, fun (x₁, x₂) ⟨(y₁, y₂), hyU, hxy⟩ ↦ ?_⟩
  /-
    α : Type u
    inst✝ : UniformSpace α
    U : Set (Prod α α)
    hU : And (Membership.mem (uniformity α) U) (IsOpen U)
    x✝¹ : Prod α α
    x₁ x₂ : α
    x✝ : Membership.mem (Set.preimage (Prod.map SeparationQuotient.mk SeparationQu …
    y₁ y₂ : α
    hyU : Membership.mem (id U) { fst := y₁, snd := y₂ }
    hxy : Eq (Prod.map SeparationQuotient.mk SeparationQuotient.mk { fst := y₁, sn …
    ⊢ Membership.mem (id U) { fst := x₁, snd := x₂ }
  -/
  simp only [Prod.map, Prod.ext_iff, mk_eq_mk] at hxy
  /-
    α : Type u
    inst✝ : UniformSpace α
    U : Set (Prod α α)
    hU : And (Membership.mem (uniformity α) U) (IsOpen U)
    x✝¹ : Prod α α
    x₁ x₂ : α
    x✝ : Membership.mem (Set.preimage (Prod.map SeparationQuotient.mk SeparationQu …
    y₁ y₂ : α
    hyU : Membership.mem (id U) { fst := y₁, snd := y₂ }
    hxy : And (Inseparable y₁ x₁) (Inseparable y₂ x₂)
    ⊢ Membership.mem (id U) { fst := x₁, snd := x₂ }
  -/
  exact ((hxy.1.prod hxy.2).mem_open_iff hU.2).1 hyU
  /-
    🎉 no goals
  -/


instance instUniformSpace : UniformSpace (SeparationQuotient α) where
  uniformity := map (Prod.map mk mk) (𝓤 α)
  symm := tendsto_map' <| tendsto_map.comp tendsto_swap_uniformity
  comp := fun t ht ↦ by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      ⊢ Membership.mem ((Filter.map (Prod.map SeparationQuotient.mk SeparationQuotie …
    -/
    rcases comp_open_symm_mem_uniformity_sets ht with ⟨U, hU, hUo, -, hUt⟩
    /-
      case intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUo : IsOpen U
      hUt : HasSubset.Subset (compRel U U) (Set.preimage (Prod.map SeparationQuotien …
      ⊢ Membership.mem ((Filter.map (Prod.map SeparationQuotient.mk SeparationQuotie …
    -/
    refine mem_of_superset (mem_lift' <| image_mem_map hU) ?_
    /-
      case intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUo : IsOpen U
      hUt : HasSubset.Subset (compRel U U) (Set.preimage (Prod.map SeparationQuotien …
      ⊢ HasSubset.Subset (compRel (Set.image (Prod.map SeparationQuotient.mk Separat …
    -/
    simp only [subset_def, Prod.forall, mem_compRel, mem_image, Prod.ext_iff]
    /-
      case intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUo : IsOpen U
      hUt : HasSubset.Subset (compRel U U) (Set.preimage (Prod.map SeparationQuotien …
      ⊢ ∀ (a b : SeparationQuotient α), (Exists fun z => And (Exists fun x => And (M …
    -/
    rintro _ _ ⟨_, ⟨⟨x, y⟩, hxyU, rfl, rfl⟩, ⟨⟨y', z⟩, hyzU, hy, rfl⟩⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.mk.intro.intro.intro.mk.intro.i …
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUo : IsOpen U
      hUt : HasSubset.Subset (compRel U U) (Set.preimage (Prod.map SeparationQuotien …
      x y : α
      hxyU : Membership.mem U { fst := x, snd := y }
      y' z : α
      hyzU : Membership.mem U { fst := y', snd := z }
      hy : Eq (Prod.map SeparationQuotient.mk SeparationQuotient.mk { fst := y', snd …
      ⊢ Membership.mem t { fst := (Prod.map SeparationQuotient.mk SeparationQuotient …
    -/
    have : y' ⤳ y := (mk_eq_mk.1 hy).specializes
    /-
      case intro.intro.intro.intro.intro.intro.intro.mk.intro.intro.intro.mk.intro.i …
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      t : Set (Prod (SeparationQuotient α) (SeparationQuotient α))
      ht : Membership.mem (Filter.map (Prod.map SeparationQuotient.mk SeparationQuot …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUo : IsOpen U
      hUt : HasSubset.Subset (compRel U U) (Set.preimage (Prod.map SeparationQuotien …
      x y : α
      hxyU : Membership.mem U { fst := x, snd := y }
      y' z : α
      hyzU : Membership.mem U { fst := y', snd := z }
      hy : Eq (Prod.map SeparationQuotient.mk SeparationQuotient.mk { fst := y', snd …
      this : Specializes y' y
      ⊢ Membership.mem t { fst := (Prod.map SeparationQuotient.mk SeparationQuotient …
    -/
    exact @hUt (x, z) ⟨y', this.mem_open (UniformSpace.isOpen_ball _ hUo) hxyU, hyzU⟩
    /-
      🎉 no goals
    -/
  nhds_eq_comap_uniformity := surjective_mk.forall.2 fun x ↦ comap_injective surjective_mk <| by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      x : α
      ⊢ Eq (Filter.comap SeparationQuotient.mk (nhds (SeparationQuotient.mk x))) (Fi …
    -/
    conv_lhs => rw [comap_mk_nhds_mk, nhds_eq_comap_uniformity, ← comap_map_mk_uniformity]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : UniformSpace γ
      x : α
      ⊢ Eq (Filter.comap (Prod.mk x) (Filter.comap (Prod.map SeparationQuotient.mk S …
    -/
    simp only [Filter.comap_comap, Function.comp_def, Prod.map_apply]
    /-
      🎉 no goals
    -/


theorem uniformity_eq : 𝓤 (SeparationQuotient α) = (𝓤 α).map (Prod.map mk mk) := rfl


theorem uniformContinuous_mk : UniformContinuous (mk : α → SeparationQuotient α) :=
  le_rfl


theorem uniformContinuous_dom {f : SeparationQuotient α → β} :
    UniformContinuous f ↔ UniformContinuous (f ∘ mk) :=
  .rfl


theorem uniformContinuous_dom₂ {f : SeparationQuotient α × SeparationQuotient β → γ} :
    UniformContinuous f ↔ UniformContinuous fun p : α × β ↦ f (mk p.1, mk p.2) := by
  simp only [UniformContinuous, uniformity_prod_eq_prod, uniformity_eq, prod_map_map_eq,
    tendsto_map'_iff]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : Prod (SeparationQuotient α) (SeparationQuotient β) → γ
    ⊢ Iff (Filter.Tendsto (Function.comp (Function.comp (fun x => { fst := f x.1,  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem uniformContinuous_lift {f : α → β} (h : ∀ a b, Inseparable a b → f a = f b) :
    UniformContinuous (lift f h) ↔ UniformContinuous f :=
  .rfl


theorem uniformContinuous_uncurry_lift₂ {f : α → β → γ}
    (h : ∀ a c b d, Inseparable a b → Inseparable c d → f a c = f b d) :
    UniformContinuous (uncurry <| lift₂ f h) ↔ UniformContinuous (uncurry f) :=
  uniformContinuous_dom₂


theorem comap_mk_uniformity : (𝓤 (SeparationQuotient α)).comap (Prod.map mk mk) = 𝓤 α :=
  comap_map_mk_uniformity


open Classical in
/-- Factoring functions to a separated space through the separation quotient.

TODO: unify with `SeparationQuotient.lift`. -/
def lift' [T0Space β] (f : α → β) : SeparationQuotient α → β :=
  if hc : UniformContinuous f then lift f fun _ _ h => (h.map hc.continuous).eq
  else fun x => f (Nonempty.some ⟨x.out⟩)


theorem lift'_mk [T0Space β] {f : α → β} (h : UniformContinuous f) (a : α) :
                               /-
                                 α : Type u
                                 β : Type v
                                 inst✝² : UniformSpace α
                                 inst✝¹ : UniformSpace β
                                 inst✝ : T0Space β
                                 f : α → β
                                 h : UniformContinuous f
                                 a : α
                                 ⊢ Eq (SeparationQuotient.lift' f (SeparationQuotient.mk a)) (f a)
                               -/
    lift' f (mk a) = f a := by rw [lift', dif_pos h, lift_mk]
                               /-
                                 🎉 no goals
                               -/


theorem uniformContinuous_lift' [T0Space β] (f : α → β) : UniformContinuous (lift' f) := by
  /-
    α : Type u
    β : Type v
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : T0Space β
    f : α → β
    ⊢ UniformContinuous (SeparationQuotient.lift' f)
  -/
  by_cases hf : UniformContinuous f
    /-
      case pos
      α : Type u
      β : Type v
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : T0Space β
      f : α → β
      hf : UniformContinuous f
      ⊢ UniformContinuous (SeparationQuotient.lift' f)
    -/
  · rwa [lift', dif_pos hf, uniformContinuous_lift]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      β : Type v
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : T0Space β
      f : α → β
      hf : Not (UniformContinuous f)
      ⊢ UniformContinuous (SeparationQuotient.lift' f)
    -/
  · rw [lift', dif_neg hf]
    /-
      case neg
      α : Type u
      β : Type v
      inst✝² : UniformSpace α
      inst✝¹ : UniformSpace β
      inst✝ : T0Space β
      f : α → β
      hf : Not (UniformContinuous f)
      ⊢ UniformContinuous fun x => f ⋯.some
    -/
    exact uniformContinuous_of_const fun a _ => rfl
    /-
      🎉 no goals
    -/


/-- The separation quotient functor acting on functions. -/
def map (f : α → β) : SeparationQuotient α → SeparationQuotient β := lift' (mk ∘ f)


theorem map_mk {f : α → β} (h : UniformContinuous f) (a : α) : map f (mk a) = mk (f a) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    h : UniformContinuous f
    a : α
    ⊢ Eq (SeparationQuotient.map f (SeparationQuotient.mk a)) (SeparationQuotient. …
  -/
  rw [map, lift'_mk (uniformContinuous_mk.comp h)]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem uniformContinuous_map (f : α → β) : UniformContinuous (map f) :=
  uniformContinuous_lift' _


theorem map_unique {f : α → β} (hf : UniformContinuous f)
    {g : SeparationQuotient α → SeparationQuotient β} (comm : mk ∘ f = g ∘ mk) : map f = g := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    hf : UniformContinuous f
    g : SeparationQuotient α → SeparationQuotient β
    comm : Eq (Function.comp SeparationQuotient.mk f) (Function.comp g SeparationQ …
    ⊢ Eq (SeparationQuotient.map f) g
  -/
  ext ⟨a⟩
  calc
    map f ⟦a⟧ = ⟦f a⟧ := map_mk hf a
    _ = g ⟦a⟧ := congr_fun comm a


@[simp]
theorem map_id : map (@id α) = id := map_unique uniformContinuous_id rfl


theorem map_comp {f : α → β} {g : β → γ} (hf : UniformContinuous f) (hg : UniformContinuous g) :
    map g ∘ map f = map (g ∘ f) :=
                                 /-
                                   α : Type u
                                   β : Type v
                                   γ : Type w
                                   inst✝² : UniformSpace α
                                   inst✝¹ : UniformSpace β
                                   inst✝ : UniformSpace γ
                                   f : α → β
                                   g : β → γ
                                   hf : UniformContinuous f
                                   hg : UniformContinuous g
                                   ⊢ Eq (Function.comp SeparationQuotient.mk (Function.comp g f)) (Function.comp  …
                                 -/
  (map_unique (hg.comp hf) <| by simp only [Function.comp_def, map_mk, hf, hg]).symm
                                 /-
                                   🎉 no goals
                                 -/


