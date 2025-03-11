/-- The unique uniform structure inducing a given compact topological structure. -/
def uniformSpaceOfCompactT2 [TopologicalSpace γ] [CompactSpace γ] [T2Space γ] : UniformSpace γ where
  uniformity := 𝓝ˢ (diagonal γ)
  symm := continuous_swap.tendsto_nhdsSet fun _ => Eq.symm
  comp := by
    /-  This is the difficult part of the proof. We need to prove that, for each neighborhood `W`
        of the diagonal `Δ`, there exists a smaller neighborhood `V` such that `V ○ V ⊆ W`.
        -/
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      ⊢ LE.le ((nhdsSet (Set.diagonal γ)).lift' fun s => compRel s s) (nhdsSet (Set. …
    -/
    set 𝓝Δ := 𝓝ˢ (diagonal γ)
    -- The filter of neighborhoods of Δ
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      ⊢ LE.le (𝓝Δ.lift' fun s => compRel s s) 𝓝Δ
    -/
    set F := 𝓝Δ.lift' fun s : Set (γ × γ) => s ○ s
    -- Compositions of neighborhoods of Δ
    -- If this weren't true, then there would be V ∈ 𝓝Δ such that F ⊓ 𝓟 Vᶜ ≠ ⊥
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      ⊢ LE.le F 𝓝Δ
    -/
    rw [le_iff_forall_inf_principal_compl]
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      ⊢ ∀ (V : Set (Prod γ γ)), Membership.mem 𝓝Δ V → Eq (Min.min F (Filter.principa …
    -/
    intro V V_in
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      ⊢ Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot
    -/
    by_contra H
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      ⊢ False
    -/
    haveI : NeBot (F ⊓ 𝓟 Vᶜ) := ⟨H⟩
    -- Hence compactness would give us a cluster point (x, y) for F ⊓ 𝓟 Vᶜ
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      ⊢ False
    -/
    obtain ⟨⟨x, y⟩, hxy⟩ : ∃ p : γ × γ, ClusterPt p (F ⊓ 𝓟 Vᶜ) := exists_clusterPt_of_compactSpace _
    -- In particular (x, y) is a cluster point of 𝓟 Vᶜ, hence is not in the interior of V,
    -- and a fortiori not in Δ, so x ≠ y
    /-
      case intro.mk
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      ⊢ False
    -/
    have clV : ClusterPt (x, y) (𝓟 <| Vᶜ) := hxy.of_inf_right
    have : (x, y) ∉ interior V := by
      have : (x, y) ∈ closure Vᶜ := by rwa [mem_closure_iff_clusterPt]
      rwa [closure_compl] at this
    /-
      case intro.mk
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      ⊢ False
    -/
    have diag_subset : diagonal γ ⊆ interior V := subset_interior_iff_mem_nhdsSet.2 V_in
    /-
      case intro.mk
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      ⊢ False
    -/
    have x_ne_y : x ≠ y := mt (@diag_subset (x, y)) this
    -- Since γ is compact and Hausdorff, it is T₄, hence T₃.
    -- So there are closed neighborhoods V₁ and V₂ of x and y contained in
    -- disjoint open neighborhoods U₁ and U₂.
    obtain
      ⟨U₁, _, V₁, V₁_in, U₂, _, V₂, V₂_in, V₁_cl, V₂_cl, U₁_op, U₂_op, VU₁, VU₂, hU₁₂⟩ :=
      disjoint_nested_nhds x_ne_y
    -- We set U₃ := (V₁ ∪ V₂)ᶜ so that W := U₁ ×ˢ U₁ ∪ U₂ ×ˢ U₂ ∪ U₃ ×ˢ U₃ is an open
    -- neighborhood of Δ.
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      ⊢ False
    -/
    let U₃ := (V₁ ∪ V₂)ᶜ
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      ⊢ False
    -/
    have U₃_op : IsOpen U₃ := (V₁_cl.union V₂_cl).isOpen_compl
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      U₃_op : IsOpen U₃
      ⊢ False
    -/
    let W := U₁ ×ˢ U₁ ∪ U₂ ×ˢ U₂ ∪ U₃ ×ˢ U₃
    have W_in : W ∈ 𝓝Δ := by
      rw [mem_nhdsSet_iff_forall]
      rintro ⟨z, z'⟩ (rfl : z = z')
      refine IsOpen.mem_nhds ?_ ?_
      · apply_rules [IsOpen.union, IsOpen.prod]
      · simp only [W, mem_union, mem_prod, and_self_iff]
        exact (_root_.em _).imp_left fun h => union_subset_union VU₁ VU₂ h
    -- So W ○ W ∈ F by definition of F
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      U₃_op : IsOpen U₃
      W : Set (Prod γ γ) := Union.union (Union.union (SProd.sprod U₁ U₁) (SProd.spro …
      W_in : Membership.mem 𝓝Δ W
      ⊢ False
    -/
    have : W ○ W ∈ F := @mem_lift' _ _ _ (fun s => s ○ s) _ W_in
      -- Porting note: was `by simpa only using mem_lift' W_in`
    -- And V₁ ×ˢ V₂ ∈ 𝓝 (x, y)
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝¹ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this✝ : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      U₃_op : IsOpen U₃
      W : Set (Prod γ γ) := Union.union (Union.union (SProd.sprod U₁ U₁) (SProd.spro …
      W_in : Membership.mem 𝓝Δ W
      this : Membership.mem F (compRel W W)
      ⊢ False
    -/
    have hV₁₂ : V₁ ×ˢ V₂ ∈ 𝓝 (x, y) := prod_mem_nhds V₁_in V₂_in
    -- But (x, y) is also a cluster point of F so (V₁ ×ˢ V₂) ∩ (W ○ W) ≠ ∅
    -- However the construction of W implies (V₁ ×ˢ V₂) ∩ (W ○ W) = ∅.
    -- Indeed assume for contradiction there is some (u, v) in the intersection.
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝¹ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this✝ : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      U₃_op : IsOpen U₃
      W : Set (Prod γ γ) := Union.union (Union.union (SProd.sprod U₁ U₁) (SProd.spro …
      W_in : Membership.mem 𝓝Δ W
      this : Membership.mem F (compRel W W)
      hV₁₂ : Membership.mem (nhds { fst := x, snd := y }) (SProd.sprod V₁ V₂)
      ⊢ False
    -/
    obtain ⟨⟨u, v⟩, ⟨u_in, v_in⟩, w, huw, hwv⟩ := clusterPt_iff.mp hxy.of_inf_left hV₁₂ this
    -- So u ∈ V₁, v ∈ V₂, and there exists some w such that (u, w) ∈ W and (w ,v) ∈ W.
    -- Because u is in V₁ which is disjoint from U₂ and U₃, (u, w) ∈ W forces (u, w) ∈ U₁ ×ˢ U₁.
    have uw_in : (u, w) ∈ U₁ ×ˢ U₁ :=
      (huw.resolve_right fun h => h.1 <| Or.inl u_in).resolve_right fun h =>
        hU₁₂.le_bot ⟨VU₁ u_in, h.1⟩
    -- Similarly, because v ∈ V₂, (w ,v) ∈ W forces (w, v) ∈ U₂ ×ˢ U₂.
    have wv_in : (w, v) ∈ U₂ ×ˢ U₂ :=
      (hwv.resolve_right fun h => h.2 <| Or.inr v_in).resolve_left fun h =>
        hU₁₂.le_bot ⟨h.2, VU₂ v_in⟩
    -- Hence w ∈ U₁ ∩ U₂ which is empty.
    -- So we have a contradiction
    /-
      case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      𝓝Δ : Filter (Prod γ γ) := nhdsSet (Set.diagonal γ)
      F : Filter (Prod γ γ) := 𝓝Δ.lift' fun s => compRel s s
      V : Set (Prod γ γ)
      V_in : Membership.mem 𝓝Δ V
      H : Not (Eq (Min.min F (Filter.principal (HasCompl.compl V))) Bot.bot)
      this✝¹ : (Min.min F (Filter.principal (HasCompl.compl V))).NeBot
      x y : γ
      hxy : ClusterPt { fst := x, snd := y } (Min.min F (Filter.principal (HasCompl. …
      clV : ClusterPt { fst := x, snd := y } (Filter.principal (HasCompl.compl V))
      this✝ : Not (Membership.mem (interior V) { fst := x, snd := y })
      diag_subset : HasSubset.Subset (Set.diagonal γ) (interior V)
      x_ne_y : Ne x y
      U₁ : Set γ
      left✝¹ : Membership.mem (nhds x) U₁
      V₁ : Set γ
      V₁_in : Membership.mem (nhds x) V₁
      U₂ : Set γ
      left✝ : Membership.mem (nhds y) U₂
      V₂ : Set γ
      V₂_in : Membership.mem (nhds y) V₂
      V₁_cl : IsClosed V₁
      V₂_cl : IsClosed V₂
      U₁_op : IsOpen U₁
      U₂_op : IsOpen U₂
      VU₁ : HasSubset.Subset V₁ U₁
      VU₂ : HasSubset.Subset V₂ U₂
      hU₁₂ : Disjoint U₁ U₂
      U₃ : Set γ := HasCompl.compl (Union.union V₁ V₂)
      U₃_op : IsOpen U₃
      W : Set (Prod γ γ) := Union.union (Union.union (SProd.sprod U₁ U₁) (SProd.spro …
      W_in : Membership.mem 𝓝Δ W
      this : Membership.mem F (compRel W W)
      hV₁₂ : Membership.mem (nhds { fst := x, snd := y }) (SProd.sprod V₁ V₂)
      u v : γ
      u_in : Membership.mem V₁ { fst := u, snd := v }.1
      v_in : Membership.mem V₂ { fst := u, snd := v }.2
      w : γ
      huw : Membership.mem W { fst := { fst := u, snd := v }.1, snd := w }
      hwv : Membership.mem W { fst := w, snd := { fst := u, snd := v }.2 }
      uw_in : Membership.mem (SProd.sprod U₁ U₁) { fst := u, snd := w }
      wv_in : Membership.mem (SProd.sprod U₂ U₂) { fst := w, snd := v }
      ⊢ False
    -/
    exact hU₁₂.le_bot ⟨uw_in.2, wv_in.1⟩
    /-
      🎉 no goals
    -/
  nhds_eq_comap_uniformity x := by
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      x : γ
      ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) (nhdsSet (Set.diagonal γ)))
    -/
    simp_rw [nhdsSet_diagonal, comap_iSup, nhds_prod_eq, comap_prod, Function.comp_def, comap_id']
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      x : γ
      ⊢ Eq (nhds x) (iSup fun i => Min.min (Filter.comap (fun x_1 => x) (nhds i)) (n …
    -/
    rw [iSup_split_single _ x, comap_const_of_mem fun V => mem_of_mem_nhds]
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      x : γ
      ⊢ Eq (nhds x) (Max.max (Min.min Top.top (nhds x)) (iSup fun i => iSup fun x_1  …
    -/
    suffices ∀ y ≠ x, comap (fun _ : γ ↦ x) (𝓝 y) ⊓ 𝓝 y ≤ 𝓝 x by simpa
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      x : γ
      ⊢ ∀ (y : γ), Ne y x → LE.le (Min.min (Filter.comap (fun x_1 => x) (nhds y)) (n …
    -/
    intro y hxy
    /-
      γ : Type u_1
      inst✝² : TopologicalSpace γ
      inst✝¹ : CompactSpace γ
      inst✝ : T2Space γ
      x y : γ
      hxy : Ne y x
      ⊢ LE.le (Min.min (Filter.comap (fun x_1 => x) (nhds y)) (nhds y)) (nhds x)
    -/
    simp [comap_const_of_not_mem (compl_singleton_mem_nhds hxy) (not_not_intro rfl)]
    /-
      🎉 no goals
    -/

