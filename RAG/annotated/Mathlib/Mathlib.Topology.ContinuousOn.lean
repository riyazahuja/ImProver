@[simp]
theorem nhds_bind_nhdsWithin {a : α} {s : Set α} : ((𝓝 a).bind fun x => 𝓝[s] x) = 𝓝[s] a :=
  bind_inf_principal.trans <| congr_arg₂ _ nhds_bind_nhds rfl


@[simp]
theorem eventually_nhds_nhdsWithin {a : α} {s : Set α} {p : α → Prop} :
    (∀ᶠ y in 𝓝 a, ∀ᶠ x in 𝓝[s] y, p x) ↔ ∀ᶠ x in 𝓝[s] a, p x :=
  Filter.ext_iff.1 nhds_bind_nhdsWithin { x | p x }


theorem eventually_nhdsWithin_iff {a : α} {s : Set α} {p : α → Prop} :
    (∀ᶠ x in 𝓝[s] a, p x) ↔ ∀ᶠ x in 𝓝 a, x ∈ s → p x :=
  eventually_inf_principal


theorem frequently_nhdsWithin_iff {z : α} {s : Set α} {p : α → Prop} :
    (∃ᶠ x in 𝓝[s] z, p x) ↔ ∃ᶠ x in 𝓝 z, p x ∧ x ∈ s :=
                                       /-
                                         α : Type u_1
                                         inst✝ : TopologicalSpace α
                                         z : α
                                         s : Set α
                                         p : α → Prop
                                         ⊢ Iff (Filter.Frequently (fun x => And (Membership.mem s x) (p x)) (nhds z)) ( …
                                       -/
  frequently_inf_principal.trans <| by simp only [and_comm]
                                       /-
                                         🎉 no goals
                                       -/


theorem mem_closure_ne_iff_frequently_within {z : α} {s : Set α} :
    z ∈ closure (s \ {z}) ↔ ∃ᶠ x in 𝓝[≠] z, x ∈ s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    z : α
    s : Set α
    ⊢ Iff (Membership.mem (closure (SDiff.sdiff s (Singleton.singleton z))) z) (Fi …
  -/
  simp [mem_closure_iff_frequently, frequently_nhdsWithin_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem eventually_eventually_nhdsWithin {a : α} {s : Set α} {p : α → Prop} :
    (∀ᶠ y in 𝓝[s] a, ∀ᶠ x in 𝓝[s] y, p x) ↔ ∀ᶠ x in 𝓝[s] a, p x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    p : α → Prop
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => p x) (nhdsWithi …
  -/
  refine ⟨fun h => ?_, fun h => (eventually_nhds_nhdsWithin.2 h).filter_mono inf_le_left⟩
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    p : α → Prop
    h : Filter.Eventually (fun y => Filter.Eventually (fun x => p x) (nhdsWithin y …
    ⊢ Filter.Eventually (fun x => p x) (nhdsWithin a s)
  -/
  simp only [eventually_nhdsWithin_iff] at h ⊢
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    p : α → Prop
    h : Filter.Eventually (fun x => Membership.mem s x → Filter.Eventually (fun x  …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (nhds a)
  -/
  exact h.mono fun x hx hxs => (hx hxs).self_of_nhds hxs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-04")]
alias eventually_nhdsWithin_nhdsWithin := eventually_eventually_nhdsWithin


@[simp]
theorem eventually_mem_nhdsWithin_iff {x : α} {s t : Set α} :
    (∀ᶠ x' in 𝓝[s] x, t ∈ 𝓝[s] x') ↔ t ∈ 𝓝[s] x :=
  eventually_eventually_nhdsWithin


theorem nhdsWithin_eq (a : α) (s : Set α) :
    𝓝[s] a = ⨅ t ∈ { t : Set α | a ∈ t ∧ IsOpen t }, 𝓟 (t ∩ s) :=
  ((nhds_basis_opens a).inf_principal s).eq_biInf


@[simp] lemma nhdsWithin_univ (a : α) : 𝓝[Set.univ] a = 𝓝 a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    ⊢ Eq (nhdsWithin a Set.univ) (nhds a)
  -/
  rw [nhdsWithin, principal_univ, inf_top_eq]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_hasBasis {p : β → Prop} {s : β → Set α} {a : α} (h : (𝓝 a).HasBasis p s)
    (t : Set α) : (𝓝[t] a).HasBasis p fun i => s i ∩ t :=
  h.inf_principal t


theorem nhdsWithin_basis_open (a : α) (t : Set α) :
    (𝓝[t] a).HasBasis (fun u => a ∈ u ∧ IsOpen u) fun u => u ∩ t :=
  nhdsWithin_hasBasis (nhds_basis_opens a) t


theorem mem_nhdsWithin {t : Set α} {a : α} {s : Set α} :
    t ∈ 𝓝[s] a ↔ ∃ u, IsOpen u ∧ a ∈ u ∧ u ∩ s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    t : Set α
    a : α
    s : Set α
    ⊢ Iff (Membership.mem (nhdsWithin a s) t) (Exists fun u => And (IsOpen u) (And …
  -/
  simpa only [and_assoc, and_left_comm] using (nhdsWithin_basis_open a s).mem_iff
  /-
    🎉 no goals
  -/


theorem mem_nhdsWithin_iff_exists_mem_nhds_inter {t : Set α} {a : α} {s : Set α} :
    t ∈ 𝓝[s] a ↔ ∃ u ∈ 𝓝 a, u ∩ s ⊆ t :=
  (nhdsWithin_hasBasis (𝓝 a).basis_sets s).mem_iff


theorem diff_mem_nhdsWithin_compl {x : α} {s : Set α} (hs : s ∈ 𝓝 x) (t : Set α) :
    s \ t ∈ 𝓝[tᶜ] x :=
  diff_mem_inf_principal_compl hs t


theorem diff_mem_nhdsWithin_diff {x : α} {s t : Set α} (hs : s ∈ 𝓝[t] x) (t' : Set α) :
    s \ t' ∈ 𝓝[t \ t'] x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s t : Set α
    hs : Membership.mem (nhdsWithin x t) s
    t' : Set α
    ⊢ Membership.mem (nhdsWithin x (SDiff.sdiff t t')) (SDiff.sdiff s t')
  -/
  rw [nhdsWithin, diff_eq, diff_eq, ← inf_principal, ← inf_assoc]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s t : Set α
    hs : Membership.mem (nhdsWithin x t) s
    t' : Set α
    ⊢ Membership.mem (Min.min (Min.min (nhds x) (Filter.principal t)) (Filter.prin …
  -/
  exact inter_mem_inf hs (mem_principal_self _)
  /-
    🎉 no goals
  -/


theorem nhds_of_nhdsWithin_of_nhds {s t : Set α} {a : α} (h1 : s ∈ 𝓝 a) (h2 : t ∈ 𝓝[s] a) :
    t ∈ 𝓝 a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    a : α
    h1 : Membership.mem (nhds a) s
    h2 : Membership.mem (nhdsWithin a s) t
    ⊢ Membership.mem (nhds a) t
  -/
  rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.mp h2 with ⟨_, Hw, hw⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    a : α
    h1 : Membership.mem (nhds a) s
    h2 : Membership.mem (nhdsWithin a s) t
    w✝ : Set α
    Hw : Membership.mem (nhds a) w✝
    hw : HasSubset.Subset (Inter.inter w✝ s) t
    ⊢ Membership.mem (nhds a) t
  -/
  exact (𝓝 a).sets_of_superset ((𝓝 a).inter_sets Hw h1) hw
  /-
    🎉 no goals
  -/


theorem mem_nhdsWithin_iff_eventually {s t : Set α} {x : α} :
    t ∈ 𝓝[s] x ↔ ∀ᶠ y in 𝓝 x, y ∈ s → y ∈ t :=
  eventually_inf_principal


theorem mem_nhdsWithin_iff_eventuallyEq {s t : Set α} {x : α} :
    t ∈ 𝓝[s] x ↔ s =ᶠ[𝓝 x] (s ∩ t : Set α) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    x : α
    ⊢ Iff (Membership.mem (nhdsWithin x s) t) ((nhds x).EventuallyEq s (Inter.inte …
  -/
  simp_rw [mem_nhdsWithin_iff_eventually, eventuallyEq_set, mem_inter_iff, iff_self_and]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_eq_iff_eventuallyEq {s t : Set α} {x : α} : 𝓝[s] x = 𝓝[t] x ↔ s =ᶠ[𝓝 x] t :=
  set_eventuallyEq_iff_inf_principal.symm


theorem nhdsWithin_le_iff {s t : Set α} {x : α} : 𝓝[s] x ≤ 𝓝[t] x ↔ t ∈ 𝓝[s] x :=
  set_eventuallyLE_iff_inf_principal_le.symm.trans set_eventuallyLE_iff_mem_inf_principal

-- Porting note: golfed, dropped an unneeded assumption

theorem preimage_nhdsWithin_coinduced' {π : α → β} {s : Set β} {t : Set α} {a : α} (h : a ∈ t)
    (hs : s ∈ @nhds β (.coinduced (fun x : t => π x) inferInstance) (π a)) :
    π ⁻¹' s ∈ 𝓝[t] a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    t : Set α
    a : α
    h : Membership.mem t a
    hs : Membership.mem (nhds (π a)) s
    ⊢ Membership.mem (nhdsWithin a t) (Set.preimage π s)
  -/
  lift a to t using h
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    t : Set α
    a : Subtype fun x => Membership.mem t x
    hs : Membership.mem (nhds (π ↑a)) s
    ⊢ Membership.mem (nhdsWithin (↑a) t) (Set.preimage π s)
  -/
  replace hs : (fun x : t => π x) ⁻¹' s ∈ 𝓝 a := preimage_nhds_coinduced hs
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    t : Set α
    a : Subtype fun x => Membership.mem t x
    hs : Membership.mem (nhds a) (Set.preimage (fun x => π ↑x) s)
    ⊢ Membership.mem (nhdsWithin (↑a) t) (Set.preimage π s)
  -/
  rwa [← map_nhds_subtype_val, mem_map]
  /-
    🎉 no goals
  -/


theorem mem_nhdsWithin_of_mem_nhds {s t : Set α} {a : α} (h : s ∈ 𝓝 a) : s ∈ 𝓝[t] a :=
  mem_inf_of_left h


theorem self_mem_nhdsWithin {a : α} {s : Set α} : s ∈ 𝓝[s] a :=
  mem_inf_of_right (mem_principal_self s)


theorem eventually_mem_nhdsWithin {a : α} {s : Set α} : ∀ᶠ x in 𝓝[s] a, x ∈ s :=
  self_mem_nhdsWithin


theorem inter_mem_nhdsWithin (s : Set α) {t : Set α} {a : α} (h : t ∈ 𝓝 a) : s ∩ t ∈ 𝓝[s] a :=
  inter_mem self_mem_nhdsWithin (mem_inf_of_left h)


theorem pure_le_nhdsWithin {a : α} {s : Set α} (ha : a ∈ s) : pure a ≤ 𝓝[s] a :=
  le_inf (pure_le_nhds a) (le_principal_iff.2 ha)


theorem mem_of_mem_nhdsWithin {a : α} {s t : Set α} (ha : a ∈ s) (ht : t ∈ 𝓝[s] a) : a ∈ t :=
  pure_le_nhdsWithin ha ht


theorem Filter.Eventually.self_of_nhdsWithin {p : α → Prop} {s : Set α} {x : α}
    (h : ∀ᶠ y in 𝓝[s] x, p y) (hx : x ∈ s) : p x :=
  mem_of_mem_nhdsWithin hx h


theorem tendsto_const_nhdsWithin {l : Filter β} {s : Set α} {a : α} (ha : a ∈ s) :
    Tendsto (fun _ : β => a) l (𝓝[s] a) :=
  tendsto_const_pure.mono_right <| pure_le_nhdsWithin ha


theorem nhdsWithin_restrict'' {a : α} (s : Set α) {t : Set α} (h : t ∈ 𝓝[s] a) :
    𝓝[s] a = 𝓝[s ∩ t] a :=
  le_antisymm (le_inf inf_le_left (le_principal_iff.mpr (inter_mem self_mem_nhdsWithin h)))
    (inf_le_inf_left _ (principal_mono.mpr Set.inter_subset_left))


theorem nhdsWithin_restrict' {a : α} (s : Set α) {t : Set α} (h : t ∈ 𝓝 a) : 𝓝[s] a = 𝓝[s ∩ t] a :=
  nhdsWithin_restrict'' s <| mem_inf_of_left h


theorem nhdsWithin_restrict {a : α} (s : Set α) {t : Set α} (h₀ : a ∈ t) (h₁ : IsOpen t) :
    𝓝[s] a = 𝓝[s ∩ t] a :=
  nhdsWithin_restrict' s (IsOpen.mem_nhds h₁ h₀)


theorem nhdsWithin_le_of_mem {a : α} {s t : Set α} (h : s ∈ 𝓝[t] a) : 𝓝[t] a ≤ 𝓝[s] a :=
  nhdsWithin_le_iff.mpr h


theorem nhdsWithin_le_nhds {a : α} {s : Set α} : 𝓝[s] a ≤ 𝓝 a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    ⊢ LE.le (nhdsWithin a s) (nhds a)
  -/
  rw [← nhdsWithin_univ]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    ⊢ LE.le (nhdsWithin a s) (nhdsWithin a Set.univ)
  -/
  apply nhdsWithin_le_of_mem
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    ⊢ Membership.mem (nhdsWithin a s) Set.univ
  -/
  exact univ_mem
  /-
    🎉 no goals
  -/


theorem nhdsWithin_eq_nhdsWithin' {a : α} {s t u : Set α} (hs : s ∈ 𝓝 a) (h₂ : t ∩ s = u ∩ s) :
                          /-
                            α : Type u_1
                            inst✝ : TopologicalSpace α
                            a : α
                            s t u : Set α
                            hs : Membership.mem (nhds a) s
                            h₂ : Eq (Inter.inter t s) (Inter.inter u s)
                            ⊢ Eq (nhdsWithin a t) (nhdsWithin a u)
                          -/
    𝓝[t] a = 𝓝[u] a := by rw [nhdsWithin_restrict' t hs, nhdsWithin_restrict' u hs, h₂]
                          /-
                            🎉 no goals
                          -/


theorem nhdsWithin_eq_nhdsWithin {a : α} {s t u : Set α} (h₀ : a ∈ s) (h₁ : IsOpen s)
    (h₂ : t ∩ s = u ∩ s) : 𝓝[t] a = 𝓝[u] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t u : Set α
    h₀ : Membership.mem s a
    h₁ : IsOpen s
    h₂ : Eq (Inter.inter t s) (Inter.inter u s)
    ⊢ Eq (nhdsWithin a t) (nhdsWithin a u)
  -/
  rw [nhdsWithin_restrict t h₀ h₁, nhdsWithin_restrict u h₀ h₁, h₂]
  /-
    🎉 no goals
  -/


@[simp] theorem nhdsWithin_eq_nhds {a : α} {s : Set α} : 𝓝[s] a = 𝓝 a ↔ s ∈ 𝓝 a :=
  inf_eq_left.trans le_principal_iff


theorem IsOpen.nhdsWithin_eq {a : α} {s : Set α} (h : IsOpen s) (ha : a ∈ s) : 𝓝[s] a = 𝓝 a :=
  nhdsWithin_eq_nhds.2 <| h.mem_nhds ha


theorem preimage_nhds_within_coinduced {π : α → β} {s : Set β} {t : Set α} {a : α} (h : a ∈ t)
    (ht : IsOpen t)
    (hs : s ∈ @nhds β (.coinduced (fun x : t => π x) inferInstance) (π a)) :
    π ⁻¹' s ∈ 𝓝 a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    t : Set α
    a : α
    h : Membership.mem t a
    ht : IsOpen t
    hs : Membership.mem (nhds (π a)) s
    ⊢ Membership.mem (nhds a) (Set.preimage π s)
  -/
  rw [← ht.nhdsWithin_eq h]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    t : Set α
    a : α
    h : Membership.mem t a
    ht : IsOpen t
    hs : Membership.mem (nhds (π a)) s
    ⊢ Membership.mem (nhdsWithin a t) (Set.preimage π s)
  -/
  exact preimage_nhdsWithin_coinduced' h hs
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : TopologicalSpace α
                                                      a : α
                                                      ⊢ Eq (nhdsWithin a EmptyCollection.emptyCollection) Bot.bot
                                                    -/
theorem nhdsWithin_empty (a : α) : 𝓝[∅] a = ⊥ := by rw [nhdsWithin, principal_empty, inf_bot_eq]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem nhdsWithin_union (a : α) (s t : Set α) : 𝓝[s ∪ t] a = 𝓝[s] a ⊔ 𝓝[t] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (nhdsWithin a (Union.union s t)) (Max.max (nhdsWithin a s) (nhdsWithin a  …
  -/
  delta nhdsWithin
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (Min.min (nhds a) (Filter.principal (Union.union s t))) (Max.max (Min.min …
  -/
  rw [← inf_sup_left, sup_principal]
  /-
    🎉 no goals
  -/


theorem nhds_eq_nhdsWithin_sup_nhdsWithin (b : α) {I₁ I₂ : Set α} (hI : Set.univ = I₁ ∪ I₂) :
    nhds b = nhdsWithin b I₁ ⊔ nhdsWithin b I₂ := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    b : α
    I₁ I₂ : Set α
    hI : Eq Set.univ (Union.union I₁ I₂)
    ⊢ Eq (nhds b) (Max.max (nhdsWithin b I₁) (nhdsWithin b I₂))
  -/
  rw [← nhdsWithin_univ b, hI, nhdsWithin_union]
  /-
    🎉 no goals
  -/


/-- If `L` and `R` are neighborhoods of `b` within sets whose union is `Set.univ`, then
`L ∪ R` is a neighborhood of `b`. -/
theorem union_mem_nhds_of_mem_nhdsWithin {b : α}
    {I₁ I₂ : Set α} (h : Set.univ = I₁ ∪ I₂)
    {L : Set α} (hL : L ∈ nhdsWithin b I₁)
    {R : Set α} (hR : R ∈ nhdsWithin b I₂) : L ∪ R ∈ nhds b := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    b : α
    I₁ I₂ : Set α
    h : Eq Set.univ (Union.union I₁ I₂)
    L : Set α
    hL : Membership.mem (nhdsWithin b I₁) L
    R : Set α
    hR : Membership.mem (nhdsWithin b I₂) R
    ⊢ Membership.mem (nhds b) (Union.union L R)
  -/
  rw [← nhdsWithin_univ b, h, nhdsWithin_union]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    b : α
    I₁ I₂ : Set α
    h : Eq Set.univ (Union.union I₁ I₂)
    L : Set α
    hL : Membership.mem (nhdsWithin b I₁) L
    R : Set α
    hR : Membership.mem (nhdsWithin b I₂) R
    ⊢ Membership.mem (Max.max (nhdsWithin b I₁) (nhdsWithin b I₂)) (Union.union L R)
  -/
  exact ⟨mem_of_superset hL (by aesop), mem_of_superset hR (by aesop)⟩
  /-
    🎉 no goals
  -/



/-- Writing a punctured neighborhood filter as a sup of left and right filters. -/
lemma punctured_nhds_eq_nhdsWithin_sup_nhdsWithin [LinearOrder α] {x : α} :
    𝓝[≠] x = 𝓝[<] x ⊔ 𝓝[>] x := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    x : α
    ⊢ Eq (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Max.max (nhdsWit …
  -/
  rw [← Iio_union_Ioi, nhdsWithin_union]
  /-
    🎉 no goals
  -/



/-- Obtain a "predictably-sided" neighborhood of `b` from two one-sided neighborhoods. -/
theorem nhds_of_Ici_Iic [LinearOrder α] {b : α}
    {L : Set α} (hL : L ∈ 𝓝[≤] b)
    {R : Set α} (hR : R ∈ 𝓝[≥] b) : L ∩ Iic b ∪ R ∩ Ici b ∈ 𝓝 b :=
  union_mem_nhds_of_mem_nhdsWithin Iic_union_Ici.symm
    (inter_mem hL self_mem_nhdsWithin) (inter_mem hR self_mem_nhdsWithin)


theorem nhdsWithin_biUnion {ι} {I : Set ι} (hI : I.Finite) (s : ι → Set α) (a : α) :
    𝓝[⋃ i ∈ I, s i] a = ⨆ i ∈ I, 𝓝[s i] a :=
                                 /-
                                   α : Type u_1
                                   inst✝ : TopologicalSpace α
                                   ι : Type u_5
                                   I : Set ι
                                   hI : I.Finite
                                   s : ι → Set α
                                   a : α
                                   ⊢ Eq (nhdsWithin a (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun i  …
                                 -/
  Set.Finite.induction_on hI (by simp) fun _ _ hT ↦ by
                                 /-
                                   🎉 no goals
                                 -/
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      ι : Type u_5
      I : Set ι
      hI : I.Finite
      s : ι → Set α
      a : α
      a✝ : ι
      s✝ : Set ι
      x✝¹ : Not (Membership.mem s✝ a✝)
      x✝ : s✝.Finite
      hT : Eq (nhdsWithin a (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun …
      ⊢ Eq (nhdsWithin a (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun i  …
    -/
    simp only [hT, nhdsWithin_union, iSup_insert, biUnion_insert]
    /-
      🎉 no goals
    -/


theorem nhdsWithin_sUnion {S : Set (Set α)} (hS : S.Finite) (a : α) :
    𝓝[⋃₀ S] a = ⨆ s ∈ S, 𝓝[s] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : S.Finite
    a : α
    ⊢ Eq (nhdsWithin a S.sUnion) (iSup fun s => iSup fun h => nhdsWithin a s)
  -/
  rw [sUnion_eq_biUnion, nhdsWithin_biUnion hS]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_iUnion {ι} [Finite ι] (s : ι → Set α) (a : α) :
    𝓝[⋃ i, s i] a = ⨆ i, 𝓝[s i] a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    ι : Sort u_5
    inst✝ : Finite ι
    s : ι → Set α
    a : α
    ⊢ Eq (nhdsWithin a (Set.iUnion fun i => s i)) (iSup fun i => nhdsWithin a (s i))
  -/
  rw [← sUnion_range, nhdsWithin_sUnion (finite_range s), iSup_range]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_inter (a : α) (s t : Set α) : 𝓝[s ∩ t] a = 𝓝[s] a ⊓ 𝓝[t] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (nhdsWithin a (Inter.inter s t)) (Min.min (nhdsWithin a s) (nhdsWithin a  …
  -/
  delta nhdsWithin
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (Min.min (nhds a) (Filter.principal (Inter.inter s t))) (Min.min (Min.min …
  -/
  rw [inf_left_comm, inf_assoc, inf_principal, ← inf_assoc, inf_idem]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_inter' (a : α) (s t : Set α) : 𝓝[s ∩ t] a = 𝓝[s] a ⊓ 𝓟 t := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (nhdsWithin a (Inter.inter s t)) (Min.min (nhdsWithin a s) (Filter.princi …
  -/
  delta nhdsWithin
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Eq (Min.min (nhds a) (Filter.principal (Inter.inter s t))) (Min.min (Min.min …
  -/
  rw [← inf_principal, inf_assoc]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_inter_of_mem {a : α} {s t : Set α} (h : s ∈ 𝓝[t] a) : 𝓝[s ∩ t] a = 𝓝[t] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    h : Membership.mem (nhdsWithin a t) s
    ⊢ Eq (nhdsWithin a (Inter.inter s t)) (nhdsWithin a t)
  -/
  rw [nhdsWithin_inter, inf_eq_right]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    h : Membership.mem (nhdsWithin a t) s
    ⊢ LE.le (nhdsWithin a t) (nhdsWithin a s)
  -/
  exact nhdsWithin_le_of_mem h
  /-
    🎉 no goals
  -/


theorem nhdsWithin_inter_of_mem' {a : α} {s t : Set α} (h : t ∈ 𝓝[s] a) : 𝓝[s ∩ t] a = 𝓝[s] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    h : Membership.mem (nhdsWithin a s) t
    ⊢ Eq (nhdsWithin a (Inter.inter s t)) (nhdsWithin a s)
  -/
  rw [inter_comm, nhdsWithin_inter_of_mem h]
  /-
    🎉 no goals
  -/


@[simp]
theorem nhdsWithin_singleton (a : α) : 𝓝[{a}] a = pure a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    ⊢ Eq (nhdsWithin a (Singleton.singleton a)) (Pure.pure a)
  -/
  rw [nhdsWithin, principal_singleton, inf_eq_right.2 (pure_le_nhds a)]
  /-
    🎉 no goals
  -/


@[simp]
theorem nhdsWithin_insert (a : α) (s : Set α) : 𝓝[insert a s] a = pure a ⊔ 𝓝[s] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s : Set α
    ⊢ Eq (nhdsWithin a (Insert.insert a s)) (Max.max (Pure.pure a) (nhdsWithin a s))
  -/
  rw [← singleton_union, nhdsWithin_union, nhdsWithin_singleton]
  /-
    🎉 no goals
  -/


theorem mem_nhdsWithin_insert {a : α} {s t : Set α} : t ∈ 𝓝[insert a s] a ↔ a ∈ t ∧ t ∈ 𝓝[s] a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    s t : Set α
    ⊢ Iff (Membership.mem (nhdsWithin a (Insert.insert a s)) t) (And (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem insert_mem_nhdsWithin_insert {a : α} {s t : Set α} (h : t ∈ 𝓝[s] a) :
                                       /-
                                         α : Type u_1
                                         inst✝ : TopologicalSpace α
                                         a : α
                                         s t : Set α
                                         h : Membership.mem (nhdsWithin a s) t
                                         ⊢ Membership.mem (nhdsWithin a (Insert.insert a s)) (Insert.insert a t)
                                       -/
    insert a t ∈ 𝓝[insert a s] a := by simp [mem_of_superset h]
                                       /-
                                         🎉 no goals
                                       -/


theorem insert_mem_nhds_iff {a : α} {s : Set α} : insert a s ∈ 𝓝 a ↔ s ∈ 𝓝[≠] a := by
  simp only [nhdsWithin, mem_inf_principal, mem_compl_iff, mem_singleton_iff, or_iff_not_imp_left,
    insert_def]


@[simp]
theorem nhdsWithin_compl_singleton_sup_pure (a : α) : 𝓝[≠] a ⊔ pure a = 𝓝 a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) (Pure.pu …
  -/
  rw [← nhdsWithin_singleton, ← nhdsWithin_union, compl_union_self, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_prod {α : Type*} [TopologicalSpace α] {β : Type*} [TopologicalSpace β]
    {s u : Set α} {t v : Set β} {a : α} {b : β} (hu : u ∈ 𝓝[s] a) (hv : v ∈ 𝓝[t] b) :
    u ×ˢ v ∈ 𝓝[s ×ˢ t] (a, b) := by
  /-
    α : Type u_5
    inst✝¹ : TopologicalSpace α
    β : Type u_6
    inst✝ : TopologicalSpace β
    s u : Set α
    t v : Set β
    a : α
    b : β
    hu : Membership.mem (nhdsWithin a s) u
    hv : Membership.mem (nhdsWithin b t) v
    ⊢ Membership.mem (nhdsWithin { fst := a, snd := b } (SProd.sprod s t)) (SProd. …
  -/
  rw [nhdsWithin_prod_eq]
  /-
    α : Type u_5
    inst✝¹ : TopologicalSpace α
    β : Type u_6
    inst✝ : TopologicalSpace β
    s u : Set α
    t v : Set β
    a : α
    b : β
    hu : Membership.mem (nhdsWithin a s) u
    hv : Membership.mem (nhdsWithin b t) v
    ⊢ Membership.mem (SProd.sprod (nhdsWithin a s) (nhdsWithin b t)) (SProd.sprod  …
  -/
  exact prod_mem_prod hu hv
  /-
    🎉 no goals
  -/


lemma Filter.EventuallyEq.mem_interior {x : α} {s t : Set α} (hst : s =ᶠ[𝓝 x] t)
    (h : x ∈ interior s) : x ∈ interior t := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s t : Set α
    hst : (nhds x).EventuallyEq s t
    h : Membership.mem (interior s) x
    ⊢ Membership.mem (interior t) x
  -/
  rw [← nhdsWithin_eq_iff_eventuallyEq] at hst
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s t : Set α
    hst : Eq (nhdsWithin x s) (nhdsWithin x t)
    h : Membership.mem (interior s) x
    ⊢ Membership.mem (interior t) x
  -/
  simpa [mem_interior_iff_mem_nhds, ← nhdsWithin_eq_nhds, hst] using h
  /-
    🎉 no goals
  -/


lemma Filter.EventuallyEq.mem_interior_iff {x : α} {s t : Set α} (hst : s =ᶠ[𝓝 x] t) :
    x ∈ interior s ↔ x ∈ interior t :=
  ⟨fun h ↦ hst.mem_interior h, fun h ↦ hst.symm.mem_interior h⟩


@[deprecated (since := "2024-11-11")]
alias EventuallyEq.mem_interior_iff := Filter.EventuallyEq.mem_interior_iff


theorem nhdsWithin_pi_eq' {I : Set ι} (hI : I.Finite) (s : ∀ i, Set (π i)) (x : ∀ i, π i) :
    𝓝[pi I s] x = ⨅ i, comap (fun x => x i) (𝓝 (x i) ⊓ ⨅ (_ : i ∈ I), 𝓟 (s i)) := by
  simp only [nhdsWithin, nhds_pi, Filter.pi, comap_inf, comap_iInf, pi_def, comap_principal, ←
    iInf_principal_finite hI, ← iInf_inf_eq]


theorem nhdsWithin_pi_eq {I : Set ι} (hI : I.Finite) (s : ∀ i, Set (π i)) (x : ∀ i, π i) :
    𝓝[pi I s] x =
      (⨅ i ∈ I, comap (fun x => x i) (𝓝[s i] x i)) ⊓
        ⨅ (i) (_ : i ∉ I), comap (fun x => x i) (𝓝 (x i)) := by
  simp only [nhdsWithin, nhds_pi, Filter.pi, pi_def, ← iInf_principal_finite hI, comap_inf,
    comap_principal, eval]
  /-
    ι : Type u_5
    π : ι → Type u_6
    inst✝ : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    hI : I.Finite
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    ⊢ Eq (Min.min (iInf fun i => Filter.comap (Function.eval i) (nhds (x i))) (iIn …
  -/
  rw [iInf_split _ fun i => i ∈ I, inf_right_comm]
  /-
    ι : Type u_5
    π : ι → Type u_6
    inst✝ : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    hI : I.Finite
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    ⊢ Eq (Min.min (Min.min (iInf fun i => iInf fun x_1 => Filter.comap (Function.e …
  -/
  simp only [iInf_inf_eq]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_pi_univ_eq [Finite ι] (s : ∀ i, Set (π i)) (x : ∀ i, π i) :
    𝓝[pi univ s] x = ⨅ i, comap (fun x => x i) (𝓝[s i] x i) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : Finite ι
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    ⊢ Eq (nhdsWithin x (Set.univ.pi s)) (iInf fun i => Filter.comap (fun x => x i) …
  -/
  simpa [nhdsWithin] using nhdsWithin_pi_eq finite_univ s x
  /-
    🎉 no goals
  -/


theorem nhdsWithin_pi_eq_bot {I : Set ι} {s : ∀ i, Set (π i)} {x : ∀ i, π i} :
    𝓝[pi I s] x = ⊥ ↔ ∃ i ∈ I, 𝓝[s i] x i = ⊥ := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    inst✝ : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    ⊢ Iff (Eq (nhdsWithin x (I.pi s)) Bot.bot) (Exists fun i => And (Membership.me …
  -/
  simp only [nhdsWithin, nhds_pi, pi_inf_principal_pi_eq_bot]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_pi_neBot {I : Set ι} {s : ∀ i, Set (π i)} {x : ∀ i, π i} :
    (𝓝[pi I s] x).NeBot ↔ ∀ i ∈ I, (𝓝[s i] x i).NeBot := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    inst✝ : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    ⊢ Iff (nhdsWithin x (I.pi s)).NeBot (∀ (i : ι), Membership.mem I i → (nhdsWith …
  -/
  simp [neBot_iff, nhdsWithin_pi_eq_bot]
  /-
    🎉 no goals
  -/


instance instNeBotNhdsWithinUnivPi {s : ∀ i, Set (π i)} {x : ∀ i, π i}
    [∀ i, (𝓝[s i] x i).NeBot] : (𝓝[pi univ s] x).NeBot := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    ι : Type u_5
    π : ι → Type u_6
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    x : (i : ι) → π i
    inst✝ : ∀ (i : ι), (nhdsWithin (x i) (s i)).NeBot
    ⊢ (nhdsWithin x (Set.univ.pi s)).NeBot
  -/
  simpa [nhdsWithin_pi_neBot]
  /-
    🎉 no goals
  -/


instance Pi.instNeBotNhdsWithinIio [Nonempty ι] [∀ i, Preorder (π i)] {x : ∀ i, π i}
    [∀ i, (𝓝[<] x i).NeBot] : (𝓝[<] x).NeBot :=
  have : (𝓝[pi univ fun i ↦ Iio (x i)] x).NeBot := inferInstance
  this.mono <| nhdsWithin_mono _ fun _y hy ↦ lt_of_strongLT fun i ↦ hy i trivial


instance Pi.instNeBotNhdsWithinIoi [Nonempty ι] [∀ i, Preorder (π i)] {x : ∀ i, π i}
    [∀ i, (𝓝[>] x i).NeBot] : (𝓝[>] x).NeBot :=
  Pi.instNeBotNhdsWithinIio (π := fun i ↦ (π i)ᵒᵈ) (x := fun i ↦ OrderDual.toDual (x i))


theorem Filter.Tendsto.piecewise_nhdsWithin {f g : α → β} {t : Set α} [∀ x, Decidable (x ∈ t)]
    {a : α} {s : Set α} {l : Filter β} (h₀ : Tendsto f (𝓝[s ∩ t] a) l)
    (h₁ : Tendsto g (𝓝[s ∩ tᶜ] a) l) : Tendsto (piecewise t f g) (𝓝[s] a) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    f g : α → β
    t : Set α
    inst✝ : (x : α) → Decidable (Membership.mem t x)
    a : α
    s : Set α
    l : Filter β
    h₀ : Filter.Tendsto f (nhdsWithin a (Inter.inter s t)) l
    h₁ : Filter.Tendsto g (nhdsWithin a (Inter.inter s (HasCompl.compl t))) l
    ⊢ Filter.Tendsto (t.piecewise f g) (nhdsWithin a s) l
  -/
                              /-
                                🎉 no goals
                              -/
  apply Tendsto.piecewise <;> rwa [← nhdsWithin_inter']
                              /-
                                🎉 no goals
                              -/


theorem Filter.Tendsto.if_nhdsWithin {f g : α → β} {p : α → Prop} [DecidablePred p] {a : α}
    {s : Set α} {l : Filter β} (h₀ : Tendsto f (𝓝[s ∩ { x | p x }] a) l)
    (h₁ : Tendsto g (𝓝[s ∩ { x | ¬p x }] a) l) :
    Tendsto (fun x => if p x then f x else g x) (𝓝[s] a) l :=
  h₀.piecewise_nhdsWithin h₁


theorem map_nhdsWithin (f : α → β) (a : α) (s : Set α) :
    map f (𝓝[s] a) = ⨅ t ∈ { t : Set α | a ∈ t ∧ IsOpen t }, 𝓟 (f '' (t ∩ s)) :=
  ((nhdsWithin_basis_open a s).map f).eq_biInf


theorem tendsto_nhdsWithin_mono_left {f : α → β} {a : α} {s t : Set α} {l : Filter β} (hst : s ⊆ t)
    (h : Tendsto f (𝓝[t] a) l) : Tendsto f (𝓝[s] a) l :=
  h.mono_left <| nhdsWithin_mono a hst


theorem tendsto_nhdsWithin_mono_right {f : β → α} {l : Filter β} {a : α} {s t : Set α} (hst : s ⊆ t)
    (h : Tendsto f l (𝓝[s] a)) : Tendsto f l (𝓝[t] a) :=
  h.mono_right (nhdsWithin_mono a hst)


theorem tendsto_nhdsWithin_of_tendsto_nhds {f : α → β} {a : α} {s : Set α} {l : Filter β}
    (h : Tendsto f (𝓝 a) l) : Tendsto f (𝓝[s] a) l :=
  h.mono_left inf_le_left


theorem eventually_mem_of_tendsto_nhdsWithin {f : β → α} {a : α} {s : Set α} {l : Filter β}
    (h : Tendsto f l (𝓝[s] a)) : ∀ᶠ i in l, f i ∈ s := by
  simp_rw [nhdsWithin_eq, tendsto_iInf, mem_setOf_eq, tendsto_principal, mem_inter_iff,
    eventually_and] at h
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    f : β → α
    a : α
    s : Set α
    l : Filter β
    h : ∀ (i : Set α), And (Membership.mem i a) (IsOpen i) → And (Filter.Eventuall …
    ⊢ Filter.Eventually (fun i => Membership.mem s (f i)) l
  -/
  exact (h univ ⟨mem_univ a, isOpen_univ⟩).2
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_of_tendsto_nhdsWithin {f : β → α} {a : α} {s : Set α} {l : Filter β}
    (h : Tendsto f l (𝓝[s] a)) : Tendsto f l (𝓝 a) :=
  h.mono_right nhdsWithin_le_nhds


theorem nhdsWithin_neBot_of_mem {s : Set α} {x : α} (hx : x ∈ s) : NeBot (𝓝[s] x) :=
  mem_closure_iff_nhdsWithin_neBot.1 <| subset_closure hx


theorem IsClosed.mem_of_nhdsWithin_neBot {s : Set α} (hs : IsClosed s) {x : α}
    (hx : NeBot <| 𝓝[s] x) : x ∈ s :=
  hs.closure_eq ▸ mem_closure_iff_nhdsWithin_neBot.2 hx


theorem DenseRange.nhdsWithin_neBot {ι : Type*} {f : ι → α} (h : DenseRange f) (x : α) :
    NeBot (𝓝[range f] x) :=
  mem_closure_iff_clusterPt.1 (h x)


theorem mem_closure_pi {ι : Type*} {α : ι → Type*} [∀ i, TopologicalSpace (α i)] {I : Set ι}
    {s : ∀ i, Set (α i)} {x : ∀ i, α i} : x ∈ closure (pi I s) ↔ ∀ i ∈ I, x i ∈ closure (s i) := by
  /-
    ι : Type u_5
    α : ι → Type u_6
    inst✝ : (i : ι) → TopologicalSpace (α i)
    I : Set ι
    s : (i : ι) → Set (α i)
    x : (i : ι) → α i
    ⊢ Iff (Membership.mem (closure (I.pi s)) x) (∀ (i : ι), Membership.mem I i → M …
  -/
  simp only [mem_closure_iff_nhdsWithin_neBot, nhdsWithin_pi_neBot]
  /-
    🎉 no goals
  -/


theorem closure_pi_set {ι : Type*} {α : ι → Type*} [∀ i, TopologicalSpace (α i)] (I : Set ι)
    (s : ∀ i, Set (α i)) : closure (pi I s) = pi I fun i => closure (s i) :=
  Set.ext fun _ => mem_closure_pi


theorem dense_pi {ι : Type*} {α : ι → Type*} [∀ i, TopologicalSpace (α i)] {s : ∀ i, Set (α i)}
    (I : Set ι) (hs : ∀ i ∈ I, Dense (s i)) : Dense (pi I s) := by
  simp only [dense_iff_closure_eq, closure_pi_set, pi_congr rfl fun i hi => (hs i hi).closure_eq,
    pi_univ]


theorem eventuallyEq_nhdsWithin_iff {f g : α → β} {s : Set α} {a : α} :
    f =ᶠ[𝓝[s] a] g ↔ ∀ᶠ x in 𝓝 a, x ∈ s → f x = g x :=
  mem_inf_principal


theorem eventuallyEq_nhdsWithin_of_eqOn {f g : α → β} {s : Set α} {a : α} (h : EqOn f g s) :
    f =ᶠ[𝓝[s] a] g :=
  mem_inf_of_right h


theorem Set.EqOn.eventuallyEq_nhdsWithin {f g : α → β} {s : Set α} {a : α} (h : EqOn f g s) :
    f =ᶠ[𝓝[s] a] g :=
  eventuallyEq_nhdsWithin_of_eqOn h


theorem tendsto_nhdsWithin_congr {f g : α → β} {s : Set α} {a : α} {l : Filter β}
    (hfg : ∀ x ∈ s, f x = g x) (hf : Tendsto f (𝓝[s] a) l) : Tendsto g (𝓝[s] a) l :=
  (tendsto_congr' <| eventuallyEq_nhdsWithin_of_eqOn hfg).1 hf


theorem eventually_nhdsWithin_of_forall {s : Set α} {a : α} {p : α → Prop} (h : ∀ x ∈ s, p x) :
    ∀ᶠ x in 𝓝[s] a, p x :=
  mem_inf_of_right h


theorem tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within {a : α} {l : Filter β} {s : Set α}
    (f : β → α) (h1 : Tendsto f l (𝓝 a)) (h2 : ∀ᶠ x in l, f x ∈ s) : Tendsto f l (𝓝[s] a) :=
  tendsto_inf.2 ⟨h1, tendsto_principal.2 h2⟩


theorem tendsto_nhdsWithin_iff {a : α} {l : Filter β} {s : Set α} {f : β → α} :
    Tendsto f l (𝓝[s] a) ↔ Tendsto f l (𝓝 a) ∧ ∀ᶠ n in l, f n ∈ s :=
  ⟨fun h => ⟨tendsto_nhds_of_tendsto_nhdsWithin h, eventually_mem_of_tendsto_nhdsWithin h⟩, fun h =>
    tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ h.1 h.2⟩


@[simp]
theorem tendsto_nhdsWithin_range {a : α} {l : Filter β} {f : β → α} :
    Tendsto f l (𝓝[range f] a) ↔ Tendsto f l (𝓝 a) :=
  ⟨fun h => h.mono_right inf_le_left, fun h =>
    tendsto_inf.2 ⟨h, tendsto_principal.2 <| Eventually.of_forall mem_range_self⟩⟩


theorem Filter.EventuallyEq.eq_of_nhdsWithin {s : Set α} {f g : α → β} {a : α} (h : f =ᶠ[𝓝[s] a] g)
    (hmem : a ∈ s) : f a = g a :=
  h.self_of_nhdsWithin hmem


theorem eventually_nhdsWithin_of_eventually_nhds {α : Type*} [TopologicalSpace α] {s : Set α}
    {a : α} {p : α → Prop} (h : ∀ᶠ x in 𝓝 a, p x) : ∀ᶠ x in 𝓝[s] a, p x :=
  mem_nhdsWithin_of_mem_nhds h


lemma Set.MapsTo.preimage_mem_nhdsWithin {f : α → β} {s : Set α} {t : Set β} {x : α}
    (hst : MapsTo f s t) : f ⁻¹' t ∈ 𝓝[s] x :=
  Filter.mem_of_superset self_mem_nhdsWithin hst


theorem mem_nhdsWithin_subtype {s : Set α} {a : { x // x ∈ s }} {t u : Set { x // x ∈ s }} :
    t ∈ 𝓝[u] a ↔ t ∈ comap ((↑) : s → α) (𝓝[(↑) '' u] a) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    a : Subtype fun x => Membership.mem s x
    t u : Set (Subtype fun x => Membership.mem s x)
    ⊢ Iff (Membership.mem (nhdsWithin a u) t) (Membership.mem (Filter.comap Subtyp …
  -/
  rw [nhdsWithin, nhds_subtype, principal_subtype, ← comap_inf, ← nhdsWithin]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_subtype (s : Set α) (a : { x // x ∈ s }) (t : Set { x // x ∈ s }) :
    𝓝[t] a = comap ((↑) : s → α) (𝓝[(↑) '' t] a) :=
  Filter.ext fun _ => mem_nhdsWithin_subtype


theorem nhdsWithin_eq_map_subtype_coe {s : Set α} {a : α} (h : a ∈ s) :
    𝓝[s] a = map ((↑) : s → α) (𝓝 ⟨a, h⟩) :=
  (map_nhds_subtype_val ⟨a, h⟩).symm


theorem mem_nhds_subtype_iff_nhdsWithin {s : Set α} {a : s} {t : Set s} :
    t ∈ 𝓝 a ↔ (↑) '' t ∈ 𝓝[s] (a : α) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    a : ↑s
    t : Set ↑s
    ⊢ Iff (Membership.mem (nhds a) t) (Membership.mem (nhdsWithin (↑a) s) (Set.ima …
  -/
  rw [← map_nhds_subtype_val, image_mem_map_iff Subtype.val_injective]
  /-
    🎉 no goals
  -/


theorem preimage_coe_mem_nhds_subtype {s t : Set α} {a : s} : (↑) ⁻¹' t ∈ 𝓝 a ↔ t ∈ 𝓝[s] ↑a := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    a : ↑s
    ⊢ Iff (Membership.mem (nhds a) (Set.preimage Subtype.val t)) (Membership.mem ( …
  -/
  rw [← map_nhds_subtype_val, mem_map]
  /-
    🎉 no goals
  -/


theorem eventually_nhds_subtype_iff (s : Set α) (a : s) (P : α → Prop) :
    (∀ᶠ x : s in 𝓝 a, P x) ↔ ∀ᶠ x in 𝓝[s] a, P x :=
  preimage_coe_mem_nhds_subtype


theorem frequently_nhds_subtype_iff (s : Set α) (a : s) (P : α → Prop) :
    (∃ᶠ x : s in 𝓝 a, P x) ↔ ∃ᶠ x in 𝓝[s] a, P x :=
  eventually_nhds_subtype_iff s a (¬ P ·) |>.not


theorem tendsto_nhdsWithin_iff_subtype {s : Set α} {a : α} (h : a ∈ s) (f : α → β) (l : Filter β) :
    Tendsto f (𝓝[s] a) l ↔ Tendsto (s.restrict f) (𝓝 ⟨a, h⟩) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    s : Set α
    a : α
    h : Membership.mem s a
    f : α → β
    l : Filter β
    ⊢ Iff (Filter.Tendsto f (nhdsWithin a s) l) (Filter.Tendsto (s.restrict f) (nh …
  -/
  rw [nhdsWithin_eq_map_subtype_coe h, tendsto_map'_iff]; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- If a function is continuous within `s` at `x`, then it tends to `f x` within `s` by definition.
We register this fact for use with the dot notation, especially to use `Filter.Tendsto.comp` as
`ContinuousWithinAt.comp` will have a different meaning. -/
theorem ContinuousWithinAt.tendsto (h : ContinuousWithinAt f s x) :
    Tendsto f (𝓝[s] x) (𝓝 (f x)) :=
  h


theorem continuousWithinAt_univ (f : α → β) (x : α) :
    ContinuousWithinAt f Set.univ x ↔ ContinuousAt f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    x : α
    ⊢ Iff (ContinuousWithinAt f Set.univ x) (ContinuousAt f x)
  -/
  rw [ContinuousAt, ContinuousWithinAt, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem continuous_iff_continuousOn_univ {f : α → β} : Continuous f ↔ ContinuousOn f univ := by
  simp [continuous_iff_continuousAt, ContinuousOn, ContinuousAt, ContinuousWithinAt,
    nhdsWithin_univ]


theorem continuousWithinAt_iff_continuousAt_restrict (f : α → β) {x : α} {s : Set α} (h : x ∈ s) :
    ContinuousWithinAt f s x ↔ ContinuousAt (s.restrict f) ⟨x, h⟩ :=
  tendsto_nhdsWithin_iff_subtype h f _


theorem ContinuousWithinAt.tendsto_nhdsWithin {t : Set β}
    (h : ContinuousWithinAt f s x) (ht : MapsTo f s t) :
    Tendsto f (𝓝[s] x) (𝓝[t] f x) :=
  tendsto_inf.2 ⟨h, tendsto_principal.2 <| mem_inf_of_right <| mem_principal.2 <| ht⟩


theorem ContinuousWithinAt.tendsto_nhdsWithin_image (h : ContinuousWithinAt f s x) :
    Tendsto f (𝓝[s] x) (𝓝[f '' s] f x) :=
  h.tendsto_nhdsWithin (mapsTo_image _ _)


theorem nhdsWithin_le_comap (ctsf : ContinuousWithinAt f s x) :
    𝓝[s] x ≤ comap f (𝓝[f '' s] f x) :=
  ctsf.tendsto_nhdsWithin_image.le_comap


theorem ContinuousWithinAt.preimage_mem_nhdsWithin {t : Set β}
    (h : ContinuousWithinAt f s x) (ht : t ∈ 𝓝 (f x)) : f ⁻¹' t ∈ 𝓝[s] x :=
  h ht


theorem ContinuousWithinAt.preimage_mem_nhdsWithin' {t : Set β}
    (h : ContinuousWithinAt f s x) (ht : t ∈ 𝓝[f '' s] f x) : f ⁻¹' t ∈ 𝓝[s] x :=
  h.tendsto_nhdsWithin (mapsTo_image _ _) ht


theorem ContinuousWithinAt.preimage_mem_nhdsWithin'' {y : β} {s t : Set β}
    (h : ContinuousWithinAt f (f ⁻¹' s) x) (ht : t ∈ 𝓝[s] y) (hxy : y = f x) :
    f ⁻¹' t ∈ 𝓝[f ⁻¹' s] x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    x : α
    y : β
    s t : Set β
    h : ContinuousWithinAt f (Set.preimage f s) x
    ht : Membership.mem (nhdsWithin y s) t
    hxy : Eq y (f x)
    ⊢ Membership.mem (nhdsWithin x (Set.preimage f s)) (Set.preimage f t)
  -/
  rw [hxy] at ht
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    x : α
    y : β
    s t : Set β
    h : ContinuousWithinAt f (Set.preimage f s) x
    ht : Membership.mem (nhdsWithin (f x) s) t
    hxy : Eq y (f x)
    ⊢ Membership.mem (nhdsWithin x (Set.preimage f s)) (Set.preimage f t)
  -/
  exact h.preimage_mem_nhdsWithin' (nhdsWithin_mono _ (image_preimage_subset f s) ht)
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_of_not_mem_closure (hx : x ∉ closure s) :
    ContinuousWithinAt f s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    x : α
    hx : Not (Membership.mem (closure s) x)
    ⊢ ContinuousWithinAt f s x
  -/
  rw [mem_closure_iff_nhdsWithin_neBot, not_neBot] at hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    x : α
    hx : Eq (nhdsWithin x s) Bot.bot
    ⊢ ContinuousWithinAt f s x
  -/
  rw [ContinuousWithinAt, hx]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    x : α
    hx : Eq (nhdsWithin x s) Bot.bot
    ⊢ Filter.Tendsto f Bot.bot (nhds (f x))
  -/
  exact tendsto_bot
  /-
    🎉 no goals
  -/


theorem continuousOn_iff :
    ContinuousOn f s ↔
      ∀ x ∈ s, ∀ t : Set β, IsOpen t → f x ∈ t → ∃ u, IsOpen u ∧ x ∈ u ∧ u ∩ s ⊆ f ⁻¹' t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    ⊢ Iff (ContinuousOn f s) (∀ (x : α), Membership.mem s x → ∀ (t : Set β), IsOpe …
  -/
  simp only [ContinuousOn, ContinuousWithinAt, tendsto_nhds, mem_nhdsWithin]
  /-
    🎉 no goals
  -/


theorem ContinuousOn.continuousWithinAt (hf : ContinuousOn f s) (hx : x ∈ s) :
    ContinuousWithinAt f s x :=
  hf x hx


theorem continuousOn_iff_continuous_restrict :
    ContinuousOn f s ↔ Continuous (s.restrict f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    ⊢ Iff (ContinuousOn f s) (Continuous (s.restrict f))
  -/
  rw [ContinuousOn, continuous_iff_continuousAt]; constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      ⊢ (∀ (x : α), Membership.mem s x → ContinuousWithinAt f s x) → ∀ (x : ↑s), Con …
    -/
  · rintro h ⟨x, xs⟩
    /-
      case mp.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ContinuousWithinAt f s x
      x : α
      xs : Membership.mem s x
      ⊢ ContinuousAt (s.restrict f) ⟨x, xs⟩
    -/
    exact (continuousWithinAt_iff_continuousAt_restrict f xs).mp (h x xs)
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    ⊢ (∀ (x : ↑s), ContinuousAt (s.restrict f) x) → ∀ (x : α), Membership.mem s x  …
  -/
  intro h x xs
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ∀ (x : ↑s), ContinuousAt (s.restrict f) x
    x : α
    xs : Membership.mem s x
    ⊢ ContinuousWithinAt f s x
  -/
  exact (continuousWithinAt_iff_continuousAt_restrict f xs).mpr (h ⟨x, xs⟩)
  /-
    🎉 no goals
  -/

-- Porting note: 2 new lemmas

alias ⟨ContinuousOn.restrict, _⟩ := continuousOn_iff_continuous_restrict


theorem ContinuousOn.restrict_mapsTo {t : Set β} (hf : ContinuousOn f s) (ht : MapsTo f s t) :
    Continuous (ht.restrict f s t) :=
  hf.restrict.codRestrict _


theorem continuousOn_iff' :
    ContinuousOn f s ↔ ∀ t : Set β, IsOpen t → ∃ u, IsOpen u ∧ f ⁻¹' t ∩ s = u ∩ s := by
  have : ∀ t, IsOpen (s.restrict f ⁻¹' t) ↔ ∃ u : Set α, IsOpen u ∧ f ⁻¹' t ∩ s = u ∩ s := by
    intro t
    rw [isOpen_induced_iff, Set.restrict_eq, Set.preimage_comp]
    simp only [Subtype.preimage_coe_eq_preimage_coe_iff]
    constructor <;>
      · rintro ⟨u, ou, useq⟩
        exact ⟨u, ou, by simpa only [Set.inter_comm, eq_comm] using useq⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    this : ∀ (t : Set β), Iff (IsOpen (Set.preimage (s.restrict f) t)) (Exists fun …
    ⊢ Iff (ContinuousOn f s) (∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpe …
  -/
  rw [continuousOn_iff_continuous_restrict, continuous_def]; simp only [this]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- If a function is continuous on a set for some topologies, then it is
continuous on the same set with respect to any finer topology on the source space. -/
theorem ContinuousOn.mono_dom {α β : Type*} {t₁ t₂ : TopologicalSpace α} {t₃ : TopologicalSpace β}
    (h₁ : t₂ ≤ t₁) {s : Set α} {f : α → β} (h₂ : @ContinuousOn α β t₁ t₃ f s) :
    @ContinuousOn α β t₂ t₃ f s := fun x hx _u hu =>
  map_mono (inf_le_inf_right _ <| nhds_mono h₁) (h₂ x hx hu)


/-- If a function is continuous on a set for some topologies, then it is
continuous on the same set with respect to any coarser topology on the target space. -/
theorem ContinuousOn.mono_rng {α β : Type*} {t₁ : TopologicalSpace α} {t₂ t₃ : TopologicalSpace β}
    (h₁ : t₂ ≤ t₃) {s : Set α} {f : α → β} (h₂ : @ContinuousOn α β t₁ t₂ f s) :
    @ContinuousOn α β t₁ t₃ f s := fun x hx _u hu =>
  h₂ x hx <| nhds_mono h₁ hu


theorem continuousOn_iff_isClosed :
    ContinuousOn f s ↔ ∀ t : Set β, IsClosed t → ∃ u, IsClosed u ∧ f ⁻¹' t ∩ s = u ∩ s := by
  have : ∀ t, IsClosed (s.restrict f ⁻¹' t) ↔ ∃ u : Set α, IsClosed u ∧ f ⁻¹' t ∩ s = u ∩ s := by
    intro t
    rw [isClosed_induced_iff, Set.restrict_eq, Set.preimage_comp]
    simp only [Subtype.preimage_coe_eq_preimage_coe_iff, eq_comm, Set.inter_comm s]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    this : ∀ (t : Set β), Iff (IsClosed (Set.preimage (s.restrict f) t)) (Exists f …
    ⊢ Iff (ContinuousOn f s) (∀ (t : Set β), IsClosed t → Exists fun u => And (IsC …
  -/
  rw [continuousOn_iff_continuous_restrict, continuous_iff_isClosed]; simp only [this]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem continuous_of_cover_nhds {ι : Sort*} {s : ι → Set α}
    (hs : ∀ x : α, ∃ i, s i ∈ 𝓝 x) (hf : ∀ i, ContinuousOn f (s i)) :
    Continuous f :=
  continuous_iff_continuousAt.mpr fun x ↦ let ⟨i, hi⟩ := hs x; by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Sort u_5
      s : ι → Set α
      hs : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (s i)
      hf : ∀ (i : ι), ContinuousOn f (s i)
      x : α
      i : ι
      hi : Membership.mem (nhds x) (s i)
      ⊢ ContinuousAt f x
    -/
    rw [ContinuousAt, ← nhdsWithin_eq_nhds.2 hi]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Sort u_5
      s : ι → Set α
      hs : ∀ (x : α), Exists fun i => Membership.mem (nhds x) (s i)
      hf : ∀ (i : ι), ContinuousOn f (s i)
      x : α
      i : ι
      hi : Membership.mem (nhds x) (s i)
      ⊢ Filter.Tendsto f (nhdsWithin x (s i)) (nhds (f x))
    -/
    exact hf _ _ (mem_of_mem_nhds hi)
    /-
      🎉 no goals
    -/


@[simp] theorem continuousOn_empty (f : α → β) : ContinuousOn f ∅ := fun _ => False.elim


@[simp]
theorem continuousOn_singleton (f : α → β) (a : α) : ContinuousOn f {a} :=
  forall_eq.2 <| by
    simpa only [ContinuousWithinAt, nhdsWithin_singleton, tendsto_pure_left] using fun s =>
      mem_of_mem_nhds


theorem Set.Subsingleton.continuousOn {s : Set α} (hs : s.Subsingleton) (f : α → β) :
    ContinuousOn f s :=
  hs.induction_on (continuousOn_empty f) (continuousOn_singleton f)


theorem continuousOn_open_iff (hs : IsOpen s) :
    ContinuousOn f s ↔ ∀ t, IsOpen t → IsOpen (s ∩ f ⁻¹' t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hs : IsOpen s
    ⊢ Iff (ContinuousOn f s) (∀ (t : Set β), IsOpen t → IsOpen (Inter.inter s (Set …
  -/
  rw [continuousOn_iff']
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hs : IsOpen s
    ⊢ Iff (∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpen u) (Eq (Inter.int …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      ⊢ (∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpen u) (Eq (Inter.inter ( …
    -/
  · intro h t ht
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      h : ∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpen u) (Eq (Inter.inter  …
      t : Set β
      ht : IsOpen t
      ⊢ IsOpen (Inter.inter s (Set.preimage f t))
    -/
    rcases h t ht with ⟨u, u_open, hu⟩
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      h : ∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpen u) (Eq (Inter.inter  …
      t : Set β
      ht : IsOpen t
      u : Set α
      u_open : IsOpen u
      hu : Eq (Inter.inter (Set.preimage f t) s) (Inter.inter u s)
      ⊢ IsOpen (Inter.inter s (Set.preimage f t))
    -/
    rw [inter_comm, hu]
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      h : ∀ (t : Set β), IsOpen t → Exists fun u => And (IsOpen u) (Eq (Inter.inter  …
      t : Set β
      ht : IsOpen t
      u : Set α
      u_open : IsOpen u
      hu : Eq (Inter.inter (Set.preimage f t) s) (Inter.inter u s)
      ⊢ IsOpen (Inter.inter u s)
    -/
    apply IsOpen.inter u_open hs
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      ⊢ (∀ (t : Set β), IsOpen t → IsOpen (Inter.inter s (Set.preimage f t))) → ∀ (t …
    -/
  · intro h t ht
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      h : ∀ (t : Set β), IsOpen t → IsOpen (Inter.inter s (Set.preimage f t))
      t : Set β
      ht : IsOpen t
      ⊢ Exists fun u => And (IsOpen u) (Eq (Inter.inter (Set.preimage f t) s) (Inter …
    -/
    refine ⟨s ∩ f ⁻¹' t, h t ht, ?_⟩
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      hs : IsOpen s
      h : ∀ (t : Set β), IsOpen t → IsOpen (Inter.inter s (Set.preimage f t))
      t : Set β
      ht : IsOpen t
      ⊢ Eq (Inter.inter (Set.preimage f t) s) (Inter.inter (Inter.inter s (Set.preim …
    -/
    rw [@inter_comm _ s (f ⁻¹' t), inter_assoc, inter_self]
    /-
      🎉 no goals
    -/


theorem ContinuousOn.isOpen_inter_preimage {t : Set β}
    (hf : ContinuousOn f s) (hs : IsOpen s) (ht : IsOpen t) : IsOpen (s ∩ f ⁻¹' t) :=
  (continuousOn_open_iff hs).1 hf t ht


theorem ContinuousOn.isOpen_preimage {t : Set β} (h : ContinuousOn f s)
    (hs : IsOpen s) (hp : f ⁻¹' t ⊆ s) (ht : IsOpen t) : IsOpen (f ⁻¹' t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    t : Set β
    h : ContinuousOn f s
    hs : IsOpen s
    hp : HasSubset.Subset (Set.preimage f t) s
    ht : IsOpen t
    ⊢ IsOpen (Set.preimage f t)
  -/
  convert (continuousOn_open_iff hs).mp h t ht
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    t : Set β
    h : ContinuousOn f s
    hs : IsOpen s
    hp : HasSubset.Subset (Set.preimage f t) s
    ht : IsOpen t
    ⊢ Eq (Set.preimage f t) (Inter.inter s (Set.preimage f t))
  -/
  rw [inter_comm, inter_eq_self_of_subset_left hp]
  /-
    🎉 no goals
  -/


theorem ContinuousOn.preimage_isClosed_of_isClosed {t : Set β}
    (hf : ContinuousOn f s) (hs : IsClosed s) (ht : IsClosed t) : IsClosed (s ∩ f ⁻¹' t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    t : Set β
    hf : ContinuousOn f s
    hs : IsClosed s
    ht : IsClosed t
    ⊢ IsClosed (Inter.inter s (Set.preimage f t))
  -/
  rcases continuousOn_iff_isClosed.1 hf t ht with ⟨u, hu⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    t : Set β
    hf : ContinuousOn f s
    hs : IsClosed s
    ht : IsClosed t
    u : Set α
    hu : And (IsClosed u) (Eq (Inter.inter (Set.preimage f t) s) (Inter.inter u s))
    ⊢ IsClosed (Inter.inter s (Set.preimage f t))
  -/
  rw [inter_comm, hu.2]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    t : Set β
    hf : ContinuousOn f s
    hs : IsClosed s
    ht : IsClosed t
    u : Set α
    hu : And (IsClosed u) (Eq (Inter.inter (Set.preimage f t) s) (Inter.inter u s))
    ⊢ IsClosed (Inter.inter u s)
  -/
  apply IsClosed.inter hu.1 hs
  /-
    🎉 no goals
  -/


theorem ContinuousOn.preimage_interior_subset_interior_preimage {t : Set β}
    (hf : ContinuousOn f s) (hs : IsOpen s) : s ∩ f ⁻¹' interior t ⊆ s ∩ interior (f ⁻¹' t) :=
  calc
    s ∩ f ⁻¹' interior t ⊆ interior (s ∩ f ⁻¹' t) :=
      interior_maximal (inter_subset_inter (Subset.refl _) (preimage_mono interior_subset))
        (hf.isOpen_inter_preimage hs isOpen_interior)
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       inst✝¹ : TopologicalSpace α
                                       inst✝ : TopologicalSpace β
                                       f : α → β
                                       s : Set α
                                       t : Set β
                                       hf : ContinuousOn f s
                                       hs : IsOpen s
                                       ⊢ Eq (interior (Inter.inter s (Set.preimage f t))) (Inter.inter s (interior (S …
                                     -/
    _ = s ∩ interior (f ⁻¹' t) := by rw [interior_inter, hs.interior_eq]
                                     /-
                                       🎉 no goals
                                     -/


theorem continuousOn_of_locally_continuousOn
    (h : ∀ x ∈ s, ∃ t, IsOpen t ∧ x ∈ t ∧ ContinuousOn f (s ∩ t)) : ContinuousOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Membe …
    ⊢ ContinuousOn f s
  -/
  intro x xs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Membe …
    x : α
    xs : Membership.mem s x
    ⊢ ContinuousWithinAt f s x
  -/
  rcases h x xs with ⟨t, open_t, xt, ct⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Membe …
    x : α
    xs : Membership.mem s x
    t : Set α
    open_t : IsOpen t
    xt : Membership.mem t x
    ct : ContinuousOn f (Inter.inter s t)
    ⊢ ContinuousWithinAt f s x
  -/
  have := ct x ⟨xs, xt⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Membe …
    x : α
    xs : Membership.mem s x
    t : Set α
    open_t : IsOpen t
    xt : Membership.mem t x
    ct : ContinuousOn f (Inter.inter s t)
    this : ContinuousWithinAt f (Inter.inter s t) x
    ⊢ ContinuousWithinAt f s x
  -/
  rwa [ContinuousWithinAt, ← nhdsWithin_restrict _ xt open_t] at this
  /-
    🎉 no goals
  -/


theorem continuousOn_to_generateFrom_iff {β : Type*} {T : Set (Set β)} {f : α → β} :
    @ContinuousOn α β _ (.generateFrom T) f s ↔ ∀ x ∈ s, ∀ t ∈ T, f x ∈ t → f ⁻¹' t ∈ 𝓝[s] x :=
  forall₂_congr fun x _ => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      β : Type u_5
      T : Set (Set β)
      f : α → β
      x : α
      x✝ : Membership.mem s x
      ⊢ Iff (ContinuousWithinAt f s x) (∀ (t : Set β), Membership.mem T t → Membersh …
    -/
    delta ContinuousWithinAt
    simp only [TopologicalSpace.nhds_generateFrom, tendsto_iInf, tendsto_principal, mem_setOf_eq,
      and_imp]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      β : Type u_5
      T : Set (Set β)
      f : α → β
      x : α
      x✝ : Membership.mem s x
      ⊢ Iff (∀ (i : Set β), Membership.mem i (f x) → Membership.mem T i → Filter.Eve …
    -/
    exact forall_congr' fun t => forall_swap
    /-
      🎉 no goals
    -/

-- Porting note: dropped an unneeded assumption

theorem continuousOn_isOpen_of_generateFrom {β : Type*} {s : Set α} {T : Set (Set β)} {f : α → β}
    (h : ∀ t ∈ T, IsOpen (s ∩ f ⁻¹' t)) :
    @ContinuousOn α β _ (.generateFrom T) f s :=
  continuousOn_to_generateFrom_iff.2 fun _x hx t ht hxt => mem_nhdsWithin.2
    ⟨_, h t ht, ⟨hx, hxt⟩, fun _y hy => hy.1.2⟩


theorem ContinuousWithinAt.mono (h : ContinuousWithinAt f t x)
    (hs : s ⊆ t) : ContinuousWithinAt f s x :=
  h.mono_left (nhdsWithin_mono x hs)


theorem ContinuousWithinAt.mono_of_mem_nhdsWithin (h : ContinuousWithinAt f t x) (hs : t ∈ 𝓝[s] x) :
    ContinuousWithinAt f s x :=
  h.mono_left (nhdsWithin_le_of_mem hs)


@[deprecated (since := "2024-10-18")]
alias ContinuousWithinAt.mono_of_mem := ContinuousWithinAt.mono_of_mem_nhdsWithin


/-- If two sets coincide around `x`, then being continuous within one or the other at `x` is
equivalent. See also `continuousWithinAt_congr_set'` which requires that the sets coincide
locally away from a point `y`, in a T1 space. -/
theorem continuousWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt f t x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    x : α
    h : (nhds x).EventuallyEq s t
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt f t x)
  -/
  simp only [ContinuousWithinAt, nhdsWithin_eq_iff_eventuallyEq.mpr h]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias continuousWithinAt_congr_nhds := continuousWithinAt_congr_set


theorem ContinuousWithinAt.congr_set (hf : ContinuousWithinAt f s x) (h : s =ᶠ[𝓝 x] t) :
    ContinuousWithinAt f t x :=
  (continuousWithinAt_congr_set h).1 hf


@[deprecated (since := "2024-10-18")]
alias ContinuousWithinAt.congr_nhds := ContinuousWithinAt.congr_set


theorem continuousWithinAt_inter' (h : t ∈ 𝓝[s] x) :
    ContinuousWithinAt f (s ∩ t) x ↔ ContinuousWithinAt f s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    x : α
    h : Membership.mem (nhdsWithin x s) t
    ⊢ Iff (ContinuousWithinAt f (Inter.inter s t) x) (ContinuousWithinAt f s x)
  -/
  simp [ContinuousWithinAt, nhdsWithin_restrict'' s h]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_inter (h : t ∈ 𝓝 x) :
    ContinuousWithinAt f (s ∩ t) x ↔ ContinuousWithinAt f s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    x : α
    h : Membership.mem (nhds x) t
    ⊢ Iff (ContinuousWithinAt f (Inter.inter s t) x) (ContinuousWithinAt f s x)
  -/
  simp [ContinuousWithinAt, nhdsWithin_restrict' s h]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_union :
    ContinuousWithinAt f (s ∪ t) x ↔ ContinuousWithinAt f s x ∧ ContinuousWithinAt f t x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    x : α
    ⊢ Iff (ContinuousWithinAt f (Union.union s t) x) (And (ContinuousWithinAt f s  …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_union, tendsto_sup]
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.union (hs : ContinuousWithinAt f s x) (ht : ContinuousWithinAt f t x) :
    ContinuousWithinAt f (s ∪ t) x :=
  continuousWithinAt_union.2 ⟨hs, ht⟩


@[simp]
theorem continuousWithinAt_singleton : ContinuousWithinAt f {x} x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    x : α
    ⊢ ContinuousWithinAt f (Singleton.singleton x) x
  -/
  simp only [ContinuousWithinAt, nhdsWithin_singleton, tendsto_pure_nhds]
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_insert_self :
    ContinuousWithinAt f (insert x s) x ↔ ContinuousWithinAt f s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    x : α
    ⊢ Iff (ContinuousWithinAt f (Insert.insert x s) x) (ContinuousWithinAt f s x)
  -/
  simp only [← singleton_union, continuousWithinAt_union, continuousWithinAt_singleton, true_and]
  /-
    🎉 no goals
  -/


protected alias ⟨_, ContinuousWithinAt.insert⟩ := continuousWithinAt_insert_self


@[deprecated (since := "2024-10-10")]
protected alias ContinuousWithinAt.insert_self := ContinuousWithinAt.insert

/- `continuousWithinAt_insert` gives the same equivalence but at a point `y` possibly different
from `x`. As this requires the space to be T1, and this property is not available in this file,
this is found in another file although it is part of the basic API for `continuousWithinAt`. -/


theorem ContinuousWithinAt.diff_iff
    (ht : ContinuousWithinAt f t x) : ContinuousWithinAt f (s \ t) x ↔ ContinuousWithinAt f s x :=
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      inst✝¹ : TopologicalSpace α
                                      inst✝ : TopologicalSpace β
                                      f : α → β
                                      s t : Set α
                                      x : α
                                      ht : ContinuousWithinAt f t x
                                      h : ContinuousWithinAt f (SDiff.sdiff s t) x
                                      ⊢ HasSubset.Subset s (Union.union (SDiff.sdiff s t) t)
                                    -/
  ⟨fun h => (h.union ht).mono <| by simp only [diff_union_self, subset_union_left], fun h =>
                                    /-
                                      🎉 no goals
                                    -/
    h.mono diff_subset⟩


/-- See also `continuousWithinAt_diff_singleton` for the case of `s \ {y}`, but
requiring `T1Space α. -/
@[simp]
theorem continuousWithinAt_diff_self :
    ContinuousWithinAt f (s \ {x}) x ↔ ContinuousWithinAt f s x :=
  continuousWithinAt_singleton.diff_iff


@[simp]
theorem continuousWithinAt_compl_self :
    ContinuousWithinAt f {x}ᶜ x ↔ ContinuousAt f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    x : α
    ⊢ Iff (ContinuousWithinAt f (HasCompl.compl (Singleton.singleton x)) x) (Conti …
  -/
  rw [compl_eq_univ_diff, continuousWithinAt_diff_self, continuousWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem ContinuousOn.mono (hf : ContinuousOn f s) (h : t ⊆ s) :
    ContinuousOn f t := fun x hx => (hf x (h hx)).mono_left (nhdsWithin_mono _ h)


theorem antitone_continuousOn {f : α → β} : Antitone (ContinuousOn f) := fun _s _t hst hf =>
  hf.mono hst


theorem ContinuousAt.continuousWithinAt (h : ContinuousAt f x) :
    ContinuousWithinAt f s x :=
  ContinuousWithinAt.mono ((continuousWithinAt_univ f x).2 h) (subset_univ _)


theorem continuousWithinAt_iff_continuousAt (h : s ∈ 𝓝 x) :
    ContinuousWithinAt f s x ↔ ContinuousAt f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    x : α
    h : Membership.mem (nhds x) s
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousAt f x)
  -/
  rw [← univ_inter s, continuousWithinAt_inter h, continuousWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.continuousAt
    (h : ContinuousWithinAt f s x) (hs : s ∈ 𝓝 x) : ContinuousAt f x :=
  (continuousWithinAt_iff_continuousAt hs).mp h


theorem IsOpen.continuousOn_iff (hs : IsOpen s) :
    ContinuousOn f s ↔ ∀ ⦃a⦄, a ∈ s → ContinuousAt f a :=
  forall₂_congr fun _ => continuousWithinAt_iff_continuousAt ∘ hs.mem_nhds


theorem ContinuousOn.continuousAt (h : ContinuousOn f s)
    (hx : s ∈ 𝓝 x) : ContinuousAt f x :=
  (h x (mem_of_mem_nhds hx)).continuousAt hx


theorem continuousOn_of_forall_continuousAt (hcont : ∀ x ∈ s, ContinuousAt f x) :
    ContinuousOn f s := fun x hx => (hcont x hx).continuousWithinAt


@[deprecated (since := "2024-10-30")]
alias ContinuousAt.continuousOn := continuousOn_of_forall_continuousAt


@[fun_prop]
theorem Continuous.continuousOn (h : Continuous f) : ContinuousOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : Continuous f
    ⊢ ContinuousOn f s
  -/
  rw [continuous_iff_continuousOn_univ] at h
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : ContinuousOn f Set.univ
    ⊢ ContinuousOn f s
  -/
  exact h.mono (subset_univ _)
  /-
    🎉 no goals
  -/


theorem Continuous.continuousWithinAt (h : Continuous f) :
    ContinuousWithinAt f s x :=
  h.continuousAt.continuousWithinAt



theorem ContinuousOn.congr_mono (h : ContinuousOn f s) (h' : EqOn g f s₁) (h₁ : s₁ ⊆ s) :
    ContinuousOn g s₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    ⊢ ContinuousOn g s₁
  -/
  intro x hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    x : α
    hx : Membership.mem s₁ x
    ⊢ ContinuousWithinAt g s₁ x
  -/
  unfold ContinuousWithinAt
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    x : α
    hx : Membership.mem s₁ x
    ⊢ Filter.Tendsto g (nhdsWithin x s₁) (nhds (g x))
  -/
  have A := (h x (h₁ hx)).mono h₁
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    x : α
    hx : Membership.mem s₁ x
    A : ContinuousWithinAt f s₁ x
    ⊢ Filter.Tendsto g (nhdsWithin x s₁) (nhds (g x))
  -/
  unfold ContinuousWithinAt at A
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    x : α
    hx : Membership.mem s₁ x
    A : Filter.Tendsto f (nhdsWithin x s₁) (nhds (f x))
    ⊢ Filter.Tendsto g (nhdsWithin x s₁) (nhds (g x))
  -/
  rw [← h' hx] at A
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s s₁ : Set α
    h : ContinuousOn f s
    h' : Set.EqOn g f s₁
    h₁ : HasSubset.Subset s₁ s
    x : α
    hx : Membership.mem s₁ x
    A : Filter.Tendsto f (nhdsWithin x s₁) (nhds (g x))
    ⊢ Filter.Tendsto g (nhdsWithin x s₁) (nhds (g x))
  -/
  exact A.congr' h'.eventuallyEq_nhdsWithin.symm
  /-
    🎉 no goals
  -/


theorem ContinuousOn.congr (h : ContinuousOn f s) (h' : EqOn g f s) :
    ContinuousOn g s :=
  h.congr_mono h' (Subset.refl _)


theorem continuousOn_congr (h' : EqOn g f s) :
    ContinuousOn g s ↔ ContinuousOn f s :=
  ⟨fun h => ContinuousOn.congr h h'.symm, fun h => h.congr h'⟩


theorem Filter.EventuallyEq.congr_continuousWithinAt (h : f =ᶠ[𝓝[s] x] g) (hx : f x = g x) :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt g s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    s : Set α
    x : α
    h : (nhdsWithin x s).EventuallyEq f g
    hx : Eq (f x) (g x)
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt g s x)
  -/
  rw [ContinuousWithinAt, hx, tendsto_congr' h, ContinuousWithinAt]
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.congr_of_eventuallyEq
    (h : ContinuousWithinAt f s x) (h₁ : g =ᶠ[𝓝[s] x] f) (hx : g x = f x) :
    ContinuousWithinAt g s x :=
  (h₁.congr_continuousWithinAt hx).2 h


theorem ContinuousWithinAt.congr_of_eventuallyEq_of_mem
    (h : ContinuousWithinAt f s x) (h₁ : g =ᶠ[𝓝[s] x] f) (hx : x ∈ s) :
    ContinuousWithinAt g s x :=
  h.congr_of_eventuallyEq h₁ (mem_of_mem_nhdsWithin hx h₁ :)


theorem Filter.EventuallyEq.congr_continuousWithinAt_of_mem (h : f =ᶠ[𝓝[s] x] g) (hx : x ∈ s) :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt g s x :=
  ⟨fun h' ↦ h'.congr_of_eventuallyEq_of_mem h.symm hx,
    fun h' ↦  h'.congr_of_eventuallyEq_of_mem h hx⟩


theorem ContinuousWithinAt.congr_of_eventuallyEq_insert
    (h : ContinuousWithinAt f s x) (h₁ : g =ᶠ[𝓝[insert x s] x] f) :
    ContinuousWithinAt g s x :=
  h.congr_of_eventuallyEq (nhdsWithin_mono _ (subset_insert _ _) h₁)
    (mem_of_mem_nhdsWithin (mem_insert _ _) h₁ :)


theorem Filter.EventuallyEq.congr_continuousWithinAt_of_insert (h : f =ᶠ[𝓝[insert x s] x] g) :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt g s x :=
  ⟨fun h' ↦ h'.congr_of_eventuallyEq_insert h.symm,
    fun h' ↦  h'.congr_of_eventuallyEq_insert h⟩


theorem ContinuousWithinAt.congr (h : ContinuousWithinAt f s x)
    (h₁ : ∀ y ∈ s, g y = f y) (hx : g x = f x) : ContinuousWithinAt g s x :=
  h.congr_of_eventuallyEq (mem_of_superset self_mem_nhdsWithin h₁) hx


theorem continuousWithinAt_congr (h₁ : ∀ y ∈ s, g y = f y) (hx : g x = f x) :
    ContinuousWithinAt g s x ↔ ContinuousWithinAt f s x :=
  ⟨fun h' ↦ h'.congr (fun x hx ↦ (h₁ x hx).symm) hx.symm, fun h' ↦  h'.congr h₁ hx⟩


theorem ContinuousWithinAt.congr_of_mem (h : ContinuousWithinAt f s x)
    (h₁ : ∀ y ∈ s, g y = f y) (hx : x ∈ s) : ContinuousWithinAt g s x :=
  h.congr h₁ (h₁ x hx)


theorem continuousWithinAt_congr_of_mem (h₁ : ∀ y ∈ s, g y = f y) (hx : x ∈ s) :
    ContinuousWithinAt g s x ↔ ContinuousWithinAt f s x :=
  continuousWithinAt_congr h₁ (h₁ x hx)


theorem ContinuousWithinAt.congr_of_insert (h : ContinuousWithinAt f s x)
    (h₁ : ∀ y ∈ insert x s, g y = f y) : ContinuousWithinAt g s x :=
  h.congr (fun y hy ↦ h₁ y (mem_insert_of_mem _ hy)) (h₁ x (mem_insert _ _))


theorem continuousWithinAt_congr_of_insert
    (h₁ : ∀ y ∈ insert x s, g y = f y) :
    ContinuousWithinAt g s x ↔ ContinuousWithinAt f s x :=
  continuousWithinAt_congr (fun y hy ↦ h₁ y (mem_insert_of_mem _ hy)) (h₁ x (mem_insert _ _))


theorem ContinuousWithinAt.congr_mono
    (h : ContinuousWithinAt f s x) (h' : EqOn g f s₁) (h₁ : s₁ ⊆ s) (hx : g x = f x) :
    ContinuousWithinAt g s₁ x :=
  (h.mono h₁).congr h' hx


theorem ContinuousAt.congr_of_eventuallyEq (h : ContinuousAt f x) (hg : g =ᶠ[𝓝 x] f) :
    ContinuousAt g x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    x : α
    h : ContinuousAt f x
    hg : (nhds x).EventuallyEq g f
    ⊢ ContinuousAt g x
  -/
  simp only [← continuousWithinAt_univ] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f g : α → β
    x : α
    hg : (nhds x).EventuallyEq g f
    h : ContinuousWithinAt f Set.univ x
    ⊢ ContinuousWithinAt g Set.univ x
  -/
  exact h.congr_of_eventuallyEq_of_mem (by rwa [nhdsWithin_univ]) (mem_univ x)
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.comp {g : β → γ} {t : Set β}
    (hg : ContinuousWithinAt g t (f x)) (hf : ContinuousWithinAt f s x) (h : MapsTo f s t) :
    ContinuousWithinAt (g ∘ f) s x :=
  hg.tendsto.comp (hf.tendsto_nhdsWithin h)


theorem ContinuousWithinAt.comp_of_eq {g : β → γ} {t : Set β} {y : β}
    (hg : ContinuousWithinAt g t y) (hf : ContinuousWithinAt f s x) (h : MapsTo f s t)
    (hy : f x = y) : ContinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    s : Set α
    x : α
    g : β → γ
    t : Set β
    y : β
    hg : ContinuousWithinAt g t y
    hf : ContinuousWithinAt f s x
    h : Set.MapsTo f s t
    hy : Eq (f x) y
    ⊢ ContinuousWithinAt (Function.comp g f) s x
  -/
  subst hy; exact hg.comp hf h
            /-
              🎉 no goals
            -/


theorem ContinuousWithinAt.comp_inter {g : β → γ} {t : Set β}
    (hg : ContinuousWithinAt g t (f x)) (hf : ContinuousWithinAt f s x) :
    ContinuousWithinAt (g ∘ f) (s ∩ f ⁻¹' t) x :=
  hg.comp (hf.mono inter_subset_left) inter_subset_right


@[deprecated (since := "2024-10-10")]
protected alias ContinuousWithinAt.comp' := ContinuousWithinAt.comp_inter


theorem ContinuousWithinAt.comp_inter_of_eq {g : β → γ} {t : Set β} {y : β}
    (hg : ContinuousWithinAt g t y) (hf : ContinuousWithinAt f s x) (hy : f x = y) :
    ContinuousWithinAt (g ∘ f) (s ∩ f ⁻¹' t) x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    s : Set α
    x : α
    g : β → γ
    t : Set β
    y : β
    hg : ContinuousWithinAt g t y
    hf : ContinuousWithinAt f s x
    hy : Eq (f x) y
    ⊢ ContinuousWithinAt (Function.comp g f) (Inter.inter s (Set.preimage f t)) x
  -/
  subst hy; exact hg.comp_inter hf
            /-
              🎉 no goals
            -/


theorem ContinuousWithinAt.comp_of_preimage_mem_nhdsWithin {g : β → γ} {t : Set β}
    (hg : ContinuousWithinAt g t (f x)) (hf : ContinuousWithinAt f s x) (h : f ⁻¹' t ∈ 𝓝[s] x) :
    ContinuousWithinAt (g ∘ f) s x :=
  hg.tendsto.comp (tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within f hf h)


theorem ContinuousWithinAt.comp_of_preimage_mem_nhdsWithin_of_eq {g : β → γ} {t : Set β} {y : β}
    (hg : ContinuousWithinAt g t y) (hf : ContinuousWithinAt f s x) (h : f ⁻¹' t ∈ 𝓝[s] x)
    (hy : f x = y) :
    ContinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    s : Set α
    x : α
    g : β → γ
    t : Set β
    y : β
    hg : ContinuousWithinAt g t y
    hf : ContinuousWithinAt f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f t)
    hy : Eq (f x) y
    ⊢ ContinuousWithinAt (Function.comp g f) s x
  -/
  subst hy; exact hg.comp_of_preimage_mem_nhdsWithin hf h
            /-
              🎉 no goals
            -/


theorem ContinuousWithinAt.comp_of_mem_nhdsWithin_image {g : β → γ} {t : Set β}
    (hg : ContinuousWithinAt g t (f x)) (hf : ContinuousWithinAt f s x)
    (hs : t ∈ 𝓝[f '' s] f x) : ContinuousWithinAt (g ∘ f) s x :=
  (hg.mono_of_mem_nhdsWithin hs).comp hf (mapsTo_image f s)


theorem ContinuousWithinAt.comp_of_mem_nhdsWithin_image_of_eq {g : β → γ} {t : Set β} {y : β}
    (hg : ContinuousWithinAt g t y) (hf : ContinuousWithinAt f s x)
    (hs : t ∈ 𝓝[f '' s] y) (hy : f x = y) : ContinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    s : Set α
    x : α
    g : β → γ
    t : Set β
    y : β
    hg : ContinuousWithinAt g t y
    hf : ContinuousWithinAt f s x
    hs : Membership.mem (nhdsWithin y (Set.image f s)) t
    hy : Eq (f x) y
    ⊢ ContinuousWithinAt (Function.comp g f) s x
  -/
  subst hy; exact hg.comp_of_mem_nhdsWithin_image hf hs
            /-
              🎉 no goals
            -/


theorem ContinuousAt.comp_continuousWithinAt {g : β → γ}
    (hg : ContinuousAt g (f x)) (hf : ContinuousWithinAt f s x) : ContinuousWithinAt (g ∘ f) s x :=
  hg.continuousWithinAt.comp hf (mapsTo_univ _ _)


theorem ContinuousAt.comp_continuousWithinAt_of_eq {g : β → γ} {y : β}
    (hg : ContinuousAt g y) (hf : ContinuousWithinAt f s x) (hy : f x = y) :
    ContinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    s : Set α
    x : α
    g : β → γ
    y : β
    hg : ContinuousAt g y
    hf : ContinuousWithinAt f s x
    hy : Eq (f x) y
    ⊢ ContinuousWithinAt (Function.comp g f) s x
  -/
  subst hy; exact hg.comp_continuousWithinAt hf
            /-
              🎉 no goals
            -/


/-- See also `ContinuousOn.comp'` using the form `fun y ↦ g (f y)` instead of `g ∘ f`. -/
theorem ContinuousOn.comp {g : β → γ} {t : Set β} (hg : ContinuousOn g t)
    (hf : ContinuousOn f s) (h : MapsTo f s t) : ContinuousOn (g ∘ f) s := fun x hx =>
  ContinuousWithinAt.comp (hg _ (h hx)) (hf x hx) h


/-- Variant of `ContinuousOn.comp` using the form `fun y ↦ g (f y)` instead of `g ∘ f`. -/
@[fun_prop]
theorem ContinuousOn.comp' {g : β → γ} {f : α → β} {s : Set α} {t : Set β} (hg : ContinuousOn g t)
    (hf : ContinuousOn f s) (h : Set.MapsTo f s t) : ContinuousOn (fun x => g (f x)) s :=
  ContinuousOn.comp hg hf h


@[fun_prop]
theorem ContinuousOn.comp_inter {g : β → γ} {t : Set β} (hg : ContinuousOn g t)
    (hf : ContinuousOn f s) : ContinuousOn (g ∘ f) (s ∩ f ⁻¹' t) :=
  hg.comp (hf.mono inter_subset_left) inter_subset_right


/-- See also `Continuous.comp_continuousOn'` using the form `fun y ↦ g (f y)`
instead of `g ∘ f`. -/
theorem Continuous.comp_continuousOn {g : β → γ} {f : α → β} {s : Set α} (hg : Continuous g)
    (hf : ContinuousOn f s) : ContinuousOn (g ∘ f) s :=
  hg.continuousOn.comp hf (mapsTo_univ _ _)


/-- Variant of `Continuous.comp_continuousOn` using the form `fun y ↦ g (f y)`
instead of `g ∘ f`. -/
@[fun_prop]
theorem Continuous.comp_continuousOn'
    {α β γ : Type*} [TopologicalSpace α] [TopologicalSpace β] [TopologicalSpace γ] {g : β → γ}
    {f : α → β} {s : Set α} (hg : Continuous g) (hf : ContinuousOn f s) :
    ContinuousOn (fun x ↦ g (f x)) s :=
  hg.comp_continuousOn hf


theorem ContinuousOn.comp_continuous {g : β → γ} {f : α → β} {s : Set β} (hg : ContinuousOn g s)
    (hf : Continuous f) (hs : ∀ x, f x ∈ s) : Continuous (g ∘ f) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ
    f : α → β
    s : Set β
    hg : ContinuousOn g s
    hf : Continuous f
    hs : ∀ (x : α), Membership.mem s (f x)
    ⊢ Continuous (Function.comp g f)
  -/
  rw [continuous_iff_continuousOn_univ] at *
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ
    f : α → β
    s : Set β
    hg : ContinuousOn g s
    hf : ContinuousOn f Set.univ
    hs : ∀ (x : α), Membership.mem s (f x)
    ⊢ ContinuousOn (Function.comp g f) Set.univ
  -/
  exact hg.comp hf fun x _ => hs x
  /-
    🎉 no goals
  -/


theorem ContinuousOn.image_comp_continuous {g : β → γ} {f : α → β} {s : Set α}
    (hg : ContinuousOn g (f '' s)) (hf : Continuous f) : ContinuousOn (g ∘ f) s :=
  hg.comp hf.continuousOn (s.mapsTo_image f)


theorem ContinuousAt.comp₂_continuousWithinAt {f : β × γ → δ} {g : α → β} {h : α → γ} {x : α}
    {s : Set α} (hf : ContinuousAt f (g x, h x)) (hg : ContinuousWithinAt g s x)
    (hh : ContinuousWithinAt h s x) :
    ContinuousWithinAt (fun x ↦ f (g x, h x)) s x :=
  ContinuousAt.comp_continuousWithinAt hf (hg.prod_mk_nhds hh)


theorem ContinuousAt.comp₂_continuousWithinAt_of_eq {f : β × γ → δ} {g : α → β}
    {h : α → γ} {x : α} {s : Set α} {y : β × γ} (hf : ContinuousAt f y)
    (hg : ContinuousWithinAt g s x) (hh : ContinuousWithinAt h s x) (e : (g x, h x) = y) :
    ContinuousWithinAt (fun x ↦ f (g x, h x)) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f : Prod β γ → δ
    g : α → β
    h : α → γ
    x : α
    s : Set α
    y : Prod β γ
    hf : ContinuousAt f y
    hg : ContinuousWithinAt g s x
    hh : ContinuousWithinAt h s x
    e : Eq { fst := g x, snd := h x } y
    ⊢ ContinuousWithinAt (fun x => f { fst := g x, snd := h x }) s x
  -/
  rw [← e] at hf
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f : Prod β γ → δ
    g : α → β
    h : α → γ
    x : α
    s : Set α
    y : Prod β γ
    hf : ContinuousAt f { fst := g x, snd := h x }
    hg : ContinuousWithinAt g s x
    hh : ContinuousWithinAt h s x
    e : Eq { fst := g x, snd := h x } y
    ⊢ ContinuousWithinAt (fun x => f { fst := g x, snd := h x }) s x
  -/
  exact hf.comp₂_continuousWithinAt hg hh
  /-
    🎉 no goals
  -/



theorem ContinuousWithinAt.mem_closure_image
    (h : ContinuousWithinAt f s x) (hx : x ∈ closure s) : f x ∈ closure (f '' s) :=
  haveI := mem_closure_iff_nhdsWithin_neBot.1 hx
  mem_closure_of_tendsto h <| mem_of_superset self_mem_nhdsWithin (subset_preimage_image f s)


theorem ContinuousWithinAt.mem_closure {t : Set β}
    (h : ContinuousWithinAt f s x) (hx : x ∈ closure s) (ht : MapsTo f s t) : f x ∈ closure t :=
  closure_mono (image_subset_iff.2 ht) (h.mem_closure_image hx)


theorem Set.MapsTo.closure_of_continuousWithinAt {t : Set β}
    (h : MapsTo f s t) (hc : ∀ x ∈ closure s, ContinuousWithinAt f s x) :
    MapsTo f (closure s) (closure t) := fun x hx => (hc x hx).mem_closure hx h


theorem Set.MapsTo.closure_of_continuousOn {t : Set β} (h : MapsTo f s t)
    (hc : ContinuousOn f (closure s)) : MapsTo f (closure s) (closure t) :=
  h.closure_of_continuousWithinAt fun x hx => (hc x hx).mono subset_closure


theorem ContinuousWithinAt.image_closure
    (hf : ∀ x ∈ closure s, ContinuousWithinAt f s x) : f '' closure s ⊆ closure (f '' s) :=
  ((mapsTo_image f s).closure_of_continuousWithinAt hf).image_subset


theorem ContinuousOn.image_closure (hf : ContinuousOn f (closure s)) :
    f '' closure s ⊆ closure (f '' s) :=
  ContinuousWithinAt.image_closure fun x hx => (hf x hx).mono subset_closure


theorem ContinuousWithinAt.prod_map {f : α → γ} {g : β → δ} {s : Set α} {t : Set β} {x : α} {y : β}
    (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g t y) :
    ContinuousWithinAt (Prod.map f g) (s ×ˢ t) (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f : α → γ
    g : β → δ
    s : Set α
    t : Set β
    x : α
    y : β
    hf : ContinuousWithinAt f s x
    hg : ContinuousWithinAt g t y
    ⊢ ContinuousWithinAt (Prod.map f g) (SProd.sprod s t) { fst := x, snd := y }
  -/
  unfold ContinuousWithinAt at *
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f : α → γ
    g : β → δ
    s : Set α
    t : Set β
    x : α
    y : β
    hf : Filter.Tendsto f (nhdsWithin x s) (nhds (f x))
    hg : Filter.Tendsto g (nhdsWithin y t) (nhds (g y))
    ⊢ Filter.Tendsto (Prod.map f g) (nhdsWithin { fst := x, snd := y } (SProd.spro …
  -/
  rw [nhdsWithin_prod_eq, Prod.map, nhds_prod_eq]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace δ
    f : α → γ
    g : β → δ
    s : Set α
    t : Set β
    x : α
    y : β
    hf : Filter.Tendsto f (nhdsWithin x s) (nhds (f x))
    hg : Filter.Tendsto g (nhdsWithin y t) (nhds (g y))
    ⊢ Filter.Tendsto (Prod.map f g) (SProd.sprod (nhdsWithin x s) (nhdsWithin y t) …
  -/
  exact hf.prod_map hg
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_prod_of_discrete_left [DiscreteTopology α]
    {f : α × β → γ} {s : Set (α × β)} {x : α × β} :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt (f ⟨x.1, ·⟩) {b | (x.1, b) ∈ s} x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    s : Set (Prod α β)
    x : Prod α β
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt (fun x_1 => f { fst := x. …
  -/
  rw [← x.eta]; simp_rw [ContinuousWithinAt, nhdsWithin, nhds_prod_eq, nhds_discrete, pure_prod,
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     γ : Type u_3
                                     inst✝³ : TopologicalSpace α
                                     inst✝² : TopologicalSpace β
                                     inst✝¹ : TopologicalSpace γ
                                     inst✝ : DiscreteTopology α
                                     f : Prod α β → γ
                                     s : Set (Prod α β)
                                     x : Prod α β
                                     ⊢ Iff (Filter.Tendsto f (Filter.map (Prod.mk x.1) (Min.min (nhds x.2) (Filter. …
                                   -/
    ← map_inf_principal_preimage]; rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem continuousWithinAt_prod_of_discrete_right [DiscreteTopology β]
    {f : α × β → γ} {s : Set (α × β)} {x : α × β} :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt (f ⟨·, x.2⟩) {a | (a, x.2) ∈ s} x.1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology β
    f : Prod α β → γ
    s : Set (Prod α β)
    x : Prod α β
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt (fun x_1 => f { fst := x_ …
  -/
  rw [← x.eta]; simp_rw [ContinuousWithinAt, nhdsWithin, nhds_prod_eq, nhds_discrete, prod_pure,
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     γ : Type u_3
                                     inst✝³ : TopologicalSpace α
                                     inst✝² : TopologicalSpace β
                                     inst✝¹ : TopologicalSpace γ
                                     inst✝ : DiscreteTopology β
                                     f : Prod α β → γ
                                     s : Set (Prod α β)
                                     x : Prod α β
                                     ⊢ Iff (Filter.Tendsto f (Filter.map (fun a => { fst := a, snd := x.2 }) (Min.m …
                                   -/
    ← map_inf_principal_preimage]; rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem continuousAt_prod_of_discrete_left [DiscreteTopology α] {f : α × β → γ} {x : α × β} :
    ContinuousAt f x ↔ ContinuousAt (f ⟨x.1, ·⟩) x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    x : Prod α β
    ⊢ Iff (ContinuousAt f x) (ContinuousAt (fun x_1 => f { fst := x.1, snd := x_1  …
  -/
  simp_rw [← continuousWithinAt_univ]; exact continuousWithinAt_prod_of_discrete_left
                                       /-
                                         🎉 no goals
                                       -/


theorem continuousAt_prod_of_discrete_right [DiscreteTopology β] {f : α × β → γ} {x : α × β} :
    ContinuousAt f x ↔ ContinuousAt (f ⟨·, x.2⟩) x.1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology β
    f : Prod α β → γ
    x : Prod α β
    ⊢ Iff (ContinuousAt f x) (ContinuousAt (fun x_1 => f { fst := x_1, snd := x.2  …
  -/
  simp_rw [← continuousWithinAt_univ]; exact continuousWithinAt_prod_of_discrete_right
                                       /-
                                         🎉 no goals
                                       -/


theorem continuousOn_prod_of_discrete_left [DiscreteTopology α] {f : α × β → γ} {s : Set (α × β)} :
    ContinuousOn f s ↔ ∀ a, ContinuousOn (f ⟨a, ·⟩) {b | (a, b) ∈ s} := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    s : Set (Prod α β)
    ⊢ Iff (ContinuousOn f s) (∀ (a : α), ContinuousOn (fun x => f { fst := a, snd  …
  -/
  simp_rw [ContinuousOn, Prod.forall, continuousWithinAt_prod_of_discrete_left]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem continuousOn_prod_of_discrete_right [DiscreteTopology β] {f : α × β → γ} {s : Set (α × β)} :
    ContinuousOn f s ↔ ∀ b, ContinuousOn (f ⟨·, b⟩) {a | (a, b) ∈ s} := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology β
    f : Prod α β → γ
    s : Set (Prod α β)
    ⊢ Iff (ContinuousOn f s) (∀ (b : β), ContinuousOn (fun x => f { fst := x, snd  …
  -/
  simp_rw [ContinuousOn, Prod.forall, continuousWithinAt_prod_of_discrete_right]; apply forall_swap
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- If a function `f a b` is such that `y ↦ f a b` is continuous for all `a`, and `a` lives in a
discrete space, then `f` is continuous, and vice versa. -/
theorem continuous_prod_of_discrete_left [DiscreteTopology α] {f : α × β → γ} :
    Continuous f ↔ ∀ a, Continuous (f ⟨a, ·⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    ⊢ Iff (Continuous f) (∀ (a : α), Continuous fun x => f { fst := a, snd := x })
  -/
  simp_rw [continuous_iff_continuousOn_univ]; exact continuousOn_prod_of_discrete_left
                                              /-
                                                🎉 no goals
                                              -/


theorem continuous_prod_of_discrete_right [DiscreteTopology β] {f : α × β → γ} :
    Continuous f ↔ ∀ b, Continuous (f ⟨·, b⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology β
    f : Prod α β → γ
    ⊢ Iff (Continuous f) (∀ (b : β), Continuous fun x => f { fst := x, snd := b })
  -/
  simp_rw [continuous_iff_continuousOn_univ]; exact continuousOn_prod_of_discrete_right
                                              /-
                                                🎉 no goals
                                              -/


theorem isOpenMap_prod_of_discrete_left [DiscreteTopology α] {f : α × β → γ} :
    IsOpenMap f ↔ ∀ a, IsOpenMap (f ⟨a, ·⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    ⊢ Iff (IsOpenMap f) (∀ (a : α), IsOpenMap fun x => f { fst := a, snd := x })
  -/
  simp_rw [isOpenMap_iff_nhds_le, Prod.forall, nhds_prod_eq, nhds_discrete, pure_prod, map_map]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : DiscreteTopology α
    f : Prod α β → γ
    ⊢ Iff (∀ (a : α) (b : β), LE.le (nhds (f { fst := a, snd := b })) (Filter.map  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isOpenMap_prod_of_discrete_right [DiscreteTopology β] {f : α × β → γ} :
    IsOpenMap f ↔ ∀ b, IsOpenMap (f ⟨·, b⟩) := by
  simp_rw [isOpenMap_iff_nhds_le, Prod.forall, forall_swap (α := α) (β := β), nhds_prod_eq,
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          γ : Type u_3
                                          inst✝³ : TopologicalSpace α
                                          inst✝² : TopologicalSpace β
                                          inst✝¹ : TopologicalSpace γ
                                          inst✝ : DiscreteTopology β
                                          f : Prod α β → γ
                                          ⊢ Iff (∀ (y : β) (x : α), LE.le (nhds (f { fst := x, snd := y })) (Filter.map  …
                                        -/
    nhds_discrete, prod_pure, map_map]; rfl
                                        /-
                                          🎉 no goals
                                        -/


theorem ContinuousOn.prod_map {f : α → γ} {g : β → δ} {s : Set α} {t : Set β}
    (hf : ContinuousOn f s) (hg : ContinuousOn g t) : ContinuousOn (Prod.map f g) (s ×ˢ t) :=
  fun ⟨x, y⟩ ⟨hx, hy⟩ => ContinuousWithinAt.prod_map (hf x hx) (hg y hy)


theorem ContinuousWithinAt.prod {f : α → β} {g : α → γ} {s : Set α} {x : α}
    (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (fun x => (f x, g x)) s x :=
  hf.prod_mk_nhds hg


@[fun_prop]
theorem ContinuousOn.prod {f : α → β} {g : α → γ} {s : Set α} (hf : ContinuousOn f s)
    (hg : ContinuousOn g s) : ContinuousOn (fun x => (f x, g x)) s := fun x hx =>
  ContinuousWithinAt.prod (hf x hx) (hg x hx)


theorem continuousOn_fst {s : Set (α × β)} : ContinuousOn Prod.fst s :=
  continuous_fst.continuousOn


theorem continuousWithinAt_fst {s : Set (α × β)} {p : α × β} : ContinuousWithinAt Prod.fst s p :=
  continuous_fst.continuousWithinAt


@[fun_prop]
theorem ContinuousOn.fst {f : α → β × γ} {s : Set α} (hf : ContinuousOn f s) :
    ContinuousOn (fun x => (f x).1) s :=
  continuous_fst.comp_continuousOn hf


theorem ContinuousWithinAt.fst {f : α → β × γ} {s : Set α} {a : α} (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => (f x).fst) s a :=
  continuousAt_fst.comp_continuousWithinAt h


theorem continuousOn_snd {s : Set (α × β)} : ContinuousOn Prod.snd s :=
  continuous_snd.continuousOn


theorem continuousWithinAt_snd {s : Set (α × β)} {p : α × β} : ContinuousWithinAt Prod.snd s p :=
  continuous_snd.continuousWithinAt


@[fun_prop]
theorem ContinuousOn.snd {f : α → β × γ} {s : Set α} (hf : ContinuousOn f s) :
    ContinuousOn (fun x => (f x).2) s :=
  continuous_snd.comp_continuousOn hf


theorem ContinuousWithinAt.snd {f : α → β × γ} {s : Set α} {a : α} (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => (f x).snd) s a :=
  continuousAt_snd.comp_continuousWithinAt h


theorem continuousWithinAt_prod_iff {f : α → β × γ} {s : Set α} {x : α} :
    ContinuousWithinAt f s x ↔
      ContinuousWithinAt (Prod.fst ∘ f) s x ∧ ContinuousWithinAt (Prod.snd ∘ f) s x :=
  ⟨fun h => ⟨h.fst, h.snd⟩, fun ⟨h1, h2⟩ => h1.prod h2⟩


theorem continuousWithinAt_pi {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    {f : α → ∀ i, π i} {s : Set α} {x : α} :
    ContinuousWithinAt f s x ↔ ∀ i, ContinuousWithinAt (fun y => f y i) s x :=
  tendsto_pi_nhds


theorem continuousOn_pi {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    {f : α → ∀ i, π i} {s : Set α} : ContinuousOn f s ↔ ∀ i, ContinuousOn (fun y => f y i) s :=
  ⟨fun h i x hx => tendsto_pi_nhds.1 (h x hx) i, fun h x hx => tendsto_pi_nhds.2 fun i => h i x hx⟩


@[fun_prop]
theorem continuousOn_pi' {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    {f : α → ∀ i, π i} {s : Set α} (hf : ∀ i, ContinuousOn (fun y => f y i) s) :
    ContinuousOn f s :=
  continuousOn_pi.2 hf


@[fun_prop]
theorem continuousOn_apply {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    (i : ι) (s) : ContinuousOn (fun p : ∀ i, π i => p i) s :=
  Continuous.continuousOn (continuous_apply i)



@[fun_prop]
theorem continuousOn_const {s : Set α} {c : β} : ContinuousOn (fun _ => c) s :=
  continuous_const.continuousOn


theorem continuousWithinAt_const {b : β} {s : Set α} {x : α} :
    ContinuousWithinAt (fun _ : α => b) s x :=
  continuous_const.continuousWithinAt


theorem continuousOn_id {s : Set α} : ContinuousOn id s :=
  continuous_id.continuousOn


@[fun_prop]
theorem continuousOn_id' (s : Set α) : ContinuousOn (fun x : α => x) s := continuousOn_id


theorem continuousWithinAt_id {s : Set α} {x : α} : ContinuousWithinAt id s x :=
  continuous_id.continuousWithinAt


protected theorem ContinuousOn.iterate {f : α → α} {s : Set α} (hcont : ContinuousOn f s)
    (hmaps : MapsTo f s s) : ∀ n, ContinuousOn (f^[n]) s
  | 0 => continuousOn_id
  | (n + 1) => (hcont.iterate hmaps n).comp hcont hmaps


theorem ContinuousWithinAt.finCons
    {f : α → π 0} {g : α → ∀ j : Fin n, π (Fin.succ j)} {a : α} {s : Set α}
    (hf : ContinuousWithinAt f s a) (hg : ContinuousWithinAt g s a) :
    ContinuousWithinAt (fun a => Fin.cons (f a) (g a)) s a :=
  hf.tendsto.finCons hg


theorem ContinuousOn.finCons {f : α → π 0} {s : Set α} {g : α → ∀ j : Fin n, π (Fin.succ j)}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a => Fin.cons (f a) (g a)) s := fun a ha =>
  (hf a ha).finCons (hg a ha)


theorem ContinuousWithinAt.matrixVecCons {f : α → β} {g : α → Fin n → β} {a : α} {s : Set α}
    (hf : ContinuousWithinAt f s a) (hg : ContinuousWithinAt g s a) :
    ContinuousWithinAt (fun a => Matrix.vecCons (f a) (g a)) s a :=
  hf.tendsto.matrixVecCons hg


theorem ContinuousOn.matrixVecCons {f : α → β} {g : α → Fin n → β} {s : Set α}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a => Matrix.vecCons (f a) (g a)) s := fun a ha =>
  (hf a ha).matrixVecCons (hg a ha)


theorem ContinuousWithinAt.finSnoc
    {f : α → ∀ j : Fin n, π (Fin.castSucc j)} {g : α → π (Fin.last _)} {a : α} {s : Set α}
    (hf : ContinuousWithinAt f s a) (hg : ContinuousWithinAt g s a) :
    ContinuousWithinAt (fun a => Fin.snoc (f a) (g a)) s a :=
  hf.tendsto.finSnoc hg


theorem ContinuousOn.finSnoc
    {f : α → ∀ j : Fin n, π (Fin.castSucc j)} {g : α → π (Fin.last _)} {s : Set α}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a => Fin.snoc (f a) (g a)) s := fun a ha =>
  (hf a ha).finSnoc (hg a ha)


theorem ContinuousWithinAt.finInsertNth
    (i : Fin (n + 1)) {f : α → π i} {g : α → ∀ j : Fin n, π (i.succAbove j)} {a : α} {s : Set α}
    (hf : ContinuousWithinAt f s a) (hg : ContinuousWithinAt g s a) :
    ContinuousWithinAt (fun a => i.insertNth (f a) (g a)) s a :=
  hf.tendsto.finInsertNth i hg


@[deprecated (since := "2025-01-02")]
alias ContinuousWithinAt.fin_insertNth := ContinuousWithinAt.finInsertNth


theorem ContinuousOn.finInsertNth
    (i : Fin (n + 1)) {f : α → π i} {g : α → ∀ j : Fin n, π (i.succAbove j)} {s : Set α}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a => i.insertNth (f a) (g a)) s := fun a ha =>
  (hf a ha).finInsertNth i (hg a ha)


@[deprecated (since := "2025-01-02")]
alias ContinuousOn.fin_insertNth := ContinuousOn.finInsertNth


theorem Set.LeftInvOn.map_nhdsWithin_eq {f : α → β} {g : β → α} {x : β} {s : Set β}
    (h : LeftInvOn f g s) (hx : f (g x) = x) (hf : ContinuousWithinAt f (g '' s) (g x))
    (hg : ContinuousWithinAt g s x) : map g (𝓝[s] x) = 𝓝[g '' s] g x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    g : β → α
    x : β
    s : Set β
    h : Set.LeftInvOn f g s
    hx : Eq (f (g x)) x
    hf : ContinuousWithinAt f (Set.image g s) (g x)
    hg : ContinuousWithinAt g s x
    ⊢ Eq (Filter.map g (nhdsWithin x s)) (nhdsWithin (g x) (Set.image g s))
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      g : β → α
      x : β
      s : Set β
      h : Set.LeftInvOn f g s
      hx : Eq (f (g x)) x
      hf : ContinuousWithinAt f (Set.image g s) (g x)
      hg : ContinuousWithinAt g s x
      ⊢ LE.le (Filter.map g (nhdsWithin x s)) (nhdsWithin (g x) (Set.image g s))
    -/
  · exact hg.tendsto_nhdsWithin (mapsTo_image _ _)
    /-
      🎉 no goals
    -/
  · have A : g ∘ f =ᶠ[𝓝[g '' s] g x] id :=
      h.rightInvOn_image.eqOn.eventuallyEq_of_mem self_mem_nhdsWithin
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      g : β → α
      x : β
      s : Set β
      h : Set.LeftInvOn f g s
      hx : Eq (f (g x)) x
      hf : ContinuousWithinAt f (Set.image g s) (g x)
      hg : ContinuousWithinAt g s x
      A : (nhdsWithin (g x) (Set.image g s)).EventuallyEq (Function.comp g f) id
      ⊢ LE.le (nhdsWithin (g x) (Set.image g s)) (Filter.map g (nhdsWithin x s))
    -/
    refine le_map_of_right_inverse A ?_
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      g : β → α
      x : β
      s : Set β
      h : Set.LeftInvOn f g s
      hx : Eq (f (g x)) x
      hf : ContinuousWithinAt f (Set.image g s) (g x)
      hg : ContinuousWithinAt g s x
      A : (nhdsWithin (g x) (Set.image g s)).EventuallyEq (Function.comp g f) id
      ⊢ Filter.Tendsto f (nhdsWithin (g x) (Set.image g s)) (nhdsWithin x s)
    -/
    simpa only [hx] using hf.tendsto_nhdsWithin (h.mapsTo (surjOn_image _ _))
    /-
      🎉 no goals
    -/


theorem Function.LeftInverse.map_nhds_eq {f : α → β} {g : β → α} {x : β}
    (h : Function.LeftInverse f g) (hf : ContinuousWithinAt f (range g) (g x))
    (hg : ContinuousAt g x) : map g (𝓝 x) = 𝓝[range g] g x := by
  simpa only [nhdsWithin_univ, image_univ] using
    (h.leftInvOn univ).map_nhdsWithin_eq (h x) (by rwa [image_univ]) hg.continuousWithinAt


lemma Topology.IsInducing.continuousWithinAt_iff {f : α → β} {g : β → γ} (hg : IsInducing g)
    {s : Set α} {x : α} : ContinuousWithinAt f s x ↔ ContinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : β → γ
    hg : Topology.IsInducing g
    s : Set α
    x : α
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt (Function.comp g f) s x)
  -/
  simp_rw [ContinuousWithinAt, hg.tendsto_nhds_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-10-28")]
alias Inducing.continuousWithinAt_iff := IsInducing.continuousWithinAt_iff


lemma Topology.IsInducing.continuousOn_iff {f : α → β} {g : β → γ} (hg : IsInducing g)
    {s : Set α} : ContinuousOn f s ↔ ContinuousOn (g ∘ f) s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : β → γ
    hg : Topology.IsInducing g
    s : Set α
    ⊢ Iff (ContinuousOn f s) (ContinuousOn (Function.comp g f) s)
  -/
  simp_rw [ContinuousOn, hg.continuousWithinAt_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.continuousOn_iff := IsInducing.continuousOn_iff


lemma Topology.IsEmbedding.continuousOn_iff {f : α → β} {g : β → γ} (hg : IsEmbedding g)
    {s : Set α} : ContinuousOn f s ↔ ContinuousOn (g ∘ f) s :=
  hg.isInducing.continuousOn_iff


@[deprecated (since := "2024-10-26")]
alias Embedding.continuousOn_iff := IsEmbedding.continuousOn_iff


lemma Topology.IsEmbedding.map_nhdsWithin_eq {f : α → β} (hf : IsEmbedding f) (s : Set α) (x : α) :
    map f (𝓝[s] x) = 𝓝[f '' s] f x := by
  rw [nhdsWithin, Filter.map_inf hf.injective, hf.map_nhds_eq, map_principal, ← nhdsWithin_inter',
    inter_eq_self_of_subset_right (image_subset_range _ _)]


@[deprecated (since := "2024-10-26")]
alias Embedding.map_nhdsWithin_eq := IsEmbedding.map_nhdsWithin_eq


theorem Topology.IsOpenEmbedding.map_nhdsWithin_preimage_eq {f : α → β} (hf : IsOpenEmbedding f)
    (s : Set β) (x : α) : map f (𝓝[f ⁻¹' s] x) = 𝓝[s] f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Topology.IsOpenEmbedding f
    s : Set β
    x : α
    ⊢ Eq (Filter.map f (nhdsWithin x (Set.preimage f s))) (nhdsWithin (f x) s)
  -/
  rw [hf.isEmbedding.map_nhdsWithin_eq, image_preimage_eq_inter_range]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Topology.IsOpenEmbedding f
    s : Set β
    x : α
    ⊢ Eq (nhdsWithin (f x) (Inter.inter s (Set.range f))) (nhdsWithin (f x) s)
  -/
  apply nhdsWithin_eq_nhdsWithin (mem_range_self _) hf.isOpen_range
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Topology.IsOpenEmbedding f
    s : Set β
    x : α
    ⊢ Eq (Inter.inter (Inter.inter s (Set.range f)) (Set.range f)) (Inter.inter s  …
  -/
  rw [inter_assoc, inter_self]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.map_nhdsWithin_preimage_eq := IsOpenEmbedding.map_nhdsWithin_preimage_eq


theorem Topology.IsQuotientMap.continuousOn_isOpen_iff {f : α → β} {g : β → γ} (h : IsQuotientMap f)
    {s : Set β} (hs : IsOpen s) : ContinuousOn g s ↔ ContinuousOn (g ∘ f) (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : β → γ
    h : Topology.IsQuotientMap f
    s : Set β
    hs : IsOpen s
    ⊢ Iff (ContinuousOn g s) (ContinuousOn (Function.comp g f) (Set.preimage f s))
  -/
  simp only [continuousOn_iff_continuous_restrict, (h.restrictPreimage_isOpen hs).continuous_iff]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : β → γ
    h : Topology.IsQuotientMap f
    s : Set β
    hs : IsOpen s
    ⊢ Iff (Continuous (Function.comp (s.restrict g) (s.restrictPreimage f))) (Cont …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-22")]
alias QuotientMap.continuousOn_isOpen_iff := IsQuotientMap.continuousOn_isOpen_iff


theorem IsOpenMap.continuousOn_image_of_leftInvOn {f : α → β} {s : Set α}
    (h : IsOpenMap (s.restrict f)) {finv : β → α} (hleft : LeftInvOn finv f s) :
    ContinuousOn finv (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    h : IsOpenMap (s.restrict f)
    finv : β → α
    hleft : Set.LeftInvOn finv f s
    ⊢ ContinuousOn finv (Set.image f s)
  -/
  refine continuousOn_iff'.2 fun t ht => ⟨f '' (t ∩ s), ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      h : IsOpenMap (s.restrict f)
      finv : β → α
      hleft : Set.LeftInvOn finv f s
      t : Set α
      ht : IsOpen t
      ⊢ IsOpen (Set.image f (Inter.inter t s))
    -/
  · rw [← image_restrict]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      h : IsOpenMap (s.restrict f)
      finv : β → α
      hleft : Set.LeftInvOn finv f s
      t : Set α
      ht : IsOpen t
      ⊢ IsOpen (Set.image (s.restrict f) (Set.preimage Subtype.val t))
    -/
    exact h _ (ht.preimage continuous_subtype_val)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      h : IsOpenMap (s.restrict f)
      finv : β → α
      hleft : Set.LeftInvOn finv f s
      t : Set α
      ht : IsOpen t
      ⊢ Eq (Inter.inter (Set.preimage finv t) (Set.image f s)) (Inter.inter (Set.ima …
    -/
  · rw [inter_eq_self_of_subset_left (image_subset f inter_subset_right), hleft.image_inter']
    /-
      🎉 no goals
    -/


theorem IsOpenMap.continuousOn_range_of_leftInverse {f : α → β} (hf : IsOpenMap f) {finv : β → α}
    (hleft : Function.LeftInverse finv f) : ContinuousOn finv (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : IsOpenMap f
    finv : β → α
    hleft : Function.LeftInverse finv f
    ⊢ ContinuousOn finv (Set.range f)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : IsOpenMap f
    finv : β → α
    hleft : Function.LeftInverse finv f
    ⊢ ContinuousOn finv (Set.image f Set.univ)
  -/
  exact (hf.restrict isOpen_univ).continuousOn_image_of_leftInvOn fun x _ => hleft x
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_update_same [DecidableEq α] {y : β} :
    ContinuousWithinAt (update f x y) s x ↔ Tendsto f (𝓝[s \ {x}] x) (𝓝 y) :=
  calc
    ContinuousWithinAt (update f x y) s x ↔ Tendsto (update f x y) (𝓝[s \ {x}] x) (𝓝 y) := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      s : Set α
      x : α
      inst✝ : DecidableEq α
      y : β
      ⊢ Iff (ContinuousWithinAt (Function.update f x y) s x) (Filter.Tendsto (Functi …
    -/
    { rw [← continuousWithinAt_diff_self, ContinuousWithinAt, update_self] }
    /-
      🎉 no goals
    -/
    _ ↔ Tendsto f (𝓝[s \ {x}] x) (𝓝 y) :=
      tendsto_congr' <| eventually_nhdsWithin_iff.2 <| Eventually.of_forall
        fun _ hz => update_of_ne hz.2 ..


@[simp]
theorem continuousAt_update_same [DecidableEq α] {y : β} :
    ContinuousAt (Function.update f x y) x ↔ Tendsto f (𝓝[≠] x) (𝓝 y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f : α → β
    x : α
    inst✝ : DecidableEq α
    y : β
    ⊢ Iff (ContinuousAt (Function.update f x y) x) (Filter.Tendsto f (nhdsWithin x …
  -/
  rw [← continuousWithinAt_univ, continuousWithinAt_update_same, compl_eq_univ_diff]
  /-
    🎉 no goals
  -/


theorem ContinuousOn.if' {s : Set α} {p : α → Prop} {f g : α → β} [∀ a, Decidable (p a)]
    (hpf : ∀ a ∈ s ∩ frontier { a | p a },
      Tendsto f (𝓝[s ∩ { a | p a }] a) (𝓝 <| if p a then f a else g a))
    (hpg :
      ∀ a ∈ s ∩ frontier { a | p a },
        Tendsto g (𝓝[s ∩ { a | ¬p a }] a) (𝓝 <| if p a then f a else g a))
    (hf : ContinuousOn f <| s ∩ { a | p a }) (hg : ContinuousOn g <| s ∩ { a | ¬p a }) :
    ContinuousOn (fun a => if p a then f a else g a) s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    s : Set α
    p : α → Prop
    f g : α → β
    inst✝ : (a : α) → Decidable (p a)
    hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
    hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
    hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
    hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
    ⊢ ContinuousOn (fun a => ite (p a) (f a) (g a)) s
  -/
  intro x hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    s : Set α
    p : α → Prop
    f g : α → β
    inst✝ : (a : α) → Decidable (p a)
    hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
    hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
    hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
    hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
    x : α
    hx : Membership.mem s x
    ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) s x
  -/
  by_cases hx' : x ∈ frontier { a | p a }
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      s : Set α
      p : α → Prop
      f g : α → β
      inst✝ : (a : α) → Decidable (p a)
      hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
      hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
      x : α
      hx : Membership.mem s x
      hx' : Membership.mem (frontier (setOf fun a => p a)) x
      ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) s x
    -/
  · exact (hpf x ⟨hx, hx'⟩).piecewise_nhdsWithin (hpg x ⟨hx, hx'⟩)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      s : Set α
      p : α → Prop
      f g : α → β
      inst✝ : (a : α) → Decidable (p a)
      hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
      hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
      x : α
      hx : Membership.mem s x
      hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
      ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) s x
    -/
  · rw [← inter_univ s, ← union_compl_self { a | p a }, inter_union_distrib_left] at hx ⊢
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      s : Set α
      p : α → Prop
      f g : α → β
      inst✝ : (a : α) → Decidable (p a)
      hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
      hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
      hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
      x : α
      hx : Membership.mem (Union.union (Inter.inter s (setOf fun a => p a)) (Inter.i …
      hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
      ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) (Union.union (Inter.inte …
    -/
    cases' hx with hx hx
      /-
        case neg.inl
        α : Type u_1
        β : Type u_2
        inst✝² : TopologicalSpace α
        inst✝¹ : TopologicalSpace β
        s : Set α
        p : α → Prop
        f g : α → β
        inst✝ : (a : α) → Decidable (p a)
        hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
        hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
        hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
        hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
        x : α
        hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
        hx : Membership.mem (Inter.inter s (setOf fun a => p a)) x
        ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) (Union.union (Inter.inte …
      -/
    · apply ContinuousWithinAt.union
        /-
          case neg.inl.hs
          α : Type u_1
          β : Type u_2
          inst✝² : TopologicalSpace α
          inst✝¹ : TopologicalSpace β
          s : Set α
          p : α → Prop
          f g : α → β
          inst✝ : (a : α) → Decidable (p a)
          hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
          hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
          hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
          hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
          x : α
          hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
          hx : Membership.mem (Inter.inter s (setOf fun a => p a)) x
          ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) (Inter.inter s (setOf fu …
        -/
      · exact (hf x hx).congr (fun y hy => if_pos hy.2) (if_pos hx.2)
        /-
          🎉 no goals
        -/
      · have : x ∉ closure { a | p a }ᶜ := fun h => hx' ⟨subset_closure hx.2, by
          rwa [closure_compl] at h⟩
        exact continuousWithinAt_of_not_mem_closure fun h =>
          this (closure_inter_subset_inter_closure _ _ h).2
      /-
        case neg.inr
        α : Type u_1
        β : Type u_2
        inst✝² : TopologicalSpace α
        inst✝¹ : TopologicalSpace β
        s : Set α
        p : α → Prop
        f g : α → β
        inst✝ : (a : α) → Decidable (p a)
        hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
        hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
        hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
        hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
        x : α
        hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
        hx : Membership.mem (Inter.inter s (HasCompl.compl (setOf fun a => p a))) x
        ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) (Union.union (Inter.inte …
      -/
    · apply ContinuousWithinAt.union
      · have : x ∉ closure { a | p a } := fun h =>
          hx' ⟨h, fun h' : x ∈ interior { a | p a } => hx.2 (interior_subset h')⟩
        exact continuousWithinAt_of_not_mem_closure fun h =>
          this (closure_inter_subset_inter_closure _ _ h).2
        /-
          case neg.inr.ht
          α : Type u_1
          β : Type u_2
          inst✝² : TopologicalSpace α
          inst✝¹ : TopologicalSpace β
          s : Set α
          p : α → Prop
          f g : α → β
          inst✝ : (a : α) → Decidable (p a)
          hpf : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
          hpg : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a)) …
          hf : ContinuousOn f (Inter.inter s (setOf fun a => p a))
          hg : ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
          x : α
          hx' : Not (Membership.mem (frontier (setOf fun a => p a)) x)
          hx : Membership.mem (Inter.inter s (HasCompl.compl (setOf fun a => p a))) x
          ⊢ ContinuousWithinAt (fun a => ite (p a) (f a) (g a)) (Inter.inter s (HasCompl …
        -/
      · exact (hg x hx).congr (fun y hy => if_neg hy.2) (if_neg hx.2)
        /-
          🎉 no goals
        -/


theorem ContinuousOn.piecewise' [∀ a, Decidable (a ∈ t)]
    (hpf : ∀ a ∈ s ∩ frontier t, Tendsto f (𝓝[s ∩ t] a) (𝓝 (piecewise t f g a)))
    (hpg : ∀ a ∈ s ∩ frontier t, Tendsto g (𝓝[s ∩ tᶜ] a) (𝓝 (piecewise t f g a)))
    (hf : ContinuousOn f <| s ∩ t) (hg : ContinuousOn g <| s ∩ tᶜ) :
    ContinuousOn (piecewise t f g) s :=
  hf.if' hpf hpg hg


theorem ContinuousOn.if {p : α → Prop} [∀ a, Decidable (p a)]
    (hp : ∀ a ∈ s ∩ frontier { a | p a }, f a = g a)
    (hf : ContinuousOn f <| s ∩ closure { a | p a })
    (hg : ContinuousOn g <| s ∩ closure { a | ¬p a }) :
    ContinuousOn (fun a => if p a then f a else g a) s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    s : Set α
    p : α → Prop
    inst✝ : (a : α) → Decidable (p a)
    hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
    hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
    hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
    ⊢ ContinuousOn (fun a => ite (p a) (f a) (g a)) s
  -/
  apply ContinuousOn.if'
    /-
      case hpf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      ⊢ ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a  …
    -/
  · rintro a ha
    /-
      case hpf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto f (nhdsWithin a (Inter.inter s (setOf fun a => p a))) (nhds ( …
    -/
    simp only [← hp a ha, ite_self]
    /-
      case hpf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto f (nhdsWithin a (Inter.inter s (setOf fun a => p a))) (nhds ( …
    -/
    apply tendsto_nhdsWithin_mono_left (inter_subset_inter_right s subset_closure)
    /-
      case hpf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto f (nhdsWithin a (Inter.inter s (closure (setOf fun a => p a)) …
    -/
    exact hf a ⟨ha.1, ha.2.1⟩
    /-
      🎉 no goals
    -/
    /-
      case hpg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      ⊢ ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a  …
    -/
  · rintro a ha
    /-
      case hpg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto g (nhdsWithin a (Inter.inter s (setOf fun a => Not (p a)))) ( …
    -/
    simp only [hp a ha, ite_self]
    /-
      case hpg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto g (nhdsWithin a (Inter.inter s (setOf fun a => Not (p a)))) ( …
    -/
    apply tendsto_nhdsWithin_mono_left (inter_subset_inter_right s subset_closure)
    /-
      case hpg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      ha : Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) a
      ⊢ Filter.Tendsto g (nhdsWithin a (Inter.inter s (closure (setOf fun a => Not ( …
    -/
    rcases ha with ⟨has, ⟨_, ha⟩⟩
    /-
      case hpg.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      has : Membership.mem s a
      left✝ : Membership.mem (closure (setOf fun a => p a)) a
      ha : Not (Membership.mem (interior (setOf fun a => p a)) a)
      ⊢ Filter.Tendsto g (nhdsWithin a (Inter.inter s (closure (setOf fun a => Not ( …
    -/
    rw [← mem_compl_iff, ← closure_compl] at ha
    /-
      case hpg.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      a : α
      has : Membership.mem s a
      left✝ : Membership.mem (closure (setOf fun a => p a)) a
      ha : Membership.mem (closure (HasCompl.compl (setOf fun a => p a))) a
      ⊢ Filter.Tendsto g (nhdsWithin a (Inter.inter s (closure (setOf fun a => Not ( …
    -/
    apply hg a ⟨has, ha⟩
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      ⊢ ContinuousOn f (Inter.inter s (setOf fun a => p a))
    -/
  · exact hf.mono (inter_subset_inter_right s subset_closure)
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      p : α → Prop
      inst✝ : (a : α) → Decidable (p a)
      hp : ∀ (a : α), Membership.mem (Inter.inter s (frontier (setOf fun a => p a))) …
      hf : ContinuousOn f (Inter.inter s (closure (setOf fun a => p a)))
      hg : ContinuousOn g (Inter.inter s (closure (setOf fun a => Not (p a))))
      ⊢ ContinuousOn g (Inter.inter s (setOf fun a => Not (p a)))
    -/
  · exact hg.mono (inter_subset_inter_right s subset_closure)
    /-
      🎉 no goals
    -/


theorem ContinuousOn.piecewise [∀ a, Decidable (a ∈ t)]
    (ht : ∀ a ∈ s ∩ frontier t, f a = g a) (hf : ContinuousOn f <| s ∩ closure t)
    (hg : ContinuousOn g <| s ∩ closure tᶜ) : ContinuousOn (piecewise t f g) s :=
  hf.if ht hg


theorem continuous_if' {p : α → Prop} [∀ a, Decidable (p a)]
    (hpf : ∀ a ∈ frontier { x | p x }, Tendsto f (𝓝[{ x | p x }] a) (𝓝 <| ite (p a) (f a) (g a)))
    (hpg : ∀ a ∈ frontier { x | p x }, Tendsto g (𝓝[{ x | ¬p x }] a) (𝓝 <| ite (p a) (f a) (g a)))
    (hf : ContinuousOn f { x | p x }) (hg : ContinuousOn g { x | ¬p x }) :
    Continuous fun a => ite (p a) (f a) (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : α → Prop
    inst✝ : (a : α) → Decidable (p a)
    hpf : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Filter.Ten …
    hpg : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Filter.Ten …
    hf : ContinuousOn f (setOf fun x => p x)
    hg : ContinuousOn g (setOf fun x => Not (p x))
    ⊢ Continuous fun a => ite (p a) (f a) (g a)
  -/
  rw [continuous_iff_continuousOn_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : α → Prop
    inst✝ : (a : α) → Decidable (p a)
    hpf : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Filter.Ten …
    hpg : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Filter.Ten …
    hf : ContinuousOn f (setOf fun x => p x)
    hg : ContinuousOn g (setOf fun x => Not (p x))
    ⊢ ContinuousOn (fun a => ite (p a) (f a) (g a)) Set.univ
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
  apply ContinuousOn.if' <;> simp [*] <;> assumption
                                          /-
                                            🎉 no goals
                                          -/


theorem continuous_if {p : α → Prop} [∀ a, Decidable (p a)]
    (hp : ∀ a ∈ frontier { x | p x }, f a = g a) (hf : ContinuousOn f (closure { x | p x }))
    (hg : ContinuousOn g (closure { x | ¬p x })) :
    Continuous fun a => if p a then f a else g a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : α → Prop
    inst✝ : (a : α) → Decidable (p a)
    hp : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Eq (f a) (g …
    hf : ContinuousOn f (closure (setOf fun x => p x))
    hg : ContinuousOn g (closure (setOf fun x => Not (p x)))
    ⊢ Continuous fun a => ite (p a) (f a) (g a)
  -/
  rw [continuous_iff_continuousOn_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : α → Prop
    inst✝ : (a : α) → Decidable (p a)
    hp : ∀ (a : α), Membership.mem (frontier (setOf fun x => p x)) a → Eq (f a) (g …
    hf : ContinuousOn f (closure (setOf fun x => p x))
    hg : ContinuousOn g (closure (setOf fun x => Not (p x)))
    ⊢ ContinuousOn (fun a => ite (p a) (f a) (g a)) Set.univ
  -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
  apply ContinuousOn.if <;> simpa
                            /-
                              🎉 no goals
                            -/


theorem Continuous.if {p : α → Prop} [∀ a, Decidable (p a)]
    (hp : ∀ a ∈ frontier { x | p x }, f a = g a) (hf : Continuous f) (hg : Continuous g) :
    Continuous fun a => if p a then f a else g a :=
  continuous_if hp hf.continuousOn hg.continuousOn


theorem continuous_if_const (p : Prop) [Decidable p] (hf : p → Continuous f)
    (hg : ¬p → Continuous g) : Continuous fun a => if p then f a else g a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : Prop
    inst✝ : Decidable p
    hf : p → Continuous f
    hg : Not p → Continuous g
    ⊢ Continuous fun a => ite p (f a) (g a)
  -/
  split_ifs with h
  /-
    case pos
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    p : Prop
    inst✝ : Decidable p
    hf : p → Continuous f
    hg : Not p → Continuous g
    h : p
    ⊢ Continuous fun a => f a
  -/
  exacts [hf h, hg h]
  /-
    🎉 no goals
  -/


theorem Continuous.if_const (p : Prop) [Decidable p] (hf : Continuous f)
    (hg : Continuous g) : Continuous fun a => if p then f a else g a :=
  continuous_if_const p (fun _ => hf) fun _ => hg


theorem continuous_piecewise [∀ a, Decidable (a ∈ s)]
    (hs : ∀ a ∈ frontier s, f a = g a) (hf : ContinuousOn f (closure s))
    (hg : ContinuousOn g (closure sᶜ)) : Continuous (piecewise s f g) :=
  continuous_if hs hf hg


theorem Continuous.piecewise [∀ a, Decidable (a ∈ s)]
    (hs : ∀ a ∈ frontier s, f a = g a) (hf : Continuous f) (hg : Continuous g) :
    Continuous (piecewise s f g) :=
  hf.if hs hg


theorem IsOpen.ite' (hs : IsOpen s) (hs' : IsOpen s')
    (ht : ∀ x ∈ frontier t, x ∈ s ↔ x ∈ s') : IsOpen (t.ite s s') := by
  classical
    simp only [isOpen_iff_continuous_mem, Set.ite] at *
    convert
      continuous_piecewise (fun x hx => propext (ht x hx)) hs.continuousOn hs'.continuousOn using 2
    rename_i x
    by_cases hx : x ∈ t <;> simp [hx]


theorem IsOpen.ite (hs : IsOpen s) (hs' : IsOpen s')
    (ht : s ∩ frontier t = s' ∩ frontier t) : IsOpen (t.ite s s') :=
                             /-
                               α : Type u_1
                               inst✝ : TopologicalSpace α
                               s s' t : Set α
                               hs : IsOpen s
                               hs' : IsOpen s'
                               ht : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
                               x : α
                               hx : Membership.mem (frontier t) x
                               ⊢ Iff (Membership.mem s x) (Membership.mem s' x)
                             -/
  hs.ite' hs' fun x hx => by simpa [hx] using Set.ext_iff.1 ht x
                             /-
                               🎉 no goals
                             -/


theorem ite_inter_closure_eq_of_inter_frontier_eq
    (ht : s ∩ frontier t = s' ∩ frontier t) : t.ite s s' ∩ closure t = s ∩ closure t := by
  rw [closure_eq_self_union_frontier, inter_union_distrib_left, inter_union_distrib_left,
    ite_inter_self, ite_inter_of_inter_eq _ ht]


theorem ite_inter_closure_compl_eq_of_inter_frontier_eq
    (ht : s ∩ frontier t = s' ∩ frontier t) : t.ite s s' ∩ closure tᶜ = s' ∩ closure tᶜ := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s s' t : Set α
    ht : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
    ⊢ Eq (Inter.inter (t.ite s s') (closure (HasCompl.compl t))) (Inter.inter s' ( …
  -/
  rw [← ite_compl, ite_inter_closure_eq_of_inter_frontier_eq]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s s' t : Set α
    ht : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
    ⊢ Eq (Inter.inter s' (frontier (HasCompl.compl t))) (Inter.inter s (frontier ( …
  -/
  rwa [frontier_compl, eq_comm]
  /-
    🎉 no goals
  -/


theorem continuousOn_piecewise_ite' [∀ x, Decidable (x ∈ t)]
    (h : ContinuousOn f (s ∩ closure t)) (h' : ContinuousOn g (s' ∩ closure tᶜ))
    (H : s ∩ frontier t = s' ∩ frontier t) (Heq : EqOn f g (s ∩ frontier t)) :
    ContinuousOn (t.piecewise f g) (t.ite s s') := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    s s' t : Set α
    inst✝ : (x : α) → Decidable (Membership.mem t x)
    h : ContinuousOn f (Inter.inter s (closure t))
    h' : ContinuousOn g (Inter.inter s' (closure (HasCompl.compl t)))
    H : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
    Heq : Set.EqOn f g (Inter.inter s (frontier t))
    ⊢ ContinuousOn (t.piecewise f g) (t.ite s s')
  -/
  apply ContinuousOn.piecewise
    /-
      case ht
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s s' t : Set α
      inst✝ : (x : α) → Decidable (Membership.mem t x)
      h : ContinuousOn f (Inter.inter s (closure t))
      h' : ContinuousOn g (Inter.inter s' (closure (HasCompl.compl t)))
      H : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
      Heq : Set.EqOn f g (Inter.inter s (frontier t))
      ⊢ ∀ (a : α), Membership.mem (Inter.inter (t.ite s s') (frontier t)) a → Eq (f  …
    -/
  · rwa [ite_inter_of_inter_eq _ H]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s s' t : Set α
      inst✝ : (x : α) → Decidable (Membership.mem t x)
      h : ContinuousOn f (Inter.inter s (closure t))
      h' : ContinuousOn g (Inter.inter s' (closure (HasCompl.compl t)))
      H : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
      Heq : Set.EqOn f g (Inter.inter s (frontier t))
      ⊢ ContinuousOn f (Inter.inter (t.ite s s') (closure t))
    -/
  · rwa [ite_inter_closure_eq_of_inter_frontier_eq H]
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s s' t : Set α
      inst✝ : (x : α) → Decidable (Membership.mem t x)
      h : ContinuousOn f (Inter.inter s (closure t))
      h' : ContinuousOn g (Inter.inter s' (closure (HasCompl.compl t)))
      H : Eq (Inter.inter s (frontier t)) (Inter.inter s' (frontier t))
      Heq : Set.EqOn f g (Inter.inter s (frontier t))
      ⊢ ContinuousOn g (Inter.inter (t.ite s s') (closure (HasCompl.compl t)))
    -/
  · rwa [ite_inter_closure_compl_eq_of_inter_frontier_eq H]
    /-
      🎉 no goals
    -/


theorem continuousOn_piecewise_ite [∀ x, Decidable (x ∈ t)]
    (h : ContinuousOn f s) (h' : ContinuousOn g s') (H : s ∩ frontier t = s' ∩ frontier t)
    (Heq : EqOn f g (s ∩ frontier t)) : ContinuousOn (t.piecewise f g) (t.ite s s') :=
  continuousOn_piecewise_ite' (h.mono inter_subset_left) (h'.mono inter_subset_left) H Heq



/-- If `f` is continuous on an open set `s` and continuous at each point of another
set `t` then `f` is continuous on `s ∪ t`. -/
lemma ContinuousOn.union_continuousAt
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    {s t : Set X} {f : X → Y} (s_op : IsOpen s)
    (hs : ContinuousOn f s) (ht : ∀ x ∈ t, ContinuousAt f x) :
    ContinuousOn f (s ∪ t) :=
  continuousOn_of_forall_continuousAt <| fun _ hx => hx.elim
  (fun h => ContinuousWithinAt.continuousAt (continuousWithinAt hs h) <| IsOpen.mem_nhds s_op h)
  (ht _)

