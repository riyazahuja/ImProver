@[to_additive]
lemma exists_openSubgroup_separating {a b : G} (h : a ≠ b) :
    ∃ V : OpenSubgroup G, Disjoint (a • (V : Set G)) (b • V) := by
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    ⊢ Exists fun V => Disjoint (HSMul.hSMul a ↑V) (HSMul.hSMul b ↑V)
  -/
  obtain ⟨u, v, _, open_v, mem_u, mem_v, dis⟩ := t2_separation (h ∘ inv_mul_eq_one.mp)
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    ⊢ Exists fun V => Disjoint (HSMul.hSMul a ↑V) (HSMul.hSMul b ↑V)
  -/
  obtain ⟨V, hV⟩ := is_nonarchimedean v (open_v.mem_nhds mem_v)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    ⊢ Exists fun V => Disjoint (HSMul.hSMul a ↑V) (HSMul.hSMul b ↑V)
  -/
  use V
  /-
    case h
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    ⊢ Disjoint (HSMul.hSMul a ↑V) (HSMul.hSMul b ↑V)
  -/
  simp only [Disjoint, Set.le_eq_subset, Set.bot_eq_empty, Set.subset_empty_iff]
  /-
    case h
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    ⊢ ∀ ⦃x : Set G⦄, HasSubset.Subset x (HSMul.hSMul a ↑V) → HasSubset.Subset x (H …
  -/
  intros x mem_aV mem_bV
  /-
    case h
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    ⊢ Eq x EmptyCollection.emptyCollection
  -/
  by_contra! con
  /-
    case h
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    con : x.Nonempty
    ⊢ False
  -/
  obtain ⟨s, hs⟩ := con
  /-
    case h.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    s : G
    hs : Membership.mem x s
    ⊢ False
  -/
  have hsa : s ∈ a • (V : Set G) := mem_aV hs
  /-
    case h.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    s : G
    hs : Membership.mem x s
    hsa : Membership.mem (HSMul.hSMul a ↑V) s
    ⊢ False
  -/
  have hsb : s ∈ b • (V : Set G) := mem_bV hs
  /-
    case h.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    s : G
    hs : Membership.mem x s
    hsa : Membership.mem (HSMul.hSMul a ↑V) s
    hsb : Membership.mem (HSMul.hSMul b ↑V) s
    ⊢ False
  -/
  rw [mem_leftCoset_iff] at hsa hsb
  /-
    case h.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    s : G
    hs : Membership.mem x s
    hsa : Membership.mem (↑V) (HMul.hMul (Inv.inv a) s)
    hsb : Membership.mem (↑V) (HMul.hMul (Inv.inv b) s)
    ⊢ False
  -/
  refine dis.subset_compl_right mem_u (hV ?_)
  /-
    case h.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : NonarchimedeanGroup G
    inst✝ : T2Space G
    a b : G
    h : Ne a b
    u v : Set G
    left✝ : IsOpen u
    open_v : IsOpen v
    mem_u : Membership.mem u (HMul.hMul (Inv.inv a) b)
    mem_v : Membership.mem v 1
    dis : Disjoint u v
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) v
    x : Set G
    mem_aV : HasSubset.Subset x (HSMul.hSMul a ↑V)
    mem_bV : HasSubset.Subset x (HSMul.hSMul b ↑V)
    s : G
    hs : Membership.mem x s
    hsa : Membership.mem (↑V) (HMul.hMul (Inv.inv a) s)
    hsb : Membership.mem (↑V) (HMul.hMul (Inv.inv b) s)
    ⊢ Membership.mem (↑V) (HMul.hMul (Inv.inv a) b)
  -/
  simpa [mul_assoc] using mul_mem hsa (inv_mem hsb)
  /-
    🎉 no goals
  -/


@[to_additive]
instance (priority := 100) instTotallySeparated : TotallySeparatedSpace G where
  isTotallySeparated_univ x _ y _ hxy := by
    /-
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : NonarchimedeanGroup G
      inst✝ : T2Space G
      x : G
      x✝¹ : Membership.mem Set.univ x
      y : G
      x✝ : Membership.mem Set.univ y
      hxy : Ne x y
      ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
    -/
    obtain ⟨V, dxy⟩ := exists_openSubgroup_separating hxy
    exact ⟨_, _, V.isOpen.smul x, (V.isClosed.smul x).isOpen_compl, mem_own_leftCoset ..,
      dxy.subset_compl_left <| mem_own_leftCoset .., by simp, disjoint_compl_right⟩


