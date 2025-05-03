@[simp]
theorem ordConnectedComponent_mem_nhds : ordConnectedComponent s a ∈ 𝓝 a ↔ s ∈ 𝓝 a := by
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s : Set X
    ⊢ Iff (Membership.mem (nhds a) (s.ordConnectedComponent a)) (Membership.mem (n …
  -/
  refine ⟨fun h => mem_of_superset h ordConnectedComponent_subset, fun h => ?_⟩
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s : Set X
    h : Membership.mem (nhds a) s
    ⊢ Membership.mem (nhds a) (s.ordConnectedComponent a)
  -/
  rcases exists_Icc_mem_subset_of_mem_nhds h with ⟨b, c, ha, ha', hs⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s : Set X
    h : Membership.mem (nhds a) s
    b c : X
    ha : Membership.mem (Set.Icc b c) a
    ha' : Membership.mem (nhds a) (Set.Icc b c)
    hs : HasSubset.Subset (Set.Icc b c) s
    ⊢ Membership.mem (nhds a) (s.ordConnectedComponent a)
  -/
  exact mem_of_superset ha' (subset_ordConnectedComponent ha hs)
  /-
    🎉 no goals
  -/


theorem compl_ordConnectedSection_ordSeparatingSet_mem_nhdsGE (hd : Disjoint s (closure t))
    (ha : a ∈ s) : (ordConnectedSection (ordSeparatingSet s t))ᶜ ∈ 𝓝[≥] a := by
  have hmem : tᶜ ∈ 𝓝[≥] a := by
    refine mem_nhdsWithin_of_mem_nhds ?_
    rw [← mem_interior_iff_mem_nhds, interior_compl]
    exact disjoint_left.1 hd ha
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s t : Set X
    hd : Disjoint s (closure t)
    ha : Membership.mem s a
    hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
    ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
  -/
  rcases exists_Icc_mem_subset_of_mem_nhdsGE hmem with ⟨b, hab, hmem', hsub⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s t : Set X
    hd : Disjoint s (closure t)
    ha : Membership.mem s a
    hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
    b : X
    hab : LE.le a b
    hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
    hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
    ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
  -/
  by_cases H : Disjoint (Icc a b) (ordConnectedSection <| ordSeparatingSet s t)
    /-
      case pos
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      H : Disjoint (Set.Icc a b) (s.ordSeparatingSet t).ordConnectedSection
      ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
    -/
  · exact mem_of_superset hmem' (disjoint_left.1 H)
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      H : Not (Disjoint (Set.Icc a b) (s.ordSeparatingSet t).ordConnectedSection)
      ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
    -/
  · simp only [Set.disjoint_left, not_forall, Classical.not_not] at H
    /-
      case neg
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      H : Exists fun x => Exists fun h => Membership.mem (s.ordSeparatingSet t).ordC …
      ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
    -/
    rcases H with ⟨c, ⟨hac, hcb⟩, hc⟩
    have hsub' : Icc a b ⊆ ordConnectedComponent tᶜ a :=
      subset_ordConnectedComponent (left_mem_Icc.2 hab) hsub
    have hd : Disjoint s (ordConnectedSection (ordSeparatingSet s t)) :=
      disjoint_left_ordSeparatingSet.mono_right ordConnectedSection_subset
    replace hac : a < c := hac.lt_of_ne <| Ne.symm <| ne_of_mem_of_not_mem hc <|
      disjoint_left.1 hd ha
    /-
      case neg.intro.intro.intro
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd✝ : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      c : X
      hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
      hcb : LE.le c b
      hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
      hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
      hac : LT.lt a c
      ⊢ Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl (s.ordSeparatingSe …
    -/
    filter_upwards [Ico_mem_nhdsGE hac] with x hx hx'
    /-
      case h
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd✝ : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      c : X
      hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
      hcb : LE.le c b
      hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
      hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
      hac : LT.lt a c
      x : X
      hx : Membership.mem (Set.Ico a c) x
      hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
      ⊢ False
    -/
    refine hx.2.ne (eq_of_mem_ordConnectedSection_of_uIcc_subset hx' hc ?_)
    /-
      case h
      X : Type u_1
      inst✝² : LinearOrder X
      inst✝¹ : TopologicalSpace X
      inst✝ : OrderTopology X
      a : X
      s t : Set X
      hd✝ : Disjoint s (closure t)
      ha : Membership.mem s a
      hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
      b : X
      hab : LE.le a b
      hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
      hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
      c : X
      hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
      hcb : LE.le c b
      hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
      hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
      hac : LT.lt a c
      x : X
      hx : Membership.mem (Set.Ico a c) x
      hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
      ⊢ HasSubset.Subset (Set.uIcc x c) (s.ordSeparatingSet t)
    -/
    refine subset_inter (subset_iUnion₂_of_subset a ha ?_) ?_
    · exact OrdConnected.uIcc_subset inferInstance (hsub' ⟨hx.1, hx.2.le.trans hcb⟩)
        (hsub' ⟨hac.le, hcb⟩)
      /-
        case h.refine_2
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        ⊢ HasSubset.Subset (Set.uIcc x c) (Set.iUnion fun x => Set.iUnion fun h => (Ha …
      -/
    · rcases mem_iUnion₂.1 (ordConnectedSection_subset hx').2 with ⟨y, hyt, hxy⟩
      /-
        case h.refine_2.intro.intro
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        y : X
        hyt : Membership.mem t y
        hxy : Membership.mem ((HasCompl.compl s).ordConnectedComponent y) x
        ⊢ HasSubset.Subset (Set.uIcc x c) (Set.iUnion fun x => Set.iUnion fun h => (Ha …
      -/
      refine subset_iUnion₂_of_subset y hyt (OrdConnected.uIcc_subset inferInstance hxy ?_)
      /-
        case h.refine_2.intro.intro
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        y : X
        hyt : Membership.mem t y
        hxy : Membership.mem ((HasCompl.compl s).ordConnectedComponent y) x
        ⊢ Membership.mem ((HasCompl.compl s).ordConnectedComponent y) c
      -/
      refine subset_ordConnectedComponent left_mem_uIcc hxy ?_
      suffices c < y by
        rw [uIcc_of_ge (hx.2.trans this).le]
        exact ⟨hx.2.le, this.le⟩
      /-
        case h.refine_2.intro.intro
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        y : X
        hyt : Membership.mem t y
        hxy : Membership.mem ((HasCompl.compl s).ordConnectedComponent y) x
        ⊢ LT.lt c y
      -/
      refine lt_of_not_le fun hyc => ?_
      /-
        case h.refine_2.intro.intro
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        y : X
        hyt : Membership.mem t y
        hxy : Membership.mem ((HasCompl.compl s).ordConnectedComponent y) x
        hyc : LE.le y c
        ⊢ False
      -/
      have hya : y < a := not_le.1 fun hay => hsub ⟨hay, hyc.trans hcb⟩ hyt
      /-
        case h.refine_2.intro.intro
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        a : X
        s t : Set X
        hd✝ : Disjoint s (closure t)
        ha : Membership.mem s a
        hmem : Membership.mem (nhdsWithin a (Set.Ici a)) (HasCompl.compl t)
        b : X
        hab : LE.le a b
        hmem' : Membership.mem (nhdsWithin a (Set.Ici a)) (Set.Icc a b)
        hsub : HasSubset.Subset (Set.Icc a b) (HasCompl.compl t)
        c : X
        hc : Membership.mem (s.ordSeparatingSet t).ordConnectedSection c
        hcb : LE.le c b
        hsub' : HasSubset.Subset (Set.Icc a b) ((HasCompl.compl t).ordConnectedCompone …
        hd : Disjoint s (s.ordSeparatingSet t).ordConnectedSection
        hac : LT.lt a c
        x : X
        hx : Membership.mem (Set.Ico a c) x
        hx' : Membership.mem (s.ordSeparatingSet t).ordConnectedSection x
        y : X
        hyt : Membership.mem t y
        hxy : Membership.mem ((HasCompl.compl s).ordConnectedComponent y) x
        hyc : LE.le y c
        hya : LT.lt y a
        ⊢ False
      -/
      exact hxy (Icc_subset_uIcc ⟨hya.le, hx.1⟩) ha
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-12-22")]
alias compl_section_ordSeparatingSet_mem_nhdsWithin_Ici :=
  compl_ordConnectedSection_ordSeparatingSet_mem_nhdsGE


theorem compl_ordConnectedSection_ordSeparatingSet_mem_nhdsLE (hd : Disjoint s (closure t))
    (ha : a ∈ s) : (ordConnectedSection <| ordSeparatingSet s t)ᶜ ∈ 𝓝[≤] a := by
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s t : Set X
    hd : Disjoint s (closure t)
    ha : Membership.mem s a
    ⊢ Membership.mem (nhdsWithin a (Set.Iic a)) (HasCompl.compl (s.ordSeparatingSe …
  -/
  have hd' : Disjoint (ofDual ⁻¹' s) (closure <| ofDual ⁻¹' t) := hd
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s t : Set X
    hd : Disjoint s (closure t)
    ha : Membership.mem s a
    hd' : Disjoint (Set.preimage (⇑OrderDual.ofDual) s) (closure (Set.preimage (⇑O …
    ⊢ Membership.mem (nhdsWithin a (Set.Iic a)) (HasCompl.compl (s.ordSeparatingSe …
  -/
  have ha' : toDual a ∈ ofDual ⁻¹' s := ha
  simpa only [dual_ordSeparatingSet, dual_ordConnectedSection] using
    compl_ordConnectedSection_ordSeparatingSet_mem_nhdsGE hd' ha'


@[deprecated (since := "2024-12-22")]
alias compl_section_ordSeparatingSet_mem_nhdsWithin_Iic :=
  compl_ordConnectedSection_ordSeparatingSet_mem_nhdsLE


theorem compl_ordConnectedSection_ordSeparatingSet_mem_nhds (hd : Disjoint s (closure t))
    (ha : a ∈ s) : (ordConnectedSection <| ordSeparatingSet s t)ᶜ ∈ 𝓝 a := by
  /-
    X : Type u_1
    inst✝² : LinearOrder X
    inst✝¹ : TopologicalSpace X
    inst✝ : OrderTopology X
    a : X
    s t : Set X
    hd : Disjoint s (closure t)
    ha : Membership.mem s a
    ⊢ Membership.mem (nhds a) (HasCompl.compl (s.ordSeparatingSet t).ordConnectedS …
  -/
  rw [← nhdsLE_sup_nhdsGE, mem_sup]
  exact ⟨compl_ordConnectedSection_ordSeparatingSet_mem_nhdsLE hd ha,
    compl_ordConnectedSection_ordSeparatingSet_mem_nhdsGE hd ha⟩


@[deprecated (since := "2024-12-22")]
alias compl_section_ordSeparatingSet_mem_nhds := compl_ordConnectedSection_ordSeparatingSet_mem_nhds


theorem ordT5Nhd_mem_nhdsSet (hd : Disjoint s (closure t)) : ordT5Nhd s t ∈ 𝓝ˢ s :=
  bUnion_mem_nhdsSet fun x hx => ordConnectedComponent_mem_nhds.2 <| inter_mem
    (by
      /-
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        s t : Set X
        hd : Disjoint s (closure t)
        x : X
        hx : Membership.mem s x
        ⊢ Membership.mem (nhds x) (HasCompl.compl t)
      -/
      rw [← mem_interior_iff_mem_nhds, interior_compl]
      /-
        X : Type u_1
        inst✝² : LinearOrder X
        inst✝¹ : TopologicalSpace X
        inst✝ : OrderTopology X
        s t : Set X
        hd : Disjoint s (closure t)
        x : X
        hx : Membership.mem s x
        ⊢ Membership.mem (HasCompl.compl (closure t)) x
      -/
      exact disjoint_left.1 hd hx)
      /-
        🎉 no goals
      -/
    (compl_ordConnectedSection_ordSeparatingSet_mem_nhds hd hx)


/-- A linear order with order topology is a completely normal Hausdorff topological space. -/
instance (priority := 100) OrderTopology.completelyNormalSpace : CompletelyNormalSpace X :=
  ⟨fun s t h₁ h₂ => Filter.disjoint_iff.2
    ⟨ordT5Nhd s t, ordT5Nhd_mem_nhdsSet h₂, ordT5Nhd t s, ordT5Nhd_mem_nhdsSet h₁.symm,
      disjoint_ordT5Nhd⟩⟩


instance (priority := 100) OrderTopology.t5Space : T5Space X := T5Space.mk

