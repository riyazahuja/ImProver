/-- A compactness property for a complete lattice is that any `sup`-closed non-empty subset
contains its `sSup`. -/
def IsSupClosedCompact : Prop :=
  ∀ (s : Set α) (_ : s.Nonempty), SupClosed s → sSup s ∈ s


/-- A compactness property for a complete lattice is that any subset has a finite subset with the
same `sSup`. -/
def IsSupFiniteCompact : Prop :=
  ∀ s : Set α, ∃ t : Finset α, ↑t ⊆ s ∧ sSup s = t.sup id


/-- An element `k` of a complete lattice is said to be compact if any set with `sSup`
above `k` has a finite subset with `sSup` above `k`.  Such an element is also called
"finite" or "S-compact". -/
def IsCompactElement {α : Type*} [CompleteLattice α] (k : α) :=
  ∀ s : Set α, k ≤ sSup s → ∃ t : Finset α, ↑t ⊆ s ∧ k ≤ t.sup id


theorem isCompactElement_iff.{u} {α : Type u} [CompleteLattice α] (k : α) :
    CompleteLattice.IsCompactElement k ↔
      ∀ (ι : Type u) (s : ι → α), k ≤ iSup s → ∃ t : Finset ι, k ≤ t.sup s := by
  classical
    constructor
    · intro H ι s hs
      obtain ⟨t, ht, ht'⟩ := H (Set.range s) hs
      have : ∀ x : t, ∃ i, s i = x := fun x => ht x.prop
      choose f hf using this
      refine ⟨Finset.univ.image f, ht'.trans ?_⟩
      rw [Finset.sup_le_iff]
      intro b hb
      rw [← show s (f ⟨b, hb⟩) = id b from hf _]
      exact Finset.le_sup (Finset.mem_image_of_mem f <| Finset.mem_univ (Subtype.mk b hb))
    · intro H s hs
      obtain ⟨t, ht⟩ :=
        H s Subtype.val
          (by
            delta iSup
            rwa [Subtype.range_coe])
      refine ⟨t.image Subtype.val, by simp, ht.trans ?_⟩
      rw [Finset.sup_le_iff]
      exact fun x hx => @Finset.le_sup _ _ _ _ _ id _ (Finset.mem_image_of_mem Subtype.val hx)


/-- An element `k` is compact if and only if any directed set with `sSup` above
`k` already got above `k` at some point in the set. -/
theorem isCompactElement_iff_le_of_directed_sSup_le (k : α) :
    IsCompactElement k ↔
      ∀ s : Set α, s.Nonempty → DirectedOn (· ≤ ·) s → k ≤ sSup s → ∃ x : α, x ∈ s ∧ k ≤ x := by
  classical
    constructor
    · intro hk s hne hdir hsup
      obtain ⟨t, ht⟩ := hk s hsup
      -- certainly every element of t is below something in s, since ↑t ⊆ s.
      have t_below_s : ∀ x ∈ t, ∃ y ∈ s, x ≤ y := fun x hxt => ⟨x, ht.left hxt, le_rfl⟩
      obtain ⟨x, ⟨hxs, hsupx⟩⟩ := Finset.sup_le_of_le_directed s hne hdir t t_below_s
      exact ⟨x, ⟨hxs, le_trans ht.right hsupx⟩⟩
    · intro hk s hsup
      -- Consider the set of finite joins of elements of the (plain) set s.
      let S : Set α := { x | ∃ t : Finset α, ↑t ⊆ s ∧ x = t.sup id }
      -- S is directed, nonempty, and still has sup above k.
      have dir_US : DirectedOn (· ≤ ·) S := by
        rintro x ⟨c, hc⟩ y ⟨d, hd⟩
        use x ⊔ y
        constructor
        · use c ∪ d
          constructor
          · simp only [hc.left, hd.left, Set.union_subset_iff, Finset.coe_union, and_self_iff]
          · simp only [hc.right, hd.right, Finset.sup_union]
        simp only [and_self_iff, le_sup_left, le_sup_right]
      have sup_S : sSup s ≤ sSup S := by
        apply sSup_le_sSup
        intro x hx
        use {x}
        simpa only [and_true, id, Finset.coe_singleton, eq_self_iff_true,
          Finset.sup_singleton, Set.singleton_subset_iff]
      have Sne : S.Nonempty := by
        suffices ⊥ ∈ S from Set.nonempty_of_mem this
        use ∅
        simp only [Set.empty_subset, Finset.coe_empty, Finset.sup_empty, eq_self_iff_true,
          and_self_iff]
      -- Now apply the defn of compact and finish.
      obtain ⟨j, ⟨hjS, hjk⟩⟩ := hk S Sne dir_US (le_trans hsup sup_S)
      obtain ⟨t, ⟨htS, htsup⟩⟩ := hjS
      use t
      exact ⟨htS, by rwa [← htsup]⟩


theorem IsCompactElement.exists_finset_of_le_iSup {k : α} (hk : IsCompactElement k) {ι : Type*}
    (f : ι → α) (h : k ≤ ⨆ i, f i) : ∃ s : Finset ι, k ≤ ⨆ i ∈ s, f i := by
  classical
    let g : Finset ι → α := fun s => ⨆ i ∈ s, f i
    have h1 : DirectedOn (· ≤ ·) (Set.range g) := by
      rintro - ⟨s, rfl⟩ - ⟨t, rfl⟩
      exact
        ⟨g (s ∪ t), ⟨s ∪ t, rfl⟩, iSup_le_iSup_of_subset Finset.subset_union_left,
          iSup_le_iSup_of_subset Finset.subset_union_right⟩
    have h2 : k ≤ sSup (Set.range g) :=
      h.trans
        (iSup_le fun i =>
          le_sSup_of_le ⟨{i}, rfl⟩
            (le_iSup_of_le i (le_iSup_of_le (Finset.mem_singleton_self i) le_rfl)))
    obtain ⟨-, ⟨s, rfl⟩, hs⟩ :=
      (isCompactElement_iff_le_of_directed_sSup_le α k).mp hk (Set.range g) (Set.range_nonempty g)
        h1 h2
    exact ⟨s, hs⟩


/-- A compact element `k` has the property that any directed set lying strictly below `k` has
its `sSup` strictly below `k`. -/
theorem IsCompactElement.directed_sSup_lt_of_lt {α : Type*} [CompleteLattice α] {k : α}
    (hk : IsCompactElement k) {s : Set α} (hemp : s.Nonempty) (hdir : DirectedOn (· ≤ ·) s)
    (hbelow : ∀ x ∈ s, x < k) : sSup s < k := by
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : CompleteLattice.IsCompactElement k
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    ⊢ LT.lt (SupSet.sSup s) k
  -/
  rw [isCompactElement_iff_le_of_directed_sSup_le] at hk
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    ⊢ LT.lt (SupSet.sSup s) k
  -/
  by_contra h
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    h : Not (LT.lt (SupSet.sSup s) k)
    ⊢ False
  -/
  have sSup' : sSup s ≤ k := sSup_le s k fun s hs => (hbelow s hs).le
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    h : Not (LT.lt (SupSet.sSup s) k)
    sSup' : LE.le (SupSet.sSup s) k
    ⊢ False
  -/
  replace sSup : sSup s = k := eq_iff_le_not_lt.mpr ⟨sSup', h⟩
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    h : Not (LT.lt (SupSet.sSup s) k)
    sSup' : LE.le (SupSet.sSup s) k
    sSup : Eq (SupSet.sSup s) k
    ⊢ False
  -/
  obtain ⟨x, hxs, hkx⟩ := hk s hemp hdir sSup.symm.le
  /-
    case intro.intro
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    h : Not (LT.lt (SupSet.sSup s) k)
    sSup' : LE.le (SupSet.sSup s) k
    sSup : Eq (SupSet.sSup s) k
    x : α
    hxs : Membership.mem s x
    hkx : LE.le k x
    ⊢ False
  -/
  obtain hxk := hbelow x hxs
  /-
    case intro.intro
    α : Type u_3
    inst✝ : CompleteLattice α
    k : α
    hk : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
    s : Set α
    hemp : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hbelow : ∀ (x : α), Membership.mem s x → LT.lt x k
    h : Not (LT.lt (SupSet.sSup s) k)
    sSup' : LE.le (SupSet.sSup s) k
    sSup : Eq (SupSet.sSup s) k
    x : α
    hxs : Membership.mem s x
    hkx : LE.le k x
    hxk : LT.lt x k
    ⊢ False
  -/
  exact hxk.ne (hxk.le.antisymm hkx)
  /-
    🎉 no goals
  -/


theorem isCompactElement_finsetSup {α β : Type*} [CompleteLattice α] {f : β → α} (s : Finset β)
    (h : ∀ x ∈ s, IsCompactElement (f x)) : IsCompactElement (s.sup f) := by
  classical
    rw [isCompactElement_iff_le_of_directed_sSup_le]
    intro d hemp hdir hsup
    rw [← Function.id_comp f]
    rw [← Finset.sup_image]
    apply Finset.sup_le_of_le_directed d hemp hdir
    rintro x hx
    obtain ⟨p, ⟨hps, rfl⟩⟩ := Finset.mem_image.mp hx
    specialize h p hps
    rw [isCompactElement_iff_le_of_directed_sSup_le] at h
    specialize h d hemp hdir (le_trans (Finset.le_sup hps) hsup)
    simpa only [exists_prop]


theorem WellFoundedGT.isSupFiniteCompact [WellFoundedGT α] :
    IsSupFiniteCompact α := fun s => by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedGT α
    s : Set α
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (Eq (SupSet.sSup s) (t.sup id))
  -/
  let S := { x | ∃ t : Finset α, ↑t ⊆ s ∧ t.sup id = x }
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedGT α
    s : Set α
    S : Set α := setOf fun x => Exists fun t => And (HasSubset.Subset (↑t) s) (Eq  …
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (Eq (SupSet.sSup s) (t.sup id))
  -/
  obtain ⟨m, ⟨t, ⟨ht₁, rfl⟩⟩, hm⟩ := wellFounded_gt.has_min S ⟨⊥, ∅, by simp⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedGT α
    s : Set α
    S : Set α := setOf fun x => Exists fun t => And (HasSubset.Subset (↑t) s) (Eq  …
    t : Finset α
    ht₁ : HasSubset.Subset (↑t) s
    hm : ∀ (x : α), Membership.mem S x → Not (GT.gt x (t.sup id))
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (Eq (SupSet.sSup s) (t.sup id))
  -/
  refine ⟨t, ht₁, (sSup_le _ _ fun y hy => ?_).antisymm ?_⟩
  · classical
    rw [eq_of_le_of_not_lt (Finset.sup_mono (t.subset_insert y))
        (hm _ ⟨insert y t, by simp [Set.insert_subset_iff, hy, ht₁]⟩)]
    simp
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : WellFoundedGT α
      s : Set α
      S : Set α := setOf fun x => Exists fun t => And (HasSubset.Subset (↑t) s) (Eq  …
      t : Finset α
      ht₁ : HasSubset.Subset (↑t) s
      hm : ∀ (x : α), Membership.mem S x → Not (GT.gt x (t.sup id))
      ⊢ LE.le (t.sup id) (SupSet.sSup s)
    -/
  · rw [Finset.sup_id_eq_sSup]
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : WellFoundedGT α
      s : Set α
      S : Set α := setOf fun x => Exists fun t => And (HasSubset.Subset (↑t) s) (Eq  …
      t : Finset α
      ht₁ : HasSubset.Subset (↑t) s
      hm : ∀ (x : α), Membership.mem S x → Not (GT.gt x (t.sup id))
      ⊢ LE.le (SupSet.sSup ↑t) (SupSet.sSup s)
    -/
    exact sSup_le_sSup ht₁
    /-
      🎉 no goals
    -/


theorem IsSupFiniteCompact.isSupClosedCompact (h : IsSupFiniteCompact α) :
    IsSupClosedCompact α := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    h : CompleteLattice.IsSupFiniteCompact α
    ⊢ CompleteLattice.IsSupClosedCompact α
  -/
  intro s hne hsc; obtain ⟨t, ht₁, ht₂⟩ := h s; clear h
  /-
    case intro.intro
    α : Type u_2
    inst✝ : CompleteLattice α
    s : Set α
    hne : s.Nonempty
    hsc : SupClosed s
    t : Finset α
    ht₁ : HasSubset.Subset (↑t) s
    ht₂ : Eq (SupSet.sSup s) (t.sup id)
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  rcases t.eq_empty_or_nonempty with h | h
    /-
      case intro.intro.inl
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      t : Finset α
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : Eq (SupSet.sSup s) (t.sup id)
      h : Eq t EmptyCollection.emptyCollection
      ⊢ Membership.mem s (SupSet.sSup s)
    -/
  · subst h
    /-
      case intro.intro.inl
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      ht₁ : HasSubset.Subset (↑EmptyCollection.emptyCollection) s
      ht₂ : Eq (SupSet.sSup s) (EmptyCollection.emptyCollection.sup id)
      ⊢ Membership.mem s (SupSet.sSup s)
    -/
    rw [Finset.sup_empty] at ht₂
    /-
      case intro.intro.inl
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      ht₁ : HasSubset.Subset (↑EmptyCollection.emptyCollection) s
      ht₂ : Eq (SupSet.sSup s) Bot.bot
      ⊢ Membership.mem s (SupSet.sSup s)
    -/
    rw [ht₂]
    /-
      case intro.intro.inl
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      ht₁ : HasSubset.Subset (↑EmptyCollection.emptyCollection) s
      ht₂ : Eq (SupSet.sSup s) Bot.bot
      ⊢ Membership.mem s Bot.bot
    -/
    simp [eq_singleton_bot_of_sSup_eq_bot_of_nonempty ht₂ hne]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      t : Finset α
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : Eq (SupSet.sSup s) (t.sup id)
      h : t.Nonempty
      ⊢ Membership.mem s (SupSet.sSup s)
    -/
  · rw [ht₂]
    /-
      case intro.intro.inr
      α : Type u_2
      inst✝ : CompleteLattice α
      s : Set α
      hne : s.Nonempty
      hsc : SupClosed s
      t : Finset α
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : Eq (SupSet.sSup s) (t.sup id)
      h : t.Nonempty
      ⊢ Membership.mem s (t.sup id)
    -/
    exact hsc.finsetSup_mem h ht₁
    /-
      🎉 no goals
    -/


theorem IsSupClosedCompact.wellFoundedGT (h : IsSupClosedCompact α) :
    WellFoundedGT α where
  wf := by
    /-
      α : Type u_2
      inst✝ : CompleteLattice α
      h : CompleteLattice.IsSupClosedCompact α
      ⊢ WellFounded fun x1 x2 => GT.gt x1 x2
    -/
    refine RelEmbedding.wellFounded_iff_no_descending_seq.mpr ⟨fun a => ?_⟩
    suffices sSup (Set.range a) ∈ Set.range a by
      obtain ⟨n, hn⟩ := Set.mem_range.mp this
      have h' : sSup (Set.range a) < a (n + 1) := by
        change _ > _
        simp [← hn, a.map_rel_iff]
      apply lt_irrefl (a (n + 1))
      apply lt_of_le_of_lt _ h'
      apply le_sSup
      apply Set.mem_range_self
    /-
      α : Type u_2
      inst✝ : CompleteLattice α
      h : CompleteLattice.IsSupClosedCompact α
      a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      ⊢ Membership.mem (Set.range ⇑a) (SupSet.sSup (Set.range ⇑a))
    -/
    apply h (Set.range a)
      /-
        case x
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        ⊢ (Set.range ⇑a).Nonempty
      -/
    · use a 37
      /-
        case h
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        ⊢ Membership.mem (Set.range ⇑a) (a 37)
      -/
      apply Set.mem_range_self
      /-
        🎉 no goals
      -/
      /-
        case a
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        ⊢ SupClosed (Set.range ⇑a)
      -/
    · rintro x ⟨m, hm⟩ y ⟨n, hn⟩
      /-
        case a.intro.intro
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        x : α
        m : Nat
        hm : Eq (a m) x
        y : α
        n : Nat
        hn : Eq (a n) y
        ⊢ Membership.mem (Set.range ⇑a) (Max.max x y)
      -/
      use m ⊔ n
      /-
        case h
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        x : α
        m : Nat
        hm : Eq (a m) x
        y : α
        n : Nat
        hn : Eq (a n) y
        ⊢ Eq (a (Max.max m n)) (Max.max x y)
      -/
      rw [← hm, ← hn]
      /-
        case h
        α : Type u_2
        inst✝ : CompleteLattice α
        h : CompleteLattice.IsSupClosedCompact α
        a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
        x : α
        m : Nat
        hm : Eq (a m) x
        y : α
        n : Nat
        hn : Eq (a n) y
        ⊢ Eq (a (Max.max m n)) (Max.max (a m) (a n))
      -/
      apply RelHomClass.map_sup a
      /-
        🎉 no goals
      -/


theorem isSupFiniteCompact_iff_all_elements_compact :
    IsSupFiniteCompact α ↔ ∀ k : α, IsCompactElement k := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    ⊢ Iff (CompleteLattice.IsSupFiniteCompact α) (∀ (k : α), CompleteLattice.IsCom …
  -/
  refine ⟨fun h k s hs => ?_, fun h s => ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : CompleteLattice α
      h : CompleteLattice.IsSupFiniteCompact α
      k : α
      s : Set α
      hs : LE.le k (SupSet.sSup s)
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (LE.le k (t.sup id))
    -/
  · obtain ⟨t, ⟨hts, htsup⟩⟩ := h s
    /-
      case refine_1.intro.intro
      α : Type u_2
      inst✝ : CompleteLattice α
      h : CompleteLattice.IsSupFiniteCompact α
      k : α
      s : Set α
      hs : LE.le k (SupSet.sSup s)
      t : Finset α
      hts : HasSubset.Subset (↑t) s
      htsup : Eq (SupSet.sSup s) (t.sup id)
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (LE.le k (t.sup id))
    -/
    use t, hts
    /-
      case right
      α : Type u_2
      inst✝ : CompleteLattice α
      h : CompleteLattice.IsSupFiniteCompact α
      k : α
      s : Set α
      hs : LE.le k (SupSet.sSup s)
      t : Finset α
      hts : HasSubset.Subset (↑t) s
      htsup : Eq (SupSet.sSup s) (t.sup id)
      ⊢ LE.le k (t.sup id)
    -/
    rwa [← htsup]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : CompleteLattice α
      h : ∀ (k : α), CompleteLattice.IsCompactElement k
      s : Set α
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (Eq (SupSet.sSup s) (t.sup id))
    -/
  · obtain ⟨t, ⟨hts, htsup⟩⟩ := h (sSup s) s (by rfl)
    have : sSup s = t.sup id := by
      suffices t.sup id ≤ sSup s by apply le_antisymm <;> assumption
      simp only [id, Finset.sup_le_iff]
      intro x hx
      exact le_sSup _ _ (hts hx)
    /-
      case refine_2.intro.intro
      α : Type u_2
      inst✝ : CompleteLattice α
      h : ∀ (k : α), CompleteLattice.IsCompactElement k
      s : Set α
      t : Finset α
      hts : HasSubset.Subset (↑t) s
      htsup : LE.le (SupSet.sSup s) (t.sup id)
      this : Eq (SupSet.sSup s) (t.sup id)
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (Eq (SupSet.sSup s) (t.sup id))
    -/
    exact ⟨t, hts, this⟩
    /-
      🎉 no goals
    -/


open List in
theorem wellFoundedGT_characterisations : List.TFAE
    [WellFoundedGT α, IsSupFiniteCompact α, IsSupClosedCompact α, ∀ k : α, IsCompactElement k] := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    ⊢ (List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteCompact  …
  -/
  tfae_have 1 → 2 := @WellFoundedGT.isSupFiniteCompact α _
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    tfae_1_to_2 : WellFoundedGT α → CompleteLattice.IsSupFiniteCompact α
    ⊢ (List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteCompact  …
  -/
  tfae_have 2 → 3 := IsSupFiniteCompact.isSupClosedCompact α
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    tfae_1_to_2 : WellFoundedGT α → CompleteLattice.IsSupFiniteCompact α
    tfae_2_to_3 : CompleteLattice.IsSupFiniteCompact α → CompleteLattice.IsSupClos …
    ⊢ (List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteCompact  …
  -/
  tfae_have 3 → 1 := IsSupClosedCompact.wellFoundedGT α
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    tfae_1_to_2 : WellFoundedGT α → CompleteLattice.IsSupFiniteCompact α
    tfae_2_to_3 : CompleteLattice.IsSupFiniteCompact α → CompleteLattice.IsSupClos …
    tfae_3_to_1 : CompleteLattice.IsSupClosedCompact α → WellFoundedGT α
    ⊢ (List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteCompact  …
  -/
  tfae_have 2 ↔ 4 := isSupFiniteCompact_iff_all_elements_compact α
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    tfae_1_to_2 : WellFoundedGT α → CompleteLattice.IsSupFiniteCompact α
    tfae_2_to_3 : CompleteLattice.IsSupFiniteCompact α → CompleteLattice.IsSupClos …
    tfae_3_to_1 : CompleteLattice.IsSupClosedCompact α → WellFoundedGT α
    tfae_2_iff_4 : Iff (CompleteLattice.IsSupFiniteCompact α) (∀ (k : α), Complete …
    ⊢ (List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteCompact  …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem wellFoundedGT_iff_isSupFiniteCompact :
    WellFoundedGT α ↔ IsSupFiniteCompact α :=
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    ⊢ Eq ((List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteComp …
  -/
  /-
    🎉 no goals
  -/
  (wellFoundedGT_characterisations α).out 0 1
  /-
    🎉 no goals
  -/


theorem isSupFiniteCompact_iff_isSupClosedCompact : IsSupFiniteCompact α ↔ IsSupClosedCompact α :=
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    ⊢ Eq ((List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteComp …
  -/
  /-
    🎉 no goals
  -/
  (wellFoundedGT_characterisations α).out 1 2
  /-
    🎉 no goals
  -/


theorem isSupClosedCompact_iff_wellFoundedGT :
    IsSupClosedCompact α ↔ WellFoundedGT α :=
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    ⊢ Eq ((List.cons (WellFoundedGT α) (List.cons (CompleteLattice.IsSupFiniteComp …
  -/
  /-
    🎉 no goals
  -/
  (wellFoundedGT_characterisations α).out 2 0
  /-
    🎉 no goals
  -/


alias ⟨_, IsSupFiniteCompact.wellFoundedGT⟩ := wellFoundedGT_iff_isSupFiniteCompact


alias ⟨_, IsSupClosedCompact.isSupFiniteCompact⟩ := isSupFiniteCompact_iff_isSupClosedCompact


alias ⟨_, WellFoundedGT.isSupClosedCompact⟩ := isSupClosedCompact_iff_wellFoundedGT


theorem WellFoundedGT.finite_of_sSupIndep [WellFoundedGT α] {s : Set α}
    (hs : sSupIndep s) : s.Finite := by
  classical
    refine Set.not_infinite.mp fun contra => ?_
    obtain ⟨t, ht₁, ht₂⟩ := CompleteLattice.WellFoundedGT.isSupFiniteCompact α s
    replace contra : ∃ x : α, x ∈ s ∧ x ≠ ⊥ ∧ x ∉ t := by
      have : (s \ (insert ⊥ t : Finset α)).Infinite := contra.diff (Finset.finite_toSet _)
      obtain ⟨x, hx₁, hx₂⟩ := this.nonempty
      exact ⟨x, hx₁, by simpa [not_or] using hx₂⟩
    obtain ⟨x, hx₀, hx₁, hx₂⟩ := contra
    replace hs : x ⊓ sSup s = ⊥ := by
      have := hs.mono (by simp [ht₁, hx₀, -Set.union_singleton] : ↑t ∪ {x} ≤ s) (by simp : x ∈ _)
      simpa [Disjoint, hx₂, ← t.sup_id_eq_sSup, ← ht₂] using this.eq_bot
    apply hx₁
    rw [← hs, eq_comm, inf_eq_left]
    exact le_sSup hx₀


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedGT.finite_of_setIndependent := WellFoundedGT.finite_of_sSupIndep


theorem WellFoundedGT.finite_ne_bot_of_iSupIndep [WellFoundedGT α]
    {ι : Type*} {t : ι → α} (ht : iSupIndep t) : Set.Finite {i | t i ≠ ⊥} := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedGT α
    ι : Type u_3
    t : ι → α
    ht : iSupIndep t
    ⊢ (setOf fun i => Ne (t i) Bot.bot).Finite
  -/
  refine Finite.of_finite_image (Finite.subset ?_ (image_subset_range t _)) ht.injOn
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedGT α
    ι : Type u_3
    t : ι → α
    ht : iSupIndep t
    ⊢ (Set.range t).Finite
  -/
  exact WellFoundedGT.finite_of_sSupIndep ht.sSupIndep_range
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedGT.finite_ne_bot_of_independent :=
  WellFoundedGT.finite_ne_bot_of_iSupIndep


theorem WellFoundedGT.finite_of_iSupIndep [WellFoundedGT α] {ι : Type*}
    {t : ι → α} (ht : iSupIndep t) (h_ne_bot : ∀ i, t i ≠ ⊥) : Finite ι :=
  haveI := (WellFoundedGT.finite_of_sSupIndep ht.sSupIndep_range).to_subtype
  Finite.of_injective_finite_range (ht.injective h_ne_bot)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedGT.finite_of_independent := WellFoundedGT.finite_of_iSupIndep


theorem WellFoundedLT.finite_of_sSupIndep [WellFoundedLT α] {s : Set α}
    (hs : sSupIndep s) : s.Finite := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    s : Set α
    hs : sSupIndep s
    ⊢ s.Finite
  -/
  by_contra inf
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    s : Set α
    hs : sSupIndep s
    inf : Not s.Finite
    ⊢ False
  -/
  let e := (Infinite.diff inf <| finite_singleton ⊥).to_subtype.natEmbedding
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    s : Set α
    hs : sSupIndep s
    inf : Not s.Finite
    e : Function.Embedding Nat ↑(SDiff.sdiff s (Singleton.singleton Bot.bot)) := I …
    ⊢ False
  -/
  let a n := ⨆ i ≥ n, (e i).1
  have sup_le n : (e n).1 ⊔ a (n + 1) ≤ a n := sup_le_iff.mpr ⟨le_iSup₂_of_le n le_rfl le_rfl,
    iSup₂_le fun i hi ↦ le_iSup₂_of_le i (n.le_succ.trans hi) le_rfl⟩
  have lt n : a (n + 1) < a n := (Disjoint.right_lt_sup_of_left_ne_bot
    ((hs (e n).2.1).mono_right <| iSup₂_le fun i hi ↦ le_sSup ?_) (e n).2.2).trans_le (sup_le n)
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : WellFoundedLT α
      s : Set α
      hs : sSupIndep s
      inf : Not s.Finite
      e : Function.Embedding Nat ↑(SDiff.sdiff s (Singleton.singleton Bot.bot)) := I …
      a : Nat → α := fun n => iSup fun i => iSup fun h => ↑(e i)
      sup_le : ∀ (n : Nat), LE.le (Max.max (↑(e n)) (a (HAdd.hAdd n 1))) (a n)
      lt : ∀ (n : Nat), LT.lt (a (HAdd.hAdd n 1)) (a n)
      ⊢ False
    -/
  · exact (RelEmbedding.natGT a lt).not_wellFounded_of_decreasing_seq wellFounded_lt
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    s : Set α
    hs : sSupIndep s
    inf : Not s.Finite
    e : Function.Embedding Nat ↑(SDiff.sdiff s (Singleton.singleton Bot.bot)) := I …
    a : Nat → α := fun n => iSup fun i => iSup fun h => ↑(e i)
    sup_le : ∀ (n : Nat), LE.le (Max.max (↑(e n)) (a (HAdd.hAdd n 1))) (a n)
    n i : Nat
    hi : GE.ge i (HAdd.hAdd n 1)
    ⊢ Membership.mem (SDiff.sdiff s (Singleton.singleton ↑(e n))) ↑(e i)
  -/
  exact ⟨(e i).2.1, fun h ↦ n.lt_succ_self.not_le <| hi.trans_eq <| e.2 <| Subtype.val_injective h⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedLT.finite_of_setIndependent := WellFoundedLT.finite_of_sSupIndep


theorem WellFoundedLT.finite_ne_bot_of_iSupIndep [WellFoundedLT α]
    {ι : Type*} {t : ι → α} (ht : iSupIndep t) : Set.Finite {i | t i ≠ ⊥} := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    ι : Type u_3
    t : ι → α
    ht : iSupIndep t
    ⊢ (setOf fun i => Ne (t i) Bot.bot).Finite
  -/
  refine Finite.of_finite_image (Finite.subset ?_ (image_subset_range t _)) ht.injOn
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : WellFoundedLT α
    ι : Type u_3
    t : ι → α
    ht : iSupIndep t
    ⊢ (Set.range t).Finite
  -/
  exact WellFoundedLT.finite_of_sSupIndep ht.sSupIndep_range
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedLT.finite_ne_bot_of_independent :=
  WellFoundedLT.finite_ne_bot_of_iSupIndep


theorem WellFoundedLT.finite_of_iSupIndep [WellFoundedLT α] {ι : Type*}
    {t : ι → α} (ht : iSupIndep t) (h_ne_bot : ∀ i, t i ≠ ⊥) : Finite ι :=
  haveI := (WellFoundedLT.finite_of_sSupIndep ht.sSupIndep_range).to_subtype
  Finite.of_injective_finite_range (ht.injective h_ne_bot)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.WellFoundedLT.finite_of_independent := WellFoundedLT.finite_of_iSupIndep


/-- A complete lattice is said to be compactly generated if any
element is the `sSup` of compact elements. -/
class IsCompactlyGenerated (α : Type*) [CompleteLattice α] : Prop where
  /-- In a compactly generated complete lattice,
    every element is the `sSup` of some set of compact elements. -/
  exists_sSup_eq : ∀ x : α, ∃ s : Set α, (∀ x ∈ s, CompleteLattice.IsCompactElement x) ∧ sSup s = x


@[simp]
theorem sSup_compact_le_eq (b) :
    sSup { c : α | CompleteLattice.IsCompactElement c ∧ c ≤ b } = b := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    b : α
    ⊢ Eq (SupSet.sSup (setOf fun c => And (CompleteLattice.IsCompactElement c) (LE …
  -/
  rcases IsCompactlyGenerated.exists_sSup_eq b with ⟨s, hs, rfl⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → CompleteLattice.IsCompactElement x
    ⊢ Eq (SupSet.sSup (setOf fun c => And (CompleteLattice.IsCompactElement c) (LE …
  -/
  exact le_antisymm (sSup_le fun c hc => hc.2) (sSup_le_sSup fun c cs => ⟨hs c cs, le_sSup cs⟩)
  /-
    🎉 no goals
  -/


@[simp]
theorem sSup_compact_eq_top : sSup { a : α | CompleteLattice.IsCompactElement a } = ⊤ := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    ⊢ Eq (SupSet.sSup (setOf fun a => CompleteLattice.IsCompactElement a)) Top.top
  -/
  refine Eq.trans (congr rfl (Set.ext fun x => ?_)) (sSup_compact_le_eq ⊤)
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    x : α
    ⊢ Iff (Membership.mem (setOf fun a => CompleteLattice.IsCompactElement a) x) ( …
  -/
  exact (and_iff_left le_top).symm
  /-
    🎉 no goals
  -/


theorem le_iff_compact_le_imp {a b : α} :
    a ≤ b ↔ ∀ c : α, CompleteLattice.IsCompactElement c → c ≤ a → c ≤ b :=
  ⟨fun ab _ _ ca => le_trans ca ab, fun h => by
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a b : α
      h : ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c a → LE.le c b
      ⊢ LE.le a b
    -/
    rw [← sSup_compact_le_eq a, ← sSup_compact_le_eq b]
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      a b : α
      h : ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c a → LE.le c b
      ⊢ LE.le (SupSet.sSup (setOf fun c => And (CompleteLattice.IsCompactElement c)  …
    -/
    exact sSup_le_sSup fun c hc => ⟨hc.1, h c hc.1 hc.2⟩⟩
    /-
      🎉 no goals
    -/


/-- This property is sometimes referred to as `α` being upper continuous. -/
theorem DirectedOn.inf_sSup_eq (h : DirectedOn (· ≤ ·) s) : a ⊓ sSup s = ⨆ b ∈ s, a ⊓ b :=
  le_antisymm
    (by
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
        ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
      -/
      rw [le_iff_compact_le_imp]
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
        ⊢ ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c (Min.min a (SupSet.s …
      -/
      by_cases hs : s.Nonempty
        /-
          case pos
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          ⊢ ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c (Min.min a (SupSet.s …
        -/
      · intro c hc hcinf
        /-
          case pos
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          c : α
          hc : CompleteLattice.IsCompactElement c
          hcinf : LE.le c (Min.min a (SupSet.sSup s))
          ⊢ LE.le c (iSup fun b => iSup fun h => Min.min a b)
        -/
        rw [le_inf_iff] at hcinf
        /-
          case pos
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          c : α
          hc : CompleteLattice.IsCompactElement c
          hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
          ⊢ LE.le c (iSup fun b => iSup fun h => Min.min a b)
        -/
        rw [CompleteLattice.isCompactElement_iff_le_of_directed_sSup_le] at hc
        /-
          case pos
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          c : α
          hc : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
          hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
          ⊢ LE.le c (iSup fun b => iSup fun h => Min.min a b)
        -/
        rcases hc s hs h hcinf.2 with ⟨d, ds, cd⟩
        /-
          case pos.intro.intro
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          c : α
          hc : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
          hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
          d : α
          ds : Membership.mem s d
          cd : LE.le c d
          ⊢ LE.le c (iSup fun b => iSup fun h => Min.min a b)
        -/
        refine (le_inf hcinf.1 cd).trans (le_trans ?_ (le_iSup₂ d ds))
        /-
          case pos.intro.intro
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : s.Nonempty
          c : α
          hc : ∀ (s : Set α), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) s → LE. …
          hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
          d : α
          ds : Membership.mem s d
          cd : LE.le c d
          ⊢ LE.le (Min.min a d) (Min.min a d)
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : Not s.Nonempty
          ⊢ ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c (Min.min a (SupSet.s …
        -/
      · rw [Set.not_nonempty_iff_eq_empty] at hs
        /-
          case neg
          α : Type u_2
          inst✝¹ : CompleteLattice α
          inst✝ : IsCompactlyGenerated α
          a : α
          s : Set α
          h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
          hs : Eq s EmptyCollection.emptyCollection
          ⊢ ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c (Min.min a (SupSet.s …
        -/
        simp [hs])
        /-
          🎉 no goals
        -/
    iSup_inf_le_inf_sSup


/-- This property is sometimes referred to as `α` being upper continuous. -/
protected theorem DirectedOn.sSup_inf_eq (h : DirectedOn (· ≤ ·) s) :
    sSup s ⊓ a = ⨆ b ∈ s, b ⊓ a := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    s : Set α
    h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq (Min.min (SupSet.sSup s) a) (iSup fun b => iSup fun h => Min.min b a)
  -/
  simp_rw [inf_comm _ a, h.inf_sSup_eq]
  /-
    🎉 no goals
  -/


protected theorem Directed.inf_iSup_eq (h : Directed (· ≤ ·) f) :
    (a ⊓ ⨆ i, f i) = ⨆ i, a ⊓ f i := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    f : ι → α
    inst✝ : IsCompactlyGenerated α
    a : α
    h : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Eq (Min.min a (iSup fun i => f i)) (iSup fun i => Min.min a (f i))
  -/
  rw [iSup, h.directedOn_range.inf_sSup_eq, iSup_range]
  /-
    🎉 no goals
  -/


protected theorem Directed.iSup_inf_eq (h : Directed (· ≤ ·) f) :
    (⨆ i, f i) ⊓ a = ⨆ i, f i ⊓ a := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    f : ι → α
    inst✝ : IsCompactlyGenerated α
    a : α
    h : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Eq (Min.min (iSup fun i => f i) a) (iSup fun i => Min.min (f i) a)
  -/
  rw [iSup, h.directedOn_range.sSup_inf_eq, iSup_range]
  /-
    🎉 no goals
  -/


protected theorem DirectedOn.disjoint_sSup_right (h : DirectedOn (· ≤ ·) s) :
    Disjoint a (sSup s) ↔ ∀ ⦃b⦄, b ∈ s → Disjoint a b := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    s : Set α
    h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    ⊢ Iff (Disjoint a (SupSet.sSup s)) (∀ ⦃b : α⦄, Membership.mem s b → Disjoint a …
  -/
  simp_rw [disjoint_iff, h.inf_sSup_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


protected theorem DirectedOn.disjoint_sSup_left (h : DirectedOn (· ≤ ·) s) :
    Disjoint (sSup s) a ↔ ∀ ⦃b⦄, b ∈ s → Disjoint b a := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    a : α
    s : Set α
    h : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    ⊢ Iff (Disjoint (SupSet.sSup s) a) (∀ ⦃b : α⦄, Membership.mem s b → Disjoint b …
  -/
  simp_rw [disjoint_iff, h.sSup_inf_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


protected theorem Directed.disjoint_iSup_right (h : Directed (· ≤ ·) f) :
    Disjoint a (⨆ i, f i) ↔ ∀ i, Disjoint a (f i) := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    f : ι → α
    inst✝ : IsCompactlyGenerated α
    a : α
    h : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Iff (Disjoint a (iSup fun i => f i)) (∀ (i : ι), Disjoint a (f i))
  -/
  simp_rw [disjoint_iff, h.inf_iSup_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


protected theorem Directed.disjoint_iSup_left (h : Directed (· ≤ ·) f) :
    Disjoint (⨆ i, f i) a ↔ ∀ i, Disjoint (f i) a := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝¹ : CompleteLattice α
    f : ι → α
    inst✝ : IsCompactlyGenerated α
    a : α
    h : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Iff (Disjoint (iSup fun i => f i) a) (∀ (i : ι), Disjoint (f i) a)
  -/
  simp_rw [disjoint_iff, h.iSup_inf_eq, iSup_eq_bot]
  /-
    🎉 no goals
  -/


/-- This property is equivalent to `α` being upper continuous. -/
theorem inf_sSup_eq_iSup_inf_sup_finset :
    a ⊓ sSup s = ⨆ (t : Finset α) (_ : ↑t ⊆ s), a ⊓ t.sup id :=
  le_antisymm
    (by
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun t => iSup fun x => Min.min a (t. …
      -/
      rw [le_iff_compact_le_imp]
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        ⊢ ∀ (c : α), CompleteLattice.IsCompactElement c → LE.le c (Min.min a (SupSet.s …
      -/
      intro c hc hcinf
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        c : α
        hc : CompleteLattice.IsCompactElement c
        hcinf : LE.le c (Min.min a (SupSet.sSup s))
        ⊢ LE.le c (iSup fun t => iSup fun x => Min.min a (t.sup id))
      -/
      rw [le_inf_iff] at hcinf
      /-
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        c : α
        hc : CompleteLattice.IsCompactElement c
        hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
        ⊢ LE.le c (iSup fun t => iSup fun x => Min.min a (t.sup id))
      -/
      rcases hc s hcinf.2 with ⟨t, ht1, ht2⟩
      /-
        case intro.intro
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        c : α
        hc : CompleteLattice.IsCompactElement c
        hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
        t : Finset α
        ht1 : HasSubset.Subset (↑t) s
        ht2 : LE.le c (t.sup id)
        ⊢ LE.le c (iSup fun t => iSup fun x => Min.min a (t.sup id))
      -/
      refine (le_inf hcinf.1 ht2).trans (le_trans ?_ (le_iSup₂ t ht1))
      /-
        case intro.intro
        α : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : IsCompactlyGenerated α
        a : α
        s : Set α
        c : α
        hc : CompleteLattice.IsCompactElement c
        hcinf : And (LE.le c a) (LE.le c (SupSet.sSup s))
        t : Finset α
        ht1 : HasSubset.Subset (↑t) s
        ht2 : LE.le c (t.sup id)
        ⊢ LE.le (Min.min a (t.sup id)) (Min.min a (t.sup id))
      -/
      rfl)
      /-
        🎉 no goals
      -/
    (iSup_le fun t =>
      iSup_le fun h => inf_le_inf_left _ ((Finset.sup_id_eq_sSup t).symm ▸ sSup_le_sSup h))


theorem sSupIndep_iff_finite {s : Set α} :
    sSupIndep s ↔
      ∀ t : Finset α, ↑t ⊆ s → sSupIndep (↑t : Set α) :=
  ⟨fun hs _ ht => hs.mono ht, fun h a ha => by
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      s : Set α
      h : ∀ (t : Finset α), HasSubset.Subset (↑t) s → sSupIndep ↑t
      a : α
      ha : Membership.mem s a
      ⊢ Disjoint a (SupSet.sSup (SDiff.sdiff s (Singleton.singleton a)))
    -/
    rw [disjoint_iff, inf_sSup_eq_iSup_inf_sup_finset, iSup_eq_bot]
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      s : Set α
      h : ∀ (t : Finset α), HasSubset.Subset (↑t) s → sSupIndep ↑t
      a : α
      ha : Membership.mem s a
      ⊢ ∀ (i : Finset α), Eq (iSup fun x => Min.min a (i.sup id)) Bot.bot
    -/
    intro t
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      s : Set α
      h : ∀ (t : Finset α), HasSubset.Subset (↑t) s → sSupIndep ↑t
      a : α
      ha : Membership.mem s a
      t : Finset α
      ⊢ Eq (iSup fun x => Min.min a (t.sup id)) Bot.bot
    -/
    rw [iSup_eq_bot, Finset.sup_id_eq_sSup]
    /-
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      s : Set α
      h : ∀ (t : Finset α), HasSubset.Subset (↑t) s → sSupIndep ↑t
      a : α
      ha : Membership.mem s a
      t : Finset α
      ⊢ HasSubset.Subset (↑t) (SDiff.sdiff s (Singleton.singleton a)) → Eq (Min.min  …
    -/
    intro ht
    classical
      have h' := (h (insert a t) ?_ (t.mem_insert_self a)).eq_bot
      · rwa [Finset.coe_insert, Set.insert_diff_self_of_not_mem] at h'
        exact fun con => ((Set.mem_diff a).1 (ht con)).2 (Set.mem_singleton a)
      · rw [Finset.coe_insert, Set.insert_subset_iff]
        exact ⟨ha, Set.Subset.trans ht diff_subset⟩⟩


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.setIndependent_iff_finite := sSupIndep_iff_finite


lemma iSupIndep_iff_supIndep_of_injOn {ι : Type*} {f : ι → α}
    (hf : InjOn f {i | f i ≠ ⊥}) :
    iSupIndep f ↔ ∀ (s : Finset ι), s.SupIndep f := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    ι : Type u_3
    f : ι → α
    hf : Set.InjOn f (setOf fun i => Ne (f i) Bot.bot)
    ⊢ Iff (iSupIndep f) (∀ (s : Finset ι), s.SupIndep f)
  -/
  refine ⟨fun h ↦ h.supIndep', fun h ↦ iSupIndep_def'.mpr fun i ↦ ?_⟩
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    ι : Type u_3
    f : ι → α
    hf : Set.InjOn f (setOf fun i => Ne (f i) Bot.bot)
    h : ∀ (s : Finset ι), s.SupIndep f
    i : ι
    ⊢ Disjoint (f i) (SupSet.sSup (Set.image f (setOf fun j => Ne j i)))
  -/
  simp_rw [disjoint_iff, inf_sSup_eq_iSup_inf_sup_finset, iSup_eq_bot, ← disjoint_iff]
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    ι : Type u_3
    f : ι → α
    hf : Set.InjOn f (setOf fun i => Ne (f i) Bot.bot)
    h : ∀ (s : Finset ι), s.SupIndep f
    i : ι
    ⊢ ∀ (i_1 : Finset α), HasSubset.Subset (↑i_1) (Set.image f (setOf fun j => Ne  …
  -/
  intro s hs
  classical
  rw [← Finset.sup_erase_bot]
  set t := s.erase ⊥
  replace hf : InjOn f (f ⁻¹' t) := fun i hi j _ hij ↦ by
    refine hf ?_ ?_ hij <;> aesop (add norm simp [t])
  have : (Finset.erase (insert i (t.preimage _ hf)) i).image f = t := by
    ext a
    simp only [Finset.mem_preimage, Finset.mem_erase, ne_eq, Finset.mem_insert, true_or, not_true,
      Finset.erase_insert_eq_erase, not_and, Finset.mem_image, t]
    refine ⟨by aesop, fun ⟨ha, has⟩ ↦ ?_⟩
    obtain ⟨j, hj, rfl⟩ := hs has
    exact ⟨j, ⟨hj, ha, has⟩, rfl⟩
  rw [← this, Finset.sup_image]
  specialize h (insert i (t.preimage _ hf))
  rw [Finset.supIndep_iff_disjoint_erase] at h
  exact h i (Finset.mem_insert_self i _)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_iff_supIndep_of_injOn := iSupIndep_iff_supIndep_of_injOn


theorem sSupIndep_iUnion_of_directed {η : Type*} {s : η → Set α}
    (hs : Directed (· ⊆ ·) s) (h : ∀ i, sSupIndep (s i)) :
    sSupIndep (⋃ i, s i) := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    η : Type u_3
    s : η → Set α
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), sSupIndep (s i)
    ⊢ sSupIndep (Set.iUnion fun i => s i)
  -/
  by_cases hη : Nonempty η
    /-
      case pos
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Nonempty η
      ⊢ sSupIndep (Set.iUnion fun i => s i)
    -/
  · rw [sSupIndep_iff_finite]
    /-
      case pos
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Nonempty η
      ⊢ ∀ (t : Finset α), HasSubset.Subset (↑t) (Set.iUnion fun i => s i) → sSupInde …
    -/
    intro t ht
    /-
      case pos
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Nonempty η
      t : Finset α
      ht : HasSubset.Subset (↑t) (Set.iUnion fun i => s i)
      ⊢ sSupIndep ↑t
    -/
    obtain ⟨I, fi, hI⟩ := Set.finite_subset_iUnion t.finite_toSet ht
    /-
      case pos.intro.intro
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Nonempty η
      t : Finset α
      ht : HasSubset.Subset (↑t) (Set.iUnion fun i => s i)
      I : Set η
      fi : I.Finite
      hI : HasSubset.Subset (↑t) (Set.iUnion fun i => Set.iUnion fun h => s i)
      ⊢ sSupIndep ↑t
    -/
    obtain ⟨i, hi⟩ := hs.finset_le fi.toFinset
    exact (h i).mono
        (Set.Subset.trans hI <| Set.iUnion₂_subset fun j hj => hi j (fi.mem_toFinset.2 hj))
    /-
      case neg
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Not (Nonempty η)
      ⊢ sSupIndep (Set.iUnion fun i => s i)
    -/
  · rintro a ⟨_, ⟨i, _⟩, _⟩
    /-
      case neg.intro.intro.intro
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Not (Nonempty η)
      a : α
      w✝ : Set α
      right✝ : Membership.mem w✝ a
      i : η
      h✝ : Eq ((fun i => s i) i) w✝
      ⊢ Disjoint a (SupSet.sSup (SDiff.sdiff (Set.iUnion fun i => s i) (Singleton.si …
    -/
    exfalso
    /-
      case neg.intro.intro.intro
      α : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : IsCompactlyGenerated α
      η : Type u_3
      s : η → Set α
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), sSupIndep (s i)
      hη : Not (Nonempty η)
      a : α
      w✝ : Set α
      right✝ : Membership.mem w✝ a
      i : η
      h✝ : Eq ((fun i => s i) i) w✝
      ⊢ False
    -/
    exact hη ⟨i⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.setIndependent_iUnion_of_directed := sSupIndep_iUnion_of_directed


theorem iSupIndep_sUnion_of_directed {s : Set (Set α)} (hs : DirectedOn (· ⊆ ·) s)
    (h : ∀ a ∈ s, sSupIndep a) : sSupIndep (⋃₀ s) := by
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set (Set α)
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set α), Membership.mem s a → sSupIndep a
    ⊢ sSupIndep s.sUnion
  -/
  rw [Set.sUnion_eq_iUnion]
  /-
    α : Type u_2
    inst✝¹ : CompleteLattice α
    inst✝ : IsCompactlyGenerated α
    s : Set (Set α)
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set α), Membership.mem s a → sSupIndep a
    ⊢ sSupIndep (Set.iUnion fun i => ↑i)
  -/
  exact sSupIndep_iUnion_of_directed hs.directed_val (by simpa using h)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_sUnion_of_directed := iSupIndep_sUnion_of_directed


theorem isCompactlyGenerated_of_wellFoundedGT [h : WellFoundedGT α] :
    IsCompactlyGenerated α := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    h : WellFoundedGT α
    ⊢ IsCompactlyGenerated α
  -/
  rw [wellFoundedGT_iff_isSupFiniteCompact, isSupFiniteCompact_iff_all_elements_compact] at h
  -- x is the join of the set of compact elements {x}
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    h : ∀ (k : α), CompleteLattice.IsCompactElement k
    ⊢ IsCompactlyGenerated α
  -/
  exact ⟨fun x => ⟨{x}, ⟨fun x _ => h x, sSup_singleton⟩⟩⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-07")]
alias WellFounded.isSupFiniteCompact := WellFoundedGT.isSupFiniteCompact

@[deprecated (since := "2024-10-07")]
alias IsSupClosedCompact.wellFounded := IsSupClosedCompact.wellFoundedGT

@[deprecated (since := "2024-10-07")]
alias wellFounded_characterisations := wellFoundedGT_characterisations

@[deprecated (since := "2024-10-07")]
alias wellFounded_iff_isSupFiniteCompact := wellFoundedGT_iff_isSupFiniteCompact

@[deprecated (since := "2024-10-07")]
alias isSupClosedCompact_iff_wellFounded := isSupClosedCompact_iff_wellFoundedGT

@[deprecated (since := "2024-10-07")]
alias IsSupFiniteCompact.wellFounded := IsSupFiniteCompact.wellFoundedGT

@[deprecated (since := "2024-10-07")]
alias _root_.WellFounded.isSupClosedCompact := WellFoundedGT.isSupClosedCompact

@[deprecated (since := "2024-10-07")]
alias WellFounded.finite_of_setIndependent := WellFoundedGT.finite_of_sSupIndep

@[deprecated (since := "2024-10-07")]
alias WellFounded.finite_ne_bot_of_independent := WellFoundedGT.finite_ne_bot_of_iSupIndep

@[deprecated (since := "2024-10-07")]
alias WellFounded.finite_of_independent := WellFoundedGT.finite_of_iSupIndep

@[deprecated (since := "2024-10-07")]
alias isCompactlyGenerated_of_wellFounded := isCompactlyGenerated_of_wellFoundedGT


/-- A compact element `k` has the property that any `b < k` lies below a "maximal element below
`k`", which is to say `[⊥, k]` is coatomic. -/
theorem Iic_coatomic_of_compact_element {k : α} (h : IsCompactElement k) :
    IsCoatomic (Set.Iic k) := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    k : α
    h : CompleteLattice.IsCompactElement k
    ⊢ IsCoatomic ↑(Set.Iic k)
  -/
  constructor
  /-
    case eq_top_or_exists_le_coatom
    α : Type u_2
    inst✝ : CompleteLattice α
    k : α
    h : CompleteLattice.IsCompactElement k
    ⊢ ∀ (b : ↑(Set.Iic k)), Or (Eq b Top.top) (Exists fun a => And (IsCoatom a) (L …
  -/
  rintro ⟨b, hbk⟩
  /-
    case eq_top_or_exists_le_coatom.mk
    α : Type u_2
    inst✝ : CompleteLattice α
    k : α
    h : CompleteLattice.IsCompactElement k
    b : α
    hbk : Membership.mem (Set.Iic k) b
    ⊢ Or (Eq ⟨b, hbk⟩ Top.top) (Exists fun a => And (IsCoatom a) (LE.le ⟨b, hbk⟩ a))
  -/
  obtain rfl | H := eq_or_ne b k
    /-
      case eq_top_or_exists_le_coatom.mk.inl
      α : Type u_2
      inst✝ : CompleteLattice α
      b : α
      h : CompleteLattice.IsCompactElement b
      hbk : Membership.mem (Set.Iic b) b
      ⊢ Or (Eq ⟨b, hbk⟩ Top.top) (Exists fun a => And (IsCoatom a) (LE.le ⟨b, hbk⟩ a))
    -/
  · left; ext; simp only [Set.Iic.coe_top, Subtype.coe_mk]
               /-
                 🎉 no goals
               -/
  /-
    case eq_top_or_exists_le_coatom.mk.inr
    α : Type u_2
    inst✝ : CompleteLattice α
    k : α
    h : CompleteLattice.IsCompactElement k
    b : α
    hbk : Membership.mem (Set.Iic k) b
    H : Ne b k
    ⊢ Or (Eq ⟨b, hbk⟩ Top.top) (Exists fun a => And (IsCoatom a) (LE.le ⟨b, hbk⟩ a))
  -/
  right
  /-
    case eq_top_or_exists_le_coatom.mk.inr.h
    α : Type u_2
    inst✝ : CompleteLattice α
    k : α
    h : CompleteLattice.IsCompactElement k
    b : α
    hbk : Membership.mem (Set.Iic k) b
    H : Ne b k
    ⊢ Exists fun a => And (IsCoatom a) (LE.le ⟨b, hbk⟩ a)
  -/
  have ⟨a, ba, h⟩ := zorn_le_nonempty₀ (Set.Iio k) ?_ b (lt_of_le_of_ne hbk H)
    /-
      case eq_top_or_exists_le_coatom.mk.inr.h.refine_2
      α : Type u_2
      inst✝ : CompleteLattice α
      k : α
      h✝ : CompleteLattice.IsCompactElement k
      b : α
      hbk : Membership.mem (Set.Iic k) b
      H : Ne b k
      a : α
      ba : LE.le b a
      h : Maximal (fun x => Membership.mem (Set.Iio k) x) a
      ⊢ Exists fun a => And (IsCoatom a) (LE.le ⟨b, hbk⟩ a)
    -/
  · refine ⟨⟨a, le_of_lt h.prop⟩, ⟨ne_of_lt h.prop, fun c hck => by_contradiction fun c₀ => ?_⟩, ba⟩
    /-
      case eq_top_or_exists_le_coatom.mk.inr.h.refine_2
      α : Type u_2
      inst✝ : CompleteLattice α
      k : α
      h✝ : CompleteLattice.IsCompactElement k
      b : α
      hbk : Membership.mem (Set.Iic k) b
      H : Ne b k
      a : α
      ba : LE.le b a
      h : Maximal (fun x => Membership.mem (Set.Iio k) x) a
      c : ↑(Set.Iic k)
      hck : LT.lt ⟨a, ⋯⟩ c
      c₀ : Not (Eq c Top.top)
      ⊢ False
    -/
    cases h.eq_of_le (y := c.1) (lt_of_le_of_ne c.2 fun con ↦ c₀ (Subtype.ext con)) hck.le
    /-
      case eq_top_or_exists_le_coatom.mk.inr.h.refine_2.refl
      α : Type u_2
      inst✝ : CompleteLattice α
      k : α
      h✝ : CompleteLattice.IsCompactElement k
      b : α
      hbk : Membership.mem (Set.Iic k) b
      H : Ne b k
      c : ↑(Set.Iic k)
      c₀ : Not (Eq c Top.top)
      ba : LE.le b ↑c
      h : Maximal (fun x => Membership.mem (Set.Iio k) x) ↑c
      hck : LT.lt ⟨↑c, ⋯⟩ c
      ⊢ False
    -/
    exact lt_irrefl _ hck
    /-
      🎉 no goals
    -/
    /-
      case eq_top_or_exists_le_coatom.mk.inr.h.refine_1
      α : Type u_2
      inst✝ : CompleteLattice α
      k : α
      h : CompleteLattice.IsCompactElement k
      b : α
      hbk : Membership.mem (Set.Iic k) b
      H : Ne b k
      ⊢ ∀ (c : Set α), HasSubset.Subset c (Set.Iio k) → IsChain (fun x1 x2 => LE.le  …
    -/
  · intro S SC cC I _
    /-
      case eq_top_or_exists_le_coatom.mk.inr.h.refine_1
      α : Type u_2
      inst✝ : CompleteLattice α
      k : α
      h : CompleteLattice.IsCompactElement k
      b : α
      hbk : Membership.mem (Set.Iic k) b
      H : Ne b k
      S : Set α
      SC : HasSubset.Subset S (Set.Iio k)
      cC : IsChain (fun x1 x2 => LE.le x1 x2) S
      I : α
      a✝ : Membership.mem S I
      ⊢ Exists fun ub => And (Membership.mem (Set.Iio k) ub) (∀ (z : α), Membership. …
    -/
    by_cases hS : S.Nonempty
      /-
        case pos
        α : Type u_2
        inst✝ : CompleteLattice α
        k : α
        h : CompleteLattice.IsCompactElement k
        b : α
        hbk : Membership.mem (Set.Iic k) b
        H : Ne b k
        S : Set α
        SC : HasSubset.Subset S (Set.Iio k)
        cC : IsChain (fun x1 x2 => LE.le x1 x2) S
        I : α
        a✝ : Membership.mem S I
        hS : S.Nonempty
        ⊢ Exists fun ub => And (Membership.mem (Set.Iio k) ub) (∀ (z : α), Membership. …
      -/
    · refine ⟨sSup S, h.directed_sSup_lt_of_lt hS cC.directedOn SC, ?_⟩
      /-
        case pos
        α : Type u_2
        inst✝ : CompleteLattice α
        k : α
        h : CompleteLattice.IsCompactElement k
        b : α
        hbk : Membership.mem (Set.Iic k) b
        H : Ne b k
        S : Set α
        SC : HasSubset.Subset S (Set.Iio k)
        cC : IsChain (fun x1 x2 => LE.le x1 x2) S
        I : α
        a✝ : Membership.mem S I
        hS : S.Nonempty
        ⊢ ∀ (z : α), Membership.mem S z → LE.le z (SupSet.sSup S)
      -/
      intro; apply le_sSup
             /-
               🎉 no goals
             -/
    exact
      ⟨b, lt_of_le_of_ne hbk H, by
        simp only [Set.not_nonempty_iff_eq_empty.mp hS, Set.mem_empty_iff_false, forall_const,
          forall_prop_of_false, not_false_iff]⟩


theorem coatomic_of_top_compact (h : IsCompactElement (⊤ : α)) : IsCoatomic α :=
  (@OrderIso.IicTop α _ _).isCoatomic_iff.mp (Iic_coatomic_of_compact_element h)


instance (priority := 100) isAtomic_of_complementedLattice [ComplementedLattice α] : IsAtomic α :=
  ⟨fun b => by
    /-
      ι : Sort u_1
      α : Type u_2
      inst✝³ : CompleteLattice α
      f : ι → α
      inst✝² : IsModularLattice α
      inst✝¹ : IsCompactlyGenerated α
      inst✝ : ComplementedLattice α
      b : α
      ⊢ Or (Eq b Bot.bot) (Exists fun a => And (IsAtom a) (LE.le a b))
    -/
    by_cases h : { c : α | CompleteLattice.IsCompactElement c ∧ c ≤ b } ⊆ {⊥}
      /-
        case pos
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElement c)  …
        ⊢ Or (Eq b Bot.bot) (Exists fun a => And (IsAtom a) (LE.le a b))
      -/
    · left
      /-
        case pos.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElement c)  …
        ⊢ Eq b Bot.bot
      -/
      rw [← sSup_compact_le_eq b, sSup_eq_bot]
      /-
        case pos.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElement c)  …
        ⊢ ∀ (a : α), Membership.mem (setOf fun c => And (CompleteLattice.IsCompactElem …
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        ⊢ Or (Eq b Bot.bot) (Exists fun a => And (IsAtom a) (LE.le a b))
      -/
    · rcases Set.not_subset.1 h with ⟨c, ⟨hc, hcb⟩, hcbot⟩
      /-
        case neg.intro.intro.intro
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        ⊢ Or (Eq b Bot.bot) (Exists fun a => And (IsAtom a) (LE.le a b))
      -/
      right
      /-
        case neg.intro.intro.intro.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      have hc' := CompleteLattice.Iic_coatomic_of_compact_element hc
      /-
        case neg.intro.intro.intro.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        hc' : IsCoatomic ↑(Set.Iic c)
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      rw [← isAtomic_iff_isCoatomic] at hc'
      /-
        case neg.intro.intro.intro.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        hc' : IsAtomic ↑(Set.Iic c)
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      haveI := hc'
      /-
        case neg.intro.intro.intro.h
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        hc' this : IsAtomic ↑(Set.Iic c)
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      obtain con | ⟨a, ha, hac⟩ := eq_bot_or_exists_atom_le (⟨c, le_refl c⟩ : Set.Iic c)
        /-
          case neg.intro.intro.intro.h.inl
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
          c : α
          hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
          hc : CompleteLattice.IsCompactElement c
          hcb : LE.le c b
          hc' this : IsAtomic ↑(Set.Iic c)
          con : Eq ⟨c, ⋯⟩ Bot.bot
          ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
        -/
      · exfalso
        /-
          case neg.intro.intro.intro.h.inl
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
          c : α
          hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
          hc : CompleteLattice.IsCompactElement c
          hcb : LE.le c b
          hc' this : IsAtomic ↑(Set.Iic c)
          con : Eq ⟨c, ⋯⟩ Bot.bot
          ⊢ False
        -/
        apply hcbot
        /-
          case neg.intro.intro.intro.h.inl
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
          c : α
          hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
          hc : CompleteLattice.IsCompactElement c
          hcb : LE.le c b
          hc' this : IsAtomic ↑(Set.Iic c)
          con : Eq ⟨c, ⋯⟩ Bot.bot
          ⊢ Membership.mem (Singleton.singleton Bot.bot) c
        -/
        simp only [Subtype.ext_iff, Set.Iic.coe_bot, Subtype.coe_mk] at con
        /-
          case neg.intro.intro.intro.h.inl
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
          c : α
          hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
          hc : CompleteLattice.IsCompactElement c
          hcb : LE.le c b
          hc' this : IsAtomic ↑(Set.Iic c)
          con : Eq c Bot.bot
          ⊢ Membership.mem (Singleton.singleton Bot.bot) c
        -/
        exact con
        /-
          🎉 no goals
        -/
      /-
        case neg.intro.intro.intro.h.inr.intro.intro
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        hc' this : IsAtomic ↑(Set.Iic c)
        a : Subtype fun x => Membership.mem (Set.Iic c) x
        ha : IsAtom a
        hac : LE.le a ⟨c, ⋯⟩
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      rw [← Subtype.coe_le_coe, Subtype.coe_mk] at hac
      /-
        case neg.intro.intro.intro.h.inr.intro.intro
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        h : Not (HasSubset.Subset (setOf fun c => And (CompleteLattice.IsCompactElemen …
        c : α
        hcbot : Not (Membership.mem (Singleton.singleton Bot.bot) c)
        hc : CompleteLattice.IsCompactElement c
        hcb : LE.le c b
        hc' this : IsAtomic ↑(Set.Iic c)
        a : Subtype fun x => Membership.mem (Set.Iic c) x
        ha : IsAtom a
        hac : LE.le ↑a ↑⟨c, ⋯⟩
        ⊢ Exists fun a => And (IsAtom a) (LE.le a b)
      -/
      exact ⟨a, ha.of_isAtom_coe_Iic, hac.trans hcb⟩⟩
      /-
        🎉 no goals
      -/


/-- See [Lemma 5.1][calugareanu]. -/
instance (priority := 100) isAtomistic_of_complementedLattice [ComplementedLattice α] :
    IsAtomistic α :=
  ⟨fun b =>
    ⟨{ a | IsAtom a ∧ a ≤ b }, by
      /-
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        ⊢ Eq b (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)))
      -/
      symm
      /-
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        ⊢ Eq (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
      -/
      have hle : sSup { a : α | IsAtom a ∧ a ≤ b } ≤ b := sSup_le fun _ => And.right
      /-
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        ⊢ Eq (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
      -/
      apply (lt_or_eq_of_le hle).resolve_left _
      /-
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        ⊢ Not (LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b)
      -/
      intro con
      /-
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        ⊢ False
      -/
      obtain ⟨c, hc⟩ := exists_isCompl (⟨sSup { a : α | IsAtom a ∧ a ≤ b }, hle⟩ : Set.Iic b)
      /-
        case intro
        ι : Sort u_1
        α : Type u_2
        inst✝³ : CompleteLattice α
        f : ι → α
        inst✝² : IsModularLattice α
        inst✝¹ : IsCompactlyGenerated α
        inst✝ : ComplementedLattice α
        b : α
        hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
        c : Subtype fun x => Membership.mem (Set.Iic b) x
        hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
        ⊢ False
      -/
      obtain rfl | ⟨a, ha, hac⟩ := eq_bot_or_exists_atom_le c
        /-
          case intro.inl
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ Bo …
          ⊢ False
        -/
      · exact ne_of_lt con (Subtype.ext_iff.1 (eq_top_of_isCompl_bot hc))
        /-
          🎉 no goals
        -/
        /-
          case intro.inr.intro.intro
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          c : Subtype fun x => Membership.mem (Set.Iic b) x
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
          a : Subtype fun x => Membership.mem (Set.Iic b) x
          ha : IsAtom a
          hac : LE.le a c
          ⊢ False
        -/
      · apply ha.1
        /-
          case intro.inr.intro.intro
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          c : Subtype fun x => Membership.mem (Set.Iic b) x
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
          a : Subtype fun x => Membership.mem (Set.Iic b) x
          ha : IsAtom a
          hac : LE.le a c
          ⊢ Eq a Bot.bot
        -/
        rw [eq_bot_iff]
        /-
          case intro.inr.intro.intro
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          c : Subtype fun x => Membership.mem (Set.Iic b) x
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
          a : Subtype fun x => Membership.mem (Set.Iic b) x
          ha : IsAtom a
          hac : LE.le a c
          ⊢ LE.le a Bot.bot
        -/
        apply le_trans (le_inf _ hac) hc.disjoint.le_bot
        /-
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          c : Subtype fun x => Membership.mem (Set.Iic b) x
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
          a : Subtype fun x => Membership.mem (Set.Iic b) x
          ha : IsAtom a
          hac : LE.le a c
          ⊢ LE.le a ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩
        -/
        rw [← Subtype.coe_le_coe, Subtype.coe_mk]
        /-
          ι : Sort u_1
          α : Type u_2
          inst✝³ : CompleteLattice α
          f : ι → α
          inst✝² : IsModularLattice α
          inst✝¹ : IsCompactlyGenerated α
          inst✝ : ComplementedLattice α
          b : α
          hle : LE.le (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          con : LT.lt (SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b))) b
          c : Subtype fun x => Membership.mem (Set.Iic b) x
          hc : IsCompl ⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩ c
          a : Subtype fun x => Membership.mem (Set.Iic b) x
          ha : IsAtom a
          hac : LE.le a c
          ⊢ LE.le ↑a ↑⟨SupSet.sSup (setOf fun a => And (IsAtom a) (LE.le a b)), hle⟩
        -/
        exact le_sSup ⟨ha.of_isAtom_coe_Iic, a.2⟩, fun _ => And.left⟩⟩
        /-
          🎉 no goals
        -/


/-- In an atomic lattice, every element `b` has a complement of the form `sSup s`, where each
element of `s` is an atom. See also `complementedLattice_of_sSup_atoms_eq_top`. -/
theorem exists_sSupIndep_isCompl_sSup_atoms (h : sSup { a : α | IsAtom a } = ⊤) (b : α) :
    ∃ s : Set α, sSupIndep s ∧
    IsCompl b (sSup s) ∧ ∀ ⦃a⦄, a ∈ s → IsAtom a := by
  -- porting note(https://github.com/leanprover-community/mathlib4/issues/5732):
  -- `obtain` chokes on the placeholder.
  have zorn := zorn_subset
    (S := {s : Set α | sSupIndep s ∧ Disjoint b (sSup s) ∧ ∀ a ∈ s, IsAtom a})
    fun c hc1 hc2 =>
      ⟨⋃₀ c,
        ⟨iSupIndep_sUnion_of_directed hc2.directedOn fun s hs => (hc1 hs).1, ?_,
          fun a ⟨s, sc, as⟩ => (hc1 sc).2.2 a as⟩,
        fun _ => Set.subset_sUnion_of_mem⟩
  /-
    case refine_2
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    zorn : Exists fun m => Maximal (fun x => Membership.mem (setOf fun s => And (s …
    ⊢ Exists fun s => And (sSupIndep s) (And (IsCompl b (SupSet.sSup s)) (∀ ⦃a : α …
  -/
  swap
    /-
      case refine_1
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      c : Set (Set α)
      hc1 : HasSubset.Subset c (setOf fun s => And (sSupIndep s) (And (Disjoint b (S …
      hc2 : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      ⊢ Disjoint b (SupSet.sSup c.sUnion)
    -/
  · rw [sSup_sUnion, ← sSup_image, DirectedOn.disjoint_sSup_right]
      /-
        case refine_1
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        c : Set (Set α)
        hc1 : HasSubset.Subset c (setOf fun s => And (sSupIndep s) (And (Disjoint b (S …
        hc2 : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        ⊢ ∀ ⦃b_1 : α⦄, Membership.mem (Set.image SupSet.sSup c) b_1 → Disjoint b b_1
      -/
    · rintro _ ⟨s, hs, rfl⟩
      /-
        case refine_1.intro.intro
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        c : Set (Set α)
        hc1 : HasSubset.Subset c (setOf fun s => And (sSupIndep s) (And (Disjoint b (S …
        hc2 : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        s : Set α
        hs : Membership.mem c s
        ⊢ Disjoint b (SupSet.sSup s)
      -/
      exact (hc1 hs).2.1
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        c : Set (Set α)
        hc1 : HasSubset.Subset c (setOf fun s => And (sSupIndep s) (And (Disjoint b (S …
        hc2 : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        ⊢ DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image SupSet.sSup c)
      -/
    · rw [directedOn_image]
      /-
        case refine_1
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        c : Set (Set α)
        hc1 : HasSubset.Subset c (setOf fun s => And (sSupIndep s) (And (Disjoint b (S …
        hc2 : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        ⊢ DirectedOn (Order.Preimage SupSet.sSup fun x1 x2 => LE.le x1 x2) c
      -/
      exact hc2.directedOn.mono @fun s t => sSup_le_sSup
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    zorn : Exists fun m => Maximal (fun x => Membership.mem (setOf fun s => And (s …
    ⊢ Exists fun s => And (sSupIndep s) (And (IsCompl b (SupSet.sSup s)) (∀ ⦃a : α …
  -/
  simp_rw [maximal_subset_iff] at zorn
  /-
    case refine_2
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    zorn : Exists fun m => And (Membership.mem (setOf fun s => And (sSupIndep s) ( …
    ⊢ Exists fun s => And (sSupIndep s) (And (IsCompl b (SupSet.sSup s)) (∀ ⦃a : α …
  -/
  obtain ⟨s, ⟨s_ind, b_inf_Sup_s, s_atoms⟩, s_max⟩ := zorn
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    ⊢ Exists fun s => And (sSupIndep s) (And (IsCompl b (SupSet.sSup s)) (∀ ⦃a : α …
  -/
  refine ⟨s, s_ind, ⟨b_inf_Sup_s, ?_⟩, s_atoms⟩
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    ⊢ Codisjoint b (SupSet.sSup s)
  -/
  rw [codisjoint_iff_le_sup, ← h, sSup_le_iff]
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    ⊢ ∀ (b_1 : α), Membership.mem (setOf fun a => IsAtom a) b_1 → LE.le b_1 (Max.m …
  -/
  intro a ha
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    ⊢ LE.le a (Max.max b (SupSet.sSup s))
  -/
  rw [← inf_eq_left]
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    ⊢ Eq (Min.min a (Max.max b (SupSet.sSup s))) a
  -/
  refine (ha.le_iff.mp inf_le_left).resolve_left fun con => ha.1 ?_
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    con : Eq (Min.min a (Max.max b (SupSet.sSup s))) Bot.bot
    ⊢ Eq a Bot.bot
  -/
  rw [← con, eq_comm, inf_eq_left]
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    con : Eq (Min.min a (Max.max b (SupSet.sSup s))) Bot.bot
    ⊢ LE.le a (Max.max b (SupSet.sSup s))
  -/
  refine (le_sSup ?_).trans le_sup_right
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    con : Eq (Min.min a (Max.max b (SupSet.sSup s))) Bot.bot
    ⊢ Membership.mem s a
  -/
  rw [← disjoint_iff] at con
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    con : Disjoint a (Max.max b (SupSet.sSup s))
    ⊢ Membership.mem s a
  -/
  have a_dis_Sup_s : Disjoint a (sSup s) := con.mono_right le_sup_right
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
    b : α
    s : Set α
    s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
    s_ind : sSupIndep s
    b_inf_Sup_s : Disjoint b (SupSet.sSup s)
    s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
    a : α
    ha : Membership.mem (setOf fun a => IsAtom a) a
    con : Disjoint a (Max.max b (SupSet.sSup s))
    a_dis_Sup_s : Disjoint a (SupSet.sSup s)
    ⊢ Membership.mem s a
  -/
  rw [s_max ⟨fun x hx => ?_, ?_, fun x hx => ?_⟩ Set.subset_union_left]
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      ⊢ Membership.mem (Union.union s ?m.79075) a
    -/
  · exact Set.mem_union_right _ (Set.mem_singleton _)
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      ⊢ Disjoint b (SupSet.sSup (Union.union s (Singleton.singleton a)))
    -/
  · rw [sSup_union, sSup_singleton]
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      ⊢ Disjoint b (Max.max (SupSet.sSup s) a)
    -/
    exact b_inf_Sup_s.disjoint_sup_right_of_disjoint_sup_left con.symm
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      x : α
      hx : Membership.mem (Union.union s (Singleton.singleton a)) x
      ⊢ Disjoint x (SupSet.sSup (SDiff.sdiff (Union.union s (Singleton.singleton a)) …
    -/
  · rw [Set.mem_union, Set.mem_singleton_iff] at hx
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      x : α
      hx : Or (Membership.mem s x) (Eq x a)
      ⊢ Disjoint x (SupSet.sSup (SDiff.sdiff (Union.union s (Singleton.singleton a)) …
    -/
    obtain rfl | xa := eq_or_ne x a
      /-
        case inl
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        x : α
        ha : Membership.mem (setOf fun a => IsAtom a) x
        con : Disjoint x (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint x (SupSet.sSup s)
        hx : Or (Membership.mem s x) (Eq x x)
        ⊢ Disjoint x (SupSet.sSup (SDiff.sdiff (Union.union s (Singleton.singleton x)) …
      -/
    · simp only [Set.mem_singleton, Set.insert_diff_of_mem, Set.union_singleton]
      /-
        case inl
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        x : α
        ha : Membership.mem (setOf fun a => IsAtom a) x
        con : Disjoint x (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint x (SupSet.sSup s)
        hx : Or (Membership.mem s x) (Eq x x)
        ⊢ Disjoint x (SupSet.sSup (SDiff.sdiff s (Singleton.singleton x)))
      -/
      exact con.mono_right ((sSup_le_sSup Set.diff_subset).trans le_sup_right)
      /-
        🎉 no goals
      -/
    · have h : (s ∪ {a}) \ {x} = s \ {x} ∪ {a} := by
        simp only [Set.union_singleton]
        rw [Set.insert_diff_of_not_mem]
        rw [Set.mem_singleton_iff]
        exact Ne.symm xa
      /-
        case inr
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h✝ : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        a : α
        ha : Membership.mem (setOf fun a => IsAtom a) a
        con : Disjoint a (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint a (SupSet.sSup s)
        x : α
        hx : Or (Membership.mem s x) (Eq x a)
        xa : Ne x a
        h : Eq (SDiff.sdiff (Union.union s (Singleton.singleton a)) (Singleton.singlet …
        ⊢ Disjoint x (SupSet.sSup (SDiff.sdiff (Union.union s (Singleton.singleton a)) …
      -/
      rw [h, sSup_union, sSup_singleton]
      apply
        (s_ind (hx.resolve_right xa)).disjoint_sup_right_of_disjoint_sup_left
          (a_dis_Sup_s.mono_right _).symm
      /-
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h✝ : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        a : α
        ha : Membership.mem (setOf fun a => IsAtom a) a
        con : Disjoint a (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint a (SupSet.sSup s)
        x : α
        hx : Or (Membership.mem s x) (Eq x a)
        xa : Ne x a
        h : Eq (SDiff.sdiff (Union.union s (Singleton.singleton a)) (Singleton.singlet …
        ⊢ LE.le (Max.max x (SupSet.sSup (SDiff.sdiff s (Singleton.singleton x)))) (Sup …
      -/
      rw [← sSup_insert, Set.insert_diff_singleton, Set.insert_eq_of_mem (hx.resolve_right xa)]
      /-
        🎉 no goals
      -/
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      x : α
      hx : Membership.mem (Union.union s (Singleton.singleton a)) x
      ⊢ IsAtom x
    -/
  · rw [Set.mem_union, Set.mem_singleton_iff] at hx
    /-
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
      b : α
      s : Set α
      s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
      s_ind : sSupIndep s
      b_inf_Sup_s : Disjoint b (SupSet.sSup s)
      s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
      a : α
      ha : Membership.mem (setOf fun a => IsAtom a) a
      con : Disjoint a (Max.max b (SupSet.sSup s))
      a_dis_Sup_s : Disjoint a (SupSet.sSup s)
      x : α
      hx : Or (Membership.mem s x) (Eq x a)
      ⊢ IsAtom x
    -/
    obtain hx | rfl := hx
      /-
        case inl
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        a : α
        ha : Membership.mem (setOf fun a => IsAtom a) a
        con : Disjoint a (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint a (SupSet.sSup s)
        x : α
        hx : Membership.mem s x
        ⊢ IsAtom x
      -/
    · exact s_atoms x hx
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_2
        inst✝² : CompleteLattice α
        inst✝¹ : IsModularLattice α
        inst✝ : IsCompactlyGenerated α
        h : Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
        b : α
        s : Set α
        s_max : ∀ ⦃t : Set α⦄, Membership.mem (setOf fun s => And (sSupIndep s) (And ( …
        s_ind : sSupIndep s
        b_inf_Sup_s : Disjoint b (SupSet.sSup s)
        s_atoms : ∀ (a : α), Membership.mem s a → IsAtom a
        x : α
        ha : Membership.mem (setOf fun a => IsAtom a) x
        con : Disjoint x (Max.max b (SupSet.sSup s))
        a_dis_Sup_s : Disjoint x (SupSet.sSup s)
        ⊢ IsAtom x
      -/
    · exact ha
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-24")]
alias exists_setIndependent_isCompl_sSup_atoms := exists_sSupIndep_isCompl_sSup_atoms


theorem exists_sSupIndep_of_sSup_atoms_eq_top (h : sSup { a : α | IsAtom a } = ⊤) :
    ∃ s : Set α, sSupIndep s ∧ sSup s = ⊤ ∧ ∀ ⦃a⦄, a ∈ s → IsAtom a :=
  let ⟨s, s_ind, s_top, s_atoms⟩ := exists_sSupIndep_isCompl_sSup_atoms h ⊥
  ⟨s, s_ind, eq_top_of_isCompl_bot s_top.symm, s_atoms⟩


@[deprecated (since := "2024-11-24")]
alias exists_setIndependent_of_sSup_atoms_eq_top := exists_sSupIndep_of_sSup_atoms_eq_top


/-- See [Theorem 6.6][calugareanu]. -/
theorem complementedLattice_of_sSup_atoms_eq_top (h : sSup { a : α | IsAtom a } = ⊤) :
    ComplementedLattice α :=
  ⟨fun b =>
    let ⟨s, _, s_top, _⟩ := exists_sSupIndep_isCompl_sSup_atoms h b
    ⟨sSup s, s_top⟩⟩


/-- See [Theorem 6.6][calugareanu]. -/
theorem complementedLattice_of_isAtomistic [IsAtomistic α] : ComplementedLattice α :=
  complementedLattice_of_sSup_atoms_eq_top sSup_atoms_eq_top


theorem complementedLattice_iff_isAtomistic : ComplementedLattice α ↔ IsAtomistic α := by
  /-
    α : Type u_2
    inst✝² : CompleteLattice α
    inst✝¹ : IsModularLattice α
    inst✝ : IsCompactlyGenerated α
    ⊢ Iff (ComplementedLattice α) (IsAtomistic α)
  -/
  constructor <;> intros
    /-
      case mp
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      a✝ : ComplementedLattice α
      ⊢ IsAtomistic α
    -/
  · exact isAtomistic_of_complementedLattice
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝² : CompleteLattice α
      inst✝¹ : IsModularLattice α
      inst✝ : IsCompactlyGenerated α
      a✝ : IsAtomistic α
      ⊢ ComplementedLattice α
    -/
  · exact complementedLattice_of_isAtomistic
    /-
      🎉 no goals
    -/


