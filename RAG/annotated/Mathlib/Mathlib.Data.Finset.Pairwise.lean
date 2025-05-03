instance [DecidableEq α] {r : α → α → Prop} [DecidableRel r] {s : Finset α} :
    Decidable ((s : Set α).Pairwise r) :=
  decidable_of_iff' (∀ a ∈ s, ∀ b ∈ s, a ≠ b → r a b) Iff.rfl


theorem Finset.pairwiseDisjoint_range_singleton :
    (Set.range (singleton : α → Finset α)).PairwiseDisjoint id := by
  /-
    α : Type u_1
    ⊢ (Set.range Singleton.singleton).PairwiseDisjoint id
  -/
  rintro _ ⟨a, rfl⟩ _ ⟨b, rfl⟩ h
  /-
    case intro.intro
    α : Type u_1
    a b : α
    h : Ne (Singleton.singleton a) (Singleton.singleton b)
    ⊢ Function.onFun Disjoint id (Singleton.singleton a) (Singleton.singleton b)
  -/
  exact disjoint_singleton.2 (ne_of_apply_ne _ h)
  /-
    🎉 no goals
  -/


theorem PairwiseDisjoint.elim_finset {s : Set ι} {f : ι → Finset α} (hs : s.PairwiseDisjoint f)
    {i j : ι} (hi : i ∈ s) (hj : j ∈ s) (a : α) (hai : a ∈ f i) (haj : a ∈ f j) : i = j :=
  hs.elim hi hj (Finset.not_disjoint_iff.2 ⟨a, hai, haj⟩)


theorem PairwiseDisjoint.image_finset_of_le [DecidableEq ι] {s : Finset ι} {f : ι → α}
    (hs : (s : Set ι).PairwiseDisjoint f) {g : ι → ι} (hf : ∀ a, f (g a) ≤ f a) :
    (s.image g : Set ι).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → α
    hs : (↑s).PairwiseDisjoint f
    g : ι → ι
    hf : ∀ (a : ι), LE.le (f (g a)) (f a)
    ⊢ (↑(Finset.image g s)).PairwiseDisjoint f
  -/
  rw [coe_image]
  /-
    α : Type u_1
    ι : Type u_2
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → α
    hs : (↑s).PairwiseDisjoint f
    g : ι → ι
    hf : ∀ (a : ι), LE.le (f (g a)) (f a)
    ⊢ (Set.image g ↑s).PairwiseDisjoint f
  -/
  exact hs.image_of_le hf
  /-
    🎉 no goals
  -/


theorem PairwiseDisjoint.attach (hs : (s : Set ι).PairwiseDisjoint f) :
    (s.attach : Set { x // x ∈ s }).PairwiseDisjoint (f ∘ Subtype.val) := fun i _ j _ hij =>
  hs i.2 j.2 <| mt Subtype.ext_val hij


/-- Bind operation for `Set.PairwiseDisjoint`. In a complete lattice, you can use
`Set.PairwiseDisjoint.biUnion`. -/
theorem PairwiseDisjoint.biUnion_finset {s : Set ι'} {g : ι' → Finset ι} {f : ι → α}
    (hs : s.PairwiseDisjoint fun i' : ι' => (g i').sup f)
    (hg : ∀ i ∈ s, (g i : Set ι).PairwiseDisjoint f) : (⋃ i ∈ s, ↑(g i)).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Set ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => (g i').sup f
    hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => ↑(g i)).PairwiseDisjoint f
  -/
  rintro a ha b hb hab
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Set ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => (g i').sup f
    hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
    a : ι
    ha : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => ↑(g i)) a
    b : ι
    hb : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => ↑(g i)) b
    hab : Ne a b
    ⊢ Function.onFun Disjoint f a b
  -/
  simp_rw [Set.mem_iUnion] at ha hb
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Set ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => (g i').sup f
    hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    ha : Exists fun i => Exists fun i_1 => Membership.mem (↑(g i)) a
    hb : Exists fun i => Exists fun i_1 => Membership.mem (↑(g i)) b
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain ⟨c, hc, ha⟩ := ha
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Set ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => (g i').sup f
    hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    hb : Exists fun i => Exists fun i_1 => Membership.mem (↑(g i)) b
    c : ι'
    hc : Membership.mem s c
    ha : Membership.mem (↑(g c)) a
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain ⟨d, hd, hb⟩ := hb
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Set ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => (g i').sup f
    hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    c : ι'
    hc : Membership.mem s c
    ha : Membership.mem (↑(g c)) a
    d : ι'
    hd : Membership.mem s d
    hb : Membership.mem (↑(g d)) b
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain hcd | hcd := eq_or_ne (g c) (g d)
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      s : Set ι'
      g : ι' → Finset ι
      f : ι → α
      hs : s.PairwiseDisjoint fun i' => (g i').sup f
      hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
      a b : ι
      hab : Ne a b
      c : ι'
      hc : Membership.mem s c
      ha : Membership.mem (↑(g c)) a
      d : ι'
      hd : Membership.mem s d
      hb : Membership.mem (↑(g d)) b
      hcd : Eq (g c) (g d)
      ⊢ Function.onFun Disjoint f a b
    -/
  · exact hg d hd (by rwa [hcd] at ha) hb hab
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      s : Set ι'
      g : ι' → Finset ι
      f : ι → α
      hs : s.PairwiseDisjoint fun i' => (g i').sup f
      hg : ∀ (i : ι'), Membership.mem s i → (↑(g i)).PairwiseDisjoint f
      a b : ι
      hab : Ne a b
      c : ι'
      hc : Membership.mem s c
      ha : Membership.mem (↑(g c)) a
      d : ι'
      hd : Membership.mem s d
      hb : Membership.mem (↑(g d)) b
      hcd : Ne (g c) (g d)
      ⊢ Function.onFun Disjoint f a b
    -/
  · exact (hs hc hd (ne_of_apply_ne _ hcd)).mono (Finset.le_sup ha) (Finset.le_sup hb)
    /-
      🎉 no goals
    -/


theorem pairwise_of_coe_toFinset_pairwise (hl : (l.toFinset : Set α).Pairwise r) (hn : l.Nodup) :
    l.Pairwise r := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    r : α → α → Prop
    l : List α
    hl : (↑l.toFinset).Pairwise r
    hn : l.Nodup
    ⊢ List.Pairwise r l
  -/
  rw [coe_toFinset] at hl
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    r : α → α → Prop
    l : List α
    hl : (setOf fun a => Membership.mem l a).Pairwise r
    hn : l.Nodup
    ⊢ List.Pairwise r l
  -/
  exact hn.pairwise_of_set_pairwise hl
  /-
    🎉 no goals
  -/


theorem pairwise_iff_coe_toFinset_pairwise (hn : l.Nodup) (hs : Symmetric r) :
    (l.toFinset : Set α).Pairwise r ↔ l.Pairwise r := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    r : α → α → Prop
    l : List α
    hn : l.Nodup
    hs : Symmetric r
    ⊢ Iff ((↑l.toFinset).Pairwise r) (List.Pairwise r l)
  -/
  letI : IsSymm α r := ⟨hs⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    r : α → α → Prop
    l : List α
    hn : l.Nodup
    hs : Symmetric r
    this : IsSymm α r := { symm := hs }
    ⊢ Iff ((↑l.toFinset).Pairwise r) (List.Pairwise r l)
  -/
  rw [coe_toFinset, hn.pairwise_coe]
  /-
    🎉 no goals
  -/


theorem pairwise_disjoint_of_coe_toFinset_pairwiseDisjoint {α ι} [SemilatticeInf α] [OrderBot α]
    [DecidableEq ι] {l : List ι} {f : ι → α} (hl : (l.toFinset : Set ι).PairwiseDisjoint f)
    (hn : l.Nodup) : l.Pairwise (_root_.Disjoint on f) :=
  pairwise_of_coe_toFinset_pairwise hl hn


theorem pairwiseDisjoint_iff_coe_toFinset_pairwise_disjoint {α ι} [SemilatticeInf α] [OrderBot α]
    [DecidableEq ι] {l : List ι} {f : ι → α} (hn : l.Nodup) :
    (l.toFinset : Set ι).PairwiseDisjoint f ↔ l.Pairwise (_root_.Disjoint on f) :=
  pairwise_iff_coe_toFinset_pairwise hn (symmetric_disjoint.comap f)


