/-- A set family is intersecting if every pair of elements is non-disjoint. -/
def Intersecting (s : Set α) : Prop :=
  ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → ¬Disjoint a b


@[mono]
theorem Intersecting.mono (h : t ⊆ s) (hs : s.Intersecting) : t.Intersecting := fun _a ha _b hb =>
  hs (h ha) (h hb)


theorem Intersecting.not_bot_mem (hs : s.Intersecting) : ⊥ ∉ s := fun h => hs h h disjoint_bot_left


theorem Intersecting.ne_bot (hs : s.Intersecting) (ha : a ∈ s) : a ≠ ⊥ :=
  ne_of_mem_of_not_mem ha hs.not_bot_mem


theorem intersecting_empty : (∅ : Set α).Intersecting := fun _ => False.elim


@[simp]
                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝¹ : SemilatticeInf α
                                                                            inst✝ : OrderBot α
                                                                            a : α
                                                                            ⊢ Iff (Singleton.singleton a).Intersecting (Ne a Bot.bot)
                                                                          -/
theorem intersecting_singleton : ({a} : Set α).Intersecting ↔ a ≠ ⊥ := by simp [Intersecting]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


protected theorem Intersecting.insert (hs : s.Intersecting) (ha : a ≠ ⊥)
    (h : ∀ b ∈ s, ¬Disjoint a b) : (insert a s).Intersecting := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    s : Set α
    a : α
    hs : s.Intersecting
    ha : Ne a Bot.bot
    h : ∀ (b : α), Membership.mem s b → Not (Disjoint a b)
    ⊢ (Insert.insert a s).Intersecting
  -/
  rintro b (rfl | hb) c (rfl | hc)
    /-
      case inl.inl
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      s : Set α
      hs : s.Intersecting
      c : α
      ha : Ne c Bot.bot
      h : ∀ (b : α), Membership.mem s b → Not (Disjoint c b)
      ⊢ Not (Disjoint c c)
    -/
  · rwa [disjoint_self]
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      s : Set α
      hs : s.Intersecting
      b : α
      ha : Ne b Bot.bot
      h : ∀ (b_1 : α), Membership.mem s b_1 → Not (Disjoint b b_1)
      c : α
      hc : Membership.mem s c
      ⊢ Not (Disjoint b c)
    -/
  · exact h _ hc
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      s : Set α
      hs : s.Intersecting
      b : α
      hb : Membership.mem s b
      c : α
      ha : Ne c Bot.bot
      h : ∀ (b : α), Membership.mem s b → Not (Disjoint c b)
      ⊢ Not (Disjoint b c)
    -/
  · exact fun H => h _ hb H.symm
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      s : Set α
      a : α
      hs : s.Intersecting
      ha : Ne a Bot.bot
      h : ∀ (b : α), Membership.mem s b → Not (Disjoint a b)
      b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem s c
      ⊢ Not (Disjoint b c)
    -/
  · exact hs hb hc
    /-
      🎉 no goals
    -/


theorem intersecting_insert :
    (insert a s).Intersecting ↔ s.Intersecting ∧ a ≠ ⊥ ∧ ∀ b ∈ s, ¬Disjoint a b :=
  ⟨fun h =>
    ⟨h.mono <| subset_insert _ _, h.ne_bot <| mem_insert _ _, fun _b hb =>
      h (mem_insert _ _) <| mem_insert_of_mem _ hb⟩,
    fun h => h.1.insert h.2.1 h.2.2⟩


theorem intersecting_iff_pairwise_not_disjoint :
    s.Intersecting ↔ (s.Pairwise fun a b => ¬Disjoint a b) ∧ s ≠ {⊥} := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    s : Set α
    ⊢ Iff s.Intersecting (And (s.Pairwise fun a b => Not (Disjoint a b)) (Ne s (Si …
  -/
  refine ⟨fun h => ⟨fun a ha b hb _ => h ha hb, ?_⟩, fun h a ha b hb hab => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      s : Set α
      h : s.Intersecting
      ⊢ Ne s (Singleton.singleton Bot.bot)
    -/
  · rintro rfl
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : SemilatticeInf α
      inst✝ : OrderBot α
      h : (Singleton.singleton Bot.bot).Intersecting
      ⊢ False
    -/
    exact intersecting_singleton.1 h rfl
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    s : Set α
    h : And (s.Pairwise fun a b => Not (Disjoint a b)) (Ne s (Singleton.singleton  …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Disjoint a b
    ⊢ False
  -/
  have := h.1.eq ha hb (Classical.not_not.2 hab)
  /-
    case refine_2
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    s : Set α
    h : And (s.Pairwise fun a b => Not (Disjoint a b)) (Ne s (Singleton.singleton  …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Disjoint a b
    this : Eq a b
    ⊢ False
  -/
  rw [this, disjoint_self] at hab
  /-
    case refine_2
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    s : Set α
    h : And (s.Pairwise fun a b => Not (Disjoint a b)) (Ne s (Singleton.singleton  …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Eq b Bot.bot
    this : Eq a b
    ⊢ False
  -/
  rw [hab] at hb
  exact
    h.2
      (eq_singleton_iff_unique_mem.2
        ⟨hb, fun c hc => not_ne_iff.1 fun H => h.1 hb hc H.symm disjoint_bot_left⟩)


protected theorem Subsingleton.intersecting (hs : s.Subsingleton) : s.Intersecting ↔ s ≠ {⊥} :=
  intersecting_iff_pairwise_not_disjoint.trans <| and_iff_right <| hs.pairwise _


theorem intersecting_iff_eq_empty_of_subsingleton [Subsingleton α] (s : Set α) :
    s.Intersecting ↔ s = ∅ := by
  refine
    subsingleton_of_subsingleton.intersecting.trans
      ⟨not_imp_comm.2 fun h => subsingleton_of_subsingleton.eq_singleton_of_mem ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝² : SemilatticeInf α
      inst✝¹ : OrderBot α
      inst✝ : Subsingleton α
      s : Set α
      h : Not (Eq s EmptyCollection.emptyCollection)
      ⊢ Membership.mem s Bot.bot
    -/
  · obtain ⟨a, ha⟩ := nonempty_iff_ne_empty.2 h
    /-
      case refine_1.intro
      α : Type u_1
      inst✝² : SemilatticeInf α
      inst✝¹ : OrderBot α
      inst✝ : Subsingleton α
      s : Set α
      h : Not (Eq s EmptyCollection.emptyCollection)
      a : α
      ha : Membership.mem s a
      ⊢ Membership.mem s Bot.bot
    -/
    rwa [Subsingleton.elim ⊥ a]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : SemilatticeInf α
      inst✝¹ : OrderBot α
      inst✝ : Subsingleton α
      s : Set α
      ⊢ Eq s EmptyCollection.emptyCollection → Ne s (Singleton.singleton Bot.bot)
    -/
  · rintro rfl
    /-
      case refine_2
      α : Type u_1
      inst✝² : SemilatticeInf α
      inst✝¹ : OrderBot α
      inst✝ : Subsingleton α
      ⊢ Ne EmptyCollection.emptyCollection (Singleton.singleton Bot.bot)
    -/
    exact (Set.singleton_nonempty _).ne_empty.symm
    /-
      🎉 no goals
    -/


/-- Maximal intersecting families are upper sets. -/
protected theorem Intersecting.isUpperSet (hs : s.Intersecting)
    (h : ∀ t : Set α, t.Intersecting → s ⊆ t → s = t) : IsUpperSet s := by
  classical
    rintro a b hab ha
    rw [h (Insert.insert b s) _ (subset_insert _ _)]
    · exact mem_insert _ _
    exact
      hs.insert (mt (eq_bot_mono hab) <| hs.ne_bot ha) fun c hc hbc => hs ha hc <| hbc.mono_left hab


/-- Maximal intersecting families are upper sets. Finset version. -/
theorem Intersecting.isUpperSet' {s : Finset α} (hs : (s : Set α).Intersecting)
    (h : ∀ t : Finset α, (t : Set α).Intersecting → s ⊆ t → s = t) : IsUpperSet (s : Set α) := by
  classical
    rintro a b hab ha
    rw [h (Insert.insert b s) _ (Finset.subset_insert _ _)]
    · exact mem_insert_self _ _
    rw [coe_insert]
    exact
      hs.insert (mt (eq_bot_mono hab) <| hs.ne_bot ha) fun c hc hbc => hs ha hc <| hbc.mono_left hab


theorem Intersecting.exists_mem_set {𝒜 : Set (Set α)} (h𝒜 : 𝒜.Intersecting) {s t : Set α}
    (hs : s ∈ 𝒜) (ht : t ∈ 𝒜) : ∃ a, a ∈ s ∧ a ∈ t :=
  not_disjoint_iff.1 <| h𝒜 hs ht


theorem Intersecting.exists_mem_finset [DecidableEq α] {𝒜 : Set (Finset α)} (h𝒜 : 𝒜.Intersecting)
    {s t : Finset α} (hs : s ∈ 𝒜) (ht : t ∈ 𝒜) : ∃ a, a ∈ s ∧ a ∈ t :=
  not_disjoint_iff.1 <| disjoint_coe.not.2 <| h𝒜 hs ht


theorem Intersecting.not_compl_mem {s : Set α} (hs : s.Intersecting) {a : α} (ha : a ∈ s) :
    aᶜ ∉ s := fun h => hs ha h disjoint_compl_right


theorem Intersecting.not_mem {s : Set α} (hs : s.Intersecting) {a : α} (ha : aᶜ ∈ s) : a ∉ s :=
  fun h => hs ha h disjoint_compl_left


theorem Intersecting.disjoint_map_compl {s : Finset α} (hs : (s : Set α).Intersecting) :
    Disjoint s (s.map ⟨compl, compl_injective⟩) := by
  /-
    α : Type u_1
    inst✝ : BooleanAlgebra α
    s : Finset α
    hs : (↑s).Intersecting
    ⊢ Disjoint s (Finset.map { toFun := HasCompl.compl, inj' := ⋯ } s)
  -/
  rw [Finset.disjoint_left]
  /-
    α : Type u_1
    inst✝ : BooleanAlgebra α
    s : Finset α
    hs : (↑s).Intersecting
    ⊢ ∀ ⦃a : α⦄, Membership.mem s a → Not (Membership.mem (Finset.map { toFun := H …
  -/
  rintro x hx hxc
  /-
    α : Type u_1
    inst✝ : BooleanAlgebra α
    s : Finset α
    hs : (↑s).Intersecting
    x : α
    hx : Membership.mem s x
    hxc : Membership.mem (Finset.map { toFun := HasCompl.compl, inj' := ⋯ } s) x
    ⊢ False
  -/
  obtain ⟨x, hx', rfl⟩ := mem_map.mp hxc
  /-
    case intro.intro
    α : Type u_1
    inst✝ : BooleanAlgebra α
    s : Finset α
    hs : (↑s).Intersecting
    x : α
    hx' : Membership.mem s x
    hx : Membership.mem s ({ toFun := HasCompl.compl, inj' := ⋯ } x)
    hxc : Membership.mem (Finset.map { toFun := HasCompl.compl, inj' := ⋯ } s) ({  …
    ⊢ False
  -/
  exact hs.not_compl_mem hx' hx
  /-
    🎉 no goals
  -/


theorem Intersecting.card_le [Fintype α] {s : Finset α} (hs : (s : Set α).Intersecting) :
    2 * #s ≤ Fintype.card α := by
  classical
    refine (s.disjUnion _ hs.disjoint_map_compl).card_le_univ.trans_eq' ?_
    rw [Nat.two_mul, card_disjUnion, card_map]


theorem Intersecting.is_max_iff_card_eq (hs : (s : Set α).Intersecting) :
    (∀ t : Finset α, (t : Set α).Intersecting → s ⊆ t → s = t) ↔ 2 * #s = Fintype.card α := by
  classical
    refine ⟨fun h ↦ ?_, fun h t ht hst ↦ Finset.eq_of_subset_of_card_le hst <|
      Nat.le_of_mul_le_mul_left (ht.card_le.trans_eq h.symm) Nat.two_pos⟩
    suffices s.disjUnion (s.map ⟨compl, compl_injective⟩) hs.disjoint_map_compl = Finset.univ by
      rw [Fintype.card, ← this, Nat.two_mul, card_disjUnion, card_map]
    rw [← coe_eq_univ, disjUnion_eq_union, coe_union, coe_map, Function.Embedding.coeFn_mk,
      image_eq_preimage_of_inverse compl_compl compl_compl]
    refine eq_univ_of_forall fun a => ?_
    simp_rw [mem_union, mem_preimage]
    by_contra! ha
    refine s.ne_insert_of_not_mem _ ha.1 (h _ ?_ <| s.subset_insert _)
    rw [coe_insert]
    refine hs.insert ?_ fun b hb hab => ha.2 <| (hs.isUpperSet' h) hab.le_compl_left hb
    rintro rfl
    have := h {⊤} (by rw [coe_singleton]; exact intersecting_singleton.2 top_ne_bot)
    rw [compl_bot] at ha
    rw [coe_eq_empty.1 ((hs.isUpperSet' h).not_top_mem.1 ha.2)] at this
    exact Finset.singleton_ne_empty _ (this <| Finset.empty_subset _).symm


theorem Intersecting.exists_card_eq (hs : (s : Set α).Intersecting) :
    ∃ t, s ⊆ t ∧ 2 * #t = Fintype.card α ∧ (t : Set α).Intersecting := by
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s : Finset α
    hs : (↑s).Intersecting
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  have := hs.card_le
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s : Finset α
    hs : (↑s).Intersecting
    this : LE.le (HMul.hMul 2 s.card) (Fintype.card α)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  rw [mul_comm, ← Nat.le_div_iff_mul_le Nat.two_pos] at this
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s : Finset α
    hs : (↑s).Intersecting
    this : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  revert hs
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s : Finset α
    this : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    ⊢ (↑s).Intersecting → Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMu …
  -/
  refine s.strongDownwardInductionOn ?_ this
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s : Finset α
    this : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    ⊢ ∀ (t₁ : Finset α), (∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.car …
  -/
  rintro s ih _hcard hs
  /-
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  by_cases h : ∀ t : Finset α, (t : Set α).Intersecting → s ⊆ t → s = t
    /-
      case pos
      α : Type u_1
      inst✝² : BooleanAlgebra α
      inst✝¹ : Nontrivial α
      inst✝ : Fintype α
      s✝ : Finset α
      this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
      s : Finset α
      ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
      _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
      hs : (↑s).Intersecting
      h : ∀ (t : Finset α), (↑t).Intersecting → HasSubset.Subset s t → Eq s t
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
    -/
  · exact ⟨s, Subset.rfl, hs.is_max_iff_card_eq.1 h, hs⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    h : Not (∀ (t : Finset α), (↑t).Intersecting → HasSubset.Subset s t → Eq s t)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  push_neg at h
  /-
    case neg
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    h : Exists fun t => And (↑t).Intersecting (And (HasSubset.Subset s t) (Ne s t))
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  obtain ⟨t, ht, hst⟩ := h
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    t : Finset α
    ht : (↑t).Intersecting
    hst : And (HasSubset.Subset s t) (Ne s t)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (HMul.hMul 2 t.card) (Fi …
  -/
  refine (ih ?_ (_root_.ssubset_iff_subset_ne.2 hst) ht).imp fun u => And.imp_left hst.1.trans
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    t : Finset α
    ht : (↑t).Intersecting
    hst : And (HasSubset.Subset s t) (Ne s t)
    ⊢ LE.le t.card (HDiv.hDiv (Fintype.card α) 2)
  -/
  rw [Nat.le_div_iff_mul_le Nat.two_pos, mul_comm]
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝² : BooleanAlgebra α
    inst✝¹ : Nontrivial α
    inst✝ : Fintype α
    s✝ : Finset α
    this : LE.le s✝.card (HDiv.hDiv (Fintype.card α) 2)
    s : Finset α
    ih : ∀ {t₂ : Finset α}, LE.le t₂.card (HDiv.hDiv (Fintype.card α) 2) → HasSSub …
    _hcard : LE.le s.card (HDiv.hDiv (Fintype.card α) 2)
    hs : (↑s).Intersecting
    t : Finset α
    ht : (↑t).Intersecting
    hst : And (HasSubset.Subset s t) (Ne s t)
    ⊢ LE.le (HMul.hMul 2 t.card) (Fintype.card α)
  -/
  exact ht.card_le
  /-
    🎉 no goals
  -/


