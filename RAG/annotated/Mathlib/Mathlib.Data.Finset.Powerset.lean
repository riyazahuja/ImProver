/-- When `s` is a finset, `s.powerset` is the finset of all subsets of `s` (seen as finsets). -/
def powerset (s : Finset α) : Finset (Finset α) :=
  ⟨(s.1.powerset.pmap Finset.mk) fun _t h => nodup_of_le (mem_powerset.1 h) s.nodup,
    s.nodup.powerset.pmap fun _a _ha _b _hb => congr_arg Finset.val⟩


@[simp]
theorem mem_powerset {s t : Finset α} : s ∈ powerset t ↔ s ⊆ t := by
  /-
    α : Type u_1
    s t : Finset α
    ⊢ Iff (Membership.mem t.powerset s) (HasSubset.Subset s t)
  -/
  cases s
  simp [powerset, mem_mk, mem_pmap, mk.injEq, mem_powerset, exists_prop, exists_eq_right,
    ← val_le_iff]


@[simp, norm_cast]
theorem coe_powerset (s : Finset α) :
    (s.powerset : Set (Finset α)) = ((↑) : Finset α → Set α) ⁻¹' (s : Set α).powerset := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (↑s.powerset) (Set.preimage Finset.toSet (↑s).powerset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s x✝ : Finset α
    ⊢ Iff (Membership.mem (↑s.powerset) x✝) (Membership.mem (Set.preimage Finset.t …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: remove @[simp], simp can prove it

theorem empty_mem_powerset (s : Finset α) : ∅ ∈ powerset s :=
  mem_powerset.2 (empty_subset _)

-- Porting note: remove @[simp], simp can prove it

theorem mem_powerset_self (s : Finset α) : s ∈ powerset s :=
  mem_powerset.2 Subset.rfl


@[aesop safe apply (rule_sets := [finsetNonempty])]
theorem powerset_nonempty (s : Finset α) : s.powerset.Nonempty :=
  ⟨∅, empty_mem_powerset _⟩


@[simp]
theorem powerset_mono {s t : Finset α} : powerset s ⊆ powerset t ↔ s ⊆ t :=
  ⟨fun h => mem_powerset.1 <| h <| mem_powerset_self _, fun st _u h =>
    mem_powerset.2 <| Subset.trans (mem_powerset.1 h) st⟩


theorem powerset_injective : Injective (powerset : Finset α → Finset (Finset α)) :=
  (injective_of_le_imp_le _) powerset_mono.1


@[simp]
theorem powerset_inj : powerset s = powerset t ↔ s = t :=
  powerset_injective.eq_iff


@[simp]
theorem powerset_empty : (∅ : Finset α).powerset = {∅} :=
  rfl


@[simp]
theorem powerset_eq_singleton_empty : s.powerset = {∅} ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Eq s.powerset (Singleton.singleton EmptyCollection.emptyCollection)) (E …
  -/
  rw [← powerset_empty, powerset_inj]
  /-
    🎉 no goals
  -/


/-- **Number of Subsets of a Set** -/
@[simp]
theorem card_powerset (s : Finset α) : card (powerset s) = 2 ^ card s :=
  (card_pmap _ _ _).trans (Multiset.card_powerset s.1)


theorem not_mem_of_mem_powerset_of_not_mem {s t : Finset α} {a : α} (ht : t ∈ s.powerset)
    (h : a ∉ s) : a ∉ t := by
  /-
    α : Type u_1
    s t : Finset α
    a : α
    ht : Membership.mem s.powerset t
    h : Not (Membership.mem s a)
    ⊢ Not (Membership.mem t a)
  -/
  apply mt _ h
  /-
    α : Type u_1
    s t : Finset α
    a : α
    ht : Membership.mem s.powerset t
    h : Not (Membership.mem s a)
    ⊢ Membership.mem t a → Membership.mem s a
  -/
  apply mem_powerset.1 ht
  /-
    🎉 no goals
  -/


theorem powerset_insert [DecidableEq α] (s : Finset α) (a : α) :
    powerset (insert a s) = s.powerset ∪ s.powerset.image (insert a) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ⊢ Eq (Insert.insert a s).powerset (Union.union s.powerset (Finset.image (Inser …
  -/
  ext t
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    t : Finset α
    ⊢ Iff (Membership.mem (Insert.insert a s).powerset t) (Membership.mem (Union.u …
  -/
  simp only [exists_prop, mem_powerset, mem_image, mem_union, subset_insert_iff]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    t : Finset α
    ⊢ Iff (HasSubset.Subset (t.erase a) s) (Or (HasSubset.Subset t s) (Exists fun  …
  -/
  by_cases h : a ∈ t
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      t : Finset α
      h : Membership.mem t a
      ⊢ Iff (HasSubset.Subset (t.erase a) s) (Or (HasSubset.Subset t s) (Exists fun  …
    -/
  · constructor
      /-
        case pos.mp
        α : Type u_1
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        t : Finset α
        h : Membership.mem t a
        ⊢ HasSubset.Subset (t.erase a) s → Or (HasSubset.Subset t s) (Exists fun a_2 = …
      -/
    · exact fun H => Or.inr ⟨_, H, insert_erase h⟩
      /-
        🎉 no goals
      -/
      /-
        case pos.mpr
        α : Type u_1
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        t : Finset α
        h : Membership.mem t a
        ⊢ Or (HasSubset.Subset t s) (Exists fun a_1 => And (HasSubset.Subset a_1 s) (E …
      -/
    · intro H
      /-
        case pos.mpr
        α : Type u_1
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        t : Finset α
        h : Membership.mem t a
        H : Or (HasSubset.Subset t s) (Exists fun a_1 => And (HasSubset.Subset a_1 s)  …
        ⊢ HasSubset.Subset (t.erase a) s
      -/
      cases' H with H H
        /-
          case pos.mpr.inl
          α : Type u_1
          inst✝ : DecidableEq α
          s : Finset α
          a : α
          t : Finset α
          h : Membership.mem t a
          H : HasSubset.Subset t s
          ⊢ HasSubset.Subset (t.erase a) s
        -/
      · exact Subset.trans (erase_subset a t) H
        /-
          🎉 no goals
        -/
        /-
          case pos.mpr.inr
          α : Type u_1
          inst✝ : DecidableEq α
          s : Finset α
          a : α
          t : Finset α
          h : Membership.mem t a
          H : Exists fun a_1 => And (HasSubset.Subset a_1 s) (Eq (Insert.insert a a_1) t)
          ⊢ HasSubset.Subset (t.erase a) s
        -/
      · rcases H with ⟨u, hu⟩
        /-
          case pos.mpr.inr.intro
          α : Type u_1
          inst✝ : DecidableEq α
          s : Finset α
          a : α
          t : Finset α
          h : Membership.mem t a
          u : Finset α
          hu : And (HasSubset.Subset u s) (Eq (Insert.insert a u) t)
          ⊢ HasSubset.Subset (t.erase a) s
        -/
        rw [← hu.2]
        /-
          case pos.mpr.inr.intro
          α : Type u_1
          inst✝ : DecidableEq α
          s : Finset α
          a : α
          t : Finset α
          h : Membership.mem t a
          u : Finset α
          hu : And (HasSubset.Subset u s) (Eq (Insert.insert a u) t)
          ⊢ HasSubset.Subset ((Insert.insert a u).erase a) s
        -/
        exact Subset.trans (erase_insert_subset a u) hu.1
        /-
          🎉 no goals
        -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      t : Finset α
      h : Not (Membership.mem t a)
      ⊢ Iff (HasSubset.Subset (t.erase a) s) (Or (HasSubset.Subset t s) (Exists fun  …
    -/
  · have : ¬∃ u : Finset α, u ⊆ s ∧ insert a u = t := by simp [Ne.symm (ne_insert_of_not_mem _ _ h)]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      t : Finset α
      h : Not (Membership.mem t a)
      this : Not (Exists fun u => And (HasSubset.Subset u s) (Eq (Insert.insert a u) …
      ⊢ Iff (HasSubset.Subset (t.erase a) s) (Or (HasSubset.Subset t s) (Exists fun  …
    -/
    simp [Finset.erase_eq_of_not_mem h, this]
    /-
      🎉 no goals
    -/


/-- For predicate `p` decidable on subsets, it is decidable whether `p` holds for any subset. -/
instance decidableExistsOfDecidableSubsets {s : Finset α} {p : ∀ t ⊆ s, Prop}
    [∀ (t) (h : t ⊆ s), Decidable (p t h)] : Decidable (∃ (t : _) (h : t ⊆ s), p t h) :=
  decidable_of_iff (∃ (t : _) (hs : t ∈ s.powerset), p t (mem_powerset.1 hs))
    ⟨fun ⟨t, _, hp⟩ => ⟨t, _, hp⟩, fun ⟨t, hs, hp⟩ => ⟨t, mem_powerset.2 hs, hp⟩⟩


/-- For predicate `p` decidable on subsets, it is decidable whether `p` holds for every subset. -/
instance decidableForallOfDecidableSubsets {s : Finset α} {p : ∀ t ⊆ s, Prop}
    [∀ (t) (h : t ⊆ s), Decidable (p t h)] : Decidable (∀ (t) (h : t ⊆ s), p t h) :=
  decidable_of_iff (∀ (t) (h : t ∈ s.powerset), p t (mem_powerset.1 h))
    ⟨fun h t hs => h t (mem_powerset.2 hs), fun h _ _ => h _ _⟩


/-- For predicate `p` decidable on subsets, it is decidable whether `p` holds for any subset. -/
instance decidableExistsOfDecidableSubsets' {s : Finset α} {p : Finset α → Prop}
    [∀ t, Decidable (p t)] : Decidable (∃ t ⊆ s, p t) :=
                                                       /-
                                                         α : Type u_1
                                                         s✝ t s : Finset α
                                                         p : Finset α → Prop
                                                         inst✝ : (t : Finset α) → Decidable (p t)
                                                         ⊢ Iff (Exists fun t => Exists fun _h => p t) (Exists fun t => And (HasSubset.S …
                                                       -/
  decidable_of_iff (∃ (t : _) (_h : t ⊆ s), p t) <| by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- For predicate `p` decidable on subsets, it is decidable whether `p` holds for every subset. -/
instance decidableForallOfDecidableSubsets' {s : Finset α} {p : Finset α → Prop}
    [∀ t, Decidable (p t)] : Decidable (∀ t ⊆ s, p t) :=
                                                       /-
                                                         α : Type u_1
                                                         s✝ t s : Finset α
                                                         p : Finset α → Prop
                                                         inst✝ : (t : Finset α) → Decidable (p t)
                                                         ⊢ Iff (∀ (t : Finset α), HasSubset.Subset t s → p t) (∀ (t : Finset α), HasSub …
                                                       -/
  decidable_of_iff (∀ (t : _) (_h : t ⊆ s), p t) <| by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- For `s` a finset, `s.ssubsets` is the finset comprising strict subsets of `s`. -/
def ssubsets (s : Finset α) : Finset (Finset α) :=
  erase (powerset s) s


@[simp]
theorem mem_ssubsets {s t : Finset α} : t ∈ s.ssubsets ↔ t ⊂ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Iff (Membership.mem s.ssubsets t) (HasSSubset.SSubset t s)
  -/
  rw [ssubsets, mem_erase, mem_powerset, ssubset_iff_subset_ne, and_comm]
  /-
    🎉 no goals
  -/


theorem empty_mem_ssubsets {s : Finset α} (h : s.Nonempty) : ∅ ∈ s.ssubsets := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    h : s.Nonempty
    ⊢ Membership.mem s.ssubsets EmptyCollection.emptyCollection
  -/
  rw [mem_ssubsets, ssubset_iff_subset_ne]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    h : s.Nonempty
    ⊢ And (HasSubset.Subset EmptyCollection.emptyCollection s) (Ne EmptyCollection …
  -/
  exact ⟨empty_subset s, h.ne_empty.symm⟩
  /-
    🎉 no goals
  -/


/-- For predicate `p` decidable on ssubsets, it is decidable whether `p` holds for any ssubset. -/
def decidableExistsOfDecidableSSubsets {s : Finset α} {p : ∀ t ⊂ s, Prop}
    [∀ t h, Decidable (p t h)] : Decidable (∃ t h, p t h) :=
  decidable_of_iff (∃ (t : _) (hs : t ∈ s.ssubsets), p t (mem_ssubsets.1 hs))
    ⟨fun ⟨t, _, hp⟩ => ⟨t, _, hp⟩, fun ⟨t, hs, hp⟩ => ⟨t, mem_ssubsets.2 hs, hp⟩⟩


/-- For predicate `p` decidable on ssubsets, it is decidable whether `p` holds for every ssubset. -/
def decidableForallOfDecidableSSubsets {s : Finset α} {p : ∀ t ⊂ s, Prop}
    [∀ t h, Decidable (p t h)] : Decidable (∀ t h, p t h) :=
  decidable_of_iff (∀ (t) (h : t ∈ s.ssubsets), p t (mem_ssubsets.1 h))
    ⟨fun h t hs => h t (mem_ssubsets.2 hs), fun h _ _ => h _ _⟩


/-- A version of `Finset.decidableExistsOfDecidableSSubsets` with a non-dependent `p`.
Typeclass inference cannot find `hu` here, so this is not an instance. -/
def decidableExistsOfDecidableSSubsets' {s : Finset α} {p : Finset α → Prop}
    (hu : ∀ t ⊂ s, Decidable (p t)) : Decidable (∃ (t : _) (_h : t ⊂ s), p t) :=
  @Finset.decidableExistsOfDecidableSSubsets _ _ _ _ hu


/-- A version of `Finset.decidableForallOfDecidableSSubsets` with a non-dependent `p`.
Typeclass inference cannot find `hu` here, so this is not an instance. -/
def decidableForallOfDecidableSSubsets' {s : Finset α} {p : Finset α → Prop}
    (hu : ∀ t ⊂ s, Decidable (p t)) : Decidable (∀ t ⊂ s, p t) :=
  @Finset.decidableForallOfDecidableSSubsets _ _ _ _ hu


/-- Given an integer `n` and a finset `s`, then `powersetCard n s` is the finset of subsets of `s`
of cardinality `n`. -/
def powersetCard (n : ℕ) (s : Finset α) : Finset (Finset α) :=
  ⟨((s.1.powersetCard n).pmap Finset.mk) fun _t h => nodup_of_le (mem_powersetCard.1 h).1 s.2,
    s.2.powersetCard.pmap fun _a _ha _b _hb => congr_arg Finset.val⟩


@[simp] lemma mem_powersetCard : s ∈ powersetCard n t ↔ s ⊆ t ∧ card s = n := by
  /-
    α : Type u_1
    n : Nat
    s t : Finset α
    ⊢ Iff (Membership.mem (Finset.powersetCard n t) s) (And (HasSubset.Subset s t) …
  -/
  cases s; simp [powersetCard, val_le_iff.symm]
           /-
             🎉 no goals
           -/


@[simp]
theorem powersetCard_mono {n} {s t : Finset α} (h : s ⊆ t) : powersetCard n s ⊆ powersetCard n t :=
  fun _u h' => mem_powersetCard.2 <|
    And.imp (fun h₂ => Subset.trans h₂ h) id (mem_powersetCard.1 h')


/-- **Formula for the Number of Combinations** -/
@[simp]
theorem card_powersetCard (n : ℕ) (s : Finset α) :
    card (powersetCard n s) = Nat.choose (card s) n :=
  (card_pmap _ _ _).trans (Multiset.card_powersetCard n s.1)


@[simp]
theorem powersetCard_zero (s : Finset α) : s.powersetCard 0 = {∅} := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (Finset.powersetCard 0 s) (Singleton.singleton EmptyCollection.emptyColle …
  -/
  ext; rw [mem_powersetCard, mem_singleton, card_eq_zero]
  refine
    ⟨fun h => h.2, fun h => by
      rw [h]
      exact ⟨empty_subset s, rfl⟩⟩


lemma powersetCard_empty_subsingleton (n : ℕ) :
    (powersetCard n (∅ : Finset α) : Set <| Finset α).Subsingleton := by
  /-
    α : Type u_1
    n : Nat
    ⊢ (↑(Finset.powersetCard n EmptyCollection.emptyCollection)).Subsingleton
  -/
  simp [Set.Subsingleton, subset_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_val_val_powersetCard (s : Finset α) (i : ℕ) :
    (s.powersetCard i).val.map Finset.val = s.1.powersetCard i := by
  /-
    α : Type u_1
    s : Finset α
    i : Nat
    ⊢ Eq (Multiset.map Finset.val (Finset.powersetCard i s).val) (Multiset.powerse …
  -/
  simp [Finset.powersetCard, map_pmap, pmap_eq_map, map_id']
  /-
    🎉 no goals
  -/


theorem powersetCard_one (s : Finset α) :
    s.powersetCard 1 = s.map ⟨_, Finset.singleton_injective⟩ :=
                                                          /-
                                                            α : Type u_1
                                                            s : Finset α
                                                            ⊢ Eq (Multiset.map Finset.val (Finset.powersetCard 1 s).val) (Multiset.map Fin …
                                                          -/
  eq_of_veq <| Multiset.map_injective val_injective <| by simp [Multiset.powersetCard_one]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
lemma powersetCard_eq_empty : powersetCard n s = ∅ ↔ s.card < n := by
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Iff (Eq (Finset.powersetCard n s) EmptyCollection.emptyCollection) (LT.lt s. …
  -/
  refine ⟨?_, fun h ↦ card_eq_zero.1 <| by rw [card_powersetCard, Nat.choose_eq_zero_of_lt h]⟩
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Eq (Finset.powersetCard n s) EmptyCollection.emptyCollection → LT.lt s.card n
  -/
  contrapose!
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ LE.le n s.card → Ne (Finset.powersetCard n s) EmptyCollection.emptyCollection
  -/
  exact fun h ↦ nonempty_iff_ne_empty.1 <| (exists_subset_card_eq h).imp <| by simp
  /-
    🎉 no goals
  -/


@[simp] lemma powersetCard_card_add (s : Finset α) (hn : 0 < n) :
                                          /-
                                            α : Type u_1
                                            n : Nat
                                            s : Finset α
                                            hn : LT.lt 0 n
                                            ⊢ Eq (Finset.powersetCard (HAdd.hAdd s.card n) s) EmptyCollection.emptyCollect …
                                          -/
    s.powersetCard (s.card + n) = ∅ := by simpa
                                          /-
                                            🎉 no goals
                                          -/


theorem powersetCard_eq_filter {n} {s : Finset α} :
    powersetCard n s = (powerset s).filter fun x => x.card = n := by
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Eq (Finset.powersetCard n s) (Finset.filter (fun x => Eq x.card n) s.powerset)
  -/
  ext
  /-
    case h
    α : Type u_1
    n : Nat
    s a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.powersetCard n s) a✝) (Membership.mem (Finset.fi …
  -/
  simp [mem_powersetCard]
  /-
    🎉 no goals
  -/


theorem powersetCard_succ_insert [DecidableEq α] {x : α} {s : Finset α} (h : x ∉ s) (n : ℕ) :
    powersetCard n.succ (insert x s) =
    powersetCard n.succ s ∪ (powersetCard n s).image (insert x) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    ⊢ Eq (Finset.powersetCard n.succ (Insert.insert x s)) (Union.union (Finset.pow …
  -/
  rw [powersetCard_eq_filter, powerset_insert, filter_union, ← powersetCard_eq_filter]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    ⊢ Eq (Union.union (Finset.powersetCard n.succ s) (Finset.filter (fun x => Eq x …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    ⊢ Eq (Finset.filter (fun x => Eq x.card n.succ) (Finset.image (Insert.insert x …
  -/
  rw [powersetCard_eq_filter, filter_image]
  /-
    case e_a
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    ⊢ Eq (Finset.image (Insert.insert x) (Finset.filter (fun a => Eq (Insert.inser …
  -/
  congr 1
  /-
    case e_a.e_s
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    ⊢ Eq (Finset.filter (fun a => Eq (Insert.insert x a).card n.succ) s.powerset)  …
  -/
  ext t
  /-
    case e_a.e_s.h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    t : Finset α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => Eq (Insert.insert x a).card n.s …
  -/
  simp only [mem_powerset, mem_filter, Function.comp_apply, and_congr_right_iff]
  /-
    case e_a.e_s.h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    t : Finset α
    ⊢ HasSubset.Subset t s → Iff (Eq (Insert.insert x t).card n.succ) (Eq t.card n)
  -/
  intro ht
  /-
    case e_a.e_s.h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    t : Finset α
    ht : HasSubset.Subset t s
    ⊢ Iff (Eq (Insert.insert x t).card n.succ) (Eq t.card n)
  -/
  have : x ∉ t := fun H => h (ht H)
  /-
    case e_a.e_s.h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    s : Finset α
    h : Not (Membership.mem s x)
    n : Nat
    t : Finset α
    ht : HasSubset.Subset t s
    this : Not (Membership.mem t x)
    ⊢ Iff (Eq (Insert.insert x t).card n.succ) (Eq t.card n)
  -/
  simp [card_insert_of_not_mem this, Nat.succ_inj']
  /-
    🎉 no goals
  -/


@[simp]
lemma powersetCard_nonempty : (powersetCard n s).Nonempty ↔ n ≤ s.card := by
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Iff (Finset.powersetCard n s).Nonempty (LE.le n s.card)
  -/
  aesop (add simp [Finset.Nonempty, exists_subset_card_eq, card_le_card])
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, powersetCard_nonempty_of_le⟩ := powersetCard_nonempty


@[simp]
theorem powersetCard_self (s : Finset α) : powersetCard s.card s = {s} := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (Finset.powersetCard s.card s) (Singleton.singleton s)
  -/
  ext
  /-
    case h
    α : Type u_1
    s a✝ : Finset α
    ⊢ Iff (Membership.mem (Finset.powersetCard s.card s) a✝) (Membership.mem (Sing …
  -/
  rw [mem_powersetCard, mem_singleton]
  /-
    case h
    α : Type u_1
    s a✝ : Finset α
    ⊢ Iff (And (HasSubset.Subset a✝ s) (Eq a✝.card s.card)) (Eq a✝ s)
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      s a✝ : Finset α
      ⊢ And (HasSubset.Subset a✝ s) (Eq a✝.card s.card) → Eq a✝ s
    -/
  · exact fun ⟨hs, hc⟩ => eq_of_subset_of_card_le hs hc.ge
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      s a✝ : Finset α
      ⊢ Eq a✝ s → And (HasSubset.Subset a✝ s) (Eq a✝.card s.card)
    -/
  · rintro rfl
    /-
      case h.mpr
      α : Type u_1
      a✝ : Finset α
      ⊢ And (HasSubset.Subset a✝ a✝) (Eq a✝.card a✝.card)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem pairwise_disjoint_powersetCard (s : Finset α) :
    Pairwise fun i j => Disjoint (s.powersetCard i) (s.powersetCard j) := fun _i _j hij =>
  Finset.disjoint_left.mpr fun _x hi hj =>
    hij <| (mem_powersetCard.mp hi).2.symm.trans (mem_powersetCard.mp hj).2


theorem powerset_card_disjiUnion (s : Finset α) :
    Finset.powerset s =
      (range (s.card + 1)).disjiUnion (fun i => powersetCard i s)
        (s.pairwise_disjoint_powersetCard.set_pairwise _) := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq s.powerset ((Finset.range (HAdd.hAdd s.card 1)).disjiUnion (fun i => Fins …
  -/
  refine ext fun a => ⟨fun ha => ?_, fun ha => ?_⟩
    /-
      case refine_1
      α : Type u_1
      s a : Finset α
      ha : Membership.mem s.powerset a
      ⊢ Membership.mem ((Finset.range (HAdd.hAdd s.card 1)).disjiUnion (fun i => Fin …
    -/
  · rw [mem_disjiUnion]
    exact
      ⟨a.card, mem_range.mpr (Nat.lt_succ_of_le (card_le_card (mem_powerset.mp ha))),
        mem_powersetCard.mpr ⟨mem_powerset.mp ha, rfl⟩⟩
    /-
      case refine_2
      α : Type u_1
      s a : Finset α
      ha : Membership.mem ((Finset.range (HAdd.hAdd s.card 1)).disjiUnion (fun i =>  …
      ⊢ Membership.mem s.powerset a
    -/
  · rcases mem_disjiUnion.mp ha with ⟨i, _hi, ha⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      s a : Finset α
      ha✝ : Membership.mem ((Finset.range (HAdd.hAdd s.card 1)).disjiUnion (fun i => …
      i : Nat
      _hi : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) i
      ha : Membership.mem (Finset.powersetCard i s) a
      ⊢ Membership.mem s.powerset a
    -/
    exact mem_powerset.mpr (mem_powersetCard.mp ha).1
    /-
      🎉 no goals
    -/


theorem powerset_card_biUnion [DecidableEq (Finset α)] (s : Finset α) :
    Finset.powerset s = (range (s.card + 1)).biUnion fun i => powersetCard i s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq (Finset α)
    s : Finset α
    ⊢ Eq s.powerset ((Finset.range (HAdd.hAdd s.card 1)).biUnion fun i => Finset.p …
  -/
  simpa only [disjiUnion_eq_biUnion] using powerset_card_disjiUnion s
  /-
    🎉 no goals
  -/


theorem powersetCard_sup [DecidableEq α] (u : Finset α) (n : ℕ) (hn : n < u.card) :
    (powersetCard n.succ u).sup id = u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    u : Finset α
    n : Nat
    hn : LT.lt n u.card
    ⊢ Eq ((Finset.powersetCard n.succ u).sup id) u
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      ⊢ LE.le ((Finset.powersetCard n.succ u).sup id) u
    -/
  · simp_rw [Finset.sup_le_iff, mem_powersetCard]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      ⊢ ∀ (b : Finset α), And (HasSubset.Subset b u) (Eq b.card n.succ) → LE.le (id  …
    -/
    rintro x ⟨h, -⟩
    /-
      case a.intro
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      x : Finset α
      h : HasSubset.Subset x u
      ⊢ LE.le (id x) u
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      ⊢ LE.le u ((Finset.powersetCard n.succ u).sup id)
    -/
  · rw [sup_eq_biUnion, le_iff_subset, subset_iff]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      ⊢ ∀ ⦃x : α⦄, Membership.mem u x → Membership.mem ((Finset.powersetCard n.succ  …
    -/
    intro x hx
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      x : α
      hx : Membership.mem u x
      ⊢ Membership.mem ((Finset.powersetCard n.succ u).biUnion id) x
    -/
    simp only [mem_biUnion, exists_prop, id]
    obtain ⟨t, ht⟩ : ∃ t, t ∈ powersetCard n (u.erase x) := powersetCard_nonempty.2
      (le_trans (Nat.le_sub_one_of_lt hn) pred_card_le_card_erase)
    /-
      case a.intro
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      x : α
      hx : Membership.mem u x
      t : Finset α
      ht : Membership.mem (Finset.powersetCard n (u.erase x)) t
      ⊢ Exists fun a => And (Membership.mem (Finset.powersetCard n.succ u) a) (Membe …
    -/
    refine ⟨insert x t, ?_, mem_insert_self _ _⟩
    /-
      case a.intro
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      x : α
      hx : Membership.mem u x
      t : Finset α
      ht : Membership.mem (Finset.powersetCard n (u.erase x)) t
      ⊢ Membership.mem (Finset.powersetCard n.succ u) (Insert.insert x t)
    -/
    rw [← insert_erase hx, powersetCard_succ_insert (not_mem_erase _ _)]
    /-
      case a.intro
      α : Type u_1
      inst✝ : DecidableEq α
      u : Finset α
      n : Nat
      hn : LT.lt n u.card
      x : α
      hx : Membership.mem u x
      t : Finset α
      ht : Membership.mem (Finset.powersetCard n (u.erase x)) t
      ⊢ Membership.mem (Union.union (Finset.powersetCard n.succ (u.erase x)) (Finset …
    -/
    exact mem_union_right _ (mem_image_of_mem _ ht)
    /-
      🎉 no goals
    -/


theorem powersetCard_map {β : Type*} (f : α ↪ β) (n : ℕ) (s : Finset α) :
    powersetCard n (s.map f) = (powersetCard n s).map (mapEmbedding f).toEmbedding :=
  ext fun t => by
    /-
      α : Type u_1
      β : Type u_2
      f : Function.Embedding α β
      n : Nat
      s : Finset α
      t : Finset β
      ⊢ Iff (Membership.mem (Finset.powersetCard n (Finset.map f s)) t) (Membership. …
    -/
    simp only [card_map, mem_powersetCard, le_eq_subset, gt_iff_lt, mem_map, mapEmbedding_apply]
    /-
      α : Type u_1
      β : Type u_2
      f : Function.Embedding α β
      n : Nat
      s : Finset α
      t : Finset β
      ⊢ Iff (And (HasSubset.Subset t (Finset.map f s)) (Eq t.card n)) (Exists fun a  …
    -/
    constructor
    · classical
      intro h
      have : map f (filter (fun x => (f x ∈ t)) s) = t := by
        ext x
        simp only [mem_map, mem_filter, decide_eq_true_eq]
        exact ⟨fun ⟨_y, ⟨_hy₁, hy₂⟩, hy₃⟩ => hy₃ ▸ hy₂,
          fun hx => let ⟨y, hy⟩ := mem_map.1 (h.1 hx); ⟨y, ⟨hy.1, hy.2 ▸ hx⟩, hy.2⟩⟩
      refine ⟨_, ?_, this⟩
      rw [← card_map f, this, h.2]; simp
      /-
        case mpr
        α : Type u_1
        β : Type u_2
        f : Function.Embedding α β
        n : Nat
        s : Finset α
        t : Finset β
        ⊢ (Exists fun a => And (And (HasSubset.Subset a s) (Eq a.card n)) (Eq ((Finset …
      -/
    · rintro ⟨a, ⟨has, rfl⟩, rfl⟩
      /-
        case mpr.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f : Function.Embedding α β
        s a : Finset α
        has : HasSubset.Subset a s
        ⊢ And (HasSubset.Subset ((Finset.mapEmbedding f).toEmbedding a) (Finset.map f  …
      -/
      dsimp [RelEmbedding.coe_toEmbedding]
      -- Porting note: Why is `rw` required here and not `simp`?
      /-
        case mpr.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f : Function.Embedding α β
        s a : Finset α
        has : HasSubset.Subset a s
        ⊢ And (HasSubset.Subset ((Finset.mapEmbedding f) a) (Finset.map f s)) (Eq ((Fi …
      -/
      rw [mapEmbedding_apply]
      /-
        case mpr.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f : Function.Embedding α β
        s a : Finset α
        has : HasSubset.Subset a s
        ⊢ And (HasSubset.Subset (Finset.map f a) (Finset.map f s)) (Eq (Finset.map f a …
      -/
      simp [has]
      /-
        🎉 no goals
      -/


