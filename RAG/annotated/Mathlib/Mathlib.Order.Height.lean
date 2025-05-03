/-- The set of strictly ascending lists of `α` contained in a `Set α`. -/
def subchain : Set (List α) :=
  { l | l.Chain' (· < ·) ∧ ∀ i ∈ l, i ∈ s }


@[simp] -- porting note: new `simp`
theorem nil_mem_subchain : [] ∈ s.subchain := ⟨trivial, fun _ ↦ nofun⟩


theorem cons_mem_subchain_iff :
    (a::l) ∈ s.subchain ↔ a ∈ s ∧ l ∈ s.subchain ∧ ∀ b ∈ l.head?, a < b := by
  simp only [subchain, mem_setOf_eq, forall_mem_cons, chain'_cons', and_left_comm, and_comm,
    and_assoc]


@[simp]
                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : LT α
                                                                      s : Set α
                                                                      a : α
                                                                      ⊢ Iff (Membership.mem s.subchain (List.cons a List.nil)) (Membership.mem s a)
                                                                    -/
theorem singleton_mem_subchain_iff : [a] ∈ s.subchain ↔ a ∈ s := by simp [cons_mem_subchain_iff]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance : Nonempty s.subchain :=
  ⟨⟨[], s.nil_mem_subchain⟩⟩


/-- The maximal length of a strictly ascending sequence in a partial order. -/
noncomputable def chainHeight : ℕ∞ :=
  ⨆ l ∈ s.subchain, length l


theorem chainHeight_eq_iSup_subtype : s.chainHeight = ⨆ l : s.subchain, ↑l.1.length :=
  iSup_subtype'


theorem exists_chain_of_le_chainHeight {n : ℕ} (hn : ↑n ≤ s.chainHeight) :
    ∃ l ∈ s.subchain, length l = n := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    hn : LE.le (↑n) s.chainHeight
    ⊢ Exists fun l => And (Membership.mem s.subchain l) (Eq l.length n)
  -/
  rcases (le_top : s.chainHeight ≤ ⊤).eq_or_lt with ha | ha <;>
    /-
      case inl
      α : Type u_1
      inst✝ : LT α
      s : Set α
      n : Nat
      hn : LE.le (↑n) s.chainHeight
      ha : Eq s.chainHeight Top.top
      ⊢ Exists fun l => And (Membership.mem s.subchain l) (Eq l.length n)
    -/
    rw [chainHeight_eq_iSup_subtype] at ha
  · obtain ⟨_, ⟨⟨l, h₁, h₂⟩, rfl⟩, h₃⟩ :=
      not_bddAbove_iff'.mp (WithTop.iSup_coe_eq_top.1 ha) n
    exact ⟨l.take n, ⟨h₁.take _, fun x h ↦ h₂ _ <| take_subset _ _ h⟩,
      (l.length_take n).trans <| min_eq_left <| le_of_not_ge h₃⟩
    /-
      case inr
      α : Type u_1
      inst✝ : LT α
      s : Set α
      n : Nat
      hn : LE.le (↑n) s.chainHeight
      ha : LT.lt (iSup fun l => ↑(↑l).length) Top.top
      ⊢ Exists fun l => And (Membership.mem s.subchain l) (Eq l.length n)
    -/
  · rw [ENat.iSup_coe_lt_top] at ha
    /-
      case inr
      α : Type u_1
      inst✝ : LT α
      s : Set α
      n : Nat
      hn : LE.le (↑n) s.chainHeight
      ha : BddAbove (Set.range fun l => (↑l).length)
      ⊢ Exists fun l => And (Membership.mem s.subchain l) (Eq l.length n)
    -/
    obtain ⟨⟨l, h₁, h₂⟩, e : l.length = _⟩ := Nat.sSup_mem (Set.range_nonempty _) ha
    refine
      ⟨l.take n, ⟨h₁.take _, fun x h ↦ h₂ _ <| take_subset _ _ h⟩,
        (l.length_take n).trans <| min_eq_left <| ?_⟩
    /-
      case inr.intro.mk.intro
      α : Type u_1
      inst✝ : LT α
      s : Set α
      n : Nat
      hn : LE.le (↑n) s.chainHeight
      ha : BddAbove (Set.range fun l => (↑l).length)
      l : List α
      h₁ : List.Chain' (fun x1 x2 => LT.lt x1 x2) l
      h₂ : ∀ (i : α), Membership.mem l i → Membership.mem s i
      e : Eq l.length (SupSet.sSup (Set.range fun l => (↑l).length))
      ⊢ LE.le n l.length
    -/
    rwa [e, ← Nat.cast_le (α := ℕ∞), sSup_range, ENat.coe_iSup ha, ← chainHeight_eq_iSup_subtype]
    /-
      🎉 no goals
    -/


theorem le_chainHeight_TFAE (n : ℕ) :
    TFAE [↑n ≤ s.chainHeight, ∃ l ∈ s.subchain, length l = n, ∃ l ∈ s.subchain, n ≤ length l] := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    ⊢ (List.cons (LE.le (↑n) s.chainHeight) (List.cons (Exists fun l => And (Membe …
  -/
  tfae_have 1 → 2 := s.exists_chain_of_le_chainHeight
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    tfae_1_to_2 : LE.le (↑n) s.chainHeight → Exists fun l => And (Membership.mem s …
    ⊢ (List.cons (LE.le (↑n) s.chainHeight) (List.cons (Exists fun l => And (Membe …
  -/
  tfae_have 2 → 3 := fun ⟨l, hls, he⟩ ↦ ⟨l, hls, he.ge⟩
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    tfae_1_to_2 : LE.le (↑n) s.chainHeight → Exists fun l => And (Membership.mem s …
    tfae_2_to_3 : (Exists fun l => And (Membership.mem s.subchain l) (Eq l.length  …
    ⊢ (List.cons (LE.le (↑n) s.chainHeight) (List.cons (Exists fun l => And (Membe …
  -/
  tfae_have 3 → 1 := fun ⟨l, hs, hn⟩ ↦ le_iSup₂_of_le l hs (WithTop.coe_le_coe.2 hn)
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    tfae_1_to_2 : LE.le (↑n) s.chainHeight → Exists fun l => And (Membership.mem s …
    tfae_2_to_3 : (Exists fun l => And (Membership.mem s.subchain l) (Eq l.length  …
    tfae_3_to_1 : (Exists fun l => And (Membership.mem s.subchain l) (LE.le n l.le …
    ⊢ (List.cons (LE.le (↑n) s.chainHeight) (List.cons (Exists fun l => And (Membe …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem le_chainHeight_iff {n : ℕ} : ↑n ≤ s.chainHeight ↔ ∃ l ∈ s.subchain, length l = n :=
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n : Nat
    ⊢ Eq ((List.cons (LE.le (↑n) s.chainHeight) (List.cons (Exists fun l => And (M …
  -/
  /-
    🎉 no goals
  -/
  (le_chainHeight_TFAE s n).out 0 1
  /-
    🎉 no goals
  -/


theorem length_le_chainHeight_of_mem_subchain (hl : l ∈ s.subchain) : ↑l.length ≤ s.chainHeight :=
  le_chainHeight_iff.mpr ⟨l, hl, rfl⟩


theorem chainHeight_eq_top_iff : s.chainHeight = ⊤ ↔ ∀ n, ∃ l ∈ s.subchain, length l = n := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    ⊢ Iff (Eq s.chainHeight Top.top) (∀ (n : Nat), Exists fun l => And (Membership …
  -/
  refine ⟨fun h n ↦ le_chainHeight_iff.1 (le_top.trans_eq h.symm), fun h ↦ ?_⟩
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
    ⊢ Eq s.chainHeight Top.top
  -/
  contrapose! h; obtain ⟨n, hn⟩ := WithTop.ne_top_iff_exists.1 h
  exact ⟨n + 1, fun l hs ↦ (Nat.lt_succ_iff.2 <| Nat.cast_le.1 <|
    (length_le_chainHeight_of_mem_subchain hs).trans_eq hn.symm).ne⟩


@[simp]
theorem one_le_chainHeight_iff : 1 ≤ s.chainHeight ↔ s.Nonempty := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    ⊢ Iff (LE.le 1 s.chainHeight) s.Nonempty
  -/
  rw [← Nat.cast_one, Set.le_chainHeight_iff]
  simp only [length_eq_one, @and_comm (_ ∈ _), @eq_comm _ _ [_], exists_exists_eq_and,
    singleton_mem_subchain_iff, Set.Nonempty]


@[simp]
theorem chainHeight_eq_zero_iff : s.chainHeight = 0 ↔ s = ∅ := by
  rw [← not_iff_not, ← Ne, ← ENat.one_le_iff_ne_zero, one_le_chainHeight_iff,
    nonempty_iff_ne_empty]


@[simp]
theorem chainHeight_empty : (∅ : Set α).chainHeight = 0 :=
  chainHeight_eq_zero_iff.2 rfl


@[simp]
theorem chainHeight_of_isEmpty [IsEmpty α] : s.chainHeight = 0 :=
  chainHeight_eq_zero_iff.mpr (Subsingleton.elim _ _)


theorem le_chainHeight_add_nat_iff {n m : ℕ} :
    ↑n ≤ s.chainHeight + m ↔ ∃ l ∈ s.subchain, n ≤ length l + m := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    n m : Nat
    ⊢ Iff (LE.le (↑n) (HAdd.hAdd s.chainHeight ↑m)) (Exists fun l => And (Membersh …
  -/
  simp_rw [← tsub_le_iff_right, ← ENat.coe_sub, (le_chainHeight_TFAE s (n - m)).out 0 2]
  /-
    🎉 no goals
  -/


theorem chainHeight_add_le_chainHeight_add (s : Set α) (t : Set β) (n m : ℕ) :
    s.chainHeight + n ≤ t.chainHeight + m ↔
      ∀ l ∈ s.subchain, ∃ l' ∈ t.subchain, length l + n ≤ length l' + m := by
  refine
    ⟨fun e l h ↦
      le_chainHeight_add_nat_iff.1
        ((add_le_add_right (length_le_chainHeight_of_mem_subchain h) _).trans e),
      fun H ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LT α
    inst✝ : LT β
    s : Set α
    t : Set β
    n m : Nat
    H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
    ⊢ LE.le (HAdd.hAdd s.chainHeight ↑n) (HAdd.hAdd t.chainHeight ↑m)
  -/
  by_cases h : s.chainHeight = ⊤
  · suffices t.chainHeight = ⊤ by
      rw [this, top_add]
      exact le_top
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : Eq s.chainHeight Top.top
      ⊢ Eq t.chainHeight Top.top
    -/
    rw [chainHeight_eq_top_iff] at h ⊢
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      ⊢ ∀ (n : Nat), Exists fun l => And (Membership.mem t.subchain l) (Eq l.length n)
    -/
    intro k
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      k : Nat
      ⊢ Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)
    -/
    have := (le_chainHeight_TFAE t k).out 1 2
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      k : Nat
      this : Iff (Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)) …
      ⊢ Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)
    -/
    rw [this]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      k : Nat
      this : Iff (Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)) …
      ⊢ Exists fun l => And (Membership.mem t.subchain l) (LE.le k l.length)
    -/
    obtain ⟨l, hs, hl⟩ := h (k + m)
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      k : Nat
      this : Iff (Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)) …
      l : List α
      hs : Membership.mem s.subchain l
      hl : Eq l.length (HAdd.hAdd k m)
      ⊢ Exists fun l => And (Membership.mem t.subchain l) (LE.le k l.length)
    -/
    obtain ⟨l', ht, hl'⟩ := H l hs
    /-
      case pos.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : ∀ (n : Nat), Exists fun l => And (Membership.mem s.subchain l) (Eq l.lengt …
      k : Nat
      this : Iff (Exists fun l => And (Membership.mem t.subchain l) (Eq l.length k)) …
      l : List α
      hs : Membership.mem s.subchain l
      hl : Eq l.length (HAdd.hAdd k m)
      l' : List β
      ht : Membership.mem t.subchain l'
      hl' : LE.le (HAdd.hAdd l.length n) (HAdd.hAdd l'.length m)
      ⊢ Exists fun l => And (Membership.mem t.subchain l) (LE.le k l.length)
    -/
    exact ⟨l', ht, (add_le_add_iff_right m).1 <| _root_.trans (hl.symm.trans_le le_self_add) hl'⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : Not (Eq s.chainHeight Top.top)
      ⊢ LE.le (HAdd.hAdd s.chainHeight ↑n) (HAdd.hAdd t.chainHeight ↑m)
    -/
  · obtain ⟨k, hk⟩ := WithTop.ne_top_iff_exists.1 h
    /-
      case neg.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : Not (Eq s.chainHeight Top.top)
      k : Nat
      hk : Eq (↑k) s.chainHeight
      ⊢ LE.le (HAdd.hAdd s.chainHeight ↑n) (HAdd.hAdd t.chainHeight ↑m)
    -/
    obtain ⟨l, hs, hl⟩ := le_chainHeight_iff.1 hk.le
    /-
      case neg.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : Not (Eq s.chainHeight Top.top)
      k : Nat
      hk : Eq (↑k) s.chainHeight
      l : List α
      hs : Membership.mem s.subchain l
      hl : Eq l.length k
      ⊢ LE.le (HAdd.hAdd s.chainHeight ↑n) (HAdd.hAdd t.chainHeight ↑m)
    -/
    rw [← hk, ← hl]
    /-
      case neg.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      s : Set α
      t : Set β
      n m : Nat
      H : ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Member …
      h : Not (Eq s.chainHeight Top.top)
      k : Nat
      hk : Eq (↑k) s.chainHeight
      l : List α
      hs : Membership.mem s.subchain l
      hl : Eq l.length k
      ⊢ LE.le (HAdd.hAdd ↑l.length ↑n) (HAdd.hAdd t.chainHeight ↑m)
    -/
    exact le_chainHeight_add_nat_iff.2 (H l hs)
    /-
      🎉 no goals
    -/


theorem chainHeight_le_chainHeight_TFAE (s : Set α) (t : Set β) :
    TFAE [s.chainHeight ≤ t.chainHeight, ∀ l ∈ s.subchain, ∃ l' ∈ t.subchain, length l = length l',
      ∀ l ∈ s.subchain, ∃ l' ∈ t.subchain, length l ≤ length l'] := by
  tfae_have 1 ↔ 3 := by
    convert ← chainHeight_add_le_chainHeight_add s t 0 0 <;> apply add_zero
  tfae_have 2 ↔ 3 := by
    refine forall₂_congr fun l _ ↦ ?_
    simp_rw [← (le_chainHeight_TFAE t l.length).out 1 2, eq_comm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LT α
    inst✝ : LT β
    s : Set α
    t : Set β
    tfae_1_iff_3 : Iff (LE.le s.chainHeight t.chainHeight) (∀ (l : List α), Member …
    tfae_2_iff_3 : Iff (∀ (l : List α), Membership.mem s.subchain l → Exists fun l …
    ⊢ (List.cons (LE.le s.chainHeight t.chainHeight) (List.cons (∀ (l : List α), M …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem chainHeight_le_chainHeight_iff {t : Set β} :
    s.chainHeight ≤ t.chainHeight ↔ ∀ l ∈ s.subchain, ∃ l' ∈ t.subchain, length l = length l' :=
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LT α
    inst✝ : LT β
    s : Set α
    t : Set β
    ⊢ Eq ((List.cons (LE.le s.chainHeight t.chainHeight) (List.cons (∀ (l : List α …
  -/
  /-
    🎉 no goals
  -/
  (chainHeight_le_chainHeight_TFAE s t).out 0 1
  /-
    🎉 no goals
  -/


theorem chainHeight_le_chainHeight_iff_le {t : Set β} :
    s.chainHeight ≤ t.chainHeight ↔ ∀ l ∈ s.subchain, ∃ l' ∈ t.subchain, length l ≤ length l' :=
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LT α
    inst✝ : LT β
    s : Set α
    t : Set β
    ⊢ Eq ((List.cons (LE.le s.chainHeight t.chainHeight) (List.cons (∀ (l : List α …
  -/
  /-
    🎉 no goals
  -/
  (chainHeight_le_chainHeight_TFAE s t).out 0 2
  /-
    🎉 no goals
  -/


theorem chainHeight_mono (h : s ⊆ t) : s.chainHeight ≤ t.chainHeight :=
  chainHeight_le_chainHeight_iff.2 fun l hl ↦ ⟨l, ⟨hl.1, fun i hi ↦ h <| hl.2 i hi⟩, rfl⟩


theorem chainHeight_image (f : α → β) (hf : ∀ {x y}, x < y ↔ f x < f y) (s : Set α) :
    (f '' s).chainHeight = s.chainHeight := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LT α
    inst✝ : LT β
    f : α → β
    hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
    s : Set α
    ⊢ Eq (Set.image f s).chainHeight s.chainHeight
  -/
  apply le_antisymm <;> rw [chainHeight_le_chainHeight_iff]
  · suffices ∀ l ∈ (f '' s).subchain, ∃ l' ∈ s.subchain, map f l' = l by
      intro l hl
      obtain ⟨l', h₁, rfl⟩ := this l hl
      exact ⟨l', h₁, length_map _ _⟩
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      f : α → β
      hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
      s : Set α
      ⊢ ∀ (l : List β), Membership.mem (Set.image f s).subchain l → Exists fun l' => …
    -/
    intro l
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      f : α → β
      hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
      s : Set α
      l : List β
      ⊢ Membership.mem (Set.image f s).subchain l → Exists fun l' => And (Membership …
    -/
    induction' l with x xs hx
      /-
        case a.nil
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        ⊢ Membership.mem (Set.image f s).subchain List.nil → Exists fun l' => And (Mem …
      -/
    · exact fun _ ↦ ⟨nil, ⟨trivial, fun x h ↦ (not_mem_nil x h).elim⟩, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case a.cons
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        x : β
        xs : List β
        hx : Membership.mem (Set.image f s).subchain xs → Exists fun l' => And (Member …
        ⊢ Membership.mem (Set.image f s).subchain (List.cons x xs) → Exists fun l' =>  …
      -/
    · intro h
      /-
        case a.cons
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        x : β
        xs : List β
        hx : Membership.mem (Set.image f s).subchain xs → Exists fun l' => And (Member …
        h : Membership.mem (Set.image f s).subchain (List.cons x xs)
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (Eq (List.map f l') (Lis …
      -/
      rw [cons_mem_subchain_iff] at h
      /-
        case a.cons
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        x : β
        xs : List β
        hx : Membership.mem (Set.image f s).subchain xs → Exists fun l' => And (Member …
        h : And (Membership.mem (Set.image f s) x) (And (Membership.mem (Set.image f s …
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (Eq (List.map f l') (Lis …
      -/
      obtain ⟨⟨x, hx', rfl⟩, h₁, h₂⟩ := h
      /-
        case a.cons.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        xs : List β
        hx : Membership.mem (Set.image f s).subchain xs → Exists fun l' => And (Member …
        x : α
        hx' : Membership.mem s x
        h₁ : Membership.mem (Set.image f s).subchain xs
        h₂ : ∀ (b : β), Membership.mem xs.head? b → LT.lt (f x) b
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (Eq (List.map f l') (Lis …
      -/
      obtain ⟨l', h₃, rfl⟩ := hx h₁
      /-
        case a.cons.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        x : α
        hx' : Membership.mem s x
        l' : List α
        h₃ : Membership.mem s.subchain l'
        hx : Membership.mem (Set.image f s).subchain (List.map f l') → Exists fun l'_1 …
        h₁ : Membership.mem (Set.image f s).subchain (List.map f l')
        h₂ : ∀ (b : β), Membership.mem (List.map f l').head? b → LT.lt (f x) b
        ⊢ Exists fun l'_1 => And (Membership.mem s.subchain l'_1) (Eq (List.map f l'_1 …
      -/
      refine ⟨x::l', Set.cons_mem_subchain_iff.mpr ⟨hx', h₃, ?_⟩, rfl⟩
      /-
        case a.cons.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        x : α
        hx' : Membership.mem s x
        l' : List α
        h₃ : Membership.mem s.subchain l'
        hx : Membership.mem (Set.image f s).subchain (List.map f l') → Exists fun l'_1 …
        h₁ : Membership.mem (Set.image f s).subchain (List.map f l')
        h₂ : ∀ (b : β), Membership.mem (List.map f l').head? b → LT.lt (f x) b
        ⊢ ∀ (b : α), Membership.mem l'.head? b → LT.lt x b
      -/
      cases l'
        /-
          case a.cons.intro.intro.intro.intro.intro.intro.nil
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          f : α → β
          hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
          s : Set α
          x : α
          hx' : Membership.mem s x
          h₃ : Membership.mem s.subchain List.nil
          hx : Membership.mem (Set.image f s).subchain (List.map f List.nil) → Exists fu …
          h₁ : Membership.mem (Set.image f s).subchain (List.map f List.nil)
          h₂ : ∀ (b : β), Membership.mem (List.map f List.nil).head? b → LT.lt (f x) b
          ⊢ ∀ (b : α), Membership.mem List.nil.head? b → LT.lt x b
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case a.cons.intro.intro.intro.intro.intro.intro.cons
          α : Type u_1
          β : Type u_2
          inst✝¹ : LT α
          inst✝ : LT β
          f : α → β
          hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
          s : Set α
          x : α
          hx' : Membership.mem s x
          head✝ : α
          tail✝ : List α
          h₃ : Membership.mem s.subchain (List.cons head✝ tail✝)
          hx : Membership.mem (Set.image f s).subchain (List.map f (List.cons head✝ tail …
          h₁ : Membership.mem (Set.image f s).subchain (List.map f (List.cons head✝ tail …
          h₂ : ∀ (b : β), Membership.mem (List.map f (List.cons head✝ tail✝)).head? b →  …
          ⊢ ∀ (b : α), Membership.mem (List.cons head✝ tail✝).head? b → LT.lt x b
        -/
      · simpa [← hf] using h₂
        /-
          🎉 no goals
        -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      f : α → β
      hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
      s : Set α
      ⊢ ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Membersh …
    -/
  · intro l hl
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : LT α
      inst✝ : LT β
      f : α → β
      hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
      s : Set α
      l : List α
      hl : Membership.mem s.subchain l
      ⊢ Exists fun l' => And (Membership.mem (Set.image f s).subchain l') (Eq l.leng …
    -/
    refine ⟨l.map f, ⟨?_, ?_⟩, ?_⟩
      /-
        case a.refine_1
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ List.Chain' (fun x1 x2 => LT.lt x1 x2) (List.map f l)
      -/
    · simp_rw [chain'_map, ← hf]
      /-
        case a.refine_1
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ List.Chain' (fun a b => LT.lt a b) l
      -/
      exact hl.1
      /-
        🎉 no goals
      -/
      /-
        case a.refine_2
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ ∀ (i : β), Membership.mem (List.map f l) i → Membership.mem (Set.image f s) i
      -/
    · intro _ e
      /-
        case a.refine_2
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        i✝ : β
        e : Membership.mem (List.map f l) i✝
        ⊢ Membership.mem (Set.image f s) i✝
      -/
      obtain ⟨a, ha, rfl⟩ := mem_map.mp e
      /-
        case a.refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        a : α
        ha : Membership.mem l a
        e : Membership.mem (List.map f l) (f a)
        ⊢ Membership.mem (Set.image f s) (f a)
      -/
      exact Set.mem_image_of_mem _ (hl.2 _ ha)
      /-
        🎉 no goals
      -/
      /-
        case a.refine_3
        α : Type u_1
        β : Type u_2
        inst✝¹ : LT α
        inst✝ : LT β
        f : α → β
        hf : ∀ {x y : α}, Iff (LT.lt x y) (LT.lt (f x) (f y))
        s : Set α
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ Eq l.length (List.map f l).length
      -/
    · rw [length_map]
      /-
        🎉 no goals
      -/


@[simp]
theorem chainHeight_dual : (ofDual ⁻¹' s).chainHeight = s.chainHeight := by
  /-
    α : Type u_1
    inst✝ : LT α
    s : Set α
    ⊢ Eq (Set.preimage (⇑OrderDual.ofDual) s).chainHeight s.chainHeight
  -/
  apply le_antisymm <;>
    /-
      case a
      α : Type u_1
      inst✝ : LT α
      s : Set α
      ⊢ LE.le (Set.preimage (⇑OrderDual.ofDual) s).chainHeight s.chainHeight
    -/
    /-
      case a
      α : Type u_1
      inst✝ : LT α
      s : Set α
      ⊢ ∀ (l : List (OrderDual α)), Membership.mem (Set.preimage (⇑OrderDual.ofDual) …
    -/
    /-
      case a
      α : Type u_1
      inst✝ : LT α
      s : Set α
      ⊢ ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Membersh …
    -/
    rintro l ⟨h₁, h₂⟩
    exact ⟨l.reverse, ⟨chain'_reverse.mpr h₁, fun i h ↦ h₂ i (mem_reverse.mp h)⟩,
      (length_reverse _).symm⟩


theorem chainHeight_eq_iSup_Ici : s.chainHeight = ⨆ i ∈ s, (s ∩ Set.Ici i).chainHeight := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    ⊢ Eq s.chainHeight (iSup fun i => iSup fun h => (Inter.inter s (Set.Ici i)).ch …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      ⊢ LE.le s.chainHeight (iSup fun i => iSup fun h => (Inter.inter s (Set.Ici i)) …
    -/
  · refine iSup₂_le ?_
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      ⊢ ∀ (i : List α), Membership.mem s.subchain i → LE.le (↑i.length) (iSup fun i  …
    -/
    rintro (_ | ⟨x, xs⟩) h
      /-
        case a.nil
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        h : Membership.mem s.subchain List.nil
        ⊢ LE.le (↑List.nil.length) (iSup fun i => iSup fun h => (Inter.inter s (Set.Ic …
      -/
    · exact zero_le _
      /-
        🎉 no goals
      -/
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        ⊢ LE.le (↑(List.cons x xs).length) (iSup fun i => iSup fun h => (Inter.inter s …
      -/
    · apply le_trans _ (le_iSup₂ x (cons_mem_subchain_iff.mp h).1)
      /-
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        ⊢ LE.le (↑(List.cons x xs).length) (Inter.inter s (Set.Ici x)).chainHeight
      -/
      apply length_le_chainHeight_of_mem_subchain
      /-
        case hl
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        ⊢ Membership.mem (Inter.inter s (Set.Ici x)).subchain (List.cons x xs)
      -/
      refine ⟨h.1, fun i hi ↦ ⟨h.2 i hi, ?_⟩⟩
      /-
        case hl
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        i : α
        hi : Membership.mem (List.cons x xs) i
        ⊢ Membership.mem (Set.Ici x) i
      -/
      cases hi
        /-
          case hl.head
          α : Type u_1
          s : Set α
          inst✝ : Preorder α
          x : α
          xs : List α
          h : Membership.mem s.subchain (List.cons x xs)
          ⊢ Membership.mem (Set.Ici x) x
        -/
      · exact left_mem_Ici
        /-
          🎉 no goals
        -/
      /-
        case hl.tail
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        i : α
        a✝ : List.Mem i xs
        ⊢ Membership.mem (Set.Ici x) i
      -/
      rename_i hi
      /-
        case hl.tail
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        i : α
        hi : List.Mem i xs
        ⊢ Membership.mem (Set.Ici x) i
      -/
      cases' chain'_iff_pairwise.mp h.1 with _ _ h'
      /-
        case hl.tail.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        x : α
        xs : List α
        h : Membership.mem s.subchain (List.cons x xs)
        i : α
        hi : List.Mem i xs
        a✝ : List.Pairwise (fun x1 x2 => LT.lt x1 x2) xs
        h' : ∀ (a' : α), Membership.mem xs a' → LT.lt x a'
        ⊢ Membership.mem (Set.Ici x) i
      -/
      exact (h' _ hi).le
      /-
        🎉 no goals
      -/
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      ⊢ LE.le (iSup fun i => iSup fun h => (Inter.inter s (Set.Ici i)).chainHeight)  …
    -/
  · exact iSup₂_le fun i _ ↦ chainHeight_mono Set.inter_subset_left
    /-
      🎉 no goals
    -/


theorem chainHeight_eq_iSup_Iic : s.chainHeight = ⨆ i ∈ s, (s ∩ Set.Iic i).chainHeight := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    ⊢ Eq s.chainHeight (iSup fun i => iSup fun h => (Inter.inter s (Set.Iic i)).ch …
  -/
  simp_rw [← chainHeight_dual (_ ∩ _)]
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    ⊢ Eq s.chainHeight (iSup fun i => iSup fun x => (Set.preimage (⇑OrderDual.ofDu …
  -/
  rw [← chainHeight_dual, chainHeight_eq_iSup_Ici]
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    ⊢ Eq (iSup fun i => iSup fun h => (Inter.inter (Set.preimage (⇑OrderDual.ofDua …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem chainHeight_insert_of_forall_gt (a : α) (hx : ∀ b ∈ s, a < b) :
    (insert a s).chainHeight = s.chainHeight + 1 := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    a : α
    hx : ∀ (b : α), Membership.mem s b → LT.lt a b
    ⊢ Eq (Insert.insert a s).chainHeight (HAdd.hAdd s.chainHeight 1)
  -/
  rw [← add_zero (insert a s).chainHeight]
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    a : α
    hx : ∀ (b : α), Membership.mem s b → LT.lt a b
    ⊢ Eq (HAdd.hAdd (Insert.insert a s).chainHeight 0) (HAdd.hAdd s.chainHeight 1)
  -/
  change (insert a s).chainHeight + (0 : ℕ) = s.chainHeight + (1 : ℕ)
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    a : α
    hx : ∀ (b : α), Membership.mem s b → LT.lt a b
    ⊢ Eq (HAdd.hAdd (Insert.insert a s).chainHeight ↑0) (HAdd.hAdd s.chainHeight ↑1)
  -/
  apply le_antisymm <;> rw [chainHeight_add_le_chainHeight_add]
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      a : α
      hx : ∀ (b : α), Membership.mem s b → LT.lt a b
      ⊢ ∀ (l : List α), Membership.mem (Insert.insert a s).subchain l → Exists fun l …
    -/
  · rintro (_ | ⟨y, ys⟩) h
      /-
        case a.nil
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        h : Membership.mem (Insert.insert a s).subchain List.nil
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (LE.le (HAdd.hAdd List.n …
      -/
    · exact ⟨[], nil_mem_subchain _, zero_le _⟩
      /-
        🎉 no goals
      -/
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        y : α
        ys : List α
        h : Membership.mem (Insert.insert a s).subchain (List.cons y ys)
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (LE.le (HAdd.hAdd (List. …
      -/
    · have h' := cons_mem_subchain_iff.mp h
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        y : α
        ys : List α
        h : Membership.mem (Insert.insert a s).subchain (List.cons y ys)
        h' : And (Membership.mem (Insert.insert a s) y) (And (Membership.mem (Insert.i …
        ⊢ Exists fun l' => And (Membership.mem s.subchain l') (LE.le (HAdd.hAdd (List. …
      -/
      refine ⟨ys, ⟨h'.2.1.1, fun i hi ↦ ?_⟩, by simp⟩
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        y : α
        ys : List α
        h : Membership.mem (Insert.insert a s).subchain (List.cons y ys)
        h' : And (Membership.mem (Insert.insert a s) y) (And (Membership.mem (Insert.i …
        i : α
        hi : Membership.mem ys i
        ⊢ Membership.mem s i
      -/
      apply (h'.2.1.2 i hi).resolve_left
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        y : α
        ys : List α
        h : Membership.mem (Insert.insert a s).subchain (List.cons y ys)
        h' : And (Membership.mem (Insert.insert a s) y) (And (Membership.mem (Insert.i …
        i : α
        hi : Membership.mem ys i
        ⊢ Not (Eq i a)
      -/
      rintro rfl
      /-
        case a.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        y : α
        ys : List α
        i : α
        hi : Membership.mem ys i
        hx : ∀ (b : α), Membership.mem s b → LT.lt i b
        h : Membership.mem (Insert.insert i s).subchain (List.cons y ys)
        h' : And (Membership.mem (Insert.insert i s) y) (And (Membership.mem (Insert.i …
        ⊢ False
      -/
      cases' chain'_iff_pairwise.mp h.1 with _ _ hy
      /-
        case a.cons.cons
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        y : α
        ys : List α
        i : α
        hi : Membership.mem ys i
        hx : ∀ (b : α), Membership.mem s b → LT.lt i b
        h : Membership.mem (Insert.insert i s).subchain (List.cons y ys)
        h' : And (Membership.mem (Insert.insert i s) y) (And (Membership.mem (Insert.i …
        a✝ : List.Pairwise (fun x1 x2 => LT.lt x1 x2) ys
        hy : ∀ (a' : α), Membership.mem ys a' → LT.lt y a'
        ⊢ False
      -/
      cases' h'.1 with h' h'
      /-
        case a.cons.cons.inl
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        y : α
        ys : List α
        i : α
        hi : Membership.mem ys i
        hx : ∀ (b : α), Membership.mem s b → LT.lt i b
        h : Membership.mem (Insert.insert i s).subchain (List.cons y ys)
        h'✝ : And (Membership.mem (Insert.insert i s) y) (And (Membership.mem (Insert. …
        a✝ : List.Pairwise (fun x1 x2 => LT.lt x1 x2) ys
        hy : ∀ (a' : α), Membership.mem ys a' → LT.lt y a'
        h' : Eq y i
        ⊢ False
      -/
      exacts [(hy _ hi).ne h', not_le_of_gt (hy _ hi) (hx _ h').le]
      /-
        🎉 no goals
      -/
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      a : α
      hx : ∀ (b : α), Membership.mem s b → LT.lt a b
      ⊢ ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Membersh …
    -/
  · intro l hl
    /-
      case a
      α : Type u_1
      s : Set α
      inst✝ : Preorder α
      a : α
      hx : ∀ (b : α), Membership.mem s b → LT.lt a b
      l : List α
      hl : Membership.mem s.subchain l
      ⊢ Exists fun l' => And (Membership.mem (Insert.insert a s).subchain l') (LE.le …
    -/
    refine ⟨a::l, ⟨?_, ?_⟩, by simp⟩
      /-
        case a.refine_1
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ List.Chain' (fun x1 x2 => LT.lt x1 x2) (List.cons a l)
      -/
    · rw [chain'_cons']
      /-
        case a.refine_1
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ And (∀ (y : α), Membership.mem l.head? y → LT.lt a y) (List.Chain' (fun x1 x …
      -/
      exact ⟨fun y hy ↦ hx _ (hl.2 _ (mem_of_mem_head? hy)), hl.1⟩
      /-
        🎉 no goals
      -/
    · -- Porting note: originally this was
        -- rintro x (rfl | hx)
        -- exacts [Or.inl (Set.mem_singleton x), Or.inr (hl.2 x hx)]
      -- but this fails because `List.Mem` is now an inductive prop.
      -- I couldn't work out how to drive `rcases` here but asked at
      -- https://leanprover.zulipchat.com/#narrow/stream/348111-std4/topic/rcases.3F/near/347976083
      /-
        case a.refine_2
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ ∀ (i : α), Membership.mem (List.cons a l) i → Membership.mem (Insert.insert  …
      -/
      rintro x (_ | _)
      /-
        case a.refine_2.head
        α : Type u_1
        s : Set α
        inst✝ : Preorder α
        a : α
        hx : ∀ (b : α), Membership.mem s b → LT.lt a b
        l : List α
        hl : Membership.mem s.subchain l
        ⊢ Membership.mem (Insert.insert a s) a
      -/
      exacts [Or.inl (Set.mem_singleton a), Or.inr (hl.2 x ‹_›)]
      /-
        🎉 no goals
      -/


theorem chainHeight_insert_of_forall_lt (a : α) (ha : ∀ b ∈ s, b < a) :
    (insert a s).chainHeight = s.chainHeight + 1 := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    a : α
    ha : ∀ (b : α), Membership.mem s b → LT.lt b a
    ⊢ Eq (Insert.insert a s).chainHeight (HAdd.hAdd s.chainHeight 1)
  -/
  rw [← chainHeight_dual, ← chainHeight_dual s]
  /-
    α : Type u_1
    s : Set α
    inst✝ : Preorder α
    a : α
    ha : ∀ (b : α), Membership.mem s b → LT.lt b a
    ⊢ Eq (Set.preimage (⇑OrderDual.ofDual) (Insert.insert a s)).chainHeight (HAdd. …
  -/
  exact chainHeight_insert_of_forall_gt _ ha
  /-
    🎉 no goals
  -/


theorem chainHeight_union_le : (s ∪ t).chainHeight ≤ s.chainHeight + t.chainHeight := by
  classical
    refine iSup₂_le fun l hl ↦ ?_
    let l₁ := l.filter (· ∈ s)
    let l₂ := l.filter (· ∈ t)
    have hl₁ : ↑l₁.length ≤ s.chainHeight := by
      apply Set.length_le_chainHeight_of_mem_subchain
      exact ⟨hl.1.sublist (filter_sublist _), fun i h ↦ by simpa using (of_mem_filter h : _)⟩
    have hl₂ : ↑l₂.length ≤ t.chainHeight := by
      apply Set.length_le_chainHeight_of_mem_subchain
      exact ⟨hl.1.sublist (filter_sublist _), fun i h ↦ by simpa using (of_mem_filter h : _)⟩
    refine le_trans ?_ (add_le_add hl₁ hl₂)
    simp_rw [l₁, l₂, ← Nat.cast_add, ← Multiset.coe_card, ← Multiset.card_add,
      ← Multiset.filter_coe]
    rw [Multiset.filter_add_filter, Multiset.filter_eq_self.mpr, Multiset.card_add, Nat.cast_add]
    exacts [le_add_right rfl.le, hl.2]


theorem chainHeight_union_eq (s t : Set α) (H : ∀ a ∈ s, ∀ b ∈ t, a < b) :
    (s ∪ t).chainHeight = s.chainHeight + t.chainHeight := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
    ⊢ Eq (Union.union s t).chainHeight (HAdd.hAdd s.chainHeight t.chainHeight)
  -/
  cases h : t.chainHeight
    /-
      case top
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      h : Eq t.chainHeight Top.top
      ⊢ Eq (Union.union s t).chainHeight (HAdd.hAdd s.chainHeight Top.top)
    -/
  · rw [add_top, eq_top_iff, ← h]
    /-
      case top
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      h : Eq t.chainHeight Top.top
      ⊢ LE.le t.chainHeight (Union.union s t).chainHeight
    -/
    exact Set.chainHeight_mono subset_union_right
    /-
      🎉 no goals
    -/
  /-
    case coe
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
    a✝ : Nat
    h : Eq t.chainHeight ↑a✝
    ⊢ Eq (Union.union s t).chainHeight (HAdd.hAdd s.chainHeight ↑a✝)
  -/
  apply le_antisymm
    /-
      case coe.a
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      a✝ : Nat
      h : Eq t.chainHeight ↑a✝
      ⊢ LE.le (Union.union s t).chainHeight (HAdd.hAdd s.chainHeight ↑a✝)
    -/
  · rw [← h]
    /-
      case coe.a
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      a✝ : Nat
      h : Eq t.chainHeight ↑a✝
      ⊢ LE.le (Union.union s t).chainHeight (HAdd.hAdd s.chainHeight t.chainHeight)
    -/
    exact chainHeight_union_le
    /-
      🎉 no goals
    -/
  rw [← add_zero (s ∪ t).chainHeight, ← WithTop.coe_zero,
    ENat.some_eq_coe, chainHeight_add_le_chainHeight_add]
  /-
    case coe.a
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
    a✝ : Nat
    h : Eq t.chainHeight ↑a✝
    ⊢ ∀ (l : List α), Membership.mem s.subchain l → Exists fun l' => And (Membersh …
  -/
  intro l hl
  /-
    case coe.a
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
    a✝ : Nat
    h : Eq t.chainHeight ↑a✝
    l : List α
    hl : Membership.mem s.subchain l
    ⊢ Exists fun l' => And (Membership.mem (Union.union s t).subchain l') (LE.le ( …
  -/
  obtain ⟨l', hl', rfl⟩ := exists_chain_of_le_chainHeight t h.symm.le
  /-
    case coe.a.intro.intro
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
    l : List α
    hl : Membership.mem s.subchain l
    l' : List α
    hl' : Membership.mem t.subchain l'
    h : Eq t.chainHeight ↑l'.length
    ⊢ Exists fun l'_1 => And (Membership.mem (Union.union s t).subchain l'_1) (LE. …
  -/
  refine ⟨l ++ l', ⟨Chain'.append hl.1 hl'.1 fun x hx y hy ↦ ?_, fun i hi ↦ ?_⟩, by simp⟩
    /-
      case coe.a.intro.intro.refine_1
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      l : List α
      hl : Membership.mem s.subchain l
      l' : List α
      hl' : Membership.mem t.subchain l'
      h : Eq t.chainHeight ↑l'.length
      x : α
      hx : Membership.mem l.getLast? x
      y : α
      hy : Membership.mem l'.head? y
      ⊢ LT.lt x y
    -/
  · exact H x (hl.2 _ <| mem_of_mem_getLast? hx) y (hl'.2 _ <| mem_of_mem_head? hy)
    /-
      🎉 no goals
    -/
    /-
      case coe.a.intro.intro.refine_2
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      l : List α
      hl : Membership.mem s.subchain l
      l' : List α
      hl' : Membership.mem t.subchain l'
      h : Eq t.chainHeight ↑l'.length
      i : α
      hi : Membership.mem (HAppend.hAppend l l') i
      ⊢ Membership.mem (Union.union s t) i
    -/
  · rw [mem_append] at hi
    /-
      case coe.a.intro.intro.refine_2
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      l : List α
      hl : Membership.mem s.subchain l
      l' : List α
      hl' : Membership.mem t.subchain l'
      h : Eq t.chainHeight ↑l'.length
      i : α
      hi : Or (Membership.mem l i) (Membership.mem l' i)
      ⊢ Membership.mem (Union.union s t) i
    -/
    cases' hi with hi hi
    /-
      case coe.a.intro.intro.refine_2.inl
      α : Type u_1
      inst✝ : Preorder α
      s t : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LT.lt a b
      l : List α
      hl : Membership.mem s.subchain l
      l' : List α
      hl' : Membership.mem t.subchain l'
      h : Eq t.chainHeight ↑l'.length
      i : α
      hi : Membership.mem l i
      ⊢ Membership.mem (Union.union s t) i
    -/
    exacts [Or.inl (hl.2 _ hi), Or.inr (hl'.2 _ hi)]
    /-
      🎉 no goals
    -/


theorem wellFoundedGT_of_chainHeight_ne_top (s : Set α) (hs : s.chainHeight ≠ ⊤) :
    WellFoundedGT s := by
  -- Porting note: added
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    hs : Ne s.chainHeight Top.top
    ⊢ WellFoundedGT ↑s
  -/
  haveI : IsTrans { x // x ∈ s } (↑· < ↑·) := inferInstance

  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    hs : Ne s.chainHeight Top.top
    this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
    ⊢ WellFoundedGT ↑s
  -/
  obtain ⟨n, hn⟩ := WithTop.ne_top_iff_exists.1 hs
  /-
    case intro
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    hs : Ne s.chainHeight Top.top
    this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
    n : Nat
    hn : Eq (↑n) s.chainHeight
    ⊢ WellFoundedGT ↑s
  -/
  refine ⟨RelEmbedding.wellFounded_iff_no_descending_seq.2 ⟨fun f ↦ ?_⟩⟩
  /-
    case intro
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    hs : Ne s.chainHeight Top.top
    this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
    n : Nat
    hn : Eq (↑n) s.chainHeight
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
    ⊢ False
  -/
  refine n.lt_succ_self.not_le (WithTop.coe_le_coe.1 <| hn.symm ▸ ?_)
  refine le_iSup₂_of_le ((ofFn (n := n.succ) fun i ↦ f i).map Subtype.val)
    ⟨chain'_map_of_chain' ((↑) : {x // x ∈ s} → α) (fun _ _ ↦ id)
      (chain'_iff_pairwise.2 <| pairwise_ofFn.2 fun i j ↦ f.map_rel_iff.2), fun i h ↦ ?_⟩ ?_
    /-
      case intro.refine_1
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      hs : Ne s.chainHeight Top.top
      this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
      n : Nat
      hn : Eq (↑n) s.chainHeight
      f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      i : α
      h : Membership.mem (List.map Subtype.val (List.ofFn fun i => f ↑i)) i
      ⊢ Membership.mem s i
    -/
  · obtain ⟨a, -, rfl⟩ := mem_map.1 h
    /-
      case intro.refine_1.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      hs : Ne s.chainHeight Top.top
      this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
      n : Nat
      hn : Eq (↑n) s.chainHeight
      f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      a : Subtype fun x => Membership.mem s x
      h : Membership.mem (List.map Subtype.val (List.ofFn fun i => f ↑i)) ↑a
      ⊢ Membership.mem s ↑a
    -/
    exact a.prop
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      hs : Ne s.chainHeight Top.top
      this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
      n : Nat
      hn : Eq (↑n) s.chainHeight
      f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      ⊢ LE.le ↑n.succ ↑(List.map Subtype.val (List.ofFn fun i => f ↑i)).length
    -/
  · rw [length_map, length_ofFn]
    /-
      case intro.refine_2
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      hs : Ne s.chainHeight Top.top
      this : IsTrans (Subtype fun x => Membership.mem s x) fun x1 x2 => LT.lt x1 x2
      n : Nat
      hn : Eq (↑n) s.chainHeight
      f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      ⊢ LE.le ↑n.succ ↑n.succ
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/


theorem wellFoundedLT_of_chainHeight_ne_top (s : Set α) (hs : s.chainHeight ≠ ⊤) :
    WellFoundedLT s :=
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : Preorder α
                                                             s : Set α
                                                             hs : Ne s.chainHeight Top.top
                                                             ⊢ Ne (Set.preimage (⇑OrderDual.ofDual) s).chainHeight Top.top
                                                           -/
  wellFoundedGT_of_chainHeight_ne_top (ofDual ⁻¹' s) <| by rwa [chainHeight_dual]
                                                           /-
                                                             🎉 no goals
                                                           -/


