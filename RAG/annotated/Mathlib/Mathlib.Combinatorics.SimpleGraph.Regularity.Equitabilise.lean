/-- Given a partition `P` of `s`, as well as a proof that `a * m + b * (m + 1) = #s`, we can
find a new partition `Q` of `s` where each part has size `m` or `m + 1`, every part of `P` is the
union of parts of `Q` plus at most `m` extra elements, there are `b` parts of size `m + 1` and
(provided `m > 0`, because a partition does not have parts of size `0`) there are `a` parts of size
`m` and hence `a + b` parts in total. -/
theorem equitabilise_aux (hs : a * m + b * (m + 1) = #s) :
    ∃ Q : Finpartition s,
      (∀ x : Finset α, x ∈ Q.parts → #x = m ∨ #x = m + 1) ∧
        (∀ x, x ∈ P.parts → #(x \ {y ∈ Q.parts | y ⊆ x}.biUnion id) ≤ m) ∧
          #{i ∈ Q.parts | #i = m + 1} = b := by
  -- Get rid of the easy case `m = 0`
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  obtain rfl | m_pos := m.eq_zero_or_pos
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a 0) (HMul.hMul b (HAdd.hAdd 0 1))) s.card
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
  · refine ⟨⊥, by simp, ?_, by simpa [Finset.filter_true_of_mem] using hs.symm⟩
    simp only [le_zero_iff, card_eq_zero, mem_biUnion, exists_prop, mem_filter, id,
      and_assoc, sdiff_eq_empty_iff_subset, subset_iff]
    exact fun x hx a ha =>
      ⟨{a}, mem_map_of_mem _ (P.le hx ha), singleton_subset_iff.2 ha, mem_singleton_self _⟩
  -- Prove the case `m > 0` by strong induction on `s`
  /-
    case inr
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    m_pos : GT.gt m 0
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  induction' s using Finset.strongInduction with s ih generalizing a b
  -- If `a = b = 0`, then `s = ∅` and we can partition into zero parts
  /-
    case inr.H
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  by_cases hab : a = 0 ∧ b = 0
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : And (Eq a 0) (Eq b 0)
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
  · simp only [hab.1, hab.2, add_zero, zero_mul, eq_comm, card_eq_zero, Finset.bot_eq_empty] at hs
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hab : And (Eq a 0) (Eq b 0)
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
    subst hs
    -- Porting note: to synthesize `Finpartition ∅`, `have` is required
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      a b : Nat
      hab : And (Eq a 0) (Eq b 0)
      ih : ∀ (t : Finset α), HasSSubset.SSubset t EmptyCollection.emptyCollection →  …
      P : Finpartition EmptyCollection.emptyCollection
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
    have : P = Finpartition.empty _ := Unique.eq_default (α := Finpartition ⊥) P
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      a b : Nat
      hab : And (Eq a 0) (Eq b 0)
      ih : ∀ (t : Finset α), HasSSubset.SSubset t EmptyCollection.emptyCollection →  …
      P : Finpartition EmptyCollection.emptyCollection
      this : Eq P (Finpartition.empty (Finset α))
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
    exact ⟨Finpartition.empty _, by simp, by simp [this], by simp [hab.2]⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Not (And (Eq a 0) (Eq b 0))
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  simp_rw [not_and_or, ← Ne.eq_def, ← pos_iff_ne_zero] at hab
  -- `n` will be the size of the smallest part
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  set n := if 0 < a then m else m + 1 with hn
  -- Some easy facts about it
  obtain ⟨hn₀, hn₁, hn₂, hn₃⟩ : 0 < n ∧ n ≤ m + 1 ∧ n ≤ a * m + b * (m + 1) ∧
      ite (0 < a) (a - 1) a * m + ite (0 < a) b (b - 1) * (m + 1) = #s - n := by
    rw [hn, ← hs]
    split_ifs with h <;> rw [tsub_mul, one_mul]
    · refine ⟨m_pos, le_succ _, le_add_right (Nat.le_mul_of_pos_left _ ‹0 < a›), ?_⟩
      rw [tsub_add_eq_add_tsub (Nat.le_mul_of_pos_left _ h)]
    · refine ⟨succ_pos', le_rfl,
        le_add_left (Nat.le_mul_of_pos_left _ <| hab.resolve_left ‹¬0 < a›), ?_⟩
      rw [← add_tsub_assoc_of_le (Nat.le_mul_of_pos_left _ <| hab.resolve_left ‹¬0 < a›)]
  /- We will call the inductive hypothesis on a partition of `s \ t` for a carefully chosen `t ⊆ s`.
    To decide which, however, we must distinguish the case where all parts of `P` have size `m` (in
    which case we take `t` to be an arbitrary subset of `s` of size `n`) from the case where at
    least one part `u` of `P` has size `m + 1` (in which case we take `t` to be an arbitrary subset
    of `u` of size `n`). The rest of each branch is just tedious calculations to satisfy the
    induction hypothesis. -/
  /-
    case neg.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  by_cases h : ∀ u ∈ P.parts, #u < m + 1
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
  · obtain ⟨t, hts, htn⟩ := exists_subset_card_eq (hn₂.trans_eq hs)
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      t : Finset α
      hts : HasSubset.Subset t s
      htn : Eq t.card n
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
    have ht : t.Nonempty := by rwa [← card_pos, htn]
    have hcard : ite (0 < a) (a - 1) a * m + ite (0 < a) b (b - 1) * (m + 1) = #(s \ t) := by
      rw [card_sdiff ‹t ⊆ s›, htn, hn₃]
    obtain ⟨R, hR₁, _, hR₃⟩ :=
      @ih (s \ t) (sdiff_ssubset hts ‹t.Nonempty›) (if 0 < a then a - 1 else a)
        (if 0 < a then b else b - 1) (P.avoid t) hcard
    /-
      case pos.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      t : Finset α
      hts : HasSubset.Subset t s
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
    -/
    refine ⟨R.extend ht.ne_empty sdiff_disjoint (sdiff_sup_cancel hts), ?_, ?_, ?_⟩
      /-
        case pos.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ⊢ ∀ (x : Finset α), Membership.mem (R.extend ⋯ ⋯ ⋯).parts x → Or (Eq x.card m) …
      -/
    · simp only [extend_parts, mem_insert, forall_eq_or_imp, and_iff_left hR₁, htn, hn]
      /-
        case pos.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ⊢ Or (Eq (ite (LT.lt 0 a) m (HAdd.hAdd m 1)) m) (Eq (ite (LT.lt 0 a) m (HAdd.h …
      -/
      exact ite_eq_or_eq _ _ _
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ⊢ ∀ (x : Finset α), Membership.mem P.parts x → LE.le (SDiff.sdiff x ((Finset.f …
      -/
    · exact fun x hx => (card_le_card sdiff_subset).trans (Nat.lt_succ_iff.1 <| h _ hx)
      /-
        🎉 no goals
      -/
    /-
      case pos.intro.intro.intro.intro.intro.refine_3
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      t : Finset α
      hts : HasSubset.Subset t s
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) (R.extend ⋯ ⋯ ⋯).part …
    -/
    simp_rw [extend_parts, filter_insert, htn, n, m.succ_ne_self.symm.ite_eq_right_iff]
    /-
      case pos.intro.intro.intro.intro.intro.refine_3
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      t : Finset α
      hts : HasSubset.Subset t s
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ Eq (ite (Not (LT.lt 0 a)) (Insert.insert t (Finset.filter (fun i => Eq i.car …
    -/
    split_ifs with ha
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ha : LT.lt 0 a
        ⊢ Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card b
      -/
    · rw [hR₃, if_pos ha]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
      t : Finset α
      hts : HasSubset.Subset t s
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ha : Not (LT.lt 0 a)
      ⊢ Eq (Insert.insert t (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.pa …
    -/
    rw [card_insert_of_not_mem, hR₃, if_neg ha, tsub_add_cancel_of_le]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ha : Not (LT.lt 0 a)
        ⊢ LE.le 1 b
      -/
    · exact hab.resolve_left ha
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        h : ∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd m 1)
        t : Finset α
        hts : HasSubset.Subset t s
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        left✝ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sd …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ha : Not (LT.lt 0 a)
        ⊢ Not (Membership.mem (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.pa …
      -/
    · intro H; exact ht.ne_empty (le_sdiff_iff.1 <| R.le <| filter_subset _ _ H)
               /-
                 🎉 no goals
               -/
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    h : Not (∀ (u : Finset α), Membership.mem P.parts u → LT.lt u.card (HAdd.hAdd  …
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  push_neg at h
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    h : Exists fun u => And (Membership.mem P.parts u) (LE.le (HAdd.hAdd m 1) u.ca …
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  obtain ⟨u, hu₁, hu₂⟩ := h
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    u : Finset α
    hu₁ : Membership.mem P.parts u
    hu₂ : LE.le (HAdd.hAdd m 1) u.card
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  obtain ⟨t, htu, htn⟩ := exists_subset_card_eq (hn₁.trans hu₂)
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    u : Finset α
    hu₁ : Membership.mem P.parts u
    hu₂ : LE.le (HAdd.hAdd m 1) u.card
    t : Finset α
    htu : HasSubset.Subset t u
    htn : Eq t.card n
    ⊢ Exists fun Q => And (∀ (x : Finset α), Membership.mem Q.parts x → Or (Eq x.c …
  -/
  have ht : t.Nonempty := by rwa [← card_pos, htn]
  have hcard : ite (0 < a) (a - 1) a * m + ite (0 < a) b (b - 1) * (m + 1) = #(s \ t) := by
    rw [card_sdiff (htu.trans <| P.le hu₁), htn, hn₃]
  obtain ⟨R, hR₁, hR₂, hR₃⟩ :=
    @ih (s \ t) (sdiff_ssubset (htu.trans <| P.le hu₁) ht) (if 0 < a then a - 1 else a)
      (if 0 < a then b else b - 1) (P.avoid t) hcard
  refine
    ⟨R.extend ht.ne_empty sdiff_disjoint (sdiff_sup_cancel <| htu.trans <| P.le hu₁), ?_, ?_, ?_⟩
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ ∀ (x : Finset α), Membership.mem (R.extend ⋯ ⋯ ⋯).parts x → Or (Eq x.card m) …
    -/
  · simp only [mem_insert, forall_eq_or_imp, extend_parts, and_iff_left hR₁, htn, hn]
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ Or (Eq (ite (LT.lt 0 a) m (HAdd.hAdd m 1)) m) (Eq (ite (LT.lt 0 a) m (HAdd.h …
    -/
    exact ite_eq_or_eq _ _ _
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ ∀ (x : Finset α), Membership.mem P.parts x → LE.le (SDiff.sdiff x ((Finset.f …
    -/
  · conv in _ ∈ _ => rw [← insert_erase hu₁]
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ ∀ (x : Finset α), Membership.mem (Insert.insert u (P.parts.erase u)) x → LE. …
    -/
    simp only [and_imp, mem_insert, forall_eq_or_imp, Ne, extend_parts]
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      ⊢ And (LE.le (SDiff.sdiff u ((Finset.filter (fun y => HasSubset.Subset y u) (I …
    -/
    refine ⟨?_, fun x hx => (card_le_card ?_).trans <| hR₂ x ?_⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        u : Finset α
        hu₁ : Membership.mem P.parts u
        hu₂ : LE.le (HAdd.hAdd m 1) u.card
        t : Finset α
        htu : HasSubset.Subset t u
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ⊢ LE.le (SDiff.sdiff u ((Finset.filter (fun y => HasSubset.Subset y u) (Insert …
      -/
    · simp only [filter_insert, if_pos htu, biUnion_insert, mem_erase, id]
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        u : Finset α
        hu₁ : Membership.mem P.parts u
        hu₂ : LE.le (HAdd.hAdd m 1) u.card
        t : Finset α
        htu : HasSubset.Subset t u
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        ⊢ LE.le (SDiff.sdiff u (Union.union t ((Finset.filter (fun y => HasSubset.Subs …
      -/
      obtain rfl | hut := eq_or_ne u t
        /-
          case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1.inl
          α : Type u_1
          inst✝ : DecidableEq α
          m : Nat
          m_pos : GT.gt m 0
          s : Finset α
          ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
          a b : Nat
          P : Finpartition s
          hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
          hab : Or (LT.lt 0 a) (LT.lt 0 b)
          n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
          hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
          hn₀ : LT.lt 0 n
          hn₁ : LE.le n (HAdd.hAdd m 1)
          hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
          hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
          u : Finset α
          hu₁ : Membership.mem P.parts u
          hu₂ : LE.le (HAdd.hAdd m 1) u.card
          htu : HasSubset.Subset u u
          htn : Eq u.card n
          ht : u.Nonempty
          hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
          R : Finpartition (SDiff.sdiff s u)
          hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
          hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid u).parts x → LE.le (SDiff.sdif …
          hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
          ⊢ LE.le (SDiff.sdiff u (Union.union u ((Finset.filter (fun y => HasSubset.Subs …
        -/
      · rw [sdiff_eq_empty_iff_subset.2 subset_union_left]
        /-
          case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1.inl
          α : Type u_1
          inst✝ : DecidableEq α
          m : Nat
          m_pos : GT.gt m 0
          s : Finset α
          ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
          a b : Nat
          P : Finpartition s
          hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
          hab : Or (LT.lt 0 a) (LT.lt 0 b)
          n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
          hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
          hn₀ : LT.lt 0 n
          hn₁ : LE.le n (HAdd.hAdd m 1)
          hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
          hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
          u : Finset α
          hu₁ : Membership.mem P.parts u
          hu₂ : LE.le (HAdd.hAdd m 1) u.card
          htu : HasSubset.Subset u u
          htn : Eq u.card n
          ht : u.Nonempty
          hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
          R : Finpartition (SDiff.sdiff s u)
          hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
          hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid u).parts x → LE.le (SDiff.sdif …
          hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
          ⊢ LE.le EmptyCollection.emptyCollection.card m
        -/
        exact bot_le
        /-
          🎉 no goals
        -/
      refine
        (card_le_card fun i => ?_).trans
          (hR₂ (u \ t) <| P.mem_avoid.2 ⟨u, hu₁, fun i => hut <| i.antisymm htu, rfl⟩)
      -- Porting note: `not_and` required because `∃ x ∈ s, p x` is defined differently
      simp only [not_exists, not_and, mem_biUnion, and_imp, mem_union, mem_filter, mem_sdiff,
        id, not_or]
      exact fun hi₁ hi₂ hi₃ =>
        ⟨⟨hi₁, hi₂⟩, fun x hx hx' => hi₃ _ hx <| hx'.trans sdiff_subset⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_2
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        u : Finset α
        hu₁ : Membership.mem P.parts u
        hu₂ : LE.le (HAdd.hAdd m 1) u.card
        t : Finset α
        htu : HasSubset.Subset t u
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        x : Finset α
        hx : Membership.mem (P.parts.erase u) x
        ⊢ HasSubset.Subset (SDiff.sdiff x ((Finset.filter (fun y => HasSubset.Subset y …
      -/
    · apply sdiff_subset_sdiff Subset.rfl (biUnion_subset_biUnion_of_subset_left _ _)
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m : Nat
        m_pos : GT.gt m 0
        s : Finset α
        ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
        a b : Nat
        P : Finpartition s
        hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
        hab : Or (LT.lt 0 a) (LT.lt 0 b)
        n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
        hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
        hn₀ : LT.lt 0 n
        hn₁ : LE.le n (HAdd.hAdd m 1)
        hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
        hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
        u : Finset α
        hu₁ : Membership.mem P.parts u
        hu₂ : LE.le (HAdd.hAdd m 1) u.card
        t : Finset α
        htu : HasSubset.Subset t u
        htn : Eq t.card n
        ht : t.Nonempty
        hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
        R : Finpartition (SDiff.sdiff s t)
        hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
        hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
        hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
        x : Finset α
        hx : Membership.mem (P.parts.erase u) x
        ⊢ HasSubset.Subset (Finset.filter (fun y => HasSubset.Subset y x) R.parts) (Fi …
      -/
      exact filter_subset_filter _ (subset_insert _ _)
      /-
        🎉 no goals
      -/
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_3
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      x : Finset α
      hx : Membership.mem (P.parts.erase u) x
      ⊢ Membership.mem (P.avoid t).parts x
    -/
    simp only [avoid, ofErase, mem_erase, mem_image, bot_eq_empty]
    exact
      ⟨(nonempty_of_mem_parts _ <| mem_of_mem_erase hx).ne_empty, _, mem_of_mem_erase hx,
        (disjoint_of_subset_right htu <|
            P.disjoint (mem_of_mem_erase hx) hu₁ <| ne_of_mem_erase hx).sdiff_eq_left⟩
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    u : Finset α
    hu₁ : Membership.mem P.parts u
    hu₂ : LE.le (HAdd.hAdd m 1) u.card
    t : Finset α
    htu : HasSubset.Subset t u
    htn : Eq t.card n
    ht : t.Nonempty
    hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
    R : Finpartition (SDiff.sdiff s t)
    hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
    hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
    hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
    ⊢ Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) (R.extend ⋯ ⋯ ⋯).part …
  -/
  simp only [extend_parts, filter_insert, htn, hn, m.succ_ne_self.symm.ite_eq_right_iff]
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    m : Nat
    m_pos : GT.gt m 0
    s : Finset α
    ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
    a b : Nat
    P : Finpartition s
    hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hab : Or (LT.lt 0 a) (LT.lt 0 b)
    n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
    hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
    hn₀ : LT.lt 0 n
    hn₁ : LE.le n (HAdd.hAdd m 1)
    hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
    hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
    u : Finset α
    hu₁ : Membership.mem P.parts u
    hu₂ : LE.le (HAdd.hAdd m 1) u.card
    t : Finset α
    htu : HasSubset.Subset t u
    htn : Eq t.card n
    ht : t.Nonempty
    hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
    R : Finpartition (SDiff.sdiff s t)
    hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
    hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
    hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
    ⊢ Eq (ite (Not (LT.lt 0 a)) (Insert.insert t (Finset.filter (fun i => Eq i.car …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      h : LT.lt 0 a
      ⊢ Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card b
    -/
  · rw [hR₃, if_pos h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      h : Not (LT.lt 0 a)
      ⊢ Eq (Insert.insert t (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.pa …
    -/
  · rw [card_insert_of_not_mem, hR₃, if_neg h, Nat.sub_add_cancel (hab.resolve_left h)]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      m : Nat
      m_pos : GT.gt m 0
      s : Finset α
      ih : ∀ (t : Finset α), HasSSubset.SSubset t s → ∀ {a b : Nat} {P : Finpartitio …
      a b : Nat
      P : Finpartition s
      hs : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
      hab : Or (LT.lt 0 a) (LT.lt 0 b)
      n : Nat := ite (LT.lt 0 a) m (HAdd.hAdd m 1)
      hn : Eq n (ite (LT.lt 0 a) m (HAdd.hAdd m 1))
      hn₀ : LT.lt 0 n
      hn₁ : LE.le n (HAdd.hAdd m 1)
      hn₂ : LE.le n (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1)))
      hn₃ : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul.hM …
      u : Finset α
      hu₁ : Membership.mem P.parts u
      hu₂ : LE.le (HAdd.hAdd m 1) u.card
      t : Finset α
      htu : HasSubset.Subset t u
      htn : Eq t.card n
      ht : t.Nonempty
      hcard : Eq (HAdd.hAdd (HMul.hMul (ite (LT.lt 0 a) (HSub.hSub a 1) a) m) (HMul. …
      R : Finpartition (SDiff.sdiff s t)
      hR₁ : ∀ (x : Finset α), Membership.mem R.parts x → Or (Eq x.card m) (Eq x.card …
      hR₂ : ∀ (x : Finset α), Membership.mem (P.avoid t).parts x → LE.le (SDiff.sdif …
      hR₃ : Eq (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.parts).card (it …
      h : Not (LT.lt 0 a)
      ⊢ Not (Membership.mem (Finset.filter (fun i => Eq i.card (HAdd.hAdd m 1)) R.pa …
    -/
    intro H; exact ht.ne_empty (le_sdiff_iff.1 <| R.le <| filter_subset _ _ H)
             /-
               🎉 no goals
             -/


/-- Given a partition `P` of `s`, as well as a proof that `a * m + b * (m + 1) = #s`, build a
new partition `Q` of `s` where each part has size `m` or `m + 1`, every part of `P` is the union of
parts of `Q` plus at most `m` extra elements, there are `b` parts of size `m + 1` and (provided
`m > 0`, because a partition does not have parts of size `0`) there are `a` parts of size `m` and
hence `a + b` parts in total. -/
noncomputable def equitabilise : Finpartition s :=
  (P.equitabilise_aux h).choose


theorem card_eq_of_mem_parts_equitabilise :
    t ∈ (P.equitabilise h).parts → #t = m ∨ #t = m + 1 :=
  (P.equitabilise_aux h).choose_spec.1 _


theorem equitabilise_isEquipartition : (P.equitabilise h).IsEquipartition :=
  Set.equitableOn_iff_exists_eq_eq_add_one.2 ⟨m, fun _ => card_eq_of_mem_parts_equitabilise⟩


theorem card_filter_equitabilise_big : #{u ∈ (P.equitabilise h).parts | #u = m + 1} = b :=
  (P.equitabilise_aux h).choose_spec.2.2


theorem card_filter_equitabilise_small (hm : m ≠ 0) :
    #{u ∈ (P.equitabilise h).parts | #u = m} = a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    ⊢ Eq (Finset.filter (fun u => Eq u.card m) (Finpartition.equitabilise h).parts …
  -/
  refine (mul_eq_mul_right_iff.1 <| (add_left_inj (b * (m + 1))).1 ?_).resolve_right hm
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Finset.filter (fun u => Eq u.card m) (Finpartition …
  -/
  rw [h, ← (P.equitabilise h).sum_card_parts]
  have hunion :
    (P.equitabilise h).parts =
      {u ∈ (P.equitabilise h).parts | #u = m} ∪ {u ∈ (P.equitabilise h).parts | #u = m + 1} := by
    rw [← filter_or, filter_true_of_mem]
    exact fun x => card_eq_of_mem_parts_equitabilise
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    hunion : Eq (Finpartition.equitabilise h).parts (Union.union (Finset.filter (f …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Finset.filter (fun u => Eq u.card m) (Finpartition …
  -/
  nth_rw 2 [hunion]
  rw [sum_union, sum_const_nat fun x hx => (mem_filter.1 hx).2,
    sum_const_nat fun x hx => (mem_filter.1 hx).2, P.card_filter_equitabilise_big]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    hunion : Eq (Finpartition.equitabilise h).parts (Union.union (Finset.filter (f …
    ⊢ Disjoint (Finset.filter (fun u => Eq u.card m) (Finpartition.equitabilise h) …
  -/
  refine disjoint_filter_filter' _ _ ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    hunion : Eq (Finpartition.equitabilise h).parts (Union.union (Finset.filter (f …
    ⊢ Disjoint (fun u => Eq u.card m) fun u => Eq u.card (HAdd.hAdd m 1)
  -/
  intro x ha hb i h
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h✝ : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    hunion : Eq (Finpartition.equitabilise h✝).parts (Union.union (Finset.filter ( …
    x : Finset α → Prop
    ha : LE.le x fun u => Eq u.card m
    hb : LE.le x fun u => Eq u.card (HAdd.hAdd m 1)
    i : Finset α
    h : x i
    ⊢ Bot.bot i
  -/
  apply succ_ne_self m _
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h✝ : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    hunion : Eq (Finpartition.equitabilise h✝).parts (Union.union (Finset.filter ( …
    x : Finset α → Prop
    ha : LE.le x fun u => Eq u.card m
    hb : LE.le x fun u => Eq u.card (HAdd.hAdd m 1)
    i : Finset α
    h : x i
    ⊢ Eq m.succ m
  -/
  exact (hb i h).symm.trans (ha i h)
  /-
    🎉 no goals
  -/


theorem card_parts_equitabilise (hm : m ≠ 0) : #(P.equitabilise h).parts = a + b := by
  rw [← filter_true_of_mem fun x => card_eq_of_mem_parts_equitabilise, filter_or,
    card_union_of_disjoint, P.card_filter_equitabilise_small _ hm, P.card_filter_equitabilise_big]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11187): was `infer_instance`
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    m a b : Nat
    P : Finpartition s
    h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b (HAdd.hAdd m 1))) s.card
    hm : Ne m 0
    ⊢ Disjoint (Finset.filter (fun x => Eq x.card m) (Finpartition.equitabilise h) …
  -/
  exact disjoint_filter.2 fun x _ h₀ h₁ => Nat.succ_ne_self m <| h₁.symm.trans h₀
  /-
    🎉 no goals
  -/


theorem card_parts_equitabilise_subset_le :
    t ∈ P.parts → #(t \ {u ∈ (P.equitabilise h).parts | u ⊆ t}.biUnion id) ≤ m :=
  (Classical.choose_spec <| P.equitabilise_aux h).2.1 t


/-- We can find equipartitions of arbitrary size. -/
theorem exists_equipartition_card_eq (hn : n ≠ 0) (hs : n ≤ #s) :
    ∃ P : Finpartition s, P.IsEquipartition ∧ #P.parts = n := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    n : Nat
    hn : Ne n 0
    hs : LE.le n s.card
    ⊢ Exists fun P => And P.IsEquipartition (Eq P.parts.card n)
  -/
  rw [← pos_iff_ne_zero] at hn
  have : (n - #s % n) * (#s / n) + #s % n * (#s / n + 1) = #s := by
    rw [tsub_mul, mul_add, ← add_assoc,
      tsub_add_cancel_of_le (Nat.mul_le_mul_right _ (mod_lt _ hn).le), mul_one, add_comm,
      mod_add_div]
  refine
    ⟨(indiscrete (card_pos.1 <| hn.trans_le hs).ne_empty).equitabilise this,
      equitabilise_isEquipartition, ?_⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    n : Nat
    hn : LT.lt 0 n
    hs : LE.le n s.card
    this : Eq (HAdd.hAdd (HMul.hMul (HSub.hSub n (HMod.hMod s.card n)) (HDiv.hDiv  …
    ⊢ Eq (Finpartition.equitabilise this).parts.card n
  -/
  rw [card_parts_equitabilise _ _ (Nat.div_pos hs hn).ne', tsub_add_cancel_of_le (mod_lt _ hn).le]
  /-
    🎉 no goals
  -/


