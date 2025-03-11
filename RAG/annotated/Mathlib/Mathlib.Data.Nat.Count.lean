/-- Count the number of naturals `k < n` satisfying `p k`. -/
def count (n : ℕ) : ℕ :=
  (List.range n).countP p


@[simp]
theorem count_zero : count p 0 = 0 := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Nat.count p 0) 0
  -/
  rw [count, List.range_zero, List.countP, List.countP.go]
  /-
    🎉 no goals
  -/


/-- A fintype instance for the set relevant to `Nat.count`. Locally an instance in locale `count` -/
def CountSet.fintype (n : ℕ) : Fintype { i // i < n ∧ p i } := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Fintype (Subtype fun i => And (LT.lt i n) (p i))
  -/
  apply Fintype.ofFinset {x ∈ range n | p x}
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ ∀ (x : Nat), Iff (Membership.mem (Finset.filter (fun x => p x) (Finset.range …
  -/
  intro x
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n x : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun x => p x) (Finset.range n)) x) (Memb …
  -/
  rw [mem_filter, mem_range]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n x : Nat
    ⊢ Iff (And (LT.lt x n) (p x)) (Membership.mem (fun x => And (LT.lt x n) (p x)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem count_eq_card_filter_range (n : ℕ) : count p n = #{x ∈ range n | p x} := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (Nat.count p n) (Finset.filter (fun x => p x) (Finset.range n)).card
  -/
  rw [count, List.countP_eq_length_filter]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (List.filter (fun b => Decidable.decide (p b)) (List.range n)).length (Fi …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `count p n` can be expressed as the cardinality of `{k // k < n ∧ p k}`. -/
theorem count_eq_card_fintype (n : ℕ) : count p n = Fintype.card { k : ℕ // k < n ∧ p k } := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (Nat.count p n) (Fintype.card (Subtype fun k => And (LT.lt k n) (p k)))
  -/
  rw [count_eq_card_filter_range, ← Fintype.card_ofFinset, ← CountSet.fintype]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (Fintype.card ↑fun x => And (LT.lt x n) (p x)) (Fintype.card (Subtype fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem count_le {n : ℕ} : count p n ≤ n := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ LE.le (Nat.count p n) n
  -/
  rw [count_eq_card_filter_range]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ LE.le (Finset.filter (fun x => p x) (Finset.range n)).card n
  -/
  exact (card_filter_le _ _).trans_eq (card_range _)
  /-
    🎉 no goals
  -/


theorem count_succ (n : ℕ) : count p (n + 1) = count p n + if p n then 1 else 0 := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (Nat.count p (HAdd.hAdd n 1)) (HAdd.hAdd (Nat.count p n) (ite (p n) 1 0))
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [count, List.range_succ, h]
                       /-
                         🎉 no goals
                       -/


@[mono]
theorem count_monotone : Monotone (count p) :=
                                     /-
                                       p : Nat → Prop
                                       inst✝ : DecidablePred p
                                       n : Nat
                                       ⊢ LE.le (Nat.count p n) (Nat.count p (HAdd.hAdd n 1))
                                     -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  monotone_nat_of_le_succ fun n ↦ by by_cases h : p n <;> simp [count_succ, h]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem count_add (a b : ℕ) : count p (a + b) = count p a + count (fun k ↦ p (a + k)) b := by
  have : Disjoint {x ∈ range a | p x} {x ∈ (range b).map <| addLeftEmbedding a | p x} := by
    apply disjoint_filter_filter
    rw [Finset.disjoint_left]
    simp_rw [mem_map, mem_range, addLeftEmbedding_apply]
    rintro x hx ⟨c, _, rfl⟩
    exact (self_le_add_right _ _).not_lt hx
  simp_rw [count_eq_card_filter_range, range_add, filter_union, card_union_of_disjoint this,
    filter_map, addLeftEmbedding, card_map]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    a b : Nat
    this : Disjoint (Finset.filter (fun x => p x) (Finset.range a)) (Finset.filter …
    ⊢ Eq (HAdd.hAdd (Finset.filter (fun x => p x) (Finset.range a)).card (Finset.f …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem count_add' (a b : ℕ) : count p (a + b) = count (fun k ↦ p (k + b)) a + count p b := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    a b : Nat
    ⊢ Eq (Nat.count p (HAdd.hAdd a b)) (HAdd.hAdd (Nat.count (fun k => p (HAdd.hAd …
  -/
  rw [add_comm, count_add, add_comm]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    a b : Nat
    ⊢ Eq (HAdd.hAdd (Nat.count (fun k => p (HAdd.hAdd b k)) a) (Nat.count p b)) (H …
  -/
  simp_rw [add_comm b]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             p : Nat → Prop
                                                             inst✝ : DecidablePred p
                                                             ⊢ Eq (Nat.count p 1) (ite (p 0) 1 0)
                                                           -/
theorem count_one : count p 1 = if p 0 then 1 else 0 := by simp [count_succ]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem count_succ' (n : ℕ) :
    count p (n + 1) = count (fun k ↦ p (k + 1)) n + if p 0 then 1 else 0 := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Eq (Nat.count p (HAdd.hAdd n 1)) (HAdd.hAdd (Nat.count (fun k => p (HAdd.hAd …
  -/
  rw [count_add', count_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_lt_count_succ_iff {n : ℕ} : count p n < count p (n + 1) ↔ p n := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Iff (LT.lt (Nat.count p n) (Nat.count p (HAdd.hAdd n 1))) (p n)
  -/
                       /-
                         🎉 no goals
                       -/
  by_cases h : p n <;> simp [count_succ, h]
                       /-
                         🎉 no goals
                       -/


theorem count_succ_eq_succ_count_iff {n : ℕ} : count p (n + 1) = count p n + 1 ↔ p n := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Iff (Eq (Nat.count p (HAdd.hAdd n 1)) (HAdd.hAdd (Nat.count p n) 1)) (p n)
  -/
                       /-
                         🎉 no goals
                       -/
  by_cases h : p n <;> simp [h, count_succ]
                       /-
                         🎉 no goals
                       -/


theorem count_succ_eq_count_iff {n : ℕ} : count p (n + 1) = count p n ↔ ¬p n := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ Iff (Eq (Nat.count p (HAdd.hAdd n 1)) (Nat.count p n)) (Not (p n))
  -/
                       /-
                         🎉 no goals
                       -/
  by_cases h : p n <;> simp [h, count_succ]
                       /-
                         🎉 no goals
                       -/


alias ⟨_, count_succ_eq_succ_count⟩ := count_succ_eq_succ_count_iff


alias ⟨_, count_succ_eq_count⟩ := count_succ_eq_count_iff


theorem count_le_cardinal (n : ℕ) : (count p n : Cardinal) ≤ Cardinal.mk { k | p k } := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ LE.le (↑(Nat.count p n)) (Cardinal.mk ↑(setOf fun k => p k))
  -/
  rw [count_eq_card_fintype, ← Cardinal.mk_fintype]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    n : Nat
    ⊢ LE.le (Cardinal.mk (Subtype fun k => And (LT.lt k n) (p k))) (Cardinal.mk ↑( …
  -/
  exact Cardinal.mk_subtype_mono fun x hx ↦ hx.2
  /-
    🎉 no goals
  -/


theorem lt_of_count_lt_count {a b : ℕ} (h : count p a < count p b) : a < b :=
  (count_monotone p).reflect_lt h


theorem count_strict_mono {m n : ℕ} (hm : p m) (hmn : m < n) : count p m < count p n :=
  (count_lt_count_succ_iff.2 hm).trans_le <| count_monotone _ (Nat.succ_le_iff.2 hmn)


theorem count_injective {m n : ℕ} (hm : p m) (hn : p n) (heq : count p m = count p n) : m = n := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    m n : Nat
    hm : p m
    hn : p n
    heq : Eq (Nat.count p m) (Nat.count p n)
    ⊢ Eq m n
  -/
  by_contra! h : m ≠ n
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    m n : Nat
    hm : p m
    hn : p n
    heq : Eq (Nat.count p m) (Nat.count p n)
    h : Ne m n
    ⊢ False
  -/
  wlog hmn : m < n
    /-
      case inr
      p : Nat → Prop
      inst✝ : DecidablePred p
      m n : Nat
      hm : p m
      hn : p n
      heq : Eq (Nat.count p m) (Nat.count p n)
      h : Ne m n
      this : ∀ {p : Nat → Prop} [inst : DecidablePred p] {m n : Nat}, p m → p n → Eq …
      hmn : Not (LT.lt m n)
      ⊢ False
    -/
  · exact this hn hm heq.symm h.symm (h.lt_or_lt.resolve_left hmn)
    /-
      🎉 no goals
    -/
    /-
      p✝ : Nat → Prop
      inst✝¹ : DecidablePred p✝
      p : Nat → Prop
      inst✝ : DecidablePred p
      m n : Nat
      hm : p m
      hn : p n
      heq : Eq (Nat.count p m) (Nat.count p n)
      h : Ne m n
      hmn : LT.lt m n
      ⊢ False
    -/
  · simpa [heq] using count_strict_mono hm hmn
    /-
      🎉 no goals
    -/


theorem count_le_card (hp : (setOf p).Finite) (n : ℕ) : count p n ≤ #hp.toFinset := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    hp : (setOf p).Finite
    n : Nat
    ⊢ LE.le (Nat.count p n) hp.toFinset.card
  -/
  rw [count_eq_card_filter_range]
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    hp : (setOf p).Finite
    n : Nat
    ⊢ LE.le (Finset.filter (fun x => p x) (Finset.range n)).card hp.toFinset.card
  -/
  exact Finset.card_mono fun x hx ↦ hp.mem_toFinset.2 (mem_filter.1 hx).2
  /-
    🎉 no goals
  -/


theorem count_lt_card {n : ℕ} (hp : (setOf p).Finite) (hpn : p n) : count p n < #hp.toFinset :=
  (count_lt_count_succ_iff.2 hpn).trans_le (count_le_card hp _)


theorem count_iff_forall {n : ℕ} : count p n = n ↔ ∀ n' < n, p n' := by
  simpa [count_eq_card_filter_range, card_range, mem_range] using
    card_filter_eq_iff (p := p) (s := range n)


alias ⟨_, count_of_forall⟩ := count_iff_forall


@[simp] theorem count_true (n : ℕ) : count (fun _ ↦ True) n = n := count_of_forall fun _ _ ↦ trivial


theorem count_iff_forall_not {n : ℕ} : count p n = 0 ↔ ∀ m < n, ¬p m := by
  simpa [count_eq_card_filter_range, mem_range] using
    card_filter_eq_zero_iff (p := p) (s := range n)


alias ⟨_, count_of_forall_not⟩ := count_iff_forall_not


@[simp] theorem count_false (n : ℕ) : count (fun _ ↦ False) n = 0 :=
  count_of_forall_not fun _ _ ↦ id


theorem count_mono_left {n : ℕ} (hpq : ∀ k, p k → q k) : count p n ≤ count q n := by
  /-
    p : Nat → Prop
    inst✝¹ : DecidablePred p
    q : Nat → Prop
    inst✝ : DecidablePred q
    n : Nat
    hpq : ∀ (k : Nat), p k → q k
    ⊢ LE.le (Nat.count p n) (Nat.count q n)
  -/
  simp only [count_eq_card_filter_range]
  /-
    p : Nat → Prop
    inst✝¹ : DecidablePred p
    q : Nat → Prop
    inst✝ : DecidablePred q
    n : Nat
    hpq : ∀ (k : Nat), p k → q k
    ⊢ LE.le (Finset.filter (fun x => p x) (Finset.range n)).card (Finset.filter (f …
  -/
  exact card_le_card ((range n).monotone_filter_right hpq)
  /-
    🎉 no goals
  -/


