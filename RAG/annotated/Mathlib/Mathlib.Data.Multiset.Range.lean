/-- `range n` is the multiset lifted from the list `range n`,
  that is, the set `{0, 1, ..., n-1}`. -/
def range (n : ℕ) : Multiset ℕ :=
  List.range n


theorem coe_range (n : ℕ) : ↑(List.range n) = range n :=
  rfl


@[simp]
theorem range_zero : range 0 = 0 :=
  rfl


@[simp]
theorem range_succ (n : ℕ) : range (succ n) = n ::ₘ range n := by
  /-
    n : Nat
    ⊢ Eq (Multiset.range n.succ) (Multiset.cons n (Multiset.range n))
  -/
  rw [range, List.range_succ, ← coe_add, add_comm]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem card_range (n : ℕ) : card (range n) = n :=
  length_range _


theorem range_subset {m n : ℕ} : range m ⊆ range n ↔ m ≤ n :=
  List.range_subset


@[simp]
theorem mem_range {m n : ℕ} : m ∈ range n ↔ m < n :=
  List.mem_range


theorem not_mem_range_self {n : ℕ} : n ∉ range n :=
  List.not_mem_range_self


theorem self_mem_range_succ (n : ℕ) : n ∈ range (n + 1) :=
  List.self_mem_range_succ n


theorem range_add (a b : ℕ) : range (a + b) = range a + (range b).map (a + ·) :=
  congr_arg ((↑) : List ℕ → Multiset ℕ) (List.range_add _ _)


theorem range_disjoint_map_add (a : ℕ) (m : Multiset ℕ) :
    Disjoint (range a) (m.map (a + ·)) := by
  /-
    a : Nat
    m : Multiset Nat
    ⊢ Disjoint (Multiset.range a) (Multiset.map (fun x => HAdd.hAdd a x) m)
  -/
  rw [disjoint_left]
  /-
    a : Nat
    m : Multiset Nat
    ⊢ ∀ {a_1 : Nat}, Membership.mem (Multiset.range a) a_1 → Not (Membership.mem ( …
  -/
  intro x hxa hxb
  /-
    a : Nat
    m : Multiset Nat
    x : Nat
    hxa : Membership.mem (Multiset.range a) x
    hxb : Membership.mem (Multiset.map (fun x => HAdd.hAdd a x) m) x
    ⊢ False
  -/
  rw [range, mem_coe, List.mem_range] at hxa
  /-
    a : Nat
    m : Multiset Nat
    x : Nat
    hxa : LT.lt x a
    hxb : Membership.mem (Multiset.map (fun x => HAdd.hAdd a x) m) x
    ⊢ False
  -/
  obtain ⟨c, _, rfl⟩ := mem_map.1 hxb
  /-
    case intro.intro
    a : Nat
    m : Multiset Nat
    c : Nat
    left✝ : Membership.mem m c
    hxa : LT.lt (HAdd.hAdd a c) a
    hxb : Membership.mem (Multiset.map (fun x => HAdd.hAdd a x) m) (HAdd.hAdd a c)
    ⊢ False
  -/
  exact (Nat.le_add_right _ _).not_lt hxa
  /-
    🎉 no goals
  -/


theorem range_add_eq_union (a b : ℕ) : range (a + b) = range a ∪ (range b).map (a + ·) := by
  /-
    a b : Nat
    ⊢ Eq (Multiset.range (HAdd.hAdd a b)) (Union.union (Multiset.range a) (Multise …
  -/
  rw [range_add, add_eq_union_iff_disjoint]
  /-
    a b : Nat
    ⊢ Disjoint (Multiset.range a) (Multiset.map (fun x => HAdd.hAdd a x) (Multiset …
  -/
  apply range_disjoint_map_add
  /-
    🎉 no goals
  -/


