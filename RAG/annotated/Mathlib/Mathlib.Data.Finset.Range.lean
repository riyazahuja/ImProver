/-- `range n` is the set of natural numbers less than `n`. -/
def range (n : ℕ) : Finset ℕ :=
  ⟨_, nodup_range n⟩


@[simp]
theorem range_val (n : ℕ) : (range n).1 = Multiset.range n :=
  rfl


@[simp]
theorem mem_range : m ∈ range n ↔ m < n :=
  Multiset.mem_range


@[simp, norm_cast]
theorem coe_range (n : ℕ) : (range n : Set ℕ) = Set.Iio n :=
  Set.ext fun _ => mem_range


@[simp]
theorem range_zero : range 0 = ∅ :=
  rfl


@[simp]
theorem range_one : range 1 = {0} :=
  rfl


theorem range_succ : range (succ n) = insert n (range n) :=
  eq_of_veq <| (Multiset.range_succ n).trans <| (ndinsert_of_not_mem not_mem_range_self).symm


theorem range_add_one : range (n + 1) = insert n (range n) :=
  range_succ


theorem not_mem_range_self : n ∉ range n :=
  Multiset.not_mem_range_self


theorem self_mem_range_succ (n : ℕ) : n ∈ range (n + 1) :=
  Multiset.self_mem_range_succ n


@[simp]
theorem range_subset {n m} : range n ⊆ range m ↔ n ≤ m :=
  Multiset.range_subset


theorem range_mono : Monotone range := fun _ _ => range_subset.2


@[gcongr] alias ⟨_, _root_.GCongr.finset_range_subset_of_le⟩ := range_subset


theorem mem_range_succ_iff {a b : ℕ} : a ∈ Finset.range b.succ ↔ a ≤ b :=
  Finset.mem_range.trans Nat.lt_succ_iff


theorem mem_range_le {n x : ℕ} (hx : x ∈ range n) : x ≤ n :=
  (mem_range.1 hx).le


theorem mem_range_sub_ne_zero {n x : ℕ} (hx : x ∈ range n) : n - x ≠ 0 :=
  _root_.ne_of_gt <| Nat.sub_pos_of_lt <| mem_range.1 hx


@[simp]
theorem nonempty_range_iff : (range n).Nonempty ↔ n ≠ 0 :=
  ⟨fun ⟨k, hk⟩ => (k.zero_le.trans_lt <| mem_range.1 hk).ne',
   fun h => ⟨0, mem_range.2 <| Nat.pos_iff_ne_zero.2 h⟩⟩


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected alias ⟨_, Aesop.range_nonempty⟩ := nonempty_range_iff


@[simp]
theorem range_eq_empty_iff : range n = ∅ ↔ n = 0 := by
  /-
    n : Nat
    ⊢ Iff (Eq (Finset.range n) EmptyCollection.emptyCollection) (Eq n 0)
  -/
  rw [← not_nonempty_iff_eq_empty, nonempty_range_iff, not_not]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
theorem nonempty_range_succ : (range <| n + 1).Nonempty :=
  nonempty_range_iff.2 n.succ_ne_zero


lemma range_nontrivial {n : ℕ} (hn : 1 < n) : (Finset.range n).Nontrivial := by
  /-
    n : Nat
    hn : LT.lt 1 n
    ⊢ (Finset.range n).Nontrivial
  -/
  rw [Finset.Nontrivial, Finset.coe_range]
  /-
    n : Nat
    hn : LT.lt 1 n
    ⊢ (Set.Iio n).Nontrivial
  -/
  exact ⟨0, Nat.zero_lt_one.trans hn, 1, hn, Nat.zero_ne_one⟩
  /-
    🎉 no goals
  -/


theorem exists_nat_subset_range (s : Finset ℕ) : ∃ n : ℕ, s ⊆ range n :=
                     /-
                       s : Finset Nat
                       ⊢ Exists fun n => HasSubset.Subset EmptyCollection.emptyCollection (Finset.ran …
                     -/
  s.induction_on (by simp)
                     /-
                       🎉 no goals
                     -/
                                                           /-
                                                             s : Finset Nat
                                                             a : Nat
                                                             x✝² : Finset Nat
                                                             x✝¹ : Not (Membership.mem x✝² a)
                                                             x✝ : Exists fun n => HasSubset.Subset x✝² (Finset.range n)
                                                             n : Nat
                                                             hn : HasSubset.Subset x✝² (Finset.range n)
                                                             ⊢ Membership.mem (Finset.range (Max.max (HAdd.hAdd a 1) n)) a
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    fun a _ _ ⟨n, hn⟩ => ⟨max (a + 1) n, insert_subset (by simp) (hn.trans (by simp))⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Equivalence between the set of natural numbers which are `≥ k` and `ℕ`, given by `n → n - k`. -/
def notMemRangeEquiv (k : ℕ) : { n // n ∉ range k } ≃ ℕ where
  toFun i := i.1 - k
                         /-
                           α : Type u_1
                           β : Type u_2
                           γ : Type u_3
                           k j : Nat
                           ⊢ Not (Membership.mem (Multiset.range k) (HAdd.hAdd j k))
                         -/
  invFun j := ⟨j + k, by simp⟩
                         /-
                           🎉 no goals
                         -/
  left_inv j := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      k : Nat
      j : Subtype fun n => Not (Membership.mem (Multiset.range k) n)
      ⊢ Eq ((fun j => ⟨HAdd.hAdd j k, ⋯⟩) ((fun i => HSub.hSub (↑i) k) j)) j
    -/
    rw [Subtype.ext_iff_val]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      k : Nat
      j : Subtype fun n => Not (Membership.mem (Multiset.range k) n)
      ⊢ Eq ↑((fun j => ⟨HAdd.hAdd j k, ⋯⟩) ((fun i => HSub.hSub (↑i) k) j)) ↑j
    -/
    apply Nat.sub_add_cancel
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      k : Nat
      j : Subtype fun n => Not (Membership.mem (Multiset.range k) n)
      ⊢ LE.le k ↑j
    -/
    simpa using j.2
    /-
      🎉 no goals
    -/
  right_inv _ := Nat.add_sub_cancel_right _ _


@[simp]
theorem coe_notMemRangeEquiv (k : ℕ) :
    (notMemRangeEquiv k : { n // n ∉ range k } → ℕ) = fun (i : { n // n ∉ range k }) => i - k :=
  rfl


@[simp]
theorem coe_notMemRangeEquiv_symm (k : ℕ) :
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   β : Type u_2
                                                                                   γ : Type u_3
                                                                                   k j : Nat
                                                                                   ⊢ Not (Membership.mem (Multiset.range k) (HAdd.hAdd j k))
                                                                                 -/
    ((notMemRangeEquiv k).symm : ℕ → { n // n ∉ range k }) = fun j => ⟨j + k, by simp⟩ :=
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  rfl

