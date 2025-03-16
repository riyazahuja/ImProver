/-- `Ico n m` is the list of natural numbers `n ≤ x < m`.
(Ico stands for "interval, closed-open".)

See also `Mathlib/Order/Interval/Basic.lean` for modelling intervals in general preorders, as well
as sibling definitions alongside it such as `Set.Ico`, `Multiset.Ico` and `Finset.Ico`
for sets, multisets and finite sets respectively.
 -/
def Ico (n m : ℕ) : List ℕ :=
  range' n (m - n)


                                                   /-
                                                     n : Nat
                                                     ⊢ Eq (List.Ico 0 n) (List.range n)
                                                   -/
theorem zero_bot (n : ℕ) : Ico 0 n = range n := by rw [Ico, Nat.sub_zero, range_eq_range']
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem length (n m : ℕ) : length (Ico n m) = m - n := by
  /-
    n m : Nat
    ⊢ Eq (List.Ico n m).length (HSub.hSub m n)
  -/
  dsimp [Ico]
  /-
    n m : Nat
    ⊢ Eq (List.range' n (HSub.hSub m n)).length (HSub.hSub m n)
  -/
  simp [length_range']
  /-
    🎉 no goals
  -/


theorem pairwise_lt (n m : ℕ) : Pairwise (· < ·) (Ico n m) := by
  /-
    n m : Nat
    ⊢ List.Pairwise (fun x1 x2 => LT.lt x1 x2) (List.Ico n m)
  -/
  dsimp [Ico]
  /-
    n m : Nat
    ⊢ List.Pairwise (fun x1 x2 => LT.lt x1 x2) (List.range' n (HSub.hSub m n))
  -/
  simp [pairwise_lt_range']
  /-
    🎉 no goals
  -/


theorem nodup (n m : ℕ) : Nodup (Ico n m) := by
  /-
    n m : Nat
    ⊢ (List.Ico n m).Nodup
  -/
  dsimp [Ico]
  /-
    n m : Nat
    ⊢ (List.range' n (HSub.hSub m n)).Nodup
  -/
  simp [nodup_range']
  /-
    🎉 no goals
  -/


@[simp]
theorem mem {n m l : ℕ} : l ∈ Ico n m ↔ n ≤ l ∧ l < m := by
  /-
    n m l : Nat
    ⊢ Iff (Membership.mem (List.Ico n m) l) (And (LE.le n l) (LT.lt l m))
  -/
  suffices n ≤ l ∧ l < n + (m - n) ↔ n ≤ l ∧ l < m by simp [Ico, this]
  /-
    n m l : Nat
    ⊢ Iff (And (LE.le n l) (LT.lt l (HAdd.hAdd n (HSub.hSub m n)))) (And (LE.le n  …
  -/
  rcases le_total n m with hnm | hmn
    /-
      case inl
      n m l : Nat
      hnm : LE.le n m
      ⊢ Iff (And (LE.le n l) (LT.lt l (HAdd.hAdd n (HSub.hSub m n)))) (And (LE.le n  …
    -/
  · rw [Nat.add_sub_cancel' hnm]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n m l : Nat
      hmn : LE.le m n
      ⊢ Iff (And (LE.le n l) (LT.lt l (HAdd.hAdd n (HSub.hSub m n)))) (And (LE.le n  …
    -/
  · rw [Nat.sub_eq_zero_iff_le.mpr hmn, Nat.add_zero]
    exact
      and_congr_right fun hnl =>
        Iff.intro (fun hln => (not_le_of_gt hln hnl).elim) fun hlm => lt_of_lt_of_le hlm hmn


theorem eq_nil_of_le {n m : ℕ} (h : m ≤ n) : Ico n m = [] := by
  /-
    n m : Nat
    h : LE.le m n
    ⊢ Eq (List.Ico n m) List.nil
  -/
  simp [Ico, Nat.sub_eq_zero_iff_le.mpr h]
  /-
    🎉 no goals
  -/


theorem map_add (n m k : ℕ) : (Ico n m).map (k + ·) = Ico (n + k) (m + k) := by
  /-
    n m k : Nat
    ⊢ Eq (List.map (fun x => HAdd.hAdd k x) (List.Ico n m)) (List.Ico (HAdd.hAdd n …
  -/
  rw [Ico, Ico, map_add_range', Nat.add_sub_add_right m k, Nat.add_comm n k]
  /-
    🎉 no goals
  -/


theorem map_sub (n m k : ℕ) (h₁ : k ≤ n) :
    ((Ico n m).map fun x => x - k) = Ico (n - k) (m - k) := by
  /-
    n m k : Nat
    h₁ : LE.le k n
    ⊢ Eq (List.map (fun x => HSub.hSub x k) (List.Ico n m)) (List.Ico (HSub.hSub n …
  -/
  rw [Ico, Ico, Nat.sub_sub_sub_cancel_right h₁, map_sub_range' _ _ _ h₁]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_empty {n : ℕ} : Ico n n = [] :=
  eq_nil_of_le (le_refl n)


@[simp]
theorem eq_empty_iff {n m : ℕ} : Ico n m = [] ↔ m ≤ n :=
                                                      /-
                                                        n m : Nat
                                                        h : Eq (List.Ico n m) List.nil
                                                        ⊢ Eq (HSub.hSub m n) 0
                                                      -/
  Iff.intro (fun h => Nat.sub_eq_zero_iff_le.mp <| by rw [← length, h, List.length]) eq_nil_of_le
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem append_consecutive {n m l : ℕ} (hnm : n ≤ m) (hml : m ≤ l) :
    Ico n m ++ Ico m l = Ico n l := by
  /-
    n m l : Nat
    hnm : LE.le n m
    hml : LE.le m l
    ⊢ Eq (HAppend.hAppend (List.Ico n m) (List.Ico m l)) (List.Ico n l)
  -/
  dsimp only [Ico]
  /-
    n m l : Nat
    hnm : LE.le n m
    hml : LE.le m l
    ⊢ Eq (HAppend.hAppend (List.range' n (HSub.hSub m n)) (List.range' m (HSub.hSu …
  -/
  convert range'_append n (m-n) (l-m) 1 using 2
    /-
      case h.e'_2.h.e'_6
      n m l : Nat
      hnm : LE.le n m
      hml : LE.le m l
      ⊢ Eq (List.range' m (HSub.hSub l m)) (List.range' (HAdd.hAdd n (HMul.hMul 1 (H …
    -/
  · rw [Nat.one_mul, Nat.add_sub_cancel' hnm]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_2
      n m l : Nat
      hnm : LE.le n m
      hml : LE.le m l
      ⊢ Eq (HSub.hSub l n) (HAdd.hAdd (HSub.hSub l m) (HSub.hSub m n))
    -/
  · rw [Nat.sub_add_sub_cancel hml hnm]
    /-
      🎉 no goals
    -/


@[simp]
theorem inter_consecutive (n m l : ℕ) : Ico n m ∩ Ico m l = [] := by
  /-
    n m l : Nat
    ⊢ Eq (Inter.inter (List.Ico n m) (List.Ico m l)) List.nil
  -/
  apply eq_nil_iff_forall_not_mem.2
  /-
    n m l : Nat
    ⊢ ∀ (a : Nat), Not (Membership.mem (Inter.inter (List.Ico n m) (List.Ico m l)) …
  -/
  intro a
  /-
    n m l a : Nat
    ⊢ Not (Membership.mem (Inter.inter (List.Ico n m) (List.Ico m l)) a)
  -/
  simp only [and_imp, not_and, not_lt, List.mem_inter_iff, List.Ico.mem]
  /-
    n m l a : Nat
    ⊢ LE.le n a → LT.lt a m → LE.le m a → LE.le l a
  -/
  intro _ h₂ h₃
  /-
    n m l a : Nat
    a✝ : LE.le n a
    h₂ : LT.lt a m
    h₃ : LE.le m a
    ⊢ LE.le l a
  -/
  exfalso
  /-
    n m l a : Nat
    a✝ : LE.le n a
    h₂ : LT.lt a m
    h₃ : LE.le m a
    ⊢ False
  -/
  exact not_lt_of_ge h₃ h₂
  /-
    🎉 no goals
  -/


@[simp]
theorem bagInter_consecutive (n m l : Nat) :
    @List.bagInter ℕ instBEqOfDecidableEq (Ico n m) (Ico m l) = [] :=
                                         /-
                                           n m l : Nat
                                           ⊢ Eq (Inter.inter (List.Ico n m) (List.Ico m l)) List.nil
                                         -/
  (bagInter_nil_iff_inter_nil _ _).2 (by convert inter_consecutive n m l)
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem succ_singleton {n : ℕ} : Ico n (n + 1) = [n] := by
  /-
    n : Nat
    ⊢ Eq (List.Ico n (HAdd.hAdd n 1)) (List.cons n List.nil)
  -/
  dsimp [Ico]
  /-
    n : Nat
    ⊢ Eq (List.range' n (HSub.hSub (HAdd.hAdd n 1) n)) (List.cons n List.nil)
  -/
  simp [range', Nat.add_sub_cancel_left]
  /-
    🎉 no goals
  -/


theorem succ_top {n m : ℕ} (h : n ≤ m) : Ico n (m + 1) = Ico n m ++ [m] := by
  /-
    n m : Nat
    h : LE.le n m
    ⊢ Eq (List.Ico n (HAdd.hAdd m 1)) (HAppend.hAppend (List.Ico n m) (List.cons m …
  -/
  rwa [← succ_singleton, append_consecutive]
  /-
    case hml
    n m : Nat
    h : LE.le n m
    ⊢ LE.le m (HAdd.hAdd m 1)
  -/
  exact Nat.le_succ _
  /-
    🎉 no goals
  -/


theorem eq_cons {n m : ℕ} (h : n < m) : Ico n m = n :: Ico (n + 1) m := by
  /-
    n m : Nat
    h : LT.lt n m
    ⊢ Eq (List.Ico n m) (List.cons n (List.Ico (HAdd.hAdd n 1) m))
  -/
  rw [← append_consecutive (Nat.le_succ n) h, succ_singleton]
  /-
    n m : Nat
    h : LT.lt n m
    ⊢ Eq (HAppend.hAppend (List.cons n List.nil) (List.Ico n.succ m)) (List.cons n …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem pred_singleton {m : ℕ} (h : 0 < m) : Ico (m - 1) m = [m - 1] := by
  /-
    m : Nat
    h : LT.lt 0 m
    ⊢ Eq (List.Ico (HSub.hSub m 1) m) (List.cons (HSub.hSub m 1) List.nil)
  -/
  simp [Ico, Nat.sub_sub_self (succ_le_of_lt h)]
  /-
    🎉 no goals
  -/


theorem chain'_succ (n m : ℕ) : Chain' (fun a b => b = succ a) (Ico n m) := by
  /-
    n m : Nat
    ⊢ List.Chain' (fun a b => Eq b a.succ) (List.Ico n m)
  -/
  by_cases h : n < m
    /-
      case pos
      n m : Nat
      h : LT.lt n m
      ⊢ List.Chain' (fun a b => Eq b a.succ) (List.Ico n m)
    -/
  · rw [eq_cons h]
    /-
      case pos
      n m : Nat
      h : LT.lt n m
      ⊢ List.Chain' (fun a b => Eq b a.succ) (List.cons n (List.Ico (HAdd.hAdd n 1)  …
    -/
    exact chain_succ_range' _ _ 1
    /-
      🎉 no goals
    -/
    /-
      case neg
      n m : Nat
      h : Not (LT.lt n m)
      ⊢ List.Chain' (fun a b => Eq b a.succ) (List.Ico n m)
    -/
  · rw [eq_nil_of_le (le_of_not_gt h)]
    /-
      case neg
      n m : Nat
      h : Not (LT.lt n m)
      ⊢ List.Chain' (fun a b => Eq b a.succ) List.nil
    -/
    trivial
    /-
      🎉 no goals
    -/


                                                  /-
                                                    n m : Nat
                                                    ⊢ Not (Membership.mem (List.Ico n m) m)
                                                  -/
theorem not_mem_top {n m : ℕ} : m ∉ Ico n m := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem filter_lt_of_top_le {n m l : ℕ} (hml : m ≤ l) :
    ((Ico n m).filter fun x => x < l) = Ico n m :=
  filter_eq_self.2 fun k hk => by
    /-
      n m l : Nat
      hml : LE.le m l
      k : Nat
      hk : Membership.mem (List.Ico n m) k
      ⊢ Eq (Decidable.decide (LT.lt k l)) Bool.true
    -/
    simp only [(lt_of_lt_of_le (mem.1 hk).2 hml), decide_true]
    /-
      🎉 no goals
    -/


theorem filter_lt_of_le_bot {n m l : ℕ} (hln : l ≤ n) : ((Ico n m).filter fun x => x < l) = [] :=
  filter_eq_nil_iff.2 fun k hk => by
     /-
       n m l : Nat
       hln : LE.le l n
       k : Nat
       hk : Membership.mem (List.Ico n m) k
       ⊢ Not (Eq (Decidable.decide (LT.lt k l)) Bool.true)
     -/
     simp only [decide_eq_true_eq, not_lt]
     /-
       n m l : Nat
       hln : LE.le l n
       k : Nat
       hk : Membership.mem (List.Ico n m) k
       ⊢ LE.le l k
     -/
     apply le_trans hln
     /-
       n m l : Nat
       hln : LE.le l n
       k : Nat
       hk : Membership.mem (List.Ico n m) k
       ⊢ LE.le n k
     -/
     exact (mem.1 hk).1
     /-
       🎉 no goals
     -/


theorem filter_lt_of_ge {n m l : ℕ} (hlm : l ≤ m) :
    ((Ico n m).filter fun x => x < l) = Ico n l := by
  /-
    n m l : Nat
    hlm : LE.le l m
    ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x l)) (List.Ico n m)) (Lis …
  -/
  rcases le_total n l with hnl | hln
  · rw [← append_consecutive hnl hlm, filter_append, filter_lt_of_top_le (le_refl l),
      filter_lt_of_le_bot (le_refl l), append_nil]
    /-
      case inr
      n m l : Nat
      hlm : LE.le l m
      hln : LE.le l n
      ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x l)) (List.Ico n m)) (Lis …
    -/
  · rw [eq_nil_of_le hln, filter_lt_of_le_bot hln]
    /-
      🎉 no goals
    -/


@[simp]
theorem filter_lt (n m l : ℕ) :
    ((Ico n m).filter fun x => x < l) = Ico n (min m l) := by
  /-
    n m l : Nat
    ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x l)) (List.Ico n m)) (Lis …
  -/
  rcases le_total m l with hml | hlm
    /-
      case inl
      n m l : Nat
      hml : LE.le m l
      ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x l)) (List.Ico n m)) (Lis …
    -/
  · rw [min_eq_left hml, filter_lt_of_top_le hml]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n m l : Nat
      hlm : LE.le l m
      ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x l)) (List.Ico n m)) (Lis …
    -/
  · rw [min_eq_right hlm, filter_lt_of_ge hlm]
    /-
      🎉 no goals
    -/


theorem filter_le_of_le_bot {n m l : ℕ} (hln : l ≤ n) :
    ((Ico n m).filter fun x => l ≤ x) = Ico n m :=
  filter_eq_self.2 fun k hk => by
    /-
      n m l : Nat
      hln : LE.le l n
      k : Nat
      hk : Membership.mem (List.Ico n m) k
      ⊢ Eq (Decidable.decide (LE.le l k)) Bool.true
    -/
    rw [decide_eq_true_eq]
    /-
      n m l : Nat
      hln : LE.le l n
      k : Nat
      hk : Membership.mem (List.Ico n m) k
      ⊢ LE.le l k
    -/
    exact le_trans hln (mem.1 hk).1
    /-
      🎉 no goals
    -/


theorem filter_le_of_top_le {n m l : ℕ} (hml : m ≤ l) : ((Ico n m).filter fun x => l ≤ x) = [] :=
  filter_eq_nil_iff.2 fun k hk => by
    /-
      n m l : Nat
      hml : LE.le m l
      k : Nat
      hk : Membership.mem (List.Ico n m) k
      ⊢ Not (Eq (Decidable.decide (LE.le l k)) Bool.true)
    -/
    rw [decide_eq_true_eq]
    /-
      n m l : Nat
      hml : LE.le m l
      k : Nat
      hk : Membership.mem (List.Ico n m) k
      ⊢ Not (LE.le l k)
    -/
    exact not_le_of_gt (lt_of_lt_of_le (mem.1 hk).2 hml)
    /-
      🎉 no goals
    -/


theorem filter_le_of_le {n m l : ℕ} (hnl : n ≤ l) :
    ((Ico n m).filter fun x => l ≤ x) = Ico l m := by
  /-
    n m l : Nat
    hnl : LE.le n l
    ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le l x)) (List.Ico n m)) (Lis …
  -/
  rcases le_total l m with hlm | hml
  · rw [← append_consecutive hnl hlm, filter_append, filter_le_of_top_le (le_refl l),
      filter_le_of_le_bot (le_refl l), nil_append]
    /-
      case inr
      n m l : Nat
      hnl : LE.le n l
      hml : LE.le m l
      ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le l x)) (List.Ico n m)) (Lis …
    -/
  · rw [eq_nil_of_le hml, filter_le_of_top_le hml]
    /-
      🎉 no goals
    -/


@[simp]
theorem filter_le (n m l : ℕ) : ((Ico n m).filter fun x => l ≤ x) = Ico (max n l) m := by
  /-
    n m l : Nat
    ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le l x)) (List.Ico n m)) (Lis …
  -/
  rcases le_total n l with hnl | hln
    /-
      case inl
      n m l : Nat
      hnl : LE.le n l
      ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le l x)) (List.Ico n m)) (Lis …
    -/
  · rw [max_eq_right hnl, filter_le_of_le hnl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n m l : Nat
      hln : LE.le l n
      ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le l x)) (List.Ico n m)) (Lis …
    -/
  · rw [max_eq_left hln, filter_le_of_le_bot hln]
    /-
      🎉 no goals
    -/


theorem filter_lt_of_succ_bot {n m : ℕ} (hnm : n < m) :
    ((Ico n m).filter fun x => x < n + 1) = [n] := by
  /-
    n m : Nat
    hnm : LT.lt n m
    ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x (HAdd.hAdd n 1))) (List. …
  -/
  have r : min m (n + 1) = n + 1 := (@inf_eq_right _ _ m (n + 1)).mpr hnm
  /-
    n m : Nat
    hnm : LT.lt n m
    r : Eq (Min.min m (HAdd.hAdd n 1)) (HAdd.hAdd n 1)
    ⊢ Eq (List.filter (fun x => Decidable.decide (LT.lt x (HAdd.hAdd n 1))) (List. …
  -/
  simp [filter_lt n m (n + 1), r]
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_le_of_bot {n m : ℕ} (hnm : n < m) : ((Ico n m).filter fun x => x ≤ n) = [n] := by
  /-
    n m : Nat
    hnm : LT.lt n m
    ⊢ Eq (List.filter (fun x => Decidable.decide (LE.le x n)) (List.Ico n m)) (Lis …
  -/
  rw [← filter_lt_of_succ_bot hnm]
  exact filter_congr fun _ _ => by
    simpa using Nat.lt_succ_iff.symm


/-- For any natural numbers n, a, and b, one of the following holds:
1. n < a
2. n ≥ b
3. n ∈ Ico a b
-/
theorem trichotomy (n a b : ℕ) : n < a ∨ b ≤ n ∨ n ∈ Ico a b := by
  /-
    n a b : Nat
    ⊢ Or (LT.lt n a) (Or (LE.le b n) (Membership.mem (List.Ico a b) n))
  -/
  by_cases h₁ : n < a
    /-
      case pos
      n a b : Nat
      h₁ : LT.lt n a
      ⊢ Or (LT.lt n a) (Or (LE.le b n) (Membership.mem (List.Ico a b) n))
    -/
  · left
    /-
      case pos.h
      n a b : Nat
      h₁ : LT.lt n a
      ⊢ LT.lt n a
    -/
    exact h₁
    /-
      🎉 no goals
    -/
    /-
      case neg
      n a b : Nat
      h₁ : Not (LT.lt n a)
      ⊢ Or (LT.lt n a) (Or (LE.le b n) (Membership.mem (List.Ico a b) n))
    -/
  · right
    /-
      case neg.h
      n a b : Nat
      h₁ : Not (LT.lt n a)
      ⊢ Or (LE.le b n) (Membership.mem (List.Ico a b) n)
    -/
    by_cases h₂ : n ∈ Ico a b
      /-
        case pos
        n a b : Nat
        h₁ : Not (LT.lt n a)
        h₂ : Membership.mem (List.Ico a b) n
        ⊢ Or (LE.le b n) (Membership.mem (List.Ico a b) n)
      -/
    · right
      /-
        case pos.h
        n a b : Nat
        h₁ : Not (LT.lt n a)
        h₂ : Membership.mem (List.Ico a b) n
        ⊢ Membership.mem (List.Ico a b) n
      -/
      exact h₂
      /-
        🎉 no goals
      -/
      /-
        case neg
        n a b : Nat
        h₁ : Not (LT.lt n a)
        h₂ : Not (Membership.mem (List.Ico a b) n)
        ⊢ Or (LE.le b n) (Membership.mem (List.Ico a b) n)
      -/
    · left
      /-
        case neg.h
        n a b : Nat
        h₁ : Not (LT.lt n a)
        h₂ : Not (Membership.mem (List.Ico a b) n)
        ⊢ LE.le b n
      -/
      simp only [Ico.mem, not_and, not_lt] at *
      /-
        case neg.h
        n a b : Nat
        h₁ : LE.le a n
        h₂ : LE.le a n → LE.le b n
        ⊢ LE.le b n
      -/
      exact h₂ h₁
      /-
        🎉 no goals
      -/


