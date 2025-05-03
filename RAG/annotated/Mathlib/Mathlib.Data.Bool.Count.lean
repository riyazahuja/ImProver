@[simp]
theorem count_not_add_count (l : List Bool) (b : Bool) : count (!b) l + count b l = length l := by
  -- Porting note: Proof re-written
  -- Old proof: simp only [length_eq_countP_add_countP (Eq (!b)), Bool.not_not_eq, count]
  /-
    l : List Bool
    b : Bool
    ⊢ Eq (HAdd.hAdd (List.count b.not l) (List.count b l)) l.length
  -/
  simp only [length_eq_countP_add_countP (· == !b), count, add_right_inj]
  /-
    l : List Bool
    b : Bool
    ⊢ Eq (List.countP (fun x => BEq.beq x b) l) (List.countP (fun a => Decidable.d …
  -/
  suffices (fun x => x == b) = (fun a => decide ¬(a == !b) = true) by rw [this]
  /-
    l : List Bool
    b : Bool
    ⊢ Eq (fun x => BEq.beq x b) fun a => Decidable.decide (Not (Eq (BEq.beq a b.no …
  -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  ext x; cases x <;> cases b <;> rfl
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem count_add_count_not (l : List Bool) (b : Bool) : count b l + count (!b) l = length l := by
  /-
    l : List Bool
    b : Bool
    ⊢ Eq (HAdd.hAdd (List.count b l) (List.count b.not l)) l.length
  -/
  rw [add_comm, count_not_add_count]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_false_add_count_true (l : List Bool) : count false l + count true l = length l :=
  count_not_add_count l true


@[simp]
theorem count_true_add_count_false (l : List Bool) : count true l + count false l = length l :=
  count_not_add_count l false


theorem Chain.count_not :
    ∀ {b : Bool} {l : List Bool}, Chain (· ≠ ·) b l → count (!b) l = count b l + length l % 2
  | _, [], _h => rfl
  | b, x :: l, h => by
    /-
      b x : Bool
      l : List Bool
      h : List.Chain (fun x1 x2 => Ne x1 x2) b (List.cons x l)
      ⊢ Eq (List.count b.not (List.cons x l)) (HAdd.hAdd (List.count b (List.cons x  …
    -/
    obtain rfl : b = !x := Bool.eq_not_iff.2 (rel_of_chain_cons h)
    rw [Bool.not_not, count_cons_self, count_cons_of_ne x.not_ne_self,
      Chain.count_not (chain_of_chain_cons h), length, add_assoc, Nat.mod_two_add_succ_mod_two]


theorem count_not_eq_count (hl : Chain' (· ≠ ·) l) (h2 : Even (length l)) (b : Bool) :
    count (!b) l = count b l := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    h2 : Even l.length
    b : Bool
    ⊢ Eq (List.count b.not l) (List.count b l)
  -/
  cases' l with x l
    /-
      case nil
      b : Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) List.nil
      h2 : Even List.nil.length
      ⊢ Eq (List.count b.not List.nil) (List.count b List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case cons
    b x : Bool
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
    h2 : Even (List.cons x l).length
    ⊢ Eq (List.count b.not (List.cons x l)) (List.count b (List.cons x l))
  -/
  rw [length_cons, Nat.even_add_one, Nat.not_even_iff] at h2
  suffices count (!x) (x :: l) = count x (x :: l) by
    -- Porting note: old proof is
    -- cases b <;> cases x <;> try exact this;
    cases b <;> cases x <;>
    revert this <;> simp only [Bool.not_false, Bool.not_true] <;> intro this <;>
    (try exact this) <;> exact this.symm
  /-
    case cons
    b x : Bool
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
    h2 : Eq (HMod.hMod l.length 2) 1
    ⊢ Eq (List.count x.not (List.cons x l)) (List.count x (List.cons x l))
  -/
  rw [count_cons_of_ne x.not_ne_self, hl.count_not, h2, count_cons_self]
  /-
    🎉 no goals
  -/


theorem count_false_eq_count_true (hl : Chain' (· ≠ ·) l) (h2 : Even (length l)) :
    count false l = count true l :=
  hl.count_not_eq_count h2 true


theorem count_not_le_count_add_one (hl : Chain' (· ≠ ·) l) (b : Bool) :
    count (!b) l ≤ count b l + 1 := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (List.count b.not l) (HAdd.hAdd (List.count b l) 1)
  -/
  cases' l with x l
    /-
      case nil
      b : Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) List.nil
      ⊢ LE.le (List.count b.not List.nil) (HAdd.hAdd (List.count b List.nil) 1)
    -/
  · exact zero_le _
    /-
      🎉 no goals
    -/
  /-
    case cons
    b x : Bool
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
    ⊢ LE.le (List.count b.not (List.cons x l)) (HAdd.hAdd (List.count b (List.cons …
  -/
  obtain rfl | rfl : b = x ∨ b = !x := by simp only [Bool.eq_not_iff, em]
    /-
      case cons.inl
      b : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons b l)
      ⊢ LE.le (List.count b.not (List.cons b l)) (HAdd.hAdd (List.count b (List.cons …
    -/
  · rw [count_cons_of_ne b.not_ne_self, count_cons_self, hl.count_not, add_assoc]
    /-
      case cons.inl
      b : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons b l)
      ⊢ LE.le (HAdd.hAdd (List.count b l) (HMod.hMod l.length 2)) (HAdd.hAdd (List.c …
    -/
    exact add_le_add_left (Nat.mod_lt _ two_pos).le _
    /-
      🎉 no goals
    -/
    /-
      case cons.inr
      x : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
      ⊢ LE.le (List.count x.not.not (List.cons x l)) (HAdd.hAdd (List.count x.not (L …
    -/
  · rw [Bool.not_not, count_cons_self, count_cons_of_ne x.not_ne_self, hl.count_not]
    /-
      case cons.inr
      x : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
      ⊢ LE.le (HAdd.hAdd (List.count x l) 1) (HAdd.hAdd (HAdd.hAdd (List.count x l)  …
    -/
    exact add_le_add_right (le_add_right le_rfl) _
    /-
      🎉 no goals
    -/


theorem count_false_le_count_true_add_one (hl : Chain' (· ≠ ·) l) :
    count false l ≤ count true l + 1 :=
  hl.count_not_le_count_add_one true


theorem count_true_le_count_false_add_one (hl : Chain' (· ≠ ·) l) :
    count true l ≤ count false l + 1 :=
  hl.count_not_le_count_add_one false


theorem two_mul_count_bool_of_even (hl : Chain' (· ≠ ·) l) (h2 : Even (length l)) (b : Bool) :
    2 * count b l = length l := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    h2 : Even l.length
    b : Bool
    ⊢ Eq (HMul.hMul 2 (List.count b l)) l.length
  -/
  rw [← count_not_add_count l b, hl.count_not_eq_count h2, two_mul]
  /-
    🎉 no goals
  -/


theorem two_mul_count_bool_eq_ite (hl : Chain' (· ≠ ·) l) (b : Bool) :
    2 * count b l =
      if Even (length l) then length l else
      if Option.some b == l.head? then length l + 1 else length l - 1 := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ Eq (HMul.hMul 2 (List.count b l)) (ite (Even l.length) l.length (ite (Eq (BE …
  -/
  by_cases h2 : Even (length l)
    /-
      case pos
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
      b : Bool
      h2 : Even l.length
      ⊢ Eq (HMul.hMul 2 (List.count b l)) (ite (Even l.length) l.length (ite (Eq (BE …
    -/
  · rw [if_pos h2, hl.two_mul_count_bool_of_even h2]
    /-
      🎉 no goals
    -/
    /-
      case neg
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
      b : Bool
      h2 : Not (Even l.length)
      ⊢ Eq (HMul.hMul 2 (List.count b l)) (ite (Even l.length) l.length (ite (Eq (BE …
    -/
  · cases' l with x l
      /-
        case neg.nil
        b : Bool
        hl : List.Chain' (fun x1 x2 => Ne x1 x2) List.nil
        h2 : Not (Even List.nil.length)
        ⊢ Eq (HMul.hMul 2 (List.count b List.nil)) (ite (Even List.nil.length) List.ni …
      -/
    · exact (h2 even_zero).elim
      /-
        🎉 no goals
      -/
    /-
      case neg.cons
      b x : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
      h2 : Not (Even (List.cons x l).length)
      ⊢ Eq (HMul.hMul 2 (List.count b (List.cons x l))) (ite (Even (List.cons x l).l …
    -/
    simp only [if_neg h2, count_cons, mul_add, head?, Option.mem_some_iff, @eq_comm _ x]
    /-
      case neg.cons
      b x : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
      h2 : Not (Even (List.cons x l).length)
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (List.count b l)) (HMul.hMul 2 (ite (Eq (BEq.beq  …
    -/
    rw [length_cons, Nat.even_add_one, not_not] at h2
    /-
      case neg.cons
      b x : Bool
      l : List Bool
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) (List.cons x l)
      h2 : Even l.length
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (List.count b l)) (HMul.hMul 2 (ite (Eq (BEq.beq  …
    -/
    replace hl : l.Chain' (· ≠ ·) := hl.tail
    /-
      case neg.cons
      b x : Bool
      l : List Bool
      h2 : Even l.length
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (List.count b l)) (HMul.hMul 2 (ite (Eq (BEq.beq  …
    -/
    rw [hl.two_mul_count_bool_of_even h2]
    /-
      case neg.cons
      b x : Bool
      l : List Bool
      h2 : Even l.length
      hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
      ⊢ Eq (HAdd.hAdd l.length (HMul.hMul 2 (ite (Eq (BEq.beq x b) Bool.true) 1 0))) …
    -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    cases b <;> cases x <;> split_ifs <;> simp <;> contradiction
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem length_sub_one_le_two_mul_count_bool (hl : Chain' (· ≠ ·) l) (b : Bool) :
    length l - 1 ≤ 2 * count b l := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (HSub.hSub l.length 1) (HMul.hMul 2 (List.count b l))
  -/
  rw [hl.two_mul_count_bool_eq_ite]
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (HSub.hSub l.length 1) (ite (Even l.length) l.length (ite (Eq (BEq.beq …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [le_tsub_add, Nat.le_succ_of_le]
                /-
                  🎉 no goals
                -/


theorem length_div_two_le_count_bool (hl : Chain' (· ≠ ·) l) (b : Bool) :
    length l / 2 ≤ count b l := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (HDiv.hDiv l.length 2) (List.count b l)
  -/
  rw [Nat.div_le_iff_le_mul_add_pred two_pos, ← tsub_le_iff_right]
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (HSub.hSub l.length (HSub.hSub 2 1)) (HMul.hMul 2 (List.count b l))
  -/
  exact length_sub_one_le_two_mul_count_bool hl b
  /-
    🎉 no goals
  -/


theorem two_mul_count_bool_le_length_add_one (hl : Chain' (· ≠ ·) l) (b : Bool) :
    2 * count b l ≤ length l + 1 := by
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (HMul.hMul 2 (List.count b l)) (HAdd.hAdd l.length 1)
  -/
  rw [hl.two_mul_count_bool_eq_ite]
  /-
    l : List Bool
    hl : List.Chain' (fun x1 x2 => Ne x1 x2) l
    b : Bool
    ⊢ LE.le (ite (Even l.length) l.length (ite (Eq (BEq.beq (Option.some b) l.head …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [Nat.le_succ_of_le]
                /-
                  🎉 no goals
                -/


