@[simp, norm_cast]
theorem cast_one [One α] [Add α] : ((1 : PosNum) : α) = 1 :=
  rfl


@[simp]
theorem cast_one' [One α] [Add α] : (PosNum.one : α) = 1 :=
  rfl


@[simp, norm_cast]
theorem cast_bit0 [One α] [Add α] (n : PosNum) : (n.bit0 : α) = (n : α) + n :=
  rfl


@[simp, norm_cast]
theorem cast_bit1 [One α] [Add α] (n : PosNum) : (n.bit1 : α) = ((n : α) + n) + 1 :=
  rfl


@[simp, norm_cast]
theorem cast_to_nat [AddMonoidWithOne α] : ∀ n : PosNum, ((n : ℕ) : α) = n
  | 1 => Nat.cast_one
                 /-
                   α : Type u_1
                   inst✝ : AddMonoidWithOne α
                   p : PosNum
                   ⊢ Eq ↑↑p.bit0 ↑p.bit0
                 -/
  | bit0 p => by dsimp; rw [Nat.cast_add, p.cast_to_nat]
                        /-
                          🎉 no goals
                        -/
                 /-
                   α : Type u_1
                   inst✝ : AddMonoidWithOne α
                   p : PosNum
                   ⊢ Eq ↑↑p.bit1 ↑p.bit1
                 -/
  | bit1 p => by dsimp; rw [Nat.cast_add, Nat.cast_add, Nat.cast_one, p.cast_to_nat]
                        /-
                          🎉 no goals
                        -/


@[norm_cast]
theorem to_nat_to_int (n : PosNum) : ((n : ℕ) : ℤ) = n :=
  cast_to_nat _


@[simp, norm_cast]
theorem cast_to_int [AddGroupWithOne α] (n : PosNum) : ((n : ℤ) : α) = n := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : PosNum
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← to_nat_to_int, Int.cast_natCast, cast_to_nat]
  /-
    🎉 no goals
  -/


theorem succ_to_nat : ∀ n, (succ n : ℕ) = n + 1
  | 1 => rfl
  | bit0 _ => rfl
  | bit1 p =>
    (congr_arg (fun n ↦ n + n) (succ_to_nat p)).trans <|
                                                /-
                                                  p : PosNum
                                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (↑p) 1) ↑p) 1) (HAdd.hAdd (HAdd.hAdd (HA …
                                                -/
      show ↑p + 1 + ↑p + 1 = ↑p + ↑p + 1 + 1 by simp [add_left_comm]
                                                /-
                                                  🎉 no goals
                                                -/


                                                    /-
                                                      n : PosNum
                                                      ⊢ Eq (HAdd.hAdd 1 n) n.succ
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem one_add (n : PosNum) : 1 + n = succ n := by cases n <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                    /-
                                                      n : PosNum
                                                      ⊢ Eq (HAdd.hAdd n 1) n.succ
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem add_one (n : PosNum) : n + 1 = succ n := by cases n <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[norm_cast]
theorem add_to_nat : ∀ m n, ((m + n : PosNum) : ℕ) = m + n
               /-
                 b : PosNum
                 ⊢ Eq (↑(HAdd.hAdd 1 b)) (HAdd.hAdd ↑1 ↑b)
               -/
  | 1, b => by rw [one_add b, succ_to_nat, add_comm, cast_one]
               /-
                 🎉 no goals
               -/
               /-
                 a : PosNum
                 ⊢ Eq (↑(HAdd.hAdd a 1)) (HAdd.hAdd ↑a ↑1)
               -/
  | a, 1 => by rw [add_one a, succ_to_nat, cast_one]
               /-
                 🎉 no goals
               -/
  | bit0 a, bit0 b => (congr_arg (fun n ↦ n + n) (add_to_nat a b)).trans <| add_add_add_comm _ _ _ _
  | bit0 a, bit1 b =>
    (congr_arg (fun n ↦ (n + n) + 1) (add_to_nat a b)).trans <|
                                                              /-
                                                                a b : PosNum
                                                                ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ↑a ↑b) (HAdd.hAdd ↑a ↑b)) 1) (HAdd.hAdd  …
                                                              -/
      show (a + b + (a + b) + 1 : ℕ) = a + a + (b + b + 1) by simp [add_left_comm]
                                                              /-
                                                                🎉 no goals
                                                              -/
  | bit1 a, bit0 b =>
    (congr_arg (fun n ↦ (n + n) + 1) (add_to_nat a b)).trans <|
                                                              /-
                                                                a b : PosNum
                                                                ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ↑a ↑b) (HAdd.hAdd ↑a ↑b)) 1) (HAdd.hAdd  …
                                                              -/
      show (a + b + (a + b) + 1 : ℕ) = a + a + 1 + (b + b) by simp [add_comm, add_left_comm]
                                                              /-
                                                                🎉 no goals
                                                              -/
  | bit1 a, bit1 b =>
    show (succ (a + b) + succ (a + b) : ℕ) = a + a + 1 + (b + b + 1) by
      /-
        a b : PosNum
        ⊢ Eq (HAdd.hAdd ↑(HAdd.hAdd a b).succ ↑(HAdd.hAdd a b).succ) (HAdd.hAdd (HAdd. …
      -/
      rw [succ_to_nat, add_to_nat a b]; simp [add_left_comm]
                                        /-
                                          🎉 no goals
                                        -/


theorem add_succ : ∀ m n : PosNum, m + succ n = succ (m + n)
               /-
                 b : PosNum
                 ⊢ Eq (HAdd.hAdd 1 b.succ) (HAdd.hAdd 1 b).succ
               -/
  | 1, b => by simp [one_add]
               /-
                 🎉 no goals
               -/
  | bit0 a, 1 => congr_arg bit0 (add_one a)
  | bit1 a, 1 => congr_arg bit1 (add_one a)
  | bit0 _, bit0 _ => rfl
  | bit0 a, bit1 b => congr_arg bit0 (add_succ a b)
  | bit1 _, bit0 _ => rfl
  | bit1 a, bit1 b => congr_arg bit1 (add_succ a b)


theorem bit0_of_bit0 : ∀ n, n + n = bit0 n
  | 1 => rfl
  | bit0 p => congr_arg bit0 (bit0_of_bit0 p)
                                              /-
                                                p : PosNum
                                                ⊢ Eq (HAdd.hAdd p p).succ.bit0 p.bit1.bit0
                                              -/
  | bit1 p => show bit0 (succ (p + p)) = _ by rw [bit0_of_bit0 p, succ]
                                              /-
                                                🎉 no goals
                                              -/


theorem bit1_of_bit1 (n : PosNum) : (n + n) + 1 = bit1 n :=
                               /-
                                 n : PosNum
                                 ⊢ Eq (HAdd.hAdd (HAdd.hAdd n n) 1) n.bit1
                               -/
  show (n + n) + 1 = bit1 n by rw [add_one, bit0_of_bit0, succ]
                               /-
                                 🎉 no goals
                               -/


@[norm_cast]
theorem mul_to_nat (m) : ∀ n, ((m * n : PosNum) : ℕ) = m * n
  | 1 => (mul_one _).symm
                                                               /-
                                                                 m p : PosNum
                                                                 ⊢ Eq (HAdd.hAdd ↑(HMul.hMul m p) ↑(HMul.hMul m p)) (HMul.hMul (↑m) (HAdd.hAdd  …
                                                               -/
  | bit0 p => show (↑(m * p) + ↑(m * p) : ℕ) = ↑m * (p + p) by rw [mul_to_nat m p, left_distrib]
                                                               /-
                                                                 🎉 no goals
                                                               -/
  | bit1 p =>
    (add_to_nat (bit0 (m * p)) m).trans <|
                                                                /-
                                                                  m p : PosNum
                                                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd ↑(HMul.hMul m p) ↑(HMul.hMul m p)) ↑m) (HAdd.hAdd ( …
                                                                -/
      show (↑(m * p) + ↑(m * p) + ↑m : ℕ) = ↑m * (p + p) + m by rw [mul_to_nat m p, left_distrib]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem to_nat_pos : ∀ n : PosNum, 0 < (n : ℕ)
  | 1 => Nat.zero_lt_one
  | bit0 p =>
    let h := to_nat_pos p
    add_pos h h
  | bit1 _p => Nat.succ_pos _


theorem cmp_to_nat_lemma {m n : PosNum} : (m : ℕ) < n → (bit1 m : ℕ) < bit0 n :=
  show (m : ℕ) < n → (m + m + 1 + 1 : ℕ) ≤ n + n by
    /-
      m n : PosNum
      ⊢ LT.lt ↑m ↑n → LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ↑m ↑m) 1) 1) (HAdd.hAdd …
    -/
    intro h; rw [Nat.add_right_comm m m 1, add_assoc]; exact Nat.add_le_add h h
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem cmp_swap (m) : ∀ n, (cmp m n).swap = cmp n m := by
  /-
    m : PosNum
    ⊢ ∀ (n : PosNum), Eq (m.cmp n).swap (n.cmp m)
  -/
  induction' m with m IH m IH <;> intro n <;> cases' n with n n <;> unfold cmp <;>
    /-
      case one.one
      ⊢ Eq Ordering.eq.swap Ordering.eq
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
    try { rfl } <;> rw [← IH] <;> cases cmp m n <;> rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem cmp_to_nat : ∀ m n, (Ordering.casesOn (cmp m n) ((m : ℕ) < n) (m = n) ((n : ℕ) < m) : Prop)
  | 1, 1 => rfl
  | bit0 a, 1 =>
    let h : (1 : ℕ) ≤ a := to_nat_pos a
    Nat.add_le_add h h
  | bit1 a, 1 => Nat.succ_lt_succ <| to_nat_pos <| bit0 a
  | 1, bit0 b =>
    let h : (1 : ℕ) ≤ b := to_nat_pos b
    Nat.add_le_add h h
  | 1, bit1 b => Nat.succ_lt_succ <| to_nat_pos <| bit0 b
  | bit0 a, bit0 b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn (a.bit0.cmp b.bit0) (LT.lt ↑a.bit0 ↑b.bit0) (Eq a.bit0 b.bi …
    -/
    dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt (HAdd.hAdd ↑a ↑a) (HAdd.hAdd ↑b ↑b)) (Eq a.bit0 b.bit0)  …
    -/
    have := cmp_to_nat a b; revert this; cases cmp a b <;> dsimp <;> intro this
      /-
        case lt
        a b : PosNum
        this : LT.lt ↑a ↑b
        ⊢ LT.lt (HAdd.hAdd ↑a ↑a) (HAdd.hAdd ↑b ↑b)
      -/
    · exact Nat.add_lt_add this this
      /-
        🎉 no goals
      -/
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ Eq a.bit0 b.bit0
      -/
    · rw [this]
      /-
        🎉 no goals
      -/
      /-
        case gt
        a b : PosNum
        this : LT.lt ↑b ↑a
        ⊢ LT.lt (HAdd.hAdd ↑b ↑b) (HAdd.hAdd ↑a ↑a)
      -/
    · exact Nat.add_lt_add this this
      /-
        🎉 no goals
      -/
  | bit0 a, bit1 b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn (a.bit0.cmp b.bit1) (LT.lt ↑a.bit0 ↑b.bit1) (Eq a.bit0 b.bi …
    -/
    dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt (HAdd.hAdd ↑a ↑a) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)) (Eq a …
    -/
    have := cmp_to_nat a b; revert this; cases cmp a b <;> dsimp <;> intro this
      /-
        case lt
        a b : PosNum
        this : LT.lt ↑a ↑b
        ⊢ LT.lt (HAdd.hAdd ↑a ↑a) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)
      -/
    · exact Nat.le_succ_of_le (Nat.add_lt_add this this)
      /-
        🎉 no goals
      -/
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ LT.lt (HAdd.hAdd ↑a ↑a) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)
      -/
    · rw [this]
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ LT.lt (HAdd.hAdd ↑b ↑b) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)
      -/
      apply Nat.lt_succ_self
      /-
        🎉 no goals
      -/
      /-
        case gt
        a b : PosNum
        this : LT.lt ↑b ↑a
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1) (HAdd.hAdd ↑a ↑a)
      -/
    · exact cmp_to_nat_lemma this
      /-
        🎉 no goals
      -/
  | bit1 a, bit0 b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn (a.bit1.cmp b.bit0) (LT.lt ↑a.bit1 ↑b.bit0) (Eq a.bit1 b.bi …
    -/
    dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1) (HAdd.hAdd ↑b ↑b)) (Eq a …
    -/
    have := cmp_to_nat a b; revert this; cases cmp a b <;> dsimp <;> intro this
      /-
        case lt
        a b : PosNum
        this : LT.lt ↑a ↑b
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1) (HAdd.hAdd ↑b ↑b)
      -/
    · exact cmp_to_nat_lemma this
      /-
        🎉 no goals
      -/
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ LT.lt (HAdd.hAdd ↑b ↑b) (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1)
      -/
    · rw [this]
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ LT.lt (HAdd.hAdd ↑b ↑b) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)
      -/
      apply Nat.lt_succ_self
      /-
        🎉 no goals
      -/
      /-
        case gt
        a b : PosNum
        this : LT.lt ↑b ↑a
        ⊢ LT.lt (HAdd.hAdd ↑b ↑b) (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1)
      -/
    · exact Nat.le_succ_of_le (Nat.add_lt_add this this)
      /-
        🎉 no goals
      -/
  | bit1 a, bit1 b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn (a.bit1.cmp b.bit1) (LT.lt ↑a.bit1 ↑b.bit1) (Eq a.bit1 b.bi …
    -/
    dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1) (HAdd.hAdd (HAdd.hAdd ↑b …
    -/
    have := cmp_to_nat a b; revert this; cases cmp a b <;> dsimp <;> intro this
      /-
        case lt
        a b : PosNum
        this : LT.lt ↑a ↑b
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1) (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1)
      -/
    · exact Nat.succ_lt_succ (Nat.add_lt_add this this)
      /-
        🎉 no goals
      -/
      /-
        case eq
        a b : PosNum
        this : Eq a b
        ⊢ Eq a.bit1 b.bit1
      -/
    · rw [this]
      /-
        🎉 no goals
      -/
      /-
        case gt
        a b : PosNum
        this : LT.lt ↑b ↑a
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd ↑b ↑b) 1) (HAdd.hAdd (HAdd.hAdd ↑a ↑a) 1)
      -/
    · exact Nat.succ_lt_succ (Nat.add_lt_add this this)
      /-
        🎉 no goals
      -/


@[norm_cast]
theorem lt_to_nat {m n : PosNum} : (m : ℕ) < n ↔ m < n :=
  show (m : ℕ) < n ↔ cmp m n = Ordering.lt from
    match cmp m n, cmp_to_nat m n with
                           /-
                             m n : PosNum
                             h : Ordering.casesOn Ordering.lt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.lt Ordering.lt)
                           -/
    | Ordering.lt, h => by simp only at h; simp [h]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : PosNum
                             h : Ordering.casesOn Ordering.eq (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.eq Ordering.lt)
                           -/
    | Ordering.eq, h => by simp only at h; simp [h, lt_irrefl]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : PosNum
                             h : Ordering.casesOn Ordering.gt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.gt Ordering.lt)
                           -/
    | Ordering.gt, h => by simp [not_lt_of_gt h]
                           /-
                             🎉 no goals
                           -/


@[norm_cast]
theorem le_to_nat {m n : PosNum} : (m : ℕ) ≤ n ↔ m ≤ n := by
  /-
    m n : PosNum
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr lt_to_nat
                 /-
                   🎉 no goals
                 -/


                                             /-
                                               n : Num
                                               ⊢ Eq (HAdd.hAdd n 0) n
                                             -/
                                                         /-
                                                           🎉 no goals
                                                         -/
theorem add_zero (n : Num) : n + 0 = n := by cases n <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


                                             /-
                                               n : Num
                                               ⊢ Eq (HAdd.hAdd 0 n) n
                                             -/
                                                         /-
                                                           🎉 no goals
                                                         -/
theorem zero_add (n : Num) : 0 + n = n := by cases n <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem add_one : ∀ n : Num, n + 1 = succ n
  | 0 => rfl
                /-
                  p : PosNum
                  ⊢ Eq (HAdd.hAdd (Num.pos p) 1) (Num.pos p).succ
                -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
  | pos p => by cases p <;> rfl
                            /-
                              🎉 no goals
                            -/


theorem add_succ : ∀ m n : Num, m + succ n = succ (m + n)
               /-
                 n : Num
                 ⊢ Eq (HAdd.hAdd 0 n.succ) (HAdd.hAdd 0 n).succ
               -/
  | 0, n => by simp [zero_add]
               /-
                 🎉 no goals
               -/
                                                       /-
                                                         p : PosNum
                                                         ⊢ Eq (Num.pos (HAdd.hAdd p 1)) (HAdd.hAdd (Num.pos p) 0).succ
                                                       -/
  | pos p, 0 => show pos (p + 1) = succ (pos p + 0) by rw [PosNum.add_one, add_zero, succ, succ']
                                                       /-
                                                         🎉 no goals
                                                       -/
  | pos _, pos _ => congr_arg pos (PosNum.add_succ _ _)


theorem bit0_of_bit0 : ∀ n : Num, n + n = n.bit0
  | 0 => rfl
  | pos p => congr_arg pos p.bit0_of_bit0


theorem bit1_of_bit1 : ∀ n : Num, (n + n) + 1 = n.bit1
  | 0 => rfl
  | pos p => congr_arg pos p.bit1_of_bit1


@[simp]
                                             /-
                                               ⊢ Eq (Num.ofNat' 0) 0
                                             -/
theorem ofNat'_zero : Num.ofNat' 0 = 0 := by simp [Num.ofNat']
                                             /-
                                               🎉 no goals
                                             -/


theorem ofNat'_bit (b n) : ofNat' (Nat.bit b n) = cond b Num.bit1 Num.bit0 (ofNat' n) :=
  Nat.binaryRec_eq _ _ (.inl rfl)


@[simp]
                                            /-
                                              ⊢ Eq (Num.ofNat' 1) 1
                                            -/
theorem ofNat'_one : Num.ofNat' 1 = 1 := by erw [ofNat'_bit true 0, cond, ofNat'_zero]; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem bit1_succ : ∀ n : Num, n.bit1.succ = n.succ.bit0
  | 0 => rfl
  | pos _n => rfl


theorem ofNat'_succ : ∀ {n}, ofNat' (n + 1) = ofNat' n + 1 :=
                      /-
                        ⊢ Eq (Num.ofNat' (HAdd.hAdd 0 1)) (HAdd.hAdd (Num.ofNat' 0) 1)
                      -/
  @(Nat.binaryRec (by simp [zero_add]) fun b n ih => by
                      /-
                        🎉 no goals
                      -/
    /-
      b : Bool
      n : Nat
      ih : Eq (Num.ofNat' (HAdd.hAdd n 1)) (HAdd.hAdd (Num.ofNat' n) 1)
      ⊢ Eq (Num.ofNat' (HAdd.hAdd (Nat.bit b n) 1)) (HAdd.hAdd (Num.ofNat' (Nat.bit  …
    -/
    cases b
      /-
        case false
        n : Nat
        ih : Eq (Num.ofNat' (HAdd.hAdd n 1)) (HAdd.hAdd (Num.ofNat' n) 1)
        ⊢ Eq (Num.ofNat' (HAdd.hAdd (Nat.bit Bool.false n) 1)) (HAdd.hAdd (Num.ofNat'  …
      -/
    · erw [ofNat'_bit true n, ofNat'_bit]
      /-
        case false
        n : Nat
        ih : Eq (Num.ofNat' (HAdd.hAdd n 1)) (HAdd.hAdd (Num.ofNat' n) 1)
        ⊢ Eq (cond Bool.true Num.bit1 Num.bit0 (Num.ofNat' n)) (HAdd.hAdd (cond Bool.f …
      -/
      simp only [← bit1_of_bit1, ← bit0_of_bit0, cond]
      /-
        🎉 no goals
      -/
    · rw [show n.bit true + 1 = (n + 1).bit false by simp [Nat.bit, mul_add],
        ofNat'_bit, ofNat'_bit, ih]
      /-
        case true
        n : Nat
        ih : Eq (Num.ofNat' (HAdd.hAdd n 1)) (HAdd.hAdd (Num.ofNat' n) 1)
        ⊢ Eq (cond Bool.false Num.bit1 Num.bit0 (HAdd.hAdd (Num.ofNat' n) 1)) (HAdd.hA …
      -/
      simp only [cond, add_one, bit1_succ])
      /-
        🎉 no goals
      -/


@[simp]
theorem add_ofNat' (m n) : Num.ofNat' (m + n) = Num.ofNat' m + Num.ofNat' n := by
  /-
    m n : Nat
    ⊢ Eq (Num.ofNat' (HAdd.hAdd m n)) (HAdd.hAdd (Num.ofNat' m) (Num.ofNat' n))
  -/
  induction n
    /-
      case zero
      m : Nat
      ⊢ Eq (Num.ofNat' (HAdd.hAdd m 0)) (HAdd.hAdd (Num.ofNat' m) (Num.ofNat' 0))
    -/
  · simp only [Nat.add_zero, ofNat'_zero, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      m n✝ : Nat
      a✝ : Eq (Num.ofNat' (HAdd.hAdd m n✝)) (HAdd.hAdd (Num.ofNat' m) (Num.ofNat' n✝))
      ⊢ Eq (Num.ofNat' (HAdd.hAdd m (HAdd.hAdd n✝ 1))) (HAdd.hAdd (Num.ofNat' m) (Nu …
    -/
  · simp only [Nat.add_succ, Nat.add_zero, ofNat'_succ, add_one, add_succ, *]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_zero [Zero α] [One α] [Add α] : ((0 : Num) : α) = 0 :=
  rfl


@[simp]
theorem cast_zero' [Zero α] [One α] [Add α] : (Num.zero : α) = 0 :=
  rfl


@[simp, norm_cast]
theorem cast_one [Zero α] [One α] [Add α] : ((1 : Num) : α) = 1 :=
  rfl


@[simp]
theorem cast_pos [Zero α] [One α] [Add α] (n : PosNum) : (Num.pos n : α) = n :=
  rfl


theorem succ'_to_nat : ∀ n, (succ' n : ℕ) = n + 1
  | 0 => (Nat.zero_add _).symm
  | pos _p => PosNum.succ_to_nat _


theorem succ_to_nat (n) : (succ n : ℕ) = n + 1 :=
  succ'_to_nat n


@[simp, norm_cast]
theorem cast_to_nat [AddMonoidWithOne α] : ∀ n : Num, ((n : ℕ) : α) = n
  | 0 => Nat.cast_zero
  | pos p => p.cast_to_nat


@[norm_cast]
theorem add_to_nat : ∀ m n, ((m + n : Num) : ℕ) = m + n
  | 0, 0 => rfl
  | 0, pos _q => (Nat.zero_add _).symm
  | pos _p, 0 => rfl
  | pos _p, pos _q => PosNum.add_to_nat _ _


@[norm_cast]
theorem mul_to_nat : ∀ m n, ((m * n : Num) : ℕ) = m * n
  | 0, 0 => rfl
  | 0, pos _q => (zero_mul _).symm
  | pos _p, 0 => rfl
  | pos _p, pos _q => PosNum.mul_to_nat _ _


theorem cmp_to_nat : ∀ m n, (Ordering.casesOn (cmp m n) ((m : ℕ) < n) (m = n) ((n : ℕ) < m) : Prop)
  | 0, 0 => rfl
  | 0, pos _ => to_nat_pos _
  | pos _, 0 => to_nat_pos _
  | pos a, pos b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn ((Num.pos a).cmp (Num.pos b)) (LT.lt ↑(Num.pos a) ↑(Num.pos …
    -/
    have := PosNum.cmp_to_nat a b; revert this; dsimp [cmp]; cases PosNum.cmp a b
    /-
      case lt
      a b : PosNum
      ⊢ Ordering.rec (LT.lt ↑a ↑b) (Eq a b) (LT.lt ↑b ↑a) Ordering.lt → Ordering.rec …
    -/
    exacts [id, congr_arg pos, id]
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem lt_to_nat {m n : Num} : (m : ℕ) < n ↔ m < n :=
  show (m : ℕ) < n ↔ cmp m n = Ordering.lt from
    match cmp m n, cmp_to_nat m n with
                           /-
                             m n : Num
                             h : Ordering.casesOn Ordering.lt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.lt Ordering.lt)
                           -/
    | Ordering.lt, h => by simp only at h; simp [h]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : Num
                             h : Ordering.casesOn Ordering.eq (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.eq Ordering.lt)
                           -/
    | Ordering.eq, h => by simp only at h; simp [h, lt_irrefl]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : Num
                             h : Ordering.casesOn Ordering.gt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.gt Ordering.lt)
                           -/
    | Ordering.gt, h => by simp [not_lt_of_gt h]
                           /-
                             🎉 no goals
                           -/


@[norm_cast]
theorem le_to_nat {m n : Num} : (m : ℕ) ≤ n ↔ m ≤ n := by
  /-
    m n : Num
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr lt_to_nat
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem of_to_nat' : ∀ n : PosNum, Num.ofNat' (n : ℕ) = Num.pos n
            /-
              ⊢ Eq (Num.ofNat' ↑1) (Num.pos 1)
            -/
  | 1 => by erw [@Num.ofNat'_bit true 0, Num.ofNat'_zero]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/
  | bit0 p => by
      /-
        p : PosNum
        ⊢ Eq (Num.ofNat' ↑p.bit0) (Num.pos p.bit0)
      -/
      simpa only [Nat.bit_false, cond_false, two_mul, of_to_nat' p] using Num.ofNat'_bit false p
      /-
        🎉 no goals
      -/
  | bit1 p => by
      /-
        p : PosNum
        ⊢ Eq (Num.ofNat' ↑p.bit1) (Num.pos p.bit1)
      -/
      simpa only [Nat.bit_true, cond_true, two_mul, of_to_nat' p] using Num.ofNat'_bit true p
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem of_to_nat' : ∀ n : Num, Num.ofNat' (n : ℕ) = n
  | 0 => ofNat'_zero
  | pos p => p.of_to_nat'


lemma toNat_injective : Injective (castNum : Num → ℕ) := LeftInverse.injective of_to_nat'


@[norm_cast]
theorem to_nat_inj {m n : Num} : (m : ℕ) = n ↔ m = n := toNat_injective.eq_iff


/-- This tactic tries to turn an (in)equality about `Num`s to one about `Nat`s by rewriting.
```lean
example (n : Num) (m : Num) : n ≤ n + m := by
  transfer_rw
  exact Nat.le_add_right _ _
```
-/
scoped macro (name := transfer_rw) "transfer_rw" : tactic => `(tactic|
    (repeat first | rw [← to_nat_inj] | rw [← lt_to_nat] | rw [← le_to_nat]
     repeat first | rw [add_to_nat] | rw [mul_to_nat] | rw [cast_one] | rw [cast_zero]))


/--
This tactic tries to prove (in)equalities about `Num`s by transferring them to the `Nat` world and
then trying to call `simp`.
```lean
example (n : Num) (m : Num) : n ≤ n + m := by transfer
```
-/
scoped macro (name := transfer) "transfer" : tactic => `(tactic|
    (intros; transfer_rw; try simp))


instance addMonoid : AddMonoid Num where
  add := (· + ·)
  zero := 0
  zero_add := zero_add
  add_zero := add_zero
                  /-
                    ⊢ ∀ (a b c : Num), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b  …
                  -/
  add_assoc := by transfer
                  /-
                    🎉 no goals
                  -/
  nsmul := nsmulRec


instance addMonoidWithOne : AddMonoidWithOne Num :=
  { Num.addMonoid with
    natCast := Num.ofNat'
    one := 1
    natCast_zero := ofNat'_zero
    natCast_succ := fun _ => ofNat'_succ }


instance commSemiring : CommSemiring Num where
  __ := Num.addMonoid
  __ := Num.addMonoidWithOne
  mul := (· * ·)
  npow := @npowRec Num ⟨1⟩ ⟨(· * ·)⟩
                   /-
                     x✝ : Num
                     ⊢ Eq (HMul.hMul x✝ 0) 0
                   -/
                   /-
                     x✝ : Num
                     ⊢ Eq (HMul.hMul 0 x✝) 0
                   -/
  mul_zero _ := by rw [← to_nat_inj, mul_to_nat, cast_zero, mul_zero]
                   /-
                     🎉 no goals
                   -/
                     /-
                       x✝¹ x✝ : Num
                       ⊢ Eq (HAdd.hAdd x✝¹ x✝) (HAdd.hAdd x✝ x✝¹)
                     -/
                   /-
                     🎉 no goals
                   -/
                     /-
                       🎉 no goals
                     -/
  zero_mul _ := by rw [← to_nat_inj, mul_to_nat, cast_zero, zero_mul]
                  /-
                    x✝ : Num
                    ⊢ Eq (HMul.hMul x✝ 1) x✝
                  -/
                           /-
                             x✝² x✝¹ x✝ : Num
                             ⊢ Eq (HMul.hMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HMul.hMul x✝² x✝¹) (HMul.h …
                           -/
                  /-
                    x✝ : Num
                    ⊢ Eq (HMul.hMul 1 x✝) x✝
                  -/
                           /-
                             🎉 no goals
                           -/
                            /-
                              x✝² x✝¹ x✝ : Num
                              ⊢ Eq (HMul.hMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HMul.hMul x✝² x✝) (HMul.hM …
                            -/
  mul_one _ := by rw [← to_nat_inj, mul_to_nat, cast_one, mul_one]
                            /-
                              🎉 no goals
                            -/
                  /-
                    🎉 no goals
                  -/
                        /-
                          x✝² x✝¹ x✝ : Num
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝² x✝¹) x✝) (HMul.hMul x✝² (HMul.hMul x✝¹ x✝))
                        -/
                  /-
                    🎉 no goals
                  -/
                        /-
                          🎉 no goals
                        -/
  one_mul _ := by rw [← to_nat_inj, mul_to_nat, cast_one, one_mul]
  add_comm _ _ := by simp_rw [← to_nat_inj, add_to_nat, add_comm]
                     /-
                       x✝¹ x✝ : Num
                       ⊢ Eq (HMul.hMul x✝¹ x✝) (HMul.hMul x✝ x✝¹)
                     -/
  mul_comm _ _ := by simp_rw [← to_nat_inj, mul_to_nat, mul_comm]
                     /-
                       🎉 no goals
                     -/
  mul_assoc _ _ _ := by simp_rw [← to_nat_inj, mul_to_nat, mul_assoc]
  left_distrib _ _ _ := by simp only [← to_nat_inj, mul_to_nat, add_to_nat, mul_add]
  right_distrib _ _ _ := by simp only [← to_nat_inj, mul_to_nat, add_to_nat, add_mul]


instance orderedCancelAddCommMonoid : OrderedCancelAddCommMonoid Num where
  le := (· ≤ ·)
  lt := (· < ·)
                             /-
                               a b : Num
                               ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
                             -/
                /-
                  ⊢ ∀ (a : Num), LE.le a a
                -/
  lt_iff_le_not_le a b := by simp only [← lt_to_nat, ← le_to_nat, lt_iff_le_not_le]
                /-
                  🎉 no goals
                -/
                       /-
                         a b c : Num
                         ⊢ LE.le a b → LE.le b c → LE.le a c
                       -/
                             /-
                               🎉 no goals
                             -/
                                    /-
                                      🎉 no goals
                                    -/
  le_refl := by transfer
  le_trans a b c := by transfer_rw; apply le_trans
                        /-
                          a b : Num
                          ⊢ LE.le a b → LE.le b a → Eq a b
                        -/
  le_antisymm a b := by transfer_rw; apply le_antisymm
                                     /-
                                       🎉 no goals
                                     -/
                                /-
                                  a b : Num
                                  h : LE.le a b
                                  c : Num
                                  ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
                                -/
  add_le_add_left a b h c := by revert h; transfer_rw; exact fun h => add_le_add_left h c
                                                       /-
                                                         🎉 no goals
                                                       -/
                                    /-
                                      a b c : Num
                                      ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a c) → LE.le b c
                                    -/
  le_of_add_le_add_left a b c := by transfer_rw; apply le_of_add_le_add_left
                                                 /-
                                                   🎉 no goals
                                                 -/


instance linearOrderedSemiring : LinearOrderedSemiring Num :=
  { Num.commSemiring,
    Num.orderedCancelAddCommMonoid with
    le_total := by
      /-
        ⊢ ∀ (a b : Num), Or (LE.le a b) (LE.le b a)
      -/
      intro a b
      /-
        a b : Num
        ⊢ Or (LE.le a b) (LE.le b a)
      -/
                      /-
                        ⊢ LE.le 0 1
                      -/
      transfer_rw
                      /-
                        🎉 no goals
                      -/
      /-
        a b : Num
        ⊢ Or (LE.le ↑a ↑b) (LE.le ↑b ↑a)
      -/
      /-
        ⊢ ∀ (a b c : Num), LT.lt a b → LT.lt 0 c → LT.lt (HMul.hMul c a) (HMul.hMul c b)
      -/
      apply le_total
      /-
        a b c : Num
        ⊢ LT.lt a b → LT.lt 0 c → LT.lt (HMul.hMul c a) (HMul.hMul c b)
      -/
      /-
        🎉 no goals
      -/
      /-
        a b c : Num
        ⊢ LT.lt ↑a ↑b → LT.lt 0 ↑c → LT.lt (HMul.hMul ↑c ↑a) (HMul.hMul ↑c ↑b)
      -/
    zero_le_one := by decide
      /-
        🎉 no goals
      -/
    mul_lt_mul_of_pos_left := by
      /-
        ⊢ ∀ (a b c : Num), LT.lt a b → LT.lt 0 c → LT.lt (HMul.hMul a c) (HMul.hMul b c)
      -/
      intro a b c
      /-
        a b c : Num
        ⊢ LT.lt a b → LT.lt 0 c → LT.lt (HMul.hMul a c) (HMul.hMul b c)
      -/
                                /-
                                  ⊢ Ne 0 1
                                -/
      transfer_rw
                                /-
                                  🎉 no goals
                                -/
      /-
        a b c : Num
        ⊢ LT.lt ↑a ↑b → LT.lt 0 ↑c → LT.lt (HMul.hMul ↑a ↑c) (HMul.hMul ↑b ↑c)
      -/
      apply mul_lt_mul_of_pos_left
      /-
        🎉 no goals
      -/
    mul_lt_mul_of_pos_right := by
      intro a b c
      transfer_rw
      apply mul_lt_mul_of_pos_right
    decidableLT := Num.decidableLT
    decidableLE := Num.decidableLE
    -- This is relying on an automatically generated instance name,
    -- generated in a `deriving` handler.
    -- See https://github.com/leanprover/lean4/issues/2343
    decidableEq := instDecidableEqNum
    exists_pair_ne := ⟨0, 1, by decide⟩ }


@[norm_cast]
theorem add_of_nat (m n) : ((m + n : ℕ) : Num) = m + n :=
  add_ofNat' _ _


@[norm_cast]
theorem to_nat_to_int (n : Num) : ((n : ℕ) : ℤ) = n :=
  cast_to_nat _


@[simp, norm_cast]
theorem cast_to_int {α} [AddGroupWithOne α] (n : Num) : ((n : ℤ) : α) = n := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : Num
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← to_nat_to_int, Int.cast_natCast, cast_to_nat]
  /-
    🎉 no goals
  -/


theorem to_of_nat : ∀ n : ℕ, ((n : Num) : ℕ) = n
            /-
              ⊢ Eq (↑↑0) 0
            -/
  | 0 => by rw [Nat.cast_zero, cast_zero]
            /-
              🎉 no goals
            -/
                /-
                  n : Nat
                  ⊢ Eq (↑↑(HAdd.hAdd n 1)) (HAdd.hAdd n 1)
                -/
  | n + 1 => by rw [Nat.cast_succ, add_one, succ_to_nat, to_of_nat n]
                /-
                  🎉 no goals
                -/


@[simp, norm_cast]
theorem of_natCast {α} [AddMonoidWithOne α] (n : ℕ) : ((n : Num) : α) = n := by
  /-
    α : Type u_1
    inst✝ : AddMonoidWithOne α
    n : Nat
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← cast_to_nat, to_of_nat]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias of_nat_cast := of_natCast


@[norm_cast]
theorem of_nat_inj {m n : ℕ} : (m : Num) = n ↔ m = n :=
  ⟨fun h => Function.LeftInverse.injective to_of_nat h, congr_arg _⟩

-- Porting note: The priority should be `high`er than `cast_to_nat`.

@[simp high, norm_cast]
theorem of_to_nat : ∀ n : Num, ((n : ℕ) : Num) = n :=
  of_to_nat'


@[norm_cast]
theorem dvd_to_nat (m n : Num) : (m : ℕ) ∣ n ↔ m ∣ n :=
                        /-
                          m n : Num
                          x✝ : Dvd.dvd ↑m ↑n
                          k : Nat
                          e : Eq (↑n) (HMul.hMul (↑m) k)
                          ⊢ Eq n (HMul.hMul m ↑k)
                        -/
                                               /-
                                                 🎉 no goals
                                               -/
  ⟨fun ⟨k, e⟩ => ⟨k, by rw [← of_to_nat n, e]; simp⟩, fun ⟨k, e⟩ => ⟨k, by simp [e, mul_to_nat]⟩⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp high, norm_cast]
theorem of_to_nat : ∀ n : PosNum, ((n : ℕ) : Num) = Num.pos n :=
  of_to_nat'


@[norm_cast]
theorem to_nat_inj {m n : PosNum} : (m : ℕ) = n ↔ m = n :=
                              /-
                                m n : PosNum
                                h : Eq ↑m ↑n
                                ⊢ Eq (Num.pos m) (Num.pos n)
                              -/
  ⟨fun h => Num.pos.inj <| by rw [← PosNum.of_to_nat, ← PosNum.of_to_nat, h], congr_arg _⟩
                              /-
                                🎉 no goals
                              -/


theorem pred'_to_nat : ∀ n, (pred' n : ℕ) = Nat.pred n
  | 1 => rfl
  | bit0 n =>
    have : Nat.succ ↑(pred' n) = ↑n := by
      /-
        n : PosNum
        ⊢ Eq (↑n.pred').succ ↑n
      -/
      rw [pred'_to_nat n, Nat.succ_pred_eq_of_pos (to_nat_pos n)]
      /-
        🎉 no goals
      -/
    match (motive :=
        ∀ k : Num, Nat.succ ↑k = ↑n → ↑(Num.casesOn k 1 bit1 : PosNum) = Nat.pred (n + n))
      pred' n, this with
                                         /-
                                           n : PosNum
                                           this : Eq (↑n.pred').succ ↑n
                                           h : Eq ↑1 ↑n
                                           ⊢ Eq (↑(Num.casesOn 0 1 PosNum.bit1)) (HAdd.hAdd ↑n ↑n).pred
                                         -/
    | 0, (h : ((1 : Num) : ℕ) = n) => by rw [← to_nat_inj.1 h]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                             /-
                                               n : PosNum
                                               this : Eq (↑n.pred').succ ↑n
                                               p : PosNum
                                               h : Eq (↑p).succ ↑n
                                               ⊢ Eq (↑(Num.casesOn (Num.pos p) 1 PosNum.bit1)) (HAdd.hAdd ↑n ↑n).pred
                                             -/
    | Num.pos p, (h : Nat.succ ↑p = n) => by rw [← h]; exact (Nat.succ_add p p).symm
                                                       /-
                                                         🎉 no goals
                                                       -/
  | bit1 _ => rfl


@[simp]
theorem pred'_succ' (n) : pred' (succ' n) = n :=
                         /-
                           n : Num
                           ⊢ Eq ↑n.succ'.pred' ↑n
                         -/
  Num.to_nat_inj.1 <| by rw [pred'_to_nat, succ'_to_nat, Nat.add_one, Nat.pred_succ]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem succ'_pred' (n) : succ' (pred' n) = n :=
  to_nat_inj.1 <| by
    /-
      n : PosNum
      ⊢ Eq ↑n.pred'.succ' ↑n
    -/
    rw [succ'_to_nat, pred'_to_nat, Nat.add_one, Nat.succ_pred_eq_of_pos (to_nat_pos _)]
    /-
      🎉 no goals
    -/


instance dvd : Dvd PosNum :=
  ⟨fun m n => pos m ∣ pos n⟩


@[norm_cast]
theorem dvd_to_nat {m n : PosNum} : (m : ℕ) ∣ n ↔ m ∣ n :=
  Num.dvd_to_nat (pos m) (pos n)


theorem size_to_nat : ∀ n, (size n : ℕ) = Nat.size n
  | 1 => Nat.size_one.symm
  | bit0 n => by
      /-
        n : PosNum
        ⊢ Eq (↑n.bit0.size) (↑n.bit0).size
      -/
      rw [size, succ_to_nat, size_to_nat n, cast_bit0, ← two_mul]
      /-
        n : PosNum
        ⊢ Eq (HAdd.hAdd (↑n).size 1) (HMul.hMul 2 ↑n).size
      -/
      erw [@Nat.size_bit false n]
      /-
        n : PosNum
        ⊢ Ne (Nat.bit Bool.false ↑n) 0
      -/
      have := to_nat_pos n
      /-
        n : PosNum
        this : LT.lt 0 ↑n
        ⊢ Ne (Nat.bit Bool.false ↑n) 0
      -/
      dsimp [Nat.bit]; omega
                       /-
                         🎉 no goals
                       -/
  | bit1 n => by
      /-
        n : PosNum
        ⊢ Eq (↑n.bit1.size) (↑n.bit1).size
      -/
      rw [size, succ_to_nat, size_to_nat n, cast_bit1, ← two_mul]
      /-
        n : PosNum
        ⊢ Eq (HAdd.hAdd (↑n).size 1) (HAdd.hAdd (HMul.hMul 2 ↑n) 1).size
      -/
      erw [@Nat.size_bit true n]
      /-
        n : PosNum
        ⊢ Ne (Nat.bit Bool.true ↑n) 0
      -/
      dsimp [Nat.bit]; omega
                       /-
                         🎉 no goals
                       -/


theorem size_eq_natSize : ∀ n, (size n : ℕ) = natSize n
  | 1 => rfl
                 /-
                   n : PosNum
                   ⊢ Eq (↑n.bit0.size) n.bit0.natSize
                 -/
  | bit0 n => by rw [size, succ_to_nat, natSize, size_eq_natSize n]
                 /-
                   🎉 no goals
                 -/
                 /-
                   n : PosNum
                   ⊢ Eq (↑n.bit1.size) n.bit1.natSize
                 -/
  | bit1 n => by rw [size, succ_to_nat, natSize, size_eq_natSize n]
                 /-
                   🎉 no goals
                 -/


                                                          /-
                                                            n : PosNum
                                                            ⊢ Eq n.natSize (↑n).size
                                                          -/
theorem natSize_to_nat (n) : natSize n = Nat.size n := by rw [← size_eq_natSize, size_to_nat]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                              /-
                                                n : PosNum
                                                ⊢ LT.lt 0 n.natSize
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
theorem natSize_pos (n) : 0 < natSize n := by cases n <;> apply Nat.succ_pos
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- This tactic tries to turn an (in)equality about `PosNum`s to one about `Nat`s by rewriting.
```lean
example (n : PosNum) (m : PosNum) : n ≤ n + m := by
  transfer_rw
  exact Nat.le_add_right _ _
```
-/
scoped macro (name := transfer_rw) "transfer_rw" : tactic => `(tactic|
    (repeat first | rw [← to_nat_inj] | rw [← lt_to_nat] | rw [← le_to_nat]
     repeat first | rw [add_to_nat] | rw [mul_to_nat] | rw [cast_one] | rw [cast_zero]))


/--
This tactic tries to prove (in)equalities about `PosNum`s by transferring them to the `Nat` world
and then trying to call `simp`.
```lean
example (n : PosNum) (m : PosNum) : n ≤ n + m := by transfer
```
-/
scoped macro (name := transfer) "transfer" : tactic => `(tactic|
    (intros; transfer_rw; try simp [add_comm, add_left_comm, mul_comm, mul_left_comm]))


instance addCommSemigroup : AddCommSemigroup PosNum where
  add := (· + ·)
                  /-
                    α : Type u_1
                    ⊢ ∀ (a b c : PosNum), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd …
                  -/
                  /-
                    🎉 no goals
                  -/
  add_assoc := by transfer
                  /-
                    🎉 no goals
                  -/
                 /-
                   α : Type u_1
                   ⊢ ∀ (a b : PosNum), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                 -/
                 /-
                   🎉 no goals
                 -/
  add_comm := by transfer
                 /-
                   🎉 no goals
                 -/


instance commMonoid : CommMonoid PosNum where
  mul := (· * ·)
  one := (1 : PosNum)
  npow := @npowRec PosNum ⟨1⟩ ⟨(· * ·)⟩
                  /-
                    α : Type u_1
                    ⊢ ∀ (a b c : PosNum), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul …
                  -/
                  /-
                    🎉 no goals
                  -/
  mul_assoc := by transfer
                  /-
                    🎉 no goals
                  -/
                /-
                  α : Type u_1
                  ⊢ ∀ (a : PosNum), Eq (HMul.hMul 1 a) a
                -/
                /-
                  🎉 no goals
                -/
  one_mul := by transfer
                /-
                  🎉 no goals
                -/
                /-
                  α : Type u_1
                  ⊢ ∀ (a : PosNum), Eq (HMul.hMul a 1) a
                -/
                /-
                  🎉 no goals
                -/
  mul_one := by transfer
                /-
                  🎉 no goals
                -/
                 /-
                   α : Type u_1
                   ⊢ ∀ (a b : PosNum), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
                 /-
                   🎉 no goals
                 -/
  mul_comm := by transfer
                 /-
                   🎉 no goals
                 -/


instance distrib : Distrib PosNum where
  add := (· + ·)
  mul := (· * ·)
                     /-
                       α : Type u_1
                       ⊢ ∀ (a b c : PosNum), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a …
                     -/
  left_distrib := by transfer; simp [mul_add]
                               /-
                                 🎉 no goals
                               -/
                      /-
                        α : Type u_1
                        ⊢ ∀ (a b c : PosNum), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a …
                      -/
  right_distrib := by transfer; simp [mul_add, mul_comm]
                                /-
                                  🎉 no goals
                                -/


instance linearOrder : LinearOrder PosNum where
  lt := (· < ·)
  lt_iff_le_not_le := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : PosNum), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    intro a b
    /-
      α : Type u_1
      a b : PosNum
      ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    transfer_rw
                /-
                  α : Type u_1
                  ⊢ ∀ (a : PosNum), LE.le a a
                -/
                /-
                  🎉 no goals
                -/
    /-
      α : Type u_1
      a b : PosNum
      ⊢ Iff (LT.lt ↑a ↑b) (And (LE.le ↑a ↑b) (Not (LE.le ↑b ↑a)))
    -/
                /-
                  🎉 no goals
                -/
    apply lt_iff_le_not_le
    /-
      α : Type u_1
      ⊢ ∀ (a b c : PosNum), LE.le a b → LE.le b c → LE.le a c
    -/
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      a b c : PosNum
      ⊢ LE.le a b → LE.le b c → LE.le a c
    -/
  le := (· ≤ ·)
    /-
      α : Type u_1
      a b c : PosNum
      ⊢ LE.le ↑a ↑b → LE.le ↑b ↑c → LE.le ↑a ↑c
    -/
  le_refl := by transfer
    /-
      🎉 no goals
    -/
  le_trans := by
    intro a b c
    transfer_rw
    apply le_trans
  le_antisymm := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : PosNum), LE.le a b → LE.le b a → Eq a b
    -/
    intro a b
    /-
      α : Type u_1
      a b : PosNum
      ⊢ LE.le a b → LE.le b a → Eq a b
    -/
    transfer_rw
    /-
      α : Type u_1
      a b : PosNum
      ⊢ LE.le ↑a ↑b → LE.le ↑b ↑a → Eq ↑a ↑b
    -/
    apply le_antisymm
    /-
      🎉 no goals
    -/
  le_total := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : PosNum), Or (LE.le a b) (LE.le b a)
    -/
    intro a b
    /-
      α : Type u_1
      a b : PosNum
      ⊢ Or (LE.le a b) (LE.le b a)
    -/
    transfer_rw
    /-
      α : Type u_1
      a b : PosNum
      ⊢ Or (LE.le ↑a ↑b) (LE.le ↑b ↑a)
    -/
    apply le_total
    /-
      🎉 no goals
    -/
                    /-
                      α : Type u_1
                      ⊢ DecidableRel fun x1 x2 => LT.lt x1 x2
                    -/
                    /-
                      α : Type u_1
                      ⊢ DecidableRel fun x1 x2 => LE.le x1 x2
                    -/
  decidableLT := by infer_instance
                    /-
                      🎉 no goals
                    -/
                    /-
                      α : Type u_1
                      ⊢ DecidableEq PosNum
                    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  decidableLE := by infer_instance
  decidableEq := by infer_instance


@[simp]
                                                        /-
                                                          n : PosNum
                                                          ⊢ Eq (↑n) (Num.pos n)
                                                        -/
theorem cast_to_num (n : PosNum) : ↑n = Num.pos n := by rw [← cast_to_nat, ← of_to_nat n]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp, norm_cast]
                                                             /-
                                                               b : Bool
                                                               n : PosNum
                                                               ⊢ Eq (↑(PosNum.bit b n)) (Nat.bit b ↑n)
                                                             -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
theorem bit_to_nat (b n) : (bit b n : ℕ) = Nat.bit b n := by cases b <;> simp [bit, two_mul]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp, norm_cast]
theorem cast_add [AddMonoidWithOne α] (m n) : ((m + n : PosNum) : α) = m + n := by
  /-
    α : Type u_1
    inst✝ : AddMonoidWithOne α
    m n : PosNum
    ⊢ Eq (↑(HAdd.hAdd m n)) (HAdd.hAdd ↑m ↑n)
  -/
  rw [← cast_to_nat, add_to_nat, Nat.cast_add, cast_to_nat, cast_to_nat]
  /-
    🎉 no goals
  -/


@[simp 500, norm_cast]
theorem cast_succ [AddMonoidWithOne α] (n : PosNum) : (succ n : α) = n + 1 := by
  /-
    α : Type u_1
    inst✝ : AddMonoidWithOne α
    n : PosNum
    ⊢ Eq (↑n.succ) (HAdd.hAdd (↑n) 1)
  -/
  rw [← add_one, cast_add, cast_one]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_inj [AddMonoidWithOne α] [CharZero α] {m n : PosNum} : (m : α) = n ↔ m = n := by
  /-
    α : Type u_1
    inst✝¹ : AddMonoidWithOne α
    inst✝ : CharZero α
    m n : PosNum
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  rw [← cast_to_nat m, ← cast_to_nat n, Nat.cast_inj, to_nat_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_le_cast [LinearOrderedSemiring α] (n : PosNum) : (1 : α) ≤ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    n : PosNum
    ⊢ LE.le 1 ↑n
  -/
  rw [← cast_to_nat, ← Nat.cast_one, Nat.cast_le (α := α)]; apply to_nat_pos
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem cast_pos [LinearOrderedSemiring α] (n : PosNum) : 0 < (n : α) :=
  lt_of_lt_of_le zero_lt_one (one_le_cast n)


@[simp, norm_cast]
theorem cast_mul [Semiring α] (m n) : ((m * n : PosNum) : α) = m * n := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    m n : PosNum
    ⊢ Eq (↑(HMul.hMul m n)) (HMul.hMul ↑m ↑n)
  -/
  rw [← cast_to_nat, mul_to_nat, Nat.cast_mul, cast_to_nat, cast_to_nat]
  /-
    🎉 no goals
  -/


@[simp]
theorem cmp_eq (m n) : cmp m n = Ordering.eq ↔ m = n := by
  /-
    m n : PosNum
    ⊢ Iff (Eq (m.cmp n) Ordering.eq) (Eq m n)
  -/
  have := cmp_to_nat m n
  -- Porting note: `cases` didn't rewrite at `this`, so `revert` & `intro` are required.
  /-
    m n : PosNum
    this : Ordering.casesOn (m.cmp n) (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
    ⊢ Iff (Eq (m.cmp n) Ordering.eq) (Eq m n)
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  revert this; cases cmp m n <;> intro this <;> simp at this ⊢ <;> try { exact this } <;>
    /-
      case lt
      m n : PosNum
      this : LT.lt ↑m ↑n
      ⊢ Not (Eq m n)
    -/
    /-
      🎉 no goals
    -/
    simp [show m ≠ n from fun e => by rw [e] at this;exact lt_irrefl _ this]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_lt [LinearOrderedSemiring α] {m n : PosNum} : (m : α) < n ↔ m < n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    m n : PosNum
    ⊢ Iff (LT.lt ↑m ↑n) (LT.lt m n)
  -/
  rw [← cast_to_nat m, ← cast_to_nat n, Nat.cast_lt (α := α), lt_to_nat]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_le [LinearOrderedSemiring α] {m n : PosNum} : (m : α) ≤ n ↔ m ≤ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    m n : PosNum
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr cast_lt
                 /-
                   🎉 no goals
                 -/


theorem bit_to_nat (b n) : (bit b n : ℕ) = Nat.bit b n := by
  /-
    b : Bool
    n : Num
    ⊢ Eq (↑(Num.bit b n)) (Nat.bit b ↑n)
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
  cases b <;> cases n <;> simp [bit, two_mul] <;> rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem cast_succ' [AddMonoidWithOne α] (n) : (succ' n : α) = n + 1 := by
  /-
    α : Type u_1
    inst✝ : AddMonoidWithOne α
    n : Num
    ⊢ Eq (↑n.succ') (HAdd.hAdd (↑n) 1)
  -/
  rw [← PosNum.cast_to_nat, succ'_to_nat, Nat.cast_add_one, cast_to_nat]
  /-
    🎉 no goals
  -/


theorem cast_succ [AddMonoidWithOne α] (n) : (succ n : α) = n + 1 :=
  cast_succ' n


@[simp, norm_cast]
theorem cast_add [Semiring α] (m n) : ((m + n : Num) : α) = m + n := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    m n : Num
    ⊢ Eq (↑(HAdd.hAdd m n)) (HAdd.hAdd ↑m ↑n)
  -/
  rw [← cast_to_nat, add_to_nat, Nat.cast_add, cast_to_nat, cast_to_nat]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_bit0 [Semiring α] (n : Num) : (n.bit0 : α) = 2 * (n : α) := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    n : Num
    ⊢ Eq (↑n.bit0) (HMul.hMul 2 ↑n)
  -/
  rw [← bit0_of_bit0, two_mul, cast_add]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_bit1 [Semiring α] (n : Num) : (n.bit1 : α) = 2 * (n : α) + 1 := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    n : Num
    ⊢ Eq (↑n.bit1) (HAdd.hAdd (HMul.hMul 2 ↑n) 1)
  -/
  rw [← bit1_of_bit1, bit0_of_bit0, cast_add, cast_bit0]; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp, norm_cast]
theorem cast_mul [Semiring α] : ∀ m n, ((m * n : Num) : α) = m * n
  | 0, 0 => (zero_mul _).symm
  | 0, pos _q => (zero_mul _).symm
  | pos _p, 0 => (mul_zero _).symm
  | pos _p, pos _q => PosNum.cast_mul _ _


theorem size_to_nat : ∀ n, (size n : ℕ) = Nat.size n
  | 0 => Nat.size_zero.symm
  | pos p => p.size_to_nat


theorem size_eq_natSize : ∀ n, (size n : ℕ) = natSize n
  | 0 => rfl
  | pos p => p.size_eq_natSize


                                                          /-
                                                            n : Num
                                                            ⊢ Eq n.natSize (↑n).size
                                                          -/
theorem natSize_to_nat (n) : natSize n = Nat.size n := by rw [← size_eq_natSize, size_to_nat]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp 999]
theorem ofNat'_eq : ∀ n, Num.ofNat' n = n :=
                    /-
                      ⊢ Eq (Num.ofNat' 0) ↑0
                    -/
  Nat.binaryRec (by simp) fun b n IH => by
                    /-
                      🎉 no goals
                    -/
    /-
      b : Bool
      n : Nat
      IH : Eq (Num.ofNat' n) ↑n
      ⊢ Eq (Num.ofNat' (Nat.bit b n)) ↑(Nat.bit b n)
    -/
    rw [ofNat'] at IH ⊢
    /-
      b : Bool
      n : Nat
      IH : Eq (Nat.binaryRec 0 (fun b x => cond b Num.bit1 Num.bit0) n) ↑n
      ⊢ Eq (Nat.binaryRec 0 (fun b x => cond b Num.bit1 Num.bit0) (Nat.bit b n)) ↑(N …
    -/
    rw [Nat.binaryRec_eq _ _ (.inl rfl), IH]
    -- Porting note: `Nat.cast_bit0` & `Nat.cast_bit1` are not `simp` theorems anymore.
    /-
      b : Bool
      n : Nat
      IH : Eq (Nat.binaryRec 0 (fun b x => cond b Num.bit1 Num.bit0) n) ↑n
      ⊢ Eq (cond b Num.bit1 Num.bit0 ↑n) ↑(Nat.bit b n)
    -/
    cases b <;> simp only [cond_false, cond_true, Nat.bit, two_mul, Nat.cast_add, Nat.cast_one]
      /-
        case false
        n : Nat
        IH : Eq (Nat.binaryRec 0 (fun b x => cond b Num.bit1 Num.bit0) n) ↑n
        ⊢ Eq (↑n).bit0 (HAdd.hAdd ↑n ↑n)
      -/
    · rw [bit0_of_bit0]
      /-
        🎉 no goals
      -/
      /-
        case true
        n : Nat
        IH : Eq (Nat.binaryRec 0 (fun b x => cond b Num.bit1 Num.bit0) n) ↑n
        ⊢ Eq (↑n).bit1 (HAdd.hAdd (HAdd.hAdd ↑n ↑n) 1)
      -/
    · rw [bit1_of_bit1]
      /-
        🎉 no goals
      -/


                                                              /-
                                                                n : Num
                                                                ⊢ Eq (Neg.neg n.toZNum) n.toZNumNeg
                                                              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
theorem zneg_toZNum (n : Num) : -n.toZNum = n.toZNumNeg := by cases n <;> rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                                 /-
                                                                   n : Num
                                                                   ⊢ Eq (Neg.neg n.toZNumNeg) n.toZNum
                                                                 -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
theorem zneg_toZNumNeg (n : Num) : -n.toZNumNeg = n.toZNum := by cases n <;> rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem toZNum_inj {m n : Num} : m.toZNum = n.toZNum ↔ m = n :=
               /-
                 m n : Num
                 h : Eq m.toZNum n.toZNum
                 ⊢ Eq m n
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
  ⟨fun h => by cases m <;> cases n <;> cases h <;> rfl, congr_arg _⟩
                                                   /-
                                                     🎉 no goals
                                                   -/



@[simp]
theorem cast_toZNum [Zero α] [One α] [Add α] [Neg α] : ∀ n : Num, (n.toZNum : α) = n
  | 0 => rfl
  | Num.pos _p => rfl


@[simp]
theorem cast_toZNumNeg [AddGroup α] [One α] : ∀ n : Num, (n.toZNumNeg : α) = -n
  | 0 => neg_zero.symm
  | Num.pos _p => rfl


@[simp]
theorem add_toZNum (m n : Num) : Num.toZNum (m + n) = m.toZNum + n.toZNum := by
  /-
    m n : Num
    ⊢ Eq (HAdd.hAdd m n).toZNum (HAdd.hAdd m.toZNum n.toZNum)
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
  cases m <;> cases n <;> rfl
                          /-
                            🎉 no goals
                          -/


theorem pred_to_nat {n : PosNum} (h : 1 < n) : (pred n : ℕ) = Nat.pred n := by
  /-
    n : PosNum
    h : LT.lt 1 n
    ⊢ Eq (↑n.pred) (↑n).pred
  -/
  unfold pred
  /-
    n : PosNum
    h : LT.lt 1 n
    ⊢ Eq (↑(Num.casesOn n.pred' 1 id)) (↑n).pred
  -/
  cases e : pred' n
    /-
      case zero
      n : PosNum
      h : LT.lt 1 n
      e : Eq n.pred' Num.zero
      ⊢ Eq (↑(Num.casesOn Num.zero 1 id)) (↑n).pred
    -/
  · have : (1 : ℕ) ≤ Nat.pred n := Nat.pred_le_pred ((@cast_lt ℕ _ _ _).2 h)
    /-
      case zero
      n : PosNum
      h : LT.lt 1 n
      e : Eq n.pred' Num.zero
      this : LE.le 1 (↑n).pred
      ⊢ Eq (↑(Num.casesOn Num.zero 1 id)) (↑n).pred
    -/
    rw [← pred'_to_nat, e] at this
    /-
      case zero
      n : PosNum
      h : LT.lt 1 n
      e : Eq n.pred' Num.zero
      this : LE.le 1 ↑Num.zero
      ⊢ Eq (↑(Num.casesOn Num.zero 1 id)) (↑n).pred
    -/
    exact absurd this (by decide)
    /-
      🎉 no goals
    -/
    /-
      case pos
      n : PosNum
      h : LT.lt 1 n
      a✝ : PosNum
      e : Eq n.pred' (Num.pos a✝)
      ⊢ Eq (↑(Num.casesOn (Num.pos a✝) 1 id)) (↑n).pred
    -/
  · rw [← pred'_to_nat, e]
    /-
      case pos
      n : PosNum
      h : LT.lt 1 n
      a✝ : PosNum
      e : Eq n.pred' (Num.pos a✝)
      ⊢ Eq ↑(Num.casesOn (Num.pos a✝) 1 id) ↑(Num.pos a✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


                                                                  /-
                                                                    a : PosNum
                                                                    ⊢ Eq (a.sub' 1) a.pred'.toZNum
                                                                  -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
theorem sub'_one (a : PosNum) : sub' a 1 = (pred' a).toZNum := by cases a <;> rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


                                                                     /-
                                                                       a : PosNum
                                                                       ⊢ Eq (PosNum.sub' 1 a) a.pred'.toZNumNeg
                                                                     -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
theorem one_sub' (a : PosNum) : sub' 1 a = (pred' a).toZNumNeg := by cases a <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem lt_iff_cmp {m n} : m < n ↔ cmp m n = Ordering.lt :=
  Iff.rfl


theorem le_iff_cmp {m n} : m ≤ n ↔ cmp m n ≠ Ordering.gt :=
                                      /-
                                        m n : PosNum
                                        ⊢ Iff (Eq (n.cmp m) Ordering.lt) (Eq (m.cmp n) Ordering.gt)
                                      -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  not_congr <| lt_iff_cmp.trans <| by rw [← cmp_swap]; cases cmp m n <;> decide
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem pred_to_nat : ∀ n : Num, (pred n : ℕ) = Nat.pred n
  | 0 => rfl
                /-
                  p : PosNum
                  ⊢ Eq (↑(Num.pos p).pred) (↑(Num.pos p)).pred
                -/
  | pos p => by rw [pred, PosNum.pred'_to_nat]; rfl
                                                /-
                                                  🎉 no goals
                                                -/


theorem ppred_to_nat : ∀ n : Num, (↑) <$> ppred n = Nat.ppred n
  | 0 => rfl
  | pos p => by
    /-
      p : PosNum
      ⊢ Eq (Functor.map castNum (Num.pos p).ppred) (↑(Num.pos p)).ppred
    -/
    rw [ppred, Option.map_some, Nat.ppred_eq_some.2]
    /-
      p : PosNum
      ⊢ Eq (↑p.pred').succ ↑(Num.pos p)
    -/
    rw [PosNum.pred'_to_nat, Nat.succ_pred_eq_of_pos (PosNum.to_nat_pos _)]
    /-
      p : PosNum
      ⊢ Eq ↑p ↑(Num.pos p)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem cmp_swap (m n) : (cmp m n).swap = cmp n m := by
  /-
    m n : Num
    ⊢ Eq (m.cmp n).swap (n.cmp m)
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
  cases m <;> cases n <;> try { rfl }; apply PosNum.cmp_swap
                                       /-
                                         🎉 no goals
                                       -/


theorem cmp_eq (m n) : cmp m n = Ordering.eq ↔ m = n := by
  /-
    m n : Num
    ⊢ Iff (Eq (m.cmp n) Ordering.eq) (Eq m n)
  -/
  have := cmp_to_nat m n
  -- Porting note: `cases` didn't rewrite at `this`, so `revert` & `intro` are required.
  /-
    m n : Num
    this : Ordering.casesOn (m.cmp n) (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
    ⊢ Iff (Eq (m.cmp n) Ordering.eq) (Eq m n)
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  revert this; cases cmp m n <;> intro this <;> simp at this ⊢ <;> try { exact this } <;>
    /-
      case lt
      m n : Num
      this : LT.lt ↑m ↑n
      ⊢ Not (Eq m n)
    -/
    /-
      🎉 no goals
    -/
    simp [show m ≠ n from fun e => by rw [e] at this; exact lt_irrefl _ this]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_lt [LinearOrderedSemiring α] {m n : Num} : (m : α) < n ↔ m < n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    m n : Num
    ⊢ Iff (LT.lt ↑m ↑n) (LT.lt m n)
  -/
  rw [← cast_to_nat m, ← cast_to_nat n, Nat.cast_lt (α := α), lt_to_nat]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_le [LinearOrderedSemiring α] {m n : Num} : (m : α) ≤ n ↔ m ≤ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    m n : Num
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr cast_lt
                 /-
                   🎉 no goals
                 -/


@[simp, norm_cast]
theorem cast_inj [LinearOrderedSemiring α] {m n : Num} : (m : α) = n ↔ m = n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemiring α
    m n : Num
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  rw [← cast_to_nat m, ← cast_to_nat n, Nat.cast_inj, to_nat_inj]
  /-
    🎉 no goals
  -/


theorem le_iff_cmp {m n} : m ≤ n ↔ cmp m n ≠ Ordering.gt :=
                                      /-
                                        m n : Num
                                        ⊢ Iff (Eq (n.cmp m) Ordering.lt) (Eq (m.cmp n) Ordering.gt)
                                      -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  not_congr <| lt_iff_cmp.trans <| by rw [← cmp_swap]; cases cmp m n <;> decide
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem castNum_eq_bitwise {f : Num → Num → Num} {g : Bool → Bool → Bool}
    (p : PosNum → PosNum → Num)
    (gff : g false false = false) (f00 : f 0 0 = 0)
    (f0n : ∀ n, f 0 (pos n) = cond (g false true) (pos n) 0)
    (fn0 : ∀ n, f (pos n) 0 = cond (g true false) (pos n) 0)
    (fnn : ∀ m n, f (pos m) (pos n) = p m n) (p11 : p 1 1 = cond (g true true) 1 0)
    (p1b : ∀ b n, p 1 (PosNum.bit b n) = bit (g true b) (cond (g false true) (pos n) 0))
    (pb1 : ∀ a m, p (PosNum.bit a m) 1 = bit (g a true) (cond (g true false) (pos m) 0))
    (pbb : ∀ a b m n, p (PosNum.bit a m) (PosNum.bit b n) = bit (g a b) (p m n)) :
    ∀ m n : Num, (f m n : ℕ) = Nat.bitwise g m n := by
  /-
    f : Num → Num → Num
    g : Bool → Bool → Bool
    p : PosNum → PosNum → Num
    gff : Eq (g Bool.false Bool.false) Bool.false
    f00 : Eq (f 0 0) 0
    f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
    fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
    fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
    p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
    p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
    pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
    pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
    ⊢ ∀ (m n : Num), Eq (↑(f m n)) (Nat.bitwise g ↑m ↑n)
  -/
  intros m n
  /-
    f : Num → Num → Num
    g : Bool → Bool → Bool
    p : PosNum → PosNum → Num
    gff : Eq (g Bool.false Bool.false) Bool.false
    f00 : Eq (f 0 0) 0
    f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
    fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
    fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
    p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
    p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
    pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
    pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
    m n : Num
    ⊢ Eq (↑(f m n)) (Nat.bitwise g ↑m ↑n)
  -/
  cases' m with m <;> cases' n with n <;>
      /-
        case zero.zero
        f : Num → Num → Num
        g : Bool → Bool → Bool
        p : PosNum → PosNum → Num
        gff : Eq (g Bool.false Bool.false) Bool.false
        f00 : Eq (f 0 0) 0
        f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
        fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
        fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
        p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
        p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
        pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
        pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
        ⊢ Eq (↑(f Num.zero Num.zero)) (Nat.bitwise g ↑Num.zero ↑Num.zero)
      -/
      try simp only [show zero = 0 from rfl, show ((0 : Num) : ℕ) = 0 from rfl]
    /-
      case zero.zero
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      ⊢ Eq (↑(f 0 0)) (Nat.bitwise g 0 0)
    -/
  · rw [f00, Nat.bitwise_zero]; rfl
                                /-
                                  🎉 no goals
                                -/
    /-
      case zero.pos
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      n : PosNum
      ⊢ Eq (↑(f 0 (Num.pos n))) (Nat.bitwise g 0 ↑(Num.pos n))
    -/
  · rw [f0n, Nat.bitwise_zero_left]
    /-
      case zero.pos
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      n : PosNum
      ⊢ Eq (↑(cond (g Bool.false Bool.true) (Num.pos n) 0)) (ite (Eq (g Bool.false B …
    -/
                           /-
                             🎉 no goals
                           -/
    cases g false true <;> rfl
                           /-
                             🎉 no goals
                           -/
    /-
      case pos.zero
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      m : PosNum
      ⊢ Eq (↑(f (Num.pos m) 0)) (Nat.bitwise g (↑(Num.pos m)) 0)
    -/
  · rw [fn0, Nat.bitwise_zero_right]
    /-
      case pos.zero
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      m : PosNum
      ⊢ Eq (↑(cond (g Bool.true Bool.false) (Num.pos m) 0)) (ite (Eq (g Bool.true Bo …
    -/
                           /-
                             🎉 no goals
                           -/
    cases g true false <;> rfl
                           /-
                             🎉 no goals
                           -/
    /-
      case pos.pos
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      m n : PosNum
      ⊢ Eq (↑(f (Num.pos m) (Num.pos n))) (Nat.bitwise g ↑(Num.pos m) ↑(Num.pos n))
    -/
  · rw [fnn]
    have : ∀ (b) (n : PosNum), (cond b (↑n) 0 : ℕ) = ↑(cond b (pos n) 0 : Num) := by
      intros b _; cases b <;> rfl
    /-
      case pos.pos
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      m n : PosNum
      this : ∀ (b : Bool) (n : PosNum), Eq (cond b (↑n) 0) ↑(cond b (Num.pos n) 0)
      ⊢ Eq (↑(p m n)) (Nat.bitwise g ↑(Num.pos m) ↑(Num.pos n))
    -/
    induction' m with m IH m IH generalizing n <;> cases' n with n n
    any_goals simp only [show one = 1 from rfl, show pos 1 = 1 from rfl,
      show PosNum.bit0 = PosNum.bit false from rfl, show PosNum.bit1 = PosNum.bit true from rfl,
      show ((1 : Num) : ℕ) = Nat.bit true 0 from rfl]
    all_goals
      repeat
        rw [show ∀ b n, (pos (PosNum.bit b n) : ℕ) = Nat.bit b ↑n by
          intros b _; cases b <;> simp_all]
      rw [Nat.bitwise_bit gff]
    /-
      case pos.pos.one.one
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      this : ∀ (b : Bool) (n : PosNum), Eq (cond b (↑n) 0) ↑(cond b (Num.pos n) 0)
      ⊢ Eq (↑(p 1 1)) (Nat.bit (g Bool.true Bool.true) (Nat.bitwise g 0 0))
    -/
    any_goals rw [Nat.bitwise_zero, p11]; cases g true true <;> rfl
    /-
      case pos.pos.one.bit1
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      this : ∀ (b : Bool) (n : PosNum), Eq (cond b (↑n) 0) ↑(cond b (Num.pos n) 0)
      n : PosNum
      ⊢ Eq (↑(p 1 (PosNum.bit Bool.true n))) (Nat.bit (g Bool.true Bool.true) (Nat.b …
    -/
    any_goals rw [Nat.bitwise_zero_left, ← Bool.cond_eq_ite, this, ← bit_to_nat, p1b]
    /-
      case pos.pos.bit1.one
      f : Num → Num → Num
      g : Bool → Bool → Bool
      p : PosNum → PosNum → Num
      gff : Eq (g Bool.false Bool.false) Bool.false
      f00 : Eq (f 0 0) 0
      f0n : ∀ (n : PosNum), Eq (f 0 (Num.pos n)) (cond (g Bool.false Bool.true) (Num …
      fn0 : ∀ (n : PosNum), Eq (f (Num.pos n) 0) (cond (g Bool.true Bool.false) (Num …
      fnn : ∀ (m n : PosNum), Eq (f (Num.pos m) (Num.pos n)) (p m n)
      p11 : Eq (p 1 1) (cond (g Bool.true Bool.true) 1 0)
      p1b : ∀ (b : Bool) (n : PosNum), Eq (p 1 (PosNum.bit b n)) (Num.bit (g Bool.tr …
      pb1 : ∀ (a : Bool) (m : PosNum), Eq (p (PosNum.bit a m) 1) (Num.bit (g a Bool. …
      pbb : ∀ (a b : Bool) (m n : PosNum), Eq (p (PosNum.bit a m) (PosNum.bit b n))  …
      this : ∀ (b : Bool) (n : PosNum), Eq (cond b (↑n) 0) ↑(cond b (Num.pos n) 0)
      m : PosNum
      IH : ∀ (n : PosNum), Eq (↑(p m n)) (Nat.bitwise g ↑(Num.pos m) ↑(Num.pos n))
      ⊢ Eq (↑(p (PosNum.bit Bool.true m) 1)) (Nat.bit (g Bool.true Bool.true) (Nat.b …
    -/
    any_goals rw [Nat.bitwise_zero_right, ← Bool.cond_eq_ite, this, ← bit_to_nat, pb1]
    all_goals
      rw [← show ∀ n : PosNum, ↑(p m n) = Nat.bitwise g ↑m ↑n from IH]
      rw [← bit_to_nat, pbb]


@[simp, norm_cast]
theorem castNum_or : ∀ m n : Num, ↑(m ||| n) = (↑m ||| ↑n : ℕ) := by
  -- Porting note: A name of an implicit local hypothesis is not available so
  --               `cases_type*` is used.
  /-
    ⊢ ∀ (m n : Num), Eq (↑(HOr.hOr m n)) (HOr.hOr ↑m ↑n)
  -/
  apply castNum_eq_bitwise fun x y => pos (PosNum.lor x y) <;>
   /-
     case gff
     ⊢ Eq (Bool.false.or Bool.false) Bool.false
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
   intros <;> (try cases_type* Bool) <;> rfl
                                         /-
                                           🎉 no goals
                                         -/


@[simp, norm_cast]
theorem castNum_and : ∀ m n : Num, ↑(m &&& n) = (↑m &&& ↑n : ℕ) := by
  /-
    ⊢ ∀ (m n : Num), Eq (↑(HAnd.hAnd m n)) (HAnd.hAnd ↑m ↑n)
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
  apply castNum_eq_bitwise PosNum.land <;> intros <;> (try cases_type* Bool) <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp, norm_cast]
theorem castNum_ldiff : ∀ m n : Num, (ldiff m n : ℕ) = Nat.ldiff m n := by
  /-
    ⊢ ∀ (m n : Num), Eq (↑(m.ldiff n)) ((↑m).ldiff ↑n)
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
  apply castNum_eq_bitwise PosNum.ldiff <;> intros <;> (try cases_type* Bool) <;> rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp, norm_cast]
theorem castNum_xor : ∀ m n : Num, ↑(m ^^^ n) = (↑m ^^^ ↑n : ℕ) := by
  /-
    ⊢ ∀ (m n : Num), Eq (↑(HXor.hXor m n)) (HXor.hXor ↑m ↑n)
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
  apply castNum_eq_bitwise PosNum.lxor <;> intros <;> (try cases_type* Bool) <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp, norm_cast]
theorem castNum_shiftLeft (m : Num) (n : Nat) : ↑(m <<< n) = (m : ℕ) <<< (n : ℕ) := by
  /-
    m : Num
    n : Nat
    ⊢ Eq (↑(HShiftLeft.hShiftLeft m n)) (HShiftLeft.hShiftLeft (↑m) n)
  -/
  cases m <;> dsimp only [← shiftl_eq_shiftLeft, shiftl]
    /-
      case zero
      n : Nat
      ⊢ Eq (↑0) (HShiftLeft.hShiftLeft (↑Num.zero) n)
    -/
  · symm
    /-
      case zero
      n : Nat
      ⊢ Eq (HShiftLeft.hShiftLeft (↑Num.zero) n) ↑0
    -/
    apply Nat.zero_shiftLeft
    /-
      🎉 no goals
    -/
  /-
    case pos
    n : Nat
    a✝ : PosNum
    ⊢ Eq (↑(Num.pos (HShiftLeft.hShiftLeft a✝ n))) (HShiftLeft.hShiftLeft (↑(Num.p …
  -/
  simp only [cast_pos]
  /-
    case pos
    n : Nat
    a✝ : PosNum
    ⊢ Eq (↑(HShiftLeft.hShiftLeft a✝ n)) (HShiftLeft.hShiftLeft (↑a✝) n)
  -/
  induction' n with n IH
    /-
      case pos.zero
      a✝ : PosNum
      ⊢ Eq (↑(HShiftLeft.hShiftLeft a✝ 0)) (HShiftLeft.hShiftLeft (↑a✝) 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  simp [PosNum.shiftl_succ_eq_bit0_shiftl, Nat.shiftLeft_succ, IH, pow_succ, ← mul_assoc, mul_comm,
        -shiftl_eq_shiftLeft, -PosNum.shiftl_eq_shiftLeft, shiftl, mul_two]


@[simp, norm_cast]
theorem castNum_shiftRight (m : Num) (n : Nat) : ↑(m >>> n) = (m : ℕ) >>> (n : ℕ) := by
  /-
    m : Num
    n : Nat
    ⊢ Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRight (↑m) n)
  -/
  cases' m with m <;> dsimp only [← shiftr_eq_shiftRight, shiftr]
    /-
      case zero
      n : Nat
      ⊢ Eq (↑0) (HShiftRight.hShiftRight (↑Num.zero) n)
    -/
  · symm
    /-
      case zero
      n : Nat
      ⊢ Eq (HShiftRight.hShiftRight (↑Num.zero) n) ↑0
    -/
    apply Nat.zero_shiftRight
    /-
      🎉 no goals
    -/
  /-
    case pos
    n : Nat
    m : PosNum
    ⊢ Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRight (↑(Num.pos m)) n)
  -/
  induction' n with n IH generalizing m
    /-
      case pos.zero
      m : PosNum
      ⊢ Eq (↑(HShiftRight.hShiftRight m 0)) (HShiftRight.hShiftRight (↑(Num.pos m)) 0)
    -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  · cases m <;> rfl
                /-
                  🎉 no goals
                -/
  /-
    case pos.succ
    n : Nat
    IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
    m : PosNum
    ⊢ Eq (↑(HShiftRight.hShiftRight m (HAdd.hAdd n 1))) (HShiftRight.hShiftRight ( …
  -/
  have hdiv2 : ∀ m, Nat.div2 (m + m) = m := by intro; rw [Nat.div2_val]; omega
  /-
    case pos.succ
    n : Nat
    IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
    m : PosNum
    hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
    ⊢ Eq (↑(HShiftRight.hShiftRight m (HAdd.hAdd n 1))) (HShiftRight.hShiftRight ( …
  -/
  cases' m with m m <;> dsimp only [PosNum.shiftr, ← PosNum.shiftr_eq_shiftRight]
    /-
      case pos.succ.one
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      ⊢ Eq (↑0) (HShiftRight.hShiftRight (↑(Num.pos PosNum.one)) (HAdd.hAdd n 1))
    -/
  · rw [Nat.shiftRight_eq_div_pow]
    /-
      case pos.succ.one
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      ⊢ Eq (↑0) (HDiv.hDiv (↑(Num.pos PosNum.one)) (HPow.hPow 2 (HAdd.hAdd n 1)))
    -/
    symm
    /-
      case pos.succ.one
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      ⊢ Eq (HDiv.hDiv (↑(Num.pos PosNum.one)) (HPow.hPow 2 (HAdd.hAdd n 1))) ↑0
    -/
    apply Nat.div_eq_of_lt
    /-
      case pos.succ.one.h₀
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      ⊢ LT.lt (↑(Num.pos PosNum.one)) (HPow.hPow 2 (HAdd.hAdd n 1))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case pos.succ.bit1
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (↑(m.shiftr n)) (HShiftRight.hShiftRight (↑(Num.pos m.bit1)) (HAdd.hAdd n …
    -/
  · trans
      /-
        n : Nat
        IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
        hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
        m : PosNum
        ⊢ Eq ↑(m.shiftr n) ?m.314125
      -/
    · apply IH
      /-
        🎉 no goals
      -/
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (HShiftRight.hShiftRight (↑(Num.pos m)) n) (HShiftRight.hShiftRight (↑(Nu …
    -/
    change Nat.shiftRight m n = Nat.shiftRight (m + m + 1) (n + 1)
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq ((↑m).shiftRight n) ((HAdd.hAdd (HAdd.hAdd ↑m ↑m) 1).shiftRight (HAdd.hAd …
    -/
    rw [add_comm n 1, @Nat.shiftRight_eq _ (1 + n), Nat.shiftRight_add]
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq ((↑m).shiftRight n) (HShiftRight.hShiftRight (HShiftRight.hShiftRight (HA …
    -/
    apply congr_arg fun x => Nat.shiftRight x n
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (↑m) (HShiftRight.hShiftRight (HAdd.hAdd (HAdd.hAdd ↑m ↑m) 1) 1)
    -/
    simp [-add_assoc, Nat.shiftRight_succ, Nat.shiftRight_zero, ← Nat.div2_val, hdiv2]
    /-
      🎉 no goals
    -/
    /-
      case pos.succ.bit0
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (↑(m.shiftr n)) (HShiftRight.hShiftRight (↑(Num.pos m.bit0)) (HAdd.hAdd n …
    -/
  · trans
      /-
        n : Nat
        IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
        hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
        m : PosNum
        ⊢ Eq ↑(m.shiftr n) ?m.315600
      -/
    · apply IH
      /-
        🎉 no goals
      -/
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (HShiftRight.hShiftRight (↑(Num.pos m)) n) (HShiftRight.hShiftRight (↑(Nu …
    -/
    change Nat.shiftRight m n = Nat.shiftRight (m + m) (n + 1)
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq ((↑m).shiftRight n) ((HAdd.hAdd ↑m ↑m).shiftRight (HAdd.hAdd n 1))
    -/
    rw [add_comm n 1,  @Nat.shiftRight_eq _ (1 + n), Nat.shiftRight_add]
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq ((↑m).shiftRight n) (HShiftRight.hShiftRight (HShiftRight.hShiftRight (HA …
    -/
    apply congr_arg fun x => Nat.shiftRight x n
    /-
      n : Nat
      IH : ∀ (m : PosNum), Eq (↑(HShiftRight.hShiftRight m n)) (HShiftRight.hShiftRi …
      hdiv2 : ∀ (m : Nat), Eq (HAdd.hAdd m m).div2 m
      m : PosNum
      ⊢ Eq (↑m) (HShiftRight.hShiftRight (HAdd.hAdd ↑m ↑m) 1)
    -/
    simp [-add_assoc, Nat.shiftRight_succ, Nat.shiftRight_zero, ← Nat.div2_val, hdiv2]
    /-
      🎉 no goals
    -/


@[simp]
theorem castNum_testBit (m n) : testBit m n = Nat.testBit m n := by
  -- Porting note: `unfold` → `dsimp only`
  cases m with dsimp only [testBit]
  | zero =>
    rw [show (Num.zero : Nat) = 0 from rfl, Nat.zero_testBit]
  | pos m =>
    rw [cast_pos]
    induction' n with n IH generalizing m <;> cases' m with m m
        <;> simp only [PosNum.testBit]
    · rfl
    · rw [PosNum.cast_bit1, ← two_mul, ← congr_fun Nat.bit_true, Nat.testBit_bit_zero]
    · rw [PosNum.cast_bit0, ← two_mul, ← congr_fun Nat.bit_false, Nat.testBit_bit_zero]
    · simp [Nat.testBit_add_one]
    · rw [PosNum.cast_bit1, ← two_mul, ← congr_fun Nat.bit_true, Nat.testBit_bit_succ, IH]
    · rw [PosNum.cast_bit0, ← two_mul, ← congr_fun Nat.bit_false, Nat.testBit_bit_succ, IH]


@[simp, norm_cast]
theorem cast_zero [Zero α] [One α] [Add α] [Neg α] : ((0 : ZNum) : α) = 0 :=
  rfl


@[simp]
theorem cast_zero' [Zero α] [One α] [Add α] [Neg α] : (ZNum.zero : α) = 0 :=
  rfl


@[simp, norm_cast]
theorem cast_one [Zero α] [One α] [Add α] [Neg α] : ((1 : ZNum) : α) = 1 :=
  rfl


@[simp]
theorem cast_pos [Zero α] [One α] [Add α] [Neg α] (n : PosNum) : (pos n : α) = n :=
  rfl


@[simp]
theorem cast_neg [Zero α] [One α] [Add α] [Neg α] (n : PosNum) : (neg n : α) = -n :=
  rfl


@[simp, norm_cast]
theorem cast_zneg [AddGroup α] [One α] : ∀ n, ((-n : ZNum) : α) = -n
  | 0 => neg_zero.symm
  | pos _p => rfl
  | neg _p => (neg_neg _).symm


theorem neg_zero : (-0 : ZNum) = 0 :=
  rfl


theorem zneg_pos (n : PosNum) : -pos n = neg n :=
  rfl


theorem zneg_neg (n : PosNum) : -neg n = pos n :=
  rfl


                                              /-
                                                n : ZNum
                                                ⊢ Eq (Neg.neg (Neg.neg n)) n
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
theorem zneg_zneg (n : ZNum) : - -n = n := by cases n <;> rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            n : ZNum
                                                            ⊢ Eq (Neg.neg n.bit1) (Neg.neg n).bitm1
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
theorem zneg_bit1 (n : ZNum) : -n.bit1 = (-n).bitm1 := by cases n <;> rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                           /-
                                                             n : ZNum
                                                             ⊢ Eq (Neg.neg n.bitm1) (Neg.neg n).bit1
                                                           -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
theorem zneg_bitm1 (n : ZNum) : -n.bitm1 = (-n).bit1 := by cases n <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem zneg_succ (n : ZNum) : -n.succ = (-n).pred := by
  /-
    n : ZNum
    ⊢ Eq (Neg.neg n.succ) (Neg.neg n).pred
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases n <;> try { rfl }; rw [succ, Num.zneg_toZNumNeg]; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem zneg_pred (n : ZNum) : -n.pred = (-n).succ := by
  /-
    n : ZNum
    ⊢ Eq (Neg.neg n.pred) (Neg.neg n).succ
  -/
  rw [← zneg_zneg (succ (-n)), zneg_succ, zneg_zneg]
  /-
    🎉 no goals
  -/


@[simp]
theorem abs_to_nat : ∀ n, (abs n : ℕ) = Int.natAbs n
  | 0 => rfl
  | pos p => congr_arg Int.natAbs p.to_nat_to_int
                                                                /-
                                                                  p : PosNum
                                                                  ⊢ Eq (↑↑p).natAbs (Neg.neg ↑p).natAbs
                                                                -/
  | neg p => show Int.natAbs ((p : ℕ) : ℤ) = Int.natAbs (-p) by rw [p.to_nat_to_int, Int.natAbs_neg]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem abs_toZNum : ∀ n : Num, abs n.toZNum = n
  | 0 => rfl
  | Num.pos _p => rfl


@[simp, norm_cast]
theorem cast_to_int [AddGroupWithOne α] : ∀ n : ZNum, ((n : ℤ) : α) = n
            /-
              α : Type u_1
              inst✝ : AddGroupWithOne α
              ⊢ Eq ↑↑0 ↑0
            -/
  | 0 => by rw [cast_zero, cast_zero, Int.cast_zero]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  inst✝ : AddGroupWithOne α
                  p : PosNum
                  ⊢ Eq ↑↑(ZNum.pos p) ↑(ZNum.pos p)
                -/
  | pos p => by rw [cast_pos, cast_pos, PosNum.cast_to_int]
                /-
                  🎉 no goals
                -/
                /-
                  α : Type u_1
                  inst✝ : AddGroupWithOne α
                  p : PosNum
                  ⊢ Eq ↑↑(ZNum.neg p) ↑(ZNum.neg p)
                -/
  | neg p => by rw [cast_neg, cast_neg, Int.cast_neg, PosNum.cast_to_int]
                /-
                  🎉 no goals
                -/


theorem bit0_of_bit0 : ∀ n : ZNum, n + n = n.bit0
  | 0 => rfl
  | pos a => congr_arg pos a.bit0_of_bit0
  | neg a => congr_arg neg a.bit0_of_bit0


theorem bit1_of_bit1 : ∀ n : ZNum, n + n + 1 = n.bit1
  | 0 => rfl
  | pos a => congr_arg pos a.bit1_of_bit1
                                               /-
                                                 a : PosNum
                                                 ⊢ Eq (PosNum.sub' 1 (HAdd.hAdd a a)) (ZNum.neg a).bit1
                                               -/
  | neg a => show PosNum.sub' 1 (a + a) = _ by rw [PosNum.one_sub', a.bit0_of_bit0]; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp, norm_cast]
theorem cast_bit0 [AddGroupWithOne α] : ∀ n : ZNum, (n.bit0 : α) = (n : α) + n
  | 0 => (add_zero _).symm
                /-
                  α : Type u_1
                  inst✝ : AddGroupWithOne α
                  p : PosNum
                  ⊢ Eq (↑(ZNum.pos p).bit0) (HAdd.hAdd ↑(ZNum.pos p) ↑(ZNum.pos p))
                -/
  | pos p => by rw [ZNum.bit0, cast_pos, cast_pos]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  | neg p => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      p : PosNum
      ⊢ Eq (↑(ZNum.neg p).bit0) (HAdd.hAdd ↑(ZNum.neg p) ↑(ZNum.neg p))
    -/
    rw [ZNum.bit0, cast_neg, cast_neg, PosNum.cast_bit0, neg_add_rev]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_bit1 [AddGroupWithOne α] : ∀ n : ZNum, (n.bit1 : α) = ((n : α) + n) + 1
            /-
              α : Type u_1
              inst✝ : AddGroupWithOne α
              ⊢ Eq (↑(ZNum.bit1 0)) (HAdd.hAdd (HAdd.hAdd ↑0 ↑0) 1)
            -/
  | 0 => by simp [ZNum.bit1]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  inst✝ : AddGroupWithOne α
                  p : PosNum
                  ⊢ Eq (↑(ZNum.pos p).bit1) (HAdd.hAdd (HAdd.hAdd ↑(ZNum.pos p) ↑(ZNum.pos p)) 1)
                -/
  | pos p => by rw [ZNum.bit1, cast_pos, cast_pos]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  | neg p => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      p : PosNum
      ⊢ Eq (↑(ZNum.neg p).bit1) (HAdd.hAdd (HAdd.hAdd ↑(ZNum.neg p) ↑(ZNum.neg p)) 1)
    -/
    rw [ZNum.bit1, cast_neg, cast_neg]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      p : PosNum
      ⊢ Eq (Neg.neg ↑(Num.casesOn p.pred' 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd (Neg …
    -/
    cases' e : pred' p with a <;>
      /-
        case zero
        α : Type u_1
        inst✝ : AddGroupWithOne α
        p : PosNum
        e : Eq p.pred' Num.zero
        ⊢ Eq (Neg.neg ↑(Num.casesOn Num.zero 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd (Ne …
      -/
      have ep : p = _ := (succ'_pred' p).symm.trans (congr_arg Num.succ' e)
      /-
        case zero
        α : Type u_1
        inst✝ : AddGroupWithOne α
        p : PosNum
        e : Eq p.pred' Num.zero
        ep : Eq p Num.zero.succ'
        ⊢ Eq (Neg.neg ↑(Num.casesOn Num.zero 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd (Ne …
      -/
    · conv at ep => change p = 1
      /-
        case zero
        α : Type u_1
        inst✝ : AddGroupWithOne α
        p : PosNum
        e : Eq p.pred' Num.zero
        ep : Eq p 1
        ⊢ Eq (Neg.neg ↑(Num.casesOn Num.zero 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd (Ne …
      -/
      subst p
      /-
        case zero
        α : Type u_1
        inst✝ : AddGroupWithOne α
        e : Eq (PosNum.pred' 1) Num.zero
        ⊢ Eq (Neg.neg ↑(Num.casesOn Num.zero 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd (Ne …
      -/
      simp
      /-
        🎉 no goals
      -/
    -- Porting note: `rw [Num.succ']` yields a `match` pattern.
      /-
        case pos
        α : Type u_1
        inst✝ : AddGroupWithOne α
        p a : PosNum
        e : Eq p.pred' (Num.pos a)
        ep : Eq p (Num.pos a).succ'
        ⊢ Eq (Neg.neg ↑(Num.casesOn (Num.pos a) 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd  …
      -/
    · dsimp only [Num.succ'] at ep
      /-
        case pos
        α : Type u_1
        inst✝ : AddGroupWithOne α
        p a : PosNum
        e : Eq p.pred' (Num.pos a)
        ep : Eq p a.succ
        ⊢ Eq (Neg.neg ↑(Num.casesOn (Num.pos a) 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd  …
      -/
      subst p
      /-
        case pos
        α : Type u_1
        inst✝ : AddGroupWithOne α
        a : PosNum
        e : Eq a.succ.pred' (Num.pos a)
        ⊢ Eq (Neg.neg ↑(Num.casesOn (Num.pos a) 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd  …
      -/
      have : (↑(-↑a : ℤ) : α) = -1 + ↑(-↑a + 1 : ℤ) := by simp [add_comm (- ↑a : ℤ) 1]
      /-
        case pos
        α : Type u_1
        inst✝ : AddGroupWithOne α
        a : PosNum
        e : Eq a.succ.pred' (Num.pos a)
        this : Eq (↑(Neg.neg ↑a)) (HAdd.hAdd (-1) ↑(HAdd.hAdd (Neg.neg ↑a) 1))
        ⊢ Eq (Neg.neg ↑(Num.casesOn (Num.pos a) 1 PosNum.bit1)) (HAdd.hAdd (HAdd.hAdd  …
      -/
      simpa using this
      /-
        🎉 no goals
      -/


@[simp]
theorem cast_bitm1 [AddGroupWithOne α] (n : ZNum) : (n.bitm1 : α) = (n : α) + n - 1 := by
  conv =>
    lhs
    rw [← zneg_zneg n]
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : ZNum
    ⊢ Eq (↑(Neg.neg (Neg.neg n)).bitm1) (HSub.hSub (HAdd.hAdd ↑n ↑n) 1)
  -/
  rw [← zneg_bit1, cast_zneg, cast_bit1]
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : ZNum
    ⊢ Eq (Neg.neg (HAdd.hAdd (HAdd.hAdd ↑(Neg.neg n) ↑(Neg.neg n)) 1)) (HSub.hSub  …
  -/
  have : ((-1 + n + n : ℤ) : α) = (n + n + -1 : ℤ) := by simp [add_comm, add_left_comm]
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : ZNum
    this : Eq ↑(HAdd.hAdd (HAdd.hAdd (-1) ↑n) ↑n) ↑(HAdd.hAdd (HAdd.hAdd ↑n ↑n) (- …
    ⊢ Eq (Neg.neg (HAdd.hAdd (HAdd.hAdd ↑(Neg.neg n) ↑(Neg.neg n)) 1)) (HSub.hSub  …
  -/
  simpa [sub_eq_add_neg] using this
  /-
    🎉 no goals
  -/


                                              /-
                                                n : ZNum
                                                ⊢ Eq (HAdd.hAdd n 0) n
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
theorem add_zero (n : ZNum) : n + 0 = n := by cases n <;> rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


                                              /-
                                                n : ZNum
                                                ⊢ Eq (HAdd.hAdd 0 n) n
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
theorem zero_add (n : ZNum) : 0 + n = n := by cases n <;> rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem add_one : ∀ n : ZNum, n + 1 = succ n
  | 0 => rfl
  | pos p => congr_arg pos p.add_one
                /-
                  p : PosNum
                  ⊢ Eq (HAdd.hAdd (ZNum.neg p) 1) (ZNum.neg p).succ
                -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
  | neg p => by cases p <;> rfl
                            /-
                              🎉 no goals
                            -/


theorem cast_to_znum : ∀ n : PosNum, (n : ZNum) = ZNum.pos n
  | 1 => rfl
  | bit0 p => by
      /-
        p : PosNum
        ⊢ Eq (↑p.bit0) (ZNum.pos p.bit0)
      -/
      have := congr_arg ZNum.bit0 (cast_to_znum p)
      /-
        p : PosNum
        this : Eq (↑p).bit0 (ZNum.pos p).bit0
        ⊢ Eq (↑p.bit0) (ZNum.pos p.bit0)
      -/
      rwa [← ZNum.bit0_of_bit0] at this
      /-
        🎉 no goals
      -/
  | bit1 p => by
      /-
        p : PosNum
        ⊢ Eq (↑p.bit1) (ZNum.pos p.bit1)
      -/
      have := congr_arg ZNum.bit1 (cast_to_znum p)
      /-
        p : PosNum
        this : Eq (↑p).bit1 (ZNum.pos p).bit1
        ⊢ Eq (↑p.bit1) (ZNum.pos p.bit1)
      -/
      rwa [← ZNum.bit1_of_bit1] at this
      /-
        🎉 no goals
      -/


theorem cast_sub' [AddGroupWithOne α] : ∀ m n : PosNum, (sub' m n : α) = m - n
  | a, 1 => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a : PosNum
      ⊢ Eq (↑(a.sub' 1)) (HSub.hSub ↑a ↑1)
    -/
    rw [sub'_one, Num.cast_toZNum, ← Num.cast_to_nat, pred'_to_nat, ← Nat.sub_one]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a : PosNum
      ⊢ Eq (↑(HSub.hSub (↑a) 1)) (HSub.hSub ↑a ↑1)
    -/
    simp [PosNum.cast_pos]
    /-
      🎉 no goals
    -/
  | 1, b => by
    rw [one_sub', Num.cast_toZNumNeg, ← neg_sub, neg_inj, ← Num.cast_to_nat, pred'_to_nat,
        ← Nat.sub_one]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      b : PosNum
      ⊢ Eq (↑(HSub.hSub (↑b) 1)) (HSub.hSub ↑b ↑1)
    -/
    simp [PosNum.cast_pos]
    /-
      🎉 no goals
    -/
  | bit0 a, bit0 b => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (↑(a.bit0.sub' b.bit0)) (HSub.hSub ↑a.bit0 ↑b.bit0)
    -/
    rw [sub', ZNum.cast_bit0, cast_sub' a b]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) (HSub.hSub ↑a.bit0 ↑b.bit0)
    -/
    have : ((a + -b + (a + -b) : ℤ) : α) = a + a + (-b + -b) := by simp [add_left_comm]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      this : Eq (↑(HAdd.hAdd (HAdd.hAdd (↑a) (Neg.neg ↑b)) (HAdd.hAdd (↑a) (Neg.neg  …
      ⊢ Eq (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) (HSub.hSub ↑a.bit0 ↑b.bit0)
    -/
    simpa [sub_eq_add_neg] using this
    /-
      🎉 no goals
    -/
  | bit0 a, bit1 b => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (↑(a.bit0.sub' b.bit1)) (HSub.hSub ↑a.bit0 ↑b.bit1)
    -/
    rw [sub', ZNum.cast_bitm1, cast_sub' a b]
    have : ((-b + (a + (-b + -1)) : ℤ) : α) = (a + -1 + (-b + -b) : ℤ) := by
      simp [add_comm, add_left_comm]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      this : Eq ↑(HAdd.hAdd (Neg.neg ↑b) (HAdd.hAdd (↑a) (HAdd.hAdd (Neg.neg ↑b) (-1 …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) 1) (HSub.hSub  …
    -/
    simpa [sub_eq_add_neg] using this
    /-
      🎉 no goals
    -/
  | bit1 a, bit0 b => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (↑(a.bit1.sub' b.bit0)) (HSub.hSub ↑a.bit1 ↑b.bit0)
    -/
    rw [sub', ZNum.cast_bit1, cast_sub' a b]
    have : ((-b + (a + (-b + 1)) : ℤ) : α) = (a + 1 + (-b + -b) : ℤ) := by
      simp [add_comm, add_left_comm]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      this : Eq ↑(HAdd.hAdd (Neg.neg ↑b) (HAdd.hAdd (↑a) (HAdd.hAdd (Neg.neg ↑b) 1)) …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) 1) (HSub.hSub  …
    -/
    simpa [sub_eq_add_neg] using this
    /-
      🎉 no goals
    -/
  | bit1 a, bit1 b => by
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (↑(a.bit1.sub' b.bit1)) (HSub.hSub ↑a.bit1 ↑b.bit1)
    -/
    rw [sub', ZNum.cast_bit0, cast_sub' a b]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      ⊢ Eq (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) (HSub.hSub ↑a.bit1 ↑b.bit1)
    -/
    have : ((-b + (a + -b) : ℤ) : α) = a + (-b + -b) := by simp [add_left_comm]
    /-
      α : Type u_1
      inst✝ : AddGroupWithOne α
      a b : PosNum
      this : Eq (↑(HAdd.hAdd (Neg.neg ↑b) (HAdd.hAdd (↑a) (Neg.neg ↑b)))) (HAdd.hAdd …
      ⊢ Eq (HAdd.hAdd (HSub.hSub ↑a ↑b) (HSub.hSub ↑a ↑b)) (HSub.hSub ↑a.bit1 ↑b.bit1)
    -/
    simpa [sub_eq_add_neg] using this
    /-
      🎉 no goals
    -/


theorem to_nat_eq_succ_pred (n : PosNum) : (n : ℕ) = n.pred' + 1 := by
  /-
    n : PosNum
    ⊢ Eq (↑n) (HAdd.hAdd (↑n.pred') 1)
  -/
  rw [← Num.succ'_to_nat, n.succ'_pred']
  /-
    🎉 no goals
  -/


theorem to_int_eq_succ_pred (n : PosNum) : (n : ℤ) = (n.pred' : ℕ) + 1 := by
  /-
    n : PosNum
    ⊢ Eq (↑n) (HAdd.hAdd (↑↑n.pred') 1)
  -/
  rw [← n.to_nat_to_int, to_nat_eq_succ_pred]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem cast_sub' [AddGroupWithOne α] : ∀ m n : Num, (sub' m n : α) = m - n
  | 0, 0 => (sub_zero _).symm
  | pos _a, 0 => (sub_zero _).symm
  | 0, pos _b => (zero_sub _).symm
  | pos _a, pos _b => PosNum.cast_sub' _ _


theorem toZNum_succ : ∀ n : Num, n.succ.toZNum = n.toZNum.succ
  | 0 => rfl
  | pos _n => rfl


theorem toZNumNeg_succ : ∀ n : Num, n.succ.toZNumNeg = n.toZNumNeg.pred
  | 0 => rfl
  | pos _n => rfl


@[simp]
theorem pred_succ : ∀ n : ZNum, n.pred.succ = n
  | 0 => rfl
                                                            /-
                                                              p : PosNum
                                                              ⊢ Eq (Num.pos p).succ'.pred'.toZNumNeg (ZNum.neg p)
                                                            -/
  | ZNum.neg p => show toZNumNeg (pos p).succ'.pred' = _ by rw [PosNum.pred'_succ']; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                     /-
                       p : PosNum
                       ⊢ Eq (ZNum.pos p).pred.succ (ZNum.pos p)
                     -/
  | ZNum.pos p => by rw [ZNum.pred, ← toZNum_succ, Num.succ, PosNum.succ'_pred', toZNum]
                     /-
                       🎉 no goals
                     -/

-- Porting note: `erw [ZNum.ofInt', ZNum.ofInt']` yields `match` so
--               `change` & `dsimp` are required.

theorem succ_ofInt' : ∀ n, ZNum.ofInt' (n + 1) = ZNum.ofInt' n + 1
  | (n : ℕ) => by
    /-
      n : Nat
      ⊢ Eq (ZNum.ofInt' (HAdd.hAdd (↑n) 1)) (HAdd.hAdd (ZNum.ofInt' ↑n) 1)
    -/
    change ZNum.ofInt' (n + 1 : ℕ) = ZNum.ofInt' (n : ℕ) + 1
    /-
      n : Nat
      ⊢ Eq (ZNum.ofInt' ↑(HAdd.hAdd n 1)) (HAdd.hAdd (ZNum.ofInt' ↑n) 1)
    -/
    dsimp only [ZNum.ofInt', ZNum.ofInt']
    /-
      n : Nat
      ⊢ Eq (Num.ofNat' (HAdd.hAdd n 1)).toZNum (HAdd.hAdd (Num.ofNat' n).toZNum 1)
    -/
    rw [Num.ofNat'_succ, Num.add_one, toZNum_succ, ZNum.add_one]
    /-
      🎉 no goals
    -/
  | -[0+1] => by
    /-
      ⊢ Eq (ZNum.ofInt' (HAdd.hAdd (Int.negSucc 0) 1)) (HAdd.hAdd (ZNum.ofInt' (Int. …
    -/
    change ZNum.ofInt' 0 = ZNum.ofInt' (-[0+1]) + 1
    /-
      ⊢ Eq (ZNum.ofInt' 0) (HAdd.hAdd (ZNum.ofInt' (Int.negSucc 0)) 1)
    -/
    dsimp only [ZNum.ofInt', ZNum.ofInt']
    /-
      ⊢ Eq (Num.ofNat' 0).toZNum (HAdd.hAdd (Num.ofNat' (HAdd.hAdd 0 1)).toZNumNeg 1)
    -/
    rw [ofNat'_succ, ofNat'_zero]; rfl
                                   /-
                                     🎉 no goals
                                   -/
  | -[(n + 1)+1] => by
    /-
      n : Nat
      ⊢ Eq (ZNum.ofInt' (HAdd.hAdd (Int.negSucc (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (ZNu …
    -/
    change ZNum.ofInt' -[n+1] = ZNum.ofInt' -[(n + 1)+1] + 1
    /-
      n : Nat
      ⊢ Eq (ZNum.ofInt' (Int.negSucc n)) (HAdd.hAdd (ZNum.ofInt' (Int.negSucc (HAdd. …
    -/
    dsimp only [ZNum.ofInt', ZNum.ofInt']
    rw [@Num.ofNat'_succ (n + 1), Num.add_one, toZNumNeg_succ,
      @ofNat'_succ n, Num.add_one, ZNum.add_one, pred_succ]


theorem ofInt'_toZNum : ∀ n : ℕ, toZNum n = ZNum.ofInt' n
  | 0 => rfl
  | n + 1 => by
    rw [Nat.cast_succ, Num.add_one, toZNum_succ, ofInt'_toZNum n, Nat.cast_succ, succ_ofInt',
      ZNum.add_one]


theorem mem_ofZNum' : ∀ {m : Num} {n : ZNum}, m ∈ ofZNum' n ↔ n = toZNum m
  | 0, 0 => ⟨fun _ => rfl, fun _ => rfl⟩
  | pos _, 0 => ⟨nofun, nofun⟩
  | m, ZNum.pos p =>
                                /-
                                  m : Num
                                  p : PosNum
                                  ⊢ Iff (Eq (Num.pos p) m) (Eq (ZNum.pos p) m.toZNum)
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
    Option.some_inj.trans <| by cases m <;> constructor <;> intro h <;> try cases h <;> rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                         /-
                                           m : Num
                                           p : PosNum
                                           h : Eq (ZNum.neg p) m.toZNum
                                           ⊢ Membership.mem (Num.ofZNum' (ZNum.neg p)) m
                                         -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  | m, ZNum.neg p => ⟨nofun, fun h => by cases m <;> cases h⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem ofZNum'_toNat : ∀ n : ZNum, (↑) <$> ofZNum' n = Int.toNat' n
  | 0 => rfl
                                           /-
                                             p : PosNum
                                             ⊢ Eq (Functor.map castNum (Num.ofZNum' (ZNum.pos p))) (↑p).toNat'
                                           -/
  | ZNum.pos p => show _ = Int.toNat' p by rw [← PosNum.to_nat_to_int p]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  | ZNum.neg p =>
    (congr_arg fun x => Int.toNat' (-x)) <|
                                          /-
                                            p : PosNum
                                            ⊢ Eq ↑(HAdd.hAdd (↑p.pred') 1) ↑p
                                          -/
      show ((p.pred' + 1 : ℕ) : ℤ) = p by rw [← succ'_to_nat]; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem ofZNum_toNat : ∀ n : ZNum, (ofZNum n : ℕ) = Int.toNat n
  | 0 => rfl
                                          /-
                                            p : PosNum
                                            ⊢ Eq (↑(Num.ofZNum (ZNum.pos p))) (↑p).toNat
                                          -/
  | ZNum.pos p => show _ = Int.toNat p by rw [← PosNum.to_nat_to_int p]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  | ZNum.neg p =>
    (congr_arg fun x => Int.toNat (-x)) <|
                                          /-
                                            p : PosNum
                                            ⊢ Eq ↑(HAdd.hAdd (↑p.pred') 1) ↑p
                                          -/
      show ((p.pred' + 1 : ℕ) : ℤ) = p by rw [← succ'_to_nat]; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem cast_ofZNum [AddGroupWithOne α] (n : ZNum) : (ofZNum n : α) = Int.toNat n := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : ZNum
    ⊢ Eq ↑(Num.ofZNum n) ↑(↑n).toNat
  -/
  rw [← cast_to_nat, ofZNum_toNat]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem sub_to_nat (m n) : ((m - n : Num) : ℕ) = m - n :=
  show (ofZNum _ : ℕ) = _ by
    /-
      m n : Num
      ⊢ Eq (↑(Num.ofZNum (m.sub' n))) (HSub.hSub ↑m ↑n)
    -/
    rw [ofZNum_toNat, cast_sub', ← to_nat_to_int, ← to_nat_to_int, Int.toNat_sub]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_add [AddGroupWithOne α] : ∀ m n, ((m + n : ZNum) : α) = m + n
               /-
                 α : Type u_1
                 inst✝ : AddGroupWithOne α
                 a : ZNum
                 ⊢ Eq (↑(HAdd.hAdd 0 a)) (HAdd.hAdd ↑0 ↑a)
               -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
  | 0, a => by cases a <;> exact (_root_.zero_add _).symm
                           /-
                             🎉 no goals
                           -/
               /-
                 α : Type u_1
                 inst✝ : AddGroupWithOne α
                 b : ZNum
                 ⊢ Eq (↑(HAdd.hAdd b 0)) (HAdd.hAdd ↑b ↑0)
               -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
  | b, 0 => by cases b <;> exact (_root_.add_zero _).symm
                           /-
                             🎉 no goals
                           -/
  | pos _, pos _ => PosNum.cast_add _ _
                       /-
                         α : Type u_1
                         inst✝ : AddGroupWithOne α
                         a b : PosNum
                         ⊢ Eq (↑(HAdd.hAdd (ZNum.pos a) (ZNum.neg b))) (HAdd.hAdd ↑(ZNum.pos a) ↑(ZNum. …
                       -/
  | pos a, neg b => by simpa only [sub_eq_add_neg] using PosNum.cast_sub' (α := α) _ _
                       /-
                         🎉 no goals
                       -/
  | neg a, pos b =>
    have : (↑b + -↑a : α) = -↑a + ↑b := by
      /-
        α : Type u_1
        inst✝ : AddGroupWithOne α
        a b : PosNum
        ⊢ Eq (HAdd.hAdd (↑b) (Neg.neg ↑a)) (HAdd.hAdd (Neg.neg ↑a) ↑b)
      -/
      rw [← PosNum.cast_to_int a, ← PosNum.cast_to_int b, ← Int.cast_neg, ← Int.cast_add (-a)]
      /-
        α : Type u_1
        inst✝ : AddGroupWithOne α
        a b : PosNum
        ⊢ Eq (HAdd.hAdd ↑↑b ↑(Neg.neg ↑a)) ↑(HAdd.hAdd (Neg.neg ↑a) ↑b)
      -/
      simp [add_comm]
      /-
        🎉 no goals
      -/
    (PosNum.cast_sub' _ _).trans <| (sub_eq_add_neg _ _).trans this
  | neg a, neg b =>
    show -(↑(a + b) : α) = -a + -b by
      rw [PosNum.cast_add, neg_eq_iff_eq_neg, neg_add_rev, neg_neg, neg_neg,
          ← PosNum.cast_to_int a, ← PosNum.cast_to_int b, ← Int.cast_add, ← Int.cast_add, add_comm]


@[simp]
theorem cast_succ [AddGroupWithOne α] (n) : ((succ n : ZNum) : α) = n + 1 := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : ZNum
    ⊢ Eq (↑n.succ) (HAdd.hAdd (↑n) 1)
  -/
  rw [← add_one, cast_add, cast_one]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem mul_to_int : ∀ m n, ((m * n : ZNum) : ℤ) = m * n
               /-
                 a : ZNum
                 ⊢ Eq (↑(HMul.hMul 0 a)) (HMul.hMul ↑0 ↑a)
               -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
  | 0, a => by cases a <;> exact (zero_mul _).symm
                           /-
                             🎉 no goals
                           -/
               /-
                 b : ZNum
                 ⊢ Eq (↑(HMul.hMul b 0)) (HMul.hMul ↑b ↑0)
               -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
  | b, 0 => by cases b <;> exact (mul_zero _).symm
                           /-
                             🎉 no goals
                           -/
  | pos a, pos b => PosNum.cast_mul a b
                                                 /-
                                                   a b : PosNum
                                                   ⊢ Eq (Neg.neg ↑(HMul.hMul a b)) (HMul.hMul (↑a) (Neg.neg ↑b))
                                                 -/
  | pos a, neg b => show -↑(a * b) = ↑a * -↑b by rw [PosNum.cast_mul, neg_mul_eq_mul_neg]
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   a b : PosNum
                                                   ⊢ Eq (Neg.neg ↑(HMul.hMul a b)) (HMul.hMul (Neg.neg ↑a) ↑b)
                                                 -/
  | neg a, pos b => show -↑(a * b) = -↑a * ↑b by rw [PosNum.cast_mul, neg_mul_eq_neg_mul]
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   a b : PosNum
                                                   ⊢ Eq (↑(HMul.hMul a b)) (HMul.hMul (Neg.neg ↑a) (Neg.neg ↑b))
                                                 -/
  | neg a, neg b => show ↑(a * b) = -↑a * -↑b by rw [PosNum.cast_mul, neg_mul_neg]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem cast_mul [Ring α] (m n) : ((m * n : ZNum) : α) = m * n := by
  /-
    α : Type u_1
    inst✝ : Ring α
    m n : ZNum
    ⊢ Eq (↑(HMul.hMul m n)) (HMul.hMul ↑m ↑n)
  -/
  rw [← cast_to_int, mul_to_int, Int.cast_mul, cast_to_int, cast_to_int]
  /-
    🎉 no goals
  -/


theorem ofInt'_neg : ∀ n : ℤ, ofInt' (-n) = -ofInt' n
                                             /-
                                               n : Nat
                                               ⊢ Eq (ZNum.ofInt' ↑(HAdd.hAdd n 1)) (Neg.neg (ZNum.ofInt' (Int.negSucc n)))
                                             -/
  | -[n+1] => show ofInt' (n + 1 : ℕ) = _ by simp only [ofInt', Num.zneg_toZNumNeg]
                                             /-
                                               🎉 no goals
                                             -/
                                                                        /-
                                                                          ⊢ Eq (Num.ofNat' 0).toZNum (Neg.neg (Num.ofNat' 0).toZNum)
                                                                        -/
  | 0 => show Num.toZNum (Num.ofNat' 0) = -Num.toZNum (Num.ofNat' 0) by rw [Num.ofNat'_zero]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
                                                           /-
                                                             n : Nat
                                                             ⊢ Eq (Num.ofNat' (HAdd.hAdd n 1)).toZNumNeg (Neg.neg (Num.ofNat' (HAdd.hAdd n  …
                                                           -/
  | (n + 1 : ℕ) => show Num.toZNumNeg _ = -Num.toZNum _ by rw [Num.zneg_toZNum]
                                                           /-
                                                             🎉 no goals
                                                           -/

-- Porting note: `erw [ofInt']` yields `match` so `dsimp` is required.

theorem of_to_int' : ∀ n : ZNum, ZNum.ofInt' n = n
            /-
              ⊢ Eq (ZNum.ofInt' ↑0) 0
            -/
  | 0 => by dsimp [ofInt', cast_zero]; erw [Num.ofNat'_zero, Num.toZNum]
                                       /-
                                         🎉 no goals
                                       -/
                /-
                  a : PosNum
                  ⊢ Eq (ZNum.ofInt' ↑(ZNum.pos a)) (ZNum.pos a)
                -/
  | pos a => by rw [cast_pos, ← PosNum.cast_to_nat, ← Num.ofInt'_toZNum, PosNum.of_to_nat]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
  | neg a => by
    /-
      a : PosNum
      ⊢ Eq (ZNum.ofInt' ↑(ZNum.neg a)) (ZNum.neg a)
    -/
    rw [cast_neg, ofInt'_neg, ← PosNum.cast_to_nat, ← Num.ofInt'_toZNum, PosNum.of_to_nat]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem to_int_inj {m n : ZNum} : (m : ℤ) = n ↔ m = n :=
  ⟨fun h => Function.LeftInverse.injective of_to_int' h, congr_arg _⟩


theorem cmp_to_int : ∀ m n, (Ordering.casesOn (cmp m n) ((m : ℤ) < n) (m = n) ((n : ℤ) < m) : Prop)
  | 0, 0 => rfl
  | pos a, pos b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn ((ZNum.pos a).cmp (ZNum.pos b)) (LT.lt ↑(ZNum.pos a) ↑(ZNum …
    -/
    have := PosNum.cmp_to_nat a b; revert this; dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt ↑a ↑b) (Eq a b) (LT.lt ↑b ↑a) (a.cmp b) → Ordering.rec ( …
    -/
    cases PosNum.cmp a b <;> dsimp <;> [simp; exact congr_arg pos; simp [GT.gt]]
    /-
      🎉 no goals
    -/
  | neg a, neg b => by
    /-
      a b : PosNum
      ⊢ Ordering.casesOn ((ZNum.neg a).cmp (ZNum.neg b)) (LT.lt ↑(ZNum.neg a) ↑(ZNum …
    -/
    have := PosNum.cmp_to_nat b a; revert this; dsimp [cmp]
    /-
      a b : PosNum
      ⊢ Ordering.rec (LT.lt ↑b ↑a) (Eq b a) (LT.lt ↑a ↑b) (b.cmp a) → Ordering.rec ( …
    -/
    cases PosNum.cmp b a <;> dsimp <;> [simp; simp +contextual; simp [GT.gt]]
    /-
      🎉 no goals
    -/
  | pos _, 0 => PosNum.cast_pos _
  | pos _, neg _ => lt_trans (neg_lt_zero.2 <| PosNum.cast_pos _) (PosNum.cast_pos _)
  | 0, neg _ => neg_lt_zero.2 <| PosNum.cast_pos _
  | neg _, 0 => neg_lt_zero.2 <| PosNum.cast_pos _
  | neg _, pos _ => lt_trans (neg_lt_zero.2 <| PosNum.cast_pos _) (PosNum.cast_pos _)
  | 0, pos _ => PosNum.cast_pos _


@[norm_cast]
theorem lt_to_int {m n : ZNum} : (m : ℤ) < n ↔ m < n :=
  show (m : ℤ) < n ↔ cmp m n = Ordering.lt from
    match cmp m n, cmp_to_int m n with
                           /-
                             m n : ZNum
                             h : Ordering.casesOn Ordering.lt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.lt Ordering.lt)
                           -/
    | Ordering.lt, h => by simp only at h; simp [h]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : ZNum
                             h : Ordering.casesOn Ordering.eq (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.eq Ordering.lt)
                           -/
    | Ordering.eq, h => by simp only at h; simp [h, lt_irrefl]
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             m n : ZNum
                             h : Ordering.casesOn Ordering.gt (LT.lt ↑m ↑n) (Eq m n) (LT.lt ↑n ↑m)
                             ⊢ Iff (LT.lt ↑m ↑n) (Eq Ordering.gt Ordering.lt)
                           -/
    | Ordering.gt, h => by simp [not_lt_of_gt h]
                           /-
                             🎉 no goals
                           -/


theorem le_to_int {m n : ZNum} : (m : ℤ) ≤ n ↔ m ≤ n := by
  /-
    m n : ZNum
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr lt_to_int
                 /-
                   🎉 no goals
                 -/


@[simp, norm_cast]
theorem cast_lt [LinearOrderedRing α] {m n : ZNum} : (m : α) < n ↔ m < n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    m n : ZNum
    ⊢ Iff (LT.lt ↑m ↑n) (LT.lt m n)
  -/
  rw [← cast_to_int m, ← cast_to_int n, Int.cast_lt, lt_to_int]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_le [LinearOrderedRing α] {m n : ZNum} : (m : α) ≤ n ↔ m ≤ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    m n : ZNum
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← not_lt]; exact not_congr cast_lt
                 /-
                   🎉 no goals
                 -/


@[simp, norm_cast]
theorem cast_inj [LinearOrderedRing α] {m n : ZNum} : (m : α) = n ↔ m = n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    m n : ZNum
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  rw [← cast_to_int m, ← cast_to_int n, Int.cast_inj (α := α), to_int_inj]
  /-
    🎉 no goals
  -/


/-- This tactic tries to turn an (in)equality about `ZNum`s to one about `Int`s by rewriting.
```lean
example (n : ZNum) (m : ZNum) : n ≤ n + m * m := by
  transfer_rw
  exact le_add_of_nonneg_right (mul_self_nonneg _)
```
-/
scoped macro (name := transfer_rw) "transfer_rw" : tactic => `(tactic|
    (repeat first | rw [← to_int_inj] | rw [← lt_to_int] | rw [← le_to_int]
     repeat first | rw [cast_add] | rw [mul_to_int] | rw [cast_one] | rw [cast_zero]))


/--
This tactic tries to prove (in)equalities about `ZNum`s by transferring them to the `Int` world and
then trying to call `simp`.
```lean
example (n : ZNum) (m : ZNum) : n ≤ n + m * m := by
  transfer
  exact mul_self_nonneg _
```
-/
scoped macro (name := transfer) "transfer" : tactic => `(tactic|
    (intros; transfer_rw; try simp [add_comm, add_left_comm, mul_comm, mul_left_comm]))


instance linearOrder : LinearOrder ZNum where
  lt := (· < ·)
  lt_iff_le_not_le := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : ZNum), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    intro a b
    /-
      α : Type u_1
      a b : ZNum
      ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    transfer_rw
                /-
                  α : Type u_1
                  ⊢ ∀ (a : ZNum), LE.le a a
                -/
    /-
      α : Type u_1
      a b : ZNum
      ⊢ Iff (LT.lt ↑a ↑b) (And (LE.le ↑a ↑b) (Not (LE.le ↑b ↑a)))
    -/
                /-
                  🎉 no goals
                -/
    apply lt_iff_le_not_le
    /-
      α : Type u_1
      ⊢ ∀ (a b c : ZNum), LE.le a b → LE.le b c → LE.le a c
    -/
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      a b c : ZNum
      ⊢ LE.le a b → LE.le b c → LE.le a c
    -/
  le := (· ≤ ·)
    /-
      α : Type u_1
      a b c : ZNum
      ⊢ LE.le ↑a ↑b → LE.le ↑b ↑c → LE.le ↑a ↑c
    -/
  le_refl := by transfer
    /-
      🎉 no goals
    -/
  le_trans := by
    intro a b c
    transfer_rw
    apply le_trans
  le_antisymm := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : ZNum), LE.le a b → LE.le b a → Eq a b
    -/
    intro a b
    /-
      α : Type u_1
      a b : ZNum
      ⊢ LE.le a b → LE.le b a → Eq a b
    -/
    transfer_rw
    /-
      α : Type u_1
      a b : ZNum
      ⊢ LE.le ↑a ↑b → LE.le ↑b ↑a → Eq ↑a ↑b
    -/
    apply le_antisymm
    /-
      🎉 no goals
    -/
  le_total := by
    /-
      α : Type u_1
      ⊢ ∀ (a b : ZNum), Or (LE.le a b) (LE.le b a)
    -/
    intro a b
    /-
      α : Type u_1
      a b : ZNum
      ⊢ Or (LE.le a b) (LE.le b a)
    -/
    transfer_rw
    /-
      α : Type u_1
      a b : ZNum
      ⊢ Or (LE.le ↑a ↑b) (LE.le ↑b ↑a)
    -/
    apply le_total
    /-
      🎉 no goals
    -/
  -- This is relying on an automatically generated instance name, generated in a `deriving` handler.
  -- See https://github.com/leanprover/lean4/issues/2343
  decidableEq := instDecidableEqZNum
  decidableLE := ZNum.decidableLE
  decidableLT := ZNum.decidableLT


instance addMonoid : AddMonoid ZNum where
  add := (· + ·)
                  /-
                    α : Type u_1
                    ⊢ ∀ (a b c : ZNum), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b …
                  -/
  add_assoc := by transfer
                  /-
                    🎉 no goals
                  -/
  zero := 0
  zero_add := zero_add
  add_zero := add_zero
  nsmul := nsmulRec


instance addCommGroup : AddCommGroup ZNum :=
  { ZNum.addMonoid with
                   /-
                     α : Type u_1
                     ⊢ ∀ (a b : ZNum), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                   -/
    add_comm := by transfer
                   /-
                     🎉 no goals
                   -/
                         /-
                           α : Type u_1
                           ⊢ ∀ (a : ZNum), Eq (HAdd.hAdd (Neg.neg a) a) 0
                         -/
    neg := Neg.neg
                         /-
                           🎉 no goals
                         -/
    zsmul := zsmulRec
    neg_add_cancel := by transfer }


instance addMonoidWithOne : AddMonoidWithOne ZNum :=
  { ZNum.addMonoid with
    one := 1
    natCast := fun n => ZNum.ofInt' n
                                                      /-
                                                        α : Type u_1
                                                        ⊢ Eq (Num.ofNat' 0).toZNum 0
                                                      -/
    natCast_zero := show (Num.ofNat' 0).toZNum = 0 by rw [Num.ofNat'_zero]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    natCast_succ := fun n =>
      show (Num.ofNat' (n + 1)).toZNum = (Num.ofNat' n).toZNum + 1 by
        /-
          α : Type u_1
          n : Nat
          ⊢ Eq (Num.ofNat' (HAdd.hAdd n 1)).toZNum (HAdd.hAdd (Num.ofNat' n).toZNum 1)
        -/
        rw [Num.ofNat'_succ, Num.add_one, Num.toZNum_succ, ZNum.add_one] }
        /-
          🎉 no goals
        -/

-- Porting note: These theorems should be declared out of the instance, otherwise timeouts.


                                                               /-
                                                                 ⊢ ∀ (a b : ZNum), Eq (HMul.hMul a b) (HMul.hMul b a)
                                                               -/
private theorem mul_comm : ∀ (a b : ZNum), a * b = b * a := by transfer
                                                               /-
                                                                 🎉 no goals
                                                               -/


private theorem add_le_add_left : ∀ (a b : ZNum), a ≤ b → ∀ (c : ZNum), c + a ≤ c + b := by
  /-
    ⊢ ∀ (a b : ZNum), LE.le a b → ∀ (c : ZNum), LE.le (HAdd.hAdd c a) (HAdd.hAdd c …
  -/
  intro a b h c
  /-
    a b : ZNum
    h : LE.le a b
    c : ZNum
    ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
  -/
  revert h
  /-
    a b c : ZNum
    ⊢ LE.le a b → LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
  -/
  transfer_rw
  /-
    a b c : ZNum
    ⊢ LE.le ↑a ↑b → LE.le (HAdd.hAdd ↑c ↑a) (HAdd.hAdd ↑c ↑b)
  -/
  exact fun h => _root_.add_le_add_left h c
  /-
    🎉 no goals
  -/


instance linearOrderedCommRing : LinearOrderedCommRing ZNum :=
  { ZNum.linearOrder, ZNum.addCommGroup, ZNum.addMonoidWithOne with
    mul := (· * ·)
                    /-
                      α : Type u_1
                      ⊢ ∀ (a b c : ZNum), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b …
                    -/
                   /-
                     α : Type u_1
                     ⊢ ∀ (a : ZNum), Eq (HMul.hMul 0 a) 0
                   -/
    mul_assoc := by transfer
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u_1
                     ⊢ ∀ (a : ZNum), Eq (HMul.hMul a 0) 0
                   -/
                    /-
                      🎉 no goals
                    -/
      /-
        α : Type u_1
        ⊢ ∀ (a b c : ZNum), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b …
      -/
                   /-
                     🎉 no goals
                   -/
      /-
        α : Type u_1
        a✝ b✝ c✝ : ZNum
        ⊢ Eq (HMul.hMul (↑a✝) (HAdd.hAdd ↑b✝ ↑c✝)) (HAdd.hAdd (HMul.hMul ↑a✝ ↑b✝) (HMu …
      -/
    zero_mul := by transfer
      /-
        🎉 no goals
      -/
    mul_zero := by transfer
      /-
        α : Type u_1
        ⊢ ∀ (a b c : ZNum), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c …
      -/
                  /-
                    α : Type u_1
                    ⊢ ∀ (a : ZNum), Eq (HMul.hMul 1 a) a
                  -/
      /-
        α : Type u_1
        a✝ b✝ c✝ : ZNum
        ⊢ Eq (HMul.hMul (↑c✝) (HAdd.hAdd ↑a✝ ↑b✝)) (HAdd.hAdd (HMul.hMul ↑a✝ ↑c✝) (HMu …
      -/
    one_mul := by transfer
      /-
        🎉 no goals
      -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    α : Type u_1
                    ⊢ ∀ (a : ZNum), Eq (HMul.hMul a 1) a
                  -/
    mul_one := by transfer
                  /-
                    🎉 no goals
                  -/
    left_distrib := by
      transfer
      simp [mul_add]
    right_distrib := by
      transfer
      simp [mul_add, _root_.mul_comm]
    mul_comm := mul_comm
                                /-
                                  α : Type u_1
                                  ⊢ Ne 0 1
                                -/
    exists_pair_ne := ⟨0, 1, by decide⟩
                                /-
                                  🎉 no goals
                                -/
    add_le_add_left := add_le_add_left
    mul_pos := fun a b =>
      show 0 < a → 0 < b → 0 < a * b by
        /-
          α : Type u_1
          a b : ZNum
          ⊢ LT.lt 0 a → LT.lt 0 b → LT.lt 0 (HMul.hMul a b)
        -/
        transfer_rw
                      /-
                        α : Type u_1
                        ⊢ LE.le 0 1
                      -/
        /-
          α : Type u_1
          a b : ZNum
          ⊢ LT.lt 0 ↑a → LT.lt 0 ↑b → LT.lt 0 (HMul.hMul ↑a ↑b)
        -/
                      /-
                        🎉 no goals
                      -/
        apply mul_pos
        /-
          🎉 no goals
        -/
    zero_le_one := by decide }


@[simp, norm_cast]
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝ : Ring α
                                                                       m n : ZNum
                                                                       ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
                                                                     -/
theorem cast_sub [Ring α] (m n) : ((m - n : ZNum) : α) = m - n := by simp [sub_eq_neg_add]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[norm_cast]
theorem neg_of_int : ∀ n, ((-n : ℤ) : ZNum) = -n
  | (_ + 1 : ℕ) => rfl
            /-
              ⊢ Eq (↑(-0)) (Neg.neg ↑0)
            -/
  | 0 => by rw [Int.cast_neg]
            /-
              🎉 no goals
            -/
  | -[_+1] => (zneg_zneg _).symm


@[simp]
theorem ofInt'_eq : ∀ n : ℤ, ZNum.ofInt' n = n
  | (n : ℕ) => rfl
  | -[n+1] => by
    /-
      n : Nat
      ⊢ Eq (ZNum.ofInt' (Int.negSucc n)) ↑(Int.negSucc n)
    -/
    show Num.toZNumNeg (n + 1 : ℕ) = -(n + 1 : ℕ)
    rw [← neg_inj, neg_neg, Nat.cast_succ, Num.add_one, Num.zneg_toZNumNeg, Num.toZNum_succ,
      Nat.cast_succ, ZNum.add_one]
    /-
      n : Nat
      ⊢ Eq (↑n).toZNum.succ (↑n).succ
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem of_nat_toZNum (n : ℕ) : Num.toZNum n = n :=
  rfl

-- Porting note: The priority should be `high`er than `cast_to_int`.

@[simp high, norm_cast]
                                                          /-
                                                            n : ZNum
                                                            ⊢ Eq (↑↑n) n
                                                          -/
theorem of_to_int (n : ZNum) : ((n : ℤ) : ZNum) = n := by rw [← ofInt'_eq, of_to_int']
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem to_of_int (n : ℤ) : ((n : ZNum) : ℤ) = n :=
                           /-
                             n : Int
                             ⊢ Eq (↑↑0) 0
                           -/
                           /-
                             🎉 no goals
                           -/
                                     /-
                                       🎉 no goals
                                     -/
  Int.inductionOn' n 0 (by simp) (by simp) (by simp)
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                              /-
                                                                n : Nat
                                                                ⊢ Eq (↑n).toZNumNeg (Neg.neg ↑n)
                                                              -/
theorem of_nat_toZNumNeg (n : ℕ) : Num.toZNumNeg n = -n := by rw [← of_nat_toZNum, Num.zneg_toZNum]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp, norm_cast]
theorem of_intCast [AddGroupWithOne α] (n : ℤ) : ((n : ZNum) : α) = n := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : Int
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← cast_to_int, to_of_int]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias of_int_cast := of_intCast


@[simp, norm_cast]
theorem of_natCast [AddGroupWithOne α] (n : ℕ) : ((n : ZNum) : α) = n := by
  /-
    α : Type u_1
    inst✝ : AddGroupWithOne α
    n : Nat
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← Int.cast_natCast, of_intCast, Int.cast_natCast]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem dvd_to_int (m n : ZNum) : (m : ℤ) ∣ n ↔ m ∣ n :=
                        /-
                          m n : ZNum
                          x✝ : Dvd.dvd ↑m ↑n
                          k : Int
                          e : Eq (↑n) (HMul.hMul (↑m) k)
                          ⊢ Eq n (HMul.hMul m ↑k)
                        -/
                                               /-
                                                 🎉 no goals
                                               -/
  ⟨fun ⟨k, e⟩ => ⟨k, by rw [← of_to_int n, e]; simp⟩, fun ⟨k, e⟩ => ⟨k, by simp [e]⟩⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem divMod_to_nat_aux {n d : PosNum} {q r : Num} (h₁ : (r : ℕ) + d * ((q : ℕ) + q) = n)
    (h₂ : (r : ℕ) < 2 * d) :
    ((divModAux d q r).2 + d * (divModAux d q r).1 : ℕ) = ↑n ∧ ((divModAux d q r).2 : ℕ) < d := by
  /-
    n d : PosNum
    q r : Num
    h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
    h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
    ⊢ And (Eq (HAdd.hAdd (↑(d.divModAux q r).2) (HMul.hMul ↑d ↑(d.divModAux q r).1 …
  -/
  unfold divModAux
  have : ∀ {r₂}, Num.ofZNum' (Num.sub' r (Num.pos d)) = some r₂ ↔ (r : ℕ) = r₂ + d := by
    intro r₂
    apply Num.mem_ofZNum'.trans
    rw [← ZNum.to_int_inj, Num.cast_toZNum, Num.cast_sub', sub_eq_iff_eq_add, ← Int.natCast_inj]
    simp
  /-
    n d : PosNum
    q r : Num
    h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
    h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
    this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
    ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) (Num. …
  -/
  cases' e : Num.ofZNum' (Num.sub' r (Num.pos d)) with r₂
    /-
      case none
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) Option.none
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) Optio …
    -/
  · rw [Num.cast_bit0, two_mul]
    /-
      case none
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) Option.none
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) Optio …
    -/
    refine ⟨h₁, lt_of_not_ge fun h => ?_⟩
    /-
      case none
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) Option.none
      h : GE.ge ↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) Option.none (fun  …
      ⊢ False
    -/
    cases' Nat.le.dest h with r₂ e'
    /-
      case none.intro
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) Option.none
      h : GE.ge ↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) Option.none (fun  …
      r₂ : Nat
      e' : Eq (HAdd.hAdd (↑d) r₂) ↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) …
      ⊢ False
    -/
    rw [← Num.to_of_nat r₂, add_comm] at e'
    /-
      case none.intro
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) Option.none
      h : GE.ge ↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) Option.none (fun  …
      r₂ : Nat
      e' : Eq (HAdd.hAdd ↑↑r₂ ↑d) ↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) …
      ⊢ False
    -/
    cases e.symm.trans (this.2 e'.symm)
    /-
      🎉 no goals
    -/
    /-
      case some
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r …
      r₂ : Num
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r₂)
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) (Opti …
    -/
  · have := this.1 e
    /-
      case some
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this✝ : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some  …
      r₂ : Num
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r₂)
      this : Eq (↑r) (HAdd.hAdd ↑r₂ ↑d)
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divModAux.match_1 (fun x => Prod Num Num) (Opti …
    -/
    simp only [Num.cast_bit1]
    /-
      case some
      n d : PosNum
      q r : Num
      h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
      h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
      this✝ : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some  …
      r₂ : Num
      e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r₂)
      this : Eq (↑r) (HAdd.hAdd ↑r₂ ↑d)
      ⊢ And (Eq (HAdd.hAdd (↑r₂) (HMul.hMul (↑d) (HAdd.hAdd (HMul.hMul 2 ↑q) 1))) ↑n …
    -/
    constructor
      /-
        case some.left
        n d : PosNum
        q r : Num
        h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
        h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
        this✝ : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some  …
        r₂ : Num
        e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r₂)
        this : Eq (↑r) (HAdd.hAdd ↑r₂ ↑d)
        ⊢ Eq (HAdd.hAdd (↑r₂) (HMul.hMul (↑d) (HAdd.hAdd (HMul.hMul 2 ↑q) 1))) ↑n
      -/
    · rwa [two_mul, add_comm _ 1, mul_add, mul_one, ← add_assoc, ← this]
      /-
        🎉 no goals
      -/
      /-
        case some.right
        n d : PosNum
        q r : Num
        h₁ : Eq (HAdd.hAdd (↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n
        h₂ : LT.lt (↑r) (HMul.hMul 2 ↑d)
        this✝ : ∀ {r₂ : Num}, Iff (Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some  …
        r₂ : Num
        e : Eq (Num.ofZNum' (r.sub' (Num.pos d))) (Option.some r₂)
        this : Eq (↑r) (HAdd.hAdd ↑r₂ ↑d)
        ⊢ LT.lt ↑r₂ ↑d
      -/
    · rwa [this, two_mul, add_lt_add_iff_right] at h₂
      /-
        🎉 no goals
      -/


theorem divMod_to_nat (d n : PosNum) :
    (n / d : ℕ) = (divMod d n).1 ∧ (n % d : ℕ) = (divMod d n).2 := by
  /-
    d n : PosNum
    ⊢ And (Eq (HDiv.hDiv ↑n ↑d) ↑(d.divMod n).1) (Eq (HMod.hMod ↑n ↑d) ↑(d.divMod  …
  -/
  rw [Nat.div_mod_unique (PosNum.cast_pos _)]
  /-
    d n : PosNum
    ⊢ And (Eq (HAdd.hAdd (↑(d.divMod n).2) (HMul.hMul ↑d ↑(d.divMod n).1)) ↑n) (LT …
  -/
  induction' n with n IH n IH
  · exact
      divMod_to_nat_aux (by simp) (Nat.mul_le_mul_left 2 (PosNum.cast_pos d : (0 : ℕ) < d))
    /-
      case bit1
      d n : PosNum
      IH : And (Eq (HAdd.hAdd (↑(d.divMod n).2) (HMul.hMul ↑d ↑(d.divMod n).1)) ↑n)  …
      ⊢ And (Eq (HAdd.hAdd (↑(d.divMod n.bit1).2) (HMul.hMul ↑d ↑(d.divMod n.bit1).1 …
    -/
  · unfold divMod
    -- Porting note: `cases'` didn't rewrite at `this`, so `revert` & `intro` are required.
    /-
      case bit1
      d n : PosNum
      IH : And (Eq (HAdd.hAdd (↑(d.divMod n).2) (HMul.hMul ↑d ↑(d.divMod n).1)) ↑n)  …
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divMod.match_1 (fun x => Prod Num Num) (d.divMo …
    -/
    revert IH; cases' divMod d n with q r; intro IH
    /-
      case bit1.mk
      d n : PosNum
      q r : Num
      IH : And (Eq (HAdd.hAdd (↑{ fst := q, snd := r }.2) (HMul.hMul ↑d ↑{ fst := q, …
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divMod.match_1 (fun x => Prod Num Num) { fst := …
    -/
    simp only [divMod] at IH ⊢
    /-
      case bit1.mk
      d n : PosNum
      q r : Num
      IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
      ⊢ And (Eq (HAdd.hAdd (↑(d.divModAux q r.bit1).2) (HMul.hMul ↑d ↑(d.divModAux q …
    -/
    apply divMod_to_nat_aux <;> simp only [Num.cast_bit1, cast_bit1]
      /-
        case bit1.mk.h₁
        d n : PosNum
        q r : Num
        IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 ↑r) 1) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑ …
      -/
    · rw [← two_mul, ← two_mul, add_right_comm, mul_left_comm, ← mul_add, IH.1]
      /-
        🎉 no goals
      -/
      /-
        case bit1.mk.h₂
        d n : PosNum
        q r : Num
        IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
        ⊢ LT.lt (HAdd.hAdd (HMul.hMul 2 ↑r) 1) (HMul.hMul 2 ↑d)
      -/
    · omega
      /-
        🎉 no goals
      -/
    /-
      case bit0
      d n : PosNum
      IH : And (Eq (HAdd.hAdd (↑(d.divMod n).2) (HMul.hMul ↑d ↑(d.divMod n).1)) ↑n)  …
      ⊢ And (Eq (HAdd.hAdd (↑(d.divMod n.bit0).2) (HMul.hMul ↑d ↑(d.divMod n.bit0).1 …
    -/
  · unfold divMod
    -- Porting note: `cases'` didn't rewrite at `this`, so `revert` & `intro` are required.
    /-
      case bit0
      d n : PosNum
      IH : And (Eq (HAdd.hAdd (↑(d.divMod n).2) (HMul.hMul ↑d ↑(d.divMod n).1)) ↑n)  …
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divMod.match_1 (fun x => Prod Num Num) (d.divMo …
    -/
    revert IH; cases' divMod d n with q r; intro IH
    /-
      case bit0.mk
      d n : PosNum
      q r : Num
      IH : And (Eq (HAdd.hAdd (↑{ fst := q, snd := r }.2) (HMul.hMul ↑d ↑{ fst := q, …
      ⊢ And (Eq (HAdd.hAdd (↑(PosNum.divMod.match_1 (fun x => Prod Num Num) { fst := …
    -/
    simp only [divMod] at IH ⊢
    /-
      case bit0.mk
      d n : PosNum
      q r : Num
      IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
      ⊢ And (Eq (HAdd.hAdd (↑(d.divModAux q r.bit0).2) (HMul.hMul ↑d ↑(d.divModAux q …
    -/
    apply divMod_to_nat_aux
      /-
        case bit0.mk.h₁
        d n : PosNum
        q r : Num
        IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
        ⊢ Eq (HAdd.hAdd (↑r.bit0) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) ↑n.bit0
      -/
    · simp only [Num.cast_bit0, cast_bit0]
      /-
        case bit0.mk.h₁
        d n : PosNum
        q r : Num
        IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
        ⊢ Eq (HAdd.hAdd (HMul.hMul 2 ↑r) (HMul.hMul (↑d) (HAdd.hAdd ↑q ↑q))) (HAdd.hAd …
      -/
      rw [← two_mul, ← two_mul, mul_left_comm, ← mul_add, ← IH.1]
      /-
        🎉 no goals
      -/
      /-
        case bit0.mk.h₂
        d n : PosNum
        q r : Num
        IH : And (Eq (HAdd.hAdd (↑r) (HMul.hMul ↑d ↑q)) ↑n) (LT.lt ↑r ↑d)
        ⊢ LT.lt (↑r.bit0) (HMul.hMul 2 ↑d)
      -/
    · simpa using IH.2
      /-
        🎉 no goals
      -/


@[simp]
theorem div'_to_nat (n d) : (div' n d : ℕ) = n / d :=
  (divMod_to_nat _ _).1.symm


@[simp]
theorem mod'_to_nat (n d) : (mod' n d : ℕ) = n % d :=
  (divMod_to_nat _ _).2.symm


@[simp]
protected theorem div_zero (n : Num) : n / 0 = 0 :=
  show n.div 0 = 0 by
    /-
      n : Num
      ⊢ Eq (n.div 0) 0
    -/
    cases n
      /-
        case zero
        ⊢ Eq (Num.zero.div 0) 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos
        a✝ : PosNum
        ⊢ Eq ((Num.pos a✝).div 0) 0
      -/
    · simp [Num.div]
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem div_to_nat : ∀ n d, ((n / d : Num) : ℕ) = n / d
               /-
                 ⊢ Eq (↑(0 / 0)) (HDiv.hDiv ↑0 ↑0)
               -/
  | 0, 0 => by simp
               /-
                 🎉 no goals
               -/
  | 0, pos _ => (Nat.zero_div _).symm
  | pos _, 0 => (Nat.div_zero _).symm
  | pos _, pos _ => PosNum.div'_to_nat _ _


@[simp]
protected theorem mod_zero (n : Num) : n % 0 = n :=
  show n.mod 0 = n by
    /-
      n : Num
      ⊢ Eq (n.mod 0) n
    -/
    cases n
      /-
        case zero
        ⊢ Eq (Num.zero.mod 0) Num.zero
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos
        a✝ : PosNum
        ⊢ Eq ((Num.pos a✝).mod 0) (Num.pos a✝)
      -/
    · simp [Num.mod]
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem mod_to_nat : ∀ n d, ((n % d : Num) : ℕ) = n % d
               /-
                 ⊢ Eq (↑(HMod.hMod 0 0)) (HMod.hMod ↑0 ↑0)
               -/
  | 0, 0 => by simp
               /-
                 🎉 no goals
               -/
  | 0, pos _ => (Nat.zero_mod _).symm
  | pos _, 0 => (Nat.mod_zero _).symm
  | pos _, pos _ => PosNum.mod'_to_nat _ _


theorem gcd_to_nat_aux :
    ∀ {n} {a b : Num}, a ≤ b → (a * b).natSize ≤ n → (gcdAux n a b : ℕ) = Nat.gcd a b
  | 0, 0, _, _ab, _h => (Nat.gcd_zero_left _).symm
  | 0, pos _, 0, ab, _h => (not_lt_of_ge ab).elim rfl
  | 0, pos _, pos _, _ab, h => (not_lt_of_le h).elim <| PosNum.natSize_pos _
  | Nat.succ _, 0, _, _ab, _h => (Nat.gcd_zero_left _).symm
  | Nat.succ n, pos a, b, ab, h => by
    /-
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
      ⊢ Eq (↑(Num.gcdAux n.succ (Num.pos a) b)) ((↑(Num.pos a)).gcd ↑b)
    -/
    simp only [gcdAux, cast_pos]
    /-
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
      ⊢ Eq (↑(Num.gcdAux n (HMod.hMod b (Num.pos a)) (Num.pos a))) ((↑a).gcd ↑b)
    -/
    rw [Nat.gcd_rec, gcd_to_nat_aux, mod_to_nat]
      /-
        n : Nat
        a : PosNum
        b : Num
        ab : LE.le (Num.pos a) b
        h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
        ⊢ Eq ((HMod.hMod ↑b ↑(Num.pos a)).gcd ↑(Num.pos a)) ((HMod.hMod ↑b ↑a).gcd ↑a)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a
        n : Nat
        a : PosNum
        b : Num
        ab : LE.le (Num.pos a) b
        h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
        ⊢ LE.le (HMod.hMod b (Num.pos a)) (Num.pos a)
      -/
    · rw [← le_to_nat, mod_to_nat]
      /-
        case a
        n : Nat
        a : PosNum
        b : Num
        ab : LE.le (Num.pos a) b
        h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
        ⊢ LE.le (HMod.hMod ↑b ↑(Num.pos a)) ↑(Num.pos a)
      -/
      exact le_of_lt (Nat.mod_lt _ (PosNum.cast_pos _))
      /-
        🎉 no goals
      -/
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LE.le (HMul.hMul (Num.pos a) b).natSize n.succ
      ⊢ LE.le (HMul.hMul (HMod.hMod b (Num.pos a)) (Num.pos a)).natSize n
    -/
    rw [natSize_to_nat, mul_to_nat, Nat.size_le] at h ⊢
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul ↑(Num.pos a) ↑b) (HPow.hPow 2 n.succ)
      ⊢ LT.lt (HMul.hMul ↑(HMod.hMod b (Num.pos a)) ↑(Num.pos a)) (HPow.hPow 2 n)
    -/
    rw [mod_to_nat, mul_comm]
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul ↑(Num.pos a) ↑b) (HPow.hPow 2 n.succ)
      ⊢ LT.lt (HMul.hMul (↑(Num.pos a)) (HMod.hMod ↑b ↑(Num.pos a))) (HPow.hPow 2 n)
    -/
    rw [pow_succ, ← Nat.mod_add_div b (pos a)] at h
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul (↑(Num.pos a)) (HAdd.hAdd (HMod.hMod ↑b ↑(Num.pos a)) (HM …
      ⊢ LT.lt (HMul.hMul (↑(Num.pos a)) (HMod.hMod ↑b ↑(Num.pos a))) (HPow.hPow 2 n)
    -/
    refine lt_of_mul_lt_mul_right (lt_of_le_of_lt ?_ h) (Nat.zero_le 2)
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul (↑(Num.pos a)) (HAdd.hAdd (HMod.hMod ↑b ↑(Num.pos a)) (HM …
      ⊢ LE.le (HMul.hMul (HMul.hMul (↑(Num.pos a)) (HMod.hMod ↑b ↑(Num.pos a))) 2) ( …
    -/
    rw [mul_two, mul_add]
    refine
      add_le_add_left
        (Nat.mul_le_mul_left _ (le_trans (le_of_lt (Nat.mod_lt _ (PosNum.cast_pos _))) ?_)) _
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul (↑(Num.pos a)) (HAdd.hAdd (HMod.hMod ↑b ↑(Num.pos a)) (HM …
      ⊢ LE.le (↑a) (HMul.hMul (↑(Num.pos a)) (HDiv.hDiv ↑b ↑(Num.pos a)))
    -/
    suffices 1 ≤ _ by simpa using Nat.mul_le_mul_left (pos a) this
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul (↑(Num.pos a)) (HAdd.hAdd (HMod.hMod ↑b ↑(Num.pos a)) (HM …
      ⊢ LE.le 1 (HDiv.hDiv ↑b ↑a)
    -/
    rw [Nat.le_div_iff_mul_le a.cast_pos, one_mul]
    /-
      case a
      n : Nat
      a : PosNum
      b : Num
      ab : LE.le (Num.pos a) b
      h : LT.lt (HMul.hMul (↑(Num.pos a)) (HAdd.hAdd (HMod.hMod ↑b ↑(Num.pos a)) (HM …
      ⊢ LE.le ↑a ↑b
    -/
    exact le_to_nat.2 ab
    /-
      🎉 no goals
    -/


@[simp]
theorem gcd_to_nat : ∀ a b, (gcd a b : ℕ) = Nat.gcd a b := by
  have : ∀ a b : Num, (a * b).natSize ≤ a.natSize + b.natSize := by
    intros
    simp only [natSize_to_nat, cast_mul]
    rw [Nat.size_le, pow_add]
    exact mul_lt_mul'' (Nat.lt_size_self _) (Nat.lt_size_self _) (Nat.zero_le _) (Nat.zero_le _)
  /-
    this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
    ⊢ ∀ (a b : Num), Eq (↑(a.gcd b)) ((↑a).gcd ↑b)
  -/
  intros
  /-
    this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
    a✝ b✝ : Num
    ⊢ Eq (↑(a✝.gcd b✝)) ((↑a✝).gcd ↑b✝)
  -/
  unfold gcd
  /-
    this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
    a✝ b✝ : Num
    ⊢ Eq (↑(ite (LE.le a✝ b✝) (Num.gcdAux (HAdd.hAdd a✝.natSize b✝.natSize) a✝ b✝) …
  -/
  split_ifs with h
    /-
      case pos
      this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
      a✝ b✝ : Num
      h : LE.le a✝ b✝
      ⊢ Eq (↑(Num.gcdAux (HAdd.hAdd a✝.natSize b✝.natSize) a✝ b✝)) ((↑a✝).gcd ↑b✝)
    -/
  · exact gcd_to_nat_aux h (this _ _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
      a✝ b✝ : Num
      h : Not (LE.le a✝ b✝)
      ⊢ Eq (↑(Num.gcdAux (HAdd.hAdd b✝.natSize a✝.natSize) b✝ a✝)) ((↑a✝).gcd ↑b✝)
    -/
  · rw [Nat.gcd_comm]
    /-
      case neg
      this : ∀ (a b : Num), LE.le (HMul.hMul a b).natSize (HAdd.hAdd a.natSize b.nat …
      a✝ b✝ : Num
      h : Not (LE.le a✝ b✝)
      ⊢ Eq (↑(Num.gcdAux (HAdd.hAdd b✝.natSize a✝.natSize) b✝ a✝)) ((↑b✝).gcd ↑a✝)
    -/
    exact gcd_to_nat_aux (le_of_not_le h) (this _ _)
    /-
      🎉 no goals
    -/


theorem dvd_iff_mod_eq_zero {m n : Num} : m ∣ n ↔ n % m = 0 := by
  /-
    m n : Num
    ⊢ Iff (Dvd.dvd m n) (Eq (HMod.hMod n m) 0)
  -/
  rw [← dvd_to_nat, Nat.dvd_iff_mod_eq_zero, ← to_nat_inj, mod_to_nat]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


instance decidableDvd : DecidableRel ((· ∣ ·) : Num → Num → Prop)
  | _a, _b => decidable_of_iff' _ dvd_iff_mod_eq_zero


instance PosNum.decidableDvd : DecidableRel ((· ∣ ·) : PosNum → PosNum → Prop)
  | _a, _b => Num.decidableDvd _ _


@[simp]
protected theorem div_zero (n : ZNum) : n / 0 = 0 :=
                      /-
                        n : ZNum
                        ⊢ Eq (n.div 0) 0
                      -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  show n.div 0 = 0 by cases n <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp, norm_cast]
theorem div_to_int : ∀ n d, ((n / d : ZNum) : ℤ) = n / d
               /-
                 ⊢ Eq (↑(0 / 0)) (HDiv.hDiv ↑0 ↑0)
               -/
  | 0, 0 => by simp [Int.ediv_zero]
               /-
                 🎉 no goals
               -/
  | 0, pos _ => (Int.zero_ediv _).symm
  | 0, neg _ => (Int.zero_ediv _).symm
  | pos _, 0 => (Int.ediv_zero _).symm
  | neg _, 0 => (Int.ediv_zero _).symm
                                                    /-
                                                      n d : PosNum
                                                      ⊢ Eq (↑(n.div' d)) (HDiv.hDiv ↑(ZNum.pos n) ↑(ZNum.pos d))
                                                    -/
  | pos n, pos d => (Num.cast_toZNum _).trans <| by rw [← Num.to_nat_to_int]; simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                       /-
                                                         n d : PosNum
                                                         ⊢ Eq (Neg.neg ↑(n.div' d)) (HDiv.hDiv ↑(ZNum.pos n) ↑(ZNum.neg d))
                                                       -/
  | pos n, neg d => (Num.cast_toZNumNeg _).trans <| by rw [← Num.to_nat_to_int]; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  | neg n, pos d =>
    show -_ = -_ / ↑d by
      rw [n.to_int_eq_succ_pred, d.to_int_eq_succ_pred, ← PosNum.to_nat_to_int, Num.succ'_to_nat,
        Num.div_to_nat]
      /-
        n d : PosNum
        ⊢ Eq (Neg.neg ↑(HAdd.hAdd (HDiv.hDiv ↑n.pred' ↑(Num.pos d)) 1)) (HDiv.hDiv (Ne …
      -/
      change -[n.pred' / ↑d+1] = -[n.pred' / (d.pred' + 1)+1]
      /-
        n d : PosNum
        ⊢ Eq (Int.negSucc (HDiv.hDiv ↑n.pred' ↑d)) (Int.negSucc (HDiv.hDiv (↑n.pred')  …
      -/
      rw [d.to_nat_eq_succ_pred]
      /-
        🎉 no goals
      -/
  | neg n, neg d =>
    show ↑(PosNum.pred' n / Num.pos d).succ' = -_ / -↑d by
      rw [n.to_int_eq_succ_pred, d.to_int_eq_succ_pred, ← PosNum.to_nat_to_int, Num.succ'_to_nat,
        Num.div_to_nat]
      /-
        n d : PosNum
        ⊢ Eq (↑(HAdd.hAdd (HDiv.hDiv ↑n.pred' ↑(Num.pos d)) 1)) (HDiv.hDiv (Neg.neg (H …
      -/
      change (Nat.succ (_ / d) : ℤ) = Nat.succ (n.pred' / (d.pred' + 1))
      /-
        n d : PosNum
        ⊢ Eq ↑(HDiv.hDiv ↑n.pred' ↑d).succ ↑(HDiv.hDiv (↑n.pred') (HAdd.hAdd (↑d.pred' …
      -/
      rw [d.to_nat_eq_succ_pred]
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem mod_to_int : ∀ n d, ((n % d : ZNum) : ℤ) = n % d
  | 0, _ => (Int.zero_emod _).symm
  | pos n, d =>
    (Num.cast_toZNum _).trans <| by
      /-
        n : PosNum
        d : ZNum
        ⊢ Eq (↑(HMod.hMod (Num.pos n) d.abs)) (HMod.hMod ↑(ZNum.pos n) ↑d)
      -/
      rw [← Num.to_nat_to_int, cast_pos, Num.mod_to_nat, ← PosNum.to_nat_to_int, abs_to_nat]
      /-
        n : PosNum
        d : ZNum
        ⊢ Eq (↑(HMod.hMod (↑(Num.pos n)) (↑d).natAbs)) (HMod.hMod ↑↑n ↑d)
      -/
      rfl
      /-
        🎉 no goals
      -/
  | neg n, d =>
    (Num.cast_sub' _ _).trans <| by
      rw [← Num.to_nat_to_int, cast_neg, ← Num.to_nat_to_int, Num.succ_to_nat, Num.mod_to_nat,
          abs_to_nat, ← Int.subNatNat_eq_coe, n.to_int_eq_succ_pred]
      /-
        n : PosNum
        d : ZNum
        ⊢ Eq (Int.subNatNat (↑d).natAbs (HAdd.hAdd (HMod.hMod (↑n.pred') (↑d).natAbs)  …
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem gcd_to_nat (a b) : (gcd a b : ℕ) = Int.gcd a b :=
                                   /-
                                     a b : ZNum
                                     ⊢ Eq ((↑a.abs).gcd ↑b.abs) ((↑a).gcd ↑b)
                                   -/
  (Num.gcd_to_nat _ _).trans <| by simp only [abs_to_nat]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem dvd_iff_mod_eq_zero {m n : ZNum} : m ∣ n ↔ n % m = 0 := by
  /-
    m n : ZNum
    ⊢ Iff (Dvd.dvd m n) (Eq (HMod.hMod n m) 0)
  -/
  rw [← dvd_to_int, Int.dvd_iff_emod_eq_zero, ← to_int_inj, mod_to_int]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance decidableDvd : DecidableRel ((· ∣ ·) : ZNum → ZNum → Prop)
  | _a, _b => decidable_of_iff' _ dvd_iff_mod_eq_zero


/-- Cast a `SNum` to the corresponding integer. -/
def ofSnum : SNum → ℤ :=
  SNum.rec' (fun a => cond a (-1) 0) fun a _p IH => cond a (2 * IH + 1) (2 * IH)


instance snumCoe : Coe SNum ℤ :=
  ⟨ofSnum⟩


instance SNum.lt : LT SNum :=
  ⟨fun a b => (a : ℤ) < b⟩


instance SNum.le : LE SNum :=
  ⟨fun a b => (a : ℤ) ≤ b⟩


