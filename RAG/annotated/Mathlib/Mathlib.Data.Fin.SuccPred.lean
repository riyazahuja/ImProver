instance : ∀ {n : ℕ}, SuccOrder (Fin n)
            /-
              ⊢ SuccOrder (Fin 0)
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
  | 0 => by constructor <;> intro a <;> exact elim0 a
                                        /-
                                          🎉 no goals
                                        -/
  | n + 1 =>
    SuccOrder.ofCore (fun i => if i < Fin.last n then i + 1 else i)
      (by
        /-
          n : Nat
          ⊢ ∀ {a : Fin (HAdd.hAdd n 1)}, Not (IsMax a) → ∀ (b : Fin (HAdd.hAdd n 1)), If …
        -/
        intro a ha b
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Not (IsMax a)
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LT.lt a b) (LE.le ((fun i => ite (LT.lt i (Fin.last n)) (HAdd.hAdd i 1) …
        -/
        rw [isMax_iff_eq_top, eq_top_iff, not_le, top_eq_last] at ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : LT.lt a (Fin.last n)
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LT.lt a b) (LE.le ((fun i => ite (LT.lt i (Fin.last n)) (HAdd.hAdd i 1) …
        -/
        dsimp
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : LT.lt a (Fin.last n)
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LT.lt a b) (LE.le (ite (LT.lt a (Fin.last n)) (HAdd.hAdd a 1) a) b)
        -/
        rw [if_pos ha, lt_iff_val_lt_val, le_iff_val_le_val, val_add_one_of_lt ha]
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : LT.lt a (Fin.last n)
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LT.lt ↑a ↑b) (LE.le (HAdd.hAdd (↑a) 1) ↑b)
        -/
        exact Nat.lt_iff_add_one_le)
        /-
          🎉 no goals
        -/
      (by
        /-
          n : Nat
          ⊢ ∀ (a : Fin (HAdd.hAdd n 1)), IsMax a → Eq ((fun i => ite (LT.lt i (Fin.last  …
        -/
        intro a ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : IsMax a
          ⊢ Eq ((fun i => ite (LT.lt i (Fin.last n)) (HAdd.hAdd i 1) i) a) a
        -/
        rw [isMax_iff_eq_top, top_eq_last] at ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Eq a (Fin.last n)
          ⊢ Eq ((fun i => ite (LT.lt i (Fin.last n)) (HAdd.hAdd i 1) i) a) a
        -/
        dsimp
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Eq a (Fin.last n)
          ⊢ Eq (ite (LT.lt a (Fin.last n)) (HAdd.hAdd a 1) a) a
        -/
        rw [if_neg ha.not_lt])
        /-
          🎉 no goals
        -/


@[simp]
theorem succ_eq {n : ℕ} : SuccOrder.succ = fun a => if a < Fin.last n then a + 1 else a :=
  rfl


@[simp]
theorem succ_apply {n : ℕ} (a) : SuccOrder.succ a = if a < Fin.last n then a + 1 else a :=
  rfl


instance : ∀ {n : ℕ}, PredOrder (Fin n)
            /-
              ⊢ PredOrder (Fin 0)
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
  | 0 => by constructor <;> first | intro a; exact elim0 a
                            /-
                              🎉 no goals
                            -/
  | n + 1 =>
    PredOrder.ofCore (fun x => if x = 0 then 0 else x - 1)
      (by
        /-
          n : Nat
          ⊢ ∀ {a : Fin (HAdd.hAdd n 1)}, Not (IsMin a) → ∀ (b : Fin (HAdd.hAdd n 1)), If …
        -/
        intro a ha b
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Not (IsMin a)
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LE.le b ((fun x => ite (Eq x 0) 0 (HSub.hSub x 1)) a)) (LT.lt b a)
        -/
        rw [isMin_iff_eq_bot, eq_bot_iff, not_le, bot_eq_zero] at ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : LT.lt 0 a
          b : Fin (HAdd.hAdd n 1)
          ⊢ Iff (LE.le b ((fun x => ite (Eq x 0) 0 (HSub.hSub x 1)) a)) (LT.lt b a)
        -/
        dsimp
        rw [if_neg ha.ne', lt_iff_val_lt_val, le_iff_val_le_val, coe_sub_one, if_neg ha.ne',
          Nat.lt_iff_add_one_le, Nat.le_sub_iff_add_le]
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : LT.lt 0 a
          b : Fin (HAdd.hAdd n 1)
          ⊢ LE.le 1 ↑a
        -/
        exact ha)
        /-
          🎉 no goals
        -/
      (by
        /-
          n : Nat
          ⊢ ∀ (a : Fin (HAdd.hAdd n 1)), IsMin a → Eq ((fun x => ite (Eq x 0) 0 (HSub.hS …
        -/
        intro a ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : IsMin a
          ⊢ Eq ((fun x => ite (Eq x 0) 0 (HSub.hSub x 1)) a) a
        -/
        rw [isMin_iff_eq_bot, bot_eq_zero] at ha
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Eq a 0
          ⊢ Eq ((fun x => ite (Eq x 0) 0 (HSub.hSub x 1)) a) a
        -/
        dsimp
        /-
          n : Nat
          a : Fin (HAdd.hAdd n 1)
          ha : Eq a 0
          ⊢ Eq (ite (Eq a 0) 0 (HSub.hSub a 1)) a
        -/
        rwa [if_pos ha, eq_comm])
        /-
          🎉 no goals
        -/


@[simp]
theorem pred_eq {n} : PredOrder.pred = fun a : Fin (n + 1) => if a = 0 then 0 else a - 1 :=
  rfl


@[simp]
theorem pred_apply {n : ℕ} (a : Fin (n + 1)) : PredOrder.pred a = if a = 0 then 0 else a - 1 :=
  rfl


