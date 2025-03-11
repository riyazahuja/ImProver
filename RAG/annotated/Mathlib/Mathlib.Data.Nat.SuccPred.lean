@[instance] abbrev instSuccOrder : SuccOrder ℕ :=
  SuccOrder.ofSuccLeIff succ Nat.succ_le


instance instSuccAddOrder : SuccAddOrder ℕ := ⟨fun _ => rfl⟩

-- so that Lean reads `Nat.pred` through `pred_order.pred`

@[instance] abbrev instPredOrder : PredOrder ℕ where
  pred := pred
  pred_le := pred_le
  min_of_le_pred {a} ha := by
    /-
      m n a : Nat
      ha : LE.le a a.pred
      ⊢ IsMin a
    -/
    cases a
      /-
        case zero
        m n : Nat
        ha : LE.le 0 (Nat.pred 0)
        ⊢ IsMin 0
      -/
    · exact isMin_bot
      /-
        🎉 no goals
      -/
      /-
        case succ
        m n n✝ : Nat
        ha : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd n✝ 1).pred
        ⊢ IsMin (HAdd.hAdd n✝ 1)
      -/
    · exact (not_succ_le_self _ ha).elim
      /-
        🎉 no goals
      -/
  le_pred_of_lt {a} {b} h := by
    /-
      m n a b : Nat
      h : LT.lt a b
      ⊢ LE.le a b.pred
    -/
    cases b
      /-
        case zero
        m n a : Nat
        h : LT.lt a 0
        ⊢ LE.le a (Nat.pred 0)
      -/
    · exact (a.not_lt_zero h).elim
      /-
        🎉 no goals
      -/
      /-
        case succ
        m n a n✝ : Nat
        h : LT.lt a (HAdd.hAdd n✝ 1)
        ⊢ LE.le a (HAdd.hAdd n✝ 1).pred
      -/
    · exact le_of_succ_le_succ h
      /-
        🎉 no goals
      -/


instance instPredSubOrder : PredSubOrder ℕ := ⟨fun _ => rfl⟩


@[simp]
theorem succ_eq_succ : Order.succ = succ :=
  rfl


@[simp]
theorem pred_eq_pred : Order.pred = pred :=
  rfl


protected theorem succ_iterate (a : ℕ) : ∀ n, succ^[n] a = a + n :=
  Order.succ_iterate a


protected theorem pred_iterate (a : ℕ) : ∀ n, pred^[n] a = a - n
  | 0 => rfl
  | n + 1 => by
    /-
      a n : Nat
      ⊢ Eq (Nat.iterate Nat.pred (HAdd.hAdd n 1) a) (HSub.hSub a (HAdd.hAdd n 1))
    -/
    rw [Function.iterate_succ', sub_succ]
    /-
      a n : Nat
      ⊢ Eq (Function.comp Nat.pred (Nat.iterate Nat.pred n) a) (HSub.hSub a n).pred
    -/
    exact congr_arg _ (Nat.pred_iterate a n)
    /-
      🎉 no goals
    -/


lemma le_succ_iff_eq_or_le : m ≤ n.succ ↔ m = n.succ ∨ m ≤ n := Order.le_succ_iff_eq_or_le


instance : IsSuccArchimedean ℕ :=
                               /-
                                 m n a b : Nat
                                 h : LE.le a b
                                 ⊢ Eq (Nat.iterate Order.succ (HSub.hSub b a) a) b
                               -/
  ⟨fun {a} {b} h => ⟨b - a, by rw [succ_eq_succ, Nat.succ_iterate, add_tsub_cancel_of_le h]⟩⟩
                               /-
                                 🎉 no goals
                               -/


instance : IsPredArchimedean ℕ :=
                               /-
                                 m n a b : Nat
                                 h : LE.le a b
                                 ⊢ Eq (Nat.iterate Order.pred (HSub.hSub b a) b) a
                               -/
  ⟨fun {a} {b} h => ⟨b - a, by rw [pred_eq_pred, Nat.pred_iterate, tsub_tsub_cancel_of_le h]⟩⟩
                               /-
                                 🎉 no goals
                               -/


lemma forall_ne_zero_iff (P : ℕ → Prop) :
    (∀ i, i ≠ 0 → P i) ↔ (∀ i, P (i + 1)) :=
  SuccOrder.forall_ne_bot_iff P


@[deprecated Order.covBy_iff_add_one_eq (since := "2024-09-04")]
protected theorem covBy_iff_succ_eq {m n : ℕ} : m ⋖ n ↔ m + 1 = n :=
  covBy_iff_add_one_eq


@[simp, norm_cast]
theorem Fin.coe_covBy_iff {n : ℕ} {a b : Fin n} : (a : ℕ) ⋖ b ↔ a ⋖ b :=
  and_congr_right' ⟨fun h _c hc => h hc, fun h c ha hb => @h ⟨c, hb.trans b.prop⟩ ha hb⟩


alias ⟨_, CovBy.coe_fin⟩ := Fin.coe_covBy_iff

