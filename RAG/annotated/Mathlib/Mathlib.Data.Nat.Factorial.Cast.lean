theorem cast_ascFactorial : (a.ascFactorial b : S) = (ascPochhammer S b).eval (a : S) := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    a b : Nat
    ⊢ Eq (↑(a.ascFactorial b)) (Polynomial.eval (↑a) (ascPochhammer S b))
  -/
  rw [← ascPochhammer_nat_eq_ascFactorial, ascPochhammer_eval_cast]
  /-
    🎉 no goals
  -/

-- Porting note: added type ascription around a - (b - 1)

theorem cast_descFactorial :
    (a.descFactorial b : S) = (ascPochhammer S b).eval (a - (b - 1) : S) := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    a b : Nat
    ⊢ Eq (↑(a.descFactorial b)) (Polynomial.eval (↑(HSub.hSub a (HSub.hSub b 1)))  …
  -/
  rw [← ascPochhammer_eval_cast, ascPochhammer_nat_eq_descFactorial]
  /-
    S : Type u_1
    inst✝ : Semiring S
    a b : Nat
    ⊢ Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b 1) …
  -/
  induction' b with b
    /-
      case zero
      S : Type u_1
      inst✝ : Semiring S
      a b : Nat
      ⊢ Eq ↑(a.descFactorial 0) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub 0 1) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      S : Type u_1
      inst✝ : Semiring S
      a b✝ b : Nat
      a✝ : Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b …
      ⊢ Eq ↑(a.descFactorial (HAdd.hAdd b 1)) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a ( …
    -/
  · simp_rw [add_succ, Nat.add_one_sub_one]
    /-
      case succ
      S : Type u_1
      inst✝ : Semiring S
      a b✝ b : Nat
      a✝ : Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b …
      ⊢ Eq ↑(a.descFactorial (HAdd.hAdd b 0).succ) ↑((HAdd.hAdd (HSub.hSub a b) b).d …
    -/
    obtain h | h := le_total a b
      /-
        case succ.inl
        S : Type u_1
        inst✝ : Semiring S
        a b✝ b : Nat
        a✝ : Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b …
        h : LE.le a b
        ⊢ Eq ↑(a.descFactorial (HAdd.hAdd b 0).succ) ↑((HAdd.hAdd (HSub.hSub a b) b).d …
      -/
    · rw [descFactorial_of_lt (lt_succ_of_le h), descFactorial_of_lt (lt_succ_of_le _)]
      /-
        S : Type u_1
        inst✝ : Semiring S
        a b✝ b : Nat
        a✝ : Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b …
        h : LE.le a b
        ⊢ LE.le (HAdd.hAdd (HSub.hSub a b) b) b
      -/
      rw [tsub_eq_zero_iff_le.mpr h, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case succ.inr
        S : Type u_1
        inst✝ : Semiring S
        a b✝ b : Nat
        a✝ : Eq ↑(a.descFactorial b) ↑((HSub.hSub (HAdd.hAdd (HSub.hSub a (HSub.hSub b …
        h : LE.le b a
        ⊢ Eq ↑(a.descFactorial (HAdd.hAdd b 0).succ) ↑((HAdd.hAdd (HSub.hSub a b) b).d …
      -/
    · rw [tsub_add_cancel_of_le h]
      /-
        🎉 no goals
      -/


theorem cast_factorial : (a ! : S) = (ascPochhammer S a).eval 1 := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    a : Nat
    ⊢ Eq (↑a.factorial) (Polynomial.eval 1 (ascPochhammer S a))
  -/
  rw [← one_ascFactorial, cast_ascFactorial, cast_one]
  /-
    🎉 no goals
  -/


/-- Convenience lemma. The `a - 1` is not using truncated subtraction, as opposed to the definition
of `Nat.descFactorial` as a natural. -/
theorem cast_descFactorial_two : (a.descFactorial 2 : S) = a * (a - 1) := by
  /-
    S : Type u_1
    inst✝ : Ring S
    a : Nat
    ⊢ Eq (↑(a.descFactorial 2)) (HMul.hMul (↑a) (HSub.hSub (↑a) 1))
  -/
  rw [cast_descFactorial]
  /-
    S : Type u_1
    inst✝ : Ring S
    a : Nat
    ⊢ Eq (Polynomial.eval (↑(HSub.hSub a (HSub.hSub 2 1))) (ascPochhammer S 2)) (H …
  -/
  cases a
    /-
      case zero
      S : Type u_1
      inst✝ : Ring S
      ⊢ Eq (Polynomial.eval (↑(HSub.hSub 0 (HSub.hSub 2 1))) (ascPochhammer S 2)) (H …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [succ_sub_succ, tsub_zero, cast_succ, add_sub_cancel_right, ascPochhammer_succ_right,
      ascPochhammer_one, Polynomial.X_mul, Polynomial.eval_mul_X, Polynomial.eval_add,
      Polynomial.eval_X, cast_one, Polynomial.eval_one]


