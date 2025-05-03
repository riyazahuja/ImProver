/-- `bernsteinPolynomial R n ν` is `(choose n ν) * X^ν * (1 - X)^(n - ν)`.

Although the coefficients are integers, it is convenient to work over an arbitrary commutative ring.
-/
def bernsteinPolynomial (n ν : ℕ) : R[X] :=
  (choose n ν : R[X]) * X ^ ν * (1 - X) ^ (n - ν)


theorem eq_zero_of_lt {n ν : ℕ} (h : n < ν) : bernsteinPolynomial R n ν = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LT.lt n ν
    ⊢ Eq (bernsteinPolynomial R n ν) 0
  -/
  simp [bernsteinPolynomial, Nat.choose_eq_zero_of_lt h]
  /-
    🎉 no goals
  -/


@[simp]
theorem map (f : R →+* S) (n ν : ℕ) :
                                                                        /-
                                                                          R : Type u_1
                                                                          inst✝¹ : CommRing R
                                                                          S : Type u_2
                                                                          inst✝ : CommRing S
                                                                          f : RingHom R S
                                                                          n ν : Nat
                                                                          ⊢ Eq (Polynomial.map f (bernsteinPolynomial R n ν)) (bernsteinPolynomial S n ν)
                                                                        -/
    (bernsteinPolynomial R n ν).map f = bernsteinPolynomial S n ν := by simp [bernsteinPolynomial]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem flip (n ν : ℕ) (h : ν ≤ n) :
    (bernsteinPolynomial R n ν).comp (1 - X) = bernsteinPolynomial R n (n - ν) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LE.le ν n
    ⊢ Eq ((bernsteinPolynomial R n ν).comp (HSub.hSub 1 Polynomial.X)) (bernsteinP …
  -/
  simp [bernsteinPolynomial, h, tsub_tsub_assoc, mul_right_comm]
  /-
    🎉 no goals
  -/


theorem flip' (n ν : ℕ) (h : ν ≤ n) :
    bernsteinPolynomial R n ν = (bernsteinPolynomial R n (n - ν)).comp (1 - X) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LE.le ν n
    ⊢ Eq (bernsteinPolynomial R n ν) ((bernsteinPolynomial R n (HSub.hSub n ν)).co …
  -/
  simp [← flip _ _ _ h, Polynomial.comp_assoc]
  /-
    🎉 no goals
  -/


theorem eval_at_0 (n ν : ℕ) : (bernsteinPolynomial R n ν).eval 0 = if ν = 0 then 1 else 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.eval 0 (bernsteinPolynomial R n ν)) (ite (Eq ν 0) 1 0)
  -/
  rw [bernsteinPolynomial]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.eval 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : Eq ν 0
      ⊢ Eq (Polynomial.eval 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
    -/
  · subst h; simp
             /-
               🎉 no goals
             -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : Not (Eq ν 0)
      ⊢ Eq (Polynomial.eval 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
    -/
  · simp [zero_pow h]
    /-
      🎉 no goals
    -/


theorem eval_at_1 (n ν : ℕ) : (bernsteinPolynomial R n ν).eval 1 = if ν = n then 1 else 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.eval 1 (bernsteinPolynomial R n ν)) (ite (Eq ν n) 1 0)
  -/
  rw [bernsteinPolynomial]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.eval 1 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : Eq ν n
      ⊢ Eq (Polynomial.eval 1 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
    -/
  · subst h; simp
             /-
               🎉 no goals
             -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : Not (Eq ν n)
      ⊢ Eq (Polynomial.eval 1 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
    -/
  · obtain hνn | hnν := Ne.lt_or_lt h
      /-
        case neg.inl
        R : Type u_1
        inst✝ : CommRing R
        n ν : Nat
        h : Not (Eq ν n)
        hνn : LT.lt ν n
        ⊢ Eq (Polynomial.eval 1 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
      -/
    · simp [zero_pow <| Nat.sub_ne_zero_of_lt hνn]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        inst✝ : CommRing R
        n ν : Nat
        h : Not (Eq ν n)
        hnν : LT.lt n ν
        ⊢ Eq (Polynomial.eval 1 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Polyn …
      -/
    · simp [Nat.choose_eq_zero_of_lt hnν]
      /-
        🎉 no goals
      -/


theorem derivative_succ_aux (n ν : ℕ) :
    Polynomial.derivative (bernsteinPolynomial R (n + 1) (ν + 1)) =
      (n + 1) * (bernsteinPolynomial R n ν - bernsteinPolynomial R n (ν + 1)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.derivative (bernsteinPolynomial R (HAdd.hAdd n 1) (HAdd.hAdd  …
  -/
  rw [bernsteinPolynomial]
  suffices ((n + 1).choose (ν + 1) : R[X]) * ((↑(ν + 1 : ℕ) : R[X]) * X ^ ν) * (1 - X) ^ (n - ν) -
      ((n + 1).choose (ν + 1) : R[X]) * X ^ (ν + 1) * ((↑(n - ν) : R[X]) * (1 - X) ^ (n - ν - 1)) =
      (↑(n + 1) : R[X]) * ((n.choose ν : R[X]) * X ^ ν * (1 - X) ^ (n - ν) -
        (n.choose (ν + 1) : R[X]) * X ^ (ν + 1) * (1 - X) ^ (n - (ν + 1))) by
    simpa [Polynomial.derivative_pow, ← sub_eq_add_neg, Nat.succ_sub_succ_eq_sub,
      Polynomial.derivative_mul, Polynomial.derivative_natCast, zero_mul,
      Nat.cast_add, algebraMap.coe_one, Polynomial.derivative_X, mul_one, zero_add,
      Polynomial.derivative_sub, Polynomial.derivative_one, zero_sub, mul_neg, Nat.sub_zero,
      bernsteinPolynomial, map_add, map_natCast, Nat.cast_one]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1 …
  -/
  conv_rhs => rw [mul_sub]
  -- We'll prove the two terms match up separately.
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1 …
  -/
  refine congr (congr_arg Sub.sub ?_) ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (HMul.h …
    -/
  · simp only [← mul_assoc]
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1) …
    -/
    apply congr (congr_arg (· * ·) (congr (congr_arg (· * ·) _) rfl)) rfl
    -- Now it's just about binomial coefficients
    /-
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul ↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1)) ↑(HAdd.hAdd ν 1)) (H …
    -/
    exact mod_cast congr_arg (fun m : ℕ => (m : R[X])) (Nat.succ_mul_choose_eq n ν).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (HPow.h …
    -/
  · rw [← tsub_add_eq_tsub_tsub, ← mul_assoc, ← mul_assoc]; congr 1
    /-
      case refine_2.e_a
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (HPow.h …
    -/
    rw [mul_comm, ← mul_assoc, ← mul_assoc]; congr 1
    /-
      case refine_2.e_a.e_a
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul ↑(HSub.hSub n ν) ↑((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (H …
    -/
    norm_cast
    /-
      case refine_2.e_a.e_a
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq ↑(HMul.hMul (HSub.hSub n ν) ((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) ↑(H …
    -/
    congr 1
    /-
      case refine_2.e_a.e_a.e_a
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      ⊢ Eq (HMul.hMul (HSub.hSub n ν) ((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (HMu …
    -/
    convert (Nat.choose_mul_succ_eq n (ν + 1)).symm using 1
    · -- Porting note: was
      -- convert mul_comm _ _ using 2
      -- simp
      /-
        case h.e'_2
        R : Type u_1
        inst✝ : CommRing R
        n ν : Nat
        ⊢ Eq (HMul.hMul (HSub.hSub n ν) ((HAdd.hAdd n 1).choose (HAdd.hAdd ν 1))) (HMu …
      -/
      rw [mul_comm, Nat.succ_sub_succ_eq_sub]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        R : Type u_1
        inst✝ : CommRing R
        n ν : Nat
        ⊢ Eq (HMul.hMul (HAdd.hAdd n 1) (n.choose (HAdd.hAdd ν 1))) (HMul.hMul (n.choo …
      -/
    · apply mul_comm
      /-
        🎉 no goals
      -/


theorem derivative_succ (n ν : ℕ) : Polynomial.derivative (bernsteinPolynomial R n (ν + 1)) =
    n * (bernsteinPolynomial R (n - 1) ν - bernsteinPolynomial R (n - 1) (ν + 1)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.derivative (bernsteinPolynomial R n (HAdd.hAdd ν 1))) (HMul.h …
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      ν : Nat
      ⊢ Eq (Polynomial.derivative (bernsteinPolynomial R 0 (HAdd.hAdd ν 1))) (HMul.h …
    -/
  · simp [bernsteinPolynomial]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      ν n✝ : Nat
      ⊢ Eq (Polynomial.derivative (bernsteinPolynomial R (HAdd.hAdd n✝ 1) (HAdd.hAdd …
    -/
  · rw [Nat.cast_succ]; apply derivative_succ_aux
                        /-
                          🎉 no goals
                        -/


theorem derivative_zero (n : ℕ) :
    Polynomial.derivative (bernsteinPolynomial R n 0) = -n * bernsteinPolynomial R (n - 1) 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq (Polynomial.derivative (bernsteinPolynomial R n 0)) (HMul.hMul (Neg.neg ↑ …
  -/
  simp [bernsteinPolynomial, Polynomial.derivative_pow]
  /-
    🎉 no goals
  -/


theorem iterate_derivative_at_0_eq_zero_of_lt (n : ℕ) {ν k : ℕ} :
    k < ν → (Polynomial.derivative^[k] (bernsteinPolynomial R n ν)).eval 0 = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν k : Nat
    ⊢ LT.lt k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) k (b …
  -/
  cases' ν with ν
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      n k : Nat
      ⊢ LT.lt k 0 → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) k (b …
    -/
  · rintro ⟨⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      n k ν : Nat
      ⊢ LT.lt k (HAdd.hAdd ν 1) → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.de …
    -/
  · rw [Nat.lt_succ_iff]
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      n k ν : Nat
      ⊢ LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) k (b …
    -/
    induction' k with k ih generalizing n ν
      /-
        case succ.zero
        R : Type u_1
        inst✝ : CommRing R
        n ν : Nat
        ⊢ LE.le 0 ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) 0 (b …
      -/
    · simp [eval_at_0]
      /-
        🎉 no goals
      -/
    · simp only [derivative_succ, Int.natCast_eq_zero, mul_eq_zero, Function.comp_apply,
        Function.iterate_succ, Polynomial.iterate_derivative_sub,
        Polynomial.iterate_derivative_natCast_mul, Polynomial.eval_mul, Polynomial.eval_natCast,
        Polynomial.eval_sub]
      /-
        case succ.succ
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        ih : ∀ (n ν : Nat), LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomia …
        n ν : Nat
        ⊢ LE.le (HAdd.hAdd k 1) ν → Eq (HMul.hMul (↑n) (HSub.hSub (Polynomial.eval 0 ( …
      -/
      intro h
      /-
        case succ.succ
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        ih : ∀ (n ν : Nat), LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomia …
        n ν : Nat
        h : LE.le (HAdd.hAdd k 1) ν
        ⊢ Eq (HMul.hMul (↑n) (HSub.hSub (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.d …
      -/
      apply mul_eq_zero_of_right
      /-
        case succ.succ.h
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        ih : ∀ (n ν : Nat), LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomia …
        n ν : Nat
        h : LE.le (HAdd.hAdd k 1) ν
        ⊢ Eq (HSub.hSub (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) k (be …
      -/
      rw [ih _ _ (Nat.le_of_succ_le h), sub_zero]
      /-
        case succ.succ.h
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        ih : ∀ (n ν : Nat), LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomia …
        n ν : Nat
        h : LE.le (HAdd.hAdd k 1) ν
        ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) k (bernsteinPoly …
      -/
      convert ih _ _ (Nat.pred_le_pred h)
      /-
        case h.e'_2.h.e'_4.h.h.e'_4.h.e'_4
        R : Type u_1
        inst✝ : CommRing R
        k : Nat
        ih : ∀ (n ν : Nat), LE.le k ν → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomia …
        n ν : Nat
        h : LE.le (HAdd.hAdd k 1) ν
        e_2✝ : Eq Ring.toSemiring CommSemiring.toSemiring
        ⊢ Eq ν (HAdd.hAdd ν.pred 1)
      -/
      exact (Nat.succ_pred_eq_of_pos (k.succ_pos.trans_le h)).symm
      /-
        🎉 no goals
      -/


@[simp]
theorem iterate_derivative_succ_at_0_eq_zero (n ν : ℕ) :
    (Polynomial.derivative^[ν] (bernsteinPolynomial R n (ν + 1))).eval 0 = 0 :=
  iterate_derivative_at_0_eq_zero_of_lt R n (lt_add_one ν)


@[simp]
theorem iterate_derivative_at_0 (n ν : ℕ) :
    (Polynomial.derivative^[ν] (bernsteinPolynomial R n ν)).eval 0 =
      (ascPochhammer R ν).eval ((n - (ν - 1) : ℕ) : R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν (bernsteinPoly …
  -/
  by_cases h : ν ≤ n
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LE.le ν n
      ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν (bernsteinPoly …
    -/
  · induction' ν with ν ih generalizing n
      /-
        case pos.zero
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        h : LE.le 0 n
        ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) 0 (bernsteinPoly …
      -/
    · simp [eval_at_0]
      /-
        🎉 no goals
      -/
      /-
        case pos.succ
        R : Type u_1
        inst✝ : CommRing R
        ν : Nat
        ih : ∀ (n : Nat), LE.le ν n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
        n : Nat
        h : LE.le (HAdd.hAdd ν 1) n
        ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) (HAdd.hAdd ν 1)  …
      -/
    · have h' : ν ≤ n - 1 := le_tsub_of_add_le_right h
      simp only [derivative_succ, ih (n - 1) h', iterate_derivative_succ_at_0_eq_zero,
        Nat.succ_sub_succ_eq_sub, tsub_zero, sub_zero, iterate_derivative_sub,
        iterate_derivative_natCast_mul, eval_one, eval_mul, eval_add, eval_sub, eval_X, eval_comp,
        eval_natCast, Function.comp_apply, Function.iterate_succ, ascPochhammer_succ_left]
      /-
        case pos.succ
        R : Type u_1
        inst✝ : CommRing R
        ν : Nat
        ih : ∀ (n : Nat), LE.le ν n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
        n : Nat
        h : LE.le (HAdd.hAdd ν 1) n
        h' : LE.le ν (HSub.hSub n 1)
        ⊢ Eq (HMul.hMul (↑n) (Polynomial.eval (↑(HSub.hSub (HSub.hSub n 1) (HSub.hSub  …
      -/
      obtain rfl | h'' := ν.eq_zero_or_pos
        /-
          case pos.succ.inl
          R : Type u_1
          inst✝ : CommRing R
          n : Nat
          ih : ∀ (n : Nat), LE.le 0 n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
          h : LE.le (HAdd.hAdd 0 1) n
          h' : LE.le 0 (HSub.hSub n 1)
          ⊢ Eq (HMul.hMul (↑n) (Polynomial.eval (↑(HSub.hSub (HSub.hSub n 1) (HSub.hSub  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case pos.succ.inr
          R : Type u_1
          inst✝ : CommRing R
          ν : Nat
          ih : ∀ (n : Nat), LE.le ν n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
          n : Nat
          h : LE.le (HAdd.hAdd ν 1) n
          h' : LE.le ν (HSub.hSub n 1)
          h'' : GT.gt ν 0
          ⊢ Eq (HMul.hMul (↑n) (Polynomial.eval (↑(HSub.hSub (HSub.hSub n 1) (HSub.hSub  …
        -/
      · have : n - 1 - (ν - 1) = n - ν := by omega
        /-
          case pos.succ.inr
          R : Type u_1
          inst✝ : CommRing R
          ν : Nat
          ih : ∀ (n : Nat), LE.le ν n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
          n : Nat
          h : LE.le (HAdd.hAdd ν 1) n
          h' : LE.le ν (HSub.hSub n 1)
          h'' : GT.gt ν 0
          this : Eq (HSub.hSub (HSub.hSub n 1) (HSub.hSub ν 1)) (HSub.hSub n ν)
          ⊢ Eq (HMul.hMul (↑n) (Polynomial.eval (↑(HSub.hSub (HSub.hSub n 1) (HSub.hSub  …
        -/
        rw [this, ascPochhammer_eval_succ]
        /-
          case pos.succ.inr
          R : Type u_1
          inst✝ : CommRing R
          ν : Nat
          ih : ∀ (n : Nat), LE.le ν n → Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial. …
          n : Nat
          h : LE.le (HAdd.hAdd ν 1) n
          h' : LE.le ν (HSub.hSub n 1)
          h'' : GT.gt ν 0
          this : Eq (HSub.hSub (HSub.hSub n 1) (HSub.hSub ν 1)) (HSub.hSub n ν)
          ⊢ Eq (HMul.hMul (↑n) (Polynomial.eval (↑(HSub.hSub n ν)) (ascPochhammer R ν))) …
        -/
        rw_mod_cast [tsub_add_cancel_of_le (h'.trans n.pred_le)]
        /-
          🎉 no goals
        -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : Not (LE.le ν n)
      ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν (bernsteinPoly …
    -/
  · simp only [not_le] at h
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LT.lt n ν
      ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν (bernsteinPoly …
    -/
    rw [tsub_eq_zero_iff_le.mpr (Nat.le_sub_one_of_lt h), eq_zero_of_lt R h]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LT.lt n ν
      ⊢ Eq (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν 0)) (Polynomia …
    -/
    simp [pos_iff_ne_zero.mp (pos_of_gt h)]
    /-
      🎉 no goals
    -/


theorem iterate_derivative_at_0_ne_zero [CharZero R] (n ν : ℕ) (h : ν ≤ n) :
    (Polynomial.derivative^[ν] (bernsteinPolynomial R n ν)).eval 0 ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ Ne (Polynomial.eval 0 (Nat.iterate (⇑Polynomial.derivative) ν (bernsteinPoly …
  -/
  simp only [Int.natCast_eq_zero, bernsteinPolynomial.iterate_derivative_at_0, Ne, Nat.cast_eq_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ Not (Eq (Polynomial.eval (↑(HSub.hSub n (HSub.hSub ν 1))) (ascPochhammer R ν …
  -/
  simp only [← ascPochhammer_eval_cast]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ Not (Eq (↑(Polynomial.eval (HSub.hSub n (HSub.hSub ν 1)) (ascPochhammer Nat  …
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ Not (Eq (Polynomial.eval (HSub.hSub n (HSub.hSub ν 1)) (ascPochhammer Nat ν) …
  -/
  apply ne_of_gt
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ LT.lt 0 (Polynomial.eval (HSub.hSub n (HSub.hSub ν 1)) (ascPochhammer Nat ν))
  -/
  obtain rfl | h' := Nat.eq_zero_or_pos ν
    /-
      case h.inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n : Nat
      h : LE.le 0 n
      ⊢ LT.lt 0 (Polynomial.eval (HSub.hSub n (HSub.hSub 0 1)) (ascPochhammer Nat 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n ν : Nat
      h : LE.le ν n
      h' : GT.gt ν 0
      ⊢ LT.lt 0 (Polynomial.eval (HSub.hSub n (HSub.hSub ν 1)) (ascPochhammer Nat ν))
    -/
  · rw [← Nat.succ_pred_eq_of_pos h'] at h
    /-
      case h.inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      n ν : Nat
      h : LE.le ν.pred.succ n
      h' : GT.gt ν 0
      ⊢ LT.lt 0 (Polynomial.eval (HSub.hSub n (HSub.hSub ν 1)) (ascPochhammer Nat ν))
    -/
    exact ascPochhammer_pos _ _ (tsub_pos_of_lt (Nat.lt_of_succ_le h))
    /-
      🎉 no goals
    -/


theorem iterate_derivative_at_1_eq_zero_of_lt (n : ℕ) {ν k : ℕ} :
    k < n - ν → (Polynomial.derivative^[k] (bernsteinPolynomial R n ν)).eval 1 = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν k : Nat
    ⊢ LT.lt k (HSub.hSub n ν) → Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.de …
  -/
  intro w
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν k : Nat
    w : LT.lt k (HSub.hSub n ν)
    ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) k (bernsteinPoly …
  -/
  rw [flip' _ _ _ (tsub_pos_iff_lt.mp (pos_of_gt w)).le]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν k : Nat
    w : LT.lt k (HSub.hSub n ν)
    ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) k ((bernsteinPol …
  -/
  simp [Polynomial.eval_comp, iterate_derivative_at_0_eq_zero_of_lt R n w]
  /-
    🎉 no goals
  -/


@[simp]
theorem iterate_derivative_at_1 (n ν : ℕ) (h : ν ≤ n) :
    (Polynomial.derivative^[n - ν] (bernsteinPolynomial R n ν)).eval 1 =
      (-1) ^ (n - ν) * (ascPochhammer R (n - ν)).eval (ν + 1 : R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LE.le ν n
    ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub n ν)  …
  -/
  rw [flip' _ _ _ h]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LE.le ν n
    ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub n ν)  …
  -/
  simp [Polynomial.eval_comp, h]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n ν : Nat
    h : LE.le ν n
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HSub.hSub n ν)) (Polynomial.eval (↑(HSub.hSub …
  -/
  obtain rfl | h' := h.eq_or_lt
    /-
      case inl
      R : Type u_1
      inst✝ : CommRing R
      ν : Nat
      h : LE.le ν ν
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HSub.hSub ν ν)) (Polynomial.eval (↑(HSub.hSub …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LE.le ν n
      h' : LT.lt ν n
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HSub.hSub n ν)) (Polynomial.eval (↑(HSub.hSub …
    -/
  · norm_cast
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LE.le ν n
      h' : LT.lt ν n
      ⊢ Eq (HMul.hMul ↑(HPow.hPow (Int.negSucc 0) (HSub.hSub n ν)) ↑(Polynomial.eval …
    -/
    congr
    /-
      case inr.e_a.e_a.e_a
      R : Type u_1
      inst✝ : CommRing R
      n ν : Nat
      h : LE.le ν n
      h' : LT.lt ν n
      ⊢ Eq (HSub.hSub n (HSub.hSub (HSub.hSub n ν) 1)) (HAdd.hAdd ν 1)
    -/
    omega
    /-
      🎉 no goals
    -/


theorem iterate_derivative_at_1_ne_zero [CharZero R] (n ν : ℕ) (h : ν ≤ n) :
    (Polynomial.derivative^[n - ν] (bernsteinPolynomial R n ν)).eval 1 ≠ 0 := by
  rw [bernsteinPolynomial.iterate_derivative_at_1 _ _ _ h, Ne, neg_one_pow_mul_eq_zero_iff, ←
    Nat.cast_succ, ← ascPochhammer_eval_cast, ← Nat.cast_zero, Nat.cast_inj]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharZero R
    n ν : Nat
    h : LE.le ν n
    ⊢ Not (Eq (Polynomial.eval ν.succ (ascPochhammer Nat (HSub.hSub n ν))) 0)
  -/
  exact (ascPochhammer_pos _ _ (Nat.succ_pos ν)).ne'
  /-
    🎉 no goals
  -/


theorem linearIndependent_aux (n k : ℕ) (h : k ≤ n + 1) :
    LinearIndependent ℚ fun ν : Fin k => bernsteinPolynomial ℚ n ν := by
  /-
    n k : Nat
    h : LE.le k (HAdd.hAdd n 1)
    ⊢ LinearIndependent Rat fun ν => bernsteinPolynomial Rat n ↑ν
  -/
  induction' k with k ih
    /-
      case zero
      n : Nat
      h : LE.le 0 (HAdd.hAdd n 1)
      ⊢ LinearIndependent Rat fun ν => bernsteinPolynomial Rat n ↑ν
    -/
  · apply linearIndependent_empty_type
    /-
      🎉 no goals
    -/
    /-
      case succ
      n k : Nat
      ih : LE.le k (HAdd.hAdd n 1) → LinearIndependent Rat fun ν => bernsteinPolynom …
      h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
      ⊢ LinearIndependent Rat fun ν => bernsteinPolynomial Rat n ↑ν
    -/
  · apply linearIndependent_fin_succ'.mpr
    /-
      case succ
      n k : Nat
      ih : LE.le k (HAdd.hAdd n 1) → LinearIndependent Rat fun ν => bernsteinPolynom …
      h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
      ⊢ And (LinearIndependent Rat (Fin.init fun ν => bernsteinPolynomial Rat n ↑ν)) …
    -/
    fconstructor
      /-
        case succ.left
        n k : Nat
        ih : LE.le k (HAdd.hAdd n 1) → LinearIndependent Rat fun ν => bernsteinPolynom …
        h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
        ⊢ LinearIndependent Rat (Fin.init fun ν => bernsteinPolynomial Rat n ↑ν)
      -/
    · exact ih (le_of_lt h)
      /-
        🎉 no goals
      -/
    · -- The actual work!
      -- We show that the (n-k)-th derivative at 1 doesn't vanish,
      -- but vanishes for everything in the span.
      /-
        case succ.right
        n k : Nat
        ih : LE.le k (HAdd.hAdd n 1) → LinearIndependent Rat fun ν => bernsteinPolynom …
        h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.range (Fin.init fun ν => bernst …
      -/
      clear ih
      /-
        case succ.right
        n k : Nat
        h : LE.le (HAdd.hAdd k 1) (HAdd.hAdd n 1)
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.range (Fin.init fun ν => bernst …
      -/
      simp only [Nat.succ_eq_add_one, add_le_add_iff_right] at h
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.range (Fin.init fun ν => bernst …
      -/
      simp only [Fin.val_last, Fin.init_def]
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolyn …
      -/
      dsimp
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolyn …
      -/
      apply not_mem_span_of_apply_not_mem_span_image (@Polynomial.derivative ℚ _ ^ (n - k))
      -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `span_image` into `span_image _`
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        ⊢ Not (Membership.mem (Submodule.span Rat (Set.image (⇑(HPow.hPow Polynomial.d …
      -/
      simp only [not_exists, not_and, Submodule.mem_map, Submodule.span_image _]
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        ⊢ ∀ (x : Polynomial Rat), Membership.mem (Submodule.span Rat (Set.range fun k_ …
      -/
      intro p m
      /-
        case succ.right
        n k : Nat
        h : LE.le k n
        p : Polynomial Rat
        m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
        ⊢ Not (Eq ((HPow.hPow Polynomial.derivative (HSub.hSub n k)) p) ((HPow.hPow Po …
      -/
      apply_fun Polynomial.eval (1 : ℚ)
      /-
        n k : Nat
        h : LE.le k n
        p : Polynomial Rat
        m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
        ⊢ Ne (Polynomial.eval 1 ((HPow.hPow Polynomial.derivative (HSub.hSub n k)) p)) …
      -/
      simp only [LinearMap.pow_apply]
      -- The right hand side is nonzero,
      -- so it will suffice to show the left hand side is always zero.
      suffices (Polynomial.derivative^[n - k] p).eval 1 = 0 by
        rw [this]
        exact (iterate_derivative_at_1_ne_zero ℚ n k h).symm
      /-
        n k : Nat
        h : LE.le k n
        p : Polynomial Rat
        m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
        ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub n k)  …
      -/
      refine span_induction ?_ ?_ ?_ ?_ m
        /-
          case refine_1
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          ⊢ ∀ (x : Polynomial Rat), Membership.mem (Set.range fun k_1 => bernsteinPolyno …
        -/
      · simp only [Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff]
        /-
          case refine_1
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          ⊢ ∀ (a : Fin k), Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) ( …
        -/
        rintro ⟨a, w⟩; simp only [Fin.val_mk]
        /-
          case refine_1.mk
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          a : Nat
          w : LT.lt a k
          ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub n k)  …
        -/
        rw [iterate_derivative_at_1_eq_zero_of_lt ℚ n ((tsub_lt_tsub_iff_left_of_le h).mpr w)]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          ⊢ Eq (Polynomial.eval 1 (Nat.iterate (⇑Polynomial.derivative) (HSub.hSub n k)  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          ⊢ ∀ (x y : Polynomial Rat), Membership.mem (Submodule.span Rat (Set.range fun  …
        -/
      · intro x y _ _ hx hy; simp [hx, hy]
                             /-
                               🎉 no goals
                             -/
        /-
          case refine_4
          n k : Nat
          h : LE.le k n
          p : Polynomial Rat
          m : Membership.mem (Submodule.span Rat (Set.range fun k_1 => bernsteinPolynomi …
          ⊢ ∀ (a : Rat) (x : Polynomial Rat), Membership.mem (Submodule.span Rat (Set.ra …
        -/
      · intro a x _ h; simp [h]
                       /-
                         🎉 no goals
                       -/


/-- The Bernstein polynomials are linearly independent.

We prove by induction that the collection of `bernsteinPolynomial n ν` for `ν = 0, ..., k`
are linearly independent.
The inductive step relies on the observation that the `(n-k)`-th derivative, evaluated at 1,
annihilates `bernsteinPolynomial n ν` for `ν < k`, but has a nonzero value at `ν = k`.
-/
theorem linearIndependent (n : ℕ) :
    LinearIndependent ℚ fun ν : Fin (n + 1) => bernsteinPolynomial ℚ n ν :=
  linearIndependent_aux n (n + 1) le_rfl


theorem sum (n : ℕ) : (∑ ν ∈ Finset.range (n + 1), bernsteinPolynomial R n ν) = 1 :=
  calc
    (∑ ν ∈ Finset.range (n + 1), bernsteinPolynomial R n ν) = (X + (1 - X)) ^ n := by
      /-
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => bernsteinPolynomial R n ν) ( …
      -/
      rw [add_pow]
      /-
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => bernsteinPolynomial R n ν) ( …
      -/
      simp only [bernsteinPolynomial, mul_comm, mul_assoc, mul_left_comm]
      /-
        🎉 no goals
      -/
                /-
                  R : Type u_1
                  inst✝ : CommRing R
                  n : Nat
                  ⊢ Eq (HPow.hPow (HAdd.hAdd Polynomial.X (HSub.hSub 1 Polynomial.X)) n) 1
                -/
    _ = 1 := by simp
                /-
                  🎉 no goals
                -/


theorem sum_smul (n : ℕ) :
    (∑ ν ∈ Finset.range (n + 1), ν • bernsteinPolynomial R n ν) = n • X := by
  -- We calculate the `x`-derivative of `(x+y)^n`, evaluated at `y=(1-x)`,
  -- either directly or by using the binomial theorem.
  -- We'll work in `MvPolynomial Bool R`.
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  let x : MvPolynomial Bool R := MvPolynomial.X true
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  let y : MvPolynomial Bool R := MvPolynomial.X false
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  have pderiv_true_x : pderiv true x = 1 := by rw [pderiv_X]; rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  have pderiv_true_y : pderiv true y = 0 := by rw [pderiv_X]; rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  let e : Bool → R[X] := fun i => cond i X (1 - X)
  -- Start with `(x+y)^n = (x+y)^n`,
  -- take the `x`-derivative, evaluate at `x=X, y=1-X`, and multiply by `X`:
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
    e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
  -/
  trans MvPolynomial.aeval e (pderiv true ((x + y) ^ n)) * X
  -- On the left hand side we'll use the binomial theorem, then simplify.
  · -- We first prepare a tedious rewrite:
    have w : ∀ k : ℕ, k • bernsteinPolynomial R n k =
        (k : R[X]) * Polynomial.X ^ (k - 1) * (1 - Polynomial.X) ^ (n - k) * (n.choose k : R[X]) *
          Polynomial.X := by
      rintro (_ | k)
      · simp
      · rw [bernsteinPolynomial]
        simp only [← natCast_mul, Nat.succ_eq_add_one, Nat.add_succ_sub_one, add_zero, pow_succ]
        push_cast
        ring
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      x : MvPolynomial Bool R := MvPolynomial.X Bool.true
      y : MvPolynomial Bool R := MvPolynomial.X Bool.false
      pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
      pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
      e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
      w : ∀ (k : Nat), Eq (HSMul.hSMul k (bernsteinPolynomial R n k)) (HMul.hMul (HM …
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
    -/
    rw [add_pow, map_sum (pderiv true), map_sum (MvPolynomial.aeval e), Finset.sum_mul]
    -- Step inside the sum:
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      x : MvPolynomial Bool R := MvPolynomial.X Bool.true
      y : MvPolynomial Bool R := MvPolynomial.X Bool.false
      pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
      pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
      e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
      w : ∀ (k : Nat), Eq (HSMul.hSMul k (bernsteinPolynomial R n k)) (HMul.hMul (HM …
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul ν (bernsteinPoly …
    -/
    refine Finset.sum_congr rfl fun k _ => (w k).trans ?_
    simp only [x, y, e, pderiv_true_x, pderiv_true_y, Algebra.id.smul_eq_mul, nsmul_eq_mul,
      Bool.cond_true, Bool.cond_false, add_zero, mul_one, mul_zero, smul_zero, MvPolynomial.aeval_X,
      MvPolynomial.pderiv_mul, Derivation.leibniz_pow, Derivation.map_natCast, map_natCast, map_pow,
      map_mul]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      x : MvPolynomial Bool R := MvPolynomial.X Bool.true
      y : MvPolynomial Bool R := MvPolynomial.X Bool.false
      pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
      pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
      e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
      ⊢ Eq (HMul.hMul ((MvPolynomial.aeval e) ((MvPolynomial.pderiv Bool.true) (HPow …
    -/
  · rw [(pderiv true).leibniz_pow, (pderiv true).map_add, pderiv_true_x, pderiv_true_y]
    simp only [x, y, e, Algebra.id.smul_eq_mul, nsmul_eq_mul, map_natCast, map_pow, map_add,
      map_mul, Bool.cond_true, Bool.cond_false, MvPolynomial.aeval_X, add_sub_cancel,
      one_pow, add_zero, mul_one]


theorem sum_mul_smul (n : ℕ) :
    (∑ ν ∈ Finset.range (n + 1), (ν * (ν - 1)) • bernsteinPolynomial R n ν) =
      (n * (n - 1)) • X ^ 2 := by
  -- We calculate the second `x`-derivative of `(x+y)^n`, evaluated at `y=(1-x)`,
  -- either directly or by using the binomial theorem.
  -- We'll work in `MvPolynomial Bool R`.
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  let x : MvPolynomial Bool R := MvPolynomial.X true
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  let y : MvPolynomial Bool R := MvPolynomial.X false
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  have pderiv_true_x : pderiv true x = 1 := by rw [pderiv_X]; rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  have pderiv_true_y : pderiv true y = 0 := by rw [pderiv_X]; rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  let e : Bool → R[X] := fun i => cond i X (1 - X)
  -- Start with `(x+y)^n = (x+y)^n`,
  -- take the second `x`-derivative, evaluate at `x=X, y=1-X`, and multiply by `X`:
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    x : MvPolynomial Bool R := MvPolynomial.X Bool.true
    y : MvPolynomial Bool R := MvPolynomial.X Bool.false
    pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
    pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
    e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
  -/
  trans MvPolynomial.aeval e (pderiv true (pderiv true ((x + y) ^ n))) * X ^ 2
  -- On the left hand side we'll use the binomial theorem, then simplify.
  · -- We first prepare a tedious rewrite:
    have w : ∀ k : ℕ, (k * (k - 1)) • bernsteinPolynomial R n k =
        (n.choose k : R[X]) * ((1 - Polynomial.X) ^ (n - k) *
          ((k : R[X]) * ((↑(k - 1) : R[X]) * Polynomial.X ^ (k - 1 - 1)))) * Polynomial.X ^ 2 := by
      rintro (_ | _ | k)
      · simp
      · simp
      · rw [bernsteinPolynomial]
        simp only [← natCast_mul, Nat.succ_eq_add_one, Nat.add_succ_sub_one, add_zero, pow_succ]
        push_cast
        ring
    rw [add_pow, map_sum (pderiv true), map_sum (pderiv true), map_sum (MvPolynomial.aeval e),
      Finset.sum_mul]
    -- Step inside the sum:
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      x : MvPolynomial Bool R := MvPolynomial.X Bool.true
      y : MvPolynomial Bool R := MvPolynomial.X Bool.false
      pderiv_true_x : Eq ((MvPolynomial.pderiv Bool.true) x) 1
      pderiv_true_y : Eq ((MvPolynomial.pderiv Bool.true) y) 0
      e : Bool → Polynomial R := fun i => cond i Polynomial.X (HSub.hSub 1 Polynomia …
      w : ∀ (k : Nat), Eq (HSMul.hSMul (HMul.hMul k (HSub.hSub k 1)) (bernsteinPolyn …
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HSMul.hSMul (HMul.hMul ν (HS …
    -/
    refine Finset.sum_congr rfl fun k _ => (w k).trans ?_
    simp only [x, y, e, pderiv_true_x, pderiv_true_y, Algebra.id.smul_eq_mul, nsmul_eq_mul,
      Bool.cond_true, Bool.cond_false, add_zero, zero_add, mul_zero, smul_zero, mul_one,
      MvPolynomial.aeval_X, MvPolynomial.pderiv_X_self, MvPolynomial.pderiv_X_of_ne,
      Derivation.leibniz_pow, Derivation.leibniz, Derivation.map_natCast, map_natCast, map_pow,
      map_mul, map_add]
  -- On the right hand side, we'll just simplify.
  · simp only [x, y, e, pderiv_one, pderiv_mul, (pderiv _).leibniz_pow, (pderiv _).map_natCast,
      (pderiv true).map_add, pderiv_true_x, pderiv_true_y, Algebra.id.smul_eq_mul, add_zero,
      mul_one, Derivation.map_smul_of_tower, map_nsmul, map_pow, map_add, Bool.cond_true,
      Bool.cond_false, MvPolynomial.aeval_X, add_sub_cancel, one_pow, smul_smul,
      smul_one_mul]


/-- A certain linear combination of the previous three identities,
which we'll want later.
-/
theorem variance (n : ℕ) :
    (∑ ν ∈ Finset.range (n + 1), (n • Polynomial.X - (ν : R[X])) ^ 2 * bernsteinPolynomial R n ν) =
      n • Polynomial.X * ((1 : R[X]) - Polynomial.X) := by
  have p : ((((Finset.range (n + 1)).sum fun ν => (ν * (ν - 1)) • bernsteinPolynomial R n ν) +
      (1 - (2 * n) • Polynomial.X) * (Finset.range (n + 1)).sum fun ν =>
        ν • bernsteinPolynomial R n ν) + n ^ 2 • X ^ 2 *
          (Finset.range (n + 1)).sum fun ν => bernsteinPolynomial R n ν) = _ :=
    rfl
  conv at p =>
    lhs
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    simp only [← natCast_mul]
    simp only [← mul_assoc]
    simp only [← add_mul]
  conv at p =>
    rhs
    rw [sum, sum_smul, sum_mul_smul, ← natCast_mul]
  calc
    _ = _ := Finset.sum_congr rfl fun k m => ?_
    _ = _ := p
    _ = _ := ?_
    /-
      case calc_1
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      p : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HAdd.hAdd (HAdd …
      k : Nat
      m : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
      ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub (HSMul.hSMul n Polynomial.X) ↑k) 2) (ber …
    -/
  · congr 1; simp only [← natCast_mul, push_cast]
    /-
      case calc_1.e_a
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      p : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HAdd.hAdd (HAdd …
      k : Nat
      m : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
      ⊢ Eq (HPow.hPow (HSub.hSub (HMul.hMul (↑n) Polynomial.X) ↑k) 2) (HAdd.hAdd (HA …
    -/
                        /-
                          🎉 no goals
                        -/
    cases k <;> · simp; ring
                        /-
                          🎉 no goals
                        -/
    /-
      case calc_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      p : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HAdd.hAdd (HAdd …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul n (HSub.hSub n 1))) (HPow.h …
    -/
  · simp only [← natCast_mul, push_cast]
    /-
      case calc_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      p : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HAdd.hAdd (HAdd …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul ↑n ↑(HSub.hSub n 1)) (HPow.hP …
    -/
    cases n
      /-
        case calc_2.zero
        R : Type u_1
        inst✝ : CommRing R
        p : Eq ((Finset.range (HAdd.hAdd 0 1)).sum fun x => HMul.hMul (HAdd.hAdd (HAdd …
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul ↑0 ↑(HSub.hSub 0 1)) (HPow.hP …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case calc_2.succ
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        p : Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)).sum fun x => HMul.hMul ( …
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul ↑(HAdd.hAdd n✝ 1) ↑(HSub.hSub …
      -/
    · simp; ring
            /-
              🎉 no goals
            -/


