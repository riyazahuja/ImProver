/-- `ascPochhammer S n` is the polynomial `X * (X + 1) * ... * (X + n - 1)`,
with coefficients in the semiring `S`.
-/
noncomputable def ascPochhammer : ℕ → S[X]
  | 0 => 1
  | n + 1 => X * (ascPochhammer n).comp (X + 1)


@[simp]
theorem ascPochhammer_zero : ascPochhammer S 0 = 1 :=
  rfl


@[simp]
                                                        /-
                                                          S : Type u
                                                          inst✝ : Semiring S
                                                          ⊢ Eq (ascPochhammer S 1) Polynomial.X
                                                        -/
theorem ascPochhammer_one : ascPochhammer S 1 = X := by simp [ascPochhammer]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem ascPochhammer_succ_left (n : ℕ) :
    ascPochhammer S (n + 1) = X * (ascPochhammer S n).comp (X + 1) := by
  /-
    S : Type u
    inst✝ : Semiring S
    n : Nat
    ⊢ Eq (ascPochhammer S (HAdd.hAdd n 1)) (HMul.hMul Polynomial.X ((ascPochhammer …
  -/
  rw [ascPochhammer]
  /-
    🎉 no goals
  -/


theorem monic_ascPochhammer (n : ℕ) [Nontrivial S] [NoZeroDivisors S] :
    Monic <| ascPochhammer S n := by
  /-
    S : Type u
    inst✝² : Semiring S
    n : Nat
    inst✝¹ : Nontrivial S
    inst✝ : NoZeroDivisors S
    ⊢ (ascPochhammer S n).Monic
  -/
  induction' n with n hn
    /-
      case zero
      S : Type u
      inst✝² : Semiring S
      inst✝¹ : Nontrivial S
      inst✝ : NoZeroDivisors S
      ⊢ (ascPochhammer S 0).Monic
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      S : Type u
      inst✝² : Semiring S
      inst✝¹ : Nontrivial S
      inst✝ : NoZeroDivisors S
      n : Nat
      hn : (ascPochhammer S n).Monic
      ⊢ (ascPochhammer S (HAdd.hAdd n 1)).Monic
    -/
  · have : leadingCoeff (X + 1 : S[X]) = 1 := leadingCoeff_X_add_C 1
    rw [ascPochhammer_succ_left, Monic.def, leadingCoeff_mul,
      leadingCoeff_comp (ne_zero_of_eq_one <| natDegree_X_add_C 1 : natDegree (X + 1) ≠ 0), hn,
      monic_X, one_mul, one_mul, this, one_pow]


@[simp]
theorem ascPochhammer_map (f : S →+* T) (n : ℕ) :
    (ascPochhammer S n).map f = ascPochhammer T n := by
  induction n with
  | zero => simp
  | succ n ih => simp [ih, ascPochhammer_succ_left, map_comp]


theorem ascPochhammer_eval₂ (f : S →+* T) (n : ℕ) (t : T) :
    (ascPochhammer T n).eval t = (ascPochhammer S n).eval₂ f t := by
  /-
    S : Type u
    inst✝¹ : Semiring S
    T : Type v
    inst✝ : Semiring T
    f : RingHom S T
    n : Nat
    t : T
    ⊢ Eq (Polynomial.eval t (ascPochhammer T n)) (Polynomial.eval₂ f t (ascPochham …
  -/
  rw [← ascPochhammer_map f]
  /-
    S : Type u
    inst✝¹ : Semiring S
    T : Type v
    inst✝ : Semiring T
    f : RingHom S T
    n : Nat
    t : T
    ⊢ Eq (Polynomial.eval t (Polynomial.map f (ascPochhammer S n))) (Polynomial.ev …
  -/
  exact eval_map f t
  /-
    🎉 no goals
  -/


theorem ascPochhammer_eval_comp {R : Type*} [CommSemiring R] (n : ℕ) (p : R[X]) [Algebra R S]
    (x : S) : ((ascPochhammer S n).comp (p.map (algebraMap R S))).eval x =
    (ascPochhammer S n).eval (p.eval₂ (algebraMap R S) x) := by
  rw [ascPochhammer_eval₂ (algebraMap R S), ← eval₂_comp', ← ascPochhammer_map (algebraMap R S),
    ← map_comp, eval_map]


@[simp, norm_cast]
theorem ascPochhammer_eval_cast (n k : ℕ) :
    (((ascPochhammer ℕ n).eval k : ℕ) : S) = ((ascPochhammer S n).eval k : S) := by
  rw [← ascPochhammer_map (algebraMap ℕ S), eval_map, ← eq_natCast (algebraMap ℕ S),
      eval₂_at_natCast,Nat.cast_id]


theorem ascPochhammer_eval_zero {n : ℕ} : (ascPochhammer S n).eval 0 = if n = 0 then 1 else 0 := by
  /-
    S : Type u
    inst✝ : Semiring S
    n : Nat
    ⊢ Eq (Polynomial.eval 0 (ascPochhammer S n)) (ite (Eq n 0) 1 0)
  -/
  cases n
    /-
      case zero
      S : Type u
      inst✝ : Semiring S
      ⊢ Eq (Polynomial.eval 0 (ascPochhammer S 0)) (ite (Eq 0 0) 1 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      S : Type u
      inst✝ : Semiring S
      n✝ : Nat
      ⊢ Eq (Polynomial.eval 0 (ascPochhammer S (HAdd.hAdd n✝ 1))) (ite (Eq (HAdd.hAd …
    -/
  · simp [X_mul, Nat.succ_ne_zero, ascPochhammer_succ_left]
    /-
      🎉 no goals
    -/


                                                                            /-
                                                                              S : Type u
                                                                              inst✝ : Semiring S
                                                                              ⊢ Eq (Polynomial.eval 0 (ascPochhammer S 0)) 1
                                                                            -/
theorem ascPochhammer_zero_eval_zero : (ascPochhammer S 0).eval 0 = 1 := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem ascPochhammer_ne_zero_eval_zero {n : ℕ} (h : n ≠ 0) : (ascPochhammer S n).eval 0 = 0 := by
  /-
    S : Type u
    inst✝ : Semiring S
    n : Nat
    h : Ne n 0
    ⊢ Eq (Polynomial.eval 0 (ascPochhammer S n)) 0
  -/
  simp [ascPochhammer_eval_zero, h]
  /-
    🎉 no goals
  -/


theorem ascPochhammer_succ_right (n : ℕ) :
    ascPochhammer S (n + 1) = ascPochhammer S n * (X + (n : S[X])) := by
  suffices h : ascPochhammer ℕ (n + 1) = ascPochhammer ℕ n * (X + (n : ℕ[X])) by
    apply_fun Polynomial.map (algebraMap ℕ S) at h
    simpa only [ascPochhammer_map, Polynomial.map_mul, Polynomial.map_add, map_X,
      Polynomial.map_natCast] using h
  induction n with
  | zero => simp
  | succ n ih =>
    conv_lhs =>
      rw [ascPochhammer_succ_left, ih, mul_comp, ← mul_assoc, ← ascPochhammer_succ_left, add_comp,
          X_comp, natCast_comp, add_assoc, add_comm (1 : ℕ[X]), ← Nat.cast_succ]


theorem ascPochhammer_succ_eval {S : Type*} [Semiring S] (n : ℕ) (k : S) :
    (ascPochhammer S (n + 1)).eval k = (ascPochhammer S n).eval k * (k + n) := by
  rw [ascPochhammer_succ_right, mul_add, eval_add, eval_mul_X, ← Nat.cast_comm, ← C_eq_natCast,
    eval_C_mul, Nat.cast_comm, ← mul_add]


theorem ascPochhammer_succ_comp_X_add_one (n : ℕ) :
    (ascPochhammer S (n + 1)).comp (X + 1) =
      ascPochhammer S (n + 1) + (n + 1) • (ascPochhammer S n).comp (X + 1) := by
  suffices (ascPochhammer ℕ (n + 1)).comp (X + 1) =
      ascPochhammer ℕ (n + 1) + (n + 1) * (ascPochhammer ℕ n).comp (X + 1)
    by simpa [map_comp] using congr_arg (Polynomial.map (Nat.castRingHom S)) this
  /-
    S : Type u
    inst✝ : Semiring S
    n : Nat
    ⊢ Eq ((ascPochhammer Nat (HAdd.hAdd n 1)).comp (HAdd.hAdd Polynomial.X 1)) (HA …
  -/
  nth_rw 2 [ascPochhammer_succ_left]
  rw [← add_mul, ascPochhammer_succ_right ℕ n, mul_comp, mul_comm, add_comp, X_comp, natCast_comp,
    add_comm, ← add_assoc]
  /-
    S : Type u
    inst✝ : Semiring S
    n : Nat
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑n) Polynomial.X) 1) ((ascPochhammer Na …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem ascPochhammer_mul (n m : ℕ) :
    ascPochhammer S n * (ascPochhammer S m).comp (X + (n : S[X])) = ascPochhammer S (n + m) := by
  /-
    S : Type u
    inst✝ : Semiring S
    n m : Nat
    ⊢ Eq (HMul.hMul (ascPochhammer S n) ((ascPochhammer S m).comp (HAdd.hAdd Polyn …
  -/
  induction' m with m ih
    /-
      case zero
      S : Type u
      inst✝ : Semiring S
      n : Nat
      ⊢ Eq (HMul.hMul (ascPochhammer S n) ((ascPochhammer S 0).comp (HAdd.hAdd Polyn …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [ascPochhammer_succ_right, Polynomial.mul_X_add_natCast_comp, ← mul_assoc, ih,
      ← add_assoc, ascPochhammer_succ_right, Nat.cast_add, add_assoc]


theorem ascPochhammer_nat_eq_ascFactorial (n : ℕ) :
    ∀ k, (ascPochhammer ℕ k).eval n = n.ascFactorial k
            /-
              n : Nat
              ⊢ Eq (Polynomial.eval n (ascPochhammer Nat 0)) (n.ascFactorial 0)
            -/
  | 0 => by rw [ascPochhammer_zero, eval_one, Nat.ascFactorial_zero]
            /-
              🎉 no goals
            -/
  | t + 1 => by
    rw [ascPochhammer_succ_right, eval_mul, ascPochhammer_nat_eq_ascFactorial n t, eval_add, eval_X,
      eval_natCast, Nat.cast_id, Nat.ascFactorial_succ, mul_comm]


theorem ascPochhammer_nat_eq_natCast_ascFactorial (S : Type*) [Semiring S] (n k : ℕ) :
    (ascPochhammer S k).eval (n : S) = n.ascFactorial k := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    n k : Nat
    ⊢ Eq (Polynomial.eval (↑n) (ascPochhammer S k)) ↑(n.ascFactorial k)
  -/
  norm_cast
  /-
    S : Type u_1
    inst✝ : Semiring S
    n k : Nat
    ⊢ Eq ↑(Polynomial.eval n (ascPochhammer Nat k)) ↑(n.ascFactorial k)
  -/
  rw [ascPochhammer_nat_eq_ascFactorial]
  /-
    🎉 no goals
  -/


theorem ascPochhammer_nat_eq_descFactorial (a b : ℕ) :
    (ascPochhammer ℕ b).eval a = (a + b - 1).descFactorial b := by
  /-
    a b : Nat
    ⊢ Eq (Polynomial.eval a (ascPochhammer Nat b)) ((HSub.hSub (HAdd.hAdd a b) 1). …
  -/
  rw [ascPochhammer_nat_eq_ascFactorial, Nat.add_descFactorial_eq_ascFactorial']
  /-
    🎉 no goals
  -/


theorem ascPochhammer_nat_eq_natCast_descFactorial (S : Type*) [Semiring S] (a b : ℕ) :
    (ascPochhammer S b).eval (a : S) = (a + b - 1).descFactorial b := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    a b : Nat
    ⊢ Eq (Polynomial.eval (↑a) (ascPochhammer S b)) ↑((HSub.hSub (HAdd.hAdd a b) 1 …
  -/
  norm_cast
  /-
    S : Type u_1
    inst✝ : Semiring S
    a b : Nat
    ⊢ Eq ↑(Polynomial.eval a (ascPochhammer Nat b)) ↑((HSub.hSub (HAdd.hAdd a b) 1 …
  -/
  rw [ascPochhammer_nat_eq_descFactorial]
  /-
    🎉 no goals
  -/


@[simp]
theorem ascPochhammer_natDegree (n : ℕ) [NoZeroDivisors S] [Nontrivial S] :
    (ascPochhammer S n).natDegree = n := by
  /-
    S : Type u
    inst✝² : Semiring S
    n : Nat
    inst✝¹ : NoZeroDivisors S
    inst✝ : Nontrivial S
    ⊢ Eq (ascPochhammer S n).natDegree n
  -/
  induction' n with n hn
    /-
      case zero
      S : Type u
      inst✝² : Semiring S
      inst✝¹ : NoZeroDivisors S
      inst✝ : Nontrivial S
      ⊢ Eq (ascPochhammer S 0).natDegree 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      S : Type u
      inst✝² : Semiring S
      inst✝¹ : NoZeroDivisors S
      inst✝ : Nontrivial S
      n : Nat
      hn : Eq (ascPochhammer S n).natDegree n
      ⊢ Eq (ascPochhammer S (HAdd.hAdd n 1)).natDegree (HAdd.hAdd n 1)
    -/
  · have : natDegree (X + (n : S[X])) = 1 := natDegree_X_add_C (n : S)
    rw [ascPochhammer_succ_right,
        natDegree_mul _ (ne_zero_of_natDegree_gt <| this.symm ▸ Nat.zero_lt_one), hn, this]
    /-
      S : Type u
      inst✝² : Semiring S
      inst✝¹ : NoZeroDivisors S
      inst✝ : Nontrivial S
      n : Nat
      hn : Eq (ascPochhammer S n).natDegree n
      this : Eq (HAdd.hAdd Polynomial.X ↑n).natDegree 1
      ⊢ Ne (ascPochhammer S n) 0
    -/
    cases n
      /-
        case zero
        S : Type u
        inst✝² : Semiring S
        inst✝¹ : NoZeroDivisors S
        inst✝ : Nontrivial S
        hn : Eq (ascPochhammer S 0).natDegree 0
        this : Eq (HAdd.hAdd Polynomial.X ↑0).natDegree 1
        ⊢ Ne (ascPochhammer S 0) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ
        S : Type u
        inst✝² : Semiring S
        inst✝¹ : NoZeroDivisors S
        inst✝ : Nontrivial S
        n✝ : Nat
        hn : Eq (ascPochhammer S (HAdd.hAdd n✝ 1)).natDegree (HAdd.hAdd n✝ 1)
        this : Eq (HAdd.hAdd Polynomial.X ↑(HAdd.hAdd n✝ 1)).natDegree 1
        ⊢ Ne (ascPochhammer S (HAdd.hAdd n✝ 1)) 0
      -/
    · refine ne_zero_of_natDegree_gt <| hn.symm ▸ Nat.add_one_pos _
      /-
        🎉 no goals
      -/


theorem ascPochhammer_pos (n : ℕ) (s : S) (h : 0 < s) : 0 < (ascPochhammer S n).eval s := by
  induction n with
  | zero =>
    simp only [ascPochhammer_zero, eval_one]
    exact zero_lt_one
  | succ n ih =>
    rw [ascPochhammer_succ_right, mul_add, eval_add, ← Nat.cast_comm, eval_natCast_mul, eval_mul_X,
      Nat.cast_comm, ← mul_add]
    exact mul_pos ih (lt_of_lt_of_le h (le_add_of_nonneg_right (Nat.cast_nonneg n)))


@[simp]
theorem ascPochhammer_eval_one (S : Type*) [Semiring S] (n : ℕ) :
    (ascPochhammer S n).eval (1 : S) = (n ! : S) := by
  /-
    S : Type u_2
    inst✝ : Semiring S
    n : Nat
    ⊢ Eq (Polynomial.eval 1 (ascPochhammer S n)) ↑n.factorial
  -/
  rw_mod_cast [ascPochhammer_nat_eq_ascFactorial, Nat.one_ascFactorial]
  /-
    🎉 no goals
  -/


theorem factorial_mul_ascPochhammer (S : Type*) [Semiring S] (r n : ℕ) :
    (r ! : S) * (ascPochhammer S n).eval (r + 1 : S) = (r + n)! := by
  /-
    S : Type u_2
    inst✝ : Semiring S
    r n : Nat
    ⊢ Eq (HMul.hMul (↑r.factorial) (Polynomial.eval (HAdd.hAdd (↑r) 1) (ascPochham …
  -/
  rw_mod_cast [ascPochhammer_nat_eq_ascFactorial, Nat.factorial_mul_ascFactorial]
  /-
    🎉 no goals
  -/


theorem ascPochhammer_nat_eval_succ (r : ℕ) :
    ∀ n : ℕ, n * (ascPochhammer ℕ r).eval (n + 1) = (n + r) * (ascPochhammer ℕ r).eval n
  | 0 => by
    /-
      r : Nat
      ⊢ Eq (HMul.hMul 0 (Polynomial.eval (HAdd.hAdd 0 1) (ascPochhammer Nat r))) (HM …
    -/
    by_cases h : r = 0
      /-
        case pos
        r : Nat
        h : Eq r 0
        ⊢ Eq (HMul.hMul 0 (Polynomial.eval (HAdd.hAdd 0 1) (ascPochhammer Nat r))) (HM …
      -/
    · simp only [h, zero_mul, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case neg
        r : Nat
        h : Not (Eq r 0)
        ⊢ Eq (HMul.hMul 0 (Polynomial.eval (HAdd.hAdd 0 1) (ascPochhammer Nat r))) (HM …
      -/
    · simp only [ascPochhammer_eval_zero, zero_mul, if_neg h, mul_zero]
      /-
        🎉 no goals
      -/
                /-
                  r k : Nat
                  ⊢ Eq (HMul.hMul (HAdd.hAdd k 1) (Polynomial.eval (HAdd.hAdd (HAdd.hAdd k 1) 1) …
                -/
  | k + 1 => by simp only [ascPochhammer_nat_eq_ascFactorial, Nat.succ_ascFactorial, add_right_comm]
                /-
                  🎉 no goals
                -/


theorem ascPochhammer_eval_succ (r n : ℕ) :
    (n : S) * (ascPochhammer S r).eval (n + 1 : S) =
    (n + r) * (ascPochhammer S r).eval (n : S) :=
  mod_cast congr_arg Nat.cast (ascPochhammer_nat_eval_succ r n)


/-- `descPochhammer R n` is the polynomial `X * (X - 1) * ... * (X - n + 1)`,
with coefficients in the ring `R`.
-/
noncomputable def descPochhammer : ℕ → R[X]
  | 0 => 1
  | n + 1 => X * (descPochhammer n).comp (X - 1)


@[simp]
theorem descPochhammer_zero : descPochhammer R 0 = 1 :=
  rfl


@[simp]
                                                          /-
                                                            R : Type u
                                                            inst✝ : Ring R
                                                            ⊢ Eq (descPochhammer R 1) Polynomial.X
                                                          -/
theorem descPochhammer_one : descPochhammer R 1 = X := by simp [descPochhammer]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem descPochhammer_succ_left (n : ℕ) :
    descPochhammer R (n + 1) = X * (descPochhammer R n).comp (X - 1) := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (descPochhammer R (HAdd.hAdd n 1)) (HMul.hMul Polynomial.X ((descPochhamm …
  -/
  rw [descPochhammer]
  /-
    🎉 no goals
  -/


theorem monic_descPochhammer (n : ℕ) [Nontrivial R] [NoZeroDivisors R] :
    Monic <| descPochhammer R n := by
  /-
    R : Type u
    inst✝² : Ring R
    n : Nat
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    ⊢ (descPochhammer R n).Monic
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u
      inst✝² : Ring R
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      ⊢ (descPochhammer R 0).Monic
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝² : Ring R
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      n : Nat
      hn : (descPochhammer R n).Monic
      ⊢ (descPochhammer R (HAdd.hAdd n 1)).Monic
    -/
  · have h : leadingCoeff (X - 1 : R[X]) = 1 := leadingCoeff_X_sub_C 1
    /-
      case succ
      R : Type u
      inst✝² : Ring R
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      n : Nat
      hn : (descPochhammer R n).Monic
      h : Eq (HSub.hSub Polynomial.X 1).leadingCoeff 1
      ⊢ (descPochhammer R (HAdd.hAdd n 1)).Monic
    -/
    have : natDegree (X - (1 : R[X])) ≠ 0 := ne_zero_of_eq_one <| natDegree_X_sub_C (1 : R)
    rw [descPochhammer_succ_left, Monic.def, leadingCoeff_mul, leadingCoeff_comp this, hn, monic_X,
        one_mul, one_mul, h, one_pow]


@[simp]
theorem descPochhammer_map (f : R →+* T) (n : ℕ) :
    (descPochhammer R n).map f = descPochhammer T n := by
  induction n with
  | zero => simp
  | succ n ih => simp [ih, descPochhammer_succ_left, map_comp]

@[simp, norm_cast]
theorem descPochhammer_eval_cast (n : ℕ) (k : ℤ) :
    (((descPochhammer ℤ n).eval k : ℤ) : R) = ((descPochhammer R n).eval k : R) := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    k : Int
    ⊢ Eq (↑(Polynomial.eval k (descPochhammer Int n))) (Polynomial.eval (↑k) (desc …
  -/
  rw [← descPochhammer_map (algebraMap ℤ R), eval_map, ← eq_intCast (algebraMap ℤ R)]
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    k : Int
    ⊢ Eq ((algebraMap Int R) (Polynomial.eval k (descPochhammer Int n))) (Polynomi …
  -/
  simp only [algebraMap_int_eq, eq_intCast, eval₂_at_intCast, Nat.cast_id, eq_natCast, Int.cast_id]
  /-
    🎉 no goals
  -/


theorem descPochhammer_eval_zero {n : ℕ} :
    (descPochhammer R n).eval 0 = if n = 0 then 1 else 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (Polynomial.eval 0 (descPochhammer R n)) (ite (Eq n 0) 1 0)
  -/
  cases n
    /-
      case zero
      R : Type u
      inst✝ : Ring R
      ⊢ Eq (Polynomial.eval 0 (descPochhammer R 0)) (ite (Eq 0 0) 1 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : Ring R
      n✝ : Nat
      ⊢ Eq (Polynomial.eval 0 (descPochhammer R (HAdd.hAdd n✝ 1))) (ite (Eq (HAdd.hA …
    -/
  · simp [X_mul, Nat.succ_ne_zero, descPochhammer_succ_left]
    /-
      🎉 no goals
    -/


                                                                              /-
                                                                                R : Type u
                                                                                inst✝ : Ring R
                                                                                ⊢ Eq (Polynomial.eval 0 (descPochhammer R 0)) 1
                                                                              -/
theorem descPochhammer_zero_eval_zero : (descPochhammer R 0).eval 0 = 1 := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem descPochhammer_ne_zero_eval_zero {n : ℕ} (h : n ≠ 0) : (descPochhammer R n).eval 0 = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    h : Ne n 0
    ⊢ Eq (Polynomial.eval 0 (descPochhammer R n)) 0
  -/
  simp [descPochhammer_eval_zero, h]
  /-
    🎉 no goals
  -/


theorem descPochhammer_succ_right (n : ℕ) :
    descPochhammer R (n + 1) = descPochhammer R n * (X - (n : R[X])) := by
  suffices h : descPochhammer ℤ (n + 1) = descPochhammer ℤ n * (X - (n : ℤ[X])) by
    apply_fun Polynomial.map (algebraMap ℤ R) at h
    simpa [descPochhammer_map, Polynomial.map_mul, Polynomial.map_add, map_X,
      Polynomial.map_intCast] using h
  induction n with
  | zero => simp [descPochhammer]
  | succ n ih =>
    conv_lhs =>
      rw [descPochhammer_succ_left, ih, mul_comp, ← mul_assoc, ← descPochhammer_succ_left, sub_comp,
          X_comp, natCast_comp]
    rw [Nat.cast_add, Nat.cast_one, sub_add_eq_sub_sub_swap]


@[simp]
theorem descPochhammer_natDegree (n : ℕ) [NoZeroDivisors R] [Nontrivial R] :
    (descPochhammer R n).natDegree = n := by
  /-
    R : Type u
    inst✝² : Ring R
    n : Nat
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    ⊢ Eq (descPochhammer R n).natDegree n
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u
      inst✝² : Ring R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      ⊢ Eq (descPochhammer R 0).natDegree 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝² : Ring R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      n : Nat
      hn : Eq (descPochhammer R n).natDegree n
      ⊢ Eq (descPochhammer R (HAdd.hAdd n 1)).natDegree (HAdd.hAdd n 1)
    -/
  · have : natDegree (X - (n : R[X])) = 1 := natDegree_X_sub_C (n : R)
    rw [descPochhammer_succ_right,
        natDegree_mul _ (ne_zero_of_natDegree_gt <| this.symm ▸ Nat.zero_lt_one), hn, this]
    /-
      R : Type u
      inst✝² : Ring R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      n : Nat
      hn : Eq (descPochhammer R n).natDegree n
      this : Eq (HSub.hSub Polynomial.X ↑n).natDegree 1
      ⊢ Ne (descPochhammer R n) 0
    -/
    cases n
      /-
        case zero
        R : Type u
        inst✝² : Ring R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        hn : Eq (descPochhammer R 0).natDegree 0
        this : Eq (HSub.hSub Polynomial.X ↑0).natDegree 1
        ⊢ Ne (descPochhammer R 0) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ
        R : Type u
        inst✝² : Ring R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        n✝ : Nat
        hn : Eq (descPochhammer R (HAdd.hAdd n✝ 1)).natDegree (HAdd.hAdd n✝ 1)
        this : Eq (HSub.hSub Polynomial.X ↑(HAdd.hAdd n✝ 1)).natDegree 1
        ⊢ Ne (descPochhammer R (HAdd.hAdd n✝ 1)) 0
      -/
    · refine ne_zero_of_natDegree_gt <| hn.symm ▸ Nat.add_one_pos _
      /-
        🎉 no goals
      -/


theorem descPochhammer_succ_eval {S : Type*} [Ring S] (n : ℕ) (k : S) :
    (descPochhammer S (n + 1)).eval k = (descPochhammer S n).eval k * (k - n) := by
  rw [descPochhammer_succ_right, mul_sub, eval_sub, eval_mul_X, ← Nat.cast_comm, ← C_eq_natCast,
    eval_C_mul, Nat.cast_comm, ← mul_sub]


theorem descPochhammer_succ_comp_X_sub_one (n : ℕ) :
    (descPochhammer R (n + 1)).comp (X - 1) =
      descPochhammer R (n + 1) - (n + (1 : R[X])) • (descPochhammer R n).comp (X - 1) := by
  suffices (descPochhammer ℤ (n + 1)).comp (X - 1) =
      descPochhammer ℤ (n + 1) - (n + 1) * (descPochhammer ℤ n).comp (X - 1)
    by simpa [map_comp] using congr_arg (Polynomial.map (Int.castRingHom R)) this
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq ((descPochhammer Int (HAdd.hAdd n 1)).comp (HSub.hSub Polynomial.X 1)) (H …
  -/
  nth_rw 2 [descPochhammer_succ_left]
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq ((descPochhammer Int (HAdd.hAdd n 1)).comp (HSub.hSub Polynomial.X 1)) (H …
  -/
  rw [← sub_mul, descPochhammer_succ_right ℤ n, mul_comp, mul_comm, sub_comp, X_comp, natCast_comp]
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub (HSub.hSub Polynomial.X 1) ↑n) ((descPochhammer Int …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem descPochhammer_eq_ascPochhammer (n : ℕ) :
    descPochhammer ℤ n = (ascPochhammer ℤ n).comp ((X : ℤ[X]) - n + 1) := by
  induction n with
  | zero => rw [descPochhammer_zero, ascPochhammer_zero, one_comp]
  | succ n ih =>
    rw [Nat.cast_succ, sub_add, add_sub_cancel_right, descPochhammer_succ_right,
      ascPochhammer_succ_left, ih, X_mul, mul_X_comp, comp_assoc, add_comp, X_comp, one_comp]


theorem descPochhammer_eval_eq_ascPochhammer (r : R) (n : ℕ) :
    (descPochhammer R n).eval r = (ascPochhammer R n).eval (r - n + 1) := by
  induction n with
  | zero => rw [descPochhammer_zero, eval_one, ascPochhammer_zero, eval_one]
  | succ n ih =>
    rw [Nat.cast_succ, sub_add, add_sub_cancel_right, descPochhammer_succ_eval, ih,
      ascPochhammer_succ_left, X_mul, eval_mul_X, show (X + 1 : R[X]) =
      (X + 1 : ℕ[X]).map (algebraMap ℕ R) by simp only [Polynomial.map_add, map_X,
      Polynomial.map_one], ascPochhammer_eval_comp, eval₂_add, eval₂_X, eval₂_one]


theorem descPochhammer_mul (n m : ℕ) :
    descPochhammer R n * (descPochhammer R m).comp (X - (n : R[X])) = descPochhammer R (n + m) := by
  /-
    R : Type u
    inst✝ : Ring R
    n m : Nat
    ⊢ Eq (HMul.hMul (descPochhammer R n) ((descPochhammer R m).comp (HSub.hSub Pol …
  -/
  induction' m with m ih
    /-
      case zero
      R : Type u
      inst✝ : Ring R
      n : Nat
      ⊢ Eq (HMul.hMul (descPochhammer R n) ((descPochhammer R 0).comp (HSub.hSub Pol …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [descPochhammer_succ_right, Polynomial.mul_X_sub_intCast_comp, ← mul_assoc, ih,
      ← add_assoc, descPochhammer_succ_right, Nat.cast_add, sub_add_eq_sub_sub]


theorem ascPochhammer_eval_neg_eq_descPochhammer (r : R) : ∀ (k : ℕ),
    (ascPochhammer R k).eval (-r) = (-1)^k * (descPochhammer R k).eval r
  | 0 => by
    /-
      R : Type u
      inst✝ : Ring R
      r : R
      ⊢ Eq (Polynomial.eval (Neg.neg r) (ascPochhammer R 0)) (HMul.hMul (HPow.hPow ( …
    -/
    rw [ascPochhammer_zero, descPochhammer_zero]
    /-
      R : Type u
      inst✝ : Ring R
      r : R
      ⊢ Eq (Polynomial.eval (Neg.neg r) 1) (HMul.hMul (HPow.hPow (-1) 0) (Polynomial …
    -/
    simp only [eval_one, pow_zero, mul_one]
    /-
      🎉 no goals
    -/
  | (k+1) => by
    rw [ascPochhammer_succ_right, mul_add, eval_add, eval_mul_X, ← Nat.cast_comm, eval_natCast_mul,
      Nat.cast_comm, ← mul_add, ascPochhammer_eval_neg_eq_descPochhammer r k, mul_assoc,
      descPochhammer_succ_right, mul_sub, eval_sub, eval_mul_X, ← Nat.cast_comm, eval_natCast_mul,
      pow_add, pow_one, mul_assoc ((-1)^k) (-1), mul_sub, neg_one_mul, neg_mul_eq_mul_neg,
      Nat.cast_comm, sub_eq_add_neg, neg_one_mul, neg_neg, ← mul_add]


theorem descPochhammer_eval_eq_descFactorial (n k : ℕ) :
    (descPochhammer R k).eval (n : R) = n.descFactorial k := by
  induction k with
  | zero => rw [descPochhammer_zero, eval_one, Nat.descFactorial_zero, Nat.cast_one]
  | succ k ih =>
    rw [descPochhammer_succ_right, Nat.descFactorial_succ, mul_sub, eval_sub, eval_mul_X,
      ← Nat.cast_comm k, eval_natCast_mul, ← Nat.cast_comm n, ← sub_mul, ih]
    by_cases h : n < k
    · rw [Nat.descFactorial_eq_zero_iff_lt.mpr h, Nat.cast_zero, mul_zero, mul_zero, Nat.cast_zero]
    · rw [Nat.cast_mul, Nat.cast_sub <| not_lt.mp h]


theorem descPochhammer_int_eq_ascFactorial (a b : ℕ) :
    (descPochhammer ℤ b).eval (a + b : ℤ) = (a + 1).ascFactorial b := by
  rw [← Nat.cast_add, descPochhammer_eval_eq_descFactorial ℤ (a + b) b,
    Nat.add_descFactorial_eq_ascFactorial]


/-- The Pochhammer polynomial of degree `n` has roots at `0`, `-1`, ..., `-(n - 1)`. -/
theorem ascPochhammer_eval_neg_coe_nat_of_lt {n k : ℕ} (h : k < n) :
    (ascPochhammer R n).eval (-(k : R)) = 0 := by
  induction n with
  | zero => contradiction
  | succ n ih =>
    rw [ascPochhammer_succ_eval]
    rcases lt_trichotomy k n with hkn | rfl | hkn
    · simp [ih hkn]
    · simp
    · omega


/-- Over an integral domain, the Pochhammer polynomial of degree `n` has roots *only* at
`0`, `-1`, ..., `-(n - 1)`. -/
@[simp]
theorem ascPochhammer_eval_eq_zero_iff [IsDomain R]
    (n : ℕ) (r : R) : (ascPochhammer R n).eval r = 0 ↔ ∃ k < n, k = -r := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : IsDomain R
    n : Nat
    r : R
    ⊢ Iff (Eq (Polynomial.eval r (ascPochhammer R n)) 0) (Exists fun k => And (LT. …
  -/
  refine ⟨fun zero' ↦ ?_, fun hrn ↦ ?_⟩
  · induction n with
    | zero => simp only [ascPochhammer_zero, Polynomial.eval_one, one_ne_zero] at zero'
    | succ n ih =>
      rw [ascPochhammer_succ_eval, mul_eq_zero] at zero'
      cases zero' with
      | inl h =>
        obtain ⟨rn, hrn, rrn⟩ := ih h
        exact ⟨rn, by omega, rrn⟩
      | inr h =>
        exact ⟨n, lt_add_one n, eq_neg_of_add_eq_zero_right h⟩
    /-
      case refine_2
      R : Type u
      inst✝¹ : Ring R
      inst✝ : IsDomain R
      n : Nat
      r : R
      hrn : Exists fun k => And (LT.lt k n) (Eq (↑k) (Neg.neg r))
      ⊢ Eq (Polynomial.eval r (ascPochhammer R n)) 0
    -/
  · obtain ⟨rn, hrn, rnn⟩ := hrn
    /-
      case refine_2.intro.intro
      R : Type u
      inst✝¹ : Ring R
      inst✝ : IsDomain R
      n : Nat
      r : R
      rn : Nat
      hrn : LT.lt rn n
      rnn : Eq (↑rn) (Neg.neg r)
      ⊢ Eq (Polynomial.eval r (ascPochhammer R n)) 0
    -/
    convert ascPochhammer_eval_neg_coe_nat_of_lt hrn
    /-
      case h.e'_2.h.e'_3
      R : Type u
      inst✝¹ : Ring R
      inst✝ : IsDomain R
      n : Nat
      r : R
      rn : Nat
      hrn : LT.lt rn n
      rnn : Eq (↑rn) (Neg.neg r)
      ⊢ Eq r (Neg.neg ↑rn)
    -/
    simp [rnn]
    /-
      🎉 no goals
    -/


