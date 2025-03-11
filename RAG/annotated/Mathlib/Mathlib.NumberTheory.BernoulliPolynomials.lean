/-- The Bernoulli polynomials are defined in terms of the negative Bernoulli numbers. -/
def bernoulli (n : ℕ) : ℚ[X] :=
  ∑ i ∈ range (n + 1), Polynomial.monomial (n - i) (_root_.bernoulli i * choose n i)


theorem bernoulli_def (n : ℕ) : bernoulli n =
    ∑ i ∈ range (n + 1), Polynomial.monomial i (_root_.bernoulli (n - i) * choose n i) := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.bernoulli n) ((Finset.range (HAdd.hAdd n 1)).sum fun i => (Po …
  -/
  rw [← sum_range_reflect, add_succ_sub_one, add_zero, bernoulli]
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => (Polynomial.monomial (HSub.h …
  -/
  apply sum_congr rfl
  /-
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq ((Polynomi …
  -/
  rintro x hx
  /-
    n x : Nat
    hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
    ⊢ Eq ((Polynomial.monomial (HSub.hSub n x)) (HMul.hMul (_root_.bernoulli x) ↑( …
  -/
  rw [mem_range_succ_iff] at hx
  /-
    n x : Nat
    hx : LE.le x n
    ⊢ Eq ((Polynomial.monomial (HSub.hSub n x)) (HMul.hMul (_root_.bernoulli x) ↑( …
  -/
  rw [choose_symm hx, tsub_tsub_cancel_of_le hx]
  /-
    🎉 no goals
  -/

/-
### examples
-/

@[simp]
                                               /-
                                                 ⊢ Eq (Polynomial.bernoulli 0) 1
                                               -/
theorem bernoulli_zero : bernoulli 0 = 1 := by simp [bernoulli]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem bernoulli_eval_zero (n : ℕ) : (bernoulli n).eval 0 = _root_.bernoulli n := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.eval 0 (Polynomial.bernoulli n)) (_root_.bernoulli n)
  -/
  rw [bernoulli, eval_finset_sum, sum_range_succ]
  have : ∑ x ∈ range n, _root_.bernoulli x * n.choose x * 0 ^ (n - x) = 0 := by
    apply sum_eq_zero fun x hx => _
    intros x hx
    simp [tsub_eq_zero_iff_le, mem_range.1 hx]
  /-
    n : Nat
    this : Eq ((Finset.range n).sum fun x => HMul.hMul (HMul.hMul (_root_.bernoull …
    ⊢ Eq (HAdd.hAdd ((Finset.range n).sum fun x => Polynomial.eval 0 ((Polynomial. …
  -/
  simp [this]
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli_eval_one (n : ℕ) : (bernoulli n).eval 1 = bernoulli' n := by
  /-
    n : Nat
    ⊢ Eq (Polynomial.eval 1 (Polynomial.bernoulli n)) (bernoulli' n)
  -/
  simp only [bernoulli, eval_finset_sum]
  simp only [← succ_eq_add_one, sum_range_succ, mul_one, cast_one, choose_self,
    (_root_.bernoulli _).mul_comm, sum_bernoulli, one_pow, mul_one, eval_C, eval_monomial, one_mul]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (ite (Eq n 1) 1 0) (_root_.bernoulli n)) (bernoulli' n)
  -/
  by_cases h : n = 1
    /-
      case pos
      n : Nat
      h : Eq n 1
      ⊢ Eq (HAdd.hAdd (ite (Eq n 1) 1 0) (_root_.bernoulli n)) (bernoulli' n)
    -/
  · norm_num [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      h : Not (Eq n 1)
      ⊢ Eq (HAdd.hAdd (ite (Eq n 1) 1 0) (_root_.bernoulli n)) (bernoulli' n)
    -/
  · simp [h, bernoulli_eq_bernoulli'_of_ne_one h]
    /-
      🎉 no goals
    -/


theorem derivative_bernoulli_add_one (k : ℕ) :
    Polynomial.derivative (bernoulli (k + 1)) = (k + 1) * bernoulli k := by
  /-
    k : Nat
    ⊢ Eq (Polynomial.derivative (Polynomial.bernoulli (HAdd.hAdd k 1))) (HMul.hMul …
  -/
  simp_rw [bernoulli, derivative_sum, derivative_monomial, Nat.sub_sub, Nat.add_sub_add_right]
  -- LHS sum has an extra term, but the coefficient is zero:
  rw [range_add_one, sum_insert not_mem_range_self, tsub_self, cast_zero, mul_zero,
    map_zero, zero_add, mul_sum]
  -- the rest of the sum is termwise equal:
  /-
    k : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd k 1)).sum fun x => (Polynomial.monomial (HSub.h …
  -/
  refine sum_congr (by rfl) fun m _ => ?_
  /-
    k m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) m
    ⊢ Eq ((Polynomial.monomial (HSub.hSub k m)) (HMul.hMul (HMul.hMul (_root_.bern …
  -/
  conv_rhs => rw [← Nat.cast_one, ← Nat.cast_add, ← C_eq_natCast, C_mul_monomial, mul_comm]
  /-
    k m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) m
    ⊢ Eq ((Polynomial.monomial (HSub.hSub k m)) (HMul.hMul (HMul.hMul (_root_.bern …
  -/
  rw [mul_assoc, mul_assoc, ← Nat.cast_mul, ← Nat.cast_mul]
  /-
    k m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) m
    ⊢ Eq ((Polynomial.monomial (HSub.hSub k m)) (HMul.hMul (_root_.bernoulli m) ↑( …
  -/
  congr 3
  /-
    case h.e_6.h.e_a.e_a
    k m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) m
    ⊢ Eq (HMul.hMul ((HAdd.hAdd k 1).choose m) (HSub.hSub (HAdd.hAdd k 1) m)) (HMu …
  -/
  rw [(choose_mul_succ_eq k m).symm]
  /-
    🎉 no goals
  -/


theorem derivative_bernoulli (k : ℕ) :
    Polynomial.derivative (bernoulli k) = k * bernoulli (k - 1) := by
  cases k with
  | zero => rw [Nat.cast_zero, zero_mul, bernoulli_zero, derivative_one]
  | succ k => exact mod_cast derivative_bernoulli_add_one k


@[simp]
nonrec theorem sum_bernoulli (n : ℕ) :
    (∑ k ∈ range (n + 1), ((n + 1).choose k : ℚ) • bernoulli k) = monomial n (n + 1 : ℚ) := by
  simp_rw [bernoulli_def, Finset.smul_sum, Finset.range_eq_Ico, ← Finset.sum_Ico_Ico_comm,
    Finset.sum_Ico_eq_sum_range]
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HSub.hSub (HAdd.hAdd n 1) 0)).sum fun k => (Finset.range  …
  -/
  simp only [add_tsub_cancel_left, tsub_zero, zero_add, map_add]
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => (Finset.range (HSub.hSub (HA …
  -/
  simp_rw [smul_monomial, mul_comm (_root_.bernoulli _) _, smul_eq_mul, ← mul_assoc]
  conv_lhs =>
    apply_congr
    · skip
    · conv =>
      apply_congr
      · skip
      · rw [← Nat.cast_mul, choose_mul ((le_tsub_iff_left <| mem_range_le (by assumption)).1 <|
            mem_range_le (by assumption)) (le.intro rfl),
          Nat.cast_mul, add_tsub_cancel_left, mul_assoc, mul_comm, ← smul_eq_mul, ←
          smul_monomial]
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => (Finset.range (HSub.hSub (HA …
  -/
  simp_rw [← sum_smul]
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HSMul.hSMul ((Finset.range ( …
  -/
  rw [sum_range_succ_comm]
  simp only [add_right_eq_self, mul_one, cast_one, cast_add, add_tsub_cancel_left,
    choose_succ_self_right, one_smul, _root_.bernoulli_zero, sum_singleton, zero_add,
    map_add, range_one, bernoulli_zero, mul_one, one_mul, add_zero, choose_self]
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun x => HSMul.hSMul ((Finset.range (HSub.hSub (HAd …
  -/
  apply sum_eq_zero fun x hx => _
  have f : ∀ x ∈ range n, ¬n + 1 - x = 1 := by
    rintro x H
    rw [mem_range] at H
    rw [eq_comm]
    exact _root_.ne_of_lt (Nat.lt_of_lt_of_le one_lt_two (le_tsub_of_add_le_left (succ_le_succ H)))
  /-
    n : Nat
    f : ∀ (x : Nat), Membership.mem (Finset.range n) x → Not (Eq (HSub.hSub (HAdd. …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → Eq (HSMul.hSMul ((Finset.ra …
  -/
  intro x hx
  /-
    n : Nat
    f : ∀ (x : Nat), Membership.mem (Finset.range n) x → Not (Eq (HSub.hSub (HAdd. …
    x : Nat
    hx : Membership.mem (Finset.range n) x
    ⊢ Eq (HSMul.hSMul ((Finset.range (HSub.hSub (HAdd.hAdd n 1) x)).sum fun i => H …
  -/
  rw [sum_bernoulli]
  have g : ite (n + 1 - x = 1) (1 : ℚ) 0 = 0 := by
    simp only [ite_eq_right_iff, one_ne_zero]
    intro h₁
    exact (f x hx) h₁
  /-
    n : Nat
    f : ∀ (x : Nat), Membership.mem (Finset.range n) x → Not (Eq (HSub.hSub (HAdd. …
    x : Nat
    hx : Membership.mem (Finset.range n) x
    g : Eq (ite (Eq (HSub.hSub (HAdd.hAdd n 1) x) 1) 1 0) 0
    ⊢ Eq (HSMul.hSMul (ite (Eq (HSub.hSub (HAdd.hAdd n 1) x) 1) 1 0) ((Polynomial. …
  -/
  rw [g, zero_smul]
  /-
    🎉 no goals
  -/


/-- Another version of `Polynomial.sum_bernoulli`. -/
theorem bernoulli_eq_sub_sum (n : ℕ) :
    (n.succ : ℚ) • bernoulli n =
      monomial n (n.succ : ℚ) - ∑ k ∈ Finset.range n, ((n + 1).choose k : ℚ) • bernoulli k := by
  rw [Nat.cast_succ, ← sum_bernoulli n, sum_range_succ, add_sub_cancel_left, choose_succ_self_right,
    Nat.cast_succ]


/-- Another version of `sum_range_pow`. -/
theorem sum_range_pow_eq_bernoulli_sub (n p : ℕ) :
    ((p + 1 : ℚ) * ∑ k ∈ range n, (k : ℚ) ^ p) = (bernoulli p.succ).eval (n : ℚ) -
    _root_.bernoulli p.succ := by
  /-
    n p : Nat
    ⊢ Eq (HMul.hMul (HAdd.hAdd (↑p) 1) ((Finset.range n).sum fun k => HPow.hPow (↑ …
  -/
  rw [sum_range_pow, bernoulli_def, eval_finset_sum, ← sum_div, mul_div_cancel₀ _ _]
    /-
      n p : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd p 1)).sum fun i => HMul.hMul (HMul.hMul (_root_ …
    -/
  · simp_rw [eval_monomial]
    /-
      n p : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd p 1)).sum fun i => HMul.hMul (HMul.hMul (_root_ …
    -/
    symm
    /-
      n p : Nat
      ⊢ Eq (HSub.hSub ((Finset.range (HAdd.hAdd p.succ 1)).sum fun x => HMul.hMul (H …
    -/
    rw [← sum_flip _, sum_range_succ]
    simp only [tsub_self, tsub_zero, choose_zero_right, cast_one, mul_one, _root_.pow_zero,
      add_tsub_cancel_right]
    /-
      n p : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd p 1)).sum fun x => HMul.hMul (HMul.hMul (_root_ …
    -/
    apply sum_congr rfl fun x hx => _
    /-
      n p : Nat
      ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd p 1)) x → Eq (HMul.hMul …
    -/
    intro x hx
    /-
      n p x : Nat
      hx : Membership.mem (Finset.range (HAdd.hAdd p 1)) x
      ⊢ Eq (HMul.hMul (HMul.hMul (_root_.bernoulli (HSub.hSub p.succ (HSub.hSub (HAd …
    -/
    apply congr_arg₂ _ (congr_arg₂ _ _ _) rfl
      /-
        n p x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd p 1)) x
        ⊢ Eq (_root_.bernoulli (HSub.hSub p.succ (HSub.hSub (HAdd.hAdd p 1) x))) (_roo …
      -/
    · rw [Nat.sub_sub_self (mem_range_le hx)]
      /-
        🎉 no goals
      -/
      /-
        n p x : Nat
        hx : Membership.mem (Finset.range (HAdd.hAdd p 1)) x
        ⊢ Eq ↑(p.succ.choose (HSub.hSub (HAdd.hAdd p 1) x)) ↑((HAdd.hAdd p 1).choose x)
      -/
    · rw [← choose_symm (mem_range_le hx)]
      /-
        🎉 no goals
      -/
    /-
      n p : Nat
      ⊢ Ne (HAdd.hAdd (↑p) 1) 0
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


/-- Rearrangement of `Polynomial.sum_range_pow_eq_bernoulli_sub`. -/
theorem bernoulli_succ_eval (n p : ℕ) : (bernoulli p.succ).eval (n : ℚ) =
    _root_.bernoulli p.succ + (p + 1 : ℚ) * ∑ k ∈ range n, (k : ℚ) ^ p := by
  /-
    n p : Nat
    ⊢ Eq (Polynomial.eval (↑n) (Polynomial.bernoulli p.succ)) (HAdd.hAdd (_root_.b …
  -/
  apply eq_add_of_sub_eq'
  /-
    case h
    n p : Nat
    ⊢ Eq (HSub.hSub (Polynomial.eval (↑n) (Polynomial.bernoulli p.succ)) (_root_.b …
  -/
  rw [sum_range_pow_eq_bernoulli_sub]
  /-
    🎉 no goals
  -/


theorem bernoulli_eval_one_add (n : ℕ) (x : ℚ) :
    (bernoulli n).eval (1 + x) = (bernoulli n).eval x + n * x ^ (n - 1) := by
  /-
    n : Nat
    x : Rat
    ⊢ Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial.bernoulli n)) (HAdd.hAdd (Po …
  -/
  refine Nat.strong_induction_on n fun d hd => ?_
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    ⊢ Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial.bernoulli d)) (HAdd.hAdd (Po …
  -/
  have nz : ((d.succ : ℕ) : ℚ) ≠ 0 := by norm_cast
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial.bernoulli d)) (HAdd.hAdd (Po …
  -/
  apply (mul_right_inj' nz).1
  rw [← smul_eq_mul, ← eval_smul, bernoulli_eq_sub_sum, mul_add, ← smul_eq_mul, ← eval_smul,
    bernoulli_eq_sub_sum, eval_sub, eval_finset_sum]
  conv_lhs =>
    congr
    · skip
    · apply_congr
      · skip
      · rw [eval_smul, hd _ (mem_range.1 (by assumption))]
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 x) ((Polynomial.monomial d) ↑d.s …
  -/
  rw [eval_sub, eval_finset_sum]
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 x) ((Polynomial.monomial d) ↑d.s …
  -/
  simp_rw [eval_smul, smul_add]
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 x) ((Polynomial.monomial d) ↑d.s …
  -/
  rw [sum_add_distrib, sub_add, sub_eq_sub_iff_sub_eq_sub, _root_.add_sub_sub_cancel]
  conv_rhs =>
    congr
    · skip
    · congr
      rw [succ_eq_add_one, ← choose_succ_self_right d]
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 x) ((Polynomial.monomial d) ↑d.s …
  -/
  rw [Nat.cast_succ, ← smul_eq_mul, ← sum_range_succ _ d, eval_monomial_one_add_sub]
  /-
    n : Nat
    x : Rat
    d : Nat
    hd : ∀ (m : Nat), LT.lt m d → Eq (Polynomial.eval (HAdd.hAdd 1 x) (Polynomial. …
    nz : Ne (↑d.succ) 0
    ⊢ Eq ((Finset.range (HAdd.hAdd d 1)).sum fun x_1 => HMul.hMul (↑((HAdd.hAdd d  …
  -/
  simp_rw [smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- The theorem that $(e^X - 1) * ∑ Bₙ(t)* X^n/n! = Xe^{tX}$ -/
theorem bernoulli_generating_function (t : A) :
    (mk fun n => aeval t ((1 / n ! : ℚ) • bernoulli n)) * (exp A - 1) =
      PowerSeries.X * rescale t (exp A) := by
  -- check equality of power series by checking coefficients of X^n
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    ⊢ Eq (HMul.hMul (PowerSeries.mk fun n => (Polynomial.aeval t) (HSMul.hSMul (HD …
  -/
  ext n
  -- n = 0 case solved by `simp`
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    ⊢ Eq ((PowerSeries.coeff A n) (HMul.hMul (PowerSeries.mk fun n => (Polynomial. …
  -/
  cases' n with n
    /-
      case h.zero
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      t : A
      ⊢ Eq ((PowerSeries.coeff A 0) (HMul.hMul (PowerSeries.mk fun n => (Polynomial. …
    -/
  · simp
    /-
      🎉 no goals
    -/
  -- n ≥ 1, the coefficients is a sum to n+2, so use `sum_range_succ` to write as
  -- last term plus sum to n+1
  rw [coeff_succ_X_mul, coeff_rescale, coeff_exp, PowerSeries.coeff_mul,
    Nat.sum_antidiagonal_eq_sum_range_succ_mk, sum_range_succ]
  -- last term is zero so kill with `add_zero`
  simp only [RingHom.map_sub, tsub_self, constantCoeff_one, constantCoeff_exp,
    coeff_zero_eq_constantCoeff, mul_zero, sub_self, add_zero]
  -- Let's multiply both sides by (n+1)! (OK because it's a unit)
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul ((PowerSeries.coef …
  -/
  have hnp1 : IsUnit ((n + 1)! : ℚ) := IsUnit.mk0 _ (mod_cast factorial_ne_zero (n + 1))
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul ((PowerSeries.coef …
  -/
  rw [← (hnp1.map (algebraMap ℚ A)).mul_right_inj]
  -- do trivial rearrangements to make RHS (n+1)*t^n
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ Eq (HMul.hMul ((algebraMap Rat A) ↑(HAdd.hAdd n 1).factorial) ((Finset.range …
  -/
  rw [mul_left_comm, ← RingHom.map_mul]
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ Eq (HMul.hMul ((algebraMap Rat A) ↑(HAdd.hAdd n 1).factorial) ((Finset.range …
  -/
  change _ = t ^ n * algebraMap ℚ A (((n + 1) * n ! : ℕ) * (1 / n !))
  rw [cast_mul, mul_assoc,
    mul_one_div_cancel (show (n ! : ℚ) ≠ 0 from cast_ne_zero.2 (factorial_ne_zero n)), mul_one,
    mul_comm (t ^ n), ← aeval_monomial, cast_add, cast_one]
  -- But this is the RHS of `Polynomial.sum_bernoulli`
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ Eq (HMul.hMul ((algebraMap Rat A) ↑(HAdd.hAdd n 1).factorial) ((Finset.range …
  -/
  rw [← sum_bernoulli, Finset.mul_sum, map_sum]
  -- and now we have to prove a sum is a sum, but all the terms are equal.
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HMul.hMul ((algebraMap Rat A …
  -/
  apply Finset.sum_congr rfl
  -- The rest is just trivialities, hampered by the fact that we're coercing
  -- factorials and binomial coefficients between ℕ and ℚ and A.
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (HMul.hMul …
  -/
  intro i hi
  -- deal with coefficients of e^X-1
  simp only [Nat.cast_choose ℚ (mem_range_le hi), coeff_mk, if_neg (mem_range_sub_ne_zero hi),
    one_div, map_smul, PowerSeries.coeff_one, coeff_exp, sub_zero, LinearMap.map_sub,
    Algebra.smul_mul_assoc, Algebra.smul_def, mul_right_comm _ ((aeval t) _), ← mul_assoc, ←
    RingHom.map_mul, succ_eq_add_one, ← Polynomial.C_eq_algebraMap, Polynomial.aeval_mul,
    Polynomial.aeval_C]
  -- finally cancel the Bernoulli polynomial and the algebra_map
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    t : A
    n : Nat
    hnp1 : IsUnit ↑(HAdd.hAdd n 1).factorial
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Eq (HMul.hMul ((algebraMap Rat A) (HMul.hMul (HMul.hMul (↑(HAdd.hAdd n 1).fa …
  -/
  field_simp
  /-
    🎉 no goals
  -/


