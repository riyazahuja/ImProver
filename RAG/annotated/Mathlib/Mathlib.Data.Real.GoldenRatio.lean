/-- The golden ratio `φ := (1 + √5)/2`. -/
abbrev goldenRatio : ℝ := (1 + √5) / 2


/-- The conjugate of the golden ratio `ψ := (1 - √5)/2`. -/
abbrev goldenConj : ℝ := (1 - √5) / 2


@[inherit_doc goldenRatio] scoped[goldenRatio] notation "φ" => goldenRatio

@[inherit_doc goldenConj] scoped[goldenRatio] notation "ψ" => goldenConj

/-- The inverse of the golden ratio is the opposite of its conjugate. -/
theorem inv_gold : φ⁻¹ = -ψ := by
  /-
    ⊢ Eq (Inv.inv goldenRatio) (Neg.neg goldenConj)
  -/
  have : 1 + √5 ≠ 0 := ne_of_gt (add_pos (by norm_num) <| Real.sqrt_pos.mpr (by norm_num))
  /-
    this : Ne (HAdd.hAdd 1 (Real.sqrt 5)) 0
    ⊢ Eq (Inv.inv goldenRatio) (Neg.neg goldenConj)
  -/
  field_simp [sub_mul, mul_add]
  /-
    this : Ne (HAdd.hAdd 1 (Real.sqrt 5)) 0
    ⊢ Eq (HMul.hMul 2 2) (HSub.hSub 5 1)
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- The opposite of the golden ratio is the inverse of its conjugate. -/
theorem inv_goldConj : ψ⁻¹ = -φ := by
  /-
    ⊢ Eq (Inv.inv goldenConj) (Neg.neg goldenRatio)
  -/
  rw [inv_eq_iff_eq_inv, ← neg_inv, ← neg_eq_iff_eq_neg]
  /-
    ⊢ Eq (Neg.neg goldenConj) (Inv.inv goldenRatio)
  -/
  exact inv_gold.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem gold_mul_goldConj : φ * ψ = -1 := by
  /-
    ⊢ Eq (HMul.hMul goldenRatio goldenConj) (-1)
  -/
  field_simp
  /-
    ⊢ Eq (HMul.hMul (HAdd.hAdd 1 (Real.sqrt 5)) (HSub.hSub 1 (Real.sqrt 5))) (Neg. …
  -/
  rw [← sq_sub_sq]
  /-
    ⊢ Eq (HSub.hSub (HPow.hPow 1 2) (HPow.hPow (Real.sqrt 5) 2)) (Neg.neg (HMul.hM …
  -/
  norm_num
  /-
    🎉 no goals
  -/


@[simp]
theorem goldConj_mul_gold : ψ * φ = -1 := by
  /-
    ⊢ Eq (HMul.hMul goldenConj goldenRatio) (-1)
  -/
  rw [mul_comm]
  /-
    ⊢ Eq (HMul.hMul goldenRatio goldenConj) (-1)
  -/
  exact gold_mul_goldConj
  /-
    🎉 no goals
  -/


@[simp]
theorem gold_add_goldConj : φ + ψ = 1 := by
  /-
    ⊢ Eq (HAdd.hAdd goldenRatio goldenConj) 1
  -/
  rw [goldenRatio, goldenConj]
  /-
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd 1 (Real.sqrt 5)) 2) (HDiv.hDiv (HSub.hSu …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem one_sub_goldConj : 1 - φ = ψ := by
  /-
    ⊢ Eq (HSub.hSub 1 goldenRatio) goldenConj
  -/
  linarith [gold_add_goldConj]
  /-
    🎉 no goals
  -/


theorem one_sub_gold : 1 - ψ = φ := by
  /-
    ⊢ Eq (HSub.hSub 1 goldenConj) goldenRatio
  -/
  linarith [gold_add_goldConj]
  /-
    🎉 no goals
  -/


@[simp]
                                             /-
                                               ⊢ Eq (HSub.hSub goldenRatio goldenConj) (Real.sqrt 5)
                                             -/
theorem gold_sub_goldConj : φ - ψ = √5 := by ring
                                             /-
                                               🎉 no goals
                                             -/


theorem gold_pow_sub_gold_pow (n : ℕ) : φ ^ (n + 2) - φ ^ (n + 1) = φ ^ n := by
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (HPow.hPow goldenRatio (HAdd.hAdd n 2)) (HPow.hPow goldenRatio …
  -/
  rw [goldenRatio]; ring_nf; norm_num; ring
                                       /-
                                         🎉 no goals
                                       -/


@[simp 1200]
theorem gold_sq : φ ^ 2 = φ + 1 := by
  /-
    ⊢ Eq (HPow.hPow goldenRatio 2) (HAdd.hAdd goldenRatio 1)
  -/
  rw [goldenRatio, ← sub_eq_zero]
  /-
    ⊢ Eq (HSub.hSub (HPow.hPow (HDiv.hDiv (HAdd.hAdd 1 (Real.sqrt 5)) 2) 2) (HAdd. …
  -/
  ring_nf
  /-
    ⊢ Eq (HAdd.hAdd (-5 / 4) (HMul.hMul (HPow.hPow (Real.sqrt 5) 2) (1 / 4))) 0
  -/
                        /-
                          🎉 no goals
                        -/
  rw [Real.sq_sqrt] <;> norm_num
                        /-
                          🎉 no goals
                        -/


@[simp 1200]
theorem goldConj_sq : ψ ^ 2 = ψ + 1 := by
  /-
    ⊢ Eq (HPow.hPow goldenConj 2) (HAdd.hAdd goldenConj 1)
  -/
  rw [goldenConj, ← sub_eq_zero]
  /-
    ⊢ Eq (HSub.hSub (HPow.hPow (HDiv.hDiv (HSub.hSub 1 (Real.sqrt 5)) 2) 2) (HAdd. …
  -/
  ring_nf
  /-
    ⊢ Eq (HAdd.hAdd (-5 / 4) (HMul.hMul (HPow.hPow (Real.sqrt 5) 2) (1 / 4))) 0
  -/
                        /-
                          🎉 no goals
                        -/
  rw [Real.sq_sqrt] <;> norm_num
                        /-
                          🎉 no goals
                        -/


theorem gold_pos : 0 < φ :=
              /-
                ⊢ LT.lt 0 (HAdd.hAdd 1 (Real.sqrt 5))
              -/
                                /-
                                  🎉 no goals
                                -/
  mul_pos (by apply add_pos <;> norm_num) <| inv_pos.2 zero_lt_two
                                /-
                                  🎉 no goals
                                -/


theorem gold_ne_zero : φ ≠ 0 :=
  ne_of_gt gold_pos


theorem one_lt_gold : 1 < φ := by
  /-
    ⊢ LT.lt 1 goldenRatio
  -/
  refine lt_of_mul_lt_mul_left ?_ (le_of_lt gold_pos)
  /-
    ⊢ LT.lt (HMul.hMul goldenRatio 1) (HMul.hMul goldenRatio goldenRatio)
  -/
  simp [← sq, gold_pos, zero_lt_one]
  /-
    🎉 no goals
  -/


theorem gold_lt_two : φ < 2 := by calc
  (1 + sqrt 5) / 2 < (1 + 3) / 2 := by gcongr; rw [sqrt_lt'] <;> norm_num
  _ = 2 := by norm_num


theorem goldConj_neg : ψ < 0 := by
  /-
    ⊢ LT.lt goldenConj 0
  -/
  linarith [one_sub_goldConj, one_lt_gold]
  /-
    🎉 no goals
  -/


theorem goldConj_ne_zero : ψ ≠ 0 :=
  ne_of_lt goldConj_neg


theorem neg_one_lt_goldConj : -1 < ψ := by
  /-
    ⊢ LT.lt (-1) goldenConj
  -/
  rw [neg_lt, ← inv_gold]
  /-
    ⊢ LT.lt (Inv.inv goldenRatio) 1
  -/
  exact inv_lt_one_of_one_lt₀ one_lt_gold
  /-
    🎉 no goals
  -/


/-- The golden ratio is irrational. -/
theorem gold_irrational : Irrational φ := by
  /-
    ⊢ Irrational goldenRatio
  -/
  have := Nat.Prime.irrational_sqrt (show Nat.Prime 5 by norm_num)
  /-
    this : Irrational (↑5).sqrt
    ⊢ Irrational goldenRatio
  -/
  have := this.rat_add 1
  /-
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HAdd.hAdd (↑1) (↑5).sqrt)
    ⊢ Irrational goldenRatio
  -/
  convert this.rat_mul (show (0.5 : ℚ) ≠ 0 by norm_num)
  /-
    case h.e'_1
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HAdd.hAdd (↑1) (↑5).sqrt)
    ⊢ Eq goldenRatio (HMul.hMul (↑0.5) (HAdd.hAdd (↑1) (↑5).sqrt))
  -/
  norm_num
  /-
    case h.e'_1
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HAdd.hAdd (↑1) (↑5).sqrt)
    ⊢ Eq goldenRatio (HMul.hMul (1 / 2) (HAdd.hAdd 1 (Real.sqrt 5)))
  -/
  field_simp
  /-
    🎉 no goals
  -/


/-- The conjugate of the golden ratio is irrational. -/
theorem goldConj_irrational : Irrational ψ := by
  /-
    ⊢ Irrational goldenConj
  -/
  have := Nat.Prime.irrational_sqrt (show Nat.Prime 5 by norm_num)
  /-
    this : Irrational (↑5).sqrt
    ⊢ Irrational goldenConj
  -/
  have := this.rat_sub 1
  /-
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HSub.hSub (↑1) (↑5).sqrt)
    ⊢ Irrational goldenConj
  -/
  convert this.rat_mul (show (0.5 : ℚ) ≠ 0 by norm_num)
  /-
    case h.e'_1
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HSub.hSub (↑1) (↑5).sqrt)
    ⊢ Eq goldenConj (HMul.hMul (↑0.5) (HSub.hSub (↑1) (↑5).sqrt))
  -/
  norm_num
  /-
    case h.e'_1
    this✝ : Irrational (↑5).sqrt
    this : Irrational (HSub.hSub (↑1) (↑5).sqrt)
    ⊢ Eq goldenConj (HMul.hMul (1 / 2) (HSub.hSub 1 (Real.sqrt 5)))
  -/
  field_simp
  /-
    🎉 no goals
  -/


/-- The recurrence relation satisfied by the Fibonacci sequence. -/
def fibRec : LinearRecurrence α where
  order := 2
  coeffs := ![1, 1]


/-- The characteristic polynomial of `fibRec` is `X² - (X + 1)`. -/
theorem fibRec_charPoly_eq {β : Type*} [CommRing β] :
    fibRec.charPoly = X ^ 2 - (X + (1 : β[X])) := by
  /-
    β : Type u_2
    inst✝ : CommRing β
    ⊢ Eq fibRec.charPoly (HSub.hSub (HPow.hPow Polynomial.X 2) (HAdd.hAdd Polynomi …
  -/
  rw [fibRec, LinearRecurrence.charPoly]
  /-
    β : Type u_2
    inst✝ : CommRing β
    ⊢ Eq (HSub.hSub ((Polynomial.monomial { order := 2, coeffs := Matrix.vecCons 1 …
  -/
  simp [Finset.sum_fin_eq_sum_range, Finset.sum_range_succ', ← smul_X_eq_monomial]
  /-
    🎉 no goals
  -/


/-- As expected, the Fibonacci sequence is a solution of `fibRec`. -/
theorem fib_isSol_fibRec : fibRec.IsSolution (fun x => x.fib : ℕ → α) := by
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    ⊢ fibRec.IsSolution fun x => ↑(Nat.fib x)
  -/
  rw [fibRec]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    ⊢ { order := 2, coeffs := Matrix.vecCons 1 (Matrix.vecCons 1 Matrix.vecEmpty)  …
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    n : Nat
    ⊢ Eq ((fun x => ↑(Nat.fib x)) (HAdd.hAdd n { order := 2, coeffs := Matrix.vecC …
  -/
  simp only
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    n : Nat
    ⊢ Eq (↑(Nat.fib (HAdd.hAdd n 2))) (Finset.univ.sum fun x => HMul.hMul (Matrix. …
  -/
  rw [Nat.fib_add_two, add_comm]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    n : Nat
    ⊢ Eq (↑(HAdd.hAdd (Nat.fib (HAdd.hAdd n 1)) (Nat.fib n))) (Finset.univ.sum fun …
  -/
  simp [Finset.sum_fin_eq_sum_range, Finset.sum_range_succ']
  /-
    🎉 no goals
  -/


/-- The geometric sequence `fun n ↦ φ^n` is a solution of `fibRec`. -/
theorem geom_gold_isSol_fibRec : fibRec.IsSolution (φ ^ ·) := by
  /-
    ⊢ fibRec.IsSolution fun x => HPow.hPow goldenRatio x
  -/
  rw [fibRec.geom_sol_iff_root_charPoly, fibRec_charPoly_eq]
  /-
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X 2) (HAdd.hAdd Polynomial.X 1)).IsRoot gol …
  -/
  simp [sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- The geometric sequence `fun n ↦ ψ^n` is a solution of `fibRec`. -/
theorem geom_goldConj_isSol_fibRec : fibRec.IsSolution (ψ ^ ·) := by
  /-
    ⊢ fibRec.IsSolution fun x => HPow.hPow goldenConj x
  -/
  rw [fibRec.geom_sol_iff_root_charPoly, fibRec_charPoly_eq]
  /-
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X 2) (HAdd.hAdd Polynomial.X 1)).IsRoot gol …
  -/
  simp [sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- Binet's formula as a function equality. -/
theorem Real.coe_fib_eq' :
    (fun n => Nat.fib n : ℕ → ℝ) = fun n => (φ ^ n - ψ ^ n) / √5 := by
  /-
    ⊢ Eq (fun n => ↑(Nat.fib n)) fun n => HDiv.hDiv (HSub.hSub (HPow.hPow goldenRa …
  -/
  rw [fibRec.sol_eq_of_eq_init]
    /-
      ⊢ Set.EqOn (fun n => ↑(Nat.fib n)) (fun n => HDiv.hDiv (HSub.hSub (HPow.hPow g …
    -/
  · intro i hi
    /-
      i : Nat
      hi : Membership.mem (↑(Finset.range fibRec.order)) i
      ⊢ Eq ((fun n => ↑(Nat.fib n)) i) ((fun n => HDiv.hDiv (HSub.hSub (HPow.hPow go …
    -/
    norm_cast at hi
    /-
      i : Nat
      hi : Membership.mem (Finset.range fibRec.order) i
      ⊢ Eq ((fun n => ↑(Nat.fib n)) i) ((fun n => HDiv.hDiv (HSub.hSub (HPow.hPow go …
    -/
    fin_cases hi
      /-
        case «0»
        ⊢ Eq ((fun n => ↑(Nat.fib n)) 0) ((fun n => HDiv.hDiv (HSub.hSub (HPow.hPow go …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case «1»
        ⊢ Eq ((fun n => ↑(Nat.fib n)) 1) ((fun n => HDiv.hDiv (HSub.hSub (HPow.hPow go …
      -/
    · simp only [goldenRatio, goldenConj]
      /-
        case «1»
        ⊢ Eq (↑(Nat.fib 1)) (HDiv.hDiv (HSub.hSub (HPow.hPow (HDiv.hDiv (HAdd.hAdd 1 ( …
      -/
      ring_nf
      /-
        case «1»
        ⊢ Eq 1 (HMul.hMul (Real.sqrt 5) (Inv.inv (Real.sqrt 5)))
      -/
      rw [mul_inv_cancel₀]; norm_num
                            /-
                              🎉 no goals
                            -/
    /-
      case hu
      ⊢ fibRec.IsSolution fun n => ↑(Nat.fib n)
    -/
  · exact fib_isSol_fibRec
    /-
      🎉 no goals
    -/
  · -- Porting note: Rewrote this proof
    suffices LinearRecurrence.IsSolution fibRec
        ((fun n ↦ (√5)⁻¹ * φ ^ n) - (fun n ↦ (√5)⁻¹ * ψ ^ n)) by
      convert this
      rw [Pi.sub_apply]
      ring
    /-
      case hv
      ⊢ fibRec.IsSolution (HSub.hSub (fun n => HMul.hMul (Inv.inv (Real.sqrt 5)) (HP …
    -/
    apply (@fibRec ℝ _).solSpace.sub_mem
      /-
        case hv.a
        ⊢ Membership.mem fibRec.solSpace fun n => HMul.hMul (Inv.inv (Real.sqrt 5)) (H …
      -/
    · exact Submodule.smul_mem fibRec.solSpace (√5)⁻¹ geom_gold_isSol_fibRec
      /-
        🎉 no goals
      -/
      /-
        case hv.a
        ⊢ Membership.mem fibRec.solSpace fun n => HMul.hMul (Inv.inv (Real.sqrt 5)) (H …
      -/
    · exact Submodule.smul_mem fibRec.solSpace (√5)⁻¹ geom_goldConj_isSol_fibRec
      /-
        🎉 no goals
      -/


/-- Binet's formula as a dependent equality. -/
theorem Real.coe_fib_eq : ∀ n, (Nat.fib n : ℝ) = (φ ^ n - ψ ^ n) / √5 := by
  /-
    ⊢ ∀ (n : Nat), Eq (↑(Nat.fib n)) (HDiv.hDiv (HSub.hSub (HPow.hPow goldenRatio  …
  -/
  rw [← funext_iff, Real.coe_fib_eq']
  /-
    🎉 no goals
  -/


/-- Relationship between the Fibonacci Sequence, Golden Ratio and its conjugate's exponents --/
theorem fib_golden_conj_exp (n : ℕ) : Nat.fib (n + 1) - φ * Nat.fib n = ψ ^ n := by
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (↑(Nat.fib (HAdd.hAdd n 1))) (HMul.hMul goldenRatio ↑(Nat.fib  …
  -/
  repeat rw [coe_fib_eq]
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub (HPow.hPow goldenRatio (HAdd.hAdd n 1))  …
  -/
  rw [mul_div, div_sub_div_same, mul_sub, ← pow_succ']
  /-
    n : Nat
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HSub.hSub (HPow.hPow goldenRatio (HAdd.hAdd n 1))  …
  -/
  ring_nf
  /-
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (Real.sqrt 5) (Inv.inv (Real.sqrt 5))) (HPow.hPow ( …
  -/
  have nz : sqrt 5 ≠ 0 := by norm_num
  /-
    n : Nat
    nz : Ne (Real.sqrt 5) 0
    ⊢ Eq (HMul.hMul (HMul.hMul (Real.sqrt 5) (Inv.inv (Real.sqrt 5))) (HPow.hPow ( …
  -/
  rw [← (mul_inv_cancel₀ nz).symm, one_mul]
  /-
    🎉 no goals
  -/


/-- Relationship between the Fibonacci Sequence, Golden Ratio and its exponents --/
theorem fib_golden_exp' (n : ℕ) : φ * Nat.fib (n + 1) + Nat.fib n = φ ^ (n + 1) := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    calc
      _ = φ * (Nat.fib n) + φ ^ 2 * (Nat.fib (n + 1)) := by
        simp only [Nat.fib_add_one (Nat.succ_ne_zero n), Nat.succ_sub_succ_eq_sub, tsub_zero,
          Nat.cast_add, gold_sq]; ring
      _ = φ * ((Nat.fib n) + φ * (Nat.fib (n + 1))) := by ring
      _ = φ ^ (n + 2) := by rw [add_comm, ih]; ring

