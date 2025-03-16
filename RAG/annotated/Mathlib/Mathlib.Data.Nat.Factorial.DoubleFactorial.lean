/-- `Nat.doubleFactorial n` is the double factorial of `n`. -/
@[simp]
def doubleFactorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | k + 2 => (k + 2) * doubleFactorial k

-- This notation is `\!!` not two !'s

@[inherit_doc] scoped notation:10000 n "‼" => Nat.doubleFactorial n


lemma doubleFactorial_pos : ∀ n, 0 < n‼
  | 0 | 1 => zero_lt_one
  | _n + 2 => mul_pos (succ_pos _) (doubleFactorial_pos _)


theorem doubleFactorial_add_two (n : ℕ) : (n + 2)‼ = (n + 2) * n‼ :=
  rfl


                                                                              /-
                                                                                n : Nat
                                                                                ⊢ Eq (HAdd.hAdd n 1).doubleFactorial (HMul.hMul (HAdd.hAdd n 1) (HSub.hSub n 1 …
                                                                              -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
theorem doubleFactorial_add_one (n : ℕ) : (n + 1)‼ = (n + 1) * (n - 1)‼ := by cases n <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem factorial_eq_mul_doubleFactorial : ∀ n : ℕ, (n + 1)! = (n + 1)‼ * n‼
  | 0 => rfl
  | k + 1 => by
    rw [doubleFactorial_add_two, factorial, factorial_eq_mul_doubleFactorial _, mul_comm _ k‼,
      mul_assoc]


lemma doubleFactorial_le_factorial : ∀ n, n‼ ≤ n !
  | 0 => le_rfl
  | n + 1 => by
    /-
      n : Nat
      ⊢ LE.le (HAdd.hAdd n 1).doubleFactorial (HAdd.hAdd n 1).factorial
    -/
    rw [factorial_eq_mul_doubleFactorial]; exact Nat.le_mul_of_pos_right _ n.doubleFactorial_pos
                                           /-
                                             🎉 no goals
                                           -/


theorem doubleFactorial_two_mul : ∀ n : ℕ, (2 * n)‼ = 2 ^ n * n !
  | 0 => rfl
  | n + 1 => by
    rw [mul_add, mul_one, doubleFactorial_add_two, factorial, pow_succ, doubleFactorial_two_mul _,
      succ_eq_add_one]
    /-
      n : Nat
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul 2 n) 2) (HMul.hMul (HPow.hPow 2 n) n.fac …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem doubleFactorial_eq_prod_even : ∀ n : ℕ, (2 * n)‼ = ∏ i ∈ Finset.range n, 2 * (i + 1)
  | 0 => rfl
  | n + 1 => by
    rw [Finset.prod_range_succ, ← doubleFactorial_eq_prod_even _, mul_comm (2 * n)‼,
      (by ring : 2 * (n + 1) = 2 * n + 2)]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 n) 2).doubleFactorial (HMul.hMul (HAdd.hAdd (HMul …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem doubleFactorial_eq_prod_odd :
    ∀ n : ℕ, (2 * n + 1)‼ = ∏ i ∈ Finset.range n, (2 * (i + 1) + 1)
  | 0 => rfl
  | n + 1 => by
    rw [Finset.prod_range_succ, ← doubleFactorial_eq_prod_odd _, mul_comm (2 * n + 1)‼,
      (by ring : 2 * (n + 1) + 1 = 2 * n + 1 + 2)]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 n) 1) 2).doubleFactorial (HMul.hMul (H …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Extension for `Nat.doubleFactorial`. -/
@[positivity Nat.doubleFactorial _]
def evalDoubleFactorial : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(Nat.doubleFactorial $n) =>
    assumeInstancesCommute
    return .positive q(Nat.doubleFactorial_pos $n)
  | _, _ => throwError "not Nat.doubleFactorial"


