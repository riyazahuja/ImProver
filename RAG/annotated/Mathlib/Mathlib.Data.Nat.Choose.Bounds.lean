theorem choose_le_pow_div (r n : ℕ) : (n.choose r : α) ≤ (n ^ r : α) / r ! := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    r n : Nat
    ⊢ LE.le (↑(n.choose r)) (HDiv.hDiv (HPow.hPow (↑n) r) ↑r.factorial)
  -/
  rw [le_div_iff₀']
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (HMul.hMul ↑r.factorial ↑(n.choose r)) (HPow.hPow (↑n) r)
    -/
  · norm_cast
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (HMul.hMul r.factorial (n.choose r)) (HPow.hPow n r)
    -/
    rw [← Nat.descFactorial_eq_factorial_mul_choose]
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (n.descFactorial r) (HPow.hPow n r)
    -/
    exact n.descFactorial_le_pow r
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    r n : Nat
    ⊢ LT.lt 0 ↑r.factorial
  -/
  exact mod_cast r.factorial_pos
  /-
    🎉 no goals
  -/


lemma choose_le_descFactorial (n k : ℕ) : n.choose k ≤ n.descFactorial k := by
  /-
    n k : Nat
    ⊢ LE.le (n.choose k) (n.descFactorial k)
  -/
  rw [choose_eq_descFactorial_div_factorial]
  /-
    n k : Nat
    ⊢ LE.le (HDiv.hDiv (n.descFactorial k) k.factorial) (n.descFactorial k)
  -/
  exact Nat.div_le_self _ _
  /-
    🎉 no goals
  -/


/-- This lemma was changed on 2024/08/29, the old statement is available
in `Nat.choose_le_pow_div`. -/
lemma choose_le_pow (n k : ℕ) : n.choose k ≤ n ^ k :=
  (choose_le_descFactorial n k).trans (descFactorial_le_pow n k)

-- horrific casting is due to ℕ-subtraction

theorem pow_le_choose (r n : ℕ) : ((n + 1 - r : ℕ) ^ r : α) / r ! ≤ n.choose r := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    r n : Nat
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑(HSub.hSub (HAdd.hAdd n 1) r)) r) ↑r.factorial …
  -/
  rw [div_le_iff₀']
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (HPow.hPow (↑(HSub.hSub (HAdd.hAdd n 1) r)) r) (HMul.hMul ↑r.factorial …
    -/
  · norm_cast
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) r) r) (HMul.hMul r.factorial (n. …
    -/
    rw [← Nat.descFactorial_eq_factorial_mul_choose]
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      r n : Nat
      ⊢ LE.le (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) r) r) (n.descFactorial r)
    -/
    exact n.pow_sub_le_descFactorial r
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    r n : Nat
    ⊢ LT.lt 0 ↑r.factorial
  -/
  exact mod_cast r.factorial_pos
  /-
    🎉 no goals
  -/


