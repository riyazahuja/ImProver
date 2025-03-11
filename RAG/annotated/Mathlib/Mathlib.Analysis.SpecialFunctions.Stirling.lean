/-- Define `stirlingSeq n` as $\frac{n!}{\sqrt{2n}(\frac{n}{e})^n}$.
Stirling's formula states that this sequence has limit $\sqrt(π)$.
-/
noncomputable def stirlingSeq (n : ℕ) : ℝ :=
  n ! / (√(2 * n : ℝ) * (n / exp 1) ^ n)


@[simp]
theorem stirlingSeq_zero : stirlingSeq 0 = 0 := by
  /-
    ⊢ Eq (Stirling.stirlingSeq 0) 0
  -/
  rw [stirlingSeq, cast_zero, mul_zero, Real.sqrt_zero, zero_mul, div_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem stirlingSeq_one : stirlingSeq 1 = exp 1 / √2 := by
  /-
    ⊢ Eq (Stirling.stirlingSeq 1) (HDiv.hDiv (Real.exp 1) (Real.sqrt 2))
  -/
  rw [stirlingSeq, pow_one, factorial_one, cast_one, mul_one, mul_one_div, one_div_div]
  /-
    🎉 no goals
  -/


theorem log_stirlingSeq_formula (n : ℕ) :
    log (stirlingSeq n) = Real.log n ! - 1 / 2 * Real.log (2 * n) - n * log (n / exp 1) := by
  /-
    n : Nat
    ⊢ Eq (Real.log (Stirling.stirlingSeq n)) (HSub.hSub (HSub.hSub (Real.log ↑n.fa …
  -/
  cases n
    /-
      case zero
      ⊢ Eq (Real.log (Stirling.stirlingSeq 0)) (HSub.hSub (HSub.hSub (Real.log ↑(Nat …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (Real.log (Stirling.stirlingSeq (HAdd.hAdd n✝ 1))) (HSub.hSub (HSub.hSub  …
    -/
  · rw [stirlingSeq, log_div, log_mul, sqrt_eq_rpow, log_rpow, Real.log_pow, tsub_tsub]
          /-
            case succ.hx
            n✝ : Nat
            ⊢ LT.lt 0 (HMul.hMul 2 ↑(HAdd.hAdd n✝ 1))
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
      <;> positivity
          /-
            🎉 no goals
          -/


/-- The sequence `log (stirlingSeq (m + 1)) - log (stirlingSeq (m + 2))` has the series expansion
   `∑ 1 / (2 * (k + 1) + 1) * (1 / 2 * (m + 1) + 1)^(2 * (k + 1))`
-/
theorem log_stirlingSeq_diff_hasSum (m : ℕ) :
    HasSum (fun k : ℕ => (1 : ℝ) / (2 * ↑(k + 1) + 1) * ((1 / (2 * ↑(m + 1) + 1)) ^ 2) ^ ↑(k + 1))
      (log (stirlingSeq (m + 1)) - log (stirlingSeq (m + 2))) := by
  /-
    m : Nat
    ⊢ HasSum (fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAdd.hAdd  …
  -/
  let f (k : ℕ) := (1 : ℝ) / (2 * k + 1) * ((1 / (2 * ↑(m + 1) + 1)) ^ 2) ^ k
  /-
    m : Nat
    f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
    ⊢ HasSum (fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAdd.hAdd  …
  -/
  change HasSum (fun k => f (k + 1)) _
  /-
    m : Nat
    f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
    ⊢ HasSum (fun k => f (HAdd.hAdd k 1)) (HSub.hSub (Real.log (Stirling.stirlingS …
  -/
  rw [hasSum_nat_add_iff]
  /-
    m : Nat
    f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
    ⊢ HasSum f (HAdd.hAdd (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd m  …
  -/
  convert (hasSum_log_one_add_inv m.cast_add_one_pos).mul_left ((↑(m + 1) : ℝ) + 1 / 2) using 1
    /-
      case h.e'_5
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      ⊢ Eq f fun i => HMul.hMul (HAdd.hAdd (↑(HAdd.hAdd m 1)) (1 / 2)) (HMul.hMul (H …
    -/
  · ext k
    /-
      case h.e'_5.h
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      k : Nat
      ⊢ Eq (f k) (HMul.hMul (HAdd.hAdd (↑(HAdd.hAdd m 1)) (1 / 2)) (HMul.hMul (HMul. …
    -/
    dsimp only [f]
    /-
      case h.e'_5.h
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      k : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k) 1)) (HPow.hPow (HPow. …
    -/
    rw [← pow_mul, pow_add]
    /-
      case h.e'_5.h
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      k : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k) 1)) (HPow.hPow (HDiv. …
    -/
    push_cast
    /-
      case h.e'_5.h
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      k : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k) 1)) (HPow.hPow (HDiv. …
    -/
    field_simp
    /-
      case h.e'_5.h
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      k : Nat
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HMul.hMul (HPow.hPow (HAdd.hAd …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      ⊢ Eq (HAdd.hAdd (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd m 1))) ( …
    -/
  · have h : ∀ x ≠ (0 : ℝ), 1 + x⁻¹ = (x + 1) / x := fun x hx ↦ by field_simp [hx]
    simp (disch := positivity) only [log_stirlingSeq_formula, log_div, log_mul, log_exp,
      factorial_succ, cast_mul, cast_succ, cast_zero, range_one, sum_singleton, h]
    /-
      case h.e'_6
      m : Nat
      f : Nat → Real := fun k => HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑k)  …
      h : ∀ (x : Real), Ne x 0 → Eq (HAdd.hAdd 1 (Inv.inv x)) (HDiv.hDiv (HAdd.hAdd  …
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (Real.log (HAdd.hA …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The sequence `log ∘ stirlingSeq ∘ succ` is monotone decreasing -/
theorem log_stirlingSeq'_antitone : Antitone (Real.log ∘ stirlingSeq ∘ succ) :=
  antitone_nat_of_succ_le fun n =>
                                                                        /-
                                                                          n m : Nat
                                                                          ⊢ LE.le 0 (HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAdd.hAdd m 1)) 1) …
                                                                        -/
    sub_nonneg.mp <| (log_stirlingSeq_diff_hasSum n).nonneg fun m => by positivity
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- We have a bound for successive elements in the sequence `log (stirlingSeq k)`.
-/
theorem log_stirlingSeq_diff_le_geo_sum (n : ℕ) :
    log (stirlingSeq (n + 1)) - log (stirlingSeq (n + 2)) ≤
      ((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2 / (1 - ((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2) := by
  /-
    n : Nat
    ⊢ LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1))) (Real.log …
  -/
  have h_nonneg : (0 : ℝ) ≤ ((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2 := sq_nonneg _
  have g : HasSum (fun k : ℕ => (((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2) ^ ↑(k + 1))
      (((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2 / (1 - ((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2)) := by
    have := (hasSum_geometric_of_lt_one h_nonneg ?_).mul_left (((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2)
    · simp_rw [← _root_.pow_succ'] at this
      exact this
    rw [one_div, inv_pow]
    exact inv_lt_one_of_one_lt₀ (one_lt_pow₀ (lt_add_of_pos_left _ <| by positivity) two_ne_zero)
  have hab (k : ℕ) : (1 : ℝ) / (2 * ↑(k + 1) + 1) * ((1 / (2 * ↑(n + 1) + 1)) ^ 2) ^ ↑(k + 1) ≤
      (((1 : ℝ) / (2 * ↑(n + 1) + 1)) ^ 2) ^ ↑(k + 1) := by
    refine mul_le_of_le_one_left (pow_nonneg h_nonneg ↑(k + 1)) ?_
    rw [one_div]
    exact inv_le_one_of_one_le₀ (le_add_of_nonneg_left <| by positivity)
  /-
    n : Nat
    h_nonneg : LE.le 0 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAdd.hAdd …
    g : HasSum (fun k => HPow.hPow (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 …
    hab : ∀ (k : Nat), LE.le (HMul.hMul (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAd …
    ⊢ LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1))) (Real.log …
  -/
  exact hasSum_le hab (log_stirlingSeq_diff_hasSum n) g
  /-
    🎉 no goals
  -/


/-- We have the bound `log (stirlingSeq n) - log (stirlingSeq (n+1))` ≤ 1/(4 n^2)
-/
theorem log_stirlingSeq_sub_log_stirlingSeq_succ (n : ℕ) :
    log (stirlingSeq (n + 1)) - log (stirlingSeq (n + 2)) ≤ 1 / (4 * (↑(n + 1) : ℝ) ^ 2) := by
  /-
    n : Nat
    ⊢ LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1))) (Real.log …
  -/
  have h₁ : (0 : ℝ) < 4 * ((n : ℝ) + 1) ^ 2 := by positivity
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    ⊢ LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1))) (Real.log …
  -/
  have h₃ : (0 : ℝ) < (2 * ((n : ℝ) + 1) + 1) ^ 2 := by positivity
  have h₂ : (0 : ℝ) < 1 - (1 / (2 * ((n : ℝ) + 1) + 1)) ^ 2 := by
    rw [← mul_lt_mul_right h₃]
    have H : 0 < (2 * ((n : ℝ) + 1) + 1) ^ 2 - 1 := by nlinarith [@cast_nonneg ℝ _ n]
    convert H using 1 <;> field_simp [h₃.ne']
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1))) (Real.log …
  -/
  refine (log_stirlingSeq_diff_le_geo_sum n).trans ?_
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑(HAdd.hAdd …
  -/
  push_cast
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd  …
  -/
  rw [div_le_div_iff₀ h₂ h₁]
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HMul.hMul (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd  …
  -/
  field_simp [h₃.ne']
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HDiv.hDiv (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HPow.hPow ( …
  -/
  rw [div_le_div_iff_of_pos_right h₃]
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HSub.hSub (HPow.hPow ( …
  -/
  ring_nf
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd 4 (HMul.hMul (↑n) 8)) (HMul.hMul (HPow.hPow (↑n) …
  -/
  norm_cast
  /-
    n : Nat
    h₁ : LT.lt 0 (HMul.hMul 4 (HPow.hPow (HAdd.hAdd (↑n) 1) 2))
    h₃ : LT.lt 0 (HPow.hPow (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) 1) 2)
    h₂ : LT.lt 0 (HSub.hSub 1 (HPow.hPow (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 (HAd …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd 4 (HMul.hMul n 8)) (HMul.hMul (HPow.hPow n 2) 4) …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- For any `n`, we have `log_stirlingSeq 1 - log_stirlingSeq n ≤ 1/4 * ∑' 1/k^2`  -/
theorem log_stirlingSeq_bounded_aux :
    ∃ c : ℝ, ∀ n : ℕ, log (stirlingSeq 1) - log (stirlingSeq (n + 1)) ≤ c := by
  /-
    ⊢ Exists fun c => ∀ (n : Nat), LE.le (HSub.hSub (Real.log (Stirling.stirlingSe …
  -/
  let d : ℝ := ∑' k : ℕ, (1 : ℝ) / (↑(k + 1) : ℝ) ^ 2
  /-
    d : Real := tsum fun k => HDiv.hDiv 1 (HPow.hPow (↑(HAdd.hAdd k 1)) 2)
    ⊢ Exists fun c => ∀ (n : Nat), LE.le (HSub.hSub (Real.log (Stirling.stirlingSe …
  -/
  use 1 / 4 * d
  /-
    case h
    d : Real := tsum fun k => HDiv.hDiv 1 (HPow.hPow (↑(HAdd.hAdd k 1)) 2)
    ⊢ ∀ (n : Nat), LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq 1)) (Real.log  …
  -/
  let log_stirlingSeq' : ℕ → ℝ := fun k => log (stirlingSeq (k + 1))
  /-
    case h
    d : Real := tsum fun k => HDiv.hDiv 1 (HPow.hPow (↑(HAdd.hAdd k 1)) 2)
    log_stirlingSeq' : Nat → Real := fun k => Real.log (Stirling.stirlingSeq (HAdd …
    ⊢ ∀ (n : Nat), LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq 1)) (Real.log  …
  -/
  intro n
  have h₁ k : log_stirlingSeq' k - log_stirlingSeq' (k + 1) ≤ 1 / 4 * (1 / (↑(k + 1) : ℝ) ^ 2) := by
    convert log_stirlingSeq_sub_log_stirlingSeq_succ k using 1; field_simp
  have h₂ : (∑ k ∈ range n, 1 / (↑(k + 1) : ℝ) ^ 2) ≤ d := by
    have := (summable_nat_add_iff 1).mpr <| Real.summable_one_div_nat_pow.mpr one_lt_two
    exact sum_le_tsum (range n) (fun k _ => by positivity) this
  calc
    log (stirlingSeq 1) - log (stirlingSeq (n + 1)) = log_stirlingSeq' 0 - log_stirlingSeq' n :=
      rfl
    _ = ∑ k ∈ range n, (log_stirlingSeq' k - log_stirlingSeq' (k + 1)) := by
      rw [← sum_range_sub' log_stirlingSeq' n]
    _ ≤ ∑ k ∈ range n, 1 / 4 * (1 / ↑((k + 1)) ^ 2) := sum_le_sum fun k _ => h₁ k
    _ = 1 / 4 * ∑ k ∈ range n, 1 / ↑((k + 1)) ^ 2 := by rw [mul_sum]
    _ ≤ 1 / 4 * d := by gcongr


/-- The sequence `log_stirlingSeq` is bounded below for `n ≥ 1`. -/
theorem log_stirlingSeq_bounded_by_constant : ∃ c, ∀ n : ℕ, c ≤ log (stirlingSeq (n + 1)) := by
  /-
    ⊢ Exists fun c => ∀ (n : Nat), LE.le c (Real.log (Stirling.stirlingSeq (HAdd.h …
  -/
  obtain ⟨d, h⟩ := log_stirlingSeq_bounded_aux
  /-
    case intro
    d : Real
    h : ∀ (n : Nat), LE.le (HSub.hSub (Real.log (Stirling.stirlingSeq 1)) (Real.lo …
    ⊢ Exists fun c => ∀ (n : Nat), LE.le c (Real.log (Stirling.stirlingSeq (HAdd.h …
  -/
  exact ⟨log (stirlingSeq 1) - d, fun n => sub_le_comm.mp (h n)⟩
  /-
    🎉 no goals
  -/


/-- The sequence `stirlingSeq` is positive for `n > 0`  -/
                                                                 /-
                                                                   n : Nat
                                                                   ⊢ LT.lt 0 (Stirling.stirlingSeq (HAdd.hAdd n 1))
                                                                 -/
theorem stirlingSeq'_pos (n : ℕ) : 0 < stirlingSeq (n + 1) := by unfold stirlingSeq; positivity
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- The sequence `stirlingSeq` has a positive lower bound.
-/
theorem stirlingSeq'_bounded_by_pos_constant : ∃ a, 0 < a ∧ ∀ n : ℕ, a ≤ stirlingSeq (n + 1) := by
  /-
    ⊢ Exists fun a => And (LT.lt 0 a) (∀ (n : Nat), LE.le a (Stirling.stirlingSeq  …
  -/
  cases' log_stirlingSeq_bounded_by_constant with c h
  /-
    case intro
    c : Real
    h : ∀ (n : Nat), LE.le c (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1)))
    ⊢ Exists fun a => And (LT.lt 0 a) (∀ (n : Nat), LE.le a (Stirling.stirlingSeq  …
  -/
  refine ⟨exp c, exp_pos _, fun n => ?_⟩
  /-
    case intro
    c : Real
    h : ∀ (n : Nat), LE.le c (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1)))
    n : Nat
    ⊢ LE.le (Real.exp c) (Stirling.stirlingSeq (HAdd.hAdd n 1))
  -/
  rw [← le_log_iff_exp_le (stirlingSeq'_pos n)]
  /-
    case intro
    c : Real
    h : ∀ (n : Nat), LE.le c (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1)))
    n : Nat
    ⊢ LE.le c (Real.log (Stirling.stirlingSeq (HAdd.hAdd n 1)))
  -/
  exact h n
  /-
    🎉 no goals
  -/


/-- The sequence `stirlingSeq ∘ succ` is monotone decreasing -/
theorem stirlingSeq'_antitone : Antitone (stirlingSeq ∘ succ) := fun n m h =>
  (log_le_log_iff (stirlingSeq'_pos m) (stirlingSeq'_pos n)).mp (log_stirlingSeq'_antitone h)


/-- The limit `a` of the sequence `stirlingSeq` satisfies `0 < a` -/
theorem stirlingSeq_has_pos_limit_a : ∃ a : ℝ, 0 < a ∧ Tendsto stirlingSeq atTop (𝓝 a) := by
  /-
    ⊢ Exists fun a => And (LT.lt 0 a) (Filter.Tendsto Stirling.stirlingSeq Filter. …
  -/
  obtain ⟨x, x_pos, hx⟩ := stirlingSeq'_bounded_by_pos_constant
  /-
    case intro.intro
    x : Real
    x_pos : LT.lt 0 x
    hx : ∀ (n : Nat), LE.le x (Stirling.stirlingSeq (HAdd.hAdd n 1))
    ⊢ Exists fun a => And (LT.lt 0 a) (Filter.Tendsto Stirling.stirlingSeq Filter. …
  -/
  have hx' : x ∈ lowerBounds (Set.range (stirlingSeq ∘ succ)) := by simpa [lowerBounds] using hx
  /-
    case intro.intro
    x : Real
    x_pos : LT.lt 0 x
    hx : ∀ (n : Nat), LE.le x (Stirling.stirlingSeq (HAdd.hAdd n 1))
    hx' : Membership.mem (lowerBounds (Set.range (Function.comp Stirling.stirlingS …
    ⊢ Exists fun a => And (LT.lt 0 a) (Filter.Tendsto Stirling.stirlingSeq Filter. …
  -/
  refine ⟨_, lt_of_lt_of_le x_pos (le_csInf (Set.range_nonempty _) hx'), ?_⟩
  /-
    case intro.intro
    x : Real
    x_pos : LT.lt 0 x
    hx : ∀ (n : Nat), LE.le x (Stirling.stirlingSeq (HAdd.hAdd n 1))
    hx' : Membership.mem (lowerBounds (Set.range (Function.comp Stirling.stirlingS …
    ⊢ Filter.Tendsto Stirling.stirlingSeq Filter.atTop (nhds (InfSet.sInf (Set.ran …
  -/
  rw [← Filter.tendsto_add_atTop_iff_nat 1]
  /-
    case intro.intro
    x : Real
    x_pos : LT.lt 0 x
    hx : ∀ (n : Nat), LE.le x (Stirling.stirlingSeq (HAdd.hAdd n 1))
    hx' : Membership.mem (lowerBounds (Set.range (Function.comp Stirling.stirlingS …
    ⊢ Filter.Tendsto (fun n => Stirling.stirlingSeq (HAdd.hAdd n 1)) Filter.atTop  …
  -/
  exact tendsto_atTop_ciInf stirlingSeq'_antitone ⟨x, hx'⟩
  /-
    🎉 no goals
  -/


/-- The sequence `n / (2 * n + 1)` tends to `1/2` -/
theorem tendsto_self_div_two_mul_self_add_one :
    Tendsto (fun n : ℕ => (n : ℝ) / (2 * n + 1)) atTop (𝓝 (1 / 2)) := by
  conv =>
    congr
    · skip
    · skip
    rw [one_div, ← add_zero (2 : ℝ)]
  refine (((tendsto_const_div_atTop_nhds_zero_nat 1).const_add (2 : ℝ)).inv₀
    ((add_zero (2 : ℝ)).symm ▸ two_ne_zero)).congr' (eventually_atTop.mpr ⟨1, fun n hn => ?_⟩)
  /-
    n : Nat
    hn : GE.ge n 1
    ⊢ Eq (Inv.inv (HAdd.hAdd 2 (HDiv.hDiv 1 ↑n))) ((fun n => HDiv.hDiv (↑n) (HAdd. …
  -/
  rw [add_div' (1 : ℝ) 2 n (cast_ne_zero.mpr (one_le_iff_ne_zero.mp hn)), inv_div]
  /-
    🎉 no goals
  -/


/-- For any `n ≠ 0`, we have the identity
`(stirlingSeq n)^4 / (stirlingSeq (2*n))^2 * (n / (2 * n + 1)) = W n`, where `W n` is the
`n`-th partial product of Wallis' formula for `π / 2`. -/
theorem stirlingSeq_pow_four_div_stirlingSeq_pow_two_eq (n : ℕ) (hn : n ≠ 0) :
    stirlingSeq n ^ 4 / stirlingSeq (2 * n) ^ 2 * (n / (2 * n + 1)) = Wallis.W n := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (Stirling.stirlingSeq n) 4) (HPow.hPow ( …
  -/
  have : 4 = 2 * 2 := by rfl
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (Stirling.stirlingSeq n) 4) (HPow.hPow ( …
  -/
  rw [stirlingSeq, this, pow_mul, stirlingSeq, Wallis.W_eq_factorial_ratio]
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (HPow.hPow (HDiv.hDiv (↑n.factorial) (HM …
  -/
  simp_rw [div_pow, mul_pow]
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HPow.hPow (HPow.hPow (↑n.factorial) 2)  …
  -/
  rw [sq_sqrt, sq_sqrt]
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HPow.hPow (HPow.hPow (↑n.factorial) 2)  …
  -/
  any_goals positivity
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HPow.hPow (HPow.hPow (↑n.factorial) 2)  …
  -/
  field_simp [← exp_nsmul]
  /-
    n : Nat
    hn : Ne n 0
    this : Eq 4 (HMul.hMul 2 2)
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HPow.hPow (↑n.fac …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- Suppose the sequence `stirlingSeq` (defined above) has the limit `a ≠ 0`.
Then the Wallis sequence `W n` has limit `a^2 / 2`.
-/
theorem second_wallis_limit (a : ℝ) (hane : a ≠ 0) (ha : Tendsto stirlingSeq atTop (𝓝 a)) :
    Tendsto Wallis.W atTop (𝓝 (a ^ 2 / 2)) := by
  refine Tendsto.congr' (eventually_atTop.mpr ⟨1, fun n hn =>
    stirlingSeq_pow_four_div_stirlingSeq_pow_two_eq n (one_le_iff_ne_zero.mp hn)⟩) ?_
  have h : a ^ 2 / 2 = a ^ 4 / a ^ 2 * (1 / 2) := by
    rw [mul_one_div, ← mul_one_div (a ^ 4) (a ^ 2), one_div, ← pow_sub_of_lt a]
    norm_num
  /-
    a : Real
    hane : Ne a 0
    ha : Filter.Tendsto Stirling.stirlingSeq Filter.atTop (nhds a)
    h : Eq (HDiv.hDiv (HPow.hPow a 2) 2) (HMul.hMul (HDiv.hDiv (HPow.hPow a 4) (HP …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HDiv.hDiv (HPow.hPow (Stirling.stirlingS …
  -/
  rw [h]
  exact ((ha.pow 4).div ((ha.comp (tendsto_id.const_mul_atTop' two_pos)).pow 2)
    (pow_ne_zero 2 hane)).mul tendsto_self_div_two_mul_self_add_one


/-- **Stirling's Formula** -/
theorem tendsto_stirlingSeq_sqrt_pi : Tendsto stirlingSeq atTop (𝓝 (√π)) := by
  /-
    ⊢ Filter.Tendsto Stirling.stirlingSeq Filter.atTop (nhds Real.pi.sqrt)
  -/
  obtain ⟨a, hapos, halimit⟩ := stirlingSeq_has_pos_limit_a
  have hπ : π / 2 = a ^ 2 / 2 :=
    tendsto_nhds_unique Wallis.tendsto_W_nhds_pi_div_two (second_wallis_limit a hapos.ne' halimit)
  /-
    case intro.intro
    a : Real
    hapos : LT.lt 0 a
    halimit : Filter.Tendsto Stirling.stirlingSeq Filter.atTop (nhds a)
    hπ : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv (HPow.hPow a 2) 2)
    ⊢ Filter.Tendsto Stirling.stirlingSeq Filter.atTop (nhds Real.pi.sqrt)
  -/
  rwa [(div_left_inj' (two_ne_zero' ℝ)).mp hπ, sqrt_sq hapos.le]
  /-
    🎉 no goals
  -/


/-- **Stirling's Formula**, formulated in terms of `Asymptotics.IsEquivalent`. -/
lemma factorial_isEquivalent_stirling :
    (fun n ↦ n ! : ℕ → ℝ) ~[atTop] fun n ↦ Real.sqrt (2 * n * π) * (n / exp 1) ^ n := by
  /-
    ⊢ Asymptotics.IsEquivalent Filter.atTop (fun n => ↑n.factorial) fun n => HMul. …
  -/
  refine Asymptotics.isEquivalent_of_tendsto_one ?_ ?_
    /-
      case refine_1
      ⊢ Filter.Eventually (fun x => Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑x) Real.p …
    -/
  · filter_upwards [eventually_ne_atTop 0] with n hn h
    /-
      case h
      n : Nat
      hn : Ne n 0
      h : Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑n) Real.pi).sqrt (HPow.hPow (HDiv.h …
      ⊢ Eq (↑n.factorial) 0
    -/
    exact absurd h (by positivity)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ Filter.Tendsto (HDiv.hDiv (fun n => ↑n.factorial) fun n => HMul.hMul (HMul.h …
    -/
  · have : sqrt π ≠ 0 := by positivity
    /-
      case refine_2
      this : Ne Real.pi.sqrt 0
      ⊢ Filter.Tendsto (HDiv.hDiv (fun n => ↑n.factorial) fun n => HMul.hMul (HMul.h …
    -/
    nth_rewrite 2 [← div_self this]
    /-
      case refine_2
      this : Ne Real.pi.sqrt 0
      ⊢ Filter.Tendsto (HDiv.hDiv (fun n => ↑n.factorial) fun n => HMul.hMul (HMul.h …
    -/
    convert tendsto_stirlingSeq_sqrt_pi.div tendsto_const_nhds this using 1
    /-
      case h.e'_3
      this : Ne Real.pi.sqrt 0
      ⊢ Eq (HDiv.hDiv (fun n => ↑n.factorial) fun n => HMul.hMul (HMul.hMul (HMul.hM …
    -/
    ext n
    /-
      case h.e'_3.h
      this : Ne Real.pi.sqrt 0
      n : Nat
      ⊢ Eq (HDiv.hDiv (fun n => ↑n.factorial) (fun n => HMul.hMul (HMul.hMul (HMul.h …
    -/
    field_simp [stirlingSeq, mul_right_comm]
    /-
      🎉 no goals
    -/


