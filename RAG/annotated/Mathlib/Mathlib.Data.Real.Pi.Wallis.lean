/-- The product of the first `k` terms in Wallis' formula for `π`. -/
noncomputable def W (k : ℕ) : ℝ :=
  ∏ i ∈ range k, (2 * i + 2) / (2 * i + 1) * ((2 * i + 2) / (2 * i + 3))


theorem W_succ (k : ℕ) :
    W (k + 1) = W k * ((2 * k + 2) / (2 * k + 1) * ((2 * k + 2) / (2 * k + 3))) :=
  prod_range_succ _ _


theorem W_pos (k : ℕ) : 0 < W k := by
  /-
    k : Nat
    ⊢ LT.lt 0 (Real.Wallis.W k)
  -/
  induction' k with k hk
    /-
      case zero
      ⊢ LT.lt 0 (Real.Wallis.W 0)
    -/
  · unfold W; simp
              /-
                🎉 no goals
              -/
    /-
      case succ
      k : Nat
      hk : LT.lt 0 (Real.Wallis.W k)
      ⊢ LT.lt 0 (Real.Wallis.W (HAdd.hAdd k 1))
    -/
  · rw [W_succ]
    /-
      case succ
      k : Nat
      hk : LT.lt 0 (Real.Wallis.W k)
      ⊢ LT.lt 0 (HMul.hMul (Real.Wallis.W k) (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
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
    refine mul_pos hk (mul_pos (div_pos ?_ ?_) (div_pos ?_ ?_)) <;> positivity
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem W_eq_factorial_ratio (n : ℕ) :
    W n = 2 ^ (4 * n) * n ! ^ 4 / ((2 * n)! ^ 2 * (2 * n + 1)) := by
  /-
    n : Nat
    ⊢ Eq (Real.Wallis.W n) (HDiv.hDiv (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n)) (HP …
  -/
  induction' n with n IH
  · simp only [W, prod_range_zero, Nat.factorial_zero, mul_zero, pow_zero,
      algebraMap.coe_one, one_pow, mul_one, algebraMap.coe_zero, zero_add, div_self, Ne,
      one_ne_zero, not_false_iff]
    /-
      case zero
      ⊢ Eq 1 (HDiv.hDiv (HMul.hMul 1 (HPow.hPow (↑1) 4)) (HMul.hMul (HPow.hPow (↑1)  …
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      IH : Eq (Real.Wallis.W n) (HDiv.hDiv (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n))  …
      ⊢ Eq (Real.Wallis.W (HAdd.hAdd n 1)) (HDiv.hDiv (HMul.hMul (HPow.hPow 2 (HMul. …
    -/
  · unfold W at IH ⊢
    /-
      case succ
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun i => HMul.hMul (HDiv.hDiv (HAdd. …
    -/
    rw [prod_range_succ, IH, _root_.div_mul_div_comm, _root_.div_mul_div_comm]
    /-
      case succ
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n)) (HPow.hPow …
    -/
    refine (div_eq_div_iff ?_ ?_).mpr ?_
    /-
      case succ.refine_1
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Ne (HMul.hMul (HMul.hMul (HPow.hPow (↑(HMul.hMul 2 n).factorial) 2) (HAdd.hA …
    -/
    any_goals exact ne_of_gt (by positivity)
    /-
      case succ.refine_3
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n)) (HPow.hPow …
    -/
    simp_rw [Nat.mul_succ, Nat.factorial_succ, pow_succ]
    /-
      case succ.refine_3
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n)) (HMul.hMul …
    -/
    push_cast
    /-
      case succ.refine_3
      n : Nat
      IH : Eq ((Finset.range n).prod fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul. …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow 2 (HMul.hMul 4 n)) (HMul.hMul …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


theorem W_eq_integral_sin_pow_div_integral_sin_pow (k : ℕ) : (π / 2)⁻¹ * W k =
    (∫ x : ℝ in (0)..π, sin x ^ (2 * k + 1)) / ∫ x : ℝ in (0)..π, sin x ^ (2 * k) := by
  /-
    k : Nat
    ⊢ Eq (HMul.hMul (Inv.inv (HDiv.hDiv Real.pi 2)) (Real.Wallis.W k)) (HDiv.hDiv  …
  -/
  rw [integral_sin_pow_even, integral_sin_pow_odd, mul_div_mul_comm, ← prod_div_distrib, inv_div]
  /-
    k : Nat
    ⊢ Eq (HMul.hMul (HDiv.hDiv 2 Real.pi) (Real.Wallis.W k)) (HMul.hMul (HDiv.hDiv …
  -/
  simp_rw [div_div_div_comm, div_div_eq_mul_div, mul_div_assoc]
  /-
    k : Nat
    ⊢ Eq (HMul.hMul (HDiv.hDiv 2 Real.pi) (Real.Wallis.W k)) (HMul.hMul (HDiv.hDiv …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem W_le (k : ℕ) : W k ≤ π / 2 := by
  /-
    k : Nat
    ⊢ LE.le (Real.Wallis.W k) (HDiv.hDiv Real.pi 2)
  -/
  rw [← div_le_one pi_div_two_pos, div_eq_inv_mul]
  /-
    k : Nat
    ⊢ LE.le (HMul.hMul (Inv.inv (HDiv.hDiv Real.pi 2)) (Real.Wallis.W k)) 1
  -/
  rw [W_eq_integral_sin_pow_div_integral_sin_pow, div_le_one (integral_sin_pow_pos _)]
  /-
    k : Nat
    ⊢ LE.le (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hM …
  -/
  apply integral_sin_pow_succ_le
  /-
    🎉 no goals
  -/


theorem le_W (k : ℕ) : ((2 : ℝ) * k + 1) / (2 * k + 2) * (π / 2) ≤ W k := by
  /-
    k : Nat
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HAdd.hAdd (HMul. …
  -/
  rw [← le_div_iff₀ pi_div_two_pos, div_eq_inv_mul (W k) _]
  /-
    k : Nat
    ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HAdd.hAdd (HMul.hMul 2 ↑k)  …
  -/
  rw [W_eq_integral_sin_pow_div_integral_sin_pow, le_div_iff₀ (integral_sin_pow_pos _)]
  /-
    k : Nat
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HAdd.hAdd (HMul. …
  -/
  convert integral_sin_pow_succ_le (2 * k + 1)
  /-
    case h.e'_3
    k : Nat
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HAdd.hAdd (HMul.hMu …
  -/
  rw [integral_sin_pow (2 * k)]
  /-
    case h.e'_3
    k : Nat
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (HAdd.hAdd (HMul.hMu …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem tendsto_W_nhds_pi_div_two : Tendsto W atTop (𝓝 <| π / 2) := by
  /-
    ⊢ Filter.Tendsto Real.Wallis.W Filter.atTop (nhds (HDiv.hDiv Real.pi 2))
  -/
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le ?_ tendsto_const_nhds le_W W_le
  /-
    ⊢ Filter.Tendsto (fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑i) 1) …
  -/
  have : 𝓝 (π / 2) = 𝓝 ((1 - 0) * (π / 2)) := by rw [sub_zero, one_mul]
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    ⊢ Filter.Tendsto (fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑i) 1) …
  -/
  rw [this]
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    ⊢ Filter.Tendsto (fun i => HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑i) 1) …
  -/
  refine Tendsto.mul ?_ tendsto_const_nhds
  have h : ∀ n : ℕ, ((2 : ℝ) * n + 1) / (2 * n + 2) = 1 - 1 / (2 * n + 2) := by
    intro n
    rw [sub_div' _ _ _ (ne_of_gt (add_pos_of_nonneg_of_pos (mul_nonneg
      (two_pos : 0 < (2 : ℝ)).le (Nat.cast_nonneg _)) two_pos)), one_mul]
    congr 1; ring
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    h : ∀ (n : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HAdd.hAdd (HMul …
    ⊢ Filter.Tendsto (fun i => HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑i) 1) (HAdd.hAdd …
  -/
  simp_rw [h]
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    h : ∀ (n : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HAdd.hAdd (HMul …
    ⊢ Filter.Tendsto (fun i => HSub.hSub 1 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 ↑i …
  -/
  refine (tendsto_const_nhds.div_atTop ?_).const_sub _
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    h : ∀ (n : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HAdd.hAdd (HMul …
    ⊢ Filter.Tendsto (fun i => HAdd.hAdd (HMul.hMul 2 ↑i) 2) Filter.atTop Filter.a …
  -/
  refine Tendsto.atTop_add ?_ tendsto_const_nhds
  /-
    this : Eq (nhds (HDiv.hDiv Real.pi 2)) (nhds (HMul.hMul (HSub.hSub 1 0) (HDiv. …
    h : ∀ (n : Nat), Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HAdd.hAdd (HMul …
    ⊢ Filter.Tendsto (fun i => HMul.hMul 2 ↑i) Filter.atTop Filter.atTop
  -/
  exact tendsto_natCast_atTop_atTop.const_mul_atTop two_pos
  /-
    🎉 no goals
  -/


/-- Wallis' product formula for `π / 2`. -/
theorem Real.tendsto_prod_pi_div_two :
    Tendsto (fun k => ∏ i ∈ range k, ((2 : ℝ) * i + 2) / (2 * i + 1) * ((2 * i + 2) / (2 * i + 3)))
      atTop (𝓝 (π / 2)) :=
  Real.Wallis.tendsto_W_nhds_pi_div_two

