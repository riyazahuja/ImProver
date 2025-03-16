theorem pow_arith_mean_le_arith_mean_pow (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) (n : ℕ) :
    (∑ i ∈ s, w i * z i) ^ n ≤ ∑ i ∈ s, w i * z i ^ n :=
  (convexOn_pow n).map_sum_le hw hw' hz


theorem pow_arith_mean_le_arith_mean_pow_of_even (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) {n : ℕ} (hn : Even n) :
    (∑ i ∈ s, w i * z i) ^ n ≤ ∑ i ∈ s, w i * z i ^ n :=
  hn.convexOn_pow.map_sum_le hw hw' fun _ _ => Set.mem_univ _


theorem zpow_arith_mean_le_arith_mean_zpow (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 < z i) (m : ℤ) :
    (∑ i ∈ s, w i * z i) ^ m ≤ ∑ i ∈ s, w i * z i ^ m :=
  (convexOn_zpow m).map_sum_le hw hw' hz


theorem rpow_arith_mean_le_arith_mean_rpow (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) {p : ℝ} (hp : 1 ≤ p) :
    (∑ i ∈ s, w i * z i) ^ p ≤ ∑ i ∈ s, w i * z i ^ p :=
  (convexOn_rpow hp).map_sum_le hw hw' hz


theorem arith_mean_le_rpow_mean (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i) (hw' : ∑ i ∈ s, w i = 1)
    (hz : ∀ i ∈ s, 0 ≤ z i) {p : ℝ} (hp : 1 ≤ p) :
    ∑ i ∈ s, w i * z i ≤ (∑ i ∈ s, w i * z i ^ p) ^ (1 / p) := by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (z i)) (HPow.hPow (s.sum fun i => HMul …
  -/
  have : 0 < p := by positivity
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    p : Real
    hp : LE.le 1 p
    this : LT.lt 0 p
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (z i)) (HPow.hPow (s.sum fun i => HMul …
  -/
  rw [← rpow_le_rpow_iff _ _ this, ← rpow_mul, one_div_mul_cancel (ne_of_gt this), rpow_one]
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      p : Real
      hp : LE.le 1 p
      this : LT.lt 0 p
      ⊢ LE.le (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) (s.sum fun i => H …
    -/
  · exact rpow_arith_mean_le_arith_mean_rpow s w z hw hw' hz hp
    /-
      🎉 no goals
    -/
  all_goals
    apply_rules [sum_nonneg, rpow_nonneg]
    intro i hi
    apply_rules [mul_nonneg, rpow_nonneg, hw i hi, hz i hi]


/-- Weighted generalized mean inequality, version sums over finite sets, with `ℝ≥0`-valued
functions and natural exponent. -/
theorem pow_arith_mean_le_arith_mean_pow (w z : ι → ℝ≥0) (hw' : ∑ i ∈ s, w i = 1) (n : ℕ) :
    (∑ i ∈ s, w i * z i) ^ n ≤ ∑ i ∈ s, w i * z i ^ n :=
  mod_cast
    Real.pow_arith_mean_le_arith_mean_pow s _ _ (fun i _ => (w i).coe_nonneg)
      (mod_cast hw') (fun i _ => (z i).coe_nonneg) n


/-- Weighted generalized mean inequality, version for sums over finite sets, with `ℝ≥0`-valued
functions and real exponents. -/
theorem rpow_arith_mean_le_arith_mean_rpow (w z : ι → ℝ≥0) (hw' : ∑ i ∈ s, w i = 1) {p : ℝ}
    (hp : 1 ≤ p) : (∑ i ∈ s, w i * z i) ^ p ≤ ∑ i ∈ s, w i * z i ^ p :=
  mod_cast
    Real.rpow_arith_mean_le_arith_mean_rpow s _ _ (fun i _ => (w i).coe_nonneg)
      (mod_cast hw') (fun i _ => (z i).coe_nonneg) hp


/-- Weighted generalized mean inequality, version for two elements of `ℝ≥0` and real exponents. -/
theorem rpow_arith_mean_le_arith_mean2_rpow (w₁ w₂ z₁ z₂ : ℝ≥0) (hw' : w₁ + w₂ = 1) {p : ℝ}
    (hp : 1 ≤ p) : (w₁ * z₁ + w₂ * z₂) ^ p ≤ w₁ * z₁ ^ p + w₂ * z₂ ^ p := by
  /-
    w₁ w₂ z₁ z₂ : NNReal
    hw' : Eq (HAdd.hAdd w₁ w₂) 1
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul w₁ z₁) (HMul.hMul w₂ z₂)) p) (HAdd.hA …
  -/
  have h := rpow_arith_mean_le_arith_mean_rpow univ ![w₁, w₂] ![z₁, z₂] ?_ hp
    /-
      case refine_2
      w₁ w₂ z₁ z₂ : NNReal
      hw' : Eq (HAdd.hAdd w₁ w₂) 1
      p : Real
      hp : LE.le 1 p
      h : LE.le (HPow.hPow (Finset.univ.sum fun i => HMul.hMul (Matrix.vecCons w₁ (M …
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul w₁ z₁) (HMul.hMul w₂ z₂)) p) (HAdd.hA …
    -/
  · simpa [Fin.sum_univ_succ] using h
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      w₁ w₂ z₁ z₂ : NNReal
      hw' : Eq (HAdd.hAdd w₁ w₂) 1
      p : Real
      hp : LE.le 1 p
      ⊢ Eq (Finset.univ.sum fun i => Matrix.vecCons w₁ (Matrix.vecCons w₂ Matrix.vec …
    -/
  · simp [hw', Fin.sum_univ_succ]
    /-
      🎉 no goals
    -/


/-- Unweighted mean inequality, version for two elements of `ℝ≥0` and real exponents. -/
theorem rpow_add_le_mul_rpow_add_rpow (z₁ z₂ : ℝ≥0) {p : ℝ} (hp : 1 ≤ p) :
    (z₁ + z₂) ^ p ≤ (2 : ℝ≥0) ^ (p - 1) * (z₁ ^ p + z₂ ^ p) := by
  /-
    z₁ z₂ : NNReal
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd z₁ z₂) p) (HMul.hMul (HPow.hPow 2 (HSub.hSub p 1 …
  -/
  rcases eq_or_lt_of_le hp with (rfl | h'p)
    /-
      case inl
      z₁ z₂ : NNReal
      hp : LE.le 1 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd z₁ z₂) 1) (HMul.hMul (HPow.hPow 2 (HSub.hSub 1 1 …
    -/
  · simp only [rpow_one, sub_self, rpow_zero, one_mul]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/
  convert rpow_arith_mean_le_arith_mean2_rpow (1 / 2) (1 / 2) (2 * z₁) (2 * z₂) (add_halves 1) hp
    using 1
  · simp only [one_div, inv_mul_cancel_left₀, Ne, mul_eq_zero, two_ne_zero, one_ne_zero,
      not_false_iff]
    /-
      case h.e'_4
      z₁ z₂ : NNReal
      p : Real
      hp : LE.le 1 p
      h'p : LT.lt 1 p
      ⊢ Eq (HMul.hMul (HPow.hPow 2 (HSub.hSub p 1)) (HAdd.hAdd (HPow.hPow z₁ p) (HPo …
    -/
  · have A : p - 1 ≠ 0 := ne_of_gt (sub_pos.2 h'p)
    /-
      case h.e'_4
      z₁ z₂ : NNReal
      p : Real
      hp : LE.le 1 p
      h'p : LT.lt 1 p
      A : Ne (HSub.hSub p 1) 0
      ⊢ Eq (HMul.hMul (HPow.hPow 2 (HSub.hSub p 1)) (HAdd.hAdd (HPow.hPow z₁ p) (HPo …
    -/
    simp only [mul_rpow, rpow_sub' A, div_eq_inv_mul, rpow_one, mul_one]
    /-
      case h.e'_4
      z₁ z₂ : NNReal
      p : Real
      hp : LE.le 1 p
      h'p : LT.lt 1 p
      A : Ne (HSub.hSub p 1) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv 2) (HPow.hPow 2 p)) (HAdd.hAdd (HPow.hPow  …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Weighted generalized mean inequality, version for sums over finite sets, with `ℝ≥0`-valued
functions and real exponents. -/
theorem arith_mean_le_rpow_mean (w z : ι → ℝ≥0) (hw' : ∑ i ∈ s, w i = 1) {p : ℝ} (hp : 1 ≤ p) :
    ∑ i ∈ s, w i * z i ≤ (∑ i ∈ s, w i * z i ^ p) ^ (1 / p) :=
  mod_cast
    Real.arith_mean_le_rpow_mean s _ _ (fun i _ => (w i).coe_nonneg) (mod_cast hw')
      (fun i _ => (z i).coe_nonneg) hp


private theorem add_rpow_le_one_of_add_le_one {p : ℝ} (a b : ℝ≥0) (hab : a + b ≤ 1) (hp1 : 1 ≤ p) :
    a ^ p + b ^ p ≤ 1 := by
  /-
    p : Real
    a b : NNReal
    hab : LE.le (HAdd.hAdd a b) 1
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) 1
  -/
  have h_le_one : ∀ x : ℝ≥0, x ≤ 1 → x ^ p ≤ x := fun x hx => rpow_le_self_of_le_one hx hp1
  /-
    p : Real
    a b : NNReal
    hab : LE.le (HAdd.hAdd a b) 1
    hp1 : LE.le 1 p
    h_le_one : ∀ (x : NNReal), LE.le x 1 → LE.le (HPow.hPow x p) x
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) 1
  -/
  have ha : a ≤ 1 := (self_le_add_right a b).trans hab
  /-
    p : Real
    a b : NNReal
    hab : LE.le (HAdd.hAdd a b) 1
    hp1 : LE.le 1 p
    h_le_one : ∀ (x : NNReal), LE.le x 1 → LE.le (HPow.hPow x p) x
    ha : LE.le a 1
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) 1
  -/
  have hb : b ≤ 1 := (self_le_add_left b a).trans hab
  /-
    p : Real
    a b : NNReal
    hab : LE.le (HAdd.hAdd a b) 1
    hp1 : LE.le 1 p
    h_le_one : ∀ (x : NNReal), LE.le x 1 → LE.le (HPow.hPow x p) x
    ha : LE.le a 1
    hb : LE.le b 1
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) 1
  -/
  exact (add_le_add (h_le_one a ha) (h_le_one b hb)).trans hab
  /-
    🎉 no goals
  -/


theorem add_rpow_le_rpow_add {p : ℝ} (a b : ℝ≥0) (hp1 : 1 ≤ p) : a ^ p + b ^ p ≤ (a + b) ^ p := by
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have hp_pos : 0 < p := by positivity
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  by_cases h_zero : a + b = 0
    /-
      case pos
      p : Real
      a b : NNReal
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      h_zero : Eq (HAdd.hAdd a b) 0
      ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
    -/
  · simp [add_eq_zero.mp h_zero, hp_pos.ne']
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have h_nonzero : ¬(a = 0 ∧ b = 0) := by rwa [add_eq_zero] at h_zero
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    h_nonzero : Not (And (Eq a 0) (Eq b 0))
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have h_add : a / (a + b) + b / (a + b) = 1 := by rw [div_add_div_same, div_self h_zero]
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    h_nonzero : Not (And (Eq a 0) (Eq b 0))
    h_add : Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a  …
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have h := add_rpow_le_one_of_add_le_one (a / (a + b)) (b / (a + b)) h_add.le hp1
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    h_nonzero : Not (And (Eq a 0) (Eq b 0))
    h_add : Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a  …
    h : LE.le (HAdd.hAdd (HPow.hPow (HDiv.hDiv a (HAdd.hAdd a b)) p) (HPow.hPow (H …
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  rw [div_rpow a (a + b), div_rpow b (a + b)] at h
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    h_nonzero : Not (And (Eq a 0) (Eq b 0))
    h_add : Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a  …
    h : LE.le (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (HPow.hPow (HAdd.hAdd a b) p)) …
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have hab_0 : (a + b) ^ p ≠ 0 := by simp [hp_pos, h_nonzero]
  /-
    case neg
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_zero : Not (Eq (HAdd.hAdd a b) 0)
    h_nonzero : Not (And (Eq a 0) (Eq b 0))
    h_add : Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a  …
    h : LE.le (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (HPow.hPow (HAdd.hAdd a b) p)) …
    hab_0 : Ne (HPow.hPow (HAdd.hAdd a b) p) 0
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have hab_0' : 0 < (a + b) ^ p := zero_lt_iff.mpr hab_0
  have h_mul : (a + b) ^ p * (a ^ p / (a + b) ^ p + b ^ p / (a + b) ^ p) ≤ (a + b) ^ p := by
    nth_rw 4 [← mul_one ((a + b) ^ p)]
    exact (mul_le_mul_left hab_0').mpr h
  rwa [div_eq_mul_inv, div_eq_mul_inv, mul_add, mul_comm (a ^ p), mul_comm (b ^ p), ← mul_assoc, ←
    mul_assoc, mul_inv_cancel₀ hab_0, one_mul, one_mul] at h_mul


theorem rpow_add_rpow_le_add {p : ℝ} (a b : ℝ≥0) (hp1 : 1 ≤ p) :
    (a ^ p + b ^ p) ^ (1 / p) ≤ a + b := by
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HDiv.hDiv 1 p) …
  -/
  rw [one_div]
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (Inv.inv p)) (H …
  -/
  rw [← @NNReal.le_rpow_inv_iff _ _ p⁻¹ (by simp [lt_of_lt_of_le zero_lt_one hp1])]
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  rw [inv_inv]
  /-
    p : Real
    a b : NNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  exact add_rpow_le_rpow_add _ _ hp1
  /-
    🎉 no goals
  -/


theorem rpow_add_rpow_le {p q : ℝ} (a b : ℝ≥0) (hp_pos : 0 < p) (hpq : p ≤ q) :
    (a ^ q + b ^ q) ^ (1 / q) ≤ (a ^ p + b ^ p) ^ (1 / p) := by
  have h_rpow : ∀ a : ℝ≥0, a ^ q = (a ^ p) ^ (q / p) := fun a => by
    rw [← NNReal.rpow_mul, div_eq_inv_mul, ← mul_assoc, mul_inv_cancel₀ hp_pos.ne.symm,
      one_mul]
  have h_rpow_add_rpow_le_add :
    ((a ^ p) ^ (q / p) + (b ^ p) ^ (q / p)) ^ (1 / (q / p)) ≤ a ^ p + b ^ p := by
    refine rpow_add_rpow_le_add (a ^ p) (b ^ p) ?_
    rwa [one_le_div hp_pos]
  rw [h_rpow a, h_rpow b, one_div p, NNReal.le_rpow_inv_iff hp_pos, ← NNReal.rpow_mul, mul_comm,
    mul_one_div]
  /-
    p q : Real
    a b : NNReal
    hp_pos : LT.lt 0 p
    hpq : LE.le p q
    h_rpow : ∀ (a : NNReal), Eq (HPow.hPow a q) (HPow.hPow (HPow.hPow a p) (HDiv.h …
    h_rpow_add_rpow_le_add : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow (HPow.hPow a p …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow (HPow.hPow a p) (HDiv.hDiv q p)) (HPo …
  -/
  rwa [one_div_div] at h_rpow_add_rpow_le_add
  /-
    🎉 no goals
  -/


theorem rpow_add_le_add_rpow {p : ℝ} (a b : ℝ≥0) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (a + b) ^ p ≤ a ^ p + b ^ p := by
  /-
    p : Real
    a b : NNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  rcases hp.eq_or_lt with (rfl | hp_pos)
    /-
      case inl
      a b : NNReal
      hp : LE.le 0 0
      hp1 : LE.le 0 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) 0) (HAdd.hAdd (HPow.hPow a 0) (HPow.hPow b  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    a b : NNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  have h := rpow_add_rpow_le a b hp_pos hp1
  /-
    case inr
    p : Real
    a b : NNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a 1) (HPow.hPow b 1)) (1 / 1)) (HPo …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  rw [one_div_one, one_div] at h
  /-
    case inr
    p : Real
    a b : NNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a 1) (HPow.hPow b 1)) 1) (HPow.hPow …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  repeat' rw [NNReal.rpow_one] at h
  /-
    case inr
    p : Real
    a b : NNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HAdd.hAdd a b) (HPow.hPow (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  exact (NNReal.le_rpow_inv_iff hp_pos).mp h
  /-
    🎉 no goals
  -/


/-- Weighted generalized mean inequality, version for sums over finite sets, with `ℝ≥0∞`-valued
functions and real exponents. -/
theorem rpow_arith_mean_le_arith_mean_rpow (w z : ι → ℝ≥0∞) (hw' : ∑ i ∈ s, w i = 1) {p : ℝ}
    (hp : 1 ≤ p) : (∑ i ∈ s, w i * z i) ^ p ≤ ∑ i ∈ s, w i * z i ^ p := by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → ENNReal
    hw' : Eq (s.sum fun i => w i) 1
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) (s.sum fun i => H …
  -/
  have hp_pos : 0 < p := by positivity
  /-
    ι : Type u
    s : Finset ι
    w z : ι → ENNReal
    hw' : Eq (s.sum fun i => w i) 1
    p : Real
    hp : LE.le 1 p
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) (s.sum fun i => H …
  -/
  have hp_nonneg : 0 ≤ p := by positivity
  /-
    ι : Type u
    s : Finset ι
    w z : ι → ENNReal
    hw' : Eq (s.sum fun i => w i) 1
    p : Real
    hp : LE.le 1 p
    hp_pos : LT.lt 0 p
    hp_nonneg : LE.le 0 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) (s.sum fun i => H …
  -/
  have hp_not_neg : ¬p < 0 := by simp [hp_nonneg]
  have h_top_iff_rpow_top : ∀ (i : ι), i ∈ s → (w i * z i = ⊤ ↔ w i * z i ^ p = ⊤) := by
    simp [ENNReal.mul_eq_top, hp_pos, hp_nonneg, hp_not_neg]
  /-
    ι : Type u
    s : Finset ι
    w z : ι → ENNReal
    hw' : Eq (s.sum fun i => w i) 1
    p : Real
    hp : LE.le 1 p
    hp_pos : LT.lt 0 p
    hp_nonneg : LE.le 0 p
    hp_not_neg : Not (LT.lt p 0)
    h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) (s.sum fun i => H …
  -/
  refine le_of_top_imp_top_of_toNNReal_le ?_ ?_
  · -- first, prove `(∑ i ∈ s, w i * z i) ^ p = ⊤ → ∑ i ∈ s, (w i * z i ^ p) = ⊤`
    /-
      case refine_1
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      ⊢ Eq (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) Top.top → Eq (s.sum  …
    -/
    rw [rpow_eq_top_iff, sum_eq_top, sum_eq_top]
    /-
      case refine_1
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      ⊢ Or (And (Eq (s.sum fun i => HMul.hMul (w i) (z i)) 0) (LT.lt p 0)) (And (Exi …
    -/
    intro h
    /-
      case refine_1
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      h : Or (And (Eq (s.sum fun i => HMul.hMul (w i) (z i)) 0) (LT.lt p 0)) (And (E …
      ⊢ Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (HPow.hPow (z  …
    -/
    simp only [and_false, hp_not_neg, false_or] at h
    /-
      case refine_1
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      h : And (Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (z a))  …
      ⊢ Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (HPow.hPow (z  …
    -/
    rcases h.left with ⟨a, H, ha⟩
    /-
      case refine_1.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      h : And (Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (z a))  …
      a : ι
      H : Membership.mem s a
      ha : Eq (HMul.hMul (w a) (z a)) Top.top
      ⊢ Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (HPow.hPow (z  …
    -/
    use a, H
    /-
      case right
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      h : And (Exists fun a => And (Membership.mem s a) (Eq (HMul.hMul (w a) (z a))  …
      a : ι
      H : Membership.mem s a
      ha : Eq (HMul.hMul (w a) (z a)) Top.top
      ⊢ Eq (HMul.hMul (w a) (HPow.hPow (z a) p)) Top.top
    -/
    rwa [← h_top_iff_rpow_top a H]
    /-
      🎉 no goals
    -/
  · -- second, suppose both `(∑ i ∈ s, w i * z i) ^ p ≠ ⊤` and `∑ i ∈ s, (w i * z i ^ p) ≠ ⊤`,
    -- and prove `((∑ i ∈ s, w i * z i) ^ p).toNNReal ≤ (∑ i ∈ s, (w i * z i ^ p)).toNNReal`,
    -- by using `NNReal.rpow_arith_mean_le_arith_mean_rpow`.
    /-
      case refine_2
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      ⊢ Ne (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) Top.top → Ne (s.sum  …
    -/
    intro h_top_rpow_sum _
    -- show hypotheses needed to put the `.toNNReal` inside the sums.
    have h_top : ∀ a : ι, a ∈ s → w a * z a ≠ ⊤ :=
      haveI h_top_sum : ∑ i ∈ s, w i * z i ≠ ⊤ := by
        intro h
        rw [h, top_rpow_of_pos hp_pos] at h_top_rpow_sum
        exact h_top_rpow_sum rfl
      fun a ha => (lt_top_of_sum_ne_top h_top_sum ha).ne
    have h_top_rpow : ∀ a : ι, a ∈ s → w a * z a ^ p ≠ ⊤ := by
      intro i hi
      specialize h_top i hi
      rwa [Ne, ← h_top_iff_rpow_top i hi]
    -- put the `.toNNReal` inside the sums.
    simp_rw [toNNReal_sum h_top_rpow, toNNReal_rpow, toNNReal_sum h_top, toNNReal_mul,
      toNNReal_rpow]
    -- use corresponding nnreal result
    refine
      NNReal.rpow_arith_mean_le_arith_mean_rpow s (fun i => (w i).toNNReal)
        (fun i => (z i).toNNReal) ?_ hp
    -- verify the hypothesis `∑ i ∈ s, (w i).toNNReal = 1`, using `∑ i ∈ s, w i = 1` .
    have h_sum_nnreal : ∑ i ∈ s, w i = ↑(∑ i ∈ s, (w i).toNNReal) := by
      rw [coe_finset_sum]
      refine sum_congr rfl fun i hi => (coe_toNNReal ?_).symm
      refine (lt_top_of_sum_ne_top ?_ hi).ne
      exact hw'.symm ▸ ENNReal.one_ne_top
    /-
      case refine_2
      ι : Type u
      s : Finset ι
      w z : ι → ENNReal
      hw' : Eq (s.sum fun i => w i) 1
      p : Real
      hp : LE.le 1 p
      hp_pos : LT.lt 0 p
      hp_nonneg : LE.le 0 p
      hp_not_neg : Not (LT.lt p 0)
      h_top_iff_rpow_top : ∀ (i : ι), Membership.mem s i → Iff (Eq (HMul.hMul (w i)  …
      h_top_rpow_sum : Ne (HPow.hPow (s.sum fun i => HMul.hMul (w i) (z i)) p) Top.top
      a✝ : Ne (s.sum fun i => HMul.hMul (w i) (HPow.hPow (z i) p)) Top.top
      h_top : ∀ (a : ι), Membership.mem s a → Ne (HMul.hMul (w a) (z a)) Top.top
      h_top_rpow : ∀ (a : ι), Membership.mem s a → Ne (HMul.hMul (w a) (HPow.hPow (z …
      h_sum_nnreal : Eq (s.sum fun i => w i) ↑(s.sum fun i => (w i).toNNReal)
      ⊢ Eq (s.sum fun i => (fun i => (w i).toNNReal) i) 1
    -/
    rwa [← coe_inj, ← h_sum_nnreal]
    /-
      🎉 no goals
    -/


/-- Weighted generalized mean inequality, version for two elements of `ℝ≥0∞` and real
exponents. -/
theorem rpow_arith_mean_le_arith_mean2_rpow (w₁ w₂ z₁ z₂ : ℝ≥0∞) (hw' : w₁ + w₂ = 1) {p : ℝ}
    (hp : 1 ≤ p) : (w₁ * z₁ + w₂ * z₂) ^ p ≤ w₁ * z₁ ^ p + w₂ * z₂ ^ p := by
  /-
    w₁ w₂ z₁ z₂ : ENNReal
    hw' : Eq (HAdd.hAdd w₁ w₂) 1
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul w₁ z₁) (HMul.hMul w₂ z₂)) p) (HAdd.hA …
  -/
  have h := rpow_arith_mean_le_arith_mean_rpow univ ![w₁, w₂] ![z₁, z₂] ?_ hp
    /-
      case refine_2
      w₁ w₂ z₁ z₂ : ENNReal
      hw' : Eq (HAdd.hAdd w₁ w₂) 1
      p : Real
      hp : LE.le 1 p
      h : LE.le (HPow.hPow (Finset.univ.sum fun i => HMul.hMul (Matrix.vecCons w₁ (M …
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul w₁ z₁) (HMul.hMul w₂ z₂)) p) (HAdd.hA …
    -/
  · simpa [Fin.sum_univ_succ] using h
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      w₁ w₂ z₁ z₂ : ENNReal
      hw' : Eq (HAdd.hAdd w₁ w₂) 1
      p : Real
      hp : LE.le 1 p
      ⊢ Eq (Finset.univ.sum fun i => Matrix.vecCons w₁ (Matrix.vecCons w₂ Matrix.vec …
    -/
  · simp [hw', Fin.sum_univ_succ]
    /-
      🎉 no goals
    -/


/-- Unweighted mean inequality, version for two elements of `ℝ≥0∞` and real exponents. -/
theorem rpow_add_le_mul_rpow_add_rpow (z₁ z₂ : ℝ≥0∞) {p : ℝ} (hp : 1 ≤ p) :
    (z₁ + z₂) ^ p ≤ (2 : ℝ≥0∞) ^ (p - 1) * (z₁ ^ p + z₂ ^ p) := by
  convert rpow_arith_mean_le_arith_mean2_rpow (1 / 2) (1 / 2) (2 * z₁) (2 * z₂)
      (ENNReal.add_halves 1) hp using 1
    /-
      case h.e'_3
      z₁ z₂ : ENNReal
      p : Real
      hp : LE.le 1 p
      ⊢ Eq (HPow.hPow (HAdd.hAdd z₁ z₂) p) (HPow.hPow (HAdd.hAdd (HMul.hMul (1 / 2)  …
    -/
  · simp [← mul_assoc, ENNReal.inv_mul_cancel two_ne_zero two_ne_top]
    /-
      🎉 no goals
    -/
  · simp only [mul_rpow_of_nonneg _ _ (zero_le_one.trans hp), rpow_sub _ _ two_ne_zero two_ne_top,
      ENNReal.div_eq_inv_mul, rpow_one, mul_one]
    /-
      case h.e'_4
      z₁ z₂ : ENNReal
      p : Real
      hp : LE.le 1 p
      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv 2) (HPow.hPow 2 p)) (HAdd.hAdd (HPow.hPow  …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem add_rpow_le_rpow_add {p : ℝ} (a b : ℝ≥0∞) (hp1 : 1 ≤ p) : a ^ p + b ^ p ≤ (a + b) ^ p := by
  /-
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  have hp_pos : 0 < p := by positivity
  /-
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  by_cases h_top : a + b = ⊤
    /-
      case pos
      p : Real
      a b : ENNReal
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      h_top : Eq (HAdd.hAdd a b) Top.top
      ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
    -/
  · rw [← @ENNReal.rpow_eq_top_iff_of_pos (a + b) p hp_pos] at h_top
    /-
      case pos
      p : Real
      a b : ENNReal
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      h_top : Eq (HPow.hPow (HAdd.hAdd a b) p) Top.top
      ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
    -/
    rw [h_top]
    /-
      case pos
      p : Real
      a b : ENNReal
      hp1 : LE.le 1 p
      hp_pos : LT.lt 0 p
      h_top : Eq (HPow.hPow (HAdd.hAdd a b) p) Top.top
      ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) Top.top
    -/
    exact le_top
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_top : Not (Eq (HAdd.hAdd a b) Top.top)
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  obtain ⟨ha_top, hb_top⟩ := add_ne_top.mp h_top
  /-
    case neg.intro
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    h_top : Not (Eq (HAdd.hAdd a b) Top.top)
    ha_top : Ne a Top.top
    hb_top : Ne b Top.top
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  lift a to ℝ≥0 using ha_top
  /-
    case neg.intro.intro
    p : Real
    b : ENNReal
    hp1 : LE.le 1 p
    hp_pos : LT.lt 0 p
    hb_top : Ne b Top.top
    a : NNReal
    h_top : Not (Eq (HAdd.hAdd (↑a) b) Top.top)
    ⊢ LE.le (HAdd.hAdd (HPow.hPow (↑a) p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd ( …
  -/
  lift b to ℝ≥0 using hb_top
  simpa [ENNReal.coe_rpow_of_nonneg _ hp_pos.le] using
    ENNReal.coe_le_coe.2 (NNReal.add_rpow_le_rpow_add a b hp1)


theorem rpow_add_rpow_le_add {p : ℝ} (a b : ℝ≥0∞) (hp1 : 1 ≤ p) :
    (a ^ p + b ^ p) ^ (1 / p) ≤ a + b := by
  /-
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HDiv.hDiv 1 p) …
  -/
  rw [one_div, ← @ENNReal.le_rpow_inv_iff _ _ p⁻¹ (by simp [lt_of_lt_of_le zero_lt_one hp1])]
  /-
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  rw [inv_inv]
  /-
    p : Real
    a b : ENNReal
    hp1 : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p)) (HPow.hPow (HAdd.hAdd a b) …
  -/
  exact add_rpow_le_rpow_add _ _ hp1
  /-
    🎉 no goals
  -/


theorem rpow_add_rpow_le {p q : ℝ} (a b : ℝ≥0∞) (hp_pos : 0 < p) (hpq : p ≤ q) :
    (a ^ q + b ^ q) ^ (1 / q) ≤ (a ^ p + b ^ p) ^ (1 / p) := by
  have h_rpow : ∀ a : ℝ≥0∞, a ^ q = (a ^ p) ^ (q / p) := fun a => by
    rw [← ENNReal.rpow_mul, mul_div_cancel₀ _ hp_pos.ne']
  have h_rpow_add_rpow_le_add :
    ((a ^ p) ^ (q / p) + (b ^ p) ^ (q / p)) ^ (1 / (q / p)) ≤ a ^ p + b ^ p := by
    refine rpow_add_rpow_le_add (a ^ p) (b ^ p) ?_
    rwa [one_le_div hp_pos]
  rw [h_rpow a, h_rpow b, one_div p, ENNReal.le_rpow_inv_iff hp_pos, ← ENNReal.rpow_mul, mul_comm,
    mul_one_div]
  /-
    p q : Real
    a b : ENNReal
    hp_pos : LT.lt 0 p
    hpq : LE.le p q
    h_rpow : ∀ (a : ENNReal), Eq (HPow.hPow a q) (HPow.hPow (HPow.hPow a p) (HDiv. …
    h_rpow_add_rpow_le_add : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow (HPow.hPow a p …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow (HPow.hPow a p) (HDiv.hDiv q p)) (HPo …
  -/
  rwa [one_div_div] at h_rpow_add_rpow_le_add
  /-
    🎉 no goals
  -/


theorem rpow_add_le_add_rpow {p : ℝ} (a b : ℝ≥0∞) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (a + b) ^ p ≤ a ^ p + b ^ p := by
  /-
    p : Real
    a b : ENNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  rcases hp.eq_or_lt with (rfl | hp_pos)
    /-
      case inl
      a b : ENNReal
      hp : LE.le 0 0
      hp1 : LE.le 0 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) 0) (HAdd.hAdd (HPow.hPow a 0) (HPow.hPow b  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    a b : ENNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  have h := rpow_add_rpow_le a b hp_pos hp1
  /-
    case inr
    p : Real
    a b : ENNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a 1) (HPow.hPow b 1)) (1 / 1)) (HPo …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  rw [one_div_one, one_div] at h
  /-
    case inr
    p : Real
    a b : ENNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow a 1) (HPow.hPow b 1)) 1) (HPow.hPow …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  repeat' rw [ENNReal.rpow_one] at h
  /-
    case inr
    p : Real
    a b : ENNReal
    hp : LE.le 0 p
    hp1 : LE.le p 1
    hp_pos : LT.lt 0 p
    h : LE.le (HAdd.hAdd a b) (HPow.hPow (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b p …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd a b) p) (HAdd.hAdd (HPow.hPow a p) (HPow.hPow b  …
  -/
  exact (ENNReal.le_rpow_inv_iff hp_pos).mp h
  /-
    🎉 no goals
  -/


