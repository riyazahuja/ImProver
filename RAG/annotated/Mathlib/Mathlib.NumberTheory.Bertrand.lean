/-- A reified version of the `Bertrand.main_inequality` below.
This is not best possible: it actually holds for 464 ≤ x.
-/
theorem real_main_inequality {x : ℝ} (x_large : (512 : ℝ) ≤ x) :
    x * (2 * x) ^ √(2 * x) * 4 ^ (2 * x / 3) ≤ 4 ^ x := by
  /-
    x : Real
    x_large : LE.le 512 x
    ⊢ LE.le (HMul.hMul (HMul.hMul x (HPow.hPow (HMul.hMul 2 x) (HMul.hMul 2 x).sqr …
  -/
  let f : ℝ → ℝ := fun x => log x + √(2 * x) * log (2 * x) - log 4 / 3 * x
  have hf' : ∀ x, 0 < x → 0 < x * (2 * x) ^ √(2 * x) / 4 ^ (x / 3) := fun x h =>
    div_pos (mul_pos h (rpow_pos_of_pos (mul_pos two_pos h) _)) (rpow_pos_of_pos four_pos _)
  have hf : ∀ x, 0 < x → f x = log (x * (2 * x) ^ √(2 * x) / 4 ^ (x / 3)) := by
    intro x h5
    have h6 := mul_pos (zero_lt_two' ℝ) h5
    have h7 := rpow_pos_of_pos h6 (√(2 * x))
    rw [log_div (mul_pos h5 h7).ne' (rpow_pos_of_pos four_pos _).ne', log_mul h5.ne' h7.ne',
      log_rpow h6, log_rpow zero_lt_four, ← mul_div_right_comm, ← mul_div, mul_comm x]
  /-
    x : Real
    x_large : LE.le 512 x
    f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
    hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
    hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
    ⊢ LE.le (HMul.hMul (HMul.hMul x (HPow.hPow (HMul.hMul 2 x) (HMul.hMul 2 x).sqr …
  -/
  have h5 : 0 < x := lt_of_lt_of_le (by norm_num1) x_large
  rw [← div_le_one (rpow_pos_of_pos four_pos x), ← div_div_eq_mul_div, ← rpow_sub four_pos, ←
    mul_div 2 x, mul_div_left_comm, ← mul_one_sub, (by norm_num1 : (1 : ℝ) - 2 / 3 = 1 / 3),
    mul_one_div, ← log_nonpos_iff (hf' x h5), ← hf x h5]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): the proof was rewritten, because it was too slow
  have h : ConcaveOn ℝ (Set.Ioi 0.5) f := by
    apply ConcaveOn.sub
    · apply ConcaveOn.add
      · exact strictConcaveOn_log_Ioi.concaveOn.subset
          (Set.Ioi_subset_Ioi (by norm_num)) (convex_Ioi 0.5)
      convert ((strictConcaveOn_sqrt_mul_log_Ioi.concaveOn.comp_linearMap
        ((2 : ℝ) • LinearMap.id))) using 1
      ext x
      simp only [Set.mem_Ioi, Set.mem_preimage, LinearMap.smul_apply,
        LinearMap.id_coe, id_eq, smul_eq_mul]
      rw [← mul_lt_mul_left (two_pos)]
      norm_num1
      rfl
    apply ConvexOn.smul
    · refine div_nonneg (log_nonneg (by norm_num1)) (by norm_num1)
    · exact convexOn_id (convex_Ioi (0.5 : ℝ))
  suffices ∃ x1 x2, 0.5 < x1 ∧ x1 < x2 ∧ x2 ≤ x ∧ 0 ≤ f x1 ∧ f x2 ≤ 0 by
    obtain ⟨x1, x2, h1, h2, h0, h3, h4⟩ := this
    exact (h.right_le_of_le_left'' h1 ((h1.trans h2).trans_le h0) h2 h0 (h4.trans h3)).trans h4
  /-
    x : Real
    x_large : LE.le 512 x
    f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
    hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
    hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
    h5 : LT.lt 0 x
    h : ConcaveOn Real (Set.Ioi 0.5) f
    ⊢ Exists fun x1 => Exists fun x2 => And (LT.lt 0.5 x1) (And (LT.lt x1 x2) (And …
  -/
  refine ⟨18, 512, by norm_num1, by norm_num1, x_large, ?_, ?_⟩
    /-
      case refine_1
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      ⊢ LE.le 0 (f 18)
    -/
  · have : √(2 * 18 : ℝ) = 6 := (sqrt_eq_iff_mul_self_eq_of_pos (by norm_num1)).mpr (by norm_num1)
    /-
      case refine_1
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 18).sqrt 6
      ⊢ LE.le 0 (f 18)
    -/
    rw [hf _ (by norm_num1), log_nonneg_iff (by positivity), this, one_le_div (by norm_num1)]
    /-
      case refine_1
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 18).sqrt 6
      ⊢ LE.le (HPow.hPow 4 (18 / 3)) (HMul.hMul 18 (HPow.hPow (HMul.hMul 2 18) 6))
    -/
    norm_num1
    /-
      🎉 no goals
    -/
  · have : √(2 * 512) = 32 :=
      (sqrt_eq_iff_mul_self_eq_of_pos (by norm_num1)).mpr (by norm_num1)
    rw [hf _ (by norm_num1), log_nonpos_iff (hf' _ (by norm_num1)), this,
        div_le_one (by positivity)]
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HMul.hMul 512 (HPow.hPow (HMul.hMul 2 512) 32)) (HPow.hPow 4 (512 / 3))
    -/
    conv in 512 => equals 2 ^ 9 => norm_num1
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HMul.hMul (HPow.hPow 2 9) (HPow.hPow (HMul.hMul 2 512) 32)) (HPow.hPo …
    -/
    conv in 2 * 512 => equals 2 ^ 10 => norm_num1
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HMul.hMul (HPow.hPow 2 9) (HPow.hPow (HPow.hPow 2 10) 32)) (HPow.hPow …
    -/
    conv in 32 => rw [← Nat.cast_ofNat]
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HMul.hMul (HPow.hPow 2 9) (HPow.hPow (HPow.hPow 2 10) ↑32)) (HPow.hPo …
    -/
    rw [rpow_natCast, ← pow_mul, ← pow_add]
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HPow.hPow 2 (HAdd.hAdd 9 (HMul.hMul 10 32))) (HPow.hPow 4 (512 / 3))
    -/
    conv in 4 => equals 2 ^ (2 : ℝ) => rw [rpow_two]; norm_num1
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HPow.hPow 2 (HAdd.hAdd 9 (HMul.hMul 10 32))) (HPow.hPow (HPow.hPow 2  …
    -/
    rw [← rpow_mul, ← rpow_natCast]
    /-
      case refine_2
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le (HPow.hPow 2 ↑(HAdd.hAdd 9 (HMul.hMul 10 32))) (HPow.hPow 2 (HMul.hMul …
    -/
    on_goal 1 => apply rpow_le_rpow_of_exponent_le
    /-
      case refine_2.hx
      x : Real
      x_large : LE.le 512 x
      f : Real → Real := fun x => HSub.hSub (HAdd.hAdd (Real.log x) (HMul.hMul (HMul …
      hf' : ∀ (x : Real), LT.lt 0 x → LT.lt 0 (HDiv.hDiv (HMul.hMul x (HPow.hPow (HM …
      hf : ∀ (x : Real), LT.lt 0 x → Eq (f x) (Real.log (HDiv.hDiv (HMul.hMul x (HPo …
      h5 : LT.lt 0 x
      h : ConcaveOn Real (Set.Ioi 0.5) f
      this : Eq (HMul.hMul 2 512).sqrt 32
      ⊢ LE.le 1 2
    -/
    all_goals norm_num1
    /-
      🎉 no goals
    -/


/-- The inequality which contradicts Bertrand's postulate, for large enough `n`.
-/
theorem bertrand_main_inequality {n : ℕ} (n_large : 512 ≤ n) :
    n * (2 * n) ^ sqrt (2 * n) * 4 ^ (2 * n / 3) ≤ 4 ^ n := by
  /-
    n : Nat
    n_large : LE.le 512 n
    ⊢ LE.le (HMul.hMul (HMul.hMul n (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).sqr …
  -/
  rw [← @cast_le ℝ]
  /-
    n : Nat
    n_large : LE.le 512 n
    ⊢ LE.le ↑(HMul.hMul (HMul.hMul n (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).sq …
  -/
  simp only [cast_add, cast_one, cast_mul, cast_pow, ← Real.rpow_natCast]
  /-
    n : Nat
    n_large : LE.le 512 n
    ⊢ LE.le (HMul.hMul (HMul.hMul (↑n) (HPow.hPow (HMul.hMul ↑2 ↑n) ↑(HMul.hMul 2  …
  -/
  refine _root_.trans ?_ (Bertrand.real_main_inequality (by exact_mod_cast n_large))
  /-
    n : Nat
    n_large : LE.le 512 n
    ⊢ LE.le (HMul.hMul (HMul.hMul (↑n) (HPow.hPow (HMul.hMul ↑2 ↑n) ↑(HMul.hMul 2  …
  -/
  gcongr
    /-
      case h₁.h.hx
      n : Nat
      n_large : LE.le 512 n
      ⊢ LE.le 1 (HMul.hMul ↑2 ↑n)
    -/
  · have n2_pos : 0 < 2 * n := by positivity
    /-
      case h₁.h.hx
      n : Nat
      n_large : LE.le 512 n
      n2_pos : LT.lt 0 (HMul.hMul 2 n)
      ⊢ LE.le 1 (HMul.hMul ↑2 ↑n)
    -/
    exact mod_cast n2_pos
    /-
      🎉 no goals
    -/
    /-
      case h₁.h.hyz
      n : Nat
      n_large : LE.le 512 n
      ⊢ LE.le (↑(HMul.hMul 2 n).sqrt) (HMul.hMul 2 ↑n).sqrt
    -/
  · exact_mod_cast Real.nat_sqrt_le_real_sqrt
    /-
      🎉 no goals
    -/
    /-
      case h₂.hx
      n : Nat
      n_large : LE.le 512 n
      ⊢ LE.le 1 ↑4
    -/
  · norm_num1
    /-
      🎉 no goals
    -/
    /-
      case h₂.hyz
      n : Nat
      n_large : LE.le 512 n
      ⊢ LE.le (↑(HDiv.hDiv (HMul.hMul 2 n) 3)) (HDiv.hDiv (HMul.hMul 2 ↑n) 3)
    -/
  · exact cast_div_le.trans (by norm_cast)
    /-
      🎉 no goals
    -/


/-- A lemma that tells us that, in the case where Bertrand's postulate does not hold, the prime
factorization of the central binomial coefficient only has factors at most `2 * n / 3 + 1`.
-/
theorem centralBinom_factorization_small (n : ℕ) (n_large : 2 < n)
    (no_prime : ¬∃ p : ℕ, p.Prime ∧ n < p ∧ p ≤ 2 * n) :
    centralBinom n = ∏ p ∈ Finset.range (2 * n / 3 + 1), p ^ (centralBinom n).factorization p := by
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    ⊢ Eq n.centralBinom ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMul 2 n) 3) 1) …
  -/
  refine (Eq.trans ?_ n.prod_pow_factorization_centralBinom).symm
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    ⊢ Eq ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMul 2 n) 3) 1)).prod fun p => …
  -/
  apply Finset.prod_subset
    /-
      case h
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      ⊢ HasSubset.Subset (Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMul 2 n) 3) 1))  …
    -/
  · exact Finset.range_subset.2 (add_le_add_right (Nat.div_le_self _ _) _)
    /-
      🎉 no goals
    -/
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)) x → …
  -/
  intro x hx h2x
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    x : Nat
    hx : Membership.mem (Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)) x
    h2x : Not (Membership.mem (Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMul 2 n)  …
    ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
  -/
  rw [Finset.mem_range, Nat.lt_succ_iff] at hx h2x
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    x : Nat
    hx : LE.le x (HMul.hMul 2 n)
    h2x : Not (LE.le x (HDiv.hDiv (HMul.hMul 2 n) 3))
    ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
  -/
  rw [not_le, div_lt_iff_lt_mul three_pos, mul_comm x] at h2x
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    x : Nat
    hx : LE.le x (HMul.hMul 2 n)
    h2x : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 x)
    ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
  -/
  replace no_prime := not_exists.mp no_prime x
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    x : Nat
    hx : LE.le x (HMul.hMul 2 n)
    h2x : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 x)
    no_prime : Not (And (Nat.Prime x) (And (LT.lt n x) (LE.le x (HMul.hMul 2 n))))
    ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
  -/
  rw [← and_assoc, not_and', not_and_or, not_lt] at no_prime
  /-
    case hf
    n : Nat
    n_large : LT.lt 2 n
    x : Nat
    hx : LE.le x (HMul.hMul 2 n)
    h2x : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 x)
    no_prime : LE.le x (HMul.hMul 2 n) → Or (Not (Nat.Prime x)) (LE.le x n)
    ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
  -/
  cases' no_prime hx with h h
    /-
      case hf.inl
      n : Nat
      n_large : LT.lt 2 n
      x : Nat
      hx : LE.le x (HMul.hMul 2 n)
      h2x : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 x)
      no_prime : LE.le x (HMul.hMul 2 n) → Or (Not (Nat.Prime x)) (LE.le x n)
      h : Not (Nat.Prime x)
      ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
    -/
  · rw [factorization_eq_zero_of_non_prime n.centralBinom h, Nat.pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case hf.inr
      n : Nat
      n_large : LT.lt 2 n
      x : Nat
      hx : LE.le x (HMul.hMul 2 n)
      h2x : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 x)
      no_prime : LE.le x (HMul.hMul 2 n) → Or (Not (Nat.Prime x)) (LE.le x n)
      h : LE.le x n
      ⊢ Eq (HPow.hPow x (n.centralBinom.factorization x)) 1
    -/
  · rw [factorization_centralBinom_of_two_mul_self_lt_three_mul n_large h h2x, Nat.pow_zero]
    /-
      🎉 no goals
    -/


/-- An upper bound on the central binomial coefficient used in the proof of Bertrand's postulate.
The bound splits the prime factors of `centralBinom n` into those
1. At most `sqrt (2 * n)`, which contribute at most `2 * n` for each such prime.
2. Between `sqrt (2 * n)` and `2 * n / 3`, which contribute at most `4^(2 * n / 3)` in total.
3. Between `2 * n / 3` and `n`, which do not exist.
4. Between `n` and `2 * n`, which would not exist in the case where Bertrand's postulate is false.
5. Above `2 * n`, which do not exist.
-/
theorem centralBinom_le_of_no_bertrand_prime (n : ℕ) (n_large : 2 < n)
    (no_prime : ¬∃ p : ℕ, Nat.Prime p ∧ n < p ∧ p ≤ 2 * n) :
    centralBinom n ≤ (2 * n) ^ sqrt (2 * n) * 4 ^ (2 * n / 3) := by
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    ⊢ LE.le n.centralBinom (HMul.hMul (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).s …
  -/
  have n_pos : 0 < n := (Nat.zero_le _).trans_lt n_large
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    n_pos : LT.lt 0 n
    ⊢ LE.le n.centralBinom (HMul.hMul (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).s …
  -/
  have n2_pos : 1 ≤ 2 * n := mul_pos (zero_lt_two' ℕ) n_pos
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    n_pos : LT.lt 0 n
    n2_pos : LE.le 1 (HMul.hMul 2 n)
    ⊢ LE.le n.centralBinom (HMul.hMul (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).s …
  -/
  let S := (Finset.range (2 * n / 3 + 1)).filter Nat.Prime
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    n_pos : LT.lt 0 n
    n2_pos : LE.le 1 (HMul.hMul 2 n)
    S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
    ⊢ LE.le n.centralBinom (HMul.hMul (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n).s …
  -/
  let f x := x ^ n.centralBinom.factorization x
  have : ∏ x ∈ S, f x = ∏ x ∈ Finset.range (2 * n / 3 + 1), f x := by
    refine Finset.prod_filter_of_ne fun p _ h => ?_
    contrapose! h; dsimp only [f]
    rw [factorization_eq_zero_of_non_prime n.centralBinom h, _root_.pow_zero]
  rw [centralBinom_factorization_small n n_large no_prime, ← this, ←
    Finset.prod_filter_mul_prod_filter_not S (· ≤ sqrt (2 * n))]
  /-
    n : Nat
    n_large : LT.lt 2 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    n_pos : LT.lt 0 n
    n2_pos : LE.le 1 (HMul.hMul 2 n)
    S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
    f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
    this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
    ⊢ LE.le (HMul.hMul ((Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S). …
  -/
  apply mul_le_mul'
    /-
      case h₁
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ LE.le ((Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S).prod fun x  …
    -/
  · refine (Finset.prod_le_prod' fun p _ => (?_ : f p ≤ 2 * n)).trans ?_
      /-
        case h₁.refine_1
        n : Nat
        n_large : LT.lt 2 n
        no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
        n_pos : LT.lt 0 n
        n2_pos : LE.le 1 (HMul.hMul 2 n)
        S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
        f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
        this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
        p : Nat
        x✝ : Membership.mem (Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S) p
        ⊢ LE.le (f p) (HMul.hMul 2 n)
      -/
    · exact pow_factorization_choose_le (mul_pos two_pos n_pos)
      /-
        🎉 no goals
      -/
    /-
      case h₁.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ LE.le ((Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S).prod fun i  …
    -/
    have : (Finset.Icc 1 (sqrt (2 * n))).card = sqrt (2 * n) := by rw [card_Icc, Nat.add_sub_cancel]
    /-
      case h₁.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this✝ : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hM …
      this : Eq (Finset.Icc 1 (HMul.hMul 2 n).sqrt).card (HMul.hMul 2 n).sqrt
      ⊢ LE.le ((Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S).prod fun i  …
    -/
    rw [Finset.prod_const]
    /-
      case h₁.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this✝ : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hM …
      this : Eq (Finset.Icc 1 (HMul.hMul 2 n).sqrt).card (HMul.hMul 2 n).sqrt
      ⊢ LE.le (HPow.hPow (HMul.hMul 2 n) (Finset.filter (fun x => LE.le x (HMul.hMul …
    -/
    refine pow_right_mono₀ n2_pos ((Finset.card_le_card fun x hx => ?_).trans this.le)
    /-
      case h₁.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this✝ : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hM …
      this : Eq (Finset.Icc 1 (HMul.hMul 2 n).sqrt).card (HMul.hMul 2 n).sqrt
      x : Nat
      hx : Membership.mem (Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S) x
      ⊢ Membership.mem (Finset.Icc 1 (HMul.hMul 2 n).sqrt) x
    -/
    obtain ⟨h1, h2⟩ := Finset.mem_filter.1 hx
    /-
      case h₁.refine_2.intro
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this✝ : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hM …
      this : Eq (Finset.Icc 1 (HMul.hMul 2 n).sqrt).card (HMul.hMul 2 n).sqrt
      x : Nat
      hx : Membership.mem (Finset.filter (fun x => LE.le x (HMul.hMul 2 n).sqrt) S) x
      h1 : Membership.mem S x
      h2 : LE.le x (HMul.hMul 2 n).sqrt
      ⊢ Membership.mem (Finset.Icc 1 (HMul.hMul 2 n).sqrt) x
    -/
    exact Finset.mem_Icc.mpr ⟨(Finset.mem_filter.1 h1).2.one_lt.le, h2⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ LE.le ((Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt)) S).prod  …
    -/
  · refine le_trans ?_ (primorial_le_4_pow (2 * n / 3))
    /-
      case h₂
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ LE.le ((Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt)) S).prod  …
    -/
    refine (Finset.prod_le_prod' fun p hp => (?_ : f p ≤ p)).trans ?_
      /-
        case h₂.refine_1
        n : Nat
        n_large : LT.lt 2 n
        no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
        n_pos : LT.lt 0 n
        n2_pos : LE.le 1 (HMul.hMul 2 n)
        S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
        f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
        this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
        p : Nat
        hp : Membership.mem (Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt …
        ⊢ LE.le (f p) p
      -/
    · obtain ⟨h1, h2⟩ := Finset.mem_filter.1 hp
      /-
        case h₂.refine_1.intro
        n : Nat
        n_large : LT.lt 2 n
        no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
        n_pos : LT.lt 0 n
        n2_pos : LE.le 1 (HMul.hMul 2 n)
        S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
        f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
        this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
        p : Nat
        hp : Membership.mem (Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt …
        h1 : Membership.mem S p
        h2 : Not (LE.le p (HMul.hMul 2 n).sqrt)
        ⊢ LE.le (f p) p
      -/
      refine (pow_right_mono₀ (Finset.mem_filter.1 h1).2.one_lt.le ?_).trans (pow_one p).le
      /-
        case h₂.refine_1.intro
        n : Nat
        n_large : LT.lt 2 n
        no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
        n_pos : LT.lt 0 n
        n2_pos : LE.le 1 (HMul.hMul 2 n)
        S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
        f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
        this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
        p : Nat
        hp : Membership.mem (Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt …
        h1 : Membership.mem S p
        h2 : Not (LE.le p (HMul.hMul 2 n).sqrt)
        ⊢ LE.le (n.centralBinom.factorization p) 1
      -/
      exact Nat.factorization_choose_le_one (sqrt_lt'.mp <| not_le.1 h2)
      /-
        🎉 no goals
      -/
    /-
      case h₂.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ LE.le ((Finset.filter (fun x => Not (LE.le x (HMul.hMul 2 n).sqrt)) S).prod  …
    -/
    refine Finset.prod_le_prod_of_subset_of_one_le' (Finset.filter_subset _ _) ?_
    /-
      case h₂.refine_2
      n : Nat
      n_large : LT.lt 2 n
      no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
      n_pos : LT.lt 0 n
      n2_pos : LE.le 1 (HMul.hMul 2 n)
      S : Finset Nat := Finset.filter Nat.Prime (Finset.range (HAdd.hAdd (HDiv.hDiv  …
      f : Nat → Nat := fun x => HPow.hPow x (n.centralBinom.factorization x)
      this : Eq (S.prod fun x => f x) ((Finset.range (HAdd.hAdd (HDiv.hDiv (HMul.hMu …
      ⊢ ∀ (i : Nat), Membership.mem (Finset.filter (fun p => Nat.Prime p) (Finset.ra …
    -/
    exact fun p hp _ => (Finset.mem_filter.1 hp).2.one_lt.le
    /-
      🎉 no goals
    -/


/-- Proves that **Bertrand's postulate** holds for all sufficiently large `n`.
-/
theorem exists_prime_lt_and_le_two_mul_eventually (n : ℕ) (n_large : 512 ≤ n) :
    ∃ p : ℕ, p.Prime ∧ n < p ∧ p ≤ 2 * n := by
  -- Assume there is no prime in the range.
  /-
    n : Nat
    n_large : LE.le 512 n
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  by_contra no_prime
  -- Then we have the above sub-exponential bound on the size of this central binomial coefficient.
  -- We now couple this bound with an exponential lower bound on the central binomial coefficient,
  -- yielding an inequality which we have seen is false for large enough n.
  /-
    n : Nat
    n_large : LE.le 512 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    ⊢ False
  -/
  have H1 : n * (2 * n) ^ sqrt (2 * n) * 4 ^ (2 * n / 3) ≤ 4 ^ n := bertrand_main_inequality n_large
  have H2 : 4 ^ n < n * n.centralBinom :=
    Nat.four_pow_lt_mul_centralBinom n (le_trans (by norm_num1) n_large)
  have H3 : n.centralBinom ≤ (2 * n) ^ sqrt (2 * n) * 4 ^ (2 * n / 3) :=
    centralBinom_le_of_no_bertrand_prime n (lt_of_lt_of_le (by norm_num1) n_large) no_prime
  /-
    n : Nat
    n_large : LE.le 512 n
    no_prime : Not (Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    H1 : LE.le (HMul.hMul (HMul.hMul n (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n). …
    H2 : LT.lt (HPow.hPow 4 n) (HMul.hMul n n.centralBinom)
    H3 : LE.le n.centralBinom (HMul.hMul (HPow.hPow (HMul.hMul 2 n) (HMul.hMul 2 n …
    ⊢ False
  -/
  rw [mul_assoc] at H1; exact not_le.2 H2 ((mul_le_mul_left' H3 n).trans H1)
                        /-
                          🎉 no goals
                        -/


/-- Proves that Bertrand's postulate holds over all positive naturals less than n by identifying a
descending list of primes, each no more than twice the next, such that the list contains a witness
for each number ≤ n.
-/
theorem exists_prime_lt_and_le_two_mul_succ {n} (q) {p : ℕ} (prime_p : Nat.Prime p)
    (covering : p ≤ 2 * q) (H : n < q → ∃ p : ℕ, p.Prime ∧ n < p ∧ p ≤ 2 * n) (hn : n < p) :
    ∃ p : ℕ, p.Prime ∧ n < p ∧ p ≤ 2 * n := by
  /-
    n q p : Nat
    prime_p : Nat.Prime p
    covering : LE.le p (HMul.hMul 2 q)
    H : LT.lt n q → Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    hn : LT.lt n p
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  by_cases h : p ≤ 2 * n; · exact ⟨p, prime_p, hn, h⟩
                            /-
                              🎉 no goals
                            -/
  /-
    case neg
    n q p : Nat
    prime_p : Nat.Prime p
    covering : LE.le p (HMul.hMul 2 q)
    H : LT.lt n q → Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (H …
    hn : LT.lt n p
    h : Not (LE.le p (HMul.hMul 2 n))
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  exact H (lt_of_mul_lt_mul_left' (lt_of_lt_of_le (not_le.1 h) covering))
  /-
    🎉 no goals
  -/


/--
**Bertrand's Postulate**: For any positive natural number, there is a prime which is greater than
it, but no more than twice as large.
-/
theorem exists_prime_lt_and_le_two_mul (n : ℕ) (hn0 : n ≠ 0) :
    ∃ p, Nat.Prime p ∧ n < p ∧ p ≤ 2 * n := by
  -- Split into cases whether `n` is large or small
  /-
    n : Nat
    hn0 : Ne n 0
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  cases' lt_or_le 511 n with h h
  -- If `n` is large, apply the lemma derived from the inequalities on the central binomial
  -- coefficient.
    /-
      case inl
      n : Nat
      hn0 : Ne n 0
      h : LT.lt 511 n
      ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
    -/
  · exact exists_prime_lt_and_le_two_mul_eventually n h
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    hn0 : Ne n 0
    h : LE.le n 511
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  replace h : n < 521 := h.trans_lt (by norm_num1)
  /-
    case inr
    n : Nat
    hn0 : Ne n 0
    h : LT.lt n 521
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (LE.le p (HMul.hMul 2 n)))
  -/
  revert h
  -- For small `n`, supply a list of primes to cover the initial cases.
  open Lean Elab Tactic in
  run_tac do
    for i in [317, 163, 83, 43, 23, 13, 7, 5, 3, 2] do
      let i : Term := quote i
      evalTactic <| ←
        `(tactic| refine exists_prime_lt_and_le_two_mul_succ $i (by norm_num1) (by norm_num1) ?_)
  exact fun h2 => ⟨2, prime_two, h2, Nat.mul_le_mul_left 2 (Nat.pos_of_ne_zero hn0)⟩


alias bertrand := Nat.exists_prime_lt_and_le_two_mul


