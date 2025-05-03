theorem X_pow_sub_X_sub_one_irreducible_aux (z : ℂ) : ¬(z ^ n = z + 1 ∧ z ^ n + z ^ 2 = 0) := by
  /-
    n : Nat
    z : Complex
    ⊢ Not (And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) …
  -/
  rintro ⟨h1, h2⟩
  replace h3 : z ^ 3 = 1 := by
    linear_combination (1 - z - z ^ 2 - z ^ n) * h1 + (z ^ n - 2) * h2
  have key : z ^ n = 1 ∨ z ^ n = z ∨ z ^ n = z ^ 2 := by
    rw [← Nat.mod_add_div n 3, pow_add, pow_mul, h3, one_pow, mul_one]
    have : n % 3 < 3 := Nat.mod_lt n zero_lt_three
    interval_cases n % 3 <;>
    simp only [this, pow_zero, pow_one, eq_self_iff_true, or_true, true_or]
  have z_ne_zero : z ≠ 0 := fun h =>
    zero_ne_one ((zero_pow three_ne_zero).symm.trans (show (0 : ℂ) ^ 3 = 1 from h ▸ h3))
  /-
    case intro
    n : Nat
    z : Complex
    h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
    h2 : Eq (HAdd.hAdd (HPow.hPow z n) (HPow.hPow z 2)) 0
    h3 : Eq (HPow.hPow z 3) 1
    key : Or (Eq (HPow.hPow z n) 1) (Or (Eq (HPow.hPow z n) z) (Eq (HPow.hPow z n) …
    z_ne_zero : Ne z 0
    ⊢ False
  -/
  rcases key with (key | key | key)
    /-
      case intro.inl
      n : Nat
      z : Complex
      h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
      h2 : Eq (HAdd.hAdd (HPow.hPow z n) (HPow.hPow z 2)) 0
      h3 : Eq (HPow.hPow z 3) 1
      z_ne_zero : Ne z 0
      key : Eq (HPow.hPow z n) 1
      ⊢ False
    -/
  · exact z_ne_zero (by rwa [key, self_eq_add_left] at h1)
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inl
      n : Nat
      z : Complex
      h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
      h2 : Eq (HAdd.hAdd (HPow.hPow z n) (HPow.hPow z 2)) 0
      h3 : Eq (HPow.hPow z 3) 1
      z_ne_zero : Ne z 0
      key : Eq (HPow.hPow z n) z
      ⊢ False
    -/
  · exact one_ne_zero (by rwa [key, self_eq_add_right] at h1)
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inr
      n : Nat
      z : Complex
      h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
      h2 : Eq (HAdd.hAdd (HPow.hPow z n) (HPow.hPow z 2)) 0
      h3 : Eq (HPow.hPow z 3) 1
      z_ne_zero : Ne z 0
      key : Eq (HPow.hPow z n) (HPow.hPow z 2)
      ⊢ False
    -/
  · exact z_ne_zero (pow_eq_zero (by rwa [key, add_self_eq_zero] at h2))
    /-
      🎉 no goals
    -/


theorem X_pow_sub_X_sub_one_irreducible (hn1 : n ≠ 1) : Irreducible (X ^ n - X - 1 : ℤ[X]) := by
  /-
    n : Nat
    hn1 : Ne n 1
    ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
  -/
  by_cases hn0 : n = 0
    /-
      case pos
      n : Nat
      hn1 : Ne n 1
      hn0 : Eq n 0
      ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
    -/
  · rw [hn0, pow_zero, sub_sub, add_comm, ← sub_sub, sub_self, zero_sub]
    /-
      case pos
      n : Nat
      hn1 : Ne n 1
      hn0 : Eq n 0
      ⊢ Irreducible (Neg.neg Polynomial.X)
    -/
    exact Associated.irreducible ⟨-1, mul_neg_one X⟩ irreducible_X
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
  -/
  have hn : 1 < n := Nat.one_lt_iff_ne_zero_and_ne_one.mpr ⟨hn0, hn1⟩
  have hp : (X ^ n - X - 1 : ℤ[X]) = trinomial 0 1 n (-1) (-1) 1 := by
    simp only [trinomial, C_neg, C_1]; ring
  /-
    case neg
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
  -/
  rw [hp]
  /-
    case neg
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    ⊢ Irreducible (Polynomial.trinomial 0 1 n (-1) (-1) 1)
  -/
  apply IsUnitTrinomial.irreducible_of_coprime' ⟨0, 1, n, zero_lt_one, hn, -1, -1, 1, rfl⟩
  /-
    case neg
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    ⊢ ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) (Polynomial.trinomial 0  …
  -/
  rintro z ⟨h1, h2⟩
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq ((Polynomial.aeval z) (Polynomial.trinomial 0 1 n ↑(-1) ↑(-1) ↑1)) 0
    h2 : Eq ((Polynomial.aeval z) (Polynomial.trinomial 0 1 n ↑(-1) ↑(-1) ↑1).mirr …
    ⊢ False
  -/
  apply X_pow_sub_X_sub_one_irreducible_aux (n := n) z
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq ((Polynomial.aeval z) (Polynomial.trinomial 0 1 n ↑(-1) ↑(-1) ↑1)) 0
    h2 : Eq ((Polynomial.aeval z) (Polynomial.trinomial 0 1 n ↑(-1) ↑(-1) ↑1).mirr …
    ⊢ And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) (HPo …
  -/
  rw [trinomial_mirror zero_lt_one hn (-1 : ℤˣ).ne_zero (1 : ℤˣ).ne_zero] at h2
  simp_rw [trinomial, aeval_add, aeval_mul, aeval_X_pow, aeval_C,
    Units.val_neg, Units.val_one, map_neg, map_one] at h1 h2
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (-1) (HPow.hPow z 0)) (HMul.hMul (-1) …
    h2 : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow z 0)) (HMul.hMul (-1) (H …
    ⊢ And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) (HPo …
  -/
  replace h1 : z ^ n = z + 1 := by linear_combination h1
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h2 : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow z 0)) (HMul.hMul (-1) (H …
    h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
    ⊢ And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) (HPo …
  -/
  replace h2 := mul_eq_zero_of_left h2 z
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
    h2 : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow z 0)) (HMul.h …
    ⊢ And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) (HPo …
  -/
  rw [add_mul, add_mul, add_zero, mul_assoc (-1 : ℂ), ← pow_succ, Nat.sub_add_cancel hn.le] at h2
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
    h2 : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul 1 (HPow.hPow z 0)) z) (HMu …
    ⊢ And (Eq (HPow.hPow z n) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HPow.hPow z n) (HPo …
  -/
  rw [h1] at h2 ⊢
  /-
    case neg.intro
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hn : LT.lt 1 n
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    z : Complex
    h1 : Eq (HPow.hPow z n) (HAdd.hAdd z 1)
    h2 : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul 1 (HPow.hPow z 0)) z) (HMu …
    ⊢ And (Eq (HAdd.hAdd z 1) (HAdd.hAdd z 1)) (Eq (HAdd.hAdd (HAdd.hAdd z 1) (HPo …
  -/
  exact ⟨rfl, by linear_combination -h2⟩
  /-
    🎉 no goals
  -/


theorem X_pow_sub_X_sub_one_irreducible_rat (hn1 : n ≠ 1) : Irreducible (X ^ n - X - 1 : ℚ[X]) := by
  /-
    n : Nat
    hn1 : Ne n 1
    ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
  -/
  by_cases hn0 : n = 0
    /-
      case pos
      n : Nat
      hn1 : Ne n 1
      hn0 : Eq n 0
      ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
    -/
  · rw [hn0, pow_zero, sub_sub, add_comm, ← sub_sub, sub_self, zero_sub]
    /-
      case pos
      n : Nat
      hn1 : Ne n 1
      hn0 : Eq n 0
      ⊢ Irreducible (Neg.neg Polynomial.X)
    -/
    exact Associated.irreducible ⟨-1, mul_neg_one X⟩ irreducible_X
    /-
      🎉 no goals
    -/
  have hp : (X ^ n - X - 1 : ℤ[X]) = trinomial 0 1 n (-1) (-1) 1 := by
    simp only [trinomial, C_neg, C_1]; ring
  /-
    case neg
    n : Nat
    hn1 : Ne n 1
    hn0 : Not (Eq n 0)
    hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
    ⊢ Irreducible (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1)
  -/
  have hn : 1 < n := Nat.one_lt_iff_ne_zero_and_ne_one.mpr ⟨hn0, hn1⟩
  have h := (IsPrimitive.Int.irreducible_iff_irreducible_map_cast ?_).mp
    (X_pow_sub_X_sub_one_irreducible hn1)
  · rwa [Polynomial.map_sub, Polynomial.map_sub, Polynomial.map_pow, Polynomial.map_one,
      Polynomial.map_X] at h
    /-
      case neg.refine_1
      n : Nat
      hn1 : Ne n 1
      hn0 : Not (Eq n 0)
      hp : Eq (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1) (Pol …
      hn : LT.lt 1 n
      ⊢ (HSub.hSub (HSub.hSub (HPow.hPow Polynomial.X n) Polynomial.X) 1).IsPrimitive
    -/
  · exact hp.symm ▸ (trinomial_monic zero_lt_one hn).isPrimitive
    /-
      🎉 no goals
    -/


