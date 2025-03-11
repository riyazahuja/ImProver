theorem FloorSemiring.eventually_mul_pow_lt_factorial_sub (a c : K) (d : ℕ) :
    ∀ᶠ n in atTop, a * c ^ n < (n - d)! := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedRing K
    inst✝ : FloorSemiring K
    a c : K
    d : Nat
    ⊢ Filter.Eventually (fun n => LT.lt (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub  …
  -/
  filter_upwards [Nat.eventually_mul_pow_lt_factorial_sub ⌈|a|⌉₊ ⌈|c|⌉₊ d] with n h
  calc a * c ^ n
    _ ≤ |a * c ^ n| := le_abs_self _
    _ ≤ ⌈|a|⌉₊ * (⌈|c|⌉₊ : K) ^ n := ?_
    _ = ↑(⌈|a|⌉₊ * ⌈|c|⌉₊ ^ n) := ?_
    _ < (n - d)! := Nat.cast_lt.mpr h
    /-
      case h.calc_1
      K : Type u_1
      inst✝¹ : LinearOrderedRing K
      inst✝ : FloorSemiring K
      a c : K
      d n : Nat
      h : LT.lt (HMul.hMul (Nat.ceil (abs a)) (HPow.hPow (Nat.ceil (abs c)) n)) (HSu …
      ⊢ LE.le (abs (HMul.hMul a (HPow.hPow c n))) (HMul.hMul (↑(Nat.ceil (abs a))) ( …
    -/
  · rw [abs_mul, abs_pow]
    /-
      case h.calc_1
      K : Type u_1
      inst✝¹ : LinearOrderedRing K
      inst✝ : FloorSemiring K
      a c : K
      d n : Nat
      h : LT.lt (HMul.hMul (Nat.ceil (abs a)) (HPow.hPow (Nat.ceil (abs c)) n)) (HSu …
      ⊢ LE.le (HMul.hMul (abs a) (HPow.hPow (abs c) n)) (HMul.hMul (↑(Nat.ceil (abs  …
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
    gcongr <;> try first | positivity | apply Nat.le_ceil
               /-
                 🎉 no goals
               -/
    /-
      case h.calc_2
      K : Type u_1
      inst✝¹ : LinearOrderedRing K
      inst✝ : FloorSemiring K
      a c : K
      d n : Nat
      h : LT.lt (HMul.hMul (Nat.ceil (abs a)) (HPow.hPow (Nat.ceil (abs c)) n)) (HSu …
      ⊢ Eq (HMul.hMul (↑(Nat.ceil (abs a))) (HPow.hPow (↑(Nat.ceil (abs c))) n)) ↑(H …
    -/
  · simp_rw [Nat.cast_mul, Nat.cast_pow]
    /-
      🎉 no goals
    -/

