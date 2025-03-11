/-- Square root function on rational numbers, defined by taking the (integer) square root of the
numerator and the square root (on natural numbers) of the denominator. -/
@[pp_nodot]
def sqrt (q : ℚ) : ℚ := mkRat (Int.sqrt q.num) (Nat.sqrt q.den)


theorem sqrt_eq (q : ℚ) : Rat.sqrt (q * q) = |q| := by
  /-
    q : Rat
    ⊢ Eq (Rat.sqrt (HMul.hMul q q)) (abs q)
  -/
  rw [sqrt, mul_self_num, mul_self_den, Int.sqrt_eq, Nat.sqrt_eq, abs_def, divInt_ofNat]
  /-
    🎉 no goals
  -/


theorem exists_mul_self (x : ℚ) : (∃ q, q * q = x) ↔ Rat.sqrt x * Rat.sqrt x = x :=
                     /-
                       x : Rat
                       x✝ : Exists fun q => Eq (HMul.hMul q q) x
                       n : Rat
                       hn : Eq (HMul.hMul n n) x
                       ⊢ Eq (HMul.hMul (Rat.sqrt x) (Rat.sqrt x)) x
                     -/
  ⟨fun ⟨n, hn⟩ => by rw [← hn, sqrt_eq, abs_mul_abs_self], fun h => ⟨Rat.sqrt x, h⟩⟩
                     /-
                       🎉 no goals
                     -/


lemma sqrt_nonneg (q : ℚ) : 0 ≤ Rat.sqrt q := mkRat_nonneg (Int.sqrt_nonneg _) _


/-- `IsSquare` can be decided on `ℚ` by checking against the square root. -/
instance : DecidablePred (IsSquare : ℚ → Prop) :=
  fun m => decidable_of_iff' (sqrt m * sqrt m = m) <| by
    /-
      m : Rat
      ⊢ Iff (IsSquare m) (Eq (HMul.hMul (Rat.sqrt m) (Rat.sqrt m)) m)
    -/
    simp_rw [← exists_mul_self m, IsSquare, eq_comm]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem sqrt_intCast (z : ℤ) : Rat.sqrt (z : ℚ) = Int.sqrt z := by
  /-
    z : Int
    ⊢ Eq (Rat.sqrt ↑z) ↑(Int.sqrt z)
  -/
  simp only [sqrt, num_intCast, den_intCast, Nat.sqrt_one, mkRat_one]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem sqrt_natCast (n : ℕ) : Rat.sqrt (n : ℚ) = Nat.sqrt n := by
  /-
    n : Nat
    ⊢ Eq (Rat.sqrt ↑n) ↑n.sqrt
  -/
  rw [← Int.cast_natCast, sqrt_intCast, Int.sqrt_natCast, Int.cast_natCast]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem sqrt_ofNat (n : ℕ) : Rat.sqrt (no_index (OfNat.ofNat n) : ℚ) = Nat.sqrt (OfNat.ofNat n) :=
  sqrt_natCast _


