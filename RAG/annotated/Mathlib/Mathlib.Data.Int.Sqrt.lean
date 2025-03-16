/-- `sqrt z` is the square root of an integer `z`. If `z` is positive, it returns the largest
integer `r` such that `r * r ≤ n`. If it is negative, it returns `0`. For example, `sqrt (-1) = 0`,
`sqrt 1 = 1`, `sqrt 2 = 1` -/
@[pp_nodot]
def sqrt (z : ℤ) : ℤ :=
  Nat.sqrt <| Int.toNat z


theorem sqrt_eq (n : ℤ) : sqrt (n * n) = n.natAbs := by
  /-
    n : Int
    ⊢ Eq (Int.sqrt (HMul.hMul n n)) ↑n.natAbs
  -/
  rw [sqrt, ← natAbs_mul_self, toNat_natCast, Nat.sqrt_eq]
  /-
    🎉 no goals
  -/


theorem exists_mul_self (x : ℤ) : (∃ n, n * n = x) ↔ sqrt x * sqrt x = x :=
                     /-
                       x : Int
                       x✝ : Exists fun n => Eq (HMul.hMul n n) x
                       n : Int
                       hn : Eq (HMul.hMul n n) x
                       ⊢ Eq (HMul.hMul (Int.sqrt x) (Int.sqrt x)) x
                     -/
  ⟨fun ⟨n, hn⟩ => by rw [← hn, sqrt_eq, ← Int.ofNat_mul, natAbs_mul_self], fun h => ⟨sqrt x, h⟩⟩
                     /-
                       🎉 no goals
                     -/


theorem sqrt_nonneg (n : ℤ) : 0 ≤ sqrt n :=
  natCast_nonneg _


@[simp, norm_cast]
                                                                   /-
                                                                     n : Nat
                                                                     ⊢ Eq (Int.sqrt ↑n) ↑n.sqrt
                                                                   -/
theorem sqrt_natCast (n : ℕ) : Int.sqrt (n : ℤ) = Nat.sqrt n := by rw [sqrt, toNat_ofNat]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem sqrt_ofNat (n : ℕ) : Int.sqrt ofNat(n) = Nat.sqrt ofNat(n) :=
  sqrt_natCast _


