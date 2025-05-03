/-- A natural number `t` is invertible in a field `K` if the characteristic of `K` does not divide
`t`. -/
def invertibleOfRingCharNotDvd {t : ℕ} (not_dvd : ¬ringChar K ∣ t) : Invertible (t : K) :=
  invertibleOfNonzero fun h => not_dvd ((ringChar.spec K t).mp h)


theorem not_ringChar_dvd_of_invertible {t : ℕ} [Invertible (t : K)] : ¬ringChar K ∣ t := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    t : Nat
    inst✝ : Invertible ↑t
    ⊢ Not (Dvd.dvd (ringChar K) t)
  -/
  rw [← ringChar.spec, ← Ne]
  /-
    K : Type u_1
    inst✝¹ : Field K
    t : Nat
    inst✝ : Invertible ↑t
    ⊢ Ne (↑t) 0
  -/
  exact Invertible.ne_zero (t : K)
  /-
    🎉 no goals
  -/


/-- A natural number `t` is invertible in a field `K` of characteristic `p` if `p` does not divide
`t`. -/
def invertibleOfCharPNotDvd {p : ℕ} [CharP K p] {t : ℕ} (not_dvd : ¬p ∣ t) : Invertible (t : K) :=
  invertibleOfNonzero fun h => not_dvd ((CharP.cast_eq_zero_iff K p t).mp h)

-- warning: this could potentially loop with `Invertible.ne_zero` - if there is weird type-class
-- loops, watch out for that.

instance invertibleOfPos [CharZero K] (n : ℕ) [NeZero n] : Invertible (n : K) :=
  invertibleOfNonzero <| NeZero.out


instance invertibleSucc (n : ℕ) : Invertible (n.succ : K) :=
  invertibleOfNonzero (Nat.cast_ne_zero.mpr (Nat.succ_ne_zero _))


instance invertibleTwo : Invertible (2 : K) :=
                                    /-
                                      K : Type u_1
                                      inst✝¹ : DivisionRing K
                                      inst✝ : CharZero K
                                      ⊢ Ne 2 0
                                    -/
  invertibleOfNonzero (mod_cast (by decide : 2 ≠ 0))
                                    /-
                                      🎉 no goals
                                    -/


instance invertibleThree : Invertible (3 : K) :=
                                    /-
                                      K : Type u_1
                                      inst✝¹ : DivisionRing K
                                      inst✝ : CharZero K
                                      ⊢ Ne 3 0
                                    -/
  invertibleOfNonzero (mod_cast (by decide : 3 ≠ 0))
                                    /-
                                      🎉 no goals
                                    -/


