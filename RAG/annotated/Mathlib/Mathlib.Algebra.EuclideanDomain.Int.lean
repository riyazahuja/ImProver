instance Int.euclideanDomain : EuclideanDomain ℤ :=
  { inferInstanceAs (CommRing Int), inferInstanceAs (Nontrivial Int) with
    quotient := (· / ·), quotient_zero := Int.ediv_zero, remainder := (· % ·),
    quotient_mul_add_remainder_eq := Int.ediv_add_emod,
    r := fun a b => a.natAbs < b.natAbs,
    r_wellFounded := (measure natAbs).wf
    remainder_lt := fun a b b0 => Int.ofNat_lt.1 <| by
      /-
        a b : Int
        b0 : Ne b 0
        ⊢ LT.lt ↑((fun x1 x2 => HMod.hMod x1 x2) a b).natAbs ↑b.natAbs
      -/
      rw [Int.natAbs_of_nonneg (Int.emod_nonneg _ b0), ← Int.abs_eq_natAbs]
      /-
        a b : Int
        b0 : Ne b 0
        ⊢ LT.lt (HMod.hMod a b) (abs b)
      -/
      exact Int.emod_lt _ b0
      /-
        🎉 no goals
      -/
    mul_left_not_lt := fun a b b0 =>
      not_lt_of_ge <| by
        /-
          a b : Int
          b0 : Ne b 0
          ⊢ GE.ge (HMul.hMul a b).natAbs a.natAbs
        -/
        rw [← mul_one a.natAbs, Int.natAbs_mul]
        /-
          a b : Int
          b0 : Ne b 0
          ⊢ GE.ge (HMul.hMul a.natAbs b.natAbs) (HMul.hMul a.natAbs 1)
        -/
        rw [← Int.natAbs_pos] at b0
        /-
          a b : Int
          b0 : LT.lt 0 b.natAbs
          ⊢ GE.ge (HMul.hMul a.natAbs b.natAbs) (HMul.hMul a.natAbs 1)
        -/
        exact Nat.mul_le_mul_left _ b0 }
        /-
          🎉 no goals
        -/

