theorem natAbs_eq_iff_mul_self_eq {a b : ℤ} : a.natAbs = b.natAbs ↔ a * a = b * b := by
  /-
    a b : Int
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_eq_iff_mul_self_eq, abs_eq_natAbs, abs_eq_natAbs]
  /-
    a b : Int
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq ↑a.natAbs ↑b.natAbs)
  -/
  exact Int.natCast_inj.symm
  /-
    🎉 no goals
  -/


theorem natAbs_lt_iff_mul_self_lt {a b : ℤ} : a.natAbs < b.natAbs ↔ a * a < b * b := by
  /-
    a b : Int
    ⊢ Iff (LT.lt a.natAbs b.natAbs) (LT.lt (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_lt_iff_mul_self_lt, abs_eq_natAbs, abs_eq_natAbs]
  /-
    a b : Int
    ⊢ Iff (LT.lt a.natAbs b.natAbs) (LT.lt ↑a.natAbs ↑b.natAbs)
  -/
  exact Int.ofNat_lt.symm
  /-
    🎉 no goals
  -/


theorem natAbs_le_iff_mul_self_le {a b : ℤ} : a.natAbs ≤ b.natAbs ↔ a * a ≤ b * b := by
  /-
    a b : Int
    ⊢ Iff (LE.le a.natAbs b.natAbs) (LE.le (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_le_iff_mul_self_le, abs_eq_natAbs, abs_eq_natAbs]
  /-
    a b : Int
    ⊢ Iff (LE.le a.natAbs b.natAbs) (LE.le ↑a.natAbs ↑b.natAbs)
  -/
  exact Int.ofNat_le.symm
  /-
    🎉 no goals
  -/


