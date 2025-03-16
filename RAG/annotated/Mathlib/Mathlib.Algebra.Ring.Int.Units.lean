lemma units_eq_one_or (u : ℤˣ) : u = 1 ∨ u = -1 := by
  /-
    u : Units Int
    ⊢ Or (Eq u 1) (Eq u (-1))
  -/
  simpa only [Units.ext_iff] using isUnit_eq_one_or u.isUnit
  /-
    🎉 no goals
  -/


lemma units_ne_iff_eq_neg {u v : ℤˣ} : u ≠ v ↔ u = -v := by
  /-
    u v : Units Int
    ⊢ Iff (Ne u v) (Eq u (Neg.neg v))
  -/
  simpa only [Ne, Units.ext_iff] using isUnit_ne_iff_eq_neg u.isUnit v.isUnit
  /-
    🎉 no goals
  -/


