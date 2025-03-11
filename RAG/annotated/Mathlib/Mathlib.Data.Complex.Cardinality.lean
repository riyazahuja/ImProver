/-- The cardinality of the complex numbers, as a type. -/
@[simp]
theorem mk_complex : #ℂ = 𝔠 := by
  /-
    ⊢ Eq (Cardinal.mk Complex) Cardinal.continuum
  -/
  rw [mk_congr Complex.equivRealProd, mk_prod, lift_id, mk_real, continuum_mul_self]
  /-
    🎉 no goals
  -/


/-- The cardinality of the complex numbers, as a set. -/
                                                        /-
                                                          ⊢ Eq (Cardinal.mk ↑Set.univ) Cardinal.continuum
                                                        -/
theorem mk_univ_complex : #(Set.univ : Set ℂ) = 𝔠 := by rw [mk_univ, mk_complex]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The complex numbers are not countable. -/
theorem not_countable_complex : ¬(Set.univ : Set ℂ).Countable := by
  /-
    ⊢ Not Set.univ.Countable
  -/
  rw [← le_aleph0_iff_set_countable, not_le, mk_univ_complex]
  /-
    ⊢ LT.lt Cardinal.aleph0 Cardinal.continuum
  -/
  apply cantor
  /-
    🎉 no goals
  -/

