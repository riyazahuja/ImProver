theorem det_eq_prod_roots_charpoly_of_splits (hAps : A.charpoly.Splits (RingHom.id R)) :
    A.det = (Matrix.charpoly A).roots.prod := by
  rw [det_eq_sign_charpoly_coeff, ← charpoly_natDegree_eq_dim A,
    Polynomial.prod_roots_eq_coeff_zero_of_monic_of_splits A.charpoly_monic hAps, ← mul_assoc,
    ← pow_two, pow_right_comm, neg_one_sq, one_pow, one_mul]


theorem trace_eq_sum_roots_charpoly_of_splits (hAps : A.charpoly.Splits (RingHom.id R)) :
    A.trace = (Matrix.charpoly A).roots.sum := by
  /-
    n : Type u_1
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    R : Type u_2
    inst✝ : Field R
    A : Matrix n n R
    hAps : Polynomial.Splits (RingHom.id R) A.charpoly
    ⊢ Eq A.trace A.charpoly.roots.sum
  -/
  cases' isEmpty_or_nonempty n with h
  · rw [Matrix.trace, Fintype.sum_empty, Matrix.charpoly,
      det_eq_one_of_card_eq_zero (Fintype.card_eq_zero_iff.2 h), Polynomial.roots_one,
      Multiset.empty_eq_zero, Multiset.sum_zero]
  · rw [trace_eq_neg_charpoly_coeff, neg_eq_iff_eq_neg,
      ← Polynomial.sum_roots_eq_nextCoeff_of_monic_of_split A.charpoly_monic hAps, nextCoeff,
      charpoly_natDegree_eq_dim, if_neg (Fintype.card_ne_zero : Fintype.card n ≠ 0)]


theorem det_eq_prod_roots_charpoly [IsAlgClosed R] : A.det = (Matrix.charpoly A).roots.prod :=
  det_eq_prod_roots_charpoly_of_splits (IsAlgClosed.splits A.charpoly)


theorem trace_eq_sum_roots_charpoly [IsAlgClosed R] : A.trace = (Matrix.charpoly A).roots.sum :=
  trace_eq_sum_roots_charpoly_of_splits (IsAlgClosed.splits A.charpoly)


