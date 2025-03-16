@[simp]
theorem eval₂_mul' :
    (p * q).eval₂ (algebraMap R S) x = p.eval₂ (algebraMap R S) x * q.eval₂ (algebraMap R S) x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    x : S
    p q : Polynomial R
    ⊢ Eq (Polynomial.eval₂ (algebraMap R S) x (HMul.hMul p q)) (HMul.hMul (Polynom …
  -/
  exact eval₂_mul_noncomm _ _ fun k => Algebra.commute_algebraMap_left (coeff q k) x
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_pow' (n : ℕ) :
    (p ^ n).eval₂ (algebraMap R S) x = (p.eval₂ (algebraMap R S) x) ^ n := by
  induction n with
  | zero => simp only [pow_zero, eval₂_one]
  | succ n ih => rw [pow_succ, pow_succ, eval₂_mul', ih]


@[simp]
theorem eval₂_comp' : eval₂ (algebraMap R S) x (p.comp q) =
    eval₂ (algebraMap R S) (eval₂ (algebraMap R S) x q) p := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs => simp only [add_comp, eval₂_add, hr, hs]
  | h_monomial n a => simp only [monomial_comp, eval₂_mul', eval₂_C, eval₂_monomial, eval₂_pow']


