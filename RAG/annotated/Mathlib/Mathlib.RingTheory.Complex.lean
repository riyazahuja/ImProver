theorem Algebra.leftMulMatrix_complex (z : ℂ) :
    Algebra.leftMulMatrix Complex.basisOneI z = !![z.re, -z.im; z.im, z.re] := by
  /-
    z : Complex
    ⊢ Eq ((Algebra.leftMulMatrix Complex.basisOneI) z) (Matrix.of (Matrix.vecCons  …
  -/
  ext i j
  rw [Algebra.leftMulMatrix_eq_repr_mul, Complex.coe_basisOneI_repr, Complex.coe_basisOneI, mul_re,
    mul_im, Matrix.of_apply]
  /-
    case a
    z : Complex
    i j : Fin 2
    ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul z.re (Matrix.vecCons 1 (Matrix.vecC …
  -/
  fin_cases j
  · simp only [Fin.zero_eta, id_eq, Matrix.cons_val_zero, one_re, mul_one, one_im, mul_zero,
      sub_zero, zero_add, Matrix.cons_val_fin_one]
    /-
      case a.«_@».Mathlib.Data.Matrix.Defs._hyg.177.«0»
      z : Complex
      i : Fin 2
      ⊢ Eq (Matrix.vecCons z.re (Matrix.vecCons z.im Matrix.vecEmpty) i) (Matrix.vec …
    -/
                    /-
                      🎉 no goals
                    -/
    fin_cases i <;> rfl
                    /-
                      🎉 no goals
                    -/
  · simp only [Fin.mk_one, id_eq, Matrix.cons_val_one, Matrix.head_cons, I_re, mul_zero, I_im,
      mul_one, zero_sub, add_zero, Matrix.cons_val_fin_one]
    /-
      case a.«_@».Mathlib.Data.Matrix.Defs._hyg.177.«1»
      z : Complex
      i : Fin 2
      ⊢ Eq (Matrix.vecCons (Neg.neg z.im) (Matrix.vecCons z.re Matrix.vecEmpty) i) ( …
    -/
                    /-
                      🎉 no goals
                    -/
    fin_cases i <;> rfl
                    /-
                      🎉 no goals
                    -/


theorem Algebra.trace_complex_apply (z : ℂ) : Algebra.trace ℝ ℂ z = 2 * z.re := by
  rw [Algebra.trace_eq_matrix_trace Complex.basisOneI, Algebra.leftMulMatrix_complex,
    Matrix.trace_fin_two]
  /-
    z : Complex
    ⊢ Eq (HAdd.hAdd (Matrix.of (Matrix.vecCons (Matrix.vecCons z.re (Matrix.vecCon …
  -/
  exact (two_mul _).symm
  /-
    🎉 no goals
  -/


theorem Algebra.norm_complex_apply (z : ℂ) : Algebra.norm ℝ z = Complex.normSq z := by
  rw [Algebra.norm_eq_matrix_det Complex.basisOneI, Algebra.leftMulMatrix_complex,
    Matrix.det_fin_two, normSq_apply]
  /-
    z : Complex
    ⊢ Eq (HSub.hSub (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons z.re (Ma …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Algebra.norm_complex_eq : Algebra.norm ℝ = normSq.toMonoidHom :=
  MonoidHom.ext Algebra.norm_complex_apply

