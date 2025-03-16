/-- The matrix [a, -b; b, a] (inspired by multiplication by a complex number); it is an element of
$GL_2(R)$ if `a ^ 2 + b ^ 2` is nonzero. -/
@[simps! (config := .asFn) val]
def planeConformalMatrix {R} [Field R] (a b : R) (hab : a ^ 2 + b ^ 2 ≠ 0) :
    Matrix.GeneralLinearGroup (Fin 2) R :=
                                                       /-
                                                         R : Type ?u.5
                                                         inst✝ : Field R
                                                         a b : R
                                                         hab : Ne (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) 0
                                                         ⊢ Ne (Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons (Neg.neg b)  …
                                                       -/
  GeneralLinearGroup.mkOfDetNeZero !![a, -b; b, a] (by simpa [det_fin_two, sq] using hab)
                                                       /-
                                                         🎉 no goals
                                                       -/

/- TODO: Add Iwasawa matrices `n_x=!![1,x; 0,1]`, `a_t=!![exp(t/2),0;0,exp(-t/2)]` and
  `k_θ=!![cos θ, sin θ; -sin θ, cos θ]`
-/

