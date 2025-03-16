theorem UpperHalfPlane.abs_exp_two_pi_I_lt_one (z : ℍ) :
    ‖(Complex.exp (2 * π * Complex.I * z))‖ < 1 := by
  simp only [coe_I, Complex.norm_eq_abs, Complex.abs_exp, mul_re, re_ofNat, ofReal_re, im_ofNat,
    ofReal_im, mul_zero, sub_zero, Complex.I_re, mul_im, zero_mul, add_zero, Complex.I_im, mul_one,
    sub_self, coe_re, coe_im, zero_sub, exp_lt_one_iff, Left.neg_neg_iff]
  /-
    z : UpperHalfPlane
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul 2 Real.pi) z.im)
  -/
  positivity
  /-
    🎉 no goals
  -/

