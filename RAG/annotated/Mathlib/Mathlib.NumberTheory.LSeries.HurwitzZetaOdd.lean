/-- Variant of `jacobiTheta₂'` which we introduce to simplify some formulae. -/
def jacobiTheta₂'' (z τ : ℂ) : ℂ :=
  cexp (π * I * z ^ 2 * τ) * (jacobiTheta₂' (z * τ) τ / (2 * π * I) + z * jacobiTheta₂ (z * τ) τ)


lemma jacobiTheta₂''_conj (z τ : ℂ) :
    conj (jacobiTheta₂'' z τ) = jacobiTheta₂'' (conj z) (-conj τ) := by
  simp only [jacobiTheta₂'', jacobiTheta₂'_conj, jacobiTheta₂_conj,
    ← exp_conj, conj_ofReal, conj_I, map_mul, map_add, map_div₀, mul_neg, map_pow, map_ofNat,
    neg_mul, div_neg, neg_div, jacobiTheta₂'_neg_left, jacobiTheta₂_neg_left]


/-- Restatement of `jacobiTheta₂'_add_left'`: the function `jacobiTheta₂''` is 1-periodic in `z`. -/
lemma jacobiTheta₂''_add_left (z τ : ℂ) : jacobiTheta₂'' (z + 1) τ = jacobiTheta₂'' z τ := by
  /-
    z τ : Complex
    ⊢ Eq (HurwitzZeta.jacobiTheta₂'' (HAdd.hAdd z 1) τ) (HurwitzZeta.jacobiTheta₂' …
  -/
  simp only [jacobiTheta₂'', add_mul z 1, one_mul, jacobiTheta₂'_add_left', jacobiTheta₂_add_left']
  /-
    z τ : Complex
    ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) Compl …
  -/
  generalize jacobiTheta₂ (z * τ) τ = J
  /-
    z τ J : Complex
    ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) Compl …
  -/
  generalize jacobiTheta₂' (z * τ) τ = J'
  -- clear denominator
  /-
    z τ J J' : Complex
    ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) Compl …
  -/
  simp_rw [div_add' _ _ _ two_pi_I_ne_zero, ← mul_div_assoc]
  /-
    z τ J J' : Complex
    ⊢ Eq (HDiv.hDiv (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (↑Rea …
  -/
  refine congr_arg (· / (2 * π * I)) ?_
  -- get all exponential terms to left
  rw [mul_left_comm _ (cexp _), ← mul_add, mul_assoc (cexp _), ← mul_add, ← mul_assoc (cexp _),
    ← Complex.exp_add]
  /-
    z τ J J' : Complex
    ⊢ Eq (HMul.hMul (Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (↑Rea …
  -/
                            /-
                              🎉 no goals
                            -/
  congrm (cexp ?_ * ?_) <;> ring
                            /-
                              🎉 no goals
                            -/


lemma jacobiTheta₂''_neg_left (z τ : ℂ) : jacobiTheta₂'' (-z) τ = -jacobiTheta₂'' z τ := by
  simp only [jacobiTheta₂'', jacobiTheta₂'_neg_left, jacobiTheta₂_neg_left,
    neg_mul, neg_div, ← neg_add, mul_neg, neg_sq]


lemma jacobiTheta₂'_functional_equation' (z τ : ℂ) :
    jacobiTheta₂' z τ = (-2 * π) / (-I * τ) ^ (3 / 2 : ℂ) * jacobiTheta₂'' z (-1 / τ) := by
  /-
    z τ : Complex
    ⊢ Eq (jacobiTheta₂' z τ) (HMul.hMul (HDiv.hDiv (HMul.hMul (-2) ↑Real.pi) (HPow …
  -/
  rcases eq_or_ne τ 0 with rfl | hτ
    /-
      case inl
      z : Complex
      ⊢ Eq (jacobiTheta₂' z 0) (HMul.hMul (HDiv.hDiv (HMul.hMul (-2) ↑Real.pi) (HPow …
    -/
  · rw [jacobiTheta₂'_undef _ (by simp), mul_zero, zero_cpow (by norm_num), div_zero, zero_mul]
    /-
      🎉 no goals
    -/
  have aux1 : (-2 * π : ℂ) / (2 * π * I) = I := by
    rw [div_eq_iff two_pi_I_ne_zero, mul_comm I, mul_assoc _ I I, I_mul_I, neg_mul, mul_neg,
      mul_one]
  rw [jacobiTheta₂'_functional_equation, ← mul_one_div _ τ, mul_right_comm _ (cexp _),
    (by rw [cpow_one, ← div_div, div_self (neg_ne_zero.mpr I_ne_zero)] :
      1 / τ = -I / (-I * τ) ^ (1 : ℂ)), div_mul_div_comm,
    ← cpow_add _ _ (mul_ne_zero (neg_ne_zero.mpr I_ne_zero) hτ), ← div_mul_eq_mul_div,
    (by norm_num : (1 / 2  + 1 : ℂ) = 3 / 2), mul_assoc (1 / _), mul_assoc (1 / _),
    ← mul_one_div (-2 * π : ℂ), mul_comm _ (1 / _), mul_assoc (1 / _)]
  /-
    case inr
    z τ : Complex
    hτ : Ne τ 0
    aux1 : Eq (HDiv.hDiv (HMul.hMul (-2) ↑Real.pi) (HMul.hMul (HMul.hMul 2 ↑Real.p …
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (HMul.hMul (Neg.neg Complex.I) τ) (3 / …
  -/
  congr 1
  rw [jacobiTheta₂'', div_add' _ _ _ two_pi_I_ne_zero, ← mul_div_assoc, ← mul_div_assoc,
    ← div_mul_eq_mul_div (-2 * π : ℂ), mul_assoc, aux1, mul_div z (-1), mul_neg_one, neg_div τ z,
    jacobiTheta₂_neg_left, jacobiTheta₂'_neg_left, neg_mul, ← mul_neg, ← mul_neg,
    mul_div, mul_neg_one, neg_div, neg_mul, neg_mul, neg_div]
  /-
    case inr.e_a
    z τ : Complex
    hτ : Ne τ 0
    aux1 : Eq (HDiv.hDiv (HMul.hMul (-2) ↑Real.pi) (HMul.hMul (HMul.hMul 2 ↑Real.p …
    ⊢ Eq (HMul.hMul Complex.I (HMul.hMul (Complex.exp (Neg.neg (HDiv.hDiv (HMul.hM …
  -/
  congr 2
  /-
    case inr.e_a.e_a.e_a
    z τ : Complex
    hτ : Ne τ 0
    aux1 : Eq (HDiv.hDiv (HMul.hMul (-2) ↑Real.pi) (HMul.hMul (HMul.hMul 2 ↑Real.p …
    ⊢ Eq (Neg.neg (HSub.hSub (jacobiTheta₂' (HDiv.hDiv z τ) (Neg.neg (HDiv.hDiv 1  …
  -/
  rw [neg_sub, ← sub_eq_neg_add, mul_comm _ (_ * I), ← mul_assoc]
  /-
    🎉 no goals
  -/


/-- Odd Hurwitz zeta kernel (function whose Mellin transform will be the odd part of the completed
Hurwitz zeta function). See `oddKernel_def` for the defining formula, and `hasSum_int_oddKernel`
for an expression as a sum over `ℤ`.
-/
@[irreducible] def oddKernel (a : UnitAddCircle) (x : ℝ) : ℝ :=
  (show Function.Periodic (fun a : ℝ ↦ re (jacobiTheta₂'' a (I * x))) 1 by
    /-
      a : UnitAddCircle
      x : Real
      ⊢ Function.Periodic (fun a => (HurwitzZeta.jacobiTheta₂'' (↑a) (HMul.hMul Comp …
    -/
    intro a; simp only [ofReal_add, ofReal_one, jacobiTheta₂''_add_left]).lift a
             /-
               🎉 no goals
             -/


lemma oddKernel_def (a x : ℝ) : ↑(oddKernel a x) = jacobiTheta₂'' a (I * x) := by
  rw [oddKernel, Function.Periodic.lift_coe, ← conj_eq_iff_re, jacobiTheta₂''_conj, map_mul,
    conj_I, neg_mul, neg_neg, conj_ofReal, conj_ofReal]


lemma oddKernel_def' (a x : ℝ) : ↑(oddKernel ↑a x) = cexp (-π * a ^ 2 * x) *
    (jacobiTheta₂' (a * I * x) (I * x) / (2 * π * I) + a * jacobiTheta₂ (a * I * x) (I * x)) := by
  rw [oddKernel_def, jacobiTheta₂'', ← mul_assoc ↑a I x,
    (by ring : ↑π * I * ↑a ^ 2 * (I * ↑x) = I ^ 2 * ↑π * ↑a ^ 2 * x), I_sq, neg_one_mul]


lemma oddKernel_undef (a : UnitAddCircle) {x : ℝ} (hx : x ≤ 0) : oddKernel a x = 0 := by
  /-
    a : UnitAddCircle
    x : Real
    hx : LE.le x 0
    ⊢ Eq (HurwitzZeta.oddKernel a x) 0
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  rw [← ofReal_eq_zero, oddKernel_def', jacobiTheta₂_undef, jacobiTheta₂'_undef, zero_div, zero_add,
    mul_zero, mul_zero] <;>
  /-
    case H.hτ
    x : Real
    hx : LE.le x 0
    a' : Real
    ⊢ LE.le (HMul.hMul Complex.I ↑x).im 0
  -/
  /-
    🎉 no goals
  -/
  rwa [I_mul_im, ofReal_re]
  /-
    🎉 no goals
  -/


/-- Auxiliary function appearing in the functional equation for the odd Hurwitz zeta kernel, equal
to `∑ (n : ℕ), 2 * n * sin (2 * π * n * a) * exp (-π * n ^ 2 * x)`. See `hasSum_nat_sinKernel`
for the defining sum. -/
@[irreducible] def sinKernel (a : UnitAddCircle) (x : ℝ) : ℝ :=
  (show Function.Periodic (fun ξ : ℝ ↦ re (jacobiTheta₂' ξ (I * x) / (-2 * π))) 1 by
    /-
      a : UnitAddCircle
      x : Real
      ⊢ Function.Periodic (fun ξ => (HDiv.hDiv (jacobiTheta₂' (↑ξ) (HMul.hMul Comple …
    -/
    intro ξ; simp_rw [ofReal_add, ofReal_one, jacobiTheta₂'_add_left]).lift a
             /-
               🎉 no goals
             -/


lemma sinKernel_def (a x : ℝ) : ↑(sinKernel ↑a x) = jacobiTheta₂' a (I * x) / (-2 * π) := by
  /-
    a x : Real
    ⊢ Eq (↑(HurwitzZeta.sinKernel (↑a) x)) (HDiv.hDiv (jacobiTheta₂' (↑a) (HMul.hM …
  -/
  rw [sinKernel, Function.Periodic.lift_coe, re_eq_add_conj, map_div₀, jacobiTheta₂'_conj]
  /-
    a x : Real
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (jacobiTheta₂' (↑a) (HMul.hMul Complex.I …
  -/
  simp_rw [map_mul, conj_I, conj_ofReal, map_neg, map_ofNat, neg_mul, neg_neg, add_self_div_two]
  /-
    🎉 no goals
  -/


lemma sinKernel_undef (a : UnitAddCircle) {x : ℝ} (hx : x ≤ 0) : sinKernel a x = 0 := by
  /-
    a : UnitAddCircle
    x : Real
    hx : LE.le x 0
    ⊢ Eq (HurwitzZeta.sinKernel a x) 0
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  rw [← ofReal_eq_zero, sinKernel_def, jacobiTheta₂'_undef _ (by rwa [I_mul_im, ofReal_re]),
    zero_div]


lemma oddKernel_neg (a : UnitAddCircle) (x : ℝ) : oddKernel (-a) x = -oddKernel a x := by
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.oddKernel (Neg.neg a) x) (Neg.neg (HurwitzZeta.oddKernel a x))
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  rw [← ofReal_inj, ← QuotientAddGroup.mk_neg, oddKernel_def, ofReal_neg, ofReal_neg, oddKernel_def,
    jacobiTheta₂''_neg_left]


lemma oddKernel_zero (x : ℝ) : oddKernel 0 x = 0 := by
  /-
    x : Real
    ⊢ Eq (HurwitzZeta.oddKernel 0 x) 0
  -/
  simpa only [neg_zero, eq_neg_self_iff] using oddKernel_neg 0 x
  /-
    🎉 no goals
  -/


lemma sinKernel_neg (a : UnitAddCircle) (x : ℝ) :
    sinKernel (-a) x = -sinKernel a x := by
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.sinKernel (Neg.neg a) x) (Neg.neg (HurwitzZeta.sinKernel a x))
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  rw [← ofReal_inj, ← QuotientAddGroup.mk_neg, ofReal_neg, sinKernel_def, sinKernel_def, ofReal_neg,
    jacobiTheta₂'_neg_left, neg_div]


lemma sinKernel_zero (x : ℝ) : sinKernel 0 x = 0 := by
  /-
    x : Real
    ⊢ Eq (HurwitzZeta.sinKernel 0 x) 0
  -/
  simpa only [neg_zero, eq_neg_self_iff] using sinKernel_neg 0 x
  /-
    🎉 no goals
  -/


/-- The odd kernel is continuous on `Ioi 0`. -/
lemma continuousOn_oddKernel (a : UnitAddCircle) : ContinuousOn (oddKernel a) (Ioi 0) := by
  /-
    a : UnitAddCircle
    ⊢ ContinuousOn (HurwitzZeta.oddKernel a) (Set.Ioi 0)
  -/
  induction' a using QuotientAddGroup.induction_on with a
  suffices ContinuousOn (fun x ↦ (oddKernel a x : ℂ)) (Ioi 0) from
    (continuous_re.comp_continuousOn this).congr fun a _ ↦ (ofReal_re _).symm
  /-
    case H
    a : Real
    ⊢ ContinuousOn (fun x => ↑(HurwitzZeta.oddKernel (↑a) x)) (Set.Ioi 0)
  -/
  simp_rw [oddKernel_def' a]
  /-
    case H
    a : Real
    ⊢ ContinuousOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg …
  -/
  refine fun x hx ↦ ((Continuous.continuousAt ?_).mul ?_).continuousWithinAt
    /-
      case H.refine_1
      a x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ Continuous fun x => Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) (HP …
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case H.refine_2
      a x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ ContinuousAt (fun x => HAdd.hAdd (HDiv.hDiv (jacobiTheta₂' (HMul.hMul (HMul. …
    -/
  · have hf : Continuous fun u : ℝ ↦ (a * I * u, I * u) := by fun_prop
    /-
      case H.refine_2
      a x : Real
      hx : Membership.mem (Set.Ioi 0) x
      hf : Continuous fun u => { fst := HMul.hMul (HMul.hMul (↑a) Complex.I) ↑u, snd …
      ⊢ ContinuousAt (fun x => HAdd.hAdd (HDiv.hDiv (jacobiTheta₂' (HMul.hMul (HMul. …
    -/
    apply ContinuousAt.add
    · exact ((continuousAt_jacobiTheta₂' (a * I * x) (by rwa [I_mul_im, ofReal_re])).comp
        (f := fun u : ℝ ↦ (a * I * u, I * u)) hf.continuousAt).div_const _
    · exact continuousAt_const.mul <| (continuousAt_jacobiTheta₂ (a * I * x)
        (by rwa [I_mul_im, ofReal_re])).comp (f := fun u : ℝ ↦ (a * I * u, I * u)) hf.continuousAt


lemma continuousOn_sinKernel (a : UnitAddCircle) : ContinuousOn (sinKernel a) (Ioi 0) := by
  /-
    a : UnitAddCircle
    ⊢ ContinuousOn (HurwitzZeta.sinKernel a) (Set.Ioi 0)
  -/
  induction' a using QuotientAddGroup.induction_on with a
  suffices ContinuousOn (fun x ↦ (sinKernel a x : ℂ)) (Ioi 0) from
    (continuous_re.comp_continuousOn this).congr fun a _ ↦ (ofReal_re _).symm
  /-
    case H
    a : Real
    ⊢ ContinuousOn (fun x => ↑(HurwitzZeta.sinKernel (↑a) x)) (Set.Ioi 0)
  -/
  simp_rw [sinKernel_def]
  /-
    case H
    a : Real
    ⊢ ContinuousOn (fun x => HDiv.hDiv (jacobiTheta₂' (↑a) (HMul.hMul Complex.I ↑x …
  -/
  apply (continuousOn_of_forall_continuousAt (fun x hx ↦ ?_)).div_const
  /-
    a x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ ContinuousAt (fun x => jacobiTheta₂' (↑a) (HMul.hMul Complex.I ↑x)) x
  -/
  have h := continuousAt_jacobiTheta₂' a (by rwa [I_mul_im, ofReal_re])
  /-
    a x : Real
    hx : Membership.mem (Set.Ioi 0) x
    h : ContinuousAt (fun p => jacobiTheta₂' p.1 p.2) { fst := ↑a, snd := HMul.hMu …
    ⊢ ContinuousAt (fun x => jacobiTheta₂' (↑a) (HMul.hMul Complex.I ↑x)) x
  -/
  fun_prop
  /-
    🎉 no goals
  -/


lemma oddKernel_functional_equation (a : UnitAddCircle) (x : ℝ) :
    oddKernel a x = 1 / x ^ (3 / 2 : ℝ) * sinKernel a (1 / x) := by
  -- first reduce to `0 < x`
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.oddKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (3 / 2)) …
  -/
  rcases le_or_lt x 0 with hx | hx
    /-
      case inl
      a : UnitAddCircle
      x : Real
      hx : LE.le x 0
      ⊢ Eq (HurwitzZeta.oddKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (3 / 2)) …
    -/
  · rw [oddKernel_undef _ hx, sinKernel_undef _ (one_div_nonpos.mpr hx), mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a : UnitAddCircle
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (HurwitzZeta.oddKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (3 / 2)) …
  -/
  induction' a using QuotientAddGroup.induction_on with a
  have h1 : -1 / (I * ↑(1 / x)) = I * x := by rw [one_div, ofReal_inv, mul_comm, ← div_div,
    div_inv_eq_mul, div_eq_mul_inv, inv_I, mul_neg, neg_one_mul, neg_mul, neg_neg, mul_comm]
  have h2 : (-I * (I * ↑(1 / x))) = 1 / x := by
    rw [← mul_assoc, neg_mul, I_mul_I, neg_neg, one_mul, ofReal_div, ofReal_one]
  have h3 : (x : ℂ) ^ (3 / 2 : ℂ) ≠ 0 := by
    simp only [Ne, cpow_eq_zero_iff, ofReal_eq_zero, hx.ne', false_and, not_false_eq_true]
  /-
    case inr.H
    x : Real
    hx : LT.lt 0 x
    a : Real
    h1 : Eq (HDiv.hDiv (-1) (HMul.hMul Complex.I ↑(HDiv.hDiv 1 x))) (HMul.hMul Com …
    h2 : Eq (HMul.hMul (Neg.neg Complex.I) (HMul.hMul Complex.I ↑(HDiv.hDiv 1 x))) …
    h3 : Ne (HPow.hPow (↑x) (3 / 2)) 0
    ⊢ Eq (HurwitzZeta.oddKernel (↑a) x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (3 /  …
  -/
  have h4 : arg x ≠ π := by rw [arg_ofReal_of_nonneg hx.le]; exact pi_ne_zero.symm
  rw [← ofReal_inj, oddKernel_def, ofReal_mul, sinKernel_def, jacobiTheta₂'_functional_equation',
    h1, h2]
  /-
    case inr.H
    x : Real
    hx : LT.lt 0 x
    a : Real
    h1 : Eq (HDiv.hDiv (-1) (HMul.hMul Complex.I ↑(HDiv.hDiv 1 x))) (HMul.hMul Com …
    h2 : Eq (HMul.hMul (Neg.neg Complex.I) (HMul.hMul Complex.I ↑(HDiv.hDiv 1 x))) …
    h3 : Ne (HPow.hPow (↑x) (3 / 2)) 0
    h4 : Ne (↑x).arg Real.pi
    ⊢ Eq (HurwitzZeta.jacobiTheta₂'' (↑a) (HMul.hMul Complex.I ↑x)) (HMul.hMul (↑( …
  -/
  generalize jacobiTheta₂'' a (I * ↑x) = J
  rw [one_div (x : ℂ), inv_cpow _ _ h4, div_inv_eq_mul, one_div, ofReal_inv, ofReal_cpow hx.le,
    ofReal_div, ofReal_ofNat, ofReal_ofNat, ← mul_div_assoc _ _ (-2 * π : ℂ),
    eq_div_iff <| mul_ne_zero (neg_ne_zero.mpr two_ne_zero) (ofReal_ne_zero.mpr pi_ne_zero),
    ← div_eq_inv_mul, eq_div_iff h3, mul_comm J _, mul_right_comm]


lemma hasSum_int_oddKernel (a : ℝ) {x : ℝ} (hx : 0 < x) :
    HasSum (fun n : ℤ ↦ (n + a) * rexp (-π * (n + a) ^ 2 * x)) (oddKernel ↑a x) := by
  /-
    a x : Real
    hx : LT.lt 0 x
    ⊢ HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) a) (Real.exp (HMul.hMul (HMul.hMu …
  -/
  rw [← hasSum_ofReal, oddKernel_def' a x]
  /-
    a x : Real
    hx : LT.lt 0 x
    ⊢ HasSum (fun x_1 => ↑(HMul.hMul (HAdd.hAdd (↑x_1) a) (Real.exp (HMul.hMul (HM …
  -/
  have h1 := hasSum_jacobiTheta₂_term (a * I * x) (by rwa [I_mul_im, ofReal_re])
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    ⊢ HasSum (fun x_1 => ↑(HMul.hMul (HAdd.hAdd (↑x_1) a) (Real.exp (HMul.hMul (HM …
  -/
  have h2 := hasSum_jacobiTheta₂'_term (a * I * x) (by rwa [I_mul_im, ofReal_re])
  refine (((h2.div_const (2 * π * I)).add (h1.mul_left ↑a)).mul_left
    (cexp (-π * a ^ 2 * x))).congr_fun (fun n ↦ ?_)
  rw [jacobiTheta₂'_term, mul_assoc (2 * π * I), mul_div_cancel_left₀ _ two_pi_I_ne_zero, ← add_mul,
    mul_left_comm, jacobiTheta₂_term, ← Complex.exp_add]
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    h2 : HasSum (fun n => jacobiTheta₂'_term n (HMul.hMul (HMul.hMul (↑a) Complex. …
    n : Int
    ⊢ Eq (↑(HMul.hMul (HAdd.hAdd (↑n) a) (Real.exp (HMul.hMul (HMul.hMul (Neg.neg  …
  -/
  push_cast
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    h2 : HasSum (fun n => jacobiTheta₂'_term n (HMul.hMul (HMul.hMul (↑a) Complex. …
    n : Int
    ⊢ Eq (HMul.hMul (HAdd.hAdd ↑n ↑a) (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg  …
  -/
  simp only [← mul_assoc, ← add_mul]
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    h2 : HasSum (fun n => jacobiTheta₂'_term n (HMul.hMul (HMul.hMul (↑a) Complex. …
    n : Int
    ⊢ Eq (HMul.hMul (HAdd.hAdd ↑n ↑a) (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg  …
  -/
  congrm _ * cexp (?_ * x)
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    h2 : HasSum (fun n => jacobiTheta₂'_term n (HMul.hMul (HMul.hMul (↑a) Complex. …
    n : Int
    ⊢ Eq (HMul.hMul (Neg.neg ↑Real.pi) (HPow.hPow (HAdd.hAdd ↑n ↑a) 2)) (HAdd.hAdd …
  -/
  simp only [mul_right_comm _ I, add_mul, mul_assoc _ I, I_mul_I]
  /-
    a x : Real
    hx : LT.lt 0 x
    h1 : HasSum (fun n => jacobiTheta₂_term n (HMul.hMul (HMul.hMul (↑a) Complex.I …
    h2 : HasSum (fun n => jacobiTheta₂'_term n (HMul.hMul (HMul.hMul (↑a) Complex. …
    n : Int
    ⊢ Eq (HMul.hMul (Neg.neg ↑Real.pi) (HPow.hPow (HAdd.hAdd ↑n ↑a) 2)) (HAdd.hAdd …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma hasSum_int_sinKernel (a : ℝ) {t : ℝ} (ht : 0 < t) : HasSum
    (fun n : ℤ ↦ -I * n * cexp (2 * π * I * a * n) * rexp (-π * n ^ 2 * t)) ↑(sinKernel a t) := by
  have h : -2 * (π : ℂ) ≠ (0 : ℂ) := by
    simp only [neg_mul, ne_eq, neg_eq_zero, mul_eq_zero,
      OfNat.ofNat_ne_zero, ofReal_eq_zero, pi_ne_zero, or_self, not_false_eq_true]
  /-
    a t : Real
    ht : LT.lt 0 t
    h : Ne (HMul.hMul (-2) ↑Real.pi) 0
    ⊢ HasSum (fun n => HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Complex.I) ↑n) (Co …
  -/
  rw [sinKernel_def]
  refine ((hasSum_jacobiTheta₂'_term a
    (by rwa [I_mul_im, ofReal_re])).div_const _).congr_fun fun n ↦ ?_
  rw [jacobiTheta₂'_term, jacobiTheta₂_term, ofReal_exp, mul_assoc (-I * n), ← Complex.exp_add,
    eq_div_iff h, ofReal_mul, ofReal_mul, ofReal_pow, ofReal_neg, ofReal_intCast,
    mul_comm _ (-2 * π : ℂ), ← mul_assoc]
  /-
    a t : Real
    ht : LT.lt 0 t
    h : Ne (HMul.hMul (-2) ↑Real.pi) 0
    n : Int
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) (HMul.hMul (Neg.neg Compl …
  -/
  congrm ?_ * cexp (?_ + ?_)
    /-
      case refine_1
      a t : Real
      ht : LT.lt 0 t
      h : Ne (HMul.hMul (-2) ↑Real.pi) 0
      n : Int
      ⊢ Eq (HMul.hMul (HMul.hMul (-2) ↑Real.pi) (HMul.hMul (Neg.neg Complex.I) ↑n))  …
    -/
  · simp only [neg_mul, mul_neg, neg_neg, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a t : Real
      ht : LT.lt 0 t
      h : Ne (HMul.hMul (-2) ↑Real.pi) 0
      n : Int
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑a) ↑n …
    -/
  · exact mul_right_comm (2 * π * I) a n
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      a t : Real
      ht : LT.lt 0 t
      h : Ne (HMul.hMul (-2) ↑Real.pi) 0
      n : Int
      ⊢ Eq (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) (HPow.hPow (↑n) 2)) ↑t) (HMul.hM …
    -/
  · simp only [← mul_assoc, mul_comm _ I, I_mul_I, neg_one_mul]
    /-
      🎉 no goals
    -/


lemma hasSum_nat_sinKernel (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℕ ↦ 2 * n * Real.sin (2 * π * a * n) * rexp (-π * n ^ 2 * t))
    (sinKernel ↑a t) := by
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => HMul.hMul (HMul.hMul (HMul.hMul 2 ↑n) (Real.sin (HMul.hMul  …
  -/
  rw [← hasSum_ofReal]
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul (HMul.hMul 2 ↑x) (Real.sin (HMul.hMu …
  -/
  have := (hasSum_int_sinKernel a ht).nat_add_neg
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Co …
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul (HMul.hMul 2 ↑x) (Real.sin (HMul.hMu …
  -/
  simp only [Int.cast_zero, sq (0 : ℂ), zero_mul, mul_zero, add_zero] at this
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Co …
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul (HMul.hMul 2 ↑x) (Real.sin (HMul.hMu …
  -/
  refine this.congr_fun fun n ↦ ?_
  simp_rw [Int.cast_neg, neg_sq, mul_neg, ofReal_mul, Int.cast_natCast, ofReal_natCast,
      ofReal_ofNat, ← add_mul, ofReal_sin, Complex.sin]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Co …
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑n) (HDiv.hDiv (HMul.hMul (HSub.hSub ( …
  -/
  push_cast
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Co …
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑n) (HDiv.hDiv (HMul.hMul (HSub.hSub ( …
  -/
  congr 1
  rw [← mul_div_assoc, ← div_mul_eq_mul_div, ← div_mul_eq_mul_div, div_self two_ne_zero, one_mul,
    neg_mul, neg_mul, neg_neg, mul_comm _ I, ← mul_assoc, mul_comm _ I, neg_mul,
    ← sub_eq_neg_add, mul_sub]
  /-
    case e_a
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (Neg.neg Co …
    n : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul Complex.I ↑n) (Complex.exp (Neg.neg (HMu …
  -/
              /-
                🎉 no goals
              -/
  congr 3 <;> ring
              /-
                🎉 no goals
              -/


/-- The function `oddKernel a` has exponential decay at `+∞`, for any `a`. -/
lemma isBigO_atTop_oddKernel (a : UnitAddCircle) :
    ∃ p, 0 < p ∧ IsBigO atTop (oddKernel a) (fun x ↦ Real.exp (-p * x)) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  induction' a using QuotientAddGroup.induction_on with b
  /-
    case H
    b : Real
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  obtain ⟨p, hp, hp'⟩ := HurwitzKernelBounds.isBigO_atTop_F_int_one b
  /-
    case H.intro.intro
    b p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_int 1 ↑b) fun t = …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  refine ⟨p, hp, (Eventually.isBigO ?_).trans hp'⟩
  /-
    case H.intro.intro
    b p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_int 1 ↑b) fun t = …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HurwitzZeta.oddKernel (↑b) x)) …
  -/
  filter_upwards [eventually_gt_atTop 0] with t ht
  simpa only [← (hasSum_int_oddKernel b ht).tsum_eq, Real.norm_eq_abs, HurwitzKernelBounds.F_int,
    HurwitzKernelBounds.f_int, pow_one, norm_mul, abs_of_nonneg (exp_pos _).le] using
    norm_tsum_le_tsum_norm (hasSum_int_oddKernel b ht).summable.norm


/-- The function `sinKernel a` has exponential decay at `+∞`, for any `a`. -/
lemma isBigO_atTop_sinKernel (a : UnitAddCircle) :
    ∃ p, 0 < p ∧ IsBigO atTop (sinKernel a) (fun x ↦ Real.exp (-p * x)) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  induction' a using QuotientAddGroup.induction_on with a
  /-
    case H
    a : Real
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  obtain ⟨p, hp, hp'⟩ := HurwitzKernelBounds.isBigO_atTop_F_nat_one (le_refl 0)
  /-
    case H.intro.intro
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (HurwitzZet …
  -/
  refine ⟨p, hp, (Eventually.isBigO ?_).trans (hp'.const_mul_left 2)⟩
  /-
    case H.intro.intro
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HurwitzZeta.sinKernel (↑a) x)) …
  -/
  filter_upwards [eventually_gt_atTop 0] with t ht
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (HurwitzZeta.sinKernel (↑a) t)) (HMul.hMul 2 (HurwitzKernel …
  -/
  rw [HurwitzKernelBounds.F_nat, ← (hasSum_nat_sinKernel a ht).tsum_eq]
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (tsum fun b => HMul.hMul (HMul.hMul (HMul.hMul 2 ↑b) (Real. …
  -/
  apply tsum_of_norm_bounded (g := fun n ↦ 2 * HurwitzKernelBounds.f_nat 1 0 t n)
    /-
      case h.hg
      a p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
      t : Real
      ht : LT.lt 0 t
      ⊢ HasSum (fun n => HMul.hMul 2 (HurwitzKernelBounds.f_nat 1 0 t n)) (HMul.hMul …
    -/
  · exact (HurwitzKernelBounds.summable_f_nat 1 0 ht).hasSum.mul_left _
    /-
      🎉 no goals
    -/
    /-
      case h.h
      a p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
      t : Real
      ht : LT.lt 0 t
      ⊢ ∀ (i : Nat), LE.le (Norm.norm (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑i) (Real.s …
    -/
  · intro n
    rw [norm_mul, norm_mul, norm_mul, norm_two, mul_assoc, mul_assoc,
      mul_le_mul_iff_of_pos_left two_pos, HurwitzKernelBounds.f_nat, pow_one, add_zero,
      norm_of_nonneg (exp_pos _).le, Real.norm_eq_abs, Nat.abs_cast, ← mul_assoc,
      mul_le_mul_iff_of_pos_right (exp_pos _)]
    /-
      case h.h
      a p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (HurwitzKernelBounds.F_nat 1 0) fun t => …
      t : Real
      ht : LT.lt 0 t
      n : Nat
      ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (Real.sin (HMul.hMul (HMul.hMul (HMul.hMul  …
    -/
    exact mul_le_of_le_one_right (Nat.cast_nonneg _) (abs_sin_le_one _)
    /-
      🎉 no goals
    -/


/-- A `StrongFEPair` structure with `f = oddKernel a` and `g = sinKernel a`. -/
@[simps]
def hurwitzOddFEPair (a : UnitAddCircle) : StrongFEPair ℂ where
  f := ofReal ∘ oddKernel a
  g := ofReal ∘ sinKernel a
  hf_int := (continuous_ofReal.comp_continuousOn (continuousOn_oddKernel a)).locallyIntegrableOn
    measurableSet_Ioi
  hg_int := (continuous_ofReal.comp_continuousOn (continuousOn_sinKernel a)).locallyIntegrableOn
    measurableSet_Ioi
  k := 3 / 2
           /-
             a : UnitAddCircle
             ⊢ LT.lt 0 (3 / 2)
           -/
  hk := by norm_num
           /-
             🎉 no goals
           -/
  hε := one_ne_zero
  f₀ := 0
  hf₀ := rfl
  g₀ := 0
  hg₀ := rfl
  hf_top r := by
    /-
      a : UnitAddCircle
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    let ⟨v, hv, hv'⟩ := isBigO_atTop_oddKernel a
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (HurwitzZeta.oddKernel a) fun x => Real. …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    rw [← isBigO_norm_left] at hv' ⊢
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.oddKern …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (Function.com …
    -/
    simp_rw [Function.comp_def, sub_zero, norm_real]
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.oddKern …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.oddKernel a …
    -/
    exact hv'.trans (isLittleO_exp_neg_mul_rpow_atTop hv _).isBigO
    /-
      🎉 no goals
    -/
  hg_top r := by
    /-
      a : UnitAddCircle
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    let ⟨v, hv, hv'⟩ := isBigO_atTop_sinKernel a
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (HurwitzZeta.sinKernel a) fun x => Real. …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    rw [← isBigO_norm_left] at hv' ⊢
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.sinKern …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (Function.com …
    -/
    simp_rw [Function.comp_def, sub_zero, norm_real]
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.sinKern …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HurwitzZeta.sinKernel a …
    -/
    exact hv'.trans (isLittleO_exp_neg_mul_rpow_atTop hv _).isBigO
    /-
      🎉 no goals
    -/
  h_feq x hx := by simp_rw [Function.comp_apply, one_mul, smul_eq_mul, ← ofReal_mul,
    oddKernel_functional_equation a, one_div x, one_div x⁻¹, inv_rpow (le_of_lt hx), one_div,
    inv_inv]


/-- The entire function of `s` which agrees with
`1 / 2 * Gamma ((s + 1) / 2) * π ^ (-(s + 1) / 2) * ∑' (n : ℤ), sgn (n + a) / |n + a| ^ s`
for `1 < re s`.
-/
def completedHurwitzZetaOdd (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzOddFEPair a).Λ ((s + 1) / 2)) / 2


lemma differentiable_completedHurwitzZetaOdd (a : UnitAddCircle) :
    Differentiable ℂ (completedHurwitzZetaOdd a) :=
  ((hurwitzOddFEPair a).differentiable_Λ.comp
    ((differentiable_id.add_const 1).div_const 2)).div_const 2


/-- The entire function of `s` which agrees with
` Gamma ((s + 1) / 2) * π ^ (-(s + 1) / 2) * ∑' (n : ℕ), sin (2 * π * a * n) / n ^ s`
for `1 < re s`.
-/
def completedSinZeta (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzOddFEPair a).symm.Λ ((s + 1) / 2)) / 2


lemma differentiable_completedSinZeta (a : UnitAddCircle) :
    Differentiable ℂ (completedSinZeta a) :=
  ((hurwitzOddFEPair a).symm.differentiable_Λ.comp
    ((differentiable_id.add_const 1).div_const 2)).div_const 2


lemma completedHurwitzZetaOdd_neg (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaOdd (-a) s = -completedHurwitzZetaOdd a s := by
  simp only [completedHurwitzZetaOdd, StrongFEPair.Λ, hurwitzOddFEPair, mellin, Function.comp_def,
    oddKernel_neg, ofReal_neg, smul_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HDiv.hDiv (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
  -/
  rw [integral_neg, neg_div]
  /-
    🎉 no goals
  -/


lemma completedSinZeta_neg (a : UnitAddCircle) (s : ℂ) :
    completedSinZeta (-a) s = -completedSinZeta a s := by
  simp only [completedSinZeta, StrongFEPair.Λ, mellin, StrongFEPair.symm, WeakFEPair.symm,
    hurwitzOddFEPair, Function.comp_def, sinKernel_neg, ofReal_neg, smul_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HDiv.hDiv (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
  -/
  rw [integral_neg, neg_div]
  /-
    🎉 no goals
  -/



/-- Functional equation for the odd Hurwitz zeta function. -/
theorem completedHurwitzZetaOdd_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaOdd a (1 - s) = completedSinZeta a s := by
  rw [completedHurwitzZetaOdd, completedSinZeta,
    (by { push_cast; ring } : (1 - s + 1) / 2 = ↑(3 / 2 : ℝ) - (s + 1) / 2),
    ← hurwitzOddFEPair_k, (hurwitzOddFEPair a).functional_equation ((s + 1) / 2),
    hurwitzOddFEPair_ε, one_smul]


/-- Functional equation for the odd Hurwitz zeta function (alternative form). -/
lemma completedSinZeta_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedSinZeta a (1 - s) = completedHurwitzZetaOdd a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedSinZeta a (HSub.hSub 1 s)) (HurwitzZeta.completedHu …
  -/
  rw [← completedHurwitzZetaOdd_one_sub, sub_sub_cancel]
  /-
    🎉 no goals
  -/


/-- Formula for `completedSinZeta` as a Dirichlet series in the convergence range
(first version, with sum over `ℤ`). -/
lemma hasSum_int_completedSinZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ Gammaℝ (s + 1) * (-I) * Int.sign n *
    cexp (2 * π * I * a * n) / (↑|n| : ℂ) ^ s / 2) (completedSinZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HAdd …
  -/
  let c (n : ℤ) : ℂ := -I * cexp (2 * π * I * a * n) / 2
  have hc (n : ℤ) : ‖c n‖ = 1 / 2 := by
    simp_rw [c, (by { push_cast; ring } : 2 * π * I * a * n = ↑(2 * π * a * n) * I), norm_div,
      RCLike.norm_ofNat, norm_mul, norm_neg, norm_I, one_mul, norm_exp_ofReal_mul_I]
  have hF t (ht : 0 < t) :
      HasSum (fun n ↦ c n * n * rexp (-π * n ^ 2 * t)) (sinKernel a t / 2) := by
    refine ((hasSum_int_sinKernel a ht).div_const 2).congr_fun fun n ↦ ?_
    rw [div_mul_eq_mul_div, div_mul_eq_mul_div, mul_right_comm (-I)]
  have h_sum : Summable fun i ↦ ‖c i‖ / |↑i| ^ s.re := by
    simp_rw [hc, div_right_comm]
    apply Summable.div_const
    apply Summable.of_nat_of_neg <;>
    · simp only [Int.cast_neg, abs_neg, Int.cast_natCast, Nat.abs_cast]
      rwa [summable_one_div_nat_rpow]
  refine (mellin_div_const .. ▸ hasSum_mellin_pi_mul_sq' (zero_lt_one.trans hs) hF h_sum).congr_fun
    fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    c : Int → Complex := fun n => HDiv.hDiv (HMul.hMul (Neg.neg Complex.I) (Comple …
    hc : ∀ (n : Int), Eq (Norm.norm (c n)) (1 / 2)
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => HMul.hMul (HMul.hMul (c n) ↑n) …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs ↑i) s.re)
    n : Int
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Ga …
  -/
  simp only [Int.sign_eq_sign, SignType.intCast_cast, sign_intCast, ← Int.cast_abs, ofReal_intCast]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    c : Int → Complex := fun n => HDiv.hDiv (HMul.hMul (Neg.neg Complex.I) (Comple …
    hc : ∀ (n : Int), Eq (Norm.norm (c n)) (1 / 2)
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => HMul.hMul (HMul.hMul (c n) ↑n) …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs ↑i) s.re)
    n : Int
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Ga …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Formula for `completedSinZeta` as a Dirichlet series in the convergence range
(second version, with sum over `ℕ`). -/
lemma hasSum_nat_completedSinZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ Gammaℝ (s + 1) * Real.sin (2 * π * a * n) / (n : ℂ) ^ s)
    (completedSinZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Real.sin (HMu …
  -/
  have := (hasSum_int_completedSinZeta a hs).nat_add_neg
  simp_rw [Int.sign_zero, Int.cast_zero, mul_zero, zero_mul, zero_div, add_zero, abs_neg,
    Int.sign_neg, Nat.abs_cast, Int.cast_neg, Int.cast_natCast, ← add_div] at this
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Real.sin (HMu …
  -/
  refine this.congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Real.sin (HMul.hMul (HMul. …
  -/
  rw [div_right_comm]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Real.sin (HMul.hMul (HMul. …
  -/
  rcases eq_or_ne n 0 with rfl | h
  · simp only [Nat.cast_zero, mul_zero, Real.sin_zero, ofReal_zero, zero_div, mul_neg,
      Int.sign_zero, Int.cast_zero, Complex.exp_zero, mul_one, neg_zero, add_zero]
  /-
    case inr
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    h : Ne n 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Real.sin (HMul.hMul (HMul. …
  -/
  simp_rw [Int.sign_natCast_of_ne_zero h, Int.cast_one, ofReal_sin, Complex.sin]
  /-
    case inr
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    h : Ne n 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (HDiv.hDiv (HMul.hMul (HSub. …
  -/
  simp only [← mul_div_assoc, push_cast, mul_assoc (Gammaℝ _), ← mul_add]
  /-
    case inr
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    h : Ne n 0
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (HMul.hMul (HSub. …
  -/
  congr 3
  rw [mul_one, mul_neg_one, neg_neg, neg_mul I, ← sub_eq_neg_add, ← mul_sub, mul_comm,
    mul_neg, neg_mul]
  /-
    case inr.e_a.e_a.e_a
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    n : Nat
    h : Ne n 0
    ⊢ Eq (HMul.hMul Complex.I (HSub.hSub (Complex.exp (Neg.neg (HMul.hMul (HMul.hM …
  -/
              /-
                🎉 no goals
              -/
  congr 3 <;> ring
              /-
                🎉 no goals
              -/


/-- Formula for `completedHurwitzZetaOdd` as a Dirichlet series in the convergence range. -/
lemma hasSum_int_completedHurwitzZetaOdd (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ Gammaℝ (s + 1) * SignType.sign (n + a) / (↑|n + a| : ℂ) ^ s / 2)
    (completedHurwitzZetaOdd a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Si …
  -/
  let r (n : ℤ) : ℝ := n + a
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    r : Int → Real := fun n => HAdd.hAdd (↑n) a
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Si …
  -/
  let c (n : ℤ) : ℂ := 1 / 2
  have hF t (ht : 0 < t) : HasSum (fun n ↦ c n * r n * rexp (-π * (r n) ^ 2 * t))
      (oddKernel a t / 2) := by
    refine ((hasSum_ofReal.mpr (hasSum_int_oddKernel a ht)).div_const 2).congr_fun fun n ↦ ?_
    simp only [r, c, push_cast, div_mul_eq_mul_div, one_mul]
  have h_sum : Summable fun i ↦ ‖c i‖ / |r i| ^ s.re := by
    simp_rw [c, ← mul_one_div ‖_‖]
    apply Summable.mul_left
    rwa [summable_one_div_int_add_rpow]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    r : Int → Real := fun n => HAdd.hAdd (↑n) a
    c : Int → Complex := fun n => 1 / 2
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => HMul.hMul (HMul.hMul (c n) ↑(r …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs (r i)) s …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Si …
  -/
  have := mellin_div_const .. ▸ hasSum_mellin_pi_mul_sq' (zero_lt_one.trans hs) hF h_sum
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    r : Int → Real := fun n => HAdd.hAdd (↑n) a
    c : Int → Complex := fun n => 1 / 2
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => HMul.hMul (HMul.hMul (c n) ↑(r …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs (r i)) s …
    this : HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(Si …
  -/
  refine this.congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    r : Int → Real := fun n => HAdd.hAdd (↑n) a
    c : Int → Complex := fun n => 1 / 2
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => HMul.hMul (HMul.hMul (c n) ↑(r …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs (r i)) s …
    this : HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ …
    n : Int
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd s 1).Gammaℝ ↑(SignType.sign ( …
  -/
  simp only [r, c, mul_one_div, div_mul_eq_mul_div, div_right_comm]
  /-
    🎉 no goals
  -/


/-- The odd part of the Hurwitz zeta function, i.e. the meromorphic function of `s` which agrees
with `1 / 2 * ∑' (n : ℤ), sign (n + a) / |n + a| ^ s` for `1 < re s`-/
noncomputable def hurwitzZetaOdd (a : UnitAddCircle) (s : ℂ) :=
  completedHurwitzZetaOdd a s / Gammaℝ (s + 1)


lemma hurwitzZetaOdd_neg (a : UnitAddCircle) (s : ℂ) :
    hurwitzZetaOdd (-a) s = -hurwitzZetaOdd a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaOdd (Neg.neg a) s) (Neg.neg (HurwitzZeta.hurwitzZ …
  -/
  simp_rw [hurwitzZetaOdd, completedHurwitzZetaOdd_neg, neg_div]
  /-
    🎉 no goals
  -/


/-- The odd Hurwitz zeta function is differentiable everywhere. -/
lemma differentiable_hurwitzZetaOdd (a : UnitAddCircle) :
    Differentiable ℂ (hurwitzZetaOdd a) :=
  (differentiable_completedHurwitzZetaOdd a).mul <| differentiable_Gammaℝ_inv.comp <|
    differentiable_id.add <| differentiable_const _


/-- The sine zeta function, i.e. the meromorphic function of `s` which agrees
with `∑' (n : ℕ), sin (2 * π * a * n) / n ^ s` for `1 < re s`. -/
noncomputable def sinZeta (a : UnitAddCircle) (s : ℂ) :=
  completedSinZeta a s / Gammaℝ (s + 1)


lemma sinZeta_neg (a : UnitAddCircle) (s : ℂ) :
    sinZeta (-a) s = -sinZeta a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.sinZeta (Neg.neg a) s) (Neg.neg (HurwitzZeta.sinZeta a s))
  -/
  simp_rw [sinZeta, completedSinZeta_neg, neg_div]
  /-
    🎉 no goals
  -/


/-- The sine zeta function is differentiable everywhere. -/
lemma differentiableAt_sinZeta (a : UnitAddCircle) :
    Differentiable ℂ (sinZeta a) :=
  (differentiable_completedSinZeta a).mul <| differentiable_Gammaℝ_inv.comp <|
    differentiable_id.add <| differentiable_const _


/-- Formula for `hurwitzZetaOdd` as a Dirichlet series in the convergence range (sum over `ℤ`). -/
theorem hasSum_int_hurwitzZetaOdd (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ SignType.sign (n + a) / (↑|n + a| : ℂ) ^ s / 2) (hurwitzZetaOdd a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a)))  …
  -/
  refine ((hasSum_int_completedHurwitzZetaOdd a hs).div_const (Gammaℝ _)).congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Int
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a))) (HPow.hPow (↑ …
  -/
  have : 0 < re (s + 1) := by rw [add_re, one_re]; positivity
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Int
    this : LT.lt 0 (HAdd.hAdd s 1).re
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a))) (HPow.hPow (↑ …
  -/
  simp only [div_right_comm _ _ (Gammaℝ _), mul_div_cancel_left₀ _ (Gammaℝ_ne_zero_of_re_pos this)]
  /-
    🎉 no goals
  -/


/-- Formula for `hurwitzZetaOdd` as a Dirichlet series in the convergence range, with sum over `ℕ`
(version with absolute values) -/
lemma hasSum_nat_hurwitzZetaOdd (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ (SignType.sign (n + a) / (↑|n + a| : ℂ) ^ s
      - SignType.sign (n + 1 - a) / (↑|n + 1 - a| : ℂ) ^ s) / 2) (hurwitzZetaOdd a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HSub.hSub (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd …
  -/
  refine (hasSum_int_hurwitzZetaOdd a hs).nat_add_neg_add_one.congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a))) (H …
  -/
  rw [Int.cast_neg, Int.cast_add, Int.cast_one, sub_div, sub_eq_add_neg, Int.cast_natCast]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a))) (H …
  -/
  have : -(n + 1) + a = -(n + 1 - a) := by ring_nf
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    this : Eq (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑n) 1)) a) (Neg.neg (HSub.hSub (HAdd …
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) a))) (H …
  -/
  rw [this, Left.sign_neg, abs_neg, SignType.coe_neg, neg_div, neg_div]
  /-
    🎉 no goals
  -/


/-- Formula for `hurwitzZetaOdd` as a Dirichlet series in the convergence range, with sum over `ℕ`
(version without absolute values, assuming `a ∈ Icc 0 1`) -/
lemma hasSum_nat_hurwitzZetaOdd_of_mem_Icc {a : ℝ} (ha : a ∈ Icc 0 1) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ (1 / (n + a : ℂ) ^ s - 1 / (n + 1 - a : ℂ) ^ s) / 2)
    (hurwitzZetaOdd a s) := by
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HSub.hSub (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n  …
  -/
  refine (hasSum_nat_hurwitzZetaOdd a hs).congr_fun fun n ↦ ?_
  suffices ∀ b : ℝ, 0 ≤ b → SignType.sign (n + b) / (↑|n + b| : ℂ) ^ s = 1 / (n + b) ^ s by
    simp only [add_sub_assoc, this a ha.1, this (1 - a) (sub_nonneg.mpr ha.2), push_cast]
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ ∀ (b : Real), LE.le 0 b → Eq (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) b)) …
  -/
  intro b hb
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    b : Real
    hb : LE.le 0 b
    ⊢ Eq (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) b))) (HPow.hPow (↑(abs (HAdd. …
  -/
  rw [abs_of_nonneg (by positivity), (by simp : (n : ℂ) + b = ↑(n + b))]
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    b : Real
    hb : LE.le 0 b
    ⊢ Eq (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) b))) (HPow.hPow (↑(HAdd.hAdd  …
  -/
  rcases lt_or_eq_of_le (by positivity : 0 ≤ n + b) with hb | hb
    /-
      case inl
      a : Real
      ha : Membership.mem (Set.Icc 0 1) a
      s : Complex
      hs : LT.lt 1 s.re
      n : Nat
      b : Real
      hb✝ : LE.le 0 b
      hb : LT.lt 0 (HAdd.hAdd (↑n) b)
      ⊢ Eq (HDiv.hDiv (↑(SignType.sign (HAdd.hAdd (↑n) b))) (HPow.hPow (↑(HAdd.hAdd  …
    -/
  · rw [sign_pos hb, SignType.coe_one]
    /-
      🎉 no goals
    -/
  · rw [← hb, ofReal_zero, zero_cpow ((not_lt.mpr zero_le_one) ∘ (zero_re ▸ · ▸ hs)),
      div_zero, div_zero]


/-- Formula for `sinZeta` as a Dirichlet series in the convergence range, with sum over `ℤ`. -/
theorem hasSum_int_sinZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ -I * n.sign * cexp (2 * π * I * a * n) / ↑|n| ^ s / 2) (sinZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (Neg.neg Complex …
  -/
  rw [sinZeta]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (Neg.neg Complex …
  -/
  refine ((hasSum_int_completedSinZeta a hs).div_const (Gammaℝ (s + 1))).congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Int
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul (Neg.neg Complex.I) ↑n.sign)  …
  -/
  have : 0 < re (s + 1) := by rw [add_re, one_re]; positivity
  simp only [mul_assoc, div_right_comm _ _ (Gammaℝ _),
    mul_div_cancel_left₀ _ (Gammaℝ_ne_zero_of_re_pos this)]


/-- Formula for `sinZeta` as a Dirichlet series in the convergence range, with sum over `ℕ`. -/
lemma hasSum_nat_sinZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ Real.sin (2 * π * a * n) / (n : ℂ) ^ s) (sinZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Re …
  -/
  have := (hasSum_int_sinZeta a hs).nat_add_neg
  simp_rw [abs_neg, Int.sign_neg, Int.cast_neg, Nat.abs_cast, Int.cast_natCast, mul_neg, abs_zero,
    Int.cast_zero, zero_cpow (ne_zero_of_one_lt_re hs), div_zero, zero_div, add_zero] at this
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Re …
  -/
  simp_rw [push_cast, Complex.sin]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul (HSub.hSub (Complex.exp (HM …
  -/
  refine this.congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul  …
    n : Nat
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HSub.hSub (Complex.exp (HMul.hMul (Neg. …
  -/
  rcases ne_or_eq n 0 with h | rfl
  · simp only [neg_mul, sub_mul, div_right_comm _ (2 : ℂ), Int.sign_natCast_of_ne_zero h,
      Int.cast_one, mul_one, mul_comm I, neg_neg, ← add_div, ← sub_eq_neg_add]
    /-
      case inl
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      this : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HMul.hMul  …
      n : Nat
      h : Ne n 0
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HSub.hSub (HMul.hMul (Complex.exp (Neg.neg (HMul.h …
    -/
                /-
                  🎉 no goals
                -/
    congr 5 <;> ring
                /-
                  🎉 no goals
                -/
  · simp only [Nat.cast_zero, Int.sign_zero, Int.cast_zero, mul_zero, zero_mul, neg_zero,
      sub_self, zero_div, zero_add]


/-- Reformulation of `hasSum_nat_sinZeta` using `LSeriesHasSum`. -/
lemma LSeriesHasSum_sin (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    LSeriesHasSum (Real.sin <| 2 * π * a * ·) s (sinZeta a s) :=
  (hasSum_nat_sinZeta a hs).congr_fun
    (LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs) _)


/-- The trivial zeroes of the odd Hurwitz zeta function. -/
theorem hurwitzZetaOdd_neg_two_mul_nat_sub_one (a : UnitAddCircle) (n : ℕ) :
    hurwitzZetaOdd a (-2 * n - 1) = 0 := by
  /-
    a : UnitAddCircle
    n : Nat
    ⊢ Eq (HurwitzZeta.hurwitzZetaOdd a (HSub.hSub (HMul.hMul (-2) ↑n) 1)) 0
  -/
  rw [hurwitzZetaOdd, Gammaℝ_eq_zero_iff.mpr ⟨n, by rw [neg_mul, sub_add_cancel]⟩, div_zero]
  /-
    🎉 no goals
  -/


/-- The trivial zeroes of the sine zeta function. -/
theorem sinZeta_neg_two_mul_nat_sub_one (a : UnitAddCircle) (n : ℕ) :
    sinZeta a (-2 * n - 1) = 0 := by
  /-
    a : UnitAddCircle
    n : Nat
    ⊢ Eq (HurwitzZeta.sinZeta a (HSub.hSub (HMul.hMul (-2) ↑n) 1)) 0
  -/
  rw [sinZeta, Gammaℝ_eq_zero_iff.mpr ⟨n, by rw [neg_mul, sub_add_cancel]⟩, div_zero]
  /-
    🎉 no goals
  -/


/-- If `s` is not in `-ℕ`, then `hurwitzZetaOdd a (1 - s)` is an explicit multiple of
`sinZeta s`. -/
lemma hurwitzZetaOdd_one_sub (a : UnitAddCircle) {s : ℂ} (hs : ∀ (n : ℕ), s ≠ -n) :
    hurwitzZetaOdd a (1 - s) = 2 * (2 * π) ^ (-s) * Gamma s * sin (π * s / 2) * sinZeta a s := by
  rw [← Gammaℂ, hurwitzZetaOdd, (by ring : 1 - s + 1 = 2 - s), div_eq_mul_inv,
    inv_Gammaℝ_two_sub hs, completedHurwitzZetaOdd_one_sub, sinZeta, ← div_eq_mul_inv,
    ← mul_div_assoc, ← mul_div_assoc, mul_comm]


/-- If `s` is not in `-ℕ`, then `sinZeta a (1 - s)` is an explicit multiple of
`hurwitzZetaOdd s`. -/
lemma sinZeta_one_sub (a : UnitAddCircle) {s : ℂ} (hs : ∀ (n : ℕ), s ≠ -n) :
    sinZeta a (1 - s) = 2 * (2 * π) ^ (-s) * Gamma s * sin (π * s / 2) * hurwitzZetaOdd a s := by
  rw [← Gammaℂ, sinZeta, (by ring : 1 - s + 1 = 2 - s), div_eq_mul_inv, inv_Gammaℝ_two_sub hs,
    completedSinZeta_one_sub, hurwitzZetaOdd, ← div_eq_mul_inv, ← mul_div_assoc, ← mul_div_assoc,
    mul_comm]


