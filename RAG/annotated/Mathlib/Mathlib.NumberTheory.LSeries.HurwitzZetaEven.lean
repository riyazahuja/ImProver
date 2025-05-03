/-- Even Hurwitz zeta kernel (function whose Mellin transform will be the even part of the
completed Hurwit zeta function). See `evenKernel_def` for the defining formula, and
`hasSum_int_evenKernel` for an expression as a sum over `ℤ`. -/
@[irreducible] def evenKernel (a : UnitAddCircle) (x : ℝ) : ℝ :=
  (show Function.Periodic
    (fun ξ : ℝ ↦ rexp (-π * ξ ^ 2 * x) * re (jacobiTheta₂ (ξ * I * x) (I * x))) 1 by
      /-
        a : UnitAddCircle
        x : Real
        ⊢ Function.Periodic (fun ξ => HMul.hMul (Real.exp (HMul.hMul (HMul.hMul (Neg.n …
      -/
      intro ξ
      /-
        a : UnitAddCircle
        x ξ : Real
        ⊢ Eq ((fun ξ => HMul.hMul (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (H …
      -/
      simp only [ofReal_add, ofReal_one, add_mul, one_mul, jacobiTheta₂_add_left']
      have : cexp (-↑π * I * ((I * ↑x) + 2 * (↑ξ * I * ↑x))) = rexp (π * (x + 2 * ξ * x)) := by
        ring_nf
        simp only [I_sq, mul_neg, mul_one, neg_mul, neg_neg, sub_neg_eq_add, ofReal_exp, ofReal_add,
          ofReal_mul, ofReal_ofNat]
      /-
        a : UnitAddCircle
        x ξ : Real
        this : Eq (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) Complex.I) (HA …
        ⊢ Eq (HMul.hMul (Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HPow.hPow ( …
      -/
      rw [this, re_ofReal_mul, ← mul_assoc, ← Real.exp_add]
      /-
        a : UnitAddCircle
        x ξ : Real
        this : Eq (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) Complex.I) (HA …
        ⊢ Eq (HMul.hMul (Real.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg Real.pi) ( …
      -/
      congr
      /-
        case e_a.e_x
        a : UnitAddCircle
        x ξ : Real
        this : Eq (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) Complex.I) (HA …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HPow.hPow (HAdd.hAdd  …
      -/
      ring).lift a
      /-
        🎉 no goals
      -/


lemma evenKernel_def (a x : ℝ) :
    ↑(evenKernel ↑a x) = cexp (-π * a ^ 2 * x) * jacobiTheta₂ (a * I * x) (I * x) := by
  /-
    a x : Real
    ⊢ Eq (↑(HurwitzZeta.evenKernel (↑a) x)) (HMul.hMul (Complex.exp (HMul.hMul (HM …
  -/
  unfold evenKernel
  simp only [neg_mul, Function.Periodic.lift_coe, ofReal_mul, ofReal_exp, ofReal_neg, ofReal_pow,
    re_eq_add_conj, jacobiTheta₂_conj, map_mul, conj_ofReal, conj_I, mul_neg, neg_neg,
    jacobiTheta₂_neg_left, ← mul_two, mul_div_cancel_right₀ _ (two_ne_zero' ℂ)]


/-- For `x ≤ 0` the defining sum diverges, so the kernel is 0. -/
lemma evenKernel_undef (a : UnitAddCircle) {x : ℝ} (hx : x ≤ 0) : evenKernel a x = 0 := by
  /-
    a : UnitAddCircle
    x : Real
    hx : LE.le x 0
    ⊢ Eq (HurwitzZeta.evenKernel a x) 0
  -/
  have : (I * ↑x).im ≤ 0 := by rwa [I_mul_im, ofReal_re]
  /-
    a : UnitAddCircle
    x : Real
    hx : LE.le x 0
    this : LE.le (HMul.hMul Complex.I ↑x).im 0
    ⊢ Eq (HurwitzZeta.evenKernel a x) 0
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  /-
    case H
    x : Real
    hx : LE.le x 0
    this : LE.le (HMul.hMul Complex.I ↑x).im 0
    a' : Real
    ⊢ Eq (HurwitzZeta.evenKernel (↑a') x) 0
  -/
  rw [← ofReal_inj, evenKernel_def, jacobiTheta₂_undef _ this, mul_zero, ofReal_zero]
  /-
    🎉 no goals
  -/


/-- Cosine Hurwitz zeta kernel. See `cosKernel_def` for the defining formula, and
`hasSum_int_cosKernel` for expression as a sum. -/
@[irreducible] def cosKernel (a : UnitAddCircle) (x : ℝ) : ℝ :=
  (show Function.Periodic (fun ξ : ℝ ↦ re (jacobiTheta₂ ξ (I * x))) 1 by
    /-
      a : UnitAddCircle
      x : Real
      ⊢ Function.Periodic (fun ξ => (jacobiTheta₂ (↑ξ) (HMul.hMul Complex.I ↑x)).re) 1
    -/
    intro ξ; simp_rw [ofReal_add, ofReal_one, jacobiTheta₂_add_left]).lift a
             /-
               🎉 no goals
             -/


lemma cosKernel_def (a x : ℝ) : ↑(cosKernel ↑a x) = jacobiTheta₂ a (I * x) := by
  /-
    a x : Real
    ⊢ Eq (↑(HurwitzZeta.cosKernel (↑a) x)) (jacobiTheta₂ (↑a) (HMul.hMul Complex.I …
  -/
  unfold cosKernel
  simp only [Function.Periodic.lift_coe, re_eq_add_conj, jacobiTheta₂_conj, conj_ofReal, map_mul,
    conj_I, neg_mul, neg_neg, ← mul_two, mul_div_cancel_right₀ _ (two_ne_zero' ℂ)]


lemma cosKernel_undef (a : UnitAddCircle) {x : ℝ} (hx : x ≤ 0) : cosKernel a x = 0 := by
  /-
    a : UnitAddCircle
    x : Real
    hx : LE.le x 0
    ⊢ Eq (HurwitzZeta.cosKernel a x) 0
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  /-
    case H
    x : Real
    hx : LE.le x 0
    a' : Real
    ⊢ Eq (HurwitzZeta.cosKernel (↑a') x) 0
  -/
  rw [← ofReal_inj, cosKernel_def, jacobiTheta₂_undef _ (by rwa [I_mul_im, ofReal_re]), ofReal_zero]
  /-
    🎉 no goals
  -/


/-- For `a = 0`, both kernels agree. -/
lemma evenKernel_eq_cosKernel_of_zero : evenKernel 0 = cosKernel 0 := by
  /-
    ⊢ Eq (HurwitzZeta.evenKernel 0) (HurwitzZeta.cosKernel 0)
  -/
  ext1 x
  simp only [← QuotientAddGroup.mk_zero, ← ofReal_inj, evenKernel_def, ofReal_zero, sq, mul_zero,
    zero_mul, Complex.exp_zero, one_mul, cosKernel_def]


lemma evenKernel_neg (a : UnitAddCircle) (x : ℝ) : evenKernel (-a) x = evenKernel a x := by
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.evenKernel (Neg.neg a) x) (HurwitzZeta.evenKernel a x)
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  simp only [← QuotientAddGroup.mk_neg, ← ofReal_inj, evenKernel_def, ofReal_neg, neg_sq, neg_mul,
    jacobiTheta₂_neg_left]


lemma cosKernel_neg (a : UnitAddCircle) (x : ℝ) : cosKernel (-a) x = cosKernel a x := by
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.cosKernel (Neg.neg a) x) (HurwitzZeta.cosKernel a x)
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  simp only [← QuotientAddGroup.mk_neg, ← ofReal_inj, cosKernel_def, ofReal_neg,
    jacobiTheta₂_neg_left]


lemma continuousOn_evenKernel (a : UnitAddCircle) : ContinuousOn (evenKernel a) (Ioi 0) := by
  /-
    a : UnitAddCircle
    ⊢ ContinuousOn (HurwitzZeta.evenKernel a) (Set.Ioi 0)
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (HurwitzZeta.evenKernel ↑a') (Set.Ioi 0)
  -/
  apply continuous_re.comp_continuousOn (f := fun x ↦ (evenKernel a' x : ℂ))
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (fun x => ↑(HurwitzZeta.evenKernel (↑a') x)) (Set.Ioi 0)
  -/
  simp only [evenKernel_def a']
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg …
  -/
  refine continuousOn_of_forall_continuousAt (fun x hx ↦ ((Continuous.continuousAt ?_).mul ?_))
    /-
      case H.refine_1
      a' x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ Continuous fun x => Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) (HP …
    -/
  · exact Complex.continuous_exp.comp (continuous_const.mul continuous_ofReal)
    /-
      🎉 no goals
    -/
    /-
      case H.refine_2
      a' x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ ContinuousAt (fun x => jacobiTheta₂ (HMul.hMul (HMul.hMul (↑a') Complex.I) ↑ …
    -/
  · have h := continuousAt_jacobiTheta₂ (a' * I * x) (?_ : 0 < im (I * x))
      /-
        case H.refine_2.refine_2
        a' x : Real
        hx : Membership.mem (Set.Ioi 0) x
        h : ContinuousAt (fun p => jacobiTheta₂ p.1 p.2) { fst := HMul.hMul (HMul.hMul …
        ⊢ ContinuousAt (fun x => jacobiTheta₂ (HMul.hMul (HMul.hMul (↑a') Complex.I) ↑ …
      -/
    · exact h.comp (f := fun u : ℝ ↦ (a' * I * u, I * u)) (by fun_prop)
      /-
        🎉 no goals
      -/
      /-
        case H.refine_2.refine_1
        a' x : Real
        hx : Membership.mem (Set.Ioi 0) x
        ⊢ LT.lt 0 (HMul.hMul Complex.I ↑x).im
      -/
    · rwa [mul_im, I_re, I_im, zero_mul, one_mul, zero_add, ofReal_re]
      /-
        🎉 no goals
      -/


lemma continuousOn_cosKernel (a : UnitAddCircle) : ContinuousOn (cosKernel a) (Ioi 0) := by
  /-
    a : UnitAddCircle
    ⊢ ContinuousOn (HurwitzZeta.cosKernel a) (Set.Ioi 0)
  -/
  induction' a using QuotientAddGroup.induction_on with a'
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (HurwitzZeta.cosKernel ↑a') (Set.Ioi 0)
  -/
  apply continuous_re.comp_continuousOn (f := fun x ↦ (cosKernel a' x : ℂ))
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (fun x => ↑(HurwitzZeta.cosKernel (↑a') x)) (Set.Ioi 0)
  -/
  simp only [cosKernel_def]
  /-
    case H
    a' : Real
    ⊢ ContinuousOn (fun x => jacobiTheta₂ (↑a') (HMul.hMul Complex.I ↑x)) (Set.Ioi …
  -/
  refine continuousOn_of_forall_continuousAt (fun x hx ↦ ?_)
  /-
    case H
    a' x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ ContinuousAt (fun x => jacobiTheta₂ (↑a') (HMul.hMul Complex.I ↑x)) x
  -/
  have : 0 < im (I * x) := by rwa [mul_im, I_re, I_im, zero_mul, one_mul, zero_add, ofReal_re]
  exact (continuousAt_jacobiTheta₂ a' this).comp
    (g := fun p : ℂ × ℂ ↦ jacobiTheta₂ p.1 p.2)
    (f := fun u : ℝ ↦ ((a' : ℂ), I * u))
    (by fun_prop)


lemma evenKernel_functional_equation (a : UnitAddCircle) (x : ℝ) :
    evenKernel a x = 1 / x ^ (1 / 2 : ℝ) * cosKernel a (1 / x) := by
  /-
    a : UnitAddCircle
    x : Real
    ⊢ Eq (HurwitzZeta.evenKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (1 / 2) …
  -/
  rcases le_or_lt x 0 with hx | hx
    /-
      case inl
      a : UnitAddCircle
      x : Real
      hx : LE.le x 0
      ⊢ Eq (HurwitzZeta.evenKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (1 / 2) …
    -/
  · rw [evenKernel_undef _ hx, cosKernel_undef, mul_zero]
    /-
      case inl.hx
      a : UnitAddCircle
      x : Real
      hx : LE.le x 0
      ⊢ LE.le (HDiv.hDiv 1 x) 0
    -/
    exact div_nonpos_of_nonneg_of_nonpos zero_le_one hx
    /-
      🎉 no goals
    -/
  /-
    case inr
    a : UnitAddCircle
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (HurwitzZeta.evenKernel a x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (1 / 2) …
  -/
  induction' a using QuotientAddGroup.induction_on with a
  /-
    case inr.H
    x : Real
    hx : LT.lt 0 x
    a : Real
    ⊢ Eq (HurwitzZeta.evenKernel (↑a) x) (HMul.hMul (HDiv.hDiv 1 (HPow.hPow x (1 / …
  -/
  rw [← ofReal_inj, ofReal_mul, evenKernel_def, cosKernel_def, jacobiTheta₂_functional_equation]
  have h1 : I * ↑(1 / x) = -1 / (I * x) := by
    push_cast
    rw [← div_div, mul_one_div, div_I, neg_one_mul, neg_neg]
  /-
    case inr.H
    x : Real
    hx : LT.lt 0 x
    a : Real
    h1 : Eq (HMul.hMul Complex.I ↑(HDiv.hDiv 1 x)) (HDiv.hDiv (-1) (HMul.hMul Comp …
    ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) (HPow.hP …
  -/
  have hx' : I * x ≠ 0 := mul_ne_zero I_ne_zero (ofReal_ne_zero.mpr hx.ne')
  have h2 : a * I * x / (I * x) = a := by
    rw [div_eq_iff hx']
    ring
  have h3 : 1 / (-I * (I * x)) ^ (1 / 2 : ℂ) = 1 / ↑(x ^ (1 / 2 : ℝ)) := by
    rw [neg_mul, ← mul_assoc, I_mul_I, neg_one_mul, neg_neg,ofReal_cpow hx.le, ofReal_div,
      ofReal_one, ofReal_ofNat]
  have h4 : -π * I * (a * I * x) ^ 2 / (I * x) = - (-π * a ^ 2 * x) := by
    rw [mul_pow, mul_pow, I_sq, div_eq_iff hx']
    ring
  rw [h1, h2, h3, h4, ← mul_assoc, mul_comm (cexp _), mul_assoc _ (cexp _) (cexp _),
    ← Complex.exp_add, neg_add_cancel, Complex.exp_zero, mul_one, ofReal_div, ofReal_one]


lemma hasSum_int_evenKernel (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℤ ↦ rexp (-π * (n + a) ^ 2 * t)) (evenKernel a t) := by
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HPow.hPow …
  -/
  rw [← hasSum_ofReal, evenKernel_def]
  have (n : ℤ) : ↑(rexp (-π * (↑n + a) ^ 2 * t)) =
      cexp (-↑π * ↑a ^ 2 * ↑t) * jacobiTheta₂_term n (↑a * I * ↑t) (I * ↑t) := by
    rw [jacobiTheta₂_term, ← Complex.exp_add]
    push_cast
    congr
    ring_nf
    simp only [I_sq, mul_neg, neg_mul, mul_one]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : ∀ (n : Int), Eq (↑(Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HP …
    ⊢ HasSum (fun x => ↑(Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HPow.hP …
  -/
  simp only [this]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : ∀ (n : Int), Eq (↑(Real.exp (HMul.hMul (HMul.hMul (Neg.neg Real.pi) (HP …
    ⊢ HasSum (fun x => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real …
  -/
  apply (hasSum_jacobiTheta₂_term _ (by rwa [I_mul_im, ofReal_re])).mul_left
  /-
    🎉 no goals
  -/


lemma hasSum_int_cosKernel (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℤ ↦ cexp (2 * π * I * a * n) * rexp (-π * n ^ 2 * t)) ↑(cosKernel a t) := by
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HM …
  -/
  rw [cosKernel_def a t]
  have (n : ℤ) : cexp (2 * ↑π * I * ↑a * ↑n) * ↑(rexp (-π * ↑n ^ 2 * t)) =
      jacobiTheta₂_term n (↑a) (I * ↑t) := by
    rw [jacobiTheta₂_term, ofReal_exp, ← Complex.exp_add]
    push_cast
    ring_nf
    simp only [I_sq, mul_neg, neg_mul, mul_one, sub_eq_add_neg]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : ∀ (n : Int), Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMu …
    ⊢ HasSum (fun n => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HM …
  -/
  simp only [this]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : ∀ (n : Int), Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMu …
    ⊢ HasSum (fun n => jacobiTheta₂_term n (↑a) (HMul.hMul Complex.I ↑t)) (jacobiT …
  -/
  exact hasSum_jacobiTheta₂_term _ (by rwa [I_mul_im, ofReal_re])
  /-
    🎉 no goals
  -/


/-- Modified version of `hasSum_int_evenKernel` omitting the constant term at `∞`. -/
lemma hasSum_int_evenKernel₀ (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℤ ↦ if n + a = 0 then 0 else rexp (-π * (n + a) ^ 2 * t))
    (evenKernel a t - if (a : UnitAddCircle) = 0 then 1 else 0) := by
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0) 0 (Real.exp (HMul.hMul (HMul. …
  -/
  haveI := Classical.propDecidable -- speed up instance search for `if / then / else`
  /-
    a t : Real
    ht : LT.lt 0 t
    this : (a : Prop) → Decidable a
    ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0) 0 (Real.exp (HMul.hMul (HMul. …
  -/
  simp_rw [AddCircle.coe_eq_zero_iff, zsmul_one]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : (a : Prop) → Decidable a
    ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0) 0 (Real.exp (HMul.hMul (HMul. …
  -/
  split_ifs with h
    /-
      case pos
      a t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      h : Exists fun n => Eq (↑n) a
      ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0) 0 (Real.exp (HMul.hMul (HMul. …
    -/
  · obtain ⟨k, rfl⟩ := h
    /-
      case pos.intro
      t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      k : Int
      ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd ↑n ↑k) 0) 0 (Real.exp (HMul.hMul (HMul.h …
    -/
    simp_rw [← Int.cast_add, Int.cast_eq_zero, add_eq_zero_iff_eq_neg]
    simpa only [Int.cast_add, neg_mul, Int.cast_neg, neg_add_cancel, ne_eq, OfNat.ofNat_ne_zero,
      not_false_eq_true, zero_pow, mul_zero, zero_mul, Real.exp_zero]
      using hasSum_ite_sub_hasSum (hasSum_int_evenKernel (k : ℝ) ht) (-k)
    /-
      case neg
      a t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      h : Not (Exists fun n => Eq (↑n) a)
      ⊢ HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0) 0 (Real.exp (HMul.hMul (HMul. …
    -/
  · suffices ∀ (n : ℤ), n + a ≠ 0 by simpa [this] using hasSum_int_evenKernel a ht
    /-
      case neg
      a t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      h : Not (Exists fun n => Eq (↑n) a)
      ⊢ ∀ (n : Int), Ne (HAdd.hAdd (↑n) a) 0
    -/
    contrapose! h
    /-
      case neg
      a t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      h : Exists fun n => Eq (HAdd.hAdd (↑n) a) 0
      ⊢ Exists fun n => Eq (↑n) a
    -/
    let ⟨n, hn⟩ := h
    /-
      case neg
      a t : Real
      ht : LT.lt 0 t
      this : (a : Prop) → Decidable a
      h : Exists fun n => Eq (HAdd.hAdd (↑n) a) 0
      n : Int
      hn : Eq (HAdd.hAdd (↑n) a) 0
      ⊢ Exists fun n => Eq (↑n) a
    -/
    exact ⟨-n, by rwa [Int.cast_neg, neg_eq_iff_add_eq_zero]⟩
    /-
      🎉 no goals
    -/


lemma hasSum_int_cosKernel₀ (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℤ ↦ if n = 0 then 0 else cexp (2 * π * I * a * n) * rexp (-π * n ^ 2 * t))
    (↑(cosKernel a t) - 1) := by
  simpa? using hasSum_ite_sub_hasSum (hasSum_int_cosKernel a ht) 0
  says simpa only [neg_mul, ofReal_exp, ofReal_neg, ofReal_mul, ofReal_pow, ofReal_intCast,
    Int.cast_zero, mul_zero, Complex.exp_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, zero_mul, Real.exp_zero, ofReal_one, mul_one] using
    hasSum_ite_sub_hasSum (hasSum_int_cosKernel a ht) 0


lemma hasSum_nat_cosKernel₀ (a : ℝ) {t : ℝ} (ht : 0 < t) :
    HasSum (fun n : ℕ ↦ 2 * Real.cos (2 * π * a * (n + 1)) * rexp (-π * (n + 1) ^ 2 * t))
    (cosKernel a t - 1) := by
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun n => HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul (HMul.hMul (HMu …
  -/
  rw [← hasSum_ofReal, ofReal_sub, ofReal_one]
  /-
    a t : Real
    ht : LT.lt 0 t
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul (HMul.hMul (H …
  -/
  have := (hasSum_int_cosKernel a ht).nat_add_neg
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HAdd.hAdd (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMu …
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul (HMul.hMul (H …
  -/
  rw [← hasSum_nat_add_iff' 1] at this
  simp_rw [Finset.sum_range_one, Nat.cast_zero, neg_zero, Int.cast_zero, zero_pow two_ne_zero,
    mul_zero, zero_mul, Complex.exp_zero, Real.exp_zero, ofReal_one, mul_one, Int.cast_neg,
    Int.cast_natCast, neg_sq, ← add_mul, add_sub_assoc, ← sub_sub, sub_self, zero_sub,
    ← sub_eq_add_neg, mul_neg] at this
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HMul.hMul (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMu …
    ⊢ HasSum (fun x => ↑(HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul (HMul.hMul (H …
  -/
  refine this.congr_fun fun n ↦ ?_
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HMul.hMul (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMu …
    n : Nat
    ⊢ Eq (↑(HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 Re …
  -/
  push_cast
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HMul.hMul (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMu …
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Complex.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑ …
  -/
  rw [Complex.cos, mul_div_cancel₀ _ two_ne_zero]
  /-
    a t : Real
    ht : LT.lt 0 t
    this : HasSum (fun n => HMul.hMul (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMu …
    n : Nat
    ⊢ Eq (HMul.hMul (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
  -/
              /-
                🎉 no goals
              -/
  congr 3 <;> ring
              /-
                🎉 no goals
              -/


/-- The function `evenKernel a - L` has exponential decay at `+∞`, where `L = 1` if
`a = 0` and `L = 0` otherwise. -/
lemma isBigO_atTop_evenKernel_sub (a : UnitAddCircle) : ∃ p : ℝ, 0 < p ∧
    (evenKernel a · - (if a = 0 then 1 else 0)) =O[atTop] (rexp <| -p * ·) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  induction' a using QuotientAddGroup.induction_on with b
  /-
    case H
    b : Real
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  obtain ⟨p, hp, hp'⟩ := HurwitzKernelBounds.isBigO_atTop_F_int_zero_sub b
  /-
    case H.intro.intro
    b p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  refine ⟨p, hp, (EventuallyEq.isBigO ?_).trans hp'⟩
  /-
    case H.intro.intro
    b p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (HurwitzZeta.evenKernel (↑b) x …
  -/
  filter_upwards [eventually_gt_atTop 0] with t ht
  simp only [← (hasSum_int_evenKernel b ht).tsum_eq, HurwitzKernelBounds.F_int,
    HurwitzKernelBounds.f_int, pow_zero, one_mul, Function.Periodic.lift_coe]


/-- The function `cosKernel a - 1` has exponential decay at `+∞`, for any `a`. -/
lemma isBigO_atTop_cosKernel_sub (a : UnitAddCircle) :
    ∃ p, 0 < p ∧ IsBigO atTop (cosKernel a · - 1) (fun x ↦ Real.exp (-p * x)) := by
  /-
    a : UnitAddCircle
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  induction' a using QuotientAddGroup.induction_on with a
  /-
    case H
    a : Real
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  obtain ⟨p, hp, hp'⟩ := HurwitzKernelBounds.isBigO_atTop_F_nat_zero_sub zero_le_one
  /-
    case H.intro.intro
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Exists fun p => And (LT.lt 0 p) (Asymptotics.IsBigO Filter.atTop (fun x => H …
  -/
  refine ⟨p, hp, (Eventually.isBigO ?_).trans (hp'.const_mul_left 2)⟩
  /-
    case H.intro.intro
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HurwitzZeta.cosKern …
  -/
  simp only [eq_false_intro one_ne_zero, if_false, sub_zero]
  /-
    case H.intro.intro
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HurwitzZeta.cosKern …
  -/
  filter_upwards [eventually_gt_atTop 0] with t ht
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (HSub.hSub (HurwitzZeta.cosKernel (↑a) t) 1)) (HMul.hMul 2  …
  -/
  rw [← (hasSum_nat_cosKernel₀ a ht).tsum_eq, HurwitzKernelBounds.F_nat]
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    t : Real
    ht : LT.lt 0 t
    ⊢ LE.le (Norm.norm (tsum fun b => HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul  …
  -/
  apply tsum_of_norm_bounded ((HurwitzKernelBounds.summable_f_nat 0 1 ht).hasSum.mul_left 2)
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    t : Real
    ht : LT.lt 0 t
    ⊢ ∀ (i : Nat), LE.le (Norm.norm (HMul.hMul (HMul.hMul 2 (Real.cos (HMul.hMul ( …
  -/
  intro n
  rw [norm_mul, norm_mul, norm_two, mul_assoc, mul_le_mul_iff_of_pos_left two_pos,
    norm_of_nonneg (exp_pos _).le, HurwitzKernelBounds.f_nat, pow_zero, one_mul, Real.norm_eq_abs]
  /-
    case h
    a p : Real
    hp : LT.lt 0 p
    hp' : Asymptotics.IsBigO Filter.atTop (fun t => HSub.hSub (HurwitzKernelBounds …
    t : Real
    ht : LT.lt 0 t
    n : Nat
    ⊢ LE.le (HMul.hMul (abs (Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi)  …
  -/
  exact mul_le_of_le_one_left (exp_pos _).le (abs_cos_le_one _)
  /-
    🎉 no goals
  -/


/-- A `WeakFEPair` structure with `f = evenKernel a` and `g = cosKernel a`. -/
def hurwitzEvenFEPair (a : UnitAddCircle) : WeakFEPair ℂ where
  f := ofReal ∘ evenKernel a
  g := ofReal ∘ cosKernel a
  hf_int := (continuous_ofReal.comp_continuousOn (continuousOn_evenKernel a)).locallyIntegrableOn
    measurableSet_Ioi
  hg_int := (continuous_ofReal.comp_continuousOn (continuousOn_cosKernel a)).locallyIntegrableOn
    measurableSet_Ioi
  hk := one_half_pos
  hε := one_ne_zero
  f₀ := if a = 0 then 1 else 0
  hf_top r := by
    /-
      a : UnitAddCircle
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    let ⟨v, hv, hv'⟩ := isBigO_atTop_evenKernel_sub a
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (HurwitzZeta.evenKer …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    rw [← isBigO_norm_left] at hv' ⊢
    conv at hv' =>
      enter [2, x]; rw [← norm_real, ofReal_sub, apply_ite ((↑) : ℝ → ℂ), ofReal_one, ofReal_zero]
    /-
      a : UnitAddCircle
      r v : Real
      hv : LT.lt 0 v
      hv' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (↑(Hurwit …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (Function.com …
    -/
    exact hv'.trans (isLittleO_exp_neg_mul_rpow_atTop hv _).isBigO
    /-
      🎉 no goals
    -/
  g₀ := 1
  hg_top r := by
    /-
      a : UnitAddCircle
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    obtain ⟨p, hp, hp'⟩ := isBigO_atTop_cosKernel_sub a
    /-
      case intro.intro
      a : UnitAddCircle
      r p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (HurwitzZeta.cosKern …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (Function.comp Complex.o …
    -/
    rw [← isBigO_norm_left] at hp' ⊢
    have (x : ℝ) : ‖(ofReal ∘ cosKernel a) x - 1‖ = ‖cosKernel a x - 1‖ := by
      rw [← norm_real, ofReal_sub, ofReal_one, Function.comp_apply]
    /-
      case intro.intro
      a : UnitAddCircle
      r p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (HurwitzZ …
      this : ∀ (x : Real), Eq (Norm.norm (HSub.hSub (Function.comp Complex.ofReal (H …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (Function.com …
    -/
    simp only [this]
    /-
      case intro.intro
      a : UnitAddCircle
      r p : Real
      hp : LT.lt 0 p
      hp' : Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (HurwitzZ …
      this : ∀ (x : Real), Eq (Norm.norm (HSub.hSub (Function.comp Complex.ofReal (H …
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => Norm.norm (HSub.hSub (HurwitzZeta. …
    -/
    exact hp'.trans (isLittleO_exp_neg_mul_rpow_atTop hp _).isBigO
    /-
      🎉 no goals
    -/
  h_feq x hx := by
    simp_rw [Function.comp_apply, one_mul, smul_eq_mul, ← ofReal_mul,
      evenKernel_functional_equation, one_div x, one_div x⁻¹, inv_rpow (le_of_lt hx),
      one_div, inv_inv]


lemma hurwitzEvenFEPair_zero_symm :
    (hurwitzEvenFEPair 0).symm = hurwitzEvenFEPair 0 := by
  /-
    ⊢ Eq (HurwitzZeta.hurwitzEvenFEPair 0).symm (HurwitzZeta.hurwitzEvenFEPair 0)
  -/
  unfold hurwitzEvenFEPair WeakFEPair.symm
  /-
    ⊢ Eq { f := { f := Function.comp Complex.ofReal (HurwitzZeta.evenKernel 0), g  …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  congr 1 <;> simp only [evenKernel_eq_cosKernel_of_zero, inv_one, if_true]
              /-
                🎉 no goals
              -/


lemma hurwitzEvenFEPair_neg (a : UnitAddCircle) : hurwitzEvenFEPair (-a) = hurwitzEvenFEPair a := by
  /-
    a : UnitAddCircle
    ⊢ Eq (HurwitzZeta.hurwitzEvenFEPair (Neg.neg a)) (HurwitzZeta.hurwitzEvenFEPai …
  -/
  unfold hurwitzEvenFEPair
  /-
    a : UnitAddCircle
    ⊢ Eq { f := Function.comp Complex.ofReal (HurwitzZeta.evenKernel (Neg.neg a)), …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  congr 1 <;> simp only [Function.comp_def, evenKernel_neg, cosKernel_neg, neg_eq_zero]
              /-
                🎉 no goals
              -/


/--
The meromorphic function of `s` which agrees with
`1 / 2 * Gamma (s / 2) * π ^ (-s / 2) * ∑' (n : ℤ), 1 / |n + a| ^ s` for `1 < re s`.
-/
def completedHurwitzZetaEven (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzEvenFEPair a).Λ (s / 2)) / 2


/-- The entire function differing from `completedHurwitzZetaEven a s` by a linear combination of
`1 / s` and `1 / (1 - s)`. -/
def completedHurwitzZetaEven₀ (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzEvenFEPair a).Λ₀ (s / 2)) / 2


lemma completedHurwitzZetaEven_eq (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaEven a s =
    completedHurwitzZetaEven₀ a s - (if a = 0 then 1 else 0) / s - 1 / (1 - s) := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedHurwitzZetaEven a s) (HSub.hSub (HSub.hSub (Hurwitz …
  -/
  rw [completedHurwitzZetaEven, WeakFEPair.Λ, sub_div, sub_div]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HSub.hSub (HSub.hSub (HDiv.hDiv ((HurwitzZeta.hurwitzEvenFEPair a).Λ₀ (H …
  -/
  congr 1
  · change completedHurwitzZetaEven₀ a s - (1 / (s / 2)) • (if a = 0 then 1 else 0) / 2 =
      completedHurwitzZetaEven₀ a s - (if a = 0 then 1 else 0) / s
    /-
      case e_a
      a : UnitAddCircle
      s : Complex
      ⊢ Eq (HSub.hSub (HurwitzZeta.completedHurwitzZetaEven₀ a s) (HDiv.hDiv (HSMul. …
    -/
    rw [smul_eq_mul, mul_comm, mul_div_assoc, div_div, div_mul_cancel₀ _ two_ne_zero, mul_one_div]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      a : UnitAddCircle
      s : Complex
      ⊢ Eq (HDiv.hDiv (HSMul.hSMul (HDiv.hDiv (HurwitzZeta.hurwitzEvenFEPair a).ε (H …
    -/
  · change (1 / (↑(1 / 2 : ℝ) - s / 2)) • 1 / 2 = 1 / (1 - s)
    /-
      case e_a
      a : UnitAddCircle
      s : Complex
      ⊢ Eq (HDiv.hDiv (HSMul.hSMul (HDiv.hDiv 1 (HSub.hSub (↑(1 / 2)) (HDiv.hDiv s 2 …
    -/
    push_cast
    /-
      case e_a
      a : UnitAddCircle
      s : Complex
      ⊢ Eq (HDiv.hDiv (HSMul.hSMul (HDiv.hDiv 1 (HSub.hSub (HDiv.hDiv 1 2) (HDiv.hDi …
    -/
    rw [smul_eq_mul, mul_one, ← sub_div, div_div, div_mul_cancel₀ _ two_ne_zero]
    /-
      🎉 no goals
    -/


/--
The meromorphic function of `s` which agrees with
`Gamma (s / 2) * π ^ (-s / 2) * ∑' n : ℕ, cos (2 * π * a * n) / n ^ s` for `1 < re s`.
-/
def completedCosZeta (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzEvenFEPair a).symm.Λ (s / 2)) / 2


/-- The entire function differing from `completedCosZeta a s` by a linear combination of
`1 / s` and `1 / (1 - s)`. -/
def completedCosZeta₀ (a : UnitAddCircle) (s : ℂ) : ℂ :=
  ((hurwitzEvenFEPair a).symm.Λ₀ (s / 2)) / 2


lemma completedCosZeta_eq (a : UnitAddCircle) (s : ℂ) :
    completedCosZeta a s =
    completedCosZeta₀ a s - 1 / s - (if a = 0 then 1 else 0) / (1 - s) := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta a s) (HSub.hSub (HSub.hSub (HurwitzZeta.com …
  -/
  rw [completedCosZeta, WeakFEPair.Λ, sub_div, sub_div]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HSub.hSub (HSub.hSub (HDiv.hDiv ((HurwitzZeta.hurwitzEvenFEPair a).symm. …
  -/
  congr 1
  · rw [completedCosZeta₀, WeakFEPair.symm, hurwitzEvenFEPair, smul_eq_mul, mul_one, div_div,
      div_mul_cancel₀ _ (two_ne_zero' ℂ)]
  · simp_rw [WeakFEPair.symm, hurwitzEvenFEPair, push_cast, inv_one, smul_eq_mul,
      mul_comm _ (if _ then _ else _), mul_div_assoc, div_div, ← sub_div,
      div_mul_cancel₀ _ (two_ne_zero' ℂ), mul_one_div]


lemma completedHurwitzZetaEven_neg (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaEven (-a) s = completedHurwitzZetaEven a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedHurwitzZetaEven (Neg.neg a) s) (HurwitzZeta.complet …
  -/
  simp only [completedHurwitzZetaEven, hurwitzEvenFEPair_neg]
  /-
    🎉 no goals
  -/


lemma completedHurwitzZetaEven₀_neg (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaEven₀ (-a) s = completedHurwitzZetaEven₀ a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedHurwitzZetaEven₀ (Neg.neg a) s) (HurwitzZeta.comple …
  -/
  simp only [completedHurwitzZetaEven₀, hurwitzEvenFEPair_neg]
  /-
    🎉 no goals
  -/


lemma completedCosZeta_neg (a : UnitAddCircle) (s : ℂ) :
    completedCosZeta (-a) s = completedCosZeta a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta (Neg.neg a) s) (HurwitzZeta.completedCosZet …
  -/
  simp only [completedCosZeta, hurwitzEvenFEPair_neg]
  /-
    🎉 no goals
  -/


lemma completedCosZeta₀_neg (a : UnitAddCircle) (s : ℂ) :
    completedCosZeta₀ (-a) s = completedCosZeta₀ a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta₀ (Neg.neg a) s) (HurwitzZeta.completedCosZe …
  -/
  simp only [completedCosZeta₀, hurwitzEvenFEPair_neg]
  /-
    🎉 no goals
  -/


/-- Functional equation for the even Hurwitz zeta function. -/
lemma completedHurwitzZetaEven_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaEven a (1 - s) = completedCosZeta a s := by
  rw [completedHurwitzZetaEven, completedCosZeta, sub_div,
    (by norm_num : (1 / 2 : ℂ) = ↑(1 / 2 : ℝ)),
    (by rfl : (1 / 2 : ℝ) = (hurwitzEvenFEPair a).k),
    (hurwitzEvenFEPair a).functional_equation (s / 2),
    (by rfl : (hurwitzEvenFEPair a).ε = 1),
    one_smul]


/-- Functional equation for the even Hurwitz zeta function with poles removed. -/
lemma completedHurwitzZetaEven₀_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedHurwitzZetaEven₀ a (1 - s) = completedCosZeta₀ a s := by
  rw [completedHurwitzZetaEven₀, completedCosZeta₀, sub_div,
    (by norm_num : (1 / 2 : ℂ) = ↑(1 / 2 : ℝ)),
    (by rfl : (1 / 2 : ℝ) = (hurwitzEvenFEPair a).k),
    (hurwitzEvenFEPair a).functional_equation₀ (s / 2),
    (by rfl : (hurwitzEvenFEPair a).ε = 1),
    one_smul]


/-- Functional equation for the even Hurwitz zeta function (alternative form). -/
lemma completedCosZeta_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedCosZeta a (1 - s) = completedHurwitzZetaEven a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta a (HSub.hSub 1 s)) (HurwitzZeta.completedHu …
  -/
  rw [← completedHurwitzZetaEven_one_sub, sub_sub_cancel]
  /-
    🎉 no goals
  -/


/-- Functional equation for the even Hurwitz zeta function with poles removed (alternative form). -/
lemma completedCosZeta₀_one_sub (a : UnitAddCircle) (s : ℂ) :
    completedCosZeta₀ a (1 - s) = completedHurwitzZetaEven₀ a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.completedCosZeta₀ a (HSub.hSub 1 s)) (HurwitzZeta.completedH …
  -/
  rw [← completedHurwitzZetaEven₀_one_sub, sub_sub_cancel]
  /-
    🎉 no goals
  -/


/--
The even Hurwitz completed zeta is differentiable away from `s = 0` and `s = 1` (and also at
`s = 0` if `a ≠ 0`)
-/
lemma differentiableAt_completedHurwitzZetaEven
    (a : UnitAddCircle) {s : ℂ} (hs : s ≠ 0 ∨ a ≠ 0) (hs' : s ≠ 1) :
    DifferentiableAt ℂ (completedHurwitzZetaEven a) s := by
  refine (((hurwitzEvenFEPair a).differentiableAt_Λ ?_ (Or.inl ?_)).comp s
      (differentiableAt_id.div_const _)).div_const _
    /-
      case refine_1
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 0) (Ne a 0)
      hs' : Ne s 1
      ⊢ Or (Ne (HDiv.hDiv s 2) 0) (Eq (HurwitzZeta.hurwitzEvenFEPair a).f₀ 0)
    -/
  · simp only [ne_eq, div_eq_zero_iff, OfNat.ofNat_ne_zero, or_false]
    /-
      case refine_1
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 0) (Ne a 0)
      hs' : Ne s 1
      ⊢ Or (Not (Eq s 0)) (Eq (HurwitzZeta.hurwitzEvenFEPair a).f₀ 0)
    -/
    rcases hs with h | h
      /-
        case refine_1.inl
        a : UnitAddCircle
        s : Complex
        hs' : Ne s 1
        h : Ne s 0
        ⊢ Or (Not (Eq s 0)) (Eq (HurwitzZeta.hurwitzEvenFEPair a).f₀ 0)
      -/
    · exact Or.inl h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        a : UnitAddCircle
        s : Complex
        hs' : Ne s 1
        h : Ne a 0
        ⊢ Or (Not (Eq s 0)) (Eq (HurwitzZeta.hurwitzEvenFEPair a).f₀ 0)
      -/
    · simp only [hurwitzEvenFEPair, one_div, h, ↓reduceIte, or_true]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 0) (Ne a 0)
      hs' : Ne s 1
      ⊢ Ne (HDiv.hDiv s 2) ↑(HurwitzZeta.hurwitzEvenFEPair a).k
    -/
  · change s / 2 ≠ ↑(1 / 2 : ℝ)
    /-
      case refine_2
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 0) (Ne a 0)
      hs' : Ne s 1
      ⊢ Ne (HDiv.hDiv s 2) ↑(1 / 2)
    -/
    rw [ofReal_div, ofReal_one, ofReal_ofNat]
    /-
      case refine_2
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 0) (Ne a 0)
      hs' : Ne s 1
      ⊢ Ne (HDiv.hDiv s 2) (HDiv.hDiv 1 2)
    -/
    exact hs' ∘ (div_left_inj' two_ne_zero).mp
    /-
      🎉 no goals
    -/


lemma differentiable_completedHurwitzZetaEven₀ (a : UnitAddCircle) :
    Differentiable ℂ (completedHurwitzZetaEven₀ a) :=
  ((hurwitzEvenFEPair a).differentiable_Λ₀.comp (differentiable_id.div_const _)).div_const _


/-- The difference of two completed even Hurwitz zeta functions is differentiable at `s = 1`. -/
lemma differentiableAt_one_completedHurwitzZetaEven_sub_completedHurwitzZetaEven
    (a b : UnitAddCircle) :
    DifferentiableAt ℂ (fun s ↦ completedHurwitzZetaEven a s - completedHurwitzZetaEven b s) 1 := by
  have (s) : completedHurwitzZetaEven a s - completedHurwitzZetaEven b s =
      completedHurwitzZetaEven₀ a s - completedHurwitzZetaEven₀ b s -
      ((if a = 0 then 1 else 0) - (if b = 0 then 1 else 0)) / s := by
    simp_rw [completedHurwitzZetaEven_eq, sub_div]
    abel
  /-
    a b : UnitAddCircle
    this : ∀ (s : Complex), Eq (HSub.hSub (HurwitzZeta.completedHurwitzZetaEven a  …
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.completedHurwitzZe …
  -/
  rw [funext this]
  /-
    a b : UnitAddCircle
    this : ∀ (s : Complex), Eq (HSub.hSub (HurwitzZeta.completedHurwitzZetaEven a  …
    ⊢ DifferentiableAt Complex (fun x => HSub.hSub (HSub.hSub (HurwitzZeta.complet …
  -/
  refine .sub ?_ <| (differentiable_const _ _).div (differentiable_id _) one_ne_zero
  /-
    a b : UnitAddCircle
    this : ∀ (s : Complex), Eq (HSub.hSub (HurwitzZeta.completedHurwitzZetaEven a  …
    ⊢ DifferentiableAt Complex (fun x => HSub.hSub (HurwitzZeta.completedHurwitzZe …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  apply DifferentiableAt.sub <;> apply differentiable_completedHurwitzZetaEven₀
                                 /-
                                   🎉 no goals
                                 -/


lemma differentiableAt_completedCosZeta
    (a : UnitAddCircle) {s : ℂ} (hs : s ≠ 0) (hs' : s ≠ 1 ∨ a ≠ 0) :
    DifferentiableAt ℂ (completedCosZeta a) s := by
  refine (((hurwitzEvenFEPair a).symm.differentiableAt_Λ (Or.inl ?_) ?_).comp s
      (differentiableAt_id.div_const _)).div_const _
    /-
      case refine_1
      a : UnitAddCircle
      s : Complex
      hs : Ne s 0
      hs' : Or (Ne s 1) (Ne a 0)
      ⊢ Ne (HDiv.hDiv s 2) 0
    -/
  · exact div_ne_zero_iff.mpr ⟨hs, two_ne_zero⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a : UnitAddCircle
      s : Complex
      hs : Ne s 0
      hs' : Or (Ne s 1) (Ne a 0)
      ⊢ Or (Ne (HDiv.hDiv s 2) ↑(HurwitzZeta.hurwitzEvenFEPair a).symm.k) (Eq (Hurwi …
    -/
  · change s / 2 ≠ ↑(1 / 2 : ℝ) ∨ (if a = 0 then 1 else 0) = 0
    /-
      case refine_2
      a : UnitAddCircle
      s : Complex
      hs : Ne s 0
      hs' : Or (Ne s 1) (Ne a 0)
      ⊢ Or (Ne (HDiv.hDiv s 2) ↑(1 / 2)) (Eq (ite (Eq a 0) 1 0) 0)
    -/
    refine Or.imp (fun h ↦ ?_) (fun ha ↦ ?_) hs'
      /-
        case refine_2.refine_1
        a : UnitAddCircle
        s : Complex
        hs : Ne s 0
        hs' : Or (Ne s 1) (Ne a 0)
        h : Ne s 1
        ⊢ Ne (HDiv.hDiv s 2) ↑(1 / 2)
      -/
    · simpa only [push_cast] using h ∘ (div_left_inj' two_ne_zero).mp
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        a : UnitAddCircle
        s : Complex
        hs : Ne s 0
        hs' : Or (Ne s 1) (Ne a 0)
        ha : Ne a 0
        ⊢ Eq (ite (Eq a 0) 1 0) 0
      -/
    · simp_rw [eq_false_intro ha, if_false]
      /-
        🎉 no goals
      -/


lemma differentiable_completedCosZeta₀ (a : UnitAddCircle) :
    Differentiable ℂ (completedCosZeta₀ a) :=
  ((hurwitzEvenFEPair a).symm.differentiable_Λ₀.comp (differentiable_id.div_const _)).div_const _


private lemma tendsto_div_two_punctured_nhds (a : ℂ) :
    Tendsto (fun s : ℂ ↦ s / 2) (𝓝[≠] a) (𝓝[≠] (a / 2)) :=
  le_of_eq ((Homeomorph.mulRight₀ _ (inv_ne_zero (two_ne_zero' ℂ))).map_punctured_nhds_eq a)


/-- The residue of `completedHurwitzZetaEven a s` at `s = 1` is equal to `1`. -/
lemma completedHurwitzZetaEven_residue_one (a : UnitAddCircle) :
    Tendsto (fun s ↦ (s - 1) * completedHurwitzZetaEven a s) (𝓝[≠] 1) (𝓝 1) := by
  have h1 : Tendsto (fun s : ℂ ↦ (s - ↑(1  / 2 : ℝ)) * _) (𝓝[≠] ↑(1  / 2 : ℝ))
    (𝓝 ((1 : ℂ) * (1 : ℂ))) := (hurwitzEvenFEPair a).Λ_residue_k
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s ↑(1 / 2)) ((HurwitzZeta.h …
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HurwitzZeta.completedHur …
  -/
  simp only [push_cast, one_mul] at h1
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s (HDiv.hDiv 1 2)) ((Hurwit …
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HurwitzZeta.completedHur …
  -/
  refine (h1.comp <| tendsto_div_two_punctured_nhds 1).congr (fun s ↦ ?_)
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s (HDiv.hDiv 1 2)) ((Hurwit …
    s : Complex
    ⊢ Eq (Function.comp (fun s => HMul.hMul (HSub.hSub s (HDiv.hDiv 1 2)) ((Hurwit …
  -/
  rw [completedHurwitzZetaEven, Function.comp_apply, ← sub_div, div_mul_eq_mul_div, mul_div_assoc]
  /-
    🎉 no goals
  -/


/-- The residue of `completedHurwitzZetaEven a s` at `s = 0` is equal to `-1` if `a = 0`, and `0`
otherwise. -/
lemma completedHurwitzZetaEven_residue_zero (a : UnitAddCircle) :
    Tendsto (fun s ↦ s * completedHurwitzZetaEven a s) (𝓝[≠] 0) (𝓝 (if a = 0 then -1 else 0)) := by
  have h1 : Tendsto (fun s : ℂ ↦ s * _) (𝓝[≠] 0)
    (𝓝 (-(if a = 0 then 1 else 0))) := (hurwitzEvenFEPair a).Λ_residue_zero
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).Λ …
    ⊢ Filter.Tendsto (fun s => HMul.hMul s (HurwitzZeta.completedHurwitzZetaEven a …
  -/
  have : -(if a = 0 then (1 : ℂ) else 0) = (if a = 0 then -1 else 0) := by { split_ifs <;> simp }
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).Λ …
    this : Eq (Neg.neg (ite (Eq a 0) 1 0)) (ite (Eq a 0) (-1) 0)
    ⊢ Filter.Tendsto (fun s => HMul.hMul s (HurwitzZeta.completedHurwitzZetaEven a …
  -/
  simp only [this, push_cast, one_mul] at h1
  /-
    a : UnitAddCircle
    this : Eq (Neg.neg (ite (Eq a 0) 1 0)) (ite (Eq a 0) (-1) 0)
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).Λ …
    ⊢ Filter.Tendsto (fun s => HMul.hMul s (HurwitzZeta.completedHurwitzZetaEven a …
  -/
  refine (h1.comp <| zero_div (2 : ℂ) ▸ (tendsto_div_two_punctured_nhds 0)).congr (fun s ↦ ?_)
  /-
    a : UnitAddCircle
    this : Eq (Neg.neg (ite (Eq a 0) 1 0)) (ite (Eq a 0) (-1) 0)
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).Λ …
    s : Complex
    ⊢ Eq (Function.comp (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).Λ …
  -/
  rw [completedHurwitzZetaEven, Function.comp_apply, div_mul_eq_mul_div, mul_div_assoc]
  /-
    🎉 no goals
  -/


lemma completedCosZeta_residue_zero (a : UnitAddCircle) :
    Tendsto (fun s ↦ s * completedCosZeta a s) (𝓝[≠] 0) (𝓝 (-1)) := by
  have h1 : Tendsto (fun s : ℂ ↦ s * _) (𝓝[≠] 0)
    (𝓝 (-1)) := (hurwitzEvenFEPair a).symm.Λ_residue_zero
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).s …
    ⊢ Filter.Tendsto (fun s => HMul.hMul s (HurwitzZeta.completedCosZeta a s)) (nh …
  -/
  refine (h1.comp <| zero_div (2 : ℂ) ▸ (tendsto_div_two_punctured_nhds 0)).congr (fun s ↦ ?_)
  /-
    a : UnitAddCircle
    h1 : Filter.Tendsto (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).s …
    s : Complex
    ⊢ Eq (Function.comp (fun s => HMul.hMul s ((HurwitzZeta.hurwitzEvenFEPair a).s …
  -/
  rw [completedCosZeta, Function.comp_apply, div_mul_eq_mul_div, mul_div_assoc]
  /-
    🎉 no goals
  -/


/-- Formula for `completedCosZeta` as a Dirichlet series in the convergence range
(first version, with sum over `ℤ`). -/
lemma hasSum_int_completedCosZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ Gammaℝ s * cexp (2 * π * I * a * n) / (↑|n| : ℂ) ^ s / 2)
    (completedCosZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (Complex.exp (HMul …
  -/
  let c (n : ℤ) : ℂ := cexp (2 * π * I * a * n) / 2
  have hF t (ht : 0 < t) : HasSum (fun n : ℤ ↦ if n = 0 then 0 else c n * rexp (-π * n ^ 2 * t))
      ((cosKernel a t - 1) / 2) := by
    refine ((hasSum_int_cosKernel₀ a ht).div_const 2).congr_fun fun n ↦ ?_
    split_ifs <;> simp only [zero_div, c, div_mul_eq_mul_div]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    c : Int → Complex := fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq n 0) 0 (HMul.hMul (c n …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (Complex.exp (HMul …
  -/
  simp only [← Int.cast_eq_zero (α := ℝ)] at hF
  rw [show completedCosZeta a s = mellin (fun t ↦ (cosKernel a t - 1 : ℂ) / 2) (s / 2) by
    rw [mellin_div_const, completedCosZeta]
    congr 1
    refine ((hurwitzEvenFEPair a).symm.hasMellin (?_ : 1 / 2 < (s / 2).re)).2.symm
    rwa [div_ofNat_re, div_lt_div_iff_of_pos_right two_pos]]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    c : Int → Complex := fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (↑n) 0) 0 (HMul.hMul ( …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (Complex.exp (HMul …
  -/
  refine (hasSum_mellin_pi_mul_sq (zero_lt_one.trans hs) hF ?_).congr_fun fun n ↦ ?_
    /-
      case refine_1
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      c : Int → Complex := fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (↑n) 0) 0 (HMul.hMul ( …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs ↑i) s.re)
    -/
  · apply (((summable_one_div_int_add_rpow 0 s.re).mpr hs).div_const 2).of_norm_bounded
    /-
      case refine_1
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      c : Int → Complex := fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (↑n) 0) 0 (HMul.hMul ( …
      ⊢ ∀ (i : Int), LE.le (Norm.norm (HDiv.hDiv (Norm.norm (c i)) (HPow.hPow (abs ↑ …
    -/
    intro i
    simp only [c, (by { push_cast; ring } : 2 * π * I * a * i = ↑(2 * π * a * i) * I), norm_div,
      RCLike.norm_ofNat, norm_norm, Complex.norm_exp_ofReal_mul_I, add_zero, norm_one,
      norm_of_nonneg (by positivity : 0 ≤ |(i : ℝ)| ^ s.re), div_right_comm, le_rfl]
    /-
      case refine_2
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      c : Int → Complex := fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (↑n) 0) 0 (HMul.hMul ( …
      n : Int
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (Complex.exp (HMul.hMul (HMul.h …
    -/
  · simp only [c, Int.cast_eq_zero, ← Int.cast_abs, ofReal_intCast, div_right_comm, mul_div_assoc]
    /-
      🎉 no goals
    -/


/-- Formula for `completedCosZeta` as a Dirichlet series in the convergence range
(second version, with sum over `ℕ`). -/
lemma hasSum_nat_completedCosZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ if n = 0 then 0 else Gammaℝ s * Real.cos (2 * π * a * n) / (n : ℂ) ^ s)
    (completedCosZeta a s) := by
  have aux : ((|0| : ℤ) : ℂ) ^ s = 0 := by
    rw [abs_zero, Int.cast_zero, zero_cpow (ne_zero_of_one_lt_re hs)]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    aux : Eq (HPow.hPow (↑(abs 0)) s) 0
    ⊢ HasSum (fun n => ite (Eq n 0) 0 (HDiv.hDiv (HMul.hMul s.Gammaℝ ↑(Real.cos (H …
  -/
  have hint := (hasSum_int_completedCosZeta a hs).nat_add_neg
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    aux : Eq (HPow.hPow (↑(abs 0)) s) 0
    hint : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (C …
    ⊢ HasSum (fun n => ite (Eq n 0) 0 (HDiv.hDiv (HMul.hMul s.Gammaℝ ↑(Real.cos (H …
  -/
  rw [aux, div_zero, zero_div, add_zero] at hint
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    aux : Eq (HPow.hPow (↑(abs 0)) s) 0
    hint : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (C …
    ⊢ HasSum (fun n => ite (Eq n 0) 0 (HDiv.hDiv (HMul.hMul s.Gammaℝ ↑(Real.cos (H …
  -/
  refine hint.congr_fun fun n ↦ ?_
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    aux : Eq (HPow.hPow (↑(abs 0)) s) 0
    hint : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (C …
    n : Nat
    ⊢ Eq (ite (Eq n 0) 0 (HDiv.hDiv (HMul.hMul s.Gammaℝ ↑(Real.cos (HMul.hMul (HMu …
  -/
  split_ifs with h
    /-
      case pos
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      aux : Eq (HPow.hPow (↑(abs 0)) s) 0
      hint : HasSum (fun n => HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (C …
      n : Nat
      h : Eq n 0
      ⊢ Eq 0 (HAdd.hAdd (HDiv.hDiv (HDiv.hDiv (HMul.hMul s.Gammaℝ (Complex.exp (HMul …
    -/
  · simp only [h, Nat.cast_zero, aux, div_zero, zero_div, neg_zero, zero_add]
    /-
      🎉 no goals
    -/
  · simp only [ofReal_cos, ofReal_mul, ofReal_ofNat, ofReal_natCast, Complex.cos,
      show 2 * π * a * n * I = 2 * π * I * a * n by ring, neg_mul, mul_div_assoc,
      div_right_comm _ (2 : ℂ), Int.cast_natCast, Nat.abs_cast, Int.cast_neg, mul_neg, abs_neg, ←
      mul_add, ← add_div]


/-- Formula for `completedHurwitzZetaEven` as a Dirichlet series in the convergence range. -/
lemma hasSum_int_completedHurwitzZetaEven (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ Gammaℝ s / (↑|n + a| : ℂ) ^ s / 2) (completedHurwitzZetaEven a s) := by
  have hF (t : ℝ) (ht : 0 < t) : HasSum (fun n : ℤ ↦ if n + a = 0 then 0
      else (1 / 2 : ℂ) * rexp (-π * (n + a) ^ 2 * t))
      ((evenKernel a t - (if (a : UnitAddCircle) = 0 then 1 else 0 : ℝ)) / 2) := by
    refine (ofReal_sub .. ▸ (hasSum_ofReal.mpr (hasSum_int_evenKernel₀ a ht)).div_const
      2).congr_fun fun n ↦ ?_
    split_ifs
    · rw [ofReal_zero, zero_div]
    · rw [mul_comm, mul_one_div]
  rw [show completedHurwitzZetaEven a s = mellin (fun t ↦ ((evenKernel (↑a) t : ℂ) -
        ↑(if (a : UnitAddCircle) = 0 then 1 else 0 : ℝ)) / 2) (s / 2) by
    simp_rw [mellin_div_const, apply_ite ofReal, ofReal_one, ofReal_zero]
    refine congr_arg (· / 2) ((hurwitzEvenFEPair a).hasMellin (?_ : 1 / 2 < (s / 2).re)).2.symm
    rwa [div_ofNat_re, div_lt_div_iff_of_pos_right two_pos]]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0)  …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv s.Gammaℝ (HPow.hPow (↑(abs (HAdd.hAdd  …
  -/
  refine (hasSum_mellin_pi_mul_sq (zero_lt_one.trans hs) hF ?_).congr_fun fun n ↦ ?_
    /-
      case refine_1
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0)  …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (1 / 2)) (HPow.hPow (abs (HAdd.hAdd ( …
    -/
  · simp_rw [← mul_one_div ‖_‖]
    /-
      case refine_1
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0)  …
      ⊢ Summable fun i => HMul.hMul (Norm.norm (1 / 2)) (HDiv.hDiv 1 (HPow.hPow (abs …
    -/
    apply Summable.mul_left
    /-
      case refine_1.hf
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0)  …
      ⊢ Summable fun i => HDiv.hDiv 1 (HPow.hPow (abs (HAdd.hAdd (↑i) a)) s.re)
    -/
    rwa [summable_one_div_int_add_rpow]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a : Real
      s : Complex
      hs : LT.lt 1 s.re
      hF : ∀ (t : Real), LT.lt 0 t → HasSum (fun n => ite (Eq (HAdd.hAdd (↑n) a) 0)  …
      n : Int
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv s.Gammaℝ (HPow.hPow (↑(abs (HAdd.hAdd (↑n) a))) s)) …
    -/
  · rw [mul_one_div, div_right_comm]
    /-
      🎉 no goals
    -/


/-- Technical lemma which will give us differentiability of Hurwitz zeta at `s = 0`. -/
lemma differentiableAt_update_of_residue
    {Λ : ℂ → ℂ} (hf : ∀ (s : ℂ) (_ : s ≠ 0) (_ : s ≠ 1), DifferentiableAt ℂ Λ s)
    {L : ℂ} (h_lim : Tendsto (fun s ↦ s * Λ s) (𝓝[≠] 0) (𝓝 L)) (s : ℂ) (hs' : s ≠ 1) :
    DifferentiableAt ℂ (Function.update (fun s ↦ Λ s / Gammaℝ s) 0 (L / 2)) s := by
  have claim (t) (ht : t ≠ 0) (ht' : t ≠ 1) : DifferentiableAt ℂ (fun u : ℂ ↦ Λ u / Gammaℝ u) t :=
    (hf t ht ht').mul differentiable_Gammaℝ_inv.differentiableAt
  have claim2 : Tendsto (fun s : ℂ ↦ Λ s / Gammaℝ s) (𝓝[≠] 0) (𝓝 <| L / 2) := by
    refine Tendsto.congr' ?_ (h_lim.div Gammaℝ_residue_zero two_ne_zero)
    filter_upwards [self_mem_nhdsWithin] with s (hs : s ≠ 0)
    rw [Pi.div_apply, ← div_div, mul_div_cancel_left₀ _ hs]
  /-
    Λ : Complex → Complex
    hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
    L : Complex
    h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
    s : Complex
    hs' : Ne s 1
    claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
    claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
    ⊢ DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ …
  -/
  rcases ne_or_eq s 0 with hs | rfl
  · -- Easy case : `s ≠ 0`
    /-
      case inl
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      s : Complex
      hs' : Ne s 1
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs : Ne s 0
      ⊢ DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ …
    -/
    refine (claim s hs hs').congr_of_eventuallyEq ?_
    /-
      case inl
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      s : Complex
      hs' : Ne s 1
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs : Ne s 0
      ⊢ (nhds s).EventuallyEq (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) 0 …
    -/
    filter_upwards [isOpen_compl_singleton.mem_nhds hs] with x hx
    /-
      case h
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      s : Complex
      hs' : Ne s 1
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs : Ne s 0
      x : Complex
      hx : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      ⊢ Eq (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) 0 (HDiv.hDiv L 2) x) …
    -/
    simp only [Function.update_of_ne hx]
    /-
      🎉 no goals
    -/
  · -- Hard case : `s = 0`
    /-
      case inr
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs' : Ne 0 1
      ⊢ DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ …
    -/
    simp_rw [← claim2.limUnder_eq]
    /-
      case inr
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs' : Ne 0 1
      ⊢ DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (Λ s) s.Gammaℝ …
    -/
    have S_nhds : {(1 : ℂ)}ᶜ ∈ 𝓝 (0 : ℂ) := isOpen_compl_singleton.mem_nhds hs'
    refine ((Complex.differentiableOn_update_limUnder_of_isLittleO S_nhds
      (fun t ht ↦ (claim t ht.2 ht.1).differentiableWithinAt) ?_) 0 hs').differentiableAt S_nhds
    /-
      case inr
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs' : Ne 0 1
      S_nhds : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton 1))
      ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0)) …
    -/
    simp only [Gammaℝ, zero_div, div_zero, Complex.Gamma_zero, mul_zero, cpow_zero, sub_zero]
    -- Remains to show completed zeta is `o (s ^ (-1))` near 0.
    /-
      case inr
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs' : Ne 0 1
      S_nhds : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton 1))
      ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0)) …
    -/
    refine (isBigO_const_of_tendsto claim2 <| one_ne_zero' ℂ).trans_isLittleO ?_
    /-
      case inr
      Λ : Complex → Complex
      hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
      L : Complex
      h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
      claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
      claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
      hs' : Ne 0 1
      S_nhds : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton 1))
      ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0)) …
    -/
    rw [isLittleO_iff_tendsto']
      /-
        case inr
        Λ : Complex → Complex
        hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
        L : Complex
        h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
        claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
        claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
        hs' : Ne 0 1
        S_nhds : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton 1))
        ⊢ Filter.Tendsto (fun x => HDiv.hDiv 1 (Inv.inv x)) (nhdsWithin 0 (HasCompl.co …
      -/
    · exact Tendsto.congr (fun x ↦ by rw [← one_div, one_div_one_div]) nhdsWithin_le_nhds
      /-
        🎉 no goals
      -/
      /-
        case inr
        Λ : Complex → Complex
        hf : ∀ (s : Complex), Ne s 0 → Ne s 1 → DifferentiableAt Complex Λ s
        L : Complex
        h_lim : Filter.Tendsto (fun s => HMul.hMul s (Λ s)) (nhdsWithin 0 (HasCompl.co …
        claim : ∀ (t : Complex), Ne t 0 → Ne t 1 → DifferentiableAt Complex (fun u =>  …
        claim2 : Filter.Tendsto (fun s => HDiv.hDiv (Λ s) s.Gammaℝ) (nhdsWithin 0 (Has …
        hs' : Ne 0 1
        S_nhds : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton 1))
        ⊢ Filter.Eventually (fun x => Eq (Inv.inv x) 0 → Eq 1 0) (nhdsWithin 0 (HasCom …
      -/
    · exact eventually_of_mem self_mem_nhdsWithin fun x hx hx' ↦ (hx <| inv_eq_zero.mp hx').elim
      /-
        🎉 no goals
      -/


/-- The even part of the Hurwitz zeta function, i.e. the meromorphic function of `s` which agrees
with `1 / 2 * ∑' (n : ℤ), 1 / |n + a| ^ s` for `1 < re s`-/
noncomputable def hurwitzZetaEven (a : UnitAddCircle) :=
  Function.update (fun s ↦ completedHurwitzZetaEven a s / Gammaℝ s)
  0 (if a = 0 then -1 / 2 else 0)


lemma hurwitzZetaEven_def_of_ne_or_ne {a : UnitAddCircle} {s : ℂ} (h : a ≠ 0 ∨ s ≠ 0) :
    hurwitzZetaEven a s = completedHurwitzZetaEven a s / Gammaℝ s := by
  /-
    a : UnitAddCircle
    s : Complex
    h : Or (Ne a 0) (Ne s 0)
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven a s) (HDiv.hDiv (HurwitzZeta.completedHurwit …
  -/
  rw [hurwitzZetaEven]
  /-
    a : UnitAddCircle
    s : Complex
    h : Or (Ne a 0) (Ne s 0)
    ⊢ Eq (Function.update (fun s => HDiv.hDiv (HurwitzZeta.completedHurwitzZetaEve …
  -/
  rcases ne_or_eq s 0 with h | rfl
    /-
      case inl
      a : UnitAddCircle
      s : Complex
      h✝ : Or (Ne a 0) (Ne s 0)
      h : Ne s 0
      ⊢ Eq (Function.update (fun s => HDiv.hDiv (HurwitzZeta.completedHurwitzZetaEve …
    -/
  · rw [Function.update_of_ne h]
    /-
      🎉 no goals
    -/
  · simpa only [Gammaℝ, Function.update_self, neg_zero, zero_div, cpow_zero, Complex.Gamma_zero,
    mul_zero, div_zero, ite_eq_right_iff, div_eq_zero_iff, neg_eq_zero, one_ne_zero,
    OfNat.ofNat_ne_zero, or_self, imp_false, ne_eq, not_true_eq_false, or_false] using h


lemma hurwitzZetaEven_apply_zero (a : UnitAddCircle) :
    hurwitzZetaEven a 0 = if a = 0 then -1 / 2 else 0 :=
  Function.update_self ..


lemma hurwitzZetaEven_neg (a : UnitAddCircle) (s : ℂ) :
    hurwitzZetaEven (-a) s = hurwitzZetaEven a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven (Neg.neg a) s) (HurwitzZeta.hurwitzZetaEven  …
  -/
  simp_rw [hurwitzZetaEven, neg_eq_zero, completedHurwitzZetaEven_neg]
  /-
    🎉 no goals
  -/


/-- The trivial zeroes of the even Hurwitz zeta function. -/
theorem hurwitzZetaEven_neg_two_mul_nat_add_one (a : UnitAddCircle) (n : ℕ) :
    hurwitzZetaEven a (-2 * (n + 1)) = 0 := by
  have : (-2 : ℂ) * (n + 1) ≠ 0 :=
    mul_ne_zero (neg_ne_zero.mpr two_ne_zero) (Nat.cast_add_one_ne_zero n)
  rw [hurwitzZetaEven, Function.update_of_ne this,
    Gammaℝ_eq_zero_iff.mpr ⟨n + 1, by rw [neg_mul, Nat.cast_add_one]⟩, div_zero]


/-- The Hurwitz zeta function is differentiable everywhere except at `s = 1`. This is true
even in the delicate case `a = 0` and `s = 0` (where the completed zeta has a pole, but this is
cancelled out by the Gamma factor). -/
lemma differentiableAt_hurwitzZetaEven (a : UnitAddCircle) {s : ℂ} (hs' : s ≠ 1) :
    DifferentiableAt ℂ (hurwitzZetaEven a) s := by
  have := differentiableAt_update_of_residue
    (fun t ht ht' ↦ differentiableAt_completedHurwitzZetaEven a (Or.inl ht) ht')
    (completedHurwitzZetaEven_residue_zero a) s hs'
  /-
    a : UnitAddCircle
    s : Complex
    hs' : Ne s 1
    this : DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (HurwitzZ …
    ⊢ DifferentiableAt Complex (HurwitzZeta.hurwitzZetaEven a) s
  -/
  simp_rw [div_eq_mul_inv, ite_mul, zero_mul, ← div_eq_mul_inv] at this
  /-
    a : UnitAddCircle
    s : Complex
    hs' : Ne s 1
    this : DifferentiableAt Complex (Function.update (fun s => HDiv.hDiv (HurwitzZ …
    ⊢ DifferentiableAt Complex (HurwitzZeta.hurwitzZetaEven a) s
  -/
  exact this
  /-
    🎉 no goals
  -/


lemma hurwitzZetaEven_residue_one (a : UnitAddCircle) :
    Tendsto (fun s ↦ (s - 1) * hurwitzZetaEven a s) (𝓝[≠] 1) (𝓝 1) := by
  have : Tendsto (fun s ↦ (s - 1) * completedHurwitzZetaEven a s / Gammaℝ s) (𝓝[≠] 1) (𝓝 1) := by
    simpa only [Gammaℝ_one, inv_one, mul_one] using (completedHurwitzZetaEven_residue_one a).mul
      <| (differentiable_Gammaℝ_inv.continuous.tendsto _).mono_left nhdsWithin_le_nhds
  /-
    a : UnitAddCircle
    this : Filter.Tendsto (fun s => HDiv.hDiv (HMul.hMul (HSub.hSub s 1) (HurwitzZ …
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HurwitzZeta.hurwitzZetaE …
  -/
  refine this.congr' ?_
  /-
    a : UnitAddCircle
    this : Filter.Tendsto (fun s => HDiv.hDiv (HMul.hMul (HSub.hSub s 1) (HurwitzZ …
    ⊢ (nhdsWithin 1 (HasCompl.compl (Singleton.singleton 1))).EventuallyEq (fun s  …
  -/
  filter_upwards [eventually_ne_nhdsWithin one_ne_zero] with s hs
  /-
    case h
    a : UnitAddCircle
    this : Filter.Tendsto (fun s => HDiv.hDiv (HMul.hMul (HSub.hSub s 1) (HurwitzZ …
    s : Complex
    hs : Ne s 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HSub.hSub s 1) (HurwitzZeta.completedHurwitzZetaEv …
  -/
  simp_rw [hurwitzZetaEven_def_of_ne_or_ne (Or.inr hs), mul_div_assoc]
  /-
    🎉 no goals
  -/


lemma differentiableAt_hurwitzZetaEven_sub_one_div (a : UnitAddCircle) :
    DifferentiableAt ℂ (fun s ↦ hurwitzZetaEven a s - 1 / (s - 1) / Gammaℝ s) 1 := by
  suffices DifferentiableAt ℂ
      (fun s ↦ completedHurwitzZetaEven a s / Gammaℝ s - 1 / (s - 1) / Gammaℝ s) 1 by
    apply this.congr_of_eventuallyEq
    filter_upwards [eventually_ne_nhds one_ne_zero] with x hx
    rw [hurwitzZetaEven, Function.update_of_ne hx]
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HDiv.hDiv (HurwitzZeta.complet …
  -/
  simp_rw [← sub_div, div_eq_mul_inv _ (Gammaℝ _)]
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HMul.hMul (HSub.hSub (HurwitzZeta.complet …
  -/
  refine DifferentiableAt.mul ?_ differentiable_Gammaℝ_inv.differentiableAt
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.completedHurwitzZe …
  -/
  simp_rw [completedHurwitzZetaEven_eq, sub_sub, add_assoc]
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.completedHurwitzZe …
  -/
  conv => enter [2, s, 2]; rw [← neg_sub, div_neg, neg_add_cancel, add_zero]
  exact (differentiable_completedHurwitzZetaEven₀ a _).sub
    <| (differentiableAt_const _).div differentiableAt_id one_ne_zero


/-- Expression for `hurwitzZetaEven a 1` as a limit. (Mathematically `hurwitzZetaEven a 1` is
undefined, but our construction assigns some value to it; this lemma is mostly of interest for
determining what that value is). -/
lemma tendsto_hurwitzZetaEven_sub_one_div_nhds_one (a : UnitAddCircle) :
    Tendsto (fun s ↦ hurwitzZetaEven a s - 1 / (s - 1) / Gammaℝ s) (𝓝 1)
    (𝓝 (hurwitzZetaEven a 1)) := by
  simpa only [one_div, sub_self, div_zero, Gammaℝ_one, div_one, sub_zero] using
    (differentiableAt_hurwitzZetaEven_sub_one_div a).continuousAt.tendsto


lemma differentiable_hurwitzZetaEven_sub_hurwitzZetaEven (a b : UnitAddCircle) :
    Differentiable ℂ (fun s ↦ hurwitzZetaEven a s - hurwitzZetaEven b s) := by
  /-
    a b : UnitAddCircle
    ⊢ Differentiable Complex fun s => HSub.hSub (HurwitzZeta.hurwitzZetaEven a s)  …
  -/
  intro z
  /-
    a b : UnitAddCircle
    z : Complex
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.hurwitzZetaEven a  …
  -/
  rcases ne_or_eq z 1 with hz | rfl
    /-
      case inl
      a b : UnitAddCircle
      z : Complex
      hz : Ne z 1
      ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.hurwitzZetaEven a  …
    -/
  · exact (differentiableAt_hurwitzZetaEven a hz).sub (differentiableAt_hurwitzZetaEven b hz)
    /-
      🎉 no goals
    -/
  · convert (differentiableAt_hurwitzZetaEven_sub_one_div a).sub
      (differentiableAt_hurwitzZetaEven_sub_one_div b) using 2 with s
    /-
      case h.e'_11.h
      a b : UnitAddCircle
      x✝ : Complex
      ⊢ Eq (HSub.hSub (HurwitzZeta.hurwitzZetaEven a x✝) (HurwitzZeta.hurwitzZetaEve …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/--
Formula for `hurwitzZetaEven` as a Dirichlet series in the convergence range, with sum over `ℤ`.
-/
lemma hasSum_int_hurwitzZetaEven (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ 1 / (↑|n + a| : ℂ) ^ s / 2) (hurwitzZetaEven a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv 1 (HPow.hPow (↑(abs (HAdd.hAdd (↑n) a) …
  -/
  rw [hurwitzZetaEven, Function.update_of_ne (ne_zero_of_one_lt_re hs)]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv 1 (HPow.hPow (↑(abs (HAdd.hAdd (↑n) a) …
  -/
  have := (hasSum_int_completedHurwitzZetaEven a hs).div_const (Gammaℝ s)
  exact this.congr_fun fun n ↦ by simp only [div_right_comm _ _ (Gammaℝ _),
    div_self (Gammaℝ_ne_zero_of_re_pos (zero_lt_one.trans hs))]


/-- Formula for `hurwitzZetaEven` as a Dirichlet series in the convergence range, with sum over `ℕ`
(version with absolute values) -/
lemma hasSum_nat_hurwitzZetaEven (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ (1 / (↑|n + a| : ℂ) ^ s + 1 / (↑|n + 1 - a| : ℂ) ^ s) / 2)
    (hurwitzZetaEven a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow (↑(abs (HAdd.h …
  -/
  refine (hasSum_int_hurwitzZetaEven a hs).nat_add_neg_add_one.congr_fun fun n ↦ ?_
  simp only [← abs_neg (n + 1 - a), neg_sub', sub_neg_eq_add, add_div, Int.cast_natCast,
    Int.cast_neg, Int.cast_add, Int.cast_one]


/-- Formula for `hurwitzZetaEven` as a Dirichlet series in the convergence range, with sum over `ℕ`
(version without absolute values, assuming `a ∈ Icc 0 1`) -/
lemma hasSum_nat_hurwitzZetaEven_of_mem_Icc {a : ℝ} (ha : a ∈ Icc 0 1) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ (1 / (n + a : ℂ) ^ s + 1 / (n + 1 - a : ℂ) ^ s) / 2)
    (hurwitzZetaEven a s) := by
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n  …
  -/
  refine (hasSum_nat_hurwitzZetaEven a hs).congr_fun fun n ↦ ?_
  /-
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n ↑a) s)) (HDiv …
  -/
  congr 2 <;>
  /-
    case e_a.e_a
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n ↑a) s)) (HDiv.hDiv 1 (HPow.hPow (↑( …
  -/
  rw [_root_.abs_of_nonneg (by linarith [ha.1, ha.2])] <;>
  /-
    case e_a.e_a
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n ↑a) s)) (HDiv.hDiv 1 (HPow.hPow (↑( …
  -/
  /-
    🎉 no goals
  -/
  simp only [one_div, ofReal_sub, ofReal_add, ofReal_natCast, ofReal_one]
  /-
    🎉 no goals
  -/


/-- The cosine zeta function, i.e. the meromorphic function of `s` which agrees
with `∑' (n : ℕ), cos (2 * π * a * n) / n ^ s` for `1 < re s`. -/
noncomputable def cosZeta (a : UnitAddCircle) :=
  Function.update (fun s : ℂ ↦ completedCosZeta a s / Gammaℝ s) 0 (-1 / 2)


lemma cosZeta_apply_zero (a : UnitAddCircle) : cosZeta a 0 = -1 / 2 :=
  Function.update_self ..


lemma cosZeta_neg (a : UnitAddCircle) (s : ℂ) :
    cosZeta (-a) s = cosZeta a s := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.cosZeta (Neg.neg a) s) (HurwitzZeta.cosZeta a s)
  -/
  simp_rw [cosZeta, completedCosZeta_neg]
  /-
    🎉 no goals
  -/


/-- The trivial zeroes of the cosine zeta function. -/
theorem cosZeta_neg_two_mul_nat_add_one (a : UnitAddCircle) (n : ℕ) :
    cosZeta a (-2 * (n + 1)) = 0 := by
  have : (-2 : ℂ) * (n + 1) ≠ 0 :=
    mul_ne_zero (neg_ne_zero.mpr two_ne_zero) (Nat.cast_add_one_ne_zero n)
  rw [cosZeta, Function.update_of_ne this,
    Gammaℝ_eq_zero_iff.mpr ⟨n + 1, by rw [neg_mul, Nat.cast_add_one]⟩, div_zero]


/-- The cosine zeta function is differentiable everywhere, except at `s = 1` if `a = 0`. -/
lemma differentiableAt_cosZeta (a : UnitAddCircle) {s : ℂ} (hs' : s ≠ 1 ∨ a ≠ 0) :
    DifferentiableAt ℂ (cosZeta a) s := by
  /-
    a : UnitAddCircle
    s : Complex
    hs' : Or (Ne s 1) (Ne a 0)
    ⊢ DifferentiableAt Complex (HurwitzZeta.cosZeta a) s
  -/
  rcases ne_or_eq s 1 with hs' | rfl
  · exact differentiableAt_update_of_residue (fun _ ht ht' ↦
      differentiableAt_completedCosZeta a ht (Or.inl ht')) (completedCosZeta_residue_zero a) s hs'
  · apply ((differentiableAt_completedCosZeta a one_ne_zero hs').mul
      (differentiable_Gammaℝ_inv.differentiableAt)).congr_of_eventuallyEq
    /-
      case inr
      a : UnitAddCircle
      hs' : Or (Ne 1 1) (Ne a 0)
      ⊢ (nhds 1).EventuallyEq (HurwitzZeta.cosZeta a) fun y => HMul.hMul (HurwitzZet …
    -/
    filter_upwards [isOpen_compl_singleton.mem_nhds one_ne_zero] with x hx
    /-
      case h
      a : UnitAddCircle
      hs' : Or (Ne 1 1) (Ne a 0)
      x : Complex
      hx : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      ⊢ Eq (HurwitzZeta.cosZeta a x) (HMul.hMul (HurwitzZeta.completedCosZeta a x) ( …
    -/
    simp_rw [cosZeta, Function.update_of_ne hx, div_eq_mul_inv]
    /-
      🎉 no goals
    -/


/-- If `a ≠ 0` then the cosine zeta function is entire. -/
lemma differentiable_cosZeta_of_ne_zero {a : UnitAddCircle} (ha : a ≠ 0) :
    Differentiable ℂ (cosZeta a) :=
  fun _ ↦ differentiableAt_cosZeta a (Or.inr ha)


/-- Formula for `cosZeta` as a Dirichlet series in the convergence range, with sum over `ℤ`. -/
lemma hasSum_int_cosZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℤ ↦ cexp (2 * π * I * a * n) / ↑|n| ^ s / 2) (cosZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
  -/
  rw [cosZeta, Function.update_of_ne (ne_zero_of_one_lt_re hs)]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HM …
  -/
  refine ((hasSum_int_completedCosZeta a hs).div_const (Gammaℝ s)).congr_fun fun n ↦ ?_
  rw [mul_div_assoc _ (cexp _), div_right_comm _ (2 : ℂ),
    mul_div_cancel_left₀ _ (Gammaℝ_ne_zero_of_re_pos (zero_lt_one.trans hs))]


/-- Formula for `cosZeta` as a Dirichlet series in the convergence range, with sum over `ℕ`. -/
lemma hasSum_nat_cosZeta (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ Real.cos (2 * π * a * n) / (n : ℂ) ^ s) (cosZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (↑(Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 Re …
  -/
  have := (hasSum_int_cosZeta a hs).nat_add_neg
  simp_rw [abs_neg, Int.cast_neg, Nat.abs_cast, Int.cast_natCast, mul_neg, abs_zero, Int.cast_zero,
    zero_cpow (ne_zero_of_one_lt_re hs), div_zero, zero_div, add_zero, ← add_div,
    div_right_comm _ _ (2 : ℂ)] at this
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMu …
    ⊢ HasSum (fun n => HDiv.hDiv (↑(Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 Re …
  -/
  simp_rw [push_cast, Complex.cos, neg_mul]
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    this : HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMu …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul (HM …
  -/
  exact this.congr_fun fun n ↦ by rw [show 2 * π * a * n * I = 2 * π * I * a * n by ring]
  /-
    🎉 no goals
  -/


/-- Reformulation of `hasSum_nat_cosZeta` using `LSeriesHasSum`. -/
lemma LSeriesHasSum_cos (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    LSeriesHasSum (Real.cos <| 2 * π * a * ·) s (cosZeta a s) :=
  (hasSum_nat_cosZeta a hs).congr_fun
    (LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs) _)


/-- If `s` is not in `-ℕ`, and either `a ≠ 0` or `s ≠ 1`, then
`hurwitzZetaEven a (1 - s)` is an explicit multiple of `cosZeta s`. -/
lemma hurwitzZetaEven_one_sub (a : UnitAddCircle) {s : ℂ}
    (hs : ∀ (n : ℕ), s ≠ -n) (hs' : a ≠ 0 ∨ s ≠ 1) :
    hurwitzZetaEven a (1 - s) = 2 * (2 * π) ^ (-s) * Gamma s * cos (π * s / 2) * cosZeta a s := by
  have : hurwitzZetaEven a (1 - s) = completedHurwitzZetaEven a (1 - s) * (Gammaℝ (1 - s))⁻¹ := by
    rw [hurwitzZetaEven_def_of_ne_or_ne, div_eq_mul_inv]
    simpa [sub_eq_zero, eq_comm (a := s)] using hs'
  rw [this, completedHurwitzZetaEven_one_sub, inv_Gammaℝ_one_sub hs, cosZeta,
    Function.update_of_ne (by simpa using hs 0), ← Gammaℂ]
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    this : Eq (HurwitzZeta.hurwitzZetaEven a (HSub.hSub 1 s)) (HMul.hMul (HurwitzZ …
    ⊢ Eq (HMul.hMul (HurwitzZeta.completedCosZeta a s) (HMul.hMul (HMul.hMul s.Gam …
  -/
  generalize Gammaℂ s * cos (π * s / 2) = A -- speeds up ring_nf call
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    this : Eq (HurwitzZeta.hurwitzZetaEven a (HSub.hSub 1 s)) (HMul.hMul (HurwitzZ …
    A : Complex
    ⊢ Eq (HMul.hMul (HurwitzZeta.completedCosZeta a s) (HMul.hMul A (Inv.inv s.Gam …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- If `s` is not of the form `1 - n` for `n ∈ ℕ`, then `cosZeta a (1 - s)` is an explicit
multiple of `hurwitzZetaEven s`. -/
lemma cosZeta_one_sub (a : UnitAddCircle) {s : ℂ} (hs : ∀ (n : ℕ), s ≠ 1 - n) :
    cosZeta a (1 - s) = 2 * (2 * π) ^ (-s) * Gamma s * cos (π * s / 2) * hurwitzZetaEven a s := by
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    ⊢ Eq (HurwitzZeta.cosZeta a (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HMul.hMul  …
  -/
  rw [← Gammaℂ]
  have : cosZeta a (1 - s) = completedCosZeta a (1 - s) * (Gammaℝ (1 - s))⁻¹ := by
    rw [cosZeta, Function.update_of_ne, div_eq_mul_inv]
    simpa [sub_eq_zero] using (hs 0).symm
  rw [this, completedCosZeta_one_sub, inv_Gammaℝ_one_sub (fun n ↦ by simpa using hs (n + 1)),
    hurwitzZetaEven_def_of_ne_or_ne (Or.inr (by simpa using hs 1))]
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    this : Eq (HurwitzZeta.cosZeta a (HSub.hSub 1 s)) (HMul.hMul (HurwitzZeta.comp …
    ⊢ Eq (HMul.hMul (HurwitzZeta.completedHurwitzZetaEven a s) (HMul.hMul (HMul.hM …
  -/
  generalize Gammaℂ s * cos (π * s / 2) = A -- speeds up ring_nf call
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    this : Eq (HurwitzZeta.cosZeta a (HSub.hSub 1 s)) (HMul.hMul (HurwitzZeta.comp …
    A : Complex
    ⊢ Eq (HMul.hMul (HurwitzZeta.completedHurwitzZetaEven a s) (HMul.hMul A (Inv.i …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


