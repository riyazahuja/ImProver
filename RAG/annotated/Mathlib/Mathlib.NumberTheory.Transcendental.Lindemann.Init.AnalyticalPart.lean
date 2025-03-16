theorem hasDerivAt_cexp_mul_sumIDeriv (p : ℂ[X]) (s : ℂ) (x : ℝ) :
    HasDerivAt (fun x : ℝ ↦ -(cexp (-(x • s)) * p.sumIDeriv.eval (x • s)))
      (s * (cexp (-(x • s)) * p.eval (x • s))) x := by
  /-
    p : Polynomial Complex
    s : Complex
    x : Real
    ⊢ HasDerivAt (fun x => Neg.neg (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x …
  -/
  have h₀ := (hasDerivAt_id' x).smul_const s
  /-
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    ⊢ HasDerivAt (fun x => Neg.neg (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x …
  -/
  have h₁ := h₀.neg.cexp
  /-
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    h₁ : HasDerivAt (fun x => Complex.exp (Neg.neg (HSMul.hSMul x s))) (HMul.hMul  …
    ⊢ HasDerivAt (fun x => Neg.neg (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x …
  -/
  have h₂ := ((sumIDeriv p).hasDerivAt (x • s)).comp x h₀
  /-
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    h₁ : HasDerivAt (fun x => Complex.exp (Neg.neg (HSMul.hSMul x s))) (HMul.hMul  …
    h₂ : HasDerivAt (Function.comp (fun x => Polynomial.eval x (Polynomial.sumIDer …
    ⊢ HasDerivAt (fun x => Neg.neg (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x …
  -/
  convert (h₁.mul h₂).neg using 1
  /-
    case h.e'_9
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    h₁ : HasDerivAt (fun x => Complex.exp (Neg.neg (HSMul.hSMul x s))) (HMul.hMul  …
    h₂ : HasDerivAt (Function.comp (fun x => Polynomial.eval x (Polynomial.sumIDer …
    ⊢ Eq (HMul.hMul s (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x s))) (Polyno …
  -/
  nth_rw 1 [sumIDeriv_eq_self_add p]
  /-
    case h.e'_9
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    h₁ : HasDerivAt (fun x => Complex.exp (Neg.neg (HSMul.hSMul x s))) (HMul.hMul  …
    h₂ : HasDerivAt (Function.comp (fun x => Polynomial.eval x (Polynomial.sumIDer …
    ⊢ Eq (HMul.hMul s (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x s))) (Polyno …
  -/
  simp only [one_smul, eval_add, Function.comp_apply]
  /-
    case h.e'_9
    p : Polynomial Complex
    s : Complex
    x : Real
    h₀ : HasDerivAt (fun y => HSMul.hSMul y s) (HSMul.hSMul 1 s) x
    h₁ : HasDerivAt (fun x => Complex.exp (Neg.neg (HSMul.hSMul x s))) (HMul.hMul  …
    h₂ : HasDerivAt (Function.comp (fun x => Polynomial.eval x (Polynomial.sumIDer …
    ⊢ Eq (HMul.hMul s (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x s))) (Polyno …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem integral_exp_mul_eval (p : ℂ[X]) (s : ℂ) :
    s * ∫ x in (0)..1, exp (-(x • s)) * p.eval (x • s) =
      -(exp (-s) * p.sumIDeriv.eval s) + p.sumIDeriv.eval 0 := by
  rw [← intervalIntegral.integral_const_mul,
    intervalIntegral.integral_eq_sub_of_hasDerivAt
      (fun x hx => hasDerivAt_cexp_mul_sumIDeriv p s x)
      (ContinuousOn.intervalIntegrable (by fun_prop))]
  /-
    p : Polynomial Complex
    s : Complex
    ⊢ Eq (HSub.hSub (Neg.neg (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul 1 s)))  …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
`P` is a slightly generalized version of `Iᵢ` in
[the wikipedia proof](https://en.wikipedia.org/wiki/Lindemann%E2%80%93Weierstrass_theorem):
`Iᵢ(s) = P(fᵢ, s)`.
-/
private def P (f : ℂ[X]) (s : ℂ) :=
  exp s * f.sumIDeriv.eval 0 - f.sumIDeriv.eval s


private theorem P_eq_integral_exp_mul_eval (f : ℂ[X]) (s : ℂ) :
    P f s = exp s * (s * ∫ x in (0)..1, exp (-(x • s)) * f.eval (x • s)) := by
  rw [integral_exp_mul_eval, mul_add, mul_neg, exp_neg, mul_inv_cancel_left₀ (exp_ne_zero s),
    neg_add_eq_sub, P]


/--
Given a sequence of complex polynomials `fₚ`, a complex constant `s`, and a real constant `c` such
that `|fₚ(xs)| ≤ c ^ p` for all `p ∈ ℕ` and `x ∈ Ioc 0 1`, then there is also a nonnegative
constant `c'` such that for all nonzero `p ∈ ℕ`, `|P(fₚ, s)| ≤ c' ^ p`.
-/
private theorem P_le_aux (f : ℕ → ℂ[X]) (s : ℂ) (c : ℝ)
    (hc : ∀ p : ℕ, ∀ x ∈ Set.Ioc (0 : ℝ) 1, Complex.abs ((f p).eval (x • s)) ≤ c ^ p) :
    ∃ c' ≥ 0, ∀ p : ℕ,
      Complex.abs (P (f p) s) ≤
        Real.exp s.re * (Real.exp (Complex.abs s) * c' ^ p * (Complex.abs s)) := by
  /-
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), LE.le (Complex.abs (Lindeman …
  -/
  refine ⟨|c|, abs_nonneg _, fun p => ?_⟩
  /-
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    ⊢ LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.hMul (Real.exp s. …
  -/
  rw [P_eq_integral_exp_mul_eval (f p) s, mul_comm s, map_mul, map_mul, abs_exp]
  /-
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    ⊢ LE.le (HMul.hMul (Real.exp s.re) (HMul.hMul (Complex.abs (intervalIntegral ( …
  -/
  gcongr
  /-
    case h.h
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    ⊢ LE.le (Complex.abs (intervalIntegral (fun x => HMul.hMul (Complex.exp (Neg.n …
  -/
  rw [intervalIntegral.integral_of_le zero_le_one, ← norm_eq_abs, ← mul_one (_ * _)]
  /-
    case h.h
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
  -/
  convert MeasureTheory.norm_setIntegral_le_of_norm_le_const' _ _ _
    /-
      case h.e'_4.h.e'_6
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      ⊢ Eq 1 (MeasureTheory.MeasureSpace.volume (Set.Ioc 0 1)).toReal
    -/
  · rw [Real.volume_Ioc, sub_zero, ENNReal.toReal_ofReal zero_le_one]
    /-
      🎉 no goals
    -/
    /-
      case h.h.convert_10
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      ⊢ LT.lt (MeasureTheory.MeasureSpace.volume (Set.Ioc 0 1)) Top.top
    -/
  · rw [Real.volume_Ioc, sub_zero]; exact ENNReal.ofReal_lt_top
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case h.h.convert_11
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      ⊢ MeasurableSet (Set.Ioc 0 1)
    -/
  · exact measurableSet_Ioc
    /-
      🎉 no goals
    -/
  /-
    case h.h.convert_12
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    ⊢ ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Norm.norm (HMul.hMul ( …
  -/
  intro x hx
  /-
    case h.h.convert_12
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    ⊢ LE.le (Norm.norm (HMul.hMul (Complex.exp (Neg.neg (HSMul.hSMul x s))) (Polyn …
  -/
  rw [norm_mul, norm_eq_abs, abs_exp]
  /-
    case h.h.convert_12
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    ⊢ LE.le (HMul.hMul (Real.exp (Neg.neg (HSMul.hSMul x s)).re) (Norm.norm (Polyn …
  -/
  gcongr
    /-
      case h.h.convert_12.h₁.h
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      ⊢ LE.le (Neg.neg (HSMul.hSMul x s)).re (Complex.abs s)
    -/
  · simp only [Set.mem_Ioc] at hx
    /-
      case h.h.convert_12.h₁.h
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : And (LT.lt 0 x) (LE.le x 1)
      ⊢ LE.le (Neg.neg (HSMul.hSMul x s)).re (Complex.abs s)
    -/
    apply (re_le_abs _).trans
    /-
      case h.h.convert_12.h₁.h
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : And (LT.lt 0 x) (LE.le x 1)
      ⊢ LE.le (Complex.abs (Neg.neg (HSMul.hSMul x s))) (Complex.abs s)
    -/
    rw [← norm_eq_abs, ← norm_eq_abs, norm_neg, norm_smul, Real.norm_of_nonneg hx.1.le]
    /-
      case h.h.convert_12.h₁.h
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : And (LT.lt 0 x) (LE.le x 1)
      ⊢ LE.le (HMul.hMul x (Norm.norm s)) (Norm.norm s)
    -/
    exact mul_le_of_le_one_left (norm_nonneg _) hx.2
    /-
      🎉 no goals
    -/
    /-
      case h.h.convert_12.h₂
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      ⊢ LE.le (Norm.norm (Polynomial.eval (HSMul.hSMul x s) (f p))) (HPow.hPow (abs  …
    -/
  · rw [← _root_.abs_pow, norm_eq_abs]
    /-
      case h.h.convert_12.h₂
      f : Nat → Polynomial Complex
      s : Complex
      c : Real
      hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      ⊢ LE.le (Complex.abs (Polynomial.eval (HSMul.hSMul x s) (f p))) (abs (HPow.hPo …
    -/
    exact (hc p x hx).trans (le_abs_self _)
    /-
      🎉 no goals
    -/


/--
Given a sequence of complex polynomials `fₚ`, a complex constant `s`, and a real constant `c` such
that `|fₚ(xs)| ≤ c ^ p` for all `p ∈ ℕ` and `x ∈ Ioc 0 1`, then there is also a nonnegative
constant `c'` such that for all nonzero `p ∈ ℕ`, `|P(fₚ, s)| ≤ c' ^ p`.
-/
private theorem P_le (f : ℕ → ℂ[X]) (s : ℂ) (c : ℝ)
    (hc : ∀ p : ℕ, ∀ x ∈ Set.Ioc (0 : ℝ) 1, Complex.abs ((f p).eval (x • s)) ≤ c ^ p) :
    ∃ c' ≥ 0, ∀ p ≠ 0, Complex.abs (P (f p) s) ≤ c' ^ p := by
  /-
    f : Nat → Polynomial Complex
    s : Complex
    c : Real
    hc : ∀ (p : Nat) (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (Complex.a …
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs  …
  -/
  obtain ⟨c', hc', h'⟩ := P_le_aux f s c hc; clear c hc
  /-
    case intro.intro
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs  …
  -/
  let c₁ := max (Real.exp s.re) 1
  /-
    case intro.intro
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs  …
  -/
  let c₂ := max (Real.exp (Complex.abs s)) 1
  /-
    case intro.intro
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs  …
  -/
  let c₃ := max (Complex.abs s) 1
  /-
    case intro.intro
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    c₃ : Real := Max.max (Complex.abs s) 1
    ⊢ Exists fun c' => And (GE.ge c' 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs  …
  -/
  use c₁ * (c₂ * c' * c₃), by positivity
  /-
    case right
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    c₃ : Real := Max.max (Complex.abs s) 1
    ⊢ ∀ (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) ( …
  -/
  intro p hp
  /-
    case right
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    c₃ : Real := Max.max (Complex.abs s) 1
    p : Nat
    hp : Ne p 0
    ⊢ LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HPow.hPow (HMul.hMul c …
  -/
  refine (h' p).trans ?_
  /-
    case right
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    c₃ : Real := Max.max (Complex.abs s) 1
    p : Nat
    hp : Ne p 0
    ⊢ LE.le (HMul.hMul (Real.exp s.re) (HMul.hMul (HMul.hMul (Real.exp (Complex.ab …
  -/
  simp_rw [mul_pow]
  have le_max_one_pow {x : ℝ} : x ≤ max x 1 ^ p :=
    (max_cases x 1).elim (fun h ↦ h.1.symm ▸ le_self_pow₀ h.2 hp)
      fun h ↦ by rw [h.1, one_pow]; exact h.2.le
  /-
    case right
    f : Nat → Polynomial Complex
    s : Complex
    c' : Real
    hc' : GE.ge c' 0
    h' : ∀ (p : Nat), LE.le (Complex.abs (LindemannWeierstrass.P (f p) s)) (HMul.h …
    c₁ : Real := Max.max (Real.exp s.re) 1
    c₂ : Real := Max.max (Real.exp (Complex.abs s)) 1
    c₃ : Real := Max.max (Complex.abs s) 1
    p : Nat
    hp : Ne p 0
    le_max_one_pow : ∀ {x : Real}, LE.le x (HPow.hPow (Max.max x 1) p)
    ⊢ LE.le (HMul.hMul (Real.exp s.re) (HMul.hMul (HMul.hMul (Real.exp (Complex.ab …
  -/
             /-
               🎉 no goals
             -/
             /-
               🎉 no goals
             -/
  gcongr <;> exact le_max_one_pow
             /-
               🎉 no goals
             -/


/--
Given a polynomial with integer coefficients `p` and a complex constant `s`, there is a nonnegative
`c` such that for all nonzero `q ∈ ℕ`, `|P(X ^ (q - 1) * p ^ q, s)| ≤ c ^ q`.

Note: Jacobson writes `h(x)` for `x ^ (q - 1) * p(x) ^ q` and `bⱼ` for its coefficients.
-/
private theorem exp_polynomial_approx_aux (f : ℤ[X]) (s : ℂ) :
    ∃ c ≥ 0,
      ∀ p ≠ 0, Complex.abs (P (map (algebraMap ℤ ℂ) (X ^ (p - 1) * f ^ p)) s) ≤ c ^ p := by
  have : Bornology.IsBounded
      ((fun x : ℝ ↦ max (x * abs s) 1 * Complex.abs (aeval (x * s) f)) '' Set.Ioc 0 1) := by
    have h :
      (fun x : ℝ ↦ max (x * abs s) 1 * Complex.abs (aeval (x * s) f)) '' Set.Ioc 0 1 ⊆
        (fun x : ℝ ↦ max (x * abs s) 1 * Complex.abs (aeval (x * s) f)) '' Set.Icc 0 1 :=
      Set.image_subset _ Set.Ioc_subset_Icc_self
    refine (IsCompact.image isCompact_Icc ?_).isBounded.subset h
    fun_prop
  /-
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    ⊢ Exists fun c => And (GE.ge c 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs (L …
  -/
  obtain ⟨c, h⟩ := this.exists_norm_le
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    h : ∀ (x : Real), Membership.mem (Set.image (fun x => HMul.hMul (Max.max (HMul …
    ⊢ Exists fun c => And (GE.ge c 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs (L …
  -/
  simp_rw [Real.norm_eq_abs] at h
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    h : ∀ (x : Real), Membership.mem (Set.image (fun x => HMul.hMul (Max.max (HMul …
    ⊢ Exists fun c => And (GE.ge c 0) (∀ (p : Nat), Ne p 0 → LE.le (Complex.abs (L …
  -/
  refine P_le _ s c (fun p x hx => ?_)
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    h : ∀ (x : Real), Membership.mem (Set.image (fun x => HMul.hMul (Max.max (HMul …
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    ⊢ LE.le (Complex.abs (Polynomial.eval (HSMul.hSMul x s) (Polynomial.map (algeb …
  -/
  specialize h (max (x * abs s) 1 * Complex.abs (aeval (x * s) f)) (Set.mem_image_of_mem _ hx)
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
    ⊢ LE.le (Complex.abs (Polynomial.eval (HSMul.hSMul x s) (Polynomial.map (algeb …
  -/
  refine le_trans ?_ (pow_le_pow_left₀ (abs_nonneg _) h _)
  simp_rw [Polynomial.map_mul, Polynomial.map_pow, map_X, eval_mul, eval_pow, eval_X, map_mul,
    Complex.abs_pow, real_smul, map_mul, abs_ofReal, ← eval₂_eq_eval_map, ← aeval_def, abs_mul,
    Complex.abs_abs, mul_pow, abs_of_pos hx.1]
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow x (HSub.hSub p 1)) (HPow.hPow (Comple …
  -/
  refine mul_le_mul_of_nonneg_right ?_ (pow_nonneg (Complex.abs.nonneg _) _)
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
    ⊢ LE.le (HMul.hMul (HPow.hPow x (HSub.hSub p 1)) (HPow.hPow (Complex.abs s) (H …
  -/
  rw [← mul_pow, _root_.abs_of_nonneg (by positivity), max_def]
  /-
    case intro
    f : Polynomial Int
    s : Complex
    this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
    c : Real
    p : Nat
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
    ⊢ LE.le (HPow.hPow (HMul.hMul x (Complex.abs s)) (HSub.hSub p 1)) (HPow.hPow ( …
  -/
  split_ifs with hx1
    /-
      case pos
      f : Polynomial Int
      s : Complex
      this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
      c : Real
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
      hx1 : LE.le (HMul.hMul x (Complex.abs s)) 1
      ⊢ LE.le (HPow.hPow (HMul.hMul x (Complex.abs s)) (HSub.hSub p 1)) (HPow.hPow 1 …
    -/
  · rw [one_pow]
    /-
      case pos
      f : Polynomial Int
      s : Complex
      this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
      c : Real
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
      hx1 : LE.le (HMul.hMul x (Complex.abs s)) 1
      ⊢ LE.le (HPow.hPow (HMul.hMul x (Complex.abs s)) (HSub.hSub p 1)) 1
    -/
    exact pow_le_one₀ (mul_nonneg hx.1.le (Complex.abs.nonneg _)) hx1
    /-
      🎉 no goals
    -/
    /-
      case neg
      f : Polynomial Int
      s : Complex
      this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
      c : Real
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
      hx1 : Not (LE.le (HMul.hMul x (Complex.abs s)) 1)
      ⊢ LE.le (HPow.hPow (HMul.hMul x (Complex.abs s)) (HSub.hSub p 1)) (HPow.hPow ( …
    -/
  · push_neg at hx1
    /-
      case neg
      f : Polynomial Int
      s : Complex
      this : Bornology.IsBounded (Set.image (fun x => HMul.hMul (Max.max (HMul.hMul  …
      c : Real
      p : Nat
      x : Real
      hx : Membership.mem (Set.Ioc 0 1) x
      h : LE.le (abs (HMul.hMul (Max.max (HMul.hMul x (Complex.abs s)) 1) (Complex.a …
      hx1 : LT.lt 1 (HMul.hMul x (Complex.abs s))
      ⊢ LE.le (HPow.hPow (HMul.hMul x (Complex.abs s)) (HSub.hSub p 1)) (HPow.hPow ( …
    -/
    exact pow_le_pow_right₀ hx1.le (Nat.sub_le _ _)
    /-
      🎉 no goals
    -/


/--
See equation (68), page 285 of [Jacobson, *Basic Algebra I, 4.12*][jacobson1974].

Given a polynomial `f` with integer coefficients, we can find a constant `c : ℝ` and for each prime
`p > |f₀|`, `nₚ : ℤ` and `gₚ : ℤ[X]` such that

* `p` does not divide `nₚ`
* `deg(gₚ) < p * deg(f)`
* all complex roots `r` of `f` satisfy `|nₚ * e ^ r - p * gₚ(r)| ≤ c ^ p / (p - 1)!`

In the proof of Lindemann-Weierstrass, we will take `f` to be a polynomial whose complex roots
are the algebraic numbers whose exponentials we want to prove to be linearly independent.

Note: Jacobson writes `Nₚ` for our `nₚ` and `M` for our `c` (modulo a constant factor).
-/
theorem exp_polynomial_approx (f : ℤ[X]) (hf : f.eval 0 ≠ 0) :
    ∃ c,
      ∀ p > (eval 0 f).natAbs, p.Prime →
        ∃ n : ℤ, ¬ ↑p ∣ n ∧ ∃ gp : ℤ[X], gp.natDegree ≤ p * f.natDegree - 1 ∧
          ∀ {r : ℂ}, r ∈ f.aroots ℂ →
            Complex.abs (n • exp r - p • aeval r gp : ℂ) ≤ c ^ p / (p - 1)! := by
  /-
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    ⊢ Exists fun c => ∀ (p : Nat), GT.gt p (Polynomial.eval 0 f).natAbs → Nat.Prim …
  -/
  simp_rw [nsmul_eq_mul, zsmul_eq_mul]
  /-
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    ⊢ Exists fun c => ∀ (p : Nat), GT.gt p (Polynomial.eval 0 f).natAbs → Nat.Prim …
  -/
  choose c' c'0 Pp'_le using exp_polynomial_approx_aux f
  use
    if h : ((f.aroots ℂ).map c').toFinset.Nonempty then ((f.aroots ℂ).map c').toFinset.max' h else 0
  /-
    case h
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    ⊢ ∀ (p : Nat), GT.gt p (Polynomial.eval 0 f).natAbs → Nat.Prime p → Exists fun …
  -/
  intro p p_gt prime_p
  /-
    case h
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    ⊢ Exists fun n => And (Not (Dvd.dvd (↑p) n)) (Exists fun gp => And (LE.le gp.n …
  -/
  obtain ⟨gp', -, h'⟩ := eval_sumIDeriv_of_pos (X ^ (p - 1) * f ^ p) prime_p.pos
  /-
    case h.intro.intro
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : ∀ (r : Int) {p' : Polynomial Int}, Eq (HMul.hMul (HPow.hPow Polynomial.X  …
    ⊢ Exists fun n => And (Not (Dvd.dvd (↑p) n)) (Exists fun gp => And (LE.le gp.n …
  -/
  specialize h' 0 (by rw [C_0, sub_zero])
  /-
    case h.intro.intro
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    ⊢ Exists fun n => And (Not (Dvd.dvd (↑p) n)) (Exists fun gp => And (LE.le gp.n …
  -/
  use f.eval 0 ^ p + p * gp'.eval 0
  /-
    case h
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    ⊢ And (Not (Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow (Polynomial.eval 0 f) p) (HMul. …
  -/
  constructor
    /-
      case h.left
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      ⊢ Not (Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow (Polynomial.eval 0 f) p) (HMul.hMul  …
    -/
  · rw [dvd_add_left (dvd_mul_right _ _)]
    /-
      case h.left
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      ⊢ Not (Dvd.dvd (↑p) (HPow.hPow (Polynomial.eval 0 f) p))
    -/
    contrapose! p_gt with h
    /-
      case h.left
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      h : Dvd.dvd (↑p) (HPow.hPow (Polynomial.eval 0 f) p)
      ⊢ LE.le p (Polynomial.eval 0 f).natAbs
    -/
    exact Nat.le_of_dvd (Int.natAbs_pos.mpr hf) (Int.natCast_dvd.mp (Int.Prime.dvd_pow' prime_p h))
    /-
      🎉 no goals
    -/
  /-
    case h.right
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub (HMul.hMul p f.natDegree …
  -/
  obtain ⟨gp, gp'_le, h⟩ := aeval_sumIDeriv ℂ (X ^ (p - 1) * f ^ p) p
  /-
    case h.right.intro.intro
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
    ⊢ Exists fun gp => And (LE.le gp.natDegree (HSub.hSub (HMul.hMul p f.natDegree …
  -/
  refine ⟨gp, ?_, ?_⟩
    /-
      case h.right.intro.intro.refine_1
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      gp : Polynomial Int
      gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
      h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
      ⊢ LE.le gp.natDegree (HSub.hSub (HMul.hMul p f.natDegree) 1)
    -/
  · refine gp'_le.trans ((tsub_le_tsub_right natDegree_mul_le p).trans ?_)
    rw [natDegree_X_pow, natDegree_pow, tsub_add_eq_add_tsub prime_p.one_le, tsub_right_comm,
      add_tsub_cancel_left]
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
    ⊢ ∀ {r : Complex}, Membership.mem (f.aroots Complex) r → LE.le (Complex.abs (H …
  -/
  intro r hr
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
    r : Complex
    hr : Membership.mem (f.aroots Complex) r
    ⊢ LE.le (Complex.abs (HSub.hSub (HMul.hMul (↑(HAdd.hAdd (HPow.hPow (Polynomial …
  -/
  specialize h r _
    /-
      case h.right.intro.intro.refine_2
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      gp : Polynomial Int
      gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
      h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
      r : Complex
      hr : Membership.mem (f.aroots Complex) r
      ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) p) (Polynomial. …
    -/
  · rw [mem_roots'] at hr
    /-
      case h.right.intro.intro.refine_2
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      gp : Polynomial Int
      gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
      h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
      r : Complex
      hr : And (Ne (Polynomial.map (algebraMap Int Complex) f) 0) ((Polynomial.map ( …
      ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) p) (Polynomial. …
    -/
    rw [Polynomial.map_mul, f.map_pow]
    /-
      case h.right.intro.intro.refine_2
      f : Polynomial Int
      hf : Ne (Polynomial.eval 0 f) 0
      c' : Complex → Real
      c'0 : ∀ (s : Complex), GE.ge (c' s) 0
      Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
      p : Nat
      p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
      prime_p : Nat.Prime p
      gp' : Polynomial Int
      h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
      gp : Polynomial Int
      gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
      h : ∀ (r : Complex), Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C  …
      r : Complex
      hr : And (Ne (Polynomial.map (algebraMap Int Complex) f) 0) ((Polynomial.map ( …
      ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C r)) p) (HMul.hMul ( …
    -/
    exact dvd_mul_of_dvd_right (pow_dvd_pow_of_dvd (dvd_iff_isRoot.mpr hr.2) _) _
    /-
      🎉 no goals
    -/
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    r : Complex
    hr : Membership.mem (f.aroots Complex) r
    h : Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polyn …
    ⊢ LE.le (Complex.abs (HSub.hSub (HMul.hMul (↑(HAdd.hAdd (HPow.hPow (Polynomial …
  -/
  rw [nsmul_eq_mul] at h
  have :
      (↑(eval 0 f ^ p + p * eval 0 gp') * cexp r - p * (aeval r) gp) * (p - 1)! =
      ((eval 0 f ^ p * cexp r) * (p - 1)! +
        ↑(p * (p - 1)!) * (eval 0 gp' * cexp r - (aeval r) gp)) := by
    push_cast; ring
  rw [le_div_iff₀ (Nat.cast_pos.mpr (Nat.factorial_pos _) : (0 : ℝ) < _), ← abs_natCast, ← map_mul,
    this, Nat.mul_factorial_pred prime_p.pos, mul_sub, ← h]
  have :
      ↑(eval 0 f) ^ p * cexp r * ↑(p - 1)! +
        (↑p ! * (↑(eval 0 gp') * cexp r) - (aeval r) (sumIDeriv (X ^ (p - 1) * f ^ p))) =
      ((p - 1)! • ↑(eval 0 (f ^ p)) + p ! • ↑(eval 0 gp') : ℤ) * cexp r -
        (aeval r) (sumIDeriv (X ^ (p - 1) * f ^ p)) := by
    simp; ring
  rw [this, ← h', mul_comm, ← eq_intCast (algebraMap ℤ ℂ),
    ← aeval_algebraMap_apply_eq_algebraMap_eval, map_zero,
    aeval_sumIDeriv_eq_eval, aeval_sumIDeriv_eq_eval, ← P]
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    r : Complex
    hr : Membership.mem (f.aroots Complex) r
    h : Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polyn …
    this✝ : Eq (HMul.hMul (HSub.hSub (HMul.hMul (↑(HAdd.hAdd (HPow.hPow (Polynomia …
    this : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (↑(Polynomial.eval 0 f)) …
    ⊢ LE.le (Complex.abs (LindemannWeierstrass.P (Polynomial.map (algebraMap Int C …
  -/
  refine (Pp'_le r p prime_p.ne_zero).trans (pow_le_pow_left₀ (c'0 r) ?_ _)
  have aux : c' r ∈ (Multiset.map c' (f.aroots ℂ)).toFinset := by
    simpa only [Multiset.mem_toFinset] using Multiset.mem_map_of_mem _ hr
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    r : Complex
    hr : Membership.mem (f.aroots Complex) r
    h : Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polyn …
    this✝ : Eq (HMul.hMul (HSub.hSub (HMul.hMul (↑(HAdd.hAdd (HPow.hPow (Polynomia …
    this : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (↑(Polynomial.eval 0 f)) …
    aux : Membership.mem (Multiset.map c' (f.aroots Complex)).toFinset (c' r)
    ⊢ LE.le (c' r) (dite (Multiset.map c' (f.aroots Complex)).toFinset.Nonempty (f …
  -/
  have h : ((f.aroots ℂ).map c').toFinset.Nonempty := ⟨c' r, aux⟩
  /-
    case h.right.intro.intro.refine_2
    f : Polynomial Int
    hf : Ne (Polynomial.eval 0 f) 0
    c' : Complex → Real
    c'0 : ∀ (s : Complex), GE.ge (c' s) 0
    Pp'_le : ∀ (s : Complex) (p : Nat), Ne p 0 → LE.le (Complex.abs (LindemannWeie …
    p : Nat
    p_gt : GT.gt p (Polynomial.eval 0 f).natAbs
    prime_p : Nat.Prime p
    gp' : Polynomial Int
    h' : Eq (Polynomial.eval 0 (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Polynom …
    gp : Polynomial Int
    gp'_le : LE.le gp.natDegree (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X (HSu …
    r : Complex
    hr : Membership.mem (f.aroots Complex) r
    h✝ : Eq ((Polynomial.aeval r) (Polynomial.sumIDeriv (HMul.hMul (HPow.hPow Poly …
    this✝ : Eq (HMul.hMul (HSub.hSub (HMul.hMul (↑(HAdd.hAdd (HPow.hPow (Polynomia …
    this : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (↑(Polynomial.eval 0 f)) …
    aux : Membership.mem (Multiset.map c' (f.aroots Complex)).toFinset (c' r)
    h : (Multiset.map c' (f.aroots Complex)).toFinset.Nonempty
    ⊢ LE.le (c' r) (dite (Multiset.map c' (f.aroots Complex)).toFinset.Nonempty (f …
  -/
  simpa only [h, ↓reduceDIte] using Finset.le_max' _ _ aux
  /-
    🎉 no goals
  -/


