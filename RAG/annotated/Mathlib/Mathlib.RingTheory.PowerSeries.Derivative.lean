/--
The formal derivative of a power series in one variable.
This is defined here as a function, but will be packaged as a
derivation `derivative` on `R⟦X⟧`.
-/
noncomputable def derivativeFun (f : R⟦X⟧) : R⟦X⟧ := mk fun n ↦ coeff R (n + 1) f * (n + 1)


theorem coeff_derivativeFun (f : R⟦X⟧) (n : ℕ) :
    coeff R n f.derivativeFun = coeff R (n + 1) f * (n + 1) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) f.derivativeFun) (HMul.hMul ((PowerSeries.coeff  …
  -/
  rw [derivativeFun, coeff_mk]
  /-
    🎉 no goals
  -/


theorem derivativeFun_coe (f : R[X]) : (f : R⟦X⟧).derivativeFun = derivative f := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : Polynomial R
    ⊢ Eq (↑f).derivativeFun ↑(Polynomial.derivative f)
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f : Polynomial R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) (↑f).derivativeFun) ((PowerSeries.coeff R n✝) ↑ …
  -/
  rw [coeff_derivativeFun, coeff_coe, coeff_coe, coeff_derivative]
  /-
    🎉 no goals
  -/


theorem derivativeFun_add (f g : R⟦X⟧) :
    derivativeFun (f + g) = derivativeFun f + derivativeFun g := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : PowerSeries R
    ⊢ Eq (HAdd.hAdd f g).derivativeFun (HAdd.hAdd f.derivativeFun g.derivativeFun)
  -/
  ext
  rw [coeff_derivativeFun, map_add, map_add, coeff_derivativeFun,
    coeff_derivativeFun, add_mul]


theorem derivativeFun_C (r : R) : derivativeFun (C R r) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq ((PowerSeries.C R) r).derivativeFun 0
  -/
  ext n
  -- Note that `map_zero` didn't get picked up, apparently due to a missing `FunLike.coe`
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    r : R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) ((PowerSeries.C R) r).derivativeFun) ((PowerSeri …
  -/
  rw [coeff_derivativeFun, coeff_succ_C, zero_mul, (coeff R n).map_zero]
  /-
    🎉 no goals
  -/


theorem trunc_derivativeFun (f : R⟦X⟧) (n : ℕ) :
    trunc n f.derivativeFun = derivative (trunc (n + 1) f) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    n : Nat
    ⊢ Eq (PowerSeries.trunc n f.derivativeFun) (Polynomial.derivative (PowerSeries …
  -/
  ext d
  /-
    case a
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    n d : Nat
    ⊢ Eq ((PowerSeries.trunc n f.derivativeFun).coeff d) ((Polynomial.derivative ( …
  -/
  rw [coeff_trunc]
  /-
    case a
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    n d : Nat
    ⊢ Eq (ite (LT.lt d n) ((PowerSeries.coeff R d) f.derivativeFun) 0) ((Polynomia …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      f : PowerSeries R
      n d : Nat
      h : LT.lt d n
      ⊢ Eq ((PowerSeries.coeff R d) f.derivativeFun) ((Polynomial.derivative (PowerS …
    -/
  · have : d + 1 < n + 1 := succ_lt_succ_iff.2 h
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      f : PowerSeries R
      n d : Nat
      h : LT.lt d n
      this : LT.lt (HAdd.hAdd d 1) (HAdd.hAdd n 1)
      ⊢ Eq ((PowerSeries.coeff R d) f.derivativeFun) ((Polynomial.derivative (PowerS …
    -/
    rw [coeff_derivativeFun, coeff_derivative, coeff_trunc, if_pos this]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      f : PowerSeries R
      n d : Nat
      h : Not (LT.lt d n)
      ⊢ Eq 0 ((Polynomial.derivative (PowerSeries.trunc (HAdd.hAdd n 1) f)).coeff d)
    -/
  · have : ¬d + 1 < n + 1 := by rwa [succ_lt_succ_iff]
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      f : PowerSeries R
      n d : Nat
      h : Not (LT.lt d n)
      this : Not (LT.lt (HAdd.hAdd d 1) (HAdd.hAdd n 1))
      ⊢ Eq 0 ((Polynomial.derivative (PowerSeries.trunc (HAdd.hAdd n 1) f)).coeff d)
    -/
    rw [coeff_derivative, coeff_trunc, if_neg this, zero_mul]
    /-
      🎉 no goals
    -/

--A special case of `derivativeFun_mul`, used in its proof.

private theorem derivativeFun_coe_mul_coe (f g : R[X]) : derivativeFun (f * g : R⟦X⟧) =
    f * derivative g + g * derivative f  := by
  rw [← coe_mul, derivativeFun_coe, derivative_mul,
    add_comm, mul_comm _ g, ← coe_mul, ← coe_mul, Polynomial.coe_add]


/-- **Leibniz rule for formal power series**. -/
theorem derivativeFun_mul (f g : R⟦X⟧) :
    derivativeFun (f * g) = f • g.derivativeFun + g • f.derivativeFun := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : PowerSeries R
    ⊢ Eq (HMul.hMul f g).derivativeFun (HAdd.hAdd (HSMul.hSMul f g.derivativeFun)  …
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f g : PowerSeries R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul f g).derivativeFun) ((PowerSeries.coe …
  -/
  have h₁ : n < n + 1 := lt_succ_self n
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f g : PowerSeries R
    n : Nat
    h₁ : LT.lt n (HAdd.hAdd n 1)
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul f g).derivativeFun) ((PowerSeries.coe …
  -/
  have h₂ : n < n + 1 + 1 := Nat.lt_add_right _ h₁
  rw [coeff_derivativeFun, map_add, coeff_mul_eq_coeff_trunc_mul_trunc _ _ (lt_succ_self _),
    smul_eq_mul, smul_eq_mul, coeff_mul_eq_coeff_trunc_mul_trunc₂ g f.derivativeFun h₂ h₁,
    coeff_mul_eq_coeff_trunc_mul_trunc₂ f g.derivativeFun h₂ h₁, trunc_derivativeFun,
    trunc_derivativeFun, ← map_add, ← derivativeFun_coe_mul_coe, coeff_derivativeFun]


theorem derivativeFun_one : derivativeFun (1 : R⟦X⟧) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (PowerSeries.derivativeFun 1) 0
  -/
  rw [← map_one (C R), derivativeFun_C (1 : R)]
  /-
    🎉 no goals
  -/


theorem derivativeFun_smul (r : R) (f : R⟦X⟧) : derivativeFun (r • f) = r • derivativeFun f := by
  rw [smul_eq_C_mul, smul_eq_C_mul, derivativeFun_mul, derivativeFun_C, smul_zero, add_zero,
    smul_eq_mul]


/-- The formal derivative of a formal power series -/
noncomputable def derivative : Derivation R R⟦X⟧ R⟦X⟧ where
  toFun             := derivativeFun
  map_add'          := derivativeFun_add
  map_smul'         := derivativeFun_smul
  map_one_eq_zero'  := derivativeFun_one
  leibniz'          := derivativeFun_mul

/-- Abbreviation of `PowerSeries.derivative`, the formal derivative on `R⟦X⟧` -/
scoped notation "d⁄dX" => derivative


@[simp] theorem derivative_C (r : R) : d⁄dX R (C R r) = 0 := derivativeFun_C r


theorem coeff_derivative (f : R⟦X⟧) (n : ℕ) :
    coeff R n (d⁄dX R f) = coeff R (n + 1) f * (n + 1) := coeff_derivativeFun f n


theorem derivative_coe (f : R[X]) : d⁄dX R f = Polynomial.derivative f := derivativeFun_coe f


@[simp] theorem derivative_X : d⁄dX R (X : R⟦X⟧) = 1 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq ((PowerSeries.derivative R) PowerSeries.X) 1
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ((PowerSeries.derivative R) PowerSeries.X)) ((P …
  -/
  rw [coeff_derivative, coeff_one, coeff_X, boole_mul]
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    ⊢ Eq (ite (Eq (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (↑n✝) 1) 0) (ite (Eq n✝ 0) 1 0)
  -/
  simp_rw [add_left_eq_self]
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    ⊢ Eq (ite (Eq n✝ 0) (HAdd.hAdd (↑n✝) 1) 0) (ite (Eq n✝ 0) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      n✝ : Nat
      h : Eq n✝ 0
      ⊢ Eq (HAdd.hAdd (↑n✝) 1) 1
    -/
  · rw [h, cast_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      n✝ : Nat
      h : Not (Eq n✝ 0)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem trunc_derivative (f : R⟦X⟧) (n : ℕ) :
    trunc n (d⁄dX R f) = Polynomial.derivative (trunc (n + 1) f) :=
  trunc_derivativeFun ..


theorem trunc_derivative' (f : R⟦X⟧) (n : ℕ) :
    trunc (n-1) (d⁄dX R f) = Polynomial.derivative (trunc n f) := by
  cases n with
  | zero =>
    simp
  | succ n =>
    rw [succ_sub_one, trunc_derivative]


/-- If `f` and `g` have the same constant term and derivative, then they are equal. -/
theorem derivative.ext {R} [CommRing R] [NoZeroSMulDivisors ℕ R] {f g} (hD : d⁄dX R f = d⁄dX R g)
    (hc : constantCoeff R f = constantCoeff R g) : f = g := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : NoZeroSMulDivisors Nat R
    f g : PowerSeries R
    hD : Eq ((PowerSeries.derivative R) f) ((PowerSeries.derivative R) g)
    hc : Eq ((PowerSeries.constantCoeff R) f) ((PowerSeries.constantCoeff R) g)
    ⊢ Eq f g
  -/
  ext n
  cases n with
  | zero =>
    rw [coeff_zero_eq_constantCoeff, hc]
  | succ n =>
    have equ : coeff R n (d⁄dX R f) = coeff R n (d⁄dX R g) := by rw [hD]
    rwa [coeff_derivative, coeff_derivative, ← cast_succ, mul_comm, ← nsmul_eq_mul,
      mul_comm, ← nsmul_eq_mul, smul_right_inj n.succ_ne_zero] at equ


@[simp] theorem derivative_inv {R} [CommRing R] (f : R⟦X⟧ˣ) :
    d⁄dX R ↑f⁻¹ = -(↑f⁻¹ : R⟦X⟧) ^ 2 * d⁄dX R f := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : Units (PowerSeries R)
    ⊢ Eq ((PowerSeries.derivative R) ↑(Inv.inv f)) (HMul.hMul (Neg.neg (HPow.hPow  …
  -/
  apply Derivation.leibniz_of_mul_eq_one
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : Units (PowerSeries R)
    ⊢ Eq (HMul.hMul ↑(Inv.inv f) ↑f) 1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] theorem derivative_invOf {R} [CommRing R] (f : R⟦X⟧) [Invertible f] :
    d⁄dX R ⅟f = - ⅟f ^ 2 * d⁄dX R f := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    f : PowerSeries R
    inst✝ : Invertible f
    ⊢ Eq ((PowerSeries.derivative R) (Invertible.invOf f)) (HMul.hMul (Neg.neg (HP …
  -/
  rw [Derivation.leibniz_invOf, smul_eq_mul]
  /-
    🎉 no goals
  -/

/-
The following theorem is stated only in the case that `R` is a field. This is because
there is currently no instance of `Inv R⟦X⟧` for more general base rings `R`.
-/

@[simp] theorem derivative_inv' {R} [Field R] (f : R⟦X⟧) : d⁄dX R f⁻¹ = -f⁻¹ ^ 2 * d⁄dX R f := by
  /-
    R : Type u_1
    inst✝ : Field R
    f : PowerSeries R
    ⊢ Eq ((PowerSeries.derivative R) (Inv.inv f)) (HMul.hMul (Neg.neg (HPow.hPow ( …
  -/
  by_cases h : constantCoeff R f = 0
  · suffices f⁻¹ = 0 by
      rw [this, pow_two, zero_mul, neg_zero, zero_mul, map_zero]
    /-
      case pos
      R : Type u_1
      inst✝ : Field R
      f : PowerSeries R
      h : Eq ((PowerSeries.constantCoeff R) f) 0
      ⊢ Eq (Inv.inv f) 0
    -/
    rwa [MvPowerSeries.inv_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Field R
    f : PowerSeries R
    h : Not (Eq ((PowerSeries.constantCoeff R) f) 0)
    ⊢ Eq ((PowerSeries.derivative R) (Inv.inv f)) (HMul.hMul (Neg.neg (HPow.hPow ( …
  -/
  apply Derivation.leibniz_of_mul_eq_one
  /-
    case neg.h
    R : Type u_1
    inst✝ : Field R
    f : PowerSeries R
    h : Not (Eq ((PowerSeries.constantCoeff R) f) 0)
    ⊢ Eq (HMul.hMul (Inv.inv f) f) 1
  -/
  exact PowerSeries.inv_mul_cancel (h := h)
  /-
    🎉 no goals
  -/


