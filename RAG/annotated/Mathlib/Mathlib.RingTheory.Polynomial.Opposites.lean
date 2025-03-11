/-- Ring isomorphism between `R[X]ᵐᵒᵖ` and `Rᵐᵒᵖ[X]` sending each coefficient of a polynomial
to the corresponding element of the opposite ring. -/
def opRingEquiv (R : Type*) [Semiring R] : R[X]ᵐᵒᵖ ≃+* Rᵐᵒᵖ[X] :=
  ((toFinsuppIso R).op.trans AddMonoidAlgebra.opRingEquiv).trans (toFinsuppIso _).symm


@[simp]
theorem opRingEquiv_op_monomial (n : ℕ) (r : R) :
    opRingEquiv R (op (monomial n r : R[X])) = monomial n (op r) := by
  simp only [opRingEquiv, RingEquiv.coe_trans, Function.comp_apply,
    AddMonoidAlgebra.opRingEquiv_apply, RingEquiv.op_apply_apply, toFinsuppIso_apply, unop_op,
    toFinsupp_monomial, Finsupp.mapRange_single, toFinsuppIso_symm_apply, ofFinsupp_single]


@[simp]
theorem opRingEquiv_op_C (a : R) : opRingEquiv R (op (C a)) = C (op a) :=
  opRingEquiv_op_monomial 0 a


@[simp]
theorem opRingEquiv_op_X : opRingEquiv R (op (X : R[X])) = X :=
  opRingEquiv_op_monomial 1 1


theorem opRingEquiv_op_C_mul_X_pow (r : R) (n : ℕ) :
    opRingEquiv R (op (C r * X ^ n : R[X])) = C (op r) * X ^ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    n : Nat
    ⊢ Eq ((Polynomial.opRingEquiv R) (MulOpposite.op (HMul.hMul (Polynomial.C r) ( …
  -/
  simp only [X_pow_mul, op_mul, op_pow, map_mul, map_pow, opRingEquiv_op_X, opRingEquiv_op_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem opRingEquiv_symm_monomial (n : ℕ) (r : Rᵐᵒᵖ) :
    (opRingEquiv R).symm (monomial n r) = op (monomial n (unop r)) :=
                                /-
                                  R : Type u_1
                                  inst✝ : Semiring R
                                  n : Nat
                                  r : MulOpposite R
                                  ⊢ Eq ((Polynomial.opRingEquiv R) ((Polynomial.opRingEquiv R).symm ((Polynomial …
                                -/
  (opRingEquiv R).injective (by simp)
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem opRingEquiv_symm_C (a : Rᵐᵒᵖ) : (opRingEquiv R).symm (C a) = op (C (unop a)) :=
  opRingEquiv_symm_monomial 0 a


@[simp]
theorem opRingEquiv_symm_X : (opRingEquiv R).symm (X : Rᵐᵒᵖ[X]) = op X :=
  opRingEquiv_symm_monomial 1 1


theorem opRingEquiv_symm_C_mul_X_pow (r : Rᵐᵒᵖ) (n : ℕ) :
    (opRingEquiv R).symm (C r * X ^ n : Rᵐᵒᵖ[X]) = op (C (unop r) * X ^ n) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : MulOpposite R
    n : Nat
    ⊢ Eq ((Polynomial.opRingEquiv R).symm (HMul.hMul (Polynomial.C r) (HPow.hPow P …
  -/
  rw [C_mul_X_pow_eq_monomial, opRingEquiv_symm_monomial, C_mul_X_pow_eq_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_opRingEquiv (p : R[X]ᵐᵒᵖ) (n : ℕ) :
    (opRingEquiv R p).coeff n = op ((unop p).coeff n) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : MulOpposite (Polynomial R)
    n : Nat
    ⊢ Eq (((Polynomial.opRingEquiv R) p).coeff n) (MulOpposite.op ((MulOpposite.un …
  -/
  induction' p with p
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    p : Polynomial R
    ⊢ Eq (((Polynomial.opRingEquiv R) (MulOpposite.op p)).coeff n) (MulOpposite.op …
  -/
  cases p
  /-
    case h.ofFinsupp
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (((Polynomial.opRingEquiv R) (MulOpposite.op { toFinsupp := toFinsupp✝ }) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem support_opRingEquiv (p : R[X]ᵐᵒᵖ) : (opRingEquiv R p).support = (unop p).support := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : MulOpposite (Polynomial R)
    ⊢ Eq ((Polynomial.opRingEquiv R) p).support (MulOpposite.unop p).support
  -/
  induction' p with p
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((Polynomial.opRingEquiv R) (MulOpposite.op p)).support (MulOpposite.unop …
  -/
  cases p
  /-
    case h.ofFinsupp
    R : Type u_1
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((Polynomial.opRingEquiv R) (MulOpposite.op { toFinsupp := toFinsupp✝ })) …
  -/
  exact Finsupp.support_mapRange_of_injective (map_zero _) _ op_injective
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_opRingEquiv (p : R[X]ᵐᵒᵖ) : (opRingEquiv R p).natDegree = (unop p).natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : MulOpposite (Polynomial R)
    ⊢ Eq ((Polynomial.opRingEquiv R) p).natDegree (MulOpposite.unop p).natDegree
  -/
  by_cases p0 : p = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : MulOpposite (Polynomial R)
      p0 : Eq p 0
      ⊢ Eq ((Polynomial.opRingEquiv R) p).natDegree (MulOpposite.unop p).natDegree
    -/
  · simp only [p0, _root_.map_zero, natDegree_zero, unop_zero]
    /-
      🎉 no goals
    -/
  · simp only [p0, natDegree_eq_support_max', Ne, EmbeddingLike.map_eq_zero_iff, not_false_iff,
      support_opRingEquiv, unop_eq_zero_iff]


@[simp]
theorem leadingCoeff_opRingEquiv (p : R[X]ᵐᵒᵖ) :
    (opRingEquiv R p).leadingCoeff = op (unop p).leadingCoeff := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : MulOpposite (Polynomial R)
    ⊢ Eq ((Polynomial.opRingEquiv R) p).leadingCoeff (MulOpposite.op (MulOpposite. …
  -/
  rw [leadingCoeff, coeff_opRingEquiv, natDegree_opRingEquiv, leadingCoeff]
  /-
    🎉 no goals
  -/


