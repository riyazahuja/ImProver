/-- (Implementation detail).
The function underlying `A ⊗[R] R[X] →ₐ[R] A[X]`,
as a bilinear function of two arguments.
-/
-- Porting note: was  `@[simps apply_apply]`
@[simps! apply_apply]
def toFunBilinear : A →ₗ[A] R[X] →ₗ[R] A[X] :=
  LinearMap.toSpanSingleton A _ (aeval (Polynomial.X : A[X])).toLinearMap


theorem toFunBilinear_apply_eq_sum (a : A) (p : R[X]) :
    toFunBilinear R A a p = p.sum fun n r => monomial n (a * algebraMap R A r) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : A
    p : Polynomial R
    ⊢ Eq (((PolyEquivTensor.toFunBilinear R A) a) p) (p.sum fun n r => (Polynomial …
  -/
  simp only [toFunBilinear_apply_apply, aeval_def, eval₂_eq_sum, Polynomial.sum, Finset.smul_sum]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : A
    p : Polynomial R
    ⊢ Eq (p.support.sum fun x => HSMul.hSMul a (HMul.hMul ((algebraMap R (Polynomi …
  -/
  congr with i : 1
  rw [← Algebra.smul_def, ← C_mul', mul_smul_comm, C_mul_X_pow_eq_monomial, ← Algebra.commutes,
    ← Algebra.smul_def, smul_monomial]


/-- (Implementation detail).
The function underlying `A ⊗[R] R[X] →ₐ[R] A[X]`,
as a linear map.
-/
def toFunLinear : A ⊗[R] R[X] →ₗ[R] A[X] :=
  TensorProduct.lift (toFunBilinear R A)


@[simp]
theorem toFunLinear_tmul_apply (a : A) (p : R[X]) :
    toFunLinear R A (a ⊗ₜ[R] p) = toFunBilinear R A a p :=
  rfl

-- We apparently need to provide the decidable instance here
-- in order to successfully rewrite by this lemma.

theorem toFunLinear_mul_tmul_mul_aux_1 (p : R[X]) (k : ℕ) (h : Decidable ¬p.coeff k = 0) (a : A) :
    ite (¬coeff p k = 0) (a * (algebraMap R A) (coeff p k)) 0 =
                                           /-
                                             R : Type u_1
                                             A : Type u_2
                                             inst✝² : CommSemiring R
                                             inst✝¹ : Semiring A
                                             inst✝ : Algebra R A
                                             p : Polynomial R
                                             k : Nat
                                             h : Decidable (Not (Eq (p.coeff k) 0))
                                             a : A
                                             ⊢ Eq (ite (Not (Eq (p.coeff k) 0)) (HMul.hMul a ((algebraMap R A) (p.coeff k)) …
                                           -/
    a * (algebraMap R A) (coeff p k) := by classical split_ifs <;> simp [*]
                                           /-
                                             🎉 no goals
                                           -/


theorem toFunLinear_mul_tmul_mul_aux_2 (k : ℕ) (a₁ a₂ : A) (p₁ p₂ : R[X]) :
    a₁ * a₂ * (algebraMap R A) ((p₁ * p₂).coeff k) =
      (Finset.antidiagonal k).sum fun x =>
        a₁ * (algebraMap R A) (coeff p₁ x.1) * (a₂ * (algebraMap R A) (coeff p₂ x.2)) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    k : Nat
    a₁ a₂ : A
    p₁ p₂ : Polynomial R
    ⊢ Eq (HMul.hMul (HMul.hMul a₁ a₂) ((algebraMap R A) ((HMul.hMul p₁ p₂).coeff k …
  -/
  simp_rw [mul_assoc, Algebra.commutes, ← Finset.mul_sum, mul_assoc, ← Finset.mul_sum]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    k : Nat
    a₁ a₂ : A
    p₁ p₂ : Polynomial R
    ⊢ Eq (HMul.hMul a₁ (HMul.hMul a₂ ((algebraMap R A) ((HMul.hMul p₁ p₂).coeff k) …
  -/
  congr
  /-
    case e_a.e_a
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    k : Nat
    a₁ a₂ : A
    p₁ p₂ : Polynomial R
    ⊢ Eq ((algebraMap R A) ((HMul.hMul p₁ p₂).coeff k)) ((Finset.HasAntidiagonal.a …
  -/
  simp_rw [Algebra.commutes (coeff p₂ _), coeff_mul, map_sum, RingHom.map_mul]
  /-
    🎉 no goals
  -/


theorem toFunLinear_mul_tmul_mul (a₁ a₂ : A) (p₁ p₂ : R[X]) :
    (toFunLinear R A) ((a₁ * a₂) ⊗ₜ[R] (p₁ * p₂)) =
      (toFunLinear R A) (a₁ ⊗ₜ[R] p₁) * (toFunLinear R A) (a₂ ⊗ₜ[R] p₂) := by
  classical
    simp only [toFunLinear_tmul_apply, toFunBilinear_apply_eq_sum]
    ext k
    simp_rw [coeff_sum, coeff_monomial, sum_def, Finset.sum_ite_eq', mem_support_iff, Ne]
    conv_rhs => rw [coeff_mul]
    simp_rw [finset_sum_coeff, coeff_monomial, Finset.sum_ite_eq', mem_support_iff, Ne, mul_ite,
      mul_zero, ite_mul, zero_mul]
    simp_rw [← ite_zero_mul (¬coeff p₁ _ = 0) (a₁ * (algebraMap R A) (coeff p₁ _))]
    simp_rw [← mul_ite_zero (¬coeff p₂ _ = 0) _ (_ * _)]
    simp_rw [toFunLinear_mul_tmul_mul_aux_1, toFunLinear_mul_tmul_mul_aux_2]


theorem toFunLinear_one_tmul_one :
    toFunLinear R A (1 ⊗ₜ[R] 1) = 1 := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq ((PolyEquivTensor.toFunLinear R A) (TensorProduct.tmul R 1 1)) 1
  -/
  rw [toFunLinear_tmul_apply, toFunBilinear_apply_apply, Polynomial.aeval_one, one_smul]
  /-
    🎉 no goals
  -/


/-- (Implementation detail).
The algebra homomorphism `A ⊗[R] R[X] →ₐ[R] A[X]`.
-/
def toFunAlgHom : A ⊗[R] R[X] →ₐ[R] A[X] :=
  algHomOfLinearMapTensorProduct (toFunLinear R A) (toFunLinear_mul_tmul_mul R A)
    (toFunLinear_one_tmul_one R A)


@[simp]
theorem toFunAlgHom_apply_tmul (a : A) (p : R[X]) :
    toFunAlgHom R A (a ⊗ₜ[R] p) = p.sum fun n r => monomial n (a * (algebraMap R A) r) :=
  toFunBilinear_apply_eq_sum R A _ _


/-- (Implementation detail.)

The bare function `A[X] → A ⊗[R] R[X]`.
(We don't need to show that it's an algebra map, thankfully --- just that it's an inverse.)
-/
def invFun (p : A[X]) : A ⊗[R] R[X] :=
  p.eval₂ (includeLeft : A →ₐ[R] A ⊗[R] R[X]) ((1 : A) ⊗ₜ[R] (X : R[X]))


@[simp]
theorem invFun_add {p q} : invFun R A (p + q) = invFun R A p + invFun R A q := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p q : Polynomial A
    ⊢ Eq (PolyEquivTensor.invFun R A (HAdd.hAdd p q)) (HAdd.hAdd (PolyEquivTensor. …
  -/
  simp only [invFun, eval₂_add]
  /-
    🎉 no goals
  -/


theorem invFun_monomial (n : ℕ) (a : A) :
    invFun R A (monomial n a) = (a ⊗ₜ[R] 1) * 1 ⊗ₜ[R] X ^ n :=
  eval₂_monomial _ _


theorem left_inv (x : A ⊗ R[X]) : invFun R A ((toFunAlgHom R A) x) = x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : TensorProduct R A (Polynomial R)
    ⊢ Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) x)) x
  -/
  refine TensorProduct.induction_on x ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      ⊢ Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) 0)) 0
    -/
  · simp [invFun]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      ⊢ ∀ (x : A) (y : Polynomial R), Eq (PolyEquivTensor.invFun R A ((PolyEquivTens …
    -/
  · intro a p
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      a : A
      p : Polynomial R
      ⊢ Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) (TensorPro …
    -/
    dsimp only [invFun]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      a : A
      p : Polynomial R
      ⊢ Eq (Polynomial.eval₂ (↑Algebra.TensorProduct.includeLeft) (TensorProduct.tmu …
    -/
    rw [toFunAlgHom_apply_tmul, eval₂_sum]
    simp_rw [eval₂_monomial, AlgHom.coe_toRingHom, Algebra.TensorProduct.tmul_pow, one_pow,
      Algebra.TensorProduct.includeLeft_apply, Algebra.TensorProduct.tmul_mul_tmul, mul_one,
      one_mul, ← Algebra.commutes, ← Algebra.smul_def, smul_tmul, sum_def, ← tmul_sum]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      a : A
      p : Polynomial R
      ⊢ Eq (TensorProduct.tmul R a (p.support.sum fun a => HSMul.hSMul (p.coeff a) ( …
    -/
    conv_rhs => rw [← sum_C_mul_X_pow_eq p]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      a : A
      p : Polynomial R
      ⊢ Eq (TensorProduct.tmul R a (p.support.sum fun a => HSMul.hSMul (p.coeff a) ( …
    -/
    simp only [Algebra.smul_def]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      a : A
      p : Polynomial R
      ⊢ Eq (TensorProduct.tmul R a (p.support.sum fun x => HMul.hMul ((algebraMap R  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : TensorProduct R A (Polynomial R)
      ⊢ ∀ (x y : TensorProduct R A (Polynomial R)), Eq (PolyEquivTensor.invFun R A ( …
    -/
  · intro p q hp hq
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x p q : TensorProduct R A (Polynomial R)
      hp : Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) p)) p
      hq : Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) q)) q
      ⊢ Eq (PolyEquivTensor.invFun R A ((PolyEquivTensor.toFunAlgHom R A) (HAdd.hAdd …
    -/
    simp only [map_add, invFun_add, hp, hq]
    /-
      🎉 no goals
    -/


theorem right_inv (x : A[X]) : (toFunAlgHom R A) (invFun R A x) = x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : Polynomial A
    ⊢ Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTensor.invFun R A x)) x
  -/
  refine Polynomial.induction_on' x ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : Polynomial A
      ⊢ ∀ (p q : Polynomial A), Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTens …
    -/
  · intro p q hp hq
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x p q : Polynomial A
      hp : Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTensor.invFun R A p)) p
      hq : Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTensor.invFun R A q)) q
      ⊢ Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTensor.invFun R A (HAdd.hAdd …
    -/
    simp only [invFun_add, map_add, hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : Polynomial A
      ⊢ ∀ (n : Nat) (a : A), Eq ((PolyEquivTensor.toFunAlgHom R A) (PolyEquivTensor. …
    -/
  · intro n a
    rw [invFun_monomial, Algebra.TensorProduct.tmul_pow,
        one_pow, Algebra.TensorProduct.tmul_mul_tmul, mul_one, one_mul, toFunAlgHom_apply_tmul,
        X_pow_eq_monomial, sum_monomial_index] <;>
      /-
        case refine_2
        R : Type u_1
        A : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        x : Polynomial A
        n : Nat
        a : A
        ⊢ Eq ((Polynomial.monomial n) (HMul.hMul a ((algebraMap R A) 1))) ((Polynomial …
      -/
      /-
        🎉 no goals
      -/
      simp
      /-
        🎉 no goals
      -/


/-- (Implementation detail)

The equivalence, ignoring the algebra structure, `(A ⊗[R] R[X]) ≃ A[X]`.
-/
def equiv : A ⊗[R] R[X] ≃ A[X] where
  toFun := toFunAlgHom R A
  invFun := invFun R A
  left_inv := left_inv R A
  right_inv := right_inv R A


/-- The `R`-algebra isomorphism `A[X] ≃ₐ[R] (A ⊗[R] R[X])`.
-/
def polyEquivTensor : A[X] ≃ₐ[R] A ⊗[R] R[X] :=
  AlgEquiv.symm { PolyEquivTensor.toFunAlgHom R A, PolyEquivTensor.equiv R A with }


@[simp]
theorem polyEquivTensor_apply (p : A[X]) :
    polyEquivTensor R A p =
      p.eval₂ (includeLeft : A →ₐ[R] A ⊗[R] R[X]) ((1 : A) ⊗ₜ[R] (X : R[X])) :=
  rfl


@[simp]
theorem polyEquivTensor_symm_apply_tmul (a : A) (p : R[X]) :
    (polyEquivTensor R A).symm (a ⊗ₜ p) = p.sum fun n r => monomial n (a * algebraMap R A r) :=
  toFunAlgHom_apply_tmul _ _ _ _


/--
The algebra isomorphism stating "matrices of polynomials are the same as polynomials of matrices".

(You probably shouldn't attempt to use this underlying definition ---
it's an algebra equivalence, and characterised extensionally by the lemma
`matPolyEquiv_coeff_apply` below.)
-/
noncomputable def matPolyEquiv : Matrix n n R[X] ≃ₐ[R] (Matrix n n R)[X] :=
  ((matrixEquivTensor R R[X] n).trans (Algebra.TensorProduct.comm R _ _)).trans
    (polyEquivTensor R (Matrix n n R)).symm


@[simp] theorem matPolyEquiv_symm_C (M : Matrix n n R) : matPolyEquiv.symm (C M) = M.map C := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (matPolyEquiv.symm (Polynomial.C M)) (M.map ⇑Polynomial.C)
  -/
  simp [matPolyEquiv, ← C_eq_algebraMap]
  /-
    🎉 no goals
  -/


@[simp] theorem matPolyEquiv_map_C (M : Matrix n n R) : matPolyEquiv (M.map C) = C M := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (matPolyEquiv (M.map ⇑Polynomial.C)) (Polynomial.C M)
  -/
  rw [← matPolyEquiv_symm_C, AlgEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp] theorem matPolyEquiv_symm_X :
    matPolyEquiv.symm X = diagonal fun _ : n => (X : R[X]) := by
  suffices (Matrix.map 1 fun x ↦ X * algebraMap R R[X] x) = diagonal fun _ : n => (X : R[X]) by
    simpa [matPolyEquiv]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    ⊢ Eq (Matrix.map 1 fun x => HMul.hMul Polynomial.X ((algebraMap R (Polynomial  …
  -/
  rw [← Matrix.diagonal_one]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    ⊢ Eq ((Matrix.diagonal fun x => 1).map fun x => HMul.hMul Polynomial.X ((algeb …
  -/
  simp [-Matrix.diagonal_one]
  /-
    🎉 no goals
  -/


@[simp] theorem matPolyEquiv_diagonal_X :
    matPolyEquiv (diagonal fun _ : n => (X : R[X])) = X := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    ⊢ Eq (matPolyEquiv (Matrix.diagonal fun x => Polynomial.X)) Polynomial.X
  -/
  rw [← matPolyEquiv_symm_X, AlgEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


unseal Algebra.TensorProduct.mul in
theorem matPolyEquiv_coeff_apply_aux_1 (i j : n) (k : ℕ) (x : R) :
    matPolyEquiv (stdBasisMatrix i j <| monomial k x) = monomial k (stdBasisMatrix i j x) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i j : n
    k : Nat
    x : R
    ⊢ Eq (matPolyEquiv (Matrix.stdBasisMatrix i j ((Polynomial.monomial k) x))) (( …
  -/
  simp only [matPolyEquiv, AlgEquiv.trans_apply, matrixEquivTensor_apply_stdBasisMatrix]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i j : n
    k : Nat
    x : R
    ⊢ Eq ((polyEquivTensor R (Matrix n n R)).symm ((Algebra.TensorProduct.comm R ( …
  -/
  apply (polyEquivTensor R (Matrix n n R)).injective
  simp only [AlgEquiv.apply_symm_apply,Algebra.TensorProduct.comm_tmul,
    polyEquivTensor_apply, eval₂_monomial]
  simp only [Algebra.TensorProduct.tmul_mul_tmul, one_pow, one_mul, Matrix.mul_one,
    Algebra.TensorProduct.tmul_pow, Algebra.TensorProduct.includeLeft_apply]
  /-
    case a
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i j : n
    k : Nat
    x : R
    ⊢ Eq (TensorProduct.tmul R (Matrix.stdBasisMatrix i j 1) ((Polynomial.monomial …
  -/
  rw [← smul_X_eq_monomial, ← TensorProduct.smul_tmul]
  /-
    case a
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i j : n
    k : Nat
    x : R
    ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul x (Matrix.stdBasisMatrix i j 1)) (HPow …
  -/
                    /-
                      🎉 no goals
                    -/
  congr with i' <;> simp [stdBasisMatrix]
                    /-
                      🎉 no goals
                    -/


theorem matPolyEquiv_coeff_apply_aux_2 (i j : n) (p : R[X]) (k : ℕ) :
    coeff (matPolyEquiv (stdBasisMatrix i j p)) k = stdBasisMatrix i j (coeff p k) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i j : n
    p : Polynomial R
    k : Nat
    ⊢ Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j p)).coeff k) (Matrix.stdBasisMa …
  -/
  refine Polynomial.induction_on' p ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p : Polynomial R
      k : Nat
      ⊢ ∀ (p q : Polynomial R), Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j p)).coe …
    -/
  · intro p q hp hq
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p✝ : Polynomial R
      k : Nat
      p q : Polynomial R
      hp : Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j p)).coeff k) (Matrix.stdBasi …
      hq : Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j q)).coeff k) (Matrix.stdBasi …
      ⊢ Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j (HAdd.hAdd p q))).coeff k) (Mat …
    -/
    ext
    /-
      case refine_1.a
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p✝ : Polynomial R
      k : Nat
      p q : Polynomial R
      hp : Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j p)).coeff k) (Matrix.stdBasi …
      hq : Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j q)).coeff k) (Matrix.stdBasi …
      i✝ j✝ : n
      ⊢ Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j (HAdd.hAdd p q))).coeff k i✝ j✝ …
    -/
    simp [hp, hq, coeff_add, DMatrix.add_apply, stdBasisMatrix_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p : Polynomial R
      k : Nat
      ⊢ ∀ (n_1 : Nat) (a : R), Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j ((Polyno …
    -/
  · intro k x
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p : Polynomial R
      k✝ k : Nat
      x : R
      ⊢ Eq ((matPolyEquiv (Matrix.stdBasisMatrix i j ((Polynomial.monomial k) x))).c …
    -/
    simp only [matPolyEquiv_coeff_apply_aux_1, coeff_monomial]
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      i j : n
      p : Polynomial R
      k✝ k : Nat
      x : R
      ⊢ Eq (ite (Eq k k✝) (Matrix.stdBasisMatrix i j x) 0) (Matrix.stdBasisMatrix i  …
    -/
    split_ifs <;>
        /-
          case pos
          R : Type u_1
          inst✝² : CommSemiring R
          n : Type w
          inst✝¹ : DecidableEq n
          inst✝ : Fintype n
          i j : n
          p : Polynomial R
          k✝ k : Nat
          x : R
          h✝ : Eq k k✝
          ⊢ Eq (Matrix.stdBasisMatrix i j x) (Matrix.stdBasisMatrix i j x)
        -/
        /-
          case pos.h.h
          R : Type u_1
          inst✝² : CommSemiring R
          n : Type w
          inst✝¹ : DecidableEq n
          inst✝ : Fintype n
          i j : n
          p : Polynomial R
          k✝ k : Nat
          x : R
          h✝ : Eq k k✝
          x✝¹ x✝ : n
          ⊢ Eq (Matrix.stdBasisMatrix i j x x✝¹ x✝) (Matrix.stdBasisMatrix i j x x✝¹ x✝)
        -/
        /-
          🎉 no goals
        -/
        /-
          case neg.h.h
          R : Type u_1
          inst✝² : CommSemiring R
          n : Type w
          inst✝¹ : DecidableEq n
          inst✝ : Fintype n
          i j : n
          p : Polynomial R
          k✝ k : Nat
          x : R
          h✝ : Not (Eq k k✝)
          x✝¹ x✝ : n
          ⊢ Eq (0 x✝¹ x✝) (Matrix.stdBasisMatrix i j 0 x✝¹ x✝)
        -/
        simp
        /-
          🎉 no goals
        -/


@[simp]
theorem matPolyEquiv_coeff_apply (m : Matrix n n R[X]) (k : ℕ) (i j : n) :
    coeff (matPolyEquiv m) k i j = coeff (m i j) k := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    k : Nat
    i j : n
    ⊢ Eq ((matPolyEquiv m).coeff k i j) ((m i j).coeff k)
  -/
  refine Matrix.induction_on' m ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j : n
      ⊢ Eq ((matPolyEquiv 0).coeff k i j) ((0 i j).coeff k)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j : n
      ⊢ ∀ (p q : Matrix n n (Polynomial R)), Eq ((matPolyEquiv p).coeff k i j) ((p i …
    -/
  · intro p q hp hq
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j : n
      p q : Matrix n n (Polynomial R)
      hp : Eq ((matPolyEquiv p).coeff k i j) ((p i j).coeff k)
      hq : Eq ((matPolyEquiv q).coeff k i j) ((q i j).coeff k)
      ⊢ Eq ((matPolyEquiv (HAdd.hAdd p q)).coeff k i j) ((HAdd.hAdd p q i j).coeff k)
    -/
    simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j : n
      ⊢ ∀ (i_1 j_1 : n) (x : Polynomial R), Eq ((matPolyEquiv (Matrix.stdBasisMatrix …
    -/
  · intro i' j' x
    /-
      case refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j i' j' : n
      x : Polynomial R
      ⊢ Eq ((matPolyEquiv (Matrix.stdBasisMatrix i' j' x)).coeff k i j) ((Matrix.std …
    -/
    rw [matPolyEquiv_coeff_apply_aux_2]
    /-
      case refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j i' j' : n
      x : Polynomial R
      ⊢ Eq (Matrix.stdBasisMatrix i' j' (x.coeff k) i j) ((Matrix.stdBasisMatrix i'  …
    -/
    dsimp [stdBasisMatrix]
    /-
      case refine_3
      R : Type u_1
      inst✝² : CommSemiring R
      n : Type w
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      m : Matrix n n (Polynomial R)
      k : Nat
      i j i' j' : n
      x : Polynomial R
      ⊢ Eq (ite (And (Eq i' i) (Eq j' j)) (x.coeff k) 0) ((ite (And (Eq i' i) (Eq j' …
    -/
    split_ifs <;> rename_i h
      /-
        case pos
        R : Type u_1
        inst✝² : CommSemiring R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        m : Matrix n n (Polynomial R)
        k : Nat
        i j i' j' : n
        x : Polynomial R
        h : And (Eq i' i) (Eq j' j)
        ⊢ Eq (x.coeff k) (x.coeff k)
      -/
    · rcases h with ⟨rfl, rfl⟩
      /-
        case pos.intro
        R : Type u_1
        inst✝² : CommSemiring R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        m : Matrix n n (Polynomial R)
        k : Nat
        i' j' : n
        x : Polynomial R
        ⊢ Eq (x.coeff k) (x.coeff k)
      -/
      simp [stdBasisMatrix]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝² : CommSemiring R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        m : Matrix n n (Polynomial R)
        k : Nat
        i j i' j' : n
        x : Polynomial R
        h : Not (And (Eq i' i) (Eq j' j))
        ⊢ Eq 0 (Polynomial.coeff 0 k)
      -/
    · simp [stdBasisMatrix, h]
      /-
        🎉 no goals
      -/


@[simp]
theorem matPolyEquiv_symm_apply_coeff (p : (Matrix n n R)[X]) (i j : n) (k : ℕ) :
    coeff (matPolyEquiv.symm p i j) k = coeff p k i j := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial (Matrix n n R)
    i j : n
    k : Nat
    ⊢ Eq ((matPolyEquiv.symm p i j).coeff k) (p.coeff k i j)
  -/
  have t : p = matPolyEquiv (matPolyEquiv.symm p) := by simp
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial (Matrix n n R)
    i j : n
    k : Nat
    t : Eq p (matPolyEquiv (matPolyEquiv.symm p))
    ⊢ Eq ((matPolyEquiv.symm p i j).coeff k) (p.coeff k i j)
  -/
  conv_rhs => rw [t]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial (Matrix n n R)
    i j : n
    k : Nat
    t : Eq p (matPolyEquiv (matPolyEquiv.symm p))
    ⊢ Eq ((matPolyEquiv.symm p i j).coeff k) ((matPolyEquiv (matPolyEquiv.symm p)) …
  -/
  simp only [matPolyEquiv_coeff_apply]
  /-
    🎉 no goals
  -/


theorem matPolyEquiv_smul_one (p : R[X]) :
    matPolyEquiv (p • (1 : Matrix n n R[X])) = p.map (algebraMap R (Matrix n n R)) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial R
    ⊢ Eq (matPolyEquiv (HSMul.hSMul p 1)) (Polynomial.map (algebraMap R (Matrix n  …
  -/
  ext m i j
  simp only [matPolyEquiv_coeff_apply, smul_apply, one_apply, smul_eq_mul, mul_ite, mul_one,
    mul_zero, coeff_map, algebraMap_matrix_apply, Algebra.id.map_eq_id, RingHom.id_apply]
  /-
    case a.a
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial R
    m : Nat
    i j : n
    ⊢ Eq ((ite (Eq i j) p 0).coeff m) (ite (Eq i j) (p.coeff m) 0)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


@[simp]
lemma matPolyEquiv_map_smul (p : R[X]) (M : Matrix n n R[X]) :
    matPolyEquiv (p • M) = p.map (algebraMap _ _) * matPolyEquiv M := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    p : Polynomial R
    M : Matrix n n (Polynomial R)
    ⊢ Eq (matPolyEquiv (HSMul.hSMul p M)) (HMul.hMul (Polynomial.map (algebraMap R …
  -/
  rw [← one_mul M, ← smul_mul_assoc, _root_.map_mul, matPolyEquiv_smul_one, one_mul]
  /-
    🎉 no goals
  -/


theorem support_subset_support_matPolyEquiv (m : Matrix n n R[X]) (i j : n) :
    support (m i j) ⊆ support (matPolyEquiv m) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    ⊢ HasSubset.Subset (m i j).support (matPolyEquiv m).support
  -/
  intro k
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    k : Nat
    ⊢ Membership.mem (m i j).support k → Membership.mem (matPolyEquiv m).support k
  -/
  contrapose
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    k : Nat
    ⊢ Not (Membership.mem (matPolyEquiv m).support k) → Not (Membership.mem (m i j …
  -/
  simp only [not_mem_support_iff]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    k : Nat
    ⊢ Eq ((matPolyEquiv m).coeff k) 0 → Eq ((m i j).coeff k) 0
  -/
  intro hk
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    k : Nat
    hk : Eq ((matPolyEquiv m).coeff k) 0
    ⊢ Eq ((m i j).coeff k) 0
  -/
  rw [← matPolyEquiv_coeff_apply, hk]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    m : Matrix n n (Polynomial R)
    i j : n
    k : Nat
    hk : Eq ((matPolyEquiv m).coeff k) 0
    ⊢ Eq (0 i j) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Extend a ring hom `A → Mₙ(R)` to a ring hom `A[X] → Mₙ(R[X])`. -/
def RingHom.polyToMatrix (f : A →+* Matrix n n R) : A[X] →+* Matrix n n R[X] :=
  matPolyEquiv.symm.toRingHom.comp (mapRingHom f)


lemma evalRingHom_mapMatrix_comp_polyToMatrix :
    (evalRingHom 0).mapMatrix.comp f.polyToMatrix = f.comp (evalRingHom 0) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    n : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    S : Type u_3
    inst✝ : CommSemiring S
    f : RingHom S (Matrix n n R)
    ⊢ Eq ((Polynomial.evalRingHom 0).mapMatrix.comp f.polyToMatrix) (f.comp (Polyn …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [RingHom.polyToMatrix, ← AlgEquiv.symm_toRingEquiv, diagonal, apply_ite]
          /-
            🎉 no goals
          -/


lemma evalRingHom_mapMatrix_comp_compRingEquiv {m} [Fintype m] [DecidableEq m] :
    (evalRingHom 0).mapMatrix.comp (compRingEquiv m n R[X]) =
      (compRingEquiv m n R).toRingHom.comp (evalRingHom 0).mapMatrix.mapMatrix := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    m : Type u_4
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    ⊢ Eq ((Polynomial.evalRingHom 0).mapMatrix.comp ↑(Matrix.compRingEquiv m n (Po …
  -/
  ext; simp
       /-
         🎉 no goals
       -/

