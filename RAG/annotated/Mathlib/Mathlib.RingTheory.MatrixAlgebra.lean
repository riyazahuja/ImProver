/-- (Implementation detail).
The function underlying `(A ⊗[R] Matrix n n R) →ₐ[R] Matrix n n A`,
as an `R`-bilinear map.
-/
def toFunBilinear : A →ₗ[R] Matrix n n R →ₗ[R] Matrix n n A :=
  (Algebra.lsmul R R (Matrix n n A)).toLinearMap.compl₂ (Algebra.linearMap R A).mapMatrix


@[simp]
theorem toFunBilinear_apply (a : A) (m : Matrix n n R) :
    toFunBilinear R A n a m = a • m.map (algebraMap R A) :=
  rfl


/-- (Implementation detail).
The function underlying `(A ⊗[R] Matrix n n R) →ₐ[R] Matrix n n A`,
as an `R`-linear map.
-/
def toFunLinear : A ⊗[R] Matrix n n R →ₗ[R] Matrix n n A :=
  TensorProduct.lift (toFunBilinear R A n)


/-- The function `(A ⊗[R] Matrix n n R) →ₐ[R] Matrix n n A`, as an algebra homomorphism.
-/
def toFunAlgHom : A ⊗[R] Matrix n n R →ₐ[R] Matrix n n A :=
  algHomOfLinearMapTensorProduct (toFunLinear R A n)
    (by
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        ⊢ ∀ (a₁ a₂ : A) (b₁ b₂ : Matrix n n R), Eq ((MatrixEquivTensor.toFunLinear R A …
      -/
      intros
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        a₁✝ a₂✝ : A
        b₁✝ b₂✝ : Matrix n n R
        ⊢ Eq ((MatrixEquivTensor.toFunLinear R A n) (TensorProduct.tmul R (HMul.hMul a …
      -/
      simp_rw [toFunLinear, lift.tmul, toFunBilinear_apply, Matrix.map_mul]
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        a₁✝ a₂✝ : A
        b₁✝ b₂✝ : Matrix n n R
        ⊢ Eq (HSMul.hSMul (HMul.hMul a₁✝ a₂✝) (HMul.hMul (b₁✝.map ⇑(algebraMap R A)) ( …
      -/
      ext
      /-
        case a
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        a₁✝ a₂✝ : A
        b₁✝ b₂✝ : Matrix n n R
        i✝ j✝ : n
        ⊢ Eq (HSMul.hSMul (HMul.hMul a₁✝ a₂✝) (HMul.hMul (b₁✝.map ⇑(algebraMap R A)) ( …
      -/
      dsimp
      simp_rw [Matrix.mul_apply, Matrix.smul_apply, Matrix.map_apply, smul_eq_mul, Finset.mul_sum,
        _root_.mul_assoc, Algebra.left_comm])
    (by
      simp_rw [toFunLinear, lift.tmul, toFunBilinear_apply,
        Matrix.map_one (algebraMap R A) (map_zero _) (map_one _), one_smul])


@[simp]
theorem toFunAlgHom_apply (a : A) (m : Matrix n n R) :
    toFunAlgHom R A n (a ⊗ₜ m) = a • m.map (algebraMap R A) := rfl


/-- (Implementation detail.)

The bare function `Matrix n n A → A ⊗[R] Matrix n n R`.
(We don't need to show that it's an algebra map, thankfully --- just that it's an inverse.)
-/
def invFun (M : Matrix n n A) : A ⊗[R] Matrix n n R :=
  ∑ p : n × n, M p.1 p.2 ⊗ₜ stdBasisMatrix p.1 p.2 1


@[simp]
                                               /-
                                                 R : Type u
                                                 inst✝⁴ : CommSemiring R
                                                 A : Type v
                                                 inst✝³ : Semiring A
                                                 inst✝² : Algebra R A
                                                 n : Type w
                                                 inst✝¹ : DecidableEq n
                                                 inst✝ : Fintype n
                                                 ⊢ Eq (MatrixEquivTensor.invFun R A n 0) 0
                                               -/
theorem invFun_zero : invFun R A n 0 = 0 := by simp [invFun]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem invFun_add (M N : Matrix n n A) :
    invFun R A n (M + N) = invFun R A n M + invFun R A n N := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M N : Matrix n n A
    ⊢ Eq (MatrixEquivTensor.invFun R A n (HAdd.hAdd M N)) (HAdd.hAdd (MatrixEquivT …
  -/
  simp [invFun, add_tmul, Finset.sum_add_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem invFun_smul (a : A) (M : Matrix n n A) :
    invFun R A n (a • M) = a ⊗ₜ 1 * invFun R A n M := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    a : A
    M : Matrix n n A
    ⊢ Eq (MatrixEquivTensor.invFun R A n (HSMul.hSMul a M)) (HMul.hMul (TensorProd …
  -/
  simp [invFun, Finset.mul_sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem invFun_algebraMap (M : Matrix n n R) : invFun R A n (M.map (algebraMap R A)) = 1 ⊗ₜ M := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (MatrixEquivTensor.invFun R A n (M.map ⇑(algebraMap R A))) (TensorProduct …
  -/
  dsimp [invFun]
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (Finset.univ.sum fun p => TensorProduct.tmul R ((algebraMap R A) (M p.1 p …
  -/
  simp only [Algebra.algebraMap_eq_smul_one, smul_tmul, ← tmul_sum, mul_boole]
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (TensorProduct.tmul R 1 (Finset.univ.sum fun a => HSMul.hSMul (M a.1 a.2) …
  -/
  congr
  /-
    case e_n
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (Finset.univ.sum fun a => HSMul.hSMul (M a.1 a.2) (Matrix.stdBasisMatrix  …
  -/
  conv_rhs => rw [matrix_eq_sum_stdBasisMatrix M]
  /-
    case e_n
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (Finset.univ.sum fun a => HSMul.hSMul (M a.1 a.2) (Matrix.stdBasisMatrix  …
  -/
  convert Finset.sum_product (β := Matrix n n R) ..; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem right_inv (M : Matrix n n A) : (toFunAlgHom R A n) (invFun R A n M) = M := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    ⊢ Eq ((MatrixEquivTensor.toFunAlgHom R A n) (MatrixEquivTensor.invFun R A n M) …
  -/
  simp only [invFun, map_sum, toFunAlgHom_apply]
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (M x.1 x.2) ((Matrix.stdBasisMatrix …
  -/
  convert Finset.sum_product (β := Matrix n n A) ..
  /-
    case h.e'_3
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    ⊢ Eq M (Finset.univ.sum fun x => Finset.univ.sum fun y => HSMul.hSMul (M { fst …
  -/
  conv_lhs => rw [matrix_eq_sum_stdBasisMatrix M]
  /-
    case h.e'_3
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    ⊢ Eq (Finset.univ.sum fun i => Finset.univ.sum fun j => Matrix.stdBasisMatrix  …
  -/
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => Matrix.ext fun a b => ?_
  /-
    case h.e'_3
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    i : n
    x✝¹ : Membership.mem Finset.univ i
    j : n
    x✝ : Membership.mem Finset.univ j
    a b : n
    ⊢ Eq (Matrix.stdBasisMatrix i j (M i j) a b) (HSMul.hSMul (M { fst := i, snd : …
  -/
  dsimp [stdBasisMatrix]
  /-
    case h.e'_3
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n A
    i : n
    x✝¹ : Membership.mem Finset.univ i
    j : n
    x✝ : Membership.mem Finset.univ j
    a b : n
    ⊢ Eq (ite (And (Eq i a) (Eq j b)) (M i j) 0) (HMul.hMul (M i j) ((algebraMap R …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> aesop
                /-
                  🎉 no goals
                -/


theorem left_inv (M : A ⊗[R] Matrix n n R) : invFun R A n (toFunAlgHom R A n M) = M := by
  induction M with
  | zero => simp
  | tmul a m => simp
  | add x y hx hy =>
    rw [map_add]
    conv_rhs => rw [← hx, ← hy, ← invFun_add]


/-- (Implementation detail)

The equivalence, ignoring the algebra structure, `(A ⊗[R] Matrix n n R) ≃ Matrix n n A`.
-/
def equiv : A ⊗[R] Matrix n n R ≃ Matrix n n A where
  toFun := toFunAlgHom R A n
  invFun := invFun R A n
  left_inv := left_inv R A n
  right_inv := right_inv R A n


/-- The `R`-algebra isomorphism `Matrix n n A ≃ₐ[R] (A ⊗[R] Matrix n n R)`.
-/
def matrixEquivTensor : Matrix n n A ≃ₐ[R] A ⊗[R] Matrix n n R :=
  AlgEquiv.symm { MatrixEquivTensor.toFunAlgHom R A n, MatrixEquivTensor.equiv R A n with }


@[simp]
theorem matrixEquivTensor_apply (M : Matrix n n A) :
    matrixEquivTensor R A n M = ∑ p : n × n, M p.1 p.2 ⊗ₜ stdBasisMatrix p.1 p.2 1 :=
  rfl

-- Porting note: short circuiting simplifier from simplifying left hand side

@[simp (high)]
theorem matrixEquivTensor_apply_stdBasisMatrix (i j : n) (x : A) :
    matrixEquivTensor R A n (stdBasisMatrix i j x) = x ⊗ₜ stdBasisMatrix i j 1 := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    i j : n
    x : A
    ⊢ Eq ((matrixEquivTensor R A n) (Matrix.stdBasisMatrix i j x)) (TensorProduct. …
  -/
  have t : ∀ p : n × n, i = p.1 ∧ j = p.2 ↔ p = (i, j) := by aesop
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    n : Type w
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    i j : n
    x : A
    t : ∀ (p : Prod n n), Iff (And (Eq i p.1) (Eq j p.2)) (Eq p { fst := i, snd := …
    ⊢ Eq ((matrixEquivTensor R A n) (Matrix.stdBasisMatrix i j x)) (TensorProduct. …
  -/
  simp [ite_tmul, t, stdBasisMatrix]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-11")] alias matrixEquivTensor_apply_std_basis :=
  matrixEquivTensor_apply_stdBasisMatrix


@[simp]
theorem matrixEquivTensor_apply_symm (a : A) (M : Matrix n n R) :
    (matrixEquivTensor R A n).symm (a ⊗ₜ M) = M.map fun x => a * algebraMap R A x :=
  rfl

