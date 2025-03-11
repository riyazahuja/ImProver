/-- The linear map built from `TensorProduct.map` corresponds to the matrix built from
`Matrix.kronecker`. -/
theorem TensorProduct.toMatrix_map (f : M →ₗ[R] M') (g : N →ₗ[R] N') :
    toMatrix (bM.tensorProduct bN) (bM'.tensorProduct bN') (TensorProduct.map f g) =
      toMatrix bM bM' f ⊗ₖ toMatrix bN bN' g := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    M' : Type u_5
    N' : Type u_6
    ι : Type u_7
    κ : Type u_8
    ι' : Type u_10
    κ' : Type u_11
    inst✝¹⁴ : DecidableEq ι
    inst✝¹³ : DecidableEq κ
    inst✝¹² : Fintype ι
    inst✝¹¹ : Fintype κ
    inst✝¹⁰ : Finite ι'
    inst✝⁹ : Finite κ'
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R M'
    inst✝ : Module R N'
    bM : Basis ι R M
    bN : Basis κ R N
    bM' : Basis ι' R M'
    bN' : Basis κ' R N'
    f : LinearMap (RingHom.id R) M M'
    g : LinearMap (RingHom.id R) N N'
    ⊢ Eq ((LinearMap.toMatrix (bM.tensorProduct bN) (bM'.tensorProduct bN')) (Tens …
  -/
  ext ⟨i, j⟩ ⟨i', j'⟩
  simp_rw [Matrix.kroneckerMap_apply, toMatrix_apply, Basis.tensorProduct_apply,
    TensorProduct.map_tmul, Basis.tensorProduct_repr_tmul_apply]
  /-
    case a.mk.mk
    R : Type u_1
    M : Type u_2
    N : Type u_3
    M' : Type u_5
    N' : Type u_6
    ι : Type u_7
    κ : Type u_8
    ι' : Type u_10
    κ' : Type u_11
    inst✝¹⁴ : DecidableEq ι
    inst✝¹³ : DecidableEq κ
    inst✝¹² : Fintype ι
    inst✝¹¹ : Fintype κ
    inst✝¹⁰ : Finite ι'
    inst✝⁹ : Finite κ'
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R M'
    inst✝ : Module R N'
    bM : Basis ι R M
    bN : Basis κ R N
    bM' : Basis ι' R M'
    bN' : Basis κ' R N'
    f : LinearMap (RingHom.id R) M M'
    g : LinearMap (RingHom.id R) N N'
    i : ι'
    j : κ'
    i' : ι
    j' : κ
    ⊢ Eq (HSMul.hSMul ((bN'.repr (g (bN j'))) j) ((bM'.repr (f (bM i'))) i)) (HMul …
  -/
  exact mul_comm _ _
  /-
    🎉 no goals
  -/


/-- The matrix built from `Matrix.kronecker` corresponds to the linear map built from
`TensorProduct.map`. -/
theorem Matrix.toLin_kronecker (A : Matrix ι' ι R) (B : Matrix κ' κ R) :
    toLin (bM.tensorProduct bN) (bM'.tensorProduct bN') (A ⊗ₖ B) =
      TensorProduct.map (toLin bM bM' A) (toLin bN bN' B) := by
  rw [← LinearEquiv.eq_symm_apply, toLin_symm, TensorProduct.toMatrix_map, toMatrix_toLin,
    toMatrix_toLin]


/-- `TensorProduct.comm` corresponds to a permutation of the identity matrix. -/
theorem TensorProduct.toMatrix_comm :
    toMatrix (bM.tensorProduct bN) (bN.tensorProduct bM) (TensorProduct.comm R M N) =
      (1 : Matrix (ι × κ) (ι × κ) R).submatrix Prod.swap _root_.id := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    ι : Type u_7
    κ : Type u_8
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    bM : Basis ι R M
    bN : Basis κ R N
    ⊢ Eq ((LinearMap.toMatrix (bM.tensorProduct bN) (bN.tensorProduct bM)) ↑(Tenso …
  -/
  ext ⟨i, j⟩ ⟨i', j'⟩
  simp only [toMatrix_apply, Basis.tensorProduct_apply, LinearEquiv.coe_coe, comm_tmul,
    Basis.tensorProduct_repr_tmul_apply, Basis.repr_self, Finsupp.single_apply, @eq_comm _ i',
    @eq_comm _ j', smul_eq_mul, mul_ite, mul_one, mul_zero, ← ite_and, and_comm, submatrix_apply,
    Matrix.one_apply, Prod.swap_prod_mk, id_eq, Prod.mk.injEq]


/-- `TensorProduct.assoc` corresponds to a permutation of the identity matrix. -/
theorem TensorProduct.toMatrix_assoc :
    toMatrix ((bM.tensorProduct bN).tensorProduct bP) (bM.tensorProduct (bN.tensorProduct bP))
        (TensorProduct.assoc R M N P) =
      (1 : Matrix (ι × κ × τ) (ι × κ × τ) R).submatrix _root_.id (Equiv.prodAssoc _ _ _) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    ι : Type u_7
    κ : Type u_8
    τ : Type u_9
    inst✝¹² : DecidableEq ι
    inst✝¹¹ : DecidableEq κ
    inst✝¹⁰ : DecidableEq τ
    inst✝⁹ : Fintype ι
    inst✝⁸ : Fintype κ
    inst✝⁷ : Fintype τ
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    bM : Basis ι R M
    bN : Basis κ R N
    bP : Basis τ R P
    ⊢ Eq ((LinearMap.toMatrix ((bM.tensorProduct bN).tensorProduct bP) (bM.tensorP …
  -/
  ext ⟨i, j, k⟩ ⟨⟨i', j'⟩, k'⟩
  simp only [toMatrix_apply, Basis.tensorProduct_apply, LinearEquiv.coe_coe, assoc_tmul,
    Basis.tensorProduct_repr_tmul_apply, Basis.repr_self, Finsupp.single_apply, @eq_comm _ k',
    @eq_comm _ j', smul_eq_mul, mul_ite, mul_one, mul_zero, ← ite_and, @eq_comm _ i',
    submatrix_apply, Matrix.one_apply, id_eq, Equiv.prodAssoc_apply, Prod.mk.injEq]

