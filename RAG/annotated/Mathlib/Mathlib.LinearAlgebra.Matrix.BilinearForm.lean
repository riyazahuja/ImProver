/-- The map from `Matrix n n R` to bilinear forms on `n → R`.

This is an auxiliary definition for the equivalence `Matrix.toBilin'`. -/
def Matrix.toBilin'Aux [Fintype n] (M : Matrix n n R₁) : BilinForm R₁ (n → R₁) :=
  Matrix.toLinearMap₂'Aux _ _ M


theorem Matrix.toBilin'Aux_single [Fintype n] [DecidableEq n] (M : Matrix n n R₁) (i j : n) :
    M.toBilin'Aux (Pi.single i 1) (Pi.single j 1) = M i j :=
  Matrix.toLinearMap₂'Aux_single _ _ _ _ _


/-- The linear map from bilinear forms to `Matrix n n R` given an `n`-indexed basis.

This is an auxiliary definition for the equivalence `Matrix.toBilin'`. -/
def BilinForm.toMatrixAux (b : n → M₁) : BilinForm R₁ M₁ →ₗ[R₁] Matrix n n R₁ :=
  LinearMap.toMatrix₂Aux R₁ b b


@[simp]
theorem LinearMap.BilinForm.toMatrixAux_apply (B : BilinForm R₁ M₁) (b : n → M₁) (i j : n) :
    -- Porting note: had to hint the base ring even though it should be clear from context...
    BilinForm.toMatrixAux (R₁ := R₁) b B i j = B (b i) (b j) :=
  LinearMap.toMatrix₂Aux_apply R₁ B _ _ _ _


theorem toBilin'Aux_toMatrixAux [DecidableEq n] (B₂ : BilinForm R₁ (n → R₁)) :
    -- Porting note: had to hint the base ring even though it should be clear from context...
    Matrix.toBilin'Aux (BilinForm.toMatrixAux (R₁ := R₁) (fun j => Pi.single j 1) B₂) = B₂ := by
  /-
    R₁ : Type u_1
    inst✝² : CommSemiring R₁
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    B₂ : LinearMap.BilinForm R₁ (n → R₁)
    ⊢ Eq ((BilinForm.toMatrixAux fun j => Pi.single j 1) B₂).toBilin'Aux B₂
  -/
  rw [BilinForm.toMatrixAux, Matrix.toBilin'Aux, toLinearMap₂'Aux_toMatrix₂Aux]
  /-
    🎉 no goals
  -/


/-- The linear equivalence between bilinear forms on `n → R` and `n × n` matrices -/
def LinearMap.BilinForm.toMatrix' : BilinForm R₁ (n → R₁) ≃ₗ[R₁] Matrix n n R₁ :=
  LinearMap.toMatrix₂' R₁


/-- The linear equivalence between `n × n` matrices and bilinear forms on `n → R` -/
def Matrix.toBilin' : Matrix n n R₁ ≃ₗ[R₁] BilinForm R₁ (n → R₁) :=
  BilinForm.toMatrix'.symm


@[simp]
theorem Matrix.toBilin'Aux_eq (M : Matrix n n R₁) : Matrix.toBilin'Aux M = Matrix.toBilin' M :=
  rfl


theorem Matrix.toBilin'_apply (M : Matrix n n R₁) (x y : n → R₁) :
    Matrix.toBilin' M x y = ∑ i, ∑ j, x i * M i j * y j :=
  (Matrix.toLinearMap₂'_apply _ _ _).trans
        /-
          R₁ : Type u_1
          inst✝² : CommSemiring R₁
          n : Type u_5
          inst✝¹ : Fintype n
          inst✝ : DecidableEq n
          M : Matrix n n R₁
          x y : n → R₁
          ⊢ Eq (Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMul (x i) (HSM …
        -/
    (by simp only [smul_eq_mul, mul_assoc, mul_comm, mul_left_comm])
        /-
          🎉 no goals
        -/


theorem Matrix.toBilin'_apply' (M : Matrix n n R₁) (v w : n → R₁) :
    Matrix.toBilin' M v w = dotProduct v (M *ᵥ w) := Matrix.toLinearMap₂'_apply' _ _ _


@[simp]
theorem Matrix.toBilin'_single (M : Matrix n n R₁) (i j : n) :
    Matrix.toBilin' M (Pi.single i 1) (Pi.single j 1) = M i j := by
  /-
    R₁ : Type u_1
    inst✝² : CommSemiring R₁
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n R₁
    i j : n
    ⊢ Eq (((Matrix.toBilin' M) (Pi.single i 1)) (Pi.single j 1)) (M i j)
  -/
  simp [Matrix.toBilin'_apply, Pi.single_apply]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[simp, deprecated Matrix.toBilin'_single (since := "2024-08-09")]
theorem Matrix.toBilin'_stdBasis (M : Matrix n n R₁) (i j : n) :
    Matrix.toBilin' M
      (LinearMap.stdBasis R₁ (fun _ ↦ R₁) i 1)
      (LinearMap.stdBasis R₁ (fun _ ↦ R₁) j 1) = M i j := Matrix.toBilin'_single _ _ _


@[simp]
theorem LinearMap.BilinForm.toMatrix'_symm :
    (BilinForm.toMatrix'.symm : Matrix n n R₁ ≃ₗ[R₁] _) = Matrix.toBilin' :=
  rfl


@[simp]
theorem Matrix.toBilin'_symm :
    (Matrix.toBilin'.symm : _ ≃ₗ[R₁] Matrix n n R₁) = BilinForm.toMatrix' :=
  BilinForm.toMatrix'.symm_symm


@[simp]
theorem Matrix.toBilin'_toMatrix' (B : BilinForm R₁ (n → R₁)) :
    Matrix.toBilin' (BilinForm.toMatrix' B) = B :=
  Matrix.toBilin'.apply_symm_apply B


@[simp]
theorem BilinForm.toMatrix'_toBilin' (M : Matrix n n R₁) :
    BilinForm.toMatrix' (Matrix.toBilin' M) = M :=
  (LinearMap.toMatrix₂' R₁).apply_symm_apply M


@[simp]
theorem BilinForm.toMatrix'_apply (B : BilinForm R₁ (n → R₁)) (i j : n) :
    BilinForm.toMatrix' B i j = B (Pi.single i 1) (Pi.single j 1) :=
  LinearMap.toMatrix₂'_apply _ _ _

-- Porting note: dot notation for bundled maps doesn't work in the rest of this section

@[simp]
theorem BilinForm.toMatrix'_comp (B : BilinForm R₁ (n → R₁)) (l r : (o → R₁) →ₗ[R₁] n → R₁) :
    BilinForm.toMatrix' (B.comp l r) =
      (LinearMap.toMatrix' l)ᵀ * BilinForm.toMatrix' B * LinearMap.toMatrix' r :=
  LinearMap.toMatrix₂'_compl₁₂ B _ _


theorem BilinForm.toMatrix'_compLeft (B : BilinForm R₁ (n → R₁)) (f : (n → R₁) →ₗ[R₁] n → R₁) :
    BilinForm.toMatrix' (B.compLeft f) = (LinearMap.toMatrix' f)ᵀ * BilinForm.toMatrix' B :=
  LinearMap.toMatrix₂'_comp B _


theorem BilinForm.toMatrix'_compRight (B : BilinForm R₁ (n → R₁)) (f : (n → R₁) →ₗ[R₁] n → R₁) :
    BilinForm.toMatrix' (B.compRight f) = BilinForm.toMatrix' B * LinearMap.toMatrix' f :=
  LinearMap.toMatrix₂'_compl₂ B _


theorem BilinForm.mul_toMatrix'_mul (B : BilinForm R₁ (n → R₁)) (M : Matrix o n R₁)
    (N : Matrix n o R₁) : M * BilinForm.toMatrix' B * N =
      BilinForm.toMatrix' (B.comp (Matrix.toLin' Mᵀ) (Matrix.toLin' N)) :=
  LinearMap.mul_toMatrix₂'_mul B _ _


theorem BilinForm.mul_toMatrix' (B : BilinForm R₁ (n → R₁)) (M : Matrix n n R₁) :
    M * BilinForm.toMatrix' B = BilinForm.toMatrix' (B.compLeft (Matrix.toLin' Mᵀ)) :=
  LinearMap.mul_toMatrix' B _


theorem BilinForm.toMatrix'_mul (B : BilinForm R₁ (n → R₁)) (M : Matrix n n R₁) :
    BilinForm.toMatrix' B * M = BilinForm.toMatrix' (B.compRight (Matrix.toLin' M)) :=
  LinearMap.toMatrix₂'_mul B _


theorem Matrix.toBilin'_comp (M : Matrix n n R₁) (P Q : Matrix n o R₁) :
    M.toBilin'.comp (Matrix.toLin' P) (Matrix.toLin' Q) = Matrix.toBilin' (Pᵀ * M * Q) :=
  BilinForm.toMatrix'.injective
        /-
          R₁ : Type u_1
          inst✝⁴ : CommSemiring R₁
          n : Type u_5
          o : Type u_6
          inst✝³ : Fintype n
          inst✝² : Fintype o
          inst✝¹ : DecidableEq n
          inst✝ : DecidableEq o
          M : Matrix n n R₁
          P Q : Matrix n o R₁
          ⊢ Eq (LinearMap.BilinForm.toMatrix' ((Matrix.toBilin' M).comp (Matrix.toLin' P …
        -/
    (by simp only [BilinForm.toMatrix'_comp, BilinForm.toMatrix'_toBilin', toMatrix'_toLin'])
        /-
          🎉 no goals
        -/


/-- `BilinForm.toMatrix b` is the equivalence between `R`-bilinear forms on `M` and
`n`-by-`n` matrices with entries in `R`, if `b` is an `R`-basis for `M`. -/
noncomputable def BilinForm.toMatrix : BilinForm R₁ M₁ ≃ₗ[R₁] Matrix n n R₁ :=
  LinearMap.toMatrix₂ b b


/-- `BilinForm.toMatrix b` is the equivalence between `R`-bilinear forms on `M` and
`n`-by-`n` matrices with entries in `R`, if `b` is an `R`-basis for `M`. -/
noncomputable def Matrix.toBilin : Matrix n n R₁ ≃ₗ[R₁] BilinForm R₁ M₁ :=
  (BilinForm.toMatrix b).symm


@[simp]
theorem BilinForm.toMatrix_apply (B : BilinForm R₁ M₁) (i j : n) :
    BilinForm.toMatrix b B i j = B (b i) (b j) :=
  LinearMap.toMatrix₂_apply _ _ B _ _


@[simp]
theorem Matrix.toBilin_apply (M : Matrix n n R₁) (x y : M₁) :
    Matrix.toBilin b M x y = ∑ i, ∑ j, b.repr x i * M i j * b.repr y j :=
  (Matrix.toLinearMap₂_apply _ _ _ _ _).trans
        /-
          R₁ : Type u_1
          M₁ : Type u_2
          inst✝⁴ : CommSemiring R₁
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R₁ M₁
          n : Type u_5
          inst✝¹ : Fintype n
          inst✝ : DecidableEq n
          b : Basis n R₁ M₁
          M : Matrix n n R₁
          x y : M₁
          ⊢ Eq (Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMul ((b.repr x …
        -/
    (by simp only [smul_eq_mul, mul_assoc, mul_comm, mul_left_comm])
        /-
          🎉 no goals
        -/

-- Not a `simp` lemma since `BilinForm.toMatrix` needs an extra argument

theorem BilinearForm.toMatrixAux_eq (B : BilinForm R₁ M₁) :
    BilinForm.toMatrixAux (R₁ := R₁) b B = BilinForm.toMatrix b B :=
  LinearMap.toMatrix₂Aux_eq _ _ B


@[simp]
theorem BilinForm.toMatrix_symm : (BilinForm.toMatrix b).symm = Matrix.toBilin b :=
  rfl


@[simp]
theorem Matrix.toBilin_symm : (Matrix.toBilin b).symm = BilinForm.toMatrix b :=
  (BilinForm.toMatrix b).symm_symm


theorem Matrix.toBilin_basisFun : Matrix.toBilin (Pi.basisFun R₁ n) = Matrix.toBilin' := by
  /-
    R₁ : Type u_1
    inst✝² : CommSemiring R₁
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq (Matrix.toBilin (Pi.basisFun R₁ n)) Matrix.toBilin'
  -/
  ext M
  simp only [coe_comp, coe_single, Function.comp_apply, toBilin_apply, Pi.basisFun_repr,
    toBilin'_apply]


theorem BilinForm.toMatrix_basisFun :
    BilinForm.toMatrix (Pi.basisFun R₁ n) = BilinForm.toMatrix' := by
  /-
    R₁ : Type u_1
    inst✝² : CommSemiring R₁
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq (BilinForm.toMatrix (Pi.basisFun R₁ n)) LinearMap.BilinForm.toMatrix'
  -/
  rw [BilinForm.toMatrix, BilinForm.toMatrix', LinearMap.toMatrix₂_basisFun]
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toBilin_toMatrix (B : BilinForm R₁ M₁) :
    Matrix.toBilin b (BilinForm.toMatrix b B) = B :=
  (Matrix.toBilin b).apply_symm_apply B


@[simp]
theorem BilinForm.toMatrix_toBilin (M : Matrix n n R₁) :
    BilinForm.toMatrix b (Matrix.toBilin b M) = M :=
  (BilinForm.toMatrix b).apply_symm_apply M


theorem BilinForm.toMatrix_comp (B : BilinForm R₁ M₁) (l r : M₂' →ₗ[R₁] M₁) :
    BilinForm.toMatrix c (B.comp l r) =
      (LinearMap.toMatrix c b l)ᵀ * BilinForm.toMatrix b B * LinearMap.toMatrix c b r :=
  LinearMap.toMatrix₂_compl₁₂ _ _ _ _ B _ _


theorem BilinForm.toMatrix_compLeft (B : BilinForm R₁ M₁) (f : M₁ →ₗ[R₁] M₁) :
    BilinForm.toMatrix b (B.compLeft f) = (LinearMap.toMatrix b b f)ᵀ * BilinForm.toMatrix b B :=
  LinearMap.toMatrix₂_comp _ _ _ B _


theorem BilinForm.toMatrix_compRight (B : BilinForm R₁ M₁) (f : M₁ →ₗ[R₁] M₁) :
    BilinForm.toMatrix b (B.compRight f) = BilinForm.toMatrix b B * LinearMap.toMatrix b b f :=
  LinearMap.toMatrix₂_compl₂ _ _ _ B _


@[simp]
theorem BilinForm.toMatrix_mul_basis_toMatrix (c : Basis o R₁ M₁) (B : BilinForm R₁ M₁) :
    (b.toMatrix c)ᵀ * BilinForm.toMatrix b B * b.toMatrix c = BilinForm.toMatrix c B :=
  LinearMap.toMatrix₂_mul_basis_toMatrix _ _ _  _ B


theorem BilinForm.mul_toMatrix_mul (B : BilinForm R₁ M₁) (M : Matrix o n R₁) (N : Matrix n o R₁) :
    M * BilinForm.toMatrix b B * N =
      BilinForm.toMatrix c (B.comp (Matrix.toLin c b Mᵀ) (Matrix.toLin c b N)) :=
  LinearMap.mul_toMatrix₂_mul _ _ _ _ B _ _


theorem BilinForm.mul_toMatrix (B : BilinForm R₁ M₁) (M : Matrix n n R₁) :
    M * BilinForm.toMatrix b B = BilinForm.toMatrix b (B.compLeft (Matrix.toLin b b Mᵀ)) :=
  LinearMap.mul_toMatrix₂ _ _ _ B _


theorem BilinForm.toMatrix_mul (B : BilinForm R₁ M₁) (M : Matrix n n R₁) :
    BilinForm.toMatrix b B * M = BilinForm.toMatrix b (B.compRight (Matrix.toLin b b M)) :=
  LinearMap.toMatrix₂_mul _ _ _  B _


theorem Matrix.toBilin_comp (M : Matrix n n R₁) (P Q : Matrix n o R₁) :
    (Matrix.toBilin b M).comp (toLin c b P) (toLin c b Q) = Matrix.toBilin c (Pᵀ * M * Q) := by
  /-
    R₁ : Type u_1
    M₁ : Type u_2
    inst✝⁸ : CommSemiring R₁
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R₁ M₁
    n : Type u_5
    o : Type u_6
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype o
    inst✝³ : DecidableEq n
    b : Basis n R₁ M₁
    M₂' : Type u_7
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : Module R₁ M₂'
    c : Basis o R₁ M₂'
    inst✝ : DecidableEq o
    M : Matrix n n R₁
    P Q : Matrix n o R₁
    ⊢ Eq (((Matrix.toBilin b) M).comp ((Matrix.toLin c b) P) ((Matrix.toLin c b) Q …
  -/
  ext x y
  rw [Matrix.toBilin, BilinForm.toMatrix, Matrix.toBilin, BilinForm.toMatrix, toMatrix₂_symm,
    toMatrix₂_symm, ← Matrix.toLinearMap₂_compl₁₂ b b c c]
  /-
    case H
    R₁ : Type u_1
    M₁ : Type u_2
    inst✝⁸ : CommSemiring R₁
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R₁ M₁
    n : Type u_5
    o : Type u_6
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype o
    inst✝³ : DecidableEq n
    b : Basis n R₁ M₁
    M₂' : Type u_7
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : Module R₁ M₂'
    c : Basis o R₁ M₂'
    inst✝ : DecidableEq o
    M : Matrix n n R₁
    P Q : Matrix n o R₁
    x y : M₂'
    ⊢ Eq (((((Matrix.toLinearMap₂ b b) M).comp ((Matrix.toLin c b) P) ((Matrix.toL …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Matrix.isAdjointPair_equiv' [DecidableEq n] (P : Matrix n n R₂) (h : IsUnit P) :
    (Pᵀ * J * P).IsAdjointPair (Pᵀ * J * P) A A' ↔
      J.IsAdjointPair J (P * A * P⁻¹) (P * A' * P⁻¹) :=
  Matrix.isAdjointPair_equiv _ _ _ _ h


theorem mem_pairSelfAdjointMatricesSubmodule' :
    A ∈ pairSelfAdjointMatricesSubmodule J J₃ ↔ Matrix.IsAdjointPair J J₃ A A := by
  /-
    R₂ : Type u_3
    inst✝² : CommRing R₂
    n : Type u_5
    inst✝¹ : Fintype n
    J J₃ A : Matrix n n R₂
    inst✝ : DecidableEq n
    ⊢ Iff (Membership.mem (pairSelfAdjointMatricesSubmodule J J₃) A) (J.IsAdjointP …
  -/
  simp only [mem_pairSelfAdjointMatricesSubmodule]
  /-
    🎉 no goals
  -/


/-- The submodule of self-adjoint matrices with respect to the bilinear form corresponding to
the matrix `J`. -/
def selfAdjointMatricesSubmodule' : Submodule R₂ (Matrix n n R₂) :=
  pairSelfAdjointMatricesSubmodule J J


theorem mem_selfAdjointMatricesSubmodule' :
    A ∈ selfAdjointMatricesSubmodule J ↔ J.IsSelfAdjoint A := by
  /-
    R₂ : Type u_3
    inst✝² : CommRing R₂
    n : Type u_5
    inst✝¹ : Fintype n
    J A : Matrix n n R₂
    inst✝ : DecidableEq n
    ⊢ Iff (Membership.mem (selfAdjointMatricesSubmodule J) A) (J.IsSelfAdjoint A)
  -/
  simp only [mem_selfAdjointMatricesSubmodule]
  /-
    🎉 no goals
  -/


/-- The submodule of skew-adjoint matrices with respect to the bilinear form corresponding to
the matrix `J`. -/
def skewAdjointMatricesSubmodule' : Submodule R₂ (Matrix n n R₂) :=
  pairSelfAdjointMatricesSubmodule (-J) J


theorem mem_skewAdjointMatricesSubmodule' :
    A ∈ skewAdjointMatricesSubmodule J ↔ J.IsSkewAdjoint A := by
  /-
    R₂ : Type u_3
    inst✝² : CommRing R₂
    n : Type u_5
    inst✝¹ : Fintype n
    J A : Matrix n n R₂
    inst✝ : DecidableEq n
    ⊢ Iff (Membership.mem (skewAdjointMatricesSubmodule J) A) (J.IsSkewAdjoint A)
  -/
  simp only [mem_skewAdjointMatricesSubmodule]
  /-
    🎉 no goals
  -/


theorem _root_.Matrix.nondegenerate_toBilin'_iff_nondegenerate_toBilin {M : Matrix ι ι R₁}
    (b : Basis ι R₁ M₁) : M.toBilin'.Nondegenerate ↔ (Matrix.toBilin b M).Nondegenerate :=
  (nondegenerate_congr_iff b.equivFun.symm).symm

-- Lemmas transferring nondegeneracy between a matrix and its associated bilinear form

theorem _root_.Matrix.Nondegenerate.toBilin' {M : Matrix ι ι R₂} (h : M.Nondegenerate) :
    M.toBilin'.Nondegenerate := fun x hx =>
                                 /-
                                   R₂ : Type u_3
                                   inst✝² : CommRing R₂
                                   ι : Type u_6
                                   inst✝¹ : DecidableEq ι
                                   inst✝ : Fintype ι
                                   M : Matrix ι ι R₂
                                   h : M.Nondegenerate
                                   x : ι → R₂
                                   hx : ∀ (n : ι → R₂), Eq (((Matrix.toBilin' M) x) n) 0
                                   y : ι → R₂
                                   ⊢ Eq (dotProduct x (M.mulVec y)) 0
                                 -/
  h.eq_zero_of_ortho fun y => by simpa only [toBilin'_apply'] using hx y
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem _root_.Matrix.nondegenerate_toBilin'_iff {M : Matrix ι ι R₂} :
    M.toBilin'.Nondegenerate ↔ M.Nondegenerate :=
  ⟨fun h v hv => h v fun w => (M.toBilin'_apply' _ _).trans <| hv w, Matrix.Nondegenerate.toBilin'⟩


theorem _root_.Matrix.Nondegenerate.toBilin {M : Matrix ι ι R₂} (h : M.Nondegenerate)
    (b : Basis ι R₂ M₂) : (Matrix.toBilin b M).Nondegenerate :=
  (Matrix.nondegenerate_toBilin'_iff_nondegenerate_toBilin b).mp h.toBilin'


@[simp]
theorem _root_.Matrix.nondegenerate_toBilin_iff {M : Matrix ι ι R₂} (b : Basis ι R₂ M₂) :
    (Matrix.toBilin b M).Nondegenerate ↔ M.Nondegenerate := by
  /-
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁴ : CommRing R₂
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R₂ M₂
    ι : Type u_6
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    M : Matrix ι ι R₂
    b : Basis ι R₂ M₂
    ⊢ Iff ((Matrix.toBilin b) M).Nondegenerate M.Nondegenerate
  -/
  rw [← Matrix.nondegenerate_toBilin'_iff_nondegenerate_toBilin, Matrix.nondegenerate_toBilin'_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem nondegenerate_toMatrix'_iff {B : BilinForm R₂ (ι → R₂)} :
    B.toMatrix'.Nondegenerate (m := ι) ↔ B.Nondegenerate :=
  Matrix.nondegenerate_toBilin'_iff.symm.trans <| (Matrix.toBilin'_toMatrix' B).symm ▸ Iff.rfl


theorem Nondegenerate.toMatrix' {B : BilinForm R₂ (ι → R₂)} (h : B.Nondegenerate) :
    B.toMatrix'.Nondegenerate :=
  nondegenerate_toMatrix'_iff.mpr h


@[simp]
theorem nondegenerate_toMatrix_iff {B : BilinForm R₂ M₂} (b : Basis ι R₂ M₂) :
    (BilinForm.toMatrix b B).Nondegenerate ↔ B.Nondegenerate :=
  (Matrix.nondegenerate_toBilin_iff b).symm.trans <| (Matrix.toBilin_toMatrix b B).symm ▸ Iff.rfl


theorem Nondegenerate.toMatrix {B : BilinForm R₂ M₂} (h : B.Nondegenerate) (b : Basis ι R₂ M₂) :
    (BilinForm.toMatrix b B).Nondegenerate :=
  (nondegenerate_toMatrix_iff b).mpr h


theorem nondegenerate_toBilin'_iff_det_ne_zero {M : Matrix ι ι A} :
    M.toBilin'.Nondegenerate ↔ M.det ≠ 0 := by
  /-
    A : Type u_5
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    ι : Type u_6
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    M : Matrix ι ι A
    ⊢ Iff (Matrix.toBilin' M).Nondegenerate (Ne M.det 0)
  -/
  rw [Matrix.nondegenerate_toBilin'_iff, Matrix.nondegenerate_iff_det_ne_zero]
  /-
    🎉 no goals
  -/


theorem nondegenerate_toBilin'_of_det_ne_zero' (M : Matrix ι ι A) (h : M.det ≠ 0) :
    M.toBilin'.Nondegenerate :=
  nondegenerate_toBilin'_iff_det_ne_zero.mpr h


theorem nondegenerate_iff_det_ne_zero {B : BilinForm A M₂} (b : Basis ι A M₂) :
    B.Nondegenerate ↔ (BilinForm.toMatrix b B).det ≠ 0 := by
  /-
    M₂ : Type u_4
    inst✝⁵ : AddCommGroup M₂
    A : Type u_5
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Module A M₂
    ι : Type u_6
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    B : LinearMap.BilinForm A M₂
    b : Basis ι A M₂
    ⊢ Iff B.Nondegenerate (Ne ((BilinForm.toMatrix b) B).det 0)
  -/
  rw [← Matrix.nondegenerate_iff_det_ne_zero, nondegenerate_toMatrix_iff]
  /-
    🎉 no goals
  -/


theorem nondegenerate_of_det_ne_zero (b : Basis ι A M₂) (h : (BilinForm.toMatrix b B₃).det ≠ 0) :
    B₃.Nondegenerate :=
  (nondegenerate_iff_det_ne_zero b).mpr h


