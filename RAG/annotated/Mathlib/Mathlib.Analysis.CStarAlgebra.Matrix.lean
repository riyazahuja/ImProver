theorem entry_norm_bound_of_unitary {U : Matrix n n 𝕜} (hU : U ∈ Matrix.unitaryGroup n 𝕜)
    (i j : n) : ‖U i j‖ ≤ 1 := by
  -- The norm squared of an entry is at most the L2 norm of its row.
  have norm_sum : ‖U i j‖ ^ 2 ≤ ∑ x, ‖U i x‖ ^ 2 := by
    apply Multiset.single_le_sum
    · intro x h_x
      rw [Multiset.mem_map] at h_x
      cases' h_x with a h_a
      rw [← h_a.2]
      apply sq_nonneg
    · rw [Multiset.mem_map]
      use j
      simp only [eq_self_iff_true, Finset.mem_univ_val, and_self_iff, sq_eq_sq₀]
  -- The L2 norm of a row is a diagonal entry of U * Uᴴ
  have diag_eq_norm_sum : (U * Uᴴ) i i = (∑ x : n, ‖U i x‖ ^ 2 : ℝ) := by
    simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, ← starRingEnd_apply, RCLike.mul_conj,
      RCLike.normSq_eq_def', RCLike.ofReal_pow]; norm_cast
  -- The L2 norm of a row is a diagonal entry of U * Uᴴ, real part
  have re_diag_eq_norm_sum : RCLike.re ((U * Uᴴ) i i) = ∑ x : n, ‖U i x‖ ^ 2 := by
    rw [RCLike.ext_iff] at diag_eq_norm_sum
    rw [diag_eq_norm_sum.1]
    norm_cast
  -- Since U is unitary, the diagonal entries of U * Uᴴ are all 1
  /-
    𝕜 : Type u_1
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    U : Matrix n n 𝕜
    hU : Membership.mem (Matrix.unitaryGroup n 𝕜) U
    i j : n
    norm_sum : LE.le (HPow.hPow (Norm.norm (U i j)) 2) (Finset.univ.sum fun x => H …
    diag_eq_norm_sum : Eq (HMul.hMul U U.conjTranspose i i) ↑(Finset.univ.sum fun  …
    re_diag_eq_norm_sum : Eq (RCLike.re (HMul.hMul U U.conjTranspose i i)) (Finset …
    ⊢ LE.le (Norm.norm (U i j)) 1
  -/
  have mul_eq_one : U * Uᴴ = 1 := unitary.mul_star_self_of_mem hU
  have diag_eq_one : RCLike.re ((U * Uᴴ) i i) = 1 := by
    simp only [mul_eq_one, eq_self_iff_true, Matrix.one_apply_eq, RCLike.one_re]
  -- Putting it all together
  /-
    𝕜 : Type u_1
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    U : Matrix n n 𝕜
    hU : Membership.mem (Matrix.unitaryGroup n 𝕜) U
    i j : n
    norm_sum : LE.le (HPow.hPow (Norm.norm (U i j)) 2) (Finset.univ.sum fun x => H …
    diag_eq_norm_sum : Eq (HMul.hMul U U.conjTranspose i i) ↑(Finset.univ.sum fun  …
    re_diag_eq_norm_sum : Eq (RCLike.re (HMul.hMul U U.conjTranspose i i)) (Finset …
    mul_eq_one : Eq (HMul.hMul U U.conjTranspose) 1
    diag_eq_one : Eq (RCLike.re (HMul.hMul U U.conjTranspose i i)) 1
    ⊢ LE.le (Norm.norm (U i j)) 1
  -/
  rw [← sq_le_one_iff₀ (norm_nonneg (U i j)), ← diag_eq_one, re_diag_eq_norm_sum]
  /-
    𝕜 : Type u_1
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    U : Matrix n n 𝕜
    hU : Membership.mem (Matrix.unitaryGroup n 𝕜) U
    i j : n
    norm_sum : LE.le (HPow.hPow (Norm.norm (U i j)) 2) (Finset.univ.sum fun x => H …
    diag_eq_norm_sum : Eq (HMul.hMul U U.conjTranspose i i) ↑(Finset.univ.sum fun  …
    re_diag_eq_norm_sum : Eq (RCLike.re (HMul.hMul U U.conjTranspose i i)) (Finset …
    mul_eq_one : Eq (HMul.hMul U U.conjTranspose) 1
    diag_eq_one : Eq (RCLike.re (HMul.hMul U U.conjTranspose i i)) 1
    ⊢ LE.le (HPow.hPow (Norm.norm (U i j)) 2) (Finset.univ.sum fun x => HPow.hPow  …
  -/
  exact norm_sum
  /-
    🎉 no goals
  -/


/-- The entrywise sup norm of a unitary matrix is at most 1. -/
theorem entrywise_sup_norm_bound_of_unitary {U : Matrix n n 𝕜} (hU : U ∈ Matrix.unitaryGroup n 𝕜) :
    ‖U‖ ≤ 1 := by
  conv => -- Porting note: was `simp_rw [pi_norm_le_iff_of_nonneg zero_le_one]`
    rw [pi_norm_le_iff_of_nonneg zero_le_one]
    intro
    rw [pi_norm_le_iff_of_nonneg zero_le_one]
  /-
    𝕜 : Type u_1
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    U : Matrix n n 𝕜
    hU : Membership.mem (Matrix.unitaryGroup n 𝕜) U
    ⊢ ∀ (i i_1 : n), LE.le (Norm.norm (U i i_1)) 1
  -/
  intros
  /-
    𝕜 : Type u_1
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    U : Matrix n n 𝕜
    hU : Membership.mem (Matrix.unitaryGroup n 𝕜) U
    i✝¹ i✝ : n
    ⊢ LE.le (Norm.norm (U i✝¹ i✝)) 1
  -/
  exact entry_norm_bound_of_unitary hU _ _
  /-
    🎉 no goals
  -/


/-- The natural star algebra equivalence between matrices and continuous linear endomoporphisms
of Euclidean space induced by the orthonormal basis `EuclideanSpace.basisFun`.

This is a more-bundled version of `Matrix.toEuclideanLin`, for the special case of square matrices,
followed by a more-bundled version of `LinearMap.toContinuousLinearMap`. -/
def toEuclideanCLM :
    Matrix n n 𝕜 ≃⋆ₐ[𝕜] (EuclideanSpace 𝕜 n →L[𝕜] EuclideanSpace 𝕜 n) :=
  toMatrixOrthonormal (EuclideanSpace.basisFun n 𝕜) |>.symm.trans <|
    { toContinuousLinearMap with
      map_mul' := fun _ _ ↦ rfl
      map_star' := adjoint_toContinuousLinearMap }


lemma coe_toEuclideanCLM_eq_toEuclideanLin (A : Matrix n n 𝕜) :
    (toEuclideanCLM (n := n) (𝕜 := 𝕜) A : _ →ₗ[𝕜] _) = toEuclideanLin A :=
  rfl


@[simp]
lemma toEuclideanCLM_piLp_equiv_symm (A : Matrix n n 𝕜) (x : n → 𝕜) :
    toEuclideanCLM (n := n) (𝕜 := 𝕜) A ((WithLp.equiv _ _).symm x) =
      (WithLp.equiv _ _).symm (toLin' A x) :=
  rfl


@[simp]
lemma piLp_equiv_toEuclideanCLM (A : Matrix n n 𝕜) (x : EuclideanSpace 𝕜 n) :
    WithLp.equiv _ _ (toEuclideanCLM (n := n) (𝕜 := 𝕜) A x) =
      toLin' A (WithLp.equiv _ _ x) :=
  rfl


/-- An auxiliary definition used only to construct the true `NormedAddCommGroup` (and `Metric`)
structure provided by `Matrix.instMetricSpaceL2Op` and `Matrix.instNormedAddCommGroupL2Op`. -/
def l2OpNormedAddCommGroupAux : NormedAddCommGroup (Matrix m n 𝕜) :=
  @NormedAddCommGroup.induced ((Matrix m n 𝕜) ≃ₗ[𝕜] (EuclideanSpace 𝕜 n →L[𝕜] EuclideanSpace 𝕜 m)) _
    _ _ _ ContinuousLinearMap.toNormedAddCommGroup.toNormedAddGroup _ _ <|
    (toEuclideanLin.trans toContinuousLinearMap).injective


/-- An auxiliary definition used only to construct the true `NormedRing` (and `Metric`) structure
provided by `Matrix.instMetricSpaceL2Op` and `Matrix.instNormedRingL2Op`. -/
def l2OpNormedRingAux : NormedRing (Matrix n n 𝕜) :=
  @NormedRing.induced ((Matrix n n 𝕜) ≃⋆ₐ[𝕜] (EuclideanSpace 𝕜 n →L[𝕜] EuclideanSpace 𝕜 n)) _
    _ _ _ ContinuousLinearMap.toNormedRing _ _ toEuclideanCLM.injective


/-- The metric on `Matrix m n 𝕜` arising from the operator norm given by the identification with
(continuous) linear maps of `EuclideanSpace`. -/
def instL2OpMetricSpace : MetricSpace (Matrix m n 𝕜) := by
  /- We first replace the topology so that we can automatically replace the uniformity using
  `UniformAddGroup.toUniformSpace_eq`. -/
  letI normed_add_comm_group : NormedAddCommGroup (Matrix m n 𝕜) :=
    { l2OpNormedAddCommGroupAux.replaceTopology <|
        (toEuclideanLin (𝕜 := 𝕜) (m := m) (n := n)).trans toContinuousLinearMap
        |>.toContinuousLinearEquiv.toHomeomorph.isInducing.eq_induced with
      norm := l2OpNormedAddCommGroupAux.norm
      dist_eq := l2OpNormedAddCommGroupAux.dist_eq }
  exact normed_add_comm_group.replaceUniformity <| by
    congr
    rw [← @UniformAddGroup.toUniformSpace_eq _ (Matrix.instUniformSpace m n 𝕜) _ _]
    rw [@UniformAddGroup.toUniformSpace_eq _ PseudoEMetricSpace.toUniformSpace _ _]


/-- The norm structure on `Matrix m n 𝕜` arising from the operator norm given by the identification
with (continuous) linear maps of `EuclideanSpace`. -/
def instL2OpNormedAddCommGroup : NormedAddCommGroup (Matrix m n 𝕜) where
  norm := l2OpNormedAddCommGroupAux.norm
  dist_eq := l2OpNormedAddCommGroupAux.dist_eq


lemma l2_opNorm_def (A : Matrix m n 𝕜) :
    ‖A‖ = ‖(toEuclideanLin (𝕜 := 𝕜) (m := m) (n := n)).trans toContinuousLinearMap A‖ := rfl


@[deprecated (since := "2024-02-02")] alias l2_op_norm_def := l2_opNorm_def


lemma l2_opNNNorm_def (A : Matrix m n 𝕜) :
    ‖A‖₊ = ‖(toEuclideanLin (𝕜 := 𝕜) (m := m) (n := n)).trans toContinuousLinearMap A‖₊ := rfl


@[deprecated (since := "2024-02-02")] alias l2_op_nnnorm_def := l2_opNNNorm_def


lemma l2_opNorm_conjTranspose [DecidableEq m] (A : Matrix m n 𝕜) : ‖Aᴴ‖ = ‖A‖ := by
  rw [l2_opNorm_def, toEuclideanLin_eq_toLin_orthonormal, LinearEquiv.trans_apply,
    toLin_conjTranspose, adjoint_toContinuousLinearMap]
  /-
    𝕜 : Type u_1
    m : Type u_2
    n : Type u_3
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    A : Matrix m n 𝕜
    ⊢ Eq (Norm.norm (ContinuousLinearMap.adjoint (LinearMap.toContinuousLinearMap  …
  -/
  exact ContinuousLinearMap.adjoint.norm_map _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias l2_op_norm_conjTranspose := l2_opNorm_conjTranspose


lemma l2_opNNNorm_conjTranspose [DecidableEq m] (A : Matrix m n 𝕜) : ‖Aᴴ‖₊ = ‖A‖₊ :=
  Subtype.ext <| l2_opNorm_conjTranspose _


@[deprecated (since := "2024-02-02")] alias l2_op_nnnorm_conjTranspose := l2_opNNNorm_conjTranspose


lemma l2_opNorm_conjTranspose_mul_self (A : Matrix m n 𝕜) : ‖Aᴴ * A‖ = ‖A‖ * ‖A‖ := by
  classical
  rw [l2_opNorm_def, toEuclideanLin_eq_toLin_orthonormal, LinearEquiv.trans_apply,
    Matrix.toLin_mul (v₂ := (EuclideanSpace.basisFun m 𝕜).toBasis), toLin_conjTranspose]
  exact ContinuousLinearMap.norm_adjoint_comp_self _


@[deprecated (since := "2024-02-02")]
alias l2_op_norm_conjTranspose_mul_self := l2_opNorm_conjTranspose_mul_self


lemma l2_opNNNorm_conjTranspose_mul_self (A : Matrix m n 𝕜) : ‖Aᴴ * A‖₊ = ‖A‖₊ * ‖A‖₊ :=
  Subtype.ext <| l2_opNorm_conjTranspose_mul_self _


@[deprecated (since := "2024-02-02")]
alias l2_op_nnnorm_conjTranspose_mul_self := l2_opNNNorm_conjTranspose_mul_self

-- note: with only a type ascription in the left-hand side, Lean picks the wrong norm.

lemma l2_opNorm_mulVec (A : Matrix m n 𝕜) (x : EuclideanSpace 𝕜 n) :
    ‖(EuclideanSpace.equiv m 𝕜).symm <| A *ᵥ x‖ ≤ ‖A‖ * ‖x‖ :=
  toEuclideanLin (n := n) (m := m) (𝕜 := 𝕜) |>.trans toContinuousLinearMap A |>.le_opNorm x


@[deprecated (since := "2024-02-02")] alias l2_op_norm_mulVec := l2_opNorm_mulVec


lemma l2_opNNNorm_mulVec (A : Matrix m n 𝕜) (x : EuclideanSpace 𝕜 n) :
    ‖(EuclideanSpace.equiv m 𝕜).symm <| A *ᵥ x‖₊ ≤ ‖A‖₊ * ‖x‖₊ :=
  A.l2_opNorm_mulVec x


@[deprecated (since := "2024-02-02")] alias l2_op_nnnorm_mulVec := l2_opNNNorm_mulVec


lemma l2_opNorm_mul (A : Matrix m n 𝕜) (B : Matrix n l 𝕜) :
    ‖A * B‖ ≤ ‖A‖ * ‖B‖ := by
  /-
    𝕜 : Type u_1
    m : Type u_2
    n : Type u_3
    l : Type u_4
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    A : Matrix m n 𝕜
    B : Matrix n l 𝕜
    ⊢ LE.le (Norm.norm (HMul.hMul A B)) (HMul.hMul (Norm.norm A) (Norm.norm B))
  -/
  simp only [l2_opNorm_def]
  have := (toEuclideanLin (n := n) (m := m) (𝕜 := 𝕜) ≪≫ₗ toContinuousLinearMap) A
    |>.opNorm_comp_le <| (toEuclideanLin (n := l) (m := n) (𝕜 := 𝕜) ≪≫ₗ toContinuousLinearMap) B
  /-
    𝕜 : Type u_1
    m : Type u_2
    n : Type u_3
    l : Type u_4
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    A : Matrix m n 𝕜
    B : Matrix n l 𝕜
    this : LE.le (Norm.norm (((Matrix.toEuclideanLin.trans LinearMap.toContinuousL …
    ⊢ LE.le (Norm.norm ((Matrix.toEuclideanLin.trans LinearMap.toContinuousLinearM …
  -/
  convert this
  /-
    case h.e'_3.h.e'_3
    𝕜 : Type u_1
    m : Type u_2
    n : Type u_3
    l : Type u_4
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    A : Matrix m n 𝕜
    B : Matrix n l 𝕜
    this : LE.le (Norm.norm (((Matrix.toEuclideanLin.trans LinearMap.toContinuousL …
    ⊢ Eq ((Matrix.toEuclideanLin.trans LinearMap.toContinuousLinearMap) (HMul.hMul …
  -/
  ext1 x
  /-
    case h.e'_3.h.e'_3.h
    𝕜 : Type u_1
    m : Type u_2
    n : Type u_3
    l : Type u_4
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    A : Matrix m n 𝕜
    B : Matrix n l 𝕜
    this : LE.le (Norm.norm (((Matrix.toEuclideanLin.trans LinearMap.toContinuousL …
    x : EuclideanSpace 𝕜 l
    ⊢ Eq (((Matrix.toEuclideanLin.trans LinearMap.toContinuousLinearMap) (HMul.hMu …
  -/
  exact congr($(Matrix.toLin'_mul A B) x)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias l2_op_norm_mul := l2_opNorm_mul


lemma l2_opNNNorm_mul (A : Matrix m n 𝕜) (B : Matrix n l 𝕜) : ‖A * B‖₊ ≤ ‖A‖₊ * ‖B‖₊ :=
  l2_opNorm_mul A B


@[deprecated (since := "2024-02-02")] alias l2_op_nnnorm_mul := l2_opNNNorm_mul


/-- The normed algebra structure on `Matrix n n 𝕜` arising from the operator norm given by the
identification with (continuous) linear endmorphisms of `EuclideanSpace 𝕜 n`. -/
def instL2OpNormedSpace : NormedSpace 𝕜 (Matrix m n 𝕜) where
  norm_smul_le r x := by
    /-
      𝕜 : Type u_1
      m : Type u_2
      n : Type u_3
      l : Type u_4
      E : Type u_5
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq n
      inst✝¹ : Fintype l
      inst✝ : DecidableEq l
      r : 𝕜
      x : Matrix m n 𝕜
      ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
    rw [l2_opNorm_def, LinearEquiv.map_smul]
    /-
      𝕜 : Type u_1
      m : Type u_2
      n : Type u_3
      l : Type u_4
      E : Type u_5
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq n
      inst✝¹ : Fintype l
      inst✝ : DecidableEq l
      r : 𝕜
      x : Matrix m n 𝕜
      ⊢ LE.le (Norm.norm (HSMul.hSMul r ((Matrix.toEuclideanLin.trans LinearMap.toCo …
    -/
    exact norm_smul_le r ((toEuclideanLin (𝕜 := 𝕜) (m := m) (n := n)).trans toContinuousLinearMap x)
    /-
      🎉 no goals
    -/


/-- The normed ring structure on `Matrix n n 𝕜` arising from the operator norm given by the
identification with (continuous) linear endmorphisms of `EuclideanSpace 𝕜 n`. -/
def instL2OpNormedRing : NormedRing (Matrix n n 𝕜) where
  dist_eq := l2OpNormedRingAux.dist_eq
  norm_mul := l2OpNormedRingAux.norm_mul


/-- This is the same as `Matrix.l2_opNorm_def`, but with a more bundled RHS for square matrices. -/
lemma cstar_norm_def (A : Matrix n n 𝕜) : ‖A‖ = ‖toEuclideanCLM (n := n) (𝕜 := 𝕜) A‖ := rfl


/-- This is the same as `Matrix.l2_opNNNorm_def`, but with a more bundled RHS for square
matrices. -/
lemma cstar_nnnorm_def (A : Matrix n n 𝕜) : ‖A‖₊ = ‖toEuclideanCLM (n := n) (𝕜 := 𝕜) A‖₊ := rfl


/-- The normed algebra structure on `Matrix n n 𝕜` arising from the operator norm given by the
identification with (continuous) linear endmorphisms of `EuclideanSpace 𝕜 n`. -/
def instL2OpNormedAlgebra : NormedAlgebra 𝕜 (Matrix n n 𝕜) where
  norm_smul_le := norm_smul_le


/-- The operator norm on `Matrix n n 𝕜` given by the identification with (continuous) linear
endmorphisms of `EuclideanSpace 𝕜 n` makes it into a `L2OpRing`. -/
lemma instCStarRing : CStarRing (Matrix n n 𝕜) where
  norm_mul_self_le M := le_of_eq <| Eq.symm <| l2_opNorm_conjTranspose_mul_self M


