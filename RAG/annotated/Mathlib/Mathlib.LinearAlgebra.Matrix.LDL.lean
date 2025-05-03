local notation "⟪" x ", " y "⟫ₑ" =>
  @inner 𝕜 _ _ ((WithLp.equiv 2 _).symm x) ((WithLp.equiv _ _).symm y)


/-- The inverse of the lower triangular matrix `L` of the LDL-decomposition. It is obtained by
applying Gram-Schmidt-Orthogonalization w.r.t. the inner product induced by `Sᵀ` on the standard
basis vectors `Pi.basisFun`. -/
noncomputable def LDL.lowerInv : Matrix n n 𝕜 :=
  @gramSchmidt 𝕜 (n → 𝕜) _ (_ : _) (InnerProductSpace.ofMatrix hS.transpose) n _ _ _
    (Pi.basisFun 𝕜 n)


theorem LDL.lowerInv_eq_gramSchmidtBasis :
    LDL.lowerInv hS =
      ((Pi.basisFun 𝕜 n).toMatrix
          (@gramSchmidtBasis 𝕜 (n → 𝕜) _ (_ : _) (InnerProductSpace.ofMatrix hS.transpose) n _ _ _
            (Pi.basisFun 𝕜 n)))ᵀ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    ⊢ Eq (LDL.lowerInv hS) ((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi.basi …
  -/
  letI := NormedAddCommGroup.ofMatrix hS.transpose
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    this : NormedAddCommGroup (n → 𝕜) := Matrix.NormedAddCommGroup.ofMatrix ⋯
    ⊢ Eq (LDL.lowerInv hS) ((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi.basi …
  -/
  letI := InnerProductSpace.ofMatrix hS.transpose
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    this✝ : NormedAddCommGroup (n → 𝕜) := Matrix.NormedAddCommGroup.ofMatrix ⋯
    this : InnerProductSpace 𝕜 (n → 𝕜) := Matrix.InnerProductSpace.ofMatrix ⋯
    ⊢ Eq (LDL.lowerInv hS) ((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi.basi …
  -/
  ext i j
  /-
    case a
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    this✝ : NormedAddCommGroup (n → 𝕜) := Matrix.NormedAddCommGroup.ofMatrix ⋯
    this : InnerProductSpace 𝕜 (n → 𝕜) := Matrix.InnerProductSpace.ofMatrix ⋯
    i j : n
    ⊢ Eq (LDL.lowerInv hS i j) (((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi …
  -/
  rw [LDL.lowerInv, Basis.coePiBasisFun.toMatrix_eq_transpose, coe_gramSchmidtBasis]
  /-
    case a
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    this✝ : NormedAddCommGroup (n → 𝕜) := Matrix.NormedAddCommGroup.ofMatrix ⋯
    this : InnerProductSpace 𝕜 (n → 𝕜) := Matrix.InnerProductSpace.ofMatrix ⋯
    i j : n
    ⊢ Eq (gramSchmidt 𝕜 (⇑(Pi.basisFun 𝕜 n)) i j) ((Matrix.transpose (gramSchmidt  …
  -/
  rfl
  /-
    🎉 no goals
  -/


noncomputable instance LDL.invertibleLowerInv : Invertible (LDL.lowerInv hS) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    ⊢ Invertible (LDL.lowerInv hS)
  -/
  rw [LDL.lowerInv_eq_gramSchmidtBasis]
  haveI :=
    Basis.invertibleToMatrix (Pi.basisFun 𝕜 n)
      (@gramSchmidtBasis 𝕜 (n → 𝕜) _ (_ : _) (InnerProductSpace.ofMatrix hS.transpose) n _ _ _
        (Pi.basisFun 𝕜 n))
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    this : Invertible ((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi.basisFun  …
    ⊢ Invertible ((Pi.basisFun 𝕜 n).toMatrix ⇑(gramSchmidtBasis (Pi.basisFun 𝕜 n)) …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem LDL.lowerInv_orthogonal {i j : n} (h₀ : i ≠ j) :
    ⟪LDL.lowerInv hS i, Sᵀ *ᵥ LDL.lowerInv hS j⟫ₑ = 0 :=
  @gramSchmidt_orthogonal 𝕜 _ _ (_ : _) (InnerProductSpace.ofMatrix hS.transpose) _ _ _ _ _ _ _ h₀


/-- The entries of the diagonal matrix `D` of the LDL decomposition. -/
noncomputable def LDL.diagEntries : n → 𝕜 := fun i =>
  ⟪star (LDL.lowerInv hS i), S *ᵥ star (LDL.lowerInv hS i)⟫ₑ


/-- The diagonal matrix `D` of the LDL decomposition. -/
noncomputable def LDL.diag : Matrix n n 𝕜 :=
  Matrix.diagonal (LDL.diagEntries hS)


theorem LDL.lowerInv_triangular {i j : n} (hij : i < j) : LDL.lowerInv hS i j = 0 := by
  rw [←
    @gramSchmidt_triangular 𝕜 (n → 𝕜) _ (_ : _) (InnerProductSpace.ofMatrix hS.transpose) n _ _ _
      i j hij (Pi.basisFun 𝕜 n),
    Pi.basisFun_repr, LDL.lowerInv]


/-- Inverse statement of **LDL decomposition**: we can conjugate a positive definite matrix
by some lower triangular matrix and get a diagonal matrix. -/
theorem LDL.diag_eq_lowerInv_conj : LDL.diag hS = LDL.lowerInv hS * S * (LDL.lowerInv hS)ᴴ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    ⊢ Eq (LDL.diag hS) (HMul.hMul (HMul.hMul (LDL.lowerInv hS) S) (LDL.lowerInv hS …
  -/
  ext i j
  /-
    case a
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    i j : n
    ⊢ Eq (LDL.diag hS i j) (HMul.hMul (HMul.hMul (LDL.lowerInv hS) S) (LDL.lowerIn …
  -/
  by_cases hij : i = j
  · simp only [diag, diagEntries, EuclideanSpace.inner_piLp_equiv_symm, star_star, hij,
    diagonal_apply_eq, Matrix.mul_assoc]
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : RCLike 𝕜
      n : Type u_2
      inst✝³ : LinearOrder n
      inst✝² : WellFoundedLT n
      inst✝¹ : LocallyFiniteOrderBot n
      S : Matrix n n 𝕜
      inst✝ : Fintype n
      hS : S.PosDef
      i j : n
      hij : Eq i j
      ⊢ Eq (dotProduct (LDL.lowerInv hS j) (S.mulVec (Star.star (LDL.lowerInv hS j)) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : RCLike 𝕜
      n : Type u_2
      inst✝³ : LinearOrder n
      inst✝² : WellFoundedLT n
      inst✝¹ : LocallyFiniteOrderBot n
      S : Matrix n n 𝕜
      inst✝ : Fintype n
      hS : S.PosDef
      i j : n
      hij : Not (Eq i j)
      ⊢ Eq (LDL.diag hS i j) (HMul.hMul (HMul.hMul (LDL.lowerInv hS) S) (LDL.lowerIn …
    -/
  · simp only [LDL.diag, hij, diagonal_apply_ne, Ne, not_false_iff, mul_mul_apply]
    rw [conjTranspose, transpose_map, transpose_transpose, dotProduct_mulVec,
      (LDL.lowerInv_orthogonal hS fun h : j = i => hij h.symm).symm, ← inner_conj_symm,
      mulVec_transpose, EuclideanSpace.inner_piLp_equiv_symm, ← RCLike.star_def, ←
      star_dotProduct_star, dotProduct_comm, star_star]
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : RCLike 𝕜
      n : Type u_2
      inst✝³ : LinearOrder n
      inst✝² : WellFoundedLT n
      inst✝¹ : LocallyFiniteOrderBot n
      S : Matrix n n 𝕜
      inst✝ : Fintype n
      hS : S.PosDef
      i j : n
      hij : Not (Eq i j)
      ⊢ Eq (dotProduct (Matrix.vecMul (LDL.lowerInv hS i) S) (Star.star (LDL.lowerIn …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The lower triangular matrix `L` of the LDL decomposition. -/
noncomputable def LDL.lower :=
  (LDL.lowerInv hS)⁻¹


/-- **LDL decomposition**: any positive definite matrix `S` can be
decomposed as `S = LDLᴴ` where `L` is a lower-triangular matrix and `D` is a diagonal matrix. -/
theorem LDL.lower_conj_diag : LDL.lower hS * LDL.diag hS * (LDL.lower hS)ᴴ = S := by
  rw [LDL.lower, conjTranspose_nonsing_inv, Matrix.mul_assoc,
    Matrix.inv_mul_eq_iff_eq_mul_of_invertible (LDL.lowerInv hS),
    Matrix.mul_inv_eq_iff_eq_mul_of_invertible]
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    n : Type u_2
    inst✝³ : LinearOrder n
    inst✝² : WellFoundedLT n
    inst✝¹ : LocallyFiniteOrderBot n
    S : Matrix n n 𝕜
    inst✝ : Fintype n
    hS : S.PosDef
    ⊢ Eq (LDL.diag hS) (HMul.hMul (HMul.hMul (LDL.lowerInv hS) S) (LDL.lowerInv hS …
  -/
  exact LDL.diag_eq_lowerInv_conj hS
  /-
    🎉 no goals
  -/


