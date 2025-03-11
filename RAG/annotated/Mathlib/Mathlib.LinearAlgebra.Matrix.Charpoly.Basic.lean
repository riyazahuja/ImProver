/-- The "characteristic matrix" of `M : Matrix n n R` is the matrix of polynomials $t I - M$.
The determinant of this matrix is the characteristic polynomial.
-/
def charmatrix (M : Matrix n n R) : Matrix n n R[X] :=
  Matrix.scalar n (X : R[X]) - (C : R →+* R[X]).mapMatrix M


theorem charmatrix_apply :
    charmatrix M i j = (Matrix.diagonal fun _ : n => X) i j - C (M i j) :=
  rfl


@[simp]
theorem charmatrix_apply_eq : charmatrix M i i = (X : R[X]) - C (M i i) := by
  simp only [charmatrix, RingHom.mapMatrix_apply, sub_apply, scalar_apply, map_apply,
    diagonal_apply_eq]


@[simp]
theorem charmatrix_apply_ne (h : i ≠ j) : charmatrix M i j = -C (M i j) := by
  simp only [charmatrix, RingHom.mapMatrix_apply, sub_apply, scalar_apply, diagonal_apply_ne _ h,
    map_apply, sub_eq_neg_self]


theorem matPolyEquiv_charmatrix : matPolyEquiv (charmatrix M) = X - C M := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (matPolyEquiv M.charmatrix) (HSub.hSub Polynomial.X (Polynomial.C M))
  -/
  ext k i j
  /-
    case a.a
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    i j : n
    ⊢ Eq ((matPolyEquiv M.charmatrix).coeff k i j) ((HSub.hSub Polynomial.X (Polyn …
  -/
  simp only [matPolyEquiv_coeff_apply, coeff_sub, Pi.sub_apply]
  /-
    case a.a
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    i j : n
    ⊢ Eq ((M.charmatrix i j).coeff k) (HSub.hSub (Polynomial.X.coeff k) ((Polynomi …
  -/
  by_cases h : i = j
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i j : n
      h : Eq i j
      ⊢ Eq ((M.charmatrix i j).coeff k) (HSub.hSub (Polynomial.X.coeff k) ((Polynomi …
    -/
  · subst h
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i : n
      ⊢ Eq ((M.charmatrix i i).coeff k) (HSub.hSub (Polynomial.X.coeff k) ((Polynomi …
    -/
    rw [charmatrix_apply_eq, coeff_sub]
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i : n
      ⊢ Eq (HSub.hSub (Polynomial.X.coeff k) ((Polynomial.C (M i i)).coeff k)) (HSub …
    -/
    simp only [coeff_X, coeff_C]
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i : n
      ⊢ Eq (HSub.hSub (ite (Eq 1 k) 1 0) (ite (Eq k 0) (M i i) 0)) (HSub.hSub (ite ( …
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
    split_ifs <;> simp
                  /-
                    🎉 no goals
                  -/
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i j : n
      h : Not (Eq i j)
      ⊢ Eq ((M.charmatrix i j).coeff k) (HSub.hSub (Polynomial.X.coeff k) ((Polynomi …
    -/
  · rw [charmatrix_apply_ne _ _ _ h, coeff_X, coeff_neg, coeff_C, coeff_C]
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      n : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      k : Nat
      i j : n
      h : Not (Eq i j)
      ⊢ Eq (Neg.neg (ite (Eq k 0) (M i j) 0)) (HSub.hSub (ite (Eq 1 k) 1 0) (ite (Eq …
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
    split_ifs <;> simp [h]
                  /-
                    🎉 no goals
                  -/


theorem charmatrix_reindex (e : n ≃ m) :
    charmatrix (reindex e e M) = reindex e e (charmatrix M) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    M : Matrix n n R
    e : Equiv n m
    ⊢ Eq ((Matrix.reindex e e) M).charmatrix ((Matrix.reindex e e) M.charmatrix)
  -/
  ext i j x
  /-
    case a.a
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    M : Matrix n n R
    e : Equiv n m
    i j : m
    x : Nat
    ⊢ Eq ((((Matrix.reindex e e) M).charmatrix i j).coeff x) (((Matrix.reindex e e …
  -/
  by_cases h : i = j
  /-
    case pos
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    M : Matrix n n R
    e : Equiv n m
    i j : m
    x : Nat
    h : Eq i j
    ⊢ Eq ((((Matrix.reindex e e) M).charmatrix i j).coeff x) (((Matrix.reindex e e …
  -/
  all_goals simp [h]
  /-
    🎉 no goals
  -/


lemma charmatrix_map (M : Matrix n n R) (f : R →+* S) :
    charmatrix (M.map f) = (charmatrix M).map (Polynomial.map f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    f : RingHom R S
    ⊢ Eq (M.map ⇑f).charmatrix (M.charmatrix.map (Polynomial.map f))
  -/
  ext i j
  /-
    case a.a
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    f : RingHom R S
    i j : n
    n✝ : Nat
    ⊢ Eq (((M.map ⇑f).charmatrix i j).coeff n✝) ((M.charmatrix.map (Polynomial.map …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> simp [h, charmatrix, diagonal]
                         /-
                           🎉 no goals
                         -/


lemma charmatrix_fromBlocks :
    charmatrix (fromBlocks M₁₁ M₁₂ M₂₁ M₂₂) =
      fromBlocks (charmatrix M₁₁) (- M₁₂.map C) (- M₂₁.map C) (charmatrix M₂₂) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    M₁₁ : Matrix m m R
    M₁₂ : Matrix m n R
    M₂₁ : Matrix n m R
    M₂₂ : Matrix n n R
    ⊢ Eq (Matrix.fromBlocks M₁₁ M₁₂ M₂₁ M₂₂).charmatrix (Matrix.fromBlocks M₁₁.cha …
  -/
  simp only [charmatrix]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    M₁₁ : Matrix m m R
    M₁₂ : Matrix m n R
    M₂₁ : Matrix n m R
    M₂₂ : Matrix n n R
    ⊢ Eq (HSub.hSub ((Matrix.scalar (Sum m n)) Polynomial.X) (Polynomial.C.mapMatr …
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
  ext (i|i) (j|j) : 2 <;> simp [diagonal]
                          /-
                            🎉 no goals
                          -/

-- TODO: importing block triangular here is somewhat expensive, if more lemmas about it are added
-- to this file, it may be worth extracting things out to Charpoly/Block.lean

@[simp]
lemma charmatrix_blockTriangular_iff {α : Type*} [Preorder α] {M : Matrix n n R} {b : n → α} :
    M.charmatrix.BlockTriangular b ↔ M.BlockTriangular b := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    α : Type u_5
    inst✝ : Preorder α
    M : Matrix n n R
    b : n → α
    ⊢ Iff (M.charmatrix.BlockTriangular b) (M.BlockTriangular b)
  -/
  rw [charmatrix, scalar_apply, RingHom.mapMatrix_apply, (blockTriangular_diagonal _).sub_iff_right]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    α : Type u_5
    inst✝ : Preorder α
    M : Matrix n n R
    b : n → α
    ⊢ Iff ((M.map ⇑Polynomial.C).BlockTriangular b) (M.BlockTriangular b)
  -/
  simp [BlockTriangular]
  /-
    🎉 no goals
  -/


alias ⟨BlockTriangular.of_charmatrix, BlockTriangular.charmatrix⟩ := charmatrix_blockTriangular_iff


/-- The characteristic polynomial of a matrix `M` is given by $\det (t I - M)$.
-/
def charpoly (M : Matrix n n R) : R[X] :=
  (charmatrix M).det


theorem charpoly_reindex (e : n ≃ m)
    (M : Matrix n n R) : (reindex e e M).charpoly = M.charpoly := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    e : Equiv n m
    M : Matrix n n R
    ⊢ Eq ((Matrix.reindex e e) M).charpoly M.charpoly
  -/
  unfold Matrix.charpoly
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    m : Type u_3
    n : Type u_4
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    e : Equiv n m
    M : Matrix n n R
    ⊢ Eq ((Matrix.reindex e e) M).charmatrix.det M.charmatrix.det
  -/
  rw [charmatrix_reindex, Matrix.det_reindex_self]
  /-
    🎉 no goals
  -/


lemma charpoly_map (M : Matrix n n R) (f : R →+* S) :
    (M.map f).charpoly = M.charpoly.map f := by
  rw [charpoly, charmatrix_map, ← Polynomial.coe_mapRingHom, charpoly, RingHom.map_det,
    RingHom.mapMatrix_apply]


@[simp]
lemma charpoly_fromBlocks_zero₁₂ :
    (fromBlocks M₁₁ 0 M₂₁ M₂₂).charpoly = (M₁₁.charpoly * M₂₂.charpoly) := by
  simp only [charpoly, charmatrix_fromBlocks, Matrix.map_zero _ (Polynomial.C_0), neg_zero,
    det_fromBlocks_zero₁₂]


@[simp]
lemma charpoly_fromBlocks_zero₂₁ :
    (fromBlocks M₁₁ M₁₂ 0 M₂₂).charpoly = (M₁₁.charpoly * M₂₂.charpoly) := by
  simp only [charpoly, charmatrix_fromBlocks, Matrix.map_zero _ (Polynomial.C_0), neg_zero,
    det_fromBlocks_zero₂₁]


lemma charmatrix_toSquareBlock {α : Type*} [DecidableEq α] {b : n → α} {a : α} :
    (M.toSquareBlock b a).charmatrix = M.charmatrix.toSquareBlock b a := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    M : Matrix n n R
    α : Type u_5
    inst✝ : DecidableEq α
    b : n → α
    a : α
    ⊢ Eq (M.toSquareBlock b a).charmatrix (M.charmatrix.toSquareBlock b a)
  -/
  ext i j : 1
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    M : Matrix n n R
    α : Type u_5
    inst✝ : DecidableEq α
    b : n → α
    a : α
    i j : Subtype fun a_1 => Eq (b a_1) a
    ⊢ Eq ((M.toSquareBlock b a).charmatrix i j) (M.charmatrix.toSquareBlock b a i j)
  -/
  simp [charmatrix_apply, toSquareBlock_def, diagonal_apply, Subtype.ext_iff]
  /-
    🎉 no goals
  -/


lemma BlockTriangular.charpoly {α : Type*} {b : n → α} [LinearOrder α] (h : M.BlockTriangular b) :
    M.charpoly = ∏ a ∈ image b univ, (M.toSquareBlock b a).charpoly := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    M : Matrix n n R
    α : Type u_5
    b : n → α
    inst✝ : LinearOrder α
    h : M.BlockTriangular b
    ⊢ Eq M.charpoly ((Finset.image b Finset.univ).prod fun a => (M.toSquareBlock b …
  -/
  simp only [Matrix.charpoly, h.charmatrix.det, charmatrix_toSquareBlock]
  /-
    🎉 no goals
  -/


lemma charpoly_of_upperTriangular [LinearOrder n] (M : Matrix n n R) (h : M.BlockTriangular id) :
    M.charpoly = ∏ i : n, (X - C (M i i)) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : LinearOrder n
    M : Matrix n n R
    h : M.BlockTriangular id
    ⊢ Eq M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial. …
  -/
  simp [charpoly, det_of_upperTriangular h.charmatrix]
  /-
    🎉 no goals
  -/

-- This proof follows http://drorbn.net/AcademicPensieve/2015-12/CayleyHamilton.pdf

/-- The **Cayley-Hamilton Theorem**, that the characteristic polynomial of a matrix,
applied to the matrix itself, is zero.

This holds over any commutative ring.

See `LinearMap.aeval_self_charpoly` for the equivalent statement about endomorphisms.
-/
theorem aeval_self_charpoly (M : Matrix n n R) : aeval M M.charpoly = 0 := by
  -- We begin with the fact $χ_M(t) I = adjugate (t I - M) * (t I - M)$,
  -- as an identity in `Matrix n n R[X]`.
  have h : M.charpoly • (1 : Matrix n n R[X]) = adjugate (charmatrix M) * charmatrix M :=
    (adjugate_mul _).symm
  -- Using the algebra isomorphism `Matrix n n R[X] ≃ₐ[R] Polynomial (Matrix n n R)`,
  -- we have the same identity in `Polynomial (Matrix n n R)`.
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (HSMul.hSMul M.charpoly 1) (HMul.hMul M.charmatrix.adjugate M.charmatrix)
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  apply_fun matPolyEquiv at h
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (matPolyEquiv (HSMul.hSMul M.charpoly 1)) (matPolyEquiv (HMul.hMul M.ch …
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  simp only [_root_.map_mul, matPolyEquiv_charmatrix] at h
  -- Because the coefficient ring `Matrix n n R` is non-commutative,
  -- evaluation at `M` is not multiplicative.
  -- However, any polynomial which is a product of the form $N * (t I - M)$
  -- is sent to zero, because the evaluation function puts the polynomial variable
  -- to the right of any coefficients, so everything telescopes.
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (matPolyEquiv (HSMul.hSMul M.charpoly 1)) (HMul.hMul (matPolyEquiv M.ch …
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  apply_fun fun p => p.eval M at h
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (Polynomial.eval M (matPolyEquiv (HSMul.hSMul M.charpoly 1))) (Polynomi …
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  rw [eval_mul_X_sub_C] at h
  -- Now $χ_M (t) I$, when thought of as a polynomial of matrices
  -- and evaluated at some `N` is exactly $χ_M (N)$.
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (Polynomial.eval M (matPolyEquiv (HSMul.hSMul M.charpoly 1))) 0
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  rw [matPolyEquiv_smul_one, eval_map] at h
  -- Thus we have $χ_M(M) = 0$, which is the desired result.
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    h : Eq (Polynomial.eval₂ (algebraMap R (Matrix n n R)) M M.charpoly) 0
    ⊢ Eq ((Polynomial.aeval M) M.charpoly) 0
  -/
  exact h
  /-
    🎉 no goals
  -/


