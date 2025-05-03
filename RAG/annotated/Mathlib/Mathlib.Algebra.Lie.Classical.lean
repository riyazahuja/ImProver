@[simp]
theorem matrix_trace_commutator_zero [Fintype n] (X Y : Matrix n n R) : Matrix.trace ⁅X, Y⁆ = 0 :=
  calc
    _ = Matrix.trace (X * Y) - Matrix.trace (Y * X) := trace_sub _ _
    _ = Matrix.trace (X * Y) - Matrix.trace (X * Y) :=
      (congr_arg (fun x => _ - x) (Matrix.trace_mul_comm Y X))
    _ = 0 := sub_self _


/-- The special linear Lie algebra: square matrices of trace zero. -/
def sl [Fintype n] : LieSubalgebra R (Matrix n n R) :=
  { LinearMap.ker (Matrix.traceLinearMap n R R) with
    lie_mem' := fun _ _ => LinearMap.mem_ker.2 <| matrix_trace_commutator_zero _ _ _ _ }


theorem sl_bracket [Fintype n] (A B : sl n R) : ⁅A, B⁆.val = A.val * B.val - B.val * A.val :=
  rfl


/-- When j ≠ i, the elementary matrices are elements of sl n R, in fact they are part of a natural
basis of `sl n R`. -/
def Eb (h : j ≠ i) : sl n R :=
  ⟨Matrix.stdBasisMatrix i j (1 : R),
    show Matrix.stdBasisMatrix i j (1 : R) ∈ LinearMap.ker (Matrix.traceLinearMap n R R) from
      Matrix.StdBasisMatrix.trace_zero i j (1 : R) h⟩


@[simp]
theorem eb_val (h : j ≠ i) : (Eb R i j h).val = Matrix.stdBasisMatrix i j 1 :=
  rfl


theorem sl_non_abelian [Fintype n] [Nontrivial R] (h : 1 < Fintype.card n) :
    ¬IsLieAbelian (sl n R) := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝³ : DecidableEq n
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    h : LT.lt 1 (Fintype.card n)
    ⊢ Not (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear …
  -/
  rcases Fintype.exists_pair_of_one_lt_card h with ⟨j, i, hij⟩
  /-
    case intro.intro
    n : Type u_1
    R : Type u₂
    inst✝³ : DecidableEq n
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    h : LT.lt 1 (Fintype.card n)
    j i : n
    hij : Ne j i
    ⊢ Not (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear …
  -/
  let A := Eb R i j hij
  /-
    case intro.intro
    n : Type u_1
    R : Type u₂
    inst✝³ : DecidableEq n
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    h : LT.lt 1 (Fintype.card n)
    j i : n
    hij : Ne j i
    A : Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl n R) x := Lie …
    ⊢ Not (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear …
  -/
  let B := Eb R j i hij.symm
  /-
    case intro.intro
    n : Type u_1
    R : Type u₂
    inst✝³ : DecidableEq n
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    h : LT.lt 1 (Fintype.card n)
    j i : n
    hij : Ne j i
    A : Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl n R) x := Lie …
    B : Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl n R) x := Lie …
    ⊢ Not (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear …
  -/
  intro c
  have c' : A.val * B.val = B.val * A.val := by
    rw [← sub_eq_zero, ← sl_bracket, c.trivial, ZeroMemClass.coe_zero]
  /-
    case intro.intro
    n : Type u_1
    R : Type u₂
    inst✝³ : DecidableEq n
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    h : LT.lt 1 (Fintype.card n)
    j i : n
    hij : Ne j i
    A : Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl n R) x := Lie …
    B : Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl n R) x := Lie …
    c : IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.SpecialLinear.sl …
    c' : Eq (HMul.hMul ↑A ↑B) (HMul.hMul ↑B ↑A)
    ⊢ False
  -/
  simpa [A, B, stdBasisMatrix, Matrix.mul_apply, hij] using congr_fun (congr_fun c' i) i
  /-
    🎉 no goals
  -/


/-- The symplectic Lie algebra: skew-adjoint matrices with respect to the canonical skew-symmetric
bilinear form. -/
def sp [Fintype l] : LieSubalgebra R (Matrix (l ⊕ l) (l ⊕ l) R) :=
  skewAdjointMatricesLieSubalgebra (Matrix.J l R)


/-- The definite orthogonal Lie subalgebra: skew-adjoint matrices with respect to the symmetric
bilinear form defined by the identity matrix. -/
def so [Fintype n] : LieSubalgebra R (Matrix n n R) :=
  skewAdjointMatricesLieSubalgebra (1 : Matrix n n R)


@[simp]
theorem mem_so [Fintype n] (A : Matrix n n R) : A ∈ so n R ↔ Aᵀ = -A := by
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    A : Matrix n n R
    ⊢ Iff (Membership.mem (LieAlgebra.Orthogonal.so n R) A) (Eq A.transpose (Neg.n …
  -/
  rw [so, mem_skewAdjointMatricesLieSubalgebra, mem_skewAdjointMatricesSubmodule]
  /-
    n : Type u_1
    R : Type u₂
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : Fintype n
    A : Matrix n n R
    ⊢ Iff (Matrix.IsSkewAdjoint 1 A) (Eq A.transpose (Neg.neg A))
  -/
  simp only [Matrix.IsSkewAdjoint, Matrix.IsAdjointPair, Matrix.mul_one, Matrix.one_mul]
  /-
    🎉 no goals
  -/


/-- The indefinite diagonal matrix with `p` 1s and `q` -1s. -/
def indefiniteDiagonal : Matrix (p ⊕ q) (p ⊕ q) R :=
  Matrix.diagonal <| Sum.elim (fun _ => 1) fun _ => -1


/-- The indefinite orthogonal Lie subalgebra: skew-adjoint matrices with respect to the symmetric
bilinear form defined by the indefinite diagonal matrix. -/
def so' [Fintype p] [Fintype q] : LieSubalgebra R (Matrix (p ⊕ q) (p ⊕ q) R) :=
  skewAdjointMatricesLieSubalgebra <| indefiniteDiagonal p q R


/-- A matrix for transforming the indefinite diagonal bilinear form into the definite one, provided
the parameter `i` is a square root of -1. -/
def Pso (i : R) : Matrix (p ⊕ q) (p ⊕ q) R :=
  Matrix.diagonal <| Sum.elim (fun _ => 1) fun _ => i


theorem pso_inv {i : R} (hi : i * i = -1) : Pso p q R i * Pso p q R (-i) = 1 := by
  /-
    p : Type u_2
    q : Type u_3
    R : Type u₂
    inst✝⁴ : DecidableEq p
    inst✝³ : DecidableEq q
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
  -/
  ext (x y); rcases x with ⟨x⟩|⟨x⟩ <;> rcases y with ⟨y⟩|⟨y⟩
  · -- x y : p
    /-
      case a.inl.inl
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : p
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    by_cases h : x = y <;>
    /-
      case pos
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : p
      h : Eq x y
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    /-
      🎉 no goals
    -/
    simp [Pso, indefiniteDiagonal, h, one_apply]
    /-
      🎉 no goals
    -/
  · -- x : p, y : q
    /-
      case a.inl.inr
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x : p
      y : q
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    simp [Pso, indefiniteDiagonal]
    /-
      🎉 no goals
    -/
  · -- x : q, y : p
    /-
      case a.inr.inl
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x : q
      y : p
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    simp [Pso, indefiniteDiagonal]
    /-
      🎉 no goals
    -/
  · -- x y : q
    /-
      case a.inr.inr
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : q
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    by_cases h : x = y <;>
    /-
      case pos
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : q
      h : Eq x y
      ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i) (LieAlgebra.Orthogonal.Pso …
    -/
    /-
      🎉 no goals
    -/
    simp [Pso, indefiniteDiagonal, h, hi, one_apply]
    /-
      🎉 no goals
    -/


/-- There is a constructive inverse of `Pso p q R i`. -/
def invertiblePso {i : R} (hi : i * i = -1) : Invertible (Pso p q R i) :=
  invertibleOfRightInverse _ _ (pso_inv p q R hi)


theorem indefiniteDiagonal_transform {i : R} (hi : i * i = -1) :
    (Pso p q R i)ᵀ * indefiniteDiagonal p q R * Pso p q R i = 1 := by
  /-
    p : Type u_2
    q : Type u_3
    R : Type u₂
    inst✝⁴ : DecidableEq p
    inst✝³ : DecidableEq q
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
  -/
  ext (x y); rcases x with ⟨x⟩|⟨x⟩ <;> rcases y with ⟨y⟩|⟨y⟩
  · -- x y : p
    /-
      case a.inl.inl
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : p
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    by_cases h : x = y <;>
    /-
      case pos
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : p
      h : Eq x y
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    /-
      🎉 no goals
    -/
    simp [Pso, indefiniteDiagonal, h, one_apply]
    /-
      🎉 no goals
    -/
  · -- x : p, y : q
    /-
      case a.inl.inr
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x : p
      y : q
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    simp [Pso, indefiniteDiagonal]
    /-
      🎉 no goals
    -/
  · -- x : q, y : p
    /-
      case a.inr.inl
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x : q
      y : p
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    simp [Pso, indefiniteDiagonal]
    /-
      🎉 no goals
    -/
  · -- x y : q
    /-
      case a.inr.inr
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : q
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    by_cases h : x = y <;>
    /-
      case pos
      p : Type u_2
      q : Type u_3
      R : Type u₂
      inst✝⁴ : DecidableEq p
      inst✝³ : DecidableEq q
      inst✝² : CommRing R
      inst✝¹ : Fintype p
      inst✝ : Fintype q
      i : R
      hi : Eq (HMul.hMul i i) (-1)
      x y : q
      h : Eq x y
      ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.Pso p q R i).transpose (LieA …
    -/
    /-
      🎉 no goals
    -/
    simp [Pso, indefiniteDiagonal, h, hi, one_apply]
    /-
      🎉 no goals
    -/


/-- An equivalence between the indefinite and definite orthogonal Lie algebras, over a ring
containing a square root of -1. -/
noncomputable def soIndefiniteEquiv {i : R} (hi : i * i = -1) : so' p q R ≃ₗ⁅R⁆ so (p ⊕ q) R := by
  apply
    (skewAdjointMatricesLieSubalgebraEquiv (indefiniteDiagonal p q R) (Pso p q R i)
        (invertiblePso p q R hi)).trans
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁶ : DecidableEq n
    inst✝⁵ : DecidableEq p
    inst✝⁴ : DecidableEq q
    inst✝³ : DecidableEq l
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (skewAdjointMatricesLieSubalgebr …
  -/
  apply LieEquiv.ofEq
  /-
    case h
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁶ : DecidableEq n
    inst✝⁵ : DecidableEq p
    inst✝⁴ : DecidableEq q
    inst✝³ : DecidableEq l
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    ⊢ Eq ↑(skewAdjointMatricesLieSubalgebra (HMul.hMul (HMul.hMul (LieAlgebra.Orth …
  -/
  ext A; rw [indefiniteDiagonal_transform p q R hi]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem soIndefiniteEquiv_apply {i : R} (hi : i * i = -1) (A : so' p q R) :
    (soIndefiniteEquiv p q R hi A : Matrix (p ⊕ q) (p ⊕ q) R) =
      (Pso p q R i)⁻¹ * (A : Matrix (p ⊕ q) (p ⊕ q) R) * Pso p q R i := by
  /-
    p : Type u_2
    q : Type u_3
    R : Type u₂
    inst✝⁴ : DecidableEq p
    inst✝³ : DecidableEq q
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    A : Subtype fun x => Membership.mem (LieAlgebra.Orthogonal.so' p q R) x
    ⊢ Eq (↑((LieAlgebra.Orthogonal.soIndefiniteEquiv p q R hi) A)) (HMul.hMul (HMu …
  -/
  rw [soIndefiniteEquiv, LieEquiv.trans_apply, LieEquiv.ofEq_apply]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    p : Type u_2
    q : Type u_3
    R : Type u₂
    inst✝⁴ : DecidableEq p
    inst✝³ : DecidableEq q
    inst✝² : CommRing R
    inst✝¹ : Fintype p
    inst✝ : Fintype q
    i : R
    hi : Eq (HMul.hMul i i) (-1)
    A : Subtype fun x => Membership.mem (LieAlgebra.Orthogonal.so' p q R) x
    ⊢ Eq (↑((skewAdjointMatricesLieSubalgebraEquiv (LieAlgebra.Orthogonal.indefini …
  -/
  erw [skewAdjointMatricesLieSubalgebraEquiv_apply]
  /-
    🎉 no goals
  -/


/-- A matrix defining a canonical even-rank symmetric bilinear form.

It looks like this as a `2l x 2l` matrix of `l x l` blocks:

   [ 0 1 ]
   [ 1 0 ]
-/
def JD : Matrix (l ⊕ l) (l ⊕ l) R :=
  Matrix.fromBlocks 0 1 1 0


/-- The classical Lie algebra of type D as a Lie subalgebra of matrices associated to the matrix
`JD`. -/
def typeD [Fintype l] :=
  skewAdjointMatricesLieSubalgebra (JD l R)


/-- A matrix transforming the bilinear form defined by the matrix `JD` into a split-signature
diagonal matrix.

It looks like this as a `2l x 2l` matrix of `l x l` blocks:

   [ 1 -1 ]
   [ 1  1 ]
-/
def PD : Matrix (l ⊕ l) (l ⊕ l) R :=
  Matrix.fromBlocks 1 (-1) 1 1


/-- The split-signature diagonal matrix. -/
def S :=
  indefiniteDiagonal l l R


theorem s_as_blocks : S l R = Matrix.fromBlocks 1 0 0 (-1) := by
  /-
    l : Type u_4
    R : Type u₂
    inst✝¹ : DecidableEq l
    inst✝ : CommRing R
    ⊢ Eq (LieAlgebra.Orthogonal.S l R) (Matrix.fromBlocks 1 0 0 (-1))
  -/
  rw [← Matrix.diagonal_one, Matrix.diagonal_neg, Matrix.fromBlocks_diagonal]
  /-
    l : Type u_4
    R : Type u₂
    inst✝¹ : DecidableEq l
    inst✝ : CommRing R
    ⊢ Eq (LieAlgebra.Orthogonal.S l R) (Matrix.diagonal (Sum.elim (fun x => 1) fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem jd_transform [Fintype l] : (PD l R)ᵀ * JD l R * PD l R = (2 : R) • S l R := by
  have h : (PD l R)ᵀ * JD l R = Matrix.fromBlocks 1 1 1 (-1) := by
    simp [PD, JD, Matrix.fromBlocks_transpose, Matrix.fromBlocks_multiply]
  /-
    l : Type u_4
    R : Type u₂
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    h : Eq (HMul.hMul (LieAlgebra.Orthogonal.PD l R).transpose (LieAlgebra.Orthogo …
    ⊢ Eq (HMul.hMul (HMul.hMul (LieAlgebra.Orthogonal.PD l R).transpose (LieAlgebr …
  -/
  rw [h, PD, s_as_blocks, Matrix.fromBlocks_multiply, Matrix.fromBlocks_smul]
  /-
    l : Type u_4
    R : Type u₂
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    h : Eq (HMul.hMul (LieAlgebra.Orthogonal.PD l R).transpose (LieAlgebra.Orthogo …
    ⊢ Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul 1 1) (HMul.hMul 1 1)) (HAdd.hAdd …
  -/
  simp [two_smul]
  /-
    🎉 no goals
  -/


theorem pd_inv [Fintype l] [Invertible (2 : R)] : PD l R * ⅟ (2 : R) • (PD l R)ᵀ = 1 := by
  rw [PD, Matrix.fromBlocks_transpose, Matrix.fromBlocks_smul,
    Matrix.fromBlocks_multiply]
  /-
    l : Type u_4
    R : Type u₂
    inst✝³ : DecidableEq l
    inst✝² : CommRing R
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul 1 (HSMul.hSMul (Invertible.invOf …
  -/
  simp
  /-
    🎉 no goals
  -/


instance invertiblePD [Fintype l] [Invertible (2 : R)] : Invertible (PD l R) :=
  invertibleOfRightInverse _ _ (pd_inv l R)


/-- An equivalence between two possible definitions of the classical Lie algebra of type D. -/
noncomputable def typeDEquivSo' [Fintype l] [Invertible (2 : R)] : typeD l R ≃ₗ⁅R⁆ so' l l R := by
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (LieAlgebra.Orthogonal.typeD l R …
  -/
  apply (skewAdjointMatricesLieSubalgebraEquiv (JD l R) (PD l R) (by infer_instance)).trans
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (skewAdjointMatricesLieSubalgebr …
  -/
  apply LieEquiv.ofEq
  /-
    case h
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ Eq ↑(skewAdjointMatricesLieSubalgebra (HMul.hMul (HMul.hMul (LieAlgebra.Orth …
  -/
  ext A
  rw [jd_transform, ← val_unitOfInvertible (2 : R), ← Units.smul_def, LieSubalgebra.mem_coe,
    mem_skewAdjointMatricesLieSubalgebra_unit_smul]
  /-
    case h.h
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    A : Matrix (Sum l l) (Sum l l) R
    ⊢ Iff (Membership.mem (skewAdjointMatricesLieSubalgebra (LieAlgebra.Orthogonal …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A matrix defining a canonical odd-rank symmetric bilinear form.

It looks like this as a `(2l+1) x (2l+1)` matrix of blocks:

   [ 2 0 0 ]
   [ 0 0 1 ]
   [ 0 1 0 ]

where sizes of the blocks are:

   [`1 x 1` `1 x l` `1 x l`]
   [`l x 1` `l x l` `l x l`]
   [`l x 1` `l x l` `l x l`]
-/
def JB :=
  Matrix.fromBlocks ((2 : R) • (1 : Matrix Unit Unit R)) 0 0 (JD l R)


/-- The classical Lie algebra of type B as a Lie subalgebra of matrices associated to the matrix
`JB`. -/
def typeB [Fintype l] :=
  skewAdjointMatricesLieSubalgebra (JB l R)


/-- A matrix transforming the bilinear form defined by the matrix `JB` into an
almost-split-signature diagonal matrix.

It looks like this as a `(2l+1) x (2l+1)` matrix of blocks:

   [ 1 0  0 ]
   [ 0 1 -1 ]
   [ 0 1  1 ]

where sizes of the blocks are:

   [`1 x 1` `1 x l` `1 x l`]
   [`l x 1` `l x l` `l x l`]
   [`l x 1` `l x l` `l x l`]
-/
def PB :=
  Matrix.fromBlocks (1 : Matrix Unit Unit R) 0 0 (PD l R)


theorem pb_inv [Invertible (2 : R)] : PB l R * Matrix.fromBlocks 1 0 0 (⅟ (PD l R)) = 1 := by
  /-
    l : Type u_4
    R : Type u₂
    inst✝³ : DecidableEq l
    inst✝² : CommRing R
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ Eq (HMul.hMul (LieAlgebra.Orthogonal.PB l R) (Matrix.fromBlocks 1 0 0 (Inver …
  -/
  rw [PB, Matrix.fromBlocks_multiply, mul_invOf_self]
  simp only [Matrix.mul_zero, Matrix.mul_one, Matrix.zero_mul, zero_add, add_zero,
    Matrix.fromBlocks_one]


instance invertiblePB [Invertible (2 : R)] : Invertible (PB l R) :=
  invertibleOfRightInverse _ _ (pb_inv l R)


theorem jb_transform : (PB l R)ᵀ * JB l R * PB l R = (2 : R) • Matrix.fromBlocks 1 0 0 (S l R) := by
  simp [PB, JB, jd_transform, Matrix.fromBlocks_transpose, Matrix.fromBlocks_multiply,
    Matrix.fromBlocks_smul]


theorem indefiniteDiagonal_assoc :
    indefiniteDiagonal (Unit ⊕ l) l R =
      Matrix.reindexLieEquiv (Equiv.sumAssoc Unit l l).symm
        (Matrix.fromBlocks 1 0 0 (indefiniteDiagonal l l R)) := by
  /-
    l : Type u_4
    R : Type u₂
    inst✝² : DecidableEq l
    inst✝¹ : CommRing R
    inst✝ : Fintype l
    ⊢ Eq (LieAlgebra.Orthogonal.indefiniteDiagonal (Sum Unit l) l R) ((Matrix.rein …
  -/
  ext ⟨⟨i₁ | i₂⟩ | i₃⟩ ⟨⟨j₁ | j₂⟩ | j₃⟩ <;>
  -- Porting note: added `Sum.inl_injective.eq_iff`, `Sum.inr_injective.eq_iff`
    simp only [indefiniteDiagonal, Matrix.diagonal_apply, Equiv.sumAssoc_apply_inl_inl,
      Matrix.reindexLieEquiv_apply, Matrix.submatrix_apply, Equiv.symm_symm, Matrix.reindex_apply,
      Sum.elim_inl, if_true, eq_self_iff_true, Matrix.one_apply_eq, Matrix.fromBlocks_apply₁₁,
      DMatrix.zero_apply, Equiv.sumAssoc_apply_inl_inr, if_false, Matrix.fromBlocks_apply₁₂,
      Matrix.fromBlocks_apply₂₁, Matrix.fromBlocks_apply₂₂, Equiv.sumAssoc_apply_inr,
      Sum.elim_inr, Sum.inl_injective.eq_iff, Sum.inr_injective.eq_iff, reduceCtorEq] <;>
    /-
      case a.inl.inl.unit.inl.inr
      l : Type u_4
      R : Type u₂
      inst✝² : DecidableEq l
      inst✝¹ : CommRing R
      inst✝ : Fintype l
      j₃ : l
      ⊢ Eq 0 (0 PUnit.unit (Sum.inl j₃))
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
    congr 1
    /-
      🎉 no goals
    -/


/-- An equivalence between two possible definitions of the classical Lie algebra of type B. -/
noncomputable def typeBEquivSo' [Invertible (2 : R)] : typeB l R ≃ₗ⁅R⁆ so' (Unit ⊕ l) l R := by
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (LieAlgebra.Orthogonal.typeB l R …
  -/
  apply (skewAdjointMatricesLieSubalgebraEquiv (JB l R) (PB l R) (by infer_instance)).trans
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (skewAdjointMatricesLieSubalgebr …
  -/
  symm
  apply
    (skewAdjointMatricesLieSubalgebraEquivTranspose (indefiniteDiagonal (Sum Unit l) l R)
        (Matrix.reindexAlgEquiv _ _ (Equiv.sumAssoc PUnit l l))
        (Matrix.transpose_reindex _ _)).trans
  /-
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (skewAdjointMatricesLieSubalgebr …
  -/
  apply LieEquiv.ofEq
  /-
    case h
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    ⊢ Eq ↑(skewAdjointMatricesLieSubalgebra ((Matrix.reindexAlgEquiv R R (Equiv.su …
  -/
  ext A
  rw [jb_transform, ← val_unitOfInvertible (2 : R), ← Units.smul_def, LieSubalgebra.mem_coe,
    LieSubalgebra.mem_coe, mem_skewAdjointMatricesLieSubalgebra_unit_smul]
  /-
    case h.h
    n : Type u_1
    p : Type u_2
    q : Type u_3
    l : Type u_4
    R : Type u₂
    inst✝⁸ : DecidableEq n
    inst✝⁷ : DecidableEq p
    inst✝⁶ : DecidableEq q
    inst✝⁵ : DecidableEq l
    inst✝⁴ : CommRing R
    inst✝³ : Fintype p
    inst✝² : Fintype q
    inst✝¹ : Fintype l
    inst✝ : Invertible 2
    A : Matrix (Sum PUnit.{1} (Sum l l)) (Sum PUnit.{1} (Sum l l)) R
    ⊢ Iff (Membership.mem (skewAdjointMatricesLieSubalgebra ((Matrix.reindexAlgEqu …
  -/
  simp [indefiniteDiagonal_assoc, S]
  /-
    🎉 no goals
  -/


