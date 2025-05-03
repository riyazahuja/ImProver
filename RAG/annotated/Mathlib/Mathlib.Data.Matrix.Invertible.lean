/-- A copy of `invOf_mul_cancel_left` for rectangular matrices. -/
protected theorem invOf_mul_cancel_left (A : Matrix n n α) (B : Matrix n m α) [Invertible A] :
                            /-
                              m : Type u_1
                              n : Type u_2
                              α : Type u_3
                              inst✝³ : Fintype n
                              inst✝² : DecidableEq n
                              inst✝¹ : Semiring α
                              A : Matrix n n α
                              B : Matrix n m α
                              inst✝ : Invertible A
                              ⊢ Eq (HMul.hMul (Invertible.invOf A) (HMul.hMul A B)) B
                            -/
    ⅟ A * (A * B) = B := by rw [← Matrix.mul_assoc, invOf_mul_self, Matrix.one_mul]
                            /-
                              🎉 no goals
                            -/


/-- A copy of `mul_invOf_cancel_left` for rectangular matrices. -/
protected theorem mul_invOf_cancel_left (A : Matrix n n α) (B : Matrix n m α) [Invertible A] :
                            /-
                              m : Type u_1
                              n : Type u_2
                              α : Type u_3
                              inst✝³ : Fintype n
                              inst✝² : DecidableEq n
                              inst✝¹ : Semiring α
                              A : Matrix n n α
                              B : Matrix n m α
                              inst✝ : Invertible A
                              ⊢ Eq (HMul.hMul A (HMul.hMul (Invertible.invOf A) B)) B
                            -/
    A * (⅟ A * B) = B := by rw [← Matrix.mul_assoc, mul_invOf_self, Matrix.one_mul]
                            /-
                              🎉 no goals
                            -/


/-- A copy of `invOf_mul_cancel_right` for rectangular matrices. -/
protected theorem invOf_mul_cancel_right (A : Matrix m n α) (B : Matrix n n α) [Invertible B] :
                          /-
                            m : Type u_1
                            n : Type u_2
                            α : Type u_3
                            inst✝³ : Fintype n
                            inst✝² : DecidableEq n
                            inst✝¹ : Semiring α
                            A : Matrix m n α
                            B : Matrix n n α
                            inst✝ : Invertible B
                            ⊢ Eq (HMul.hMul (HMul.hMul A (Invertible.invOf B)) B) A
                          -/
    A * ⅟ B * B = A := by rw [Matrix.mul_assoc, invOf_mul_self, Matrix.mul_one]
                          /-
                            🎉 no goals
                          -/


/-- A copy of `mul_invOf_cancel_right` for rectangular matrices. -/
protected theorem mul_invOf_cancel_right (A : Matrix m n α) (B : Matrix n n α) [Invertible B] :
                          /-
                            m : Type u_1
                            n : Type u_2
                            α : Type u_3
                            inst✝³ : Fintype n
                            inst✝² : DecidableEq n
                            inst✝¹ : Semiring α
                            A : Matrix m n α
                            B : Matrix n n α
                            inst✝ : Invertible B
                            ⊢ Eq (HMul.hMul (HMul.hMul A B) (Invertible.invOf B)) A
                          -/
    A * B * ⅟ B = A := by rw [Matrix.mul_assoc, mul_invOf_self, Matrix.mul_one]
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-09-07")]
protected alias invOf_mul_self_assoc := Matrix.invOf_mul_cancel_left

@[deprecated (since := "2024-09-07")]
protected alias mul_invOf_self_assoc := Matrix.mul_invOf_cancel_left

@[deprecated (since := "2024-09-07")]
protected alias mul_invOf_mul_self_cancel := Matrix.invOf_mul_cancel_right

@[deprecated (since := "2024-09-07")]
protected alias mul_mul_invOf_self_cancel := Matrix.mul_invOf_cancel_right


/-- The conjugate transpose of an invertible matrix is invertible. -/
instance invertibleConjTranspose [Invertible A] : Invertible Aᴴ := Invertible.star _


lemma conjTranspose_invOf [Invertible A] [Invertible Aᴴ] : (⅟A)ᴴ = ⅟(Aᴴ) := star_invOf _


/-- A matrix is invertible if the conjugate transpose is invertible. -/
def invertibleOfInvertibleConjTranspose [Invertible Aᴴ] : Invertible A := by
  /-
    m : Type u_1
    n : Type u_2
    α : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : Semiring α
    inst✝¹ : StarRing α
    A : Matrix n n α
    inst✝ : Invertible A.conjTranspose
    ⊢ Invertible A
  -/
  rw [← conjTranspose_conjTranspose A, ← star_eq_conjTranspose]
  /-
    m : Type u_1
    n : Type u_2
    α : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : Semiring α
    inst✝¹ : StarRing α
    A : Matrix n n α
    inst✝ : Invertible A.conjTranspose
    ⊢ Invertible (Star.star A.conjTranspose)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp] lemma isUnit_conjTranspose : IsUnit Aᴴ ↔ IsUnit A := isUnit_star


/-- The transpose of an invertible matrix is invertible. -/
instance invertibleTranspose [Invertible A] : Invertible Aᵀ where
  invOf := (⅟A)ᵀ
                       /-
                         m : Type u_1
                         n : Type u_2
                         α : Type u_3
                         inst✝³ : Fintype n
                         inst✝² : DecidableEq n
                         inst✝¹ : CommSemiring α
                         A : Matrix n n α
                         inst✝ : Invertible A
                         ⊢ Eq (HMul.hMul (Invertible.invOf A).transpose A.transpose) 1
                       -/
  invOf_mul_self := by rw [← transpose_mul, mul_invOf_self, transpose_one]
                       /-
                         🎉 no goals
                       -/
                       /-
                         m : Type u_1
                         n : Type u_2
                         α : Type u_3
                         inst✝³ : Fintype n
                         inst✝² : DecidableEq n
                         inst✝¹ : CommSemiring α
                         A : Matrix n n α
                         inst✝ : Invertible A
                         ⊢ Eq (HMul.hMul A.transpose (Invertible.invOf A).transpose) 1
                       -/
  mul_invOf_self := by rw [← transpose_mul, invOf_mul_self, transpose_one]
                       /-
                         🎉 no goals
                       -/


lemma transpose_invOf [Invertible A] [Invertible Aᵀ] : (⅟A)ᵀ = ⅟(Aᵀ) := by
  /-
    n : Type u_2
    α : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommSemiring α
    A : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible A.transpose
    ⊢ Eq (Invertible.invOf A).transpose (Invertible.invOf A.transpose)
  -/
  letI := invertibleTranspose A
  /-
    n : Type u_2
    α : Type u_3
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommSemiring α
    A : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible A.transpose
    this : Invertible A.transpose := A.invertibleTranspose
    ⊢ Eq (Invertible.invOf A).transpose (Invertible.invOf A.transpose)
  -/
  convert (rfl : _ = ⅟(Aᵀ))
  /-
    🎉 no goals
  -/


/-- `Aᵀ` is invertible when `A` is. -/
def invertibleOfInvertibleTranspose [Invertible Aᵀ] : Invertible A where
  invOf := (⅟(Aᵀ))ᵀ
                       /-
                         m : Type u_1
                         n : Type u_2
                         α : Type u_3
                         inst✝³ : Fintype n
                         inst✝² : DecidableEq n
                         inst✝¹ : CommSemiring α
                         A : Matrix n n α
                         inst✝ : Invertible A.transpose
                         ⊢ Eq (HMul.hMul (Invertible.invOf A.transpose).transpose A) 1
                       -/
  invOf_mul_self := by rw [← transpose_one, ← mul_invOf_self Aᵀ, transpose_mul, transpose_transpose]
                       /-
                         🎉 no goals
                       -/
                       /-
                         m : Type u_1
                         n : Type u_2
                         α : Type u_3
                         inst✝³ : Fintype n
                         inst✝² : DecidableEq n
                         inst✝¹ : CommSemiring α
                         A : Matrix n n α
                         inst✝ : Invertible A.transpose
                         ⊢ Eq (HMul.hMul A (Invertible.invOf A.transpose).transpose) 1
                       -/
  mul_invOf_self := by rw [← transpose_one, ← invOf_mul_self Aᵀ, transpose_mul, transpose_transpose]
                       /-
                         🎉 no goals
                       -/


/-- Together `Matrix.invertibleTranspose` and `Matrix.invertibleOfInvertibleTranspose` form an
equivalence, although both sides of the equiv are subsingleton anyway. -/
@[simps]
def transposeInvertibleEquivInvertible : Invertible Aᵀ ≃ Invertible A where
  toFun := @invertibleOfInvertibleTranspose _ _ _ _ _ _
  invFun := @invertibleTranspose _ _ _ _ _ _
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


@[simp] lemma isUnit_transpose : IsUnit Aᵀ ↔ IsUnit A := by
  simp only [← nonempty_invertible_iff_isUnit,
    (transposeInvertibleEquivInvertible A).nonempty_congr]


lemma add_mul_mul_invOf_mul_eq_one :
    (A + U*C*V)*(⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A) = 1 := by
  calc
    (A + U*C*V)*(⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A)
    _ = A*⅟A - A*⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A + U*C*V*⅟A - U*C*V*⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A := by
      simp_rw [add_sub_assoc, add_mul, mul_sub, Matrix.mul_assoc]
    _ = (1 + U*C*V*⅟A) - (U*⅟(⅟C + V*⅟A*U)*V*⅟A + U*C*V*⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A) := by
      rw [mul_invOf_self, Matrix.one_mul]
      abel
    _ = 1 + U*C*V*⅟A - (U + U*C*V*⅟A*U)*⅟(⅟C + V*⅟A*U)*V*⅟A := by
      rw [sub_right_inj, Matrix.add_mul, Matrix.add_mul, Matrix.add_mul]
    _ = 1 + U*C*V*⅟A - U*C*(⅟C + V*⅟A*U)*⅟(⅟C + V*⅟A*U)*V*⅟A := by
      congr
      simp only [Matrix.mul_add, Matrix.mul_invOf_cancel_right, ← Matrix.mul_assoc]
    _ = 1 := by
      rw [Matrix.mul_invOf_cancel_right]
      abel


/-- Like `add_mul_mul_invOf_mul_eq_one`, but with multiplication reversed. -/
lemma add_mul_mul_invOf_mul_eq_one' :
    (⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A)*(A + U*C*V) = 1 := by
  calc
    (⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A)*(A + U*C*V)
    _ = ⅟A*A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A*A + ⅟A*U*C*V - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A*U*C*V := by
      simp_rw [add_sub_assoc, _root_.mul_add, _root_.sub_mul, Matrix.mul_assoc]
    _ = (1 + ⅟A*U*C*V) - (⅟A*U*⅟(⅟C + V*⅟A*U)*V + ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A*U*C*V) := by
      rw [invOf_mul_self, Matrix.invOf_mul_cancel_right]
      abel
    _ = 1 + ⅟A*U*C*V - ⅟A*U*⅟(⅟C + V*⅟A*U)*(V + V*⅟A*U*C*V) := by
      rw [sub_right_inj, Matrix.mul_add]
      simp_rw [Matrix.mul_assoc]
    _ = 1 + ⅟A*U*C*V - ⅟A*U*⅟(⅟C + V*⅟A*U)*(⅟C + V*⅟A*U)*C*V := by
      congr 1
      simp only [Matrix.mul_add, Matrix.add_mul, ← Matrix.mul_assoc,
        Matrix.invOf_mul_cancel_right]
    _ = 1 := by
      rw [Matrix.invOf_mul_cancel_right]
      abel


/-- If matrices `A`, `C`, and `C⁻¹ + V * A⁻¹ * U` are invertible, then so is `A + U * C * V`-/
def invertibleAddMulMul : Invertible (A + U*C*V) where
  invOf := ⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A
  invOf_mul_self := add_mul_mul_invOf_mul_eq_one' _ _ _ _
  mul_invOf_self := add_mul_mul_invOf_mul_eq_one _ _ _ _


/-- The **Woodbury Identity** (`⅟` version). -/
theorem invOf_add_mul_mul [Invertible (A + U*C*V)] :
    ⅟(A + U*C*V) = ⅟A - ⅟A*U*⅟(⅟C + V*⅟A*U)*V*⅟A := by
  /-
    m : Type u_1
    n : Type u_2
    α : Type u_3
    inst✝⁸ : Fintype n
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Ring α
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    inst✝³ : Invertible A
    inst✝² : Invertible C
    inst✝¹ : Invertible (HAdd.hAdd (Invertible.invOf C) (HMul.hMul (HMul.hMul V (I …
    inst✝ : Invertible (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))
    ⊢ Eq (Invertible.invOf (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub …
  -/
  letI := invertibleAddMulMul A U C V
  /-
    m : Type u_1
    n : Type u_2
    α : Type u_3
    inst✝⁸ : Fintype n
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Ring α
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    inst✝³ : Invertible A
    inst✝² : Invertible C
    inst✝¹ : Invertible (HAdd.hAdd (Invertible.invOf C) (HMul.hMul (HMul.hMul V (I …
    inst✝ : Invertible (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))
    this : Invertible (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V)) := A.invertibleA …
    ⊢ Eq (Invertible.invOf (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub …
  -/
  convert (rfl : ⅟(A + U*C*V) = _)
  /-
    🎉 no goals
  -/


