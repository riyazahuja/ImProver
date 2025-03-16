/-- LDU decomposition of a block matrix with an invertible top-left corner, using the
Schur complement. -/
theorem fromBlocks_eq_of_invertible₁₁ (A : Matrix m m α) (B : Matrix m n α) (C : Matrix l m α)
    (D : Matrix l n α) [Invertible A] :
    fromBlocks A B C D =
      fromBlocks 1 0 (C * ⅟ A) 1 * fromBlocks A 0 0 (D - C * ⅟ A * B) *
        fromBlocks 1 (⅟ A * B) 0 1 := by
  simp only [fromBlocks_multiply, Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add,
    Matrix.one_mul, Matrix.mul_one, invOf_mul_self, Matrix.mul_invOf_cancel_left,
    Matrix.invOf_mul_cancel_right, Matrix.mul_assoc, add_sub_cancel]


/-- LDU decomposition of a block matrix with an invertible bottom-right corner, using the
Schur complement. -/
theorem fromBlocks_eq_of_invertible₂₂ (A : Matrix l m α) (B : Matrix l n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] :
    fromBlocks A B C D =
      fromBlocks 1 (B * ⅟ D) 0 1 * fromBlocks (A - B * ⅟ D * C) 0 0 D *
        fromBlocks 1 0 (⅟ D * C) 1 :=
  (Matrix.reindex (Equiv.sumComm _ _) (Equiv.sumComm _ _)).injective <| by
    simpa [reindex_apply, Equiv.sumComm_symm, ← submatrix_mul_equiv _ _ _ (Equiv.sumComm n m), ←
      submatrix_mul_equiv _ _ _ (Equiv.sumComm n l), Equiv.sumComm_apply,
      fromBlocks_submatrix_sum_swap_sum_swap] using fromBlocks_eq_of_invertible₁₁ D C B A


/-- An upper-block-triangular matrix is invertible if its diagonal is. -/
def fromBlocksZero₂₁Invertible (A : Matrix m m α) (B : Matrix m n α) (D : Matrix n n α)
    [Invertible A] [Invertible D] : Invertible (fromBlocks A B 0 D) :=
  invertibleOfLeftInverse _ (fromBlocks (⅟ A) (-(⅟ A * B * ⅟ D)) 0 (⅟ D)) <| by
    simp_rw [fromBlocks_multiply, Matrix.mul_zero, Matrix.zero_mul, zero_add, add_zero,
      Matrix.neg_mul, invOf_mul_self, Matrix.invOf_mul_cancel_right, add_neg_cancel,
      fromBlocks_one]


/-- A lower-block-triangular matrix is invertible if its diagonal is. -/
def fromBlocksZero₁₂Invertible (A : Matrix m m α) (C : Matrix n m α) (D : Matrix n n α)
    [Invertible A] [Invertible D] : Invertible (fromBlocks A 0 C D) :=
  invertibleOfLeftInverse _
      (fromBlocks (⅟ A) 0 (-(⅟ D * C * ⅟ A))
        (⅟ D)) <| by -- a symmetry argument is more work than just copying the proof
    simp_rw [fromBlocks_multiply, Matrix.mul_zero, Matrix.zero_mul, zero_add, add_zero,
      Matrix.neg_mul, invOf_mul_self, Matrix.invOf_mul_cancel_right, neg_add_cancel,
      fromBlocks_one]


theorem invOf_fromBlocks_zero₂₁_eq (A : Matrix m m α) (B : Matrix m n α) (D : Matrix n n α)
    [Invertible A] [Invertible D] [Invertible (fromBlocks A B 0 D)] :
    ⅟ (fromBlocks A B 0 D) = fromBlocks (⅟ A) (-(⅟ A * B * ⅟ D)) 0 (⅟ D) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Invert …
  -/
  letI := fromBlocksZero₂₁Invertible A B D
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
    this : Invertible (Matrix.fromBlocks A B 0 D) := A.fromBlocksZero₂₁Invertible  …
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Invert …
  -/
  convert (rfl : ⅟ (fromBlocks A B 0 D) = _)
  /-
    🎉 no goals
  -/


theorem invOf_fromBlocks_zero₁₂_eq (A : Matrix m m α) (C : Matrix n m α) (D : Matrix n n α)
    [Invertible A] [Invertible D] [Invertible (fromBlocks A 0 C D)] :
    ⅟ (fromBlocks A 0 C D) = fromBlocks (⅟ A) 0 (-(⅟ D * C * ⅟ A)) (⅟ D) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Invert …
  -/
  letI := fromBlocksZero₁₂Invertible A C D
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
    this : Invertible (Matrix.fromBlocks A 0 C D) := A.fromBlocksZero₁₂Invertible  …
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Invert …
  -/
  convert (rfl : ⅟ (fromBlocks A 0 C D) = _)
  /-
    🎉 no goals
  -/


/-- Both diagonal entries of an invertible upper-block-triangular matrix are invertible (by reading
off the diagonal entries of the inverse). -/
def invertibleOfFromBlocksZero₂₁Invertible (A : Matrix m m α) (B : Matrix m n α) (D : Matrix n n α)
    [Invertible (fromBlocks A B 0 D)] : Invertible A × Invertible D where
  fst :=
    invertibleOfLeftInverse _ (⅟ (fromBlocks A B 0 D)).toBlocks₁₁ <| by
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₁₁ A) 1
      -/
      have := invOf_mul_self (fromBlocks A B 0 D)
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        this : Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A B 0 D)) (Matrix.fr …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₁₁ A) 1
      -/
      rw [← fromBlocks_toBlocks (⅟ (fromBlocks A B 0 D)), fromBlocks_multiply] at this
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        this : Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul (Invertible.invOf (Matrix.f …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₁₁ A) 1
      -/
      replace := congr_arg Matrix.toBlocks₁₁ this
      simpa only [Matrix.toBlocks_fromBlocks₁₁, Matrix.mul_zero, add_zero, ← fromBlocks_one] using
        this
  snd :=
    invertibleOfRightInverse _ (⅟ (fromBlocks A B 0 D)).toBlocks₂₂ <| by
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        ⊢ Eq (HMul.hMul D (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₂₂) 1
      -/
      have := mul_invOf_self (fromBlocks A B 0 D)
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        this : Eq (HMul.hMul (Matrix.fromBlocks A B 0 D) (Invertible.invOf (Matrix.fro …
        ⊢ Eq (HMul.hMul D (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₂₂) 1
      -/
      rw [← fromBlocks_toBlocks (⅟ (fromBlocks A B 0 D)), fromBlocks_multiply] at this
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        B : Matrix m n α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A B 0 D)
        this : Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul A (Invertible.invOf (Matrix …
        ⊢ Eq (HMul.hMul D (Invertible.invOf (Matrix.fromBlocks A B 0 D)).toBlocks₂₂) 1
      -/
      replace := congr_arg Matrix.toBlocks₂₂ this
      simpa only [Matrix.toBlocks_fromBlocks₂₂, Matrix.zero_mul, zero_add, ← fromBlocks_one] using
        this


/-- Both diagonal entries of an invertible lower-block-triangular matrix are invertible (by reading
off the diagonal entries of the inverse). -/
def invertibleOfFromBlocksZero₁₂Invertible (A : Matrix m m α) (C : Matrix n m α) (D : Matrix n n α)
    [Invertible (fromBlocks A 0 C D)] : Invertible A × Invertible D where
  fst :=
    invertibleOfRightInverse _ (⅟ (fromBlocks A 0 C D)).toBlocks₁₁ <| by
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        ⊢ Eq (HMul.hMul A (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₁₁) 1
      -/
      have := mul_invOf_self (fromBlocks A 0 C D)
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        this : Eq (HMul.hMul (Matrix.fromBlocks A 0 C D) (Invertible.invOf (Matrix.fro …
        ⊢ Eq (HMul.hMul A (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₁₁) 1
      -/
      rw [← fromBlocks_toBlocks (⅟ (fromBlocks A 0 C D)), fromBlocks_multiply] at this
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        this : Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul A (Invertible.invOf (Matrix …
        ⊢ Eq (HMul.hMul A (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₁₁) 1
      -/
      replace := congr_arg Matrix.toBlocks₁₁ this
      simpa only [Matrix.toBlocks_fromBlocks₁₁, Matrix.zero_mul, add_zero, ← fromBlocks_one] using
        this
  snd :=
    invertibleOfLeftInverse _ (⅟ (fromBlocks A 0 C D)).toBlocks₂₂ <| by
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₂₂ D) 1
      -/
      have := invOf_mul_self (fromBlocks A 0 C D)
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        this : Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A 0 C D)) (Matrix.fr …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₂₂ D) 1
      -/
      rw [← fromBlocks_toBlocks (⅟ (fromBlocks A 0 C D)), fromBlocks_multiply] at this
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁷ : Fintype l
        inst✝⁶ : Fintype m
        inst✝⁵ : Fintype n
        inst✝⁴ : DecidableEq l
        inst✝³ : DecidableEq m
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A : Matrix m m α
        C : Matrix n m α
        D : Matrix n n α
        inst✝ : Invertible (Matrix.fromBlocks A 0 C D)
        this : Eq (Matrix.fromBlocks (HAdd.hAdd (HMul.hMul (Invertible.invOf (Matrix.f …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.fromBlocks A 0 C D)).toBlocks₂₂ D) 1
      -/
      replace := congr_arg Matrix.toBlocks₂₂ this
      simpa only [Matrix.toBlocks_fromBlocks₂₂, Matrix.mul_zero, zero_add, ← fromBlocks_one] using
        this


/-- `invertibleOfFromBlocksZero₂₁Invertible` and `Matrix.fromBlocksZero₂₁Invertible` form
an equivalence. -/
def fromBlocksZero₂₁InvertibleEquiv (A : Matrix m m α) (B : Matrix m n α) (D : Matrix n n α) :
    Invertible (fromBlocks A B 0 D) ≃ Invertible A × Invertible D where
  toFun _ := invertibleOfFromBlocksZero₂₁Invertible A B D
  invFun i := by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      ⊢ Invertible (Matrix.fromBlocks A B 0 D)
    -/
    letI := i.1
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      this : Invertible A := i.1
      ⊢ Invertible (Matrix.fromBlocks A B 0 D)
    -/
    letI := i.2
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      this✝ : Invertible A := i.1
      this : Invertible D := i.2
      ⊢ Invertible (Matrix.fromBlocks A B 0 D)
    -/
    exact fromBlocksZero₂₁Invertible A B D
    /-
      🎉 no goals
    -/
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- `invertibleOfFromBlocksZero₁₂Invertible` and `Matrix.fromBlocksZero₁₂Invertible` form
an equivalence. -/
def fromBlocksZero₁₂InvertibleEquiv (A : Matrix m m α) (C : Matrix n m α) (D : Matrix n n α) :
    Invertible (fromBlocks A 0 C D) ≃ Invertible A × Invertible D where
  toFun _ := invertibleOfFromBlocksZero₁₂Invertible A C D
  invFun i := by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      ⊢ Invertible (Matrix.fromBlocks A 0 C D)
    -/
    letI := i.1
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      this : Invertible A := i.1
      ⊢ Invertible (Matrix.fromBlocks A 0 C D)
    -/
    letI := i.2
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁶ : Fintype l
      inst✝⁵ : Fintype m
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq l
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      i : Prod (Invertible A) (Invertible D)
      this✝ : Invertible A := i.1
      this : Invertible D := i.2
      ⊢ Invertible (Matrix.fromBlocks A 0 C D)
    -/
    exact fromBlocksZero₁₂Invertible A C D
    /-
      🎉 no goals
    -/
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- An upper block-triangular matrix is invertible iff both elements of its diagonal are.

This is a propositional form of `Matrix.fromBlocksZero₂₁InvertibleEquiv`. -/
@[simp]
theorem isUnit_fromBlocks_zero₂₁ {A : Matrix m m α} {B : Matrix m n α} {D : Matrix n n α} :
    IsUnit (fromBlocks A B 0 D) ↔ IsUnit A ∧ IsUnit D := by
  simp only [← nonempty_invertible_iff_isUnit, ← nonempty_prod,
    (fromBlocksZero₂₁InvertibleEquiv _ _ _).nonempty_congr]


/-- A lower block-triangular matrix is invertible iff both elements of its diagonal are.

This is a propositional form of `Matrix.fromBlocksZero₁₂InvertibleEquiv` forms an `iff`. -/
@[simp]
theorem isUnit_fromBlocks_zero₁₂ {A : Matrix m m α} {C : Matrix n m α} {D : Matrix n n α} :
    IsUnit (fromBlocks A 0 C D) ↔ IsUnit A ∧ IsUnit D := by
  simp only [← nonempty_invertible_iff_isUnit, ← nonempty_prod,
    (fromBlocksZero₁₂InvertibleEquiv _ _ _).nonempty_congr]


/-- An expression for the inverse of an upper block-triangular matrix, when either both elements of
diagonal are invertible, or both are not. -/
theorem inv_fromBlocks_zero₂₁_of_isUnit_iff (A : Matrix m m α) (B : Matrix m n α) (D : Matrix n n α)
    (hAD : IsUnit A ↔ IsUnit D) :
    (fromBlocks A B 0 D)⁻¹ = fromBlocks A⁻¹ (-(A⁻¹ * B * D⁻¹)) 0 D⁻¹ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    D : Matrix n n α
    hAD : Iff (IsUnit A) (IsUnit D)
    ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
  -/
  by_cases hA : IsUnit A
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
  · have hD := hAD.mp hA
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
    cases hA.nonempty_invertible
    /-
      case pos.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝ : Invertible A
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
    cases hD.nonempty_invertible
    /-
      case pos.intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝¹ : Invertible A
      val✝ : Invertible D
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
    letI := fromBlocksZero₂₁Invertible A B D
    /-
      case pos.intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝¹ : Invertible A
      val✝ : Invertible D
      this : Invertible (Matrix.fromBlocks A B 0 D) := A.fromBlocksZero₂₁Invertible  …
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
    simp_rw [← invOf_eq_nonsing_inv, invOf_fromBlocks_zero₂₁_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : Not (IsUnit A)
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A B 0 D)) (Matrix.fromBlocks (Inv.inv A) (Neg …
    -/
  · have hD := hAD.not.mp hA
    have : ¬IsUnit (fromBlocks A B 0 D) :=
      isUnit_fromBlocks_zero₂₁.not.mpr (not_and'.mpr fun _ => hA)
    simp_rw [nonsing_inv_eq_ring_inverse, Ring.inverse_non_unit _ hA, Ring.inverse_non_unit _ hD,
      Ring.inverse_non_unit _ this, Matrix.zero_mul, neg_zero, fromBlocks_zero]


/-- An expression for the inverse of a lower block-triangular matrix, when either both elements of
diagonal are invertible, or both are not. -/
theorem inv_fromBlocks_zero₁₂_of_isUnit_iff (A : Matrix m m α) (C : Matrix n m α) (D : Matrix n n α)
    (hAD : IsUnit A ↔ IsUnit D) :
    (fromBlocks A 0 C D)⁻¹ = fromBlocks A⁻¹ 0 (-(D⁻¹ * C * A⁻¹)) D⁻¹ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m m α
    C : Matrix n m α
    D : Matrix n n α
    hAD : Iff (IsUnit A) (IsUnit D)
    ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
  -/
  by_cases hA : IsUnit A
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
  · have hD := hAD.mp hA
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
    cases hA.nonempty_invertible
    /-
      case pos.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝ : Invertible A
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
    cases hD.nonempty_invertible
    /-
      case pos.intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝¹ : Invertible A
      val✝ : Invertible D
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
    letI := fromBlocksZero₁₂Invertible A C D
    /-
      case pos.intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : IsUnit A
      hD : IsUnit D
      val✝¹ : Invertible A
      val✝ : Invertible D
      this : Invertible (Matrix.fromBlocks A 0 C D) := A.fromBlocksZero₁₂Invertible  …
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
    simp_rw [← invOf_eq_nonsing_inv, invOf_fromBlocks_zero₁₂_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix m m α
      C : Matrix n m α
      D : Matrix n n α
      hAD : Iff (IsUnit A) (IsUnit D)
      hA : Not (IsUnit A)
      ⊢ Eq (Inv.inv (Matrix.fromBlocks A 0 C D)) (Matrix.fromBlocks (Inv.inv A) 0 (N …
    -/
  · have hD := hAD.not.mp hA
    have : ¬IsUnit (fromBlocks A 0 C D) :=
      isUnit_fromBlocks_zero₁₂.not.mpr (not_and'.mpr fun _ => hA)
    simp_rw [nonsing_inv_eq_ring_inverse, Ring.inverse_non_unit _ hA, Ring.inverse_non_unit _ hD,
      Ring.inverse_non_unit _ this, Matrix.zero_mul, neg_zero, fromBlocks_zero]


/-- A block matrix is invertible if the bottom right corner and the corresponding schur complement
is. -/
def fromBlocks₂₂Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] [Invertible (A - B * ⅟ D * C)] :
    Invertible (fromBlocks A B C D) := by
  -- factor `fromBlocks` via `fromBlocks_eq_of_invertible₂₂`, and state the inverse we expect
  convert Invertible.copy' _ _ (fromBlocks (⅟ (A - B * ⅟ D * C)) (-(⅟ (A - B * ⅟ D * C) * B * ⅟ D))
    (-(⅟ D * C * ⅟ (A - B * ⅟ D * C))) (⅟ D + ⅟ D * C * ⅟ (A - B * ⅟ D * C) * B * ⅟ D))
      (fromBlocks_eq_of_invertible₂₂ _ _ _ _) _
  · -- the product is invertible because all the factors are
    /-
      case convert_1
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁸ : Fintype l
      inst✝⁷ : Fintype m
      inst✝⁶ : Fintype n
      inst✝⁵ : DecidableEq l
      inst✝⁴ : DecidableEq m
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      C : Matrix n m α
      D : Matrix n n α
      inst✝¹ : Invertible D
      inst✝ : Invertible (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertible.invOf D))  …
      ⊢ Invertible (HMul.hMul (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Invertib …
    -/
    letI : Invertible (1 : Matrix n n α) := invertibleOne
    /-
      case convert_1
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁸ : Fintype l
      inst✝⁷ : Fintype m
      inst✝⁶ : Fintype n
      inst✝⁵ : DecidableEq l
      inst✝⁴ : DecidableEq m
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      C : Matrix n m α
      D : Matrix n n α
      inst✝¹ : Invertible D
      inst✝ : Invertible (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertible.invOf D))  …
      this : Invertible 1 := invertibleOne
      ⊢ Invertible (HMul.hMul (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Invertib …
    -/
    letI : Invertible (1 : Matrix m m α) := invertibleOne
    /-
      case convert_1
      l : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_4
      inst✝⁸ : Fintype l
      inst✝⁷ : Fintype m
      inst✝⁶ : Fintype n
      inst✝⁵ : DecidableEq l
      inst✝⁴ : DecidableEq m
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      A : Matrix m m α
      B : Matrix m n α
      C : Matrix n m α
      D : Matrix n n α
      inst✝¹ : Invertible D
      inst✝ : Invertible (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertible.invOf D))  …
      this✝ : Invertible 1 := invertibleOne
      this : Invertible 1 := invertibleOne
      ⊢ Invertible (HMul.hMul (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Invertib …
    -/
    refine Invertible.mul ?_ (fromBlocksZero₁₂Invertible _ _ _)
    exact
      Invertible.mul (fromBlocksZero₂₁Invertible _ _ _)
        (fromBlocksZero₂₁Invertible _ _ _)
  · -- unfold the `Invertible` instances to get the raw factors
    show
      _ =
        fromBlocks 1 0 (-(1 * (⅟ D * C) * 1)) 1 *
          (fromBlocks (⅟ (A - B * ⅟ D * C)) (-(⅟ (A - B * ⅟ D * C) * 0 * ⅟ D)) 0 (⅟ D) *
            fromBlocks 1 (-(1 * (B * ⅟ D) * 1)) 0 1)
    -- combine into a single block matrix
    simp only [fromBlocks_multiply, invOf_one, Matrix.one_mul, Matrix.mul_one, Matrix.zero_mul,
      Matrix.mul_zero, add_zero, zero_add, neg_zero, Matrix.mul_neg, Matrix.neg_mul, neg_neg, ←
      Matrix.mul_assoc, add_comm (⅟D)]


/-- A block matrix is invertible if the top left corner and the corresponding schur complement
is. -/
def fromBlocks₁₁Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible A] [Invertible (D - C * ⅟ A * B)] :
    Invertible (fromBlocks A B C D) := by
  -- we argue by symmetry
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible (HSub.hSub D (HMul.hMul (HMul.hMul C (Invertible.invOf A))  …
    ⊢ Invertible (Matrix.fromBlocks A B C D)
  -/
  letI := fromBlocks₂₂Invertible D C B A
  letI iDCBA :=
    submatrixEquivInvertible (fromBlocks D C B A) (Equiv.sumComm _ _) (Equiv.sumComm _ _)
  exact
    iDCBA.copy' _
      (fromBlocks (⅟ A + ⅟ A * B * ⅟ (D - C * ⅟ A * B) * C * ⅟ A) (-(⅟ A * B * ⅟ (D - C * ⅟ A * B)))
        (-(⅟ (D - C * ⅟ A * B) * C * ⅟ A)) (⅟ (D - C * ⅟ A * B)))
      (fromBlocks_submatrix_sum_swap_sum_swap _ _ _ _).symm
      (fromBlocks_submatrix_sum_swap_sum_swap _ _ _ _).symm


theorem invOf_fromBlocks₂₂_eq (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] [Invertible (A - B * ⅟ D * C)]
    [Invertible (fromBlocks A B C D)] :
    ⅟ (fromBlocks A B C D) =
      fromBlocks (⅟ (A - B * ⅟ D * C)) (-(⅟ (A - B * ⅟ D * C) * B * ⅟ D))
        (-(⅟ D * C * ⅟ (A - B * ⅟ D * C))) (⅟ D + ⅟ D * C * ⅟ (A - B * ⅟ D * C) * B * ⅟ D) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible D
    inst✝¹ : Invertible (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertible.invOf D)) …
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (Invert …
  -/
  letI := fromBlocks₂₂Invertible A B C D
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible D
    inst✝¹ : Invertible (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertible.invOf D)) …
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this : Invertible (Matrix.fromBlocks A B C D) := A.fromBlocks₂₂Invertible B C D
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (Invert …
  -/
  convert (rfl : ⅟ (fromBlocks A B C D) = _)
  /-
    🎉 no goals
  -/


theorem invOf_fromBlocks₁₁_eq (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible A] [Invertible (D - C * ⅟ A * B)]
    [Invertible (fromBlocks A B C D)] :
    ⅟ (fromBlocks A B C D) =
      fromBlocks (⅟ A + ⅟ A * B * ⅟ (D - C * ⅟ A * B) * C * ⅟ A) (-(⅟ A * B * ⅟ (D - C * ⅟ A * B)))
        (-(⅟ (D - C * ⅟ A * B) * C * ⅟ A)) (⅟ (D - C * ⅟ A * B)) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible (HSub.hSub D (HMul.hMul (HMul.hMul C (Invertible.invOf A)) …
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (HAdd.h …
  -/
  letI := fromBlocks₁₁Invertible A B C D
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : DecidableEq n
    inst✝³ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝² : Invertible A
    inst✝¹ : Invertible (HSub.hSub D (HMul.hMul (HMul.hMul C (Invertible.invOf A)) …
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this : Invertible (Matrix.fromBlocks A B C D) := A.fromBlocks₁₁Invertible B C D
    ⊢ Eq (Invertible.invOf (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (HAdd.h …
  -/
  convert (rfl : ⅟ (fromBlocks A B C D) = _)
  /-
    🎉 no goals
  -/


/-- If a block matrix is invertible and so is its bottom left element, then so is the corresponding
Schur complement. -/
def invertibleOfFromBlocks₂₂Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] [Invertible (fromBlocks A B C D)] :
    Invertible (A - B * ⅟ D * C) := by
  suffices Invertible (fromBlocks (A - B * ⅟ D * C) 0 0 D) by
    exact (invertibleOfFromBlocksZero₁₂Invertible (A - B * ⅟ D * C) 0 D).1
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    ⊢ Invertible (Matrix.fromBlocks (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertib …
  -/
  letI : Invertible (1 : Matrix n n α) := invertibleOne
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this : Invertible 1 := invertibleOne
    ⊢ Invertible (Matrix.fromBlocks (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertib …
  -/
  letI : Invertible (1 : Matrix m m α) := invertibleOne
  letI iDC : Invertible (fromBlocks 1 0 (⅟ D * C) 1 : Matrix (m ⊕ n) (m ⊕ n) α) :=
    fromBlocksZero₁₂Invertible _ _ _
  letI iBD : Invertible (fromBlocks 1 (B * ⅟ D) 0 1 : Matrix (m ⊕ n) (m ⊕ n) α) :=
    fromBlocksZero₂₁Invertible _ _ _
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this✝ : Invertible 1 := invertibleOne
    this : Invertible 1 := invertibleOne
    iDC : Invertible (Matrix.fromBlocks 1 0 (HMul.hMul (Invertible.invOf D) C) 1)  …
    iBD : Invertible (Matrix.fromBlocks 1 (HMul.hMul B (Invertible.invOf D)) 0 1)  …
    ⊢ Invertible (Matrix.fromBlocks (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertib …
  -/
  letI iBDC := Invertible.copy ‹_› _ (fromBlocks_eq_of_invertible₂₂ A B C D).symm
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this✝ : Invertible 1 := invertibleOne
    this : Invertible 1 := invertibleOne
    iDC : Invertible (Matrix.fromBlocks 1 0 (HMul.hMul (Invertible.invOf D) C) 1)  …
    iBD : Invertible (Matrix.fromBlocks 1 (HMul.hMul B (Invertible.invOf D)) 0 1)  …
    iBDC : Invertible (HMul.hMul (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Inv …
    ⊢ Invertible (Matrix.fromBlocks (HSub.hSub A (HMul.hMul (HMul.hMul B (Invertib …
  -/
  refine (iBD.mulLeft _).symm ?_
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible D
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    this✝ : Invertible 1 := invertibleOne
    this : Invertible 1 := invertibleOne
    iDC : Invertible (Matrix.fromBlocks 1 0 (HMul.hMul (Invertible.invOf D) C) 1)  …
    iBD : Invertible (Matrix.fromBlocks 1 (HMul.hMul B (Invertible.invOf D)) 0 1)  …
    iBDC : Invertible (HMul.hMul (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Inv …
    ⊢ Invertible (HMul.hMul (Matrix.fromBlocks 1 (HMul.hMul B (Invertible.invOf D) …
  -/
  exact (iDC.mulRight _).symm iBDC
  /-
    🎉 no goals
  -/


/-- If a block matrix is invertible and so is its bottom left element, then so is the corresponding
Schur complement. -/
def invertibleOfFromBlocks₁₁Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible A] [Invertible (fromBlocks A B C D)] :
    Invertible (D - C * ⅟ A * B) := by
  -- another symmetry argument
  letI iABCD' :=
    submatrixEquivInvertible (fromBlocks A B C D) (Equiv.sumComm _ _) (Equiv.sumComm _ _)
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    iABCD' : Invertible ((Matrix.fromBlocks A B C D).submatrix ⇑(Equiv.sumComm n m …
    ⊢ Invertible (HSub.hSub D (HMul.hMul (HMul.hMul C (Invertible.invOf A)) B))
  -/
  letI iDCBA := iABCD'.copy _ (fromBlocks_submatrix_sum_swap_sum_swap _ _ _ _).symm
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁸ : Fintype l
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq l
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible (Matrix.fromBlocks A B C D)
    iABCD' : Invertible ((Matrix.fromBlocks A B C D).submatrix ⇑(Equiv.sumComm n m …
    iDCBA : Invertible (Matrix.fromBlocks D C B A) := iABCD'.copy (Matrix.fromBloc …
    ⊢ Invertible (HSub.hSub D (HMul.hMul (HMul.hMul C (Invertible.invOf A)) B))
  -/
  exact invertibleOfFromBlocks₂₂Invertible D C B A
  /-
    🎉 no goals
  -/


/-- `Matrix.invertibleOfFromBlocks₂₂Invertible` and `Matrix.fromBlocks₂₂Invertible` as an
equivalence. -/
def invertibleEquivFromBlocks₂₂Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] :
    Invertible (fromBlocks A B C D) ≃ Invertible (A - B * ⅟ D * C) where
  toFun _iABCD := invertibleOfFromBlocks₂₂Invertible _ _ _ _
  invFun _i_schur := fromBlocks₂₂Invertible _ _ _ _
  left_inv _iABCD := Subsingleton.elim _ _
  right_inv _i_schur := Subsingleton.elim _ _


/-- `Matrix.invertibleOfFromBlocks₁₁Invertible` and `Matrix.fromBlocks₁₁Invertible` as an
equivalence. -/
def invertibleEquivFromBlocks₁₁Invertible (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible A] :
    Invertible (fromBlocks A B C D) ≃ Invertible (D - C * ⅟ A * B) where
  toFun _iABCD := invertibleOfFromBlocks₁₁Invertible _ _ _ _
  invFun _i_schur := fromBlocks₁₁Invertible _ _ _ _
  left_inv _iABCD := Subsingleton.elim _ _
  right_inv _i_schur := Subsingleton.elim _ _


/-- If the bottom-left element of a block matrix is invertible, then the whole matrix is invertible
iff the corresponding schur complement is. -/
theorem isUnit_fromBlocks_iff_of_invertible₂₂ {A : Matrix m m α} {B : Matrix m n α}
    {C : Matrix n m α} {D : Matrix n n α} [Invertible D] :
    IsUnit (fromBlocks A B C D) ↔ IsUnit (A - B * ⅟ D * C) := by
  simp only [← nonempty_invertible_iff_isUnit,
    (invertibleEquivFromBlocks₂₂Invertible A B C D).nonempty_congr]


/-- If the top-right element of a block matrix is invertible, then the whole matrix is invertible
iff the corresponding schur complement is. -/
theorem isUnit_fromBlocks_iff_of_invertible₁₁ {A : Matrix m m α} {B : Matrix m n α}
    {C : Matrix n m α} {D : Matrix n n α} [Invertible A] :
    IsUnit (fromBlocks A B C D) ↔ IsUnit (D - C * ⅟ A * B) := by
  simp only [← nonempty_invertible_iff_isUnit,
    (invertibleEquivFromBlocks₁₁Invertible A B C D).nonempty_congr]


/-- Determinant of a 2×2 block matrix, expanded around an invertible top left element in terms of
the Schur complement. -/
theorem det_fromBlocks₁₁ (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible A] :
    (Matrix.fromBlocks A B C D).det = det A * det (D - C * ⅟ A * B) := by
  rw [fromBlocks_eq_of_invertible₁₁ (A := A), det_mul, det_mul, det_fromBlocks_zero₂₁,
    det_fromBlocks_zero₂₁, det_fromBlocks_zero₁₂, det_one, det_one, one_mul, one_mul, mul_one]


@[simp]
theorem det_fromBlocks_one₁₁ (B : Matrix m n α) (C : Matrix n m α) (D : Matrix n n α) :
    (Matrix.fromBlocks 1 B C D).det = det (D - C * B) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    ⊢ Eq (Matrix.fromBlocks 1 B C D).det (HSub.hSub D (HMul.hMul C B)).det
  -/
  haveI : Invertible (1 : Matrix m m α) := invertibleOne
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    this : Invertible 1
    ⊢ Eq (Matrix.fromBlocks 1 B C D).det (HSub.hSub D (HMul.hMul C B)).det
  -/
  rw [det_fromBlocks₁₁, invOf_one, Matrix.mul_one, det_one, one_mul]
  /-
    🎉 no goals
  -/


/-- Determinant of a 2×2 block matrix, expanded around an invertible bottom right element in terms
of the Schur complement. -/
theorem det_fromBlocks₂₂ (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α)
    (D : Matrix n n α) [Invertible D] :
    (Matrix.fromBlocks A B C D).det = det D * det (A - B * ⅟ D * C) := by
  have : fromBlocks A B C D =
      (fromBlocks D C B A).submatrix (Equiv.sumComm _ _) (Equiv.sumComm _ _) := by
    ext (i j)
    cases i <;> cases j <;> rfl
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    D : Matrix n n α
    inst✝ : Invertible D
    this : Eq (Matrix.fromBlocks A B C D) ((Matrix.fromBlocks D C B A).submatrix ⇑ …
    ⊢ Eq (Matrix.fromBlocks A B C D).det (HMul.hMul D.det (HSub.hSub A (HMul.hMul  …
  -/
  rw [this, det_submatrix_equiv_self, det_fromBlocks₁₁]
  /-
    🎉 no goals
  -/


@[simp]
theorem det_fromBlocks_one₂₂ (A : Matrix m m α) (B : Matrix m n α) (C : Matrix n m α) :
    (Matrix.fromBlocks A B C 1).det = det (A - B * C) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    ⊢ Eq (Matrix.fromBlocks A B C 1).det (HSub.hSub A (HMul.hMul B C)).det
  -/
  haveI : Invertible (1 : Matrix n n α) := invertibleOne
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m m α
    B : Matrix m n α
    C : Matrix n m α
    this : Invertible 1
    ⊢ Eq (Matrix.fromBlocks A B C 1).det (HSub.hSub A (HMul.hMul B C)).det
  -/
  rw [det_fromBlocks₂₂, invOf_one, Matrix.mul_one, det_one, one_mul]
  /-
    🎉 no goals
  -/


/-- The **Weinstein–Aronszajn identity**. Note the `1` on the LHS is of shape m×m, while the `1` on
the RHS is of shape n×n. -/
theorem det_one_add_mul_comm (A : Matrix m n α) (B : Matrix n m α) :
    det (1 + A * B) = det (1 + B * A) :=
  calc
    det (1 + A * B) = det (fromBlocks 1 (-A) B 1) := by
      /-
        m : Type u_2
        n : Type u_3
        α : Type u_4
        inst✝⁴ : Fintype m
        inst✝³ : Fintype n
        inst✝² : DecidableEq m
        inst✝¹ : DecidableEq n
        inst✝ : CommRing α
        A : Matrix m n α
        B : Matrix n m α
        ⊢ Eq (HAdd.hAdd 1 (HMul.hMul A B)).det (Matrix.fromBlocks 1 (Neg.neg A) B 1).det
      -/
      rw [det_fromBlocks_one₂₂, Matrix.neg_mul, sub_neg_eq_add]
      /-
        🎉 no goals
      -/
                              /-
                                m : Type u_2
                                n : Type u_3
                                α : Type u_4
                                inst✝⁴ : Fintype m
                                inst✝³ : Fintype n
                                inst✝² : DecidableEq m
                                inst✝¹ : DecidableEq n
                                inst✝ : CommRing α
                                A : Matrix m n α
                                B : Matrix n m α
                                ⊢ Eq (Matrix.fromBlocks 1 (Neg.neg A) B 1).det (HAdd.hAdd 1 (HMul.hMul B A)).det
                              -/
    _ = det (1 + B * A) := by rw [det_fromBlocks_one₁₁, Matrix.mul_neg, sub_neg_eq_add]
                              /-
                                🎉 no goals
                              -/


/-- Alternate statement of the **Weinstein–Aronszajn identity** -/
theorem det_mul_add_one_comm (A : Matrix m n α) (B : Matrix n m α) :
                                            /-
                                              m : Type u_2
                                              n : Type u_3
                                              α : Type u_4
                                              inst✝⁴ : Fintype m
                                              inst✝³ : Fintype n
                                              inst✝² : DecidableEq m
                                              inst✝¹ : DecidableEq n
                                              inst✝ : CommRing α
                                              A : Matrix m n α
                                              B : Matrix n m α
                                              ⊢ Eq (HAdd.hAdd (HMul.hMul A B) 1).det (HAdd.hAdd (HMul.hMul B A) 1).det
                                            -/
    det (A * B + 1) = det (B * A + 1) := by rw [add_comm, det_one_add_mul_comm, add_comm]
                                            /-
                                              🎉 no goals
                                            -/


theorem det_one_sub_mul_comm (A : Matrix m n α) (B : Matrix n m α) :
    det (1 - A * B) = det (1 - B * A) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m n α
    B : Matrix n m α
    ⊢ Eq (HSub.hSub 1 (HMul.hMul A B)).det (HSub.hSub 1 (HMul.hMul B A)).det
  -/
  rw [sub_eq_add_neg, ← Matrix.neg_mul, det_one_add_mul_comm, Matrix.mul_neg, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- A special case of the **Matrix determinant lemma** for when `A = I`. -/
theorem det_one_add_col_mul_row {ι : Type*} [Unique ι] (u v : m → α) :
    det (1 + col ι u * row ι v) = 1 + v ⬝ᵥ u := by
  rw [det_one_add_mul_comm, det_unique, Pi.add_apply, Pi.add_apply, Matrix.one_apply_eq,
    Matrix.row_mul_col_apply]


/-- The **Matrix determinant lemma**

TODO: show the more general version without `hA : IsUnit A.det` as
`(A + col u * row v).det = A.det + v ⬝ᵥ (adjugate A) *ᵥ u`.
-/
theorem det_add_col_mul_row {ι : Type*} [Unique ι]
    {A : Matrix m m α} (hA : IsUnit A.det) (u v : m → α) :
    (A + col ι u * row ι v).det = A.det * (1 + row ι v * A⁻¹ * col ι u).det := by
  /-
    m : Type u_2
    α : Type u_4
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : CommRing α
    ι : Type u_5
    inst✝ : Unique ι
    A : Matrix m m α
    hA : IsUnit A.det
    u v : m → α
    ⊢ Eq (HAdd.hAdd A (HMul.hMul (Matrix.col ι u) (Matrix.row ι v))).det (HMul.hMu …
  -/
  nth_rewrite 1 [← Matrix.mul_one A]
  rwa [← Matrix.mul_nonsing_inv_cancel_left A (col ι u * row ι v),
    ← Matrix.mul_add, det_mul, ← Matrix.mul_assoc, det_one_add_mul_comm,
    ← Matrix.mul_assoc]


/-- A generalization of the **Matrix determinant lemma** -/
theorem det_add_mul {A : Matrix m m α} (U : Matrix m n α)
    (V : Matrix n m α) (hA : IsUnit A.det) :
    (A + U * V).det = A.det * (1 + V * A⁻¹ * U).det := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_4
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix m m α
    U : Matrix m n α
    V : Matrix n m α
    hA : IsUnit A.det
    ⊢ Eq (HAdd.hAdd A (HMul.hMul U V)).det (HMul.hMul A.det (HAdd.hAdd 1 (HMul.hMu …
  -/
  nth_rewrite 1 [← Matrix.mul_one A]
  rwa [← Matrix.mul_nonsing_inv_cancel_left A (U * V), ← Matrix.mul_add,
    det_mul, ← Matrix.mul_assoc, det_one_add_mul_comm, ← Matrix.mul_assoc]


scoped infixl:65 " ⊕ᵥ " => Sum.elim


theorem schur_complement_eq₁₁ [Fintype m] [DecidableEq m] [Fintype n] {A : Matrix m m 𝕜}
    (B : Matrix m n 𝕜) (D : Matrix n n 𝕜) (x : m → 𝕜) (y : n → 𝕜) [Invertible A]
    (hA : A.IsHermitian) :
    (star (x ⊕ᵥ y)) ᵥ* (fromBlocks A B Bᴴ D) ⬝ᵥ (x ⊕ᵥ y) =
      (star (x + (A⁻¹ * B) *ᵥ y)) ᵥ* A ⬝ᵥ (x + (A⁻¹ * B) *ᵥ y) +
        (star y) ᵥ* (D - Bᴴ * A⁻¹ * B) ⬝ᵥ y := by
  simp [Function.star_sum_elim, fromBlocks_mulVec, vecMul_fromBlocks, add_vecMul,
    dotProduct_mulVec, vecMul_sub, Matrix.mul_assoc, vecMul_mulVec, hA.eq,
    conjTranspose_nonsing_inv, star_mulVec]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝⁵ : CommRing 𝕜
    inst✝⁴ : StarRing 𝕜
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    x : m → 𝕜
    y : n → 𝕜
    inst✝ : Invertible A
    hA : A.IsHermitian
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (dotProduct (Matrix.vecMul (Star.star x) A) x) (dot …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem schur_complement_eq₂₂ [Fintype m] [Fintype n] [DecidableEq n] (A : Matrix m m 𝕜)
    (B : Matrix m n 𝕜) {D : Matrix n n 𝕜} (x : m → 𝕜) (y : n → 𝕜) [Invertible D]
    (hD : D.IsHermitian) :
    (star (x ⊕ᵥ y)) ᵥ* (fromBlocks A B Bᴴ D) ⬝ᵥ (x ⊕ᵥ y) =
      (star ((D⁻¹ * Bᴴ) *ᵥ x + y)) ᵥ* D ⬝ᵥ ((D⁻¹ * Bᴴ) *ᵥ x + y) +
        (star x) ᵥ* (A - B * D⁻¹ * Bᴴ) ⬝ᵥ x := by
  simp [Function.star_sum_elim, fromBlocks_mulVec, vecMul_fromBlocks, add_vecMul,
    dotProduct_mulVec, vecMul_sub, Matrix.mul_assoc, vecMul_mulVec, hD.eq,
    conjTranspose_nonsing_inv, star_mulVec]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝⁵ : CommRing 𝕜
    inst✝⁴ : StarRing 𝕜
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    x : m → 𝕜
    y : n → 𝕜
    inst✝ : Invertible D
    hD : D.IsHermitian
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (dotProduct (Matrix.vecMul (Star.star x) A) x) (dot …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem IsHermitian.fromBlocks₁₁ [Fintype m] [DecidableEq m] {A : Matrix m m 𝕜} (B : Matrix m n 𝕜)
    (D : Matrix n n 𝕜) (hA : A.IsHermitian) :
    (Matrix.fromBlocks A B Bᴴ D).IsHermitian ↔ (D - Bᴴ * A⁻¹ * B).IsHermitian := by
  have hBAB : (Bᴴ * A⁻¹ * B).IsHermitian := by
    apply isHermitian_conjTranspose_mul_mul
    apply hA.inv
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝³ : CommRing 𝕜
    inst✝² : StarRing 𝕜
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hA : A.IsHermitian
    hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
    ⊢ Iff (Matrix.fromBlocks A B B.conjTranspose D).IsHermitian (HSub.hSub D (HMul …
  -/
  rw [isHermitian_fromBlocks_iff]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝³ : CommRing 𝕜
    inst✝² : StarRing 𝕜
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hA : A.IsHermitian
    hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
    ⊢ Iff (And A.IsHermitian (And (Eq B.conjTranspose B.conjTranspose) (And (Eq B. …
  -/
  constructor
    /-
      case mp
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      ⊢ And A.IsHermitian (And (Eq B.conjTranspose B.conjTranspose) (And (Eq B.conjT …
    -/
  · intro h
    /-
      case mp
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      h : And A.IsHermitian (And (Eq B.conjTranspose B.conjTranspose) (And (Eq B.con …
      ⊢ (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHermit …
    -/
    apply IsHermitian.sub h.2.2.2 hBAB
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      ⊢ (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHermit …
    -/
  · intro h
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHerm …
      ⊢ And A.IsHermitian (And (Eq B.conjTranspose B.conjTranspose) (And (Eq B.conjT …
    -/
    refine ⟨hA, rfl, conjTranspose_conjTranspose B, ?_⟩
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHerm …
      ⊢ D.IsHermitian
    -/
    rw [← sub_add_cancel D]
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝³ : CommRing 𝕜
      inst✝² : StarRing 𝕜
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.IsHermitian
      hBAB : (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B).IsHermitian
      h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHerm …
      ⊢ (HAdd.hAdd (HSub.hSub D ?mpr) ?mpr).IsHermitian
    -/
    apply IsHermitian.add h hBAB
    /-
      🎉 no goals
    -/


theorem IsHermitian.fromBlocks₂₂ [Fintype n] [DecidableEq n] (A : Matrix m m 𝕜) (B : Matrix m n 𝕜)
    {D : Matrix n n 𝕜} (hD : D.IsHermitian) :
    (Matrix.fromBlocks A B Bᴴ D).IsHermitian ↔ (A - B * D⁻¹ * Bᴴ).IsHermitian := by
  rw [← isHermitian_submatrix_equiv (Equiv.sumComm n m), Equiv.sumComm_apply,
    fromBlocks_submatrix_sum_swap_sum_swap]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝³ : CommRing 𝕜
    inst✝² : StarRing 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hD : D.IsHermitian
    ⊢ Iff (Matrix.fromBlocks D B.conjTranspose B A).IsHermitian (HSub.hSub A (HMul …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  convert IsHermitian.fromBlocks₁₁ _ _ hD <;> simp
                                              /-
                                                🎉 no goals
                                              -/


theorem PosSemidef.fromBlocks₁₁ [Fintype m] [DecidableEq m] [Fintype n] {A : Matrix m m 𝕜}
    (B : Matrix m n 𝕜) (D : Matrix n n 𝕜) (hA : A.PosDef) [Invertible A] :
    (fromBlocks A B Bᴴ D).PosSemidef ↔ (D - Bᴴ * A⁻¹ * B).PosSemidef := by
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝⁷ : CommRing 𝕜
    inst✝⁶ : StarRing 𝕜
    inst✝⁵ : PartialOrder 𝕜
    inst✝⁴ : StarOrderedRing 𝕜
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hA : A.PosDef
    inst✝ : Invertible A
    ⊢ Iff (Matrix.fromBlocks A B B.conjTranspose D).PosSemidef (HSub.hSub D (HMul. …
  -/
  rw [PosSemidef, IsHermitian.fromBlocks₁₁ _ _ hA.1]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝⁷ : CommRing 𝕜
    inst✝⁶ : StarRing 𝕜
    inst✝⁵ : PartialOrder 𝕜
    inst✝⁴ : StarOrderedRing 𝕜
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hA : A.PosDef
    inst✝ : Invertible A
    ⊢ Iff (And (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)) …
  -/
  constructor
    /-
      case mp
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      ⊢ And (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).IsHe …
    -/
  · refine fun h => ⟨h.1, fun x => ?_⟩
    /-
      case mp
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      h : And (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).Is …
      x : n → 𝕜
      ⊢ LE.le 0 (dotProduct (Star.star x) ((HSub.hSub D (HMul.hMul (HMul.hMul B.conj …
    -/
    have := h.2 (-((A⁻¹ * B) *ᵥ x) ⊕ᵥ x)
    rw [dotProduct_mulVec, schur_complement_eq₁₁ B D _ _ hA.1, neg_add_cancel, dotProduct_zero,
      zero_add] at this
    /-
      case mp
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      h : And (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).Is …
      x : n → 𝕜
      this : LE.le 0 (dotProduct (Matrix.vecMul (Star.star x) (HSub.hSub D (HMul.hMu …
      ⊢ LE.le 0 (dotProduct (Star.star x) ((HSub.hSub D (HMul.hMul (HMul.hMul B.conj …
    -/
    rw [dotProduct_mulVec]; exact this
                            /-
                              🎉 no goals
                            -/
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      ⊢ (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSemid …
    -/
  · refine fun h => ⟨h.1, fun x => ?_⟩
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
      x : Sum m n → 𝕜
      ⊢ LE.le 0 (dotProduct (Star.star x) ((Matrix.fromBlocks A B B.conjTranspose D) …
    -/
    rw [dotProduct_mulVec, ← Sum.elim_comp_inl_inr x, schur_complement_eq₁₁ B D _ _ hA.1]
    /-
      case mpr
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : DecidableEq m
      inst✝¹ : Fintype n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hA : A.PosDef
      inst✝ : Invertible A
      h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
      x : Sum m n → 𝕜
      ⊢ LE.le 0 (HAdd.hAdd (dotProduct (Matrix.vecMul (Star.star (HAdd.hAdd (Functio …
    -/
    apply le_add_of_nonneg_of_le
      /-
        case mpr.ha
        m : Type u_2
        n : Type u_3
        𝕜 : Type u_5
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : StarRing 𝕜
        inst✝⁵ : PartialOrder 𝕜
        inst✝⁴ : StarOrderedRing 𝕜
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        A : Matrix m m 𝕜
        B : Matrix m n 𝕜
        D : Matrix n n 𝕜
        hA : A.PosDef
        inst✝ : Invertible A
        h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
        x : Sum m n → 𝕜
        ⊢ LE.le 0 (dotProduct (Matrix.vecMul (Star.star (HAdd.hAdd (Function.comp x Su …
      -/
    · rw [← dotProduct_mulVec]
      /-
        case mpr.ha
        m : Type u_2
        n : Type u_3
        𝕜 : Type u_5
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : StarRing 𝕜
        inst✝⁵ : PartialOrder 𝕜
        inst✝⁴ : StarOrderedRing 𝕜
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        A : Matrix m m 𝕜
        B : Matrix m n 𝕜
        D : Matrix n n 𝕜
        hA : A.PosDef
        inst✝ : Invertible A
        h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
        x : Sum m n → 𝕜
        ⊢ LE.le 0 (dotProduct (Star.star (HAdd.hAdd (Function.comp x Sum.inl) ((HMul.h …
      -/
      apply hA.posSemidef.2
      /-
        🎉 no goals
      -/
      /-
        case mpr.hbc
        m : Type u_2
        n : Type u_3
        𝕜 : Type u_5
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : StarRing 𝕜
        inst✝⁵ : PartialOrder 𝕜
        inst✝⁴ : StarOrderedRing 𝕜
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        A : Matrix m m 𝕜
        B : Matrix m n 𝕜
        D : Matrix n n 𝕜
        hA : A.PosDef
        inst✝ : Invertible A
        h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
        x : Sum m n → 𝕜
        ⊢ LE.le 0 (dotProduct (Matrix.vecMul (Star.star (Function.comp x Sum.inr)) (HS …
      -/
    · rw [← dotProduct_mulVec (star (x ∘ Sum.inr))]
      /-
        case mpr.hbc
        m : Type u_2
        n : Type u_3
        𝕜 : Type u_5
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : StarRing 𝕜
        inst✝⁵ : PartialOrder 𝕜
        inst✝⁴ : StarOrderedRing 𝕜
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        A : Matrix m m 𝕜
        B : Matrix m n 𝕜
        D : Matrix n n 𝕜
        hA : A.PosDef
        inst✝ : Invertible A
        h : (HSub.hSub D (HMul.hMul (HMul.hMul B.conjTranspose (Inv.inv A)) B)).PosSem …
        x : Sum m n → 𝕜
        ⊢ LE.le 0 (dotProduct (Star.star (Function.comp x Sum.inr)) ((HSub.hSub D (HMu …
      -/
      apply h.2
      /-
        🎉 no goals
      -/


theorem PosSemidef.fromBlocks₂₂ [Fintype m] [Fintype n] [DecidableEq n] (A : Matrix m m 𝕜)
    (B : Matrix m n 𝕜) {D : Matrix n n 𝕜} (hD : D.PosDef) [Invertible D] :
    (fromBlocks A B Bᴴ D).PosSemidef ↔ (A - B * D⁻¹ * Bᴴ).PosSemidef := by
  rw [← posSemidef_submatrix_equiv (Equiv.sumComm n m), Equiv.sumComm_apply,
    fromBlocks_submatrix_sum_swap_sum_swap]
  /-
    m : Type u_2
    n : Type u_3
    𝕜 : Type u_5
    inst✝⁷ : CommRing 𝕜
    inst✝⁶ : StarRing 𝕜
    inst✝⁵ : PartialOrder 𝕜
    inst✝⁴ : StarOrderedRing 𝕜
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    A : Matrix m m 𝕜
    B : Matrix m n 𝕜
    D : Matrix n n 𝕜
    hD : D.PosDef
    inst✝ : Invertible D
    ⊢ Iff (Matrix.fromBlocks D B.conjTranspose B A).PosSemidef (HSub.hSub A (HMul. …
  -/
  convert PosSemidef.fromBlocks₁₁ Bᴴ A hD <;>
    /-
      case h.e'_1.h.e'_7.h.e'_8
      m : Type u_2
      n : Type u_3
      𝕜 : Type u_5
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : StarRing 𝕜
      inst✝⁵ : PartialOrder 𝕜
      inst✝⁴ : StarOrderedRing 𝕜
      inst✝³ : Fintype m
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      A : Matrix m m 𝕜
      B : Matrix m n 𝕜
      D : Matrix n n 𝕜
      hD : D.PosDef
      inst✝ : Invertible D
      ⊢ Eq B B.conjTranspose.conjTranspose
    -/
    /-
      🎉 no goals
    -/
    simp
    /-
      🎉 no goals
    -/


