/--
`Matrix.col ι u` the matrix with all columns equal to the vector `u`.

To get a column matrix with exactly one column, `Matrix.col (Fin 1) u` is the canonical choice.
-/
def col (ι : Type*) (w : m → α) : Matrix m ι α :=
  of fun x _ => w x

-- TODO: set as an equation lemma for `col`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem col_apply {ι : Type*} (w : m → α) (i) (j : ι) : col ι w i j = w i :=
  rfl


/--
`Matrix.row ι u` the matrix with all rows equal to the vector `u`.

To get a row matrix with exactly one row, `Matrix.row (Fin 1) u` is the canonical choice.
-/
def row (ι : Type*) (v : n → α) : Matrix ι n α :=
  of fun _ y => v y


@[simp]
theorem row_apply (v : n → α) (i : ι) (j) : row ι v i j = v j :=
  rfl


theorem col_injective [Nonempty ι] : Function.Injective (col ι : (m → α) → Matrix m ι α) := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Nonempty ι
    ⊢ Function.Injective (Matrix.col ι)
  -/
  inhabit ι
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Nonempty ι
    inhabited_h : Inhabited ι
    ⊢ Function.Injective (Matrix.col ι)
  -/
  exact fun _x _y h => funext fun i => congr_fun₂ h i default
  /-
    🎉 no goals
  -/


@[simp] theorem col_inj [Nonempty ι] {v w : m → α} : col ι v = col ι w ↔ v = w :=
  col_injective.eq_iff


@[simp] theorem col_zero [Zero α] : col ι (0 : m → α) = 0 := rfl


@[simp] theorem col_eq_zero [Zero α] [Nonempty ι] (v : m → α) : col ι v = 0 ↔ v = 0 := col_inj


@[simp]
theorem col_add [Add α] (v w : m → α) : col ι (v + w) = col ι v + col ι w := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Add α
    v w : m → α
    ⊢ Eq (Matrix.col ι (HAdd.hAdd v w)) (HAdd.hAdd (Matrix.col ι v) (Matrix.col ι  …
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Add α
    v w : m → α
    i✝ : m
    j✝ : ι
    ⊢ Eq (Matrix.col ι (HAdd.hAdd v w) i✝ j✝) (HAdd.hAdd (Matrix.col ι v) (Matrix. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem col_smul [SMul R α] (x : R) (v : m → α) : col ι (x • v) = x • col ι v := by
  /-
    m : Type u_2
    R : Type u_5
    α : Type v
    ι : Type u_6
    inst✝ : SMul R α
    x : R
    v : m → α
    ⊢ Eq (Matrix.col ι (HSMul.hSMul x v)) (HSMul.hSMul x (Matrix.col ι v))
  -/
  ext
  /-
    case a
    m : Type u_2
    R : Type u_5
    α : Type v
    ι : Type u_6
    inst✝ : SMul R α
    x : R
    v : m → α
    i✝ : m
    j✝ : ι
    ⊢ Eq (Matrix.col ι (HSMul.hSMul x v) i✝ j✝) (HSMul.hSMul x (Matrix.col ι v) i✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem row_injective [Nonempty ι] : Function.Injective (row ι : (n → α) → Matrix ι n α) := by
  /-
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝ : Nonempty ι
    ⊢ Function.Injective (Matrix.row ι)
  -/
  inhabit ι
  /-
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝ : Nonempty ι
    inhabited_h : Inhabited ι
    ⊢ Function.Injective (Matrix.row ι)
  -/
  exact fun _x _y h => funext fun j => congr_fun₂ h default j
  /-
    🎉 no goals
  -/


@[simp] theorem row_inj [Nonempty ι] {v w : n → α} : row ι v = row ι w ↔ v = w :=
  row_injective.eq_iff


@[simp] theorem row_zero [Zero α] : row ι (0 : n → α) = 0 := rfl


@[simp] theorem row_eq_zero [Zero α] [Nonempty ι] (v : n → α) : row ι v = 0 ↔ v = 0 := row_inj


@[simp]
theorem row_add [Add α] (v w : m → α) : row ι (v + w) = row ι v + row ι w := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Add α
    v w : m → α
    ⊢ Eq (Matrix.row ι (HAdd.hAdd v w)) (HAdd.hAdd (Matrix.row ι v) (Matrix.row ι  …
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Add α
    v w : m → α
    i✝ : ι
    j✝ : m
    ⊢ Eq (Matrix.row ι (HAdd.hAdd v w) i✝ j✝) (HAdd.hAdd (Matrix.row ι v) (Matrix. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem row_smul [SMul R α] (x : R) (v : m → α) : row ι (x • v) = x • row ι v := by
  /-
    m : Type u_2
    R : Type u_5
    α : Type v
    ι : Type u_6
    inst✝ : SMul R α
    x : R
    v : m → α
    ⊢ Eq (Matrix.row ι (HSMul.hSMul x v)) (HSMul.hSMul x (Matrix.row ι v))
  -/
  ext
  /-
    case a
    m : Type u_2
    R : Type u_5
    α : Type v
    ι : Type u_6
    inst✝ : SMul R α
    x : R
    v : m → α
    i✝ : ι
    j✝ : m
    ⊢ Eq (Matrix.row ι (HSMul.hSMul x v) i✝ j✝) (HSMul.hSMul x (Matrix.row ι v) i✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem transpose_col (v : m → α) : (Matrix.col ι v)ᵀ = Matrix.row ι v := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    v : m → α
    ⊢ Eq (Matrix.col ι v).transpose (Matrix.row ι v)
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    v : m → α
    i✝ : ι
    j✝ : m
    ⊢ Eq ((Matrix.col ι v).transpose i✝ j✝) (Matrix.row ι v i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem transpose_row (v : m → α) : (Matrix.row ι v)ᵀ = Matrix.col ι v := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    v : m → α
    ⊢ Eq (Matrix.row ι v).transpose (Matrix.col ι v)
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    v : m → α
    i✝ : m
    j✝ : ι
    ⊢ Eq ((Matrix.row ι v).transpose i✝ j✝) (Matrix.col ι v i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem conjTranspose_col [Star α] (v : m → α) : (col ι v)ᴴ = row ι (star v) := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Star α
    v : m → α
    ⊢ Eq (Matrix.col ι v).conjTranspose (Matrix.row ι (Star.star v))
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Star α
    v : m → α
    i✝ : ι
    j✝ : m
    ⊢ Eq ((Matrix.col ι v).conjTranspose i✝ j✝) (Matrix.row ι (Star.star v) i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem conjTranspose_row [Star α] (v : m → α) : (row ι v)ᴴ = col ι (star v) := by
  /-
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Star α
    v : m → α
    ⊢ Eq (Matrix.row ι v).conjTranspose (Matrix.col ι (Star.star v))
  -/
  ext
  /-
    case a
    m : Type u_2
    α : Type v
    ι : Type u_6
    inst✝ : Star α
    v : m → α
    i✝ : m
    j✝ : ι
    ⊢ Eq ((Matrix.row ι v).conjTranspose i✝ j✝) (Matrix.col ι (Star.star v) i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem row_vecMul [Fintype m] [NonUnitalNonAssocSemiring α] (M : Matrix m n α) (v : m → α) :
    Matrix.row ι (v ᵥ* M) = Matrix.row ι v * M := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : m → α
    ⊢ Eq (Matrix.row ι (Matrix.vecMul v M)) (HMul.hMul (Matrix.row ι v) M)
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : m → α
    i✝ : ι
    j✝ : n
    ⊢ Eq (Matrix.row ι (Matrix.vecMul v M) i✝ j✝) (HMul.hMul (Matrix.row ι v) M i✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem col_vecMul [Fintype m] [NonUnitalNonAssocSemiring α] (M : Matrix m n α) (v : m → α) :
    Matrix.col ι (v ᵥ* M) = (Matrix.row ι v * M)ᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : m → α
    ⊢ Eq (Matrix.col ι (Matrix.vecMul v M)) (HMul.hMul (Matrix.row ι v) M).transpose
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : m → α
    i✝ : n
    j✝ : ι
    ⊢ Eq (Matrix.col ι (Matrix.vecMul v M) i✝ j✝) ((HMul.hMul (Matrix.row ι v) M). …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem col_mulVec [Fintype n] [NonUnitalNonAssocSemiring α] (M : Matrix m n α) (v : n → α) :
    Matrix.col ι (M *ᵥ v) = M * Matrix.col ι v := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : n → α
    ⊢ Eq (Matrix.col ι (M.mulVec v)) (HMul.hMul M (Matrix.col ι v))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : n → α
    i✝ : m
    j✝ : ι
    ⊢ Eq (Matrix.col ι (M.mulVec v) i✝ j✝) (HMul.hMul M (Matrix.col ι v) i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem row_mulVec [Fintype n] [NonUnitalNonAssocSemiring α] (M : Matrix m n α) (v : n → α) :
    Matrix.row ι (M *ᵥ v) = (M * Matrix.col ι v)ᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : n → α
    ⊢ Eq (Matrix.row ι (M.mulVec v)) (HMul.hMul M (Matrix.col ι v)).transpose
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    M : Matrix m n α
    v : n → α
    i✝ : ι
    j✝ : m
    ⊢ Eq (Matrix.row ι (M.mulVec v) i✝ j✝) ((HMul.hMul M (Matrix.col ι v)).transpo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem row_mulVec_eq_const [Fintype m] [NonUnitalNonAssocSemiring α] (v w : m → α) :
    Matrix.row ι v *ᵥ w = Function.const _ (v ⬝ᵥ w) := rfl


theorem mulVec_col_eq_const [Fintype m] [NonUnitalNonAssocSemiring α] (v w : m → α) :
    v ᵥ* Matrix.col ι w = Function.const _ (v ⬝ᵥ w) := rfl


theorem row_mul_col [Fintype m] [Mul α] [AddCommMonoid α] (v w : m → α) :
    row ι v * col ι w = of fun _ _ => v ⬝ᵥ w :=
  rfl


@[simp]
theorem row_mul_col_apply [Fintype m] [Mul α] [AddCommMonoid α] (v w : m → α) (i j) :
    (row ι v * col ι w) i j = v ⬝ᵥ w :=
  rfl


@[simp]
theorem diag_col_mul_row [Mul α] [AddCommMonoid α] [Unique ι] (a b : n → α) :
    diag (col ι a * row ι b) = a * b := by
  /-
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝² : Mul α
    inst✝¹ : AddCommMonoid α
    inst✝ : Unique ι
    a b : n → α
    ⊢ Eq (HMul.hMul (Matrix.col ι a) (Matrix.row ι b)).diag (HMul.hMul a b)
  -/
  ext
  /-
    case h
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝² : Mul α
    inst✝¹ : AddCommMonoid α
    inst✝ : Unique ι
    a b : n → α
    x✝ : n
    ⊢ Eq ((HMul.hMul (Matrix.col ι a) (Matrix.row ι b)).diag x✝) (HMul.hMul a b x✝)
  -/
  simp [Matrix.mul_apply, col, row]
  /-
    🎉 no goals
  -/


theorem vecMulVec_eq [Mul α] [AddCommMonoid α] [Unique ι] (w : m → α) (v : n → α) :
    vecMulVec w v = col ι w * row ι v := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝² : Mul α
    inst✝¹ : AddCommMonoid α
    inst✝ : Unique ι
    w : m → α
    v : n → α
    ⊢ Eq (Matrix.vecMulVec w v) (HMul.hMul (Matrix.col ι w) (Matrix.row ι v))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    ι : Type u_6
    inst✝² : Mul α
    inst✝¹ : AddCommMonoid α
    inst✝ : Unique ι
    w : m → α
    v : n → α
    i✝ : m
    j✝ : n
    ⊢ Eq (Matrix.vecMulVec w v i✝ j✝) (HMul.hMul (Matrix.col ι w) (Matrix.row ι v) …
  -/
  simp [vecMulVec, mul_apply]
  /-
    🎉 no goals
  -/


/-- Update, i.e. replace the `i`th row of matrix `A` with the values in `b`. -/
def updateRow [DecidableEq m] (M : Matrix m n α) (i : m) (b : n → α) : Matrix m n α :=
  of <| Function.update M i b


/-- Update, i.e. replace the `j`th column of matrix `A` with the values in `b`. -/
def updateCol [DecidableEq n] (M : Matrix m n α) (j : n) (b : m → α) : Matrix m n α :=
  of fun i => Function.update (M i) j (b i)


@[deprecated (since := "2024-12-11")] alias updateColumn := updateCol


@[simp]
theorem updateRow_self [DecidableEq m] : updateRow M i b i = b :=
  -- Porting note: (implicit arg) added `(β := _)`
  Function.update_self (β := fun _ => (n → α)) i b M


@[simp]
theorem updateCol_self [DecidableEq n] : updateCol M j c i j = c i :=
  -- Porting note: (implicit arg) added `(β := _)`
  Function.update_self (β := fun _ => α) j (c i) (M i)


@[deprecated (since := "2024-12-11")] alias updateColumn_self := updateCol_self


@[simp]
theorem updateRow_ne [DecidableEq m] {i' : m} (i_ne : i' ≠ i) : updateRow M i b i' = M i' :=
  -- Porting note: (implicit arg) added `(β := _)`
  Function.update_of_ne (β := fun _ => (n → α)) i_ne b M


@[simp]
theorem updateCol_ne [DecidableEq n] {j' : n} (j_ne : j' ≠ j) :
    updateCol M j c i j' = M i j' :=
  -- Porting note: (implicit arg) added `(β := _)`
  Function.update_of_ne (β := fun _ => α) j_ne (c i) (M i)


@[deprecated (since := "2024-12-11")] alias updateColumn_ne := updateCol_ne


theorem updateRow_apply [DecidableEq m] {i' : m} :
    updateRow M i b i' j = if i' = i then b j else M i' j := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    j : n
    b : n → α
    inst✝ : DecidableEq m
    i' : m
    ⊢ Eq (M.updateRow i b i' j) (ite (Eq i' i) (b j) (M i' j))
  -/
  by_cases h : i' = i
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type v
      M : Matrix m n α
      i : m
      j : n
      b : n → α
      inst✝ : DecidableEq m
      i' : m
      h : Eq i' i
      ⊢ Eq (M.updateRow i b i' j) (ite (Eq i' i) (b j) (M i' j))
    -/
  · rw [h, updateRow_self, if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      n : Type u_3
      α : Type v
      M : Matrix m n α
      i : m
      j : n
      b : n → α
      inst✝ : DecidableEq m
      i' : m
      h : Not (Eq i' i)
      ⊢ Eq (M.updateRow i b i' j) (ite (Eq i' i) (b j) (M i' j))
    -/
  · rw [updateRow_ne h, if_neg h]
    /-
      🎉 no goals
    -/


theorem updateCol_apply [DecidableEq n] {j' : n} :
    updateCol M j c i j' = if j' = j then c i else M i j' := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    j : n
    c : m → α
    inst✝ : DecidableEq n
    j' : n
    ⊢ Eq (M.updateCol j c i j') (ite (Eq j' j) (c i) (M i j'))
  -/
  by_cases h : j' = j
    /-
      case pos
      m : Type u_2
      n : Type u_3
      α : Type v
      M : Matrix m n α
      i : m
      j : n
      c : m → α
      inst✝ : DecidableEq n
      j' : n
      h : Eq j' j
      ⊢ Eq (M.updateCol j c i j') (ite (Eq j' j) (c i) (M i j'))
    -/
  · rw [h, updateCol_self, if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      n : Type u_3
      α : Type v
      M : Matrix m n α
      i : m
      j : n
      c : m → α
      inst✝ : DecidableEq n
      j' : n
      h : Not (Eq j' j)
      ⊢ Eq (M.updateCol j c i j') (ite (Eq j' j) (c i) (M i j'))
    -/
  · rw [updateCol_ne h, if_neg h]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-11")] alias updateColumn_apply := updateCol_apply


@[simp]
theorem updateCol_subsingleton [Subsingleton n] (A : Matrix m n R) (i : n) (b : m → R) :
    A.updateCol i b = (col (Fin 1) b).submatrix id (Function.const n 0) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝ : Subsingleton n
    A : Matrix m n R
    i : n
    b : m → R
    ⊢ Eq (A.updateCol i b) ((Matrix.col (Fin 1) b).submatrix id (Function.const n  …
  -/
  ext x y
  /-
    case a
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝ : Subsingleton n
    A : Matrix m n R
    i : n
    b : m → R
    x : m
    y : n
    ⊢ Eq (A.updateCol i b x y) ((Matrix.col (Fin 1) b).submatrix id (Function.cons …
  -/
  simp [updateCol_apply, Subsingleton.elim i y]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias updateColumn_subsingleton := updateCol_subsingleton


@[simp]
theorem updateRow_subsingleton [Subsingleton m] (A : Matrix m n R) (i : m) (b : n → R) :
    A.updateRow i b = (row (Fin 1) b).submatrix (Function.const m 0) id := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝ : Subsingleton m
    A : Matrix m n R
    i : m
    b : n → R
    ⊢ Eq (A.updateRow i b) ((Matrix.row (Fin 1) b).submatrix (Function.const m 0)  …
  -/
  ext x y
  /-
    case a
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝ : Subsingleton m
    A : Matrix m n R
    i : m
    b : n → R
    x : m
    y : n
    ⊢ Eq (A.updateRow i b x y) ((Matrix.row (Fin 1) b).submatrix (Function.const m …
  -/
  simp [updateCol_apply, Subsingleton.elim i x]
  /-
    🎉 no goals
  -/


theorem map_updateRow [DecidableEq m] (f : α → β) :
    map (updateRow M i b) f = updateRow (M.map f) i (f ∘ b) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    f : α → β
    ⊢ Eq ((M.updateRow i b).map f) ((M.map f).updateRow i (Function.comp f b))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    f : α → β
    i✝ : m
    j✝ : n
    ⊢ Eq ((M.updateRow i b).map f i✝ j✝) ((M.map f).updateRow i (Function.comp f b …
  -/
  rw [updateRow_apply, map_apply, map_apply, updateRow_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    f : α → β
    i✝ : m
    j✝ : n
    ⊢ Eq (f (ite (Eq i✝ i) (b j✝) (M i✝ j✝))) (ite (Eq i✝ i) (Function.comp f b j✝ …
  -/
  exact apply_ite f _ _ _
  /-
    🎉 no goals
  -/


theorem map_updateCol [DecidableEq n] (f : α → β) :
    map (updateCol M j c) f = updateCol (M.map f) j (f ∘ c) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    f : α → β
    ⊢ Eq ((M.updateCol j c).map f) ((M.map f).updateCol j (Function.comp f c))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    f : α → β
    i✝ : m
    j✝ : n
    ⊢ Eq ((M.updateCol j c).map f i✝ j✝) ((M.map f).updateCol j (Function.comp f c …
  -/
  rw [updateCol_apply, map_apply, map_apply, updateCol_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    β : Type w
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    f : α → β
    i✝ : m
    j✝ : n
    ⊢ Eq (f (ite (Eq j✝ j) (c i✝) (M i✝ j✝))) (ite (Eq j✝ j) (Function.comp f c i✝ …
  -/
  exact apply_ite f _ _ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias map_updateColumn := map_updateCol


theorem updateRow_transpose [DecidableEq n] : updateRow Mᵀ j c = (updateCol M j c)ᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    ⊢ Eq (M.transpose.updateRow j c) (M.updateCol j c).transpose
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    i✝ : n
    j✝ : m
    ⊢ Eq (M.transpose.updateRow j c i✝ j✝) ((M.updateCol j c).transpose i✝ j✝)
  -/
  rw [transpose_apply, updateRow_apply, updateCol_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    j : n
    c : m → α
    inst✝ : DecidableEq n
    i✝ : n
    j✝ : m
    ⊢ Eq (ite (Eq i✝ j) (c j✝) (M.transpose i✝ j✝)) (ite (Eq i✝ j) (c j✝) (M j✝ i✝))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem updateCol_transpose [DecidableEq m] : updateCol Mᵀ i b = (updateRow M i b)ᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    ⊢ Eq (M.transpose.updateCol i b) (M.updateRow i b).transpose
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    i✝ : n
    j✝ : m
    ⊢ Eq (M.transpose.updateCol i b i✝ j✝) ((M.updateRow i b).transpose i✝ j✝)
  -/
  rw [transpose_apply, updateRow_apply, updateCol_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    b : n → α
    inst✝ : DecidableEq m
    i✝ : n
    j✝ : m
    ⊢ Eq (ite (Eq j✝ i) (b i✝) (M.transpose i✝ j✝)) (ite (Eq j✝ i) (b i✝) (M j✝ i✝))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias updateColumn_transpose := updateCol_transpose


theorem updateRow_conjTranspose [DecidableEq n] [Star α] :
    updateRow Mᴴ j (star c) = (updateCol M j c)ᴴ := by
  rw [conjTranspose, conjTranspose, transpose_map, transpose_map, updateRow_transpose,
    map_updateCol]
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    j : n
    c : m → α
    inst✝¹ : DecidableEq n
    inst✝ : Star α
    ⊢ Eq ((M.map Star.star).updateCol j (Star.star c)).transpose ((M.map Star.star …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem updateCol_conjTranspose [DecidableEq m] [Star α] :
    updateCol Mᴴ i (star b) = (updateRow M i b)ᴴ := by
  rw [conjTranspose, conjTranspose, transpose_map, transpose_map, updateCol_transpose,
    map_updateRow]
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    M : Matrix m n α
    i : m
    b : n → α
    inst✝¹ : DecidableEq m
    inst✝ : Star α
    ⊢ Eq ((M.map Star.star).updateRow i (Star.star b)).transpose ((M.map Star.star …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias updateColumn_conjTranspose := updateCol_conjTranspose


@[simp]
theorem updateRow_eq_self [DecidableEq m] (A : Matrix m n α) (i : m) : A.updateRow i (A i) = A :=
  Function.update_eq_self i A


@[simp]
theorem updateCol_eq_self [DecidableEq n] (A : Matrix m n α) (i : n) :
    (A.updateCol i fun j => A j i) = A :=
  funext fun j => Function.update_eq_self i (A j)


@[deprecated (since := "2024-12-11")] alias updateColumn_eq_self := updateCol_eq_self


theorem diagonal_updateCol_single [DecidableEq n] [Zero α] (v : n → α) (i : n) (x : α) :
    (diagonal v).updateCol i (Pi.single i x) = diagonal (Function.update v i x) := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    v : n → α
    i : n
    x : α
    ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x)) (Matrix.diagonal (Funct …
  -/
  ext j k
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    v : n → α
    i : n
    x : α
    j k : n
    ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j k) (Matrix.diagonal (F …
  -/
  obtain rfl | hjk := eq_or_ne j k
    /-
      case a.inl
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i : n
      x : α
      j : n
      ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j j) (Matrix.diagonal (F …
    -/
  · rw [diagonal_apply_eq]
    /-
      case a.inl
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i : n
      x : α
      j : n
      ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j j) (Function.update v  …
    -/
    obtain rfl | hji := eq_or_ne j i
      /-
        case a.inl.inl
        n : Type u_3
        α : Type v
        inst✝¹ : DecidableEq n
        inst✝ : Zero α
        v : n → α
        x : α
        j : n
        ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single j x) j j) (Function.update v  …
      -/
    · rw [updateCol_self, Pi.single_eq_same, Function.update_self]
      /-
        🎉 no goals
      -/
      /-
        case a.inl.inr
        n : Type u_3
        α : Type v
        inst✝¹ : DecidableEq n
        inst✝ : Zero α
        v : n → α
        i : n
        x : α
        j : n
        hji : Ne j i
        ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j j) (Function.update v  …
      -/
    · rw [updateCol_ne hji, diagonal_apply_eq, Function.update_of_ne hji]
      /-
        🎉 no goals
      -/
    /-
      case a.inr
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i : n
      x : α
      j k : n
      hjk : Ne j k
      ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j k) (Matrix.diagonal (F …
    -/
  · rw [diagonal_apply_ne _ hjk]
    /-
      case a.inr
      n : Type u_3
      α : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Zero α
      v : n → α
      i : n
      x : α
      j k : n
      hjk : Ne j k
      ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j k) 0
    -/
    obtain rfl | hki := eq_or_ne k i
      /-
        case a.inr.inl
        n : Type u_3
        α : Type v
        inst✝¹ : DecidableEq n
        inst✝ : Zero α
        v : n → α
        x : α
        j k : n
        hjk : Ne j k
        ⊢ Eq ((Matrix.diagonal v).updateCol k (Pi.single k x) j k) 0
      -/
    · rw [updateCol_self, Pi.single_eq_of_ne hjk]
      /-
        🎉 no goals
      -/
      /-
        case a.inr.inr
        n : Type u_3
        α : Type v
        inst✝¹ : DecidableEq n
        inst✝ : Zero α
        v : n → α
        i : n
        x : α
        j k : n
        hjk : Ne j k
        hki : Ne k i
        ⊢ Eq ((Matrix.diagonal v).updateCol i (Pi.single i x) j k) 0
      -/
    · rw [updateCol_ne hki, diagonal_apply_ne _ hjk]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-12-11")]
alias diagonal_updateColumn_single := diagonal_updateCol_single


theorem diagonal_updateRow_single [DecidableEq n] [Zero α] (v : n → α) (i : n) (x : α) :
    (diagonal v).updateRow i (Pi.single i x) = diagonal (Function.update v i x) := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    v : n → α
    i : n
    x : α
    ⊢ Eq ((Matrix.diagonal v).updateRow i (Pi.single i x)) (Matrix.diagonal (Funct …
  -/
  rw [← diagonal_transpose, updateRow_transpose, diagonal_updateCol_single, diagonal_transpose]
  /-
    🎉 no goals
  -/


theorem updateRow_submatrix_equiv [DecidableEq l] [DecidableEq m] (A : Matrix m n α) (i : l)
    (r : o → α) (e : l ≃ m) (f : o ≃ n) :
    updateRow (A.submatrix e f) i r = (A.updateRow (e i) fun j => r (f.symm j)).submatrix e f := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : DecidableEq l
    inst✝ : DecidableEq m
    A : Matrix m n α
    i : l
    r : o → α
    e : Equiv l m
    f : Equiv o n
    ⊢ Eq ((A.submatrix ⇑e ⇑f).updateRow i r) ((A.updateRow (e i) fun j => r (f.sym …
  -/
  ext i' j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : DecidableEq l
    inst✝ : DecidableEq m
    A : Matrix m n α
    i : l
    r : o → α
    e : Equiv l m
    f : Equiv o n
    i' : l
    j : o
    ⊢ Eq ((A.submatrix ⇑e ⇑f).updateRow i r i' j) ((A.updateRow (e i) fun j => r ( …
  -/
  simp only [submatrix_apply, updateRow_apply, Equiv.apply_eq_iff_eq, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem submatrix_updateRow_equiv [DecidableEq l] [DecidableEq m] (A : Matrix m n α) (i : m)
    (r : n → α) (e : l ≃ m) (f : o ≃ n) :
    (A.updateRow i r).submatrix e f = updateRow (A.submatrix e f) (e.symm i) fun i => r (f i) :=
               /-
                 l : Type u_1
                 m : Type u_2
                 n : Type u_3
                 o : Type u_4
                 α : Type v
                 inst✝¹ : DecidableEq l
                 inst✝ : DecidableEq m
                 A : Matrix m n α
                 i : m
                 r : n → α
                 e : Equiv l m
                 f : Equiv o n
                 ⊢ Eq ((A.updateRow i r).submatrix ⇑e ⇑f) ((A.updateRow (e (e.symm i)) fun j => …
               -/
  Eq.trans (by simp_rw [Equiv.apply_symm_apply]) (updateRow_submatrix_equiv A _ _ e f).symm
               /-
                 🎉 no goals
               -/


theorem updateCol_submatrix_equiv [DecidableEq o] [DecidableEq n] (A : Matrix m n α) (j : o)
    (c : l → α) (e : l ≃ m) (f : o ≃ n) : updateCol (A.submatrix e f) j c =
    (A.updateCol (f j) fun i => c (e.symm i)).submatrix e f := by
  simpa only [← transpose_submatrix, updateRow_transpose] using
    congr_arg transpose (updateRow_submatrix_equiv Aᵀ j c f e)


@[deprecated (since := "2024-12-11")]
alias updateColumn_submatrix_equiv := updateCol_submatrix_equiv


theorem submatrix_updateCol_equiv [DecidableEq o] [DecidableEq n] (A : Matrix m n α) (j : n)
    (c : m → α) (e : l ≃ m) (f : o ≃ n) : (A.updateCol j c).submatrix e f =
    updateCol (A.submatrix e f) (f.symm j) fun i => c (e i) :=
               /-
                 l : Type u_1
                 m : Type u_2
                 n : Type u_3
                 o : Type u_4
                 α : Type v
                 inst✝¹ : DecidableEq o
                 inst✝ : DecidableEq n
                 A : Matrix m n α
                 j : n
                 c : m → α
                 e : Equiv l m
                 f : Equiv o n
                 ⊢ Eq ((A.updateCol j c).submatrix ⇑e ⇑f) ((A.updateCol (f (f.symm j)) fun i => …
               -/
  Eq.trans (by simp_rw [Equiv.apply_symm_apply]) (updateCol_submatrix_equiv A _ _ e f).symm
               /-
                 🎉 no goals
               -/


@[deprecated (since := "2024-12-11")]
alias submatrix_updateColumn_equiv := submatrix_updateCol_equiv


theorem updateRow_reindex [DecidableEq l] [DecidableEq m] (A : Matrix m n α) (i : l) (r : o → α)
    (e : m ≃ l) (f : n ≃ o) :
    updateRow (reindex e f A) i r = reindex e f (A.updateRow (e.symm i) fun j => r (f j)) :=
  updateRow_submatrix_equiv _ _ _ _ _


theorem reindex_updateRow [DecidableEq l] [DecidableEq m] (A : Matrix m n α) (i : m) (r : n → α)
    (e : m ≃ l) (f : n ≃ o) :
    reindex e f (A.updateRow i r) = updateRow (reindex e f A) (e i) fun i => r (f.symm i) :=
  submatrix_updateRow_equiv _ _ _ _ _


theorem updateCol_reindex [DecidableEq o] [DecidableEq n] (A : Matrix m n α) (j : o) (c : l → α)
    (e : m ≃ l) (f : n ≃ o) :
    updateCol (reindex e f A) j c = reindex e f (A.updateCol (f.symm j) fun i => c (e i)) :=
  updateCol_submatrix_equiv _ _ _ _ _


@[deprecated (since := "2024-12-11")] alias updateColumn_reindex := updateCol_reindex


theorem reindex_updateCol [DecidableEq o] [DecidableEq n] (A : Matrix m n α) (j : n) (c : m → α)
    (e : m ≃ l) (f : n ≃ o) :
    reindex e f (A.updateCol j c) = updateCol (reindex e f A) (f j) fun i => c (e.symm i) :=
  submatrix_updateCol_equiv _ _ _ _ _


@[deprecated (since := "2024-12-11")] alias reindex_updateColumn := reindex_updateCol


