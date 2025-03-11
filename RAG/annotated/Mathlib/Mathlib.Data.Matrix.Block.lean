theorem dotProduct_block [Fintype m] [Fintype n] [Mul α] [AddCommMonoid α] (v w : m ⊕ n → α) :
    v ⬝ᵥ w = v ∘ Sum.inl ⬝ᵥ w ∘ Sum.inl + v ∘ Sum.inr ⬝ᵥ w ∘ Sum.inr :=
  Fintype.sum_sum_type _


/-- We can form a single large matrix by flattening smaller 'block' matrices of compatible
dimensions. -/
@[pp_nodot]
def fromBlocks (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α) :
    Matrix (n ⊕ o) (l ⊕ m) α :=
  of <| Sum.elim (fun i => Sum.elim (A i) (B i)) (fun j => Sum.elim (C j) (D j))


@[simp]
theorem fromBlocks_apply₁₁ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (i : n) (j : l) : fromBlocks A B C D (Sum.inl i) (Sum.inl j) = A i j :=
  rfl


@[simp]
theorem fromBlocks_apply₁₂ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (i : n) (j : m) : fromBlocks A B C D (Sum.inl i) (Sum.inr j) = B i j :=
  rfl


@[simp]
theorem fromBlocks_apply₂₁ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (i : o) (j : l) : fromBlocks A B C D (Sum.inr i) (Sum.inl j) = C i j :=
  rfl


@[simp]
theorem fromBlocks_apply₂₂ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (i : o) (j : m) : fromBlocks A B C D (Sum.inr i) (Sum.inr j) = D i j :=
  rfl


/-- Given a matrix whose row and column indexes are sum types, we can extract the corresponding
"top left" submatrix. -/
def toBlocks₁₁ (M : Matrix (n ⊕ o) (l ⊕ m) α) : Matrix n l α :=
  of fun i j => M (Sum.inl i) (Sum.inl j)


/-- Given a matrix whose row and column indexes are sum types, we can extract the corresponding
"top right" submatrix. -/
def toBlocks₁₂ (M : Matrix (n ⊕ o) (l ⊕ m) α) : Matrix n m α :=
  of fun i j => M (Sum.inl i) (Sum.inr j)


/-- Given a matrix whose row and column indexes are sum types, we can extract the corresponding
"bottom left" submatrix. -/
def toBlocks₂₁ (M : Matrix (n ⊕ o) (l ⊕ m) α) : Matrix o l α :=
  of fun i j => M (Sum.inr i) (Sum.inl j)


/-- Given a matrix whose row and column indexes are sum types, we can extract the corresponding
"bottom right" submatrix. -/
def toBlocks₂₂ (M : Matrix (n ⊕ o) (l ⊕ m) α) : Matrix o m α :=
  of fun i j => M (Sum.inr i) (Sum.inr j)


theorem fromBlocks_toBlocks (M : Matrix (n ⊕ o) (l ⊕ m) α) :
    fromBlocks M.toBlocks₁₁ M.toBlocks₁₂ M.toBlocks₂₁ M.toBlocks₂₂ = M := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    M : Matrix (Sum n o) (Sum l m) α
    ⊢ Eq (Matrix.fromBlocks M.toBlocks₁₁ M.toBlocks₁₂ M.toBlocks₂₁ M.toBlocks₂₂) M
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    M : Matrix (Sum n o) (Sum l m) α
    i : Sum n o
    j : Sum l m
    ⊢ Eq (Matrix.fromBlocks M.toBlocks₁₁ M.toBlocks₁₂ M.toBlocks₂₁ M.toBlocks₂₂ i  …
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
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> rfl
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem toBlocks_fromBlocks₁₁ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D).toBlocks₁₁ = A :=
  rfl


@[simp]
theorem toBlocks_fromBlocks₁₂ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D).toBlocks₁₂ = B :=
  rfl


@[simp]
theorem toBlocks_fromBlocks₂₁ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D).toBlocks₂₁ = C :=
  rfl


@[simp]
theorem toBlocks_fromBlocks₂₂ (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D).toBlocks₂₂ = D :=
  rfl


/-- Two block matrices are equal if their blocks are equal. -/
theorem ext_iff_blocks {A B : Matrix (n ⊕ o) (l ⊕ m) α} :
    A = B ↔
      A.toBlocks₁₁ = B.toBlocks₁₁ ∧
        A.toBlocks₁₂ = B.toBlocks₁₂ ∧ A.toBlocks₂₁ = B.toBlocks₂₁ ∧ A.toBlocks₂₂ = B.toBlocks₂₂ :=
  ⟨fun h => h ▸ ⟨rfl, rfl, rfl, rfl⟩, fun ⟨h₁₁, h₁₂, h₂₁, h₂₂⟩ => by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      o : Type u_4
      α : Type u_12
      A B : Matrix (Sum n o) (Sum l m) α
      x✝ : And (Eq A.toBlocks₁₁ B.toBlocks₁₁) (And (Eq A.toBlocks₁₂ B.toBlocks₁₂) (A …
      h₁₁ : Eq A.toBlocks₁₁ B.toBlocks₁₁
      h₁₂ : Eq A.toBlocks₁₂ B.toBlocks₁₂
      h₂₁ : Eq A.toBlocks₂₁ B.toBlocks₂₁
      h₂₂ : Eq A.toBlocks₂₂ B.toBlocks₂₂
      ⊢ Eq A B
    -/
    rw [← fromBlocks_toBlocks A, ← fromBlocks_toBlocks B, h₁₁, h₁₂, h₂₁, h₂₂]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem fromBlocks_inj {A : Matrix n l α} {B : Matrix n m α} {C : Matrix o l α} {D : Matrix o m α}
    {A' : Matrix n l α} {B' : Matrix n m α} {C' : Matrix o l α} {D' : Matrix o m α} :
    fromBlocks A B C D = fromBlocks A' B' C' D' ↔ A = A' ∧ B = B' ∧ C = C' ∧ D = D' :=
  ext_iff_blocks


theorem fromBlocks_map (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α)
    (f : α → β) : (fromBlocks A B C D).map f =
      fromBlocks (A.map f) (B.map f) (C.map f) (D.map f) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    β : Type u_13
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    f : α → β
    ⊢ Eq ((Matrix.fromBlocks A B C D).map f) (Matrix.fromBlocks (A.map f) (B.map f …
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
  ext i j; rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp [fromBlocks]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem fromBlocks_transpose (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D)ᵀ = fromBlocks Aᵀ Cᵀ Bᵀ Dᵀ := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    ⊢ Eq (Matrix.fromBlocks A B C D).transpose (Matrix.fromBlocks A.transpose C.tr …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    i : Sum l m
    j : Sum n o
    ⊢ Eq ((Matrix.fromBlocks A B C D).transpose i j) (Matrix.fromBlocks A.transpos …
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
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp [fromBlocks]
                                            /-
                                              🎉 no goals
                                            -/


theorem fromBlocks_conjTranspose [Star α] (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : (fromBlocks A B C D)ᴴ = fromBlocks Aᴴ Cᴴ Bᴴ Dᴴ := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝ : Star α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    ⊢ Eq (Matrix.fromBlocks A B C D).conjTranspose (Matrix.fromBlocks A.conjTransp …
  -/
  simp only [conjTranspose, fromBlocks_transpose, fromBlocks_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem fromBlocks_submatrix_sum_swap_left (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (f : p → l ⊕ m) :
    (fromBlocks A B C D).submatrix Sum.swap f = (fromBlocks C D A B).submatrix id f := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    f : p → Sum l m
    ⊢ Eq ((Matrix.fromBlocks A B C D).submatrix Sum.swap f) ((Matrix.fromBlocks C  …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    f : p → Sum l m
    i : Sum o n
    j : p
    ⊢ Eq ((Matrix.fromBlocks A B C D).submatrix Sum.swap f i j) ((Matrix.fromBlock …
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
  cases i <;> dsimp <;> cases f j <;> rfl
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem fromBlocks_submatrix_sum_swap_right (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (f : p → n ⊕ o) :
    (fromBlocks A B C D).submatrix f Sum.swap = (fromBlocks B A D C).submatrix f id := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    f : p → Sum n o
    ⊢ Eq ((Matrix.fromBlocks A B C D).submatrix f Sum.swap) ((Matrix.fromBlocks B  …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    f : p → Sum n o
    i : p
    j : Sum m l
    ⊢ Eq ((Matrix.fromBlocks A B C D).submatrix f Sum.swap i j) ((Matrix.fromBlock …
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
  cases j <;> dsimp <;> cases f i <;> rfl
                                      /-
                                        🎉 no goals
                                      -/


theorem fromBlocks_submatrix_sum_swap_sum_swap {l m n o α : Type*} (A : Matrix n l α)
    (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α) :
                                                                                /-
                                                                                  l : Type u_14
                                                                                  m : Type u_15
                                                                                  n : Type u_16
                                                                                  o : Type u_17
                                                                                  α : Type u_18
                                                                                  A : Matrix n l α
                                                                                  B : Matrix n m α
                                                                                  C : Matrix o l α
                                                                                  D : Matrix o m α
                                                                                  ⊢ Eq ((Matrix.fromBlocks A B C D).submatrix Sum.swap Sum.swap) (Matrix.fromBlo …
                                                                                -/
    (fromBlocks A B C D).submatrix Sum.swap Sum.swap = fromBlocks D C B A := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- A 2x2 block matrix is block diagonal if the blocks outside of the diagonal vanish -/
def IsTwoBlockDiagonal [Zero α] (A : Matrix (n ⊕ o) (l ⊕ m) α) : Prop :=
  toBlocks₁₂ A = 0 ∧ toBlocks₂₁ A = 0


/-- Let `p` pick out certain rows and `q` pick out certain columns of a matrix `M`. Then
  `toBlock M p q` is the corresponding block matrix. -/
def toBlock (M : Matrix m n α) (p : m → Prop) (q : n → Prop) : Matrix { a // p a } { a // q a } α :=
  M.submatrix (↑) (↑)


@[simp]
theorem toBlock_apply (M : Matrix m n α) (p : m → Prop) (q : n → Prop) (i : { a // p a })
    (j : { a // q a }) : toBlock M p q i j = M ↑i ↑j :=
  rfl


/-- Let `p` pick out certain rows and columns of a square matrix `M`. Then
  `toSquareBlockProp M p` is the corresponding block matrix. -/
def toSquareBlockProp (M : Matrix m m α) (p : m → Prop) : Matrix { a // p a } { a // p a } α :=
  toBlock M _ _


theorem toSquareBlockProp_def (M : Matrix m m α) (p : m → Prop) :
    -- Porting note: added missing `of`
    toSquareBlockProp M p = of (fun i j : { a // p a } => M ↑i ↑j) :=
  rfl


/-- Let `b` map rows and columns of a square matrix `M` to blocks. Then
  `toSquareBlock M b k` is the block `k` matrix. -/
def toSquareBlock (M : Matrix m m α) (b : m → β) (k : β) :
    Matrix { a // b a = k } { a // b a = k } α :=
  toSquareBlockProp M _


theorem toSquareBlock_def (M : Matrix m m α) (b : m → β) (k : β) :
    -- Porting note: added missing `of`
    toSquareBlock M b k = of (fun i j : { a // b a = k } => M ↑i ↑j) :=
  rfl


theorem fromBlocks_smul [SMul R α] (x : R) (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) : x • fromBlocks A B C D = fromBlocks (x • A) (x • B) (x • C) (x • D) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_10
    α : Type u_12
    inst✝ : SMul R α
    x : R
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    ⊢ Eq (HSMul.hSMul x (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (HSMul.hSM …
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
  ext i j; rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp [fromBlocks]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem fromBlocks_neg [Neg R] (A : Matrix n l R) (B : Matrix n m R) (C : Matrix o l R)
    (D : Matrix o m R) : -fromBlocks A B C D = fromBlocks (-A) (-B) (-C) (-D) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_10
    inst✝ : Neg R
    A : Matrix n l R
    B : Matrix n m R
    C : Matrix o l R
    D : Matrix o m R
    ⊢ Eq (Neg.neg (Matrix.fromBlocks A B C D)) (Matrix.fromBlocks (Neg.neg A) (Neg …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_10
    inst✝ : Neg R
    A : Matrix n l R
    B : Matrix n m R
    C : Matrix o l R
    D : Matrix o m R
    i : Sum n o
    j : Sum l m
    ⊢ Eq (Neg.neg (Matrix.fromBlocks A B C D) i j) (Matrix.fromBlocks (Neg.neg A)  …
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
  cases i <;> cases j <;> simp [fromBlocks]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem fromBlocks_zero [Zero α] : fromBlocks (0 : Matrix n l α) 0 0 (0 : Matrix o m α) = 0 := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝ : Zero α
    ⊢ Eq (Matrix.fromBlocks 0 0 0 0) 0
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝ : Zero α
    i : Sum n o
    j : Sum l m
    ⊢ Eq (Matrix.fromBlocks 0 0 0 0 i j) (0 i j)
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
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> rfl
                                            /-
                                              🎉 no goals
                                            -/


theorem fromBlocks_add [Add α] (A : Matrix n l α) (B : Matrix n m α) (C : Matrix o l α)
    (D : Matrix o m α) (A' : Matrix n l α) (B' : Matrix n m α) (C' : Matrix o l α)
    (D' : Matrix o m α) : fromBlocks A B C D + fromBlocks A' B' C' D' =
      fromBlocks (A + A') (B + B') (C + C') (D + D') := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝ : Add α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    A' : Matrix n l α
    B' : Matrix n m α
    C' : Matrix o l α
    D' : Matrix o m α
    ⊢ Eq (HAdd.hAdd (Matrix.fromBlocks A B C D) (Matrix.fromBlocks A' B' C' D')) ( …
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
  ext i j; rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem fromBlocks_multiply [Fintype l] [Fintype m] [NonUnitalNonAssocSemiring α] (A : Matrix n l α)
    (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α) (A' : Matrix l p α) (B' : Matrix l q α)
    (C' : Matrix m p α) (D' : Matrix m q α) :
    fromBlocks A B C D * fromBlocks A' B' C' D' =
      fromBlocks (A * A' + B * C') (A * B' + B * D') (C * A' + D * C') (C * B' + D * D') := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    q : Type u_6
    α : Type u_12
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    A' : Matrix l p α
    B' : Matrix l q α
    C' : Matrix m p α
    D' : Matrix m q α
    ⊢ Eq (HMul.hMul (Matrix.fromBlocks A B C D) (Matrix.fromBlocks A' B' C' D')) ( …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    q : Type u_6
    α : Type u_12
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    A' : Matrix l p α
    B' : Matrix l q α
    C' : Matrix m p α
    D' : Matrix m q α
    i : Sum n o
    j : Sum p q
    ⊢ Eq (HMul.hMul (Matrix.fromBlocks A B C D) (Matrix.fromBlocks A' B' C' D') i  …
  -/
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp only [fromBlocks, mul_apply, of_apply,
      Sum.elim_inr, Fintype.sum_sum_type, Sum.elim_inl, add_apply]


theorem fromBlocks_mulVec [Fintype l] [Fintype m] [NonUnitalNonAssocSemiring α] (A : Matrix n l α)
    (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α) (x : l ⊕ m → α) :
    (fromBlocks A B C D) *ᵥ x =
      Sum.elim (A *ᵥ (x ∘ Sum.inl) + B *ᵥ (x ∘ Sum.inr))
        (C *ᵥ (x ∘ Sum.inl) + D *ᵥ (x ∘ Sum.inr)) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    x : Sum l m → α
    ⊢ Eq ((Matrix.fromBlocks A B C D).mulVec x) (Sum.elim (HAdd.hAdd (A.mulVec (Fu …
  -/
  ext i
  /-
    case h
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝² : Fintype l
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    x : Sum l m → α
    i : Sum n o
    ⊢ Eq ((Matrix.fromBlocks A B C D).mulVec x i) (Sum.elim (HAdd.hAdd (A.mulVec ( …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp [mulVec, dotProduct]
              /-
                🎉 no goals
              -/


theorem vecMul_fromBlocks [Fintype n] [Fintype o] [NonUnitalNonAssocSemiring α] (A : Matrix n l α)
    (B : Matrix n m α) (C : Matrix o l α) (D : Matrix o m α) (x : n ⊕ o → α) :
    x ᵥ* fromBlocks A B C D =
      Sum.elim ((x ∘ Sum.inl) ᵥ* A + (x ∘ Sum.inr) ᵥ* C)
        ((x ∘ Sum.inl) ᵥ* B + (x ∘ Sum.inr) ᵥ* D) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝² : Fintype n
    inst✝¹ : Fintype o
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    x : Sum n o → α
    ⊢ Eq (Matrix.vecMul x (Matrix.fromBlocks A B C D)) (Sum.elim (HAdd.hAdd (Matri …
  -/
  ext i
  /-
    case h
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝² : Fintype n
    inst✝¹ : Fintype o
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix n l α
    B : Matrix n m α
    C : Matrix o l α
    D : Matrix o m α
    x : Sum n o → α
    i : Sum l m
    ⊢ Eq (Matrix.vecMul x (Matrix.fromBlocks A B C D) i) (Sum.elim (HAdd.hAdd (Mat …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp [vecMul, dotProduct]
              /-
                🎉 no goals
              -/


theorem toBlock_diagonal_self (d : m → α) (p : m → Prop) :
    Matrix.toBlock (diagonal d) p p = diagonal fun i : Subtype p => d ↑i := by
  /-
    m : Type u_2
    α : Type u_12
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    p : m → Prop
    ⊢ Eq ((Matrix.diagonal d).toBlock p p) (Matrix.diagonal fun i => d ↑i)
  -/
  ext i j
  /-
    case a
    m : Type u_2
    α : Type u_12
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    p : m → Prop
    i j : Subtype fun a => p a
    ⊢ Eq ((Matrix.diagonal d).toBlock p p i j) (Matrix.diagonal (fun i => d ↑i) i j)
  -/
  by_cases h : i = j
    /-
      case pos
      m : Type u_2
      α : Type u_12
      inst✝¹ : DecidableEq m
      inst✝ : Zero α
      d : m → α
      p : m → Prop
      i j : Subtype fun a => p a
      h : Eq i j
      ⊢ Eq ((Matrix.diagonal d).toBlock p p i j) (Matrix.diagonal (fun i => d ↑i) i j)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      α : Type u_12
      inst✝¹ : DecidableEq m
      inst✝ : Zero α
      d : m → α
      p : m → Prop
      i j : Subtype fun a => p a
      h : Not (Eq i j)
      ⊢ Eq ((Matrix.diagonal d).toBlock p p i j) (Matrix.diagonal (fun i => d ↑i) i j)
    -/
  · simp [One.one, h, Subtype.val_injective.ne h]
    /-
      🎉 no goals
    -/


theorem toBlock_diagonal_disjoint (d : m → α) {p q : m → Prop} (hpq : Disjoint p q) :
    Matrix.toBlock (diagonal d) p q = 0 := by
  /-
    m : Type u_2
    α : Type u_12
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    p q : m → Prop
    hpq : Disjoint p q
    ⊢ Eq ((Matrix.diagonal d).toBlock p q) 0
  -/
  ext ⟨i, hi⟩ ⟨j, hj⟩
  /-
    case a.mk.mk
    m : Type u_2
    α : Type u_12
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    p q : m → Prop
    hpq : Disjoint p q
    i : m
    hi : p i
    j : m
    hj : q j
    ⊢ Eq ((Matrix.diagonal d).toBlock p q ⟨i, hi⟩ ⟨j, hj⟩) (0 ⟨i, hi⟩ ⟨j, hj⟩)
  -/
  have : i ≠ j := fun heq => hpq.le_bot i ⟨hi, heq.symm ▸ hj⟩
  /-
    case a.mk.mk
    m : Type u_2
    α : Type u_12
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d : m → α
    p q : m → Prop
    hpq : Disjoint p q
    i : m
    hi : p i
    j : m
    hj : q j
    this : Ne i j
    ⊢ Eq ((Matrix.diagonal d).toBlock p q ⟨i, hi⟩ ⟨j, hj⟩) (0 ⟨i, hi⟩ ⟨j, hj⟩)
  -/
  simp [diagonal_apply_ne d this]
  /-
    🎉 no goals
  -/


@[simp]
theorem fromBlocks_diagonal (d₁ : l → α) (d₂ : m → α) :
    fromBlocks (diagonal d₁) 0 0 (diagonal d₂) = diagonal (Sum.elim d₁ d₂) := by
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d₁ : l → α
    d₂ : m → α
    ⊢ Eq (Matrix.fromBlocks (Matrix.diagonal d₁) 0 0 (Matrix.diagonal d₂)) (Matrix …
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    d₁ : l → α
    d₂ : m → α
    i j : Sum l m
    ⊢ Eq (Matrix.fromBlocks (Matrix.diagonal d₁) 0 0 (Matrix.diagonal d₂) i j) (Ma …
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
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp [diagonal]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
lemma toBlocks₁₁_diagonal (v : l ⊕ m → α) :
    toBlocks₁₁ (diagonal v) = diagonal (fun i => v (Sum.inl i)) := by
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    ⊢ Eq (Matrix.diagonal v).toBlocks₁₁ (Matrix.diagonal fun i => v (Sum.inl i))
  -/
  unfold toBlocks₁₁
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    ⊢ Eq (Matrix.of fun i j => Matrix.diagonal v (Sum.inl i) (Sum.inl j)) (Matrix. …
  -/
  funext i j
  /-
    case h.h
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    i j : l
    ⊢ Eq (Matrix.of (fun i j => Matrix.diagonal v (Sum.inl i) (Sum.inl j)) i j) (M …
  -/
  simp only [ne_eq, Sum.inl.injEq, of_apply, diagonal_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma toBlocks₂₂_diagonal (v : l ⊕ m → α) :
    toBlocks₂₂ (diagonal v) = diagonal (fun i => v (Sum.inr i)) := by
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    ⊢ Eq (Matrix.diagonal v).toBlocks₂₂ (Matrix.diagonal fun i => v (Sum.inr i))
  -/
  unfold toBlocks₂₂
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    ⊢ Eq (Matrix.of fun i j => Matrix.diagonal v (Sum.inr i) (Sum.inr j)) (Matrix. …
  -/
  funext i j
  /-
    case h.h
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝² : DecidableEq l
    inst✝¹ : DecidableEq m
    inst✝ : Zero α
    v : Sum l m → α
    i j : m
    ⊢ Eq (Matrix.of (fun i j => Matrix.diagonal v (Sum.inr i) (Sum.inr j)) i j) (M …
  -/
  simp only [ne_eq, Sum.inr.injEq, of_apply, diagonal_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma toBlocks₁₂_diagonal (v : l ⊕ m → α) : toBlocks₁₂ (diagonal v) = 0 := rfl


@[simp]
lemma toBlocks₂₁_diagonal (v : l ⊕ m → α) : toBlocks₂₁ (diagonal v) = 0 := rfl


@[simp]
theorem fromBlocks_one : fromBlocks (1 : Matrix l l α) 0 0 (1 : Matrix m m α) = 1 := by
  /-
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝³ : DecidableEq l
    inst✝² : DecidableEq m
    inst✝¹ : Zero α
    inst✝ : One α
    ⊢ Eq (Matrix.fromBlocks 1 0 0 1) 1
  -/
  ext i j
  /-
    case a
    l : Type u_1
    m : Type u_2
    α : Type u_12
    inst✝³ : DecidableEq l
    inst✝² : DecidableEq m
    inst✝¹ : Zero α
    inst✝ : One α
    i j : Sum l m
    ⊢ Eq (Matrix.fromBlocks 1 0 0 1 i j) (1 i j)
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
  rcases i with ⟨⟩ <;> rcases j with ⟨⟩ <;> simp [one_apply]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem toBlock_one_self (p : m → Prop) : Matrix.toBlock (1 : Matrix m m α) p p = 1 :=
  toBlock_diagonal_self _ p


theorem toBlock_one_disjoint {p q : m → Prop} (hpq : Disjoint p q) :
    Matrix.toBlock (1 : Matrix m m α) p q = 0 :=
  toBlock_diagonal_disjoint _ hpq


/-- `Matrix.blockDiagonal M` turns a homogeneously-indexed collection of matrices
`M : o → Matrix m n α'` into an `m × o`-by-`n × o` block matrix which has the entries of `M` along
the diagonal and zero elsewhere.

See also `Matrix.blockDiagonal'` if the matrices may not have the same size everywhere.
-/
def blockDiagonal (M : o → Matrix m n α) : Matrix (m × o) (n × o) α :=
  of <| (fun ⟨i, k⟩ ⟨j, k'⟩ => if k = k' then M k i j else 0 : m × o → n × o → α)

-- TODO: set as an equation lemma for `blockDiagonal`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem blockDiagonal_apply' (M : o → Matrix m n α) (i k j k') :
    blockDiagonal M ⟨i, k⟩ ⟨j, k'⟩ = if k = k' then M k i j else 0 :=
  rfl


theorem blockDiagonal_apply (M : o → Matrix m n α) (ik jk) :
    blockDiagonal M ik jk = if ik.2 = jk.2 then M ik.2 ik.1 jk.1 else 0 := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    ik : Prod m o
    jk : Prod n o
    ⊢ Eq (Matrix.blockDiagonal M ik jk) (ite (Eq ik.2 jk.2) (M ik.2 ik.1 jk.1) 0)
  -/
  cases ik
  /-
    case mk
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    jk : Prod n o
    fst✝ : m
    snd✝ : o
    ⊢ Eq (Matrix.blockDiagonal M { fst := fst✝, snd := snd✝ } jk) (ite (Eq { fst : …
  -/
  cases jk
  /-
    case mk.mk
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    fst✝¹ : m
    snd✝¹ : o
    fst✝ : n
    snd✝ : o
    ⊢ Eq (Matrix.blockDiagonal M { fst := fst✝¹, snd := snd✝¹ } { fst := fst✝, snd …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal_apply_eq (M : o → Matrix m n α) (i j k) :
    blockDiagonal M (i, k) (j, k) = M k i j :=
  if_pos rfl


theorem blockDiagonal_apply_ne (M : o → Matrix m n α) (i j) {k k'} (h : k ≠ k') :
    blockDiagonal M (i, k) (j, k') = 0 :=
  if_neg h


theorem blockDiagonal_map (M : o → Matrix m n α) (f : α → β) (hf : f 0 = 0) :
    (blockDiagonal M).map f = blockDiagonal fun k => (M k).map f := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : o → Matrix m n α
    f : α → β
    hf : Eq (f 0) 0
    ⊢ Eq ((Matrix.blockDiagonal M).map f) (Matrix.blockDiagonal fun k => (M k).map …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : o → Matrix m n α
    f : α → β
    hf : Eq (f 0) 0
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq ((Matrix.blockDiagonal M).map f i✝ j✝) (Matrix.blockDiagonal (fun k => (M …
  -/
  simp only [map_apply, blockDiagonal_apply, eq_comm]
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : o → Matrix m n α
    f : α → β
    hf : Eq (f 0) 0
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (f (ite (Eq i✝.2 j✝.2) (M i✝.2 i✝.1 j✝.1) 0)) (ite (Eq i✝.2 j✝.2) (f (M i …
  -/
  rw [apply_ite f, hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal_transpose (M : o → Matrix m n α) :
    (blockDiagonal M)ᵀ = blockDiagonal fun k => (M k)ᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    ⊢ Eq (Matrix.blockDiagonal M).transpose (Matrix.blockDiagonal fun k => (M k).t …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    i✝ : Prod n o
    j✝ : Prod m o
    ⊢ Eq ((Matrix.blockDiagonal M).transpose i✝ j✝) (Matrix.blockDiagonal (fun k = …
  -/
  simp only [transpose_apply, blockDiagonal_apply, eq_comm]
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : o → Matrix m n α
    i✝ : Prod n o
    j✝ : Prod m o
    ⊢ Eq (ite (Eq i✝.2 j✝.2) (M i✝.2 j✝.1 i✝.1) 0) (ite (Eq i✝.2 j✝.2) (M j✝.2 j✝. …
  -/
  split_ifs with h
    /-
      case pos
      m : Type u_2
      n : Type u_3
      o : Type u_4
      α : Type u_12
      inst✝¹ : DecidableEq o
      inst✝ : Zero α
      M : o → Matrix m n α
      i✝ : Prod n o
      j✝ : Prod m o
      h : Eq i✝.2 j✝.2
      ⊢ Eq (M i✝.2 j✝.1 i✝.1) (M j✝.2 j✝.1 i✝.1)
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u_2
      n : Type u_3
      o : Type u_4
      α : Type u_12
      inst✝¹ : DecidableEq o
      inst✝ : Zero α
      M : o → Matrix m n α
      i✝ : Prod n o
      j✝ : Prod m o
      h : Not (Eq i✝.2 j✝.2)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem blockDiagonal_conjTranspose {α : Type*} [AddMonoid α] [StarAddMonoid α]
    (M : o → Matrix m n α) : (blockDiagonal M)ᴴ = blockDiagonal fun k => (M k)ᴴ := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    inst✝² : DecidableEq o
    α : Type u_14
    inst✝¹ : AddMonoid α
    inst✝ : StarAddMonoid α
    M : o → Matrix m n α
    ⊢ Eq (Matrix.blockDiagonal M).conjTranspose (Matrix.blockDiagonal fun k => (M  …
  -/
  simp only [conjTranspose, blockDiagonal_transpose]
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    inst✝² : DecidableEq o
    α : Type u_14
    inst✝¹ : AddMonoid α
    inst✝ : StarAddMonoid α
    M : o → Matrix m n α
    ⊢ Eq ((Matrix.blockDiagonal fun k => (M k).transpose).map Star.star) (Matrix.b …
  -/
  rw [blockDiagonal_map _ star (star_zero α)]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal_zero : blockDiagonal (0 : o → Matrix m n α) = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    ⊢ Eq (Matrix.blockDiagonal 0) 0
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (Matrix.blockDiagonal 0 i✝ j✝) (0 i✝ j✝)
  -/
  simp [blockDiagonal_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal_diagonal [DecidableEq m] (d : o → m → α) :
    (blockDiagonal fun k => diagonal (d k)) = diagonal fun ik => d ik.2 ik.1 := by
  /-
    m : Type u_2
    o : Type u_4
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : DecidableEq m
    d : o → m → α
    ⊢ Eq (Matrix.blockDiagonal fun k => Matrix.diagonal (d k)) (Matrix.diagonal fu …
  -/
  ext ⟨i, k⟩ ⟨j, k'⟩
  /-
    case a.mk.mk
    m : Type u_2
    o : Type u_4
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : DecidableEq m
    d : o → m → α
    i : m
    k : o
    j : m
    k' : o
    ⊢ Eq (Matrix.blockDiagonal (fun k => Matrix.diagonal (d k)) { fst := i, snd := …
  -/
  simp only [blockDiagonal_apply, diagonal_apply, Prod.mk.inj_iff, ← ite_and]
  /-
    case a.mk.mk
    m : Type u_2
    o : Type u_4
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : DecidableEq m
    d : o → m → α
    i : m
    k : o
    j : m
    k' : o
    ⊢ Eq (ite (And (Eq k k') (Eq i j)) (d k i) 0) (ite (And (Eq i j) (Eq k k')) (d …
  -/
  congr 1
  /-
    case a.mk.mk.e_c
    m : Type u_2
    o : Type u_4
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : DecidableEq m
    d : o → m → α
    i : m
    k : o
    j : m
    k' : o
    ⊢ Eq (And (Eq k k') (Eq i j)) (And (Eq i j) (Eq k k'))
  -/
  rw [and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal_one [DecidableEq m] [One α] : blockDiagonal (1 : o → Matrix m m α) = 1 :=
  show (blockDiagonal fun _ : o => diagonal fun _ : m => (1 : α)) = diagonal fun _ => 1 by
    /-
      m : Type u_2
      o : Type u_4
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : Zero α
      inst✝¹ : DecidableEq m
      inst✝ : One α
      ⊢ Eq (Matrix.blockDiagonal fun x => Matrix.diagonal fun x => 1) (Matrix.diagon …
    -/
    rw [blockDiagonal_diagonal]
    /-
      🎉 no goals
    -/


@[simp]
theorem blockDiagonal_add [AddZeroClass α] (M N : o → Matrix m n α) :
    blockDiagonal (M + N) = blockDiagonal M + blockDiagonal N := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : o → Matrix m n α
    ⊢ Eq (Matrix.blockDiagonal (HAdd.hAdd M N)) (HAdd.hAdd (Matrix.blockDiagonal M …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : o → Matrix m n α
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (Matrix.blockDiagonal (HAdd.hAdd M N) i✝ j✝) (HAdd.hAdd (Matrix.blockDiag …
  -/
  simp only [blockDiagonal_apply, Pi.add_apply, add_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : o → Matrix m n α
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (ite (Eq i✝.2 j✝.2) (HAdd.hAdd (M i✝.2 i✝.1 j✝.1) (N i✝.2 i✝.1 j✝.1)) 0)  …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


/-- `Matrix.blockDiagonal` as an `AddMonoidHom`. -/
@[simps]
def blockDiagonalAddMonoidHom [AddZeroClass α] :
    (o → Matrix m n α) →+ Matrix (m × o) (n × o) α where
  toFun := blockDiagonal
  map_zero' := blockDiagonal_zero
  map_add' := blockDiagonal_add


@[simp]
theorem blockDiagonal_neg [AddGroup α] (M : o → Matrix m n α) :
    blockDiagonal (-M) = -blockDiagonal M :=
  map_neg (blockDiagonalAddMonoidHom m n o α) M


@[simp]
theorem blockDiagonal_sub [AddGroup α] (M N : o → Matrix m n α) :
    blockDiagonal (M - N) = blockDiagonal M - blockDiagonal N :=
  map_sub (blockDiagonalAddMonoidHom m n o α) M N


@[simp]
theorem blockDiagonal_mul [Fintype n] [Fintype o] [NonUnitalNonAssocSemiring α]
    (M : o → Matrix m n α) (N : o → Matrix n p α) :
    (blockDiagonal fun k => M k * N k) = blockDiagonal M * blockDiagonal N := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : Fintype n
    inst✝¹ : Fintype o
    inst✝ : NonUnitalNonAssocSemiring α
    M : o → Matrix m n α
    N : o → Matrix n p α
    ⊢ Eq (Matrix.blockDiagonal fun k => HMul.hMul (M k) (N k)) (HMul.hMul (Matrix. …
  -/
  ext ⟨i, k⟩ ⟨j, k'⟩
  /-
    case a.mk.mk
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : Fintype n
    inst✝¹ : Fintype o
    inst✝ : NonUnitalNonAssocSemiring α
    M : o → Matrix m n α
    N : o → Matrix n p α
    i : m
    k : o
    j : p
    k' : o
    ⊢ Eq (Matrix.blockDiagonal (fun k => HMul.hMul (M k) (N k)) { fst := i, snd := …
  -/
  simp only [blockDiagonal_apply, mul_apply, ← Finset.univ_product_univ, Finset.sum_product]
  /-
    case a.mk.mk
    m : Type u_2
    n : Type u_3
    o : Type u_4
    p : Type u_5
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : Fintype n
    inst✝¹ : Fintype o
    inst✝ : NonUnitalNonAssocSemiring α
    M : o → Matrix m n α
    N : o → Matrix n p α
    i : m
    k : o
    j : p
    k' : o
    ⊢ Eq (ite (Eq k k') (Finset.univ.sum fun j_1 => HMul.hMul (M k i j_1) (N k j_1 …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


/-- `Matrix.blockDiagonal` as a `RingHom`. -/
@[simps]
def blockDiagonalRingHom [DecidableEq m] [Fintype o] [Fintype m] [NonAssocSemiring α] :
    (o → Matrix m m α) →+* Matrix (m × o) (m × o) α :=
  { blockDiagonalAddMonoidHom m m o α with
    toFun := blockDiagonal
    map_one' := blockDiagonal_one
    map_mul' := blockDiagonal_mul }


@[simp]
theorem blockDiagonal_pow [DecidableEq m] [Fintype o] [Fintype m] [Semiring α]
    (M : o → Matrix m m α) (n : ℕ) : blockDiagonal (M ^ n) = blockDiagonal M ^ n :=
  map_pow (blockDiagonalRingHom m o α) M n


@[simp]
theorem blockDiagonal_smul {R : Type*} [Monoid R] [AddMonoid α] [DistribMulAction R α] (x : R)
    (M : o → Matrix m n α) : blockDiagonal (x • M) = x • blockDiagonal M := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Monoid R
    inst✝¹ : AddMonoid α
    inst✝ : DistribMulAction R α
    x : R
    M : o → Matrix m n α
    ⊢ Eq (Matrix.blockDiagonal (HSMul.hSMul x M)) (HSMul.hSMul x (Matrix.blockDiag …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Monoid R
    inst✝¹ : AddMonoid α
    inst✝ : DistribMulAction R α
    x : R
    M : o → Matrix m n α
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (Matrix.blockDiagonal (HSMul.hSMul x M) i✝ j✝) (HSMul.hSMul x (Matrix.blo …
  -/
  simp only [blockDiagonal_apply, Pi.smul_apply, smul_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Monoid R
    inst✝¹ : AddMonoid α
    inst✝ : DistribMulAction R α
    x : R
    M : o → Matrix m n α
    i✝ : Prod m o
    j✝ : Prod n o
    ⊢ Eq (ite (Eq i✝.2 j✝.2) (HSMul.hSMul x (M i✝.2 i✝.1 j✝.1)) 0) (HSMul.hSMul x  …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


/-- Extract a block from the diagonal of a block diagonal matrix.

This is the block form of `Matrix.diag`, and the left-inverse of `Matrix.blockDiagonal`. -/
def blockDiag (M : Matrix (m × o) (n × o) α) (k : o) : Matrix m n α :=
  of fun i j => M (i, k) (j, k)

-- TODO: set as an equation lemma for `blockDiag`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem blockDiag_apply (M : Matrix (m × o) (n × o) α) (k : o) (i j) :
    blockDiag M k i j = M (i, k) (j, k) :=
  rfl


theorem blockDiag_map (M : Matrix (m × o) (n × o) α) (f : α → β) :
    blockDiag (M.map f) = fun k => (blockDiag M k).map f :=
  rfl


@[simp]
theorem blockDiag_transpose (M : Matrix (m × o) (n × o) α) (k : o) :
    blockDiag Mᵀ k = (blockDiag M k)ᵀ :=
  ext fun _ _ => rfl


@[simp]
theorem blockDiag_conjTranspose {α : Type*} [AddMonoid α] [StarAddMonoid α]
    (M : Matrix (m × o) (n × o) α) (k : o) : blockDiag Mᴴ k = (blockDiag M k)ᴴ :=
  ext fun _ _ => rfl


@[simp]
theorem blockDiag_zero : blockDiag (0 : Matrix (m × o) (n × o) α) = 0 :=
  rfl


@[simp]
theorem blockDiag_diagonal [DecidableEq o] [DecidableEq m] (d : m × o → α) (k : o) :
    blockDiag (diagonal d) k = diagonal fun i => d (i, k) :=
  ext fun i j => by
    /-
      m : Type u_2
      o : Type u_4
      α : Type u_12
      inst✝² : Zero α
      inst✝¹ : DecidableEq o
      inst✝ : DecidableEq m
      d : Prod m o → α
      k : o
      i j : m
      ⊢ Eq ((Matrix.diagonal d).blockDiag k i j) (Matrix.diagonal (fun i => d { fst  …
    -/
    obtain rfl | hij := Decidable.eq_or_ne i j
      /-
        case inl
        m : Type u_2
        o : Type u_4
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : DecidableEq m
        d : Prod m o → α
        k : o
        i : m
        ⊢ Eq ((Matrix.diagonal d).blockDiag k i i) (Matrix.diagonal (fun i => d { fst  …
      -/
    · rw [blockDiag_apply, diagonal_apply_eq, diagonal_apply_eq]
      /-
        🎉 no goals
      -/
      /-
        case inr
        m : Type u_2
        o : Type u_4
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : DecidableEq m
        d : Prod m o → α
        k : o
        i j : m
        hij : Ne i j
        ⊢ Eq ((Matrix.diagonal d).blockDiag k i j) (Matrix.diagonal (fun i => d { fst  …
      -/
    · rw [blockDiag_apply, diagonal_apply_ne _ hij, diagonal_apply_ne _ (mt _ hij)]
      /-
        m : Type u_2
        o : Type u_4
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : DecidableEq m
        d : Prod m o → α
        k : o
        i j : m
        hij : Ne i j
        ⊢ Eq { fst := i, snd := k } { fst := j, snd := k } → Eq i j
      -/
      exact Prod.fst_eq_iff.mpr
      /-
        🎉 no goals
      -/


@[simp]
theorem blockDiag_blockDiagonal [DecidableEq o] (M : o → Matrix m n α) :
    blockDiag (blockDiagonal M) = M :=
  funext fun _ => ext fun i j => blockDiagonal_apply_eq M i j _


theorem blockDiagonal_injective [DecidableEq o] :
    Function.Injective (blockDiagonal : (o → Matrix m n α) → Matrix _ _ α) :=
  Function.LeftInverse.injective blockDiag_blockDiagonal


@[simp]
theorem blockDiagonal_inj [DecidableEq o] {M N : o → Matrix m n α} :
    blockDiagonal M = blockDiagonal N ↔ M = N :=
  blockDiagonal_injective.eq_iff


@[simp]
theorem blockDiag_one [DecidableEq o] [DecidableEq m] [One α] :
    blockDiag (1 : Matrix (m × o) (m × o) α) = 1 :=
  funext <| blockDiag_diagonal _


@[simp]
theorem blockDiag_add [AddZeroClass α] (M N : Matrix (m × o) (n × o) α) :
    blockDiag (M + N) = blockDiag M + blockDiag N :=
  rfl


/-- `Matrix.blockDiag` as an `AddMonoidHom`. -/
@[simps]
def blockDiagAddMonoidHom [AddZeroClass α] : Matrix (m × o) (n × o) α →+ o → Matrix m n α where
  toFun := blockDiag
  map_zero' := blockDiag_zero
  map_add' := blockDiag_add


@[simp]
theorem blockDiag_neg [AddGroup α] (M : Matrix (m × o) (n × o) α) : blockDiag (-M) = -blockDiag M :=
  map_neg (blockDiagAddMonoidHom m n o α) M


@[simp]
theorem blockDiag_sub [AddGroup α] (M N : Matrix (m × o) (n × o) α) :
    blockDiag (M - N) = blockDiag M - blockDiag N :=
  map_sub (blockDiagAddMonoidHom m n o α) M N


@[simp]
theorem blockDiag_smul {R : Type*} [Monoid R] [AddMonoid α] [DistribMulAction R α] (x : R)
    (M : Matrix (m × o) (n × o) α) : blockDiag (x • M) = x • blockDiag M :=
  rfl


/-- `Matrix.blockDiagonal' M` turns `M : Π i, Matrix (m i) (n i) α` into a
`Σ i, m i`-by-`Σ i, n i` block matrix which has the entries of `M` along the diagonal
and zero elsewhere.

This is the dependently-typed version of `Matrix.blockDiagonal`. -/
def blockDiagonal' (M : ∀ i, Matrix (m' i) (n' i) α) : Matrix (Σi, m' i) (Σi, n' i) α :=
  of <|
    (fun ⟨k, i⟩ ⟨k', j⟩ => if h : k = k' then M k i (cast (congr_arg n' h.symm) j) else 0 :
      (Σi, m' i) → (Σi, n' i) → α)

-- TODO: set as an equation lemma for `blockDiagonal'`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem blockDiagonal'_apply' (M : ∀ i, Matrix (m' i) (n' i) α) (k i k' j) :
    blockDiagonal' M ⟨k, i⟩ ⟨k', j⟩ =
      if h : k = k' then M k i (cast (congr_arg n' h.symm) j) else 0 :=
  rfl


theorem blockDiagonal'_eq_blockDiagonal (M : o → Matrix m n α) {k k'} (i j) :
    blockDiagonal M (i, k) (j, k') = blockDiagonal' M ⟨k, i⟩ ⟨k', j⟩ :=
  rfl


theorem blockDiagonal'_submatrix_eq_blockDiagonal (M : o → Matrix m n α) :
    (blockDiagonal' M).submatrix (Prod.toSigma ∘ Prod.swap) (Prod.toSigma ∘ Prod.swap) =
      blockDiagonal M :=
  Matrix.ext fun ⟨_, _⟩ ⟨_, _⟩ => rfl


theorem blockDiagonal'_apply (M : ∀ i, Matrix (m' i) (n' i) α) (ik jk) :
    blockDiagonal' M ik jk =
      if h : ik.1 = jk.1 then M ik.1 ik.2 (cast (congr_arg n' h.symm) jk.2) else 0 := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    ik : Sigma fun i => m' i
    jk : Sigma fun i => n' i
    ⊢ Eq (Matrix.blockDiagonal' M ik jk) (dite (Eq ik.fst jk.fst) (fun h => M ik.f …
  -/
  cases ik
  /-
    case mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    jk : Sigma fun i => n' i
    fst✝ : o
    snd✝ : m' fst✝
    ⊢ Eq (Matrix.blockDiagonal' M ⟨fst✝, snd✝⟩ jk) (dite (Eq ⟨fst✝, snd✝⟩.fst jk.f …
  -/
  cases jk
  /-
    case mk.mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    fst✝¹ : o
    snd✝¹ : m' fst✝¹
    fst✝ : o
    snd✝ : n' fst✝
    ⊢ Eq (Matrix.blockDiagonal' M ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩) (dite (Eq ⟨fst✝¹, s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal'_apply_eq (M : ∀ i, Matrix (m' i) (n' i) α) (k i j) :
    blockDiagonal' M ⟨k, i⟩ ⟨k, j⟩ = M k i j :=
  dif_pos rfl


theorem blockDiagonal'_apply_ne (M : ∀ i, Matrix (m' i) (n' i) α) {k k'} (i j) (h : k ≠ k') :
    blockDiagonal' M ⟨k, i⟩ ⟨k', j⟩ = 0 :=
  dif_neg h


theorem blockDiagonal'_map (M : ∀ i, Matrix (m' i) (n' i) α) (f : α → β) (hf : f 0 = 0) :
    (blockDiagonal' M).map f = blockDiagonal' fun k => (M k).map f := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : (i : o) → Matrix (m' i) (n' i) α
    f : α → β
    hf : Eq (f 0) 0
    ⊢ Eq ((Matrix.blockDiagonal' M).map f) (Matrix.blockDiagonal' fun k => (M k).m …
  -/
  ext
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : (i : o) → Matrix (m' i) (n' i) α
    f : α → β
    hf : Eq (f 0) 0
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq ((Matrix.blockDiagonal' M).map f i✝ j✝) (Matrix.blockDiagonal' (fun k =>  …
  -/
  simp only [map_apply, blockDiagonal'_apply, eq_comm]
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    β : Type u_13
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : Zero β
    M : (i : o) → Matrix (m' i) (n' i) α
    f : α → β
    hf : Eq (f 0) 0
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (f (dite (Eq i✝.fst j✝.fst) (fun h => M i✝.fst i✝.snd (cast ⋯ j✝.snd)) fu …
  -/
  rw [apply_dite f, hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal'_transpose (M : ∀ i, Matrix (m' i) (n' i) α) :
    (blockDiagonal' M)ᵀ = blockDiagonal' fun k => (M k)ᵀ := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    ⊢ Eq (Matrix.blockDiagonal' M).transpose (Matrix.blockDiagonal' fun k => (M k) …
  -/
  ext ⟨ii, ix⟩ ⟨ji, jx⟩
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    ii : o
    ix : n' ii
    ji : o
    jx : m' ji
    ⊢ Eq ((Matrix.blockDiagonal' M).transpose ⟨ii, ix⟩ ⟨ji, jx⟩) (Matrix.blockDiag …
  -/
  simp only [transpose_apply, blockDiagonal'_apply]
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    M : (i : o) → Matrix (m' i) (n' i) α
    ii : o
    ix : n' ii
    ji : o
    jx : m' ji
    ⊢ Eq (dite (Eq ji ii) (fun h => M ji jx (cast ⋯ ix)) fun h => 0) (dite (Eq ii  …
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
  split_ifs <;> cc
                /-
                  🎉 no goals
                -/


@[simp]
theorem blockDiagonal'_conjTranspose {α} [AddMonoid α] [StarAddMonoid α]
    (M : ∀ i, Matrix (m' i) (n' i) α) : (blockDiagonal' M)ᴴ = blockDiagonal' fun k => (M k)ᴴ := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    inst✝² : DecidableEq o
    α : Type u_14
    inst✝¹ : AddMonoid α
    inst✝ : StarAddMonoid α
    M : (i : o) → Matrix (m' i) (n' i) α
    ⊢ Eq (Matrix.blockDiagonal' M).conjTranspose (Matrix.blockDiagonal' fun k => ( …
  -/
  simp only [conjTranspose, blockDiagonal'_transpose]
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    inst✝² : DecidableEq o
    α : Type u_14
    inst✝¹ : AddMonoid α
    inst✝ : StarAddMonoid α
    M : (i : o) → Matrix (m' i) (n' i) α
    ⊢ Eq ((Matrix.blockDiagonal' fun k => (M k).transpose).map Star.star) (Matrix. …
  -/
  exact blockDiagonal'_map _ star (star_zero α)
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal'_zero : blockDiagonal' (0 : ∀ i, Matrix (m' i) (n' i) α) = 0 := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    ⊢ Eq (Matrix.blockDiagonal' 0) 0
  -/
  ext
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : Zero α
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (Matrix.blockDiagonal' 0 i✝ j✝) (0 i✝ j✝)
  -/
  simp [blockDiagonal'_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem blockDiagonal'_diagonal [∀ i, DecidableEq (m' i)] (d : ∀ i, m' i → α) :
    (blockDiagonal' fun k => diagonal (d k)) = diagonal fun ik => d ik.1 ik.2 := by
  /-
    o : Type u_4
    m' : o → Type u_7
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : (i : o) → DecidableEq (m' i)
    d : (i : o) → m' i → α
    ⊢ Eq (Matrix.blockDiagonal' fun k => Matrix.diagonal (d k)) (Matrix.diagonal f …
  -/
  ext ⟨i, k⟩ ⟨j, k'⟩
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : (i : o) → DecidableEq (m' i)
    d : (i : o) → m' i → α
    i : o
    k : m' i
    j : o
    k' : m' j
    ⊢ Eq (Matrix.blockDiagonal' (fun k => Matrix.diagonal (d k)) ⟨i, k⟩ ⟨j, k'⟩) ( …
  -/
  simp only [blockDiagonal'_apply, diagonal]
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    α : Type u_12
    inst✝² : DecidableEq o
    inst✝¹ : Zero α
    inst✝ : (i : o) → DecidableEq (m' i)
    d : (i : o) → m' i → α
    i : o
    k : m' i
    j : o
    k' : m' j
    ⊢ Eq (dite (Eq i j) (fun h => Matrix.of (fun i_1 j => ite (Eq i_1 j) (d i i_1) …
  -/
  obtain rfl | hij := Decidable.eq_or_ne i j
    /-
      case a.mk.mk.inl
      o : Type u_4
      m' : o → Type u_7
      α : Type u_12
      inst✝² : DecidableEq o
      inst✝¹ : Zero α
      inst✝ : (i : o) → DecidableEq (m' i)
      d : (i : o) → m' i → α
      i : o
      k k' : m' i
      ⊢ Eq (dite (Eq i i) (fun h => Matrix.of (fun i_1 j => ite (Eq i_1 j) (d i i_1) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.mk.mk.inr
      o : Type u_4
      m' : o → Type u_7
      α : Type u_12
      inst✝² : DecidableEq o
      inst✝¹ : Zero α
      inst✝ : (i : o) → DecidableEq (m' i)
      d : (i : o) → m' i → α
      i : o
      k : m' i
      j : o
      k' : m' j
      hij : Ne i j
      ⊢ Eq (dite (Eq i j) (fun h => Matrix.of (fun i_1 j => ite (Eq i_1 j) (d i i_1) …
    -/
  · simp [hij]
    /-
      🎉 no goals
    -/


@[simp]
theorem blockDiagonal'_one [∀ i, DecidableEq (m' i)] [One α] :
    blockDiagonal' (1 : ∀ i, Matrix (m' i) (m' i) α) = 1 :=
  show (blockDiagonal' fun i : o => diagonal fun _ : m' i => (1 : α)) = diagonal fun _ => 1 by
    /-
      o : Type u_4
      m' : o → Type u_7
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : Zero α
      inst✝¹ : (i : o) → DecidableEq (m' i)
      inst✝ : One α
      ⊢ Eq (Matrix.blockDiagonal' fun i => Matrix.diagonal fun x => 1) (Matrix.diago …
    -/
    rw [blockDiagonal'_diagonal]
    /-
      🎉 no goals
    -/


@[simp]
theorem blockDiagonal'_add [AddZeroClass α] (M N : ∀ i, Matrix (m' i) (n' i) α) :
    blockDiagonal' (M + N) = blockDiagonal' M + blockDiagonal' N := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : (i : o) → Matrix (m' i) (n' i) α
    ⊢ Eq (Matrix.blockDiagonal' (HAdd.hAdd M N)) (HAdd.hAdd (Matrix.blockDiagonal' …
  -/
  ext
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : (i : o) → Matrix (m' i) (n' i) α
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (Matrix.blockDiagonal' (HAdd.hAdd M N) i✝ j✝) (HAdd.hAdd (Matrix.blockDia …
  -/
  simp only [blockDiagonal'_apply, Pi.add_apply, add_apply]
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝¹ : DecidableEq o
    inst✝ : AddZeroClass α
    M N : (i : o) → Matrix (m' i) (n' i) α
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (dite (Eq i✝.fst j✝.fst) (fun h => HAdd.hAdd (M i✝.fst i✝.snd (cast ⋯ j✝. …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


/-- `Matrix.blockDiagonal'` as an `AddMonoidHom`. -/
@[simps]
def blockDiagonal'AddMonoidHom [AddZeroClass α] :
    (∀ i, Matrix (m' i) (n' i) α) →+ Matrix (Σi, m' i) (Σi, n' i) α where
  toFun := blockDiagonal'
  map_zero' := blockDiagonal'_zero
  map_add' := blockDiagonal'_add


@[simp]
theorem blockDiagonal'_neg [AddGroup α] (M : ∀ i, Matrix (m' i) (n' i) α) :
    blockDiagonal' (-M) = -blockDiagonal' M :=
  map_neg (blockDiagonal'AddMonoidHom m' n' α) M


@[simp]
theorem blockDiagonal'_sub [AddGroup α] (M N : ∀ i, Matrix (m' i) (n' i) α) :
    blockDiagonal' (M - N) = blockDiagonal' M - blockDiagonal' N :=
  map_sub (blockDiagonal'AddMonoidHom m' n' α) M N


@[simp]
theorem blockDiagonal'_mul [NonUnitalNonAssocSemiring α] [∀ i, Fintype (n' i)] [Fintype o]
    (M : ∀ i, Matrix (m' i) (n' i) α) (N : ∀ i, Matrix (n' i) (p' i) α) :
    (blockDiagonal' fun k => M k * N k) = blockDiagonal' M * blockDiagonal' N := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    p' : o → Type u_9
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : (i : o) → Fintype (n' i)
    inst✝ : Fintype o
    M : (i : o) → Matrix (m' i) (n' i) α
    N : (i : o) → Matrix (n' i) (p' i) α
    ⊢ Eq (Matrix.blockDiagonal' fun k => HMul.hMul (M k) (N k)) (HMul.hMul (Matrix …
  -/
  ext ⟨k, i⟩ ⟨k', j⟩
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    p' : o → Type u_9
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : (i : o) → Fintype (n' i)
    inst✝ : Fintype o
    M : (i : o) → Matrix (m' i) (n' i) α
    N : (i : o) → Matrix (n' i) (p' i) α
    k : o
    i : m' k
    k' : o
    j : p' k'
    ⊢ Eq (Matrix.blockDiagonal' (fun k => HMul.hMul (M k) (N k)) ⟨k, i⟩ ⟨k', j⟩) ( …
  -/
  simp only [blockDiagonal'_apply, mul_apply, ← Finset.univ_sigma_univ, Finset.sum_sigma]
  /-
    case a.mk.mk
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    p' : o → Type u_9
    α : Type u_12
    inst✝³ : DecidableEq o
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : (i : o) → Fintype (n' i)
    inst✝ : Fintype o
    M : (i : o) → Matrix (m' i) (n' i) α
    N : (i : o) → Matrix (n' i) (p' i) α
    k : o
    i : m' k
    k' : o
    j : p' k'
    ⊢ Eq (dite (Eq k k') (fun h => Finset.univ.sum fun j_1 => HMul.hMul (M k i j_1 …
  -/
  rw [Fintype.sum_eq_single k]
    /-
      case a.mk.mk
      o : Type u_4
      m' : o → Type u_7
      n' : o → Type u_8
      p' : o → Type u_9
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : NonUnitalNonAssocSemiring α
      inst✝¹ : (i : o) → Fintype (n' i)
      inst✝ : Fintype o
      M : (i : o) → Matrix (m' i) (n' i) α
      N : (i : o) → Matrix (n' i) (p' i) α
      k : o
      i : m' k
      k' : o
      j : p' k'
      ⊢ Eq (dite (Eq k k') (fun h => Finset.univ.sum fun j_1 => HMul.hMul (M k i j_1 …
    -/
  · simp only [if_pos, dif_pos] -- Porting note: added
    /-
      case a.mk.mk
      o : Type u_4
      m' : o → Type u_7
      n' : o → Type u_8
      p' : o → Type u_9
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : NonUnitalNonAssocSemiring α
      inst✝¹ : (i : o) → Fintype (n' i)
      inst✝ : Fintype o
      M : (i : o) → Matrix (m' i) (n' i) α
      N : (i : o) → Matrix (n' i) (p' i) α
      k : o
      i : m' k
      k' : o
      j : p' k'
      ⊢ Eq (dite (Eq k k') (fun h => Finset.univ.sum fun j_1 => HMul.hMul (M k i j_1 …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp
                  /-
                    🎉 no goals
                  -/
    /-
      case a.mk.mk
      o : Type u_4
      m' : o → Type u_7
      n' : o → Type u_8
      p' : o → Type u_9
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : NonUnitalNonAssocSemiring α
      inst✝¹ : (i : o) → Fintype (n' i)
      inst✝ : Fintype o
      M : (i : o) → Matrix (m' i) (n' i) α
      N : (i : o) → Matrix (n' i) (p' i) α
      k : o
      i : m' k
      k' : o
      j : p' k'
      ⊢ ∀ (x : o), Ne x k → Eq (Finset.univ.sum fun x_1 => HMul.hMul (dite (Eq k x)  …
    -/
  · intro j' hj'
    /-
      case a.mk.mk
      o : Type u_4
      m' : o → Type u_7
      n' : o → Type u_8
      p' : o → Type u_9
      α : Type u_12
      inst✝³ : DecidableEq o
      inst✝² : NonUnitalNonAssocSemiring α
      inst✝¹ : (i : o) → Fintype (n' i)
      inst✝ : Fintype o
      M : (i : o) → Matrix (m' i) (n' i) α
      N : (i : o) → Matrix (n' i) (p' i) α
      k : o
      i : m' k
      k' : o
      j : p' k'
      j' : o
      hj' : Ne j' k
      ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (dite (Eq k j') (fun h => M k i (cast …
    -/
    exact Finset.sum_eq_zero fun _ _ => by rw [dif_neg hj'.symm, zero_mul]
    /-
      🎉 no goals
    -/


/-- `Matrix.blockDiagonal'` as a `RingHom`. -/
@[simps]
def blockDiagonal'RingHom [∀ i, DecidableEq (m' i)] [Fintype o] [∀ i, Fintype (m' i)]
    [NonAssocSemiring α] : (∀ i, Matrix (m' i) (m' i) α) →+* Matrix (Σi, m' i) (Σi, m' i) α :=
  { blockDiagonal'AddMonoidHom m' m' α with
    toFun := blockDiagonal'
    map_one' := blockDiagonal'_one
    map_mul' := blockDiagonal'_mul }


@[simp]
theorem blockDiagonal'_pow [∀ i, DecidableEq (m' i)] [Fintype o] [∀ i, Fintype (m' i)] [Semiring α]
    (M : ∀ i, Matrix (m' i) (m' i) α) (n : ℕ) : blockDiagonal' (M ^ n) = blockDiagonal' M ^ n :=
  map_pow (blockDiagonal'RingHom m' α) M n


@[simp]
theorem blockDiagonal'_smul {R : Type*} [Semiring R] [AddCommMonoid α] [Module R α] (x : R)
    (M : ∀ i, Matrix (m' i) (n' i) α) : blockDiagonal' (x • M) = x • blockDiagonal' M := by
  /-
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid α
    inst✝ : Module R α
    x : R
    M : (i : o) → Matrix (m' i) (n' i) α
    ⊢ Eq (Matrix.blockDiagonal' (HSMul.hSMul x M)) (HSMul.hSMul x (Matrix.blockDia …
  -/
  ext
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid α
    inst✝ : Module R α
    x : R
    M : (i : o) → Matrix (m' i) (n' i) α
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (Matrix.blockDiagonal' (HSMul.hSMul x M) i✝ j✝) (HSMul.hSMul x (Matrix.bl …
  -/
  simp only [blockDiagonal'_apply, Pi.smul_apply, smul_apply]
  /-
    case a
    o : Type u_4
    m' : o → Type u_7
    n' : o → Type u_8
    α : Type u_12
    inst✝³ : DecidableEq o
    R : Type u_14
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid α
    inst✝ : Module R α
    x : R
    M : (i : o) → Matrix (m' i) (n' i) α
    i✝ : Sigma fun i => m' i
    j✝ : Sigma fun i => n' i
    ⊢ Eq (dite (Eq i✝.fst j✝.fst) (fun h => HSMul.hSMul x (M i✝.fst i✝.snd (cast ⋯ …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


/-- Extract a block from the diagonal of a block diagonal matrix.

This is the block form of `Matrix.diag`, and the left-inverse of `Matrix.blockDiagonal'`. -/
def blockDiag' (M : Matrix (Σi, m' i) (Σi, n' i) α) (k : o) : Matrix (m' k) (n' k) α :=
  of fun i j => M ⟨k, i⟩ ⟨k, j⟩

-- TODO: set as an equation lemma for `blockDiag'`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem blockDiag'_apply (M : Matrix (Σi, m' i) (Σi, n' i) α) (k : o) (i j) :
    blockDiag' M k i j = M ⟨k, i⟩ ⟨k, j⟩ :=
  rfl


theorem blockDiag'_map (M : Matrix (Σi, m' i) (Σi, n' i) α) (f : α → β) :
    blockDiag' (M.map f) = fun k => (blockDiag' M k).map f :=
  rfl


@[simp]
theorem blockDiag'_transpose (M : Matrix (Σi, m' i) (Σi, n' i) α) (k : o) :
    blockDiag' Mᵀ k = (blockDiag' M k)ᵀ :=
  ext fun _ _ => rfl


@[simp]
theorem blockDiag'_conjTranspose {α : Type*} [AddMonoid α] [StarAddMonoid α]
    (M : Matrix (Σi, m' i) (Σi, n' i) α) (k : o) : blockDiag' Mᴴ k = (blockDiag' M k)ᴴ :=
  ext fun _ _ => rfl


@[simp]
theorem blockDiag'_zero : blockDiag' (0 : Matrix (Σi, m' i) (Σi, n' i) α) = 0 :=
  rfl


@[simp]
theorem blockDiag'_diagonal [DecidableEq o] [∀ i, DecidableEq (m' i)] (d : (Σi, m' i) → α) (k : o) :
    blockDiag' (diagonal d) k = diagonal fun i => d ⟨k, i⟩ :=
  ext fun i j => by
    /-
      o : Type u_4
      m' : o → Type u_7
      α : Type u_12
      inst✝² : Zero α
      inst✝¹ : DecidableEq o
      inst✝ : (i : o) → DecidableEq (m' i)
      d : (Sigma fun i => m' i) → α
      k : o
      i j : m' k
      ⊢ Eq ((Matrix.diagonal d).blockDiag' k i j) (Matrix.diagonal (fun i => d ⟨k, i …
    -/
    obtain rfl | hij := Decidable.eq_or_ne i j
      /-
        case inl
        o : Type u_4
        m' : o → Type u_7
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : (i : o) → DecidableEq (m' i)
        d : (Sigma fun i => m' i) → α
        k : o
        i : m' k
        ⊢ Eq ((Matrix.diagonal d).blockDiag' k i i) (Matrix.diagonal (fun i => d ⟨k, i …
      -/
    · rw [blockDiag'_apply, diagonal_apply_eq, diagonal_apply_eq]
      /-
        🎉 no goals
      -/
      /-
        case inr
        o : Type u_4
        m' : o → Type u_7
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : (i : o) → DecidableEq (m' i)
        d : (Sigma fun i => m' i) → α
        k : o
        i j : m' k
        hij : Ne i j
        ⊢ Eq ((Matrix.diagonal d).blockDiag' k i j) (Matrix.diagonal (fun i => d ⟨k, i …
      -/
    · rw [blockDiag'_apply, diagonal_apply_ne _ hij, diagonal_apply_ne _ (mt (fun h => ?_) hij)]
      /-
        o : Type u_4
        m' : o → Type u_7
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : (i : o) → DecidableEq (m' i)
        d : (Sigma fun i => m' i) → α
        k : o
        i j : m' k
        hij : Ne i j
        h : Eq ⟨k, i⟩ ⟨k, j⟩
        ⊢ Eq i j
      -/
      cases h
      /-
        case refl
        o : Type u_4
        m' : o → Type u_7
        α : Type u_12
        inst✝² : Zero α
        inst✝¹ : DecidableEq o
        inst✝ : (i : o) → DecidableEq (m' i)
        d : (Sigma fun i => m' i) → α
        k : o
        i : m' k
        hij : Ne i i
        ⊢ Eq i i
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem blockDiag'_blockDiagonal' [DecidableEq o] (M : ∀ i, Matrix (m' i) (n' i) α) :
    blockDiag' (blockDiagonal' M) = M :=
  funext fun _ => ext fun _ _ => blockDiagonal'_apply_eq M _ _ _


theorem blockDiagonal'_injective [DecidableEq o] :
    Function.Injective (blockDiagonal' : (∀ i, Matrix (m' i) (n' i) α) → Matrix _ _ α) :=
  Function.LeftInverse.injective blockDiag'_blockDiagonal'


@[simp]
theorem blockDiagonal'_inj [DecidableEq o] {M N : ∀ i, Matrix (m' i) (n' i) α} :
    blockDiagonal' M = blockDiagonal' N ↔ M = N :=
  blockDiagonal'_injective.eq_iff


@[simp]
theorem blockDiag'_one [DecidableEq o] [∀ i, DecidableEq (m' i)] [One α] :
    blockDiag' (1 : Matrix (Σi, m' i) (Σi, m' i) α) = 1 :=
  funext <| blockDiag'_diagonal _


@[simp]
theorem blockDiag'_add [AddZeroClass α] (M N : Matrix (Σi, m' i) (Σi, n' i) α) :
    blockDiag' (M + N) = blockDiag' M + blockDiag' N :=
  rfl


/-- `Matrix.blockDiag'` as an `AddMonoidHom`. -/
@[simps]
def blockDiag'AddMonoidHom [AddZeroClass α] :
    Matrix (Σi, m' i) (Σi, n' i) α →+ ∀ i, Matrix (m' i) (n' i) α where
  toFun := blockDiag'
  map_zero' := blockDiag'_zero
  map_add' := blockDiag'_add


@[simp]
theorem blockDiag'_neg [AddGroup α] (M : Matrix (Σi, m' i) (Σi, n' i) α) :
    blockDiag' (-M) = -blockDiag' M :=
  map_neg (blockDiag'AddMonoidHom m' n' α) M


@[simp]
theorem blockDiag'_sub [AddGroup α] (M N : Matrix (Σi, m' i) (Σi, n' i) α) :
    blockDiag' (M - N) = blockDiag' M - blockDiag' N :=
  map_sub (blockDiag'AddMonoidHom m' n' α) M N


@[simp]
theorem blockDiag'_smul {R : Type*} [Monoid R] [AddMonoid α] [DistribMulAction R α] (x : R)
    (M : Matrix (Σi, m' i) (Σi, n' i) α) : blockDiag' (x • M) = x • blockDiag' M :=
  rfl


theorem toBlock_mul_eq_mul {m n k : Type*} [Fintype n] (p : m → Prop) (q : k → Prop)
    (A : Matrix m n R) (B : Matrix n k R) :
    (A * B).toBlock p q = A.toBlock p ⊤ * B.toBlock ⊤ q := by
  /-
    R : Type u_10
    inst✝¹ : CommRing R
    m : Type u_14
    n : Type u_15
    k : Type u_16
    inst✝ : Fintype n
    p : m → Prop
    q : k → Prop
    A : Matrix m n R
    B : Matrix n k R
    ⊢ Eq ((HMul.hMul A B).toBlock p q) (HMul.hMul (A.toBlock p Top.top) (B.toBlock …
  -/
  ext i k
  /-
    case a
    R : Type u_10
    inst✝¹ : CommRing R
    m : Type u_14
    n : Type u_15
    k✝ : Type u_16
    inst✝ : Fintype n
    p : m → Prop
    q : k✝ → Prop
    A : Matrix m n R
    B : Matrix n k✝ R
    i : Subtype fun a => p a
    k : Subtype fun a => q a
    ⊢ Eq ((HMul.hMul A B).toBlock p q i k) (HMul.hMul (A.toBlock p Top.top) (B.toB …
  -/
  simp only [toBlock_apply, mul_apply]
  /-
    case a
    R : Type u_10
    inst✝¹ : CommRing R
    m : Type u_14
    n : Type u_15
    k✝ : Type u_16
    inst✝ : Fintype n
    p : m → Prop
    q : k✝ → Prop
    A : Matrix m n R
    B : Matrix n k✝ R
    i : Subtype fun a => p a
    k : Subtype fun a => q a
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (A (↑i) j) (B j ↑k)) (Finset.univ.sum …
  -/
  rw [Finset.sum_subtype]
  /-
    case a.h
    R : Type u_10
    inst✝¹ : CommRing R
    m : Type u_14
    n : Type u_15
    k✝ : Type u_16
    inst✝ : Fintype n
    p : m → Prop
    q : k✝ → Prop
    A : Matrix m n R
    B : Matrix n k✝ R
    i : Subtype fun a => p a
    k : Subtype fun a => q a
    ⊢ ∀ (x : n), Iff (Membership.mem Finset.univ x) (Top.top x)
  -/
  simp [Pi.top_apply, Prop.top_eq_true]
  /-
    🎉 no goals
  -/


theorem toBlock_mul_eq_add {m n k : Type*} [Fintype n] (p : m → Prop) (q : n → Prop)
    [DecidablePred q] (r : k → Prop) (A : Matrix m n R) (B : Matrix n k R) : (A * B).toBlock p r =
    A.toBlock p q * B.toBlock q r + (A.toBlock p fun i => ¬q i) * B.toBlock (fun i => ¬q i) r := by
  classical
    ext i k
    simp only [toBlock_apply, mul_apply, Pi.add_apply]
    exact (Fintype.sum_subtype_add_sum_subtype q fun x => A (↑i) x * B x ↑k).symm


lemma Matrix.map_toSquareBlock
    (f : α → β) {M : Matrix m m α} {ι} {b : m → ι} {i : ι} :
    (M.map f).toSquareBlock b i = (M.toSquareBlock b i).map f :=
  submatrix_map _ _ _ _


lemma Matrix.comp_toSquareBlock {b : m → α}
    (M : Matrix m m (Matrix n n R)) (a : α) :
    letI equiv := Equiv.prodSubtypeFstEquivSubtypeProd.symm
    (M.comp m m n n R).toSquareBlock (fun i ↦ b i.1) a =
      ((M.toSquareBlock b a).comp _ _ n n R).reindex equiv equiv :=
  rfl


lemma Matrix.comp_diagonal (d) :
    comp m m n n R (diagonal d) =
      (blockDiagonal d).reindex (.prodComm ..) (.prodComm ..) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_14
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    d : m → Matrix n n R
    ⊢ Eq ((Matrix.comp m m n n R) (Matrix.diagonal d)) ((Matrix.reindex (Equiv.pro …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    R : Type u_14
    inst✝¹ : Zero R
    inst✝ : DecidableEq m
    d : m → Matrix n n R
    i✝ j✝ : Prod m n
    ⊢ Eq ((Matrix.comp m m n n R) (Matrix.diagonal d) i✝ j✝) ((Matrix.reindex (Equ …
  -/
  simp [diagonal, blockDiagonal, Matrix.ite_apply]
  /-
    🎉 no goals
  -/


