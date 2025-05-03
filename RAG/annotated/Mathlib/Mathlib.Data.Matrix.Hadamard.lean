/-- `Matrix.hadamard` defines the Hadamard product,
    which is the pointwise product of two matrices of the same size. -/
def hadamard [Mul α] (A : Matrix m n α) (B : Matrix m n α) : Matrix m n α :=
  of fun i j => A i j * B i j

-- TODO: set as an equation lemma for `hadamard`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem hadamard_apply [Mul α] (A : Matrix m n α) (B : Matrix m n α) (i j) :
    hadamard A B i j = A i j * B i j :=
  rfl


scoped infixl:100 " ⊙ " => Matrix.hadamard


theorem hadamard_comm [CommSemigroup α] : A ⊙ B = B ⊙ A :=
  ext fun _ _ => mul_comm _ _

-- associativity

theorem hadamard_assoc [Semigroup α] : A ⊙ B ⊙ C = A ⊙ (B ⊙ C) :=
  ext fun _ _ => mul_assoc _ _ _

-- distributivity

theorem hadamard_add [Distrib α] : A ⊙ (B + C) = A ⊙ B + A ⊙ C :=
  ext fun _ _ => left_distrib _ _ _


theorem add_hadamard [Distrib α] : (B + C) ⊙ A = B ⊙ A + C ⊙ A :=
  ext fun _ _ => right_distrib _ _ _

-- scalar multiplication

@[simp]
theorem smul_hadamard [Mul α] [SMul R α] [IsScalarTower R α α] (k : R) : (k • A) ⊙ B = k • A ⊙ B :=
  ext fun _ _ => smul_mul_assoc _ _ _


@[simp]
theorem hadamard_smul [Mul α] [SMul R α] [SMulCommClass R α α] (k : R) : A ⊙ (k • B) = k • A ⊙ B :=
  ext fun _ _ => mul_smul_comm _ _ _


@[simp]
theorem hadamard_zero : A ⊙ (0 : Matrix m n α) = 0 :=
  ext fun _ _ => mul_zero _


@[simp]
theorem zero_hadamard : (0 : Matrix m n α) ⊙ A = 0 :=
  ext fun _ _ => zero_mul _


theorem hadamard_one : M ⊙ (1 : Matrix n n α) = diagonal fun i => M i i := by
  /-
    α : Type u_1
    n : Type u_3
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    M : Matrix n n α
    ⊢ Eq (M.hadamard 1) (Matrix.diagonal fun i => M i i)
  -/
  ext i j
  /-
    case a
    α : Type u_1
    n : Type u_3
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    M : Matrix n n α
    i j : n
    ⊢ Eq (M.hadamard 1 i j) (Matrix.diagonal (fun i => M i i) i j)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> simp [h]
                         /-
                           🎉 no goals
                         -/


theorem one_hadamard : (1 : Matrix n n α) ⊙ M = diagonal fun i => M i i := by
  /-
    α : Type u_1
    n : Type u_3
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    M : Matrix n n α
    ⊢ Eq (Matrix.hadamard 1 M) (Matrix.diagonal fun i => M i i)
  -/
  ext i j
  /-
    case a
    α : Type u_1
    n : Type u_3
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    M : Matrix n n α
    i j : n
    ⊢ Eq (Matrix.hadamard 1 M i j) (Matrix.diagonal (fun i => M i i) i j)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> simp [h]
                         /-
                           🎉 no goals
                         -/


theorem diagonal_hadamard_diagonal (v : n → α) (w : n → α) :
    diagonal v ⊙ diagonal w = diagonal (v * w) :=
  ext fun _ _ => (apply_ite₂ _ _ _ _ _ _).trans (congr_arg _ <| zero_mul 0)


theorem sum_hadamard_eq : (∑ i : m, ∑ j : n, (A ⊙ B) i j) = trace (A * Bᵀ) :=
  rfl


theorem dotProduct_vecMul_hadamard [DecidableEq m] [DecidableEq n] (v : m → α) (w : n → α) :
    dotProduct (v ᵥ* (A ⊙ B)) w = trace (diagonal v * A * (B * diagonal w)ᵀ) := by
  /-
    α : Type u_1
    m : Type u_2
    n : Type u_3
    A B : Matrix m n α
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : Semiring α
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    v : m → α
    w : n → α
    ⊢ Eq (dotProduct (Matrix.vecMul v (A.hadamard B)) w) (HMul.hMul (HMul.hMul (Ma …
  -/
  rw [← sum_hadamard_eq, Finset.sum_comm]
  /-
    α : Type u_1
    m : Type u_2
    n : Type u_3
    A B : Matrix m n α
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : Semiring α
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    v : m → α
    w : n → α
    ⊢ Eq (dotProduct (Matrix.vecMul v (A.hadamard B)) w) (Finset.univ.sum fun y => …
  -/
  simp [dotProduct, vecMul, Finset.sum_mul, mul_assoc]
  /-
    🎉 no goals
  -/


