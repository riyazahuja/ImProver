/-- The trace of a square matrix. For more bundled versions, see:
* `Matrix.traceAddMonoidHom`
* `Matrix.traceLinearMap`
-/
def trace (A : Matrix n n R) : R :=
  ∑ i, diag A i


lemma trace_diagonal {o} [Fintype o] [DecidableEq o] (d : o → R) :
    trace (diagonal d) = ∑ i, d i := by
  /-
    R : Type u_6
    inst✝² : AddCommMonoid R
    o : Type u_8
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    d : o → R
    ⊢ Eq (Matrix.diagonal d).trace (Finset.univ.sum fun i => d i)
  -/
  simp only [trace, diag_apply, diagonal_apply_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_zero : trace (0 : Matrix n n R) = 0 :=
  (Finset.sum_const (0 : R)).trans <| smul_zero _


@[simp]
                                                                                  /-
                                                                                    n : Type u_3
                                                                                    R : Type u_6
                                                                                    inst✝² : Fintype n
                                                                                    inst✝¹ : AddCommMonoid R
                                                                                    inst✝ : IsEmpty n
                                                                                    A : Matrix n n R
                                                                                    ⊢ Eq A.trace 0
                                                                                  -/
lemma trace_eq_zero_of_isEmpty [IsEmpty n] (A : Matrix n n R) : trace A = 0 := by simp [trace]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem trace_add (A B : Matrix n n R) : trace (A + B) = trace A + trace B :=
  Finset.sum_add_distrib


@[simp]
theorem trace_smul [Monoid α] [DistribMulAction α R] (r : α) (A : Matrix n n R) :
    trace (r • A) = r • trace A :=
  Finset.smul_sum.symm


@[simp]
theorem trace_transpose (A : Matrix n n R) : trace Aᵀ = trace A :=
  rfl


@[simp]
theorem trace_conjTranspose [StarAddMonoid R] (A : Matrix n n R) : trace Aᴴ = star (trace A) :=
  (star_sum _ _).symm


/-- `Matrix.trace` as an `AddMonoidHom` -/
@[simps]
def traceAddMonoidHom : Matrix n n R →+ R where
  toFun := trace
  map_zero' := trace_zero n R
  map_add' := trace_add


/-- `Matrix.trace` as a `LinearMap` -/
@[simps]
def traceLinearMap [Semiring α] [Module α R] : Matrix n n R →ₗ[α] R where
  toFun := trace
  map_add' := trace_add
  map_smul' := trace_smul


@[simp]
theorem trace_list_sum (l : List (Matrix n n R)) : trace l.sum = (l.map trace).sum :=
  map_list_sum (traceAddMonoidHom n R) l


@[simp]
theorem trace_multiset_sum (s : Multiset (Matrix n n R)) : trace s.sum = (s.map trace).sum :=
  map_multiset_sum (traceAddMonoidHom n R) s


@[simp]
theorem trace_sum (s : Finset ι) (f : ι → Matrix n n R) :
    trace (∑ i ∈ s, f i) = ∑ i ∈ s, trace (f i) :=
  map_sum (traceAddMonoidHom n R) f s


theorem _root_.AddMonoidHom.map_trace [AddCommMonoid S] {F : Type*} [FunLike F R S]
    [AddMonoidHomClass F R S] (f : F) (A : Matrix n n R) :
    f (trace A) = trace ((f : R →+ S).mapMatrix A) :=
  map_sum f (fun i => diag A i) Finset.univ


lemma trace_blockDiagonal [DecidableEq p] (M : p → Matrix n n R) :
    trace (blockDiagonal M) = ∑ i, trace (M i) := by
  /-
    n : Type u_3
    p : Type u_4
    R : Type u_6
    inst✝³ : Fintype n
    inst✝² : Fintype p
    inst✝¹ : AddCommMonoid R
    inst✝ : DecidableEq p
    M : p → Matrix n n R
    ⊢ Eq (Matrix.blockDiagonal M).trace (Finset.univ.sum fun i => (M i).trace)
  -/
  simp [blockDiagonal, trace, Finset.sum_comm (γ := n), Fintype.sum_prod_type]
  /-
    🎉 no goals
  -/


lemma trace_blockDiagonal' [DecidableEq p] {m : p → Type*} [∀ i, Fintype (m i)]
    (M : ∀ i, Matrix (m i) (m i) R) :
    trace (blockDiagonal' M) = ∑ i, trace (M i) := by
  /-
    p : Type u_4
    R : Type u_6
    inst✝³ : Fintype p
    inst✝² : AddCommMonoid R
    inst✝¹ : DecidableEq p
    m : p → Type u_8
    inst✝ : (i : p) → Fintype (m i)
    M : (i : p) → Matrix (m i) (m i) R
    ⊢ Eq (Matrix.blockDiagonal' M).trace (Finset.univ.sum fun i => (M i).trace)
  -/
  simp [blockDiagonal', trace, Finset.sum_sigma']
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_sub (A B : Matrix n n R) : trace (A - B) = trace A - trace B :=
  Finset.sum_sub_distrib


@[simp]
theorem trace_neg (A : Matrix n n R) : trace (-A) = -trace A :=
  Finset.sum_neg_distrib


@[simp]
theorem trace_one : trace (1 : Matrix n n R) = Fintype.card n := by
  /-
    n : Type u_3
    R : Type u_6
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : AddCommMonoidWithOne R
    ⊢ Eq (Matrix.trace 1) ↑(Fintype.card n)
  -/
  simp_rw [trace, diag_one, Pi.one_def, Finset.sum_const, nsmul_one, Finset.card_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_transpose_mul [AddCommMonoid R] [Mul R] (A : Matrix m n R) (B : Matrix n m R) :
    trace (Aᵀ * Bᵀ) = trace (A * B) :=
  Finset.sum_comm


theorem trace_mul_comm [AddCommMonoid R] [CommSemigroup R] (A : Matrix m n R) (B : Matrix n m R) :
                                        /-
                                          m : Type u_2
                                          n : Type u_3
                                          R : Type u_6
                                          inst✝³ : Fintype m
                                          inst✝² : Fintype n
                                          inst✝¹ : AddCommMonoid R
                                          inst✝ : CommSemigroup R
                                          A : Matrix m n R
                                          B : Matrix n m R
                                          ⊢ Eq (HMul.hMul A B).trace (HMul.hMul B A).trace
                                        -/
    trace (A * B) = trace (B * A) := by rw [← trace_transpose, ← trace_transpose_mul, transpose_mul]
                                        /-
                                          🎉 no goals
                                        -/


theorem trace_mul_cycle [NonUnitalCommSemiring R] (A : Matrix m n R) (B : Matrix n p R)
    (C : Matrix p m R) : trace (A * B * C) = trace (C * A * B) := by
  /-
    m : Type u_2
    n : Type u_3
    p : Type u_4
    R : Type u_6
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : Fintype p
    inst✝ : NonUnitalCommSemiring R
    A : Matrix m n R
    B : Matrix n p R
    C : Matrix p m R
    ⊢ Eq (HMul.hMul (HMul.hMul A B) C).trace (HMul.hMul (HMul.hMul C A) B).trace
  -/
  rw [trace_mul_comm, Matrix.mul_assoc]
  /-
    🎉 no goals
  -/


theorem trace_mul_cycle' [NonUnitalCommSemiring R] (A : Matrix m n R) (B : Matrix n p R)
    (C : Matrix p m R) : trace (A * (B * C)) = trace (C * (A * B)) := by
  /-
    m : Type u_2
    n : Type u_3
    p : Type u_4
    R : Type u_6
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : Fintype p
    inst✝ : NonUnitalCommSemiring R
    A : Matrix m n R
    B : Matrix n p R
    C : Matrix p m R
    ⊢ Eq (HMul.hMul A (HMul.hMul B C)).trace (HMul.hMul C (HMul.hMul A B)).trace
  -/
  rw [← Matrix.mul_assoc, trace_mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_col_mul_row {ι : Type*} [Unique ι] [NonUnitalNonAssocSemiring R] (a b : n → R) :
    trace (col ι a * row ι b) = dotProduct a b := by
  /-
    n : Type u_3
    R : Type u_6
    inst✝² : Fintype n
    ι : Type u_8
    inst✝¹ : Unique ι
    inst✝ : NonUnitalNonAssocSemiring R
    a b : n → R
    ⊢ Eq (HMul.hMul (Matrix.col ι a) (Matrix.row ι b)).trace (dotProduct a b)
  -/
  apply Finset.sum_congr rfl
  /-
    n : Type u_3
    R : Type u_6
    inst✝² : Fintype n
    ι : Type u_8
    inst✝¹ : Unique ι
    inst✝ : NonUnitalNonAssocSemiring R
    a b : n → R
    ⊢ ∀ (x : n), Membership.mem Finset.univ x → Eq ((HMul.hMul (Matrix.col ι a) (M …
  -/
  simp [mul_apply]
  /-
    🎉 no goals
  -/


lemma trace_submatrix_succ {n : ℕ} [NonUnitalNonAssocSemiring R]
    (M : Matrix (Fin n.succ) (Fin n.succ) R) :
    M 0 0 + trace (submatrix M Fin.succ Fin.succ) = trace M := by
  /-
    R : Type u_6
    n : Nat
    inst✝ : NonUnitalNonAssocSemiring R
    M : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq (HAdd.hAdd (M 0 0) (M.submatrix Fin.succ Fin.succ).trace) M.trace
  -/
  delta trace
  /-
    R : Type u_6
    n : Nat
    inst✝ : NonUnitalNonAssocSemiring R
    M : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq (HAdd.hAdd (M 0 0) (Finset.univ.sum fun i => (M.submatrix Fin.succ Fin.su …
  -/
  rw [← (finSuccEquiv n).symm.sum_comp]
  /-
    R : Type u_6
    n : Nat
    inst✝ : NonUnitalNonAssocSemiring R
    M : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq (HAdd.hAdd (M 0 0) (Finset.univ.sum fun i => (M.submatrix Fin.succ Fin.su …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem trace_units_conj (M : (Matrix m m R)ˣ) (N : Matrix m m R) :
    trace ((M : Matrix _ _ _) * N * (↑M⁻¹ : Matrix _ _ _)) = trace N := by
  /-
    m : Type u_2
    R : Type u_6
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : CommSemiring R
    M : Units (Matrix m m R)
    N : Matrix m m R
    ⊢ Eq (HMul.hMul (HMul.hMul (↑M) N) ↑(Inv.inv M)).trace N.trace
  -/
  rw [trace_mul_cycle, Units.inv_mul, one_mul]
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
-- TODO(https://github.com/leanprover-community/mathlib4/issues/6607): fix elaboration so that the ascription isn't needed
theorem trace_units_conj' (M : (Matrix m m R)ˣ) (N : Matrix m m R) :
    trace ((↑M⁻¹ : Matrix _ _ _) * N * (↑M : Matrix _ _ _)) = trace N :=
  trace_units_conj M⁻¹ N


@[simp]
theorem trace_fin_zero (A : Matrix (Fin 0) (Fin 0) R) : trace A = 0 :=
  rfl


theorem trace_fin_one (A : Matrix (Fin 1) (Fin 1) R) : trace A = A 0 0 :=
  add_zero _


theorem trace_fin_two (A : Matrix (Fin 2) (Fin 2) R) : trace A = A 0 0 + A 1 1 :=
  congr_arg (_ + ·) (add_zero (A 1 1))


theorem trace_fin_three (A : Matrix (Fin 3) (Fin 3) R) : trace A = A 0 0 + A 1 1 + A 2 2 := by
  /-
    R : Type u_6
    inst✝ : AddCommMonoid R
    A : Matrix (Fin 3) (Fin 3) R
    ⊢ Eq A.trace (HAdd.hAdd (HAdd.hAdd (A 0 0) (A 1 1)) (A 2 2))
  -/
  rw [← add_zero (A 2 2), add_assoc]
  /-
    R : Type u_6
    inst✝ : AddCommMonoid R
    A : Matrix (Fin 3) (Fin 3) R
    ⊢ Eq A.trace (HAdd.hAdd (A 0 0) (HAdd.hAdd (A 1 1) (HAdd.hAdd (A 2 2) 0)))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_fin_one_of (a : R) : trace !![a] = a :=
  trace_fin_one _


@[simp]
theorem trace_fin_two_of (a b c d : R) : trace !![a, b; c, d] = a + d :=
  trace_fin_two _


@[simp]
theorem trace_fin_three_of (a b c d e f g h i : R) :
    trace !![a, b, c; d, e, f; g, h, i] = a + e + i :=
  trace_fin_three _


@[simp]
theorem trace_zero (h : j ≠ i) : trace (stdBasisMatrix i j c) = 0 := by
  -- Porting note: added `-diag_apply`
  /-
    n : Type u_10
    α : Type u_12
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : AddCommMonoid α
    i j : n
    c : α
    h : Ne j i
    ⊢ Eq (Matrix.stdBasisMatrix i j c).trace 0
  -/
  simp [trace, -diag_apply, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_eq : trace (stdBasisMatrix i i c) = c := by
  -- Porting note: added `-diag_apply`
  /-
    n : Type u_10
    α : Type u_12
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : AddCommMonoid α
    i : n
    c : α
    ⊢ Eq (Matrix.stdBasisMatrix i i c).trace c
  -/
  simp [trace, -diag_apply]
  /-
    🎉 no goals
  -/


