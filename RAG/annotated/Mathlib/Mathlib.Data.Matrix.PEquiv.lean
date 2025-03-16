/-- `toMatrix` returns a matrix containing ones and zeros. `f.toMatrix i j` is `1` if
  `f i = some j` and `0` otherwise -/
def toMatrix [DecidableEq n] [Zero α] [One α] (f : m ≃. n) : Matrix m n α :=
  of fun i j => if j ∈ f i then (1 : α) else 0

-- TODO: set as an equation lemma for `toMatrix`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem toMatrix_apply [DecidableEq n] [Zero α] [One α] (f : m ≃. n) (i j) :
    toMatrix f i j = if j ∈ f i then (1 : α) else 0 :=
  rfl


theorem mul_matrix_apply [Fintype m] [DecidableEq m] [Semiring α] (f : l ≃. m) (M : Matrix m n α)
    (i j) : (f.toMatrix * M :) i j = Option.casesOn (f i) 0 fun fi => M fi j := by
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    f : PEquiv l m
    M : Matrix m n α
    i : l
    j : n
    ⊢ Eq (HMul.hMul f.toMatrix M i j) (Option.casesOn (f i) 0 fun fi => M fi j)
  -/
  dsimp [toMatrix, Matrix.mul_apply]
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    f : PEquiv l m
    M : Matrix m n α
    i : l
    j : n
    ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (ite (Membership.mem (f i) j_1) 1 0 …
  -/
  cases' h : f i with fi
    /-
      case none
      l : Type u_2
      m : Type u_3
      n : Type u_4
      α : Type v
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      inst✝ : Semiring α
      f : PEquiv l m
      M : Matrix m n α
      i : l
      j : n
      h : Eq (f i) Option.none
      ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (ite (Membership.mem Option.none j_ …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case some
      l : Type u_2
      m : Type u_3
      n : Type u_4
      α : Type v
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      inst✝ : Semiring α
      f : PEquiv l m
      M : Matrix m n α
      i : l
      j : n
      fi : m
      h : Eq (f i) (Option.some fi)
      ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (ite (Membership.mem (Option.some f …
    -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  · rw [Finset.sum_eq_single fi] <;> simp +contextual [h, eq_comm]
                                     /-
                                       🎉 no goals
                                     -/


theorem toMatrix_symm [DecidableEq m] [DecidableEq n] [Zero α] [One α] (f : m ≃. n) :
    (f.symm.toMatrix : Matrix n m α) = f.toMatrixᵀ := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    f : PEquiv m n
    ⊢ Eq f.symm.toMatrix f.toMatrix.transpose
  -/
  ext
  /-
    case a
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    f : PEquiv m n
    i✝ : n
    j✝ : m
    ⊢ Eq (f.symm.toMatrix i✝ j✝) (f.toMatrix.transpose i✝ j✝)
  -/
  simp only [transpose, mem_iff_mem f, toMatrix_apply]
  /-
    case a
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    f : PEquiv m n
    i✝ : n
    j✝ : m
    ⊢ Eq (ite (Membership.mem (f j✝) i✝) 1 0) (Matrix.of (fun x y => ite (Membersh …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem toMatrix_refl [DecidableEq n] [Zero α] [One α] :
    ((PEquiv.refl n).toMatrix : Matrix n n α) = 1 := by
  /-
    n : Type u_4
    α : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    ⊢ Eq (PEquiv.refl n).toMatrix 1
  -/
  ext
  /-
    case a
    n : Type u_4
    α : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : One α
    i✝ j✝ : n
    ⊢ Eq ((PEquiv.refl n).toMatrix i✝ j✝) (1 i✝ j✝)
  -/
  simp [toMatrix_apply, one_apply]
  /-
    🎉 no goals
  -/


theorem matrix_mul_apply [Fintype m] [Semiring α] [DecidableEq n] (M : Matrix l m α) (f : m ≃. n)
    (i j) : (M * f.toMatrix :) i j = Option.casesOn (f.symm j) 0 fun fj => M i fj := by
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Semiring α
    inst✝ : DecidableEq n
    M : Matrix l m α
    f : PEquiv m n
    i : l
    j : n
    ⊢ Eq (HMul.hMul M f.toMatrix i j) (Option.casesOn (f.symm j) 0 fun fj => M i fj)
  -/
  dsimp [toMatrix, Matrix.mul_apply]
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Semiring α
    inst✝ : DecidableEq n
    M : Matrix l m α
    f : PEquiv m n
    i : l
    j : n
    ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (M i j_1) (ite (Membership.mem (f j …
  -/
  cases' h : f.symm j with fj
    /-
      case none
      l : Type u_2
      m : Type u_3
      n : Type u_4
      α : Type v
      inst✝² : Fintype m
      inst✝¹ : Semiring α
      inst✝ : DecidableEq n
      M : Matrix l m α
      f : PEquiv m n
      i : l
      j : n
      h : Eq (f.symm j) Option.none
      ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (M i j_1) (ite (Membership.mem (f j …
    -/
  · simp [h, ← f.eq_some_iff]
    /-
      🎉 no goals
    -/
    /-
      case some
      l : Type u_2
      m : Type u_3
      n : Type u_4
      α : Type v
      inst✝² : Fintype m
      inst✝¹ : Semiring α
      inst✝ : DecidableEq n
      M : Matrix l m α
      f : PEquiv m n
      i : l
      j : n
      fj : m
      h : Eq (f.symm j) (Option.some fj)
      ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (M i j_1) (ite (Membership.mem (f j …
    -/
  · rw [Finset.sum_eq_single fj]
      /-
        case some
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type v
        inst✝² : Fintype m
        inst✝¹ : Semiring α
        inst✝ : DecidableEq n
        M : Matrix l m α
        f : PEquiv m n
        i : l
        j : n
        fj : m
        h : Eq (f.symm j) (Option.some fj)
        ⊢ Eq (HMul.hMul (M i fj) (ite (Membership.mem (f fj) j) 1 0)) (Option.rec 0 (f …
      -/
    · simp [h, ← f.eq_some_iff]
      /-
        🎉 no goals
      -/
      /-
        case some.h₀
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type v
        inst✝² : Fintype m
        inst✝¹ : Semiring α
        inst✝ : DecidableEq n
        M : Matrix l m α
        f : PEquiv m n
        i : l
        j : n
        fj : m
        h : Eq (f.symm j) (Option.some fj)
        ⊢ ∀ (b : m), Membership.mem Finset.univ b → Ne b fj → Eq (HMul.hMul (M i b) (i …
      -/
    · rintro b - n
      /-
        case some.h₀
        l : Type u_2
        m : Type u_3
        n✝ : Type u_4
        α : Type v
        inst✝² : Fintype m
        inst✝¹ : Semiring α
        inst✝ : DecidableEq n✝
        M : Matrix l m α
        f : PEquiv m n✝
        i : l
        j : n✝
        fj : m
        h : Eq (f.symm j) (Option.some fj)
        b : m
        n : Ne b fj
        ⊢ Eq (HMul.hMul (M i b) (ite (Membership.mem (f b) j) 1 0)) 0
      -/
      simp [h, ← f.eq_some_iff, n.symm]
      /-
        🎉 no goals
      -/
      /-
        case some.h₁
        l : Type u_2
        m : Type u_3
        n : Type u_4
        α : Type v
        inst✝² : Fintype m
        inst✝¹ : Semiring α
        inst✝ : DecidableEq n
        M : Matrix l m α
        f : PEquiv m n
        i : l
        j : n
        fj : m
        h : Eq (f.symm j) (Option.some fj)
        ⊢ Not (Membership.mem Finset.univ fj) → Eq (HMul.hMul (M i fj) (ite (Membershi …
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem toPEquiv_mul_matrix [Fintype m] [DecidableEq m] [Semiring α] (f : m ≃ m)
    (M : Matrix m n α) : f.toPEquiv.toMatrix * M = M.submatrix f id := by
  /-
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    f : Equiv m m
    M : Matrix m n α
    ⊢ Eq (HMul.hMul f.toPEquiv.toMatrix M) (M.submatrix (⇑f) id)
  -/
  ext i j
  /-
    case a
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    f : Equiv m m
    M : Matrix m n α
    i : m
    j : n
    ⊢ Eq (HMul.hMul f.toPEquiv.toMatrix M i j) (M.submatrix (⇑f) id i j)
  -/
  rw [mul_matrix_apply, Equiv.toPEquiv_apply, submatrix_apply, id]
  /-
    🎉 no goals
  -/


theorem mul_toPEquiv_toMatrix {m n α : Type*} [Fintype n] [DecidableEq n] [Semiring α] (f : n ≃ n)
    (M : Matrix m n α) : M * f.toPEquiv.toMatrix = M.submatrix id f.symm :=
  Matrix.ext fun i j => by
    rw [PEquiv.matrix_mul_apply, ← Equiv.toPEquiv_symm, Equiv.toPEquiv_apply,
      Matrix.submatrix_apply, id]


theorem toMatrix_trans [Fintype m] [DecidableEq m] [DecidableEq n] [Semiring α] (f : l ≃. m)
    (g : m ≃. n) : ((f.trans g).toMatrix : Matrix l n α) = f.toMatrix * g.toMatrix := by
  /-
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Semiring α
    f : PEquiv l m
    g : PEquiv m n
    ⊢ Eq (f.trans g).toMatrix (HMul.hMul f.toMatrix g.toMatrix)
  -/
  ext i j
  /-
    case a
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Semiring α
    f : PEquiv l m
    g : PEquiv m n
    i : l
    j : n
    ⊢ Eq ((f.trans g).toMatrix i j) (HMul.hMul f.toMatrix g.toMatrix i j)
  -/
  rw [mul_matrix_apply]
  /-
    case a
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Semiring α
    f : PEquiv l m
    g : PEquiv m n
    i : l
    j : n
    ⊢ Eq ((f.trans g).toMatrix i j) (Option.casesOn (f i) 0 fun fi => g.toMatrix f …
  -/
  dsimp [toMatrix, PEquiv.trans]
  /-
    case a
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Semiring α
    f : PEquiv l m
    g : PEquiv m n
    i : l
    j : n
    ⊢ Eq (ite (Membership.mem ((f i).bind ⇑g) j) 1 0) (Option.rec 0 (fun val => it …
  -/
                /-
                  🎉 no goals
                -/
  cases f i <;> simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem toMatrix_bot [DecidableEq n] [Zero α] [One α] :
    ((⊥ : PEquiv m n).toMatrix : Matrix m n α) = 0 :=
  rfl


theorem toMatrix_injective [DecidableEq n] [MonoidWithZero α] [Nontrivial α] :
    Function.Injective (@toMatrix m n α _ _ _) := by
  classical
    intro f g
    refine not_imp_not.1 ?_
    simp only [Matrix.ext_iff.symm, toMatrix_apply, PEquiv.ext_iff, not_forall, exists_imp]
    intro i hi
    use i
    cases' hf : f i with fi
    · cases' hg : g i with gi
      · rw [hf, hg] at hi; exact (hi rfl).elim
      · use gi
        simp
    · use fi
      simp [hf.symm, Ne.symm hi]


theorem toMatrix_swap [DecidableEq n] [Ring α] (i j : n) :
    (Equiv.swap i j).toPEquiv.toMatrix =
      (1 : Matrix n n α) - (single i i).toMatrix - (single j j).toMatrix + (single i j).toMatrix +
        (single j i).toMatrix := by
  /-
    n : Type u_4
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Ring α
    i j : n
    ⊢ Eq (Equiv.toPEquiv (Equiv.swap i j)).toMatrix (HAdd.hAdd (HAdd.hAdd (HSub.hS …
  -/
  ext
  /-
    case a
    n : Type u_4
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Ring α
    i j i✝ j✝ : n
    ⊢ Eq ((Equiv.toPEquiv (Equiv.swap i j)).toMatrix i✝ j✝) (HAdd.hAdd (HAdd.hAdd  …
  -/
  dsimp [toMatrix, single, Equiv.swap_apply_def, Equiv.toPEquiv, one_apply]
  /-
    case a
    n : Type u_4
    α : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Ring α
    i j i✝ j✝ : n
    ⊢ Eq (ite (Membership.mem (Option.some (ite (Eq i✝ i) j (ite (Eq i✝ j) i i✝))) …
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


@[simp]
theorem single_mul_single [Fintype n] [DecidableEq k] [DecidableEq m] [DecidableEq n] [Semiring α]
    (a : m) (b : n) (c : k) :
    ((single a b).toMatrix : Matrix _ _ α) * (single b c).toMatrix = (single a c).toMatrix := by
  /-
    k : Type u_1
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq k
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Semiring α
    a : m
    b : n
    c : k
    ⊢ Eq (HMul.hMul (PEquiv.single a b).toMatrix (PEquiv.single b c).toMatrix) (PE …
  -/
  rw [← toMatrix_trans, single_trans_single]
  /-
    🎉 no goals
  -/


theorem single_mul_single_of_ne [Fintype n] [DecidableEq n] [DecidableEq k] [DecidableEq m]
    [Semiring α] {b₁ b₂ : n} (hb : b₁ ≠ b₂) (a : m) (c : k) :
    (single a b₁).toMatrix * (single b₂ c).toMatrix = (0 : Matrix _ _ α) := by
  /-
    k : Type u_1
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq k
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    b₁ b₂ : n
    hb : Ne b₁ b₂
    a : m
    c : k
    ⊢ Eq (HMul.hMul (PEquiv.single a b₁).toMatrix (PEquiv.single b₂ c).toMatrix) 0
  -/
  rw [← toMatrix_trans, single_trans_single_of_ne hb, toMatrix_bot]
  /-
    🎉 no goals
  -/


/-- Restatement of `single_mul_single`, which will simplify expressions in `simp` normal form,
  when associativity may otherwise need to be carefully applied. -/
@[simp]
theorem single_mul_single_right [Fintype n] [Fintype k] [DecidableEq n] [DecidableEq k]
    [DecidableEq m] [Semiring α] (a : m) (b : n) (c : k) (M : Matrix k l α) :
    (single a b).toMatrix * ((single b c).toMatrix * M) = (single a c).toMatrix * M := by
  /-
    k : Type u_1
    l : Type u_2
    m : Type u_3
    n : Type u_4
    α : Type v
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype k
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq k
    inst✝¹ : DecidableEq m
    inst✝ : Semiring α
    a : m
    b : n
    c : k
    M : Matrix k l α
    ⊢ Eq (HMul.hMul (PEquiv.single a b).toMatrix (HMul.hMul (PEquiv.single b c).to …
  -/
  rw [← Matrix.mul_assoc, single_mul_single]
  /-
    🎉 no goals
  -/


/-- We can also define permutation matrices by permuting the rows of the identity matrix. -/
theorem equiv_toPEquiv_toMatrix [DecidableEq n] [Zero α] [One α] (σ : Equiv n n) (i j : n) :
    σ.toPEquiv.toMatrix i j = (1 : Matrix n n α) (σ i) j :=
  if_congr Option.some_inj rfl rfl


