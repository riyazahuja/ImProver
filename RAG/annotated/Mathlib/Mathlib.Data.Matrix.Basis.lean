/-- `stdBasisMatrix i j a` is the matrix with `a` in the `i`-th row, `j`-th column,
and zeroes elsewhere.
-/
def stdBasisMatrix (i : m) (j : n) (a : α) : Matrix m n α :=
  of <| fun i' j' => if i = i' ∧ j = j' then a else 0


theorem stdBasisMatrix_eq_of_single_single (i : m) (j : n) (a : α) :
    stdBasisMatrix i j a = Matrix.of (Pi.single i (Pi.single j a)) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    a : α
    ⊢ Eq (Matrix.stdBasisMatrix i j a) (Matrix.of (Pi.single i (Pi.single j a)))
  -/
  ext a b
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    a✝ : α
    a : m
    b : n
    ⊢ Eq (Matrix.stdBasisMatrix i j a✝ a b) (Matrix.of (Pi.single i (Pi.single j a …
  -/
  unfold stdBasisMatrix
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    a✝ : α
    a : m
    b : n
    ⊢ Eq (Matrix.of (fun i' j' => ite (And (Eq i i') (Eq j j')) a✝ 0) a b) (Matrix …
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
  by_cases hi : i = a <;> by_cases hj : j = b <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem smul_stdBasisMatrix [SMulZeroClass R α] (r : R) (i : m) (j : n) (a : α) :
    r • stdBasisMatrix i j a = stdBasisMatrix i j (r • a) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_4
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : SMulZeroClass R α
    r : R
    i : m
    j : n
    a : α
    ⊢ Eq (HSMul.hSMul r (Matrix.stdBasisMatrix i j a)) (Matrix.stdBasisMatrix i j  …
  -/
  unfold stdBasisMatrix
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_4
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : SMulZeroClass R α
    r : R
    i : m
    j : n
    a : α
    ⊢ Eq (HSMul.hSMul r (Matrix.of fun i' j' => ite (And (Eq i i') (Eq j j')) a 0) …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    R : Type u_4
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Zero α
    inst✝ : SMulZeroClass R α
    r : R
    i : m
    j : n
    a : α
    i✝ : m
    j✝ : n
    ⊢ Eq (HSMul.hSMul r (Matrix.of fun i' j' => ite (And (Eq i i') (Eq j j')) a 0) …
  -/
  simp [smul_ite]
  /-
    🎉 no goals
  -/


@[simp]
theorem stdBasisMatrix_zero (i : m) (j : n) : stdBasisMatrix i j (0 : α) = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    ⊢ Eq (Matrix.stdBasisMatrix i j 0) 0
  -/
  unfold stdBasisMatrix
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    ⊢ Eq (Matrix.of fun i' j' => ite (And (Eq i i') (Eq j j')) 0 0) 0
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    i✝ : m
    j✝ : n
    ⊢ Eq (Matrix.of (fun i' j' => ite (And (Eq i i') (Eq j j')) 0 0) i✝ j✝) (0 i✝  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem stdBasisMatrix_add [AddZeroClass α] (i : m) (j : n) (a b : α) :
    stdBasisMatrix i j (a + b) = stdBasisMatrix i j a + stdBasisMatrix i j b := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    i : m
    j : n
    a b : α
    ⊢ Eq (Matrix.stdBasisMatrix i j (HAdd.hAdd a b)) (HAdd.hAdd (Matrix.stdBasisMa …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    i : m
    j : n
    a b : α
    i✝ : m
    j✝ : n
    ⊢ Eq (Matrix.stdBasisMatrix i j (HAdd.hAdd a b) i✝ j✝) (HAdd.hAdd (Matrix.stdB …
  -/
  simp only [stdBasisMatrix, of_apply]
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : AddZeroClass α
    i : m
    j : n
    a b : α
    i✝ : m
    j✝ : n
    ⊢ Eq (ite (And (Eq i i✝) (Eq j j✝)) (HAdd.hAdd a b) 0) (HAdd.hAdd (Matrix.of f …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem mulVec_stdBasisMatrix [NonUnitalNonAssocSemiring α] [Fintype m]
    (i : n) (j : m) (c : α) (x : m → α) :
    mulVec (stdBasisMatrix i j c) x = Function.update (0 : n → α) i (c * x j) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    i : n
    j : m
    c : α
    x : m → α
    ⊢ Eq ((Matrix.stdBasisMatrix i j c).mulVec x) (Function.update 0 i (HMul.hMul  …
  -/
  ext i'
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    i : n
    j : m
    c : α
    x : m → α
    i' : n
    ⊢ Eq ((Matrix.stdBasisMatrix i j c).mulVec x i') (Function.update 0 i (HMul.hM …
  -/
  simp [stdBasisMatrix, mulVec, dotProduct]
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    i : n
    j : m
    c : α
    x : m → α
    i' : n
    ⊢ Eq (Finset.univ.sum fun x_1 => ite (And (Eq i i') (Eq j x_1)) (HMul.hMul c ( …
  -/
  rcases eq_or_ne i i' with rfl|h
    /-
      case h.inl
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype m
      i : n
      j : m
      c : α
      x : m → α
      ⊢ Eq (Finset.univ.sum fun x_1 => ite (And (Eq i i) (Eq j x_1)) (HMul.hMul c (x …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    i : n
    j : m
    c : α
    x : m → α
    i' : n
    h : Ne i i'
    ⊢ Eq (Finset.univ.sum fun x_1 => ite (And (Eq i i') (Eq j x_1)) (HMul.hMul c ( …
  -/
  simp [h, h.symm]
  /-
    🎉 no goals
  -/


theorem matrix_eq_sum_stdBasisMatrix [AddCommMonoid α] [Fintype m] [Fintype n] (x : Matrix m n α) :
    x = ∑ i : m, ∑ j : n, stdBasisMatrix i j (x i j) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    x : Matrix m n α
    ⊢ Eq x (Finset.univ.sum fun i => Finset.univ.sum fun j => Matrix.stdBasisMatri …
  -/
  ext i j
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    x : Matrix m n α
    i : m
    j : n
    ⊢ Eq (x i j) (Finset.univ.sum (fun i => Finset.univ.sum fun j => Matrix.stdBas …
  -/
  rw [← Fintype.sum_prod_type']
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    x : Matrix m n α
    i : m
    j : n
    ⊢ Eq (x i j) (Finset.univ.sum (fun x_1 => Matrix.stdBasisMatrix x_1.1 x_1.2 (x …
  -/
  simp [stdBasisMatrix, Matrix.sum_apply, Matrix.of_apply, ← Prod.mk.inj_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-11")] alias matrix_eq_sum_std_basis := matrix_eq_sum_stdBasisMatrix


theorem stdBasisMatrix_eq_single_vecMulVec_single [MulZeroOneClass α] (i : m) (j : n) :
    stdBasisMatrix i j (1 : α) = vecMulVec (Pi.single i 1) (Pi.single j 1) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    i : m
    j : n
    ⊢ Eq (Matrix.stdBasisMatrix i j 1) (Matrix.vecMulVec (Pi.single i 1) (Pi.singl …
  -/
  ext i' j'
  -- Porting note: lean3 didn't apply `mul_ite`.
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    i : m
    j : n
    i' : m
    j' : n
    ⊢ Eq (Matrix.stdBasisMatrix i j 1 i' j') (Matrix.vecMulVec (Pi.single i 1) (Pi …
  -/
  simp [-mul_ite, stdBasisMatrix, vecMulVec, ite_and, Pi.single_apply, eq_comm]
  /-
    🎉 no goals
  -/

-- TODO: tie this up with the `Basis` machinery of linear algebra
-- this is not completely trivial because we are indexing by two types, instead of one

@[deprecated stdBasisMatrix_eq_single_vecMulVec_single (since := "2024-08-11")]
theorem std_basis_eq_basis_mul_basis [MulZeroOneClass α] (i : m) (j : n) :
    stdBasisMatrix i j (1 : α) =
      vecMulVec (fun i' => ite (i = i') 1 0) fun j' => ite (j = j') 1 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    i : m
    j : n
    ⊢ Eq (Matrix.stdBasisMatrix i j 1) (Matrix.vecMulVec (fun i' => ite (Eq i i')  …
  -/
  rw [stdBasisMatrix_eq_single_vecMulVec_single]
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : MulZeroOneClass α
    i : m
    j : n
    ⊢ Eq (Matrix.vecMulVec (Pi.single i 1) (Pi.single j 1)) (Matrix.vecMulVec (fun …
  -/
                    /-
                      🎉 no goals
                    -/
  congr! with i <;> simp only [Pi.single_apply, eq_comm]
                    /-
                      🎉 no goals
                    -/

-- todo: the old proof used fintypes, I don't know `Finsupp` but this feels generalizable

@[elab_as_elim]
protected theorem induction_on'
    [AddCommMonoid α] [Finite m] [Finite n] {P : Matrix m n α → Prop} (M : Matrix m n α)
    (h_zero : P 0) (h_add : ∀ p q, P p → P q → P (p + q))
    (h_std_basis : ∀ (i : m) (j : n) (x : α), P (stdBasisMatrix i j x)) : P M := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Finite m
    inst✝ : Finite n
    P : Matrix m n α → Prop
    M : Matrix m n α
    h_zero : P 0
    h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
    h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
    ⊢ P M
  -/
  cases nonempty_fintype m; cases nonempty_fintype n
  /-
    case intro.intro
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Finite m
    inst✝ : Finite n
    P : Matrix m n α → Prop
    M : Matrix m n α
    h_zero : P 0
    h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
    h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
    val✝¹ : Fintype m
    val✝ : Fintype n
    ⊢ P M
  -/
  rw [matrix_eq_sum_stdBasisMatrix M, ← Finset.sum_product']
  /-
    case intro.intro
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq m
    inst✝³ : DecidableEq n
    inst✝² : AddCommMonoid α
    inst✝¹ : Finite m
    inst✝ : Finite n
    P : Matrix m n α → Prop
    M : Matrix m n α
    h_zero : P 0
    h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
    h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
    val✝¹ : Fintype m
    val✝ : Fintype n
    ⊢ P ((SProd.sprod Finset.univ Finset.univ).sum fun x => Matrix.stdBasisMatrix  …
  -/
  apply Finset.sum_induction _ _ h_add h_zero
    /-
      case intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq m
      inst✝³ : DecidableEq n
      inst✝² : AddCommMonoid α
      inst✝¹ : Finite m
      inst✝ : Finite n
      P : Matrix m n α → Prop
      M : Matrix m n α
      h_zero : P 0
      h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
      h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
      val✝¹ : Fintype m
      val✝ : Fintype n
      ⊢ ∀ (x : Prod m n), Membership.mem (SProd.sprod Finset.univ Finset.univ) x → P …
    -/
  · intros
    /-
      case intro.intro
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq m
      inst✝³ : DecidableEq n
      inst✝² : AddCommMonoid α
      inst✝¹ : Finite m
      inst✝ : Finite n
      P : Matrix m n α → Prop
      M : Matrix m n α
      h_zero : P 0
      h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
      h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
      val✝¹ : Fintype m
      val✝ : Fintype n
      x✝ : Prod m n
      a✝ : Membership.mem (SProd.sprod Finset.univ Finset.univ) x✝
      ⊢ P (Matrix.stdBasisMatrix x✝.1 x✝.2 (M x✝.1 x✝.2))
    -/
    apply h_std_basis
    /-
      🎉 no goals
    -/


@[elab_as_elim]
protected theorem induction_on
    [AddCommMonoid α] [Finite m] [Finite n] [Nonempty m] [Nonempty n]
    {P : Matrix m n α → Prop} (M : Matrix m n α) (h_add : ∀ p q, P p → P q → P (p + q))
    (h_std_basis : ∀ i j x, P (stdBasisMatrix i j x)) : P M :=
  Matrix.induction_on' M
    (by
      /-
        m : Type u_2
        n : Type u_3
        α : Type u_5
        inst✝⁶ : DecidableEq m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : AddCommMonoid α
        inst✝³ : Finite m
        inst✝² : Finite n
        inst✝¹ : Nonempty m
        inst✝ : Nonempty n
        P : Matrix m n α → Prop
        M : Matrix m n α
        h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
        h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
        ⊢ P 0
      -/
      inhabit m
      /-
        m : Type u_2
        n : Type u_3
        α : Type u_5
        inst✝⁶ : DecidableEq m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : AddCommMonoid α
        inst✝³ : Finite m
        inst✝² : Finite n
        inst✝¹ : Nonempty m
        inst✝ : Nonempty n
        P : Matrix m n α → Prop
        M : Matrix m n α
        h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
        h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
        inhabited_h : Inhabited m
        ⊢ P 0
      -/
      inhabit n
      /-
        m : Type u_2
        n : Type u_3
        α : Type u_5
        inst✝⁶ : DecidableEq m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : AddCommMonoid α
        inst✝³ : Finite m
        inst✝² : Finite n
        inst✝¹ : Nonempty m
        inst✝ : Nonempty n
        P : Matrix m n α → Prop
        M : Matrix m n α
        h_add : ∀ (p q : Matrix m n α), P p → P q → P (HAdd.hAdd p q)
        h_std_basis : ∀ (i : m) (j : n) (x : α), P (Matrix.stdBasisMatrix i j x)
        inhabited_h✝ : Inhabited m
        inhabited_h : Inhabited n
        ⊢ P 0
      -/
      simpa using h_std_basis default default 0)
      /-
        🎉 no goals
      -/
    h_add h_std_basis


@[simp]
theorem apply_same : stdBasisMatrix i j c i j = c :=
  if_pos (And.intro rfl rfl)


@[simp]
theorem apply_of_ne (h : ¬(i = i' ∧ j = j')) : stdBasisMatrix i j c i' j' = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    c : α
    i' : m
    j' : n
    h : Not (And (Eq i i') (Eq j j'))
    ⊢ Eq (Matrix.stdBasisMatrix i j c i' j') 0
  -/
  simp only [stdBasisMatrix, and_imp, ite_eq_right_iff, of_apply]
  /-
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : m
    j : n
    c : α
    i' : m
    j' : n
    h : Not (And (Eq i i') (Eq j j'))
    ⊢ Eq i i' → Eq j j' → Eq c 0
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem apply_of_row_ne {i i' : m} (hi : i ≠ i') (j j' : n) (a : α) :
                                         /-
                                           m : Type u_2
                                           n : Type u_3
                                           α : Type u_5
                                           inst✝² : DecidableEq m
                                           inst✝¹ : DecidableEq n
                                           inst✝ : Zero α
                                           i i' : m
                                           hi : Ne i i'
                                           j j' : n
                                           a : α
                                           ⊢ Eq (Matrix.stdBasisMatrix i j a i' j') 0
                                         -/
    stdBasisMatrix i j a i' j' = 0 := by simp [hi]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem apply_of_col_ne (i i' : m) {j j' : n} (hj : j ≠ j') (a : α) :
                                         /-
                                           m : Type u_2
                                           n : Type u_3
                                           α : Type u_5
                                           inst✝² : DecidableEq m
                                           inst✝¹ : DecidableEq n
                                           inst✝ : Zero α
                                           i i' : m
                                           j j' : n
                                           hj : Ne j j'
                                           a : α
                                           ⊢ Eq (Matrix.stdBasisMatrix i j a i' j') 0
                                         -/
    stdBasisMatrix i j a i' j' = 0 := by simp [hj]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem diag_zero (h : j ≠ i) : diag (stdBasisMatrix i j c) = 0 :=
  funext fun _ => if_neg fun ⟨e₁, e₂⟩ => h (e₂.trans e₁.symm)


@[simp]
theorem diag_same : diag (stdBasisMatrix i i c) = Pi.single i c := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : n
    c : α
    ⊢ Eq (Matrix.stdBasisMatrix i i c).diag (Pi.single i c)
  -/
  ext j
  /-
    case h
    n : Type u_3
    α : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Zero α
    i : n
    c : α
    j : n
    ⊢ Eq ((Matrix.stdBasisMatrix i i c).diag j) (Pi.single i c j)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  by_cases hij : i = j <;> (try rw [hij]) <;> simp [hij]
                                              /-
                                                🎉 no goals
                                              -/


omit [DecidableEq n] in
@[simp]
theorem mul_left_apply_same (i : l) (j : m) (b : n) (M : Matrix m n α) :
                                                     /-
                                                       l : Type u_1
                                                       m : Type u_2
                                                       n : Type u_3
                                                       α : Type u_5
                                                       inst✝³ : DecidableEq l
                                                       inst✝² : DecidableEq m
                                                       inst✝¹ : Fintype m
                                                       inst✝ : NonUnitalNonAssocSemiring α
                                                       c : α
                                                       i : l
                                                       j : m
                                                       b : n
                                                       M : Matrix m n α
                                                       ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) M i b) (HMul.hMul c (M j b))
                                                     -/
    (stdBasisMatrix i j c * M) i b = c * M j b := by simp [mul_apply, stdBasisMatrix]
                                                     /-
                                                       🎉 no goals
                                                     -/


omit [DecidableEq l] in
@[simp]
theorem mul_right_apply_same (i : m) (j : n) (a : l) (M : Matrix l m α) :
                                                     /-
                                                       l : Type u_1
                                                       m : Type u_2
                                                       n : Type u_3
                                                       α : Type u_5
                                                       inst✝³ : DecidableEq m
                                                       inst✝² : DecidableEq n
                                                       inst✝¹ : Fintype m
                                                       inst✝ : NonUnitalNonAssocSemiring α
                                                       c : α
                                                       i : m
                                                       j : n
                                                       a : l
                                                       M : Matrix l m α
                                                       ⊢ Eq (HMul.hMul M (Matrix.stdBasisMatrix i j c) a j) (HMul.hMul (M a i) c)
                                                     -/
    (M * stdBasisMatrix i j c) a j = M a i * c := by simp [mul_apply, stdBasisMatrix, mul_comm]
                                                     /-
                                                       🎉 no goals
                                                     -/


omit [DecidableEq n] in
@[simp]
theorem mul_left_apply_of_ne (i : l) (j : m) (a : l) (b : n) (h : a ≠ i) (M : Matrix m n α) :
                                             /-
                                               l : Type u_1
                                               m : Type u_2
                                               n : Type u_3
                                               α : Type u_5
                                               inst✝³ : DecidableEq l
                                               inst✝² : DecidableEq m
                                               inst✝¹ : Fintype m
                                               inst✝ : NonUnitalNonAssocSemiring α
                                               c : α
                                               i : l
                                               j : m
                                               a : l
                                               b : n
                                               h : Ne a i
                                               M : Matrix m n α
                                               ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) M a b) 0
                                             -/
    (stdBasisMatrix i j c * M) a b = 0 := by simp [mul_apply, h.symm]
                                             /-
                                               🎉 no goals
                                             -/


omit [DecidableEq l] in
@[simp]
theorem mul_right_apply_of_ne (i : m) (j : n) (a : l) (b : n) (hbj : b ≠ j) (M : Matrix l m α) :
                                             /-
                                               l : Type u_1
                                               m : Type u_2
                                               n : Type u_3
                                               α : Type u_5
                                               inst✝³ : DecidableEq m
                                               inst✝² : DecidableEq n
                                               inst✝¹ : Fintype m
                                               inst✝ : NonUnitalNonAssocSemiring α
                                               c : α
                                               i : m
                                               j : n
                                               a : l
                                               b : n
                                               hbj : Ne b j
                                               M : Matrix l m α
                                               ⊢ Eq (HMul.hMul M (Matrix.stdBasisMatrix i j c) a b) 0
                                             -/
    (M * stdBasisMatrix i j c) a b = 0 := by simp [mul_apply, hbj.symm]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem mul_same (i : l) (j : m) (k : n) (d : α) :
    stdBasisMatrix i j c * stdBasisMatrix j k d = stdBasisMatrix i k (c * d) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l
    j : m
    k : n
    d : α
    ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) (Matrix.stdBasisMatrix j k d)) ( …
  -/
  ext a b
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l
    j : m
    k : n
    d : α
    a : l
    b : n
    ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) (Matrix.stdBasisMatrix j k d) a  …
  -/
  simp only [mul_apply, stdBasisMatrix, boole_mul]
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l
    j : m
    k : n
    d : α
    a : l
    b : n
    ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (Matrix.of (fun i' j' => ite (And ( …
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
  by_cases h₁ : i = a <;> by_cases h₂ : k = b <;> simp [h₁, h₂]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem mul_of_ne (i : l) (j k : m) {l : n} (h : j ≠ k) (d : α) :
    stdBasisMatrix i j c * stdBasisMatrix k l d = 0 := by
  /-
    l✝ : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l✝
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l✝
    j k : m
    l : n
    h : Ne j k
    d : α
    ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) (Matrix.stdBasisMatrix k l d)) 0
  -/
  ext a b
  /-
    case a
    l✝ : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l✝
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l✝
    j k : m
    l : n
    h : Ne j k
    d : α
    a : l✝
    b : n
    ⊢ Eq (HMul.hMul (Matrix.stdBasisMatrix i j c) (Matrix.stdBasisMatrix k l d) a  …
  -/
  simp only [mul_apply, boole_mul, stdBasisMatrix, of_apply]
  /-
    case a
    l✝ : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type u_5
    inst✝⁴ : DecidableEq l✝
    inst✝³ : DecidableEq m
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    c : α
    i : l✝
    j k : m
    l : n
    h : Ne j k
    d : α
    a : l✝
    b : n
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (ite (And (Eq i a) (Eq j x)) c 0) (it …
  -/
  by_cases h₁ : i = a
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [h₁, h, h.symm]`
    /-
      case pos
      l✝ : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq l✝
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring α
      c : α
      i : l✝
      j k : m
      l : n
      h : Ne j k
      d : α
      a : l✝
      b : n
      h₁ : Eq i a
      ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (ite (And (Eq i a) (Eq j x)) c 0) (it …
    -/
  · simp only [h₁, true_and, mul_ite, ite_mul, zero_mul, mul_zero, ← ite_and, zero_apply]
    /-
      case pos
      l✝ : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq l✝
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring α
      c : α
      i : l✝
      j k : m
      l : n
      h : Ne j k
      d : α
      a : l✝
      b : n
      h₁ : Eq i a
      ⊢ Eq (Finset.univ.sum fun x => ite (And (And (Eq k x) (Eq l b)) (Eq j x)) (HMu …
    -/
    refine Finset.sum_eq_zero (fun x _ => ?_)
    /-
      case pos
      l✝ : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq l✝
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring α
      c : α
      i : l✝
      j k : m
      l : n
      h : Ne j k
      d : α
      a : l✝
      b : n
      h₁ : Eq i a
      x : m
      x✝ : Membership.mem Finset.univ x
      ⊢ Eq (ite (And (And (Eq k x) (Eq l b)) (Eq j x)) (HMul.hMul c d) 0) 0
    -/
    apply if_neg
    /-
      case pos.hnc
      l✝ : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq l✝
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring α
      c : α
      i : l✝
      j k : m
      l : n
      h : Ne j k
      d : α
      a : l✝
      b : n
      h₁ : Eq i a
      x : m
      x✝ : Membership.mem Finset.univ x
      ⊢ Not (And (And (Eq k x) (Eq l b)) (Eq j x))
    -/
    rintro ⟨⟨rfl, rfl⟩, h⟩
    /-
      case pos.hnc.intro.intro
      l✝ : Type u_1
      m : Type u_2
      n : Type u_3
      α : Type u_5
      inst✝⁴ : DecidableEq l✝
      inst✝³ : DecidableEq m
      inst✝² : DecidableEq n
      inst✝¹ : Fintype m
      inst✝ : NonUnitalNonAssocSemiring α
      c : α
      i : l✝
      j k : m
      l : n
      h✝ : Ne j k
      d : α
      a : l✝
      h₁ : Eq i a
      x✝ : Membership.mem Finset.univ k
      h : Eq j k
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
  · simp only [h₁, false_and, ite_false, mul_ite, zero_mul, mul_zero, ite_self,
      Finset.sum_const_zero, zero_apply]


theorem row_eq_zero_of_commute_stdBasisMatrix {i j k : n} {M : Matrix n n α}
    (hM : Commute (stdBasisMatrix i j 1) M) (hkj : k ≠ j) : M j k = 0 := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j k : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    hkj : Ne k j
    ⊢ Eq (M j k) 0
  -/
  have := ext_iff.mpr hM i k
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j k : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    hkj : Ne k j
    this : Eq (HMul.hMul (Matrix.stdBasisMatrix i j 1) M i k) (HMul.hMul M (Matrix …
    ⊢ Eq (M j k) 0
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem col_eq_zero_of_commute_stdBasisMatrix {i j k : n} {M : Matrix n n α}
    (hM : Commute (stdBasisMatrix i j 1) M) (hki : k ≠ i) : M k i = 0 := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j k : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    hki : Ne k i
    ⊢ Eq (M k i) 0
  -/
  have := ext_iff.mpr hM k j
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j k : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    hki : Ne k i
    this : Eq (HMul.hMul (Matrix.stdBasisMatrix i j 1) M k j) (HMul.hMul M (Matrix …
    ⊢ Eq (M k i) 0
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem diag_eq_of_commute_stdBasisMatrix {i j : n} {M : Matrix n n α}
    (hM : Commute (stdBasisMatrix i j 1) M) : M i i = M j j := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    ⊢ Eq (M i i) (M j j)
  -/
  have := ext_iff.mpr hM i j
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    i j : n
    M : Matrix n n α
    hM : Commute (Matrix.stdBasisMatrix i j 1) M
    this : Eq (HMul.hMul (Matrix.stdBasisMatrix i j 1) M i j) (HMul.hMul M (Matrix …
    ⊢ Eq (M i i) (M j j)
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- `M` is a scalar matrix if it commutes with every non-diagonal `stdBasisMatrix`. -/
theorem mem_range_scalar_of_commute_stdBasisMatrix {M : Matrix n n α}
    (hM : Pairwise fun i j => Commute (stdBasisMatrix i j 1) M) :
    M ∈ Set.range (Matrix.scalar n) := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
    ⊢ Membership.mem (Set.range ⇑(Matrix.scalar n)) M
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      n : Type u_3
      α : Type u_5
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Semiring α
      M : Matrix n n α
      hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
      h✝ : IsEmpty n
      ⊢ Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    -/
  · exact ⟨0, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
    h✝ : Nonempty n
    ⊢ Membership.mem (Set.range ⇑(Matrix.scalar n)) M
  -/
  obtain ⟨i⟩ := ‹Nonempty n›
  /-
    case inr.intro
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
    h✝ : Nonempty n
    i : n
    ⊢ Membership.mem (Set.range ⇑(Matrix.scalar n)) M
  -/
  refine ⟨M i i, Matrix.ext fun j k => ?_⟩
  /-
    case inr.intro
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
    h✝ : Nonempty n
    i j k : n
    ⊢ Eq ((Matrix.scalar n) (M i i) j k) (M j k)
  -/
  simp only [scalar_apply]
  /-
    case inr.intro
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
    h✝ : Nonempty n
    i j k : n
    ⊢ Eq (Matrix.diagonal (fun x => M i i) j k) (M j k)
  -/
  obtain rfl | hkl := Decidable.eq_or_ne j k
    /-
      case inr.intro.inl
      n : Type u_3
      α : Type u_5
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Semiring α
      M : Matrix n n α
      hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
      h✝ : Nonempty n
      i j : n
      ⊢ Eq (Matrix.diagonal (fun x => M i i) j j) (M j j)
    -/
  · rw [diagonal_apply_eq]
    /-
      case inr.intro.inl
      n : Type u_3
      α : Type u_5
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Semiring α
      M : Matrix n n α
      hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
      h✝ : Nonempty n
      i j : n
      ⊢ Eq (M i i) (M j j)
    -/
    obtain rfl | hij := Decidable.eq_or_ne i j
      /-
        case inr.intro.inl.inl
        n : Type u_3
        α : Type u_5
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Semiring α
        M : Matrix n n α
        hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
        h✝ : Nonempty n
        i : n
        ⊢ Eq (M i i) (M i i)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.inl.inr
        n : Type u_3
        α : Type u_5
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Semiring α
        M : Matrix n n α
        hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
        h✝ : Nonempty n
        i j : n
        hij : Ne i j
        ⊢ Eq (M i i) (M j j)
      -/
    · exact diag_eq_of_commute_stdBasisMatrix (hM hij)
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.inr
      n : Type u_3
      α : Type u_5
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Semiring α
      M : Matrix n n α
      hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
      h✝ : Nonempty n
      i j k : n
      hkl : Ne j k
      ⊢ Eq (Matrix.diagonal (fun x => M i i) j k) (M j k)
    -/
  · rw [diagonal_apply_ne _ hkl]
    /-
      case inr.intro.inr
      n : Type u_3
      α : Type u_5
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Semiring α
      M : Matrix n n α
      hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
      h✝ : Nonempty n
      i j k : n
      hkl : Ne j k
      ⊢ Eq 0 (M j k)
    -/
    obtain rfl | hij := Decidable.eq_or_ne i j
      /-
        case inr.intro.inr.inl
        n : Type u_3
        α : Type u_5
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Semiring α
        M : Matrix n n α
        hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
        h✝ : Nonempty n
        i k : n
        hkl : Ne i k
        ⊢ Eq 0 (M i k)
      -/
    · rw [col_eq_zero_of_commute_stdBasisMatrix (hM hkl.symm) hkl]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.inr.inr
        n : Type u_3
        α : Type u_5
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Semiring α
        M : Matrix n n α
        hM : Pairwise fun i j => Commute (Matrix.stdBasisMatrix i j 1) M
        h✝ : Nonempty n
        i j k : n
        hkl : Ne j k
        hij : Ne i j
        ⊢ Eq 0 (M j k)
      -/
    · rw [row_eq_zero_of_commute_stdBasisMatrix (hM hij) hkl.symm]
      /-
        🎉 no goals
      -/


theorem mem_range_scalar_iff_commute_stdBasisMatrix {M : Matrix n n α} :
    M ∈ Set.range (Matrix.scalar n) ↔ ∀ (i j : n), i ≠ j → Commute (stdBasisMatrix i j 1) M := by
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    ⊢ Iff (Membership.mem (Set.range ⇑(Matrix.scalar n)) M) (∀ (i j : n), Ne i j → …
  -/
  refine ⟨fun ⟨r, hr⟩ i j _ => hr ▸ Commute.symm ?_, mem_range_scalar_of_commute_stdBasisMatrix⟩
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    x✝¹ : Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    i j : n
    x✝ : Ne i j
    r : α
    hr : Eq ((Matrix.scalar n) r) M
    ⊢ Commute ((Matrix.scalar n) r) (Matrix.stdBasisMatrix i j 1)
  -/
  rw [scalar_commute_iff]
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    x✝¹ : Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    i j : n
    x✝ : Ne i j
    r : α
    hr : Eq ((Matrix.scalar n) r) M
    ⊢ Eq (HSMul.hSMul r (Matrix.stdBasisMatrix i j 1)) (HSMul.hSMul (MulOpposite.o …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `M` is a scalar matrix if and only if it commutes with every `stdBasisMatrix`. -/
theorem mem_range_scalar_iff_commute_stdBasisMatrix' {M : Matrix n n α} :
    M ∈ Set.range (Matrix.scalar n) ↔ ∀ (i j : n), Commute (stdBasisMatrix i j 1) M := by
  refine ⟨fun ⟨r, hr⟩ i j => hr ▸ Commute.symm ?_,
    fun hM => mem_range_scalar_iff_commute_stdBasisMatrix.mpr <| fun i j _ => hM i j⟩
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    x✝ : Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    i j : n
    r : α
    hr : Eq ((Matrix.scalar n) r) M
    ⊢ Commute ((Matrix.scalar n) r) (Matrix.stdBasisMatrix i j 1)
  -/
  rw [scalar_commute_iff]
  /-
    n : Type u_3
    α : Type u_5
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Semiring α
    M : Matrix n n α
    x✝ : Membership.mem (Set.range ⇑(Matrix.scalar n)) M
    i j : n
    r : α
    hr : Eq ((Matrix.scalar n) r) M
    ⊢ Eq (HSMul.hSMul r (Matrix.stdBasisMatrix i j 1)) (HSMul.hSMul (MulOpposite.o …
  -/
  simp
  /-
    🎉 no goals
  -/


