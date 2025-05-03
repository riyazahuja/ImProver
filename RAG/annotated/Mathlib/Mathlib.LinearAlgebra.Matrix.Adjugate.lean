/-- `cramerMap A b i` is the determinant of the matrix `A` with column `i` replaced with `b`,
  and thus `cramerMap A b` is the vector output by Cramer's rule on `A` and `b`.

  If `A * x = b` has a unique solution in `x`, `cramerMap A` sends the vector `b` to `A.det • x`.
  Otherwise, the outcome of `cramerMap` is well-defined but not necessarily useful.
-/
def cramerMap (i : n) : α :=
  (A.updateCol i b).det


theorem cramerMap_is_linear (i : n) : IsLinearMap α fun b => cramerMap A b i :=
  { map_add := det_updateCol_add _ _
    map_smul := det_updateCol_smul _ _ }


theorem cramer_is_linear : IsLinearMap α (cramerMap A) := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ IsLinearMap α A.cramerMap
  -/
  constructor <;> intros <;> ext i
    /-
      case map_add.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      x✝ y✝ : n → α
      i : n
      ⊢ Eq (A.cramerMap (HAdd.hAdd x✝ y✝) i) (HAdd.hAdd (A.cramerMap x✝) (A.cramerMa …
    -/
  · apply (cramerMap_is_linear A i).1
    /-
      🎉 no goals
    -/
    /-
      case map_smul.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      c✝ : α
      x✝ : n → α
      i : n
      ⊢ Eq (A.cramerMap (HSMul.hSMul c✝ x✝) i) (HSMul.hSMul c✝ (A.cramerMap x✝) i)
    -/
  · apply (cramerMap_is_linear A i).2
    /-
      🎉 no goals
    -/


/-- `cramer A b i` is the determinant of the matrix `A` with column `i` replaced with `b`,
  and thus `cramer A b` is the vector output by Cramer's rule on `A` and `b`.

  If `A * x = b` has a unique solution in `x`, `cramer A` sends the vector `b` to `A.det • x`.
  Otherwise, the outcome of `cramer` is well-defined but not necessarily useful.
 -/
def cramer (A : Matrix n n α) : (n → α) →ₗ[α] (n → α) :=
  IsLinearMap.mk' (cramerMap A) (cramer_is_linear A)


theorem cramer_apply (i : n) : cramer A b i = (A.updateCol i b).det :=
  rfl


theorem cramer_transpose_apply (i : n) : cramer Aᵀ b i = (A.updateRow i b).det := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    i : n
    ⊢ Eq (A.transpose.cramer b i) (A.updateRow i b).det
  -/
  rw [cramer_apply, updateCol_transpose, det_transpose]
  /-
    🎉 no goals
  -/


theorem cramer_transpose_row_self (i : n) : Aᵀ.cramer (A i) = Pi.single i A.det := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    ⊢ Eq (A.transpose.cramer (A i)) (Pi.single i A.det)
  -/
  ext j
  /-
    case h
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.transpose.cramer (A i) j) (Pi.single i A.det j)
  -/
  rw [cramer_apply, Pi.single_apply]
  /-
    case h
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.transpose.updateCol j (A i)).det (ite (Eq j i) A.det 0)
  -/
  split_ifs with h
  · -- i = j: this entry should be `A.det`
    /-
      case pos
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      h : Eq j i
      ⊢ Eq (A.transpose.updateCol j (A i)).det A.det
    -/
    subst h
    /-
      case pos
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      ⊢ Eq (A.transpose.updateCol j (A j)).det A.det
    -/
    simp only [updateCol_transpose, det_transpose, updateRow_eq_self]
    /-
      🎉 no goals
    -/
  · -- i ≠ j: this entry should be 0
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      h : Not (Eq j i)
      ⊢ Eq (A.transpose.updateCol j (A i)).det 0
    -/
    rw [updateCol_transpose, det_transpose]
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      h : Not (Eq j i)
      ⊢ Eq (A.updateRow j (A i)).det 0
    -/
    apply det_zero_of_row_eq h
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      h : Not (Eq j i)
      ⊢ Eq (A.updateRow j (A i) j) (A.updateRow j (A i) i)
    -/
    rw [updateRow_self, updateRow_ne (Ne.symm h)]
    /-
      🎉 no goals
    -/


theorem cramer_row_self (i : n) (h : ∀ j, b j = A j i) : A.cramer b = Pi.single i A.det := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    i : n
    h : ∀ (j : n), Eq (b j) (A j i)
    ⊢ Eq (A.cramer b) (Pi.single i A.det)
  -/
  rw [← transpose_transpose A, det_transpose]
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    i : n
    h : ∀ (j : n), Eq (b j) (A j i)
    ⊢ Eq (A.transpose.transpose.cramer b) (Pi.single i A.transpose.det)
  -/
  convert cramer_transpose_row_self Aᵀ i
  /-
    case h.e'_2.h.e'_6
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    i : n
    h : ∀ (j : n), Eq (b j) (A j i)
    ⊢ Eq b (A.transpose i)
  -/
  exact funext h
  /-
    🎉 no goals
  -/


@[simp]
theorem cramer_one : cramer (1 : Matrix n n α) = 1 := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    ⊢ Eq (Matrix.cramer 1) 1
  -/
  ext i j
  /-
    case h.h.h
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    i j : n
    ⊢ Eq (((Matrix.cramer 1).comp (LinearMap.single α (fun i => α) i)) 1 j) ((Line …
  -/
  convert congr_fun (cramer_row_self (1 : Matrix n n α) (Pi.single i 1) i _) j
    /-
      case h.e'_3.h.e'_1
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      i j : n
      ⊢ Eq 1 (Matrix.det 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.h.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      i j : n
      ⊢ ∀ (j : n), Eq (Pi.single i 1 j) (1 j i)
    -/
  · intro j
    /-
      case h.h.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      i j✝ j : n
      ⊢ Eq (Pi.single i 1 j) (1 j i)
    -/
    rw [Matrix.one_eq_pi_single, Pi.single_comm]
    /-
      🎉 no goals
    -/


theorem cramer_smul (r : α) (A : Matrix n n α) :
    cramer (r • A) = r ^ (Fintype.card n - 1) • cramer A :=
  LinearMap.ext fun _ => funext fun _ => det_updateCol_smul_left _ _ _ _


@[simp]
theorem cramer_subsingleton_apply [Subsingleton n] (A : Matrix n n α) (b : n → α) (i : n) :
                             /-
                               n : Type v
                               α : Type w
                               inst✝³ : DecidableEq n
                               inst✝² : Fintype n
                               inst✝¹ : CommRing α
                               inst✝ : Subsingleton n
                               A : Matrix n n α
                               b : n → α
                               i : n
                               ⊢ Eq (A.cramer b i) (b i)
                             -/
    cramer A b i = b i := by rw [cramer_apply, det_eq_elem_of_subsingleton _ i, updateCol_self]
                             /-
                               🎉 no goals
                             -/


theorem cramer_zero [Nontrivial n] : cramer (0 : Matrix n n α) = 0 := by
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    ⊢ Eq (Matrix.cramer 0) 0
  -/
  ext i j
  /-
    case h.h.h
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j : n
    ⊢ Eq (((Matrix.cramer 0).comp (LinearMap.single α (fun i => α) i)) 1 j) ((Line …
  -/
  obtain ⟨j', hj'⟩ : ∃ j', j' ≠ j := exists_ne j
  /-
    case h.h.h.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    ⊢ Eq (((Matrix.cramer 0).comp (LinearMap.single α (fun i => α) i)) 1 j) ((Line …
  -/
  apply det_eq_zero_of_column_eq_zero j'
  /-
    case h.h.h.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    ⊢ ∀ (i_1 : n), Eq (Matrix.updateCol 0 j ((LinearMap.single α (fun i => α) i) 1 …
  -/
  intro j''
  /-
    case h.h.h.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    j'' : n
    ⊢ Eq (Matrix.updateCol 0 j ((LinearMap.single α (fun i => α) i) 1) j'' j') 0
  -/
  simp [updateCol_ne hj']
  /-
    🎉 no goals
  -/


/-- Use linearity of `cramer` to take it out of a summation. -/
theorem sum_cramer {β} (s : Finset β) (f : β → n → α) :
    (∑ x ∈ s, cramer A (f x)) = cramer A (∑ x ∈ s, f x) :=
  (map_sum (cramer A) ..).symm


/-- Use linearity of `cramer` and vector evaluation to take `cramer A _ i` out of a summation. -/
theorem sum_cramer_apply {β} (s : Finset β) (f : n → β → α) (i : n) :
    (∑ x ∈ s, cramer A (fun j => f j x) i) = cramer A (fun j : n => ∑ x ∈ s, f j x) i :=
  calc
    (∑ x ∈ s, cramer A (fun j => f j x) i) = (∑ x ∈ s, cramer A fun j => f j x) i :=
      (Finset.sum_apply i s _).symm
    _ = cramer A (fun j : n => ∑ x ∈ s, f j x) i := by
      /-
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        β : Type u_1
        s : Finset β
        f : n → β → α
        i : n
        ⊢ Eq (s.sum (fun x => A.cramer fun j => f j x) i) (A.cramer (fun j => s.sum fu …
      -/
      rw [sum_cramer, cramer_apply, cramer_apply]
      /-
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        β : Type u_1
        s : Finset β
        f : n → β → α
        i : n
        ⊢ Eq (A.updateCol i (s.sum fun x j => f j x)).det (A.updateCol i fun j => s.su …
      -/
      simp only [updateCol]
      /-
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        β : Type u_1
        s : Finset β
        f : n → β → α
        i : n
        ⊢ Eq (Matrix.of fun i_1 => Function.update (A i_1) i (s.sum (fun x j => f j x) …
      -/
      congr with j
      /-
        case e_M.h.e_6.h.h.h
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        β : Type u_1
        s : Finset β
        f : n → β → α
        i j x✝ : n
        ⊢ Eq (Function.update (A j) i (s.sum (fun x j => f j x) j) x✝) (Function.updat …
      -/
      congr
      /-
        case e_M.h.e_6.h.h.h.e_v
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        β : Type u_1
        s : Finset β
        f : n → β → α
        i j x✝ : n
        ⊢ Eq (s.sum (fun x j => f j x) j) (s.sum fun x => f j x)
      -/
      apply Finset.sum_apply
      /-
        🎉 no goals
      -/


theorem cramer_submatrix_equiv (A : Matrix m m α) (e : n ≃ m) (b : n → α) :
    cramer (A.submatrix e e) b = cramer A (b ∘ e.symm) ∘ e := by
  /-
    m : Type u
    n : Type v
    α : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : CommRing α
    A : Matrix m m α
    e : Equiv n m
    b : n → α
    ⊢ Eq ((A.submatrix ⇑e ⇑e).cramer b) (Function.comp (A.cramer (Function.comp b  …
  -/
  ext i
  simp_rw [Function.comp_apply, cramer_apply, updateCol_submatrix_equiv,
    det_submatrix_equiv_self e, Function.comp_def]


theorem cramer_reindex (e : m ≃ n) (A : Matrix m m α) (b : n → α) :
    cramer (reindex e e A) b = cramer A (b ∘ e) ∘ e.symm :=
  cramer_submatrix_equiv _ _ _


/-- The adjugate matrix is the transpose of the cofactor matrix.

  Typically, the cofactor matrix is defined by taking minors,
  i.e. the determinant of the matrix with a row and column removed.
  However, the proof of `mul_adjugate` becomes a lot easier if we use the
  matrix replacing a column with a basis vector, since it allows us to use
  facts about the `cramer` map.
-/
def adjugate (A : Matrix n n α) : Matrix n n α :=
  of fun i => cramer Aᵀ (Pi.single i 1)


theorem adjugate_def (A : Matrix n n α) : adjugate A = of fun i => cramer Aᵀ (Pi.single i 1) :=
  rfl


theorem adjugate_apply (A : Matrix n n α) (i j : n) :
    adjugate A i j = (A.updateRow j (Pi.single i 1)).det := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.adjugate i j) (A.updateRow j (Pi.single i 1)).det
  -/
  rw [adjugate_def, of_apply, cramer_apply, updateCol_transpose, det_transpose]
  /-
    🎉 no goals
  -/


theorem adjugate_transpose (A : Matrix n n α) : (adjugate A)ᵀ = adjugate Aᵀ := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq A.adjugate.transpose A.transpose.adjugate
  -/
  ext i j
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.adjugate.transpose i j) (A.transpose.adjugate i j)
  -/
  rw [transpose_apply, adjugate_apply, adjugate_apply, updateRow_transpose, det_transpose]
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.updateRow i (Pi.single j 1)).det (A.updateCol j (Pi.single i 1)).det
  -/
  rw [det_apply', det_apply']
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  apply Finset.sum_congr rfl
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ ∀ (x : Equiv.Perm n), Membership.mem Finset.univ x → Eq (HMul.hMul (↑↑(Equiv …
  -/
  intro σ _
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    σ : Equiv.Perm n
    a✝ : Membership.mem Finset.univ σ
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i_1 => A.updateR …
  -/
  congr 1
  /-
    case a.e_a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    σ : Equiv.Perm n
    a✝ : Membership.mem Finset.univ σ
    ⊢ Eq (Finset.univ.prod fun i_1 => A.updateRow i (Pi.single j 1) (σ i_1) i_1) ( …
  -/
  by_cases h : i = σ j
  · -- Everything except `(i , j)` (= `(σ j , j)`) is given by A, and the rest is a single `1`.
    /-
      case pos
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Eq i (σ j)
      ⊢ Eq (Finset.univ.prod fun i_1 => A.updateRow i (Pi.single j 1) (σ i_1) i_1) ( …
    -/
    congr
    /-
      case pos.e_f
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Eq i (σ j)
      ⊢ Eq (fun i_1 => A.updateRow i (Pi.single j 1) (σ i_1) i_1) fun i_1 => A.updat …
    -/
    ext j'
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Eq i (σ j)
      j' : n
      ⊢ Eq (A.updateRow i (Pi.single j 1) (σ j') j') (A.updateCol j (Pi.single i 1)  …
    -/
    subst h
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      ⊢ Eq (A.updateRow (σ j) (Pi.single j 1) (σ j') j') (A.updateCol j (Pi.single ( …
    -/
    have : σ j' = σ j ↔ j' = j := σ.injective.eq_iff
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      this : Iff (Eq (σ j') (σ j)) (Eq j' j)
      ⊢ Eq (A.updateRow (σ j) (Pi.single j 1) (σ j') j') (A.updateCol j (Pi.single ( …
    -/
    rw [updateRow_apply, updateCol_apply]
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      this : Iff (Eq (σ j') (σ j)) (Eq j' j)
      ⊢ Eq (ite (Eq (σ j') (σ j)) (Pi.single j 1 j') (A (σ j') j')) (ite (Eq j' j) ( …
    -/
    simp_rw [this]
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      this : Iff (Eq (σ j') (σ j)) (Eq j' j)
      ⊢ Eq (ite (Eq j' j) (Pi.single j 1 j') (A (σ j') j')) (ite (Eq j' j) (Pi.singl …
    -/
    rw [← dite_eq_ite, ← dite_eq_ite]
    /-
      case pos.e_f.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      this : Iff (Eq (σ j') (σ j)) (Eq j' j)
      ⊢ Eq (dite (Eq j' j) (fun x => Pi.single j 1 j') fun x => A (σ j') j') (dite ( …
    -/
    congr 1 with rfl
    /-
      case pos.e_f.h.e_t.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      j' : n
      this : Iff (Eq (σ j') (σ j')) (Eq j' j')
      ⊢ Eq (Pi.single j' 1 j') (Pi.single (σ j') 1 (σ j'))
    -/
    rw [Pi.single_eq_same, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
  · -- Otherwise, we need to show that there is a `0` somewhere in the product.
    have : (∏ j' : n, updateCol A j (Pi.single i 1) (σ j') j') = 0 := by
      apply prod_eq_zero (mem_univ j)
      rw [updateCol_self, Pi.single_eq_of_ne' h]
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      ⊢ Eq (Finset.univ.prod fun i_1 => A.updateRow i (Pi.single j 1) (σ i_1) i_1) ( …
    -/
    rw [this]
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      ⊢ Eq (Finset.univ.prod fun i_1 => A.updateRow i (Pi.single j 1) (σ i_1) i_1) 0
    -/
    apply prod_eq_zero (mem_univ (σ⁻¹ i))
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      ⊢ Eq (A.updateRow i (Pi.single j 1) (σ ((Inv.inv σ) i)) ((Inv.inv σ) i)) 0
    -/
    erw [apply_symm_apply σ i, updateRow_self]
    /-
      case neg
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      ⊢ Eq (Pi.single j 1 ((Inv.inv σ) i)) 0
    -/
    apply Pi.single_eq_of_ne
    /-
      case neg.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      ⊢ Ne ((Inv.inv σ) i) j
    -/
    intro h'
    /-
      case neg.h
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      i j : n
      σ : Equiv.Perm n
      a✝ : Membership.mem Finset.univ σ
      h : Not (Eq i (σ j))
      this : Eq (Finset.univ.prod fun j' => A.updateCol j (Pi.single i 1) (σ j') j') 0
      h' : Eq ((Inv.inv σ) i) j
      ⊢ False
    -/
    exact h ((symm_apply_eq σ).mp h')
    /-
      🎉 no goals
    -/


@[simp]
theorem adjugate_submatrix_equiv_self (e : n ≃ m) (A : Matrix m m α) :
    adjugate (A.submatrix e e) = (adjugate A).submatrix e e := by
  /-
    m : Type u
    n : Type v
    α : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : CommRing α
    e : Equiv n m
    A : Matrix m m α
    ⊢ Eq (A.submatrix ⇑e ⇑e).adjugate (A.adjugate.submatrix ⇑e ⇑e)
  -/
  ext i j
  rw [adjugate_apply, submatrix_apply, adjugate_apply, ← det_submatrix_equiv_self e,
    updateRow_submatrix_equiv]
  -- Porting note: added
  suffices (fun j => Pi.single i 1 (e.symm j)) = Pi.single (e i) 1 by
    erw [this]
  /-
    case a
    m : Type u
    n : Type v
    α : Type w
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    inst✝ : CommRing α
    e : Equiv n m
    A : Matrix m m α
    i j : n
    ⊢ Eq (fun j => Pi.single i 1 (e.symm j)) (Pi.single (e i) 1)
  -/
  exact Function.update_comp_equiv _ e.symm _ _
  /-
    🎉 no goals
  -/


theorem adjugate_reindex (e : m ≃ n) (A : Matrix m m α) :
    adjugate (reindex e e A) = reindex e e (adjugate A) :=
  adjugate_submatrix_equiv_self _ _


/-- Since the map `b ↦ cramer A b` is linear in `b`, it must be multiplication by some matrix. This
matrix is `A.adjugate`. -/
theorem cramer_eq_adjugate_mulVec (A : Matrix n n α) (b : n → α) :
    cramer A b = A.adjugate *ᵥ b := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    ⊢ Eq (A.cramer b) (A.adjugate.mulVec b)
  -/
  nth_rw 2 [← A.transpose_transpose]
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    ⊢ Eq (A.cramer b) (A.transpose.transpose.adjugate.mulVec b)
  -/
  rw [← adjugate_transpose, adjugate_def]
  have : b = ∑ i, b i • (Pi.single i 1 : n → α) := by
    refine (pi_eq_sum_univ b).trans ?_
    congr with j
    -- Porting note: needed to help `Pi.smul_apply`
    simp [Pi.single_apply, eq_comm, Pi.smul_apply (b j)]
  conv_lhs =>
    rw [this]
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    this : Eq b (Finset.univ.sum fun i => HSMul.hSMul (b i) (Pi.single i 1))
    ⊢ Eq (A.cramer (Finset.univ.sum fun i => HSMul.hSMul (b i) (Pi.single i 1))) ( …
  -/
  ext k
  /-
    case h
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    this : Eq b (Finset.univ.sum fun i => HSMul.hSMul (b i) (Pi.single i 1))
    k : n
    ⊢ Eq (A.cramer (Finset.univ.sum fun i => HSMul.hSMul (b i) (Pi.single i 1)) k) …
  -/
  simp [mulVec, dotProduct, mul_comm]
  /-
    🎉 no goals
  -/


theorem mul_adjugate_apply (A : Matrix n n α) (i j k) :
    A i k * adjugate A k j = cramer Aᵀ (Pi.single k (A i k)) j := by
  rw [← smul_eq_mul, adjugate, of_apply, ← Pi.smul_apply, ← LinearMap.map_smul, ← Pi.single_smul',
    smul_eq_mul, mul_one]


theorem mul_adjugate (A : Matrix n n α) : A * adjugate A = A.det • (1 : Matrix n n α) := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (HMul.hMul A A.adjugate) (HSMul.hSMul A.det 1)
  -/
  ext i j
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (HMul.hMul A A.adjugate i j) (HSMul.hSMul A.det 1 i j)
  -/
  rw [mul_apply, Pi.smul_apply, Pi.smul_apply, one_apply, smul_eq_mul, mul_boole]
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i j : n
    ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (A i j_1) (A.adjugate j_1 j)) (ite  …
  -/
  simp [mul_adjugate_apply, sum_cramer_apply, cramer_transpose_row_self, Pi.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


theorem adjugate_mul (A : Matrix n n α) : adjugate A * A = A.det • (1 : Matrix n n α) :=
  calc
    adjugate A * A = (Aᵀ * adjugate Aᵀ)ᵀ := by
      /-
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        A : Matrix n n α
        ⊢ Eq (HMul.hMul A.adjugate A) (HMul.hMul A.transpose A.transpose.adjugate).tra …
      -/
      rw [← adjugate_transpose, ← transpose_mul, transpose_transpose]
      /-
        🎉 no goals
      -/
                /-
                  n : Type v
                  α : Type w
                  inst✝² : DecidableEq n
                  inst✝¹ : Fintype n
                  inst✝ : CommRing α
                  A : Matrix n n α
                  ⊢ Eq (HMul.hMul A.transpose A.transpose.adjugate).transpose (HSMul.hSMul A.det …
                -/
    _ = _ := by rw [mul_adjugate Aᵀ, det_transpose, transpose_smul, transpose_one]
                /-
                  🎉 no goals
                -/


theorem adjugate_smul (r : α) (A : Matrix n n α) :
    adjugate (r • A) = r ^ (Fintype.card n - 1) • adjugate A := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    r : α
    A : Matrix n n α
    ⊢ Eq (HSMul.hSMul r A).adjugate (HSMul.hSMul (HPow.hPow r (HSub.hSub (Fintype. …
  -/
  rw [adjugate, adjugate, transpose_smul, cramer_smul]
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    r : α
    A : Matrix n n α
    ⊢ Eq (Matrix.of fun i => (HSMul.hSMul (HPow.hPow r (HSub.hSub (Fintype.card n) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A stronger form of **Cramer's rule** that allows us to solve some instances of `A * x = b` even
if the determinant is not a unit. A sufficient (but still not necessary) condition is that `A.det`
divides `b`. -/
@[simp]
theorem mulVec_cramer (A : Matrix n n α) (b : n → α) : A *ᵥ cramer A b = A.det • b := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    b : n → α
    ⊢ Eq (A.mulVec (A.cramer b)) (HSMul.hSMul A.det b)
  -/
  rw [cramer_eq_adjugate_mulVec, mulVec_mulVec, mul_adjugate, smul_mulVec_assoc, one_mulVec]
  /-
    🎉 no goals
  -/


theorem adjugate_subsingleton [Subsingleton n] (A : Matrix n n α) : adjugate A = 1 := by
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Subsingleton n
    A : Matrix n n α
    ⊢ Eq A.adjugate 1
  -/
  ext i j
  /-
    case a
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Subsingleton n
    A : Matrix n n α
    i j : n
    ⊢ Eq (A.adjugate i j) (1 i j)
  -/
  simp [Subsingleton.elim i j, adjugate_apply, det_eq_elem_of_subsingleton _ i, one_apply]
  /-
    🎉 no goals
  -/


theorem adjugate_eq_one_of_card_eq_one {A : Matrix n n α} (h : Fintype.card n = 1) :
    adjugate A = 1 :=
  haveI : Subsingleton n := Fintype.card_le_one_iff_subsingleton.mp h.le
  adjugate_subsingleton _


@[simp]
theorem adjugate_zero [Nontrivial n] : adjugate (0 : Matrix n n α) = 0 := by
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    ⊢ Eq (Matrix.adjugate 0) 0
  -/
  ext i j
  /-
    case a
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j : n
    ⊢ Eq (Matrix.adjugate 0 i j) (0 i j)
  -/
  obtain ⟨j', hj'⟩ : ∃ j', j' ≠ j := exists_ne j
  /-
    case a.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    ⊢ Eq (Matrix.adjugate 0 i j) (0 i j)
  -/
  apply det_eq_zero_of_column_eq_zero j'
  /-
    case a.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    ⊢ ∀ (i_1 : n), Eq ((Matrix.transpose 0).updateCol j (Pi.single i 1) i_1 j') 0
  -/
  intro j''
  /-
    case a.intro
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : Nontrivial n
    i j j' : n
    hj' : Ne j' j
    j'' : n
    ⊢ Eq ((Matrix.transpose 0).updateCol j (Pi.single i 1) j'' j') 0
  -/
  simp [updateCol_ne hj']
  /-
    🎉 no goals
  -/


@[simp]
theorem adjugate_one : adjugate (1 : Matrix n n α) = 1 := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    ⊢ Eq (Matrix.adjugate 1) 1
  -/
  ext
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    i✝ j✝ : n
    ⊢ Eq (Matrix.adjugate 1 i✝ j✝) (1 i✝ j✝)
  -/
  simp [adjugate_def, Matrix.one_apply, Pi.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem adjugate_diagonal (v : n → α) :
    adjugate (diagonal v) = diagonal fun i => ∏ j ∈ Finset.univ.erase i, v j := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    v : n → α
    ⊢ Eq (Matrix.diagonal v).adjugate (Matrix.diagonal fun i => (Finset.univ.erase …
  -/
  ext i j
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    v : n → α
    i j : n
    ⊢ Eq ((Matrix.diagonal v).adjugate i j) (Matrix.diagonal (fun i => (Finset.uni …
  -/
  simp only [adjugate_def, cramer_apply, diagonal_transpose, of_apply]
  /-
    case a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    v : n → α
    i j : n
    ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single i 1)).det (Matrix.diagonal (f …
  -/
  obtain rfl | hij := eq_or_ne i j
  · rw [diagonal_apply_eq, diagonal_updateCol_single, det_diagonal,
      prod_update_of_mem (Finset.mem_univ _), sdiff_singleton_eq_erase, one_mul]
    /-
      case a.inr
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      v : n → α
      i j : n
      hij : Ne i j
      ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single i 1)).det (Matrix.diagonal (f …
    -/
  · rw [diagonal_apply_ne _ hij]
    /-
      case a.inr
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      v : n → α
      i j : n
      hij : Ne i j
      ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single i 1)).det 0
    -/
    refine det_eq_zero_of_row_eq_zero j fun k => ?_
    /-
      case a.inr
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      v : n → α
      i j : n
      hij : Ne i j
      k : n
      ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single i 1) j k) 0
    -/
    obtain rfl | hjk := eq_or_ne k j
      /-
        case a.inr.inl
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        v : n → α
        i k : n
        hij : Ne i k
        ⊢ Eq ((Matrix.diagonal v).updateCol k (Pi.single i 1) k k) 0
      -/
    · rw [updateCol_self, Pi.single_eq_of_ne' hij]
      /-
        🎉 no goals
      -/
      /-
        case a.inr.inr
        n : Type v
        α : Type w
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : CommRing α
        v : n → α
        i j : n
        hij : Ne i j
        k : n
        hjk : Ne k j
        ⊢ Eq ((Matrix.diagonal v).updateCol j (Pi.single i 1) j k) 0
      -/
    · rw [updateCol_ne hjk, diagonal_apply_ne' _ hjk]
      /-
        🎉 no goals
      -/


theorem _root_.RingHom.map_adjugate {R S : Type*} [CommRing R] [CommRing S] (f : R →+* S)
    (M : Matrix n n R) : f.mapMatrix M.adjugate = Matrix.adjugate (f.mapMatrix M) := by
  /-
    n : Type v
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : Matrix n n R
    ⊢ Eq (f.mapMatrix M.adjugate) (f.mapMatrix M).adjugate
  -/
  ext i k
  have : Pi.single i (1 : S) = f ∘ Pi.single i 1 := by
    rw [← f.map_one]
    exact Pi.single_op (fun _ => f) (fun _ => f.map_zero) i (1 : R)
  rw [adjugate_apply, RingHom.mapMatrix_apply, map_apply, RingHom.mapMatrix_apply, this, ←
    map_updateRow, ← RingHom.mapMatrix_apply, ← RingHom.map_det, ← adjugate_apply]


theorem _root_.AlgHom.map_adjugate {R A B : Type*} [CommSemiring R] [CommRing A] [CommRing B]
    [Algebra R A] [Algebra R B] (f : A →ₐ[R] B) (M : Matrix n n A) :
    f.mapMatrix M.adjugate = Matrix.adjugate (f.mapMatrix M) :=
  f.toRingHom.map_adjugate _


theorem det_adjugate (A : Matrix n n α) : (adjugate A).det = A.det ^ (Fintype.card n - 1) := by
  -- get rid of the `- 1`
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq A.adjugate.det (HPow.hPow A.det (HSub.hSub (Fintype.card n) 1))
  -/
  rcases (Fintype.card n).eq_zero_or_pos with h_card | h_card
    /-
      case inl
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h_card : Eq (Fintype.card n) 0
      ⊢ Eq A.adjugate.det (HPow.hPow A.det (HSub.hSub (Fintype.card n) 1))
    -/
  · haveI : IsEmpty n := Fintype.card_eq_zero_iff.mp h_card
    /-
      case inl
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h_card : Eq (Fintype.card n) 0
      this : IsEmpty n
      ⊢ Eq A.adjugate.det (HPow.hPow A.det (HSub.hSub (Fintype.card n) 1))
    -/
    rw [h_card, Nat.zero_sub, pow_zero, adjugate_subsingleton, det_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h_card : GT.gt (Fintype.card n) 0
    ⊢ Eq A.adjugate.det (HPow.hPow A.det (HSub.hSub (Fintype.card n) 1))
  -/
  replace h_card := tsub_add_cancel_of_le h_card.nat_succ_le
  -- express `A` as an evaluation of a polynomial in n^2 variables, and solve in the polynomial ring
  -- where `A'.det` is non-zero.
  /-
    case inr
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h_card : Eq (HAdd.hAdd (HSub.hSub (Fintype.card n) (Nat.succ 0)) (Nat.succ 0)) …
    ⊢ Eq A.adjugate.det (HPow.hPow A.det (HSub.hSub (Fintype.card n) 1))
  -/
  let A' := mvPolynomialX n n ℤ
  suffices A'.adjugate.det = A'.det ^ (Fintype.card n - 1) by
    rw [← mvPolynomialX_mapMatrix_aeval ℤ A, ← AlgHom.map_adjugate, ← AlgHom.map_det, ←
      AlgHom.map_det, ← map_pow, this]
  /-
    case inr
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h_card : Eq (HAdd.hAdd (HSub.hSub (Fintype.card n) (Nat.succ 0)) (Nat.succ 0)) …
    A' : Matrix n n (MvPolynomial (Prod n n) Int) := Matrix.mvPolynomialX n n Int
    ⊢ Eq A'.adjugate.det (HPow.hPow A'.det (HSub.hSub (Fintype.card n) 1))
  -/
  apply mul_left_cancel₀ (show A'.det ≠ 0 from det_mvPolynomialX_ne_zero n ℤ)
  calc
    A'.det * A'.adjugate.det = (A' * adjugate A').det := (det_mul _ _).symm
    _ = A'.det ^ Fintype.card n := by rw [mul_adjugate, det_smul, det_one, mul_one]
    _ = A'.det * A'.det ^ (Fintype.card n - 1) := by rw [← pow_succ', h_card]


@[simp]
theorem adjugate_fin_zero (A : Matrix (Fin 0) (Fin 0) α) : adjugate A = 0 :=
  Subsingleton.elim _ _


@[simp]
theorem adjugate_fin_one (A : Matrix (Fin 1) (Fin 1) α) : adjugate A = 1 :=
  adjugate_subsingleton A


theorem adjugate_fin_succ_eq_det_submatrix {n : ℕ} (A : Matrix (Fin n.succ) (Fin n.succ) α) (i j) :
    adjugate A i j = (-1) ^ (j + i : ℕ) * det (A.submatrix j.succAbove i.succAbove) := by
  /-
    α : Type w
    inst✝ : CommRing α
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) α
    i j : Fin n.succ
    ⊢ Eq (A.adjugate i j) (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd ↑j ↑i)) (A.submatr …
  -/
  simp_rw [adjugate_apply, det_succ_row _ j, updateRow_self, submatrix_updateRow_succAbove]
  /-
    α : Type w
    inst✝ : CommRing α
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) α
    i j : Fin n.succ
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd …
  -/
  rw [Fintype.sum_eq_single i fun h hjk => ?_, Pi.single_eq_same, mul_one]
  /-
    α : Type w
    inst✝ : CommRing α
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) α
    i j h : Fin n.succ
    hjk : Ne h i
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd ↑j ↑h)) (Pi.single i 1 h …
  -/
  rw [Pi.single_eq_of_ne hjk, mul_zero, zero_mul]
  /-
    🎉 no goals
  -/


theorem adjugate_fin_two (A : Matrix (Fin 2) (Fin 2) α) :
    adjugate A = !![A 1 1, -A 0 1; -A 1 0, A 0 0] := by
  /-
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 2) (Fin 2) α
    ⊢ Eq A.adjugate (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 1 1) (Matrix.vec …
  -/
  ext i j
  /-
    case a
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 2) (Fin 2) α
    i j : Fin 2
    ⊢ Eq (A.adjugate i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 1 1) (Matr …
  -/
  rw [adjugate_fin_succ_eq_det_submatrix]
  /-
    case a
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 2) (Fin 2) α
    i j : Fin 2
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd ↑j ↑i)) (A.submatrix j.succAbove i. …
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
  fin_cases i <;> fin_cases j <;> simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem adjugate_fin_two_of (a b c d : α) : adjugate !![a, b; c, d] = !![d, -b; -c, a] :=
  adjugate_fin_two _


theorem adjugate_fin_three (A : Matrix (Fin 3) (Fin 3) α) :
    adjugate A =
    !![A 1 1 * A 2 2 - A 1 2 * A 2 1,
      -(A 0 1 * A 2 2) + A 0 2 * A 2 1,
      A 0 1 * A 1 2 - A 0 2 * A 1 1;
      -(A 1 0 * A 2 2) + A 1 2 * A 2 0,
      A 0 0 * A 2 2 - A 0 2 * A 2 0,
      -(A 0 0 * A 1 2) + A 0 2 * A 1 0;
      A 1 0 * A 2 1 - A 1 1 * A 2 0,
      -(A 0 0 * A 2 1) + A 0 1 * A 2 0,
      A 0 0 * A 1 1 - A 0 1 * A 1 0] := by
  /-
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 3) (Fin 3) α
    ⊢ Eq A.adjugate (Matrix.of (Matrix.vecCons (Matrix.vecCons (HSub.hSub (HMul.hM …
  -/
  ext i j
  /-
    case a
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 3) (Fin 3) α
    i j : Fin 3
    ⊢ Eq (A.adjugate i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (HSub.hSub (H …
  -/
  rw [adjugate_fin_succ_eq_det_submatrix, det_fin_two]
  /-
    case a
    α : Type w
    inst✝ : CommRing α
    A : Matrix (Fin 3) (Fin 3) α
    i j : Fin 3
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd ↑j ↑i)) (HSub.hSub (HMul.hMul (A.su …
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
  fin_cases i <;> fin_cases j <;> simp [updateRow, Fin.succAbove, Fin.lt_def] <;> ring
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem adjugate_fin_three_of (a b c d e f g h i : α) :
    adjugate !![a, b, c; d, e, f; g, h, i] =
      !![  e * i  - f * h, -(b * i) + c * h,   b * f  - c * e;
         -(d * i) + f * g,   a * i  - c * g, -(a * f) + c * d;
           d * h  - e * g, -(a * h) + b * g,   a * e  - b * d] :=
  adjugate_fin_three _


theorem det_eq_sum_mul_adjugate_row (A : Matrix n n α) (i : n) :
    det A = ∑ j : n, A i j * adjugate A j i := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (A i j) (A.adjugate j i))
  -/
  haveI : Nonempty n := ⟨i⟩
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    this : Nonempty n
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (A i j) (A.adjugate j i))
  -/
  obtain ⟨n', hn'⟩ := Nat.exists_eq_succ_of_ne_zero (Fintype.card_ne_zero : Fintype.card n ≠ 0)
  /-
    case intro
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    this : Nonempty n
    n' : Nat
    hn' : Eq (Fintype.card n) n'.succ
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (A i j) (A.adjugate j i))
  -/
  obtain ⟨e⟩ := Fintype.truncEquivFinOfCardEq hn'
  /-
    case intro.mk
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    this : Nonempty n
    n' : Nat
    hn' : Eq (Fintype.card n) n'.succ
    x✝ : Trunc (Equiv n (Fin n'.succ))
    e : Equiv n (Fin n'.succ)
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (A i j) (A.adjugate j i))
  -/
  let A' := reindex e e A
  suffices det A' = ∑ j : Fin n'.succ, A' (e i) j * adjugate A' j (e i) by
    simp_rw [A', det_reindex_self, adjugate_reindex, reindex_apply, submatrix_apply, ← e.sum_comp,
      Equiv.symm_apply_apply] at this
    exact this
  /-
    case intro.mk
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    this : Nonempty n
    n' : Nat
    hn' : Eq (Fintype.card n) n'.succ
    x✝ : Trunc (Equiv n (Fin n'.succ))
    e : Equiv n (Fin n'.succ)
    A' : Matrix (Fin n'.succ) (Fin n'.succ) α := (Matrix.reindex e e) A
    ⊢ Eq A'.det (Finset.univ.sum fun j => HMul.hMul (A' (e i) j) (A'.adjugate j (e …
  -/
  rw [det_succ_row A' (e i)]
  /-
    case intro.mk
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    i : n
    this : Nonempty n
    n' : Nat
    hn' : Eq (Fintype.card n) n'.succ
    x✝ : Trunc (Equiv n (Fin n'.succ))
    e : Equiv n (Fin n'.succ)
    A' : Matrix (Fin n'.succ) (Fin n'.succ) α := (Matrix.reindex e e) A
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd …
  -/
  simp_rw [mul_assoc, mul_left_comm _ (A' _ _), ← adjugate_fin_succ_eq_det_submatrix]
  /-
    🎉 no goals
  -/


theorem det_eq_sum_mul_adjugate_col (A : Matrix n n α) (j : n) :
    det A = ∑ i : n, A i j * adjugate A j i := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    j : n
    ⊢ Eq A.det (Finset.univ.sum fun i => HMul.hMul (A i j) (A.adjugate j i))
  -/
  simpa only [det_transpose, ← adjugate_transpose] using det_eq_sum_mul_adjugate_row Aᵀ j
  /-
    🎉 no goals
  -/


theorem adjugate_conjTranspose [StarRing α] (A : Matrix n n α) : A.adjugateᴴ = adjugate Aᴴ := by
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    ⊢ Eq A.adjugate.conjTranspose A.conjTranspose.adjugate
  -/
  dsimp only [conjTranspose]
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    ⊢ Eq (A.adjugate.transpose.map Star.star) (A.transpose.map Star.star).adjugate
  -/
  have : Aᵀ.adjugate.map star = adjugate (Aᵀ.map star) := (starRingEnd α).map_adjugate Aᵀ
  /-
    n : Type v
    α : Type w
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : CommRing α
    inst✝ : StarRing α
    A : Matrix n n α
    this : Eq (A.transpose.adjugate.map Star.star) (A.transpose.map Star.star).adj …
    ⊢ Eq (A.adjugate.transpose.map Star.star) (A.transpose.map Star.star).adjugate
  -/
  rw [A.adjugate_transpose, this]
  /-
    🎉 no goals
  -/


theorem isRegular_of_isLeftRegular_det {A : Matrix n n α} (hA : IsLeftRegular A.det) :
    IsRegular A := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    hA : IsLeftRegular A.det
    ⊢ IsRegular A
  -/
  constructor
    /-
      case left
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      ⊢ IsLeftRegular A
    -/
  · intro B C h
    /-
      case left
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      B C : Matrix n n α
      h : Eq ((fun x => HMul.hMul A x) B) ((fun x => HMul.hMul A x) C)
      ⊢ Eq B C
    -/
    refine hA.matrix ?_
    /-
      case left
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      B C : Matrix n n α
      h : Eq ((fun x => HMul.hMul A x) B) ((fun x => HMul.hMul A x) C)
      ⊢ Eq ((fun x => HSMul.hSMul A.det x) B) ((fun x => HSMul.hSMul A.det x) C)
    -/
    simp only at h ⊢
    rw [← Matrix.one_mul B, ← Matrix.one_mul C, ← Matrix.smul_mul, ← Matrix.smul_mul, ←
      adjugate_mul, Matrix.mul_assoc, Matrix.mul_assoc, h]
    /-
      case right
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      ⊢ IsRightRegular A
    -/
  · intro B C (h : B * A = C * A)
    /-
      case right
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      B C : Matrix n n α
      h : Eq (HMul.hMul B A) (HMul.hMul C A)
      ⊢ Eq B C
    -/
    refine hA.matrix ?_
    /-
      case right
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      hA : IsLeftRegular A.det
      B C : Matrix n n α
      h : Eq (HMul.hMul B A) (HMul.hMul C A)
      ⊢ Eq ((fun x => HSMul.hSMul A.det x) B) ((fun x => HSMul.hSMul A.det x) C)
    -/
    simp only
    rw [← Matrix.mul_one B, ← Matrix.mul_one C, ← Matrix.mul_smul, ← Matrix.mul_smul, ←
      mul_adjugate, ← Matrix.mul_assoc, ← Matrix.mul_assoc, h]


theorem adjugate_mul_distrib_aux (A B : Matrix n n α) (hA : IsLeftRegular A.det)
    (hB : IsLeftRegular B.det) : adjugate (A * B) = adjugate B * adjugate A := by
  have hAB : IsLeftRegular (A * B).det := by
    rw [det_mul]
    exact hA.mul hB
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    hA : IsLeftRegular A.det
    hB : IsLeftRegular B.det
    hAB : IsLeftRegular (HMul.hMul A B).det
    ⊢ Eq (HMul.hMul A B).adjugate (HMul.hMul B.adjugate A.adjugate)
  -/
  refine (isRegular_of_isLeftRegular_det hAB).left ?_
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    hA : IsLeftRegular A.det
    hB : IsLeftRegular B.det
    hAB : IsLeftRegular (HMul.hMul A B).det
    ⊢ Eq ((fun x => HMul.hMul (HMul.hMul A B) x) (HMul.hMul A B).adjugate) ((fun x …
  -/
  simp only
  rw [mul_adjugate, Matrix.mul_assoc, ← Matrix.mul_assoc B, mul_adjugate,
    smul_mul, Matrix.one_mul, mul_smul, mul_adjugate, smul_smul, mul_comm, ← det_mul]


/-- Proof follows from "The trace Cayley-Hamilton theorem" by Darij Grinberg, Section 5.3
-/
theorem adjugate_mul_distrib (A B : Matrix n n α) : adjugate (A * B) = adjugate B * adjugate A := by
  let g : Matrix n n α → Matrix n n α[X] := fun M =>
    M.map Polynomial.C + (Polynomial.X : α[X]) • (1 : Matrix n n α[X])
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Matrix n n α → Matrix n n (Polynomial α) := fun M => HAdd.hAdd (M.map ⇑Pol …
    ⊢ Eq (HMul.hMul A B).adjugate (HMul.hMul B.adjugate A.adjugate)
  -/
  let f' : Matrix n n α[X] →+* Matrix n n α := (Polynomial.evalRingHom 0).mapMatrix
  have f'_inv : ∀ M, f' (g M) = M := by
    intro
    ext
    simp [f', g]
  have f'_adj : ∀ M : Matrix n n α, f' (adjugate (g M)) = adjugate M := by
    intro
    rw [RingHom.map_adjugate, f'_inv]
  have f'_g_mul : ∀ M N : Matrix n n α, f' (g M * g N) = M * N := by
    intros M N
    rw [RingHom.map_mul, f'_inv, f'_inv]
  have hu : ∀ M : Matrix n n α, IsRegular (g M).det := by
    intro M
    refine Polynomial.Monic.isRegular ?_
    simp only [g, Polynomial.Monic.def, ← Polynomial.leadingCoeff_det_X_one_add_C M, add_comm]
  rw [← f'_adj, ← f'_adj, ← f'_adj, ← f'.map_mul, ←
    adjugate_mul_distrib_aux _ _ (hu A).left (hu B).left, RingHom.map_adjugate,
    RingHom.map_adjugate, f'_inv, f'_g_mul]


@[simp]
theorem adjugate_pow (A : Matrix n n α) (k : ℕ) : adjugate (A ^ k) = adjugate A ^ k := by
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    k : Nat
    ⊢ Eq (HPow.hPow A k).adjugate (HPow.hPow A.adjugate k)
  -/
  induction' k with k IH
    /-
      case zero
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      ⊢ Eq (HPow.hPow A 0).adjugate (HPow.hPow A.adjugate 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      k : Nat
      IH : Eq (HPow.hPow A k).adjugate (HPow.hPow A.adjugate k)
      ⊢ Eq (HPow.hPow A (HAdd.hAdd k 1)).adjugate (HPow.hPow A.adjugate (HAdd.hAdd k …
    -/
  · rw [pow_succ', adjugate_mul_distrib, IH, pow_succ]
    /-
      🎉 no goals
    -/


theorem det_smul_adjugate_adjugate (A : Matrix n n α) :
    det A • adjugate (adjugate A) = det A ^ (Fintype.card n - 1) • A := by
  have : A * (A.adjugate * A.adjugate.adjugate) =
      A * (A.det ^ (Fintype.card n - 1) • (1 : Matrix n n α)) := by
    rw [← adjugate_mul_distrib, adjugate_mul, adjugate_smul, adjugate_one]
  rwa [← Matrix.mul_assoc, mul_adjugate, Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul,
    Matrix.one_mul] at this


/-- Note that this is not true for `Fintype.card n = 1` since `1 - 2 = 0` and not `-1`. -/
theorem adjugate_adjugate (A : Matrix n n α) (h : Fintype.card n ≠ 1) :
    adjugate (adjugate A) = det A ^ (Fintype.card n - 2) • A := by
  -- get rid of the `- 2`
  /-
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub (Fintype.car …
  -/
  cases' h_card : Fintype.card n with n'
    /-
      case zero
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Ne (Fintype.card n) 1
      h_card : Eq (Fintype.card n) 0
      ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub 0 2)) A)
    -/
  · subsingleton [Fintype.card_eq_zero_iff.mp h_card]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n' : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd n' 1)
    ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub (HAdd.hAdd n …
  -/
  cases n'
    /-
      case succ.zero
      n : Type v
      α : Type w
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Ne (Fintype.card n) 1
      h_card : Eq (Fintype.card n) (HAdd.hAdd 0 1)
      ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub (HAdd.hAdd 0 …
    -/
  · exact (h h_card).elim
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub (HAdd.hAdd ( …
  -/
  rw [← h_card]
  -- express `A` as an evaluation of a polynomial in n^2 variables, and solve in the polynomial ring
  -- where `A'.det` is non-zero.
  /-
    case succ.succ
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq A.adjugate.adjugate (HSMul.hSMul (HPow.hPow A.det (HSub.hSub (Fintype.car …
  -/
  let A' := mvPolynomialX n n ℤ
  suffices adjugate (adjugate A') = det A' ^ (Fintype.card n - 2) • A' by
    rw [← mvPolynomialX_mapMatrix_aeval ℤ A, ← AlgHom.map_adjugate, ← AlgHom.map_adjugate, this,
      ← AlgHom.map_det, ← map_pow (MvPolynomial.aeval fun p : n × n ↦ A p.1 p.2),
      AlgHom.mapMatrix_apply, AlgHom.mapMatrix_apply, Matrix.map_smul' _ _ _ (_root_.map_mul _)]
  /-
    case succ.succ
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    A' : Matrix n n (MvPolynomial (Prod n n) Int) := Matrix.mvPolynomialX n n Int
    ⊢ Eq A'.adjugate.adjugate (HSMul.hSMul (HPow.hPow A'.det (HSub.hSub (Fintype.c …
  -/
  have h_card' : Fintype.card n - 2 + 1 = Fintype.card n - 1 := by simp [h_card]
  have is_reg : IsSMulRegular (MvPolynomial (n × n) ℤ) (det A') := fun x y =>
    mul_left_cancel₀ (det_mvPolynomialX_ne_zero n ℤ)
  /-
    case succ.succ
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    A' : Matrix n n (MvPolynomial (Prod n n) Int) := Matrix.mvPolynomialX n n Int
    h_card' : Eq (HAdd.hAdd (HSub.hSub (Fintype.card n) 2) 1) (HSub.hSub (Fintype. …
    is_reg : IsSMulRegular (MvPolynomial (Prod n n) Int) A'.det
    ⊢ Eq A'.adjugate.adjugate (HSMul.hSMul (HPow.hPow A'.det (HSub.hSub (Fintype.c …
  -/
  apply is_reg.matrix
  /-
    case succ.succ.a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    A' : Matrix n n (MvPolynomial (Prod n n) Int) := Matrix.mvPolynomialX n n Int
    h_card' : Eq (HAdd.hAdd (HSub.hSub (Fintype.card n) 2) 1) (HSub.hSub (Fintype. …
    is_reg : IsSMulRegular (MvPolynomial (Prod n n) Int) A'.det
    ⊢ Eq ((fun x => HSMul.hSMul A'.det x) A'.adjugate.adjugate) ((fun x => HSMul.h …
  -/
  simp only
  /-
    case succ.succ.a
    n : Type v
    α : Type w
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Ne (Fintype.card n) 1
    n✝ : Nat
    h_card : Eq (Fintype.card n) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    A' : Matrix n n (MvPolynomial (Prod n n) Int) := Matrix.mvPolynomialX n n Int
    h_card' : Eq (HAdd.hAdd (HSub.hSub (Fintype.card n) 2) 1) (HSub.hSub (Fintype. …
    is_reg : IsSMulRegular (MvPolynomial (Prod n n) Int) A'.det
    ⊢ Eq (HSMul.hSMul A'.det A'.adjugate.adjugate) (HSMul.hSMul A'.det (HSMul.hSMu …
  -/
  rw [smul_smul, ← pow_succ', h_card', det_smul_adjugate_adjugate]
  /-
    🎉 no goals
  -/


/-- A weaker version of `Matrix.adjugate_adjugate` that uses `Nontrivial`. -/
theorem adjugate_adjugate' (A : Matrix n n α) [Nontrivial n] :
    adjugate (adjugate A) = det A ^ (Fintype.card n - 2) • A :=
  adjugate_adjugate _ <| Fintype.one_lt_card.ne'


