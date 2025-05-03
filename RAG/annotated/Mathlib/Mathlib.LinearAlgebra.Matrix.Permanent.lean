/-- The permanent of a square matrix defined as a sum over all permutations. This is analogous to
the determinant but without alternating signs. -/
def permanent (M : Matrix n n R) : R := ∑ σ : Perm n, ∏ i, M (σ i) i


@[simp]
theorem permanent_diagonal {d : n → R} : permanent (diagonal d) = ∏ i, d i := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    d : n → R
    ⊢ Eq (Matrix.diagonal d).permanent (Finset.univ.prod fun i => d i)
  -/
  refine (sum_eq_single 1 (fun σ _ hσ ↦ ?_) (fun h ↦ (h <| mem_univ _).elim)).trans ?_
  · match not_forall.mp (mt Equiv.ext hσ) with
    | ⟨x, hx⟩ => exact Finset.prod_eq_zero (mem_univ x) (if_neg hx)
    /-
      case refine_2
      n : Type u_1
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type u_2
      inst✝ : CommSemiring R
      d : n → R
      ⊢ Eq (Finset.univ.prod fun i => Matrix.diagonal d (1 i) i) (Finset.univ.prod f …
    -/
  · simp only [Perm.one_apply, diagonal_apply_eq]
    /-
      🎉 no goals
    -/


@[simp]
                                                                             /-
                                                                               n : Type u_1
                                                                               inst✝³ : DecidableEq n
                                                                               inst✝² : Fintype n
                                                                               R : Type u_2
                                                                               inst✝¹ : CommSemiring R
                                                                               inst✝ : Nonempty n
                                                                               ⊢ Eq (Matrix.permanent 0) 0
                                                                             -/
theorem permanent_zero [Nonempty n] : permanent (0 : Matrix n n R) = 0 := by simp [permanent]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem permanent_one : permanent (1 : Matrix n n R) = 1 := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    ⊢ Eq (Matrix.permanent 1) 1
  -/
  rw [← diagonal_one]; simp [-diagonal_one]
                       /-
                         🎉 no goals
                       -/


                                                                                 /-
                                                                                   n : Type u_1
                                                                                   inst✝³ : DecidableEq n
                                                                                   inst✝² : Fintype n
                                                                                   R : Type u_2
                                                                                   inst✝¹ : CommSemiring R
                                                                                   inst✝ : IsEmpty n
                                                                                   A : Matrix n n R
                                                                                   ⊢ Eq A.permanent 1
                                                                                 -/
theorem permanent_isEmpty [IsEmpty n] {A : Matrix n n R} : permanent A = 1 := by simp [permanent]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem permanent_eq_one_of_card_eq_zero {A : Matrix n n R} (h : card n = 0) : permanent A = 1 :=
  haveI : IsEmpty n := card_eq_zero_iff.mp h
  permanent_isEmpty


/-- If `n` has only one element, the permanent of an `n` by `n` matrix is just that element.
Although `Unique` implies `DecidableEq` and `Fintype`, the instances might
not be syntactically equal. Thus, we need to fill in the args explicitly. -/
@[simp]
theorem permanent_unique {n : Type*} [Unique n] [DecidableEq n] [Fintype n] (A : Matrix n n R) :
                                          /-
                                            R : Type u_2
                                            inst✝³ : CommSemiring R
                                            n : Type u_3
                                            inst✝² : Unique n
                                            inst✝¹ : DecidableEq n
                                            inst✝ : Fintype n
                                            A : Matrix n n R
                                            ⊢ Eq A.permanent (A Inhabited.default Inhabited.default)
                                          -/
    permanent A = A default default := by simp [permanent, univ_unique]
                                          /-
                                            🎉 no goals
                                          -/


theorem permanent_eq_elem_of_subsingleton [Subsingleton n] (A : Matrix n n R) (k : n) :
    permanent A = A k k := by
  /-
    n : Type u_1
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Subsingleton n
    A : Matrix n n R
    k : n
    ⊢ Eq A.permanent (A k k)
  -/
  have := uniqueOfSubsingleton k
  /-
    n : Type u_1
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Subsingleton n
    A : Matrix n n R
    k : n
    this : Unique n
    ⊢ Eq A.permanent (A k k)
  -/
  convert permanent_unique A
  /-
    🎉 no goals
  -/


theorem permanent_eq_elem_of_card_eq_one {A : Matrix n n R} (h : card n = 1) (k : n) :
    permanent A = A k k :=
  haveI : Subsingleton n := card_le_one_iff_subsingleton.mp h.le
  permanent_eq_elem_of_subsingleton _ _


/-- Transposing a matrix preserves the permanent. -/
@[simp]
theorem permanent_transpose (M : Matrix n n R) : Mᵀ.permanent = M.permanent := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    ⊢ Eq M.transpose.permanent M.permanent
  -/
  refine sum_bijective _ inv_involutive.bijective _ _ ?_
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    ⊢ ∀ (x : Equiv.Perm n), Eq (Finset.univ.prod fun i => M.transpose (x i) i) (Fi …
  -/
  intro σ
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => M.transpose (σ i) i) (Finset.univ.prod fun i = …
  -/
  apply Fintype.prod_equiv σ
  /-
    case h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ ∀ (x : n), Eq (M.transpose (σ x) x) (M ((Inv.inv σ) (σ x)) (σ x))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Permuting the columns does not change the permanent. -/
theorem permanent_permute_cols (σ : Perm n) (M : Matrix n n R) :
    (M.submatrix σ id).permanent = M.permanent :=
  (Group.mulLeft_bijective σ).sum_comp fun τ ↦ ∏ i : n, M (τ i) i


/-- Permuting the rows does not change the permanent. -/
theorem permanent_permute_rows (σ : Perm n) (M : Matrix n n R) :
    (M.submatrix id σ).permanent = M.permanent := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    σ : Equiv.Perm n
    M : Matrix n n R
    ⊢ Eq (M.submatrix id ⇑σ).permanent M.permanent
  -/
  rw [← permanent_transpose, transpose_submatrix, permanent_permute_cols, permanent_transpose]
  /-
    🎉 no goals
  -/


@[simp]
theorem permanent_smul (M : Matrix n n R) (c : R) :
    permanent (c • M) = c ^ Fintype.card n * permanent M := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    ⊢ Eq (HSMul.hSMul c M).permanent (HMul.hMul (HPow.hPow c (Fintype.card n)) M.p …
  -/
  simp only [permanent, smul_apply, smul_eq_mul, Finset.mul_sum]
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.prod fun x_1 => HMul.hMul c (M (x x …
  -/
  congr
  /-
    case e_f
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    ⊢ Eq (fun x => Finset.univ.prod fun x_1 => HMul.hMul c (M (x x_1) x_1)) fun i  …
  -/
  ext
  /-
    case e_f.h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    x✝ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun x => HMul.hMul c (M (x✝ x) x)) (HMul.hMul (HPow.hPo …
  -/
  rw [mul_comm]
  /-
    case e_f.h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    x✝ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun x => HMul.hMul c (M (x✝ x) x)) (HMul.hMul (Finset.u …
  -/
  conv in ∏ _ , c * _ => simp [mul_comm c];
  /-
    case e_f.h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    c : R
    x✝ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun x => HMul.hMul (M (x✝ x) x) c) (HMul.hMul (Finset.u …
  -/
  exact prod_mul_pow_card.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem permanent_updateCol_smul (M : Matrix n n R) (j : n) (c : R) (u : n → R) :
    permanent (updateCol M j <| c • u) = c * permanent (updateCol M j u) := by
  simp only [permanent, ← mul_prod_erase _ _ (mem_univ j), updateCol_self, Pi.smul_apply,
    smul_eq_mul, mul_sum, ← mul_assoc]
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    j : n
    c : R
    u : n → R
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (HMul.hMul c (u (x j))) ((Finset.univ …
  -/
  congr 1 with p
  /-
    case e_f.h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    j : n
    c : R
    u : n → R
    p : Equiv.Perm n
    ⊢ Eq (HMul.hMul (HMul.hMul c (u (p j))) ((Finset.univ.erase j).prod fun i => M …
  -/
  rw [Finset.prod_congr rfl (fun i hi ↦ ?_)]
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommSemiring R
    M : Matrix n n R
    j : n
    c : R
    u : n → R
    p : Equiv.Perm n
    i : n
    hi : Membership.mem (Finset.univ.erase j) i
    ⊢ Eq (M.updateCol j (HSMul.hSMul c u) (p i) i) (M.updateCol j u (p i) i)
  -/
  simp only [ne_eq, ne_of_mem_erase hi, not_false_eq_true, updateCol_ne]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")]
alias permanent_updateColumn_smul := permanent_updateCol_smul


@[simp]
theorem permanent_updateRow_smul (M : Matrix n n R) (j : n) (c : R) (u : n → R) :
    permanent (updateRow M j <| c • u) = c * permanent (updateRow M j u) := by
  rw [← permanent_transpose, ← updateCol_transpose, permanent_updateCol_smul,
    updateCol_transpose, permanent_transpose]


