/-- Let `M` be a `(n+1) × n` matrix whose row sums to zero. Then all the matrices obtained by
deleting one row have the same determinant up to a sign. -/
theorem submatrix_succAbove_det_eq_negOnePow_submatrix_succAbove_det {n : ℕ}
    (M : Matrix (Fin (n + 1)) (Fin n) R) (hv : ∑ j, M j = 0) (j₁ j₂ : Fin (n + 1)) :
    (M.submatrix (Fin.succAbove j₁) id).det =
      Int.negOnePow (j₁ - j₂) • (M.submatrix (Fin.succAbove j₂) id).det := by
  suffices ∀ j, (M.submatrix (Fin.succAbove j) id).det =
      Int.negOnePow j • (M.submatrix (Fin.succAbove 0) id).det by
    rw [this j₁, this j₂, smul_smul, ← Int.negOnePow_add, sub_add_cancel]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    M : Matrix (Fin (HAdd.hAdd n 1)) (Fin n) R
    hv : Eq (Finset.univ.sum fun j => M j) 0
    j₁ j₂ : Fin (HAdd.hAdd n 1)
    ⊢ ∀ (j : Fin (HAdd.hAdd n 1)), Eq (M.submatrix j.succAbove id).det (HSMul.hSMu …
  -/
  intro j
  induction j using Fin.induction with
  | zero => rw [Fin.val_zero, Nat.cast_zero, Int.negOnePow_zero, one_smul]
  | succ i h_ind =>
      rw [Fin.val_succ, Nat.cast_add, Nat.cast_one, Int.negOnePow_succ, Units.neg_smul,
        ← neg_eq_iff_eq_neg, ← neg_one_smul R,
        ← det_updateRow_sum (M.submatrix i.succ.succAbove id) i (fun _ ↦ -1),
        ← Fin.coe_castSucc i, ← h_ind]
      congr
      ext a b
      simp_rw [neg_one_smul, updateRow_apply, Finset.sum_neg_distrib, Pi.neg_apply,
        Finset.sum_apply, submatrix_apply, id_eq]
      split_ifs with h
      · replace hv := congr_fun hv b
        rw [Fin.sum_univ_succAbove _ i.succ, Pi.add_apply, Finset.sum_apply] at hv
        rwa [h, Fin.succAbove_castSucc_self, neg_eq_iff_add_eq_zero, add_comm]
      · obtain h|h := ne_iff_lt_or_gt.mp h
        · rw [Fin.succAbove_castSucc_of_lt _ _ h,
            Fin.succAbove_of_succ_le _ _ (Fin.succ_lt_succ_iff.mpr h).le]
        · rw [Fin.succAbove_succ_of_lt _ _ h, Fin.succAbove_castSucc_of_le _ _ h.le]


/-- Let `M` be a `(n+1) × n` matrix whose column sums to zero. Then all the matrices obtained by
deleting one column have the same determinant up to a sign. -/
theorem submatrix_succAbove_det_eq_negOnePow_submatrix_succAbove_det' {n : ℕ}
    (M : Matrix (Fin n) (Fin (n + 1)) R) (hv : ∀ i, ∑ j, M i j = 0) (j₁ j₂ : Fin (n + 1)) :
    (M.submatrix id (Fin.succAbove j₁)).det =
      Int.negOnePow (j₁ - j₂) • (M.submatrix id (Fin.succAbove j₂)).det := by
  rw [← det_transpose, transpose_submatrix,
    submatrix_succAbove_det_eq_negOnePow_submatrix_succAbove_det M.transpose ?_ j₁ j₂,
    ← det_transpose, transpose_submatrix, transpose_transpose]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    M : Matrix (Fin n) (Fin (HAdd.hAdd n 1)) R
    hv : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => M i j) 0
    j₁ j₂ : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Finset.univ.sum fun j => M.transpose j) 0
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    M : Matrix (Fin n) (Fin (HAdd.hAdd n 1)) R
    hv : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => M i j) 0
    j₁ j₂ : Fin (HAdd.hAdd n 1)
    x✝ : Fin n
    ⊢ Eq (Finset.univ.sum (fun j => M.transpose j) x✝) (0 x✝)
  -/
  simp_rw [Finset.sum_apply, transpose_apply, hv, Pi.zero_apply]
  /-
    🎉 no goals
  -/


