/-- **Gershgorin's circle theorem**: for any eigenvalue `μ` of a square matrix `A`, there exists an
index `k` such that `μ` lies in the closed ball of center the diagonal term `A k k` and of
radius the sum of the norms `∑ j ≠ k, ‖A k j‖. -/
theorem eigenvalue_mem_ball {μ : K} (hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ) :
    ∃ k, μ ∈ Metric.closedBall (A k k) (∑ j ∈ Finset.univ.erase k, ‖A k j‖) := by
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    μ : K
    hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
    ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : IsEmpty n
      ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
    -/
  · exfalso
    /-
      case inl
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : IsEmpty n
      ⊢ False
    -/
    exact hμ Submodule.eq_bot_of_subsingleton
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : Nonempty n
      ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
    -/
  · obtain ⟨v, h_eg, h_nz⟩ := hμ.exists_hasEigenvector
    /-
      case inr.intro.intro
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : Nonempty n
      v : n → K
      h_eg : Membership.mem ((Module.End.genEigenspace (Matrix.toLin' A) μ) 1) v
      h_nz : Ne v 0
      ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
    -/
    obtain ⟨i, -, h_i⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (fun i => ‖v i‖)
    have h_nz : v i ≠ 0 := by
      contrapose! h_nz
      ext j
      rw [Pi.zero_apply, ← norm_le_zero_iff]
      refine (h_i ▸ Finset.le_sup' (fun i => ‖v i‖) (Finset.mem_univ j)).trans ?_
      exact norm_le_zero_iff.mpr h_nz
    have h_le : ∀ j, ‖v j * (v i)⁻¹‖ ≤ 1 := fun j => by
      rw [norm_mul, norm_inv, mul_inv_le_iff₀ (norm_pos_iff.mpr h_nz), one_mul]
      exact h_i ▸ Finset.le_sup' (fun i => ‖v i‖) (Finset.mem_univ j)
    /-
      case inr.intro.intro.intro.intro
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : Nonempty n
      v : n → K
      h_eg : Membership.mem ((Module.End.genEigenspace (Matrix.toLin' A) μ) 1) v
      h_nz✝ : Ne v 0
      i : n
      h_i : Eq (Finset.univ.sup' ⋯ fun i => Norm.norm (v i)) (Norm.norm (v i))
      h_nz : Ne (v i) 0
      h_le : ∀ (j : n), LE.le (Norm.norm (HMul.hMul (v j) (Inv.inv (v i)))) 1
      ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
    -/
    simp_rw [mem_closedBall_iff_norm']
    /-
      case inr.intro.intro.intro.intro
      K : Type u_1
      n : Type u_2
      inst✝² : NormedField K
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n K
      μ : K
      hμ : Module.End.HasEigenvalue (Matrix.toLin' A) μ
      h✝ : Nonempty n
      v : n → K
      h_eg : Membership.mem ((Module.End.genEigenspace (Matrix.toLin' A) μ) 1) v
      h_nz✝ : Ne v 0
      i : n
      h_i : Eq (Finset.univ.sup' ⋯ fun i => Norm.norm (v i)) (Norm.norm (v i))
      h_nz : Ne (v i) 0
      h_le : ∀ (j : n), LE.le (Norm.norm (HMul.hMul (v j) (Inv.inv (v i)))) 1
      ⊢ Exists fun k => LE.le (Norm.norm (HSub.hSub (A k k) μ)) ((Finset.univ.erase  …
    -/
    refine ⟨i, ?_⟩
    calc
      _ = ‖(A i i * v i - μ * v i) * (v i)⁻¹‖ := by congr; field_simp [h_nz]; ring
      _ = ‖(A i i * v i - ∑ j, A i j * v j) * (v i)⁻¹‖ := by
                rw [show μ * v i = ∑ x : n, A i x * v x by
                  rw [← dotProduct, ← Matrix.mulVec]
                  exact (congrFun (Module.End.mem_eigenspace_iff.mp h_eg) i).symm]
      _ = ‖(∑ j ∈ Finset.univ.erase i, A i j * v j) * (v i)⁻¹‖ := by
                rw [Finset.sum_erase_eq_sub (Finset.mem_univ i), ← neg_sub, neg_mul, norm_neg]
      _ ≤ ∑ j ∈ Finset.univ.erase i, ‖A i j‖ * ‖v j * (v i)⁻¹‖ := by
                rw [Finset.sum_mul]
                exact (norm_sum_le _ _).trans (le_of_eq (by simp_rw [mul_assoc, norm_mul]))
      _ ≤ ∑ j ∈ Finset.univ.erase i, ‖A i j‖ :=
                (Finset.sum_le_sum fun j _ => mul_le_of_le_one_right (norm_nonneg _) (h_le j))


/-- If `A` is a row strictly dominant diagonal matrix, then it's determinant is nonzero. -/
theorem det_ne_zero_of_sum_row_lt_diag (h : ∀ k, ∑ j ∈ Finset.univ.erase k, ‖A k j‖ < ‖A k k‖) :
    A.det ≠ 0 := by
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : ∀ (k : n), LT.lt ((Finset.univ.erase k).sum fun j => Norm.norm (A k j)) (N …
    ⊢ Ne A.det 0
  -/
  contrapose! h
  suffices ∃ k, 0 ∈ Metric.closedBall (A k k) (∑ j ∈ Finset.univ.erase k, ‖A k j‖) by
    exact this.imp (fun a h ↦ by rwa [mem_closedBall_iff_norm', sub_zero] at h)
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : Eq A.det 0
    ⊢ Exists fun k => Membership.mem (Metric.closedBall (A k k) ((Finset.univ.eras …
  -/
  refine eigenvalue_mem_ball ?_
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : Eq A.det 0
    ⊢ Module.End.HasEigenvalue (Matrix.toLin' A) 0
  -/
  rw [Module.End.hasEigenvalue_iff, Module.End.eigenspace_zero, ne_comm]
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : Eq A.det 0
    ⊢ Ne Bot.bot (LinearMap.ker (Matrix.toLin' A))
  -/
  exact ne_of_lt (LinearMap.bot_lt_ker_of_det_eq_zero (by rwa [LinearMap.det_toLin']))
  /-
    🎉 no goals
  -/


/-- If `A` is a column strictly dominant diagonal matrix, then it's determinant is nonzero. -/
theorem det_ne_zero_of_sum_col_lt_diag (h : ∀ k, ∑ i ∈ Finset.univ.erase k, ‖A i k‖ < ‖A k k‖) :
    A.det ≠ 0 := by
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : ∀ (k : n), LT.lt ((Finset.univ.erase k).sum fun i => Norm.norm (A i k)) (N …
    ⊢ Ne A.det 0
  -/
  rw [← Matrix.det_transpose]
  /-
    K : Type u_1
    n : Type u_2
    inst✝² : NormedField K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n K
    h : ∀ (k : n), LT.lt ((Finset.univ.erase k).sum fun i => Norm.norm (A i k)) (N …
    ⊢ Ne A.transpose.det 0
  -/
  exact det_ne_zero_of_sum_row_lt_diag (by simp_rw [Matrix.transpose_apply]; exact h)
  /-
    🎉 no goals
  -/

